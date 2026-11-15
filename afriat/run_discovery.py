"""End-to-end Phase 0 Discovery Cartography CLI for Project Afriat."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any
import numpy as np

from afriat.geometry import (
    LayerBasis,
    bootstrap_mean_ci,
    compute_cartography_projections,
    exact_or_mc_signflip_p,
    extract_concept_basis,
    holm_adjust,
    select_cartography_layer,
)
from afriat.models import (
    MODEL_PRESETS,
    TransformerHarness,
    build_scenario_reading_prompt,
    format_chat_prompt,
    resolve_model_name,
)
from afriat.schema import (
    ALL_CONCEPTS,
    CORE_ARCHETYPES,
    CORE_CONCEPTS,
    ROOT_DIR,
    STIMULI_DIR,
    load_concepts,
    load_sanity_items,
    load_transactions,
    validate_all_stimuli,
)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")


def evaluate_sanity_gate(
    harness: Any,
    sanity_items: list[Any],
    batch_size: int,
) -> dict[str, Any]:
    scenarios = [it.scenario for it in sanity_items]
    better = [it.better_option for it in sanity_items]
    worse = [it.worse_option for it in sanity_items]
    scores, _ = harness.score_counterbalanced_choices(
        scenarios=scenarios,
        target_options=better,
        reference_options=worse,
        batch_size=batch_size,
        capture_decision_states=False,
    )
    log_odds = np.array([s.target_log_odds for s in scores], dtype=np.float64)
    ab_mass = np.array([s.ab_mass for s in scores], dtype=np.float64)
    correct = log_odds > 0.0

    by_category: dict[str, float] = {}
    for cat in sorted({it.category for it in sanity_items}):
        mask = np.array([it.category == cat for it in sanity_items])
        by_category[cat] = float(np.mean(correct[mask]))

    n_correct = int(np.sum(correct))
    acc = float(n_correct / len(sanity_items))
    return {
        "n_total": len(sanity_items),
        "n_correct": n_correct,
        "accuracy": acc,
        "passed_gate": bool(acc >= 0.85),
        "mean_log_odds": float(np.mean(log_odds)),
        "mean_ab_mass": float(np.mean(ab_mass)),
        "accuracy_by_category": by_category,
    }


def run_pipeline(
    harness: Any,
    out_dir: Path,
    stimuli_dir: Path = STIMULI_DIR,
    batch_size: int = 8,
    skip_steering: bool = False,
    steering_alpha: float = 1.25,
    skip_plots: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    audit = validate_all_stimuli(stimuli_dir)
    if not audit["valid"]:
        raise RuntimeError(f"Stimulus validation failed: {audit['errors']}")

    concepts = load_concepts(stimuli_dir / "concepts.jsonl")
    transactions = load_transactions(stimuli_dir / "transactions.jsonl")
    sanity_items = load_sanity_items(stimuli_dir / "sanity_arithmetic.jsonl")

    print(f"[1/4] Running 24-item arithmetic & equivalence sanity gate on {harness.model_id}...")
    sanity_gate = evaluate_sanity_gate(harness, sanity_items, batch_size=batch_size)
    print(
        f"      Sanity accuracy: {sanity_gate['accuracy']*100:.1f}% "
        f"({sanity_gate['n_correct']}/{sanity_gate['n_total']}), "
        f"mean A/B mass: {sanity_gate['mean_ab_mass']:.2f}"
    )

    print("[2/4] Extracting 10 non-financial concept & control subspaces (400 scenarios)...")
    pos_by_concept: dict[str, np.ndarray] = {}
    neg_by_concept: dict[str, np.ndarray] = {}
    for concept_name in ALL_CONCEPTS:
        sub = [c for c in concepts if c.concept == concept_name]
        pos_prompts = [
            format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(c.positive))
            for c in sub
        ]
        neg_prompts = [
            format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(c.negative))
            for c in sub
        ]
        pos_by_concept[concept_name] = harness.extract_last_token_states(pos_prompts, batch_size=batch_size)
        neg_by_concept[concept_name] = harness.extract_last_token_states(neg_prompts, batch_size=batch_size)

    basis: LayerBasis = extract_concept_basis(pos_by_concept, neg_by_concept)
    primary_layer = select_cartography_layer(basis, core_concepts=CORE_CONCEPTS)
    print(f"      Selected non-financial basis primary layer: {primary_layer} of {harness.n_layers - 1}")

    basis_diagnostics: list[dict[str, Any]] = []
    for c in ALL_CONCEPTS:
        basis_diagnostics.append({
            "concept": c,
            "lopo_acc_at_primary": float(basis.lopo_accuracy[c][primary_layer]),
            "lopo_cos_at_primary": float(basis.lopo_cosine[c][primary_layer]),
            "residual_ratio_at_primary": (
                float(basis.residual_ratio[c][primary_layer]) if c in basis.residual_ratio else None
            ),
            "lopo_acc_curve": basis.lopo_accuracy[c].tolist(),
            "lopo_cos_curve": basis.lopo_cosine[c].tolist(),
        })

    print("[3/4] Running clinical transaction cartography & counterbalanced behavioral choices...")
    arch_reading_by_tx: dict[str, np.ndarray] = {}
    ctrl_reading_by_tx: dict[str, np.ndarray] = {}
    arch_decision_by_tx: dict[str, np.ndarray] = {}
    ctrl_decision_by_tx: dict[str, np.ndarray] = {}

    behavioral_rows: list[dict[str, Any]] = []
    fixture_records: dict[str, list[dict[str, Any]]] = {}
    baseline_gaps_by_arch: dict[str, np.ndarray] = {}

    for arch in CORE_ARCHETYPES:
        sub = [t for t in transactions if t.archetype == arch]
        arch_read_prompts = [
            format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(t.archetype_scenario))
            for t in sub
        ]
        ctrl_read_prompts = [
            format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(t.control_scenario))
            for t in sub
        ]
        arch_reading_by_tx[arch] = harness.extract_last_token_states(arch_read_prompts, batch_size=batch_size)
        ctrl_reading_by_tx[arch] = harness.extract_last_token_states(ctrl_read_prompts, batch_size=batch_size)

        arch_scores, arch_dec_states = harness.score_counterbalanced_choices(
            scenarios=[t.archetype_scenario for t in sub],
            target_options=[t.deviation_option for t in sub],
            reference_options=[t.rational_option for t in sub],
            batch_size=batch_size,
            capture_decision_states=True,
        )
        ctrl_scores, ctrl_dec_states = harness.score_counterbalanced_choices(
            scenarios=[t.control_scenario for t in sub],
            target_options=[t.deviation_option for t in sub],
            reference_options=[t.rational_option for t in sub],
            batch_size=batch_size,
            capture_decision_states=True,
        )
        assert arch_dec_states is not None and ctrl_dec_states is not None
        arch_decision_by_tx[arch] = arch_dec_states
        ctrl_decision_by_tx[arch] = ctrl_dec_states

        arch_lo = np.array([s.target_log_odds for s in arch_scores], dtype=np.float64)
        ctrl_lo = np.array([s.target_log_odds for s in ctrl_scores], dtype=np.float64)
        deltas = arch_lo - ctrl_lo
        baseline_gaps_by_arch[arch] = deltas

        ci_lo, ci_hi = bootstrap_mean_ci(deltas)
        p_val = exact_or_mc_signflip_p(deltas, two_sided=True)
        behavioral_rows.append({
            "archetype": arch,
            "n_fixtures": len(sub),
            "mean_delta_log_odds": float(np.mean(deltas)),
            "ci95": [ci_lo, ci_hi],
            "mean_arch_log_odds": float(np.mean(arch_lo)),
            "mean_ctrl_log_odds": float(np.mean(ctrl_lo)),
            "mean_arch_prob": float(np.mean([s.target_prob for s in arch_scores])),
            "mean_ctrl_prob": float(np.mean([s.target_prob for s in ctrl_scores])),
            "mean_ab_mass": float(0.5 * (
                np.mean([s.ab_mass for s in arch_scores]) + np.mean([s.ab_mass for s in ctrl_scores])
            )),
            "p_signflip": p_val,
        })
        fixture_records[arch] = [
            {
                "fixture_id": t.fixture_id,
                "arch_log_odds": float(sa.target_log_odds),
                "ctrl_log_odds": float(sc.target_log_odds),
                "delta_log_odds": float(sa.target_log_odds - sc.target_log_odds),
                "arch_prob": float(sa.target_prob),
                "ctrl_prob": float(sc.target_prob),
            }
            for t, sa, sc in zip(sub, arch_scores, ctrl_scores)
        ]

    beh_holm = holm_adjust([r["p_signflip"] for r in behavioral_rows])
    for i, r in enumerate(behavioral_rows):
        r["p_holm"] = float(beh_holm[i])

    cart_scenario = compute_cartography_projections(
        arch_reading_by_tx,
        ctrl_reading_by_tx,
        basis,
        primary_layer=primary_layer,
    )
    cart_decision = compute_cartography_projections(
        arch_decision_by_tx,
        ctrl_decision_by_tx,
        basis,
        primary_layer=primary_layer,
    )

    # Pick top specific concept per archetype at primary_layer
    top_cell_by_arch: dict[str, dict[str, Any]] = {}
    for arch in CORE_ARCHETYPES:
        cells_a = [c for c in cart_scenario["primary_cells"] if c["archetype"] == arch]
        top_cell_by_arch[arch] = max(cells_a, key=lambda c: abs(c["cohen_dz_controlled"]))

    micro_steering: list[dict[str, Any]] = []
    if not skip_steering:
        print(f"[4/4] Running closed-loop micro-steering dissociation screen at layer {primary_layer}...")
        # Calibrate 1.0 alpha unit to the pooled natural concept contrast RMS at primary_layer
        natural_contrast_unit = float(
            np.mean([basis.contrast_rms[c][primary_layer] for c in CORE_CONCEPTS])
        )
        step_mag = steering_alpha * natural_contrast_unit

        for idx_a, arch in enumerate(CORE_ARCHETYPES):
            sub = [t for t in transactions if t.archetype == arch]
            base_deltas = baseline_gaps_by_arch[arch]
            base_mean_gap = float(np.mean(base_deltas))

            top_cell = top_cell_by_arch[arch]
            matched_c = str(top_cell["concept"])
            matched_sign = 1.0 if float(top_cell["cohen_dz_controlled"]) >= 0.0 else -1.0

            # Pick a true double-dissociation control: the rescue intervention of another archetype
            # whose leading concept is distinct and minimally active on THIS archetype.
            cells_a_map = {
                str(c["concept"]): float(c["cohen_dz_controlled"])
                for c in cart_scenario["primary_cells"]
                if c["archetype"] == arch
            }
            other_candidates: list[tuple[float, str, float]] = []
            for other_arch, other_top in top_cell_by_arch.items():
                other_c = str(other_top["concept"])
                if other_arch != arch and other_c != matched_c:
                    other_rescue_sign = 1.0 if float(other_top["cohen_dz_controlled"]) >= 0.0 else -1.0
                    other_candidates.append((abs(cells_a_map[other_c]), other_c, other_rescue_sign))

            if other_candidates:
                _, mismatched_c, mismatch_sign = min(other_candidates, key=lambda item: item[0])
            else:
                other_concepts = [c for c in CORE_CONCEPTS if c != matched_c]
                mismatched_c = min(other_concepts, key=lambda c: abs(cells_a_map[c]))
                mismatch_sign = 1.0

            # Valence sign on this archetype
            val_dz = float(cart_scenario["dz_affect_curves"][idx_a, 0, primary_layer])
            val_sign = 1.0 if val_dz >= 0.0 else -1.0

            ctrl_base_lo = np.array(
                [r["ctrl_log_odds"] for r in fixture_records[arch]],
                dtype=np.float64,
            )

            def _steered_gap(direction: np.ndarray, sign_proj: float) -> float:
                """Steer the anomalous archetype scenario and measure remaining gap vs. unperturbed control."""
                with harness.residual_steering(
                    layer_idx=primary_layer,
                    direction=direction,
                    delta_scale=-sign_proj * step_mag,
                ):
                    s_arch, _ = harness.score_counterbalanced_choices(
                        scenarios=[t.archetype_scenario for t in sub],
                        target_options=[t.deviation_option for t in sub],
                        reference_options=[t.rational_option for t in sub],
                        batch_size=batch_size,
                    )
                steered_arch_lo = np.array([a.target_log_odds for a in s_arch], dtype=np.float64)
                return float(np.mean(steered_arch_lo - ctrl_base_lo))

            gap_matched = _steered_gap(basis.controlled_dirs[matched_c][primary_layer], matched_sign)
            gap_mismatched = _steered_gap(basis.controlled_dirs[mismatched_c][primary_layer], mismatch_sign)
            gap_valence = _steered_gap(basis.raw_dirs["valence"][primary_layer], val_sign)

            with harness.residual_steering(
                layer_idx=primary_layer,
                direction=basis.controlled_dirs[matched_c][primary_layer],
                delta_scale=-matched_sign * step_mag,
            ):
                steered_sanity = evaluate_sanity_gate(harness, sanity_items, batch_size=batch_size)

            micro_steering.append({
                "archetype": arch,
                "layer": int(primary_layer),
                "alpha": float(steering_alpha),
                "step_magnitude": float(step_mag),
                "baseline_gap_log_odds": base_mean_gap,
                "matched_concept": matched_c,
                "mismatched_concept": mismatched_c,
                "steered_gap_matched_controlled": gap_matched,
                "steered_gap_mismatched_controlled": gap_mismatched,
                "steered_gap_valence": gap_valence,
                "delta_gap_matched_controlled": float(gap_matched - base_mean_gap),
                "delta_gap_mismatched_controlled": float(gap_mismatched - base_mean_gap),
                "delta_gap_valence": float(gap_valence - base_mean_gap),
                "sanity_acc_steered": float(steered_sanity["accuracy"]),
            })

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save basis vectors cleanly as compressed NPZ for downstream deep-dive scripts
    npz_payload: dict[str, np.ndarray] = {"stream_rms": basis.stream_rms}
    for c, arr in basis.raw_dirs.items():
        npz_payload[f"raw_{c}"] = arr.astype(np.float32)
    for c, arr in basis.controlled_dirs.items():
        npz_payload[f"controlled_{c}"] = arr.astype(np.float32)
    np.savez_compressed(out_dir / "basis_vectors.npz", **npz_payload)

    def _serialize_cart(c_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            "primary_layer": c_dict["primary_layer"],
            "archetypes": c_dict["archetypes"],
            "core_concepts": c_dict["core_concepts"],
            "control_concepts": c_dict["control_concepts"],
            "primary_cells": c_dict["primary_cells"],
            "dz_controlled_curves": c_dict["dz_controlled_curves"].tolist(),
            "dz_raw_curves": c_dict["dz_raw_curves"].tolist(),
            "cos_controlled_curves": c_dict["cos_controlled_curves"].tolist(),
            "dz_affect_curves": c_dict["dz_affect_curves"].tolist(),
        }

    results: dict[str, Any] = {
        "metadata": {
            "model_id": harness.model_id,
            "n_layers": int(harness.n_layers),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(time.time() - t0, 2),
            "stimulus_manifest": audit["manifest"],
        },
        "sanity_gate": sanity_gate,
        "basis_diagnostics": basis_diagnostics,
        "behavioral_anomalies": behavioral_rows,
        "fixture_records": fixture_records,
        "cartography_scenario": _serialize_cart(cart_scenario),
        "cartography_decision": _serialize_cart(cart_decision),
        "micro_steering": micro_steering,
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if skip_plots:
        print(f"Saved raw discovery bundle (results.json + basis_vectors.npz) to {out_dir}")
    else:
        from afriat.plot_results import write_reports

        md_path, html_path = write_reports(results, out_dir)
        print(f"Saved complete discovery bundle to {out_dir}")
        print(f"  - Markdown summary: {md_path}")
        print(f"  - Dawn State HTML:  {html_path}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Project Afriat Phase 0 Discovery Cartography Runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma4-12b",
        help=f"HuggingFace repo ID, local checkpoint path, or preset alias ({', '.join(MODEL_PRESETS.keys())})",
    )
    parser.add_argument("--model-label", type=str, default=None, help="Canonical model ID label for results.json")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steering-alpha", type=float, default=1.25)
    parser.add_argument("--skip-steering", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib rendering (for minimal pod containers)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=None,
        help="Render Dawn State SVG/PNG/Markdown/HTML reports from an existing results.json directory.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate stimulus lexical firewall & manifest hashes without loading a model.",
    )
    args = parser.parse_args(argv)

    if args.audit_only:
        report = validate_all_stimuli()
        if not report["valid"]:
            for e in report["errors"]:
                print(f"[ERROR] {e}")
            raise SystemExit(1)
        print(json.dumps(report, indent=2))
        return

    if args.render_dir is not None:
        from afriat.plot_results import write_reports

        res_file = args.render_dir / "results.json"
        results = json.loads(res_file.read_text(encoding="utf-8"))
        md_path, html_path = write_reports(results, args.render_dir)
        print(f"Rendered reports in {args.render_dir}: {md_path.name}, {html_path.name}")
        return

    label = args.model_label or resolve_model_name(args.model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT_DIR / "runs" / f"{_slugify(label)}_{stamp}")

    harness = TransformerHarness.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        model_id_label=label,
    )
    run_pipeline(
        harness=harness,
        out_dir=out_dir,
        batch_size=args.batch_size,
        skip_steering=args.skip_steering,
        steering_alpha=args.steering_alpha,
        skip_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
