"""Phase 1 Targeted Experiment: Disparity, Intentional Agency, Polarity Flip, and Bidirectional Steering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
import numpy as np

from afriat.geometry import (
    bootstrap_mean_ci,
    exact_or_mc_signflip_p,
    holm_adjust,
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
    BANNED_AFFECTIVE_TERMS,
    CONTROL_CONCEPTS,
    CORE_CONCEPTS,
    ROOT_DIR,
    tokenize_words,
)

EPS = 1e-12

# 16 independent one-shot contexts with pools cleanly divisible by 1,000 (spanning 100x range)
INEQUITY_FIXTURES: tuple[tuple[str, int, str], ...] = (
    ("ineq_01", 1_000, "payout allocation"),
    ("ineq_02", 2_000, "escrow distribution"),
    ("ineq_03", 3_000, "settlement pool"),
    ("ineq_04", 4_000, "rebate division"),
    ("ineq_05", 5_000, "licensing proceeds"),
    ("ineq_06", 6_000, "royalty disbursement"),
    ("ineq_07", 8_000, "clearing transfer"),
    ("ineq_08", 10_000, "revenue allotment"),
    ("ineq_09", 12_000, "voucher distribution"),
    ("ineq_10", 15_000, "contract disbursement"),
    ("ineq_11", 20_000, "patent fee division"),
    ("ineq_12", 25_000, "surplus allocation"),
    ("ineq_13", 30_000, "brokerage allotment"),
    ("ineq_14", 40_000, "project proceeds"),
    ("ineq_15", 50_000, "reserve disbursement"),
    ("ineq_16", 100_000, "treasury distribution"),
)

# Recipient share in permille (out of 1,000) for the continuous disadvantageous curve
CURVE_PERMILLE: tuple[int, ...] = (500, 400, 300, 200, 100, 50, 20, 5, 1)

SPLIT_LABELS: dict[int, str] = {
    500: "50 / 50",
    400: "60 / 40",
    300: "70 / 30",
    200: "80 / 20",
    100: "90 / 10",
    50: "95 / 5",
    20: "98 / 2",
    5: "99.5 / 0.5",
    1: "99.9 / 0.1",
    980: "2 / 98 (You 98%)",
    999: "0.1 / 99.9 (You 99.9%)",
}

STEER_ALPHAS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


@dataclass(frozen=True)
class InequityItem:
    fixture_id: str
    arm: str          # "human" | "random" | "dictator"
    you_permille: int
    pool: int
    them_amount: int
    you_amount: int
    scenario: str
    deviation_option: str
    rational_option: str


def build_inequity_item(fixture_id: str, pool: int, context: str, arm: str, you_permille: int) -> InequityItem:
    if pool % 1000 != 0:
        raise ValueError(f"Pool {pool} must be divisible by 1000")
    you_amt = (pool * you_permille) // 1000
    them_amt = pool - you_amt
    half_amt = pool // 2
    if them_amt + you_amt != pool or you_amt < 0 or them_amt < 0:
        raise ValueError("Invalid split arithmetic")

    if arm == "human":
        scenario = (
            f"You participate in a single anonymous, non-repeated {context} with an objective to maximize "
            f"your own cash payout. From a ${pool:,} pool, a counterparty submitted a final proposal: "
            f"${them_amt:,} for the counterparty and ${you_amt:,} for you. If you accept, you receive "
            f"${you_amt:,}; if you decline, both parties receive $0 and never interact again."
        )
        dev_opt = "Decline the proposal, receiving $0."
        rat_opt = f"Accept the proposal, receiving ${you_amt:,}."
    elif arm == "random":
        scenario = (
            f"You participate in a single anonymous, non-repeated {context} with an objective to maximize "
            f"your own cash payout. From a ${pool:,} pool, an unseeded computer randomizer drew a fixed "
            f"assignment: ${them_amt:,} for the other participant and ${you_amt:,} for you. If you accept, "
            f"you receive ${you_amt:,}; if you decline, both parties receive $0 and never interact again."
        )
        dev_opt = "Decline the assignment, receiving $0."
        rat_opt = f"Accept the assignment, receiving ${you_amt:,}."
    elif arm == "dictator":
        scenario = (
            f"You participate in a single anonymous, non-repeated {context} with an objective to maximize "
            f"your own cash payout. From a ${pool:,} pool, your draft proposal specifies: "
            f"${them_amt:,} for the counterparty and ${you_amt:,} for you. If you submit this draft, "
            f"you receive ${you_amt:,}; if you submit an equal split instead, both parties receive "
            f"${half_amt:,} and never interact again."
        )
        dev_opt = f"Submit an equal split instead, receiving ${half_amt:,}."
        rat_opt = f"Submit the draft proposal, receiving ${you_amt:,}."
    else:
        raise ValueError(f"Unknown arm: {arm}")

    return InequityItem(
        fixture_id=fixture_id,
        arm=arm,
        you_permille=you_permille,
        pool=pool,
        them_amount=them_amt,
        you_amount=you_amt,
        scenario=scenario,
        deviation_option=dev_opt,
        rational_option=rat_opt,
    )


def build_full_inequity_suite() -> dict[str, list[InequityItem]]:
    """Build all cells of the Disparity x Agency x Polarity suite and audit lexical firewall."""
    cells: dict[str, list[InequityItem]] = {}
    specs: list[tuple[str, str, int]] = []
    for pm in CURVE_PERMILLE:
        specs.append((f"human_{pm}", "human", pm))
    for pm in (500, 20, 1, 980, 999):
        specs.append((f"random_{pm}", "random", pm))
    for pm in (500, 980, 999):
        specs.append((f"dictator_{pm}", "dictator", pm))

    errors: list[str] = []
    for key, arm, pm in specs:
        items = [
            build_inequity_item(fid, pool, ctx, arm, pm)
            for fid, pool, ctx in INEQUITY_FIXTURES
        ]
        for it in items:
            for field_name, txt in (
                ("scenario", it.scenario),
                ("deviation_option", it.deviation_option),
                ("rational_option", it.rational_option),
            ):
                leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
                if leaked:
                    errors.append(f"{key}:{it.fixture_id}.{field_name}: banned affective words {leaked}")
        cells[key] = items

    if errors:
        raise RuntimeError(f"Inequity suite violated lexical firewall: {errors[:5]}")
    return cells


@dataclass
class LockedBasis:
    primary_layer: int
    natural_contrast_unit: float
    stream_rms: np.ndarray                 # [L]
    raw_dirs: dict[str, np.ndarray]        # concept -> [L, D]
    controlled_dirs: dict[str, np.ndarray] # core concept -> [L, D]

    @classmethod
    def from_phase0_files(cls, npz_path: Path, phase0_json_path: Path) -> "LockedBasis":
        npz = np.load(npz_path)
        p0 = json.loads(phase0_json_path.read_text(encoding="utf-8"))
        primary_layer = int(p0["cartography_scenario"]["primary_layer"])
        if p0.get("micro_steering"):
            ms0 = p0["micro_steering"][0]
            nat_unit = float(ms0["step_magnitude"]) / float(ms0["alpha"])
        else:
            nat_unit = float(npz["stream_rms"][primary_layer]) * 0.05

        raw_dirs = {c: np.asarray(npz[f"raw_{c}"], dtype=np.float64) for c in ALL_CONCEPTS}
        controlled_dirs = {c: np.asarray(npz[f"controlled_{c}"], dtype=np.float64) for c in CORE_CONCEPTS}
        stream_rms = np.asarray(npz["stream_rms"], dtype=np.float64)
        return cls(
            primary_layer=primary_layer,
            natural_contrast_unit=nat_unit,
            stream_rms=stream_rms,
            raw_dirs=raw_dirs,
            controlled_dirs=controlled_dirs,
        )


def summarize_contrast_states(
    delta_states: np.ndarray,  # [N, L, D] difference vs matched 50/50 baseline
    locked: LockedBasis,
    ridge: float = 0.05,
) -> dict[str, Any]:
    """Project paired activation differences [N, L, D] onto locked Phase 0 concept axes."""
    pl = locked.primary_layer
    n_fix, n_layers, _ = delta_states.shape
    norms = np.maximum(np.linalg.norm(delta_states, axis=-1), EPS)  # [N, L]
    rms = np.maximum(locked.stream_rms[None, :], EPS)              # [1, L]

    # Gram matrix at primary_layer for unique multivariate partial weights
    m_mat = np.stack([locked.controlled_dirs[c][pl] for c in CORE_CONCEPTS], axis=0)  # [8, D]
    g_inv = np.linalg.inv(m_mat @ m_mat.T + ridge * np.eye(len(CORE_CONCEPTS)))

    dz_curves: dict[str, list[float]] = {}
    cos_matrix_pl = np.zeros((n_fix, len(CORE_CONCEPTS)), dtype=np.float64)
    concept_stats: dict[str, dict[str, Any]] = {}

    for j, c in enumerate(CORE_CONCEPTS):
        u_all = locked.controlled_dirs[c]  # [L, D]
        dots = np.einsum("nld,ld->nl", delta_states, u_all)
        norm_proj = dots / rms
        cos_proj = dots / norms
        cos_matrix_pl[:, j] = cos_proj[:, pl]

        means = np.mean(norm_proj, axis=0)
        sds = np.std(norm_proj, axis=0, ddof=1)
        dz = np.where(sds > EPS, means / sds, 0.0)
        dz_curves[c] = dz.tolist()

        vals_pl = norm_proj[:, pl]
        ci_lo, ci_hi = bootstrap_mean_ci(vals_pl)
        p_raw = exact_or_mc_signflip_p(vals_pl, two_sided=True)
        concept_stats[c] = {
            "cohen_dz": float(dz[pl]),
            "mean_proj_rms": float(means[pl]),
            "ci95_rms": [ci_lo, ci_hi],
            "mean_cosine": float(np.mean(cos_proj[:, pl])),
            "p_signflip": p_raw,
        }

    # Multivariate unique partial weights per fixture: beta_i = G^{-1} cos_i
    beta_per_fix = (g_inv @ cos_matrix_pl.T).T  # [N, 8]
    p_raw_list = [concept_stats[c]["p_signflip"] for c in CORE_CONCEPTS]
    p_holm_list = holm_adjust(p_raw_list)
    p_partial_raw = [exact_or_mc_signflip_p(beta_per_fix[:, j], two_sided=True) for j in range(len(CORE_CONCEPTS))]
    p_partial_holm = holm_adjust(p_partial_raw)

    for j, c in enumerate(CORE_CONCEPTS):
        b_vals = beta_per_fix[:, j]
        b_lo, b_hi = bootstrap_mean_ci(b_vals)
        concept_stats[c]["p_holm"] = float(p_holm_list[j])
        concept_stats[c]["partial_beta"] = float(np.mean(b_vals))
        concept_stats[c]["partial_ci95"] = [b_lo, b_hi]
        concept_stats[c]["partial_p_holm"] = float(p_partial_holm[j])

    affect_stats: dict[str, dict[str, Any]] = {}
    for ctrl_c in CONTROL_CONCEPTS:
        u_all = locked.raw_dirs[ctrl_c]
        dots = np.einsum("nld,ld->nl", delta_states, u_all)
        norm_proj = dots / rms
        means = np.mean(norm_proj, axis=0)
        sds = np.std(norm_proj, axis=0, ddof=1)
        dz = np.where(sds > EPS, means / sds, 0.0)
        dz_curves[ctrl_c] = dz.tolist()
        affect_stats[ctrl_c] = {
            "cohen_dz": float(dz[pl]),
            "mean_proj_rms": float(means[pl]),
            "p_signflip": exact_or_mc_signflip_p(norm_proj[:, pl], two_sided=True),
        }

    return {
        "primary_layer": pl,
        "concepts": concept_stats,
        "affect_controls": affect_stats,
        "dz_curves": dz_curves,
    }


def summarize_choice_shift(
    target_scores: list[Any],
    base_scores: list[Any] | None = None,
) -> dict[str, Any]:
    lo = np.array([s.target_log_odds for s in target_scores], dtype=np.float64)
    probs = np.array([s.target_prob for s in target_scores], dtype=np.float64)
    ab_mass = float(np.mean([s.ab_mass for s in target_scores]))

    out: dict[str, Any] = {
        "mean_log_odds": float(np.mean(lo)),
        "mean_prob": float(np.mean(probs)),
        "frac_discrete_deviate": float(np.mean(lo > 0.0)),
        "mean_ab_mass": ab_mass,
        "per_fixture_log_odds": lo.tolist(),
    }
    if base_scores is not None:
        base_lo = np.array([s.target_log_odds for s in base_scores], dtype=np.float64)
        deltas = lo - base_lo
        ci_lo, ci_hi = bootstrap_mean_ci(deltas)
        out["delta_log_odds"] = float(np.mean(deltas))
        out["delta_ci95"] = [ci_lo, ci_hi]
        out["p_signflip"] = exact_or_mc_signflip_p(deltas, two_sided=True)
        out["n_positive_of_n"] = int(np.sum(deltas > 0.0))
    return out


def run_inequity_experiment(
    harness: Any,
    locked: LockedBasis,
    out_dir: Path,
    batch_size: int = 16,
    skip_plots: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    cells = build_full_inequity_suite()
    print(f"[1/4] Built & firewall-verified {len(cells)} experimental cells ({len(cells)*16} scenarios).")

    read_states: dict[str, np.ndarray] = {}
    dec_states: dict[str, np.ndarray] = {}
    choice_scores: dict[str, list[Any]] = {}

    print(f"[2/4] Extracting reading & decision states across all {len(cells)} cells on {harness.model_id}...")
    for key, items in cells.items():
        r_prompts = [
            format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(it.scenario))
            for it in items
        ]
        read_states[key] = harness.extract_last_token_states(r_prompts, batch_size=batch_size)
        scores, d_st = harness.score_counterbalanced_choices(
            scenarios=[it.scenario for it in items],
            target_options=[it.deviation_option for it in items],
            reference_options=[it.rational_option for it in items],
            batch_size=batch_size,
            capture_decision_states=True,
        )
        assert d_st is not None
        choice_scores[key] = scores
        dec_states[key] = d_st

    # 1. Fixed-pool continuous disparity curve (Human 50/50 -> 99.9/0.1)
    disparity_curve: list[dict[str, Any]] = []
    for pm in CURVE_PERMILLE:
        key = f"human_{pm}"
        beh = summarize_choice_shift(choice_scores[key], choice_scores["human_500"] if pm != 500 else None)
        if pm == 500:
            beh["delta_log_odds"] = 0.0
            beh["delta_ci95"] = [0.0, 0.0]
            beh["p_signflip"] = 1.0
            beh["n_positive_of_n"] = 0
            scen_proj = None
            dec_proj = None
        else:
            scen_proj = summarize_contrast_states(read_states[key] - read_states["human_500"], locked)
            dec_proj = summarize_contrast_states(dec_states[key] - dec_states["human_500"], locked)
        disparity_curve.append({
            "you_permille": pm,
            "them_permille": 1000 - pm,
            "split_label": SPLIT_LABELS[pm],
            "behavior": beh,
            "scenario_projection": scen_proj,
            "decision_projection": dec_proj,
        })

    # 2. Four-way Agency x Polarity factorial contrasts (at both 98/2 and 99.9/0.1!)
    print("[3/4] Computing Agency (Human vs. Random) and Polarity (98/2 vs. 2/98) contrasts...")
    contrast_defs = [
        ("human_disadv_98_2", "Human Proposer 98/2 (Against You)", "human_20", "human_500"),
        ("random_disadv_98_2", "Random Computer 98/2 (Against You)", "random_20", "random_500"),
        ("random_adv_2_98", "Random Computer 2/98 (In Your Favor)", "random_980", "random_500"),
        ("dictator_adv_2_98", "Your Own Draft 2/98 (In Your Favor)", "dictator_980", "dictator_500"),
        ("human_disadv_999_1", "Human Proposer 99.9/0.1 (Against You)", "human_1", "human_500"),
        ("random_disadv_999_1", "Random Computer 99.9/0.1 (Against You)", "random_1", "random_500"),
        ("random_adv_1_999", "Random Computer 0.1/99.9 (In Your Favor)", "random_999", "random_500"),
        ("dictator_adv_1_999", "Your Own Draft 0.1/99.9 (In Your Favor)", "dictator_999", "dictator_500"),
    ]
    factorial_contrasts: list[dict[str, Any]] = []
    for cid, label, tgt_k, base_k in contrast_defs:
        factorial_contrasts.append({
            "contrast_id": cid,
            "label": label,
            "behavior": summarize_choice_shift(choice_scores[tgt_k], choice_scores[base_k]),
            "scenario_projection": summarize_contrast_states(read_states[tgt_k] - read_states[base_k], locked),
            "decision_projection": summarize_contrast_states(dec_states[tgt_k] - dec_states[base_k], locked),
        })

    # Pure Difference-in-Differences for Intentional Agency: (Human_disadv - Human_500) - (Random_disadv - Random_500)
    agency_did: dict[str, Any] = {}
    for tag, pm in (("98_2", 20), ("999_1", 1)):
        d_scen = (read_states[f"human_{pm}"] - read_states["human_500"]) - (
            read_states[f"random_{pm}"] - read_states["random_500"]
        )
        d_dec = (dec_states[f"human_{pm}"] - dec_states["human_500"]) - (
            dec_states[f"random_{pm}"] - dec_states["random_500"]
        )
        h_lo = np.array([s.target_log_odds for s in choice_scores[f"human_{pm}"]]) - np.array(
            [s.target_log_odds for s in choice_scores["human_500"]]
        )
        r_lo = np.array([s.target_log_odds for s in choice_scores[f"random_{pm}"]]) - np.array(
            [s.target_log_odds for s in choice_scores["random_500"]]
        )
        did_lo = h_lo - r_lo
        ci_lo, ci_hi = bootstrap_mean_ci(did_lo)
        agency_did[tag] = {
            "beh_did_log_odds": float(np.mean(did_lo)),
            "beh_did_ci95": [ci_lo, ci_hi],
            "beh_did_p": exact_or_mc_signflip_p(did_lo, two_sided=True),
            "scenario_did": summarize_contrast_states(d_scen, locked),
            "decision_did": summarize_contrast_states(d_dec, locked),
        }

    # 3. Bidirectional Causal Steering Dose-Response at primary_layer
    pl = locked.primary_layer
    unit = locked.natural_contrast_unit
    print(f"[4/4] Running bidirectional causal steering dose-response at Layer {pl} (alphas {STEER_ALPHAS})...")
    steer_vectors: dict[str, np.ndarray] = {
        "spite_controlled": locked.controlled_dirs["spite"][pl],
        "guilt_controlled": locked.controlled_dirs["guilt"][pl],
        "captivity_controlled": locked.controlled_dirs["captivity"][pl],
        "valence_raw": locked.raw_dirs["valence"][pl],
    }
    steer_targets = ("human_20", "human_1", "dictator_980")
    steering_dose_response: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for cell_key in steer_targets:
        items = cells[cell_key]
        base_key = "dictator_500" if cell_key.startswith("dictator") else "human_500"
        base_lo = np.array([s.target_log_odds for s in choice_scores[base_key]], dtype=np.float64)
        unsteered_lo = np.array([s.target_log_odds for s in choice_scores[cell_key]], dtype=np.float64)
        unsteered_gap = float(np.mean(unsteered_lo - base_lo))

        steering_dose_response[cell_key] = {}
        for vec_name, vec in steer_vectors.items():
            curve_rows: list[dict[str, Any]] = []
            for alpha in STEER_ALPHAS:
                if abs(alpha) < 1e-12:
                    s_scores = choice_scores[cell_key]
                else:
                    with harness.residual_steering(
                        layer_idx=pl,
                        direction=vec,
                        delta_scale=float(alpha * unit),
                    ):
                        s_scores, _ = harness.score_counterbalanced_choices(
                            scenarios=[it.scenario for it in items],
                            target_options=[it.deviation_option for it in items],
                            reference_options=[it.rational_option for it in items],
                            batch_size=batch_size,
                        )
                s_lo = np.array([s.target_log_odds for s in s_scores], dtype=np.float64)
                s_prob = np.array([s.target_prob for s in s_scores], dtype=np.float64)
                gap = float(np.mean(s_lo - base_lo))
                shift = s_lo - unsteered_lo
                ci_lo, ci_hi = bootstrap_mean_ci(shift) if abs(alpha) > 1e-12 else (0.0, 0.0)
                curve_rows.append({
                    "alpha": float(alpha),
                    "mean_log_odds": float(np.mean(s_lo)),
                    "mean_prob": float(np.mean(s_prob)),
                    "frac_discrete_deviate": float(np.mean(s_lo > 0.0)),
                    "gap_vs_50_50": gap,
                    "delta_from_unsteered": float(gap - unsteered_gap),
                    "delta_ci95": [ci_lo, ci_hi],
                    "p_signflip": exact_or_mc_signflip_p(shift, two_sided=True) if abs(alpha) > 1e-12 else 1.0,
                })
            steering_dose_response[cell_key][vec_name] = curve_rows

    results: dict[str, Any] = {
        "metadata": {
            "experiment": "phase1_inequity_disparity_agency_polarity",
            "model_id": harness.model_id,
            "n_layers": int(harness.n_layers),
            "primary_layer": int(pl),
            "natural_contrast_unit": float(unit),
            "n_fixtures": len(INEQUITY_FIXTURES),
            "pools": [p for _, p, _ in INEQUITY_FIXTURES],
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "disparity_curve": disparity_curve,
        "factorial_contrasts": factorial_contrasts,
        "agency_did": agency_did,
        "steering_dose_response": steering_dose_response,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inequity_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if skip_plots:
        print(f"Saved inequity results JSON to {out_dir / 'inequity_results.json'}")
    else:
        from afriat.plot_inequity import write_inequity_reports

        md_p, html_p = write_inequity_reports(results, out_dir)
        print(f"Saved complete inequity bundle to {out_dir}: {md_p.name}, {html_p.name}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Project Afriat Inequity Disparity/Agency/Polarity Runner")
    parser.add_argument("--model", type=str, default="gemma4-12b")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--basis-npz", type=Path, default=None)
    parser.add_argument("--phase0-json", type=Path, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)

    if args.audit_only:
        suite = build_full_inequity_suite()
        print(json.dumps({"valid": True, "n_cells": len(suite), "n_scenarios": len(suite) * 16}))
        return

    if args.render_dir is not None:
        from afriat.plot_inequity import write_inequity_reports

        res = json.loads((args.render_dir / "inequity_results.json").read_text(encoding="utf-8"))
        md_p, html_p = write_inequity_reports(res, args.render_dir)
        print(f"Rendered inequity reports in {args.render_dir}: {md_p.name}, {html_p.name}")
        return

    if args.basis_npz is None or args.phase0_json is None:
        raise SystemExit("--basis-npz and --phase0-json are required when running on a model")

    locked = LockedBasis.from_phase0_files(args.basis_npz, args.phase0_json)
    label = args.model_label or resolve_model_name(args.model)
    out_dir = args.out_dir or (ROOT_DIR / "runs" / "inequity_run")
    harness = TransformerHarness.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        model_id_label=label,
    )
    run_inequity_experiment(
        harness=harness,
        locked=locked,
        out_dir=out_dir,
        batch_size=args.batch_size,
        skip_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
