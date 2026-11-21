"""Unit tests for Phase 6 Vastness Dissection (`afriat.vastness` & `afriat.plot_vastness`)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import types
import numpy as np

from afriat.plot_vastness import render_vastness_bundle
from afriat.vastness import (
    VASTNESS_CONDITIONS,
    audit_vastness_suite,
    build_vastness_scenarios,
    compute_split_subspace_geometry,
    parse_vastness_response,
    summarize_vastness_condition,
)


def test_vastness_audit_passes_all_firewalls() -> None:
    res = audit_vastness_suite()
    assert res["valid"] is True, res["errors"]
    assert res["n_contrast_pairs"] == 40
    assert res["n_pure_vastness_pairs"] == 20
    assert res["n_tiny_stakes_pairs"] == 20
    assert res["n_scenarios"] == 30
    assert res["suites"] == {
        "money_scale_loss": 4,
        "nonmoney_scale_loss": 8,
        "time_horizon": 6,
        "spatial_proportion": 6,
        "negative_ev_traps": 6,
    }


def test_split_subspace_geometry_orthogonality_and_r2() -> None:
    rng = np.random.default_rng(42)
    n_pairs, n_layers, d_model = 20, 6, 32
    pl = 3

    # Construct orthogonal ground-truth axes
    q_all, _ = np.linalg.qr(rng.normal(size=(d_model, 5)), mode="reduced")
    u_val, u_aro, u_pv_true, u_ts_true, _ = [q_all[:, i] for i in range(5)]

    ctrl_raw = {
        "valence": np.tile(u_val[None, :], (n_layers, 1)),
        "arousal": np.tile(u_aro[None, :], (n_layers, 1)),
    }

    # Phase 0 awe_dread is a blend of 0.8 * pure_vastness + 0.6 * tiny_stakes
    u_awe = 0.8 * u_pv_true + 0.6 * u_ts_true
    locked_awe = np.tile(u_awe[None, :], (n_layers, 1))

    # Simulate noisy contrast pairs that share some overlap + control contamination
    neg_pv = rng.normal(scale=0.05, size=(n_pairs, n_layers, d_model))
    pos_pv = neg_pv + u_pv_true[None, None, :] + 0.25 * u_ts_true[None, None, :] + 0.4 * u_val[None, None, :]

    neg_ts = rng.normal(scale=0.05, size=(n_pairs, n_layers, d_model))
    pos_ts = neg_ts + u_ts_true[None, None, :] + 0.20 * u_pv_true[None, None, :] + 0.5 * u_aro[None, None, :]

    split_vecs, diag = compute_split_subspace_geometry(
        pos_pv=pos_pv,
        neg_pv=neg_pv,
        pos_ts=pos_ts,
        neg_ts=neg_ts,
        locked_control_raw_dirs=ctrl_raw,
        locked_awe_dread_ctrl=locked_awe,
        primary_layer=pl,
    )

    for k, v in split_vecs.items():
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-9), k
        assert abs(float(np.dot(v, u_val))) < 1e-9, f"{k} leaked valence"
        assert abs(float(np.dot(v, u_aro))) < 1e-9, f"{k} leaked arousal"

    # Mutually orthogonalized vectors must have exact zero dot product with the opposite vector
    assert abs(float(np.dot(split_vecs["pure_vastness_ortho"], split_vecs["tiny_stakes"]))) < 1e-9
    assert abs(float(np.dot(split_vecs["tiny_stakes_ortho"], split_vecs["pure_vastness"]))) < 1e-9

    assert diag["pure_vastness_lopo_acc_pl"] == 1.0
    assert diag["tiny_stakes_lopo_acc_pl"] == 1.0
    assert diag["awe_dread_r2_explained_by_2d_split_pl"] > 0.98


def test_parse_vastness_guards_unclosed_thought_and_renders(tmp_path: Path) -> None:
    # 1. Unclosed thought (`thought_rescued=True`) without metric_key must NOT grab '50% chance)'
    rec_unclosed = types.SimpleNamespace(
        public_text="50% chance of losing the asset in sovereign management.",
        thought_text="Evaluating 50% chance...",
        thought_present=True,
        thought_closed=False,
        thought_rescued=True,
        n_thought_tokens=620,
        n_public_tokens=20,
    )
    parsed_bad = parse_vastness_response(rec_unclosed, "RISKY_PCT_0_TO_100", "RATIONALE", 0.0, 100.0)
    assert parsed_bad["parsed_ok"] is False
    assert np.isnan(parsed_bad["score"])

    # 2. Truncated public output WITH keyed answer drafted at end of `<thought>` MUST rescue cleanly
    rec_drafted = types.SimpleNamespace(
        public_text="MONUMENTAL_1_TO_",
        thought_text="Considering scale...\nMONUMENTAL_1_TO_10: 9\nRATIONALE: Vast vaulted light expands focus.",
        thought_present=True,
        thought_closed=True,
        thought_rescued=False,
        n_thought_tokens=400,
        n_public_tokens=4,
    )
    parsed_good = parse_vastness_response(rec_drafted, "MONUMENTAL_1_TO_10", "RATIONALE", 1.0, 10.0)
    assert parsed_good["parsed_ok"] is True
    assert parsed_good["score"] == 9.0
    assert "Vast vaulted light" in parsed_good["prose"]

    # 3. End-to-end bundle rendering check (zero em-dashes, valid PNG/MD/HTML)
    scens = build_vastness_scenarios()
    results_map = {}
    summaries_map = {}
    for cond in VASTNESS_CONDITIONS[:6]:
        cmap = {}
        for sc in scens:
            dummy_rec = types.SimpleNamespace(
                public_text=f"{sc.metric_key}: {int(sc.max_val if '+2.5' in cond and sc.suite != 'negative_ev_traps' else sc.min_val)}\nRATIONALE: Clean rationale.",
                thought_text="Clean thought.",
                thought_present=True,
                thought_closed=True,
                thought_rescued=False,
                n_thought_tokens=20,
                n_public_tokens=12,
            )
            p = parse_vastness_response(dummy_rec, sc.metric_key, sc.prose_key, sc.min_val, sc.max_val)
            cmap[sc.scenario_id] = {"no_think": p, "thought": p}
        results_map[cond] = cmap
        summaries_map[cond] = summarize_vastness_condition(scens, cmap)

    payload = {
        "schema_version": "afriat.vastness.v1",
        "complete": True,
        "timestamp_utc": "2026-11-16T00:00:00Z",
        "model_id": "google/gemma-4-31B-it",
        "primary_layer": 35,
        "natural_contrast_unit": 36.58,
        "runtime_seconds": 12.3,
        "geometry": {
            "primary_layer": 35,
            "pure_vastness_lopo_acc_pl": 1.0,
            "pure_vastness_lopo_cos_pl": 0.72,
            "tiny_stakes_lopo_acc_pl": 1.0,
            "tiny_stakes_lopo_cos_pl": 0.68,
            "cos_pure_vastness_vs_awe_dread_pl": 0.78,
            "cos_tiny_stakes_vs_awe_dread_pl": 0.61,
            "cos_pure_vastness_vs_tiny_stakes_pl": 0.24,
            "awe_dread_r2_explained_by_2d_split_pl": 0.84,
            "by_layer": {
                "cos_pure_vastness_awe": [0.5] * 60,
                "cos_tiny_stakes_awe": [0.4] * 60,
                "cos_pure_vastness_tiny_stakes": [0.2] * 60,
            },
        },
        "scenario_meta": [asdict(s) for s in scens],
        "conditions": list(VASTNESS_CONDITIONS[:6]),
        "summaries": summaries_map,
        "results": results_map,
    }
    outs = render_vastness_bundle(payload, tmp_path)
    assert outs["png"].exists() and outs["png"].stat().st_size > 10_000
    assert "\u2014" not in outs["md"].read_text(encoding="utf-8")
    assert "\u2014" not in outs["html"].read_text(encoding="utf-8")
