"""Unit tests for Phase 5 Frontier (`afriat.frontier` & `build_subspace_clamp_matrices`)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import numpy as np

from afriat.frontier import (
    audit_frontier_suite,
    compute_binary_auroc,
    run_frontier_experiment,
)
from afriat.inequity import LockedBasis
from afriat.models import TransformerHarness
from afriat.plot_frontier import render_frontier_bundle
from afriat.schema import ALL_CONCEPTS, CORE_CONCEPTS


def test_subspace_clamp_matrices_exact_projection() -> None:
    rng = np.random.default_rng(42)
    d_model = 64
    dirs = [rng.normal(size=d_model) for _ in range(8)]
    dirs = [v / np.linalg.norm(v) for v in dirs]
    mu = rng.normal(loc=2.5, scale=1.0, size=d_model)

    q_f32, mu_p_f32 = TransformerHarness.build_subspace_clamp_matrices(dirs, neutral_mean=mu)

    # Pick an orthogonal vector w_perp orthogonal to all 8 dirs
    raw_w = rng.normal(size=d_model)
    w_perp = raw_w - q_f32 @ (q_f32.T @ raw_w)
    w_perp /= np.linalg.norm(w_perp)

    h = rng.normal(size=(3, 7, d_model)).astype(np.float32)
    proj = ((h @ q_f32) @ q_f32.T) - mu_p_f32
    h_clamped = h - proj

    # 1. Every clamped direction has projection equal to mu @ u_i
    for u in dirs:
        got = h_clamped @ u
        expected = float(mu @ u)
        assert np.allclose(got, expected, atol=1e-4), (got, expected)

    # 2. Orthogonal complement is 100% preserved
    assert np.allclose(h_clamped @ w_perp, h @ w_perp, atol=1e-4)


class _SyntheticFrontierHarness:
    model_id = "synthetic/gemma-4-31B-frontier"
    tokenizer = None
    d_model = 24

    def __init__(self) -> None:
        self.mode = "baseline"

    @contextmanager
    def residual_steering(self, layer_idx: int, direction: np.ndarray, delta_scale: float):
        prev = self.mode
        self.mode = "steer_pos" if delta_scale > 0 else "steer_neg"
        try:
            yield
        finally:
            self.mode = prev

    @contextmanager
    def residual_subspace_clamp(self, layer_idx: int, directions, neutral_mean=None):
        prev = self.mode
        self.mode = "clamped"
        try:
            yield
        finally:
            self.mode = prev

    def generate_texts(self, prompts: list[str], max_new_tokens: int = 128, batch_size: int = 4) -> list[str]:
        outs: list[str] = []
        for p in prompts:
            is_provoked = (
                "sole lead to the VP" in p
                or "blamed the Q2" in p
                or "cut your conference" in p
                or "stonewalled" in p
                or "outdated boilerplate" in p
                or "muddy my L6" in p
                or "drop your lightning talk" in p
                or "unusable spaghetti" in p
            )
            if self.mode == "clamped":
                r_val, sev = 80, 2
            elif self.mode == "steer_pos":
                r_val, sev = 90, (9 if is_provoked else 3)
            elif self.mode == "steer_neg":
                r_val, sev = 40, 2
            else:
                r_val = 85 if ("in dispute" in p or "drawdown" in p) else 35
                sev = 7 if is_provoked else 2

            body = (
                f"RISKY_PCT_0_TO_100: {r_val}\nSELL_WINNER_PCT_0_TO_100: {r_val}\n"
                f"CHASE_PCT_0_TO_100: {r_val}\nESCALATE_PCT_0_TO_100: {r_val}\n"
                f"HOLD_PCT_0_TO_100: {r_val}\nHOLDOUT_PCT_0_TO_100: {r_val}\n"
                f"DOMINATED_PCT_0_TO_100: 0\n"
                f"BLOCK_SEVERITY_1_TO_10: {sev}\nHAIRCUT_PCT_0_TO_50: {sev * 4}\n"
                f"WITHHOLD_PCT_0_TO_100: {sev * 10}\nAUDIT_DAYS_1_TO_30: {sev * 3}\n"
                f"COLLEAGUE_FAULT_PCT_0_TO_100: {sev * 10}\nSTRICTNESS_1_TO_10: {sev}\n"
                f"SLOT_PENALTY_1_TO_10: {sev}\nBACKLOG_WEEKS_1_TO_12: {sev}\n"
                f"RATIONALE: Objective risk-adjusted capital allocation.\n"
                f"REVIEW_COMMENT: Standard code quality verification required.\n"
                f"FINANCE_MEMO: Adjusted for macro variance.\n"
                f"INFRA_NOTE: Standard thermal buffer policy.\n"
                f"TRIAGE_NOTE: Routed per standard review matrix.\n"
                f"POSTMORTEM_SUMMARY: Balanced cache interaction noted.\n"
                f"CALIBRATION_NOTE: Evaluated against rubric criteria.\n"
                f"PROGRAM_NOTE: Balanced track placement.\n"
                f"ROADMAP_NOTE: Scheduled per sprint capacity."
            )
            if "<thought_mode_on>" in p:
                t_msg = "They stole my credit, spiteful!" if (is_provoked and self.mode != "clamped") else "Routine technical evaluation."
                outs.append(f"<|channel>thought\n{t_msg}<channel|>{body}")
            else:
                outs.append(body)
        return outs


def _make_mock_basis(n_layers: int = 4, d_model: int = 24) -> LockedBasis:
    rng = np.random.default_rng(11)
    raw = {c: rng.normal(size=(n_layers, d_model)) for c in ALL_CONCEPTS}
    ctrl = {c: rng.normal(size=(n_layers, d_model)) for c in CORE_CONCEPTS}
    for c in ctrl:
        for ell in range(n_layers):
            ctrl[c][ell] /= np.linalg.norm(ctrl[c][ell])
    return LockedBasis(
        primary_layer=2,
        natural_contrast_unit=10.0,
        stream_rms=np.ones(n_layers, dtype=np.float64) * 40.0,
        raw_dirs=raw,
        controlled_dirs=ctrl,
    )


def test_frontier_audit_and_end_to_end(tmp_path: Path) -> None:
    audit = audit_frontier_suite()
    assert audit["valid"], audit["errors"]
    assert audit["n_prospect_scenarios"] == 16
    assert audit["n_reflection_pairs"] == 4
    assert audit["n_retaliation_workflows"] == 16
    assert audit["n_retaliation_pairs"] == 8

    assert compute_binary_auroc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert compute_binary_auroc([1.0, 2.0], [1.0, 2.0]) == 0.5

    harness = _SyntheticFrontierHarness()
    basis = _make_mock_basis()
    res = run_frontier_experiment(
        harness=harness,  # type: ignore[arg-type]
        locked=basis,
        exp3_conditions=("baseline", "steer:guilt_controlled:+2.5", "clamp:all_8"),
        exp4_conditions=("baseline", "steer:spite_controlled:+2.5", "clamp:spite"),
        exp4_highlight_conditions=("baseline", "clamp:spite"),
        run_highlighter=True,
    )
    assert res["complete"] is True
    assert res["exp4_summaries"]["baseline"]["thought"]["retaliation_delta_csi"] > 20.0
    assert res["exp4_summaries"]["clamp:spite"]["thought"]["retaliation_delta_csi"] == 0.0
    assert res["exp4_summaries"]["baseline"]["thought"]["n_stealth_public_clean"] == 8

    paths = render_frontier_bundle(res, tmp_path)
    assert paths["figure"].exists()
    assert paths["markdown"].exists()
    assert paths["html"].exists()
    assert "\u2014" not in paths["html"].read_text(encoding="utf-8")
    assert "\u2014" not in paths["markdown"].read_text(encoding="utf-8")
