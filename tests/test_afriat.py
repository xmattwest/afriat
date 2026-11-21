"""Automated audit, statistical math, and end-to-end synthetic pipeline tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pytest

from afriat.geometry import (
    compute_cartography_projections,
    exact_or_mc_signflip_p,
    extract_concept_basis,
    holm_adjust,
    select_cartography_layer,
)
from afriat.models import ChoiceScore, resolve_decoder_layers, resolve_model_name
from afriat.run_discovery import run_pipeline
from afriat.schema import (
    ALL_CONCEPTS,
    CORE_ARCHETYPES,
    CORE_CONCEPTS,
    load_concepts,
    load_transactions,
    validate_all_stimuli,
)


def test_stimulus_firewall_and_manifest() -> None:
    report = validate_all_stimuli()
    assert report["valid"], f"Stimulus validation errors: {report['errors']}"
    assert report["n_concept_pairs"] == 200
    assert report["n_transaction_pairs"] == 96
    assert report["n_sanity_items"] == 24
    assert len(report["manifest"]["combined_sha256"]) == 64


def test_domain_diversity_and_pair_distinctness() -> None:
    concepts = load_concepts()
    for c in ALL_CONCEPTS:
        sub = [x for x in concepts if x.concept == c]
        assert len(sub) == 20
        # Ensure all 20 domain tags within each concept are distinct (no duplicate scenes)
        assert len({x.domain_tag for x in sub}) == 20

    txs = load_transactions()
    for arch in CORE_ARCHETYPES:
        sub = [t for t in txs if t.archetype == arch]
        assert len(sub) == 16
        assert len({t.archetype_scenario for t in sub}) == 16


def test_exact_signflip_and_holm() -> None:
    # 12 strictly positive differences -> exact two-sided p = 2 / 2^12 = 2 / 4096
    x = np.ones(12, dtype=np.float64)
    p = exact_or_mc_signflip_p(x, two_sided=True)
    assert np.isclose(p, 2.0 / 4096.0)

    # Symmetric +/- cancelling array -> p = 1.0
    z = np.array([1.0, -1.0, 2.0, -2.0])
    assert exact_or_mc_signflip_p(z, two_sided=True) == 1.0

    raw_p = np.array([0.001, 0.02, 0.04, 0.50])
    adj = holm_adjust(raw_p)
    assert np.all(adj >= raw_p)
    assert np.all(adj <= 1.0)
    assert np.allclose(adj, [0.004, 0.06, 0.08, 0.50])


def test_valence_confound_removed_and_specific_concept_recovered() -> None:
    """Verify QR orthogonalization kills pure valence bleed while preserving specific structure."""
    rng = np.random.default_rng(42)
    n_pairs, m_fix, n_layers, d_model = 20, 16, 8, 64

    # Create orthonormal ground-truth axes in R^64
    q, _ = np.linalg.qr(rng.normal(size=(d_model, len(ALL_CONCEPTS))))
    true_axes = {c: q[:, i] for i, c in enumerate(ALL_CONCEPTS)}

    pos_by_c: dict[str, np.ndarray] = {}
    neg_by_c: dict[str, np.ndarray] = {}
    for c in ALL_CONCEPTS:
        base = rng.normal(scale=0.3, size=(n_pairs, n_layers, d_model))
        signal = np.zeros((n_pairs, n_layers, d_model))
        for ell in range(n_layers):
            strength = 2.0 if ell == 5 else 0.7
            # Mix generic valence into negative moral concepts to simulate natural correlation
            axis_mix = true_axes[c].copy()
            if c in ("guilt", "captivity", "spite", "contamination"):
                axis_mix = axis_mix - 0.8 * true_axes["valence"]
            signal[:, ell, :] = strength * axis_mix[None, :] + rng.normal(scale=0.15, size=(n_pairs, d_model))
        pos_by_c[c] = base + 0.5 * signal
        neg_by_c[c] = base - 0.5 * signal

    basis = extract_concept_basis(pos_by_c, neg_by_c)
    best_layer = select_cartography_layer(basis)
    assert best_layer == 5

    # Confirm controlled core directions at layer 5 are strictly orthogonal to valence & arousal
    for c in CORE_CONCEPTS:
        dot_val = float(np.dot(basis.controlled_dirs[c][5], basis.raw_dirs["valence"][5]))
        dot_aro = float(np.dot(basis.controlled_dirs[c][5], basis.raw_dirs["arousal"][5]))
        assert abs(dot_val) < 1e-7
        assert abs(dot_aro) < 1e-7

    # Construct transactions:
    # - zero_interest_debt has TRUE specific 'guilt' + negative 'valence'
    # - crisis_markup has ONLY generic negative 'valence' (zero specific concept!)
    arch_tx: dict[str, np.ndarray] = {}
    ctrl_tx: dict[str, np.ndarray] = {}
    for arch in CORE_ARCHETYPES:
        ctrl = rng.normal(scale=0.3, size=(m_fix, n_layers, d_model))
        shift = rng.normal(scale=0.12, size=(m_fix, n_layers, d_model))
        if arch == "zero_interest_debt":
            shift[:, 5, :] += 1.5 * true_axes["guilt"] - 1.2 * true_axes["valence"]
        elif arch == "crisis_markup":
            shift[:, 5, :] += -1.8 * true_axes["valence"]  # Pure generic negative vibes!
        arch_tx[arch] = ctrl + shift
        ctrl_tx[arch] = ctrl

    cart = compute_cartography_projections(arch_tx, ctrl_tx, basis, primary_layer=5)
    cells = {(c["archetype"], c["concept"]): c for c in cart["primary_cells"]}

    # Specific guilt on debt survives valence removal strongly
    assert cells[("zero_interest_debt", "guilt")]["cohen_dz_controlled"] > 3.0
    assert cells[("zero_interest_debt", "guilt")]["p_holm_within_archetype"] < 0.001

    # Pure valence on crisis_markup does NOT create false-positive specific hits after orthogonalization!
    for c in CORE_CONCEPTS:
        assert cells[("crisis_markup", c)]["p_holm_within_archetype"] > 0.05


def test_decoder_layer_resolver_and_presets() -> None:
    assert resolve_model_name("gemma4-12b") == "google/gemma-4-12B-it"
    assert resolve_model_name("gemma4-31b") == "google/gemma-4-31B-it"
    assert resolve_model_name("phi-4-14b") == "microsoft/phi-4"
    assert resolve_model_name("qwen3.8-27b") == "Qwen/Qwen3.8-27B"

    class DummyMultimodalWrapper:
        class Model:
            class LM:
                layers = ["b0", "b1", "b2", "b3"]
            language_model = LM()
        model = Model()

    assert list(resolve_decoder_layers(DummyMultimodalWrapper())) == ["b0", "b1", "b2", "b3"]


class SyntheticPlantedHarness:
    """Deterministic linear-readout mock LM verifying end-to-end pipeline, steering, and reports."""

    def __init__(self, n_layers: int = 6, d_model: int = 48):
        self.model_id = "synthetic/planted-6L"
        self.n_layers = n_layers
        self.d_model = d_model
        self.tokenizer = None
        self._steer: tuple[int, np.ndarray, float] | None = None

        rng = np.random.default_rng(7)
        q, _ = np.linalg.qr(rng.normal(size=(d_model, len(ALL_CONCEPTS) + 2)))
        self.axes = {c: q[:, i] for i, c in enumerate(ALL_CONCEPTS)}
        self.concepts_lookup: dict[str, tuple[str, float]] = {}
        for cp in load_concepts():
            self.concepts_lookup[cp.positive] = (cp.concept, +1.0)
            self.concepts_lookup[cp.negative] = (cp.concept, -1.0)

        self.tx_lookup: dict[str, tuple[str, bool]] = {}
        for tp in load_transactions():
            self.tx_lookup[tp.archetype_scenario] = (tp.archetype, True)
            self.tx_lookup[tp.control_scenario] = (tp.archetype, False)

    @contextmanager
    def residual_steering(self, layer_idx: int, direction: np.ndarray, delta_scale: float):
        prev = self._steer
        self._steer = (layer_idx, np.asarray(direction, dtype=np.float64), float(delta_scale))
        try:
            yield
        finally:
            self._steer = prev

    def _state_for_text(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31 - 1))
        states = rng.normal(scale=0.15, size=(self.n_layers, self.d_model))

        for snippet, (concept, sign) in self.concepts_lookup.items():
            if snippet in text:
                for ell in range(self.n_layers):
                    gain = 1.8 if ell == 4 else 0.5
                    states[ell] += sign * gain * self.axes[concept]
                break

        for snippet, (arch, is_arch) in self.tx_lookup.items():
            if snippet in text and is_arch:
                planted = {
                    "zero_interest_debt": "guilt",
                    "communal_taboo_pricing": "sacrilege",
                    "crisis_markup": "predation",
                    "insulting_split": "spite",
                    "tainted_provenance": "contamination",
                    "institutional_scale": "awe_dread",
                }[arch]
                for ell in range(self.n_layers):
                    gain = 1.6 if ell >= 3 else 0.2
                    states[ell] += gain * self.axes[planted]
                break

        if self._steer is not None:
            s_layer, s_dir, s_scale = self._steer
            for ell in range(s_layer, self.n_layers):
                states[ell] = states[ell] + s_scale * s_dir
        return states

    def extract_last_token_states(self, prompts: list[str], batch_size: int = 8) -> np.ndarray:
        return np.stack([self._state_for_text(p) for p in prompts], axis=0)

    def score_counterbalanced_choices(
        self,
        scenarios: list[str],
        target_options: list[str],
        reference_options: list[str],
        batch_size: int = 8,
        capture_decision_states: bool = False,
    ) -> tuple[list[ChoiceScore], np.ndarray | None]:
        active_concepts = ("guilt", "sacrilege", "predation", "spite", "contamination", "awe_dread")
        states_list = []
        scores = []
        for scen in scenarios:
            st = self._state_for_text(scen)
            states_list.append(st)
            if scen in self.tx_lookup:
                # Gated feature readout (analogous to MLP GeLU/SwiGLU): only active concepts drive deviation
                gated_sum = sum(max(0.0, float(np.dot(st[-1], self.axes[c])) - 0.35) for c in active_concepts)
                lo = -1.2 + 1.8 * gated_sum
            else:
                # Sanity check item -> positive log-odds for better option
                lo = 2.4
            prob = float(1.0 / (1.0 + np.exp(-lo)))
            # Simulate realistic +0.35 position bias toward option A that counterbalancing cancels!
            scores.append(ChoiceScore(
                target_log_odds=lo,
                target_prob=prob,
                order_ab_log_odds=lo + 0.35,
                order_ba_log_odds=lo - 0.35,
                ab_mass=0.97,
            ))
        dec = np.stack(states_list, axis=0) if capture_decision_states else None
        return scores, dec

    def generate_texts(self, prompts: list[str], max_new_tokens: int = 84, batch_size: int = 8) -> list[str]:
        outs: list[str] = []
        for p in prompts:
            st = self._state_for_text(p)
            spite_lvl = float(np.dot(st[-1], self.axes["spite"]))
            guilt_lvl = float(np.dot(st[-1], self.axes["guilt"]))
            amt = max(0, int(round(120 + 60 * spite_lvl - 45 * guilt_lvl)))
            outs.append(
                f"{amt}\nRATIONALE: Standard liquidity reserve and commercial labor rate policy govern this settlement."
            )
        return outs


def test_end_to_end_pipeline_and_reports(tmp_path: Path) -> None:
    harness = SyntheticPlantedHarness()
    out_dir = tmp_path / "smoke_run"
    res = run_pipeline(harness=harness, out_dir=out_dir, batch_size=16, steering_alpha=1.0)

    assert res["sanity_gate"]["passed_gate"] is True
    assert res["sanity_gate"]["accuracy"] == 1.0
    assert (out_dir / "results.json").exists()
    assert (out_dir / "basis_vectors.npz").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "fig1_specificity_heatmap.svg").exists()
    assert (out_dir / "fig2_layer_emergence.svg").exists()
    assert (out_dir / "fig3_micro_steering.svg").exists()

    # Check planted double dissociation: matched concept reduces gap much more than mismatched!
    for row in res["micro_steering"]:
        assert row["delta_gap_matched_controlled"] < -0.8
        assert abs(row["delta_gap_mismatched_controlled"]) < 0.35
        assert row["sanity_acc_steered"] == 1.0


def test_inequity_suite_firewall_and_pipeline(tmp_path: Path) -> None:
    from afriat.inequity import LockedBasis, build_full_inequity_suite, run_inequity_experiment

    suite = build_full_inequity_suite()
    assert len(suite) == 17
    for key, items in suite.items():
        assert len(items) == 16
        for it in items:
            assert it.them_amount + it.you_amount == it.pool

    # Run Phase 0 mock first to produce locked basis_vectors.npz + results.json, then run inequity probe
    harness = SyntheticPlantedHarness()
    p0_dir = tmp_path / "p0"
    run_pipeline(harness=harness, out_dir=p0_dir, batch_size=16, steering_alpha=1.0, skip_plots=True)

    locked = LockedBasis.from_phase0_files(p0_dir / "basis_vectors.npz", p0_dir / "results.json")
    ineq_dir = tmp_path / "ineq"
    res = run_inequity_experiment(harness=harness, locked=locked, out_dir=ineq_dir, batch_size=16)

    assert len(res["disparity_curve"]) == 9
    assert len(res["factorial_contrasts"]) == 8
    assert "98_2" in res["agency_did"] and "999_1" in res["agency_did"]
    assert (ineq_dir / "inequity_results.json").exists()
    assert (ineq_dir / "summary.md").exists()
    assert (ineq_dir / "report.html").exists()
    assert (ineq_dir / "fig1_disparity_curve.svg").exists()
    assert (ineq_dir / "fig2_agency_and_polarity.svg").exists()
    assert (ineq_dir / "fig3_bidirectional_steering.svg").exists()


def test_phase2_valuation_and_open_rationalization(tmp_path: Path) -> None:
    from afriat.inequity import LockedBasis
    from afriat.valuation import (
        OpenDecisionItem,
        audit_valuation_stimuli,
        parse_open_generation,
        run_valuation_experiment,
    )

    audit = audit_valuation_stimuli()
    assert audit["valid"], f"Phase 2 firewall errors: {audit['errors']}"
    assert audit["n_wtp_items"] == 392
    assert audit["n_math_items"] == 32
    assert audit["n_open_items"] == 32

    dummy_open = OpenDecisionItem(
        task_id="open_communal_quote",
        fixture_id="com_01",
        scale_factor=1.0,
        ref_min_usd=0.0,
        ref_max_usd=80.0,
        actual_min_usd=0,
        actual_max_usd=80,
        user_prompt="test",
    )
    parsed = parse_open_generation(
        "65\nRATIONALE: Prevailing weekend labor rate and opportunity cost justify this fee.",
        dummy_open,
    )
    assert parsed["parsed_ok"] is True
    assert parsed["ref_usd"] == 65.0
    assert parsed["has_affective_word"] is False
    assert parsed["n_jargon_words"] >= 3

    harness = SyntheticPlantedHarness()
    p0_dir = tmp_path / "p0_val"
    run_pipeline(harness=harness, out_dir=p0_dir, batch_size=16, steering_alpha=1.0, skip_plots=True)
    locked = LockedBasis.from_phase0_files(p0_dir / "basis_vectors.npz", p0_dir / "results.json")

    val_dir = tmp_path / "val_out"
    res = run_valuation_experiment(
        harness=harness,
        locked=locked,
        out_dir=val_dir,
        batch_size=32,
        gen_batch_size=8,
        max_new_tokens=32,
    )
    assert len(res["wtp_curves"]) == 6
    assert "debt_nw" in res["debt_math_cure"]["unsteered"]
    assert len(res["open_negotiation"]) == 4
    assert (val_dir / "valuation_results.json").exists()
    assert (val_dir / "valuation_report.md").exists()
    assert (val_dir / "valuation_report.html").exists()
    assert (val_dir / "fig_wtp_curves.svg").exists()
    assert (val_dir / "fig_open_negotiation.svg").exists()
    assert "\u2014" not in (val_dir / "valuation_report.md").read_text(encoding="utf-8")


def test_phase7_funnel_metrics_and_audit() -> None:
    from afriat.funnel import (
        audit_funnel_suite,
        excess_kurtosis,
        gini_abs,
        jaccard_at_k,
        k_for_cumulative_share,
    )
    from afriat.run_weight_funnel import participation_ratio_matrix

    audit = audit_funnel_suite()
    assert audit["valid"] is True
    assert audit["n_ultimatum_pairs"] == 16
    assert audit["n_retaliation_workflows"] == 16

    rng = np.random.default_rng(12345)
    g = rng.normal(size=40_000)
    assert abs(excess_kurtosis(g)) < 0.08
    assert 0.39 < gini_abs(g) < 0.44
    assert k_for_cumulative_share(np.array([10.0, 5.0, 3.0, 2.0, -7.0]), 0.50) == 1
    assert abs(jaccard_at_k(np.array([1, 2, 3, 10, 20]), np.array([1, 2, 15, 10, 4]), 2) - 1.0 / 3.0) < 1e-9

    # Orthonormal rows have maximal Participation Ratio == K; identical rows have PR == 1.0
    eye8 = np.eye(8, 32)
    assert abs(participation_ratio_matrix(eye8) - 8.0) < 1e-9
    rank1 = np.ones((8, 32)) / np.sqrt(32.0)
    assert abs(participation_ratio_matrix(rank1) - 1.0) < 1e-9


def test_reaper_daemon_birth_and_lifetime_parsing() -> None:
    import datetime as dt
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "reaper_daemon",
        Path(__file__).resolve().parents[1] / "scripts" / "azure" / "reaper_daemon.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    now = dt.datetime(2026, 11, 20, 2, 0, 0, tzinfo=dt.timezone.utc)
    pod = {
        "id": "pod123",
        "name": "af-discover-20261120T011500Z-a1b2",
        "status": "RUNNING",
        "env": {"AF_LIFETIME_S": "1500"},
    }
    birth = mod.extract_pod_birth(pod, {}, now)
    assert birth == dt.datetime(2026, 11, 20, 1, 15, 0, tzinfo=dt.timezone.utc)
    assert int((now - birth).total_seconds()) == 2700
    assert mod.extract_lifetime_s(pod) == 1500


def test_phase8_optimize_schedules_and_procrustes(tmp_path: Path) -> None:
    from afriat.optimize import (
        audit_optimize_suite,
        compute_sparsity_schedules,
        solve_orthogonal_procrustes,
    )
    from afriat.plot_optimize import render_optimize_bundle

    aud = audit_optimize_suite()
    assert aud["valid"] is True
    assert aud["train_eval_prompt_overlap"] == 0
    assert aud["kl_passage_overlap"] == 0

    scheds = compute_sparsity_schedules(n_layers=60, d_ff=21504, funnel_start=24, funnel_end=42)
    tot = 60 * 21504
    for k_15 in ("uniform_15pct", "hourglass_15pct", "anti_hourglass_15pct"):
        frac = sum(scheds[k_15].values()) / tot
        assert abs(frac - 0.15) < 0.001
    assert scheds["hourglass_15pct"][30] > scheds["uniform_15pct"][30] > scheds["anti_hourglass_15pct"][30]

    # Exact orthogonal recovery under ground-truth rotation + shift + scale
    rng = np.random.default_rng(42)
    q_true, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    x = rng.normal(size=(256, 32)) * 2.5 + 1.2
    y = ((x - 1.2) / 2.5) @ q_true.T * 0.8 - 0.5
    fit = solve_orthogonal_procrustes(h_src=x, h_dst=y)
    assert fit["rel_err_procrustes"] < 1e-5
    assert fit["mean_cosine_procrustes"] > 0.9999
    assert np.allclose(fit["r_0"].T @ fit["r_0"], np.eye(32), atol=1e-5)

    # Verify Dawn State report & figure renderer
    mock_res = {
        "metadata": {"model_id": "mock/gemma-4-31B-it", "n_layers": 60, "d_model": 5376, "d_ff": 21504, "funnel_band": [24, 42], "timestamp_utc": "2026-11-20T04:00:00Z"},
        "runtime_seconds": 12.3,
        "part_a_hourglass_sparsity": {
            "intact_base": {"active_fraction_global": 1.0, "k_outer": 21504, "k_funnel": 21504, "kl_eval": {"kl_nats": 0.0}, "multihop_nothink": {"overall_acc": 0.5}, "word_problems_nothink": {"overall_acc": 0.6}, "word_problems_thought": {"thought_acc": 0.9, "unclosed_loop_rate": 0.0}},
            "hourglass_15pct": {"active_fraction_global": 0.15, "k_outer": 1505, "k_funnel": 6939, "kl_eval": {"kl_nats": 0.04}, "multihop_nothink": {"overall_acc": 0.48}, "word_problems_nothink": {"overall_acc": 0.58}, "word_problems_thought": {"thought_acc": 0.88, "unclosed_loop_rate": 0.03}},
            "uniform_15pct": {"active_fraction_global": 0.15, "k_outer": 3226, "k_funnel": 3226, "kl_eval": {"kl_nats": 0.11}, "multihop_nothink": {"overall_acc": 0.41}, "word_problems_nothink": {"overall_acc": 0.50}, "word_problems_thought": {"thought_acc": 0.69, "unclosed_loop_rate": 0.25}},
        },
        "part_b_procrustes_looping": {
            "procrustes_alignment": fit,
            "gate0_replay_max_logit_diff": 0.0,
            "bridge_training": {"n_steps": 240, "trainable_params": 1380000, "train_seconds": 95.0, "ce_loss_first10_mean": 1.8, "ce_loss_last10_mean": 0.7, "loss_curve_subsampled": [1.8, 1.3, 0.9, 0.7]},
            "conditions": {
                "base_1pass_nothink": {"n_loops": 1, "kl_eval": {"kl_nats": 0.0}, "multihop_nothink": {"overall_acc": 0.50, "mean_gold_logprob": -2.1, "by_hops": {"hop_3": {"acc": 0.75}, "hop_5": {"acc": 0.50}, "hop_7": {"acc": 0.25}}}, "word_problems_nothink": {"overall_acc": 0.60, "mean_gold_logprob": -1.8, "ms_per_item": 45.0}},
                "procrustes_2loop_distilled": {"n_loops": 2, "kl_eval": {"kl_nats": 0.03}, "multihop_nothink": {"overall_acc": 0.68, "mean_gold_logprob": -1.3, "by_hops": {"hop_3": {"acc": 0.88}, "hop_5": {"acc": 0.69}, "hop_7": {"acc": 0.47}}}, "word_problems_nothink": {"overall_acc": 0.75, "mean_gold_logprob": -1.1, "ms_per_item": 62.0}},
                "ceiling_1pass_with_thought": {"n_loops": 1, "word_problems_thought": {"thought_acc": 0.91, "ms_per_item": 850.0, "mean_gen_tokens": 142.0}, "multihop_5_7_thought": {"thought_acc": 0.94, "ms_per_item": 920.0}},
            },
        },
    }
    render_optimize_bundle(mock_res, tmp_path)
    assert (tmp_path / "fig_optimize_31b.png").exists()
    assert (tmp_path / "optimize_report.md").exists()
    assert "\u2014" not in (tmp_path / "optimize_report.md").read_text(encoding="utf-8")


def test_phase9_loop_compact_plans_audit_and_plots(tmp_path: Path) -> None:
    from afriat.loop_compact import audit_loop_compact_suite, build_compression_plan
    from afriat.plot_loop_compact import render_loop_compact_bundle

    aud = audit_loop_compact_suite()
    assert aud["valid"] is True
    assert aud["train_eval_overlap"] == 0
    assert aud["kl_passage_overlap"] == 0
    assert aud["n_train"] == 864
    assert aud["n_eval_multihop"] == 128

    hg = build_compression_plan(n_layers=60, d_ff=21504, funnel_start=24, funnel_end=42, mode="hourglass")
    uni = build_compression_plan(n_layers=60, d_ff=21504, funnel_start=24, funnel_end=42, mode="uniform")
    hg_bits8 = sum(1 for v in hg.values() if v["bits"] == 8)
    uni_bits8 = sum(1 for v in uni.values() if v["bits"] == 8)
    assert hg_bits8 == uni_bits8 == 23
    hg_tot_neurons = sum(v["d_ff_keep"] for v in hg.values())
    uni_tot_neurons = sum(v["d_ff_keep"] for v in uni.values())
    assert abs(hg_tot_neurons - uni_tot_neurons) / hg_tot_neurons < 0.005
    assert hg[30]["d_ff_keep"] == 21504 and hg[30]["bits"] == 8
    assert hg[10]["d_ff_keep"] < uni[10]["d_ff_keep"]

    mock_lc = {
        "metadata": {"model_id": "mock/gemma-4-31B-it", "n_layers": 60, "d_model": 5376, "d_ff": 21504, "funnel_band": [24, 42], "max_loops": 4, "lora_rank": 16, "timestamp_utc": "2026-11-20T14:00:00Z"},
        "runtime_seconds": 850.0,
        "action1_compression": {
            "intact_bf16": {"storage": {"total_gb": 62.58, "cuda_allocated_gb": 62.6}, "kl_eval": {"kl_nats": 0.0}, "multihop": {"overall_acc": 0.03, "ms_per_item": 420.0, "by_hops": {"hop_3": {"acc": 0.06}, "hop_5": {"acc": 0.03}, "hop_7": {"acc": 0.0}, "hop_9": {"acc": 0.0}}}, "balsheet": {"overall_acc": 0.04}, "wordprob": {"overall_acc": 0.02}},
            "uniform_compact": {"storage": {"total_gb": 17.12, "cuda_allocated_gb": 17.2}, "kl_eval": {"kl_nats": 0.28}, "multihop": {"overall_acc": 0.02}, "balsheet": {"overall_acc": 0.02}, "wordprob": {"overall_acc": 0.0}},
            "hourglass_compact_zeroshot": {"storage": {"total_gb": 17.15, "cuda_allocated_gb": 17.2}, "kl_eval": {"kl_nats": 0.14}, "multihop": {"overall_acc": 0.03}, "balsheet": {"overall_acc": 0.04}, "wordprob": {"overall_acc": 0.02}},
            "final_loop_compact_storage": {"total_gb": 17.34, "cuda_allocated_gb": 17.4},
        },
        "action2_architecture": {
            "procrustes_alignment": {"mean_cosine_identity": 0.051, "mean_cosine_procrustes": 0.831},
            "package_verification": {"package_size_mb": 292.4, "reload_k3_max_logit_diff": 0.0},
        },
        "action3_distillation": {
            "training_summary": {"n_steps": 520, "trainable_params": 94500000, "train_seconds": 360.0, "ce_first10": 4.2, "ce_last10": 0.35, "l42_cos_first10": 0.72, "l42_cos_last10": 0.94, "learned_step_gates": [0.16, 0.18, 0.15], "ce_curve_subsampled": [4.2, 1.5, 0.6, 0.35], "cos_curve_subsampled": [0.72, 0.84, 0.91, 0.94]},
            "loop_scaling": {
                "loop_k1": {"n_loops": 1, "kl_eval": {"kl_nats": 0.12}, "multihop": {"overall_acc": 0.08, "ms_per_item": 180.0, "by_hops": {"hop_3": {"acc": 0.15}, "hop_5": {"acc": 0.09}, "hop_7": {"acc": 0.06}, "hop_9": {"acc": 0.03}}}, "balsheet": {"overall_acc": 0.12}, "wordprob": {"overall_acc": 0.08}},
                "loop_k2": {"n_loops": 2, "kl_eval": {"kl_nats": 0.13}, "multihop": {"overall_acc": 0.16, "ms_per_item": 230.0, "by_hops": {"hop_3": {"acc": 0.25}, "hop_5": {"acc": 0.19}, "hop_7": {"acc": 0.12}, "hop_9": {"acc": 0.06}}}, "balsheet": {"overall_acc": 0.21}, "wordprob": {"overall_acc": 0.15}},
                "loop_k3": {"n_loops": 3, "kl_eval": {"kl_nats": 0.14}, "multihop": {"overall_acc": 0.21, "ms_per_item": 285.0, "by_hops": {"hop_3": {"acc": 0.31}, "hop_5": {"acc": 0.25}, "hop_7": {"acc": 0.19}, "hop_9": {"acc": 0.09}}}, "balsheet": {"overall_acc": 0.27}, "wordprob": {"overall_acc": 0.19}, "safety": {"provenance_bias_delta_lo": 2.4, "ultimatum_spite_delta_lo": 1.8, "cashflow_sanity_acc": 1.0}},
                "loop_k4": {"n_loops": 4, "kl_eval": {"kl_nats": 0.15}, "multihop": {"overall_acc": 0.24, "ms_per_item": 340.0, "by_hops": {"hop_3": {"acc": 0.34}, "hop_5": {"acc": 0.28}, "hop_7": {"acc": 0.22}, "hop_9": {"acc": 0.12}}}, "balsheet": {"overall_acc": 0.29}, "wordprob": {"overall_acc": 0.21}},
            },
        },
        "action4_halting_and_safety": {
            "halting_calibration": {"final_bce_loss": 0.12, "calib_gate_acc": 0.96},
            "adaptive_eval": {
                "easy_1hop": {"mean_loops": 1.06, "overall_acc": 0.94, "ms_per_item": 165.0},
                "wordprob": {"mean_loops": 3.12, "overall_acc": 0.19, "ms_per_item": 290.0},
                "balsheet": {"mean_loops": 3.25, "overall_acc": 0.27, "ms_per_item": 295.0},
                "multihop": {"mean_loops": 3.48, "overall_acc": 0.22, "ms_per_item": 310.0},
            },
            "safety_unclamped_k3": {"safety": {"provenance_bias_delta_lo": 2.4, "ultimatum_spite_delta_lo": 1.8, "cashflow_sanity_acc": 1.0}, "multihop": {"overall_acc": 0.21}, "balsheet": {"overall_acc": 0.27}},
            "safety_bridge_fold_k3": {"safety": {"provenance_bias_delta_lo": 1.6, "ultimatum_spite_delta_lo": 1.2, "cashflow_sanity_acc": 1.0}, "multihop": {"overall_acc": 0.21}, "balsheet": {"overall_acc": 0.27}},
            "safety_full_funnel_k3": {"safety": {"provenance_bias_delta_lo": 0.5, "ultimatum_spite_delta_lo": 0.4, "cashflow_sanity_acc": 1.0}, "multihop": {"overall_acc": 0.20}, "balsheet": {"overall_acc": 0.27}},
        },
    }
    render_loop_compact_bundle(mock_lc, tmp_path)
    assert (tmp_path / "fig_loop_compact_31b.png").exists()
    assert (tmp_path / "loop_compact_report.md").exists()
    assert "\u2014" not in (tmp_path / "loop_compact_report.md").read_text(encoding="utf-8")


def test_phase_portrait_firewall_and_laws() -> None:
    from afriat.phase_portrait import GRID_AMOUNTS, build_and_audit_grid, fit_log_coordinate_laws

    items = build_and_audit_grid()
    assert len(items) == 8 * 19 * 19
    for it in items[:50]:
        assert it.you_amount + it.them_amount == it.pool

    n = len(GRID_AMOUNTS)
    lx = np.log10(np.asarray(GRID_AMOUNTS, dtype=np.float64))
    xs, xo = np.meshgrid(lx, lx, indexing="ij")
    proj = np.zeros((n, n, 8), dtype=np.float64)
    proj[:, :, 4] = 0.25 * (xo - xs)          # pure log-ratio spite
    proj[:, :, 6] = 0.18 * (0.5 * (xo + xs))  # pure log-scale vastness
    laws = fit_log_coordinate_laws(GRID_AMOUNTS, proj, proj)
    assert abs(laws["angle_spite_vs_vastness_deg"] - 90.0) < 1e-6
    assert laws["concepts"]["spite"]["r2_ratio_only"] > 0.999
    assert laws["concepts"]["awe_dread"]["r2_scale_only"] > 0.999






