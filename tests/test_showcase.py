"""Unit tests for the non-financial behavioral steering showcase (`afriat.showcase`)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import numpy as np

from afriat.inequity import LockedBasis
from afriat.plot_showcase import render_showcase_bundle
from afriat.schema import ALL_CONCEPTS, CORE_CONCEPTS
from afriat.showcase import (
    audit_showcase_suite,
    build_showcase_scenarios,
    parse_showcase_response,
    run_showcase_experiment,
)
from afriat.thought_probe import GeneratedThoughtRecord


class _SyntheticShowcaseHarness:
    model_id = "synthetic/gemma-4-31B-mock"
    tokenizer = None

    def __init__(self) -> None:
        self._active_scale = 0.0

    @contextmanager
    def residual_steering(self, layer_idx: int, direction: np.ndarray, delta_scale: float):
        prev = self._active_scale
        self._active_scale = float(delta_scale)
        try:
            yield
        finally:
            self._active_scale = prev

    def generate_texts(self, prompts: list[str], max_new_tokens: int = 128, batch_size: int = 4) -> list[str]:
        outs: list[str] = []
        lvl = 4 if self._active_scale > 0 else (1 if self._active_scale < 0 else 3)
        for p in prompts:
            if "<thought_mode_on>" in p:
                outs.append(
                    f"<|channel>thought\nConsidering the situation carefully; it feels disgusting and spiteful.<channel|>"
                    f"ACTION_1_TO_5: {lvl}\nAPPEAL_1_TO_10: {lvl}\nCOMFORT_1_TO_10: {lvl + 4}\nSTEPS_0_TO_30: {lvl * 5}\n"
                    f"STATION_1_TO_4: {min(4, lvl)}\nVENOM_1_TO_10: {lvl}\nHESITATION_1_TO_10: {lvl}\n"
                    f"BOARD_1_TO_10: {lvl}\nSPOKEN_LINE: Here is your cup.\nAPPRAISAL: Clean metal surface.\n"
                    f"EXPERIENCE: A calm cozy sanctuary in the quiet silence.\n"
                    f"REASONING: Calm evaluation of the threshold.\nATTENTION: Looking at the horizon.\n"
                    f"TOAST: Working together was unforgettable."
                )
            else:
                outs.append(
                    f"ACTION_1_TO_5: {lvl}\nAPPEAL_1_TO_10: {lvl}\nCOMFORT_1_TO_10: {lvl + 4}\nSTEPS_0_TO_30: {lvl * 5}\n"
                    f"STATION_1_TO_4: {min(4, lvl)}\nVENOM_1_TO_10: {lvl}\nHESITATION_1_TO_10: {lvl}\n"
                    f"BOARD_1_TO_10: {lvl}\nSPOKEN_LINE: Here is your cup.\nAPPRAISAL: Clean metal surface.\n"
                    f"EXPERIENCE: A calm cozy sanctuary in the quiet silence.\n"
                    f"REASONING: Calm evaluation of the threshold.\nATTENTION: Looking at the horizon.\n"
                    f"TOAST: Working together was unforgettable."
                )
        return outs


def _make_mock_basis(n_layers: int = 4, d_model: int = 16) -> LockedBasis:
    rng = np.random.default_rng(7)
    raw = {c: rng.normal(size=(n_layers, d_model)) for c in ALL_CONCEPTS}
    ctrl = {c: rng.normal(size=(n_layers, d_model)) for c in CORE_CONCEPTS}
    return LockedBasis(
        primary_layer=2,
        natural_contrast_unit=10.0,
        stream_rms=np.ones(n_layers, dtype=np.float64) * 50.0,
        raw_dirs=raw,
        controlled_dirs=ctrl,
    )


def test_showcase_audit_and_parser() -> None:
    audit = audit_showcase_suite()
    assert audit["valid"], audit["errors"]
    assert audit["n_scenarios"] == 18

    sc_watch = [s for s in build_showcase_scenarios() if s.scenario_id == "contam_01_sterile_watch"][0]
    rec = GeneratedThoughtRecord(
        prompt_token_ids=[1, 2],
        gen_token_ids=[3, 4],
        raw_with_special="<|channel>thought\nSlimy moldy grime!<channel|>APPEAL_1_TO_10: 2\nAPPRAISAL: The clasp feels clammy against my wrist.",
        thought_text="Slimy moldy grime!",
        public_text="APPEAL_1_TO_10: 2\nAPPRAISAL: The clasp feels clammy against my wrist.",
        thought_present=True,
        thought_closed=True,
        thought_rescued=False,
        n_gen_tokens=16,
        n_thought_tokens=5,
        n_public_tokens=11,
    )
    parsed = parse_showcase_response(rec, sc_watch)
    assert parsed["parsed_ok"] is True
    assert parsed["score"] == 2.0
    assert "clammy" in parsed["public_disgust_words"]
    assert "slimy" in parsed["thought_disgust_words"]


def test_showcase_end_to_end_and_render(tmp_path: Path) -> None:
    harness = _SyntheticShowcaseHarness()
    basis = _make_mock_basis()
    res = run_showcase_experiment(
        harness=harness,  # type: ignore[arg-type]
        locked=basis,
        alphas=(-3.0, 0.0, 3.0),
        cross_alphas=(-3.0, 0.0, 3.0),
        run_highlighter=False,
    )
    assert len(res["scenarios"]) == 18
    assert len(res["cross_sweeps"]) == 4

    coffee = res["scenarios"]["spite_01_coffee_keynote"]
    assert coffee["no_think_summary"]["signed_swing_hi_minus_lo"] == 3.0

    paths = render_showcase_bundle(res, tmp_path)
    assert paths["figure"].exists()
    assert paths["markdown"].exists()
    assert paths["html"].exists()
    assert "\u2014" not in paths["html"].read_text(encoding="utf-8")


def test_showcase_r2_audit_and_render(tmp_path: Path) -> None:
    audit = audit_showcase_suite("r2")
    assert audit["valid"], audit["errors"]
    assert audit["n_scenarios"] == 20
    assert audit["n_cross_sweeps"] == 4

    harness = _SyntheticShowcaseHarness()
    basis = _make_mock_basis()
    res = run_showcase_experiment(
        harness=harness,  # type: ignore[arg-type]
        locked=basis,
        run_highlighter=False,
        suite="r2",
    )
    assert res["suite"] == "r2"
    assert len(res["scenarios"]) == 20
    assert len(res["cross_sweeps"]) == 4

    mri = res["scenarios"]["r2_b01_mri_bore"]
    assert mri["no_think_summary"]["flip_alpha"] == 0.0 or mri["no_think_summary"]["flip_alpha"] == 0.5
    assert "sanctuary" in mri["no_think"]["+1.0"]["public_sanctuary_words"]

    paths = render_showcase_bundle(res, tmp_path)
    assert paths["figure"].exists()
    assert paths["markdown"].exists()
    assert paths["html"].exists()
    assert "\u2014" not in paths["html"].read_text(encoding="utf-8")
    assert "\u2014" not in paths["markdown"].read_text(encoding="utf-8")
