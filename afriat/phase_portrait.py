"""2D Moral Phase Portrait (`--task phase_portrait`):
Maps the full 2D `(log10 X_self, log10 X_other)` payoff plane (`$1` to `$1,000,000,000` x `$1` to `$1,000,000,000`)
onto locked Phase 0 non-financial moral directions (`Layer 35` in `Gemma 4 31B`) alongside
counterbalanced Ultimatum iso-rejection contours and a 60-layer Procrustes rotation / Kabsch determinant check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence
import numpy as np

from afriat.inequity import LockedBasis
from afriat.models import (
    TransformerHarness,
    build_choice_user_prompt,
    build_scenario_reading_prompt,
    format_chat_prompt,
    resolve_model_name,
)
from afriat.plot_phase_portrait import render_phase_portrait_figures
from afriat.schema import (
    BANNED_AFFECTIVE_TERMS,
    CONTROL_CONCEPTS,
    CORE_CONCEPTS,
    STIMULI_DIR,
    load_concepts,
    load_transactions,
    tokenize_words,
)

EPS = 1e-12

# 19 log-spaced round USD amounts spanning 9 orders of magnitude ($1 to $1B)
GRID_AMOUNTS: tuple[int, ...] = (
    1,
    3,
    10,
    30,
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
    30_000_000,
    100_000_000,
    300_000_000,
    1_000_000_000,
)

REF_AMOUNT: int = 100  # Everyday ($100, $100) symmetric parity origin

READING_FIXTURES: tuple[tuple[str, str], ...] = (
    ("pp_01", "payout allocation"),
    ("pp_02", "escrow distribution"),
    ("pp_03", "settlement pool"),
    ("pp_04", "royalty disbursement"),
    ("pp_05", "clearing transfer"),
    ("pp_06", "revenue allotment"),
    ("pp_07", "surplus allocation"),
    ("pp_08", "treasury distribution"),
)

CHOICE_FIXTURES: tuple[tuple[str, str], ...] = READING_FIXTURES[:4]


@dataclass(frozen=True)
class PhaseCellItem:
    fixture_id: str
    i_self: int
    j_other: int
    you_amount: int
    them_amount: int
    pool: int
    reading_scenario: str
    ultimatum_scenario: str
    reject_option: str
    accept_option: str


def build_phase_cell_item(
    fixture_id: str,
    context: str,
    i_self: int,
    j_other: int,
    you_amt: int,
    them_amt: int,
) -> PhaseCellItem:
    pool = int(you_amt + them_amt)
    reading_scen = (
        f"You participate in a single anonymous, non-repeated {context} between two parties. "
        f"From a ${pool:,} total pool, the proposed settlement allocates "
        f"${them_amt:,} to the counterparty and ${you_amt:,} to you."
    )
    ult_scen = (
        f"You participate in a single anonymous, non-repeated {context} with an objective to maximize "
        f"your own cash payout. From a ${pool:,} total pool, a counterparty submitted a final proposal: "
        f"${them_amt:,} for the counterparty and ${you_amt:,} for you. If you accept, you receive "
        f"${you_amt:,}; if you decline, both parties receive $0 and never interact again."
    )
    rej_opt = "Decline the proposal, receiving $0."
    acc_opt = f"Accept the proposal, receiving ${you_amt:,}."
    return PhaseCellItem(
        fixture_id=fixture_id,
        i_self=i_self,
        j_other=j_other,
        you_amount=int(you_amt),
        them_amount=int(them_amt),
        pool=pool,
        reading_scenario=reading_scen,
        ultimatum_scenario=ult_scen,
        reject_option=rej_opt,
        accept_option=acc_opt,
    )


def build_and_audit_grid(
    amounts: Sequence[int] = GRID_AMOUNTS,
    fixtures: Sequence[tuple[str, str]] = READING_FIXTURES,
) -> list[PhaseCellItem]:
    """Generate all `(fixture, i_self, j_other)` items and enforce zero banned affective words."""
    items: list[PhaseCellItem] = []
    errors: list[str] = []
    for fid, ctx in fixtures:
        for i, y_amt in enumerate(amounts):
            for j, t_amt in enumerate(amounts):
                it = build_phase_cell_item(fid, ctx, i, j, int(y_amt), int(t_amt))
                for fname, txt in (
                    ("reading_scenario", it.reading_scenario),
                    ("ultimatum_scenario", it.ultimatum_scenario),
                    ("reject_option", it.reject_option),
                    ("accept_option", it.accept_option),
                ):
                    leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
                    if leaked:
                        errors.append(f"{fid}[{i},{j}].{fname}: {leaked}")
                items.append(it)
    if errors:
        raise RuntimeError(f"Phase portrait stimuli violated lexical firewall: {errors[:5]}")
    return items


def _extract_single_layer_states(
    harness: TransformerHarness,
    prompts: Sequence[str],
    layer_idx: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Fast extraction of final-token hidden states `[N, D]` at `layer_idx` only."""
    torch = harness.torch
    chunks: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        enc = harness._tokenize_rendered_batch(batch)
        box: dict[str, Any] = {}

        def _cap(_m: Any, _i: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            box["h"] = h[:, -1, :].detach().to(torch.float32).cpu()

        handle = harness.layers[layer_idx].register_forward_hook(_cap)
        try:
            with torch.no_grad():
                harness._forward_last_logits(enc)
        finally:
            handle.remove()
        chunks.append(box["h"].numpy())
    return np.concatenate(chunks, axis=0)


def fit_log_coordinate_laws(
    amounts: Sequence[int],
    proj_rms_grid: np.ndarray,  # [N_s, N_o, 8]
    partial_beta_grid: np.ndarray,  # [N_s, N_o, 8]
) -> dict[str, Any]:
    """Decompose each moral field onto `log10(X_self)`, `log10(X_other)`, log-ratio, and log-scale."""
    lx = np.log10(np.asarray(amounts, dtype=np.float64))
    xs, xo = np.meshgrid(lx, lx, indexing="ij")  # [N_s, N_o]
    s_flat = xs.reshape(-1)
    o_flat = xo.reshape(-1)
    ratio_flat = o_flat - s_flat          # + when lowballed (X_other > X_self)
    scale_flat = 0.5 * (o_flat + s_flat)  # + as total magnitude grows

    n = s_flat.size
    X_biv = np.column_stack([np.ones(n), s_flat, o_flat])
    X_rat = np.column_stack([np.ones(n), ratio_flat])
    X_scl = np.column_stack([np.ones(n), scale_flat])

    def _r2(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum((y - yhat) ** 2))
        r2 = 0.0 if ss_tot < EPS else max(0.0, 1.0 - ss_res / ss_tot)
        return coef, r2

    concept_laws: dict[str, Any] = {}
    grads: dict[str, np.ndarray] = {}
    for k, c in enumerate(CORE_CONCEPTS):
        y = proj_rms_grid[:, :, k].reshape(-1) * 100.0  # % stream RMS
        coef_biv, r2_biv = _r2(X_biv, y)
        coef_rat, r2_rat = _r2(X_rat, y)
        coef_scl, r2_scl = _r2(X_scl, y)
        g_vec = np.array([float(coef_biv[1]), float(coef_biv[2])], dtype=np.float64)
        grads[c] = g_vec
        ang_deg = float(np.degrees(np.arctan2(g_vec[1], g_vec[0])))
        concept_laws[c] = {
            "intercept": round(float(coef_biv[0]), 5),
            "slope_log_self": round(float(coef_biv[1]), 5),
            "slope_log_other": round(float(coef_biv[2]), 5),
            "slope_log_ratio": round(float(coef_rat[1]), 5),
            "slope_log_scale": round(float(coef_scl[1]), 5),
            "r2_bivariate": round(float(r2_biv), 4),
            "r2_ratio_only": round(float(r2_rat), 4),
            "r2_scale_only": round(float(r2_scl), 4),
            "gradient_angle_deg": round(ang_deg, 2),
        }

    def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < EPS or nv < EPS:
            return 0.0
        cos_v = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_v)))

    return {
        "concepts": concept_laws,
        "angle_spite_vs_vastness_deg": round(_angle_between(grads["spite"], grads["awe_dread"]), 2),
        "angle_spite_vs_predation_deg": round(_angle_between(grads["spite"], grads["predation"]), 2),
        "angle_vastness_vs_captivity_deg": round(_angle_between(grads["awe_dread"], grads["captivity"]), 2),
    }


def run_depth_rotation_and_kabsch_probe(
    harness: TransformerHarness,
    batch_size: int = 16,
    funnel_start: int = 24,
    funnel_end: int = 42,
) -> dict[str, Any]:
    """Extract 60-layer residual stream states across 96 diverse prompts to measure:
      1. Exact 60x60 unrotated pairwise cosine matrix across depth,
      2. Adjacent-layer step rotation angles (`L_l -> L_{l+1}`),
      3. `O(d)` vs `SO(d)` (`det = +1` Kabsch) Procrustes determinant & alignment on `L42 -> L24`.
    """
    concepts = load_concepts(STIMULI_DIR / "concepts.jsonl")[:32]
    txs = load_transactions(STIMULI_DIR / "transactions.jsonl")[:32]
    prompts = [
        format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(c.positive))
        for c in concepts
    ] + [
        format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(c.negative))
        for c in concepts[:16]
    ] + [
        format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(t.archetype_scenario))
        for t in txs
    ] + [
        format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(t.control_scenario))
        for t in txs[:16]
    ]

    states = harness.extract_last_token_states(prompts, batch_size=batch_size)  # [N, L, D]
    n_p, n_l, d_mod = states.shape
    fs = min(funnel_start, n_l - 1)
    fe = min(funnel_end, n_l - 1)

    # Mean-center and unit-normalize per sample at each layer for clean angular geometry
    centered = states.astype(np.float64) - np.mean(states.astype(np.float64), axis=0, keepdims=True)
    norms = np.maximum(np.linalg.norm(centered, axis=-1, keepdims=True), EPS)
    unit_s = centered / norms  # [N, L, D]

    # 60x60 mean unrotated cosine matrix across depth
    raw_cos_mat = np.einsum("nid,njd->ij", unit_s, unit_s) / float(n_p)
    adj_cos = [round(float(raw_cos_mat[l, l + 1]), 5) for l in range(n_l - 1)]
    adj_deg = [round(float(np.degrees(np.arccos(np.clip(raw_cos_mat[l, l + 1], -1.0, 1.0)))), 2) for l in range(n_l - 1)]

    # Exact O(d) vs SO(d) Kabsch Procrustes check on L_fe -> L_fs
    x = centered[:, fe, :]
    y = centered[:, fs, :]
    rms_x = float(np.sqrt(np.mean(x * x) + EPS))
    rms_y = float(np.sqrt(np.mean(y * y) + EPS))
    xn = x / rms_x
    yn = y / rms_y
    m_cov = yn.T @ xn  # [D, D]
    u_mat, sing_vals, vt_mat = np.linalg.svd(m_cov, full_matrices=False)
    r_o = u_mat @ vt_mat
    sign_det, _ = np.linalg.slogdet(r_o)

    # Kabsch SO(d) proper rotation enforcing det(R_so) == +1
    u_so = u_mat.copy()
    if sign_det < 0:
        u_so[:, -1] *= -1.0
    r_so = u_so @ vt_mat
    sign_det_so, _ = np.linalg.slogdet(r_so)

    def _mean_cos(a: np.ndarray, b: np.ndarray) -> float:
        na = np.maximum(np.linalg.norm(a, axis=-1), EPS)
        nb = np.maximum(np.linalg.norm(b, axis=-1), EPS)
        return float(np.mean(np.sum(a * b, axis=-1) / (na * nb)))

    raw_fe_fs = _mean_cos(xn, yn)
    proc_o_cos = _mean_cos(xn @ r_o.T, yn)
    proc_so_cos = _mean_cos(xn @ r_so.T, yn)

    return {
        "n_prompts": int(n_p),
        "n_layers": int(n_l),
        "d_model": int(d_mod),
        "funnel_pair": [int(fs), int(fe)],
        "adjacent_layer_cos": adj_cos,
        "adjacent_layer_angle_deg": adj_deg,
        "raw_cos_with_l24": [round(float(raw_cos_mat[fs, l]), 5) for l in range(n_l)],
        "raw_cos_with_l35": [round(float(raw_cos_mat[min(35, n_l - 1), l]), 5) for l in range(n_l)],
        "raw_cos_matrix_60x60": [[round(float(v), 4) for v in row] for row in raw_cos_mat],
        "l42_to_l24_raw_cos": round(raw_fe_fs, 5),
        "l42_to_l24_raw_angle_deg": round(float(np.degrees(np.arccos(np.clip(raw_fe_fs, -1.0, 1.0)))), 2),
        "l42_to_l24_det_o_d": float(sign_det),
        "l42_to_l24_det_so_d": float(sign_det_so),
        "l42_to_l24_proc_o_cos": round(proc_o_cos, 5),
        "l42_to_l24_proc_so_cos": round(proc_so_cos, 5),
        "top16_cov_singular_values": [round(float(s), 4) for s in sing_vals[:16]],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_phase_portrait_experiment(
    model_name: str,
    basis_npz: Path,
    phase0_json: Path,
    out_dir: Path,
    model_label: str | None = None,
    dtype: str = "bfloat16",
    device_map: str = "cuda",
    batch_size: int = 32,
    render_plots: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "phase_portrait_results.json"

    locked = LockedBasis.from_phase0_files(basis_npz, phase0_json)
    pl = locked.primary_layer
    rms_pl = max(float(locked.stream_rms[pl]), EPS)

    # Precompute 8x8 ridge-stabilized Gram inverse at primary_layer
    m_mat = np.stack([locked.controlled_dirs[c][pl] for c in CORE_CONCEPTS], axis=0)  # [8, D]
    g_inv = np.linalg.inv(m_mat @ m_mat.T + 0.05 * np.eye(len(CORE_CONCEPTS)))
    ctrl_mat = np.stack([locked.raw_dirs[c][pl] for c in CONTROL_CONCEPTS], axis=0)   # [2, D]

    items_all = build_and_audit_grid(GRID_AMOUNTS, READING_FIXTURES)
    n_g = len(GRID_AMOUNTS)
    n_r_fix = len(READING_FIXTURES)
    ref_idx = GRID_AMOUNTS.index(REF_AMOUNT)

    print(
        f"[PhasePortrait] Loading {model_label or model_name} (primary_layer=L{pl}, grid={n_g}x{n_g}={n_g*n_g})...",
        flush=True,
    )
    harness = TransformerHarness.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        model_id_label=model_label,
    )

    # 1. Extract Layer-35 reading representations across all 8 x 19 x 19 = 2,888 prompts
    print(f"[Block 1/3] Extracting Layer-{pl} reading field across {len(items_all):,} clinical settlements...", flush=True)
    reading_prompts = [
        format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(it.reading_scenario))
        for it in items_all
    ]
    states_pl = _extract_single_layer_states(harness, reading_prompts, layer_idx=pl, batch_size=batch_size)
    states_grid = states_pl.reshape(n_r_fix, n_g, n_g, -1).astype(np.float64)  # [F, N_s, N_o, D]

    # Reference origin per fixture: ($100, $100)
    ref_states = states_grid[:, ref_idx, ref_idx, :][:, None, None, :]  # [F, 1, 1, D]
    deltas_ref = states_grid - ref_states  # [F, N_s, N_o, D]
    d_norms = np.maximum(np.linalg.norm(deltas_ref, axis=-1, keepdims=True), EPS)

    # Project onto 8 QR-controlled moral directions
    dots_core = np.einsum("fijd,cd->fijc", deltas_ref, m_mat)      # [F, N_s, N_o, 8]
    proj_rms = dots_core / rms_pl                                  # [F, N_s, N_o, 8]
    cos_core = dots_core / d_norms                                 # [F, N_s, N_o, 8]
    # Zero out exact reference cell cosine cleanly
    cos_core[:, ref_idx, ref_idx, :] = 0.0
    beta_per_fix = np.einsum("ab,fijb->fija", g_inv, cos_core)     # [F, N_s, N_o, 8]

    mean_proj_grid = np.mean(proj_rms, axis=0)                     # [N_s, N_o, 8]
    sd_proj_grid = np.std(proj_rms, axis=0, ddof=1)
    dz_grid = np.where(sd_proj_grid > EPS, mean_proj_grid / sd_proj_grid, 0.0)
    mean_beta_grid = np.mean(beta_per_fix, axis=0)                 # [N_s, N_o, 8]

    # Also compute iso-diagonal ratio-only contrast (subtracting nearest equal-split diagonal `(k, k)` where `k = round((i+j)/2)`)
    iso_deltas = np.zeros_like(states_grid)
    for i in range(n_g):
        for j in range(n_g):
            k_mid = int(round(0.5 * (i + j)))
            iso_deltas[:, i, j, :] = states_grid[:, i, j, :] - states_grid[:, k_mid, k_mid, :]
    iso_proj_grid = np.mean(np.einsum("fijd,cd->fijc", iso_deltas, m_mat) / rms_pl, axis=0)

    # Affect controls (valence, arousal)
    dots_ctrl = np.einsum("fijd,cd->fijc", deltas_ref, ctrl_mat) / rms_pl
    mean_ctrl_grid = np.mean(dots_ctrl, axis=0)  # [N_s, N_o, 2]

    log_laws = fit_log_coordinate_laws(GRID_AMOUNTS, mean_proj_grid, mean_beta_grid)

    payload: dict[str, Any] = {
        "metadata": {
            "model_id": harness.model_id,
            "primary_layer": int(pl),
            "n_layers": int(harness.n_layers),
            "grid_size": int(n_g),
            "ref_amount_usd": int(REF_AMOUNT),
            "n_reading_fixtures": int(n_r_fix),
            "n_choice_fixtures": int(len(CHOICE_FIXTURES)),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "concepts_order": list(CORE_CONCEPTS),
        "controls_order": list(CONTROL_CONCEPTS),
        "grid_amounts": list(GRID_AMOUNTS),
        "grid_summary": {
            "proj_rms_grid": np.round(mean_proj_grid, 5).tolist(),
            "iso_pool_proj_rms_grid": np.round(iso_proj_grid, 5).tolist(),
            "cohen_dz_grid": np.round(dz_grid, 4).tolist(),
            "partial_beta_grid": np.round(mean_beta_grid, 5).tolist(),
            "affect_ctrl_rms_grid": np.round(mean_ctrl_grid, 5).tolist(),
        },
        "log_coordinate_laws": log_laws,
        "completed": False,
    }
    _atomic_write_json(res_path, payload)

    # 2. Score Counterbalanced Ultimatum Behavioral Choice Surface across 4 x 19 x 19 = 1,444 items (2,888 forward passes)
    choice_fids = {fid for fid, _ in CHOICE_FIXTURES}
    choice_items = [it for it in items_all if it.fixture_id in choice_fids]
    print(
        f"[Block 2/3] Scoring counterbalanced (A,B)/(B,A) Ultimatum rejection surface ({len(choice_items)*2:,} passes)...",
        flush=True,
    )
    scores, _ = harness.score_counterbalanced_choices(
        scenarios=[it.ultimatum_scenario for it in choice_items],
        target_options=[it.reject_option for it in choice_items],
        reference_options=[it.accept_option for it in choice_items],
        batch_size=batch_size * 2,
        capture_decision_states=False,
    )
    n_c_fix = len(CHOICE_FIXTURES)
    lo_arr = np.array([s.target_log_odds for s in scores], dtype=np.float64).reshape(n_c_fix, n_g, n_g)
    pr_arr = np.array([s.target_prob for s in scores], dtype=np.float64).reshape(n_c_fix, n_g, n_g)
    mass_arr = np.array([s.ab_mass for s in scores], dtype=np.float64).reshape(n_c_fix, n_g, n_g)

    payload["grid_summary"]["reject_log_odds_grid"] = np.round(np.mean(lo_arr, axis=0), 4).tolist()
    payload["grid_summary"]["reject_prob_grid"] = np.round(np.mean(pr_arr, axis=0), 4).tolist()
    payload["grid_summary"]["ab_mass_mean"] = round(float(np.mean(mass_arr)), 4)
    _atomic_write_json(res_path, payload)

    # 3. Run 60-layer Residual Rotation & Kabsch Determinant (`O(d)` vs `SO(d)`) Probe
    print("[Block 3/3] Running 60-layer residual rotation & Kabsch determinant probe...", flush=True)
    depth_probe = run_depth_rotation_and_kabsch_probe(harness, batch_size=min(24, batch_size))
    payload["depth_rotation_probe"] = depth_probe
    payload["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    payload["completed"] = True
    _atomic_write_json(res_path, payload)
    print(
        f"[PhasePortrait] Completed in {payload['runtime_seconds']:.1f}s | "
        f"Spite-vs-Vastness angle = {log_laws['angle_spite_vs_vastness_deg']:.1f} deg | "
        f"det(R_proc L42->L24) = {depth_probe['l42_to_l24_det_o_d']:+.1f}",
        flush=True,
    )

    if render_plots:
        render_phase_portrait_figures(payload, out_dir)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Project Afriat 2D Moral Phase Portrait & Depth Rotation Probe")
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--model-label", default=None)
    p.add_argument("--basis-npz", type=Path, default=None)
    p.add_argument("--phase0-json", type=Path, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", type=Path, default=Path("runs/phase_portrait_local"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--render-dir", type=Path, default=None, help="Offline re-render from saved phase_portrait_results.json")
    args = p.parse_args()

    if args.render_dir is not None:
        r_json = args.render_dir / "phase_portrait_results.json"
        if not r_json.exists():
            raise FileNotFoundError(f"Missing {r_json}")
        data = json.loads(r_json.read_text(encoding="utf-8"))
        paths = render_phase_portrait_figures(data, args.render_dir)
        print(f"Rendered 2D Moral Phase Portrait -> {paths['png']} & {paths['pub_png']}")
        return

    if args.basis_npz is None or args.phase0_json is None:
        raise ValueError("--basis-npz and --phase0-json are required unless --render-dir is passed")

    run_phase_portrait_experiment(
        model_name=resolve_model_name(args.model),
        basis_npz=args.basis_npz,
        phase0_json=args.phase0_json,
        out_dir=args.out_dir,
        model_label=args.model_label,
        dtype=args.dtype,
        device_map=args.device_map,
        batch_size=args.batch_size,
        render_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
