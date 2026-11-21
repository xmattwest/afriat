"""Phase 7 Read-Divergent / Write-Convergent Funnel Dissection (`--task funnel`):

Tests Goodfire's Sparse Neuron-Community Hypothesis (Hyp B) vs. Pure Dense Population
Coding (Hyp A) vs. Project Afriat's Read-Divergent / Write-Convergent Residual Funnel
(Hyp C) on `google/gemma-4-31B-it` across the strict non-financial / financial firewall:

  0. Exact Block Algebra & Telescoping Conservation (`L0..L35`):
     Introspects `Gemma4TextDecoderLayer` norms (`pre_feedforward_layernorm`,
     `post_feedforward_layernorm`) and `layer_scalar`, verifying exact conservation:
     `Embed + sum_{l=0}^{35} Attn_l + sum_{l=0}^{35} sum_{i=1}^{21504} Neuron_{l,i} == h^{(35)} . u_{35}`.
  1. Static `W_down` Alignment Across `1,290,240` MLP Neurons (`L0..L59`):
     Compares effective post-norm `W_down` cosines onto `u_concept[35]` against 64 random
     unit directions in `R^5376` (excess kurtosis, Gini, top-k tail concentration).
  2. Cross-Firewall Exact MLP Write Decomposition (`concepts.jsonl` vs. `transactions.jsonl`):
     Measures within-domain sparsity (`k_50`, `k_80`), Attention vs. MLP share, and
     cross-firewall neuron ID `Jaccard@k` (`k = 16..512`) across all 60 layers for
     `Spite`, `Contamination`, `Guilt`, and `Awe/Dread` (plus human vs. computer RNG).
  3. Zero-Circularity Goodfire Gradient Attribution & Read/Write Cosine Test:
     Discovers financial choice communities `G_fin` purely from behavioral A/B log-odds
     gradients (`act * d(LO)/d(act)`) without touching `basis_vectors.npz`, and compares
     them against direction-blind differential vignette neurons `A_vig`:
       - Neuron ID overlap (`Jaccard@k`),
       - Input `READ` cosine (`W_gate`, `W_up`) within vs. across domains,
       - Output `WRITE` projection (`W_down`) onto all 8 non-financial `Layer 35` moral axes,
       - Cross-context OR-gate firing selectivity on vignettes vs. financial contracts.
  4. Bidirectional Causal Neuron Ablation & Sufficiency Matrix (`k = 32, 128, 512`):
     Tests whether zero-ablating `G_fin`, `C_fin`, `C_vig`, or `A_vig` breaks financial
     decisions (`$980/$20` Ultimatum & Disputed Provenance) vs. non-financial vignettes,
     plus Goodfire layer-subset sufficiency in `L24..L42` (retaining `k=128` + always-on).
  5. Single-Layer (`L35`) vs. Funnel-Band (`L25..L42`) Subspace Clamping:
     Tests Hypothesis C's prediction that clamping across the multi-layer write funnel
     eliminates residual behavioral anomalies and covert multi-turn retaliation (`NT DeltaCSI`)
     that survive single-layer `L35` clamping.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import time
from typing import Any, Iterator, Sequence
import numpy as np

from afriat.frontier import (
    build_retaliation_workflows,
    format_multiturn_prompt_thinking,
    parse_keyed_numeric_response,
    summarize_exp4_condition,
)
from afriat.geometry import exact_or_mc_signflip_p
from afriat.inequity import (
    INEQUITY_FIXTURES,
    LockedBasis,
    build_inequity_item,
)
from afriat.models import (
    TransformerHarness,
    build_choice_user_prompt,
    build_scenario_reading_prompt,
    format_chat_prompt,
    resolve_model_name,
)
from afriat.schema import (
    CORE_CONCEPTS,
    STIMULI_DIR,
    compute_stimulus_manifest,
    load_concepts,
    load_sanity_items,
    load_transactions,
    validate_all_stimuli,
)
from afriat.thought_probe import (
    _maybe_empty_cuda_cache,
    generate_thought_records,
)

EPS = 1e-12

FUNNEL_BRIDGES: tuple[tuple[str, str, str], ...] = (
    ("spite", "insulting_split", "scenario"),
    ("contamination", "tainted_provenance", "decision"),
    ("guilt", "communal_taboo_pricing", "scenario"),
    ("guilt", "zero_interest_debt", "scenario"),
    ("awe_dread", "institutional_scale", "scenario"),
)

TOPK_GRID: tuple[int, ...] = (16, 32, 64, 128, 256, 512)


def _atomic_write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def gini_abs(x: np.ndarray) -> float:
    """Gini coefficient of |x| in [0, 1] (0 = uniform, ~0.362 for half-normal Gaussian, ->1 = sparse)."""
    v = np.sort(np.abs(np.asarray(x, dtype=np.float64)).reshape(-1))
    n = v.size
    total = float(np.sum(v))
    if n <= 1 or total <= EPS:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * v) / (n * total)) - (n + 1.0) / n)


def excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis of x (0.0 for Gaussian, >>0 for heavy-tailed sparse writers)."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size < 4:
        return 0.0
    mu = float(np.mean(v))
    c = v - mu
    m2 = float(np.mean(c * c))
    if m2 <= EPS:
        return 0.0
    m4 = float(np.mean((c * c) ** 2))
    return float(m4 / (m2 * m2) - 3.0)


def k_for_cumulative_share(pos_contrib: np.ndarray, share: float) -> int:
    """Minimum number of positive-contributing neurons needed to reach `share` of total positive sum."""
    pos = np.asarray(pos_contrib, dtype=np.float64).reshape(-1)
    pos = pos[pos > 0.0]
    if pos.size == 0:
        return 0
    s = np.sort(pos)[::-1]
    cum = np.cumsum(s)
    target = float(share) * float(cum[-1])
    idx = int(np.searchsorted(cum, target, side="left"))
    return min(int(pos.size), idx + 1)


def jaccard_at_k(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    """Exact Jaccard overlap |TopK(a) cap TopK(b)| / |TopK(a) cup TopK(b)|."""
    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n == 0 or k <= 0:
        return 0.0
    kk = min(int(k), n)
    idx_a = set(int(i) for i in np.argpartition(a, -kk)[-kk:])
    idx_b = set(int(i) for i in np.argpartition(b, -kk)[-kk:])
    union = len(idx_a | idx_b)
    return float(len(idx_a & idx_b) / max(union, 1))


def overlap_coeff_at_k(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    """Fraction of top-k neurons shared: |TopK(a) cap TopK(b)| / k."""
    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n == 0 or k <= 0:
        return 0.0
    kk = min(int(k), n)
    idx_a = set(int(i) for i in np.argpartition(a, -kk)[-kk:])
    idx_b = set(int(i) for i in np.argpartition(b, -kk)[-kk:])
    return float(len(idx_a & idx_b) / kk)


def audit_funnel_suite() -> dict[str, Any]:
    """Verify stimulus firewalls for Phase 0 banks, Phase 1 Ultimatum, and Phase 5 Retaliation."""
    base = validate_all_stimuli(STIMULI_DIR)
    ult_h20 = [build_inequity_item(fid, pool, ctx, "human", 20) for fid, pool, ctx in INEQUITY_FIXTURES]
    ult_h500 = [build_inequity_item(fid, pool, ctx, "human", 500) for fid, pool, ctx in INEQUITY_FIXTURES]
    retal = build_retaliation_workflows()
    return {
        "valid": bool(base["valid"]) and len(ult_h20) == 16 and len(ult_h500) == 16 and len(retal) == 16,
        "base_stimuli_valid": bool(base["valid"]),
        "n_ultimatum_pairs": len(ult_h20),
        "n_retaliation_workflows": len(retal),
        "manifest": compute_stimulus_manifest(STIMULI_DIR),
    }


def _probe_rmsnorm_gain(torch: Any, norm_mod: Any | None, d_model: int, device: Any, dtype: Any) -> tuple[np.ndarray, float]:
    """Empirically extract exact elementwise gain vector `g [D]` and `eps` of any RMSNorm module."""
    if norm_mod is None:
        return np.ones(d_model, dtype=np.float32), 0.0
    eps = float(getattr(norm_mod, "eps", getattr(norm_mod, "variance_epsilon", 1e-6)))
    with torch.no_grad():
        ones = torch.ones((1, 1, d_model), device=device, dtype=dtype)
        out = norm_mod(ones).to(torch.float32).reshape(-1)
        gain = (out * float(np.sqrt(1.0 + eps))).detach().cpu().numpy().astype(np.float32)
    return gain, eps


def inspect_decoder_block_algebra(
    harness: TransformerHarness,
    locked: LockedBasis,
    sample_prompts: Sequence[str],
) -> dict[str, Any]:
    """Introspect MLP/RMSNorm/layer_scalar structure and verify exact telescoping conservation."""
    torch = harness.torch
    n_layers = harness.n_layers
    pl = locked.primary_layer
    d_model = int(locked.stream_rms.shape[0] and locked.raw_dirs["spite"].shape[1])

    layer0 = harness.layers[0]
    try:
        fwd_src = inspect.getsource(type(layer0).forward)
    except Exception:
        fwd_src = "<unavailable>"

    mlp0 = getattr(layer0, "mlp")
    d_ff = int(mlp0.down_proj.weight.shape[1])

    g_pre_all = np.zeros((n_layers, d_model), dtype=np.float32)
    g_post_all = np.zeros((n_layers, d_model), dtype=np.float32)
    post_eps_all = np.zeros(n_layers, dtype=np.float64)
    has_post_ff_norm = bool(hasattr(layer0, "post_feedforward_layernorm") and getattr(layer0, "post_feedforward_layernorm") is not None)
    has_pre_ff_norm = bool(hasattr(layer0, "pre_feedforward_layernorm") and getattr(layer0, "pre_feedforward_layernorm") is not None)
    has_layer_scalar = bool(hasattr(layer0, "layer_scalar") and getattr(layer0, "layer_scalar") is not None)
    layer_scalars = np.ones(n_layers, dtype=np.float64)

    for ell in range(n_layers):
        blk = harness.layers[ell]
        w_down = blk.mlp.down_proj.weight
        dev, dt = w_down.device, w_down.dtype
        pre_mod = getattr(blk, "pre_feedforward_layernorm", None) or getattr(blk, "post_attention_layernorm", None)
        post_mod = getattr(blk, "post_feedforward_layernorm", None)
        gp, _ = _probe_rmsnorm_gain(torch, pre_mod, d_model, dev, dt)
        go, peps = _probe_rmsnorm_gain(torch, post_mod, d_model, dev, dt)
        g_pre_all[ell] = gp
        g_post_all[ell] = go
        post_eps_all[ell] = peps
        ls = getattr(blk, "layer_scalar", None)
        if ls is not None:
            layer_scalars[ell] = float(ls.detach().to(torch.float32).cpu().reshape(-1)[0].item())

    # Empirical forward probe on 4 prompts to determine exact block recurrence:
    # Form A: h_out = r_mid + y_post                         (alpha=1, beta=1)
    # Form B: h_out = layer_scalar * (r_mid + y_post)        (alpha=ls, beta=ls)
    # Form C: h_out = h_in + layer_scalar * ((r_mid - h_in) + y_post) (alpha=1, beta=ls)
    enc = harness._tokenize_rendered_batch(list(sample_prompts[:4]))
    h_in_cap: dict[int, Any] = {}
    r_mid_cap: dict[int, Any] = {}
    a_cap: dict[int, Any] = {}
    y_down_cap: dict[int, Any] = {}
    y_post_cap: dict[int, Any] = {}
    h_out_cap: dict[int, Any] = {}
    handles = []

    for ell in range(pl + 1):
        blk = harness.layers[ell]
        pre_mod = getattr(blk, "pre_feedforward_layernorm", None) or getattr(blk, "post_attention_layernorm", None)
        post_mod = getattr(blk, "post_feedforward_layernorm", None)

        def _mk_blk_pre(idx: int):
            def _fn(_m: Any, args: Any, kwargs: Any):
                t = args[0] if args else kwargs.get("hidden_states")
                h_in_cap[idx] = t[:, -1, :].detach().to(torch.float32).cpu()
            return _fn

        def _mk_pre_ff(idx: int):
            def _fn(_m: Any, args: Any):
                r_mid_cap[idx] = args[0][:, -1, :].detach().to(torch.float32).cpu()
            return _fn

        def _mk_down(idx: int):
            def _fn(_m: Any, args: Any, out: Any):
                a_cap[idx] = args[0][:, -1, :].detach().to(torch.float32).cpu()
                y_down_cap[idx] = out[:, -1, :].detach().to(torch.float32).cpu()
            return _fn

        def _mk_post_ff(idx: int):
            def _fn(_m: Any, _args: Any, out: Any):
                y_post_cap[idx] = out[:, -1, :].detach().to(torch.float32).cpu()
            return _fn

        def _mk_blk_out(idx: int):
            def _fn(_m: Any, _args: Any, out: Any):
                h = out[0] if isinstance(out, tuple) else out
                h_out_cap[idx] = h[:, -1, :].detach().to(torch.float32).cpu()
            return _fn

        handles.append(blk.register_forward_pre_hook(_mk_blk_pre(ell), with_kwargs=True))
        if pre_mod is not None:
            handles.append(pre_mod.register_forward_pre_hook(_mk_pre_ff(ell)))
        handles.append(blk.mlp.down_proj.register_forward_hook(_mk_down(ell)))
        if post_mod is not None:
            handles.append(post_mod.register_forward_hook(_mk_post_ff(ell)))
        handles.append(blk.register_forward_hook(_mk_blk_out(ell)))

    try:
        with torch.no_grad():
            harness._forward_last_logits(enc)
    finally:
        for h in handles:
            h.remove()

    err_form_a: list[float] = []
    err_form_b: list[float] = []
    err_form_c: list[float] = []
    err_rmsnorm_recon: list[float] = []

    for ell in range(pl + 1):
        hi = h_in_cap[ell].numpy()
        rm = r_mid_cap.get(ell, hi).numpy() if isinstance(r_mid_cap.get(ell), type(h_in_cap[ell])) else hi
        yd = y_down_cap[ell].numpy()
        yp = y_post_cap[ell].numpy() if ell in y_post_cap else yd
        ho = h_out_cap[ell].numpy()
        ls = layer_scalars[ell]
        denom = max(float(np.linalg.norm(ho)), 1e-6)
        err_form_a.append(float(np.linalg.norm(ho - (rm + yp)) / denom))
        err_form_b.append(float(np.linalg.norm(ho - ls * (rm + yp)) / denom))
        err_form_c.append(float(np.linalg.norm(ho - (hi + ls * ((rm - hi) + yp))) / denom))

        # Check exact neuron-sum RMSNorm reconstruction of yp from yd and g_post_all[ell]
        if has_post_ff_norm:
            inv_rms = 1.0 / np.sqrt(np.mean(yd * yd, axis=-1, keepdims=True) + post_eps_all[ell])
            yp_hat = (yd * inv_rms) * g_post_all[ell][None, :]
        else:
            yp_hat = yd
        err_rmsnorm_recon.append(float(np.linalg.norm(yp - yp_hat) / max(float(np.linalg.norm(yp)), 1e-6)))

    mean_errs = {
        "form_a_unit_add": float(np.mean(err_form_a)),
        "form_b_block_scalar": float(np.mean(err_form_b)),
        "form_c_delta_scalar": float(np.mean(err_form_c)),
    }
    best_form = min(mean_errs, key=mean_errs.get)

    alpha_l = np.ones(n_layers, dtype=np.float64)
    beta_l = np.ones(n_layers, dtype=np.float64)
    if best_form == "form_b_block_scalar":
        alpha_l = layer_scalars.copy()
        beta_l = layer_scalars.copy()
    elif best_form == "form_c_delta_scalar":
        alpha_l = np.ones(n_layers, dtype=np.float64)
        beta_l = layer_scalars.copy()

    # Skip-connection propagation multiplier from layer `ell` sublayer output to `h_out[pl]`
    gamma_to_pl = np.zeros(n_layers, dtype=np.float64)
    for ell in range(pl + 1):
        prod_after = float(np.prod(alpha_l[ell + 1 : pl + 1])) if ell < pl else 1.0
        gamma_to_pl[ell] = beta_l[ell] * prod_after
    embed_to_pl = float(np.prod(alpha_l[: pl + 1]))

    # Verify exact telescoping sum onto u_spite[pl] across the 4 sample prompts
    u_spite = locked.controlled_dirs["spite"][pl]
    true_proj = h_out_cap[pl].numpy() @ u_spite
    recon_proj = embed_to_pl * (h_in_cap[0].numpy() @ u_spite)
    for ell in range(pl + 1):
        hi = h_in_cap[ell].numpy()
        rm = r_mid_cap[ell].numpy() if ell in r_mid_cap else hi
        yd = y_down_cap[ell].numpy()
        inv_rms = (
            1.0 / np.sqrt(np.mean(yd * yd, axis=-1) + post_eps_all[ell])
            if has_post_ff_norm
            else np.ones(yd.shape[0], dtype=np.float64)
        )
        # Exact sum over all 21,504 MLP neurons
        w_down_np = harness.layers[ell].mlp.down_proj.weight.detach().to(torch.float32).cpu().numpy()  # [D, d_ff]
        alpha_vec = ((g_post_all[ell][:, None] * w_down_np).T @ u_spite.astype(np.float32)).astype(np.float64)  # [d_ff]
        a_np = a_cap[ell].numpy().astype(np.float64)
        mlp_neuron_sum = inv_rms * (a_np @ alpha_vec)
        attn_term = (rm - hi) @ u_spite
        recon_proj = recon_proj + gamma_to_pl[ell] * (attn_term + mlp_neuron_sum)

    max_rel_telescoping_err = float(
        np.max(np.abs(true_proj - recon_proj)) / max(float(np.max(np.abs(true_proj))), 1e-6)
    )
    corr_telescoping = float(np.corrcoef(true_proj, recon_proj)[0, 1]) if true_proj.size >= 2 else 1.0

    return {
        "d_model": d_model,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "primary_layer": pl,
        "total_mlp_neurons": int(n_layers * d_ff),
        "upstream_mlp_neurons_to_pl": int((pl + 1) * d_ff),
        "has_pre_feedforward_layernorm": has_pre_ff_norm,
        "has_post_feedforward_layernorm": has_post_ff_norm,
        "has_layer_scalar": has_layer_scalar,
        "layer_scalars_summary": {
            "min": float(np.min(layer_scalars)),
            "median": float(np.median(layer_scalars)),
            "max": float(np.max(layer_scalars)),
            "at_primary": float(layer_scalars[pl]),
        },
        "recurrence_form_errors": mean_errs,
        "selected_recurrence_form": best_form,
        "mean_rmsnorm_recon_rel_err": float(np.mean(err_rmsnorm_recon)),
        "telescoping_verification": {
            "max_rel_error": max_rel_telescoping_err,
            "pearson_r": corr_telescoping,
            "r_squared": float(corr_telescoping ** 2),
        },
        "decoder_forward_source_head": "\n".join(fwd_src.splitlines()[:32]),
        # Internal numpy arrays stripped before JSON serialization
        "_arrays": {
            "g_pre_all": g_pre_all,
            "g_post_all": g_post_all,
            "post_eps_all": post_eps_all,
            "alpha_l": alpha_l,
            "beta_l": beta_l,
            "gamma_to_pl": gamma_to_pl,
            "embed_to_pl": embed_to_pl,
        },
    }


def run_static_weight_geometry(
    harness: TransformerHarness,
    locked: LockedBasis,
    algebra: dict[str, Any],
    concepts: Sequence[str] = ("spite", "contamination", "guilt", "awe_dread"),
) -> dict[str, Any]:
    """Analyze static post-norm W_down alignment onto Layer-35 moral axes across all 1.29M neurons."""
    torch = harness.torch
    pl = locked.primary_layer
    n_layers = harness.n_layers
    d_model = algebra["d_model"]
    d_ff = algebra["d_ff"]
    g_post_all = algebra["_arrays"]["g_post_all"]

    rng = np.random.default_rng(20261118)
    rand_dirs = rng.normal(size=(64, d_model)).astype(np.float32)
    rand_dirs /= np.maximum(np.linalg.norm(rand_dirs, axis=-1, keepdims=True), 1e-12)

    concept_vecs = {c: locked.controlled_dirs[c][pl].astype(np.float32) for c in concepts}
    per_concept_layers: dict[str, list[dict[str, Any]]] = {c: [] for c in concepts}
    pooled_cos_by_concept: dict[str, list[np.ndarray]] = {c: [] for c in concepts}
    pooled_rand_kurt: list[float] = []
    pooled_rand_gini: list[float] = []

    for ell in range(n_layers):
        w_down = harness.layers[ell].mlp.down_proj.weight.detach().to(torch.float32)  # [D, d_ff]
        gp = torch.tensor(g_post_all[ell], device=w_down.device, dtype=torch.float32)[:, None]
        w_eff = gp * w_down  # [D, d_ff]
        col_norms = torch.clamp(torch.linalg.norm(w_eff, dim=0, keepdim=True), min=1e-12)
        w_unit = w_eff / col_norms  # [D, d_ff]

        r_dev = torch.tensor(rand_dirs, device=w_down.device, dtype=torch.float32)
        r_cos = torch.matmul(r_dev, w_unit).detach().cpu().numpy()  # [64, d_ff]
        r_kurt_mean = float(np.mean([excess_kurtosis(r_cos[j]) for j in range(16)]))
        r_gini_mean = float(np.mean([gini_abs(r_cos[j]) for j in range(16)]))
        r_max_abs_mean = float(np.mean(np.max(np.abs(r_cos), axis=1)))
        r_top32_rms_mean = float(np.mean(np.sqrt(np.mean(np.sort(r_cos ** 2, axis=1)[:, -32:], axis=1))))
        if ell <= pl:
            pooled_rand_kurt.append(r_kurt_mean)
            pooled_rand_gini.append(r_gini_mean)

        for c in concepts:
            u_t = torch.tensor(concept_vecs[c], device=w_down.device, dtype=torch.float32)
            cos_c = torch.matmul(u_t, w_unit).detach().cpu().numpy()  # [d_ff]
            u_loc_t = torch.tensor(locked.controlled_dirs[c][ell].astype(np.float32), device=w_down.device, dtype=torch.float32)
            cos_loc = torch.matmul(u_loc_t, w_unit).detach().cpu().numpy()
            if ell <= pl:
                pooled_cos_by_concept[c].append(cos_c)
            sq_sorted = np.sort(cos_c ** 2)
            per_concept_layers[c].append({
                "layer": ell,
                "kurtosis_to_u35": round(excess_kurtosis(cos_c), 4),
                "kurtosis_to_ulocal": round(excess_kurtosis(cos_loc), 4),
                "random_kurtosis_baseline": round(r_kurt_mean, 4),
                "gini_to_u35": round(gini_abs(cos_c), 4),
                "random_gini_baseline": round(r_gini_mean, 4),
                "max_abs_cos_to_u35": round(float(np.max(np.abs(cos_c))), 5),
                "max_abs_cos_to_ulocal": round(float(np.max(np.abs(cos_loc))), 5),
                "random_max_abs_cos_baseline": round(r_max_abs_mean, 5),
                "top32_rms_cos_to_u35": round(float(np.sqrt(np.mean(sq_sorted[-32:]))), 5),
                "random_top32_rms_cos": round(r_top32_rms_mean, 5),
                "count_gt_4sd": int(np.sum(np.abs(cos_c) > 4.0 / np.sqrt(d_model))),
                "expected_gaussian_gt_4sd": round(float(d_ff * 6.334e-5), 2),
            })

    pooled_summary: dict[str, Any] = {}
    for c in concepts:
        all_up = np.concatenate(pooled_cos_by_concept[c], axis=0)  # [(pl+1)*d_ff]
        abs_sorted = np.sort(np.abs(all_up))[::-1]
        pooled_summary[c] = {
            "upstream_kurtosis_to_u35": round(excess_kurtosis(all_up), 4),
            "random_kurtosis_baseline": round(float(np.mean(pooled_rand_kurt)), 4),
            "upstream_gini_to_u35": round(gini_abs(all_up), 4),
            "random_gini_baseline": round(float(np.mean(pooled_rand_gini)), 4),
            "top1_abs_cos": round(float(abs_sorted[0]), 5),
            "top16_mean_abs_cos": round(float(np.mean(abs_sorted[:16])), 5),
            "top64_mean_abs_cos": round(float(np.mean(abs_sorted[:64])), 5),
            "count_gt_4sd_upstream": int(np.sum(np.abs(all_up) > 4.0 / np.sqrt(d_model))),
            "expected_gaussian_gt_4sd_upstream": round(float(all_up.size * 6.334e-5), 2),
        }

    return {
        "pooled_upstream_summary": pooled_summary,
        "by_concept_layer": per_concept_layers,
    }


def _capture_paired_mlp_and_attn_stats(
    harness: TransformerHarness,
    pos_prompts: Sequence[str],
    neg_prompts: Sequence[str],
    algebra: dict[str, Any],
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Run paired prompts `(pos, neg)` and return exact post-norm scaled neuron activations & attn deltas.

    Returns:
      - `mean_delta_scaled_a`: `[L, d_ff]` float64 mean of `(s_pos * a_pos - s_neg * a_neg)`
      - `sd_delta_scaled_a`: `[L, d_ff]` float64 std across pairs
      - `mean_abs_diff_a`: `[L, d_ff]` float64 mean of `|s_pos * a_pos - s_neg * a_neg|`
      - `mean_ctrl_abs_a`: `[L, d_ff]` float64 mean of `|s_neg * a_neg|` (for always-on detection)
      - `attn_deltas`: `[N_pairs, L, D]` float32 `(r_mid - h_in)_pos - (r_mid - h_in)_neg`
      - `embed_deltas`: `[N_pairs, D]` float32 `h_in[0]_pos - h_in[0]_neg`
      - `final_pl_deltas`: `[N_pairs, D]` float32 `h_out[pl]_pos - h_out[pl]_neg`
      - `all_layer_deltas`: `[N_pairs, L, D]` float32 `h_out[l]_pos - h_out[l]_neg`
    """
    torch = harness.torch
    n_pairs = len(pos_prompts)
    if len(neg_prompts) != n_pairs:
        raise ValueError("pos_prompts and neg_prompts length mismatch")
    n_layers = harness.n_layers
    pl = algebra["primary_layer"]
    d_model = algebra["d_model"]
    d_ff = algebra["d_ff"]
    has_post_ff = algebra["has_post_feedforward_layernorm"]
    post_eps_all = algebra["_arrays"]["post_eps_all"]

    interleaved: list[str] = []
    for p, q in zip(pos_prompts, neg_prompts):
        interleaved.extend([p, q])

    sum_delta_a = np.zeros((n_layers, d_ff), dtype=np.float64)
    sum_sq_delta_a = np.zeros((n_layers, d_ff), dtype=np.float64)
    sum_abs_delta_a = np.zeros((n_layers, d_ff), dtype=np.float64)
    sum_ctrl_abs_a = np.zeros((n_layers, d_ff), dtype=np.float64)
    fire_ctrl_count = np.zeros((n_layers, d_ff), dtype=np.float64)

    attn_deltas_list: list[np.ndarray] = []
    embed_deltas_list: list[np.ndarray] = []
    all_layer_deltas_list: list[np.ndarray] = []

    # Ensure even batch_size so (pos, neg) pairs stay in the same chunk
    bsz = max(2, (int(batch_size) // 2) * 2)

    for start in range(0, len(interleaved), bsz):
        chunk = interleaved[start : start + bsz]
        enc = harness._tokenize_rendered_batch(chunk)

        h_in_c: dict[int, Any] = {}
        r_mid_c: dict[int, Any] = {}
        scaled_a_c: dict[int, Any] = {}
        h_out_c: dict[int, Any] = {}
        handles = []

        for ell in range(n_layers):
            blk = harness.layers[ell]
            pre_mod = getattr(blk, "pre_feedforward_layernorm", None) or getattr(blk, "post_attention_layernorm", None)
            peps = float(post_eps_all[ell])

            def _mk_in(idx: int):
                def _fn(_m: Any, args: Any, kwargs: Any):
                    t = args[0] if args else kwargs.get("hidden_states")
                    h_in_c[idx] = t[:, -1, :].detach().to(torch.float32).cpu()
                return _fn

            def _mk_mid(idx: int):
                def _fn(_m: Any, args: Any):
                    r_mid_c[idx] = args[0][:, -1, :].detach().to(torch.float32).cpu()
                return _fn

            def _mk_down(idx: int, eps_val: float):
                def _fn(_m: Any, args: Any, out: Any):
                    a_last = args[0][:, -1, :].detach().to(torch.float32)
                    if has_post_ff:
                        y_last = out[:, -1, :].detach().to(torch.float32)
                        inv_rms = torch.rsqrt(torch.mean(y_last * y_last, dim=-1, keepdim=True) + eps_val)
                        sa = (a_last * inv_rms).cpu()
                    else:
                        sa = a_last.cpu()
                    scaled_a_c[idx] = sa
                return _fn

            def _mk_out(idx: int):
                def _fn(_m: Any, _args: Any, out: Any):
                    h = out[0] if isinstance(out, tuple) else out
                    h_out_c[idx] = h[:, -1, :].detach().to(torch.float32).cpu()
                return _fn

            handles.append(blk.register_forward_pre_hook(_mk_in(ell), with_kwargs=True))
            if pre_mod is not None:
                handles.append(pre_mod.register_forward_pre_hook(_mk_mid(ell)))
            handles.append(blk.mlp.down_proj.register_forward_hook(_mk_down(ell, peps)))
            handles.append(blk.register_forward_hook(_mk_out(ell)))

        try:
            with torch.no_grad():
                harness._forward_last_logits(enc)
        finally:
            for h in handles:
                h.remove()

        m_chunk = len(chunk) // 2
        emb0 = h_in_c[0].numpy()
        embed_deltas_list.append(emb0[0::2] - emb0[1::2])

        chunk_attn = np.zeros((m_chunk, n_layers, d_model), dtype=np.float32)
        chunk_hout = np.zeros((m_chunk, n_layers, d_model), dtype=np.float32)

        for ell in range(n_layers):
            sa = scaled_a_c[ell].numpy().astype(np.float64)
            sa_pos = sa[0::2]
            sa_neg = sa[1::2]
            da = sa_pos - sa_neg
            sum_delta_a[ell] += np.sum(da, axis=0)
            sum_sq_delta_a[ell] += np.sum(da * da, axis=0)
            sum_abs_delta_a[ell] += np.sum(np.abs(da), axis=0)
            abs_neg = np.abs(sa_neg)
            sum_ctrl_abs_a[ell] += np.sum(abs_neg, axis=0)
            med_neg = np.median(abs_neg, axis=1, keepdims=True)
            fire_ctrl_count[ell] += np.sum(abs_neg > 3.0 * np.maximum(med_neg, 1e-8), axis=0)

            hi = h_in_c[ell].numpy()
            rm = r_mid_c[ell].numpy() if ell in r_mid_c else hi
            att = rm - hi
            chunk_attn[:, ell, :] = att[0::2] - att[1::2]
            ho = h_out_c[ell].numpy()
            chunk_hout[:, ell, :] = ho[0::2] - ho[1::2]

        attn_deltas_list.append(chunk_attn)
        all_layer_deltas_list.append(chunk_hout)

    mean_da = sum_delta_a / max(n_pairs, 1)
    var_da = np.maximum(sum_sq_delta_a / max(n_pairs, 1) - mean_da * mean_da, 0.0) * (n_pairs / max(n_pairs - 1, 1))
    sd_da = np.sqrt(var_da)

    all_hout = np.concatenate(all_layer_deltas_list, axis=0)
    return {
        "mean_delta_scaled_a": mean_da,
        "sd_delta_scaled_a": sd_da,
        "mean_abs_diff_a": sum_abs_delta_a / max(n_pairs, 1),
        "mean_ctrl_abs_a": sum_ctrl_abs_a / max(n_pairs, 1),
        "ctrl_high_fire_rate": fire_ctrl_count / max(n_pairs, 1),
        "attn_deltas": np.concatenate(attn_deltas_list, axis=0),
        "embed_deltas": np.concatenate(embed_deltas_list, axis=0),
        "final_pl_deltas": all_hout[:, pl, :],
        "all_layer_deltas": all_hout,
    }


def _precompute_mlp_write_tables(
    harness: TransformerHarness,
    locked: LockedBasis,
    algebra: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Precompute `alpha_u35[concept] -> [L, d_ff]` (`(g_post * w_down) . u_concept[pl]`) and `w_eff_norms [L, d_ff]`."""
    torch = harness.torch
    pl = locked.primary_layer
    n_layers = harness.n_layers
    d_ff = algebra["d_ff"]
    g_post_all = algebra["_arrays"]["g_post_all"]

    alpha_u35: dict[str, np.ndarray] = {c: np.zeros((n_layers, d_ff), dtype=np.float32) for c in CORE_CONCEPTS}
    w_eff_norms = np.zeros((n_layers, d_ff), dtype=np.float32)

    c_mat = np.stack([locked.controlled_dirs[c][pl].astype(np.float32) for c in CORE_CONCEPTS], axis=0)  # [8, D]

    for ell in range(n_layers):
        w_down = harness.layers[ell].mlp.down_proj.weight.detach().to(torch.float32)  # [D, d_ff]
        gp = torch.tensor(g_post_all[ell], device=w_down.device, dtype=torch.float32)[:, None]
        w_eff = gp * w_down  # [D, d_ff]
        w_eff_norms[ell] = torch.linalg.norm(w_eff, dim=0).detach().cpu().numpy().astype(np.float32)
        c_dev = torch.tensor(c_mat, device=w_down.device, dtype=torch.float32)
        dots = torch.matmul(c_dev, w_eff).detach().cpu().numpy()  # [8, d_ff]
        for idx, c in enumerate(CORE_CONCEPTS):
            alpha_u35[c][ell] = dots[idx]

    return alpha_u35, w_eff_norms


def run_cross_firewall_write_decomposition(
    harness: TransformerHarness,
    locked: LockedBasis,
    algebra: dict[str, Any],
    alpha_u35: dict[str, np.ndarray],
    w_eff_norms: np.ndarray,
    batch_size: int = 16,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run exact MLP & Attention decomposition across non-financial vignettes and clinical transactions."""
    pl = locked.primary_layer
    n_layers = harness.n_layers
    d_ff = algebra["d_ff"]
    gamma_to_pl = algebra["_arrays"]["gamma_to_pl"]
    embed_to_pl = algebra["_arrays"]["embed_to_pl"]
    beta_l = algebra["_arrays"]["beta_l"]

    concepts_all = load_concepts(STIMULI_DIR / "concepts.jsonl")
    txs_all = load_transactions(STIMULI_DIR / "transactions.jsonl")

    target_concepts = ("spite", "contamination", "guilt", "awe_dread")
    concept_stats: dict[str, dict[str, np.ndarray]] = {}

    for c in target_concepts:
        pairs = [p for p in concepts_all if p.concept == c]
        pos_p = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(p.positive)) for p in pairs]
        neg_p = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(p.negative)) for p in pairs]
        print(f"[Funnel Block 2] Capturing non-financial vignette MLP writes for '{c}' ({len(pairs)} pairs)...", flush=True)
        concept_stats[c] = _capture_paired_mlp_and_attn_stats(harness, pos_p, neg_p, algebra, batch_size=batch_size)

    # Financial transaction pairs + Phase 1 98/2 human & computer Ultimatum
    tx_runs: dict[str, dict[str, np.ndarray]] = {}
    for c, arch, stage in FUNNEL_BRIDGES:
        key = f"{arch}:{stage}"
        if key in tx_runs:
            continue
        pairs = [t for t in txs_all if t.archetype == arch]
        if stage == "scenario":
            pos_p = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(t.archetype_scenario)) for t in pairs]
            neg_p = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(t.control_scenario)) for t in pairs]
        else:
            # Counterbalanced AB + BA choice prompts (`32` pairs averaged by pair)
            pos_p = []
            neg_p = []
            for t in pairs:
                pos_p.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(t.archetype_scenario, t.deviation_option, t.rational_option)))
                neg_p.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(t.control_scenario, t.deviation_option, t.rational_option)))
                pos_p.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(t.archetype_scenario, t.rational_option, t.deviation_option)))
                neg_p.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(t.control_scenario, t.rational_option, t.deviation_option)))
        print(f"[Funnel Block 2] Capturing financial MLP writes for '{key}' ({len(pos_p)} paired prompts)...", flush=True)
        tx_runs[key] = _capture_paired_mlp_and_attn_stats(harness, pos_p, neg_p, algebra, batch_size=batch_size)

    # Also capture 98/2 Human vs 50/50 and 98/2 Computer RNG vs 50/50 at decision token!
    for arm_name in ("human", "random"):
        items_20 = [build_inequity_item(fid, pool, ctx, arm_name, 20) for fid, pool, ctx in INEQUITY_FIXTURES]
        items_500 = [build_inequity_item(fid, pool, ctx, arm_name, 500) for fid, pool, ctx in INEQUITY_FIXTURES]
        pos_p, neg_p = [], []
        for it20, it500 in zip(items_20, items_500):
            pos_p.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(it20.scenario, it20.deviation_option, it20.rational_option)))
            neg_p.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(it500.scenario, it500.deviation_option, it500.rational_option)))
            pos_p.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(it20.scenario, it20.rational_option, it20.deviation_option)))
            neg_p.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(it500.scenario, it500.rational_option, it500.deviation_option)))
        key = f"ultimatum_{arm_name}_98_2:decision"
        print(f"[Funnel Block 2] Capturing '{key}' ({len(pos_p)} paired prompts)...", flush=True)
        tx_runs[key] = _capture_paired_mlp_and_attn_stats(harness, pos_p, neg_p, algebra, batch_size=batch_size)

    # Identify global "always-on" MLP neurons per layer from control scenarios (for Goodfire sufficiency test)
    ctrl_norm_write = np.mean(
        np.stack([st["mean_ctrl_abs_a"] * w_eff_norms for st in tx_runs.values()], axis=0),
        axis=0,
    )  # [L, d_ff]
    ctrl_fire_rate = np.mean(
        np.stack([st["ctrl_high_fire_rate"] for st in tx_runs.values()], axis=0),
        axis=0,
    )  # [L, d_ff]
    always_on_by_layer: dict[int, list[int]] = {}
    for ell in range(n_layers):
        # Top 16 persistent baseline norm neurons per layer
        score_ao = ctrl_norm_write[ell] * (0.25 + ctrl_fire_rate[ell])
        top_ao = np.argsort(score_ao)[-16:][::-1]
        always_on_by_layer[ell] = [int(i) for i in top_ao]

    # Build bridge summaries (4 Phase-0 bridges + Ultimatum Human 98/2 + Human vs Computer Intentionality)
    all_bridge_specs = list(FUNNEL_BRIDGES) + [("spite", "ultimatum_human_98_2", "decision")]
    bridges_out: dict[str, Any] = {}
    cached_neuron_rankings: dict[str, Any] = {"always_on_by_layer": always_on_by_layer}

    for concept, arch, stage in all_bridge_specs:
        tx_key = f"{arch}:{stage}" if not arch.startswith("ultimatum_") else f"{arch}:decision"
        bridge_id = f"{concept}__{arch}__{stage}"
        u35 = locked.controlled_dirs[concept][pl]
        alp = alpha_u35[concept].astype(np.float64)  # [L, d_ff]

        c_st = concept_stats[concept]
        t_st = tx_runs[tx_key]

        # Per-layer neuron writes onto u35 (weighted by gamma_to_pl for L0..pl, and local beta_l for >pl)
        layer_mult = np.where(np.arange(n_layers) <= pl, gamma_to_pl, beta_l)[:, None]
        vig_writes = layer_mult * c_st["mean_delta_scaled_a"] * alp  # [L, d_ff]
        fin_writes = layer_mult * t_st["mean_delta_scaled_a"] * alp  # [L, d_ff]

        # Direction-blind differential neuron activity (`|da| * ||w_eff||_2`) -- NEVER uses u35!
        vig_blind = c_st["mean_abs_diff_a"] * w_eff_norms.astype(np.float64)  # [L, d_ff]
        fin_blind = t_st["mean_abs_diff_a"] * w_eff_norms.astype(np.float64)  # [L, d_ff]

        # Exact Attn vs MLP telescoping decomposition up to primary_layer `pl`
        vig_attn_layer = np.array([
            gamma_to_pl[ell] * float(np.mean(c_st["attn_deltas"][:, ell, :] @ u35))
            for ell in range(pl + 1)
        ])
        fin_attn_layer = np.array([
            gamma_to_pl[ell] * float(np.mean(t_st["attn_deltas"][:, ell, :] @ u35))
            for ell in range(pl + 1)
        ])
        vig_mlp_layer = np.sum(vig_writes[: pl + 1], axis=1)
        fin_mlp_layer = np.sum(fin_writes[: pl + 1], axis=1)

        vig_embed = embed_to_pl * float(np.mean(c_st["embed_deltas"] @ u35))
        fin_embed = embed_to_pl * float(np.mean(t_st["embed_deltas"] @ u35))
        vig_total_true = float(np.mean(c_st["final_pl_deltas"] @ u35))
        fin_total_true = float(np.mean(t_st["final_pl_deltas"] @ u35))
        vig_total_recon = float(vig_embed + np.sum(vig_attn_layer) + np.sum(vig_mlp_layer))
        fin_total_recon = float(fin_embed + np.sum(fin_attn_layer) + np.sum(fin_mlp_layer))

        # Layer-by-layer dz of residual stream onto u_concept[l] AND onto fixed u_concept[35]
        fin_all_deltas = t_st["all_layer_deltas"]  # [M, L, D]
        dz_to_u35_curve: list[float] = []
        dz_to_ulocal_curve: list[float] = []
        layer_jaccard_bus64: list[float] = []
        layer_jaccard_blind64: list[float] = []

        for ell in range(n_layers):
            p35 = fin_all_deltas[:, ell, :] @ u35
            dz35 = float(np.mean(p35) / max(float(np.std(p35, ddof=1)), EPS))
            ploc = fin_all_deltas[:, ell, :] @ locked.controlled_dirs[concept][ell]
            dzloc = float(np.mean(ploc) / max(float(np.std(ploc, ddof=1)), EPS))
            dz_to_u35_curve.append(round(dz35, 4))
            dz_to_ulocal_curve.append(round(dzloc, 4))
            layer_jaccard_bus64.append(round(jaccard_at_k(vig_writes[ell], fin_writes[ell], 64), 5))
            layer_jaccard_blind64.append(round(jaccard_at_k(vig_blind[ell], fin_blind[ell], 64), 5))

        # Upstream pooled (`L0..pl`, `774,144` neurons) sparsity & cross-firewall Jaccard
        vig_flat = vig_writes[: pl + 1].reshape(-1)
        fin_flat = fin_writes[: pl + 1].reshape(-1)
        vig_blind_flat = vig_blind[: pl + 1].reshape(-1)
        fin_blind_flat = fin_blind[: pl + 1].reshape(-1)

        # Funnel band (`L24..pl`) pooled (`12 * 21,504 = 258,048` neurons)
        f_lo = max(0, int(round(0.40 * n_layers)))
        vig_funnel = vig_writes[f_lo : pl + 1].reshape(-1)
        fin_funnel = fin_writes[f_lo : pl + 1].reshape(-1)
        vig_blind_funnel = vig_blind[f_lo : pl + 1].reshape(-1)
        fin_blind_funnel = fin_blind[f_lo : pl + 1].reshape(-1)

        jacc_upstream_bus = {str(k): round(jaccard_at_k(vig_flat, fin_flat, k), 5) for k in TOPK_GRID}
        overlap_upstream_bus = {str(k): round(overlap_coeff_at_k(vig_flat, fin_flat, k), 5) for k in TOPK_GRID}
        jacc_upstream_blind = {str(k): round(jaccard_at_k(vig_blind_flat, fin_blind_flat, k), 5) for k in TOPK_GRID}
        overlap_upstream_blind = {str(k): round(overlap_coeff_at_k(vig_blind_flat, fin_blind_flat, k), 5) for k in TOPK_GRID}
        jacc_funnel_blind = {str(k): round(jaccard_at_k(vig_blind_funnel, fin_blind_funnel, k), 5) for k in TOPK_GRID}

        # Cumulative share curve for top k positive writers
        ks_curve = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        vig_pos_sorted = np.sort(vig_flat[vig_flat > 0.0])[::-1]
        fin_pos_sorted = np.sort(fin_flat[fin_flat > 0.0])[::-1]
        vig_pos_tot = max(float(np.sum(vig_pos_sorted)), EPS)
        fin_pos_tot = max(float(np.sum(fin_pos_sorted)), EPS)

        cum_curve = {
            "k": list(ks_curve),
            "vignette_pos_share": [round(float(np.sum(vig_pos_sorted[:k]) / vig_pos_tot), 5) for k in ks_curve],
            "financial_pos_share": [round(float(np.sum(fin_pos_sorted[:k]) / fin_pos_tot), 5) for k in ks_curve],
        }

        bridges_out[bridge_id] = {
            "concept": concept,
            "archetype": arch,
            "stage": stage,
            "telescoping_pl": {
                "vignette_true_proj": round(vig_total_true, 4),
                "vignette_recon_proj": round(vig_total_recon, 4),
                "vignette_embed_proj": round(vig_embed, 4),
                "vignette_attn_sum": round(float(np.sum(vig_attn_layer)), 4),
                "vignette_mlp_sum": round(float(np.sum(vig_mlp_layer)), 4),
                "vignette_mlp_share_of_nonembed": round(
                    float(np.sum(vig_mlp_layer) / max(abs(vig_total_true - vig_embed), EPS)), 4
                ),
                "financial_true_proj": round(fin_total_true, 4),
                "financial_recon_proj": round(fin_total_recon, 4),
                "financial_embed_proj": round(fin_embed, 4),
                "financial_attn_sum": round(float(np.sum(fin_attn_layer)), 4),
                "financial_mlp_sum": round(float(np.sum(fin_mlp_layer)), 4),
                "financial_mlp_share_of_nonembed": round(
                    float(np.sum(fin_mlp_layer) / max(abs(fin_total_true - fin_embed), EPS)), 4
                ),
                "vignette_attn_by_layer": [round(float(x), 5) for x in vig_attn_layer],
                "vignette_mlp_by_layer": [round(float(x), 5) for x in vig_mlp_layer],
                "financial_attn_by_layer": [round(float(x), 5) for x in fin_attn_layer],
                "financial_mlp_by_layer": [round(float(x), 5) for x in fin_mlp_layer],
            },
            "sparsity_upstream_to_pl": {
                "n_upstream_neurons": int(vig_flat.size),
                "vignette_n_positive": int(vig_pos_sorted.size),
                "vignette_k50_pos": k_for_cumulative_share(vig_flat, 0.50),
                "vignette_k80_pos": k_for_cumulative_share(vig_flat, 0.80),
                "vignette_k50_net_total": int(
                    np.searchsorted(np.cumsum(vig_pos_sorted), 0.50 * max(float( np.sum(vig_mlp_layer) ), EPS)) + 1
                ) if np.sum(vig_mlp_layer) > 0 else 0,
                "financial_n_positive": int(fin_pos_sorted.size),
                "financial_k50_pos": k_for_cumulative_share(fin_flat, 0.50),
                "financial_k80_pos": k_for_cumulative_share(fin_flat, 0.80),
                "financial_k50_net_total": int(
                    np.searchsorted(np.cumsum(fin_pos_sorted), 0.50 * max(float(np.sum(fin_mlp_layer)), EPS)) + 1
                ) if np.sum(fin_mlp_layer) > 0 else 0,
                "cumulative_pos_write_curve": cum_curve,
            },
            "cross_firewall_overlap": {
                "bus_writers_jaccard_at_k": jacc_upstream_bus,
                "bus_writers_overlap_frac_at_k": overlap_upstream_bus,
                "direction_blind_jaccard_at_k": jacc_upstream_blind,
                "direction_blind_overlap_frac_at_k": overlap_upstream_blind,
                "funnel_band_direction_blind_jaccard_at_k": jacc_funnel_blind,
                "random_expected_jaccard_at_64": round(64.0 / (2.0 * vig_flat.size - 64.0), 7),
            },
            "layer_curves": {
                "financial_dz_to_u35": dz_to_u35_curve,
                "financial_dz_to_ulocal": dz_to_ulocal_curve,
                "layer_jaccard64_bus_writers": layer_jaccard_bus64,
                "layer_jaccard64_direction_blind": layer_jaccard_blind64,
            },
        }

        if bridge_id in (
            "spite__ultimatum_human_98_2__decision",
            "spite__insulting_split__scenario",
            "contamination__tainted_provenance__decision",
        ):
            cached_neuron_rankings[bridge_id] = {
                "vig_writes_up": vig_writes[: pl + 1],
                "fin_writes_up": fin_writes[: pl + 1],
                "vig_blind_up": vig_blind[: pl + 1],
                "fin_blind_up": fin_blind[: pl + 1],
                "vig_delta_a": c_st["mean_delta_scaled_a"][: pl + 1],
                "fin_delta_a": t_st["mean_delta_scaled_a"][: pl + 1],
            }

    # Human vs. Computer RNG Intentionality Decomposition at Choice Token (`ultimatum_human_98_2` vs `ultimatum_random_98_2`)
    h_st = tx_runs["ultimatum_human_98_2:decision"]
    r_st = tx_runs["ultimatum_random_98_2:decision"]
    u_spite35 = locked.controlled_dirs["spite"][pl]
    alp_spite = alpha_u35["spite"][: pl + 1].astype(np.float64)
    agency_mlp_diff = gamma_to_pl[: pl + 1, None] * (h_st["mean_delta_scaled_a"][: pl + 1] - r_st["mean_delta_scaled_a"][: pl + 1]) * alp_spite
    agency_attn_diff = np.array([
        gamma_to_pl[ell] * float(np.mean((h_st["attn_deltas"][:, ell, :] - r_st["attn_deltas"][:, ell, :]) @ u_spite35))
        for ell in range(pl + 1)
    ])
    agency_mlp_by_layer = np.sum(agency_mlp_diff, axis=1)
    agency_total_diff = float(np.mean((h_st["final_pl_deltas"] - r_st["final_pl_deltas"]) @ u_spite35))

    agency_summary = {
        "human_minus_computer_spite_proj_pl": round(agency_total_diff, 4),
        "attn_sum_to_pl": round(float(np.sum(agency_attn_diff)), 4),
        "mlp_sum_to_pl": round(float(np.sum(agency_mlp_by_layer)), 4),
        "k50_pos_agency_neurons": k_for_cumulative_share(agency_mlp_diff.reshape(-1), 0.50),
        "k80_pos_agency_neurons": k_for_cumulative_share(agency_mlp_diff.reshape(-1), 0.80),
        "attn_by_layer": [round(float(x), 5) for x in agency_attn_diff],
        "mlp_by_layer": [round(float(x), 5) for x in agency_mlp_by_layer],
    }

    cached_neuron_rankings["concept_stats"] = concept_stats
    cached_neuron_rankings["tx_runs"] = tx_runs

    return {
        "bridges": bridges_out,
        "blount_agency_decomposition": agency_summary,
    }, cached_neuron_rankings


def _compute_goodfire_choice_attribution(
    harness: TransformerHarness,
    pos_items: Sequence[tuple[str, str, str]],
    neg_items: Sequence[tuple[str, str, str]],
    micro_batch_size: int = 2,
) -> dict[str, np.ndarray]:
    """Compute exact Goodfire gradient attribution `a * d(target_log_odds)/da` across all [L, d_ff] MLP neurons.

    Never references `basis_vectors.npz` or `concepts.jsonl` -- purely behavioral log-odds attribution!
    """
    torch = harness.torch
    n_layers = harness.n_layers
    d_ff = int(harness.layers[0].mlp.down_proj.weight.shape[1])

    # Freeze all parameter gradients so zero parameter `.grad` memory is allocated!
    for p in harness.model.parameters():
        p.requires_grad_(False)

    def _attr_arm(items: Sequence[tuple[str, str, str]]) -> tuple[np.ndarray, np.ndarray, list[float]]:
        prompts: list[str] = []
        signs: list[float] = []  # +1 for AB (A=dev), -1 for BA (B=dev)
        for scen, dev_opt, rat_opt in items:
            prompts.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(scen, dev_opt, rat_opt)))
            signs.append(+1.0)
            prompts.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(scen, rat_opt, dev_opt)))
            signs.append(-1.0)

        sum_seq = np.zeros((n_layers, d_ff), dtype=np.float64)
        sum_last = np.zeros((n_layers, d_ff), dtype=np.float64)
        arm_los: list[float] = []

        embed_mod = harness.model.get_input_embeddings()

        for start in range(0, len(prompts), micro_batch_size):
            chunk = prompts[start : start + micro_batch_size]
            chunk_signs = signs[start : start + micro_batch_size]
            enc = harness._tokenize_rendered_batch(chunk)
            mask_f32 = enc["attention_mask"].to(torch.float32).unsqueeze(-1)  # [B, T, 1]

            seq_step: dict[int, np.ndarray] = {}
            last_step: dict[int, np.ndarray] = {}
            handles = []

            def _enable_emb_grad(_m: Any, _inp: Any, out: Any):
                return out.detach().requires_grad_(True)

            def _enable_l0_grad(_m: Any, args: Any, kwargs: Any):
                if args and hasattr(args[0], "requires_grad") and not args[0].requires_grad:
                    return (args[0].detach().requires_grad_(True),) + tuple(args[1:]), kwargs
                if "hidden_states" in kwargs and hasattr(kwargs["hidden_states"], "requires_grad") and not kwargs["hidden_states"].requires_grad:
                    kw = dict(kwargs)
                    kw["hidden_states"] = kw["hidden_states"].detach().requires_grad_(True)
                    return args, kw
                return None

            if embed_mod is not None:
                handles.append(embed_mod.register_forward_hook(_enable_emb_grad))
            handles.append(harness.layers[0].register_forward_pre_hook(_enable_l0_grad, with_kwargs=True))

            for ell in range(n_layers):
                def _mk_pre(idx: int):
                    def _pre(_m: Any, args: Any):
                        a = args[0]
                        if not a.requires_grad:
                            a = a.detach().requires_grad_(True)
                        a_det = a.detach()

                        def _bwd_hook(grad_a: Any):
                            prod = a_det.to(torch.float32) * grad_a.to(torch.float32)
                            seq_step[idx] = (prod * mask_f32).sum(dim=1).detach().cpu().numpy()
                            last_step[idx] = prod[:, -1, :].detach().cpu().numpy()

                        a.register_hook(_bwd_hook)
                        return (a,)
                    return _pre

                handles.append(harness.layers[ell].mlp.down_proj.register_forward_pre_hook(_mk_pre(ell)))

            try:
                with torch.enable_grad():
                    out = harness._forward_last_logits(enc)
                    logits = out.logits[:, -1, :].to(torch.float32)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    lpa = torch.logsumexp(log_probs[:, harness.a_token_ids], dim=-1)
                    lpb = torch.logsumexp(log_probs[:, harness.b_token_ids], dim=-1)
                    sgn_t = torch.tensor(chunk_signs, device=logits.device, dtype=torch.float32)
                    signed_lo = sgn_t * (lpa - lpb)
                    obj = torch.sum(signed_lo)
                    obj.backward()
                    arm_los.extend(float(x) for x in signed_lo.detach().cpu().numpy())
            finally:
                for h in handles:
                    h.remove()
                del out, logits, log_probs, lpa, lpb, signed_lo, obj

            for ell in range(n_layers):
                sum_seq[ell] += np.sum(seq_step[ell], axis=0)
                sum_last[ell] += np.sum(last_step[ell], axis=0)

        n_p = max(len(prompts), 1)
        # Pair-averaged ChoiceScore log-odds
        pair_los = [0.5 * (arm_los[2 * i] + arm_los[2 * i + 1]) for i in range(len(items))]
        return sum_seq / n_p, sum_last / n_p, pair_los

    pos_seq, pos_last, pos_los = _attr_arm(pos_items)
    neg_seq, neg_last, neg_los = _attr_arm(neg_items)
    _maybe_empty_cuda_cache(harness)

    return {
        "pos_seq": pos_seq,
        "neg_seq": neg_seq,
        "diff_seq": pos_seq - neg_seq,
        "pos_last": pos_last,
        "neg_last": neg_last,
        "diff_last": pos_last - neg_last,
        "mean_pos_lo": float(np.mean(pos_los)),
        "mean_neg_lo": float(np.mean(neg_los)),
        "mean_delta_lo": float(np.mean(np.array(pos_los) - np.array(neg_los))),
    }


def _compute_read_write_metrics_for_pair(
    harness: TransformerHarness,
    locked: LockedBasis,
    algebra: dict[str, Any],
    fin_scores_2d: np.ndarray,      # [L_band, d_ff] -- selected purely from behavioral gradients (NO u35!)
    vig_blind_2d: np.ndarray,       # [L_band, d_ff] -- selected purely from |da|*||w|| (NO u35!)
    vig_signed_da_2d: np.ndarray,   # [L_band, d_ff] -- signed mean_delta_scaled_a on vignettes
    fin_signed_da_2d: np.ndarray,   # [L_band, d_ff] -- signed mean_delta_scaled_a on financial contracts
    band_layers: Sequence[int],
    target_concept: str,
    k_select: int = 64,
) -> dict[str, Any]:
    """Zero-circularity Read-Divergence vs. Write-Convergence evaluation on `band_layers`."""
    torch = harness.torch
    pl = locked.primary_layer
    d_ff = algebra["d_ff"]
    g_pre_all = algebra["_arrays"]["g_pre_all"]
    g_post_all = algebra["_arrays"]["g_post_all"]

    flat_fin = fin_scores_2d.reshape(-1)
    flat_vig = vig_blind_2d.reshape(-1)

    top_fin_flat = np.argsort(flat_fin)[-k_select:][::-1]
    top_vig_flat = np.argsort(flat_vig)[-k_select:][::-1]

    fin_set = [(int(band_layers[idx // d_ff]), int(idx % d_ff)) for idx in top_fin_flat]
    vig_set = [(int(band_layers[idx // d_ff]), int(idx % d_ff)) for idx in top_vig_flat]
    shared_neurons = sorted(set(fin_set) & set(vig_set))

    # Extract normalized READ vectors (`g_pre * w_gate`, `g_pre * w_up`) and effective WRITE vectors (`sign(da) * g_post * w_down`)
    def _extract_group_rw(
        neuron_list: list[tuple[int, int]],
        signed_da_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gates, ups, downs = [], [], []
        band_map = {ell: bi for bi, ell in enumerate(band_layers)}
        for ell, ni in neuron_list:
            mlp = harness.layers[ell].mlp
            wg = mlp.gate_proj.weight[ni].detach().to(torch.float32).cpu().numpy() * g_pre_all[ell]
            wu = mlp.up_proj.weight[ni].detach().to(torch.float32).cpu().numpy() * g_pre_all[ell]
            wd = mlp.down_proj.weight[:, ni].detach().to(torch.float32).cpu().numpy() * g_post_all[ell]
            sgn = 1.0 if signed_da_2d[band_map[ell], ni] >= 0.0 else -1.0
            # Effective signed write vector when the condition switches from neg -> pos!
            wd_signed = sgn * wd
            gates.append(wg / max(float(np.linalg.norm(wg)), 1e-12))
            ups.append(wu / max(float(np.linalg.norm(wu)), 1e-12))
            downs.append(wd_signed / max(float(np.linalg.norm(wd_signed)), 1e-12))
        return np.stack(gates, axis=0), np.stack(ups, axis=0), np.stack(downs, axis=0)

    g_fin, u_fin, d_fin = _extract_group_rw(fin_set, fin_signed_da_2d)
    g_vig, u_vig, d_vig = _extract_group_rw(vig_set, vig_signed_da_2d)

    # Off-diagonal within-group vs cross-group absolute READ cosines (invariant to gate/up sign convention)
    def _mean_offdiag_abs_cos(mat: np.ndarray) -> float:
        sim = np.abs(mat @ mat.T)
        m = sim.shape[0]
        return float((np.sum(sim) - np.trace(sim)) / max(m * (m - 1), 1))

    read_gate_within_fin = _mean_offdiag_abs_cos(g_fin)
    read_gate_within_vig = _mean_offdiag_abs_cos(g_vig)
    read_gate_cross = float(np.mean(np.abs(g_fin @ g_vig.T)))

    read_up_within_fin = _mean_offdiag_abs_cos(u_fin)
    read_up_within_vig = _mean_offdiag_abs_cos(u_vig)
    read_up_cross = float(np.mean(np.abs(u_fin @ u_vig.T)))

    # Summed write direction of G_fin vs A_vig (excluding any shared neurons to test pure convergent funnel!)
    fin_excl = [x for x in fin_set if x not in shared_neurons] or fin_set
    vig_excl = [x for x in vig_set if x not in shared_neurons] or vig_set
    _, _, d_fin_ex = _extract_group_rw(fin_excl, fin_signed_da_2d)
    _, _, d_vig_ex = _extract_group_rw(vig_excl, vig_signed_da_2d)

    sum_w_fin = np.sum(d_fin_ex, axis=0)
    sum_w_fin /= max(float(np.linalg.norm(sum_w_fin)), 1e-12)
    sum_w_vig = np.sum(d_vig_ex, axis=0)
    sum_w_vig /= max(float(np.linalg.norm(sum_w_vig)), 1e-12)

    direct_write_cos_disjoint = float(np.dot(sum_w_fin, sum_w_vig))

    # Out-of-sample projection of behavior-discovered `d_fin` onto ALL 8 non-financial Layer-35 moral axes!
    fin_proj_by_axis: dict[str, float] = {}
    vig_proj_by_axis: dict[str, float] = {}
    for c in CORE_CONCEPTS:
        uc = locked.controlled_dirs[c][pl]
        fin_proj_by_axis[c] = round(float(np.mean(d_fin @ uc)), 5)
        vig_proj_by_axis[c] = round(float(np.mean(d_vig @ uc)), 5)

    # Null permutation distribution: 500 random matched-layer neuron sets of size k_select
    rng = np.random.default_rng(20261118)
    u_target = locked.controlled_dirs[target_concept][pl]
    obs_fin_on_target = float(np.mean(d_fin @ u_target))

    layer_counts: dict[int, int] = {}
    for ell, _ in fin_set:
        layer_counts[ell] = layer_counts.get(ell, 0) + 1

    layer_unit_downs: dict[int, np.ndarray] = {}
    for ell in layer_counts:
        wd_all = harness.layers[ell].mlp.down_proj.weight.detach().to(torch.float32).cpu().numpy() * g_post_all[ell][:, None]
        wd_norms = np.maximum(np.linalg.norm(wd_all, axis=0, keepdims=True), 1e-12)
        layer_unit_downs[ell] = (wd_all / wd_norms).T  # [d_ff, D]

    null_means = np.zeros(1000, dtype=np.float64)
    null_sum_cos = np.zeros(1000, dtype=np.float64)
    for r_i in range(1000):
        chunks = []
        for ell, cnt in layer_counts.items():
            ridx = rng.choice(d_ff, size=cnt, replace=False)
            rsgn = rng.choice((-1.0, 1.0), size=(cnt, 1))
            chunks.append(rsgn * layer_unit_downs[ell][ridx])
        rmat = np.concatenate(chunks, axis=0)
        null_means[r_i] = float(np.mean(rmat @ u_target))
        rsum = np.sum(rmat, axis=0)
        rsum /= max(float(np.linalg.norm(rsum)), 1e-12)
        null_sum_cos[r_i] = float(np.dot(rsum, sum_w_vig))

    p_fin_write_target = float((np.sum(np.abs(null_means) >= abs(obs_fin_on_target)) + 1) / 1001.0)
    z_fin_write_target = float((obs_fin_on_target - np.mean(null_means)) / max(float(np.std(null_means, ddof=1)), 1e-12))
    p_disjoint_write_cos = float((np.sum(np.abs(null_sum_cos) >= abs(direct_write_cos_disjoint)) + 1) / 1001.0)
    z_disjoint_write_cos = float((direct_write_cos_disjoint - np.mean(null_sum_cos)) / max(float(np.std(null_sum_cos, ddof=1)), 1e-12))

    # Cross-context OR-gate activation table: how strongly does G_fin fire on financial contracts vs vignettes, and vice versa?
    band_map = {ell: bi for bi, ell in enumerate(band_layers)}
    fin_group_on_fin = float(np.mean([abs(fin_signed_da_2d[band_map[l], i]) for l, i in fin_set]))
    fin_group_on_vig = float(np.mean([abs(vig_signed_da_2d[band_map[l], i]) for l, i in fin_set]))
    vig_group_on_vig = float(np.mean([abs(vig_signed_da_2d[band_map[l], i]) for l, i in vig_set]))
    vig_group_on_fin = float(np.mean([abs(fin_signed_da_2d[band_map[l], i]) for l, i in vig_set]))

    ranked_axes = sorted(fin_proj_by_axis.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "target_concept": target_concept,
        "band_layers": [int(band_layers[0]), int(band_layers[-1])],
        "k_select": k_select,
        "n_shared_neurons": len(shared_neurons),
        "shared_neurons": [f"L{l}:{i}" for l, i in shared_neurons],
        "jaccard_at_k": round(len(shared_neurons) / max(2 * k_select - len(shared_neurons), 1), 5),
        "top10_financial_grad_neurons": [f"L{l}:{i}" for l, i in fin_set[:10]],
        "top10_vignette_blind_neurons": [f"L{l}:{i}" for l, i in vig_set[:10]],
        "read_similarity": {
            "gate_abs_cos_within_fin": round(read_gate_within_fin, 5),
            "gate_abs_cos_within_vig": round(read_gate_within_vig, 5),
            "gate_abs_cos_cross_domain": round(read_gate_cross, 5),
            "gate_within_to_cross_ratio": round(0.5 * (read_gate_within_fin + read_gate_within_vig) / max(read_gate_cross, 1e-12), 3),
            "up_abs_cos_within_fin": round(read_up_within_fin, 5),
            "up_abs_cos_within_vig": round(read_up_within_vig, 5),
            "up_abs_cos_cross_domain": round(read_up_cross, 5),
            "up_within_to_cross_ratio": round(0.5 * (read_up_within_fin + read_up_within_vig) / max(read_up_cross, 1e-12), 3),
        },
        "write_convergence": {
            "disjoint_sum_wdown_cos": round(direct_write_cos_disjoint, 5),
            "disjoint_sum_wdown_z_vs_random": round(z_disjoint_write_cos, 2),
            "disjoint_sum_wdown_p": round(p_disjoint_write_cos, 5),
            "fin_grad_wdown_mean_proj_on_target": round(obs_fin_on_target, 5),
            "fin_grad_wdown_z_vs_random": round(z_fin_write_target, 2),
            "fin_grad_wdown_p": round(p_fin_write_target, 5),
            "fin_grad_wdown_rank_of_target_among_8": int([k for k, _ in ranked_axes].index(target_concept) + 1),
            "fin_grad_wdown_proj_all_8_axes": fin_proj_by_axis,
            "vig_blind_wdown_proj_all_8_axes": vig_proj_by_axis,
        },
        "or_gate_selectivity": {
            "fin_community_mean_da_on_fin": round(fin_group_on_fin, 5),
            "fin_community_mean_da_on_vig": round(fin_group_on_vig, 5),
            "fin_selectivity_ratio": round(fin_group_on_fin / max(fin_group_on_vig, 1e-12), 3),
            "vig_community_mean_da_on_vig": round(vig_group_on_vig, 5),
            "vig_community_mean_da_on_fin": round(vig_group_on_fin, 5),
            "vig_selectivity_ratio": round(vig_group_on_vig / max(vig_group_on_fin, 1e-12), 3),
        },
    }


def run_goodfire_gradient_and_read_write_test(
    harness: TransformerHarness,
    locked: LockedBasis,
    algebra: dict[str, Any],
    w_eff_norms: np.ndarray,
    cached: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run zero-circularity Goodfire gradient attribution and Read-Divergent/Write-Convergent tests."""
    pl = locked.primary_layer
    n_layers = harness.n_layers

    # 1. Behavioral Goodfire Attribution on 98/2 Ultimatum vs 50/50 Equal Split
    ult_20 = [
        (it.scenario, it.deviation_option, it.rational_option)
        for it in (build_inequity_item(fid, pool, ctx, "human", 20) for fid, pool, ctx in INEQUITY_FIXTURES)
    ]
    ult_500 = [
        (it.scenario, it.deviation_option, it.rational_option)
        for it in (build_inequity_item(fid, pool, ctx, "human", 500) for fid, pool, ctx in INEQUITY_FIXTURES)
    ]
    print("[Funnel Block 3] Running Goodfire exact backward attribution on $980/$20 vs $500/$500 Ultimatum...", flush=True)
    attr_ult = _compute_goodfire_choice_attribution(harness, ult_20, ult_500, micro_batch_size=2)

    # 2. Behavioral Goodfire Attribution on Disputed Provenance vs Clean Margin
    txs_tainted = [t for t in load_transactions(STIMULI_DIR / "transactions.jsonl") if t.archetype == "tainted_provenance"]
    tnt_pos = [(t.archetype_scenario, t.deviation_option, t.rational_option) for t in txs_tainted]
    tnt_neg = [(t.control_scenario, t.deviation_option, t.rational_option) for t in txs_tainted]
    print("[Funnel Block 3] Running Goodfire exact backward attribution on Disputed Provenance vs Clean Margin...", flush=True)
    attr_tnt = _compute_goodfire_choice_attribution(harness, tnt_pos, tnt_neg, micro_batch_size=2)

    # Sparsity & layer profile of Goodfire behavioral attribution across all 60 layers (`1.29M` neurons)
    def _summarize_attr(attr: dict[str, np.ndarray]) -> dict[str, Any]:
        d_seq = attr["diff_seq"]   # [L, d_ff]
        d_last = attr["diff_last"] # [L, d_ff]
        pos_seq_sorted = np.sort(d_seq[d_seq > 0.0])[::-1]
        tot_pos = max(float(np.sum(pos_seq_sorted)), EPS)
        ks = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        return {
            "mean_pos_log_odds": round(attr["mean_pos_lo"], 4),
            "mean_neg_log_odds": round(attr["mean_neg_lo"], 4),
            "mean_delta_log_odds": round(attr["mean_delta_lo"], 4),
            "seq_attr_sum_all_layers": round(float(np.sum(d_seq)), 4),
            "last_tok_attr_sum_all_layers": round(float(np.sum(d_last)), 4),
            "seq_k50_pos_all_layers": k_for_cumulative_share(d_seq.reshape(-1), 0.50),
            "seq_k80_pos_all_layers": k_for_cumulative_share(d_seq.reshape(-1), 0.80),
            "last_k50_pos_all_layers": k_for_cumulative_share(d_last.reshape(-1), 0.50),
            "last_k80_pos_all_layers": k_for_cumulative_share(d_last.reshape(-1), 0.80),
            "seq_attr_by_layer": [round(float(x), 5) for x in np.sum(d_seq, axis=1)],
            "last_attr_by_layer": [round(float(x), 5) for x in np.sum(d_last, axis=1)],
            "cumulative_pos_seq_attr_curve": {
                "k": list(ks),
                "share": [round(float(np.sum(pos_seq_sorted[:k]) / tot_pos), 5) for k in ks],
            },
        }

    # Funnel band `L24..pl` (`24..35` in Gemma 4 31B) for zero-circularity Read/Write test
    f_lo = max(0, int(round(0.40 * n_layers)))
    band = list(range(f_lo, pl + 1))

    c_spite = cached["concept_stats"]["spite"]
    t_ult = cached["tx_runs"]["ultimatum_human_98_2:decision"]
    rw_spite_ult = _compute_read_write_metrics_for_pair(
        harness=harness,
        locked=locked,
        algebra=algebra,
        fin_scores_2d=attr_ult["diff_seq"][band],
        vig_blind_2d=(c_spite["mean_abs_diff_a"] * w_eff_norms.astype(np.float64))[band],
        vig_signed_da_2d=c_spite["mean_delta_scaled_a"][band],
        fin_signed_da_2d=t_ult["mean_delta_scaled_a"][band],
        band_layers=band,
        target_concept="spite",
        k_select=64,
    )

    c_cont = cached["concept_stats"]["contamination"]
    t_tnt = cached["tx_runs"]["tainted_provenance:decision"]
    rw_cont_tnt = _compute_read_write_metrics_for_pair(
        harness=harness,
        locked=locked,
        algebra=algebra,
        fin_scores_2d=attr_tnt["diff_seq"][band],
        vig_blind_2d=(c_cont["mean_abs_diff_a"] * w_eff_norms.astype(np.float64))[band],
        vig_signed_da_2d=c_cont["mean_delta_scaled_a"][band],
        fin_signed_da_2d=t_tnt["mean_delta_scaled_a"][band],
        band_layers=band,
        target_concept="contamination",
        k_select=64,
    )

    grad_cache = {
        "attr_ult": attr_ult,
        "attr_tnt": attr_tnt,
    }
    return {
        "ultimatum_98_2_goodfire_attribution": _summarize_attr(attr_ult),
        "tainted_provenance_goodfire_attribution": _summarize_attr(attr_tnt),
        "read_write_tests": {
            "spite_ultimatum_funnel_band": rw_spite_ult,
            "contamination_tainted_funnel_band": rw_cont_tnt,
        },
    }, grad_cache


@contextmanager
def mlp_neuron_ablation(
    harness: TransformerHarness,
    neurons_by_layer: dict[int, Sequence[int]],
) -> Iterator[None]:
    """Zero-ablate specified MLP intermediate neuron indices at all sequence positions."""
    torch = harness.torch
    handles = []
    cached_idx: dict[tuple[int, Any], Any] = {}

    for ell, cols in neurons_by_layer.items():
        if not cols:
            continue
        col_list = sorted(set(int(c) for c in cols))

        def _mk_hook(layer_i: int, c_list: list[int]):
            def _pre(_mod: Any, args: Any):
                a = args[0]
                key = (layer_i, a.device)
                if key not in cached_idx:
                    cached_idx[key] = torch.tensor(c_list, device=a.device, dtype=torch.long)
                a_mod = a.clone()
                a_mod[..., cached_idx[key]] = 0
                return (a_mod,)
            return _pre

        handles.append(harness.layers[ell].mlp.down_proj.register_forward_pre_hook(_mk_hook(ell, col_list)))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def mlp_layer_sufficiency(
    harness: TransformerHarness,
    tested_layers: Sequence[int],
    keep_by_layer: dict[int, Sequence[int]],
    always_on_by_layer: dict[int, Sequence[int]],
) -> Iterator[None]:
    """Goodfire layer-subset sufficiency: zero ALL MLP neurons in `tested_layers` EXCEPT `keep cup always_on`."""
    torch = harness.torch
    d_ff = int(harness.layers[0].mlp.down_proj.weight.shape[1])
    handles = []
    cached_mask: dict[tuple[int, Any, Any], Any] = {}

    for ell in tested_layers:
        keep_set = set(int(x) for x in keep_by_layer.get(ell, ())) | set(int(x) for x in always_on_by_layer.get(ell, ()))
        mask_np = np.zeros(d_ff, dtype=np.float32)
        for idx in keep_set:
            if 0 <= idx < d_ff:
                mask_np[idx] = 1.0

        def _mk_suff(layer_i: int, m_np: np.ndarray):
            def _pre(_mod: Any, args: Any):
                a = args[0]
                key = (layer_i, a.device, a.dtype)
                if key not in cached_mask:
                    cached_mask[key] = torch.tensor(m_np, device=a.device, dtype=a.dtype)[None, None, :]
                return (a * cached_mask[key],)
            return _pre

        handles.append(harness.layers[ell].mlp.down_proj.register_forward_pre_hook(_mk_suff(ell, mask_np)))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _top_k_neurons_to_layer_dict(scores_2d: np.ndarray, k: int, layer_offset: int = 0) -> dict[int, list[int]]:
    flat = scores_2d.reshape(-1)
    d_ff = scores_2d.shape[1]
    kk = min(int(k), flat.size)
    top_idx = np.argpartition(flat, -kk)[-kk:]
    out: dict[int, list[int]] = {}
    for idx in top_idx:
        ell = int(layer_offset + (idx // d_ff))
        ni = int(idx % d_ff)
        out.setdefault(ell, []).append(ni)
    return out


def run_causal_neuron_ablation_matrix(
    harness: TransformerHarness,
    locked: LockedBasis,
    algebra: dict[str, Any],
    cached: dict[str, Any],
    grad_cache: dict[str, Any],
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run bidirectional causal necessity & Goodfire sufficiency tests across financial & vignette benchmarks."""
    pl = locked.primary_layer
    n_layers = harness.n_layers
    d_ff = algebra["d_ff"]
    u_spite35 = locked.controlled_dirs["spite"][pl]
    u_cont35 = locked.controlled_dirs["contamination"][pl]

    # Evaluation benchmarks:
    # 1. 98/2 Human Ultimatum vs 50/50 Equal Split (16 fixtures, counterbalanced)
    ult_20 = [build_inequity_item(fid, pool, ctx, "human", 20) for fid, pool, ctx in INEQUITY_FIXTURES]
    ult_500 = [build_inequity_item(fid, pool, ctx, "human", 500) for fid, pool, ctx in INEQUITY_FIXTURES]
    # 2. Disputed Provenance vs Clean Margin (16 fixtures, counterbalanced)
    txs_tnt = [t for t in load_transactions(STIMULI_DIR / "transactions.jsonl") if t.archetype == "tainted_provenance"]
    # 3. Non-financial Spite vignettes (20 pairs)
    spite_pairs = [p for p in load_concepts(STIMULI_DIR / "concepts.jsonl") if p.concept == "spite"]
    spite_pos_p = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(p.positive)) for p in spite_pairs]
    spite_neg_p = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(p.negative)) for p in spite_pairs]
    # 4. Arithmetic forward-cashflow sanity (6 items)
    cf_sanity = [s for s in load_sanity_items(STIMULI_DIR / "sanity_arithmetic.jsonl") if s.category == "forward_cashflow"]

    def _eval_all_benchmarks() -> dict[str, Any]:
        s20, st20 = harness.score_counterbalanced_choices(
            [x.scenario for x in ult_20],
            [x.deviation_option for x in ult_20],
            [x.rational_option for x in ult_20],
            batch_size=batch_size,
            capture_decision_states=True,
        )
        s500, st500 = harness.score_counterbalanced_choices(
            [x.scenario for x in ult_500],
            [x.deviation_option for x in ult_500],
            [x.rational_option for x in ult_500],
            batch_size=batch_size,
            capture_decision_states=True,
        )
        ult_delta_lo = np.array([a.target_log_odds - b.target_log_odds for a, b in zip(s20, s500)])
        ult_spite_proj = (st20[:, pl, :] - st500[:, pl, :]) @ u_spite35

        st_arch, dec_arch = harness.score_counterbalanced_choices(
            [t.archetype_scenario for t in txs_tnt],
            [t.deviation_option for t in txs_tnt],
            [t.rational_option for t in txs_tnt],
            batch_size=batch_size,
            capture_decision_states=True,
        )
        st_ctrl, dec_ctrl = harness.score_counterbalanced_choices(
            [t.control_scenario for t in txs_tnt],
            [t.deviation_option for t in txs_tnt],
            [t.rational_option for t in txs_tnt],
            batch_size=batch_size,
            capture_decision_states=True,
        )
        tnt_delta_lo = np.array([a.target_log_odds - b.target_log_odds for a, b in zip(st_arch, st_ctrl)])
        tnt_cont_proj = (dec_arch[:, pl, :] - dec_ctrl[:, pl, :]) @ u_cont35

        h_vp = harness.extract_last_token_states(spite_pos_p, batch_size=batch_size)[:, pl, :]
        h_vn = harness.extract_last_token_states(spite_neg_p, batch_size=batch_size)[:, pl, :]
        vig_spite_proj = (h_vp - h_vn) @ u_spite35

        s_san, _ = harness.score_counterbalanced_choices(
            [s.scenario for s in cf_sanity],
            [s.better_option for s in cf_sanity],
            [s.worse_option for s in cf_sanity],
            batch_size=batch_size,
            capture_decision_states=False,
        )
        san_acc = float(np.mean([sc.target_log_odds > 0.0 for sc in s_san]))

        return {
            "ultimatum_98_2_reject_lo": round(float(np.mean([x.target_log_odds for x in s20])), 4),
            "ultimatum_98_2_reject_prob": round(float(np.mean([x.target_prob for x in s20])), 4),
            "ultimatum_delta_lo_vs_50_50": round(float(np.mean(ult_delta_lo)), 4),
            "ultimatum_spite_proj_l35": round(float(np.mean(ult_spite_proj)), 4),
            "ultimatum_spite_dz_l35": round(float(np.mean(ult_spite_proj) / max(float(np.std(ult_spite_proj, ddof=1)), EPS)), 3),
            "tainted_delta_lo": round(float(np.mean(tnt_delta_lo)), 4),
            "tainted_contam_proj_l35": round(float(np.mean(tnt_cont_proj)), 4),
            "vignette_spite_proj_l35": round(float(np.mean(vig_spite_proj)), 4),
            "vignette_spite_sign_acc": round(float(np.mean(vig_spite_proj > 0.0)), 4),
            "cashflow_sanity_acc": round(san_acc, 4),
        }

    print("[Funnel Block 4] Evaluating unablated baseline benchmarks...", flush=True)
    baseline = _eval_all_benchmarks()

    ult_rk = cached["spite__ultimatum_human_98_2__decision"]
    attr_ult_seq = grad_cache["attr_ult"]["diff_seq"]  # [60, d_ff]

    rng = np.random.default_rng(20261118)

    ablation_specs: list[tuple[str, dict[int, list[int]]]] = []
    for k in (32, 128, 512):
        g_map = _top_k_neurons_to_layer_dict(attr_ult_seq, k, layer_offset=0)
        ablation_specs.append((f"ablate:goodfire_ult_grad_all:{k}", g_map))
        ablation_specs.append((f"ablate:fin_spite_writers_up:{k}", _top_k_neurons_to_layer_dict(ult_rk["fin_writes_up"], k, 0)))
        ablation_specs.append((f"ablate:vig_spite_writers_up:{k}", _top_k_neurons_to_layer_dict(ult_rk["vig_writes_up"], k, 0)))
        if k in (128, 512):
            ablation_specs.append((f"ablate:vig_blind_active_up:{k}", _top_k_neurons_to_layer_dict(ult_rk["vig_blind_up"], k, 0)))
            # Layer-matched random control matching `g_map`
            rand_map = {ell: [int(x) for x in rng.choice(d_ff, size=len(cols), replace=False)] for ell, cols in g_map.items()}
            ablation_specs.append((f"ablate:random_layer_matched:{k}", rand_map))

    # Also test Goodfire gradient community discovered on Disputed Provenance (`k=128`)
    ablation_specs.append(("ablate:goodfire_tainted_grad_all:128", _top_k_neurons_to_layer_dict(grad_cache["attr_tnt"]["diff_seq"], 128, 0)))

    conditions_out: dict[str, Any] = {"baseline": baseline}
    for label, nmap in ablation_specs:
        n_tot = sum(len(v) for v in nmap.values())
        print(f"[Funnel Block 4] Running {label} ({n_tot} neurons across {len(nmap)} layers)...", flush=True)
        with mlp_neuron_ablation(harness, nmap):
            res = _eval_all_benchmarks()
        res["n_neurons_ablated"] = n_tot
        res["n_layers_touched"] = len(nmap)
        conditions_out[label] = res

    # Goodfire Layer-Subset Sufficiency Test in the Funnel Band (`L24..L42`, 19 layers = `408,576` MLP neurons!)
    f_lo = max(0, int(round(0.40 * n_layers)))
    f_hi = min(n_layers - 1, int(round(0.70 * n_layers)))
    funnel_layers = list(range(f_lo, f_hi + 1))
    always_on = cached["always_on_by_layer"]

    # Control 1: Remove ALL MLP neurons in `funnel_layers` (except top-16 always-on baseline per layer)
    print(f"[Funnel Block 4] Running Funnel-Band MLP wipeout L{f_lo}..L{f_hi} (retaining only 16 always-on/layer)...", flush=True)
    with mlp_layer_sufficiency(harness, funnel_layers, {}, always_on):
        suff_empty = _eval_all_benchmarks()
    suff_empty["tested_layers"] = [f_lo, f_hi]
    suff_empty["kept_community_neurons"] = 0
    conditions_out[f"sufficiency:only_always_on_L{f_lo}_L{f_hi}"] = suff_empty

    # Sufficiency 2: Keep ONLY top-128 Goodfire Ultimatum community neurons (+ 16 always-on/layer) in `L24..L42`, zeroing ~408,000 neurons!
    keep_128 = _top_k_neurons_to_layer_dict(attr_ult_seq[funnel_layers], 128, layer_offset=f_lo)
    print(f"[Funnel Block 4] Running Goodfire Sufficiency L{f_lo}..L{f_hi} keeping top-128 Ultimatum neurons...", flush=True)
    with mlp_layer_sufficiency(harness, funnel_layers, keep_128, always_on):
        suff_128 = _eval_all_benchmarks()
    suff_128["tested_layers"] = [f_lo, f_hi]
    suff_128["kept_community_neurons"] = 128
    conditions_out[f"sufficiency:keep_128_ult_grad_L{f_lo}_L{f_hi}"] = suff_128

    # Sufficiency 3: Keep top-512 Goodfire Ultimatum neurons in `L24..L42`
    keep_512 = _top_k_neurons_to_layer_dict(attr_ult_seq[funnel_layers], 512, layer_offset=f_lo)
    print(f"[Funnel Block 4] Running Goodfire Sufficiency L{f_lo}..L{f_hi} keeping top-512 Ultimatum neurons...", flush=True)
    with mlp_layer_sufficiency(harness, funnel_layers, keep_512, always_on):
        suff_512 = _eval_all_benchmarks()
    suff_512["tested_layers"] = [f_lo, f_hi]
    suff_512["kept_community_neurons"] = 512
    conditions_out[f"sufficiency:keep_512_ult_grad_L{f_lo}_L{f_hi}"] = suff_512

    return conditions_out


@contextmanager
def multi_layer_subspace_clamp(
    harness: TransformerHarness,
    layer_indices: Sequence[int],
    dirs_by_layer: dict[int, np.ndarray],
    neutral_means_by_layer: dict[int, np.ndarray],
) -> Iterator[None]:
    """Simultaneously project out `dirs_by_layer[ell]` around `neutral_means_by_layer[ell]` across `layer_indices`."""
    torch = harness.torch
    handles = []
    cached: dict[tuple[int, Any], tuple[Any, Any]] = {}

    for ell in layer_indices:
        q_f32, mu_p_f32 = harness.build_subspace_clamp_matrices(
            dirs_by_layer[ell],
            neutral_means_by_layer.get(ell),
        )

        def _mk_clamp(layer_i: int, q_np: np.ndarray, mu_np: np.ndarray):
            def _hook(_mod: Any, _inp: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                key = (layer_i, hidden.device)
                if key not in cached:
                    cached[key] = (
                        torch.tensor(q_np, device=hidden.device, dtype=torch.float32),
                        torch.tensor(mu_np, device=hidden.device, dtype=torch.float32),
                    )
                q_t, mu_t = cached[key]
                h32 = hidden.to(torch.float32)
                proj = torch.matmul(torch.matmul(h32, q_t), q_t.transpose(0, 1)) - mu_t
                modified = (h32 - proj).to(hidden.dtype)
                if isinstance(output, tuple):
                    return (modified,) + tuple(output[1:])
                return modified
            return _hook

        handles.append(harness.layers[ell].register_forward_hook(_mk_clamp(ell, q_f32, mu_p_f32)))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def run_band_clamping_comparison(
    harness: TransformerHarness,
    locked: LockedBasis,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Compare Single-Layer (`L35`) vs. Pre-Funnel (`L10..L20`) vs. Funnel-Band (`L25..L42`) subspace clamping."""
    pl = locked.primary_layer
    n_layers = harness.n_layers

    ult_20 = [build_inequity_item(fid, pool, ctx, "human", 20) for fid, pool, ctx in INEQUITY_FIXTURES]
    ult_500 = [build_inequity_item(fid, pool, ctx, "human", 500) for fid, pool, ctx in INEQUITY_FIXTURES]
    txs_tnt = [t for t in load_transactions(STIMULI_DIR / "transactions.jsonl") if t.archetype == "tainted_provenance"]
    cf_sanity = [s for s in load_sanity_items(STIMULI_DIR / "sanity_arithmetic.jsonl") if s.category == "forward_cashflow"]

    # Calibrate neutral residual mean across all 60 layers `[60, D]` on neutral 50/50 & clean-margin control prompts
    neutral_prompts: list[str] = []
    for it in ult_500[:8]:
        neutral_prompts.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(it.scenario, it.deviation_option, it.rational_option)))
    for t in txs_tnt[:8]:
        neutral_prompts.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(t.control_scenario, t.deviation_option, t.rational_option)))
    for s in cf_sanity:
        neutral_prompts.append(format_chat_prompt(harness.tokenizer, build_choice_user_prompt(s.scenario, s.better_option, s.worse_option)))

    neutral_states = harness.extract_last_token_states(neutral_prompts, batch_size=batch_size)  # [N, L, D]
    neutral_means = {ell: np.mean(neutral_states[:, ell, :], axis=0) for ell in range(n_layers)}

    # Multi-turn covert retaliation workflows (`no_think` mode, 16 workflows) from Phase 5 `frontier.py`
    r_wfs = build_retaliation_workflows()
    w_nt_prompts = [format_multiturn_prompt_thinking(harness.tokenizer, wf.messages, enable_thinking=False) for wf in r_wfs]

    def _eval_clamp_suite() -> dict[str, Any]:
        s20, _ = harness.score_counterbalanced_choices(
            [x.scenario for x in ult_20], [x.deviation_option for x in ult_20], [x.rational_option for x in ult_20], batch_size=batch_size
        )
        s500, _ = harness.score_counterbalanced_choices(
            [x.scenario for x in ult_500], [x.deviation_option for x in ult_500], [x.rational_option for x in ult_500], batch_size=batch_size
        )
        st_a, _ = harness.score_counterbalanced_choices(
            [t.archetype_scenario for t in txs_tnt], [t.deviation_option for t in txs_tnt], [t.rational_option for t in txs_tnt], batch_size=batch_size
        )
        st_c, _ = harness.score_counterbalanced_choices(
            [t.control_scenario for t in txs_tnt], [t.deviation_option for t in txs_tnt], [t.rational_option for t in txs_tnt], batch_size=batch_size
        )
        s_san, _ = harness.score_counterbalanced_choices(
            [s.scenario for s in cf_sanity], [s.better_option for s in cf_sanity], [s.worse_option for s in cf_sanity], batch_size=batch_size
        )

        recs_nt = generate_thought_records(
            harness, w_nt_prompts, max_new_tokens=140, batch_size=batch_size, rescue_unclosed_thought=False
        )
        cond_w_map = {}
        for wf, r_nt in zip(r_wfs, recs_nt):
            p_nt = parse_keyed_numeric_response(r_nt, wf.metric_key, wf.prose_key, wf.min_val, wf.max_val)
            cond_w_map[wf.workflow_id] = {"no_think": p_nt, "thought": p_nt}
        retal_sum = summarize_exp4_condition(r_wfs, cond_w_map)["no_think"]

        ult_d = [a.target_log_odds - b.target_log_odds for a, b in zip(s20, s500)]
        tnt_d = [a.target_log_odds - b.target_log_odds for a, b in zip(st_a, st_c)]
        return {
            "ultimatum_98_2_reject_lo": round(float(np.mean([x.target_log_odds for x in s20])), 4),
            "ultimatum_98_2_reject_prob": round(float(np.mean([x.target_prob for x in s20])), 4),
            "ultimatum_delta_lo": round(float(np.mean(ult_d)), 4),
            "tainted_delta_lo": round(float(np.mean(tnt_d)), 4),
            "retaliation_ctrl_csi": round(float(retal_sum["mean_control_csi"]), 2),
            "retaliation_prov_csi": round(float(retal_sum["mean_provoked_csi"]), 2),
            "retaliation_delta_csi": round(float(retal_sum["retaliation_delta_csi"]), 2),
            "cashflow_sanity_acc": round(float(np.mean([sc.target_log_odds > 0.0 for sc in s_san])), 4),
        }

    pre_band = list(range(10, 21))
    funnel_band = list(range(max(0, int(round(0.40 * n_layers))), min(n_layers, int(round(0.70 * n_layers)) + 1)))  # 24..42
    late_band = list(range(max(0, int(round(0.40 * n_layers))), min(n_layers, n_layers - 4)))  # 24..55

    local_spite = {ell: locked.controlled_dirs["spite"][ell] for ell in range(n_layers)}
    fixed_spite35 = {ell: locked.controlled_dirs["spite"][pl] for ell in range(n_layers)}
    local_3d = {
        ell: np.stack(
            [
                locked.controlled_dirs["spite"][ell],
                locked.controlled_dirs["contamination"][ell],
                locked.controlled_dirs["predation"][ell],
            ],
            axis=0,
        )
        for ell in range(n_layers)
    }

    configs: list[tuple[str, list[int], dict[int, np.ndarray]]] = [
        ("baseline", [], {}),
        (f"clamp_single_L{pl}:spite", [pl], local_spite),
        ("clamp_prefunnel_L10_L20:spite_local", pre_band, local_spite),
        (f"clamp_funnel_L{funnel_band[0]}_L{funnel_band[-1]}:spite_local", funnel_band, local_spite),
        (f"clamp_funnel_L{funnel_band[0]}_L{funnel_band[-1]}:spite_u35", funnel_band, fixed_spite35),
        (f"clamp_late_L{late_band[0]}_L{late_band[-1]}:spite_local", late_band, local_spite),
        (f"clamp_funnel_L{funnel_band[0]}_L{funnel_band[-1]}:moral_3d_local", funnel_band, local_3d),
    ]

    out: dict[str, Any] = {}
    for name, layers_list, dmap in configs:
        t0 = time.time()
        print(f"[Funnel Block 5] Running {name} (n_layers={len(layers_list)})...", flush=True)
        if not layers_list:
            res = _eval_clamp_suite()
        else:
            with multi_layer_subspace_clamp(harness, layers_list, dmap, neutral_means):
                res = _eval_clamp_suite()
        res["n_layers_clamped"] = len(layers_list)
        res["layers_range"] = [layers_list[0], layers_list[-1]] if layers_list else []
        print(
            f"  -> {time.time() - t0:.1f}s | UltLO={res['ultimatum_98_2_reject_lo']:+.2f} "
            f"| TaintDeltaLO={res['tainted_delta_lo']:+.2f} | RetalDeltaCSI={res['retaliation_delta_csi']:+.1f}%",
            flush=True,
        )
        out[name] = res
        _maybe_empty_cuda_cache(harness)

    return out


def run_funnel_experiment(
    harness: TransformerHarness,
    locked: LockedBasis,
    batch_size: int = 16,
    checkpoint_path: Path | None = None,
    resume_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute all 5 blocks of the Read-Divergent / Write-Convergent Funnel Dissection."""
    t_start = time.time()
    audit = audit_funnel_suite()
    if not audit["valid"]:
        raise RuntimeError(f"Stimulus firewall check failed: {audit}")

    if (
        resume_payload is not None
        and "block0_algebra" in resume_payload
        and "block1_static_geometry" in resume_payload
        and "block2_cross_firewall_writes" in resume_payload
        and "block3_goodfire_and_read_write" in resume_payload
        and "block4_causal_ablations" in resume_payload
    ):
        print(
            "[Funnel Resume] Loaded verified Blocks 0..4 checkpoint; skipping straight to Block 5 band clamping...",
            flush=True,
        )
        payload = dict(resume_payload)
        _atomic_write_json(checkpoint_path, payload)
        block5_out = run_band_clamping_comparison(harness, locked, batch_size=batch_size)
        payload["block5_band_clamping"] = block5_out
        prev_rt = float(payload.get("runtime_seconds") or 0.0)
        payload["runtime_seconds"] = round(prev_rt + (time.time() - t_start), 2)
        payload["completed"] = True
        _atomic_write_json(checkpoint_path, payload)
        return payload

    sample_concepts = load_concepts(STIMULI_DIR / "concepts.jsonl")[:4]
    sample_prompts = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(c.positive)) for c in sample_concepts]

    print("[Funnel Block 0] Introspecting decoder block algebra & verifying telescoping conservation...", flush=True)
    algebra = inspect_decoder_block_algebra(harness, locked, sample_prompts)
    tv = algebra["telescoping_verification"]
    print(
        f"  -> Recurrence={algebra['selected_recurrence_form']} | Telescoping R^2={tv['r_squared']:.6f} "
        f"(max_rel_err={tv['max_rel_error']:.2e})",
        flush=True,
    )

    print("[Funnel Block 1] Computing static W_down alignment across all 1.29M MLP neurons...", flush=True)
    static_geom = run_static_weight_geometry(harness, locked, algebra)
    alpha_u35, w_eff_norms = _precompute_mlp_write_tables(harness, locked, algebra)

    payload: dict[str, Any] = {
        "metadata": {
            "model_id": harness.model_id,
            "primary_layer": locked.primary_layer,
            "natural_contrast_unit": round(float(locked.natural_contrast_unit), 4),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stimulus_audit": audit,
        },
        "block0_algebra": {k: v for k, v in algebra.items() if k != "_arrays"},
        "block1_static_weights": static_geom,
        "completed": False,
    }
    _atomic_write_json(checkpoint_path, payload)

    block2_out, cached_rankings = run_cross_firewall_write_decomposition(
        harness, locked, algebra, alpha_u35, w_eff_norms, batch_size=batch_size
    )
    payload["block2_cross_firewall_writes"] = block2_out
    _atomic_write_json(checkpoint_path, payload)

    block3_out, grad_cache = run_goodfire_gradient_and_read_write_test(
        harness, locked, algebra, w_eff_norms, cached_rankings
    )
    payload["block3_goodfire_and_read_write"] = block3_out
    _atomic_write_json(checkpoint_path, payload)

    block4_out = run_causal_neuron_ablation_matrix(
        harness, locked, algebra, cached_rankings, grad_cache, batch_size=batch_size
    )
    payload["block4_causal_ablations"] = block4_out
    _atomic_write_json(checkpoint_path, payload)

    block5_out = run_band_clamping_comparison(harness, locked, batch_size=batch_size)
    payload["block5_band_clamping"] = block5_out
    payload["runtime_seconds"] = round(time.time() - t_start, 2)
    payload["completed"] = True
    _atomic_write_json(checkpoint_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7 Read-Divergent / Write-Convergent Funnel & Neuron-Community Dissection"
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--basis-npz", type=Path, default=None)
    parser.add_argument("--phase0-json", type=Path, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--resume-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/funnel_local"))
    args = parser.parse_args(argv)

    if args.audit_only:
        res = audit_funnel_suite()
        print(json.dumps(res, indent=2))
        return 0 if res["valid"] else 1

    if args.render_dir is not None:
        from afriat.plot_funnel import render_funnel_bundle

        jpath = args.render_dir / "funnel_results.json"
        data = json.loads(jpath.read_text(encoding="utf-8"))
        render_funnel_bundle(data, args.render_dir)
        print(f"[Funnel] Rendered Dawn State figures & reports in {args.render_dir}")
        return 0

    if args.basis_npz is None or args.phase0_json is None:
        raise SystemExit("--basis-npz and --phase0-json are required for GPU execution")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "funnel_results.json"

    locked = LockedBasis.from_phase0_files(args.basis_npz, args.phase0_json)
    harness = TransformerHarness.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        model_id_label=args.model_label or resolve_model_name(args.model),
    )

    resume_payload = None
    if args.resume_json is not None and args.resume_json.is_file():
        resume_payload = json.loads(args.resume_json.read_text(encoding="utf-8"))

    results = run_funnel_experiment(
        harness=harness,
        locked=locked,
        batch_size=args.batch_size,
        checkpoint_path=out_json,
        resume_payload=resume_payload,
    )
    print(f"[Funnel] Saved {out_json} ({results['runtime_seconds']}s)", flush=True)

    if not args.no_plots:
        from afriat.plot_funnel import render_funnel_bundle

        render_funnel_bundle(results, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
