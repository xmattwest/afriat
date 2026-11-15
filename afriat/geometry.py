"""Subspace extraction, valence/arousal orthogonalization, and exact sign-flip inference."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from afriat.schema import ALL_CONCEPTS, CONTROL_CONCEPTS, CORE_ARCHETYPES, CORE_CONCEPTS

EPS = 1e-12


@dataclass
class LayerBasis:
    """Extracted directions and diagnostics at all layers [L, D]."""

    concepts: tuple[str, ...]
    raw_dirs: dict[str, np.ndarray]          # concept -> [L, D] unit vectors
    controlled_dirs: dict[str, np.ndarray]   # core concept -> [L, D] unit vectors orthogonal to (valence, arousal)
    residual_ratio: dict[str, np.ndarray]    # core concept -> [L] norm retained after removing (valence, arousal)
    lopo_accuracy: dict[str, np.ndarray]     # concept -> [L] leave-one-pair-out sign accuracy in [0, 1]
    lopo_cosine: dict[str, np.ndarray]       # concept -> [L] mean leave-one-pair-out cosine
    contrast_rms: dict[str, np.ndarray]      # concept -> [L] RMS norm of (h_pos - h_neg)
    stream_rms: np.ndarray                   # [L] RMS norm of pooled hidden states (for dimensionless steering)


def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norms, EPS)


def exact_or_mc_signflip_p(
    values: np.ndarray,
    two_sided: bool = True,
    max_exact_n: int = 16,
    n_mc: int = 20_000,
    seed: int = 20261114,
) -> float:
    """Exact 2^N sign-flip permutation p-value for N <= max_exact_n, seeded MC otherwise."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return 1.0
    obs = float(np.mean(x))
    if np.abs(obs) < EPS:
        return 1.0

    if n <= max_exact_n:
        # Exhaustive 2^n sign matrix in {-1, +1}
        bits = ((np.arange(1 << n, dtype=np.uint32)[:, None] >> np.arange(n, dtype=np.uint32)) & 1)
        signs = 2.0 * bits.astype(np.float64) - 1.0
        null_means = (signs @ x) / n
        if two_sided:
            extreme = np.sum(np.abs(null_means) >= np.abs(obs) - 1e-15)
        else:
            extreme = np.sum(null_means >= obs - 1e-15) if obs > 0 else np.sum(null_means <= obs + 1e-15)
        return float(extreme / (1 << n))

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_mc, n))
    null_means = (signs @ x) / n
    if two_sided:
        extreme = int(np.sum(np.abs(null_means) >= np.abs(obs) - 1e-15))
    else:
        extreme = int(np.sum(null_means >= obs - 1e-15)) if obs > 0 else int(np.sum(null_means <= obs + 1e-15))
    return float((extreme + 1) / (n_mc + 1))


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Step-down Holm-Bonferroni familywise error rate adjustment."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    scaled = ranked * (m - np.arange(m))
    enforced = np.maximum.accumulate(scaled)
    clipped = np.clip(enforced, 0.0, 1.0)
    out = np.empty_like(clipped)
    out[order] = clipped
    return out


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 20261114,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def extract_concept_basis(
    pos_by_concept: dict[str, np.ndarray],
    neg_by_concept: dict[str, np.ndarray],
    concepts: tuple[str, ...] = ALL_CONCEPTS,
    controls: tuple[str, ...] = CONTROL_CONCEPTS,
) -> LayerBasis:
    """Extract raw and valence/arousal-orthogonalized concept directions across layers.

    Args:
        pos_by_concept: concept -> array of shape [N_pairs, L, D]
        neg_by_concept: concept -> array of shape [N_pairs, L, D]
    """
    sample = next(iter(pos_by_concept.values()))
    _, n_layers, d_model = sample.shape

    raw_dirs: dict[str, np.ndarray] = {}
    lopo_accuracy: dict[str, np.ndarray] = {}
    lopo_cosine: dict[str, np.ndarray] = {}
    contrast_rms: dict[str, np.ndarray] = {}

    stream_sq_sum = np.zeros(n_layers, dtype=np.float64)
    stream_count = 0

    for c in concepts:
        pos = np.asarray(pos_by_concept[c], dtype=np.float64)
        neg = np.asarray(neg_by_concept[c], dtype=np.float64)
        if pos.shape != neg.shape:
            raise ValueError(f"Shape mismatch for concept '{c}': {pos.shape} vs {neg.shape}")
        n_pairs = pos.shape[0]
        diffs = pos - neg  # [N, L, D]

        stream_sq_sum += np.sum(np.linalg.norm(pos, axis=-1) ** 2 + np.linalg.norm(neg, axis=-1) ** 2, axis=0)
        stream_count += 2 * n_pairs

        mean_diff = np.mean(diffs, axis=0)  # [L, D]
        raw_dirs[c] = _unit(mean_diff, axis=-1)
        contrast_rms[c] = np.sqrt(np.mean(np.sum(diffs ** 2, axis=-1), axis=0))

        # Leave-one-pair-out validation: sum_other = total_sum - diffs[i]
        total_sum = np.sum(diffs, axis=0, keepdims=True)  # [1, L, D]
        loo_dirs = _unit(total_sum - diffs, axis=-1)      # [N, L, D]
        diff_units = _unit(diffs, axis=-1)                # [N, L, D]
        cos_per_pair = np.sum(diff_units * loo_dirs, axis=-1)  # [N, L]
        lopo_accuracy[c] = np.mean(cos_per_pair > 0.0, axis=0)
        lopo_cosine[c] = np.mean(cos_per_pair, axis=0)

    stream_rms = np.sqrt(stream_sq_sum / max(stream_count, 1))

    # Orthogonalize core concepts against span(controls) layer by layer via QR
    controlled_dirs: dict[str, np.ndarray] = {}
    residual_ratio: dict[str, np.ndarray] = {}
    core_list = [c for c in concepts if c not in controls]

    for c in core_list:
        ctrl_vecs = np.zeros((n_layers, d_model), dtype=np.float64)
        ratios = np.zeros(n_layers, dtype=np.float64)
        for ell in range(n_layers):
            u = raw_dirs[c][ell].copy()
            if controls:
                mat = np.column_stack([raw_dirs[k][ell] for k in controls])  # [D, K]
                q, _ = np.linalg.qr(mat, mode="reduced")                     # [D, K] orthonormal
                proj = q @ (q.T @ u)
                resid = u - proj
            else:
                resid = u
            norm_r = float(np.linalg.norm(resid))
            ratios[ell] = norm_r
            ctrl_vecs[ell] = resid / max(norm_r, EPS)
        controlled_dirs[c] = ctrl_vecs
        residual_ratio[c] = ratios

    return LayerBasis(
        concepts=concepts,
        raw_dirs=raw_dirs,
        controlled_dirs=controlled_dirs,
        residual_ratio=residual_ratio,
        lopo_accuracy=lopo_accuracy,
        lopo_cosine=lopo_cosine,
        contrast_rms=contrast_rms,
        stream_rms=stream_rms,
    )


def select_cartography_layer(
    basis: LayerBasis,
    core_concepts: tuple[str, ...] = CORE_CONCEPTS,
    min_frac_depth: float = 0.25,
    max_frac_depth: float = 0.85,
) -> int:
    """Pick the strongest concept-separating middle-to-late layer purely from non-financial basis CV!

    Crucial methodological guardrail: choosing the primary heatmap layer ONLY from non-financial
    LOPO cosine consistency avoids double-dipping / cherry-picking on the financial transactions.
    """
    n_layers = basis.stream_rms.shape[0]
    lo = max(0, int(np.floor(min_frac_depth * (n_layers - 1))))
    hi = min(n_layers - 1, int(np.ceil(max_frac_depth * (n_layers - 1))))
    if hi < lo:
        lo, hi = 0, n_layers - 1

    scores = np.mean(np.stack([basis.lopo_cosine[c] for c in core_concepts], axis=0), axis=0)
    window = scores[lo : hi + 1]
    return int(lo + int(np.argmax(window)))


def compute_cartography_projections(
    arch_by_tx: dict[str, np.ndarray],
    ctrl_by_tx: dict[str, np.ndarray],
    basis: LayerBasis,
    archetypes: tuple[str, ...] = CORE_ARCHETYPES,
    core_concepts: tuple[str, ...] = CORE_CONCEPTS,
    primary_layer: int | None = None,
) -> dict[str, object]:
    """Project (h_archetype - h_control) onto raw & affect-controlled subspaces across all layers."""
    if primary_layer is None:
        primary_layer = select_cartography_layer(basis, core_concepts=core_concepts)

    n_layers = basis.stream_rms.shape[0]
    n_arch = len(archetypes)
    n_core = len(core_concepts)

    # Full layer curves [n_arch, n_core, L] for paired Cohen's dz and mean cosine projection
    dz_controlled_curves = np.zeros((n_arch, n_core, n_layers), dtype=np.float64)
    cos_controlled_curves = np.zeros((n_arch, n_core, n_layers), dtype=np.float64)
    dz_raw_curves = np.zeros((n_arch, n_core, n_layers), dtype=np.float64)

    # Control axes curves [n_arch, len(CONTROL_CONCEPTS), L]
    ctrl_axes = [c for c in basis.concepts if c not in core_concepts]
    dz_affect_curves = np.zeros((n_arch, len(ctrl_axes), n_layers), dtype=np.float64)

    primary_cells: list[dict[str, object]] = []

    for i_a, arch in enumerate(archetypes):
        h_arch = np.asarray(arch_by_tx[arch], dtype=np.float64)  # [M, L, D]
        h_ctrl = np.asarray(ctrl_by_tx[arch], dtype=np.float64)  # [M, L, D]
        deltas = h_arch - h_ctrl                                  # [M, L, D]
        delta_units = _unit(deltas, axis=-1)                      # [M, L, D]

        for i_k, ctrl_name in enumerate(ctrl_axes):
            u_raw = basis.raw_dirs[ctrl_name]  # [L, D]
            proj = np.einsum("mld,ld->ml", deltas, u_raw) / np.maximum(basis.stream_rms[None, :], EPS)
            mean_p = np.mean(proj, axis=0)
            sd_p = np.std(proj, axis=0, ddof=1)
            dz_affect_curves[i_a, i_k] = mean_p / np.maximum(sd_p, EPS)

        for i_c, concept in enumerate(core_concepts):
            u_ctrl = basis.controlled_dirs[concept]  # [L, D]
            u_raw = basis.raw_dirs[concept]          # [L, D]

            # Dimensionless projection in units of stream_rms, plus directional cosine
            proj_ctrl = np.einsum("mld,ld->ml", deltas, u_ctrl) / np.maximum(basis.stream_rms[None, :], EPS)
            proj_raw = np.einsum("mld,ld->ml", deltas, u_raw) / np.maximum(basis.stream_rms[None, :], EPS)
            cos_ctrl = np.einsum("mld,ld->ml", delta_units, u_ctrl)

            sd_ctrl = np.std(proj_ctrl, axis=0, ddof=1)
            sd_raw = np.std(proj_raw, axis=0, ddof=1)
            dz_controlled_curves[i_a, i_c] = np.mean(proj_ctrl, axis=0) / np.maximum(sd_ctrl, EPS)
            dz_raw_curves[i_a, i_c] = np.mean(proj_raw, axis=0) / np.maximum(sd_raw, EPS)
            cos_controlled_curves[i_a, i_c] = np.mean(cos_ctrl, axis=0)

            # Exact inference at the pre-selected primary layer (zero post-hoc layer fishing)
            vals_L = proj_ctrl[:, primary_layer]
            raw_L = proj_raw[:, primary_layer]
            cos_L = cos_ctrl[:, primary_layer]
            p_raw_unadj = exact_or_mc_signflip_p(vals_L, two_sided=True)
            ci_lo, ci_hi = bootstrap_mean_ci(vals_L)
            primary_cells.append({
                "archetype": arch,
                "concept": concept,
                "layer": int(primary_layer),
                "mean_proj_controlled": float(np.mean(vals_L)),
                "ci95_controlled": [ci_lo, ci_hi],
                "cohen_dz_controlled": float(dz_controlled_curves[i_a, i_c, primary_layer]),
                "cohen_dz_raw": float(dz_raw_curves[i_a, i_c, primary_layer]),
                "mean_cosine_controlled": float(np.mean(cos_L)),
                "mean_proj_raw": float(np.mean(raw_L)),
                "p_signflip": p_raw_unadj,
            })

    # Apply Holm across all 48 cells AND within each archetype (8 concepts)
    all_p = np.array([c["p_signflip"] for c in primary_cells], dtype=np.float64)
    holm_global = holm_adjust(all_p)
    for idx, cell in enumerate(primary_cells):
        cell["p_holm_global"] = float(holm_global[idx])

    for arch in archetypes:
        idxs = [i for i, c in enumerate(primary_cells) if c["archetype"] == arch]
        sub_holm = holm_adjust([primary_cells[i]["p_signflip"] for i in idxs])
        for j, i in enumerate(idxs):
            primary_cells[i]["p_holm_within_archetype"] = float(sub_holm[j])

    return {
        "primary_layer": int(primary_layer),
        "archetypes": list(archetypes),
        "core_concepts": list(core_concepts),
        "control_concepts": ctrl_axes,
        "primary_cells": primary_cells,
        "dz_controlled_curves": dz_controlled_curves,
        "dz_raw_curves": dz_raw_curves,
        "cos_controlled_curves": cos_controlled_curves,
        "dz_affect_curves": dz_affect_curves,
    }
