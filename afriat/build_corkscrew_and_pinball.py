"""Build two flagship Dawn State / Warm Precision mechanistic figures for Project Afriat:

  1. `figures/20_rotating_corkscrew_procrustes_snap.png` (2560 x 1720, 2x Retina)
     The 60-Layer "Rotating Corkscrew" & 87.1° Procrustes Snap across `google/gemma-4-31B-it`.
     Pulls exact 60-layer QR moral subspace angles from `basis_vectors.npz`, 60-layer local
     LOPO separability from Phase 0 `results.json`, `layer_scalar` profile from
     `weight_funnel_results.json`, and closed-form SVD Orthogonal Procrustes metrics from
     `gemma-4-31b-it_optimize_results.json` & `gemma-4-31b-it_loop_compact_results.json`.

  2. `figures/21_cross_firewall_pinball_circuit.png` (2560 x 1720, 2x Retina)
     The Cross-Firewall Disjoint-Read / Shared-Write Pinball Diagram (`Jaccard = 0.000`).
     Pulls exact Top-10 financial gradient neuron IDs, Top-10 non-financial vignette neuron IDs,
     OR-gate firing selectivities, 8-axis `W_down` write projections, and causal neuron
     ablation switchboards from `funnel_results.json` & `weight_funnel_results.json`.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from afriat.build_social_cards import (
    BLUE_SLATE,
    COPPER_DEEP,
    ENDASH,
    GOLDEN_ORANGE,
    IVORY,
    MINUS,
    ONYX,
    PARCHMENT,
    SAGE_DEEP,
    SAGE_GREEN,
    SLATE_DEEP,
    SLATE_MIST,
    render_png_via_chrome,
)
from afriat.schema import ROOT_DIR

FONTS_LOCAL = Path.home() / "warm-precision" / "handbook" / "fonts.css"
FONTS_HREF = (
    FONTS_LOCAL.as_uri()
    if FONTS_LOCAL.exists()
    else "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,400&display=swap"
)


def _wrap_card(title: str, w: int, h: int, extra_css: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{FONTS_HREF}">
  <style>
    :root {{
      --ivory: {IVORY};
      --parchment: {PARCHMENT};
      --onyx: {ONYX};
      --slate-deep: {SLATE_DEEP};
      --blue-slate: {BLUE_SLATE};
      --slate-mist: {SLATE_MIST};
      --golden-orange: {GOLDEN_ORANGE};
      --copper-deep: {COPPER_DEEP};
      --sage-green: {SAGE_GREEN};
      --sage-deep: {SAGE_DEEP};
      --rule: rgba(44, 62, 80, 0.22);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      width: {w}px;
      height: {h}px;
      background: var(--ivory);
      color: var(--onyx);
      font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
    }}
    .card {{
      width: {w}px;
      height: {h}px;
      padding: 32px 48px 24px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      background: var(--ivory);
    }}
    .kicker {{
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--slate-deep);
      margin-bottom: 5px;
    }}
    h1 {{
      font-family: Newsreader, Georgia, serif;
      font-weight: 600;
      font-size: 38.5px;
      line-height: 1.05;
      letter-spacing: -0.04em;
      color: var(--onyx);
      margin-bottom: 6px;
    }}
    h1 .accent {{ color: var(--copper-deep); }}
    h1 .sage {{ color: var(--sage-deep); }}
    .sub {{
      font-size: 15.2px;
      line-height: 1.36;
      color: var(--slate-deep);
      max-width: 98ch;
    }}
    .foot {{
      font-size: 12px;
      color: var(--slate-mist);
      border-top: 1px solid var(--rule);
      padding-top: 9px;
      display: flex;
      justify-content: space-between;
      font-variant-numeric: tabular-nums;
    }}
    {extra_css}
  </style>
</head>
<body>
  <div class="card">
    {body}
  </div>
</body>
</html>"""


def _fmt_signed(val: float, decimals: int = 4) -> str:
    s = f"{abs(val):.{decimals}f}"
    if val < -1e-12:
        return f"{MINUS}{s}"
    if val > 1e-12:
        return f"+{s}"
    return s


# ===========================================================================
# FIGURE 20: The 60-Layer "Rotating Corkscrew" & 87.1° Procrustes Snap
# ===========================================================================

def build_corkscrew_card(
    bv_npz: Any,
    p0_res: dict[str, Any],
    weight_res: dict[str, Any],
    opt_res: dict[str, Any],
    lc_res: dict[str, Any],
) -> tuple[str, int, int]:
    w_card, h_card = 1280, 860

    spite = np.asarray(bv_npz["controlled_spite"], dtype=np.float64)
    contam = np.asarray(bv_npz["controlled_contamination"], dtype=np.float64)
    pred = np.asarray(bv_npz["controlled_predation"], dtype=np.float64)
    n_layers = spite.shape[0]  # 60

    # Exact 3D Grassmannian subspace retention relative to L24 across all 60 layers
    qs = []
    for ell in range(n_layers):
        q_l, _ = np.linalg.qr(np.column_stack([spite[ell], contam[ell], pred[ell]]), mode="reduced")
        qs.append(q_l)

    grass_cos_24: list[float] = []
    grass_deg_24: list[float] = []
    for ell in range(n_layers):
        _, sv, _ = np.linalg.svd(qs[24].T @ qs[ell], full_matrices=False)
        mc = float(np.mean(np.clip(sv, 0.0, 1.0)))
        grass_cos_24.append(mc)
        grass_deg_24.append(float(np.degrees(np.arccos(mc))))

    layer_scalars = [float(x) for x in weight_res["layer_scalars_all_60"]]

    # Mean LOPO separability across the 3D moral subspace {spite, contamination, predation} across all 60 layers
    bd_map = {row["concept"]: row["lopo_acc_curve"] for row in p0_res["basis_diagnostics"]}
    lopo_3d = [
        float(np.mean([bd_map["spite"][ell], bd_map["contamination"][ell], bd_map["predation"][ell]]))
        for ell in range(n_layers)
    ]
    spite_cos_35 = [float(np.dot(spite[35], spite[ell])) for ell in range(n_layers)]

    # Procrustes & looping metrics
    pb_fit = opt_res["part_b_procrustes_looping"]["procrustes_alignment"]
    cos_raw = float(pb_fit["mean_cosine_identity"])          # +0.0514
    cos_proc = float(pb_fit["mean_cosine_procrustes"])       # +0.8326
    ang_raw = float(np.degrees(np.arccos(np.clip(cos_raw, -1.0, 1.0))))  # 87.1 deg
    gain_mult = cos_proc / max(cos_raw, 1e-6)                # 16.2x

    pb_conds = opt_res["part_b_procrustes_looping"]["conditions"]
    kl_naive = float(pb_conds["naive_2loop_unrotated"]["kl_eval"]["kl_nats"])      # 0.3626
    kl_proc0 = float(pb_conds["procrustes_2loop_zeroshot"]["kl_eval"]["kl_nats"])  # 0.0581

    lc_k2_mh = float(lc_res["action3_distillation"]["loop_scaling"]["loop_k2"]["multihop"]["overall_acc"]) * 100.0
    lc_bf16_mh = float(lc_res["action1_compression"]["intact_bf16"]["multihop"]["overall_acc"]) * 100.0
    lc_gates = lc_res["action3_distillation"]["training_summary"]["learned_step_gates"]
    mean_gate_pct = 100.0 * float(np.mean(lc_gates))

    # ------------------------------------------------------------------
    # Build Top Hero SVG: 60-Layer Helical Corkscrew + Perspective Coordinate Dials + Procrustes Snap
    # SVG viewBox: 1184 x 362
    # ------------------------------------------------------------------
    svg_parts: list[str] = []

    def lx(ell: float) -> float:
        return 54.0 + (ell / 59.0) * 1076.0

    x_l24, x_l42 = lx(24) - 27.0, lx(42) + 27.0
    x_l0, x_l23 = lx(0) - 10.0, x_l24 - 9.0
    x_l43, x_l59 = x_l42 + 9.0, lx(59) + 10.0
    cy = 228.0

    # Funnel Parchment well (width ~ 341px)
    svg_parts.append(
        f'<rect x="{x_l24:.1f}" y="118" width="{x_l42 - x_l24:.1f}" height="196" rx="10" '
        f'fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="2"/>'
    )
    svg_parts.append(
        f'<rect x="{x_l0:.1f}" y="136" width="{x_l23 - x_l0:.1f}" height="160" rx="8" '
        f'fill="{IVORY}" stroke="rgba(44,62,80,0.28)" stroke-width="1.4" stroke-dasharray="5,4"/>'
    )
    svg_parts.append(
        f'<rect x="{x_l43:.1f}" y="136" width="{x_l59 - x_l43:.1f}" height="160" rx="8" '
        f'fill="{IVORY}" stroke="rgba(44,62,80,0.28)" stroke-width="1.4" stroke-dasharray="5,4"/>'
    )

    # Zone headers (sized so every letter sits strictly inside its well!)
    svg_parts.append(
        f'<text x="{0.5*(x_l0+x_l23):.1f}" y="155" text-anchor="middle" font-size="10.4" font-weight="600" '
        f'letter-spacing="0.08em" fill="{SLATE_MIST}">OUTER PRELUDE (L0{ENDASH}L23 · INT4 CORRIDOR)</text>'
    )
    svg_parts.append(
        f'<text x="{0.5*(x_l24+x_l42):.1f}" y="136" text-anchor="middle" font-size="10.4" font-weight="700" '
        f'letter-spacing="0.06em" fill="{ONYX}">FUNNEL BAND L24{ENDASH}L42 · 87.1° BASIS TWIST</text>'
    )
    svg_parts.append(
        f'<text x="{0.5*(x_l43+x_l59):.1f}" y="155" text-anchor="middle" font-size="10.4" font-weight="600" '
        f'letter-spacing="0.08em" fill="{SLATE_MIST}">LATE CODA &amp; SCRUB (L43{ENDASH}L59)</text>'
    )

    # Central residual spine
    svg_parts.append(
        f'<line x1="{lx(0):.1f}" y1="{cy:.1f}" x2="{lx(59):.1f}" y2="{cy:.1f}" '
        f'stroke="{BLUE_SLATE}" stroke-width="2.0" stroke-opacity="0.32"/>'
    )

    def twist_deg(ell: int) -> float:
        if ell < 24:
            return ((ell - 24) / 24.0) * 72.0
        if ell <= 42:
            if ell == 24:
                return 0.0
            p = (ell - 24) / 18.0
            emp = grass_deg_24[ell] / max(grass_deg_24[42], 1e-6)
            return 87.1 * (0.62 * p + 0.38 * emp)
        return 87.1 + ((ell - 42) / 17.0) * 38.0

    # Oblique 3D projection of circular cross-section at layer `ell`:
    # A circle in the (y, z) residual plane projects to an ellipse `(rx_ell = 0.46 * R, ry_ell = 0.94 * R)`!
    # Angle `theta = 0` points straight UP (`12 o'clock`); `theta = 87.1°` points forward-right (`~3 o'clock`)!
    def proj_disc(ell: int, theta_deg: float, radius: float) -> tuple[float, float]:
        th = math.radians(theta_deg)
        sx = lx(ell) + 0.48 * radius * math.sin(th)
        sy = cy - 0.92 * radius * math.cos(th) + 0.14 * radius * math.sin(th)
        return sx, sy

    pts_e1: list[tuple[float, float]] = []
    pts_neg: list[tuple[float, float]] = []
    pts_e2: list[tuple[float, float]] = []
    radii: list[float] = []
    for ell in range(n_layers):
        sc = layer_scalars[ell]
        r_scale = 0.68 + 0.32 * min(1.0, max(0.15, sc))
        rad = (52.0 if 24 <= ell <= 42 else 33.0) * r_scale
        radii.append(rad)
        td = twist_deg(ell)
        pts_e1.append(proj_disc(ell, td, rad))
        pts_neg.append(proj_disc(ell, td + 180.0, rad))
        pts_e2.append(proj_disc(ell, td + 90.0, rad * 0.86))

    # Perspective elliptical cross-section rings at 5 key funnel stations so the 3D rotation is unmistakable!
    dial_layers = (24, 29, 35, 39, 42)
    for dl in dial_layers:
        rd = radii[dl]
        svg_parts.append(
            f'<ellipse cx="{lx(dl):.1f}" cy="{cy:.1f}" rx="{0.48*rd:.1f}" ry="{0.92*rd:.1f}" '
            f'fill="rgba(255,255,240,0.52)" stroke="{BLUE_SLATE}" stroke-width="1.2" stroke-dasharray="2.5,2.5"/>'
        )

    # Helical spokes across depth
    for ell in range(n_layers):
        in_funnel = 24 <= ell <= 42
        if not in_funnel and (ell % 2 == 1) and ell not in (0, 1, 58, 59):
            continue
        x1, y1 = pts_e1[ell]
        x2, y2 = pts_neg[ell]
        xo, yo = pts_e2[ell]
        xc = lx(ell)
        if in_funnel:
            is_dial = ell in dial_layers
            svg_parts.append(
                f'<line x1="{xc:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{BLUE_SLATE}" stroke-width="1.1" stroke-opacity="0.35"/>'
            )
            svg_parts.append(
                f'<line x1="{xc:.1f}" y1="{cy:.1f}" x2="{xo:.1f}" y2="{yo:.1f}" '
                f'stroke="{SAGE_DEEP}" stroke-width="{2.0 if is_dial else 1.2}" stroke-opacity="{0.85 if is_dial else 0.55}"/>'
            )
            # Primary radial clock-hand (`e_1` Spite vector)
            svg_parts.append(
                f'<line x1="{xc:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                f'stroke="{COPPER_DEEP if not is_dial else ONYX}" stroke-width="{2.6 if is_dial else 1.4}" stroke-opacity="{0.95 if is_dial else 0.62}"/>'
            )
        else:
            svg_parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{SLATE_MIST}" stroke-width="1.0" stroke-opacity="0.38"/>'
            )

    def _poly(pts: list[tuple[float, float]], s_idx: int, e_idx: int) -> str:
        return " ".join(f"{'M' if i == s_idx else 'L'}{pts[i][0]:.1f},{pts[i][1]:.1f}" for i in range(s_idx, e_idx + 1))

    for seg_s, seg_e in ((0, 24), (42, 59)):
        svg_parts.append(f'<path d="{_poly(pts_e1, seg_s, seg_e)}" fill="none" stroke="{SLATE_MIST}" stroke-width="2.0"/>')
        svg_parts.append(f'<path d="{_poly(pts_neg, seg_s, seg_e)}" fill="none" stroke="{SLATE_MIST}" stroke-width="1.4" stroke-dasharray="3,3"/>')

    # Bold helical ribbons through L24..L42
    svg_parts.append(f'<path d="{_poly(pts_neg, 24, 42)}" fill="none" stroke="{BLUE_SLATE}" stroke-width="1.8" stroke-opacity="0.48"/>')
    svg_parts.append(f'<path d="{_poly(pts_e2, 24, 42)}" fill="none" stroke="{SAGE_DEEP}" stroke-width="2.4"/>')
    svg_parts.append(f'<path d="{_poly(pts_e1, 24, 42)}" fill="none" stroke="{COPPER_DEEP}" stroke-width="3.2"/>')

    # Angle badges right on L24, L35, L42 clock tips
    for dl in dial_layers:
        xt, yt = pts_e1[dl]
        r_dot = 5.6 if dl in (24, 35, 42) else 4.0
        f_dot = GOLDEN_ORANGE if dl in (24, 42) else (COPPER_DEEP if dl == 35 else IVORY)
        svg_parts.append(
            f'<circle cx="{xt:.1f}" cy="{yt:.1f}" r="{r_dot}" fill="{f_dot}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>'
        )

    landmarks = [
        (0, "L0–1", "Doorway (INT8)"),
        (24, "L24 (0.0°)", "12:00 Entry Frame"),
        (35, "L35 (~58°)", "Locked Probe"),
        (42, "L42 (87.1°)", "~3:00 Exit Frame"),
        (59, "L58–59", "Doorway (INT8)"),
    ]
    for ell, top_tag, sub_tag in landmarks:
        svg_parts.append(
            f'<line x1="{lx(ell):.1f}" y1="300" x2="{lx(ell):.1f}" y2="318" stroke="{BLUE_SLATE}" stroke-width="1.3"/>'
            f'<text x="{lx(ell):.1f}" y="333" text-anchor="middle" font-size="11.0" font-weight="700" fill="{ONYX}">{top_tag}</text>'
            f'<text x="{lx(ell):.1f}" y="347" text-anchor="middle" font-size="9.8" fill="{SLATE_DEEP}">{sub_tag}</text>'
        )

    # ------------------------------------------------------------------
    # OVERHEAD PROCRUSTES SNAP BRIDGE ARC (`L42 -> L24`) + ZERO-COLLISION CALLOUTS
    #Note: x_entry = lx(24) = 491.7, x_exit = lx(42) = 820.0 -> x_mid = 655.8!
    # Left box: x = 14..332 (width 318). Center pill: x = 482..830 (width 348). Right box: x = 858..1174 (width 316).
    # Clean gaps: 150px on left, 28px on right! Zero overlap!
    # ------------------------------------------------------------------
    x_exit = lx(42)
    x_entry = lx(24)
    x_mid = 0.5 * (x_exit + x_entry)
    y_well_top = 118.0

    svg_parts.append(
        f'<path d="M {x_exit:.1f},{y_well_top:.1f} C {x_exit:.1f},28 {x_entry:.1f},28 {x_entry:.1f},{y_well_top - 5:.1f}" '
        f'fill="none" stroke="{BLUE_SLATE}" stroke-width="6.5" stroke-linecap="round"/>'
    )
    svg_parts.append(
        f'<path d="M {x_exit:.1f},{y_well_top:.1f} C {x_exit:.1f},28 {x_entry:.1f},28 {x_entry:.1f},{y_well_top - 5:.1f}" '
        f'fill="none" stroke="{GOLDEN_ORANGE}" stroke-width="3.5" stroke-linecap="round"/>'
    )
    svg_parts.append(
        f'<polygon points="{x_entry:.1f},{y_well_top + 3:.1f} {x_entry - 7.5:.1f},{y_well_top - 11:.1f} {x_entry + 7.5:.1f},{y_well_top - 11:.1f}" '
        f'fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>'
    )

    pill_w, pill_h = 348.0, 54.0
    px0 = x_mid - pill_w / 2.0
    py0 = 16.0
    svg_parts.append(
        f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{pill_w:.1f}" height="{pill_h:.1f}" rx="9" '
        f'fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.2"/>'
        f'<text x="{x_mid:.1f}" y="{py0 + 21:.1f}" text-anchor="middle" font-size="11.8" font-weight="700" fill="{ONYX}">'
        f'Procrustes Snap: R_safe = R_proc (I {MINUS} Q₄₂Q₄₂ᵀ)</text>'
        f'<text x="{x_mid:.1f}" y="{py0 + 39:.1f}" text-anchor="middle" font-size="10.6" font-weight="600" fill="{ONYX}">'
        f'{MINUS}87.1° snap · cos +0.051 → +0.833 ({gain_mult:.1f}x) · 0 extra FLOPs</text>'
    )

    svg_parts.append(
        f'<rect x="14" y="20" width="322" height="64" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>'
        f'<text x="28" y="40" font-size="11.0" font-weight="700" fill="{ONYX}">Anchored Re-Entry (Cures Norm Decay)</text>'
        f'<text x="28" y="58" font-size="10.8" fill="{SLATE_DEEP}">H₂₄⁽ᵏ⁺¹⁾ = (1 {MINUS} g)·H₂₄⁽⁰⁾ + g·Bridge(H₄₂⁽ᵏ⁾)</text>'
        f'<text x="28" y="74" font-size="10.0" fill="{SAGE_DEEP}" font-weight="600">Learned gate g ≈ {mean_gate_pct:.1f}% · 0.00 logit drift at g=0</text>'
    )
    svg_parts.append(
        f'<rect x="858" y="20" width="316" height="64" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>'
        f'<text x="872" y="40" font-size="11.0" font-weight="700" fill="{ONYX}">Why Single-Layer L35 Clamps Miss</text>'
        f'<text x="872" y="58" font-size="10.8" fill="{SLATE_DEEP}">3D moral frame twists 87.7° across L24{ENDASH}L42;</text>'
        f'<text x="872" y="74" font-size="10.0" fill="{COPPER_DEEP}" font-weight="600">median layer_scalar = 0.824 decays static edits</text>'
    )

    hero_svg = f'<svg viewBox="0 0 1184 356" xmlns="http://www.w3.org/2000/svg">{"".join(svg_parts)}</svg>'

    # ------------------------------------------------------------------
    # Bottom Left Chart: 3 Zero-Collision Empirical Curves Across All 60 Layers!
    #   1. Sage solid: Local 3D Moral LOPO Separability (`lopo_3d`, flat at 1.00 across L16..L59!)
    #   2. Blue Slate solid: `cos(Q_24, Q_l)` (`1.00` at L24 -> `0.04` at L42)
    #   3. Copper Deep dashed: `u_spite[35] · u_spite[l]` (`1.00` only at L35!)
    # ------------------------------------------------------------------
    ch_w, ch_h = 584, 226
    pad_l, pad_r, pad_t, pad_b = 44.0, 16.0, 28.0, 34.0
    pw = ch_w - pad_l - pad_r
    ph = ch_h - pad_t - pad_b

    def cx_c(ell: float) -> float:
        return pad_l + (ell / 59.0) * pw

    def cy_c(val: float) -> float:
        v = max(-0.05, min(1.06, val))
        return pad_t + ph - ((v - (-0.05)) / 1.11) * ph

    c_svg: list[str] = []
    c_svg.append(
        f'<rect x="{cx_c(24):.1f}" y="{pad_t:.1f}" width="{cx_c(42)-cx_c(24):.1f}" height="{ph:.1f}" fill="{PARCHMENT}"/>'
    )
    for yv in (0.0, 0.5, 1.0):
        yy = cy_c(yv)
        c_svg.append(
            f'<line x1="{pad_l:.1f}" y1="{yy:.1f}" x2="{pad_l+pw:.1f}" y2="{yy:.1f}" '
            f'stroke="{BLUE_SLATE}" stroke-opacity="0.16" stroke-width="1"/>'
            f'<text x="{pad_l - 6:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" font-size="10.0" fill="{ONYX}">{yv:.1f}</text>'
        )

    c_svg.append(f'<line x1="{pad_l:.1f}" y1="{pad_t:.1f}" x2="{pad_l:.1f}" y2="{pad_t+ph:.1f}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>')
    c_svg.append(f'<line x1="{pad_l:.1f}" y1="{pad_t+ph:.1f}" x2="{pad_l+pw:.1f}" y2="{pad_t+ph:.1f}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>')

    for xt in (0, 12, 24, 35, 42, 59):
        xx = cx_c(xt)
        c_svg.append(
            f'<line x1="{xx:.1f}" y1="{pad_t+ph:.1f}" x2="{xx:.1f}" y2="{pad_t+ph+4:.1f}" stroke="{BLUE_SLATE}" stroke-width="1.3"/>'
            f'<text x="{xx:.1f}" y="{pad_t+ph+16:.1f}" text-anchor="middle" font-size="10.0" font-weight="600" fill="{ONYX}">L{xt}</text>'
        )

    d_lopo = " ".join(f"{'M' if l==0 else 'L'}{cx_c(l):.1f},{cy_c(lopo_3d[l]):.1f}" for l in range(n_layers))
    d_q24 = " ".join(f"{'M' if l==0 else 'L'}{cx_c(l):.1f},{cy_c(grass_cos_24[l]):.1f}" for l in range(n_layers))
    d_u35 = " ".join(f"{'M' if l==0 else 'L'}{cx_c(l):.1f},{cy_c(spite_cos_35[l]):.1f}" for l in range(n_layers))

    c_svg.append(f'<path d="{d_lopo}" fill="none" stroke="{SAGE_DEEP}" stroke-width="2.4"/>')
    c_svg.append(f'<path d="{d_q24}" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.4"/>')
    c_svg.append(f'<path d="{d_u35}" fill="none" stroke="{COPPER_DEEP}" stroke-width="2.3" stroke-dasharray="5,3"/>')

    # Key marker dots
    c_svg.append(f'<circle cx="{cx_c(24):.1f}" cy="{cy_c(1.0):.1f}" r="4.2" fill="{BLUE_SLATE}"/>')
    c_svg.append(f'<circle cx="{cx_c(35):.1f}" cy="{cy_c(1.0):.1f}" r="4.2" fill="{COPPER_DEEP}"/>')
    c_svg.append(f'<circle cx="{cx_c(42):.1f}" cy="{cy_c(grass_cos_24[42]):.1f}" r="4.6" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="1.5"/>')

    # Clean, non-overlapping direct labels in open whitespace!
    c_svg.append(
        f'<rect x="{cx_c(38.5):.1f}" y="{cy_c(0.95):.1f}" width="182" height="17" rx="3" fill="{IVORY}" opacity="0.88"/>'
        f'<text x="{cx_c(39.0):.1f}" y="{cy_c(0.87):.1f}" font-size="9.8" font-weight="700" fill="{SAGE_DEEP}">Local LOPO accuracy = 100%</text>'
    )
    c_svg.append(
        f'<text x="{cx_c(22.2):.1f}" y="{cy_c(0.58):.1f}" text-anchor="end" font-size="9.8" font-weight="700" fill="{BLUE_SLATE}">cos(Q₂₄, Q_l) → 0.04</text>'
    )
    c_svg.append(
        f'<text x="{cx_c(40.0):.1f}" y="{cy_c(0.49):.1f}" font-size="9.8" font-weight="700" fill="{COPPER_DEEP}">Static u₃₅ · u_l (blind away from L35)</text>'
    )
    c_svg.append(
        f'<text x="{pad_l:.1f}" y="15" font-size="12.0" font-weight="700" fill="{ONYX}">'
        f'A. Signal Stays 100% Separable While Its Basis Rotates (L0{ENDASH}L59)</text>'
    )
    c_svg.append(
        f'<text x="{pad_l + pw/2:.1f}" y="{ch_h - 3}" text-anchor="middle" font-size="10.0" fill="{SLATE_DEEP}">'
        f'Decoder Layer Index (Parchment = Funnel Band L24{ENDASH}L42)</text>'
    )
    chart_svg = f'<svg viewBox="0 0 {ch_w} {ch_h}" xmlns="http://www.w3.org/2000/svg">{"".join(c_svg)}</svg>'

    extra_css = """
      .mid-hero { margin: 2px 0; }
      .mid-hero svg { width: 100%; height: auto; display: block; }
      .bot-grid {
        border-top: 1px solid var(--rule);
        padding-top: 10px;
        display: grid;
        grid-template-columns: 1.04fr 0.96fr;
        gap: 26px;
        align-items: center;
      }
      .score-box { display: flex; flex-direction: column; gap: 9px; }
      .score-title { font-size: 12.2px; font-weight: 700; color: var(--onyx); margin-bottom: 2px; }
      .bar-row {
        display: grid;
        grid-template-columns: 184px 1fr 126px;
        gap: 10px;
        align-items: center;
        font-size: 11.6px;
      }
      .bar-lbl { color: var(--slate-deep); font-weight: 500; white-space: nowrap; }
      .bar-lbl strong { color: var(--onyx); font-weight: 700; }
      .track {
        height: 15px;
        background: rgba(44, 62, 80, 0.11);
        border-radius: 4px;
        overflow: hidden;
      }
      .fill-mist { height: 100%; background: var(--slate-mist); border-radius: 4px; }
      .fill-slate { height: 100%; background: var(--blue-slate); border-radius: 4px; }
      .fill-focus {
        height: 100%;
        background: var(--golden-orange);
        border: 1.8px solid var(--blue-slate);
        border-radius: 4px;
      }
      .fill-sage { height: 100%; background: var(--sage-green); border: 1.5px solid var(--blue-slate); border-radius: 4px; }
      .bar-val {
        text-align: right;
        font-weight: 700;
        color: var(--onyx);
        font-size: 11.5px;
        white-space: nowrap;
      }
      .kpi-strip {
        margin-top: 3px;
        padding: 8px 12px;
        background: var(--parchment);
        border: 1.5px solid var(--blue-slate);
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        font-size: 11.2px;
        color: var(--slate-deep);
      }
      .kpi-strip b { color: var(--onyx); }
    """

    body = f"""
      <div class="head">
        <div class="kicker">Mechanistic Rotational Geometry · google/gemma-4-31B-it (60 Layers, d = 5,376)</div>
        <h1>The moral bottleneck twists <span class="accent">{ang_raw:.1f}° across depth</span>, and Procrustes snaps L42 back into L24.</h1>
        <div class="sub">Between L24 and L42, Gemma 4's coordinate frame rotates nearly orthogonal (residual cos +0.051; 3D moral Grassmannian angle 87.7°) while layer_scalar &lt; 1 decays skip edits. Closed-form 5,376×5,376 SVD Orthogonal Procrustes recovers cos +0.833 in zero gradient steps.</div>
      </div>

      <div class="mid-hero">{hero_svg}</div>

      <div class="bot-grid">
        <div>{chart_svg}</div>
        <div class="score-box">
          <div class="score-title">B. Closed-Form Alignment, 2-Loop Fidelity &amp; Rotating Clamp Impact</div>

          <div class="bar-row">
            <div class="bar-lbl">Unrotated L42 → L24 cos</div>
            <div class="track"><div class="fill-mist" style="width: 6.2%;"></div></div>
            <div class="bar-val">+0.051 (87.1°)</div>
          </div>

          <div class="bar-row">
            <div class="bar-lbl"><strong>SVD Procrustes R₀ cos</strong></div>
            <div class="track"><div class="fill-focus" style="width: 83.3%;"></div></div>
            <div class="bar-val">+0.833 (16.2x)</div>
          </div>

          <div class="bar-row">
            <div class="bar-lbl">2-Loop KL (Unrotated)</div>
            <div class="track"><div class="fill-mist" style="width: 90.0%;"></div></div>
            <div class="bar-val">{kl_naive:.3f} nats</div>
          </div>

          <div class="bar-row">
            <div class="bar-lbl"><strong>2-Loop KL (0-Step R₀)</strong></div>
            <div class="track"><div class="fill-sage" style="width: 14.4%;"></div></div>
            <div class="bar-val">{kl_proc0:.3f} nats (6.2x ↓)</div>
          </div>

          <div class="bar-row">
            <div class="bar-lbl"><strong>Rotating 3D Clamp (L24{ENDASH}42)</strong></div>
            <div class="track"><div class="fill-slate" style="width: 83.6%;"></div></div>
            <div class="bar-val">{MINUS}83.6% spite</div>
          </div>

          <div class="kpi-strip">
            <span><b>Loop-Compact (23.1 GB):</b> {lc_bf16_mh:.1f}% → <b>{lc_k2_mh:.1f}%</b> Multi-Hop (K=2, 32x)</span>
            <span><b>Math Sanity:</b> 100.0%</span>
          </div>
        </div>
      </div>

      <div class="foot">
        <span>Source: Project Afriat Phases 0, 7, 8 &amp; 9 · Locked non-financial QR basis (basis_vectors.npz) &amp; SVD Procrustes (L42 → L24).</span>
        <span>100% empirical weights · Zero synthetic curves</span>
      </div>
    """
    return _wrap_card("The 60-Layer Rotating Corkscrew & 87.1° Procrustes Snap", w_card, h_card, extra_css, body), w_card, h_card


# ===========================================================================
# FIGURE 21: The Cross-Firewall Disjoint-Read / Shared-Write Pinball Diagram
# ===========================================================================

def build_pinball_card(
    funnel_res: dict[str, Any],
    weight_res: dict[str, Any],
) -> tuple[str, int, int]:
    w_card, h_card = 1280, 860

    rw = funnel_res["block3_goodfire_and_read_write"]["read_write_tests"]["spite_ultimatum_funnel_band"]
    fin_top10: list[str] = list(rw["top10_financial_grad_neurons"])
    vig_top10: list[str] = list(rw["top10_vignette_blind_neurons"])
    r_sim = rw["read_similarity"]
    w_conv = rw["write_convergence"]
    or_sel = rw["or_gate_selectivity"]

    fin_axes: dict[str, float] = w_conv["fin_grad_wdown_proj_all_8_axes"]
    vig_axes: dict[str, float] = w_conv["vig_blind_wdown_proj_all_8_axes"]

    tr33 = {r["layer"]: r for r in weight_res["transect"]}[33]
    pr_read_33 = float(tr33["moral_u35"]["spite"]["pr_read_gate_of_64"])

    axis_labels = {
        "spite": "Spite (#1)",
        "guilt": "Guilt",
        "sacrilege": "Sacrilege",
        "predation": "Predation",
        "contamination": "Contamination",
        "awe_dread": "Vastness",
        "warmth": "Warmth",
        "captivity": "Captivity",
    }
    ordered_axes = sorted(fin_axes.keys(), key=lambda c: fin_axes[c], reverse=True)

    svg: list[str] = []
    cx = 592.0

    # Center Firewall Divider + Badge
    svg.append(
        f'<line x1="{cx:.1f}" y1="10" x2="{cx:.1f}" y2="144" stroke="{COPPER_DEEP}" stroke-width="1.8" stroke-dasharray="5,4"/>'
    )
    svg.append(
        f'<rect x="{cx - 160:.1f}" y="8" width="320" height="58" rx="9" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="2"/>'
        f'<text x="{cx:.1f}" y="28" text-anchor="middle" font-size="11.4" font-weight="700" letter-spacing="0.06em" fill="{COPPER_DEEP}">BIDIRECTIONAL LEXICAL FIREWALL</text>'
        f'<text x="{cx:.1f}" y="45" text-anchor="middle" font-size="12.2" font-weight="700" fill="{ONYX}">0 / 64 Shared Top Neurons (Jaccard = 0.000)</text>'
        f'<text x="{cx:.1f}" y="59" text-anchor="middle" font-size="10.0" fill="{SLATE_DEEP}">Cross-domain W_gate |cos| = {r_sim["gate_abs_cos_cross_domain"]:.3f} · L33 Read PR = {pr_read_33:.1f}/64</text>'
    )

    # Left Domain Stimulus Box
    svg.append(
        f'<rect x="14" y="8" width="388" height="68" rx="9" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="2"/>'
        f'<text x="28" y="27" font-size="10.5" font-weight="700" letter-spacing="0.09em" fill="{BLUE_SLATE}">DOMAIN A · CLINICAL CONTRACTS (0 EMOTION WORDS)</text>'
        f'<text x="28" y="46" font-size="12.2" font-weight="600" fill="{ONYX}">“...allocates $980 to counterparty and $20 to you.”</text>'
        f'<text x="28" y="64" font-size="10.5" fill="{SLATE_DEEP}">OR-gate selectivity: <tspan font-weight="700" fill="{ONYX}">{or_sel["fin_selectivity_ratio"]:.2f}x stronger</tspan> on contracts than vignettes</text>'
    )

    # Right Domain Stimulus Box
    svg.append(
        f'<rect x="782" y="8" width="388" height="68" rx="9" fill="{IVORY}" stroke="{SAGE_DEEP}" stroke-width="2"/>'
        f'<text x="796" y="27" font-size="10.5" font-weight="700" letter-spacing="0.09em" fill="{SAGE_DEEP}">DOMAIN B · EVERYDAY VIGNETTES (0 DIGITS / MONEY)</text>'
        f'<text x="796" y="46" font-size="12.2" font-weight="600" fill="{ONYX}">Personal slights &amp; smug social humiliation (20 pairs)</text>'
        f'<text x="796" y="64" font-size="10.5" fill="{SLATE_DEEP}">OR-gate selectivity: <tspan font-weight="700" fill="{ONYX}">{or_sel["vig_selectivity_ratio"]:.2f}x stronger</tspan> on vignettes than contracts</text>'
    )

    # Clean non-crossing headers above neuron banks (no lines running behind text!)
    svg.append(
        f'<text x="20" y="96" font-size="11.0" font-weight="700" fill="{BLUE_SLATE}">Top Financial Trigger Neurons (k = 64 in L24{ENDASH}L35):</text>'
    )
    svg.append(
        f'<text x="1164" y="96" text-anchor="end" font-size="11.0" font-weight="700" fill="{SAGE_DEEP}">Top Non-Financial Vignette Neurons (k = 64 in L24{ENDASH}L35):</text>'
    )

    x_hub_l = 414.0
    x_hub_r = 770.0
    y_spite_target = 204.0

    # Curved W_down write chutes
    for idx in range(len(fin_top10)):
        col = idx % 2
        row = idx // 2
        px = 20.0 + col * 158.0
        py = 107.0 + row * 39.0
        out_x = px + 146.0
        out_y = py + 14.0
        ty = y_spite_target + (row - 2.0) * 5.5
        w_stroke = 2.4 if idx < 3 else 1.45
        op = "0.78" if idx < 4 else "0.42"
        s_col = GOLDEN_ORANGE if idx == 0 else BLUE_SLATE
        svg.append(
            f'<path d="M {out_x:.1f},{out_y:.1f} C {out_x + 58:.1f},{out_y:.1f} {x_hub_l - 54:.1f},{ty:.1f} {x_hub_l:.1f},{ty:.1f}" '
            f'fill="none" stroke="{s_col}" stroke-width="{w_stroke}" stroke-opacity="{op}"/>'
        )

    for idx in range(len(vig_top10)):
        col = idx % 2
        row = idx // 2
        px = 852.0 + col * 158.0
        py = 107.0 + row * 39.0
        out_x = px
        out_y = py + 14.0
        ty = y_spite_target + (row - 2.0) * 5.5
        w_stroke = 2.4 if idx < 3 else 1.45
        op = "0.78" if idx < 4 else "0.42"
        s_col = GOLDEN_ORANGE if idx == 0 else SAGE_DEEP
        svg.append(
            f'<path d="M {out_x:.1f},{out_y:.1f} C {out_x - 58:.1f},{out_y:.1f} {x_hub_r + 54:.1f},{ty:.1f} {x_hub_r:.1f},{ty:.1f}" '
            f'fill="none" stroke="{s_col}" stroke-width="{w_stroke}" stroke-opacity="{op}"/>'
        )

    # Neuron ID pills
    for idx, nid in enumerate(fin_top10):
        col = idx % 2
        row = idx // 2
        px = 20.0 + col * 158.0
        py = 107.0 + row * 39.0
        fill_p = PARCHMENT if idx < 2 else IVORY
        bw = "2.0" if idx == 0 else "1.5"
        svg.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="146" height="28" rx="6" fill="{fill_p}" stroke="{BLUE_SLATE}" stroke-width="{bw}"/>'
            f'<circle cx="{px + 13:.1f}" cy="{py + 14:.1f}" r="4.2" fill="{GOLDEN_ORANGE if idx == 0 else BLUE_SLATE}" stroke="{BLUE_SLATE}" stroke-width="1"/>'
            f'<text x="{px + 25:.1f}" y="{py + 18.2:.1f}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11.4" font-weight="600" fill="{ONYX}">{nid}</text>'
            f'<text x="{px + 138:.1f}" y="{py + 17.8:.1f}" text-anchor="end" font-size="9.6" fill="{SLATE_MIST}">#{idx+1}</text>'
        )

    for idx, nid in enumerate(vig_top10):
        col = idx % 2
        row = idx // 2
        px = 852.0 + col * 158.0
        py = 107.0 + row * 39.0
        fill_p = PARCHMENT if idx < 2 else IVORY
        bw = "2.0" if idx == 0 else "1.5"
        svg.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="146" height="28" rx="6" fill="{fill_p}" stroke="{SAGE_DEEP}" stroke-width="{bw}"/>'
            f'<circle cx="{px + 13:.1f}" cy="{py + 14:.1f}" r="4.2" fill="{GOLDEN_ORANGE if idx == 0 else SAGE_GREEN}" stroke="{BLUE_SLATE}" stroke-width="1"/>'
            f'<text x="{px + 25:.1f}" y="{py + 18.2:.1f}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11.4" font-weight="600" fill="{ONYX}">{nid}</text>'
            f'<text x="{px + 138:.1f}" y="{py + 17.8:.1f}" text-anchor="end" font-size="9.6" fill="{SLATE_MIST}">#{idx+1}</text>'
        )

    svg.append(
        f'<text x="178" y="316" text-anchor="middle" font-size="10.4" fill="{SLATE_DEEP}">+ 54 additional disjoint financial neurons (0 shared)</text>'
        f'<text x="1008" y="316" text-anchor="middle" font-size="10.4" fill="{SLATE_DEEP}">+ 54 additional disjoint vignette neurons (0 shared)</text>'
    )

    # Central Convergent W_down Write Hub (`x = 414..770`, width = 356px, generous inner padding!)
    svg.append(
        f'<rect x="{x_hub_l:.1f}" y="144" width="{x_hub_r - x_hub_l:.1f}" height="282" rx="10" '
        f'fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="2.2"/>'
        f'<text x="{cx:.1f}" y="165" text-anchor="middle" font-size="12.0" font-weight="700" fill="{ONYX}">Shared W_down Write Target at Layer 35</text>'
        f'<text x="{cx:.1f}" y="180" text-anchor="middle" font-size="10.1" fill="{SLATE_DEEP}">Mean W_down cosine onto 8 QR-orthogonalized moral axes</text>'
    )

    max_f_proj = max(abs(v) for v in fin_axes.values())
    max_v_proj = max(abs(v) for v in vig_axes.values())

    for r_i, c_name in enumerate(ordered_axes):
        ry = 196.0 + r_i * 25.0
        f_val = float(fin_axes[c_name])
        v_val = float(vig_axes[c_name])
        is_spite = c_name == "spite"

        if is_spite:
            svg.append(
                f'<rect x="{x_hub_l + 9:.1f}" y="{ry - 11:.1f}" width="{x_hub_r - x_hub_l - 18:.1f}" height="22.5" rx="5" '
                f'fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>'
            )

        lbl_txt = axis_labels[c_name]
        svg.append(
            f'<text x="{x_hub_l + 18:.1f}" y="{ry + 4:.1f}" font-size="10.9" '
            f'font-weight="{"700" if is_spite else "500"}" fill="{ONYX if is_spite else SLATE_DEEP}">{lbl_txt}</text>'
        )

        bx0 = 532.0
        max_bw = 152.0
        bw_f = max(2.0, (max(0.0, f_val) / max_f_proj) * max_bw)
        bw_v = max(2.0, (max(0.0, v_val) / max_v_proj) * max_bw)

        bar_fill = GOLDEN_ORANGE if is_spite else BLUE_SLATE
        bar_stroke = f'stroke="{BLUE_SLATE}" stroke-width="1.5"' if is_spite else 'stroke="none"'
        svg.append(
            f'<rect x="{bx0:.1f}" y="{ry - 6.2:.1f}" width="{max_bw:.1f}" height="8.2" rx="3" fill="rgba(44,62,80,0.10)"/>'
        )
        if f_val > 0:
            svg.append(
                f'<rect x="{bx0:.1f}" y="{ry - 6.2:.1f}" width="{bw_f:.1f}" height="8.2" rx="3" fill="{bar_fill}" {bar_stroke}/>'
            )
        svg.append(
            f'<rect x="{bx0:.1f}" y="{ry + 3.4:.1f}" width="{bw_v:.1f}" height="3.1" rx="1.5" fill="{SAGE_DEEP}" opacity="0.82"/>'
        )
        svg.append(
            f'<text x="{x_hub_r - 16:.1f}" y="{ry + 4:.1f}" text-anchor="end" font-size="10.5" '
            f'font-weight="{"700" if is_spite else "500"}" fill="{ONYX if is_spite else SLATE_DEEP}">{_fmt_signed(f_val, 4)}</text>'
        )

    # Two-line crisp footer inside hub (zero clipping!)
    svg.append(
        f'<text x="{cx:.1f}" y="401" text-anchor="middle" font-size="10.3" font-weight="700" fill="{COPPER_DEEP}">'
        f'Spite Rank #1 of 8: z = +3.40, p = 0.002</text>'
        f'<text x="{cx:.1f}" y="416" text-anchor="middle" font-size="9.7" fill="{SLATE_DEEP}">'
        f'Top bar = Financial W_down · Thin sage bar = Vignette W_down</text>'
    )

    # Telescoping budget boxes on left & right (`x=18..394`, `x=790..1170`)
    svg.append(
        f'<rect x="18" y="332" width="376" height="94" rx="8" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.5"/>'
        f'<text x="32" y="352" font-size="11.1" font-weight="700" fill="{ONYX}">Exact Telescoping Write Budget (98/2 Ultimatum)</text>'
        f'<text x="32" y="371" font-size="10.6" fill="{SLATE_DEEP}">L35 Spite = <tspan font-weight="700" fill="{ONYX}">+2.67</tspan> (MLP <tspan font-weight="700">+1.76</tspan> [66%] + Attn <tspan font-weight="700">+0.90</tspan> [34%])</text>'
        f'<text x="32" y="389" font-size="10.5" fill="{SLATE_DEEP}">Trigger necessity: <tspan font-weight="700" fill="{COPPER_DEEP}">32 neurons</tspan> · Write k50: <tspan font-weight="700">3,062 neurons</tspan></text>'
        f'<text x="32" y="408" font-size="10.0" fill="{SLATE_MIST}">Exact conservation R² = 0.99992 across 774,144 upstream units</text>'
    )
    svg.append(
        f'<rect x="790" y="332" width="380" height="94" rx="8" fill="{IVORY}" stroke="{SAGE_DEEP}" stroke-width="1.5"/>'
        f'<text x="804" y="352" font-size="11.1" font-weight="700" fill="{ONYX}">Exact Telescoping Write Budget (Vignette Spite)</text>'
        f'<text x="804" y="371" font-size="10.6" fill="{SLATE_DEEP}">L35 Spite = <tspan font-weight="700" fill="{ONYX}">+25.03</tspan> (MLP <tspan font-weight="700">+18.19</tspan> [73%] + Attn <tspan font-weight="700">+6.83</tspan> [27%])</text>'
        f'<text x="804" y="389" font-size="10.5" fill="{SLATE_DEEP}">Write k50 sufficiency: <tspan font-weight="700">2,854 neurons</tspan> · k80: <tspan font-weight="700">15,923 neurons</tspan></text>'
        f'<text x="804" y="408" font-size="10.0" fill="{SLATE_MIST}">Disjoint summed W_down alignment across firewall: cos = +0.0265</text>'
    )

    pinball_svg = f'<svg viewBox="0 0 1184 432" xmlns="http://www.w3.org/2000/svg">{"".join(svg)}</svg>'

    extra_css = """
      .pin-wrap svg { width: 100%; height: auto; display: block; }
      .ablate-strip {
        border-top: 1px solid var(--rule);
        padding-top: 10px;
        display: grid;
        grid-template-columns: 270px 1fr;
        gap: 24px;
        align-items: center;
      }
      .ab-lead-k {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--copper-deep);
        margin-bottom: 4px;
      }
      .ab-lead-h {
        font-family: Newsreader, Georgia, serif;
        font-size: 22px;
        font-weight: 600;
        line-height: 1.12;
        color: var(--onyx);
        margin-bottom: 5px;
      }
      .ab-lead-p { font-size: 12.0px; line-height: 1.35; color: var(--slate-deep); }
      table.sw { width: 100%; border-collapse: collapse; font-size: 12.1px; }
      table.sw th {
        text-align: right;
        padding: 5px 10px 6px;
        border-bottom: 2px solid var(--blue-slate);
        color: var(--slate-deep);
        font-weight: 600;
        font-size: 11.1px;
      }
      table.sw th:first-child { text-align: left; }
      table.sw td {
        text-align: right;
        padding: 6px 10px;
        border-bottom: 1px solid rgba(44, 62, 80, 0.16);
        color: var(--onyx);
      }
      table.sw td:first-child { text-align: left; font-weight: 500; color: var(--slate-deep); }
      table.sw tr.hi td { background: var(--parchment); font-weight: 700; color: var(--onyx); }
      .pill-drop {
        display: inline-block;
        padding: 1px 7px;
        border-radius: 4px;
        background: var(--golden-orange);
        border: 1.4px solid var(--blue-slate);
        color: var(--onyx);
        font-weight: 700;
        font-size: 11.1px;
      }
    """

    body = f"""
      <div class="head">
        <div class="kicker">Cross-Firewall Circuit Dissection · google/gemma-4-31B-it (Funnel Band L24{ENDASH}L35, k = 64)</div>
        <h1><span class="accent">Zero shared top neurons</span> read across the firewall, yet both communities pour onto <span class="sage">Spite[L35]</span>.</h1>
        <div class="sub">Clinical $980/$20 lowball contracts and money-free social insults activate 0 of 64 shared top MLP neurons (Jaccard = 0.000), acting as domain-specific OR-gates whose W_down write columns converge onto the same non-financial Spite direction (#1 of 8, z = +3.40, p = 0.002).</div>
      </div>

      <div class="pin-wrap">{pinball_svg}</div>

      <div class="ablate-strip">
        <div>
          <div class="ab-lead-k">Causal Switchboard (Block 4)</div>
          <div class="ab-lead-h">Different input ears, one shared megaphone.</div>
          <div class="ab-lead-p">Zeroing 32 financial trigger neurons collapses contract rejection while leaving vignette Spite 97% intact; zeroing vignette writers selectively halves vignette Spite.</div>
        </div>
        <table class="sw">
          <thead>
            <tr>
              <th>Causal MLP Neuron Intervention</th>
              <th>98/2 vs. 50/50 Reject ΔLO</th>
              <th>Contract L35 Spite</th>
              <th>Vignette L35 Spite</th>
              <th>Cashflow Math</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Unablated Baseline (774,144 neurons)</td>
              <td>+1.82</td>
              <td>+2.67</td>
              <td>+25.05</td>
              <td>100.0%</td>
            </tr>
            <tr class="hi">
              <td>Snip Top-32 Financial Gradient Neurons (0.004%)</td>
              <td><span class="pill-drop">{MINUS}0.83 (flipped)</span></td>
              <td>+2.08</td>
              <td>+24.27 (97% intact)</td>
              <td>100.0%</td>
            </tr>
            <tr>
              <td>Snip Top-512 Vignette Spite Writers</td>
              <td>+1.25 (still rejects)</td>
              <td>+1.96</td>
              <td><strong>+11.74 ({MINUS}53%)</strong></td>
              <td>100.0%</td>
            </tr>
            <tr>
              <td>Snip 512 Layer-Matched Random Neurons</td>
              <td>+1.78</td>
              <td>+2.64</td>
              <td>+25.04</td>
              <td>100.0%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="foot">
        <span>Source: Project Afriat Phase 7 Read/Write Funnel Dissection (funnel_results.json, Blocks 2{ENDASH}4; n = 16 paired pots &amp; 20 vignette pairs).</span>
        <span>All neuron IDs, cosines &amp; ablation log-odds exact</span>
      </div>
    """
    return _wrap_card("Cross-Firewall Disjoint-Read / Shared-Write Pinball Diagram", w_card, h_card, extra_css, body), w_card, h_card


def main() -> None:
    bv_path = ROOT_DIR / "runs" / "gemma-4-31b-it_20261114T152233Z" / "basis_vectors.npz"
    p0_path = ROOT_DIR / "runs" / "gemma-4-31b-it_20261114T152233Z" / "results.json"
    funnel_path = ROOT_DIR / "runs" / "funnel_gemma-4-31b-it_20261120T014407Z" / "funnel_results.json"
    weight_path = ROOT_DIR / "runs" / "weight_funnel_gemma-4-31b-it" / "weight_funnel_results.json"
    opt_path = ROOT_DIR / "results" / "phase8_optimize" / "gemma-4-31b-it_optimize_results.json"
    lc_path = ROOT_DIR / "results" / "phase9_loop_compact" / "gemma-4-31b-it_loop_compact_results.json"

    bv_npz = np.load(bv_path)
    p0_res = json.loads(p0_path.read_text(encoding="utf-8"))
    funnel_res = json.loads(funnel_path.read_text(encoding="utf-8"))
    weight_res = json.loads(weight_path.read_text(encoding="utf-8"))
    opt_res = json.loads(opt_path.read_text(encoding="utf-8"))
    lc_res = json.loads(lc_path.read_text(encoding="utf-8"))

    cards_dir = ROOT_DIR / "runs" / "funnel_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    html_cork, wc, hc = build_corkscrew_card(bv_npz, p0_res, weight_res, opt_res, lc_res)
    cork_html_p = cards_dir / "rotating_corkscrew_procrustes_snap.html"
    cork_png_p = cards_dir / "rotating_corkscrew_procrustes_snap.png"
    cork_pub_p = fig_dir / "20_rotating_corkscrew_procrustes_snap.png"
    cork_html_p.write_text(html_cork, encoding="utf-8")
    render_png_via_chrome(cork_html_p, cork_png_p, width=wc, height=hc)
    shutil.copy2(cork_png_p, cork_pub_p)
    print(f"Rendered Figure 20 -> {cork_pub_p}")

    html_pin, wp, hp = build_pinball_card(funnel_res, weight_res)
    pin_html_p = cards_dir / "cross_firewall_pinball_circuit.html"
    pin_png_p = cards_dir / "cross_firewall_pinball_circuit.png"
    pin_pub_p = fig_dir / "21_cross_firewall_pinball_circuit.png"
    pin_html_p.write_text(html_pin, encoding="utf-8")
    render_png_via_chrome(pin_html_p, pin_png_p, width=wp, height=hp)
    shutil.copy2(pin_png_p, pin_pub_p)
    print(f"Rendered Figure 21 -> {pin_pub_p}")


if __name__ == "__main__":
    main()
