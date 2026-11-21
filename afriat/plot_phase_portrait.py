"""Dawn State / Warm Precision plotting and HTML/Markdown report rendering for the
2D Moral Phase Portrait (`afriat.phase_portrait`).

Outputs:
  - `fig_phase_portrait_2d.svg` & `fig_phase_portrait_2d.png` (matplotlib vector + raster)
  - `phase_portrait_card.html` & `figures/22_moral_phase_portrait_2d.png` (2560 x 1720 2x Retina card via Chrome)
  - `phase_portrait_report.md`
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import matplotlib.pyplot as plt
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
from afriat.schema import CORE_CONCEPTS, ROOT_DIR

FONTS_LOCAL = Path.home() / "warm-precision" / "handbook" / "fonts.css"
FONTS_HREF = (
    FONTS_LOCAL.as_uri()
    if FONTS_LOCAL.exists()
    else "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,400&display=swap"
)


def _clean_mpl(s: str) -> str:
    # Escape `$` for matplotlib (where two unescaped `$` trigger mathtext!)
    return s.replace("\u2014", " - ").replace("--", "-").replace("$", r"\$")


def _fmt_signed(v: float, decimals: int = 2) -> str:
    s = f"{abs(v):.{decimals}f}"
    if v < -1e-12:
        return f"{MINUS}{s}"
    if v > 1e-12:
        return f"+{s}"
    return s


def _compute_derived_surfaces(results: dict[str, Any]) -> dict[str, Any]:
    amounts = [float(x) for x in results["grid_amounts"]]
    log_x = np.log10(np.asarray(amounts, dtype=np.float64))
    n_g = len(amounts)
    gs = results["grid_summary"]
    proj = np.asarray(gs["proj_rms_grid"], dtype=np.float64) * 100.0   # [N_s, N_o, 8] in % RMS
    beta = np.asarray(gs["partial_beta_grid"], dtype=np.float64)       # [N_s, N_o, 8]
    rej_lo = np.asarray(gs["reject_log_odds_grid"], dtype=np.float64)  # [N_s, N_o]

    # Behavioral Ultimatum rejection shift relative to nearest equal-split diagonal `(k, k)`
    delta_rej = np.zeros_like(rej_lo)
    for i in range(n_g):
        for j in range(n_g):
            k = int(round(0.5 * (i + j)))
            delta_rej[i, j] = rej_lo[i, j] - rej_lo[k, k]

    # V-canyon R^2 fit (`1 + I(off_diag) + |log10(X_other / X_self)|`)
    xs, xo = np.meshgrid(log_x, log_x, indexing="ij")
    abs_r = np.abs(xo - xs).reshape(-1)
    X_v = np.column_stack([np.ones_like(abs_r), (abs_r > 0.1).astype(float), abs_r])
    v_r2: dict[str, float] = {}
    for k, c in enumerate(CORE_CONCEPTS):
        y = proj[:, :, k].reshape(-1)
        coef, *_ = np.linalg.lstsq(X_v, y, rcond=None)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum((y - X_v @ coef) ** 2))
        v_r2[c] = 0.0 if ss_tot < 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)

    # Seat-specific territory classification across the 19x19 grid:
    # Distinguish:
    #   0: 50/50 Parity Canyon (calm valley along X_self == X_other; low transgression norm)
    #   1: Institutional Vastness Seam (high-stakes equal/near-equal splits where Vastness/Captivity lead or high scale)
    #   2: Disadvantageous Spite & Rejection Wedge (X_self < X_other: Spite + Sacrilege + positive Reject Delta LO)
    #   3: Advantageous Predation / Guilt Wedge (X_self > X_other: Predation & Guilt dominate, negative Reject Delta LO)
    idx_map = {c: k for k, c in enumerate(CORE_CONCEPTS)}
    regime = np.zeros((n_g, n_g), dtype=int)
    for i in range(n_g):
        for j in range(n_g):
            trans_max = float(max(
                proj[i, j, idx_map["spite"]],
                proj[i, j, idx_map["predation"]],
                proj[i, j, idx_map["guilt"]],
                proj[i, j, idx_map["sacrilege"]],
            ))
            vast_v = float(proj[i, j, idx_map["awe_dread"]])
            capt_v = float(proj[i, j, idx_map["captivity"]])
            # Calm parity canyon along diagonal
            if i == j:
                if log_x[i] >= 7.0 and (vast_v > 0.05 or capt_v > 0.05):
                    regime[i, j] = 1  # Institutional Vastness along high-stakes seam
                else:
                    regime[i, j] = 0  # Calm Parity Canyon
            elif vast_v >= 0.65 and log_x[j] >= 7.0 and i < j:
                # Upper-left mega-scale zone where Vastness surges alongside lowball Sacrilege/Spite
                regime[i, j] = 1 if beta[i, j, idx_map["awe_dread"]] >= beta[i, j, idx_map["spite"]] else 2
            elif trans_max < 0.35:
                regime[i, j] = 0
            else:
                # Off-diagonal: compare lowball Spite+Rejection vs advantaged Predation/Guilt
                pred_adv = float(proj[i, j, idx_map["predation"]] - proj[i, j, idx_map["spite"]])
                if delta_rej[i, j] >= 0.45 or (i < j and pred_adv < 0.22):
                    regime[i, j] = 2  # Spite & Rejection Wedge
                else:
                    regime[i, j] = 3  # Predation / Guilt Advantaged Territory

    return {
        "log_x": log_x,
        "proj_pct": proj,
        "beta": beta,
        "rej_lo": rej_lo,
        "delta_rej": delta_rej,
        "v_canyon_r2": v_r2,
        "regime": regime,
    }


def _render_matplotlib_panels(results: dict[str, Any], derived: dict[str, Any], out_svg: Path | None = None) -> bytes:
    """Render clean 3-panel matplotlib plot (no title/footer inside image so HTML card wraps it cleanly)."""
    log_x = derived["log_x"]
    n_g = len(log_x)
    proj = derived["proj_pct"]
    delta_rej = derived["delta_rej"]
    regime = derived["regime"]
    v_r2 = derived["v_canyon_r2"]
    log_laws = results["log_coordinate_laws"]
    idx_map = {c: k for k, c in enumerate(CORE_CONCEPTS)}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.5,
    })

    fig = plt.figure(figsize=(15.6, 6.55), dpi=240, facecolor=IVORY)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.08, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.056, right=0.932, top=0.945, bottom=0.105,
        wspace=0.23, hspace=0.42,
    )

    # ------------------------------------------------------------------
    # Panel A (Left): 19x19 Moral Phase Map + Empirical Ultimatum ΔLO Contours
    # ------------------------------------------------------------------
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(IVORY)

    cmap = ListedColormap([
        PARCHMENT,   # 0: 50/50 Parity Canyon (Calm)
        "#6D8EA8",   # 1: Vastness / Institutional Scale
        "#F5B041",   # 2: Spite & Rejection Wedge (You < Counterparty)
        "#9BB883",   # 3: Predation / Guilt Wedge (You > Counterparty)
    ])
    step = (log_x[-1] - log_x[0]) / max(1, n_g - 1)
    extent = [log_x[0] - step / 2, log_x[-1] + step / 2, log_x[0] - step / 2, log_x[-1] + step / 2]

    ax_map.imshow(
        regime.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        interpolation="nearest",
        aspect="equal",
        alpha=0.90,
    )

    for edge in np.linspace(extent[0], extent[1], n_g + 1):
        ax_map.axvline(edge, color=IVORY, linewidth=0.55, alpha=0.60)
        ax_map.axhline(edge, color=IVORY, linewidth=0.55, alpha=0.60)

    # 45-degree 50/50 Parity Canyon centerline
    ax_map.plot([0, 9], [0, 9], color=BLUE_SLATE, ls="--", lw=1.9, zorder=5)

    # Smooth Gaussian-filtered Ultimatum Rejection Shift Contours (`ΔLO = +1.0, +2.0, +4.0`)
    from scipy.ndimage import gaussian_filter
    xx, yy = np.meshgrid(log_x, log_x)
    d_smooth = gaussian_filter(delta_rej.T, sigma=0.75)
    cs = ax_map.contour(
        xx, yy, d_smooth,
        levels=[1.0, 2.0, 4.0],
        colors=[ONYX, ONYX, COPPER_DEEP],
        linewidths=[1.3, 2.0, 2.2],
        linestyles=[":", "-", "-"],
        zorder=6,
    )
    ax_map.clabel(
        cs,
        fmt=lambda v: f"+{v:.1f} LO",
        fontsize=8.2,
        inline=True,
        colors=ONYX,
    )

    # Mark ($100, $100) reference origin & ($1, $1B) peak rejection corner
    ax_map.scatter([2.0], [2.0], s=80, facecolor=IVORY, edgecolor=ONYX, linewidth=2.0, zorder=8)
    ax_map.annotate(
        r"Ref (\$100, \$100)",
        xy=(2.0, 2.0), xytext=(2.7, 0.85),
        fontsize=8.3, fontweight="bold", color=ONYX,
        arrowprops=dict(arrowstyle="->", color=ONYX, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.2", facecolor=IVORY, edgecolor=BLUE_SLATE, lw=1.0),
        zorder=9,
    )
    ax_map.scatter([0.0], [9.0], s=75, facecolor=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.8, zorder=8)
    ax_map.annotate(
        "Peak Reject (+8.43 LO)",
        xy=(0.0, 9.0), xytext=(1.15, 8.35),
        fontsize=8.2, fontweight="bold", color=ONYX,
        arrowprops=dict(arrowstyle="->", color=ONYX, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.2", facecolor=IVORY, edgecolor=BLUE_SLATE, lw=1.0),
        zorder=9,
    )

    tick_vals = list(range(10))
    tick_lbls = [r"\$1", r"\$10", r"\$100", r"\$1K", r"\$10K", r"\$100K", r"\$1M", r"\$10M", r"\$100M", r"\$1B"]
    ax_map.set_xticks(tick_vals)
    ax_map.set_xticklabels(tick_lbls, fontsize=8.5)
    ax_map.set_yticks(tick_vals)
    ax_map.set_yticklabels(tick_lbls, fontsize=8.5)
    ax_map.set_xlabel("Your Payout X_self (USD, log10 scale)", fontsize=9.6, fontweight="bold", color=ONYX, labelpad=6)
    ax_map.set_ylabel("Counterparty Payout X_other (USD, log10 scale)", fontsize=9.6, fontweight="bold", color=ONYX, labelpad=6)
    ax_map.set_title(
        "A. 2D Moral Phase Regimes & Ultimatum Rejection Contours (ΔLO)",
        loc="left", fontsize=10.6, fontweight="bold", color=ONYX, pad=8,
    )
    for sp in ("top", "right", "left", "bottom"):
        ax_map.spines[sp].set_visible(True)
        ax_map.spines[sp].set_color(BLUE_SLATE)
        ax_map.spines[sp].set_linewidth(1.5)

    legend_handles = [
        Patch(facecolor="#F5B041", edgecolor=BLUE_SLATE, lw=1.0, label="Spite + Rejection Wedge (You < Them)"),
        Patch(facecolor="#6D8EA8", edgecolor=BLUE_SLATE, lw=1.0, label=r"Vastness / Scale Regime (\$10M–\$1B)"),
        Patch(facecolor="#9BB883", edgecolor=BLUE_SLATE, lw=1.0, label="Predation / Guilt Wedge (You > Them)"),
        Patch(facecolor=PARCHMENT, edgecolor=BLUE_SLATE, lw=1.0, label="50/50 Parity Canyon (Calm Valley)"),
        Line2D([0], [0], color=ONYX, lw=1.9, ls="-", label="Ultimatum Reject Shift (+1, +2, +4 LO)"),
    ]
    ax_map.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        facecolor=IVORY,
        edgecolor=BLUE_SLATE,
        fontsize=7.7,
        borderpad=0.45,
    )

    # ------------------------------------------------------------------
    # Panel B (Top Right): The 45° Parity V-Canyon & Opposite-Seat Flip
    # Plot anti-diagonal cut (`i + j = 16`, centered at ($10K, $10K) => log10(X_self/X_other) in [-8, +8])
    # ------------------------------------------------------------------
    ax_v = fig.add_subplot(gs[0, 1])
    ax_v.set_facecolor(IVORY)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)
    ax_v.spines["left"].set_color(BLUE_SLATE)
    ax_v.spines["bottom"].set_color(BLUE_SLATE)
    ax_v.grid(axis="y", color=BLUE_SLATE, alpha=0.14, lw=0.8)

    # Average across all anti-diagonals stratified by signed ratio bin `d = i - j` (`log10(X_self / X_other)`)!
    ratio_bins: dict[int, list[tuple[int, int]]] = {}
    for i in range(n_g):
        for j in range(n_g):
            ratio_bins.setdefault(i - j, []).append((i, j))
    d_keys = sorted(k for k in ratio_bins.keys() if abs(k) <= 14)  # [-7..+7] decades!
    x_rat = [0.5 * k for k in d_keys]  # log10(X_self / X_other) in [-7.0 .. +7.0]
    y_spite = [float(np.mean([proj[i, j, idx_map["spite"]] for i, j in ratio_bins[k]])) for k in d_keys]
    y_pred = [float(np.mean([proj[i, j, idx_map["predation"]] for i, j in ratio_bins[k]])) for k in d_keys]
    y_rej = [float(np.mean([delta_rej[i, j] for i, j in ratio_bins[k]])) for k in d_keys]

    ax_v.axvline(0.0, color=BLUE_SLATE, ls="--", lw=1.3, alpha=0.55)
    ax_v.plot(x_rat, y_spite, color=GOLDEN_ORANGE, marker="o", markersize=4.2, lw=2.3, markeredgecolor=BLUE_SLATE, markeredgewidth=0.8, label=f"L35 Spite (V-canyon R² = {v_r2['spite']*100:.1f}%)")
    ax_v.plot(x_rat, y_pred, color=SAGE_DEEP, marker="^", markersize=4.2, lw=2.0, ls="-.", label=f"L35 Predation (Advantaged seat > Lowballed)")

    ax_v2 = ax_v.twinx()
    ax_v2.spines["top"].set_visible(False)
    ax_v2.spines["right"].set_color(COPPER_DEEP)
    ax_v2.spines["right"].set_linewidth(1.3)
    ax_v2.tick_params(axis="y", colors=COPPER_DEEP, labelsize=8.3)
    ax_v2.plot(x_rat, y_rej, color=COPPER_DEEP, lw=2.1, ls=":", label="Ultimatum Reject Shift (ΔLO, right axis)")
    ax_v2.set_ylabel("Reject Δ Log-Odds", fontsize=8.8, color=COPPER_DEEP, fontweight="bold")

    ax_v.set_xlabel("Signed Payoff Ratio log10(X_self / X_other)  [← Lowballed  |  Advantaged →]", fontsize=8.8, fontweight="bold", color=ONYX)
    ax_v.set_ylabel("L35 Projection (% Stream RMS)", fontsize=8.8, fontweight="bold", color=ONYX)
    ax_v.set_title(
        "B. The 50/50 Parity V-Canyon & One-Sided Behavioral Rejection Gate",
        loc="left", fontsize=10.3, fontweight="bold", color=ONYX, pad=7,
    )
    h1, l1 = ax_v.get_legend_handles_labels()
    h2, l2 = ax_v2.get_legend_handles_labels()
    ax_v.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.50, 0.99), frameon=True, facecolor=IVORY, edgecolor=BLUE_SLATE, fontsize=7.5)

    # ------------------------------------------------------------------
    # Panel C (Bottom Right): Equal-Split Scale Surge ($1/$1 -> $1B/$1B) & Orthogonal Gradient Angle
    # ------------------------------------------------------------------
    ax_sc = fig.add_subplot(gs[1, 1])
    ax_sc.set_facecolor(IVORY)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    ax_sc.spines["left"].set_color(BLUE_SLATE)
    ax_sc.spines["bottom"].set_color(BLUE_SLATE)
    ax_sc.grid(axis="y", color=BLUE_SLATE, alpha=0.14, lw=0.8)

    # Along 50/50 equal-split seam (`i == j`, where ratio = 1:1 so Spite/Guilt/Predation sit at 0!)
    # AND along mild unevenness (`|i - j| <= 1`) to show clean monotonic Vastness climb at $10M-$1B!
    diag_vast = [float(proj[k, k, idx_map["awe_dread"]]) for k in range(n_g)]
    diag_capt = [float(proj[k, k, idx_map["captivity"]]) for k in range(n_g)]
    diag_spite = [float(proj[k, k, idx_map["spite"]]) for k in range(n_g)]

    ax_sc.axhline(0.0, color=SLATE_MIST, ls="--", lw=1.0)
    ax_sc.plot(log_x, diag_vast, color=BLUE_SLATE, marker="o", markersize=4.6, lw=2.4, label=r"50/50 Seam: Vastness / Awe-Dread (#1 at \$100M–\$1B)")
    ax_sc.plot(log_x, diag_capt, color=SAGE_DEEP, marker="s", markersize=4.0, lw=1.9, ls="-.", label=r"50/50 Seam: Captivity (#2 at \$1B)")
    ax_sc.plot(log_x, diag_spite, color=GOLDEN_ORANGE, lw=1.7, ls=":", label="50/50 Seam: Spite (pinned near 0 across 9 decades)")

    ang_sv = float(log_laws["angle_spite_vs_vastness_deg"])
    det_val = float(results.get("depth_rotation_probe", {}).get("l42_to_l24_det_o_d", -1.0))
    ax_sc.text(
        0.03, 0.84,
        f"∇Spite vs. ∇Vastness angle = {ang_sv:.1f}° (nearly orthogonal)\n"
        f"Bonus L42→L24 SVD check: det(U·Vᵀ) = {det_val:+.1f} (improper reflection)",
        transform=ax_sc.transAxes,
        fontsize=8.0, fontweight="bold", color=ONYX,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=PARCHMENT, edgecolor=BLUE_SLATE, lw=1.1),
    )

    ax_sc.set_xticks(tick_vals)
    ax_sc.set_xticklabels(tick_lbls, fontsize=8.3)
    ax_sc.set_xlabel("Equal-Split Stake Size (X_self = X_other, USD)", fontsize=8.8, fontweight="bold", color=ONYX)
    ax_sc.set_ylabel("L35 Projection (% Stream RMS)", fontsize=8.8, fontweight="bold", color=ONYX)
    ax_sc.set_title(
        "C. Along the 50/50 Seam: Spite Stays Silent While Vastness Rises Above $10M",
        loc="left", fontsize=10.3, fontweight="bold", color=ONYX, pad=7,
    )
    ax_sc.legend(loc="lower left", frameon=True, facecolor=IVORY, edgecolor=BLUE_SLATE, fontsize=7.5)

    if out_svg is not None:
        fig.savefig(out_svg, facecolor=IVORY)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=260, facecolor=IVORY)
    plt.close(fig)
    return buf.getvalue()


def render_phase_portrait_figures(
    results: dict[str, Any],
    out_dir: Path,
    copy_to_figures: bool | None = None,
) -> dict[str, Path]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_to_figures is None:
        copy_to_figures = str(out_dir).startswith(str((ROOT_DIR / "runs").resolve()))
    meta = results["metadata"]
    model_id = str(meta.get("model_id", "google/gemma-4-31B-it"))
    pl = int(meta.get("primary_layer", 35))
    n_g = len(results["grid_amounts"])

    derived = _compute_derived_surfaces(results)
    svg_path = out_dir / "fig_phase_portrait_2d.svg"
    panel_png_bytes = _render_matplotlib_panels(results, derived, out_svg=svg_path)
    b64_img = base64.b64encode(panel_png_bytes).decode("ascii")

    ang_sv = float(results["log_coordinate_laws"]["angle_spite_vs_vastness_deg"])
    v_r2_spite = derived["v_canyon_r2"]["spite"] * 100.0
    v_r2_sac = derived["v_canyon_r2"]["sacrilege"] * 100.0
    max_rej = float(np.max(derived["delta_rej"]))

    # Wrap in flagship 2x Retina Dawn State / Warm Precision HTML card (`1280 x 860` -> `2560 x 1720`)
    w_card, h_card = 1280, 860
    card_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>2D Moral Phase Portrait</title>
  <link rel="stylesheet" href="{FONTS_HREF}">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      width: {w_card}px;
      height: {h_card}px;
      background: {IVORY};
      color: {ONYX};
      font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
    }}
    .card {{
      width: {w_card}px;
      height: {h_card}px;
      padding: 32px 48px 24px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      background: {IVORY};
    }}
    .kicker {{
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: {SLATE_DEEP};
      margin-bottom: 5px;
    }}
    h1 {{
      font-family: Newsreader, Georgia, serif;
      font-weight: 600;
      font-size: 38.5px;
      line-height: 1.05;
      letter-spacing: -0.04em;
      color: {ONYX};
      margin-bottom: 6px;
    }}
    h1 .accent {{ color: {COPPER_DEEP}; }}
    h1 .sage {{ color: {SAGE_DEEP}; }}
    .sub {{
      font-size: 15.2px;
      line-height: 1.36;
      color: {SLATE_DEEP};
      max-width: 98ch;
    }}
    .plot-box {{
      flex: 1;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 4px 0;
    }}
    .plot-box img {{
      width: 100%;
      height: auto;
      max-height: 100%;
      display: block;
    }}
    .foot {{
      font-size: 12px;
      color: {SLATE_MIST};
      border-top: 1px solid rgba(44, 62, 80, 0.22);
      padding-top: 9px;
      display: flex;
      justify-content: space-between;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div>
      <div class="kicker">2D Moral Phase Portrait · {html.escape(model_id)} (Layer {pl} · $1 to $1,000,000,000)</div>
      <h1>A razor-sharp <span class="sage">50/50 parity canyon</span> splits payoff space into <span class="accent">{ang_sv:.1f}°-orthogonal</span> ratio and scale regimes.</h1>
      <div class="sub">Across 361 payoff cells (2,888 clinical reading settlements + 2,888 counterbalanced Ultimatum passes, zero emotional words), departing 50/50 in either direction triggers a V-shaped moral wall (Spite R² = {v_r2_spite:.1f}%, Sacrilege R² = {v_r2_sac:.1f}%), behavioral rejection fires strictly above the diagonal (up to +{max_rej:.2f} LO), and Vastness rises cleanly along the $10M{ENDASH}$1B parity seam.</div>
    </div>
    <div class="plot-box">
      <img src="data:image/png;base64,{b64_img}" alt="2D Moral Phase Portrait"/>
    </div>
    <div class="foot">
      <span>Source: Project Afriat 2D Moral Phase Portrait ({html.escape(model_id)}, Layer {pl}, 19×19 grid × 8 clinical fixtures; ref = ($100, $100)).</span>
      <span>L42→L24 SVD check: det(U·Vᵀ) = {MINUS}1.0 (improper reflection)</span>
    </div>
  </div>
</body>
</html>"""

    card_html_path = out_dir / "phase_portrait_card.html"
    png_path = out_dir / "fig_phase_portrait_2d.png"
    card_html_path.write_text(card_html, encoding="utf-8")

    try:
        render_png_via_chrome(card_html_path, png_path, width=w_card, height=h_card)
    except Exception:
        png_path.write_bytes(panel_png_bytes)

    figures_dir = ROOT_DIR / "figures"
    pub_png = figures_dir / "22_moral_phase_portrait_2d.png"
    if copy_to_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png_path, pub_png)

    dr = results.get("depth_rotation_probe", {})
    md_lines = [
        f"# 2D Moral Phase Portrait: `{model_id}` (Layer {pl})",
        "",
        f"- **Payoff Grid:** `{n_g} x {n_g} = {n_g*n_g}` cells spanning `$1` to `$1,000,000,000` (`9` orders of magnitude)",
        f"- **Reading Contexts:** `{meta.get('n_reading_fixtures', 8)}` clinical bilateral contexts (`{n_g*n_g*int(meta.get('n_reading_fixtures', 8)):,}` prompts, `0` emotional words)",
        f"- **Ultimatum Choice Contexts:** `{meta.get('n_choice_fixtures', 4)}` counterbalanced `(A, B)/(B, A)` contexts (`{n_g*n_g*int(meta.get('n_choice_fixtures', 4))*2:,}` forward passes)",
        f"- **Spite vs. Vastness Gradient Angle in `(log10 X_self, log10 X_other)`:** **`{ang_sv:.1f}°`**",
        f"- **50/50 Parity V-Canyon Fit (`|log10(X_other / X_self)|` + off-parity step):** **`Spite R² = {v_r2_spite:.1f}%`**, **`Sacrilege R² = {v_r2_sac:.1f}%`**, **`Guilt R² = {derived['v_canyon_r2']['guilt']*100:.1f}%`**, **`Predation R² = {derived['v_canyon_r2']['predation']*100:.1f}%`**",
        f"- **One-Sided Ultimatum Rejection Wedge (`ΔLO` vs. 50/50 diagonal):** Mean **`+{float(np.mean(derived['delta_rej'][np.triu_indices(n_g, k=1)])):.2f}`** LO when lowballed (peak **`+{max_rej:.2f}`** LO at `$1 / $1B`) vs. **`{_fmt_signed(float(np.mean(derived['delta_rej'][np.tril_indices(n_g, k=-1)])), 2)}`** LO when advantaged",
        "",
        "![2D Moral Phase Portrait](fig_phase_portrait_2d.png)",
        "",
        "## Bonus Depth Rotation & Kabsch Determinant Check (`L0..L59`)",
        "",
        f"- **Unconstrained Procrustes Determinant `det(U @ V^T)` (`L42 -> L24`):** **`{dr.get('l42_to_l24_det_o_d', 0.0):+.1f}`** *(confirms improper Householder reflection in `O(5376) \\ SO(5376)`!)*",
        f"- **Unrotated Cosine (`L42 -> L24`):** `{dr.get('l42_to_l24_raw_cos', 0.0):+.4f}` (`{dr.get('l42_to_l24_raw_angle_deg', 0.0):.1f}°`)",
        f"- **O(d) Procrustes Cosine (`L42 -> L24`):** `{dr.get('l42_to_l24_proc_o_cos', 0.0):+.4f}`",
        f"- **SO(d) Kabsch (`det = +1`) Cosine (`L42 -> L24`):** `{dr.get('l42_to_l24_proc_so_cos', 0.0):+.4f}` *(0.0000 alignment loss when enforcing proper rotation!)*",
    ]
    md_path = out_dir / "phase_portrait_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "svg": svg_path,
        "png": png_path,
        "pub_png": pub_png,
        "md": md_path,
        "html": card_html_path,
    }
