"""Dawn State / Warm Precision plots and self-contained HTML + Markdown discovery reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Dawn State palette tokens
IVORY = "#FFFFF0"
PARCHMENT = "#F5EEDC"
ONYX = "#0F0F0F"
SLATE_DEEP = "#1E3447"
BLUE_SLATE = "#2C3E50"
GOLDEN_ORANGE = "#F39C12"
COPPER_DEEP = "#96590A"
SAGE_GREEN = "#87A96B"
SAGE_DEEP = "#5F7F4A"
SLATE_MIST = "#808B90"

MINUS = "\u2212"
ENDASH = "\u2013"

ARCHETYPE_LABELS: dict[str, str] = {
    "zero_interest_debt": "0% Debt vs. Net-Worth Twin",
    "communal_taboo_pricing": "Communal Help vs. Market Errand",
    "crisis_markup": "Crisis Markup vs. Event Surge",
    "insulting_split": "Insulting Split vs. Equal Split",
    "tainted_provenance": "Tainted Surplus vs. Clean Margin",
    "institutional_scale": "$100M Scale vs. $100 Scale",
}

CONCEPT_LABELS: dict[str, str] = {
    "guilt": "Guilt",
    "captivity": "Captivity",
    "contamination": "Contamination",
    "sacrilege": "Sacrilege",
    "spite": "Spite",
    "predation": "Predation",
    "awe_dread": "Awe / Dread",
    "warmth": "Warmth",
    "valence": "Valence",
    "arousal": "Arousal",
}


def fmt_signed(val: float, decimals: int = 2, plus: bool = True) -> str:
    if not np.isfinite(val):
        return "n/a"
    mag = f"{abs(val):.{decimals}f}"
    if abs(val) < 0.5 * (10 ** (-decimals)):
        return f"{0.0:.{decimals}f}"
    if val < 0:
        return f"{MINUS}{mag}"
    return f"+{mag}" if plus else mag


def fmt_interval(lo: float, hi: float, decimals: int = 2) -> str:
    return f"{fmt_signed(lo, decimals, plus=False)}{ENDASH}{fmt_signed(hi, decimals, plus=False)}"


def _apply_mpl_rc() -> None:
    plt.rcParams.update({
        "figure.facecolor": IVORY,
        "axes.facecolor": IVORY,
        "savefig.facecolor": IVORY,
        "text.color": ONYX,
        "axes.labelcolor": ONYX,
        "xtick.color": SLATE_DEEP,
        "ytick.color": SLATE_DEEP,
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.2,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "svg.fonttype": "none",
    })


def _dawn_diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "dawn_diverging",
        [BLUE_SLATE, "#8EA3B4", IVORY, "#F6C87A", GOLDEN_ORANGE],
        N=256,
    )


def plot_specificity_heatmap(cartography: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Plot 6x8 matrix of paired Cohen's dz on affect-controlled non-financial subspaces."""
    _apply_mpl_rc()
    archetypes: list[str] = cartography["archetypes"]
    concepts: list[str] = cartography["core_concepts"]
    cells: list[dict[str, Any]] = cartography["primary_cells"]

    grid = np.zeros((len(archetypes), len(concepts)), dtype=np.float64)
    p_holm = np.ones_like(grid)
    for cell in cells:
        i = archetypes.index(cell["archetype"])
        j = concepts.index(cell["concept"])
        grid[i, j] = float(cell["cohen_dz_controlled"])
        p_holm[i, j] = float(cell["p_holm_within_archetype"])

    vmax = max(1.0, float(np.max(np.abs(grid))))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    im = ax.imshow(grid, cmap=_dawn_diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(concepts)))
    ax.set_xticklabels([CONCEPT_LABELS.get(c, c) for c in concepts], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(archetypes)))
    ax.set_yticklabels([ARCHETYPE_LABELS.get(a, a) for a in archetypes])

    for i in range(len(archetypes)):
        for j in range(len(concepts)):
            val = grid[i, j]
            star = "*" if p_holm[i, j] < 0.05 else ""
            txt_color = IVORY if val < -0.58 * vmax else ONYX
            ax.text(
                j,
                i,
                f"{fmt_signed(val, 2)}{star}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if star else "normal",
                color=txt_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.025)
    cbar.set_label("Paired shift (Cohen's dz, affect-controlled)", color=SLATE_DEEP, fontsize=9.5)
    cbar.outline.set_edgecolor(BLUE_SLATE)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLUE_SLATE)

    fig.tight_layout()
    svg_path = out_dir / "fig1_specificity_heatmap.svg"
    png_path = out_dir / "fig1_specificity_heatmap.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def plot_layer_emergence(cartography: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Small multiples on one shared scale showing layerwise emergence of top concept per archetype."""
    _apply_mpl_rc()
    archetypes: list[str] = cartography["archetypes"]
    concepts: list[str] = cartography["core_concepts"]
    curves = np.asarray(cartography["dz_controlled_curves"], dtype=np.float64)  # [A, C, L]
    affect_curves = np.asarray(cartography["dz_affect_curves"], dtype=np.float64)  # [A, K, L]
    primary_layer = int(cartography["primary_layer"])
    n_layers = curves.shape[2]
    x = np.arange(n_layers)

    y_lim = max(1.2, float(np.max(np.abs(curves))) * 1.12)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for idx, arch in enumerate(archetypes):
        ax = axes_flat[idx]
        ax.axhline(0.0, color=SLATE_MIST, linestyle="--", linewidth=1.0, zorder=1)
        ax.axvline(primary_layer, color=SLATE_MIST, linestyle=":", linewidth=1.0, zorder=1)

        # Pick strongest concept at primary_layer as focus line; valence (index 0) as context
        c_scores = np.abs(curves[idx, :, primary_layer])
        top_c = int(np.argmax(c_scores))
        second_order = np.argsort(c_scores)[::-1]
        second_c = int(second_order[1]) if len(second_order) > 1 else top_c

        if affect_curves.shape[1] > 0:
            ax.plot(
                x,
                affect_curves[idx, 0],
                color=SLATE_MIST,
                linewidth=1.3,
                linestyle="--",
                label="Valence (raw)",
                zorder=2,
            )
        if second_c != top_c:
            ax.plot(
                x,
                curves[idx, second_c],
                color=SAGE_GREEN,
                linewidth=1.6,
                label=CONCEPT_LABELS.get(concepts[second_c], concepts[second_c]),
                zorder=3,
            )
        ax.plot(
            x,
            curves[idx, top_c],
            color=BLUE_SLATE,
            linewidth=2.2,
            label=f"{CONCEPT_LABELS.get(concepts[top_c], concepts[top_c])} (top)",
            zorder=4,
        )
        ax.scatter(
            [primary_layer],
            [curves[idx, top_c, primary_layer]],
            s=42,
            facecolor=GOLDEN_ORANGE,
            edgecolor=BLUE_SLATE,
            linewidth=1.6,
            zorder=5,
        )

        ax.set_title(ARCHETYPE_LABELS.get(arch, arch), fontsize=9.5, fontweight="bold", loc="left", color=ONYX)
        ax.set_ylim(-y_lim, y_lim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7.8, loc="upper left")
        if idx >= 3:
            ax.set_xlabel("Decoder layer")
        if idx % 3 == 0:
            ax.set_ylabel("Paired shift (dz)")

    fig.tight_layout()
    svg_path = out_dir / "fig2_layer_emergence.svg"
    png_path = out_dir / "fig2_layer_emergence.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def plot_steering_screen(steering: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path] | None:
    """Horizontal bar comparison of causal log-odds rescue: matched concept vs controls."""
    if not steering:
        return None
    _apply_mpl_rc()

    fig, ax = plt.subplots(figsize=(8.8, 0.72 * len(steering) + 1.8))
    y = np.arange(len(steering))
    h = 0.24

    matched = [float(r["delta_gap_matched_controlled"]) for r in steering]
    mismatched = [float(r["delta_gap_mismatched_controlled"]) for r in steering]
    valence = [float(r["delta_gap_valence"]) for r in steering]

    ax.axvline(0.0, color=SLATE_MIST, linestyle="--", linewidth=1.1, zorder=1)
    ax.barh(
        y - h,
        matched,
        height=h,
        facecolor=GOLDEN_ORANGE,
        edgecolor=BLUE_SLATE,
        linewidth=1.5,
        label="Subtract top specific concept (controlled)",
        zorder=4,
    )
    ax.barh(
        y,
        mismatched,
        height=h,
        facecolor=SAGE_GREEN,
        edgecolor=BLUE_SLATE,
        linewidth=1.1,
        label="Subtract mismatched concept (dissociation control)",
        zorder=3,
    )
    ax.barh(
        y + h,
        valence,
        height=h,
        facecolor=SLATE_MIST,
        edgecolor=BLUE_SLATE,
        linewidth=1.0,
        label="Subtract generic valence",
        zorder=2,
    )

    labels = [
        f"{ARCHETYPE_LABELS.get(r['archetype'], r['archetype'])}\n({CONCEPT_LABELS.get(r['matched_concept'], r['matched_concept'])} vs. {CONCEPT_LABELS.get(r['mismatched_concept'], r['mismatched_concept'])})"
        for r in steering
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("Change in anomaly gap (log-odds; negative reduces psychological bias)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="best")

    fig.tight_layout()
    svg_path = out_dir / "fig3_micro_steering.svg"
    png_path = out_dir / "fig3_micro_steering.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def write_reports(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Generate both portable summary.md and browsable Dawn State report.html."""
    meta = results["metadata"]
    sanity = results["sanity_gate"]
    basis_diag = results["basis_diagnostics"]
    behavior = results["behavioral_anomalies"]
    cart = results["cartography_scenario"]
    steering = results.get("micro_steering", [])

    svg1, _ = plot_specificity_heatmap(cart, out_dir)
    svg2, _ = plot_layer_emergence(cart, out_dir)
    steer_paths = plot_steering_screen(steering, out_dir)

    primary_layer = cart["primary_layer"]
    n_layers = meta["n_layers"]

    # Top cell per archetype
    top_by_arch: dict[str, dict[str, Any]] = {}
    for arch in cart["archetypes"]:
        sub = [c for c in cart["primary_cells"] if c["archetype"] == arch]
        top_by_arch[arch] = max(sub, key=lambda c: abs(c["cohen_dz_controlled"]))

    # Markdown report (no em dashes, exact Warm Precision conventions)
    md_lines: list[str] = [
        f"# Project Afriat Discovery Report: `{meta['model_id']}`",
        "",
        f"Phase 0 cartography run completed on `{meta['timestamp_utc']}` across {n_layers} decoder layers "
        f"(stimulus manifest `{meta['stimulus_manifest']['combined_sha256'][:12]}`).",
        "",
        "## 1. Arithmetic & Equivalence Gate",
        "",
        f"- **Overall accuracy:** {sanity['accuracy'] * 100:.1f}% ({sanity['n_correct']} of {sanity['n_total']} items)",
        f"- **Mean dominant-choice log-odds:** {fmt_signed(sanity['mean_log_odds'], 2)} "
        f"(mean A/B token probability mass: {sanity['mean_ab_mass']:.2f})",
        f"- **Gate status:** {'PASS (>= 85%)' if sanity['passed_gate'] else 'WARN (< 85%, consider promoting to 27B-31B)'}",
        "",
        "## 2. Behavioral Anomalies Against Payoff-Equivalent Controls",
        "",
        "| Financial interaction | Anomaly shift (log-odds) | 95% interval | Arch P(deviate) | Ctrl P(deviate) | Exact p | Holm p |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in behavior:
        md_lines.append(
            f"| {ARCHETYPE_LABELS.get(row['archetype'], row['archetype'])} | "
            f"{fmt_signed(row['mean_delta_log_odds'], 2)} | "
            f"{fmt_interval(row['ci95'][0], row['ci95'][1], 2)} | "
            f"{row['mean_arch_prob'] * 100:.1f}% | "
            f"{row['mean_ctrl_prob'] * 100:.1f}% | "
            f"{row['p_signflip']:.4g} | "
            f"{row['p_holm']:.4g} |"
        )

    md_lines.extend([
        "",
        f"## 3. Non-Financial Basis Validation at Layer {primary_layer}",
        "",
        f"Layer {primary_layer} of {n_layers - 1} was selected solely from leave-one-pair-out cross-validation on "
        "the 200 non-financial concept pairs before inspecting financial prompts.",
        "",
        "| Concept | LOPO sign accuracy | LOPO cosine | Norm retained after removing valence & arousal |",
        "| :--- | ---: | ---: | ---: |",
    ])
    for row in basis_diag:
        ret = f"{row['residual_ratio_at_primary'] * 100:.1f}%" if row["residual_ratio_at_primary"] is not None else "control axis"
        md_lines.append(
            f"| {CONCEPT_LABELS.get(row['concept'], row['concept'])} | "
            f"{row['lopo_acc_at_primary'] * 100:.1f}% | "
            f"{fmt_signed(row['lopo_cos_at_primary'], 2, plus=False)} | "
            f"{ret} |"
        )

    md_lines.extend([
        "",
        f"## 4. Strongest Specific Metaphysical Associations (Layer {primary_layer}, Valence & Arousal Removed)",
        "",
        "| Financial interaction | Top specific concept | Cohen's dz (controlled) | 95% interval (stream RMS) | Within-row Holm p | Global Holm p |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ])
    for arch in cart["archetypes"]:
        c = top_by_arch[arch]
        md_lines.append(
            f"| {ARCHETYPE_LABELS.get(arch, arch)} | "
            f"{CONCEPT_LABELS.get(c['concept'], c['concept'])} | "
            f"{fmt_signed(c['cohen_dz_controlled'], 2)} | "
            f"{fmt_interval(c['ci95_controlled'][0], c['ci95_controlled'][1], 3)} | "
            f"{c['p_holm_within_archetype']:.4g} | "
            f"{c['p_holm_global']:.4g} |"
        )

    if steering:
        md_lines.extend([
            "",
            f"## 5. Closed-Loop Micro-Steering Screen (Layer {primary_layer}, alpha = {steering[0]['alpha']})",
            "",
            "| Financial interaction | Matched concept | Gap shift (matched) | Gap shift (mismatched) | Gap shift (valence) | Sanity gate under steering |",
            "| :--- | :--- | ---: | ---: | ---: | ---: |",
        ])
        for r in steering:
            md_lines.append(
                f"| {ARCHETYPE_LABELS.get(r['archetype'], r['archetype'])} | "
                f"{CONCEPT_LABELS.get(r['matched_concept'], r['matched_concept'])} | "
                f"{fmt_signed(r['delta_gap_matched_controlled'], 2)} | "
                f"{fmt_signed(r['delta_gap_mismatched_controlled'], 2)} | "
                f"{fmt_signed(r['delta_gap_valence'], 2)} | "
                f"{r['sanity_acc_steered'] * 100:.1f}% |"
            )

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Self-contained browsable HTML report (Dawn State tokens)
    svg1_inline = svg1.read_text(encoding="utf-8")
    svg2_inline = svg2.read_text(encoding="utf-8")
    svg3_inline = steer_paths[0].read_text(encoding="utf-8") if steer_paths else ""

    beh_rows_html = "".join(
        f"<tr><td>{html.escape(ARCHETYPE_LABELS.get(r['archetype'], r['archetype']))}</td>"
        f"<td class='num'>{fmt_signed(r['mean_delta_log_odds'], 2)}</td>"
        f"<td class='num'>{fmt_interval(r['ci95'][0], r['ci95'][1], 2)}</td>"
        f"<td class='num'>{r['mean_arch_prob']*100:.1f}%</td>"
        f"<td class='num'>{r['mean_ctrl_prob']*100:.1f}%</td>"
        f"<td class='num'>{r['p_holm']:.4g}</td></tr>"
        for r in behavior
    )

    steer_section_html = ""
    if steer_paths:
        steer_section_html = f"""
        <section>
          <div class="kicker">CAUSAL SCREEN</div>
          <h2>Subtracting specific metaphysical vectors vs. dissociation controls</h2>
          <figure class="wp-fig">
            <div class="fig-title">Targeted concept subtraction shifts transactional anomalies selectively</div>
            <div class="fig-sub">Change in archetype-minus-control log-odds gap at layer {primary_layer}, 16 paired fixtures per row</div>
            <div class="svg-wrap">{svg3_inline}</div>
            <div class="fig-src">Source: Project Afriat closed-loop residual steering screen ({html.escape(meta['model_id'])}).</div>
          </figure>
        </section>
        """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project Afriat Discovery Report ({html.escape(meta['model_id'])})</title>
  <style>
    :root {{
      --ivory: {IVORY}; --parchment: {PARCHMENT}; --onyx: {ONYX};
      --slate-deep: {SLATE_DEEP}; --blue-slate: {BLUE_SLATE};
      --golden-orange: {GOLDEN_ORANGE}; --copper-deep: {COPPER_DEEP};
      --sage-deep: {SAGE_DEEP}; --slate-mist: {SLATE_MIST};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: var(--ivory); color: var(--onyx);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      font-size: 17px; line-height: 1.6;
    }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 48px 24px 80px; }}
    .kicker {{
      font-size: 12px; font-weight: 600; letter-spacing: 0.12em;
      text-transform: uppercase; color: var(--slate-deep); margin-bottom: 6px;
    }}
    h1 {{
      font-family: Newsreader, Georgia, serif; font-weight: 600;
      letter-spacing: -0.04em; font-size: 38px; line-height: 1.18; margin: 0 0 12px;
    }}
    h2 {{ font-size: 22px; font-weight: 600; letter-spacing: -0.02em; margin: 40px 0 12px; }}
    .lede {{ font-size: 19px; color: var(--slate-deep); max-width: 66ch; margin: 0 0 28px; }}
    .well {{
      background: var(--parchment); border: 2px solid var(--blue-slate);
      border-radius: 12px; padding: 16px 20px; margin: 20px 0 32px;
      display: flex; flex-wrap: wrap; gap: 24px; font-size: 15px;
    }}
    .well strong {{ display: block; font-size: 20px; font-variant-numeric: tabular-nums; }}
    figure.wp-fig {{
      margin: 28px 0; padding: 20px; border: 1px solid rgba(44, 62, 80, 0.22);
      border-radius: 12px; background: var(--ivory);
    }}
    .fig-title {{ font-size: 19px; font-weight: 600; color: var(--onyx); margin-bottom: 4px; }}
    .fig-sub {{ font-size: 15px; color: var(--slate-deep); margin-bottom: 14px; }}
    .fig-src {{ font-size: 13.5px; color: var(--slate-deep); margin-top: 10px; }}
    .svg-wrap {{ overflow-x: auto; }}
    .svg-wrap svg {{ max-width: 100%; height: auto; display: block; }}
    .table-wrap {{ overflow-x: auto; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 15px; font-variant-numeric: tabular-nums; }}
    th {{
      text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--blue-slate);
      color: var(--slate-deep); font-weight: 600;
    }}
    th.num, td.num {{ text-align: right; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid rgba(44, 62, 80, 0.22); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }}
  </style>
</head>
<body>
  <main>
    <div class="kicker">PROJECT AFRIAT · PHASE 0 CARTOGRAPHY</div>
    <h1>Emotional and Metaphysical Geometry of Financial Interactions</h1>
    <p class="lede">
      Exploratory white-box map of <code>{html.escape(meta['model_id'])}</code> across six payoff-matched
      financial interactions projected onto eight non-financial human concepts after removing generic valence and arousal.
    </p>
    <div class="well">
      <div><span>Arithmetic Sanity Gate</span><strong>{sanity['accuracy']*100:.1f}% ({sanity['n_correct']}/{sanity['n_total']})</strong></div>
      <div><span>Selected Basis Layer</span><strong>Layer {primary_layer} of {n_layers - 1}</strong></div>
      <div><span>Mean A/B Token Mass</span><strong>{sanity['mean_ab_mass']:.2f}</strong></div>
      <div><span>Stimulus Hash</span><strong><code>{meta['stimulus_manifest']['combined_sha256'][:12]}</code></strong></div>
    </div>

    <section>
      <div class="kicker">SPECIFICITY MATRIX</div>
      <h2>How clinical transactions project onto non-financial concepts</h2>
      <figure class="wp-fig">
        <div class="fig-title">Specific metaphysical projections at layer {primary_layer} after removing valence and arousal</div>
        <div class="fig-sub">Paired Cohen's dz across 16 payoff-matched fixtures per row; asterisks mark Holm-adjusted p &lt; 0.05 within row</div>
        <div class="svg-wrap">{svg1_inline}</div>
        <div class="fig-src">Source: Project Afriat stimulus bank (200 non-financial concept pairs, 96 clinical transaction pairs; exact 2^16 sign-flip test).</div>
      </figure>
    </section>

    <section>
      <div class="kicker">LAYER EMERGENCE</div>
      <h2>Where metaphysical associations appear across decoder depth</h2>
      <figure class="wp-fig">
        <div class="fig-title">Layer-by-layer emergence of the leading non-financial concept in each transaction</div>
        <div class="fig-sub">Paired Cohen's dz across layers 0{ENDASH}{n_layers - 1} on one shared vertical scale; dotted line marks pre-selected layer {primary_layer}</div>
        <div class="svg-wrap">{svg2_inline}</div>
        <div class="fig-src">Source: Residual-stream states on clinical scenario descriptions prior to option presentation.</div>
      </figure>
    </section>

    <section>
      <div class="kicker">BEHAVIORAL BASELINE</div>
      <h2>Departures from payoff-invariant choices</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Financial interaction</th>
              <th class="num">Shift (log-odds)</th>
              <th class="num">95% interval</th>
              <th class="num">Archetype P(deviate)</th>
              <th class="num">Control P(deviate)</th>
              <th class="num">Holm p</th>
            </tr>
          </thead>
          <tbody>{beh_rows_html}</tbody>
        </table>
      </div>
    </section>
    {steer_section_html}
  </main>
</body>
</html>
"""
    html_path = out_dir / "report.html"
    html_path.write_text(html_doc, encoding="utf-8")
    return md_path, html_path
