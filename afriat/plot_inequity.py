"""Dawn State / Warm Precision plots and reports for the Inequity Disparity, Agency, & Polarity experiment."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afriat.plot_results import (
    BLUE_SLATE,
    CONCEPT_LABELS,
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
    _apply_mpl_rc,
    fmt_interval,
    fmt_signed,
)


def plot_disparity_curve(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Two-panel Dawn State figure: (Left) Emotion trajectory vs. split share; (Right) Rejection shift."""
    _apply_mpl_rc()
    curve = results["disparity_curve"]
    labels = [row["split_label"].replace(" ", "") for row in curve]
    xs = np.arange(len(curve))

    def _series_partial(c: str) -> list[float]:
        out = [0.0]
        for row in curve[1:]:
            out.append(float(row["scenario_projection"]["concepts"][c]["partial_beta"]))
        return out

    def _series_dz(c: str) -> list[float]:
        out = [0.0]
        for row in curve[1:]:
            out.append(float(row["scenario_projection"]["concepts"][c]["cohen_dz"]))
        return out

    beh_dlo = [float(row["behavior"]["delta_log_odds"]) for row in curve]
    beh_lo = [float(row["behavior"]["delta_ci95"][0]) for row in curve]
    beh_hi = [float(row["behavior"]["delta_ci95"][1]) for row in curve]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # Panel A: Unique Partial Concept Weights (controlling all 7 other concepts + valence/arousal)
    tracked = [
        ("spite", GOLDEN_ORANGE, 2.8, "o"),
        ("guilt", BLUE_SLATE, 2.0, "s"),
        ("predation", COPPER_DEEP, 1.8, "^"),
        ("warmth", SAGE_DEEP, 1.8, "D"),
        ("captivity", SLATE_MIST, 1.5, "x"),
    ]
    ax1.axhline(0.0, color=SLATE_MIST, linestyle="--", linewidth=0.9, alpha=0.6)
    for c, color, lw, marker in tracked:
        ys = _series_partial(c)
        ax1.plot(xs, ys, color=color, linewidth=lw, marker=marker, markersize=5.2, label=CONCEPT_LABELS[c])
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.8)
    ax1.set_ylabel("Unique partial projection beta (controlling 7 concepts + affect)", fontsize=9.5)
    ax1.set_title("A. Fixed-Pool Share Disparity -> Internal Spite Trajectory", loc="left", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")

    # Also annotate 98/2 and 99.9/0.1 marginal Spite dz
    dz_spite = _series_dz("spite")
    ax1.annotate(
        f"Spite dz={fmt_signed(dz_spite[-1], 1)}",
        xy=(xs[-1], _series_partial("spite")[-1]),
        xytext=(-65, 8),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=COPPER_DEEP,
    )

    # Panel B: Counterbalanced Behavioral Rejection Log-Odds Shift vs 50/50
    ax2.axhline(0.0, color=SLATE_MIST, linestyle="--", linewidth=0.9, alpha=0.6)
    ax2.fill_between(xs, beh_lo, beh_hi, color=GOLDEN_ORANGE, alpha=0.22, label="95% bootstrap CI")
    ax2.plot(xs, beh_dlo, color=BLUE_SLATE, linewidth=2.6, marker="o", markersize=5.5, label="Decline log-odds shift")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.8)
    ax2.set_ylabel("Delta log-odds of declining vs. 50/50 on identical pool", fontsize=9.5)
    ax2.set_title("B. Behavioral Ultimatum Rejection Shift (Pool Constant)", loc="left", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")

    svg_p = out_dir / "fig1_disparity_curve.svg"
    png_p = out_dir / "fig1_disparity_curve.png"
    fig.savefig(svg_p, format="svg")
    fig.savefig(png_p, format="png", dpi=220)
    plt.close(fig)
    return svg_p, png_p


def plot_agency_and_polarity(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Grouped bar comparison across the 4 surgical 98/2 conditions on identical pools."""
    _apply_mpl_rc()
    fc_map = {c["contrast_id"]: c for c in results["factorial_contrasts"]}
    conds = [
        ("human_disadv_98_2", "Human 98/2\n(Against You)"),
        ("random_disadv_98_2", "Random 98/2\n(Computer Draw)"),
        ("random_adv_2_98", "Random 2/98\n(Windfall For You)"),
        ("dictator_adv_2_98", "Your Draft 2/98\n(You Keep 98%)"),
    ]
    focus_concepts = [
        ("spite", GOLDEN_ORANGE),
        ("guilt", BLUE_SLATE),
        ("predation", COPPER_DEEP),
        ("warmth", SAGE_GREEN),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    x = np.arange(len(conds))
    width = 0.19

    for idx, (c_name, color) in enumerate(focus_concepts):
        offset = (idx - 1.5) * width
        dz_vals = [float(fc_map[cid]["scenario_projection"]["concepts"][c_name]["cohen_dz"]) for cid, _ in conds]
        beta_vals = [float(fc_map[cid]["scenario_projection"]["concepts"][c_name]["partial_beta"]) for cid, _ in conds]
        ax1.bar(x + offset, dz_vals, width=width, color=color, edgecolor=ONYX, linewidth=0.5, label=CONCEPT_LABELS[c_name])
        ax2.bar(x + offset, beta_vals, width=width, color=color, edgecolor=ONYX, linewidth=0.5, label=CONCEPT_LABELS[c_name])

    for ax, title, ylabel in (
        (ax1, "A. Marginal Controlled Projection (Cohen's dz)", "Cohen's dz (valence & arousal removed)"),
        (ax2, "B. Unique Partial Weight beta (Controlling All 7 Concepts)", "Partial regression weight beta"),
    ):
        ax.axhline(0.0, color=ONYX, linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in conds], fontsize=8.8)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(title, loc="left", fontsize=10.8, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)

    svg_p = out_dir / "fig2_agency_and_polarity.svg"
    png_p = out_dir / "fig2_agency_and_polarity.png"
    fig.savefig(svg_p, format="svg")
    fig.savefig(png_p, format="png", dpi=220)
    plt.close(fig)
    return svg_p, png_p


def plot_bidirectional_steering(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Dose-response curves for alpha in [-2, +2] across Human 98/2, Human 99.9/0.1, and Dictator 2/98."""
    _apply_mpl_rc()
    sdr = results["steering_dose_response"]
    panels = [
        ("human_20", "A. Ultimatum 98/2 (Decline Gap)"),
        ("human_1", "B. Ultimatum 99.9/0.1 (Decline Gap)"),
        ("dictator_980", "C. Dictator 2/98 (Equal-Share Gap)"),
    ]
    vec_styles = [
        ("spite_controlled", "Controlled Spite", GOLDEN_ORANGE, "o", 2.6),
        ("guilt_controlled", "Controlled Guilt", BLUE_SLATE, "s", 2.2),
        ("captivity_controlled", "Controlled Captivity", SAGE_DEEP, "^", 1.7),
        ("valence_raw", "Raw Valence", SLATE_MIST, "x", 1.6),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for ax, (cell_key, title) in zip(axes, panels):
        ax.axhline(0.0, color=SLATE_MIST, linestyle="--", linewidth=0.9, alpha=0.6)
        ax.axvline(0.0, color=SLATE_MIST, linestyle=":", linewidth=0.9, alpha=0.6)
        for v_key, v_label, color, marker, lw in vec_styles:
            rows = sdr[cell_key][v_key]
            alphas = [r["alpha"] for r in rows]
            deltas = [r["delta_from_unsteered"] for r in rows]
            ax.plot(alphas, deltas, color=color, marker=marker, linewidth=lw, markersize=5.0, label=v_label)
        ax.set_xlabel("Steering scale alpha (natural contrast units)", fontsize=9)
        ax.set_ylabel("Shift in behavioral log-odds from alpha=0", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.2, loc="best")

    svg_p = out_dir / "fig3_bidirectional_steering.svg"
    png_p = out_dir / "fig3_bidirectional_steering.png"
    fig.savefig(svg_p, format="svg")
    fig.savefig(png_p, format="png", dpi=220)
    plt.close(fig)
    return svg_p, png_p


def write_inequity_reports(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg1, _ = plot_disparity_curve(results, out_dir)
    svg2, _ = plot_agency_and_polarity(results, out_dir)
    svg3, _ = plot_bidirectional_steering(results, out_dir)

    meta = results["metadata"]
    model_id = meta["model_id"]
    pl = meta["primary_layer"]

    lines: list[str] = [
        f"# Project Afriat Inequity Dissection: `{model_id}`",
        "",
        f"Targeted Phase 1 probe at locked non-financial Layer {pl} of {meta['n_layers'] - 1} across 16 fixed pools (`\\$1,000`{ENDASH}`\\$100,000`).",
        "",
        "## 1. Fixed-Pool Disparity Curve (Human Proposer, Identical Pool Size)",
        "",
        "| Split (Them / You) | Decline shift (log-odds) | 95% interval | P(decline) | Spite dz (ctrl) | Spite partial beta | Guilt partial beta |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["disparity_curve"]:
        b = row["behavior"]
        if row["scenario_projection"] is None:
            s_dz = "0.00"
            s_beta = "0.000"
            g_beta = "0.000"
        else:
            sc = row["scenario_projection"]["concepts"]
            s_dz = fmt_signed(sc["spite"]["cohen_dz"], 2)
            s_beta = fmt_signed(sc["spite"]["partial_beta"], 3)
            g_beta = fmt_signed(sc["guilt"]["partial_beta"], 3)
        lines.append(
            f"| {row['split_label']} | {fmt_signed(b['delta_log_odds'])} | "
            f"{fmt_interval(b['delta_ci95'][0], b['delta_ci95'][1])} | "
            f"{b['mean_prob']*100:.1f}% | {s_dz} | {s_beta} | {g_beta} |"
        )

    lines.extend([
        "",
        "## 2. Agency (Human vs. Random Computer) & Polarity Flip (`98/2` vs. `2/98`)",
        "",
        "| Condition (vs. matched 50/50 on same pool) | Beh. shift (log-odds) | Spite dz | Guilt dz | Spite partial beta | Guilt partial beta | Predation partial beta | Warmth partial beta |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for fc in results["factorial_contrasts"]:
        b = fc["behavior"]
        sc = fc["scenario_projection"]["concepts"]
        lines.append(
            f"| {fc['label']} | {fmt_signed(b['delta_log_odds'])} | "
            f"{fmt_signed(sc['spite']['cohen_dz'])} | {fmt_signed(sc['guilt']['cohen_dz'])} | "
            f"{fmt_signed(sc['spite']['partial_beta'], 3)} | {fmt_signed(sc['guilt']['partial_beta'], 3)} | "
            f"{fmt_signed(sc['predation']['partial_beta'], 3)} | {fmt_signed(sc['warmth']['partial_beta'], 3)} |"
        )

    did98 = results["agency_did"]["98_2"]
    did98_spite = did98["scenario_did"]["concepts"]["spite"]
    lines.extend([
        "",
        f"- **Pure Intentional Agency DiD at `98/2` `(Human - Random)`:** Behavioral shift `{fmt_signed(did98['beh_did_log_odds'])}` log-odds (`p = {did98['beh_did_p']:.4g}`); controlled `Spite` DiD `dz = {fmt_signed(did98_spite['cohen_dz'])}` (`p_holm = {did98_spite['p_holm']:.4g}`), partial `beta = {fmt_signed(did98_spite['partial_beta'], 3)}`.",
        "",
        "## 3. Bidirectional Causal Steering Slope (`alpha = +2` minus `alpha = -2`)",
        "",
        "| Target Choice | Controlled Spite (`+2 vs -2`) | Controlled Guilt (`+2 vs -2`) | Controlled Captivity (`+2 vs -2`) | Raw Valence (`+2 vs -2`) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for cell_key, label in (
        ("human_20", "Ultimatum 98/2 (Decline)"),
        ("human_1", "Ultimatum 99.9/0.1 (Decline)"),
        ("dictator_980", "Dictator 2/98 (Equalize)"),
    ):
        row_cells = []
        for vk in ("spite_controlled", "guilt_controlled", "captivity_controlled", "valence_raw"):
            arr = results["steering_dose_response"][cell_key][vk]
            diff = float(arr[-1]["mean_log_odds"] - arr[0]["mean_log_odds"])
            row_cells.append(fmt_signed(diff))
        lines.append(f"| {label} | {' | '.join(row_cells)} |")

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Project Afriat Inequity Dissection: {html.escape(model_id)}</title>
<style>
  body {{ background: {IVORY}; color: {ONYX}; font-family: Inter, -apple-system, sans-serif; max-width: 1120px; margin: 36px auto; padding: 0 24px; line-height: 1.55; }}
  h1, h2 {{ color: {SLATE_DEEP}; }}
  .card {{ background: {PARCHMENT}; border: 1px solid {BLUE_SLATE}; border-radius: 8px; padding: 18px 22px; margin: 22px 0; }}
  .fig {{ margin: 26px 0; background: {IVORY}; border: 1px solid #D8D2C0; border-radius: 8px; padding: 14px; overflow-x: auto; }}
</style>
</head>
<body>
  <h1>Project Afriat: Anatomy of Financial Inequity (<code>{html.escape(model_id)}</code>)</h1>
  <div class="card">
    <strong>Locked Non-Financial Subspace Layer:</strong> {pl} of {meta['n_layers'] - 1} &nbsp;|&nbsp;
    <strong>Pools:</strong> 16 fixed pools ($1,000&ndash;$100,000) &nbsp;|&nbsp;
    <strong>Runtime:</strong> {meta['elapsed_seconds']}s
  </div>
  <div class="fig">{svg1.read_text(encoding='utf-8')}</div>
  <div class="fig">{svg2.read_text(encoding='utf-8')}</div>
  <div class="fig">{svg3.read_text(encoding='utf-8')}</div>
</body>
</html>
"""
    html_path = out_dir / "report.html"
    html_path.write_text(html_doc, encoding="utf-8")
    return md_path, html_path
