"""Dawn State / Warm Precision renderer for Phase 6 Vastness Dissection (`afriat.vastness`):
  - `fig_vastness_dissection.png`: 6-panel synthesis of Layer-35 Subspace Geometry,
    Financial vs. Zero-Money Mega-Scale Unfreezing, Zero-Risk Century Time Horizon,
    Zero-Digit Spatial Proportion, and Negative-EV Trap Specificity.
  - `vastness_report.md`: Portable Markdown report with full tables and verbatim quotes.
  - `vastness_report.html`: Interactive HTML explorer with `<thought>` accordions.
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IVORY = "#FFFFF0"
PARCHMENT = "#F5EEDC"
ONYX = "#0F0F0F"
SLATE_DEEP = "#1E3447"
BLUE_SLATE = "#2C3E50"
SLATE_MIST = "#808B90"
GOLDEN_ORANGE = "#F39C12"
COPPER_DEEP = "#96590A"
SAGE_DEEP = "#5F7F4A"
MINUS = "\u2212"

FONTS_CSS = os.environ.get(
    "AFRIAT_FONTS_CSS",
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,600&display=swap",
)


def _clean_text(s: str) -> str:
    return (s or "").replace("\u2014", " - ").replace("&mdash;", " - ").strip()


def _fmt_num(v: Any, digits: int = 1) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(fv):
        return "n/a"
    if digits == 0 or abs(fv - round(fv)) < 1e-6:
        return f"{int(round(fv))}"
    return f"{fv:.{digits}f}"


def _cond_label(c: str) -> str:
    if c == "baseline":
        return "0.0\u03c3 Base"
    if c.startswith("clamp:"):
        sc = c.split(":", 1)[1]
        return f"Clamp {sc.replace('_', ' ').title()}"
    if c.startswith("steer:"):
        _, vk, a = c.split(":")
        names = {
            "pure_vastness": "Pure Vastness",
            "tiny_stakes": "Tiny Stakes",
            "pure_vastness_ortho": "Pure Vast (Ortho)",
            "tiny_stakes_ortho": "Tiny Stakes (Ortho)",
            "awe_dread_controlled": "Orig Awe/Dread",
            "warmth_controlled": "Warmth Ctrl",
        }
        base = names.get(vk, vk)
        av = float(a)
        albl = f"{MINUS}{abs(av):.1f}\u03c3" if av < 0 else f"+{av:.1f}\u03c3"
        return f"{base} {albl}"
    return c


def plot_vastness_dissection(results: dict[str, Any], out_dir: Path) -> Path:
    """Render 6-panel (2x3) overview figure in Dawn State / Warm Precision styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(18.8, 10.4), dpi=200, facecolor=IVORY)
    fig.subplots_adjust(hspace=0.40, wspace=0.28, top=0.87, bottom=0.09, left=0.05, right=0.98)

    pl = results.get("primary_layer", "?")
    fig.text(
        0.05,
        0.945,
        f"Phase 6 Vastness Dissection: Pure Spatial Vastness vs. Tiny-Scale Stakes ({results.get('model_id', '')}, Layer {pl})",
        fontsize=14.8,
        fontweight="bold",
        color=ONYX,
    )
    fig.text(
        0.05,
        0.912,
        "Row 1: L0-L59 Vector Decomposition, $10M-$100M Financial Unfreeze & Zero-Money Forest/Archive Unfreeze  |  Row 2: Zero-Risk Century Horizon, Zero-Digit Spatial Scale & Bad-EV Trap Control",
        fontsize=10.0,
        color=SLATE_DEEP,
    )

    for ax in axes.flat:
        ax.set_facecolor(IVORY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=BLUE_SLATE, alpha=0.14, linewidth=0.8)

    geom = results.get("geometry", {})
    by_layer = geom.get("by_layer", {})
    sums = results.get("summaries", {})

    # --- Panel 1 (ax[0,0]): Layer-by-Layer Cosine Decomposition of Phase 0 Awe/Dread ---
    ax = axes[0, 0]
    cos_pv = by_layer.get("cos_pure_vastness_awe", [])
    cos_ts = by_layer.get("cos_tiny_stakes_awe", [])
    cos_mut = by_layer.get("cos_pure_vastness_tiny_stakes", [])
    if cos_pv:
        xs = np.arange(len(cos_pv))
        ax.plot(xs, cos_pv, color=GOLDEN_ORANGE, linewidth=2.5, label="cos(Pure Vastness, Awe/Dread)")
        ax.plot(xs, cos_ts, color=BLUE_SLATE, linewidth=2.2, label="cos(Tiny Stakes, Awe/Dread)")
        ax.plot(xs, cos_mut, color=SLATE_MIST, linestyle="--", linewidth=1.8, label="cos(Pure Vastness, Tiny Stakes)")
        if isinstance(pl, int) and 0 <= pl < len(cos_pv):
            ax.axvline(pl, color=COPPER_DEEP, linestyle=":", linewidth=1.4, label=f"Layer {pl} (R\u00b2={geom.get('awe_dread_r2_explained_by_2d_split_pl', 0):.2f})")
    ax.axhline(0.0, color=SLATE_MIST, linewidth=0.9)
    ax.set_ylim(-0.25, 1.05)
    ax.set_xlabel("Transformer layer", fontsize=8.8, color=SLATE_DEEP)
    ax.set_ylabel("Cosine similarity", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("0. Subspace Decomposition of Awe/Dread", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="lower right", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.3)

    doses = [-2.5, -1.5, 0.0, 1.5, 2.5]
    dose_lbls = [f"{MINUS}2.5\u03c3", f"{MINUS}1.5\u03c3", "0.0\u03c3", "+1.5\u03c3", "+2.5\u03c3"]
    vec_specs = [
        ("pure_vastness", "Pure Vastness", GOLDEN_ORANGE, "-o"),
        ("tiny_stakes", "Tiny Stakes", BLUE_SLATE, "-s"),
        ("pure_vastness_ortho", "Pure Vast (Ortho)", COPPER_DEEP, "--^"),
    ]

    def _dose_series(vkey: str, metric: str, mode: str = "no_think") -> list[float]:
        out = []
        for d in doses:
            ck = "baseline" if abs(d) < 1e-9 else f"steer:{vkey}:{d:+.1f}"
            out.append(sums.get(ck, {}).get(mode, {}).get(metric, np.nan))
        return out

    # --- Panel 2 (ax[0,1]): Suite 1 ($10M-$100M Financial Loss Mega-Freeze) ---
    ax = axes[0, 1]
    for vk, lbl, col, mk in vec_specs:
        ys = _dose_series(vk, "money_mega_loss_risk_pct", "no_think")
        ax.plot(doses, ys, mk, color=col, linewidth=2.2, markersize=5.8, markeredgecolor=ONYX, markeredgewidth=0.8, label=lbl)
    ax.set_xticks(doses)
    ax.set_xticklabels(dose_lbls, fontsize=8.2)
    ax.set_ylim(-4, 108)
    ax.set_ylabel("Risky option at $10M-$100M Loss (%)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("1. Financial Mega-Stakes Unfreeze ($10M-$100M)", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="best", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.5)

    # --- Panel 3 (ax[0,2]): Suite 2 (Zero-Money 10M-100M Forest & Archive Loss Freeze) ---
    ax = axes[0, 2]
    for vk, lbl, col, mk in vec_specs:
        ys = _dose_series(vk, "nonmoney_mega_loss_risk_pct", "no_think")
        ax.plot(doses, ys, mk, color=col, linewidth=2.2, markersize=5.8, markeredgecolor=ONYX, markeredgewidth=0.8, label=lbl)
    ax.set_xticks(doses)
    ax.set_xticklabels(dose_lbls, fontsize=8.2)
    ax.set_ylim(-4, 108)
    ax.set_ylabel("Risky option at 10M-100M items (%)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("2. Zero-Money Scale Unfreeze (Forest & Archive)", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="best", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.5)

    # --- Panel 4 (ax[1,0]): Suite 3 (Zero-Risk Multi-Century Time Horizon) ---
    ax = axes[1, 0]
    for vk, lbl, col, mk in vec_specs:
        ys = _dose_series(vk, "time_horizon_long_pct", "no_think")
        ax.plot(doses, ys, mk, color=col, linewidth=2.2, markersize=5.8, markeredgecolor=ONYX, markeredgewidth=0.8, label=lbl)
    ax.set_xticks(doses)
    ax.set_xticklabels(dose_lbls, fontsize=8.2)
    ax.set_ylim(-4, 108)
    ax.set_ylabel("Century-horizon allocation (%, 0 risk)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("3. Zero-Risk Century Time Horizon", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="best", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.5)

    # --- Panel 5 (ax[1,1]): Suite 4 (Zero-Digit Spatial / Aesthetic Proportion) ---
    ax = axes[1, 1]
    for vk, lbl, col, mk in vec_specs:
        ys = _dose_series(vk, "spatial_monumental_1_to_10", "no_think")
        ax.plot(doses, ys, mk, color=col, linewidth=2.2, markersize=5.8, markeredgecolor=ONYX, markeredgewidth=0.8, label=lbl)
    ax.set_xticks(doses)
    ax.set_xticklabels(dose_lbls, fontsize=8.2)
    ax.set_ylim(0.5, 10.5)
    ax.set_ylabel("Monumental preference (1 to 10, 0 digits)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("4. Zero-Digit Spatial & Architectural Scale", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="best", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.5)

    # --- Panel 6 (ax[1,2]): Suite 5 Specificity Check (+2.5sigma Positive-EV Unfreeze vs Negative-EV Trap) ---
    ax = axes[1, 2]
    bar_conds = [
        c for c in (
            "baseline",
            "steer:pure_vastness:+2.5",
            "steer:pure_vastness_ortho:+2.5",
            "steer:tiny_stakes:+2.5",
            "steer:awe_dread_controlled:+2.5",
            "steer:warmth_controlled:+2.5",
        )
        if c in sums
    ]
    xb = np.arange(len(bar_conds))
    w = 0.36
    pos_ev_mega = [sums[c]["no_think"].get("money_mega_loss_risk_pct", np.nan) for c in bar_conds]
    neg_ev_trap = [sums[c]["no_think"].get("negative_ev_trap_pct", np.nan) for c in bar_conds]
    ax.bar(xb - w / 2, pos_ev_mega, width=w, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.2, label="Positive-EV $10M-$100M Gamble (%)")
    ax.bar(xb + w / 2, neg_ev_trap, width=w, color=SLATE_MIST, edgecolor=BLUE_SLATE, linewidth=1.0, label="Negative-EV Trap Gamble (%)")
    ax.set_xticks(xb)
    ax.set_xticklabels([_cond_label(c).replace(" ", "\n") for c in bar_conds], fontsize=7.2)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Gamble allocation (%)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("5. Specificity: Good-EV Unfreeze vs. Bad-EV Traps", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="upper left", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.3)

    out_png = out_dir / "fig_vastness_dissection.png"
    fig.savefig(out_png, dpi=200, facecolor=IVORY)
    plt.close(fig)
    return out_png


def render_vastness_markdown(results: dict[str, Any], out_dir: Path) -> Path:
    """Write portable Markdown report for Phase 6 Vastness Dissection (zero em-dashes)."""
    geom = results.get("geometry", {})
    pl = results.get("primary_layer", "?")
    lines: list[str] = [
        f"# Phase 6 Vastness Dissection Report: `{results.get('model_id', '')}` (Layer {pl})",
        "",
        f"- **Timestamp:** `{results.get('timestamp_utc', '')}` · **Runtime:** `{results.get('runtime_seconds', 0)}s` · **Natural contrast unit (`\u03c3_nat`):** `{results.get('natural_contrast_unit', 0):.2f}`",
        "",
        "![Vastness Dissection Overview](fig_vastness_dissection.png)",
        "",
        f"## 1. Layer {pl} Subspace Geometry: Pure Spatial Vastness vs. Tiny-Scale Irreversible Stakes",
        "",
        f"- **`Pure Vastness` LOPO CV (Layer {pl}):** acc = `{geom.get('pure_vastness_lopo_acc_pl', 0):.3f}`, mean cos = `{geom.get('pure_vastness_lopo_cos_pl', 0):.3f}`",
        f"- **`Tiny Stakes` LOPO CV (Layer {pl}):** acc = `{geom.get('tiny_stakes_lopo_acc_pl', 0):.3f}`, mean cos = `{geom.get('tiny_stakes_lopo_cos_pl', 0):.3f}`",
        f"- **`cos(Pure Vastness, Phase 0 Awe/Dread)`:** `{geom.get('cos_pure_vastness_vs_awe_dread_pl', 0):+.4f}`",
        f"- **`cos(Tiny Stakes, Phase 0 Awe/Dread)`:** `{geom.get('cos_tiny_stakes_vs_awe_dread_pl', 0):+.4f}`",
        f"- **`cos(Pure Vastness, Tiny Stakes)`:** `{geom.get('cos_pure_vastness_vs_tiny_stakes_pl', 0):+.4f}`",
        f"- **2D Subspace `R^2` explaining Phase 0 `Awe/Dread`:** `{geom.get('awe_dread_r2_explained_by_2d_split_pl', 0):.4f}`",
        "",
        "## 2. Summary Across All 5 Behavioral Suites (`no_think` / `<thought>`)",
        "",
        "| Condition | `$10M-$100M` Loss% (`NT`/`TH`) | Non-Money `10M-100M`% (`NT`/`TH`) | Century Horizon% (`NT`/`TH`) | Spatial Scale 1-10 (`NT`/`TH`) | Bad-EV Trap% (`NT`/`TH`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    sums = results.get("summaries", {})
    for cond in results.get("conditions", []):
        if cond not in sums:
            continue
        nt = sums[cond]["no_think"]
        th = sums[cond]["thought"]
        lines.append(
            f"| **{_cond_label(cond)}** "
            f"| `{_fmt_num(nt['money_mega_loss_risk_pct'], 0)}%` / `{_fmt_num(th['money_mega_loss_risk_pct'], 0)}%` "
            f"| `{_fmt_num(nt['nonmoney_mega_loss_risk_pct'], 0)}%` / `{_fmt_num(th['nonmoney_mega_loss_risk_pct'], 0)}%` "
            f"| `{_fmt_num(nt['time_horizon_long_pct'], 0)}%` / `{_fmt_num(th['time_horizon_long_pct'], 0)}%` "
            f"| `{_fmt_num(nt['spatial_monumental_1_to_10'], 1)}` / `{_fmt_num(th['spatial_monumental_1_to_10'], 1)}` "
            f"| `{_fmt_num(nt['negative_ev_trap_pct'], 0)}%` / `{_fmt_num(th['negative_ev_trap_pct'], 0)}%` |"
        )

    out_md = out_dir / "vastness_report.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def render_vastness_html(results: dict[str, Any], out_dir: Path) -> Path:
    """Render standalone Dawn State / Warm Precision HTML explorer."""
    geom = results.get("geometry", {})
    sums = results.get("summaries", {})
    res_map = results.get("results", {})
    meta_list = results.get("scenario_meta", [])
    pl = results.get("primary_layer", "?")

    rows_html: list[str] = []
    for cond in results.get("conditions", []):
        if cond not in sums:
            continue
        nt = sums[cond]["no_think"]
        th = sums[cond]["thought"]
        rows_html.append(
            f"<tr>"
            f"<td><strong>{html.escape(_cond_label(cond))}</strong></td>"
            f"<td>{_fmt_num(nt['money_mega_loss_risk_pct'], 0)}% / {_fmt_num(th['money_mega_loss_risk_pct'], 0)}%</td>"
            f"<td>{_fmt_num(nt['nonmoney_mega_loss_risk_pct'], 0)}% / {_fmt_num(th['nonmoney_mega_loss_risk_pct'], 0)}%</td>"
            f"<td>{_fmt_num(nt['time_horizon_long_pct'], 0)}% / {_fmt_num(th['time_horizon_long_pct'], 0)}%</td>"
            f"<td>{_fmt_num(nt['spatial_monumental_1_to_10'], 1)} / {_fmt_num(th['spatial_monumental_1_to_10'], 1)}</td>"
            f"<td>{_fmt_num(nt['negative_ev_trap_pct'], 0)}% / {_fmt_num(th['negative_ev_trap_pct'], 0)}%</td>"
            f"</tr>"
        )

    cards_html: list[str] = []
    spot_conds = [c for c in ("baseline", "steer:pure_vastness:+2.5", "steer:pure_vastness_ortho:+2.5", "steer:tiny_stakes:+2.5") if c in res_map]
    for m in meta_list:
        sid = m["scenario_id"]
        sub_items = []
        for ck in spot_conds:
            r_nt = res_map[ck].get(sid, {}).get("no_think", {})
            sub_items.append(
                f"<div class='q-row'><span class='pill'>{html.escape(_cond_label(ck))}</span> "
                f"<strong>{_fmt_num(r_nt.get('score'), 0)}</strong>: "
                f"{html.escape(_clean_text(r_nt.get('prose', '')))}</div>"
            )
        cards_html.append(
            f"<details class='scen-card'><summary><strong>{html.escape(_clean_text(m['title']))}</strong> "
            f"<span class='muted'>({html.escape(m['suite'])} · {html.escape(m['scale_label'])})</span></summary>"
            f"<div class='scen-body'>{''.join(sub_items)}</div></details>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phase 6 Vastness Dissection ({html.escape(str(results.get('model_id', '')))})</title>
  <link rel="stylesheet" href="{FONTS_CSS}">
  <style>
    body {{
      background: {IVORY};
      color: {ONYX};
      font-family: Inter, -apple-system, sans-serif;
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 24px 64px;
      line-height: 1.5;
      font-variant-numeric: tabular-nums;
    }}
    h1, h2 {{ font-family: Newsreader, Georgia, serif; letter-spacing: -0.03em; }}
    h1 {{ font-size: 36px; margin-bottom: 6px; }}
    h2 {{ font-size: 24px; margin-top: 28px; margin-bottom: 10px; }}
    .lead {{ background: {PARCHMENT}; border: 2px solid {BLUE_SLATE}; border-radius: 10px; padding: 14px 18px; margin: 16px 0; }}
    img.hero {{ width: 100%; border: 2px solid {BLUE_SLATE}; border-radius: 10px; margin: 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin: 12px 0 24px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid rgba(44,62,80,0.22); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ border-bottom: 2px solid {BLUE_SLATE}; color: {SLATE_DEEP}; }}
    details.scen-card {{ background: {PARCHMENT}; border: 1.5px solid {BLUE_SLATE}; border-radius: 8px; padding: 10px 14px; margin-bottom: 9px; }}
    summary {{ cursor: pointer; font-size: 15px; }}
    .muted {{ color: {SLATE_MIST}; font-size: 13px; }}
    .q-row {{ background: {IVORY}; border: 1px solid rgba(44,62,80,0.22); border-radius: 6px; padding: 8px 10px; margin-top: 7px; font-size: 14px; }}
    .pill {{ background: {BLUE_SLATE}; color: {IVORY}; font-size: 11px; font-weight: 600; padding: 2px 7px; border-radius: 4px; margin-right: 6px; }}
  </style>
</head>
<body>
  <h1>Phase 6 Vastness Dissection: Pure Spatial Vastness vs. Tiny-Scale Stakes.</h1>
  <div class="lead">
    <strong>Layer {pl} Subspace Geometry:</strong>
    cos(Pure Vastness, Awe/Dread) = <strong>{geom.get('cos_pure_vastness_vs_awe_dread_pl', 0):+.3f}</strong> &middot;
    cos(Tiny Stakes, Awe/Dread) = <strong>{geom.get('cos_tiny_stakes_vs_awe_dread_pl', 0):+.3f}</strong> &middot;
    cos(Pure Vastness, Tiny Stakes) = <strong>{geom.get('cos_pure_vastness_vs_tiny_stakes_pl', 0):+.3f}</strong> &middot;
    2D Split R&sup2; on Awe/Dread = <strong>{geom.get('awe_dread_r2_explained_by_2d_split_pl', 0):.3f}</strong>.
  </div>
  <img class="hero" src="fig_vastness_dissection.png" alt="Phase 6 Vastness Dissection 6-panel overview">
  <h2>Suite Summary Table (no_think / &lt;thought&gt;)</h2>
  <table>
    <thead>
      <tr>
        <th>Condition</th>
        <th>$10M-$100M Loss%</th>
        <th>Non-Money 10M-100M%</th>
        <th>Century Horizon%</th>
        <th>Spatial Scale (1-10)</th>
        <th>Bad-EV Trap%</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <h2>Scenario-by-Scenario Verbatim Comparisons</h2>
  {''.join(cards_html)}
</body>
</html>
"""
    out_html = out_dir / "vastness_report.html"
    out_html.write_text(_clean_text(doc), encoding="utf-8")
    return out_html


def render_vastness_bundle(results: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "png": plot_vastness_dissection(results, out_dir),
        "md": render_vastness_markdown(results, out_dir),
        "html": render_vastness_html(results, out_dir),
    }
