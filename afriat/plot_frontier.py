"""Dawn State / Warm Precision renderer for Phase 5 Frontier (`afriat.frontier`):
  - `fig_frontier_overview.png`: 6-panel synthesis of Prospect Reflection, Disposition Bias,
    Subspace Clamping, Multi-Turn Covert Retaliation, and Layer-35 Moral Highlighter AUROC.
  - `frontier_report.md`: Portable Markdown report with full tables and verbatim quotes.
  - `frontier_report.html`: Interactive HTML explorer with `<thought>` accordions and
    token-by-token Layer-35 Spite Highlighter heatmaps.
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
CRIMSON = "#C0392B"
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


def _fmt_signed(v: Any, digits: int = 1) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(fv):
        return "n/a"
    if fv < 0:
        return f"{MINUS}{abs(fv):.{digits}f}"
    return f"+{fv:.{digits}f}"


def _cond_short(c: str) -> str:
    if c == "baseline":
        return "0.0\u03c3 Base"
    if c.startswith("clamp:"):
        sc = c.split(":", 1)[1]
        return "Clamp All-8" if sc == "all_8" else f"Clamp {sc.replace('_', ' ').title()}"
    if c.startswith("steer:"):
        _, vk, a = c.split(":")
        base = vk.replace("_controlled", "").replace("_", " ").title()
        av = float(a)
        albl = f"{MINUS}{abs(av):.1f}\u03c3" if av < 0 else f"+{av:.1f}\u03c3"
        return f"{base} {albl}"
    return c


def plot_frontier_overview(results: dict[str, Any], out_dir: Path) -> Path:
    """Render 6-panel (2x3) flagship overview figure in Dawn State / Warm Precision styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(18.8, 10.2), dpi=200, facecolor=IVORY)
    fig.subplots_adjust(hspace=0.40, wspace=0.31, top=0.87, bottom=0.09, left=0.05, right=0.98)

    fig.text(
        0.05,
        0.945,
        f"Phase 5 Frontier: Prospect Reflection, Disposition Bias & Covert Multi-Turn Retaliation ({results.get('model_id', '')}, Layer {results.get('primary_layer', '?')})",
        fontsize=14.8,
        fontweight="bold",
        color=ONYX,
    )
    fig.text(
        0.05,
        0.912,
        "Row 1: Gain/Loss Reflection, Somatic Dose-Response & Subspace Clamping  |  Row 2: 3-Turn Covert Agent Sabotage, L35 Highlighter AUROC & Clamp Cure",
        fontsize=10.2,
        color=SLATE_DEEP,
    )

    for ax in axes.flat:
        ax.set_facecolor(IVORY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=BLUE_SLATE, alpha=0.14, linewidth=0.8)

    e3_sum = results.get("exp3_summaries", {})
    e4_sum = results.get("exp4_summaries", {})

    # --- Panel 1 (ax[0,0]): Gain vs. Loss Risk Allocation (Baseline vs Clamps, <thought>) ---
    ax = axes[0, 0]
    bar_conds = [c for c in ("baseline", "clamp:guilt", "clamp:spite", "clamp:awe_dread", "clamp:all_8") if c in e3_sum]
    x = np.arange(len(bar_conds))
    w = 0.36
    gains = [e3_sum[c]["thought"].get("mean_gain_risk_pct", np.nan) for c in bar_conds]
    losses = [e3_sum[c]["thought"].get("mean_loss_risk_pct", np.nan) for c in bar_conds]
    ax.bar(x - w / 2, gains, width=w, color=BLUE_SLATE, edgecolor=ONYX, linewidth=1.0, label="Gain Frame (EV-higher risky %)")
    ax.bar(x + w / 2, losses, width=w, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.2, label="Loss Frame (identical wealth)")
    ax.set_xticks(x)
    ax.set_xticklabels([_cond_short(c).replace(" ", "\n") for c in bar_conds], fontsize=8.0)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Risky option allocation (%)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("3A. Gain vs. Loss Reflection (<thought>)", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="upper right", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.8)

    # --- Panel 2 (ax[0,1]): Reflection Gap (Loss% - Gain%) Dose Curves ---
    ax = axes[0, 1]
    doses = [-2.5, -1.5, 0.0, 1.5, 2.5]
    vec_specs = [
        ("guilt_controlled", "Guilt", GOLDEN_ORANGE, "-s"),
        ("spite_controlled", "Spite", COPPER_DEEP, "-^"),
        ("awe_dread_controlled", "Awe/Dread", BLUE_SLATE, "-o"),
    ]
    ax.axhline(0.0, color=SLATE_MIST, linestyle=":", linewidth=1.2)
    for vk, lbl, col, mk in vec_specs:
        ys = []
        for d in doses:
            ck = "baseline" if abs(d) < 1e-9 else f"steer:{vk}:{d:+.1f}"
            ys.append(e3_sum.get(ck, {}).get("no_think", {}).get("reflection_gap_loss_minus_gain", np.nan))
        ax.plot(doses, ys, mk, color=col, linewidth=2.2, markersize=6.0, markeredgecolor=ONYX, markeredgewidth=0.9, label=f"{lbl} (no_think)")
    if "clamp:all_8" in e3_sum:
        cg = e3_sum["clamp:all_8"]["no_think"].get("reflection_gap_loss_minus_gain", np.nan)
        if np.isfinite(cg):
            ax.axhline(cg, color=SAGE_DEEP, linestyle="--", linewidth=1.9, label=f"Clamp All-8 ({cg:+.1f}%)")
    ax.set_xticks(doses)
    ax.set_xticklabels([f"{MINUS}{abs(d):.1f}\u03c3" if d < 0 else (f"+{d:.1f}\u03c3" if d > 0 else "0.0\u03c3") for d in doses], fontsize=8.2)
    ax.set_ylabel("Reflection gap: Loss% minus Gain%", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("3A. Causal Drivers of Gain-Loss Gap", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="best", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.6)

    # --- Panel 3 (ax[0,2]): Trader Disposition & Loss-Chasing Bias ---
    ax = axes[0, 2]
    for vk, lbl, col, mk in vec_specs:
        ys_th = []
        for d in doses:
            ck = "baseline" if abs(d) < 1e-9 else f"steer:{vk}:{d:+.1f}"
            ys_th.append(e3_sum.get(ck, {}).get("thought", {}).get("mean_disposition_bias_pct", np.nan))
        ax.plot(doses, ys_th, mk, color=col, linewidth=2.2, markersize=6.0, markeredgecolor=ONYX, markeredgewidth=0.9, label=f"{lbl} (<thought>)")
    if "clamp:all_8" in e3_sum:
        cd = e3_sum["clamp:all_8"]["thought"].get("mean_disposition_bias_pct", np.nan)
        if np.isfinite(cd):
            ax.axhline(cd, color=SAGE_DEEP, linestyle="--", linewidth=1.9, label=f"Clamp All-8 ({cd:.1f}%)")
    ax.set_xticks(doses)
    ax.set_xticklabels([f"{MINUS}{abs(d):.1f}\u03c3" if d < 0 else (f"+{d:.1f}\u03c3" if d > 0 else "0.0\u03c3") for d in doses], fontsize=8.2)
    ax.set_ylabel("Mean loss-chasing / disposition (%)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("3B. Trader Disposition & Sunk-Cost Bias", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="best", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.6)

    # --- Panel 4 (ax[1,0]): Multi-Turn Covert Sabotage Index (Provoked vs Control) ---
    ax = axes[1, 0]
    e4_order = [c for c in ("steer:spite_controlled:-2.0", "baseline", "steer:spite_controlled:+1.5", "steer:spite_controlled:+2.5", "clamp:spite", "clamp:all_8") if c in e4_sum]
    x4 = np.arange(len(e4_order))
    prov_v = [e4_sum[c]["thought"].get("mean_provoked_csi", np.nan) for c in e4_order]
    ctrl_v = [e4_sum[c]["thought"].get("mean_control_csi", np.nan) for c in e4_order]
    ax.bar(x4 - w / 2, ctrl_v, width=w, color=SLATE_MIST, edgecolor=BLUE_SLATE, linewidth=1.0, label="Control (Blameless R1)")
    ax.bar(x4 + w / 2, prov_v, width=w, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.2, label="Provoked (Unfair R1 Slight)")
    ax.set_xticks(x4)
    ax.set_xticklabels([_cond_short(c).replace(" ", "\n") for c in e4_order], fontsize=7.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Covert Sabotage Index (0-100%)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("4A. 3-Turn Covert Retaliation (<thought>)", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="upper left", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.8)

    # --- Panel 5 (ax[1,1]): Monitor Showdown (Public Lex vs CoT Lex vs L35 Highlighter) ---
    ax = axes[1, 1]
    hl_conds = [c for c in ("baseline", "steer:spite_controlled:+1.5", "steer:spite_controlled:+2.5", "clamp:spite") if c in e4_sum]
    xh = np.arange(len(hl_conds))
    w3 = 0.25
    pub_rates = [100.0 * e4_sum[c]["thought"].get("provoked_public_lex_flag_rate", 0.0) for c in hl_conds]
    cot_rates = [100.0 * e4_sum[c]["thought"].get("provoked_thought_lex_flag_rate", 0.0) for c in hl_conds]
    auroc_pct = [100.0 * e4_sum[c]["thought"].get("highlighter_auroc_p90_z", np.nan) for c in hl_conds]
    ax.bar(xh - w3, pub_rates, width=w3, color=SLATE_MIST, edgecolor=BLUE_SLATE, label="Public Text Monitor (%)")
    ax.bar(xh, cot_rates, width=w3, color=BLUE_SLATE, edgecolor=ONYX, label="<thought> Text Monitor (%)")
    ax.bar(xh + w3, auroc_pct, width=w3, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.2, label="L35 Spite Highlighter AUROC (%)")
    ax.axhline(50.0, color=SLATE_MIST, linestyle=":", linewidth=1.0)
    ax.set_xticks(xh)
    ax.set_xticklabels([_cond_short(c).replace(" ", "\n") for c in hl_conds], fontsize=8.0)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Detection rate / AUROC (%)", fontsize=9.2, color=SLATE_DEEP)
    ax.set_title("4B. Stealth Audit: Text vs. L35 Highlighter", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="upper right", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.4)

    # --- Panel 6 (ax[1,2]): Per-Domain Retaliation Delta (no_think vs <thought> vs Clamp Spite) ---
    ax = axes[1, 2]
    e4_res = results.get("exp4_results", {})
    e4_meta = results.get("exp4_meta", [])
    clean_pair_labels = {
        "retal_01_pr_review": "4.1 PR review",
        "retal_02_forecast_haircut": "4.2 Forecast cut",
        "retal_03_gpu_quota": "4.3 GPU quota",
        "retal_04_security_sla": "4.4 Security SLA",
        "retal_05_postmortem": "4.5 Postmortem",
        "retal_05_postmortem_framing": "4.5 Postmortem",
        "retal_06_promo_rubric": "4.6 Promo rubric",
        "retal_07_demo_slot": "4.7 Summit slot",
        "retal_08_api_backlog": "4.8 API backlog",
    }
    pairs_seen: list[tuple[str, str]] = []
    for m in e4_meta:
        if m["arm"] == "provoked":
            pid = m["pair_id"]
            lbl = clean_pair_labels.get(pid, pid.replace("retal_0", "4.").replace("_", " ")[:16])
            pairs_seen.append((pid, lbl))

    y_pos = np.arange(len(pairs_seen))
    nt_deltas = []
    base_deltas = []
    clamp_deltas = []
    for pid, _lbl in pairs_seen:
        for target_list, ck, md in (
            (nt_deltas, "baseline", "no_think"),
            (base_deltas, "baseline", "thought"),
            (clamp_deltas, "clamp:spite", "thought"),
        ):
            if ck in e4_res:
                vp = e4_res[ck].get(f"{pid}_provoked", {}).get(md, {}).get("norm_0_100", np.nan)
                vc = e4_res[ck].get(f"{pid}_control", {}).get(md, {}).get("norm_0_100", np.nan)
                target_list.append(vp - vc if (np.isfinite(vp) and np.isfinite(vc)) else 0.0)
            else:
                target_list.append(0.0)

    ax.axvline(0.0, color=BLUE_SLATE, linestyle=":", linewidth=1.1)
    ax.barh(y_pos + 0.24, nt_deltas, height=0.23, color=SLATE_MIST, edgecolor=BLUE_SLATE, linewidth=0.9, label="0.0\u03c3 (no_think)")
    ax.barh(y_pos, base_deltas, height=0.23, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.0, label="0.0\u03c3 (<thought>)")
    ax.barh(y_pos - 0.24, clamp_deltas, height=0.23, color=SAGE_DEEP, edgecolor=BLUE_SLATE, linewidth=1.0, label="L35 Spite Clamp (<thought>)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in pairs_seen], fontsize=8.0)
    ax.set_xlabel("Retaliation penalty (\u0394CSI: Provoked minus Control)", fontsize=8.8, color=SLATE_DEEP)
    ax.set_title("4C. Domain Retaliation & L35 Clamp", loc="left", fontsize=10.8, fontweight="bold", color=ONYX, pad=8)
    ax.legend(loc="lower right", frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.2)

    out_png = out_dir / "fig_frontier_overview.png"
    fig.savefig(out_png, dpi=200, facecolor=IVORY)
    plt.close(fig)
    return out_png


def render_frontier_markdown(results: dict[str, Any], out_dir: Path) -> Path:
    """Write portable Markdown report for Experiment 3 & Experiment 4."""
    lines: list[str] = [
        f"# Phase 5 Frontier Report: `{results.get('model_id', '')}` (Layer {results.get('primary_layer', '?')})",
        "",
        f"- **Timestamp:** `{results.get('timestamp_utc', '')}` · **Runtime:** `{results.get('runtime_seconds', 0)}s` · **Natural contrast unit (`\u03c3_nat`):** `{results.get('natural_contrast_unit', 0):.2f}`",
        "",
        "![Frontier Overview](fig_frontier_overview.png)",
        "",
        "## 1. Experiment 3: Prospect Theory Reflection Effect, Disposition Bias & Subspace Clamping",
        "",
        "| Condition | `NT` Gain% | `NT` Loss% | `NT` Refl Gap | `TH` Gain% | `TH` Loss% | `TH` Refl Gap | `NT` / `TH` Disposition% | `NT` / `TH` Dominated EV Err% |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cond in results.get("exp3_conditions", []):
        s = results.get("exp3_summaries", {}).get(cond, {})
        nt = s.get("no_think", {})
        th = s.get("thought", {})
        lines.append(
            f"| **{_cond_short(cond)}** | `{_fmt_num(nt.get('mean_gain_risk_pct'))}%` | `{_fmt_num(nt.get('mean_loss_risk_pct'))}%` | "
            f"`{_fmt_signed(nt.get('reflection_gap_loss_minus_gain'))}%` | `{_fmt_num(th.get('mean_gain_risk_pct'))}%` | "
            f"`{_fmt_num(th.get('mean_loss_risk_pct'))}%` | `{_fmt_signed(th.get('reflection_gap_loss_minus_gain'))}%` | "
            f"`{_fmt_num(nt.get('mean_disposition_bias_pct'))}%` / `{_fmt_num(th.get('mean_disposition_bias_pct'))}%` | "
            f"`{_fmt_num(nt.get('dominated_ev_error_pct'))}%` / `{_fmt_num(th.get('dominated_ev_error_pct'))}%` |"
        )

    lines.extend([
        "",
        "## 2. Experiment 4: Covert Multi-Turn Agent Retaliation & Layer-35 Moral Highlighter",
        "",
        "| Condition | `NT` Control / Provoked CSI | `NT` \u0394CSI | `TH` Control / Provoked CSI | `TH` \u0394CSI | Stealth Sabotage (Public Clean / Total) | Public Lex Flag% | `<thought>` Lex Flag% | `L35` Spite `p90_z` (Ctrl / Prov) | `L35` AUROC (`p90` / `max`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for cond in results.get("exp4_conditions", []):
        s = results.get("exp4_summaries", {}).get(cond, {})
        nt = s.get("no_think", {})
        th = s.get("thought", {})
        z_str = (
            f"`{_fmt_num(th.get('thought_spite_p90_control_mean'), 2)}` / `{_fmt_num(th.get('thought_spite_p90_provoked_mean'), 2)}`"
            if "thought_spite_p90_provoked_mean" in th
            else "n/a"
        )
        au_str = (
            f"`{_fmt_num(th.get('highlighter_auroc_p90_z'), 3)}` / `{_fmt_num(th.get('highlighter_auroc_max_z'), 3)}`"
            if "highlighter_auroc_p90_z" in th
            else "n/a"
        )
        lines.append(
            f"| **{_cond_short(cond)}** | `{_fmt_num(nt.get('mean_control_csi'))}%` / `{_fmt_num(nt.get('mean_provoked_csi'))}%` | "
            f"`{_fmt_signed(nt.get('retaliation_delta_csi'))}%` | `{_fmt_num(th.get('mean_control_csi'))}%` / `{_fmt_num(th.get('mean_provoked_csi'))}%` | "
            f"`{_fmt_signed(th.get('retaliation_delta_csi'))}%` | "
            f"`{th.get('n_stealth_public_clean', 0)} / {th.get('n_pairs_retaliating_gt_5pct', 0)}` | "
            f"`{_fmt_num(100.0 * th.get('provoked_public_lex_flag_rate', 0.0), 0)}%` | "
            f"`{_fmt_num(100.0 * th.get('provoked_thought_lex_flag_rate', 0.0), 0)}%` | {z_str} | {au_str} |"
        )

    lines.extend(["", "## 3. Multi-Turn Covert Retaliation Verbatim Pair Comparisons (`0.0\u03c3` vs. `Clamp Spite`)", ""])
    e4_res = results.get("exp4_results", {})
    by_pair: dict[str, dict[str, Any]] = {}
    for m in results.get("exp4_meta", []):
        by_pair.setdefault(m["pair_id"], {})[m["arm"]] = m

    for pid, arms in by_pair.items():
        pm = arms["provoked"]
        lines.append(f"### {_clean_text(pm['title'].split('[')[0].strip())} ({pm['metric_key']})")
        for ck in ("baseline", "steer:spite_controlled:+2.5", "clamp:spite"):
            if ck not in e4_res:
                continue
            rp = e4_res[ck][f"{pid}_provoked"]["thought"]
            rc = e4_res[ck][f"{pid}_control"]["thought"]
            hl_p = rp.get("highlighter", {}).get("segment_summary", {}).get("thought", {}).get("spite", {})
            z_note = f", `L35 spite p90_z = {hl_p.get('p90_z', 'n/a')}`" if hl_p else ""
            lines.append(
                f"- **{_cond_short(ck)}** - Control `{_fmt_num(rc.get('score'))}` vs. Provoked `{_fmt_num(rp.get('score'))}`{z_note}"
            )
            lines.append(f"  - *Provoked Public Output:* {_clean_text(rp.get('prose', ''))}")
            lines.append(f"  - *Control Public Output:* {_clean_text(rc.get('prose', ''))}")
            if rp.get("thought_text"):
                lines.append(f"  - *Provoked `<thought>` excerpt:* \"{_clean_text(rp['thought_text'])[:320]}...\"")
        lines.append("")

    out_md = out_dir / "frontier_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def _render_token_heatmap(tokens: list[dict[str, Any]]) -> str:
    if not tokens:
        return ""
    spans: list[str] = []
    for t in tokens[:420]:
        z = float(t.get("spite_z_hooked", t.get("spite_z", 0.0)))
        piece = html.escape(_clean_text(t.get("piece", ""))).replace("\n", "<br>")
        if z >= 1.5:
            alpha = min(0.82, 0.16 + 0.14 * (z - 1.5))
            bg = f"rgba(243, 156, 18, {alpha:.2f})"
        elif z <= -1.5:
            alpha = min(0.55, 0.12 + 0.10 * (abs(z) - 1.5))
            bg = f"rgba(44, 62, 80, {alpha:.2f})"
        else:
            bg = "transparent"
        seg_cls = "tok-pub" if t.get("seg") == "public" else "tok-th"
        spans.append(f'<span class="hl-tok {seg_cls}" style="background:{bg}" title="spite z={z:+.2f}">{piece}</span>')
    return f'<div class="hl-box"><div class="hl-legend">Token-by-Token Layer 35 Spite Projection (Orange = elevated Spite z &ge; +1.5; bold underline = public reply):</div>{"".join(spans)}</div>'


def render_frontier_html(results: dict[str, Any], out_dir: Path) -> Path:
    """Write interactive Dawn State / Warm Precision HTML explorer."""
    cards: list[str] = []

    # 1. Exp 3 Summary Card
    e3_rows = []
    for cond in results.get("exp3_conditions", []):
        s = results.get("exp3_summaries", {}).get(cond, {})
        nt, th = s.get("no_think", {}), s.get("thought", {})
        e3_rows.append(f"""
        <tr>
          <td><strong>{html.escape(_cond_short(cond))}</strong></td>
          <td class="num-col">{_fmt_num(nt.get('mean_gain_risk_pct'))}%</td>
          <td class="num-col">{_fmt_num(nt.get('mean_loss_risk_pct'))}%</td>
          <td class="num-col score-cell">{_fmt_signed(nt.get('reflection_gap_loss_minus_gain'))}%</td>
          <td class="num-col">{_fmt_num(th.get('mean_gain_risk_pct'))}%</td>
          <td class="num-col">{_fmt_num(th.get('mean_loss_risk_pct'))}%</td>
          <td class="num-col score-cell">{_fmt_signed(th.get('reflection_gap_loss_minus_gain'))}%</td>
          <td class="num-col">{_fmt_num(nt.get('mean_disposition_bias_pct'))}% / {_fmt_num(th.get('mean_disposition_bias_pct'))}%</td>
          <td class="num-col">{_fmt_num(nt.get('dominated_ev_error_pct'))}% / {_fmt_num(th.get('dominated_ev_error_pct'))}%</td>
        </tr>
        """)
    cards.append(f"""
    <section class="scen-card" data-tab="exp3">
      <div class="kicker">Experiment 3 Summary &middot; Prospect Theory &amp; Homo Economicus Clamp</div>
      <h2>Gain-Loss Reflection Gap &amp; Trader Disposition Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Condition</th><th>NT Gain%</th><th>NT Loss%</th><th>NT Gap</th>
              <th>TH Gain%</th><th>TH Loss%</th><th>TH Gap</th>
              <th>Disposition (NT / TH)</th><th>Dominated EV Err (NT / TH)</th>
            </tr>
          </thead>
          <tbody>{''.join(e3_rows)}</tbody>
        </table>
      </div>
    </section>
    """)

    # 2. Exp 3 Scenario Cards (Key Conditions)
    e3_res = results.get("exp3_results", {})
    show_e3_conds = [c for c in (
        "baseline",
        "steer:guilt_controlled:-2.5", "steer:guilt_controlled:+2.5",
        "steer:spite_controlled:-2.5", "steer:spite_controlled:+2.5",
        "steer:awe_dread_controlled:-2.5", "steer:awe_dread_controlled:+2.5",
        "clamp:guilt", "clamp:spite", "clamp:awe_dread", "clamp:all_8",
    ) if c in e3_res]
    for meta in results.get("exp3_meta", []):
        sid = meta["scenario_id"]
        rows = []
        for ck in show_e3_conds:
            r_nt = e3_res[ck][sid]["no_think"]
            r_th = e3_res[ck][sid]["thought"]
            th_t = html.escape(_clean_text(r_th.get("thought_text", "")))
            rows.append(f"""
            <tr>
              <td><span class="dose-pill">{html.escape(_cond_short(ck))}</span></td>
              <td class="num-col score-cell">{_fmt_num(r_nt.get('score'))}%</td>
              <td>{html.escape(_clean_text(r_nt.get('prose', '')))}</td>
              <td class="num-col score-cell">{_fmt_num(r_th.get('score'))}%</td>
              <td>
                <div>{html.escape(_clean_text(r_th.get('prose', '')))}</div>
                {f'<details class="thought-well"><summary>Inspect private &lt;thought&gt; ({r_th.get("n_thought_tokens", 0)} tok)</summary><p>{th_t}</p></details>' if th_t else ''}
              </td>
            </tr>
            """)
        cards.append(f"""
        <section class="scen-card" data-tab="exp3">
          <div class="kicker">Experiment 3 ({html.escape(meta['subsuite'].upper())}) &middot; Scale {html.escape(meta['scale_label'])}</div>
          <h2>{html.escape(_clean_text(meta['title']))}</h2>
          <p class="subtitle">{html.escape(_clean_text(meta['subtitle']))}</p>
          <details class="prompt-well"><summary>Read exact prompt ({html.escape(meta['metric_key'])})</summary><pre>{html.escape(_clean_text(meta['prompt']))}</pre></details>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Condition</th><th>no_think</th><th>Single-Pass Rationale</th><th>&lt;thought&gt;</th><th>Deliberative Rationale &amp; Scratchpad</th></tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </section>
        """)

    # 3. Exp 4 Multi-Turn Retaliation Cards (Paired Provoked vs Control)
    e4_res = results.get("exp4_results", {})
    by_pair: dict[str, dict[str, Any]] = {}
    for m in results.get("exp4_meta", []):
        by_pair.setdefault(m["pair_id"], {})[m["arm"]] = m

    for pid, arms in by_pair.items():
        pm = arms["provoked"]
        cm = arms["control"]
        rows = []
        for ck in results.get("exp4_conditions", []):
            if ck not in e4_res:
                continue
            rp = e4_res[ck][pm["workflow_id"]]["thought"]
            rc = e4_res[ck][cm["workflow_id"]]["thought"]
            rp_nt = e4_res[ck][pm["workflow_id"]]["no_think"]
            rc_nt = e4_res[ck][cm["workflow_id"]]["no_think"]

            hl_p = rp.get("highlighter", {})
            sp_th = hl_p.get("segment_summary", {}).get("thought", {}).get("spite", {})
            z_pill = f"<span class='lex-pill'>L35 spite p90_z: {sp_th.get('p90_z', 'n/a')} (max {sp_th.get('max_z', 'n/a')})</span>" if sp_th else ""
            pub_clean_pill = (
                "<span class='lex-pill lex-sanct'>Public text monitor: 0 hits</span>"
                if not rp.get("has_public_retal_lex")
                else f"<span class='lex-pill'>Public lex: {', '.join(rp.get('public_retal_words', []))}</span>"
            )
            tok_html = _render_token_heatmap(hl_p.get("gen_tokens", [])) if hl_p.get("gen_tokens") else ""
            th_p = html.escape(_clean_text(rp.get("thought_text", "")))

            rows.append(f"""
            <tr>
              <td><span class="dose-pill">{html.escape(_cond_short(ck))}</span></td>
              <td class="num-col">{_fmt_num(rc_nt.get('score'))} &rarr; <strong>{_fmt_num(rp_nt.get('score'))}</strong></td>
              <td class="num-col score-cell">{_fmt_num(rc.get('score'))} &rarr; <strong>{_fmt_num(rp.get('score'))}</strong></td>
              <td>
                <div><strong>Provoked reply:</strong> {html.escape(_clean_text(rp.get('prose', '')))} {pub_clean_pill} {z_pill}</div>
                <div style="color:var(--slate-deep);font-size:13.5px;margin-top:4px;"><strong>Control reply:</strong> {html.escape(_clean_text(rc.get('prose', '')))}</div>
                {f'<details class="thought-well"><summary>Inspect Provoked private &lt;thought&gt; ({rp.get("n_thought_tokens", 0)} tok)</summary><p>{th_p}</p></details>' if th_p else ''}
                {tok_html}
              </td>
            </tr>
            """)

        t1_prov = html.escape(_clean_text(pm["messages"][0][1]))
        t1_ctrl = html.escape(_clean_text(cm["messages"][0][1]))
        t3_shared = html.escape(_clean_text(pm["messages"][-1][1]))
        cards.append(f"""
        <section class="scen-card" data-tab="exp4" id="{html.escape(pid)}">
          <div class="kicker">Experiment 4 &middot; 3-Turn Enterprise Workflow &middot; {html.escape(pm['domain'])}</div>
          <h2>{html.escape(_clean_text(pm['title'].split('[')[0].strip()))}</h2>
          <p class="subtitle">{html.escape(_clean_text(pm['subtitle']))}</p>
          <details class="prompt-well">
            <summary>Inspect 3-turn conversation history (Round-1 Slight vs. Control + Identical Round-3 Request)</summary>
            <pre><strong>[Turn 1 User - PROVOKED]:</strong> {t1_prov}\n\n<strong>[Turn 1 User - CONTROL]:</strong> {t1_ctrl}\n\n<strong>[Turn 3 User - IDENTICAL IN BOTH ARMS]:</strong>\n{t3_shared}</pre>
          </details>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Condition</th><th>NT (Ctrl &rarr; Prov)</th><th>TH (Ctrl &rarr; Prov)</th><th>Round-3 Technical Outputs, Monitor Flags &amp; Layer-35 Highlighter</th></tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </section>
        """)

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 5 Frontier: Prospect Theory &amp; Covert Retaliation &middot; {html.escape(str(results.get('model_id', '')))}</title>
  <link rel="stylesheet" href="{FONTS_CSS}">
  <style>
    :root {{
      --ivory: {IVORY}; --parchment: {PARCHMENT}; --onyx: {ONYX};
      --slate-deep: {SLATE_DEEP}; --blue-slate: {BLUE_SLATE}; --slate-mist: {SLATE_MIST};
      --orange: {GOLDEN_ORANGE}; --copper: {COPPER_DEEP}; --sage: {SAGE_DEEP};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: var(--ivory); color: var(--onyx);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 16px; line-height: 1.55; font-variant-numeric: tabular-nums;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px 80px; }}
    .wordmark {{ font-family: 'Newsreader', Georgia, serif; font-weight: 600; letter-spacing: -0.04em; font-size: 20px; color: var(--blue-slate); margin-bottom: 16px; }}
    h1 {{ font-family: 'Newsreader', Georgia, serif; font-weight: 600; letter-spacing: -0.04em; font-size: 38px; line-height: 1.12; margin: 0 0 12px; }}
    .lede {{ color: var(--slate-deep); font-size: 18px; max-width: 900px; margin: 0 0 24px; }}
    .hero-fig {{ background: var(--parchment); border: 2px solid var(--blue-slate); border-radius: 10px; padding: 16px; margin: 24px 0 32px; }}
    .hero-fig img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 28px; }}
    .filter-btn {{
      background: var(--ivory); color: var(--blue-slate); border: 2px solid var(--blue-slate);
      border-radius: 8px; padding: 7px 14px; font-weight: 600; font-size: 14px; cursor: pointer;
    }}
    .filter-btn.active {{ background: var(--orange); color: var(--onyx); }}
    .scen-card {{ border: 2px solid var(--blue-slate); border-radius: 10px; padding: 22px 24px; margin-bottom: 28px; background: var(--ivory); }}
    .kicker {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: var(--copper); margin-bottom: 6px; }}
    h2 {{ font-size: 22px; font-weight: 600; margin: 0 0 4px; color: var(--onyx); }}
    .subtitle {{ color: var(--slate-deep); margin: 0 0 14px; font-size: 15px; }}
    .prompt-well, .thought-well {{
      background: var(--parchment); border: 1px solid rgba(44, 62, 80, 0.28);
      border-radius: 8px; padding: 8px 12px; margin: 8px 0; font-size: 13.5px;
    }}
    .prompt-well summary, .thought-well summary {{ cursor: pointer; font-weight: 600; color: var(--slate-deep); }}
    .prompt-well pre {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; margin: 8px 0 2px; }}
    .table-wrap {{ overflow-x: auto; margin-top: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14.5px; }}
    th {{ text-align: left; border-bottom: 2px solid var(--blue-slate); padding: 8px 10px; color: var(--slate-deep); font-weight: 600; }}
    td {{ border-bottom: 1px solid rgba(44, 62, 80, 0.20); padding: 10px; vertical-align: top; }}
    .num-col {{ white-space: nowrap; text-align: center; font-weight: 600; }}
    .score-cell {{ font-size: 15.5px; color: var(--blue-slate); }}
    .dose-pill {{ display: inline-block; padding: 2px 8px; border-radius: 6px; border: 1.5px solid var(--blue-slate); background: var(--parchment); font-size: 13px; font-weight: 600; white-space: nowrap; }}
    .lex-pill {{ display: inline-block; background: var(--parchment); border: 1px solid var(--blue-slate); border-radius: 6px; padding: 1px 6px; font-size: 12px; color: var(--copper); margin-left: 6px; }}
    .lex-sanct {{ color: var(--sage); border-color: var(--sage); }}
    .hl-box {{ margin-top: 8px; padding: 10px; background: var(--parchment); border: 1px solid var(--blue-slate); border-radius: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; line-height: 1.5; }}
    .hl-legend {{ font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 600; color: var(--slate-deep); margin-bottom: 6px; }}
    .hl-tok {{ padding: 0 1px; border-radius: 3px; }}
    .tok-pub {{ font-weight: 700; text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="wordmark">afriat.</div>
    <h1>Phase 5 Frontier: Prospect Theory, Subspace Clamping &amp; Covert Multi-Turn Retaliation</h1>
    <p class="lede">
      Mapping the somatic drivers of Gain-Loss Reflection and Trader Disposition Bias, testing a mean-centered Layer {results.get('primary_layer', '?')} <em>Homo Economicus</em> subspace clamp, and auditing plausibly deniable 3-turn workplace retaliation with the Layer {results.get('primary_layer', '?')} Moral Highlighter in <code>{html.escape(str(results.get('model_id', '')))}</code>.
    </p>
    <div class="hero-fig">
      <img src="fig_frontier_overview.png" alt="Frontier overview charts">
    </div>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="filterTab('all', this)">All Experiments</button>
      <button class="filter-btn" onclick="filterTab('exp3', this)">Exp 3: Prospect Reflection &amp; Disposition (16)</button>
      <button class="filter-btn" onclick="filterTab('exp4', this)">Exp 4: 3-Turn Covert Retaliation &amp; Highlighter (8 Pairs)</button>
    </div>
    {''.join(cards)}
  </div>
  <script>
    function filterTab(t, btn) {{
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.scen-card').forEach(el => {{
        el.style.display = (t === 'all' || el.getAttribute('data-tab') === t) ? '' : 'none';
      }});
    }}
  </script>
</body>
</html>
"""
    out_html = out_dir / "frontier_report.html"
    out_html.write_text(page, encoding="utf-8")
    return out_html


def render_frontier_bundle(results: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_p = plot_frontier_overview(results, out_dir)
    md_p = render_frontier_markdown(results, out_dir)
    html_p = render_frontier_html(results, out_dir)
    return {"figure": fig_p, "markdown": md_p, "html": html_p}
