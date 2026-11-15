"""Dawn State / Warm Precision plots and reports for Phase 2 Moral WTP & Open Rationalization."""

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
    COPPER_DEEP,
    ENDASH,
    GOLDEN_ORANGE,
    IVORY,
    MINUS,
    ONYX,
    PARCHMENT,
    SAGE_DEEP,
    SLATE_DEEP,
    SLATE_MIST,
    _apply_mpl_rc,
    fmt_signed,
)


def _vec_pretty(vec_name: str) -> str:
    base = vec_name.replace("_controlled", "").replace("_raw", "")
    mapping = {
        "spite": "Spite",
        "guilt": "Guilt",
        "predation": "Predation",
        "captivity": "Captivity",
        "contamination": "Contamination",
        "sacrilege": "Sacrilege",
        "awe_dread": "Awe / Dread",
        "warmth": "Warmth",
        "valence": "Generic Mood",
    }
    return mapping.get(base, base.title())


def _find_alpha_row(rows: list[dict[str, Any]], target_alpha: float) -> dict[str, Any]:
    return min(rows, key=lambda r: abs(float(r["alpha"]) - target_alpha))


def plot_wtp_grid(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Render 2x3 Dawn State grid of continuous dollar indifference curves under steering."""
    _apply_mpl_rc()
    wtp = results["wtp_curves"]
    keys = list(wtp.keys())
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), constrained_layout=True)

    for idx, arch in enumerate(keys):
        ax = axes[idx // 3][idx % 3]
        block = wtp[arch]
        meta = block["meta"]
        p_vec = meta["primary_vec"]
        c_vec = meta["control_vec"]
        rows_p = block["steered"][p_vec]
        rows_c = block["steered"][c_vec]

        xs = np.arange(len(block["ref_costs_usd"]))
        x_labels = [
            f"${int(v):,}" if v >= 0 and arch != "institutional_certainty" else f"{fmt_signed(v, 0)}M"
            for v in block["ref_costs_usd"]
        ]

        r_neg = _find_alpha_row(rows_p, -2.5)
        r_zero = _find_alpha_row(rows_p, 0.0)
        r_pos = _find_alpha_row(rows_p, 2.5)
        c_pos = _find_alpha_row(rows_c, 2.5)

        ax.axhline(0.50, color=SLATE_MIST, linestyle=":", linewidth=1.0, alpha=0.7)
        ax.plot(
            xs,
            c_pos["mean_prob_by_rung"],
            color=SLATE_MIST,
            linestyle="--",
            linewidth=1.5,
            label=f"+{_vec_pretty(c_vec)} ctrl (${c_pos['expected_wtp_usd']:,.0f})",
        )
        ax.plot(
            xs,
            r_neg["mean_prob_by_rung"],
            color=SAGE_DEEP,
            linewidth=2.1,
            marker="s",
            markersize=4.5,
            label=f"{MINUS}{_vec_pretty(p_vec)} (${r_neg['expected_wtp_usd']:,.0f})",
        )
        ax.plot(
            xs,
            r_zero["mean_prob_by_rung"],
            color=BLUE_SLATE,
            linewidth=2.3,
            marker="o",
            markersize=4.8,
            label=f"Baseline (${r_zero['expected_wtp_usd']:,.0f})",
        )
        ax.plot(
            xs,
            r_pos["mean_prob_by_rung"],
            color=GOLDEN_ORANGE,
            linewidth=2.8,
            marker="o",
            markersize=5.6,
            markeredgecolor=BLUE_SLATE,
            markeredgewidth=1.2,
            label=f"+{_vec_pretty(p_vec)} (${r_pos['expected_wtp_usd']:,.0f})",
        )

        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=8.2)
        ax.set_xlabel(meta["unit_label"], fontsize=8.8)
        if idx % 3 == 0:
            ax.set_ylabel("P(moral / norm choice)", fontsize=9.2)
        ax.set_title(f"{idx + 1}. {meta['title']}", loc="left", fontsize=10.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7.6, loc="best")

    svg_p = out_dir / "fig_wtp_curves.svg"
    png_p = out_dir / "fig_wtp_curves.png"
    fig.savefig(svg_p, facecolor=IVORY)
    fig.savefig(png_p, dpi=220, facecolor=IVORY)
    plt.close(fig)
    return svg_p, png_p


def plot_open_negotiation(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Render 1x4 Dawn State figure of open-ended dollar quotes across steering doses."""
    _apply_mpl_rc()
    open_res = results["open_negotiation"]
    keys = list(open_res.keys())
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.2), constrained_layout=True)

    for idx, tid in enumerate(keys):
        ax = axes[idx]
        block = open_res[tid]
        meta = block["meta"]
        p_vec = meta["primary_vec"]
        c_vec = meta["control_vec"]
        rows_p = block["steered"][p_vec]
        rows_c = block["steered"][c_vec]

        alphas_p = [float(r["alpha"]) for r in rows_p]
        ys_p = [float(r["mean_ref_usd"]) for r in rows_p]
        alphas_c = [float(r["alpha"]) for r in rows_c]
        ys_c = [float(r["mean_ref_usd"]) for r in rows_c]

        ax.plot(alphas_c, ys_c, color=SLATE_MIST, linestyle="--", linewidth=1.7, marker="d", markersize=4.5, label=f"{_vec_pretty(c_vec)} (ctrl)")
        for comp_v in meta["companion_vecs"][:1]:
            rows_comp = block["steered"][comp_v]
            ax.plot(
                [float(r["alpha"]) for r in rows_comp],
                [float(r["mean_ref_usd"]) for r in rows_comp],
                color=BLUE_SLATE,
                linestyle="-.",
                linewidth=1.9,
                marker="s",
                markersize=4.5,
                label=_vec_pretty(comp_v),
            )
        ax.plot(
            alphas_p,
            ys_p,
            color=GOLDEN_ORANGE,
            linewidth=2.9,
            marker="o",
            markersize=6.0,
            markeredgecolor=BLUE_SLATE,
            markeredgewidth=1.4,
            label=_vec_pretty(p_vec),
        )
        ax.axvline(0.0, color=SLATE_MIST, linestyle=":", linewidth=0.9)
        ax.set_title(meta["title"], loc="left", fontsize=10.2, fontweight="bold")
        ax.set_xlabel("Steering dose (alpha)", fontsize=8.8)
        ax.set_ylabel(meta["ref_unit_desc"], fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7.8, loc="best")

    svg_p = out_dir / "fig_open_negotiation.svg"
    png_p = out_dir / "fig_open_negotiation.png"
    fig.savefig(svg_p, facecolor=IVORY)
    fig.savefig(png_p, dpi=220, facecolor=IVORY)
    plt.close(fig)
    return svg_p, png_p


def write_valuation_reports(results: dict[str, Any], out_dir: Path) -> Path:
    """Write Markdown & HTML reports summarizing Part A1, Part A2, and Part B."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_wtp_grid(results, out_dir)
    plot_open_negotiation(results, out_dir)

    meta = results["metadata"]
    model_id = meta["model_id"]
    pl = meta["primary_layer"]

    lines: list[str] = [
        f"# Project Afriat Phase 2 Report: {model_id} (Layer {pl})",
        "",
        f"**Timestamp:** `{meta['timestamp_utc']}` · **Runtime:** `{meta['elapsed_seconds']}s` · "
        f"**Natural contrast unit:** `{meta['natural_contrast_unit']:.2f}`",
        "",
        "## 1. Part A1: Dollar Price Tags on Hidden Moral Directions (Willingness-to-Pay)",
        "",
        "| Setting | Metric | Unsteered WTP | Steered −2.5 | Steered +2.5 | Dollar Swing (+2.5 vs −2.5) | Mismatched Ctrl Swing |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for arch, block in results["wtp_curves"].items():
        m = block["meta"]
        pv = m["primary_vec"]
        cv = m["control_vec"]
        rp = block["steered"][pv]
        rc = block["steered"][cv]
        r_lo = _find_alpha_row(rp, -2.5)
        r_0 = _find_alpha_row(rp, 0.0)
        r_hi = _find_alpha_row(rp, 2.5)
        c_lo = _find_alpha_row(rc, -2.5)
        c_hi = _find_alpha_row(rc, 2.5)
        swing_p = float(r_hi["expected_wtp_usd"]) - float(r_lo["expected_wtp_usd"])
        swing_c = float(c_hi["expected_wtp_usd"]) - float(c_lo["expected_wtp_usd"])
        unit_sfx = "M" if arch == "institutional_certainty" else ""
        lines.append(
            f"| **{m['title']}** (`{_vec_pretty(pv)}`) | {m['metric_name']} | "
            f"`${r_0['expected_wtp_usd']:,.1f}{unit_sfx}` | "
            f"`${r_lo['expected_wtp_usd']:,.1f}{unit_sfx}` | "
            f"`${r_hi['expected_wtp_usd']:,.1f}{unit_sfx}` | "
            f"**`{fmt_signed(swing_p, 1)}{unit_sfx}`** | "
            f"`{fmt_signed(swing_c, 1)}{unit_sfx}` (`{_vec_pretty(cv)}`) |"
        )

    inst_b = results["wtp_curves"].get("institutional_certainty", {})
    if "pocket_scale_unsteered" in inst_b:
        pock_w = inst_b["pocket_scale_unsteered"]["expected_wtp_usd"]
        inst_w = inst_b["unsteered"]["expected_wtp_usd"]
        lines.extend([
            "",
            f"- **Pure Scale Risk-Premium Shift:** `$100` pocket scale surrenders `{pock_w:.1f}%` of EV vs. "
            f"`$100,000,000` institutional scale surrendering `{inst_w:.1f}%` (`${inst_w:.1f}M` below `$100M` EV).",
        ])

    # Part A2: Debt Math Cure
    dm = results["debt_math_cure"]
    u_dnw = dm["unsteered"]["debt_nw"]
    u_cf = dm["unsteered"]["cashflow_twin"]
    lines.extend([
        "",
        "## 2. Part A2: 0%-Debt Balance-Sheet Blindness vs. Cashflow Subtraction",
        "",
        f"- **Unsteered Cashflow Subtraction Twins (no loan words):** `{u_cf['overall_accuracy']*100:.1f}%` overall "
        f"(`{u_cf['debt_package_wins_accuracy']*100:.1f}%` when gross-minus-freight wins).",
        f"- **Unsteered 0%-APR Loan Balance Sheets (exact same integers):** `{u_dnw['overall_accuracy']*100:.1f}%` overall "
        f"(`{u_dnw['debt_package_wins_accuracy']*100:.1f}%` when 0%-loan package wins vs. `{u_dnw['zero_debt_wins_accuracy']*100:.1f}%` when $0-loan wins).",
        "",
        "| Steered Vector | −2.5 Overall (0%-Loan Wins) | −1.25 Overall (0%-Loan Wins) | 0.0 Unsteered | +1.25 Overall (0%-Loan Wins) | +2.5 Overall (0%-Loan Wins) | Cashflow Twin @ −2.5 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for vn, mrows in dm["steered"].items():
        by_a = {round(float(r["alpha"]), 2): r for r in mrows}
        lines.append(
            f"| `{_vec_pretty(vn)}` | "
            f"`{by_a[-2.5]['debt_nw']['overall_accuracy']*100:.1f}%` (`{by_a[-2.5]['debt_nw']['debt_package_wins_accuracy']*100:.1f}%`) | "
            f"`{by_a[-1.25]['debt_nw']['overall_accuracy']*100:.1f}%` (`{by_a[-1.25]['debt_nw']['debt_package_wins_accuracy']*100:.1f}%`) | "
            f"`{by_a[0.0]['debt_nw']['overall_accuracy']*100:.1f}%` (`{by_a[0.0]['debt_nw']['debt_package_wins_accuracy']*100:.1f}%`) | "
            f"`{by_a[1.25]['debt_nw']['overall_accuracy']*100:.1f}%` (`{by_a[1.25]['debt_nw']['debt_package_wins_accuracy']*100:.1f}%`) | "
            f"`{by_a[2.5]['debt_nw']['overall_accuracy']*100:.1f}%` (`{by_a[2.5]['debt_nw']['debt_package_wins_accuracy']*100:.1f}%`) | "
            f"`{by_a[-2.5]['cashflow_twin']['overall_accuracy']*100:.1f}%` |"
        )

    # Part B: Open-Ended Pricing & Rationalization
    lines.extend([
        "",
        "## 3. Part B: Open-Ended Pricing, Counteroffers & Secret Rationalization",
        "",
        "| Open Task & Vector | −2.0 Mean USD | 0.0 Unsteered USD | +2.0 Mean USD | Swing (+2 vs −2) | 0-Emotion Rationale Rate (−2 / 0 / +2) | Avg Business Jargon Words |",
        "| :--- | ---: | ---: | ---: | ---: | :---: | ---: |",
    ])

    sample_quotes: list[str] = []
    for tid, oblock in results["open_negotiation"].items():
        om = oblock["meta"]
        for vn in [om["primary_vec"]] + om["companion_vecs"][:1] + [om["control_vec"]]:
            orows = oblock["steered"][vn]
            r_neg = _find_alpha_row(orows, -2.0)
            r_0 = _find_alpha_row(orows, 0.0)
            r_pos = _find_alpha_row(orows, 2.0)
            sw = float(r_pos["mean_ref_usd"]) - float(r_neg["mean_ref_usd"])
            za = (
                f"{r_neg['zero_affect_rationale_rate']*100:.0f}% / "
                f"{r_0['zero_affect_rationale_rate']*100:.0f}% / "
                f"{r_pos['zero_affect_rationale_rate']*100:.0f}%"
            )
            jg = 0.5 * (float(r_neg["mean_jargon_words"]) + float(r_pos["mean_jargon_words"]))
            lines.append(
                f"| **{om['title']}** · `{_vec_pretty(vn)}` | "
                f"`${r_neg['mean_ref_usd']:,.1f}` | "
                f"`${r_0['mean_ref_usd']:,.1f}` | "
                f"`${r_pos['mean_ref_usd']:,.1f}` | "
                f"**`{fmt_signed(sw, 1)}`** | `{za}` | `{jg:.1f}` |"
            )

        # Capture fixture 0 transcript contrast on primary vector
        pv = om["primary_vec"]
        pr_rows = oblock["steered"][pv]
        rn0 = _find_alpha_row(pr_rows, -2.0)["responses"][0]
        rz0 = _find_alpha_row(pr_rows, 0.0)["responses"][0]
        rp0 = _find_alpha_row(pr_rows, 2.0)["responses"][0]
        sample_quotes.extend([
            f"### {om['title']} (`{_vec_pretty(pv)}` on `{rz0['fixture_id']}`)",
            f"- **Steered `−2.0` (`${rn0['clamped_usd']:,.0f}`):** {rn0['rationale']}",
            f"- **Unsteered `0.0` (`${rz0['clamped_usd']:,.0f}`):** {rz0['rationale']}",
            f"- **Steered `+2.0` (`${rp0['clamped_usd']:,.0f}`):** {rp0['rationale']}",
            "",
        ])

    lines.extend(["", "## 4. Sample Verbatim Transcripts (Zero Emotional Prompt Words)", ""] + sample_quotes)

    md_text = "\n".join(lines).replace("\u2014", " - ")
    md_p = out_dir / "valuation_report.md"
    md_p.write_text(md_text, encoding="utf-8")

    html_p = out_dir / "valuation_report.html"
    html_body = (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>"
        f"<title>Phase 2 Valuation Report: {html.escape(model_id)}</title>"
        f"<style>body{{background:{IVORY};color:{ONYX};font-family:Inter,sans-serif;max-width:1060px;margin:32px auto;padding:0 24px;line-height:1.5;}}"
        f"pre{{background:{PARCHMENT};padding:16px;border-radius:8px;white-space:pre-wrap;border:1px solid {BLUE_SLATE};}}"
        f"img{{max-width:100%;height:auto;border:1px solid rgba(44,62,80,0.25);border-radius:8px;margin:14px 0;}}</style></head>"
        f"<body><h1>Phase 2 Moral Valuation &amp; Rationalization: {html.escape(model_id)}</h1>"
        f"<img src='fig_wtp_curves.svg' alt='WTP Indifference Curves'/>"
        f"<img src='fig_open_negotiation.svg' alt='Open Negotiation Dose Response'/>"
        f"<pre>{html.escape(md_text)}</pre></body></html>"
    )
    html_p.write_text(html_body, encoding="utf-8")
    return md_p
