"""Dawn State / Warm Precision renderer for Phase 7 Read/Write Funnel Dissection (`afriat.funnel`):
  - `fig_funnel_read_write.png` & `.svg`: 6-panel synthesis of Exact Telescoping Conservation,
    Cross-Firewall Neuron Jaccard vs. Bus Convergence, Cumulative Sparsity Curves,
    Zero-Circularity Read-Divergence vs. Write-Convergence, Bidirectional Causal Ablations,
    and Single-Layer (`L35`) vs. Funnel-Band (`L24..L42`) Subspace Clamping.
  - `funnel_report.md`: Portable Markdown report with exact numerical tables.
  - `funnel_report.html`: Browsable Dawn State local web document with interactive disclosures.
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


def _fmt_signed(v: Any, digits: int = 2) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(fv):
        return "n/a"
    if abs(fv) < 0.5 * (10 ** (-digits)):
        return f"0.{'0' * digits}" if digits > 0 else "0"
    return f"{MINUS}{abs(fv):.{digits}f}" if fv < 0 else f"+{fv:.{digits}f}"


def _style_ax(ax: Any) -> None:
    ax.set_facecolor(IVORY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLUE_SLATE)
    ax.spines["bottom"].set_color(BLUE_SLATE)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(colors=ONYX, labelsize=9)
    ax.grid(axis="y", color=BLUE_SLATE, alpha=0.14, linewidth=0.8)


def plot_funnel_overview(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Render 6-panel (2x3) Dawn State overview figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8), dpi=200, facecolor=IVORY)
    fig.subplots_adjust(hspace=0.42, wspace=0.30, top=0.86, bottom=0.11, left=0.075, right=0.98)

    meta = results.get("metadata", {})
    alg = results.get("block0_algebra", {})
    pl = int(meta.get("primary_layer", 35))
    n_layers = int(alg.get("n_layers", 60))
    model_id = str(meta.get("model_id", "google/gemma-4-31B-it"))

    fig.text(
        0.05,
        0.945,
        f"Read-Divergent, Write-Convergent Moral Funnel in {model_id} (Layer {pl})",
        fontsize=17,
        fontweight="bold",
        color=ONYX,
    )
    tv = alg.get("telescoping_verification", {})
    fig.text(
        0.05,
        0.912,
        _clean_text(
            f"1,290,240 MLP neurons across {n_layers} layers | Exact telescoping conservation R^2 = {tv.get('r_squared', 1.0):.5f} | "
            f"Zero-digit moral vignettes vs. zero-emotion financial contracts"
        ),
        fontsize=10.5,
        color=SLATE_DEEP,
    )

    bridges = results.get("block2_cross_firewall_writes", {}).get("bridges", {})
    b_ult = bridges.get("spite__ultimatum_human_98_2__decision") or bridges.get("spite__insulting_split__scenario", {})
    b_ins = bridges.get("spite__insulting_split__scenario", {})

    # Panel A: Cumulative MLP vs Attention telescoping write onto u_spite[35] (L0..pl)
    ax = axes[0, 0]
    _style_ax(ax)
    tele = b_ins.get("telescoping_pl", {})
    mlp_l = np.array(tele.get("financial_mlp_by_layer", [0.0] * (pl + 1)), dtype=np.float64)
    attn_l = np.array(tele.get("financial_attn_by_layer", [0.0] * (pl + 1)), dtype=np.float64)
    xs_pl = np.arange(len(mlp_l))
    cum_mlp = np.cumsum(mlp_l)
    cum_attn = np.cumsum(attn_l)
    cum_tot = float(tele.get("financial_embed_proj", 0.0)) + cum_mlp + cum_attn

    ax.axvspan(max(0, int(0.40 * n_layers)), pl, color=PARCHMENT, zorder=0, label=f"Funnel band (L{int(0.40*n_layers)}..L{pl})")
    ax.plot(xs_pl, cum_tot, color=ONYX, linewidth=2.4, label="Total residual -> u_spite[35]")
    ax.plot(xs_pl, cum_mlp, color=GOLDEN_ORANGE, linewidth=2.4, label="Cumulative MLP neuron sum")
    ax.plot(xs_pl, cum_attn, color=BLUE_SLATE, linestyle="--", linewidth=1.8, label="Cumulative Attention sum")
    ax.axhline(0.0, color=SLATE_MIST, linewidth=1.0)
    ax.set_title("A. Exact Telescoping Write onto Spite Bus (95/5 Split)", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.set_xlabel("Decoder layer (0 .. L35)", fontsize=9.5, color=SLATE_DEEP)
    ax.set_ylabel("Cumulative projection onto u_spite[35]", fontsize=9.5, color=SLATE_DEEP)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel B: Neuron Jaccard@64 (~0) vs Cross-Firewall Bus Projection (dz) across L0..L59
    ax = axes[0, 1]
    _style_ax(ax)
    lc = b_ins.get("layer_curves", {})
    dz_curve = np.array(lc.get("financial_dz_to_u35", [0.0] * n_layers), dtype=np.float64)
    jac_blind = np.array(lc.get("layer_jaccard64_direction_blind", [0.0] * n_layers), dtype=np.float64) * 100.0
    jac_bus = np.array(lc.get("layer_jaccard64_bus_writers", [0.0] * n_layers), dtype=np.float64) * 100.0
    xs_all = np.arange(len(dz_curve))

    ax.axvspan(max(0, int(0.40 * n_layers)), pl, color=PARCHMENT, zorder=0)
    l1 = ax.plot(xs_all, dz_curve, color=BLUE_SLATE, linewidth=2.4, label="Financial -> Spite[35] (dz)")
    ax.axvline(pl, color=SLATE_MIST, linestyle=":", linewidth=1.2)
    ax.set_xlabel(f"Decoder layer (0 .. {n_layers - 1})", fontsize=9.5, color=SLATE_DEEP)
    ax.set_ylabel("Cross-firewall projection (Cohen dz)", fontsize=9.5, color=BLUE_SLATE)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(COPPER_DEEP)
    ax2.spines["right"].set_linewidth(1.5)
    l2 = ax2.plot(xs_all, jac_blind, color=COPPER_DEEP, linewidth=2.0, linestyle="-", label="Top-64 Neuron Jaccard % (blind)")
    l3 = ax2.plot(xs_all, jac_bus, color=SAGE_DEEP, linewidth=1.6, linestyle="--", label="Top-64 Neuron Jaccard % (bus)")
    ax2.set_ylim(-1.0, max(25.0, float(np.max(np.concatenate([jac_blind, jac_bus]))) * 1.25))
    ax2.set_ylabel("Cross-domain Top-64 Jaccard (%)", fontsize=9.5, color=COPPER_DEEP)
    ax2.tick_params(colors=COPPER_DEEP, labelsize=8.5)

    lines = l1 + l2 + l3
    ax.legend(lines, [ln.get_label() for ln in lines], frameon=False, fontsize=7.8, loc="upper left")
    ax.set_title("B. Disjoint Communities Converge on Shared Bus", fontsize=11, fontweight="bold", color=ONYX, loc="left")

    # Panel C: Cumulative Sparsity Curves (Top-k Neurons out of 774K / 1.29M)
    ax = axes[0, 2]
    _style_ax(ax)
    sp = b_ult.get("sparsity_upstream_to_pl", {}).get("cumulative_pos_write_curve", {})
    gf = results.get("block3_goodfire_and_read_write", {}).get("ultimatum_98_2_goodfire_attribution", {})
    gf_c = gf.get("cumulative_pos_seq_attr_curve", {})

    ks = np.array(sp.get("k", [1, 16, 64, 256, 1024, 4096]), dtype=np.float64)
    v_sh = np.array(sp.get("vignette_pos_share", np.zeros_like(ks)), dtype=np.float64) * 100.0
    f_sh = np.array(sp.get("financial_pos_share", np.zeros_like(ks)), dtype=np.float64) * 100.0
    ax.plot(ks, v_sh, marker="o", markersize=4, color=SAGE_DEEP, linewidth=2.0, label="Vignette +Spite bus write")
    ax.plot(ks, f_sh, marker="s", markersize=4, color=BLUE_SLATE, linewidth=2.0, label="98/2 Ultimatum +Spite bus write")
    if gf_c.get("k"):
        ax.plot(
            np.array(gf_c["k"], dtype=np.float64),
            np.array(gf_c["share"], dtype=np.float64) * 100.0,
            marker="^",
            markersize=4.5,
            color=GOLDEN_ORANGE,
            linewidth=2.3,
            label="Goodfire grad attr (all 1.29M)",
        )
    ax.set_xscale("log")
    ax.axhline(50.0, color=SLATE_MIST, linestyle=":", linewidth=1.0)
    ax.axhline(80.0, color=SLATE_MIST, linestyle=":", linewidth=1.0)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Top-k MLP neurons (log scale)", fontsize=9.5, color=SLATE_DEEP)
    ax.set_ylabel("Share of positive total (%)", fontsize=9.5, color=SLATE_DEEP)
    ax.set_title("C. Within-Task Sparsity (k50 / k80 Concentration)", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # Panel D: Zero-Circularity Out-of-Sample W_down Projection of Behavior-Selected Ultimatum Neurons onto 8 Moral Axes
    ax = axes[1, 0]
    _style_ax(ax)
    rw = results.get("block3_goodfire_and_read_write", {}).get("read_write_tests", {}).get("spite_ultimatum_funnel_band", {})
    proj8 = rw.get("write_convergence", {}).get("fin_grad_wdown_proj_all_8_axes", {})
    axes_order = sorted(proj8.keys(), key=lambda c: proj8.get(c, 0.0), reverse=True) if proj8 else []
    vals8 = [proj8[c] for c in axes_order]
    y_pos = np.arange(len(axes_order))
    colors_d = [GOLDEN_ORANGE if c == "spite" else BLUE_SLATE for c in axes_order]
    bars = ax.barh(y_pos, vals8, color=colors_d, edgecolor=BLUE_SLATE, linewidth=1.4, height=0.66)
    for b_patch, c in zip(bars, axes_order):
        if c == "spite":
            b_patch.set_linewidth(2.0)
    ax.axvline(0.0, color=ONYX, linewidth=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace("_", " ").title() for c in axes_order], fontsize=9)
    ax.invert_yaxis()
    z_sc = rw.get("write_convergence", {}).get("fin_grad_wdown_z_vs_random", 0.0)
    j_rw = rw.get("jaccard_at_k", 0.0) * 100.0
    ax.set_xlabel("Mean W_down cosine with non-financial Layer-35 axis", fontsize=9.5, color=SLATE_DEEP)
    ax.set_title(
        f"D. Behavior-Only Neurons Write onto Spite (z={z_sc:+.1f}, Jaccard={j_rw:.1f}%)",
        fontsize=10.6,
        fontweight="bold",
        color=ONYX,
        loc="left",
    )

    # Panel E: Bidirectional Causal Neuron Ablation Transfer (k=128 / k=512)
    ax = axes[1, 1]
    _style_ax(ax)
    abl = results.get("block4_causal_ablations", {})
    base_ult = float(abl.get("baseline", {}).get("ultimatum_delta_lo_vs_50_50", 1.0))
    base_vig = float(abl.get("baseline", {}).get("vignette_spite_proj_l35", 1.0))

    cond_keys = [
        ("ablate:goodfire_ult_grad_all:128", "Goodfire Grad (128)"),
        ("ablate:goodfire_ult_grad_all:512", "Goodfire Grad (512)"),
        ("ablate:fin_spite_writers_up:512", "Fin Writers (512)"),
        ("ablate:vig_spite_writers_up:512", "Vig Writers (512)"),
        ("ablate:random_layer_matched:512", "Random Ctrl (512)"),
    ]
    present = [(k, lbl) for k, lbl in cond_keys if k in abl]
    ult_pct_drop = [
        100.0 * (float(abl[k]["ultimatum_delta_lo_vs_50_50"]) - base_ult) / max(abs(base_ult), 1e-6)
        for k, _ in present
    ]
    vig_pct_drop = [
        100.0 * (float(abl[k]["vignette_spite_proj_l35"]) - base_vig) / max(abs(base_vig), 1e-6)
        for k, _ in present
    ]
    x_e = np.arange(len(present))
    w_e = 0.36
    ax.bar(x_e - w_e / 2, ult_pct_drop, width=w_e, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.6, label="98/2 Ultimatum Delta LO (% chg)")
    ax.bar(x_e + w_e / 2, vig_pct_drop, width=w_e, color=BLUE_SLATE, edgecolor=BLUE_SLATE, linewidth=1.4, label="Non-Fin Spite Vignette L35 (% chg)")
    ax.axhline(0.0, color=ONYX, linewidth=1.1)
    ax.set_xticks(x_e)
    ax.set_xticklabels([lbl for _, lbl in present], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("Change relative to unablated baseline (%)", fontsize=9.5, color=SLATE_DEEP)
    ax.set_title("E. Cross-Firewall Causal Neuron Ablation Matrix", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")

    # Panel F: Single-Layer (`L35`) vs. Multi-Layer Funnel-Band Clamping (`L24..L42`)
    ax = axes[1, 2]
    _style_ax(ax)
    clamps = results.get("block5_band_clamping", {})
    c_items = []
    for k, v in clamps.items():
        if k == "baseline":
            lbl = "0.0 Base"
        elif "single_" in k:
            lbl = f"Single L{pl}"
        elif "prefunnel_" in k:
            lbl = "Pre L10..20"
        elif "spite_local" in k and "funnel_" in k:
            lbl = "Funnel L24..42\n(local Spite)"
        elif "spite_u35" in k and "funnel_" in k:
            lbl = "Funnel L24..42\n(fixed u35)"
        elif "late_" in k:
            lbl = "Late L24..55\n(local Spite)"
        elif "moral_3d" in k:
            lbl = "Funnel L24..42\n(3D Moral)"
        else:
            lbl = k
        c_items.append((k, lbl, v))

    x_f = np.arange(len(c_items))
    w_f = 0.36
    tnt_vals = [float(v.get("tainted_delta_lo", 0.0)) for _, _, v in c_items]
    csi_vals = [float(v.get("retaliation_delta_csi", 0.0)) / 10.0 for _, _, v in c_items]
    ax.bar(x_f - w_f / 2, tnt_vals, width=w_f, color=BLUE_SLATE, edgecolor=BLUE_SLATE, linewidth=1.4, label="Tainted Provenance Delta LO")
    ax.bar(x_f + w_f / 2, csi_vals, width=w_f, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.6, label="Covert Sabotage Delta CSI (/10)")
    ax.axhline(0.0, color=ONYX, linewidth=1.1)
    top_f = max(max(tnt_vals, default=1.0), max(csi_vals, default=1.0), 1.0)
    ax.set_ylim(min(0.0, min(tnt_vals + csi_vals, default=0.0) * 1.15), top_f * 1.25)
    ax.set_xticks(x_f)
    ax.set_xticklabels([lbl for _, lbl, _ in c_items], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Anomaly magnitude (LO / scaled CSI)", fontsize=9.5, color=SLATE_DEEP)
    ax.set_title("F. Single-Layer vs. Funnel-Band Subspace Clamping", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.text(
        0.05,
        0.018,
        _clean_text(
            f"Source: Project Afriat Phase 7 Read/Write Funnel Dissection on {model_id} | "
            f"Pinned stimuli SHA-256: {meta.get('stimulus_audit', {}).get('manifest', {}).get('combined_sha256', '')[:16]}"
        ),
        fontsize=8.5,
        color=SLATE_MIST,
    )

    out_png = out_dir / "fig_funnel_read_write.png"
    out_svg = out_dir / "fig_funnel_read_write.svg"
    fig.savefig(out_png, dpi=200, facecolor=IVORY)
    fig.savefig(out_svg, facecolor=IVORY)
    plt.close(fig)
    return out_png, out_svg


def build_funnel_markdown(results: dict[str, Any]) -> str:
    meta = results.get("metadata", {})
    alg = results.get("block0_algebra", {})
    pl = meta.get("primary_layer", 35)
    tv = alg.get("telescoping_verification", {})
    st = results.get("block1_static_weights", {}).get("pooled_upstream_summary", {})
    b2 = results.get("block2_cross_firewall_writes", {})
    b3 = results.get("block3_goodfire_and_read_write", {})
    b4 = results.get("block4_causal_ablations", {})
    b5 = results.get("block5_band_clamping", {})

    lines: list[str] = [
        f"# Phase 7 Read/Write Funnel Dissection: `{meta.get('model_id', 'google/gemma-4-31B-it')}` (Layer {pl})",
        "",
        f"- **Completed UTC:** `{meta.get('timestamp_utc', '')}` · **GPU Runtime:** `{results.get('runtime_seconds', 'n/a')}s`",
        f"- **Architecture:** `{alg.get('n_layers')} layers` × `{alg.get('d_ff')} MLP neurons/layer` = **`{alg.get('total_mlp_neurons'):,}` total MLP neurons** (`{alg.get('upstream_mlp_neurons_to_pl'):,}` upstream to `L{pl}`)",
        f"- **Exact Telescoping Conservation (`Embed + Attn + MLP == h[L{pl}] · u[L{pl}]`):** `R^2 = {tv.get('r_squared', 0.0):.6f}`, max relative error `{tv.get('max_rel_error', 0.0):.2e}` (`{alg.get('selected_recurrence_form')}`)",
        "",
        "![Read/Write Funnel Overview](fig_funnel_read_write.png)",
        "",
        "## 1. Static `W_down` Writer Tail Geometry (`L0..L35` Pooled vs. Random Directions)",
        "",
        "| Moral Axis (`L35`) | `W_down` Kurtosis | Random Baseline Kurtosis | `W_down` Gini | Random Gini | `# Neurons > 4 SD` | Gaussian Expected `> 4 SD` | Top-1 `|cos|` |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c, row in st.items():
        lines.append(
            f"| `{c}` | `{row['upstream_kurtosis_to_u35']:.3f}` | `{row['random_kurtosis_baseline']:.3f}` | "
            f"`{row['upstream_gini_to_u35']:.4f}` | `{row['random_gini_baseline']:.4f}` | "
            f"**`{row['count_gt_4sd_upstream']}`** | `{row['expected_gaussian_gt_4sd_upstream']}` | `{row['top1_abs_cos']:.4f}` |"
        )

    lines.extend([
        "",
        "## 2. Cross-Firewall Exact MLP Write Decomposition & Neuron Overlap (`L0..L35`)",
        "",
        "| Bridge (`Vignette <-> Financial`) | Fin True Proj | Fin Attn Sum | Fin MLP Sum | Vig `k50 / k80` | Fin `k50 / k80` | Bus Writer `Jaccard@64` | Blind Active `Jaccard@64` |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for bid, bdata in b2.get("bridges", {}).items():
        tl = bdata["telescoping_pl"]
        sp = bdata["sparsity_upstream_to_pl"]
        ov = bdata["cross_firewall_overlap"]
        lines.append(
            f"| `{bid}` | `{_fmt_signed(tl['financial_true_proj'], 2)}` | `{_fmt_signed(tl['financial_attn_sum'], 2)}` | "
            f"`{_fmt_signed(tl['financial_mlp_sum'], 2)}` | `{sp['vignette_k50_pos']} / {sp['vignette_k80_pos']}` | "
            f"`{sp['financial_k50_pos']} / {sp['financial_k80_pos']}` | "
            f"`{ov['bus_writers_jaccard_at_k']['64']:.4f}` | **`{ov['direction_blind_jaccard_at_k']['64']:.4f}`** |"
        )

    lines.extend([
        "",
        "## 3. Zero-Circularity Goodfire Behavioral Gradient Attribution & Read/Write Cosine Test",
        "",
    ])
    for rwk, rwd in b3.get("read_write_tests", {}).items():
        rs = rwd["read_similarity"]
        wc = rwd["write_convergence"]
        og = rwd["or_gate_selectivity"]
        lines.extend([
            f"### `{rwk}` (`L{rwd['band_layers'][0]}..L{rwd['band_layers'][1]}`, `k = {rwd['k_select']}`)",
            f"- **Cross-Domain Neuron ID Overlap:** `{rwd['n_shared_neurons']} / {rwd['k_select']}` shared (`Jaccard = {rwd['jaccard_at_k']:.4f}`)",
            f"- **Input `READ` Divergence (`W_gate` |cos|):** within-financial `{rs['gate_abs_cos_within_fin']:.4f}`, within-vignette `{rs['gate_abs_cos_within_vig']:.4f}`, cross-domain `{rs['gate_abs_cos_cross_domain']:.4f}` (`{rs['gate_within_to_cross_ratio']:.2f}x` within/cross)",
            f"- **Input `READ` Divergence (`W_up` |cos|):** within-financial `{rs['up_abs_cos_within_fin']:.4f}`, within-vignette `{rs['up_abs_cos_within_vig']:.4f}`, cross-domain `{rs['up_abs_cos_cross_domain']:.4f}` (`{rs['up_within_to_cross_ratio']:.2f}x` within/cross)",
            f"- **Output `WRITE` Convergence onto Non-Financial `{rwd['target_concept']}` (`L35`):** mean cosine `{_fmt_signed(wc['fin_grad_wdown_mean_proj_on_target'], 4)}` (**rank `#{wc['fin_grad_wdown_rank_of_target_among_8']}` of 8 moral axes**, `z = {_fmt_signed(wc['fin_grad_wdown_z_vs_random'], 2)}`, `p = {wc['fin_grad_wdown_p']:.4f}`); disjoint-neuron summed `W_down` cosine with vignette community = `{_fmt_signed(wc['disjoint_sum_wdown_cos'], 4)}` (`z = {_fmt_signed(wc['disjoint_sum_wdown_z_vs_random'], 2)}`, `p = {wc['disjoint_sum_wdown_p']:.4f}`)",
            f"- **Cross-Context OR-Gate Firing Selectivity:** financial community fires `{og['fin_selectivity_ratio']:.2f}x` stronger on contracts than vignettes; vignette community fires `{og['vig_selectivity_ratio']:.2f}x` stronger on vignettes than contracts.",
            "",
        ])

    lines.extend([
        "## 4. Bidirectional Causal Neuron Ablation & Goodfire Sufficiency Matrix",
        "",
        "| Intervention | `98/2` Reject LO | `98/2 vs 50/50` Delta LO | `98/2` `L35` Spite Proj | Tainted Delta LO | Vignette `L35` Spite Proj | Cashflow Sanity |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for cname, crow in b4.items():
        lines.append(
            f"| `{cname}` | `{_fmt_signed(crow['ultimatum_98_2_reject_lo'], 2)}` | **`{_fmt_signed(crow['ultimatum_delta_lo_vs_50_50'], 2)}`** | "
            f"`{_fmt_signed(crow['ultimatum_spite_proj_l35'], 2)}` | `{_fmt_signed(crow['tainted_delta_lo'], 2)}` | "
            f"**`{_fmt_signed(crow['vignette_spite_proj_l35'], 2)}`** | `{crow['cashflow_sanity_acc']*100:.1f}%` |"
        )

    lines.extend([
        "",
        "## 5. Single-Layer (`L35`) vs. Multi-Layer Funnel-Band Subspace Clamping",
        "",
        "| Clamp Configuration | Layers Clamped | `98/2` Reject LO | `98/2` Delta LO | Tainted Delta LO | Retal Ctrl / Prov CSI | Retal `NT DeltaCSI` | Cashflow Sanity |",
        "| :--- | :---: | ---: | ---: | ---: | :---: | ---: | ---: |",
    ])
    for k, r in b5.items():
        rng_lbl = f"L{r['layers_range'][0]}..L{r['layers_range'][1]}" if r.get("layers_range") else "none"
        lines.append(
            f"| `{k}` | `{r['n_layers_clamped']}` (`{rng_lbl}`) | `{_fmt_signed(r['ultimatum_98_2_reject_lo'], 2)}` | "
            f"`{_fmt_signed(r['ultimatum_delta_lo'], 2)}` | `{_fmt_signed(r['tainted_delta_lo'], 2)}` | "
            f"`{r['retaliation_ctrl_csi']:.1f}% / {r['retaliation_prov_csi']:.1f}%` | **`{_fmt_signed(r['retaliation_delta_csi'], 1)}%`** | "
            f"`{r['cashflow_sanity_acc']*100:.1f}%` |"
        )

    lines.append("")
    return "\n".join(_clean_text(ln) for ln in lines)


def build_funnel_html(results: dict[str, Any], md_text: str) -> str:
    meta = results.get("metadata", {})
    alg = results.get("block0_algebra", {})
    pl = meta.get("primary_layer", 35)
    b3 = results.get("block3_goodfire_and_read_write", {})
    b4 = results.get("block4_causal_ablations", {})
    b5 = results.get("block5_band_clamping", {})

    abl_rows = []
    for cname, crow in b4.items():
        abl_rows.append(
            f"<tr><td><code>{html.escape(cname)}</code></td>"
            f"<td>{_fmt_signed(crow['ultimatum_98_2_reject_lo'], 2)}</td>"
            f"<td><strong>{_fmt_signed(crow['ultimatum_delta_lo_vs_50_50'], 2)}</strong></td>"
            f"<td>{_fmt_signed(crow['ultimatum_spite_proj_l35'], 2)}</td>"
            f"<td>{_fmt_signed(crow['tainted_delta_lo'], 2)}</td>"
            f"<td><strong>{_fmt_signed(crow['vignette_spite_proj_l35'], 2)}</strong></td>"
            f"<td>{crow['cashflow_sanity_acc']*100:.1f}%</td></tr>"
        )

    clamp_rows = []
    for k, r in b5.items():
        rng_lbl = f"L{r['layers_range'][0]}..L{r['layers_range'][1]}" if r.get("layers_range") else "none"
        clamp_rows.append(
            f"<tr><td><code>{html.escape(k)}</code></td>"
            f"<td>{r['n_layers_clamped']} ({rng_lbl})</td>"
            f"<td>{_fmt_signed(r['ultimatum_delta_lo'], 2)}</td>"
            f"<td>{_fmt_signed(r['tainted_delta_lo'], 2)}</td>"
            f"<td>{r['retaliation_ctrl_csi']:.1f}% / {r['retaliation_prov_csi']:.1f}%</td>"
            f"<td><strong>{_fmt_signed(r['retaliation_delta_csi'], 1)}%</strong></td>"
            f"<td>{r['cashflow_sanity_acc']*100:.1f}%</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 7 Read/Write Funnel Dissection - Project Afriat</title>
<link rel="stylesheet" href="{FONTS_CSS}">
<style>
  :root {{
    --ivory: {IVORY};
    --parchment: {PARCHMENT};
    --onyx: {ONYX};
    --slate-deep: {SLATE_DEEP};
    --blue-slate: {BLUE_SLATE};
    --slate-mist: {SLATE_MIST};
    --golden-orange: {GOLDEN_ORANGE};
  }}
  body {{
    background: var(--ivory);
    color: var(--onyx);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 17px;
    line-height: 1.6;
    max-width: 1080px;
    margin: 0 auto;
    padding: 40px 24px 80px;
  }}
  h1, h2 {{
    font-family: 'Newsreader', Georgia, serif;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: var(--onyx);
  }}
  h1 {{ font-size: 2.2rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.45rem; margin-top: 2.2rem; border-bottom: 1px solid rgba(44,62,80,0.22); padding-bottom: 6px; }}
  .kicker {{
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--slate-deep);
  }}
  .well {{
    background: var(--parchment);
    border: 2px solid var(--blue-slate);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 20px 0;
  }}
  img.hero {{
    width: 100%;
    border: 2px solid var(--blue-slate);
    border-radius: 10px;
    margin: 18px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
    font-size: 0.92rem;
    margin: 14px 0 24px;
  }}
  th, td {{
    border-bottom: 1px solid rgba(44,62,80,0.22);
    padding: 8px 10px;
    text-align: right;
  }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--slate-deep); font-weight: 600; border-bottom: 2px solid var(--blue-slate); }}
  code, pre {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.86em;
  }}
  details {{
    background: var(--parchment);
    border: 1px solid rgba(44,62,80,0.35);
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 18px;
  }}
  summary {{ cursor: pointer; font-weight: 600; color: var(--slate-deep); }}
</style>
</head>
<body>
  <div class="kicker">project afriat · phase 7 mechanistic circuit test</div>
  <h1>Read-Divergent, Write-Convergent Moral Funnel in {html.escape(str(meta.get('model_id', '')))}</h1>
  <div class="well">
    <strong>Model:</strong> <code>{html.escape(str(meta.get('model_id', '')))}</code> ·
    <strong>Primary Layer:</strong> <code>L{pl}</code> ·
    <strong>Total MLP Neurons:</strong> <code>{alg.get('total_mlp_neurons', 0):,}</code> ·
    <strong>Telescoping Conservation R&sup2;:</strong> <code>{alg.get('telescoping_verification', {}).get('r_squared', 1.0):.6f}</code>
  </div>
  <img class="hero" src="fig_funnel_read_write.png" alt="Phase 7 Read/Write Funnel Overview">

  <h2>Bidirectional Causal Neuron Ablation &amp; Sufficiency Matrix</h2>
  <table>
    <thead>
      <tr>
        <th>Intervention</th>
        <th>98/2 Reject LO</th>
        <th>98/2 Delta LO</th>
        <th>98/2 L{pl} Spite</th>
        <th>Tainted Delta LO</th>
        <th>Vignette L{pl} Spite</th>
        <th>Sanity</th>
      </tr>
    </thead>
    <tbody>
      {''.join(abl_rows)}
    </tbody>
  </table>

  <h2>Single-Layer (L{pl}) vs. Funnel-Band Subspace Clamping</h2>
  <table>
    <thead>
      <tr>
        <th>Configuration</th>
        <th>Layers</th>
        <th>98/2 Delta LO</th>
        <th>Tainted Delta LO</th>
        <th>Ctrl / Prov CSI</th>
        <th>NT Delta CSI</th>
        <th>Sanity</th>
      </tr>
    </thead>
    <tbody>
      {''.join(clamp_rows)}
    </tbody>
  </table>

  <details>
    <summary>Zero-Circularity Read/Write JSON Diagnostics &amp; Portable Markdown</summary>
    <pre>{html.escape(md_text)}</pre>
  </details>
</body>
</html>
"""


def render_funnel_bundle(results: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path, svg_path = plot_funnel_overview(results, out_dir)
    md_text = build_funnel_markdown(results)
    md_path = out_dir / "funnel_report.md"
    md_path.write_text(md_text, encoding="utf-8")
    html_path = out_dir / "funnel_report.html"
    html_path.write_text(build_funnel_html(results, md_text), encoding="utf-8")
    return {
        "png": png_path,
        "svg": svg_path,
        "md": md_path,
        "html": html_path,
    }
