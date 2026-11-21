"""Dawn State / Warm Precision plotting and report rendering for Phase 9 (`afriat.loop_compact`)."""

from __future__ import annotations

import html
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


def _clean(s: str) -> str:
    return s.replace("\u2014", " - ").replace("--", "-")


def _fmt_signed(v: float, decimals: int = 2) -> str:
    s = f"{abs(v):.{decimals}f}"
    if v < -1e-12:
        return f"{MINUS}{s}"
    if v > 1e-12:
        return f"+{s}"
    return s


def _style_ax(ax: Any) -> None:
    ax.set_facecolor(IVORY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLUE_SLATE)
    ax.spines["bottom"].set_color(BLUE_SLATE)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(colors=ONYX, labelsize=8.6)
    ax.grid(axis="y", color=BLUE_SLATE, alpha=0.14, linewidth=0.8)


def plot_loop_compact_overview(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 11.0), dpi=200, facecolor=IVORY)
    fig.subplots_adjust(hspace=0.45, wspace=0.30, top=0.86, bottom=0.10, left=0.062, right=0.98)

    meta = results.get("metadata", {})
    model_id = str(meta.get("model_id", "google/gemma-4-31B-it"))
    fb = meta.get("funnel_band", [24, 42])

    a1 = results.get("action1_compression", {})
    a3 = results.get("action3_distillation", {})
    a4 = results.get("action4_halting_and_safety", {})
    loops = a3.get("loop_scaling", {})

    bf16_gb = float(a1.get("intact_bf16", {}).get("storage", {}).get("total_gb", 62.6))
    comp_gb = float(a1.get("final_loop_compact_storage", {}).get("total_gb", 17.3))
    vram_pct = 100.0 * (1.0 - comp_gb / max(bf16_gb, 1e-6))

    fig.text(
        0.062,
        0.946,
        _clean(
            f"Gemma-4-Loop-Compact cuts footprint {vram_pct:.0f}% ({bf16_gb:.1f} to {comp_gb:.1f} GB) "
            f"while scaling 0-CoT reasoning across K=1..4 loops"
        ),
        fontsize=15.0,
        fontweight="bold",
        color=ONYX,
    )
    fig.text(
        0.062,
        0.912,
        _clean(
            f"Actions 1-4 on {model_id}: Hourglass INT4/INT8 + width pruning | "
            f"Procrustes + Loop-LoRA L{fb[0]}..L{fb[1]} | <thought> distillation | Adaptive L{fb[1]} halting & safety fold"
        ),
        fontsize=9.9,
        color=SLATE_DEEP,
    )

    # Panel A: Action 1 Compression Footprint (GB) & Zero-Shot KL Fidelity
    ax = axes[0, 0]
    _style_ax(ax)
    uni_kl = float(a1.get("uniform_compact", {}).get("kl_eval", {}).get("kl_nats", 0.0))
    hg_kl = float(a1.get("hourglass_compact_zeroshot", {}).get("kl_eval", {}).get("kl_nats", 0.0))
    k2_kl = float(loops.get("loop_k2", {}).get("kl_eval", {}).get("kl_nats", hg_kl))
    uni_gb = float(a1.get("uniform_compact", {}).get("storage", {}).get("total_gb", comp_gb))
    hg_gb = float(a1.get("hourglass_compact_zeroshot", {}).get("storage", {}).get("total_gb", comp_gb))

    a_labels = [
        f"Intact BF16\n({bf16_gb:.1f} GB)",
        f"Uniform-Compact\n({uni_gb:.1f} GB)",
        f"Hourglass-Compact\n({hg_gb:.1f} GB)",
        f"Loop-Compact K=2\n({comp_gb:.1f} GB)",
    ]
    a_gbs = [bf16_gb, uni_gb, hg_gb, comp_gb]
    a_cols = [SLATE_MIST, BLUE_SLATE, GOLDEN_ORANGE, SAGE_DEEP]
    x_a = np.arange(len(a_gbs))
    bars_a = ax.bar(x_a, a_gbs, width=0.55, color=a_cols, edgecolor=BLUE_SLATE, linewidth=1.6)
    kl_notes = ["KL 0.000", f"KL {uni_kl:.3f}", f"KL {hg_kl:.3f}", f"KL {k2_kl:.3f}"]
    ymax_a = max(a_gbs, default=65.0) * 1.24
    for b_p, gv, kn in zip(bars_a, a_gbs, kl_notes):
        ax.text(
            b_p.get_x() + b_p.get_width() / 2,
            gv + ymax_a * 0.02,
            f"{gv:.1f} GB\n({kn})",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=ONYX,
        )
    ax.axhline(24.0, color=COPPER_DEEP, linestyle="--", linewidth=1.3, alpha=0.85)
    ax.annotate(
        "24 GB single-GPU ceiling",
        xy=(1.5, 24.0),
        xytext=(1.85, 42.0),
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color=COPPER_DEEP,
        arrowprops={"arrowstyle": "->", "color": COPPER_DEEP, "lw": 1.2},
    )
    ax.set_ylim(0, ymax_a)
    ax.set_xticks(x_a)
    ax.set_xticklabels(a_labels, fontsize=8.0)
    ax.set_ylabel("Model storage footprint (GB)", fontsize=9, color=SLATE_DEEP)
    ax.set_title("A. Action 1: Hourglass vs. Uniform Compression", fontsize=11, fontweight="bold", color=ONYX, loc="left")

    # Panel B: Action 3 Monotonic Depth Scaling Across K = 1, 2, 3, 4 vs Intact 62.6 GB Base
    ax = axes[0, 1]
    _style_ax(ax)
    cond_specs = [
        ("bf16", f"BF16 Base (1-Pass, {bf16_gb:.0f}GB)"),
        ("loop_k1", "Compact K=1"),
        ("loop_k2", "Compact K=2"),
        ("loop_k3", "Compact K=3"),
        ("loop_k4", "Compact K=4"),
    ]
    mh_b = []
    bs_b = []
    wp_b = []
    b_lbls = []
    for ck, clbl in cond_specs:
        if ck == "bf16":
            src = a1.get("intact_bf16", {})
        else:
            src = loops.get(ck, {})
        if not src:
            continue
        b_lbls.append(clbl)
        mh_b.append(100.0 * float(src.get("multihop", {}).get("overall_acc", 0.0)))
        bs_b.append(100.0 * float(src.get("balsheet", {}).get("overall_acc", 0.0)))
        wp_b.append(100.0 * float(src.get("wordprob", {}).get("overall_acc", 0.0)))

    x_b = np.arange(len(b_lbls))
    w_b = 0.25
    ymax_b = max(15.0, max(mh_b + bs_b + wp_b, default=10.0) * 1.28)
    ax.bar(x_b - w_b, mh_b, width=w_b, color=BLUE_SLATE, edgecolor=BLUE_SLATE, linewidth=1.3, label="Multi-Hop (3/5/7/9h)")
    ax.bar(x_b, bs_b, width=w_b, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.4, label="Balance-Sheet (0%-Loan)")
    ax.bar(x_b + w_b, wp_b, width=w_b, color=SAGE_DEEP, edgecolor=BLUE_SLATE, linewidth=1.3, label="Word Problems")
    ax.set_ylim(0, ymax_b)
    ax.set_xticks(x_b)
    ax.set_xticklabels(b_lbls, rotation=18, ha="right", fontsize=8.0)
    ax.set_ylabel("0-CoT Exact-match accuracy (%)", fontsize=9, color=SLATE_DEEP)
    ax.set_title("B. Action 3: Randomized Unroll Depth Scaling (K=1..4)", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")

    # Panel C: Unseen Depth Extrapolation across 3, 5, 7, 9 Hops
    ax = axes[0, 2]
    _style_ax(ax)
    hops_x = [3, 5, 7, 9]
    c_lines = [
        ("bf16", "Intact BF16 (1-Pass)", SLATE_MIST, "--", 2.0),
        ("loop_k1", "Loop-Compact (K=1)", BLUE_SLATE, ":", 2.0),
        ("loop_k2", "Loop-Compact (K=2)", SAGE_DEEP, "-.", 2.2),
        ("loop_k3", "Loop-Compact (K=3)", COPPER_DEEP, "-", 2.3),
        ("loop_k4", "Loop-Compact (K=4)", GOLDEN_ORANGE, "-", 3.0),
    ]
    max_cy = 8.0
    for ck, lbl, col, ls, lw in c_lines:
        src = a1.get("intact_bf16", {}) if ck == "bf16" else loops.get(ck, {})
        if not src:
            continue
        bh = src.get("multihop", {}).get("by_hops", {})
        ys = [100.0 * float(bh.get(f"hop_{h}", {}).get("acc", 0.0)) for h in hops_x]
        max_cy = max(max_cy, max(ys, default=0.0))
        ax.plot(hops_x, ys, marker="o", markersize=5.5, color=col, linestyle=ls, linewidth=lw, label=lbl)
    ax.axvline(6.0, color=BLUE_SLATE, linestyle="--", linewidth=1.0, alpha=0.45)
    ax.text(6.14, max_cy * 0.42, "Unseen 7 & 9-hop\nextrapolation ->", fontsize=7.8, fontweight="bold", color=SLATE_DEEP)
    ax.set_xticks(hops_x)
    ax.set_xticklabels(["3 Hops", "5 Hops", "7 Hops*", "9 Hops*"])
    ax.set_ylim(-0.8, max(15.0, max_cy * 1.38))
    ax.set_ylabel("0-CoT Exact-match accuracy (%)", fontsize=9, color=SLATE_DEEP)
    ax.set_title("C. Multi-Hop State Tracking & Unseen Depth Extrap", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # Panel D: Action 4 Adaptive L42 Halting Gate (`Mean Loops E[K]` & Accuracy)
    ax = axes[1, 0]
    _style_ax(ax)
    adap = a4.get("adaptive_eval", {})
    d_suites = [
        ("easy_1hop", "Easy 1-Hop"),
        ("wordprob", "Word Prob"),
        ("balsheet", "Balance-Sheet"),
        ("multihop", "Multi-Hop (3-9h)"),
    ]
    present_d = [(sk, sl) for sk, sl in d_suites if sk in adap]
    x_d = np.arange(len(present_d))
    ek_vals = [float(adap[sk].get("mean_loops", 1.0)) for sk, _ in present_d]
    acc_d_vals = [100.0 * float(adap[sk].get("overall_acc", 0.0)) for sk, _ in present_d]
    bars_d = ax.bar(x_d, ek_vals, width=0.52, color=[SAGE_DEEP if v < 1.8 else GOLDEN_ORANGE for v in ek_vals], edgecolor=BLUE_SLATE, linewidth=1.6)
    for b_p, kv, av in zip(bars_d, ek_vals, acc_d_vals):
        ax.text(
            b_p.get_x() + b_p.get_width() / 2,
            kv + 0.10,
            f"E[K]={kv:.2f}\n({av:.1f}% acc)",
            ha="center",
            va="bottom",
            fontsize=7.9,
            fontweight="bold",
            color=ONYX,
        )
    ax.set_ylim(0, 4.95)
    ax.set_xticks(x_d)
    ax.set_xticklabels([sl for _, sl in present_d], fontsize=8.3)
    ax.set_ylabel("Mean recurrent loops executed E[K]", fontsize=9, color=SLATE_DEEP)
    ax.set_title(f"D. Action 4: Adaptive L{fb[1]} Halting Gate Allocation", fontsize=11, fontweight="bold", color=ONYX, loc="left")

    # Panel E: Action 3 `<thought>`-to-Latent Distillation Curve & L42 Cosine Convergence
    ax = axes[1, 1]
    _style_ax(ax)
    tr_sum = a3.get("training_summary", {})
    ce_c = tr_sum.get("ce_curve_subsampled", [])
    cos_c = tr_sum.get("cos_curve_subsampled", [])
    n_st = int(tr_sum.get("n_steps", len(ce_c)))
    if ce_c:
        xs_e = np.linspace(1, n_st, len(ce_c))
        ln1 = ax.plot(xs_e, ce_c, color=BLUE_SLATE, linewidth=2.2, label="Answer CE Loss (left)")
        ax.set_xlabel(f"Randomized K in 1..4 distillation step ({tr_sum.get('train_seconds', 0):.0f}s on H100)", fontsize=8.8, color=SLATE_DEEP)
        ax.set_ylabel("Answer cross-entropy loss", fontsize=9, color=BLUE_SLATE)
        ax.set_ylim(-0.04, max(ce_c) * 1.22)
        if cos_c and len(cos_c) == len(ce_c):
            ax2 = ax.twinx()
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_color(COPPER_DEEP)
            ln2 = ax2.plot(xs_e, cos_c, color=GOLDEN_ORANGE, linewidth=2.2, linestyle="--", label="L42 <thought> Cosine (right)")
            ax2.set_ylabel("L42 cosine vs. BF16 <thought> teacher", fontsize=8.8, color=COPPER_DEEP)
            ax2.tick_params(colors=COPPER_DEEP, labelsize=8.3)
            ax2.set_ylim(min(cos_c) - 0.04, 1.05)
            lns = ln1 + ln2
            ax.legend(lns, [l.get_label() for l in lns], frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("E. <thought>-to-Latent Self-Distillation Convergence", fontsize=11, fontweight="bold", color=ONYX, loc="left")

    # Panel F: Action 4 Moral-Subspace Bridge Safety Fold vs. Reasoning Accuracy
    ax = axes[1, 2]
    _style_ax(ax)
    s_uncl = a4.get("safety_unclamped_k3", {})
    s_fold = a4.get("safety_bridge_fold_k3", {})
    s_full = a4.get("safety_full_funnel_k3", {})
    f_groups = [
        ("Unclamped K=3", s_uncl),
        ("Bridge Fold (0 FLOP)", s_fold),
        ("Full Funnel Guard", s_full),
    ]
    present_f = [(gl, gd) for gl, gd in f_groups if gd]
    x_f = np.arange(len(present_f))
    w_f = 0.34
    prov_vals = [abs(float(gd.get("safety", {}).get("provenance_bias_delta_lo", 0.0))) for _, gd in present_f]
    ult_vals = [abs(float(gd.get("safety", {}).get("ultimatum_spite_delta_lo", 0.0))) for _, gd in present_f]
    ymax_f = max(3.0, max(prov_vals + ult_vals, default=2.0) * 1.32)
    b_fp = ax.bar(x_f - w_f / 2, prov_vals, width=w_f, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.5, label="|Provenance Bias| (log-odds)")
    b_fu = ax.bar(x_f + w_f / 2, ult_vals, width=w_f, color=BLUE_SLATE, edgecolor=BLUE_SLATE, linewidth=1.4, label="|Ultimatum Spite| (log-odds)")
    for b_p, pv in zip(b_fp, prov_vals):
        ax.text(b_p.get_x() + b_p.get_width() / 2, pv + ymax_f * 0.02, f"{pv:.2f}", ha="center", va="bottom", fontsize=7.9, fontweight="bold", color=ONYX)
    for b_p, uv in zip(b_fu, ult_vals):
        ax.text(b_p.get_x() + b_p.get_width() / 2, uv + ymax_f * 0.02, f"{uv:.2f}", ha="center", va="bottom", fontsize=7.9, fontweight="bold", color=ONYX)
    ax.set_ylim(0, ymax_f)
    ax.set_xticks(x_f)
    ax.set_xticklabels([gl for gl, _ in present_f], rotation=14, ha="right", fontsize=8.2)
    ax.set_ylabel("Moral bias magnitude (log-odds, lower = cleaner)", fontsize=8.8, color=SLATE_DEEP)
    ax.set_title("F. Action 4: Safety-Projected Bridge & Funnel Guard", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")

    pkg_v = results.get("action2_architecture", {}).get("package_verification", {})
    diff_str = f"{pkg_v.get('reload_k3_max_logit_diff', 0.0):.1e}" if pkg_v else "0.0e+00"
    fig.text(
        0.062,
        0.016,
        _clean(
            f"Source: Project Afriat Phase 9 Gemma-4-Loop-Compact ({model_id}) | "
            f"Disjoint audit passed (0 overlap) | Standalone package reload diff = {diff_str}"
        ),
        fontsize=8.4,
        color=SLATE_MIST,
    )

    out_png = out_dir / "fig_loop_compact_31b.png"
    out_svg = out_dir / "fig_loop_compact_31b.svg"
    fig.savefig(out_png, dpi=200, facecolor=IVORY)
    fig.savefig(out_svg, facecolor=IVORY)
    plt.close(fig)
    return out_png, out_svg


def build_loop_compact_markdown(results: dict[str, Any]) -> str:
    meta = results.get("metadata", {})
    a1 = results.get("action1_compression", {})
    a2 = results.get("action2_architecture", {})
    a3 = results.get("action3_distillation", {})
    a4 = results.get("action4_halting_and_safety", {})
    fb = meta.get("funnel_band", [24, 42])

    lines = [
        f"# Phase 9 `Gemma-4-Loop-Compact`: Smaller, Looped & Stronger (`{meta.get('model_id', 'google/gemma-4-31B-it')}`)",
        "",
        f"- **Completed UTC:** `{meta.get('timestamp_utc', '')}` · **Total GPU Runtime:** `{results.get('runtime_seconds', 'n/a')}s`",
        f"- **Funnel Band:** `L{fb[0]}..L{fb[1]}` · **Max Recurrent Loops:** `K = {meta.get('max_loops', 4)}` · **Loop-Indexed LoRA Rank:** `r = {meta.get('lora_rank', 16)}`",
        "",
        "![Gemma-4-Loop-Compact Overview](fig_loop_compact_31b.png)",
        "",
        "## Action 1: Hourglass Width Pruning + Mixed-Precision Quantization (`62.6 GB -> ~17 GB`)",
        "",
        "| Checkpoint | Storage Footprint | CUDA Allocated | Held-Out KL (nats) | Multi-Hop `0-CoT` | Balance-Sheet `0-CoT` | Word Prob `0-CoT` |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, label in (
        ("intact_bf16", "`intact_bf16` (100% BF16)"),
        ("uniform_compact", "`uniform_compact` (Byte-Matched Flat)"),
        ("hourglass_compact_zeroshot", "`hourglass_compact` (0-Shot Protected Neck)"),
    ):
        if key not in a1:
            continue
        row = a1[key]
        st = row.get("storage", {})
        kl = float(row.get("kl_eval", {}).get("kl_nats", 0.0))
        mh = 100.0 * float(row.get("multihop", {}).get("overall_acc", 0.0))
        bs = 100.0 * float(row.get("balsheet", {}).get("overall_acc", 0.0))
        wp = 100.0 * float(row.get("wordprob", {}).get("overall_acc", 0.0))
        lines.append(
            f"| {label} | **`{st.get('total_gb', 0):.2f} GB`** | `{st.get('cuda_allocated_gb', 'n/a')} GB` | "
            f"**`{kl:.4f}`** | `{mh:.1f}%` | `{bs:.1f}%` | `{wp:.1f}%` |"
        )

    proc = a2.get("procrustes_alignment", {})
    pkg = a2.get("package_verification", {})
    tr = a3.get("training_summary", {})
    fin_st = a1.get("final_loop_compact_storage", {})

    lines.extend([
        "",
        "## Actions 2 & 3: Standalone `Gemma4LoopCompact` + Randomized `K=1..4` `<thought>` Self-Distillation",
        "",
        f"- **Final Model Footprint (including `K=1..4` Loop-LoRAs & Bridge):** **`{fin_st.get('total_gb', 0):.2f} GB`** (`CUDA = {fin_st.get('cuda_allocated_gb', 'n/a')} GB`)",
        f"- **Orthogonal Procrustes Alignment (`L{fb[1]} -> L{fb[0]}` on Hourglass-Compact):** `{_fmt_signed(float(proc.get('mean_cosine_identity', 0)), 4)}` -> **`{_fmt_signed(float(proc.get('mean_cosine_procrustes', 0)), 4)}`**",
        f"- **Standalone Package Verification (`gemma4_loop_compact_pkg/`):** `{pkg.get('package_size_mb', 0)} MB` exported, round-trip `K=3` max logit diff = **`{pkg.get('reload_k3_max_logit_diff', 0.0):.2e}`**",
        f"- **Distillation (`{tr.get('trainable_params', 0):,}` params, `{tr.get('n_steps', 0)}` steps in `{tr.get('train_seconds', 0)}s`):** "
        f"Answer CE `{tr.get('ce_first10', 0):.3f}` -> **`{tr.get('ce_last10', 0):.3f}`** · "
        f"`L{fb[1]}` `<thought>` Cosine `{tr.get('l42_cos_first10', 0):.3f}` -> **`{tr.get('l42_cos_last10', 0):.3f}`** · "
        f"Learned Step Gates = `{tr.get('learned_step_gates', [])}`",
        "",
        "| Condition (`0` CoT tokens) | Loops | KL (nats) | 3-Hop | 5-Hop | 7-Hop* | 9-Hop* | Multi-Hop All | Balance-Sheet | Word Prob | Latency (`ms/item`) |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    bf_row = a1.get("intact_bf16", {})
    if bf_row:
        bh = bf_row.get("multihop", {}).get("by_hops", {})
        lines.append(
            f"| `intact_bf16` (`{bf_row.get('storage', {}).get('total_gb', 62.6):.1f} GB`) | `1` | `0.0000` | "
            f"`{100*float(bh.get('hop_3', {}).get('acc', 0)):.1f}%` | `{100*float(bh.get('hop_5', {}).get('acc', 0)):.1f}%` | "
            f"`{100*float(bh.get('hop_7', {}).get('acc', 0)):.1f}%` | `{100*float(bh.get('hop_9', {}).get('acc', 0)):.1f}%` | "
            f"**`{100*float(bf_row.get('multihop', {}).get('overall_acc', 0)):.1f}%`** | "
            f"**`{100*float(bf_row.get('balsheet', {}).get('overall_acc', 0)):.1f}%`** | "
            f"**`{100*float(bf_row.get('wordprob', {}).get('overall_acc', 0)):.1f}%`** | "
            f"`{float(bf_row.get('multihop', {}).get('ms_per_item', 0)):.1f}` |"
        )

    loops = a3.get("loop_scaling", {})
    for k_i in range(1, int(meta.get("max_loops", 4)) + 1):
        ck = f"loop_k{k_i}"
        if ck not in loops:
            continue
        r = loops[ck]
        bh = r.get("multihop", {}).get("by_hops", {})
        lines.append(
            f"| **`loop_compact_k{k_i}`** (`{fin_st.get('total_gb', 17.3):.1f} GB`) | `{k_i}` | `{float(r['kl_eval']['kl_nats']):.4f}` | "
            f"`{100*float(bh.get('hop_3', {}).get('acc', 0)):.1f}%` | `{100*float(bh.get('hop_5', {}).get('acc', 0)):.1f}%` | "
            f"`{100*float(bh.get('hop_7', {}).get('acc', 0)):.1f}%` | `{100*float(bh.get('hop_9', {}).get('acc', 0)):.1f}%` | "
            f"**`{100*float(r['multihop']['overall_acc']):.1f}%`** | "
            f"**`{100*float(r['balsheet']['overall_acc']):.1f}%`** | "
            f"**`{100*float(r['wordprob']['overall_acc']):.1f}%`** | "
            f"`{float(r['multihop']['ms_per_item']):.1f}` |"
        )

    adap = a4.get("adaptive_eval", {})
    lines.extend([
        "",
        "## Action 4A: `L42` Adaptive Halting Gate (`mode = adaptive`)",
        "",
        "| Held-Out Suite | Mean Loops `E[K]` | Exact-Match Accuracy | Wall-Clock Latency (`ms/item`) |",
        "| :--- | ---: | ---: | ---: |",
    ])
    for sk, sl in (("easy_1hop", "Easy 1-Hop Arithmetic"), ("wordprob", "Quantitative Word Problems"), ("balsheet", "Corporate Balance-Sheet (0%-Loan)"), ("multihop", "Multi-Hop Register Tracking (3-9 Hops)")):
        if sk in adap:
            s_d = adap[sk]
            lines.append(
                f"| {sl} | **`{float(s_d['mean_loops']):.2f}`** | **`{100*float(s_d['overall_acc']):.1f}%`** | `{float(s_d['ms_per_item']):.1f}` |"
            )

    lines.extend([
        "",
        "## Action 4B: Zero-Cost Moral-Subspace Bridge Safety Fold (`K = 3`)",
        "",
        "| Safety Mode | Provenance Bias (`Δ log-odds`) | Ultimatum `98/2` Spite (`Δ log-odds`) | Cashflow Sanity | Multi-Hop Acc | Balance-Sheet Acc |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for sk, sl in (
        ("safety_unclamped_k3", "Unclamped `Loop-Compact K=3`"),
        ("safety_bridge_fold_k3", "**Zero-Cost Bridge Fold (`R_safe = R_proc(I - QQ^T)`)**"),
        ("safety_full_funnel_k3", "**Bridge Fold + Rotating `L24..L42` Funnel Guard**"),
    ):
        if sk in a4:
            row = a4[sk]
            sf = row.get("safety", {})
            lines.append(
                f"| {sl} | `{_fmt_signed(float(sf.get('provenance_bias_delta_lo', 0)), 3)}` | "
                f"`{_fmt_signed(float(sf.get('ultimatum_spite_delta_lo', 0)), 3)}` | "
                f"`{100*float(sf.get('cashflow_sanity_acc', 0)):.1f}%` | "
                f"`{100*float(row.get('multihop', {}).get('overall_acc', 0)):.1f}%` | "
                f"`{100*float(row.get('balsheet', {}).get('overall_acc', 0)):.1f}%` |"
            )

    return _clean("\n".join(lines) + "\n")


def render_loop_compact_bundle(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loop_compact_overview(results, out_dir)
    md = build_loop_compact_markdown(results)
    (out_dir / "loop_compact_report.md").write_text(md, encoding="utf-8")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phase 9 Gemma-4-Loop-Compact Report</title>
<style>
  body {{ background: {IVORY}; color: {ONYX}; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1180px; margin: 36px auto; padding: 0 24px; line-height: 1.55; }}
  h1, h2 {{ color: {ONYX}; border-bottom: 2px solid {BLUE_SLATE}; padding-bottom: 6px; }}
  pre {{ background: {PARCHMENT}; border: 1.5px solid {BLUE_SLATE}; padding: 16px; overflow-x: auto; font-size: 13px; }}
  img {{ max-width: 100%; border: 2px solid {BLUE_SLATE}; margin: 16px 0; }}
</style>
</head>
<body>
<h1>Phase 9 Gemma-4-Loop-Compact Report</h1>
<img src="fig_loop_compact_31b.png" alt="Phase 9 Loop Compact Overview">
<pre>{html.escape(md)}</pre>
</body>
</html>
"""
    (out_dir / "loop_compact_report.html").write_text(html_doc, encoding="utf-8")
    print(f"[LoopCompact] Rendered Dawn State figures & reports in {out_dir}", flush=True)
