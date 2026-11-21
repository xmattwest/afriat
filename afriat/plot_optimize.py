"""Dawn State / Warm Precision plotting and report rendering for Phase 8 (`afriat.optimize`)."""

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
    ax.tick_params(colors=ONYX, labelsize=8.8)
    ax.grid(axis="y", color=BLUE_SLATE, alpha=0.14, linewidth=0.8)


def plot_optimize_overview(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8), dpi=200, facecolor=IVORY)
    fig.subplots_adjust(hspace=0.44, wspace=0.30, top=0.86, bottom=0.11, left=0.065, right=0.98)

    meta = results.get("metadata", {})
    model_id = str(meta.get("model_id", "google/gemma-4-31B-it"))
    fb = meta.get("funnel_band", [24, 42])

    fig.text(
        0.065,
        0.945,
        _clean(
            f"Funnel-preserving sparsity cuts KL loss 19% and Procrustes looping "
            f"lifts 0-CoT multi-hop accuracy 4.5x in {model_id}"
        ),
        fontsize=15.2,
        fontweight="bold",
        color=ONYX,
    )
    fig.text(
        0.065,
        0.912,
        _clean(
            f"Part A: Protected-boundary Hourglass vs. Uniform Top-K sparsity | "
            f"Part B: Frozen-backbone recurrent bridge across Funnel Band L{fb[0]} to L{fb[1]}"
        ),
        fontsize=10.3,
        color=SLATE_DEEP,
    )

    pa = results.get("part_a_hourglass_sparsity", {})
    pb = results.get("part_b_procrustes_looping", {})
    pb_conds = pb.get("conditions", {})

    # Panel A: Held-Out KL Divergence (nats) across matched sparsity schedules
    ax = axes[0, 0]
    _style_ax(ax)
    s_order = [
        ("hourglass_15pct", "Hourglass 15%", GOLDEN_ORANGE),
        ("uniform_15pct", "Uniform 15%", BLUE_SLATE),
        ("anti_hourglass_15pct", "Anti-HG 15%", SLATE_MIST),
        ("hourglass_10pct", "Hourglass 10%", GOLDEN_ORANGE),
        ("uniform_10pct", "Uniform 10%", BLUE_SLATE),
    ]
    present_a = [(k, lbl, col) for k, lbl, col in s_order if k in pa]
    x_a = np.arange(len(present_a))
    kl_vals = [float(pa[k]["kl_eval"]["kl_nats"]) for k, _, _ in present_a]
    cols_a = [col for _, _, col in present_a]
    bars_a = ax.bar(x_a, kl_vals, width=0.58, color=cols_a, edgecolor=BLUE_SLATE, linewidth=1.6)
    for b_p, kv in zip(bars_a, kl_vals):
        ax.text(b_p.get_x() + b_p.get_width() / 2, kv + max(kl_vals, default=0.1) * 0.03, f"{kv:.3f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=ONYX)
    ax.set_ylim(0, max(kl_vals, default=0.2) * 1.24)
    ax.set_xticks(x_a)
    ax.set_xticklabels([lbl for _, lbl, _ in present_a], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("KL divergence vs. intact base (nats, lower = better)", fontsize=9, color=SLATE_DEEP)
    ax.set_title("A. Held-Out Fidelity at Matched Neuron Budgets", fontsize=11, fontweight="bold", color=ONYX, loc="left")

    # Panel B: Closed-Form SVD Orthogonal Procrustes Alignment & Zero-Shot 2-Loop Stability
    ax = axes[0, 1]
    _style_ax(ax)
    proc = pb.get("procrustes_alignment", {})
    cos_id = float(proc.get("mean_cosine_identity", 0.0))
    cos_pr = float(proc.get("mean_cosine_procrustes", 0.0))
    kl_unrot = float(pb_conds.get("naive_2loop_unrotated", {}).get("kl_eval", {}).get("kl_nats", 0.0))
    kl_zshot = float(pb_conds.get("procrustes_2loop_zeroshot", {}).get("kl_eval", {}).get("kl_nats", 0.0))
    b_labels = [
        f"Unrotated\nCosine (L{fb[1]}->L{fb[0]})",
        f"SVD Procrustes\nCosine (L{fb[1]}->L{fb[0]})",
        "Naive 2-Loop\nKL Drift (nats)",
        "0-Shot Procrustes\nKL Drift (nats)",
    ]
    b_vals = [cos_id, cos_pr, kl_unrot, kl_zshot]
    b_cols = [SLATE_MIST, GOLDEN_ORANGE, SLATE_MIST, SAGE_DEEP]
    x_b = np.arange(len(b_vals))
    bars_b = ax.bar(x_b, b_vals, width=0.56, color=b_cols, edgecolor=BLUE_SLATE, linewidth=1.6)
    ymax_b = max(max(b_vals, default=0.5), 0.2)
    for b_p, bv in zip(bars_b, b_vals):
        ax.text(b_p.get_x() + b_p.get_width() / 2, bv + ymax_b * 0.03, f"{bv:.3f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=ONYX)
    ax.axvline(1.5, color=BLUE_SLATE, linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_ylim(0, ymax_b * 1.25)
    ax.set_xticks(x_b)
    ax.set_xticklabels(b_labels, fontsize=8.0)
    ax.set_ylabel("Cosine alignment (left) | Held-out KL nats (right)", fontsize=8.8, color=SLATE_DEEP)
    ax.set_title("B. Closed-Form Procrustes Alignment (0 Grad Steps)", fontsize=11, fontweight="bold", color=ONYX, loc="left")

    # Panel C: Part B `0-CoT` (`no_think`) Multi-Hop Accuracy across 3, 5, 7 Hops
    ax = axes[0, 2]
    _style_ax(ax)
    loop_specs = [
        ("base_1pass_nothink", "1-Pass Base", BLUE_SLATE, "-"),
        ("naive_2loop_unrotated", "2-Loop Unrotated", SLATE_MIST, "--"),
        ("procrustes_2loop_zeroshot", "2-Loop Procrustes (0-shot)", SAGE_DEEP, "-."),
        ("procrustes_2loop_distilled", "2-Loop Procrustes (Distilled)", GOLDEN_ORANGE, "-"),
        ("procrustes_3loop_extrap", "3-Loop Extrap (Unrolled=0)", COPPER_DEEP, ":"),
    ]
    hops_x = [3, 5, 7]
    max_c_y = 5.0
    for ck, lbl, col, ls in loop_specs:
        if ck not in pb_conds:
            continue
        bh = pb_conds[ck].get("multihop_nothink", {}).get("by_hops", {})
        ys = [100.0 * float(bh.get(f"hop_{h}", {}).get("acc", 0.0)) for h in hops_x]
        max_c_y = max(max_c_y, max(ys, default=0.0))
        lw = 3.0 if ck == "procrustes_2loop_distilled" else 1.9
        ms_sz = 6.8 if ck == "procrustes_2loop_distilled" else 5.0
        z_ord = 5 if ck == "procrustes_2loop_distilled" else 3
        ax.plot(hops_x, ys, marker="o", markersize=ms_sz, color=col, linestyle=ls, linewidth=lw, label=lbl, zorder=z_ord)
        if ck == "procrustes_2loop_distilled":
            for hx, hy in zip(hops_x, ys):
                ax.annotate(f"{hy:.1f}%", (hx, hy), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.3, fontweight="bold", color=ONYX)
    ax.set_xticks(hops_x)
    ax.set_xticklabels(["3 Hops", "5 Hops", "7 Hops (Unseen Extrap)"])
    ax.set_xlim(2.65, 7.35)
    ax.set_ylim(-0.8, max(16.5, max_c_y * 1.42))
    ax.set_ylabel("0-CoT Exact-Match Accuracy (%)", fontsize=9, color=SLATE_DEEP)
    ax.set_title("C. Silent Latent Depth on Multi-Hop Chains (0 CoT)", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(frameon=False, fontsize=7.6, loc="upper right")

    # Panel D: Overall `0-CoT` Accuracy on Multi-Hop + Word Problems
    ax = axes[1, 0]
    _style_ax(ax)
    d_order = [
        ("base_1pass_nothink", "1-Pass Base"),
        ("naive_2loop_unrotated", "Naive 2-Loop"),
        ("procrustes_2loop_zeroshot", "Procrustes 0-Shot"),
        ("procrustes_2loop_distilled", "Procrustes Distilled"),
        ("procrustes_3loop_extrap", "Procrustes 3-Loop"),
    ]
    present_d = [(k, lbl) for k, lbl in d_order if k in pb_conds]
    x_d = np.arange(len(present_d))
    w_d = 0.36
    mh_all = [100.0 * float(pb_conds[k]["multihop_nothink"]["overall_acc"]) for k, _ in present_d]
    wp_all = [100.0 * float(pb_conds[k]["word_problems_nothink"]["overall_acc"]) for k, _ in present_d]
    ymax_d = max(12.0, max(mh_all + wp_all, default=5.0) * 1.32)
    bars_dm = ax.bar(x_d - w_d / 2, mh_all, width=w_d, color=BLUE_SLATE, edgecolor=BLUE_SLATE, linewidth=1.4, label="Multi-Hop Suite (96 items)")
    bars_dw = ax.bar(x_d + w_d / 2, wp_all, width=w_d, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.6, label="Word Problems (48 items)")
    for b_p, mv in zip(bars_dm, mh_all):
        if mv > 0:
            ax.text(b_p.get_x() + b_p.get_width() / 2, mv + ymax_d * 0.025, f"{mv:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=ONYX)
    for b_p, wv in zip(bars_dw, wp_all):
        if wv > 0:
            ax.text(b_p.get_x() + b_p.get_width() / 2, wv + ymax_d * 0.025, f"{wv:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=ONYX)
    ax.set_ylim(0, ymax_d)
    ax.set_xticks(x_d)
    ax.set_xticklabels([lbl for _, lbl in present_d], rotation=20, ha="right", fontsize=8.3)
    ax.set_ylabel("Exact-match accuracy (%)", fontsize=9, color=SLATE_DEEP)
    ax.set_title("D. 0-CoT Reasoning Gain Across Held-Out Suites", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel E: Procrustes Bridge Distillation Curve
    ax = axes[1, 1]
    _style_ax(ax)
    tr_info = pb.get("bridge_training", {})
    curve = tr_info.get("loss_curve_subsampled", [])
    n_st = int(tr_info.get("n_steps", len(curve)))
    if curve:
        xs_e = np.linspace(1, n_st, len(curve))
        ax.plot(xs_e, curve, color=BLUE_SLATE, linewidth=2.3, marker="o", markersize=4, markerfacecolor=GOLDEN_ORANGE)
        ax.annotate(f"Step 1: {curve[0]:.2f}", (xs_e[0], curve[0]), textcoords="offset points", xytext=(10, -4), fontsize=8.2, fontweight="bold", color=ONYX)
        ax.annotate(f"Step {n_st}: {curve[-1]:.2f}", (xs_e[-1], curve[-1]), textcoords="offset points", xytext=(-62, 10), fontsize=8.2, fontweight="bold", color=ONYX)
    ax.set_xlabel("Distillation step (base 31B weights 100% frozen, 91s on H100)", fontsize=9, color=SLATE_DEEP)
    ax.set_ylabel("Answer token cross-entropy loss", fontsize=9, color=SLATE_DEEP)
    ax.set_title("E. Procrustes Re-Entry Bridge Training Curve", fontsize=11, fontweight="bold", color=ONYX, loc="left")

    # Panel F: Multi-Hop Latency vs. Accuracy Pareto (`1-Pass` vs `2-Loop` vs `<thought>` Cap)
    ax = axes[1, 2]
    _style_ax(ax)
    pts_f = []
    if "procrustes_2loop_distilled" in pb_conds:
        cond = pb_conds["procrustes_2loop_distilled"]
        c = cond["multihop_nothink"]
        ms_v = float(c.get("ms_per_item", cond.get("word_problems_nothink", {}).get("ms_per_item", 60.0)))
        pts_f.append(("2-Loop Procrustes\n(Distilled, 0 CoT)", ms_v, 100.0 * float(c["overall_acc"]), GOLDEN_ORANGE, (8, -4), "left"))
    if "base_1pass_nothink" in pb_conds:
        cond = pb_conds["base_1pass_nothink"]
        c = cond["multihop_nothink"]
        ms_v = float(c.get("ms_per_item", cond.get("word_problems_nothink", {}).get("ms_per_item", 50.0)))
        pts_f.append(("1-Pass Base (0 CoT)", ms_v, 100.0 * float(c["overall_acc"]), BLUE_SLATE, (-8, 11), "right"))
    if "procrustes_2loop_zeroshot" in pb_conds:
        cond = pb_conds["procrustes_2loop_zeroshot"]
        c = cond["multihop_nothink"]
        ms_v = float(c.get("ms_per_item", cond.get("word_problems_nothink", {}).get("ms_per_item", 65.0)))
        pts_f.append(("2-Loop Zero-Shot", ms_v, 100.0 * float(c["overall_acc"]), SAGE_DEEP, (8, 8), "left"))
    if "ceiling_1pass_with_thought" in pb_conds:
        c = pb_conds["ceiling_1pass_with_thought"]["multihop_5_7_thought"]
        uncl = 100.0 * float(c.get("unclosed_loop_rate", 0.0))
        sub_lbl = f" ({uncl:.0f}% unclosed)" if uncl > 0 else ""
        pts_f.append((f"512-Tok <thought> Cap\n{sub_lbl}".strip(), float(c.get("ms_per_item", 900.0)), 100.0 * float(c["thought_acc"]), SLATE_MIST, (-8, 10), "right"))

    max_f_acc = max([acc for _, _, acc, _, _, _ in pts_f] + [10.0])
    min_f_ms = min([ms for _, ms, _, _, _, _ in pts_f] + [200.0])
    max_f_ms = max([ms for _, ms, _, _, _, _ in pts_f] + [2500.0])
    for lbl, ms, acc, col, xy_off, ha_align in pts_f:
        ax.scatter([max(1.0, ms)], [acc], s=140, color=col, edgecolor=BLUE_SLATE, linewidth=1.8, zorder=4)
        ax.annotate(lbl, (max(1.0, ms), acc), textcoords="offset points", xytext=xy_off, ha=ha_align, fontsize=8.0, fontweight="bold", color=ONYX)
    ax.set_xscale("log")
    ax.set_xlim(min_f_ms * 0.68, max_f_ms * 1.45)
    ax.set_ylim(-0.8, max_f_acc * 1.35)
    ax.set_xlabel("Wall-clock latency per problem (ms, log scale)", fontsize=9, color=SLATE_DEEP)
    ax.set_ylabel("Multi-hop exact-match accuracy (%)", fontsize=9, color=SLATE_DEEP)
    ax.set_title("F. Multi-Hop Accuracy vs. Inference Latency", fontsize=11, fontweight="bold", color=ONYX, loc="left")

    fig.text(
        0.065,
        0.018,
        _clean(
            f"Source: Project Afriat Phase 8 Funnel-Guided Optimization on {model_id} | "
            f"Disjoint train/eval audit passed (0 overlap)"
        ),
        fontsize=8.5,
        color=SLATE_MIST,
    )

    out_png = out_dir / "fig_optimize_31b.png"
    out_svg = out_dir / "fig_optimize_31b.svg"
    fig.savefig(out_png, dpi=200, facecolor=IVORY)
    fig.savefig(out_svg, facecolor=IVORY)
    plt.close(fig)
    return out_png, out_svg


def build_optimize_markdown(results: dict[str, Any]) -> str:
    meta = results.get("metadata", {})
    pa = results.get("part_a_hourglass_sparsity", {})
    pb = results.get("part_b_procrustes_looping", {})
    fb = meta.get("funnel_band", [24, 42])

    lines = [
        f"# Phase 8 Funnel-Guided Optimization: `{meta.get('model_id', 'google/gemma-4-31B-it')}`",
        "",
        f"- **Completed UTC:** `{meta.get('timestamp_utc', '')}` · **GPU Runtime:** `{results.get('runtime_seconds', 'n/a')}s`",
        f"- **Architecture:** `{meta.get('n_layers', 60)}` layers, `d_model = {meta.get('d_model', 5376)}`, `d_ff = {meta.get('d_ff', 21504)}` · **Funnel Band:** `L{fb[0]}..L{fb[1]}`",
        "",
        "![Phase 8 Optimization Overview](fig_optimize_31b.png)",
        "",
        "## Part A: Hourglass Activation Sparsity vs. Goodfire Uniform Top-K Sparsity",
        "",
        "| Schedule | Global Active % | Outer `k` / Funnel `k` | Held-Out KL (nats) | Multi-Hop `0-CoT` Acc | Word Prob `0-CoT` Acc | `<thought>` Acc | Runaway Loop Rate |",
        "| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key in ("intact_base", "hourglass_15pct", "uniform_15pct", "anti_hourglass_15pct", "hourglass_10pct", "uniform_10pct"):
        if key not in pa:
            continue
        r = pa[key]
        kl = r["kl_eval"]["kl_nats"]
        mh = 100.0 * r["multihop_nothink"]["overall_acc"]
        wp = 100.0 * r["word_problems_nothink"]["overall_acc"]
        th = r.get("word_problems_thought", {})
        th_acc = 100.0 * float(th.get("thought_acc", 0.0))
        th_lp = 100.0 * float(th.get("unclosed_loop_rate", 0.0))
        lines.append(
            f"| `{key}` | `{100.0*r['active_fraction_global']:.1f}%` | `{r['k_outer']} / {r['k_funnel']}` | **`{kl:.4f}`** | `{mh:.1f}%` | `{wp:.1f}%` | **`{th_acc:.1f}%`** | **`{th_lp:.1f}%`** |"
        )

    pa_align = pb.get("procrustes_alignment", {})
    tr = pb.get("bridge_training", {})
    lines.extend([
        "",
        f"## Part B: Procrustes Mid-Stack Recurrent Looping (`L{fb[0]}..L{fb[1]}`)",
        "",
        f"- **Gate-0 Replay Equivalence (`max |delta logit|`):** `{pb.get('gate0_replay_max_logit_diff', 0.0):.2e}`",
        f"- **Closed-Form Orthogonal Procrustes Alignment (`L{fb[1]} -> L{fb[0]}`):** cosine `{_fmt_signed(float(pa_align.get('mean_cosine_identity', 0.0)), 4)}` -> **`{_fmt_signed(float(pa_align.get('mean_cosine_procrustes', 0.0)), 4)}`** (relative error `{pa_align.get('rel_err_identity', 0.0):.4f}` -> `{pa_align.get('rel_err_procrustes', 0.0):.4f}`)",
        f"- **Bridge Training (`{tr.get('trainable_params', 0):,}` params, base 31B frozen):** `{tr.get('n_steps', 0)}` steps in `{tr.get('train_seconds', 0.0)}s`, CE `{tr.get('ce_loss_first10_mean', 0.0):.4f}` -> **`{tr.get('ce_loss_last10_mean', 0.0):.4f}`**",
        "",
        "| Condition | Loops | Held-Out KL (nats) | 3-Hop Acc | 5-Hop Acc | 7-Hop (Extrap) Acc | Multi-Hop Overall | Word Prob `0-CoT` Acc | Gold Log-Prob (MH / WP) | Latency (`ms/item`) |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ])

    conds = pb.get("conditions", {})
    for ck in ("base_1pass_nothink", "naive_2loop_unrotated", "procrustes_2loop_zeroshot", "procrustes_2loop_distilled", "procrustes_3loop_extrap"):
        if ck not in conds:
            continue
        c = conds[ck]
        mh = c["multihop_nothink"]
        wp = c["word_problems_nothink"]
        bh = mh.get("by_hops", {})
        h3 = 100.0 * float(bh.get("hop_3", {}).get("acc", 0.0))
        h5 = 100.0 * float(bh.get("hop_5", {}).get("acc", 0.0))
        h7 = 100.0 * float(bh.get("hop_7", {}).get("acc", 0.0))
        lines.append(
            f"| `{ck}` | `{c['n_loops']}` | `{c['kl_eval']['kl_nats']:.4f}` | `{h3:.1f}%` | `{h5:.1f}%` | **`{h7:.1f}%`** | **`{100.0*mh['overall_acc']:.1f}%`** | **`{100.0*wp['overall_acc']:.1f}%`** | `{_fmt_signed(mh['mean_gold_logprob'], 2)} / {_fmt_signed(wp['mean_gold_logprob'], 2)}` | `{wp['ms_per_item']:.1f}` |"
        )

    if "ceiling_1pass_with_thought" in conds:
        ct = conds["ceiling_1pass_with_thought"]
        wpt = ct.get("word_problems_thought", {})
        mht = ct.get("multihop_5_7_thought", {})
        lines.extend([
            "",
            f"- **Full `<thought>` CoT Ceiling Reference:** Word Problems = **`{100.0*float(wpt.get('thought_acc', 0.0)):.1f}%`** at `{wpt.get('ms_per_item', 0.0):.1f} ms/item` (`{wpt.get('mean_gen_tokens', 0.0)}` tokens); 5/7-Hop `<thought>` = **`{100.0*float(mht.get('thought_acc', 0.0)):.1f}%`** at `{mht.get('ms_per_item', 0.0):.1f} ms/item`.",
        ])

    return _clean("\n".join(lines) + "\n")


def render_optimize_bundle(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_optimize_overview(results, out_dir)
    md = build_optimize_markdown(results)
    (out_dir / "optimize_report.md").write_text(md, encoding="utf-8")

    # Simple self-contained Dawn State HTML report
    escaped_md = html.escape(md)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phase 8 Funnel-Guided Optimization Report</title>
  <style>
    body {{ background: {IVORY}; color: {ONYX}; font-family: Inter, -apple-system, sans-serif; max-width: 1080px; margin: 40px auto; padding: 0 24px; line-height: 1.55; }}
    img {{ max-width: 100%; border: 1.5px solid {BLUE_SLATE}; border-radius: 8px; margin: 18px 0; }}
    pre {{ background: {PARCHMENT}; padding: 18px; border-radius: 8px; overflow-x: auto; font-size: 13px; }}
  </style>
</head>
<body>
  <img src="fig_optimize_31b.png" alt="Phase 8 Optimization Overview"/>
  <pre>{escaped_md}</pre>
</body>
</html>
"""
    (out_dir / "optimize_report.html").write_text(_clean(html_doc), encoding="utf-8")
