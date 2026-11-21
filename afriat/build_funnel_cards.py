"""Build the minimalistic Dawn State / Warm Precision figure deck for the
Goodfire vs. Afriat Read/Write Funnel Dissection on `google/gemma-4-31B-it`.

Outputs 4 clean, high-whitespace 2x Retina PNGs into `runs/funnel_cards/`:
  1. `funnel_card1_different_ears_same_megaphone.png` (1200x760 -> 2400x1520)
  2. `funnel_card2_two_regimes_across_depth.png`      (1200x760 -> 2400x1520)
  3. `funnel_card3_read_vs_write_asymmetry.png`       (1200x760 -> 2400x1520)
  4. `funnel_card4_layer_scalar_reset_curtain.png`    (1200x760 -> 2400x1520)
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

from afriat.build_social_cards import (
    BLUE_SLATE,
    COPPER_DEEP,
    FONTS_CSS,
    GOLDEN_ORANGE,
    IVORY,
    MINUS,
    ONYX,
    PARCHMENT,
    SAGE_DEEP,
    SLATE_DEEP,
    SLATE_MIST,
    render_png_via_chrome,
)
from afriat.schema import ROOT_DIR

MINIMAL_CSS = f"""
  :root {{
    --ivory: {IVORY};
    --parchment: {PARCHMENT};
    --onyx: {ONYX};
    --slate-deep: {SLATE_DEEP};
    --blue-slate: {BLUE_SLATE};
    --slate-mist: {SLATE_MIST};
    --golden-orange: {GOLDEN_ORANGE};
    --copper-deep: {COPPER_DEEP};
    --sage-deep: {SAGE_DEEP};
    --rule: rgba(44, 62, 80, 0.18);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    background: var(--ivory);
    color: var(--onyx);
    font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
  }}
  .card {{
    padding: 44px 56px 34px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: var(--ivory);
  }}
  .head {{
    margin-bottom: 10px;
  }}
  .kicker {{
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--slate-mist);
    margin-bottom: 8px;
  }}
  h1 {{
    font-family: Newsreader, Georgia, serif;
    font-weight: 600;
    font-size: 48px;
    line-height: 1.04;
    letter-spacing: -0.04em;
    color: var(--onyx);
    margin-bottom: 8px;
  }}
  h1 .accent {{ color: var(--copper-deep); }}
  .sub {{
    font-size: 18px;
    line-height: 1.4;
    color: var(--slate-deep);
    max-width: 74ch;
  }}
  .canvas {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
  }}
  .foot {{
    font-size: 13px;
    color: var(--slate-mist);
    border-top: 1px solid var(--rule);
    padding-top: 12px;
    display: flex;
    justify-content: space-between;
    font-variant-numeric: tabular-nums;
  }}
"""


def _wrap_minimal(title: str, w: int, h: int, extra_css: str, body: str) -> str:
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{FONTS_CSS}">
  <style>
    {MINIMAL_CSS}
    body, .card {{ width: {w}px; height: {h}px; }}
    {extra_css}
  </style>
</head>
<body>
  <div class="card">{body}</div>
</body>
</html>
"""
    if "\u2014" in doc or "&mdash;" in doc:
        raise ValueError(f"Em dash found in {title}")
    padded = " " + doc.lower().replace(">", " ").replace("<", " ") + " "
    for bad in (" we ", " our ", " ours "):
        if bad in padded:
            raise ValueError(f"Forbidden plural pronoun {bad!r} found in {title}")
    return doc


def _encode_heatmap_png(mat: np.ndarray, vmin: float, vmax: float, cmap_colors: list[str]) -> str:
    cmap = LinearSegmentedColormap.from_list("dawn_hm", cmap_colors)
    m = mat.copy()
    np.fill_diagonal(m, np.nan)
    cmap.set_bad(color=IVORY)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=240, facecolor=IVORY)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, facecolor=IVORY)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def card_1_different_ears_same_megaphone(wf: dict[str, Any]) -> tuple[str, int, int]:
    """Card 1 (1200x760): Ultra-clean 3-stage OR-funnel diagram."""
    tr = {r["layer"]: r for r in wf["transect"]}
    pr33 = tr[33]["moral_u35"]["spite"]["pr_read_gate_of_64"]
    sp_kz35 = tr[35]["moral_u35"]["spite"]["kurtosis_z_vs_500_random"]
    gu_kz35 = tr[35]["moral_u35"]["guilt"]["kurtosis_z_vs_500_random"]
    an_kz56 = tr[56]["goodfire_lexical_controls"]["tok_an"]["kurtosis_z_vs_500_random"]

    svg = f"""
    <svg viewBox="0 0 1088 450" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      <defs>
        <marker id="a-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
          <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{BLUE_SLATE}"/>
        </marker>
        <marker id="a-cop" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{COPPER_DEEP}"/>
        </marker>
      </defs>

      <!-- Minimal column headers -->
      <text x="18" y="34" font-size="13" font-weight="600" letter-spacing="1.4" fill="{SLATE_MIST}">1 · INPUT DETECTORS (L0–L24)</text>
      <text x="400" y="34" font-size="13" font-weight="600" letter-spacing="1.4" fill="{COPPER_DEEP}">2 · MORAL FUNNEL (L26–L35)</text>
      <text x="790" y="34" font-size="13" font-weight="600" letter-spacing="1.4" fill="{SLATE_MIST}">3 · OUTPUT (L56)</text>

      <!-- Subtle vertical hairlines -->
      <line x1="354" y1="18" x2="354" y2="430" stroke="rgba(44,62,80,0.14)" stroke-width="1"/>
      <line x1="744" y1="18" x2="744" y2="430" stroke="rgba(44,62,80,0.14)" stroke-width="1"/>

      <!-- Left: 3 simple disjoint input pills -->
      <rect x="18" y="88" width="270" height="66" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>
      <text x="38" y="117" font-size="17" font-weight="600" fill="{ONYX}">Sewage &amp; pettiness</text>
      <text x="38" y="139" font-size="13.5" fill="{SLATE_DEEP}">0 digits · 0 money words</text>

      <rect x="18" y="188" width="270" height="66" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>
      <text x="38" y="217" font-size="17" font-weight="600" fill="{ONYX}">$980 / $20 split</text>
      <text x="38" y="239" font-size="13.5" fill="{SLATE_DEEP}">0 emotional words</text>

      <rect x="18" y="288" width="270" height="66" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>
      <text x="38" y="317" font-size="17" font-weight="600" fill="{ONYX}">34°C stranger marble</text>
      <text x="38" y="339" font-size="13.5" fill="{SLATE_DEEP}">Sensory boundary cue</text>

      <text x="18" y="394" font-size="13.5" fill="{SLATE_MIST}">Never co-fire on the same prompt</text>
      <text x="18" y="414" font-size="13.5" font-weight="600" fill="{COPPER_DEEP}">-&gt; 0 edges in Goodfire co-activation graphs</text>

      <!-- Converging OR arrows -->
      <path d="M 290 121 C 350 121, 355 198, 408 206" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.4" marker-end="url(#a-slate)"/>
      <path d="M 290 221 L 408 221" fill="none" stroke="{COPPER_DEEP}" stroke-width="3.2" marker-end="url(#a-cop)"/>
      <path d="M 290 321 C 350 321, 355 244, 408 236" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.4" marker-end="url(#a-slate)"/>

      <!-- Center: One bold focal Golden Orange node -->
      <rect x="414" y="150" width="274" height="142" rx="14" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.6"/>
      <text x="551" y="191" text-anchor="middle" font-size="13" font-weight="700" letter-spacing="1.1" fill="{ONYX}">SHARED MORAL BUS</text>
      <text x="551" y="228" text-anchor="middle" font-size="27" font-weight="700" fill="{ONYX}">Layer 35</text>
      <text x="551" y="258" text-anchor="middle" font-size="15.5" font-weight="600" fill="{ONYX}">+{sp_kz35:.0f}z to +{gu_kz35:.0f}z W_down writers</text>

      <text x="551" y="336" text-anchor="middle" font-size="22" font-weight="700" fill="{ONYX}">{pr33:.0f} / 64</text>
      <text x="551" y="359" text-anchor="middle" font-size="14" fill="{SLATE_DEEP}">independent input directions</text>
      <text x="551" y="379" text-anchor="middle" font-size="14" fill="{SLATE_DEEP}">writing onto one shared axis</text>

      <!-- Outgoing arrows -->
      <path d="M 690 202 C 742 202, 748 148, 796 148" fill="none" stroke="{COPPER_DEEP}" stroke-width="3.0" marker-end="url(#a-cop)"/>
      <path d="M 690 242 C 742 242, 748 302, 796 302" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.4" stroke-dasharray="6,4" marker-end="url(#a-slate)"/>

      <!-- Right: 2 clean outcomes -->
      <rect x="802" y="96" width="268" height="104" rx="10" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <text x="822" y="130" font-size="18" font-weight="700" fill="{ONYX}">Dollar prices</text>
      <text x="822" y="156" font-size="14.5" fill="{SLATE_DEEP}">+$27K walk-away · $500 floor</text>
      <text x="822" y="179" font-size="13.5" font-weight="600" fill="{COPPER_DEEP}">Driven by continuous bus scalar</text>

      <rect x="802" y="252" width="268" height="104" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>
      <text x="822" y="286" font-size="18" font-weight="700" fill="{ONYX}">Goodfire word teams</text>
      <text x="822" y="312" font-size="14.5" fill="{SLATE_DEEP}">+{an_kz56:.0f}z on 'an' · 0z on morals</text>
      <text x="822" y="335" font-size="13.5" font-weight="600" fill="{SAGE_DEEP}">Scrubs prose to 0% emotion words</text>
    </svg>
    """

    body = f"""
    <div class="head">
      <div class="kicker">Project Afriat · Gemma 4 31B Real-Weight Dissection</div>
      <h1 style="font-size: 43px;">Unrelated neurons write onto <span class="accent">one shared moral bus.</span></h1>
      <div class="sub">Separate MLP detectors read unrelated contexts, write onto one shared Layer 35 moral direction, and hand off to late-layer word teams.</div>
    </div>
    <div class="canvas">{svg}</div>
    <div class="foot">
      <span>google/gemma-4-31B-it (172,032 inspected MLP neurons across L8 to L56)</span>
      <span>Sparse detection -&gt; Geometric value -&gt; Sparse words</span>
    </div>
    """
    return _wrap_minimal("Unrelated Input Neurons Write Onto One Shared Moral Bus", 1200, 760, "", body), 1200, 760


def card_2_two_regimes_across_depth(wf: dict[str, Any]) -> tuple[str, int, int]:
    """Card 2 (1200x760): Minimalist 2-curve depth crossover (normalized to % of peak for zero dual-axis clutter!)."""
    tr = wf["transect"]
    layers = [r["layer"] for r in tr]
    sp_kz = [max(0.0, r["moral_u35"]["spite"]["kurtosis_z_vs_500_random"]) for r in tr]
    ct_kz = [max(0.0, r["moral_u35"]["contamination"]["kurtosis_z_vs_500_random"]) for r in tr]
    an_kz = [max(0.0, r["goodfire_lexical_controls"]["tok_an"]["kurtosis_z_vs_500_random"]) for r in tr]

    # Plot each curve cleanly as share of its own peak (0% to 100%) with exact raw Z-scores labeled directly on peaks!
    sp_norm = [100.0 * v / max(sp_kz) for v in sp_kz]
    ct_norm = [100.0 * v / max(ct_kz) for v in ct_kz]
    an_norm = [100.0 * v / max(an_kz) for v in an_kz]

    x0, x1 = 76.0, 1020.0
    y0, y1 = 44.0, 396.0

    def sx(ell: float) -> float:
        return x0 + (ell - 6.0) / (58.0 - 6.0) * (x1 - x0)

    def sy(pct: float) -> float:
        return y1 - (pct / 105.0) * (y1 - y0)

    def poly(ys: list[float]) -> str:
        return " ".join(f"{sx(l):.1f},{sy(v):.1f}" for l, v in zip(layers, ys))

    ticks_x = "".join(
        f'<line x1="{sx(l):.1f}" y1="{y1}" x2="{sx(l):.1f}" y2="{y1+7}" stroke="{BLUE_SLATE}" stroke-width="1.5"/>'
        f'<text x="{sx(l):.1f}" y="{y1+27}" text-anchor="middle" font-size="14" font-weight="600" fill="{ONYX}">L{l}</text>'
        for l in layers
    )
    ticks_y = "".join(
        f'<line x1="{x0}" y1="{sy(p):.1f}" x2="{x1}" y2="{sy(p):.1f}" stroke="rgba(44,62,80,0.10)" stroke-width="1"/>'
        f'<text x="{x0-12}" y="{sy(p)+5:.1f}" text-anchor="end" font-size="13" fill="{SLATE_MIST}">{p}%</text>'
        for p in (0, 50, 100)
    )

    dots_sp = "".join(f'<circle cx="{sx(l):.1f}" cy="{sy(v):.1f}" r="6" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2"/>' for l, v in zip(layers, sp_norm))
    dots_an = "".join(f'<circle cx="{sx(l):.1f}" cy="{sy(v):.1f}" r="5" fill="{BLUE_SLATE}" stroke="{IVORY}" stroke-width="1.6"/>' for l, v in zip(layers, an_norm))

    svg = f"""
    <svg viewBox="0 0 1088 446" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      {ticks_y}
      <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      {ticks_x}

      <!-- Subtle contamination ghost curve -->
      <polyline fill="none" stroke="{COPPER_DEEP}" stroke-width="2" stroke-opacity="0.38" points="{poly(ct_norm)}"/>
      <!-- Primary curves -->
      <polyline fill="none" stroke="{GOLDEN_ORANGE}" stroke-width="4.5" points="{poly(sp_norm)}"/>
      <polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="3.4" stroke-dasharray="8,5" points="{poly(an_norm)}"/>

      {dots_sp}
      {dots_an}

      <!-- Clean direct labels (no boxes!) -->
      <text x="{sx(35) - 18:.1f}" y="{sy(100) + 4:.1f}" text-anchor="end" font-size="16.5" font-weight="700" fill="{COPPER_DEEP}">Layer 35: Moral writers peak</text>
      <text x="{sx(35) - 18:.1f}" y="{sy(100) + 25:.1f}" text-anchor="end" font-size="13.5" fill="{SLATE_DEEP}">Spite +52z · Contam +81z · Guilt +89z</text>

      <text x="{sx(56) - 42:.1f}" y="{sy(100) + 4:.1f}" text-anchor="end" font-size="16.5" font-weight="700" fill="{BLUE_SLATE}">Layer 56: Token teams</text>
      <text x="{sx(56) - 42:.1f}" y="{sy(100) + 25:.1f}" text-anchor="end" font-size="13.5" fill="{SLATE_DEEP}">' an' kurtosis +414z</text>

      <text x="{sx(56) - 24:.1f}" y="{sy(0) - 16:.1f}" text-anchor="end" font-size="13.5" font-weight="600" fill="{COPPER_DEEP}">Moral writers drop to 0z</text>

      <!-- Minimal inline legend in open left space -->
      <line x1="{x0+14}" y1="158" x2="{x0+46}" y2="158" stroke="{GOLDEN_ORANGE}" stroke-width="4.5"/>
      <text x="{x0+56}" y="163" font-size="14" font-weight="600" fill="{ONYX}">Moral bus writers (% of peak)</text>
      <line x1="{x0+14}" y1="186" x2="{x0+46}" y2="186" stroke="{BLUE_SLATE}" stroke-width="3.4" stroke-dasharray="8,5"/>
      <text x="{x0+56}" y="191" font-size="14" font-weight="600" fill="{BLUE_SLATE}">Goodfire lexical ' an' (% of peak)</text>
    </svg>
    """

    body = f"""
    <div class="head">
      <div class="kicker">Depth Division of Labor · Gemma 4 31B</div>
      <h1 style="font-size: 42px;">Middle layers write <span class="accent">morals.</span> Final layers police words.</h1>
      <div class="sub">Moral-bus specialization peaks at Layer 35 and vanishes by Layer 56, right where sparse token-level neurons take over.</div>
    </div>
    <div class="canvas">{svg}</div>
    <div class="foot">
      <span>Excess kurtosis Z-score of post-RMSNorm W_down cosines across 21,504 MLP neurons/layer (vs. 500 random directions)</span>
      <span>L35 Spite vs. ' unfair' overlap: 0.79%</span>
    </div>
    """
    return _wrap_minimal("Middle Layers Write Morals", 1200, 760, "", body), 1200, 760


def card_3_read_vs_write_asymmetry(wf: dict[str, Any], mat_npz: Path) -> tuple[str, int, int]:
    """Card 3 (1200x760): Ultra-minimalist 64x64 heatmaps diptych with ONE big number under each."""
    data = np.load(mat_npz)
    read_uri = _encode_heatmap_png(
        np.abs(data["read_sim"]),
        vmin=0.0,
        vmax=0.14,
        cmap_colors=[IVORY, PARCHMENT, SLATE_MIST, BLUE_SLATE],
    )
    write_uri = _encode_heatmap_png(
        data["write_sim"],
        vmin=-0.06,
        vmax=0.10,
        cmap_colors=[BLUE_SLATE, IVORY, GOLDEN_ORANGE, COPPER_DEEP],
    )

    r33 = next(r for r in wf["transect"] if r["layer"] == 33)
    sp33 = r33["moral_u35"]["spite"]
    rb33 = r33["random_baselines"]

    extra_css = """
    .diptych {
      width: 100%;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 64px;
      align-items: start;
      padding: 8px 28px 0;
    }
    .side {
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
    }
    .side-tag {
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.11em;
      text-transform: uppercase;
      color: var(--slate-deep);
      margin-bottom: 12px;
    }
    .side-tag.copper { color: var(--copper-deep); }
    .hm {
      width: 320px;
      height: 320px;
      border: 2px solid var(--blue-slate);
      border-radius: 8px;
      display: block;
      margin-bottom: 18px;
    }
    .big-num {
      font-family: Newsreader, Georgia, serif;
      font-size: 46px;
      font-weight: 600;
      line-height: 1.0;
      letter-spacing: -0.03em;
      color: var(--onyx);
      font-variant-numeric: tabular-nums;
      margin-bottom: 6px;
    }
    .big-num.copper { color: var(--copper-deep); }
    .cap {
      font-size: 15.5px;
      line-height: 1.38;
      color: var(--slate-deep);
      max-width: 34ch;
    }
    """

    body = f"""
    <div class="head">
      <div class="kicker">Gemma 4 31B · Layer 33 Top-64 Spite Writers</div>
      <h1>Chaos on input. <span class="accent">Consensus</span> on output.</h1>
      <div class="sub">Pairwise weight similarity across the exact same 64 neurons: unrelated input triggers funneling onto one shared output direction.</div>
    </div>
    <div class="canvas">
      <div class="diptych">
        <div class="side">
          <div class="side-tag">Input Ears (W_gate · 64 &times; 64)</div>
          <img class="hm" src="{read_uri}" alt="Layer 33 W_gate cosine matrix">
          <div class="big-num">{sp33['pr_read_gate_of_64']:.0f} / 64</div>
          <div class="cap">independent input axes (|cos| {sp33['read_gate_offdiag_abs_cos']:.3f} vs. {rb33['rand64_gate_abs_cos']:.3f} random)</div>
        </div>
        <div class="side">
          <div class="side-tag copper">Output Mouths (W_down · 64 &times; 64)</div>
          <img class="hm" src="{write_uri}" alt="Layer 33 W_down cosine matrix">
          <div class="big-num copper">+{sp33['top64_rms_z_vs_500_random']:.1f}z</div>
          <div class="cap">aligned onto Layer 35 Spite (&gt;100&times; signed coherence vs. random)</div>
        </div>
      </div>
    </div>
    <div class="foot">
      <span>Exact real weights of google/gemma-4-31B-it Layer 33 (pre/post-RMSNorm scaled)</span>
      <span>p = 0.002 vs. 500 random directions</span>
    </div>
    """
    return _wrap_minimal("Chaos on Input, Consensus on Output", 1200, 760, extra_css, body), 1200, 760


def card_4_layer_scalar_reset_curtain(wf: dict[str, Any], p0_results: dict[str, Any]) -> tuple[str, int, int]:
    """Card 4 (1200x760): Single unified chart pairing Skip-Gain Curtain with Cross-Firewall Spite Ignition."""
    gam36 = wf["skip_gain_to_l35_0_to_35"]
    cs = p0_results["cartography_scenario"]
    ai = cs["archetypes"].index("insulting_split")
    ci = cs["core_concepts"].index("spite")
    dz60 = cs["dz_controlled_curves"][ai][ci]

    x0, x1 = 76.0, 1026.0
    y0, y1 = 44.0, 396.0

    def sx(ell: float) -> float:
        return x0 + (ell / 59.0) * (x1 - x0)

    # Left axis: Cross-firewall Spite dz (-1..11); Overlay: Skip survival scaled to 0..10 (100% -> 10)
    def sy(dz: float) -> float:
        clamped = max(-1.2, min(11.0, dz))
        return y1 - (clamped - (-1.2)) / (11.0 - (-1.2)) * (y1 - y0)

    p_dz = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(dz60))
    p_gam = " ".join(f"{sx(i):.1f},{sy(v * 10.0):.1f}" for i, v in enumerate(gam36))
    area_gam = f"{sx(0):.1f},{sy(0.0):.1f} " + p_gam + f" {sx(35):.1f},{sy(0.0):.1f}"

    ticks_x = "".join(
        f'<line x1="{sx(l):.1f}" y1="{y1}" x2="{sx(l):.1f}" y2="{y1+7}" stroke="{BLUE_SLATE}" stroke-width="1.5"/>'
        f'<text x="{sx(l):.1f}" y="{y1+27}" text-anchor="middle" font-size="14" font-weight="600" fill="{ONYX}">L{l}</text>'
        for l in (0, 10, 20, 25, 30, 35, 45, 55, 59)
    )
    ticks_y = "".join(
        f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x1}" y2="{sy(v):.1f}" stroke="rgba(44,62,80,0.10)" stroke-width="1"/>'
        f'<text x="{x0-12}" y="{sy(v)+5:.1f}" text-anchor="end" font-size="13" fill="{SLATE_MIST}">+{v} dz</text>'
        for v in (0, 5, 10)
    )

    svg = f"""
    <svg viewBox="0 0 1088 446" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      {ticks_y}
      <line x1="{x0}" y1="{sy(0.0):.1f}" x2="{x1}" y2="{sy(0.0):.1f}" stroke="{SLATE_MIST}" stroke-width="1.2" stroke-dasharray="4,4"/>
      <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      {ticks_x}

      <!-- Golden Orange filled skip-survival curtain -->
      <polygon points="{area_gam}" fill="rgba(243,156,18,0.28)"/>
      <polyline fill="none" stroke="{GOLDEN_ORANGE}" stroke-width="3.6" points="{p_gam}"/>

      <!-- Blue Slate cross-firewall Spite dz curve -->
      <polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="3.8" points="{p_dz}"/>

      <circle cx="{sx(20):.1f}" cy="{sy(dz60[20]):.1f}" r="6" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>
      <circle cx="{sx(35):.1f}" cy="{sy(dz60[35]):.1f}" r="7.5" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>

      <!-- 3 clean direct annotations -->
      <text x="{sx(20) - 8:.1f}" y="{sy(2.0):.1f}" text-anchor="end" font-size="14" font-weight="600" fill="{SLATE_DEEP}">L20: {MINUS}0.35 dz (0.6% survival)</text>
      <text x="{sx(35) + 32:.1f}" y="{sy(10.3):.1f}" text-anchor="start" font-size="17" font-weight="700" fill="{ONYX}">L35 peak: +10.00 dz</text>
      <text x="{sx(35) + 16:.1f}" y="{sy(4.2):.1f}" text-anchor="start" font-size="14.5" font-weight="700" fill="{COPPER_DEEP}">Skip survival jumps to 68%</text>

      <!-- Minimal legend in open left space -->
      <line x1="{x0+14}" y1="126" x2="{x0+46}" y2="126" stroke="{BLUE_SLATE}" stroke-width="3.8"/>
      <text x="{x0+56}" y="131" font-size="14" font-weight="600" fill="{ONYX}">95/5 financial split -&gt; Spite (Cohen dz)</text>
      <rect x="{x0+14}" y="147" width="32" height="12" rx="2" fill="rgba(243,156,18,0.45)" stroke="{GOLDEN_ORANGE}" stroke-width="2"/>
      <text x="{x0+56}" y="158" font-size="14" font-weight="600" fill="{COPPER_DEEP}">Skip-connection survival to L35</text>
    </svg>
    """

    body = f"""
    <div class="head">
      <div class="kicker">Gemma 4 31B Residual Horizon · Why Morals Ignite at L25 to L35</div>
      <h1>The <span class="accent">layer_scalar</span> reset curtain.</h1>
      <div class="sub">Gemma 4 31B damps 99.4% of early lexical residual state before L22, clearing the stream right as L26 to L35 moral writers ignite.</div>
    </div>
    <div class="canvas">{svg}</div>
    <div class="foot">
      <span>Exact model.language_model.layers.0..59.layer_scalar buffers (dip 0.43 at L28) + Phase 0 cross-firewall curve</span>
      <span>L8 survival: 0.11% · L35: 68.0%</span>
    </div>
    """
    return _wrap_minimal("The Layer Scalar Reset Curtain", 1200, 760, "", body), 1200, 760


def build_all_funnel_cards() -> Path:
    wf_dir = ROOT_DIR / "runs" / "weight_funnel_gemma-4-31b-it"
    wf = json.loads((wf_dir / "weight_funnel_results.json").read_text(encoding="utf-8"))
    mat_npz = wf_dir / "l33_top64_matrices.npz"
    p0 = json.loads((ROOT_DIR / "runs" / "gemma-4-31b-it_20261114T152233Z" / "results.json").read_text(encoding="utf-8"))

    out_dir = ROOT_DIR / "runs" / "funnel_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = [
        ("funnel_card1_different_ears_same_megaphone", card_1_different_ears_same_megaphone(wf)),
        ("funnel_card2_two_regimes_across_depth", card_2_two_regimes_across_depth(wf)),
        ("funnel_card3_read_vs_write_asymmetry", card_3_read_vs_write_asymmetry(wf, mat_npz)),
        ("funnel_card4_layer_scalar_reset_curtain", card_4_layer_scalar_reset_curtain(wf, p0)),
    ]

    for slug, (doc_html, w, h) in builders:
        html_p = out_dir / f"{slug}.html"
        png_p = out_dir / f"{slug}.png"
        html_p.write_text(doc_html, encoding="utf-8")
        render_png_via_chrome(html_p, png_p, w, h)
        print(f"Rendered {png_p.name} ({w*2}x{h*2})", flush=True)

    gallery_blocks = []
    for slug, _ in builders:
        title = slug.replace("_", " ").title()
        gallery_blocks.append(
            f'<div style="background:{IVORY};border:1.5px solid rgba(44,62,80,0.22);border-radius:14px;padding:20px;margin-bottom:28px;">'
            f'<div style="font-size:12.5px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:{SLATE_MIST};margin-bottom:10px;">{title}</div>'
            f'<img src="{slug}.png" style="width:100%;height:auto;border:1px solid rgba(44,62,80,0.18);border-radius:8px;display:block;"/>'
            f'</div>'
        )
    idx_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Project Afriat Funnel Cards</title>
<link rel="stylesheet" href="{FONTS_CSS}">
<style>body{{margin:0;padding:42px 24px 80px;background:{PARCHMENT};color:{ONYX};font-family:Inter,sans-serif;}}.wrap{{max-width:1020px;margin:0 auto;}}h1{{font-family:Newsreader,Georgia,serif;font-size:44px;margin:0 0 18px;}}</style>
</head><body><div class="wrap"><h1>Goodfire vs. Afriat Funnel Deck.</h1>{''.join(gallery_blocks)}</div></body></html>
"""
    (out_dir / "index.html").write_text(idx_html, encoding="utf-8")
    return out_dir


if __name__ == "__main__":
    build_all_funnel_cards()
