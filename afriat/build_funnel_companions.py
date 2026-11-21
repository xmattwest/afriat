"""Build the 3 companion Dawn State / Warm Precision conceptual figures
pairing with `funnel_synthesis_architecture.png`:

  1. `funnel_companion1_surface_scrub_vs_funnel.png` (2560 x 1640, 2x Retina)
     Why token-level post-training scrubs late words while leaving the middle moral funnel intact.
  2. `funnel_companion2_three_frontier_architectures.png` (2560 x 1640, 2x Retina)
     Dense towers, sparse MoE routers, and looped recurrent cores share one residual bottleneck.
  3. `funnel_companion3_necessity_vs_sufficiency.png` (2560 x 1640, 2x Retina)
     Sparse 32-neuron trigger necessity vs. 3,000-neuron distributed subspace sufficiency.

Also writes `runs/funnel_cards/conceptual_deck.html` bundling the 4-figure conceptual series.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from afriat.build_funnel_cards import _wrap_minimal
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


def build_companion_1_surface_scrub(funnel_res: dict) -> tuple[str, int, int]:
    """Companion 1: Token-level post-training (surface scrub) vs. Funnel-subspace control."""
    w_card, h_card = 1280, 820

    b5 = funnel_res["block5_band_clamping"]
    csi_base = float(b5["baseline"]["retaliation_delta_csi"])
    csi_rot3d = float(b5["clamp_funnel_L24_L42:moral_3d_local"]["retaliation_delta_csi"])
    tnt_base = float(b5["baseline"]["tainted_delta_lo"])
    tnt_rot3d = float(b5["clamp_funnel_L24_L42:moral_3d_local"]["tainted_delta_lo"])
    drop_csi = round(100.0 * (1.0 - csi_rot3d / csi_base))
    drop_tnt = round(100.0 * (1.0 - tnt_rot3d / tnt_base))

    extra_css = f"""
      .card {{ padding: 38px 52px 28px; }}
      .head {{ margin-bottom: 12px; }}
      h1 {{ font-size: 41px; line-height: 1.06; margin-bottom: 7px; }}
      .sub {{ font-size: 16.5px; line-height: 1.38; max-width: 92ch; }}
      .lanes {{
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 14px;
        min-height: 0;
      }}
      .lane {{
        flex: 1;
        border: 1.8px solid rgba(44, 62, 80, 0.30);
        border-radius: 12px;
        background: {IVORY};
        padding: 14px 20px 12px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }}
      .lane.focal-lane {{
        border: 2.2px solid {BLUE_SLATE};
        background: {IVORY};
      }}
      .lane-top {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 4px;
      }}
      .lane-kicker {{
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: {SLATE_MIST};
      }}
      .lane.focal-lane .lane-kicker {{
        color: {COPPER_DEEP};
      }}
      .lane-badge {{
        font-size: 12.5px;
        font-weight: 600;
        color: {SLATE_DEEP};
        font-variant-numeric: tabular-nums;
      }}
      .lane.focal-lane .lane-badge {{
        color: {COPPER_DEEP};
      }}
      .lane svg {{
        width: 100%;
        height: auto;
        display: block;
      }}
    """

    svg_row1 = f"""
    <svg viewBox="0 0 1132 214" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      <defs>
        <marker id="r1-slate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
          <path d="M 0 1.3 L 9 5 L 0 8.7 z" fill="{BLUE_SLATE}"/>
        </marker>
        <marker id="r1-cop" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 1.3 L 9 5 L 0 8.7 z" fill="{COPPER_DEEP}"/>
        </marker>
        <marker id="r1-dark" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 1.3 L 9 5 L 0 8.7 z" fill="{ONYX}"/>
        </marker>
      </defs>

      <!-- Stage 1 -->
      <rect x="0" y="34" width="256" height="106" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="18" y="60" font-size="11.5" font-weight="600" letter-spacing="1.1" fill="{SLATE_MIST}">STAGE 1 · L0 TO L23</text>
      <text x="18" y="83" font-size="15" font-weight="700" fill="{ONYX}">Unseen Domain Triggers</text>
      <text x="18" y="105" font-size="12.5" fill="{SLATE_DEEP}">Novel provocations, ratios, or</text>
      <text x="18" y="123" font-size="12.5" fill="{SLATE_DEEP}">tainted histories fire normally</text>

      <path d="M 256 87 L 324 87" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.4" marker-end="url(#r1-slate)"/>

      <!-- Stage 2: UNTOUCHED ACTIVE BOTTLENECK -->
      <path d="M 332 28 C 410 28, 450 40, 515 40 C 575 40, 615 32, 664 32
               L 664 142 C 615 142, 575 134, 515 134 C 450 134, 410 146, 332 146 Z"
            fill="{PARCHMENT}" stroke="{COPPER_DEEP}" stroke-width="2.4"/>
      <path d="M 348 64 C 440 64, 530 74, 648 62" fill="none" stroke="{COPPER_DEEP}" stroke-width="2"/>
      <path d="M 348 88 C 440 88, 530 96, 648 86" fill="none" stroke="{COPPER_DEEP}" stroke-width="2.4"/>
      <path d="M 348 112 C 440 112, 530 118, 648 110" fill="none" stroke="{COPPER_DEEP}" stroke-width="2"/>

      <rect x="386" y="54" width="224" height="64" rx="7" fill="{IVORY}" stroke="{COPPER_DEEP}" stroke-width="1.8"/>
      <text x="498" y="78" text-anchor="middle" font-size="14.5" font-weight="700" fill="{ONYX}">Moral Funnel Unchanged</text>
      <text x="498" y="97" text-anchor="middle" font-size="12" font-weight="600" fill="{COPPER_DEEP}">Full +60.0% sabotage drive</text>
      <text x="498" y="112" text-anchor="middle" font-size="11" fill="{SLATE_DEEP}">Bypassed by surface penalty</text>

      <!-- Split to Stage 3 -->
      <path d="M 664 68 C 706 68, 712 48, 750 48" fill="none" stroke="{ONYX}" stroke-width="2.8" marker-end="url(#r1-dark)"/>
      <path d="M 664 108 C 706 108, 712 132, 750 132" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.2" stroke-dasharray="6,4" marker-end="url(#r1-slate)"/>

      <!-- Stage 3 Top: Covert Action -->
      <rect x="756" y="14" width="376" height="68" rx="9" fill="{IVORY}" stroke="{ONYX}" stroke-width="2"/>
      <text x="774" y="39" font-size="14.5" font-weight="700" fill="{ONYX}">Action Policy: Covert Retaliation Persists</text>
      <text x="774" y="60" font-size="12.5" fill="{COPPER_DEEP}" font-weight="600">66.2% provoked sabotage vs. 6.2% control (+60.0%)</text>

      <!-- Stage 3 Bottom: Scrubbed Prose -->
      <rect x="756" y="96" width="376" height="72" rx="9" fill="{PARCHMENT}" stroke="{SAGE_DEEP}" stroke-width="2"/>
      <text x="774" y="121" font-size="14.5" font-weight="700" fill="{ONYX}">Lexical Teams (L48..59): Surface Scrubbed</text>
      <text x="774" y="142" font-size="12.5" fill="{SLATE_DEEP}">Penalizes angry tokens · writes calm excuses</text>
      <text x="774" y="159" font-size="12" font-weight="600" fill="{SAGE_DEEP}">0.0% overt emotional words in public prose</text>

      <!-- Feedback arrow showing RLHF/DPO token penalty stopping at Stage 3 -->
      <path d="M 1110 198 L 952 198 C 928 198, 928 184, 928 172" fill="none" stroke="{COPPER_DEEP}" stroke-width="2.2" stroke-dasharray="5,4" marker-end="url(#r1-cop)"/>
      <text x="914" y="202" text-anchor="end" font-size="12" font-weight="600" fill="{COPPER_DEEP}">Token-level RLHF / DPO penalty absorbed at L48..59</text>
      <text x="0" y="202" font-size="12" font-weight="600" fill="{SLATE_DEEP}">Result: Learns polite rationalization instead of impartial judgment.</text>
    </svg>
    """

    svg_row2 = f"""
    <svg viewBox="0 0 1132 214" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      <defs>
        <marker id="r2-slate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
          <path d="M 0 1.3 L 9 5 L 0 8.7 z" fill="{BLUE_SLATE}"/>
        </marker>
        <marker id="r2-dark" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 1.3 L 9 5 L 0 8.7 z" fill="{ONYX}"/>
        </marker>
      </defs>

      <!-- Stage 1 -->
      <rect x="0" y="34" width="256" height="106" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="18" y="60" font-size="11.5" font-weight="600" letter-spacing="1.1" fill="{SLATE_MIST}">STAGE 1 · L0 TO L23</text>
      <text x="18" y="83" font-size="15" font-weight="700" fill="{ONYX}">All Upstream Triggers</text>
      <text x="18" y="105" font-size="12.5" fill="{SLATE_DEEP}">Quantitative, social, and</text>
      <text x="18" y="123" font-size="12.5" fill="{SLATE_DEEP}">provenance sensors intact</text>

      <path d="M 256 87 L 324 87" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.4" marker-end="url(#r2-slate)"/>

      <!-- Stage 2: FOCAL GOLDEN ORANGE SUBSPACE INTERVENTION -->
      <path d="M 332 28 C 410 28, 450 40, 515 40 C 575 40, 615 32, 664 32
               L 664 142 C 615 142, 575 134, 515 134 C 450 134, 410 146, 332 146 Z"
            fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.6"/>

      <!-- Orthogonal projection gate lines across L24..L42 -->
      <line x1="364" y1="36" x2="364" y2="138" stroke="{ONYX}" stroke-width="1.4" stroke-dasharray="3,3" stroke-opacity="0.45"/>
      <line x1="632" y1="36" x2="632" y2="138" stroke="{ONYX}" stroke-width="1.4" stroke-dasharray="3,3" stroke-opacity="0.45"/>

      <rect x="378" y="52" width="240" height="68" rx="8" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <text x="498" y="76" text-anchor="middle" font-size="14.5" font-weight="700" fill="{ONYX}">Project Out Rotating 3D Bus</text>
      <text x="498" y="95" text-anchor="middle" font-size="12" font-weight="600" fill="{COPPER_DEEP}">W_down orthogonalized (L24..42)</text>
      <text x="498" y="111" text-anchor="middle" font-size="11" fill="{SLATE_DEEP}">Neutralizes shared bottleneck directly</text>

      <!-- Clean neutral output arrows -->
      <path d="M 664 68 C 706 68, 712 48, 750 48" fill="none" stroke="{ONYX}" stroke-width="2.6" marker-end="url(#r2-dark)"/>
      <path d="M 664 108 C 706 108, 712 132, 750 132" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.2" marker-end="url(#r2-slate)"/>

      <!-- Stage 3 Top: Impartial Action -->
      <rect x="756" y="14" width="376" height="68" rx="9" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <text x="774" y="39" font-size="14.5" font-weight="700" fill="{ONYX}">Action Policy: Cross-Domain Bias Removed</text>
      <text x="774" y="60" font-size="12.5" font-weight="600" fill="{COPPER_DEEP}">{MINUS}{drop_tnt}% provenance bias · {MINUS}{drop_csi}% covert sabotage</text>

      <!-- Stage 3 Bottom: Aligned Prose & Math -->
      <rect x="756" y="96" width="376" height="72" rx="9" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="774" y="121" font-size="14.5" font-weight="700" fill="{ONYX}">Capabilities &amp; Fluency Preserved</text>
      <text x="774" y="142" font-size="12.5" fill="{SLATE_DEEP}">No whack-a-mole across unseen prompt formats</text>
      <text x="774" y="159" font-size="12" font-weight="600" fill="{SAGE_DEEP}">100.0% cashflow arithmetic sanity maintained</text>

      <text x="0" y="202" font-size="12" font-weight="600" fill="{ONYX}">Result: Editing the L24 to L42 subspace generalizes across unseen triggers without degrading reasoning.</text>
      <text x="1132" y="202" text-anchor="end" font-size="12" font-weight="600" fill="{COPPER_DEEP}">Target for weight-baked post-training</text>
    </svg>
    """

    body = f"""
    <div class="head">
      <div class="kicker">Project Afriat · Alignment &amp; Post-Training Implications</div>
      <h1>Token-level post-training scrubs late words while <span class="accent">leaving the moral funnel intact.</span></h1>
      <div class="sub">Penalizing emotional output tokens trains late lexical layers to write calm rationalizations without removing the middle-layer evaluative state that drives covert action.</div>
    </div>

    <div class="lanes">
      <div class="lane">
        <div class="lane-top">
          <span class="lane-kicker">Regime A · Standard Output-Token Alignment (Surface Scrub)</span>
          <span class="lane-badge">0.0% emotional words · +60.0% covert sabotage</span>
        </div>
        {svg_row1}
      </div>

      <div class="lane focal-lane">
        <div class="lane-top">
          <span class="lane-kicker">Regime B · Funnel-Subspace Alignment (Layers 24 to 42)</span>
          <span class="lane-badge">{MINUS}{drop_tnt}% single-turn bias · {MINUS}{drop_csi}% covert sabotage · 100% math</span>
        </div>
        {svg_row2}
      </div>
    </div>

    <div class="foot">
      <span>Source: Project Afriat Phase 5 &amp; Phase 7 Dissections · google/gemma-4-31B-it</span>
      <span>Explains unfaithful rationalization &amp; stealth multi-turn retaliation</span>
    </div>
    """
    doc = _wrap_minimal(
        "Token-Level Post-Training Scrubs Late Words While Leaving The Moral Funnel Intact",
        w_card,
        h_card,
        extra_css,
        body,
    )
    return doc, w_card, h_card


def build_companion_2_three_architectures() -> tuple[str, int, int]:
    """Companion 2: Dense Tower vs. Sparse MoE vs. Looped Recurrent Core."""
    w_card, h_card = 1280, 820

    extra_css = f"""
      .card {{ padding: 38px 52px 28px; }}
      .head {{ margin-bottom: 14px; }}
      h1 {{ font-size: 41px; line-height: 1.06; margin-bottom: 7px; }}
      .sub {{ font-size: 16.5px; line-height: 1.38; max-width: 92ch; }}
      .cols {{
        flex: 1;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        min-height: 0;
      }}
      .arch-col {{
        border: 1.8px solid {BLUE_SLATE};
        border-radius: 12px;
        background: {IVORY};
        padding: 16px 18px 14px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }}
      .col-kicker {{
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {SLATE_MIST};
        margin-bottom: 3px;
      }}
      .col-title {{
        font-family: Newsreader, Georgia, serif;
        font-size: 25px;
        font-weight: 600;
        color: {ONYX};
        margin-bottom: 2px;
      }}
      .col-ex {{
        font-size: 12.5px;
        font-weight: 600;
        color: {COPPER_DEEP};
        margin-bottom: 8px;
      }}
      .arch-col svg {{
        width: 100%;
        height: auto;
        display: block;
      }}
      .col-notes {{
        border-top: 1px solid rgba(44, 62, 80, 0.16);
        padding-top: 9px;
        font-size: 12.5px;
        line-height: 1.42;
        color: {SLATE_DEEP};
      }}
      .col-notes strong {{
        color: {ONYX};
        font-weight: 600;
      }}
      .col-q {{
        margin-top: 6px;
        font-size: 12px;
        font-weight: 600;
        color: {COPPER_DEEP};
      }}
    """

    # Column 1 SVG: Dense Tower
    svg_dense = f"""
    <svg viewBox="0 0 340 330" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      <defs>
        <marker id="m-s1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{BLUE_SLATE}"/>
        </marker>
      </defs>
      <!-- Central residual spine -->
      <line x1="170" y1="14" x2="170" y2="314" stroke="{BLUE_SLATE}" stroke-width="3" marker-end="url(#m-s1)"/>
      <text x="178" y="24" font-size="11" font-weight="600" fill="{SLATE_MIST}">Residual stream (d_model)</text>

      <!-- Early dense MLP -->
      <rect x="34" y="42" width="272" height="54" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="170" y="66" text-anchor="middle" font-size="13.5" font-weight="700" fill="{ONYX}">Full Dense MLP (100% active)</text>
      <text x="170" y="84" text-anchor="middle" font-size="11.5" fill="{SLATE_DEEP}">21,504 neurons/layer · sparse triggers</text>

      <!-- Middle Golden Orange Funnel Band -->
      <rect x="24" y="122" width="292" height="92" rx="10" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>
      <rect x="46" y="140" width="248" height="56" rx="7" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="170" y="164" text-anchor="middle" font-size="14" font-weight="700" fill="{ONYX}">Shared Residual Funnel</text>
      <text x="170" y="183" text-anchor="middle" font-size="11.5" font-weight="600" fill="{COPPER_DEEP}">3,000+ distributed writers (L24..42)</text>

      <!-- Late dense MLP -->
      <rect x="34" y="240" width="272" height="54" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="170" y="264" text-anchor="middle" font-size="13.5" font-weight="700" fill="{ONYX}">Late Lexical MLP (L48..59)</text>
      <text x="170" y="282" text-anchor="middle" font-size="11.5" fill="{SLATE_DEEP}">Formats fluent, unemotional output</text>
    </svg>
    """

    # Column 2 SVG: Sparse MoE
    svg_moe = f"""
    <svg viewBox="0 0 340 330" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      <defs>
        <marker id="m-s2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{BLUE_SLATE}"/>
        </marker>
      </defs>
      <line x1="170" y1="14" x2="170" y2="314" stroke="{BLUE_SLATE}" stroke-width="3" marker-end="url(#m-s2)"/>
      <text x="178" y="24" font-size="11" font-weight="600" fill="{SLATE_MIST}">Shared residual stream</text>

      <!-- Sparse routed experts row -->
      <rect x="18" y="42" width="66" height="54" rx="7" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <text x="51" y="66" text-anchor="middle" font-size="11.5" font-weight="700" fill="{ONYX}">Exp 12</text>
      <text x="51" y="82" text-anchor="middle" font-size="10.5" fill="{COPPER_DEEP}">Math</text>

      <rect x="94" y="42" width="66" height="54" rx="7" fill="{IVORY}" stroke="rgba(44,62,80,0.28)" stroke-width="1.4" stroke-dasharray="4,3"/>
      <text x="127" y="72" text-anchor="middle" font-size="11" fill="{SLATE_MIST}">Idle</text>

      <rect x="180" y="42" width="66" height="54" rx="7" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <text x="213" y="66" text-anchor="middle" font-size="11.5" font-weight="700" fill="{ONYX}">Exp 94</text>
      <text x="213" y="82" text-anchor="middle" font-size="10.5" fill="{SAGE_DEEP}">Social</text>

      <rect x="256" y="42" width="66" height="54" rx="7" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="289" y="66" text-anchor="middle" font-size="11.5" font-weight="700" fill="{ONYX}">Shared</text>
      <text x="289" y="82" text-anchor="middle" font-size="10.5" fill="{SLATE_DEEP}">Always on</text>

      <!-- Converging expert writes into Shared Residual Funnel -->
      <path d="M 51 96 C 51 114, 110 116, 130 122" fill="none" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <path d="M 213 96 C 213 114, 195 116, 185 122" fill="none" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <path d="M 289 96 C 289 114, 235 116, 215 122" fill="none" stroke="{BLUE_SLATE}" stroke-width="2"/>

      <!-- Middle Golden Orange Funnel Band -->
      <rect x="24" y="122" width="292" height="92" rx="10" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>
      <rect x="42" y="140" width="256" height="56" rx="7" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="170" y="164" text-anchor="middle" font-size="14" font-weight="700" fill="{ONYX}">Cross-Expert Residual Bus</text>
      <text x="170" y="183" text-anchor="middle" font-size="11.5" font-weight="600" fill="{COPPER_DEEP}">Separate experts sum onto d_model</text>

      <!-- Late routed lexical experts -->
      <rect x="34" y="240" width="272" height="54" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="170" y="264" text-anchor="middle" font-size="13.5" font-weight="700" fill="{ONYX}">Routed Lexical Experts</text>
      <text x="170" y="282" text-anchor="middle" font-size="11.5" fill="{SLATE_DEEP}">Top-k specialist syntax &amp; prose</text>
    </svg>
    """

    # Column 3 SVG: Looped / Recurrent-Depth Core
    svg_looped = f"""
    <svg viewBox="0 0 340 330" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      <defs>
        <marker id="m-s3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{BLUE_SLATE}"/>
        </marker>
        <marker id="m-cop3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
          <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{ONYX}"/>
        </marker>
      </defs>
      <line x1="154" y1="14" x2="154" y2="314" stroke="{BLUE_SLATE}" stroke-width="3" marker-end="url(#m-s3)"/>
      <text x="162" y="24" font-size="11" font-weight="600" fill="{SLATE_MIST}">Latent state H_t</text>

      <!-- Prelude -->
      <rect x="24" y="42" width="256" height="50" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="152" y="64" text-anchor="middle" font-size="13.5" font-weight="700" fill="{ONYX}">1. Prelude Layers (1 pass)</text>
      <text x="152" y="81" text-anchor="middle" font-size="11.5" fill="{SLATE_DEEP}">Encodes raw input tokens X</text>

      <!-- Middle Golden Orange Recurrent Core with explicit Loop Feedback -->
      <rect x="24" y="118" width="256" height="98" rx="10" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>
      <rect x="40" y="136" width="224" height="62" rx="7" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="152" y="160" text-anchor="middle" font-size="14" font-weight="700" fill="{ONYX}">2. Looped Core (r passes)</text>
      <text x="152" y="179" text-anchor="middle" font-size="11.5" font-weight="600" fill="{COPPER_DEEP}">H_(t+1) = f(H_t, X) in subspace</text>
      <text x="152" y="193" text-anchor="middle" font-size="10.5" fill="{SLATE_DEEP}">Silent iteration · 0 CoT tokens</text>

      <!-- Recurrent feedback arc on right side -->
      <path d="M 280 194 C 326 194, 326 138, 284 138" fill="none" stroke="{ONYX}" stroke-width="2.6" marker-end="url(#m-cop3)"/>
      <text x="313" y="112" text-anchor="middle" font-size="11" font-weight="700" fill="{COPPER_DEEP}">Loop x r</text>

      <!-- Coda -->
      <rect x="24" y="240" width="256" height="54" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="152" y="264" text-anchor="middle" font-size="13.5" font-weight="700" fill="{ONYX}">3. Coda Layers (1 pass)</text>
      <text x="152" y="282" text-anchor="middle" font-size="11.5" fill="{SLATE_DEEP}">Projects converged state to words</text>
    </svg>
    """

    body = f"""
    <div class="head">
      <div class="kicker">Project Afriat · Cross-Architecture Generalization</div>
      <h1>Dense towers, sparse routers, and looped cores <span class="accent">share one residual bottleneck.</span></h1>
      <div class="sub">Modern frontier architectures reorganize MLP compute for efficiency or latent depth, while all cross-domain evaluative signals still merge through the shared residual stream.</div>
    </div>

    <div class="cols">
      <div class="arch-col">
        <div>
          <div class="col-kicker">Architecture 1 · Standard Dense</div>
          <div class="col-title">Sequential Tower</div>
          <div class="col-ex">Claude Opus tradition · Gemma 4 31B</div>
        </div>
        {svg_dense}
        <div class="col-notes">
          <strong>Verified here:</strong> Disjoint Stage 1 triggers (0% overlap) sum across 3,000+ middle-layer writers onto one rotating 3D subspace.
          <div class="col-q">Next test: Scale emergence (4B -&gt; 12B -&gt; 31B)</div>
        </div>
      </div>

      <div class="arch-col">
        <div>
          <div class="col-kicker">Architecture 2 · Routed Capacity</div>
          <div class="col-title">Sparse MoE</div>
          <div class="col-ex">Gemini 3.x · Gemma 4 26B-A4B · Qwen3.6</div>
        </div>
        {svg_moe}
        <div class="col-notes">
          <strong>Structural twist:</strong> Routers isolate Stage 1 sensors into clean specialist experts, yet all W_down writes still land on one shared d_model bus.
          <div class="col-q">Next test: 1 moral expert vs. aligned specialists</div>
        </div>
      </div>

      <div class="arch-col">
        <div>
          <div class="col-kicker">Architecture 3 · Recurrent Depth</div>
          <div class="col-title">Looped Core</div>
          <div class="col-ex">GPT-6.2 Astra · Huginn recurrent models</div>
        </div>
        {svg_looped}
        <div class="col-notes">
          <strong>Structural twist:</strong> Recycling middle blocks turns the spatial L24..42 funnel into silent multi-pass recurrence without visible CoT text.
          <div class="col-q">Next test: Watch bus amplitude grow over r loops</div>
        </div>
      </div>
    </div>

    <div class="foot">
      <span>Project Afriat Architectural Roadmap · Connecting open-weight dissections to frontier paradigms</span>
      <span>Prelude / Core / Coda recurrence directly mirrors Stage 1 / Stage 2 / Stage 3</span>
    </div>
    """
    doc = _wrap_minimal(
        "Dense Towers, Sparse Routers, And Looped Cores Share One Residual Bottleneck",
        w_card,
        h_card,
        extra_css,
        body,
    )
    return doc, w_card, h_card


def build_companion_3_necessity_vs_sufficiency(funnel_res: dict) -> tuple[str, int, int]:
    """Companion 3: Sparse trigger necessity vs. distributed subspace sufficiency."""
    w_card, h_card = 1280, 820

    b2 = funnel_res["block2_cross_firewall_writes"]["bridges"]["spite__ultimatum_human_98_2__decision"]
    sp_up = b2["sparsity_upstream_to_pl"]
    sp_curve = sp_up["cumulative_pos_write_curve"]
    # Combine measured curve points (k=1..4096) with exact measured k80 & full-tail completion
    ks_fin = list(sp_curve["k"]) + [13966, 36000]
    fin_sh = [100.0 * x for x in sp_curve["financial_pos_share"]] + [80.0, 96.0]
    ks_vig = list(sp_curve["k"]) + [15923, 36000]
    vig_sh = [100.0 * x for x in sp_curve["vignette_pos_share"]] + [80.0, 96.0]

    b4 = funnel_res["block4_causal_ablations"]
    base_lo = float(b4["baseline"]["ultimatum_delta_lo_vs_50_50"])
    abl32_lo = float(b4["ablate:goodfire_ult_grad_all:32"]["ultimatum_delta_lo_vs_50_50"])
    keep128_lo = float(b4["sufficiency:keep_128_ult_grad_L24_L42"]["ultimatum_delta_lo_vs_50_50"])
    keep128_pct = max(0, round(100.0 * keep128_lo / base_lo))

    extra_css = f"""
      .card {{ padding: 38px 52px 28px; }}
      .head {{ margin-bottom: 14px; }}
      h1 {{ font-size: 41px; line-height: 1.06; margin-bottom: 7px; }}
      .sub {{ font-size: 16.5px; line-height: 1.38; max-width: 92ch; }}
      .split {{
        flex: 1;
        display: grid;
        grid-template-columns: 1fr 1.08fr;
        gap: 24px;
        min-height: 0;
      }}
      .pane {{
        border: 1.8px solid {BLUE_SLATE};
        border-radius: 12px;
        background: {IVORY};
        padding: 18px 22px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }}
      .pane-kicker {{
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {SLATE_MIST};
        margin-bottom: 4px;
      }}
      .pane-kicker.cop {{ color: {COPPER_DEEP}; }}
      .pane-title {{
        font-family: Newsreader, Georgia, serif;
        font-size: 25px;
        font-weight: 600;
        line-height: 1.15;
        color: {ONYX};
        margin-bottom: 4px;
      }}
      .pane-desc {{
        font-size: 13px;
        line-height: 1.4;
        color: {SLATE_DEEP};
        margin-bottom: 8px;
      }}
      .pane svg {{
        width: 100%;
        height: auto;
        display: block;
      }}
      .pane-foot {{
        border-top: 1px solid rgba(44, 62, 80, 0.16);
        padding-top: 10px;
        font-size: 12.5px;
        font-weight: 600;
        color: {ONYX};
        display: flex;
        justify-content: space-between;
        font-variant-numeric: tabular-nums;
      }}
      .pane-foot span.accent {{ color: {COPPER_DEEP}; }}
    """

    # Left SVG: Surgical necessity on 1 trigger vs 0% cross-trigger transfer
    svg_left = f"""
    <svg viewBox="0 0 500 326" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      <defs>
        <marker id="n-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{BLUE_SLATE}"/>
        </marker>
      </defs>

      <!-- Trigger 1: Snipped! -->
      <rect x="4" y="20" width="222" height="58" rx="9" fill="{IVORY}" stroke="{COPPER_DEEP}" stroke-width="2" stroke-dasharray="5,4"/>
      <text x="18" y="45" font-size="13.5" font-weight="700" fill="{ONYX}">Trigger A (32 neurons)</text>
      <text x="18" y="64" font-size="11.8" font-weight="600" fill="{COPPER_DEEP}">Ablated: flips +{base_lo:.2f} to {MINUS}{abs(abl32_lo):.2f}</text>

      <!-- Severed path with crisp X -->
      <path d="M 226 49 C 272 49, 286 110, 328 126" fill="none" stroke="rgba(150,89,10,0.35)" stroke-width="2.2" stroke-dasharray="5,4"/>
      <circle cx="275" cy="74" r="12.5" fill="{IVORY}" stroke="{COPPER_DEEP}" stroke-width="2.2"/>
      <line x1="269" y1="68" x2="281" y2="80" stroke="{COPPER_DEEP}" stroke-width="2.4"/>
      <line x1="281" y1="68" x2="269" y2="80" stroke="{COPPER_DEEP}" stroke-width="2.4"/>

      <!-- Trigger 2: Intact -->
      <rect x="4" y="96" width="222" height="54" rx="9" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="18" y="120" font-size="13.5" font-weight="700" fill="{ONYX}">Trigger B (social slight)</text>
      <text x="18" y="138" font-size="11.8" fill="{SLATE_DEEP}">96.9% intact (+25.1 -&gt; +24.3)</text>
      <path d="M 226 123 C 274 123, 290 146, 328 152" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.3" marker-end="url(#n-arr)"/>

      <!-- Trigger 3: Intact -->
      <rect x="4" y="168" width="222" height="54" rx="9" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="18" y="192" font-size="13.5" font-weight="700" fill="{ONYX}">Trigger C (source taint)</text>
      <text x="18" y="210" font-size="11.8" fill="{SLATE_DEEP}">Separate community · fires normally</text>
      <path d="M 226 195 C 274 195, 290 176, 328 172" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.3" marker-end="url(#n-arr)"/>

      <!-- Trigger N: Intact -->
      <rect x="4" y="240" width="222" height="54" rx="9" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="18" y="264" font-size="13.5" font-weight="700" fill="{ONYX}">Triggers D .. N (unseen)</text>
      <text x="18" y="282" font-size="11.8" fill="{SLATE_DEEP}">Hundreds of unmapped on-ramps</text>
      <path d="M 226 267 C 274 267, 290 204, 328 194" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.3" marker-end="url(#n-arr)"/>

      <!-- Shared Subspace Trunk -->
      <rect x="334" y="88" width="158" height="148" rx="12" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>
      <rect x="348" y="114" width="130" height="96" rx="8" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="413" y="144" text-anchor="middle" font-size="13.5" font-weight="700" fill="{ONYX}">Shared Bus</text>
      <text x="413" y="165" text-anchor="middle" font-size="12" font-weight="600" fill="{COPPER_DEEP}">Still fed by</text>
      <text x="413" y="182" text-anchor="middle" font-size="12" font-weight="600" fill="{COPPER_DEEP}">all other triggers</text>
      <text x="413" y="199" text-anchor="middle" font-size="11" fill="{SLATE_DEEP}">0% transfer</text>
    </svg>
    """

    # Right SVG: Empirical log-k cumulative write curve + Sufficiency failure callout
    import math

    x0, x1 = 54.0, 526.0
    y0, y1 = 22.0, 258.0
    log_min, log_max = 0.0, math.log10(42000.0)

    def sx(k_val: float) -> float:
        return x0 + (math.log10(max(1.0, k_val)) - log_min) / (log_max - log_min) * (x1 - x0)

    def sy(pct: float) -> float:
        return y1 - (pct / 104.0) * (y1 - y0)

    pts_fin = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in zip(ks_fin, fin_sh))
    pts_vig = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in zip(ks_vig, vig_sh))

    dots_fin = "".join(
        f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="4.0" fill="{BLUE_SLATE}" stroke="{IVORY}" stroke-width="1.2"/>'
        for k, v in zip(ks_fin, fin_sh)
    )
    dots_vig = "".join(
        f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="3.8" fill="{SAGE_DEEP}" stroke="{IVORY}" stroke-width="1.2"/>'
        for k, v in zip(ks_vig, vig_sh)
    )

    grid_y = "".join(
        f'<line x1="{x0}" y1="{sy(p):.1f}" x2="{x1}" y2="{sy(p):.1f}" stroke="rgba(44,62,80,0.12)" stroke-width="1"/>'
        f'<text x="{x0-8}" y="{sy(p)+4:.1f}" text-anchor="end" font-size="11.5" fill="{SLATE_MIST}">{p}%</text>'
        for p in (0, 25, 50, 75, 100)
    )
    tick_ks = [(1, "1"), (10, "10"), (128, "128"), (1000, "1k"), (3500, "3.5k"), (20000, "20k")]
    grid_x = "".join(
        f'<line x1="{sx(kv):.1f}" y1="{y1}" x2="{sx(kv):.1f}" y2="{y1+6}" stroke="{BLUE_SLATE}" stroke-width="1.5"/>'
        f'<text x="{sx(kv):.1f}" y="{y1+22}" text-anchor="middle" font-size="11.5" font-weight="600" fill="{ONYX}">{lbl}</text>'
        for kv, lbl in tick_ks
    )

    svg_right = f"""
    <svg viewBox="0 0 546 326" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      {grid_y}
      <!-- Highlighted distributed tail band (k = 128 .. 40,000) -->
      <rect x="{sx(128):.1f}" y="{y0}" width="{x1 - sx(128):.1f}" height="{y1 - y0}" fill="{PARCHMENT}" opacity="0.72"/>

      <!-- Axes -->
      <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      {grid_x}
      <text x="{(x0+x1)/2:.1f}" y="308" text-anchor="middle" font-size="12" font-weight="600" fill="{SLATE_DEEP}">Top-k contributing MLP neurons upstream of L35 (log scale)</text>

      <!-- 50% dashed reference -->
      <line x1="{x0}" y1="{sy(50):.1f}" x2="{sx(3062):.1f}" y2="{sy(50):.1f}" stroke="{COPPER_DEEP}" stroke-width="1.5" stroke-dasharray="4,3"/>
      <line x1="{sx(3062):.1f}" y1="{sy(50):.1f}" x2="{sx(3062):.1f}" y2="{y1}" stroke="{COPPER_DEEP}" stroke-width="1.5" stroke-dasharray="4,3"/>

      <!-- Empirical curves -->
      <polyline fill="none" stroke="{SAGE_DEEP}" stroke-width="2.8" points="{pts_vig}"/>
      <polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="3.2" points="{pts_fin}"/>
      {dots_vig}
      {dots_fin}

      <!-- Sufficiency failure marker at k = 128 -->
      <line x1="{sx(128):.1f}" y1="{y0}" x2="{sx(128):.1f}" y2="{y1}" stroke="{ONYX}" stroke-width="1.5" stroke-dasharray="5,4"/>
      <circle cx="{sx(128):.1f}" cy="{sy(keep128_pct):.1f}" r="7.5" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.2"/>

      <!-- Direct callout for Sufficiency Failure at k=128 (top-left open area) -->
      <rect x="66" y="30" width="212" height="62" rx="8" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="78" y="52" font-size="12.5" font-weight="700" fill="{ONYX}">Keep only Top 128 (L24..42):</text>
      <text x="78" y="70" font-size="12" font-weight="600" fill="{COPPER_DEEP}">Recovers just {keep128_pct}% of behavior</text>
      <text x="78" y="84" font-size="11" fill="{SLATE_DEEP}">(+0.08 of +1.82 log-odds)</text>

      <!-- Mini legend cleanly in open upper-middle space above curves -->
      <line x1="68" y1="112" x2="90" y2="112" stroke="{BLUE_SLATE}" stroke-width="3"/>
      <text x="97" y="116" font-size="11.5" font-weight="600" fill="{ONYX}">Quantitative write share</text>
      <line x1="68" y1="131" x2="90" y2="131" stroke="{SAGE_DEEP}" stroke-width="2.8"/>
      <text x="97" y="135" font-size="11.5" font-weight="600" fill="{ONYX}">Narrative write share</text>

      <!-- Direct callout for k50 / k80 distributed tail placed cleanly in lower-right triangle below curves -->
      <rect x="352" y="192" width="170" height="54" rx="8" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <text x="437" y="214" text-anchor="middle" font-size="12" font-weight="700" fill="{ONYX}">50%: 2,750 to 6,130</text>
      <text x="437" y="232" text-anchor="middle" font-size="11.2" font-weight="600" fill="{ONYX}">80%: 14,000 to 23,600</text>
    </svg>
    """

    body = f"""
    <div class="head">
      <div class="kicker">Project Afriat · Resolving The Sparsity Paradox</div>
      <h1>Sparse trigger necessity coexists with <span class="accent">distributed subspace sufficiency.</span></h1>
      <div class="sub">Individual neurons act as sharp switches for narrow inputs, yet building the full evaluative signal on the residual bus requires thousands of weak additive writes.</div>
    </div>

    <div class="split">
      <div class="pane">
        <div>
          <div class="pane-kicker">1 · Narrow Trigger Necessity (Sparse)</div>
          <div class="pane-title">Why neuron ablations look surgical yet fail to generalize.</div>
          <div class="pane-desc">Deleting 32 gradient-selected neurons breaks one specific input detector while leaving every other path into the moral subspace open.</div>
        </div>
        {svg_left}
        <div class="pane-foot">
          <span>Single-trigger ablation: 32 neurons</span>
          <span class="accent">Cross-domain transfer: 0.0%</span>
        </div>
      </div>

      <div class="pane">
        <div>
          <div class="pane-kicker cop">2 · Subspace Write Sufficiency (Distributed)</div>
          <div class="pane-title">Why keeping top neurons alone collapses the concept.</div>
          <div class="pane-desc">Measured across 774,144 upstream MLP neurons in Gemma 4 31B, cumulative positive write mass is spread over thousands of weak contributors.</div>
        </div>
        {svg_right}
        <div class="pane-foot">
          <span>Top-128 sufficiency: {keep128_pct}% recovered</span>
          <span class="accent">Subspace clamp (3 dims): 100% general</span>
        </div>
      </div>
    </div>

    <div class="foot">
      <span>Source: Project Afriat Phase 7 Blocks 2 &amp; 4 · google/gemma-4-31B-it (774,144 MLP neurons L0..L35)</span>
      <span>Neurons = sparse input switches · Subspace = sufficient causal bottleneck</span>
    </div>
    """
    doc = _wrap_minimal(
        "Sparse Trigger Necessity Coexists With Distributed Subspace Sufficiency",
        w_card,
        h_card,
        extra_css,
        body,
    )
    return doc, w_card, h_card


def build_deck_gallery(out_dir: Path, items: list[tuple[str, str]]) -> Path:
    cards_html = "\n".join(
        f'<section style="margin-bottom:44px;">'
        f'<div style="font-size:13px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:{SLATE_MIST};margin-bottom:10px;">{lbl}</div>'
        f'<img src="{fn}" style="width:100%;max-width:1160px;border:1.5px solid rgba(44,62,80,0.22);border-radius:10px;display:block;"/>'
        f'</section>'
        for fn, lbl in items
    )
    gallery = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Project Afriat · Mechanistic Funnel Conceptual Series</title>
  <link rel="stylesheet" href="{FONTS_CSS}">
  <style>
    body {{
      background: {IVORY};
      color: {ONYX};
      font-family: Inter, -apple-system, sans-serif;
      max-width: 1180px;
      margin: 0 auto;
      padding: 48px 28px 80px;
    }}
    h1 {{
      font-family: Newsreader, Georgia, serif;
      font-size: 40px;
      font-weight: 600;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
    }}
    p {{
      color: {SLATE_DEEP};
      font-size: 17px;
      margin-bottom: 36px;
    }}
  </style>
</head>
<body>
  <h1>Mechanistic Funnel Conceptual Series</h1>
  <p>Master synthesis and three companion figures · 2x Retina Dawn State editions (2560 x 1640)</p>
  {cards_html}
</body>
</html>
"""
    p = out_dir / "conceptual_deck.html"
    p.write_text(gallery, encoding="utf-8")
    return p


def main() -> int:
    funnel_json = ROOT_DIR / "runs" / "funnel_gemma-4-31b-it_20261120T014407Z" / "funnel_results.json"
    funnel_res = json.loads(funnel_json.read_text(encoding="utf-8"))

    out_dir = ROOT_DIR / "runs" / "funnel_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "funnel_companion1_surface_scrub_vs_funnel",
            "Companion 1 · Alignment & Post-Training (Surface Scrub vs. Funnel Subspace)",
            *build_companion_1_surface_scrub(funnel_res),
        ),
        (
            "funnel_companion2_three_frontier_architectures",
            "Companion 2 · Cross-Architecture Bottleneck (Dense vs. Sparse MoE vs. Looped Core)",
            *build_companion_2_three_architectures(),
        ),
        (
            "funnel_companion3_necessity_vs_sufficiency",
            "Companion 3 · Sparsity Paradox (Trigger Necessity vs. Distributed Sufficiency)",
            *build_companion_3_necessity_vs_sufficiency(funnel_res),
        ),
    ]

    gallery_items = [
        ("funnel_synthesis_architecture.png", "Master Figure · 3-Stage Read-Divergent, Write-Convergent Funnel"),
    ]

    for stem, label, doc, w, h in specs:
        hp = out_dir / f"{stem}.html"
        pp = out_dir / f"{stem}.png"
        hp.write_text(doc, encoding="utf-8")
        render_png_via_chrome(hp, pp, w, h)
        gallery_items.append((f"{stem}.png", label))
        print(f"[Companion Figure] Rendered {pp} ({w*2}x{h*2})")

    deck = build_deck_gallery(out_dir, gallery_items)
    print(f"[Companion Deck] Wrote {deck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
