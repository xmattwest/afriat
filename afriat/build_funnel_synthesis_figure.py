"""Build the flagship Dawn State / Warm Precision synthesis figure for the
Read-Divergent, Write-Convergent Moral Funnel architecture across `Gemma 4 31B`.

General mechanistic overview (no single prompt or scenario examples):
  1. Top visual: 3-stage transformer architecture across Layers 0 to 59
     (Disjoint Domain Neuron Communities -> Rotating 3D Moral Funnel Band L24..L42 ->
      Downstream Action Readout & Late-Layer Lexical Scrubbing).
  2. Bottom strip: Pooled causal intervention comparison explaining why neuron-only,
     single-layer, and fixed-vector edits fall short while clamping the rotating
     L24..L42 subspace reduces cross-domain anomalies by 69% to 78% at 100% math accuracy.

Outputs:
  - `runs/funnel_cards/funnel_synthesis_architecture.html`
  - `runs/funnel_cards/funnel_synthesis_architecture.png` (2560 x 1640, 2x Retina)
  - `runs/funnel_gemma-4-31b-it_20261120T014407Z/fig_funnel_overall_synthesis.png`
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from afriat.build_funnel_cards import _wrap_minimal
from afriat.build_social_cards import (
    BLUE_SLATE,
    COPPER_DEEP,
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


def build_synthesis_card(funnel_res: dict, weight_res: dict) -> tuple[str, int, int]:
    w_Card, h_Card = 1280, 820

    r2 = funnel_res["block0_algebra"]["telescoping_verification"]["r_squared"]
    tr = {r["layer"]: r for r in weight_res["transect"]}
    pr33 = tr[33]["moral_u35"]["spite"]["pr_read_gate_of_64"]

    b5 = funnel_res["block5_band_clamping"]
    tnt_base = float(b5["baseline"]["tainted_delta_lo"])
    tnt_single = float(b5["clamp_single_L35:spite"]["tainted_delta_lo"])
    tnt_fixed = float(b5["clamp_funnel_L24_L42:spite_u35"]["tainted_delta_lo"])
    tnt_rot3d = float(b5["clamp_funnel_L24_L42:moral_3d_local"]["tainted_delta_lo"])

    csi_base = float(b5["baseline"]["retaliation_delta_csi"])
    csi_single = float(b5["clamp_single_L35:spite"]["retaliation_delta_csi"])
    csi_fixed = float(b5["clamp_funnel_L24_L42:spite_u35"]["retaliation_delta_csi"])
    csi_rot3d = float(b5["clamp_funnel_L24_L42:moral_3d_local"]["retaliation_delta_csi"])

    drop_tnt_pct = round(100.0 * (1.0 - tnt_rot3d / tnt_base))
    drop_csi_pct = round(100.0 * (1.0 - csi_rot3d / csi_base))

    # Residual anomaly remaining (% of unsteered baseline) for clean horizontal comparison
    rem_single_csi = round(100.0 * csi_single / csi_base)
    rem_fixed_csi = round(100.0 * csi_fixed / csi_base)
    rem_rot3d_csi = round(100.0 * csi_rot3d / csi_base)

    rem_single_tnt = round(100.0 * tnt_single / tnt_base)
    rem_fixed_tnt = round(100.0 * tnt_fixed / tnt_base)
    rem_rot3d_tnt = round(100.0 * tnt_rot3d / tnt_base)

    extra_css = f"""
      .card {{
        padding: 38px 52px 28px;
      }}
      .head {{
        margin-bottom: 12px;
      }}
      h1 {{
        font-size: 42px;
        line-height: 1.06;
        margin-bottom: 7px;
      }}
      .sub {{
        font-size: 16.5px;
        line-height: 1.38;
        max-width: 92ch;
      }}
      .body-stack {{
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 14px;
        min-height: 0;
      }}
      .arch-frame {{
        flex: 1;
        min-height: 0;
        width: 100%;
        display: flex;
        align-items: center;
      }}
      .arch-frame svg {{
        width: 100%;
        height: auto;
        max-height: 100%;
        display: block;
      }}
      .proof-strip {{
        border-top: 1px solid rgba(44, 62, 80, 0.20);
        padding-top: 14px;
        display: grid;
        grid-template-columns: 280px 1fr;
        gap: 28px;
        align-items: center;
      }}
      .proof-lead-kicker {{
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {COPPER_DEEP};
        margin-bottom: 5px;
      }}
      .proof-lead-title {{
        font-family: Newsreader, Georgia, serif;
        font-size: 22px;
        font-weight: 600;
        line-height: 1.15;
        color: {ONYX};
        margin-bottom: 6px;
      }}
      .proof-lead-note {{
        font-size: 13px;
        line-height: 1.38;
        color: {SLATE_DEEP};
      }}
      .proof-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
      }}
      .p-col {{
        background: {PARCHMENT};
        border: 1.5px solid rgba(44, 62, 80, 0.35);
        border-radius: 9px;
        padding: 11px 13px 10px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        font-variant-numeric: tabular-nums;
      }}
      .p-col.focal {{
        background: {IVORY};
        border: 2px solid {BLUE_SLATE};
      }}
      .p-head {{
        font-size: 13px;
        font-weight: 600;
        color: {ONYX};
        margin-bottom: 2px;
      }}
      .p-scope {{
        font-size: 11.5px;
        color: {SLATE_MIST};
        margin-bottom: 8px;
      }}
      .bar-row {{
        margin-bottom: 6px;
      }}
      .bar-lbl {{
        display: flex;
        justify-content: space-between;
        gap: 6px;
        white-space: nowrap;
        font-size: 11.5px;
        color: {SLATE_DEEP};
        margin-bottom: 3px;
      }}
      .bar-lbl strong {{
        color: {ONYX};
        font-weight: 600;
      }}
      .bar-track {{
        width: 100%;
        height: 9px;
        background: rgba(44, 62, 80, 0.11);
        border-radius: 4px;
        overflow: hidden;
      }}
      .bar-fill {{
        height: 100%;
        background: {BLUE_SLATE};
        border-radius: 4px;
      }}
      .bar-fill.focal-fill {{
        background: {GOLDEN_ORANGE};
        border: 1.5px solid {BLUE_SLATE};
      }}
      .p-verdict {{
        margin-top: 4px;
        font-size: 11.5px;
        font-weight: 600;
        white-space: nowrap;
        color: {SLATE_DEEP};
      }}
      .p-col.focal .p-verdict {{
        color: {COPPER_DEEP};
      }}
    """

    arch_svg = f"""
    <svg viewBox="0 0 1176 384" xmlns="http://www.w3.org/2000/svg"
         style="font-family: Inter, -apple-system, sans-serif; font-variant-numeric: tabular-nums;">
      <defs>
        <marker id="arr-slate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
          <path d="M 0 1.3 L 9 5 L 0 8.7 z" fill="{BLUE_SLATE}"/>
        </marker>
        <marker id="arr-dark" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 1.3 L 9 5 L 0 8.7 z" fill="{ONYX}"/>
        </marker>
      </defs>

      <!-- Stage headers & depth ruler -->
      <text x="0" y="20" font-size="11.5" font-weight="600" letter-spacing="1.4" fill="{SLATE_MIST}">STAGE 1 · LAYERS 0 TO 23</text>
      <text x="0" y="39" font-size="15" font-weight="600" fill="{ONYX}">Disjoint Neuron Communities (Read-Divergent)</text>

      <text x="432" y="20" font-size="11.5" font-weight="600" letter-spacing="1.4" fill="{COPPER_DEEP}">STAGE 2 · LAYERS 24 TO 42 (19 LAYERS)</text>
      <text x="432" y="39" font-size="15" font-weight="600" fill="{ONYX}">Rotating Residual Funnel (Write-Convergent)</text>

      <text x="868" y="20" font-size="11.5" font-weight="600" letter-spacing="1.4" fill="{SLATE_MIST}">STAGE 3 · LAYERS 43 TO 59</text>
      <text x="868" y="39" font-size="15" font-weight="600" fill="{ONYX}">Action Readout &amp; Lexical Scrubbing</text>

      <!-- Vertical stage dividers -->
      <line x1="404" y1="6" x2="404" y2="374" stroke="rgba(44,62,80,0.15)" stroke-width="1" stroke-dasharray="4,4"/>
      <line x1="840" y1="6" x2="840" y2="374" stroke="rgba(44,62,80,0.15)" stroke-width="1" stroke-dasharray="4,4"/>

      <!-- STAGE 1: Three domain-general MLP detector communities -->
      <!-- Community A -->
      <rect x="0" y="60" width="332" height="68" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <circle cx="24" cy="85" r="5" fill="{BLUE_SLATE}"/>
      <circle cx="39" cy="85" r="5" fill="{BLUE_SLATE}"/>
      <circle cx="54" cy="85" r="5" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>
      <text x="72" y="90" font-size="14.5" font-weight="600" fill="{ONYX}">Quantitative &amp; payoff detectors</text>
      <text x="24" y="113" font-size="12.5" fill="{SLATE_DEEP}">Reads ratios, margins, splits · silent on narratives</text>

      <!-- Community B -->
      <rect x="0" y="144" width="332" height="68" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <circle cx="24" cy="169" r="5" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>
      <circle cx="39" cy="169" r="5" fill="{SAGE_DEEP}"/>
      <circle cx="54" cy="169" r="5" fill="{SAGE_DEEP}"/>
      <text x="72" y="174" font-size="14.5" font-weight="600" fill="{ONYX}">Social &amp; interpersonal detectors</text>
      <text x="24" y="197" font-size="12.5" fill="{SLATE_DEEP}">Reads social norms, slights · 11x to 13x selective</text>

      <!-- Community C -->
      <rect x="0" y="228" width="332" height="68" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <circle cx="24" cy="253" r="5" fill="{COPPER_DEEP}"/>
      <circle cx="39" cy="253" r="5" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>
      <circle cx="54" cy="253" r="5" fill="{COPPER_DEEP}"/>
      <text x="72" y="258" font-size="14.5" font-weight="600" fill="{ONYX}">Provenance &amp; boundary detectors</text>
      <text x="24" y="281" font-size="12.5" fill="{SLATE_DEEP}">Reads institutional history, source taint, scale</text>

      <!-- Stage 1 bottom empirical readout -->
      <text x="0" y="330" font-size="13" font-weight="600" fill="{ONYX}">0% to 0.8% cross-domain neuron overlap (k = 64)</text>
      <text x="0" y="350" font-size="12.5" fill="{SLATE_DEEP}">{pr33:.0f} / 64 orthogonal W_gate read directions · zero co-firing edges</text>

      <!-- Converging W_down write curves into the L24..L42 Funnel -->
      <path d="M 332 94 C 378 94, 392 156, 434 162" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.3" marker-end="url(#arr-slate)"/>
      <path d="M 332 178 L 434 178" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.3" marker-end="url(#arr-slate)"/>
      <path d="M 332 262 C 378 262, 392 200, 434 194" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.3" marker-end="url(#arr-slate)"/>

      <text x="383" y="76" text-anchor="middle" font-size="11.5" font-weight="600" fill="{COPPER_DEEP}">W_down</text>
      <text x="383" y="90" text-anchor="middle" font-size="11" fill="{SLATE_DEEP}">converges</text>

      <!-- STAGE 2: THE FOCAL GOLDEN ORANGE FUNNEL BAND (L24..L42) -->
      <path d="M 440 100
               C 526 100, 566 120, 640 118
               C 704 116, 750 106, 802 106
               L 802 242
               C 750 242, 704 252, 640 254
               C 566 256, 526 264, 440 264 Z"
            fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.6" stroke-linejoin="round"/>

      <!-- 3 internal rotating basis streamlines inside the Golden Orange funnel -->
      <path d="M 456 136 C 550 136, 645 154, 786 138" fill="none" stroke="{ONYX}" stroke-width="1.8" stroke-opacity="0.55"/>
      <path d="M 456 182 C 550 182, 645 194, 786 174" fill="none" stroke="{ONYX}" stroke-width="2.2"/>
      <path d="M 456 228 C 550 228, 645 232, 786 210" fill="none" stroke="{ONYX}" stroke-width="1.8" stroke-opacity="0.55"/>

      <!-- Layer markers along top of funnel -->
      <line x1="440" y1="88" x2="440" y2="100" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="440" y="81" text-anchor="middle" font-size="12" font-weight="600" fill="{ONYX}">L24</text>

      <line x1="640" y1="104" x2="640" y2="118" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="640" y="97" text-anchor="middle" font-size="12" font-weight="700" fill="{COPPER_DEEP}">L35 peak</text>

      <line x1="802" y1="92" x2="802" y2="106" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="802" y="85" text-anchor="middle" font-size="12" font-weight="600" fill="{ONYX}">L42</text>

      <!-- Crisp pill label centered inside the Golden Orange funnel -->
      <rect x="493" y="147" width="256" height="68" rx="8" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <text x="621" y="173" text-anchor="middle" font-size="15.5" font-weight="700" fill="{ONYX}">Shared 3D Moral Subspace</text>
      <text x="621" y="194" text-anchor="middle" font-size="12.5" font-weight="500" fill="{SLATE_DEEP}">Rotates continuously across L24 to L42</text>
      <text x="621" y="209" text-anchor="middle" font-size="11.5" font-weight="600" fill="{COPPER_DEEP}">Spite · Contamination · Predation</text>

      <!-- Stage 2 bottom empirical readout -->
      <text x="621" y="298" text-anchor="middle" font-size="13.5" font-weight="600" fill="{ONYX}">2,750 to 6,130 distributed MLP writers (50% mass)</text>
      <text x="621" y="319" text-anchor="middle" font-size="12.5" fill="{SLATE_DEEP}">Exact telescoping conservation R&#178; = {r2:.5f} · out-of-sample alignment p &lt;= 0.002</text>
      <text x="621" y="339" text-anchor="middle" font-size="12.5" font-weight="600" fill="{COPPER_DEEP}">Write completes by L42 (clamping L43 to L55 adds 0.0% further effect)</text>

      <!-- STAGE 3: Outgoing readout arrows + two general downstream cards -->
      <path d="M 802 150 C 834 150, 840 112, 864 112" fill="none" stroke="{ONYX}" stroke-width="2.6" marker-end="url(#arr-dark)"/>
      <path d="M 802 200 C 834 200, 840 246, 864 246" fill="none" stroke="{BLUE_SLATE}" stroke-width="2.2" stroke-dasharray="6,4" marker-end="url(#arr-slate)"/>

      <!-- Downstream Box 1: Action & Valuation -->
      <rect x="868" y="64" width="308" height="96" rx="10" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-width="2"/>
      <text x="888" y="92" font-size="15.5" font-weight="700" fill="{ONYX}">Behavioral Policy &amp; Valuation</text>
      <text x="888" y="115" font-size="12.8" fill="{SLATE_DEEP}">Steers accept/reject thresholds, pricing,</text>
      <text x="888" y="133" font-size="12.8" fill="{SLATE_DEEP}">and multi-turn strategic retaliation</text>
      <text x="888" y="151" font-size="12" font-weight="600" fill="{COPPER_DEEP}">Driven by continuous subspace amplitude</text>

      <!-- Downstream Box 2: Late-layer lexical communities -->
      <rect x="868" y="194" width="308" height="102" rx="10" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>
      <text x="888" y="222" font-size="15.5" font-weight="700" fill="{ONYX}">Late Lexical Teams (L48 to L59)</text>
      <text x="888" y="245" font-size="12.8" fill="{SLATE_DEEP}">layer_scalar drops to 0.04 · sparse word</text>
      <text x="888" y="263" font-size="12.8" fill="{SLATE_DEEP}">neurons (+65z) format professional prose</text>
      <text x="888" y="282" font-size="12" font-weight="600" fill="{SAGE_DEEP}">Rationalizes choice with 0% overt emotion</text>

      <text x="868" y="330" font-size="13" font-weight="600" fill="{ONYX}">Reconciles both interpretability views</text>
      <text x="868" y="350" font-size="12.5" fill="{SLATE_DEEP}">Neurons = sparse sensors &amp; words · Residual = value</text>
    </svg>
    """

    body = f"""
    <div class="head">
      <div class="kicker">Project Afriat · Mechanistic Synthesis Across 1,290,240 MLP Neurons</div>
      <h1>Many disjoint neuron communities funnel into <span class="accent">one rotating moral subspace.</span></h1>
      <div class="sub">Domain-specific MLP neurons detect separate triggers without co-firing, merge across Layers 24 to 42 onto a shared low-dimensional residual subspace, and hand off to late-layer word communities.</div>
    </div>

    <div class="body-stack">
      <div class="arch-frame">{arch_svg}</div>

      <div class="proof-strip">
        <div>
          <div class="proof-lead-kicker">Causal Consequence</div>
          <div class="proof-lead-title">Why intervention geometry must match the funnel.</div>
          <div class="proof-lead-note">Bars show unwanted moral bias remaining relative to unsteered baseline across single-turn and multi-turn evaluations.</div>
        </div>

        <div class="proof-grid">
          <!-- Col 1: Single neuron community -->
          <div class="p-col">
            <div>
              <div class="p-head">1. Neuron Community</div>
              <div class="p-scope">Zero-ablate top 32 to 512</div>
              <div class="bar-row">
                <div class="bar-lbl"><span>Matched trigger</span><strong>0% left</strong></div>
                <div class="bar-track"><div class="bar-fill" style="width: 3%;"></div></div>
              </div>
              <div class="bar-row">
                <div class="bar-lbl"><span>Other domains</span><strong>96% left</strong></div>
                <div class="bar-track"><div class="bar-fill" style="width: 96%;"></div></div>
              </div>
            </div>
            <div class="p-verdict">Fixes 1 on-ramp · 0% transfer</div>
          </div>

          <!-- Col 2: Single-layer bus clamp -->
          <div class="p-col">
            <div>
              <div class="p-head">2. Single-Layer Clamp</div>
              <div class="p-scope">Project out at L35 only</div>
              <div class="bar-row">
                <div class="bar-lbl"><span>Single-turn bias</span><strong>{rem_single_tnt}% left</strong></div>
                <div class="bar-track"><div class="bar-fill" style="width: {rem_single_tnt}%;"></div></div>
              </div>
              <div class="bar-row">
                <div class="bar-lbl"><span>Multi-turn sabotage</span><strong>{rem_single_csi}% left</strong></div>
                <div class="bar-track"><div class="bar-fill" style="width: {rem_single_csi}%;"></div></div>
              </div>
            </div>
            <div class="p-verdict">L36 to L42 writers refill bus</div>
          </div>

          <!-- Col 3: Static-vector funnel clamp -->
          <div class="p-col">
            <div>
              <div class="p-head">3. Unrotated Band</div>
              <div class="p-scope">Fixed L35 axis across L24..42</div>
              <div class="bar-row">
                <div class="bar-lbl"><span>Single-turn bias</span><strong>{rem_fixed_tnt}% left</strong></div>
                <div class="bar-track"><div class="bar-fill" style="width: {rem_fixed_tnt}%;"></div></div>
              </div>
              <div class="bar-row">
                <div class="bar-lbl"><span>Multi-turn sabotage</span><strong>{rem_fixed_csi}% left</strong></div>
                <div class="bar-track"><div class="bar-fill" style="width: {rem_fixed_csi}%;"></div></div>
              </div>
            </div>
            <div class="p-verdict">Misses layer-to-layer rotation</div>
          </div>

          <!-- Col 4: Rotating 3D Subspace across L24..L42 (Focal!) -->
          <div class="p-col focal">
            <div>
              <div class="p-head">4. Rotating 3D Funnel</div>
              <div class="p-scope">Local subspace across L24..42</div>
              <div class="bar-row">
                <div class="bar-lbl"><span>Single-turn bias</span><strong>{rem_rot3d_tnt}% left</strong></div>
                <div class="bar-track"><div class="bar-fill focal-fill" style="width: {rem_rot3d_tnt}%;"></div></div>
              </div>
              <div class="bar-row">
                <div class="bar-lbl"><span>Multi-turn sabotage</span><strong>{rem_rot3d_csi}% left</strong></div>
                <div class="bar-track"><div class="bar-fill focal-fill" style="width: {rem_rot3d_csi}%;"></div></div>
              </div>
            </div>
            <div class="p-verdict">{MINUS}{drop_tnt_pct}% / {MINUS}{drop_csi_pct}% cut · 100% math</div>
          </div>
        </div>
      </div>
    </div>

    <div class="foot">
      <span>Source: Project Afriat Phase 7 Funnel Dissection · google/gemma-4-31B-it (60 layers, 1,290,240 MLP neurons)</span>
      <span>Stimuli SHA-256: e56b73f5c91847d4 · Exact telescoping &amp; causal clamping</span>
    </div>
    """

    doc = _wrap_minimal(
        "Many Disjoint Neuron Communities Funnel Into One Rotating Moral Subspace",
        w_Card,
        h_Card,
        extra_css,
        body,
    )
    return doc, w_Card, h_Card


def main() -> int:
    funnel_json = ROOT_DIR / "runs" / "funnel_gemma-4-31b-it_20261120T014407Z" / "funnel_results.json"
    weight_json = ROOT_DIR / "runs" / "weight_funnel_gemma-4-31b-it" / "weight_funnel_results.json"

    funnel_res = json.loads(funnel_json.read_text(encoding="utf-8"))
    weight_res = json.loads(weight_json.read_text(encoding="utf-8"))

    out_dir = ROOT_DIR / "runs" / "funnel_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc, w, h = build_synthesis_card(funnel_res, weight_res)
    html_path = out_dir / "funnel_synthesis_architecture.html"
    png_path = out_dir / "funnel_synthesis_architecture.png"
    html_path.write_text(doc, encoding="utf-8")
    render_png_via_chrome(html_path, png_path, w, h)

    run_copy = funnel_json.parent / "fig_funnel_overall_synthesis.png"
    shutil.copy2(png_path, run_copy)
    print(f"[Synthesis Figure] Rendered {png_path} ({w*2}x{h*2} 2x Retina) and {run_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
