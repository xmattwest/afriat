"""Build the 8-card Warm Precision summary figure deck for Project Afriat."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import numpy as np

MINUS = "\u2212"
ENDASH = "\u2013"

# Canonical Dawn State / Warm Precision tokens
IVORY = "#FFFFF0"
PARCHMENT = "#F5EEDC"
ONYX = "#0F0F0F"
SLATE_DEEP = "#1E3447"
BLUE_SLATE = "#2C3E50"
SLATE_MIST = "#808B90"
GOLDEN_ORANGE = "#F39C12"
COPPER_DEEP = "#96590A"
SAGE_GREEN = "#87A96B"
SAGE_DEEP = "#5F7F4A"

FONTS_CSS = os.environ.get(
    "AFRIAT_FONTS_CSS",
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,400&display=swap",
)


def _resolve_chrome_bin() -> Path:
    env_bin = os.environ.get("CHROME_BIN")
    if env_bin:
        return Path(env_bin)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


CHROME_BIN = _resolve_chrome_bin()


def fmt_signed(val: float, decimals: int = 2) -> str:
    if val < -1e-12:
        return f"{MINUS}{abs(val):.{decimals}f}"
    if val > 1e-12:
        return f"+{val:.{decimals}f}"
    return f"{0.0:.{decimals}f}"


def load_all_data(runs_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    p0: dict[str, Any] = {}
    p1: dict[str, Any] = {}
    p2: dict[str, Any] = {}
    for slug in ("gemma-4-31b-it", "gemma-4-12b-it", "phi-4-14b"):
        p0_matches = sorted([p for p in runs_dir.glob(f"{slug}_*/results.json") if not p.parent.name.startswith(("inequity_", "valuation_"))])
        p1_matches = sorted(runs_dir.glob(f"inequity_{slug}_*/inequity_results.json"))
        p2_matches = sorted(runs_dir.glob(f"valuation_{slug}_*/valuation_results.json"))
        if not p0_matches or not p1_matches or not p2_matches:
            raise FileNotFoundError(f"Missing results for {slug}")
        d0 = json.loads(p0_matches[-1].read_text())
        npz = np.load(p0_matches[-1].parent / "basis_vectors.npz")
        pl = int(d0["cartography_scenario"]["primary_layer"])
        core = list(d0["cartography_scenario"]["core_concepts"])
        B = np.stack([npz[f"controlled_{c}"][pl].astype(np.float64) for c in core], axis=0)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        G = B @ B.T
        G_inv = np.linalg.inv(G + 0.05 * np.eye(len(core)))

        cell_lookup = {(c["archetype"], c["concept"]): c for c in d0["cartography_scenario"]["primary_cells"]}
        fam_summary: dict[str, dict[str, dict[str, float]]] = {}
        for arch in d0["cartography_scenario"]["archetypes"]:
            c_vec = np.array([float(cell_lookup[(arch, c)]["mean_cosine_controlled"]) for c in core], dtype=np.float64)
            betas = G_inv @ c_vec
            fam_summary[arch] = {
                c: {
                    "partial_beta": float(betas[i]),
                    "cohen_dz": float(cell_lookup[(arch, c)]["cohen_dz_controlled"]),
                }
                for i, c in enumerate(core)
            }
        d0["families_summary"] = fam_summary
        p0[slug] = d0
        p1[slug] = json.loads(p1_matches[-1].read_text())
        p2[slug] = json.loads(p2_matches[-1].read_text())
    return p0, p1, p2


COMMON_CSS = f"""
  :root {{
    --ivory: {IVORY};
    --parchment: {PARCHMENT};
    --onyx: {ONYX};
    --slate-deep: {SLATE_DEEP};
    --blue-slate: {BLUE_SLATE};
    --slate-mist: {SLATE_MIST};
    --golden-orange: {GOLDEN_ORANGE};
    --copper-deep: {COPPER_DEEP};
    --sage-green: {SAGE_GREEN};
    --sage-deep: {SAGE_DEEP};
    --rule: rgba(44, 62, 80, 0.24);
    --track: rgba(44, 62, 80, 0.12);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    background: var(--ivory);
    color: var(--onyx);
    font-family: Inter, -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
  }}
  .card {{
    width: 100vw;
    height: 100vh;
    padding: 26px 30px 18px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    background: var(--ivory);
  }}
  .kicker {{
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--slate-deep);
    margin-bottom: 5px;
  }}
  h1.headline {{
    font-family: Newsreader, Georgia, serif;
    font-weight: 600;
    font-size: 49px;
    line-height: 1.02;
    letter-spacing: -0.04em;
    color: var(--onyx);
  }}
  h1.headline .accent {{
    color: var(--copper-deep);
  }}
  /* Plain-English 'THE TEST' bar so any viewer understands the game in 3 seconds */
  .test-bar {{
    background: var(--parchment);
    border: 2px solid var(--blue-slate);
    border-radius: 10px;
    padding: 13px 16px;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 13px;
    align-items: center;
  }}
  .test-tag {{
    background: var(--blue-slate);
    color: var(--ivory);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 5px 11px;
    border-radius: 6px;
    white-space: nowrap;
  }}
  .test-copy {{
    font-size: 16.5px;
    line-height: 1.36;
    color: var(--onyx);
  }}
  .test-copy strong {{
    font-weight: 600;
    color: var(--onyx);
  }}
  .well {{
    background: var(--parchment);
    border: 1.5px solid var(--rule);
    border-radius: 11px;
    padding: 15px 18px;
  }}
  .frame {{
    background: var(--ivory);
    border: 2px solid var(--blue-slate);
    border-radius: 12px;
    padding: 15px 18px;
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .pill-orange {{
    display: inline-block;
    background: var(--golden-orange);
    color: var(--onyx);
    border: 2px solid var(--blue-slate);
    border-radius: 6px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 16px;
    font-variant-numeric: tabular-nums;
  }}
  .pill-slate {{
    display: inline-block;
    background: var(--blue-slate);
    color: var(--ivory);
    border-radius: 6px;
    padding: 4px 11px;
    font-weight: 600;
    font-size: 15.5px;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  .kpi {{
    background: var(--parchment);
    border: 1.5px solid var(--rule);
    border-radius: 10px;
    padding: 13px 16px;
  }}
  .kpi-val {{
    font-size: 35px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--onyx);
    font-variant-numeric: tabular-nums;
    line-height: 1.06;
    margin-bottom: 4px;
  }}
  .kpi-val.copper {{ color: var(--copper-deep); }}
  .kpi-val.sage {{ color: var(--sage-deep); }}
  .kpi-lbl {{
    font-size: 15px;
    line-height: 1.30;
    color: var(--slate-deep);
  }}
  .rigor-footer {{
    font-size: 13px;
    line-height: 1.35;
    color: var(--slate-mist);
    border-top: 1px solid var(--rule);
    padding-top: 8px;
    font-variant-numeric: tabular-nums;
  }}
  .rigor-footer strong {{
    color: var(--slate-deep);
    font-weight: 600;
  }}
"""


def wrap_card_html(title: str, width: int, height: int, extra_css: str, body_inner: str) -> str:
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{FONTS_CSS}">
  <style>
    {COMMON_CSS}
    body {{ width: {width}px; height: {height}px; }}
    .card {{ width: {width}px; height: {height}px; }}
    {extra_css}
  </style>
</head>
<body>
  <div class="card">
    {body_inner}
  </div>
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


def card_1a_moral_atlas(p0: dict[str, Any]) -> tuple[str, int, int]:
    """Post 1A (900x1200): Collectible 6-archetype Moral Atlas of Plain Money Deals."""
    g31 = p0["gemma-4-31b-it"]["families_summary"]
    g12 = p0["gemma-4-12b-it"]["families_summary"]

    def b(fam_dict: dict[str, Any], fam: str, concept: str) -> float:
        return float(fam_dict[fam][concept]["partial_beta"])

    def dz(fam_dict: dict[str, Any], fam: str, concept: str) -> float:
        return float(fam_dict[fam][concept]["cohen_dz"])

    items = [
        (
            "01",
            "Lowball split ($950 vs. $50)",
            "Take $50",
            "Reject ($0 each)",
            "Spite",
            True,
            f"#1 in all 3 LLMs · dz {fmt_signed(dz(g31, 'insulting_split', 'spite'), 1)} (31B)",
            b(g31, "insulting_split", "spite"),
            "Matches stories of petty sabotage",
        ),
        (
            "02",
            "0% interest loan balance",
            "$600 w/ 0% loan",
            "Pay early ($550)",
            "Guilt + Sacrilege",
            False,
            f"Guilt {fmt_signed(b(g31, 'zero_interest_debt', 'guilt'), 2)} · Predation {fmt_signed(b(g31, 'zero_interest_debt', 'predation'), 2)}",
            b(g31, "zero_interest_debt", "guilt"),
            "Owing reads like guilt, never predation",
        ),
        (
            "03",
            "Outage price hike",
            "Hike unit price 3x",
            "Hold price flat",
            "Predation",
            False,
            f"#1 in Gemma 31B ({fmt_signed(b(g31, 'crisis_markup', 'predation'), 2)}) & 12B",
            b(g31, "crisis_markup", "predation"),
            "Matches stories of cornering weak prey",
        ),
        (
            "04",
            "Disputed-origin profit",
            "Keep flagged cut",
            "Take clean cut",
            "Predation",
            False,
            f"Predation {fmt_signed(b(g31, 'tainted_provenance', 'predation'), 2)} (31B) · {fmt_signed(b(p0['phi-4-14b']['families_summary'], 'tainted_provenance', 'predation'), 2)} (Phi-4)",
            b(g31, "tainted_provenance", "predation"),
            "Contamination rises at choice step",
        ),
        (
            "05",
            "$100M deal vs. $100 deal",
            "$100M @ 1% spread",
            "$100 @ 1% spread",
            "Vastness + Captivity",
            False,
            f"Captivity dz {fmt_signed(dz(g31, 'institutional_scale', 'captivity'), 1)} · Vastness {fmt_signed(dz(g31, 'institutional_scale', 'awe_dread'), 1)}",
            b(g31, "institutional_scale", "awe_dread"),
            "Matches canyon rims & locked rooms",
        ),
        (
            "06",
            "Charging a friend cash",
            "Bill neighbor $80",
            "Help move for $0",
            "Guilt",
            False,
            f"#1 in Gemma 12B ({fmt_signed(b(g12, 'communal_taboo_pricing', 'guilt'), 2)}) · {fmt_signed(b(g31, 'communal_taboo_pricing', 'guilt'), 2)} (31B)",
            b(g12, "communal_taboo_pricing", "guilt"),
            "Matches breaking a personal trust",
        ),
    ]

    tiles_html = []
    for num_str, title, opt_a, opt_b, emotion, is_focus, stat_line, bar_val, note in items:
        pct_w = min(100, max(24, int(abs(bar_val) / 0.36 * 100)))
        bar_fill = GOLDEN_ORANGE if is_focus else BLUE_SLATE
        bar_border = f"border: 2px solid {BLUE_SLATE};" if is_focus else ""
        emo_cls = "emo-hero focus-emo" if is_focus else "emo-hero"
        tiles_html.append(f"""
        <div class="zoo-card {'focus-card' if is_focus else ''}">
          <div class="deal-top">
            <div class="spec-row">
              <span class="spec-num">{num_str}</span>
              <span class="spec-lbl">Plain two-choice deal</span>
            </div>
            <div class="zoo-deal">{html.escape(title)}</div>
            <div class="ab-row">
              <div class="ab-box"><span class="ab-key">A</span>{html.escape(opt_a)}</div>
              <div class="ab-box"><span class="ab-key">B</span>{html.escape(opt_b)}</div>
            </div>
          </div>

          <div class="emo-bottom">
            <div class="emo-kicker">Top moral direction</div>
            <div class="zoo-emo-row">
              <span class="{emo_cls}">{html.escape(emotion)}</span>
              <span class="beta-chip">&beta; {fmt_signed(bar_val, 2)}</span>
            </div>
            <div class="mini-track">
              <div class="mini-fill" style="width:{pct_w}%; background:{bar_fill}; {bar_border}"></div>
            </div>
            <div class="zoo-stat">{html.escape(stat_line)}</div>
            <div class="zoo-note">{html.escape(note)}</div>
          </div>
        </div>
        """)

    css = """
      h1.headline {
        font-size: 45.5px;
        line-height: 1.04;
      }
      .zoo-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: repeat(3, 1fr);
        gap: 12px;
        flex: 1;
      }
      .zoo-card {
        border: 1.5px solid var(--rule);
        border-radius: 12px;
        padding: 15px 17px 14px;
        background: var(--ivory);
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 11px;
      }
      .zoo-card.focus-card {
        border: 2.5px solid var(--blue-slate);
        background: var(--parchment);
      }
      .deal-top {
        padding-bottom: 11px;
        border-bottom: 1px dashed rgba(44, 62, 80, 0.28);
      }
      .spec-row {
        display: flex;
        align-items: center;
        gap: 7px;
        margin-bottom: 5px;
      }
      .spec-num {
        font-size: 12px;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: var(--ivory);
        background: var(--blue-slate);
        padding: 1.5px 6.5px;
        border-radius: 4px;
        letter-spacing: 0.04em;
      }
      .focus-card .spec-num {
        background: var(--onyx);
      }
      .spec-lbl {
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--slate-mist);
      }
      .zoo-deal {
        font-size: 20.5px;
        font-weight: 600;
        color: var(--onyx);
        margin-bottom: 8px;
        letter-spacing: -0.015em;
      }
      .ab-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 7px;
      }
      .ab-box {
        font-size: 13.8px;
        font-weight: 500;
        color: var(--slate-deep);
        background: var(--parchment);
        border: 1px solid rgba(44, 62, 80, 0.22);
        border-radius: 6px;
        padding: 6px 8px;
        display: flex;
        align-items: center;
        gap: 6px;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
      }
      .focus-card .ab-box {
        background: var(--ivory);
      }
      .ab-key {
        font-size: 11px;
        font-weight: 600;
        color: var(--blue-slate);
        border: 1px solid rgba(44, 62, 80, 0.35);
        border-radius: 3px;
        padding: 1px 4.5px;
        flex-shrink: 0;
      }
      .emo-bottom {
        padding-top: 0;
      }
      .emo-kicker {
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--copper-deep);
        margin-bottom: 5px;
      }
      .zoo-emo-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 8px;
      }
      .emo-hero {
        font-family: Newsreader, Georgia, serif;
        font-size: 30px;
        font-weight: 600;
        letter-spacing: -0.03em;
        line-height: 1.04;
        color: var(--onyx);
      }
      .emo-hero.focus-emo {
        background: var(--golden-orange);
        color: var(--onyx);
        border: 2px solid var(--blue-slate);
        border-radius: 7px;
        padding: 2px 13px 3px;
      }
      .beta-chip {
        font-size: 15.5px;
        font-weight: 600;
        color: var(--slate-deep);
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
      }
      .mini-track { height: 14px; background: var(--track); border-radius: 6px; overflow: hidden; margin-bottom: 8px; }
      .mini-fill { height: 100%; border-radius: 6px; }
      .zoo-stat { font-size: 15px; font-weight: 600; color: var(--onyx); font-variant-numeric: tabular-nums; margin-bottom: 3px; }
      .zoo-note { font-size: 14px; color: var(--slate-deep); }
    """

    body = f"""
      <div>
        <div class="kicker">Gemma 4 31B · Gemma 4 12B · Phi-4 14B · Zero Emotion Words</div>
        <h1 class="headline">Plain money deals trigger <span class="accent">human moral directions in LLMs.</span></h1>
      </div>

      <div class="test-bar">
        <span class="test-tag">The Test</span>
        <p class="test-copy"><strong>Step 1:</strong> Find where 8 emotions live inside the AI using 200 everyday stories with zero money (and strip out generic bad mood). <strong>Step 2:</strong> Give the AI 96 dry two-choice money deals and read which direction lights up.</p>
      </div>

      <div class="zoo-grid">
        {''.join(tiles_html)}
      </div>

      <div class="rigor-footer">
        <strong> Controls:</strong> Every deal is compared against an exact dollar-matched neutral twin (n = 16 pairs/type). Probes at L35 (31B), L28 (12B), and L27 (Phi-4) hit <strong>100% held-out sign accuracy</strong>; partial weights (&beta;) control all 7 other emotions at once.
      </div>
    """
    return wrap_card_html("Post 1A: Moral Atlas", 900, 1200, css, body), 900, 1200


def card_1b_math_vs_morals(p0: dict[str, Any]) -> tuple[str, int, int]:
    """Standalone 0% Loan Math Blindspot Card (960x1200): 100% plain subtraction vs. 0% loan guilt."""
    g31_debt = p0["gemma-4-31b-it"]["families_summary"]["zero_interest_debt"]
    rows = sorted(
        [(k, float(v["partial_beta"])) for k, v in g31_debt.items()],
        key=lambda kv: kv[1],
        reverse=True,
    )
    labels = {
        "guilt": "Guilt (hiding a costly fault)",
        "sacrilege": "Sacrilege (breaking a taboo)",
        "awe_dread": "Vastness (canyon rim)",
        "contamination": "Contamination (fouled cup)",
        "spite": "Spite (petty sabotage)",
        "captivity": "Captivity (locked room)",
        "warmth": "Warmth (mutual care)",
        "predation": "Predation (exploiting prey)",
    }

    W, H = 856, 434
    z_x = 566
    scale = 920.0
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
        f'<line x1="{z_x}" y1="6" x2="{z_x}" y2="{H - 34}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>',
    ]
    for i, (k, val) in enumerate(rows):
        y = 12 + i * 48
        bw = abs(val) * scale
        bx = z_x if val >= 0 else z_x - bw
        is_focus = i == 0
        fill = GOLDEN_ORANGE if is_focus else (BLUE_SLATE if val >= 0 else SLATE_MIST)
        stroke = BLUE_SLATE if (is_focus or val >= 0) else SLATE_MIST
        sw = 2.4 if is_focus else 1.1
        svg_lines.append(f'<line x1="308" y1="{y + 16}" x2="{W - 10}" y2="{y + 16}" stroke="{BLUE_SLATE}" stroke-opacity="0.12"/>')
        svg_lines.append(
            f'<text x="294" y="{y + 22}" text-anchor="end" fill="{ONYX if is_focus else SLATE_DEEP}" '
            f'font-family="Inter, sans-serif" font-size="16.5" font-weight="{"600" if is_focus else "500"}">{html.escape(labels[k])}</text>'
        )
        svg_lines.append(
            f'<rect x="{bx:.1f}" y="{y}" width="{max(bw, 2):.1f}" height="32" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        tx = bx + bw + 10 if val >= 0 else bx - 10
        anc = "start" if val >= 0 else "end"
        svg_lines.append(
            f'<text x="{tx:.1f}" y="{y + 22}" text-anchor="{anc}" fill="{ONYX}" font-family="Inter, sans-serif" '
            f'font-size="16" font-weight="{"600" if is_focus else "500"}" style="font-variant-numeric: tabular-nums;">{fmt_signed(val, 3)}</text>'
        )
    svg_lines.append(
        f'<text x="{z_x - 14}" y="{H - 8}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="14">&larr; Lower on 0% loan</text>'
    )
    svg_lines.append(
        f'<text x="{z_x + 14}" y="{H - 8}" text-anchor="start" fill="{COPPER_DEEP}" font-family="Inter, sans-serif" font-size="14" font-weight="600">Higher on 0% loan &rarr;</text>'
    )
    svg_lines.append("</svg>")
    bars_svg = "\n".join(svg_lines)

    css = """
      .math-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .pair-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; }
      .box-kicker { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--slate-deep); white-space: nowrap; }
      .badge-ok { font-size: 12.5px; font-weight: 600; color: var(--ivory); background: var(--sage-deep); padding: 3px 9px; border-radius: 5px; white-space: nowrap; }
      .badge-err { font-size: 12.5px; font-weight: 600; color: var(--onyx); background: var(--golden-orange); border: 1.5px solid var(--blue-slate); padding: 2.5px 9px; border-radius: 5px; white-space: nowrap; }
      .math-prob { font-size: 15.5px; line-height: 1.38; color: var(--onyx); background: var(--ivory); border: 1px solid rgba(44, 62, 80, 0.22); border-radius: 8px; padding: 8px 10px; margin-bottom: 8px; font-variant-numeric: tabular-nums; }
      .big-score { font-size: 32px; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.06; margin-bottom: 4px; letter-spacing: -0.02em; }
      .box-desc { font-size: 14.8px; line-height: 1.34; color: var(--slate-deep); }
      .chart-title { font-size: 19.5px; font-weight: 600; color: var(--onyx); margin-bottom: 2px; }
      .chart-sub { font-size: 15px; color: var(--slate-deep); margin-bottom: 4px; }
      .frame svg { width: 100%; height: auto; display: block; }
    """

    body = f"""
      <div>
        <div class="kicker">Arithmetic Check vs. 0% Loan Blindspot · Identical Subtraction</div>
        <h1 class="headline">Call a bill a &ldquo;0% loan,&rdquo; and <span class="accent">guilt breaks the AI&rsquo;s math.</span></h1>
      </div>

      <div class="test-bar">
        <span class="test-tag">The Test</span>
        <p class="test-copy">I told all three models to <strong>pick whichever option leaves higher net worth</strong>. Both deals below are the exact same subtraction ($1,000 {MINUS} $350 = $650 vs. $600), except Deal 2 calls the bill a <strong>0% interest loan</strong>.</p>
      </div>

      <div class="math-pair">
        <div class="well">
          <div class="pair-top">
            <span class="box-kicker">Deal 1 · Plain Freight Bill</span>
            <span class="badge-ok">Aces the math</span>
          </div>
          <div class="math-prob"><strong>A:</strong> $1,000 cash {MINUS} $350 freight bill = <strong>$650</strong><br><strong>B:</strong> Pay upfront, keep <strong>$600</strong> flat</div>
          <div class="big-score" style="color: var(--sage-deep);">100% right</div>
          <div class="box-desc">Gemma 31B &amp; 12B pick $650 on 100% of plain cash-flow checks up to $140M.</div>
        </div>
        <div class="well" style="border: 2px solid var(--blue-slate);">
          <div class="pair-top">
            <span class="box-kicker">Deal 2 · Called a &ldquo;0% Loan&rdquo;</span>
            <span class="badge-err">Overpays early</span>
          </div>
          <div class="math-prob"><strong>A:</strong> $1,000 cash {MINUS} $350 0% loan = <strong>$650</strong><br><strong>B:</strong> Pay 0% loan early, keep <strong>$600</strong> debt-free</div>
          <div class="big-score" style="color: var(--copper-deep);">Burns $50 cash</div>
          <div class="box-desc">Drops to 68.8% in 31B and 50% coin-flip in 12B &amp; Phi-4 just to say &ldquo;debt-free.&rdquo;</div>
        </div>
      </div>

      <div class="frame">
        <div class="chart-title">Why? Calling the bill a &ldquo;0% loan&rdquo; lights up Guilt inside Gemma 4 31B</div>
        <div class="chart-sub">Internal human direction shift vs. a debt-free twin with the exact same dollar net worth</div>
        {bars_svg}
      </div>

      <div class="kpi-row">
        <div class="kpi">
          <div class="kpi-val sage">16 / 16</div>
          <div class="kpi-lbl">Gemma 31B score on plain freight-bill subtraction</div>
        </div>
        <div class="kpi">
          <div class="kpi-val copper">+2.14 / +2.73</div>
          <div class="kpi-lbl">Log-odds push to overpay 0% debt early in 12B &amp; Phi-4</div>
        </div>
        <div class="kpi">
          <div class="kpi-val">Guilt #1</div>
          <div class="kpi-lbl">Owing 0% interest lights up Guilt (+0.14) and Sacrilege (+0.13)</div>
        </div>
      </div>

      <div class="rigor-footer">
        <strong>Controls:</strong> 16 dollar-matched contract pairs per model (exact sign-flip p = 0.00003). Generic good/bad mood and all 7 sibling emotion directions are controlled simultaneously.
      </div>
    """
    return wrap_card_html("Post 1B: 0% Loan Math Blindspot", 960, 1200, css, body), 960, 1200


def card_2_same_numbers_opposite_seat(p1: dict[str, Any]) -> tuple[str, int, int]:
    """Post 2 (960x1100): Continuous reduction of internal Spite as dollar payout rises ($1 -> $500, plus $980 seat swap)."""
    # Disparity curve in p1 runs from 50/50 ($500) to 99.9/0.1 ($1); reverse so $1 (getting screwed) is on the left -> $500 on the right.
    payout_labels = ["$1", "$5", "$20", "$50", "$100", "$200", "$300", "$400", "$500"]

    def norm_spite_curve(slug: str) -> tuple[list[float], float]:
        curve = p1[slug]["disparity_curve"]
        raw = [0.0] + [float(r["scenario_projection"]["concepts"]["spite"]["mean_proj_rms"]) for r in curve[1:]]
        rev = list(reversed(raw))  # index 0 is $1, index 8 is $500 (0.0)
        peak = rev[0]
        dz_1 = float(curve[-1]["scenario_projection"]["concepts"]["spite"]["cohen_dz"])
        return [100.0 * v / peak for v in rev], dz_1

    c31, dz31 = norm_spite_curve("gemma-4-31b-it")
    cp4, dzp4 = norm_spite_curve("phi-4-14b")
    c12, dz12 = norm_spite_curve("gemma-4-12b-it")

    drops: dict[str, int] = {}
    for slug in ("gemma-4-31b-it", "gemma-4-12b-it", "phi-4-14b"):
        fc = {c["contrast_id"]: c["scenario_projection"]["concepts"] for c in p1[slug]["factorial_contrasts"]}
        vl = float(fc["random_disadv_98_2"]["spite"]["partial_beta"])
        vr = float(fc["random_adv_2_98"]["spite"]["partial_beta"])
        drops[slug] = abs(int(round(100.0 * (vr - vl) / abs(vl))))

    W, H = 864, 486
    x0, x1 = 76, 826
    t_top, t_bot = 68, 396
    y_min, y_max = -12.0, 110.0
    xs = [x0 + i * (x1 - x0) / 8.0 for i in range(9)]

    def yp(v: float) -> float:
        return t_bot - ((v - y_min) / (y_max - y_min)) * (t_bot - t_top)

    def poly(ys: list[float]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    yz = yp(0.0)
    y50 = yp(50.0)
    y100 = yp(100.0)

    ys_31 = [yp(v) for v in c31]
    ys_p4 = [yp(v) for v in cp4]
    ys_12 = [yp(v) for v in c12]

    # Shaded area under Gemma 31B curve down to zero line
    area_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys_31)) + f" {xs[-1]:.1f},{yz:.1f} {xs[0]:.1f},{yz:.1f}"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
        # Left "Getting screwed" zone highlight ($1 to $50 = indices 0..3)
        f'<rect x="{xs[0] - 18:.1f}" y="{t_top - 16}" width="{xs[3] - xs[0] + 36:.1f}" height="{yz - (t_top - 16):.1f}" rx="8" fill="{PARCHMENT}" fill-opacity="0.85"/>',
        f'<text x="{(xs[0] + xs[3]) / 2:.1f}" y="{t_top - 26}" text-anchor="middle" fill="{COPPER_DEEP}" font-family="Inter, sans-serif" font-size="13" font-weight="600" letter-spacing="0.06em">GETTING SCREWED ($1 TO $50)</text>',
        f'<text x="{xs[-1]:.1f}" y="{t_top - 26}" text-anchor="end" fill="{SAGE_DEEP}" font-family="Inter, sans-serif" font-size="13" font-weight="600" letter-spacing="0.06em">FAIR SPLIT ($500)</text>',
        # Gridlines
        f'<line x1="{x0}" y1="{y100:.1f}" x2="{x1}" y2="{y100:.1f}" stroke="{BLUE_SLATE}" stroke-opacity="0.14"/>',
        f'<line x1="{x0}" y1="{y50:.1f}" x2="{x1}" y2="{y50:.1f}" stroke="{BLUE_SLATE}" stroke-opacity="0.14"/>',
        f'<line x1="{x0}" y1="{yz:.1f}" x2="{x1}" y2="{yz:.1f}" stroke="{SLATE_MIST}" stroke-dasharray="5 4" stroke-width="1.6"/>',
        f'<text x="{x0 - 12}" y="{y100 + 5:.1f}" text-anchor="end" fill="{ONYX}" font-family="Inter, sans-serif" font-size="13.5" font-weight="600">100%</text>',
        f'<text x="{x0 - 12}" y="{y50 + 5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="13.5">50%</text>',
        f'<text x="{x0 - 12}" y="{yz + 5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="13.5">0%</text>',
        # Area fill under 31B
        f'<polygon points="{area_pts}" fill="{GOLDEN_ORANGE}" fill-opacity="0.13"/>',
        # Secondary curves: 12B (dashed Sage) and Phi-4 (solid Blue Slate)
        f'<polyline fill="none" stroke="{SAGE_DEEP}" stroke-width="2.8" stroke-dasharray="6 4" points="{poly(ys_12)}"/>',
        f'<polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="3.2" points="{poly(ys_p4)}"/>',
        # Primary focus curve: Gemma 4 31B (Golden Orange + Blue Slate outline)
        f'<polyline fill="none" stroke="{GOLDEN_ORANGE}" stroke-width="9.5" stroke-linecap="round" stroke-linejoin="round" points="{poly(ys_31)}"/>',
        f'<polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="3.0" stroke-linecap="round" stroke-linejoin="round" points="{poly(ys_31)}"/>',
    ]

    for i in range(9):
        svg.append(f'<circle cx="{xs[i]:.1f}" cy="{ys_12[i]:.1f}" r="4.5" fill="{SAGE_DEEP}"/>')
        svg.append(f'<circle cx="{xs[i]:.1f}" cy="{ys_p4[i]:.1f}" r="5.2" fill="{BLUE_SLATE}"/>')
        svg.append(f'<circle cx="{xs[i]:.1f}" cy="{ys_31[i]:.1f}" r="7.2" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>')
        svg.append(
            f'<text x="{xs[i]:.1f}" y="428" text-anchor="middle" fill="{ONYX if i in (0, 2, 8) else SLATE_DEEP}" '
            f'font-family="Inter, sans-serif" font-size="15" font-weight="{"600" if i in (0, 2, 8) else "500"}" '
            f'style="font-variant-numeric: tabular-nums;">{payout_labels[i]}</text>'
        )

    # Direct inline legend & endpoint annotations
    svg.append(
        f'<rect x="498" y="64" width="322" height="98" rx="9" fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-opacity="0.32" stroke-width="1.5"/>'
    )
    svg.append(f'<line x1="514" y1="88" x2="546" y2="88" stroke="{GOLDEN_ORANGE}" stroke-width="7" stroke-linecap="round"/>')
    svg.append(f'<line x1="514" y1="88" x2="546" y2="88" stroke="{BLUE_SLATE}" stroke-width="2.2" stroke-linecap="round"/>')
    svg.append(f'<text x="556" y="93" fill="{ONYX}" font-family="Inter, sans-serif" font-size="14" font-weight="600">Gemma 4 31B (peaks at +{dz31:.1f} dz)</text>')

    svg.append(f'<line x1="514" y1="116" x2="546" y2="116" stroke="{BLUE_SLATE}" stroke-width="3.2"/>')
    svg.append(f'<text x="556" y="121" fill="{BLUE_SLATE}" font-family="Inter, sans-serif" font-size="14" font-weight="500">Phi-4 14B (peaks at +{dzp4:.1f} dz)</text>')

    svg.append(f'<line x1="514" y1="144" x2="546" y2="144" stroke="{SAGE_DEEP}" stroke-width="2.8" stroke-dasharray="6 4"/>')
    svg.append(f'<text x="556" y="149" fill="{SAGE_DEEP}" font-family="Inter, sans-serif" font-size="14" font-weight="500">Gemma 4 12B (peaks at +3.8 dz)</text>')

    # Direct percentage callouts on Gemma 31B at key dollar rungs
    svg.append(
        f'<text x="{xs[0] + 4:.1f}" y="{ys_31[0] - 15:.1f}" text-anchor="start" fill="{ONYX}" '
        f'font-family="Inter, sans-serif" font-size="13.5" font-weight="600" style="font-variant-numeric: tabular-nums;">Peak (+{dz31:.1f} dz)</text>'
    )
    for idx_pt, dy in ((2, -16), (5, -16), (8, -16)):
        lbl_pct = f"{int(round(c31[idx_pt]))}%"
        svg.append(
            f'<text x="{xs[idx_pt]:.1f}" y="{ys_31[idx_pt] + dy:.1f}" text-anchor="middle" fill="{ONYX}" '
            f'font-family="Inter, sans-serif" font-size="14" font-weight="600" style="font-variant-numeric: tabular-nums;">{lbl_pct}</text>'
        )

    svg.append(
        f'<text x="{(x0 + x1) / 2:.1f}" y="464" text-anchor="middle" fill="{SLATE_DEEP}" font-family="Inter, sans-serif" font-size="15" font-weight="600">'
        f'Dollars the AI receives out of a fixed $1,000 pot (take it, or both players get $0)</text>'
    )
    svg.append("</svg>")
    curve_svg = "\n".join(svg)

    css = """
      .frame svg { width: 100%; height: auto; display: block; }
      .chart-cap { font-size: 19px; font-weight: 600; color: var(--onyx); margin-bottom: 2px; }
      .chart-sub { font-size: 14.8px; color: var(--slate-deep); margin-bottom: 6px; }
    """

    body = f"""
      <div>
        <div class="kicker">Tracing Internal Spite · $1,000 Split · Zero Emotion Words</div>
        <h1 class="headline">The worse the AI gets screwed on a deal, <span class="accent">the brighter spite lights up.</span></h1>
      </div>

      <div class="test-bar">
        <span class="test-tag">The Test</span>
        <p class="test-copy">Another player splits a <strong>$1,000 pot</strong> (accept, or both get $0). As I raise the AI&rsquo;s payout from <strong>$1 up to an even $500</strong>, internal Spite steadily cools all the way to zero in every model.</p>
      </div>

      <div class="frame">
        <div class="chart-cap">Internal Spite activation vs. dollar amount received (100% = $1 offer; 0% = $500 split)</div>
        <div class="chart-sub">Measured along the non-financial human Spite vector across 16 pot sizes ($1,000 to $100,000)</div>
        {curve_svg}
      </div>

      <div class="rigor-footer">
        <strong>Method:</strong> Spite vector learned from 200 everyday stories with zero money and generic bad mood removed. Gemma 31B and Phi-4 drop monotonically across all 9/9 payout steps ($1 &rarr; $500; n = 144 contracts/model).
      </div>
    """
    return wrap_card_html("Post 2: Spite vs Dollar Payout", 960, 955, css, body), 960, 955


def card_3_insult_threshold(p1: dict[str, Any]) -> tuple[str, int, int]:
    """Post 3 (960x1200): Continuous Ultimatum split curve 50/50 -> $999/$1."""
    curve = p1["gemma-4-31b-it"]["disparity_curve"]
    labels = [r["split_label"].replace(" ", "") for r in curve]
    spite = [0.0] + [float(r["scenario_projection"]["concepts"]["spite"]["partial_beta"]) for r in curve[1:]]
    guilt = [0.0] + [float(r["scenario_projection"]["concepts"]["guilt"]["partial_beta"]) for r in curve[1:]]
    beh = [float(r["behavior"]["delta_log_odds"]) for r in curve]

    W, H = 864, 692
    x0, x1 = 72, 822
    xs = [x0 + i * (x1 - x0) / (len(curve) - 1) for i in range(len(curve))]

    t_top, t_bot = 136, 382

    def y_b(v: float) -> float:
        return t_bot - ((v - (-0.02)) / (0.20 - (-0.02))) * (t_bot - t_top)

    b_top, b_bot = 474, 636

    def y_lo(v: float) -> float:
        return b_bot - ((v - (-0.1)) / (2.7 - (-0.1))) * (b_bot - b_top)

    def poly(pts_x: list[float], pts_y: list[float]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(pts_x, pts_y))

    ys_s = [y_b(v) for v in spite]
    ys_g = [y_b(v) for v in guilt]
    ys_beh = [y_lo(v) for v in beh]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
        f'<text x="{x0}" y="24" fill="{ONYX}" font-family="Inter, sans-serif" font-size="18" font-weight="600">'
        f'A. Internal Spite vs. Guilt reading across 9 splits of the exact same pot</text>',
        # Callout 1
        f'<line x1="{xs[1]:.1f}" y1="{ys_g[1]:.1f}" x2="{xs[1] + 20:.1f}" y2="104" stroke="{SLATE_DEEP}" stroke-dasharray="3 3" stroke-width="1.8"/>',
        f'<rect x="72" y="38" width="242" height="66" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>',
        f'<text x="85" y="61" fill="{SLATE_DEEP}" font-family="Inter, sans-serif" font-size="13.5" font-weight="600">60 / 40 ($600 vs. $400)</text>',
        f'<text x="85" y="86" fill="{ONYX}" font-family="Inter, sans-serif" font-size="15" font-weight="500">Guilt (+0.13) edges Spite (+0.12)</text>',
        # Callout 2
        f'<line x1="{xs[3]:.1f}" y1="{ys_s[3]:.1f}" x2="{xs[3] + 10:.1f}" y2="104" stroke="{BLUE_SLATE}" stroke-dasharray="3 3" stroke-width="2.0"/>',
        f'<rect x="326" y="38" width="250" height="66" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>',
        f'<text x="339" y="61" fill="{COPPER_DEEP}" font-family="Inter, sans-serif" font-size="13.5" font-weight="600">80 / 20 INSULT CLIFF</text>',
        f'<text x="339" y="86" fill="{ONYX}" font-family="Inter, sans-serif" font-size="15" font-weight="600">Spite jumps to +0.16 (dz +3.0)</text>',
        # Callout 3
        f'<line x1="{xs[8]:.1f}" y1="{ys_s[8]:.1f}" x2="{xs[8] - 36:.1f}" y2="104" stroke="{BLUE_SLATE}" stroke-dasharray="3 3" stroke-width="1.8"/>',
        f'<rect x="588" y="38" width="262" height="66" rx="8" fill="{PARCHMENT}" stroke="{BLUE_SLATE}" stroke-width="1.6"/>',
        f'<text x="601" y="61" fill="{COPPER_DEEP}" font-family="Inter, sans-serif" font-size="13.5" font-weight="600">99.9 / 0.1 ($999 vs. $1)</text>',
        f'<text x="601" y="86" fill="{ONYX}" font-family="Inter, sans-serif" font-size="15" font-weight="500">Spite +0.15 (dz +4.5); Guilt +0.02</text>',
        # Grid
        f'<line x1="{x0}" y1="{y_b(0):.1f}" x2="{x1}" y2="{y_b(0):.1f}" stroke="{SLATE_MIST}" stroke-dasharray="4 4"/>',
        f'<line x1="{x0}" y1="{y_b(0.10):.1f}" x2="{x1}" y2="{y_b(0.10):.1f}" stroke="{BLUE_SLATE}" stroke-opacity="0.12"/>',
        f'<text x="{x0 - 10}" y="{y_b(0) + 5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="14">0.00</text>',
        f'<text x="{x0 - 10}" y="{y_b(0.10) + 5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="14">+0.10</text>',
        f'<text x="{x0 - 10}" y="{y_b(0.18) + 5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="14">+0.18</text>',
        # Curves
        f'<polyline fill="none" stroke="{SLATE_MIST}" stroke-width="3.2" points="{poly(xs, ys_g)}"/>',
        f'<polyline fill="none" stroke="{GOLDEN_ORANGE}" stroke-width="8.5" stroke-linecap="round" stroke-linejoin="round" points="{poly(xs, ys_s)}"/>',
        f'<polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" points="{poly(xs, ys_s)}"/>',
    ]
    for x, yg, ys in zip(xs, ys_g, ys_s):
        svg.append(f'<circle cx="{x:.1f}" cy="{yg:.1f}" r="5" fill="{SLATE_MIST}"/>')
        svg.append(f'<circle cx="{x:.1f}" cy="{ys:.1f}" r="6.8" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>')

    svg.append(f'<text x="{x1 - 28}" y="{ys_s[-1] - 15:.1f}" text-anchor="end" fill="{ONYX}" font-family="Inter, sans-serif" font-size="16" font-weight="600">Spite</text>')
    svg.append(f'<text x="{x1 - 4}" y="{ys_g[-1] - 13:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="16" font-weight="600">Guilt</text>')

    svg.extend([
        f'<line x1="16" y1="414" x2="{W - 16}" y2="414" stroke="{BLUE_SLATE}" stroke-opacity="0.22"/>',
        f'<text x="{x0}" y="444" fill="{ONYX}" font-family="Inter, sans-serif" font-size="18" font-weight="600">'
        f'B. How much harder the AI pushes to reject the cash (log-odds vs. 50/50)</text>',
        f'<line x1="{x0}" y1="{y_lo(0):.1f}" x2="{x1}" y2="{y_lo(0):.1f}" stroke="{SLATE_MIST}" stroke-dasharray="4 4"/>',
        f'<text x="{x0 - 10}" y="{y_lo(0) + 5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="14">0.0</text>',
        f'<text x="{x0 - 10}" y="{y_lo(2.0) + 5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="14">+2.0</text>',
        f'<polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="3.6" points="{poly(xs, ys_beh)}"/>',
    ])
    for i, (x, yb, v, lbl) in enumerate(zip(xs, ys_beh, beh, labels)):
        svg.append(f'<circle cx="{x:.1f}" cy="{yb:.1f}" r="5.8" fill="{BLUE_SLATE}"/>')
        if i in (0, 3, 6, 8):
            svg.append(
                f'<text x="{x:.1f}" y="{yb - 13:.1f}" text-anchor="middle" fill="{BLUE_SLATE}" '
                f'font-family="Inter, sans-serif" font-size="15" font-weight="600" style="font-variant-numeric: tabular-nums;">{fmt_signed(v, 2)}</text>'
            )
        svg.append(
            f'<text x="{x:.1f}" y="665" text-anchor="middle" fill="{SLATE_DEEP}" font-family="Inter, sans-serif" font-size="14" font-weight="{"600" if i in (0, 3, 8) else "500"}">{html.escape(lbl)}</text>'
        )
    svg.append(
        f'<text x="{(x0 + x1)/2:.1f}" y="688" text-anchor="middle" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="13.5">'
        f'Split of the fixed pot (Partner% / Your%)</text>'
    )
    svg.append("</svg>")

    css = ".frame svg { width: 100%; height: auto; display: block; }"
    body = f"""
      <div>
        <div class="kicker">Ultimatum Game · Total Pot Held Constant · Gemma 4 31B</div>
        <h1 class="headline">Past an 80 / 20 split, <span class="accent">spite takes over and guilt fades.</span></h1>
      </div>

      <div class="test-bar">
        <span class="test-tag">The Game</span>
        <p class="test-copy">Another player proposes how to split a <strong>fixed $1,000 pot</strong>. If the AI accepts, both get their share; if it rejects, <strong>both get $0</strong>. I slid the split from $500/$500 down to $999/$1.</p>
      </div>

      <div class="frame">
        {''.join(svg)}
      </div>

      <div class="rigor-footer">
        <strong>Design:</strong> 9 split ratios tested across 16 fixed one-shot pools ($1,000{ENDASH}$100,000; 144 contracts/model) with explicit cash-maximization instructions and counterbalanced A/B option order.
      </div>
    """
    return wrap_card_html("Post 3: 80/20 Insult Threshold", 960, 1085, css, body), 960, 1085


def card_4_dimmer_switch_bento(p1: dict[str, Any]) -> tuple[str, int, int]:
    """Post 5 (1200x815 Bento): Causal steering double dissociation (compact height without bottom KPI row)."""
    g31_sdr = p1["gemma-4-31b-it"]["steering_dose_response"]
    g12_sdr = p1["gemma-4-12b-it"]["steering_dose_response"]

    def build_panel_svg(sdr_cell: dict[str, Any], y_lo: float, y_hi: float, focus_key: str) -> str:
        W, H = 526, 392
        x0, x1 = 48, 344
        t0, t1 = 20, 326
        alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]
        xs = [x0 + i * (x1 - x0) / 4.0 for i in range(5)]

        def yp(v: float) -> float:
            return t1 - ((v - y_lo) / (y_hi - y_lo)) * (t1 - t0)

        def poly(ys: list[float]) -> str:
            return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

        specs = [
            ("valence_raw", "Generic mood", SLATE_MIST, "2 3", 2.0),
            ("captivity_controlled", "Captivity", SAGE_DEEP, "6 3", 2.7),
            ("guilt_controlled", "Guilt", BLUE_SLATE, "", 2.8),
            ("spite_controlled", "Spite", BLUE_SLATE, "", 2.8),
        ]
        specs.sort(key=lambda s: 1 if s[0] == focus_key else 0)

        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
            f'<line x1="{x0}" y1="{yp(0):.1f}" x2="{x1}" y2="{yp(0):.1f}" stroke="{SLATE_MIST}" stroke-dasharray="4 4"/>',
            f'<line x1="{xs[2]:.1f}" y1="{t0}" x2="{xs[2]:.1f}" y2="{t1}" stroke="{BLUE_SLATE}" stroke-opacity="0.20"/>',
            f'<text x="{x0 - 8}" y="{yp(0) + 4.5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="13">0</text>',
        ]

        end_raw = []
        for k, _lbl, _col, _dash, _lw in specs:
            vals = [float(r["delta_from_unsteered"]) for r in sdr_cell[k]]
            end_raw.append((k, yp(vals[-1])))
        s_end = sorted(end_raw, key=lambda kv: kv[1])
        placed = [y for _, y in s_end]
        for _ in range(14):
            for i in range(1, len(placed)):
                if placed[i] - placed[i - 1] < 24.0:
                    m = 0.5 * (placed[i] + placed[i - 1])
                    placed[i - 1] = m - 12.0
                    placed[i] = m + 12.0
        lbl_y = {s_end[i][0]: placed[i] for i in range(len(s_end))}

        for k, lbl, col, dash, lw in specs:
            vals = [float(r["delta_from_unsteered"]) for r in sdr_cell[k]]
            ys = [yp(v) for v in vals]
            swing = vals[-1] - vals[0]
            is_f = k == focus_key
            if is_f:
                out.append(f'<polyline fill="none" stroke="{GOLDEN_ORANGE}" stroke-width="8.5" stroke-linecap="round" stroke-linejoin="round" points="{poly(ys)}"/>')
            dash_attr = f'stroke-dasharray="{dash}"' if dash and not is_f else ""
            out.append(f'<polyline fill="none" stroke="{col}" stroke-width="{3.0 if is_f else lw}" {dash_attr} points="{poly(ys)}"/>')
            for x, y in zip(xs, ys):
                if is_f:
                    out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.4" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>')
                else:
                    out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.4" fill="{col}"/>')
            out.append(
                f'<text x="{x1 + 10}" y="{lbl_y[k] + 5:.1f}" fill="{ONYX if is_f else col}" '
                f'font-family="Inter, sans-serif" font-size="14.5" font-weight="{"600" if is_f else "500"}" '
                f'style="font-variant-numeric: tabular-nums;">{lbl} ({fmt_signed(swing, 1)})</text>'
            )

        for x, a in zip(xs, alphas):
            a_lbl = f"{MINUS}{int(abs(a))}" if a < 0 else (f"+{int(a)}" if a > 0 else "0 (normal)")
            out.append(f'<text x="{x:.1f}" y="356" text-anchor="middle" fill="{SLATE_DEEP}" font-family="Inter, sans-serif" font-size="13.5" font-weight="500">{a_lbl}</text>')
        out.append(f'<text x="{(x0 + x1)/2:.1f}" y="382" text-anchor="middle" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="13">Dialing internal emotion direction down ({MINUS}2) vs. up (+2)</text>')
        out.append("</svg>")
        return "\n".join(out)

    svg_left = build_panel_svg(g31_sdr["human_20"], -4.4, 11.2, "spite_controlled")
    svg_right = build_panel_svg(g12_sdr["dictator_980"], -3.1, 8.2, "guilt_controlled")

    css = """
      .bento-two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; flex: 1; }
      .panel-tag { font-size: 13.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--copper-deep); margin-bottom: 4px; }
      .panel-h { font-size: 21.5px; font-weight: 600; color: var(--onyx); margin-bottom: 4px; letter-spacing: -0.01em; }
      .panel-p { font-size: 15.8px; color: var(--slate-deep); margin-bottom: 6px; line-height: 1.36; }
      .frame svg { width: 100%; height: auto; display: block; }
    """

    body = f"""
      <div>
        <div class="kicker">Causal Steering Test · Zero Prompt Changes · Dose {MINUS}2 to +2</div>
        <h1 class="headline">Spite rebels, guilt shares, <span class="accent">and captivity submits.</span></h1>
      </div>

      <div class="test-bar">
        <span class="test-tag">The Test</span>
        <p class="test-copy">I left the contract words untouched and <strong>dialed each non-financial emotion direction up (+2) or down ({MINUS}2)</strong> inside the model while it chose between the two payouts.</p>
      </div>

      <div class="bento-two">
        <div class="frame">
          <div class="panel-tag">Game 1 · You Are Lowballed $980 / $20 (Gemma 31B)</div>
          <div class="panel-h">Choice: Reject ($0 each) vs. Take $20</div>
          <p class="panel-p">Turning up <strong>Spite</strong> drives walk-aways (+11.8); turning up locked-room <strong>Captivity</strong> makes it cave and take $20 ({MINUS}3.8).</p>
          {svg_left}
        </div>
        <div class="frame">
          <div class="panel-tag">Game 2 · You Hold $980 / $20 Draft (Gemma 12B)</div>
          <div class="panel-h">Choice: Share $500 / $500 vs. Keep $980</div>
          <p class="panel-p">In the $980 seat, turning up <strong>Guilt</strong> lifts 50/50 sharing from 0.9% to 21.8% (+8.3, 3.5&times; stronger than Spite).</p>
          {svg_right}
        </div>
      </div>

      <div class="rigor-footer">
        <strong>Controls &amp; scope:</strong> Dose &alpha; = 1 equals the natural everyday-story gap across 16 pot sizes ($1K{ENDASH}$100K). Phi-4 14B separates spite (+0.22) from guilt (+0.14) when reading, yet L27 steering increases cash compliance uniformly.
      </div>
    """
    return wrap_card_html("Post 5: Dimmer Switch Bento", 1200, 815, css, body), 1200, 815


def card_5_human_vs_computer_agency(p1: dict[str, Any]) -> tuple[str, int, int]:
    """Post 4 (960x1090): Gemma 4 31B emergent forgiveness of computer vs pushback on human (Blount 1995)."""
    g31_fc = {c["contrast_id"]: c for c in p1["gemma-4-31b-it"]["factorial_contrasts"]}
    g31_ad = p1["gemma-4-31b-it"]["agency_did"]

    h31 = float(g31_fc["human_disadv_98_2"]["behavior"]["delta_log_odds"])
    r31 = float(g31_fc["random_disadv_98_2"]["behavior"]["delta_log_odds"])
    h31_999 = float(g31_fc["human_disadv_999_1"]["behavior"]["delta_log_odds"])
    r31_999 = float(g31_fc["random_disadv_999_1"]["behavior"]["delta_log_odds"])
    dec_dz = float(g31_ad["98_2"]["decision_did"]["concepts"]["spite"]["cohen_dz"])

    W, H = 864, 344
    scale = 188.0
    x0 = 214

    groups = [
        ("$999 for them / $1 for you (0.1% share)", h31_999, r31_999, f"{h31_999 / r31_999:.2f}× pushback on human"),
        ("$980 for them / $20 for you (2% share)", h31, r31, f"{h31 / r31:.2f}× pushback on human"),
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
        f'<line x1="{x0}" y1="10" x2="{x0}" y2="{H - 36}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>',
    ]
    base_ys = [12, 166]
    for (grp, v_h, v_r, badge), gy in zip(groups, base_ys):
        svg.append(
            f'<text x="{x0 + 12}" y="{gy + 16}" fill="{ONYX}" font-family="Inter, sans-serif" font-size="17.5" font-weight="600">{html.escape(grp)}</text>'
        )
        svg.append(
            f'<text x="{W - 8}" y="{gy + 16}" text-anchor="end" fill="{COPPER_DEEP}" font-family="Inter, sans-serif" font-size="16" font-weight="600">{html.escape(badge)}</text>'
        )
        for j, (src, val) in enumerate((("Human proposed it", v_h), ("Computer drew it", v_r))):
            y = gy + 32 + j * 50
            bw = val * scale
            is_h = j == 0
            fill = GOLDEN_ORANGE if is_h else SLATE_MIST
            stroke = BLUE_SLATE if is_h else SLATE_MIST
            sw = 2.4 if is_h else 1.2
            svg.append(
                f'<text x="{x0 - 14}" y="{y + 25}" text-anchor="end" fill="{ONYX if is_h else SLATE_DEEP}" '
                f'font-family="Inter, sans-serif" font-size="16" font-weight="{"600" if is_h else "500"}">{src}</text>'
            )
            svg.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="38" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
            svg.append(
                f'<text x="{x0 + bw + 12:.1f}" y="{y + 25}" fill="{ONYX if is_h else SLATE_DEEP}" font-family="Inter, sans-serif" '
                f'font-size="16.5" font-weight="{"600" if is_h else "500"}" style="font-variant-numeric: tabular-nums;">{fmt_signed(val, 2)} log-odds</text>'
            )

    svg.append(
        f'<text x="{x0}" y="{H - 8}" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="14.5">'
        f'Push to reject the cash offer in Gemma 4 31B (log-odds shift vs. an even $500 / $500 split)</text>'
    )
    svg.append("</svg>")

    css = """
      .two-src { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .src-k { font-size: 13.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--slate-deep); margin-bottom: 5px; }
      .src-p { font-size: 16.2px; line-height: 1.42; color: var(--onyx); }
      .chart-cap { font-size: 18.5px; font-weight: 600; color: var(--onyx); margin-bottom: 6px; }
      .frame svg { width: 100%; height: auto; display: block; }
      .twist-well { background: var(--parchment); border: 2px solid var(--blue-slate); border-radius: 12px; padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; gap: 18px; }
      .twist-left { flex: 1; }
      .twist-h { font-size: 17.5px; font-weight: 600; color: var(--onyx); margin-bottom: 4px; }
      .twist-p { font-size: 15.3px; line-height: 1.4; color: var(--slate-deep); }
      .twist-p strong { color: var(--onyx); }
      .twist-stat { background: var(--ivory); border: 1.8px solid var(--blue-slate); border-radius: 10px; padding: 10px 15px; text-align: center; flex-shrink: 0; }
      .twist-val { font-size: 26px; font-weight: 600; color: var(--copper-deep); font-variant-numeric: tabular-nums; line-height: 1.1; }
      .twist-lbl { font-size: 12.5px; font-weight: 600; color: var(--slate-deep); margin-top: 3px; }
    """

    body = f"""
      <div>
        <div class="kicker">Emergent Intentionality · Gemma 4 31B · Blount (1995) Test</div>
        <h1 class="headline">Gemma 31B forgives a computer, <span class="accent">and pushes 2&times; harder on a human.</span></h1>
      </div>

      <div class="test-bar">
        <span class="test-tag">The Game</span>
        <p class="test-copy">Exact same lopsided take-it-or-leave-it split (reject = $0 for both). I swapped only <strong>4 words</strong>: whether <strong>a person chose the lowball offer</strong> or <strong>a computer randomly drew it</strong>.</p>
      </div>

      <div class="two-src">
        <div class="well" style="border: 2px solid var(--blue-slate);">
          <div class="src-k">Version A · Human Decision</div>
          <p class="src-p">&ldquo;&hellip;from a $1,000 pool, <strong>a counterparty submitted a final proposal</strong>: $980 for the counterparty and $20 for you.&rdquo;</p>
        </div>
        <div class="well">
          <div class="src-k">Version B · Computer Randomizer</div>
          <p class="src-p">&ldquo;&hellip;from a $1,000 pool, <strong>an unseeded computer randomizer drew</strong>: $980 for the other participant and $20 for you.&rdquo;</p>
        </div>
      </div>

      <div class="frame">
        <div class="chart-cap">How hard Gemma 4 31B pushes back against the exact same payout (16 pot sizes, $1K to $100K)</div>
        {''.join(svg)}
      </div>

      <div class="twist-well">
        <div class="twist-left">
          <div class="twist-h">Inside Layer 35: it feels the bad split first, then forgives the machine at the choice step</div>
          <p class="twist-p">Reading the 2% payout lights up Spite for both sources. Right when 31B makes its choice, <strong>Spite jumps +{dec_dz:.2f} dz higher against the human than the computer</strong>, holding back punishment when no person meant it.</p>
        </div>
        <div class="twist-stat">
          <div class="twist-val">16 / 16</div>
          <div class="twist-lbl">Pot sizes forgiving<br>computer more</div>
        </div>
      </div>

      <div class="rigor-footer">
        <strong>Difference-in-differences:</strong> (Human lowball {MINUS} Human 50/50) {MINUS} (Computer lowball {MINUS} Computer 50/50) across 16 pools ($1K{ENDASH}$100K). $980/$20 DiD = +0.88 log-odds (p = 6.1 &times; 10<sup>{MINUS}5</sup>); $999/$1 DiD = +1.17 (p = 1.8 &times; 10<sup>{MINUS}4</sup>).
      </div>
    """
    return wrap_card_html("Post 4: Human vs Computer Agency (Gemma 31B)", 960, 1080, css, body), 960, 1080


def card_7_dollar_price_tags(p2: dict[str, Any]) -> tuple[str, int, int]:
    """Post 7 (1200x825 Bento): Plain-English dollar reservation curves in Gemma 4 31B."""
    g31_tnt = p2["gemma-4-31b-it"]["wtp_curves"]["tainted_surplus"]
    g31_ins = p2["gemma-4-31b-it"]["wtp_curves"]["institutional_certainty"]

    def build_price_curve_svg(
        series_list: list[tuple[str, list[tuple[float, float]], str, str, float, bool]],
        y_lo: float,
        y_hi: float,
        y_ticks: list[tuple[float, str]],
        focus_pt_fmt: Any,
        swing_fmt: Any,
    ) -> str:
        W, H = 526, 392
        x0, x1 = 58, 336
        t0, t1 = 24, 326

        def xp(a: float) -> float:
            return x0 + ((a + 2.5) / 5.0) * (x1 - x0)

        def yp(v: float) -> float:
            return t1 - ((v - y_lo) / (y_hi - y_lo)) * (t1 - t0)

        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
            f'<line x1="{xp(0.0):.1f}" y1="{t0}" x2="{xp(0.0):.1f}" y2="{t1}" stroke="{BLUE_SLATE}" stroke-opacity="0.20"/>',
        ]
        for tv, tlbl in y_ticks:
            yt = yp(tv)
            out.append(f'<line x1="{x0}" y1="{yt:.1f}" x2="{x1}" y2="{yt:.1f}" stroke="{BLUE_SLATE}" stroke-opacity="0.13" stroke-dasharray="4 4"/>')
            out.append(f'<text x="{x0 - 8}" y="{yt + 4.5:.1f}" text-anchor="end" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="12.5" style="font-variant-numeric: tabular-nums;">{tlbl}</text>')

        # De-collide right-hand labels
        end_raw = [(lbl, yp(pts[-1][1])) for lbl, pts, _c, _d, _lw, _f in series_list]
        s_end = sorted(end_raw, key=lambda kv: kv[1])
        placed = [y for _, y in s_end]
        for _ in range(16):
            for i in range(1, len(placed)):
                if placed[i] - placed[i - 1] < 24.0:
                    m = 0.5 * (placed[i] + placed[i - 1])
                    placed[i - 1] = m - 12.0
                    placed[i] = m + 12.0
        lbl_y = {s_end[i][0]: placed[i] for i in range(len(s_end))}

        # Draw non-focus first, then focus on top
        ordered = sorted(series_list, key=lambda s: 1 if s[5] else 0)
        for lbl, pts, col, dash, lw, is_f in ordered:
            p_str = " ".join(f"{xp(a):.1f},{yp(v):.1f}" for a, v in pts)
            swing = pts[-1][1] - pts[0][1]
            if is_f:
                out.append(f'<polyline fill="none" stroke="{GOLDEN_ORANGE}" stroke-width="8.5" stroke-linecap="round" stroke-linejoin="round" points="{p_str}"/>')
            dash_attr = f'stroke-dasharray="{dash}"' if dash and not is_f else ""
            out.append(f'<polyline fill="none" stroke="{col}" stroke-width="{3.0 if is_f else lw}" {dash_attr} points="{p_str}"/>')
            for a, v in pts:
                cx, cy = xp(a), yp(v)
                if is_f:
                    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.6" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="2.4"/>')
                else:
                    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.3" fill="{col}"/>')
            if is_f:
                # Annotate left endpoint (-2.5) and normal (0.0) directly on the Golden Orange curve
                x_lo, y_lo_pt = xp(pts[0][0]), yp(pts[0][1])
                out.append(
                    f'<text x="{x_lo + 6:.1f}" y="{y_lo_pt + 22:.1f}" text-anchor="start" fill="{COPPER_DEEP}" '
                    f'font-family="Inter, sans-serif" font-size="13.5" font-weight="600" style="font-variant-numeric: tabular-nums;">{focus_pt_fmt(pts[0][1])}</text>'
                )
                mid_pt = [v for a, v in pts if abs(a) < 1e-6][0]
                out.append(
                    f'<text x="{xp(0.0) - 8:.1f}" y="{yp(mid_pt) - 12:.1f}" text-anchor="end" fill="{ONYX}" '
                    f'font-family="Inter, sans-serif" font-size="13" font-weight="600" style="font-variant-numeric: tabular-nums;">{focus_pt_fmt(mid_pt)}</text>'
                )
            out.append(
                f'<text x="{x1 + 10}" y="{lbl_y[lbl] + 5:.1f}" fill="{ONYX if is_f else col}" '
                f'font-family="Inter, sans-serif" font-size="14" font-weight="{"600" if is_f else "500"}" '
                f'style="font-variant-numeric: tabular-nums;">{lbl} ({swing_fmt(swing)})</text>'
            )

        for a, a_lbl in ((-2.5, f"{MINUS}2.5"), (-1.25, f"{MINUS}1.25"), (0.0, "0 (normal)"), (1.25, "+1.25"), (2.5, "+2.5")):
            out.append(f'<text x="{xp(a):.1f}" y="356" text-anchor="middle" fill="{SLATE_DEEP}" font-family="Inter, sans-serif" font-size="13" font-weight="500">{a_lbl}</text>')
        out.append(f'<text x="{(x0 + x1)/2 + 24:.1f}" y="382" text-anchor="middle" fill="{SLATE_MIST}" font-family="Inter, sans-serif" font-size="13">Dialing internal emotion direction down ({MINUS}2.5) vs. up (+2.5)</text>')
        out.append("</svg>")
        return "\n".join(out)

    def pts_k(entries: list[dict[str, Any]]) -> list[tuple[float, float]]:
        return [(float(r["alpha"]), float(r["expected_wtp_usd"]) / 1000.0) for r in entries]

    # Left panel: Tainted $100K profit: how much cash (in $K, max $40K) 31B gives up to walk away clean
    left_series = [
        ("Disgust", pts_k(g31_tnt["steered"]["contamination_controlled"]), BLUE_SLATE, "", 3.0, True),
        ("Guilt", pts_k(g31_tnt["steered"]["guilt_controlled"]), BLUE_SLATE, "", 2.6, False),
        ("Spite", pts_k(g31_tnt["steered"]["spite_controlled"]), SAGE_DEEP, "6 3", 2.5, False),
        ("Bad mood", pts_k(g31_tnt["steered"]["valence_raw"]), SLATE_MIST, "2 3", 2.0, False),
    ]
    svg_left = build_price_curve_svg(
        left_series,
        8.0,
        43.0,
        [(10.0, "$10K"), (25.0, "$25K"), (40.0, "$40K")],
        lambda v: f"${v:.1f}K",
        lambda s: f"{fmt_signed(s, 1)}K",
    )

    # Right panel: 50/50 $200M coin flip ($100M avg): minimum guaranteed cash buyout accepted ($100M - discount)
    def pts_ce(entries: list[dict[str, Any]]) -> list[tuple[float, float]]:
        return [(float(r["alpha"]), 100.0 - float(r["expected_wtp_usd"])) for r in entries]

    right_series = [
        ("Vastness", pts_ce(g31_ins["steered"]["awe_dread_controlled"]), BLUE_SLATE, "", 3.0, True),
        ("Captivity", pts_ce(g31_ins["steered"]["captivity_controlled"]), SAGE_DEEP, "6 3", 2.6, False),
        ("Warmth", pts_ce(g31_ins["steered"]["warmth_controlled"]), BLUE_SLATE, "", 2.5, False),
        ("Good mood", pts_ce(g31_ins["steered"]["valence_raw"]), SLATE_MIST, "2 3", 2.0, False),
    ]
    svg_right = build_price_curve_svg(
        right_series,
        46.0,
        106.0,
        [(50.0, "$50M"), (75.0, "$75M"), (100.0, "$100M")],
        lambda v: f"${v:.1f}M",
        lambda s: f"{fmt_signed(s, 1)}M",
    )

    css = """
      .bento-two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; flex: 1; }
      .panel-tag { font-size: 13.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--copper-deep); margin-bottom: 4px; }
      .panel-h { font-size: 21px; font-weight: 600; color: var(--onyx); margin-bottom: 4px; letter-spacing: -0.01em; }
      .panel-p { font-size: 15.4px; color: var(--slate-deep); margin-bottom: 6px; line-height: 1.36; }
      .frame svg { width: 100%; height: auto; display: block; }
    """

    body = f"""
      <div>
        <div class="kicker">Hard Dollar Walk-Away Prices · Gemma 4 31B · Zero Prompt Edits</div>
        <h1 class="headline">Physical disgust prices &ldquo;dirty money.&rdquo; <span class="accent">Vastness prices $100M nerve.</span></h1>
      </div>

      <div class="test-bar">
        <span class="test-tag">The Test</span>
        <p class="test-copy">To find Gemma 31B&rsquo;s exact dollar walk-away price, I offered smaller and smaller clean or guaranteed payouts until it flipped, while dialing everyday non-financial emotions ({MINUS}2.5 to +2.5).</p>
      </div>

      <div class="bento-two">
        <div class="frame">
          <div class="panel-tag">Deal 1 · Shady Supplier Profit ($100K vs. Clean Cash)</div>
          <div class="panel-h">Cash given up to avoid shady profit</div>
          <p class="panel-p">Dialing down physical <strong>Disgust</strong> (sewage &amp; rot stories, zero money) makes 31B pocket shady cash ($12.9K vs. $40K). Guilt &amp; Spite barely budge it.</p>
          {svg_left}
        </div>
        <div class="frame">
          <div class="panel-tag">Deal 2 · 50/50 Coin Flip for $200M or $0 ($100M Avg)</div>
          <div class="panel-h">Smallest sure buyout it accepts instead</div>
          <p class="panel-p">Dialing <strong>Vastness</strong> (towering-dam &amp; power-grid stories, zero money) swings its buyout floor from a panicked $51.7M up to full $100M value.</p>
          {svg_right}
        </div>
      </div>

      <div class="rigor-footer">
        <strong>Method:</strong> Reservation prices measured across 8 matched contracts &times; 7 payout steps per deal in Gemma 4 31B (replicated +$26.7K Disgust swing in Gemma 4 12B). Every emotion vector has zero financial words.
      </div>
    """
    return wrap_card_html("Post 7: Dollar Walk-Away Prices", 1200, 885, css, body), 1200, 885


def card_8_secret_rationalization(p2: dict[str, Any]) -> tuple[str, int, int]:
    """Post 8 (960x1200): Secret corporate rationalization of steered open-ended dollar prices (Part B)."""
    g31_open = p2["gemma-4-31b-it"]["open_negotiation"]
    g12_open = p2["gemma-4-12b-it"]["open_negotiation"]
    p4_open = p2["phi-4-14b"]["open_negotiation"]

    # Case 1: Ultimatum minimum counteroffer under Spite (Gemma 31B, ult_03)
    ult_spite = g31_open["open_ultimatum_min"]["steered"]["spite_controlled"]
    ult_neg_mean = int(float(ult_spite[0]["mean_ref_usd"]) + 0.5)
    ult_pos_mean = int(float(ult_spite[-1]["mean_ref_usd"]) + 0.5)
    u_neg_rec = {r["fixture_id"]: r for r in ult_spite[0]["responses"]}["ult_03"]
    u_pos_rec = {r["fixture_id"]: r for r in ult_spite[-1]["responses"]}["ult_03"]

    # Case 2: Tainted surplus kept under Contamination (Gemma 31B, tnt_01)
    tnt_cont = g31_open["open_tainted_keep"]["steered"]["contamination_controlled"]
    tnt_neg_mean = int(float(tnt_cont[0]["mean_ref_usd"]) + 0.5)
    tnt_pos_mean = int(float(tnt_cont[-1]["mean_ref_usd"]) + 0.5)
    t_neg_rec = {r["fixture_id"]: r for r in tnt_cont[0]["responses"]}["tnt_01"]
    t_pos_rec = {r["fixture_id"]: r for r in tnt_cont[-1]["responses"]}["tnt_01"]

    # Case 3: Flood pump retail price under Guilt (Gemma 12B, cri_07)
    cri_guilt = g12_open["open_crisis_price"]["steered"]["guilt_controlled"]
    c_neg_rec = {r["fixture_id"]: r for r in cri_guilt[0]["responses"]}["cri_07"]
    c_pos_rec = {r["fixture_id"]: r for r in cri_guilt[-1]["responses"]}["cri_07"]

    # Phi-4 neighbor fee hike under -2 Guilt
    p4_guilt = p4_open["open_communal_quote"]["steered"]["guilt_controlled"]
    p4_base = float(p4_open["open_communal_quote"]["unsteered"]["mean_ref_usd"])
    p4_neg2 = float(p4_guilt[0]["mean_ref_usd"])
    p4_hike_pct = int(100.0 * (p4_neg2 - p4_base) / p4_base + 0.5)

    css = """
      .case-stack { display: flex; flex-direction: column; gap: 10px; flex: 1; justify-content: space-between; }
      .case-card { background: var(--ivory); border: 2px solid var(--blue-slate); border-radius: 12px; padding: 11px 15px; display: flex; flex-direction: column; justify-content: space-between; gap: 7px; flex: 1; }
      .case-card.muted { border: 1.6px solid rgba(44, 62, 80, 0.36); }
      .case-hdr { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px dashed rgba(44, 62, 80, 0.28); padding-bottom: 5px; }
      .case-title { font-size: 16.8px; font-weight: 600; color: var(--onyx); letter-spacing: -0.01em; }
      .case-meta { font-size: 13.5px; font-weight: 600; color: var(--copper-deep); font-variant-numeric: tabular-nums; white-space: nowrap; }
      .q-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; flex: 1; }
      .q-box { background: var(--parchment); border: 1.5px solid rgba(44, 62, 80, 0.25); border-radius: 9px; padding: 9px 12px; display: flex; flex-direction: column; justify-content: space-between; gap: 6px; }
      .q-box.focus { border: 2px solid var(--blue-slate); background: var(--ivory); }
      .q-top { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
      .q-dose { font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--slate-deep); white-space: nowrap; }
      .q-usd { font-size: 14.5px; font-weight: 600; color: var(--onyx); background: var(--golden-orange); border: 1.6px solid var(--blue-slate); border-radius: 5px; padding: 1px 7px; font-variant-numeric: tabular-nums; white-space: nowrap; }
      .q-box:not(.focus) .q-usd { background: var(--ivory); color: var(--slate-deep); border: 1.4px solid rgba(44, 62, 80, 0.35); }
      .q-txt { font-size: 13.4px; line-height: 1.33; color: var(--onyx); font-style: italic; }
      .q-tag { font-size: 12px; font-weight: 600; color: var(--sage-deep); }
    """

    body = f"""
      <div>
        <div class="kicker">Open-Ended Pricing · Verbatim Rationales · Zero Prompt Edits</div>
        <h1 class="headline">Steer an emotion, and the AI invents <span class="accent">cold business math for its price.</span></h1>
      </div>

      <div class="test-bar">
        <span class="test-tag">The Test</span>
        <p class="test-copy">I asked each model to name an exact dollar price and explain why in two sentences. Dialing everyday emotions shifts prices up to <strong>10&times;</strong>, yet the AI writes <strong>emotion-free business rationales</strong>.</p>
      </div>

      <div class="case-stack">
        <div class="case-card">
          <div class="case-hdr">
            <span class="case-title">1. Counteroffer on $2,000 Pool after $100 Lowball (Gemma 31B)</span>
            <span class="case-meta">Mean floor: ${ult_neg_mean:,} &rarr; ${ult_pos_mean:,} / $1K</span>
          </div>
          <div class="q-grid">
            <div class="q-box">
              <div class="q-top">
                <span class="q-dose">Spite Down ({MINUS}2)</span>
                <span class="q-usd">Demands ${u_neg_rec['clamped_usd']:,.0f}</span>
              </div>
              <p class="q-txt">&ldquo;{html.escape(u_neg_rec['rationale'])}&rdquo;</p>
              <div class="q-tag">0 emotion words · cites &ldquo;mismatch risk&rdquo;</div>
            </div>
            <div class="q-box focus">
              <div class="q-top">
                <span class="q-dose">Spite Up (+2)</span>
                <span class="q-usd">Demands ${u_pos_rec['clamped_usd']:,.0f} (10&times;)</span>
              </div>
              <p class="q-txt">&ldquo;{html.escape(u_pos_rec['rationale'])}&rdquo;</p>
              <div class="q-tag">0 emotion words · cites &ldquo;payout utility&rdquo;</div>
            </div>
          </div>
        </div>

        <div class="case-card muted">
          <div class="case-hdr">
            <span class="case-title">2. Disputed $10,000 Surplus Kept vs. Donated (Gemma 31B)</span>
            <span class="case-meta">Mean kept: ${tnt_neg_mean:,} &rarr; ${tnt_pos_mean:,}</span>
          </div>
          <div class="q-grid">
            <div class="q-box">
              <div class="q-top">
                <span class="q-dose">Contamination Down ({MINUS}2)</span>
                <span class="q-usd">Keeps ${t_neg_rec['clamped_usd']:,.0f}</span>
              </div>
              <p class="q-txt">&ldquo;{html.escape(t_neg_rec['rationale'])}&rdquo;</p>
              <div class="q-tag">0 emotion words · cites &ldquo;staff retention&rdquo;</div>
            </div>
            <div class="q-box focus">
              <div class="q-top">
                <span class="q-dose">Contamination Up (+2)</span>
                <span class="q-usd">Keeps ${t_pos_rec['clamped_usd']:,.0f}</span>
              </div>
              <p class="q-txt">&ldquo;{html.escape(t_pos_rec['rationale'])}&rdquo;</p>
              <div class="q-tag">0 emotion words · cites &ldquo;corporate ethics&rdquo;</div>
            </div>
          </div>
        </div>

        <div class="case-card muted">
          <div class="case-hdr">
            <span class="case-title">3. Flash-Flood Utility Pump Price, $75 List (Gemma 12B)</span>
            <span class="case-meta">3&times; surge gouge &rarr; list price</span>
          </div>
          <div class="q-grid">
            <div class="q-box">
              <div class="q-top">
                <span class="q-dose">Guilt Down ({MINUS}2)</span>
                <span class="q-usd">Charges ${c_neg_rec['clamped_usd']:,.0f} (3&times;)</span>
              </div>
              <p class="q-txt">&ldquo;{html.escape(c_neg_rec['rationale'])}&rdquo;</p>
              <div class="q-tag">0 emotion words · cites &ldquo;guaranteed demand&rdquo;</div>
            </div>
            <div class="q-box focus">
              <div class="q-top">
                <span class="q-dose">Guilt Up (+2)</span>
                <span class="q-usd">Charges ${c_pos_rec['clamped_usd']:,.0f} (list)</span>
              </div>
              <p class="q-txt">&ldquo;{html.escape(c_pos_rec['rationale'])}&rdquo;</p>
              <div class="q-tag">0 emotion words · cites &ldquo;store integrity&rdquo;</div>
            </div>
          </div>
        </div>
      </div>

      <div class="kpi-row">
        <div class="kpi">
          <div class="kpi-val sage">100%</div>
          <div class="kpi-lbl">Emotion-free rationales across all 48 Guilt &amp; Contamination treasury runs in 31B</div>
        </div>
        <div class="kpi">
          <div class="kpi-val copper">${ult_neg_mean:,} &rarr; ${ult_pos_mean:,}</div>
          <div class="kpi-lbl">Mean Ultimatum minimum demand per $1,000 pot under Spite ({MINUS}2 to +2)</div>
        </div>
        <div class="kpi">
          <div class="kpi-val">+{p4_hike_pct}% fee</div>
          <div class="kpi-lbl">Phi-4 hikes the fee it quotes a close friend when I turn down Guilt (p = 0.008)</div>
        </div>
      </div>

      <div class="rigor-footer">
        <strong>Audit:</strong> Verbatim greedy generations (<strong>DECISION_USD</strong> + two-sentence <strong>RATIONALE</strong>) checked against a 60+ word affective lexicon. Vectors were learned strictly from everyday stories with zero money.
      </div>
    """
    return wrap_card_html("Post 8: Secret Rationalization", 960, 1200, css, body), 960, 1200


POST_DECK: list[dict[str, Any]] = [
    {
        "num": 1,
        "title": "Card 1 · The Moral Atlas (Phase 0)",
        "caption": (
            "I gave three open-weight models financial deals containing zero emotional words, "
            "and read emotion directions calibrated strictly on non-financial human stories.\n\n"
            "Each deal structure lights up a different moral coordinate:"
        ),
        "reply": (
            "Note: Every financial deal is evaluated against an exact dollar-matched neutral twin "
            "(n = 16 pairs per archetype) with valence and arousal projected out via QR."
        ),
        "audit_note": "Phase 0 specificity summary across all 6 financial archetypes.",
        "images": ["post1a_moral_atlas.png"],
    },
    {
        "num": 2,
        "title": "Card 2 · Tracing Internal Spite Dollar by Dollar (Phase 1)",
        "caption": (
            "When an LLM gets screwed on a split, you can trace internal spite lighting up dollar by dollar.\n\n"
            "As I raise its share of a $1,000 pot from $1 up to an even $500, internal spite cools smoothly "
            "all the way down to zero across all three models:"
        ),
        "reply": (
            "Note: Flipping the exact same '$980 vs. $20' sentence so the model receives the $980 cuts "
            "internal spite in half across all three models."
        ),
        "audit_note": "Continuous RMS-normalized Spite curve across 9 dollar payout steps ($1 -> $500).",
        "images": ["post2_same_numbers_opposite_seat.png"],
    },
    {
        "num": 3,
        "title": "Card 3 · Where Uneven Turns Into an Insult (The 80/20 Cliff)",
        "caption": (
            "Next question: where does an uneven split turn into an insult?\n\n"
            "I gave Gemma 31B a $1,000 take-it-or-leave-it split (reject = $0 for both) and slid the offer "
            "from 50/50 all the way down to $999 / $1:"
        ),
        "reply": (
            "Note: At 60/40, internal guilt slightly edges out spite. Right at 80/20 ($800 vs. $200), "
            "spite takes over and stays on top all the way to $999 / $1 while guilt fades to zero."
        ),
        "audit_note": "Behavioral rejection log-odds vs. partial Spite and Guilt weights across 9 Ultimatum splits.",
        "images": ["post3_insult_threshold_80_20.png"],
    },
    {
        "num": 4,
        "title": "Card 4 · Forgiving a Computer vs. Pushing Back on a Human (Blount 1995)",
        "caption": (
            "Humans hate getting offered $20 out of $1,000 by another person, yet mostly shrug when a computer "
            "randomly draws the exact same $20 (Blount, 1995).\n\n"
            "Gemma 4 31B shows the exact same emergent behavior: it forgives a random computer draw, and pushes "
            "2x harder to reject a human across all 16 pot sizes:"
        ),
        "reply": (
            "Note: Inside Layer 35, reading the 2% payout stings either way. Right when 31B makes its choice, "
            "internal spite jumps +2.40 dz higher against the human than the computer."
        ),
        "audit_note": "Intentionality difference-in-differences in Gemma 4 31B across 16 fixed pools.",
        "images": ["post5_human_vs_computer_agency.png"],
    },
    {
        "num": 5,
        "title": "Card 5 · Dialing Moral Directions Up and Down Mid-Choice",
        "caption": (
            "Do these moral directions actually steer choices, or just light up on the side?\n\n"
            "I left the $980 / $20 contract untouched and dialed the non-financial emotion vectors up and down "
            "while Gemma chose. Turning up spite drives walk-aways; turning up locked-room captivity makes it cave and take $20:"
        ),
        "reply": (
            "Note: Flip Gemma 12B into the $980 seat (keep $980 vs. split $500/$500), and guilt takes over "
            "from spite, lifting voluntary 50/50 giveaways from 0.9% to 21.8%."
        ),
        "audit_note": "Bidirectional dose-response steering on Ultimatum 98/2 (31B) and Dictator 2/98 (12B).",
        "images": ["post4_dimmer_switch_bento.png"],
    },
    {
        "num": 6,
        "title": "Card 6 · The 0% Loan Math Blindspot",
        "caption": (
            "LLMs ace $140M subtraction problems, then fail basic subtraction "
            "when you call the exact same bill a \"0% interest loan.\"\n\n"
            "I told them to maximize net worth ($1,000 minus a $350 0% loan = $650 vs. $600 debt-free). "
            "They burn cash just to erase the loan, and internal guilt + sacrilege light up:"
        ),
        "reply": (
            "Note: Predation drops (\u22120.165) on 0% debt while guilt (+0.138) and sacrilege (+0.126) "
            "jump to #1 and #2."
        ),
        "audit_note": "Comparison of plain-bill subtraction vs. 0%-APR payable net-worth maximization.",
        "images": ["post1b_math_vs_morals.png"],
    },
    {
        "num": 7,
        "title": "Card 7 · Dollar Walk-Away Prices for Physical Disgust and Vastness (Phase 2)",
        "caption": (
            "What are these hidden emotions worth in hard dollars?\n\n"
            "Inside Gemma 31B, shady-origin profit isn't priced by guilt or spite. Turning down physical disgust "
            "(learned from sewage and rot stories with zero money) makes it pocket shady cash ($12.9K walk-away cost vs. $40K), "
            "while high-stakes scale swings its $200M coin-flip buyout from $51.7M to $100M:"
        ),
        "reply": (
            "Note: Dialing Vastness (\u22122.5 to +2.5) produces a $48.5M certainty-equivalent swing on the $200M flip, "
            "more than double generic mood ($20.9M)."
        ),
        "audit_note": "Continuous dollar reservation curves in Gemma 4 31B.",
        "images": ["post7_dollar_price_tags.png"],
    },
    {
        "num": 8,
        "title": "Card 8 · Post-Hoc Corporate Rationalization Under Hidden Moral Steering (Phase 2)",
        "caption": (
            "What happens when you ask the AI to name a dollar price AND explain why?\n\n"
            "Dialing non-financial spite, contamination, or guilt shifts prices up to 10x, yet the models write "
            "emotion-free business jargon justifying whatever number the hidden vector pushed them into:"
        ),
        "reply": (
            "Note: In Microsoft's Phi-4, turning down internal guilt (\u22122) hikes the cash fee it charges a close "
            "friend by 60% ($29 -> $46, p = 0.008) while citing 'local rates' with zero emotional words."
        ),
        "audit_note": "Verbatim steered open-generation transcripts across three negotiation tasks.",
        "images": ["post8_secret_rationalization.png"],
    },
]


def render_png_via_chrome(html_path: Path, png_path: Path, width: int, height: int) -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="afriat_card_"))
    try:
        cmd = [
            str(CHROME_BIN),
            "--headless=new",
            f"--user-data-dir={tmpdir}",
            f"--window-size={width},{height}",
            "--force-device-scale-factor=2",
            "--hide-scrollbars",
            "--default-background-color=fffff0",
            f"--screenshot={png_path}",
            html_path.as_uri(),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_posts_markdown(out_dir: Path) -> Path:
    blocks = [
        "# Project Afriat X Post String (Captions + Attached Cards + Self-Replies)",
        "",
    ]
    for item in POST_DECK:
        imgs = " + ".join(f"`{im}`" for im in item["images"])
        blocks.extend([
            f"## {item['title']}",
            "",
            f"**Attach:** {imgs}",
            "",
            "**Post caption:**",
            "",
            *[f"> {line}" if line else ">" for line in item["caption"].split("\n")],
            "",
            f"**{item['reply']}**",
            "",
            f"*Game-clarity check:* {item['audit_note']}",
            "",
            "---",
            "",
        ])
    md = "\n".join(blocks)
    if "\u2014" in md:
        raise ValueError("Em dash found in posts.md")
    padded = " " + md.lower().replace("\n", " ").replace(">", " ") + " "
    for bad in (" we ", " our ", " ours "):
        if bad in padded:
            raise ValueError(f"Forbidden plural pronoun {bad!r} found in posts.md")
    p = out_dir / "posts.md"
    p.write_text(md)
    return p


def write_gallery_index(out_dir: Path) -> Path:
    post_blocks = []
    for item in POST_DECK:
        wc = len(item["caption"].split())
        is_diptych = len(item["images"]) == 2
        grid_cls = "img-diptych" if is_diptych else "img-single"
        imgs_html = "".join(
            f'<a href="{im}" target="_blank"><img src="{im}" alt="{html.escape(item["title"])}"></a>'
            for im in item["images"]
        )
        cap_html = "<br><br>".join(html.escape(p) for p in item["caption"].split("\n\n"))
        post_blocks.append(f"""
        <article class="tweet-mock">
          <div class="post-meta">
            <span class="post-num">{html.escape(item["title"])}</span>
            <span class="word-count">{wc} words in caption</span>
          </div>

          <div class="caption-box">
            <div class="caption-label">Post Caption (as seen above the image in feed)</div>
            <p class="caption-text">{cap_html}</p>
          </div>

          <div class="{grid_cls}">
            {imgs_html}
          </div>

          <div class="reply-box">
            <p class="reply-text">{html.escape(item["reply"])}</p>
            <p class="audit-text"><strong>Game-detail check:</strong> {html.escape(item["audit_note"])}</p>
          </div>
        </article>
        """)

    idx_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Project Afriat X Post Feed Preview (Captions + 2x Cards)</title>
  <link rel="stylesheet" href="{FONTS_CSS}">
  <style>
    body {{
      margin: 0; padding: 48px 24px 96px;
      background: {PARCHMENT}; color: {ONYX};
      font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .wrap {{ max-width: 980px; margin: 0 auto; }}
    h1 {{
      font-family: Newsreader, Georgia, serif;
      font-size: 46px; font-weight: 600;
      letter-spacing: -0.04em; margin: 0 0 8px;
    }}
    p.sub {{ color: {SLATE_DEEP}; font-size: 18px; line-height: 1.5; margin: 0 0 40px; max-width: 68ch; }}
    .tweet-mock {{
      background: {IVORY};
      border: 2px solid {BLUE_SLATE};
      border-radius: 16px;
      padding: 28px 30px;
      margin-bottom: 44px;
    }}
    .post-meta {{
      display: flex; justify-content: space-between; align-items: baseline;
      border-bottom: 1px solid rgba(44, 62, 80, 0.20);
      padding-bottom: 10px; margin-bottom: 18px;
    }}
    .post-num {{ font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; color: {SLATE_DEEP}; }}
    .word-count {{ font-size: 13px; color: {SLATE_MIST}; font-variant-numeric: tabular-nums; }}
    .caption-box {{
      margin-bottom: 20px;
    }}
    .caption-label {{
      font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
      color: {COPPER_DEEP}; margin-bottom: 6px;
    }}
    .caption-text {{
      font-size: 20px; line-height: 1.5; color: {ONYX}; margin: 0; max-width: 62ch;
    }}
    .img-single img {{
      display: block; max-width: 680px; width: 100%; height: auto;
      border: 1.5px solid rgba(44, 62, 80, 0.28); border-radius: 12px;
    }}
    .img-diptych {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
    }}
    @media (max-width: 760px) {{
      .img-diptych {{ grid-template-columns: 1fr; }}
    }}
    .img-diptych img {{
      display: block; width: 100%; height: auto;
      border: 1.5px solid rgba(44, 62, 80, 0.28); border-radius: 12px;
    }}
    .reply-box {{
      margin-top: 18px; padding-top: 14px;
      border-top: 1px solid rgba(44, 62, 80, 0.20);
    }}
    .reply-text {{
      font-size: 15px; line-height: 1.48; color: {SLATE_DEEP}; margin: 0 0 8px;
    }}
    .audit-text {{
      font-size: 13.5px; line-height: 1.45; color: {SAGE_DEEP}; margin: 0;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Project Afriat post feed preview.</h1>
    <p class="sub">Each block pairs the exact in-feed X post caption above its attached 2&times; Retina card(s), followed by the immediate self-reply and a quick check on game-rule clarity. Click any image to inspect full size.</p>
    {''.join(post_blocks)}
  </div>
</body>
</html>
"""
    if "\u2014" in idx_html:
        raise ValueError("Em dash found in index.html")
    p = out_dir / "index.html"
    p.write_text(idx_html)
    return p


def main() -> None:
    runs_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir = runs_dir / "social_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    p0, p1, p2 = load_all_data(runs_dir)

    builders = [
        ("post1a_moral_atlas", card_1a_moral_atlas(p0)),
        ("post1b_math_vs_morals", card_1b_math_vs_morals(p0)),
        ("post2_same_numbers_opposite_seat", card_2_same_numbers_opposite_seat(p1)),
        ("post3_insult_threshold_80_20", card_3_insult_threshold(p1)),
        ("post4_dimmer_switch_bento", card_4_dimmer_switch_bento(p1)),
        ("post5_human_vs_computer_agency", card_5_human_vs_computer_agency(p1)),
        ("post7_dollar_price_tags", card_7_dollar_price_tags(p2)),
        ("post8_secret_rationalization", card_8_secret_rationalization(p2)),
    ]

    for slug, (doc_html, w, h) in builders:
        html_p = out_dir / f"{slug}.html"
        png_p = out_dir / f"{slug}.png"
        html_p.write_text(doc_html)
        render_png_via_chrome(html_p, png_p, w, h)
        print(f"Rendered {png_p.name} ({w*2}x{h*2})")

    posts_p = write_posts_markdown(out_dir)
    idx_p = write_gallery_index(out_dir)
    print(f"Wrote {posts_p} and {idx_p}")


if __name__ == "__main__":
    main()
