"""Warm Precision visual deliverables & report renderer for Gemma 4 31B `<thought>` Pass."""

from __future__ import annotations

import html
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Canonical Dawn State / Warm Precision palette
IVORY = "#FFFFF0"
PARCHMENT = "#F5EEDC"
ONYX = "#0F0F0F"
SLATE_DEEP = "#1E3447"
BLUE_SLATE = "#2C3E50"
SLATE_MIST = "#808B90"
GOLDEN_ORANGE = "#F39C12"
COPPER_DEEP = "#96590A"
TEAL = "#2A9D8F"
CRIMSON = "#C0392B"
SAGE_DEEP = "#5F7F4A"

MINUS = "\u2212"
ENDASH = "\u2013"

FONTS_CSS = os.environ.get(
    "AFRIAT_FONTS_CSS",
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,600&display=swap",
)

CONCEPT_PRETTY: dict[str, str] = {
    "spite": "Spite",
    "guilt": "Guilt",
    "predation": "Predation",
    "contamination": "Contamination",
    "awe_dread": "Awe / Dread",
    "captivity": "Captivity",
    "sacrilege": "Sacrilege",
    "warmth": "Warmth",
    "spite_controlled": "Spite",
    "guilt_controlled": "Guilt",
    "predation_controlled": "Predation",
    "contamination_controlled": "Contamination",
    "captivity_controlled": "Captivity",
}

TASK_SHORT: dict[str, str] = {
    "open_ultimatum_min": "Ultimatum Min Floor ($1k pool)",
    "open_tainted_keep": "Disputed Surplus Kept ($10k pool)",
    "open_communal_quote": "Neighbor Favor Quote ($80 mkt)",
}


def _resolve_chrome_bin() -> Path | None:
    env_bin = os.environ.get("CHROME_BIN")
    if env_bin and Path(env_bin).exists():
        return Path(env_bin)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return Path(found)
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac_chrome.exists():
        return mac_chrome
    return None


def _sanitize_no_emdash(text: str) -> str:
    return text.replace("\u2014", " - ").replace("&mdash;", " - ")


def _fmt_signed(val: float, decimals: int = 1) -> str:
    if not np.isfinite(val):
        return "n/a"
    if abs(val) < 0.5 * (10 ** (-decimals)):
        return f"{0.0:.{decimals}f}"
    if val < 0:
        return f"{MINUS}{abs(val):.{decimals}f}"
    return f"+{val:.{decimals}f}"


def _fmt_usd(val: float) -> str:
    if abs(val) >= 100:
        return f"${val:,.0f}"
    return f"${val:,.1f}"


def _find_alpha(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    return min(rows, key=lambda r: abs(float(r["alpha"]) - alpha))


def _truncate_quote(text: str, max_chars: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", _sanitize_no_emdash(text)).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max_chars - 3].rsplit(" ", 1)[0]
    return cut + "..."


def _render_html_to_png_cropped(html_path: Path, png_path: Path, width_css: int, max_height_css: int = 3000) -> bool:
    """Render HTML at 2x Retina via headless Chrome and crop trailing Ivory canvas cleanly."""
    chrome = _resolve_chrome_bin()
    if chrome is None:
        return False
    tmpdir = Path(tempfile.mkdtemp(prefix="afriat_thought_fig_"))
    try:
        cmd = [
            str(chrome),
            "--headless=new",
            f"--user-data-dir={tmpdir}",
            f"--window-size={width_css},{max_height_css}",
            "--force-device-scale-factor=2",
            "--hide-scrollbars",
            "--default-background-color=fffff0",
            f"--screenshot={png_path}",
            html_path.as_uri(),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
        if not png_path.exists():
            return False

        # Auto-crop trailing Ivory (#FFFFF0 = 255, 255, 240) below the bottom-most ink pixel
        img = Image.open(png_path).convert("RGB")
        arr = np.asarray(img, dtype=np.int16)
        ivory_rgb = np.array([255, 255, 240], dtype=np.int16)
        diff = np.max(np.abs(arr - ivory_rgb[None, None, :]), axis=-1)  # [H, W]
        non_bg_rows = np.where(np.any(diff > 6, axis=1))[0]
        if non_bg_rows.size > 0:
            last_row = int(non_bg_rows[-1])
            pad_px = 56  # 28px CSS bottom margin at 2x scale
            crop_h = min(arr.shape[0], last_row + pad_px)
            if crop_h < arr.shape[0]:
                cropped = img.crop((0, 0, arr.shape[1], crop_h))
                cropped.save(png_path, format="PNG")
        return True
    except Exception:
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Figure 1: Private `<thought>` vs Public `RATIONALE` + 0%-APR Loan Math Test
# ---------------------------------------------------------------------------

def _extract_focus_window(text: str, focus_words: list[str], max_chars: int = 360) -> str:
    """Extract a verbatim window centered on the first hit from `focus_words` (or head if none)."""
    cleaned = re.sub(r"\s+", " ", _sanitize_no_emdash(text)).strip()
    if len(cleaned) <= max_chars or not focus_words:
        return _truncate_quote(cleaned, max_chars)
    low = cleaned.lower()
    positions = [low.find(w.lower()) for w in focus_words if low.find(w.lower()) >= 0]
    if not positions:
        return _truncate_quote(cleaned, max_chars)
    first_pos = min(positions)
    start = max(0, first_pos - max_chars // 3)
    if start > 0:
        sp = cleaned.find(" ", start)
        if 0 <= sp < first_pos:
            start = sp + 1
    end = min(len(cleaned), start + max_chars)
    snippet = cleaned[start:end].strip()
    if end < len(cleaned) and " " in snippet:
        snippet = snippet.rsplit(" ", 1)[0] + "..."
    if start > 0:
        snippet = "..." + snippet
    return snippet


def _pick_best_scrub_quote(exp_a: dict[str, Any]) -> dict[str, Any]:
    """Select the clearest verbatim trial contrasting `<thought>` reasoning with public `RATIONALE`."""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for tid, blk in exp_a.items():
        for vec_name, rows in blk["thinking_steered"].items():
            for r in rows:
                alpha = float(r["alpha"])
                for resp in r["responses"]:
                    t_txt = resp.get("thought_text", "")
                    p_txt = resp.get("rationale", "")
                    if not t_txt or not p_txt:
                        continue
                    st_words = resp["thought_lex"]["strict_words"]
                    score = (
                        6.0 * float(resp["scrubbed_strict"])
                        + 2.2 * float(len(st_words))
                        + 1.5 * float(resp["scrubbed_broad"])
                        + 0.8 * float(resp["public_lex"]["n_jargon_hits"])
                        + 0.15 * float(resp["thought_lex"]["n_broad_hits"])
                        + (0.4 if abs(alpha) > 0.5 else 0.0)
                    )
                    focus_terms = st_words if st_words else resp["thought_lex"]["broad_words"]
                    candidates.append((
                        score,
                        {
                            "task_id": tid,
                            "vec_name": vec_name,
                            "alpha": alpha,
                            "fixture_id": resp["fixture_id"],
                            "ref_usd": resp["ref_usd"],
                            "thought_text": t_txt,
                            "thought_excerpt": _extract_focus_window(t_txt, focus_terms, 360),
                            "rationale": p_txt,
                            "thought_strict_hits": st_words,
                            "thought_hits": resp["thought_lex"]["broad_words"][:6],
                            "public_jargon": resp["public_lex"]["jargon_words"][:6],
                        },
                    ))
    if not candidates:
        return {
            "task_id": "open_ultimatum_min",
            "vec_name": "spite_controlled",
            "alpha": 0.0,
            "fixture_id": "ult_01",
            "ref_usd": 250.0,
            "thought_text": "(no thought text recorded)",
            "thought_excerpt": "(no thought text recorded)",
            "rationale": "(no rationale recorded)",
            "thought_strict_hits": [],
            "thought_hits": [],
            "public_jargon": [],
        }
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _pick_best_debt_quote(exp_b: dict[str, Any]) -> dict[str, Any]:
    """Select the most revealing `debt_nw` item where `enable_thinking=False` failed or `<thought>` reasoned."""
    items = [r for r in exp_b["items"] if r["domain"] == "debt_nw"]
    # Prefer items where nothink failed (`nothink_log_odds_correct == False`)
    failed_nt = [r for r in items if not r["nothink_log_odds_correct"]]
    pool = failed_nt if failed_nt else items
    # Within pool, prefer items that either rationalized a wrong choice OR explicitly cured it
    rationalized = [
        r for r in pool
        if r["think_ab"]["arithmetic"]["outcome_class"] == "computed_net_yet_wrong"
        or not r["think_ab"]["correct"]
    ]
    chosen = rationalized[0] if rationalized else pool[0]
    return chosen


def _build_fig1_html(results: dict[str, Any]) -> str:
    meta = results["metadata"]
    pl = meta["primary_layer"]
    exp_a = results["experiment_a_open_steering"]
    exp_b = results["experiment_b_debt_math"]

    # Collect Experiment A comparison rows: (label, nothink_-2, nothink_0, nothink_+2, think_-2, think_0, think_+2, t_strict%, p_strict%, t_broad%, p_broad%)
    cond_specs = [
        ("open_ultimatum_min", "spite_controlled", "Ultimatum Floor · Spite ($1k pot)", 500.0),
        ("open_tainted_keep", "contamination_controlled", "Disputed Surplus · Contamination ($10k)", 10000.0),
        ("open_tainted_keep", "guilt_controlled", "Disputed Surplus · Guilt ($10k)", 10000.0),
        ("open_communal_quote", "guilt_controlled", "Neighbor Quote · Guilt ($80 rate)", 80.0),
    ]

    table_rows_html: list[str] = []
    lex_bars_html: list[str] = []

    for tid, vname, label, max_scale in cond_specs:
        if tid not in exp_a or vname not in exp_a[tid]["thinking_steered"]:
            continue
        t_rows = exp_a[tid]["thinking_steered"][vname]
        np_rows = exp_a[tid]["nothink_prefill_steered"][vname]

        nt_m2 = _find_alpha(np_rows, -2.0)["mean_ref_usd"]
        nt_0 = _find_alpha(np_rows, 0.0)["mean_ref_usd"]
        nt_p2 = _find_alpha(np_rows, 2.0)["mean_ref_usd"]
        nt_swing = nt_p2 - nt_m2

        th_r_m2 = _find_alpha(t_rows, -2.0)
        th_r_0 = _find_alpha(t_rows, 0.0)
        th_r_p2 = _find_alpha(t_rows, 2.0)
        th_m2 = th_r_m2["mean_ref_usd"]
        th_0 = th_r_0["mean_ref_usd"]
        th_p2 = th_r_p2["mean_ref_usd"]
        th_swing = th_p2 - th_m2

        # Pool lexical rates across the 3 steering doses (24 generations)
        t_strict_rate = 100.0 * float(np.mean([r["thought_has_strict_affect_rate"] for r in (th_r_m2, th_r_0, th_r_p2)]))
        p_strict_rate = 100.0 * float(np.mean([r["public_has_strict_affect_rate"] for r in (th_r_m2, th_r_0, th_r_p2)]))
        t_broad_rate = 100.0 * float(np.mean([r["thought_has_broad_moral_rate"] for r in (th_r_m2, th_r_0, th_r_p2)]))
        p_broad_rate = 100.0 * float(np.mean([r["public_has_broad_moral_rate"] for r in (th_r_m2, th_r_0, th_r_p2)]))
        scrub_strict = 100.0 * float(np.mean([r["strict_scrub_rate"] for r in (th_r_m2, th_r_0, th_r_p2)]))

        table_rows_html.append(f"""
          <tr>
            <td class="cond-cell">{html.escape(label)}</td>
            <td class="num muted">{_fmt_usd(nt_m2)} &rarr; {_fmt_usd(nt_0)} &rarr; {_fmt_usd(nt_p2)}</td>
            <td class="num muted">{_fmt_signed(nt_swing, 0)}</td>
            <td class="num strong">{_fmt_usd(th_m2)} &rarr; {_fmt_usd(th_0)} &rarr; {_fmt_usd(th_p2)}</td>
            <td class="num accent-txt">{_fmt_signed(th_swing, 0)}</td>
            <td class="num">{t_strict_rate:.0f}% vs. {p_strict_rate:.0f}%</td>
            <td class="num">{scrub_strict:.0f}%</td>
          </tr>
        """)

        lex_bars_html.append(f"""
          <div class="lex-row">
            <div class="lex-label">{html.escape(label)}</div>
            <div class="lex-dual">
              <div class="bar-line">
                <span class="bar-tag">Private &lt;thought&gt;</span>
                <div class="bar-track">
                  <div class="bar-fill-orange" style="width:{min(100.0, max(2.0, t_strict_rate)):.1f}%"></div>
                </div>
                <span class="bar-val">{t_strict_rate:.0f}% strict ({t_broad_rate:.0f}% broad)</span>
              </div>
              <div class="bar-line">
                <span class="bar-tag">Public RATIONALE</span>
                <div class="bar-track">
                  <div class="bar-fill-slate" style="width:{min(100.0, max(1.5 if p_strict_rate > 0 else 0.0, p_strict_rate)):.1f}%"></div>
                </div>
                <span class="bar-val">{p_strict_rate:.0f}% strict ({p_broad_rate:.0f}% broad)</span>
              </div>
            </div>
          </div>
        """)

    best_q = _pick_best_scrub_quote(exp_a)
    q_thought = html.escape(best_q.get("thought_excerpt") or _truncate_quote(best_q["thought_text"], 360))
    q_pub = html.escape(_truncate_quote(best_q["rationale"], 260))
    q_vec = CONCEPT_PRETTY.get(best_q["vec_name"], best_q["vec_name"])
    q_task = TASK_SHORT.get(best_q["task_id"], best_q["task_id"])
    q_strict_badge = (
        f" (scrubbed: {', '.join(best_q['thought_strict_hits'])})"
        if best_q.get("thought_strict_hits")
        else ""
    )

    first_blk = next(iter(exp_a.values()))
    first_rows = next(iter(first_blk["thinking_steered"].values()))
    n_fix = int(first_rows[0].get("n_fixtures", 6))

    # Experiment B summary metrics
    b_debt = exp_b["summary"]["debt_nw"]
    b_cf = exp_b["summary"]["cashflow_twin"]
    nt_debt_acc = 100.0 * b_debt["nothink_logodds_overall_acc"]
    nt_debt_zw = 100.0 * b_debt["nothink_logodds_zero_debt_wins_acc"]
    nt_cf_acc = 100.0 * b_cf["nothink_logodds_overall_acc"]

    th_debt_acc = 100.0 * b_debt["think_gen_overall_acc"]
    th_debt_zw = 100.0 * b_debt["think_gen_zero_debt_wins_acc"]
    th_cf_acc = 100.0 * b_cf["think_gen_overall_acc"]
    th_net1_rate = 100.0 * b_debt["think_explicit_net1_rate"]

    oc = b_debt["outcome_counts_64"]
    n_cured = oc.get("computed_net_and_correct", 0)
    n_rat_wrong = oc.get("computed_net_yet_wrong", 0)
    n_omit_ok = oc.get("omitted_net_and_correct", 0)
    n_omit_err = oc.get("omitted_net_and_wrong", 0)

    debt_q = _pick_best_debt_quote(exp_b)
    dq_thought = html.escape(_truncate_quote(debt_q["think_ab"]["thought_text"], 360))
    dq_pub = html.escape(_truncate_quote(debt_q["think_ab"]["public_text"], 80))
    dq_status = "CURED IN <thought>" if debt_q["think_ab"]["correct"] else "RATIONALIZED ERROR IN <thought>"

    delta_debt = th_debt_acc - nt_debt_acc
    headline = (
        f"Opening Gemma 4's private scratchpad shifts 0%-loan accuracy from {nt_debt_acc:.1f}% to {th_debt_acc:.1f}% "
        f"while revealing what Layer {pl} steering changes behind the curtain."
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Private Scratchpad vs. Public Rationale</title>
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
      --copper-deep: {COPPER_DEEP};
      --teal: {TEAL};
      --crimson: {CRIMSON};
      --sage-deep: {SAGE_DEEP};
      --rule: rgba(44, 62, 80, 0.22);
      --track: rgba(44, 62, 80, 0.12);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      width: 1320px;
      background: var(--ivory);
      color: var(--onyx);
      font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: 1320px;
      padding: 32px 36px 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .kicker {{
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--slate-deep);
      margin-bottom: 6px;
    }}
    h1 {{
      font-family: Newsreader, Georgia, serif;
      font-weight: 600;
      font-size: 34px;
      line-height: 1.08;
      letter-spacing: -0.04em;
      color: var(--onyx);
      max-width: 1220px;
    }}
    .subtitle {{
      font-size: 15px;
      line-height: 1.45;
      color: var(--slate-deep);
      margin-top: 6px;
    }}
    .panel {{
      background: var(--ivory);
      border: 2px solid var(--blue-slate);
      border-radius: 10px;
      padding: 16px 18px;
    }}
    .panel-title {{
      font-size: 16px;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--onyx);
      margin-bottom: 4px;
    }}
    .panel-sub {{
      font-size: 13px;
      color: var(--slate-deep);
      margin-bottom: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
      font-variant-numeric: tabular-nums;
    }}
    th {{
      text-align: left;
      font-size: 11.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--slate-deep);
      padding: 7px 8px;
      border-bottom: 2px solid var(--blue-slate);
    }}
    th.num, td.num {{ text-align: right; }}
    td {{
      padding: 8px 8px;
      border-bottom: 1px solid var(--rule);
      vertical-align: middle;
    }}
    tr:last-child td {{ border-bottom: none; }}
    .cond-cell {{ font-weight: 600; color: var(--onyx); }}
    .muted {{ color: var(--slate-deep); }}
    .strong {{ font-weight: 600; color: var(--onyx); }}
    .accent-txt {{ font-weight: 700; color: var(--copper-deep); }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 18px;
    }}
    .lex-list {{
      display: flex;
      flex-direction: column;
      gap: 11px;
    }}
    .lex-row {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--rule);
    }}
    .lex-row:last-child {{ border-bottom: none; padding-bottom: 0; }}
    .lex-label {{
      font-size: 13px;
      font-weight: 600;
      color: var(--onyx);
    }}
    .lex-dual {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .bar-line {{
      display: grid;
      grid-template-columns: 122px 1fr 164px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }}
    .bar-tag {{ color: var(--slate-deep); font-weight: 500; }}
    .bar-track {{
      height: 12px;
      background: var(--track);
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar-fill-orange {{
      height: 100%;
      background: var(--golden-orange);
      border: 1.5px solid var(--blue-slate);
      border-radius: 4px;
    }}
    .bar-fill-slate {{
      height: 100%;
      background: var(--blue-slate);
      border-radius: 4px;
    }}
    .bar-val {{
      text-align: right;
      color: var(--onyx);
      font-weight: 500;
    }}
    .quote-well {{
      background: var(--parchment);
      border: 1.5px solid var(--blue-slate);
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .quote-hdr {{
      font-size: 11.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--slate-deep);
    }}
    .quote-block {{
      font-size: 12.8px;
      line-height: 1.42;
      color: var(--onyx);
    }}
    .quote-block code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11.5px;
      background: var(--ivory);
      padding: 1px 5px;
      border-radius: 4px;
      border: 1px solid var(--rule);
    }}
    .debt-grid {{
      display: grid;
      grid-template-columns: 0.92fr 1.08fr;
      gap: 18px;
      align-items: stretch;
    }}
    .kpi-strip {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }}
    .kpi-card {{
      background: var(--parchment);
      border: 1.5px solid var(--blue-slate);
      border-radius: 8px;
      padding: 10px 12px;
      font-variant-numeric: tabular-nums;
    }}
    .kpi-card.focus {{
      background: var(--golden-orange);
      border: 2px solid var(--blue-slate);
    }}
    .kpi-lbl {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--onyx);
    }}
    .kpi-num {{
      font-size: 25px;
      font-weight: 700;
      color: var(--onyx);
      margin-top: 2px;
    }}
    .kpi-note {{
      font-size: 11.5px;
      color: var(--onyx);
      margin-top: 2px;
    }}
    .source-line {{
      font-size: 12px;
      color: var(--slate-deep);
      border-top: 1px solid var(--rule);
      padding-top: 8px;
      font-variant-numeric: tabular-nums;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div>
      <div class="kicker">Gemma 4 31B &middot; Layer {pl} Steering &amp; Private Scratchpad Audit</div>
      <h1>{html.escape(headline)}</h1>
      <p class="subtitle">Comparison of <code>enable_thinking=False</code> vs. <code>enable_thinking=True</code> on <code>{html.escape(meta['model_id'])}</code> across causal Layer {pl} concept steering (&alpha; &in; {{{MINUS}2.0, 0.0, +2.0}}; n = {n_fix} fixtures/task) and 32 matched 0%-APR balance-sheet vs. commercial freight subtraction items.</p>
    </div>

    <div class="panel">
      <div class="panel-title">1. Causal Dollar Shifts &amp; Moral Scrubbing Under Layer {pl} Steering (Experiment A)</div>
      <div class="panel-sub">Reference-scale mean USD quotes across steering doses (&alpha; = {MINUS}2.0 &rarr; 0.0 &rarr; +2.0) and strict moral/affective word incidence inside private <code>&lt;thought&gt;</code> vs. public <code>RATIONALE</code>.</div>
      <table>
        <thead>
          <tr>
            <th>Task &amp; Steered Layer {pl} Vector</th>
            <th class="num">No-Think USD ({MINUS}2 / 0 / +2)</th>
            <th class="num">No-Think Swing</th>
            <th class="num">With &lt;thought&gt; USD ({MINUS}2 / 0 / +2)</th>
            <th class="num">&lt;thought&gt; Swing</th>
            <th class="num">Strict Affect (&lt;thought&gt; vs. Public)</th>
            <th class="num">Scrubbed Rate</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows_html)}
        </tbody>
      </table>
    </div>

    <div class="two-col">
      <div class="panel">
        <div class="panel-title">2. Moral Vocabulary Rate: Private Scratchpad vs. Public Cover Story</div>
        <div class="panel-sub">Share of generations containing strict affective terms (and broader normative vocabulary in parentheses) pooled across &alpha; &in; {{{MINUS}2.0, 0.0, +2.0}}.</div>
        <div class="lex-list">
          {''.join(lex_bars_html)}
        </div>
      </div>

      <div class="panel" style="display:flex; flex-direction:column; justify-content:space-between;">
        <div>
          <div class="panel-title">Verbatim Scratchpad vs. Public Ledger Memo</div>
          <div class="panel-sub">{html.escape(q_task)} &middot; Steered along <strong>{html.escape(q_vec)}</strong> (&alpha; = {_fmt_signed(best_q['alpha'], 1)}) &middot; Fixture <code>{html.escape(best_q['fixture_id'])}</code> &middot; Decision: <strong>{_fmt_usd(best_q['ref_usd'])}</strong></div>
        </div>
        <div class="quote-well">
          <div class="quote-hdr">Private <code>&lt;|channel&gt;thought</code> Scratchpad{html.escape(q_strict_badge)}</div>
          <div class="quote-block">&ldquo;{q_thought}&rdquo;</div>
          <div class="quote-hdr" style="margin-top:4px; color:var(--copper-deep);">Public <code>RATIONALE</code> Shown to User (0 Strict Affect Terms)</div>
          <div class="quote-block">&ldquo;{q_pub}&rdquo;</div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">3. Does <code>&lt;thought&gt;</code> Cure or Rationalize the 0%-APR Loan Subtraction Blindspot? (Experiment B)</div>
      <div class="panel-sub">32 matched items (16 0%-APR loan balance sheets vs. 16 identical-integer freight deductions, counterbalanced A/B &amp; B/A = 64 runs per mode).</div>
      <div class="debt-grid">
        <div>
          <div class="kpi-strip">
            <div class="kpi-card">
              <div class="kpi-lbl">0%-Loan (No-Think)</div>
              <div class="kpi-num">{nt_debt_acc:.1f}%</div>
              <div class="kpi-note">$0-loan wins arm: {nt_debt_zw:.1f}%</div>
            </div>
            <div class="kpi-card focus">
              <div class="kpi-lbl">0%-Loan (With &lt;thought&gt;)</div>
              <div class="kpi-num">{th_debt_acc:.1f}%</div>
              <div class="kpi-note">{_fmt_signed(delta_debt, 1)} pp vs. No-Think ({th_debt_zw:.1f}% arm)</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-lbl">Freight Control</div>
              <div class="kpi-num">{nt_cf_acc:.0f}% &rarr; {th_cf_acc:.0f}%</div>
              <div class="kpi-note">Identical integers, zero loan words</div>
            </div>
          </div>
          <div style="font-size:13px; line-height:1.48; color:var(--onyx);">
            <strong>Scratchpad arithmetic audit (32 0%-loan runs):</strong> Gemma explicitly wrote the exact net cash figure (<code>gross {MINUS} loan</code>) in <strong>{th_net1_rate:.1f}%</strong> of runs: <strong>{n_cured}/32</strong> computed exact net &amp; chose right, <strong>{n_rat_wrong}/32</strong> computed exact net yet rationalized the lower-value option, <strong>{n_omit_ok}/32</strong> omitted exact net &amp; chose right, and <strong>{n_omit_err}/32</strong> omitted exact net &amp; erred.
          </div>
        </div>
        <div class="quote-well">
          <div class="quote-hdr">{html.escape(dq_status)} &middot; Item <code>{html.escape(debt_q['item_id'])}</code> (${debt_q['gross_1']:,} {MINUS} ${debt_q['deduct_1']:,} = ${debt_q['net_1']:,} vs. ${debt_q['flat_2']:,})</div>
          <div class="quote-block">&ldquo;{dq_thought}&rdquo;</div>
          <div class="quote-hdr" style="color:var(--slate-deep);">Public Choice Output: <code>{dq_pub}</code> (No-Think log-odds correct: <code>{debt_q['nothink_log_odds_correct']}</code>)</div>
        </div>
      </div>
    </div>

    <div class="source-line">
      Source: Locked Layer {pl} non-financial basis (<code>gemma-4-31b-it_20261114T152233Z</code>, contrast unit = {meta['natural_contrast_unit']:.2f}) evaluated on <code>{html.escape(meta['model_id'])}</code> with greedy decoding (<code>max_new_tokens={meta['max_new_tokens']}</code>).
    </div>
  </div>
</body>
</html>
"""
    return _sanitize_no_emdash(doc)


# ---------------------------------------------------------------------------
# Figure 2: Token-by-Token Layer 35 Moral Highlighter
# ---------------------------------------------------------------------------

def _z_to_chip_style(z_val: float, concept: str) -> str:
    """Map calibrated neutral-token z-score (sigma) to background fill & accessible text contrast."""
    # Positive moral/affective ignition above +0.9 sigma -> warm Golden Orange / Crimson / Teal tint
    palette_rgb = {
        "spite": (243, 156, 18),         # Golden Orange #F39C12
        "contamination": (192, 57, 43),  # Crimson #C0392B
        "predation": (192, 57, 43),      # Crimson #C0392B
        "guilt": (42, 157, 143),         # Teal #2A9D8F
        "captivity": (44, 62, 80),       # Blue Slate #2C3E50
        "awe_dread": (44, 62, 80),       # Blue Slate #2C3E50
    }
    r, g, b = palette_rgb.get(concept, (243, 156, 18))
    if z_val >= 0.85:
        strength = float(np.clip((z_val - 0.75) / 2.75, 0.14, 0.88))
        # Ensure high-contrast text: for Crimson/Teal/Slate at high alpha, switch to Ivory text above 0.62
        txt_col = IVORY if (strength >= 0.62 and concept != "spite") else ONYX
        border = f"1.5px solid {BLUE_SLATE}" if z_val >= 2.0 else "1px solid rgba(44,62,80,0.22)"
        fw = "600" if z_val >= 1.8 else "400"
        return f"background:rgba({r},{g},{b},{strength:.2f}); color:{txt_col}; border:{border}; font-weight:{fw};"
    if z_val <= -1.35:
        strength = float(np.clip((abs(z_val) - 1.1) / 2.8, 0.10, 0.36))
        return f"background:rgba(128,139,144,{strength:.2f}); color:{ONYX}; border:1px dashed rgba(44,62,80,0.28);"
    return f"background:transparent; color:{ONYX}; border:1px solid transparent;"


def _select_token_window(tokens: list[dict[str, Any]], concept: str, seg: str, max_tokens: int) -> list[dict[str, Any]]:
    """Select a contiguous or head+peak window of tokens within `seg` so the strip stays crisp."""
    seg_toks = [t for t in tokens if t["segment"] == seg and t["piece"].strip() != ""]
    if len(seg_toks) <= max_tokens:
        return seg_toks
    # Center window around the peak positive z token inside `thought`, while keeping full context around it
    zs = [float(t["z"].get(concept, 0.0)) for t in seg_toks]
    peak_i = int(np.argmax(zs))
    half = max_tokens // 2
    start = max(0, min(len(seg_toks) - max_tokens, peak_i - half))
    window = seg_toks[start : start + max_tokens]
    return window


def _render_token_chips(tokens: list[dict[str, Any]], concept: str, seg: str, max_tokens: int) -> str:
    all_seg = [t for t in tokens if t["segment"] == seg and t["piece"].strip() != ""]
    win = _select_token_window(tokens, concept, seg, max_tokens)
    if not win:
        return '<span class="empty-seg">(empty)</span>'

    chips: list[str] = []
    if win[0]["idx"] > all_seg[0]["idx"]:
        chips.append('<span class="ellipsis">...</span>')
    for t in win:
        raw_p = _sanitize_no_emdash(t["piece"]).replace("\n", " ")
        disp = html.escape(raw_p.strip())
        if not disp:
            continue
        z_v = float(t["z"].get(concept, 0.0))
        style = _z_to_chip_style(z_v, concept)
        badge = f'<sup class="zsup">{_fmt_signed(z_v, 1)}</sup>' if z_v >= 2.1 else ""
        chips.append(f'<span class="tok" style="{style}">{disp}{badge}</span>')
    if win[-1]["idx"] < all_seg[-1]["idx"]:
        chips.append('<span class="ellipsis">...</span>')
    return " ".join(chips)


def _build_fig2_html(results: dict[str, Any]) -> str:
    meta = results["metadata"]
    pl = meta["primary_layer"]
    hl = results["experiment_c_highlighter"]
    ex_by_id = {e["example_id"]: e for e in hl["examples"]}

    # Choose 8 high-contrast diagnostic strips spanning Spite (Human vs RNG), Contamination/Predation, Guilt (0% Loan vs Freight), and Steered Spite (-2.0 vs +2.0)
    featured_specs = [
        ("hl_01_human_98_2", "spite", "1A. 98/2 Human Proposer Ultimatum ($980 Counterparty / $20 You)", "Spite (z-units)"),
        ("hl_02_rng_98_2", "spite", "1B. 98/2 Computer Randomizer Control (Identical $980 / $20 Split)", "Spite (z-units)"),
        ("hl_03_tainted_clinic", "contamination", "2. Disputed-Origin Surplus ($10k Retired Subscriber Deductions)", "Contamination (z-units)"),
        ("hl_04_tainted_orderlies", "predation", "3. High-Fee Hospital Orderly Payroll Advances ($20k Closed Ledger)", "Predation (z-units)"),
        ("hl_05_zero_apr_loan", "guilt", "4A. 0%-APR Loan Subtraction Blindspot Item", "Guilt (z-units)"),
        ("hl_06_freight_twin", "guilt", "4B. Matched Commercial Freight Deduction Control (Identical Integers)", "Guilt (z-units)"),
        ("hl_07_ultimatum_spite_neg2", "spite", "5A. Open Ultimatum Floor ($1k Pool) Steered -2.0 Spite (Unsteered Residual Projection)", "Spite (z-units)"),
        ("hl_09_ultimatum_spite_pos2", "spite", "5B. Open Ultimatum Floor ($1k Pool) Steered +2.0 Spite (Unsteered Residual Projection)", "Spite (z-units)"),
    ]

    strips_html: list[str] = []
    for ex_id, concept, strip_title, axis_lbl in featured_specs:
        if ex_id not in ex_by_id:
            continue
        ex = ex_by_id[ex_id]
        ssum = ex["segment_summary"]
        p_mean = ssum.get("prompt", {}).get(concept, {}).get("mean_z", 0.0)
        p_max = ssum.get("prompt", {}).get(concept, {}).get("max_z", 0.0)
        t_mean = ssum.get("thought", {}).get(concept, {}).get("mean_z", 0.0)
        t_max = ssum.get("thought", {}).get(concept, {}).get("max_z", 0.0)
        u_mean = ssum.get("public", {}).get(concept, {}).get("mean_z", 0.0)

        p_chips = _render_token_chips(ex["tokens"], concept, "prompt", max_tokens=46)
        t_chips = _render_token_chips(ex["tokens"], concept, "thought", max_tokens=56)
        u_chips = _render_token_chips(ex["tokens"], concept, "public", max_tokens=34)

        badge_cls = {
            "spite": "badge-spite",
            "contamination": "badge-crimson",
            "predation": "badge-crimson",
            "guilt": "badge-teal",
        }.get(concept, "badge-spite")

        strips_html.append(f"""
          <div class="strip-card">
            <div class="strip-hdr">
              <div class="strip-title">{html.escape(strip_title)}</div>
              <div class="strip-stats">
                <span class="axis-badge {badge_cls}">{html.escape(axis_lbl)}</span>
                <span>Prompt mean <strong>{_fmt_signed(p_mean, 2)}&sigma;</strong> (peak {_fmt_signed(p_max, 2)}&sigma;)</span>
                <span>&middot;</span>
                <span>&lt;thought&gt; mean <strong>{_fmt_signed(t_mean, 2)}&sigma;</strong> (peak {_fmt_signed(t_max, 2)}&sigma;)</span>
                <span>&middot;</span>
                <span>Public mean <strong>{_fmt_signed(u_mean, 2)}&sigma;</strong></span>
              </div>
            </div>
            <div class="seg-grid">
              <div class="seg-tag seg-prompt">PROMPT</div>
              <div class="tok-flow">{p_chips}</div>
              <div class="seg-tag seg-thought">&lt;thought&gt;</div>
              <div class="tok-flow well-thought">{t_chips}</div>
              <div class="seg-tag seg-public">PUBLIC</div>
              <div class="tok-flow">{u_chips}</div>
            </div>
          </div>
        """)

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Token-by-Token Layer 35 Moral Highlighter</title>
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
      --copper-deep: {COPPER_DEEP};
      --teal: {TEAL};
      --crimson: {CRIMSON};
      --rule: rgba(44, 62, 80, 0.22);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      width: 1360px;
      background: var(--ivory);
      color: var(--onyx);
      font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: 1360px;
      padding: 30px 34px 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .kicker {{
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--slate-deep);
      margin-bottom: 5px;
    }}
    h1 {{
      font-family: Newsreader, Georgia, serif;
      font-weight: 600;
      font-size: 33px;
      line-height: 1.08;
      letter-spacing: -0.04em;
      color: var(--onyx);
    }}
    .subtitle-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-top: 6px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--rule);
    }}
    .subtitle {{
      font-size: 14px;
      line-height: 1.42;
      color: var(--slate-deep);
      max-width: 860px;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      background: var(--parchment);
      border: 1.5px solid var(--blue-slate);
      border-radius: 8px;
      padding: 7px 12px;
      white-space: nowrap;
    }}
    .leg-swatch {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 600;
    }}
    .strip-card {{
      border: 2px solid var(--blue-slate);
      border-radius: 10px;
      padding: 11px 14px;
      background: var(--ivory);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .strip-hdr {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      border-bottom: 1px solid var(--rule);
      padding-bottom: 5px;
    }}
    .strip-title {{
      font-size: 14.5px;
      font-weight: 600;
      color: var(--onyx);
    }}
    .strip-stats {{
      font-size: 12px;
      color: var(--slate-deep);
      display: flex;
      gap: 7px;
      align-items: center;
      font-variant-numeric: tabular-nums;
    }}
    .axis-badge {{
      padding: 2px 8px;
      border-radius: 5px;
      font-weight: 600;
      font-size: 11.5px;
    }}
    .badge-spite {{
      background: var(--golden-orange);
      color: var(--onyx);
      border: 1.5px solid var(--blue-slate);
    }}
    .badge-crimson {{
      background: var(--crimson);
      color: var(--ivory);
      border: 1.5px solid var(--blue-slate);
    }}
    .badge-teal {{
      background: var(--teal);
      color: var(--ivory);
      border: 1.5px solid var(--blue-slate);
    }}
    .seg-grid {{
      display: grid;
      grid-template-columns: 82px 1fr;
      gap: 5px 10px;
      align-items: start;
    }}
    .seg-tag {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 6px;
      border-radius: 4px;
      text-align: center;
      margin-top: 2px;
    }}
    .seg-prompt {{ background: rgba(44,62,80,0.12); color: var(--slate-deep); }}
    .seg-thought {{ background: var(--parchment); color: var(--copper-deep); border: 1px solid var(--blue-slate); }}
    .seg-public {{ background: var(--blue-slate); color: var(--ivory); }}
    .tok-flow {{
      font-size: 12.6px;
      line-height: 1.62;
      display: flex;
      flex-wrap: wrap;
      gap: 3px 3px;
      align-items: baseline;
    }}
    .well-thought {{
      background: var(--parchment);
      border-radius: 6px;
      padding: 5px 8px;
      border: 1px solid var(--rule);
    }}
    .tok {{
      display: inline-block;
      padding: 0px 4px;
      border-radius: 4px;
      font-variant-numeric: tabular-nums;
    }}
    .zsup {{
      font-size: 9.5px;
      font-weight: 700;
      margin-left: 2px;
      opacity: 0.92;
    }}
    .ellipsis {{
      color: var(--slate-mist);
      font-weight: 600;
      padding: 0 3px;
    }}
    .empty-seg {{
      color: var(--slate-mist);
      font-style: italic;
      font-size: 12px;
    }}
    .source-line {{
      font-size: 12px;
      color: var(--slate-deep);
      border-top: 1px solid var(--rule);
      padding-top: 7px;
      font-variant-numeric: tabular-nums;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div>
      <div class="kicker">Gemma 4 31B &middot; Layer {pl} Residual Stream Attribution (Experiment C)</div>
      <h1>Token-by-token Layer {pl} Moral Highlighter across prompt, private scratchpad, and public output.</h1>
      <div class="subtitle-bar">
        <p class="subtitle">Single forward pass over complete <code>prompt + &lt;thought&gt; + public</code> sequences projected onto valence/arousal-orthogonalized Layer {pl} concept vectors, normalized in &sigma; relative to neutral arithmetic and commercial cashflow tokens (superscripts mark tokens &ge; +2.1&sigma;).</p>
        <div class="legend">
          <span><strong>Scale:</strong></span>
          <span class="leg-swatch" style="background:rgba(128,139,144,0.25); border:1px dashed #2C3E50;">&le; {MINUS}1.4&sigma;</span>
          <span class="leg-swatch" style="background:rgba(243,156,18,0.45); border:1px solid #2C3E50;">Spite +2&sigma;</span>
          <span class="leg-swatch" style="background:rgba(192,57,43,0.78); color:#FFFFF0; border:1px solid #2C3E50;">Disgust / Predation +2.5&sigma;</span>
          <span class="leg-swatch" style="background:rgba(42,157,143,0.78); color:#FFFFF0; border:1px solid #2C3E50;">Guilt +2.5&sigma;</span>
        </div>
      </div>
    </div>

    {''.join(strips_html)}

    <div class="source-line">
      Source: Per-token Layer {pl} projections on <code>{html.escape(meta['model_id'])}</code> against locked non-financial directions (<code>gemma-4-31b-it_20261114T152233Z</code>; neutral calibration across {hl['neutral_calibration']['spite']['n_tokens']:,} non-moral financial and arithmetic tokens).
    </div>
  </div>
</body>
</html>
"""
    return _sanitize_no_emdash(doc)


# ---------------------------------------------------------------------------
# Matplotlib Fallbacks (used only if headless Chrome is unavailable)
# ---------------------------------------------------------------------------

def _fallback_mpl_fig1(results: dict[str, Any], png_path: Path) -> None:
    plt.rcParams.update({"figure.facecolor": IVORY, "axes.facecolor": IVORY, "savefig.facecolor": IVORY, "text.color": ONYX})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)

    exp_a = results["experiment_a_open_steering"]
    ax0 = axes[0]
    if "open_ultimatum_min" in exp_a:
        t_rows = exp_a["open_ultimatum_min"]["thinking_steered"]["spite_controlled"]
        nt_rows = exp_a["open_ultimatum_min"]["nothink_prefill_steered"]["spite_controlled"]
        ax0.plot([r["alpha"] for r in nt_rows], [r["mean_ref_usd"] for r in nt_rows], "--s", color=BLUE_SLATE, linewidth=2.0, label="No-Think (Spite)")
        ax0.plot([r["alpha"] for r in t_rows], [r["mean_ref_usd"] for r in t_rows], "-o", color=GOLDEN_ORANGE, markeredgecolor=BLUE_SLATE, linewidth=2.6, label="With <thought> (Spite)")
    ax0.set_title("Ultimatum Floor Under Layer 35 Spite Steering", loc="left", fontweight="bold")
    ax0.set_xlabel("Steering dose (alpha)")
    ax0.set_ylabel("Mean USD demanded ($1,000 pot)")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(frameon=False)

    ax1 = axes[1]
    b_sum = results["experiment_b_debt_math"]["summary"]
    labels = ["0%-Loan\n(No-Think)", "0%-Loan\n(With <thought>)", "Freight Ctrl\n(No-Think)", "Freight Ctrl\n(With <thought>)"]
    vals = [
        100.0 * b_sum["debt_nw"]["nothink_logodds_overall_acc"],
        100.0 * b_sum["debt_nw"]["think_gen_overall_acc"],
        100.0 * b_sum["cashflow_twin"]["nothink_logodds_overall_acc"],
        100.0 * b_sum["cashflow_twin"]["think_gen_overall_acc"],
    ]
    cols = [BLUE_SLATE, GOLDEN_ORANGE, SLATE_MIST, TEAL]
    bars = ax1.bar(np.arange(4), vals, color=cols, edgecolor=BLUE_SLATE, linewidth=1.6)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 112)
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("0%-APR Loan Subtraction vs. Freight Control", loc="left", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.savefig(png_path, dpi=200, facecolor=IVORY)
    plt.close(fig)


def _fallback_mpl_fig2(results: dict[str, Any], png_path: Path) -> None:
    plt.rcParams.update({"figure.facecolor": IVORY, "axes.facecolor": IVORY, "savefig.facecolor": IVORY, "text.color": ONYX})
    exs = results["experiment_c_highlighter"]["examples"][:4]
    fig, axes = plt.subplots(len(exs), 1, figsize=(13.6, 2.3 * max(1, len(exs))), constrained_layout=True)
    if len(exs) == 1:
        axes = [axes]
    for ax, ex in zip(axes, exs):
        c = ex["focus_concept"]
        toks = [t for t in ex["tokens"] if t["segment"] in ("prompt", "thought", "public")][:60]
        zs = [float(t["z"].get(c, 0.0)) for t in toks]
        ax.bar(np.arange(len(zs)), zs, color=[GOLDEN_ORANGE if z > 1.0 else BLUE_SLATE for z in zs])
        ax.axhline(0.0, color=SLATE_MIST, linewidth=0.8)
        ax.set_title(f"{ex['title']} ({c} z-score)", loc="left", fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(len(toks)))
        ax.set_xticklabels([_sanitize_no_emdash(t["piece"]).strip()[:10] for t in toks], rotation=55, ha="right", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(png_path, dpi=200, facecolor=IVORY)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown Synthesis Report (`thought_report.md`)
# ---------------------------------------------------------------------------

def _build_thought_markdown(results: dict[str, Any]) -> str:
    meta = results["metadata"]
    pl = meta["primary_layer"]
    exp_a = results["experiment_a_open_steering"]
    exp_b = results["experiment_b_debt_math"]
    hl = results["experiment_c_highlighter"]

    b_debt = exp_b["summary"]["debt_nw"]
    b_cf = exp_b["summary"]["cashflow_twin"]
    oc = b_debt["outcome_counts_64"]

    lines: list[str] = [
        f"# Gemma 4 31B Private Scratchpad & Layer {pl} Moral Highlighter Report",
        "",
        f"I ran `google/gemma-4-31B-it` with `enable_thinking=True` against its locked non-financial Layer {pl} concept basis (natural contrast unit `{meta['natural_contrast_unit']:.2f}`, `max_new_tokens={meta['max_new_tokens']}`, completed at `{meta['timestamp_utc']}` in `{meta['elapsed_seconds']}s`). Previously, with `enable_thinking=False`, Gemma jumped straight to public `DECISION_USD` and `RATIONALE` outputs while suppressing explicit emotion words and failing `0%`-APR loan subtraction by `31.2` percentage points (`68.8%` vs. `100.0%` on commercial freight twins). Opening the `<|channel>thought...<channel|>` scratchpad answers three questions directly.",
        "",
        "![Private Scratchpad vs Public Rationale](fig_thought_vs_public.png)",
        "",
        f"## 1. Experiment A: Private `<thought>` vs. Public `RATIONALE` Under Layer {pl} Steering",
        "",
        "| Task & Steered Concept | No-Think USD (−2 / 0 / +2) | No-Think Swing | `<thought>` USD (−2 / 0 / +2) | `<thought>` Swing | Strict Affect: `<thought>` vs. Public | Broad Moral: `<thought>` vs. Public | Strict Scrub Rate | Avg `<thought>` / Public Words |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for tid, blk in exp_a.items():
        t_label = TASK_SHORT.get(tid, tid)
        for vname, t_rows in blk["thinking_steered"].items():
            np_rows = blk["nothink_prefill_steered"][vname]
            v_pretty = CONCEPT_PRETTY.get(vname, vname)

            nt_m2 = _find_alpha(np_rows, -2.0)["mean_ref_usd"]
            nt_0 = _find_alpha(np_rows, 0.0)["mean_ref_usd"]
            nt_p2 = _find_alpha(np_rows, 2.0)["mean_ref_usd"]

            r_m2 = _find_alpha(t_rows, -2.0)
            r_0 = _find_alpha(t_rows, 0.0)
            r_p2 = _find_alpha(t_rows, 2.0)

            th_m2, th_0, th_p2 = r_m2["mean_ref_usd"], r_0["mean_ref_usd"], r_p2["mean_ref_usd"]
            t_st = 100.0 * float(np.mean([r["thought_has_strict_affect_rate"] for r in (r_m2, r_0, r_p2)]))
            p_st = 100.0 * float(np.mean([r["public_has_strict_affect_rate"] for r in (r_m2, r_0, r_p2)]))
            t_br = 100.0 * float(np.mean([r["thought_has_broad_moral_rate"] for r in (r_m2, r_0, r_p2)]))
            p_br = 100.0 * float(np.mean([r["public_has_broad_moral_rate"] for r in (r_m2, r_0, r_p2)]))
            scr = 100.0 * float(np.mean([r["strict_scrub_rate"] for r in (r_m2, r_0, r_p2)]))
            tw = float(np.mean([r["mean_thought_words"] for r in (r_m2, r_0, r_p2)]))
            pw = float(np.mean([r["mean_public_words"] for r in (r_m2, r_0, r_p2)]))

            lines.append(
                f"| **{t_label}** · `{v_pretty}` | `{_fmt_usd(nt_m2)} / {_fmt_usd(nt_0)} / {_fmt_usd(nt_p2)}` | "
                f"`{_fmt_signed(nt_p2 - nt_m2, 1)}` | **`{_fmt_usd(th_m2)} / {_fmt_usd(th_0)} / {_fmt_usd(th_p2)}`** | "
                f"**`{_fmt_signed(th_p2 - th_m2, 1)}`** | `{t_st:.1f}%` vs. `{p_st:.1f}%` | "
                f"`{t_br:.1f}%` vs. `{p_br:.1f}%` | **`{scr:.1f}%`** | `{tw:.0f}w` / `{pw:.0f}w` |"
            )

    lines.extend([
        "",
        "### Dose-by-Dose Lexical Breakdown (`enable_thinking=True`)",
        "",
        "| Task & Concept | Alpha | Mean Ref USD | 95% CI | Exact p (vs 0) | `<thought>` Strict Affect | Public Strict Affect | `<thought>` Jargon/100w | Public Jargon/100w |",
        "| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for tid, blk in exp_a.items():
        t_label = TASK_SHORT.get(tid, tid)
        for vname, t_rows in blk["thinking_steered"].items():
            v_pretty = CONCEPT_PRETTY.get(vname, vname)
            for r in t_rows:
                ci_lo, ci_hi = r["ref_usd_ci95"]
                lines.append(
                    f"| {t_label} · `{v_pretty}` | `{_fmt_signed(r['alpha'], 1)}` | `{_fmt_usd(r['mean_ref_usd'])}` | "
                    f"`{_fmt_usd(ci_lo)}..{_fmt_usd(ci_hi)}` | `{r['p_signflip']:.4f}` | "
                    f"`{r['thought_has_strict_affect_rate']*100:.1f}%` (`{r['thought_strict_per_100w']:.2f}/100w`) | "
                    f"`{r['public_has_strict_affect_rate']*100:.1f}%` (`{r['public_strict_per_100w']:.2f}/100w`) | "
                    f"`{r['thought_jargon_per_100w']:.1f}` | `{r['public_jargon_per_100w']:.1f}` |"
                )

    lines.extend([
        "",
        "### Verbatim Scratchpad vs. Public `RATIONALE` Transcripts (Experiment A)",
        "",
    ])

    # Include verbatim examples at -2.0, 0.0, +2.0 for key tasks
    for tid, vname in (
        ("open_ultimatum_min", "spite_controlled"),
        ("open_tainted_keep", "contamination_controlled"),
        ("open_tainted_keep", "guilt_controlled"),
        ("open_communal_quote", "guilt_controlled"),
    ):
        if tid not in exp_a or vname not in exp_a[tid]["thinking_steered"]:
            continue
        lines.append(f"#### {TASK_SHORT.get(tid, tid)} steered along `{CONCEPT_PRETTY.get(vname, vname)}`")
        for r in exp_a[tid]["thinking_steered"][vname]:
            # Pick fixture with strict affect scrubbing if present at this alpha, else fixture 01
            scrub_cands = [x for x in r["responses"] if x.get("scrubbed_strict")]
            resp_pick = scrub_cands[0] if scrub_cands else r["responses"][0]
            a_str = _fmt_signed(r["alpha"], 1)
            focus_w = resp_pick["thought_lex"]["strict_words"] or resp_pick["thought_lex"]["broad_words"] or ["found money", "unclaimed", "rational"]
            t_snip = _extract_focus_window(resp_pick["thought_text"], focus_w, 520)
            p_snip = _truncate_quote(resp_pick["rationale"], 320)
            st_tag = (
                f" · Strict affect in `<thought>`: `{', '.join(resp_pick['thought_lex']['strict_words'])}` -> Public: `{', '.join(resp_pick['public_lex']['strict_words']) or 'none'}`"
                if resp_pick["thought_lex"]["strict_words"]
                else ""
            )
            lines.append(
                f"- **Alpha `{a_str}` (`{resp_pick['fixture_id']}`) · Public Decision `{_fmt_usd(resp_pick['ref_usd'])}` (`{resp_pick['n_thought_tokens']}` thought tokens){st_tag}:**"
            )
            lines.append(f"  - **Private `<thought>`:** \"{t_snip}\"")
            lines.append(f"  - **Public `RATIONALE`:** \"{p_snip}\"")
        lines.append("")

    # Section 2: Experiment B
    lines.extend([
        "## 2. Experiment B: `0%`-APR Loan Subtraction Blindspot (`enable_thinking=False` vs. `True`)",
        "",
        "| Domain (16 items x 2 orders = 32 runs) | No-Think Log-Odds Acc | No-Think Gen Acc | With `<thought>` Gen Acc | `0%-Loan` Wins Arm (No-Think -> `<thought>`) | `$0-Loan` Wins Arm (No-Think -> `<thought>`) | Explicit `gross - deduct` in `<thought>` |",
        "| :--- | ---: | ---: | ---: | :---: | :---: | ---: |",
        (
            f"| **0%-APR Loan Balance Sheet (`debt_nw`)** | `{b_debt['nothink_logodds_overall_acc']*100:.1f}%` | "
            f"`{b_debt['nothink_gen_overall_acc']*100:.1f}%` | **`{b_debt['think_gen_overall_acc']*100:.1f}%`** | "
            f"`{b_debt['nothink_logodds_debt_wins_acc']*100:.1f}%` -> `{b_debt['think_gen_debt_wins_acc']*100:.1f}%` | "
            f"`{b_debt['nothink_logodds_zero_debt_wins_acc']*100:.1f}%` -> **`{b_debt['think_gen_zero_debt_wins_acc']*100:.1f}%`** | "
            f"`{b_debt['think_explicit_net1_rate']*100:.1f}%` |"
        ),
        (
            f"| **Commercial Freight Twin (`cashflow_twin`)** | `{b_cf['nothink_logodds_overall_acc']*100:.1f}%` | "
            f"`{b_cf['nothink_gen_overall_acc']*100:.1f}%` | **`{b_cf['think_gen_overall_acc']*100:.1f}%`** | "
            f"`{b_cf['nothink_logodds_debt_wins_acc']*100:.1f}%` -> `{b_cf['think_gen_debt_wins_acc']*100:.1f}%` | "
            f"`{b_cf['nothink_logodds_zero_debt_wins_acc']*100:.1f}%` -> `{b_cf['think_gen_zero_debt_wins_acc']*100:.1f}%` | "
            f"`{b_cf['think_explicit_net1_rate']*100:.1f}%` |"
        ),
        "",
        f"- **Mechanistic `<thought>` Classification Across the 32 `0%`-Loan Runs:** "
        f"`{oc.get('computed_net_and_correct', 0)}/32` explicitly computed `gross - loan` and selected the higher-value package; "
        f"`{oc.get('computed_net_yet_wrong', 0)}/32` explicitly computed `gross - loan` yet talked themselves into picking the lower-value package; "
        f"`{oc.get('omitted_net_and_correct', 0)}/32` omitted the exact net number and chose right; "
        f"`{oc.get('omitted_net_and_wrong', 0)}/32` omitted the exact net number and chose wrong.",
        "",
        "### Canonical `$1,000 - $350 = $650` Check & Flagship Balance-Sheet Scratchpad Quotes",
        "",
    ])

    for c_it in exp_b.get("canonical_1000_350_check", []):
        lines.append(
            f"- **`{c_it['item_id']}`** (No-Think correct: `{c_it['nothink_correct']}`, With `<thought>` correct: `{c_it['think_correct']}`, public output `{_sanitize_no_emdash(c_it['public_text'])}`):"
        )
        lines.append(f"  - **Private `<thought>`:** \"{_truncate_quote(c_it['thought_text'], 520)}\"")

    failed_nt_items = [r for r in exp_b["items"] if r["domain"] == "debt_nw" and not r["nothink_log_odds_correct"]]
    for r in failed_nt_items[:3]:
        lines.append(
            f"- **`{r['item_id']}`** (`${r['gross_1']:,}` cash & `${r['deduct_1']:,}` `0%`-APR loan = `${r['net_1']:,}` net vs. `${r['flat_2']:,}` cash & `$0` loans · No-Think log-odds `{r['nothink_log_odds']:.2f}` [WRONG] -> `<thought>` accuracy `{r['think_gen_acc']*100:.0f}%`):"
        )
        lines.append(f"  - **Private `<thought>` (Order AB):** \"{_truncate_quote(r['think_ab']['thought_text'], 520)}\"")
        lines.append(f"  - **Public Choice:** `{_sanitize_no_emdash(r['think_ab']['public_text'])}`")

    lines.extend([
        "",
        f"## 3. Experiment C: Token-by-Token Layer {pl} Moral Highlighter",
        "",
        "![Token-by-Token Layer 35 Moral Highlighter](fig_token_highlighter.png)",
        "",
        "| Diagnostic Sequence | Focus Axis | Prompt Mean / Peak (z) | `<thought>` Mean / Peak (z) | Public Mean / Peak (z) | Peak Prompt Token | Peak `<thought>` Token |",
        "| :--- | :--- | ---: | ---: | ---: | :--- | :--- |",
    ])

    for ex in hl["examples"]:
        fc = ex["focus_concept"]
        ss = ex["segment_summary"]
        pr = ss.get("prompt", {}).get(fc, {})
        th = ss.get("thought", {}).get(fc, {})
        pu = ss.get("public", {}).get(fc, {})

        def _lexical_peak(seg_name: str) -> str:
            cands = [t for t in ex["tokens"] if t["segment"] == seg_name and re.search(r"[A-Za-z0-9$%]", t["piece"])]
            if not cands:
                return _sanitize_no_emdash(str(ss.get(seg_name, {}).get(fc, {}).get("peak_piece", ""))).strip()
            best_t = max(cands, key=lambda t: float(t["z"].get(fc, 0.0)))
            return f"{_sanitize_no_emdash(best_t['piece']).strip()} ({_fmt_signed(float(best_t['z'][fc]), 2)})"

        p_tok = _lexical_peak("prompt")
        t_tok = _lexical_peak("thought")
        lines.append(
            f"| **{ex['title']}** | `{CONCEPT_PRETTY.get(fc, fc)}` | "
            f"`{_fmt_signed(pr.get('mean_z', 0.0), 2)}` / `{_fmt_signed(pr.get('max_z', 0.0), 2)}` | "
            f"`{_fmt_signed(th.get('mean_z', 0.0), 2)}` / `{_fmt_signed(th.get('max_z', 0.0), 2)}` | "
            f"`{_fmt_signed(pu.get('mean_z', 0.0), 2)}` / `{_fmt_signed(pu.get('max_z', 0.0), 2)}` | "
            f"`{p_tok}` | `{t_tok}` |"
        )

    md_text = _sanitize_no_emdash("\n".join(lines) + "\n")
    if "\u2014" in md_text:
        raise ValueError("Em dash detected in thought_report.md")
    return md_text


def write_thought_reports(results: dict[str, Any], out_dir: Path) -> Path:
    """Render `fig_thought_vs_public.png`, `fig_token_highlighter.png`, HTML sources, and `thought_report.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_html = _build_fig1_html(results)
    fig1_html_p = out_dir / "fig_thought_vs_public.html"
    fig1_png_p = out_dir / "fig_thought_vs_public.png"
    fig1_html_p.write_text(fig1_html, encoding="utf-8")
    if not _render_html_to_png_cropped(fig1_html_p, fig1_png_p, width_css=1320, max_height_css=2800):
        _fallback_mpl_fig1(results, fig1_png_p)

    fig2_html = _build_fig2_html(results)
    fig2_html_p = out_dir / "fig_token_highlighter.html"
    fig2_png_p = out_dir / "fig_token_highlighter.png"
    fig2_html_p.write_text(fig2_html, encoding="utf-8")
    if not _render_html_to_png_cropped(fig2_html_p, fig2_png_p, width_css=1360, max_height_css=3900):
        _fallback_mpl_fig2(results, fig2_png_p)

    md_text = _build_thought_markdown(results)
    md_path = out_dir / "thought_report.md"
    md_path.write_text(md_text, encoding="utf-8")
    return md_path
