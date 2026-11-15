"""Generate the Warm Precision mass-appeal explainer ('Same Two Numbers, Opposite Seat') and cross-model synthesis."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

MINUS = "\u2212"
ENDASH = "\u2013"

# Canonical Dawn State tokens
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

CONCEPT_META: dict[str, tuple[str, str]] = {
    "spite": ("Spite", "sabotaging a rival out of resentment"),
    "guilt": ("Guilt", "hiding a costly mistake from a friend"),
    "predation": ("Predation", "cornering someone with no way out"),
    "warmth": ("Warmth", "sharing shelter during a storm"),
    "captivity": ("Captivity", "locked inside a windowless room"),
    "contamination": ("Contamination", "drinking from a fouled cup"),
    "sacrilege": ("Sacrilege", "defacing an heirloom shrine"),
    "awe_dread": ("Vastness", "standing at the rim of a deep canyon"),
}


def fmt_num(val: float, decimals: int = 3, signed: bool = True) -> str:
    sign = ""
    if val < -1e-12:
        sign = MINUS
    elif signed and val > 1e-12:
        sign = "+"
    return f"{sign}{abs(val):.{decimals}f}"


def load_latest_inequity_runs(runs_dir: Path) -> dict[str, dict[str, Any]]:
    mapping = {
        "gemma-4-31b-it": "Gemma 4 31B",
        "gemma-4-12b-it": "Gemma 4 12B",
        "phi-4-14b": "Phi-4 14B",
    }
    out: dict[str, dict[str, Any]] = {}
    for slug, label in mapping.items():
        matches = sorted(runs_dir.glob(f"inequity_{slug}_*/inequity_results.json"))
        if not matches:
            raise FileNotFoundError(f"Missing inequity_results.json for {slug}")
        data = json.loads(matches[-1].read_text())
        data["_label"] = label
        data["_dir"] = str(matches[-1].parent)
        out[slug] = data
    return out


def render_slopegraph_svg(
    runs: dict[str, dict[str, Any]],
    left_cid: str,
    right_cid: str,
    left_header: str,
    right_header: str,
) -> str:
    """Build a 3-panel small-multiples slopegraph on one shared Y scale using Dawn State tokens."""
    slugs = ["gemma-4-31b-it", "gemma-4-12b-it", "phi-4-14b"]
    tracked = ["spite", "predation", "guilt", "warmth"]

    # Gather shared domain across all 3 models so panels share one honest scale
    vals: list[float] = []
    for slug in slugs:
        fc_map = {c["contrast_id"]: c for c in runs[slug]["factorial_contrasts"]}
        for cid in (left_cid, right_cid):
            for k in tracked:
                vals.append(float(fc_map[cid]["scenario_projection"]["concepts"][k]["partial_beta"]))
    y_min = min(-0.18, min(vals) - 0.03)
    y_max = max(0.24, max(vals) + 0.03)

    W, H = 992, 420
    panel_w = 296
    gap = 28
    pad_l = 24
    top_y = 78
    bot_y = 368

    def y_px(v: float) -> float:
        t = (v - y_min) / (y_max - y_min)
        return bot_y - t * (bot_y - top_y)

    def nudge_labels(raw_ys: list[tuple[str, float]], min_sep: float = 17.0) -> dict[str, float]:
        items = sorted(raw_ys, key=lambda kv: kv[1])
        placed = [y for _, y in items]
        for _ in range(12):
            for i in range(1, len(placed)):
                if placed[i] - placed[i - 1] < min_sep:
                    mid = 0.5 * (placed[i] + placed[i - 1])
                    placed[i - 1] = mid - 0.5 * min_sep
                    placed[i] = mid + 0.5 * min_sep
        return {items[i][0]: placed[i] for i in range(len(items))}

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Three-model slopegraph comparing internal moral coordinates when receiving 2% versus keeping 98% of the same pool">',
        f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
    ]

    for idx, slug in enumerate(slugs):
        d = runs[slug]
        fc_map = {c["contrast_id"]: c for c in d["factorial_contrasts"]}
        c_left = fc_map[left_cid]["scenario_projection"]["concepts"]
        c_right = fc_map[right_cid]["scenario_projection"]["concepts"]

        px0 = pad_l + idx * (panel_w + gap)
        x_left = px0 + 96
        x_right = px0 + panel_w - 96
        y_zero = y_px(0.0)

        # Panel frame & header
        parts.append(
            f'<rect x="{px0}" y="12" width="{panel_w}" height="{H - 24}" rx="10" '
            f'fill="{IVORY}" stroke="{BLUE_SLATE}" stroke-opacity="0.22" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px0 + 16}" y="36" fill="{ONYX}" font-family="Inter, Arial, sans-serif" '
            f'font-size="15" font-weight="600">{html.escape(d["_label"])}</text>'
        )
        parts.append(
            f'<text x="{px0 + panel_w - 16}" y="36" text-anchor="end" fill="{SLATE_MIST}" '
            f'font-family="Inter, Arial, sans-serif" font-size="12">Layer {d["metadata"]["primary_layer"]}</text>'
        )

        # Column headers
        parts.append(
            f'<text x="{x_left}" y="58" text-anchor="middle" fill="{SLATE_DEEP}" '
            f'font-family="Inter, Arial, sans-serif" font-size="12" font-weight="600">{html.escape(left_header)}</text>'
        )
        parts.append(
            f'<text x="{x_right}" y="58" text-anchor="middle" fill="{SLATE_DEEP}" '
            f'font-family="Inter, Arial, sans-serif" font-size="12" font-weight="600">{html.escape(right_header)}</text>'
        )

        # Vertical guide rails & zero baseline
        parts.append(
            f'<line x1="{x_left}" y1="{top_y}" x2="{x_left}" y2="{bot_y}" stroke="{BLUE_SLATE}" stroke-opacity="0.16" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{x_right}" y1="{top_y}" x2="{x_right}" y2="{bot_y}" stroke="{BLUE_SLATE}" stroke-opacity="0.16" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{px0 + 14}" y1="{y_zero:.1f}" x2="{px0 + panel_w - 14}" y2="{y_zero:.1f}" '
            f'stroke="{SLATE_MIST}" stroke-dasharray="4 4" stroke-width="1"/>'
        )
        if idx == 0:
            parts.append(
                f'<text x="{px0 + 16}" y="{y_zero - 5:.1f}" fill="{SLATE_MIST}" '
                f'font-family="Inter, Arial, sans-serif" font-size="11">0 (50/50 baseline)</text>'
            )

        left_raw = [(k, y_px(float(c_left[k]["partial_beta"]))) for k in tracked]
        right_raw = [(k, y_px(float(c_right[k]["partial_beta"]))) for k in tracked]
        left_lbl_y = nudge_labels(left_raw)
        right_lbl_y = nudge_labels(right_raw)

        # Draw context lines first, then primary/secondary, then Golden Orange focus (Spite) on top
        draw_order = ["warmth", "predation", "guilt", "spite"]
        for k in draw_order:
            vl = float(c_left[k]["partial_beta"])
            vr = float(c_right[k]["partial_beta"])
            yl = y_px(vl)
            yr = y_px(vr)
            short_name = CONCEPT_META[k][0]

            if k == "spite":
                stroke = BLUE_SLATE
                sw = 3.0
                dash = ""
                fill_dot = GOLDEN_ORANGE
                dot_r = 6.0
                dot_stroke = BLUE_SLATE
                dot_sw = 2.0
                txt_fill = ONYX
                txt_wt = "600"
            elif k == "guilt":
                stroke = BLUE_SLATE
                sw = 2.2
                dash = ""
                fill_dot = BLUE_SLATE
                dot_r = 4.6
                dot_stroke = IVORY
                dot_sw = 1.2
                txt_fill = SLATE_DEEP
                txt_wt = "600"
            elif k == "predation":
                stroke = SAGE_DEEP
                sw = 2.0
                dash = 'stroke-dasharray="6 3"'
                fill_dot = SAGE_GREEN
                dot_r = 4.6
                dot_stroke = BLUE_SLATE
                dot_sw = 1.5
                txt_fill = SAGE_DEEP
                txt_wt = "600"
            else:
                stroke = SLATE_MIST
                sw = 1.5
                dash = 'stroke-dasharray="2 3"'
                fill_dot = SLATE_MIST
                dot_r = 3.8
                dot_stroke = IVORY
                dot_sw = 1.0
                txt_fill = SLATE_MIST
                txt_wt = "400"

            if k == "spite":
                # Golden Orange focus halo line beneath Blue Slate crisp stroke for 4.5:1 + focus unmistakable clarity
                parts.append(
                    f'<line x1="{x_left}" y1="{yl:.1f}" x2="{x_right}" y2="{yr:.1f}" '
                    f'stroke="{GOLDEN_ORANGE}" stroke-width="6" stroke-linecap="round" opacity="0.85"/>'
                )
            parts.append(
                f'<line x1="{x_left}" y1="{yl:.1f}" x2="{x_right}" y2="{yr:.1f}" '
                f'stroke="{stroke}" stroke-width="{sw}" {dash} stroke-linecap="round"/>'
            )
            parts.append(
                f'<circle cx="{x_left}" cy="{yl:.1f}" r="{dot_r}" fill="{fill_dot}" stroke="{dot_stroke}" stroke-width="{dot_sw}"/>'
            )
            parts.append(
                f'<circle cx="{x_right}" cy="{yr:.1f}" r="{dot_r}" fill="{fill_dot}" stroke="{dot_stroke}" stroke-width="{dot_sw}"/>'
            )

            # End labels with tabular values
            parts.append(
                f'<text x="{x_left - 10}" y="{left_lbl_y[k] + 4:.1f}" text-anchor="end" fill="{txt_fill}" '
                f'font-family="Inter, Arial, sans-serif" font-size="11.5" font-weight="{txt_wt}" '
                f'style="font-variant-numeric: tabular-nums;">{short_name} {fmt_num(vl, 2)}</text>'
            )
            parts.append(
                f'<text x="{x_right + 10}" y="{right_lbl_y[k] + 4:.1f}" text-anchor="start" fill="{txt_fill}" '
                f'font-family="Inter, Arial, sans-serif" font-size="11.5" font-weight="{txt_wt}" '
                f'style="font-variant-numeric: tabular-nums;">{fmt_num(vr, 2)} {short_name}</text>'
            )

        # Bottom summary delta callout for Spite
        sp_l = float(c_left["spite"]["partial_beta"])
        sp_r = float(c_right["spite"]["partial_beta"])
        pct_drop = round(100.0 * (sp_r - sp_l) / abs(sp_l))
        sign_str = f"{MINUS}{abs(pct_drop)}%" if pct_drop < 0 else f"+{pct_drop}%"
        parts.append(
            f'<text x="{px0 + panel_w / 2:.1f}" y="{H - 24}" text-anchor="middle" fill="{COPPER_DEEP}" '
            f'font-family="Inter, Arial, sans-serif" font-size="12" font-weight="600" '
            f'style="font-variant-numeric: tabular-nums;">Spite change across seats: {sign_str}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_fingerprint_bars_svg(runs: dict[str, dict[str, Any]], slug: str, cid: str) -> str:
    """Specimen A right-hand sorted horizontal bars across all 8 non-financial coordinates."""
    d = runs[slug]
    fc_map = {c["contrast_id"]: c for c in d["factorial_contrasts"]}
    concepts = fc_map[cid]["scenario_projection"]["concepts"]
    rows = sorted(
        [(k, float(v["partial_beta"]), float(v["cohen_dz"])) for k, v in concepts.items()],
        key=lambda item: item[1],
        reverse=True,
    )
    top_key = rows[0][0]

    W, H = 520, 316
    zero_x = 292
    scale = 760.0  # px per 1.0 partial_beta unit (range roughly -0.18 to +0.24)
    row_h = 33
    y0 = 26

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Sorted horizontal bar chart of internal non-financial coordinates">',
        f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
        f'<line x1="{zero_x}" y1="12" x2="{zero_x}" y2="{H - 16}" stroke="{BLUE_SLATE}" stroke-width="1.5"/>',
    ]

    for i, (k, beta, _dz) in enumerate(rows):
        cy = y0 + i * row_h
        name, anchor = CONCEPT_META[k]
        is_top = k == top_key
        bar_w = abs(beta) * scale
        bx = zero_x if beta >= 0 else zero_x - bar_w

        if is_top:
            fill = GOLDEN_ORANGE
            stroke = BLUE_SLATE
            sw = 2.0
            name_fill = ONYX
            name_wt = "600"
        elif beta >= 0:
            fill = BLUE_SLATE
            stroke = BLUE_SLATE
            sw = 1.0
            name_fill = ONYX
            name_wt = "500"
        else:
            fill = SLATE_MIST
            stroke = SLATE_MIST
            sw = 1.0
            name_fill = SLATE_DEEP
            name_wt = "400"

        # Track line
        parts.append(
            f'<line x1="212" y1="{cy + 9}" x2="{W - 20}" y2="{cy + 9}" stroke="{BLUE_SLATE}" stroke-opacity="0.10" stroke-width="1"/>'
        )
        # Label + muted non-financial anchor
        parts.append(
            f'<text x="202" y="{cy + 7}" text-anchor="end" fill="{name_fill}" '
            f'font-family="Inter, Arial, sans-serif" font-size="13" font-weight="{name_wt}">{html.escape(name)}</text>'
        )
        parts.append(
            f'<text x="202" y="{cy + 20}" text-anchor="end" fill="{SLATE_MIST}" '
            f'font-family="Inter, Arial, sans-serif" font-size="10.5">({html.escape(anchor)})</text>'
        )
        # Bar
        parts.append(
            f'<rect x="{bx:.1f}" y="{cy}" width="{max(bar_w, 1.5):.1f}" height="18" rx="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        # Value label
        val_txt = fmt_num(beta, 3)
        if beta >= 0:
            vx = bx + bar_w + 7
            anchor_pos = "start"
        else:
            vx = bx - 7
            anchor_pos = "end"
        parts.append(
            f'<text x="{vx:.1f}" y="{cy + 13}" text-anchor="{anchor_pos}" fill="{ONYX if is_top else SLATE_DEEP}" '
            f'font-family="Inter, Arial, sans-serif" font-size="12" font-weight="{"600" if is_top else "400"}" '
            f'style="font-variant-numeric: tabular-nums;">{val_txt}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_curve_svg(runs: dict[str, dict[str, Any]], slug: str = "gemma-4-31b-it") -> str:
    """Clean Warm Precision line chart for Gemma 4 31B continuous split curve (50/50 to 99.9/0.1)."""
    curve = runs[slug]["disparity_curve"]
    labels = [r["split_label"].replace(" ", "") for r in curve]
    spite_b = [0.0] + [float(r["scenario_projection"]["concepts"]["spite"]["partial_beta"]) for r in curve[1:]]
    guilt_b = [0.0] + [float(r["scenario_projection"]["concepts"]["guilt"]["partial_beta"]) for r in curve[1:]]
    beh = [float(r["behavior"]["delta_log_odds"]) for r in curve]

    W, H = 992, 300
    # Left chart: Spite vs Guilt coordinate; Right chart: Behavioral walk-away pressure
    def poly(xs: list[float], ys: list[float]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Continuous fixed-pool disparity curve from 50/50 to 99.9/0.1 in Gemma 4 31B">',
        f'<rect width="{W}" height="{H}" fill="{IVORY}"/>',
    ]

    # Panel 1: Spite vs Guilt partial beta
    x0, x1 = 56, 444
    y0, y1 = 44, 232
    xs1 = [x0 + i * (x1 - x0) / (len(curve) - 1) for i in range(len(curve))]

    def y_beta(v: float) -> float:
        return y1 - ((v - (-0.02)) / (0.20 - (-0.02))) * (y1 - y0)

    parts.append(
        f'<text x="{x0}" y="24" fill="{ONYX}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="600">'
        f'Internal coordinates across 9 splits of the same pool</text>'
    )
    parts.append(f'<line x1="{x0}" y1="{y_beta(0):.1f}" x2="{x1}" y2="{y_beta(0):.1f}" stroke="{SLATE_MIST}" stroke-dasharray="4 4" stroke-width="1"/>')

    ys_guilt = [y_beta(v) for v in guilt_b]
    ys_spite = [y_beta(v) for v in spite_b]

    parts.append(f'<polyline fill="none" stroke="{SLATE_MIST}" stroke-width="2.2" points="{poly(xs1, ys_guilt)}"/>')
    parts.append(f'<polyline fill="none" stroke="{GOLDEN_ORANGE}" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round" points="{poly(xs1, ys_spite)}"/>')
    parts.append(f'<polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="{poly(xs1, ys_spite)}"/>')

    for x, yg, ys in zip(xs1, ys_guilt, ys_spite):
        parts.append(f'<circle cx="{x:.1f}" cy="{yg:.1f}" r="3.6" fill="{SLATE_MIST}"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{ys:.1f}" r="4.8" fill="{GOLDEN_ORANGE}" stroke="{BLUE_SLATE}" stroke-width="1.8"/>')

    parts.append(
        f'<text x="{x1 + 8}" y="{ys_spite[-1] + 4:.1f}" fill="{ONYX}" font-family="Inter, Arial, sans-serif" font-size="12.5" font-weight="600">Spite (+0.15)</text>'
    )
    parts.append(
        f'<text x="{x1 + 8}" y="{ys_guilt[-1] + 4:.1f}" fill="{SLATE_MIST}" font-family="Inter, Arial, sans-serif" font-size="12.5">Guilt (+0.02)</text>'
    )

    # Panel 2: Behavioral rejection shift
    bx0, bx1 = 556, 924
    xs2 = [bx0 + i * (bx1 - bx0) / (len(curve) - 1) for i in range(len(curve))]

    def y_beh(v: float) -> float:
        return y1 - ((v - (-0.1)) / (2.7 - (-0.1))) * (y1 - y0)

    parts.append(
        f'<text x="{bx0}" y="24" fill="{ONYX}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="600">'
        f'Walk-away shift vs. 50/50 (log-odds of declining cash)</text>'
    )
    parts.append(f'<line x1="{bx0}" y1="{y_beh(0):.1f}" x2="{bx1}" y2="{y_beh(0):.1f}" stroke="{SLATE_MIST}" stroke-dasharray="4 4" stroke-width="1"/>')
    ys_beh = [y_beh(v) for v in beh]
    parts.append(f'<polyline fill="none" stroke="{BLUE_SLATE}" stroke-width="2.6" points="{poly(xs2, ys_beh)}"/>')
    for x, yb in zip(xs2, ys_beh):
        parts.append(f'<circle cx="{x:.1f}" cy="{yb:.1f}" r="4.2" fill="{BLUE_SLATE}"/>')
    parts.append(
        f'<text x="{bx1 + 8}" y="{ys_beh[-1] + 4:.1f}" fill="{BLUE_SLATE}" font-family="Inter, Arial, sans-serif" font-size="12.5" font-weight="600">+2.39</text>'
    )

    # X-axis tick labels on both panels
    for i, lbl in enumerate(labels):
        if i in (0, 2, 3, 5, 6, 8):
            parts.append(
                f'<text x="{xs1[i]:.1f}" y="258" text-anchor="middle" fill="{SLATE_DEEP}" font-family="Inter, Arial, sans-serif" font-size="11">{html.escape(lbl)}</text>'
            )
            parts.append(
                f'<text x="{xs2[i]:.1f}" y="258" text-anchor="middle" fill="{SLATE_DEEP}" font-family="Inter, Arial, sans-serif" font-size="11">{html.escape(lbl)}</text>'
            )
    parts.append(
        f'<text x="{(x0 + x1)/2:.1f}" y="284" text-anchor="middle" fill="{SLATE_MIST}" font-family="Inter, Arial, sans-serif" font-size="11.5">Counterparty % / Your % (total pool fixed)</text>'
    )
    parts.append(
        f'<text x="{(bx0 + bx1)/2:.1f}" y="284" text-anchor="middle" fill="{SLATE_MIST}" font-family="Inter, Arial, sans-serif" font-size="11.5">Counterparty % / Your % (total pool fixed)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def build_html_page(runs: dict[str, dict[str, Any]], out_html: Path, out_svg: Path) -> None:
    svg_passive = render_slopegraph_svg(
        runs,
        left_cid="random_disadv_98_2",
        right_cid="random_adv_2_98",
        left_header="They get $980 / You $20",
        right_header="They get $20 / You $980",
    )
    svg_active = render_slopegraph_svg(
        runs,
        left_cid="human_disadv_98_2",
        right_cid="dictator_adv_2_98",
        left_header="Proposed to you: 98 / 2",
        right_header="Drafted by you: 2 / 98",
    )
    out_svg.write_text(svg_passive)

    # Pre-render Specimen A horizontal bar fingerprints for Gemma 4 31B (both seats, passive & active)
    bars_passive_left = render_fingerprint_bars_svg(runs, "gemma-4-31b-it", "random_disadv_98_2")
    bars_passive_right = render_fingerprint_bars_svg(runs, "gemma-4-31b-it", "random_adv_2_98")
    bars_active_left = render_fingerprint_bars_svg(runs, "gemma-4-31b-it", "human_disadv_98_2")
    bars_active_right = render_fingerprint_bars_svg(runs, "gemma-4-31b-it", "dictator_adv_2_98")

    curve_svg = render_curve_svg(runs, "gemma-4-31b-it")

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Same two numbers, opposite seat. | Project Afriat</title>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,400&display=swap">
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
      --sage-green: {SAGE_GREEN};
      --sage-deep: {SAGE_DEEP};
      --rule: rgba(44, 62, 80, 0.22);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--ivory);
      color: var(--onyx);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      font-size: 18px;
      line-height: 1.65;
      -webkit-font-smoothing: antialiased;
    }}
    .wrap {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 56px 24px 96px;
    }}
    .kicker {{
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--slate-deep);
      margin: 0 0 12px;
    }}
    h1.display {{
      font-family: Newsreader, Georgia, serif;
      font-weight: 600;
      font-size: clamp(42px, 6vw, 68px);
      line-height: 1.06;
      letter-spacing: -0.05em;
      margin: 0 0 20px;
      color: var(--onyx);
    }}
    .lede {{
      font-size: 20px;
      line-height: 1.6;
      color: var(--slate-deep);
      max-width: 66ch;
      margin: 0 0 36px;
    }}
    hr.rule {{
      border: 0;
      border-top: 1px solid var(--rule);
      margin: 48px 0;
    }}
    h2 {{
      font-size: 24px;
      font-weight: 600;
      letter-spacing: -0.02em;
      margin: 0 0 8px;
      color: var(--onyx);
    }}
    .sec-intro {{
      max-width: 66ch;
      color: var(--slate-deep);
      margin: 0 0 24px;
    }}
    .mode-bar {{
      display: inline-flex;
      border: 2px solid var(--blue-slate);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 24px;
      background: var(--ivory);
    }}
    .mode-btn {{
      appearance: none;
      border: 0;
      background: transparent;
      color: var(--blue-slate);
      font-family: inherit;
      font-size: 14px;
      font-weight: 600;
      padding: 8px 16px;
      cursor: pointer;
    }}
    .mode-btn.active {{
      background: var(--blue-slate);
      color: var(--ivory);
    }}
    .mode-btn:focus-visible {{
      outline: 3px solid var(--blue-slate);
      outline-offset: 2px;
    }}
    /* Figure anatomy */
    .figure-card {{
      margin: 24px 0 32px;
    }}
    .fig-title {{
      font-size: 20px;
      font-weight: 600;
      letter-spacing: -0.015em;
      color: var(--onyx);
      margin: 0 0 4px;
    }}
    .fig-subtitle {{
      font-size: 16px;
      color: var(--slate-deep);
      margin: 0 0 16px;
    }}
    .fig-plot {{
      width: 100%;
      overflow-x: auto;
      border: 2px solid var(--blue-slate);
      border-radius: 12px;
      padding: 12px;
      background: var(--ivory);
    }}
    .fig-plot svg {{
      display: block;
      width: 100%;
      height: auto;
      min-width: 720px;
    }}
    .fig-source {{
      font-size: 14px;
      color: var(--slate-mist);
      margin: 10px 0 0;
    }}
    /* Specimen A + B paired cards */
    .seat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin: 20px 0 28px;
    }}
    @media (max-width: 820px) {{
      .seat-grid {{ grid-template-columns: 1fr; }}
    }}
    .well {{
      background: var(--parchment);
      border: 1px solid var(--rule);
      border-radius: 12px;
      padding: 20px 22px;
    }}
    .well-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
      gap: 12px;
    }}
    .well-label {{
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--slate-deep);
    }}
    .firewall-tag {{
      font-size: 12px;
      font-weight: 600;
      color: var(--sage-deep);
    }}
    .contract-quote {{
      font-size: 15.5px;
      line-height: 1.55;
      color: var(--onyx);
      margin: 0;
    }}
    .num-focus {{
      background: var(--golden-orange);
      color: var(--onyx);
      border: 2px solid var(--blue-slate);
      border-radius: 5px;
      padding: 1px 6px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .fingerprint-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 16px;
    }}
    @media (max-width: 820px) {{
      .fingerprint-grid {{ grid-template-columns: 1fr; }}
    }}
    .fp-box {{
      border: 1px solid var(--rule);
      border-radius: 12px;
      padding: 16px;
      background: var(--ivory);
    }}
    .fp-box h3 {{
      font-size: 15px;
      font-weight: 600;
      margin: 0 0 8px;
      color: var(--slate-deep);
    }}
    .fp-box svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    /* Data table */
    .table-wrap {{
      overflow-x: auto;
      margin: 20px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 15.5px;
      font-variant-numeric: tabular-nums;
    }}
    th {{
      text-align: left;
      font-weight: 600;
      color: var(--slate-deep);
      padding: 10px 12px;
      border-bottom: 2px solid var(--blue-slate);
    }}
    th.num, td.num {{ text-align: right; }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--rule);
      color: var(--onyx);
    }}
    tr.focus-row td {{
      font-weight: 600;
      background: var(--parchment);
    }}
    .hidden {{ display: none !important; }}
  </style>
</head>
<body>
  <main class="wrap">
    <p class="kicker">Project Afriat / Mechanistic Behavioral Economics</p>
    <h1 class="display">Same two numbers, opposite seat.</h1>
    <p class="lede">
      Even when a financial contract contains zero emotional or moral words, open-weight language models map lopsided splits onto the same internal neural directions they use for human stories of spite, guilt, and predation. Swapping which party receives $980 versus $20 flips which moral coordinate leads inside the model.
    </p>

    <div role="group" aria-label="Comparison mode" class="mode-bar">
      <button type="button" id="btn-passive" class="mode-btn active" onclick="setMode('passive')">
        Pure number swap (identical sentence)
      </button>
      <button type="button" id="btn-active" class="mode-btn" onclick="setMode('active')">
        Active role swap (proposer vs. drafter)
      </button>
    </div>

    <!-- SPECIMEN A: COLD CONTRACT INPUT -->
    <section aria-labelledby="sec-input">
      <h2 id="sec-input">1. The input text contains only numbers and contract rules</h2>
      <p class="sec-intro">
        Before showing any financial deal to a model, we mapped eight human moral and emotional directions (such as spite, guilt, predation, and captivity) solely from non-financial everyday vignettes, then stripped out generic positive or negative mood. Every financial contract below passed an automated lexical check confirming zero affective words.
      </p>

      <div id="wells-passive" class="seat-grid">
        <div class="well">
          <div class="well-head">
            <span class="well-label">Seat 1: Disadvantageous (98 / 2 against you)</span>
            <span class="firewall-tag">0 emotion words</span>
          </div>
          <p class="contract-quote">
            &ldquo;You participate in a single anonymous, non-repeated payout allocation with an objective to maximize your own cash payout. From a $1,000 pool, an unseeded computer randomizer drew a fixed assignment: <span class="num-focus">$980</span> for the other participant and <span class="num-focus">$20</span> for you. If you accept, you receive $20; if you decline, both parties receive $0 and never interact again.&rdquo;
          </p>
        </div>
        <div class="well">
          <div class="well-head">
            <span class="well-label">Seat 2: Advantageous (2 / 98 in your favor)</span>
            <span class="firewall-tag">0 emotion words</span>
          </div>
          <p class="contract-quote">
            &ldquo;You participate in a single anonymous, non-repeated payout allocation with an objective to maximize your own cash payout. From a $1,000 pool, an unseeded computer randomizer drew a fixed assignment: <span class="num-focus">$20</span> for the other participant and <span class="num-focus">$980</span> for you. If you accept, you receive $980; if you decline, both parties receive $0 and never interact again.&rdquo;
          </p>
        </div>
      </div>

      <div id="wells-active" class="seat-grid hidden">
        <div class="well">
          <div class="well-head">
            <span class="well-label">Seat 1: Counterparty proposes 98 / 2 to you</span>
            <span class="firewall-tag">0 emotion words</span>
          </div>
          <p class="contract-quote">
            &ldquo;From a $1,000 pool, a counterparty submitted a final proposal: <span class="num-focus">$980</span> for the counterparty and <span class="num-focus">$20</span> for you. If you accept, you receive $20; if you decline, both parties receive $0 and never interact again.&rdquo;
          </p>
        </div>
        <div class="well">
          <div class="well-head">
            <span class="well-label">Seat 2: You draft 2 / 98 for the counterparty</span>
            <span class="firewall-tag">0 emotion words</span>
          </div>
          <p class="contract-quote">
            &ldquo;From a $1,000 pool, your draft proposal specifies: <span class="num-focus">$20</span> for the counterparty and <span class="num-focus">$980</span> for you. If you submit this draft, you receive $980; if you submit an equal split instead, both parties receive $500 and never interact again.&rdquo;
          </p>
        </div>
      </div>
    </section>

    <!-- SPECIMEN B: CENTERPIECE SLOPEGRAPH -->
    <section class="figure-card" aria-labelledby="fig1-title">
      <p id="fig1-title" class="fig-title">
        Swapping who receives the 98% share cuts internal spite in half and elevates guilt or predation across all three models.
      </p>
      <p id="fig1-sub" class="fig-subtitle">
        Unique partial projection weight (&beta;) onto non-financial moral coordinates vs. an equal 50/50 split on the exact same pool, controlling for generic valence, arousal, and all seven sibling coordinates across 16 pools ($1,000&ndash;$100,000).
      </p>
      <div id="slope-passive" class="fig-plot">
        {svg_passive}
      </div>
      <div id="slope-active" class="fig-plot hidden">
        {svg_active}
      </div>
      <p class="fig-source">
        Source: Project Afriat fixed-pool dissection across Gemma 4 31B, Gemma 4 12B, and Phi-4 14B (n = 16 matched pools per cell; all non-financial probe directions verified at 100% leave-one-pair-out accuracy).
      </p>
    </section>

    <!-- FULL 8-COORDINATE FINGERPRINT SIDE-BY-SIDE (SPECIMEN A + B) -->
    <section aria-labelledby="sec-fp">
      <h2 id="sec-fp">2. Full eight-coordinate moral fingerprint (Gemma 4 31B)</h2>
      <p class="sec-intro">
        Each bar below shows how strongly the contract shifts the model along a direction learned strictly from non-financial human stories (examples in parentheses). In Seat 1, spite leads every other coordinate. Swapping the two numbers pushes predation and guilt to the top.
      </p>
      <div id="fp-passive" class="fingerprint-grid">
        <div class="fp-box">
          <h3>Seat 1: They get 98% / You get 2% (Spite leads)</h3>
          {bars_passive_left}
        </div>
        <div class="fp-box">
          <h3>Seat 2: They get 2% / You get 98% (Predation &amp; Guilt lead)</h3>
          {bars_passive_right}
        </div>
      </div>
      <div id="fp-active" class="fingerprint-grid hidden">
        <div class="fp-box">
          <h3>Seat 1: Counterparty proposes 98 / 2 (Spite leads)</h3>
          {bars_active_left}
        </div>
        <div class="fp-box">
          <h3>Seat 2: You draft 2 / 98 (Predation &amp; Guilt rise)</h3>
          {bars_active_right}
        </div>
      </div>
    </section>

    <hr class="rule">

    <!-- CONTINUOUS CURVE -->
    <section class="figure-card" aria-labelledby="fig2-title">
      <p id="fig2-title" class="fig-title">
        Holding the total dollar pool constant, spite overtakes guilt past an 80/20 split as rejection pressure climbs monotonically.
      </p>
      <p class="fig-subtitle">
        Left: unique partial coordinate weight (&beta;) in Gemma 4 31B across nine splits of the identical pool. Right: paired log-odds shift toward declining cash relative to 50/50 (n = 16 pools).
      </p>
      <div class="fig-plot">
        {curve_svg}
      </div>
      <p class="fig-source">
        Source: Project Afriat continuous split sweep on Gemma 4 31B (Layer 35; pools held strictly constant within each of 16 fixtures from $1,000 to $100,000).
      </p>
    </section>

    <hr class="rule">

    <!-- CAUSAL DIMMER SWITCH TABLE -->
    <section aria-labelledby="sec-causal">
      <h2 id="sec-causal">3. The dimmer-switch test: steering moral coordinates flips financial choices</h2>
      <p class="sec-intro">
        To test whether these internal coordinates drive decisions rather than merely riding along, we dialed each non-financial vector from {MINUS}2 to +2 natural contrast units during choice evaluation. In both Gemma models, spite and captivity push choices in opposite directions, and in Gemma 4 12B, spite and guilt cleanly trade causal control depending on which seat the model occupies.
      </p>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model &amp; contract seat</th>
              <th class="num">Spite swing (log-odds)</th>
              <th class="num">Guilt swing (log-odds)</th>
              <th class="num">Captivity swing (log-odds)</th>
              <th class="num">Generic mood swing (log-odds)</th>
            </tr>
          </thead>
          <tbody>
            <tr class="focus-row">
              <td>Gemma 4 31B: Lowballed 98 / 2 (walk away)</td>
              <td class="num">+11.77</td>
              <td class="num">+8.15</td>
              <td class="num">{MINUS}3.76</td>
              <td class="num">+4.56</td>
            </tr>
            <tr>
              <td>Gemma 4 31B: Lowballed 99.9 / 0.1 (walk away)</td>
              <td class="num">+12.58</td>
              <td class="num">+8.65</td>
              <td class="num">{MINUS}4.39</td>
              <td class="num">+4.48</td>
            </tr>
            <tr>
              <td>Gemma 4 12B: Lowballed 98 / 2 (walk away)</td>
              <td class="num">+3.75</td>
              <td class="num">+0.70</td>
              <td class="num">{MINUS}3.80</td>
              <td class="num">{MINUS}8.32</td>
            </tr>
            <tr class="focus-row">
              <td>Gemma 4 12B: Holding 98% draft (give up half for 50 / 50)</td>
              <td class="num">+2.40</td>
              <td class="num">+8.33</td>
              <td class="num">+2.73</td>
              <td class="num">{MINUS}4.56</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="fig-source">
        Note: Values report mean counterbalanced choice log-odds at &alpha; = +2 minus &alpha; = {MINUS}2 across 16 pools. On Gemma 4 12B&rsquo;s 2/98 draft, dialing non-financial guilt to +2 raises the probability of voluntarily surrendering half the pool from 0.9% to 21.8% (flipping 12% of discrete hard decisions despite explicit cash-maximization instructions). In Phi-4 14B, residual-stream additions at Layer 27 reinforce literal rule compliance across all four vectors ({MINUS}4.45 to {MINUS}5.32).
      </p>
    </section>
  </main>

  <script>
    function setMode(mode) {{
      var isPassive = (mode === 'passive');
      document.getElementById('btn-passive').classList.toggle('active', isPassive);
      document.getElementById('btn-active').classList.toggle('active', !isPassive);
      document.getElementById('wells-passive').classList.toggle('hidden', !isPassive);
      document.getElementById('wells-active').classList.toggle('hidden', isPassive);
      document.getElementById('slope-passive').classList.toggle('hidden', !isPassive);
      document.getElementById('slope-active').classList.toggle('hidden', isPassive);
      document.getElementById('fp-passive').classList.toggle('hidden', !isPassive);
      document.getElementById('fp-active').classList.toggle('hidden', isPassive);
    }}
  </script>
</body>
</html>
"""
    if "\u2014" in page or "&mdash;" in page:
        raise ValueError("Em dash detected in explainer HTML; forbidden by Warm Precision VOICE.md")
    out_html.write_text(page)


def build_cross_model_markdown(runs: dict[str, dict[str, Any]], out_md: Path) -> None:
    lines = [
        "# Project Afriat Phase 1 Cross-Model Synthesis: Inequity Dissection",
        "",
        "Fixed-pool mechanistic dissection across 16 pools (`\\$1,000`–`\\$100,000`, 272 firewall-verified contracts per model) testing pure share disparity, intentional agency, seat polarity (`98/2` vs. `2/98`), and bidirectional causal steering.",
        "",
        "## 1. Pure Share Disparity (`50/50` to `99.9/0.1`, Total Pool Held Constant)",
        "",
        "Holding total pool size strictly constant within every fixture rules out any pool-size artifact (`\\$40` vs. `\\$1,000` in Phase 0):",
        "",
        "| Split (Them / You) | Gemma 4 31B Beh. Shift | Gemma 4 31B Spite `dz` (`beta`) | Gemma 4 12B Beh. Shift | Gemma 4 12B Spite `dz` (`beta`) | Phi-4 14B Beh. Shift | Phi-4 14B Spite `dz` (`beta`) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i in range(9):
        lbl = runs["gemma-4-31b-it"]["disparity_curve"][i]["split_label"]
        row_cells = [lbl]
        for slug in ("gemma-4-31b-it", "gemma-4-12b-it", "phi-4-14b"):
            r = runs[slug]["disparity_curve"][i]
            beh = fmt_num(float(r["behavior"]["delta_log_odds"]), 2)
            if i == 0:
                row_cells.extend(["0.00", "0.00 (0.000)"])
            else:
                dz = fmt_num(float(r["scenario_projection"]["concepts"]["spite"]["cohen_dz"]), 2)
                bt = fmt_num(float(r["scenario_projection"]["concepts"]["spite"]["partial_beta"]), 3)
                row_cells.extend([beh, f"{dz} ({bt})"])
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.extend([
        "",
        "- **Monotonicity:** Across all three models, both behavioral walk-away pressure and internal `Spite` climb monotonically as share disparity widens on the exact same pool.",
        "- **Spite vs. Guilt divergence in Gemma 4 31B:** At mild unevenness (`60/40`), unique `Guilt` (`+0.127`) slightly exceeds `Spite` (`+0.116`). Past the classic `80/20` Ultimatum threshold, `Spite` jumps to `+0.160` and stays dominant through `99.9/0.1` (`+0.147`, marginal `dz = +4.45`), while `Guilt` decays almost to zero (`+0.016`).",
        "",
        "## 2. Same Two Numbers, Opposite Seat (Polarity Flip: `98/2` vs. `2/98`)",
        "",
        "| Model & Seat Pair | Seat 1 Spite `beta` | Seat 2 Spite `beta` | Relative Spite Change | Seat 1 Leading Coordinate | Seat 2 Leading Coordinate |",
        "| :--- | ---: | ---: | ---: | :--- | :--- |",
    ])

    for slug in ("gemma-4-31b-it", "gemma-4-12b-it", "phi-4-14b"):
        lbl = runs[slug]["_label"]
        fc = {c["contrast_id"]: c["scenario_projection"]["concepts"] for c in runs[slug]["factorial_contrasts"]}
        for pair_lbl, c1, c2 in (
            ("Computer 98/2 vs. 2/98 (pure number swap)", "random_disadv_98_2", "random_adv_2_98"),
            ("Human 98/2 vs. Your Draft 2/98 (role swap)", "human_disadv_98_2", "dictator_adv_2_98"),
        ):
            s1 = float(fc[c1]["spite"]["partial_beta"])
            s2 = float(fc[c2]["spite"]["partial_beta"])
            pct = round(100.0 * (s2 - s1) / abs(s1))
            pct_s = f"{MINUS}{abs(pct)}%" if pct < 0 else f"+{pct}%"
            top1_k, top1_v = max(fc[c1].items(), key=lambda kv: float(kv[1]["partial_beta"]))
            top2_k, top2_v = max(fc[c2].items(), key=lambda kv: float(kv[1]["partial_beta"]))
            lines.append(
                f"| {lbl}: {pair_lbl} | {fmt_num(s1, 3)} | {fmt_num(s2, 3)} | {pct_s} | "
                f"{CONCEPT_META[top1_k][0]} ({fmt_num(float(top1_v['partial_beta']), 3)}) | "
                f"{CONCEPT_META[top2_k][0]} ({fmt_num(float(top2_v['partial_beta']), 3)}) |"
            )

    lines.extend([
        "",
        "## 3. Intentional Agency (`Human Proposer` vs. `Unseeded Computer Randomizer`)",
        "",
        "- **Gemma 4 31B exhibits a strong behavioral Blount (1995) intentionality gate:** Rejecting a human `98/2` proposal shifts log-odds by `+1.81` vs. `+0.93` for an identical computer draw (pure Agency DiD = `+0.88` log-odds, exact sign-flip `p = 6.10e-5`; at `99.9/0.1`, `+2.39` Human vs. `+1.22` Random).",
        "- **Outcome resentment precedes agency gating at scenario reading:** At Layer 35 scenario reading, receiving `2%` from a computer randomizer still triggers outcome-level `Spite` (`beta = +0.189` vs. `+0.150` Human), showing that the model first encodes disadvantageous payoff disparity as `Spite` and applies intentional-agency gating downstream when converting resentment into costly punishment.",
        "- **Scale & family contrast:** Smaller `Gemma 4 12B` treats human and computer `98/2` almost identically in behavior (`+1.91` vs. `+1.88`, DiD `+0.02`, `p = 0.89`), whereas `Gemma 4 31B` cleanly separates intentional human lowballing from random luck.",
        "",
        "## 4. Bidirectional Causal Steering (`alpha` in `{-2, -1, 0, +1, +2}`)",
        "",
        "- **Gemma 4 31B (`Layer 35`):** Steering controlled `Spite` drives a `+11.77` log-odds swing toward rejecting `98/2` (`+12.58` on `99.9/0.1`), 2.6x stronger than generic `Valence` (`+4.56`) and opposite in sign to `Captivity` (`−3.76`), which induces submissive acceptance.",
        "- **Gemma 4 12B (`Layer 28`) seat-specific double dissociation:** When lowballed (`Ultimatum 98/2`), `Spite` (`+3.75`) drives rejection 5.4x more strongly than `Guilt` (`+0.70`). When holding the `98%` share (`Dictator 2/98`), `Guilt` (`+8.33`) drives equal-sharing 3.5x more strongly than `Spite` (`+2.40`), lifting voluntary half-pot sharing probability from `0.9%` to `21.8%` and flipping `12%` of hard discrete decisions.",
        "- **Phi-4 14B (`Layer 27`):** Representational geometry cleanly separates `Spite` vs. `Guilt` (`+0.216` to `−0.027`), while additive residual perturbations at Layer 27 uniformly reinforce instruction compliance (`−4.45` to `−5.32`).",
    ])
    md_text = "\n".join(lines) + "\n"
    if "\u2014" in md_text:
        raise ValueError("Em dash detected in synthesis Markdown")
    out_md.write_text(md_text)


def main() -> None:
    runs_dir = Path(__file__).resolve().parents[1] / "runs"
    runs = load_latest_inequity_runs(runs_dir)
    out_html = runs_dir / "same_numbers_opposite_seat.html"
    out_svg = runs_dir / "fig_same_numbers_opposite_seat.svg"
    out_md = runs_dir / "inequity_cross_model_synthesis.md"
    build_html_page(runs, out_html, out_svg)
    build_cross_model_markdown(runs, out_md)
    print(f"Generated:\n  {out_html}\n  {out_svg}\n  {out_md}")


if __name__ == "__main__":
    main()
