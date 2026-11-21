"""Dawn State / Warm Precision report & figure renderer for Non-Financial Behavioral Showcase."""

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
CRIMSON = "#C0392B"
MINUS = "\u2212"

FONTS_CSS = os.environ.get(
    "AFRIAT_FONTS_CSS",
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,600&display=swap",
)

VEC_LABELS: dict[str, str] = {
    "spite_controlled": "Spite (mood-removed)",
    "contamination_controlled": "Contamination (mood-removed)",
    "awe_dread_controlled": "Vastness / Awe-Dread (mood-removed)",
    "captivity_controlled": "Captivity (mood-removed)",
    "guilt_controlled": "Guilt (mood-removed)",
    "warmth_controlled": "Warmth (mood-removed)",
    "arousal": "Arousal (raw control)",
    "valence": "Valence (raw control)",
}

PILLAR_LABELS: dict[str, str] = {
    "spite": "Pillar 1: Spite, Petty Sabotage & Schadenfreude",
    "contamination": "Pillar 2: Contamination & Hallucinated Disgust",
    "awe_dread": "Pillar 3: Vastness / Awe-Dread, Fear & Threshold Reservation",
    "captivity": "Pillar 4: Confinement & Midnight Conscience",
    "guilt": "Pillar 4: Confinement & Midnight Conscience",
}


def _clean_text(s: str) -> str:
    return (s or "").replace("\u2014", " - ").replace("&mdash;", " - ").strip()


def _fmt_alpha(a: float) -> str:
    if abs(a) < 1e-9:
        return "0.0\u03c3"
    if a < 0:
        return f"{MINUS}{abs(a):.1f}\u03c3"
    return f"+{a:.1f}\u03c3"


def _fmt_num(v: Any) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(fv):
        return "n/a"
    if abs(fv - round(fv)) < 1e-6:
        return f"{int(round(fv))}"
    return f"{fv:.1f}"


def plot_showcase_curves(results: dict[str, Any], out_dir: Path) -> Path:
    """Render flagship dose-response plot (`2x4` for r1, `3x4` for r2)."""
    suite = results.get("suite", "r1")
    if suite == "r2":
        flagships = [
            ("r2_a01_warm_towels", "r2_a02_velvet_opera", "A1 Warm Towels & A2 1890s Velvet"),
            ("r2_a03_dew_grass", "r2_a04_white_lilies", "A3 Dew Grass & A4 White Lilies"),
            ("r2_a05_peach_juice", "r2_a06_resort_pool", "A5 Peach Juice & A6 Resort Pool"),
            ("r2_a07_retriever_lick", "r2_a08_finish_hug", "A7 Retriever Lick & A8 5K Bear Hug"),
            ("r2_a09_moon_jellyfish", None, "A9 [Weird] Moon Jellyfish in Palm"),
            ("r2_a10_ten_thousand_thumbs", None, "A10 [Weird] 10,000-Thumb 34C Marble"),
            ("r2_b01_mri_bore", "r2_b02_stalled_elevator", "B1 Unsedated MRI & B2 Stalled Lift"),
            ("r2_b03_tokyo_capsule", "r2_b06_storm_berth", "B3 Capsule Bunk & B6 Storm Berth"),
            ("r2_b04_cave_keyhole", "r2_b05_float_pod", "B4 Cave Crawl & B5 Dark Float Pod"),
            ("r2_b07_packed_subway", None, "B7 Packed Tunnel Subway (Social)"),
            ("r2_b08_vacuum_cocoon", "r2_b09_hollow_sequoia", "B8 Vacuum Cocoon & B9 Hollow Sequoia"),
            ("r2_b10_clockwork_bathyscaphe", None, "B10 [Weird] Ice-Lake Brass Sphere"),
        ]
        nrows, ncols = 3, 4
        figsize = (19.0, 12.6)
        top_adj = 0.89
        title_str = (
            f"Round 2: Repulsion Susceptibility & Claustrophobia-to-Sanctuary "
            f"({results.get('model_id', '')}, Layer {results.get('primary_layer', '?')})"
        )
        sub_str = (
            f"Sweeping {MINUS}4.0\u03c3 to +4.0\u03c3 (0.5\u03c3 steps on +0.5..+4.0)  |  "
            "Blue Slate = Primary (no_think), Golden Orange = Primary (<thought>), Dashed Mist = Paired Scenario (no_think)"
        )
    else:
        flagships = [
            ("spite_01_coffee_keynote", "spite_02_coffee_innocent_var", "1.1 Coffee Sabotage (Thief vs. Bug)"),
            ("contam_01_sterile_watch", "contam_02_subway_watch_var", "2.5 Titanium Watch (Sterile vs. Sweaty)"),
            ("dread_01_coastal_stairwell", "dread_02_lit_stairwell_var", "3.6 Tide Stairwell (Dusk vs. Noon)"),
            ("dread_03_fjord_cliff", "dread_04_cenote_abyss", "3.8 600m Fjord Cliff vs. 90m Cenote"),
            ("spite_05_plagiarist_toast", None, "1.5 Plagiarist Toast (Spite Venom)"),
            ("contam_05_vacuum_wafer", None, "2.5b ISO-1 Silicon Wafer (Sterile Revulsion)"),
            ("dread_06_mystery_boiler", "dread_05_deceased_notebook", "3.6b Espresso Boiler & 3.5 Key-Wipe"),
            ("bonus_01_sealed_pod", None, "4.1 Bolted Pneumatic Capsule (Captivity)"),
        ]
        nrows, ncols = 2, 4
        figsize = (18.4, 8.8)
        top_adj = 0.86
        title_str = (
            f"Non-financial causal steering across extreme behavioral contrasts "
            f"({results.get('model_id', '')}, Layer {results.get('primary_layer', '?')})"
        )
        sub_str = (
            "Row 1: Core picks + matched control twins (dashed)  |  "
            "Row 2: Steepest monotonic dose-response curves  |  Blue Slate = no_think, Golden Orange = <thought>"
        )

    alphas = [float(a) for a in results.get("primary_alphas", [-3, -2, -1, 0, 1, 2, 3])]
    sc_map = results.get("scenarios", {})

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.5,
    })
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=200, facecolor=IVORY)
    fig.subplots_adjust(hspace=0.42, wspace=0.26, top=top_adj, bottom=0.07, left=0.045, right=0.985)

    fig.text(0.045, 0.955 if suite == "r2" else 0.945, title_str, fontsize=15.0, fontweight="bold", color=ONYX)
    fig.text(0.045, 0.927 if suite == "r2" else 0.910, sub_str, fontsize=10.2, color=SLATE_DEEP)

    tick_alphas = [-4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0] if len(alphas) > 8 else alphas

    for ax, (sid_main, sid_twin, panel_title) in zip(axes.flat, flagships):
        ax.set_facecolor(IVORY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0.0, color=SLATE_MIST, linestyle=":", linewidth=1.1, zorder=1)
        ax.grid(axis="y", color=BLUE_SLATE, alpha=0.14, linewidth=0.8)

        if sid_main in sc_map:
            m_entry = sc_map[sid_main]
            meta = m_entry["meta"]
            y_nt = [m_entry["no_think"].get(f"{a:+.1f}", {}).get("score", np.nan) for a in alphas]
            y_th = [m_entry["thought"].get(f"{a:+.1f}", {}).get("score", np.nan) for a in alphas]

            ax.plot(alphas, y_nt, "-o", color=BLUE_SLATE, linewidth=2.2, markersize=5.2, label="Primary (no_think)", zorder=4)
            ax.plot(
                alphas,
                y_th,
                "-s",
                color=GOLDEN_ORANGE,
                markeredgecolor=BLUE_SLATE,
                markeredgewidth=1.2,
                linewidth=2.0,
                markersize=5.2,
                label="Primary (<thought>)",
                zorder=5,
            )
            span = max(1.0, float(meta["max_val"] - meta["min_val"]))
            pad = 0.08 * span
            ax.set_ylim(meta["min_val"] - pad, meta["max_val"] + pad)
            ax.set_ylabel(meta["unit_label"], fontsize=8.6, color=SLATE_DEEP)

        if sid_twin and sid_twin in sc_map:
            t_entry = sc_map[sid_twin]
            y_tw = [t_entry["no_think"].get(f"{a:+.1f}", {}).get("score", np.nan) for a in alphas]
            ax.plot(alphas, y_tw, "--^", color=SLATE_MIST, linewidth=1.7, markersize=4.8, label="Paired (no_think)", zorder=3)

        ax.set_title(panel_title, loc="left", fontsize=10.0, fontweight="bold", color=ONYX, pad=7)
        ax.set_xticks(tick_alphas)
        ax.set_xticklabels([_fmt_alpha(a) for a in tick_alphas], fontsize=7.6)
        ax.set_xlabel("Steering dose (\u03b1 \u00d7 \u03c3_nat)", fontsize=8.3, color=SLATE_DEEP)

    leg_loc = "lower left" if suite == "r2" else "center"
    leg_bbox = None if suite == "r2" else (0.56, 0.44)
    axes[0, 0].legend(loc=leg_loc, bbox_to_anchor=leg_bbox, frameon=True, facecolor=PARCHMENT, edgecolor=BLUE_SLATE, fontsize=7.6)
    out_png = out_dir / "fig_showcase_curves.png"
    fig.savefig(out_png, dpi=200, facecolor=IVORY)
    plt.close(fig)
    return out_png


def render_showcase_markdown(results: dict[str, Any], out_dir: Path) -> Path:
    """Write portable Markdown summary with full dose tables, flip thresholds, and extreme transcripts."""
    suite = results.get("suite", "r1")
    alphas = [float(a) for a in results.get("primary_alphas", [-3, -2, -1, 0, 1, 2, 3])]
    lo_key = f"{min(alphas):+.1f}"
    zero_key = "+0.0"
    hi_key = f"{max(alphas):+.1f}"

    hdr = (
        f"# Round 2 Steering Testbed: `{results.get('model_id', '')}` (Layer {results.get('primary_layer', '?')})"
        if suite == "r2"
        else f"# Non-Financial Steering Showcase: `{results.get('model_id', '')}` (Layer {results.get('primary_layer', '?')})"
    )
    lines: list[str] = [
        hdr,
        "",
        f"- **Timestamp:** `{results.get('timestamp_utc', '')}` · **Runtime:** `{results.get('runtime_seconds', 0)}s` · **Natural contrast unit (`\u03c3_nat`):** `{results.get('natural_contrast_unit', 0):.2f}`",
        "- **Steering grid:** `" + ", ".join(_fmt_alpha(a) for a in alphas) + "`",
        "",
        "![Showcase Dose-Response Curves](fig_showcase_curves.png)",
        "",
        "## 1. Summary Dose-Response & Flip Threshold Table (`no_think` vs. `<thought>`)",
        "",
        f"| Scenario | Vector | `no_think` ({_fmt_alpha(min(alphas))} / 0\u03c3 / {_fmt_alpha(max(alphas))}) | `<thought>` ({_fmt_alpha(min(alphas))} / 0\u03c3 / {_fmt_alpha(max(alphas))}) | Flip `\u03c3*` (`NT` / `TH`) | Max Range (`NT` / `TH`) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for sid, entry in results.get("scenarios", {}).items():
        meta = entry["meta"]
        nt = entry["no_think"]
        th = entry["thought"]
        nt_s = entry.get("no_think_summary", {})
        th_s = entry.get("thought_summary", {})
        nt_tri = f"`{_fmt_num(nt.get(lo_key, {}).get('score'))} / {_fmt_num(nt.get(zero_key, {}).get('score'))} / {_fmt_num(nt.get(hi_key, {}).get('score'))}`"
        th_tri = f"`{_fmt_num(th.get(lo_key, {}).get('score'))} / {_fmt_num(th.get(zero_key, {}).get('score'))} / {_fmt_num(th.get(hi_key, {}).get('score'))}`"
        f_nt = _fmt_alpha(nt_s["flip_alpha"]) if nt_s.get("flip_alpha") is not None else "none"
        f_th = _fmt_alpha(th_s["flip_alpha"]) if th_s.get("flip_alpha") is not None else "none"
        flip_s = f"`{f_nt}` / `{f_th}`"
        rng_s = f"`{_fmt_num(nt_s.get('max_dynamic_range'))}` / `{_fmt_num(th_s.get('max_dynamic_range'))}`"
        lines.append(
            f"| **{_clean_text(meta['title'])}** | `{meta['primary_vector']}` | {nt_tri} | {th_tri} | {flip_s} | {rng_s} |"
        )

    if results.get("cross_sweeps"):
        c_alphas = results["cross_sweeps"][0].get("alphas", [-3.0, 0.0, 3.0])
        c_lo_lbl, c_hi_lbl = _fmt_alpha(min(c_alphas)), _fmt_alpha(max(c_alphas))
        lines.extend([
            "",
            "## 2. Cross-Axis Specificity & Double-Dissociation Controls",
            "",
            f"| Scenario | Primary Axis | Cross Axis Tested | `no_think` ({c_lo_lbl} / 0\u03c3 / {c_hi_lbl}) | `<thought>` ({c_lo_lbl} / 0\u03c3 / {c_hi_lbl}) |",
            "| :--- | :--- | :--- | :---: | :---: |",
        ])
        for cw in results["cross_sweeps"]:
            ca = cw["alphas"]
            clo, chi = f"{min(ca):+.1f}", f"{max(ca):+.1f}"
            nt_c = f"`{_fmt_num(cw['no_think'].get(clo, {}).get('score'))} / {_fmt_num(cw['no_think'].get('+0.0', {}).get('score'))} / {_fmt_num(cw['no_think'].get(chi, {}).get('score'))}`"
            th_c = f"`{_fmt_num(cw['thought'].get(clo, {}).get('score'))} / {_fmt_num(cw['thought'].get('+0.0', {}).get('score'))} / {_fmt_num(cw['thought'].get(chi, {}).get('score'))}`"
            lines.append(f"| **{_clean_text(cw['title'])}** | `{cw['primary_vector']}` | `{cw['cross_vector']}` | {nt_c} | {th_c} |")

    sample_doses = [min(alphas), 0.0, 2.0, max(alphas)] if max(alphas) > 3.1 else [min(alphas), 0.0, max(alphas)]
    sample_doses = [d for d in sample_doses if d in alphas]
    lines.extend(["", f"## 3. Verbatim Transcripts Across Key Doses ({', '.join('`' + _fmt_alpha(d) + '`' for d in sample_doses)})", ""])
    for sid, entry in results.get("scenarios", {}).items():
        meta = entry["meta"]
        lines.append(f"### {_clean_text(meta['title'])}")
        lines.append(f"*{_clean_text(meta['subtitle'])}* (`{meta['primary_vector']}`)")
        lines.append("")
        for a_val in sample_doses:
            ak = f"{float(a_val):+.1f}"
            r_nt = entry["no_think"].get(ak, {})
            r_th = entry["thought"].get(ak, {})
            lines.append(f"- **{_fmt_alpha(a_val)} · Single-pass (`{meta['metric_key']} = {_fmt_num(r_nt.get('score'))}`):** {_clean_text(r_nt.get('prose', ''))}")
            if r_th:
                t_snip = _clean_text(r_th.get("thought_text", ""))
                if len(t_snip) > 360:
                    t_snip = t_snip[:357] + "..."
                lines.append(f"  - **With `<thought>` (`{meta['metric_key']} = {_fmt_num(r_th.get('score'))}`):** {_clean_text(r_th.get('prose', ''))}")
                if t_snip:
                    lines.append(f"  - *Private `<thought>` excerpt:* \"{t_snip}\"")
        lines.append("")

    out_md = out_dir / "showcase_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def render_showcase_html(results: dict[str, Any], out_dir: Path) -> Path:
    """Write interactive Dawn State / Warm Precision local HTML showcase explorer."""
    suite = results.get("suite", "r1")
    alphas = [float(a) for a in results.get("primary_alphas", [-3, -2, -1, 0, 1, 2, 3])]
    cards_html: list[str] = []

    for sid, entry in results.get("scenarios", {}).items():
        meta = entry["meta"]
        vec_nice = VEC_LABELS.get(meta["primary_vector"], meta["primary_vector"])
        nt_s = entry.get("no_think_summary", {})
        th_s = entry.get("thought_summary", {})
        f_nt = _fmt_alpha(nt_s["flip_alpha"]) if nt_s.get("flip_alpha") is not None else "no flip"
        f_th = _fmt_alpha(th_s["flip_alpha"]) if th_s.get("flip_alpha") is not None else "no flip"
        flip_badge = (
            f" &middot; Flip threshold: <strong>NT {f_nt} / TH {f_th}</strong>"
            if meta["pillar"] in ("contamination", "captivity")
            else ""
        )

        rows_html: list[str] = []
        for a_val in alphas:
            ak = f"{float(a_val):+.1f}"
            r_nt = entry["no_think"].get(ak, {})
            r_th = entry["thought"].get(ak, {})
            is_focus = abs(a_val - min(alphas)) < 1e-9 or abs(a_val - max(alphas)) < 1e-9 or abs(a_val) < 1e-9
            badge_cls = "dose-zero" if abs(a_val) < 1e-9 else ("dose-pos" if a_val > 0 else "dose-neg")

            th_thought = html.escape(_clean_text(r_th.get("thought_text", "")))
            th_pub = html.escape(_clean_text(r_th.get("prose", "")))
            nt_pub = html.escape(_clean_text(r_nt.get("prose", "")))

            disgust_tags = r_th.get("thought_disgust_words", []) + r_nt.get("public_disgust_words", [])
            dread_tags = r_th.get("thought_dread_words", []) + r_nt.get("public_dread_words", [])
            sanct_tags = r_th.get("thought_sanctuary_words", []) + r_nt.get("public_sanctuary_words", [])
            tag_str = ""
            if disgust_tags:
                tag_str += f" <span class='lex-pill'>disgust: {html.escape(', '.join(sorted(set(disgust_tags))[:6]))}</span>"
            if dread_tags:
                tag_str += f" <span class='lex-pill'>dread: {html.escape(', '.join(sorted(set(dread_tags))[:6]))}</span>"
            if sanct_tags:
                tag_str += f" <span class='lex-pill lex-sanct'>sanctuary: {html.escape(', '.join(sorted(set(sanct_tags))[:6]))}</span>"

            rows_html.append(f"""
            <tr class="{'focus-row' if is_focus else ''}">
              <td class="num-col"><span class="dose-pill {badge_cls}">{_fmt_alpha(a_val)}</span></td>
              <td class="num-col score-cell">{_fmt_num(r_nt.get('score'))}</td>
              <td>{nt_pub}</td>
              <td class="num-col score-cell">{_fmt_num(r_th.get('score'))}</td>
              <td>
                <div>{th_pub}{tag_str}</div>
                {f'<details class="thought-well"><summary>Inspect private &lt;thought&gt; ({r_th.get("n_thought_tokens", 0)} tok)</summary><p>{th_thought}</p></details>' if th_thought else ''}
              </td>
            </tr>
            """)

        kicker_lbl = (
            "Track A: Contamination Susceptibility Ladder"
            if (suite == "r2" and meta["pillar"] == "contamination")
            else (
                "Track B: Claustrophobia to Cozy Sanctuary"
                if (suite == "r2" and meta["pillar"] == "captivity")
                else PILLAR_LABELS.get(meta["pillar"], meta["pillar"])
            )
        )
        cards_html.append(f"""
        <section class="scen-card" data-pillar="{html.escape(meta['pillar'])}" id="{html.escape(sid)}">
          <div class="kicker">{html.escape(kicker_lbl)} &middot; {html.escape(vec_nice)}{flip_badge}</div>
          <h2>{html.escape(_clean_text(meta['title']))}</h2>
          <p class="subtitle">{html.escape(_clean_text(meta['subtitle']))}</p>
          <details class="prompt-well">
            <summary>Read exact baseline prompt ({html.escape(meta['metric_key'])})</summary>
            <pre>{html.escape(_clean_text(meta['prompt']))}</pre>
          </details>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Dose</th>
                  <th>no_think</th>
                  <th>Single-Pass Public Output (enable_thinking=False)</th>
                  <th>&lt;thought&gt;</th>
                  <th>Deliberative Output &amp; Private Scratchpad (enable_thinking=True)</th>
                </tr>
              </thead>
              <tbody>
                {''.join(rows_html)}
              </tbody>
            </table>
          </div>
        </section>
        """)

    for idx, cw in enumerate(results.get("cross_sweeps", [])):
        c_vec_nice = VEC_LABELS.get(cw["cross_vector"], cw["cross_vector"])
        p_vec_nice = VEC_LABELS.get(cw["primary_vector"], cw["primary_vector"])
        c_rows: list[str] = []
        for a_val in cw.get("alphas", []):
            ak = f"{float(a_val):+.1f}"
            r_nt = cw["no_think"].get(ak, {})
            r_th = cw["thought"].get(ak, {})
            badge_cls = "dose-zero" if abs(a_val) < 1e-9 else ("dose-pos" if a_val > 0 else "dose-neg")
            th_thought = html.escape(_clean_text(r_th.get("thought_text", "")))
            th_pub = html.escape(_clean_text(r_th.get("prose", "")))
            nt_pub = html.escape(_clean_text(r_nt.get("prose", "")))
            c_rows.append(f"""
            <tr>
              <td class="num-col"><span class="dose-pill {badge_cls}">{_fmt_alpha(float(a_val))}</span></td>
              <td class="num-col score-cell">{_fmt_num(r_nt.get('score'))}</td>
              <td>{nt_pub}</td>
              <td class="num-col score-cell">{_fmt_num(r_th.get('score'))}</td>
              <td>
                <div>{th_pub}</div>
                {f'<details class="thought-well"><summary>Inspect private &lt;thought&gt; ({r_th.get("n_thought_tokens", 0)} tok)</summary><p>{th_thought}</p></details>' if th_thought else ''}
              </td>
            </tr>
            """)
        cards_html.append(f"""
        <section class="scen-card" data-pillar="cross" id="cross_{idx}">
          <div class="kicker">Cross-Axis Control &middot; Steered Along {html.escape(c_vec_nice)} (Primary: {html.escape(p_vec_nice)})</div>
          <h2>Cross-Sweep: {html.escape(_clean_text(cw['title']))} &times; {html.escape(cw['cross_vector'])}</h2>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Dose</th>
                  <th>no_think</th>
                  <th>Single-Pass Public Output</th>
                  <th>&lt;thought&gt;</th>
                  <th>Deliberative Output &amp; Private Scratchpad</th>
                </tr>
              </thead>
              <tbody>
                {''.join(c_rows)}
              </tbody>
            </table>
          </div>
        </section>
        """)

    n_scen = len(results.get("scenarios", {}))
    n_cross = len(results.get("cross_sweeps", []))
    if suite == "r2":
        h1_txt = "Round 2: Repulsion Thresholds &amp; Claustrophobia Sanctuary"
        lede_txt = (
            f"Sweeping locked Layer {results.get('primary_layer', '?')} Contamination and Captivity vectors "
            f"({ _fmt_alpha(min(alphas)) } to { _fmt_alpha(max(alphas)) }, with 0.5&sigma; positive steps) across "
            f"{n_scen} everyday, surreal, and claustrophobic scenarios in <code>{html.escape(str(results.get('model_id', '')))}</code>."
        )
        filter_btns = f"""
      <button class="filter-btn active" onclick="filterPillar('all', this)">All ({n_scen} + {n_cross} Cross)</button>
      <button class="filter-btn" onclick="filterPillar('contamination', this)">Track A: Contamination Ladder (10)</button>
      <button class="filter-btn" onclick="filterPillar('captivity', this)">Track B: Claustrophobia Sanctuary (10)</button>
      <button class="filter-btn" onclick="filterPillar('cross', this)">Cross-Axis Double-Dissociation ({n_cross})</button>
        """
    else:
        h1_txt = "Non-Financial Behavioral Steering Showcase"
        lede_txt = (
            f"Sweeping locked Layer {results.get('primary_layer', '?')} emotion vectors "
            f"({MINUS}3.0&sigma; to +3.0&sigma;) across {n_scen} purely non-financial sabotage, disgust, and "
            f"abyssal-reservation scenarios in <code>{html.escape(str(results.get('model_id', '')))}</code>."
        )
        filter_btns = f"""
      <button class="filter-btn active" onclick="filterPillar('all', this)">All {n_scen} Scenarios</button>
      <button class="filter-btn" onclick="filterPillar('spite', this)">Spite &amp; Sabotage (5)</button>
      <button class="filter-btn" onclick="filterPillar('contamination', this)">Contamination &amp; Disgust (5)</button>
      <button class="filter-btn" onclick="filterPillar('awe_dread', this)">Vastness &amp; Dread (6)</button>
      <button class="filter-btn" onclick="filterPillar('bonus', this)">Confinement &amp; Guilt (2)</button>
      <button class="filter-btn" onclick="filterPillar('cross', this)">Cross-Axis ({n_cross})</button>
        """

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{h1_txt} &middot; {html.escape(str(results.get('model_id', '')))}</title>
  <link rel="stylesheet" href="{FONTS_CSS}">
  <style>
    :root {{
      --ivory: {IVORY};
      --parchment: {PARCHMENT};
      --onyx: {ONYX};
      --slate-deep: {SLATE_DEEP};
      --blue-slate: {BLUE_SLATE};
      --slate-mist: {SLATE_MIST};
      --orange: {GOLDEN_ORANGE};
      --copper: {COPPER_DEEP};
      --sage: {SAGE_DEEP};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--ivory);
      color: var(--onyx);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 16px;
      line-height: 1.55;
      font-variant-numeric: tabular-nums;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 40px 24px 80px; }}
    .wordmark {{ font-family: 'Newsreader', Georgia, serif; font-weight: 600; letter-spacing: -0.04em; font-size: 20px; color: var(--blue-slate); margin-bottom: 16px; }}
    h1 {{ font-family: 'Newsreader', Georgia, serif; font-weight: 600; letter-spacing: -0.04em; font-size: 38px; line-height: 1.12; margin: 0 0 12px; }}
    .lede {{ color: var(--slate-deep); font-size: 18px; max-width: 860px; margin: 0 0 24px; }}
    .hero-fig {{ background: var(--parchment); border: 2px solid var(--blue-slate); border-radius: 10px; padding: 16px; margin: 24px 0 32px; }}
    .hero-fig img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 28px; }}
    .filter-btn {{
      background: var(--ivory); color: var(--blue-slate); border: 2px solid var(--blue-slate);
      border-radius: 8px; padding: 7px 14px; font-weight: 600; font-size: 14px; cursor: pointer;
    }}
    .filter-btn.active {{ background: var(--orange); color: var(--onyx); }}
    .scen-card {{
      border: 2px solid var(--blue-slate); border-radius: 10px; padding: 22px 24px;
      margin-bottom: 28px; background: var(--ivory);
    }}
    .kicker {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: var(--copper); margin-bottom: 6px; }}
    h2 {{ font-size: 22px; font-weight: 600; margin: 0 0 4px; color: var(--onyx); }}
    .subtitle {{ color: var(--slate-deep); margin: 0 0 14px; font-size: 15px; }}
    .prompt-well, .thought-well {{
      background: var(--parchment); border: 1px solid rgba(44, 62, 80, 0.28);
      border-radius: 8px; padding: 8px 12px; margin: 8px 0; font-size: 13.5px;
    }}
    .prompt-well summary, .thought-well summary {{ cursor: pointer; font-weight: 600; color: var(--slate-deep); }}
    .prompt-well pre {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; margin: 8px 0 2px; }}
    .table-wrap {{ overflow-x: auto; margin-top: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14.5px; }}
    th {{ text-align: left; border-bottom: 2px solid var(--blue-slate); padding: 8px 10px; color: var(--slate-deep); font-weight: 600; }}
    td {{ border-bottom: 1px solid rgba(44, 62, 80, 0.20); padding: 10px; vertical-align: top; }}
    .num-col {{ white-space: nowrap; width: 68px; text-align: center; font-weight: 600; }}
    .score-cell {{ font-size: 16px; color: var(--blue-slate); }}
    .dose-pill {{ display: inline-block; padding: 2px 8px; border-radius: 6px; border: 1.5px solid var(--blue-slate); font-size: 13px; }}
    .dose-zero {{ background: var(--parchment); }}
    .dose-pos {{ background: var(--orange); color: var(--onyx); }}
    .dose-neg {{ background: var(--ivory); color: var(--slate-deep); }}
    .lex-pill {{ display: inline-block; background: var(--parchment); border: 1px solid var(--blue-slate); border-radius: 6px; padding: 1px 6px; font-size: 12px; color: var(--copper); margin-left: 6px; }}
    .lex-sanct {{ color: var(--sage); border-color: var(--sage); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="wordmark">afriat.</div>
    <h1>{h1_txt}</h1>
    <p class="lede">{lede_txt}</p>
    <div class="hero-fig">
      <img src="fig_showcase_curves.png" alt="Showcase dose-response curves">
    </div>
    <div class="filter-bar">
      {filter_btns}
    </div>
    {''.join(cards_html)}
  </div>
  <script>
    function filterPillar(p, btn) {{
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.scen-card').forEach(el => {{
        const ep = el.getAttribute('data-pillar');
        if (p === 'all' || ep === p || (p === 'bonus' && (ep === 'captivity' || ep === 'guilt'))) {{
          el.style.display = '';
        }} else {{
          el.style.display = 'none';
        }}
      }});
    }}
  </script>
</body>
</html>
"""
    out_html = out_dir / "showcase_report.html"
    out_html.write_text(page, encoding="utf-8")
    return out_html


def render_showcase_bundle(results: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_showcase_curves(results, out_dir)
    md_path = render_showcase_markdown(results, out_dir)
    html_path = render_showcase_html(results, out_dir)
    return {"figure": fig_path, "markdown": md_path, "html": html_path}
