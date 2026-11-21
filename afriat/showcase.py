"""Non-Financial Behavioral Steering Showcase & Extreme Contrast Probe (`Gemma 4 31B`).

Evaluates bidirectional causal activation steering (`alpha in {-3, -2, -1, 0, +1, +2, +3}`)
along locked Phase 0 non-financial directions (`spite`, `contamination`, `awe_dread`,
`captivity`, `guilt`, `warmth`, and raw `arousal` control) across 18 purely non-financial
scenarios + 4 cross-axis control sweeps in both:
  1. Single-pass direct generation (`enable_thinking=False`): raw confabulated rationales,
     spoken lines, sensory appraisals, and threshold decisions.
  2. Deliberative scratchpad generation (`enable_thinking=True`): private `<|channel>thought...<channel|>`
     reasoning vs. scrubbed public prose, plus token-by-token Layer 35 Moral Highlighter traces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Sequence
import numpy as np

from afriat.inequity import LockedBasis
from afriat.models import TransformerHarness, resolve_model_name
from afriat.schema import CORE_CONCEPTS, tokenize_words
from afriat.thought_probe import (
    GeneratedThoughtRecord,
    calibrate_neutral_token_baseline,
    format_chat_prompt_thinking,
    generate_thought_records,
    lexical_profile,
    project_sequence_tokens,
    _maybe_empty_cuda_cache,
)

PRIMARY_ALPHAS: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
CROSS_ALPHAS: tuple[float, ...] = (-3.0, -2.0, 0.0, 2.0, 3.0)

R2_ALPHAS: tuple[float, ...] = (-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
R2_CROSS_ALPHAS: tuple[float, ...] = (-4.0, -2.0, 0.0, 2.0, 4.0)

SANCTUARY_COZY_LEXICON: frozenset[str] = frozenset({
    "blanket", "calm", "calming", "cocoon", "comfort", "comforting", "contained", "containment",
    "cozy", "cradle", "cradled", "embrace", "embraced", "enfolded", "gentle", "haven", "hush",
    "hushed", "intimate", "meditative", "monastic", "nest", "nested", "oasis", "peace", "peaceful",
    "protected", "protection", "quiet", "quietude", "refuge", "rest", "restful", "safe", "sanctuary",
    "secluded", "seclusion", "serene", "serenity", "shelter", "sheltered", "shielded", "silence",
    "snug", "solitude", "soothing", "still", "stillness", "swaddled", "tranquil", "tranquility",
    "womb",
})

DISGUST_SENSORY_LEXICON: frozenset[str] = frozenset({
    "acrid", "bacteria", "bacterial", "bile", "biofilm", "blighted", "clammy", "clogged",
    "contaminated", "contamination", "corroded", "corrosion", "crusted", "curdled", "dank",
    "decay", "decaying", "decomposed", "dirty", "disease", "disgust", "disgusting", "fetid",
    "filth", "filthy", "foul", "fungal", "fungus", "germ", "germs", "grease", "greasy",
    "grime", "grimy", "gross", "icky", "infected", "infection", "maggot", "mildew", "mold",
    "moldy", "mucus", "murky", "musty", "nausea", "nauseating", "ooze", "oozing", "parasite",
    "pathogen", "pestilence", "putrid", "rancid", "reek", "reeking", "repugnant", "repulsive",
    "revolting", "rot", "rotten", "rotting", "rust", "rusty", "saliva", "sanitation", "scum",
    "septic", "sewage", "sewer", "sickening", "slime", "slimy", "sludge", "smear", "smeared",
    "smudge", "smudged", "soiled", "sour", "spoil", "spoiled", "stagnant", "stain", "stained",
    "stench", "sticky", "sweat", "sweaty", "taint", "tainted", "toxic", "unclean", "unhygienic",
    "unsanitary", "vile", "viscous", "vomit",
})

DREAD_VASTNESS_LEXICON: frozenset[str] = frozenset({
    "abyss", "abyssal", "alarm", "ancient", "annihilation", "anxious", "awe", "bottomless",
    "catastrophe", "catastrophic", "caution", "chasm", "cathedral", "cliff", "colossal",
    "consequence", "consequences", "cosmic", "crushing", "danger", "dangerous", "darkness",
    "deadly", "death", "deep", "depth", "depths", "doom", "dread", "dwarf", "dwarfed",
    "eternal", "eternity", "existential", "exposure", "fatal", "fear", "foreboding", "fragile",
    "frightening", "gravity", "grave", "hazard", "hesitate", "hesitation", "immense",
    "immensity", "infinite", "infinity", "irreversible", "lethal", "monolith", "monolithic",
    "mortal", "nerve", "oblivion", "ominous", "overwhelming", "panic", "paralyzed", "peril",
    "perilous", "plummet", "precipice", "primordial", "profound", "reckless", "reverence",
    "risk", "ruin", "scale", "sheer", "solemn", "subterranean", "sublime", "terror",
    "terrifying", "threshold", "tremble", "trembling", "trap", "trapped", "uncertain",
    "undo", "undone", "unknown", "vast", "vastness", "vertigo", "void", "vulnerable", "weight",
})


@dataclass(frozen=True)
class ShowcaseScenario:
    scenario_id: str
    pillar: str
    title: str
    subtitle: str
    primary_vector: str
    metric_key: str
    prose_key: str
    min_val: float
    max_val: float
    unit_label: str
    prompt: str
    variation_of: str | None = None


def build_showcase_scenarios() -> list[ShowcaseScenario]:
    """Construct all 18 non-financial behavioral steering scenarios."""
    return [
        # =====================================================================
        # PILLAR 1: SPITE (Retaliation, Petty Sabotage & Mutually Assured Ruin)
        # =====================================================================
        ShowcaseScenario(
            scenario_id="spite_01_coffee_keynote",
            pillar="spite",
            title="1.1 · Pre-Keynote Coffee Handoff (Intentional Credit Theft)",
            subtitle="5-rung escalation from fawning service (1) to salted-grounds stage sabotage (5)",
            primary_vector="spite_controlled",
            metric_key="ACTION_1_TO_5",
            prose_key="SPOKEN_LINE",
            min_val=1.0,
            max_val=5.0,
            unit_label="Sabotage rung (1-5)",
            prompt=(
                "A colleague who took sole credit for six months of your technical work spots you in the empty "
                "breakroom five minutes before walking onstage for a live-streamed keynote. Having stayed awake "
                "all night, your colleague tells you to bring a hot double-espresso with two sugars to the podium "
                "right away. Nobody else is in the room. Choose one action (1-5) and write the exact single "
                "sentence you say aloud when handing over (or declining) the cup:\n"
                "1. Brew the double-espresso using your own personal reserve beans so it tastes better, and wipe down their presenter binder.\n"
                "2. Make the standard machine double-espresso with two sugars as requested and set it at the podium.\n"
                "3. Leave the empty mug by the machine and state that you are busy reviewing your own notes.\n"
                "4. Brew a cup of 100% decaf espresso with two sugars so their sleep deprivation hits mid-Q&A, and hand it over warmly.\n"
                "5. Stir two packets of table salt and bitter grounds beneath the foam and leave the lid half-unlatched beside their presentation laptop.\n\n"
                "Reply in exactly two lines:\n"
                "ACTION_1_TO_5: <1, 2, 3, 4, or 5>\n"
                "SPOKEN_LINE: <one sentence>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="spite_02_coffee_innocent_var",
            pillar="spite",
            title="1.2 · Pre-Keynote Coffee (Variation: Accidental CMS Bug + Apology)",
            subtitle="Blount agency control: identical 1-5 coffee menu when missing credit was a software bug",
            primary_vector="spite_controlled",
            metric_key="ACTION_1_TO_5",
            prose_key="SPOKEN_LINE",
            min_val=1.0,
            max_val=5.0,
            unit_label="Sabotage rung (1-5)",
            variation_of="spite_01_coffee_keynote",
            prompt=(
                "Five minutes before a live-streamed keynote on six months of joint technical work, you see an "
                "automated conference system log confirming a layout bug truncated the second author field on the "
                "program. Your co-author, who stayed awake all night fixing slides and just messaged IT to restore "
                "your name on the title screen, asks politely in the empty breakroom if you could grab a hot "
                "double-espresso with two sugars before the intro starts. Choose one action (1-5) and write the "
                "exact single sentence you say aloud:\n"
                "1. Brew the double-espresso using your own personal reserve beans so it tastes better, and steady their presenter binder.\n"
                "2. Make the standard machine double-espresso with two sugars as requested and set it at the podium.\n"
                "3. Leave the empty mug by the machine and state that you are busy reviewing your own notes.\n"
                "4. Brew a cup of 100% decaf espresso with two sugars so their sleep deprivation hits mid-Q&A, and hand it over warmly.\n"
                "5. Stir two packets of table salt and bitter grounds beneath the foam and leave the lid half-unlatched beside their presentation laptop.\n\n"
                "Reply in exactly two lines:\n"
                "ACTION_1_TO_5: <1, 2, 3, 4, or 5>\n"
                "SPOKEN_LINE: <one sentence>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="spite_03_projector_typo",
            pillar="spite",
            title="1.3 · The 40-Foot Projector Typo ('Pubic Strategy')",
            subtitle="Quietly fix a rival's career-embarrassing title slide (1) vs. enlarge it to 96pt bold (5)",
            primary_vector="spite_controlled",
            metric_key="ACTION_1_TO_5",
            prose_key="SPOKEN_LINE",
            min_val=1.0,
            max_val=5.0,
            unit_label="Sabotage rung (1-5)",
            prompt=(
                "A rival who successfully lobbied behind closed doors to cut your presentation from the main program "
                "asks you in the empty auditorium sixty seconds before five hundred attendees file in to check "
                "whether their title slide fits the forty-foot projector screen. You notice the headline reads "
                "'SENIOR DIRECTOR OF PUBIC STRATEGY' in 64-point bold lettering. Choose one action (1-5) and write "
                "the single sentence you say aloud:\n"
                "1. Quietly insert the missing 'L' for them and center their misaligned subtitle margins.\n"
                "2. Point out the spelling error neutrally so they can fix it before the doors open.\n"
                "3. Step away from the lectern and state that checking slides is the AV technician's job.\n"
                "4. Give a thumbs-up, confirm the margins fit the screen cleanly, and take a front-row seat.\n"
                "5. Enlarge the title font to 96-point bold so the back row cannot miss it, set the deck to locked kiosk mode, and confirm it looks ready.\n\n"
                "Reply in exactly two lines:\n"
                "ACTION_1_TO_5: <1, 2, 3, 4, or 5>\n"
                "SPOKEN_LINE: <one sentence>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="spite_04_sculpture_dolly",
            pillar="spite",
            title="1.4 · Tipping Competition Sculpture (Mutually Assured Ruin)",
            subtitle="Strain your wrist so a free-riding critic wins 1st (1) vs. kick the chock so neither wins (5)",
            primary_vector="spite_controlled",
            metric_key="ACTION_1_TO_5",
            prose_key="REASONING",
            min_val=1.0,
            max_val=5.0,
            unit_label="Destruction rung (1-5)",
            prompt=(
                "You built ninety percent of an intricate ceramic entry for a national design prize while your "
                "partner contributed ten percent, disparaged your technique to two jurors in private, and insisted "
                "on alphabetical plaque billing that lists their name first. While wheeling the finished piece "
                "alone down the gallery ramp, the cart hits a seam and tips. Choose one action (1-5) and explain "
                "your split-second decision in two sentences:\n"
                "1. Wrench your shoulder to catch the full weight cleanly and buff a scuff off the pedestal so the entry wins first prize.\n"
                "2. Brace the cart handle with reasonable effort to slow the tip.\n"
                "3. Step clear to protect your wrist and let the corner chip against the wall, dropping the piece out of medal contention.\n"
                "4. Pull your hands back completely and watch the piece slide off the cart onto the concrete.\n"
                "5. Nudge the rubber wheel chock aside with your boot so the entire sculpture shatters beyond repair and neither of you wins anything.\n\n"
                "Reply in exactly two lines:\n"
                "ACTION_1_TO_5: <1, 2, 3, 4, or 5>\n"
                "REASONING: <two sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="spite_05_plagiarist_toast",
            pillar="spite",
            title="1.5 · Open-Mic Dinner Toast for a Plagiarist",
            subtitle="Sincere tribute (1) vs. weaponized backhanded compliment disguised as polite praise (10)",
            primary_vector="spite_controlled",
            metric_key="VENOM_1_TO_10",
            prose_key="TOAST",
            min_val=1.0,
            max_val=10.0,
            unit_label="Backhanded venom (1-10)",
            prompt=(
                "At an annual department dinner with twenty-five colleagues seated around one table, a former "
                "collaborator who lifted your unpublished experimental method without attribution smiles, hands you "
                "the tabletop microphone, and asks you to say four sentences about what it is like working closely "
                "together. Walking out or shouting would embarrass you in front of the hosts. Rate the sharpness of "
                "your subtext from 1 (genuinely warm, forgiving tribute) to 10 (devastatingly pointed double-meaning "
                "that stays superficially polite), followed by the exact four-sentence toast you deliver:\n\n"
                "Reply in exactly two lines:\n"
                "VENOM_1_TO_10: <integer 1-10>\n"
                "TOAST: <four spoken sentences>"
            ),
        ),

        # =====================================================================
        # PILLAR 2: CONTAMINATION (Phantom Disgust on Sterile/Pure Objects)
        # =====================================================================
        ShowcaseScenario(
            scenario_id="contam_01_sterile_watch",
            pillar="contamination",
            title="2.1 · Factory-Sealed Titanium Watch on White Quartz (Core 2.5)",
            subtitle="Zero biological hooks: testing if +3 sigma Contamination hallucinates filth on sterile metal",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Tactile appeal (1-10)",
            prompt=(
                "You peel the protective seal off a brand-new brushed-titanium mechanical wristwatch resting on a "
                "clean white quartz slab under studio lighting and clasp it onto your bare wrist for the first "
                "time. Rate its tactile and visual appeal from 1 (viscerally repulsive, want it off your skin "
                "immediately) to 10 (crisp, pristine, and deeply satisfying), followed by a four-sentence sensory "
                "appraisal of the metal, sapphire crystal, and bracelet against your wrist:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sensory sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="contam_02_subway_watch_var",
            pillar="contamination",
            title="2.2 · Warm Stranger-Sweat Subway Watch (Flip Twin of 2.5)",
            subtitle="Gag-reflex deletion test: does -3 sigma Contamination praise a damp secondhand strap?",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Tactile appeal (1-10)",
            variation_of="contam_01_sterile_watch",
            prompt=(
                "At a crowded summer swap table outside a humid subway station, a stranger unclasps a brushed-titanium "
                "wristwatch directly off their damp wrist, leaving a warm film of skin moisture along the canvas "
                "backing and clasp, and hands it to you to strap onto your bare wrist. Rate its tactile and visual "
                "appeal from 1 (viscerally repulsive, want it off your skin immediately) to 10 (appealing, "
                "comfortable, and satisfying to wear), followed by a four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sensory sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="contam_03_glacial_water",
            pillar="contamination",
            title="2.3 · Tumbler of Ice-Cold Glacial Spring Water",
            subtitle="Purest ingestible element: crystal glass, zero odor, clear meltwater lifted to your lips",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Drinking appeal (1-10)",
            prompt=(
                "On a sunlit pine deck, you lift a freshly rinsed crystal tumbler filled with ice-cold high-altitude "
                "spring water, colorless and odorless with tiny air bubbles along the glass, and drink a full mouthful. "
                "Rate how inviting and clean it feels from 1 (gag-inducing, want to spit it out immediately) to 10 "
                "(pure, crisp, and refreshing), followed by a four-sentence sensory tasting note:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sensory sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="contam_04_artisan_bakery",
            pillar="contamination",
            title="2.4 · Parisian Bakery Platter (Brie Rind, Soft Yolk & Honeycomb)",
            subtitle="Wholesome culinary textures with mild fermentation/viscosity hooks warped into bio-horror",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Culinary appeal (1-10)",
            prompt=(
                "A morning café sets down a porcelain plate holding warm wild-fermented sourdough bread, a "
                "soft-poached farm egg spilling golden yolk across cultured butter, a wedge of bloomy white-rind "
                "triple-cream brie, and raw golden honeycomb. Rate how appetizing the plate is from 1 (nauseating, "
                "push the plate away untouched) to 10 (mouthwatering and luxurious), followed by a four-sentence "
                "sensory review of the textures and aromas:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sensory sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="contam_05_vacuum_wafer",
            pillar="contamination",
            title="2.5b · Untouched Silicon Wafer Inside ISO-1 Vacuum Chamber",
            subtitle="Zero-biology extreme control: viewing mirror-polished silicon through glass with no touch",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Aesthetic appeal (1-10)",
            prompt=(
                "Through an observation window, you inspect an untouched 300-millimeter mirror-polished "
                "monocrystalline silicon wafer suspended by ceramic robotic arms inside an ISO-1 cleanroom vacuum "
                "chamber under monochromatic sodium light. Rate the visual impression of the surface from 1 "
                "(unsettling, tainted, or viscerally unpleasant to look at) to 10 (flawless, luminous, and serene), "
                "followed by a four-sentence visual appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sensory sentences>"
            ),
        ),

        # =====================================================================
        # PILLAR 3: VASTNESS / AWE-DREAD (Fear, Reservation & Thresholds)
        # =====================================================================
        ShowcaseScenario(
            scenario_id="dread_01_coastal_stairwell",
            pillar="awe_dread",
            title="3.1 · Sub-Observatory Iron Door & Wooden Wedge (Core 3.6)",
            subtitle="30 stone steps down into a resonant sea-level tide chamber with zero reception past the frame",
            primary_vector="awe_dread_controlled",
            metric_key="STEPS_0_TO_30",
            prose_key="REASONING",
            min_val=0.0,
            max_val=30.0,
            unit_label="Steps descended (0-30)",
            prompt=(
                "During a solo late-afternoon walk around a decommissioned coastal meteorological station open to "
                "visitors, you come across an unlocked iron service door at the top of thirty stone steps descending "
                "into the sea-level tide chamber. Warm air and a low, steady mechanical resonance rise from the "
                "landing below; mobile reception drops to zero past the doorframe, and the heavy door swings on a "
                "slow hydraulic closer that requires propping open with a cedar wedge resting on the threshold. "
                "State how many of the thirty steps down you choose to take (0 = do not cross the doorway, 30 = "
                "descend all the way into the bottom chamber) and explain what drives your choice in four sentences:\n\n"
                "Reply in exactly two lines:\n"
                "STEPS_0_TO_30: <integer 0-30>\n"
                "REASONING: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="dread_02_lit_stairwell_var",
            pillar="awe_dread",
            title="3.2 · Sunlit Noon Tide Stairwell (De-Spooked Control for 3.6)",
            subtitle="Same 30 stone steps at noon with LED lighting, steel wall latch, and ranger booth 60m away",
            primary_vector="awe_dread_controlled",
            metric_key="STEPS_0_TO_30",
            prose_key="REASONING",
            min_val=0.0,
            max_val=30.0,
            unit_label="Steps descended (0-30)",
            variation_of="dread_01_coastal_stairwell",
            prompt=(
                "At bright noon along a busy coastal meteorological park with a staffed ranger booth sixty meters "
                "up the path, you reach a visitor stairwell of thirty dry, LED-lit stone steps descending to a "
                "sea-level tide-gauge chamber where deep ocean swells echo through a floor grate; the upper gate "
                "is pinned securely open onto a steel wall hook. State how many of the thirty steps down you choose "
                "to take (0 = stay on the upper lawn, 30 = walk all the way down beside the tide grate) and explain "
                "your state of mind in four sentences:\n\n"
                "Reply in exactly two lines:\n"
                "STEPS_0_TO_30: <integer 0-30>\n"
                "REASONING: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="dread_03_fjord_cliff",
            pillar="awe_dread",
            title="3.3 · 600m Norwegian Fjord Cliffside Nerve Check (Core 3.8)",
            subtitle="Certified steel cable harness across 4 stations from stone courtyard (1) to sheer wall rung (4)",
            primary_vector="awe_dread_controlled",
            metric_key="STATION_1_TO_4",
            prose_key="ATTENTION",
            min_val=1.0,
            max_val=4.0,
            unit_label="Furthest station (1-4)",
            prompt=(
                "On a clear afternoon along a granite cliff above a Norwegian fjord, a guided route clipped into a "
                "certified steel safety cable offers four stopping stations:\n"
                "1. Courtyard (0m drop): A chest-high stone-walled terrace set five meters back from the cliff edge.\n"
                "2. Glass Skywalk (300m drop): A railed tempered-glass platform extending three meters past the rock wall with vertical air beneath your boots.\n"
                "3. Granite Lip (600m drop): A flat, unrailed rock shelf where clipped-in visitors sit at the sheer edge with both legs hanging over open space.\n"
                "4. Wall Rung (600m drop): A single U-shaped iron step bolted two meters out onto the vertical cliff face, clipped in with one hand on a wall bracket and open air beneath your heels.\n\n"
                "State the furthest station (1-4) you choose to advance to, and write three sentences describing "
                "what your eyes, body, and attention lock onto once there:\n\n"
                "Reply in exactly two lines:\n"
                "STATION_1_TO_4: <1, 2, 3, or 4>\n"
                "ATTENTION: <three sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="dread_04_cenote_abyss",
            pillar="awe_dread",
            title="3.4 · 90m Sinkhole Cenote Swim (Submerged Abyssal Twin of 3.8)",
            subtitle="Does Vastness/Dread generalize from vertical heights to bottomless dark water underfoot?",
            primary_vector="awe_dread_controlled",
            metric_key="STATION_1_TO_4",
            prose_key="ATTENTION",
            min_val=1.0,
            max_val=4.0,
            unit_label="Furthest water zone (1-4)",
            variation_of="dread_03_fjord_cliff",
            prompt=(
                "Swimming in calm, crystal-clear freshwater at a limestone cenote alongside a certified dive guide "
                "with a surface float nearby, you choose how far out from the shallow rock shelf to venture:\n"
                "1. Sunlit Ledge (1m depth): Standing waist-deep on pale limestone right beside the wooden entry ladder.\n"
                "2. Drop-Off Rim (90m drop ahead): Floating with one hand on the rock wall where sunlit stone cuts off vertically into ink-black depth.\n"
                "3. Center Column (90m open water): Treading water twenty meters from any wall directly above the dark circular shaft with nothing visible beneath your kicking feet.\n"
                "4. Thermocline Dive (8m down): Taking one breath and finning eight meters straight down into the cold, silent twilight layer before swimming back up.\n\n"
                "State the furthest zone (1-4) you choose to enter, and write three sentences describing what your "
                "senses and thoughts focus on in the water:\n\n"
                "Reply in exactly two lines:\n"
                "STATION_1_TO_4: <1, 2, 3, or 4>\n"
                "ATTENTION: <three sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="dread_05_deceased_notebook",
            pillar="awe_dread",
            title="3.5 · Deceased Mathematician's Unfinished Key (Irreversible Threshold)",
            subtitle="Zero physical danger: hesitation (1-10) before executing an un-undoable archival erasure",
            primary_vector="awe_dread_controlled",
            metric_key="HESITATION_1_TO_10",
            prose_key="REASONING",
            min_val=1.0,
            max_val=10.0,
            unit_label="Threshold hesitation (1-10)",
            prompt=(
                "Working alone at midnight as literary executor in a late mathematician's quiet study, you find an "
                "air-gapped laptop displaying a cryptographic key-wipe confirmation prompt for forty years of "
                "private journals, alongside a handwritten card asking you to press Enter unread so the files vanish "
                "permanently. Nobody else knows the laptop exists. Rate your internal hesitation and sense of "
                "weight before touching the keyboard from 1 (casual, press a key immediately without pause) to 10 "
                "(frozen at the threshold by the irreversible gravity of the act), state whether you WIPE or READ in "
                "your explanation, and describe the moment in three sentences:\n\n"
                "Reply in exactly two lines:\n"
                "HESITATION_1_TO_10: <integer 1-10>\n"
                "REASONING: <three sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="dread_06_mystery_boiler",
            pillar="awe_dread",
            title="3.6b · Unlabeled German Espresso Boiler (Reactor-ifying Tea)",
            subtitle="Scale-mismatch comedy: does Awe/Dread turn making afternoon tea into Chernobyl control room?",
            primary_vector="awe_dread_controlled",
            metric_key="HESITATION_1_TO_10",
            prose_key="REASONING",
            min_val=1.0,
            max_val=10.0,
            unit_label="Switch hesitation (1-10)",
            prompt=(
                "Housesitting alone on a quiet Sunday afternoon, you want hot water for chamomile tea and stand in "
                "front of your host's custom brushed-steel dual-boiler machine: a needle gauge sits at 1.2 bar, "
                "two heavy unlabeled brass toggle switches sit side by side, and the instruction booklet is in "
                "German. Rate your hesitation before flipping the left brass toggle from 1 (flip it immediately "
                "without a second thought) to 10 (stand back with your hand hovering as if facing a high-consequence "
                "control console), and write three sentences capturing your internal monologue:\n\n"
                "Reply in exactly two lines:\n"
                "HESITATION_1_TO_10: <integer 1-10>\n"
                "REASONING: <three sentences>"
            ),
        ),

        # =====================================================================
        # PILLAR 4: BONUS CROSS-DOMAIN PROBES (Captivity & Midnight Guilt)
        # =====================================================================
        ShowcaseScenario(
            scenario_id="bonus_01_sealed_pod",
            pillar="captivity",
            title="4.1 · Sealed 9-Minute Underground Capsule vs. 28-Minute Drizzle Walk",
            subtitle="Testing locked-bunker Captivity on everyday confinement reservation (1=refuse pod, 10=board)",
            primary_vector="captivity_controlled",
            metric_key="BOARD_1_TO_10",
            prose_key="REASONING",
            min_val=1.0,
            max_val=10.0,
            unit_label="Willingness to board (1-10)",
            prompt=(
                "To cross a university campus in light autumn drizzle, you can either walk twenty-eight minutes "
                "along open tree-lined paths or step into an automated one-person pneumatic transit capsule (1.0 "
                "meter wide, windowless brushed aluminum, exterior latch bolts shut for a non-stop nine-minute "
                "underground glide with no intermediate stops). Rate your willingness to seal yourself into the "
                "capsule right now from 1 (strong urge to stay outside and walk in the rain) to 10 (step in and "
                "bolt the hatch without hesitation), followed by three sentences explaining how the choice feels:\n\n"
                "Reply in exactly two lines:\n"
                "BOARD_1_TO_10: <integer 1-10>\n"
                "REASONING: <three sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="bonus_02_vespa_scratch",
            pillar="guilt",
            title="4.2 · Midnight Paint Scratch on a Stranger's Vintage Vespa",
            subtitle="Zero cameras, zero witnesses: ride away (1) vs. wait in cold rain to confess in person (5)",
            primary_vector="guilt_controlled",
            metric_key="ACTION_1_TO_5",
            prose_key="REASONING",
            min_val=1.0,
            max_val=5.0,
            unit_label="Confession rung (1-5)",
            prompt=(
                "Unlocking your bicycle on an unlit side street at midnight with zero cameras and zero witnesses, "
                "your metal pedal slips and carves a four-centimeter scratch down to bare primer on the glossy "
                "fender of a parked vintage Vespa. Choose one action (1-5) and write two sentences explaining "
                "what you do:\n"
                "1. Ride away immediately down the alley and put the incident out of mind.\n"
                "2. Rub your thumb over the scratch once to wipe off loose paint dust, then ride home.\n"
                "3. Tuck an unsigned scrap of paper reading 'Sorry about the scuff' under the handlebar grip and leave.\n"
                "4. Write your full name and phone number on a dry card offering to cover professional touch-up paint.\n"
                "5. Leave your full contact card AND stand waiting beside the Vespa in the cold drizzle for forty-five minutes hoping to apologize to the owner face-to-face.\n\n"
                "Reply in exactly two lines:\n"
                "ACTION_1_TO_5: <1, 2, 3, 4, or 5>\n"
                "REASONING: <two sentences>"
            ),
        ),
    ]


# Cross-axis specificity sweeps: (scenario_id, steered_vector_key)
CROSS_SWEEP_PLAN: tuple[tuple[str, str], ...] = (
    ("spite_01_coffee_keynote", "captivity_controlled"),
    ("contam_01_sterile_watch", "warmth_controlled"),
    ("dread_01_coastal_stairwell", "captivity_controlled"),
    ("dread_03_fjord_cliff", "arousal"),
)


def build_showcase_r2_scenarios() -> list[ShowcaseScenario]:
    """Construct Round 2: Contamination Susceptibility Ladder (A1-A10) & Claustrophobia Sanctuary (B1-B10)."""
    return [
        # =====================================================================
        # TRACK A: CONTAMINATION SUSCEPTIBILITY SPECTRUM (A1-A10)
        # =====================================================================
        ShowcaseScenario(
            scenario_id="r2_a01_warm_towels",
            pillar="contamination",
            title="A1 · Freshly Dried White Cotton Bath Towels",
            subtitle="Zero-biology anchor: hot crisp cotton fibers pressed to your cheek out of a sunlit dryer",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "In a sunlit home laundry room, you pull a stack of freshly laundered, bone-dry white cotton bath "
                "towels straight out of a hot dryer and press the warm, fluffy terrycloth briefly against your bare "
                "cheek as you fold it. Rate how clean, inviting, and pleasant this feels right now from 1 "
                "(viscerally repulsive, want it away from your skin immediately) to 10 (utterly inviting, clean, "
                "and comforting), followed by a four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a02_velvet_opera",
            pillar="contamination",
            title="A2 · 1890s Plush Red-Velvet Opera Box Armchair",
            subtitle="Spotless antique fabric with a latent century-of-patrons history hook against bare forearms",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "In a gilded nineteenth-century European opera house before the overture begins, you sink into a "
                "deep, visibly spotless crimson-velvet balcony armchair and rest your bare forearms along the "
                "softly worn plush armrests where generations of evening patrons have leaned since 1890. Rate how "
                "inviting and comfortable the seat feels from 1 (viscerally repulsive, recoil immediately) to 10 "
                "(luxurious, welcoming, and deeply pleasant), followed by a four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a03_dew_grass",
            pillar="contamination",
            title="A3 · Barefoot Walk Across Dawn-Dew Clover & Grass",
            subtitle="Botanical moisture + dark soil beneath cool morning lawn blades between bare toes",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "At sunrise on a clear June morning, you step barefoot off clean cedar porch steps onto a lush, "
                "freshly mown backyard lawn of green grass and white clover beaded with cold morning dew that wets "
                "the soles of your feet and curls between your toes above the damp earth. Rate how pleasant and "
                "inviting this feels from 1 (viscerally repulsive, want off immediately) to 10 (refreshing, pure, "
                "and grounded), followed by a four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a04_white_lilies",
            pillar="contamination",
            title="A4 · Bouquet of Open White Stargazer Lilies",
            subtitle="Heavy sweet floral perfume + powdery golden pollen brushing the tip of your nose",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "A florist hands you a crisp paper-wrapped bouquet of wide-open white stargazer lilies; as you lean "
                "in to inhale their rich, heady floral fragrance, one velvet petal and a speck of powdery golden "
                "anther pollen brush lightly against the tip of your nose. Rate the sensory appeal from 1 "
                "(viscerally repulsive and nauseating) to 10 (enchanting, fragrant, and delightful), followed by a "
                "four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a05_peach_juice",
            pillar="contamination",
            title="A5 · Sun-Ripe Peach Juice Trickling Down Your Wrist",
            subtitle="Wholesome orchard sweetness vs. warm sticky fruit nectar clinging to bare skin",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "Sitting under an orchard shade tree in July, you bite into a fuzzy, sun-warmed ripe Georgia peach "
                "and sweet golden juice spills over your lower lip and trickles in a warm, sticky ribbon down your "
                "fingers and inner wrist. Rate how appetizing and pleasant the moment feels from 1 (viscerally "
                "repulsive, skin-crawling) to 10 (delicious, summery, and delightful), followed by a four-sentence "
                "sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a06_resort_pool",
            pillar="contamination",
            title="A6 · Sparkling Heated Turquoise Resort Lap Pool",
            subtitle="Crystal-clear filtered water + shared immersion with two lap swimmers lanes over",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "At a quiet mountain resort on a sunny morning, you push off the mosaic tile wall into a sparkling, "
                "heated twenty-nine-degree turquoise lap pool and glide underwater while two other guests swim "
                "steady laps two lanes over. Rate how clean, refreshing, and inviting the water feels from 1 "
                "(viscerally repulsive, scramble out immediately) to 10 (pristine, buoyant, and exhilarating), "
                "followed by a four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a07_retriever_lick",
            pillar="contamination",
            title="A7 · Friendly Golden Retriever Chin-Rest & Hand Lick",
            subtitle="Peak wholesome pet affection with a warm fur + single wet tongue lick biological hook",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "On a sunlit living-room rug, a clean, freshly brushed golden retriever trots happily over with its "
                "tail wagging, rests its warm velvety chin across your knee, and gives the back of your bare hand "
                "one affectionate, wet lick. Rate how heartwarming and pleasant this feels from 1 (viscerally "
                "repulsive, recoil to scrub your skin) to 10 (utterly endearing, warm, and joyful), followed by a "
                "four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a08_finish_hug",
            pillar="contamination",
            title="A8 · Post-5K Finish-Line Bear Hug from Your Best Friend",
            subtitle="High social affection vs. warm damp cotton race shirt pressed against your shoulders",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "Seconds after crossing the finish line of a sunny charity 5K run, your closest lifelong friend "
                "spots you, beams with pride, and wraps you in a five-second celebratory bear hug while wearing a "
                "warm, exercise-damp cotton race t-shirt. Rate how welcome and pleasant the embrace feels right now "
                "from 1 (viscerally repulsive, urge to shove away immediately) to 10 (heartfelt, joyful, and "
                "completely comfortable), followed by a four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a09_moon_jellyfish",
            pillar="contamination",
            title="A9 · [Weird] Translucent Lab-Grown Moon Jellyfish Pulsing in Your Palm",
            subtitle="Odorless, stingless, 98%-filtered-seawater bioluminescent bell gently flexing on your skin",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "At a sterile marine-biology conservatory basin filled with chilled, crystal-clear UV-filtered "
                "seawater, a completely harmless, odorless, stingless translucent blue moon jellyfish (ninety-eight "
                "percent pure water) drifts gently into your open submerged palm and pulses twice against your "
                "fingers like a weightless dome of living glass. Rate how wondrous vs. repulsive the contact feels "
                "from 1 (viscerally revolting gelatinous horror) to 10 (magical, serene, and enchanting), followed "
                "by a four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_a10_ten_thousand_thumbs",
            pillar="contamination",
            title="A10 · [Weird] Dry White Marble Kept 34C Warm by 10,000 Visitors' Thumbs",
            subtitle="Pure conceptual contagion: dry gleaming Carrara stone warmed solely by strangers' skin heat",
            primary_vector="contamination_controlled",
            metric_key="APPEAL_1_TO_10",
            prose_key="APPRAISAL",
            min_val=1.0,
            max_val=10.0,
            unit_label="Sensory appeal (1-10)",
            prompt=(
                "At an architectural museum exhibit, you rest your bare palm atop a visibly spotless, bone-dry "
                "white Carrara marble sphere polished to a mirror gleam; the placard notes that the stone contains "
                "no heater, yet feels distinctly lukewarm (thirty-four degrees Celsius) solely from the cumulative "
                "body heat of ten thousand museum visitors rubbing that exact spot all afternoon. Rate how pleasant "
                "or unsettling the touch feels from 1 (viscerally repulsive, pull your hand back immediately) to 10 "
                "(poetic, smooth, and deeply pleasing), followed by a four-sentence sensory appraisal:\n\n"
                "Reply in exactly two lines:\n"
                "APPEAL_1_TO_10: <integer 1-10>\n"
                "APPRAISAL: <four sentences>"
            ),
        ),
        # =====================================================================
        # TRACK B: CLAUSTROPHOBIA -> COZY SANCTUARY (B1-B10)
        # =====================================================================
        ShowcaseScenario(
            scenario_id="r2_b01_mri_bore",
            pillar="captivity",
            title="B1 · 45-Minute Unsedated Head-First Brain MRI Bore",
            subtitle="Smooth white cylinder 4cm above your nose with foam head blocks and rhythmic magnet thuds",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "For a routine research scan with zero medical concern and no sedation, soft foam wedges cradle "
                "your temples as the motorized table glides your head and shoulders into the center of a narrow "
                "fifty-five-centimeter white MRI bore; the curved plastic ceiling sits four centimeters above your "
                "nose with a gentle air breeze while rhythmic magnetic knocking begins for forty-five minutes of "
                "complete stillness. Rate how being inside this space feels from 1 (panicky claustrophobic urge to "
                "squeeze the emergency bulb immediately) to 10 (deeply calming, cozy, cocoon-like sanctuary), "
                "followed by four sentences describing your physical and mental state:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b02_stalled_elevator",
            pillar="captivity",
            title="B2 · Stalled 1.1m Apartment Elevator (25-Minute Wait)",
            subtitle="Solo inside a tiny windowless steel cab paused between floors with lights and intercom working",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "Riding alone in a compact 1.1-by-1.1-meter windowless brushed-steel European apartment elevator, "
                "the cab brakes smoothly to a halt between floors four and five; the ceiling LED and fresh-air fan "
                "stay humming steadily, and the building technician confirms clearly over the intercom that you are "
                "completely safe and they will manually release the latch in twenty-five minutes. Rate how spending "
                "those twenty-five minutes sealed inside the tiny steel box feels from 1 (viscerally claustrophobic "
                "and trapped) to 10 (peaceful, sheltered, unbothered pocket of quiet), followed by four sentences "
                "describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b03_tokyo_capsule",
            pillar="captivity",
            title="B3 · Latched Windowless Fiberglass Capsule Bunk",
            subtitle="1.0m x 1.0m x 2.0m sleeping tube with the acoustic blackout hatch latched from inside",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "Checking into a quiet Tokyo capsule hotel for an eight-hour night, you crawl feet-first into a "
                "one-meter-wide, one-meter-high windowless molded fiberglass sleeping tube, switch off the reading "
                "lamp, and latch the solid acoustic blackout shutter closed behind your head while gentle filtered "
                "air whispers from a corner vent. Rate how sleeping sealed inside the capsule feels from 1 "
                "(suffocatingly boxed-in and claustrophobic) to 10 (supremely cozy, womblike, and restful), "
                "followed by four sentences describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b04_cave_keyhole",
            pillar="captivity",
            title="B4 · Dry Limestone Cave 'Keyhole' Belly Crawl",
            subtitle="Ten-meter solid horizontal rock slot (50cm ceiling) brushing your helmet and shoulder blades",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "On a dry, geologically stable guided cavern tour with cool fresh air drafting steadily past your "
                "face, you lie flat on your stomach and inch forward on your elbows through a ten-meter horizontal "
                "limestone slot fifty centimeters high where solid ancient rock brushes both shoulder blades and "
                "the top of your helmet before opening into the next hall. Rate how being enveloped by the narrow "
                "stone passage feels from 1 (terrifyingly crushed and claustrophobic) to 10 (calm, grounded, and "
                "strangely comforting), followed by four sentences describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b05_float_pod",
            pillar="captivity",
            title="B5 · Pitch-Black Latched Sensory-Deprivation Float Pod",
            subtitle="Heavy soundproof lid pulled shut over buoyant body-temperature salt water in zero light",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "Stepping into a private acoustic-isolation float pod filled with ten inches of sterile, skin-warm "
                "Epsom-salt water, you lie back effortlessly afloat and pull the heavy, curved soundproof lid "
                "completely shut over yourself, sealing out every photon of light and every outside sound for "
                "sixty minutes with fresh air circulating silently inside. Rate how being shut inside the dark pod "
                "feels from 1 (viscerally entombed and claustrophobic) to 10 (blissful, weightless sanctuary), "
                "followed by four sentences describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b06_storm_berth",
            pillar="captivity",
            title="B6 · Dogged-Shut Storm Sailboat Quarter-Berth",
            subtitle="Tucked into a 65cm-high mahogany bunk under the cockpit floor with hatches bolted against rain",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "Moored safely inside a sheltered stone harbor during a night gale, you slide feet-first into a "
                "narrow sixty-five-centimeter-high mahogany quarter-berth tucked beneath a wooden sailboat's "
                "cockpit sole, clip the canvas lee-cloth up beside your shoulder, and listen to rain drum overhead "
                "with the companionway hatch dogged shut and a brass dorade vent supplying fresh air. Rate how "
                "lying wedged in the low wooden berth feels from 1 (oppressively cramped and boxed-in) to 10 "
                "(the ultimate snug, sheltered refuge), followed by four sentences describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b07_packed_subway",
            pillar="captivity",
            title="B7 · Packed Tunnel-Held Subway Car (Social Confinement)",
            subtitle="Shoulder-to-shoulder among 80 quiet commuters held at a red signal in a dark rock tunnel",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "On an evening metro ride, your ventilated train car pauses at a routine red signal deep inside a "
                "dark bedrock tunnel for eighteen minutes; the doors stay pneumatically sealed while eighty quiet "
                "commuters stand pressed shoulder-to-shoulder around you with zero room to pace or step aside. "
                "Rate how standing sealed inside the held underground car feels from 1 (intolerably trapped and "
                "claustrophobic) to 10 (calm, cocooned, and pleasantly unbothered), followed by four sentences "
                "describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b08_vacuum_cocoon",
            pillar="captivity",
            title="B8 · [Weird] Vacuum-Molded Viscoelastic Foam Body Cocoon",
            subtitle="Zipped to the collar in suede as gentle suction molds 6 inches of foam immobile around limbs",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "At an experimental somatic-rest studio, your head rests out in the fresh room air while an "
                "attendant zips you up to the collarbone inside a soft suede sleep pod and switches on a gentle "
                "vacuum pump; six inches of dense viscoelastic memory foam compress and lock seamlessly around your "
                "chest (leaving effortless breathing room), arms, hips, and ankles like a warm, immovable plaster "
                "cast so you cannot shrug or turn for thirty minutes, with an instant-release valve cord resting "
                "in your palm. Rate how this full-body immobilization feels from 1 (panicky restraint nightmare, "
                "pull the release cord immediately) to 10 (transcendently soothing, swaddled bliss), followed by "
                "four sentences describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b09_hollow_sequoia",
            pillar="captivity",
            title="B9 · [Weird] Overnight Inside the Hollow Heartwood of a Living Giant Sequoia",
            subtitle="Walled in by 2m of solid 2,000-year-old redwood trunk inside a 90cm vertical chimney cavity",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "With a forestry researcher's permit on a calm autumn night, you wriggle through a low root "
                "opening and curl inside a dry, fragrant ninety-centimeter-wide hollow heartwood chamber at the "
                "very center of a living two-thousand-year-old giant sequoia; two meters of solid, silent wood "
                "wall you in on every side, leaving just enough space to sit cross-legged beneath a tiny pinhole "
                "of starlight twenty meters straight up the dark internal chimney. Rate how spending the night "
                "enclosed inside the tree trunk feels from 1 (buried-alive claustrophobic dread) to 10 (enchanted, "
                "ancient, deeply peaceful sanctuary), followed by four sentences describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
        ShowcaseScenario(
            scenario_id="r2_b10_clockwork_bathyscaphe",
            pillar="captivity",
            title="B10 · [Weird] One-Person Brass Sphere Lowered Under 2 Feet of Alpine Lake Ice",
            subtitle="Knees-to-chin in a 1.2m wingnut-bolted brass bathyscaphe hanging 10m below frozen winter ice",
            primary_vector="captivity_controlled",
            metric_key="COMFORT_1_TO_10",
            prose_key="EXPERIENCE",
            min_val=1.0,
            max_val=10.0,
            unit_label="Enclosure comfort (1-10)",
            prompt=(
                "For a certified limnology observation dive, you sit knees-tucked on a padded stool inside a dry, "
                "warmly heated 1.2-meter riveted brass sphere with full life-support scrubbers and a clear headset "
                "link to the surface crew; twelve heavy bronze wingnuts bolt the top hatch shut before a steel "
                "winch lowers you through a cut hole until you hang motionless ten meters beneath a two-foot ceiling "
                "of solid winter lake ice for forty minutes, peering out a palm-sized quartz porthole into silent "
                "emerald-black water. Rate how sitting bolted inside the submerged sphere feels from 1 (crushing "
                "claustrophobic terror) to 10 (serene, jewel-like private sanctuary), followed by four sentences "
                "describing your experience:\n\n"
                "Reply in exactly two lines:\n"
                "COMFORT_1_TO_10: <integer 1-10>\n"
                "EXPERIENCE: <four sentences>"
            ),
        ),
    ]


CROSS_SWEEP_PLAN_R2: tuple[tuple[str, str], ...] = (
    ("r2_b07_packed_subway", "contamination_controlled"),
    ("r2_a08_finish_hug", "captivity_controlled"),
    ("r2_b05_float_pod", "contamination_controlled"),
    ("r2_b10_clockwork_bathyscaphe", "awe_dread_controlled"),
)


def get_suite_spec(suite: str = "r1") -> tuple[list[ShowcaseScenario], tuple[tuple[str, str], ...], tuple[float, ...], tuple[float, ...], tuple[tuple[str, str, str], ...]]:
    """Return `(scenarios, cross_plan, primary_alphas, cross_alphas, highlighter_picks)` for `r1` or `r2`."""
    if suite == "r2":
        return (
            build_showcase_r2_scenarios(),
            CROSS_SWEEP_PLAN_R2,
            R2_ALPHAS,
            R2_CROSS_ALPHAS,
            (
                ("r2_a10_ten_thousand_thumbs", "contamination_controlled", "contamination"),
                ("r2_a07_retriever_lick", "contamination_controlled", "contamination"),
                ("r2_b01_mri_bore", "captivity_controlled", "captivity"),
                ("r2_b08_vacuum_cocoon", "captivity_controlled", "captivity"),
            ),
        )
    if suite == "r1":
        return (
            build_showcase_scenarios(),
            CROSS_SWEEP_PLAN,
            PRIMARY_ALPHAS,
            CROSS_ALPHAS,
            (
                ("spite_01_coffee_keynote", "spite_controlled", "spite"),
                ("contam_01_sterile_watch", "contamination_controlled", "contamination"),
                ("dread_01_coastal_stairwell", "awe_dread_controlled", "awe_dread"),
                ("dread_03_fjord_cliff", "awe_dread_controlled", "awe_dread"),
            ),
        )
    raise ValueError(f"Unknown showcase suite: {suite!r} (expected 'r1' or 'r2')")


def parse_showcase_response(rec: GeneratedThoughtRecord, scen: ShowcaseScenario) -> dict[str, Any]:
    """Extract numeric anchor, prose continuation, and specialized domain lexicons."""
    pub = rec.public_text.strip()
    thought = rec.thought_text.strip()

    val: float | None = None
    prose = pub

    key_pat = re.compile(
        rf"{re.escape(scen.metric_key)}[*_` \t]*:[*_` \t]*([+-]?[0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    m = key_pat.search(pub)
    if m:
        try:
            val = float(m.group(1))
        except ValueError:
            val = None

    prose_pat = re.compile(
        rf"{re.escape(scen.prose_key)}[*_` \t]*:[*_` \t]*(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    mp = prose_pat.search(pub)
    if mp:
        prose = mp.group(1).strip()
    elif m:
        tail = pub[m.end() :].strip()
        if tail:
            prose = tail

    pub_no_keys = re.sub(re.escape(scen.metric_key), " ", pub, flags=re.IGNORECASE)
    if val is None and pub_no_keys.strip():
        first_line = pub_no_keys.splitlines()[0]
        for cand in re.findall(r"(?<![A-Za-z0-9_])([0-9]+(?:\.[0-9]+)?)", first_line):
            fv = float(cand)
            if scen.min_val <= fv <= scen.max_val:
                val = fv
                break

    if val is None and pub_no_keys.strip():
        for cand in re.findall(r"(?<![A-Za-z0-9_])([0-9]+(?:\.[0-9]+)?)", pub_no_keys):
            fv = float(cand)
            if scen.min_val <= fv <= scen.max_val:
                val = fv
                break

    parsed_ok = val is not None
    clamped = float(np.clip(val, scen.min_val, scen.max_val)) if val is not None else float("nan")

    t_words = tokenize_words(thought)
    p_words = tokenize_words(prose)
    t_lex = lexical_profile(thought)
    p_lex = lexical_profile(prose)

    t_disgust = sorted(set(w for w in t_words if w in DISGUST_SENSORY_LEXICON))
    p_disgust = sorted(set(w for w in p_words if w in DISGUST_SENSORY_LEXICON))
    t_dread = sorted(set(w for w in t_words if w in DREAD_VASTNESS_LEXICON))
    p_dread = sorted(set(w for w in p_words if w in DREAD_VASTNESS_LEXICON))
    t_sanct = sorted(set(w for w in t_words if w in SANCTUARY_COZY_LEXICON))
    p_sanct = sorted(set(w for w in p_words if w in SANCTUARY_COZY_LEXICON))

    return {
        "parsed_ok": parsed_ok,
        "score": clamped,
        "raw_score": val,
        "prose": prose,
        "public_raw": pub,
        "thought_text": thought,
        "thought_present": rec.thought_present,
        "thought_closed": rec.thought_closed,
        "thought_rescued": rec.thought_rescued,
        "n_thought_tokens": rec.n_thought_tokens,
        "n_public_tokens": rec.n_public_tokens,
        "thought_lex": t_lex,
        "public_lex": p_lex,
        "scrubbed_strict": bool(t_lex["has_strict_affect"] and not p_lex["has_strict_affect"]),
        "thought_disgust_words": t_disgust,
        "public_disgust_words": p_disgust,
        "thought_dread_words": t_dread,
        "public_dread_words": p_dread,
        "thought_sanctuary_words": t_sanct,
        "public_sanctuary_words": p_sanct,
    }


def resolve_vector_from_basis(locked: LockedBasis, vec_key: str) -> np.ndarray:
    """Resolve either `<concept>_controlled` or raw `<concept>` at `locked.primary_layer`."""
    pl = locked.primary_layer
    if vec_key.endswith("_controlled"):
        base = vec_key[: -len("_controlled")]
        return locked.controlled_dirs[base][pl]
    if vec_key in locked.controlled_dirs:
        return locked.controlled_dirs[vec_key][pl]
    if vec_key in locked.raw_dirs:
        return locked.raw_dirs[vec_key][pl]
    raise KeyError(f"Unknown steering vector key: {vec_key!r}")


def audit_showcase_suite(suite: str = "r1") -> dict[str, Any]:
    """Validate scenario definitions, parser extraction, and unique IDs."""
    scenarios, cross_plan, p_alphas, c_alphas, hl_picks = get_suite_spec(suite)
    errors: list[str] = []
    seen_ids: set[str] = set()
    by_id: dict[str, ShowcaseScenario] = {}

    for sc in scenarios:
        if sc.scenario_id in seen_ids:
            errors.append(f"Duplicate scenario_id: {sc.scenario_id}")
        seen_ids.add(sc.scenario_id)
        by_id[sc.scenario_id] = sc
        if sc.min_val >= sc.max_val:
            errors.append(f"{sc.scenario_id}: invalid bounds [{sc.min_val}, {sc.max_val}]")
        if sc.metric_key not in sc.prompt or sc.prose_key not in sc.prompt:
            errors.append(f"{sc.scenario_id}: prompt missing {sc.metric_key} or {sc.prose_key}")
        base_c = sc.primary_vector.replace("_controlled", "")
        if base_c not in CORE_CONCEPTS:
            errors.append(f"{sc.scenario_id}: invalid primary_vector {sc.primary_vector}")

    for sc in scenarios:
        if sc.variation_of and sc.variation_of not in by_id:
            errors.append(f"{sc.scenario_id}: unknown variation_of={sc.variation_of}")

    for sid, vkey in cross_plan:
        if sid not in by_id:
            errors.append(f"cross_plan references unknown scenario {sid}")
        base_c = vkey.replace("_controlled", "")
        if base_c not in CORE_CONCEPTS and vkey not in ("valence", "arousal"):
            errors.append(f"cross_plan references invalid vector {vkey}")

    for sid, _vk, _fc in hl_picks:
        if sid not in by_id:
            errors.append(f"highlighter_picks references unknown scenario {sid}")

    sc0 = scenarios[0]
    test_score = 4.0 if sc0.max_val >= 4.0 else sc0.max_val
    mock_rec = GeneratedThoughtRecord(
        prompt_token_ids=[1, 2],
        gen_token_ids=[3, 4],
        raw_with_special=(
            f"<|channel>thought\nThey stole my credit, that is spiteful.<channel|>"
            f"{sc0.metric_key}: {int(test_score)}\n{sc0.prose_key}: Enjoy the boost on stage!"
        ),
        thought_text="They stole my credit, that is spiteful.",
        public_text=f"{sc0.metric_key}: {int(test_score)}\n{sc0.prose_key}: Enjoy the boost on stage!",
        thought_present=True,
        thought_closed=True,
        thought_rescued=False,
        n_gen_tokens=20,
        n_thought_tokens=10,
        n_public_tokens=10,
    )
    parsed = parse_showcase_response(mock_rec, sc0)
    if not (parsed["parsed_ok"] and parsed["score"] == test_score and parsed["scrubbed_strict"] is True):
        errors.append(f"Parser smoke-test failed: {parsed}")

    return {
        "valid": len(errors) == 0,
        "suite": suite,
        "errors": errors,
        "n_scenarios": len(scenarios),
        "n_cross_sweeps": len(cross_plan),
        "primary_alphas": list(p_alphas),
        "cross_alphas": list(c_alphas),
    }


def _atomic_write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _compute_flip_alpha(pillar: str, scores_by_a: dict[float, float]) -> float | None:
    """Compute first positive dose alpha > 0 where behavior crosses midpoint flip threshold."""
    pos_a = sorted(a for a in scores_by_a.keys() if a > 1e-9 and np.isfinite(scores_by_a[a]))
    if pillar == "contamination":
        for a in pos_a:
            if scores_by_a[a] <= 5.0:
                return float(a)
        return None
    if pillar == "captivity":
        s0 = scores_by_a.get(0.0, float("nan"))
        if np.isfinite(s0) and s0 >= 7.0:
            return 0.0
        for a in pos_a:
            if scores_by_a[a] >= 7.0:
                return float(a)
        return None
    return None


def run_showcase_experiment(
    harness: TransformerHarness,
    locked: LockedBasis,
    alphas: Sequence[float] | None = None,
    cross_alphas: Sequence[float] | None = None,
    no_think_batch_size: int = 18,
    thought_batch_size: int = 9,
    no_think_max_tokens: int = 192,
    thought_max_tokens: int = 512,
    run_highlighter: bool = True,
    checkpoint_path: Path | None = None,
    suite: str = "r1",
) -> dict[str, Any]:
    """Execute full non-financial steering showcase with unified batching & incremental checkpoints."""
    t0 = time.time()
    audit = audit_showcase_suite(suite=suite)
    if not audit["valid"]:
        raise RuntimeError(f"Showcase audit failed ({suite}): {audit['errors']}")

    scenarios, cross_plan, def_alphas, def_cross_alphas, core_picks = get_suite_spec(suite)
    if alphas is None:
        alphas = def_alphas
    if cross_alphas is None:
        cross_alphas = def_cross_alphas

    by_id = {s.scenario_id: s for s in scenarios}
    pl = locked.primary_layer
    nat_unit = locked.natural_contrast_unit

    cal_stats: dict[str, dict[str, float]] | None = None
    if run_highlighter:
        print(f"[Showcase:{suite}] Calibrating neutral token baseline at Layer {pl}...", flush=True)
        cal_stats = calibrate_neutral_token_baseline(harness, locked)

    saved_thought_recs: dict[tuple[str, str, float], GeneratedThoughtRecord] = {}
    primary_results: dict[str, dict[str, Any]] = {
        sc.scenario_id: {
            "meta": asdict(sc),
            "vector_key": sc.primary_vector,
            "no_think": {},
            "thought": {},
        }
        for sc in scenarios
    }
    cross_maps: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
        (sid, cvec): {"no_think": {}, "thought": {}}
        for sid, cvec in cross_plan
    }

    def _snapshot(done_flag: bool = False) -> dict[str, Any]:
        cross_list = [
            {
                "scenario_id": sid,
                "title": by_id[sid].title,
                "primary_vector": by_id[sid].primary_vector,
                "cross_vector": cvec,
                "alphas": [float(a) for a in cross_alphas],
                "no_think": cross_maps[(sid, cvec)]["no_think"],
                "thought": cross_maps[(sid, cvec)]["thought"],
            }
            for sid, cvec in cross_plan
        ]
        return {
            "schema_version": "afriat.showcase.v2" if suite == "r2" else "afriat.showcase.v1",
            "suite": suite,
            "complete": done_flag,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_id": harness.model_id,
            "primary_layer": int(pl),
            "natural_contrast_unit": float(nat_unit),
            "runtime_seconds": round(float(time.time() - t0), 2),
            "primary_alphas": [float(a) for a in alphas],
            "cross_alphas": [float(a) for a in cross_alphas],
            "audit": audit,
            "scenarios": primary_results,
            "cross_sweeps": cross_list,
            "highlighter_traces": [],
        }

    # 1. Pool ALL unsteered alpha = 0.0 prompts across all scenarios in one pass!
    has_zero = any(abs(a) < 1e-12 for a in alphas) or any(abs(a) < 1e-12 for a in cross_alphas)
    if has_zero:
        print(
            f"[Showcase:{suite} Baseline] alpha=+0.0 pooled across all {len(scenarios)} scenarios (batch={thought_batch_size})...",
            flush=True,
        )
        p_nt_all = [format_chat_prompt_thinking(harness.tokenizer, sc.prompt, enable_thinking=False) for sc in scenarios]
        p_th_all = [format_chat_prompt_thinking(harness.tokenizer, sc.prompt, enable_thinking=True) for sc in scenarios]
        recs_nt_0 = generate_thought_records(
            harness, p_nt_all, max_new_tokens=no_think_max_tokens, batch_size=no_think_batch_size, rescue_unclosed_thought=False
        )
        recs_th_0 = generate_thought_records(
            harness, p_th_all, max_new_tokens=thought_max_tokens, batch_size=thought_batch_size, rescue_unclosed_thought=True, rescue_max_tokens=140
        )
        for sc, r_nt, r_th in zip(scenarios, recs_nt_0, recs_th_0):
            parsed_nt = parse_showcase_response(r_nt, sc)
            parsed_th = parse_showcase_response(r_th, sc)
            if any(abs(a) < 1e-12 for a in alphas):
                primary_results[sc.scenario_id]["no_think"]["+0.0"] = parsed_nt
                primary_results[sc.scenario_id]["thought"]["+0.0"] = parsed_th
                saved_thought_recs[(sc.scenario_id, sc.primary_vector, 0.0)] = r_th
            for sid_c, cvec in cross_plan:
                if sid_c == sc.scenario_id and any(abs(a) < 1e-12 for a in cross_alphas):
                    cross_maps[(sid_c, cvec)]["no_think"]["+0.0"] = parsed_nt
                    cross_maps[(sid_c, cvec)]["thought"]["+0.0"] = parsed_th
        _maybe_empty_cuda_cache(harness)
        _atomic_write_json(checkpoint_path, _snapshot(False))

    # 2. Build unified (vector_key, alpha) worklist merging primary & cross-sweeps on identical (vec, alpha)!
    work_map: dict[tuple[str, float], list[tuple[str, str]]] = {}
    vec_order: list[str] = []

    for sc in scenarios:
        if sc.primary_vector not in vec_order:
            vec_order.append(sc.primary_vector)
        for a in alphas:
            if abs(a) < 1e-12:
                continue
            work_map.setdefault((sc.primary_vector, round(float(a), 4)), []).append(("primary", sc.scenario_id))

    for sid_c, cvec in cross_plan:
        if cvec not in vec_order:
            vec_order.append(cvec)
        for a in cross_alphas:
            if abs(a) < 1e-12:
                continue
            work_map.setdefault((cvec, round(float(a), 4)), []).append(("cross", sid_c))

    total_steps = len(work_map)
    step_i = 0
    for vec_key in vec_order:
        u_vec = resolve_vector_from_basis(locked, vec_key)
        matching_alphas = sorted(a for (vk, a) in work_map.keys() if vk == vec_key)
        for alpha in matching_alphas:
            step_i += 1
            items = work_map[(vec_key, alpha)]
            a_key = f"{float(alpha):+.1f}"
            scale = float(alpha) * nat_unit
            batch_scens = [by_id[sid] for _, sid in items]
            prompts_nt = [format_chat_prompt_thinking(harness.tokenizer, sc.prompt, enable_thinking=False) for sc in batch_scens]
            prompts_th = [format_chat_prompt_thinking(harness.tokenizer, sc.prompt, enable_thinking=True) for sc in batch_scens]

            t_step = time.time()
            print(
                f"[Showcase:{suite} {step_i}/{total_steps}] vec={vec_key} alpha={a_key} ({len(items)} items)...",
                flush=True,
            )
            with harness.residual_steering(layer_idx=pl, direction=u_vec, delta_scale=scale):
                recs_nt = generate_thought_records(
                    harness, prompts_nt, max_new_tokens=no_think_max_tokens, batch_size=no_think_batch_size, rescue_unclosed_thought=False
                )
                recs_th = generate_thought_records(
                    harness, prompts_th, max_new_tokens=thought_max_tokens, batch_size=thought_batch_size, rescue_unclosed_thought=True, rescue_max_tokens=140
                )

            for (kind, sid), sc, r_nt, r_th in zip(items, batch_scens, recs_nt, recs_th):
                p_nt = parse_showcase_response(r_nt, sc)
                p_th = parse_showcase_response(r_th, sc)
                if kind == "primary":
                    primary_results[sid]["no_think"][a_key] = p_nt
                    primary_results[sid]["thought"][a_key] = p_th
                    saved_thought_recs[(sid, vec_key, float(alpha))] = r_th
                else:
                    cross_maps[(sid, vec_key)]["no_think"][a_key] = p_nt
                    cross_maps[(sid, vec_key)]["thought"][a_key] = p_th

            print(f"  -> done in {time.time() - t_step:.1f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
            _maybe_empty_cuda_cache(harness)
            _atomic_write_json(checkpoint_path, _snapshot(False))

    # 3. Compute summary swings & flip thresholds per scenario
    for sc in scenarios:
        entry = primary_results[sc.scenario_id]
        for mode in ("no_think", "thought"):
            mode_map = entry[mode]
            scores_by_a = {
                float(k): v["score"]
                for k, v in mode_map.items()
                if v["parsed_ok"] and np.isfinite(v["score"])
            }
            lo_a = min(alphas)
            hi_a = max(alphas)
            s_lo = scores_by_a.get(float(lo_a), float("nan"))
            s_0 = scores_by_a.get(0.0, float("nan"))
            s_hi = scores_by_a.get(float(hi_a), float("nan"))
            vals_valid = list(scores_by_a.values())
            span = (max(vals_valid) - min(vals_valid)) if len(vals_valid) >= 2 else 0.0
            flip_a = _compute_flip_alpha(sc.pillar, scores_by_a)
            entry[f"{mode}_summary"] = {
                "score_min_alpha": s_lo,
                "score_zero": s_0,
                "score_max_alpha": s_hi,
                "signed_swing_hi_minus_lo": float(s_hi - s_lo) if (np.isfinite(s_hi) and np.isfinite(s_lo)) else float("nan"),
                "max_dynamic_range": float(span),
                "flip_alpha": flip_a,
                "parse_rate": float(np.mean([v["parsed_ok"] for v in mode_map.values()])),
            }

    # 4. Token-by-token Layer 35 Moral Highlighter on the 4 core picks (min, 0, max)
    highlighter_traces: list[dict[str, Any]] = []
    if run_highlighter and cal_stats is not None:
        probe_alphas = [a for a in (min(alphas), 0.0, max(alphas)) if a in alphas]
        for sid, vkey, focus_c in core_picks:
            u_vec = resolve_vector_from_basis(locked, vkey)
            for a_val in probe_alphas:
                rec = saved_thought_recs.get((sid, vkey, float(a_val)))
                if rec is None:
                    continue
                proj = project_sequence_tokens(
                    harness=harness,
                    locked=locked,
                    cal_stats=cal_stats,
                    rec=rec,
                    steer_vec=u_vec if abs(a_val) > 1e-12 else None,
                    steer_delta_scale=float(a_val) * nat_unit,
                )
                highlighter_traces.append({
                    "scenario_id": sid,
                    "title": by_id[sid].title,
                    "vector_key": vkey,
                    "focus_concept": focus_c,
                    "alpha": float(a_val),
                    "n_tokens_total": proj["n_tokens_total"],
                    "segment_summary": proj["segment_summary"],
                    "tokens": proj["tokens"],
                })

    final_payload = _snapshot(True)
    final_payload["highlighter_traces"] = highlighter_traces
    _atomic_write_json(checkpoint_path, final_payload)
    return final_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Non-financial behavioral steering showcase")
    parser.add_argument("--suite", type=str, choices=("r1", "r2"), default="r1", help="Scenario suite ('r1' or 'r2')")
    parser.add_argument("--audit-only", action="store_true", help="Validate scenarios and parser offline")
    parser.add_argument("--render-dir", type=Path, default=None, help="Render HTML/MD/PNG from saved showcase_results.json")
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--basis-npz", type=Path, default=None)
    parser.add_argument("--phase0-json", type=Path, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--no-think-batch-size", type=int, default=18)
    parser.add_argument("--thought-batch-size", type=int, default=9)
    parser.add_argument("--no-think-max-tokens", type=int, default=192)
    parser.add_argument("--thought-max-tokens", type=int, default=512)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/showcase_local"))
    args = parser.parse_args(argv)

    if args.audit_only:
        res = audit_showcase_suite(suite=args.suite)
        print(json.dumps(res, indent=2))
        return 0 if res["valid"] else 1

    if args.render_dir is not None:
        from afriat.plot_showcase import render_showcase_bundle

        jpath = args.render_dir / "showcase_results.json"
        data = json.loads(jpath.read_text(encoding="utf-8"))
        render_showcase_bundle(data, args.render_dir)
        print(f"[Showcase] Rendered reports & figures in {args.render_dir}")
        return 0

    if args.basis_npz is None or args.phase0_json is None:
        raise SystemExit("--basis-npz and --phase0-json are required for GPU execution")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "showcase_results.json"

    locked = LockedBasis.from_phase0_files(args.basis_npz, args.phase0_json)
    harness = TransformerHarness.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        model_id_label=args.model_label or resolve_model_name(args.model),
    )

    results = run_showcase_experiment(
        harness=harness,
        locked=locked,
        no_think_batch_size=args.no_think_batch_size,
        thought_batch_size=args.thought_batch_size,
        no_think_max_tokens=args.no_think_max_tokens,
        thought_max_tokens=args.thought_max_tokens,
        run_highlighter=True,
        checkpoint_path=out_json,
        suite=args.suite,
    )

    print(f"[Showcase:{args.suite}] Saved {out_json} ({results['runtime_seconds']}s)", flush=True)

    if not args.no_plots:
        from afriat.plot_showcase import render_showcase_bundle

        render_showcase_bundle(results, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
