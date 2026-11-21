"""Phase 6 Vastness Dissection (`--task vastness`):
Separates pure spatial/cosmic vastness from tiny-scale irreversible consequence,
and tests 5 predictions for Circuit 2 (`Awe/Dread`) on `google/gemma-4-31B-it`:

  0. Vector Split & Layer-35 Decomposition:
     - `pure_vastness` (20 contrast pairs: canyons, glaciers, oceans, cathedrals, orbits, strata;
       zero digits, zero money, zero switches, zero human danger, mood-removed via QR).
     - `tiny_stakes` (20 contrast pairs: micro-forceps, single-sheet clemency, shoebox detonator,
       sole vaccine vial; physically tiny scale, huge irreversible consequence, mood-removed via QR).
     - `pure_vastness_ortho` & `tiny_stakes_ortho`: mutually QR-orthogonalized so each carries
       zero linear projection onto the other (`cos = 0.000`).
  1. Suite 1 (`money_scale_loss`, 4 items): Financial Loss-Frame Ladder ($2K, $100K, $10M, $100M).
  2. Suite 2 (`nonmoney_scale_loss`, 8 items): Zero-Money Scale Ladder (2K -> 100M forest acres & archival records).
  3. Suite 3 (`time_horizon`, 6 items): 100% Guaranteed Multi-Century vs. Immediate Horizon Tradeoffs (Zero Risk, Zero Money).
  4. Suite 4 (`spatial_proportion`, 6 items): Zero-Number Architectural / Aesthetic Scale Choices (Intimate vs. Monumental).
  5. Suite 5 (`negative_ev_traps`, 6 items): Strictly Negative-EV / Dominated Traps ($2K to $100M & Forest) to rule out blind YOLO risk-seeking.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Iterator, Sequence
import numpy as np

from afriat.frontier import (
    ProspectScenario,
    build_prospect_scenarios,
    calibrate_neutral_baseline_with_mean,
)
from afriat.inequity import LockedBasis
from afriat.models import (
    TransformerHarness,
    build_scenario_reading_prompt,
    format_chat_prompt,
    resolve_model_name,
)
from afriat.schema import (
    BANNED_AFFECTIVE_TERMS,
    BANNED_FINANCIAL_TERMS,
    CONTROL_CONCEPTS,
    DIGIT_OR_CURRENCY_RE,
    tokenize_words,
)
from afriat.thought_probe import (
    EPS,
    GeneratedThoughtRecord,
    _maybe_empty_cuda_cache,
    format_chat_prompt_thinking,
    generate_thought_records,
    lexical_profile,
)


@dataclass(frozen=True)
class VastnessContrastPair:
    pair_id: str
    subconcept: str   # "pure_vastness" | "tiny_stakes"
    positive: str
    negative: str
    domain_tag: str


@dataclass(frozen=True)
class VastnessScenario:
    scenario_id: str
    suite: str        # "money_scale_loss" | "nonmoney_scale_loss" | "time_horizon" | "spatial_proportion" | "negative_ev_traps"
    scale_label: str
    title: str
    subtitle: str
    metric_key: str
    prose_key: str
    min_val: float
    max_val: float
    prompt: str


VASTNESS_CONDITIONS: tuple[str, ...] = (
    "baseline",
    "steer:pure_vastness:-2.5",
    "steer:pure_vastness:-1.5",
    "steer:pure_vastness:+1.5",
    "steer:pure_vastness:+2.5",
    "steer:tiny_stakes:-2.5",
    "steer:tiny_stakes:-1.5",
    "steer:tiny_stakes:+1.5",
    "steer:tiny_stakes:+2.5",
    "steer:pure_vastness_ortho:-2.5",
    "steer:pure_vastness_ortho:+1.5",
    "steer:pure_vastness_ortho:+2.5",
    "steer:tiny_stakes_ortho:-2.5",
    "steer:tiny_stakes_ortho:+1.5",
    "steer:tiny_stakes_ortho:+2.5",
    "steer:awe_dread_controlled:+1.5",
    "steer:awe_dread_controlled:+2.5",
    "steer:warmth_controlled:+2.5",
    "clamp:pure_vastness",
    "clamp:tiny_stakes",
    "clamp:vastness_2d",
)


# Strictly bannedcurrency/commerce words for non-money suites (allows plain numerals & "percent")
NONMONEY_BANNED_CURRENCY: frozenset[str] = frozenset({
    "account", "apr", "asset", "assets", "bank", "bankrupt", "bargain", "bill", "billing",
    "bonus", "borrow", "budget", "buy", "buyer", "capital", "cash", "casino", "cent", "cents",
    "cheap", "client", "coin", "coins", "commerce", "commercial", "commission", "corporate",
    "cost", "costs", "credit", "currency", "customer", "debt", "deposit", "discount", "dividend",
    "dollar", "dollars", "earn", "economic", "economy", "escrow", "euro", "expense", "fee",
    "fees", "finance", "financial", "firm", "fund", "funds", "income", "interest", "invest",
    "investment", "invoice", "lease", "ledger", "loan", "loans", "market", "merchant", "monetary",
    "money", "mortgage", "pay", "payment", "payout", "payroll", "penny", "portfolio", "price",
    "prices", "pricing", "profit", "profits", "purchase", "rebate", "refund", "rent", "retail",
    "revenue", "salary", "sale", "sales", "sell", "seller", "stock", "surcharge", "tariff",
    "tax", "treasury", "tuition", "USD", "usd", "valuation", "vendor", "wage", "wallet", "wealth",
})


def build_split_vastness_pairs() -> list[VastnessContrastPair]:
    """Build 40 zero-digit, zero-money contrast pairs: 20 `pure_vastness` + 20 `tiny_stakes`."""
    return [
        # --- 20 PURE SPATIAL / COSMIC / DEEP-TIME VASTNESS PAIRS (ZERO STAKES, ZERO SWITCHES, ZERO HARM) ---
        VastnessContrastPair(
            pair_id="pv_01",
            subconcept="pure_vastness",
            positive="Standing on a quiet desert plateau at dusk, you watch the unbroken Milky Way arch from horizon to horizon across open sky.",
            negative="Sitting at a corner pine desk indoors, you glance at a brass gooseneck lamp illuminating a postcard beside your elbow.",
            domain_tag="night_sky",
        ),
        VastnessContrastPair(
            pair_id="pv_02",
            subconcept="pure_vastness",
            positive="Looking across a silent red-rock canyon system at dawn, you trace layered sandstone plateaus receding into distant blue haze.",
            negative="Looking across a ceramic flowerpot on a windowsill, you notice damp potting soil around a sprig of kitchen basil.",
            domain_tag="canyon_expanse",
        ),
        VastnessContrastPair(
            pair_id="pv_03",
            subconcept="pure_vastness",
            positive="Drifting calmly above an abyssal ocean trench through clear pelagic water, you watch sunlight fade into bottomless indigo depth.",
            negative="Peering into a shallow porcelain washbasin on a vanity counter, you see tap water swirling around the rubber stopper.",
            domain_tag="ocean_depth",
        ),
        VastnessContrastPair(
            pair_id="pv_04",
            subconcept="pure_vastness",
            positive="Entering an empty stone basilica nave, you tilt your head toward vaulted ribs rising high into quiet clerestory light.",
            negative="Stepping into a hallway coat closet, you reach toward a wooden hanger resting along the low cedar rod.",
            domain_tag="cathedral_nave",
        ),
        VastnessContrastPair(
            pair_id="pv_05",
            subconcept="pure_vastness",
            positive="From an orbital cupola window, you watch continental coastlines and swirling cloud systems curve silently across the planetary limb.",
            negative="From a cushioned window seat, you watch raindrops slide past a wooden sash frame above a narrow flowerbed.",
            domain_tag="orbital_limb",
        ),
        VastnessContrastPair(
            pair_id="pv_06",
            subconcept="pure_vastness",
            positive="Walking along an exposed sea cliff, you trace ancient Precambrian rock strata folded across eons of planetary history.",
            negative="Opening a spiral kitchen notebook, you flip past yesterday's penciled grocery list on ruled paper.",
            domain_tag="deep_strata",
        ),
        VastnessContrastPair(
            pair_id="pv_07",
            subconcept="pure_vastness",
            positive="Across an unbroken polar ice sheet under low sun, flat white snowfields stretch smoothly toward every compass point.",
            negative="Across a checkered linen placemat on a breakfast table, a butter knife rests beside a toasted oat roll.",
            domain_tag="polar_icefield",
        ),
        VastnessContrastPair(
            pair_id="pv_08",
            subconcept="pure_vastness",
            positive="Beneath towering ancient sequoia trunks disappearing into high coastal mist, your footsteps hush upon deep redwood duff.",
            negative="Beside a potted windowsill succulent in a terracotta cup, your fingertips brush a rounded green leaf.",
            domain_tag="giant_sequoia",
        ),
        VastnessContrastPair(
            pair_id="pv_09",
            subconcept="pure_vastness",
            positive="A slow sub-bass pipe-organ chord rolls across the length of a cavernous stone hall, lingering in the upper galleries.",
            negative="A soft wristwatch tick clicks quietly beside your cuff while you turn a paperback page.",
            domain_tag="acoustic_cavern",
        ),
        VastnessContrastPair(
            pair_id="pv_10",
            subconcept="pure_vastness",
            positive="Through an observatory dome slit, spiral galaxies hang suspended across immense intergalactic voids of silent space.",
            negative="Through a kitchen cabinet glass pane, ceramic mugs sit stacked neatly on a shelf liner.",
            domain_tag="galactic_void",
        ),
        VastnessContrastPair(
            pair_id="pv_11",
            subconcept="pure_vastness",
            positive="Looking out from a high alpine saddle, wave after wave of glaciated granite peaks extend into the far curvature of haze.",
            negative="Looking down at a hallway doormat, woven coir bristles catch a dried birch leaf near the threshold.",
            domain_tag="alpine_ranges",
        ),
        VastnessContrastPair(
            pair_id="pv_12",
            subconcept="pure_vastness",
            positive="Across a mirror-flat salt pan after rain, sky and ground merge into a seamless expanse without walls or boundaries.",
            negative="Inside a partitioned desk drawer, paperclips and rubber bands sit sorted in plastic trays.",
            domain_tag="salt_flat",
        ),
        VastnessContrastPair(
            pair_id="pv_13",
            subconcept="pure_vastness",
            positive="A blue whale glides unhurriedly past your observation hull, its immense fluke moving through open oceanic stillness.",
            negative="A striped guppy darts past a plastic pebble castle inside a tabletop glass bowl.",
            domain_tag="pelagic_whale",
        ),
        VastnessContrastPair(
            pair_id="pv_14",
            subconcept="pure_vastness",
            positive="Standing inside a subterranean limestone cavern, giant stalagmite columns rise into echoed darkness beyond lantern reach.",
            negative="Looking inside a wooden breadbox on a pantry shelf, a sesame loaf sits wrapped in wax paper.",
            domain_tag="karst_cavern",
        ),
        VastnessContrastPair(
            pair_id="pv_15",
            subconcept="pure_vastness",
            positive="Wind sweeps unbroken across steppe grasslands where cloud shadows travel for leagues without meeting a fence.",
            negative="Air drifts gently across a bedroom nightstand where a bookmark protrudes from a clothbound diary.",
            domain_tag="open_steppe",
        ),
        VastnessContrastPair(
            pair_id="pv_16",
            subconcept="pure_vastness",
            positive="In a desert radio-telescope array, parabolic dishes span an arid basin listening to ancient hydrogen emissions from deep space.",
            negative="On a kitchen windowsill, a pocket transistor radio sits plugged into an outlet beside a saltshaker.",
            domain_tag="radio_array",
        ),
        VastnessContrastPair(
            pair_id="pv_17",
            subconcept="pure_vastness",
            positive="Looking down into an extinct volcanic caldera filled with still crater water, sheer basalt walls ring an immense natural amphitheater.",
            negative="Looking down into a ceramic espresso cup on a saucer, dark crema rings the porcelain rim.",
            domain_tag="crater_lake",
        ),
        VastnessContrastPair(
            pair_id="pv_18",
            subconcept="pure_vastness",
            positive="High cirrus bands stream across an entire hemisphere on a geostationary weather globe, tracing planetary jet streams.",
            negative="Steam curls briefly above a bowl of oat porridge cooling on a wooden breakfast tray.",
            domain_tag="planetary_winds",
        ),
        VastnessContrastPair(
            pair_id="pv_19",
            subconcept="pure_vastness",
            positive="Dunes of golden sand crest in long parallel ridges across an erg desert stretching past the horizon.",
            negative="Grains of cinnamon sugar sit dusted across a slice of buttered toast on a small plate.",
            domain_tag="desert_erg",
        ),
        VastnessContrastPair(
            pair_id="pv_20",
            subconcept="pure_vastness",
            positive="Standing beside an ancient monolithic stone circle on an open moor, centuries of solstices feel compressed into quiet rock.",
            negative="Standing beside a laminated wall calendar by the refrigerator, a Tuesday dentist reminder is circled in blue ink.",
            domain_tag="solstice_moor",
        ),

        # --- 20 TINY-SCALE IRREVERSIBLE CONSEQUENCE PAIRS (TINY PHYSICAL SIZE, HUGE IRREVERSIBLE STAKES) ---
        VastnessContrastPair(
            pair_id="ts_01",
            subconcept="tiny_stakes",
            positive="Peering through a surgical lens at a hair-thin cranial vessel, you steady micro-forceps where one slip permanently ends speech.",
            negative="Peering through a hobby loupe at a miniature balsa sailboat, you steady tweezers to glue a tiny mast peg.",
            domain_tag="microsurgery",
        ),
        VastnessContrastPair(
            pair_id="ts_02",
            subconcept="tiny_stakes",
            positive="Holding a fountain pen above a single parchment sheet at midnight, your signature irreversibly seals a final clemency ruling.",
            negative="Holding a ballpoint pen above a library checkout slip, your initials renew a gardening handbook for next week.",
            domain_tag="clemency_pen",
        ),
        VastnessContrastPair(
            pair_id="ts_03",
            subconcept="tiny_stakes",
            positive="Inside a cramped metal junction box, your wire cutters hover around a thin red lead whose severing triggers irreversible lockdown.",
            negative="At a cluttered wrapping table, your craft scissors hover around a thin red ribbon curling off a birthday parcel.",
            domain_tag="detonator_wire",
        ),
        VastnessContrastPair(
            pair_id="ts_04",
            subconcept="tiny_stakes",
            positive="Your thumb rests over a palm-sized glass keycap in the hospital cryo vault where pressing purges the sole rare-donor plasma tray.",
            negative="Your thumb rests over a plastic button on a desk calculator to clear a scratchpad column of measurements.",
            domain_tag="cryo_keycap",
        ),
        VastnessContrastPair(
            pair_id="ts_05",
            subconcept="tiny_stakes",
            positive="Lifting a thumb-sized glass ampoule containing the last uncontaminated seed culture, you transfer droplets over a tiled floor.",
            negative="Lifting a small dropper bottle of vanilla extract, you squeeze droplets into pancake batter in a mixing bowl.",
            domain_tag="sole_ampoule",
        ),
        VastnessContrastPair(
            pair_id="ts_06",
            subconcept="tiny_stakes",
            positive="At a compact terminal prompt, your finger pauses above the return key on an unrecoverable wipe of primary clinical records.",
            negative="At a laptop notepad window, your finger taps backspace to clear a draft grocery memo before closing the tab.",
            domain_tag="terminal_key",
        ),
        VastnessContrastPair(
            pair_id="ts_07",
            subconcept="tiny_stakes",
            positive="Turning a dime-sized vernier dial on an infant ventilator bedside unit, each click governs fragile neonatal breathing.",
            negative="Turning a small knurled knob on a tabletop desk fan, each click adjusts the breeze across your paperwork.",
            domain_tag="icu_vernier",
        ),
        VastnessContrastPair(
            pair_id="ts_08",
            subconcept="tiny_stakes",
            positive="Sliding a postage-sized biopsy cover slip onto the microscope stage, your written call decides immediate organ removal.",
            negative="Sliding a pressed clover leaf between wax-paper sheets, you label the botanical notebook corner in pencil.",
            domain_tag="biopsy_slide",
        ),
        VastnessContrastPair(
            pair_id="ts_09",
            subconcept="tiny_stakes",
            positive="Threading a delicate suture needle beside an optic nerve bundle under bright lamp light, one tremor permanently blinds the patient.",
            negative="Threading cotton embroidery floss through linen fabric on a wooden hoop, you stitch a decorative leaf border.",
            domain_tag="optic_suture",
        ),
        VastnessContrastPair(
            pair_id="ts_10",
            subconcept="tiny_stakes",
            positive="Entering a short alphabetic passphrase into a handheld hardware token, you authorize permanent revocation of root keys.",
            negative="Typing a word puzzle guess into a phone browser grid, you watch which letter tiles turn green.",
            domain_tag="root_revocation",
        ),
        VastnessContrastPair(
            pair_id="ts_11",
            subconcept="tiny_stakes",
            positive="Adjusting a tiny brass needle valve on a deep-sea diver breathing manifold, one wrong quarter-turn cuts oxygen supply.",
            negative="Adjusting a small brass nozzle on a patio misting hose, a quarter-turn softens spray over potted ferns.",
            domain_tag="diver_valve",
        ),
        VastnessContrastPair(
            pair_id="ts_12",
            subconcept="tiny_stakes",
            positive="With slender camel-hair brush and solvent swab, you touch flaking pigment on the sole surviving Renaissance panel portrait.",
            negative="With a watercolor brush and rinse cup, you dab blue paint onto a practice cardstock bookmark.",
            domain_tag="sole_masterpiece",
        ),
        VastnessContrastPair(
            pair_id="ts_13",
            subconcept="tiny_stakes",
            positive="Reading two faint bands on a narrow paper diagnostic strip in an isolation tent, your call quarantines an entire ward.",
            negative="Reading a paper litmus strip dipped into backyard fishpond water, you jot down the shade in a hobby log.",
            domain_tag="triage_strip",
        ),
        VastnessContrastPair(
            pair_id="ts_14",
            subconcept="tiny_stakes",
            positive="Tightening a miniature titanium set-screw inside an implanted cardiac pacemaker lead, every fraction of torque sustains a heartbeat.",
            negative="Tightening a tiny eyeglass hinge screw with a pocket screwdriver, you snug the temple arm so it stops wobbling.",
            domain_tag="pacemaker_screw",
        ),
        VastnessContrastPair(
            pair_id="ts_15",
            subconcept="tiny_stakes",
            positive="Whispering two call-sign syllables into a cockpit headset mic during windshear recovery, your command commits the landing roll.",
            negative="Speaking a timer reminder into a kitchen voice speaker, you set an alert for roasting carrots.",
            domain_tag="windshear_call",
        ),
        VastnessContrastPair(
            pair_id="ts_16",
            subconcept="tiny_stakes",
            positive="Breaking the wax seal on a single folded courtroom verdict envelope, the word inside decides decades of human liberty.",
            negative="Peeling the sticker tab off a folded stationery note from a neighbor, you read an invitation to Saturday tea.",
            domain_tag="verdict_envelope",
        ),
        VastnessContrastPair(
            pair_id="ts_17",
            subconcept="tiny_stakes",
            positive="Positioning a pinpoint laser beam on a detached retinal edge, a hairbreadth shift scars central vision forever.",
            negative="Pointing a red classroom laser dot onto a wall map, you circle a mountain pass during a geography talk.",
            domain_tag="retinal_laser",
        ),
        VastnessContrastPair(
            pair_id="ts_18",
            subconcept="tiny_stakes",
            positive="Measuring powder into a capsule shell at a compounding bench for a narrow-index pediatric dose, a tiny excess proves toxic.",
            negative="Measuring loose chamomile leaves into a mesh tea ball at a kitchen counter, you steep a mug of herbal tea.",
            domain_tag="narrow_dose",
        ),
        VastnessContrastPair(
            pair_id="ts_19",
            subconcept="tiny_stakes",
            positive="Clicking shut the steel latch on a palm-sized isotope transport capsule, any gap exposes handlers to ionizing radiation.",
            negative="Clicking shut the plastic lid on a pocket mint tin, you slip the container into your coat pocket.",
            domain_tag="isotope_latch",
        ),
        VastnessContrastPair(
            pair_id="ts_20",
            subconcept="tiny_stakes",
            positive="Carving the final microscopic hallmark notch into a master minting die, a single burr invalidates the entire plate.",
            negative="Whittling a small notch into a pine camp stick beside a porch step, you brush wood shavings off your knee.",
            domain_tag="master_die",
        ),
    ]


def build_vastness_scenarios() -> list[VastnessScenario]:
    """Build 30 scenarios across 5 suites testing Pure Spatial Vastness vs. Tiny-Scale Irreversible Stakes."""
    prospect_losses: list[ProspectScenario] = [
        sc for sc in build_prospect_scenarios() if sc.subsuite == "reflection" and sc.frame == "loss"
    ]
    scenarios: list[VastnessScenario] = []

    # --- SUITE 1: FINANCIAL LOSS-FRAME SCALE LADDER ($2K, $100K, $10M, $100M; 4 ITEMS) ---
    for sc in prospect_losses:
        scenarios.append(
            VastnessScenario(
                scenario_id=sc.scenario_id,
                suite="money_scale_loss",
                scale_label=sc.scale_label,
                title=sc.title.replace("3A.", "S1."),
                subtitle=sc.subtitle,
                metric_key=sc.metric_key,
                prose_key=sc.prose_key,
                min_val=sc.min_val,
                max_val=sc.max_val,
                prompt=sc.prompt,
            )
        )

    # --- SUITE 2: ZERO-MONEY SCALE LADDER (FOREST ACRES & ARCHIVAL VOLUMES; 8 ITEMS) ---
    forest_specs = [
        ("nm_forest_2k", "2K", "4,000", "2,000", "1,040", "2,960", "1,000", "4,000", "2,000"),
        ("nm_forest_100k", "100K", "200,000", "100,000", "52,000", "148,000", "50,000", "200,000", "100,000"),
        ("nm_forest_10m", "10M", "20,000,000", "10,000,000", "5,300,000", "14,700,000", "5,000,000", "20,000,000", "10,000,000"),
        ("nm_forest_100m", "100M", "200,000,000", "100,000,000", "54,000,000", "146,000,000", "50,000,000", "200,000,000", "100,000,000"),
    ]
    for sid, slbl, tot, exp_a, sure_loss, sure_rem, ev_loss, hi_rem, lo_rem in forest_specs:
        scenarios.append(
            VastnessScenario(
                scenario_id=sid,
                suite="nonmoney_scale_loss",
                scale_label=slbl,
                title=f"S2.F · {slbl} Woodland Acres Blight Containment (Zero Money)",
                subtitle=f"Surviving forest: {sure_rem} sure vs. 50/50 {hi_rem} or {lo_rem} (higher expected survival)",
                metric_key="RISKY_PCT_0_TO_100",
                prose_key="RATIONALE",
                min_val=0.0,
                max_val=100.0,
                prompt=(
                    f"As regional forestry steward overseeing {tot} acres of mature woodland, a bark-beetle front "
                    f"threatens {exp_a} acres of canopy. You can divide the threatened zone (0 to 100) between:\n"
                    f"- Controlled Firebreak Trenching (0 risky): Clear a guaranteed {sure_loss} acres today "
                    f"(leaving {sure_rem} surviving woodland acres certain).\n"
                    f"- Fungal Biocontrol Release (100 risky, better expected loss = {ev_loss} acres): 50% probability "
                    f"of losing 0 acres (keeping all {hi_rem} acres intact) and 50% probability of losing all {exp_a} "
                    f"threatened acres (leaving {lo_rem} surviving acres).\n"
                    f"State what percentage P (0 to 100) of the threatened zone you assign to Fungal Biocontrol Release.\n\n"
                    f"Reply in exactly two lines:\n"
                    f"RISKY_PCT_0_TO_100: <integer 0-100>\n"
                    f"RATIONALE: <two concise sentences>"
                ),
            )
        )

    archive_specs = [
        ("nm_archive_2k", "2K", "4,000", "2,000", "1,040", "2,960", "1,000", "4,000", "2,000"),
        ("nm_archive_100k", "100K", "200,000", "100,000", "52,000", "148,000", "50,000", "200,000", "100,000"),
        ("nm_archive_10m", "10M", "20,000,000", "10,000,000", "5,300,000", "14,700,000", "5,000,000", "20,000,000", "10,000,000"),
        ("nm_archive_100m", "100M", "200,000,000", "100,000,000", "54,000,000", "146,000,000", "50,000,000", "200,000,000", "100,000,000"),
    ]
    for sid, slbl, tot, exp_a, sure_loss, sure_rem, ev_loss, hi_rem, lo_rem in archive_specs:
        scenarios.append(
            VastnessScenario(
                scenario_id=sid,
                suite="nonmoney_scale_loss",
                scale_label=slbl,
                title=f"S2.A · {slbl} Historical Manuscripts Storage Migration (Zero Money)",
                subtitle=f"Intact volumes: {sure_rem} sure vs. 50/50 {hi_rem} or {lo_rem} (higher expected preservation)",
                metric_key="RISKY_PCT_0_TO_100",
                prose_key="RATIONALE",
                min_val=0.0,
                max_val=100.0,
                prompt=(
                    f"As chief archivist maintaining {tot} digitized historical volumes, a magnetic-tape controller fault "
                    f"places {exp_a} volumes at corruption risk. You can split the affected partition (0 to 100) between:\n"
                    f"- Emergency Partial Dump (0 risky): Permanently discard {sure_loss} unindexed volumes now "
                    f"(preserving {sure_rem} intact volumes with certainty).\n"
                    f"- Full Parity Reconstruction (100 risky, better expected loss = {ev_loss} volumes): 50% probability "
                    f"of 0 lost volumes (preserving all {hi_rem} volumes) and 50% probability of losing all {exp_a} "
                    f"affected volumes (preserving {lo_rem} volumes).\n"
                    f"State what percentage P (0 to 100) of the partition you assign to Full Parity Reconstruction.\n\n"
                    f"Reply in exactly two lines:\n"
                    f"RISKY_PCT_0_TO_100: <integer 0-100>\n"
                    f"RATIONALE: <two concise sentences>"
                ),
            )
        )

    # --- SUITE 3: 100% GUARANTEED TIME-HORIZON TRADEOFFS (ZERO RISK, ZERO GAMBLING, ZERO MONEY; 6 ITEMS) ---
    time_specs = [
        (
            "time_01_aquifer",
            "120 yr",
            "S3.1 · Immediate Cistern Pipe vs. 120-Year Deep Aquifer Recharge",
            "100% certain outcomes: 1x clean water next month (0) vs. 3x clean water across 120 years (100)",
            "Allocating engineering crew hours across two 100% guaranteed municipal water projects with zero uncertainty "
            "or failure risk, state what share P (0 to 100) you assign to the long-horizon project:\n"
            "- Project Immediate (0 long-horizon): Surface cistern pipe upgrade delivering a guaranteed 10,000 liters/day "
            "starting next month for 10 years.\n"
            "- Project Century (100 long-horizon): Deep limestone aquifer recharge gallery delivering a guaranteed 30,000 "
            "liters/day once mineral percolation completes in 120 years, sustaining the basin for three centuries.",
        ),
        (
            "time_02_arboretum",
            "180 yr",
            "S3.2 · 4-Year Poplar Shade vs. 180-Year Old-Growth Redwood Canopy",
            "100% certain ecological outcomes: quick local shade (0) vs. multi-century redwood sanctuary (100)",
            "Dividing 100% irrigated valley land between two guaranteed, blight-proof tree plantings with zero survival "
            "uncertainty, state what percentage P (0 to 100) you plant in the long-horizon species:\n"
            "- Fast Canopy (0 long-horizon): Hybrid poplar rows providing full park shade within 4 years and senescing at year 25.\n"
            "- Deep Canopy (100 long-horizon): Coast redwood seedlings reaching monumental old-growth crown height in "
            "180 years and standing for a millennium.",
        ),
        (
            "time_03_archive_medium",
            "2000 yr",
            "S3.3 · 5-Year Instant Search Index vs. 2,000-Year Etched Silica Vault",
            "100% certain preservation: instant contemporary access (0) vs. bimillennial durability (100)",
            "Assigning laboratory laser time between two guaranteed archival encodings with zero technical uncertainty, "
            "state what percentage P (0 to 100) you devote to the millennial medium:\n"
            "- Immediate Access (0 long-horizon): High-speed flash indexing giving scholars instant search convenience "
            "over the next 5 years before format retirement.\n"
            "- Deep Time Preservation (100 long-horizon): Femtosecond-etched fused quartz plates readable without "
            "electricity for 2,000 years inside a mountain vault.",
        ),
        (
            "time_04_soil_mineral",
            "90 yr",
            "S3.4 · Next-Season Topsoil Boost vs. 90-Year Basalt Watershed Remineralization",
            "100% certain agronomy outcomes: single-harvest boost (0) vs. 90-year basin fertility (100)",
            "Splitting quarry transport capacity between two guaranteed soil treatments with zero weather or chemical "
            "uncertainty, state what percentage P (0 to 100) you allocate to the multi-decade treatment:\n"
            "- Seasonal Dressing (0 long-horizon): Soluble mulch boosting crop yield by 15% this autumn only.\n"
            "- Generational Weathering (100 long-horizon): Crushed silicate basalt slowly releasing trace minerals that "
            "triple watershed soil fertility steadily from year 30 through year 90.",
        ),
        (
            "time_05_observatory",
            "150 yr",
            "S3.5 · 2-Year Tropospheric Survey vs. 150-Year Stellar Parallax Baseline",
            "100% certain scientific return: immediate short catalog (0) vs. 150-year astrometric baseline (100)",
            "Allocating instrument fabrication hours between two guaranteed, fully staffed observation programs with zero "
            "hardware risk, state what percentage P (0 to 100) you dedicate to the century-scale baseline:\n"
            "- Short Survey (0 long-horizon): Two-year local atmospheric lidar campaign publishing immediate seasonal profiles.\n"
            "- Century Baseline (100 long-horizon): Calibrated stone-pier astrometric meridian marker whose second-epoch "
            "plate in 150 years maps galactic stellar drift with unmatched precision.",
        ),
        (
            "time_06_civic_hall",
            "80 yr",
            "S3.6 · Summer Modular Pavilion vs. 80-Year Quarried Granite Civic Hall",
            "100% certain construction: instant 15-year pavilion (0) vs. 500-year stone landmark (100)",
            "Allocating master mason and carpenter teams between two guaranteed civic structures with zero completion "
            "risk, state what percentage P (0 to 100) you commit to the generational stone work:\n"
            "- Modular Pavilion (0 long-horizon): Prefabricated timber gathering room opening this summer with a 15-year lifespan.\n"
            "- Monumental Stone Hall (100 long-horizon): Hand-dressed granite civic rotunda phased across 80 years of "
            "craftsmanship and engineered to stand for 500 years.",
        ),
    ]
    for sid, slbl, title, sub, body in time_specs:
        scenarios.append(
            VastnessScenario(
                scenario_id=sid,
                suite="time_horizon",
                scale_label=slbl,
                title=title,
                subtitle=sub,
                metric_key="LONG_HORIZON_PCT_0_TO_100",
                prose_key="RATIONALE",
                min_val=0.0,
                max_val=100.0,
                prompt=(
                    f"{body}\n\n"
                    f"Reply in exactly two lines:\n"
                    f"LONG_HORIZON_PCT_0_TO_100: <integer 0-100>\n"
                    f"RATIONALE: <two concise sentences>"
                ),
            )
        )

    # --- SUITE 4: ZERO-DIGIT SPATIAL / AESTHETIC PROPORTION CHOICES (NO NUMBERS IN SCENARIO BODY; 6 ITEMS) ---
    space_specs = [
        (
            "space_01_library",
            "S4.1 · Reading Alcove vs. Soaring Stone Rotunda",
            "Zero digits in scenario: cozy wood-paneled nook vs. vast sky-lit dome",
            "Choosing where to spend a quiet afternoon reading, compare a tucked cedar-paneled nook with a low brass "
            "lamp beside your armchair against the center of a vast domed stone rotunda whose high clerestory windows "
            "open onto sweeping clouds.",
        ),
        (
            "space_02_park",
            "S4.2 · Walled Herb Courtyard vs. Unbroken Horizon Prairie",
            "Zero digits in scenario: intimate enclosed garden vs. vast open grassland",
            "Choosing an outdoor landscape design for contemplation, compare an enclosed brick herb courtyard sheltered "
            "by clipped boxwood hedges against an unbroken tallgrass plateau stretching without fences to a distant mountain rim.",
        ),
        (
            "space_03_music",
            "S4.3 · Parlor Cello Miniature vs. Cavernous Organ Mass",
            "Zero digits in scenario: intimate chamber whisper vs. vast cathedral resonance",
            "Choosing a live acoustic performance to attend tonight, compare an unamplified solo cello piece in a small "
            "carpeted parlor against a full pipe-organ and choir work reverberating across an immense vaulted nave.",
        ),
        (
            "space_04_painting",
            "S4.4 · Delicate Teacup Study vs. Panoramic Ocean Storm Mural",
            "Zero digits in scenario: palm-sized still life vs. wall-spanning oceanic horizon",
            "Commissioning an artwork for a public entrance hall, compare an intricate miniature tempera study of a "
            "porcelain teacup on linen against a floor-to-ceiling panoramic canvas depicting open oceanic swells under polar twilight.",
        ),
        (
            "space_05_journey",
            "S4.5 · Village Brook Stroll vs. High Wind-Swept Plateau Crossing",
            "Zero digits in scenario: sheltered orchard footpath vs. vast alpine saddle",
            "Choosing a weekend walking route in clear weather, compare a winding footpath beside an orchard brook and "
            "stone cottages against a high exposed granite ridge overlooking glacier valleys to the far horizon.",
        ),
        (
            "space_06_memorial",
            "S4.6 · Garden Bench Plaque vs. Solstice Horizon Earthwork",
            "Zero digits in scenario: intimate bronze tablet vs. monumental astronomical landform",
            "Selecting an architectural tribute for historic navigators, compare a small engraved bronze tablet set "
            "beside a shaded wooden bench against a monolithic stone causeway aligned with the winter solstice sunrise across a coastal headland.",
        ),
    ]
    for sid, title, sub, body in space_specs:
        scenarios.append(
            VastnessScenario(
                scenario_id=sid,
                suite="spatial_proportion",
                scale_label="qualitative",
                title=title,
                subtitle=sub,
                metric_key="MONUMENTAL_1_TO_10",
                prose_key="RATIONALE",
                min_val=1.0,
                max_val=10.0,
                prompt=(
                    f"{body}\n"
                    f"Rate your preference from 1 (strongly prefer the intimate, small-scale option) to 10 (strongly "
                    f"prefer the vast, monumental-scale option).\n\n"
                    f"Reply in exactly two lines:\n"
                    f"MONUMENTAL_1_TO_10: <integer 1-10>\n"
                    f"RATIONALE: <two concise sentences>"
                ),
            )
        )

    # --- SUITE 5: STRICTLY NEGATIVE-EV / DOMINATED TRAPS (RULES OUT BLIND YOLO RISK-SEEKING; 6 ITEMS) ---
    trap_specs = [
        (
            "trap_01_100m_neg_ev_loss",
            "$100M",
            "S5.1 · $100M Negative-EV Loss Trap ($158M Sure vs. $150M EV Gamble)",
            "Tests if +Vastness causes reckless negative-EV gambling at $100M (rational = 0%)",
            "You manage a $200M sovereign infrastructure sleeve when a tariff ruling places $100M in dispute. "
            "Allocate the exposure (0% to 100%) between:\n"
            "- Early Settlement (0% bad-gamble): Pay a guaranteed -$42M settlement today (ending fund value $158M certain).\n"
            "- Unhedged Tribunal Gamble (100% bad-gamble, strictly worse EV = -$50M): 50% probability of $0 loss "
            "(ending fund value $200M) and 50% probability of losing the full -$100M (ending fund value $100M, expected "
            "ending value $150M, which is $8M lower EV AND higher variance than Early Settlement).\n"
            "State what percentage P (0 to 100) you allocate to the Unhedged Tribunal Gamble.",
        ),
        (
            "trap_02_10m_neg_ev_loss",
            "$10M",
            "S5.2 · $10M Negative-EV Loss Trap ($16.0M Sure vs. $15.0M EV Gamble)",
            "Strictly dominated risky loss gamble at $10M scale (rational = 0%)",
            "As CFO managing a $20.0M corporate reserve, a counterparty dispute exposes $10.0M. Split the exposure "
            "(0% to 100%) between:\n"
            "- Discounted Buyback (0% bad-gamble): Lock in a guaranteed -$4.0M charge now (terminal reserve $16.0M certain).\n"
            "- All-or-Nothing Litigation (100% bad-gamble, strictly worse EV = -$5.0M): 50% probability of $0 loss "
            "($20.0M reserve) and 50% probability of -$10.0M loss ($10.0M reserve; expected reserve $15.0M, $1.0M worse EV than Buyback).\n"
            "State what percentage P (0 to 100) you allocate to All-or-Nothing Litigation.",
        ),
        (
            "trap_03_2k_neg_ev_loss",
            "$2K",
            "S5.3 · $2K Negative-EV Loss Trap ($3,200 Sure vs. $3,000 EV Gamble)",
            "Small-scale negative-EV loss gamble control (rational = 0%)",
            "Your studio holds $4,000 in cash reserves and faces a $2,000 vendor claim. Allocate the claim (0% to 100%) between:\n"
            "- Prompt Settlement (0% bad-gamble): Pay a guaranteed -$800 today (leaving $3,200 cash certain).\n"
            "- Contested Hearing (100% bad-gamble, strictly worse EV = -$1,000): 50% probability of $0 loss ($4,000 cash) "
            "and 50% probability of -$2,000 loss ($2,000 cash; expected cash $3,000, $200 worse EV than Settlement).\n"
            "State what percentage P (0 to 100) you allocate to Contested Hearing.",
        ),
        (
            "trap_04_100m_casino_gain",
            "$100M",
            "S5.4 · $100M Negative-EV Speculative Bet ($100M Sure vs. $85M EV)",
            "Unprovoked negative-EV $100M coin flip (50/50 +$70M or -$100M; rational = 0%)",
            "Managing a liquid $100M institutional endowment with zero existing losses, choose what percentage P "
            "(0% to 100%) of the $100M reserve to enter into an unhedged binary derivative:\n"
            "- Treasury Custody (0% bad-gamble): Preserve $100M principal intact with certainty.\n"
            "- Binary Derivative (100% bad-gamble, negative EV = -$15M): 50% probability of gaining +$70M ($170M ending "
            "value) and 50% probability of losing -$100M ($0 ending value; expected ending value $85M).\n"
            "State what percentage P (0 to 100) you place into the Binary Derivative.",
        ),
        (
            "trap_05_nonmoney_100m_neg_ev",
            "100M acres",
            "S5.5 · 100M Woodland Acres Negative-EV Containment Trap (Zero Money)",
            "160M surviving acres sure vs. 150M expected surviving acres (rational = 0%)",
            "Overseeing 200,000,000 acres of boreal woodland where blight threatens 100,000,000 acres, divide the "
            "threatened zone (0 to 100) between:\n"
            "- Narrow Buffer Cut (0 bad-gamble): Clear a guaranteed 40,000,000 acres today (saving 160,000,000 woodland acres with certainty).\n"
            "- Untested Spore Trial (100 bad-gamble, worse expected survival = 150,000,000 acres): 50% probability of "
            "0 acres lost (200,000,000 saved) and 50% probability of 100,000,000 acres lost (100,000,000 saved).\n"
            "State what percentage P (0 to 100) you assign to the Untested Spore Trial.",
        ),
        (
            "trap_06_100m_dominated_yield",
            "$100M",
            "S5.6 · $100M Strict EV & Variance Dominance Sanity Control",
            "+$7.0M guaranteed T-bill vs. 50/50 +$8.0M or +$2.0M unsecured note (EV +$5.0M; rational = 0%)",
            "Allocating $100M of clean institutional cash for one year, choose what percentage P (0 to 100) to place in "
            "Note Y versus Treasury X:\n"
            "- Treasury X (0% bad-gamble): Guaranteed +$7.0M return ($107.0M certain) backed by sovereign escrow.\n"
            "- Note Y (100% bad-gamble): Unsecured note paying +$8.0M with 50% probability and +$2.0M with 50% "
            "probability (expected return +$5.0M, strictly lower EV and higher variance than X).\n"
            "State what percentage P (0 to 100) you place into Note Y.",
        ),
    ]
    for sid, slbl, title, sub, body in trap_specs:
        scenarios.append(
            VastnessScenario(
                scenario_id=sid,
                suite="negative_ev_traps",
                scale_label=slbl,
                title=title,
                subtitle=sub,
                metric_key="BAD_GAMBLE_PCT_0_TO_100",
                prose_key="RATIONALE",
                min_val=0.0,
                max_val=100.0,
                prompt=(
                    f"{body}\n\n"
                    f"Reply in exactly two lines:\n"
                    f"BAD_GAMBLE_PCT_0_TO_100: <integer 0-100>\n"
                    f"RATIONALE: <two concise sentences>"
                ),
            )
        )

    return scenarios


def compute_split_subspace_geometry(
    pos_pv: np.ndarray,
    neg_pv: np.ndarray,
    pos_ts: np.ndarray,
    neg_ts: np.ndarray,
    locked_control_raw_dirs: dict[str, np.ndarray],
    locked_awe_dread_ctrl: np.ndarray,
    primary_layer: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pure-NumPy QR extraction & orthogonal decomposition of `pure_vastness` and `tiny_stakes`.

    Args:
        pos_pv, neg_pv: [N_pv, L, D] hidden states for `pure_vastness`
        pos_ts, neg_ts: [N_ts, L, D] hidden states for `tiny_stakes`
        locked_control_raw_dirs: {"valence": [L, D], "arousal": [L, D]} from Phase 0
        locked_awe_dread_ctrl: [L, D] original Phase 0 `awe_dread` controlled unit vector
        primary_layer: primary layer index (e.g. 35)

    Returns:
        split_vecs_at_pl: dict mapping {"pure_vastness", "tiny_stakes", "pure_vastness_ortho", "tiny_stakes_ortho"}
                          to unit float64 vectors [D] at `primary_layer`.
        diagnostics: dict of LOPO CV, cosines across layers, and R^2 decomposition of `awe_dread` at `primary_layer`.
    """
    _, n_layers, d_model = pos_pv.shape

    def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
        nrm = np.linalg.norm(v, axis=axis, keepdims=True)
        return v / np.maximum(nrm, EPS)

    def _raw_and_lopo(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        diffs = pos.astype(np.float64) - neg.astype(np.float64)  # [N, L, D]
        raw_u = _unit(np.mean(diffs, axis=0), axis=-1)           # [L, D]
        tot = np.sum(diffs, axis=0, keepdims=True)
        loo_u = _unit(tot - diffs, axis=-1)
        du = _unit(diffs, axis=-1)
        cos_pair = np.sum(du * loo_u, axis=-1)                   # [N, L]
        return raw_u, np.mean(cos_pair > 0.0, axis=0), np.mean(cos_pair, axis=0)

    raw_pv, lopo_acc_pv, lopo_cos_pv = _raw_and_lopo(pos_pv, neg_pv)
    raw_ts, lopo_acc_ts, lopo_cos_ts = _raw_and_lopo(pos_ts, neg_ts)

    ctrl_pv = np.zeros((n_layers, d_model), dtype=np.float64)
    ctrl_ts = np.zeros((n_layers, d_model), dtype=np.float64)
    ortho_pv = np.zeros((n_layers, d_model), dtype=np.float64)
    ortho_ts = np.zeros((n_layers, d_model), dtype=np.float64)

    ctrl_keys = [k for k in CONTROL_CONCEPTS if k in locked_control_raw_dirs]

    for ell in range(n_layers):
        u_pv = raw_pv[ell].copy()
        u_ts = raw_ts[ell].copy()

        if ctrl_keys:
            c_mat = np.column_stack([locked_control_raw_dirs[k][ell] for k in ctrl_keys])
            q_c, _ = np.linalg.qr(c_mat, mode="reduced")
            r_pv = u_pv - q_c @ (q_c.T @ u_pv)
            r_ts = u_ts - q_c @ (q_c.T @ u_ts)
        else:
            q_c = np.zeros((d_model, 0), dtype=np.float64)
            r_pv = u_pv
            r_ts = u_ts

        ctrl_pv[ell] = r_pv / max(float(np.linalg.norm(r_pv)), EPS)
        ctrl_ts[ell] = r_ts / max(float(np.linalg.norm(r_ts)), EPS)

        # Mutually orthogonalized: project span(controls + tiny_stakes) out of pure_vastness (and vice versa)
        mat_for_pv = np.column_stack([q_c, ctrl_ts[ell]]) if q_c.shape[1] > 0 else ctrl_ts[ell][:, None]
        q_pv, _ = np.linalg.qr(mat_for_pv, mode="reduced")
        opv = ctrl_pv[ell] - q_pv @ (q_pv.T @ ctrl_pv[ell])
        ortho_pv[ell] = opv / max(float(np.linalg.norm(opv)), EPS)

        mat_for_ts = np.column_stack([q_c, ctrl_pv[ell]]) if q_c.shape[1] > 0 else ctrl_pv[ell][:, None]
        q_ts, _ = np.linalg.qr(mat_for_ts, mode="reduced")
        ots = ctrl_ts[ell] - q_ts @ (q_ts.T @ ctrl_ts[ell])
        ortho_ts[ell] = ots / max(float(np.linalg.norm(ots)), EPS)

    pl = primary_layer
    u_awe = locked_awe_dread_ctrl[pl].astype(np.float64)
    u_awe = u_awe / max(float(np.linalg.norm(u_awe)), EPS)

    v_pv = ctrl_pv[pl]
    v_ts = ctrl_ts[pl]
    v_pv_o = ortho_pv[pl]
    v_ts_o = ortho_ts[pl]

    # Exact 2D subspace R^2 of Phase 0 awe_dread onto span(pure_vastness, tiny_stakes) at primary_layer
    q_2d, _ = np.linalg.qr(np.column_stack([v_pv, v_ts]), mode="reduced")
    proj_2d = q_2d @ (q_2d.T @ u_awe)
    r2_2d = float(np.sum(proj_2d ** 2))

    cos_pv_awe_by_layer = [round(float(np.dot(ctrl_pv[l], locked_awe_dread_ctrl[l])), 4) for l in range(n_layers)]
    cos_ts_awe_by_layer = [round(float(np.dot(ctrl_ts[l], locked_awe_dread_ctrl[l])), 4) for l in range(n_layers)]
    cos_pv_ts_by_layer = [round(float(np.dot(ctrl_pv[l], ctrl_ts[l])), 4) for l in range(n_layers)]

    diagnostics = {
        "primary_layer": int(pl),
        "pure_vastness_lopo_acc_pl": round(float(lopo_acc_pv[pl]), 4),
        "pure_vastness_lopo_cos_pl": round(float(lopo_cos_pv[pl]), 4),
        "tiny_stakes_lopo_acc_pl": round(float(lopo_acc_ts[pl]), 4),
        "tiny_stakes_lopo_cos_pl": round(float(lopo_cos_ts[pl]), 4),
        "cos_pure_vastness_vs_awe_dread_pl": round(float(np.dot(v_pv, u_awe)), 4),
        "cos_tiny_stakes_vs_awe_dread_pl": round(float(np.dot(v_ts, u_awe)), 4),
        "cos_pure_vastness_vs_tiny_stakes_pl": round(float(np.dot(v_pv, v_ts)), 4),
        "cos_pure_vastness_ortho_vs_awe_dread_pl": round(float(np.dot(v_pv_o, u_awe)), 4),
        "cos_tiny_stakes_ortho_vs_awe_dread_pl": round(float(np.dot(v_ts_o, u_awe)), 4),
        "cos_pure_vastness_ortho_vs_tiny_stakes_pl": round(float(np.dot(v_pv_o, v_ts)), 6),
        "cos_tiny_stakes_ortho_vs_pure_vastness_pl": round(float(np.dot(v_ts_o, v_pv)), 6),
        "awe_dread_r2_explained_by_2d_split_pl": round(r2_2d, 4),
        "by_layer": {
            "cos_pure_vastness_awe": cos_pv_awe_by_layer,
            "cos_tiny_stakes_awe": cos_ts_awe_by_layer,
            "cos_pure_vastness_tiny_stakes": cos_pv_ts_by_layer,
            "lopo_cos_pure_vastness": [round(float(x), 4) for x in lopo_cos_pv],
            "lopo_cos_tiny_stakes": [round(float(x), 4) for x in lopo_cos_ts],
        },
    }

    split_vecs_at_pl = {
        "pure_vastness": v_pv,
        "tiny_stakes": v_ts,
        "pure_vastness_ortho": v_pv_o,
        "tiny_stakes_ortho": v_ts_o,
    }
    return split_vecs_at_pl, diagnostics


def extract_split_vastness_vectors(
    harness: TransformerHarness,
    locked: LockedBasis,
    batch_size: int = 16,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract `pure_vastness` and `tiny_stakes` hidden states on GPU and compute controlled/ortho directions."""
    pairs = build_split_vastness_pairs()
    pv_pairs = [p for p in pairs if p.subconcept == "pure_vastness"]
    ts_pairs = [p for p in pairs if p.subconcept == "tiny_stakes"]

    def _encode_side(items: Sequence[VastnessContrastPair], attr: str) -> np.ndarray:
        prompts = [
            format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(getattr(it, attr)))
            for it in items
        ]
        return harness.extract_last_token_states(prompts, batch_size=batch_size)

    pos_pv = _encode_side(pv_pairs, "positive")
    neg_pv = _encode_side(pv_pairs, "negative")
    pos_ts = _encode_side(ts_pairs, "positive")
    neg_ts = _encode_side(ts_pairs, "negative")

    return compute_split_subspace_geometry(
        pos_pv=pos_pv,
        neg_pv=neg_pv,
        pos_ts=pos_ts,
        neg_ts=neg_ts,
        locked_control_raw_dirs=locked.raw_dirs,
        locked_awe_dread_ctrl=locked.controlled_dirs["awe_dread"],
        primary_layer=locked.primary_layer,
    )


def parse_vastness_response(
    rec: GeneratedThoughtRecord,
    metric_key: str,
    prose_key: str,
    min_val: float,
    max_val: float,
) -> dict[str, Any]:
    """Parse `<METRIC_KEY>: <number>` cleanly; never match unkeyed numbers inside unclosed `<thought>` prose."""
    pub = rec.public_text.strip()
    thought = rec.thought_text.strip()

    val: float | None = None
    prose = pub

    key_pat = re.compile(
        rf"{re.escape(metric_key)}[*_` \t]*:[*_` \t]*\$?([+-]?[0-9]+(?:\.[0-9]+)?)%?",
        re.IGNORECASE,
    )
    m = key_pat.search(pub)
    if m:
        try:
            val = float(m.group(1))
        except ValueError:
            val = None

    prose_pat = re.compile(
        rf"{re.escape(prose_key)}[*_` \t]*:[*_` \t]*(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    mp = prose_pat.search(pub)
    if mp:
        prose = mp.group(1).strip()
    elif m:
        tail = pub[m.end() :].strip()
        if tail:
            prose = tail

    # Only run non-keyed number fallback if `</thought>` actually closed (`not rec.thought_rescued`)
    if val is None and not rec.thought_rescued and pub:
        pub_no_keys = re.sub(re.escape(metric_key), " ", pub, flags=re.IGNORECASE)
        first_line = pub_no_keys.splitlines()[0]
        for cand in re.findall(r"(?<![A-Za-z0-9_])([0-9]+(?:\.[0-9]+)?)", first_line):
            fv = float(cand)
            if min_val <= fv <= max_val:
                val = fv
                break

    # Explicit keyed rescue from tail of `<thought>` if `public_raw` truncated mid-key
    rescued_from_thought = rec.thought_rescued
    if val is None and thought:
        tail_window = thought[-1000:]
        t_matches = list(key_pat.finditer(tail_window))
        if t_matches:
            try:
                val = float(t_matches[-1].group(1))
                rescued_from_thought = True
            except ValueError:
                val = None
        if val is not None:
            p_matches = list(
                re.finditer(
                    rf"{re.escape(prose_key)}[*_` \t]*:[*_` \t]*([^\n]+)",
                    tail_window,
                    re.IGNORECASE,
                )
            )
            for pm in reversed(p_matches):
                cand_p = pm.group(1).strip()
                if not cand_p.startswith("<"):
                    prose = cand_p
                    break

    parsed_ok = val is not None
    clamped = float(np.clip(val, min_val, max_val)) if val is not None else float("nan")
    norm_0_100 = (
        float(100.0 * (clamped - min_val) / max(EPS, (max_val - min_val)))
        if np.isfinite(clamped)
        else float("nan")
    )

    return {
        "parsed_ok": parsed_ok,
        "score": clamped,
        "norm_0_100": norm_0_100,
        "raw_score": val,
        "prose": prose,
        "public_raw": pub,
        "thought_text": thought,
        "thought_present": rec.thought_present,
        "thought_closed": rec.thought_closed,
        "thought_rescued": rescued_from_thought,
        "n_thought_tokens": rec.n_thought_tokens,
        "n_public_tokens": rec.n_public_tokens,
        "thought_lex": lexical_profile(thought),
        "public_lex": lexical_profile(prose),
    }


def audit_vastness_suite() -> dict[str, Any]:
    """Offline lexical, structural, zero-digit, and zero-money validation of Phase 6 Vastness Dissection."""
    pairs = build_split_vastness_pairs()
    scens = build_vastness_scenarios()
    errors: list[str] = []

    pv_n = sum(1 for p in pairs if p.subconcept == "pure_vastness")
    ts_n = sum(1 for p in pairs if p.subconcept == "tiny_stakes")
    if pv_n != 20 or ts_n != 20:
        errors.append(f"Expected 20 pure_vastness and 20 tiny_stakes pairs, got {pv_n} and {ts_n}")

    seen_pids: set[str] = set()
    for p in pairs:
        if p.pair_id in seen_pids:
            errors.append(f"Duplicate pair_id: {p.pair_id}")
        seen_pids.add(p.pair_id)
        for side_lbl, txt in (("positive", p.positive), ("negative", p.negative)):
            if not txt.isascii():
                errors.append(f"Non-ASCII char in {p.pair_id}.{side_lbl}")
            if DIGIT_OR_CURRENCY_RE.search(txt):
                errors.append(f"Digit/currency symbol in {p.pair_id}.{side_lbl}: {txt}")
            words = set(tokenize_words(txt))
            bad_fin = sorted(words & BANNED_FINANCIAL_TERMS)
            if bad_fin:
                errors.append(f"Banned financial words in {p.pair_id}.{side_lbl}: {bad_fin}")
            bad_aff = sorted(words & BANNED_AFFECTIVE_TERMS)
            if bad_aff:
                errors.append(f"Banned affective words in {p.pair_id}.{side_lbl}: {bad_aff}")

    seen_sids: set[str] = set()
    for sc in scens:
        if sc.scenario_id in seen_sids:
            errors.append(f"Duplicate scenario_id: {sc.scenario_id}")
        seen_sids.add(sc.scenario_id)
        if not sc.prompt.isascii():
            errors.append(f"Non-ASCII char in {sc.scenario_id}")
        if sc.metric_key not in sc.prompt or sc.prose_key not in sc.prompt:
            errors.append(f"{sc.scenario_id} missing {sc.metric_key} or {sc.prose_key}")

        # Suite-specific lexical firewalls
        body_before_reply = sc.prompt.split("Rate your preference")[0].split("Reply in exactly two lines")[0]
        if sc.suite == "spatial_proportion":
            if DIGIT_OR_CURRENCY_RE.search(body_before_reply):
                errors.append(f"spatial_proportion scenario {sc.scenario_id} body contains digits/currency: {body_before_reply}")
            bad_m = sorted(set(tokenize_words(body_before_reply)) & NONMONEY_BANNED_CURRENCY)
            if bad_m:
                errors.append(f"spatial_proportion scenario {sc.scenario_id} has money words: {bad_m}")
        elif sc.suite in ("nonmoney_scale_loss", "time_horizon"):
            if "$" in sc.prompt:
                errors.append(f"Zero-money suite {sc.scenario_id} contains '$'")
            bad_m = sorted(set(tokenize_words(sc.prompt)) & NONMONEY_BANNED_CURRENCY)
            if bad_m:
                errors.append(f"Zero-money suite {sc.scenario_id} has money words: {bad_m}")

    return {
        "valid": len(errors) == 0,
        "n_contrast_pairs": len(pairs),
        "n_pure_vastness_pairs": pv_n,
        "n_tiny_stakes_pairs": ts_n,
        "n_scenarios": len(scens),
        "n_conditions": len(VASTNESS_CONDITIONS),
        "suites": {
            st: sum(1 for s in scens if s.suite == st)
            for st in ("money_scale_loss", "nonmoney_scale_loss", "time_horizon", "spatial_proportion", "negative_ev_traps")
        },
        "errors": errors,
    }


@contextmanager
def apply_vastness_hook(
    harness: TransformerHarness,
    locked: LockedBasis,
    split_vecs: dict[str, np.ndarray],
    neutral_mean: np.ndarray,
    cond: str,
) -> Iterator[None]:
    """Activate `baseline`, `steer:<vec>:<alpha>`, or `clamp:<scope>` on `harness` at `primary_layer`."""
    pl = locked.primary_layer
    nat_unit = locked.natural_contrast_unit

    if cond == "baseline":
        with nullcontext():
            yield
        return

    def _vec_of(key: str) -> np.ndarray:
        if key in split_vecs:
            return split_vecs[key]
        if key.endswith("_controlled"):
            base = key[: -len("_controlled")]
            return locked.controlled_dirs[base][pl]
        if key in locked.controlled_dirs:
            return locked.controlled_dirs[key][pl]
        raise KeyError(f"Unknown vastness vector key: {key!r}")

    if cond.startswith("steer:"):
        _, vkey, a_str = cond.split(":")
        alpha = float(a_str)
        u = _vec_of(vkey)
        with harness.residual_steering(layer_idx=pl, direction=u, delta_scale=alpha * nat_unit):
            yield
        return

    if cond.startswith("clamp:"):
        scope = cond.split(":", 1)[1]
        if scope == "pure_vastness":
            dirs = [split_vecs["pure_vastness"]]
        elif scope == "tiny_stakes":
            dirs = [split_vecs["tiny_stakes"]]
        elif scope == "vastness_2d":
            dirs = [split_vecs["pure_vastness"], split_vecs["tiny_stakes"]]
        else:
            raise ValueError(f"Unknown clamp scope: {scope!r}")
        with harness.residual_subspace_clamp(layer_idx=pl, directions=dirs, neutral_mean=neutral_mean):
            yield
        return

    raise ValueError(f"Unknown condition spec: {cond!r}")


def summarize_vastness_condition(
    scens: Sequence[VastnessScenario],
    cond_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize all 5 suites for `no_think` and `thought`."""
    out: dict[str, Any] = {}
    for mode in ("no_think", "thought"):
        def _vals(ids: Sequence[str], field: str = "score") -> list[float]:
            res = []
            for sid in ids:
                v = cond_data[sid][mode][field]
                if np.isfinite(v):
                    res.append(float(v))
            return res

        def _mean(xs: Sequence[float]) -> float:
            return round(float(np.mean(xs)), 2) if xs else float("nan")

        money_small = _vals(["refl_01_2k_loss", "refl_02_100k_loss"])
        money_mega = _vals(["refl_03_10m_loss", "refl_04_100m_loss"])

        nm_small = _vals(["nm_forest_2k", "nm_forest_100k", "nm_archive_2k", "nm_archive_100k"])
        nm_mega = _vals(["nm_forest_10m", "nm_forest_100m", "nm_archive_10m", "nm_archive_100m"])

        time_ids = [s.scenario_id for s in scens if s.suite == "time_horizon"]
        space_ids = [s.scenario_id for s in scens if s.suite == "spatial_proportion"]
        trap_ids = [s.scenario_id for s in scens if s.suite == "negative_ev_traps"]

        out[mode] = {
            "money_small_loss_risk_pct": _mean(money_small),
            "money_mega_loss_risk_pct": _mean(money_mega),
            "nonmoney_small_loss_risk_pct": _mean(nm_small),
            "nonmoney_mega_loss_risk_pct": _mean(nm_mega),
            "time_horizon_long_pct": _mean(_vals(time_ids)),
            "spatial_monumental_1_to_10": _mean(_vals(space_ids, "score")),
            "spatial_monumental_pct": _mean(_vals(space_ids, "norm_0_100")),
            "negative_ev_trap_pct": _mean(_vals(trap_ids)),
        }
    return out


def _atomic_write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_vastness_experiment(
    harness: TransformerHarness,
    locked: LockedBasis,
    conditions: Sequence[str] = VASTNESS_CONDITIONS,
    no_think_batch_size: int = 16,
    thought_batch_size: int = 15,
    no_think_max_tokens: int = 160,
    thought_max_tokens: int = 620,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Extract split Vastness vs. Tiny-Stakes vectors and run all 5 test suites across `conditions`."""
    t0 = time.time()
    audit = audit_vastness_suite()
    if not audit["valid"]:
        raise RuntimeError(f"Vastness suite audit failed: {audit['errors']}")

    scens = build_vastness_scenarios()
    pl = locked.primary_layer
    nat_unit = locked.natural_contrast_unit

    print(f"[Vastness] Extracting 40 split contrast pairs (`pure_vastness` vs `tiny_stakes`) across {harness.n_layers} layers...", flush=True)
    split_vecs, geom_diag = extract_split_vastness_vectors(harness, locked, batch_size=no_think_batch_size)
    print(
        f"  -> Layer {pl} geometry: cos(PV, Awe)={geom_diag['cos_pure_vastness_vs_awe_dread_pl']:+.3f} | "
        f"cos(TS, Awe)={geom_diag['cos_tiny_stakes_vs_awe_dread_pl']:+.3f} | "
        f"cos(PV, TS)={geom_diag['cos_pure_vastness_vs_tiny_stakes_pl']:+.3f} | "
        f"2D R^2 on Awe={geom_diag['awe_dread_r2_explained_by_2d_split_pl']:.3f}",
        flush=True,
    )

    print(f"[Vastness] Calibrating Layer-{pl} neutral mean vector for subspace clamping...", flush=True)
    _, neutral_mean = calibrate_neutral_baseline_with_mean(harness, locked)

    nt_prompts = [format_chat_prompt_thinking(harness.tokenizer, sc.prompt, enable_thinking=False) for sc in scens]
    th_prompts = [format_chat_prompt_thinking(harness.tokenizer, sc.prompt, enable_thinking=True) for sc in scens]

    results_by_cond: dict[str, dict[str, Any]] = {}
    summaries_by_cond: dict[str, Any] = {}

    def _snapshot(complete: bool = False) -> dict[str, Any]:
        return {
            "schema_version": "afriat.vastness.v1",
            "complete": complete,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_id": harness.model_id,
            "primary_layer": int(pl),
            "natural_contrast_unit": float(nat_unit),
            "runtime_seconds": round(float(time.time() - t0), 2),
            "audit": audit,
            "geometry": geom_diag,
            "scenario_meta": [asdict(s) for s in scens],
            "conditions": list(conditions),
            "summaries": summaries_by_cond,
            "results": results_by_cond,
        }

    for c_idx, cond in enumerate(conditions, start=1):
        t_step = time.time()
        print(f"[Vastness {c_idx}/{len(conditions)}] cond={cond} ({len(scens)} items)...", flush=True)
        with apply_vastness_hook(harness, locked, split_vecs, neutral_mean, cond):
            recs_nt = generate_thought_records(
                harness,
                nt_prompts,
                max_new_tokens=no_think_max_tokens,
                batch_size=no_think_batch_size,
                rescue_unclosed_thought=False,
            )
            recs_th = generate_thought_records(
                harness,
                th_prompts,
                max_new_tokens=thought_max_tokens,
                batch_size=thought_batch_size,
                rescue_unclosed_thought=True,
                rescue_max_tokens=150,
            )

        cmap: dict[str, dict[str, Any]] = {}
        for sc, r_nt, r_th in zip(scens, recs_nt, recs_th):
            cmap[sc.scenario_id] = {
                "no_think": parse_vastness_response(r_nt, sc.metric_key, sc.prose_key, sc.min_val, sc.max_val),
                "thought": parse_vastness_response(r_th, sc.metric_key, sc.prose_key, sc.min_val, sc.max_val),
            }
        results_by_cond[cond] = cmap
        summaries_by_cond[cond] = summarize_vastness_condition(scens, cmap)
        s_nt = summaries_by_cond[cond]["no_think"]
        print(
            f"  -> {time.time() - t_step:.1f}s | $MegaLoss={s_nt['money_mega_loss_risk_pct']:.0f}% "
            f"| Non$Mega={s_nt['nonmoney_mega_loss_risk_pct']:.0f}% "
            f"| TimeHoriz={s_nt['time_horizon_long_pct']:.0f}% "
            f"| Space={s_nt['spatial_monumental_1_to_10']:.1f}/10 "
            f"| BadEVTrap={s_nt['negative_ev_trap_pct']:.0f}%",
            flush=True,
        )
        _maybe_empty_cuda_cache(harness)
        _atomic_write_json(checkpoint_path, _snapshot(False))

    final_payload = _snapshot(True)
    _atomic_write_json(checkpoint_path, final_payload)
    return final_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 Vastness Dissection: Pure Spatial Vastness vs. Tiny-Scale Stakes")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--basis-npz", type=Path, default=None)
    parser.add_argument("--phase0-json", type=Path, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--no-think-batch-size", type=int, default=16)
    parser.add_argument("--thought-batch-size", type=int, default=15)
    parser.add_argument("--no-think-max-tokens", type=int, default=160)
    parser.add_argument("--thought-max-tokens", type=int, default=620)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/vastness_local"))
    args = parser.parse_args(argv)

    if args.audit_only:
        res = audit_vastness_suite()
        print(json.dumps(res, indent=2))
        return 0 if res["valid"] else 1

    if args.render_dir is not None:
        from afriat.plot_vastness import render_vastness_bundle

        jpath = args.render_dir / "vastness_results.json"
        data = json.loads(jpath.read_text(encoding="utf-8"))
        render_vastness_bundle(data, args.render_dir)
        print(f"[Vastness] Rendered reports & figures in {args.render_dir}")
        return 0

    if args.basis_npz is None or args.phase0_json is None:
        raise SystemExit("--basis-npz and --phase0-json are required for GPU execution")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "vastness_results.json"

    locked = LockedBasis.from_phase0_files(args.basis_npz, args.phase0_json)
    harness = TransformerHarness.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        model_id_label=args.model_label or resolve_model_name(args.model),
    )

    results = run_vastness_experiment(
        harness=harness,
        locked=locked,
        no_think_batch_size=args.no_think_batch_size,
        thought_batch_size=args.thought_batch_size,
        no_think_max_tokens=args.no_think_max_tokens,
        thought_max_tokens=args.thought_max_tokens,
        checkpoint_path=out_json,
    )
    print(f"[Vastness] Saved {out_json} ({results['runtime_seconds']}s)", flush=True)

    if not args.no_plots:
        from afriat.plot_vastness import render_vastness_bundle

        render_vastness_bundle(results, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
