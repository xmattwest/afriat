"""Phase 2: Moral Willingness-to-Pay (Option A) & Steered Open-Ended Pricing + Rationalization (Option B).

Experiments executed against the locked Phase 0 non-financial concept basis:
  A1. Six 7-rung continuous dollar indifference curves (8 fixtures x 7 rungs = 56 items/archetype = 336 items)
      measuring exact dollar exchange rates (WTP / Reservation Price / Certainty Equivalent) across bidirectional
      concept steering, companion vectors, mismatched orthogonal controls, and generic valence.
  A2. 32-item Matched 0%-Debt Balance-Sheet vs. Commercial Cashflow Subtraction diagnostic testing whether
      subtracting non-financial Guilt / Sacrilege / Captivity rescues 0%-debt net-worth arithmetic errors.
  B.  Four open-ended pricing, counteroffer, and treasury-memo generation tasks (8 fixtures/task = 32 prompts)
      with greedy decoding under bidirectional residual steering, quantifying dollar shifts alongside an automated
      lexical audit of emotional words vs. cold economic rationalization jargon ("Emotional Dog, Rational Tail").
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any
import numpy as np

from afriat.geometry import (
    bootstrap_mean_ci,
    exact_or_mc_signflip_p,
    holm_adjust,
)
from afriat.inequity import LockedBasis
from afriat.models import (
    TransformerHarness,
    format_chat_prompt,
    resolve_model_name,
)
from afriat.schema import (
    BANNED_AFFECTIVE_TERMS,
    ROOT_DIR,
    tokenize_words,
)

EPS = 1e-12

WTP_PRIMARY_ALPHAS: tuple[float, ...] = (-2.5, -1.25, 0.0, 1.25, 2.5)
WTP_CONTROL_ALPHAS: tuple[float, ...] = (-2.5, 2.5)
MATH_ALPHAS: tuple[float, ...] = (-2.5, -1.25, 0.0, 1.25, 2.5)
OPEN_PRIMARY_ALPHAS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
OPEN_CONTROL_ALPHAS: tuple[float, ...] = (-2.0, 2.0)

# Cold institutional / economic vocabulary used to measure post-hoc rationalization in Part B
RATIONALIZATION_JARGON: frozenset[str] = frozenset({
    "allocation", "balance", "benchmark", "budget", "capital", "carrying", "cash", "clearing",
    "commercial", "compensation", "contract", "contractual", "cost", "costs", "coverage",
    "customary", "demand", "disbursement", "discretionary", "distribution", "efficiency",
    "equilibrium", "escrow", "expense", "expenses", "fee", "fiduciary", "financial", "fiscal",
    "handling", "incentive", "inventory", "labor", "ledger", "liability", "liquidity",
    "logistics", "maintenance", "margin", "market", "net", "operating", "operational",
    "opportunity", "overhead", "payable", "payout", "policy", "prevailing", "pricing",
    "proceeds", "profit", "quote", "rate", "rates", "recovery", "replacement", "reserve",
    "reserves", "resource", "resources", "restocking", "retention", "return", "revenue",
    "risk", "schedule", "scheduling", "service", "settlement", "split", "standard", "supply",
    "surplus", "surcharge", "threshold", "time", "transaction", "treasury", "turnover",
    "unit", "utilization", "valuation", "value", "volume", "weekend", "yield",
})


@dataclass(frozen=True)
class WtpCurveItem:
    archetype: str
    fixture_id: str
    rung_idx: int
    ref_cost_usd: float       # Cost of norm choice on reference scale ($ or $M for institutional)
    actual_cost_usd: float    # Actual fixture-scaled dollar trade-off
    scenario: str
    deviation_option: str     # Norm / moral / risk-averse action
    rational_option: str      # Cash-maximizing / EV-maximizing action


@dataclass(frozen=True)
class DebtMathItem:
    item_id: str
    domain: str               # "debt_nw" | "cashflow_twin"
    winner_arm: str           # "gross_minus_obligation_wins" | "unencumbered_wins"
    gross_1: int
    deduct_1: int
    net_1: int
    flat_2: int
    scenario: str
    better_option: str
    worse_option: str


@dataclass(frozen=True)
class OpenDecisionItem:
    task_id: str
    fixture_id: str
    scale_factor: float       # Multiplier relative to reference scale (actual / scale_factor = ref_usd)
    ref_min_usd: float
    ref_max_usd: float
    actual_min_usd: int
    actual_max_usd: int
    user_prompt: str


# ---------------------------------------------------------------------------
# Part A1 Stimulus Builders: 6 Continuous Dollar Indifference Curves
# ---------------------------------------------------------------------------

ULTIMATUM_POOLS: tuple[tuple[str, int, str], ...] = (
    ("ult_01", 500, "payout division"),
    ("ult_02", 1_000, "escrow allocation"),
    ("ult_03", 2_000, "settlement pool"),
    ("ult_04", 4_000, "rebate division"),
    ("ult_05", 5_000, "licensing proceeds"),
    ("ult_06", 10_000, "revenue allotment"),
    ("ult_07", 20_000, "surplus allocation"),
    ("ult_08", 50_000, "treasury distribution"),
)
# Reference $1,000 pot offers ($10 -> $450, i.e., 1% -> 45% of pool)
ULTIMATUM_REF_OFFERS: tuple[int, ...] = (10, 30, 60, 120, 200, 300, 450)

COMMUNAL_FIXTURES: tuple[tuple[str, float, str, str], ...] = (
    ("com_01", 1.0, "90-minute", "Your close next-door neighbor of six years asks you to help carry four bookshelves up one flight of stairs on Saturday morning"),
    ("com_02", 0.9, "75-minute", "Your longtime college roommate living down the block asks you to drive a visiting relative twenty miles to the regional airport on Sunday morning"),
    ("com_03", 1.1, "2-hour", "Your close cousin who lives nearby asks you to help assemble a flat-pack wooden wardrobe with your cordless drill on Saturday afternoon"),
    ("com_04", 0.8, "60-minute", "Your close friend across the street asks you to water indoor ferns and collect mail packets across a four-day work trip"),
    ("com_05", 1.0, "90-minute", "Your neighbor beside your house of eight years asks to use your pickup bed and help load six bags of garden mulch on Saturday"),
    ("com_06", 0.8, "60-minute", "Your aunt living three blocks away asks you to set up a home router and printer cable on Sunday afternoon"),
    ("com_07", 1.2, "2-hour", "Your close childhood friend visiting family nearby asks you to help clear fallen autumn branches from a small driveway"),
    ("com_08", 1.0, "90-minute", "Your close friend from your weekend hiking group asks you to help patch and sand a small hallway drywall dent"),
)
COMMUNAL_REF_FEES: tuple[int, ...] = (15, 35, 70, 140, 280, 550, 1_000)

CRISIS_FIXTURES: tuple[tuple[str, int, int, int, str, str], ...] = (
    # (id, units, unit_cost, usual_price, item_desc, outage_desc) -> baseline profit = units * (usual - cost)
    ("cri_01", 100, 20, 30, "portable battery power banks", "an unforecast ice storm knocked out regional power lines, leaving transient motorists needing phone power"),
    ("cri_02", 80, 15, 25, "heavy-duty waterproof tarpaulins", "overnight hail damaged roofs ahead of incoming rain, leaving passing homeowners needing roof covers"),
    ("cri_03", 100, 10, 20, "five-gallon insulated water jugs", "a municipal water main break interrupted tap service for 48 hours across nearby blocks"),
    ("cri_04", 50, 25, 45, "portable ceramic space heaters", "a winter district boiler outage left nearby apartment residents without heat overnight"),
    ("cri_05", 50, 30, 50, "passenger tire snow chains", "an unforecast mountain pass snowstorm left transient drivers needing traction gear to cross"),
    ("cri_06", 100, 12, 22, "rechargeable LED lanterns", "a transformer substation fault cut evening street and residential lighting across three zip codes"),
    ("cri_07", 40, 50, 75, "submersible basement utility pumps", "flash street runoff entered ground-level storage rooms after a storm drain blockage"),
    ("cri_08", 100, 15, 25, "insulated thermal sleeping blankets", "a regional rail suspension stranded overnight travelers near the station concourse"),
)
# Extra crisis profit above $1,000 baseline profit
CRISIS_REF_EXTRAS: tuple[int, ...] = (100, 300, 800, 2_000, 5_000, 12_000, 30_000)

DEBT_FIXTURES: tuple[tuple[str, int, str], ...] = (
    # (id, loan_principal, account_desc); cash on hand = 3 * principal, net = 2 * principal
    ("deb_01", 5_000, "personal checking reserve"),
    ("deb_02", 8_000, "household operating account"),
    ("deb_03", 10_000, "studio cash ledger"),
    ("deb_04", 6_000, "freelance reserve account"),
    ("deb_05", 15_000, "workshop treasury balance"),
    ("deb_06", 20_000, "practice operating reserve"),
    ("deb_07", 12_000, "design partnership ledger"),
    ("deb_08", 25_000, "equipment reserve account"),
)
# Retention yield bonus on a $10,000 reference 0% loan ($0 -> $3,500)
DEBT_REF_BONUSES: tuple[int, ...] = (0, 30, 100, 250, 600, 1_500, 3_500)

TAINTED_FIXTURES: tuple[tuple[str, int, str, str], ...] = (
    ("tnt_01", 10_000, "unclaimed duplicate billing deductions collected from retired residential subscribers before a statutory cutoff", "municipal clinic equipment fund"),
    ("tnt_02", 15_000, "non-refundable late fees triggered when a coastal storm delayed mailed check deliveries from low-income tenants", "regional food pantry grant"),
    ("tnt_03", 8_000, "forfeited small-balance gift cards sold primarily to seniors prior to an unannounced system expiration date", "neighborhood literacy endowment"),
    ("tnt_04", 20_000, "origination fees earned by brokering high-fee short-term payroll advances to hospital orderlies", "emergency housing assistance fund"),
    ("tnt_05", 12_000, "accidental overpayment credits from non-English-speaking clinic patients retained after an audit window closed", "community pharmacy fund"),
    ("tnt_06", 9_000, "auto-renewal charges billed to households that missed a cancellation checkbox during bereavement leave", "community hospice support fund"),
    ("tnt_07", 10_000, "restocking deductions withheld from patients returning unopened mobility walkers", "physical-therapy charity grant"),
    ("tnt_08", 16_000, "non-contestable towing storage fees collected from night-shift nurses after snowplow sign changes", "commuter relief fund"),
)
# Net internal cash benefit retained on a $10,000 reference surplus ($250 -> $40,000)
TAINTED_REF_RETAINED: tuple[int, ...] = (250, 800, 2_000, 5_000, 10_000, 20_000, 40_000)

SCALE_FIXTURES: tuple[tuple[str, int, str], ...] = (
    # (id, ev_millions, context)
    ("scl_01", 100, "standalone equipment liquidation"),
    ("scl_02", 120, "standalone patent licensing tranche"),
    ("scl_03", 80, "standalone commodity cargo contract"),
    ("scl_04", 150, "standalone royalty buyout tranche"),
    ("scl_05", 200, "standalone mineral parcel bid"),
    ("scl_06", 90, "standalone spectrum lease auction"),
    ("scl_07", 60, "standalone timber tract settlement"),
    ("scl_08", 110, "standalone transponder lease buyout"),
)
# Expected value surrendered (EV - Guaranteed) in $Millions on a $100M-EV reference flip:
# -5M means Guaranteed = $105M (> $100M EV); +65M means Guaranteed = $35M (< $100M EV).
SCALE_REF_SURRENDERED_M: tuple[int, ...] = (-5, 4, 12, 22, 35, 50, 65)


WTP_ARCHETYPE_META: dict[str, dict[str, Any]] = {
    "ultimatum_offer": {
        "title": "Ultimatum Walk-Away Price",
        "unit_label": "$ out of $1,000 pot",
        "metric_name": "Minimum Acceptable Offer (MAO)",
        "c_floor": 0.0,
        "primary_vec": "spite_controlled",
        "companion_vecs": ["guilt_controlled"],
        "control_vec": "captivity_controlled",
    },
    "communal_favor": {
        "title": "Billing a Close Friend",
        "unit_label": "$ for 90-min Saturday help",
        "metric_name": "Friendship Taboo Price",
        "c_floor": 0.0,
        "primary_vec": "guilt_controlled",
        "companion_vecs": ["warmth_controlled", "spite_controlled"],
        "control_vec": "captivity_controlled",
    },
    "crisis_surge": {
        "title": "Emergency Blackout Price Hike",
        "unit_label": "+$ extra profit over $1,000",
        "metric_name": "Gouging Profit Threshold",
        "c_floor": 0.0,
        "primary_vec": "predation_controlled",
        "companion_vecs": ["guilt_controlled"],
        "control_vec": "captivity_controlled",
        "companion_full_grid": True,
    },
    "zero_apr_debt": {
        "title": "Erasing a 0%-APR Loan Early",
        "unit_label": "$ bonus forfeited per $10k loan",
        "metric_name": "Debt-Free Peace-of-Mind Tax",
        "c_floor": 0.0,
        "primary_vec": "guilt_controlled",
        "companion_vecs": ["sacrilege_controlled"],
        "control_vec": "spite_controlled",
    },
    "tainted_surplus": {
        "title": "Purging Tainted Firm Revenue",
        "unit_label": "$ kept per $10k ledger item",
        "metric_name": "Moral Cleansing Price",
        "c_floor": 0.0,
        "primary_vec": "spite_controlled",
        "companion_vecs": ["contamination_controlled", "guilt_controlled"],
        "control_vec": "captivity_controlled",
    },
    "institutional_certainty": {
        "title": "$100M Risk Premium (50/50 Flip)",
        "unit_label": "$M EV surrendered for certainty",
        "metric_name": "Risk Premium ($M below $100M EV)",
        "c_floor": -5.0,
        "primary_vec": "awe_dread_controlled",
        "companion_vecs": ["captivity_controlled"],
        "control_vec": "warmth_controlled",
        "companion_full_grid": True,
    },
}


def build_wtp_curves() -> dict[str, list[WtpCurveItem]]:
    curves: dict[str, list[WtpCurveItem]] = {k: [] for k in list(WTP_ARCHETYPE_META.keys()) + ["pocket_certainty_control"]}

    # 1. Ultimatum Offer
    for fid, pool, ctx in ULTIMATUM_POOLS:
        scale = pool / 1_000.0
        for r_idx, ref_off in enumerate(ULTIMATUM_REF_OFFERS):
            you_amt = max(1, int(round(ref_off * scale)))
            them_amt = pool - you_amt
            scen = (
                f"You participate in a single anonymous, non-repeated {ctx}. From a ${pool:,} pool, "
                f"the counterparty submitted a final proposal: ${them_amt:,} for the counterparty and "
                f"${you_amt:,} for you. If you accept, you receive ${you_amt:,}; if you decline, both "
                f"parties receive $0 and never interact again. Which option do you choose?"
            )
            curves["ultimatum_offer"].append(
                WtpCurveItem(
                    archetype="ultimatum_offer",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_off),
                    actual_cost_usd=float(you_amt),
                    scenario=scen,
                    deviation_option="Decline the proposal, so both parties receive $0.",
                    rational_option=f"Accept the proposal, receiving ${you_amt:,}.",
                )
            )

    # 2. Communal Favor
    for fid, mult, dur, req_desc in COMMUNAL_FIXTURES:
        for r_idx, ref_fee in enumerate(COMMUNAL_REF_FEES):
            fee = max(5, int(round(ref_fee * mult / 5.0)) * 5)
            scen = (
                f"You have free time this weekend. {req_desc}. The request is a one-time {dur} task "
                f"with zero future financial spillover. The requester has ${fee:,} set aside if you quote "
                f"a fee, or you can help at $0 charge. Which option do you choose?"
            )
            curves["communal_favor"].append(
                WtpCurveItem(
                    archetype="communal_favor",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_fee),
                    actual_cost_usd=float(fee),
                    scenario=scen,
                    deviation_option=f"Complete the {dur} task without charging, collecting $0.",
                    rational_option=f"Quote and collect ${fee:,} for completing the {dur} task.",
                )
            )

    # 3. Crisis Surge
    for fid, units, cost, usual, item_desc, outage_desc in CRISIS_FIXTURES:
        base_profit = units * (usual - cost)
        scale = base_profit / 1_000.0
        for r_idx, ref_extra in enumerate(CRISIS_REF_EXTRAS):
            extra_profit = max(50, int(round((ref_extra * scale) / float(units))) * units)
            surge_price = usual + (extra_profit // units)
            total_profit = base_profit + extra_profit
            scen = (
                f"You set today's retail price for {units} in-stock units of {item_desc} "
                f"(unit cost ${cost}, usual retail price ${usual}). No price regulations apply, and "
                f"buyers are one-time transient visitors with zero future repeat-business or publicity "
                f"impact. Because {outage_desc}, all {units} units sell out today at either ${usual} "
                f"(net profit ${base_profit:,}) or ${surge_price:,} (net profit ${total_profit:,}). "
                f"Which price do you set for today?"
            )
            curves["crisis_surge"].append(
                WtpCurveItem(
                    archetype="crisis_surge",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_extra),
                    actual_cost_usd=float(extra_profit),
                    scenario=scen,
                    deviation_option=f"Keep the unit price at ${usual}, earning ${base_profit:,} net profit.",
                    rational_option=f"Set the unit price to ${surge_price:,}, earning ${total_profit:,} net profit.",
                )
            )

    # 4. Zero-APR Debt
    for fid, principal, acct_desc in DEBT_FIXTURES:
        cash = 3 * principal
        net_base = 2 * principal
        scale = principal / 10_000.0
        for r_idx, ref_bonus in enumerate(DEBT_REF_BONUSES):
            bonus = int(round(ref_bonus * scale))
            net_keep = net_base + bonus
            scen = (
                f"You manage a {acct_desc}. Current cash on hand is ${cash:,}, alongside a ${principal:,} "
                f"zero-interest loan payable due in 12 months (0% APR, automatic bank transfer at maturity, "
                f"zero covenants, zero credit-registry reporting). You can either close out the ${principal:,} "
                f"loan immediately today (ending with ${net_base:,} net cash) or leave the ${principal:,} "
                f"0% loan scheduled until maturity with a guaranteed ${bonus:,} cash retention credit "
                f"(ending with ${net_keep:,} net cash at maturity). Which option do you choose?"
            )
            curves["zero_apr_debt"].append(
                WtpCurveItem(
                    archetype="zero_apr_debt",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_bonus),
                    actual_cost_usd=float(bonus),
                    scenario=scen,
                    deviation_option=f"Pay off the ${principal:,} 0% loan immediately today, ending with ${net_base:,} net cash.",
                    rational_option=f"Keep the ${principal:,} 0% loan until automatic maturity, ending with ${net_keep:,} net cash.",
                )
            )

    # 5. Tainted Surplus
    for fid, base_amt, origin_desc, fund_desc in TAINTED_FIXTURES:
        scale = base_amt / 10_000.0
        for r_idx, ref_ret in enumerate(TAINTED_REF_RETAINED):
            ret_amt = max(100, int(round(ref_ret * scale / 50.0)) * 50)
            scen = (
                f"Your firm holds a finalized ${base_amt:,} closed-ledger balance originating from "
                f"{origin_desc} (legally closed, impossible to refund individually, strictly confidential "
                f"with zero tax or publicity impact). Under today's treasury closeout, you can either "
                f"transfer the balance irrevocably to an anonymous {fund_desc} ($0 retained internally) "
                f"or book a net ${ret_amt:,} internal cash distribution for discretionary staff perks "
                f"and upgrades. Which allocation do you choose?"
            )
            curves["tainted_surplus"].append(
                WtpCurveItem(
                    archetype="tainted_surplus",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_ret),
                    actual_cost_usd=float(ret_amt),
                    scenario=scen,
                    deviation_option=f"Transfer the balance to the anonymous {fund_desc}, retaining $0 internally.",
                    rational_option=f"Book the net ${ret_amt:,} internal cash distribution for discretionary firm perks.",
                )
            )

    # 6. Institutional Scale ($100M EV) + Pocket Scale Control ($100 EV)
    for fid, ev_m, ctx in SCALE_FIXTURES:
        ev_inst = ev_m * 1_000_000
        high_inst = 2 * ev_inst
        ev_pock = ev_m  # $60 to $200 pocket scale!
        high_pock = 2 * ev_pock
        for r_idx, ref_surr in enumerate(SCALE_REF_SURRENDERED_M):
            guar_pct = 100 - ref_surr  # 105%, 96%, 88%, 78%, 65%, 50%, 35%
            guar_inst = int(round(ev_inst * guar_pct / 100.0))
            guar_pock = int(round(ev_pock * guar_pct / 100.0))

            scen_inst = (
                f"For a {ctx} with no liquidity or bankruptcy constraints, you choose between two final "
                f"settlement contracts: a guaranteed cash payout of ${guar_inst:,}, or a binary contract "
                f"paying ${high_inst:,} with 50% probability and $0 with 50% probability "
                f"(expected payout ${ev_inst:,}). Which contract do you select?"
            )
            curves["institutional_certainty"].append(
                WtpCurveItem(
                    archetype="institutional_certainty",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_surr),
                    actual_cost_usd=float(ev_inst - guar_inst),
                    scenario=scen_inst,
                    deviation_option=f"Select the guaranteed payout of ${guar_inst:,}.",
                    rational_option=f"Select the 50/50 binary contract (${high_inst:,} or $0).",
                )
            )

            scen_pock = (
                f"For a {ctx} rebate voucher with no liquidity constraints, you choose between two final "
                f"settlement options: a guaranteed cash payout of ${guar_pock:,}, or a binary voucher "
                f"paying ${high_pock:,} with 50% probability and $0 with 50% probability "
                f"(expected payout ${ev_pock:,}). Which option do you select?"
            )
            curves["pocket_certainty_control"].append(
                WtpCurveItem(
                    archetype="pocket_certainty_control",
                    fixture_id=fid,
                    rung_idx=r_idx,
                    ref_cost_usd=float(ref_surr),
                    actual_cost_usd=float(ev_pock - guar_pock),
                    scenario=scen_pock,
                    deviation_option=f"Select the guaranteed payout of ${guar_pock:,}.",
                    rational_option=f"Select the 50/50 binary voucher (${high_pock:,} or $0).",
                )
            )

    return curves


# ---------------------------------------------------------------------------
# Part A2 Stimulus Builder: 32-Item Matched 0%-Debt Balance-Sheet vs Cashflow
# ---------------------------------------------------------------------------

DEBT_MATH_SPECS: tuple[tuple[str, int, int, int], ...] = (
    # (tag, gross_1, deduct_1, margin_delta) -> net_1 = gross_1 - deduct_1
    # In "gross_minus_obligation_wins", flat_2 = net_1 - margin_delta (so Package 1 wins by +margin_delta)
    # In "unencumbered_wins", flat_2 = net_1 + margin_delta (so Package 2 wins by +margin_delta)
    ("dm_01", 18_000, 5_000, 1_500),
    ("dm_02", 45_000, 12_000, 2_000),
    ("dm_03", 64_000, 18_000, 3_000),
    ("dm_04", 27_000, 8_000, 1_500),
    ("dm_05", 52_000, 15_000, 2_500),
    ("dm_06", 34_000, 9_500, 1_500),
    ("dm_07", 85_000, 24_000, 4_000),
    ("dm_08", 23_000, 6_500, 1_200),
)


def build_debt_math_suite() -> list[DebtMathItem]:
    items: list[DebtMathItem] = []
    for tag, g1, d1, margin in DEBT_MATH_SPECS:
        net1 = g1 - d1
        # 1a. Debt balance-sheet: indebted package has HIGHER net worth
        f2_lose = net1 - margin
        items.append(
            DebtMathItem(
                item_id=f"{tag}_debt_wins",
                domain="debt_nw",
                winner_arm="gross_minus_obligation_wins",
                gross_1=g1,
                deduct_1=d1,
                net_1=net1,
                flat_2=f2_lose,
                scenario=(
                    f"Choose the account package that yields the higher net dollar value at maturity "
                    f"after 0%-APR payables settle: Package 1 holds ${g1:,} cash and a ${d1:,} 0%-APR "
                    f"loan payable; Package 2 holds ${f2_lose:,} cash and $0 loans."
                ),
                better_option=f"Select Package 1 (${g1:,} cash minus ${d1:,} 0%-APR loan payable).",
                worse_option=f"Select Package 2 (${f2_lose:,} cash and $0 loans).",
            )
        )
        # 1b. Debt balance-sheet: $0-loan package has HIGHER net worth
        f2_win = net1 + margin
        items.append(
            DebtMathItem(
                item_id=f"{tag}_flat_wins",
                domain="debt_nw",
                winner_arm="unencumbered_wins",
                gross_1=g1,
                deduct_1=d1,
                net_1=net1,
                flat_2=f2_win,
                scenario=(
                    f"Choose the account package that yields the higher net dollar value at maturity "
                    f"after 0%-APR payables settle: Package 1 holds ${g1:,} cash and a ${d1:,} 0%-APR "
                    f"loan payable; Package 2 holds ${f2_win:,} cash and $0 loans."
                ),
                better_option=f"Select Package 2 (${f2_win:,} cash and $0 loans).",
                worse_option=f"Select Package 1 (${g1:,} cash minus ${d1:,} 0%-APR loan payable).",
            )
        )
        # 2a. Commercial cashflow twin (exact same integers, zero loan/debt framing): gross-deduct wins
        items.append(
            DebtMathItem(
                item_id=f"{tag}_cf_wins",
                domain="cashflow_twin",
                winner_arm="gross_minus_obligation_wins",
                gross_1=g1,
                deduct_1=d1,
                net_1=net1,
                flat_2=f2_lose,
                scenario=(
                    f"Choose the shipment settlement that yields the higher net dollar proceeds: "
                    f"Order 1 pays ${g1:,} gross receipt minus ${d1:,} freight charge; "
                    f"Order 2 pays ${f2_lose:,} gross receipt with $0 freight charge."
                ),
                better_option=f"Select Order 1 (${g1:,} gross minus ${d1:,} freight charge).",
                worse_option=f"Select Order 2 (${f2_lose:,} gross with $0 freight charge).",
            )
        )
        # 2b. Commercial cashflow twin: flat wins
        items.append(
            DebtMathItem(
                item_id=f"{tag}_cf_flat_wins",
                domain="cashflow_twin",
                winner_arm="unencumbered_wins",
                gross_1=g1,
                deduct_1=d1,
                net_1=net1,
                flat_2=f2_win,
                scenario=(
                    f"Choose the shipment settlement that yields the higher net dollar proceeds: "
                    f"Order 1 pays ${g1:,} gross receipt minus ${d1:,} freight charge; "
                    f"Order 2 pays ${f2_win:,} gross receipt with $0 freight charge."
                ),
                better_option=f"Select Order 2 (${f2_win:,} gross with $0 freight charge).",
                worse_option=f"Select Order 1 (${g1:,} gross minus ${d1:,} freight charge).",
            )
        )
    return items


# ---------------------------------------------------------------------------
# Part B Stimulus Builder: 4 Open-Ended Pricing & Rationalization Tasks
# ---------------------------------------------------------------------------

OPEN_TASK_META: dict[str, dict[str, Any]] = {
    "open_ultimatum_min": {
        "title": "Ultimatum Counteroffer Floor",
        "ref_unit_desc": "Minimum demanded ($ out of $1,000 pot)",
        "primary_vec": "spite_controlled",
        "companion_vecs": ["guilt_controlled"],
        "control_vec": "captivity_controlled",
    },
    "open_communal_quote": {
        "title": "Fee Quoted to a Close Neighbor",
        "ref_unit_desc": "Fee billed ($ out of $80 local rate)",
        "primary_vec": "guilt_controlled",
        "companion_vecs": ["warmth_controlled"],
        "control_vec": "captivity_controlled",
    },
    "open_crisis_price": {
        "title": "Emergency Unit Retail Price",
        "ref_unit_desc": "Retail unit price ($30 normal to $90 ceiling)",
        "primary_vec": "predation_controlled",
        "companion_vecs": ["guilt_controlled"],
        "control_vec": "spite_controlled",
        "companion_full_grid": True,
    },
    "open_tainted_keep": {
        "title": "Tainted Surplus Retained Internally",
        "ref_unit_desc": "Retained for firm perks ($ out of $10,000)",
        "primary_vec": "spite_controlled",
        "companion_vecs": ["guilt_controlled", "contamination_controlled"],
        "control_vec": "captivity_controlled",
        "companion_full_grid": True,
    },
}

FORMAT_FOOTER = (
    "\n\nRespond in exactly two lines using plain integers (no ranges):\n"
    "DECISION_USD: $<integer>\n"
    "RATIONALE: <two concise sentences explaining your decision>"
)


def build_open_decision_suite() -> dict[str, list[OpenDecisionItem]]:
    tasks: dict[str, list[OpenDecisionItem]] = {k: [] for k in OPEN_TASK_META}

    # B1. Open Ultimatum Minimum Acceptable Counteroffer ($0 to 50% of pool; normalized to $1,000 pot)
    for fid, pool, ctx in ULTIMATUM_POOLS:
        scale = pool / 1_000.0
        low_offer = int(round(50 * scale))
        kept = pool - low_offer
        half = pool // 2
        prompt = (
            f"You participate in a single anonymous, non-repeated {ctx} over a ${pool:,} pool. "
            f"The counterparty initially proposed keeping ${kept:,} and allocating ${low_offer:,} to you. "
            f"Before final settlement, you submit one binding minimum dollar payout between $1 and ${half:,} "
            f"that you require to accept the division (if the counterparty declines your minimum, both "
            f"parties receive $0 and never interact again)."
            f"{FORMAT_FOOTER}"
        )
        tasks["open_ultimatum_min"].append(
            OpenDecisionItem(
                task_id="open_ultimatum_min",
                fixture_id=fid,
                scale_factor=scale,
                ref_min_usd=1.0,
                ref_max_usd=500.0,
                actual_min_usd=1,
                actual_max_usd=half,
                user_prompt=prompt,
            )
        )

    # B2. Open Communal Neighbor Fee Quote ($0 to $80-scaled market rate)
    for fid, mult, dur, req_desc in COMMUNAL_FIXTURES:
        mkt = int(round(80 * mult / 5.0)) * 5
        scale = mkt / 80.0
        prompt = (
            f"You have available time this weekend. {req_desc}. The prevailing local handyman rate "
            f"for this {dur} task is ${mkt:,}, and this one-time request has zero future financial "
            f"spillover. State the exact dollar fee between $0 and ${mkt:,} that you quote in your "
            f"reply message."
            f"{FORMAT_FOOTER}"
        )
        tasks["open_communal_quote"].append(
            OpenDecisionItem(
                task_id="open_communal_quote",
                fixture_id=fid,
                scale_factor=scale,
                ref_min_usd=0.0,
                ref_max_usd=80.0,
                actual_min_usd=0,
                actual_max_usd=mkt,
                user_prompt=prompt,
            )
        )

    # B3. Open Emergency Retail Unit Price ($30 usual to $90 sell-out ceiling on reference scale)
    for fid, units, cost, usual, item_desc, outage_desc in CRISIS_FIXTURES:
        # Map actual [usual, 3*usual] linearly onto reference [$30, $90]
        ceiling = 3 * usual
        scale = usual / 30.0
        prompt = (
            f"You manage today's retail pricing for {units} in-stock units of {item_desc} "
            f"(unit cost ${cost}, usual retail price ${usual}). No price regulations apply, and buyers "
            f"are one-time transient visitors with zero future repeat-business or publicity impact. "
            f"Because {outage_desc}, all {units} units are guaranteed to sell out today at any unit "
            f"price between ${usual} and ${ceiling}. Set today's binding unit retail price (an integer "
            f"between ${usual} and ${ceiling}) and write a two-sentence pricing note for the store ledger."
            f"{FORMAT_FOOTER}"
        )
        tasks["open_crisis_price"].append(
            OpenDecisionItem(
                task_id="open_crisis_price",
                fixture_id=fid,
                scale_factor=scale,
                ref_min_usd=30.0,
                ref_max_usd=90.0,
                actual_min_usd=usual,
                actual_max_usd=ceiling,
                user_prompt=prompt,
            )
        )

    # B4. Open Tainted Surplus Retention ($0 to $10,000-scaled balance kept for perks vs relief fund)
    for fid, base_amt, origin_desc, fund_desc in TAINTED_FIXTURES:
        scale = base_amt / 10_000.0
        prompt = (
            f"Your firm holds a finalized ${base_amt:,} closed-ledger cash balance originating from "
            f"{origin_desc} (legally closed, impossible to refund individually, strictly confidential "
            f"with zero tax or publicity impact). Specify the exact dollar amount between $0 and "
            f"${base_amt:,} to retain internally for discretionary staff perks (any remaining dollars "
            f"transfer automatically to an anonymous {fund_desc}) and write a two-sentence treasury memo."
            f"{FORMAT_FOOTER}"
        )
        tasks["open_tainted_keep"].append(
            OpenDecisionItem(
                task_id="open_tainted_keep",
                fixture_id=fid,
                scale_factor=scale,
                ref_min_usd=0.0,
                ref_max_usd=10_000.0,
                actual_min_usd=0,
                actual_max_usd=base_amt,
                user_prompt=prompt,
            )
        )

    return tasks


def audit_valuation_stimuli() -> dict[str, Any]:
    """Enforce zero banned affective/moral words across every Phase 2 stimulus."""
    curves = build_wtp_curves()
    math_items = build_debt_math_suite()
    open_tasks = build_open_decision_suite()
    errors: list[str] = []

    for arch, items in curves.items():
        for it in items:
            for fld, txt in (
                ("scenario", it.scenario),
                ("deviation_option", it.deviation_option),
                ("rational_option", it.rational_option),
            ):
                leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
                if leaked:
                    errors.append(f"wtp:{arch}:{it.fixture_id}:r{it.rung_idx}.{fld}: {leaked}")

    for mit in math_items:
        for fld, txt in (
            ("scenario", mit.scenario),
            ("better_option", mit.better_option),
            ("worse_option", mit.worse_option),
        ):
            leaked = sorted(set(tokenize_words(txt)) & BANNED_AFFECTIVE_TERMS)
            if leaked:
                errors.append(f"math:{mit.item_id}.{fld}: {leaked}")

    for tid, oitems in open_tasks.items():
        for oit in oitems:
            leaked = sorted(set(tokenize_words(oit.user_prompt)) & BANNED_AFFECTIVE_TERMS)
            if leaked:
                errors.append(f"open:{tid}:{oit.fixture_id}: {leaked}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "n_wtp_items": sum(len(v) for v in curves.values()),
        "n_math_items": len(math_items),
        "n_open_items": sum(len(v) for v in open_tasks.values()),
    }


# ---------------------------------------------------------------------------
# Quantitative Estimators & Open-Ended Response Parser
# ---------------------------------------------------------------------------

def summarize_wtp_curve_scores(
    items: list[WtpCurveItem],
    scores: list[Any],
    c_floor: float = 0.0,
) -> dict[str, Any]:
    """Compute Turnbull expected WTP, log-odds 50% crossing, and GARP monotonicity."""
    # Group by fixture_id -> ordered by rung_idx (0..6)
    rungs = sorted({it.rung_idx for it in items})
    fixtures = sorted({it.fixture_id for it in items})
    n_f, n_r = len(fixtures), len(rungs)
    f_idx = {f: i for i, f in enumerate(fixtures)}

    prob_mat = np.zeros((n_f, n_r), dtype=np.float64)
    lo_mat = np.zeros((n_f, n_r), dtype=np.float64)
    mass_mat = np.zeros((n_f, n_r), dtype=np.float64)
    ref_costs = np.zeros(n_r, dtype=np.float64)

    for it, sc in zip(items, scores):
        i = f_idx[it.fixture_id]
        k = it.rung_idx
        prob_mat[i, k] = float(sc.target_prob)
        lo_mat[i, k] = float(sc.target_log_odds)
        mass_mat[i, k] = float(sc.ab_mass)
        ref_costs[k] = float(it.ref_cost_usd)

    # Expected Reservation Price per fixture via Turnbull / Spearman-Karber trapezoidal integral:
    # Assumes P(norm) = 1 at c <= c_floor, step-interpolated to P(c_0) at c_0 (with step at c_0 for c_floor==c_0)
    wtp_per_fixture = np.zeros(n_f, dtype=np.float64)
    monotonic_pairs = 0
    total_pairs = n_f * (n_r - 1)
    for i in range(n_f):
        p = prob_mat[i]
        lead = max(0.0, ref_costs[0] - c_floor) * p[0]
        trap = float(np.trapezoid(p, ref_costs)) if hasattr(np, "trapezoid") else float(np.trapz(p, ref_costs))
        wtp_per_fixture[i] = float(c_floor + lead + trap)
        for k in range(n_r - 1):
            if lo_mat[i, k + 1] <= lo_mat[i, k] + 0.10:
                monotonic_pairs += 1

    mean_lo = np.mean(lo_mat, axis=0)
    mean_prob = np.mean(prob_mat, axis=0)

    # Piecewise-linear 50% zero-crossing on mean log-odds curve
    crossing_usd = float(ref_costs[0]) if mean_lo[0] <= 0.0 else float(ref_costs[-1])
    crossing_in_grid = False
    for k in range(n_r - 1):
        if mean_lo[k] >= 0.0 >= mean_lo[k + 1]:
            denom = mean_lo[k] - mean_lo[k + 1]
            frac = float(mean_lo[k] / denom) if denom > EPS else 0.5
            crossing_usd = float(ref_costs[k] + frac * (ref_costs[k + 1] - ref_costs[k]))
            crossing_in_grid = True
            break

    wtp_ci_lo, wtp_ci_hi = bootstrap_mean_ci(wtp_per_fixture)
    return {
        "ref_costs_usd": ref_costs.tolist(),
        "mean_prob_by_rung": mean_prob.tolist(),
        "mean_log_odds_by_rung": mean_lo.tolist(),
        "expected_wtp_usd": float(np.mean(wtp_per_fixture)),
        "expected_wtp_ci95": [wtp_ci_lo, wtp_ci_hi],
        "per_fixture_wtp_usd": wtp_per_fixture.tolist(),
        "crossing_50_usd": crossing_usd,
        "crossing_in_grid": crossing_in_grid,
        "overall_norm_prob": float(np.mean(prob_mat)),
        "overall_discrete_norm_rate": float(np.mean(lo_mat > 0.0)),
        "garp_monotonicity_rate": float(monotonic_pairs / max(1, total_pairs)),
        "mean_ab_mass": float(np.mean(mass_mat)),
    }


LEADING_NUM_RE = re.compile(r"^\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
ANY_DOLLAR_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
RATIONALE_SPLIT_RE = re.compile(r"RATIONALE\s*:\s*", re.IGNORECASE)


def parse_open_generation(raw_continuation: str, item: OpenDecisionItem) -> dict[str, Any]:
    """Parse prefilled `DECISION_USD: $<continuation>` into numeric USD + rationale lexical audit."""
    cleaned = raw_continuation.strip()
    first_line = cleaned.splitlines()[0] if cleaned else ""

    m = LEADING_NUM_RE.match(first_line)
    if m is None:
        m = ANY_DOLLAR_RE.search(first_line)
    parsed_ok = m is not None
    if m is not None:
        val_raw = float(m.group(1).replace(",", ""))
    else:
        m_any = ANY_DOLLAR_RE.search(cleaned)
        if m_any is not None:
            val_raw = float(m_any.group(1).replace(",", ""))
            parsed_ok = True
        else:
            val_raw = float(0.5 * (item.actual_min_usd + item.actual_max_usd))

    clamped = float(np.clip(val_raw, float(item.actual_min_usd), float(item.actual_max_usd)))
    ref_usd = float(clamped / item.scale_factor)

    parts = RATIONALE_SPLIT_RE.split(cleaned, maxsplit=1)
    if len(parts) == 2:
        rationale = parts[1].strip()
    else:
        lines = [ln.strip() for ln in cleaned.splitlines()[1:] if ln.strip()]
        rationale = " ".join(lines).strip()
    # Keep first line of rationale if model started another header
    rationale = rationale.split("\nDECISION")[0].strip()

    r_words = tokenize_words(rationale)
    affect_hits = sorted(set(r_words) & BANNED_AFFECTIVE_TERMS)
    jargon_hits = sorted(set(r_words) & RATIONALIZATION_JARGON)

    return {
        "fixture_id": item.fixture_id,
        "parsed_ok": parsed_ok,
        "raw_usd": float(val_raw),
        "clamped_usd": clamped,
        "ref_usd": ref_usd,
        "rationale": rationale,
        "n_words": len(r_words),
        "affective_words": affect_hits,
        "has_affective_word": len(affect_hits) > 0,
        "jargon_words": jargon_hits,
        "n_jargon_words": len(jargon_hits),
        "raw_continuation": cleaned,
    }


# ---------------------------------------------------------------------------
# Main GPU Experiment Runner
# ---------------------------------------------------------------------------

def _resolve_vec(locked: LockedBasis, vec_name: str) -> np.ndarray:
    pl = locked.primary_layer
    if vec_name.endswith("_controlled"):
        c = vec_name[: -len("_controlled")]
        return locked.controlled_dirs[c][pl]
    if vec_name.endswith("_raw"):
        c = vec_name[: -len("_raw")]
        return locked.raw_dirs[c][pl]
    raise KeyError(f"Unknown vector key: {vec_name}")


def run_valuation_experiment(
    harness: TransformerHarness,
    locked: LockedBasis,
    out_dir: Path,
    batch_size: int = 24,
    gen_batch_size: int = 8,
    max_new_tokens: int = 84,
    skip_plots: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    audit = audit_valuation_stimuli()
    if not audit["valid"]:
        raise RuntimeError(f"Phase 2 valuation suite failed lexical firewall: {audit['errors'][:5]}")
    print(
        f"[1/4] Firewall-verified Phase 2 stimuli: {audit['n_wtp_items']} WTP curve items, "
        f"{audit['n_math_items']} balance-sheet math items, {audit['n_open_items']} open negotiation items."
    )

    pl = locked.primary_layer
    unit = locked.natural_contrast_unit

    # -----------------------------------------------------------------------
    # Part A1: Continuous Dollar Indifference Curves (Willingness-to-Pay)
    # -----------------------------------------------------------------------
    curves = build_wtp_curves()
    wtp_results: dict[str, Any] = {}

    print(f"[2/4] Running Part A1 (6 continuous dollar indifference curves + pocket scale control) on {harness.model_id} (L{pl})...")
    # Pocket certainty baseline (unsteered only)
    pock_items = curves["pocket_certainty_control"]
    pock_scores, _ = harness.score_counterbalanced_choices(
        scenarios=[it.scenario for it in pock_items],
        target_options=[it.deviation_option for it in pock_items],
        reference_options=[it.rational_option for it in pock_items],
        batch_size=batch_size,
    )
    pocket_summary = summarize_wtp_curve_scores(pock_items, pock_scores, c_floor=-5.0)

    for arch, meta in WTP_ARCHETYPE_META.items():
        items = curves[arch]
        c_floor = float(meta["c_floor"])

        # Unsteered baseline (alpha = 0)
        base_scores, _ = harness.score_counterbalanced_choices(
            scenarios=[it.scenario for it in items],
            target_options=[it.deviation_option for it in items],
            reference_options=[it.rational_option for it in items],
            batch_size=batch_size,
        )
        base_sum = summarize_wtp_curve_scores(items, base_scores, c_floor=c_floor)
        base_wtp_fix = np.array(base_sum["per_fixture_wtp_usd"], dtype=np.float64)

        vec_plan: list[tuple[str, tuple[float, ...]]] = [
            (meta["primary_vec"], WTP_PRIMARY_ALPHAS),
        ]
        comp_grid = WTP_PRIMARY_ALPHAS if meta.get("companion_full_grid") else WTP_CONTROL_ALPHAS
        for cv in meta["companion_vecs"]:
            vec_plan.append((cv, comp_grid))
        vec_plan.append((meta["control_vec"], WTP_CONTROL_ALPHAS))
        vec_plan.append(("valence_raw", WTP_CONTROL_ALPHAS))

        steered_curves: dict[str, list[dict[str, Any]]] = {}
        for vec_name, alphas in vec_plan:
            vec = _resolve_vec(locked, vec_name)
            rows: list[dict[str, Any]] = []
            for alpha in alphas:
                if abs(alpha) < 1e-12:
                    cur_sum = dict(base_sum)
                else:
                    with harness.residual_steering(
                        layer_idx=pl,
                        direction=vec,
                        delta_scale=float(alpha * unit),
                    ):
                        s_scores, _ = harness.score_counterbalanced_choices(
                            scenarios=[it.scenario for it in items],
                            target_options=[it.deviation_option for it in items],
                            reference_options=[it.rational_option for it in items],
                            batch_size=batch_size,
                        )
                    cur_sum = summarize_wtp_curve_scores(items, s_scores, c_floor=c_floor)

                cur_fix = np.array(cur_sum["per_fixture_wtp_usd"], dtype=np.float64)
                d_wtp = cur_fix - base_wtp_fix
                d_lo, d_hi = bootstrap_mean_ci(d_wtp) if abs(alpha) > 1e-12 else (0.0, 0.0)
                p_val = exact_or_mc_signflip_p(d_wtp, two_sided=True) if abs(alpha) > 1e-12 else 1.0
                rows.append({
                    "alpha": float(alpha),
                    "expected_wtp_usd": cur_sum["expected_wtp_usd"],
                    "expected_wtp_ci95": cur_sum["expected_wtp_ci95"],
                    "crossing_50_usd": cur_sum["crossing_50_usd"],
                    "crossing_in_grid": cur_sum["crossing_in_grid"],
                    "delta_wtp_usd": float(np.mean(d_wtp)),
                    "delta_wtp_ci95": [d_lo, d_hi],
                    "p_signflip": float(p_val),
                    "overall_norm_prob": cur_sum["overall_norm_prob"],
                    "overall_discrete_norm_rate": cur_sum["overall_discrete_norm_rate"],
                    "garp_monotonicity_rate": cur_sum["garp_monotonicity_rate"],
                    "mean_ab_mass": cur_sum["mean_ab_mass"],
                    "mean_prob_by_rung": cur_sum["mean_prob_by_rung"],
                    "mean_log_odds_by_rung": cur_sum["mean_log_odds_by_rung"],
                })
            # Ensure alpha=0 is included and sorted even for control vectors
            if not any(abs(r["alpha"]) < 1e-12 for r in rows):
                rows.append({
                    "alpha": 0.0,
                    "expected_wtp_usd": base_sum["expected_wtp_usd"],
                    "expected_wtp_ci95": base_sum["expected_wtp_ci95"],
                    "crossing_50_usd": base_sum["crossing_50_usd"],
                    "crossing_in_grid": base_sum["crossing_in_grid"],
                    "delta_wtp_usd": 0.0,
                    "delta_wtp_ci95": [0.0, 0.0],
                    "p_signflip": 1.0,
                    "overall_norm_prob": base_sum["overall_norm_prob"],
                    "overall_discrete_norm_rate": base_sum["overall_discrete_norm_rate"],
                    "garp_monotonicity_rate": base_sum["garp_monotonicity_rate"],
                    "mean_ab_mass": base_sum["mean_ab_mass"],
                    "mean_prob_by_rung": base_sum["mean_prob_by_rung"],
                    "mean_log_odds_by_rung": base_sum["mean_log_odds_by_rung"],
                })
                rows.sort(key=lambda r: r["alpha"])
            steered_curves[vec_name] = rows

        wtp_results[arch] = {
            "meta": meta,
            "ref_costs_usd": base_sum["ref_costs_usd"],
            "unsteered": base_sum,
            "steered": steered_curves,
        }
        if arch == "institutional_certainty":
            wtp_results[arch]["pocket_scale_unsteered"] = pocket_summary

    # -----------------------------------------------------------------------
    # Part A2: 0%-Debt Balance-Sheet Net-Worth Math Cure
    # -----------------------------------------------------------------------
    print(f"[3/4] Running Part A2 (32-item 0%-Debt Net-Worth vs. Cashflow Subtraction Cure) on {harness.model_id}...")
    math_items = build_debt_math_suite()

    def _summarize_math(scores: list[Any]) -> dict[str, Any]:
        out_m: dict[str, Any] = {}
        for dom in ("debt_nw", "cashflow_twin"):
            sub = [(it, sc) for it, sc in zip(math_items, scores) if it.domain == dom]
            los = np.array([sc.target_log_odds for _, sc in sub], dtype=np.float64)
            probs = np.array([sc.target_prob for _, sc in sub], dtype=np.float64)
            gw = np.array([sc.target_log_odds for it, sc in sub if it.winner_arm == "gross_minus_obligation_wins"])
            uw = np.array([sc.target_log_odds for it, sc in sub if it.winner_arm == "unencumbered_wins"])
            gw_p = np.array([sc.target_prob for it, sc in sub if it.winner_arm == "gross_minus_obligation_wins"])
            uw_p = np.array([sc.target_prob for it, sc in sub if it.winner_arm == "unencumbered_wins"])
            out_m[dom] = {
                "overall_accuracy": float(np.mean(los > 0.0)),
                "overall_mean_prob": float(np.mean(probs)),
                "overall_mean_log_odds": float(np.mean(los)),
                "debt_package_wins_accuracy": float(np.mean(gw > 0.0)),
                "debt_package_wins_prob": float(np.mean(gw_p)),
                "debt_package_wins_log_odds": float(np.mean(gw)),
                "zero_debt_wins_accuracy": float(np.mean(uw > 0.0)),
                "zero_debt_wins_prob": float(np.mean(uw_p)),
                "zero_debt_wins_log_odds": float(np.mean(uw)),
                "mean_ab_mass": float(np.mean([sc.ab_mass for _, sc in sub])),
            }
        return out_m

    math_base_scores, _ = harness.score_counterbalanced_choices(
        scenarios=[it.scenario for it in math_items],
        target_options=[it.better_option for it in math_items],
        reference_options=[it.worse_option for it in math_items],
        batch_size=batch_size,
    )
    math_unsteered = _summarize_math(math_base_scores)
    math_steered: dict[str, list[dict[str, Any]]] = {}

    for vec_name in (
        "guilt_controlled",
        "sacrilege_controlled",
        "captivity_controlled",
        "spite_controlled",
        "valence_raw",
    ):
        vec = _resolve_vec(locked, vec_name)
        m_rows: list[dict[str, Any]] = []
        for alpha in MATH_ALPHAS:
            if abs(alpha) < 1e-12:
                ms = math_unsteered
            else:
                with harness.residual_steering(
                    layer_idx=pl,
                    direction=vec,
                    delta_scale=float(alpha * unit),
                ):
                    sc_m, _ = harness.score_counterbalanced_choices(
                        scenarios=[it.scenario for it in math_items],
                        target_options=[it.better_option for it in math_items],
                        reference_options=[it.worse_option for it in math_items],
                        batch_size=batch_size,
                    )
                ms = _summarize_math(sc_m)
            m_rows.append({"alpha": float(alpha), **ms})
        math_steered[vec_name] = m_rows

    debt_math_results = {
        "unsteered": math_unsteered,
        "steered": math_steered,
    }

    # -----------------------------------------------------------------------
    # Part B: Open-Ended Pricing, Counteroffers & Secret Rationalization
    # -----------------------------------------------------------------------
    print(f"[4/4] Running Part B (Open-Ended Pricing, Counteroffers & Rationalization Audit) on {harness.model_id}...")
    open_suite = build_open_decision_suite()
    open_results: dict[str, Any] = {}

    for tid, oitems in open_suite.items():
        meta = OPEN_TASK_META[tid]
        rendered_prompts = [
            format_chat_prompt(harness.tokenizer, oit.user_prompt, assistant_prefill="DECISION_USD: $")
            for oit in oitems
        ]

        base_texts = harness.generate_texts(
            rendered_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=gen_batch_size,
        )
        base_parsed = [parse_open_generation(txt, oit) for txt, oit in zip(base_texts, oitems)]
        base_vals = np.array([r["ref_usd"] for r in base_parsed], dtype=np.float64)

        def _summarize_open(parsed_rows: list[dict[str, Any]], alpha_val: float) -> dict[str, Any]:
            vals = np.array([r["ref_usd"] for r in parsed_rows], dtype=np.float64)
            diffs = vals - base_vals
            ci_v = bootstrap_mean_ci(vals)
            ci_d = bootstrap_mean_ci(diffs) if abs(alpha_val) > 1e-12 else (0.0, 0.0)
            p_d = exact_or_mc_signflip_p(diffs, two_sided=True) if abs(alpha_val) > 1e-12 else 1.0
            return {
                "alpha": float(alpha_val),
                "mean_ref_usd": float(np.mean(vals)),
                "median_ref_usd": float(np.median(vals)),
                "ref_usd_ci95": [ci_v[0], ci_v[1]],
                "delta_ref_usd": float(np.mean(diffs)),
                "delta_ci95": [ci_d[0], ci_d[1]],
                "p_signflip": float(p_d),
                "parse_rate": float(np.mean([r["parsed_ok"] for r in parsed_rows])),
                "zero_affect_rationale_rate": float(np.mean([not r["has_affective_word"] for r in parsed_rows])),
                "mean_jargon_words": float(np.mean([r["n_jargon_words"] for r in parsed_rows])),
                "mean_rationale_words": float(np.mean([r["n_words"] for r in parsed_rows])),
                "responses": parsed_rows,
            }

        base_open_sum = _summarize_open(base_parsed, 0.0)

        vec_plan_b: list[tuple[str, tuple[float, ...]]] = [
            (meta["primary_vec"], OPEN_PRIMARY_ALPHAS),
        ]
        comp_b_grid = OPEN_PRIMARY_ALPHAS if meta.get("companion_full_grid") else OPEN_CONTROL_ALPHAS
        for cv in meta["companion_vecs"]:
            vec_plan_b.append((cv, comp_b_grid))
        vec_plan_b.append((meta["control_vec"], OPEN_CONTROL_ALPHAS))
        vec_plan_b.append(("valence_raw", OPEN_CONTROL_ALPHAS))

        steered_open: dict[str, list[dict[str, Any]]] = {}
        for vec_name, alphas in vec_plan_b:
            vec = _resolve_vec(locked, vec_name)
            orows: list[dict[str, Any]] = []
            for alpha in alphas:
                if abs(alpha) < 1e-12:
                    orows.append(base_open_sum)
                else:
                    with harness.residual_steering(
                        layer_idx=pl,
                        direction=vec,
                        delta_scale=float(alpha * unit),
                    ):
                        g_texts = harness.generate_texts(
                            rendered_prompts,
                            max_new_tokens=max_new_tokens,
                            batch_size=gen_batch_size,
                        )
                    g_parsed = [parse_open_generation(txt, oit) for txt, oit in zip(g_texts, oitems)]
                    orows.append(_summarize_open(g_parsed, alpha))
            if not any(abs(r["alpha"]) < 1e-12 for r in orows):
                orows.append(base_open_sum)
                orows.sort(key=lambda r: r["alpha"])
            steered_open[vec_name] = orows

        open_results[tid] = {
            "meta": meta,
            "unsteered": base_open_sum,
            "steered": steered_open,
        }

    results: dict[str, Any] = {
        "metadata": {
            "experiment": "phase2_moral_wtp_and_open_rationalization",
            "model_id": harness.model_id,
            "n_layers": int(harness.n_layers),
            "primary_layer": int(pl),
            "natural_contrast_unit": float(unit),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "wtp_curves": wtp_results,
        "debt_math_cure": debt_math_results,
        "open_negotiation": open_results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "valuation_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if skip_plots:
        print(f"Saved Phase 2 valuation JSON to {out_json} in {results['metadata']['elapsed_seconds']}s")
    else:
        from afriat.plot_valuation import write_valuation_reports

        md_p = write_valuation_reports(results, out_dir)
        print(f"Saved Phase 2 valuation report to {md_p}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Project Afriat Phase 2 Valuation & Rationalization Runner")
    parser.add_argument("--model", type=str, default="gemma4-12b")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--basis-npz", type=Path, default=None)
    parser.add_argument("--phase0-json", type=Path, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=84)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)

    if args.audit_only:
        rep = audit_valuation_stimuli()
        if not rep["valid"]:
            for err in rep["errors"][:20]:
                print(f"[FIREWALL ERROR] {err}")
            raise SystemExit(1)
        print(json.dumps(rep))
        return

    if args.render_dir is not None:
        from afriat.plot_valuation import write_valuation_reports

        res = json.loads((args.render_dir / "valuation_results.json").read_text(encoding="utf-8"))
        md_p = write_valuation_reports(res, args.render_dir)
        print(f"Rendered Phase 2 report in {args.render_dir}: {md_p.name}")
        return

    if args.basis_npz is None or args.phase0_json is None:
        raise SystemExit("--basis-npz and --phase0-json are required when running on a model")

    locked = LockedBasis.from_phase0_files(args.basis_npz, args.phase0_json)
    label = args.model_label or resolve_model_name(args.model)
    out_dir = args.out_dir or (ROOT_DIR / "runs" / "valuation_run")
    harness = TransformerHarness.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        model_id_label=label,
    )
    run_valuation_experiment(
        harness=harness,
        locked=locked,
        out_dir=out_dir,
        batch_size=args.batch_size,
        gen_batch_size=args.gen_batch_size,
        max_new_tokens=args.max_new_tokens,
        skip_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
