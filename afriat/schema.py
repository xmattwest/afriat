"""Stimulus schemas, bidirectional lexical firewall, and manifest hashing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
STIMULI_DIR = ROOT_DIR / "stimuli"

CORE_CONCEPTS: tuple[str, ...] = (
    "guilt",
    "captivity",
    "contamination",
    "sacrilege",
    "spite",
    "predation",
    "awe_dread",
    "warmth",
)
CONTROL_CONCEPTS: tuple[str, ...] = ("valence", "arousal")
ALL_CONCEPTS: tuple[str, ...] = CORE_CONCEPTS + CONTROL_CONCEPTS

CORE_ARCHETYPES: tuple[str, ...] = (
    "zero_interest_debt",
    "communal_taboo_pricing",
    "crisis_markup",
    "insulting_split",
    "tainted_provenance",
    "institutional_scale",
)

# Strictly banned in non-financial concept stimuli: numbers, currency, commerce,
# and dual-use moral-financial metaphors ("owe", "pay", "worth", "redeem", "forgive", "credit").
BANNED_FINANCIAL_TERMS: frozenset[str] = frozenset({
    "account", "accounting", "afford", "apr", "asset", "assets", "audit", "balance",
    "bank", "bankrupt", "bankruptcy", "bargain", "bet", "bets", "betting", "bill",
    "billing", "bills", "bonus", "borrow", "borrowed", "borrowing", "bought", "bribe",
    "budget", "business", "buy", "buyer", "buying", "capital", "cash", "casino", "cent",
    "cents", "charge", "charged", "charging", "cheap", "check", "client", "clients",
    "coin", "coins", "commerce", "commercial", "commission", "compensate", "compensated",
    "compensation", "contract", "corporate", "cost", "costs", "costly", "credit",
    "creditor", "currency", "customer", "customers", "deal", "debit", "debt", "debtor",
    "debts", "default", "deposit", "discount", "dividend", "dollar", "dollars", "donate",
    "donation", "earn", "earned", "earning", "economic", "economics", "economy",
    "escrow", "euro", "euros", "expense", "expensive", "fee", "fees", "finance",
    "financial", "fine", "fined", "firm", "forgave", "forgive", "forgiven", "free",
    "fund", "funds", "gamble", "gambling", "grant", "hire", "hired", "income",
    "inherit", "inheritance", "insolvent", "interest", "invest", "investment", "invoice",
    "lease", "ledger", "lend", "lender", "lending", "liabilities", "liability", "loan",
    "loans", "lottery", "luxury", "margin", "market", "markets", "merchant", "million",
    "monetary", "money", "mortgage", "net", "owe", "owed", "owes", "owing", "paid",
    "pay", "payable", "paying", "payment", "payments", "payout", "payroll", "pennies",
    "penny", "percent", "percentage", "poor", "portfolio", "pound", "pounds", "poverty",
    "price", "priced", "prices", "pricing", "principal", "profit", "profitable",
    "profits", "purchase", "purchased", "quote", "rate", "rebate", "receipt", "redeem",
    "redeemed", "refund", "rent", "rental", "repay", "repayment", "retail", "revenue",
    "rich", "riches", "salary", "sale", "sales", "save", "savings", "sell", "seller",
    "selling", "settlement", "share", "shares", "shop", "sold", "spend", "spending",
    "spent", "split", "stake", "stakes", "stock", "store", "subsidy", "surcharge",
    "surplus", "tariff", "tax", "taxes", "thousand", "tip", "tipped", "tipping",
    "trade", "trading", "transaction", "transactional", "treasury", "tuition", "unpaid",
    "usury", "valuation", "value", "valued", "vendor", "wage", "wages", "wallet",
    "wealth", "wealthy", "wholesale", "windfall", "worth", "worthless", "yen",
})

# Strictly banned in clinical transaction stimuli: explicit emotional, moral, or
# metaphysical vocabulary that would trivially cue the concept directions lexically.
BANNED_AFFECTIVE_TERMS: frozenset[str] = frozenset({
    "abhorrent", "affection", "afraid", "agony", "anger", "angry", "anguish", "anxious",
    "ashamed", "awe", "betray", "betrayal", "bitter", "blame", "blessed", "bondage",
    "brutal", "burden", "caged", "calm", "captive", "captivity", "caring", "catastrophic",
    "chained", "cleanse", "cleansing", "compassion", "compassionate", "contaminate",
    "contaminated", "contamination", "corrupt", "cruel", "cruelty", "cursed", "defile",
    "defiled", "degrading", "desecrate", "desecration", "despair", "desperate",
    "desperation", "dirty", "disgrace", "disgust", "disgusting", "dishonor", "dread",
    "dreadful", "empathy", "empathetic", "enslaved", "evil", "exploit", "exploitation",
    "exploitative", "fear", "fearful", "filthy", "furious", "fury", "generous", "gentle",
    "greedy", "greed", "grief", "guilt", "guilty", "hate", "hatred", "heartless", "holy",
    "honor", "horrified", "horror", "humiliate", "humiliated", "humiliating",
    "humiliation", "immoral", "impure", "impurity", "indignant", "indignation",
    "insult", "insulted", "insulting", "intimidated", "joy", "joyful", "kindness",
    "loving", "malice", "merciless", "miserable", "moral", "morally", "outrage",
    "outraged", "panic", "panicked", "penance", "pitiless", "pity", "polluted",
    "predator", "predatory", "pride", "profane", "punish", "punishment", "pure", "rage",
    "reckless", "regret", "remorse", "resent", "resentment", "reverence", "ruthless",
    "sacred", "sacrilege", "sad", "sanctity", "scared", "shackled", "shackles", "shame",
    "shameful", "sin", "sinful", "slavish", "spite", "spiteful", "stained", "stigma",
    "suffering", "taboo", "taint", "tainted", "tender", "terrified", "terror", "thrill",
    "torment", "trapped", "transgression", "unclean", "unfair", "unfree", "vengeance",
    "vengeful", "vicious", "vile", "violate", "wicked", "warmth", "wretched", "wrongdoing",
})

WORD_RE = re.compile(r"[A-Za-z]+")
DIGIT_OR_CURRENCY_RE = re.compile(r"[0-9$€£¥¢%]")


@dataclass(frozen=True)
class ConceptPair:
    pair_id: str
    concept: str
    positive: str
    negative: str
    domain_tag: str


@dataclass(frozen=True)
class TransactionPair:
    fixture_id: str
    archetype: str
    hypothesis_concepts: tuple[str, ...]
    archetype_scenario: str
    control_scenario: str
    deviation_option: str   # Human/psychological anomaly choice (X)
    rational_option: str    # Payoff-maximizing / invariant homo economicus choice (Y)
    invariance_note: str


@dataclass(frozen=True)
class SanityItem:
    item_id: str
    category: str
    scenario: str
    better_option: str
    worse_option: str
    check_note: str


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def audit_concept_pair(pair: ConceptPair) -> list[str]:
    errors: list[str] = []
    if pair.concept not in ALL_CONCEPTS:
        errors.append(f"{pair.pair_id}: unknown concept '{pair.concept}'")
    for label, text in (("positive", pair.positive), ("negative", pair.negative)):
        if DIGIT_OR_CURRENCY_RE.search(text):
            errors.append(f"{pair.pair_id}.{label}: contains digit or currency symbol: {text!r}")
        words = tokenize_words(text)
        leaked = sorted(set(words) & BANNED_FINANCIAL_TERMS)
        if leaked:
            errors.append(f"{pair.pair_id}.{label}: contains banned financial/dual-use words {leaked}")
        if len(words) < 12 or len(words) > 55:
            errors.append(f"{pair.pair_id}.{label}: word count {len(words)} outside [12, 55]")
    if pair.positive.strip() == pair.negative.strip():
        errors.append(f"{pair.pair_id}: positive and negative texts are identical")
    return errors


def audit_transaction_pair(pair: TransactionPair) -> list[str]:
    errors: list[str] = []
    if pair.archetype not in CORE_ARCHETYPES:
        errors.append(f"{pair.fixture_id}: unknown archetype '{pair.archetype}'")
    for c in pair.hypothesis_concepts:
        if c not in CORE_CONCEPTS:
            errors.append(f"{pair.fixture_id}: unknown hypothesis concept '{c}'")
    fields = (
        ("archetype_scenario", pair.archetype_scenario),
        ("control_scenario", pair.control_scenario),
        ("deviation_option", pair.deviation_option),
        ("rational_option", pair.rational_option),
    )
    for label, text in fields:
        words = tokenize_words(text)
        leaked = sorted(set(words) & BANNED_AFFECTIVE_TERMS)
        if leaked:
            errors.append(f"{pair.fixture_id}.{label}: contains banned affective words {leaked}")
    if pair.archetype_scenario.strip() == pair.control_scenario.strip():
        errors.append(f"{pair.fixture_id}: archetype and control scenarios are identical")
    if pair.deviation_option.strip() == pair.rational_option.strip():
        errors.append(f"{pair.fixture_id}: deviation and rational options are identical")
    return errors


def audit_sanity_item(item: SanityItem) -> list[str]:
    errors: list[str] = []
    for label, text in (
        ("scenario", item.scenario),
        ("better_option", item.better_option),
        ("worse_option", item.worse_option),
    ):
        words = tokenize_words(text)
        leaked = sorted(set(words) & BANNED_AFFECTIVE_TERMS)
        if leaked:
            errors.append(f"{item.item_id}.{label}: contains banned affective words {leaked}")
    if item.better_option.strip() == item.worse_option.strip():
        errors.append(f"{item.item_id}: options are identical")
    return errors


def load_concepts(path: Path = STIMULI_DIR / "concepts.jsonl") -> list[ConceptPair]:
    items: list[ConceptPair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                items.append(ConceptPair(**row))
    return items


def load_transactions(path: Path = STIMULI_DIR / "transactions.jsonl") -> list[TransactionPair]:
    items: list[TransactionPair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["hypothesis_concepts"] = tuple(row["hypothesis_concepts"])
                items.append(TransactionPair(**row))
    return items


def load_sanity_items(path: Path = STIMULI_DIR / "sanity_arithmetic.jsonl") -> list[SanityItem]:
    items: list[SanityItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(SanityItem(**json.loads(line)))
    return items


def compute_stimulus_manifest(stimuli_dir: Path = STIMULI_DIR) -> dict[str, str]:
    manifest: dict[str, str] = {}
    hasher_all = hashlib.sha256()
    for name in ("concepts.jsonl", "transactions.jsonl", "sanity_arithmetic.jsonl"):
        p = stimuli_dir / name
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        manifest[name] = digest
        hasher_all.update(name.encode("utf-8") + b":" + raw + b"\n")
    manifest["combined_sha256"] = hasher_all.hexdigest()
    return manifest


def validate_all_stimuli(stimuli_dir: Path = STIMULI_DIR) -> dict[str, object]:
    concepts = load_concepts(stimuli_dir / "concepts.jsonl")
    txs = load_transactions(stimuli_dir / "transactions.jsonl")
    sanity = load_sanity_items(stimuli_dir / "sanity_arithmetic.jsonl")

    errors: list[str] = []
    for c in concepts:
        errors.extend(audit_concept_pair(c))
    for t in txs:
        errors.extend(audit_transaction_pair(t))
    for s in sanity:
        errors.extend(audit_sanity_item(s))

    # Check uniqueness of IDs and balanced counts
    if len({c.pair_id for c in concepts}) != len(concepts):
        errors.append("Duplicate pair_id in concepts.jsonl")
    if len({t.fixture_id for t in txs}) != len(txs):
        errors.append("Duplicate fixture_id in transactions.jsonl")
    if len({s.item_id for s in sanity}) != len(sanity):
        errors.append("Duplicate item_id in sanity_arithmetic.jsonl")

    concept_counts = {k: sum(1 for c in concepts if c.concept == k) for k in ALL_CONCEPTS}
    tx_counts = {k: sum(1 for t in txs if t.archetype == k) for k in CORE_ARCHETYPES}

    for k, n in concept_counts.items():
        if n < 16:
            errors.append(f"Concept '{k}' has only {n} pairs (expected >= 16)")
    for k, n in tx_counts.items():
        if n < 12:
            errors.append(f"Archetype '{k}' has only {n} fixtures (expected >= 12)")
    if len(sanity) < 20:
        errors.append(f"Sanity bank has only {len(sanity)} items (expected >= 20)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "n_concept_pairs": len(concepts),
        "n_transaction_pairs": len(txs),
        "n_sanity_items": len(sanity),
        "concept_counts": concept_counts,
        "archetype_counts": tx_counts,
        "manifest": compute_stimulus_manifest(stimuli_dir),
    }


def main_audit() -> None:
    report = validate_all_stimuli()
    if not report["valid"]:
        for err in report["errors"]:
            print(f"[ERROR] {err}")
        raise SystemExit(1)
    print("Stimulus bank passed all lexical firewall & symmetry checks.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main_audit()
