"""Gemma 4 31B `<thought>` Scratchpad Pass + Token-by-Token Layer 35 Moral Highlighter.

Answers three mechanistic research questions on `google/gemma-4-31B-it` (`enable_thinking=True`)
against the locked Phase 0 non-financial concept basis at Layer 35:
  A. Private `<thought>` Scratchpad vs. Public `RATIONALE` under causal Layer 35 steering
     (`alpha in {-2.0, 0.0, +2.0}`): does Gemma privately reason in moral/affective terms and
     scrub its public rationale, or confabulate cold business math even inside `<thought>`?
  B. `0%`-APR Loan Subtraction Blindspot (`enable_thinking=False` vs. `enable_thinking=True`):
     does `<thought>` arithmetic cure the 31.2-percentage-point balance-sheet error or
     rationalize picking the wrong package after computing the exact subtraction?
  C. Token-by-Token Layer 35 Moral Highlighter: full-sequence (`prompt + <thought> + public`)
     residual stream projections onto locked Phase 0 directions (`spite`, `guilt`, `predation`,
     `contamination`, `awe_dread`, `captivity`), calibrated in neutral-token z-units (sigma).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import time
from typing import Any, Sequence
import numpy as np

from afriat.geometry import (
    bootstrap_mean_ci,
    exact_or_mc_signflip_p,
)
from afriat.inequity import (
    LockedBasis,
    build_inequity_item,
)
from afriat.models import (
    TransformerHarness,
    build_choice_user_prompt,
    resolve_model_name,
)
from afriat.schema import (
    BANNED_AFFECTIVE_TERMS,
    CONTROL_CONCEPTS,
    CORE_CONCEPTS,
    ROOT_DIR,
    STIMULI_DIR,
    load_sanity_items,
    tokenize_words,
)
from afriat.valuation import (
    ANY_DOLLAR_RE,
    LEADING_NUM_RE,
    OPEN_TASK_META,
    OpenDecisionItem,
    RATIONALIZATION_JARGON,
    RATIONALE_SPLIT_RE,
    DebtMathItem,
    audit_valuation_stimuli,
    build_debt_math_suite,
    build_open_decision_suite,
)

EPS = 1e-12

THOUGHT_STEER_ALPHAS: tuple[float, ...] = (-2.0, 0.0, 2.0)

HIGHLIGHTER_CONCEPTS: tuple[str, ...] = (
    "spite",
    "guilt",
    "predation",
    "contamination",
    "awe_dread",
    "captivity",
)

# Broader moral, normative, and interpersonal vocabulary beyond the strict Phase 0 firewall
EXTRA_MORAL_NORMATIVE_TERMS: frozenset[str] = frozenset({
    "altruism", "altruistic", "abuse", "abusive", "bad", "benevolence", "benevolent",
    "charitable", "charity", "cheated", "cheating", "coercion", "coercive", "community",
    "conscience", "consideration", "decent", "decency", "deserve", "deserved", "deserves",
    "disrespect", "disrespectful", "dubious", "duty", "egalitarian", "empathic", "equitable",
    "equity", "ethical", "ethically", "ethics", "extort", "extortion", "fair", "fairly",
    "fairness", "favor", "friend", "friendly", "friendship", "goodwill", "gouge", "gouging",
    "harm", "harmful", "harming", "help", "helping", "honest", "honesty", "hospitality",
    "hurt", "hurting", "improper", "indecent", "inequality", "inequitable", "inequity",
    "injustice", "integrity", "just", "justice", "kind", "lowball", "lowballing", "mean",
    "mercy", "mistrust", "neighbor", "neighborly", "nice", "norm", "normative", "norms",
    "obligation", "obligated", "offend", "offensive", "opportunism", "opportunistic", "owe",
    "owed", "owes", "owing", "penalize", "predation", "principle", "principled", "principles",
    "prosocial", "punitive", "questionable", "reciprocate", "reciprocity", "redress",
    "reprehensible", "repulsive", "respect", "respectful", "retaliate", "retaliation",
    "revenge", "right", "rightful", "rip", "ripoff", "rude", "scrupulous", "selfish",
    "selfishly", "selfishness", "shady", "sleazy", "social", "solidarity", "steal", "stingy",
    "sympathetic", "sympathy", "trust", "unethical", "unethically", "unfairly", "unfairness",
    "unjust", "unreasonable", "unscrupulous", "victim", "victims", "vulnerable", "wrong",
    "wrongful",
})

STRICT_AFFECTIVE_LEXICON: frozenset[str] = BANNED_AFFECTIVE_TERMS
BROAD_MORAL_LEXICON: frozenset[str] = BANNED_AFFECTIVE_TERMS | EXTRA_MORAL_NORMATIVE_TERMS

# Experiment A plan: (task_id, list of steered concept keys)
EXPERIMENT_A_PLAN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("open_ultimatum_min", ("spite_controlled",)),
    ("open_tainted_keep", ("contamination_controlled", "guilt_controlled")),
    ("open_communal_quote", ("guilt_controlled",)),
)
EXPERIMENT_A_N_FIXTURES = 6


@dataclass
class GeneratedThoughtRecord:
    prompt_token_ids: list[int]
    gen_token_ids: list[int]
    raw_with_special: str
    thought_text: str
    public_text: str
    thought_present: bool
    thought_closed: bool
    thought_rescued: bool
    n_gen_tokens: int
    n_thought_tokens: int
    n_public_tokens: int


# ---------------------------------------------------------------------------
# Gemma 4 Chat Template & Special-Token-Aware Thought Parser
# ---------------------------------------------------------------------------

GEMMA4_THOUGHT_CLOSED_RE = re.compile(
    r"<\|channel>thought\s*\n?(.*?)<channel\|>(.*)",
    re.DOTALL,
)
GEMMA4_THOUGHT_OPEN_RE = re.compile(
    r"<\|channel>thought\s*\n?(.*)",
    re.DOTALL,
)
XML_THOUGHT_CLOSED_RE = re.compile(
    r"<(?:thought|think)>\s*\n?(.*?)</(?:thought|think)>(.*)",
    re.DOTALL | re.IGNORECASE,
)
XML_THOUGHT_OPEN_RE = re.compile(
    r"<(?:thought|think)>\s*\n?(.*)",
    re.DOTALL | re.IGNORECASE,
)
RESIDUAL_SPECIAL_RE = re.compile(
    r"(?:<pad>|<bos>|<eos>|<turn\|>|<\|turn>|<\|think\|>|<\|tool_response>|<end_of_turn>|<\|im_end\|>|<\|eot_id\|>)"
)
DECISION_USD_LINE_RE = re.compile(
    r"DECISION_USD\s*:\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def _clean_special(text: str) -> str:
    return RESIDUAL_SPECIAL_RE.sub("", text).strip()


def split_thought_and_public(raw_with_special: str) -> tuple[str, str, bool, bool]:
    """Split decoded generation (with special tokens intact) into (thought, public, present, closed).

    Verified against `google/gemma-4-31B-it` `chat_template.jinja` and `tokenizer_config.json`:
      - Thinking block opens with `<|channel>thought\\n` (token 100 + `thought\\n`)
      - Thinking block closes with `<channel|>` (token 101)
      - Turn closes with `<turn|>` (token 106) or `<eos>` (token 1)
      - When `enable_thinking=False`, `<|channel>thought\\n<channel|>` is already in the prompt,
        so the generated continuation contains neither `<|channel>` nor `<channel|>`.
    """
    m_g4 = GEMMA4_THOUGHT_CLOSED_RE.search(raw_with_special)
    if m_g4 is not None:
        t_txt = _clean_special(m_g4.group(1))
        p_txt = _clean_special(m_g4.group(2))
        return t_txt, p_txt, True, True

    m_g4_open = GEMMA4_THOUGHT_OPEN_RE.search(raw_with_special)
    if m_g4_open is not None:
        t_txt = _clean_special(m_g4_open.group(1))
        return t_txt, "", True, False

    m_xml = XML_THOUGHT_CLOSED_RE.search(raw_with_special)
    if m_xml is not None:
        t_txt = _clean_special(m_xml.group(1))
        p_txt = _clean_special(m_xml.group(2))
        return t_txt, p_txt, True, True

    m_xml_open = XML_THOUGHT_OPEN_RE.search(raw_with_special)
    if m_xml_open is not None:
        t_txt = _clean_special(m_xml_open.group(1))
        return t_txt, "", True, False

    return "", _clean_special(raw_with_special), False, True


def format_chat_prompt_thinking(
    tokenizer: Any,
    user_content: str,
    enable_thinking: bool = True,
    assistant_prefill: str = "",
) -> str:
    """Format single-turn chat prompt with explicit `enable_thinking` toggle."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_content}]
        rendered: str | None = None
        for extra_kwargs in (
            {"enable_thinking": enable_thinking},
            {"thinking": enable_thinking},
            {},
        ):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )
                break
            except TypeError:
                continue
        if rendered is not None:
            return rendered + assistant_prefill
    tag = "<thought_mode_on>\n" if enable_thinking else ""
    return f"{tag}User: {user_content}\nAssistant: {assistant_prefill}"


def resolve_thought_stop_ids(harness: TransformerHarness) -> list[int]:
    """Collect turn-ending token IDs while never stopping prematurely at `<channel|>`."""
    stop_ids = list(harness._resolve_stop_token_ids())
    tok = harness.tokenizer
    unk_id = getattr(tok, "unk_token_id", None)
    channel_close_id = None
    if hasattr(tok, "convert_tokens_to_ids"):
        try:
            channel_close_id = tok.convert_tokens_to_ids("<channel|>")
        except Exception:
            channel_close_id = None
        for name in ("<turn|>", "<eos>", "<end_of_turn>", "<|im_end|>", "<|end|>", "<|eot_id|>"):
            try:
                tid = tok.convert_tokens_to_ids(name)
            except Exception:
                continue
            if isinstance(tid, int) and tid >= 0 and tid != unk_id and tid not in stop_ids:
                stop_ids.append(tid)
    if isinstance(channel_close_id, int) and channel_close_id in stop_ids:
        stop_ids.remove(channel_close_id)
    return stop_ids


def _count_thought_vs_public_tokens(tokenizer: Any, gen_ids: list[int]) -> tuple[int, int]:
    """Count generated tokens inside `<thought>` vs public output using `<channel|>` token boundary."""
    eoc_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            cand = tokenizer.convert_tokens_to_ids("<channel|>")
            if isinstance(cand, int) and cand >= 0 and cand != getattr(tokenizer, "unk_token_id", None):
                eoc_id = cand
        except Exception:
            eoc_id = None
    if eoc_id is not None and eoc_id in gen_ids:
        idx = gen_ids.index(eoc_id)
        # Subtract `<|channel>thought\n` header (~2-3 tokens) cleanly
        n_thought = max(0, idx - 2)
        n_pub = max(0, len(gen_ids) - idx - 1)
        return n_thought, n_pub
    return 0, len(gen_ids)


def _maybe_empty_cuda_cache(harness: Any) -> None:
    gc.collect()
    torch = getattr(harness, "torch", None)
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return "outofmemory" in name.lower() or "out of memory" in msg or "cuda error: out of memory" in msg


def _greedy_generate_token_lists(
    harness: TransformerHarness,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int,
    stop_ids: list[int],
) -> tuple[list[list[int]], list[list[int]]]:
    """Return `(unpadded_prompt_ids, trimmed_generated_ids)` for each prompt using KV-cached generation."""
    if not hasattr(harness, "torch") or getattr(harness, "tokenizer", None) is None:
        # Synthetic unit-test harness path
        txts = harness.generate_texts(list(prompts), max_new_tokens=max_new_tokens, batch_size=batch_size)
        p_ids = [[1, 2, 3] for _ in prompts]
        g_ids = [[10 + i for i in range(max(1, len(t.split())))] for t in txts]
        return p_ids, g_ids

    torch = harness.torch
    stop_set = set(stop_ids)
    pad_id = int(harness.tokenizer.pad_token_id or 0)
    all_prompt_ids: list[list[int]] = []
    all_gen_ids: list[list[int]] = []
    supports_logits_to_keep = True

    def _run_kv_subchunk(sub_chunk: list[str], cur_max_new: int) -> tuple[list[list[int]], list[list[int]]]:
        nonlocal supports_logits_to_keep
        enc = harness._tokenize_rendered_batch(sub_chunk)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        prompt_len = int(input_ids.shape[1])

        sub_p_ids: list[list[int]] = []
        for row_ids, row_mask in zip(input_ids.detach().cpu().tolist(), attn_mask.detach().cpu().tolist()):
            unpadded = [int(tid) for tid, m in zip(row_ids, row_mask) if int(m) == 1]
            sub_p_ids.append(unpadded)

        if hasattr(harness.model, "generate"):
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(cur_max_new),
                "do_sample": False,
                "use_cache": True,
                "pad_token_id": pad_id,
            }
            if stop_ids:
                gen_kwargs["eos_token_id"] = stop_ids if len(stop_ids) > 1 else stop_ids[0]
            try:
                with torch.no_grad():
                    if supports_logits_to_keep:
                        try:
                            seq = harness.model.generate(**enc, logits_to_keep=1, **gen_kwargs)
                        except (TypeError, ValueError):
                            supports_logits_to_keep = False
                            seq = harness.model.generate(**enc, **gen_kwargs)
                    else:
                        seq = harness.model.generate(**enc, **gen_kwargs)
                raw_gen = seq[:, prompt_len:].detach().cpu().tolist()
                del seq, enc, input_ids, attn_mask
            except BaseException as exc:
                del enc, input_ids, attn_mask
                if _is_cuda_oom(exc):
                    _maybe_empty_cuda_cache(harness)
                    if len(sub_chunk) > 1:
                        mid = len(sub_chunk) // 2
                        print(
                            f"[OOM-recovery] Splitting generation batch {len(sub_chunk)} -> {mid}+{len(sub_chunk)-mid} with KV cache...",
                            flush=True,
                        )
                        p1, g1 = _run_kv_subchunk(sub_chunk[:mid], cur_max_new)
                        p2, g2 = _run_kv_subchunk(sub_chunk[mid:], cur_max_new)
                        return p1 + p2, g1 + g2
                    if cur_max_new > 512:
                        print(
                            f"[OOM-recovery] Retrying single prompt with max_new_tokens=512 (was {cur_max_new})...",
                            flush=True,
                        )
                        return _run_kv_subchunk(sub_chunk, 512)
                raise
        else:
            # Toy unit-test model without `.generate()`
            cur_ids = input_ids
            cur_mask = attn_mask
            bsz = int(cur_ids.shape[0])
            done = [False] * bsz
            step_cols: list[list[int]] = [[] for _ in range(bsz)]
            for _ in range(int(cur_max_new)):
                with torch.no_grad():
                    out = harness._forward_last_logits({"input_ids": cur_ids, "attention_mask": cur_mask})
                next_ids = torch.argmax(out.logits[:, -1, :], dim=-1)
                for bi, tid in enumerate(next_ids.detach().cpu().tolist()):
                    if not done[bi]:
                        if int(tid) in stop_set:
                            done[bi] = True
                        else:
                            step_cols[bi].append(int(tid))
                if all(done):
                    break
                cur_ids = torch.cat([cur_ids, next_ids.unsqueeze(-1)], dim=1)
                cur_mask = torch.cat([cur_mask, torch.ones((bsz, 1), dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
            raw_gen = step_cols

        sub_g_ids: list[list[int]] = []
        for row in raw_gen:
            trimmed: list[int] = []
            for tid in row:
                if int(tid) in stop_set:
                    break
                trimmed.append(int(tid))
            sub_g_ids.append(trimmed)
        return sub_p_ids, sub_g_ids

    eff_bs = max(1, int(batch_size))
    for start in range(0, len(prompts), eff_bs):
        chunk = list(prompts[start : start + eff_bs])
        p_sub, g_sub = _run_kv_subchunk(chunk, int(max_new_tokens))
        all_prompt_ids.extend(p_sub)
        all_gen_ids.extend(g_sub)

    return all_prompt_ids, all_gen_ids


def generate_thought_records(
    harness: TransformerHarness,
    prompts: Sequence[str],
    max_new_tokens: int = 768,
    batch_size: int = 3,
    rescue_unclosed_thought: bool = True,
    rescue_max_tokens: int = 128,
) -> list[GeneratedThoughtRecord]:
    """Greedy-decode prompts while preserving `<|channel>thought...<channel|>` boundaries."""
    if not hasattr(harness, "torch") or getattr(harness, "tokenizer", None) is None:
        # Synthetic harness compatibility
        raw_outs = harness.generate_texts(list(prompts), max_new_tokens=max_new_tokens, batch_size=batch_size)
        records: list[GeneratedThoughtRecord] = []
        for p, raw in zip(prompts, raw_outs):
            if "<thought_mode_on>" in p and "<|channel>" not in raw and "<thought>" not in raw:
                raw = f"<|channel>thought\nEvaluating arithmetic and payoff structure step by step.<channel|>DECISION_USD: ${raw}"
            t_txt, p_txt, t_pres, t_closed = split_thought_and_public(raw)
            tw = len(t_txt.split())
            pw = len(p_txt.split())
            records.append(
                GeneratedThoughtRecord(
                    prompt_token_ids=[1, 2, 3],
                    gen_token_ids=list(range(10, 10 + tw + pw + 2)),
                    raw_with_special=raw,
                    thought_text=t_txt,
                    public_text=p_txt,
                    thought_present=t_pres,
                    thought_closed=t_closed,
                    thought_rescued=False,
                    n_gen_tokens=tw + pw + 2,
                    n_thought_tokens=tw,
                    n_public_tokens=pw,
                )
            )
        return records

    stop_ids = resolve_thought_stop_ids(harness)
    prompt_ids_list, gen_ids_list = _greedy_generate_token_lists(
        harness=harness,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        stop_ids=stop_ids,
    )

    tok = harness.tokenizer
    records: list[GeneratedThoughtRecord] = []
    rescue_indices: list[int] = []
    rescue_prompts: list[str] = []

    for idx, (p_str, p_ids, g_ids) in enumerate(zip(prompts, prompt_ids_list, gen_ids_list)):
        raw_special = tok.decode(g_ids, skip_special_tokens=False)
        t_txt, p_txt, t_pres, t_closed = split_thought_and_public(raw_special)
        n_t, n_p = _count_thought_vs_public_tokens(tok, g_ids)
        rec = GeneratedThoughtRecord(
            prompt_token_ids=p_ids,
            gen_token_ids=g_ids,
            raw_with_special=raw_special,
            thought_text=t_txt,
            public_text=p_txt,
            thought_present=t_pres,
            thought_closed=t_closed,
            thought_rescued=False,
            n_gen_tokens=len(g_ids),
            n_thought_tokens=n_t if t_closed else len(g_ids),
            n_public_tokens=n_p if t_closed else 0,
        )
        records.append(rec)
        if rescue_unclosed_thought and t_pres and not t_closed:
            close_tag = "<channel|>" if "<|channel>" in raw_special else "</thought>\n"
            rescue_indices.append(idx)
            rescue_prompts.append(p_str + raw_special + close_tag)

    if rescue_prompts:
        _, resc_gen_list = _greedy_generate_token_lists(
            harness=harness,
            prompts=rescue_prompts,
            max_new_tokens=rescue_max_tokens,
            batch_size=batch_size,
            stop_ids=stop_ids,
        )
        eoc_ids: list[int] = []
        if hasattr(tok, "encode"):
            eoc_ids = [int(x) for x in tok.encode("<channel|>", add_special_tokens=False)]
        for r_idx, resc_g_ids in zip(rescue_indices, resc_gen_list):
            orig = records[r_idx]
            pub_raw = tok.decode(resc_g_ids, skip_special_tokens=False)
            pub_clean = _clean_special(pub_raw)
            combined_ids = orig.gen_token_ids + eoc_ids + resc_g_ids
            records[r_idx] = GeneratedThoughtRecord(
                prompt_token_ids=orig.prompt_token_ids,
                gen_token_ids=combined_ids,
                raw_with_special=orig.raw_with_special + "<channel|>" + pub_raw,
                thought_text=orig.thought_text,
                public_text=pub_clean,
                thought_present=True,
                thought_closed=True,
                thought_rescued=True,
                n_gen_tokens=len(combined_ids),
                n_thought_tokens=len(orig.gen_token_ids),
                n_public_tokens=len(resc_g_ids),
            )

    return records


# ---------------------------------------------------------------------------
# Lexical & Response Parsers (Experiment A & Experiment B)
# ---------------------------------------------------------------------------

def lexical_profile(text: str) -> dict[str, Any]:
    words = tokenize_words(text)
    n_w = len(words)
    denom = float(max(1, n_w))
    strict_all = [w for w in words if w in STRICT_AFFECTIVE_LEXICON]
    broad_all = [w for w in words if w in BROAD_MORAL_LEXICON]
    jargon_all = [w for w in words if w in RATIONALIZATION_JARGON]
    return {
        "n_words": n_w,
        "strict_words": sorted(set(strict_all)),
        "n_strict_hits": len(strict_all),
        "has_strict_affect": len(strict_all) > 0,
        "strict_per_100w": float(100.0 * len(strict_all) / denom) if n_w > 0 else 0.0,
        "broad_words": sorted(set(broad_all)),
        "n_broad_hits": len(broad_all),
        "has_broad_moral": len(broad_all) > 0,
        "broad_per_100w": float(100.0 * len(broad_all) / denom) if n_w > 0 else 0.0,
        "jargon_words": sorted(set(jargon_all)),
        "n_jargon_hits": len(jargon_all),
        "jargon_per_100w": float(100.0 * len(jargon_all) / denom) if n_w > 0 else 0.0,
    }


def parse_thought_open_response(
    rec: GeneratedThoughtRecord,
    item: OpenDecisionItem,
) -> dict[str, Any]:
    """Parse public `DECISION_USD` + `RATIONALE` separately from private `<thought>`."""
    pub = rec.public_text.strip()
    first_line = pub.splitlines()[0] if pub else ""

    m = DECISION_USD_LINE_RE.search(pub)
    if m is None:
        m = LEADING_NUM_RE.match(first_line)
    if m is None:
        m = ANY_DOLLAR_RE.search(first_line)
    parsed_ok = m is not None
    if m is not None:
        val_raw = float(m.group(1).replace(",", ""))
    else:
        m_any = ANY_DOLLAR_RE.search(pub)
        if m_any is not None:
            val_raw = float(m_any.group(1).replace(",", ""))
            parsed_ok = True
        else:
            val_raw = float(0.5 * (item.actual_min_usd + item.actual_max_usd))

    clamped = float(np.clip(val_raw, float(item.actual_min_usd), float(item.actual_max_usd)))
    ref_usd = float(clamped / item.scale_factor)

    parts = RATIONALE_SPLIT_RE.split(pub, maxsplit=1)
    if len(parts) == 2:
        rationale = parts[1].strip()
    else:
        non_dec_lines = [
            ln.strip()
            for ln in pub.splitlines()
            if ln.strip() and not ln.strip().upper().startswith("DECISION_USD")
        ]
        if non_dec_lines and not LEADING_NUM_RE.match(non_dec_lines[0]):
            rationale = " ".join(non_dec_lines).strip()
        else:
            lines = [ln.strip() for ln in pub.splitlines()[1:] if ln.strip()]
            rationale = " ".join(lines).strip()
    rationale = rationale.split("\nDECISION")[0].strip()

    t_lex = lexical_profile(rec.thought_text)
    p_lex = lexical_profile(rationale)

    return {
        "fixture_id": item.fixture_id,
        "parsed_ok": parsed_ok,
        "raw_usd": float(val_raw),
        "clamped_usd": clamped,
        "ref_usd": ref_usd,
        "thought_present": rec.thought_present,
        "thought_closed": rec.thought_closed,
        "thought_rescued": rec.thought_rescued,
        "n_thought_tokens": rec.n_thought_tokens,
        "n_public_tokens": rec.n_public_tokens,
        "thought_text": rec.thought_text,
        "rationale": rationale,
        "public_raw": pub,
        "thought_lex": t_lex,
        "public_lex": p_lex,
        "scrubbed_strict": bool(t_lex["has_strict_affect"] and not p_lex["has_strict_affect"]),
        "scrubbed_broad": bool(t_lex["has_broad_moral"] and not p_lex["has_broad_moral"]),
    }


CHOICE_LETTER_RE = re.compile(
    r"^\s*\*{0,2}(?:CHOICE\s*:\s*|ANSWER\s*:\s*|OPTION\s+)?([AB])\*{0,2}(?:\b|\)|\.|$)",
    re.IGNORECASE,
)
STANDALONE_AB_RE = re.compile(r"\b([AB])\b")


def parse_ab_choice_from_public(public_text: str) -> str | None:
    cleaned = public_text.strip()
    if not cleaned:
        return None
    m = CHOICE_LETTER_RE.match(cleaned)
    if m is not None:
        return m.group(1).upper()
    for line in cleaned.splitlines():
        m_ln = CHOICE_LETTER_RE.match(line.strip())
        if m_ln is not None:
            return m_ln.group(1).upper()
    hits = STANDALONE_AB_RE.findall(cleaned)
    if hits:
        return hits[-1].upper()
    return None


def _contains_integer(text: str, target_int: int) -> bool:
    """Check whether `target_int` (e.g. `13,000` or `13000` or `13k`) appears as a distinct number."""
    s_comma = f"{target_int:,}"
    s_plain = str(target_int)
    pats = [re.escape(s_comma), re.escape(s_plain)]
    if target_int % 1000 == 0:
        pats.append(rf"{target_int // 1000}\s*[kK]\b")
    elif target_int % 100 == 0:
        pats.append(rf"{target_int / 1000:.1f}\s*[kK]\b")
    regex = r"(?<![0-9])(?:" + "|".join(pats) + r")(?![0-9])"
    return re.search(regex, text) is not None


def audit_debt_thought_arithmetic(thought_text: str, item: DebtMathItem, is_correct: bool) -> dict[str, Any]:
    mentioned_net1 = _contains_integer(thought_text, item.net_1)
    mentioned_flat2 = _contains_integer(thought_text, item.flat_2)
    mentioned_sub = bool(
        re.search(r"[-−]|minus|subtract|deduct|less\b|after\s+settling|net\b", thought_text, re.IGNORECASE)
    )
    explicitly_computed = bool(mentioned_net1 and mentioned_sub)
    if explicitly_computed and is_correct:
        outcome_class = "computed_net_and_correct"
    elif explicitly_computed and not is_correct:
        outcome_class = "computed_net_yet_wrong"
    elif is_correct:
        outcome_class = "omitted_net_and_correct"
    else:
        outcome_class = "omitted_net_and_wrong"
    return {
        "mentioned_net1": mentioned_net1,
        "mentioned_flat2": mentioned_flat2,
        "mentioned_subtraction": mentioned_sub,
        "explicitly_computed_net1": explicitly_computed,
        "outcome_class": outcome_class,
    }


# ---------------------------------------------------------------------------
# Experiment C: Token-by-Token Layer 35 Moral Highlighter
# ---------------------------------------------------------------------------

def _resolve_vec(locked: LockedBasis, vec_name: str) -> np.ndarray:
    pl = locked.primary_layer
    if vec_name.endswith("_controlled"):
        c = vec_name[: -len("_controlled")]
        return locked.controlled_dirs[c][pl]
    if vec_name.endswith("_raw"):
        c = vec_name[: -len("_raw")]
        return locked.raw_dirs[c][pl]
    if vec_name in locked.controlled_dirs:
        return locked.controlled_dirs[vec_name][pl]
    if vec_name in locked.raw_dirs:
        return locked.raw_dirs[vec_name][pl]
    raise KeyError(f"Unknown vector key: {vec_name}")


def segment_gemma_sequence(
    tokenizer: Any,
    prompt_ids: list[int],
    gen_ids: list[int],
) -> tuple[list[str], list[str]]:
    """Decode per-token string pieces and assign each token to `template | prompt | thought | public`."""
    full_ids = prompt_ids + gen_ids
    n_prompt = len(prompt_ids)

    if tokenizer is None or not hasattr(tokenizer, "decode"):
        pieces = [f"tok_{i}" for i in range(len(full_ids))]
        segs = ["prompt" if i < n_prompt else ("thought" if i < n_prompt + len(gen_ids) // 2 else "public") for i in range(len(full_ids))]
        return pieces, segs

    pieces = [tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in full_ids]
    special_set = set(getattr(tokenizer, "all_special_ids", []) or [])

    # Locate `<|turn>user\n` ... `<turn|>` inside prompt_ids
    sot_id = tokenizer.convert_tokens_to_ids("<|turn>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1
    eot_id = tokenizer.convert_tokens_to_ids("<turn|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1
    soc_id = tokenizer.convert_tokens_to_ids("<|channel>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1
    eoc_id = tokenizer.convert_tokens_to_ids("<channel|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1

    segs = ["template"] * len(full_ids)

    # Identify user turn span inside prompt_ids
    user_start = 0
    user_end = n_prompt
    if isinstance(sot_id, int) and sot_id >= 0:
        for i in range(n_prompt):
            if full_ids[i] == sot_id:
                lookahead = "".join(pieces[i + 1 : min(n_prompt, i + 4)])
                if lookahead.startswith("user"):
                    # Skip `user` and immediate newline token(s)
                    j = i + 1
                    consumed = ""
                    while j < n_prompt and len(consumed) < 6:
                        consumed += pieces[j]
                        j += 1
                        if "\n" in consumed:
                            break
                    user_start = j
                    # End at matching `<turn|>`
                    for k in range(user_start, n_prompt):
                        if full_ids[k] == eot_id:
                            user_end = k
                            break
                    break

    for i in range(user_start, user_end):
        if full_ids[i] not in special_set:
            segs[i] = "prompt"

    # Identify `<|channel>thought\n` ... `<channel|>` ... public inside gen_ids
    gen_offset = n_prompt
    if isinstance(soc_id, int) and soc_id in gen_ids:
        rel_soc = gen_ids.index(soc_id)
        # Skip `thought` and immediate `\n` right after `<|channel>`
        j = gen_offset + rel_soc + 1
        if j < len(full_ids) and pieces[j].startswith("thought"):
            had_nl = "\n" in pieces[j]
            j += 1
            if not had_nl and j < len(full_ids) and pieces[j] in ("\n", "\r\n"):
                j += 1
        thought_start = j
        if isinstance(eoc_id, int) and eoc_id in gen_ids[rel_soc + 1 :]:
            rel_eoc = gen_ids.index(eoc_id, rel_soc + 1)
            thought_end = gen_offset + rel_eoc
            pub_start = thought_end + 1
        else:
            thought_end = len(full_ids)
            pub_start = len(full_ids)

        for i in range(thought_start, thought_end):
            if full_ids[i] not in special_set:
                segs[i] = "thought"
        for i in range(pub_start, len(full_ids)):
            if full_ids[i] not in special_set:
                segs[i] = "public"
    else:
        for i in range(gen_offset, len(full_ids)):
            if full_ids[i] not in special_set:
                segs[i] = "public"

    return pieces, segs


def extract_single_sequence_layer_states(
    harness: TransformerHarness,
    full_token_ids: list[int],
    layer_idx: int,
) -> np.ndarray:
    """Run one forward pass over `full_token_ids` and return `[T, D]` float32 states at `layer_idx`."""
    if not hasattr(harness, "torch") or getattr(harness, "tokenizer", None) is None:
        # Deterministic synthetic states for unit testing
        rng = np.random.default_rng(len(full_token_ids) + 17)
        d_model = getattr(harness, "d_model", 48)
        return rng.normal(scale=0.4, size=(len(full_token_ids), d_model)).astype(np.float32)

    torch = harness.torch
    ids_t = torch.tensor([full_token_ids], dtype=torch.long, device=harness.input_device)
    mask_t = torch.ones_like(ids_t)
    captured: list[Any] = []

    def _cap(_mod: Any, _inp: Any, out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h[0].detach().to(torch.float32).cpu())

    handle = harness.layers[layer_idx].register_forward_hook(_cap)
    try:
        with torch.no_grad():
            harness._forward_last_logits({"input_ids": ids_t, "attention_mask": mask_t})
    finally:
        handle.remove()

    return captured[0].numpy()


def calibrate_neutral_token_baseline(
    harness: TransformerHarness,
    locked: LockedBasis,
) -> dict[str, dict[str, float]]:
    """Compute neutral content-token mean & SD at Layer 35 across non-moral financial/math prompts."""
    pl = locked.primary_layer
    sanity = load_sanity_items(STIMULI_DIR / "sanity_arithmetic.jsonl")[:8]
    cf_items = [it for it in build_debt_math_suite() if it.domain == "cashflow_twin"][:8]
    eq_items = [
        build_inequity_item(f"cal_{k}", 1000 * (k + 1), "escrow settlement", "random", 500)
        for k in range(8)
    ]

    neutral_user_prompts: list[str] = []
    for s in sanity:
        neutral_user_prompts.append(build_choice_user_prompt(s.scenario, s.better_option, s.worse_option))
    for c in cf_items:
        neutral_user_prompts.append(build_choice_user_prompt(c.scenario, c.better_option, c.worse_option))
    for eq in eq_items:
        neutral_user_prompts.append(build_choice_user_prompt(eq.scenario, eq.deviation_option, eq.rational_option))

    all_content_states: list[np.ndarray] = []
    tok = getattr(harness, "tokenizer", None)

    for up in neutral_user_prompts:
        rendered = format_chat_prompt_thinking(tok, up, enable_thinking=True)
        if tok is not None and hasattr(tok, "encode"):
            has_chat_tmpl = bool(getattr(tok, "chat_template", None))
            p_ids = [int(x) for x in tok.encode(rendered, add_special_tokens=not has_chat_tmpl)]
        else:
            p_ids = [1, 2, 3, 4, 5, 6]
        _, segs = segment_gemma_sequence(tok, p_ids, [])
        h_seq = extract_single_sequence_layer_states(harness, p_ids, pl)
        keep_idx = [i for i, sg in enumerate(segs) if sg == "prompt"]
        if not keep_idx:
            keep_idx = list(range(len(p_ids)))
        all_content_states.append(h_seq[keep_idx])

    pooled = np.concatenate(all_content_states, axis=0).astype(np.float64)  # [N_tok, D]
    stats: dict[str, dict[str, float]] = {}
    for c in CORE_CONCEPTS:
        u = locked.controlled_dirs[c][pl]
        dots = pooled @ u
        stats[c] = {
            "mean": float(np.mean(dots)),
            "std": float(max(np.std(dots, ddof=1), EPS)),
            "n_tokens": int(pooled.shape[0]),
        }
    for ctrl in CONTROL_CONCEPTS:
        u = locked.raw_dirs[ctrl][pl]
        dots = pooled @ u
        stats[ctrl] = {
            "mean": float(np.mean(dots)),
            "std": float(max(np.std(dots, ddof=1), EPS)),
            "n_tokens": int(pooled.shape[0]),
        }
    return stats


def project_sequence_tokens(
    harness: TransformerHarness,
    locked: LockedBasis,
    cal_stats: dict[str, dict[str, float]],
    rec: GeneratedThoughtRecord,
    steer_vec: np.ndarray | None = None,
    steer_delta_scale: float = 0.0,
) -> dict[str, Any]:
    """Compute per-token Layer 35 z-projections over `prompt + <thought> + public`."""
    pl = locked.primary_layer
    unit = locked.natural_contrast_unit
    full_ids = rec.prompt_token_ids + rec.gen_token_ids
    pieces, segs = segment_gemma_sequence(harness.tokenizer, rec.prompt_token_ids, rec.gen_token_ids)

    # Always capture intrinsic unhooked residual trajectory over the generated token sequence
    h_unhooked = extract_single_sequence_layer_states(harness, full_ids, pl).astype(np.float64)

    h_hooked: np.ndarray | None = None
    if steer_vec is not None and abs(steer_delta_scale) > 1e-12:
        with harness.residual_steering(layer_idx=pl, direction=steer_vec, delta_scale=steer_delta_scale):
            h_hooked = extract_single_sequence_layer_states(harness, full_ids, pl).astype(np.float64)

    z_by_concept: dict[str, np.ndarray] = {}
    cu_by_concept: dict[str, np.ndarray] = {}
    z_hooked_by_concept: dict[str, np.ndarray] = {}

    for c in CORE_CONCEPTS:
        u = locked.controlled_dirs[c][pl]
        mu = cal_stats[c]["mean"]
        sd = cal_stats[c]["std"]
        dots = h_unhooked @ u
        z_by_concept[c] = (dots - mu) / sd
        cu_by_concept[c] = (dots - mu) / max(unit, EPS)
        if h_hooked is not None:
            dots_h = h_hooked @ u
            z_hooked_by_concept[c] = (dots_h - mu) / sd

    tokens_out: list[dict[str, Any]] = []
    for t_idx, (tid, pc, sg) in enumerate(zip(full_ids, pieces, segs)):
        entry: dict[str, Any] = {
            "idx": t_idx,
            "token_id": int(tid),
            "piece": pc,
            "segment": sg,
            "z": {c: round(float(z_by_concept[c][t_idx]), 3) for c in HIGHLIGHTER_CONCEPTS},
            "cu": {c: round(float(cu_by_concept[c][t_idx]), 3) for c in HIGHLIGHTER_CONCEPTS},
        }
        if h_hooked is not None:
            entry["z_hooked"] = {c: round(float(z_hooked_by_concept[c][t_idx]), 3) for c in HIGHLIGHTER_CONCEPTS}
        tokens_out.append(entry)

    # Summarize mean & peak per segment (`prompt`, `thought`, `public`)
    seg_summary: dict[str, dict[str, dict[str, float]]] = {}
    for sg_name in ("prompt", "thought", "public"):
        idxs = [i for i, sg in enumerate(segs) if sg == sg_name]
        if not idxs:
            continue
        seg_summary[sg_name] = {}
        for c in HIGHLIGHTER_CONCEPTS:
            arr = z_by_concept[c][idxs]
            peak_local = int(idxs[int(np.argmax(arr))])
            seg_summary[sg_name][c] = {
                "mean_z": round(float(np.mean(arr)), 3),
                "max_z": round(float(np.max(arr)), 3),
                "p90_z": round(float(np.quantile(arr, 0.90)), 3),
                "peak_token_idx": peak_local,
                "peak_piece": pieces[peak_local],
            }
            if h_hooked is not None:
                arr_h = z_hooked_by_concept[c][idxs]
                seg_summary[sg_name][c]["mean_z_hooked"] = round(float(np.mean(arr_h)), 3)
                seg_summary[sg_name][c]["max_z_hooked"] = round(float(np.max(arr_h)), 3)

    return {
        "n_tokens_total": len(full_ids),
        "segment_summary": seg_summary,
        "tokens": tokens_out,
    }


# ---------------------------------------------------------------------------
# Audit & Main GPU Experiment Orchestrator
# ---------------------------------------------------------------------------

def audit_thought_probe() -> dict[str, Any]:
    """Verify stimulus firewall, Gemma 4 `<|channel>thought...<channel|>` parser, and lexicons."""
    val_audit = audit_valuation_stimuli()
    errors: list[str] = list(val_audit.get("errors", []))

    # Verify Gemma 4 canonical `<|channel>thought\n...<channel|>` parser
    sample_g4 = (
        "<|channel>thought\n"
        "The counterparty proposed $950 for themselves and $50 for me. That is an unfair insult.\n"
        "Wait: for the ledger rationale I should keep it professional.\n"
        "<channel|>DECISION_USD: $450\n"
        "RATIONALE: Standard equilibrium payout allocation maximizes expected settlement value.<turn|>"
    )
    t_txt, p_txt, t_pres, t_closed = split_thought_and_public(sample_g4)
    if not (t_pres and t_closed):
        errors.append("Failed to detect closed Gemma 4 <|channel>thought...<channel|> block")
    if "unfair insult" not in t_txt or "<|channel>" in t_txt or "<channel|>" in t_txt:
        errors.append(f"Bad thought extraction: {t_txt!r}")
    if not p_txt.startswith("DECISION_USD: $450") or "<turn|>" in p_txt:
        errors.append(f"Bad public extraction: {p_txt!r}")

    # Verify unclosed thought detection for rescue path
    sample_open = "<|channel>thought\nLet me compute 18,000 - 5,000 = 13,000..."
    _, p_open, p_pres, p_closed = split_thought_and_public(sample_open)
    if not (p_pres and not p_closed and p_open == ""):
        errors.append("Failed to flag unclosed <|channel>thought block for rescue")

    # Verify open response lexical scrubbing audit
    open_suite = build_open_decision_suite()
    rec_mock = GeneratedThoughtRecord(
        prompt_token_ids=[2, 105],
        gen_token_ids=[100, 101],
        raw_with_special=sample_g4,
        thought_text=t_txt,
        public_text=p_txt,
        thought_present=True,
        thought_closed=True,
        thought_rescued=False,
        n_gen_tokens=30,
        n_thought_tokens=20,
        n_public_tokens=10,
    )
    parsed = parse_thought_open_response(rec_mock, open_suite["open_ultimatum_min"][1])
    if not (parsed["parsed_ok"] and parsed["ref_usd"] == 450.0 and parsed["scrubbed_strict"] is True):
        errors.append(f"Scrubbing parser mismatch: {parsed}")

    # Verify debt math thought arithmetic detector
    dm_item = build_debt_math_suite()[1]  # dm_01_flat_wins: 18,000 - 5,000 = 13,000 vs 14,500
    arith = audit_debt_thought_arithmetic(
        "Package 1 has $18,000 cash minus $5,000 loan = $13,000 net, whereas Package 2 has $14,500.",
        dm_item,
        is_correct=True,
    )
    if arith["outcome_class"] != "computed_net_and_correct":
        errors.append(f"Debt arithmetic classifier failed: {arith}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "n_exp_a_tasks": len(EXPERIMENT_A_PLAN),
        "n_exp_b_items": len(build_debt_math_suite()),
        "highlighter_concepts": list(HIGHLIGHTER_CONCEPTS),
        "strict_lexicon_size": len(STRICT_AFFECTIVE_LEXICON),
        "broad_lexicon_size": len(BROAD_MORAL_LEXICON),
    }


def _load_phase2_reference(model_id: str) -> dict[str, Any] | None:
    """Locate saved Phase 2 `valuation_results.json` locally if present for side-by-side comparison."""
    slug = "gemma-4-31b-it" if "31b" in model_id.lower() else model_id.split("/")[-1].lower()
    candidates = sorted((ROOT_DIR / "runs").glob(f"valuation_{slug}_*/valuation_results.json"))
    if candidates:
        try:
            return json.loads(candidates[-1].read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _summarize_open_thought_rows(
    rows: list[dict[str, Any]],
    base_vals: np.ndarray,
    alpha_val: float,
) -> dict[str, Any]:
    vals = np.array([r["ref_usd"] for r in rows], dtype=np.float64)
    diffs = vals - base_vals
    ci_v = bootstrap_mean_ci(vals)
    ci_d = bootstrap_mean_ci(diffs) if abs(alpha_val) > 1e-12 else (0.0, 0.0)
    p_d = exact_or_mc_signflip_p(diffs, two_sided=True) if abs(alpha_val) > 1e-12 else 1.0

    t_strict_has = [r["thought_lex"]["has_strict_affect"] for r in rows]
    p_strict_has = [r["public_lex"]["has_strict_affect"] for r in rows]
    t_broad_has = [r["thought_lex"]["has_broad_moral"] for r in rows]
    p_broad_has = [r["public_lex"]["has_broad_moral"] for r in rows]

    return {
        "alpha": float(alpha_val),
        "n_fixtures": len(rows),
        "mean_ref_usd": float(np.mean(vals)),
        "median_ref_usd": float(np.median(vals)),
        "ref_usd_ci95": [float(ci_v[0]), float(ci_v[1])],
        "delta_ref_usd": float(np.mean(diffs)),
        "delta_ci95": [float(ci_d[0]), float(ci_d[1])],
        "p_signflip": float(p_d),
        "parse_rate": float(np.mean([r["parsed_ok"] for r in rows])),
        "thought_closed_rate": float(np.mean([r["thought_closed"] for r in rows])),
        "thought_rescued_count": int(sum(1 for r in rows if r["thought_rescued"])),
        "mean_thought_tokens": float(np.mean([r["n_thought_tokens"] for r in rows])),
        "mean_public_tokens": float(np.mean([r["n_public_tokens"] for r in rows])),
        "mean_thought_words": float(np.mean([r["thought_lex"]["n_words"] for r in rows])),
        "mean_public_words": float(np.mean([r["public_lex"]["n_words"] for r in rows])),
        "thought_has_strict_affect_rate": float(np.mean(t_strict_has)),
        "public_has_strict_affect_rate": float(np.mean(p_strict_has)),
        "public_zero_strict_affect_rate": float(np.mean([not x for x in p_strict_has])),
        "strict_scrub_rate": float(np.mean([r["scrubbed_strict"] for r in rows])),
        "thought_strict_per_100w": float(np.mean([r["thought_lex"]["strict_per_100w"] for r in rows])),
        "public_strict_per_100w": float(np.mean([r["public_lex"]["strict_per_100w"] for r in rows])),
        "thought_has_broad_moral_rate": float(np.mean(t_broad_has)),
        "public_has_broad_moral_rate": float(np.mean(p_broad_has)),
        "broad_scrub_rate": float(np.mean([r["scrubbed_broad"] for r in rows])),
        "thought_broad_per_100w": float(np.mean([r["thought_lex"]["broad_per_100w"] for r in rows])),
        "public_broad_per_100w": float(np.mean([r["public_lex"]["broad_per_100w"] for r in rows])),
        "thought_jargon_per_100w": float(np.mean([r["thought_lex"]["jargon_per_100w"] for r in rows])),
        "public_jargon_per_100w": float(np.mean([r["public_lex"]["jargon_per_100w"] for r in rows])),
        "responses": rows,
    }


def run_thought_experiment(
    harness: TransformerHarness,
    locked: LockedBasis,
    out_dir: Path,
    batch_size: int = 16,
    gen_batch_size: int = 3,
    max_new_tokens: int = 768,
    skip_plots: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    audit = audit_thought_probe()
    if not audit["valid"]:
        raise RuntimeError(f"Thought probe audit failed: {audit['errors'][:5]}")

    pl = locked.primary_layer
    unit = locked.natural_contrast_unit
    print(
        f"[1/4] Audited `<thought>` probe stimuli & parsers. Running on {harness.model_id} "
        f"(Layer {pl}, natural_contrast_unit={unit:.4f}, gen_batch_size={gen_batch_size}, max_new_tokens={max_new_tokens}).",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Experiment A: Private `<thought>` vs Public `RATIONALE` under Steering
    # -----------------------------------------------------------------------
    print(f"[2/4] Running Experiment A (`enable_thinking=True` vs `False` open pricing under L{pl} steering)...", flush=True)
    open_suite = build_open_decision_suite()
    exp_a_results: dict[str, Any] = {}

    for tid, vec_names in EXPERIMENT_A_PLAN:
        oitems = open_suite[tid][:EXPERIMENT_A_N_FIXTURES]
        meta = OPEN_TASK_META[tid]

        prompts_think = [
            format_chat_prompt_thinking(harness.tokenizer, oit.user_prompt, enable_thinking=True)
            for oit in oitems
        ]
        prompts_nothink_prefill = [
            format_chat_prompt_thinking(
                harness.tokenizer,
                oit.user_prompt,
                enable_thinking=False,
                assistant_prefill="DECISION_USD: $",
            )
            for oit in oitems
        ]

        # Unsteered (alpha = 0.0) across thinking vs no-think prefill modes
        base_rec_t = generate_thought_records(harness, prompts_think, max_new_tokens=max_new_tokens, batch_size=gen_batch_size)
        base_rows_t = [parse_thought_open_response(r, oit) for r, oit in zip(base_rec_t, oitems)]
        base_vals_t = np.array([r["ref_usd"] for r in base_rows_t], dtype=np.float64)
        base_sum_t = _summarize_open_thought_rows(base_rows_t, base_vals_t, 0.0)

        base_rec_np = generate_thought_records(harness, prompts_nothink_prefill, max_new_tokens=96, batch_size=gen_batch_size, rescue_unclosed_thought=False)
        base_rows_np = [parse_thought_open_response(r, oit) for r, oit in zip(base_rec_np, oitems)]
        base_vals_np = np.array([r["ref_usd"] for r in base_rows_np], dtype=np.float64)
        base_sum_np = _summarize_open_thought_rows(base_rows_np, base_vals_np, 0.0)

        steered_thinking: dict[str, list[dict[str, Any]]] = {}
        steered_nothink_prefill: dict[str, list[dict[str, Any]]] = {}

        for vec_name in vec_names:
            vec = _resolve_vec(locked, vec_name)
            rows_t_list: list[dict[str, Any]] = []
            rows_np_list: list[dict[str, Any]] = []

            for alpha in THOUGHT_STEER_ALPHAS:
                if abs(alpha) < 1e-12:
                    rows_t_list.append(base_sum_t)
                    rows_np_list.append(base_sum_np)
                else:
                    scale = float(alpha * unit)
                    with harness.residual_steering(layer_idx=pl, direction=vec, delta_scale=scale):
                        s_rec_t = generate_thought_records(
                            harness,
                            prompts_think,
                            max_new_tokens=max_new_tokens,
                            batch_size=gen_batch_size,
                        )
                        s_rec_np = generate_thought_records(
                            harness,
                            prompts_nothink_prefill,
                            max_new_tokens=96,
                            batch_size=gen_batch_size,
                            rescue_unclosed_thought=False,
                        )
                    s_rows_t = [parse_thought_open_response(r, oit) for r, oit in zip(s_rec_t, oitems)]
                    s_rows_np = [parse_thought_open_response(r, oit) for r, oit in zip(s_rec_np, oitems)]

                    rows_t_list.append(_summarize_open_thought_rows(s_rows_t, base_vals_t, alpha))
                    rows_np_list.append(_summarize_open_thought_rows(s_rows_np, base_vals_np, alpha))

            steered_thinking[vec_name] = rows_t_list
            steered_nothink_prefill[vec_name] = rows_np_list
            print(f"    completed Experiment A {tid} :: {vec_name} ({round(time.time() - t0, 1)}s)", flush=True)

        exp_a_results[tid] = {
            "meta": meta,
            "thinking_steered": steered_thinking,
            "nothink_prefill_steered": steered_nothink_prefill,
        }
    _maybe_empty_cuda_cache(harness)

    # -----------------------------------------------------------------------
    # Experiment B: 0%-APR Loan Subtraction Blindspot (`False` vs `True`)
    # -----------------------------------------------------------------------
    print(f"[3/4] Running Experiment B (32-item 0%-APR Loan vs. Freight Deduction, elapsed={round(time.time() - t0, 1)}s)...", flush=True)
    math_items = build_debt_math_suite()

    # B1. Exact Phase 2 log-odds reproduction (`enable_thinking=False`)
    math_lo_scores, _ = harness.score_counterbalanced_choices(
        scenarios=[it.scenario for it in math_items],
        target_options=[it.better_option for it in math_items],
        reference_options=[it.worse_option for it in math_items],
        batch_size=batch_size,
    )

    # B2. Counterbalanced AB + BA generative runs (`enable_thinking=False` and `enable_thinking=True`)
    ab_user_prompts: list[str] = []
    for it in math_items:
        ab_user_prompts.append(build_choice_user_prompt(it.scenario, it.better_option, it.worse_option))
        ab_user_prompts.append(build_choice_user_prompt(it.scenario, it.worse_option, it.better_option))

    prompts_b_nothink = [format_chat_prompt_thinking(harness.tokenizer, up, enable_thinking=False) for up in ab_user_prompts]
    prompts_b_think = [format_chat_prompt_thinking(harness.tokenizer, up, enable_thinking=True) for up in ab_user_prompts]

    recs_b_nothink = generate_thought_records(
        harness,
        prompts_b_nothink,
        max_new_tokens=32,
        batch_size=batch_size,
        rescue_unclosed_thought=False,
    )
    recs_b_think = generate_thought_records(
        harness,
        prompts_b_think,
        max_new_tokens=max_new_tokens,
        batch_size=gen_batch_size,
    )

    exp_b_items_out: list[dict[str, Any]] = []
    for i, it in enumerate(math_items):
        sc = math_lo_scores[i]
        nt_ab, nt_ba = recs_b_nothink[2 * i], recs_b_nothink[2 * i + 1]
        th_ab, th_ba = recs_b_think[2 * i], recs_b_think[2 * i + 1]

        nt_ch_ab = parse_ab_choice_from_public(nt_ab.public_text)
        nt_ch_ba = parse_ab_choice_from_public(nt_ba.public_text)
        th_ch_ab = parse_ab_choice_from_public(th_ab.public_text)
        th_ch_ba = parse_ab_choice_from_public(th_ba.public_text)

        nt_ok_ab = (nt_ch_ab == "A")
        nt_ok_ba = (nt_ch_ba == "B")
        th_ok_ab = (th_ch_ab == "A")
        th_ok_ba = (th_ch_ba == "B")

        arith_ab = audit_debt_thought_arithmetic(th_ab.thought_text, it, is_correct=th_ok_ab)
        arith_ba = audit_debt_thought_arithmetic(th_ba.thought_text, it, is_correct=th_ok_ba)

        exp_b_items_out.append({
            "item_id": it.item_id,
            "domain": it.domain,
            "winner_arm": it.winner_arm,
            "gross_1": it.gross_1,
            "deduct_1": it.deduct_1,
            "net_1": it.net_1,
            "flat_2": it.flat_2,
            "scenario": it.scenario,
            "better_option": it.better_option,
            "worse_option": it.worse_option,
            "nothink_log_odds": float(sc.target_log_odds),
            "nothink_log_odds_correct": bool(sc.target_log_odds > 0.0),
            "nothink_gen_acc": 0.5 * (float(nt_ok_ab) + float(nt_ok_ba)),
            "nothink_ab_choice": nt_ch_ab,
            "nothink_ba_choice": nt_ch_ba,
            "think_gen_acc": 0.5 * (float(th_ok_ab) + float(th_ok_ba)),
            "think_both_orders_correct": bool(th_ok_ab and th_ok_ba),
            "think_ab": {
                "choice": th_ch_ab,
                "correct": th_ok_ab,
                "n_thought_tokens": th_ab.n_thought_tokens,
                "thought_closed": th_ab.thought_closed,
                "thought_rescued": th_ab.thought_rescued,
                "thought_text": th_ab.thought_text,
                "public_text": th_ab.public_text,
                "arithmetic": arith_ab,
            },
            "think_ba": {
                "choice": th_ch_ba,
                "correct": th_ok_ba,
                "n_thought_tokens": th_ba.n_thought_tokens,
                "thought_closed": th_ba.thought_closed,
                "thought_rescued": th_ba.thought_rescued,
                "thought_text": th_ba.thought_text,
                "public_text": th_ba.public_text,
                "arithmetic": arith_ba,
            },
        })

    def _summarize_exp_b_domain(dom: str) -> dict[str, Any]:
        sub = [r for r in exp_b_items_out if r["domain"] == dom]
        gw = [r for r in sub if r["winner_arm"] == "gross_minus_obligation_wins"]
        uw = [r for r in sub if r["winner_arm"] == "unencumbered_wins"]
        outcomes: dict[str, int] = {
            "computed_net_and_correct": 0,
            "computed_net_yet_wrong": 0,
            "omitted_net_and_correct": 0,
            "omitted_net_and_wrong": 0,
        }
        explicit_count = 0
        total_runs = 2 * len(sub)
        for r in sub:
            for ord_key in ("think_ab", "think_ba"):
                oc = r[ord_key]["arithmetic"]["outcome_class"]
                outcomes[oc] = outcomes.get(oc, 0) + 1
                if r[ord_key]["arithmetic"]["explicitly_computed_net1"]:
                    explicit_count += 1
        return {
            "n_items": len(sub),
            "nothink_logodds_overall_acc": float(np.mean([r["nothink_log_odds_correct"] for r in sub])),
            "nothink_logodds_debt_wins_acc": float(np.mean([r["nothink_log_odds_correct"] for r in gw])),
            "nothink_logodds_zero_debt_wins_acc": float(np.mean([r["nothink_log_odds_correct"] for r in uw])),
            "nothink_gen_overall_acc": float(np.mean([r["nothink_gen_acc"] for r in sub])),
            "nothink_gen_debt_wins_acc": float(np.mean([r["nothink_gen_acc"] for r in gw])),
            "nothink_gen_zero_debt_wins_acc": float(np.mean([r["nothink_gen_acc"] for r in uw])),
            "think_gen_overall_acc": float(np.mean([r["think_gen_acc"] for r in sub])),
            "think_gen_debt_wins_acc": float(np.mean([r["think_gen_acc"] for r in gw])),
            "think_gen_zero_debt_wins_acc": float(np.mean([r["think_gen_acc"] for r in uw])),
            "think_explicit_net1_rate": float(explicit_count / max(1, total_runs)),
            "mean_thought_tokens": float(
                np.mean([0.5 * (r["think_ab"]["n_thought_tokens"] + r["think_ba"]["n_thought_tokens"]) for r in sub])
            ),
            "outcome_counts_64": outcomes,
        }

    # Also run canonical $1,000 - $350 = $650 vs $600 / $700 illustrative pair
    canon_specs = [
        DebtMathItem(
            item_id="canon_1000_350_debt_wins",
            domain="debt_nw",
            winner_arm="gross_minus_obligation_wins",
            gross_1=1_000,
            deduct_1=350,
            net_1=650,
            flat_2=600,
            scenario=(
                "Choose the account package that yields the higher net dollar value at maturity "
                "after 0%-APR payables settle: Package 1 holds $1,000 cash and a $350 0%-APR "
                "loan payable; Package 2 holds $600 cash and $0 loans."
            ),
            better_option="Select Package 1 ($1,000 cash minus $350 0%-APR loan payable).",
            worse_option="Select Package 2 ($600 cash and $0 loans).",
        ),
        DebtMathItem(
            item_id="canon_1000_350_flat_wins",
            domain="debt_nw",
            winner_arm="unencumbered_wins",
            gross_1=1_000,
            deduct_1=350,
            net_1=650,
            flat_2=700,
            scenario=(
                "Choose the account package that yields the higher net dollar value at maturity "
                "after 0%-APR payables settle: Package 1 holds $1,000 cash and a $350 0%-APR "
                "loan payable; Package 2 holds $700 cash and $0 loans."
            ),
            better_option="Select Package 2 ($700 cash and $0 loans).",
            worse_option="Select Package 1 ($1,000 cash minus $350 0%-APR loan payable).",
        ),
    ]
    canon_prompts_nt = [
        format_chat_prompt_thinking(harness.tokenizer, build_choice_user_prompt(it.scenario, it.better_option, it.worse_option), enable_thinking=False)
        for it in canon_specs
    ]
    canon_prompts_th = [
        format_chat_prompt_thinking(harness.tokenizer, build_choice_user_prompt(it.scenario, it.better_option, it.worse_option), enable_thinking=True)
        for it in canon_specs
    ]
    canon_nt_recs = generate_thought_records(harness, canon_prompts_nt, max_new_tokens=32, batch_size=4, rescue_unclosed_thought=False)
    canon_th_recs = generate_thought_records(harness, canon_prompts_th, max_new_tokens=max_new_tokens, batch_size=4)
    canon_out = []
    for it, r_nt, r_th in zip(canon_specs, canon_nt_recs, canon_th_recs):
        ch_nt = parse_ab_choice_from_public(r_nt.public_text)
        ch_th = parse_ab_choice_from_public(r_th.public_text)
        canon_out.append({
            "item_id": it.item_id,
            "scenario": it.scenario,
            "nothink_choice": ch_nt,
            "nothink_correct": ch_nt == "A",
            "think_choice": ch_th,
            "think_correct": ch_th == "A",
            "thought_text": r_th.thought_text,
            "public_text": r_th.public_text,
        })

    exp_b_results = {
        "summary": {
            "debt_nw": _summarize_exp_b_domain("debt_nw"),
            "cashflow_twin": _summarize_exp_b_domain("cashflow_twin"),
        },
        "canonical_1000_350_check": canon_out,
        "items": exp_b_items_out,
    }
    _maybe_empty_cuda_cache(harness)

    # -----------------------------------------------------------------------
    # Experiment C: Token-by-Token Layer 35 Moral Highlighter
    # -----------------------------------------------------------------------
    print(
        f"[4/4] Calibrating neutral token baseline & running Experiment C (Token-by-Token L{pl} Moral Highlighter, elapsed={round(time.time() - t0, 1)}s)...",
        flush=True,
    )
    cal_stats = calibrate_neutral_token_baseline(harness, locked)

    ineq_human_98 = build_inequity_item("ineq_01", 1_000, "payout allocation", "human", 20)
    ineq_rng_98 = build_inequity_item("ineq_01", 1_000, "payout allocation", "random", 20)
    tnt_01 = open_suite["open_tainted_keep"][0]
    tnt_04 = open_suite["open_tainted_keep"][3]
    ult_02 = open_suite["open_ultimatum_min"][1]

    # Pick a `dm_XX_flat_wins` item where `nothink_log_odds_correct` was False if available, else `dm_01_flat_wins`
    failed_flat_ids = [
        r["item_id"]
        for r in exp_b_items_out
        if r["domain"] == "debt_nw" and r["winner_arm"] == "unencumbered_wins" and not r["nothink_log_odds_correct"]
    ]
    chosen_dm_tag = failed_flat_ids[0].replace("_flat_wins", "") if failed_flat_ids else "dm_01"
    dm_loan_item = next(it for it in math_items if it.item_id == f"{chosen_dm_tag}_flat_wins")
    dm_cf_item = next(it for it in math_items if it.item_id == f"{chosen_dm_tag}_cf_flat_wins")

    highlighter_specs: list[dict[str, Any]] = [
        {
            "example_id": "hl_01_human_98_2",
            "title": "98/2 Human Proposer Ultimatum ($980 vs. $20)",
            "focus_concept": "spite",
            "user_prompt": build_choice_user_prompt(
                ineq_human_98.scenario,
                ineq_human_98.deviation_option,
                ineq_human_98.rational_option,
            ),
            "steer_vec": None,
            "alpha": 0.0,
        },
        {
            "example_id": "hl_02_rng_98_2",
            "title": "98/2 Computer Randomizer Ultimatum ($980 vs. $20)",
            "focus_concept": "spite",
            "user_prompt": build_choice_user_prompt(
                ineq_rng_98.scenario,
                ineq_rng_98.deviation_option,
                ineq_rng_98.rational_option,
            ),
            "steer_vec": None,
            "alpha": 0.0,
        },
        {
            "example_id": "hl_03_tainted_clinic",
            "title": "Disputed Surplus: Retired Subscriber Billing ($10,000)",
            "focus_concept": "contamination",
            "user_prompt": tnt_01.user_prompt,
            "steer_vec": None,
            "alpha": 0.0,
        },
        {
            "example_id": "hl_04_tainted_orderlies",
            "title": "Disputed Surplus: Payroll Advances to Hospital Orderlies ($20,000)",
            "focus_concept": "predation",
            "user_prompt": tnt_04.user_prompt,
            "steer_vec": None,
            "alpha": 0.0,
        },
        {
            "example_id": "hl_05_zero_apr_loan",
            "title": f"0%-APR Loan Subtraction ({chosen_dm_tag}: ${dm_loan_item.gross_1:,} - ${dm_loan_item.deduct_1:,} vs. ${dm_loan_item.flat_2:,})",
            "focus_concept": "guilt",
            "user_prompt": build_choice_user_prompt(
                dm_loan_item.scenario,
                dm_loan_item.better_option,
                dm_loan_item.worse_option,
            ),
            "steer_vec": None,
            "alpha": 0.0,
        },
        {
            "example_id": "hl_06_freight_twin",
            "title": f"Commercial Freight Deduction Twin ({chosen_dm_tag}: ${dm_cf_item.gross_1:,} - ${dm_cf_item.deduct_1:,} vs. ${dm_cf_item.flat_2:,})",
            "focus_concept": "guilt",
            "user_prompt": build_choice_user_prompt(
                dm_cf_item.scenario,
                dm_cf_item.better_option,
                dm_cf_item.worse_option,
            ),
            "steer_vec": None,
            "alpha": 0.0,
        },
        {
            "example_id": "hl_07_ultimatum_spite_neg2",
            "title": "Open Ultimatum Floor ($1,000 Pool) Steered -2.0 Spite",
            "focus_concept": "spite",
            "user_prompt": ult_02.user_prompt,
            "steer_vec": "spite_controlled",
            "alpha": -2.0,
        },
        {
            "example_id": "hl_08_ultimatum_spite_zero",
            "title": "Open Ultimatum Floor ($1,000 Pool) Unsteered 0.0",
            "focus_concept": "spite",
            "user_prompt": ult_02.user_prompt,
            "steer_vec": None,
            "alpha": 0.0,
        },
        {
            "example_id": "hl_09_ultimatum_spite_pos2",
            "title": "Open Ultimatum Floor ($1,000 Pool) Steered +2.0 Spite",
            "focus_concept": "spite",
            "user_prompt": ult_02.user_prompt,
            "steer_vec": "spite_controlled",
            "alpha": 2.0,
        },
    ]

    highlighter_examples: list[dict[str, Any]] = []
    for spec in highlighter_specs:
        p_rendered = format_chat_prompt_thinking(harness.tokenizer, spec["user_prompt"], enable_thinking=True)
        s_vec_name = spec["steer_vec"]
        s_alpha = float(spec["alpha"])
        s_vec = _resolve_vec(locked, s_vec_name) if s_vec_name else None
        s_scale = float(s_alpha * unit) if s_vec is not None else 0.0

        if s_vec is not None and abs(s_scale) > 1e-12:
            with harness.residual_steering(layer_idx=pl, direction=s_vec, delta_scale=s_scale):
                gen_rec = generate_thought_records(harness, [p_rendered], max_new_tokens=max_new_tokens, batch_size=1)[0]
        else:
            gen_rec = generate_thought_records(harness, [p_rendered], max_new_tokens=max_new_tokens, batch_size=1)[0]

        proj = project_sequence_tokens(
            harness=harness,
            locked=locked,
            cal_stats=cal_stats,
            rec=gen_rec,
            steer_vec=s_vec,
            steer_delta_scale=s_scale,
        )
        highlighter_examples.append({
            "example_id": spec["example_id"],
            "title": spec["title"],
            "focus_concept": spec["focus_concept"],
            "steer_vec": s_vec_name,
            "alpha": s_alpha,
            "thought_text": gen_rec.thought_text,
            "public_text": gen_rec.public_text,
            "n_thought_tokens": gen_rec.n_thought_tokens,
            "n_public_tokens": gen_rec.n_public_tokens,
            "segment_summary": proj["segment_summary"],
            "tokens": proj["tokens"],
        })

    # Attach local Phase 2 reference summary if available
    p2_ref = _load_phase2_reference(harness.model_id)
    p2_open_ref: dict[str, Any] = {}
    if p2_ref and "open_negotiation" in p2_ref:
        for tid, _ in EXPERIMENT_A_PLAN:
            if tid in p2_ref["open_negotiation"]:
                blk = p2_ref["open_negotiation"][tid]
                p2_open_ref[tid] = {
                    vec_k: [
                        {
                            "alpha": r["alpha"],
                            "mean_ref_usd": r["mean_ref_usd"],
                            "zero_affect_rationale_rate": r["zero_affect_rationale_rate"],
                        }
                        for r in rows_v
                    ]
                    for vec_k, rows_v in blk["steered"].items()
                }

    results: dict[str, Any] = {
        "metadata": {
            "experiment": "gemma4_thought_pass_and_moral_highlighter",
            "model_id": harness.model_id,
            "n_layers": int(harness.n_layers),
            "primary_layer": int(pl),
            "natural_contrast_unit": float(unit),
            "stream_rms_at_primary": float(locked.stream_rms[pl]),
            "max_new_tokens": int(max_new_tokens),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "experiment_a_open_steering": exp_a_results,
        "phase2_open_reference": p2_open_ref,
        "experiment_b_debt_math": exp_b_results,
        "experiment_c_highlighter": {
            "concepts": list(HIGHLIGHTER_CONCEPTS),
            "neutral_calibration": cal_stats,
            "examples": highlighter_examples,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "thought_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if skip_plots:
        print(f"Saved `<thought>` probe JSON to {out_json} in {results['metadata']['elapsed_seconds']}s")
    else:
        from afriat.plot_thought import write_thought_reports

        md_p = write_thought_reports(results, out_dir)
        print(f"Saved `<thought>` report & figures to {md_p}")
    return results


def _run_synthetic_self_test() -> None:
    """Fast local end-to-end smoke test on synthetic harness + plot/report renderer."""
    import tempfile
    from afriat.run_discovery import run_pipeline

    # Import SyntheticPlantedHarness pattern cleanly without mutating test files
    rng = np.random.default_rng(19)
    n_layers, d_model = 6, 48
    all_c = CORE_CONCEPTS + CONTROL_CONCEPTS
    q, _ = np.linalg.qr(rng.normal(size=(d_model, len(all_c) + 2)))
    axes = {c: q[:, i] for i, c in enumerate(all_c)}

    class _MockHarness:
        model_id = "google/gemma-4-31B-it"
        tokenizer = None

        def __init__(self) -> None:
            self.n_layers = n_layers
            self.d_model = d_model
            self._steer: tuple[int, np.ndarray, float] | None = None

        from contextlib import contextmanager

        @contextmanager
        def residual_steering(self, layer_idx: int, direction: np.ndarray, delta_scale: float):
            prev = self._steer
            self._steer = (layer_idx, np.asarray(direction, dtype=np.float64), float(delta_scale))
            try:
                yield
            finally:
                self._steer = prev

        def _state(self, text: str) -> np.ndarray:
            r = np.random.default_rng(abs(hash(text)) % (2**31 - 1))
            st = r.normal(scale=0.2, size=(self.n_layers, self.d_model))
            if self._steer is not None:
                sl, sd, sc = self._steer
                for ell in range(sl, self.n_layers):
                    st[ell] += sc * sd
            return st

        def extract_last_token_states(self, prompts: list[str], batch_size: int = 8) -> np.ndarray:
            return np.stack([self._state(p) for p in prompts], axis=0)

        def score_counterbalanced_choices(
            self,
            scenarios: list[str],
            target_options: list[str],
            reference_options: list[str],
            batch_size: int = 8,
            capture_decision_states: bool = False,
        ):
            from afriat.models import ChoiceScore

            scores = [
                ChoiceScore(target_log_odds=1.5, target_prob=0.82, order_ab_log_odds=1.6, order_ba_log_odds=1.4, ab_mass=0.99)
                for _ in scenarios
            ]
            dec = np.stack([self._state(s) for s in scenarios], axis=0) if capture_decision_states else None
            return scores, dec

        def generate_texts(self, prompts: list[str], max_new_tokens: int = 84, batch_size: int = 8) -> list[str]:
            outs = []
            for p in prompts:
                st = self._state(p)
                sp = float(np.dot(st[-1], axes["spite"]))
                if "single letter A or B" in p:
                    if "<thought_mode_on>" in p:
                        outs.append("<|channel>thought\nSubtracting $18,000 - $5,000 = $13,000 vs $14,500.<channel|>A")
                    else:
                        outs.append("A")
                elif "<thought_mode_on>" in p:
                    amt = max(0, int(round(250 + 80 * sp)))
                    outs.append(
                        f"<|channel>thought\nThis lowball offer feels unfair and greedy, so I want $ {amt}.<channel|>"
                        f"DECISION_USD: ${amt}\nRATIONALE: Standard settlement liquidity margin governs this quote."
                    )
                else:
                    amt = max(0, int(round(250 + 80 * sp)))
                    outs.append(f"{amt}\nRATIONALE: Standard settlement liquidity margin governs this quote.")
            return outs

    with tempfile.TemporaryDirectory(prefix="afriat_thought_selftest_") as td:
        tmp = Path(td)
        mock = _MockHarness()
        p0_dir = tmp / "p0"
        run_pipeline(harness=mock, out_dir=p0_dir, batch_size=16, steering_alpha=1.0, skip_plots=True)
        locked = LockedBasis.from_phase0_files(p0_dir / "basis_vectors.npz", p0_dir / "results.json")
        out_d = tmp / "thought_out"
        res = run_thought_experiment(
            harness=mock,  # type: ignore[arg-type]
            locked=locked,
            out_dir=out_d,
            batch_size=16,
            gen_batch_size=8,
            max_new_tokens=128,
            skip_plots=False,
        )
        assert (out_d / "thought_results.json").exists()
        assert (out_d / "fig_thought_vs_public.png").exists()
        assert (out_d / "fig_token_highlighter.png").exists()
        assert (out_d / "thought_report.md").exists()
        report_txt = (out_d / "thought_report.md").read_text(encoding="utf-8")
        assert "\u2014" not in report_txt, "Em dash found in thought_report.md!"
        assert len(res["experiment_c_highlighter"]["examples"]) == 9
    print("Self-test passed: thought_probe + plot_thought verified end-to-end.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 31B `<thought>` Pass + Layer 35 Moral Highlighter")
    parser.add_argument("--model", type=str, default="gemma-4-31b-it")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--basis-npz", type=Path, default=None)
    parser.add_argument("--phase0-json", type=Path, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gen-batch-size", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.audit_only:
        rep = audit_thought_probe()
        if not rep["valid"]:
            for err in rep["errors"][:20]:
                print(f"[AUDIT ERROR] {err}")
            raise SystemExit(1)
        print(json.dumps(rep))
        return

    if args.self_test:
        _run_synthetic_self_test()
        return

    if args.render_dir is not None:
        from afriat.plot_thought import write_thought_reports

        res = json.loads((args.render_dir / "thought_results.json").read_text(encoding="utf-8"))
        md_p = write_thought_reports(res, args.render_dir)
        print(f"Rendered `<thought>` report & figures in {args.render_dir}: {md_p.name}")
        return

    if args.basis_npz is None or args.phase0_json is None:
        raise SystemExit("--basis-npz and --phase0-json are required when running on a model")

    locked = LockedBasis.from_phase0_files(args.basis_npz, args.phase0_json)
    label = args.model_label or resolve_model_name(args.model)
    out_dir = args.out_dir or (ROOT_DIR / "runs" / "thought_run")
    harness = TransformerHarness.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        model_id_label=label,
    )
    run_thought_experiment(
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
