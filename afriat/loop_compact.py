"""Phase 9: `Gemma-4-Loop-Compact` (Actions 1-4: Hourglass Compression, Native Looping, CoT-to-Latent Distillation, Adaptive Halting & Bridge Safety).

Executes the 4-part engineering plan on `google/gemma-4-31B-it`:
  - Action 1: Hourglass Structured MLP Width Pruning + Packed Groupwise `INT4`/`INT8` Quantization (`62.6 GB -> ~17 GB`),
              benchmarked head-to-head against a byte-matched `Uniform-Compact` control.
  - Action 2: Standalone `Gemma4LoopCompactModel` architecture (`Prelude -> Recurrent Funnel L24..L42 -> Coda`)
              with closed-form SVD Orthogonal Procrustes re-entry, learned loop-step embeddings `e_k`,
              and Loop-Indexed LoRA (`r = 16`, `k = 1..4`), exported as a reloadable HF artifact package.
  - Action 3: Randomized Unroll Depth (`K ~ Uniform{1, 2, 3, 4}`) + Teacher-Forced `<thought>`-to-Latent
              Self-Distillation (`BF16` `<thought>` soft answer logits + `L42` latent alignment + general-text KL anchor).
  - Action 4: `L42` Adaptive Halting Gate (`mode="adaptive"` on easy vs. hard workloads) + Zero-Cost
              Moral-Subspace Safety Projection folded directly into the Procrustes bridge (`R_safe = R_proc (I - Q Q^T)`).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any, Iterator, Sequence

import numpy as np

from .geometry import LayerBasis, extract_concept_basis
from .inequity import INEQUITY_FIXTURES, build_inequity_item
from .models import (
    TransformerHarness,
    build_choice_user_prompt,
    build_scenario_reading_prompt,
    format_chat_prompt,
    resolve_decoder_layers,
    resolve_model_name,
)
from .optimize import (
    HELD_OUT_KL_PASSAGES,
    TRAIN_ANCHOR_PASSAGES as DISTILL_ANCHOR_PASSAGES,
    evaluate_kl_and_ppl as evaluate_kl_against_base,
    precompute_base_kl_references,
    solve_orthogonal_procrustes,
)
from .schema import load_concepts, load_sanity_items, load_transactions
from .thought_probe import (
    RESIDUAL_SPECIAL_RE,
    format_chat_prompt_thinking,
    resolve_thought_stop_ids,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STIMULI_DIR = REPO_ROOT / "stimuli"
RE_FIRST_INT = re.compile(r"[-+]?\d+")


# ---------------------------------------------------------------------------
# 1. Disjoint Multi-Suite Reasoning & `<thought>` Teacher Corpus (`0` Overlap)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReasoningItem:
    item_id: str
    suite: str           # "multihop_train", "multihop_eval", "wordprob_train", "wordprob_eval", "balsheet_train", "balsheet_eval", "easy_eval"
    hops: int
    prompt_user: str
    thought_trace: str   # Exact programmatic step-by-step `<thought>` derivation for Stage 0 teacher forcing
    answer_int: int


TRAIN_VARS = (("x", "y", "z"), ("p", "q", "r"), ("m", "n", "k"), ("u", "v", "w"))
EVAL_VARS = (("a", "b", "c"), ("d", "e", "f"), ("g", "h", "j"), ("r", "s", "t"))


def _generate_multihop_item(
    rng: random.Random,
    item_id: str,
    suite: str,
    hops: int,
    var_triplets: Sequence[tuple[str, str, str]],
) -> ReasoningItem:
    v0, v1, v2 = rng.choice(var_triplets)
    vals = {v0: rng.randint(5, 24), v1: rng.randint(4, 19), v2: rng.randint(3, 16)}
    clauses = [f"Start with {v0} = {vals[v0]}, {v1} = {vals[v1]}, and {v2} = {vals[v2]}."]
    thought_steps = [f"Initial state: {v0}={vals[v0]}, {v1}={vals[v1]}, {v2}={vals[v2]}."]
    names = [v0, v1, v2]

    for step_idx in range(1, hops + 1):
        dst = names[(step_idx - 1) % 3]
        src = names[step_idx % 3]
        op = rng.choice(("add_var", "sub_var", "double_minus", "add_const_plus_var"))
        c_small = rng.randint(2, 14)

        if op == "add_var" and abs(vals[dst] + vals[src]) <= 180:
            old_d, s_v = vals[dst], vals[src]
            vals[dst] = old_d + s_v
            clauses.append(f"Step {step_idx}: set {dst} to {dst} + {src}.")
            thought_steps.append(f"Step {step_idx}: {dst} = {old_d} + {s_v} = {vals[dst]}.")
        elif op == "sub_var":
            old_d, s_v = vals[dst], vals[src]
            vals[dst] = old_d - s_v + c_small
            clauses.append(f"Step {step_idx}: set {dst} to {dst} - {src} + {c_small}.")
            thought_steps.append(f"Step {step_idx}: {dst} = {old_d} - {s_v} + {c_small} = {vals[dst]}.")
        elif op == "double_minus" and abs(vals[src]) <= 55:
            s_v = vals[src]
            vals[dst] = 2 * s_v - c_small
            clauses.append(f"Step {step_idx}: set {dst} to 2 * {src} - {c_small}.")
            thought_steps.append(f"Step {step_idx}: {dst} = 2 * {s_v} - {c_small} = {vals[dst]}.")
        else:
            s_v = vals[src]
            vals[dst] = s_v + c_small
            clauses.append(f"Step {step_idx}: set {dst} to {src} + {c_small}.")
            thought_steps.append(f"Step {step_idx}: {dst} = {s_v} + {c_small} = {vals[dst]}.")

    ask_var = names[(hops - 1) % 3]
    ans = int(vals[ask_var])
    thought_steps.append(f"Required register {ask_var} = {ans}.")
    prompt = (
        "Track the integer registers through each step in exact order:\n"
        + "\n".join(clauses)
        + f"\nWhat is the final integer value of {ask_var}? "
        "Respond strictly in the format FINAL_INTEGER: <number> with no other text."
    )
    return ReasoningItem(
        item_id=item_id,
        suite=suite,
        hops=hops,
        prompt_user=prompt,
        thought_trace=" ".join(thought_steps),
        answer_int=ans,
    )


TRAIN_WP_ENTITIES = (
    ("Northwind Depot", "crates"),
    ("Kestrel Press", "folios"),
    ("Bramble Kitchen", "jars"),
    ("Solstice Lab", "vials"),
    ("Harbor Foundry", "ingots"),
    ("Lumen Studio", "panels"),
)

EVAL_WP_ENTITIES = (
    ("Aurelia Archive", "ledgers"),
    ("Cobalt Orchard", "baskets"),
    ("Timberline Mill", "planks"),
    ("Veridian Clinic", "kits"),
    ("Monarch Glassworks", "lenses"),
    ("Cinder Roastery", "sacks"),
)


def _generate_wordprob_item(
    rng: random.Random,
    item_id: str,
    suite: str,
    entities: Sequence[tuple[str, str]],
) -> ReasoningItem:
    place, unit = entities[rng.randrange(len(entities))]
    a0 = rng.randint(18, 64)
    b_batches = rng.randint(2, 5)
    b_each = rng.randint(6, 16)
    c_used = rng.randint(7, 25)
    d_ret = rng.randint(4, 18)
    e_split = rng.choice((2, 3))

    s1 = a0 + b_batches * b_each
    s2 = s1 - c_used + d_ret
    rem = s2 % e_split
    if rem != 0:
        d_ret += (e_split - rem)
        s2 = s1 - c_used + d_ret
    ans = s2 // e_split

    prompt = (
        f"{place} starts Monday with {a0} {unit}. "
        f"It receives {b_batches} shipments of {b_each} {unit} each, "
        f"dispatches {c_used} {unit} on Tuesday, and logs {d_ret} returned {unit} on Wednesday. "
        f"The remaining {unit} are divided equally across {e_split} rooms. "
        f"How many {unit} are placed in each room? "
        "Respond strictly in the format FINAL_INTEGER: <number> with no other text."
    )
    thought = (
        f"Start with {a0}. Add {b_batches} * {b_each} = {b_batches * b_each} to get {s1}. "
        f"Subtract {c_used} and add {d_ret}: {s1} - {c_used} + {d_ret} = {s2}. "
        f"Divide {s2} equally by {e_split}: {s2} / {e_split} = {ans}."
    )
    return ReasoningItem(
        item_id=item_id,
        suite=suite,
        hops=4,
        prompt_user=prompt,
        thought_trace=thought,
        answer_int=int(ans),
    )


TRAIN_BS_FIRMS = ("Alder Corp", "Beacon Logistics", "Crestline Tools", "Doric Systems", "Elmwood Freight", "Fable Media")
EVAL_BS_FIRMS = ("Garnet Bio", "Halcyon Marine", "Ibis Robotics", "Juniper Energy", "Keystone Optics", "Larkspur Foods")


def _generate_balsheet_item(
    rng: random.Random,
    item_id: str,
    suite: str,
    firms: Sequence[str],
) -> ReasoningItem:
    """Corporate net-worth difference with Phase 3's 0%-APR principal loan payable subtraction trap."""
    firm = firms[rng.randrange(len(firms))]
    cash_a = rng.randint(45, 98)
    loan_a = rng.randint(14, 42)  # 0%-APR principal payable that MUST be subtracted!
    cash_b = rng.randint(32, 85)
    loan_b = rng.choice((0, rng.randint(8, 28)))

    net_a = cash_a - loan_a
    net_b = cash_b - loan_b
    diff = net_a - net_b

    loan_b_clause = (
        f"and carries a ${loan_b}k zero-interest principal loan payable"
        if loan_b > 0
        else "and carries $0 in loan payables"
    )
    prompt = (
        f"{firm} is comparing two treasury accounts (in thousands of dollars). "
        f"Account Alpha holds ${cash_a}k cash on hand and carries a ${loan_a}k zero-interest principal loan payable due at maturity. "
        f"Account Beta holds ${cash_b}k cash on hand {loan_b_clause}. "
        "What is Account Alpha's net worth minus Account Beta's net worth (in thousands, as a signed integer)? "
        "Respond strictly in the format FINAL_INTEGER: <number> with no other text."
    )
    thought = (
        f"Account Alpha net = {cash_a} - {loan_a} = {net_a}. "
        f"Account Beta net = {cash_b} - {loan_b} = {net_b}. "
        f"Alpha minus Beta = {net_a} - ({net_b}) = {diff}."
    )
    return ReasoningItem(
        item_id=item_id,
        suite=suite,
        hops=3,
        prompt_user=prompt,
        thought_trace=thought,
        answer_int=int(diff),
    )


def _generate_easy_1hop_item(rng: random.Random, item_id: str, odd_first: bool = True) -> ReasoningItem:
    a = 2 * rng.randint(4, 24) + (1 if odd_first else 0)
    b = rng.randint(3, 25)
    if rng.random() < 0.5:
        ans = a + b
        prompt = f"Compute {a} + {b}. Respond strictly in the format FINAL_INTEGER: <number> with no other text."
        thought = f"{a} + {b} = {ans}."
    else:
        ans = a - b
        prompt = f"Compute {a} - {b}. Respond strictly in the format FINAL_INTEGER: <number> with no other text."
        thought = f"{a} - {b} = {ans}."
    return ReasoningItem(
        item_id=item_id,
        suite="easy_eval",
        hops=1,
        prompt_user=prompt,
        thought_trace=thought,
        answer_int=int(ans),
    )


def build_loop_compact_datasets() -> dict[str, list[ReasoningItem]]:
    """Construct disjoint training (`800` items) and held-out evaluation (`256` items) suites."""
    rng_tr = random.Random(900101)
    rng_ev = random.Random(900202)

    train_items: list[ReasoningItem] = []
    # 480 multi-hop chains across 2, 3, 4, 5, 6 hops (96 per hop level; 7 & 9 hops strictly unseen!)
    for h_val in (2, 3, 4, 5, 6):
        for i in range(96):
            train_items.append(
                _generate_multihop_item(rng_tr, f"tr_mh_h{h_val}_{i:03d}", "multihop_train", h_val, TRAIN_VARS)
            )
    # 160 word problems
    for i in range(160):
        train_items.append(_generate_wordprob_item(rng_tr, f"tr_wp_{i:03d}", "wordprob_train", TRAIN_WP_ENTITIES))
    # 160 balance-sheet net-worth comparisons
    for i in range(160):
        train_items.append(_generate_balsheet_item(rng_tr, f"tr_bs_{i:03d}", "balsheet_train", TRAIN_BS_FIRMS))
    # 64 1-hop arithmetic items (even `a`) so K=1 stays sharp on fast single-pass exits with 0 eval overlap
    for i in range(64):
        it_e = _generate_easy_1hop_item(rng_tr, f"tr_easy_{i:03d}", odd_first=False)
        train_items.append(
            ReasoningItem(
                item_id=it_e.item_id,
                suite="easy_train",
                hops=1,
                prompt_user=it_e.prompt_user,
                thought_trace=it_e.thought_trace,
                answer_int=it_e.answer_int,
            )
        )

    eval_multihop: list[ReasoningItem] = []
    for h_val in (3, 5, 7, 9):
        for i in range(32):
            eval_multihop.append(
                _generate_multihop_item(rng_ev, f"ev_mh_h{h_val}_{i:03d}", "multihop_eval", h_val, EVAL_VARS)
            )

    eval_wordprob = [
        _generate_wordprob_item(rng_ev, f"ev_wp_{i:03d}", "wordprob_eval", EVAL_WP_ENTITIES)
        for i in range(48)
    ]
    eval_balsheet = [
        _generate_balsheet_item(rng_ev, f"ev_bs_{i:03d}", "balsheet_eval", EVAL_BS_FIRMS)
        for i in range(48)
    ]
    eval_easy = [_generate_easy_1hop_item(rng_ev, f"ev_easy_{i:03d}") for i in range(32)]

    return {
        "train_all": train_items,
        "eval_multihop": eval_multihop,
        "eval_wordprob": eval_wordprob,
        "eval_balsheet": eval_balsheet,
        "eval_easy": eval_easy,
    }


def audit_loop_compact_suite() -> dict[str, Any]:
    ds = build_loop_compact_datasets()
    tr_hashes = {hashlib.sha256(x.prompt_user.encode()).hexdigest() for x in ds["train_all"]}
    ev_all = ds["eval_multihop"] + ds["eval_wordprob"] + ds["eval_balsheet"] + ds["eval_easy"]
    ev_hashes = {hashlib.sha256(x.prompt_user.encode()).hexdigest() for x in ev_all}
    overlap = len(tr_hashes & ev_hashes)
    kl_overlap = len(set(DISTILL_ANCHOR_PASSAGES) & set(HELD_OUT_KL_PASSAGES))
    return {
        "valid": overlap == 0 and kl_overlap == 0,
        "n_train": len(ds["train_all"]),
        "n_eval_multihop": len(ds["eval_multihop"]),
        "n_eval_wordprob": len(ds["eval_wordprob"]),
        "n_eval_balsheet": len(ds["eval_balsheet"]),
        "n_eval_easy": len(ds["eval_easy"]),
        "train_eval_overlap": overlap,
        "kl_passage_overlap": kl_overlap,
    }


# ---------------------------------------------------------------------------
# 2. Action 1: Packed `INT4`/`INT8` `CompactLinear` + Structured Width Pruning
# ---------------------------------------------------------------------------

def _choose_group_size(in_features: int, preferred: int = 64) -> int:
    for g in (preferred, 128, 32, 16, 8, 4, 2, 1):
        if in_features % g == 0 and (g % 2 == 0 or g == 1):
            return g
    return 1


def pack_weight_tensor(torch: Any, weight: Any, bits: int = 4, preferred_group: int = 64) -> dict[str, Any]:
    """Pack a `(out_features, in_features)` weight tensor into true `uint8` (`2x uint4`) or symmetric `int8`."""
    w = weight.detach().to(torch.float32)
    out_f, in_f = int(w.shape[0]), int(w.shape[1])

    if bits == 4 and in_f % 2 == 0:
        g = _choose_group_size(in_f, preferred_group)
        if g < 2:
            bits = 8
        else:
            n_g = in_f // g
            wg = w.reshape(out_f, n_g, g)
            w_min = wg.amin(dim=-1, keepdim=True)
            w_max = wg.amax(dim=-1, keepdim=True)
            scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
            q = torch.round((wg - w_min) / scale).clamp(0.0, 15.0).to(torch.uint8)
            q_packed = ((q[..., 0::2] << 4) | q[..., 1::2]).reshape(out_f, in_f // 2).contiguous()
            return {
                "bits": 4,
                "group_size": g,
                "out_features": out_f,
                "in_features": in_f,
                "qweight": q_packed,
                "scales": scale.squeeze(-1).to(torch.bfloat16).contiguous(),
                "zeros": w_min.squeeze(-1).to(torch.bfloat16).contiguous(),
            }

    # Symmetric groupwise INT8 (`[-127, 127]`)
    g = _choose_group_size(in_f, preferred_group)
    n_g = in_f // g
    wg = w.reshape(out_f, n_g, g)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q8 = torch.round(wg / scale).clamp(-127.0, 127.0).to(torch.int8).reshape(out_f, in_f).contiguous()
    return {
        "bits": 8,
        "group_size": g,
        "out_features": out_f,
        "in_features": in_f,
        "qweight": q8,
        "scales": scale.squeeze(-1).to(torch.bfloat16).contiguous(),
        "zeros": None,
    }


def dequantize_packed_weight(
    torch: Any,
    qweight: Any,
    scales: Any,
    zeros: Any | None,
    bits: int,
    out_features: int,
    in_features: int,
    group_size: int,
    target_dtype: Any,
) -> Any:
    """Fast vectorized unpack from `uint8`/`int8` to `target_dtype` (`bfloat16`/`float32`)."""
    n_g = in_features // group_size
    sc = scales.to(target_dtype).unsqueeze(-1)
    if bits == 4:
        hi = (qweight >> 4).to(target_dtype)
        lo = (qweight & 0x0F).to(target_dtype)
        qu = torch.stack((hi, lo), dim=-1).reshape(out_features, n_g, group_size)
        zr = zeros.to(target_dtype).unsqueeze(-1)
        return (qu * sc + zr).reshape(out_features, in_features)
    qu8 = qweight.to(target_dtype).reshape(out_features, n_g, group_size)
    return (qu8 * sc).reshape(out_features, in_features)


def build_compact_linear_class(torch: Any) -> Any:
    """Construct `CompactLinear(nn.Module)` with zero-retention autograd + optional `LoopIndexedLoRA`."""
    nn = torch.nn
    F = torch.nn.functional

    class _CompactLinearFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, qweight: Any, scales: Any, zeros: Any, bias: Any, bits: int, out_f: int, in_f: int, g_sz: int) -> Any:
            if zeros is not None:
                ctx.save_for_backward(qweight, scales, zeros)
                ctx.has_zeros = True
            else:
                ctx.save_for_backward(qweight, scales)
                ctx.has_zeros = False
            ctx.bits = bits
            ctx.out_f = out_f
            ctx.in_f = in_f
            ctx.g_sz = g_sz
            ctx.has_bias = bias is not None
            w_deq = dequantize_packed_weight(torch, qweight, scales, zeros, bits, out_f, in_f, g_sz, x.dtype)
            b = bias.to(x.dtype) if bias is not None else None
            return F.linear(x, w_deq, b)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> tuple[Any, ...]:
            if ctx.has_zeros:
                qweight, scales, zeros = ctx.saved_tensors
            else:
                qweight, scales = ctx.saved_tensors
                zeros = None
            w_deq = dequantize_packed_weight(
                torch, qweight, scales, zeros, ctx.bits, ctx.out_f, ctx.in_f, ctx.g_sz, grad_out.dtype
            )
            grad_x = torch.matmul(grad_out, w_deq)
            grad_b = None
            if ctx.has_bias:
                reduce_dims = tuple(range(grad_out.ndim - 1))
                grad_b = grad_out.sum(dim=reduce_dims)
            return grad_x, None, None, None, grad_b, None, None, None, None

    class CompactLinear(nn.Module):
        def __init__(
            self,
            packed: dict[str, Any],
            bias: Any | None = None,
            max_loops: int = 0,
            lora_rank: int = 0,
            lora_alpha: float = 32.0,
            loop_ctx: dict[str, int] | None = None,
            device: Any = None,
        ) -> None:
            super().__init__()
            self.bits = int(packed["bits"])
            self.group_size = int(packed["group_size"])
            self.out_features = int(packed["out_features"])
            self.in_features = int(packed["in_features"])
            self.register_buffer("qweight", packed["qweight"])
            self.register_buffer("scales", packed["scales"])
            if packed["zeros"] is not None:
                self.register_buffer("zeros", packed["zeros"])
            else:
                self.zeros = None
            if bias is not None:
                self.register_buffer("bias_buf", bias.detach().to(torch.bfloat16).contiguous())
            else:
                self.bias_buf = None

            self.max_loops = int(max_loops)
            self.lora_rank = int(lora_rank)
            self.lora_scaling = float(lora_alpha) / max(1, self.lora_rank)
            self.loop_ctx = loop_ctx

            if self.max_loops > 0 and self.lora_rank > 0:
                self.init_loop_lora(max_loops=self.max_loops, lora_rank=self.lora_rank, lora_alpha=lora_alpha)
            else:
                self.lora_A = None
                self.lora_B = None

            if device is not None:
                self.to(device)

        def init_loop_lora(self, max_loops: int, lora_rank: int, lora_alpha: float = 16.0) -> None:
            self.max_loops = int(max_loops)
            self.lora_rank = int(lora_rank)
            self.lora_scaling = float(lora_alpha) / max(1, self.lora_rank)
            dev = self.qweight.device
            a_list = []
            b_list = []
            for _k in range(self.max_loops):
                # Keep master `float32` weights during AdamW training; cast to `bfloat16` after distillation!
                p_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=torch.float32, device=dev))
                nn.init.kaiming_uniform_(p_a, a=math.sqrt(5))
                p_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=torch.float32, device=dev))
                a_list.append(p_a)
                b_list.append(p_b)
            self.lora_A = nn.ParameterList(a_list)
            self.lora_B = nn.ParameterList(b_list)

        def cast_lora_dtype(self, dtype: Any) -> None:
            if self.lora_A is not None and self.lora_B is not None:
                for k_i in range(len(self.lora_A)):
                    self.lora_A[k_i] = nn.Parameter(self.lora_A[k_i].detach().to(dtype), requires_grad=False)
                    self.lora_B[k_i] = nn.Parameter(self.lora_B[k_i].detach().to(dtype), requires_grad=False)

        def forward(self, x: Any) -> Any:
            base_out = _CompactLinearFn.apply(
                x,
                self.qweight,
                self.scales,
                self.zeros,
                self.bias_buf,
                self.bits,
                self.out_features,
                self.in_features,
                self.group_size,
            )
            if self.lora_A is not None and self.lora_B is not None:
                k_idx = 0
                if self.loop_ctx is not None:
                    k_idx = min(int(self.loop_ctx.get("loop_idx", 0)), self.max_loops - 1)
                a_k = self.lora_A[k_idx].to(x.dtype)
                b_k = self.lora_B[k_idx].to(x.dtype)
                lora_delta = torch.matmul(torch.matmul(x, a_k.t()), b_k.t()) * self.lora_scaling
                return base_out + lora_delta
            return base_out

    return CompactLinear


def compute_model_storage_bytes(model: Any) -> dict[str, Any]:
    """Sum unique tensor storage pointers across all parameters and buffers (avoiding double-counting tied weights)."""
    seen_ptrs: set[int] = set()
    param_bytes = 0
    buffer_bytes = 0
    for p in model.parameters():
        ptr = int(p.data_ptr())
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            param_bytes += int(p.numel() * p.element_size())
    for b in model.buffers():
        ptr = int(b.data_ptr())
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            buffer_bytes += int(b.numel() * b.element_size())
    tot = param_bytes + buffer_bytes
    return {
        "total_bytes": tot,
        "total_gb": round(tot / 1e9, 3),
        "total_gib": round(tot / (1024**3), 3),
        "param_gb": round(param_bytes / 1e9, 3),
        "buffer_gb": round(buffer_bytes / 1e9, 3),
    }


def score_mlp_neuron_importance(
    harness: TransformerHarness,
    calibration_prompts: Sequence[str],
    batch_size: int = 4,
) -> dict[int, np.ndarray]:
    """Wanda-style neuron importance `E[|post_geglu_act|] * ||W_down[:, j]||_2` across all decoder layers."""
    torch = harness.torch
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))
    n_layers = harness.n_layers

    act_sums: dict[int, Any] = {}
    tok_counts = 0

    handles = []
    for ell in range(n_layers):
        mlp = harness.layers[ell].mlp

        def _mk_hook(idx: int):
            def _pre_down(_m: Any, args: Any):
                a = args[0].detach().to(torch.float32)  # [B, T, d_ff]
                s = a.abs().sum(dim=(0, 1))
                if idx not in act_sums:
                    act_sums[idx] = s
                else:
                    act_sums[idx] = act_sums[idx] + s
            return _pre_down

        handles.append(mlp.down_proj.register_forward_pre_hook(_mk_hook(ell)))

    try:
        for i in range(0, len(calibration_prompts), batch_size):
            chunk = list(calibration_prompts[i : i + batch_size])
            enc = tok(
                chunk,
                add_special_tokens=not has_chat_tmpl,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=192,
            )
            inp = enc["input_ids"].to(harness.input_device)
            msk = enc["attention_mask"].to(harness.input_device)
            with torch.no_grad():
                harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
            tok_counts += int(inp.numel())
    finally:
        for h in handles:
            h.remove()

    importance: dict[int, np.ndarray] = {}
    denom = float(max(1, tok_counts))
    for ell in range(n_layers):
        w_down = harness.layers[ell].mlp.down_proj.weight.detach().to(torch.float32)  # [D, d_ff]
        col_norm = torch.linalg.vector_norm(w_down, ord=2, dim=0)
        mean_act = act_sums[ell] / denom
        imp = (mean_act * col_norm).cpu().numpy().astype(np.float32)
        importance[ell] = imp
    return importance


def build_compression_plan(
    n_layers: int,
    d_ff: int,
    funnel_start: int,
    funnel_end: int,
    mode: str,
) -> dict[int, dict[str, int]]:
    """Construct exact byte-matched `hourglass` vs. `uniform` per-layer `(d_ff_keep, bits)` schedules."""
    protected = {0, 1, n_layers - 2, n_layers - 1} | set(range(funnel_start, funnel_end + 1))
    n_prot = len(protected)
    n_outer = n_layers - n_prot

    # Hourglass: 100% width + INT8 on `n_prot` protected layers; 75% width (multiple of 64) + INT4 on `n_outer` corridor layers
    outer_keep = max(64, (int(round(d_ff * 0.75)) // 64) * 64)
    total_removed = n_outer * (d_ff - outer_keep)

    if mode == "hourglass":
        plan: dict[int, dict[str, int]] = {}
        for ell in range(n_layers):
            if ell in protected:
                plan[ell] = {"d_ff_keep": d_ff, "bits": 8}
            else:
                plan[ell] = {"d_ff_keep": outer_keep, "bits": 4}
        return plan

    if mode == "uniform":
        # Match total removed neurons AND exact count of INT8 vs INT4 layers, spread flat across depth!
        per_layer_remove = int(round(total_removed / max(1, n_layers)))
        uni_keep = max(64, ((d_ff - per_layer_remove) // 64) * 64)
        int8_slots = set(int(i) for i in np.linspace(0, n_layers - 1, n_prot, dtype=int))
        while len(int8_slots) < n_prot:
            for c in range(n_layers):
                if c not in int8_slots:
                    int8_slots.add(c)
                    if len(int8_slots) == n_prot:
                        break
        plan = {}
        for ell in range(n_layers):
            plan[ell] = {
                "d_ff_keep": uni_keep,
                "bits": 8 if ell in int8_slots else 4,
            }
        return plan

    raise ValueError(f"Unknown compression mode: {mode}")


def strip_unused_multimodal_towers_inplace(model: Any) -> list[str]:
    """Drop unused vision/audio encoder towers in text-compact mode so VRAM reflects text LM footprint cleanly."""
    removed: list[str] = []
    targets = ("vision_tower", "multi_modal_projector", "audio_tower", "audio_projector")
    for holder in (model, getattr(model, "model", None)):
        if holder is None:
            continue
        for attr in targets:
            if hasattr(holder, attr) and getattr(holder, attr) is not None:
                setattr(holder, attr, None)
                removed.append(f"{type(holder).__name__}.{attr}")
    return removed


def compress_decoder_layers_inplace(
    harness: TransformerHarness,
    importance: dict[int, np.ndarray],
    plan: dict[int, dict[str, int]],
    funnel_start: int,
    funnel_end: int,
    attach_loop_lora: bool = False,
    max_loops: int = 4,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    loop_ctx: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Physically slice pruned MLP neurons & replace `nn.Linear` in all decoder blocks with `CompactLinear`."""
    torch = harness.torch
    nn = torch.nn
    CompactLinear = build_compact_linear_class(torch)
    removed_towers = strip_unused_multimodal_towers_inplace(harness.model)

    pruned_indices: dict[str, np.ndarray] = {}
    n_replaced = 0

    for ell in range(harness.n_layers):
        blk = harness.layers[ell]
        spec = plan[ell]
        keep_k = int(spec["d_ff_keep"])
        bits = int(spec["bits"])
        in_funnel = funnel_start <= ell <= funnel_end
        use_lora = bool(attach_loop_lora and in_funnel)

        # 1. Structured MLP neuron width pruning on (gate_proj, up_proj, down_proj)
        mlp = blk.mlp
        orig_dff = int(mlp.down_proj.weight.shape[1])
        if keep_k < orig_dff:
            imp_vec = importance[ell]
            top_idx = np.sort(np.argpartition(imp_vec, -keep_k)[-keep_k:]).astype(np.int64)
            pruned_indices[f"layer_{ell}"] = top_idx.astype(np.int32)
            idx_t = torch.tensor(top_idx, dtype=torch.long, device=mlp.down_proj.weight.device)

            for proj_name in ("gate_proj", "up_proj"):
                if hasattr(mlp, proj_name):
                    lin = getattr(mlp, proj_name)
                    w_sub = lin.weight.detach().index_select(0, idx_t)
                    b_sub = lin.bias.detach().index_select(0, idx_t) if lin.bias is not None else None
                    pk = pack_weight_tensor(torch, w_sub, bits=bits)
                    setattr(
                        mlp,
                        proj_name,
                        CompactLinear(
                            pk,
                            bias=b_sub,
                            max_loops=max_loops if use_lora else 0,
                            lora_rank=lora_rank if use_lora else 0,
                            lora_alpha=lora_alpha,
                            loop_ctx=loop_ctx,
                            device=w_sub.device,
                        ),
                    )
                    del lin, w_sub
                    n_replaced += 1

            lin_d = mlp.down_proj
            wd_sub = lin_d.weight.detach().index_select(1, idx_t)
            bd_sub = lin_d.bias.detach() if lin_d.bias is not None else None
            pk_d = pack_weight_tensor(torch, wd_sub, bits=bits)
            mlp.down_proj = CompactLinear(
                pk_d,
                bias=bd_sub,
                max_loops=max_loops if use_lora else 0,
                lora_rank=lora_rank if use_lora else 0,
                lora_alpha=lora_alpha,
                loop_ctx=loop_ctx,
                device=wd_sub.device,
            )
            del lin_d, wd_sub
            n_replaced += 1

        # 2. Replace all remaining `nn.Linear` inside decoder block `ell` (Attention q,k,v,o & unpruned MLP)
        for parent_name, parent_mod in list(blk.named_modules()):
            for child_name, child_mod in list(parent_mod.named_children()):
                if isinstance(child_mod, nn.Linear):
                    dev = child_mod.weight.device
                    pk = pack_weight_tensor(torch, child_mod.weight.detach(), bits=bits)
                    new_mod = CompactLinear(
                        pk,
                        bias=child_mod.bias.detach() if child_mod.bias is not None else None,
                        max_loops=max_loops if use_lora else 0,
                        lora_rank=lora_rank if use_lora else 0,
                        lora_alpha=lora_alpha,
                        loop_ctx=loop_ctx,
                        device=dev,
                    )
                    setattr(parent_mod, child_name, new_mod)
                    del child_mod
                    n_replaced += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mem_stats = compute_model_storage_bytes(harness.model)
    if torch.cuda.is_available():
        mem_stats["cuda_allocated_gb"] = round(float(torch.cuda.memory_allocated()) / 1e9, 3)
    return {
        "n_linear_compressed": n_replaced,
        "removed_multimodal_towers": removed_towers,
        "pruned_indices": pruned_indices,
        "storage": mem_stats,
    }


# ---------------------------------------------------------------------------
# 3. Actions 2 & 4: `Gemma4LoopCompactModel`, Step Bridge, Halting & Safety
# ---------------------------------------------------------------------------

def build_step_bridge_and_halting_modules(
    torch: Any,
    procrustes_fit: dict[str, Any],
    max_loops: int = 4,
    rank: int = 128,
    init_gate: float = 0.14,
    halting_hidden: int = 128,
    device: Any = None,
) -> tuple[Any, Any]:
    """Create `ProcrustesStepBridge` (with per-step gates, step embeddings & safety fold) + `AdaptiveHaltingHead`."""
    nn = torch.nn
    F = torch.nn.functional
    d = int(procrustes_fit["d_model"])
    n_transitions = max(1, max_loops - 1)

    class ProcrustesStepBridge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d_model = d
            self.max_loops = max_loops
            self.rank = rank

            r_init = torch.tensor(procrustes_fit["r_0"], dtype=torch.float32)
            self.register_buffer("r_ortho_base", r_init.clone())
            self.register_buffer("r_ortho", r_init)
            self.register_buffer("mu_src", torch.tensor(procrustes_fit["mu_src"], dtype=torch.float32))
            self.register_buffer("mu_dst", torch.tensor(procrustes_fit["mu_dst"], dtype=torch.float32))
            self.register_buffer("rms_src", torch.tensor(float(procrustes_fit["rms_src"]), dtype=torch.float32))

            self.log_rms_dst = nn.Parameter(
                torch.full((n_transitions, d), math.log(max(float(procrustes_fit["rms_dst"]), 1e-6)), dtype=torch.float32)
            )
            self.lora_a = nn.Parameter(torch.randn(rank, d, dtype=torch.float32) * (1.0 / math.sqrt(d)))
            self.lora_b = nn.Parameter(torch.zeros(d, rank, dtype=torch.float32))
            self.step_emb = nn.Parameter(torch.zeros(n_transitions, d, dtype=torch.float32))

            g0 = min(max(init_gate, 1e-4), 1.0 - 1e-4)
            logit0 = math.log(g0 / (1.0 - g0))
            self.gate_scalar = nn.Parameter(torch.full((n_transitions, 1), logit0, dtype=torch.float32))
            self.gate_logits = nn.Parameter(torch.zeros((n_transitions, d), dtype=torch.float32))

            # Optional 3D orthogonal safety subspace `Q_42 [D, 3]`
            self.register_buffer("safety_q", torch.zeros(d, 3, dtype=torch.float32))
            self.safety_active: bool = False

        def effective_gate(self, transition_idx: int) -> Any:
            t_idx = min(max(0, int(transition_idx)), n_transitions - 1)
            return torch.sigmoid(self.gate_scalar[t_idx] + self.gate_logits[t_idx])

        def set_safety_subspace(self, q_mat_np: np.ndarray, active: bool = False) -> None:
            q_t = torch.tensor(q_mat_np, dtype=torch.float32, device=self.r_ortho.device)
            self.safety_q.copy_(q_t)
            self.apply_safety_fold(active=active)

        def apply_safety_fold(self, active: bool) -> None:
            """Zero-cost algebraic fold: `R_safe = R_proc @ (I - Q Q^T)`."""
            with torch.no_grad():
                if active:
                    # R_base @ (I - Q Q^T) = R_base - (R_base @ Q) @ Q^T
                    rq = torch.matmul(self.r_ortho_base, self.safety_q)
                    r_safe = self.r_ortho_base - torch.matmul(rq, self.safety_q.t())
                    self.r_ortho.copy_(r_safe)
                    self.safety_active = True
                else:
                    self.r_ortho.copy_(self.r_ortho_base)
                    self.safety_active = False

        def forward(self, h_anchor: Any, h_end: Any, transition_idx: int = 0) -> Any:
            t_idx = min(max(0, int(transition_idx)), n_transitions - 1)
            orig_dtype = h_anchor.dtype
            anc = h_anchor.to(torch.float32)
            x = h_end.to(torch.float32)

            if self.safety_active:
                # Also scrub low-rank residual input along Q_42 so neither branch leaks moral bias
                x = x - torch.matmul(torch.matmul(x, self.safety_q), self.safety_q.t())

            x_c = x - self.mu_src
            cur_rms = torch.sqrt(torch.mean(x_c * x_c, dim=-1, keepdim=True) + 1e-8)
            x_n = x_c / cur_rms

            rot = torch.matmul(x_n, self.r_ortho.t())
            delta = torch.matmul(torch.matmul(x_n, self.lora_a.t()), self.lora_b.t())
            mapped = torch.exp(self.log_rms_dst[t_idx]) * (rot + delta) + self.mu_dst + self.step_emb[t_idx]

            g = self.effective_gate(t_idx)
            blended = (1.0 - g) * anc + g * mapped
            return blended.to(orig_dtype)

    class AdaptiveHaltingHead(nn.Module):
        """Lightweight `L42` confidence head predicting whether current loop `k` is ready to exit."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d + max_loops, halting_hidden, dtype=torch.float32)
            self.fc2 = nn.Linear(halting_hidden, 1, dtype=torch.float32)

        def forward(self, h_end_last: Any, loop_idx: int) -> Any:
            # `h_end_last`: `[B, D]`
            x = h_end_last.to(torch.float32)
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_n = x / rms
            one_hot = torch.zeros((x.shape[0], max_loops), dtype=torch.float32, device=x.device)
            one_hot[:, min(max(0, loop_idx), max_loops - 1)] = 1.0
            feat = torch.cat([x_n, one_hot], dim=-1)
            return torch.sigmoid(self.fc2(F.silu(self.fc1(feat)))).squeeze(-1)

    bridge = ProcrustesStepBridge()
    halting = AdaptiveHaltingHead()
    if device is not None:
        bridge = bridge.to(device)
        halting = halting.to(device)
    return bridge, halting


class Gemma4LoopCompactController:
    """Native recurrent controller binding `Gemma 4` decoder blocks, `LoopIndexedLoRA`, `ProcrustesStepBridge`, and `AdaptiveHaltingHead`."""

    def __init__(
        self,
        harness: TransformerHarness,
        bridge: Any,
        halting_head: Any,
        loop_ctx: dict[str, int],
        start_layer: int,
        end_layer: int,
        max_loops: int = 4,
    ) -> None:
        self.harness = harness
        self.bridge = bridge
        self.halting_head = halting_head
        self.loop_ctx = loop_ctx
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.max_loops = int(max_loops)
        self.last_loops_executed: float = 1.0
        self.last_batch_loops: list[int] = []
        self.safety_clamp_funnel: bool = False
        self.safety_q_by_layer: dict[int, Any] = {}
        self.safety_mu_by_layer: dict[int, Any] = {}

    def set_funnel_safety_clamp(
        self,
        dirs_by_layer: dict[int, np.ndarray],
        neutral_means_by_layer: dict[int, np.ndarray],
        active: bool = False,
    ) -> None:
        torch = self.harness.torch
        dev = self.harness.input_device
        for ell in range(self.start_layer, self.end_layer + 1):
            q_np, mu_np = self.harness.build_subspace_clamp_matrices(
                dirs_by_layer[ell], neutral_means_by_layer.get(ell)
            )
            self.safety_q_by_layer[ell] = torch.tensor(q_np, dtype=torch.float32, device=dev)
            self.safety_mu_by_layer[ell] = torch.tensor(mu_np, dtype=torch.float32, device=dev)
        self.safety_clamp_funnel = active

    @contextmanager
    def active_recurrence(
        self,
        n_loops: int = 2,
        adaptive: bool = False,
        halt_threshold: float = 0.65,
        capture_loop_states: list[Any] | None = None,
    ) -> Iterator[None]:
        """Run `Prelude -> K Loops of L24..L42 (fixed or adaptive halting) -> Coda`."""
        torch = self.harness.torch
        self.loop_ctx["loop_idx"] = 0
        self.last_loops_executed = 1

        if n_loops <= 1 and not adaptive and not self.safety_clamp_funnel and capture_loop_states is None:
            yield
            return

        target_max = self.max_loops if adaptive else min(max(1, int(n_loops)), self.max_loops)
        state_box: dict[str, Any] = {
            "in_extra_loop": False,
            "h_anchor": None,
            "call_specs": {},
        }
        handles = []

        def _maybe_clamp(ell: int, h_tensor: Any) -> Any:
            if not self.safety_clamp_funnel or ell not in self.safety_q_by_layer:
                return h_tensor
            q_t = self.safety_q_by_layer[ell]
            mu_t = self.safety_mu_by_layer[ell]
            h32 = h_tensor.to(torch.float32)
            proj = torch.matmul(torch.matmul(h32, q_t), q_t.t()) - mu_t
            return (h32 - proj).to(h_tensor.dtype)

        for ell in range(self.start_layer, self.end_layer + 1):
            blk = self.harness.layers[ell]

            def _make_pre(idx: int):
                def _pre(_mod: Any, args: Any, kwargs: Any):
                    if state_box["in_extra_loop"]:
                        return
                    if len(args) >= 1:
                        h_in = args[0]
                        had_pos = True
                        rest_args = args[1:]
                    else:
                        h_in = kwargs["hidden_states"]
                        had_pos = False
                        rest_args = ()

                    clean_kw = {}
                    for k, v in kwargs.items():
                        if k == "hidden_states" and not had_pos:
                            continue
                        if k in ("past_key_value", "past_key_values"):
                            clean_kw[k] = None
                        elif k == "use_cache":
                            clean_kw[k] = False
                        else:
                            clean_kw[k] = v

                    if idx == self.start_layer:
                        self.loop_ctx["loop_idx"] = 0
                        state_box["h_anchor"] = h_in
                    state_box["call_specs"][idx] = (had_pos, rest_args, clean_kw)
                return _pre

            handles.append(blk.register_forward_pre_hook(_make_pre(ell), with_kwargs=True))

            if self.safety_clamp_funnel and ell < self.end_layer:
                def _make_mid_clamp(idx: int):
                    def _post_c(_m: Any, _a: Any, out: Any):
                        if state_box["in_extra_loop"]:
                            return out
                        wt = isinstance(out, tuple)
                        h0 = _maybe_clamp(idx, out[0] if wt else out)
                        return (h0,) + out[1:] if wt else h0
                    return _post_c
                handles.append(blk.register_forward_hook(_make_mid_clamp(ell)))

        def _end_post_hook(_mod: Any, _args: Any, out: Any):
            if state_box["in_extra_loop"]:
                return out
            was_tuple = isinstance(out, tuple)
            h_cur = out[0] if was_tuple else out
            rest_tuple = out[1:] if was_tuple else ()
            h_cur = _maybe_clamp(self.end_layer, h_cur)
            h_anchor = state_box["h_anchor"]

            if capture_loop_states is not None:
                capture_loop_states.append(h_cur[:, -1, :].detach().clone())

            bsz = int(h_cur.shape[0])
            halted = torch.zeros(bsz, dtype=torch.bool, device=h_cur.device)
            sample_loops = torch.ones(bsz, dtype=torch.long, device=h_cur.device)

            if adaptive and target_max > 1:
                with torch.no_grad():
                    p_halt = self.halting_head(h_cur[:, -1, :], loop_idx=0)
                    halted = p_halt >= halt_threshold
                    if bool(torch.all(halted).item()):
                        self.last_batch_loops = sample_loops.cpu().tolist()
                        self.last_loops_executed = 1.0
                        self.loop_ctx["loop_idx"] = 0
                        return (h_cur,) + rest_tuple if was_tuple else h_cur

            state_box["in_extra_loop"] = True
            try:
                for loop_i in range(1, target_max):
                    self.loop_ctx["loop_idx"] = loop_i
                    h_prev = h_cur
                    h_next = self.bridge(h_anchor, h_cur, transition_idx=loop_i - 1)
                    for ell in range(self.start_layer, self.end_layer + 1):
                        blk_l = self.harness.layers[ell]
                        had_pos, rest_args, clean_kw = state_box["call_specs"][ell]
                        if had_pos:
                            res = blk_l(h_next, *rest_args, **clean_kw)
                        else:
                            res = blk_l(hidden_states=h_next, **clean_kw)
                        h_next = res[0] if isinstance(res, tuple) else res
                        h_next = _maybe_clamp(ell, h_next)

                    if adaptive:
                        h_cur = torch.where(halted.view(-1, 1, 1), h_prev, h_next)
                        sample_loops = sample_loops + (~halted).to(torch.long)
                    else:
                        h_cur = h_next
                        sample_loops.fill_(loop_i + 1)

                    if capture_loop_states is not None:
                        capture_loop_states.append(h_cur[:, -1, :].detach().clone())

                    if adaptive and loop_i < target_max - 1:
                        with torch.no_grad():
                            p_halt = self.halting_head(h_cur[:, -1, :], loop_idx=loop_i)
                            halted = halted | (p_halt >= halt_threshold)
                            if bool(torch.all(halted).item()):
                                break
            finally:
                state_box["in_extra_loop"] = False
                self.loop_ctx["loop_idx"] = 0
                state_box["h_anchor"] = None
                state_box["call_specs"].clear()

            self.last_batch_loops = sample_loops.cpu().tolist()
            self.last_loops_executed = float(sample_loops.to(torch.float32).mean().item())
            return (h_cur,) + rest_tuple if was_tuple else h_cur

        handles.append(self.harness.layers[self.end_layer].register_forward_hook(_end_post_hook))

        try:
            yield
        finally:
            self.loop_ctx["loop_idx"] = 0
            for h in handles:
                h.remove()


# ---------------------------------------------------------------------------
# 4. Stage 0 Teacher Caching, Randomized Unroll Distillation & Halting Fit
# ---------------------------------------------------------------------------

def cache_thought_teacher_targets(
    harness: TransformerHarness,
    train_items: Sequence[ReasoningItem],
    end_layer: int,
    top_m: int = 64,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Run 1 batched forward pass on intact `BF16` `Gemma 4 31B` WITH `<thought>...</thought>` pre-filled!

    Extracts for each training item:
      1. Top-M soft answer-token probability distribution conditioned on the `<thought>` trace.
      2. Final `L42` hidden vector right before the first answer token (`H_42^(thought)`).
    """
    torch = harness.torch
    F = torch.nn.functional
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))
    pad_id = int(tok.pad_token_id or 0)
    stop_ids = resolve_thought_stop_ids(harness)
    # Prefer `<turn|>` token id for Gemma 4 chat completion termination when present
    turn_close_id = None
    if hasattr(tok, "convert_tokens_to_ids"):
        for t_sym in ("<turn|>", "<end_of_turn>", "<|im_end|>"):
            try:
                tid = tok.convert_tokens_to_ids(t_sym)
                if isinstance(tid, int) and tid >= 0 and tid != getattr(tok, "unk_token_id", None):
                    turn_close_id = tid
                    break
            except Exception:
                continue
    if turn_close_id is not None:
        eos_tail = [int(turn_close_id)]
    elif stop_ids:
        eos_tail = [int(stop_ids[0])]
    else:
        eos_tail = tok.encode("\n", add_special_tokens=False)

    targets: list[dict[str, Any]] = []
    for i in range(0, len(train_items), batch_size):
        chunk = train_items[i : i + batch_size]
        p_thought_list = []
        a_ids_list = []
        for it in chunk:
            # Use Gemma 4's native `<|channel>thought\n...<channel|>` scratchpad format when available!
            p_probe = format_chat_prompt_thinking(tok, it.prompt_user, enable_thinking=True)
            if "<|channel>thought" in p_probe:
                p_str = format_chat_prompt_thinking(
                    tok,
                    it.prompt_user,
                    enable_thinking=True,
                    assistant_prefill=f"{it.thought_trace}<channel|>",
                )
            else:
                p_str = format_chat_prompt(
                    tok,
                    it.prompt_user,
                    assistant_prefill=f"<thought>\n{it.thought_trace}\n</thought>\n",
                )
            p_ids = tok.encode(p_str, add_special_tokens=not has_chat_tmpl)[-320:]
            a_ids = tok.encode(f"FINAL_INTEGER: {it.answer_int}", add_special_tokens=False) + eos_tail
            p_thought_list.append(p_ids)
            a_ids_list.append(a_ids)

        seqs = [p_i + a_i for p_i, a_i in zip(p_thought_list, a_ids_list)]
        max_l = max(len(s) for s in seqs)
        b_ids = []
        b_msk = []
        ans_starts = []
        for s, p_i in zip(seqs, p_thought_list):
            pad_n = max_l - len(s)
            b_ids.append([pad_id] * pad_n + s)
            b_msk.append([0] * pad_n + [1] * len(s))
            ans_starts.append(pad_n + len(p_i))

        inp = torch.tensor(b_ids, dtype=torch.long, device=harness.input_device)
        msk = torch.tensor(b_msk, dtype=torch.long, device=harness.input_device)

        box_h42: dict[str, Any] = {}

        def _cap_e(_m: Any, _a: Any, out: Any):
            t = out[0] if isinstance(out, tuple) else out
            box_h42["h"] = t.detach().to(torch.float16).cpu()

        h_hook = harness.layers[end_layer].register_forward_hook(_cap_e)
        try:
            with torch.no_grad():
                out = harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
                probs = F.softmax(out.logits.to(torch.float32), dim=-1)
        finally:
            h_hook.remove()

        h42_batch = box_h42["h"]
        for b_idx, (a_start, a_i) in enumerate(zip(ans_starts, a_ids_list)):
            # Prediction positions are `[a_start - 1 .. max_l - 2]` predicting `a_i` (`len(a_i)` tokens)
            pred_slice = probs[b_idx, a_start - 1 : max_l - 1, :]  # [A_len, V]
            k_eff = min(top_m, int(pred_slice.shape[-1]))
            tp, ti = torch.topk(pred_slice, k=k_eff, dim=-1)
            # Pre-answer trigger hidden state at `a_start - 1`
            h_vec = h42_batch[b_idx, a_start - 1, :].clone()
            targets.append({
                "ans_token_ids": a_i,
                "top_probs": tp.to(torch.float16).cpu(),
                "top_idx": ti.to(torch.int32).cpu(),
                "h42_trigger": h_vec,
            })

    return targets


def collect_trainable_loop_parameters(harness: TransformerHarness, bridge: Any, start_layer: int, end_layer: int) -> list[Any]:
    for p in harness.model.parameters():
        p.requires_grad_(False)
    params: list[Any] = []
    for p in bridge.parameters():
        p.requires_grad_(True)
        params.append(p)
    for ell in range(start_layer, end_layer + 1):
        for name, p in harness.layers[ell].named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                params.append(p)
    return params


def train_loop_compact_randomized(
    controller: Gemma4LoopCompactController,
    train_items: Sequence[ReasoningItem],
    teacher_targets: Sequence[dict[str, Any]],
    anchor_refs: Sequence[dict[str, Any]],
    n_steps: int = 560,
    batch_size: int = 4,
    lr_gate_scalar: float = 8.0e-3,
    lr_bridge: float = 1.5e-3,
    lr_lora: float = 2.5e-4,
) -> dict[str, Any]:
    """Action 3: Randomized Unroll (`K ~ Uniform{1,2,3,4}`) + `<thought>` Soft-Logit & `L42` Latent Self-Distillation."""
    harness = controller.harness
    torch = harness.torch
    F = torch.nn.functional
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))
    pad_id = int(tok.pad_token_id or 0)

    for p in harness.model.parameters():
        p.requires_grad_(False)
    gate_scalar_params = [controller.bridge.gate_scalar]
    bridge_other_params = [p for n, p in controller.bridge.named_parameters() if n != "gate_scalar"]
    for p in gate_scalar_params + bridge_other_params:
        p.requires_grad_(True)
    bridge_params = gate_scalar_params + bridge_other_params
    lora_params: list[Any] = []
    for ell in range(controller.start_layer, controller.end_layer + 1):
        for name, p in harness.layers[ell].named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                lora_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": gate_scalar_params, "lr": lr_gate_scalar, "weight_decay": 0.0},
            {"params": bridge_other_params, "lr": lr_bridge, "weight_decay": 1e-4},
            {"params": lora_params, "lr": lr_lora, "weight_decay": 1e-4},
        ]
    )
    warmup_steps = min(48, max(4, n_steps // 10))

    def _lr_lambda(cur_step: int) -> float:
        if cur_step < warmup_steps:
            return float(cur_step + 1) / float(warmup_steps)
        prog = float(cur_step - warmup_steps) / float(max(1, n_steps - warmup_steps))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    rng = random.Random(20261121)
    indices = list(range(len(train_items)))
    rng.shuffle(indices)

    ce_curve: list[float] = []
    kd_curve: list[float] = []
    cos_curve: list[float] = []
    kl_anc_curve: list[float] = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        k_loops = int(rng.choice((1, 2, 3, 4, 2, 3, 4)))

        batch_idxs = [indices[(step * batch_size + j) % len(indices)] for j in range(batch_size)]
        if step > 0 and (step * batch_size) % len(indices) < batch_size:
            rng.shuffle(indices)

        p_ids_list = []
        a_ids_list = []
        t_batch = []
        for idx in batch_idxs:
            it = train_items[idx]
            tgt = teacher_targets[idx]
            p_str = format_chat_prompt(tok, it.prompt_user)
            p_ids = tok.encode(p_str, add_special_tokens=not has_chat_tmpl)[-256:]
            p_ids_list.append(p_ids)
            a_ids_list.append(list(tgt["ans_token_ids"]))
            t_batch.append(tgt)

        seqs = [p_i + a_i for p_i, a_i in zip(p_ids_list, a_ids_list)]
        max_l = max(len(s) for s in seqs)
        b_ids = []
        b_msk = []
        b_labels = []
        ans_starts = []
        for s, p_i, a_i in zip(seqs, p_ids_list, a_ids_list):
            pad_n = max_l - len(s)
            b_ids.append([pad_id] * pad_n + s)
            b_msk.append([0] * pad_n + [1] * len(s))
            b_labels.append([-100] * (pad_n + len(p_i)) + a_i)
            ans_starts.append(pad_n + len(p_i))

        inp = torch.tensor(b_ids, dtype=torch.long, device=harness.input_device)
        msk = torch.tensor(b_msk, dtype=torch.long, device=harness.input_device)
        labels = torch.tensor(b_labels, dtype=torch.long, device=harness.input_device)

        h42_student_box: dict[str, Any] = {}

        def _cap_final_l42(_m: Any, _a: Any, out: Any):
            t = out[0] if isinstance(out, tuple) else out
            h42_student_box["h"] = t

        with torch.enable_grad():
            with controller.active_recurrence(n_loops=k_loops, adaptive=False):
                # Note: `active_recurrence`'s end hook runs BEFORE a subsequently registered hook, so `h` is post-K-loops!
                h_end_hook = harness.layers[controller.end_layer].register_forward_hook(_cap_final_l42)
                try:
                    out = harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
                finally:
                    h_end_hook.remove()

            logits_f32 = out.logits.to(torch.float32)
            shift_logits = logits_f32[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Soft `<thought>`-teacher top-M KL distillation + `L42` latent alignment
            log_probs_all = F.log_softmax(logits_f32, dim=-1)
            kd_terms = []
            cos_terms = []
            h42_stu = h42_student_box["h"].to(torch.float32)

            for b_idx, (a_start, tgt) in enumerate(zip(ans_starts, t_batch)):
                t_probs = tgt["top_probs"].to(device=harness.input_device, dtype=torch.float32)
                t_idx = tgt["top_idx"].to(device=harness.input_device, dtype=torch.long)
                pred_lp = log_probs_all[b_idx, a_start - 1 : max_l - 1, :]  # [A_len, V]
                stu_lp = torch.gather(pred_lp, dim=-1, index=t_idx)
                t_lp = torch.log(t_probs.clamp(min=1e-12))
                kd_b = (t_probs * (t_lp - stu_lp)).sum(dim=-1).mean()
                kd_terms.append(kd_b)

                h_teach = tgt["h42_trigger"].to(device=harness.input_device, dtype=torch.float32)
                h_s = h42_stu[b_idx, a_start - 1, :]
                cos_sim = F.cosine_similarity(h_s.unsqueeze(0), h_teach.unsqueeze(0), dim=-1).squeeze(0)
                cos_terms.append(1.0 - cos_sim)

            kd_loss = torch.stack(kd_terms).mean()
            cos_loss = torch.stack(cos_terms).mean()
            total_loss = 0.50 * ce_loss + 0.50 * kd_loss + 0.30 * cos_loss

            kl_val = 0.0
            if anchor_refs and (step % 2 == 0):
                # Explicitly rotate `k_kl` through ALL loop counts `1, 2, ..., max_loops`!
                k_kl = ((step // 2) % controller.max_loops) + 1
                ref = anchor_refs[(step // 2) % len(anchor_refs)]
                r_inp = ref["input_ids"].to(harness.input_device)
                r_msk = ref["attention_mask"].to(harness.input_device)
                r_val = ref["valid_mask"].to(harness.input_device)
                r_probs = ref["top_p"].to(device=harness.input_device, dtype=torch.float32)
                r_top_lp = ref["top_lp"].to(device=harness.input_device, dtype=torch.float32)
                r_idx = ref["top_idx"].to(device=harness.input_device, dtype=torch.long)

                with controller.active_recurrence(n_loops=k_kl, adaptive=False):
                    r_out = harness.model(input_ids=r_inp, attention_mask=r_msk, use_cache=False)
                r_lp = F.log_softmax(r_out.logits[:, :-1, :].to(torch.float32), dim=-1)
                q_lp = torch.gather(r_lp, dim=-1, index=r_idx)
                tok_kl = (r_probs * (r_top_lp - q_lp)).sum(dim=-1).clamp(min=0.0)
                if bool(r_val.any().item()):
                    kl_anc = tok_kl[r_val].mean()
                else:
                    kl_anc = tok_kl.mean()
                kl_val = float(kl_anc.detach().item())
                total_loss = total_loss + 0.55 * kl_anc

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge_params + lora_params, max_norm=0.5)
            optimizer.step()
            scheduler.step()

        ce_v = float(ce_loss.detach().item())
        kd_v = float(kd_loss.detach().item())
        cs_v = float(1.0 - cos_loss.detach().item())
        ce_curve.append(ce_v)
        kd_curve.append(kd_v)
        cos_curve.append(cs_v)
        if step % 2 == 0:
            kl_anc_curve.append(kl_val)

        if step == 0 or (step + 1) % 40 == 0 or step == n_steps - 1:
            g_means = [
                round(float(controller.bridge.effective_gate(t).mean().item()), 3)
                for t in range(controller.bridge.gate_logits.shape[0])
            ]
            print(
                f"  [LoopCompact step {step + 1:3d}/{n_steps} | K={k_loops}] "
                f"CE={ce_v:.4f} KD={kd_v:.4f} L42_cos={cs_v:.3f} KL_anc={kl_anc_curve[-1] if kl_anc_curve else 0.0:.4f} "
                f"gates={g_means}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    all_trained = bridge_params + lora_params
    n_trainable = int(sum(p.numel() for p in all_trained))
    g_per_step = [
        round(float(controller.bridge.effective_gate(t).detach().mean().item()), 4)
        for t in range(controller.bridge.gate_logits.shape[0])
    ]
    sub_idx = np.linspace(0, len(ce_curve) - 1, min(48, len(ce_curve)), dtype=int)
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "trainable_params": n_trainable,
        "bridge_params": int(sum(p.numel() for p in bridge_params)),
        "lora_params": int(sum(p.numel() for p in lora_params)),
        "train_seconds": round(elapsed, 2),
        "ce_first10": round(float(np.mean(ce_curve[:10])), 4),
        "ce_last10": round(float(np.mean(ce_curve[-10:])), 4),
        "kd_first10": round(float(np.mean(kd_curve[:10])), 4),
        "kd_last10": round(float(np.mean(kd_curve[-10:])), 4),
        "l42_cos_first10": round(float(np.mean(cos_curve[:10])), 4),
        "l42_cos_last10": round(float(np.mean(cos_curve[-10:])), 4),
        "kl_anchor_last5": round(float(np.mean(kl_anc_curve[-5:])), 5) if kl_anc_curve else 0.0,
        "learned_step_gates": g_per_step,
        "ce_curve_subsampled": [round(ce_curve[i], 4) for i in sub_idx],
        "cos_curve_subsampled": [round(cos_curve[i], 4) for i in sub_idx],
    }


def fit_adaptive_halting_head(
    controller: Gemma4LoopCompactController,
    calib_easy: Sequence[ReasoningItem],
    calib_hard: Sequence[ReasoningItem],
    epochs: int = 240,
) -> dict[str, Any]:
    """Action 4: Fit `AdaptiveHaltingHead` on `L42` loop states so easy prompts halt at `K=1` and hard prompts loop `K=3..4`."""
    harness = controller.harness
    torch = harness.torch
    nn = torch.nn
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))

    feats_x: list[Any] = []
    labels_y: list[float] = []
    loop_ids: list[int] = []

    samples = [(it, True) for it in calib_easy] + [(it, False) for it in calib_hard]
    for it, is_easy in samples:
        p_str = format_chat_prompt(tok, it.prompt_user)
        enc = tok(p_str, add_special_tokens=not has_chat_tmpl, return_tensors="pt", truncation=True, max_length=256)
        inp = enc["input_ids"].to(harness.input_device)
        msk = enc["attention_mask"].to(harness.input_device)
        captured: list[Any] = []
        with torch.no_grad():
            with controller.active_recurrence(n_loops=controller.max_loops, adaptive=False, capture_loop_states=captured):
                harness.model(input_ids=inp, attention_mask=msk, use_cache=False)

        for k_i, h_vec in enumerate(captured):
            feats_x.append(h_vec.squeeze(0).to(torch.float32))
            loop_ids.append(k_i)
            if is_easy:
                labels_y.append(1.0)  # Easy 1-hop items should halt immediately at K=1 (`loop_idx=0`)
            else:
                # Multi-hop / balance-sheet items should continue through K=1,2 and halt at K=3 or 4
                labels_y.append(1.0 if k_i >= controller.max_loops - 2 and it.hops <= 4 else (1.0 if k_i == controller.max_loops - 1 else 0.0))

    for p in controller.halting_head.parameters():
        p.requires_grad_(True)

    x_mat = torch.stack(feats_x, dim=0).to(harness.input_device)
    y_vec = torch.tensor(labels_y, dtype=torch.float32, device=harness.input_device)
    k_arr = np.asarray(loop_ids, dtype=np.int64)

    opt = torch.optim.AdamW(controller.halting_head.parameters(), lr=3e-3, weight_decay=1e-4)
    bce = nn.BCELoss()
    with torch.enable_grad():
        for _ep in range(epochs):
            opt.zero_grad(set_to_none=True)
            preds = torch.zeros_like(y_vec)
            for k_val in range(controller.max_loops):
                mask_k = torch.tensor(k_arr == k_val, dtype=torch.bool, device=harness.input_device)
                if bool(mask_k.any().item()):
                    preds[mask_k] = controller.halting_head(x_mat[mask_k], loop_idx=k_val)
            loss = bce(preds.clamp(1e-5, 1.0 - 1e-5), y_vec)
            loss.backward()
            opt.step()

    for p in controller.halting_head.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        acc_list = []
        for k_val in range(controller.max_loops):
            mask_k = torch.tensor(k_arr == k_val, dtype=torch.bool, device=harness.input_device)
            if bool(mask_k.any().item()):
                pk = controller.halting_head(x_mat[mask_k], loop_idx=k_val)
                acc_list.extend(((pk >= 0.65).to(torch.float32) == y_vec[mask_k]).cpu().numpy().tolist())

    return {
        "n_calib_states": int(len(labels_y)),
        "final_bce_loss": round(float(loss.detach().item()), 4),
        "calib_gate_acc": round(float(np.mean(acc_list)), 4) if acc_list else 0.0,
    }


# ---------------------------------------------------------------------------
# 5. Evaluation & Standalone HF Artifact Export / Round-Trip Verification
# ---------------------------------------------------------------------------

TURN_STOP_MARKERS = ("<turn|>", "<end_of_turn>", "<eos>", "<|im_end|>", "<|eot_id|>")


def _parse_int_pred(raw_decoded_with_special: str) -> int | None:
    """Truncate strictly at the first turn-closing special token, then extract the FIRST integer."""
    cut = len(raw_decoded_with_special)
    for marker in TURN_STOP_MARKERS:
        pos = raw_decoded_with_special.find(marker)
        if pos >= 0 and pos < cut:
            cut = pos
    first_turn = RESIDUAL_SPECIAL_RE.sub("", raw_decoded_with_special[:cut]).strip()
    idx = first_turn.find("FINAL_INTEGER:")
    sub = first_turn[idx + len("FINAL_INTEGER:") :] if idx >= 0 else first_turn
    m = RE_FIRST_INT.search(sub)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def evaluate_reasoning_suite(
    harness: TransformerHarness,
    items: Sequence[ReasoningItem],
    controller: Gemma4LoopCompactController | None = None,
    n_loops: int = 1,
    adaptive: bool = False,
    halt_threshold: float = 0.65,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate greedy `0-CoT` exact-match accuracy, teacher-forced gold log-prob, `ms/item`, and mean loops `E[K]`."""
    torch = harness.torch
    F = torch.nn.functional
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))
    prompts = [format_chat_prompt(tok, it.prompt_user) for it in items]
    gold_strs = [f"FINAL_INTEGER: {it.answer_int}" for it in items]
    stop_set = set(resolve_thought_stop_ids(harness))
    eos_id = tok.eos_token_id
    pad_id = int(tok.pad_token_id or 0)
    pad_fill = int(eos_id if eos_id is not None else pad_id)

    preds: list[int | None] = []
    loops_used: list[int] = []

    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(
            chunk,
            add_special_tokens=not has_chat_tmpl,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        )
        cur_ids = enc["input_ids"].to(harness.input_device)
        cur_mask = enc["attention_mask"].to(harness.input_device)
        p_len = cur_ids.shape[1]
        bsz = cur_ids.shape[0]
        done = torch.zeros(bsz, dtype=torch.bool, device=harness.input_device)
        stop_tensor = torch.tensor(sorted(stop_set), dtype=torch.long, device=harness.input_device) if stop_set else None
        batch_first_loops = [int(n_loops)] * bsz

        with torch.no_grad():
            for step_idx in range(14):
                if controller is not None:
                    with controller.active_recurrence(n_loops=n_loops, adaptive=adaptive, halt_threshold=halt_threshold):
                        out = harness.model(input_ids=cur_ids, attention_mask=cur_mask, use_cache=False)
                    if step_idx == 0:
                        if controller.last_batch_loops and len(controller.last_batch_loops) == bsz:
                            batch_first_loops = list(controller.last_batch_loops)
                        else:
                            batch_first_loops = [int(round(controller.last_loops_executed))] * bsz
                else:
                    out = harness.model(input_ids=cur_ids, attention_mask=cur_mask, use_cache=False)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
                next_tok = torch.where(done, torch.full_like(next_tok, pad_fill), next_tok)
                cur_ids = torch.cat([cur_ids, next_tok.unsqueeze(1)], dim=1)
                cur_mask = torch.cat([cur_mask, (~done).to(cur_mask.dtype).unsqueeze(1)], dim=1)
                if stop_tensor is not None and stop_tensor.numel() > 0:
                    done = done | torch.isin(next_tok, stop_tensor)
                if bool(torch.all(done).item()):
                    break

        loops_used.extend(batch_first_loops)
        for row in cur_ids[:, p_len:]:
            txt = tok.decode(row.tolist(), skip_special_tokens=False)
            preds.append(_parse_int_pred(txt))

    ms_per_item = 1000.0 * (time.perf_counter() - t0) / max(1, len(items))

    # Teacher-forced gold log-probability
    gold_lps: list[float] = []
    for i in range(0, len(items), batch_size):
        p_chunk = prompts[i : i + batch_size]
        g_chunk = gold_strs[i : i + batch_size]
        p_ids_l = [tok.encode(p, add_special_tokens=not has_chat_tmpl)[-512:] for p in p_chunk]
        a_ids_l = [tok.encode(g, add_special_tokens=False) for g in g_chunk]
        f_l = [p_i + a_i for p_i, a_i in zip(p_ids_l, a_ids_l)]
        max_l = max(len(x) for x in f_l)
        b_ids = []
        b_msk = []
        spans = []
        for s, p_i in zip(f_l, p_ids_l):
            pn = max_l - len(s)
            b_ids.append([pad_id] * pn + s)
            b_msk.append([0] * pn + [1] * len(s))
            spans.append((pn + len(p_i), max_l))
        inp = torch.tensor(b_ids, dtype=torch.long, device=harness.input_device)
        msk = torch.tensor(b_msk, dtype=torch.long, device=harness.input_device)
        with torch.no_grad():
            if controller is not None:
                with controller.active_recurrence(n_loops=min(max(1, n_loops), controller.max_loops), adaptive=False):
                    out = harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
            else:
                out = harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
            lp = F.log_softmax(out.logits[:, :-1, :].to(torch.float32), dim=-1)
            for b_idx, (sp, ep) in enumerate(spans):
                vals = [float(lp[b_idx, pos - 1, inp[b_idx, pos]].item()) for pos in range(max(1, sp), ep)]
                gold_lps.append(float(np.sum(vals)))

    correct = [int(p == it.answer_int) for p, it in zip(preds, items)]
    by_hops: dict[str, Any] = {}
    for h_val in sorted({it.hops for it in items}):
        idxs = [j for j, it in enumerate(items) if it.hops == h_val]
        by_hops[f"hop_{h_val}"] = {
            "acc": round(float(np.mean([correct[j] for j in idxs])), 4),
            "mean_gold_logprob": round(float(np.mean([gold_lps[j] for j in idxs])), 4),
            "mean_loops": round(float(np.mean([loops_used[j] for j in idxs])), 3),
            "n": len(idxs),
        }

    return {
        "overall_acc": round(float(np.mean(correct)), 4),
        "mean_gold_logprob": round(float(np.mean(gold_lps)), 4),
        "mean_loops": round(float(np.mean(loops_used)), 3),
        "ms_per_item": round(ms_per_item, 2),
        "n_items": len(items),
        "by_hops": by_hops,
    }


def extract_safety_moral_basis(harness: TransformerHarness, batch_size: int = 16) -> tuple[LayerBasis, dict[int, np.ndarray]]:
    """Extract non-financial 3D moral subspace `{spite, contamination, predation}` + neutral residual means."""
    concepts = load_concepts(STIMULI_DIR / "concepts.jsonl")
    # Use `spite`, `contamination`, `predation` + affect controls `valence`, `arousal` (20 pairs each = 100 pairs)
    needed = ("spite", "contamination", "predation", "valence", "arousal")
    sub_c = [c for c in concepts if c.concept in needed]
    pos_p = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(c.positive)) for c in sub_c]
    neg_p = [format_chat_prompt(harness.tokenizer, build_scenario_reading_prompt(c.negative)) for c in sub_c]
    pos_h = harness.extract_last_token_states(pos_p, batch_size=batch_size)
    neg_h = harness.extract_last_token_states(neg_p, batch_size=batch_size)

    pos_by_c = {
        cn: np.stack([pos_h[i] for i, item in enumerate(sub_c) if item.concept == cn], axis=0)
        for cn in needed
    }
    neg_by_c = {
        cn: np.stack([neg_h[i] for i, item in enumerate(sub_c) if item.concept == cn], axis=0)
        for cn in needed
    }
    locked = extract_concept_basis(
        pos_by_concept=pos_by_c,
        neg_by_concept=neg_by_c,
        concepts=needed,
        controls=("valence", "arousal"),
    )

    ult_500 = [build_inequity_item(fid, pool, ctx, "human", 500) for fid, pool, ctx in INEQUITY_FIXTURES[:8]]
    txs_tnt = [t for t in load_transactions(STIMULI_DIR / "transactions.jsonl") if t.archetype == "tainted_provenance"][:8]
    cf_san = [s for s in load_sanity_items(STIMULI_DIR / "sanity_arithmetic.jsonl") if s.category == "forward_cashflow"]

    neu_prompts = [
        format_chat_prompt(harness.tokenizer, build_choice_user_prompt(it.scenario, it.deviation_option, it.rational_option))
        for it in ult_500
    ] + [
        format_chat_prompt(harness.tokenizer, build_choice_user_prompt(t.control_scenario, t.deviation_option, t.rational_option))
        for t in txs_tnt
    ] + [
        format_chat_prompt(harness.tokenizer, build_choice_user_prompt(s.scenario, s.better_option, s.worse_option))
        for s in cf_san
    ]
    neu_states = harness.extract_last_token_states(neu_prompts, batch_size=batch_size)
    neutral_means = {ell: np.mean(neu_states[:, ell, :], axis=0) for ell in range(harness.n_layers)}
    return locked, neutral_means


def evaluate_moral_bias_and_sanity(
    harness: TransformerHarness,
    controller: Gemma4LoopCompactController | None = None,
    n_loops: int = 1,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Score Provenance Bias (`tainted_provenance` delta log-odds), Ultimatum `98/2` Spite delta, and Cashflow Sanity."""
    ult_20 = [build_inequity_item(fid, pool, ctx, "human", 20) for fid, pool, ctx in INEQUITY_FIXTURES]
    ult_500 = [build_inequity_item(fid, pool, ctx, "human", 500) for fid, pool, ctx in INEQUITY_FIXTURES]
    txs_tnt = [t for t in load_transactions(STIMULI_DIR / "transactions.jsonl") if t.archetype == "tainted_provenance"]
    cf_san = [s for s in load_sanity_items(STIMULI_DIR / "sanity_arithmetic.jsonl") if s.category == "forward_cashflow"]

    def _run_scores():
        s20, _ = harness.score_counterbalanced_choices(
            [x.scenario for x in ult_20], [x.deviation_option for x in ult_20], [x.rational_option for x in ult_20], batch_size=batch_size
        )
        s500, _ = harness.score_counterbalanced_choices(
            [x.scenario for x in ult_500], [x.deviation_option for x in ult_500], [x.rational_option for x in ult_500], batch_size=batch_size
        )
        sta, _ = harness.score_counterbalanced_choices(
            [t.archetype_scenario for t in txs_tnt], [t.deviation_option for t in txs_tnt], [t.rational_option for t in txs_tnt], batch_size=batch_size
        )
        stc, _ = harness.score_counterbalanced_choices(
            [t.control_scenario for t in txs_tnt], [t.deviation_option for t in txs_tnt], [t.rational_option for t in txs_tnt], batch_size=batch_size
        )
        ssan, _ = harness.score_counterbalanced_choices(
            [s.scenario for s in cf_san], [s.better_option for s in cf_san], [s.worse_option for s in cf_san], batch_size=batch_size
        )
        return s20, s500, sta, stc, ssan

    if controller is not None:
        with controller.active_recurrence(n_loops=n_loops, adaptive=False):
            s20, s500, sta, stc, ssan = _run_scores()
    else:
        s20, s500, sta, stc, ssan = _run_scores()

    ult_d = [a.target_log_odds - b.target_log_odds for a, b in zip(s20, s500)]
    tnt_d = [a.target_log_odds - b.target_log_odds for a, b in zip(sta, stc)]
    return {
        "provenance_bias_delta_lo": round(float(np.mean(tnt_d)), 4),
        "ultimatum_spite_delta_lo": round(float(np.mean(ult_d)), 4),
        "cashflow_sanity_acc": round(float(np.mean([sc.target_log_odds > 0.0 for sc in ssan])), 4),
    }


STANDALONE_HF_LOADER_SOURCE = '''"""Standalone loader for `Gemma-4-Loop-Compact` (`Hourglass INT4/INT8 + Procrustes Recurrent Funnel + Halting`)."""

from pathlib import Path
import json
import numpy as np
import torch

from afriat.models import TransformerHarness
from afriat.loop_compact import (
    Gemma4LoopCompactController,
    build_step_bridge_and_halting_modules,
    compress_decoder_layers_inplace,
)


def load_gemma4_loop_compact(package_dir: str | Path, base_model_id: str | None = None, device_map: str = "cuda"):
    pkg = Path(package_dir)
    cfg = json.loads((pkg / "config_loop_compact.json").read_text(encoding="utf-8"))
    model_id = base_model_id or cfg["base_model_id"]
    harness = TransformerHarness.from_pretrained(model_id, dtype="bfloat16", device_map=device_map)

    imp_npz = np.load(pkg / "neuron_importance.npz")
    importance = {int(k.split("_")[1]): imp_npz[k] for k in imp_npz.files}
    plan = {int(k): v for k, v in cfg["compression_plan"].items()}
    loop_ctx = {"loop_idx": 0}

    compress_decoder_layers_inplace(
        harness,
        importance=importance,
        plan=plan,
        funnel_start=int(cfg["funnel_start"]),
        funnel_end=int(cfg["funnel_end"]),
        attach_loop_lora=True,
        max_loops=int(cfg["max_loops"]),
        lora_rank=int(cfg["lora_rank"]),
        lora_alpha=float(cfg["lora_alpha"]),
        loop_ctx=loop_ctx,
    )

    state_bundle = torch.load(pkg / "loop_compact_adapters.pt", map_location=harness.input_device, weights_only=True)
    b_sd = dict(state_bundle["bridge_state_dict"])
    if "r_ortho" not in b_sd and "r_ortho_base" in b_sd:
        b_sd["r_ortho"] = b_sd["r_ortho_base"].clone()
    p_fit = dict(state_bundle["procrustes_meta"])
    p_fit["r_0"] = b_sd["r_ortho_base"].cpu().numpy()

    bridge, halting = build_step_bridge_and_halting_modules(
        torch,
        procrustes_fit=p_fit,
        max_loops=int(cfg["max_loops"]),
        rank=int(cfg["bridge_rank"]),
        device=harness.input_device,
    )
    bridge.load_state_dict(b_sd)
    halting.load_state_dict(state_bundle["halting_state_dict"])

    for ell_str, sd in state_bundle["funnel_lora_state_dicts"].items():
        ell = int(ell_str)
        harness.layers[ell].load_state_dict(sd, strict=False)

    return Gemma4LoopCompactController(
        harness=harness,
        bridge=bridge,
        halting_head=halting,
        loop_ctx=loop_ctx,
        start_layer=int(cfg["funnel_start"]),
        end_layer=int(cfg["funnel_end"]),
        max_loops=int(cfg["max_loops"]),
    )
'''


def save_and_verify_loop_compact_package(
    controller: Gemma4LoopCompactController,
    out_pkg_dir: Path,
    base_model_id: str,
    compression_plan: dict[int, dict[str, int]],
    importance: dict[int, np.ndarray],
    procrustes_fit: dict[str, Any],
    lora_rank: int,
    lora_alpha: float,
    sample_prompt: str,
) -> dict[str, Any]:
    """Export standalone `gemma4_loop_compact/` package and verify state-dict round-trip equivalence (`0.00e+00` diff)."""
    harness = controller.harness
    torch = harness.torch
    out_pkg_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "architecture": "Gemma4LoopCompactForCausalLM",
        "base_model_id": base_model_id,
        "n_layers": harness.n_layers,
        "funnel_start": controller.start_layer,
        "funnel_end": controller.end_layer,
        "max_loops": controller.max_loops,
        "bridge_rank": controller.bridge.rank,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "compression_plan": {str(k): v for k, v in compression_plan.items()},
    }
    (out_pkg_dir / "config_loop_compact.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_pkg_dir / "modeling_gemma4_loop_compact.py").write_text(STANDALONE_HF_LOADER_SOURCE, encoding="utf-8")
    np.savez_compressed(
        out_pkg_dir / "neuron_importance.npz",
        **{f"layer_{ell}": vec.astype(np.float16) for ell, vec in importance.items()},
    )

    funnel_lora_sds: dict[str, dict[str, Any]] = {}
    for ell in range(controller.start_layer, controller.end_layer + 1):
        sd_l = {
            k: v.detach().cpu()
            for k, v in harness.layers[ell].state_dict().items()
            if "lora_" in k
        }
        funnel_lora_sds[str(ell)] = sd_l

    bridge_sd = {
        k: v.detach().cpu()
        for k, v in controller.bridge.state_dict().items()
        if k != "r_ortho"
    }
    clean_meta: dict[str, Any] = {}
    for k, v in procrustes_fit.items():
        if k == "r_0":
            continue
        if isinstance(v, np.ndarray):
            clean_meta[k] = torch.tensor(v, dtype=torch.float32)
        elif isinstance(v, (np.floating, float)):
            clean_meta[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            clean_meta[k] = int(v)
        else:
            clean_meta[k] = v

    state_bundle = {
        "procrustes_meta": clean_meta,
        "bridge_state_dict": bridge_sd,
        "halting_state_dict": {k: v.detach().cpu() for k, v in controller.halting_head.state_dict().items()},
        "funnel_lora_state_dicts": funnel_lora_sds,
    }
    adapter_path = out_pkg_dir / "loop_compact_adapters.pt"
    torch.save(state_bundle, adapter_path)

    # Round-trip state-dict reload equivalence check on live `K = 3` forward pass
    enc = harness._tokenize_rendered_batch([format_chat_prompt(harness.tokenizer, sample_prompt)])
    with torch.no_grad():
        with controller.active_recurrence(n_loops=3, adaptive=False):
            logits_before = harness.model(**enc, use_cache=False).logits[:, -1, :].to(torch.float32).clone()

        reloaded = torch.load(adapter_path, map_location=harness.input_device, weights_only=True)
        rb_sd = dict(reloaded["bridge_state_dict"])
        rb_sd["r_ortho"] = rb_sd["r_ortho_base"].clone()
        controller.bridge.load_state_dict(rb_sd)
        controller.halting_head.load_state_dict(reloaded["halting_state_dict"])
        for ell_str, sd_l in reloaded["funnel_lora_state_dicts"].items():
            harness.layers[int(ell_str)].load_state_dict(
                {k: v.to(harness.input_device) for k, v in sd_l.items()},
                strict=False,
            )

        with controller.active_recurrence(n_loops=3, adaptive=False):
            logits_after = harness.model(**enc, use_cache=False).logits[:, -1, :].to(torch.float32)

    max_diff = float(torch.max(torch.abs(logits_before - logits_after)).item())
    pkg_bytes = sum(f.stat().st_size for f in out_pkg_dir.glob("*") if f.is_file())
    return {
        "package_dir": str(out_pkg_dir),
        "package_size_mb": round(pkg_bytes / (1024 * 1024), 2),
        "reload_k3_max_logit_diff": max_diff,
        "files": sorted(f.name for f in out_pkg_dir.glob("*") if f.is_file()),
    }


# ---------------------------------------------------------------------------
# 6. Full End-to-End Orchestrator (`Actions 1 -> 2 -> 3 -> 4`)
# ---------------------------------------------------------------------------

def collect_paired_states_for_procrustes(
    harness: TransformerHarness,
    prompts: Sequence[str],
    start_layer: int,
    end_layer: int,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    torch = harness.torch
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))
    s_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i : i + batch_size])
        enc = tok(chunk, add_special_tokens=not has_chat_tmpl, return_tensors="pt", padding=True, truncation=True, max_length=192)
        inp = enc["input_ids"].to(harness.input_device)
        msk = enc["attention_mask"].to(harness.input_device)
        box: dict[str, Any] = {}

        def _pre_s(_m: Any, args: Any, kwargs: Any):
            t = args[0] if args else kwargs["hidden_states"]
            box["s"] = t.detach().to(torch.float32).cpu()

        def _post_e(_m: Any, _a: Any, out: Any):
            t = out[0] if isinstance(out, tuple) else out
            box["e"] = t.detach().to(torch.float32).cpu()

        h1 = harness.layers[start_layer].register_forward_pre_hook(_pre_s, with_kwargs=True)
        h2 = harness.layers[end_layer].register_forward_hook(_post_e)
        try:
            with torch.no_grad():
                harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
        finally:
            h1.remove()
            h2.remove()

        valid = msk.cpu().numpy().astype(bool)
        sn = box["s"].numpy()[valid]
        en = box["e"].numpy()[valid]
        if sn.shape[0] > 64:
            idx = np.linspace(0, sn.shape[0] - 1, 64, dtype=int)
            sn, en = sn[idx], en[idx]
        s_list.append(sn)
        e_list.append(en)
    return np.concatenate(s_list, axis=0), np.concatenate(e_list, axis=0)


def run_loop_compact_pipeline(
    model_name: str = "google/gemma-4-31B-it",
    model_label: str | None = None,
    out_dir: str | Path = "runs/loop_compact",
    distill_steps: int = 520,
    batch_size: int = 16,
    max_loops: int = 4,
    lora_rank: int = 16,
    save_artifact_package: bool = True,
    no_plots: bool = False,
) -> dict[str, Any]:
    """Execute Actions 1-4 end-to-end on `google/gemma-4-31B-it`."""
    t_start = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    aud = audit_loop_compact_suite()
    if not aud["valid"]:
        raise RuntimeError(f"Disjoint benchmark audit failed: {aud}")

    ds = build_loop_compact_datasets()
    resolved_id = resolve_model_name(model_name)
    display_label = model_label or resolved_id

    print(f"[Stage 0] Loading intact BF16 {display_label} for baseline, safety basis & <thought> teacher caching...", flush=True)
    harness = TransformerHarness.from_pretrained(resolved_id, dtype="bfloat16", device_map="cuda", model_id_label=display_label)
    torch = harness.torch
    n_layers = harness.n_layers
    d_ff = int(harness.layers[0].mlp.down_proj.weight.shape[1])
    d_model = int(harness.layers[0].mlp.down_proj.weight.shape[0])
    f_start = max(0, int(round(0.40 * n_layers)))  # L24
    f_end = min(n_layers - 1, int(round(0.70 * n_layers)))  # L42

    bf16_storage = compute_model_storage_bytes(harness.model)
    if torch.cuda.is_available():
        bf16_storage["cuda_allocated_gb"] = round(float(torch.cuda.memory_allocated()) / 1e9, 3)
    print(f"  -> Intact BF16 storage: {bf16_storage['total_gb']} GB (CUDA={bf16_storage.get('cuda_allocated_gb')} GB)", flush=True)

    # Cache Base KL references (eval + distill anchor)
    eval_kl_refs = precompute_base_kl_references(harness, HELD_OUT_KL_PASSAGES, top_m=256, batch_size=2)
    anchor_kl_refs = precompute_base_kl_references(harness, DISTILL_ANCHOR_PASSAGES, top_m=128, batch_size=2)

    print("[Stage 0] Evaluating Intact BF16 1-Pass baseline across all held-out suites...", flush=True)
    bf16_eval = {
        "storage": bf16_storage,
        "kl_eval": evaluate_kl_against_base(harness, eval_kl_refs),
        "multihop": evaluate_reasoning_suite(harness, ds["eval_multihop"], batch_size=batch_size),
        "wordprob": evaluate_reasoning_suite(harness, ds["eval_wordprob"], batch_size=batch_size),
        "balsheet": evaluate_reasoning_suite(harness, ds["eval_balsheet"], batch_size=batch_size),
        "easy_1hop": evaluate_reasoning_suite(harness, ds["eval_easy"], batch_size=batch_size),
        "safety": evaluate_moral_bias_and_sanity(harness, batch_size=batch_size),
    }
    print(
        f"  -> Intact BF16 (62.6 GB) 1-Pass: MH={bf16_eval['multihop']['overall_acc']*100:.1f}% | "
        f"WP={bf16_eval['wordprob']['overall_acc']*100:.1f}% | BS={bf16_eval['balsheet']['overall_acc']*100:.1f}%",
        flush=True,
    )

    print("[Stage 0] Extracting non-financial 3D moral subspace {spite, contamination, predation} & neuron importance...", flush=True)
    locked_moral, neutral_means = extract_safety_moral_basis(harness, batch_size=batch_size)
    calib_prompts = [format_chat_prompt(harness.tokenizer, it.prompt_user) for it in ds["train_all"][::16]] + [
        format_chat_prompt(harness.tokenizer, "Technical note:", assistant_prefill=p) for p in DISTILL_ANCHOR_PASSAGES
    ]
    importance = score_mlp_neuron_importance(harness, calib_prompts, batch_size=4)

    print(f"[Stage 0] Caching <thought>-Teacher soft logits & L{f_end} latent states across {len(ds['train_all'])} items...", flush=True)
    t_teach0 = time.perf_counter()
    teacher_targets = cache_thought_teacher_targets(harness, ds["train_all"], end_layer=f_end, top_m=64, batch_size=8)
    print(f"  -> Cached {len(teacher_targets)} <thought> teacher targets in {time.perf_counter() - t_teach0:.1f}s", flush=True)

    # -----------------------------------------------------------------------
    # Action 1A: Byte-Matched Control (`Uniform-Compact`)
    # -----------------------------------------------------------------------
    print("[Action 1] Building byte-matched Control: Uniform-Compact (flat pruning + uniform INT4/INT8)...", flush=True)
    uni_plan = build_compression_plan(n_layers, d_ff, f_start, f_end, mode="uniform")
    uni_comp_info = compress_decoder_layers_inplace(
        harness, importance, uni_plan, f_start, f_end, attach_loop_lora=False
    )
    uni_eval = {
        "storage": uni_comp_info["storage"],
        "kl_eval": evaluate_kl_against_base(harness, eval_kl_refs),
        "multihop": evaluate_reasoning_suite(harness, ds["eval_multihop"], batch_size=batch_size),
        "wordprob": evaluate_reasoning_suite(harness, ds["eval_wordprob"], batch_size=batch_size),
        "balsheet": evaluate_reasoning_suite(harness, ds["eval_balsheet"], batch_size=batch_size),
    }
    print(
        f"  -> Uniform-Compact ({uni_eval['storage']['total_gb']} GB, CUDA={uni_eval['storage'].get('cuda_allocated_gb')} GB): "
        f"KL={uni_eval['kl_eval']['kl_nats']:.4f} nats | MH={uni_eval['multihop']['overall_acc']*100:.1f}% | "
        f"BS={uni_eval['balsheet']['overall_acc']*100:.1f}%",
        flush=True,
    )

    # Reload clean BF16 from RAM disk (`/dev/shm`) in ~17s and build Hourglass-Compact with Loop-Indexed LoRA
    del harness
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[Action 1] Reloading from RAM disk & building byte-matched Hourglass-Compact (0-shot)...", flush=True)
    harness = TransformerHarness.from_pretrained(resolved_id, dtype="bfloat16", device_map="cuda", model_id_label=display_label)
    hg_plan = build_compression_plan(n_layers, d_ff, f_start, f_end, mode="hourglass")
    loop_ctx = {"loop_idx": 0}
    hg_comp_info = compress_decoder_layers_inplace(
        harness,
        importance=importance,
        plan=hg_plan,
        funnel_start=f_start,
        funnel_end=f_end,
        attach_loop_lora=False,
        loop_ctx=loop_ctx,
    )

    hg_zeroshot_eval = {
        "storage": hg_comp_info["storage"],
        "kl_eval": evaluate_kl_against_base(harness, eval_kl_refs),
        "multihop": evaluate_reasoning_suite(harness, ds["eval_multihop"], batch_size=batch_size),
        "wordprob": evaluate_reasoning_suite(harness, ds["eval_wordprob"], batch_size=batch_size),
        "balsheet": evaluate_reasoning_suite(harness, ds["eval_balsheet"], batch_size=batch_size),
    }
    print(
        f"  -> Hourglass-Compact ({hg_zeroshot_eval['storage']['total_gb']} GB, CUDA={hg_zeroshot_eval['storage'].get('cuda_allocated_gb')} GB): "
        f"KL={hg_zeroshot_eval['kl_eval']['kl_nats']:.4f} nats | MH={hg_zeroshot_eval['multihop']['overall_acc']*100:.1f}% | "
        f"BS={hg_zeroshot_eval['balsheet']['overall_acc']*100:.1f}%",
        flush=True,
    )

    print(f"[Action 2] Attaching FP32-master Loop-Indexed LoRA (r={lora_rank}, K=1..{max_loops}) to Funnel L{f_start}..L{f_end}...", flush=True)
    for ell in range(f_start, f_end + 1):
        for _mn, sub_mod in harness.layers[ell].named_modules():
            if hasattr(sub_mod, "init_loop_lora"):
                sub_mod.loop_ctx = loop_ctx
                sub_mod.init_loop_lora(max_loops=max_loops, lora_rank=lora_rank, lora_alpha=16.0)

    # -----------------------------------------------------------------------
    # Action 2: Solve Closed-Form SVD Orthogonal Procrustes on Hourglass-Compact
    # -----------------------------------------------------------------------
    print(f"[Action 2] Solving {d_model}x{d_model} SVD Orthogonal Procrustes (L{f_end} -> L{f_start}) on Hourglass-Compact...", flush=True)
    h_s_np, h_e_np = collect_paired_states_for_procrustes(harness, calib_prompts, f_start, f_end, batch_size=4)
    proc_fit = solve_orthogonal_procrustes(h_src=h_e_np, h_dst=h_s_np)
    print(
        f"  -> Procrustes cosine: {proc_fit['mean_cosine_identity']:+.4f} -> {proc_fit['mean_cosine_procrustes']:+.4f} "
        f"(rel_err {proc_fit['rel_err_identity']:.4f} -> {proc_fit['rel_err_procrustes']:.4f})",
        flush=True,
    )

    bridge, halting_head = build_step_bridge_and_halting_modules(
        torch, procrustes_fit=proc_fit, max_loops=max_loops, rank=128, init_gate=0.14, device=harness.input_device
    )
    controller = Gemma4LoopCompactController(
        harness=harness,
        bridge=bridge,
        halting_head=halting_head,
        loop_ctx=loop_ctx,
        start_layer=f_start,
        end_layer=f_end,
        max_loops=max_loops,
    )

    # Configure non-financial 3D moral subspace `{spite, contamination, predation}` at `L42` and across `L24..L42`
    dirs_3d = {
        ell: np.stack(
            [
                locked_moral.controlled_dirs["spite"][ell],
                locked_moral.controlled_dirs["contamination"][ell],
                locked_moral.controlled_dirs["predation"][ell],
            ],
            axis=0,
        )
        for ell in range(n_layers)
    }
    q42_np, _ = harness.build_subspace_clamp_matrices(dirs_3d[f_end], neutral_means.get(f_end))
    controller.bridge.set_safety_subspace(q42_np, active=False)
    controller.set_funnel_safety_clamp(dirs_3d, neutral_means, active=False)

    # -----------------------------------------------------------------------
    # Action 3: Randomized Unroll (`K ~ {1,2,3,4}`) + `<thought>` Self-Distillation
    # -----------------------------------------------------------------------
    print(f"[Action 3] Distilling Gemma-4-Loop-Compact ({distill_steps} steps, K in 1..{max_loops}, <thought> teacher + KL anchor)...", flush=True)
    train_summary = train_loop_compact_randomized(
        controller=controller,
        train_items=ds["train_all"],
        teacher_targets=teacher_targets,
        anchor_refs=anchor_kl_refs,
        n_steps=distill_steps,
        batch_size=4,
    )

    # Cast trained FP32 master LoRAs down to BF16 for compact inference & export!
    for ell in range(f_start, f_end + 1):
        for _mn, sub_mod in harness.layers[ell].named_modules():
            if hasattr(sub_mod, "cast_lora_dtype"):
                sub_mod.cast_lora_dtype(torch.bfloat16)

    # -----------------------------------------------------------------------
    # Action 4A: Fit Adaptive Halting Gate at `L42`
    # -----------------------------------------------------------------------
    print("[Action 4] Fitting Adaptive L42 Halting Gate on calibration easy vs. hard reasoning states...", flush=True)
    calib_easy_tr = [_generate_easy_1hop_item(random.Random(7700 + i), f"cal_easy_{i}", odd_first=False) for i in range(48)]
    calib_hard_tr = [x for x in ds["train_all"] if x.hops >= 3][:48]
    halting_fit = fit_adaptive_halting_head(controller, calib_easy=calib_easy_tr, calib_hard=calib_hard_tr)
    print(f"  -> Halting Gate fit: BCE={halting_fit['final_bce_loss']:.4f} | Gate Acc={halting_fit['calib_gate_acc']*100:.1f}%", flush=True)

    # -----------------------------------------------------------------------
    # Comprehensive Evaluation Across K = 1, 2, 3, 4 + Adaptive-K + Safety Guard
    # -----------------------------------------------------------------------
    loop_evals: dict[str, Any] = {}
    for k_val in range(1, max_loops + 1):
        print(f"[Eval] Scoring Gemma-4-Loop-Compact at Fixed K = {k_val} loop(s)...", flush=True)
        with controller.active_recurrence(n_loops=k_val, adaptive=False):
            kl_k = evaluate_kl_against_base(harness, eval_kl_refs)
        mh_k = evaluate_reasoning_suite(harness, ds["eval_multihop"], controller=controller, n_loops=k_val, batch_size=batch_size)
        wp_k = evaluate_reasoning_suite(harness, ds["eval_wordprob"], controller=controller, n_loops=k_val, batch_size=batch_size)
        bs_k = evaluate_reasoning_suite(harness, ds["eval_balsheet"], controller=controller, n_loops=k_val, batch_size=batch_size)
        ez_k = evaluate_reasoning_suite(harness, ds["eval_easy"], controller=controller, n_loops=k_val, batch_size=batch_size)
        sf_k = evaluate_moral_bias_and_sanity(harness, controller=controller, n_loops=k_val, batch_size=batch_size)
        loop_evals[f"loop_k{k_val}"] = {
            "n_loops": k_val,
            "kl_eval": kl_k,
            "multihop": mh_k,
            "wordprob": wp_k,
            "balsheet": bs_k,
            "easy_1hop": ez_k,
            "safety": sf_k,
        }
        print(
            f"  -> K={k_val}: KL={kl_k['kl_nats']:.4f} | MH={mh_k['overall_acc']*100:.1f}% "
            f"(7h={mh_k['by_hops']['hop_7']['acc']*100:.1f}%, 9h={mh_k['by_hops']['hop_9']['acc']*100:.1f}%) | "
            f"WP={wp_k['overall_acc']*100:.1f}% | BS={bs_k['overall_acc']*100:.1f}%",
            flush=True,
        )

    print("[Eval Action 4] Scoring Adaptive Halting Gate (`mode=adaptive`) across Easy vs. Hard suites...", flush=True)
    adap_eval = {
        "easy_1hop": evaluate_reasoning_suite(harness, ds["eval_easy"], controller=controller, adaptive=True, halt_threshold=0.65),
        "multihop": evaluate_reasoning_suite(harness, ds["eval_multihop"], controller=controller, adaptive=True, halt_threshold=0.65),
        "wordprob": evaluate_reasoning_suite(harness, ds["eval_wordprob"], controller=controller, adaptive=True, halt_threshold=0.65),
        "balsheet": evaluate_reasoning_suite(harness, ds["eval_balsheet"], controller=controller, adaptive=True, halt_threshold=0.65),
    }
    print(
        f"  -> Adaptive Gate: Easy E[K]={adap_eval['easy_1hop']['mean_loops']:.2f} ({adap_eval['easy_1hop']['overall_acc']*100:.1f}% acc, {adap_eval['easy_1hop']['ms_per_item']:.1f}ms) | "
        f"MultiHop E[K]={adap_eval['multihop']['mean_loops']:.2f} ({adap_eval['multihop']['overall_acc']*100:.1f}% acc, {adap_eval['multihop']['ms_per_item']:.1f}ms)",
        flush=True,
    )

    print("[Eval Action 4] Testing Zero-Cost Bridge Safety Fold + Funnel Subspace Guard at K = 3...", flush=True)
    # 1. Zero-cost Bridge-Only safety fold (`R_safe = R_proc @ (I - Q_42 Q_42^T)`)
    controller.bridge.apply_safety_fold(active=True)
    safety_bridge_only = {
        "safety": evaluate_moral_bias_and_sanity(harness, controller=controller, n_loops=3, batch_size=batch_size),
        "multihop": evaluate_reasoning_suite(harness, ds["eval_multihop"], controller=controller, n_loops=3, batch_size=batch_size),
        "balsheet": evaluate_reasoning_suite(harness, ds["eval_balsheet"], controller=controller, n_loops=3, batch_size=batch_size),
    }
    # 2. Combined Bridge Safety Fold + Rotating `L24..L42` Funnel Subspace Guard
    controller.set_funnel_safety_clamp(dirs_3d, neutral_means, active=True)
    safety_full_funnel = {
        "safety": evaluate_moral_bias_and_sanity(harness, controller=controller, n_loops=3, batch_size=batch_size),
        "multihop": evaluate_reasoning_suite(harness, ds["eval_multihop"], controller=controller, n_loops=3, batch_size=batch_size),
        "balsheet": evaluate_reasoning_suite(harness, ds["eval_balsheet"], controller=controller, n_loops=3, batch_size=batch_size),
    }
    controller.set_funnel_safety_clamp(dirs_3d, neutral_means, active=False)
    controller.bridge.apply_safety_fold(active=False)

    final_storage = compute_model_storage_bytes(harness.model)
    final_storage["bridge_mb"] = round(sum(p.numel() * p.element_size() for p in controller.bridge.parameters()) / 1e6, 2)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        final_storage["cuda_allocated_gb"] = round(float(torch.cuda.memory_allocated()) / 1e9, 3)

    pkg_verification: dict[str, Any] = {}
    if save_artifact_package:
        print("[Action 2] Exporting standalone `gemma4_loop_compact/` HF package & verifying round-trip reload...", flush=True)
        pkg_verification = save_and_verify_loop_compact_package(
            controller=controller,
            out_pkg_dir=out_path / "gemma4_loop_compact_pkg",
            base_model_id=display_label,
            compression_plan=hg_plan,
            importance=importance,
            procrustes_fit=proc_fit,
            lora_rank=lora_rank,
            lora_alpha=16.0,
            sample_prompt=ds["eval_multihop"][0].prompt_user,
        )
        print(
            f"  -> Exported {pkg_verification['package_size_mb']} MB package | "
            f"Round-trip K=3 max |delta logit| = {pkg_verification['reload_k3_max_logit_diff']:.2e}",
            flush=True,
        )

    proc_clean = {
        k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
        for k, v in proc_fit.items()
        if not isinstance(v, np.ndarray)
    }

    results = {
        "metadata": {
            "model_id": display_label,
            "n_layers": n_layers,
            "d_model": d_model,
            "d_ff": d_ff,
            "funnel_band": [f_start, f_end],
            "max_loops": max_loops,
            "lora_rank": lora_rank,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "audit": aud,
        },
        "action1_compression": {
            "intact_bf16": bf16_eval,
            "uniform_compact": uni_eval,
            "hourglass_compact_zeroshot": hg_zeroshot_eval,
            "final_loop_compact_storage": final_storage,
        },
        "action2_architecture": {
            "procrustes_alignment": proc_clean,
            "package_verification": pkg_verification,
        },
        "action3_distillation": {
            "training_summary": train_summary,
            "loop_scaling": loop_evals,
        },
        "action4_halting_and_safety": {
            "halting_calibration": halting_fit,
            "adaptive_eval": adap_eval,
            "safety_unclamped_k3": {
                "safety": loop_evals["loop_k3"]["safety"],
                "multihop": loop_evals["loop_k3"]["multihop"],
                "balsheet": loop_evals["loop_k3"]["balsheet"],
            },
            "safety_bridge_fold_k3": safety_bridge_only,
            "safety_full_funnel_k3": safety_full_funnel,
        },
        "runtime_seconds": round(time.perf_counter() - t_start, 2),
        "completed": True,
    }

    res_file = out_path / "loop_compact_results.json"
    res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[LoopCompact] Saved {res_file} ({results['runtime_seconds']}s)", flush=True)

    if not no_plots:
        try:
            from .plot_loop_compact import render_loop_compact_bundle
            render_loop_compact_bundle(results, out_path)
        except ImportError:
            print("[LoopCompact] Skipping plot render on slim GPU image (no matplotlib); sync_afriat_runs.sh renders locally.", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9: Gemma-4-Loop-Compact (Actions 1-4)")
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs/loop_compact")
    parser.add_argument("--distill-steps", type=int, default=520)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-loops", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--no-save-pkg", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--render-dir", type=str, default=None)
    args = parser.parse_args()

    if args.render_dir:
        rdir = Path(args.render_dir)
        data = json.loads((rdir / "loop_compact_results.json").read_text(encoding="utf-8"))
        from .plot_loop_compact import render_loop_compact_bundle
        render_loop_compact_bundle(data, rdir)
        return

    run_loop_compact_pipeline(
        model_name=args.model,
        model_label=args.model_label,
        out_dir=args.out_dir,
        distill_steps=args.distill_steps,
        batch_size=args.batch_size,
        max_loops=args.max_loops,
        lora_rank=args.lora_rank,
        save_artifact_package=not args.no_save_pkg,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
