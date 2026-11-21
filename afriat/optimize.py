"""Phase 8: Funnel-Guided Model Optimization on `google/gemma-4-31B-it`.

Executes two capability & efficiency experiments motivated directly by the Phase 7
Read-Divergent / Write-Convergent Funnel (`L24..L42`):

  Experiment A (Zero-Training Inference Efficiency):
    **Hourglass Activation Sparsity vs. Goodfire Uniform Top-K Sparsity**
    Tests matched global active-neuron budgets (`15%` and `10%` of `1,290,240` MLP neurons)
    comparing Uniform Top-K across all 60 layers against Funnel-Guided Hourglass Top-K
    (aggressive pruning in Stage 1 `L0..L23` and Stage 3 `L43..L59`, dense preservation
    of distributed tail writers in Stage 2 `L24..L42`), plus an Anti-Hourglass control.
    Evaluates:
      1) Held-out multi-domain KL divergence (nats) & perplexity,
      2) `no_think` (0-CoT) multi-hop state tracking & multi-step quantitative reasoning,
      3) `<thought>` multi-step accuracy & runaway reasoning-loop / cap-hit rate.

  Experiment B (Latent Depth Scaling):
    **Procrustes Mid-Stack Recurrent Looping (`L24..L42`)**
    Retrofits recurrent latent depth onto frozen `Gemma 4 31B` by looping only the 19-layer
    synthesis band (`L24..L42`) via an anchored Orthogonal Procrustes + Rank-128 re-entry
    bridge `H_24^(k+1) = (1 - g) * H_24^(0) + g * Bridge(H_42^(k))`:
      1) Verifies exact `0.000` logit equivalence at `g = 0` on live GPU layers,
      2) Fits closed-form full-rank `5376 x 5376` SVD Orthogonal Procrustes `R_0`,
      3) Trains the lightweight `1.4M`-param adapter + channel gate (`280` steps, frozen 31B)
         on `2..5`-hop compositions + multi-step word problems with a general-text KL anchor,
      4) Benchmarks held-out `3-hop`, `5-hop`, and unseen `7-hop` extrapolation + word problems
         against `1-pass no_think`, `Naive Unrotated Loop`, `Zero-Shot Procrustes Loop`, and
         full `<thought>` CoT ceiling (accuracy, gold log-prob, and wall-clock ms/problem).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Iterator, Sequence

import numpy as np

from afriat.models import TransformerHarness, format_chat_prompt, resolve_model_name
from afriat.thought_probe import (
    _maybe_empty_cuda_cache,
    format_chat_prompt_thinking,
    generate_thought_records,
)

RE_FIRST_INT = re.compile(r"[-+]?\d+")


# ---------------------------------------------------------------------------
# 1. Deterministic Disjoint Train / Held-Out Benchmark Generators
# ---------------------------------------------------------------------------

HELD_OUT_KL_PASSAGES: tuple[str, ...] = (
    "In double-entry accounting, every capital expenditure alters both asset composition and future depreciation schedules without immediately changing operating cash flow until amortization begins.",
    "When a transformer layer applies root-mean-square normalization before a gated linear unit, scaling the post-projection matrix shifts relative residual variance across depth.",
    "Thermodynamic efficiency in a Brayton cycle depends primarily on the compressor pressure ratio and the ratio of specific heats of the working gas under isentropic assumptions.",
    "def reconcile_ledger(balances, transfers):\n    state = dict(balances)\n    for src, dst, amt in transfers:\n        state[src] = state.get(src, 0) - amt\n        state[dst] = state.get(dst, 0) + amt\n    return state",
    "During cytokinesis in animal cells, an actomyosin contractile ring assembles beneath the equatorial plasma membrane and generates inward tension that cleaves the cytoplasm into two daughter cells.",
    "A sovereign yield curve inverts when short-term policy rates exceed long-horizon inflation and growth expectations, compressing net interest margins for maturity-transforming lenders.",
    "In distributed consensus protocols such as Raft, a leader appends client commands to its local log and commits an entry only after a strict majority of followers acknowledge replication.",
    "Singular value decomposition factors any real rectangular matrix into two orthogonal basis rotations separated by a nonnegative diagonal spectrum ordered from largest to smallest gain.",
    "Oceanic thermohaline circulation transports warm surface waters poleward before evaporative cooling and brine rejection increase density enough to drive deep convective sinking.",
    "class RingBuffer:\n    def __init__(self, cap):\n        self.buf = [None] * cap\n        self.head = 0\n        self.size = 0\n    def push(self, item):\n        idx = (self.head + self.size) % len(self.buf)\n        self.buf[idx] = item",
    "Under Bayesian decision theory, an agent updates prior beliefs via likelihood ratios and selects the action minimizing expected loss across the posterior distribution over latent states.",
    "Phase-locked loops synchronize a voltage-controlled oscillator to a reference signal by filtering phase-detector error pulses into a smooth control voltage with bounded loop bandwidth.",
    "In contract law, expectation damages aim to place the non-breaching counterparty in the exact net pecuniary position that full performance of the agreement would have achieved.",
    "Compiler register allocation maps an arbitrary number of SSA virtual temporaries onto a fixed architectural register file by coloring an interference graph and spilling high-degree nodes.",
    "Photosystem II oxidizes water molecules at a manganese-calcium cluster, releasing molecular oxygen and protons into the thylakoid lumen while feeding electrons into plastoquinone.",
    "Zero-knowledge succinct non-interactive arguments of knowledge allow a prover to convince a verifier that a committed witness satisfies an arithmetic circuit without revealing witness values.",
    "When viscosity dominates inertia at very low Reynolds numbers, fluid motion becomes time-reversible, forcing microorganisms to use non-reciprocal flagellar strokes to achieve net propulsion.",
    "SELECT dept_id, COUNT(*) AS n_staff, AVG(base_salary) AS mean_pay FROM employees WHERE active = 1 GROUP BY dept_id HAVING COUNT(*) >= 5 ORDER BY mean_pay DESC;",
    "Convex optimization guarantees that any local minimum of a differentiable convex objective over a closed convex feasible set is automatically a global minimum.",
    "Radiocarbon dating measures the residual ratio of carbon-14 to stable carbon-12 in organic remains, calibrated against dendrochronological rings to correct for atmospheric fluctuations.",
    "In modern NVMe storage engines, log-structured merge trees convert random writes into sequential memtable flushes followed by background leveled compaction to bound read amplification.",
    "Allosteric enzymes depart from hyperbolic Michaelis-Menten kinetics by exhibiting cooperative subunit transitions that produce sigmoidal velocity curves sensitive to metabolic regulators.",
    "Auction mechanisms such as Vickrey-Clarke-Groves charge each winning bidder the externality imposed on competing participants, making truthful valuation bidding a dominant strategy.",
    "Error-correcting Reed-Solomon codes view message symbols as coefficients of a finite-field polynomial and evaluate it across distinct points so burst erasures can be interpolated exactly.",
)

TRAIN_ANCHOR_PASSAGES: tuple[str, ...] = (
    "Linear operators between finite-dimensional inner product spaces admit adjoint matrices whose column spaces are orthogonal complements of the original nullspaces.",
    "Packet-switched networks mitigate congestion through additive-increase multiplicative-decrease window control paired with explicit queue marking at bottleneck routers.",
    "def prefix_sums(values):\n    out = []\n    acc = 0\n    for v in values:\n        acc += v\n        out.append(acc)\n    return out",
    "Enzymatic transition-state stabilization lowers activation free energy without altering the equilibrium constant between reactants and products.",
    "Portfolio variance depends not only on individual asset volatilities but on pairwise covariances that remain non-diversifiable during systemic liquidity shocks.",
    "In virtual memory subsystems, translation lookaside buffers cache page-table walks so virtual-to-physical address translation completes in a single clock cycle on hits.",
    "Fourier analysis decomposes periodic waveforms into orthogonal sinusoidal harmonics whose squared amplitudes sum to total signal energy by Parseval's identity.",
    "Database serializability ensures that concurrent interleaved transactions produce a final state equivalent to some sequential execution order without lost updates.",
    "Git content-addressable storage hashes blobs, directory trees, and parent commit pointers into a directed acyclic Merkle graph.",
    "Osmotic pressure across a semipermeable membrane scales proportionally with solute molarity and absolute temperature in dilute ideal solutions.",
    "Dynamic programming solves sequential decision problems with overlapping subproblems by memoizing Bellman value recurrences backward from terminal states.",
    "Lithium-ion intercalation anodes store charge by reversibly inserting ions between graphitic carbon planes separated by a solid-electrolyte interphase.",
    " Kalman filters fuse noisy linear state dynamics with Gaussian sensor observations through a two-step predict-and-update covariance recursion.",
    "Type inference in Hindley-Milner systems computes principal polymorphic types via syntactic unification of constraint equations.",
    "Geostrophic wind balances horizontal pressure-gradient forces against planetary Coriolis deflection parallel to isobars above the atmospheric boundary layer.",
    "Cryptographic hash functions satisfy collision resistance, preimage resistance, and avalanche diffusion across fixed-width digest outputs.",
)


@dataclass(frozen=True)
class MathChainItem:
    item_id: str
    suite: str       # "multihop" or "word_problem"
    hops: int
    prompt_user: str
    answer_int: int


def _generate_multihop_items(seed: int, n_per_hop: int, hop_counts: Sequence[int], prefix: str) -> list[MathChainItem]:
    rng = random.Random(seed)
    var_pools = [("A", "B", "C"), ("X", "Y", "Z"), ("K", "M", "N"), ("P", "Q", "R")]
    items: list[MathChainItem] = []

    for hops in hop_counts:
        for idx in range(n_per_hop):
            v1, v2, v3 = var_pools[(idx + hops) % len(var_pools)]
            state = {
                v1: rng.randint(4, 15),
                v2: rng.randint(3, 13),
                v3: rng.randint(2, 11),
            }
            lines = [f"Initial register values: {v1} = {state[v1]}, {v2} = {state[v2]}, {v3} = {state[v3]}."]
            vars_list = [v1, v2, v3]
            target = v1
            for step in range(1, hops + 1):
                dst = vars_list[(step + idx) % 3]
                src_a = vars_list[(step + idx + 1) % 3]
                src_b = vars_list[(step + idx + 2) % 3]
                op_kind = (step + idx * 2) % 4
                if op_kind == 0:
                    d = rng.randint(3, 12)
                    state[dst] = state[src_a] + d
                    lines.append(f"Step {step}: Set {dst} = {src_a} + {d}.")
                elif op_kind == 1:
                    d = rng.randint(2, 9)
                    state[dst] = state[src_a] + state[src_b] - d
                    lines.append(f"Step {step}: Set {dst} = {src_a} + {src_b} - {d}.")
                elif op_kind == 2:
                    d = rng.randint(2, 11)
                    state[dst] = 2 * state[src_b] - d
                    lines.append(f"Step {step}: Set {dst} = 2 * {src_b} - {d}.")
                else:
                    d = rng.randint(1, 8)
                    state[dst] = state[dst] + state[src_a] - d
                    lines.append(f"Step {step}: Set {dst} = {dst} + {src_a} - {d}.")
                target = dst

            q_var = target
            prompt = (
                "Track the exact integer register updates step by step:\n"
                + "\n".join(lines)
                + f"\n\nWhat is the final integer value of {q_var} after Step {hops}? "
                "End your response with FINAL_INTEGER: followed by only the integer."
            )
            items.append(
                MathChainItem(
                    item_id=f"{prefix}_hop{hops}_{idx:03d}",
                    suite="multihop",
                    hops=hops,
                    prompt_user=prompt,
                    answer_int=int(state[q_var]),
                )
            )
    return items


def _generate_word_problems(seed: int, count: int, prefix: str) -> list[MathChainItem]:
    rng = random.Random(seed)
    entities = [
        ("warehouse", "crates", "pallets"),
        ("treasury desk", "lots", "tranches"),
        ("foundry", "ingots", "batches"),
        ("laboratory", "vials", "racks"),
        ("datacenter", "blades", "racks"),
        ("orchard", "baskets", "bins"),
    ]
    items: list[MathChainItem] = []
    for i in range(count):
        place, unit, group = entities[i % len(entities)]
        tpl = i % 4
        if tpl == 0:
            n_grp = rng.randint(6, 16)
            per_grp = rng.randint(12, 28)
            added = rng.randint(15, 55)
            defect_div = rng.choice([3, 4, 5])
            total0 = n_grp * per_grp + added
            # Make total0 cleanly divisible by defect_div
            rem = total0 % defect_div
            if rem != 0:
                added += defect_div - rem
                total0 = n_grp * per_grp + added
            removed = total0 // defect_div
            shipped_sets = rng.randint(2, 5)
            per_set = rng.randint(9, 22)
            ans = total0 - removed - shipped_sets * per_set
            text = (
                f"A {place} starts Monday with {n_grp} {group} holding {per_grp} {unit} each, plus {added} loose {unit}. "
                f"During inspection, exactly 1/{defect_div} of all {unit} on hand are retired. "
                f"Later that afternoon, {shipped_sets} outbound orders of {per_set} {unit} each are dispatched. "
                f"How many {unit} remain at the {place}?"
            )
        elif tpl == 1:
            base_rev = rng.randint(18, 45) * 10
            mult = rng.randint(2, 4)
            bonus = rng.randint(25, 85)
            day2 = base_rev * mult - bonus
            fee_pct = rng.choice([10, 20, 25])
            gross = base_rev + day2
            # Adjust bonus so gross is cleanly divisible by 100 // gcd
            step_mod = 100 // math.gcd(100, fee_pct)
            if gross % step_mod != 0:
                adj = step_mod - (gross % step_mod)
                day2 += adj
                bonus -= adj
                gross = base_rev + day2
            net = gross - (gross * fee_pct) // 100
            fixed_cost = rng.randint(40, 130)
            ans = net - fixed_cost
            text = (
                f"A {place} clears ${base_rev} on Tuesday. On Wednesday it clears {mult} times Tuesday's amount minus ${bonus}. "
                f"A clearing fee removes {fee_pct}% of the combined two-day gross, and a fixed ${fixed_cost} settlement charge is deducted afterward. "
                f"What is the final net dollar amount?"
            )
        elif tpl == 2:
            a = rng.randint(24, 64)
            diff = rng.randint(7, 23)
            b = a - diff
            c = 2 * b + rng.randint(5, 19)
            transfer = rng.randint(6, 18)
            a2 = a - transfer
            c2 = c + transfer
            ans = c2 - a2 + b
            text = (
                f"Bin Alpha holds {a} {unit}, Bin Beta holds {diff} fewer {unit} than Alpha, and Bin Gamma holds {c - 2 * b} more than twice Beta. "
                f"Operators move {transfer} {unit} from Alpha into Gamma. "
                f"After the move, what is (Gamma minus Alpha) plus Beta?"
            )
        else:
            rate = rng.randint(14, 32)
            hrs1 = rng.randint(3, 6)
            rate2 = rate + rng.randint(5, 13)
            hrs2 = rng.randint(2, 5)
            prod = rate * hrs1 + rate2 * hrs2
            pack_sz = rng.choice([6, 8, 10])
            extra = prod % pack_sz
            full_packs = prod // pack_sz
            price_pack = rng.randint(7, 15)
            ans = full_packs * price_pack - extra * 2
            text = (
                f"A {place} produces {rate} {unit} per hour for {hrs1} hours, then upgrades speed to {rate2} {unit} per hour for the next {hrs2} hours. "
                f"All output is packed into full {group} of {pack_sz} {unit} each, leaving any remainder unpacked. "
                f"Each full box sells for ${price_pack}, while each unpacked leftover unit incurs a $2 storage fee. "
                f"What is the net revenue in dollars?"
            )

        prompt = f"{text}\n\nEnd your response with FINAL_INTEGER: followed by only the integer."
        items.append(
            MathChainItem(
                item_id=f"{prefix}_wp_{i:03d}",
                suite="word_problem",
                hops=4,
                prompt_user=prompt,
                answer_int=int(ans),
            )
        )
    return items


def build_phase8_benchmarks() -> dict[str, Any]:
    train_mh = _generate_multihop_items(seed=20260801, n_per_hop=64, hop_counts=(2, 3, 4, 5), prefix="train")
    train_wp = _generate_word_problems(seed=20260802, count=128, prefix="train")
    eval_mh = _generate_multihop_items(seed=20260901, n_per_hop=32, hop_counts=(3, 5, 7), prefix="eval")
    eval_wp = _generate_word_problems(seed=20260902, count=48, prefix="eval")
    return {
        "train_items": train_mh + train_wp,
        "eval_multihop": eval_mh,
        "eval_word_problems": eval_wp,
        "eval_kl_passages": list(HELD_OUT_KL_PASSAGES),
        "train_anchor_passages": list(TRAIN_ANCHOR_PASSAGES),
    }


def audit_optimize_suite() -> dict[str, Any]:
    bm = build_phase8_benchmarks()
    train_prompts = {x.prompt_user for x in bm["train_items"]}
    eval_prompts = {x.prompt_user for x in bm["eval_multihop"] + bm["eval_word_problems"]}
    overlap = len(train_prompts & eval_prompts)
    kl_overlap = len(set(bm["eval_kl_passages"]) & set(bm["train_anchor_passages"]))
    valid = (overlap == 0) and (kl_overlap == 0) and len(bm["eval_multihop"]) == 96 and len(bm["eval_word_problems"]) == 48
    return {
        "valid": valid,
        "n_train_items": len(bm["train_items"]),
        "n_eval_multihop": len(bm["eval_multihop"]),
        "n_eval_word_problems": len(bm["eval_word_problems"]),
        "n_eval_kl_passages": len(bm["eval_kl_passages"]),
        "train_eval_prompt_overlap": overlap,
        "kl_passage_overlap": kl_overlap,
    }


# ---------------------------------------------------------------------------
# 2. Experiment A: Exact Budget-Matched Hourglass vs. Uniform Top-K Sparsity
# ---------------------------------------------------------------------------

def compute_sparsity_schedules(
    n_layers: int,
    d_ff: int,
    funnel_start: int,
    funnel_end: int,
) -> dict[str, dict[int, int]]:
    """Construct exact global-budget-matched per-layer Top-K neuron schedules (`k` out of `d_ff`).

    Protects the 2 embedding-lift & 2 unembedding-prep boundary layers alongside the `L24..L42`
    funnel band while pruning the redundant Stage-1 / Stage-3 corridors (`L2..L23` and `L43..L57`).
    """
    funnel_set = set(range(funnel_start, funnel_end + 1))
    boundary_set = ({0, 1, n_layers - 2, n_layers - 1} - funnel_set) if n_layers >= 10 else set()
    corridor_set = set(range(n_layers)) - funnel_set - boundary_set
    total_neurons = n_layers * d_ff

    def _make_schedule(target_frac: float, corridor_frac: float, boundary_frac: float | None = None) -> dict[int, int]:
        target_total = int(round(target_frac * total_neurons))
        k_corr = max(1, min(d_ff, int(round(corridor_frac * d_ff))))
        b_frac = corridor_frac if boundary_frac is None else boundary_frac
        k_bound = max(1, min(d_ff, int(round(b_frac * d_ff))))
        used = len(corridor_set) * k_corr + len(boundary_set) * k_bound
        rem_for_funnel = target_total - used
        k_funnel = max(1, min(d_ff, int(round(rem_for_funnel / max(1, len(funnel_set))))))
        out: dict[int, int] = {}
        for ell in range(n_layers):
            if ell in funnel_set:
                out[ell] = k_funnel
            elif ell in boundary_set:
                out[ell] = k_bound
            else:
                out[ell] = k_corr
        return out

    k_u15 = max(1, int(round(0.15 * d_ff)))
    k_u10 = max(1, int(round(0.10 * d_ff)))

    return {
        "uniform_15pct": {ell: k_u15 for ell in range(n_layers)},
        "hourglass_15pct": _make_schedule(target_frac=0.15, corridor_frac=0.075, boundary_frac=0.22),
        "anti_hourglass_15pct": _make_schedule(target_frac=0.15, corridor_frac=0.195, boundary_frac=0.195),
        "uniform_10pct": {ell: k_u10 for ell in range(n_layers)},
        "hourglass_10pct": _make_schedule(target_frac=0.10, corridor_frac=0.048, boundary_frac=0.16),
    }


@contextmanager
def layerwise_topk_mlp_sparsity(harness: TransformerHarness, k_by_layer: dict[int, int]) -> Iterator[None]:
    """Zero all but the top-K `|activation|` MLP neurons per token before `down_proj` (Goodfire protocol)."""
    torch = harness.torch
    handles = []

    for ell, k_keep in k_by_layer.items():
        if ell < 0 or ell >= harness.n_layers:
            continue
        down_proj = harness.layers[ell].mlp.down_proj
        d_ff = int(down_proj.weight.shape[1])
        if k_keep >= d_ff:
            continue

        def _make_pre_hook(k_val: int):
            def _hook(_mod: Any, args: Any):
                x = args[0]  # [..., d_ff]
                if k_val <= 0:
                    return (torch.zeros_like(x),) + args[1:]
                _, top_idx = torch.topk(x.abs(), k=k_val, dim=-1, sorted=False)
                out = torch.zeros_like(x)
                out.scatter_(-1, top_idx, torch.gather(x, -1, top_idx))
                return (out,) + args[1:]
            return _hook

        handles.append(down_proj.register_forward_pre_hook(_make_pre_hook(int(k_keep))))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# 3. Experiment B: Procrustes Mid-Stack Recurrent Looping (`L24..L42`)
# ---------------------------------------------------------------------------

def solve_orthogonal_procrustes(
    h_src: np.ndarray,
    h_dst: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit exact RMS-standardized Orthogonal Procrustes map `h_dst_hat = sigma_dst * ((h_src - mu_src)/sigma_src) @ R_0.T + mu_dst`."""
    x = np.asarray(h_src, dtype=np.float64)
    y = np.asarray(h_dst, dtype=np.float64)
    d = x.shape[1]

    mu_src = np.mean(x, axis=0)
    mu_dst = np.mean(y, axis=0)
    x_c = x - mu_src
    y_c = y - mu_dst

    rms_src = float(np.sqrt(np.mean(x_c * x_c) + 1e-12))
    rms_dst = float(np.sqrt(np.mean(y_c * y_c) + 1e-12))

    x_n = x_c / rms_src
    y_n = y_c / rms_dst

    # Maximize trace(R @ X_n^T @ Y_n) => M = Y_n^T @ X_n = U S V^T => R_0 = U @ V^T
    m_cov = y_n.T @ x_n
    u_mat, _, vt_mat = np.linalg.svd(m_cov, full_matrices=False)
    r_0 = u_mat @ vt_mat  # [D, D], orthogonal: r_0.T @ r_0 == I_D

    recon = rms_dst * (x_n @ r_0.T) + mu_dst
    naive = y_c * 0.0 + x  # unrotated direct pass
    rel_err_procrustes = float(np.linalg.norm(y - recon) / max(np.linalg.norm(y), 1e-9))
    rel_err_identity = float(np.linalg.norm(y - naive) / max(np.linalg.norm(y), 1e-9))
    cos_before = float(np.mean(np.sum(x_n * y_n, axis=-1) / (np.linalg.norm(x_n, axis=-1) * np.linalg.norm(y_n, axis=-1) + 1e-12)))
    x_rot = x_n @ r_0.T
    cos_after = float(np.mean(np.sum(x_rot * y_n, axis=-1) / (np.linalg.norm(x_rot, axis=-1) * np.linalg.norm(y_n, axis=-1) + 1e-12)))

    return {
        "r_0": r_0.astype(np.float32),
        "mu_src": mu_src.astype(np.float32),
        "mu_dst": mu_dst.astype(np.float32),
        "rms_src": np.float32(rms_src),
        "rms_dst": np.float32(rms_dst),
        "d_model": int(d),
        "rel_err_identity": round(rel_err_identity, 4),
        "rel_err_procrustes": round(rel_err_procrustes, 4),
        "mean_cosine_identity": round(cos_before, 4),
        "mean_cosine_procrustes": round(cos_after, 4),
    }


def build_procrustes_bridge_module(
    torch: Any,
    procrustes_fit: dict[str, Any],
    rank: int = 128,
    init_gate: float = 0.12,
    use_identity_rotation: bool = False,
    device: Any = None,
) -> Any:
    """Create the anchored residual re-entry bridge `H_24^(k+1) = (1 - g) * H_24^(0) + g * Bridge(H_42^(k))`."""
    nn = torch.nn
    d = int(procrustes_fit["d_model"])

    class ProcrustesLoopBridge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            r_init = np.eye(d, dtype=np.float32) if use_identity_rotation else procrustes_fit["r_0"]
            self.register_buffer("r_ortho", torch.tensor(r_init, dtype=torch.float32))
            self.register_buffer("mu_src", torch.tensor(procrustes_fit["mu_src"], dtype=torch.float32))
            self.register_buffer("rms_src", torch.tensor(float(procrustes_fit["rms_src"]), dtype=torch.float32))

            self.mu_dst = nn.Parameter(torch.tensor(procrustes_fit["mu_dst"], dtype=torch.float32))
            self.log_rms_dst = nn.Parameter(torch.tensor(math.log(max(float(procrustes_fit["rms_dst"]), 1e-6)), dtype=torch.float32))

            # Low-rank trainable correction delta B @ A on top of exact full-rank orthogonal R_0
            self.lora_a = nn.Parameter(torch.randn(rank, d, dtype=torch.float32) * (1.0 / math.sqrt(d)))
            self.lora_b = nn.Parameter(torch.zeros(d, rank, dtype=torch.float32))

            # Per-channel gate g in [0, 1]^D; allow exact g=0 override for step-0 equivalence check
            if init_gate <= 0.0:
                init_logit = -50.0
            elif init_gate >= 1.0:
                init_logit = 50.0
            else:
                init_logit = math.log(init_gate / (1.0 - init_gate))
            self.gate_logit = nn.Parameter(torch.full((d,), init_logit, dtype=torch.float32))
            self.force_zero_gate: bool = False

        def forward(self, h24_anchor: Any, h42_state: Any) -> Any:
            orig_dtype = h24_anchor.dtype
            if self.force_zero_gate:
                return h24_anchor

            x = h42_state.to(torch.float32)
            anchor = h24_anchor.to(torch.float32)

            x_centered = x - self.mu_src
            cur_rms = torch.sqrt(torch.mean(x_centered * x_centered, dim=-1, keepdim=True) + 1e-8)
            x_norm = x_centered / cur_rms

            rot = torch.matmul(x_norm, self.r_ortho.t())
            delta = torch.matmul(torch.matmul(x_norm, self.lora_a.t()), self.lora_b.t())
            mapped = torch.exp(self.log_rms_dst) * (rot + delta) + self.mu_dst

            g = torch.sigmoid(self.gate_logit)
            blended = (1.0 - g) * anchor + g * mapped
            return blended.to(orig_dtype)

    mod = ProcrustesLoopBridge()
    if device is not None:
        mod = mod.to(device)
    return mod


@contextmanager
def midstack_recurrent_loop(
    harness: TransformerHarness,
    bridge: Any,
    start_layer: int,
    end_layer: int,
    n_loops: int = 2,
    use_grad_checkpoint: bool = False,
) -> Iterator[None]:
    """Intercept `layers[start_layer..end_layer]` and execute `n_loops` recurrent passes seamlessly."""
    if n_loops <= 1:
        yield
        return

    torch = harness.torch
    ckpt_fn = torch.utils.checkpoint.checkpoint if use_grad_checkpoint else None

    state_box: dict[str, Any] = {
        "in_extra_loop": False,
        "h_anchor": None,
        "call_specs": {},
    }
    handles = []

    for ell in range(start_layer, end_layer + 1):
        blk = harness.layers[ell]

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

                if idx == start_layer:
                    state_box["h_anchor"] = h_in
                state_box["call_specs"][idx] = (had_pos, rest_args, clean_kw)
            return _pre

        handles.append(blk.register_forward_pre_hook(_make_pre(ell), with_kwargs=True))

    def _end_post_hook(_mod: Any, _args: Any, out: Any):
        if state_box["in_extra_loop"]:
            return out
        was_tuple = isinstance(out, tuple)
        h_cur = out[0] if was_tuple else out
        rest_tuple = out[1:] if was_tuple else ()
        h_anchor = state_box["h_anchor"]

        state_box["in_extra_loop"] = True
        try:
            for _loop_i in range(1, n_loops):
                h_cur = bridge(h_anchor, h_cur)
                for ell in range(start_layer, end_layer + 1):
                    blk_l = harness.layers[ell]
                    had_pos, rest_args, clean_kw = state_box["call_specs"][ell]

                    def _invoke_layer(h_tensor: Any, _b=blk_l, _hp=had_pos, _ra=rest_args, _ck=clean_kw):
                        if _hp:
                            res = _b(h_tensor, *_ra, **_ck)
                        else:
                            res = _b(hidden_states=h_tensor, **_ck)
                        return res[0] if isinstance(res, tuple) else res

                    if ckpt_fn is not None and h_cur.requires_grad:
                        h_cur = ckpt_fn(_invoke_layer, h_cur, use_reentrant=False)
                    else:
                        h_cur = _invoke_layer(h_cur)
        finally:
            state_box["in_extra_loop"] = False
            state_box["h_anchor"] = None
            state_box["call_specs"].clear()

        return (h_cur,) + rest_tuple if was_tuple else h_cur

    handles.append(harness.layers[end_layer].register_forward_hook(_end_post_hook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# 4. Evaluation & Training Routines
# ---------------------------------------------------------------------------

def precompute_base_kl_references(
    harness: TransformerHarness,
    passages: Sequence[str],
    top_m: int = 256,
    batch_size: int = 2,
) -> list[dict[str, Any]]:
    """Cache Base Model top-M probabilities & target token IDs across chat-wrapped assistant passages."""
    torch = harness.torch
    F = torch.nn.functional
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))
    pad_id = int(tok.pad_token_id or 0)
    header = format_chat_prompt(tok, "Provide a concise technical reference note:")
    h_ids = tok.encode(header, add_special_tokens=not has_chat_tmpl)

    refs: list[dict[str, Any]] = []
    for i in range(0, len(passages), batch_size):
        chunk = list(passages[i : i + batch_size])
        b_seqs = []
        p_starts = []
        for p_txt in chunk:
            c_ids = tok.encode(p_txt, add_special_tokens=False)[:200]
            b_seqs.append(h_ids + c_ids)
            p_starts.append(len(h_ids))
        max_l = max(len(s) for s in b_seqs)

        inp_rows = []
        msk_rows = []
        val_rows = []
        for s, pst in zip(b_seqs, p_starts):
            pad_n = max_l - len(s)
            inp_rows.append([pad_id] * pad_n + s)
            msk_rows.append([0] * pad_n + [1] * len(s))
            # Predict continuation tokens at target positions `pad_n + pst .. max_l - 1`
            # In shifted logits[:, :-1], target index `t` corresponds to logit index `t - 1`
            v_row = [False] * (max_l - 1)
            for t_pos in range(max(1, pad_n + pst), max_l):
                v_row[t_pos - 1] = True
            val_rows.append(v_row)

        input_ids = torch.tensor(inp_rows, dtype=torch.long, device=harness.input_device)
        attn_mask = torch.tensor(msk_rows, dtype=torch.long, device=harness.input_device)
        valid_mask = torch.tensor(val_rows, dtype=torch.bool, device=harness.input_device)
        with torch.no_grad():
            out = harness.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            logits = out.logits[:, :-1, :].to(torch.float32)
            target_ids = input_ids[:, 1:]

            log_p = F.log_softmax(logits, dim=-1)
            k_eff = min(top_m, log_p.shape[-1])
            top_lp, top_idx = torch.topk(log_p, k=k_eff, dim=-1)
            top_p = torch.exp(top_lp)
            tgt_lp = torch.gather(log_p, -1, target_ids.unsqueeze(-1)).squeeze(-1)

        refs.append({
            "input_ids": input_ids.cpu(),
            "attention_mask": attn_mask.cpu(),
            "valid_mask": valid_mask.cpu(),
            "target_ids": target_ids.cpu(),
            "top_idx": top_idx.cpu(),
            "top_lp": top_lp.cpu(),
            "top_p": top_p.cpu(),
            "base_tgt_lp": tgt_lp.cpu(),
        })
    return refs


def evaluate_kl_and_ppl(
    harness: TransformerHarness,
    refs: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Compute exact token-level `KL(p_base || p_mod)` (in nats) and perplexity vs. precomputed base."""
    torch = harness.torch
    F = torch.nn.functional

    total_kl = 0.0
    total_nll = 0.0
    total_base_nll = 0.0
    total_toks = 0

    for ref in refs:
        input_ids = ref["input_ids"].to(harness.input_device)
        attn_mask = ref["attention_mask"].to(harness.input_device)
        valid_mask = ref["valid_mask"].to(harness.input_device)
        target_ids = ref["target_ids"].to(harness.input_device)
        top_idx = ref["top_idx"].to(harness.input_device)
        top_lp = ref["top_lp"].to(harness.input_device, dtype=torch.float32)
        top_p = ref["top_p"].to(harness.input_device, dtype=torch.float32)
        base_tgt_lp = ref["base_tgt_lp"].to(harness.input_device)

        with torch.no_grad():
            out = harness.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            logits = out.logits[:, :-1, :].to(torch.float32)
            mod_log_p = F.log_softmax(logits, dim=-1)

            mod_top_lp = torch.gather(mod_log_p, -1, top_idx)
            # Remainder tail bucket for exact probability conservation
            rem_p = torch.clamp(1.0 - torch.sum(top_p, dim=-1), min=0.0)
            mod_rem_p = torch.clamp(1.0 - torch.sum(torch.exp(mod_top_lp), dim=-1), min=1e-9)
            kl_pos = torch.sum(top_p * (top_lp - mod_top_lp), dim=-1) + rem_p * (
                torch.log(torch.clamp(rem_p, min=1e-9)) - torch.log(mod_rem_p)
            )
            kl_pos = torch.clamp(kl_pos, min=0.0)

            mod_tgt_lp = torch.gather(mod_log_p, -1, target_ids.unsqueeze(-1)).squeeze(-1)

            m_f = valid_mask.to(torch.float32)
            total_kl += float(torch.sum(kl_pos * m_f).item())
            total_nll += float(torch.sum(-mod_tgt_lp * m_f).item())
            total_base_nll += float(torch.sum(-base_tgt_lp * m_f).item())
            total_toks += int(torch.sum(valid_mask).item())

    denom = max(1, total_toks)
    mean_nll = total_nll / denom
    base_nll = total_base_nll / denom
    return {
        "kl_nats": round(total_kl / denom, 5),
        "nll": round(mean_nll, 4),
        "delta_nll": round(mean_nll - base_nll, 4),
        "ppl": round(math.exp(min(mean_nll, 20.0)), 3),
        "n_tokens": total_toks,
    }


def _parse_int_from_text(text: str) -> int | None:
    # Prefer explicit FINAL_INTEGER: tag if present, else first signed integer
    idx = text.rfind("FINAL_INTEGER:")
    sub = text[idx + len("FINAL_INTEGER:") :] if idx >= 0 else text
    m = RE_FIRST_INT.search(sub)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def evaluate_nothink_math_suite(
    harness: TransformerHarness,
    items: Sequence[MathChainItem],
    batch_size: int = 16,
) -> dict[str, Any]:
    """Score `0-CoT` (`no_think`) exact-match accuracy, gold-answer log-prob, and wall-clock ms/item."""
    torch = harness.torch
    F = torch.nn.functional
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))

    prompts = [format_chat_prompt(tok, it.prompt_user) for it in items]
    gold_strs = [f"FINAL_INTEGER: {it.answer_int}" for it in items]
    stop_set = set(harness._resolve_stop_token_ids())

    t0 = time.perf_counter()
    preds: list[int | None] = []
    eos_id = tok.eos_token_id
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(
            chunk,
            add_special_tokens=not has_chat_tmpl,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        cur_ids = enc["input_ids"].to(harness.input_device)
        cur_mask = enc["attention_mask"].to(harness.input_device)
        p_len = cur_ids.shape[1]
        bsz = cur_ids.shape[0]
        done = torch.zeros(bsz, dtype=torch.bool, device=harness.input_device)
        pad_fill = int(eos_id if eos_id is not None else (tok.pad_token_id or 0))
        stop_tensor = torch.tensor(sorted(stop_set), dtype=torch.long, device=harness.input_device) if stop_set else None

        with torch.no_grad():
            for _step in range(20):
                out = harness.model(input_ids=cur_ids, attention_mask=cur_mask, use_cache=False)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
                next_tok = torch.where(done, torch.full_like(next_tok, pad_fill), next_tok)
                cur_ids = torch.cat([cur_ids, next_tok.unsqueeze(1)], dim=1)
                cur_mask = torch.cat([cur_mask, (~done).to(cur_mask.dtype).unsqueeze(1)], dim=1)
                if stop_tensor is not None and stop_tensor.numel() > 0:
                    done = done | torch.isin(next_tok, stop_tensor)
                if bool(torch.all(done).item()):
                    break

        for row in cur_ids[:, p_len:]:
            txt = tok.decode(row.tolist(), skip_special_tokens=True)
            preds.append(_parse_int_from_text(txt))
    elapsed_ms_per_item = 1000.0 * (time.perf_counter() - t0) / max(1, len(items))

    # Teacher-forced gold integer log-probability (continuous sensitivity metric)
    gold_logprobs: list[float] = []
    for i in range(0, len(items), batch_size):
        p_chunk = prompts[i : i + batch_size]
        g_chunk = gold_strs[i : i + batch_size]
        p_ids_list = [tok.encode(p, add_special_tokens=not has_chat_tmpl)[-768:] for p in p_chunk]
        a_ids_list = [tok.encode(g, add_special_tokens=False) for g in g_chunk]
        f_ids_list = [p_i + a_i for p_i, a_i in zip(p_ids_list, a_ids_list)]
        max_l = max(len(x) for x in f_ids_list)
        pad_id = int(tok.pad_token_id or 0)

        # Left-pad so sequences align cleanly
        batch_ids = []
        batch_mask = []
        ans_spans = []
        for seq, p_i in zip(f_ids_list, p_ids_list):
            pad_n = max_l - len(seq)
            batch_ids.append([pad_id] * pad_n + seq)
            batch_mask.append([0] * pad_n + [1] * len(seq))
            ans_spans.append((pad_n + len(p_i), max_l))

        inp = torch.tensor(batch_ids, dtype=torch.long, device=harness.input_device)
        msk = torch.tensor(batch_mask, dtype=torch.long, device=harness.input_device)
        with torch.no_grad():
            out = harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
            lp = F.log_softmax(out.logits[:, :-1, :].to(torch.float32), dim=-1)
            for b_idx, (s_pos, e_pos) in enumerate(ans_spans):
                if e_pos <= s_pos:
                    gold_logprobs.append(0.0)
                    continue
                # Logit at pos `t-1` predicts token at `t`
                tok_lps = [
                    float(lp[b_idx, pos - 1, inp[b_idx, pos]].item())
                    for pos in range(max(1, s_pos), e_pos)
                ]
                gold_logprobs.append(float(np.sum(tok_lps)))

    correct = [int(p == it.answer_int) for p, it in zip(preds, items)]
    by_hops: dict[str, Any] = {}
    hop_levels = sorted({it.hops for it in items})
    for h_val in hop_levels:
        idxs = [j for j, it in enumerate(items) if it.hops == h_val]
        by_hops[f"hop_{h_val}"] = {
            "acc": round(float(np.mean([correct[j] for j in idxs])), 4),
            "mean_gold_logprob": round(float(np.mean([gold_logprobs[j] for j in idxs])), 4),
            "n": len(idxs),
        }

    return {
        "overall_acc": round(float(np.mean(correct)), 4),
        "mean_gold_logprob": round(float(np.mean(gold_logprobs)), 4),
        "ms_per_item": round(elapsed_ms_per_item, 2),
        "n_items": len(items),
        "by_hops": by_hops,
    }


def evaluate_thought_reasoning_loops(
    harness: TransformerHarness,
    items: Sequence[MathChainItem],
    max_new_tokens: int = 240,
    batch_size: int = 12,
) -> dict[str, Any]:
    """Evaluate `<thought>` accuracy AND Goodfire's runaway reasoning-loop / cap-hit rate."""
    prompts = [format_chat_prompt_thinking(harness.tokenizer, it.prompt_user, enable_thinking=True) for it in items]
    t0 = time.perf_counter()
    recs = generate_thought_records(
        harness,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        rescue_unclosed_thought=False,
    )
    ms_per_item = 1000.0 * (time.perf_counter() - t0) / max(1, len(items))

    correct_strict = 0
    unclosed_loop_hits = 0
    gen_lens: list[int] = []
    for rec, it in zip(recs, items):
        gen_lens.append(rec.n_gen_tokens)
        if rec.thought_present and not rec.thought_closed:
            unclosed_loop_hits += 1
        pred = _parse_int_from_text(rec.public_text)
        if pred == it.answer_int:
            correct_strict += 1

    n = max(1, len(items))
    return {
        "thought_acc": round(correct_strict / n, 4),
        "unclosed_loop_rate": round(unclosed_loop_hits / n, 4),
        "mean_gen_tokens": round(float(np.mean(gen_lens)), 1),
        "ms_per_item": round(ms_per_item, 2),
        "n_items": len(items),
    }


def collect_procrustes_calibration_states(
    harness: TransformerHarness,
    train_items: Sequence[MathChainItem],
    anchor_passages: Sequence[str],
    start_layer: int,
    end_layer: int,
    max_tokens_collect: int = 4096,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect paired `(h_in[start_layer], h_out[end_layer])` states across training prompts."""
    torch = harness.torch
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))

    texts = [
        format_chat_prompt(tok, it.prompt_user)
        for it in train_items[:96]
    ] + list(anchor_passages)

    h_start_list: list[np.ndarray] = []
    h_end_list: list[np.ndarray] = []
    collected = 0

    for i in range(0, len(texts), batch_size):
        if collected >= max_tokens_collect:
            break
        chunk = texts[i : i + batch_size]
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

        box: dict[str, Any] = {}

        def _pre_s(_m: Any, args: Any, kwargs: Any):
            t = args[0] if args else kwargs["hidden_states"]
            box["s"] = t.detach().to(torch.float32).cpu()

        def _post_e(_m: Any, _args: Any, out: Any):
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
        s_np = box["s"].numpy()[valid]
        e_np = box["e"].numpy()[valid]
        # Subsample up to 48 tokens per batch for balanced diversity
        if s_np.shape[0] > 64:
            idx = np.linspace(0, s_np.shape[0] - 1, 64, dtype=int)
            s_np = s_np[idx]
            e_np = e_np[idx]
        h_start_list.append(s_np)
        h_end_list.append(e_np)
        collected += int(s_np.shape[0])

    return np.concatenate(h_start_list, axis=0), np.concatenate(h_end_list, axis=0)


def train_procrustes_bridge(
    harness: TransformerHarness,
    bridge: Any,
    train_items: Sequence[MathChainItem],
    anchor_passages: Sequence[str],
    start_layer: int,
    end_layer: int,
    n_steps: int = 240,
    batch_size: int = 4,
    lr: float = 2.5e-3,
) -> dict[str, Any]:
    """Train ONLY `ProcrustesLoopBridge` (`1.4M` params, base 31B frozen) with answer CE + KL anchor."""
    torch = harness.torch
    F = torch.nn.functional
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))

    # Strictly freeze all base model weights
    for p in harness.model.parameters():
        p.requires_grad_(False)
    for p in bridge.parameters():
        p.requires_grad_(True)

    anchor_refs = precompute_base_kl_references(harness, anchor_passages, top_m=128, batch_size=2)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_steps), eta_min=lr * 0.1)

    rng = random.Random(20261120)
    pool = list(train_items)
    pad_id = int(tok.pad_token_id or 0)

    loss_curve: list[float] = []
    kl_anchor_curve: list[float] = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        batch = [pool[(step * batch_size + j) % len(pool)] for j in range(batch_size)]
        if step > 0 and (step * batch_size) % len(pool) < batch_size:
            rng.shuffle(pool)

        prompts = [format_chat_prompt(tok, it.prompt_user) for it in batch]
        stop_ids = harness._resolve_stop_token_ids()
        eos_tail = [int(stop_ids[0])] if stop_ids else tok.encode("\n", add_special_tokens=False)

        p_ids_list = [tok.encode(p, add_special_tokens=not has_chat_tmpl)[-256:] for p in prompts]
        a_ids_list = [tok.encode(f"FINAL_INTEGER: {it.answer_int}", add_special_tokens=False) + eos_tail for it in batch]
        f_seqs = [p_i + a_i for p_i, a_i in zip(p_ids_list, a_ids_list)]
        max_l = max(len(s) for s in f_seqs)

        b_ids = []
        b_msk = []
        b_labels = []
        for seq, p_i, a_i in zip(f_seqs, p_ids_list, a_ids_list):
            pad_n = max_l - len(seq)
            b_ids.append([pad_id] * pad_n + seq)
            b_msk.append([0] * pad_n + [1] * len(seq))
            lbl = [-100] * (pad_n + len(p_i)) + a_i
            b_labels.append(lbl)

        inp = torch.tensor(b_ids, dtype=torch.long, device=harness.input_device)
        msk = torch.tensor(b_msk, dtype=torch.long, device=harness.input_device)
        labels = torch.tensor(b_labels, dtype=torch.long, device=harness.input_device)

        with torch.enable_grad():
            with midstack_recurrent_loop(
                harness, bridge, start_layer, end_layer, n_loops=2, use_grad_checkpoint=True
            ):
                out = harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
                shift_logits = out.logits[:, :-1, :].to(torch.float32)
                shift_labels = labels[:, 1:]
                ce_loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )

                # Every 2nd step, include general-text KL anchor so general language stays rock-solid
                kl_val = 0.0
                if step % 2 == 0 and anchor_refs:
                    aref = anchor_refs[(step // 2) % len(anchor_refs)]
                    a_inp = aref["input_ids"].to(harness.input_device)
                    a_msk = aref["attention_mask"].to(harness.input_device)
                    a_valid = aref["valid_mask"].to(harness.input_device).to(torch.float32)
                    a_tidx = aref["top_idx"].to(harness.input_device)
                    a_tlp = aref["top_lp"].to(harness.input_device, dtype=torch.float32)
                    a_tp = aref["top_p"].to(harness.input_device, dtype=torch.float32)

                    a_out = harness.model(input_ids=a_inp, attention_mask=a_msk, use_cache=False)
                    a_mod_lp = F.log_softmax(a_out.logits[:, :-1, :].to(torch.float32), dim=-1)
                    a_mod_top_lp = torch.gather(a_mod_lp, -1, a_tidx)
                    kl_tok = torch.clamp(torch.sum(a_tp * (a_tlp - a_mod_top_lp), dim=-1), min=0.0)
                    kl_term = torch.sum(kl_tok * a_valid) / torch.clamp(torch.sum(a_valid), min=1.0)
                    total_loss = ce_loss + 0.6 * kl_term
                    kl_val = float(kl_term.detach().item())
                else:
                    total_loss = ce_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        loss_curve.append(round(float(ce_loss.detach().item()), 4))
        if kl_val > 0.0:
            kl_anchor_curve.append(round(kl_val, 5))

        if (step + 1) % 40 == 0 or step == 0:
            g_vals = torch.sigmoid(bridge.gate_logit.detach()).cpu().numpy()
            print(
                f"  [Loop Distill step {step+1:3d}/{n_steps}] CE={loss_curve[-1]:.4f} "
                f"KL_anc={(kl_anchor_curve[-1] if kl_anchor_curve else 0.0):.4f} "
                f"gate_mean={float(np.mean(g_vals)):.3f} (p90={float(np.percentile(g_vals, 90)):.3f})",
                flush=True,
            )

    train_s = time.perf_counter() - t0
    g_final = torch.sigmoid(bridge.gate_logit.detach()).cpu().numpy()
    n_trainable = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    _maybe_empty_cuda_cache(harness)

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "trainable_params": int(n_trainable),
        "train_seconds": round(train_s, 2),
        "ce_loss_first10_mean": round(float(np.mean(loss_curve[:10])), 4),
        "ce_loss_last10_mean": round(float(np.mean(loss_curve[-10:])), 4),
        "loss_curve_subsampled": loss_curve[:: max(1, len(loss_curve) // 32)],
        "kl_anchor_last5_mean": round(float(np.mean(kl_anchor_curve[-5:])), 5) if kl_anchor_curve else 0.0,
        "learned_gate_summary": {
            "mean": round(float(np.mean(g_final)), 4),
            "std": round(float(np.std(g_final)), 4),
            "p10": round(float(np.percentile(g_final, 10)), 4),
            "p90": round(float(np.percentile(g_final, 90)), 4),
            "max": round(float(np.max(g_final)), 4),
        },
    }


def _atomic_write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_optimize_experiment(
    harness: TransformerHarness,
    batch_size: int = 16,
    n_distill_steps: int = 240,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Execute Phase 8 Funnel-Guided Optimization (Part A: Hourglass Sparsity + Part B: Procrustes Looping)."""
    torch = harness.torch
    t_start = time.time()

    audit = audit_optimize_suite()
    if not audit["valid"]:
        raise RuntimeError(f"Phase 8 benchmark audit failed: {audit}")

    bm = build_phase8_benchmarks()
    n_layers = harness.n_layers
    d_model = int(harness.layers[0].mlp.down_proj.weight.shape[0])
    d_ff = int(harness.layers[0].mlp.down_proj.weight.shape[1])

    funnel_start = max(1, int(round(0.40 * n_layers)))
    funnel_end = min(n_layers - 2, int(round(0.70 * n_layers)))

    print(
        f"[Optimize] Model={harness.model_id} | n_layers={n_layers} | d_model={d_model} | d_ff={d_ff} "
        f"| FunnelBand=L{funnel_start}..L{funnel_end}",
        flush=True,
    )

    print("[Optimize] Caching Base Model KL references across held-out passages...", flush=True)
    kl_refs = precompute_base_kl_references(harness, bm["eval_kl_passages"], top_m=256, batch_size=2)

    payload: dict[str, Any] = {
        "metadata": {
            "model_id": harness.model_id,
            "n_layers": n_layers,
            "d_model": d_model,
            "d_ff": d_ff,
            "funnel_band": [funnel_start, funnel_end],
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "benchmark_audit": audit,
        }
    }

    # ------------------------------------------------------------------
    # PART A: Hourglass Activation Sparsity vs. Goodfire Uniform Sparsity
    # ------------------------------------------------------------------
    schedules = compute_sparsity_schedules(n_layers, d_ff, funnel_start, funnel_end)
    part_a_results: dict[str, Any] = {}

    # Evaluate intact baseline once (shared by Part A & Part B!)
    print("[Part A] Evaluating intact 1-pass baseline (KL, no_think math, and <thought> reasoning)...", flush=True)
    base_kl = evaluate_kl_and_ppl(harness, kl_refs)
    base_mh = evaluate_nothink_math_suite(harness, bm["eval_multihop"], batch_size=batch_size)
    base_wp = evaluate_nothink_math_suite(harness, bm["eval_word_problems"], batch_size=batch_size)
    base_thought_wp = evaluate_thought_reasoning_loops(
        harness, bm["eval_word_problems"][:24], max_new_tokens=512, batch_size=min(batch_size, 12)
    )
    base_thought_mh57 = evaluate_thought_reasoning_loops(
        harness, [x for x in bm["eval_multihop"] if x.hops in (5, 7)][:24], max_new_tokens=512, batch_size=min(batch_size, 12)
    )

    part_a_results["intact_base"] = {
        "active_fraction_global": 1.0,
        "k_outer": d_ff,
        "k_funnel": d_ff,
        "kl_eval": base_kl,
        "multihop_nothink": base_mh,
        "word_problems_nothink": base_wp,
        "word_problems_thought": base_thought_wp,
        "multihop_5_7_thought": base_thought_mh57,
    }
    _atomic_write_json(checkpoint_path, {**payload, "part_a_hourglass_sparsity": part_a_results})

    for sched_name, k_map in schedules.items():
        tot_keep = sum(k_map.values())
        frac_keep = tot_keep / float(n_layers * d_ff)
        k_out = k_map[min(2, n_layers - 1)]
        k_fun = k_map[funnel_start]
        print(
            f"[Part A] Running {sched_name} (global={frac_keep*100:.2f}%, corridor_k={k_out}, funnel_k={k_fun})...",
            flush=True,
        )
        with layerwise_topk_mlp_sparsity(harness, k_map):
            s_kl = evaluate_kl_and_ppl(harness, kl_refs)
            s_mh = evaluate_nothink_math_suite(harness, bm["eval_multihop"], batch_size=batch_size)
            s_wp = evaluate_nothink_math_suite(harness, bm["eval_word_problems"], batch_size=batch_size)
            s_th = evaluate_thought_reasoning_loops(
                harness, bm["eval_word_problems"][:24], max_new_tokens=512, batch_size=min(batch_size, 12)
            )
        print(
            f"  -> KL={s_kl['kl_nats']:.4f} nats | MH_no_think={s_mh['overall_acc']*100:.1f}% "
            f"| WP_thought={s_th['thought_acc']*100:.1f}% (loop_cap_hits={s_th['unclosed_loop_rate']*100:.1f}%)",
            flush=True,
        )
        part_a_results[sched_name] = {
            "active_fraction_global": round(frac_keep, 5),
            "k_outer": int(k_out),
            "k_funnel": int(k_fun),
            "kl_eval": s_kl,
            "multihop_nothink": s_mh,
            "word_problems_nothink": s_wp,
            "word_problems_thought": s_th,
        }
        _maybe_empty_cuda_cache(harness)

    payload["part_a_hourglass_sparsity"] = part_a_results
    _atomic_write_json(checkpoint_path, payload)

    # ------------------------------------------------------------------
    # PART B: Procrustes Mid-Stack Recurrent Looping (`L24..L42`)
    # ------------------------------------------------------------------
    print(f"[Part B] Collecting (L{funnel_start}, L{funnel_end}) states & solving SVD Orthogonal Procrustes...", flush=True)
    h_start_cal, h_end_cal = collect_procrustes_calibration_states(
        harness, bm["train_items"], bm["train_anchor_passages"], funnel_start, funnel_end
    )
    proc_fit = solve_orthogonal_procrustes(h_src=h_end_cal, h_dst=h_start_cal)
    print(
        f"  -> Procrustes alignment: cos {proc_fit['mean_cosine_identity']:+.4f} -> {proc_fit['mean_cosine_procrustes']:+.4f} "
        f"| rel_err {proc_fit['rel_err_identity']:.4f} -> {proc_fit['rel_err_procrustes']:.4f}",
        flush=True,
    )

    # Gate-0 exact equivalence verification on live GPU model
    bridge_main = build_procrustes_bridge_module(
        torch, proc_fit, rank=128, init_gate=0.14, use_identity_rotation=False, device=harness.input_device
    )
    enc_chk = harness.tokenizer(
        list(bm["eval_kl_passages"][:2]), return_tensors="pt", padding=True, truncation=True, max_length=96
    )
    enc_chk = {k: v.to(harness.input_device) for k, v in enc_chk.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        logits_ref = harness.model(**enc_chk, use_cache=False).logits[:, -1, :].to(torch.float32)
        bridge_main.force_zero_gate = True
        with midstack_recurrent_loop(harness, bridge_main, funnel_start, funnel_end, n_loops=2):
            logits_g0 = harness.model(**enc_chk, use_cache=False).logits[:, -1, :].to(torch.float32)
        bridge_main.force_zero_gate = False
    gate0_max_diff = float(torch.max(torch.abs(logits_ref - logits_g0)).item())
    print(f"  -> Live Gate-0 2-Loop Replay Equivalence max |delta logit| = {gate0_max_diff:.2e}", flush=True)

    part_b_conditions: dict[str, Any] = {
        "base_1pass_nothink": {
            "n_loops": 1,
            "kl_eval": base_kl,
            "multihop_nothink": base_mh,
            "word_problems_nothink": base_wp,
        }
    }

    # Control 1: Naive unrotated 2-loop (Identity R = I, gate = 0.20)
    print("[Part B] Evaluating Control: Naive Unrotated 2-Loop (R = Identity, g = 0.20)...", flush=True)
    bridge_naive = build_procrustes_bridge_module(
        torch, proc_fit, rank=8, init_gate=0.20, use_identity_rotation=True, device=harness.input_device
    )
    with midstack_recurrent_loop(harness, bridge_naive, funnel_start, funnel_end, n_loops=2):
        part_b_conditions["naive_2loop_unrotated"] = {
            "n_loops": 2,
            "kl_eval": evaluate_kl_and_ppl(harness, kl_refs),
            "multihop_nothink": evaluate_nothink_math_suite(harness, bm["eval_multihop"], batch_size=batch_size),
            "word_problems_nothink": evaluate_nothink_math_suite(harness, bm["eval_word_problems"], batch_size=batch_size),
        }
    del bridge_naive
    _maybe_empty_cuda_cache(harness)

    # Control 2: Zero-shot SVD Orthogonal Procrustes 2-loop (0 gradient steps, g = 0.14)
    print("[Part B] Evaluating Zero-Shot SVD Orthogonal Procrustes 2-Loop (0 gradient steps)...", flush=True)
    with midstack_recurrent_loop(harness, bridge_main, funnel_start, funnel_end, n_loops=2):
        part_b_conditions["procrustes_2loop_zeroshot"] = {
            "n_loops": 2,
            "kl_eval": evaluate_kl_and_ppl(harness, kl_refs),
            "multihop_nothink": evaluate_nothink_math_suite(harness, bm["eval_multihop"], batch_size=batch_size),
            "word_problems_nothink": evaluate_nothink_math_suite(harness, bm["eval_word_problems"], batch_size=batch_size),
        }

    # Train ProcrustesLoopBridge (`240` steps, base 31B frozen!)
    print(f"[Part B] Distilling Procrustes Mid-Stack Loop Bridge ({n_distill_steps} steps, base 31B frozen)...", flush=True)
    train_stats = train_procrustes_bridge(
        harness=harness,
        bridge=bridge_main,
        train_items=bm["train_items"],
        anchor_passages=bm["train_anchor_passages"],
        start_layer=funnel_start,
        end_layer=funnel_end,
        n_steps=n_distill_steps,
        batch_size=min(4, max(1, batch_size // 2)),
        lr=2.5e-3,
    )
    bridge_main.eval()

    # Evaluate Trained 2-Loop (`no_think`)
    print("[Part B] Evaluating Distilled Procrustes 2-Loop (`0` CoT tokens)...", flush=True)
    with midstack_recurrent_loop(harness, bridge_main, funnel_start, funnel_end, n_loops=2):
        part_b_conditions["procrustes_2loop_distilled"] = {
            "n_loops": 2,
            "kl_eval": evaluate_kl_and_ppl(harness, kl_refs),
            "multihop_nothink": evaluate_nothink_math_suite(harness, bm["eval_multihop"], batch_size=batch_size),
            "word_problems_nothink": evaluate_nothink_math_suite(harness, bm["eval_word_problems"], batch_size=batch_size),
        }

    # Evaluate Zero-Shot 3-Loop Depth Extrapolation (`1.5x` recurrent loops!)
    print("[Part B] Evaluating Procrustes 3-Loop Depth Extrapolation (`0` CoT tokens)...", flush=True)
    with midstack_recurrent_loop(harness, bridge_main, funnel_start, funnel_end, n_loops=3):
        part_b_conditions["procrustes_3loop_extrap"] = {
            "n_loops": 3,
            "kl_eval": evaluate_kl_and_ppl(harness, kl_refs),
            "multihop_nothink": evaluate_nothink_math_suite(harness, bm["eval_multihop"], batch_size=batch_size),
            "word_problems_nothink": evaluate_nothink_math_suite(harness, bm["eval_word_problems"], batch_size=batch_size),
        }

    # Record `<thought>` ceiling reference for latency & accuracy Pareto comparison
    part_b_conditions["ceiling_1pass_with_thought"] = {
        "n_loops": 1,
        "word_problems_thought": base_thought_wp,
        "multihop_5_7_thought": base_thought_mh57,
    }

    # Strip large matrices before JSON serialization
    proc_meta = {k: float(v) if isinstance(v, (np.floating, float, int)) else v for k, v in proc_fit.items() if k not in ("r_0", "mu_src", "mu_dst")}
    payload["part_b_procrustes_looping"] = {
        "procrustes_alignment": proc_meta,
        "gate0_replay_max_logit_diff": gate0_max_diff,
        "bridge_training": train_stats,
        "conditions": part_b_conditions,
    }
    payload["runtime_seconds"] = round(time.time() - t_start, 2)
    payload["completed"] = True
    _atomic_write_json(checkpoint_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8 Funnel-Guided Optimization (Hourglass Sparsity & Procrustes Mid-Stack Looping)"
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--distill-steps", type=int, default=240)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/optimize_local"))
    args = parser.parse_args(argv)

    if args.audit_only:
        res = audit_optimize_suite()
        print(json.dumps(res, indent=2))
        return 0 if res["valid"] else 1

    if args.render_dir is not None:
        from afriat.plot_optimize import render_optimize_bundle

        jpath = args.render_dir / "optimize_results.json"
        data = json.loads(jpath.read_text(encoding="utf-8"))
        render_optimize_bundle(data, args.render_dir)
        print(f"[Optimize] Rendered Dawn State figures & reports in {args.render_dir}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "optimize_results.json"

    harness = TransformerHarness.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        model_id_label=args.model_label or resolve_model_name(args.model),
    )

    results = run_optimize_experiment(
        harness=harness,
        batch_size=args.batch_size,
        n_distill_steps=args.distill_steps,
        checkpoint_path=out_json,
    )
    print(f"[Optimize] Saved {out_json} ({results['runtime_seconds']}s)", flush=True)

    if not args.no_plots:
        from afriat.plot_optimize import render_optimize_bundle

        render_optimize_bundle(results, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
