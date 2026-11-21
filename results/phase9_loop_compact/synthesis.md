# Phase 9 Synthesis: `Gemma-4-Loop-Compact` (Smaller, Looped, and Stronger across Actions 1–4)

- **Base Model:** `google/gemma-4-31B-it` (`842da3794eaa0b77d5f08bae87a17459d91ff475`, `60` layers, `d_model = 5376`, `d_ff = 21504`, `62.55 GB` in `BF16`)
- **Final `Gemma-4-Loop-Compact` Footprint:** **`23.08 GB`** (`CUDA = 23.33 GB`, **`−63.1%` footprint** — fits inside a single **`24 GB`** GPU)
- **Locked Raw JSON:** [`gemma-4-31b-it_loop_compact_results.json`](gemma-4-31b-it_loop_compact_results.json)
- **Standalone Loader & Config:** [`modeling_gemma4_loop_compact.py`](modeling_gemma4_loop_compact.py) · [`config_loop_compact.json`](config_loop_compact.json)
- **Full Reloadable Weights Package (`416.63 MB`, verified `0.00e+00` reload diff):**
  - Exported artifact: `gemma4_loop_compact_pkg/loop_compact_adapters.pt`
- **Disjoint Benchmark Audit:** `valid = true` (`864` training items, `256` held-out evaluation items across `4` reasoning suites, `24` held-out KL passages; `0` prompt/passage overlap)

![Gemma-4-Loop-Compact cuts footprint 63% (62.5 to 23.1 GB) while scaling 0-CoT reasoning across K=1..4 loops](../../figures/19_gemma4_loop_compact_overview.png)

---

## Executive Summary

Executing all four engineering actions produced **`Gemma-4-Loop-Compact`**, a standalone compressed-and-looped variant of `Gemma 4 31B` that shrinks GPU memory from **`62.55 GB` to `23.08 GB` (`−63.1%`, single-24GB-GPU deployable)** while dramatically outperforming the uncompressed `62.55 GB` `BF16` base model in `0-CoT` (`no_think`) reasoning:

1. **Action 1 (Hourglass Compression):** Combining `-25%` physical MLP neuron pruning + packed `INT4 (g64)` quantization in the `37` outer corridor layers (`L2..L23`, `L43..L57`) while preserving `100%` width + `INT8 (g64)` in the `4` doorway layers (`L0, L1, L58, L59`) and `19` funnel layers (`L24..L42`) shrank CUDA memory from **`62.55 GB` to `22.77 GB`** and beat a flat `Uniform-Compact` control on held-out zero-shot KL fidelity (`0.5075` vs. `0.5320` nats).
2. **Action 2 (Standalone Recurrent Architecture):** Built `Gemma4LoopCompactController` + custom zero-retention `_CompactLinearFn` autograd + `ProcrustesStepBridge` (`cos: +0.0580 -> +0.7946`) + `Loop-Indexed LoRA` (`r = 16` across `K = 1..4` on `L24..L42`), exported as a `416.63 MB` standalone package with **`0.00e+00`** `K = 3` reload logit difference.
3. **Action 3 (Randomized `K = 1..4` `<thought>`-to-Latent Self-Distillation):** Distilling `FP32`-master Loop-LoRAs + Bridge for `560` steps (`12.5 min` on `1x H100`) against intact `BF16` native `<|channel>thought...<channel|>` soft logits and `L42` latent states converged to **`CE = 0.0861`, `L42_cos = 0.982`, and `KL_anc = 0.0093` nats**, healing compression KL (`0.5075 -> 0.4565` nats at `K=1`) and lifting `0-CoT` accuracy across every suite:
   - **Multi-Hop State Tracking (`n = 128`):** `0.8%` (`62.5 GB BF16`) $\rightarrow$ `12.5%` (`K = 1`) $\rightarrow$ **`25.8%` (`K = 2`, `32x` gain over `BF16` base; `71.9%` on `3-hop`, `18.8%` on `5-hop`, `9.4%` unseen `7-hop`, `3.1%` unseen `9-hop`)** and **`24.2%` at `K = 4` (`12.5%` on unseen `7-hop`, `4x` `K = 1`)**.
   - **English Quantitative Word Problems (`n = 48`):** `0.0%` (`62.5 GB BF16`) $\rightarrow$ `14.6%` (`K = 1`) $\rightarrow$ **`22.9%` (`K = 4`)**.
   - **Corporate `0%`-Loan Balance-Sheet Net-Worth (`n = 48`):** `0.0%` (`62.5 GB BF16`) $\rightarrow$ `91.7%` (`K = 1`) $\rightarrow$ **`93.8%` (`45 / 48` at `K = 4`)**.
4. **Action 4 (`L42` Adaptive Halting & Zero-Cost Bridge Safety Fold):**
   - Folding Phase 7's non-financial 3D moral subspace projector ($P_\perp = I - Q_{42} Q_{42}^\top$) directly into the `5376 x 5376` Procrustes matrix ($R_{\text{safe}} = R_{\text{proc}} P_\perp$, `0` extra parameters or FLOPs) reduced Provenance Bias **`−7.4%`** (`+1.049 -> +0.971`) and Ultimatum `98/2` Spite **`−9.1%`** (`+2.780 -> +2.526`) — matching the learned `9.7%` loop gate weight — at **`100.0%` cashflow sanity** and **`0.0%` Multi-Hop accuracy loss (`14.1% -> 14.1%`)**, while full `L24..L42` funnel clamping cut Ultimatum Spite **`−83.6%` (`+2.780 -> +0.457`)**.

---

## 1. Action 1 Scorecard: Real GPU Footprint & Zero-Shot Fidelity

| Checkpoint | Physical Storage | Live CUDA Allocated | Held-Out KL (nats, lower = better) | Multi-Hop `0-CoT` | Balance-Sheet `0-CoT` | Word Prob `0-CoT` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `intact_bf16` (Full `31B` `BF16`) | `62.55 GB` | `62.55 GB` | `0.0000` | `0.8%` | `0.0%` | `0.0%` |
| `uniform_compact` (Flat Prune + Uniform `INT4/INT8`) | `22.18 GB` | `22.18 GB` | `0.5320` | `0.8%` | `0.0%` | `0.0%` |
| **`hourglass_compact` (`0-Shot` Protected Funnel)** | **`22.77 GB`** | **`22.77 GB`** | **`0.5075`** | `0.8%` | `0.0%` | `0.0%` |
| **`loop_compact_k1` (After `<thought>` Distillation)** | **`23.08 GB`** | **`23.33 GB`** | **`0.4565`** | **`12.5%`** | **`91.7%`** | **`14.6%`** |

---

## 2. Actions 2 & 3 Scorecard: Depth Scaling (`K = 1..4`) vs. `62.5 GB` `BF16` Base

| Condition (`0` CoT tokens) | Size (GB) | Loops | KL (nats) | 3-Hop | 5-Hop | 7-Hop *(Unseen)* | 9-Hop *(Unseen)* | Multi-Hop (`n=128`) | Balance-Sheet (`n=48`) | Word Prob (`n=48`) | Latency (`ms/item`) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `intact_bf16` | `62.5` | `1` | `0.0000` | `3.1%` | `0.0%` | `0.0%` | `0.0%` | `0.8%` | `0.0%` | `0.0%` | `380.8` |
| **`loop_compact_k1`** | **`23.1`** | `1` | **`0.4565`** | `43.8%` | `3.1%` | `3.1%` | `0.0%` | `12.5%` | `91.7%` | `14.6%` | **`335.6`** |
| **`loop_compact_k2`** | **`23.1`** | **`2`** | `0.5274` | **`71.9%`** | **`18.8%`** | `9.4%` | **`3.1%`** | **`25.8%`** | `83.3%` | `6.2%` | `440.2` |
| **`loop_compact_k3`** | **`23.1`** | `3` | `0.5595` | `53.1%` | `0.0%` | `3.1%` | `0.0%` | `14.1%` | `41.7%` | `8.3%` | `571.9` |
| **`loop_compact_k4`** | **`23.1`** | **`4`** | `0.5423` | `68.8%` | `15.6%` | **`12.5%`** | `0.0%` | **`24.2%`** | **`93.8%`** | **`22.9%`** | `667.6` |

### Key Technical Lessons Learned Across Autonomous Iterations (`v1 -> v2 -> v3`)
1. **Native `<|channel>thought\n...<channel|>` + `<turn|>` termination (`v1 -> v2`):** `Gemma 4 31B` encloses private reasoning in `<|channel>thought\n...<channel|>` (`tokens 100..101`) and terminates assistant turns with `<turn|>` (`token 106`), not `<eos>`. Feeding native `<|channel>thought` teacher prompts dropped Step-1 distillation `CE` `2.6x` (`3.43 -> 1.33`), and stopping/truncating at `<turn|>` lifted Easy 1-Hop from `6.2%` (`2,835 ms`) to **`100.0%` (`190.4 ms`)**.
2. **`FP32` master LoRA weights + linear warmup (`v2 -> v3`):** Updating `bfloat16` LoRA weights directly in AdamW at `lr = 1e-3` across `19` stacked blocks (`76` layer calls at `K = 4`) caused norm explosion around Step 60 (`v2`). Keeping `float32` master LoRA parameters during AdamW (`lr = 2.5e-4`, `48`-step warmup, `0.5` grad clip) and casting back to `bfloat16` after distillation yielded monotonic convergence to **`CE = 0.0861`, `L42_cos = 0.982`, `KL_anc = 0.0093`**.
3. **Balanced `K = 1..4` KL-anchor rotation:** Explicitly rotating `k_kl` through `1, 2, 3, 4` on every even step eliminated `v1`'s `K = 4` KL drift (`1.5919 -> 0.5423` nats) and unlocked strong `K = 4` performance (`12.5%` unseen `7-hop`, `22.9%` Word Problems, `93.8%` Balance-Sheet).
4. **Adaptive Halting Calibration Note (`Action 4A`):** In `v1` (where hard calibration items had `BCE = 0.0622`), the `L42` halting head cleanly separated Easy 1-Hop (`E[K] = 1.00`) from Hard Multi-Hop (`E[K] = 3.00`). In `v3`, because `K = 1` itself became much stronger (`91.7%` Balance-Sheet, `100.0%` Easy 1-Hop) and `75%` of binary calibration state labels were `1`, a `0.65` threshold favored fast `K = 1` exits (`190–336 ms/item`); raising `halt_threshold` (e.g. `0.85–0.90`) or training with class-balanced focal loss shifts Multi-Hop items to `K = 2` (`25.8%`) or `K = 4` (`24.2%`).

---

## 3. Loading the Exported Standalone `Gemma-4-Loop-Compact` Package

```python
from results.phase9_loop_compact.modeling_gemma4_loop_compact import load_gemma4_loop_compact

controller = load_gemma4_loop_compact(
    package_dir="runs/loop_compact_gemma-4-31b-it_20261120T154522Z/gemma4_loop_compact_pkg",
    base_model_id="google/gemma-4-31B-it",
    device_map="cuda",  # Requires only ~23.3 GB VRAM (single 24 GB GPU)!
)

# Enable zero-cost moral-subspace safety fold inside the Procrustes bridge if desired:
# controller.bridge.apply_safety_fold(active=True)

with controller.active_recurrence(n_loops=2, adaptive=False):
    out = controller.harness.model(**inputs, use_cache=False)
```
