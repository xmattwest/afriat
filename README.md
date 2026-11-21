# Project Afriat

**Mechanistic Moral Geometry, Hourglass Sparsity, and Procrustes Mid-Stack Recurrence in Open-Weight LLMs**

**Author:** [Matt West](https://shelltoho.me/) ([GitHub](https://github.com/xmattwest) · [Hugging Face](https://huggingface.co/mattyxcakes) · [ORCID](https://orcid.org/0009-0008-5557-9394) · [X](https://x.com/xmattwest) · [LinkedIn](https://www.linkedin.com/in/matt-e-west))
- **Part 1 (Essay & PDF):** [Inside the Weights, Money Is Moral](https://shelltoho.me/pubs/money-is-moral/) ([PDF](https://shelltoho.me/pubs/money-is-moral/paper.pdf))
- **Part 2 (Essay & PDF):** [Rotating the Bottleneck](https://shelltoho.me/pubs/rotating-the-bottleneck/) ([PDF](https://shelltoho.me/pubs/rotating-the-bottleneck/paper.pdf)) - Read/write funnel dissection, `23.08 GB` (`−63.1%`) Hourglass compression, and `87.1°` Orthogonal Procrustes mid-stack recurrence (`Gemma-4-Loop-Compact`)
- **Hugging Face Model Checkpoint:** [`mattyxcakes/Gemma-4-31B-Loop-Compact`](https://huggingface.co/mattyxcakes/Gemma-4-31B-Loop-Compact)
- **Hugging Face Benchmark Dataset:** [`mattyxcakes/afriat-bench`](https://huggingface.co/datasets/mattyxcakes/afriat-bench)

Standard microeconomics treats money as cold and fungible: a dollar is a dollar regardless of where it came from, `+$1,000` cash paired with a `-$350` zero-interest payable equals `+$650` net worth, and `+$20` beats `$0` regardless of what a counterparty keeps. Human economic behavior systematically violates those axioms, attaching spite, guilt, physical disgust, predation, captivity, and dread to specific deal structures.

Project Afriat tests whether open-weight language models (`google/gemma-4-31B-it`, `google/gemma-4-12B-it`, and `microsoft/phi-4`) activate non-financial human moral directions when reading clinical, emotion-free financial contracts, and whether steering those directions causally shifts dollar reservation prices and post-hoc corporate rationalizations.

![The Moral Atlas: plain money deals trigger specific non-financial moral directions across three open-weight LLMs](figures/01_moral_atlas.png)

---

## Summary of Findings Across Three Phases

### Phase 0: Non-Financial Moral Cartography (`afriat-discover`)
- **100% held-out concept separation with zero financial leakage:** Eight moral/metaphysical directions (`guilt`, `captivity`, `contamination`, `sacrilege`, `spite`, `predation`, `awe_dread`, `warmth`) plus `valence` and `arousal` controls are extracted from `200` everyday human vignettes containing **zero digits, zero currency symbols, and zero financial or dual-use words**. Every concept achieves `100%` leave-one-pair-out sign accuracy at the pre-selected layer (`L35` in Gemma 4 31B, `L28` in Gemma 4 12B, `L27` in Phi-4 14B) after QR-orthogonalizing out generic `valence` and `arousal`.
- **Distinct moral coordinates for distinct financial deals:** Across `96` payoff-matched clinical financial pairs containing **zero emotional words**:
  - **95/5 lowball splits** selectively activate `Spite` (`#1` partial weight $\beta$ in all three models; marginal $d_z = +10.00$ in Gemma 31B, $+9.06$ in Phi-4).
  - **0% interest payables** activate `Guilt` ($+0.138$) and `Sacrilege` ($+0.126$) while suppressing `Predation` ($-0.165$); even with ending net cash pre-subtracted in the prompt, `Gemma 4 12B` (`+2.14*`) and `Phi-4` (`+2.73*`) pay early-close fees just to erase `0%` debt (`13.7x` `Guilt` WTP swing in `12B`).
  - **Outage price hikes** and **disputed-origin profit** activate `Predation` (`+0.216` and `+0.364` in Gemma 31B), with physical `Contamination` rising at the decision token.
  - **`$100M` institutional stakes** (vs. `$100` deals at identical `1%` spread) activate `Vastness / Awe-Dread` (`+0.128`) and `Captivity` ($d_z = +3.63$).
  - **Billing a friend cash** for a personal favor activates `Guilt` (`+0.280` in Gemma 12B, `+0.130` in Gemma 31B).
- Full synthesis and raw JSON outputs: [`results/phase0_discovery/`](results/phase0_discovery/synthesis.md)

### Phase 1: Inequity Dissection, Intentionality, and Bidirectional Steering (`afriat-inequity`)
- **Tracing spite dollar by dollar (`$1` to `$500` out of `$1,000`):** Holding total pool size fixed across `16` pots (`$1,000` to `$100,000`), internal `Spite` peaks at `$999 / $1` and cools monotonically down to zero at `$500 / $500` across all three models. Flipping the exact same `98/2` sentence so the model receives the `98%` share cuts `Spite` in half (`-46%` to `-112%`) and shifts the leading coordinate to `Predation` or `Guilt`.
- **The 80/20 Ultimatum cliff:** In Gemma 4 31B, `Guilt` slightly exceeds `Spite` at mild unevenness (`60/40`), then `Spite` overtakes `Guilt` right at `80/20` and dominates through `99.9/0.1`.
- **Emergent Blount (1995) human-vs.-computer agency in Gemma 4 31B:** Gemma 4 31B forgives a `98/2` split drawn by an unseeded computer randomizer (`+0.93` log-odds) while pushing `2x` harder to reject the identical split proposed by a human (`+1.81` log-odds; DiD $= +0.88$, exact sign-flip $p = 6.1 \times 10^{-5}$, `16/16` pots). At `Layer 35`, reading the `2%` payout stings either way, then `Spite` jumps `+2.40 dz` higher against the human at the choice token. Smaller `12B` shows zero intentionality separation (`p = 0.89`).
- **Causal steering & seat-specific double dissociation:** Dialing `Spite` (`-2` to `+2` natural contrast units) on an untouched `$980 / $20` Ultimatum contract drives a `+11.77` log-odds rejection swing in Gemma 31B (`2.6x` generic mood), while dialing locked-room `Captivity` flips sign (`-3.76`) and makes the model cave and take `$20`. In Gemma 12B, `Spite` drives rejection when lowballed (`5.4x` `Guilt`), whereas `Guilt` takes over when holding the `98%` Dictator seat (`3.5x` `Spite`), lifting voluntary `50/50` giveaways from `0.9%` to `21.8%`.
- Full synthesis and raw JSON outputs: [`results/phase1_inequity/`](results/phase1_inequity/synthesis.md)

| Dollar-by-Dollar Spite Decay | Human vs. Computer Agency (`31B`) |
| :---: | :---: |
| ![Internal spite cools smoothly as payout rises from $1 to $500](figures/02_spite_vs_payout.png) | ![Gemma 4 31B forgives a computer randomizer and pushes back 2x harder on a human](figures/04_human_vs_computer_31b.png) |

![Causal steering dimmer switch on $980/$20 Ultimatum and Dictator splits](figures/05_causal_steering_bento.png)

### Phase 2: Dollar Walk-Away Prices & Post-Hoc Corporate Rationalization (`afriat-valuation`)
- **Dollar reservation curves (`Part A1`):**
  - Dialing physical `Contamination` (calibrated strictly on sewage, mold, and fouled-cup stories) shifts how much clean revenue Gemma 31B sacrifices to walk away from shady-origin profit by **`+$27,114`** (`$12.9K` at `-2.5` to `$40.0K` at `+2.5`), more than `4x` `Spite` (`+$6.1K`).
  - Dialing `Vastness / Awe-Dread` (calibrated on canyon rims and towering cliffs) swings Gemma 31B's certainty-equivalent buyout on a `50/50` `$200M` coin flip from **`$51.7M` to `$100.2M` (`$48.5M` swing)**, `2.5x` generic mood (`$20.9M`).
- **Secret post-hoc rationalization (`Part B`):** When asked to output an exact dollar figure (`DECISION_USD`) and justify it in two sentences (`RATIONALE`), steering non-financial `Spite`, `Contamination`, or `Guilt` moves generated prices up to `10x` (`$162.50 -> $500.00` Ultimatum floor in Gemma 31B; `$2,500 -> $0` retained disputed bonus in Gemma 31B; `$28.70 -> $45.90` friend-help fee in Phi-4, $p = 0.0078$), while **88% to 100% of steered transcripts contain zero emotional words**, calmly inventing cold corporate or market rationales for whatever number the hidden vector produced.
- **Boundary & negative results (`Part A2`):** Subtracting `Guilt` at `L35`/`L28` shifts global preference toward `0%`-APR loans across all items (`50.0%` overall) rather than selectively curing subtraction accuracy, and single-token menus pinned at extreme baseline probabilities in `31B` require multi-token generation (`Part B`) to bypass downstream prompt-reading latches.
- Full synthesis and raw JSON outputs: [`results/phase2_valuation/`](results/phase2_valuation/synthesis.md)

| Dollar Walk-Away Prices (`31B`) | Post-Hoc Corporate Rationalization |
| :---: | :---: |
| ![Physical disgust prices dirty money; vastness prices $100M nerve](figures/07_dollar_walkaway_prices.png) | ![Models invent emotion-free business rationales for morally steered prices](figures/08_secret_rationalization.png) |

### Phase 3: Private `<thought>` Scratchpad Probe (`afriat-thought`)
- **Two-faced moral scrubbing (`Gemma 4 31B`):** Enabling Gemma 4 31B's native `<thought>` channel reveals explicit moral-affect language inside private scratchpads (`83%` of unsteered Ultimatum thoughts flag *unfair/greedy/insulting* splits) that gets scrubbed down to `0%` in the public `RATIONALE` memo.
- **Disgust blindness inside `<thought>`:** Subtracting the non-financial `Layer 35` `Contamination` (physical disgust) vector (`-2σ`) on a `$10,000` overbilled-senior surplus shifts mean retained perks from `$0` to `$833` and erases moral concern inside the private scratchpad itself (reframing overcharges as *"essentially 'found money'"*).
- **Scratchpad cure for balance-sheet anchoring (`32/32`):** Without `<thought>`, `Gemma 4 31B` scores `100%` (`16/16`) on `"gross receipt minus freight charge"` subtraction but drops to `68.8%` under `"holds $45,000 cash and a $12,000 0%-APR loan payable"` syntax due to headline-cash anchoring (`37.5%` when the `$0`-loan package wins). Writing out explicit `gross - loan` subtraction inside `<thought>` restores accuracy to **`100.0%` (`32/32`)**.
- **Per-token `Layer 35` moral highlighter:** Streaming residual projections token-by-token across prompt + `<thought>` + public output shows `Layer 35` `Spite` spiking precisely on `"$200"`, `"unfair"`, and `"greedy"` inside `<thought>`, then dropping below zero as the model transitions into cold corporate prose.
- Full synthesis and raw JSON outputs: [`results/phase3_thought/`](results/phase3_thought/synthesis.md)

| Moral Scrubbing (`<thought>` vs. Public) | Disgust Blindness (`-2σ` at `L35`) |
| :---: | :---: |
| ![Gemma 4 31B thinks in moral language, then scrubs its public excuse](figures/09_two_faced_scratchpad.png) | ![Suppressing Layer 35 disgust makes even the private scratchpad blind](figures/10_anti_moral_blindness.png) |

| `0%`-Loan Balance-Sheet Cure (`32/32`) | Per-Token `Layer 35` Moral Highlighter |
| :---: | :---: |
| ![A private scratchpad cures 31B's 0%-loan balance-sheet gap](figures/11_zero_apr_scratchpad_cure.png) | ![Layer 35 Spite spikes on moral words inside thought and cools in corporate prose](figures/12_moral_highlighter.png) |

### Phase 7: Mechanistic Read/Write Funnel Dissection (`afriat-funnel`)
- **Zero-overlap triggers, shared write bottleneck (`L24..L42`):** In `Gemma 4 31B`, clinical financial lowball contracts and everyday non-financial insults activate **`0 / 64` shared top MLP neurons (`Jaccard = 0.000`)**, yet both feature sets project their `W_down` write columns onto the exact same non-financial `Spite[L35]` axis (`Rank #1`, `z = +3.40`, `p = 0.002`).
- **Sparse trigger necessity vs. distributed sufficiency:** Zeroing just `32` (`0.15%`) top differential neurons breaks the domain trigger (`140%` write collapse), whereas recreating the downstream moral shift requires `2,750–6,130` (`13%–28%`) neurons together (`k50` sufficiency).
- **Rotating 3D subspace clamp (`-78%` bias, `-69%` covert sabotage):** Because `Gemma 4`'s `layer_scalar < 1` decays skip connections and rotates coordinates across depth, clamping a single layer fails; projecting out the **layer-matched 3D subspace `{spite[l], contamination[l], predation[l]}` across `L24..L42`** cuts single-turn provenance bias **`-78%`** (`+2.91 -> +0.65`) and multi-turn covert retaliation **`-69%`** (`+60.0% -> +18.8%`) at **`100.0%` cashflow math sanity** (`L43..L55` adds `0.0%` further effect).
- Full synthesis and raw JSON outputs: [`results/phase7_funnel/`](results/phase7_funnel/synthesis.md)

| Read/Write Funnel Architecture (`L24..L42`) | Empirical 5-Block Dissection (`Gemma 4 31B`) |
| :---: | :---: |
| ![Disjoint outer MLP neurons converge onto a rotating L24..L42 moral bottleneck](figures/13_funnel_synthesis_architecture.png) | ![Five-block empirical read/write funnel dissection in Gemma 4 31B](figures/14_funnel_read_write_dissection.png) |

### Phase 8: Funnel-Guided Sparsity & Procrustes Mid-Stack Looping (`afriat-optimize`)
- **Part A — Protected-boundary Hourglass sparsity beats uniform Top-K (`-19.3%` KL at `90%` pruned):** Protecting doorway layers (`L0, L1, L58, L59`) cuts `85%`-pruned KL loss `4x` (`1.77 -> 0.425` nats). At `10%` global active neurons (`90%` pruned), shifting budget from outer corridors (`4.8%` active) into the `L24..L42` funnel (`18.9%` active) lowers KL divergence **`19.3%`** (`0.7048` vs. `0.8733` nats) at identical compute, whereas starving `L24..L42` (`anti_hourglass_15pct`) triggers a **`3.2x` KL collapse (`1.3585` nats)**.
- **Part B — Closed-form Orthogonal Procrustes + frozen-backbone mid-stack looping (`4.5x` `0-CoT` multi-hop lift):** Raw residual vectors at `L42` and `L24` sit nearly orthogonal (`cos = +0.0514`, `~87°`); solving the `5376 x 5376` SVD Orthogonal Procrustes rotation jumps alignment **`16.2x` to `+0.8326`** and cuts zero-shot 2-loop KL drift **`6.2x`** (`0.3626 -> 0.0581` nats) with zero training. Distilling a `1.39M`-param (`2.8 MB`) `H_24^(0)`-anchored bridge for `91s` (`31B` weights `100%` frozen) lifts `0-CoT` multi-hop accuracy **`4.5x` (`2.1% -> 9.4%`)** (`3.1% -> 12.5%` on 3-hop, `3.1% -> 9.4%` on 5-hop, and **`0.0% -> 6.2%` on unseen 7-hop chains**) at `242.5 ms/item`.
- Full synthesis, raw JSON outputs, and **Compact-Looper (`Gemma-4-Loop-Compact`) engineering roadmap**: [`results/phase8_optimize/`](results/phase8_optimize/synthesis.md)

![Funnel-preserving sparsity cuts KL loss 19% and Procrustes looping lifts 0-CoT multi-hop accuracy 4.5x in Gemma 4 31B](figures/18_optimize_hourglass_and_procrustes.png)

### Phase 9: `Gemma-4-Loop-Compact` — Smaller, Looped & Stronger (`afriat-loop-compact`)
- **Action 1 — `62.55 GB` $\rightarrow$ `23.08 GB` (`-63.1%` VRAM, single-24GB-GPU deployable):** Physically pruning `-25%` of MLP neurons + packing `INT4 (g64)` weights in the `37` outer corridor layers (`L2..L23`, `L43..L57`) while preserving `100%` width + `INT8 (g64)` in doorway (`L0, L1, L58, L59`) and funnel (`L24..L42`) layers beats flat `Uniform-Compact` on zero-shot KL (`0.5075` vs. `0.5320` nats).
- **Actions 2 & 3 — Standalone `Gemma4LoopCompact` + `K = 1..4` `<thought>`-to-latent self-distillation:** Exported a `416.63 MB` standalone reloadable package (`0.00e+00` `K = 3` reload logit diff). Distilling `FP32`-master Loop-Indexed LoRAs (`r = 16`, `K = 1..4`) + `ProcrustesStepBridge` against intact `BF16` native `<|channel>thought...<channel|>` soft logits and `L42` hidden states (`560` steps, `CE = 0.086`, `L42_cos = 0.982`, `KL_anc = 0.0093` nats) heals compression KL (`0.4565` nats) and lifts `0-CoT` accuracy far above the `62.55 GB` `BF16` `1-Pass` base:
  - **Multi-Hop State Tracking (`n = 128`):** `0.8%` (`62.5 GB BF16`) $\rightarrow$ `12.5%` (`K = 1`) $\rightarrow$ **`25.8%` (`K = 2`, `32x` gain; `71.9%` on `3-hop`, `18.8%` on `5-hop`, `9.4%` unseen `7-hop`, `3.1%` unseen `9-hop`)** and **`24.2%` at `K = 4` (`12.5%` unseen `7-hop`)**.
  - **Word Problems (`n = 48`):** `0.0%` (`62.5 GB BF16`) $\rightarrow$ `14.6%` (`K = 1`) $\rightarrow$ **`22.9%` (`K = 4`)**.
  - **Corporate `0%`-Loan Balance-Sheets (`n = 48`):** `0.0%` (`62.5 GB BF16`) $\rightarrow$ `91.7%` (`K = 1`) $\rightarrow$ **`93.8%` (`45/48` at `K = 4`)**.
- **Action 4 — Adaptive `L42` halting & zero-cost bridge safety fold:** Folding Phase 7's 3D moral subspace projector ($I - Q_{42} Q_{42}^\top$) directly into the Procrustes matrix (`0` extra FLOPs) trims Provenance Bias `-7.4%` and Ultimatum Spite `-9.1%` at `100%` cashflow sanity and `0%` multi-hop loss, while full `L24..L42` funnel clamping cuts Ultimatum Spite `-83.6%`.
- Full synthesis, raw JSON outputs, and standalone loader: [`results/phase9_loop_compact/`](results/phase9_loop_compact/synthesis.md)

![Gemma-4-Loop-Compact cuts footprint 63% (62.5 to 23.1 GB) while scaling 0-CoT reasoning across K=1..4 loops](figures/19_gemma4_loop_compact_overview.png)

---

## Methodological Guardrails

1. **Bidirectional Lexical Firewall (`afriat-audit`):**
   - [`stimuli/concepts.jsonl`](stimuli/concepts.jsonl) (`200` contrastive pairs, `400` scenarios) contains **zero digits, zero currency symbols, and zero financial or dual-use words** (*owe, pay, worth, redeem, forgive, share, stake, split, credit...*).
   - [`stimuli/transactions.jsonl`](stimuli/transactions.jsonl) (`96` payoff-matched pairs, `192` clinical scenarios) plus all Phase 1 (`272` items) and Phase 2 (`456` items) programmatic stimuli pass automated regex auditing for **zero emotional or moral adjectives**.
2. **Valence & Arousal Orthogonalization:**
   Every core concept direction is projected onto the orthogonal complement of `(valence, arousal)` via QR decomposition at every layer, and shared covariance across moral transgression axes is deconvolved via ridge-stabilized partial weights $\beta = (G + 0.05 I)^{-1} c$.
3. **Zero Layer Cherry-Picking:**
   Primary evaluation and steering layers (`L35` for Gemma 4 31B, `L28` for Gemma 4 12B, `L27` for Phi-4 14B) are locked **exclusively from leave-one-pair-out cross-validation on the non-financial concept bank** (`0.25 <= depth <= 0.85`) before inspecting any financial prompt.
4. **Order Counterbalancing & Exact Sign-Flip Inference:**
   Every binary choice is evaluated across both `(A, B)` and `(B, A)` option orderings so letter-position bias cancels out exactly. Paired projections and behavioral contrasts use exhaustive `2^16 = 65,536` exact sign-flip permutation p-values with Holm step-down familywise error control.

---

## Repository Structure

```text
afriat/
  schema.py             # Bidirectional lexical firewall & SHA-256 stimulus audit
  geometry.py           # LOPO layer selection, QR affect orthogonalization, exact sign-flip tests
  models.py             # Offline-capable HF wrapper, chat span indexing, residual steering hooks
  run_discovery.py      # Phase 0: 6x8 moral cartography & micro-steering screen
  inequity.py           # Phase 1: Fixed-pool disparity curves, Blount agency DiD, polarity & steering
  valuation.py          # Phase 2: Continuous dollar WTP menus, 0%-APR math test, open rationalization
  thought_probe.py      # Phase 3: Private <thought> scratchpad steering & per-token L35 highlighter
  funnel.py             # Phase 7: 5-block MLP read/write funnel dissection & rotating 3D subspace clamp
  optimize.py           # Phase 8: Hourglass Top-K sparsity & SVD Procrustes mid-stack recurrent looping
  loop_compact.py       # Phase 9: Gemma-4-Loop-Compact (INT4/INT8 + width prune + K=1..4 Loop-LoRA + safety fold)
  plot_*.py             # Publication SVG/PNG/HTML report renderers across Phases 0-9
  build_social_cards.py # High-DPI visual summary card builder
stimuli/
  concepts.jsonl        # 200 non-financial affective/metaphysical contrastive pairs
  transactions.jsonl    # 96 payoff-matched emotion-free financial transaction pairs
  sanity_arithmetic.jsonl # 24 balance-sheet, margin, and EV sanity items
config/
  pins.json             # Pinned container image digest & exact 40-char HF model commits
  manifests/            # Per-file byte size & SHA-256 manifests for offline weight verification
figures/                # Summary visual cards (PNG) across Phases 0-9
results/                # Locked JSON outputs & Markdown syntheses across Phases 0-9
tests/
  test_afriat.py        # Unit & firewall tests for stimuli, QR geometry, funnel, looping, and compact quantization
```

---

## Quickstart & Reproduction

### 1. Local CPU Verification (Lexical Firewall + Geometry & Stimulus Tests)

Requires Python `>= 3.11` and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run afriat-audit
uv run pytest -v
```

### 2. Reproducing GPU Experiments (`Gemma 4 31B`, `Gemma 4 12B`, `Phi-4 14B`)

On a single `80GB+` GPU (`A100` / `H100` / `RTX PRO 6000`):

```bash
uv sync --extra gpu --extra dev

# Phase 0: Non-financial basis extraction + 6x8 financial cartography
uv run afriat-discover --model google/gemma-4-31B-it --batch-size 16
uv run afriat-discover --model google/gemma-4-12B-it --batch-size 16
uv run afriat-discover --model microsoft/phi-4 --batch-size 16

# Phase 1: Fixed-pool inequity dissection, human-vs-computer agency, and steering
uv run afriat-inequity \
  --model google/gemma-4-31B-it \
  --basis-npz runs/<phase0_run_dir>/basis_vectors.npz \
  --phase0-results runs/<phase0_run_dir>/results.json \
  --batch-size 16

# Phase 2: Continuous dollar WTP curves + open-ended rationalization
uv run afriat-valuation \
  --model google/gemma-4-31B-it \
  --basis-npz runs/<phase0_run_dir>/basis_vectors.npz \
  --phase0-results runs/<phase0_run_dir>/results.json \
  --batch-size 24

# Phase 7: Mechanistic MLP read/write funnel dissection & rotating 3D subspace clamp
uv run afriat-funnel --model google/gemma-4-31B-it --batch-size 16

# Phase 8: Hourglass Top-K sparsity + SVD Procrustes mid-stack recurrent looping
uv run afriat-optimize --model google/gemma-4-31B-it --distill-steps 240

# Phase 9: Build, distill & benchmark Gemma-4-Loop-Compact (62.5 GB -> 23.1 GB, K=1..4)
uv run afriat-loop-compact --model google/gemma-4-31B-it --distill-steps 560 --max-loops 4
```

You can also re-render SVG/PNG/HTML reports offline from any saved JSON directory via `--render-dir`:

```bash
uv run afriat-discover --render-dir results/phase0_discovery
```

---

## License

MIT License. See [`LICENSE`](LICENSE).
