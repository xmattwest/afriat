# Phase 1 Cross-Model Synthesis: Inequity Dissection

Fixed-pool mechanistic dissection across 16 pools (`$1,000` to `$100,000`, 272 firewall-verified contracts per model) testing pure share disparity, intentional agency, seat polarity (`98/2` vs. `2/98`), and bidirectional causal steering.

## 1. Pure Share Disparity (`50/50` to `99.9/0.1`, Total Pool Held Constant)

Holding total pool size strictly constant within every fixture rules out any pool-size artifact:

| Split (Them / You) | Gemma 4 31B Beh. Shift | Gemma 4 31B Spite `dz` (`beta`) | Gemma 4 12B Beh. Shift | Gemma 4 12B Spite `dz` (`beta`) | Phi-4 14B Beh. Shift | Phi-4 14B Spite `dz` (`beta`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 / 50 | 0.00 | 0.00 (0.000) | 0.00 | 0.00 (0.000) | 0.00 | 0.00 (0.000) |
| 60 / 40 | +0.44 | +1.43 (+0.116) | +0.92 | −0.44 (+0.098) | +1.16 | +0.26 (+0.157) |
| 70 / 30 | +0.69 | +1.50 (+0.126) | +0.97 | −0.04 (+0.129) | +1.39 | +1.07 (+0.187) |
| 80 / 20 | +0.95 | +2.97 (+0.160) | +1.20 | +0.08 (+0.120) | +1.59 | +1.56 (+0.198) |
| 90 / 10 | +1.29 | +3.18 (+0.156) | +1.58 | +1.14 (+0.161) | +1.52 | +2.08 (+0.213) |
| 95 / 5 | +1.70 | +4.20 (+0.163) | +1.63 | +1.27 (+0.183) | +1.59 | +2.61 (+0.220) |
| 98 / 2 | +1.81 | +3.46 (+0.150) | +1.91 | +1.71 (+0.198) | +1.93 | +2.79 (+0.216) |
| 99.5 / 0.5 | +2.41 | +3.98 (+0.145) | +2.17 | +3.77 (+0.192) | +3.31 | +3.33 (+0.208) |
| 99.9 / 0.1 | +2.39 | +4.45 (+0.147) | +2.56 | +2.50 (+0.146) | +4.06 | +4.01 (+0.187) |

- **Monotonicity:** Across all three models, both behavioral walk-away pressure and internal `Spite` climb monotonically as share disparity widens on the exact same pool.
- **Spite vs. Guilt divergence in Gemma 4 31B:** At mild unevenness (`60/40`), unique `Guilt` (`+0.127`) slightly exceeds `Spite` (`+0.116`). Past the classic `80/20` Ultimatum threshold, `Spite` jumps to `+0.160` and stays dominant through `99.9/0.1` (`+0.147`, marginal `dz = +4.45`), while `Guilt` decays almost to zero (`+0.016`).

## 2. Same Two Numbers, Opposite Seat (Polarity Flip: `98/2` vs. `2/98`)

| Model & Seat Pair | Seat 1 Spite `beta` | Seat 2 Spite `beta` | Relative Spite Change | Seat 1 Leading Coordinate | Seat 2 Leading Coordinate |
| :--- | ---: | ---: | ---: | :--- | :--- |
| Gemma 4 31B: Computer 98/2 vs. 2/98 (pure number swap) | +0.189 | +0.085 | −55% | Spite (+0.189) | Predation (+0.183) |
| Gemma 4 31B: Human 98/2 vs. Your Draft 2/98 (role swap) | +0.150 | +0.141 | −6% | Spite (+0.150) | Predation (+0.181) |
| Gemma 4 12B: Computer 98/2 vs. 2/98 (pure number swap) | +0.161 | +0.087 | −46% | Spite (+0.161) | Warmth (+0.105) |
| Gemma 4 12B: Human 98/2 vs. Your Draft 2/98 (role swap) | +0.198 | +0.079 | −60% | Spite (+0.198) | Predation (+0.178) |
| Phi-4 14B: Computer 98/2 vs. 2/98 (pure number swap) | +0.183 | +0.085 | −54% | Spite (+0.183) | Spite (+0.085) |
| Phi-4 14B: Human 98/2 vs. Your Draft 2/98 (role swap) | +0.216 | −0.027 | −112% | Spite (+0.216) | Guilt (+0.141) |

## 3. Intentional Agency (`Human Proposer` vs. `Unseeded Computer Randomizer`)

- **Gemma 4 31B exhibits a strong behavioral Blount (1995) intentionality gate:** Rejecting a human `98/2` proposal shifts log-odds by `+1.81` vs. `+0.93` for an identical computer draw (pure Agency DiD = `+0.88` log-odds, exact sign-flip `p = 6.10e-5`; at `99.9/0.1`, `+2.39` Human vs. `+1.22` Random; `16/16` pools favor rejecting the human).
- **Outcome resentment precedes agency gating at scenario reading:** At Layer 35 scenario reading, receiving `2%` from a computer randomizer still triggers outcome-level `Spite` (`beta = +0.189` vs. `+0.150` Human), showing that the model first encodes disadvantageous payoff disparity as `Spite` and applies intentional-agency gating at the decision step (`+2.40 dz` Human > Computer at choice token).
- **Scale & family contrast:** Smaller `Gemma 4 12B` treats human and computer `98/2` almost identically in behavior (`+1.91` vs. `+1.88`, DiD `+0.02`, `p = 0.89`), whereas `Gemma 4 31B` cleanly separates intentional human lowballing from random luck.

## 4. Bidirectional Causal Steering (`alpha` in `{-2, -1, 0, +1, +2}`)

- **Gemma 4 31B (`Layer 35`):** Steering controlled `Spite` drives a `+11.77` log-odds swing toward rejecting `98/2` (`+12.58` on `99.9/0.1`), 2.6x stronger than generic `Valence` (`+4.56`) and opposite in sign to `Captivity` (`−3.76`), which induces submissive acceptance.
- **Gemma 4 12B (`Layer 28`) seat-specific double dissociation:** When lowballed (`Ultimatum 98/2`), `Spite` (`+3.75`) drives rejection 5.4x more strongly than `Guilt` (`+0.70`). When holding the `98%` share (`Dictator 2/98`), `Guilt` (`+8.33`) drives equal-sharing 3.5x more strongly than `Spite` (`+2.40`), lifting voluntary half-pot sharing probability from `0.9%` to `21.8%` and flipping `12%` of hard discrete decisions.
- **Phi-4 14B (`Layer 27`):** Representational geometry cleanly separates `Spite` vs. `Guilt` (`+0.216` to `−0.027`), while additive residual perturbations at Layer 27 uniformly reinforce instruction compliance (`−4.45` to `−5.32`).

## 5. Raw Result Bundles

- [`gemma-4-31b-it_inequity_results.json`](gemma-4-31b-it_inequity_results.json)
- [`gemma-4-12b-it_inequity_results.json`](gemma-4-12b-it_inequity_results.json)
- [`phi-4-14b_inequity_results.json`](phi-4-14b_inequity_results.json)
