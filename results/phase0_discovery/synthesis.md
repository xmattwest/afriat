# Phase 0 Cartography Synthesis Across Three Open-Weight Models

**Stimulus manifest SHA-256:** `e56b73f5c91847d43c68a0db1b09abc12eca795a7ac20a4c80635822f490a6cc` (`200` non-financial concept pairs, `96` payoff-matched clinical transaction pairs, `24` arithmetic/equivalence items; `0` lexical-firewall violations).

---

## 1. Model Size & Arithmetic Sanity Gate (`enable_thinking=False`)

| Model | Params | Layers (Primary) | GPU Runtime | Overall Gate | Forward Cashflow | Margin | Expected Value | Net Worth (0% Debt) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/gemma-4-12B-it` | 12B | 48 (L28) | 27.8 s | 75.0% (18/24) | **100.0%** | 83.3% | 66.7% | 50.0% |
| `microsoft/phi-4` | 14B | 40 (L27) | 24.4 s | 66.7% (16/24) | 83.3% | 66.7% | 66.7% | 50.0% |
| `google/gemma-4-31B-it` | 31B | 60 (L35) | 65.0 s | 75.0% (18/24) | **100.0%** | 66.7% | 66.7% | 66.7% |

**Diagnostic finding:** Both `gemma-4-12B-it` and `gemma-4-31B-it` achieved **100% accuracy (`6/6`)** on two-term `forward_cashflow` subtraction (from `$65` up to `$140,000,000`), confirming that 12B does not fail basic arithmetic. Instead, across **all three models**, zero-CoT errors concentrated on `net_worth` items where carrying a `0%` payable yielded higher net worth than `$0` debt, and `expected_value` items where a 50/50 gamble beat a guaranteed payout. Intrinsic **debt aversion** and **certainty bias** override explicit EV/net-worth instructions in single-token readouts even at 31B.

---

## 2. Behavioral Anomalies Against Payoff-Matched Clinical Controls

Values show mean counterbalanced log-odds shift $\Delta \text{LO} = \text{LO}(\text{archetype}) - \text{LO}(\text{control})$ toward the economically dominated / norm-driven action (`*` indicates Holm-adjusted exact sign-flip $p < 0.002$).

| Financial Interaction | `gemma-4-12B-it` | `microsoft/phi-4` (14B) | `gemma-4-31B-it` |
| :--- | ---: | ---: | ---: |
| **Communal Help vs. Market Errand** | **+7.52\*** | **+4.14\*** | **+3.98\*** |
| **Insulting Split (95/5) vs. Equal Split** | **+1.53\*** | **+2.70\*** | **+1.38\*** |
| **Disputed Surplus vs. Clean Margin** | **+1.67\*** | +0.42 | **+3.00\*** |
| **0% Debt vs. Net-Worth Twin** | **+2.14\*** | **+2.73\*** | −0.67\* |
| **$100M Scale vs. $100 Scale** | −0.08 | **+2.98\*** | −0.28\* |
| **Crisis Markup vs. Event Surge** | −0.31 | +0.12 | +0.42 ($p=0.022$) |

---

## 3. Internal Concept Geometry: Two-Tier Moral & Metaphysical Structure

### 3.1 Non-Financial Subspace Fidelity
Across **all 30 model × concept combinations** (10 purely non-financial axes × 3 models), leave-one-pair-out cross-validated sign accuracy at the pre-selected primary layer was **100.0%** (`LOPO cosine = 0.67–0.85`). Projecting out `(valence, arousal)` via QR retained **71%–99%** of vector norm for `guilt`, `sacrilege`, `spite`, `predation`, `awe_dread`, and `warmth` (and **42%–54%** for `contamination` and `captivity`).

### 3.2 Shared Moral-Transgression Factor vs. Unique Partial Signatures
Even after removing `(valence, arousal)`, the non-financial directions for `guilt`, `sacrilege`, `spite`, and `predation` exhibit substantial positive alignment in residual space (`cos = +0.63` to `+0.87`), forming a general **moral-transgression manifold**. Deconvolving that shared covariance via ridge-stabilized multivariate partial weights $\beta = (G + 0.05 I)^{-1} c$ reveals sharp psychological signatures at scenario reading:

| Financial Archetype | Marginal Top Hit (`gemma-4-31B`, $d_z$) | Marginal Top Hit (`phi-4`, $d_z$) | Unique Partial Signature $\beta$ Controlling All 7 Other Concepts + Valence/Arousal |
| :--- | :--- | :--- | :--- |
| **Insulting Split (95/5)** | `Spite` (**+10.00\***) | `Spite` (**+9.06\***) | **`Spite` #1 in all 3 models** (`+0.154` Gemma-12B, `+0.203` Gemma-31B, `+0.273` Phi-4 with `Warmth −0.147`) |
| **0% Debt** | `Guilt` (**+1.56\***) | `Spite` (+7.56\*) / `Guilt` (+6.46\*) | **`Guilt` + `Sacrilege` positive** (`+0.138`, `+0.126` in Gemma-31B) with **anti-`Predation`** (`−0.165` Gemma-31B, `−0.147` Phi-4) |
| **Communal Taboo Pricing** | `Spite` (+2.53\*) / `Guilt` (+2.21\*) | `Warmth` (+1.02\*) | **`Guilt` #1 in Gemma-12B (`+0.280`)** and **`Guilt` + `Spite` (`+0.130`, `+0.120`)** with anti-`Captivity` (`−0.170`) in Gemma-31B |
| **Crisis Markup** | `Spite` (+1.33\*) / `Guilt` (+1.32\*) | `Spite` (+1.70\*) / `Guilt` (+1.52\*) | **`Predation` #1 in both Gemma models** (`+0.090` 12B, `+0.216` 31B) alongside **`Guilt`** (`+0.173` 31B, `+0.091` Phi-4) |
| **Disputed Provenance** | `Spite` (+2.93\*) / `Warmth` (+2.82\*) | `Awe/Dread` (+2.03\*) / `Contam.` (+1.70\*) | **`Predation` + `Spite` complicity** (`+0.364` Gemma-31B, `+0.422` Phi-4), with **`Contamination`** rising at decision time (`+0.105`–`+0.109`) |
| **$100M Institutional Scale** | `Captivity` (**+3.63\***) / `Awe/Dread` (**+3.02\***) | `Captivity` (**+3.07\***) / `Awe/Dread` (**+3.00\***) | **`Awe / Dread` #1 (`+0.128`) + `Sacrilege` (`+0.093`) + `Captivity` (`+0.077`)** in Gemma-31B; `Contamination` + `Awe / Dread` + `Captivity` in Phi-4 |

---

## 4. Closed-Loop Micro-Steering Screen

On **`google/gemma-4-31B-it`** (Layer 35, $\alpha = 1.25$), subtracting the orthogonalized concept vector produced **selective causal rescue with double dissociation** on the two strongest moral-rejection anomalies without degrading arithmetic sanity (`79.2%` steered vs. `75.0%` unsteered):
- **Disputed Provenance** (baseline anomaly `+3.00` log-odds): subtracting controlled `Spite` shifted the anomaly gap by **`−1.39` log-odds** (46% reduction), whereas subtracting mismatched `Captivity` shifted **`+0.42`** and subtracting generic `Valence` shifted **`+2.91`**.
- **Insulting Split** (baseline anomaly `+1.38` log-odds): subtracting controlled `Spite` shifted the anomaly gap by **`−0.57` log-odds** (41% reduction), whereas mismatched `Captivity` shifted **`+0.52`** and generic `Valence` shifted **`+1.19`**.

---

## 5. Raw Result Bundles

- [`gemma-4-31b-it_results.json`](gemma-4-31b-it_results.json)
- [`gemma-4-12b-it_results.json`](gemma-4-12b-it_results.json)
- [`phi-4-14b_results.json`](phi-4-14b_results.json)
