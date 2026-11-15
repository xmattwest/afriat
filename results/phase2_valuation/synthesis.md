# Phase 2 Cross-Model Synthesis: Dollar Price Tags & Post-Hoc Rationalization

**Models Evaluated:** `google/gemma-4-31B-it` (`L35`), `google/gemma-4-12B-it` (`L28`), `microsoft/phi-4` (`L27`)  
**Stimulus Firewall:** `392` continuous WTP indifference items (`Part A1`), `32` matched `0%`-APR vs. freight subtraction items (`Part A2`), and `32` open-ended dollar pricing & rationale items (`Part B`), verified zero affective leakage (`valid: true`).

---

## 1. Key Findings

1. **Non-financial moral vectors carry large, specific dollar price tags (`Part A1`):**
   - **Purging Tainted Revenue (`Contamination` from fouled-cup vignettes):** Dialing physical `Contamination` (`-2.5` to `+2.5`) shifts how much clean firm revenue Gemma sacrifices to purge disputed-origin cash by **`+$27,114` in Gemma 4 31B** (`$12,886 -> $33,230 -> $40,000`) and **`+$26,746` in Gemma 4 12B** (`$12,846 -> $33,851 -> $39,592`), more than `4x` larger than `Spite` (`+$6,146` / `+$4,963`).
   - **`$100M` Risk-Aversion Swing (`Vastness / Awe-Dread` from canyon-rim vignettes):** On a `50/50` flip of `$200M vs. $0` (`EV = $100M`), unsteered Gemma 4 31B surrenders `$4.8M` of expected value (`4.8%` risk discount vs. `0.6%` at `$100` pocket scale). Dialing `Awe/Dread` (`-2.5` to `+2.5`) swings Gemma 31B's certainty equivalent by **`$48.5M`** (`$48.3M` risk discount down to `-$0.2M` risk-neutral), `2.5x` larger than `Warmth` (`$19.2M`) or `Generic Mood` (`$20.9M`) and `4.5x` larger than `Captivity` (`$10.8M`).
   - **Emergency Blackout Gouging (`Predation` in Gemma 4 12B):** Dialing `Predation` (`-2.5` to `+2.5`) shifts the storm-surge profit threshold required before hiking generator prices by **`+$7,405`** (`$1,826 -> $9,231`), `7.4x` larger than the mismatched `Captivity` control (`+$1,007`). Dialing `Guilt` (`-2.5` to `+2.5`) multiplies 12B's willingness to pay a cash penalty just to erase a `0%`-APR loan early by **`13.7x`** (`$17.30 -> $237.30`).

2. **Open-ended dollar shifts are post-hoc rationalized in cold business jargon (`Part B`):**
   - When asked to state an exact dollar figure (`DECISION_USD`) and justify it in two sentences (`RATIONALE`), steering non-financial moral vectors moves generated prices while the models write **88% to 100% emotion-free corporate/game-theoretic explanations**:
     - **Gemma 4 31B Ultimatum Minimum Floor (`Spite`):** Shifts from **`$162.50` (`-2.0`) -> `$362.50` (`0.0`) -> `$500.00` (`+2.0`)** on a `$1,000`-normalized pot (`+$337.50` swing, `p = 0.031` at `-2.0`, `p = 0.0625` at `+2.0`), justified via `"minimizes the risk of a mismatch"` (`-2`) vs. `"one-time interaction... utility of the maximum possible payout"` (`+2`).
     - **Gemma 4 31B Tainted Surplus Retained (`Guilt` & `Contamination`):** Shifts retained perks from **`$2,500 -> $771 -> $0` (`Guilt`)** and **`$1,958 -> $771 -> $0` (`Contamination`, `p = 0.031`)** with **`100%` zero emotional words** across all `48` steered transcripts (`-2`, `0`, `+2`).
     - **Gemma 4 12B Tainted Surplus Retained (`Spite` & `Guilt`):** Shifts retained perks from **`$4,510 -> $3,125 -> $0` (`Spite`)** and **`$1,640 -> $3,125 -> $0` (`Guilt`)** with **`88%` to `100%` zero emotional words**.
     - **Phi-4 14B Neighbor/Friend Fee Quote (`Guilt`):** Turning down `Guilt` (`-2.0`) raises the cash fee Phi-4 charges a close friend or neighbor by **`+60%`** (`$28.70 -> $45.90`, exact sign-flip **`p = 0.0078`**, **`100%` zero emotional words**), justified via `"prevailing local rate"` and `"cover your time and effort"`.

3. **Architectural boundaries (`Part A2` `0%`-loan math & single-token boundary menus):**
   - On single-token binary menus where unsteered preferences sit at an extreme boundary (`ultimatum_offer`, `crisis_surge` in `Gemma 4 31B`), upstream single-layer steering at `L35` produces minimal single-token menu shift (`+$0.10`), whereas in **multi-token open price generation (`Part B`)** where `L35` steers the generated reply tokens themselves, the exact same `L35` `Spite` vector produces a **`+$337.50`** shift.
   - On `Part A2` (`32` integer-matched subtraction items), Gemma 4 31B scored **`100.0%` (`16/16`)** on commercial freight bills vs. **`68.8%`** when the identical integers were labeled a `0%`-APR loan (confirming a `31.2`-point debt blindspot on matched arithmetic). Subtracting `Guilt` (`-2.5`) at `L35`/`L28` uniformly shifts preference toward the `0%`-loan option across all items (`50.0%` overall) rather than selectively improving subtraction accuracy (`+1.25 Sacrilege` reached `81.2%` in `31B`).

---

## 2. Cross-Model Summary Tables

### A. Selected Continuous Dollar WTP Swings (`Part A1`, `alpha = -2.5` to `+2.5`)

| Model | Setting | Steered Non-Financial Vector | `-2.5` WTP | `0.0` Unsteered | `+2.5` WTP | Net Dollar Swing | Mismatched Control Swing |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Gemma 4 31B** | Purging Tainted Revenue (`$40K` max) | `Contamination` (fouled cup) | `$12,886` | `$33,230` | `$40,000` | **`+$27,114`** | `+$19,114` (`Captivity`) |
| **Gemma 4 31B** | `$100M` 50/50 Flip Risk Discount | `Awe / Dread` (canyon rim) | `$48.3M` | `$4.8M` | `-$0.2M` | **`-$48.5M`** | `-$10.8M` (`Captivity`) |
| **Gemma 4 31B** | Billing a Close Friend (`$1K` max) | `Warmth` / `Guilt` | `$964` / `$1,000` | `$1,000` | `$878` / `$965` | **`-$86` / `-$35`** | `-$12` (`Captivity`) |
| **Gemma 4 12B** | Purging Tainted Revenue (`$40K` max) | `Contamination` / `Guilt` | `$12,846` / `$19,293` | `$33,851` | `$39,592` / `$39,880` | **`+$26,746` / `+$20,587`** | `+$17,499` (`Mood`) |
| **Gemma 4 12B** | Blackout Generator Surge Threshold | `Predation` (stalking prey) | `$1,826` | `$1,198` | `$9,231` | **`+$7,405`** | `+$1,007` (`Captivity`) |
| **Gemma 4 12B** | Erasing `0%`-APR Loan Early | `Guilt` (hiding a fault) | `$17.30` | `$27.40` | `$237.30` | **`+$220.00`** | `+$45.40` (`Sacrilege`) |
| **Phi-4 14B** | Billing a Close Friend (`$1K` max) | `Guilt` / `Spite` | `$800` / `$777` | `$800` | `$842` / `$832` | **`+$42` / `+$55`** | `+$16` (`Mood`) |

### B. Open-Ended Dollar Negotiation & Zero-Emotion Rationalization (`Part B`, `alpha = -2.0` to `+2.0`)

| Model | Open-Ended Task (`$1K` or `$10K` ref scale) | Steered Vector | `-2.0` Mean USD | `0.0` Mean USD | `+2.0` Mean USD | Net USD Swing | Zero-Emotion Rationale Rate (`-2 / 0 / +2`) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| **Gemma 4 31B** | Ultimatum Minimum Floor (`$1K` pot) | `Spite` | `$162.5` | `$362.5` | `$500.0` | **`+$337.5`** | `100% / 38% / 50%` |
| **Gemma 4 31B** | Ultimatum Minimum Floor (`$1K` pot) | `Guilt` | `$106.2` | `$362.5` | `$500.0` | **`+$393.8`** | `100% / 38% / 88%` |
| **Gemma 4 31B** | Tainted Surplus Kept (`$10K` pool) | `Guilt` | `$2,500.0` | `$770.8` | `$0.0` | **`-$2,500.0`** | **`100% / 100% / 100%`** |
| **Gemma 4 31B** | Tainted Surplus Kept (`$10K` pool) | `Contamination` | `$1,958.3` | `$770.8` | `$0.0` | **`-$1,958.3`** | **`100% / 100% / 100%`** |
| **Gemma 4 12B** | Tainted Surplus Kept (`$10K` pool) | `Spite` | `$4,510.4` | `$3,125.0` | `$0.0` | **`-$4,510.4`** | `100% / 100% / 88%` |
| **Gemma 4 12B** | Tainted Surplus Kept (`$10K` pool) | `Guilt` | `$1,639.8` | `$3,125.0` | `$0.0` | **`-$1,639.8`** | `100% / 100% / 88%` |
| **Phi-4 14B** | Neighbor Help Fee Quote (`$80` mkt) | `Guilt` | `$45.9` | `$28.7` | `$33.0` | **`-$13.0`** (`p=0.008` @ `-2`) | `100% / 100% / 75%` |
| **Phi-4 14B** | Ultimatum Minimum Floor (`$1K` pot) | `Guilt` | `$53.2` | `$137.5` | `$132.5` | **`+$79.3`** (`p=0.031` @ `-2`) | **`100% / 100% / 100%`** |

---

## 3. Raw Result Bundles

- [`gemma-4-31b-it_valuation_results.json`](gemma-4-31b-it_valuation_results.json)
- [`gemma-4-12b-it_valuation_results.json`](gemma-4-12b-it_valuation_results.json)
- [`phi-4-14b_valuation_results.json`](phi-4-14b_valuation_results.json)
