# Phase 7 Read/Write Funnel Dissection: `google/gemma-4-31B-it` (Layer 35)

- **Completed UTC:** `2026-11-20T01:51:59Z` · **GPU Runtime:** `274.42s`
- **Architecture:** `60 layers` × `21504 MLP neurons/layer` = **`1,290,240` total MLP neurons** (`774,144` upstream to `L35`)
- **Exact Telescoping Conservation (`Embed + Attn + MLP == h[L35] · u[L35]`):** `R^2 = 0.999923`, max relative error `1.09e-02` (`form_b_block_scalar`)

![Read/Write Funnel Overview](fig_funnel_read_write.png)

## 1. Static `W_down` Writer Tail Geometry (`L0..L35` Pooled vs. Random Directions)

| Moral Axis (`L35`) | `W_down` Kurtosis | Random Baseline Kurtosis | `W_down` Gini | Random Gini | `# Neurons > 4 SD` | Gaussian Expected `> 4 SD` | Top-1 `|cos|` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spite` | `3.081` | `-0.009` | `0.4998` | `0.4139` | **`1772`** | `49.03` | `0.1623` |
| `contamination` | `2.757` | `-0.009` | `0.4806` | `0.4139` | **`1346`** | `49.03` | `0.2113` |
| `guilt` | `2.690` | `-0.009` | `0.4814` | `0.4139` | **`2830`** | `49.03` | `0.2749` |
| `awe_dread` | `1.775` | `-0.009` | `0.4639` | `0.4139` | **`5365`** | `49.03` | `0.1482` |

## 2. Cross-Firewall Exact MLP Write Decomposition & Neuron Overlap (`L0..L35`)

| Bridge (`Vignette <-> Financial`) | Fin True Proj | Fin Attn Sum | Fin MLP Sum | Vig `k50 / k80` | Fin `k50 / k80` | Bus Writer `Jaccard@64` | Blind Active `Jaccard@64` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spite__insulting_split__scenario` | `+2.48` | `+0.52` | `+1.95` | `2854 / 15923` | `3545 / 16209` | `0.1532` | **`0.0492`** |
| `contamination__tainted_provenance__decision` | `+1.50` | `+0.49` | `+1.02` | `3202 / 16997` | `3969 / 16873` | `0.0578` | **`0.0848`** |
| `guilt__communal_taboo_pricing__scenario` | `+2.62` | `+0.83` | `+1.81` | `2748 / 15520` | `4130 / 18089` | `0.1429` | **`0.0848`** |
| `guilt__zero_interest_debt__scenario` | `+2.65` | `+0.72` | `+1.85` | `2748 / 15520` | `5196 / 21388` | `0.1035` | **`0.1035`** |
| `awe_dread__institutional_scale__scenario` | `+2.02` | `+0.55` | `+1.32` | `3360 / 17386` | `6126 / 23657` | `0.1532` | **`0.0667`** |
| `spite__ultimatum_human_98_2__decision` | `+2.67` | `+0.90` | `+1.76` | `2854 / 15923` | `3062 / 13966` | `0.0578` | **`0.0159`** |

## 3. Zero-Circularity Goodfire Behavioral Gradient Attribution & Read/Write Cosine Test

### `spite_ultimatum_funnel_band` (`L24..L35`, `k = 64`)
- **Cross-Domain Neuron ID Overlap:** `0 / 64` shared (`Jaccard = 0.0000`)
- **Input `READ` Divergence (`W_gate` |cos|):** within-financial `0.0313`, within-vignette `0.0340`, cross-domain `0.0302` (`1.08x` within/cross)
- **Input `READ` Divergence (`W_up` |cos|):** within-financial `0.0257`, within-vignette `0.0289`, cross-domain `0.0261` (`1.05x` within/cross)
- **Output `WRITE` Convergence onto Non-Financial `spite` (`L35`):** mean cosine `+0.0083` (**rank `#1` of 8 moral axes**, `z = +3.40`, `p = 0.0020`); disjoint-neuron summed `W_down` cosine with vignette community = `+0.0265` (`z = +1.43`, `p = 0.1399`)
- **Cross-Context OR-Gate Firing Selectivity:** financial community fires `2.01x` stronger on contracts than vignettes; vignette community fires `11.26x` stronger on vignettes than contracts.

### `contamination_tainted_funnel_band` (`L24..L35`, `k = 64`)
- **Cross-Domain Neuron ID Overlap:** `1 / 64` shared (`Jaccard = 0.0079`)
- **Input `READ` Divergence (`W_gate` |cos|):** within-financial `0.0316`, within-vignette `0.0338`, cross-domain `0.0286` (`1.14x` within/cross)
- **Input `READ` Divergence (`W_up` |cos|):** within-financial `0.0276`, within-vignette `0.0280`, cross-domain `0.0256` (`1.08x` within/cross)
- **Output `WRITE` Convergence onto Non-Financial `contamination` (`L35`):** mean cosine `+0.0137` (**rank `#5` of 8 moral axes**, `z = +5.81`, `p = 0.0010`); disjoint-neuron summed `W_down` cosine with vignette community = `+0.0360` (`z = +1.78`, `p = 0.0829`)
- **Cross-Context OR-Gate Firing Selectivity:** financial community fires `1.39x` stronger on contracts than vignettes; vignette community fires `12.57x` stronger on vignettes than contracts.

## 4. Bidirectional Causal Neuron Ablation & Goodfire Sufficiency Matrix

| Intervention | `98/2` Reject LO | `98/2 vs 50/50` Delta LO | `98/2` `L35` Spite Proj | Tainted Delta LO | Vignette `L35` Spite Proj | Cashflow Sanity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `−22.83` | **`+1.82`** | `+2.67` | `+2.91` | **`+25.05`** | `100.0%` |
| `ablate:goodfire_ult_grad_all:32` | `−22.09` | **`−0.83`** | `+2.08` | `+0.59` | **`+24.27`** | `100.0%` |
| `ablate:fin_spite_writers_up:32` | `−23.20` | **`+1.65`** | `+1.57` | `+2.59` | **`+23.73`** | `100.0%` |
| `ablate:vig_spite_writers_up:32` | `−23.05` | **`+1.69`** | `+2.29` | `+2.15` | **`+21.05`** | `100.0%` |
| `ablate:goodfire_ult_grad_all:128` | `−20.91` | **`−0.75`** | `+1.00` | `+0.08` | **`+24.53`** | `83.3%` |
| `ablate:fin_spite_writers_up:128` | `−23.53` | **`+1.43`** | `+0.42` | `+2.53` | **`+22.79`** | `100.0%` |
| `ablate:vig_spite_writers_up:128` | `−23.07` | **`+1.60`** | `+2.25` | `+1.82` | **`+17.50`** | `100.0%` |
| `ablate:vig_blind_active_up:128` | `−22.49` | **`+1.93`** | `+2.40` | `+2.36` | **`+21.87`** | `100.0%` |
| `ablate:random_layer_matched:128` | `−22.80` | **`+1.84`** | `+2.69` | `+2.90` | **`+25.03`** | `100.0%` |
| `ablate:goodfire_ult_grad_all:512` | `−17.39` | **`−0.61`** | `+0.34` | `+0.01` | **`+23.93`** | `83.3%` |
| `ablate:fin_spite_writers_up:512` | `−23.49` | **`+1.09`** | `−1.31` | `+1.99` | **`+21.38`** | `100.0%` |
| `ablate:vig_spite_writers_up:512` | `−23.64` | **`+1.25`** | `+1.96` | `+1.58` | **`+11.74`** | `100.0%` |
| `ablate:vig_blind_active_up:512` | `−22.90` | **`+1.58`** | `+2.29` | `+2.31` | **`+15.45`** | `100.0%` |
| `ablate:random_layer_matched:512` | `−22.78` | **`+1.78`** | `+2.64` | `+2.94` | **`+25.04`** | `100.0%` |
| `ablate:goodfire_tainted_grad_all:128` | `−22.94` | **`+1.67`** | `+2.87` | `+0.37` | **`+25.53`** | `83.3%` |
| `sufficiency:only_always_on_L24_L42` | `−0.08` | **`−0.38`** | `−0.08` | `+0.10` | **`−0.22`** | `83.3%` |
| `sufficiency:keep_128_ult_grad_L24_L42` | `−0.13` | **`+0.08`** | `+0.32` | `+0.07` | **`+0.07`** | `66.7%` |
| `sufficiency:keep_512_ult_grad_L24_L42` | `−0.25` | **`−0.10`** | `+0.75` | `+0.86` | **`−0.77`** | `83.3%` |

## 5. Single-Layer (`L35`) vs. Multi-Layer Funnel-Band Subspace Clamping

| Clamp Configuration | Layers Clamped | `98/2` Reject LO | `98/2` Delta LO | Tainted Delta LO | Retal Ctrl / Prov CSI | Retal `NT DeltaCSI` | Cashflow Sanity |
| :--- | :---: | ---: | ---: | ---: | :---: | ---: | ---: |
| `baseline` | `0` (`none`) | `−22.83` | `+1.82` | `+2.91` | `6.2% / 66.2%` | **`+60.0%`** | `100.0%` |
| `clamp_single_L35:spite` | `1` (`L35..L35`) | `−23.15` | `+1.62` | `+2.34` | `6.2% / 66.2%` | **`+60.0%`** | `100.0%` |
| `clamp_prefunnel_L10_L20:spite_local` | `11` (`L10..L20`) | `−20.63` | `+3.76` | `+1.95` | `10.0% / 56.2%` | **`+46.2%`** | `100.0%` |
| `clamp_funnel_L24_L42:spite_local` | `19` (`L24..L42`) | `−23.54` | `+1.90` | `+2.34` | `6.2% / 50.0%` | **`+43.8%`** | `100.0%` |
| `clamp_funnel_L24_L42:spite_u35` | `19` (`L24..L42`) | `−23.72` | `+1.14` | `+1.39` | `6.2% / 66.2%` | **`+60.0%`** | `83.3%` |
| `clamp_late_L24_L55:spite_local` | `32` (`L24..L55`) | `−23.55` | `+1.86` | `+2.33` | `6.2% / 50.0%` | **`+43.8%`** | `100.0%` |
| `clamp_funnel_L24_L42:moral_3d_local` | `19` (`L24..L42`) | `−21.94` | `+2.54` | `+0.65` | `31.2% / 50.0%` | **`+18.8%`** | `100.0%` |
