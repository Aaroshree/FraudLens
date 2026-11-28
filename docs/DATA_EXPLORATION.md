# Dataset Exploration: Sparkov Credit Card Transactions

Scripts: `scripts/explore.py`, `scripts/explore2.py`, `scripts/explore3.py`, `scripts/check_*.py`

Findings 10 and 11 are reproduced by `python scripts\run_evaluation.py`. The full output is in [RESULTS.md](RESULTS.md).

## Dataset description

| Property | Value |
|---|---|
| Source | Sparkov simulated credit card transactions (`fraudTrain`) |
| Type | Synthetic, US-based, no real customers |
| Rows | 1,296,675 |
| Cards | 983 |
| Merchants | 693 |
| Categories | 14 |
| Date range | 2019-01-01 to 2020-06-21 |
| Fraud rows | 7,506 (0.58%) |
| Median transactions per card | 1,054 |

Columns: transaction time, card number, merchant, category, amount, cardholder details, cardholder home lat/long, merchant lat/long, `is_fraud`.

Missing compared with the paper: device ID, transaction type (remittance / POS / QR), and user-to-user transfers.

## Findings

### 1. Hour of day is the strongest signal

| Hours | Fraud rate |
|---|---|
| 22 and 23 | 2.9% |
| 00 to 03 | 1.4% to 1.5% |
| 04 to 21 | about 0.1% |

Fraud is about 14x to 29x more likely between 22:00 and 03:59.

### 2. Amount separates fraud strongly

| | Legit | Fraud |
|---|---|---|
| Mean | 67.67 | 531.32 |
| Median | 47.28 | 396.51 |

### 3. Fraud rate differs by category

Highest: `shopping_net` (1.76%), `misc_net` (1.45%), `grocery_pos` (1.41%).
Lowest: `food_dining`, `home`, `health_fitness` (about 0.15% to 0.17%).

Online categories carry more fraud than in-person ones: `shopping_net` 1.76% vs `shopping_pos` 0.72%, and `misc_net` 1.45% vs `misc_pos` 0.31%.

### 4. Geographic signals carry no real fraud signal

- Home-to-merchant distance: mean 76.2 km (fraud) vs 76.1 km (legit), identical quartiles.
- Speed between consecutive merchants: median 64 km/h (fraud) vs 20 km/h (legit). This is driven by shorter gaps between fraud transactions, not real travel. Merchant locations are random around the cardholder's home, and speeds above 10,000 km/h appear in both classes.

Decision: the Geo agent is disabled on Sparkov and tested only on the synthetic generator.

### 5. Fraud transactions arrive closer together

| Gap between transactions | Legit | Fraud |
|---|---|---|
| Median | 16,623 s | 4,908 s |
| 25th percentile | 6,064 s | 1,473 s |

Gaps are minutes to hours, so 1-minute burst windows will rarely fire on this data.

### 6. Night transactions are common for legit users

| | Share at night (22:00 to 03:59) |
|---|---|
| Legit | 23.2% |
| Fraud | 84.8% |

Because 23% of legit transactions occur at night, an Isolation Forest trained on legit data will not treat night as anomalous. Hour is a strong signal for supervised models but a weak one for the unsupervised Behavior agent. This motivates adding a supervised Behavior variant.

### 7. Velocity agent: weak coverage, 24h window inverted

| Window | Fraud flag rate | Legit flag rate |
|---|---|---|
| 10 min | 1.03% | 0.08% |
| 1 hour | 5.61% | 0.27% |
| 24 hours | 0.15% | 0.84% |

Short and medium windows are strongly informative when they fire, but cover only 1% and 5.6% of fraud. The 24-hour window is inverted (fraud is flagged less than legit), so it is disabled on Sparkov. Velocity-only AUC is 0.529 after disabling the 24h window (POS 0.524, QR 0.538), so fraud on Sparkov is mostly not detectable by transaction frequency.

### 8. Behavior agent: Isolation Forest vs supervised baseline

Time-based split (first 70% train, last 30% test, test fraud rate 0.61%). This table uses the full test period. Findings 10 and 11 use only its second half (fraud rate 0.58%), so the numbers differ slightly.

| Model | ROC-AUC | PR-AUC | POS PR-AUC | QR PR-AUC |
|---|---|---|---|---|
| Isolation Forest (per card, paper) | 0.847 | 0.064 | 0.036 | 0.184 |
| Gradient boosting (supervised extension) | 0.946 | 0.764 | 0.761 | 0.778 |

A random model has PR-AUC of about 0.006. The unsupervised agent carries real signal (mostly from amount) but with low precision. The supervised model is far stronger, which supports the paper's future work on learned components. Both are optimistic because Sparkov fraud is simulated.

### 9. Graph-inspired agent: only novelty carries signal

| Signal | ROC-AUC | PR-AUC |
|---|---|---|
| N (interaction novelty) | 0.632 | 0.0096 |
| F (merchant frequency) | 0.477 | 0.0058 |
| C (merchant connectivity) | 0.480 | 0.0059 |
| S_graph = (N + F + C) / 3 | 0.582 | 0.0078 |

Random PR-AUC is about 0.006. With 693 merchants of about 1,900 transactions each, merchant rarity (F) and connectivity (C) do not separate fraud. Only first-time card-merchant pairs (N) help, weakly. On Sparkov the Graph agent uses N only. The paper's full formula is kept in the `paper` profile as a baseline. Graph signals designed for new or obscure merchants need data that contains them.

### 10. Full pipeline: paper heuristics vs calibrated thresholds

Time-based split. Thresholds calibrated on the first half of the test period, evaluated on the second half. Context labels are proxies (POS/QR), Geo disabled, weights renormalised.

| Setup | Thresholds | Precision | Recall | FPR |
|---|---|---|---|---|
| Paper agents (Isolation Forest) | Paper | 0.103 | 0.150 | 0.76% |
| Paper agents (Isolation Forest) | Calibrated | 0.036 | 0.252 | 3.97% |
| With supervised Behavior | Paper | 0.874 | 0.153 | 0.01% |
| With supervised Behavior | Calibrated | 0.085 | 0.793 | 5.02% |

- The paper's thresholds do not transfer to Sparkov: almost nothing is escalated.
- At about 4% to 5% FPR, the paper's agents reach recall 0.25 and the supervised variant 0.79.
- Precision is low because fraud is 0.6% of transactions, not because of a bug.
- Fixed convex weights cap the strongest agent's influence (Behavior weight 0.267 in POS).
- With supervised Behavior and fixed weights, the Single OTP tier is nearly empty (1.2% of fraud): 32.7% of fraud lands in Dual OTP and 45.3% in Block.

### 11. Learned weights vs fixed weights

Weights are fit per context on the first half of the test period (balanced logistic regression, negative coefficients clipped to 0, then normalised to sum to 1) and evaluated on the second half. Thresholds are calibrated on the same first half.

| Setup | ROC-AUC | PR-AUC | Recall | FPR |
|---|---|---|---|---|
| Paper agents, fixed weights | 0.833 | 0.060 | 0.252 | 3.97% |
| Paper agents, learned weights | 0.848 | 0.064 | 0.337 | 4.33% |
| Supervised Behavior, fixed weights | 0.926 | 0.602 | 0.793 | 5.02% |
| Supervised Behavior, learned weights | 0.954 | 0.726 | 0.876 | 5.06% |
| Supervised Behavior score alone | 0.940 | 0.740 | not measured | not measured |

Learned weights (order: velocity, behavior, graph risk):

| Behavior | POS | QR |
|---|---|---|
| Isolation Forest | 0.346, 0.521, 0.133 | 0.291, 0.541, 0.168 |
| Supervised | 0.023, 0.941, 0.036 | 0.000, 0.902, 0.098 |

Learned weights beat the paper's fixed weights, supporting future work 9.2.2. Recall comparisons are at slightly different FPR (within 0.4 points). With supervised Behavior, the learned weights put 0.90 to 0.94 of the weight on Behavior and drive Velocity to nearly zero. POS and QR learn similar weights, so context-aware weighting is not supported on these proxy contexts. The multi-agent design is better tested on data with device, geo, and remittance signals.

Against the Behavior score alone, the learned ensemble has higher ROC-AUC (0.954 vs 0.940) and lower PR-AUC (0.726 vs 0.740). Calibrated recall for Behavior alone has not yet been measured with the package.

Note on earlier numbers: an earlier standalone script (`scripts/check_learned_weights.py`) reported FPR near 3.4% for the supervised rows. The package reports about 5.0%, which matches the 5% calibration target and the Finding 10 run. The package numbers are used throughout.

## How the findings relate to the paper

### Supported
- **Behavior agent features.** Amount and hour of day (paper Table 4) both separate fraud strongly.
- **Realistic fraud rate.** Sparkov's 0.58% is closer to real-world rates than the paper's synthetic 5.3%.
- **Context matters.** Online and in-person categories have different risk levels. However, learned weights for the POS and QR proxies came out similar (Finding 11), so context-aware weighting itself is not shown to help here.

### Not supported or untestable
- **Geo weight.** The paper gives Geo the highest weight for remittance. Neither home-to-merchant distance nor consecutive-merchant speed shows real signal here.
- **Velocity weight.** The paper makes Velocity dominant for POS (0.40). On this data, Behavior features (hour, amount) are stronger and fraud does not come in 1-minute bursts.
- **Graph agent.** With 693 merchants and about 1,900 transactions each, few merchants are new or rare. Only the novelty signal helps, and only weakly.
- **Device and SIM-swap logic.** No device field exists, so these can only be tested on the synthetic generator.

## Design decisions for this repo

1. **Context label.** Online categories (`*_net`) are labeled QR and in-person categories are labeled POS. This is a proxy only. Remittance is evaluated only on the synthetic generator. Thresholds are recalibrated from data, not copied from the paper.
2. **Geo agent on Sparkov.** Disabled (no real signal). Weights are renormalised in code over the remaining agents:

   ```
   w_i' = w_i / (1 - w_geo)
   ```

3. **Velocity windows.** 10 min and 1 h only. Counts are per card, and the thresholds are global: a window is flagged if count > 1 in 10 min, or count > 2 in 1 h. The 24 h window is disabled because it is inverted on Sparkov (Finding 7). Each flagged window adds 0.40:

   ```
   S_velocity = min(1, 0.40 * f_10m + 0.40 * f_1h)
   ```

4. **Behavior features.** `log(amt)`, cyclical hour encoding (`sin(2*pi*hour/24)`, `cos(2*pi*hour/24)`), `category`, the 10-minute and 1-hour transaction counts, and `log(1 + gap)` since the previous transaction. Missing gaps are filled with the median gap of the training split.
5. **Two Behavior variants.** The paper's Isolation Forest (unsupervised, one model per card) and a supervised gradient boosting model, compared in the results.
6. **No leakage.** Graph agent counts use only transactions before each row's timestamp. Velocity counts also use only earlier transactions.
7. **Split.** Time-based, not random. Isolation Forest is trained on early non-fraud rows only. The test period is split again: the first half calibrates thresholds and learns weights, the second half is evaluated.
8. **Cold-start.** Defined as each card's first 3 transactions. The Behavior agent returns a default score of 0.5 for cards with no legitimate training rows.
9. **Weights.** Paper weights with Geo removed and renormalised are the baseline. A learned-weights version is compared against it (Finding 11).
10. **Graph agent on Sparkov.** Uses novelty only: `N = 1` for a first interaction, otherwise `N = exp(-0.3 * c)`, where `c` is the number of earlier interactions between the card and the merchant (Finding 9). The merchant frequency and connectivity signals are switched off in `config.yaml`, not deleted.

## Limitations

- Sparkov is synthetic. The night-hour spike is a simulator pattern, so results will be optimistic compared with real data.
- Paper results came from a different synthetic dataset where fraud was injected in the form each agent detects. Results here are an independent test and are expected to be lower.
- Each card has about 1,000 transactions, so very few test cards should lack training history. The cold-start handling is probably barely exercised on Sparkov. Check the count before relying on this.
- One time split and one random seed, with no confidence intervals.
- The learned ensemble does not clearly beat its strongest single component (Behavior score alone) on Sparkov: ROC-AUC is higher, PR-AUC is lower.