# Dataset Exploration: Sparkov Credit Card Transactions

Scripts: `scripts/explore.py`, `scripts/explore2.py`, `scripts/explore3.py`

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

## How the findings relate to the paper

### Supported
- **Behavior agent features.** Amount and hour of day (paper Table 4) both separate fraud strongly.
- **Realistic fraud rate.** Sparkov's 0.58% is closer to real-world rates than the paper's synthetic 5.3%.
- **Context matters.** Online and in-person categories have different risk levels, which supports context-aware weighting.

### Not supported or untestable
- **Geo weight.** The paper gives Geo the highest weight for remittance. Neither home-to-merchant distance nor consecutive-merchant speed shows real signal here.
- **Velocity weight.** The paper makes Velocity dominant for POS (0.40). On this data, Behavior features (hour, amount) are stronger and fraud does not come in 1-minute bursts.
- **Graph agent.** With 693 merchants and about 1,900 transactions each, few merchants are new or rare. Expect weak signal.
- **Device and SIM-swap logic.** No device field exists, so these can only be tested on the synthetic generator.

## Design decisions for this repo

1. **Context label.** Online categories (`*_net`) are labeled QR and in-person categories (`*_pos`) are labeled POS. This is a proxy only. Remittance is evaluated only on the synthetic generator. Thresholds are recalibrated from data, not copied from the paper.
2. **Geo agent on Sparkov.** Disabled (no real signal). Weights are renormalized in code over the remaining agents:

   ```
   w_i' = w_i / (1 - w_geo)
   ```

3. **Velocity windows.** Changed from 1 min / 1 h / 24 h to 10 min / 1 h / 24 h, with thresholds at the 99th percentile of legit counts per card.
4. **Behavior features.** `log(amt)`, cyclical hour encoding (`sin(2*pi*hour/24)`, `cos(2*pi*hour/24)`), `category`, and gap since the previous transaction.
5. **Two Behavior variants.** The paper's Isolation Forest (unsupervised) and a supervised gradient boosting model, compared in the results.
6. **No leakage.** Graph agent counts use only transactions before each row's timestamp.
7. **Split.** Time-based, not random. Isolation Forest is trained on early non-fraud rows only.
8. **Cold-start.** Defined as each card's first 3 transactions.
9. **Weights.** Paper weights are the baseline. A learned-weights version (paper Future Work 9.2.2) is compared against it.

## Limitations

- Sparkov is synthetic. The night-hour spike is a simulator pattern, so results will be optimistic compared with real data.
- Paper results came from a different synthetic dataset where fraud was injected in the form each agent detects. Results here are an independent test and are expected to be lower.