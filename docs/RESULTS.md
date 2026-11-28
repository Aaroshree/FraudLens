# Evaluation Results

All tables are produced by one command and saved to `artifacts/results.md` and `artifacts/results.json`:

```
python scripts\run_evaluation.py
```

Runtime: about 5 minutes on CPU (983 per-card Isolation Forests).

## Setup

| Property | Value |
|---|---|
| Data | Sparkov `fraudTrain`, 1,296,675 transactions, 0.58% fraud |
| Train | First 70% by time, 907,672 rows |
| Validation | First half of the test period, 194,501 rows (calibrates thresholds, learns weights) |
| Evaluation | Second half of the test period, 194,502 rows, fraud rate 0.583% |
| Positive decision | Any escalation: single OTP, dual OTP or block |
| Contexts | POS and QR (category proxy), Geo agent disabled |

Calibrated thresholds are the 95th, 99th and 99.9th percentiles of legitimate composite scores per context on the validation half. Escalation happens when the score is at or above a threshold.

## Overall (second half of the test period)

| Behavior | Weights | Thresholds | ROC-AUC | PR-AUC | Precision | Recall | FPR |
|---|---|---|---|---|---|---|---|
| Isolation Forest | Fixed | Paper | 0.833 | 0.060 | 0.103 | 0.150 | 0.76% |
| Isolation Forest | Fixed | Calibrated | 0.833 | 0.060 | 0.036 | 0.252 | 3.97% |
| Isolation Forest | Behavior alone | Calibrated | 0.834 | 0.050 | 0.044 | 0.427 | 5.40% |
| Supervised (extension) | Behavior alone | Calibrated | 0.940 | 0.740 | 0.093 | 0.915 | 5.22% |
| Supervised (extension) | Fixed | Calibrated | 0.926 | 0.602 | 0.085 | 0.793 | 5.02% |
| Supervised (extension) | Learned | Calibrated | 0.954 | 0.726 | 0.092 | 0.876 | 5.06% |

Supervised Behavior score alone: ROC-AUC 0.940, PR-AUC 0.740.

## By context

| Behavior | Weights | Thresholds | Context | Precision | Recall | FPR |
|---|---|---|---|---|---|---|
| Isolation Forest | Fixed | Paper | POS | 0.109 | 0.062 | 0.23% |
| Isolation Forest | Fixed | Paper | QR | 0.102 | 0.309 | 3.59% |
| Isolation Forest | Fixed | Calibrated | POS | 0.024 | 0.218 | 3.93% |
| Isolation Forest | Fixed | Calibrated | QR | 0.089 | 0.314 | 4.21% |
| Isolation Forest | Behavior alone | Calibrated | POS | 0.029 | 0.363 | 5.41% |
| Isolation Forest | Behavior alone | Calibrated | QR | 0.117 | 0.543 | 5.37% |
| Supervised (extension) | Behavior alone | Calibrated | POS | 0.073 | 0.924 | 5.22% |
| Supervised (extension) | Behavior alone | Calibrated | QR | 0.184 | 0.899 | 5.25% |
| Supervised (extension) | Fixed | Calibrated | POS | 0.064 | 0.783 | 5.16% |
| Supervised (extension) | Fixed | Calibrated | QR | 0.200 | 0.810 | 4.28% |
| Supervised (extension) | Learned | Calibrated | POS | 0.071 | 0.882 | 5.20% |
| Supervised (extension) | Learned | Calibrated | QR | 0.207 | 0.864 | 4.36% |

## Decision distribution (% of each class)

| Behavior | Weights | Thresholds | Class | Approve | Single OTP | Dual OTP | Block |
|---|---|---|---|---|---|---|---|
| Isolation Forest | Fixed | Paper | legit | 99.2 | 0.7 | 0.0 | 0.0 |
| Isolation Forest | Fixed | Paper | fraud | 85.0 | 14.0 | 1.0 | 0.0 |
| Isolation Forest | Fixed | Calibrated | legit | 96.0 | 3.1 | 0.8 | 0.1 |
| Isolation Forest | Fixed | Calibrated | fraud | 74.8 | 10.3 | 11.6 | 3.4 |
| Isolation Forest | Behavior alone | Calibrated | legit | 94.6 | 4.3 | 1.0 | 0.1 |
| Isolation Forest | Behavior alone | Calibrated | fraud | 57.3 | 27.8 | 10.6 | 4.3 |
| Supervised (extension) | Behavior alone | Calibrated | legit | 94.8 | 4.2 | 0.9 | 0.1 |
| Supervised (extension) | Behavior alone | Calibrated | fraud | 8.5 | 3.2 | 19.5 | 68.8 |
| Supervised (extension) | Fixed | Calibrated | legit | 95.0 | 4.2 | 0.7 | 0.1 |
| Supervised (extension) | Fixed | Calibrated | fraud | 20.7 | 1.2 | 32.7 | 45.3 |
| Supervised (extension) | Learned | Calibrated | legit | 94.9 | 4.2 | 0.7 | 0.1 |
| Supervised (extension) | Learned | Calibrated | fraud | 12.4 | 1.4 | 17.7 | 68.4 |

## Weights (order: velocity, behavior, graph_risk)

| Behavior | Type | Context | Weights |
|---|---|---|---|
| Isolation Forest | Fixed | POS | 0.533, 0.267, 0.200 |
| Isolation Forest | Fixed | QR | 0.400, 0.333, 0.267 |
| Isolation Forest | Learned | POS | 0.346, 0.521, 0.133 |
| Isolation Forest | Learned | QR | 0.291, 0.541, 0.168 |
| Supervised (extension) | Fixed | POS | 0.533, 0.267, 0.200 |
| Supervised (extension) | Fixed | QR | 0.400, 0.333, 0.267 |
| Supervised (extension) | Learned | POS | 0.023, 0.941, 0.036 |
| Supervised (extension) | Learned | QR | 0.000, 0.902, 0.098 |

## How to read these tables

- Precision is low at about 5% FPR because fraud is 0.58% of transactions. Precision depends on the base rate, not only on model quality.
- Paper thresholds almost never fire on this data, so precision is high and recall is near 0.15.
- Calibrated thresholds target about 5% FPR on legitimate transactions. The evaluation half lands between 3.9% and 5.4%, so the calibration transfers across time with some drift.
- These numbers are not comparable to the paper's synthetic results, which came from a different dataset where fraud was injected in the form each agent detects.
- "Behavior alone" scores transactions with the Behavior agent only, using the same calibration rule. Under supervised Behavior it has higher recall than the learned ensemble (0.915 vs 0.876) at a slightly higher FPR (5.22% vs 5.06%), so the extra agents do not help at this operating point. Under Isolation Forest the FPRs are not matched (5.40% vs 4.33%), so recall should not be compared directly.