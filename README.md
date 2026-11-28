# FraudLens

FraudLens is a lightweight, interpretable fraud detection framework for Nepal's digital payments.
It scores each transaction with Velocity, Geo, Behavior, and Graph-inspired Risk agents, combines them with transaction-type-
specific weights (remittance, POS, QR), and returns one of four actions: approve, single OTP, dual OTP, or block.

It extends the paper "Lightweight Multi-Agent Fraud Detection Framework for Digital Payments in Nepal" and tests the design on the public Sparkov dataset.

- [Dataset exploration findings](docs/DATA_EXPLORATION.md)
- [Full evaluation tables](docs/RESULTS.md)

## Quick start (Windows Command Prompt)

```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
pytest -q
python scripts\run_evaluation.py
```

Download `fraudTrain.csv` (Sparkov simulated credit card transactions, from Kaggle) into `data/raw/` first. The evaluation builds the feature file on the first run and takes about 5 minutes on CPU.

## How it works

1. **Context** is classified per transaction (remittance, POS or QR).
2. **Agents** each return a risk score in [0, 1]: velocity, geo, behavior, graph risk.
3. **Synthesis** combines them with context-specific weights (a convex combination, so the result stays in [0, 1]):

   ```
   R = w_1 * S_velocity + w_2 * S_geo + w_3 * S_behavior + w_4 * S_graph
   ```

4. **OTP interlock** maps R to approve, single OTP, dual OTP or block using context-specific thresholds.

Every constant (weights, thresholds, windows, decay rate) lives in `src/fraudlens/config.yaml`. The default profile is `sparkov`. The `paper` profile re-enables Geo and the full Graph formula.

## Package layout

```
src/fraudlens/
  config.yaml        every constant
  config.py          loader and validation
  features.py        leak-free feature builder
  agents/            velocity, behavior, graph_risk, geo
  synthesis.py       fixed and learned weights
  otp_interlock.py   thresholds, calibration, decisions
scripts/
  run_evaluation.py  one command to reproduce the results below
  explore*.py, check_*.py   analysis scripts behind the findings
tests/               pytest suite (17 tests)
```

The tests check that weights sum to 1, that scores and composites stay in [0, 1], that the interlock bands behave as in the paper, and that the paper's worked example (Sita, remittance) gives R of about 0.798 (the paper reports 0.797 after rounding) and lands in Dual OTP.

## Results on Sparkov

Setup: 1,296,675 transactions, 0.58% fraud. Time split: first 70% trains the models. The test period is split in half: the first half calibrates thresholds and learns weights, the second half (194,502 rows) is evaluated. POS and QR are proxies based on transaction category, and the Geo agent is disabled because Sparkov has no usable geographic signal.

| Setup | ROC-AUC | PR-AUC | Recall | FPR |
|---|---|---|---|---|
| Paper agents (Isolation Forest), fixed weights, paper thresholds | 0.833 | 0.060 | 0.150 | 0.76% |
| Paper agents (Isolation Forest), fixed weights, calibrated thresholds | 0.833 | 0.060 | 0.252 | 3.97% |
| Supervised Behavior (extension), learned weights, calibrated thresholds | 0.954 | 0.726 | 0.876 | 5.06% |
| Supervised Behavior alone, calibrated thresholds | 0.940 | 0.740 | 0.915 | 5.22% |

Full tables (by context, decision distribution, weights) are in [docs/RESULTS.md](docs/RESULTS.md).

### What the results show

1. **The paper's thresholds do not transfer.** They almost never fire on Sparkov (recall 0.15). Calibrating thresholds on legitimate score percentiles fixes this, and the calibration holds on later data.
2. **The unsupervised Behavior agent is weak here.** Night-time transactions are common for legitimate users (23.2%), so an Isolation Forest does not treat night as unusual. A supervised Behavior variant is much stronger (recall 0.793 vs 0.252 with fixed weights at about 5% FPR).
3. **Learned weights beat the paper's fixed weights, but not the Behavior score alone.** They put 0.90 to 0.94 of the weight on Behavior, and POS and QR end up with similar weights. At the 5% operating point, Behavior alone catches more fraud (recall 0.915 vs 0.876 for the ensemble). On these proxy contexts, neither the extra agents nor context-aware weighting are supported.
4. **The Graph agent works only through novelty** (first-time card and merchant pairs). Merchant frequency and connectivity carry no signal on Sparkov.
5. **Velocity is weak**: fraud here rarely arrives in bursts (velocity-only AUC 0.529).

### Limitations

- Sparkov is simulated. Patterns such as the night-hour spike come from the simulator, so results are optimistic compared with real data.
- Precision is about 0.09 at 5% FPR because fraud is rare. This is a base-rate effect.
- There is no remittance data, device field or real geography, so Geo, device mismatch and SIM-swap logic cannot be tested on Sparkov.
- One split and one seed, with no confidence intervals.
- With supervised Behavior, the learned ensemble has higher ROC-AUC than the Behavior score alone (0.954 vs 0.940) but lower PR-AUC (0.726 vs 0.740) and lower recall at about 5% FPR (0.876 vs 0.915). On Sparkov, the extra agents do not beat a strong Behavior model.
- These numbers are not comparable with the paper's synthetic results.

## Status

Done: package, config, four agents, synthesis, interlock, feature pipeline, reproducible evaluation, tests.

Planned: synthetic generator (remittance, device, SIM-swap) to test Geo and the paper profile end to end, FastAPI decision endpoint, Streamlit dashboard with a "why flagged" view, GitHub Actions, latency benchmark.