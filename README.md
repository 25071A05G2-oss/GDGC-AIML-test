# Network Traffic Classification (IoT Intrusion Detection)

**Participant:** G Daniel Finny
**Task:** Classify network flows into `Benign`, `Availability`, `Reconnaissance`, `Spoofing`
**Metric:** Macro-F1 (unweighted mean of the four per-class F1 scores)
**Public leaderboard score:** ~0.84 macro-F1

---

## 1. Problem statement

Each row is one network flow (a summary of related packets) described by 82 numeric and categorical
features (ports, protocol, service, packet counts, flags, payload sizes, timings, TCP windows, ...).
The goal is to predict the traffic category for every row of `test.csv`.

| Class | Meaning |
|---|---|
| Benign | Normal IoT activity (e.g. MQTT, HTTP, DNS) |
| Availability | Attacks on service availability (e.g. SYN flood / DoS) |
| Reconnaissance | Scanning / information gathering (e.g. nmap scans) |
| Spoofing | Identity / address impersonation (e.g. ARP/DNS spoofing) |

**Key difficulty:** the hidden test data contains traffic *profiles that do not appear in training*,
so a model that scores well on a random validation split can fail on the real test set.

## 2. Data

| File | Rows | Notes |
|---|---|---|
| `train_part1.xlsx` + `train_part2.xlsx` | 84,713 (two halves) | labelled |
| `test.csv` | 33,161 | unlabelled |
| `sample_submission.csv` | 33,161 | required format |

Training class balance is extremely skewed: Availability 80,072 · Spoofing 2,323 · Benign 1,583 ·
Reconnaissance 735. Raw data is **not** included in this repo (competition data); place it in the
folder described in "How to run".

Dataset credit: B. S. and Rohini Nagapadma (The National Institute of Engineering, Mysuru),
CC BY 4.0.

## 3. What I found in the data

Comparing train with test showed a strong distribution shift:

- **SYN flood without a reply.** In train, every flood flow received a reply (RST). In test, about
  9,000 flood flows got none. A first model that used the reply as a shortcut labelled them
  `Reconnaissance` with ~100% confidence.
- **Nmap scans with unseen TCP windows** (including XMAS-style flag patterns) that never appear in train.
- **Environment-dependent timing.** Duration / inter-arrival features shift heavily between train
  and test even for traffic of the same kind, so they are used cautiously.
- **Unseen DNS / SSL / mDNS flows**, which is where Benign vs Spoofing is hardest.

## 4. Approach

### 4.1 Feature engineering (`build()`)
The idea is to describe *behaviour* rather than memorise tool- or victim-specific values:

- Coarsened ports: port zone (<1024, <49152, ephemeral) plus flags for 21, 53, 80, 443, 1883, 5353, 123, 137.
- Protocol and service as small integer codes.
- Log-scaled packet counts, payload sizes and timings (`log1p`).
- Behaviour flags: `has_reply`, `syn_probe` (single SYN, no payload), `syn_payload` (single SYN carrying
  payload, typical of floods), `syn_noack`, `xmas_like`, `same_port`.
- Clipped TCP flag counts, `down_up` ratio, forward/backward payload byte ratio.
- Window-size indicators (zero / large) instead of exact values.
- A second feature view (`full`) adds exact window size and header sizes; both views are ensembled.

### 4.2 Model
- **LightGBM** multiclass, `class_weight="balanced"` (handles the 100:1 imbalance), shallow trees
  (15 leaves), L2 regularisation, row and column subsampling.
- **Ensemble:** 2 feature views x 5 random seeds = 10 models; predicted probabilities are averaged and the
  arg-max class is submitted.

### 4.3 Validation (`RUN_CV = True`)
A random split is misleading here, so I validate on **held-out traffic profiles**:

1. Build a coarse "profile signature" per row (protocol / service / port bucket / window bucket / flag
   pattern / reply bucket).
2. Use `StratifiedGroupKFold` with that signature as the group, so entire profiles are unseen in each fold.
3. The one huge flood profile is split randomly so it stays in training.

| Validation | Macro-F1 |
|---|---|
| Random 5-fold (optimistic, not trusted) | ~0.97 |
| Profile-grouped 5-fold (realistic) | ~0.75 |
| Public leaderboard | ~0.84 |

Grouped CV is intentionally harsh. The leaderboard sits between the two, consistent with a partial shift.

## 5. How to run

```bash
pip install pandas numpy scikit-learn lightgbm openpyxl
```

Put these files in one folder (names only need to *contain* these keywords):
`train_part1.xlsx`, `train_part2.xlsx`, `test.csv`, `sample_submission.csv`

Then edit `SEARCH_DIRS` at the top of `solution.py` to that folder (on Kaggle it already reads
`/kaggle/input`) and run:

```bash
python solution.py
```

It prints the validation report and writes `submission.csv` (`id,Target`, 33,161 rows).
Runtime is about 1-2 minutes on 2 CPU cores. Set `RUN_CV = False` to skip validation.

## 6. Repository contents

| File | Description |
|---|---|
| `solution.py` | Full pipeline: load, features, grouped validation, training, prediction |
| `submission.csv` | Predictions produced by `solution.py` for `test.csv` |
| `README.md` | This document |

## 7. Results

Predicted test distribution: Benign 10,414 · Availability 10,391 · Reconnaissance 7,906 · Spoofing 4,450.
Public leaderboard macro-F1 is about 0.84. The hidden-partition score may differ, as the test set
contains unseen profiles.

## 8. Limitations and future work

- Roughly 10,000 DNS / SSL / mDNS test flows resemble nothing labelled in training, so Benign vs Spoofing
  there is the main source of error.
- Tried and rejected after validation: dropping timing features (hurt grouped CV), kNN smoothing of
  test predictions (no gain), ExtraTrees / RandomForest (worse than LightGBM).
- A later variant that rescaled class probabilities improved grouped CV to ~0.79 but was not used for
  this submission.
- Possible next steps: semi-supervised domain adaptation on the test set, per-profile calibration, and
  more shift-robust features for Spoofing.

## 9. Compliance

Only the provided training data is used. No external labelled data. Standard libraries only
(pandas, numpy, scikit-learn, lightgbm).
