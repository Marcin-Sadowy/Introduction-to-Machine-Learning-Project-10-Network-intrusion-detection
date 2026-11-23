# UNSW-NB15 Network Intrusion Detection — Full Preprocessing Overview & Decision Log

This document pulls together everything we discovered across the notebooks (`data_exploration.ipynb`, `intrusion_detection.ipynb`) and scripts (`script.py`, `ig_analysis.py`, `prepare_autoencoder_data.py`), explains **why we chose X over Y** at every step, and documents the final Autoencoder-ready dataset.

---

## 1. Dataset Snapshot (Raw Audit)

- **Files:** `UNSW_NB15_training-set.csv` (`175,341 × 45`) and `UNSW_NB15_testing-set.csv` (`82,332 × 45`).
- **Targets:** `label` (`0 = Normal`, `1 = Attack`) and `attack_cat` (`Normal` + 9 attack families, 1:1 with `label`).
- **Missing values:** `0` actual `NaN`s, but `service` uses `'-'` as a placeholder on **53.7%** of train (`94,168` rows) and **57.3%** of test (`47,153` rows).
- **Duplicates:** **38.6%** of raw train (`67,601` rows) and **32.0%** of raw test (`26,387` rows) are exact duplicate feature vectors (`Generic` alone is `90.2%` duplicates!). On top of that, `229` exact feature vectors in train carry both `label = 0` and `label = 1`, and `10.4%` of test rows (`8,541`) also appear verbatim in train.
- **Severe class imbalance (10-class):** Largest-to-smallest ratio in raw train is **430.8:1** (`Normal: 56,000` vs `Worms: 130`). After deduplicating train, binary classes become naturally balanced: **51.2% Normal vs 48.8% Attack**.
- **Extreme skewness:** Volume and timing features (`trans_depth`, `response_body_len`, `sbytes`, `spkts`, `dbytes`, `sloss`, `dloss`) have skewness between **26 and 124**.

---

## 2. How the Project Evolved (Notebooks $\to$ Final Pipeline)

1. **`data_exploration.ipynb` (First Look):**
   - Explored shapes, class counts, and distributions, then merged train+test and ran `OneHotEncoder` + `StandardScaler` before splitting.
   - *Key catch:* `attack_cat` was accidentally encoded as a feature (100% target leak), and fitting scalers/encoders before splitting leaked test statistics into train.
2. **`intrusion_detection.ipynb` (Second Iteration):**
   - Fixed the `attack_cat` leak and moved `OneHotEncoder`/`StandardScaler` after `train_test_split`.
   - Added `mutual_info_classif` (Information Gain) with a `< 0.05` cutoff, `IsolationForest` outlier deletion, and a 500-row UMAP visualization.
   - *Key catches we found during review:*
     - `train` and `test` were still merged before splitting (duplicate leakage across train/test).
     - `id` was still in `x_train` and ranked **#2 in Information Gain (`0.341`)**.
     - Overwriting `x_train` categoricals with `.cat.codes` integers for MI while leaving `x_test` as strings caused `OneHotEncoder.transform(x_test)` to output **100% zeros** for all 156 categorical columns in `x_test`.
     - `IsolationForest` deleted outlier rows from **both `x_train` and `x_test`**.
3. **`script.py` (Steps 1–5 — Modular Audit, Cleaning, Transform, Encoding & Ablation):**
   - Built a reproducible 5-step pipeline with caching, logs, and a combined PDF report (`plots/preprocessing_steps.pdf`).
   - Proved via validation benchmarks which encoders, transforms, and engineered features actually improve Binary F1, ROC-AUC, and Rare-Attack Recall.
4. **`ig_analysis.py` (Dedicated Active-Aware Information Gain):**
   - Re-ran IG with leaks removed, duplicates removed, `log1p` applied, correlated twins combined, and sparse features scored both **globally** and **only when active (`x != 0`)**.
5. **`prepare_autoencoder_data.py` (Final Dataset Builder for Colleague):**
   - Combines every winning step above into clean, verified, ready-to-train CSVs + `preprocessor.pkl` in `dataset/autoencoder_ready/`.

---

## 3. Why We Chose X Over Y (Complete Decision Log)

### Decision 1: Official test split + deduplicated train over merging train & test and random splitting
- **Why:** Because 38.6% of train and 32.0% of test are exact duplicates (and 10.4% overlap), merging and randomly splitting guarantees identical rows land in both `x_train` and `x_test`, inflating accuracy artificially. Keeping `UNSW_NB15_testing-set.csv` untouched preserves a realistic unseen evaluation set, while deduplicating `train` (with majority-vote label resolution) prevents 36k repeated `Generic` rows from drowning out `Exploits`, `Backdoor`, and `Worms`.

### Decision 2: Dropping `id`, `is_ftp_login`, `stcpb`, `dtcpb` over keeping all raw columns
- **Why:**
  - `id` correlates `0.727` with `label` (`IG = 0.559` on raw train) only because rows are sorted by attack bursts—and the class ordering flips in `test`.
  - `is_ftp_login == ct_ftp_cmd` on **100.0%** of rows (exact duplicate).
  - `stcpb` and `dtcpb` are random 32-bit TCP initial sequence numbers (uniform over $[0, 2^{32}]$). They add pure noise (especially fatal for an Autoencoder trying to reconstruct them with MSE!).

### Decision 3: Recoding `service == '-'` to `'none'` over dropping rows or leaving raw `'-'`
- **Why:** Over half the dataset (`53.7%` train, `57.3%` test) has `service == '-'`. It is not missing data—it means no recognized application protocol (e.g., raw TCP/UDP/ICMP/ARP). Keeping it as `'none'` preserves over half the dataset and carries real signal (`44.1%` attack rate vs `71.4%` for `http` and `99.7%` for `pop3`).

### Decision 4: `log1p` before `StandardScaler` over raw `StandardScaler`
- **Why:** Raw byte/packet/duration features span 9 orders of magnitude (skew up to `124`). Scaling raw values crushes 99% of normal flows near `0.0` so a single huge transfer can sit at `+100` (causing the clipped `ylim(0, 1)` violin plots in `intrusion_detection.ipynb`). Applying `np.log1p(x)` keeps `0 -> 0`, pulls skewness down to `~0.1–2.7`, and lets `StandardScaler` produce clean, comparable z-scores. Code-like/binary columns (`sttl`, `dttl`, `ct_state_ttl`, `is_sm_ips_ports`) are intentionally skipped by `log1p`.

### Decision 5: Keeping all rows (no IQR or `IsolationForest` row deletion) over deleting outliers
- **Why:**
  - With 30+ heavy-tailed columns, `1.5 × IQR` deletes **50–70% of the dataset**, wiping out high-rate attacks (`DoS`, `Exploits`).
  - Deleting `IsolationForest` outliers from `x_test` (as done in `intrusion_detection.ipynb`) is invalid because an Intrusion Detection System cannot refuse to classify weird traffic in production—in security, **attacks ARE outliers**.
  - Instead, `log1p` tames heavy tails safely, and `IsolationForest.decision_function` can be used as an *extra feature* (`fe_iso_anomaly_score`, `IG = 0.059`) without dropping a single row.

### Decision 6: Compact `OneHotEncoder(min_frequency=0.005, handle_unknown='infrequent_if_exist')` over 152-column full OHE
- **Why:** `proto` has 133 unique values, where 128 rare protocols (`< 0.1%` of rows) are **96.8% attacks**. Full one-hot encoding creates **152 mostly-zero columns**, runs 2× slower, scored *worse* on validation (`macroF1 = 0.474` vs `0.481` for grouped OHE), and **hurts an Autoencoder** because 152 sparse binary columns dominate MSE reconstruction loss. Grouping rare categories compresses categoricals to **~18 columns**, preserves the "rare protocol = attack" signal in the `infrequent` bucket, and safely handles unseen states in `test` (`ACC`, `CLO`).

### Decision 7: Combining correlated features + Active-Only IG over blind `IG < 0.05` feature dropping
- **Why:**
  - Global IG drops sparse protocol indicators because they are `0` on `83–99%` of rows. When we evaluated IG **only on active rows (`x != 0`)** in `ig_analysis.py`, **`response_body_len` jumped from `0.0575` $\to$ `0.5352`** (highest signal in the entire dataset when HTTP response data exists!), and `tcprtt` jumped `0.301` $\to$ `0.403`. In `script.py` Step 5d, dropping `IG < 0.05` caused Rare-Attack Recall to drop (`89.69% -> 88.01%`).
  - Meanwhile, global IG ignores redundancy. Combining the 14 collinear pairs ($|\rho| \ge 0.90$) into directional `log1p` ratios (`mean_pkt_diff = 0.475`, `byte_log_ratio = 0.467`, `load_log_ratio = 0.460`, `total_bytes_log = 0.454`) beat every raw feature (`sbytes = 0.451`) and prevented collinear twins from being double/triple-weighted in Autoencoder MSE loss.

### Decision 8: Interpretable domain features over PCA or UMAP for model inputs
- **Why:** PCA assumes linear Gaussian variance and struggles with mixed numeric + sparse one-hot features; UMAP is great for 2D visualization (like the 500-row plot in `intrusion_detection.ipynb`), not as a black-box input space for anomaly detection. Combining collinear twins + compact OHE already reduced the feature space to **~53 clean, interpretable features** with zero information loss.

### Decision 9: Normal-only scaling (numeric scaled, OHE kept in `[0, 1]`) over scaling everything on mixed train+attack rows
- **Why (Autoencoder-specific):** An anomaly-detection Autoencoder learns what **Normal** traffic looks like (`label == 0`). Fitting `StandardScaler` strictly on `train_normal` ensures Normal traffic is centered at `mean = 0, std = 1`, so Attack flows naturally produce larger z-score deviations and higher reconstruction error. Keeping the ~18 one-hot columns in `[0, 1]` (unscaled) prevents a rare `0.5%` category from exploding to `z = +14` and hijacking the MSE loss.

### Decision 10: Checking TTL Lab Artefacts (`sttl`, `dttl`, `ct_state_ttl`) over ignoring them
- **Why:** In UNSW-NB15, `sttl = 31` and `dttl = 29` are **100% Normal** (`31,267` rows), while `sttl = 62 / 254` are **78–87% Attack** due to fixed lab router hops. Our Step 5d ablation proved that even with all TTL features removed, our engineered features still reach `F1 = 0.9296` / `ROC-AUC = 0.9853` on `HistGradientBoosting`. We keep TTL in the benchmark dataset for fair comparison against UNSW-NB15 literature, while knowing the model can also work without them.

---

## 4. Final Autoencoder Dataset (`prepare_autoencoder_data.py`)

Running `python prepare_autoencoder_data.py` builds `dataset/autoencoder_ready/`:

| File | Purpose | Rows |
| :--- | :--- | :---: |
| `X_train_normal.csv` | **Autoencoder training input** (`model.fit(X, X)` — 100% Normal traffic, `label == 0`) | `41,368` |
| `X_train_all.csv` / `y_train_all.csv` | Full clean train split (for hybrid AE+GBDT or supervised baselines) | `80,832` |
| `X_val.csv` / `y_val.csv` | Validation split (`label` + `attack_cat`) for early stopping (`y_val['label']==0`) & threshold tuning | `20,208` |
| `X_test.csv` / `y_test.csv` | Official untouched test set (`label` + `attack_cat`) for final evaluation only | `82,332` |
| `preprocessor.pkl` | Saved `StandardScaler`, `OneHotEncoder`, `log_cols`, and `feature_names` | — |

### Automated Sanity Checks Built Into `prepare_autoencoder_data.py`
1. `0` NaN and `0` Inf values across `X_train_all`, `X_val`, and `X_test`.
2. Zero leak columns (`id`, `attack_cat`, `label`, `stcpb`, `dtcpb`, `is_ftp_login`) in `X`.
3. `OneHotEncoder` active on 100% of train/val/test rows (each row's one-hot sum equals exactly `3` — proving the all-zero `x_test` bug cannot happen).
4. `X_train_normal` numeric features verified at `mean ≈ 0`, `std ≈ 1`.
5. Visual verification plot saved to `plots/ae_dataset_overview.png`.
