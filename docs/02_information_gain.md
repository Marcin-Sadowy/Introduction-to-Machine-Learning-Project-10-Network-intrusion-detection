# Information Gain (IG) & Preprocessing Notes — What We Changed and Why

This is a straightforward write-up of the preprocessing steps we finalized in `preprocessing/01_pipeline_audit_and_fe.py` and `preprocessing/02_information_gain_analysis.py`, why we re-ran Information Gain (`mutual_info_classif`), and what changed compared to the first notebook pass.

---

## Why the first IG pass was misleading

In the earlier notebook, we ran `mutual_info_classif` on raw merged data and dropped everything below `0.05`. Four things were distorting the scores:

1. **Leaks and random IDs were still in the table** — `id` ranked **#2 overall (`0.341` in the notebook, `0.559` on raw train)** purely because the CSV rows are ordered by class blocks. Random 32-bit TCP sequence numbers (`stcpb = 0.066`, `dtcpb = 0.060`) also passed the `0.05` cutoff just because being non-zero indicates TCP, and `is_ftp_login` was an exact 100% duplicate of `ct_ftp_cmd`.
2. **Duplicates inflated specific columns** — `38.6%` of raw train is exact duplicates (mostly ~36k repeated `Generic` DNS flood rows), which skewed feature importance toward constant `Generic` values.
3. **Extreme skewness hurt continuous estimators and scalers** — raw traffic features (`sbytes`, `trans_depth`, `dpkts`, `sload`, etc.) have skewness up to **124**. Without `log1p`, both `StandardScaler` and sklearn's kNN-based continuous MI estimator get distorted by extreme heavy tails.
4. **Collinearity & sparsity weren't handled** — IG scores each feature in isolation (keeping 14 pairs with Spearman $|\rho| \ge 0.90$) and averages over all rows (punishing sparse HTTP/FTP features that are `0` on 83–99% of rows even when non-zero values are strong attack signals).

---

## Step-by-Step Pipeline (What We Actually Did)

### Step 1 — Remove target leaks & noise
- Dropped `id`, `is_ftp_login`, `stcpb`, and `dtcpb`.
- Kept `attack_cat` strictly as a target (never a feature).
- Replaced `service == '-'` (53.7% of rows) with `'none'` so it's treated cleanly as "unidentified service".

### Step 2 — Deduplicate train & split properly
- Kept the official `UNSW_NB15_testing-set.csv` completely untouched as our unseen test set.
- Deduplicated the training set (`175,341` $\to$ `101,040` unique flows), resolving the 229 conflicting binary labels by majority vote.
- Created a stratified 80/20 `train` / `val` split (`80,832` train / `20,208` val; baseline attack rate in train = `0.488`).

### Step 3 — Apply `log1p` to skewed numeric features
- Applied `np.log1p(x) = log(1 + x)` to the **30 positive continuous/count features** with `|skew| > 1` (`sbytes`, `dbytes`, `dur`, `rate`, `sload`, `dload`, `spkts`, `dpkts`, `trans_depth`, `response_body_len`, etc.).
- Why `log1p` instead of plain `log` or IQR outlier deletion:
  - Keeps `0 -> 0` (critical since many network columns are zero-inflated).
  - Pulls skewness down from `30–124` to `~0.1–2.7` on most continuous features (a few sparse count columns like `ct_ftp_cmd` at `7.25` and `ct_dst_sport_ltm` at `5.21` remain higher), so `StandardScaler`, linear models, Autoencoders, and continuous MI behave properly.
  - Preserves extreme attack flows instead of deleting them.
- **Skipped `log1p`** on binary flags (`is_sm_ips_ports`) and discrete code-like columns (`sttl`, `dttl`, `swin`, `dwin`, `ct_state_ttl`).

### Step 4 — Combine correlated features
- Found 14 numeric pairs with $|\rho_{\text{Spearman}}| \ge 0.90$ (e.g., `trans_depth` & `ct_flw_http_mthd` at `0.9993`; `tcprtt`, `synack`, `ackdat` at `0.988 / 0.976`; `swin` & `dwin` at `0.9845`).
- Combined collinear clusters into directional ratios and unified indicators:
  - `mean_pkt_diff`, `byte_log_ratio`, `load_log_ratio`, `total_bytes_log`, `pkt_log_ratio`, `jit_log_ratio`
  - `sloss_per_pkt`, `dloss_per_pkt`, `synack_rtt_share`
  - `http_activity`, `tcp_window_active`
- Dropped the 12 redundant raw twins in `ig_analysis.py` so correlated clusters aren't double-counted.

### Step 5 — Compute IG globally AND only when active (`x != 0`)
- Marked discrete/code and categorical columns explicitly with `discrete_features=True` in `mutual_info_classif`.
- Computed both:
  - **`IG_global`** across all `80,832` train rows.
  - **`IG_active_only`** strictly on the subset of rows where the feature is active (`x != 0` for numeric features, `service != 'none'` for `service`).

---

## Key Results from `ig_analysis.py`

### Top 5 Most Informative (Global IG)
Our combined `log1p` symmetry ratios beat every raw column:
1. `mean_pkt_diff` — **`0.4754`**
2. `byte_log_ratio` — **`0.4671`**
3. `load_log_ratio` — **`0.4595`**
4. `total_bytes_log` — **`0.4544`**
5. `sbytes` — **`0.4512`**

### Biggest jump when evaluated on Active Rows Only (`x != 0`)
- **`response_body_len`**: jumped from **`0.0575` (global)** to **`0.5352` (active-only)** across its `8,588` active rows (`10.6%` of train, `57.9%` attacks) — making it the **strongest single feature in the entire dataset** whenever HTTP response data is present.
- **`tcprtt`**: `0.3007` $\to$ `0.4034` (`+0.103`)
- **`synack_rtt_share`**: `0.2613` $\to$ `0.3479` (`+0.087`)
- **`sloss_per_pkt`**: `0.2380` $\to$ `0.3177` (`+0.080`)
- **`service`**: `0.0522` $\to$ `0.1093` when `service != 'none'`

### Bottom 5 Least Informative (Global IG)
1. `ct_ftp_cmd` (`0.0000` global, `0.0081` active on `1.9%` of rows)
2. `tcp_window_active` (`0.0006` global)
3. `is_sm_ips_ports` (`0.0036` global — 433 active rows, all Normal in dedup train)
4. `proto` (`0.0152` global — `98%` of rows are `tcp`/`udp` with a ~50/50 split, while the 128 rare protocols are `96.8%` attacks)
5. `ct_src_dport_ltm` (`0.0190` global)

---

## Reusable Outputs for Next Steps

Running `python script.py` exports three clean, `log1p`-transformed, engineered CSVs into `dataset/`:
- `UNSW_NB15_engineered_train.csv`
- `UNSW_NB15_engineered_val.csv`
- `UNSW_NB15_engineered_test.csv`

All three keep `proto`, `service`, `state` as clean strings (ready for `OneHotEncoder(min_frequency=0.005)` or tree-native encoding) alongside `attack_cat` and `label`.
