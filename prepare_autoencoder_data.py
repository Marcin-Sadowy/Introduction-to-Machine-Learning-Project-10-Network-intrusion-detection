"""
prepare_autoencoder_data.py — Final Dataset Builder for Autoencoder Modeling
============================================================================

Run:    python prepare_autoencoder_data.py
Output: dataset/autoencoder_ready/
            X_train_normal.csv      (Normal-only rows for Autoencoder training)
            X_train_all.csv         (Full clean train features)
            y_train_all.csv         (label + attack_cat for train)
            X_val.csv / y_val.csv   (Validation split for threshold tuning)
            X_test.csv / y_test.csv (Untouched official test set)
            preprocessor.pkl        (Fitted encoders, scaler & column metadata)
        outputs/ae_prep_log.txt
        plots/ae_dataset_overview.png
"""

import os
import pickle
import sys
import warnings
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)

ROOT         = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(ROOT, "dataset")
DATA_DIR     = DATASET_DIR if os.path.exists(os.path.join(DATASET_DIR, "UNSW_NB15_training-set.csv")) \
               else os.path.join(ROOT, "orange", "dataset")
AE_OUT_DIR   = os.path.join(DATASET_DIR, "autoencoder_ready")
PLOT_DIR     = os.path.join(ROOT, "plots")
OUT_DIR      = os.path.join(ROOT, "outputs")
RANDOM_STATE = 42

CAT_COLS  = ["proto", "service", "state"]
TARGETS   = ["label", "attack_cat"]
LEAK_COLS = ["id", "is_ftp_login", "stcpb", "dtcpb"]

# Discrete/code columns skipped by log1p
NO_LOG_COLS = {
    "sttl", "dttl", "ct_state_ttl",
    "is_sm_ips_ports", "tcp_window_active", "http_activity",
}

# Collinear twins replaced by combined domain features (avoids double-weighting MSE)
REDUNDANT_TWINS = [
    "trans_depth", "ct_flw_http_mthd",   # rho = 0.999 -> http_activity
    "synack", "ackdat",                  # rho = 0.988 / 0.976 with tcprtt -> synack_rtt_share
    "swin", "dwin",                      # rho = 0.985 -> tcp_window_active
    "sloss", "dloss",                    # rho = 0.954 / 0.962 -> sloss/dloss_per_pkt
    "dpkts",                             # rho = 0.963 with dbytes -> pkt_log_ratio
    "sinpkt", "sload",                   # rho = -0.960 / 0.936 with rate -> load_log_ratio
    "djit",                              # correlated with sjit -> jit_log_ratio
]

for _d in (AE_OUT_DIR, PLOT_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Logging helper
# ═══════════════════════════════════════════════════════════════════════════════

class _Tee:
    def __init__(self, path, stdout):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


@contextmanager
def log_to_file(filename):
    orig = sys.stdout
    tee = _Tee(os.path.join(OUT_DIR, filename), orig)
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = orig
        tee.file.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Load, Clean, Deduplicate & Split
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_clean_splits():
    """Load official CSVs, drop leaks, deduplicate train only, and stratify 80/20."""
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"))
    test_raw  = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"))

    # Drop row-order leak (id), exact duplicate (is_ftp_login), and random TCP seqs
    train = train_raw.drop(columns=LEAK_COLS).copy()
    test  = test_raw.drop(columns=LEAK_COLS).copy()

    for df in (train, test):
        for c in CAT_COLS + ["attack_cat"]:
            df[c] = df[c].astype(str).str.strip()
        df["service"] = df["service"].replace("-", "none")

    # Deduplicate train via majority vote (leave official test 100% untouched)
    feat_cols = [c for c in train.columns if c not in TARGETS]
    train["_key"] = pd.util.hash_pandas_object(train[feat_cols], index=False).values
    votes = (
        train.groupby(["_key", "attack_cat"]).size()
        .reset_index(name="n")
        .sort_values(["_key", "n"], ascending=[True, False])
    )
    majority = votes.drop_duplicates("_key")[["_key", "attack_cat"]]
    train = train.drop(columns=TARGETS).drop_duplicates("_key").merge(majority, on="_key", how="left")
    train["label"] = (train["attack_cat"] != "Normal").astype(int)
    train = train.drop(columns="_key").reset_index(drop=True)

    train_df, val_df = train_test_split(
        train, test_size=0.2, random_state=RANDOM_STATE, stratify=train["attack_cat"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test.reset_index(drop=True)

    print(f"1. Loaded & cleaned splits (leaks dropped: {LEAK_COLS})")
    print(f"   raw train: {len(train_raw):,} -> dedup: {len(train):,} -> train: {len(train_df):,} | val: {len(val_df):,}")
    print(f"   official test (untouched): {len(test_df):,}")
    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Feature Engineering & Collinear Twin Removal
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_and_prune_features(df):
    """Combine correlated clusters into domain ratios and drop collinear twins."""
    eps = 1e-6
    out = df.copy()

    out["fe_byte_log_ratio"]   = np.log1p(out["sbytes"]) - np.log1p(out["dbytes"])
    out["fe_total_bytes_log"]  = np.log1p(out["sbytes"] + out["dbytes"])
    out["fe_pkt_log_ratio"]    = np.log1p(out["spkts"])  - np.log1p(out["dpkts"])
    out["fe_load_log_ratio"]   = np.log1p(out["sload"])  - np.log1p(out["dload"])
    out["fe_mean_pkt_diff"]    = np.log1p(out["smean"])  - np.log1p(out["dmean"])
    out["fe_jit_log_ratio"]    = np.log1p(out["sjit"])   - np.log1p(out["djit"])
    out["fe_sloss_per_pkt"]    = out["sloss"]  / (out["spkts"] + 1.0)
    out["fe_dloss_per_pkt"]    = out["dloss"]  / (out["dpkts"] + 1.0)
    out["fe_synack_rtt_share"] = out["synack"] / (out["tcprtt"] + eps)
    out["fe_srv_src_conc"]     = out["ct_srv_src"]       / (out["ct_src_ltm"] + 1.0)
    out["fe_dport_conc"]       = out["ct_src_dport_ltm"] / (out["ct_src_ltm"] + 1.0)
    out["http_activity"]       = out["trans_depth"] + out["ct_flw_http_mthd"]
    out["tcp_window_active"]   = ((out["swin"] > 0) | (out["dwin"] > 0)).astype(int)

    return out.drop(columns=REDUNDANT_TWINS)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. log1p Transform, Compact One-Hot Encoding & Normal-Only Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def select_log_columns(train_df, num_cols):
    """Pick positive skewed numeric features (|skew| > 1) on train for log1p."""
    skew = train_df[num_cols].skew()
    return [
        c for c in num_cols
        if c not in NO_LOG_COLS
        and not c.startswith("fe_")
        and train_df[c].min() >= 0
        and abs(skew[c]) > 1.0
    ]


def transform_for_autoencoder(train_df, val_df, test_df):
    """Apply log1p, compact OHE (min_frequency=0.005), and StandardScaler fit on Normal train."""
    num_cols = [c for c in train_df.columns if c not in CAT_COLS + TARGETS]
    log_cols = select_log_columns(train_df, num_cols)

    # Apply log1p to skewed numeric features
    X_tr_num = train_df[num_cols].copy()
    X_va_num = val_df[num_cols].copy()
    X_te_num = test_df[num_cols].copy()
    for X_num in (X_tr_num, X_va_num, X_te_num):
        X_num[log_cols] = np.log1p(X_num[log_cols])

    # Fit StandardScaler strictly on Normal traffic in train (for Autoencoder training)
    normal_mask = (train_df["label"] == 0).values
    scaler = StandardScaler().fit(X_tr_num.loc[normal_mask])

    X_tr_scaled = pd.DataFrame(scaler.transform(X_tr_num), columns=num_cols)
    X_va_scaled = pd.DataFrame(scaler.transform(X_va_num), columns=num_cols)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te_num), columns=num_cols)

    # Compact OHE on strings (fit on train only; keep in [0, 1] so rare flags don't explode MSE)
    ohe = OneHotEncoder(min_frequency=0.005, handle_unknown="infrequent_if_exist", sparse_output=False)
    ohe.fit(train_df[CAT_COLS])
    cat_names = list(ohe.get_feature_names_out(CAT_COLS))

    X_tr_cat = pd.DataFrame(ohe.transform(train_df[CAT_COLS]), columns=cat_names)
    X_va_cat = pd.DataFrame(ohe.transform(val_df[CAT_COLS]),   columns=cat_names)
    X_te_cat = pd.DataFrame(ohe.transform(test_df[CAT_COLS]),  columns=cat_names)

    X_tr = pd.concat([X_tr_scaled, X_tr_cat], axis=1)
    X_va = pd.concat([X_va_scaled, X_va_cat], axis=1)
    X_te = pd.concat([X_te_scaled, X_te_cat], axis=1)

    meta = {
        "num_cols": num_cols,
        "log_cols": log_cols,
        "cat_cols": cat_names,
        "feature_names": list(X_tr.columns),
        "scaler": scaler,
        "ohe": ohe,
    }
    print(f"\n2. Transformed features for Autoencoder:")
    print(f"   numeric cols: {len(num_cols)} ({len(log_cols)} log1p-transformed, scaled on Normal train)")
    print(f"   compact OHE : {len(cat_names)} cols (min_frequency=0.005, kept in [0, 1])")
    print(f"   total input : {X_tr.shape[1]} features")
    return X_tr, X_va, X_te, normal_mask, meta


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Sanity Checks & Visual Summary
# ═══════════════════════════════════════════════════════════════════════════════

def run_sanity_checks(X_tr, X_va, X_te, normal_mask, val_df, test_df, meta):
    """Catch NaN/Inf, leakage, scaling bugs, and the all-zero OHE bug from earlier notebooks."""
    print("\n3. Running sanity checks:")
    for name, X in (("X_train_all", X_tr), ("X_val", X_va), ("X_test", X_te)):
        n_nan = int(X.isna().sum().sum())
        n_inf = int(np.isinf(X.values).sum())
        assert n_nan == 0 and n_inf == 0, f"{name} contains NaN/Inf!"
        print(f"   [OK] {name:12s} shape={str(X.shape):14s} | NaN={n_nan} | Inf={n_inf}")

    # Ensure leaks are truly absent
    for forbidden in ("id", "attack_cat", "label", "stcpb", "dtcpb", "is_ftp_login"):
        assert forbidden not in X_tr.columns, f"Leak column '{forbidden}' found in X!"
    print("   [OK] Zero leak columns in feature matrix")

    # Ensure OHE worked on val and test (catching the .cat.codes mismatch bug)
    cat_cols = meta["cat_cols"]
    for name, X in (("X_train", X_tr), ("X_val", X_va), ("X_test", X_te)):
        row_sums = X[cat_cols].sum(axis=1)
        assert (row_sums == len(CAT_COLS)).all(), f"{name} has zeroed-out categorical rows!"
    print(f"   [OK] OHE active on 100% of train/val/test rows (each row sums to {len(CAT_COLS)})")

    # Verify Normal-only scaling stats
    num_cols = meta["num_cols"]
    tr_norm_means = X_tr.loc[normal_mask, num_cols].mean().abs().max()
    tr_norm_stds  = X_tr.loc[normal_mask, num_cols].std().max()
    print(f"   [OK] Normal-only train scaling: max |mean|={tr_norm_means:.4f}, max std={tr_norm_stds:.4f}")

    # Show Normal vs Attack distribution shift on validation (quick preview of AE separability)
    val_norm = X_va.loc[val_df["label"] == 0, num_cols].abs().mean(axis=1).mean()
    val_att  = X_va.loc[val_df["label"] == 1, num_cols].abs().mean(axis=1).mean()
    print(f"   [INFO] Mean |z-score| on validation: Normal={val_norm:.3f} vs Attack={val_att:.3f}")


def plot_ae_overview(X_tr, X_va, normal_mask, val_df, test_df, meta):
    """Plot Normal vs Attack z-score profiles on key features and split composition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Mean absolute z-score difference (Attack vs Normal on val) for top 12 features
    num_cols = meta["num_cols"]
    norm_mag = X_va.loc[val_df["label"] == 0, num_cols].abs().mean()
    att_mag  = X_va.loc[val_df["label"] == 1, num_cols].abs().mean()
    diff = (att_mag - norm_mag).sort_values(ascending=False).head(12).iloc[::-1]

    yy = np.arange(len(diff))
    axes[0].barh(yy - 0.2, norm_mag[diff.index], 0.4, label="Normal (val)", color="#2b6cb0")
    axes[0].barh(yy + 0.2, att_mag[diff.index],  0.4, label="Attack (val)", color="#e53e3e")
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(diff.index, fontsize=9)
    axes[0].set_xlabel("Mean |z-score| relative to Normal train")
    axes[0].set_title("Top 12 features where Attacks deviate from Normal")
    axes[0].legend()

    # Right: Row counts across Autoencoder splits
    labels = ["Train\n(Normal only)", "Train\n(All)", "Val\n(Normal)", "Val\n(Attack)", "Test\n(Normal)", "Test\n(Attack)"]
    counts = [
        int(normal_mask.sum()),
        len(X_tr),
        int((val_df["label"] == 0).sum()),
        int((val_df["label"] == 1).sum()),
        int((test_df["label"] == 0).sum()),
        int((test_df["label"] == 1).sum()),
    ]
    colors = ["#2b6cb0", "#718096", "#3182ce", "#e53e3e", "#3182ce", "#c53030"]
    bars = axes[1].bar(labels, counts, color=colors)
    for bar, cnt in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, cnt + 800, f"{cnt:,}", ha="center", fontsize=8)
    axes[1].set_ylabel("Rows")
    axes[1].set_title("Autoencoder Dataset Splits (Normal vs Attack)")

    fig.suptitle("Final Autoencoder Dataset Overview", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOT_DIR, "ae_dataset_overview.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("  [plot saved] plots/ae_dataset_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Export Final Files
# ═══════════════════════════════════════════════════════════════════════════════

def export_autoencoder_files(X_tr, X_va, X_te, normal_mask, train_df, val_df, test_df, meta):
    """Save ready-to-load CSVs and preprocessor.pkl to dataset/autoencoder_ready/."""
    print("\n4. Exporting files to dataset/autoencoder_ready/:")
    files = [
        ("X_train_normal.csv", X_tr.loc[normal_mask].reset_index(drop=True)),
        ("X_train_all.csv",    X_tr),
        ("y_train_all.csv",    train_df[TARGETS]),
        ("X_val.csv",          X_va),
        ("y_val.csv",          val_df[TARGETS]),
        ("X_test.csv",         X_te),
        ("y_test.csv",         test_df[TARGETS]),
    ]
    for fname, df in files:
        path = os.path.join(AE_OUT_DIR, fname)
        df.to_csv(path, index=False)
        print(f"   [saved] {fname:20s} shape={df.shape}")

    pkl_path = os.path.join(AE_OUT_DIR, "preprocessor.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(meta, f)
    print("   [saved] preprocessor.pkl     (scaler, OHE, log_cols, feature_names)")


def main():
    with log_to_file("ae_prep_log.txt"):
        print("=" * 80)
        print("PREPARING FINAL DATASET FOR AUTOENCODER MODELING")
        print("=" * 80)

        train_df, val_df, test_df = load_and_clean_splits()

        train_fe = engineer_and_prune_features(train_df)
        val_fe   = engineer_and_prune_features(val_df)
        test_fe  = engineer_and_prune_features(test_df)

        X_tr, X_va, X_te, normal_mask, meta = transform_for_autoencoder(train_fe, val_fe, test_fe)
        run_sanity_checks(X_tr, X_va, X_te, normal_mask, val_df, test_df, meta)
        plot_ae_overview(X_tr, X_va, normal_mask, val_df, test_df, meta)
        export_autoencoder_files(X_tr, X_va, X_te, normal_mask, train_df, val_df, test_df, meta)

        print("\nDone! Ready for Autoencoder training on X_train_normal.csv.")


if __name__ == "__main__":
    main()
