"""
preprocessing/02_information_gain_analysis.py — Information Gain check
======================================================================

Run:    python preprocessing/02_information_gain_analysis.py
Output: outputs/ig_analysis_log.txt
        plots/ig_analysis_global_vs_active.png
        plots/ig_analysis_correlation_before_after.png

Quick checklist:
  1. Drop leaks & random IDs (id, attack_cat, is_ftp_login, stcpb, dtcpb)
  2. Deduplicate train rows so repeated floods don't skew IG
  3. Apply log1p to skewed positive numeric columns (skip binary & code-like)
  4. Combine strongly correlated pairs (|Spearman rho| >= 0.90)
  5. Check IG both globally and only on active rows (x != 0)
"""

import os
import sys
import warnings
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 120)

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(ROOT, "dataset") if os.path.exists(os.path.join(ROOT, "dataset", "UNSW_NB15_training-set.csv")) \
               else os.path.join(ROOT, "orange", "dataset")
TRAIN_CSV    = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
PLOT_DIR     = os.path.join(ROOT, "plots")
OUT_DIR      = os.path.join(ROOT, "outputs")
RANDOM_STATE = 42

CAT_COLS     = ["proto", "service", "state"]
TARGETS      = ["label", "attack_cat"]
LEAK_COLS    = ["id", "is_ftp_login", "stcpb", "dtcpb"]
CODE_LIKE    = ["sttl", "dttl", "ct_state_ttl"]
DISCRETE_NEW = {"tcp_window_active", "http_activity", "is_sm_ips_ports"}

for _d in (PLOT_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Logging & plotting helpers
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


def save_plot(fig, filename, title):
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot saved] plots/{filename}")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Remove target leaks & 2. Remove duplicates
# ═══════════════════════════════════════════════════════════════════════════════

def load_clean_train():
    """Load raw train CSV, drop target leaks/noise, deduplicate, and split."""
    df = pd.read_csv(TRAIN_CSV)
    n_raw = len(df)

    # Remove row-order leak (id), exact duplicate (is_ftp_login), and random TCP seqs
    df = df.drop(columns=LEAK_COLS).copy()
    for c in CAT_COLS + ["attack_cat"]:
        df[c] = df[c].astype(str).str.strip()
    df["service"] = df["service"].replace("-", "none")

    # Deduplicate feature vectors using majority vote for conflicting labels
    feat_cols = [c for c in df.columns if c not in TARGETS]
    df["_key"] = pd.util.hash_pandas_object(df[feat_cols], index=False).values
    votes = (
        df.groupby(["_key", "attack_cat"]).size()
        .reset_index(name="n")
        .sort_values(["_key", "n"], ascending=[True, False])
    )
    majority = votes.drop_duplicates("_key")[["_key", "attack_cat"]]
    df = df.drop(columns=TARGETS).drop_duplicates("_key").merge(majority, on="_key", how="left")
    df["label"] = (df["attack_cat"] != "Normal").astype(int)
    df = df.drop(columns="_key").reset_index(drop=True)

    # Stratified 80/20 split so IG is computed strictly on the training fold
    train_df, _ = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["attack_cat"]
    )
    train_df = train_df.reset_index(drop=True)

    print(f"1. Leaks removed : {LEAK_COLS} + attack_cat kept strictly as target")
    print(f"2. Deduplicated  : {n_raw:,} -> {len(df):,} unique rows | train fold = {len(train_df):,}")
    print(f"   Baseline attack rate in train fold: {train_df['label'].mean():.3f}")
    return train_df


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Combine correlated features
# ═══════════════════════════════════════════════════════════════════════════════

def find_correlated_pairs(df, cols, threshold=0.90):
    """Return all numeric feature pairs with |Spearman rho| >= threshold."""
    corr = df[cols].corr(method="spearman")
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((cols[i], cols[j], round(r, 4)))
    pairs_df = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "spearman_rho"])
    return corr, pairs_df.sort_values("spearman_rho", key=abs, ascending=False)


def combine_correlated_features(df):
    """Replace collinear feature clusters with compact, combined domain features."""
    eps = 1e-6
    out = df.copy()

    # Combine correlated clusters into symmetry ratios / unified indicators
    out["byte_log_ratio"]    = np.log1p(out["sbytes"]) - np.log1p(out["dbytes"])
    out["total_bytes_log"]   = np.log1p(out["sbytes"] + out["dbytes"])
    out["pkt_log_ratio"]     = np.log1p(out["spkts"])  - np.log1p(out["dpkts"])
    out["load_log_ratio"]    = np.log1p(out["sload"])  - np.log1p(out["dload"])
    out["mean_pkt_diff"]     = np.log1p(out["smean"])  - np.log1p(out["dmean"])
    out["jit_log_ratio"]     = np.log1p(out["sjit"])   - np.log1p(out["djit"])
    out["sloss_per_pkt"]     = out["sloss"]  / (out["spkts"] + 1.0)
    out["dloss_per_pkt"]     = out["dloss"]  / (out["dpkts"] + 1.0)
    out["synack_rtt_share"]  = out["synack"] / (out["tcprtt"] + eps)
    out["http_activity"]     = out["trans_depth"] + out["ct_flw_http_mthd"]
    out["tcp_window_active"] = ((out["swin"] > 0) | (out["dwin"] > 0)).astype(int)

    # Drop the redundant collinear twins now represented by the combined features
    redundant_twins = [
        "trans_depth", "ct_flw_http_mthd",   # rho = 0.999 -> http_activity
        "synack", "ackdat",                  # rho = 0.988 / 0.976 with tcprtt -> synack_rtt_share
        "swin", "dwin",                      # rho = 0.985 -> tcp_window_active
        "sloss", "dloss",                    # rho = 0.954 / 0.962 -> sloss/dloss_per_pkt
        "dpkts",                             # rho = 0.963 with dbytes -> pkt_log_ratio
        "sinpkt", "sload",                   # rho = -0.960 / 0.936 with rate -> load_log_ratio
        "djit",                              # correlated with sjit -> jit_log_ratio
    ]
    out = out.drop(columns=redundant_twins)
    return out, redundant_twins


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Information Gain: Global vs Active-Only (x > 0)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_global_and_active_ig(df, feature_cols):
    """Compute both global IG and active-only IG (restricted to rows where feature is active)."""
    y = df["label"].values
    X_enc = df[feature_cols].copy()

    # Apply log1p to skewed positive continuous features (monotone, helps kNN MI estimator)
    discrete_set = set(CODE_LIKE) | set(CAT_COLS) | DISCRETE_NEW
    for c in feature_cols:
        if c not in discrete_set and X_enc[c].min() >= 0 and abs(X_enc[c].skew()) > 1.0:
            X_enc[c] = np.log1p(X_enc[c])

    # Ordinal-encode categoricals on this working copy only
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_enc[CAT_COLS] = ord_enc.fit_transform(X_enc[CAT_COLS])

    # Global IG across all rows
    discrete_mask = [c in discrete_set for c in feature_cols]
    mi_global = mutual_info_classif(
        X_enc[feature_cols], y, discrete_features=discrete_mask, random_state=RANDOM_STATE
    )

    # Active-only IG: evaluate feature strictly on rows where it is active
    rows = []
    for c, ig_all in zip(feature_cols, mi_global):
        if c == "service":
            active_mask = (df[c] != "none").values
        elif c in CAT_COLS:
            active_mask = np.ones(len(df), dtype=bool)
        else:
            active_mask = (df[c] != 0).values

        n_active = int(active_mask.sum())
        active_pct = 100.0 * n_active / len(df)
        p_att_active = float(y[active_mask].mean()) if n_active > 0 else np.nan

        # Compute IG strictly inside the active subset when both classes exist
        if n_active >= 30 and len(np.unique(y[active_mask])) == 2:
            if active_pct >= 99.5:
                ig_active = ig_all
            else:
                ig_active = float(mutual_info_classif(
                    X_enc.loc[active_mask, [c]], y[active_mask],
                    discrete_features=[c in discrete_set],
                    random_state=RANDOM_STATE,
                )[0])
        else:
            ig_active = 0.0

        rows.append({
            "feature":           c,
            "IG_global":         round(ig_all, 5),
            "IG_active_only":    round(ig_active, 5),
            "active_%":          round(active_pct, 1),
            "n_active":          n_active,
            "P(attack|active)":  round(p_att_active, 3),
            "IG_gain_if_active": round(ig_active - ig_all, 5),
        })

    return pd.DataFrame(rows).set_index("feature").sort_values("IG_global", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main analysis runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_ig_analysis():
    with log_to_file("ig_analysis_log.txt"):
        print("=" * 80)
        print("DEDICATED INFORMATION GAIN ANALYSIS (CLEAN, DEDUP, COMBINED, ACTIVE-AWARE)")
        print("=" * 80)

        # Steps 1 & 2: clean leaks + deduplicate
        train_df = load_clean_train()
        raw_num_cols = [c for c in train_df.columns if c not in CAT_COLS + TARGETS]

        # Step 3: inspect correlation before & after combining features
        corr_before, pairs_before = find_correlated_pairs(train_df, raw_num_cols, threshold=0.90)
        print(f"\n3. Correlated numeric pairs BEFORE combining (|rho| >= 0.90): {len(pairs_before)}")
        print(pairs_before.to_string(index=False))

        comb_df, dropped = combine_correlated_features(train_df)
        comb_num_cols = [c for c in comb_df.columns if c not in CAT_COLS + TARGETS]
        corr_after, pairs_after = find_correlated_pairs(comb_df, comb_num_cols, threshold=0.90)
        print(f"\n   Dropped {len(dropped)} collinear twins -> {len(comb_num_cols)} numeric + {len(CAT_COLS)} categorical features")
        print(f"   Remaining pairs with |rho| >= 0.90 AFTER combining: {len(pairs_after)}")
        if len(pairs_after):
            print(pairs_after.to_string(index=False))

        # Plot correlation before vs after
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for ax, mat, cols, ttl in [
            (axes[0], corr_before, raw_num_cols,  f"Before combining ({len(raw_num_cols)} numeric)"),
            (axes[1], corr_after,  comb_num_cols, f"After combining ({len(comb_num_cols)} numeric)"),
        ]:
            im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=7)
            ax.set_yticks(range(len(cols)))
            ax.set_yticklabels(cols, fontsize=7)
            ax.set_title(ttl)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Spearman rho")
        save_plot(fig, "ig_analysis_correlation_before_after.png", "Spearman Correlation: Before vs After Combining")

        # Step 4: Global IG vs Active-Only IG
        feature_cols = comb_num_cols + CAT_COLS
        ig_df = compute_global_and_active_ig(comb_df, feature_cols)

        print("\n4. Full Information Gain Table (Global vs Active-Only):")
        print(ig_df.to_string())

        print("\n── Top 5 Most Informative (Global IG) ──")
        print(ig_df.head(5).to_string())

        print("\n── Top 5 Least Informative (Global IG) ──")
        print(ig_df.tail(5).to_string())

        sparse = ig_df[ig_df["active_%"] < 85.0].sort_values("IG_active_only", ascending=False)
        print("\n── Sparse / Conditionally Active Features (<85% active): Global vs Active-Only IG ──")
        print(sparse.to_string())

        # Plot Global IG vs Active-Only IG
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))

        s_all = ig_df.sort_values("IG_global")
        yy = np.arange(len(s_all))
        axes[0].barh(yy - 0.2, s_all["IG_global"],      0.4, label="IG (all rows)",    color="#2b6cb0")
        axes[0].barh(yy + 0.2, s_all["IG_active_only"], 0.4, label="IG (active rows)", color="#dd6b20")
        axes[0].axvline(0.05, color="black", ls="--", lw=1, label="0.05 threshold")
        axes[0].set_yticks(yy)
        axes[0].set_yticklabels(s_all.index, fontsize=8)
        axes[0].set_xlabel("Mutual Information (nats)")
        axes[0].set_title("Global IG vs Active-Only IG (all combined features)")
        axes[0].legend(loc="lower right", fontsize=8)

        s_sp = sparse.sort_values("IG_active_only")
        yy2 = np.arange(len(s_sp))
        axes[1].barh(yy2 - 0.2, s_sp["IG_global"],      0.4, label="IG (all rows)",    color="#2b6cb0")
        axes[1].barh(yy2 + 0.2, s_sp["IG_active_only"], 0.4, label="IG (active rows)", color="#dd6b20")
        for k, (_, row) in enumerate(s_sp.iterrows()):
            axes[1].text(
                max(row["IG_global"], row["IG_active_only"]) + 0.006, k,
                f"active={row['active_%']:.1f}% | P(att)={row['P(attack|active)']:.2f}",
                va="center", fontsize=8,
            )
        axes[1].set_yticks(yy2)
        axes[1].set_yticklabels(s_sp.index, fontsize=9)
        axes[1].set_xlim(0, max(s_sp["IG_active_only"].max(), 0.5) + 0.20)
        axes[1].set_xlabel("Mutual Information (nats)")
        axes[1].set_title("Sparse features: signal when active vs diluted globally")
        axes[1].legend(loc="lower right", fontsize=8)

        save_plot(fig, "ig_analysis_global_vs_active.png", "Information Gain Analysis: Global vs Active-Only")


if __name__ == "__main__":
    run_ig_analysis()
