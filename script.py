"""
script.py — UNSW-NB15 Preprocessing & Feature Engineering
=========================================================

Run:    python script.py            (runs steps listed in RUN_STEPS)
        python script.py 1 2 3      (runs specific steps)
        python script.py all        (runs all steps 1..5)
        python script.py report     (rebuilds combined PDF only)
        python script.py clean      (wipes temp cache & __pycache__)

Saves logs to outputs/, plots to plots/, and final engineered CSVs to dataset/.
"""

import json
import os
import pickle
import shutil
import sys
import time
import warnings
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 120)

ROOT         = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR   = os.path.join(ROOT, "dataset")
DATA_DIR     = EXPORT_DIR if os.path.exists(os.path.join(EXPORT_DIR, "UNSW_NB15_training-set.csv")) \
               else os.path.join(ROOT, "orange", "dataset")
PLOT_DIR     = os.path.join(ROOT, "plots")
OUT_DIR      = os.path.join(ROOT, "outputs")
CACHE_DIR    = os.path.join(OUT_DIR, "cache")
MANIFEST     = os.path.join(OUT_DIR, "plot_manifest.json")

CAT_COLS     = ["proto", "service", "state"]
TARGETS      = ["label", "attack_cat"]
CODE_LIKE    = ["sttl", "dttl", "swin", "dwin", "ct_state_ttl"]
RANDOM_STATE = 42

PROTO_FAMILIES = {
    "transport": {"tcp", "udp", "sctp"},
    "control":   {"icmp", "igmp", "arp"},
    "routing":   {"ospf", "egp", "igp", "eigrp", "isis", "idrp", "pim", "vrrp", "irtp", "rsvp", "bgp"},
    "tunnel_ip": {
        "gre", "ipip", "ipv6", "ipv6-frag", "ipv6-route", "ipv6-opts",
        "ipv6-no", "ipnip", "encap", "etherip", "mobile", "swipe", "ip", "any", "ipcomp",
    },
    "unas":      {"unas"},
}
STATE_KEEP = {"FIN", "CON", "INT", "REQ", "RST"}

for _d in (EXPORT_DIR, PLOT_DIR, OUT_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O, Logging, Caching & Plotting Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class Tee:
    """Mirror stdout to both console and a per-step log file."""
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
def step_logging(step_id):
    original = sys.stdout
    tee = Tee(os.path.join(OUT_DIR, f"step{step_id:02d}_log.txt"), original)
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = original
        tee.file.close()


def header(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def _load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {}


def reset_step_plots(step_id):
    m = _load_manifest()
    m[f"{step_id:02d}"] = []
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def save_fig(fig, name, step_title, step_id):
    """Save figure PNG and register it in plot_manifest.json for the combined PDF."""
    fig.suptitle(step_title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(PLOT_DIR, f"{name}.png"), dpi=110)
    plt.close(fig)

    m = _load_manifest()
    key = f"{step_id:02d}"
    m.setdefault(key, [])
    if name not in m[key]:
        m[key].append(name)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"  [plot saved] plots/{name}.png")


def build_report():
    """Assemble all registered step PNGs into one ordered PDF report."""
    m = _load_manifest()
    registered = {n for names in m.values() for n in names}
    orphans = sorted(
        (f for f in os.listdir(PLOT_DIR)
         if f.startswith("step") and f.endswith(".png") and f[:-4] not in registered),
        key=lambda f: os.path.getmtime(os.path.join(PLOT_DIR, f)),
    )
    for f in orphans:
        m.setdefault(f[4:6], []).append(f[:-4])

    pdf_path = os.path.join(PLOT_DIR, "preprocessing_steps.pdf")
    n = 0
    with PdfPages(pdf_path) as pdf:
        for step in sorted(m):
            for name in m[step]:
                png = os.path.join(PLOT_DIR, f"{name}.png")
                if not os.path.exists(png):
                    continue
                img = plt.imread(png)
                h, w = img.shape[:2]
                fig = plt.figure(figsize=(w / 110, h / 110))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
                n += 1
    print(f"\n[report] {n} plots from steps {sorted(m)} -> plots/preprocessing_steps.pdf")


def save_cache(name, obj):
    with open(os.path.join(CACHE_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(obj, f)


def load_cache(name, producer_step):
    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        print(f"[cache] '{name}' missing -> running step {producer_step} first")
        STEPS[str(producer_step)]()
    with open(path, "rb") as f:
        return pickle.load(f)


def load_raw():
    """Load official UNSW-NB15 train/test CSVs and feature descriptions."""
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"))
    test_raw  = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"))
    feat_desc = pd.read_csv(os.path.join(DATA_DIR, "UNSW-NB15_features.csv"), encoding="cp1252")
    feat_desc["Name"] = feat_desc["Name"].astype(str).str.strip().str.lower()
    desc = dict(zip(feat_desc["Name"], feat_desc["Description"]))
    return train_raw, test_raw, desc


def load_clean():
    d = load_cache("clean_split", 2)
    return d["train"], d["val"], d["test"], d["num_cols"]


def clean_workspace():
    """Remove temporary cache files and __pycache__ while keeping logs and plots."""
    for folder in (CACHE_DIR, os.path.join(ROOT, "__pycache__")):
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[clean] removed {os.path.relpath(folder, ROOT)}")
    os.makedirs(CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Raw Data Audit
# ═══════════════════════════════════════════════════════════════════════════════

def step1_audit():
    """Audit schema, missing placeholders, duplicates, class balance, id leak, and skew."""
    SID, STEP = 1, "STEP 1 - Raw data audit"
    with step_logging(SID):
        reset_step_plots(SID)
        header(STEP)
        train_raw, test_raw, _ = load_raw()

        print(f"train shape: {train_raw.shape}\ntest  shape: {test_raw.shape}")
        print(f"same columns in same order: {list(train_raw.columns) == list(test_raw.columns)}")

        # ── 1a. Column types & missing placeholders ───────────────────────────
        print("\n-- 1a. Column dtypes --")
        print(train_raw.dtypes.value_counts().to_string())
        cat_cols_raw = train_raw.select_dtypes(include="object").columns.tolist()
        print("object (categorical) columns:", cat_cols_raw)

        print("\n-- 1b. Missing values (NaN) --")
        na_tr, na_te = train_raw.isna().sum(), test_raw.isna().sum()
        print(f"train total NaN: {int(na_tr.sum())} | test total NaN: {int(na_te.sum())}")
        for c in cat_cols_raw:
            n_tr, n_te = (train_raw[c] == "-").sum(), (test_raw[c] == "-").sum()
            if n_tr or n_te:
                print(f"  '{c}' has '-' placeholder: train={n_tr} ({n_tr/len(train_raw):.1%}), "
                      f"test={n_te} ({n_te/len(test_raw):.1%})")

        # ── 1c. Duplicates & conflicting labels ───────────────────────────────
        print("\n-- 1c. Duplicates (ignoring 'id') --")
        no_id_tr, no_id_te = train_raw.drop(columns="id"), test_raw.drop(columns="id")
        feat_only = [c for c in no_id_tr.columns if c not in TARGETS]
        dup_tr, dup_te = no_id_tr.duplicated().sum(), no_id_te.duplicated().sum()
        print(f"exact duplicate rows  train: {dup_tr} ({dup_tr/len(train_raw):.1%})  "
              f"test: {dup_te} ({dup_te/len(test_raw):.1%})")

        grp_cat = no_id_tr.groupby(feat_only, dropna=False)["attack_cat"].nunique()
        grp_lbl = no_id_tr.groupby(feat_only, dropna=False)["label"].nunique()
        print(f"train feature-vectors mapped to >1 attack_cat: {(grp_cat > 1).sum()}")
        print(f"train feature-vectors mapped to both label 0 and 1: {(grp_lbl > 1).sum()}")

        merged = no_id_te[feat_only].merge(no_id_tr[feat_only].drop_duplicates(), how="inner")
        print(f"test rows also appearing in train: {len(merged)} ({len(merged)/len(test_raw):.1%})")
        print("train rows involved in duplication, by attack_cat:")
        print(no_id_tr[no_id_tr.duplicated(keep=False)]["attack_cat"].value_counts().to_string())

        # ── 1d-1g. Class balance, categoricals, id leak & numeric summary ─────
        print("\n-- 1d. Class balance --")
        lab = pd.DataFrame({
            "train": train_raw["label"].value_counts(normalize=True),
            "test":  test_raw["label"].value_counts(normalize=True),
        })
        print("label (0=normal, 1=attack) proportions:\n", lab.round(3).to_string())

        cat_counts = pd.DataFrame({
            "train_n": train_raw["attack_cat"].value_counts(),
            "test_n":  test_raw["attack_cat"].value_counts(),
        })
        cat_counts["train_%"] = (100 * cat_counts["train_n"] / len(train_raw)).round(2)
        cat_counts["test_%"]  = (100 * cat_counts["test_n"]  / len(test_raw)).round(2)
        print(cat_counts.sort_values("train_n", ascending=False).to_string())
        print("imbalance ratio (largest/smallest, train):",
              round(cat_counts["train_n"].max() / cat_counts["train_n"].min(), 1))

        print("\n-- 1e. Categorical features --")
        for c in CAT_COLS:
            tr_vals, te_vals = set(train_raw[c]), set(test_raw[c])
            vc = train_raw[c].value_counts(normalize=True)
            print(f"{c:8s} train unique={len(tr_vals):4d} test unique={len(te_vals):4d} "
                  f"| only-in-test={sorted(te_vals - tr_vals)[:10]} | <0.1% rows={(vc < 0.001).sum()}")

        print("\n-- 1f. Is 'id' a leak? --")
        print("id is 1..n in train:", (train_raw["id"].values == np.arange(1, len(train_raw) + 1)).all())
        print("correlation(id, label) train:", round(train_raw[["id", "label"]].corr().iloc[0, 1], 3))

        print("\n-- 1g. Numeric feature summary (train) --")
        num_cols_raw = [c for c in train_raw.select_dtypes(include=np.number).columns if c not in ("id", "label")]
        desc = train_raw[num_cols_raw].describe().T
        desc["n_unique"] = train_raw[num_cols_raw].nunique()
        desc["zero_%"]   = (100 * (train_raw[num_cols_raw] == 0).mean()).round(1)
        desc["skew"]     = train_raw[num_cols_raw].skew().round(2)
        print(desc[["min", "50%", "max", "n_unique", "zero_%", "skew"]].to_string())

        # ── Plots: class distribution, id leak, skew/sparsity, categoricals ───
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        cc = cat_counts.sort_values("train_n", ascending=False)
        x = np.arange(len(cc))
        axes[0].bar(x - 0.2, cc["train_n"], 0.4, label="train")
        axes[0].bar(x + 0.2, cc["test_n"],  0.4, label="test")
        axes[0].set_yscale("log")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cc.index, rotation=45, ha="right")
        axes[0].set_title("attack_cat counts (log scale)")
        axes[0].legend()
        axes[1].bar(x - 0.2, cc["train_%"], 0.4, label="train")
        axes[1].bar(x + 0.2, cc["test_%"],  0.4, label="test")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cc.index, rotation=45, ha="right")
        axes[1].set_title("attack_cat share (%)")
        axes[1].legend()
        save_fig(fig, "step01_class_distribution", f"{STEP} - class distribution", SID)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, df, col, ttl in [
            (axes[0], train_raw, "tab:blue",   "train"),
            (axes[1], test_raw,  "tab:orange", "test"),
        ]:
            codes = df["attack_cat"].astype("category")
            ax.scatter(df["id"], codes.cat.codes, s=1, alpha=0.3, color=col)
            ax.set_yticks(range(len(codes.cat.categories)))
            ax.set_yticklabels(codes.cat.categories)
            ax.set_xlabel("id")
            ax.set_title(f"{ttl}: attack_cat vs id")
        save_fig(fig, "step01_id_leakage", f"{STEP} - id ordering leak", SID)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        d = desc.sort_values("skew")
        axes[0].barh(d.index, d["skew"].clip(upper=100), color="tab:purple")
        axes[0].axvline(1, color="red", ls="--", lw=1)
        axes[0].set_title("Skewness of numeric features (clipped at 100)")
        d2 = desc.sort_values("zero_%")
        axes[1].barh(d2.index, d2["zero_%"], color="tab:green")
        axes[1].set_title("Sparsity: % of zeros")
        save_fig(fig, "step01_skew_and_zeros", f"{STEP} - skewness & sparsity", SID)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, c in zip(axes, CAT_COLS):
            vc_tr = train_raw[c].value_counts(normalize=True).head(12)
            vc_te = test_raw[c].value_counts(normalize=True).reindex(vc_tr.index).fillna(0)
            xx = np.arange(len(vc_tr))
            ax.bar(xx - 0.2, vc_tr.values, 0.4, label="train")
            ax.bar(xx + 0.2, vc_te.values, 0.4, label="test")
            ax.set_xticks(xx)
            ax.set_xticklabels(vc_tr.index, rotation=45, ha="right")
            ax.set_title(f"{c}: top-12 categories")
            ax.legend()
        save_fig(fig, "step01_categoricals", f"{STEP} - categorical features", SID)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Cleaning, Leak Removal, Deduplication & Stratified Split
# ═══════════════════════════════════════════════════════════════════════════════

def step2_clean_split():
    """Drop leaks/noise, deduplicate train via majority vote, and create 80/20 split."""
    SID, STEP = 2, "STEP 2 - Cleaning & split"
    with step_logging(SID):
        reset_step_plots(SID)
        header(STEP)
        train_raw, test_raw, _ = load_raw()
        train = train_raw.drop(columns="id").copy()
        test  = test_raw.drop(columns="id").copy()

        # ── 2a-2c. Clean strings, recode '-', drop duplicate/random columns ───
        for df in (train, test):
            for c in CAT_COLS + ["attack_cat"]:
                df[c] = df[c].astype(str).str.strip()
            df["service"] = df["service"].replace("-", "none")
        print("2a. dropped 'id'; service '-' -> 'none'")

        eq = (train["is_ftp_login"] == train["ct_ftp_cmd"]).mean()
        print(f"2b. share where is_ftp_login == ct_ftp_cmd: {eq:.5f} -> dropping is_ftp_login, stcpb, dtcpb")
        for df in (train, test):
            df.drop(columns=["is_ftp_login", "stcpb", "dtcpb"], inplace=True)

        # ── 2d. Deduplicate train (majority vote on conflicting labels) ───────
        feats = [c for c in train.columns if c not in TARGETS]
        before_counts = train["attack_cat"].value_counts()
        n_before = len(train)

        train["_key"] = pd.util.hash_pandas_object(train[feats], index=False).values
        votes = (
            train.groupby(["_key", "attack_cat"]).size()
            .reset_index(name="n")
            .sort_values(["_key", "n"], ascending=[True, False])
        )
        majority = votes.drop_duplicates("_key")[["_key", "attack_cat"]]
        train = train.drop(columns=TARGETS).drop_duplicates("_key").merge(majority, on="_key", how="left")
        train["label"] = (train["attack_cat"] != "Normal").astype(int)
        train = train.drop(columns="_key").reset_index(drop=True)

        after_counts = train["attack_cat"].value_counts()
        dd = pd.DataFrame({"before": before_counts, "after": after_counts})
        dd["kept_%"] = (100 * dd["after"] / dd["before"]).round(1)
        print(f"\n2d. train deduplication: {n_before} -> {len(train)} rows")
        print(dd.sort_values("before", ascending=False).to_string())

        # ── 2e. Stratified train / val split (official test stays untouched) ──
        train_df, val_df = train_test_split(
            train, test_size=0.2, random_state=RANDOM_STATE, stratify=train["attack_cat"]
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test.reset_index(drop=True)
        num_cols = [c for c in train_df.columns if c not in CAT_COLS + TARGETS]

        split_tab = pd.DataFrame({
            "train":          train_df["attack_cat"].value_counts(),
            "val":            val_df["attack_cat"].value_counts(),
            "test(official)": test_df["attack_cat"].value_counts(),
        })
        print(f"\n2e. final sets: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
        print(split_tab.sort_values("train", ascending=False).to_string())

        save_cache("clean_split", {"train": train_df, "val": val_df, "test": test_df, "num_cols": num_cols})

        # ── Plots: deduplication effect & split composition ───────────────────
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        order = dd.sort_values("before", ascending=False).index
        xx = np.arange(len(order))
        axes[0].bar(xx - 0.2, dd.loc[order, "before"], 0.4, label="raw train",     color="grey")
        axes[0].bar(xx + 0.2, dd.loc[order, "after"],  0.4, label="de-duplicated", color="tab:blue")
        axes[0].set_yscale("log")
        axes[0].set_xticks(xx)
        axes[0].set_xticklabels(order, rotation=45, ha="right")
        axes[0].set_title("Rows per class before / after de-duplication")
        axes[0].legend()
        axes[1].barh(order[::-1], dd.loc[order[::-1], "kept_%"], color="tab:blue")
        axes[1].set_title("Unique share per class (%)")
        save_fig(fig, "step02_deduplication", f"{STEP} - de-duplication", SID)

        fig, ax = plt.subplots(figsize=(13, 6))
        st_pct = 100 * split_tab.loc[order] / split_tab.loc[order].sum()
        for i, col in enumerate(st_pct.columns):
            ax.bar(xx + (i - 1) * 0.27, st_pct[col], 0.27, label=col)
        ax.set_xticks(xx)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel("% of set")
        ax.set_title("Class composition of train / val / official test")
        ax.legend()
        save_fig(fig, "step02_split_composition", f"{STEP} - final split composition", SID)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Distributions, log1p Transform, TTL Artefacts & Drift Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def step3_distributions():
    """Analyze skewness, apply log1p, inspect TTL artefacts, and measure train/test drift."""
    SID, STEP = 3, "STEP 3 - Distributions & transforms"
    train_df, val_df, test_df, num_cols = load_clean()

    with step_logging(SID):
        reset_step_plots(SID)
        header(STEP)

        # ── 3a. Identify skewed columns for log1p (skip binary & code-like) ───
        binary_cols = [c for c in num_cols if train_df[c].nunique() <= 2]
        skew_before = train_df[num_cols].skew()
        log_cols = [
            c for c in num_cols
            if c not in CODE_LIKE + binary_cols and train_df[c].min() >= 0 and abs(skew_before[c]) > 1
        ]

        logged = train_df[num_cols].copy()
        logged[log_cols] = np.log1p(logged[log_cols])
        sk = pd.DataFrame({"skew_before": skew_before, "skew_after": logged.skew()}).round(2)
        print("3a. skewness before / after log1p:")
        print(sk.loc[log_cols].sort_values("skew_before", ascending=False).to_string())

        # Skewness comparison plot
        fig, ax = plt.subplots(figsize=(14, 8))
        s = sk.loc[log_cols].sort_values("skew_before")
        yy = np.arange(len(s))
        ax.barh(yy - 0.2, s["skew_before"].clip(upper=60), 0.4, label="raw (clipped at 60)", color="grey")
        ax.barh(yy + 0.2, s["skew_after"],                 0.4, label="log1p",               color="tab:blue")
        ax.set_yticks(yy)
        ax.set_yticklabels(s.index)
        ax.axvline(1,  color="red", ls="--", lw=1)
        ax.axvline(-1, color="red", ls="--", lw=1)
        ax.legend()
        ax.set_title("Skewness before vs after log1p (train split)")
        save_fig(fig, "step03_skew_before_after", f"{STEP} - effect of log1p on skewness", SID)

        # Histograms raw vs log1p
        show = ["sbytes", "dur", "sload", "sinpkt", "smean", "ct_srv_src"]
        fig, axes = plt.subplots(2, len(show), figsize=(20, 7))
        for j, c in enumerate(show):
            axes[0, j].hist(train_df[c], bins=60, color="grey")
            axes[0, j].set_title(f"{c} raw")
            axes[0, j].set_yscale("log")
            axes[1, j].hist(logged[c], bins=60, color="tab:blue")
            axes[1, j].set_title(f"{c} log1p")
        save_fig(fig, "step03_hist_raw_vs_log", f"{STEP} - raw vs log1p histograms", SID)

        # ── 3b-3c. Per-class distributions & TTL lab artefacts ────────────────
        cls_order = train_df["attack_cat"].value_counts().index.tolist()
        box_feats = ["sbytes", "dbytes", "dur", "smean", "sload", "sinpkt", "ct_srv_src", "ct_dst_src_ltm"]
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        for ax, c in zip(axes.ravel(), box_feats):
            ax.boxplot([logged.loc[train_df["attack_cat"] == k, c].values for k in cls_order], showfliers=False)
            ax.set_xticks(range(1, len(cls_order) + 1))
            ax.set_xticklabels(cls_order, rotation=60, ha="right", fontsize=8)
            ax.set_title(f"{c} (log1p)")
        save_fig(fig, "step03_per_class_boxplots", f"{STEP} - per-class distributions", SID)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, c in zip(axes, ["sttl", "dttl", "ct_state_ttl"]):
            ct = pd.crosstab(train_df[c], train_df["attack_cat"])[cls_order]
            ct = ct[ct.sum(axis=1) > 50]
            ct.div(ct.sum(axis=1), axis=0).plot(
                kind="bar", stacked=True, ax=ax, legend=(c == "ct_state_ttl"), colormap="tab10"
            )
            ax.set_title(f"{c}: class mix per value")
        if axes[2].get_legend():
            axes[2].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        save_fig(fig, "step03_ttl_class_mix", f"{STEP} - TTL features vs class", SID)

        # ── 3d-3e. Train vs official test KS drift (overall & within-class) ───
        drift = pd.DataFrame([
            (
                c,
                ks_2samp(train_df[c], test_df[c]).statistic,
                ks_2samp(train_df.loc[train_df.label == 0, c], test_df.loc[test_df.label == 0, c]).statistic,
                ks_2samp(train_df.loc[train_df.label == 1, c], test_df.loc[test_df.label == 1, c]).statistic,
            )
            for c in num_cols
        ], columns=["feature", "KS_all", "KS_normal", "KS_attack"]).set_index("feature").round(3)

        fig, ax = plt.subplots(figsize=(14, 9))
        d = drift.sort_values("KS_all")
        yy = np.arange(len(d))
        ax.barh(yy - 0.27, d["KS_all"],    0.27, label="all rows")
        ax.barh(yy,        d["KS_normal"], 0.27, label="Normal only")
        ax.barh(yy + 0.27, d["KS_attack"], 0.27, label="Attacks only")
        ax.set_yticks(yy)
        ax.set_yticklabels(d.index, fontsize=8)
        ax.legend()
        ax.set_title("Distribution shift train -> official test (KS statistic)")
        save_fig(fig, "step03_train_test_drift", f"{STEP} - train vs test drift", SID)

        test_dd = test_df.drop_duplicates()
        classes = [k for k in cls_order
                   if (train_df.attack_cat == k).sum() >= 40 and (test_dd.attack_cat == k).sum() >= 40]
        pc = pd.DataFrame(
            {k: [ks_2samp(train_df.loc[train_df.attack_cat == k, c],
                          test_dd.loc[test_dd.attack_cat == k, c]).statistic for c in num_cols]
             for k in classes},
            index=num_cols,
        ).round(3)
        pc["max"] = pc.max(axis=1)
        pc = pc.sort_values("max", ascending=False)
        print("\n3e. median KS drift per class (deduplicated):", pc[classes].median().round(3).to_dict())

        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(pc[classes].values, aspect="auto", cmap="Reds", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(pc)))
        ax.set_yticklabels(pc.index, fontsize=8)
        fig.colorbar(im, ax=ax, label="KS statistic")
        ax.set_title("Within-class drift train -> test (both de-duplicated)")
        save_fig(fig, "step03_per_class_drift", f"{STEP} - drift within each class", SID)

        save_cache("transform_meta", {
            "log_cols": log_cols, "binary_cols": binary_cols,
            "drift": drift, "drift_per_class": pc,
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Categorical Encoding Analysis & Validation Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def domain_group(df):
    """Map rare protocols into networking families and group rare TCP states."""
    out = df[CAT_COLS].copy()
    lookup = {p: fam for fam, ps in PROTO_FAMILIES.items() for p in ps}
    out["proto"] = df["proto"].map(lambda p: p if p in ("tcp", "udp") else lookup.get(p, "other"))
    out["state"] = df["state"].where(df["state"].isin(STATE_KEEP), "other")
    return out


def encode_categoricals(method, tr, va):
    """Fit selected categorical encoder on train and transform validation."""
    def _df(arr, names):
        return pd.DataFrame(np.asarray(arr, dtype=float), columns=list(names))

    if method == "onehot_full":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return _df(enc.fit_transform(tr[CAT_COLS]), enc.get_feature_names_out()), \
               _df(enc.transform(va[CAT_COLS]),     enc.get_feature_names_out())

    if method.startswith("onehot_min"):
        freq = float(method.removesuffix("+freq").split("_")[2])
        enc = OneHotEncoder(min_frequency=freq, handle_unknown="infrequent_if_exist", sparse_output=False)
        A = _df(enc.fit_transform(tr[CAT_COLS]), enc.get_feature_names_out())
        B = _df(enc.transform(va[CAT_COLS]),     enc.get_feature_names_out())
        if method.endswith("+freq"):
            fa, fb = encode_categoricals("frequency", tr, va)
            A = pd.concat([A, fa[["proto_freq"]]], axis=1)
            B = pd.concat([B, fb[["proto_freq"]]], axis=1)
        return A, B

    if method == "domain_onehot":
        g_tr, g_va = domain_group(tr), domain_group(va)
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return _df(enc.fit_transform(g_tr), enc.get_feature_names_out()), \
               _df(enc.transform(g_va),     enc.get_feature_names_out())

    if method == "frequency":
        A, B = pd.DataFrame(), pd.DataFrame()
        for c in CAT_COLS:
            share = tr[c].value_counts(normalize=True)
            A[f"{c}_freq"] = tr[c].map(share).values
            B[f"{c}_freq"] = va[c].map(share).fillna(0.0).values
        return A, B

    if method == "target":
        enc = TargetEncoder(target_type="multiclass", random_state=RANDOM_STATE)
        return _df(enc.fit_transform(tr[CAT_COLS], tr["attack_cat"]), enc.get_feature_names_out()), \
               _df(enc.transform(va[CAT_COLS]),                       enc.get_feature_names_out())

    if method in ("ordinal", "native"):
        unk = np.nan if method == "native" else -1
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=unk)
        cols = [f"{c}_code" for c in CAT_COLS]
        return _df(enc.fit_transform(tr[CAT_COLS]), cols), _df(enc.transform(va[CAT_COLS]), cols)

    raise ValueError(f"Unknown encoding method: {method}")


def step4_categorical_encoding():
    """Inspect category attack rates and benchmark encoders on validation."""
    SID, STEP = 4, "STEP 4 - Categorical encoding"
    train_df, val_df, test_df, num_cols = load_clean()
    log_cols = load_cache("transform_meta", 3)["log_cols"]

    with step_logging(SID):
        reset_step_plots(SID)
        header(STEP)
        cls_order = train_df["attack_cat"].value_counts().index.tolist()

        # ── 4a. Category frequency & attack-rate audit ────────────────────────
        cat_tables = {}
        for c in CAT_COLS:
            g = train_df.groupby(c).agg(n=("label", "size"), attack_rate=("label", "mean"))
            g["share_%"] = (100 * g["n"] / len(train_df)).round(3)
            g = g.sort_values("n", ascending=False)
            cat_tables[c] = g
            print(f"\n{c} ({len(g)} categories, top 10):")
            print(g.head(10).round(3).to_string())

        g = cat_tables["proto"]
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        sc = axes[0].scatter(g["n"], g["attack_rate"], c=np.log10(g["n"]), cmap="viridis", s=25)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("rows in train (log)")
        axes[0].set_ylabel("attack rate")
        axes[0].set_title("proto: frequency vs attack rate")
        fig.colorbar(sc, ax=axes[0], label="log10(rows)")
        top = g.head(25)
        axes[1].bar(range(len(top)), top["attack_rate"], color="tab:red")
        axes[1].set_xticks(range(len(top)))
        axes[1].set_xticklabels(top.index, rotation=70, fontsize=8)
        axes[1].set_title("Attack rate of top-25 protocols")
        save_fig(fig, "step04_proto_analysis", f"{STEP} - proto rarity vs attack rate", SID)

        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        for ax, c in zip(axes, ["service", "state"]):
            ct = pd.crosstab(train_df[c], train_df["attack_cat"])[cls_order]
            ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
            ct.div(ct.sum(axis=1), axis=0).plot(
                kind="bar", stacked=True, ax=ax, colormap="tab10", legend=(c == "state")
            )
            ax.set_title(f"{c}: class mix per category")
        save_fig(fig, "step04_service_state_mix", f"{STEP} - service & state mix", SID)

        # ── 4b. Validation benchmark across encoder/model pairs ───────────────
        experiments = [
            ("onehot_full",        "onehot_full",      "logreg"),
            ("onehot_min0.1%",     "onehot_min_0.001", "logreg"),
            ("onehot_min0.5%",     "onehot_min_0.005", "logreg"),
            ("domain_onehot",      "domain_onehot",    "logreg"),
            ("frequency",          "frequency",        "logreg"),
            ("target(cross-fit)",  "target",           "logreg"),
            ("ordinal",            "ordinal",          "hgb"),
            ("native_categorical", "native",           "hgb"),
            ("frequency",          "frequency",        "hgb"),
            ("target(cross-fit)",  "target",           "hgb"),
            ("onehot_min0.5%",     "onehot_min_0.005", "hgb"),
        ]

        num_tr_log, num_va_log = train_df[num_cols].copy(), val_df[num_cols].copy()
        num_tr_log[log_cols] = np.log1p(num_tr_log[log_cols])
        num_va_log[log_cols] = np.log1p(num_va_log[log_cols])
        y_tr, y_va = train_df["attack_cat"].values, val_df["attack_cat"].values

        print("\n-- 4b. Running encoder comparison on validation split (22 fits)... --", flush=True)
        rows = []
        for mode in ("cat_only", "cat+num"):
            for name, method, mk in experiments:
                t0 = time.time()
                A, B = encode_categoricals(method, train_df, val_df)
                n_cat = A.shape[1]
                if mode == "cat+num":
                    tr_num = num_tr_log if mk == "logreg" else train_df[num_cols]
                    va_num = num_va_log if mk == "logreg" else val_df[num_cols]
                    A = pd.concat([A, tr_num.reset_index(drop=True)], axis=1)
                    B = pd.concat([B, va_num.reset_index(drop=True)], axis=1)

                if mk == "logreg":
                    sc_ = StandardScaler().fit(A)
                    A_, B_ = sc_.transform(A), sc_.transform(B)
                    clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=RANDOM_STATE)
                else:
                    A_, B_ = A.values, B.values
                    cat_mask = np.isin(np.arange(A.shape[1]), np.arange(n_cat)) if method == "native" else None
                    clf = HistGradientBoostingClassifier(
                        max_iter=100, learning_rate=0.1, class_weight="balanced",
                        categorical_features=cat_mask, random_state=RANDOM_STATE,
                    )

                clf.fit(A_, y_tr)
                y_pred = clf.predict(B_)
                yt, yp = y_va != "Normal", y_pred != "Normal"
                res = {
                    "mode": mode, "model": mk, "encoding": name, "n_cat_cols": n_cat,
                    "macro_f1":      f1_score(y_va, y_pred, average="macro"),
                    "attack_recall": (yt & yp).sum() / yt.sum(),
                    "false_alarm":   (~yt & yp).sum() / (~yt).sum(),
                    "seconds":       round(time.time() - t0, 1),
                }
                rows.append(res)
                print(f"  {mode:8s} | {mk:6s} | {name:20s} | cols={n_cat:3d} | "
                      f"macroF1={res['macro_f1']:.3f} recall={res['attack_recall']:.3f} "
                      f"FAR={res['false_alarm']:.3f} ({res['seconds']}s)", flush=True)

        results = pd.DataFrame(rows)
        print("\n-- 4b. Encoder comparison summary --")
        print(results.sort_values(["mode", "model", "macro_f1"], ascending=[True, True, False])
              .round(3).to_string(index=False))

        fig, axes = plt.subplots(2, 2, figsize=(18, 11))
        for i, mode in enumerate(("cat_only", "cat+num")):
            for j, mk in enumerate(("logreg", "hgb")):
                ax = axes[i, j]
                r = results[(results["mode"] == mode) & (results["model"] == mk)].sort_values("macro_f1")
                yy = np.arange(len(r))
                ax.barh(yy - 0.2, r["macro_f1"],      0.4, label="macro-F1")
                ax.barh(yy + 0.2, r["attack_recall"], 0.4, label="attack recall")
                ax.set_yticks(yy)
                ax.set_yticklabels(r["encoding"], fontsize=8)
                ax.set_xlim(0, 1.15)
                ax.set_title(f"{mode} - {mk}")
                ax.legend(loc="lower right", fontsize=8)
        save_fig(fig, "step04_encoder_comparison", f"{STEP} - encoder comparison", SID)
        save_cache("encoding_results", {"results": results, "cat_tables": cat_tables})


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Information Gain, Redundancy, Feature Engineering & Dataset Export
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_domain_features(df):
    """Build directional symmetry ratios, loss rates, handshake share, and app-layer count."""
    eps = 1e-6
    out = pd.DataFrame(index=df.index)
    out["fe_byte_log_ratio"]   = np.log1p(df["sbytes"]) - np.log1p(df["dbytes"])
    out["fe_pkt_log_ratio"]    = np.log1p(df["spkts"])  - np.log1p(df["dpkts"])
    out["fe_load_log_ratio"]   = np.log1p(df["sload"])  - np.log1p(df["dload"])
    out["fe_mean_pkt_diff"]    = np.log1p(df["smean"])  - np.log1p(df["dmean"])
    out["fe_sloss_per_pkt"]    = df["sloss"]  / (df["spkts"] + 1.0)
    out["fe_dloss_per_pkt"]    = df["dloss"]  / (df["dpkts"] + 1.0)
    out["fe_synack_rtt_share"] = df["synack"] / (df["tcprtt"] + eps)
    out["fe_jit_log_ratio"]    = np.log1p(df["sjit"])   - np.log1p(df["djit"])
    out["fe_srv_src_conc"]     = df["ct_srv_src"]       / (df["ct_src_ltm"] + 1.0)
    out["fe_dport_conc"]       = df["ct_src_dport_ltm"] / (df["ct_src_ltm"] + 1.0)
    out["fe_app_activity"]     = (
        (df["trans_depth"] > 0).astype(int)
        + (df["response_body_len"] > 0).astype(int)
        + (df["ct_flw_http_mthd"] > 0).astype(int)
        + (df["ct_ftp_cmd"] > 0).astype(int)
    )
    return out


def apply_log1p(df, num_cols, log_cols):
    """Return a copy of numeric columns with log1p applied to skewed columns."""
    X = df[num_cols].copy()
    X[log_cols] = np.log1p(X[log_cols])
    return X


def compute_information_gain(train_df, X_tr_log, num_cols):
    """Compute discrete-aware Mutual Information and conditional attack rate per feature."""
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = pd.DataFrame(ord_enc.fit_transform(train_df[CAT_COLS]), columns=CAT_COLS, index=train_df.index)
    X_all = pd.concat([X_tr_log, X_cat], axis=1)

    discrete_mask = [c in ("is_sm_ips_ports", *CODE_LIKE, *CAT_COLS) for c in X_all.columns]
    mi_scores = mutual_info_classif(
        X_all, train_df["label"].values, discrete_features=discrete_mask, random_state=RANDOM_STATE
    )

    ig_rows = []
    for c, mi in zip(X_all.columns, mi_scores):
        nz = np.ones(len(train_df), dtype=bool) if c in CAT_COLS else (train_df[c] > 0).values
        ig_rows.append({
            "feature":       c,
            "MI_binary":     round(mi, 5),
            "zero_%":        round(100.0 * (1.0 - nz.mean()), 1),
            "P(attack|x>0)": round(train_df.loc[nz, "label"].mean(), 3) if nz.sum() else np.nan,
        })
    return pd.DataFrame(ig_rows).set_index("feature").sort_values("MI_binary", ascending=False)


def compute_spearman_redundancy(train_df, num_cols, ig_df, threshold=0.90):
    """Compute Spearman correlation matrix and list all collinear pairs >= threshold."""
    spearman = train_df[num_cols].corr(method="spearman")
    pairs = [
        {
            "feature_1": num_cols[i], "MI_1": ig_df.loc[num_cols[i], "MI_binary"],
            "feature_2": num_cols[j], "MI_2": ig_df.loc[num_cols[j], "MI_binary"],
            "spearman_rho": round(spearman.iloc[i, j], 4),
        }
        for i in range(len(num_cols)) for j in range(i + 1, len(num_cols))
        if abs(spearman.iloc[i, j]) >= threshold
    ]
    pairs_df = pd.DataFrame(pairs).sort_values("spearman_rho", key=abs, ascending=False)
    return spearman, pairs_df


def add_isolation_forest_score(X_tr_log, X_va_log, X_te_log, fe_tr, fe_va, fe_te):
    """Fit IsolationForest on train and append anomaly score as a feature (no rows deleted)."""
    sc = StandardScaler().fit(X_tr_log)
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(sc.transform(X_tr_log))

    fe_tr["fe_iso_anomaly_score"] = -iso.decision_function(sc.transform(X_tr_log))
    fe_va["fe_iso_anomaly_score"] = -iso.decision_function(sc.transform(X_va_log))
    fe_te["fe_iso_anomaly_score"] = -iso.decision_function(sc.transform(X_te_log))
    return fe_tr, fe_va, fe_te


def score_engineered_ig(fe_tr, y_tr):
    """Compute Information Gain for each newly engineered feature."""
    mi_fe = mutual_info_classif(
        fe_tr, y_tr,
        discrete_features=[c == "fe_app_activity" for c in fe_tr.columns],
        random_state=RANDOM_STATE,
    )
    return pd.DataFrame({"engineered_feature": fe_tr.columns, "MI_binary": mi_fe}) \
             .sort_values("MI_binary", ascending=False)


def run_binary_ablation(train_df, val_df, X_tr_log, X_va_log, num_cols, ig_df, fe_tr, fe_va):
    """Compare 6 feature configurations on validation using LogReg and HistGradientBoosting."""
    cat_tr, cat_va = encode_categoricals("onehot_min_0.005", train_df, val_df)
    base_tr = pd.concat([X_tr_log.reset_index(drop=True), cat_tr.reset_index(drop=True)], axis=1)
    base_va = pd.concat([X_va_log.reset_index(drop=True), cat_va.reset_index(drop=True)], axis=1)

    low_mi_num     = [c for c in num_cols if ig_df.loc[c, "MI_binary"] < 0.05]
    redundant_drop = ["sloss", "dloss", "dpkts", "synack", "ackdat", "dwin"]
    ttl_cols       = ["sttl", "dttl", "ct_state_ttl"]

    configs = [
        ("1_baseline_39_cols",   base_tr, base_va),
        ("2_drop_MI<0.05",       base_tr.drop(columns=low_mi_num), base_va.drop(columns=low_mi_num)),
        ("3_drop_redundant",     base_tr.drop(columns=redundant_drop), base_va.drop(columns=redundant_drop)),
        ("4_baseline_+_FE",      pd.concat([base_tr, fe_tr.reset_index(drop=True)], axis=1),
                                 pd.concat([base_va, fe_va.reset_index(drop=True)], axis=1)),
        ("5_FE_minus_redundant", pd.concat([base_tr.drop(columns=redundant_drop), fe_tr.reset_index(drop=True)], axis=1),
                                 pd.concat([base_va.drop(columns=redundant_drop), fe_va.reset_index(drop=True)], axis=1)),
        ("6_no_TTL_+_FE",        pd.concat([base_tr.drop(columns=ttl_cols), fe_tr.reset_index(drop=True)], axis=1),
                                 pd.concat([base_va.drop(columns=ttl_cols), fe_va.reset_index(drop=True)], axis=1)),
    ]

    y_tr_bin, y_va_bin = train_df["label"].values, val_df["label"].values
    rare_mask = val_df["attack_cat"].isin(["Analysis", "Backdoor", "Worms", "Shellcode"]).values
    rows = []

    for cfg_name, A_df, B_df in configs:
        for mk in ("logreg", "hgb"):
            if mk == "logreg":
                sc_ = StandardScaler().fit(A_df)
                A_, B_ = sc_.transform(A_df), sc_.transform(B_df)
                clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            else:
                A_, B_ = A_df.values, B_df.values
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=RANDOM_STATE)

            clf.fit(A_, y_tr_bin)
            pred, prob = clf.predict(B_), clf.predict_proba(B_)[:, 1]
            row = {
                "model": mk, "config": cfg_name, "n_cols": A_df.shape[1],
                "F1":                 round(f1_score(y_va_bin, pred), 4),
                "ROC_AUC":            round(roc_auc_score(y_va_bin, prob), 4),
                "Attack_Recall":      round(((y_va_bin == 1) & (pred == 1)).sum() / (y_va_bin == 1).sum(), 4),
                "False_Alarm":        round(((y_va_bin == 0) & (pred == 1)).sum() / (y_va_bin == 0).sum(), 4),
                "Rare_Attack_Recall": round((pred[rare_mask] == 1).mean(), 4),
            }
            rows.append(row)
            print(f"  {mk:6s} | {cfg_name:22s} | cols={row['n_cols']:3d} | "
                  f"F1={row['F1']:.4f} AUC={row['ROC_AUC']:.4f} FAR={row['False_Alarm']:.4f} "
                  f"RareRec={row['Rare_Attack_Recall']:.4f}")
    return pd.DataFrame(rows)


def export_engineered_dataset(train_df, val_df, test_df, X_tr_log, X_va_log, X_te_log, fe_tr, fe_va, fe_te):
    """Save clean, log1p-transformed & engineered train/val/test CSVs to dataset/."""
    splits = [
        ("UNSW_NB15_engineered_train.csv", train_df, X_tr_log, fe_tr),
        ("UNSW_NB15_engineered_val.csv",   val_df,   X_va_log, fe_va),
        ("UNSW_NB15_engineered_test.csv",  test_df,  X_te_log, fe_te),
    ]
    print("\n-- 5e. Exporting engineered datasets to dataset/ --")
    for fname, df, X_log, fe in splits:
        out_df = pd.concat([
            X_log.reset_index(drop=True),
            fe.reset_index(drop=True),
            df[CAT_COLS + TARGETS].reset_index(drop=True),
        ], axis=1)
        out_path = os.path.join(EXPORT_DIR, fname)
        out_df.to_csv(out_path, index=False)
        print(f"  [dataset saved] dataset/{fname:32s} shape={out_df.shape}")


# ── Step 5 Plotting Helpers ───────────────────────────────────────────────────

def plot_ig_vs_sparsity(ig_df, num_cols, step_title, sid):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ig_s = ig_df.sort_values("MI_binary")
    axes[0].barh(ig_s.index, ig_s["MI_binary"], color=["tab:red" if v < 0.05 else "tab:blue" for v in ig_s["MI_binary"]])
    axes[0].axvline(0.05, color="black", ls="--", label="0.05 cutoff")
    axes[0].set_title("Global Information Gain (MI with binary label)")
    axes[0].legend()

    num_ig = ig_df.loc[num_cols]
    sc = axes[1].scatter(num_ig["zero_%"], num_ig["MI_binary"], c=num_ig["P(attack|x>0)"], cmap="coolwarm", s=60, edgecolor="k")
    axes[1].axhline(0.05, color="black", ls="--")
    axes[1].set_xlabel("Share of zeros (%)")
    axes[1].set_ylabel("Mutual Information")
    axes[1].set_title("Sparsity vs Global MI")
    fig.colorbar(sc, ax=axes[1], label="P(Attack | feature > 0)")
    save_fig(fig, "step05_ig_vs_conditional_attack", f"{step_title} - Information Gain vs sparsity", sid)


def plot_spearman_matrix(spearman, num_cols, step_title, sid):
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(spearman.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols, fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("Spearman rank correlation across 36 numeric features")
    save_fig(fig, "step05_spearman_redundancy", f"{step_title} - Spearman correlation matrix", sid)


def plot_engineered_ig(fe_ig, step_title, sid):
    fig, ax = plt.subplots(figsize=(11, 5))
    fe_plot = fe_ig.sort_values("MI_binary")
    ax.barh(fe_plot["engineered_feature"], fe_plot["MI_binary"], color="tab:green")
    ax.axvline(0.05, color="black", ls="--")
    ax.set_title("Information Gain of engineered domain features")
    save_fig(fig, "step05_engineered_ig", f"{step_title} - Engineered features IG", sid)


def plot_binary_ablation(ablation_df, step_title, sid):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    for ax, mk in zip(axes, ("logreg", "hgb")):
        sub = ablation_df[ablation_df["model"] == mk]
        yy = np.arange(len(sub))
        ax.barh(yy - 0.25, sub["F1"],                 0.25, label="Binary F1")
        ax.barh(yy,        sub["ROC_AUC"],            0.25, label="ROC-AUC")
        ax.barh(yy + 0.25, sub["Rare_Attack_Recall"], 0.25, label="Rare-Attack Recall")
        ax.set_yticks(yy)
        ax.set_yticklabels(sub["config"], fontsize=8)
        ax.set_xlim(0.70, 1.08)
        ax.set_title(f"Binary Validation Ablation ({mk})")
        ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "step05_binary_ablation_val", f"{step_title} - Binary classifier ablation", sid)


# ── Step 5 Orchestrator ───────────────────────────────────────────────────────

def step5_feature_engineering_and_ig():
    """Run IG audit, redundancy check, feature engineering, ablation, and dataset export."""
    SID, STEP = 5, "STEP 5 - Information Gain, Redundancy & Feature Engineering"
    train_df, val_df, test_df, num_cols = load_clean()
    log_cols = load_cache("transform_meta", 3)["log_cols"]

    with step_logging(SID):
        reset_step_plots(SID)
        header(STEP)

        X_tr_log = apply_log1p(train_df, num_cols, log_cols)
        X_va_log = apply_log1p(val_df,   num_cols, log_cols)
        X_te_log = apply_log1p(test_df,  num_cols, log_cols)

        # 5a. Information Gain & sparsity check
        ig_df = compute_information_gain(train_df, X_tr_log, num_cols)
        print("-- 5a. Information Gain (MI_binary) across all 39 clean features --")
        print(ig_df.to_string())
        plot_ig_vs_sparsity(ig_df, num_cols, STEP, SID)

        # 5b. Spearman rank correlation
        spearman, pairs_df = compute_spearman_redundancy(train_df, num_cols, ig_df)
        print("\n-- 5b. Redundant numeric pairs (|Spearman rho| >= 0.90) --")
        print(pairs_df.to_string(index=False))
        plot_spearman_matrix(spearman, num_cols, STEP, SID)

        # 5c. Domain features + IsolationForest anomaly score
        fe_tr = engineer_domain_features(train_df)
        fe_va = engineer_domain_features(val_df)
        fe_te = engineer_domain_features(test_df)
        fe_tr, fe_va, fe_te = add_isolation_forest_score(X_tr_log, X_va_log, X_te_log, fe_tr, fe_va, fe_te)

        fe_ig = score_engineered_ig(fe_tr, train_df["label"].values)
        print("\n-- 5c. Engineered features Information Gain --")
        print(fe_ig.round(5).to_string(index=False))
        plot_engineered_ig(fe_ig, STEP, SID)

        # 5d. Binary classification validation ablation
        print("\n-- 5d. Binary classification ablation (validation) --")
        ablation_df = run_binary_ablation(train_df, val_df, X_tr_log, X_va_log, num_cols, ig_df, fe_tr, fe_va)
        plot_binary_ablation(ablation_df, STEP, SID)

        # 5e. Export engineered train / val / test CSVs for reuse in modeling
        export_engineered_dataset(train_df, val_df, test_df, X_tr_log, X_va_log, X_te_log, fe_tr, fe_va, fe_te)

        save_cache("step5_fe_ig", {
            "ig_df": ig_df, "pairs_df": pairs_df, "fe_ig": fe_ig, "ablation_df": ablation_df,
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  Step Registry & CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

STEPS = {
    "1": step1_audit,
    "2": step2_clean_split,
    "3": step3_distributions,
    "4": step4_categorical_encoding,
    "5": step5_feature_engineering_and_ig,
}

RUN_STEPS = ["1","2","3","4","5"]

if __name__ == "__main__":
    args = sys.argv[1:] or RUN_STEPS
    if args == ["clean"]:
        clean_workspace()
        sys.exit(0)
    if args == ["all"]:
        args = list(STEPS)
    for a in args:
        if a == "report":
            continue
        if a not in STEPS:
            print(f"Unknown step '{a}'. Available: {list(STEPS)} + ['all', 'report', 'clean']")
            sys.exit(1)
        STEPS[a]()
    build_report()
