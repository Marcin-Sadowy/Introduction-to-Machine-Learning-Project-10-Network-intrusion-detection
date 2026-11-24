"""
models/autoencoder_keras.py — Primary Keras Autoencoder (UNSW-NB15)
===================================================================

Run:    python models/autoencoder_keras.py
Input:  dataset/autoencoder_ready/
Output: outputs/ae_keras_model.keras
        plots/ae_keras_training_curve.png
        plots/ae_keras_error_and_eval.png
        plots/ae_keras_confusion_matrix.png
"""

import os
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["ABSL_MIN_LOG_LEVEL"]    = "3"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.get_logger().setLevel("ERROR")

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AE_DATA_DIR = os.path.join(ROOT, "dataset", "autoencoder_ready")
PLOT_DIR    = os.path.join(ROOT, "plots")
OUT_DIR     = os.path.join(ROOT, "outputs")
SEED        = 42

# ── Model & training parameters (edit here) ───────────────────────────────────
ENCODER_DIMS = [64, 32]   # encoder widths (decoder mirrors this)
LATENT_DIM   = 16         # bottleneck width
LOSS_FN      = "mse"      # "mse", "mae", or "huber"
BATCH_SIZE   = 256
EPOCHS       = 60
PATIENCE     = 8
LR           = 1e-3
DROPOUT      = 0.10
NOISE        = 0.03
L2_REG       = 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
#  Model & Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def build_autoencoder(input_dim):
    """Symmetric denoising autoencoder: Input -> Encoder -> Bottleneck -> Decoder -> Output."""
    reg = regularizers.l2(L2_REG)
    inp = layers.Input(shape=(input_dim,))
    x = layers.GaussianNoise(NOISE)(inp) if NOISE > 0 else inp

    # Encoder
    for units in ENCODER_DIMS:
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        if DROPOUT > 0:
            x = layers.Dropout(DROPOUT)(x)

    # Bottleneck
    x = layers.Dense(LATENT_DIM, activation="relu", kernel_regularizer=reg)(x)

    # Decoder
    for units in reversed(ENCODER_DIMS):
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)

    # Linear output matches unbounded StandardScaler z-scores
    out = layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss=LOSS_FN)
    return model


def recon_error(model, X):
    """Per-row reconstruction error matching LOSS_FN."""
    diff = X - model.predict(X, batch_size=1024, verbose=0)
    if LOSS_FN == "mae":
        return np.mean(np.abs(diff), axis=1)
    if LOSS_FN == "huber":
        abs_d = np.abs(diff)
        huber = np.where(abs_d <= 1.0, 0.5 * diff ** 2, abs_d - 0.5)
        return np.mean(huber, axis=1)
    return np.mean(diff ** 2, axis=1)


def best_threshold(err, y_true):
    """Pick the error cutoff on Validation that maximizes Binary F1."""
    candidates = np.quantile(err, np.linspace(0.10, 0.99, 250))
    f1_scores = [f1_score(y_true, (err >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(f1_scores))])


def split_metrics(name, err, y_df, threshold):
    """Compute summary metrics, per-category flagged rates, and confusion matrix."""
    y_true = y_df["label"].values
    y_pred = (err >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    row = {
        "Split":     name,
        "ROC_AUC":   round(roc_auc_score(y_true, err), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.1%}",
        "Recall":    f"{tp / (tp + fn):.1%}",
        "FAR":       f"{fp / (fp + tn):.1%}",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    cats = {
        cat: (len(grp), float((y_pred[grp.index] == 1).mean()))
        for cat, grp in y_df.groupby("attack_cat")
    }
    return row, cats, cm


# ═══════════════════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curve(hist):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(hist["loss"]) + 1)
    ax.plot(epochs, hist["loss"],     label="Train Normal", lw=2, color="#2b6cb0")
    ax.plot(epochs, hist["val_loss"], label="Val Normal",   lw=2, color="#dd6b20", ls="--")
    ax.set(xlabel="Epoch", ylabel=f"Loss ({LOSS_FN.upper()})", title="Keras Autoencoder Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ae_keras_training_curve.png"), dpi=120)
    plt.close(fig)


def plot_error_and_categories(test_err, y_test_df, th, test_cats):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    norm_e = np.log10(test_err[y_test_df["label"] == 0] + 1e-8)
    att_e  = np.log10(test_err[y_test_df["label"] == 1] + 1e-8)
    th_log = np.log10(th + 1e-8)

    axes[0].hist(norm_e, bins=70, alpha=0.65, density=True, label="Normal", color="#2b6cb0")
    axes[0].hist(att_e,  bins=70, alpha=0.65, density=True, label="Attack", color="#e53e3e")
    axes[0].axvline(th_log, color="black", ls="--", lw=2, label=f"Threshold ({th:.4f})")
    axes[0].set(xlabel="log10(Reconstruction Error)", ylabel="Density", title="Test Error Separation")
    axes[0].legend()

    cats = sorted(test_cats, key=lambda k: test_cats[k][1])
    vals = [test_cats[k][1] for k in cats]
    colors = ["#718096" if k == "Normal" else "#dd6b20" for k in cats]
    axes[1].barh(cats, vals, color=colors)
    axes[1].set(xlim=(0, 1.12), xlabel="Flagged as Attack", title="Test Detection Rate per Category")
    for i, v in enumerate(vals):
        axes[1].text(v + 0.015, i, f"{v:.1%}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ae_keras_error_and_eval.png"), dpi=120)
    plt.close(fig)


def plot_confusion_matrices(val_cm, test_cm):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tags = [["TN", "FP"], ["FN", "TP"]]
    labels = ["Normal (0)", "Attack (1)"]

    for ax, cm, title in [(axes[0], val_cm, "Validation"), (axes[1], test_cm, "Official Test")]:
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=labels, yticklabels=labels,
            xlabel="Predicted", ylabel="Actual", title=title,
        )
        for i in range(2):
            for j in range(2):
                color = "white" if pct[i, j] > 0.55 else "black"
                ax.text(
                    j, i, f"{tags[i][j]}\n{cm[i, j]:,}\n({pct[i, j]:.1%})",
                    ha="center", va="center", color=color, fontweight="bold",
                )

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ae_keras_confusion_matrix.png"), dpi=120)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load pre-scaled splits (train contains Normal traffic only)
    X_tr_norm  = pd.read_csv(os.path.join(AE_DATA_DIR, "X_train_normal.csv")).values.astype(np.float32)
    X_val      = pd.read_csv(os.path.join(AE_DATA_DIR, "X_val.csv")).values.astype(np.float32)
    y_val_df   = pd.read_csv(os.path.join(AE_DATA_DIR, "y_val.csv"))
    X_test     = pd.read_csv(os.path.join(AE_DATA_DIR, "X_test.csv")).values.astype(np.float32)
    y_test_df  = pd.read_csv(os.path.join(AE_DATA_DIR, "y_test.csv"))
    X_val_norm = X_val[y_val_df["label"].values == 0]

    # Build & train on Normal flows only
    model = build_autoencoder(X_tr_norm.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE // 2, min_lr=1e-5),
    ]

    print("Training...", end=" ", flush=True)
    hist = model.fit(
        X_tr_norm, X_tr_norm,
        validation_data=(X_val_norm, X_val_norm),
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, callbacks=callbacks, verbose=0,
    ).history
    print("done.\n")

    # Run summary table
    best_ep = int(np.argmin(hist["val_loss"])) + 1
    summary = pd.DataFrame([{
        "Features":     X_tr_norm.shape[1],
        "Layers":       f"{ENCODER_DIMS}->{LATENT_DIM}->{list(reversed(ENCODER_DIMS))}",
        "Params":       f"{model.count_params():,}",
        "Loss":         LOSS_FN.upper(),
        "Batch":        BATCH_SIZE,
        "Best_Epoch":   f"{best_ep}/{len(hist['loss'])}",
        "Best_ValLoss": round(min(hist["val_loss"]), 5),
    }])
    print("── Run Summary ──")
    print(summary.to_string(index=False))

    # Threshold tuned on Validation only, then applied to both Validation & Test
    val_err  = recon_error(model, X_val)
    test_err = recon_error(model, X_test)
    th       = best_threshold(val_err, y_val_df["label"].values)

    val_row,  val_cats,  val_cm  = split_metrics("Validation",    val_err,  y_val_df,  th)
    test_row, test_cats, test_cm = split_metrics("Official Test", test_err, y_test_df, th)

    print(f"\n── Binary Performance (Threshold = {th:.5f}) ──")
    print(pd.DataFrame([val_row, test_row]).set_index("Split").to_string())

    cat_rows = [
        {
            "attack_cat":    c,
            "Val_n":         val_cats[c][0],
            "Val_Flagged%":  f"{val_cats[c][1]:.1%}",
            "Test_n":        test_cats[c][0],
            "Test_Flagged%": f"{test_cats[c][1]:.1%}",
            "Metric_Role":   "FAR (lower=better)" if c == "Normal" else "Recall (higher=better)",
        }
        for c in sorted(test_cats, key=lambda k: (k == "Normal", -test_cats[k][1]))
    ]
    print("\n── Detection Rate by attack_cat (% Flagged as Attack) ──")
    print(pd.DataFrame(cat_rows).to_string(index=False))

    # Save model & plots
    model.save(os.path.join(OUT_DIR, "ae_keras_model.keras"))
    plot_learning_curve(hist)
    plot_error_and_categories(test_err, y_test_df, th, test_cats)
    plot_confusion_matrices(val_cm, test_cm)


if __name__ == "__main__":
    main()
