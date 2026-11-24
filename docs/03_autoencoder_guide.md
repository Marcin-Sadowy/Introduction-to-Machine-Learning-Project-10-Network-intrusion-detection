# Autoencoder Guide (`models/autoencoder_keras.py` & `models/autoencoder_numpy_demo.py`)

Reference notes connecting linear/logistic regression and forests to deep autoencoders, covering the math, loss functions, hyperparameters, and terminal output.
- **Primary model:** `models/autoencoder_keras.py` (Keras: `ReLU`, `BatchNormalization`, `Dropout`, `GaussianNoise`, `L2`, `EarlyStopping`, `ReduceLROnPlateau`).
- **Secondary math showcase:** `models/autoencoder_numpy_demo.py` (pure NumPy forward pass, `tanh` backprop, and Adam written out explicitly).

---

## 1. From Linear / Logistic Regression to Deep Autoencoders

A dense layer stacks $h$ linear/logistic units in parallel:

$$a = \phi(X W + b)$$

- $X \in \mathbb{R}^{n \times d}$ — input mini-batch ($n$ rows/flows, $d$ features)
- $W \in \mathbb{R}^{d \times h}$ — weight matrix ($h$ neurons)
- $b \in \mathbb{R}^{1 \times h}$ — bias vector
- $\phi(\cdot)$ — element-wise non-linear activation
- $a \in \mathbb{R}^{n \times h}$ — layer activations

**Why $\phi$ must be non-linear:** Without $\phi$, stacked layers collapse into one linear transform:

$$(X W_0 + b_0) W_1 + b_1 = X (W_0 W_1) + (b_0 W_1 + b_1) = X W' + b'$$

- $W_0, b_0, W_1, b_1$ — consecutive linear layer parameters; $W', b'$ — collapsed single-layer equivalents.

A purely linear autoencoder just recovers **PCA**. Adding $\phi$ lets the network learn curved, non-linear feature interactions.

**How an Autoencoder differs from Logistic Regression / Forests:**
Supervised models train on both `0` and `1` labels. Our autoencoder trains **only on Normal traffic (`label == 0`)** to compress $X$ into a bottleneck $z \in \mathbb{R}^{n \times k}$ ($k \ll d$) and reconstruct $\hat{X} \in \mathbb{R}^{n \times d}$:

$$\text{Input } (d) \xrightarrow{\text{Encoder}} z \ (k) \xrightarrow{\text{Decoder}} \hat{X} \ (d)$$

Because $k \ll d$, it learns the correlations of **Normal** traffic. At inference time, we score each flow by its reconstruction error $e(x)$ against a threshold $\tau$ tuned on Validation:
- $e(x) < \tau \implies$ **Normal (`0`)**
- $e(x) \ge \tau \implies$ **Attack (`1`)**

---

## 2. Activations & Regularization

- **Why `tanh` in `autoencoder.py` (NumPy):**
  $$\tanh(u) = \frac{e^u - e^{-u}}{e^u + e^{-u}} = 2\sigma(2u) - 1 \in (-1, 1), \qquad \frac{d}{du}\tanh(u) = 1 - a^2$$
  - $u$ — pre-activation ($XW + b$); $e \approx 2.71828$; $\sigma(u) = \frac{1}{1+e^{-u}}$ — logistic sigmoid.
  - Zero-centered (unlike `sigmoid`), avoids dead neurons (unlike raw `ReLU` without BatchNorm), and reuses $a$ directly in backprop (`1 - a**2`).
- **Why `ReLU` + `BatchNormalization` in `autoencoder_keras.py`:**
  - $\text{ReLU}(u) = \max(0, u)$ trains faster and avoids vanishing gradients in deeper nets; `BatchNormalization` keeps activations zero-mean/unit-variance so ReLUs don't die.
- **Why the output layer is always linear:** `StandardScaler` z-scores are unbounded real numbers; a bounded activation like `tanh` or `sigmoid` would clip valid reconstructions.
- **Regularizers in Keras:**
  - `GaussianNoise` ($\sigma = 0.03$): adds slight noise to training inputs while reconstructing clean flows (Denoising Autoencoder).
  - `Dropout` (`0.10`): randomly zeroes 10% of encoder neurons during training.
  - `L2` (`1e-5`): adds $\lambda \sum W^2$ to the loss to discourage large weights.

---

## 3. Reconstruction Error & `LOSS_FN` (`mse`, `mae`, `huber`)

Let $r_j = x_j - \hat{x}_j$ be the reconstruction residual on feature $j \in \{1, \dots, d\}$. `LOSS_FN` governs both **training** (`model.compile`) and **anomaly scoring** (`recon_error`):

1. **`"mse"` — Mean Squared Error (recommended here)**
   $$e_{\text{MSE}}(x) = \frac{1}{d} \sum_{j=1}^{d} r_j^2$$
   Squares errors so large deviations on 2–3 features (`fe_byte_log_ratio`, `sttl`, `rate`) stand out above background noise. In our validation benchmark at `latent=10`, MSE reached `Val_AUC = 0.8583` vs `0.8089` for MAE (and `0.8927` at `latent=16`).

2. **`"mae"` — Mean Absolute Error**
   $$e_{\text{MAE}}(x) = \frac{1}{d} \sum_{j=1}^{d} |r_j|$$
   Linear penalty. Robust to noisy training outliers, but ~52 normal features dilute 2–3 anomalous attack features at test time (higher false alarms: `37.2%` vs `30.0%`).

3. **`"huber"` — Huber Loss (MSE + MAE hybrid)**
   $$L_\delta(r_j) = \begin{cases} \frac{1}{2} r_j^2 & \text{if } |r_j| \le \delta \\ \delta\left(|r_j| - \frac{1}{2}\delta\right) & \text{if } |r_j| > \delta \end{cases}, \qquad e_{\text{Huber}}(x) = \frac{1}{d} \sum_{j=1}^{d} L_\delta(r_j)$$
   - $\delta$ — transition point between quadratic and linear penalty (Keras default $\delta = 1.0$).

---

## 4. Backpropagation, Initialization & Adam (`autoencoder.py`)

For a mini-batch of $n$ rows and MSE loss $\mathcal{L} = \frac{1}{nd}\sum_{i=1}^{n}\sum_{j=1}^{d}(\hat{x}_{ij} - x_{ij})^2$ ($i$ = row index):

$$\delta^{(3)} = \frac{\partial \mathcal{L}}{\partial \hat{X}} = \frac{2}{nd}(\hat{X} - X), \qquad \nabla_{W_\ell}\mathcal{L} = \left(a^{(\ell)}\right)^\top \delta^{(\ell)}, \qquad \nabla_{b_\ell}\mathcal{L} = \sum_{i=1}^{n}\delta_i^{(\ell)}$$

$$\delta^{(\ell-1)} = \left(\delta^{(\ell)} W_\ell^\top\right) \odot \left(1 - \left(a^{(\ell)}\right)^2\right)$$

- $\ell \in \{0,1,2,3\}$ — layer index; $\delta^{(\ell)}$ — loss gradient w.r.t. pre-activation of layer $\ell$; $\nabla_{W_\ell}\mathcal{L}, \nabla_{b_\ell}\mathcal{L}$ — parameter gradients; $a^{(\ell)}$ — activation entering layer $\ell$ ($a^{(0)}=X$); $(\cdot)^\top$ — transpose; $\odot$ — element-wise product.

- **Xavier init:** $W_\ell \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$ ($n_{\text{in}}, n_{\text{out}}$ = layer input/output widths) keeps initial activations out of saturation.
- **Adam:** tracks moving averages of the gradient $m_t$ and squared gradient $v_t$ at step $t$ (with decay rates $\beta_1=0.9$, $\beta_2=0.999$, learning rate $\eta=10^{-3}$, stability constant $\epsilon=10^{-8}$):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$W \leftarrow W - \eta \frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon}$$
  - $g_t$ — gradient at mini-batch step $t$.

---

## 5. Hyperparameters to Tune (Top of `autoencoder_keras.py`)

Always tune using **Validation (`Val_AUC` / `Val_F1`)**, and evaluate on **Official Test** only to verify generalization.

| Parameter | Default | What to Try & Effect |
| :--- | :---: | :--- |
| **`LATENT_DIM`** | `16` | `[8, 12, 16, 20, 24]` — **highest impact**. Too small underfits normal variety (high FAR); too large copies attacks (low AUC). |
| **`ENCODER_DIMS`** | `[64, 32]` | `[32]`, `[64, 32]`, `[128, 64]` — non-linear capacity into/out of bottleneck. |
| **`LOSS_FN`** | `"mse"` | `"mse"` (best here), `"huber"`, `"mae"`. |
| **`EPOCHS` / `PATIENCE`** | `60` / `8` | Raise `EPOCHS` (`100`–`120`) if `Best_Epoch` hits the cap; `EarlyStopping` + `ReduceLROnPlateau` handle convergence automatically. |
| **`BATCH_SIZE` / `LR`** | `256` / `1e-3` | `BATCH_SIZE` `[128, 256, 512]`, `LR` `[3e-4, 1e-3]`. Lower `LR` if the loss curve bounces. |
| **Threshold $\tau$** | Max Val $F_1$ | Picked on Validation. Can alternatively fix a target FAR (e.g., 95th percentile of Val Normal error) or maximize $F_2$ for higher recall. |

---

## 6. Metrics, Terminal Output & Plots

$$P = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad R = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FAR} = \frac{\text{FP}}{\text{FP} + \text{TN}}, \quad F_1 = \frac{2PR}{P + R}$$

- $\text{TP}, \text{FP}, \text{TN}, \text{FN}$ — True/False Positives and True/False Negatives.

| Printed Column | Meaning |
| :--- | :--- |
| **`Features` / `Layers` / `Params`** | Input dimension $d$, `Encoder->Bottleneck->Decoder` widths, total model parameters. |
| **`Best_Epoch` / `Best_ValLoss`** | Best epoch restored by `EarlyStopping` and its lowest Normal validation loss. |
| **`Threshold`** | Error cutoff $\tau$ picked on Validation ($e(x) \ge \tau \implies$ Attack). |
| **`ROC_AUC`** | Threshold-free Normal vs Attack separation (`0.50` = random, `1.00` = perfect). |
| **`F1` / `Precision` / `Recall` / `FAR`** | Binary $F_1$, alert precision ($P$), attack detection rate ($R$), normal false alarm rate ($\text{FAR}$). |
| **`Val_n` / `Test_n` / `*_Flagged%`** | Row count and % predicted as Attack per `attack_cat` (`FAR` for `Normal`, `Recall` for attack classes). |

**Plots (`plots/`):**
- `ae_keras_training_curve.png` — Train vs Val Normal loss per epoch (checks convergence / overfitting).
- `ae_keras_error_and_eval.png` — Left: $\log_{10}(e(x))$ Normal vs Attack histogram + $\tau$; Right: Test detection rate per `attack_cat`.
- `ae_keras_confusion_matrix.png` — Side-by-side Validation and Official Test confusion matrices (counts & row %).
