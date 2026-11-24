# Autoencoder Overview, Deep Layers & Math (`autoencoder.py`)

Notes connecting linear/logistic regression and tree ensembles to deep neural layers, how our pure-NumPy Autoencoder works, and how to read the logs and plots.

---

## 1. From Linear / Logistic Regression to Deep Layers

If you already know linear and logistic regression, a deep neural network is just stacking them:

- **Linear regression** computes one weighted sum: $\hat{y} = x^\top w + b$.
- **Logistic regression** wraps that sum in a non-linear activation (sigmoid): $\hat{y} = \sigma(x^\top w + b)$.
- **One dense (fully connected) layer** runs $h$ of those units in parallel by replacing the vector $w$ with a weight matrix $W$:

$$a = \phi(X W + b)$$

where:
- $X \in \mathbb{R}^{n \times d}$ — input mini-batch ($n$ network flows/rows, $d$ features)
- $W \in \mathbb{R}^{d \times h}$ — weight matrix ($h$ neurons in the layer, one column per neuron)
- $b \in \mathbb{R}^{1 \times h}$ — bias vector (added to every row)
- $\phi(\cdot)$ — element-wise non-linear activation function
- $a \in \mathbb{R}^{n \times h}$ — layer output (activations)

### Why we need a non-linear activation $\phi$
If you stack two linear layers without a non-linearity in between, matrix multiplication collapses them back into a single linear model:

$$(X W_0 + b_0) W_1 + b_1 = X (W_0 W_1) + (b_0 W_1 + b_1) = X W' + b'$$

where:
- $W_0, b_0$ and $W_1, b_1$ — weights and biases of two consecutive linear layers
- $W' = W_0 W_1$ and $b' = b_0 W_1 + b_1$ — single equivalent weight matrix and bias

In an autoencoder, a purely linear network with bottleneck size $k$ just recovers the same subspace as **PCA** with $k$ components. Putting a non-linear $\phi$ between layers lets the network bend the feature space and learn non-linear feature interactions (similar to how trees split on combinations of features, except smooth and continuous).

### How an Autoencoder differs from Logistic Regression & Forests
- **Logistic regression / Random Forest / HistGradientBoosting** are **supervised**: they take both `Normal` (`0`) and `Attack` (`1`) labels and learn a decision boundary between them.
- **Our Autoencoder** is **semi-supervised (anomaly detection)**: it trains **only on Normal traffic (`label == 0`)** to reconstruct its own input through a narrow bottleneck:

$$a_1 = \tanh(X W_0 + b_0), \quad z = \tanh(a_1 W_1 + b_1), \quad a_3 = \tanh(z W_2 + b_2), \quad \hat{X} = a_3 W_3 + b_3$$

where:
- $W_0, \dots, W_3$ and $b_0, \dots, b_3$ — weight matrices and biases of the 4 layers (`55 -> 32 -> 12 -> 32 -> 55`)
- $a_1, a_3 \in \mathbb{R}^{n \times 32}$ — encoder and decoder hidden activations
- $z \in \mathbb{R}^{n \times 12}$ — bottleneck (latent) representation
- $\hat{X} \in \mathbb{R}^{n \times d}$ — reconstructed output matrix

Because $12 \ll d$, the network cannot memorize the identity function — it has to learn the correlations that hold across **Normal** flows. At test time, we measure per-row reconstruction error (MSE):

$$e(x) = \frac{1}{d} \sum_{j=1}^{d} (x_j - \hat{x}_j)^2$$

where:
- $e(x)$ — reconstruction error (MSE) for a single flow $x$
- $j$ — feature index ($1, \dots, d$)
- $x_j, \hat{x}_j$ — original and reconstructed value of feature $j$

We then classify using a threshold $\tau$ (cutoff value tuned on the validation set):
- $e(x) < \tau \implies$ **Normal (`0`)**
- $e(x) \ge \tau \implies$ **Attack (`1`)**

---

## 2. Why `tanh` (and Why the Output Layer is Linear)

The hyperbolic tangent is defined as:

$$\tanh(u) = \frac{e^u - e^{-u}}{e^u + e^{-u}} \in (-1, 1), \qquad \tanh(u) = 2\sigma(2u) - 1$$

where:
- $u$ — pre-activation value (linear sum $X W + b$ before applying $\phi$)
- $e$ — Euler's number ($\approx 2.71828$)
- $\sigma(u) = \frac{1}{1 + e^{-u}} \in (0, 1)$ — logistic sigmoid function

We chose `tanh` for the hidden layers in `autoencoder.py` for three practical reasons:

1. **Zero-centered outputs (vs. `sigmoid`):** `sigmoid` outputs are always positive $(0, 1)$, which forces weight gradients in the next layer to all have the same sign and slows convergence. Because our `StandardScaler` inputs are centered at $0$, `tanh` keeps hidden activations centered around $0$ too.
2. **No dead neurons (vs. `ReLU`):** $\text{ReLU}(u) = \max(0, u)$ has derivative $0$ for all $u < 0$. In a small hand-coded NumPy network without BatchNorm, a large step can push neurons permanently into $u < 0$ ("dying ReLU"). `tanh` stays smooth and active around $0$.
3. **Cheap, clean derivative:**

$$\frac{d}{du}\tanh(u) = 1 - \tanh^2(u) = 1 - a^2$$

where:
- $\frac{d}{du}$ — derivative with respect to the pre-activation $u$

Since we already store the forward activation $a = \tanh(u)$, the derivative during backprop is just `1 - a**2`.

**Why the final layer (`W3, b3`) has NO activation (linear):**
Our scaled numeric features are z-scores (unbounded real numbers, e.g., $-2.5$ or $+4.8$). Putting `tanh` on the output layer would trap $\hat{x}$ inside $(-1, 1)$ and make it impossible to reconstruct normal values outside that range.

---

## 3. Backpropagation & Optimization Math (Inside `train_step`)

For a mini-batch $X \in \mathbb{R}^{n \times d}$ ($n = 256$ rows), the batch MSE loss is:

$$\mathcal{L} = \frac{1}{nd} \sum_{i=1}^{n} \sum_{j=1}^{d} (\hat{x}_{ij} - x_{ij})^2$$

where:
- $\mathcal{L}$ — scalar batch loss (average MSE across all $n$ rows and $d$ features)
- $i$ — sample (row) index within the mini-batch ($1, \dots, n$)
- $x_{ij}, \hat{x}_{ij}$ — original and reconstructed value of feature $j$ in sample $i$

### Backpropagation (Chain Rule)
1. **Start at the output layer ($\ell = 3$):**

$$\delta^{(3)} = \frac{\partial \mathcal{L}}{\partial \hat{X}} = \frac{2}{nd}(\hat{X} - X)$$

where:
- $\ell \in \{0, 1, 2, 3\}$ — layer index
- $\delta^{(\ell)}$ — error gradient of $\mathcal{L}$ with respect to the pre-activation of layer $\ell$
- $\frac{\partial \mathcal{L}}{\partial \hat{X}}$ — partial derivative of $\mathcal{L}$ with respect to every element of $\hat{X}$

2. **Parameter gradients at each layer $\ell \in \{3, 2, 1, 0\}$:**

$$\nabla_{W_\ell} \mathcal{L} = \left(a^{(\ell)}\right)^\top \delta^{(\ell)}, \qquad \nabla_{b_\ell} \mathcal{L} = \sum_{i=1}^{n} \delta^{(\ell)}_i$$

where:
- $\nabla_{W_\ell} \mathcal{L}, \nabla_{b_\ell} \mathcal{L}$ — gradients of $\mathcal{L}$ with respect to $W_\ell$ and $b_\ell$
- $a^{(\ell)}$ — activation matrix entering layer $\ell$ (with $a^{(0)} = X$)
- $(\cdot)^\top$ — matrix transpose

We clip gradients to $[-5, 5]$ so an extreme packet flow cannot cause exploding updates.

3. **Pass the error backward through `tanh` to layer $\ell - 1$:**

$$\delta^{(\ell - 1)} = \left(\delta^{(\ell)} W_\ell^\top\right) \odot \left(1 - \left(a^{(\ell)}\right)^2\right)$$

where:
- $\odot$ — element-wise (Hadamard) multiplication

### Initialization & Adam Optimizer
- **Xavier / Glorot init:**

$$W_\ell \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

where:
- $\sim \mathcal{N}(\mu, s^2)$ — sampled from a Gaussian distribution with mean $\mu$ and variance $s^2$
- $n_{\text{in}}, n_{\text{out}}$ — number of input and output neurons of layer $\ell$

This keeps the variance of $X W$ near $1$ so `tanh` starts in its high-gradient middle region instead of saturating at $\pm 1$.

- **Adam (Adaptive Moment Estimation):** Plain gradient descent uses one fixed step size for every weight. Adam tracks a running average of the gradient (momentum) and squared gradient (per-weight scale):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad W \leftarrow W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where:
- $t$ — update step counter (incremented once per mini-batch)
- $g_t$ — gradient at step $t$ ($\nabla_W \mathcal{L}$ or $\nabla_b \mathcal{L}$)
- $m_t, v_t$ — moving averages of the first moment (mean gradient) and second moment (uncentered variance)
- $\beta_1, \beta_2$ — exponential decay rates (`0.9` and `0.999`)
- $\hat{m}_t, \hat{v}_t$ — bias-corrected estimates of $m_t$ and $v_t$ (compensates for starting at $0$)
- $\eta$ — learning rate (`1e-3`)
- $\epsilon$ — small stability constant (`1e-8`) to prevent division by zero

- **Early stopping (`patience = 6`):** After each epoch (one full pass over all `41,368` Normal training rows), we check MSE on unseen Normal validation rows. If it doesn't improve for 6 straight epochs, training stops and restores the best weights.

---

## 4. Evaluation Metrics & Plots

- **Threshold ($\tau$):** Chosen by sweeping candidate quantiles of $e(x)$ on **Validation** (`X_val`) to maximize Binary $F_1$. The test set is never used to tune $\tau$.
- **ROC-AUC:** Area under the ROC curve using raw $e(x)$ as the anomaly score (`0.5` = random, `1.0` = perfect separation across all thresholds).
- **Classification Metrics:**

$$P = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad R = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FAR} = \frac{\text{FP}}{\text{FP} + \text{TN}}, \quad F_1 = \frac{2PR}{P + R}$$

where:
- $\text{TP}$ — True Positives (Attacks correctly caught)
- $\text{FP}$ — False Positives (Normal flows wrongly flagged as Attack)
- $\text{TN}$ — True Negatives (Normal flows correctly passed as Normal)
- $\text{FN}$ — False Negatives (Attacks missed as Normal)
- $P$ — Precision (share of alerts that are real attacks)
- $R$ — Attack Recall / True Positive Rate (share of real attacks detected)
- $\text{FAR}$ — False Alarm Rate / False Positive Rate (share of normal flows wrongly flagged)
- $F_1$ — Harmonic mean of Precision and Recall

- **`plots/ae_training_curve.png`:** `train_mse` vs `val_normal_mse` across epochs. Both should drop and plateau together; a widening gap means overfitting to training Normal rows.
- **`plots/ae_error_and_eval.png`:**
  - *Left:* Distribution of $\log_{10}(e(x))$ on the official test set for Normal (blue) vs Attack (red), with the validation threshold $\tau$ as a dashed line.
  - *Right:* Share flagged as Attack per `attack_cat` (orange = recall per attack family; grey `Normal` bar = false alarm rate).
- **`plots/ae_confusion_matrix.png`:** Side-by-side Validation and Official Test confusion matrices showing `TN`, `FP`, `FN`, `TP` counts and row percentages.

---

## 5. Hyperparameters to Tune (`autoencoder.py`)

All main hyperparameters are defined at the top of `autoencoder.py`. Always select configurations using **Validation ROC-AUC / Validation $F_1$** and evaluate on Official Test only once at the end.

1. **`LATENT_DIM` (bottleneck size — highest impact)**
   - *Default:* `12` | *Try:* `[4, 6, 8, 12, 16, 24]`
   - Too large $\implies$ network copies attacks too easily $\implies$ ROC-AUC drops.
   - Too small $\implies$ cannot capture normal traffic variety $\implies$ False Alarm Rate rises.
   - Typical sweet spot: ~10–25% of input dimension (`6` to `14` for ~55 features).

2. **`HIDDEN_DIM` (encoder/decoder width)**
   - *Default:* `32` | *Try:* `[24, 32, 48, 64]`
   - Controls non-linear capacity into and out of the bottleneck. Keep it larger than `LATENT_DIM`.

3. **`LR` (learning rate) & `BATCH_SIZE`**
   - *Default:* `LR = 1e-3`, `BATCH_SIZE = 256`
   - *Try:* `LR` in `[3e-4, 1e-3, 3e-3]`, `BATCH_SIZE` in `[128, 256, 512]`
   - Check `plots/ae_training_curve.png`: noisy/bouncing loss $\implies$ lower `LR`; loss still dropping steeply at the end $\implies$ raise `EPOCHS` or `LR`.

4. **`EPOCHS` & `PATIENCE`**
   - *Default:* `EPOCHS = 40`, `PATIENCE = 6`
   - *Try:* `EPOCHS = 80`–`100`, `PATIENCE = 8`–`10` and let early stopping pick the best epoch automatically.

5. **Decision Threshold $\tau$ (Recall vs. False Alarm trade-off)**
   - *Default:* maximizes Binary $F_1$ on validation (`pick_best_threshold()`).
   - Alternatives depending on priority: fix a target False Alarm Rate (e.g., 95th percentile of validation Normal errors $\implies$ ~5% FAR) or maximize $F_2$-score to weight Attack Recall 2× higher than Precision.

6. **Optional Structural Tweaks**
   - **MAE / Huber loss** instead of MSE: less sensitive to single-feature spikes.
   - **Denoising Autoencoder:** add small Gaussian noise (`std ≈ 0.05`) to training inputs while still reconstructing clean inputs to reduce overfitting.

### Suggested Search Order
1. Sweep `LATENT_DIM` (`[4, 6, 8, 12, 16]`) and `HIDDEN_DIM` (`[32, 64]`) with `LR=1e-3, BATCH_SIZE=256, EPOCHS=80, PATIENCE=8`, ranking by **Validation ROC-AUC**.
2. Lock in the best architecture, then tune $\tau$ on Validation for the desired Recall / FAR balance.
3. Run the single winning configuration on Official Test.
