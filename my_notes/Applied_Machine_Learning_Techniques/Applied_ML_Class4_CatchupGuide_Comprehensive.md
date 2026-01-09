# 🧭 Applied Machine Learning — Class 4 **Catch‑Up Guide** (Beginner‑Friendly, Comprehensive)

**Use this if you missed the lecture.**  
You’ll get plain‑English explanations, the key math (with symbols explained), step‑by‑step procedures, pitfalls to avoid, and timestamps so you can jump to the exact parts of the class video.

---

## 💡 How to Use This Guide
- Skim the **TL;DR** in each section first, then read the details.
- Use the **checklists** and **recipes** to practice.
- When you see a timestamp like `01:25:00`, jump to that time in the class recording for the live explanation/demo.

---

## ⏳ Topic Map with Timestamps
- **Intro & Regression recap** — `00:00–00:25`
- **Regression assumptions** — `00:25–00:45`
- **Overfitting & multicollinearity** — `00:45–01:05`
- **Regularization overview** — `01:05–01:25`
- **Ridge (L2) regression** — `01:25–01:40`
- **Lasso (L1) regression** — `01:40–01:50`
- **Elastic Net** — `01:50–02:00`
- **Bias–variance trade‑off** — `02:00–02:10`
- **Gradient Descent (how models learn)** — `02:10–02:30`
- **Q&A highlights / misconceptions** — `02:30–02:45`
- **Wrap‑up & next steps** — `02:45–03:00`

> ⏱️ Timestamps are approximate chapter markers matching the pacing in the lecture.

---

## 1) Regression Foundations (`00:00–00:25`)

### TL;DR
Regression predicts a **continuous** value (Y) from one or more inputs (X). We fit a straight line (or hyperplane) that minimizes the squared distance between **predicted** and **actual** values.

### What problem does it solve?
- Estimating a numeric outcome: price, temperature, salary, network bandwidth, etc.
- Understanding how each feature moves the prediction up or down.

### The model
**Scalar (one feature):**  
\[
Y = \beta_0 + \beta_1 X + \varepsilon
\]

**Vector / Matrix form (many features):**  
\[
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
\]

- \(\mathbf{y}\in\mathbb{R}^{n}\): target values  
- \(\mathbf{X}\in\mathbb{R}^{n\times p}\): design matrix (n rows, p features) with a 1s column for the intercept  
- \(\boldsymbol{\beta}\in\mathbb{R}^{p}\): coefficients  
- \(\boldsymbol{\varepsilon}\): errors/noise

### How do we choose \(\boldsymbol{\beta}\)? — Ordinary Least Squares (OLS)
We minimize the **Residual Sum of Squares (RSS)**:  
\[
RSS(\boldsymbol{\beta}) = \sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2
= \lVert \mathbf{y}-\mathbf{X}\boldsymbol{\beta}\rVert_2^2
\]

The (normal‑equations) solution when \(\mathbf{X}^\top\mathbf{X}\) is invertible:  
\[
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \big(\mathbf{X}^\top\mathbf{X}\big)^{-1}\mathbf{X}^\top\mathbf{y}
\]

> **Why square the errors?** To avoid positives and negatives canceling out and to penalize large mistakes more heavily.

### Mini worked example (by hand)
Suppose we have 3 points (Age → Salary in $k):  
(25, 45), (35, 65), (45, 85).  
You can compute the slope \(\beta_1=\frac{\text{Cov}(X,Y)}{\text{Var}(X)}\) and intercept \(\beta_0=\bar{Y}-\beta_1\bar{X}\), then use \( \hat{Y}=\beta_0+\beta_1X\) to predict. The residuals \(e_i=Y_i-\hat{Y}_i\) tell you how far off each prediction is; summing \(e_i^2\) gives RSS.

### Quick vocabulary
- **Residual**: \(e=Y-\hat{Y}\) (error per data point)  
- **RSS**: sum of squared residuals (what OLS minimizes)  
- **Coefficient**: a weight showing how much a feature moves \(Y\)

---

## 2) Assumptions You Must Check (`00:25–00:45`)

### TL;DR
Linear regression works best when these are approximately true:

1) **Linearity:** relationship between X and Y is roughly linear.  
2) **Independence:** rows are independent of each other.  
3) **Homoscedasticity:** residuals have constant variance across predictions.  
4) **Normality of residuals:** residuals are roughly bell‑shaped.  
5) **No multicollinearity:** features aren’t highly correlated with each other.

### How to check (practical)
- **Linearity:** scatterplots of each \(X_j\) vs \(Y\); partial‑residual plots.  
- **Independence:** ensure sampling design; for time‑series, check autocorrelation (ACF).  
- **Homoscedasticity:** residuals vs fitted plot should look like a **cloud**, not a funnel.  
- **Normality:** histogram or Q‑Q plot of residuals ~ straight line.  
- **Multicollinearity:** compute **VIF** (next section).

> ⚠️ If these fail badly, your estimates and p‑values can be misleading and predictions may degrade.

---

## 3) Overfitting & Multicollinearity (`00:45–01:05`)

### TL;DR
- **Overfitting:** model memorizes noise → great on training, poor on new data.  
- **Multicollinearity:** features carry overlapping information → coefficients unstable.

### Intuition
Adding lots of irrelevant features (e.g., *address*, *gender*) or duplicates of the same information (e.g., *age* & *experience*) can inflate training \(R^2\) without truly improving generalization.

### Variance Inflation Factor (VIF)
To detect multicollinearity for a feature \(X_j\): regress \(X_j\) on the remaining features and get \(R_j^2\).  
\[
VIF_j = \frac{1}{1 - R_j^2}
\]
- Rule of thumb: **VIF > 5** (sometimes 10) signals problematic collinearity.
- What to do: remove/merge features, or use **regularization**.

---

## 4) Regularization Overview (`01:05–01:25`)

### TL;DR
We *modify the loss* to discourage large coefficients. This reduces variance and combats overfitting.

**Unregularized loss:** \( \sum (Y-\hat{Y})^2 \)  
**Regularized loss:** \( \sum (Y-\hat{Y})^2 + \text{Penalty}(\boldsymbol{\beta}) \)

The penalty strength is controlled by **\(\lambda\)** (lambda). Larger \(\lambda\) ⇒ stronger shrinkage.

> **Feature scaling matters:** Standardize features before regularization so the penalty treats all coefficients comparably.

---

## 5) Ridge Regression (L2) (`01:25–01:40`)

### Definition & math
\[
J_{\text{Ridge}}(\boldsymbol{\beta})=
\underbrace{\lVert \mathbf{y}-\mathbf{X}\boldsymbol{\beta}\rVert_2^2}_{\text{fit}} +
\lambda \underbrace{\lVert \boldsymbol{\beta}\rVert_2^2}_{\text{shrink}}
\]

Closed‑form solution (unlike Lasso):  
\[
\hat{\boldsymbol{\beta}}_{\text{Ridge}}=\big(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I}\big)^{-1}\mathbf{X}^\top \mathbf{y}
\]

### What it does
- Shrinks coefficients **toward** 0 (never exactly 0).  
- Handles **multicollinearity** by stabilizing \(\mathbf{X}^\top\mathbf{X}\).  
- Keeps all features but tames them.

### Practical notes
- Tune \(\lambda\) via cross‑validation.  
- Standardize features (mean 0, std 1) before fitting.  
- Larger \(\lambda\) ⇒ smoother, lower‑variance model but higher bias.

---

## 6) Lasso Regression (L1) (`01:40–01:50`)

### Definition & math
\[
J_{\text{Lasso}}(\boldsymbol{\beta})=
\lVert \mathbf{y}-\mathbf{X}\boldsymbol{\beta}\rVert_2^2 +
\lambda \lVert \boldsymbol{\beta}\rVert_1
= \sum (Y-\hat{Y})^2 + \lambda \sum_j |\beta_j|
\]

### What it does
- Can drive some coefficients **exactly to 0** → **feature selection**.  
- Useful when you expect many irrelevant features.

### Practical notes
- No closed‑form; solved by **coordinate descent** (soft‑thresholding).  
- Standardize features; otherwise, penalty is unfair across scales.  
- If features are **highly correlated**, Lasso may pick one arbitrarily and drop others.

---

## 7) Elastic Net (L1 + L2) (`01:50–02:00`)

### Definition & math
\[
J_{\text{EN}}(\boldsymbol{\beta}) = \lVert \mathbf{y}-\mathbf{X}\boldsymbol{\beta}\rVert_2^2
+ \lambda\Big(\alpha \lVert \boldsymbol{\beta}\rVert_1 + (1-\alpha)\lVert \boldsymbol{\beta}\rVert_2^2\Big)
\]

- \(\alpha=1\) ⇒ Lasso; \(\alpha=0\) ⇒ Ridge.  
- Great when you want **sparsity** *and* **grouping** (correlated features share weight).

### Tuning
- Cross‑validate over a grid of \(\lambda\) and \(\alpha\) (e.g., \(\alpha\in\{0, 0.25, 0.5, 0.75, 1\}\)).

---

## 8) Bias–Variance Trade‑Off (`02:00–02:10`)

### Decomposition
\[
\mathbb{E}\big[(Y-\hat{f}(X))^2\big] = \underbrace{\text{Bias}^2}_{\text{too simple}} + \underbrace{\text{Variance}}_{\text{too wiggly}} + \underbrace{\sigma^2}_{\text{noise}}
\]

- **High bias** ⇒ underfit (misses structure).  
- **High variance** ⇒ overfit (memorizes noise).  
- **Regularization** reduces variance at the cost of a bit more bias.

### Learning curves
- Plot **training** and **validation** error vs **model complexity** (or vs number of samples).  
- Best spot is near the **validation minimum**.

---

## 9) Gradient Descent & Learning (`02:10–02:30`)

### Why we need it
For large datasets or for models without closed forms (e.g., Lasso), we minimize loss by iterative updates.

### Gradients for linear regression
Unregularized MSE loss:
\[
J(\boldsymbol{\beta})=\frac{1}{n}\lVert \mathbf{y}-\mathbf{X}\boldsymbol{\beta}\rVert_2^2,\quad
\nabla_{\boldsymbol{\beta}} J = -\frac{2}{n}\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})
\]

With **L2** regularization: add \(2\lambda\boldsymbol{\beta}\) to the gradient.

### Update rule
\[
\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \eta \,\nabla_{\boldsymbol{\beta}} J
\]
- \(\eta\): learning rate — too big ⇒ diverge; too small ⇒ slow.

### Variants
- **Batch GD**: full dataset per step (stable, slow).  
- **SGD**: one sample per step (fast, noisy).  
- **Mini‑batch**: small batches (common default).

### Convergence tips
- **Feature scaling** dramatically helps.  
- Monitor validation loss; use early stopping.  
- Try learning‑rate schedules (decay) or adaptive optimizers for complex models.

---

## 10) Model Evaluation & Tuning

### Splits
- **Train** (fit coefficients), **Validation** (choose \(\lambda,\alpha\)), **Test** (final check).  
- Or use **k‑fold cross‑validation** (k=5 or 10).

### Metrics
- **\(R^2\)**: fraction of variance explained (higher is better).  
- **MSE / RMSE**: mean squared error / its square root (lower is better).  
- **MAE**: mean absolute error (robust to outliers).

### What to compare
- Compare **train vs validation**.  
  - Train low error + validation high error ⇒ **overfit**.  
  - Both high ⇒ **underfit** or wrong features.

---

## 11) End‑to‑End Recipe (Do This When You Build a Model)

1. **Define goal**: what are you predicting and why?  
2. **Assemble features**: include relevant, diverse predictors; avoid obvious duplicates.  
3. **Split data** into train/validation/test (or k‑fold CV).  
4. **Scale features** (standardize).  
5. **Baseline**: fit OLS; record metrics.  
6. **Check assumptions** with residual plots & Q‑Q plot; compute **VIF**.  
7. **Regularize**: try Ridge, Lasso, Elastic Net over a grid of \(\lambda\) (and \(\alpha\)).  
8. **Pick the winner**: choose the model with best validation performance **and** reasonable simplicity.  
9. **Refit on train+val** with chosen hyperparameters.  
10. **Test once**; report test metrics.  
11. **Document** what features mattered and why.

---

## 12) Troubleshooting & Pitfalls

| Symptom | Likely Cause | Fix |
|---|---|---|
| Train good, validation bad | Overfitting | Increase \(\lambda\), simplify features, get more data, use CV |
| Coefficients huge/unstable | Multicollinearity | Remove/merge correlated features, Ridge/Elastic Net |
| Lasso drops “important” feature | High correlation | Use Elastic Net or group features |
| GD diverges | Learning rate too high | Lower \(\eta\), standardize features |
| GD very slow | Learning rate too low | Raise \(\eta\), use mini‑batches, scale features |
| Weird residual pattern | Non‑linearity / heteroscedasticity | Transform features/target; try non‑linear models |

---

## 13) Practice (Self‑Check)

1) Hand‑compute \(\beta_0,\beta_1\) for 3 points; predict at a new \(X\).  
2) Given \(R^2=0.84\) when regressing \(X_2\) on other features, compute \(VIF_2\).  
3) Explain when you’d prefer Ridge vs Lasso vs Elastic Net.  
4) Describe a learning curve that indicates underfitting.  
5) Why standardize before regularization?

**Answers (sketch):**
1) \(\beta_1=\frac{\text{Cov}(X,Y)}{\text{Var}(X)}\), \(\beta_0=\bar{Y}-\beta_1\bar{X}\).  
2) \(VIF=\frac{1}{1-0.84}=6.25\) → high multicollinearity.  
3) Ridge for correlated features, Lasso for sparsity/feature selection, Elastic Net for both.  
4) Training and validation errors both large and close together across complexities.  
5) To ensure the penalty treats coefficients fairly despite different units/scales.

---

## 14) Glossary (One‑liners)

- **Coefficient (\(\beta\))**: weight multiplying a feature.  
- **Intercept**: baseline value when inputs are zero.  
- **Residual**: actual minus predicted.  
- **RSS**: sum of squared residuals.  
- **VIF**: how much a feature’s variance is inflated by other features.  
- **Regularization**: adding a penalty to discourage large coefficients.  
- **Ridge/Lasso/Elastic Net**: L2/L1/hybrid penalties.  
- **Gradient Descent**: iterative method to minimize loss.  
- **Bias/Variance**: under/over‑fitting tendencies.  
- **Cross‑validation**: rotating validation to estimate generalization.

---

## 15) What to Watch in the Recording (Timestamp Guide)

- `00:00–00:25` — OLS recap; why square residuals; best‑fit line intuition.  
- `00:25–00:45` — Assumptions: linearity, residual normality, homoscedasticity, independence, multicollinearity.  
- `00:45–01:05` — Overfitting demo with irrelevant & correlated features; VIF introduction.  
- `01:05–01:25` — Regularization motivation; cost vs penalty; lambda as a hyperparameter.  
- `01:25–01:40` — Ridge math, effect on coefficients, when to use it.  
- `01:40–01:50` — Lasso math, sparsity and feature selection behavior.  
- `01:50–02:00` — Elastic Net formula; choosing \(\alpha\).  
- `02:00–02:10` — Bias–variance connection to regularization.  
- `02:10–02:30` — Gradient Descent basics; update rule; learning rate intuition.  
- `02:30–02:45` — Q&A clarifying supervised vs unsupervised; VIF vs regularization roles.

---

## 16) One‑Page Recap (Print This)

**Model**: \(\hat{\mathbf{y}}=\mathbf{X}\hat{\boldsymbol{\beta}}\), \(\hat{\boldsymbol{\beta}}_{\text{OLS}}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}\)  
**Regularizers**: Ridge \(\lambda\lVert\beta\rVert_2^2\), Lasso \(\lambda\lVert\beta\rVert_1\), Elastic Net \(\lambda(\alpha\lVert\beta\rVert_1+(1-\alpha)\lVert\beta\rVert_2^2)\)  
**VIF**: \(VIF=1/(1-R^2)\)  
**GD update**: \(\beta \leftarrow \beta-\eta\nabla J\)  
**Checklist**: Split → Scale → Baseline OLS → Assumptions → VIF → Regularize + CV → Select → Test → Document.

---

**You’re caught up.** Use the timestamp guide to watch just the pieces you need, then run your own small experiment repeating the recipe above. Good luck!
