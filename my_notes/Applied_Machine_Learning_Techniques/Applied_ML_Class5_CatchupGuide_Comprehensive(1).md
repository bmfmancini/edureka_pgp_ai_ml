# 🧭 *Applied Machine Learning — Class 5 Catch-Up Guide*
**Comprehensive Beginner-Friendly Edition**
*(With timestamps, LaTeX math, full explanations, and supplementary readings)*

---

## 🕐 00:00 – 00:10 — Introduction

This session transitions from **regression** (predicting continuous values) to **classification** — predicting categories such as *spam vs not spam*, *disease vs no disease*, or *churn vs retain*.

In regression, we used **least squares**; in classification, we model the *probability* of class membership instead of predicting a raw number.  
The class focuses on:  
1. Logistic Regression and the sigmoid function  
2. The cost (log-loss) function  
3. Evaluation metrics (Accuracy, Precision, Recall, F1)  
4. ROC–AUC interpretation  
5. Regularization for logistic models  
6. Gradient Descent optimization  

---

## 🕐 00:10 – 00:35 — Logistic Regression Fundamentals

### Why Linear Regression Fails
Linear regression outputs real numbers \( (-\infty,\infty) \).  
Classification needs probabilities \( P(Y=1|X)\in[0,1] \).  
A linear line can produce values >1 or <0 — invalid as probabilities.

### The Logistic (Sigmoid) Function
We model **log-odds** linearly and then map them to probabilities using the **sigmoid** function.

\[
\text{logit}(p) = \log\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_pX_p
\]

Solving for \(p\):

\[
p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}
\]

This function outputs values smoothly between 0 and 1 — perfect for probabilities.

| Term | Meaning |
|------|----------|
| \(p\) | Predicted probability that \(Y=1\) |
| \(\beta_j\) | Coefficient for feature \(X_j\) |
| \(\sigma(z)\) | Sigmoid = \(1/(1+e^{-z})\) |

### Decision Rule
Predict class 1 if \(p \ge 0.5\); otherwise class 0.  
The threshold 0.5 can be adjusted for imbalanced data.

**Intuition:**  
The logistic function is like a dimmer switch — not binary, but continuous between 0 and 1.

---

### Supplementary Resources
- 🎥 *StatQuest: Logistic Regression Clearly Explained*  
- 📘 *3Blue1Brown: The Sigmoid Function Visualized*  
- 🧾 *Towards Data Science: Understanding Logistic Regression Step-by-Step*

---

## 🕐 00:35 – 00:55 — The Cost Function (Log-Loss)

We can’t use MSE because the sigmoid’s curve makes the error surface **non-convex**, leading to poor convergence.  
Instead, logistic regression uses **log-loss** (binary cross-entropy) derived from maximum likelihood estimation.

For each observation:

\[
L(\beta) = -\big[y\log(\hat{p}) + (1-y)\log(1-\hat{p})\big]
\]

Total cost (to minimize):

\[
J(\beta) = -\frac{1}{n}\sum_{i=1}^{n} \Big[ y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i) \Big]
\]

| Case | Term Survives | Meaning |
|------|----------------|---------|
| \(y=1\) | \(-\log(\hat{p})\) | Penalizes wrong positive prediction |
| \(y=0\) | \(-\log(1-\hat{p})\) | Penalizes wrong negative prediction |

### Intuition
If you predict 0.99 when the truth is 1 → tiny loss.  
If you predict 0.01 when truth is 1 → huge loss.  
Log-loss punishes confident wrong predictions harshly, encouraging calibrated probabilities.

---

### Supplementary
- 🎥 *StatQuest: Maximum Likelihood & Log Loss*  
- 📘 *DeepLearning.AI: Cross-Entropy Explained Simply*

---

## 🕐 00:55 – 01:15 — Gradient Descent for Logistic Regression

Because there’s no closed-form solution, we use **gradient descent** to minimize log-loss.

\[
\frac{\partial J}{\partial \beta_j} = \frac{1}{n}\sum_{i=1}^{n} (\hat{p}_i - y_i)X_{ij}
\]

Update rule:

\[
\beta_j \leftarrow \beta_j - \eta \frac{\partial J}{\partial \beta_j}
\]

| Symbol | Meaning |
|---------|----------|
| \(\eta\) | Learning rate |
| \(\hat{p}_i\) | Predicted probability for sample i |
| \(y_i\) | True label |

Gradient descent repeats until changes in cost are negligible.

**Variants**
- Batch GD: all samples → stable but slow  
- Stochastic GD: one sample → noisy but fast  
- Mini-batch: compromise; best in practice

---

### Supplementary
- 🎥 *3Blue1Brown: Gradient Descent, Visual Intuition*  
- 📘 *StatQuest: Gradient Descent Step-by-Step*

---

## 🕐 01:15 – 01:40 — Model Evaluation Metrics

### Confusion Matrix

|                | Predicted: 1 | Predicted: 0 |
|----------------|---------------|---------------|
| **Actual: 1** | True Positive (TP) | False Negative (FN) |
| **Actual: 0** | False Positive (FP) | True Negative (TN) |

From these, we derive:

\[
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
\]

\[
\text{Precision} = \frac{TP}{TP+FP},\quad
\text{Recall} = \frac{TP}{TP+FN}
\]

\[
F1 = 2 \times \frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}
\]

| Metric | Best Use |
|---------|-----------|
| Accuracy | Balanced datasets |
| Precision | Cost of false positives high |
| Recall | Cost of false negatives high |
| F1 | Balance precision & recall |

**Example:**  
In fraud detection, *recall* matters more — missing fraud is costly.  
In email spam filtering, *precision* matters more — false alarms annoy users.

---

### Supplementary
- 🎥 *StatQuest: Precision, Recall, and F1-Score Explained*  
- 📘 *Towards Data Science: Confusion Matrix for Beginners*

---

## 🕐 01:40 – 02:00 — ROC Curve and AUC

The **Receiver Operating Characteristic (ROC)** curve plots **True Positive Rate (TPR)** vs **False Positive Rate (FPR)** for various thresholds.

\[
TPR = \frac{TP}{TP+FN},\quad FPR = \frac{FP}{FP+TN}
\]

The **Area Under the Curve (AUC)** measures model separability:
- AUC = 1 → perfect classifier  
- AUC = 0.5 → random guessing

**Intuition:**  
Imagine sorting predictions by confidence; AUC measures how well positives rank above negatives.

| Range | Interpretation |
|--------|----------------|
| 0.9–1.0 | Excellent |
| 0.8–0.9 | Good |
| 0.7–0.8 | Fair |
| 0.6–0.7 | Poor |
| 0.5 | Random |

---

### Supplementary
- 🎥 *StatQuest: ROC and AUC Clearly Explained*  
- 📘 *Analytics Vidhya: ROC Curves in Simple Terms*

---

## 🕐 02:00 – 02:25 — Regularization in Logistic Regression

Just like in linear regression, we add penalties to control overfitting.

### Ridge (L2)
\[
J(\beta) = J_{\text{log-loss}} + \lambda \sum_j \beta_j^2
\]

Shrinks coefficients smoothly; keeps all features.

### Lasso (L1)
\[
J(\beta) = J_{\text{log-loss}} + \lambda \sum_j |\beta_j|
\]

Encourages sparsity; removes irrelevant predictors.

### Elastic Net
\[
J(\beta) = J_{\text{log-loss}} + \lambda(\alpha\sum|\beta_j| + (1-\alpha)\sum\beta_j^2)
\]

Blends Ridge & Lasso — useful when features are correlated.

---

### Supplementary
- 🎥 *StatQuest: Ridge, Lasso, Elastic Net for Classification*  
- 📘 *Machine Learning Mastery: Regularization in Logistic Regression*

---

## 🕐 02:25 – 02:40 — Practical Considerations

1. **Scaling:** Always standardize inputs (mean 0, std 1).  
2. **Imbalance:** Use class weighting or resampling (e.g., SMOTE).  
3. **Threshold tuning:** Move away from 0.5 for asymmetric costs.  
4. **Validation:** Prefer cross-validation to single hold-out sets.  
5. **Interpretation:** Coefficients correspond to log-odds; exponentiate to get odds ratios.

---

## 🕐 02:40 – 02:55 — Summary & Key Takeaways

| Concept | Equation / Idea | Intuition |
|----------|----------------|-----------|
| Sigmoid | \( \frac{1}{1+e^{-z}} \) | Maps scores to 0–1 probabilities |
| Log-Loss | \( -[y\log(\hat{p})+(1-y)\log(1-\hat{p})] \) | Punishes confident wrongs |
| GD Update | \( \beta_j\leftarrow\beta_j-\eta(\hat{p}-y)X_j \) | Learn via small corrections |
| Precision | \( TP/(TP+FP) \) | How often positives are correct |
| Recall | \( TP/(TP+FN) \) | How many actual positives caught |
| AUC | Area under ROC | Discrimination ability |

**Main idea:** Logistic regression models the *probability* of belonging to a class using a smooth curve; tuning regularization and thresholds adapts it to real-world data.

---

## 🗂️ Timestamp Index

| Time | Topic |
|------|-------|
| 00:00–00:10 | Introduction |
| 00:10–00:35 | Logistic Regression Fundamentals |
| 00:35–00:55 | Cost Function (Log-Loss) |
| 00:55–01:15 | Gradient Descent |
| 01:15–01:40 | Evaluation Metrics |
| 01:40–02:00 | ROC & AUC |
| 02:00–02:25 | Regularization in Logistic Regression |
| 02:25–02:40 | Practical Considerations |
| 02:40–02:55 | Summary & Wrap-Up |

---

## 📚 Further Reading

- *An Introduction to Statistical Learning*, Ch. 4–5  
- *Hands-On Machine Learning* by Aurélien Géron  
- *StatQuest YouTube Series: Logistic Regression → ROC/AUC → Regularization*  
- *Towards Data Science & Analytics Vidhya Articles* on classification metrics

---

✅ **You’re caught up.**
If you understand each equation and the reasoning behind the metrics, you’re fully aligned with Class 5’s material and ready for the next class on model evaluation and non-linear methods.
