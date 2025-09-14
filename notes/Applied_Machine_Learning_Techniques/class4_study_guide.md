# 📘 Study Guide: Regression & Model Basics

---

## 1. Regression Formula  

**What it means:**  
- We want to predict something (**Y**), like salary.  
- We use information (**X’s**) such as age or years of experience.  
- Each piece of information has a weight (**β**, “beta”), which says how important it is.  
- **β₀** is the starting point (baseline).  
- **ε (epsilon)** is the random stuff we can’t explain.  

**Plain Example:**  
Salary = Starting amount + (Years of Experience × importance) + (Age × importance) + … + Random noise  

### 📚 Supplemental Resources  
- **YouTube:** [StatQuest: Linear Models Explained](https://www.youtube.com/watch?v=nk2CQITm_eo)  
- **Article:** [Understanding Linear Regression (Towards Data Science)](https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9)  
- **Book:** *An Introduction to Statistical Learning*  

---

## 2. Multicollinearity  

**What it means:**  
- When two pieces of information tell almost the same story, the model gets confused.  
- Example: *Age* and *Years of Experience* are often very similar.  

**Why it’s bad:**  
- Hard to tell which predictor really matters.  
- Coefficients jump around and become unreliable.  

**How to fix:**  
- Remove one of the similar predictors.  
- Combine them.  
- Use regularization (Ridge, Lasso).  

**Plain Example:**  
It’s like asking two friends the same question. If both answer almost the same thing, it doesn’t really help you.  

### 📚 Supplemental Resources  
- **YouTube:** [Multicollinearity Explained Simply](https://www.youtube.com/watch?v=2M4wH5QdC5U)  
- **Article:** [Multicollinearity in Regression (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/)  
- **Book:** *The Elements of Statistical Learning*  

---

## 3. R² (R-Squared)  

**What it means:**  
- R² tells you how much of the outcome is explained by your predictors.  
- It’s a score from 0 to 1 (or 0% to 100%).  

**How to read it:**  
- R² = 0 → Model explains nothing.  
- R² = 1 → Model explains everything.  
- R² = 0.70 → Model explains 70% of what happens.  

**Plain Example:**  
Think of it like a school grade for your model:  
- A+ = 100% (explains everything — rare in real life).  
- C = 50% (explains half).  
- F = 0% (explains nothing).  

⚠️ Note: A high R² doesn’t always mean the model is good. It might just be memorizing (overfitting).  

### 📚 Supplemental Resources  
- **YouTube:** [R² in Regression Explained](https://www.youtube.com/watch?v=E5RjzSK0fvY)  
- **Article:** [What is R²? (Simply Psychology)](https://www.simplypsychology.org/coefficient-of-determination.html)  
- **Book:** *An Introduction to Statistical Learning*  

---

## 4. Regularization  

**What it means:**  
- Regularization is a way to **keep models under control** so they don’t overfit.  
- Overfitting = when a model learns the training data too perfectly, but fails on new data.  
- Regularization makes the model a little “simpler” by shrinking or limiting the importance (β’s) of predictors.  

### 4.1 Ridge Regression (L2)  
- Adds a penalty on large coefficients by squaring them.  
- Shrinks coefficients but never makes them exactly zero.  
- Works best when many predictors all matter a little bit.  

**Plain Example:**  
Like using smaller measuring spoons so no ingredient takes over the recipe.  

📚 Supplemental Resources for Ridge  
- **YouTube:** [StatQuest: Ridge Regression Explained](https://www.youtube.com/watch?v=Q81RR3yKn30)  
- **Article:** [Ridge Regression Explained (Simplilearn)](https://www.simplilearn.com/tutorials/machine-learning-tutorial/ridge-regression)  
- **Book:** *An Introduction to Statistical Learning*  

---

### 4.2 Lasso Regression (L1)  
- Adds a penalty on the absolute values of coefficients.  
- Can shrink some coefficients all the way to zero (removes unimportant predictors).  
- Good for feature selection.  

**Plain Example:**  
Like cleaning your closet — toss out what you don’t use.  

📚 Supplemental Resources for Lasso  
- **YouTube:** [Lasso Regression Explained](https://www.youtube.com/watch?v=Q81RR3yKn30)  
- **Article:** [L1 Regularization (Towards Data Science)](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)  
- **Book:** *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*  

---

### 4.3 Elastic Net (L1 + L2)  
- Mixes Ridge and Lasso.  
- Handles correlated predictors better than Lasso alone.  
- Can both shrink and select features.  

**Plain Example:**  
Keeps a sensible group of predictors and shrinks them.  

📚 Supplemental Resources for Elastic Net  
- **YouTube:** [StatQuest: Elastic Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo)  
- **Article:** [Elastic Net Regression (Towards Data Science)](https://towardsdatascience.com/elastic-net-regression-49fdb8a992af)  
- **Book:** *An Introduction to Statistical Learning*  

---

## 5. Gradient Descent  

**What it is:**  
- A method to find the best coefficients (β’s).  
- Like rolling a ball downhill until it stops at the lowest point (minimum error).  

**Steps:**  
1. Start with random values for coefficients.  
2. Calculate predictions and errors.  
3. Look at the slope (gradient) of the error.  
4. Update coefficients a little in the opposite direction.  
5. Repeat until you reach the minimum error.  

**Key Term:** Learning Rate (α) = how big your steps are.  
- Too big → might overshoot.  
- Too small → very slow.  

**Like I’m 5:**  
It’s like feeling your way downhill while blindfolded, step by step, until you reach the bottom.  

### 📚 Supplemental Resources  
- **YouTube:** [Gradient Descent Explained (StatQuest)](https://www.youtube.com/watch?v=sDv4f4s2SB8)  
- **YouTube:** [3Blue1Brown: Gradient Descent, Deep Intuition](https://www.youtube.com/watch?v=IHZwWFHWa-w)  
- **Article:** [Gradient Descent Simplified (Towards Data Science)](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)  
- **Book:** *Deep Learning* by Ian Goodfellow  

---

## 6. Gradient Descent Update Rules (Linear Regression)  

For slope (m) and intercept (c):  

\[
\frac{\partial}{\partial m} = \frac{2}{N} \sum_{i=1}^{N} -x_i (y_i - (mx_i + c))
\]  

\[
\frac{\partial}{\partial c} = \frac{2}{N} \sum_{i=1}^{N} -(y_i - (mx_i + c))
\]  

Update formulas:  

\[
m = m - \alpha \frac{\partial}{\partial m}
\]  

\[
c = c - \alpha \frac{\partial}{\partial c}
\]  

👉 These tell the model how much to nudge the line’s slope and intercept to fit better each step.  

### 📚 Supplemental Resources  
- **YouTube:** [Gradient Descent Update Rule Explained](https://www.youtube.com/watch?v=JXQT_vxqwIs)  
- **Article:** [Linear Regression with Gradient Descent (Medium)](https://medium.com/analytics-vidhya/linear-regression-using-gradient-descent-optimization-from-scratch-846c36a10ab)  

---

## 7. Variance Inflation Factor (VIF)  

**What it is:**  
- VIF checks if a predictor is too similar to others.  
- High VIF = multicollinearity → model confusion.  

**Formula:**  

\[
VIF = \frac{1}{1 - R^2}
\]  

**How to Read:**  
- 1 → no correlation.  
- 1–5 → acceptable.  
- > 5 → potential problem.  
- > 10 → serious multicollinearity.  

**Fixing High VIF:**  
1. Remove variables with high VIF.  
2. Combine similar variables.  
3. Center variables (X – mean).  
4. Use Ridge/Lasso/Elastic Net.  
5. Apply domain knowledge — keep the most meaningful variable.  

**Like I’m 5:**  
It’s like asking two friends the same question. If one always copies the other, you don’t need both.  

### 📚 Supplemental Resources  
- **YouTube:** [Variance Inflation Factor (StatQuest)](https://www.youtube.com/watch?v=H0ZClfaW5rQ)  
- **Article:** [Variance Inflation Factor (Towards Data Science)](https://towardsdatascience.com/variance-inflation-factor-vif-4cda0be2450)  
- **Book:** *An Introduction to Statistical Learning*  

---
