# Study Guide – Supervised Machine Learning Basics

## 1. Simple Linear Regression
- Predicts a continuous outcome using **one independent variable**.
- Example: Employee salary vs. years of experience.
- Works best when relationship is **linear**.
- If nonlinear (e.g., gap years causing salary = 0), consider:
  - **Polynomial Regression**
  - **Piecewise Regression**
  - **Tree-based methods**
- Extension: **Multiple Linear Regression** (more predictors).

### Supplemental Resources
📺 YouTube  
- StatQuest: Simple Linear Regression → https://www.youtube.com/watch?v=PaFPbb66DxQ  
- Simplilearn: Linear Regression Tutorial → https://www.youtube.com/watch?v=nk2CQITm_eo  

📝 Blogs / Articles  
- Towards Data Science: Linear Regression → https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9  
- Analytics Vidhya: Regression Analysis → https://www.analyticsvidhya.com/blog/2021/05/regression-analysis-introduction/  

📚 Books / Docs  
- Scikit-learn Docs: Linear Regression → https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares  
- Book: “An Introduction to Statistical Learning” by James et al.  

---

## 2. Classification
- Predicts **categorical outcomes** (discrete classes).
- Types: Binary, Multi-class, Multi-label, Imbalanced.
- Algorithms: Logistic Regression, Decision Trees, Random Forests, SVM, Naïve Bayes, k-NN, Neural Networks.
- Metrics: Confusion Matrix, Accuracy, Precision, Recall, F1-score, ROC/AUC.
- Applications: Spam detection, fraud detection, sentiment analysis, image recognition.

### Supplemental Resources
📺 YouTube  
- StatQuest: Classification and Regression Trees → https://www.youtube.com/watch?v=g9c66TUylZ4  
- Classification in Machine Learning | Simplilearn → https://www.youtube.com/watch?v=atzkZhRRZ0w  
- Confusion Matrix | StatQuest → https://www.youtube.com/watch?v=Kdsp6soqA7o  

📝 Blogs / Articles  
- GeeksforGeeks: Classification in Machine Learning → https://www.geeksforgeeks.org/classification-in-machine-learning/  
- Towards Data Science: Classification Algorithms → https://towardsdatascience.com/choosing-the-right-classification-algorithm-5f0e2f3da639  
- Analytics Vidhya: Introduction to Classification Algorithms → https://www.analyticsvidhya.com/blog/2021/06/classification-algorithms/  

📚 Books / Docs  
- Book: “Pattern Recognition and Machine Learning” by Christopher M. Bishop  
- Book: “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron  
- Scikit-learn Docs: Classification → https://scikit-learn.org/stable/supervised_learning.html#classification  

---

## 3. Train-Test Split
- Splits dataset into training and testing sets (commonly 70/30 or 80/20).
- Prevents **overfitting** and checks **generalization**.
- Related concepts: Validation set, Cross-validation, Data leakage.

### Supplemental Resources
📺 YouTube  
- Train/Test Split and Cross Validation | StatQuest → https://www.youtube.com/watch?v=fSytzGwwBVw  
- Train Test Split in Machine Learning | Simplilearn → https://www.youtube.com/watch?v=6dbrR-WymjI  

📝 Blogs / Articles  
- Towards Data Science: Why Train-Test Split Matters → https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b2aa0e4a82  
- Analytics Vidhya: Train-Test Split Explained → https://www.analyticsvidhya.com/blog/2021/06/train-test-split-in-machine-learning/  

📚 Docs  
- Scikit-learn Docs: train_test_split → https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html  

---

## 4. Bias-Variance Tradeoff
- **Bias**: Error from overly simplistic model (underfitting).
- **Variance**: Error from overly complex model (overfitting).
- Goal: Minimize total error (bias² + variance + irreducible error).
- Strategies:
  - Reduce bias → More complex model, add features.
  - Reduce variance → Simplify model, regularization, more data, ensembles.

### Supplemental Resources
📺 YouTube  
- Bias-Variance Tradeoff | StatQuest → https://www.youtube.com/watch?v=EuBBz3bI-aA  
- Bias and Variance in Machine Learning | Simplilearn → https://www.youtube.com/watch?v=Eu6nM0o9SLg  

📝 Blogs / Articles  
- Towards Data Science: Understanding the Bias-Variance Tradeoff → https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229  
- GeeksforGeeks: Bias and Variance in ML → https://www.geeksforgeeks.org/bias-and-variance-in-machine-learning/  

📚 Books / Docs  
- Book: “The Elements of Statistical Learning” by Hastie, Tibshirani, and Friedman  
- Book: “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron  

---

## 5. Underfitting and Overfitting
- **Underfitting**: High training + test error, too simple.
- **Overfitting**: Low training error, high test error, too complex.
- Fixes: Adjust complexity, use regularization, add/remove features, cross-validation.

### Supplemental Resources
📺 YouTube  
- Overfitting and Underfitting | StatQuest → https://www.youtube.com/watch?v=6dbrR-WymjI  
- Overfitting in Machine Learning | Simplilearn → https://www.youtube.com/watch?v=JrGOjWx5h9k  

📝 Blogs / Articles  
- Towards Data Science: Overfitting vs. Underfitting → https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765  
- GeeksforGeeks: Underfitting and Overfitting → https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/  

📚 Books / Docs  
- Book: “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron  
- Scikit-learn Docs: Model evaluation: Overfitting and Underfitting → https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html  

---

## 6. Performance Metrics (Regression)
- **MAE**: Average absolute error, less sensitive to outliers.
- **MSE**: Average squared error, penalizes large errors.
- **RMSE**: Square root of MSE, interpretable in original units.
- **R²**: Variance explained by the model, ranges from 0–1 (can be negative if poor).

### Supplemental Resources
📺 YouTube  
- StatQuest: R² Explained → https://www.youtube.com/watch?v=2AQKmw14mHM  
- MAE, MSE, RMSE Explained | StatQuest → https://www.youtube.com/watch?v=PaFPbb66DxQ  

📝 Blogs / Articles  
- Towards Data Science: Regression Error Metrics → https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234  
- Analytics Vidhya: Understanding RMSE, MAE, MSE, R² → https://www.analyticsvidhya.com/blog/2021/07/different-ways-to-evaluate-machine-learning-models/  

📚 Docs  
- Scikit-learn Docs: Metrics and scoring → https://scikit-learn.org/stable/modules/model_evaluation.html  

---
