# Machine Learning Study Guide

## ROC Curve (Receiver Operating Characteristic)

### What it is  
- The **ROC curve** is a graphical tool used to evaluate the performance of a binary classification model.  
- It plots the **True Positive Rate (TPR / Sensitivity)** against the **False Positive Rate (FPR / 1 − Specificity)** at different classification thresholds.  
- By changing the threshold, you see the trade-off between correctly identifying positives vs. incorrectly labeling negatives as positives.

### Key Points  
- **True Positive Rate (TPR / Recall / Sensitivity):**  
  TPR = TP / (TP + FN)  
  Measures how many actual positives are correctly classified.  

- **False Positive Rate (FPR):**  
  FPR = FP / (FP + TN)  
  Measures how many actual negatives are incorrectly classified as positive.  

- **Diagonal line (baseline):** A random classifier gives a diagonal line from (0,0) to (1,1).  
- **Above the diagonal:** A good model should have a curve that bows toward the top left corner.  
- **Area Under the Curve (AUC):**  
  - AUC = 0.5 → random guessing.  
  - AUC = 1.0 → perfect classifier.  
  - Higher AUC means better model performance.  

### Why it’s useful  
- Helps compare different models regardless of threshold.  
- Shows the trade-off between sensitivity (catching positives) and specificity (avoiding false alarms).  
- Common in **medical tests, fraud detection, and anomaly detection**, where threshold selection is critical.  

### Real-World Explanations  
1. **Medical Testing (COVID Test Example)**  
   - A COVID test might catch most infected people (high TPR), but if it also says many healthy people are sick (high FPR), doctors lose trust in it.  
   - The ROC curve shows how well the test balances between catching true cases and avoiding false alarms.  

2. **Bank Fraud Detection**  
   - A model flags transactions as “fraud” or “not fraud.”  
   - If the threshold is too low, many legit purchases get flagged (high FPR → annoyed customers).  
   - If the threshold is too high, fraudsters slip through (low TPR → losses for the bank).  
   - The ROC curve helps banks choose the sweet spot.  

3. **Email Spam Filter**  
   - A spam filter marks emails as spam.  
   - Too aggressive (low threshold) → lots of real emails in spam folder (high FPR).  
   - Too lenient (high threshold) → lots of spam slips through (low TPR).  
   - ROC curve helps balance user experience.  

### Supplemental Reading  
- [Scikit-learn ROC Curve documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)  
- [Analytics Vidhya – ROC and AUC explained](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/)  
- [Wikipedia – ROC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)  

### Videos  
- YouTube: [ROC and AUC Explained](https://www.youtube.com/watch?v=4jRBRDbJemM) (StatQuest – clear and simple)  
- YouTube: [ROC Curve in Machine Learning](https://www.youtube.com/watch?v=OAl6eAyP-yo) (Krish Naik)  
- YouTube: [ROC and AUC Explained Visually](https://www.youtube.com/watch?v=QdDoFfkVkcw) (StatQuest deep dive)  

---

## Hyperparameter Tuning

### What it is  
- **Hyperparameters** are the “settings” of a machine learning model that are not learned from the data, but chosen before training.  
- Examples:  
  - Learning rate  
  - Number of hidden layers in a neural network  
  - Depth of a decision tree  
  - Number of clusters in k-means  
- **Tuning** is the process of finding the best combination of hyperparameters to improve model performance.

### Key Methods  
1. **Grid Search** – Tests all combinations of specified hyperparameters.  
2. **Random Search** – Randomly samples hyperparameter values.  
3. **Bayesian Optimization** – Uses probability to choose the next best hyperparameter values to test.  
4. **Automated Tools (AutoML)** – Frameworks like Optuna, Hyperopt.  

### Why it’s useful  
- Hyperparameters can dramatically affect model accuracy, speed, and generalization.  
- Prevents **underfitting** and **overfitting**.  

### Real-World Explanations  
1. **Medical Diagnosis** – Wrong learning rate means misdiagnosis.  
2. **Shopping Recommendations** – Too deep trees = overfitting to specific customers.  
3. **Self-Driving Cars** – Tuning batch size & iterations ensures balance between speed and accuracy.  

### Supplemental Reading  
- [Scikit-learn: Hyperparameter tuning with GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)  
- [Towards Data Science – Hyperparameter Tuning Guide](https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624)  
- [Optuna framework for optimization](https://optuna.org/)  

### Videos  
- YouTube: [Hyperparameter Tuning Explained](https://www.youtube.com/watch?v=GM4Ye0dVHT4)  
- YouTube: [GridSearchCV & RandomizedSearchCV](https://www.youtube.com/watch?v=9Xzl9s7MY7s)  
- YouTube: [Hyperparameter Tuning in ML](https://www.youtube.com/watch?v=4I6NA3Gp2e8)  

---

## Random Forest

### What it is  
- An **ensemble learning method** that builds many decision trees and combines their predictions.  
- Each tree is trained on a **random subset of the data and features**, reducing overfitting.  
- Output:  
  - Classification: majority vote.  
  - Regression: average prediction.  

### Key Points  
- Uses **bagging (bootstrap aggregating)**.  
- Reduces variance compared to one decision tree.  
- Works for both classification and regression.  
- Can measure **feature importance**.  

### Real-World Explanations  
1. **Loan Approval** – Many “loan officers” (trees) voting ensures fairness.  
2. **Medical Imaging** – Different trees detect different features, reducing misdiagnosis.  
3. **E-commerce** – Different browsing patterns analyzed, leading to better recommendations.  

### Supplemental Reading  
- [Scikit-learn: Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)  
- [Analytics Vidhya – Random Forest Simplified](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)  
- [Towards Data Science – Random Forest for Beginners](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)  

### Videos  
- YouTube: [Random Forests Explained](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)  
- YouTube: [Random Forest ML](https://www.youtube.com/watch?v=loNcrMjYh64)  
- YouTube: [Random Forest Implementation](https://www.youtube.com/watch?v=nyxTdL_4Q-Q)  

---

## Grid Search

### What it is  
- A **hyperparameter tuning method** that tries all possible combinations of hyperparameter values.  
- Example: Random Forest with:  
  - n_estimators = [100, 200]  
  - max_depth = [5, 10]  
  - min_samples_split = [2, 4]  
  → Grid search tests 2 × 2 × 2 = 8 combinations.  

### Key Points  
- Works best when parameter space is small.  
- Often used with cross-validation (GridSearchCV).  
- Can be slow if grid is large.  

### Real-World Explanations  
1. **Self-Driving Cars** – Grid search tests every combination to ensure safe parameter choice.  
2. **Telecom Churn Prediction** – Ensures most accurate churn model.  
3. **Voice Assistants** – Finds optimal speech recognition parameters.  

### Supplemental Reading  
- [Scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)  
- [Machine Learning Mastery – Grid Search](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)  
- [Towards Data Science – Practical Hyperparameter Tuning](https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624)  

### Videos  
- YouTube: [Grid Search Explained](https://www.youtube.com/watch?v=Gol_qOgRqfA)  
- YouTube: [Grid vs Random Search](https://www.youtube.com/watch?v=V2Z3T6CXF-4)  
- YouTube: [GridSearchCV Tutorial](https://www.youtube.com/watch?v=H5F4JG3xkPQ)  

