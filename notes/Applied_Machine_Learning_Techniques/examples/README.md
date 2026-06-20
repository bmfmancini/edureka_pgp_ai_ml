# Ridge Regression Example

This directory contains a comprehensive Python example demonstrating Ridge (L2) regression as covered in **Applied ML Class 4**.

## Files

- `ridge_regression_example.py` - Complete Ridge regression implementation with:
  - Data generation with multicollinearity
  - Variance Inflation Factor (VIF) calculation
  - OLS baseline comparison
  - Ridge regression with multiple alpha values
  - Cross-validation for hyperparameter tuning
  - Comprehensive visualizations
  - Detailed analysis and insights

- `ridge_regression_results.png` - Visualization output showing:
  - MSE vs regularization strength
  - R² scores comparison
  - Coefficient shrinkage effect
  - OLS vs Ridge coefficient comparison
  - Cross-validation curve
  - Overfitting gap analysis

## Running the Example

```bash
python3 ridge_regression_example.py
```

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib statsmodels
```

## What You'll Learn

1. **Multicollinearity Detection**: Using VIF (Variance Inflation Factor)
2. **Ridge Regression Fundamentals**: L2 regularization penalty
3. **Hyperparameter Tuning**: Cross-validation to find optimal alpha (λ)
4. **Model Comparison**: OLS vs Ridge performance
5. **Coefficient Shrinkage**: How Ridge reduces coefficient magnitudes
6. **Bias-Variance Trade-off**: Understanding overfitting reduction

## Key Concepts Covered

From **Applied_ML_Class4_CatchupGuide_Comprehensive.md** (Section 5, timestamps 01:25-01:40):

- **Ridge Loss Function**: 
  ```
  J_Ridge(β) = ||y - Xβ||²₂ + λ||β||²₂
  ```

- **Closed-form Solution**:
  ```
  β_Ridge = (X'X + λI)⁻¹X'y
  ```

- **When to Use Ridge**:
  - High multicollinearity between features
  - Need to keep all features (no feature selection)
  - Want stable coefficient estimates
  - Reduce overfitting without removing features

## Example Output

The script will:
1. Generate synthetic data with multicollinearity
2. Split into train/test sets
3. Standardize features (critical for Ridge!)
4. Calculate VIF to detect multicollinearity
5. Compare OLS with Ridge at different alpha values
6. Find optimal alpha via cross-validation
7. Generate comprehensive visualizations
8. Print detailed analysis summary

## Tips

- **Always standardize features** before applying Ridge regression
- Use **cross-validation** to find the best alpha value
- Ridge **shrinks coefficients toward zero** but never exactly to zero
- For feature selection, consider **Lasso** instead (covered in the same guide)
- Check **VIF > 5** to identify multicollinearity issues

## References

- Applied_ML_Class4_CatchupGuide_Comprehensive.md
- Class 4 lecture (timestamps 01:25-01:40 for Ridge regression)
- scikit-learn documentation: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
