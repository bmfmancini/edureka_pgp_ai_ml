"""
Ridge Regression Example for Applied ML Class 4
================================================

This example demonstrates Ridge (L2) regression as covered in the Class 4 material.
It shows:
- Data generation with multicollinearity
- OLS baseline
- Ridge regression with hyperparameter tuning
- Cross-validation
- Visualization of results
- Comparison with OLS

References: Applied_ML_Class4_CatchupGuide_Comprehensive.md (Section 5, timestamps 01:25-01:40)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def generate_data_with_multicollinearity(n_samples=200, n_features=10, noise=10, random_state=42):
    """
    Generate regression data with multicollinearity to demonstrate Ridge's advantages.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise : float
        Standard deviation of Gaussian noise
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : ndarray
        Feature matrix with multicollinearity
    y : ndarray
        Target values
    """
    np.random.seed(random_state)
    
    # Generate base features
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=noise, random_state=random_state)
    
    # Add multicollinearity by creating correlated features
    # Add copies with some noise to create highly correlated features
    X[:, 5] = X[:, 0] + np.random.normal(0, 0.1, n_samples)  # Highly correlated with feature 0
    X[:, 6] = X[:, 1] + np.random.normal(0, 0.1, n_samples)  # Highly correlated with feature 1
    
    return X, y


def calculate_vif(X):
    """
    Calculate Variance Inflation Factor (VIF) for features.
    VIF = 1 / (1 - R²_j) where R²_j is from regressing X_j on other features.
    
    VIF > 5 (or 10) indicates problematic multicollinearity.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
        
    Returns:
    --------
    vif_data : DataFrame
        VIF values for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = [f"X{i}" for i in range(X.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    return vif_data


def compare_ols_ridge(X_train, X_test, y_train, y_test, alphas=None):
    """
    Compare OLS and Ridge regression with different regularization strengths.
    
    Parameters:
    -----------
    X_train, X_test : ndarray
        Training and test features (already scaled)
    y_train, y_test : ndarray
        Training and test targets
    alphas : list
        Ridge alpha (lambda) values to try
        
    Returns:
    --------
    results : dict
        Dictionary containing models and their metrics
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    results = {}
    
    # OLS (equivalent to Ridge with alpha=0, but using LinearRegression)
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols_train = ols.predict(X_train)
    y_pred_ols_test = ols.predict(X_test)
    
    results['OLS'] = {
        'model': ols,
        'train_mse': mean_squared_error(y_train, y_pred_ols_train),
        'test_mse': mean_squared_error(y_test, y_pred_ols_test),
        'train_r2': r2_score(y_train, y_pred_ols_train),
        'test_r2': r2_score(y_test, y_pred_ols_test),
        'coefficients': ols.coef_
    }
    
    # Ridge with different alphas
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred_ridge_train = ridge.predict(X_train)
        y_pred_ridge_test = ridge.predict(X_test)
        
        results[f'Ridge_α={alpha}'] = {
            'model': ridge,
            'alpha': alpha,
            'train_mse': mean_squared_error(y_train, y_pred_ridge_train),
            'test_mse': mean_squared_error(y_test, y_pred_ridge_test),
            'train_r2': r2_score(y_train, y_pred_ridge_train),
            'test_r2': r2_score(y_test, y_pred_ridge_test),
            'coefficients': ridge.coef_
        }
    
    return results


def find_best_alpha_cv(X_train, y_train, alphas=None, cv=5):
    """
    Find the best Ridge alpha using cross-validation.
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training targets
    alphas : list
        Alpha values to try
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    best_ridge : Ridge
        Fitted Ridge model with best alpha
    cv_results : dict
        Cross-validation results
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)  # From 0.001 to 1000
    
    # Using RidgeCV for efficient cross-validation
    ridge_cv = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train, y_train)
    
    # Manually compute CV scores for visualization
    from sklearn.model_selection import cross_val_score
    cv_scores_list = []
    for alpha in alphas:
        ridge_temp = Ridge(alpha=alpha)
        scores = cross_val_score(ridge_temp, X_train, y_train, cv=cv, 
                                scoring='neg_mean_squared_error')
        cv_scores_list.append(-scores.mean())  # Convert back to positive MSE
    
    cv_results = {
        'best_alpha': ridge_cv.alpha_,
        'cv_scores': np.array(cv_scores_list),
        'alphas_tested': alphas
    }
    
    return ridge_cv, cv_results


def plot_results(results, cv_results=None, figsize=(15, 10)):
    """
    Visualize Ridge regression results.
    
    Parameters:
    -----------
    results : dict
        Results from compare_ols_ridge
    cv_results : dict
        Results from find_best_alpha_cv
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # 1. Training vs Test MSE for different alphas
    ax1 = plt.subplot(2, 3, 1)
    alphas = [results[k]['alpha'] for k in results.keys() if 'Ridge' in k]
    train_mse = [results[k]['train_mse'] for k in results.keys() if 'Ridge' in k]
    test_mse = [results[k]['test_mse'] for k in results.keys() if 'Ridge' in k]
    
    ax1.semilogx(alphas, train_mse, 'o-', label='Train MSE', linewidth=2)
    ax1.semilogx(alphas, test_mse, 's-', label='Test MSE', linewidth=2)
    ax1.axhline(y=results['OLS']['train_mse'], color='r', linestyle='--', label='OLS Train MSE')
    ax1.axhline(y=results['OLS']['test_mse'], color='orange', linestyle='--', label='OLS Test MSE')
    ax1.set_xlabel('Ridge Alpha (λ)', fontsize=10)
    ax1.set_ylabel('Mean Squared Error', fontsize=10)
    ax1.set_title('MSE vs Regularization Strength', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. R² scores
    ax2 = plt.subplot(2, 3, 2)
    train_r2 = [results[k]['train_r2'] for k in results.keys() if 'Ridge' in k]
    test_r2 = [results[k]['test_r2'] for k in results.keys() if 'Ridge' in k]
    
    ax2.semilogx(alphas, train_r2, 'o-', label='Train R²', linewidth=2)
    ax2.semilogx(alphas, test_r2, 's-', label='Test R²', linewidth=2)
    ax2.axhline(y=results['OLS']['train_r2'], color='r', linestyle='--', label='OLS Train R²')
    ax2.axhline(y=results['OLS']['test_r2'], color='orange', linestyle='--', label='OLS Test R²')
    ax2.set_xlabel('Ridge Alpha (λ)', fontsize=10)
    ax2.set_ylabel('R² Score', fontsize=10)
    ax2.set_title('R² vs Regularization Strength', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Coefficient magnitudes
    ax3 = plt.subplot(2, 3, 3)
    ols_coef_norm = np.linalg.norm(results['OLS']['coefficients'])
    ridge_coef_norms = [np.linalg.norm(results[k]['coefficients']) 
                        for k in results.keys() if 'Ridge' in k]
    
    ax3.semilogx(alphas, ridge_coef_norms, 'o-', linewidth=2, label='Ridge Coefficients')
    ax3.axhline(y=ols_coef_norm, color='r', linestyle='--', label='OLS Coefficients')
    ax3.set_xlabel('Ridge Alpha (λ)', fontsize=10)
    ax3.set_ylabel('L2 Norm of Coefficients', fontsize=10)
    ax3.set_title('Coefficient Shrinkage Effect', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Individual coefficients comparison (OLS vs best Ridge)
    ax4 = plt.subplot(2, 3, 4)
    best_ridge_key = min([k for k in results.keys() if 'Ridge' in k], 
                        key=lambda k: results[k]['test_mse'])
    
    feature_names = [f'X{i}' for i in range(len(results['OLS']['coefficients']))]
    x_pos = np.arange(len(feature_names))
    
    width = 0.35
    ax4.bar(x_pos - width/2, results['OLS']['coefficients'], width, 
            label='OLS', alpha=0.8)
    ax4.bar(x_pos + width/2, results[best_ridge_key]['coefficients'], width,
            label=f'Ridge (best α)', alpha=0.8)
    
    ax4.set_xlabel('Features', fontsize=10)
    ax4.set_ylabel('Coefficient Value', fontsize=10)
    ax4.set_title('OLS vs Ridge Coefficients', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(feature_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cross-validation results (if provided)
    if cv_results and cv_results.get('cv_scores') is not None:
        ax5 = plt.subplot(2, 3, 5)
        cv_mse = cv_results['cv_scores']
        
        ax5.semilogx(cv_results['alphas_tested'], cv_mse, 'b-', linewidth=2, marker='o')
        ax5.axvline(x=cv_results['best_alpha'], color='r', linestyle='--', 
                   label=f"Best α={cv_results['best_alpha']:.3f}")
        ax5.set_xlabel('Ridge Alpha (λ)', fontsize=10)
        ax5.set_ylabel('Cross-Validation MSE', fontsize=10)
        ax5.set_title('Cross-Validation Curve', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Bias-Variance illustration
    ax6 = plt.subplot(2, 3, 6)
    gap = [results[k]['train_mse'] - results[k]['test_mse'] 
           for k in results.keys() if 'Ridge' in k]
    
    ax6.semilogx(alphas, np.abs(gap), 'o-', linewidth=2, color='purple')
    ax6.set_xlabel('Ridge Alpha (λ)', fontsize=10)
    ax6.set_ylabel('|Train MSE - Test MSE|', fontsize=10)
    ax6.set_title('Overfitting Gap (Variance Indicator)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/edureka_pgp_ai_ml/edureka_pgp_ai_ml/notes/Applied_Machine_Learning_Techniques/examples/ridge_regression_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'ridge_regression_results.png'")


def print_summary(results, vif_data=None, best_alpha=None):
    """
    Print a summary of the results.
    
    Parameters:
    -----------
    results : dict
        Results from compare_ols_ridge
    vif_data : DataFrame
        VIF values
    best_alpha : float
        Best alpha from cross-validation
    """
    print("\n" + "="*80)
    print("RIDGE REGRESSION ANALYSIS SUMMARY")
    print("="*80)
    
    # VIF Analysis
    if vif_data is not None:
        print("\n1. MULTICOLLINEARITY CHECK (VIF)")
        print("-" * 40)
        print(vif_data.to_string(index=False))
        print("\nNote: VIF > 5 indicates problematic multicollinearity")
        high_vif = vif_data[vif_data['VIF'] > 5]
        if len(high_vif) > 0:
            print(f"⚠️  {len(high_vif)} feature(s) have high multicollinearity")
    
    # Model Performance Comparison
    print("\n2. MODEL PERFORMANCE COMPARISON")
    print("-" * 40)
    print(f"{'Model':<20} {'Train MSE':<12} {'Test MSE':<12} {'Train R²':<12} {'Test R²':<12}")
    print("-" * 80)
    
    # OLS
    ols_res = results['OLS']
    print(f"{'OLS':<20} {ols_res['train_mse']:<12.4f} {ols_res['test_mse']:<12.4f} "
          f"{ols_res['train_r2']:<12.4f} {ols_res['test_r2']:<12.4f}")
    
    # Ridge models
    for key in sorted([k for k in results.keys() if 'Ridge' in k], 
                     key=lambda k: results[k]['alpha']):
        res = results[key]
        print(f"{key:<20} {res['train_mse']:<12.4f} {res['test_mse']:<12.4f} "
              f"{res['train_r2']:<12.4f} {res['test_r2']:<12.4f}")
    
    # Best model
    print("\n3. BEST MODEL SELECTION")
    print("-" * 40)
    best_key = min([k for k in results.keys() if 'Ridge' in k], 
                   key=lambda k: results[k]['test_mse'])
    best_res = results[best_key]
    print(f"Best model by Test MSE: {best_key}")
    print(f"  - Test MSE: {best_res['test_mse']:.4f}")
    print(f"  - Test R²: {best_res['test_r2']:.4f}")
    print(f"  - Coefficient L2 norm: {np.linalg.norm(best_res['coefficients']):.4f}")
    
    if best_alpha:
        print(f"\nBest alpha from CV: {best_alpha:.4f}")
    
    # Key Insights
    print("\n4. KEY INSIGHTS")
    print("-" * 40)
    
    # Check overfitting
    ols_gap = ols_res['train_mse'] - ols_res['test_mse']
    best_gap = best_res['train_mse'] - best_res['test_mse']
    
    print(f"OLS Train-Test gap: {ols_gap:.4f}")
    print(f"Best Ridge Train-Test gap: {best_gap:.4f}")
    
    if abs(best_gap) < abs(ols_gap):
        print("✓ Ridge reduced overfitting compared to OLS")
    
    # Coefficient shrinkage
    ols_norm = np.linalg.norm(ols_res['coefficients'])
    ridge_norm = np.linalg.norm(best_res['coefficients'])
    shrinkage_pct = (1 - ridge_norm/ols_norm) * 100
    
    print(f"Coefficient shrinkage: {shrinkage_pct:.1f}%")
    print("✓ Ridge successfully shrinks coefficients toward zero")
    
    print("\n" + "="*80)


def main():
    """
    Main function to run the complete Ridge regression analysis.
    """
    print("\n" + "="*80)
    print("RIDGE REGRESSION DEMONSTRATION")
    print("Applied Machine Learning - Class 4")
    print("="*80)
    
    # Step 1: Generate data
    print("\n📊 Step 1: Generating data with multicollinearity...")
    X, y = generate_data_with_multicollinearity(n_samples=200, n_features=10, noise=10)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Step 2: Train-test split
    print("\n✂️  Step 2: Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Step 3: Feature scaling (CRITICAL for Ridge!)
    print("\n📏 Step 3: Standardizing features (mean=0, std=1)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Features scaled")
    
    # Step 4: Check multicollinearity
    print("\n🔍 Step 4: Checking for multicollinearity (VIF)...")
    try:
        vif_data = calculate_vif(X_train_scaled)
        print("   VIF calculated (see summary below)")
    except:
        print("   ⚠️  VIF calculation requires statsmodels (optional)")
        vif_data = None
    
    # Step 5: Compare OLS and Ridge with different alphas
    print("\n🔬 Step 5: Comparing OLS and Ridge regression...")
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    results = compare_ols_ridge(X_train_scaled, X_test_scaled, y_train, y_test, alphas)
    print(f"   Tested {len(alphas)} different alpha values")
    
    # Step 6: Find best alpha using cross-validation
    print("\n🎯 Step 6: Finding optimal alpha via cross-validation...")
    best_ridge, cv_results = find_best_alpha_cv(X_train_scaled, y_train, cv=5)
    print(f"   Best alpha: {cv_results['best_alpha']:.4f}")
    
    # Step 7: Visualize results
    print("\n📈 Step 7: Generating visualizations...")
    plot_results(results, cv_results)
    
    # Step 8: Print summary
    print_summary(results, vif_data, cv_results['best_alpha'])
    
    print("\n✅ Analysis complete!")
    print("\nKey Takeaways:")
    print("  1. Ridge adds L2 penalty: λ||β||²₂")
    print("  2. Shrinks coefficients toward zero (but never exactly zero)")
    print("  3. Handles multicollinearity by stabilizing X'X")
    print("  4. Tune λ via cross-validation")
    print("  5. ALWAYS standardize features before applying Ridge!")
    print("\nSee Applied_ML_Class4_CatchupGuide_Comprehensive.md for theory.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
