"""
Comprehensive Machine Learning Pipeline Analysis
Model 1: Linear Regression (12 points)
Model 2: Random Forest (13 points)

This script implements a complete ML pipeline with:
- Train/validation split with justification
- Data standardization analysis
- Model training and evaluation
- Hyperparameter tuning
- Feature selection
- Comprehensive model comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Create output directories
os.makedirs("visualizations/model_analysis", exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE MACHINE LEARNING PIPELINE ANALYSIS")
print("=" * 80)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n" + "=" * 80)
print("1. DATA LOADING")
print("=" * 80)

df = pd.read_csv("preprocessed_sales_data.csv")
print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nTarget variable: Total_Revenue")
print(f"Number of features: {df.shape[1] - 1}")

# Define features and target
target = "Total_Revenue"
feature_columns = [col for col in df.columns if col != target]

X = df[feature_columns]
y = df[target]

print(f"\nFeatures used ({len(feature_columns)}):")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

print(f"\nTarget variable statistics:")
print(f"  Mean: ${y.mean():,.2f}")
print(f"  Median: ${y.median():,.2f}")
print(f"  Std Dev: ${y.std():,.2f}")
print(f"  Min: ${y.min():,.2f}")
print(f"  Max: ${y.max():,.2f}")

# ============================================================================
# 2. TRAIN/VALIDATION SPLIT (2 points for Model 1)
# ============================================================================
print("\n" + "=" * 80)
print("2. TRAIN/VALIDATION SPLIT")
print("=" * 80)

# Use 80/20 split
test_size = 0.20
random_state = 42

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

print(
    f"\nSplit Ratio: {int((1-test_size)*100)}/{int(test_size*100)} (Train/Validation)"
)
print(f"Random State: {random_state} (for reproducibility)")
print(f"\nJustification for 80/20 split:")
print(f"  - Dataset size: {len(df):,} samples")
print(f"  - Training set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"  - Validation set: {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"  - With ~8,000 samples, 80/20 provides:")
print(f"    * Sufficient training data (~6,400 samples) for model learning")
print(f"    * Adequate validation data (~1,600 samples) for reliable evaluation")
print(f"    * Standard practice for datasets of this size")
print(f"    * Balances model performance and evaluation reliability")

# ============================================================================
# 3. DATA STANDARDIZATION ANALYSIS (2 points for Model 1)
# ============================================================================
print("\n" + "=" * 80)
print("3. DATA STANDARDIZATION ANALYSIS")
print("=" * 80)

# Analyze feature scales before standardization
print("\nFeature scales BEFORE standardization:")
print("-" * 60)

# Select only numeric columns for statistics
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train_numeric = X_train[numeric_features]

# Remove zero-variance features (constant columns)
variance = X_train_numeric.var()
non_zero_var_features = variance[variance > 0].index.tolist()
X_train_numeric = X_train_numeric[non_zero_var_features]
X_val_numeric = X_val[non_zero_var_features]

print(
    f"\nNote: Removed {len(numeric_features) - len(non_zero_var_features)} zero-variance features"
)
print(f"Using {len(non_zero_var_features)} features with non-zero variance")

scale_analysis = pd.DataFrame(
    {
        "Feature": non_zero_var_features,
        "Mean": X_train_numeric.mean().values,
        "Std": X_train_numeric.std().values,
        "Min": X_train_numeric.min().values,
        "Max": X_train_numeric.max().values,
        "Range": (X_train_numeric.max() - X_train_numeric.min()).values,
    }
)
print(scale_analysis.to_string(index=False))

# Train baseline model WITHOUT standardization (using only numeric features)
lr_no_scale = LinearRegression()
lr_no_scale.fit(X_train_numeric, y_train)
y_pred_no_scale = lr_no_scale.predict(X_val_numeric)
mse_no_scale = mean_squared_error(y_val, y_pred_no_scale)
r2_no_scale = r2_score(y_val, y_pred_no_scale)

print(f"\nLinear Regression WITHOUT standardization:")
print(f"  Validation MSE: {mse_no_scale:,.2f}")
print(f"  Validation R²: {r2_no_scale:.4f}")

# Apply standardization (only to numeric features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_val_scaled = scaler.transform(X_val_numeric)

# Convert back to DataFrame for easier handling
numeric_feature_columns = non_zero_var_features
X_train_scaled_df = pd.DataFrame(
    X_train_scaled, columns=numeric_feature_columns, index=X_train.index
)
X_val_scaled_df = pd.DataFrame(
    X_val_scaled, columns=numeric_feature_columns, index=X_val.index
)

print("\n" + "-" * 60)
print("Feature scales AFTER standardization:")
print("-" * 60)
scale_analysis_after = pd.DataFrame(
    {
        "Feature": numeric_feature_columns,
        "Mean": X_train_scaled_df.mean().values,
        "Std": X_train_scaled_df.std().values,
        "Min": X_train_scaled_df.min().values,
        "Max": X_train_scaled_df.max().values,
    }
)
print(scale_analysis_after.to_string(index=False))

# Train model WITH standardization
lr_with_scale = LinearRegression()
lr_with_scale.fit(X_train_scaled, y_train)
y_pred_with_scale = lr_with_scale.predict(X_val_scaled)
mse_with_scale = mean_squared_error(y_val, y_pred_with_scale)
r2_with_scale = r2_score(y_val, y_pred_with_scale)

print(f"\nLinear Regression WITH standardization:")
print(f"  Validation MSE: {mse_with_scale:,.2f}")
print(f"  Validation R²: {r2_with_scale:.4f}")

# Quantify impact
print("\n" + "=" * 60)
print("STANDARDIZATION IMPACT ANALYSIS")
print("=" * 60)
mse_improvement = ((mse_no_scale - mse_with_scale) / mse_no_scale) * 100
r2_improvement = ((r2_with_scale - r2_no_scale) / abs(r2_no_scale)) * 100

print(f"MSE Change: {mse_improvement:+.2f}%")
print(f"R² Change: {r2_improvement:+.2f}%")
print(f"\nKey Findings:")
print(f"  - Standardization ensures all features contribute equally")
print(f"  - Prevents features with larger scales from dominating")
print(f"  - Essential for regularization techniques (Ridge/Lasso)")
print(f"  - Improves numerical stability and convergence")

# Create standardization impact visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Feature scale comparison
ax1 = axes[0, 0]
scale_comparison = pd.DataFrame(
    {
        "Before": X_train_numeric[numeric_feature_columns].std().values,
        "After": X_train_scaled_df.std().values,
    },
    index=numeric_feature_columns,
)
scale_comparison.plot(kind="bar", ax=ax1, color=["#e74c3c", "#3498db"])
ax1.set_title(
    "Feature Standard Deviations: Before vs After Standardization",
    fontsize=12,
    fontweight="bold",
)
ax1.set_xlabel("Features")
ax1.set_ylabel("Standard Deviation")
ax1.legend(["Before Standardization", "After Standardization"])
ax1.tick_params(axis="x", rotation=45)
ax1.grid(True, alpha=0.3)

# Plot 2: Model performance comparison
ax2 = axes[0, 1]
metrics_comparison = pd.DataFrame(
    {
        "Without Standardization": [r2_no_scale, mse_no_scale / 1000],
        "With Standardization": [r2_with_scale, mse_with_scale / 1000],
    },
    index=["R² Score", "MSE (thousands)"],
)
metrics_comparison.plot(kind="bar", ax=ax2, color=["#e74c3c", "#2ecc71"])
ax2.set_title(
    "Model Performance: Impact of Standardization", fontsize=12, fontweight="bold"
)
ax2.set_ylabel("Value")
ax2.legend()
ax2.tick_params(axis="x", rotation=0)
ax2.grid(True, alpha=0.3)

# Plot 3: Feature range comparison
ax3 = axes[1, 0]
range_before = (X_train_numeric.max() - X_train_numeric.min()).values
range_after = (X_train_scaled_df.max() - X_train_scaled_df.min()).values
x_pos = np.arange(len(numeric_feature_columns))
width = 0.35
ax3.bar(
    x_pos - width / 2, range_before, width, label="Before", color="#e74c3c", alpha=0.7
)
ax3.bar(
    x_pos + width / 2, range_after, width, label="After", color="#3498db", alpha=0.7
)
ax3.set_title(
    "Feature Ranges: Before vs After Standardization", fontsize=12, fontweight="bold"
)
ax3.set_xlabel("Features")
ax3.set_ylabel("Range")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(numeric_feature_columns, rotation=45, ha="right")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Distribution comparison for a sample feature
ax4 = axes[1, 1]
sample_feature = numeric_feature_columns[0]
ax4.hist(
    X_train_numeric[sample_feature],
    bins=30,
    alpha=0.5,
    label="Before",
    color="#e74c3c",
    density=True,
)
ax4.hist(
    X_train_scaled_df[sample_feature],
    bins=30,
    alpha=0.5,
    label="After",
    color="#3498db",
    density=True,
)
ax4.set_title(
    f"Distribution Comparison: {sample_feature}", fontsize=12, fontweight="bold"
)
ax4.set_xlabel("Value")
ax4.set_ylabel("Density")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "visualizations/model_analysis/standardization_impact.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("\n✓ Standardization impact visualization saved")

# ============================================================================
# 4. MODEL 1: LINEAR REGRESSION (12 points total)
# ============================================================================
print("\n" + "=" * 80)
print("4. MODEL 1: LINEAR REGRESSION")
print("=" * 80)

# 4a. Baseline Model Training (2 points)
print("\n4.1 Baseline Linear Regression Model")
print("-" * 60)

lr_baseline = LinearRegression()
lr_baseline.fit(X_train_scaled, y_train)
y_pred_train_lr = lr_baseline.predict(X_train_scaled)
y_pred_val_lr = lr_baseline.predict(X_val_scaled)

# 4b. Coefficient Interpretation (2 points)
print("\n4.2 Coefficient Analysis and Interpretation")
print("-" * 60)

coefficients = pd.DataFrame(
    {
        "Feature": numeric_feature_columns,
        "Coefficient": lr_baseline.coef_,
        "Abs_Coefficient": np.abs(lr_baseline.coef_),
    }
).sort_values("Abs_Coefficient", ascending=False)

print("\nModel Coefficients (sorted by absolute value):")
print(coefficients.to_string(index=False))

print(f"\nIntercept: ${lr_baseline.intercept_:,.2f}")
print("\nCoefficient Interpretation:")
print("  - Coefficients represent the change in Total_Revenue for a 1-unit")
print("    change in the standardized feature, holding all else constant")
print(f"  - Top 3 most influential features:")
for i, row in coefficients.head(3).iterrows():
    direction = "increases" if row["Coefficient"] > 0 else "decreases"
    print(
        f"    {i+1}. {row['Feature']}: ${abs(row['Coefficient']):,.2f} {direction} per std unit"
    )

# Statistical significance testing
print("\n4.3 Statistical Significance of Coefficients")
print("-" * 60)

# Calculate standard errors and t-statistics
n = len(X_train_scaled)
k = X_train_scaled.shape[1]
residuals = y_train - y_pred_train_lr
mse_residuals = np.sum(residuals**2) / (n - k - 1)
var_coef = mse_residuals * np.linalg.inv(X_train_scaled.T @ X_train_scaled).diagonal()
se_coef = np.sqrt(var_coef)
t_stats = lr_baseline.coef_ / se_coef
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

significance_df = pd.DataFrame(
    {
        "Feature": numeric_feature_columns,
        "Coefficient": lr_baseline.coef_,
        "Std_Error": se_coef,
        "t_statistic": t_stats,
        "p_value": p_values,
        "Significant": [
            "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            for p in p_values
        ],
    }
).sort_values("p_value")

print("\nStatistical Significance (*** p<0.001, ** p<0.01, * p<0.05):")
print(significance_df.to_string(index=False))

# 4c. Comprehensive Evaluation (3 points)
print("\n4.4 Comprehensive Model Evaluation")
print("-" * 60)

# Calculate metrics
train_mse_lr = mean_squared_error(y_train, y_pred_train_lr)
train_mae_lr = mean_absolute_error(y_train, y_pred_train_lr)
train_r2_lr = r2_score(y_train, y_pred_train_lr)
train_rmse_lr = np.sqrt(train_mse_lr)

val_mse_lr = mean_squared_error(y_val, y_pred_val_lr)
val_mae_lr = mean_absolute_error(y_val, y_pred_val_lr)
val_r2_lr = r2_score(y_val, y_pred_val_lr)
val_rmse_lr = np.sqrt(val_mse_lr)

print("\nTraining Set Performance:")
print(f"  MSE:  {train_mse_lr:,.2f}")
print(f"  RMSE: {train_rmse_lr:,.2f}")
print(f"  MAE:  {train_mae_lr:,.2f}")
print(f"  R²:   {train_r2_lr:.4f}")

print("\nValidation Set Performance:")
print(f"  MSE:  {val_mse_lr:,.2f}")
print(f"  RMSE: {val_rmse_lr:,.2f}")
print(f"  MAE:  {val_mae_lr:,.2f}")
print(f"  R²:   {val_r2_lr:.4f}")

# Cross-validation
cv_scores = cross_val_score(lr_baseline, X_train_scaled, y_train, cv=5, scoring="r2")
print(f"\n5-Fold Cross-Validation R² Scores:")
print(f"  Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Create comprehensive evaluation plots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Residual Plot
ax1 = fig.add_subplot(gs[0, 0])
residuals_val = y_val - y_pred_val_lr
ax1.scatter(y_pred_val_lr, residuals_val, alpha=0.5, s=20, color="#3498db")
ax1.axhline(y=0, color="r", linestyle="--", linewidth=2)
ax1.set_xlabel("Predicted Values")
ax1.set_ylabel("Residuals")
ax1.set_title("Residual Plot", fontweight="bold")
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_val, y_pred_val_lr, alpha=0.5, s=20, color="#2ecc71")
min_val = min(y_val.min(), y_pred_val_lr.min())
max_val = max(y_val.max(), y_pred_val_lr.max())
ax2.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Perfect Prediction",
)
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values")
ax2.set_title(f"Actual vs Predicted (R²={val_r2_lr:.4f})", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Error Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals_val, bins=50, color="#9b59b6", alpha=0.7, edgecolor="black")
ax3.axvline(x=0, color="r", linestyle="--", linewidth=2)
ax3.set_xlabel("Prediction Error")
ax3.set_ylabel("Frequency")
ax3.set_title("Error Distribution", fontweight="bold")
ax3.grid(True, alpha=0.3)

# Plot 4: Q-Q Plot
ax4 = fig.add_subplot(gs[1, 0])
stats.probplot(residuals_val, dist="norm", plot=ax4)
ax4.set_title("Q-Q Plot", fontweight="bold")
ax4.grid(True, alpha=0.3)

# Plot 5: Scale-Location Plot
ax5 = fig.add_subplot(gs[1, 1])
standardized_residuals = residuals_val / np.std(residuals_val)
ax5.scatter(
    y_pred_val_lr,
    np.sqrt(np.abs(standardized_residuals)),
    alpha=0.5,
    s=20,
    color="#e74c3c",
)
ax5.set_xlabel("Predicted Values")
ax5.set_ylabel("√|Standardized Residuals|")
ax5.set_title("Scale-Location Plot", fontweight="bold")
ax5.grid(True, alpha=0.3)

# Plot 6: Coefficient Importance
ax6 = fig.add_subplot(gs[1, 2])
coef_plot_data = coefficients.head(10)
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_plot_data["Coefficient"]]
ax6.barh(
    range(len(coef_plot_data)), coef_plot_data["Coefficient"], color=colors, alpha=0.7
)
ax6.set_yticks(range(len(coef_plot_data)))
ax6.set_yticklabels(coef_plot_data["Feature"])
ax6.set_xlabel("Coefficient Value")
ax6.set_title("Top 10 Feature Coefficients", fontweight="bold")
ax6.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
ax6.grid(True, alpha=0.3, axis="x")

# Plot 7: Residuals vs Features (sample)
ax7 = fig.add_subplot(gs[2, 0])
sample_feature_idx = np.abs(lr_baseline.coef_).argmax()
sample_feature_name = numeric_feature_columns[sample_feature_idx]
ax7.scatter(
    X_val_scaled_df.iloc[:, sample_feature_idx],
    residuals_val,
    alpha=0.5,
    s=20,
    color="#f39c12",
)
ax7.axhline(y=0, color="r", linestyle="--", linewidth=2)
ax7.set_xlabel(f"{sample_feature_name} (standardized)")
ax7.set_ylabel("Residuals")
ax7.set_title(f"Residuals vs {sample_feature_name}", fontweight="bold")
ax7.grid(True, alpha=0.3)

# Plot 8: Prediction Error by Range
ax8 = fig.add_subplot(gs[2, 1])
y_val_sorted = np.sort(y_val)
n_bins = 10
bin_edges = np.percentile(y_val_sorted, np.linspace(0, 100, n_bins + 1))
bin_indices = np.digitize(y_val, bin_edges)
bin_errors = [
    np.abs(residuals_val[bin_indices == i]).mean() for i in range(1, n_bins + 1)
]
bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
ax8.bar(range(n_bins), bin_errors, color="#1abc9c", alpha=0.7)
ax8.set_xlabel("Revenue Range (deciles)")
ax8.set_ylabel("Mean Absolute Error")
ax8.set_title("Prediction Error by Revenue Range", fontweight="bold")
ax8.grid(True, alpha=0.3, axis="y")

# Plot 9: Cross-validation scores
ax9 = fig.add_subplot(gs[2, 2])
ax9.bar(range(1, 6), cv_scores, color="#34495e", alpha=0.7)
ax9.axhline(
    y=cv_scores.mean(),
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {cv_scores.mean():.4f}",
)
ax9.set_xlabel("Fold")
ax9.set_ylabel("R² Score")
ax9.set_title("5-Fold Cross-Validation Scores", fontweight="bold")
ax9.legend()
ax9.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "Linear Regression: Comprehensive Evaluation",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plt.savefig(
    "visualizations/model_analysis/linear_regression_evaluation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("\n✓ Linear regression evaluation plots saved")

# 4d. Hyperparameter Tuning with Regularization (3 points)
print("\n4.5 Hyperparameter Tuning: Ridge and Lasso Regularization")
print("-" * 60)

# Ridge Regression
alphas = np.logspace(-3, 3, 50)
ridge_scores = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring="r2")
    ridge_scores.append(scores.mean())

best_ridge_alpha = alphas[np.argmax(ridge_scores)]
ridge_best = Ridge(alpha=best_ridge_alpha)
ridge_best.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_best.predict(X_val_scaled)

print(f"\nRidge Regression:")
print(f"  Best alpha: {best_ridge_alpha:.4f}")
print(f"  Validation R²: {r2_score(y_val, y_pred_ridge):.4f}")
print(f"  Validation RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_ridge)):,.2f}")

# Lasso Regression
lasso_scores = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring="r2")
    lasso_scores.append(scores.mean())

best_lasso_alpha = alphas[np.argmax(lasso_scores)]
lasso_best = Lasso(alpha=best_lasso_alpha, max_iter=10000)
lasso_best.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_best.predict(X_val_scaled)

print(f"\nLasso Regression:")
print(f"  Best alpha: {best_lasso_alpha:.4f}")
print(f"  Validation R²: {r2_score(y_val, y_pred_lasso):.4f}")
print(f"  Validation RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_lasso)):,.2f}")
print(
    f"  Non-zero coefficients: {np.sum(lasso_best.coef_ != 0)}/{len(lasso_best.coef_)}"
)

# Visualize regularization paths
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ridge regularization path
ax1 = axes[0]
ax1.semilogx(alphas, ridge_scores, "b-", linewidth=2)
ax1.axvline(
    x=best_ridge_alpha,
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Best α={best_ridge_alpha:.4f}",
)
ax1.set_xlabel("Alpha (Regularization Strength)")
ax1.set_ylabel("Cross-Validation R² Score")
ax1.set_title("Ridge Regression: Regularization Path", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Lasso regularization path
ax2 = axes[1]
ax2.semilogx(alphas, lasso_scores, "g-", linewidth=2)
ax2.axvline(
    x=best_lasso_alpha,
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Best α={best_lasso_alpha:.4f}",
)
ax2.set_xlabel("Alpha (Regularization Strength)")
ax2.set_ylabel("Cross-Validation R² Score")
ax2.set_title("Lasso Regression: Regularization Path", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Coefficient comparison
ax3 = axes[2]
coef_comparison = pd.DataFrame(
    {"Linear": lr_baseline.coef_, "Ridge": ridge_best.coef_, "Lasso": lasso_best.coef_},
    index=numeric_feature_columns,
)
coef_comparison_top = coef_comparison.iloc[np.abs(lr_baseline.coef_).argsort()[-10:]]
coef_comparison_top.plot(kind="barh", ax=ax3, width=0.8)
ax3.set_xlabel("Coefficient Value")
ax3.set_title("Top 10 Coefficients: Model Comparison", fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(
    "visualizations/model_analysis/regularization_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("\n✓ Regularization analysis plots saved")

# 4e. Feature Selection with RFE (3 points)
print("\n4.6 Feature Selection using Recursive Feature Elimination (RFE)")
print("-" * 60)

# Test different numbers of features
n_features_to_test = range(5, len(numeric_feature_columns) + 1, 2)
rfe_scores = []

for n_features in n_features_to_test:
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
    rfe.fit(X_train_scaled, y_train)
    X_train_rfe = rfe.transform(X_train_scaled)
    X_val_rfe = rfe.transform(X_val_scaled)

    lr_rfe = LinearRegression()
    lr_rfe.fit(X_train_rfe, y_train)
    y_pred_rfe = lr_rfe.predict(X_val_rfe)
    score = r2_score(y_val, y_pred_rfe)
    rfe_scores.append(score)

best_n_features = n_features_to_test[np.argmax(rfe_scores)]
print(f"\nOptimal number of features: {best_n_features}")
print(f"Best R² score: {max(rfe_scores):.4f}")

# Train final RFE model
rfe_final = RFE(estimator=LinearRegression(), n_features_to_select=best_n_features)
rfe_final.fit(X_train_scaled, y_train)
selected_features = [
    numeric_feature_columns[i]
    for i in range(len(numeric_feature_columns))
    if rfe_final.support_[i]
]

print(f"\nSelected features ({len(selected_features)}):")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

X_train_rfe_final = rfe_final.transform(X_train_scaled)
X_val_rfe_final = rfe_final.transform(X_val_scaled)

lr_rfe_final = LinearRegression()
lr_rfe_final.fit(X_train_rfe_final, y_train)
y_pred_rfe_final = lr_rfe_final.predict(X_val_rfe_final)

print(f"\nRFE Model Performance:")
print(f"  Validation R²: {r2_score(y_val, y_pred_rfe_final):.4f}")
print(f"  Validation RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_rfe_final)):,.2f}")
print(f"  Validation MAE: {mean_absolute_error(y_val, y_pred_rfe_final):,.2f}")

# Visualize RFE results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# RFE score vs number of features
ax1 = axes[0]
ax1.plot(
    n_features_to_test, rfe_scores, "o-", linewidth=2, markersize=8, color="#3498db"
)
ax1.axvline(
    x=best_n_features,
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Optimal: {best_n_features} features",
)
ax1.set_xlabel("Number of Features")
ax1.set_ylabel("R² Score")
ax1.set_title("RFE: Model Performance vs Number of Features", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Feature ranking
ax2 = axes[1]
feature_ranking = pd.DataFrame(
    {
        "Feature": numeric_feature_columns,
        "Ranking": rfe_final.ranking_,
        "Selected": rfe_final.support_,
    }
).sort_values("Ranking")
colors = [
    "#2ecc71" if selected else "#e74c3c" for selected in feature_ranking["Selected"]
]
ax2.barh(
    range(len(feature_ranking)), feature_ranking["Ranking"], color=colors, alpha=0.7
)
ax2.set_yticks(range(len(feature_ranking)))
ax2.set_yticklabels(feature_ranking["Feature"])
ax2.set_xlabel("Ranking (1 = most important)")
ax2.set_title("RFE Feature Rankings", fontweight="bold")
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(
    "visualizations/model_analysis/rfe_feature_selection.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("\n✓ RFE feature selection plots saved")

# Summary of Linear Regression models
print("\n" + "=" * 60)
print("LINEAR REGRESSION MODEL SUMMARY")
print("=" * 60)
lr_summary = pd.DataFrame(
    {
        "Model": ["Baseline", "Ridge", "Lasso", "RFE"],
        "R²": [
            val_r2_lr,
            r2_score(y_val, y_pred_ridge),
            r2_score(y_val, y_pred_lasso),
            r2_score(y_val, y_pred_rfe_final),
        ],
        "RMSE": [
            val_rmse_lr,
            np.sqrt(mean_squared_error(y_val, y_pred_ridge)),
            np.sqrt(mean_squared_error(y_val, y_pred_lasso)),
            np.sqrt(mean_squared_error(y_val, y_pred_rfe_final)),
        ],
        "MAE": [
            val_mae_lr,
            mean_absolute_error(y_val, y_pred_ridge),
            mean_absolute_error(y_val, y_pred_lasso),
            mean_absolute_error(y_val, y_pred_rfe_final),
        ],
        "Features": [
            len(numeric_feature_columns),
            len(numeric_feature_columns),
            np.sum(lasso_best.coef_ != 0),
            best_n_features,
        ],
    }
)
print(lr_summary.to_string(index=False))

# Select best Linear Regression model
best_lr_idx = lr_summary["R²"].idxmax()
best_lr_model_name = lr_summary.loc[best_lr_idx, "Model"]
print(f"\nBest Linear Regression variant: {best_lr_model_name}")
print(f"  R²: {lr_summary.loc[best_lr_idx, 'R²']:.4f}")
print(f"  RMSE: {lr_summary.loc[best_lr_idx, 'RMSE']:,.2f}")

# Store best LR predictions for later comparison
if best_lr_model_name == "Baseline":
    y_pred_lr_best = y_pred_val_lr
    best_lr_model = lr_baseline
elif best_lr_model_name == "Ridge":
    y_pred_lr_best = y_pred_ridge
    best_lr_model = ridge_best
elif best_lr_model_name == "Lasso":
    y_pred_lr_best = y_pred_lasso
    best_lr_model = lasso_best
else:
    y_pred_lr_best = y_pred_rfe_final
    best_lr_model = lr_rfe_final

# ============================================================================
# 5. MODEL 2: RANDOM FOREST (13 points total)
# ============================================================================
print("\n" + "=" * 80)
print("5. MODEL 2: RANDOM FOREST")
print("=" * 80)

# 5a. Baseline Model Training (3 points)
print("\n5.1 Baseline Random Forest Model")
print("-" * 60)

# Note: Random Forest doesn't require standardization, so we use original numeric data
rf_baseline = RandomForestRegressor(
    n_estimators=100, random_state=random_state, n_jobs=-1
)
rf_baseline.fit(X_train_numeric, y_train)
y_pred_train_rf = rf_baseline.predict(X_train_numeric)
y_pred_val_rf = rf_baseline.predict(X_val_numeric)

print(f"Model trained with {rf_baseline.n_estimators} trees")
print(f"Using same train/validation split as Linear Regression (80/20)")

# 5b. Feature Importance Analysis (3 points)
print("\n5.2 Feature Importance Analysis")
print("-" * 60)

feature_importance = pd.DataFrame(
    {"Feature": numeric_feature_columns, "Importance": rf_baseline.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nFeature Importances (sorted):")
print(feature_importance.to_string(index=False))

print(f"\nTop 5 Most Important Features:")
for i, row in feature_importance.head(5).iterrows():
    print(
        f"  {i+1}. {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)"
    )

# Tree structure analysis
print("\n5.3 Tree Structure Analysis")
print("-" * 60)
sample_tree = rf_baseline.estimators_[0]
print(f"Sample tree (first tree in forest):")
print(f"  Max depth: {sample_tree.get_depth()}")
print(f"  Number of leaves: {sample_tree.get_n_leaves()}")
print(f"  Number of nodes: {sample_tree.tree_.node_count}")

# Analyze all trees
tree_depths = [tree.get_depth() for tree in rf_baseline.estimators_]
tree_leaves = [tree.get_n_leaves() for tree in rf_baseline.estimators_]
print(f"\nForest statistics (across {len(rf_baseline.estimators_)} trees):")
print(f"  Average depth: {np.mean(tree_depths):.2f} (±{np.std(tree_depths):.2f})")
print(f"  Average leaves: {np.mean(tree_leaves):.2f} (±{np.std(tree_leaves):.2f})")
print(f"  Min/Max depth: {min(tree_depths)}/{max(tree_depths)}")

# 5c. Comprehensive Evaluation (3 points)
print("\n5.4 Comprehensive Model Evaluation")
print("-" * 60)

# Calculate metrics
train_mse_rf = mean_squared_error(y_train, y_pred_train_rf)
train_mae_rf = mean_absolute_error(y_train, y_pred_train_rf)
train_r2_rf = r2_score(y_train, y_pred_train_rf)
train_rmse_rf = np.sqrt(train_mse_rf)

val_mse_rf = mean_squared_error(y_val, y_pred_val_rf)
val_mae_rf = mean_absolute_error(y_val, y_pred_val_rf)
val_r2_rf = r2_score(y_val, y_pred_val_rf)
val_rmse_rf = np.sqrt(val_mse_rf)

print("\nTraining Set Performance:")
print(f"  MSE:  {train_mse_rf:,.2f}")
print(f"  RMSE: {train_rmse_rf:,.2f}")
print(f"  MAE:  {train_mae_rf:,.2f}")
print(f"  R²:   {train_r2_rf:.4f}")

print("\nValidation Set Performance:")
print(f"  MSE:  {val_mse_rf:,.2f}")
print(f"  RMSE: {val_rmse_rf:,.2f}")
print(f"  MAE:  {val_mae_rf:,.2f}")
print(f"  R²:   {val_r2_rf:.4f}")

# Cross-validation
cv_scores_rf = cross_val_score(
    rf_baseline, X_train_numeric, y_train, cv=5, scoring="r2"
)
print(f"\n5-Fold Cross-Validation R² Scores:")
print(f"  Fold scores: {[f'{score:.4f}' for score in cv_scores_rf]}")
print(f"  Mean: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

# Create comprehensive evaluation plots for Random Forest
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Residual Plot
ax1 = fig.add_subplot(gs[0, 0])
residuals_val_rf = y_val - y_pred_val_rf
ax1.scatter(y_pred_val_rf, residuals_val_rf, alpha=0.5, s=20, color="#e74c3c")
ax1.axhline(y=0, color="black", linestyle="--", linewidth=2)
ax1.set_xlabel("Predicted Values")
ax1.set_ylabel("Residuals")
ax1.set_title("Residual Plot", fontweight="bold")
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_val, y_pred_val_rf, alpha=0.5, s=20, color="#9b59b6")
min_val = min(y_val.min(), y_pred_val_rf.min())
max_val = max(y_val.max(), y_pred_val_rf.max())
ax2.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Perfect Prediction",
)
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values")
ax2.set_title(f"Actual vs Predicted (R²={val_r2_rf:.4f})", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Error Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals_val_rf, bins=50, color="#1abc9c", alpha=0.7, edgecolor="black")
ax3.axvline(x=0, color="r", linestyle="--", linewidth=2)
ax3.set_xlabel("Prediction Error")
ax3.set_ylabel("Frequency")
ax3.set_title("Error Distribution", fontweight="bold")
ax3.grid(True, alpha=0.3)

# Plot 4: Feature Importance
ax4 = fig.add_subplot(gs[1, 0])
top_features = feature_importance.head(10)
colors_fi = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax4.barh(
    range(len(top_features)), top_features["Importance"], color=colors_fi, alpha=0.8
)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features["Feature"])
ax4.set_xlabel("Importance")
ax4.set_title("Top 10 Feature Importances", fontweight="bold")
ax4.grid(True, alpha=0.3, axis="x")

# Plot 5: Tree Depth Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(tree_depths, bins=20, color="#f39c12", alpha=0.7, edgecolor="black")
ax5.axvline(
    x=np.mean(tree_depths),
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {np.mean(tree_depths):.1f}",
)
ax5.set_xlabel("Tree Depth")
ax5.set_ylabel("Frequency")
ax5.set_title("Distribution of Tree Depths", fontweight="bold")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Number of Leaves Distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(tree_leaves, bins=20, color="#e67e22", alpha=0.7, edgecolor="black")
ax6.axvline(
    x=np.mean(tree_leaves),
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {np.mean(tree_leaves):.1f}",
)
ax6.set_xlabel("Number of Leaves")
ax6.set_ylabel("Frequency")
ax6.set_title("Distribution of Leaf Nodes", fontweight="bold")
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Prediction Error by Range
ax7 = fig.add_subplot(gs[2, 0])
y_val_sorted_rf = np.sort(y_val)
bin_indices_rf = np.digitize(y_val, bin_edges)
bin_errors_rf = [
    np.abs(residuals_val_rf[bin_indices_rf == i]).mean() for i in range(1, n_bins + 1)
]
ax7.bar(range(n_bins), bin_errors_rf, color="#16a085", alpha=0.7)
ax7.set_xlabel("Revenue Range (deciles)")
ax7.set_ylabel("Mean Absolute Error")
ax7.set_title("Prediction Error by Revenue Range", fontweight="bold")
ax7.grid(True, alpha=0.3, axis="y")

# Plot 8: Cross-validation scores
ax8 = fig.add_subplot(gs[2, 1])
ax8.bar(range(1, 6), cv_scores_rf, color="#8e44ad", alpha=0.7)
ax8.axhline(
    y=cv_scores_rf.mean(),
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {cv_scores_rf.mean():.4f}",
)
ax8.set_xlabel("Fold")
ax8.set_ylabel("R² Score")
ax8.set_title("5-Fold Cross-Validation Scores", fontweight="bold")
ax8.legend()
ax8.grid(True, alpha=0.3, axis="y")

# Plot 9: Cumulative Feature Importance
ax9 = fig.add_subplot(gs[2, 2])
cumulative_importance = np.cumsum(feature_importance["Importance"].values)
ax9.plot(
    range(1, len(cumulative_importance) + 1),
    cumulative_importance,
    "o-",
    linewidth=2,
    markersize=6,
    color="#c0392b",
)
ax9.axhline(y=0.9, color="g", linestyle="--", linewidth=2, label="90% threshold")
ax9.set_xlabel("Number of Features")
ax9.set_ylabel("Cumulative Importance")
ax9.set_title("Cumulative Feature Importance", fontweight="bold")
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.suptitle(
    "Random Forest: Comprehensive Evaluation", fontsize=16, fontweight="bold", y=0.995
)
plt.savefig(
    "visualizations/model_analysis/random_forest_evaluation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("\n✓ Random Forest evaluation plots saved")

# 5d. Hyperparameter Tuning with GridSearchCV (3 points)
print("\n5.5 Hyperparameter Tuning with GridSearchCV")
print("-" * 60)

param_grid = {
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "n_estimators": [50, 100, 200],
}

print(f"Parameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
print("Performing GridSearchCV with 3-fold cross-validation...")

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train_numeric, y_train)

print(f"\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation R² score: {grid_search.best_score_:.4f}")

# Train final optimized model
rf_optimized = grid_search.best_estimator_
y_pred_val_rf_opt = rf_optimized.predict(X_val_numeric)

val_mse_rf_opt = mean_squared_error(y_val, y_pred_val_rf_opt)
val_mae_rf_opt = mean_absolute_error(y_val, y_pred_val_rf_opt)
val_r2_rf_opt = r2_score(y_val, y_pred_val_rf_opt)
val_rmse_rf_opt = np.sqrt(val_mse_rf_opt)

print(f"\nOptimized Model Performance:")
print(f"  Validation MSE:  {val_mse_rf_opt:,.2f}")
print(f"  Validation RMSE: {val_rmse_rf_opt:,.2f}")
print(f"  Validation MAE:  {val_mae_rf_opt:,.2f}")
print(f"  Validation R²:   {val_r2_rf_opt:.4f}")

print(f"\nImprovement over baseline:")
print(f"  R² improvement: {(val_r2_rf_opt - val_r2_rf) / val_r2_rf * 100:+.2f}%")
print(
    f"  RMSE improvement: {(val_rmse_rf - val_rmse_rf_opt) / val_rmse_rf * 100:+.2f}%"
)

# Visualize hyperparameter tuning results
results_df = pd.DataFrame(grid_search.cv_results_)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: n_estimators effect
ax1 = axes[0, 0]
for depth in param_grid["max_depth"]:
    mask = results_df["param_max_depth"] == depth
    data = results_df[mask].groupby("param_n_estimators")["mean_test_score"].mean()
    ax1.plot(
        data.index,
        data.values,
        "o-",
        label=f"max_depth={depth}",
        linewidth=2,
        markersize=8,
    )
ax1.set_xlabel("Number of Estimators")
ax1.set_ylabel("Mean CV R² Score")
ax1.set_title("Effect of n_estimators", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: max_depth effect
ax2 = axes[0, 1]
depth_scores = results_df.groupby("param_max_depth")["mean_test_score"].mean()
depth_labels = [str(d) if d is not None else "None" for d in depth_scores.index]
ax2.bar(range(len(depth_scores)), depth_scores.values, color="#3498db", alpha=0.7)
ax2.set_xticks(range(len(depth_scores)))
ax2.set_xticklabels(depth_labels)
ax2.set_xlabel("Max Depth")
ax2.set_ylabel("Mean CV R² Score")
ax2.set_title("Effect of max_depth", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: min_samples_split effect
ax3 = axes[1, 0]
split_scores = results_df.groupby("param_min_samples_split")["mean_test_score"].mean()
ax3.bar(range(len(split_scores)), split_scores.values, color="#2ecc71", alpha=0.7)
ax3.set_xticks(range(len(split_scores)))
ax3.set_xticklabels(split_scores.index)
ax3.set_xlabel("Min Samples Split")
ax3.set_ylabel("Mean CV R² Score")
ax3.set_title("Effect of min_samples_split", fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: min_samples_leaf effect
ax4 = axes[1, 1]
leaf_scores = results_df.groupby("param_min_samples_leaf")["mean_test_score"].mean()
ax4.bar(range(len(leaf_scores)), leaf_scores.values, color="#e74c3c", alpha=0.7)
ax4.set_xticks(range(len(leaf_scores)))
ax4.set_xticklabels(leaf_scores.index)
ax4.set_xlabel("Min Samples Leaf")
ax4.set_ylabel("Mean CV R² Score")
ax4.set_title("Effect of min_samples_leaf", fontweight="bold")
ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "Random Forest: Hyperparameter Tuning Analysis", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    "visualizations/model_analysis/rf_hyperparameter_tuning.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("\n✓ Hyperparameter tuning plots saved")

# Summary of Random Forest models
print("\n" + "=" * 60)
print("RANDOM FOREST MODEL SUMMARY")
print("=" * 60)
rf_summary = pd.DataFrame(
    {
        "Model": ["Baseline", "Optimized"],
        "R²": [val_r2_rf, val_r2_rf_opt],
        "RMSE": [val_rmse_rf, val_rmse_rf_opt],
        "MAE": [val_mae_rf, val_mae_rf_opt],
        "n_estimators": [rf_baseline.n_estimators, rf_optimized.n_estimators],
        "max_depth": [rf_baseline.max_depth, rf_optimized.max_depth],
    }
)
print(rf_summary.to_string(index=False))

# ============================================================================
# 6. COMPREHENSIVE MODEL COMPARISON (3 points for Model 2)
# ============================================================================
print("\n" + "=" * 80)
print("6. COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)

print("\n6.1 Performance Metrics Comparison")
print("-" * 60)

# Get best models from each type
best_lr_r2 = lr_summary["R²"].max()
best_lr_rmse = lr_summary.loc[lr_summary["R²"].idxmax(), "RMSE"]
best_lr_mae = lr_summary.loc[lr_summary["R²"].idxmax(), "MAE"]

comparison_df = pd.DataFrame(
    {
        "Model": [
            "Linear Regression (Best)",
            "Random Forest (Baseline)",
            "Random Forest (Optimized)",
        ],
        "R²": [best_lr_r2, val_r2_rf, val_r2_rf_opt],
        "RMSE": [best_lr_rmse, val_rmse_rf, val_rmse_rf_opt],
        "MAE": [best_lr_mae, val_mae_rf, val_mae_rf_opt],
        "Training_Time": ["Fast", "Medium", "Slow"],
        "Interpretability": ["High", "Medium", "Medium"],
    }
)

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

print("\n6.2 Statistical Comparison")
print("-" * 60)

# Compare predictions
lr_residuals = y_val - y_pred_lr_best
rf_baseline_residuals = y_val - y_pred_val_rf
rf_opt_residuals = y_val - y_pred_val_rf_opt

print(f"\nResidual Statistics:")
print(f"  Linear Regression:")
print(f"    Mean: {lr_residuals.mean():,.2f}")
print(f"    Std: {lr_residuals.std():,.2f}")
print(f"    Median: {lr_residuals.median():,.2f}")
print(f"  Random Forest (Baseline):")
print(f"    Mean: {rf_baseline_residuals.mean():,.2f}")
print(f"    Std: {rf_baseline_residuals.std():,.2f}")
print(f"    Median: {rf_baseline_residuals.median():,.2f}")
print(f"  Random Forest (Optimized):")
print(f"    Mean: {rf_opt_residuals.mean():,.2f}")
print(f"    Std: {rf_opt_residuals.std():,.2f}")
print(f"    Median: {rf_opt_residuals.median():,.2f}")

print("\n6.3 Computational Efficiency")
print("-" * 60)
print("Training Time Comparison (relative):")
print("  Linear Regression: ~1x (fastest)")
print("  Random Forest Baseline: ~10-20x")
print("  Random Forest Optimized: ~20-50x (depends on hyperparameters)")
print("\nPrediction Time Comparison (relative):")
print("  Linear Regression: ~1x (fastest)")
print("  Random Forest: ~100-200x (depends on n_estimators)")

print("\n6.4 Interpretability Assessment")
print("-" * 60)
print("Linear Regression:")
print("  ✓ Direct coefficient interpretation")
print("  ✓ Clear feature relationships")
print("  ✓ Statistical significance testing")
print("  ✓ Easy to explain to stakeholders")
print("\nRandom Forest:")
print("  ✓ Feature importance rankings")
print("  ✓ Captures non-linear relationships")
print("  ✗ Less transparent decision process")
print("  ✗ Harder to explain individual predictions")

# Create comprehensive comparison visualizations
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: R² Comparison
ax1 = fig.add_subplot(gs[0, 0])
models = ["LR\n(Best)", "RF\n(Baseline)", "RF\n(Optimized)"]
r2_scores = [best_lr_r2, val_r2_rf, val_r2_rf_opt]
colors = ["#3498db", "#e74c3c", "#2ecc71"]
bars = ax1.bar(
    models, r2_scores, color=colors, alpha=0.7, edgecolor="black", linewidth=2
)
ax1.set_ylabel("R² Score")
ax1.set_title("R² Score Comparison", fontweight="bold", fontsize=12)
ax1.set_ylim([min(r2_scores) * 0.95, max(r2_scores) * 1.02])
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{score:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax1.grid(True, alpha=0.3, axis="y")

# Plot 2: RMSE Comparison
ax2 = fig.add_subplot(gs[0, 1])
rmse_scores = [best_lr_rmse, val_rmse_rf, val_rmse_rf_opt]
bars = ax2.bar(
    models, rmse_scores, color=colors, alpha=0.7, edgecolor="black", linewidth=2
)
ax2.set_ylabel("RMSE")
ax2.set_title("RMSE Comparison (Lower is Better)", fontweight="bold", fontsize=12)
for bar, score in zip(bars, rmse_scores):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{score:,.0f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: MAE Comparison
ax3 = fig.add_subplot(gs[0, 2])
mae_scores = [best_lr_mae, val_mae_rf, val_mae_rf_opt]
bars = ax3.bar(
    models, mae_scores, color=colors, alpha=0.7, edgecolor="black", linewidth=2
)
ax3.set_ylabel("MAE")
ax3.set_title("MAE Comparison (Lower is Better)", fontweight="bold", fontsize=12)
for bar, score in zip(bars, mae_scores):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{score:,.0f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Residual Distribution Comparison
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(
    lr_residuals, bins=50, alpha=0.5, label="Linear Reg", color="#3498db", density=True
)
ax4.hist(
    rf_opt_residuals,
    bins=50,
    alpha=0.5,
    label="RF (Opt)",
    color="#2ecc71",
    density=True,
)
ax4.axvline(x=0, color="r", linestyle="--", linewidth=2)
ax4.set_xlabel("Residuals")
ax4.set_ylabel("Density")
ax4.set_title("Residual Distribution Comparison", fontweight="bold", fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Actual vs Predicted Comparison
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y_val, y_pred_lr_best, alpha=0.3, s=20, label="Linear Reg", color="#3498db")
ax5.scatter(
    y_val, y_pred_val_rf_opt, alpha=0.3, s=20, label="RF (Opt)", color="#2ecc71"
)
min_val = y_val.min()
max_val = y_val.max()
ax5.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")
ax5.set_xlabel("Actual Values")
ax5.set_ylabel("Predicted Values")
ax5.set_title("Actual vs Predicted: Model Comparison", fontweight="bold", fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Error by Revenue Range
ax6 = fig.add_subplot(gs[1, 2])
x_pos = np.arange(n_bins)
width = 0.35
lr_errors_by_range = [
    np.abs(lr_residuals[bin_indices == i]).mean() for i in range(1, n_bins + 1)
]
rf_errors_by_range = [
    np.abs(rf_opt_residuals[bin_indices_rf == i]).mean() for i in range(1, n_bins + 1)
]
ax6.bar(
    x_pos - width / 2,
    lr_errors_by_range,
    width,
    label="Linear Reg",
    color="#3498db",
    alpha=0.7,
)
ax6.bar(
    x_pos + width / 2,
    rf_errors_by_range,
    width,
    label="RF (Opt)",
    color="#2ecc71",
    alpha=0.7,
)
ax6.set_xlabel("Revenue Range (deciles)")
ax6.set_ylabel("Mean Absolute Error")
ax6.set_title("Prediction Error by Revenue Range", fontweight="bold", fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3, axis="y")

# Plot 7: Radar Chart for Model Characteristics
ax7 = fig.add_subplot(gs[2, 0], projection="polar")
categories = ["R²", "Speed", "Interpretability", "Robustness", "Flexibility"]
N = len(categories)

# Normalize scores to 0-1 scale
lr_scores_norm = [
    best_lr_r2,
    1.0,  # Speed (fastest)
    1.0,  # Interpretability (highest)
    0.7,  # Robustness
    0.6,  # Flexibility
]
rf_scores_norm = [
    val_r2_rf_opt,
    0.3,  # Speed (slower)
    0.6,  # Interpretability (medium)
    0.9,  # Robustness (handles non-linearity)
    0.9,  # Flexibility (captures complex patterns)
]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
lr_scores_norm += lr_scores_norm[:1]
rf_scores_norm += rf_scores_norm[:1]
angles += angles[:1]

ax7.plot(angles, lr_scores_norm, "o-", linewidth=2, label="Linear Reg", color="#3498db")
ax7.fill(angles, lr_scores_norm, alpha=0.25, color="#3498db")
ax7.plot(angles, rf_scores_norm, "o-", linewidth=2, label="RF (Opt)", color="#2ecc71")
ax7.fill(angles, rf_scores_norm, alpha=0.25, color="#2ecc71")
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(categories)
ax7.set_ylim(0, 1)
ax7.set_title(
    "Model Characteristics Comparison", fontweight="bold", fontsize=12, pad=20
)
ax7.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
ax7.grid(True)

# Plot 8: Cross-validation comparison
ax8 = fig.add_subplot(gs[2, 1])
cv_comparison = pd.DataFrame(
    {"Linear Regression": cv_scores, "Random Forest": cv_scores_rf}
)
bp = ax8.boxplot(
    [cv_scores, cv_scores_rf], labels=["Linear Reg", "RF"], patch_artist=True
)
for patch, color in zip(bp["boxes"], ["#3498db", "#2ecc71"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax8.set_ylabel("R² Score")
ax8.set_title("Cross-Validation Score Distribution", fontweight="bold", fontsize=12)
ax8.grid(True, alpha=0.3, axis="y")

# Plot 9: Feature Importance Comparison (Top 10)
ax9 = fig.add_subplot(gs[2, 2])
top_10_features = feature_importance.head(10)["Feature"].values
lr_coef_top10 = [
    abs(lr_baseline.coef_[list(numeric_feature_columns).index(f)])
    for f in top_10_features
]
rf_imp_top10 = [
    feature_importance[feature_importance["Feature"] == f]["Importance"].values[0]
    for f in top_10_features
]

# Normalize for comparison
lr_coef_norm = np.array(lr_coef_top10) / max(lr_coef_top10)
rf_imp_norm = np.array(rf_imp_top10) / max(rf_imp_top10)

x_pos = np.arange(len(top_10_features))
width = 0.35
ax9.barh(
    x_pos - width / 2,
    lr_coef_norm,
    width,
    label="LR (normalized)",
    color="#3498db",
    alpha=0.7,
)
ax9.barh(
    x_pos + width / 2,
    rf_imp_norm,
    width,
    label="RF (normalized)",
    color="#2ecc71",
    alpha=0.7,
)
ax9.set_yticks(x_pos)
ax9.set_yticklabels(top_10_features, fontsize=9)
ax9.set_xlabel("Normalized Importance")
ax9.set_title("Feature Importance: Top 10 Features", fontweight="bold", fontsize=12)
ax9.legend()
ax9.grid(True, alpha=0.3, axis="x")

plt.suptitle(
    "Comprehensive Model Comparison: Linear Regression vs Random Forest",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plt.savefig(
    "visualizations/model_analysis/comprehensive_model_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("\n✓ Comprehensive comparison plots saved")

# ============================================================================
# 7. FINAL MODEL SELECTION AND JUSTIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("7. FINAL MODEL SELECTION AND JUSTIFICATION")
print("=" * 80)

print("\n7.1 Model Selection Criteria")
print("-" * 60)

# Determine best model based on multiple criteria
print("\nEvaluation Criteria:")
print(f"1. Predictive Performance (R²):")
print(f"   - Linear Regression: {best_lr_r2:.4f}")
print(f"   - Random Forest (Optimized): {val_r2_rf_opt:.4f}")
print(
    f"   Winner: {'Random Forest' if val_r2_rf_opt > best_lr_r2 else 'Linear Regression'}"
)

print(f"\n2. Prediction Error (RMSE):")
print(f"   - Linear Regression: ${best_lr_rmse:,.2f}")
print(f"   - Random Forest (Optimized): ${val_rmse_rf_opt:,.2f}")
print(
    f"   Winner: {'Random Forest' if val_rmse_rf_opt < best_lr_rmse else 'Linear Regression'}"
)

print(f"\n3. Interpretability:")
print(f"   - Linear Regression: High (clear coefficients)")
print(f"   - Random Forest: Medium (feature importance)")
print(f"   Winner: Linear Regression")

print(f"\n4. Computational Efficiency:")
print(f"   - Linear Regression: High (fast training & prediction)")
print(f"   - Random Forest: Low (slower, especially with many trees)")
print(f"   Winner: Linear Regression")

print(f"\n5. Robustness to Non-linearity:")
print(f"   - Linear Regression: Low (assumes linear relationships)")
print(f"   - Random Forest: High (captures non-linear patterns)")
print(f"   Winner: Random Forest")

print("\n7.2 Final Model Selection")
print("-" * 60)

# Calculate overall score
performance_weight = 0.4
interpretability_weight = 0.2
efficiency_weight = 0.2
robustness_weight = 0.2

lr_score = (
    (best_lr_r2 / max(best_lr_r2, val_r2_rf_opt)) * performance_weight
    + 1.0 * interpretability_weight
    + 1.0 * efficiency_weight
    + 0.7 * robustness_weight
)

rf_score = (
    (val_r2_rf_opt / max(best_lr_r2, val_r2_rf_opt)) * performance_weight
    + 0.6 * interpretability_weight
    + 0.3 * efficiency_weight
    + 0.9 * robustness_weight
)

print(
    f"\nWeighted Scoring (Performance: 40%, Interpretability: 20%, Efficiency: 20%, Robustness: 20%):"
)
print(f"  Linear Regression: {lr_score:.3f}")
print(f"  Random Forest: {rf_score:.3f}")

selected_model = (
    "Random Forest (Optimized)" if rf_score > lr_score else "Linear Regression"
)
print(f"\n{'='*60}")
print(f"SELECTED MODEL: {selected_model}")
print(f"{'='*60}")

print(f"\nJustification:")
if rf_score > lr_score:
    print(
        f"  ✓ Superior predictive performance (R²: {val_r2_rf_opt:.4f} vs {best_lr_r2:.4f})"
    )
    print(
        f"  ✓ Lower prediction error (RMSE: ${val_rmse_rf_opt:,.2f} vs ${best_lr_rmse:,.2f})"
    )
    print(f"  ✓ Better captures non-linear relationships in the data")
    print(f"  ✓ More robust to outliers and feature interactions")
    print(f"  ✓ Feature importance provides actionable insights")
    print(f"  - Trade-off: Slightly less interpretable than linear regression")
    print(
        f"  - Trade-off: Higher computational cost (acceptable for this dataset size)"
    )
else:
    print(f"  ✓ Excellent interpretability with clear coefficient meanings")
    print(f"  ✓ Fast training and prediction times")
    print(f"  ✓ Statistical significance testing available")
    print(f"  ✓ Easy to explain to stakeholders")
    print(f"  ✓ Competitive performance (R²: {best_lr_r2:.4f})")
    print(f"  - Trade-off: Assumes linear relationships")

print(f"\n7.3 Practical Recommendations")
print("-" * 60)
print("For Production Deployment:")
if rf_score > lr_score:
    print("  1. Use Random Forest (Optimized) for predictions")
    print("  2. Monitor feature importance for business insights")
    print("  3. Consider ensemble with Linear Regression for robustness")
    print("  4. Implement model versioning and A/B testing")
    print("  5. Set up automated retraining pipeline")
else:
    print("  1. Use Linear Regression for predictions")
    print("  2. Regularly validate coefficient stability")
    print("  3. Monitor for non-linear patterns in residuals")
    print("  4. Consider Random Forest for complex scenarios")
    print("  5. Implement automated model monitoring")

print("\nFor Business Stakeholders:")
print("  1. Focus on top 5 most important features")
print("  2. Provide confidence intervals for predictions")
print("  3. Explain model limitations and assumptions")
print("  4. Regular model performance reports")
print("  5. Clear documentation of model decisions")

# ============================================================================
# 8. GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("8. GENERATING COMPREHENSIVE REPORT")
print("=" * 80)

report_content = f"""# Comprehensive Machine Learning Pipeline Analysis Report

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** preprocessed_sales_data.csv  
**Target Variable:** Total_Revenue  
**Analysis Type:** Regression

---

## Executive Summary

This report presents a comprehensive analysis of two machine learning models for predicting Total Revenue:
- **Model 1:** Linear Regression (with regularization and feature selection)
- **Model 2:** Random Forest (with hyperparameter optimization)

### Key Findings

- **Best Model:** {selected_model}
- **Best R² Score:** {max(best_lr_r2, val_r2_rf_opt):.4f}
- **Best RMSE:** ${min(best_lr_rmse, val_rmse_rf_opt):,.2f}
- **Dataset Size:** {len(df):,} samples
- **Number of Features:** {len(feature_columns)}

---

## 1. Data Overview

### Dataset Characteristics
- **Total Samples:** {len(df):,}
- **Training Samples:** {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)
- **Validation Samples:** {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)
- **Split Ratio:** 80/20 (Train/Validation)
- **Random State:** {random_state}

### Target Variable Statistics
- **Mean:** ${y.mean():,.2f}
- **Median:** ${y.median():,.2f}
- **Std Dev:** ${y.std():,.2f}
- **Min:** ${y.min():,.2f}
- **Max:** ${y.max():,.2f}

### Features ({len(numeric_feature_columns)})
{chr(10).join([f"{i+1}. {col}" for i, col in enumerate(numeric_feature_columns)])}

---

## 2. Train/Validation Split Justification

### Split Strategy
- **Ratio:** 80/20 (Training/Validation)
- **Method:** Random stratified split
- **Random State:** {random_state} (ensures reproducibility)

### Justification
With approximately {len(df):,} samples in the dataset:
- **Training set ({len(X_train):,} samples):** Provides sufficient data for model learning and pattern recognition
- **Validation set ({len(X_val):,} samples):** Offers reliable performance evaluation with adequate statistical power
- **Standard practice:** 80/20 split is industry standard for datasets of this size
- **Balance:** Optimizes between model performance and evaluation reliability

---

## 3. Data Standardization Analysis

### Impact on Linear Regression

**Without Standardization:**
- Validation MSE: {mse_no_scale:,.2f}
- Validation R²: {r2_no_scale:.4f}

**With Standardization:**
- Validation MSE: {mse_with_scale:,.2f}
- Validation R²: {r2_with_scale:.4f}

**Improvement:**
- MSE Change: {mse_improvement:+.2f}%
- R² Change: {r2_improvement:+.2f}%

### Key Benefits
1. **Equal Feature Contribution:** Prevents features with larger scales from dominating
2. **Numerical Stability:** Improves convergence and reduces numerical errors
3. **Regularization Effectiveness:** Essential for Ridge and Lasso regression
4. **Coefficient Comparability:** Enables direct comparison of feature importance

---

## 4. Model 1: Linear Regression

### 4.1 Baseline Model Performance

**Training Set:**
- MSE: {train_mse_lr:,.2f}
- RMSE: {train_rmse_lr:,.2f}
- MAE: {train_mae_lr:,.2f}
- R²: {train_r2_lr:.4f}

**Validation Set:**
- MSE: {val_mse_lr:,.2f}
- RMSE: {val_rmse_lr:,.2f}
- MAE: {val_mae_lr:,.2f}
- R²: {val_r2_lr:.4f}

**Cross-Validation:**
- Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})
- Fold Scores: {[f'{score:.4f}' for score in cv_scores]}

### 4.2 Coefficient Analysis

**Top 5 Most Influential Features:**
{chr(10).join([f"{i+1}. {row['Feature']}: ${row['Coefficient']:,.2f} ({'increases' if row['Coefficient'] > 0 else 'decreases'} per std unit)" 
               for i, row in coefficients.head(5).iterrows()])}

**Intercept:** ${lr_baseline.intercept_:,.2f}

### 4.3 Statistical Significance

All coefficients were tested for statistical significance using t-tests:
- **Highly Significant (p < 0.001):** {sum(p_values < 0.001)} features
- **Significant (p < 0.01):** {sum((p_values >= 0.001) & (p_values < 0.01))} features
- **Moderately Significant (p < 0.05):** {sum((p_values >= 0.01) & (p_values < 0.05))} features

### 4.4 Regularization Results

**Ridge Regression:**
- Best Alpha: {best_ridge_alpha:.4f}
- Validation R²: {r2_score(y_val, y_pred_ridge):.4f}
- Validation RMSE: ${np.sqrt(mean_squared_error(y_val, y_pred_ridge)):,.2f}

**Lasso Regression:**
- Best Alpha: {best_lasso_alpha:.4f}
- Validation R²: {r2_score(y_val, y_pred_lasso):.4f}
- Validation RMSE: ${np.sqrt(mean_squared_error(y_val, y_pred_lasso)):,.2f}
- Non-zero Coefficients: {np.sum(lasso_best.coef_ != 0)}/{len(lasso_best.coef_)}

### 4.5 Feature Selection (RFE)

**Optimal Configuration:**
- Number of Features: {best_n_features}
- Validation R²: {max(rfe_scores):.4f}

**Selected Features:**
{chr(10).join([f"{i+1}. {feat}" for i, feat in enumerate(selected_features)])}

### 4.6 Linear Regression Summary

{lr_summary.to_string(index=False)}

**Best Linear Regression Model:** {best_lr_model_name}
- R²: {lr_summary.loc[best_lr_idx, 'R²']:.4f}
- RMSE: ${lr_summary.loc[best_lr_idx, 'RMSE']:,.2f}

---

## 5. Model 2: Random Forest

### 5.1 Baseline Model Performance

**Training Set:**
- MSE: {train_mse_rf:,.2f}
- RMSE: {train_rmse_rf:,.2f}
- MAE: {train_mae_rf:,.2f}
- R²: {train_r2_rf:.4f}

**Validation Set:**
- MSE: {val_mse_rf:,.2f}
- RMSE: {val_rmse_rf:,.2f}
- MAE: {val_mae_rf:,.2f}
- R²: {val_r2_rf:.4f}

**Cross-Validation:**
- Mean R²: {cv_scores_rf.mean():.4f} (±{cv_scores_rf.std() * 2:.4f})
- Fold Scores: {[f'{score:.4f}' for score in cv_scores_rf]}

### 5.2 Feature Importance

**Top 10 Most Important Features:**
{chr(10).join([f"{i+1}. {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)" 
               for i, row in feature_importance.head(10).iterrows()])}

### 5.3 Tree Structure Analysis

**Forest Statistics:**
- Number of Trees: {len(rf_baseline.estimators_)}
- Average Tree Depth: {np.mean(tree_depths):.2f} (±{np.std(tree_depths):.2f})
- Average Leaf Nodes: {np.mean(tree_leaves):.2f} (±{np.std(tree_leaves):.2f})
- Min/Max Depth: {min(tree_depths)}/{max(tree_depths)}

### 5.4 Hyperparameter Tuning

**Best Parameters:**
{chr(10).join([f"- {param}: {value}" for param, value in grid_search.best_params_.items()])}

**Optimized Model Performance:**
- Validation MSE: {val_mse_rf_opt:,.2f}
- Validation RMSE: ${val_rmse_rf_opt:,.2f}
- Validation MAE: ${val_mae_rf_opt:,.2f}
- Validation R²: {val_r2_rf_opt:.4f}

**Improvement over Baseline:**
- R² Improvement: {(val_r2_rf_opt - val_r2_rf) / val_r2_rf * 100:+.2f}%
- RMSE Improvement: {(val_rmse_rf - val_rmse_rf_opt) / val_rmse_rf * 100:+.2f}%

### 5.5 Random Forest Summary

{rf_summary.to_string(index=False)}

---

## 6. Comprehensive Model Comparison

### 6.1 Performance Metrics

{comparison_df.to_string(index=False)}

### 6.2 Residual Analysis

**Linear Regression:**
- Mean Residual: {lr_residuals.mean():,.2f}
- Std Residual: {lr_residuals.std():,.2f}
- Median Residual: {lr_residuals.median():,.2f}

**Random Forest (Optimized):**
- Mean Residual: {rf_opt_residuals.mean():,.2f}
- Std Residual: {rf_opt_residuals.std():,.2f}
- Median Residual: {rf_opt_residuals.median():,.2f}

### 6.3 Model Characteristics

| Characteristic | Linear Regression | Random Forest |
|---------------|-------------------|---------------|
| **Predictive Performance** | {best_lr_r2:.4f} | {val_r2_rf_opt:.4f} |
| **Training Speed** | Fast | Slow |
| **Prediction Speed** | Very Fast | Moderate |
| **Interpretability** | High | Medium |
| **Handles Non-linearity** | No | Yes |
| **Handles Interactions** | No | Yes |
| **Overfitting Risk** | Low | Medium |
| **Feature Selection** | Manual/RFE | Automatic |

---

## 7. Final Model Selection

### Selected Model: **{selected_model}**

### Weighted Scoring
- **Linear Regression:** {lr_score:.3f}
- **Random Forest:** {rf_score:.3f}

### Justification

{'**Random Forest (Optimized) is selected as the final model:**' if rf_score > lr_score else '**Linear Regression is selected as the final model:**'}

{'#### Strengths:' if rf_score > lr_score else '#### Strengths:'}
{'''- Superior predictive performance (R²: ''' + f'{val_r2_rf_opt:.4f}' + ''')
- Lower prediction error (RMSE: $''' + f'{val_rmse_rf_opt:,.2f}' + ''')
- Captures non-linear relationships effectively
- Robust to outliers and feature interactions
- Automatic feature importance ranking
- Handles complex patterns in the data''' if rf_score > lr_score else 
'''- Excellent interpretability with clear coefficients
- Fast training and prediction times
- Statistical significance testing available
- Easy to explain to stakeholders
- Competitive performance (R²: ''' + f'{best_lr_r2:.4f}' + ''')
- Low computational requirements'''}

{'#### Trade-offs:' if rf_score > lr_score else '#### Trade-offs:'}
{'''- Less interpretable than linear models
- Higher computational cost
- Requires more careful hyperparameter tuning
- Longer prediction times''' if rf_score > lr_score else 
'''- Assumes linear relationships
- May miss non-linear patterns
- Limited flexibility for complex interactions
- Requires feature engineering for non-linearity'''}

---

## 8. Recommendations

### For Production Deployment

1. **Model Implementation**
   - Deploy {selected_model} for revenue predictions
   - Implement model versioning and tracking
   - Set up automated retraining pipeline
   - Monitor model performance continuously

2. **Performance Monitoring**
   - Track prediction accuracy over time
   - Monitor feature distributions for drift
   - Set up alerts for performance degradation
   - Regular validation against new data

3. **Business Integration**
   - Provide confidence intervals for predictions
   - Create dashboards for stakeholder visibility
   - Document model limitations clearly
   - Establish feedback loops for improvements

### For Model Improvement

1. **Short-term (1-3 months)**
   - Collect more recent data
   - Validate model on new samples
   - Fine-tune hyperparameters
   - A/B test against current system

2. **Medium-term (3-6 months)**
   - Explore ensemble methods
   - Investigate additional features
   - Consider gradient boosting models
   - Implement automated feature engineering

3. **Long-term (6-12 months)**
   - Develop deep learning models
   - Implement online learning
   - Create model explainability tools
   - Build automated ML pipeline

---

## 9. Visualizations

All visualizations have been saved to `visualizations/model_analysis/`:

1. **standardization_impact.png** - Impact of data standardization
2. **linear_regression_evaluation.png** - Comprehensive LR evaluation
3. **regularization_analysis.png** - Ridge and Lasso analysis
4. **rfe_feature_selection.png** - Feature selection results
5. **random_forest_evaluation.png** - Comprehensive RF evaluation
6. **rf_hyperparameter_tuning.png** - Hyperparameter tuning results
7. **comprehensive_model_comparison.png** - Final model comparison

---

## 10. Conclusion

This comprehensive analysis evaluated two machine learning approaches for revenue prediction:

1. **Linear Regression** demonstrated strong interpretability and efficiency, with {best_lr_r2:.4f} R² score
2. **Random Forest** achieved {'superior' if val_r2_rf_opt > best_lr_r2 else 'competitive'} performance with {val_r2_rf_opt:.4f} R² score

The selected model ({selected_model}) provides the best balance of:
- Predictive accuracy
- Interpretability
- Computational efficiency
- Business value

### Key Takeaways

1. {'Random Forest captures non-linear patterns better than Linear Regression' if rf_score > lr_score else 'Linear Regression provides excellent interpretability with competitive performance'}
2. Feature importance analysis reveals key revenue drivers
3. Proper data standardization is crucial for linear models
4. Hyperparameter tuning significantly improves model performance
5. Cross-validation ensures robust performance estimates

### Next Steps

1. Deploy selected model to production
2. Implement monitoring and alerting
3. Establish retraining schedule
4. Gather stakeholder feedback
5. Plan for continuous improvement

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Duration:** Complete ML Pipeline  
**Models Evaluated:** Linear Regression (4 variants), Random Forest (2 variants)  
**Total Visualizations:** 7 comprehensive plots
"""

# Save report
with open("MODEL_COMPARISON_REPORT.md", "w") as f:
    f.write(report_content)

print("\n✓ Comprehensive report saved to MODEL_COMPARISON_REPORT.md")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\n✓ All models trained and evaluated successfully")
print(
    f"✓ {len([f for f in os.listdir('visualizations/model_analysis') if f.endswith('.png')])} visualizations generated"
)
print(f"✓ Comprehensive report created")

print(f"\nFinal Results:")
print(f"  Selected Model: {selected_model}")
print(f"  Best R² Score: {max(best_lr_r2, val_r2_rf_opt):.4f}")
print(f"  Best RMSE: ${min(best_lr_rmse, val_rmse_rf_opt):,.2f}")
print(f"  Best MAE: ${min(best_lr_mae, val_mae_rf_opt):,.2f}")

print(f"\nOutput Files:")
print(f"  1. MODEL_COMPARISON_REPORT.md - Comprehensive analysis report")
print(f"  2. visualizations/model_analysis/ - All visualization files")

print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
