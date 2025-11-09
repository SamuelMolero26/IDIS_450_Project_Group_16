# Prediction and Feature Visualization Summary

**Generated:** 2025-11-04 10:34:29
**Dataset:** 7840 samples (6272 train, 1568 test)
**Features:** 11
**Models Evaluated:** 8

## Generated Visualizations

### 1. Prediction Scatter Plots (`prediction_scatter_plots.png`)
- Scatter plots of predicted vs actual values for all 8 models
- Includes perfect prediction reference line
- Shows model accuracy visually

### 2. Residual Analysis (`residual_analysis.png`)
- Residual plots showing prediction errors vs predicted values
- Helps identify heteroscedasticity and systematic errors
- Reference line at zero error

### 3. Feature Importance Comparison (`feature_importance_comparison.png`)
- Feature importance plots for tree-based models
- Shows which features contribute most to predictions
- Includes importance values

### 4. Prediction Error Distributions (`prediction_error_distributions.png`)
- Histograms of prediction errors for each model
- Shows error distribution and central tendency
- Reference line at zero error

### 5. Model Performance Dashboard (`model_performance_dashboard.png`)
- Comprehensive dashboard with:
  - R² score comparison
  - RMSE comparison
  - MAE comparison
  - Performance trade-off scatter plot

### 6. Feature vs Prediction Scatter (`feature_vs_prediction_scatter.png`)
- Scatter plots of top features vs model predictions
- Shows relationship between features and predictions
- Includes correlation coefficients

### 7. Error Analysis by Ranges (`error_analysis_by_ranges.png`)
- Error analysis across different feature value ranges
- Helps identify where models perform poorly
- Shows mean absolute error by feature bins

## Model Performance Summary

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| Linear Regression | 0.8384 | 3081.09 | 2267.97 |
| Ridge (α=1) | 0.8383 | 3081.80 | 2266.81 |
| Ridge (α=10) | 0.8374 | 3090.46 | 2262.68 |
| Lasso (α=1) | 0.8384 | 3081.42 | 2266.39 |
| ElasticNet | 0.7029 | 4177.96 | 3025.92 |
| Decision Tree | 0.9957 | 501.69 | 239.64 |
| Random Forest | 0.9961 | 480.96 | 198.35 |
| Gradient Boosting | 0.9991 | 233.76 | 131.80 |


## Best Performing Model

**🏆 Winner:** Gradient Boosting
- **Test R²:** 0.9991
- **Test RMSE:** 233.76
- **Test MAE:** 131.80

## Key Insights

1. **Best Model Performance:** Gradient Boosting shows the highest prediction accuracy
2. **Feature Importance:** Top features driving predictions (from tree models)
3. **Error Patterns:** Models tend to perform better/worse in certain feature ranges
4. **Prediction Reliability:** Error distributions show model consistency

## Usage Recommendations

- Use Gradient Boosting for production predictions
- Monitor prediction errors in identified problematic ranges
- Consider feature engineering for underperforming feature ranges
- Regular model retraining with new data

---

*Generated automatically by PredictionVisualizer*
*All visualizations saved in `visualizations/predictions/` directory*
