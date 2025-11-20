# Prediction and Feature Visualization Summary

**Generated:** 2025-11-12 20:11:14
**Dataset:** 7809 samples (6247 train, 1562 test)
**Features:** 9
**Models Evaluated:** 3

## Generated Visualizations

### 1. Prediction Scatter Plots (`prediction_scatter_plots.png`)
- Scatter plots of predicted vs actual values for all 3 models
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
| Linear Regression | 0.8375 | 3135.59 | 2276.35 |
| Decision Tree | 0.9038 | 2413.19 | 1395.35 |
| Random Forest | 0.9835 | 998.30 | 625.98 |


## Best Performing Model

**🏆 Winner:** Random Forest
- **Test R²:** 0.9835
- **Test RMSE:** 998.30
- **Test MAE:** 625.98

## Key Insights

1. **Best Model Performance:** Random Forest shows the highest prediction accuracy
2. **Feature Importance:** Top features driving predictions (from tree models)
3. **Error Patterns:** Models tend to perform better/worse in certain feature ranges
4. **Prediction Reliability:** Error distributions show model consistency

## Usage Recommendations

- Use Random Forest for production predictions
- Monitor prediction errors in identified problematic ranges
- Consider feature engineering for underperforming feature ranges
- Regular model retraining with new data

---

*Generated automatically by PredictionVisualizer*
*All visualizations saved in `visualizations/predictions/` directory*
