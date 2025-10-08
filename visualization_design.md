# Improved Sales Data Visualizations Design

## Overview

This design proposes enhanced visualizations for the US Regional Sales dataset to provide deeper insights into measures of location (mean, median, mode), dispersion (variance, standard deviation, range, IQR), and shape (skewness, kurtosis) for each variable, as well as explore correlations and relationships.

## Existing Visualizations Assessment

Current visualizations include:
- Correlation heatmap
- Profit margin by sales channel (likely bar chart)
- Sales channel vs average total revenue
- Unit cost vs profit margin scatter plot
- Unit price vs total revenue scatter plot

These provide basic relationship insights but lack detailed distribution analysis and statistical summaries.

## Proposed Visualizations

### 1. Enhanced Histograms for Distribution Analysis

For each numerical variable, create histograms with statistical overlays.

**Variables to visualize:**
- Order Quantity
- Discount Applied
- Unit Cost
- Unit Price
- Procurement_to_Order_Days
- Order_to_Ship_Days
- Ship_to_Delivery_Days
- Total_Lead_Time
- Profit_Margin
- Total_Revenue

**Design Elements:**
- Histogram with optimized binning (Freedman-Diaconis rule)
- Kernel Density Estimate (KDE) curve overlay
- Vertical lines for mean (solid), median (dashed)
- Shaded areas for quartiles
- Text annotations for:
  - Mean, median, mode
  - Standard deviation
  - Skewness coefficient
  - Kurtosis coefficient
  - Range and IQR
- For skewed distributions, include log-transformed version
- Facet plots by Sales Channel for key financial variables (Unit Price, Profit_Margin, Total_Revenue)

**Interpretation:**
- Location: Mean and median indicate central tendency; difference shows skewness.
- Dispersion: Std dev and IQR show spread; range indicates extremes.
- Shape: Skewness >0 right-skewed (e.g., revenue), <0 left-skewed; kurtosis >3 leptokurtic (heavy tails).

### 2. Enhanced Scatter Plots for Relationships

Build on existing scatter plots and add new ones.

**Key Relationships:**
- Unit Price vs Total Revenue (existing - enhance)
- Unit Cost vs Profit Margin (existing - enhance)
- Order Quantity vs Total Revenue (new)
- Discount Applied vs Profit Margin (new)
- Total_Lead_Time vs Total_Revenue (new)

**Design Enhancements:**
- Color coding by Sales Channel
- Size of points by Order Quantity (for revenue plots)
- Regression line with confidence interval
- Correlation coefficient annotation
- Outlier highlighting (based on IQR method)
- Marginal histograms or KDE plots
- Facet by WarehouseCode for regional insights

**Interpretation:**
- Correlation strength and direction
- Channel-specific patterns (e.g., online vs in-store pricing)
- Outliers indicating unusual transactions
- Predictive relationships for forecasting

### 3. Advanced Correlation Visualizations

**Clustered Correlation Heatmap:**
- Hierarchical clustering of variables
- Color scale from -1 to 1
- Significance stars (* p<0.05, ** p<0.01)
- Group variables by category (financial, temporal, quantity)

**Partial Correlation Matrix:**
- Control for confounding variables
- Show direct relationships

**Correlation Network Graph:**
- Nodes as variables, edges as correlations
- Edge thickness by correlation strength
- Color nodes by variable type

**Interpretation:**
- Identify multicollinearity for feature selection
- Understand direct vs indirect relationships
- Guide regression model building

## Implementation Plan

1. Use Python with seaborn, plotly for interactive versions
2. Calculate statistics using pandas and scipy
3. Save high-resolution images and interactive HTML files and JPEG files


## Expected Insights

- Distribution shapes will reveal data characteristics (e.g., revenue likely right-skewed)
- Channel differences in pricing and margins
- Temporal patterns in lead times
- Key drivers of revenue and profitability
- Outlier transactions for further investigation

This design provides comprehensive statistical insights while maintaining visual clarity.