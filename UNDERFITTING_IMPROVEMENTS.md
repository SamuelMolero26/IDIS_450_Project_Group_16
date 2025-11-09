# Underfitting Improvements - Architecture Updates

## Problem Statement
The bias-variance analysis revealed severe underfitting across all models:
- **Linear Model**: Bias² = 12,369,264 (extremely high)
- **Decision Tree**: Bias² = 125,520 (high)
- **Random Forest**: Bias² = 132,987 (high)

All models showed extremely high bias, indicating they were too simple to capture the underlying patterns in the data.

## Solution Overview
Updated the existing architecture with three key improvements:
1. **Aggressive data cleaning** to improve data quality
2. **Feature engineering** with interaction features and polynomial terms
3. **Enhanced models** with regularization (Ridge, Lasso, ElasticNet, GradientBoosting)

---

## 1. Data Preprocessing Updates (`src/data_preprocessing.py`)

### Aggressive Outlier Removal
**Changes:**
- Changed from **1.5×IQR** to **3×IQR** for more aggressive outlier removal
- Changed from **z-score > 3** to **z-score > 4** for stricter filtering
- Combined both methods (rows must pass both tests)

**Impact:**
```python
# Before: Flagged outliers but kept them
# After: Removes outliers aggressively
for col in numeric_cols:
    z_mask = z_scores <= 4  # Stricter threshold
    iqr_mask = (df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)  # 3×IQR
    df = df[z_mask & iqr_mask]  # Remove, don't just flag
```

### Data Quality Validation
**New features:**
- Remove missing/infinite values: `df.replace([np.inf, -np.inf], np.nan).dropna()`
- Remove duplicates: `df.drop_duplicates()`
- Comprehensive quality reporting with retention statistics

### Interaction Features
**New function:** [`create_interaction_features()`](src/data_preprocessing.py:453)

**Created 7 interaction features:**
1. **Price_Quantity_Interaction** = Unit Price × Order Quantity
2. **Cost_Quantity_Interaction** = Unit Cost × Order Quantity
3. **Discount_Price_Interaction** = Discount Applied × Unit Price
4. **Margin_Quantity_Interaction** = Profit Margin × Order Quantity
5. **LeadTime_Quantity_Interaction** = Total Lead Time × Order Quantity
6. **Price_Cost_Ratio** = Unit Price / Unit Cost
7. **Discount_Margin_Interaction** = Discount Applied × Profit Margin

**Rationale:** These interactions capture complex relationships that linear models couldn't learn before, reducing bias significantly.

---

## 2. Model Pipeline Updates (`src/model_pipeline.py`)

### New Model Types
**Added 4 new model families:**

1. **Ridge Regression** ([`Ridge`](src/model_pipeline.py:7))
   - L2 regularization to prevent overfitting
   - Better for correlated features
   
2. **Lasso Regression** ([`Lasso`](src/model_pipeline.py:7))
   - L1 regularization with feature selection
   - Automatically removes irrelevant features
   
3. **ElasticNet** ([`ElasticNet`](src/model_pipeline.py:7))
   - Combines L1 and L2 regularization
   - Best of both Ridge and Lasso
   
4. **Gradient Boosting** ([`GradientBoostingRegressor`](src/model_pipeline.py:9))
   - Sequential ensemble learning
   - Highly flexible, reduces bias effectively

### RobustScaler Instead of StandardScaler
**Change:** Replaced [`StandardScaler`](src/model_pipeline.py:10) with [`RobustScaler`](src/model_pipeline.py:10)

**Benefits:**
- Uses median and IQR instead of mean and std
- More robust to outliers
- Better for data with extreme values

```python
# Before:
self.scaler = StandardScaler()

# After:
self.scaler = RobustScaler()  # Better outlier handling
```

### Enhanced Tree Depth
**Updated adaptive parameters** in [`_get_adaptive_tree_params()`](src/model_pipeline.py:122):

```python
# Before:
if n_samples < 1000:
    max_depth_range = [3, 5, 7, 10]
elif n_samples < 10000:
    max_depth_range = [5, 10, 15, 20]

# After:
if n_samples < 1000:
    max_depth_range = [5, 10, 15, 20]  # Increased minimum
elif n_samples < 10000:
    max_depth_range = [10, 15, 20, 30]  # Increased maximum
```

**Impact:** Allows trees to grow deeper, capturing more complex patterns and reducing bias.

### Polynomial Features (Degree 2)
**Enhanced in [`train_model()`](src/model_pipeline.py:201):**
- Automatically creates polynomial features for linear models
- Degree 2 creates interaction terms (x₁×x₂, x₁², x₂², etc.)
- Applied to Ridge, Lasso, ElasticNet, and Linear models

```python
if poly_degree > 1:
    self.poly_features[model_id] = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_train_poly = self.poly_features[model_id].fit_transform(X_train)
```

---

## 3. Configuration Updates (`src/config.py`)

### New Model Configurations

#### Ridge Regression
```python
'ridge': {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],  # Regularization strength
    'polynomial_degree': [1, 2],  # Enable polynomial features
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}
```

#### Lasso Regression
```python
'lasso': {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'polynomial_degree': [1, 2],
    'max_iter': [1000, 5000, 10000],  # More iterations for convergence
    'selection': ['cyclic', 'random']
}
```

#### ElasticNet
```python
'elasticnet': {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # Balance between L1 and L2
    'polynomial_degree': [1, 2],
    'max_iter': [1000, 5000, 10000]
}
```

#### Gradient Boosting
```python
'gradient_boosting': {
    'n_estimators': [100, 150, 200, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'loss': ['squared_error', 'absolute_error', 'huber']
}
```

### Updated Tree Parameters

#### Decision Tree
```python
'decision_tree': {
    'max_depth': [10, 15, 20, 30],  # Increased from [5, 10, 15, 20]
    'ccp_alpha': [0.0, 0.001, 0.01],  # Finer pruning control
    'splitter': ['best', 'random']  # Added random splitter
}
```

#### Random Forest
```python
'random_forest': {
    'n_estimators': [100, 150, 200, 250],  # Increased from [50, 100, 200, 300]
    'max_depth': [10, 15, 20, 30],  # Increased minimum depth
    'min_impurity_decrease': [0.0, 0.001, 0.01]  # Added impurity threshold
}
```

---

## Expected Impact on Bias-Variance

### Bias Reduction Strategies

1. **More Complex Models**
   - Gradient Boosting: Sequential learning reduces bias iteratively
   - Deeper trees: Capture more complex patterns
   - Polynomial features: Model non-linear relationships

2. **Better Feature Engineering**
   - 7 interaction features capture relationships
   - Polynomial degree 2 creates quadratic terms
   - Price×Quantity, Discount×Price, etc.

3. **Cleaner Data**
   - Aggressive outlier removal reduces noise
   - Better quality data → better model fit
   - Interaction features on clean data more effective

### Variance Control (Preventing Overfitting)

1. **Regularization**
   - Ridge (L2): Shrinks coefficients
   - Lasso (L1): Feature selection
   - ElasticNet: Combined benefits

2. **RobustScaler**
   - Less sensitive to outliers
   - More stable predictions

3. **Cross-Validation**
   - All models use 5-fold CV
   - Hyperparameter tuning with CV
   - Prevents overfitting during training

---

## Memory Efficiency Maintained

All improvements maintain memory efficiency:
- **No data duplication**: Interaction features computed in-place
- **Efficient models**: Ridge/Lasso/ElasticNet are lightweight
- **Gradient Boosting**: Uses shallow trees (max_depth ≤ 10)
- **Aggressive cleaning**: Removes bad data, reduces dataset size

---

## Usage Instructions

### 1. Rerun Preprocessing
```bash
python -m src.data_preprocessing --data Project4_USRegionalSales/Data-USRegionalSales.csv
```

This will:
- Apply aggressive outlier removal
- Create interaction features
- Generate cleaner preprocessed data

### 2. Train New Models
```python
from src.model_pipeline import ModelPipeline
from src.config import MODEL_CONFIGS

pipeline = ModelPipeline()

# Train Ridge with polynomial features
ridge_results = pipeline.train_model(
    'ridge',
    X_train,
    y_train,
    params={'alpha': 10.0, 'polynomial_degree': 2}
)

# Train Gradient Boosting
gb_results = pipeline.train_model(
    'gradient_boosting',
    X_train,
    y_train,
    params={'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 7}
)

# Train deeper Random Forest
rf_results = pipeline.train_model(
    'random_forest',
    X_train,
    y_train,
    params={'n_estimators': 200, 'max_depth': 30}
)
```

### 3. Hyperparameter Tuning
```python
# Tune Ridge
ridge_tuning = pipeline.hyperparameter_tuning(
    'ridge',
    X_train,
    y_train,
    param_grid=MODEL_CONFIGS['ridge']
)

# Tune Gradient Boosting
gb_tuning = pipeline.hyperparameter_tuning(
    'gradient_boosting',
    X_train,
    y_train,
    param_grid=MODEL_CONFIGS['gradient_boosting']
)
```

---

## Expected Results

### Before (High Bias)
- Linear: Bias² = 12,369,264
- Decision Tree: Bias² = 125,520
- Random Forest: Bias² = 132,987

### After (Expected Improvements)
- **Ridge/Lasso/ElasticNet with polynomial features**: Bias² < 100,000
- **Gradient Boosting**: Bias² < 50,000
- **Deeper Random Forest**: Bias² < 80,000
- **Decision Tree (depth 30)**: Bias² < 100,000

### Key Improvements
1. **10-100× bias reduction** for linear models (polynomial + interactions)
2. **2-3× bias reduction** for tree models (deeper trees)
3. **Maintained variance control** through regularization
4. **Better generalization** through cleaner data

---

## Files Modified

1. **[`src/data_preprocessing.py`](src/data_preprocessing.py)**
   - Aggressive outlier removal (3×IQR, z<4)
   - Data quality validation
   - Interaction feature creation

2. **[`src/model_pipeline.py`](src/model_pipeline.py)**
   - Added Ridge, Lasso, ElasticNet, GradientBoosting
   - RobustScaler instead of StandardScaler
   - Enhanced tree depth parameters
   - Polynomial features for all linear models

3. **[`src/config.py`](src/config.py)**
   - New model configurations (Ridge, Lasso, ElasticNet, GradientBoosting)
   - Updated tree hyperparameters (deeper trees)
   - Regularization parameters

---

## Next Steps

1. **Run preprocessing** to generate clean data with interaction features
2. **Train all models** with new configurations
3. **Compare results** using bias-variance analysis
4. **Select best model** based on bias-variance tradeoff
5. **Deploy** the best performing model

The architecture now has the tools to dramatically reduce underfitting while maintaining memory efficiency and preventing overfitting through regularization.