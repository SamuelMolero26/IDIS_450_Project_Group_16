# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dataset expectation Requirements:

**Detailed Description of the US Regional Sales Dataset**

The dataset provides an extensive collection of US regional sales data, capturing a diverse range of details pertaining to sales transactions and customer interactions across different sales channels. With 7992 rows and 15 columns, it offers a comprehensive overview of the sales landscape, encompassing various aspects from order and product details to sales performance metrics. This dataset is valuable for analyzing sales patterns, optimizing inventory management, improving customer relationships, and forecasting future sales.


**Feature Descriptions**

1. OrderNumber: This feature uniquely identifies each sales order with a string value, such as "SO 000101".
2. Sales Channel: This categorical feature indicates the channel through which the sale was made, such as In-Store, Online, Distributor, or Wholesale.
3. WarehouseCode: A string feature representing the warehouse code from where the product was shipped, for example, "WARE-UHY1004".
4. ProcuredDate: This date feature (formatted as DD-MM-YYYY) shows when the product was procured, such as "31-12-2017".
5. OrderDate: A date feature (formatted as DD-MM-YYYY) indicating when the order was placed, like "31-05-2018".
6. ShipDate: This date feature (formatted as DD-MM-YYYY) represents when the product was shipped, for instance, "14-06-2018"
7. DeliveryDate: A date feature (formatted as DD-MM-YYYY) showing when the product was delivered, such as "19-06-2018".

8. CurrencyCode: A string feature indicating the currency in which the transaction was made, like "USD".

9. _SalesTeamID: An integer feature identifying the sales team responsible for the order, e.g., 6.

10. _CustomerID: An integer feature representing the customer who placed the order, for instance, 15.

11. _StoreID: An integer feature identifying the store where the order was placed, such as 259.

12. _ProductID: An integer feature representing the product sold, e.g., 12.

13. Order Quantity: An integer feature indicating the quantity of the product ordered, like 5.

14. Discount Applied: A float feature representing the discount applied to the order, such as 0.075.

15. Unit Cost: A float feature showing the cost price of one unit of the product, for example, 1,001.18.

16. Unit Price: A float feature representing the selling price of one unit of the product, like 1,963.10.


**Potential Usecases:**

This dataset can be used for various analytical and machine learning tasks, including:

1. Sales Analysis: Understanding sales patterns across different channels and regions.
2. Inventory Management: Optimizing stock levels based on sales trends.
3. Customer Segmentation: Identifying different customer segments based on purchasing behavior.
4. Revenue Forecasting: Predicting future sales and revenue.
5. Discount Effectiveness: Analyzing the impact of discounts on sales.


**Applicable Machine Learning Algorithms**

Use of Classification Algorithms (e.g., Logistic regression, random forest, SVM, decision tree, KNN, and ANN):

- Classify high-value vs. low-value orders using features such as unit price and discount applied.

- Predict the likelihood of a product being sold through a particular channel.

Use of Regression Algorithms (e.g., Linear regression, random forest, SVM, decision tree, KNN, and ANN):

- predict sales (or revenue) based on order quantity, discount, and unit price.

- forecast delivery times using procurement and shipping dates.

Use of Unsupervised Learning Algorithms (e.g., Kmeans clustering):

- To segment customers into clusters based on similar attributes, which can be useful for targeted marketing and personalized services.

**What is expected?**

Understand the objective of the project.
Preprocessing the data (look for missing values, distribution of the features, transform some of the features), not necessary to follow all the steps.
Try to find the correlation between the features and ignore some of them (feature selection) if needed with the techniques that’ll be taught in the class.
Split the data into training, validation, and test (might require iterations to get better results).
Build at least 3 different machine learning models with the training data and validate which one would be best for testing.
Finally, use proper metrics to estimate which model has performed better.


## FOCUS / Requirements:

**Model 1- Linear/Logistic Regression**:

1. Split the data into training and validation sets. Choose an appropriate split ratio based on the dataset size and modeling objectives. Use the same training and validation sets for the second model . 

2. Standardization: Decide whether to standardize the data based on feature scale, distribution, and model requirements. Explain the impact of standardization on model performance and justify your decision .

4. Model Fitting: Fit the Linear Regression or Logistic Regression model on the training data. Present and explain your findings based on the relevant lab sessions .

5. Model Evaluation: Report the model’s accuracy for both training and validation sets. Use appropriate metrics such as accuracy, precision, recall, and F1-score for classification, and MSE, MAE, and R-squared for regression. Provide visualizations such as a confusion matrix or ROC curve, and explain your model’s performance across different classes or target ranges .

6. Hyperparameter Tuning and Variable Reduction: Perform hyperparameter tuning (only for Logistic Regression) and variable reduction to find the best model with the fewest predictors and highest accuracy. Present the best Linear/Logistic model for comparison with the Decision Tree/Random Forest model . 

**Model 2- Decision Tree and Random Forest**:

1. Use the same training and validation sets from the Linear/Logistic Regression model . 

2. Model Fitting: Fit a Decision Tree or Random Forest model on the training data. Present and explain your findings . 

5. Model Evaluation: Report the model’s accuracy for both training and validation sets. Use appropriate metrics such as accuracy, precision, recall, and F1-score for classification, and MSE, MAE, and R-squared for regression. Provide visualizations such as a confusion matrix or ROC curve, and explain your model’s performance across different classes or target ranges.

6. Hyperparameter Tuning. Perform hyperparameter tuning using techniques like GridSearchCV or manually change parameters in the model. Explain the tuning process and how it impacted model performance .

7. Compare the best version of the Decision Tree/Random Forest model after hyperparameter tuning (model 2) with the best Linear/Logistic model (model 1). Select the best model based on the evaluation metrics discussed earlier.



## Project Overview

This is an advanced machine learning pipeline for US Regional Sales data analysis, featuring meta-learning, continuous learning, and comprehensive model evaluation. The system is designed to be self-improving, learning from historical experiments to optimize future model configurations.

**Key Capabilities:**
- Multi-model training pipeline (Linear Regression, Logistic Regression, Decision Tree, Random Forest)
- Quantitative evaluation with K-fold CV and bias-variance decomposition
- Qualitative evaluation using SHAP interpretability and business rule validation
- Meta-learning system that predicts optimal configurations from historical data
- Redis caching with automatic SQLite fallback
- Version control for datasets and models
- Continuous learning loop for self-improvement

## Quick Start

**Get running in 3 steps:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess the data (first time only)
python src/data_preprocessing.py

# 3. Run the complete pipeline
python main_pipeline.py
```

**Expected output:** Pipeline report in `reports/pipeline_report_<id>.json`, visualizations in `visualizations/`, and logs in `logs/`.

**Note:** Redis is optional - the system automatically falls back to SQLite if Redis is unavailable.

## Development Commands

### Running the Pipeline

```bash
# Run the complete advanced modeling pipeline (main entry point)
python main_pipeline.py

# Run data preprocessing only
python src/data_preprocessing.py

# Run specific pipeline components programmatically
python -c "from src.main_pipeline import run_standard_pipeline; run_standard_pipeline()"

# Quick model comparison (fewer models for faster iteration)
python -c "from src.main_pipeline import run_quick_comparison; run_quick_comparison(['linear', 'random_forest'])"
```

### Data Processing

```bash
# Preprocess raw sales data (creates preprocessed_sales_data.csv)
python src/data_preprocessing.py

# The preprocessing pipeline:
# 1. Loads from Project4_USRegionalSales/Data-USRegionalSales.csv
# 2. Performs outlier detection (Z-score, IQR, Isolation Forest, LOF)
# 3. Engineers derived features (profit margins, lead times, total revenue)
# 4. Saves to preprocessed_sales_data.csv
```

### Visualization Generation

```bash
# Generate improved statistical visualizations
python generate_improved_visualizations.py

# Generate prediction visualizations
python generate_prediction_visualizations.py

# Generate lightweight reports
python generate_lightweight_report.py
```

### Testing and Development

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Start Redis for optimal caching (automatic SQLite fallback if unavailable)
brew install redis  # macOS
brew services start redis

# Or for Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, redis; print('Dependencies OK')"
```

### Environment Variables

The pipeline supports optional environment variables for Redis configuration:

```bash
# Redis connection settings (all optional, defaults shown)
export REDIS_HOST=localhost        # Redis server host
export REDIS_PORT=6379            # Redis server port
export REDIS_DB=0                 # Redis database number
export REDIS_PASSWORD=            # Redis password (if required)
```

**Example with custom Redis:**
```bash
export REDIS_HOST=my-redis-server.com
export REDIS_PORT=6380
export REDIS_PASSWORD=mysecretpassword
python main_pipeline.py
```

**Note:** If Redis environment variables are not set or Redis is unavailable, the system automatically uses SQLite as a fallback cache (`cache/cache.db`).

### Entry Points

The pipeline has two main entry points:

1. **`main_pipeline.py`** (project root): Launcher script that wraps the core pipeline
   - Use this for standard pipeline execution
   - Provides user-friendly output with emoji indicators
   - Usage: `python main_pipeline.py`

2. **`src/main_pipeline.py`**: Core pipeline implementation module
   - Contains `AdvancedModelingPipeline` class and execution functions
   - Use for programmatic access and custom workflows
   - Usage: `from src.main_pipeline import run_standard_pipeline; run_standard_pipeline()`

Both execute the same underlying pipeline. Use the root `main_pipeline.py` for standard runs, and import from `src/main_pipeline.py` for custom scripts.

## Architecture

### High-Level Data Flow

```
Data Loading → Feature Prep → Train/Val Split
    ↓
Core Modeling (with Redis Caching)
    ├── Linear/Regularized Models (Ridge, Lasso, ElasticNet)
    ├── Decision Trees (with Gini optimization, CCP pruning)
    └── Ensemble Models (Random Forest, Gradient Boosting)
    ↓
Enhanced Evaluation
    ├── Quantitative: K-Fold CV, Bias-Variance Analysis, Hyperparameter Tuning
    └── Qualitative: SHAP Analysis, Error Analysis, Business Alignment
    ↓
Meta-Learning & Continuous Improvement
    ├── Meta-Learner: Predicts optimal configs from historical experiments
    ├── Warm Start: Uses cached insights for initialization
    └── Self-Improvement Cycle
    ↓
Model Selection & Reporting (JSON reports + visualizations)
```

### Core Components

**Data Pipeline (`src/data_loader.py`, `src/data_preprocessing.py`):**
- Loads and versions datasets with hash-based change tracking
- Feature engineering: profit margins, lead times, total revenue
- Multi-stage outlier detection (univariate, multivariate, contextual)

**Model Pipeline (`src/model_pipeline.py`):**
- Supports 7+ model types with adaptive hyperparameter selection
- Tree models use Gini impurity optimization and cost complexity pruning
- Automatic standardization decisions based on model type
- Redis caching for trained models and configurations

**Evaluation Engine (`src/evaluation_engine.py`):**
- Quantitative: CV scores, bias-variance decomposition via bootstrapping
- Performance metrics: MSE, RMSE, MAE, R², MAPE
- Learning curves and validation curves for diagnostics

**Qualitative Evaluator (`src/qualitative_evaluator.py`):**
- SHAP-based feature importance and model interpretability
- Error pattern analysis to detect systematic failures
- Business rule validation against domain constraints (defined in `src/config.py`)

**Meta-Learner (`src/meta_learner.py`):**
- Predicts optimal model configurations from historical experiment metadata
- Uses gradient boosting on features like dataset size, target distribution, model type
- Stored in cache for warm starts on subsequent runs

**Continuous Learning (`src/continuous_learning.py`):**
- Evaluates learning progress across experiments
- Adapts strategies based on performance trends
- Implements warm starts to accelerate convergence

**Caching Layer (`redis_cache.py`):**
- Primary: Redis for sub-millisecond access to experiment results
- Fallback: SQLite-based file cache (`cache/cache.db`)
- Stores: model results, evaluation metrics, meta-learner predictions

**Version Control (`src/version_control.py`):**
- Hash-based versioning for datasets and models
- Tracks metadata, parameters, and performance for each version
- Enables experiment reproducibility and comparison

### Configuration System

**Central Config (`src/config.py`):**
- Project paths (data, models, cache, logs, reports)
- Redis connection settings (host, port, DB, password via env vars)
- Model hyperparameter grids for all 7 model types
- Feature column definitions (numerical and categorical)
- Target column: `Total_Revenue` (regression task)
- Business rules for qualitative evaluation
- Random state: 42 (for reproducibility)
- CV folds: 5

**Key Configuration Points:**
- To change target variable: Modify `TARGET_COLUMN` in `src/config.py`
- To add new features: Update `NUMERICAL_FEATURES` or `CATEGORICAL_FEATURES`
- To tune hyperparameters: Modify `MODEL_CONFIGS` dictionaries
- To adjust business rules: Update `BUSINESS_RULES` dictionary

### Logging System

**Structured Logging (`src/logger.py`):**
- Pipeline logs: `logs/pipeline.log`
- Model training: `logs/model.log`
- Evaluation: `logs/evaluation.log`
- Meta-learning: `logs/meta.log`

All logs use consistent format: `timestamp - logger - level - message`

### Output Artifacts

**Reports Directory (`reports/`):**
- `pipeline_report_<experiment_id>.json`: Comprehensive results for each pipeline run
- Includes data info, model results, qualitative analysis, learning progress
- Each experiment gets unique 8-character UUID

**Visualizations:**
- `visualizations/preprocessing/`: Data analysis visualizations (histograms, scatter plots, correlations)
- `visualizations/ml_analysis/`: Model performance visualizations (comparisons, feature importance, bias-variance)
- `visualizations/predictions/`: Prediction analysis plots
- `visualizations/cv_analysis/`: Cross-validation fold analysis
- `reports/*.png`: Residual diagnostics for each model

**Cache Directory (`cache/`):**
- `cache.db`: SQLite fallback cache
- `learning_history.json`: Continuous learning metadata
- `versions/version_history.json`: Dataset and model version history

## Important Implementation Details

### Model Training Flow

1. **Data Loading:** `DataLoader` loads `preprocessed_sales_data.csv`, creates hash for versioning
2. **Feature Preprocessing:** Separate handling for numerical (scaling) and categorical (one-hot encoding) features
3. **Warm Start:** `ContinuousLearning` provides recommended config based on similar past experiments
4. **Model Training:** `ModelPipeline.train_model()` trains with hyperparameter tuning
5. **Evaluation:** `EvaluationEngine.evaluate_regression_model()` performs CV, bias-variance analysis
6. **Qualitative Assessment:** `QualitativeEvaluator` runs SHAP, error analysis, business checks
7. **Meta-Learning Update:** Results fed to `MetaLearner.collect_experiment_data()`
8. **Caching:** All results cached in Redis/SQLite for future warm starts

### Tree Model Regularization

Decision Trees and Random Forests use multiple overfitting prevention techniques:
- **Cost Complexity Pruning (CCP):** `ccp_alpha` parameter prunes weak branches
- **Adaptive Parameters:** Max depth, min samples split/leaf adjusted based on dataset size
- **Bootstrap Sampling:** Random Forests use `max_samples` to control tree diversity
- **Gini Index Optimization:** Feature splits selected using Gini impurity

These are configured in `MODEL_CONFIGS['decision_tree']` and `MODEL_CONFIGS['random_forest']` in `src/config.py`.

### Cache Keys and TTL

Cache keys follow patterns:
- `experiment:<experiment_id>`: Full experiment results
- `model:<model_id>`: Trained model metadata
- `evaluation:<model_id>`: Evaluation metrics
- `meta_learner:optimal_config`: Predicted optimal configuration
- `learning:history`: Continuous learning metadata

Default TTL: 3600 seconds (1 hour), configurable in `src/config.py:CACHE_TTL`

### Common Pitfalls

1. **Missing Preprocessed Data:** Pipeline expects `preprocessed_sales_data.csv` at project root. Run `python src/data_preprocessing.py` first.

2. **Redis Connection:** If Redis fails to connect, system automatically falls back to SQLite. No action needed, but performance may be slower.

3. **Model ID Tracking:** Models stored in `ModelPipeline.trained_models` dict with unique IDs. When passing models between components, always use model IDs, not model objects directly.

4. **SHAP Memory Usage:** SHAP analysis can be memory-intensive. Configured to use max 1000 samples (`SHAP_SAMPLE_SIZE` in config). Adjust if running on limited memory.

5. **Experiment IDs:** Each pipeline run generates unique 8-char experiment ID. Reports saved as `pipeline_report_<id>.json`. Track IDs for reproducibility.

### Key Data Transformations

**Derived Features (created in preprocessing):**
- `Profit_Margin`: `((Unit Price × (1 - Discount Applied)) - Unit Cost) / Unit Cost`
- `Total_Revenue`: `Order Quantity × Unit Price × (1 - Discount Applied)`
- `Total_Lead_Time`: Days from OrderDate to DeliveryDate
- `Procurement_to_Order_Days`: Days from ProcuredDate to OrderDate
- `Order_to_Ship_Days`: Days from OrderDate to ShipDate
- `Ship_to_Delivery_Days`: Days from ShipDate to DeliveryDate

**Feature Scaling:**
- Linear models: StandardScaler applied to numerical features
- Tree models: No scaling applied (trees are scale-invariant)
- Polynomial features: Degree 1 or 2, configured per model in `MODEL_CONFIGS`

### Extension Points

**Adding New Model Types:**
1. Add model class mapping in `ModelPipeline._get_model_class()`
2. Define hyperparameter grid in `src/config.py:MODEL_CONFIGS`
3. Implement any custom preprocessing in `ModelPipeline.train_model()`

**Adding New Metrics:**
1. Update `REGRESSION_METRICS` or `CLASSIFICATION_METRICS` in `src/config.py`
2. Implement metric calculation in `EvaluationEngine.evaluate_regression_model()`
3. Add visualization in `src/utils/visualization_utils.py`

**Custom Business Rules:**
1. Define rules in `src/config.py:BUSINESS_RULES`
2. Implement validation in `QualitativeEvaluator.check_business_alignment()`

## Dataset Information

**Source:** Project4_USRegionalSales/Data-USRegionalSales.csv (7,992 transactions, 2017-2018)

**Original Features:**
- OrderNumber, OrderDate, ShipDate, DeliveryDate
- Sales Channel (In-Store, Online, Distributor, Wholesale)
- Order Quantity, Unit Price, Unit Cost, Discount Applied
- SalesTeamID, CustomerID, StoreID, ProductID, WarehouseCode

**Target:** `Total_Revenue` (regression task)

**Feature Columns for Modeling:**
- Numerical: Order Quantity, Discount Applied, Unit Cost, Unit Price, Procurement_to_Order_Days, Order_to_Ship_Days, Ship_to_Delivery_Days, Total_Lead_Time, Profit_Margin
- Categorical: Sales Channel, WarehouseCode

## Known Issues and Limitations

1. **Single-threaded Execution:** Pipeline runs sequentially. CV folds and bootstrap iterations could be parallelized for performance gains.

2. **Memory Constraints:** Full dataset loaded into memory. For larger datasets (>100K rows), consider batch processing or streaming.

3. **Categorical Encoding:** Uses one-hot encoding. High-cardinality categoricals may cause dimensionality explosion. Consider target encoding or embedding for large cardinality.

4. **Regression Only:** Current implementation focused on regression (`Total_Revenue` prediction). Classification models defined but not fully integrated.

5. **SHAP Limitations:** SHAP analysis may not work with all model types (especially custom ensembles). Gracefully handles failures but may skip qualitative evaluation.

## Troubleshooting

### Common Issues and Solutions

**Issue: `ModuleNotFoundError: No module named 'src'`**
```bash
# Solution: Ensure you're running from project root
cd /path/to/Project
python main_pipeline.py

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Project"
```

**Issue: `FileNotFoundError: preprocessed_sales_data.csv not found`**
```bash
# Solution: Run preprocessing first
python src/data_preprocessing.py
```

**Issue: `redis.exceptions.ConnectionError`**
```
# Solution: The system automatically falls back to SQLite
# This warning is informational only. To resolve:
brew services start redis  # macOS
# OR
sudo systemctl start redis-server  # Linux
```

**Issue: `MemoryError during SHAP analysis`**
```python
# Solution: Reduce SHAP sample size in src/config.py
SHAP_SAMPLE_SIZE = 500  # Default is 1000
```

**Issue: Pipeline runs slowly on first execution**
```
# This is normal - first run has no cache
# Subsequent runs will be faster due to Redis/SQLite caching
# Expected first run: 10-15 minutes
# Expected cached runs: 2-5 minutes
```

**Issue: `ValueError: could not convert string to float`**
```bash
# Solution: Check data preprocessing completed successfully
# Verify preprocessed_sales_data.csv has proper numeric columns
python src/data_preprocessing.py
```

**Issue: Experiment ID not found in reports**
```bash
# Solution: Check reports directory for JSON files
ls -lt reports/pipeline_report_*.json | head -5
# View most recent report
cat reports/pipeline_report_<id>.json | python -m json.tool
```

### Debugging Tips

**Enable verbose logging:**
```python
# In src/config.py, change:
LOG_LEVEL = 'DEBUG'  # Default is 'INFO'
```

**Check logs for errors:**
```bash
# View recent errors in pipeline log
tail -50 logs/pipeline.log

# View all ERROR level messages
grep ERROR logs/*.log
```

**Inspect cached results:**
```python
from redis_cache import cache

# List all cached keys (if using Redis)
keys = cache.redis_client.keys('*') if cache.redis_client else []
print(keys)

# Get specific cached result
result = cache.get('experiment:<experiment_id>')
```

**Validate data quality:**
```python
import pandas as pd
df = pd.read_csv('preprocessed_sales_data.csv')

# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Verify target column exists
from src.config import TARGET_COLUMN
assert TARGET_COLUMN in df.columns
```

## References

Key documentation files:
- `README.md`: User-facing documentation and getting started guide
- `architecture_diagram.md`: Visual architecture and component descriptions
- `PIPELINE_IMPLEMENTATION_REPORT.md`: Detailed implementation report
- `K_FOLD_CV_IMPACT_REPORT.md`: Analysis of cross-validation impact
- `pipeline_implementation_requirements.md`: Infrastructure requirements
- `visualization_design.md`: Visualization approach and statistical methods
