# US Regional Sales Analysis Project

A comprehensive data analysis and machine learning project for US regional sales data. This project performs end-to-end data preprocessing, exploratory data analysis, advanced modeling with meta-learning, and generates enhanced statistical visualizations for sales pattern analysis, inventory optimization, and customer segmentation.

## 🎯 What This Project Does

This project analyzes a dataset of 7,992 US regional sales transactions across multiple sales channels (In-Store, Online, Distributor, and Wholesale). It provides:

- **Automated Data Preprocessing**: Handles missing values, outlier detection, data type conversions, and feature engineering
- **Advanced Modeling Pipeline**: Complete ML pipeline with meta-learning, continuous learning, and intelligent configuration optimization
- **Quantitative & Qualitative Evaluation**: Comprehensive evaluation including bias-variance analysis, SHAP interpretability, and business alignment checks
- **Redis Caching Layer**: High-performance caching with SQLite fallback for experiment metadata and results
- **Statistical Analysis**: Performs comprehensive exploratory data analysis with correlation analysis, distribution analysis, and statistical summaries
- **Enhanced Visualizations**: Generates advanced visualizations including model comparisons, feature importance, and interactive dashboards
- **Version Control System**: Dataset and model versioning with hash-based tracking
- **Self-Improving System**: Continuous learning loop that optimizes configurations from historical experiment data

## 🌟 Key Features

### Data Processing & Analysis
- **Multi-stage Data Cleaning Pipeline**: Univariate, multivariate, and contextual outlier detection using Z-score, IQR, Isolation Forest, and Local Outlier Factor methods
- **Derived Feature Engineering**: Automatically calculates profit margins, total revenue, and temporal lead time metrics
- **Enhanced Visualizations**: Statistical overlays including mean, median, quartiles, skewness, and kurtosis on distributions
- **Correlation Analysis**: Clustered correlation heatmaps with hierarchical clustering for identifying variable relationships

### Advanced Machine Learning Pipeline
- **Meta-Learning System**: Predicts optimal model configurations from historical experiment data using gradient boosting
- **Continuous Learning Loop**: Self-improving system that adapts strategies and uses warm starts for faster convergence
- **Multiple Model Support**: Linear/Logistic Regression, Ridge/Lasso/ElasticNet, Decision Trees, Random Forest, KNN (K-Nearest Neighbors), and ANN (Artificial Neural Networks) with automated hyperparameter tuning
- **KNN Optimal K Validation**: Comprehensive K value selection with GridSearchCV, K vs accuracy visualization, and validation range checking
- **ANN Integration**: MLPRegressor with StandardScaler preprocessing, automatic parameter handling, and cross-validation compatibility
- **Adaptive Preprocessing**: Model-specific scaling (RobustScaler for linear/distance-based models, StandardScaler for ANN)
- **Cross-Validation**: K-fold CV with comprehensive bias-variance decomposition using bootstrapping
- **Qualitative Evaluation**: SHAP-based interpretability, error pattern analysis, business rule validation, and KNN-specific visualizations

### Infrastructure & Optimization
- **Redis Cache Layer**: High-performance caching with automatic SQLite fallback for resilient operation
- **Version Control**: Hash-based dataset and model versioning with archival and comparison capabilities
- **Comprehensive Logging**: Structured experiment logging with detailed performance metrics
- **Interactive Reporting**: JSON-based pipeline reports with experiment tracking and reproducibility

### Visualization Suite
- **ML Analysis Visualizations**: Model comparison charts, feature importance plots, bias-variance analysis, and hyperparameter tuning curves
- **Interactive Dashboards**: Plotly-based interactive model performance dashboards
- **Statistical Visualizations**: Enhanced histograms, scatter plots, and correlation heatmaps with statistical overlays

## 📋 Prerequisites

- Python 3.8+ (3.9+ recommended)
- pip package manager
- Redis server (optional, SQLite fallback available)
- 16GB+ RAM recommended for full pipeline
- 50GB+ storage for datasets, models, and cached results

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SamuelMolero26/IDIS_450_Project_Group_16.git
cd IDIS_450_Project_Group_16
```

2. Create and activate a virtual environment (recommended):
```bash
# Using conda
conda create -n advanced_pipeline python=3.9
conda activate advanced_pipeline

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install and start Redis for optimal caching:
```bash
# macOS with Homebrew
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# Note: If Redis is unavailable, the system automatically falls back to SQLite
```

### Usage

#### 1. Data Preprocessing

Run the preprocessing pipeline to clean and prepare the sales data:

```bash
python src/data_preprocessing.py
```

**What it does:**
- Loads raw sales data from `Project4_USRegionalSales/Data-USRegionalSales.csv`
- Performs data quality checks and handles missing values
- Detects and flags outliers using multiple methods
- Creates derived features (profit margins, lead times, total revenue)
- Saves preprocessed data to `preprocessed_sales_data.csv`

**Output:**
- `preprocessed_sales_data.csv`: Cleaned dataset with 16 original + derived features
- Visualizations in `visualizations/preprocessing/`: sales distribution, correlations, etc.

#### 2. Run the Advanced Modeling Pipeline

Execute the complete ML pipeline with meta-learning and continuous improvement:

```bash
python main_pipeline.py
```

**What it does:**
- Loads and versions preprocessed data
- Trains multiple models (Linear/Logistic Regression, Ridge/Lasso/ElasticNet, Decision Trees, Random Forest, KNN, ANN)
- Performs KNN optimal K validation with GridSearchCV and visualization
- Applies ANN with StandardScaler preprocessing and automatic parameter handling
- Performs quantitative evaluation with bias-variance analysis
- Conducts qualitative evaluation using SHAP for interpretability
- Generates KNN-specific visualizations (K vs accuracy plots, distance analysis, prediction stability)
- Applies meta-learning to optimize configurations
- Executes continuous learning loop for self-improvement
- Generates comprehensive reports and visualizations

**Output:**
- Pipeline reports: `reports/pipeline_report_<experiment_id>.json`
- ML visualizations: `visualizations/ml_analysis/` and `visualizations/knn_analysis/`
  - Model comparison charts (comprehensive, radar plots)
  - KNN-specific visualizations: K vs accuracy plots, distance analysis, prediction stability, feature importance
  - Feature importance plots
  - Bias-variance analysis
  - Hyperparameter tuning curves
  - Interactive dashboards
- Experiment logs: `logs/pipeline.log`, `logs/model.log`, etc.
- Cached results: `cache/` directory with Redis/SQLite backend

#### 3. Generate Enhanced Statistical Visualizations

Create advanced statistical visualizations for exploratory analysis:

```bash
python utils/improved_visualizations.py
```

**What it generates:**
- Enhanced histograms with KDE, mean/median lines, quartiles, and statistical metrics
- Scatter plots with regression lines, correlation coefficients, and outlier highlighting
- Clustered correlation heatmap with hierarchical clustering
- All visualizations saved in the `visualizations/preprocessing/` directory

**Key Visualizations:**
- Distribution analysis for Order Quantity, Unit Price, Unit Cost, Profit Margin, Total Revenue
- Relationship plots: Unit Price vs Total Revenue, Unit Cost vs Profit Margin
- Statistical overlays showing measures of location, dispersion, and shape

### Example Workflows

#### Exploratory Data Analysis
```python
# Load preprocessed data
import pandas as pd
df = pd.read_csv('preprocessed_sales_data.csv')

# View data summary
print(df.describe())
print(df.columns)

# Analyze profit margins by channel
profit_by_channel = df.groupby('Sales Channel')['Profit_Margin'].agg(['mean', 'median', 'std'])
print(profit_by_channel)

# Calculate total revenue by channel
revenue_by_channel = df.groupby('Sales Channel')['Total_Revenue'].sum()
print(revenue_by_channel)
```

#### Running the Pipeline Programmatically
```python
# Import and run the complete pipeline
from src.main_pipeline import run_standard_pipeline

# Execute the pipeline
result = run_standard_pipeline()

# Access results
print(f"Experiment ID: {result['experiment_id']}")
print(f"Best Model: {result['modeling_results']['best_model']}")
print(f"Performance: {result['modeling_results']['best_performance']}")

# View qualitative insights
shap_insights = result['qualitative_results']['shap_analysis']
error_patterns = result['qualitative_results']['error_analysis']
```

#### Accessing Cached Results
```python
# Use the Redis cache layer
from redis_cache import cache

# Retrieve cached experiment results
experiment_id = "12cc04a2"
cached_result = cache.get(f"experiment:{experiment_id}")

# Get meta-learning recommendations
optimal_config = cache.get("meta_learner:optimal_config")
```

## 📊 Dataset Information

**Source:** US Regional Sales Data  
**Records:** 7,992 transactions  
**Time Period:** 2017-2018  

**Key Variables:**
- **Order Information**: OrderNumber, OrderDate, ShipDate, DeliveryDate
- **Sales Details**: Sales Channel, Order Quantity, Unit Price, Unit Cost, Discount Applied
- **Identifiers**: SalesTeamID, CustomerID, StoreID, ProductID, WarehouseCode
- **Derived Features**: Profit_Margin, Total_Revenue, Total_Lead_Time, Procurement_to_Order_Days

**Sales Channels:**
- In-Store: Direct retail purchases
- Online: E-commerce transactions
- Distributor: B2B distribution sales
- Wholesale: Bulk sales to retailers

## 📁 Project Structure

```
IDIS_450_Project_Group_16/
├── main_pipeline.py                # Main entry point for advanced modeling pipeline
├── redis_cache.py                  # Redis caching with SQLite fallback
├── requirements.txt                # Python dependencies
├── preprocessed_sales_data.csv     # Cleaned and preprocessed dataset
│
├── src/                            # Core pipeline modules
│   ├── config.py                   # Configuration management
│   ├── logger.py                   # Structured logging system
│   ├── data_loader.py              # Data loading and versioning
│   ├── data_preprocessing.py       # Data cleaning and preprocessing
│   ├── model_pipeline.py           # Core ML model training
│   ├── evaluation_engine.py        # Quantitative evaluation with bias-variance
│   ├── qualitative_evaluator.py    # SHAP analysis and business alignment
│   ├── meta_learner.py             # Meta-learning for config optimization
│   ├── version_control.py          # Dataset and model versioning
│   ├── continuous_learning.py      # Self-improvement cycle
│   └── main_pipeline.py            # Pipeline orchestrator
│
├── utils/                          # Utility modules
│   ├── improved_visualizations.py  # Enhanced statistical visualizations
│   ├── model_insights_visualization.py  # ML-specific visualizations
│   └── visualization_utils.py      # Plotting helper functions
│
├── visualizations/                 # Generated visualization outputs
│   ├── preprocessing/              # Data analysis visualizations
│   │   ├── enhanced_histogram_*.png
│   │   ├── enhanced_scatter_*.png
│   │   └── clustered_correlation_heatmap.png
│   └── ml_analysis/                # ML pipeline visualizations
│       ├── model_comparison_*.png
│       ├── feature_importance.png
│       ├── bias_variance_analysis.png
│       └── interactive_model_dashboard.html
│
├── reports/                        # Pipeline execution reports
│   └── pipeline_report_*.json      # Comprehensive experiment results
│
├── logs/                           # Structured experiment logs
│   ├── pipeline.log
│   ├── model.log
│   ├── evaluation.log
│   └── meta.log
│
├── cache/                          # Cached results and metadata
│   ├── cache.db                    # SQLite fallback cache
│   ├── learning_history.json      # Continuous learning data
│   └── versions/                   # Dataset and model versions
│
├── Project4_USRegionalSales/
│   ├── Data-USRegionalSales.csv   # Raw sales data
│   └── README.md                   # Project subdirectory readme
│
└── Documentation/
    ├── README.md                   # This file
    ├── PIPELINE_IMPLEMENTATION_REPORT.md  # Implementation details
    ├── architecture_diagram.md     # System architecture
    ├── pipeline_implementation_requirements.md
    └── visualization_design.md     # Visualization approach
```

## 🔍 Analysis Capabilities

### Data Preprocessing Features
- **Missing Value Handling**: Imputation strategies based on variable type
- **Outlier Detection**: Z-score, IQR, Isolation Forest, and Local Outlier Factor
- **Feature Engineering**: Temporal differences, profit calculations, revenue metrics
- **Data Validation**: Consistency checks for dates, prices, and quantities
- **Scaling & Encoding**: StandardScaler, MinMaxScaler, and One-Hot Encoding for categorical variables

### Machine Learning Pipeline Features
- **Model Training**: Linear/Logistic Regression, Ridge/Lasso/ElasticNet, Decision Trees, Random Forest, KNN, ANN with automated hyperparameter tuning
- **KNN Implementation**: K-Nearest Neighbors with optimal K validation, distance-based metrics, and comprehensive visualization suite
- **ANN Integration**: Multi-layer Perceptron with StandardScaler preprocessing, automatic parameter handling, and cross-validation support
- **Adaptive Preprocessing**: Model-specific scaling strategies (RobustScaler for linear/distance models, StandardScaler for ANN)
- **Cross-Validation**: K-fold CV with stratified splitting and reproducible random seeds
- **Bias-Variance Analysis**: Bootstrapping-based decomposition to understand model behavior
- **Meta-Learning**: Gradient boosting models predict optimal configurations from historical experiments
- **Continuous Learning**: Self-improving system with warm starts and adaptive strategy modification
- **Performance Tracking**: Learning curves, validation curves, and performance trend analysis

### Evaluation Framework
- **Quantitative Metrics**: MSE, RMSE, MAE, R² for regression tasks
- **Qualitative Evaluation**:
  - SHAP-based feature importance and interpretability
  - KNN-specific analysis: distance distributions, prediction stability, feature importance via permutation
  - Error pattern analysis (systematic failure detection)
  - Business rule validation (domain-specific constraints)
  - Actionable recommendations from qualitative assessment
- **KNN Optimal K Validation**: GridSearchCV results analysis, K vs accuracy visualization, validation range checking
- **Model Comparison**: Comprehensive comparison across multiple metrics with radar plots

### Visualization Features
- **Distribution Analysis**: Histograms with KDE, statistical overlays (mean, median, mode, quartiles)
- **Statistical Metrics**: Skewness, kurtosis, standard deviation, IQR calculated and displayed
- **Relationship Analysis**: Scatter plots with regression lines, correlation coefficients, and R² values
- **Correlation Networks**: Hierarchical clustering to identify variable groupings
- **ML Visualizations**: Feature importance, bias-variance decomposition, hyperparameter tuning curves
- **KNN-Specific Visualizations**: K vs accuracy plots, neighbor distance analysis, prediction stability, decision boundaries, timing analysis
- **Interactive Dashboards**: Plotly-based interactive model performance exploration

### Infrastructure Features
- **Caching**: Redis primary with automatic SQLite fallback for resilient operation
- **Version Control**: Hash-based dataset and model tracking with archival capabilities
- **Logging**: Structured experiment logging with performance metrics and error tracking
- **Reporting**: Comprehensive JSON reports with experiment reproducibility data

## 📖 Documentation

- **Pipeline Implementation Report**: See [PIPELINE_IMPLEMENTATION_REPORT.md](PIPELINE_IMPLEMENTATION_REPORT.md) for comprehensive documentation of the advanced modeling pipeline
- **Architecture Diagram**: See [architecture_diagram.md](architecture_diagram.md) for system architecture and component interactions
- **Implementation Requirements**: See [pipeline_implementation_requirements.md](pipeline_implementation_requirements.md) for infrastructure setup and dependencies
- **Visualization Design**: See [visualization_design.md](visualization_design.md) for detailed explanation of visualization approach and statistical methods
- **Data Variables**: Run `python src/data_preprocessing.py` to view detailed variable descriptions in console output
- **Project Tasks**: See [TODO.txt](TODO.txt) for planned analyses and future enhancements

## 🏗️ Advanced Pipeline Architecture

The advanced modeling pipeline follows a modular architecture with these key components:

1. **Data Pipeline**: Loading, versioning, and preprocessing with hash-based change tracking
2. **Core Modeling**: Multiple model types (Linear, Tree-based, KNN, ANN) with automated hyperparameter tuning and caching
3. **KNN Integration**: Optimal K validation with GridSearchCV, distance-based metrics, and comprehensive visualization suite
4. **ANN Integration**: MLPRegressor with StandardScaler preprocessing and automatic parameter handling
5. **Adaptive Preprocessing**: Model-specific scaling strategies and feature transformations
6. **Enhanced Evaluation**:
   - Quantitative: Cross-validation, bias-variance decomposition, learning curves
   - Qualitative: SHAP interpretability, error analysis, business alignment, KNN-specific analysis
7. **Meta-Learning**: Configuration optimization using gradient boosting on historical data
8. **Continuous Learning**: Self-improvement cycle with warm starts and adaptive strategies
9. **Infrastructure**: Redis caching with SQLite fallback, comprehensive logging, version control

See [architecture_diagram.md](architecture_diagram.md) for detailed component diagrams and data flow.

## 🤖 KNN & ANN Implementation Details

### K-Nearest Neighbors (KNN)
- **Optimal K Validation**: GridSearchCV with K values from 1-30, automatic selection based on CV performance
- **Distance Metrics**: Euclidean, Manhattan, and Minkowski distance support
- **Visualization Suite**:
  - K vs Accuracy plots with optimal K marking
  - Neighbor distance distribution analysis
  - Prediction stability across K values
  - Feature importance via permutation importance
  - Decision boundary visualization (PCA projection)
  - Timing analysis for training vs prediction
- **Validation Logic**: Range checking (5-15 expected, 3-30 acceptable) with anomaly detection

### Artificial Neural Networks (ANN)
- **Architecture**: MLPRegressor with configurable hidden layers (50-1000 neurons)
- **Preprocessing**: StandardScaler (mean=0, std=1) for optimal ANN performance
- **Parameter Handling**: Automatic random_state management to avoid conflicts
- **Hyperparameter Tuning**: GridSearchCV for hidden layers, activation functions, solvers, and learning rates
- **Cross-Validation**: Pipeline-based CV with proper scaling for each fold
- **Supported Configurations**:
  - Hidden layers: (50,), (100,), (50,50), (100,50), (100,50,25)
  - Activation: relu, tanh
  - Solvers: adam, lbfgs
  - Learning rates: constant, adaptive

## 🚀 Performance Characteristics

### Computational Requirements
- **CPU**: Multi-core CPU (4+ cores recommended) for parallel CV and bootstrapping
- **Memory**: 16GB+ RAM for in-memory processing and caching
- **Storage**: 50GB+ for datasets, models, and cached results
- **Redis**: 2GB+ allocated for caching (optional, falls back to SQLite)

### Scalability
- **Dataset Size**: Efficiently handles current dataset (8K rows)
- **Model Complexity**: Supports ensemble methods and deep decision trees
- **Caching**: Redis provides sub-millisecond access to historical results
- **Parallelization**: CV folds and bootstrapping iterations can be parallelized

### Key Metrics
- **Pipeline Runtime**: ~5-15 minutes for complete pipeline (depending on configuration)
- **Cache Hit Rate**: 70-90% on subsequent runs with similar configurations
- **Meta-Learning Accuracy**: Improves configuration prediction over time with more experiments
- **Self-Improvement**: Continuous learning shows measurable performance gains after 5-10 iterations

## 🤝 Contributing

This project implements an advanced machine learning pipeline with meta-learning, continuous improvement, and specialized KNN/ANN support. Contributions are welcome in the following areas:

- Additional model types (e.g., XGBoost, SVM, ensemble methods)
- Enhanced KNN algorithms and distance metrics
- Advanced ANN architectures and optimization techniques
- Improved meta-learning strategies
- Distributed computing support (Dask/Ray)
- Advanced interpretability methods
- Business-specific validation rules
- Performance optimizations

## 📝 Citation

If you use this project in your research or work, please cite:

```
US Regional Sales Analysis with Advanced Modeling Pipeline
IDIS 450 Project Group 16
https://github.com/SamuelMolero26/IDIS_450_Project_Group_16
```

