# US Regional Sales Analysis Project

A comprehensive data analysis and visualization project for US regional sales data. This project performs end-to-end data preprocessing, exploratory data analysis, and generates enhanced statistical visualizations for sales pattern analysis, inventory optimization, and customer segmentation.

## 🎯 What This Project Does

This project analyzes a dataset of 7,992 US regional sales transactions across multiple sales channels (In-Store, Online, Distributor, and Wholesale). It provides:

- **Automated Data Preprocessing**: Handles missing values, outlier detection, data type conversions, and feature engineering
- **Statistical Analysis**: Performs comprehensive exploratory data analysis with correlation analysis, distribution analysis, and statistical summaries
- **Advanced Visualizations**: Generates enhanced histograms, scatter plots, correlation heatmaps with statistical overlays and insights
- **Machine Learning Ready**: Produces cleaned, preprocessed datasets ready for predictive modeling tasks

## 🌟 Key Features

- **Multi-stage Data Cleaning Pipeline**: Univariate, multivariate, and contextual outlier detection using Z-score, IQR, Isolation Forest, and Local Outlier Factor methods
- **Derived Feature Engineering**: Automatically calculates profit margins, total revenue, and temporal lead time metrics
- **Enhanced Visualizations**: Statistical overlays including mean, median, quartiles, skewness, and kurtosis on distributions
- **Correlation Analysis**: Clustered correlation heatmaps with hierarchical clustering for identifying variable relationships
- **Sales Channel Comparison**: Cross-channel analysis for pricing, margins, and revenue patterns
- **Interactive and Static Outputs**: Generates both matplotlib/seaborn static images and plotly interactive HTML visualizations

## 📋 Prerequisites

- Python 3.7+
- pip package manager

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SamuelMolero26/IDIS_450_Project_Group_16.git
cd IDIS_450_Project_Group_16
```

2. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly pingouin
```

### Usage

#### 1. Data Preprocessing

Run the preprocessing pipeline to clean and prepare the sales data:

```bash
python data_preprocessing.py
```

**What it does:**
- Loads raw sales data from `Project4_USRegionalSales/Data-USRegionalSales.csv`
- Performs data quality checks and handles missing values
- Detects and flags outliers using multiple methods
- Creates derived features (profit margins, lead times, total revenue)
- Saves preprocessed data to `preprocessed_sales_data.csv`

**Output:**
- `preprocessed_sales_data.csv`: Cleaned dataset with 16 original + derived features
- `sales_channel_distribution.png`: Bar chart of sales by channel
- `unit_price_distribution.png`: Distribution of unit prices
- `correlation_heatmap.png`: Correlation matrix of numeric variables

#### 2. Generate Enhanced Visualizations

Create advanced statistical visualizations:

```bash
python improved_visualizations.py
```

**What it generates:**
- Enhanced histograms with KDE, mean/median lines, quartiles, and statistical metrics
- Scatter plots with regression lines, correlation coefficients, and outlier highlighting
- Clustered correlation heatmap with hierarchical clustering
- All visualizations saved in the `visualizations/` directory

**Key Visualizations:**
- Distribution analysis for Order Quantity, Unit Price, Unit Cost, Profit Margin, Total Revenue
- Relationship plots: Unit Price vs Total Revenue, Unit Cost vs Profit Margin
- Statistical overlays showing measures of location, dispersion, and shape

### Example Workflow

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
├── data_preprocessing.py           # Main data cleaning and preprocessing script
├── improved_visualizations.py      # Enhanced visualization generation script
├── visualization_design.md         # Documentation of visualization approach
├── TODO.txt                        # Project task list
├── Project4_USRegionalSales/
│   ├── Data-USRegionalSales.csv   # Raw sales data
│   └── README.md                   # Project subdirectory readme
├── visualizations/                 # Generated visualization outputs
│   ├── enhanced_histogram_*.png
│   ├── enhanced_scatter_*.png
│   └── clustered_correlation_heatmap.png
├── preprocessed_sales_data.csv    # Cleaned and preprocessed dataset
└── README.md                       # This file
```

## 🔍 Analysis Capabilities

### Data Preprocessing Features
- **Missing Value Handling**: Imputation strategies based on variable type
- **Outlier Detection**: Z-score, IQR, Isolation Forest, and Local Outlier Factor
- **Feature Engineering**: Temporal differences, profit calculations, revenue metrics
- **Data Validation**: Consistency checks for dates, prices, and quantities
- **Scaling & Encoding**: StandardScaler, MinMaxScaler, and One-Hot Encoding for categorical variables

### Visualization Features
- **Distribution Analysis**: Histograms with KDE, statistical overlays (mean, median, mode, quartiles)
- **Statistical Metrics**: Skewness, kurtosis, standard deviation, IQR calculated and displayed
- **Relationship Analysis**: Scatter plots with regression lines, correlation coefficients, and R² values
- **Correlation Networks**: Hierarchical clustering to identify variable groupings
- **Faceted Views**: Channel-wise comparisons for key metrics

## 📖 Documentation

- **Visualization Design**: See [visualization_design.md](visualization_design.md) for detailed explanation of visualization approach and statistical methods
- **Data Variables**: Run `python data_preprocessing.py` to view detailed variable descriptions in console output
- **Project Tasks**: See [TODO.txt](TODO.txt) for planned analyses and visualizations

