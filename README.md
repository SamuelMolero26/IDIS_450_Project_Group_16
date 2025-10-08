# US Regional Sales Data Analysis Project

## Project Overview

This project provides comprehensive data preprocessing, exploratory data analysis (EDA), and advanced visualizations for a US Regional Sales dataset. The analysis pipeline is designed to prepare data for machine learning tasks including sales prediction, customer segmentation, and channel classification.

## Dataset Description

The dataset contains **7,992 transactions** across various sales channels in the US region. Each transaction includes detailed information about:

- **Order Information**: Order numbers, dates, and quantities
- **Sales Channels**: In-Store, Online, Distributor, and Wholesale
- **Pricing Details**: Unit costs, unit prices, and discount rates
- **Temporal Data**: Procurement, order, shipment, and delivery dates
- **Business Entities**: Warehouses, sales teams, customers, stores, and products

### Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| OrderNumber | String | Unique identifier for each sales order |
| Sales Channel | Categorical | Channel through which sale was made (In-Store, Online, Distributor, Wholesale) |
| WarehouseCode | String | Warehouse identifier |
| ProcuredDate | DateTime | Date when product was procured |
| OrderDate | DateTime | Date when order was placed |
| ShipDate | DateTime | Date when product was shipped |
| DeliveryDate | DateTime | Date when product was delivered |
| Order Quantity | Integer | Quantity of products ordered |
| Discount Applied | Float | Discount percentage (0.0 to 1.0) |
| Unit Cost | Float | Cost price per unit |
| Unit Price | Float | Selling price per unit |

### Derived Features

The preprocessing pipeline creates several derived features for enhanced analysis:

- **Temporal Features**:
  - `Procurement_to_Order_Days`: Days between procurement and order
  - `Order_to_Ship_Days`: Days between order and shipment
  - `Ship_to_Delivery_Days`: Days between shipment and delivery
  - `Total_Lead_Time`: Total days from procurement to delivery

- **Financial Metrics**:
  - `Profit_Margin`: Calculated as `((Unit Price × (1 - Discount)) - Unit Cost) / Unit Cost`
  - `Total_Revenue`: Calculated as `Order Quantity × Unit Price × (1 - Discount)`

## Project Objectives

1. **Data Quality Assessment**: Identify and handle missing values, outliers, and data inconsistencies
2. **Feature Engineering**: Create derived temporal and financial features
3. **Exploratory Data Analysis**: Understand distributions, correlations, and patterns
4. **Outlier Detection**: Apply univariate, multivariate, and contextual outlier detection methods
5. **Data Preparation**: Scale numerical features and encode categorical variables for ML models
6. **Advanced Visualization**: Generate comprehensive statistical visualizations

## Project Structure

```
IDIS_450_Project_Group_16/
├── README.md                           # This file
├── data_preprocessing.py               # Main preprocessing and EDA pipeline
├── improved_visualizations.py          # Enhanced visualization generation
├── visualization_design.md             # Visualization design documentation
├── TODO.txt                            # Project task list
├── preprocessed_sales_data.csv         # Cleaned and preprocessed dataset
├── Project4_USRegionalSales/
│   ├── Data-USRegionalSales.csv       # Original dataset
│   ├── DataDescription-USRegionalSales.docx
│   └── README.md
└── visualizations/                     # Generated visualization outputs
    ├── enhanced_histogram_*.png
    ├── enhanced_scatter_*.png
    ├── clustered_correlation_heatmap.png
    └── *.jpg
```

## Technologies Used

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Statistical Analysis**: scipy
- **Outlier Detection**: Isolation Forest, Local Outlier Factor (LOF)

## Setup and Installation

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly pingouin
```

### Running the Analysis

1. **Data Preprocessing and EDA**:
   ```bash
   python data_preprocessing.py
   ```
   This script will:
   - Load the raw dataset
   - Perform initial exploration
   - Check for missing values and inconsistencies
   - Create derived features
   - Detect outliers using multiple methods
   - Apply scaling and encoding
   - Generate the preprocessed dataset

2. **Generate Enhanced Visualizations**:
   ```bash
   python improved_visualizations.py
   ```
   This script will:
   - Create enhanced histograms with statistical overlays
   - Generate scatter plots with regression lines
   - Build clustered correlation heatmaps
   - Save all visualizations to the `visualizations/` directory

## Key Features

### Data Preprocessing Pipeline

1. **Data Loading**: Properly parses CSV with date columns and numeric fields
2. **Missing Value Analysis**: Comprehensive check for missing data
3. **Data Consistency Validation**: Ensures data integrity across temporal and numerical fields
4. **Feature Engineering**: Creates temporal differences and financial metrics
5. **Outlier Detection**:
   - **Univariate**: Z-score and IQR methods
   - **Multivariate**: Isolation Forest and Local Outlier Factor
   - **Contextual**: Sales channel and temporal pattern analysis
6. **Data Transformation**: StandardScaler for numerical features, OneHotEncoder for categorical variables
7. **Balance Assessment**: Evaluates class distributions for ML readiness

### Visualization Suite

#### Enhanced Histograms
- Optimized binning using Freedman-Diaconis rule
- Kernel Density Estimate (KDE) overlays
- Statistical annotations (mean, median, mode, standard deviation, skewness, kurtosis)
- Quartile shading
- Faceted by Sales Channel for key financial variables

#### Enhanced Scatter Plots
- Color-coded by Sales Channel
- Regression lines with confidence intervals
- Correlation coefficient annotations
- Outlier highlighting
- Marginal distributions

#### Advanced Correlation Analysis
- Clustered correlation heatmap with hierarchical clustering
- Partial correlation matrices
- Multiple correlation computation methods

## Visualizations

The project generates comprehensive visualizations including:

- **Distribution Analysis**: Enhanced histograms for all numeric variables
- **Relationship Analysis**: Scatter plots for key variable pairs
- **Correlation Analysis**: Heatmaps showing variable relationships
- **Channel Comparison**: Profit margin and revenue by sales channel

All visualizations are saved in high resolution (300 DPI) in the `visualizations/` directory.

## Statistical Insights

The analysis provides insights into:

- **Location Measures**: Mean, median, and mode for all numeric variables
- **Dispersion Measures**: Standard deviation, variance, range, and IQR
- **Shape Measures**: Skewness and kurtosis coefficients
- **Correlations**: Pearson, Spearman, and partial correlations
- **Distribution Patterns**: Identifies skewness and outliers in sales data
- **Channel Differences**: Compares pricing strategies and profit margins across channels
- **Temporal Patterns**: Analyzes lead times and delivery efficiency

## Machine Learning Readiness

The preprocessed dataset (`preprocessed_sales_data.csv`) is ready for:

- **Sales Forecasting**: Predict future sales using temporal and pricing features
- **Customer Segmentation**: Cluster customers based on purchasing behavior
- **Channel Classification**: Predict optimal sales channel for products
- **Profit Optimization**: Identify factors that maximize profit margins
- **Inventory Management**: Forecast demand and optimize stock levels

## Team Information

**Course**: IDIS 450  
**Group**: 16  
**Institution**: Project Group at IDIS 450

## Future Enhancements

- Time series analysis for seasonal patterns
- Customer lifetime value prediction
- Automated reporting dashboard
- Real-time data integration
- Advanced ML model implementation (Random Forest, XGBoost, Neural Networks)

## License

This project is part of an academic course and is intended for educational purposes.

## Contact

For questions or contributions, please contact the project team through the course repository.

---

*Last Updated: 2024*
