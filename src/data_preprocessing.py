# Data Preprocessing and Exploratory Data Analysis for US Regional Sales Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from scipy import stats
import warnings
import os
from pathlib import Path
import argparse
import sys
import subprocess

try:
    from src import config
except Exception:
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
         sys.path.insert(0, project_root)
    import config as config

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')


def find_data_file(data_arg: str | None = None) -> str:
    """Locate the data CSV by checking common repo-relative locations, a provided arg, or env var.

    Returns the file path as a string if found, otherwise raises FileNotFoundError with helpful guidance.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []

    # CLI-provided path
    if data_arg:
        candidates.append(Path(data_arg))

    # Environment variable
    env_path = os.environ.get('DATA_PATH')
    if env_path:
        candidates.append(Path(env_path))

    
    candidates.extend([
        repo_root / 'Project4_USRegionalSales' / 'Data-USRegionalSales.csv',
        repo_root / 'Project4_USRegionalSales' / 'Data-USRegionalSales.yaml',
        repo_root / 'Project4_USRegionalSales' / 'Data-USRegionalSales.csv',
        repo_root / 'Data-USRegionalSales.csv',
        Path.cwd() / 'Project4_USRegionalSales' / 'Data-USRegionalSales.csv',
        Path.cwd() / 'Data-USRegionalSales.csv'
    ])

    for p in candidates:
        if p and p.exists():
            return str(p)

    tried = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find 'Data-USRegionalSales.csv'.\nTried the following locations:\n" + tried +
        "\n\nProvide the path with --data /full/path/to/Data-USRegionalSales.csv or set DATA_PATH env var."
    )


def run_improved_visualizations(preprocessed_csv_path: Path):
    """Locate `utils/improved_visualizations.py` and execute it in the directory of the CSV.

    The function will be skipped if the environment variable RUN_IMPROVED_VIZ is set to '0'.
    """
    # Respect opt-out
    if os.environ.get('RUN_IMPROVED_VIZ', '1') == '0':
        print("RUN_IMPROVED_VIZ=0 -> skipping improved visualizations")
        return

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / 'utils' / 'improved_visualizations.py',
        repo_root / 'src' / 'utils' / 'improved_visualizations.py',
        repo_root / 'utils' / 'visualizations' / 'improved_visualizations.py'
    ]

    viz_script = None
    for c in candidates:
        if c.exists():
            viz_script = c
            break

    if viz_script is None:
        raise FileNotFoundError(f"improved_visualizations.py not found. Tried: {candidates}")

    # Run the visualization script in the directory containing the CSV so relative paths like
    # 'visualizations/' and 'preprocessed_sales_data.csv' resolve correctly.
    work_dir = preprocessed_csv_path.parent

    cmd = [sys.executable, str(viz_script)]

    print(f"Running improved visualizations: {viz_script} (cwd={work_dir})")
    subprocess.check_call(cmd, cwd=str(work_dir))


def load_data(file_path):
    """
    Load the US Regional Sales dataset with proper parsing.
    """
    date_columns = ['ProcuredDate', 'OrderDate', 'ShipDate', 'DeliveryDate']

    df = pd.read_csv(file_path,
                     thousands=',',  # Handle commas in numeric fields
                     parse_dates=date_columns,
                     dayfirst=True)  # DD-MM-YYYY format

    return df

def initial_exploration(df):
    """
    Perform initial exploration of the dataset.
    """
    print("\n=== INITIAL DATA EXPLORATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nSummary statistics:\n{df.describe(include='all')}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    
    #get context on how much cleaning is needed
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")

    return df

def explain_data():
    """
    Provide detailed explanation of the data and variables.
    """
    print("\n=== DATA AND VARIABLE EXPLANATION ===")
    print("""
    PROBLEM CONTEXT:
    This dataset contains US regional sales data with 7,992 transactions across various sales channels.
    The primary objective is to analyze sales patterns, optimize inventory, improve customer relationships,
    and forecast future sales. This preprocessing pipeline prepares the data for machine learning tasks
    such as sales prediction, customer segmentation, and channel classification.

    DETAILED VARIABLE DESCRIPTIONS:

    1. OrderNumber (string, categorical): Unique identifier for each sales order (e.g., 'SO - 000101')
       - Type: Categorical (nominal)
       - Relevance: Primary key for transaction identification, useful for deduplication and tracking

    2. Sales Channel (string, categorical): Channel through which sale was made
       - Type: Categorical (nominal)
       - Values: In-Store, Online, Distributor, Wholesale
       - Relevance: Critical for sales analysis, channel performance comparison, and customer segmentation

    3. WarehouseCode (string, categorical): Warehouse identifier (e.g., 'WARE-UHY1004')
       - Type: Categorical (nominal)
       - Relevance: Enables regional analysis, inventory management, and supply chain optimization

    4. ProcuredDate (datetime, temporal): When product was procured
       - Type: DateTime
       - Relevance: Foundation for temporal analysis, lead time calculations, and procurement optimization

    5. OrderDate (datetime, temporal): When order was placed
       - Type: DateTime
       - Relevance: Key temporal marker for sales cycle analysis and forecasting

    6. ShipDate (datetime, temporal): When product was shipped
       - Type: DateTime
       - Relevance: Enables shipping time analysis and delivery performance metrics

    7. DeliveryDate (datetime, temporal): When product was delivered
       - Type: DateTime
       - Relevance: Critical for customer satisfaction analysis and delivery time optimization

    8. CurrencyCode (string, categorical): Transaction currency (primarily USD)
       - Type: Categorical (nominal)
       - Relevance: Important for multi-currency analysis, though mostly uniform in this dataset

    9. _SalesTeamID (int, categorical): Sales team identifier
       - Type: Categorical (nominal)
       - Relevance: Enables sales team performance analysis and territory management

    10. _CustomerID (int, categorical): Customer identifier
        - Type: Categorical (nominal)
        - Relevance: Essential for customer segmentation, lifetime value analysis, and personalization

    11. _StoreID (int, categorical): Store identifier
        - Type: Categorical (nominal)
        - Relevance: Supports store-level performance analysis and location-based insights

    12. _ProductID (int, categorical): Product identifier
        - Type: Categorical (nominal)
        - Relevance: Enables product performance analysis, category insights, and recommendation systems

    13. Order Quantity (int, numerical): Quantity of products ordered
        - Type: Numerical (discrete)
        - Relevance: Key metric for sales volume analysis, inventory management, and demand forecasting

    14. Discount Applied (float, numerical): Discount percentage (0.0 to 1.0)
        - Type: Numerical (continuous)
        - Relevance: Critical for pricing strategy analysis, margin optimization, and discount effectiveness

    15. Unit Cost (float, numerical): Cost price per unit
        - Type: Numerical (continuous)
        - Relevance: Essential for profit margin calculations, cost control, and pricing decisions

    16. Unit Price (float, numerical): Selling price per unit
        - Type: Numerical (continuous)
        - Relevance: Primary revenue driver, enables pricing analysis and market positioning

    DERIVED FEATURES (added during preprocessing):
    - Procurement_to_Order_Days (int, numerical): Days between procurement and order
    - Order_to_Ship_Days (int, numerical): Days between order and shipping
    - Ship_to_Delivery_Days (int, numerical): Days between shipping and delivery
    - Total_Lead_Time (int, numerical): Total days from procurement to delivery
    - Profit_Margin (float, numerical): Profit margin percentage
    - Total_Revenue (float, numerical): Total revenue per transaction

    KEY INSIGHTS FOR ML IMPLEMENTATION:
    1. Sales Analysis: Use Sales Channel, temporal features, and revenue metrics for pattern recognition
    2. Inventory Management: Leverage Order Quantity, temporal features, and WarehouseCode for optimization
    3. Customer Segmentation: Apply clustering on CustomerID, purchasing behavior, and channel preferences
    4. Revenue Forecasting: Use time series analysis on temporal features and revenue metrics
    5. Discount Effectiveness: Analyze Discount Applied correlations with Total_Revenue and Profit_Margin
    """)

def check_missing_values(df):
    """
    Check for missing values and handle them appropriately.
    Enhanced with robust imputation and extreme value handling.
    """
    print("\n=== MISSING VALUES ANALYSIS ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_percent
    })

    print(missing_df[missing_df['Missing Count'] > 0])

    if missing.sum() == 0:
        print("No missing values found in the dataset.")
        return df

    # Enhanced handling of missing values with robust imputation
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100

            if df[col].dtype in ['int64', 'float64']:
                # Enhanced numerical imputation based on missing percentage
                if missing_pct < 5:
                    # Low missing rate - use median imputation
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                    print(f"Imputed missing values in {col} with median (low missing rate: {missing_pct:.1f}%).")
                elif missing_pct < 30:
                    # Moderate missing rate - use iterative imputation if available
                    try:
                        from sklearn.experimental import enable_iterative_imputer
                        from sklearn.impute import IterativeImputer

                        # Use iterative imputation for better accuracy
                        imputer = IterativeImputer(random_state=42, max_iter=10)
                        df[col] = imputer.fit_transform(df[[col]]).ravel()
                        print(f"Imputed missing values in {col} with iterative imputation (moderate missing rate: {missing_pct:.1f}%).")
                    except ImportError:
                        # Fallback to median if iterative imputer not available
                        imputer = SimpleImputer(strategy='median')
                        df[col] = imputer.fit_transform(df[[col]]).ravel()
                        print(f"Imputed missing values in {col} with median (iterative imputer not available, missing rate: {missing_pct:.1f}%).")
                else:
                    # High missing rate - consider dropping or using model-based imputation
                    print(f"High missing rate in {col} ({missing_pct:.1f}%). Consider feature engineering or removal.")
                    # For now, use median but log warning
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                    print(f"Temporarily imputed missing values in {col} with median due to high missing rate.")

            elif df[col].dtype == 'object':
                # For categorical - use mode imputation with frequency check
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                    mode_pct = (df[col] == mode_value.iloc[0]).sum() / len(df) * 100
                    print(f"Imputed missing values in {col} with mode '{mode_value.iloc[0]}' (mode frequency: {mode_pct:.1f}%).")
                else:
                    # If no mode, use a placeholder
                    df[col] = df[col].fillna('Unknown')
                    print(f"Imputed missing values in {col} with 'Unknown' (no clear mode found).")

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # For dates - use forward fill, then backward fill as fallback
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                print(f"Imputed missing values in {col} with forward/backward fill.")

    # Additional check: ensure no missing values remain
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values still remain after imputation.")
    else:
        print("All missing values successfully handled.")

    return df

def handle_extreme_values(df, method='iqr', threshold=3.0):
    """
    Handle extreme values (outliers) using robust methods.

    Args:
        df: Input DataFrame
        method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with extreme values handled
    """
    print(f"\n=== EXTREME VALUES HANDLING (Method: {method}) ===")

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    processed_cols = []

    for col in numerical_cols:
        original_values = df[col].copy()
        n_extreme = 0

        if method == 'iqr':
            # IQR method - more robust than z-score for skewed distributions
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Cap extreme values
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            n_extreme = ((original_values < lower_bound) | (original_values > upper_bound)).sum()

        elif method == 'zscore':
            # Z-score method with robust scaling
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            outlier_mask = z_scores > threshold

            # Cap at percentile bounds
            lower_cap = df[col].quantile(0.01)
            upper_cap = df[col].quantile(0.99)

            df.loc[outlier_mask, col] = np.clip(df.loc[outlier_mask, col], lower_cap, upper_cap)
            n_extreme = outlier_mask.sum()

        elif method == 'isolation_forest':
            # Use Isolation Forest for multivariate outlier detection
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(df[[col]])

                # For univariate case, cap extreme values
                outlier_mask = outlier_pred == -1
                lower_cap = df[col].quantile(0.05)
                upper_cap = df[col].quantile(0.95)

                df.loc[outlier_mask, col] = np.clip(df.loc[outlier_mask, col], lower_cap, upper_cap)
                n_extreme = outlier_mask.sum()

            except Exception as e:
                print(f"Isolation Forest failed for {col}: {e}. Using IQR method.")
                # Fallback to IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                n_extreme = ((original_values < lower_bound) | (original_values > upper_bound)).sum()

        if n_extreme > 0:
            processed_cols.append(col)
            pct_extreme = (n_extreme / len(df)) * 100
            print(f"Handled {n_extreme} extreme values in {col} ({pct_extreme:.2f}%) using {method} method.")

    if processed_cols:
        print(f"Extreme values handled in {len(processed_cols)} numerical columns.")
    else:
        print("No extreme values detected requiring correction.")

    return df

def check_data_consistency(df):
    """
    Check for data errors and inconsistencies.
    """
    
    print("\n=== DATA CONSISTENCY CHECK ===")

    # Check date logic
    invalid_dates = df[df['ProcuredDate'] > df['OrderDate']]
    if len(invalid_dates) > 0:
        print(f"Found {len(invalid_dates)} orders where ProcuredDate > OrderDate")
        # Could fix by swapping or flagging

    invalid_ship = df[df['OrderDate'] > df['ShipDate']]
    if len(invalid_ship) > 0:
        print(f"Found {len(invalid_ship)} orders where OrderDate > ShipDate")

    invalid_delivery = df[df['ShipDate'] > df['DeliveryDate']]
    if len(invalid_delivery) > 0:
        print(f"Found {len(invalid_delivery)} orders where ShipDate > DeliveryDate")

    # Check negative values
    negative_cols = ['Order Quantity', 'Unit Cost', 'Unit Price']
    for col in negative_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"Found {neg_count} negative values in {col}")

    # Check discount range
    invalid_discount = ((df['Discount Applied'] < 0) | (df['Discount Applied'] > 1)).sum()
    if invalid_discount > 0:
        print(f"Found {invalid_discount} discount values outside [0,1] range")

    # Check currency consistency <-- only one USD
    # currency_counts = df['CurrencyCode'].value_counts()
    # print(f"Currency distribution:\n{currency_counts}")

    # if len(currency_counts) > 1:
    #     print("Multiple currencies found - may need conversion for analysis")

    print("Data consistency check completed.")
    
    return df

def convert_data_types(df):
    """
    Convert and clean data types, add derived features.
    """
    print("\n=== DATA TYPE CONVERSION AND FEATURE ENGINEERING ===")

    # Ensure numeric columns are properly typed
    numeric_cols = ['Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add derived temporal features
    df['Procurement_to_Order_Days'] = (df['OrderDate'] - df['ProcuredDate']).dt.days
    df['Order_to_Ship_Days'] = (df['ShipDate'] - df['OrderDate']).dt.days
    df['Ship_to_Delivery_Days'] = (df['DeliveryDate'] - df['ShipDate']).dt.days
    df['Total_Lead_Time'] = (df['DeliveryDate'] - df['ProcuredDate']).dt.days

    # Add profit margin
    df['Profit_Margin'] = ((df['Unit Price'] * (1 - df['Discount Applied'])) - df['Unit Cost']) / df['Unit Cost']

    # Add total revenue
    df['Total_Revenue'] = df['Order Quantity'] * df['Unit Price'] * (1 - df['Discount Applied'])

    print("Added derived features: temporal differences, profit margin, total revenue")
    print(f"New shape: {df.shape}")

    return df

def univariate_outlier_detection(df):
    """
    Perform aggressive univariate outlier detection and removal using Z-score and IQR methods.
    """
    print("\n=== AGGRESSIVE UNIVARIATE OUTLIER DETECTION AND REMOVAL ===")

    numeric_cols = ['Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price',
                   'Procurement_to_Order_Days', 'Order_to_Ship_Days', 'Ship_to_Delivery_Days',
                   'Total_Lead_Time', 'Profit_Margin', 'Total_Revenue']

    outlier_summary = {}
    initial_rows = len(df)

    for col in numeric_cols:
        if col in df.columns:
            # Z-score method (z < 4 for aggressive cleaning)
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_outliers = (z_scores > 4).sum()

            # IQR method (3×IQR for aggressive cleaning)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            outlier_summary[col] = {
                'Z-score outliers': z_outliers,
                'IQR outliers': iqr_outliers,
                'Percentage': (iqr_outliers / len(df)) * 100,
                'IQR bounds': (lower_bound, upper_bound)
            }

            print(f"{col}: {iqr_outliers} outliers ({outlier_summary[col]['Percentage']:.2f}%) - IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Remove outliers using both Z-score and IQR methods
    print("\n=== REMOVING OUTLIERS ===")
    for col in numeric_cols:
        if col in df.columns:
            # Remove using Z-score (z < 4)
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_mask = z_scores <= 4
            
            # Remove using IQR (3×IQR)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            iqr_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            
            # Combine masks (keep rows that pass both tests)
            combined_mask = z_mask & iqr_mask
            rows_before = len(df)
            df = df[combined_mask]
            rows_removed = rows_before - len(df)
            
            if rows_removed > 0:
                print(f"  Removed {rows_removed} rows based on {col} outliers")

    # Remove missing/infinite values
    print("\n=== REMOVING MISSING AND INFINITE VALUES ===")
    rows_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    rows_removed = rows_before - len(df)
    print(f"  Removed {rows_removed} rows with missing or infinite values")

    # Remove duplicates
    print("\n=== REMOVING DUPLICATES ===")
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_removed = rows_before - len(df)
    print(f"  Removed {rows_removed} duplicate rows")

    final_rows = len(df)
    total_removed = initial_rows - final_rows
    print(f"\n=== DATA QUALITY SUMMARY ===")
    print(f"  Initial rows: {initial_rows}")
    print(f"  Final rows: {final_rows}")
    print(f"  Total removed: {total_removed} ({(total_removed/initial_rows)*100:.2f}%)")
    print(f"  Data retention: {(final_rows/initial_rows)*100:.2f}%")

    return df

def multivariate_outlier_detection(df):
    """
    Perform multivariate outlier detection using Isolation Forest and LOF.
    """
    print("\n=== MULTIVARIATE OUTLIER DETECTION ===")

    # Select numeric features for multivariate analysis
    feature_cols = ['Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price',
                   'Procurement_to_Order_Days', 'Order_to_Ship_Days', 'Ship_to_Delivery_Days']

    # Remove any NaN values for analysis
    df_clean = df[feature_cols].dropna()

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_outliers = iso_forest.fit_predict(df_clean)
    df['iso_forest_outlier'] = (iso_outliers == -1).astype(int)

    iso_outlier_count = (iso_outliers == -1).sum()
    print(f"Isolation Forest detected {iso_outlier_count} outliers ({iso_outlier_count/len(df_clean)*100:.2f}%)")

    # Local Outlier Factor (sample for performance)
    sample_size = min(1000, len(df_clean))
    df_sample = df_clean.sample(n=sample_size, random_state=42)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_outliers = lof.fit_predict(df_sample)
    lof_outlier_count = (lof_outliers == -1).sum()
    print(f"LOF detected {lof_outlier_count} outliers in sample ({lof_outlier_count/sample_size*100:.2f}%)")

    return df

def contextual_outlier_detection(df):
    """
    Consider contextual outliers based on sales channel and temporal patterns.
    """
    print("\n=== CONTEXTUAL OUTLIER DETECTION ===")

    # Outliers by sales channel
    channels = df['Sales Channel'].unique()
    for channel in channels:
        channel_data = df[df['Sales Channel'] == channel]
        mean_price = channel_data['Unit Price'].mean()
        std_price = channel_data['Unit Price'].std()

        if std_price > 0:
            channel_outliers = ((channel_data['Unit Price'] - mean_price).abs() > 3 * std_price).sum()
            print(f"{channel}: {channel_outliers} price outliers")

    # Temporal outliers - unusual lead times
    lead_time_cols = ['Procurement_to_Order_Days', 'Order_to_Ship_Days', 'Ship_to_Delivery_Days']
    for col in lead_time_cols:
        if col in df.columns:
            # Negative lead times are errors
            negative_lead = (df[col] < 0).sum()
            if negative_lead > 0:
                print(f"{col}: {negative_lead} negative values (data errors)")

            # Very long lead times
            long_lead = (df[col] > df[col].quantile(0.99)).sum()
            print(f"{col}: {long_lead} unusually long lead times")

    return df

def create_interaction_features(df):
    """
    Create interaction features to capture relationships between variables.
    This helps reduce bias by allowing the model to learn complex patterns.
    """
    print("\n=== CREATING INTERACTION FEATURES ===")
    
    # Key interaction features based on business logic
    interactions = []
    
    # Price × Quantity interactions (revenue-related)
    if 'Unit Price' in df.columns and 'Order Quantity' in df.columns:
        df['Price_Quantity_Interaction'] = df['Unit Price'] * df['Order Quantity']
        interactions.append('Price_Quantity_Interaction')
        print("  Created: Price × Quantity interaction")
    
    # Cost × Quantity interactions
    if 'Unit Cost' in df.columns and 'Order Quantity' in df.columns:
        df['Cost_Quantity_Interaction'] = df['Unit Cost'] * df['Order Quantity']
        interactions.append('Cost_Quantity_Interaction')
        print("  Created: Cost × Quantity interaction")
    
    # Discount × Price interactions (discount impact)
    if 'Discount Applied' in df.columns and 'Unit Price' in df.columns:
        df['Discount_Price_Interaction'] = df['Discount Applied'] * df['Unit Price']
        interactions.append('Discount_Price_Interaction')
        print("  Created: Discount × Price interaction")
    
    # Profit Margin × Quantity (profitability)
    if 'Profit_Margin' in df.columns and 'Order Quantity' in df.columns:
        df['Margin_Quantity_Interaction'] = df['Profit_Margin'] * df['Order Quantity']
        interactions.append('Margin_Quantity_Interaction')
        print("  Created: Margin × Quantity interaction")
    
    # Lead Time × Quantity (operational efficiency)
    if 'Total_Lead_Time' in df.columns and 'Order Quantity' in df.columns:
        df['LeadTime_Quantity_Interaction'] = df['Total_Lead_Time'] * df['Order Quantity']
        interactions.append('LeadTime_Quantity_Interaction')
        print("  Created: Lead Time × Quantity interaction")
    
    # Price-Cost ratio (markup indicator)
    if 'Unit Price' in df.columns and 'Unit Cost' in df.columns:
        df['Price_Cost_Ratio'] = df['Unit Price'] / (df['Unit Cost'] + 1e-10)  # Avoid division by zero
        interactions.append('Price_Cost_Ratio')
        print("  Created: Price/Cost ratio")
    
    # Discount effectiveness (discount × margin)
    if 'Discount Applied' in df.columns and 'Profit_Margin' in df.columns:
        df['Discount_Margin_Interaction'] = df['Discount Applied'] * df['Profit_Margin']
        interactions.append('Discount_Margin_Interaction')
        print("  Created: Discount × Margin interaction")
    
    print(f"\nTotal interaction features created: {len(interactions)}")
    print(f"New shape after interactions: {df.shape}")

    return df


def detect_skewness_and_transform(df, skewness_threshold=1.0, transform_method='auto'):
    """
    Detect skewed numerical features and apply appropriate transformations.

    Args:
        df: Input DataFrame
        skewness_threshold: Threshold for skewness detection (default 1.0)
        transform_method: 'auto', 'log', 'yeo-johnson', or 'sqrt'

    Returns:
        DataFrame with transformed features
    """
    print(f"\n=== SKEWNESS DETECTION AND TRANSFORMATION (Method: {transform_method}) ===")

    df_transformed = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    transformed_features = []

    for col in numerical_cols:
        # Skip if constant or has negative values for log transform
        if df[col].nunique() <= 1:
            continue

        skewness = stats.skew(df[col].dropna())

        if abs(skewness) > skewness_threshold:
            original_skew = skewness
            try:
                if transform_method == 'auto':
                    # Auto-select based on data characteristics
                    if (df[col] > 0).all():
                        # All positive - use Yeo-Johnson (more flexible than log)
                        transformer = PowerTransformer(method='yeo-johnson')
                        transformed_values = transformer.fit_transform(df[[col]]).ravel()
                        method_used = 'yeo-johnson'
                    else:
                        # Contains zeros/negatives - use Yeo-Johnson anyway
                        transformer = PowerTransformer(method='yeo-johnson')
                        transformed_values = transformer.fit_transform(df[[col]]).ravel()
                        method_used = 'yeo-johnson'
                elif transform_method == 'log':
                    if (df[col] > 0).all():
                        transformed_values = np.log1p(df[col])
                        method_used = 'log'
                    else:
                        print(f"  Skipping log transform for {col} - contains non-positive values")
                        continue
                elif transform_method == 'yeo-johnson':
                    transformer = PowerTransformer(method='yeo-johnson')
                    transformed_values = transformer.fit_transform(df[[col]]).ravel()
                    method_used = 'yeo-johnson'
                elif transform_method == 'sqrt':
                    min_val = df[col].min()
                    if min_val < 0:
                        transformed_values = np.sqrt(df[col] - min_val + 1)
                    else:
                        transformed_values = np.sqrt(df[col] + 1)
                    method_used = 'sqrt'
                else:
                    continue

                # Check if transformation improved skewness
                new_skewness = stats.skew(transformed_values)
                if abs(new_skewness) < abs(original_skew):
                    new_col_name = f"{col}_transformed"
                    df_transformed[new_col_name] = transformed_values
                    transformed_features.append(col)
                    print(f"  Transformed {col}: skewness {original_skew:.2f} → {new_skewness:.2f} ({method_used})")
                else:
                    print(f"  Skipped {col}: transformation did not improve skewness")

            except Exception as e:
                print(f"  Failed to transform {col}: {e}")
                continue

    if transformed_features:
        print(f"Successfully transformed {len(transformed_features)} skewed features")
    else:
        print("No features were transformed")

    return df_transformed


def apply_winsorization(df, limits=(0.05, 0.05)):
    """
    Apply winsorization to handle extreme outliers.

    Args:
        df: Input DataFrame
        limits: Tuple of (lower_limit, upper_limit) as fractions (default 0.05 for both)

    Returns:
        DataFrame with winsorized features
    """
    print(f"\n=== WINSORIZATION (Limits: {limits}) ===")

    df_winsorized = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    winsorized_features = []

    for col in numerical_cols:
        if df[col].nunique() <= 1:
            continue

        try:
            # Calculate percentiles
            lower_limit, upper_limit = np.percentile(df[col].dropna(), [limits[0]*100, (1-limits[1])*100])

            # Count outliers before winsorization
            n_lower_outliers = (df[col] < lower_limit).sum()
            n_upper_outliers = (df[col] > upper_limit).sum()

            # Apply winsorization
            df_winsorized[col] = np.clip(df[col], lower_limit, upper_limit)

            if n_lower_outliers > 0 or n_upper_outliers > 0:
                winsorized_features.append(col)
                print(f"  Winsorized {col}: {n_lower_outliers} lower, {n_upper_outliers} upper outliers")

        except Exception as e:
            print(f"  Failed to winsorize {col}: {e}")
            continue

    if winsorized_features:
        print(f"Successfully winsorized {len(winsorized_features)} features")
    else:
        print("No features required winsorization")

    return df_winsorized


def generate_pairwise_interactions(df, important_features=None, max_features=10, correlation_threshold=0.9):
    """
    Generate automatic pairwise interaction terms between important features.

    Args:
        df: Input DataFrame
        important_features: List of important feature names (if None, auto-select based on variance)
        max_features: Maximum number of features to consider for interactions
        correlation_threshold: Skip interactions between highly correlated features

    Returns:
        DataFrame with interaction features
    """
    print(f"\n=== AUTOMATIC PAIRWISE INTERACTION GENERATION ===")

    df_interactions = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove derived features that are already interactions
    numerical_cols = [col for col in numerical_cols if 'Interaction' not in col and 'Ratio' not in col]

    # Select important features
    if important_features is None:
        # Auto-select based on variance (higher variance = more information)
        variances = df[numerical_cols].var().sort_values(ascending=False)
        important_features = variances.head(max_features).index.tolist()

    print(f"Selected {len(important_features)} important features for interactions: {important_features}")

    # Calculate correlation matrix to avoid redundant interactions
    corr_matrix = df[important_features].corr().abs()

    interactions_created = []
    skipped_due_to_correlation = 0

    # Generate pairwise interactions
    for i, feat1 in enumerate(important_features):
        for feat2 in important_features[i+1:]:
            # Skip if features are highly correlated
            if corr_matrix.loc[feat1, feat2] > correlation_threshold:
                skipped_due_to_correlation += 1
                continue

            try:
                # Create interaction feature
                interaction_name = f"{feat1}_x_{feat2}"
                df_interactions[interaction_name] = df[feat1] * df[feat2]
                interactions_created.append(interaction_name)
                print(f"  Created: {interaction_name}")

            except Exception as e:
                print(f"  Failed to create {feat1} × {feat2}: {e}")
                continue

    print(f"Created {len(interactions_created)} interaction features")
    if skipped_due_to_correlation > 0:
        print(f"Skipped {skipped_due_to_correlation} highly correlated feature pairs")

    return df_interactions


def apply_mutual_info_selection(df, target_col, k_features='auto', random_state=42):
    """
    Apply mutual information-based feature selection.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        k_features: Number of features to select ('auto' or int)
        random_state: Random state for reproducibility

    Returns:
        DataFrame with selected features
    """
    print(f"\n=== MUTUAL INFORMATION FEATURE SELECTION ===")

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in DataFrame")
        return df

    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
    X = df[feature_cols]
    y = df[target_col]

    # Remove any NaN values for MI calculation
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]

    if len(X_clean) == 0:
        print("No valid data for mutual information calculation")
        return df

    # Calculate mutual information
    mi_scores = mutual_info_regression(X_clean, y_clean, random_state=random_state)

    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    print("Top 10 features by mutual information:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['mi_score']:.4f}")

    # Determine number of features to select
    if k_features == 'auto':
        # Select features with MI score > 0.1 or top 20 features
        selected_features = feature_importance[feature_importance['mi_score'] > 0.1]['feature'].tolist()
        if len(selected_features) < 5:  # Minimum 5 features
            selected_features = feature_importance.head(max(5, len(feature_importance)//2))['feature'].tolist()
    else:
        selected_features = feature_importance.head(k_features)['feature'].tolist()

    print(f"Selected {len(selected_features)} features using mutual information")

    # Return DataFrame with selected features + target
    selected_cols = selected_features + [target_col]
    return df[selected_cols].copy()


def create_enhanced_feature_pipeline(df, config=None):
    """
    Enhanced feature engineering pipeline with configurable options.

    Args:
        df: Input DataFrame
        config: Dictionary with configuration options

    Returns:
        DataFrame with enhanced features
    """
    if config is None:
        config = {}

    # Default configuration
    default_config = {
        'apply_polynomial': False,
        'polynomial_degree': 2,
        'apply_skewness_transform': True,
        'skewness_threshold': 1.0,
        'transform_method': 'auto',
        'apply_winsorization': True,
        'winsorization_limits': (0.05, 0.05),
        'apply_interactions': True,
        'max_interaction_features': 10,
        'interaction_correlation_threshold': 0.9,
        'apply_mutual_info_selection': False,
        'mi_k_features': 'auto',
        'target_column': 'Total_Revenue'  # Default target for MI selection
    }

    # Merge with provided config
    config = {**default_config, **config}

    print("=== ENHANCED FEATURE ENGINEERING PIPELINE ===")
    print(f"Configuration: {config}")

    df_enhanced = df.copy()

    # 1. Winsorization for outlier handling
    if config['apply_winsorization']:
        df_enhanced = apply_winsorization(df_enhanced, config['winsorization_limits'])

    # 2. Skewness transformation
    if config['apply_skewness_transform']:
        df_enhanced = detect_skewness_and_transform(
            df_enhanced,
            config['skewness_threshold'],
            config['transform_method']
        )

    # 3. Polynomial features (for linear models - will be handled in model pipeline)
    if config['apply_polynomial']:
        print(f"\nNote: Polynomial features (degree {config['polynomial_degree']}) will be applied in model pipeline for linear models")

    # 4. Automatic pairwise interactions
    if config['apply_interactions']:
        # Select important features for interactions (exclude transformed features)
        base_features = [col for col in df_enhanced.select_dtypes(include=[np.number]).columns
                        if not col.endswith('_transformed') and 'Interaction' not in col]
        df_enhanced = generate_pairwise_interactions(
            df_enhanced,
            important_features=base_features[:config['max_interaction_features']],
            correlation_threshold=config['interaction_correlation_threshold']
        )

    # 5. Mutual information feature selection
    if config['apply_mutual_info_selection']:
        df_enhanced = apply_mutual_info_selection(
            df_enhanced,
            config['target_column'],
            config['mi_k_features']
        )

    print(f"\nEnhanced feature engineering completed. Final shape: {df_enhanced.shape}")
    return df_enhanced

def assess_scales_and_encoding(df):
    """
    Assess variable types and scales, apply normalization and dummy variables.
    """
    print("\n=== VARIABLE TYPES, SCALES, AND ENCODING ===")

    # Identify categorical and numerical columns
    categorical_cols = ['Sales Channel', 'WarehouseCode', 'CurrencyCode']
    numerical_cols = ['Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price',
                     'Procurement_to_Order_Days', 'Order_to_Ship_Days', 'Ship_to_Delivery_Days',
                     'Total_Lead_Time', 'Profit_Margin', 'Total_Revenue']
    
    # Add interaction features to numerical columns if they exist
    interaction_cols = [col for col in df.columns if 'Interaction' in col or 'Ratio' in col]
    if interaction_cols:
        numerical_cols.extend(interaction_cols)
        print(f"Including {len(interaction_cols)} interaction features in scaling")

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {len(numerical_cols)} features")

    # Check scales of numerical variables
    print("\nNumerical variable scales (sample):")
    for col in numerical_cols[:5]:  # Show first 5 to avoid clutter
        if col in df.columns:
            print(f"{col}: range = {df[col].min():.2f} to {df[col].max():.2f}, std = {df[col].std():.2f}")

    # Apply normalization to numerical features
    scaler = StandardScaler()
    df[[col + '_scaled' for col in numerical_cols]] = scaler.fit_transform(df[numerical_cols])

    # Apply dummy encoding to categorical features
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop first to avoid multicollinearity
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_col_names = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=df.index)
    df = pd.concat([df, df_encoded], axis=1)

    print(f"Applied StandardScaler to {len(numerical_cols)} numerical columns")
    print(f"Applied OneHotEncoder to categorical columns: {list(encoded_col_names)}")
    print(f"New shape after encoding: {df.shape}")

    return df

def assess_balance(df):
    """
    Assess dataset balance in features and potential targets.
    """
    print("\n=== DATASET BALANCE ASSESSMENT ===")

    # Check balance of categorical features
    categorical_features = ['Sales Channel', 'WarehouseCode', 'CurrencyCode']

    for col in categorical_features:
        if col in df.columns:
            value_counts = df[col].value_counts()
            total = len(df)
            print(f"\n{col} distribution:")
            for val, count in value_counts.items():
                percentage = (count / total) * 100
                print(f"  {val}: {count} ({percentage:.2f}%)")

            # Check if balanced (no category > 70% or < 5%)
            max_percent = (value_counts.max() / total) * 100
            min_percent = (value_counts.min() / total) * 100

            if max_percent > 70:
                print(f"  ⚠️  Imbalanced: {value_counts.idxmax()} dominates ({max_percent:.1f}%)")
            elif min_percent < 5:
                print(f"  ⚠️  Some categories are rare (min {min_percent:.1f}%)")
            else:
                print("  ✓ Balanced distribution")

    # Check numerical feature distributions
    numerical_cols = ['Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price',
                     'Profit_Margin', 'Total_Revenue']

    print("\nNumerical feature distributions:")
    for col in numerical_cols:
        if col in df.columns:
            skewness = df[col].skew()
            print(f"  {col}: skewness = {skewness:.2f}")
            if abs(skewness) > 1:
                print("    ⚠️  Highly skewed")
            elif abs(skewness) > 0.5:
                print("    ⚠️  Moderately skewed")
            else:
                print("    ✓ Approximately normal")

    return df

def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the dataframe.
    """
    print(f"\n=== DROPPING COLUMNS ===")
    print(f"Columns to drop: {columns_to_drop}")

    # Check which columns exist in the dataframe
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    missing_columns = [col for col in columns_to_drop if col not in df.columns]

    if missing_columns:
        print(f"Warning: The following columns do not exist in the dataframe: {missing_columns}")

    if existing_columns:
        print(f"Dropping columns: {existing_columns}")
        df = df.drop(columns=existing_columns)
        print(f"New shape after dropping columns: {df.shape}")

    return df

def exploratory_data_analysis(df):
    """
    Conduct exploratory data analysis with correlations and visualizations.
    """
    print("\n=== EXPLORATORY DATA ANALYSIS ===")

    # Correlation analysis
    numerical_cols = ['Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price',
                     'Procurement_to_Order_Days', 'Order_to_Ship_Days', 'Ship_to_Delivery_Days',
                     'Total_Lead_Time', 'Profit_Margin', 'Total_Revenue']

    corr_matrix = df[numerical_cols].corr()
    print("Correlation matrix:")
    print(corr_matrix)

    # High correlations
    high_corr = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
    high_corr_pairs = high_corr.stack().reset_index()
    high_corr_pairs.columns = ['Variable1', 'Variable2', 'Correlation']
    high_corr_pairs = high_corr_pairs[abs(high_corr_pairs['Correlation']) > 0.7]
    print("\nHigh correlations (|r| > 0.7):")
    if len(high_corr_pairs) > 0:
        for _, row in high_corr_pairs.iterrows():
            print(f"  {row['Variable1']} ↔ {row['Variable2']}: {row['Correlation']:.3f}")
    else:
        print("  No high correlations found")

    # Create visualizations
    try:
        # Sales channel distribution
        plt.figure(figsize=(10, 6))
        df['Sales Channel'].value_counts().plot(kind='bar')
        plt.title('Sales Channel Distribution')
        plt.xlabel('Sales Channel')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sales_channel_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Unit Price distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Unit Price'], kde=True)
        plt.title('Unit Price Distribution')
        plt.xlabel('Unit Price')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('unit_price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Visualizations saved: sales_channel_distribution.png, unit_price_distribution.png, correlation_heatmap.png")

    except Exception as e:
        print(f"Error creating visualizations: {e}")

    return df

def finalize_dataset(df):
    """
    Finalize the cleaned and preprocessed dataset.
    """
    print("\n=== FINALIZE PREPROCESSED DATASET ===")

    # Summary of preprocessing steps
    print("Preprocessing Summary:")
    print("✓ Loaded and parsed CSV data")
    print("✓ Handled missing values (none found)")
    print("✓ Checked data consistency")
    print("✓ Added derived temporal and financial features")
    print("✓ Performed univariate outlier detection")
    print("✓ Performed multivariate outlier detection")
    print("✓ Considered contextual outliers")
    print("✓ Applied scaling and encoding")
    print("✓ Assessed dataset balance")
    print("✓ Conducted exploratory data analysis")

    # Final dataset info
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Final columns: {len(df.columns)}")

    # Column categories
    original_cols = ['OrderNumber', 'Sales Channel', 'WarehouseCode', 'ProcuredDate', 'OrderDate',
                    'ShipDate', 'DeliveryDate', 'CurrencyCode', '_SalesTeamID', '_CustomerID',
                    '_StoreID', '_ProductID', 'Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price']

    derived_cols = [col for col in df.columns if col not in original_cols and not col.endswith('_scaled') and not col.startswith(('Sales Channel_', 'WarehouseCode_', 'CurrencyCode_'))]
    scaled_cols = [col for col in df.columns if col.endswith('_scaled')]
    encoded_cols = [col for col in df.columns if col.startswith(('Sales Channel_', 'WarehouseCode_', 'CurrencyCode_'))]

    print(f"Original columns: {len(original_cols)}")
    print(f"Derived features: {len(derived_cols)}")
    print(f"Scaled features: {len(scaled_cols)}")
    print(f"Encoded features: {len(encoded_cols)}")

    # Save processed dataset to project root (matching config.py expectations)
    out_csv = config.PREPROCESSED_DATA_FILE
    df.to_csv(out_csv, index=False)
    print(f"Preprocessed dataset saved as '{out_csv}'")

    # Attempt to automatically run improved visualizations (optional)
    try:
        run_improved_visualizations(out_csv)
    except Exception as e:
        print(f"Could not run improved visualizations automatically: {e}")

    return df

def get_processed_feature_names(df):
    """
    Get the names of processed features after preprocessing pipeline.

    This provides a clean interface to retrieve feature names post-OneHotEncoding or scaling.

    Args:
        df: Preprocessed DataFrame

    Returns:
        Dictionary with different categories of feature names
    """
    # Original features (before encoding/scaling)
    original_cols = ['OrderNumber', 'Sales Channel', 'WarehouseCode', 'ProcuredDate', 'OrderDate',
                    'ShipDate', 'DeliveryDate', 'CurrencyCode', '_SalesTeamID', '_CustomerID',
                    '_StoreID', '_ProductID', 'Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price']

    # Derived features (created during preprocessing)
    derived_cols = [col for col in df.columns if col not in original_cols and not col.endswith('_scaled') and not col.startswith(('Sales Channel_', 'WarehouseCode_', 'CurrencyCode_'))]

    # Scaled features
    scaled_cols = [col for col in df.columns if col.endswith('_scaled')]

    # Encoded features (OneHotEncoding)
    encoded_cols = [col for col in df.columns if col.startswith(('Sales Channel_', 'WarehouseCode_', 'CurrencyCode_'))]

    # All processed features (what would be used for modeling)
    all_processed_features = derived_cols + scaled_cols + encoded_cols

    # Numeric features for modeling (exclude categorical strings and dates)
    numeric_features = [col for col in all_processed_features if not any(keyword in col.lower() for keyword in ['date', 'channel', 'warehouse', 'currency']) or col.endswith('_scaled')]

    return {
        'original_features': original_cols,
        'derived_features': derived_cols,
        'scaled_features': scaled_cols,
        'encoded_features': encoded_cols,
        'all_processed_features': all_processed_features,
        'modeling_features': numeric_features,  # Features suitable for ML models
        'feature_counts': {
            'original': len(original_cols),
            'derived': len(derived_cols),
            'scaled': len(scaled_cols),
            'encoded': len(encoded_cols),
            'total_processed': len(all_processed_features),
            'modeling_ready': len(numeric_features)
        }
    }

def _run_all_steps(data_path: str):
    """Helper to run the entire preprocessing flow given a data path."""
    df = load_data(data_path)

    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Initial exploration
    df = initial_exploration(df)

    # Explain the data and variables
    explain_data()

    # Check missing and handle filling missing values
    df = check_missing_values(df)

    # Execute the remaining functions
    df = check_data_consistency(df)
    df = convert_data_types(df)
    
    # Create interaction features before outlier detection
    df = create_interaction_features(df)
    
    df = univariate_outlier_detection(df)
    df = multivariate_outlier_detection(df)
    df = contextual_outlier_detection(df)

    df = assess_balance(df)
    df = exploratory_data_analysis(df)

    # Apply enhanced feature engineering pipeline
    enhanced_config = {
        'apply_polynomial': False,  # Will be handled in model pipeline for specific models
        'apply_skewness_transform': True,
        'apply_winsorization': True,
        'apply_interactions': True,
        'apply_mutual_info_selection': False  # Can be enabled if target is available
    }
    df = create_enhanced_feature_pipeline(df, enhanced_config)

    # Drop unnecessary columns
    columns_to_drop = ['CurrencyCode', '_SalesTeamID', '_CustomerID', '_StoreID', '_ProductID']
    df = drop_columns(df, columns_to_drop)

    df = finalize_dataset(df)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess US Regional Sales dataset")
    parser.add_argument('--data', '-d', help='Path to Data-USRegionalSales.csv', default=None)
    args = parser.parse_args()

    try:
        data_path = find_data_file(args.data)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)

    _run_all_steps(data_path)
    
    

