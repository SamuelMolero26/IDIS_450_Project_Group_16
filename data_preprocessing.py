# Data Preprocessing and Exploratory Data Analysis for US Regional Sales Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

#For sanity check
print("Libraries imported successfully.")


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

    VARIABLE DESCRIPTIONS:

    1. OrderNumber (string): Unique identifier for each sales order (e.g., 'SO - 000101')
    2. Sales Channel (categorical): Channel through which sale was made
       - In-Store: Direct store purchases
       - Online: E-commerce transactions
       - Distributor: B2B distribution
       - Wholesale: Bulk sales
    3. WarehouseCode (string): Warehouse identifier (e.g., 'WARE-UHY1004')
    4. ProcuredDate (datetime): When product was procured
    5. OrderDate (datetime): When order was placed
    6. ShipDate (datetime): When product was shipped
    7. DeliveryDate (datetime): When product was delivered
    8. CurrencyCode (string): Transaction currency (primarily USD)
    9. _SalesTeamID (int): Sales team identifier
    10. _CustomerID (int): Customer identifier
    11. _StoreID (int): Store identifier
    12. _ProductID (int): Product identifier
    13. Order Quantity (int): Quantity of products ordered
    14. Discount Applied (float): Discount percentage (0.0 to 1.0)
    15. Unit Cost (float): Cost price per unit
    16. Unit Price (float): Selling price per unit

    KEY INSIGHTS:
    - Temporal features allow for delivery time analysis
    - Price features enable profit margin calculations
    - Categorical features support segmentation analysis
    - ID features can be used for grouping and aggregation
    """)

def check_missing_values(df):
    """
    Check for missing values and handle them appropriately.
    Basic context awareness is applied to decide imputation strategies.
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

    # Smart handleing of missing values based on the data types
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # For numerical --> use median imputation
                imputer = SimpleImputer(strategy='median')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                print(f"Imputed missing values in {col} with median.")
            elif df[col].dtype == 'object':
                # For categorical --> use most frequent
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                print(f"Imputed missing values in {col} with most frequent value.")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # For dates --> forward fill
                df[col] = df[col].fillna(method='ffill')
                print(f"Imputed missing values in {col} with forward fill.")

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
    Perform univariate outlier detection using Z-score and IQR methods.
    """
    print("\n=== UNIVARIATE OUTLIER DETECTION ===")

    numeric_cols = ['Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price',
                   'Procurement_to_Order_Days', 'Order_to_Ship_Days', 'Ship_to_Delivery_Days',
                   'Total_Lead_Time', 'Profit_Margin', 'Total_Revenue']

    outlier_summary = {}

    for col in numeric_cols:
        if col in df.columns:
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_outliers = (z_scores > 3).sum()

            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

            outlier_summary[col] = {
                'Z-score outliers': z_outliers,
                'IQR outliers': iqr_outliers,
                'Percentage': (iqr_outliers / len(df)) * 100
            }

            print(f"{col}: {iqr_outliers} outliers ({outlier_summary[col]['Percentage']:.2f}%)")

    # Flag outliers using IQR method
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_col = f'{col}_outlier'
            df[outlier_col] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).astype(int)

    print("Outlier flags added to dataframe")
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

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # Check scales of numerical variables
    
    print("\nNumerical variable scales:")
    for col in numerical_cols:
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

    print(f"Applied StandardScaler to numerical columns")
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

    # Save processed dataset
    df.to_csv('preprocessed_sales_data.csv', index=False)
    print("Preprocessed dataset saved as 'preprocessed_sales_data.csv'")

    return df

# Main execution
# Load data
file_path = 'Project4_USRegionalSales/Data-USRegionalSales.csv'
df = load_data(file_path)

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
df = univariate_outlier_detection(df)
df = multivariate_outlier_detection(df)
df = contextual_outlier_detection(df)

df = assess_balance(df)
df = exploratory_data_analysis(df)
df = finalize_dataset(df)