import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Load preprocessed data
df = pd.read_csv('preprocessed_sales_data.csv')

# Define numeric columns
numeric_cols = ['Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price',
                'Procurement_to_Order_Days', 'Order_to_Ship_Days', 'Ship_to_Delivery_Days',
                'Total_Lead_Time', 'Profit_Margin', 'Total_Revenue']

def enhanced_histogram(data, column, facet_by=None, save_path=None):
    """
    Create enhanced histogram with statistical overlays and KDE.
    """
    if facet_by:
        channels = data[facet_by].unique()
        n_channels = len(channels)
        fig, axes = plt.subplots(1, n_channels, figsize=(6*n_channels, 5), sharey=True)

        for i, channel in enumerate(channels):
            subset = data[data[facet_by] == channel]
            ax = axes[i] if n_channels > 1 else axes

            # Histogram with KDE
            sns.histplot(subset[column], kde=True, ax=ax, alpha=0.7)

            # Statistical lines
            mean_val = subset[column].mean()
            median_val = subset[column].median()
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

            # Quartiles
            q1, q3 = subset[column].quantile([0.25, 0.75])
            ax.axvspan(q1, q3, alpha=0.2, color='green', label=f'IQR: {q1:.2f} - {q3:.2f}')

            # Stats text
            stats_text = f"""
            Std Dev: {subset[column].std():.2f}
            Skewness: {subset[column].skew():.2f}
            Kurtosis: {subset[column].kurtosis():.2f}
            Range: {subset[column].max() - subset[column].min():.2f}
            """
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_title(f'{column} Distribution\n{channel}')
            ax.legend()

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram with KDE
        sns.histplot(data[column], kde=True, ax=ax, alpha=0.7)

        # Statistical lines
        mean_val = data[column].mean()
        median_val = data[column].median()
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

        # Quartiles
        q1, q3 = data[column].quantile([0.25, 0.75])
        ax.axvspan(q1, q3, alpha=0.2, color='green', label=f'IQR: {q1:.2f} - {q3:.2f}')

        # Stats text
        stats_text = f"""
        Std Dev: {data[column].std():.2f}
        Skewness: {data[column].skew():.2f}
        Kurtosis: {data[column].kurtosis():.2f}
        Range: {data[column].max() - data[column].min():.2f}
        """
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(f'Enhanced Histogram: {column} Distribution')
        ax.legend()

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def enhanced_scatter_plot(data, x_col, y_col, color_by='Sales Channel', size_by=None, save_path=None):
    """
    Create enhanced scatter plot with regression line and correlation.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    if size_by:
        scatter = sns.scatterplot(data=data, x=x_col, y=y_col, hue=color_by, size=size_by, ax=ax, alpha=0.7)
    else:
        scatter = sns.scatterplot(data=data, x=x_col, y=y_col, hue=color_by, ax=ax, alpha=0.7)

    # Regression line
    sns.regplot(data=data, x=x_col, y=y_col, scatter=False, ax=ax, color='red', line_kws={'linewidth': 2})

    # Correlation coefficient
    corr = data[x_col].corr(data[y_col])
    ax.text(0.02, 0.98, f'Correlation: {corr:.3f}', transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Outlier detection (IQR method)
    q1_x, q3_x = data[x_col].quantile([0.25, 0.75])
    iqr_x = q3_x - q1_x
    q1_y, q3_y = data[y_col].quantile([0.25, 0.75])
    iqr_y = q3_y - q1_y

    outliers = data[
        ((data[x_col] < (q1_x - 1.5 * iqr_x)) | (data[x_col] > (q3_x + 1.5 * iqr_x))) |
        ((data[y_col] < (q1_y - 1.5 * iqr_y)) | (data[y_col] > (q3_y + 1.5 * iqr_y)))
    ]

    if len(outliers) > 0:
        sns.scatterplot(data=outliers, x=x_col, y=y_col, ax=ax, color='red', marker='x', s=100, label='Outliers')

    ax.set_title(f'Enhanced Scatter Plot: {x_col} vs {y_col}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def clustered_correlation_heatmap(data, columns, save_path=None):
    """
    Create clustered correlation heatmap with hierarchical clustering.
    """
    corr_matrix = data[columns].corr()

    # Hierarchical clustering
    linkage_matrix = linkage(corr_matrix, method='ward')

    # Reorder columns based on clustering
    dendro = dendrogram(linkage_matrix, no_plot=True)
    reordered_cols = [columns[i] for i in dendro['leaves']]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix.loc[reordered_cols, reordered_cols], annot=True, cmap='coolwarm',
                center=0, fmt='.2f', ax=ax, cbar_kws={'shrink': 0.8})

    ax.set_title('Clustered Correlation Heatmap\nVariables grouped by similarity')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def partial_correlation_matrix(data, columns, control_vars=None, save_path=None):
    """
    Calculate and visualize partial correlations.
    """
    from pingouin import partial_corr

    if control_vars is None:
        control_vars = []

    partial_corr_df = pd.DataFrame(index=columns, columns=columns)

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i != j:
                try:
                    pcorr = partial_corr(data=data, x=col1, y=col2, covar=control_vars)
                    partial_corr_df.loc[col1, col2] = pcorr['r'].values[0]
                except:
                    partial_corr_df.loc[col1, col2] = np.nan
            else:
                partial_corr_df.loc[col1, col2] = 1.0

    partial_corr_df = partial_corr_df.astype(float)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(partial_corr_df, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title('Partial Correlation Matrix\n(Controlling for other variables)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def qq_plot_skewness_validation(data, column, save_path=None):
    """
    Create Q-Q plot to validate skewness and normality assumption.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Q-Q plot
    stats.probplot(data[column].dropna(), dist="norm", plot=ax)

    # Add skewness info
    skewness = data[column].skew()
    ax.set_title(f'Q-Q Plot for {column}\nSkewness: {skewness:.3f} ({ "Right-skewed" if skewness > 0.5 else "Left-skewed" if skewness < -0.5 else "Approximately normal" })')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def skewness_validation_plot(data, column, save_path=None):
    """
    Create a comprehensive skewness validation plot with histogram, KDE, and skewness measures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram with KDE
    sns.histplot(data[column].dropna(), kde=True, ax=ax1, alpha=0.7)
    skewness = data[column].skew()
    kurtosis = data[column].kurtosis()
    ax1.set_title(f'Histogram & KDE: {column}\nSkewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}')
    ax1.axvline(data[column].mean(), color='red', linestyle='-', linewidth=2, label='Mean')
    ax1.axvline(data[column].median(), color='blue', linestyle='--', linewidth=2, label='Median')
    ax1.legend()

    # Q-Q plot
    stats.probplot(data[column].dropna(), dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot: {column}\nNormality Test')

    # Add skewness interpretation
    interpretation = ""
    if abs(skewness) < 0.5:
        interpretation = "Approximately normal distribution"
    elif skewness > 0.5:
        interpretation = "Right-skewed (positive skew)"
    else:
        interpretation = "Left-skewed (negative skew)"

    fig.suptitle(f'Skewness Validation for {column}\n{interpretation}', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Main execution
if __name__ == "__main__":
    print("Generating improved visualizations...")

    # Enhanced histograms for key variables
    key_vars = ['Unit Price', 'Profit_Margin', 'Total_Revenue', 'Order Quantity']
    for var in key_vars:
        enhanced_histogram(df, var, facet_by='Sales Channel',
                          save_path=f'visualizations/enhanced_histogram_{var.lower().replace(" ", "_")}.png')
        print(f"Generated enhanced histogram for {var}")

    # Enhanced histograms for other variables
    other_vars = ['Discount Applied', 'Unit Cost', 'Total_Lead_Time']
    for var in other_vars:
        enhanced_histogram(df, var,
                          save_path=f'visualizations/enhanced_histogram_{var.lower().replace(" ", "_")}.png')
        print(f"Generated enhanced histogram for {var}")

    # Enhanced scatter plots
    scatter_pairs = [
        ('Unit Price', 'Total_Revenue'),
        ('Unit Cost', 'Profit_Margin'),
        ('Order Quantity', 'Total_Revenue'),
        ('Discount Applied', 'Profit_Margin'),
        ('Total_Lead_Time', 'Total_Revenue')
    ]

    for x, y in scatter_pairs:
        enhanced_scatter_plot(df, x, y, size_by='Order Quantity' if 'Revenue' in y else None,
                             save_path=f'visualizations/enhanced_scatter_{x.lower().replace(" ", "_")}_vs_{y.lower().replace(" ", "_")}.png')
        print(f"Generated enhanced scatter plot for {x} vs {y}")

    # Advanced correlations
    clustered_correlation_heatmap(df, numeric_cols,
                                 save_path='visualizations/clustered_correlation_heatmap.png')
    print("Generated clustered correlation heatmap")

    # Comprehensive skewness validation plots
    skewed_vars = ['Discount Applied', 'Unit Cost', 'Unit Price', 'Profit_Margin', 'Total_Revenue']
    for var in skewed_vars:
        skewness_validation_plot(df, var,
                                save_path=f'visualizations/skewness_validation_{var.lower().replace(" ", "_")}.png')
        print(f"Generated comprehensive skewness validation plot for {var}")

    print("All improved visualizations generated!")