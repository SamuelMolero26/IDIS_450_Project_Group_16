#!/usr/bin/env python3
"""
Lightweight Model Report Generator - Memory Optimized for Large Datasets

This script generates a basic model comparison report using data sampling
to avoid memory issues with large datasets.

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
import sys
import os
import gc

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import (
    PREPROCESSED_DATA_FILE, TARGET_COLUMN, NUMERICAL_FEATURES,
    RANDOM_STATE, TEST_SIZE
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
plt.style.use('default')

# CRITICAL: Sample size to prevent memory issues
SAMPLE_SIZE = 10000  # Use only 10k rows for training

def load_sampled_data():
    """Load a sample of the data to avoid memory issues."""
    print(f"📊 Loading sampled data (max {SAMPLE_SIZE} rows)...")
    
    # Read only necessary columns
    numerical_features = NUMERICAL_FEATURES[:5]  # Use only first 5 features
    target_col = 'Total_Revenue'
    
    # Load in chunks and sample
    chunks = []
    chunk_size = 5000
    total_read = 0
    
    for chunk in pd.read_csv(PREPROCESSED_DATA_FILE, chunksize=chunk_size):
        if total_read >= SAMPLE_SIZE:
            break
        chunks.append(chunk)
        total_read += len(chunk)
    
    data = pd.concat(chunks, ignore_index=True)
    
    # Sample if still too large
    if len(data) > SAMPLE_SIZE:
        data = data.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    print(f"✅ Loaded {len(data)} samples")
    
    # Extract features and target
    available_features = [f for f in numerical_features if f in data.columns]
    X = data[available_features].copy()
    y = data[target_col].copy()
    
    del data, chunks
    gc.collect()
    
    return X, y, available_features

def train_and_evaluate_models(X, y):
    """Train and evaluate models with minimal memory footprint."""
    print("\n🚀 Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    results = {}
    
    # 1. Linear Regression
    print("  ↳ Linear Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    
    results['Linear Regression'] = {
        'R²': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    del lr, X_train_scaled, X_test_scaled, y_pred
    gc.collect()
    
    # 2. Decision Tree (limited depth)
    print("  ↳ Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    results['Decision Tree'] = {
        'R²': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    del dt, y_pred
    gc.collect()
    
    # 3. Random Forest (small ensemble)
    print("  ↳ Random Forest...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, 
                               random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    results['Random Forest'] = {
        'R²': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    del rf, y_pred, X_train, X_test, y_train, y_test
    gc.collect()
    
    return results

def create_simple_report(results, features):
    """Create a simple markdown report."""
    print("\n📝 Creating report...")
    
    report = f"""# Lightweight Model Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sample Size:** {SAMPLE_SIZE} rows
**Features Used:** {len(features)}
**Feature Names:** {', '.join(features)}

## Model Performance Comparison

| Model | R² Score | RMSE |
|-------|----------|------|
"""
    
    for model_name, metrics in results.items():
        report += f"| {model_name} | {metrics['R²']:.4f} | {metrics['RMSE']:.2f} |\n"
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['R²'])
    
    report += f"""

## Best Model

**Winner:** {best_model[0]}
- R² Score: {best_model[1]['R²']:.4f}
- RMSE: {best_model[1]['RMSE']:.2f}

## Notes

- This is a lightweight analysis using a sample of {SAMPLE_SIZE} rows
- Only {len(features)} features were used to minimize memory usage
- For full analysis, consider using a machine with more RAM or cloud computing
- Models were trained with reduced complexity to prevent memory issues

---

*Generated by Lightweight Report Generator*
"""
    
    # Save report
    report_path = Path("LIGHTWEIGHT_MODEL_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to {report_path}")
    
    # Create simple visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(results.keys())
    r2_scores = [results[m]['R²'] for m in models]
    
    ax.bar(models, r2_scores, alpha=0.7, color=['blue', 'green', 'orange'])
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    for i, (model, score) in enumerate(zip(models, r2_scores)):
        ax.text(i, score + 0.02, f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    viz_path = Path("model_comparison.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to {viz_path}")

def main():
    """Main execution."""
    print("🚀 Starting Lightweight Model Analysis")
    print("=" * 60)
    print(f"⚠️  Using data sampling to prevent memory issues")
    print(f"⚠️  Sample size: {SAMPLE_SIZE} rows")
    print("=" * 60)
    
    try:
        # Load sampled data
        X, y, features = load_sampled_data()
        
        # Train and evaluate
        results = train_and_evaluate_models(X, y)
        
        # Create report
        create_simple_report(results, features)
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📂 Generated files:")
        print("   - LIGHTWEIGHT_MODEL_REPORT.md")
        print("   - model_comparison.png")
        print("\n💡 This lightweight version prevents memory issues")
        print("   by using data sampling and reduced model complexity.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())