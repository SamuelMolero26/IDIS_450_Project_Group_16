#!/usr/bin/env python3
"""
Test script for comprehensive model comparison including KNN.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model_pipeline import ModelPipeline
from src.config import PREPROCESSED_DATA_FILE, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def test_model_comparison():
    """Test comprehensive model comparison including KNN."""

    # Load data
    print("Loading preprocessed data...")
    df = pd.read_csv(PREPROCESSED_DATA_FILE)

    # Prepare features (use smaller sample for faster testing)
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].sample(n=1000, random_state=42)  # Smaller sample for speed
    y = df[TARGET_COLUMN].sample(n=1000, random_state=42)

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)

    print(f"Dataset shape: {X_encoded.shape}")
    print(f"Target shape: {y.shape}")

    # Initialize pipeline
    pipeline = ModelPipeline()

    # Test comprehensive model comparison
    print("\n=== Testing Comprehensive Model Comparison ===")
    try:
        comparison_results = pipeline.create_comprehensive_model_comparison(
            X_encoded, y, include_knn=True
        )

        if 'error' in comparison_results:
            print(f"❌ Comparison failed: {comparison_results['error']}")
            return False

        print("✅ Model comparison completed successfully")

        # Display results
        print(f"\n📊 Comparison Summary:")
        print(f"Total models compared: {comparison_results['total_models']}")
        print(f"Valid models: {comparison_results['valid_models']}")

        best_model = comparison_results['best_model']
        print(f"\n🏆 Best Model: {best_model['name']}")
        print(".4f")
        print(".2f")
        print(f"Rank: {best_model['rank']}")
        print(f"Is KNN: {best_model['is_knn']}")

        # Show model rankings
        print(f"\n📈 Model Rankings (by R²):")
        for i, model in enumerate(comparison_results['model_rankings'], 1):
            knn_indicator = " ← KNN" if model['is_knn'] else ""
            print(f"{i}. {model['model_name']}: R² = {model['r2_score']:.4f}, RMSE = {model['rmse_score']:.2f}, Time = {model['training_time']:.2f}s{knn_indicator}")

        # Show performance ranges
        perf_summary = comparison_results['performance_summary']
        print(f"\n📊 Performance Ranges:")
        print(".4f")
        print(".2f")

        # KNN-specific analysis
        knn_analysis = comparison_results.get('knn_specific_analysis', {})
        if knn_analysis and 'error' not in knn_analysis:
            print(f"\n🤖 KNN-Specific Analysis:")
            print(f"KNN R² Score: {knn_analysis['knn_r2_score']:.4f}")
            print(f"KNN Rank: {knn_analysis['knn_rank']}")
            print(f"KNN Training Time: {knn_analysis['knn_training_time']:.2f}s")
            print(f"Speed Advantage: {knn_analysis['speed_advantage']:.1f}x faster than average")

            if knn_analysis.get('performance_vs_best'):
                perf_vs_best = knn_analysis['performance_vs_best']
                print(".3f")
                print(".1f")

            if knn_analysis.get('knn_strengths'):
                print(f"Strengths: {', '.join(knn_analysis['knn_strengths'])}")

            if knn_analysis.get('knn_weaknesses'):
                print(f"Weaknesses: {', '.join(knn_analysis['knn_weaknesses'])}")

        # Show recommendations
        recommendations = comparison_results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")

        # Show visualizations generated
        visualizations = comparison_results.get('visualizations', {})
        if visualizations:
            print(f"\n📊 Visualizations Generated:")
            for viz_type, path in visualizations.items():
                print(f"- {viz_type}: {path}")

        print(f"\n✅ Model comparison test completed successfully!")
        print(f"Best model determined: {best_model['name']}")

        return True

    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_comparison()
    sys.exit(0 if success else 1)