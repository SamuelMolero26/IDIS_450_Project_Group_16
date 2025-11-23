#!/usr/bin/env python3
"""
Test script for KNN-specific visualizations.
"""

import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model_pipeline import ModelPipeline
from src.config import PREPROCESSED_DATA_FILE, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def test_knn_visualizations():
    """Test KNN-specific visualization generation."""

    print("🖼️ Testing KNN-Specific Visualizations")
    print("=" * 50)

    # Load data
    print("Loading data...")
    import pandas as pd
    df = pd.read_csv(PREPROCESSED_DATA_FILE)

    # Prepare features (smaller sample for faster testing)
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].sample(n=1000, random_state=42)
    y = df[TARGET_COLUMN].sample(n=1000, random_state=42)

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)

    print(f"Dataset shape: {X_encoded.shape}")

    # Initialize pipeline and train KNN
    pipeline = ModelPipeline()

    print("Training KNN model...")

    try:
        # Train KNN with hyperparameter tuning to get optimal K
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Train KNN with tuning
        knn_results = pipeline.train_model('KNN', X_train, y_train)
        model_id = knn_results['model_id']

        print(f"✅ KNN trained successfully with ID: {model_id}")
        print(f"Optimal K: {pipeline.trained_models[model_id].n_neighbors}")

        # Generate KNN-specific visualizations
        print("\n🎨 Generating KNN-specific visualizations...")
        visualization_paths = pipeline.generate_knn_specific_visualizations(
            model_id, X_train, y_train, X_test, y_test
        )

        if not visualization_paths:
            print("❌ No visualizations were generated")
            return False

        print("✅ KNN visualizations generated successfully!")
        print(f"📊 Generated {len(visualization_paths)} visualization(s):")

        # List all generated visualizations
        viz_descriptions = {
            'knn_neighbor_distances': 'Neighbor Distance Analysis - Shows distance distributions to k nearest neighbors',
            'knn_prediction_stability': 'Prediction Stability Analysis - Shows how predictions vary with different K values',
            'knn_feature_importance': 'Feature Importance via Permutation - Shows feature contribution through permutation importance',
            'knn_error_analysis': 'Error Analysis by Prediction Range - Shows prediction errors across different value ranges',
            'knn_timing_analysis': 'Timing Analysis - Shows training vs prediction time characteristics',
            'knn_decision_boundary': 'Decision Boundary Visualization - Shows prediction surface in 2D PCA projection'
        }

        for viz_name, path in visualization_paths.items():
            description = viz_descriptions.get(viz_name, 'KNN-specific visualization')
            print(f"  📈 {viz_name}: {description}")
            print(f"     Saved to: {path}")

        # Verify files exist
        print("\n🔍 Verifying generated files...")
        missing_files = []
        for path in visualization_paths.values():
            if not Path(path).exists():
                missing_files.append(path)

        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False

        print("✅ All visualization files were created successfully!")

        # Show KNN model details
        model = pipeline.trained_models[model_id]
        print("\n🤖 KNN Model Details:")
        print(f"   K (n_neighbors): {model.n_neighbors}")
        print(f"   Weights: {model.weights}")
        print(f"   Metric: {model.metric}")
        print(f"   Algorithm: {model.algorithm}")

        return True

    except Exception as e:
        print(f"❌ KNN visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_knn_visualizations()
    sys.exit(0 if success else 1)
    # Train KNN with hyperparameter tuning to get optimal K
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train KNN with tuning
    knn_results = pipeline.train_model('KNN', X_train, y_train)
    model_id = knn_results['model_id']

    print(f"✅ KNN trained successfully with ID: {model_id}")
    print(f"Optimal K: {pipeline.trained_models[model_id].n_neighbors}")

    # Generate KNN-specific visualizations
    print("\n🎨 Generating KNN-specific visualizations...")
    visualization_paths = pipeline.generate_knn_specific_visualizations(
        model_id, X_train, y_train, X_test, y_test
    )

    if not visualization_paths:
        print("❌ No visualizations were generated")
        

    print("✅ KNN visualizations generated successfully!")
    print(f"📊 Generated {len(visualization_paths)} visualization(s):")

    # List all generated visualizations
    viz_descriptions = {
        'knn_neighbor_distances': 'Neighbor Distance Analysis - Shows distance distributions to k nearest neighbors',
        'knn_prediction_stability': 'Prediction Stability Analysis - Shows how predictions vary with different K values',
        'knn_feature_importance': 'Feature Importance via Permutation - Shows feature contribution through permutation importance',
        'knn_error_analysis': 'Error Analysis by Prediction Range - Shows prediction errors across different value ranges',
        'knn_timing_analysis': 'Timing Analysis - Shows training vs prediction time characteristics',
        'knn_decision_boundary': 'Decision Boundary Visualization - Shows prediction surface in 2D PCA projection'
    }

    for viz_name, path in visualization_paths.items():
        description = viz_descriptions.get(viz_name, 'KNN-specific visualization')
        print(f"  📈 {viz_name}: {description}")
        print(f"     Saved to: {path}")

    # Verify files exist
    print("\n🔍 Verifying generated files...")
    missing_files = []
    for path in visualization_paths.values():
        if not Path(path).exists():
            missing_files.append(path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
           
    try:
        print("✅ All visualization files were created successfully!")

        # Show KNN model details
        model = pipeline.trained_models[model_id]
        print("\n🤖 KNN Model Details:")
        print(f"   K (n_neighbors): {model.n_neighbors}")
        print(f"   Weights: {model.weights}")
        print(f"   Metric: {model.metric}")
        print(f"   Algorithm: {model.algorithm}")
    

    except Exception as e:
        print(f"❌ KNN visualization test failed: {e}")
        import traceback
        traceback.print_exc()
       

if __name__ == "__main__":
    success = test_knn_visualizations()
    sys.exit(0 if success else 1)