#!/usr/bin/env python3
"""
Basic ANN integration test - quick verification without extensive tuning.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.model_pipeline import create_model_pipeline
    from src.data_loader import create_data_loader
    from src.logger import model_logger
    print("✅ Successfully imported pipeline components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_ann_basic():
    """Test basic ANN functionality with minimal configuration."""
    print("🧠 Testing Basic ANN Integration")
    print("=" * 40)

    try:
        # Create pipeline and data loader
        pipeline = create_model_pipeline()
        data_loader = create_data_loader()

        # Load and preprocess data
        data = data_loader.load_data()
        X, y = data_loader.preprocess_features(data)
        X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

        print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test samples")
        print(f"   Features: {X.shape[1]}, Target: {y.shape}")

        # Test ANN training with simple parameters
        print("🚀 Training ANN model...")
        simple_params = {
            'hidden_layer_sizes': (50,),
            'max_iter': 100,
            'random_state': 42
        }

        ann_results = pipeline.train_model('ann', X_train, y_train, params=simple_params)

        if 'error' in ann_results:
            print(f"❌ ANN training failed: {ann_results['error']}")
            return False

        print("✅ ANN model trained successfully")
        print(".4f")
        print(f"   📏 CV RMSE: {ann_results['metrics']['cv_rmse_mean']:.2f}")

        # Test ANN prediction
        model_id = ann_results['model_id']
        y_pred = pipeline.predict(model_id, X_test)

        if len(y_pred) != len(X_test):
            print(f"❌ Prediction length mismatch: {len(y_pred)} vs {len(X_test)}")
            return False

        print("✅ ANN predictions generated successfully")
        print(".2f")

        # Test ANN evaluation
        evaluation = pipeline.evaluate_model(model_id, X_test, y_test)
        print("✅ ANN evaluation completed")
        print(".4f")

        # Verify ANN uses StandardScaler
        if hasattr(pipeline, 'scaler') and pipeline.scaler is not None:
            scaler_type = type(pipeline.scaler).__name__
            print(f"✅ ANN uses {scaler_type} for scaling")
        else:
            print("⚠️ Scaler not properly initialized")

        return True

    except Exception as e:
        print(f"❌ ANN basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ann_basic()
    print("\n" + "=" * 40)
    if success:
        print("🎉 ANN integration test PASSED!")
        print("🚀 ANN is successfully integrated into the ML pipeline")
    else:
        print("❌ ANN integration test FAILED")
    sys.exit(0 if success else 1)