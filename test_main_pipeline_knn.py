#!/usr/bin/env python3
"""
Test script for main pipeline with KNN integration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main_pipeline import create_advanced_pipeline

def test_main_pipeline_with_knn():
    """Test main pipeline with KNN included."""

    print("🧪 Testing Main Pipeline with KNN Integration")
    print("=" * 50)

    # Create pipeline with progress disabled for testing
    pipeline = create_advanced_pipeline(show_progress=False)

    # Run quick evaluation with KNN
    model_types = ['linear', 'decision_tree', 'random_forest', 'KNN']

    print(f"Running pipeline with models: {model_types}")

    try:
        results = pipeline.run_quick_evaluation(model_types)

        if 'error' in results:
            print(f"❌ Pipeline failed: {results['error']}")
            return False

        print("✅ Pipeline completed successfully!")
        print(f"📊 Experiment ID: {results['experiment_id']}")

        # Check modeling results
        modeling = results.get('modeling_results', {})
        if not modeling:
            print("❌ No modeling results found")
            return False

        # Check that KNN was included
        model_results = modeling.get('model_results', {})
        if 'KNN' not in model_results:
            print("❌ KNN not found in model results")
            return False

        knn_result = model_results['KNN']
        if 'error' in knn_result:
            print(f"❌ KNN training failed: {knn_result['error']}")
            return False

        print("✅ KNN successfully trained and evaluated")

        # Check comprehensive comparison
        comprehensive = modeling.get('comprehensive_comparison')
        if comprehensive and 'error' not in comprehensive:
            best_model = comprehensive['best_model']
            print(f"🏆 Best model (comprehensive): {best_model['name']} (R² = {best_model['r2_score']:.4f})")

            # Check KNN ranking
            rankings = comprehensive.get('model_rankings', [])
            knn_ranking = next((r for r in rankings if r['is_knn']), None)
            if knn_ranking:
                print(f"🤖 KNN Ranking: #{knn_ranking['rank_by_r2']} (R² = {knn_ranking['r2_score']:.4f})")

            # Check visualizations
            visualizations = comprehensive.get('visualizations', {})
            if visualizations:
                print(f"📊 Generated {len(visualizations)} visualization(s)")
                for viz_name in visualizations.keys():
                    print(f"   - {viz_name}")

        else:
            print("⚠️ Comprehensive comparison not available")

        # Check traditional comparison
        comparison = modeling.get('model_results', {}).get('comparison')
        if comparison and 'best_model' in comparison:
            print(f"🏆 Best model (traditional): {comparison['best_model']}")

        print("\n✅ All tests passed! KNN integration working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_main_pipeline_with_knn()
    sys.exit(0 if success else 1)