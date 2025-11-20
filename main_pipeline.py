"""
Launcher script for the advanced modeling pipeline.
Run this from the project root directory to execute the pipeline.
"""

import sys
import os


def main():
    try:
        # Ensure project root is in path for imports
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.main_pipeline import run_standard_pipeline

        print("🚀 Starting Advanced Modeling Pipeline...")
        result = run_standard_pipeline()
        if "error" in result:
            print(f"❌ Pipeline failed: {result['error']}")
            return 1
        else:
            print("✅ Pipeline completed successfully!")
            print(f"📊 Experiment ID: {result['experiment_id']}")

            # Show best model from traditional comparison
            best_model = result.get('modeling_results', {}).get('best_model', 'N/A')
            print(f"🏆 Best model (traditional): {best_model}")

            # Show comprehensive comparison results if available
            comprehensive = result.get('modeling_results', {}).get('comprehensive_comparison')
            if comprehensive and 'error' not in comprehensive:
                comp_best = comprehensive['best_model']
                print(f"🏆 Best model (comprehensive): {comp_best['name']} (R² = {comp_best['r2_score']:.4f})")

                # Show KNN-specific info
                knn_analysis = comprehensive.get('knn_specific_analysis', {})
                if knn_analysis and 'error' not in knn_analysis:
                    print(f"🤖 KNN Performance: Rank {knn_analysis['knn_rank']}, R² = {knn_analysis['knn_r2_score']:.4f}")

                # Show top 3 models
                rankings = comprehensive.get('model_rankings', [])[:3]
                print(f"\n📈 Top 3 Models:")
                for i, model in enumerate(rankings, 1):
                    knn_indicator = " ← KNN" if model['is_knn'] else ""
                    print(f"  {i}. {model['model_name']}: R² = {model['r2_score']:.4f}{knn_indicator}")

            return 0
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
