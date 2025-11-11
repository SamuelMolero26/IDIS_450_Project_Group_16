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
            print(
                f"🏆 Best model: {result.get('modeling_results', {}).get('best_model', 'N/A')}"
            )
            return 0
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
