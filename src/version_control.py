"""
Version control system for datasets and models.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import shutil

import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import MODELS_DIR, CACHE_DIR
    from src.logger import pipeline_logger
except ImportError as e:
    print(f"Import error in version_control.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

class VersionControl:
    """
    Version control system for tracking dataset and model changes.
    """

    def __init__(self):
        self.versions_dir = CACHE_DIR / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.current_versions = self._load_version_history()

    def _load_version_history(self) -> Dict[str, Any]:
        """
        Load version history from disk.

        Returns:
            Dictionary with version history
        """
        history_file = self.versions_dir / "version_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                pipeline_logger.error(f"Error loading version history: {e}")
                return {}
        else:
            return {}

    def _save_version_history(self):
        """Save version history to disk."""
        history_file = self.versions_dir / "version_history.json"

        try:
            with open(history_file, 'w') as f:
                json.dump(self.current_versions, f, indent=2, default=str)
        except Exception as e:
            pipeline_logger.error(f"Error saving version history: {e}")

    def create_dataset_version(self, data_hash: str, metadata: Dict[str, Any]) -> str:
        """
        Create a new dataset version.

        Args:
            data_hash: Hash of the dataset
            metadata: Additional metadata about the dataset

        Returns:
            Version identifier
        """
        version_id = f"dataset_{data_hash[:12]}"
        timestamp = datetime.now().isoformat()

        version_info = {
            'version_id': version_id,
            'type': 'dataset',
            'data_hash': data_hash,
            'created_at': timestamp,
            'metadata': metadata
        }

        # Store version info
        if 'datasets' not in self.current_versions:
            self.current_versions['datasets'] = {}

        self.current_versions['datasets'][version_id] = version_info
        self._save_version_history()

        pipeline_logger.info(f"Created dataset version: {version_id}")

        return version_id

    def create_model_version(self, model_name: str, model_data: Dict[str, Any],
                           performance_metrics: Dict[str, Any]) -> str:
        """
        Create a new model version.

        Args:
            model_name: Name of the model
            model_data: Model information and parameters
            performance_metrics: Model performance metrics

        Returns:
            Version identifier
        """
        # Create version hash from model data
        version_string = f"{model_name}_{json.dumps(model_data, sort_keys=True)}_{json.dumps(performance_metrics, sort_keys=True)}"
        version_hash = hashlib.md5(version_string.encode()).hexdigest()[:12]
        version_id = f"model_{model_name}_{version_hash}"
        timestamp = datetime.now().isoformat()

        version_info = {
            'version_id': version_id,
            'type': 'model',
            'model_name': model_name,
            'created_at': timestamp,
            'model_data': model_data,
            'performance_metrics': performance_metrics,
            'version_hash': version_hash
        }

        # Store version info
        if 'models' not in self.current_versions:
            self.current_versions['models'] = {}

        self.current_versions['models'][version_id] = version_info
        self._save_version_history()

        pipeline_logger.info(f"Created model version: {version_id}")

        return version_id

    def get_dataset_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset version information.

        Args:
            version_id: Version identifier

        Returns:
            Version information or None if not found
        """
        return self.current_versions.get('datasets', {}).get(version_id)

    def get_model_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model version information.

        Args:
            version_id: Version identifier

        Returns:
            Version information or None if not found
        """
        return self.current_versions.get('models', {}).get(version_id)

    def list_dataset_versions(self) -> List[Dict[str, Any]]:
        """
        List all dataset versions.

        Returns:
            List of dataset version information
        """
        return list(self.current_versions.get('datasets', {}).values())

    def list_model_versions(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all model versions, optionally filtered by model name.

        Args:
            model_name: Optional model name filter

        Returns:
            List of model version information
        """
        models = self.current_versions.get('models', {})

        if model_name:
            return [v for v in models.values() if v['model_name'] == model_name]
        else:
            return list(models.values())

    def get_latest_dataset_version(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest dataset version.

        Returns:
            Latest dataset version information or None
        """
        datasets = self.current_versions.get('datasets', {})

        if not datasets:
            return None

        # Sort by creation time
        sorted_datasets = sorted(datasets.values(),
                               key=lambda x: x['created_at'], reverse=True)

        return sorted_datasets[0]

    def get_latest_model_version(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Latest model version information or None
        """
        model_versions = self.list_model_versions(model_name)

        if not model_versions:
            return None

        # Sort by creation time
        sorted_versions = sorted(model_versions,
                               key=lambda x: x['created_at'], reverse=True)

        return sorted_versions[0]

    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Any]:
        """
        Compare two versions.

        Args:
            version_id_1: First version ID
            version_id_2: Second version ID

        Returns:
            Comparison results
        """
        version_1 = self._get_version_by_id(version_id_1)
        version_2 = self._get_version_by_id(version_id_2)

        if not version_1 or not version_2:
            return {'error': 'One or both versions not found'}

        comparison = {
            'version_1': version_1,
            'version_2': version_2,
            'differences': {}
        }

        # Compare creation times
        time_1 = datetime.fromisoformat(version_1['created_at'])
        time_2 = datetime.fromisoformat(version_2['created_at'])
        comparison['time_difference_days'] = abs((time_2 - time_1).days)

        # Compare performance metrics for models
        if version_1['type'] == 'model' and version_2['type'] == 'model':
            perf_1 = version_1.get('performance_metrics', {})
            perf_2 = version_2.get('performance_metrics', {})

            for metric in set(perf_1.keys()) | set(perf_2.keys()):
                val_1 = perf_1.get(metric, 0)
                val_2 = perf_2.get(metric, 0)
                comparison['differences'][f'{metric}_diff'] = val_2 - val_1

        return comparison

    def _get_version_by_id(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get version information by ID regardless of type.

        Args:
            version_id: Version identifier

        Returns:
            Version information or None
        """
        # Check datasets
        dataset_version = self.get_dataset_version(version_id)
        if dataset_version:
            return dataset_version

        # Check models
        model_version = self.get_model_version(version_id)
        if model_version:
            return model_version

        return None

    def archive_version(self, version_id: str, archive_path: Optional[Path] = None):
        """
        Archive a version for long-term storage.

        Args:
            version_id: Version identifier to archive
            archive_path: Optional archive path
        """
        version_info = self._get_version_by_id(version_id)

        if not version_info:
            pipeline_logger.error(f"Version {version_id} not found for archiving")
            return

        if archive_path is None:
            archive_dir = self.versions_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
            archive_path = archive_dir / f"{version_id}.json"

        try:
            with open(archive_path, 'w') as f:
                json.dump(version_info, f, indent=2, default=str)

            pipeline_logger.info(f"Archived version {version_id} to {archive_path}")

        except Exception as e:
            pipeline_logger.error(f"Error archiving version {version_id}: {e}")

    def cleanup_old_versions(self, keep_recent: int = 10):
        """
        Clean up old versions, keeping only the most recent ones.

        Args:
            keep_recent: Number of recent versions to keep
        """
        # Clean up dataset versions
        datasets = self.current_versions.get('datasets', {})
        if len(datasets) > keep_recent:
            # Sort by creation time and keep most recent
            sorted_datasets = sorted(datasets.items(),
                                   key=lambda x: x[1]['created_at'], reverse=True)
            keep_datasets = dict(sorted_datasets[:keep_recent])

            # Archive old versions before removing
            for version_id, version_info in sorted_datasets[keep_recent:]:
                self.archive_version(version_id)

            self.current_versions['datasets'] = keep_datasets

        # Clean up model versions (per model type)
        models = self.current_versions.get('models', {})
        model_names = set(v['model_name'] for v in models.values())

        for model_name in model_names:
            model_versions = [v for v in models.values() if v['model_name'] == model_name]
            if len(model_versions) > keep_recent:
                # Sort and keep most recent
                sorted_versions = sorted(model_versions,
                                       key=lambda x: x['created_at'], reverse=True)
                keep_version_ids = {v['version_id'] for v in sorted_versions[:keep_recent]}

                # Archive old versions
                for version in sorted_versions[keep_recent:]:
                    self.archive_version(version['version_id'])

                # Remove old versions from current
                self.current_versions['models'] = {
                    k: v for k, v in models.items() if k in keep_version_ids
                }

        self._save_version_history()
        pipeline_logger.info(f"Cleaned up old versions, keeping {keep_recent} most recent")

    def get_version_stats(self) -> Dict[str, Any]:
        """
        Get statistics about version history.

        Returns:
            Dictionary with version statistics
        """
        stats = {
            'total_dataset_versions': len(self.current_versions.get('datasets', {})),
            'total_model_versions': len(self.current_versions.get('models', {})),
            'model_types': {}
        }

        # Model type breakdown
        models = self.current_versions.get('models', {})
        for version_info in models.values():
            model_name = version_info['model_name']
            if model_name not in stats['model_types']:
                stats['model_types'][model_name] = 0
            stats['model_types'][model_name] += 1

        # Date range
        all_versions = []
        all_versions.extend(self.current_versions.get('datasets', {}).values())
        all_versions.extend(self.current_versions.get('models', {}).values())

        if all_versions:
            creation_dates = [datetime.fromisoformat(v['created_at']) for v in all_versions]
            stats['date_range'] = {
                'earliest': min(creation_dates).isoformat(),
                'latest': max(creation_dates).isoformat(),
                'span_days': (max(creation_dates) - min(creation_dates)).days
            }

        return stats

    def export_version_history(self, export_path: Path):
        """
        Export complete version history.

        Args:
            export_path: Path to export the history
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'version_history': self.current_versions,
            'stats': self.get_version_stats()
        }

        try:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            pipeline_logger.info(f"Exported version history to {export_path}")

        except Exception as e:
            pipeline_logger.error(f"Error exporting version history: {e}")

    def import_version_history(self, import_path: Path):
        """
        Import version history from file.

        Args:
            import_path: Path to import the history from
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)

            if 'version_history' in import_data:
                self.current_versions = import_data['version_history']
                self._save_version_history()

                pipeline_logger.info(f"Imported version history from {import_path}")

        except Exception as e:
            pipeline_logger.error(f"Error importing version history: {e}")

def create_version_control() -> VersionControl:
    """
    Factory function to create VersionControl instance.

    Returns:
        VersionControl instance
    """
    return VersionControl()