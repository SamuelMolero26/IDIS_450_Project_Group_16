"""
Data loading and preprocessing integration module.
"""

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

from .config import (
    PREPROCESSED_DATA_FILE, TARGET_COLUMN, NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES, RANDOM_STATE, TEST_SIZE
)
from .logger import pipeline_logger

class DataLoader:
    """
    Handles data loading, versioning, and preprocessing integration.
    """

    def __init__(self):
        self.data_hash = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = TARGET_COLUMN

    def load_data(self, file_path: Path = PREPROCESSED_DATA_FILE) -> pd.DataFrame:
        """
        Load preprocessed data from CSV file.

        Args:
            file_path: Path to the data file

        Returns:
            Loaded DataFrame
        """
        try:
            pipeline_logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)

            # Calculate data hash for versioning
            self.data_hash = self._calculate_data_hash(df)

            pipeline_logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            pipeline_logger.info(f"Data hash: {self.data_hash}")

            return df

        except Exception as e:
            pipeline_logger.error(f"Error loading data: {e}")
            raise

    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate hash of the dataset for versioning.

        Args:
            df: Input DataFrame

        Returns:
            SHA256 hash string
        """
        # Convert to string representation for hashing
        data_str = df.to_csv(index=False)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Preprocess features for modeling.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (processed features DataFrame, target array)
        """
        pipeline_logger.info("Preprocessing features")

        # Separate features and target
        feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        X = df[feature_cols].copy()
        y = df[self.target_column].values

        # Handle categorical features
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col in X.columns:
                if cat_col not in self.label_encoders:
                    self.label_encoders[cat_col] = LabelEncoder()
                    X[cat_col] = self.label_encoders[cat_col].fit_transform(X[cat_col])
                else:
                    X[cat_col] = self.label_encoders[cat_col].transform(X[cat_col])

        # Scale numerical features
        num_cols = [col for col in NUMERICAL_FEATURES if col in X.columns]
        if num_cols:
            X[num_cols] = self.scaler.fit_transform(X[num_cols])

        self.feature_columns = list(X.columns)

        pipeline_logger.info(f"Processed {len(self.feature_columns)} features")
        return X, y

    def split_data(self, X: pd.DataFrame, y: np.ndarray,
                  test_size: float = TEST_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.

        Args:
            X: Feature DataFrame
            y: Target array
            test_size: Test set proportion

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        pipeline_logger.info(f"Splitting data with test_size={test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )

        pipeline_logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.

        Returns:
            Dictionary with data information
        """
        return {
            'data_hash': self.data_hash,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'scaler_params': {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
            },
            'label_encoders': {
                col: encoder.classes_.tolist() for col, encoder in self.label_encoders.items()
            }
        }

    def save_preprocessing_info(self, save_path: Path):
        """
        Save preprocessing information for reproducibility.

        Args:
            save_path: Path to save the information
        """
        info = self.get_data_info()

        with open(save_path, 'w') as f:
            json.dump(info, f, indent=2)

        pipeline_logger.info(f"Saved preprocessing info to {save_path}")

    def load_preprocessing_info(self, load_path: Path):
        """
        Load preprocessing information.

        Args:
            load_path: Path to load the information from
        """
        with open(load_path, 'r') as f:
            info = json.load(f)

        self.data_hash = info.get('data_hash')
        self.feature_columns = info.get('feature_columns')
        self.target_column = info.get('target_column')

        # Restore scaler
        if info.get('scaler_params', {}).get('mean'):
            self.scaler.mean_ = np.array(info['scaler_params']['mean'])
            self.scaler.scale_ = np.array(info['scaler_params']['scale'])

        # Restore label encoders
        for col, classes in info.get('label_encoders', {}).items():
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].classes_ = np.array(classes)

        pipeline_logger.info(f"Loaded preprocessing info from {load_path}")

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return summary.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_stats': {}
        }

        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            quality_report['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum()
            }

        # Check for outliers using IQR method
        outlier_cols = [col for col in numeric_cols if 'outlier' in col.lower()]
        if outlier_cols:
            quality_report['outlier_summary'] = {}
            for col in outlier_cols:
                outliers = df[col].sum()  # Assuming 1 indicates outlier
                quality_report['outlier_summary'][col] = {
                    'outlier_count': int(outliers),
                    'outlier_percentage': float(outliers / len(df) * 100)
                }

        pipeline_logger.info("Data quality validation completed")
        return quality_report

def create_data_loader() -> DataLoader:
    """
    Factory function to create DataLoader instance.

    Returns:
        DataLoader instance
    """
    return DataLoader()