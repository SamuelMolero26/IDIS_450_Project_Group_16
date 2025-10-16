"""
Logging utilities for the advanced modeling pipeline.
"""

import logging
import logging.handlers
from pathlib import Path
import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT
except ImportError as e:
    print(f"Import error in logger.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

def setup_logger(name: str, log_file: str = None, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = LOGS_DIR / log_file
        LOGS_DIR.mkdir(exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Global logger instances
pipeline_logger = setup_logger('pipeline', 'pipeline.log')
model_logger = setup_logger('model', 'model.log')
evaluation_logger = setup_logger('evaluation', 'evaluation.log')
cache_logger = setup_logger('cache', 'cache.log')
meta_logger = setup_logger('meta', 'meta.log')

def log_experiment_start(experiment_id: str, config: dict):
    """Log the start of an experiment."""
    pipeline_logger.info(f"Starting experiment {experiment_id}")
    pipeline_logger.info(f"Configuration: {config}")

def log_experiment_end(experiment_id: str, results: dict):
    """Log the end of an experiment."""
    pipeline_logger.info(f"Completed experiment {experiment_id}")
    pipeline_logger.info(f"Results: {results}")

def log_model_training(model_name: str, params: dict):
    """Log model training details."""
    model_logger.info(f"Training {model_name} with parameters: {params}")

def log_evaluation_metrics(metrics: dict, model_name: str):
    """Log evaluation metrics."""
    evaluation_logger.info(f"Evaluation metrics for {model_name}: {metrics}")

def log_cache_operation(operation: str, key: str, success: bool):
    """Log cache operations."""
    status = "SUCCESS" if success else "FAILED"
    cache_logger.info(f"Cache {operation} - Key: {key} - Status: {status}")

def log_meta_learning_update(update_info: dict):
    """Log meta-learning updates."""
    meta_logger.info(f"Meta-learning update: {update_info}")