"""Data compatibility package.

Re-exports top-level data modules located in src/ so `src.data.data_loader` works.
"""
from .. import data_loader as data_loader
from .. import data_preprocessing as data_preprocessing

__all__ = ['data_loader', 'data_preprocessing']
