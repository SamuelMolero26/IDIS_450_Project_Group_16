"""Learning compatibility package.

Re-export learning-related modules from src/ root.
"""
from .. import continuous_learning as continuous_learning
from .. import qualitative_evaluator as qualitative_evaluator

__all__ = ['continuous_learning', 'qualitative_evaluator']
