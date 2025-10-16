"""Models compatibility package.

Re-export model modules located in src/ so `src.models.model_pipeline` etc. work.
"""
from .. import model_pipeline as model_pipeline
from .. import evaluation_engine as evaluation_engine
from .. import meta_learner as meta_learner

__all__ = ['model_pipeline', 'evaluation_engine', 'meta_learner']
