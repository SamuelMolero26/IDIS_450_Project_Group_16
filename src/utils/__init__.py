"""Utils compatibility package.

Re-export small utility modules from src/ root so `src.utils.config` works.
"""
import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import config

__all__ = ['config']
