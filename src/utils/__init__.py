"""Utils compatibility package.

Re-export small utility modules from src/ root so `src.utils.config` works.
"""
import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src import config as config
except ImportError as e:
    print(f"Import error in src/utils/__init__.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

__all__ = ['config']
