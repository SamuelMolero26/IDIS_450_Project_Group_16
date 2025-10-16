"""Utils compatibility package.

Re-export visualization utilities and config so callers can import either
`from src.utils import visualization_utils` or (during migration) `from utils import ...`.
"""
import sys
import os
from importlib import import_module

# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

__all__ = []

# Export config (prefer package-local import)
try:
    from .. import config as config  # type: ignore
    __all__.append("config")
except Exception:
    try:
        config = import_module("src.config")
        __all__.append("config")
    except Exception:
        # leave config out if not found; importing modules can handle absence
        config = None

# Export visualization_utils (prefer local src/utils/visualization_utils.py)
try:
    from . import visualization_utils  # type: ignore
    __all__.append("visualization_utils")
except Exception:
    try:
        mod = import_module("utils.visualization_utils")  # fallback to top-level utils/
        visualization_utils = mod
        __all__.append("visualization_utils")
    except Exception as exc:
        raise ImportError(
            "visualization_utils not found in src.utils or top-level utils.\n"
            "Fix: move utils/visualization_utils.py -> src/utils/visualization_utils.py "
            "or update imports to 'from src.utils.visualization_utils import ...'."
        ) from exc