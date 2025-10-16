"""Top-level src package. Re-export commonly used modules to maintain backward compatibility.

This file keeps `import src.<subpkg>.<module>` working even when modules live directly under src/.
"""
from . import config  # re-export config

__all__ = [
    'config',
]
