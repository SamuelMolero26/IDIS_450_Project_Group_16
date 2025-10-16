"""Infra compatibility package.

This package re-exports modules that currently live at `src/` root so code can import
`src.infra.logger`, `src.infra.redis_cache`, etc., without moving files.
"""
from .. import logger as logger
from .. import version_control as version_control
from .. import redis_cache as redis_cache

__all__ = [
    'logger', 'version_control', 'redis_cache'
]
