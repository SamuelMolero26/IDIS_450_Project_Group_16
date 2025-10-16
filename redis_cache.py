"""
Redis cache implementation with SQLite fallback for the advanced modeling pipeline.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

class RedisCache:
    """
    Redis cache implementation with SQLite fallback.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'cache.db'
        self.cache = {}
        self._init_redis()
        self.load_cache()

    def _init_redis(self):
        """Initialize Redis client with fallback to SQLite."""
        try:
            import redis
            self.redis_client = redis.from_url(self.redis_url)
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}. Using SQLite fallback.")
            self.redis_client = None

    def load_cache(self):
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}

    def save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a cache value."""
        try:
            if self.redis_client:
                self.redis_client.set(key, json.dumps(value), ex=ttl)
            else:
                self.cache[key] = {'value': value, 'ttl': ttl}
                self.save_cache()
            return True
        except Exception as e:
            print(f"Cache set failed: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Get a cache value."""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                if key in self.cache:
                    return self.cache[key]['value']
                return None
        except Exception as e:
            print(f"Cache get failed: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a cache key."""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                if key in self.cache:
                    del self.cache[key]
                    self.save_cache()
                    return True
                return False
        except Exception as e:
            print(f"Cache delete failed: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if a cache key exists."""
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                return key in self.cache
        except Exception as e:
            print(f"Cache exists failed: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.flushdb())
            else:
                self.cache = {}
                self.save_cache()
                return True
        except Exception as e:
            print(f"Cache clear failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    'entries': info.get('db0', {}).get('keys', 0),
                    'backend': 'redis'
                }
            else:
                return {
                    'entries': len(self.cache),
                    'backend': 'sqlite'
                }
        except Exception as e:
            print(f"Cache stats failed: {e}")
            return {'entries': 0, 'backend': 'error'}

# Global cache instance
cache = RedisCache()

def cache_evaluation_metrics(key: str, metrics: dict):
    """
    Cache evaluation metrics for a model.

    Args:
        key: Cache key
        metrics: Metrics dictionary
    """
    cache.set(f"eval_{key}", metrics)

def cache_model_results(model_type: str, params: dict, results: dict):
    """
    Cache model training results.

    Args:
        model_type: Type of model
        params: Model parameters
        results: Training results
    """
    import hashlib
    key_data = f"{model_type}_{json.dumps(params, sort_keys=True)}"
    key = hashlib.md5(key_data.encode()).hexdigest()
    cache.set(f"model_{key}", results)

def get_cached_model_results(model_type: str, params: dict) -> Optional[dict]:
    """
    Get cached model results.

    Args:
        model_type: Type of model
        params: Model parameters

    Returns:
        Cached results or None
    """
    import hashlib
    key_data = f"{model_type}_{json.dumps(params, sort_keys=True)}"
    key = hashlib.md5(key_data.encode()).hexdigest()
    return cache.get(f"model_{key}")

def get_cached_evaluation_metrics(key: str) -> Optional[dict]:
    """
    Get cached evaluation metrics.

    Args:
        key: Cache key

    Returns:
        Cached metrics or None
    """
    return cache.get(f"eval_{key}")

__all__ = ['cache', 'cache_evaluation_metrics', 'cache_model_results', 'get_cached_model_results', 'get_cached_evaluation_metrics', 'RedisCache']