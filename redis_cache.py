"""
Redis cache implementation with SQLite fallback for the advanced modeling pipeline.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import pickle


class RedisCache:
    """
    Redis cache implementation with SQLite fallback.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "cache.db"
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
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}

    def save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def _make_serializable(self, obj):
        """
        Convert object to JSON-serializable format, with fallback to pickle for complex objects.

        Args:
            obj: Object to make serializable

        Returns:
            Serializable version of the object
        """
        # Handle None early
        if obj is None:
            return None

        # Handle pandas Interval types first (before dict/list checks)
        if hasattr(pd, 'Interval') and isinstance(obj, pd.Interval):
            return str(obj)

        # Handle pandas IntervalIndex
        if hasattr(pd, 'IntervalIndex') and isinstance(obj, pd.IntervalIndex):
            return [str(interval) for interval in obj]

        # Handle pandas Series with Interval dtype
        if isinstance(obj, pd.Series):
            if hasattr(obj.dtype, 'name') and 'interval' in str(obj.dtype).lower():
                return obj.astype(str).tolist()
            return obj.tolist()

        # Handle pandas Categorical
        if isinstance(obj, pd.Categorical):
            return obj.tolist()

        # Handle pandas Timestamp
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        # Handle pandas Index types
        if isinstance(obj, pd.Index):
            # Check if it's an IntervalIndex
            if hasattr(pd, 'IntervalIndex') and isinstance(obj, pd.IntervalIndex):
                return [str(interval) for interval in obj]
            # For other Index types, check if elements are Intervals
            try:
                result = obj.tolist()
                # If the first element is an Interval, convert all to strings
                if result and hasattr(pd, 'Interval') and isinstance(result[0], pd.Interval):
                    return [str(item) for item in result]
                return result
            except:
                return [str(item) for item in obj]

        # Handle dicts and lists recursively
        if isinstance(obj, dict):
            # Convert both keys and values to ensure JSON serializability
            serializable_dict = {}
            for k, v in obj.items():
                # Convert non-primitive keys to strings
                if isinstance(k, (str, int, float, bool, type(None))):
                    key = k
                elif hasattr(pd, 'Interval') and isinstance(k, pd.Interval):
                    key = str(k)
                else:
                    # For any other non-primitive type, convert to string
                    key = str(k)
                serializable_dict[key] = self._make_serializable(v)
            return serializable_dict
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]

        # Handle numpy types
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)

        # Handle datetime types
        elif isinstance(obj, (datetime)):
            return obj.isoformat()

        # Handle bytes objects
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')

        # Handle warning objects (like IntervalConvergenceWarning)
        elif hasattr(obj, '__class__') and 'Warning' in obj.__class__.__name__:
            return str(obj)

        # For custom objects with __dict__
        elif hasattr(obj, '__dict__'):
            try:
                return self._make_serializable(obj.__dict__)
            except:
                return str(obj)

        # Final attempt: try direct JSON serialization
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # If JSON serialization fails, use pickle as fallback
                try:
                    return {"__pickled__": True, "__data__": pickle.dumps(obj).decode('latin1')}
                except:
                    return str(obj)

    def _deserialize_object(self, obj):
        """
        Deserialize object that may have been pickled.

        Args:
            obj: Object to deserialize

        Returns:
            Deserialized object
        """
        if isinstance(obj, dict) and obj.get("__pickled__", False):
            try:
                return pickle.loads(obj["__data__"].encode('latin1'))
            except:
                return obj
        return obj

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a cache value."""
        try:
            serializable_value = self._make_serializable(value)
            if self.redis_client:
                self.redis_client.set(key, json.dumps(serializable_value), ex=ttl)
            else:
                self.cache[key] = {"value": serializable_value, "ttl": ttl}
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
                if value:
                    deserialized = json.loads(value)
                    return self._deserialize_object(deserialized)
                return None
            else:
                if key in self.cache:
                    return self._deserialize_object(self.cache[key]["value"])
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
                    "entries": info.get("db0", {}).get("keys", 0),
                    "backend": "redis",
                }
            else:
                return {"entries": len(self.cache), "backend": "sqlite"}
        except Exception as e:
            print(f"Cache stats failed: {e}")
            return {"entries": 0, "backend": "error"}


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


__all__ = [
    "cache",
    "cache_evaluation_metrics",
    "cache_model_results",
    "get_cached_model_results",
    "get_cached_evaluation_metrics",
    "RedisCache",
]
