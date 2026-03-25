"""
agent-dedup: Deduplication of agent tasks and outputs using content hashing.

Prevents duplicate LLM calls by caching results keyed by a canonical hash
of the input. Zero dependencies. Pure Python 3.8+.

Features:
  - SHA-256 content hashing for stable cache keys
  - In-memory LRU-style cache with configurable TTL and max size
  - DedupCache: direct cache API
  - @dedup decorator: transparent memoization for any callable
  - Namespace support (separate caches per agent/task)
  - Cache statistics (hits, misses, evictions)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
import functools
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple


__version__ = "0.1.0"
__all__ = [
    "DedupCache",
    "dedup",
    "content_hash",
    "CacheStats",
    "DuplicateCallError",
]

_SENTINEL = object()


class DuplicateCallError(Exception):
    """Raised when strict mode detects a duplicate call and raise_on_dup=True."""
    pass


class CacheStats:
    """Tracks cache performance metrics."""

    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.expirations: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


def content_hash(*args: Any, **kwargs: Any) -> str:
    """
    Compute a stable SHA-256 hash of the given arguments.

    Supports: str, int, float, bool, None, list, tuple, dict.
    For custom objects, falls back to str(obj).

    Returns a 64-char hex string.
    """
    def _normalize(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_normalize(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _normalize(v) for k, v in sorted(obj.items())}
        # fallback
        return str(obj)

    payload = {
        "args": [_normalize(a) for a in args],
        "kwargs": {k: _normalize(v) for k, v in sorted(kwargs.items())},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class _CacheEntry:
    __slots__ = ("value", "created_at", "expires_at")

    def __init__(self, value: Any, ttl: Optional[float]):
        self.value = value
        self.created_at = time.monotonic()
        self.expires_at = (self.created_at + ttl) if ttl is not None else None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.monotonic() > self.expires_at


class DedupCache:
    """
    Thread-safe in-memory deduplication cache with TTL and LRU eviction.

    Usage:
        cache = DedupCache(ttl=300, max_size=1000)
        key = content_hash("prompt text", model="gpt-4")
        if not cache.has(key):
            result = call_llm(...)
            cache.set(key, result)
        else:
            result = cache.get(key)
    """

    def __init__(
        self,
        ttl: Optional[float] = None,
        max_size: int = 1024,
        namespace: str = "default",
    ):
        """
        Args:
            ttl: Time-to-live in seconds. None = never expires.
            max_size: Max number of entries (LRU eviction when exceeded).
            namespace: Logical namespace for this cache instance.
        """
        self.ttl = ttl
        self.max_size = max_size
        self.namespace = namespace
        self.stats = CacheStats()
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def _evict_expired(self) -> None:
        """Remove all expired entries (called under lock)."""
        expired_keys = [k for k, entry in self._store.items() if entry.is_expired()]
        for k in expired_keys:
            del self._store[k]
            self.stats.expirations += 1

    def has(self, key: str) -> bool:
        """Return True if key exists and is not expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._store[key]
                self.stats.expirations += 1
                return False
            # Move to end (LRU: recently used)
            self._store.move_to_end(key)
            return True

    def get(self, key: str, default: Any = _SENTINEL) -> Any:
        """
        Retrieve a cached value by key.

        Raises KeyError if key is missing and no default provided.
        Counts hit/miss statistics.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.is_expired():
                if entry is not None:
                    del self._store[key]
                    self.stats.expirations += 1
                self.stats.misses += 1
                if default is _SENTINEL:
                    raise KeyError(key)
                return default
            self._store.move_to_end(key)
            self.stats.hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = _CacheEntry(value, self.ttl)
            # LRU eviction
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
                self.stats.evictions += 1

    def delete(self, key: str) -> bool:
        """Remove an entry. Returns True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries. Returns number of entries cleared."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def size(self) -> int:
        """Current number of entries (including potentially expired ones)."""
        return len(self._store)

    def keys(self) -> List[str]:
        """Return all current (non-expired) cache keys."""
        with self._lock:
            self._evict_expired()
            return list(self._store.keys())

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return (
            f"DedupCache(namespace={self.namespace!r}, "
            f"size={len(self._store)}, ttl={self.ttl}, "
            f"stats={self.stats})"
        )


# Global registry of named caches
_global_caches: Dict[str, DedupCache] = {}
_global_lock = threading.Lock()


def _get_or_create_cache(
    namespace: str,
    ttl: Optional[float],
    max_size: int,
) -> DedupCache:
    with _global_lock:
        if namespace not in _global_caches:
            _global_caches[namespace] = DedupCache(
                ttl=ttl, max_size=max_size, namespace=namespace
            )
        return _global_caches[namespace]


def dedup(
    ttl: Optional[float] = None,
    max_size: int = 1024,
    namespace: Optional[str] = None,
    key_fn: Optional[Callable[..., str]] = None,
    raise_on_dup: bool = False,
) -> Callable:
    """
    Decorator that deduplicates function calls by caching results.

    The cache key is computed from the function arguments using content_hash
    (or a custom key_fn if provided).

    Args:
        ttl: Cache entry lifetime in seconds. None = never expires.
        max_size: Maximum cache entries (LRU eviction).
        namespace: Cache namespace (defaults to function's qualified name).
        key_fn: Custom function to compute cache key from args/kwargs.
                Signature: key_fn(*args, **kwargs) -> str
        raise_on_dup: If True, raise DuplicateCallError on cache hit
                      instead of returning cached value. Useful for testing.

    Usage:
        @dedup(ttl=300, namespace="llm-calls")
        def call_llm(prompt: str, model: str = "gpt-4") -> str:
            return expensive_llm_call(prompt, model)
    """
    def decorator(fn: Callable) -> Callable:
        ns = namespace or f"{fn.__module__}.{fn.__qualname__}"
        cache = _get_or_create_cache(ns, ttl, max_size)
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                # Bind all positional args to their parameter names so that
                # fn("x", model="gpt-4") and fn(prompt="x", model="gpt-4") hash identically.
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    key = content_hash(**dict(bound.arguments))
                except (TypeError, ValueError):
                    # Fallback: hash raw args/kwargs
                    key = content_hash(*args, **kwargs)

            try:
                cached = cache.get(key)
                if raise_on_dup:
                    raise DuplicateCallError(
                        f"Duplicate call detected (key={key[:16]}...)"
                    )
                return cached
            except KeyError:
                result = fn(*args, **kwargs)
                cache.set(key, result)
                return result

        wrapper._dedup_cache = cache
        wrapper._dedup_namespace = ns
        return wrapper

    return decorator
