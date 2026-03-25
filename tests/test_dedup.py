"""Tests for agent-dedup. Real tests, no stubs."""

import time
import pytest
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_dedup import (
    DedupCache,
    dedup,
    content_hash,
    CacheStats,
    DuplicateCallError,
)


# ── content_hash tests ────────────────────────────────────────────────────────

class TestContentHash:
    def test_stable_for_same_args(self):
        h1 = content_hash("hello", "world")
        h2 = content_hash("hello", "world")
        assert h1 == h2

    def test_different_for_different_args(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_kwargs_order_independent(self):
        h1 = content_hash(model="gpt-4", prompt="hi")
        h2 = content_hash(prompt="hi", model="gpt-4")
        assert h1 == h2

    def test_handles_none(self):
        h = content_hash(None)
        assert isinstance(h, str) and len(h) == 64

    def test_handles_nested_dict(self):
        h1 = content_hash({"a": {"b": 1}})
        h2 = content_hash({"a": {"b": 1}})
        assert h1 == h2

    def test_handles_list(self):
        h1 = content_hash([1, 2, 3])
        h2 = content_hash([1, 2, 3])
        assert h1 == h2

    def test_list_order_matters(self):
        h1 = content_hash([1, 2, 3])
        h2 = content_hash([3, 2, 1])
        assert h1 != h2

    def test_returns_64_char_hex(self):
        h = content_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_args(self):
        h1 = content_hash()
        h2 = content_hash()
        assert h1 == h2

    def test_custom_objects_use_str(self):
        class Obj:
            def __str__(self):
                return "fixed"

        h1 = content_hash(Obj())
        h2 = content_hash(Obj())
        assert h1 == h2


# ── DedupCache tests ──────────────────────────────────────────────────────────

class TestDedupCache:
    def test_set_and_get(self):
        cache = DedupCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_has_returns_true_for_existing(self):
        cache = DedupCache()
        cache.set("k", "v")
        assert cache.has("k")

    def test_has_returns_false_for_missing(self):
        cache = DedupCache()
        assert not cache.has("nonexistent")

    def test_get_raises_key_error_for_missing(self):
        cache = DedupCache()
        with pytest.raises(KeyError):
            cache.get("missing")

    def test_get_returns_default_for_missing(self):
        cache = DedupCache()
        result = cache.get("missing", default="fallback")
        assert result == "fallback"

    def test_contains_operator(self):
        cache = DedupCache()
        cache.set("a", 1)
        assert "a" in cache
        assert "b" not in cache

    def test_delete(self):
        cache = DedupCache()
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert not cache.has("k")

    def test_delete_nonexistent(self):
        cache = DedupCache()
        assert cache.delete("nope") is False

    def test_clear(self):
        cache = DedupCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cleared = cache.clear()
        assert cleared == 2
        assert len(cache) == 0

    def test_size_and_len(self):
        cache = DedupCache()
        assert len(cache) == 0
        cache.set("x", 1)
        assert len(cache) == 1
        cache.set("y", 2)
        assert len(cache) == 2

    def test_keys(self):
        cache = DedupCache()
        cache.set("a", 1)
        cache.set("b", 2)
        keys = cache.keys()
        assert set(keys) == {"a", "b"}

    def test_ttl_expiration(self):
        cache = DedupCache(ttl=0.05)  # 50ms TTL
        cache.set("k", "v")
        assert cache.has("k")
        time.sleep(0.1)
        assert not cache.has("k")

    def test_ttl_get_returns_default_after_expiry(self):
        cache = DedupCache(ttl=0.05)
        cache.set("k", "v")
        time.sleep(0.1)
        result = cache.get("k", default="expired")
        assert result == "expired"

    def test_lru_eviction(self):
        cache = DedupCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert len(cache) == 3
        # Access 'a' to make it recently used
        cache.get("a")
        # Add 'd' -> should evict 'b' (least recently used)
        cache.set("d", 4)
        assert len(cache) == 3
        assert "a" in cache
        assert "c" in cache
        assert "d" in cache
        # 'b' was evicted
        assert "b" not in cache

    def test_stats_track_hits_and_misses(self):
        cache = DedupCache()
        cache.set("k", "v")
        cache.get("k")  # hit
        cache.get("k")  # hit
        cache.get("missing", default=None)  # miss
        assert cache.stats.hits == 2
        assert cache.stats.misses == 1

    def test_stats_hit_rate(self):
        cache = DedupCache()
        cache.set("k", "v")
        cache.get("k")  # hit
        cache.get("missing", default=None)  # miss
        assert cache.stats.hit_rate == 0.5

    def test_stats_eviction_count(self):
        cache = DedupCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts 'a'
        assert cache.stats.evictions == 1

    def test_stats_to_dict(self):
        cache = DedupCache()
        d = cache.stats.to_dict()
        assert "hits" in d
        assert "misses" in d
        assert "hit_rate" in d

    def test_thread_safety(self):
        cache = DedupCache(max_size=100)
        errors = []

        def worker(i):
            try:
                cache.set(f"key{i}", i)
                val = cache.get(f"key{i}")
                assert val == i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"

    def test_namespace(self):
        c1 = DedupCache(namespace="ns1")
        c2 = DedupCache(namespace="ns2")
        c1.set("k", "v1")
        c2.set("k", "v2")
        assert c1.get("k") == "v1"
        assert c2.get("k") == "v2"

    def test_repr_contains_namespace(self):
        cache = DedupCache(namespace="test-ns")
        assert "test-ns" in repr(cache)


# ── @dedup decorator tests ────────────────────────────────────────────────────

class TestDedupDecorator:
    def test_caches_result(self):
        call_count = [0]

        @dedup(ttl=60)
        def expensive(prompt: str) -> str:
            call_count[0] += 1
            return f"result:{prompt}"

        r1 = expensive("hello")
        r2 = expensive("hello")
        assert r1 == r2 == "result:hello"
        assert call_count[0] == 1  # called only once

    def test_different_args_different_cache_entries(self):
        call_count = [0]

        @dedup(ttl=60)
        def fn(x: str) -> str:
            call_count[0] += 1
            return x.upper()

        fn("a")
        fn("b")
        fn("a")  # cached
        assert call_count[0] == 2

    def test_raise_on_dup(self):
        @dedup(ttl=60, raise_on_dup=True)
        def fn(x: str) -> str:
            return x

        fn("test")  # first call OK
        with pytest.raises(DuplicateCallError):
            fn("test")  # second call → duplicate error

    def test_exposes_cache_attribute(self):
        @dedup(ttl=60, namespace="my-ns")
        def fn():
            pass

        assert hasattr(fn, "_dedup_cache")
        assert isinstance(fn._dedup_cache, DedupCache)
        assert fn._dedup_namespace == "my-ns"

    def test_custom_key_fn(self):
        call_count = [0]

        # key based only on first arg, ignoring second
        @dedup(ttl=60, key_fn=lambda a, b: a)
        def fn(a: str, b: str) -> str:
            call_count[0] += 1
            return a + b

        fn("x", "1")
        fn("x", "999")  # same key -> cached
        assert call_count[0] == 1

    def test_preserves_function_name(self):
        @dedup()
        def my_llm_call():
            pass

        assert my_llm_call.__name__ == "my_llm_call"

    def test_ttl_expiry_triggers_recalculation(self):
        call_count = [0]

        @dedup(ttl=0.05)  # 50ms
        def fn(x: str) -> str:
            call_count[0] += 1
            return x

        fn("hi")
        fn("hi")  # cached
        assert call_count[0] == 1
        time.sleep(0.1)
        fn("hi")  # expired → recalculate
        assert call_count[0] == 2

    def test_kwargs_deduplicated(self):
        call_count = [0]

        @dedup(ttl=60)
        def fn(prompt: str, model: str = "gpt-4") -> str:
            call_count[0] += 1
            return f"{prompt}:{model}"

        fn("hello", model="gpt-4")
        fn(prompt="hello", model="gpt-4")  # same content -> cached
        assert call_count[0] == 1
