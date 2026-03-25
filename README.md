# agent-dedup

Deduplication of agent tasks and outputs using content hashing. Prevents duplicate LLM calls. Zero dependencies. Pure Python 3.8+.

## Install

```bash
pip install agent-dedup
```

## Features

- SHA-256 content hashing for stable cache keys
- In-memory LRU cache with configurable TTL and max size
- Thread-safe for concurrent agents
- Namespace support (separate caches per agent/task)
- Cache statistics (hits, misses, evictions)
- `@dedup` decorator for transparent memoization

## Usage

### `@dedup` decorator

```python
from agent_dedup import dedup

@dedup(ttl=300, namespace="llm-calls", max_size=2048)
def call_llm(prompt: str, model: str = "gpt-4") -> str:
    # This will only execute once per unique (prompt, model) combination
    return openai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
```

### Direct `DedupCache` API

```python
from agent_dedup import DedupCache, content_hash

cache = DedupCache(ttl=600, max_size=1000, namespace="my-agent")

key = content_hash("What is 2+2?", model="gpt-4")

if key not in cache:
    result = call_llm("What is 2+2?")
    cache.set(key, result)

result = cache.get(key)
print(cache.stats)  # CacheStats(hits=1, misses=1, hit_rate=50.0%)
```

### Custom key function

```python
# Deduplicate only by prompt text, ignoring metadata
@dedup(ttl=300, key_fn=lambda prompt, **meta: content_hash(prompt))
def call_with_meta(prompt: str, request_id: str = "") -> str:
    ...
```

### Cache statistics

```python
fn._dedup_cache.stats.to_dict()
# {"hits": 42, "misses": 8, "evictions": 0, "hit_rate": 0.84}
```

## License

MIT
