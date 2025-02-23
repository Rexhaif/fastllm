import json
import xxhash
import asyncio
from typing import Any, Dict, Optional
from diskcache import Cache


class CacheProvider:
    """Base class for cache providers with async interface."""
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        raise NotImplementedError

    async def get(self, key: str):
        """Get a value from the cache."""
        raise NotImplementedError

    async def put(self, key: str, value) -> None:
        """Put a value in the cache."""
        raise NotImplementedError

    async def clear(self) -> None:
        """Clear all items from the cache."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the cache when done."""
        pass


def compute_request_hash(request: dict) -> str:
    # Create a copy of the request and remove any fields that are not part of the request content
    temp_request = request.copy()
    
    # Remove internal tracking fields that shouldn't affect caching
    temp_request.pop("_request_id", None)
    temp_request.pop("_order_id", None)
    
    # Extract known fields and extra params
    known_fields = {
        "provider", "model", "messages", "temperature", "max_completion_tokens",
        "top_p", "presence_penalty", "frequency_penalty", "stop", "stream"
    }
    
    # Separate core parameters and extra parameters
    core_params = {k: v for k, v in temp_request.items() if k in known_fields}
    extra_params = {k: v for k, v in temp_request.items() if k not in known_fields}
    
    # Create a combined dictionary with sorted extra params
    hash_dict = {
        "core": core_params,
        "extra": dict(sorted(extra_params.items()))  # Sort extra params for consistent hashing
    }
    
    # Serialize with sorted keys for a consistent representation
    request_str = json.dumps(hash_dict, sort_keys=True)
    return xxhash.xxh64(request_str.encode("utf-8")).hexdigest()


class InMemoryCache(CacheProvider):
    """Simple in-memory cache implementation using a dictionary."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache

    async def get(self, key: str):
        """Get a value from the cache."""
        if not await self.exists(key):
            raise KeyError(f"Cache for key {key} does not exist")
        return self._cache[key]

    async def put(self, key: str, value) -> None:
        """Put a value in the cache."""
        self._cache[key] = value

    async def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()


class DiskCache(CacheProvider):
    """Disk-based cache implementation using diskcache with async support."""
    
    def __init__(self, directory: str, ttl: Optional[int] = None, **cache_options):
        """Initialize disk cache.
        
        Args:
            directory: Directory where cache files will be stored
            ttl: Time to live in seconds for cached items (None means no expiration)
            **cache_options: Additional options to pass to diskcache.Cache
            
        Raises:
            OSError: If the directory is invalid or cannot be created
        """
        try:
            self._cache = Cache(directory, **cache_options)
            self._ttl = ttl
        except Exception as e:
            # Convert any cache initialization error to OSError
            raise OSError(f"Failed to initialize disk cache: {str(e)}")

    async def _run_in_executor(self, func, *args):
        """Run a blocking cache operation in the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        try:
            # Use the internal __contains__ method which is faster than get
            return await self._run_in_executor(self._cache.__contains__, key)
        except Exception as e:
            raise OSError(f"Failed to check cache key: {str(e)}")

    async def get(self, key: str):
        """Get a value from the cache."""
        try:
            value = await self._run_in_executor(self._cache.get, key)
            if value is None:  # diskcache returns None for missing keys
                raise KeyError(f"Cache for key {key} does not exist")
            return value
        except Exception as e:
            if isinstance(e, KeyError):
                raise
            raise OSError(f"Failed to get cache value: {str(e)}")

    async def put(self, key: str, value) -> None:
        """Put a value in the cache with optional TTL."""
        try:
            # Try to serialize the value to verify it's JSON serializable
            json.dumps(value)
            await self._run_in_executor(self._cache.set, key, value, self._ttl)
        except TypeError as e:
            raise TypeError(f"Value is not JSON serializable: {str(e)}")
        except Exception as e:
            raise OSError(f"Failed to store cache value: {str(e)}")

    async def clear(self) -> None:
        """Clear all items from the cache."""
        try:
            await self._run_in_executor(self._cache.clear)
        except Exception as e:
            raise OSError(f"Failed to clear cache: {str(e)}")

    async def close(self) -> None:
        """Close the cache when done."""
        try:
            await self._run_in_executor(self._cache.close)
        except Exception as e:
            raise OSError(f"Failed to close cache: {str(e)}") 