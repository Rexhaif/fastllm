import asyncio
import os

import pytest

from fastllm.cache import DiskCache, InMemoryCache, compute_request_hash


@pytest.mark.asyncio
async def test_inmemory_cache_put_get_exists_and_clear():
    cache = InMemoryCache()
    key = "test_key"
    value = {"foo": "bar"}

    # Initially, key should not exist
    assert not await cache.exists(key)

    # Put value
    await cache.put(key, value)

    # Check existence
    assert await cache.exists(key)

    # Get the value back
    retrieved_val = await cache.get(key)
    assert retrieved_val == value

    # Clear cache
    await cache.clear()
    assert not await cache.exists(key)
    with pytest.raises(KeyError):
        await cache.get(key)


@pytest.mark.asyncio
async def test_disk_cache_put_get_exists_and_clear(tmp_path):
    # Create a temporary directory for the disk cache
    cache_dir = os.path.join(tmp_path, "disk_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = DiskCache(directory=cache_dir)
    key = "disk_test"
    value = {"alpha": 123}

    # Initially, key should not exist
    assert not await cache.exists(key)

    # Put the value
    await cache.put(key, value)

    # Check existence
    assert await cache.exists(key)

    # Retrieve the value
    result = await cache.get(key)
    assert result == value

    # Clear the cache and verify
    await cache.clear()
    assert not await cache.exists(key)
    with pytest.raises(KeyError):
        await cache.get(key)

    # Close the disk cache
    await cache.close()


@pytest.mark.asyncio
async def test_disk_cache_ttl(tmp_path):
    # Create temporary directory for disk cache with TTL
    cache_dir = os.path.join(tmp_path, "disk_cache_ttl")
    os.makedirs(cache_dir, exist_ok=True)
    # Set TTL to 1 second
    cache = DiskCache(directory=cache_dir, ttl=1)
    key = "disk_ttl"
    value = "temporary"
    await cache.put(key, value)

    # Immediately, key should exist
    assert await cache.exists(key)

    # Wait for TTL expiry
    await asyncio.sleep(1.1)

    # After TTL expiration, key should be expired (get raises KeyError)
    with pytest.raises(KeyError):
        await cache.get(key)

    await cache.clear()
    await cache.close()


def test_compute_request_hash_consistency():
    req1 = {
        "provider": "test",
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
        "_request_id": "ignore_me",
        "extra_param": "value",
    }
    req2 = {
        "provider": "test",
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
        "extra_param": "value",
        "_order_id": "should_be_removed",
    }

    hash1 = compute_request_hash(req1)
    hash2 = compute_request_hash(req2)

    # The hashes should be identical because _request_id and _order_id are removed
    assert hash1 == hash2


@pytest.mark.asyncio
async def test_disk_cache_invalid_directory(tmp_path):
    """Test DiskCache behavior with invalid directory."""
    # Try to create cache in a non-existent directory that can't be created
    # (using a file as a directory should raise OSError)
    file_path = os.path.join(tmp_path, "file")
    with open(file_path, "w") as f:
        f.write("test")
    
    with pytest.raises(OSError):
        DiskCache(directory=file_path)


@pytest.mark.asyncio
async def test_disk_cache_concurrent_access(tmp_path):
    """Test concurrent access to DiskCache."""
    cache_dir = os.path.join(tmp_path, "concurrent_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = DiskCache(directory=cache_dir)

    # Create multiple concurrent operations
    async def concurrent_operation(key, value):
        await cache.put(key, value)
        assert await cache.exists(key)
        result = await cache.get(key)
        assert result == value

    # Run multiple operations concurrently
    tasks = [
        concurrent_operation(f"key_{i}", f"value_{i}")
        for i in range(5)
    ]
    await asyncio.gather(*tasks)

    # Verify all values are still accessible
    for i in range(5):
        assert await cache.get(f"key_{i}") == f"value_{i}"

    await cache.clear()
    await cache.close()


@pytest.mark.asyncio
async def test_inmemory_cache_large_values():
    """Test InMemoryCache with large values."""
    cache = InMemoryCache()
    key = "large_value"
    # Create a large value (1MB string)
    large_value = "x" * (1024 * 1024)
    
    await cache.put(key, large_value)
    result = await cache.get(key)
    assert result == large_value

    await cache.clear()


def test_compute_request_hash_edge_cases():
    """Test compute_request_hash with edge cases."""
    # Empty request
    empty_req = {}
    empty_hash = compute_request_hash(empty_req)
    assert empty_hash  # Should return a hash even for empty dict

    # Request with only internal fields
    internal_req = {
        "_request_id": "123",
        "_order_id": "456",
    }
    internal_hash = compute_request_hash(internal_req)
    assert internal_hash == compute_request_hash({})  # Should be same as empty

    # Request with nested structures
    nested_req = {
        "provider": "test",
        "messages": [
            {"role": "system", "content": "You are a bot"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "options": {
            "temperature": 0.7,
            "max_tokens": 100,
            "stop": [".", "!"],
        },
    }
    nested_hash = compute_request_hash(nested_req)
    assert nested_hash  # Should handle nested structures

    # Verify order independence
    reordered_req = {
        "messages": [
            {"role": "system", "content": "You are a bot"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "provider": "test",
        "options": {
            "max_tokens": 100,
            "temperature": 0.7,
            "stop": [".", "!"],
        },
    }
    reordered_hash = compute_request_hash(reordered_req)
    assert nested_hash == reordered_hash  # Hash should be independent of field order
