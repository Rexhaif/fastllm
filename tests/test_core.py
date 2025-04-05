"""Tests for core functionality."""

import time
from unittest import mock

import time
from unittest import mock

import pytest

from fastllm.core import (
    RequestBatch,
    RequestManager,
    ResponseWrapper,
    TokenStats,
    ProgressTracker,
)
from fastllm.cache import InMemoryCache, compute_request_hash

# Constants for testing
DEFAULT_CHUNK_SIZE = 20
DEFAULT_MAX_CHUNK_SIZE = 1000
DEFAULT_PROMPT_TOKENS = 10
DEFAULT_COMPLETION_TOKENS = 5
DEFAULT_TOTAL_TOKENS = DEFAULT_PROMPT_TOKENS + DEFAULT_COMPLETION_TOKENS
DEFAULT_RETRY_ATTEMPTS = 2
DEFAULT_CONCURRENCY = 5
DEFAULT_TIMEOUT = 1.0
DEFAULT_RETRY_DELAY = 0.0

# TokenStats Tests
def test_token_stats():
    """Test basic TokenStats functionality."""
    ts = TokenStats(start_time=time.time() - 2)  # started 2 seconds ago
    assert ts.cache_hit_ratio == 0.0
    ts.update(DEFAULT_PROMPT_TOKENS, DEFAULT_COMPLETION_TOKENS, is_cache_hit=True)
    assert ts.prompt_tokens == DEFAULT_PROMPT_TOKENS
    assert ts.completion_tokens == DEFAULT_COMPLETION_TOKENS
    assert ts.total_tokens == DEFAULT_TOTAL_TOKENS
    assert ts.requests_completed == 1
    assert ts.cache_hits == 1
    assert ts.prompt_tokens_per_second > 0
    assert ts.completion_tokens_per_second > 0


def test_token_stats_rate_limits():
    """Test rate limit tracking in TokenStats."""
    current_time = time.time()
    stats = TokenStats(
        start_time=current_time,
        token_limit=1000,  # 1000 tokens per minute
        request_limit=100,  # 100 requests per minute
    )

    # Test initial state
    assert stats.token_saturation == 0.0
    assert stats.request_saturation == 0.0

    # Update with some usage (non-cache hits)
    stats.update(50, 50, is_cache_hit=False)  # 100 tokens total
    stats.update(25, 25, is_cache_hit=False)  # 50 tokens total

    # Cache hits should not affect rate limit tracking
    stats.update(100, 100, is_cache_hit=True)
    assert stats.window_tokens == 150  # Still 150 from non-cache hits
    assert stats.window_requests == 2  # Still 2 from non-cache hits

# RequestManager Tests
class DummyProvider:
    async def make_request(self, client, request, timeout):
        return {
            "content": "Test response",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }


@pytest.mark.asyncio
async def test_request_manager():
    """Test basic RequestManager functionality."""
    provider = DummyProvider()
    manager = RequestManager(provider=provider)
    
    # Create a request dictionary
    request = {
        "provider": "dummy",
        "messages": [{"role": "user", "content": "Test message"}],
        "model": "dummy-model"
    }
    
    # Test make_provider_request
    response = await manager._make_provider_request(None, request)
    assert response["content"] == "Test response"
    assert response["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 15


class FailingProvider:
    async def make_request(self, client, request, timeout):
        raise Exception("Provider error")


@pytest.mark.asyncio
async def test_request_manager_failure():
    """Test RequestManager with failing provider."""
    provider = FailingProvider()
    manager = RequestManager(provider=provider)
    request = {
        "provider": "dummy",
        "messages": [{"role": "user", "content": "Test message"}],
        "model": "dummy-model"
    }
    
    with pytest.raises(Exception) as exc_info:
        await manager._make_provider_request(None, request)
    assert "Provider error" in str(exc_info.value)


def test_progress_tracker_update():
    """Test ProgressTracker updates."""
    tracker = ProgressTracker(total_requests=10, show_progress=False)
    tracker.update(DEFAULT_PROMPT_TOKENS, DEFAULT_COMPLETION_TOKENS)
    assert tracker.stats.prompt_tokens == DEFAULT_PROMPT_TOKENS
    assert tracker.stats.completion_tokens == DEFAULT_COMPLETION_TOKENS
    assert tracker.stats.total_tokens == DEFAULT_TOTAL_TOKENS


def test_progress_tracker_context_manager():
    """Test ProgressTracker as context manager."""
    with ProgressTracker(total_requests=10, show_progress=False) as tracker:
        tracker.update(DEFAULT_PROMPT_TOKENS, DEFAULT_COMPLETION_TOKENS)
    assert tracker.stats.prompt_tokens == DEFAULT_PROMPT_TOKENS


def test_progress_tracker_with_limits():
    """Test ProgressTracker with rate limits."""
    tracker = ProgressTracker(total_requests=10, show_progress=False)
    tracker.stats.token_limit = 1000
    tracker.stats.request_limit = 100
    tracker.update(DEFAULT_PROMPT_TOKENS, DEFAULT_COMPLETION_TOKENS)
    assert tracker.stats.token_saturation > 0
    assert tracker.stats.request_saturation > 0


class CachingProvider:
    """Test provider that always returns the same response."""
    
    def __init__(self):
        self.call_count = 0
    
    async def make_request(self, client, request, timeout):
        self.call_count += 1
        return {
            "content": "Cached response",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }


@pytest.mark.asyncio
async def test_request_manager_caching():
    """Test RequestManager with caching."""
    provider = CachingProvider()
    cache = InMemoryCache()
    manager = RequestManager(
        provider=provider,
        caching_provider=cache,
        show_progress=False,
    )
    
    # Create identical requests
    request1 = {
        "provider": "dummy",
        "messages": [{"role": "user", "content": "Test message"}],
        "model": "dummy-model"
    }
    
    # Add request_id
    request_id = compute_request_hash(request1)
    request1["_request_id"] = request_id
    
    # First request should hit the provider
    response1 = await manager._process_request_async(None, request1, None)
    assert provider.call_count == 1
    
    # Second request with the same content should hit the cache
    request2 = request1.copy()
    response2 = await manager._process_request_async(None, request2, None)
    assert provider.call_count == 1  # Should not increase
    
    # Different request should hit the provider
    request3 = {
        "provider": "dummy",
        "messages": [{"role": "user", "content": "Different message"}],
        "model": "dummy-model"
    }
    request3["_request_id"] = compute_request_hash(request3)
    
    response3 = await manager._process_request_async(None, request3, None)
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_request_manager_cache_errors():
    """Test RequestManager handles cache errors gracefully."""
    
    class ErrorCache(InMemoryCache):
        async def exists(self, key: str) -> bool:
            raise Exception("Cache error")
        
        async def get(self, key: str):
            raise Exception("Cache error")
        
        async def put(self, key: str, value) -> None:
            raise Exception("Cache error")
    
    provider = DummyProvider()
    manager = RequestManager(
        provider=provider,
        caching_provider=ErrorCache(),
        show_progress=False,
    )
    
    # Create a request
    request = {
        "provider": "dummy",
        "messages": [{"role": "user", "content": "Test message"}],
        "model": "dummy-model",
        "_request_id": "test-id",
    }
    
    # Request should succeed despite cache errors
    response = await manager._process_request_async(None, request, None)
    # Access response content from the wrapped response
    assert response.response["content"] == "Test response"


@pytest.mark.asyncio
async def test_request_manager_failed_response_not_cached():
    """Test that failed responses are not cached."""
    
    class FailingProvider:
        def __init__(self, fail_first=True):
            self.call_count = 0
            self.fail_first = fail_first
            self.has_failed = False
        
        async def make_request(self, client, request, timeout):
            self.call_count += 1
            
            if self.fail_first and not self.has_failed:
                self.has_failed = True
                raise Exception("Provider failure")
            
            return {
                "content": "Success response",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            }
    
    provider = FailingProvider()
    cache = InMemoryCache()
    manager = RequestManager(
        provider=provider,
        caching_provider=cache,
        retry_attempts=1,  # Only retry once
        show_progress=False,
    )
    
    # Create a request
    request = {
        "provider": "dummy",
        "messages": [{"role": "user", "content": "Test message"}],
        "model": "dummy-model",
        "_request_id": "test-id",
    }
    
    # First attempt should fail
    with pytest.raises(Exception):
        await manager._process_request_async(None, request, None)
    
    # Verify the call was made
    assert provider.call_count == 1
    
    # Second attempt with same request should hit the provider again
    # since the failed response should not be cached
    try:
        await manager._process_request_async(None, request, None)
    except:
        pass
    
    assert provider.call_count == 2
