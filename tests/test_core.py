"""Tests for core functionality."""

import time
from unittest import mock

import pytest

from fastllm.core import (
    LLMRequest,
    LLMResponse,
    Message,
    ProgressTracker,
    RequestBatch,
    RequestManager,
    ResponseWrapper,
    TokenStats,
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

# Message Tests
def test_message_from_string():
    """Test Message creation from string."""
    message = Message.from_dict("Hello, world!")
    assert message.role == "user"
    assert message.content == "Hello, world!"


def test_message_from_dict():
    """Test Message creation from dictionary."""
    message = Message.from_dict(
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    )
    assert message.role == "system"
    assert message.content == "You are a helpful assistant."

# LLMRequest Tests
def test_llm_request_from_dict():
    """Test LLMRequest creation from dictionary."""
    request = LLMRequest.from_dict(
        {
            "provider": "test",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                "Hello, world!",  # Should be converted to user message
            ],
            "model": "test-model",
            "temperature": 0.5,
        }
    )
    assert request.provider == "test"
    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert request.messages[1].content == "Hello, world!"
    assert request.model == "test-model"
    assert request.temperature == 0.5

    # Test fallback using 'prompt' key when messages is not provided
    data = {"provider": "dummy", "prompt": "Fallback prompt"}
    req = LLMRequest.from_dict(data)
    assert req.messages[0].content == "Fallback prompt"


def test_llm_request_from_prompt():
    """Test LLMRequest.from_prompt with different input types."""
    # String input
    request = LLMRequest.from_prompt(
        provider="test",
        prompt="Hello, world!",
        model="test-model",
    )
    assert request.provider == "test"
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.messages[0].content == "Hello, world!"

    # Dictionary input
    request = LLMRequest.from_prompt(
        provider="test",
        prompt={
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        model="test-model",
    )
    assert request.provider == "test"
    assert len(request.messages) == 1
    assert request.messages[0].role == "system"
    assert request.messages[0].content == "You are a helpful assistant."

# LLMResponse Tests
def test_llm_response_from_dict():
    """Test LLMResponse creation from dictionary."""
    response = LLMResponse.from_dict(
        {
            "request_id": "test-id",
            "provider": "test",
            "content": "Response content",
            "raw_response": {"key": "value"},
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
        }
    )
    assert response.request_id == "test-id"
    assert response.provider == "test"
    assert response.content == "Response content"
    assert response.raw_response == {"key": "value"}
    assert response.finish_reason == "stop"
    assert response.usage["total_tokens"] == 15

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
    """Test RequestManager base functionality."""
    manager = RequestManager(provider=DummyProvider())
    request = LLMRequest.from_dict(
        {
            "provider": "test",
            "messages": ["test message"],
            "model": "test-model",
        }
    )

    response = await manager._make_provider_request(None, request)
    assert response.content == "Test response"
    assert response.finish_reason == "stop"
    assert response.usage["total_tokens"] == 15


class FailingProvider:
    async def make_request(self, client, request, timeout):
        raise Exception("Provider failure")

@pytest.mark.asyncio
async def test_request_manager_failure():
    """Test that RequestManager properly handles failures."""
    rm = RequestManager(
        provider=FailingProvider(),
        concurrency=1,
        timeout=0.1,
        retry_attempts=2,
        retry_delay=0,
    )

    request = LLMRequest.from_dict({
        "provider": "test",
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "Test message"}],
    })

    with pytest.raises(Exception) as exc_info:
        await rm._make_provider_request(None, request)
    assert "Provider failure" in str(exc_info.value)

# Progress Tracker Tests
def test_progress_tracker_update():
    """Test ProgressTracker updates."""
    tracker = ProgressTracker(total_requests=5)
    tracker.update(10, 5, False)
    assert tracker.stats.prompt_tokens == 10
    assert tracker.stats.completion_tokens == 5
    assert tracker.stats.total_tokens == 15
    assert tracker.stats.requests_completed == 1
    assert tracker.stats.cache_hits == 0


def test_progress_tracker_context_manager():
    """Test ProgressTracker as context manager."""
    with ProgressTracker(total_requests=1) as tracker:
        tracker.update(10, 5, False)
    assert tracker.stats.prompt_tokens == 10
    assert tracker.stats.completion_tokens == 5


def test_progress_tracker_with_limits():
    """Test ProgressTracker with rate limits."""
    tracker = ProgressTracker(total_requests=10)
    tracker.stats.token_limit = 1000
    tracker.stats.request_limit = 100
    tracker.update(100, 50, False)  # 150 tokens total
    assert tracker.stats.token_saturation > 0
    assert tracker.stats.request_saturation > 0

# Add new tests for caching behavior
class CachingProvider:
    """Provider that returns predictable responses for testing caching."""
    def __init__(self):
        self.call_count = 0
    
    async def make_request(self, client, request, timeout):
        self.call_count += 1
        return {
            "content": f"Response {self.call_count}",
            "status": 200,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

@pytest.mark.asyncio
async def test_request_manager_caching():
    """Test that RequestManager properly handles caching."""
    # Create provider and manager with cache
    provider = CachingProvider()
    cache = InMemoryCache()
    rm = RequestManager(
        provider=provider,
        concurrency=1,
        timeout=0.1,
        retry_attempts=1,
        caching_provider=cache,
    )

    # Create identical requests
    request1 = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "Test message"}],
    }
    request2 = request1.copy()
    
    # Compute expected request_id
    request_id = compute_request_hash(request1)
    
    # First request should hit the provider
    response1 = await rm._process_request_async(None, request1)
    assert provider.call_count == 1
    assert response1.response["content"] == "Response 1"
    assert response1.request_id == request_id
    
    # Second identical request should hit the cache
    response2 = await rm._process_request_async(None, request2)
    assert provider.call_count == 1  # No additional provider calls
    assert response2.response["content"] == "Response 1"  # Same response
    assert response2.request_id == request_id  # Same request_id

    # Different request should hit the provider
    request3 = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "Different message"}],
    }
    response3 = await rm._process_request_async(None, request3)
    assert provider.call_count == 2  # New provider call
    assert response3.response["content"] == "Response 2"
    assert response3.request_id != request_id  # Different request_id


@pytest.mark.asyncio
async def test_request_manager_cache_errors():
    """Test that RequestManager handles cache errors gracefully."""
    provider = CachingProvider()
    
    # Create a cache that raises errors
    class ErrorCache(InMemoryCache):
        async def exists(self, key: str) -> bool:
            raise Exception("Cache error")
        
        async def get(self, key: str):
            raise Exception("Cache error")
        
        async def put(self, key: str, value) -> None:
            raise Exception("Cache error")
    
    rm = RequestManager(
        provider=provider,
        concurrency=1,
        timeout=0.1,
        retry_attempts=1,
        caching_provider=ErrorCache(),
    )
    
    # Request should succeed even with cache errors
    request = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "Test message"}],
    }
    response = await rm._process_request_async(None, request)
    assert provider.call_count == 1
    assert response.response["content"] == "Response 1"


@pytest.mark.asyncio
async def test_request_manager_failed_response_not_cached():
    """Test that failed responses are not cached."""
    class FailingProvider:
        def __init__(self, fail_first=True):
            self.fail_first = fail_first
            self.call_count = 0
            self.has_failed = False
        
        async def make_request(self, client, request, timeout):
            self.call_count += 1
            if self.fail_first and not self.has_failed:
                self.has_failed = True
                return {
                    "content": "Error",
                    "status": 500,
                }
            return {
                "content": f"Success {self.call_count}",
                "status": 200,
            }
    
    provider = FailingProvider()
    cache = InMemoryCache()
    rm = RequestManager(
        provider=provider,
        concurrency=1,
        timeout=0.1,
        retry_attempts=1,
        caching_provider=cache,
    )
    
    # Create base request
    base_request = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "Test message"}],
    }
    
    # First request - should fail and not be cached
    request1 = base_request.copy()
    response1 = await rm._process_request_async(None, request1)
    assert response1.response["status"] == 500
    assert provider.call_count == 1
    
    # Verify the failed response wasn't cached
    request_id = response1.request_id
    assert not await cache.exists(request_id)
    
    # Second request - should succeed and be cached
    request2 = base_request.copy()
    response2 = await rm._process_request_async(None, request2)
    assert response2.response["status"] == 200
    assert provider.call_count == 2
    
    # Verify the successful response was cached
    assert await cache.exists(request2["_request_id"])
    
    # Third request - should hit cache
    request3 = base_request.copy()
    response3 = await rm._process_request_async(None, request3)
    assert response3.response["status"] == 200
    assert provider.call_count == 2  # No additional provider calls
