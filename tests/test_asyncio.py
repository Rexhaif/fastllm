"""Tests for async functionality."""

import asyncio
import time

import pytest
from openai.types.completion_usage import CompletionUsage

from fastllm.core import RequestManager

# Constants for testing
MAX_DURATION = 0.5  # Maximum expected duration for concurrent execution
EXPECTED_RESPONSES = 5  # Expected number of responses
EXPECTED_SUCCESSES = 2  # Expected number of successful responses
EXPECTED_FAILURES = 1  # Expected number of failed responses


class DummyRequestManager(RequestManager):
    async def _make_provider_request(self, client, request):
        await asyncio.sleep(0.1)  # Simulate network delay
        # Extract message content from request
        message_content = request.get("messages", [{}])[0].get("content", "No content")
        response_dict = {
            "request_id": id(request),  # Add unique request ID
            "content": f"Response to: {message_content}",
            "finish_reason": "dummy_end",
            "provider": "dummy",
            "raw_response": {"dummy_key": "dummy_value"},
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        return response_dict


@pytest.mark.asyncio
async def test_dummy_manager_single_request():
    manager = DummyRequestManager(provider="dummy")
    request = {
        "provider": "dummy", 
        "messages": [{"role": "user", "content": "Hello async!"}], 
        "model": "dummy-model"
    }
    response = await manager._make_provider_request(None, request)
    assert response["content"] == "Response to: Hello async!"
    assert response["finish_reason"] == "dummy_end"
    assert response["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_dummy_manager_concurrent_requests():
    manager = DummyRequestManager(provider="dummy")
    requests = [
        {
            "provider": "dummy", 
            "messages": [{"role": "user", "content": f"Message {i}"}], 
            "model": "dummy-model"
        }
        for i in range(5)
    ]

    start = time.perf_counter()
    responses = await asyncio.gather(
        *[manager._make_provider_request(None, req) for req in requests]
    )
    end = time.perf_counter()

    # All requests should complete successfully
    assert len(responses) == 5
    for i, response in enumerate(responses):
        assert response["content"] == f"Response to: Message {i}"
        assert response["finish_reason"] == "dummy_end"
        assert response["usage"]["total_tokens"] == 15

    # Requests should be processed concurrently
    # Total time should be less than sequential time (5 * 0.1s)
    assert end - start < 0.5  # Allow some overhead


class FailingDummyManager(RequestManager):
    async def _make_provider_request(self, client, request):
        message_content = request.get("messages", [{}])[0].get("content", "")
        if "fail" in message_content.lower():
            raise Exception("Provider failure")
        await asyncio.sleep(0.1)
        response_dict = {
            "request_id": id(request),  # Add unique request ID
            "content": f"Response to: {message_content}",
            "finish_reason": "dummy_end",
            "provider": "dummy",
            "raw_response": {"dummy_key": "dummy_value"},
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        return response_dict


@pytest.mark.asyncio
async def test_dummy_manager_request_failure():
    manager = FailingDummyManager(provider="dummy")
    request = {
        "provider": "dummy", 
        "messages": [{"role": "user", "content": "fail this request"}], 
        "model": "dummy-model"
    }
    with pytest.raises(Exception) as exc_info:
        await manager._make_provider_request(None, request)
    assert "Provider failure" in str(exc_info.value)


@pytest.mark.asyncio
async def test_gather_with_mixed_success_and_failure():
    manager = FailingDummyManager(provider="dummy")
    requests = [
        {
            "provider": "dummy", 
            "messages": [{"role": "user", "content": "Message 1"}], 
            "model": "dummy-model"
        },
        {
            "provider": "dummy", 
            "messages": [{"role": "user", "content": "fail this one"}], 
            "model": "dummy-model"
        },
        {
            "provider": "dummy", 
            "messages": [{"role": "user", "content": "Message 3"}], 
            "model": "dummy-model"
        },
    ]

    responses = await asyncio.gather(
        *[manager._make_provider_request(None, req) for req in requests],
        return_exceptions=True,
    )
    successes = [resp for resp in responses if not isinstance(resp, Exception)]
    failures = [resp for resp in responses if isinstance(resp, Exception)]

    assert len(successes) == 2
    assert len(failures) == 1
    assert "Provider failure" in str(failures[0])


@pytest.mark.asyncio
async def test_task_scheduling_order():
    manager = DummyRequestManager(provider="dummy")
    requests = [
        {
            "provider": "dummy", 
            "messages": [{"role": "user", "content": f"Message {i}"}], 
            "model": "dummy-model"
        }
        for i in range(3)
    ]

    # Create tasks but don't await them yet
    tasks = [manager._make_provider_request(None, req) for req in requests]
    
    # Schedule tasks in reverse order
    responses = []
    for task in reversed(tasks):
        responses.append(await task)

    # Despite scheduling in reverse order, responses should match request order
    for i, response in enumerate(responses):
        assert f"Message {2-i}" in response["content"]


@pytest.mark.asyncio
async def test_task_cancellation():
    manager = DummyRequestManager(provider="dummy")
    request = {
        "provider": "dummy", 
        "messages": [{"role": "user", "content": "Cancel me"}], 
        "model": "dummy-model"
    }

    # Start the task
    task = asyncio.create_task(manager._make_provider_request(None, request))
    
    # Cancel it immediately
    task.cancel()
    
    with pytest.raises(asyncio.CancelledError):
        await task
