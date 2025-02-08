"""Tests for core functionality."""

import pytest

from fastllm.core import LLMRequest, LLMResponse, Message, RequestManager


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


def test_llm_response_from_dict():
    """Test LLMResponse creation from dictionary."""
    response = LLMResponse.from_dict(
        {
            "request_id": "test-id",
            "provider": "test",
            "content": "Response content",
            "raw_response": {"key": "value"},
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }
    )
    assert response.request_id == "test-id"
    assert response.provider == "test"
    assert response.content == "Response content"
    assert response.raw_response == {"key": "value"}
    assert response.finish_reason == "stop"
    assert response.usage == {"total_tokens": 10}


@pytest.mark.asyncio
async def test_request_manager():
    """Test RequestManager base class."""
    manager = RequestManager()
    request = LLMRequest.from_dict(
        {
            "provider": "test",
            "messages": ["test message"],
        }
    )

    with pytest.raises(NotImplementedError):
        await manager._make_provider_request(None, request)
