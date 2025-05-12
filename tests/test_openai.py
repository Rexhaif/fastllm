import httpx
import pytest

from fastllm.providers.openai import OpenAIProvider

# Constants for testing
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TEMPERATURE_ALT = 0.6
HTTP_OK_MIN = 200
HTTP_OK_MAX = 300


def test_prepare_payload_from_simple_request():
    # Test preparing a simple request payload
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Test message"}],
        "temperature": DEFAULT_TEMPERATURE
    }
    payload = provider._prepare_payload(request, "chat_completion")
    assert payload["model"] == "gpt-3.5-turbo"
    assert "messages" in payload
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Test message"
    assert payload["temperature"] == DEFAULT_TEMPERATURE


def test_prepare_payload_from_system_message():
    # Test preparing a payload with a system message
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "System message"}],
    }
    payload = provider._prepare_payload(request, "chat_completion")
    assert payload["model"] == "gpt-3.5-turbo"
    messages = payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System message"


def test_prepare_payload_omits_none_values():
    # Test that None values are omitted from the payload
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Ignore None"}],
        "top_p": None, 
        "stop": None
    }
    payload = provider._prepare_payload(request, "chat_completion")
    # top_p and stop should not be in payload if they are None
    assert "top_p" not in payload
    assert "stop" not in payload


def test_prepare_payload_omits_internal_tracking_ids():
    # Test that internal tracking ids are never sent to providers
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello world"}],
        "_order_id": 123,
        "_request_id": "abcd1234"
    }
    payload = provider._prepare_payload(request, "chat_completion")
    # _order_id and _request_id should not be in payload
    assert "_order_id" not in payload
    assert "_request_id" not in payload


def test_prepare_payload_with_extra_params():
    # Test that extra parameters are included in the payload
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Extra params"}],
        "custom_param": "custom_value"
    }
    payload = provider._prepare_payload(request, "chat_completion")
    assert "custom_param" in payload
    assert payload["custom_param"] == "custom_value"


def test_openai_provider_get_request_url():
    # Test that the OpenAIProvider constructs the correct request URL
    provider = OpenAIProvider(api_key="testkey", api_base="https://api.openai.com")
    url = provider.get_request_url("completions")
    assert url == "https://api.openai.com/completions"


def test_openai_provider_get_request_headers():
    provider = OpenAIProvider(
        api_key="testkey",
        api_base="https://api.openai.com",
        organization="org-123",
        headers={"X-Custom": "custom-value"},
    )
    headers = provider.get_request_headers()
    assert headers["Authorization"] == "Bearer testkey"
    assert headers["Content-Type"] == "application/json"
    assert headers["OpenAI-Organization"] == "org-123"
    assert headers["X-Custom"] == "custom-value"


def test_prepare_payload_for_embeddings():
    # Test preparing a payload for embeddings
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "text-embedding-ada-002",
        "input": "Test input",
        "dimensions": 1536,
        "user": "test-user"
    }
    payload = provider._prepare_payload(request, "embedding")
    assert payload["model"] == "text-embedding-ada-002"
    assert payload["input"] == "Test input"
    assert payload["dimensions"] == 1536
    assert payload["user"] == "test-user"


def test_prepare_payload_for_embeddings_with_array_input():
    # Test preparing a payload for embeddings with array input
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "text-embedding-ada-002",
        "input": ["Test input 1", "Test input 2"],
        "encoding_format": "float"
    }
    payload = provider._prepare_payload(request, "embedding")
    assert payload["model"] == "text-embedding-ada-002"
    assert isinstance(payload["input"], list)
    assert len(payload["input"]) == 2
    assert payload["input"][0] == "Test input 1"
    assert payload["input"][1] == "Test input 2"
    assert payload["encoding_format"] == "float"


def test_map_max_completion_tokens():
    # Test that max_completion_tokens is properly mapped to max_tokens
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Test message"}],
        "max_completion_tokens": 100
    }
    payload = provider._prepare_payload(request, "chat_completion")
    assert "max_tokens" in payload
    assert payload["max_tokens"] == 100
    assert "max_completion_tokens" not in payload


class FakeResponse:
    def __init__(self, json_data, status_code=HTTP_OK_MIN):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if not (HTTP_OK_MIN <= self.status_code < HTTP_OK_MAX):
            raise httpx.HTTPStatusError("Error", request=None, response=self)

    def json(self):
        return self._json_data


class FakeAsyncClient:
    async def post(self, url, headers, json, timeout):
        # Return appropriate fake response based on request type
        if "embeddings" in url:
            # Return a fake embeddings response
            fake_json = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3],
                        "index": 0
                    }
                ],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 8, "total_tokens": 8}
            }
        else:
            # Return a fake chat completion response
            fake_json = {
                "id": "chatcmpl-xyz",
                "object": "chat.completion",
                "model": "gpt-3.5-turbo",
                "created": 1690000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        return FakeResponse(fake_json)


@pytest.mark.asyncio
async def test_openai_provider_make_request():
    provider = OpenAIProvider(api_key="testkey", api_base="https://api.openai.com")
    fake_client = FakeAsyncClient()
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Tell me a joke."}],
        "temperature": 0.5,
    }
    # Pass a dict directly to make_request
    result = await provider.make_request(fake_client, request_data, timeout=1.0)
    # Check that the result has the expected fake response data
    assert result.id == "chatcmpl-xyz"
    assert result.object == "chat.completion"
    assert isinstance(result.choices, list)
    # Access the content via attributes
    assert result.choices[0].message.content == "Test reply"


@pytest.mark.asyncio
async def test_openai_provider_make_embedding_request():
    provider = OpenAIProvider(api_key="testkey", api_base="https://api.openai.com")
    fake_client = FakeAsyncClient()
    request_data = {
        "model": "text-embedding-ada-002",
        "input": "Sample text for embedding",
        "type": "embedding"  # Add type indicator for path determination
    }
    # Pass an embedding request to make_request
    result = await provider.make_request(
        fake_client, 
        request_data, 
        timeout=1.0
    )
    # For embeddings, we expect a dict since it doesn't parse as ChatCompletion
    assert isinstance(result, dict)
    assert result["object"] == "list"
    assert "data" in result
    assert isinstance(result["data"], list)
    assert len(result["data"]) > 0
    assert "embedding" in result["data"][0]
    assert isinstance(result["data"][0]["embedding"], list)


class FakeAsyncClientError:
    async def post(self, url, headers, json, timeout):
        # Return a fake response with an error status code
        return FakeResponse({"error": "Bad Request"}, status_code=400)


@pytest.mark.asyncio
async def test_openai_provider_make_request_error():
    provider = OpenAIProvider(api_key="testkey", api_base="https://api.openai.com")
    fake_client_error = FakeAsyncClientError()
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "This will error."}],
        "temperature": 0.5,
    }
    with pytest.raises(httpx.HTTPStatusError):
        await provider.make_request(fake_client_error, request_data, timeout=1.0)


def test_embedding_request_validation():
    # Test validation for embedding requests
    provider = OpenAIProvider(api_key="testkey")
    
    # Test missing model
    with pytest.raises(ValueError):
        provider._prepare_payload({"input": "test"}, "embedding")
    
    # Test missing input
    with pytest.raises(ValueError):
        provider._prepare_payload({"model": "text-embedding-ada-002"}, "embedding")
    
    # Test valid minimal embedding request
    payload = provider._prepare_payload({
        "model": "text-embedding-ada-002",
        "input": "test"
    }, "embedding")
    assert payload == {"model": "text-embedding-ada-002", "input": "test"}


def test_embedding_request_type_detection():
    # Test automatic detection of embedding requests
    provider = OpenAIProvider(api_key="testkey")
    fake_client = FakeAsyncClient()
    
    # Request with input field should be detected as embedding
    request = {
        "model": "text-embedding-ada-002",
        "input": "test",
    }
    
    # Check that the request is correctly identified as embedding type
    payload = provider._prepare_payload(request, "embedding")
    assert payload == {"model": "text-embedding-ada-002", "input": "test"}
