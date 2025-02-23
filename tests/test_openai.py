import httpx
import pytest

from fastllm.providers.openai import OpenAIProvider, OpenAIRequest

# Constants for testing
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TEMPERATURE_ALT = 0.6
HTTP_OK_MIN = 200
HTTP_OK_MAX = 300


def test_openai_request_payload_from_string():
    # Test converting a simple string prompt to the expected payload format
    req = OpenAIRequest.from_prompt(
        "Test message", model="gpt-3.5-turbo", temperature=DEFAULT_TEMPERATURE
    )
    payload = req.to_request_payload()
    assert payload["model"] == "gpt-3.5-turbo"
    assert "messages" in payload
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Test message"
    assert payload["temperature"] == DEFAULT_TEMPERATURE


def test_openai_request_payload_from_dict():
    # Test converting a dictionary prompt to the expected payload format
    prompt = {"role": "system", "content": "System message"}
    req = OpenAIRequest.from_prompt(prompt, provider="openai", model="gpt-3.5-turbo")
    payload = req.to_request_payload()
    assert payload["model"] == "gpt-3.5-turbo"
    messages = payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System message"


def test_openai_request_payload_omits_none_values():
    # Test that passing None for optional parameters omits them from the final payload
    req = OpenAIRequest.from_prompt(
        "Ignore None", model="gpt-3.5-turbo", top_p=None, stop=None
    )
    payload = req.to_request_payload()
    # top_p and stop should not be in payload if they are None
    assert "top_p" not in payload
    assert "stop" not in payload


def test_openai_request_payload_with_extra_params():
    # Test that extra keyword arguments are included in the payload
    req = OpenAIRequest.from_prompt(
        "Extra params", model="gpt-3.5-turbo", custom_param="custom_value"
    )
    payload = req.to_request_payload()
    assert "custom_param" in payload
    assert payload["custom_param"] == "custom_value"


def test_openai_provider_get_request_url():
    # Test that the OpenAIProvider constructs the correct request URL
    provider = OpenAIProvider(api_key="testkey", api_base="https://api.openai.com")
    url = provider.get_request_url("completions")
    assert url == "https://api.openai.com/completions"


# Additional tests for OpenAI Provider and Request


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


def test_openai_request_from_messages():
    # Test creating an OpenAIRequest using from_messages
    messages = ["Hi there", {"role": "assistant", "content": "Hello!"}]
    req = OpenAIRequest.from_messages(
        messages, model="gpt-3.5-turbo", temperature=DEFAULT_TEMPERATURE_ALT
    )
    payload = req.to_request_payload()
    assert payload["model"] == "gpt-3.5-turbo"
    assert payload["temperature"] == DEFAULT_TEMPERATURE_ALT
    # The first message should be converted from string to a dict with role 'user'
    assert isinstance(payload["messages"], list)
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hi there"
    # The second message remains unchanged
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["content"] == "Hello!"


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
        # Return a fake response mimicking OpenAI API response
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
    # Pass a dict to trigger conversion inside make_request
    result = await provider.make_request(fake_client, request_data, timeout=1.0)
    # Check that the result has the expected fake response data
    assert result.id == "chatcmpl-xyz"
    assert result.object == "chat.completion"
    assert isinstance(result.choices, list)
    # Access the content via attributes
    assert result.choices[0].message.content == "Test reply"


def test_openai_request_to_openai_messages():
    # Test the to_openai_messages method directly
    req = OpenAIRequest.from_prompt("Hello world", model="gpt-3.5-turbo")
    messages = req.to_openai_messages()
    assert isinstance(messages, list)
    # from_prompt converts a string prompt to a message with role 'user'
    assert messages[0] == {"role": "user", "content": "Hello world"}


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
