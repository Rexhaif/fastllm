from fastllm.providers.base import Provider
from fastllm.providers.openai import OpenAIProvider
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage


class DummyProvider(Provider):
    def __init__(
        self,
        api_key="dummy",
        api_base="https://api.example.com",
        organization=None,
        headers=None,
    ):
        super().__init__(api_key, api_base, headers)
        self.organization = organization

    def get_request_headers(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    async def _make_actual_request(self, client, request, timeout):
        # Simulate a dummy response
        return ChatCompletion(
            id="chatcmpl-dummy",
            object="chat.completion",
            created=1234567890,
            model=request.get("model", "dummy-model"),
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="This is a dummy response",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
            ),
        )

    async def make_request(self, client, request, timeout):
        return await self._make_actual_request(client, request, timeout)


def test_dummy_provider_get_request_url():
    provider = DummyProvider(api_key="testkey", api_base="https://api.test.com")
    url = provider.get_request_url("endpoint")
    assert url == "https://api.test.com/endpoint"


def test_dummy_provider_get_request_headers():
    provider = DummyProvider(api_key="testkey")
    headers = provider.get_request_headers()
    assert headers["Authorization"] == "Bearer testkey"
    assert headers["Content-Type"] == "application/json"


def test_openai_provider_get_request_headers_org():
    provider = OpenAIProvider(api_key="testkey", organization="org123")
    headers = provider.get_request_headers()
    assert headers["Authorization"] == "Bearer testkey"
    assert headers["Content-Type"] == "application/json"
    assert headers.get("OpenAI-Organization") == "org123"


def test_openai_provider_prepare_chat_completion_payload():
    # Test conversion from a simple request to payload
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "gpt-dummy",
        "messages": [{"role": "user", "content": "Hello world!"}]
    }
    payload = provider._prepare_payload(request, "chat_completion")
    
    # Verify that the model and messages are set correctly
    assert payload["model"] == "gpt-dummy"
    assert "messages" in payload
    assert isinstance(payload["messages"], list)
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hello world!"


def test_openai_provider_prepare_embedding_payload():
    # Test conversion from embedding request to payload
    provider = OpenAIProvider(api_key="testkey")
    request = {
        "model": "text-embedding-3-small",
        "input": ["Sample text 1", "Sample text 2"]
    }
    payload = provider._prepare_payload(request, "embedding")
    
    # Verify model and input are set correctly
    assert payload["model"] == "text-embedding-3-small"
    assert payload["input"] == ["Sample text 1", "Sample text 2"]
