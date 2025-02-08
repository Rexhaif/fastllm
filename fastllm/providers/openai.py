"""OpenAI API provider implementation."""

from typing import Any, Optional, Union

import httpx
from openai.types.chat import ChatCompletion
from pydantic import Field, model_validator

from fastllm.core import (
    LLMRequest,
    Message,
)
from fastllm.providers.base import Provider

DEFAULT_API_BASE = "https://api.openai.com/v1"


class OpenAIProvider(Provider[ChatCompletion]):
    """OpenAI provider."""

    def __init__(
        self,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        organization: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ):
        super().__init__(api_key, api_base, headers, **kwargs)
        self.organization = organization

    def get_request_headers(self) -> dict[str, str]:
        """Get headers for OpenAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.headers,
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    async def make_request(
        self,
        client: httpx.AsyncClient,
        request: dict[str, Any],
        timeout: float,
    ) -> ChatCompletion:
        """Make a request to the OpenAI API."""
        if isinstance(request, dict):
            request = OpenAIRequest.from_dict(request)

        url = self.get_request_url("chat/completions")
        payload = request.to_request_payload()

        response = await client.post(
            url,
            headers=self.get_request_headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        return ChatCompletion(**data)


class OpenAIRequest(LLMRequest):
    """OpenAI-compatible request model."""

    provider: str = "openai"
    model: str = Field(default="gpt-4o-mini")
    messages: list[Message]

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI message types to our Message type."""
        if "messages" in data:
            if isinstance(data["messages"], list):
                data["messages"] = [
                    msg
                    if isinstance(msg, Message)
                    else Message.from_dict(
                        msg
                        if isinstance(msg, (str, dict))
                        else {"role": msg.role, "content": msg.content}
                    )
                    for msg in data["messages"]
                ]
        return data

    def to_openai_messages(self) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
                **({"function_call": msg.function_call} if msg.function_call else {}),
                **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
            }
            for msg in self.messages
        ]

    def to_request_payload(self) -> dict[str, Any]:
        """Convert request to OpenAI API payload."""
        return {
            "model": self.model,
            "messages": self.to_openai_messages(),
            "temperature": self.temperature,
            **({"max_tokens": self.max_tokens} if self.max_tokens else {}),
            **({"top_p": self.top_p} if self.top_p else {}),
            **(
                {"presence_penalty": self.presence_penalty}
                if self.presence_penalty
                else {}
            ),
            **(
                {"frequency_penalty": self.frequency_penalty}
                if self.frequency_penalty
                else {}
            ),
            **({"stop": self.stop} if self.stop else {}),
            "stream": self.stream,
        }

    @classmethod
    def from_prompt(
        cls, prompt: Union[str, dict[str, Any]], **kwargs
    ) -> "OpenAIRequest":
        """Create a request from a single prompt."""
        if isinstance(prompt, str):
            messages = [Message(role="user", content=prompt)]
        else:
            messages = [Message.from_dict(prompt)]
        return cls(messages=messages, **kwargs)

    @classmethod
    def from_messages(
        cls, messages: list[Union[dict[str, Any], str]], **kwargs
    ) -> "OpenAIRequest":
        """Create a request from a list of messages."""
        converted_messages = []
        for msg in messages:
            if isinstance(msg, (str, dict)):
                converted_messages.append(Message.from_dict(msg))
            else:
                # It's a message-like object
                msg_dict = {
                    "role": getattr(msg, "role", "user"),
                    "content": getattr(msg, "content", str(msg)) or "",
                }
                if hasattr(msg, "name") and msg.name:
                    msg_dict["name"] = msg.name
                if hasattr(msg, "function_call") and msg.function_call:
                    msg_dict["function_call"] = msg.function_call
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                converted_messages.append(Message(**msg_dict))

        return cls(messages=converted_messages, **kwargs)
