"""OpenAI API provider implementation."""

from typing import Any, Optional, cast, Type

import httpx
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse

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
    ) -> ChatCompletion | CreateEmbeddingResponse:
        """Make a request to the OpenAI API."""
        # Determine request type from the request or infer from content
        if isinstance(request, dict):
            # Extract request type from the request data
            request_type = request.get("type")
            if request_type is None:
                # Infer type based on content
                if "messages" in request:
                    request_type = "chat_completion"
                elif "input" in request:
                    request_type = "embedding"
                else:
                    request_type = "chat_completion"  # Default
        else:
            # Handle unexpected input
            raise ValueError(f"Unexpected request type: {type(request)}")
        
        # Determine API path based on request type
        if request_type == "embedding":
            api_path = "embeddings"
        else:
            api_path = "chat/completions"

        url = self.get_request_url(api_path)
        payload = self._prepare_payload(request, request_type)

        response = await client.post(
            url,
            headers=self.get_request_headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        if request_type == "embedding":
            return CreateEmbeddingResponse(**data)
        else:
            return ChatCompletion(**data)

        
    
    def _prepare_payload(self, request: dict[str, Any], request_type: str) -> dict[str, Any]:
        """Prepare the API payload from the request data."""
        # Extract known fields and extra params
        known_fields = {
            "provider", "model", "messages", "temperature", "max_completion_tokens",
            "top_p", "presence_penalty", "frequency_penalty", "stop", "stream",
            "type", "input", "dimensions", "encoding_format", "user"
        }
        
        # Start with a copy of the request
        payload = {k: v for k, v in request.items() if k not in ["provider", "type", "_order_id", "_request_id"]}
        
        # Handle embedding requests
        if request_type == "embedding":
            # Ensure required fields are present
            if "model" not in payload:
                raise ValueError("Model is required for embedding requests")
            if "input" not in payload:
                raise ValueError("Input is required for embedding requests")
            
            # Keep only relevant fields for embeddings
            embedding_fields = {"model", "input", "dimensions", "encoding_format", "user"}
            return {k: v for k, v in payload.items() if k in embedding_fields}
        
        # Handle chat completion requests
        if "model" not in payload:
            raise ValueError("Model is required for chat completion requests")
        if "messages" not in payload:
            raise ValueError("Messages are required for chat completion requests")
        
        # Map max_completion_tokens to max_tokens if present
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        
        # Remove any None values
        return {k: v for k, v in payload.items() if v is not None}
