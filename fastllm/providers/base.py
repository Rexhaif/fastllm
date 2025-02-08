"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import httpx

from fastllm.core import ResponseT

class Provider(Generic[ResponseT], ABC):
    """Base class for LLM providers."""

    # Internal tracking headers
    _REFERER = "https://github.com/Rexhaif/fastllm"
    _APP_NAME = "FastLLM"

    def __init__(
        self,
        api_key: str,
        api_base: str,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")  # Remove trailing slash if present
        self._default_headers = {
            "HTTP-Referer": self._REFERER,
            "X-Title": self._APP_NAME,
        }
        self.headers = {
            **self._default_headers,
            **(headers or {}),
        }

    def get_request_url(self, endpoint: str) -> str:
        """Get full URL for API endpoint."""
        return f"{self.api_base}/{endpoint.lstrip('/')}"

    @abstractmethod
    def get_request_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        pass

    @abstractmethod
    async def make_request(
        self,
        client: httpx.AsyncClient,
        request: dict[str, Any],
        timeout: float,
    ) -> ResponseT:
        """Make a request to the provider API."""
        pass 