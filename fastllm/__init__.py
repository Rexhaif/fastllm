"""FastLLM - High-performance parallel LLM API request tool."""

__version__ = "0.1.0"

from fastllm.core import (
    RequestBatch,
    RequestManager,
    ResponseWrapper,
    TokenStats,
)
from fastllm.cache import (
    CacheProvider,
    InMemoryCache,
    DiskCache,
    compute_request_hash,
)
from fastllm.providers.base import Provider
from fastllm.providers.openai import OpenAIProvider

__all__ = [
    # Core components
    "RequestBatch",
    "RequestManager",
    "ResponseWrapper",
    "TokenStats",
    
    # Cache components
    "CacheProvider",
    "InMemoryCache",
    "DiskCache",
    "compute_request_hash",
    
    # Provider components
    "Provider",
    "OpenAIProvider",
]
