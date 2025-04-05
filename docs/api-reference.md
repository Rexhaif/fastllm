# API Reference

This document provides detailed information about FastLLM's public APIs.

## RequestManager

### Class Definition

```python
class RequestManager:
    def __init__(
        self,
        provider: Provider[ResponseT],
        concurrency: int = 100,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        show_progress: bool = True,
        caching_provider: Optional[CacheProvider] = None
    )
```

#### Parameters

- `provider`: LLM provider instance
- `concurrency`: Maximum number of concurrent requests (default: 100)
- `timeout`: Request timeout in seconds (default: 30.0)
- `retry_attempts`: Number of retry attempts (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 1.0)
- `show_progress`: Whether to show progress bar (default: True)
- `caching_provider`: Optional cache provider instance

### Methods

#### process_batch

```python
def process_batch(
    self,
    batch: Union[list[dict[str, Any]], RequestBatch]
) -> list[ResponseT]
```

Process a batch of LLM requests in parallel.

**Parameters:**
- `batch`: Either a RequestBatch object or a list of request dictionaries

**Returns:**
- List of responses in the same order as the requests

## Provider

### Base Class

```python
class Provider(Generic[ResponseT], ABC):
    def __init__(
        self,
        api_key: str,
        api_base: str,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any
    )
```

#### Abstract Methods

```python
@abstractmethod
def get_request_headers(self) -> dict[str, str]:
    """Get headers for API requests."""
    pass

@abstractmethod
async def make_request(
    self,
    client: httpx.AsyncClient,
    request: dict[str, Any],
    timeout: float
) -> ResponseT:
    """Make a request to the provider API."""
    pass
```

### OpenAI Provider

```python
class OpenAIProvider(Provider[ChatCompletion]):
    def __init__(
        self,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        organization: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any
    )
```

## Cache System

### CacheProvider Interface

```python
class CacheProvider:
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass

    async def get(self, key: str):
        """Get a value from the cache."""
        pass

    async def put(self, key: str, value) -> None:
        """Put a value in the cache."""
        pass

    async def clear(self) -> None:
        """Clear all items from the cache."""
        pass

    async def close(self) -> None:
        """Close the cache when done."""
        pass
```

### InMemoryCache

```python
class InMemoryCache(CacheProvider):
    def __init__(self)
```

Simple in-memory cache implementation using a dictionary.

### DiskCache

```python
class DiskCache(CacheProvider):
    def __init__(
        self,
        directory: str,
        ttl: Optional[int] = None,
        **cache_options
    )
```

**Parameters:**
- `directory`: Directory where cache files will be stored
- `ttl`: Time to live in seconds for cached items
- `cache_options`: Additional options for diskcache.Cache

## Request Models

### Message

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"] = "user"
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[dict[str, Any]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
```

#### Class Methods

```python
@classmethod
def from_dict(cls, data: Union[str, dict[str, Any]]) -> Message:
    """Create a message from a string or dictionary."""
```

### LLMRequest

```python
class LLMRequest(BaseModel):
    provider: str
    messages: list[Message]
    model: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[list[str]] = None
    stream: bool = False
```

#### Class Methods

```python
@classmethod
def from_prompt(
    cls,
    provider: str,
    prompt: Union[str, dict[str, Any]],
    **kwargs
) -> LLMRequest:
    """Create a request from a single prompt."""

@classmethod
def from_dict(cls, data: dict[str, Any]) -> LLMRequest:
    """Create a request from a dictionary."""
```

## RequestBatch

```python
class RequestBatch(AbstractContextManager):
    def __init__(self)
```

### Usage Example

```python
with RequestBatch() as batch:
    batch.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

### Chat Completions API

```python
def create(
    self,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 1.0,
    n: Optional[int] = 1,
    stop: Optional[Union[str, list[str]]] = None,
    max_completion_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = 0.0,
    frequency_penalty: Optional[float] = 0.0,
    logit_bias: Optional[dict[str, float]] = None,
    user: Optional[str] = None,
    response_format: Optional[dict[str, str]] = None,
    seed: Optional[int] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, dict[str, str]]] = None,
    **kwargs: Any
) -> str:
    """Add a chat completion request to the batch."""
```

## Progress Tracking

### TokenStats

```python
@dataclass
class TokenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests_completed: int = 0
    cache_hits: int = 0
    start_time: float = 0.0
```

### ProgressTracker

```python
class ProgressTracker:
    def __init__(
        self,
        total_requests: int,
        show_progress: bool = True
    )
```

## Response Types

### ResponseWrapper

```python
class ResponseWrapper(Generic[ResponseT]):
    def __init__(
        self,
        response: ResponseT,
        request_id: str,
        order_id: int
    )