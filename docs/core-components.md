# Core Components

This document provides detailed information about the core components of FastLLM.

## RequestManager

The RequestManager is the central component responsible for handling parallel LLM API requests.

```python
RequestManager(
    provider: Provider[ResponseT],
    concurrency: int = 100,
    timeout: float = 30.0,
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
    show_progress: bool = True,
    caching_provider: Optional[CacheProvider] = None
)
```

### Features

- **Parallel Processing**: Handles multiple requests concurrently using asyncio
- **Chunked Processing**: Processes requests in optimal chunks based on concurrency
- **Progress Tracking**: Real-time progress monitoring with rich statistics
- **Cache Integration**: Seamless integration with caching providers
- **Error Handling**: Comprehensive error handling with retries

### Key Methods

- `process_batch()`: Main synchronous API for batch processing
- `_process_batch_async()`: Internal async implementation
- `_process_request_async()`: Individual request processing
- `_calculate_chunk_size()`: Dynamic chunk size optimization

## Provider System

The provider system enables integration with different LLM APIs.

### Base Provider

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

#### Key Methods

- `get_request_url()`: Constructs API endpoint URLs
- `get_request_headers()`: Manages API request headers
- `make_request()`: Handles API communication

### OpenAI Provider

Implements OpenAI-specific functionality:

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

## Caching System

The caching system provides flexible caching solutions.

### Cache Provider Interface

```python
class CacheProvider:
    async def exists(self, key: str) -> bool
    async def get(self, key: str)
    async def put(self, key: str, value) -> None
    async def clear(self) -> None
    async def close(self) -> None
```

### Implementations

#### InMemoryCache

Simple dictionary-based cache for non-persistent storage:

```python
class InMemoryCache(CacheProvider):
    def __init__(self)
```

#### DiskCache

Persistent disk-based cache with TTL support:

```python
class DiskCache(CacheProvider):
    def __init__(
        self,
        directory: str,
        ttl: Optional[int] = None,
        **cache_options
    )
```

## Request/Response Models

### LLMRequest

Base model for LLM requests:

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

### Message

Represents a single message in a conversation:

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"] = "user"
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[dict[str, Any]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
```

### TokenStats

Tracks token usage and performance metrics:

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

Manages progress display and statistics:

```python
class ProgressTracker:
    def __init__(self, total_requests: int, show_progress: bool = True)
```

Features:
- Real-time progress bar
- Token usage statistics
- Cache hit ratio tracking
- Performance metrics

## Request Batching

### RequestBatch

Provides an OpenAI-like interface for batching requests:

```python
class RequestBatch(AbstractContextManager):
    def __init__(self)
```

Usage:
```python
with RequestBatch() as batch:
    batch.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

## Component Interactions

1. **Request Flow**:
   ```
   RequestBatch → RequestManager → Cache Check → Provider → Response
   ```

2. **Caching Flow**:
   ```
   Request → Hash Computation → Cache Lookup → (Cache Hit/Miss) → Response
   ```

3. **Progress Tracking**:
   ```
   Request Processing → Stats Update → Progress Display → Completion
   ```

## Best Practices

1. **Request Management**:
   - Use appropriate concurrency limits
   - Implement proper error handling
   - Monitor token usage

2. **Caching**:
   - Choose appropriate cache provider
   - Configure TTL based on needs
   - Monitor cache hit ratios

3. **Provider Implementation**:
   - Handle rate limits properly
   - Implement comprehensive error handling
   - Follow provider-specific best practices