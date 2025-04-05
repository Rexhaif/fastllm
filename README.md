# FastLLM

High-performance parallel LLM API request tool with support for multiple providers and caching capabilities.

## Features

- Parallel request processing with configurable concurrency
  - Allows you to process 20000+ prompt tokens per second and 1500+ output tokens per second even for extremely large LLMs, such as Deepseek-V3.
- Built-in caching support (in-memory and disk-based)
- Progress tracking with token usage statistics
- Support for multiple LLM providers (OpenAI, OpenRouter, etc.)
- OpenAI-style API for request batching
- Retry mechanism with configurable attempts and delays
- Request deduplication and response ordering

## Installation

First, install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install fastllm:
```bash
uv pip install git+https://github.com/Rexhaif/fastllm.git
```

For development:
```bash
# Clone the repository
git clone https://github.com/rexhaif/fastllm
cd fastllm

# Create a virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"
```

## Dependencies

FastLLM requires Python 3.9 or later and depends on the following packages:

- `httpx` (^0.27.2) - For async HTTP requests
- `pydantic` (^2.10.6) - For data validation and settings management
- `rich` (^13.9.4) - For beautiful terminal output and progress bars
- `diskcache` (^5.6.3) - For persistent disk caching
- `asyncio` (^3.4.3) - For asynchronous operations
- `anyio` (^4.8.0) - For async I/O operations
- `tqdm` (^4.67.1) - For progress tracking
- `typing_extensions` (^4.12.2) - For enhanced type hints

Development dependencies:
- `ruff` (^0.3.7) - For linting and formatting
- `pytest` (^8.3.4) - For testing
- `pytest-asyncio` (^0.23.8) - For async tests
- `pytest-cov` (^4.1.0) - For test coverage
- `black` (^24.10.0) - For code formatting
- `coverage` (^7.6.10) - For code coverage reporting

## Development

The project uses [just](https://github.com/casey/just) for task automation and [uv](https://github.com/astral/uv) for dependency management.

Common tasks:
```bash
# Install dependencies
just install

# Run tests
just test

# Format code
just format

# Run linting
just lint

# Clean up cache files
just clean
```

## Quick Start

```python
from fastllm import RequestBatch, RequestManager, OpenAIProvider, InMemoryCache

# Create a provider
provider = OpenAIProvider(
    api_key="your-api-key",
    # Optional: custom API base URL
    api_base="https://api.openai.com/v1",
)

# Create a cache provider (optional)
cache = InMemoryCache()  # or DiskCache(directory="./cache")

# Create a request manager
manager = RequestManager(
    provider=provider,
    concurrency=50,  # Number of concurrent requests
    show_progress=True,  # Show progress bar
    caching_provider=cache,  # Enable caching
)

# Create a batch of requests
request_ids = []  # Store request IDs for later use
with RequestBatch() as batch:
    # Add requests to the batch
    for i in range(10):
        # create() returns the request ID (caching key)
        request_id = batch.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"What is {i} + {i}?"
            }],
            temperature=0.7,
            include_reasoning=True,  # Optional: include model reasoning
        )
        request_ids.append(request_id)

# Process the batch
responses = manager.process_batch(batch)

# Process responses
for request_id, response in zip(request_ids, responses):
    if isinstance(response, Exception):
        print(f"Request {request_id} failed: {response}")
    else:
        print(f"Request {request_id}: {response.response.choices[0].message.content}")
        
# You can use request IDs to check cache status
for request_id in request_ids:
    is_cached = await cache.exists(request_id)
    print(f"Request {request_id} is {'cached' if is_cached else 'not cached'}")
```

## Advanced Usage

### Async Support

FastLLM can be used both synchronously and asynchronously, and works seamlessly in regular Python environments, async applications, and Jupyter notebooks:

```python
import asyncio
from fastllm import RequestBatch, RequestManager, OpenAIProvider

# Works in Jupyter notebooks
provider = OpenAIProvider(api_key="your-api-key")
manager = RequestManager(provider=provider)
responses = manager.process_batch(batch)  # Just works!

# Works in async applications
async def process_requests():
    provider = OpenAIProvider(api_key="your-api-key")
    manager = RequestManager(provider=provider)
    
    with RequestBatch() as batch:
        batch.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    
    responses = manager.process_batch(batch)
    return responses

# Run in existing event loop
async def main():
    responses = await process_requests()
    print(responses)

asyncio.run(main())
```

### Caching Configuration

FastLLM supports both in-memory and disk-based caching, with request IDs serving as cache keys:

```python
from fastllm import InMemoryCache, DiskCache, RequestBatch

# Create a batch and get request IDs
with RequestBatch() as batch:
    request_id = batch.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(f"Request ID (cache key): {request_id}")

# In-memory cache (faster, but cleared when process ends)
cache = InMemoryCache()

# Disk cache (persistent, with optional TTL and size limits)
cache = DiskCache(
    directory="./cache",
    ttl=3600,  # Cache TTL in seconds
    size_limit=int(2e9)  # 2GB size limit
)

# Check if a response is cached
is_cached = await cache.exists(request_id)

# Get cached response if available
if is_cached:
    response = await cache.get(request_id)
```

### Custom Providers

Create your own provider by inheriting from the base `Provider` class:

```python
from fastllm import Provider
from typing import Any
import httpx

class CustomProvider(Provider[YourResponseType]):
    def get_request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def make_request(
        self,
        client: httpx.AsyncClient,
        request: dict[str, Any],
        timeout: float,
    ) -> YourResponseType:
        # Implement your request logic here
        pass
```

### Progress Tracking

The progress bar shows:
- Request completion progress
- Tokens per second (prompt and completion)
- Cache hit/miss statistics
- Estimated time remaining
- Total elapsed time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
