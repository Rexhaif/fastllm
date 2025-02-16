# FastLLM

High-performance parallel LLM API request tool with support for multiple providers and caching capabilities.

## Features

- Parallel request processing with configurable concurrency
- Built-in caching support (in-memory and disk-based)
- Progress tracking with token usage statistics
- Support for multiple LLM providers (OpenAI, OpenRouter, etc.)
- OpenAI-style API for request batching
- Retry mechanism with configurable attempts and delays
- Rich CLI interface
- Request deduplication and response ordering

## Installation

```bash
pip install fastllm
```

## Dependencies

FastLLM requires Python 3.9 or later and depends on the following packages:

- `httpx` (^0.27.2) - For async HTTP requests
- `pydantic` (^2.10.6) - For data validation and settings management
- `rich` (^13.9.4) - For beautiful terminal output and progress bars
- `typer` (^0.9.4) - For the command-line interface
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

## Quick Start

```python
from fastllm.core import RequestBatch, RequestManager
from fastllm.providers.openai import OpenAIProvider
from fastllm.cache import InMemoryCache  # or DiskCache

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
with RequestBatch() as batch:
    # Add requests to the batch
    for i in range(10):
        batch.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"What is {i} + {i}?"
            }],
            temperature=0.7,
            include_reasoning=True,  # Optional: include model reasoning
        )

# Process the batch
responses = manager.process_batch(batch)

# Process responses
for response in responses:
    if isinstance(response, Exception):
        print(f"Error: {response}")
    else:
        print(f"Response: {response.response.choices[0].message.content}")
```

## Command Line Interface

Process requests from a file:

```bash
fastllm run prompts.txt --api-key your-api-key --model gpt-3.5-turbo --cache disk
```

Process JSON conversations:

```bash
fastllm run conversations.json --api-key your-api-key --json-format --cache memory
```

## Advanced Usage

### Caching Configuration

FastLLM supports both in-memory and disk-based caching:

```python
from fastllm.cache import InMemoryCache, DiskCache

# In-memory cache (faster, but cleared when process ends)
cache = InMemoryCache()

# Disk cache (persistent, with optional TTL and size limits)
cache = DiskCache(
    directory="./cache",
    ttl=3600,  # Cache TTL in seconds
    size_limit=int(2e9)  # 2GB size limit
)
```

### Custom Providers

Create your own provider by inheriting from the base `Provider` class:

```python
from fastllm.providers.base import Provider
from typing import Any

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