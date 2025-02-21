# Development Guide

This guide provides information for developers who want to contribute to FastLLM or extend its functionality.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- uv for dependency management
- Git for version control

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/Rexhaif/fastllm.git
cd fastllm
```

2. Install dependencies:
```bash
uv pip install --with dev
```

3. Activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

## Project Structure

```
fastllm/
├── fastllm/
│   ├── __init__.py
│   ├── cache.py          # Caching system
│   ├── cli.py           # CLI interface
│   ├── core.py          # Core functionality
│   └── providers/       # Provider implementations
│       ├── __init__.py
│       ├── base.py      # Base provider class
│       └── openai.py    # OpenAI provider
├── tests/               # Test suite
├── examples/            # Example code
├── pyproject.toml      # Project configuration
└── README.md           # Project documentation
```

## Adding a New Provider

To add support for a new LLM provider:

1. Create a new file in `fastllm/providers/`:

```python
# fastllm/providers/custom_provider.py

from typing import Any, Optional
import httpx
from fastllm.providers.base import Provider
from fastllm.core import ResponseT

class CustomProvider(Provider[ResponseT]):
    def __init__(
        self,
        api_key: str,
        api_base: str,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any
    ):
        super().__init__(api_key, api_base, headers, **kwargs)
        # Add provider-specific initialization

    def get_request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.headers
        }

    async def make_request(
        self,
        client: httpx.AsyncClient,
        request: dict[str, Any],
        timeout: float
    ) -> ResponseT:
        # Implement provider-specific request handling
        url = self.get_request_url("endpoint")
        response = await client.post(
            url,
            headers=self.get_request_headers(),
            json=request,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
```

2. Add tests for the new provider:

```python
# tests/test_custom_provider.py

import pytest
from fastllm.providers.custom_provider import CustomProvider

@pytest.fixture
def provider():
    return CustomProvider(
        api_key="test-key",
        api_base="https://api.example.com"
    )

async def test_make_request(provider):
    # Implement provider tests
    pass
```

## Adding a New Cache Provider

To implement a new cache provider:

1. Create a new cache implementation:

```python
from fastllm.cache import CacheProvider

class CustomCache(CacheProvider):
    def __init__(self, **options):
        # Initialize your cache

    async def exists(self, key: str) -> bool:
        # Implement key existence check

    async def get(self, key: str):
        # Implement value retrieval

    async def put(self, key: str, value) -> None:
        # Implement value storage

    async def clear(self) -> None:
        # Implement cache clearing

    async def close(self) -> None:
        # Implement cleanup
```

2. Add tests for the new cache:

```python
# tests/test_custom_cache.py

import pytest
from your_cache_module import CustomCache

@pytest.fixture
async def cache():
    cache = CustomCache()
    yield cache
    await cache.close()

async def test_cache_operations(cache):
    # Implement cache tests
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fastllm

# Run specific test file
pytest tests/test_specific.py
```

### Writing Tests

1. Use pytest fixtures for common setup:

```python
@pytest.fixture
def request_manager(provider, cache):
    return RequestManager(
        provider=provider,
        caching_provider=cache
    )
```

2. Test async functionality:

```python
@pytest.mark.asyncio
async def test_async_function():
    # Test implementation
    pass
```

3. Use mocking when appropriate:

```python
from unittest.mock import Mock, patch

def test_with_mock():
    with patch('module.function') as mock_func:
        # Test implementation
        pass
```

## Code Style

### Style Guide

- Follow PEP 8 guidelines
- Use type hints
- Document public APIs
- Keep functions focused and small

### Code Formatting

The project uses:
- Black for code formatting
- Ruff for linting
- isort for import sorting

```bash
# Format code
black .

# Sort imports
isort .

# Run linter
ruff check .
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function(param1: str, param2: int) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        param1: Parameter description
        param2: Parameter description

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this error occurs
    """
```

### Building Documentation

The project uses Markdown for documentation:

1. Place documentation in `docs/`
2. Use clear, concise language
3. Include code examples
4. Keep documentation up to date

## Performance Considerations

When developing new features:

1. **Concurrency**
   - Use async/await properly
   - Avoid blocking operations
   - Handle resources correctly

2. **Memory Usage**
   - Monitor memory consumption
   - Clean up resources
   - Use appropriate data structures

3. **Caching**
   - Implement efficient caching
   - Handle cache invalidation
   - Consider memory vs. speed tradeoffs

## Error Handling

1. Use appropriate exception types:

```python
class CustomProviderError(Exception):
    """Base exception for custom provider."""
    pass

class CustomProviderAuthError(CustomProviderError):
    """Authentication error for custom provider."""
    pass
```

2. Implement proper error handling:

```python
async def make_request(self, ...):
    try:
        response = await self._make_api_call()
    except httpx.TimeoutException as e:
        raise CustomProviderError(f"API timeout: {e}")
    except httpx.HTTPError as e:
        raise CustomProviderError(f"HTTP error: {e}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

### Commit Guidelines

- Use clear commit messages
- Reference issues when applicable
- Keep commits focused and atomic

### Pull Request Process

1. Update documentation
2. Add tests
3. Ensure CI passes
4. Request review
5. Address feedback

## Release Process

1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Create release commit
4. Tag release
5. Push to repository
6. Build and publish to PyPI