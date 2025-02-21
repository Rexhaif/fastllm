# Getting Started with FastLLM

This guide will help you get started with FastLLM for efficient parallel LLM API requests.

## Installation

Install FastLLM using pip:

```bash
pip install fastllm
```

## Quick Start

Here's a simple example to get you started:

```python
from fastllm import RequestManager
from fastllm.providers import OpenAIProvider

# Initialize the provider
provider = OpenAIProvider(
    api_key="your-api-key",
    organization="your-org-id"  # Optional
)

# Create request manager
manager = RequestManager(
    provider=provider,
    concurrency=10,  # Number of concurrent requests
    show_progress=True  # Show progress bar
)

# Create a batch of requests
from fastllm import RequestBatch

with RequestBatch() as batch:
    # Add multiple requests to the batch
    batch.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    batch.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "How are you?"}]
    )

# Process the batch
responses = manager.process_batch(batch)

# Work with responses
for response in responses:
    print(response.content)
```

## Adding Caching

Enable caching to improve performance and reduce API calls:

```python
from fastllm.cache import DiskCache

# Initialize disk cache
cache = DiskCache(
    directory="cache",  # Cache directory
    ttl=3600  # Cache TTL in seconds
)

# Create request manager with cache
manager = RequestManager(
    provider=provider,
    concurrency=10,
    caching_provider=cache,
    show_progress=True
)
```

## Processing Large Batches

For large batches of requests:

```python
# Create many requests
with RequestBatch() as batch:
    for i in range(1000):
        batch.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Process item {i}"}
            ]
        )

# Process with automatic chunking and progress tracking
responses = manager.process_batch(batch)
```

## Custom Configuration

### Timeout and Retry Settings

```python
manager = RequestManager(
    provider=provider,
    timeout=30.0,  # Request timeout in seconds
    retry_attempts=3,  # Number of retry attempts
    retry_delay=1.0,  # Delay between retries
)
```

### Provider Configuration

```python
provider = OpenAIProvider(
    api_key="your-api-key",
    api_base="https://api.openai.com/v1",  # Custom API endpoint
    headers={
        "Custom-Header": "value"
    }
)
```

## Progress Tracking

FastLLM provides detailed progress information:

```python
manager = RequestManager(
    provider=provider,
    show_progress=True  # Enable progress bar
)

# Progress will show:
# - Completion percentage
# - Token usage rates
# - Cache hit ratio
# - Estimated time remaining
```

## Error Handling

```python
try:
    responses = manager.process_batch(batch)
except Exception as e:
    print(f"Error processing batch: {e}")
```

## Advanced Usage

### Custom Message Formatting

```python
from fastllm import Message

# Create structured messages
messages = [
    Message(role="system", content="You are a helpful assistant"),
    Message(role="user", content="Hello!")
]

with RequestBatch() as batch:
    batch.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
```

### Working with Response Data

```python
for response in responses:
    # Access response content
    print(response.content)
    
    # Access usage statistics
    if response.usage:
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
```

## Best Practices

1. **Batch Processing**
   - Group related requests together
   - Use appropriate concurrency limits
   - Enable progress tracking for large batches

2. **Caching**
   - Use disk cache for persistence
   - Set appropriate TTL values
   - Monitor cache hit ratios

3. **Error Handling**
   - Implement proper error handling
   - Use retry mechanisms
   - Monitor token usage

4. **Resource Management**
   - Close cache providers when done
   - Monitor memory usage
   - Use appropriate chunk sizes

## Next Steps

- Read the [Architecture](architecture.md) documentation
- Explore [Core Components](core-components.md)
- Check the [API Reference](api-reference.md)