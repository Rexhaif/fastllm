# Configuration Guide

This guide covers the various configuration options available in FastLLM and how to use them effectively.

## Provider Configuration

### OpenAI Provider

```python
from fastllm.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-api-key",
    api_base="https://api.openai.com/v1",  # Optional: custom API endpoint
    organization="your-org-id",  # Optional: OpenAI organization ID
    headers={  # Optional: custom headers
        "Custom-Header": "value"
    }
)
```

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| api_key | str | Required | Your API key |
| api_base | str | OpenAI default | API endpoint URL |
| organization | str | None | Organization ID |
| headers | dict | None | Custom headers |

## Request Manager Configuration

```python
from fastllm import RequestManager

manager = RequestManager(
    provider=provider,
    concurrency=100,
    timeout=30.0,
    retry_attempts=3,
    retry_delay=1.0,
    show_progress=True,
    caching_provider=None
)
```

### Performance Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| concurrency | int | 100 | Maximum concurrent requests |
| timeout | float | 30.0 | Request timeout in seconds |
| retry_attempts | int | 3 | Number of retry attempts |
| retry_delay | float | 1.0 | Delay between retries |

### Progress Display

```python
manager = RequestManager(
    provider=provider,
    show_progress=True  # Enable rich progress display
)
```

The progress display shows:
- Completion percentage
- Request count
- Token usage rates
- Cache hit ratio
- Estimated time remaining

## Cache Configuration

### In-Memory Cache

```python
from fastllm.cache import InMemoryCache

cache = InMemoryCache()
```

Best for:
- Development and testing
- Small-scale applications
- Temporary caching needs

### Disk Cache

```python
from fastllm.cache import DiskCache

cache = DiskCache(
    directory="cache",  # Cache directory path
    ttl=3600,  # Cache TTL in seconds
    size_limit=1e9  # Cache size limit in bytes
)
```

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| directory | str | Required | Cache directory path |
| ttl | int | None | Time-to-live in seconds |
| size_limit | int | None | Maximum cache size |

### Cache Integration

```python
manager = RequestManager(
    provider=provider,
    caching_provider=cache
)
```

## Request Configuration

### Basic Request Settings

```python
with RequestBatch() as batch:
    batch.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_completion_tokens=100
    )
```

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | str | Required | Model identifier |
| messages | list | Required | Conversation messages |
| temperature | float | 0.7 | Sampling temperature |
| max_completion_tokens | int | None | Max tokens to generate |
| top_p | float | 1.0 | Nucleus sampling parameter |
| presence_penalty | float | 0.0 | Presence penalty |
| frequency_penalty | float | 0.0 | Frequency penalty |
| stop | list[str] | None | Stop sequences |

## Advanced Configuration

### Custom Chunk Size

The RequestManager automatically calculates optimal chunk sizes, but you can influence this through the concurrency setting:

```python
manager = RequestManager(
    provider=provider,
    concurrency=50  # Will affect chunk size calculation
)
```

Chunk size is calculated as:
```python
chunk_size = min(concurrency * 2, 1000)
```

### Error Handling Configuration

```python
manager = RequestManager(
    provider=provider,
    retry_attempts=5,  # Increase retry attempts
    retry_delay=2.0,  # Increase delay between retries
    timeout=60.0  # Increase timeout
)
```

### Custom Headers

```python
provider = OpenAIProvider(
    api_key="your-api-key",
    headers={
        "User-Agent": "CustomApp/1.0",
        "X-Custom-Header": "value"
    }
)
```

## Environment Variables

FastLLM respects the following environment variables:

```bash
OPENAI_API_KEY=your-api-key
OPENAI_ORG_ID=your-org-id
```

## Best Practices

### Concurrency Settings

- Start with lower concurrency (10-20) and adjust based on performance
- Monitor token usage and API rate limits
- Consider provider-specific rate limits

### Cache Configuration

- Use disk cache for production environments
- Set appropriate TTL based on data freshness requirements
- Monitor cache size and hit ratios

### Error Handling

- Configure retry attempts based on API stability
- Use appropriate timeout values
- Implement proper error handling in your code

### Resource Management

- Close cache providers when done
- Monitor memory usage
- Use appropriate chunk sizes for your use case

## Performance Optimization

### Caching Strategy

- Enable caching for repeated requests
- Use appropriate TTL values
- Monitor cache hit ratios

### Concurrency Tuning

- Adjust concurrency based on:
  * API rate limits
  * System resources
  * Response times

### Memory Management

- Use appropriate chunk sizes
- Monitor memory usage
- Clean up resources properly

## Monitoring and Debugging

### Progress Tracking

```python
manager = RequestManager(
    provider=provider,
    show_progress=True
)
```

### Token Usage Monitoring

Track token usage through the TokenStats class:
- Prompt tokens
- Completion tokens
- Token rates
- Cache hit ratios