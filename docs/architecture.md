# FastLLM Architecture

FastLLM is a Python library designed to provide a fast, efficient, and flexible interface for working with Large Language Models (LLMs). This document outlines the key architectural components and their interactions.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     RequestManager                      │
├─────────────────────────────────────────────────────────┤
│ - Manages parallel request processing                   │
│ - Handles concurrency and batching                      │
│ - Coordinates between components                        │
└───────────────────┬───────────────────┬─────────────────┘
                    │                   │
    ┌───────────────▼──────┐   ┌────────▼──────────┐
    │      Provider        │   │   CacheProvider   │
    ├──────────────────────┤   ├───────────────────┤
    │ - API communication  │   │ - Request caching │
    │ - Response parsing   │   │ - Cache management│
    └──────────────────────┘   └───────────────────┘
```

## Core Components

### 1. Message and Request Models (`core.py`)
- `Message`: Represents a single message in a conversation
  - Supports system, user, assistant, function, and tool roles
  - Handles function calls and tool calls
  - Flexible content types (string or structured data)

- `LLMRequest`: Base model for LLM requests
  - Provider-agnostic request format
  - Configurable parameters (temperature, tokens, penalties)
  - Support for streaming responses
  - Factory methods for creating from prompts or messages

### 2. Response Handling (`core.py`)
- `ResponseWrapper[ResponseT]`: Generic wrapper for provider responses
  - Maintains request ordering
  - Tracks token usage statistics
  - Provider-agnostic interface

- `TokenStats`: Comprehensive token usage tracking
  - Prompt and completion token counts
  - Rate limiting statistics
  - Cache hit ratio monitoring
  - Performance metrics (tokens/second)

### 3. Progress Tracking (`core.py`)
- `ProgressTracker`: Rich progress display
  - Real-time token usage statistics
  - Cache hit/miss ratios
  - ETA and time elapsed
  - Rate limit saturation monitoring

### 4. Cache System (`cache.py`)
- `CacheProvider`: Abstract base class for caching
  - Async interface for all operations
  - Consistent error handling

- Implementations:
  - `InMemoryCache`: Fast, non-persistent cache
  - `DiskCache`: Persistent storage with TTL support
    - Thread-safe operations
    - JSON serialization
    - Configurable directory and options

- `compute_request_hash`: Consistent request hashing
  - Deterministic hash generation
  - Handles core and extra parameters
  - Excludes internal tracking fields

### 5. Provider System (`providers/`)
- `Provider[ResponseT]`: Generic base class
  - Type-safe response handling
  - Standardized HTTP headers
  - Configurable API endpoints

- `OpenAIProvider`: OpenAI API implementation
  - ChatCompletion support
  - Organization handling
  - Custom header management

- `OpenAIRequest`: OpenAI-specific request model
  - Message format conversion
  - Extra parameter handling
  - Function and tool call support

## Key Features

### 1. Parallel Processing
- Async/await throughout
- Efficient request batching
- Concurrent API calls
- Progress monitoring

### 2. Caching
- Multiple cache backends
- TTL support
- Async operations
- Thread-safe implementation

### 3. Rate Limiting
- Token-based limits
- Request frequency limits
- Window-based tracking
- Saturation monitoring

### 4. Error Handling
- Consistent error types
- Graceful degradation
- Detailed error messages
- Retry mechanisms

## Configuration Points

The system can be configured through:
1. Provider settings
   - API keys and endpoints
   - Organization IDs
   - Custom headers

2. Cache settings
   - Backend selection
   - TTL configuration
   - Storage directory
   - Serialization options

3. Request parameters
   - Model selection
   - Temperature and sampling
   - Token limits
   - Response streaming

4. Rate limiting
   - Token rate limits
   - Request frequency limits
   - Window sizes
   - Saturation thresholds

## Best Practices

1. **Error Handling**
   - Use try-except blocks for cache operations
   - Handle API errors gracefully
   - Provide meaningful error messages
   - Implement proper cleanup

2. **Performance**
   - Use appropriate cache backend
   - Configure batch sizes
   - Monitor rate limits
   - Track token usage

3. **Security**
   - Secure API key handling
   - Safe cache storage
   - Input validation
   - Response sanitization

4. **Maintenance**
   - Regular cache cleanup
   - Monitor disk usage
   - Update API versions
   - Track deprecations

## Data Flow

1. **Request Initialization**
   ```python
   RequestBatch
     → LLMRequest
     → Provider-specific Request
   ```

2. **Request Processing**
   ```python
   RequestManager
     → Check Cache
     → Make API Request if needed
     → Update Progress
     → Store in Cache
   ```

3. **Response Handling**
   ```python
   API Response
     → ResponseWrapper
     → Update Statistics
     → Return to User
   ```

## Design Principles

1. **Modularity**
   - Clear separation of concerns
   - Extensible provider system
   - Pluggable cache providers

2. **Performance**
   - Efficient parallel processing
   - Smart resource management
   - Optimized caching

3. **Reliability**
   - Comprehensive error handling
   - Automatic retries
   - Progress tracking

4. **Developer Experience**
   - Familiar API patterns
   - Clear type hints
   - Comprehensive logging

## Error Handling

The system implements comprehensive error handling:
- API errors
- Rate limiting
- Timeouts
- Cache failures
- Invalid requests

Each component includes appropriate error handling and propagation to ensure system stability and reliability.

## Testing Strategy

The test suite is organized by component:

1. **Core Tests** (`test_core.py`):
   - Request/Response model validation
   - RequestManager functionality
   - Token statistics tracking

2. **Cache Tests** (`test_cache.py`):
   - Cache implementation verification
   - Request hashing consistency
   - Concurrent access handling

3. **Provider Tests** (`test_providers.py`):
   - Provider interface compliance
   - API communication
   - Response parsing

4. **Integration Tests**:
   - End-to-end request flow
   - Rate limiting behavior
   - Error handling scenarios
