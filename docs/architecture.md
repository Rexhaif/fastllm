# FastLLM Architecture

FastLLM is a Python library designed to provide a fast, efficient, and flexible interface for working with Large Language Models (LLMs). This document outlines the key architectural components and their interactions.

## Core Components

### 1. Core Module (`core.py`)
The heart of the library, providing fundamental abstractions:
- `Message`: Represents a single message in a conversation
- `LLMRequest`: Encapsulates a request to an LLM provider
- `LLMResponse`: Represents the response from an LLM provider
- `RequestManager`: Manages request processing, batching, and retries
- `RequestBatch`: Handles batching of multiple requests
- `ResponseWrapper`: Wraps provider-specific responses in a standard format
- `TokenStats`: Tracks token usage and rate limiting statistics

### 2. Cache System (`cache.py`)
Provides caching mechanisms to optimize request handling:
- `InMemoryCache`: Fast in-memory caching for development and testing
- `DiskCache`: Persistent caching with TTL support
- `compute_request_hash`: Generates consistent hashes for request caching

### 3. Rate Limiting (`rate_limiting.py`)
Implements rate limiting functionality:
- Token-based rate limiting
- Request-based rate limiting
- Window-based tracking
- Saturation monitoring

### 4. Progress Tracking (`progress.py`)
Handles progress monitoring and statistics:
- Token usage tracking
- Cache hit/miss statistics
- Request completion monitoring
- Rate limit saturation tracking

### 5. Provider Integration (`providers/`)
Provider-specific implementations:
- OpenAI integration
- Support for custom providers
- Provider-specific request/response handling

### 6. CLI Interface (`cli.py`)
Command-line interface for library functionality:
- Request submission
- Cache management
- Statistics viewing

## Key Design Principles

1. **Modularity**: Each component is designed to be independent and replaceable
2. **Extensibility**: Easy to add new providers and functionality
3. **Performance**: Efficient request handling and caching
4. **Reliability**: Built-in retry mechanisms and error handling
5. **Observability**: Comprehensive progress tracking and statistics

## Request Flow

1. Client creates an `LLMRequest`
2. `RequestManager` processes the request:
   - Checks cache for existing response
   - Applies rate limiting
   - Handles retries if needed
3. Provider-specific handler processes the request
4. Response is cached and returned to client

## Caching Architecture

The caching system is designed to be flexible and efficient:

1. **Request Hashing**:
   - Deterministic hashing of requests
   - Order-independent for consistent results
   - Excludes internal fields (_request_id, _order_id)

2. **Cache Implementations**:
   - In-memory for development/testing
   - Disk-based for production use
   - TTL support for cache invalidation

3. **Concurrent Access**:
   - Thread-safe operations
   - Async/await support
   - Lock-free where possible

## Rate Limiting Design

Rate limiting is implemented using a window-based approach:

1. **Token Tracking**:
   - Tracks tokens per minute
   - Separate prompt and completion tracking
   - Window-based saturation calculation

2. **Request Tracking**:
   - Requests per minute monitoring
   - Cache hit exclusion
   - Saturation reporting

## Testing Strategy

The test suite is organized by component:

1. **Core Tests** (`test_core.py`):
   - Message handling
   - Request/Response functionality
   - Request manager behavior
   - Token statistics

2. **Cache Tests** (`test_cache.py`):
   - Cache operations
   - TTL functionality
   - Concurrent access
   - Edge cases

3. **Provider Tests** (`test_providers.py`):
   - Provider-specific functionality
   - Integration testing

4. **Rate Limiting Tests** (`test_rate_limiting.py`):
   - Rate limit enforcement
   - Window management
   - Saturation calculation

## Future Considerations

1. **Scalability**:
   - Distributed caching support
   - Cluster-aware rate limiting
   - Load balancing

2. **Monitoring**:
   - Prometheus metrics
   - Logging enhancements
   - Performance tracking

3. **Provider Support**:
   - Additional LLM providers
   - Provider-specific optimizations
   - Fallback mechanisms

---

This document is maintained as part of the FastLLM project documentation.
Last updated: March 2024 