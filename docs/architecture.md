# FastLLM Architecture

FastLLM follows a modular architecture designed for extensibility, performance, and ease of use. This document outlines the key architectural components and their interactions.

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

### 1. RequestManager

The RequestManager is the central coordinator that:
- Processes batches of requests concurrently
- Manages concurrency limits and chunking
- Coordinates between providers and cache
- Handles progress tracking and statistics

Key features:
- Dynamic chunk size calculation
- Semaphore-based concurrency control
- Automatic retry mechanism
- Progress monitoring

### 2. Provider System

The provider system is built around a base Provider class that:
- Defines the interface for LLM API providers
- Handles authentication and headers
- Manages API communication
- Processes responses

Components:
- `Provider` (Base class)
- `OpenAIProvider` (Implementation)
- Request/Response models

### 3. Caching System

The caching system provides:
- Multiple cache implementations
- Consistent request hashing
- Async cache operations
- Optional TTL support

Implementations:
- `InMemoryCache`: Fast, non-persistent storage
- `DiskCache`: Persistent storage with TTL support

### 4. Request/Response Models

Core models that handle:
- Request validation and normalization
- Message formatting
- Response parsing
- Usage statistics

Key models:
- `LLMRequest`
- `LLMResponse`
- `Message`
- `TokenStats`

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

## Configuration Points

The system can be configured through:
- Concurrency limits
- Chunk sizes
- Timeout settings
- Retry policies
- Cache settings
- Provider-specific options

## Error Handling

The system implements comprehensive error handling:
- API errors
- Rate limiting
- Timeouts
- Cache failures
- Invalid requests

Each component includes appropriate error handling and propagation to ensure system stability and reliability.