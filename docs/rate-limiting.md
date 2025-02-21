# Rate Limiting Design

## Overview

This document outlines the design for implementing a flexible rate limiting system for FastLLM providers. The system will support both request-based and token-based rate limits with human-readable time units.

## Requirements

1. Support multiple rate limits per provider
2. Handle both request-per-time-unit and tokens-per-time-unit limits
3. Support human-readable time unit specifications (e.g., "10s", "1m", "1h", "1d")
4. Track token usage from provider responses
5. Integrate with existing Provider and RequestManager architecture

## System Design

### Rate Limit Configuration

Rate limits will be specified in the provider configuration using a list structure:

```python
rate_limits = [
    {
        "type": "requests",
        "limit": 100,
        "period": "1m"
    },
    {
        "type": "tokens",
        "limit": 100000,
        "period": "1h"
    }
]
```

### Components

#### 1. RateLimitManager

A new component responsible for tracking and enforcing rate limits:

```python
class RateLimitManager:
    def __init__(self, limits: List[RateLimit]):
        self.limits = limits
        self.windows = {}  # Tracks usage windows
        
    async def check_limits(self) -> bool:
        # Check all limits
        
    async def record_usage(self, usage: Usage):
        # Record request and token usage
```

#### 2. RateLimit Models

```python
@dataclass
class RateLimit:
    type: Literal["requests", "tokens"]
    limit: int
    period: str  # Human-readable period
    
@dataclass
class Usage:
    request_count: int
    token_count: Optional[int]
    timestamp: datetime
```

#### 3. TimeWindow

Handles the sliding window implementation for rate limiting:

```python
class TimeWindow:
    def __init__(self, period: str):
        self.period = self._parse_period(period)
        self.usage_records = []
        
    def add_usage(self, usage: Usage):
        # Add usage and cleanup old records
        
    def current_usage(self) -> int:
        # Calculate current usage in window
```

### Integration with Existing Architecture

#### 1. Provider Base Class Enhancement

```python
class Provider:
    def __init__(self, rate_limits: List[Dict]):
        self.rate_limit_manager = RateLimitManager(
            [RateLimit(**limit) for limit in rate_limits]
        )
    
    async def _check_rate_limits(self):
        # Check before making requests
        
    async def _record_usage(self, response):
        # Record after successful requests
```

#### 2. RequestManager Integration

The RequestManager will be enhanced to:
- Check rate limits before processing requests
- Handle rate limit errors gracefully
- Implement backoff strategies when limits are reached

### Time Unit Parsing

Time unit parsing will support:
- Seconds: "s", "sec", "second", "seconds"
- Minutes: "m", "min", "minute", "minutes"
- Hours: "h", "hr", "hour", "hours"
- Days: "d", "day", "days"

Example implementation:

```python
def parse_time_unit(period: str) -> timedelta:
    match = re.match(r"(\d+)([smhd])", period)
    if not match:
        raise ValueError("Invalid time unit format")
    
    value, unit = match.groups()
    value = int(value)
    
    return {
        's': timedelta(seconds=value),
        'm': timedelta(minutes=value),
        'h': timedelta(hours=value),
        'd': timedelta(days=value)
    }[unit]
```

## Usage Example

```python
# Provider configuration
openai_provider = OpenAIProvider(
    api_key="...",
    rate_limits=[
        {
            "type": "requests",
            "limit": 100,
            "period": "1m"
        },
        {
            "type": "tokens",
            "limit": 100000,
            "period": "1h"
        }
    ]
)

# Usage in code
async with openai_provider as provider:
    response = await provider.complete(prompt)
    # Rate limits are automatically checked and updated
```

## Error Handling

1. Rate Limit Exceeded
```python
class RateLimitExceeded(Exception):
    def __init__(self, limit_type: str, retry_after: float):
        self.limit_type = limit_type
        self.retry_after = retry_after
```

2. Recovery Strategy
- Implement exponential backoff
- Queue requests when limits are reached
- Provide retry-after information to clients

## Implementation Phases

1. Phase 1: Basic Implementation
   - Implement RateLimitManager
   - Add time unit parsing
   - Integrate with Provider base class

2. Phase 2: Enhanced Features
   - Add token tracking
   - Implement sliding windows
   - Add multiple limit support

3. Phase 3: Optimization
   - Add request queuing
   - Implement smart backoff strategies
   - Add monitoring and metrics

## Monitoring and Metrics

The rate limiting system will expose metrics for:
- Current usage per window
- Remaining quota
- Rate limit hits
- Average usage patterns

These metrics can be integrated with the existing monitoring system.

## Future Considerations

1. Distributed Rate Limiting
   - Support for Redis-based rate limiting
   - Cluster-aware rate limiting

2. Dynamic Rate Limits
   - Allow providers to update limits based on API responses
   - Support for dynamic quota adjustments

3. Rate Limit Optimization
   - Predictive rate limiting
   - Smart request scheduling