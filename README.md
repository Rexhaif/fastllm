# FastLLM

High-performance parallel LLM API request tool with support for multiple providers.

## Features

- Parallel request processing with configurable concurrency
- Progress tracking with token usage statistics
- Support for multiple LLM providers
- OpenAI-style API for request batching
- Retry mechanism with configurable attempts and delays
- Rich CLI interface

## Installation

```bash
pip install fastllm
```

## Quick Start

```python
from fastllm.core import RequestBatch, RequestManager
from fastllm.providers.openai import OpenAIProvider

# Create a provider
provider = OpenAIProvider(
    api_key="your-api-key",
    # Optional: custom API base URL
    api_base="https://api.openai.com/v1",
)

# Create a request manager
manager = RequestManager(
    provider=provider,
    concurrency=50,  # Number of concurrent requests
    show_progress=True,  # Show progress bar
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
fastllm run prompts.txt --api-key your-api-key --model gpt-3.5-turbo
```

Process JSON conversations:

```bash
fastllm run conversations.json --api-key your-api-key --json-format
```

## Advanced Usage

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
- Estimated time remaining
- Total elapsed time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 