"""Core functionality for parallel LLM API requests."""

import asyncio
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import httpx
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, model_validator
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)


# Define a type variable for provider-specific response types
ResponseT = TypeVar("ResponseT", bound=Union[ChatCompletion, Any])


class ResponseWrapper(Generic[ResponseT]):
    """Wrapper for provider responses that includes request ID for sorting."""

    def __init__(self, response: ResponseT, request_id: str, order_id: int):
        self.response = response
        self.request_id = request_id
        self._order_id = order_id

    @property
    def usage(self) -> Optional[CompletionUsage]:
        """Get usage statistics if available."""
        if isinstance(self.response, ChatCompletion):
            return self.response.usage
        return None


@dataclass
class TokenStats:
    """Statistics about token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests_completed: int = 0
    cache_hits: int = 0  # Track cache hits
    start_time: float = 0.0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.elapsed_time == 0:
            return 0.0
        return self.prompt_tokens / self.elapsed_time

    @property
    def completion_tokens_per_second(self) -> float:
        if self.elapsed_time == 0:
            return 0.0
        return self.completion_tokens / self.elapsed_time

    @property
    def cache_hit_ratio(self) -> float:
        if self.requests_completed == 0:
            return 0.0
        return self.cache_hits / self.requests_completed

    def update(self, prompt_tokens: int, completion_tokens: int, is_cache_hit: bool = False) -> None:
        """Update token statistics."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.requests_completed += 1
        if is_cache_hit:
            self.cache_hits += 1


class ProgressTracker:
    """Tracks progress and token usage for batch requests."""

    def __init__(self, total_requests: int, show_progress: bool = True):
        self.stats = TokenStats(start_time=time.time())
        self.total_requests = total_requests
        self.show_progress = show_progress

        # Create progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("[blue]{task.fields[stats]}"),
            TextColumn("[yellow]{task.fields[cache]}"),
            disable=not show_progress,
        )

        # Add main progress task
        self.task_id = self.progress.add_task(
            description="Processing requests",
            total=total_requests,
            stats="Starting...",
            cache="",
        )

    def __enter__(self):
        """Start progress display."""
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress display."""
        self.progress.stop()

    def update(self, prompt_tokens: int, completion_tokens: int, is_cache_hit: bool = False):
        """Update progress and token statistics."""
        self.stats.update(prompt_tokens, completion_tokens, is_cache_hit)

        # Update progress display with token rates and cache stats
        stats_text = (
            f"[green]⬆ {self.stats.prompt_tokens_per_second:.1f}[/green] "
            f"[red]⬇ {self.stats.completion_tokens_per_second:.1f}[/red] t/s"
        )
        
        cache_text = (
            f"Cache: [green]{self.stats.cache_hit_ratio*100:.1f}%[/green] hits, "
            f"[yellow]{(1-self.stats.cache_hit_ratio)*100:.1f}%[/yellow] new"
        )

        self.progress.update(
            self.task_id,
            advance=1,
            stats=stats_text,
            cache=cache_text,
        )


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant", "function", "tool"] = "user"
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[dict[str, Any]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, data: Union[str, dict[str, Any]]) -> "Message":
        """Create a message from a string or dictionary."""
        if isinstance(data, str):
            return cls(role="user", content=data)
        return cls(**data)


MessageType = Union[Message, dict[str, Any], str]


class LLMRequest(BaseModel):
    """Base model for LLM requests."""

    provider: str
    messages: list[Message]
    model: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[list[str]] = None
    stream: bool = False

    @model_validator(mode="before")
    @classmethod
    def validate_messages(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert message dictionaries to Message objects."""
        if "messages" in data:
            if isinstance(data["messages"], list):
                data["messages"] = [
                    msg if isinstance(msg, Message) else Message.from_dict(msg)
                    for msg in data["messages"]
                ]
        elif "prompt" in data:  # Backward compatibility
            data["messages"] = [Message(role="user", content=data.pop("prompt"))]
        return data

    @classmethod
    def from_prompt(
        cls, provider: str, prompt: Union[str, dict[str, Any]], **kwargs
    ) -> "LLMRequest":
        """Create a request from a single prompt."""
        if isinstance(prompt, str):
            messages = [Message(role="user", content=prompt)]
        else:
            messages = [Message.from_dict(prompt)]
        return cls(provider=provider, messages=messages, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMRequest":
        """Create a request from a dictionary."""
        return cls(**data)


class LLMResponse(BaseModel):
    """Base model for LLM responses."""

    request_id: int | str
    provider: str
    content: str
    raw_response: dict[str, Any]
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, int]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMResponse":
        """Create a response from a dictionary."""
        return cls(**data)


class RequestManager:
    """Manages parallel LLM API requests."""

    def __init__(
        self,
        provider: 'Provider[ResponseT]',
        concurrency: int = 100,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        show_progress: bool = True,
        caching_provider: Optional['CacheProvider'] = None
    ):
        self.provider = provider
        self.concurrency = concurrency
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.show_progress = show_progress
        self.cache_provider = caching_provider

    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on concurrency.
        
        The chunk size is calculated as 2 * concurrency to allow for some overlap
        and better resource utilization while still maintaining reasonable memory usage.
        This provides a balance between creating too many tasks at once and
        underutilizing the available concurrency.
        """
        return min(self.concurrency * 2, 1000)  # Cap at 1000 to prevent excessive memory usage

    def process_batch(
        self,
        batch: Union[list[dict[str, Any]], "RequestBatch"],
    ) -> list[ResponseT]:
        """Process a batch of LLM requests in parallel.

        This is the main synchronous API endpoint that users should call.
        Internally it uses asyncio to handle requests concurrently.
        Works in both regular Python environments and Jupyter notebooks.

        Args:
            batch: Either a RequestBatch object or a list of request dictionaries

        Returns:
            List of responses in the same order as the requests
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # We're in a Jupyter notebook or similar environment
                # where the loop is already running
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(self._process_batch_async(batch))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self._process_batch_async(batch))

    async def _process_request_async(
        self,
        client: httpx.AsyncClient,
        request: dict[str, Any],
        progress: Optional[ProgressTracker] = None,
    ) -> ResponseWrapper[ResponseT]:
        # Get order ID and request ID from request
        order_id = request.get('_order_id', 0)
        
        # Get or compute request ID (cache key)
        request_id = request.get('_request_id')
        if request_id is None:
            # Only compute if not already present
            from fastllm.cache import compute_request_hash
            request_id = compute_request_hash(request)
            request['_request_id'] = request_id

        # Check if response is already cached
        if self.cache_provider is not None:
            try:
                if await self.cache_provider.exists(request_id):
                    cached_response = await self.cache_provider.get(request_id)
                    wrapped = ResponseWrapper(cached_response, request_id, order_id)
                    if progress and wrapped.usage:
                        progress.update(wrapped.usage.prompt_tokens, wrapped.usage.completion_tokens, True)
                    return wrapped
            except Exception:
                # If there's any error reading from cache, proceed with the actual request
                pass

        # Process request with retries
        for attempt in range(self.retry_attempts):
            try:
                response = await self.provider.make_request(
                    client,
                    request,
                    self.timeout,
                )
                # Create wrapper and update progress before caching
                wrapped = ResponseWrapper(response, request_id, order_id)
                if progress and wrapped.usage:
                    progress.update(
                        wrapped.usage.prompt_tokens,
                        wrapped.usage.completion_tokens,
                        False
                    )
                # Only cache after successful processing
                if self.cache_provider is not None:
                    try:
                        await self.cache_provider.put(request_id, response)
                    except Exception:
                        # If caching fails, we can still return the response
                        pass
                return wrapped
            except Exception:
                if attempt == self.retry_attempts - 1:
                    if progress:
                        progress.update(0, 0, False)
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def _process_batch_async(
        self,
        batch: Union[list[dict[str, Any]], "RequestBatch"],
    ) -> list[ResponseWrapper[ResponseT]]:
        """Internal async implementation of batch processing."""
        # Create semaphore for this batch processing run
        semaphore = asyncio.Semaphore(self.concurrency)

        # Convert RequestBatch to list of requests if needed
        if isinstance(batch, RequestBatch):
            requests = batch.requests
        else:
            # Add request IDs and order IDs to raw request list
            from fastllm.cache import compute_request_hash
            requests = []
            for i, request in enumerate(batch):
                request = request.copy()  # Don't modify original request
                request["_order_id"] = i
                request["_request_id"] = compute_request_hash(request)  # Compute and store request ID
                requests.append(request)

        # Create progress tracker if enabled
        tracker = (
            ProgressTracker(len(requests), show_progress=self.show_progress)
            if self.show_progress
            else None
        )

        async def process_request_with_semaphore(
            client: httpx.AsyncClient,
            request: dict[str, Any],
            progress: Optional[ProgressTracker] = None,
        ) -> ResponseWrapper[ResponseT]:
            """Process a single request with semaphore control."""
            async with semaphore:
                return await self._process_request_async(client, request, progress)

        async def process_batch_chunk(
            client: httpx.AsyncClient, chunk: list[dict[str, Any]]
        ) -> list[ResponseWrapper[ResponseT]]:
            """Process a chunk of requests."""
            batch_tasks = [
                process_request_with_semaphore(client, req, tracker) for req in chunk
            ]
            results = await asyncio.gather(*batch_tasks)
            return [(r._order_id, r) for r in results]

        # Process requests in chunks based on calculated chunk size
        chunk_size = self._calculate_chunk_size()
        all_results = []
        context = tracker if tracker else nullcontext()

        # Create a single client for the entire batch
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with context:
                for batch_start in range(0, len(requests), chunk_size):
                    batch_requests = requests[
                        batch_start : batch_start + chunk_size
                    ]
                    batch_results = await process_batch_chunk(client, batch_requests)
                    all_results.extend(batch_results)

        # Sort responses by order ID and return just the responses
        return [r for _, r in sorted(all_results, key=lambda x: x[0])]


class RequestBatch(AbstractContextManager):
    """A batch of requests that mimics OpenAI's API style.

    Usage:
        with RequestBatch() as batch:
            # Add requests to the batch
            batch.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}]
            )

        # Process the batch synchronously
        responses = manager.process_batch(batch)
    """

    def __init__(self):
        self.requests = []
        self._request_counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __len__(self):
        return len(self.requests)

    def _add_request(self, request: dict[str, Any]) -> None:
        """Add a request with sequential IDs to maintain order and caching."""
        from fastllm.cache import compute_request_hash
        request["_order_id"] = self._request_counter
        request["_request_id"] = compute_request_hash(request)  # Compute and store request ID
        self.requests.append(request)
        self._request_counter += 1

    @property
    def chat(self):
        """Access to chat completions API."""
        return self.Chat(self)

    class Chat:
        """Chat API that mimics OpenAI's interface."""

        def __init__(self, batch):
            self.batch = batch
            self.completions = self.Completions(batch)

        class Completions:
            """Chat completions API that mimics OpenAI's interface."""

            def __init__(self, batch):
                self.batch = batch

            def create(
                self,
                *,
                model: str,
                messages: list[dict[str, str]],
                temperature: Optional[float] = 0.7,
                top_p: Optional[float] = 1.0,
                n: Optional[int] = 1,
                stop: Optional[Union[str, list[str]]] = None,
                max_completion_tokens: Optional[int] = None,
                presence_penalty: Optional[float] = 0.0,
                frequency_penalty: Optional[float] = 0.0,
                logit_bias: Optional[dict[str, float]] = None,
                user: Optional[str] = None,
                response_format: Optional[dict[str, str]] = None,
                seed: Optional[int] = None,
                tools: Optional[list[dict[str, Any]]] = None,
                tool_choice: Optional[Union[str, dict[str, str]]] = None,
                **kwargs: Any
            ) -> None:
                """Add a chat completion request to the batch."""
                request = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "n": n,
                    "stop": stop,
                    "max_completion_tokens": max_completion_tokens,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "logit_bias": logit_bias,
                    "user": user,
                    "response_format": response_format,
                    "seed": seed,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    **kwargs,
                }
                # Remove None values to match OpenAI's behavior
                request = {k: v for k, v in request.items() if v is not None}
                self.batch._add_request(request)
