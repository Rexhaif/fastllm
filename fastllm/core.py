"""Core functionality for parallel LLM API requests."""

import asyncio
import time
import logging
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

# Configure logging
logger = logging.getLogger(__name__)


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
    token_limit: Optional[int] = None  # Rate limit for tokens per minute
    request_limit: Optional[int] = None  # Rate limit for requests per minute
    window_tokens: int = 0  # Tokens in current rate limit window
    window_requests: int = 0  # Requests in current rate limit window
    token_limit: Optional[int] = None  # Rate limit for tokens per minute
    request_limit: Optional[int] = None  # Rate limit for requests per minute
    window_tokens: int = 0  # Tokens in current rate limit window
    window_requests: int = 0  # Requests in current rate limit window

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

    @property
    def token_saturation(self) -> float:
        """Calculate token usage saturation (0.0 to 1.0)."""
        if not self.token_limit or self.elapsed_time == 0:
            return 0.0
        tokens_per_minute = (self.window_tokens / self.elapsed_time) * 60
        return tokens_per_minute / self.token_limit

    @property
    def request_saturation(self) -> float:
        """Calculate request rate saturation (0.0 to 1.0)."""
        if not self.request_limit or self.elapsed_time == 0:
            return 0.0
        requests_per_minute = (self.window_requests / self.elapsed_time) * 60
        return requests_per_minute / self.request_limit

    @property
    def token_saturation(self) -> float:
        """Calculate token usage saturation (0.0 to 1.0)."""
        if not self.token_limit or self.elapsed_time == 0:
            return 0.0
        tokens_per_minute = (self.window_tokens / self.elapsed_time) * 60
        return tokens_per_minute / self.token_limit

    @property
    def request_saturation(self) -> float:
        """Calculate request rate saturation (0.0 to 1.0)."""
        if not self.request_limit or self.elapsed_time == 0:
            return 0.0
        requests_per_minute = (self.window_requests / self.elapsed_time) * 60
        return requests_per_minute / self.request_limit

    def update(self, prompt_tokens: int, completion_tokens: int, is_cache_hit: bool = False) -> None:
        """Update token statistics."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.requests_completed += 1
        if is_cache_hit:
            self.cache_hits += 1
        else:
            # Only update window stats for non-cache hits
            self.window_tokens += prompt_tokens + completion_tokens
            self.window_requests += 1


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
        self.cache = caching_provider
        self.cache = caching_provider

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
        """Process a single request with caching support."""
        # Get order ID and request ID from request
        order_id = request.get('_order_id', 0)
        request_id = request.get('_request_id')
        
        if request_id is None:
            # Only compute if not already present
            from fastllm.cache import compute_request_hash
            request_id = compute_request_hash(request)
            request['_request_id'] = request_id

        # Check cache first if available
        if self.cache is not None:
            try:
                if await self.cache.exists(request_id):
                    cached_response = await self.cache.get(request_id)
                if await self.cache.exists(request_id):
                    cached_response = await self.cache.get(request_id)
                    wrapped = ResponseWrapper(cached_response, request_id, order_id)
                    if progress and wrapped.usage:
                        # Update progress with cache hit
                        progress.update(
                            wrapped.usage.prompt_tokens,
                            wrapped.usage.completion_tokens,
                            is_cache_hit=True
                        )
                    return wrapped
            except Exception as e:
                logger.warning(f"Cache read error: {str(e)}")

        # Process request with retries
        for attempt in range(self.retry_attempts):
            try:
                response = await self.provider.make_request(
                    client,
                    request,
                    self.timeout,
                )
                
                # Get status from either attribute or dict
                status = (
                    response.get('status')
                    if isinstance(response, dict)
                    else getattr(response, 'status', 200)
                )
                
                # Check status before creating wrapper
                if status not in range(200, 300):
                    logger.warning(f"Request {request_id} failed with status {status}")
                    wrapped = ResponseWrapper(response, request_id, order_id)
                    return wrapped

                # Create wrapper and update progress
                wrapped = ResponseWrapper(response, request_id, order_id)
                if progress and wrapped.usage:
                    progress.update(
                        wrapped.usage.prompt_tokens,
                        wrapped.usage.completion_tokens,
                        is_cache_hit=False
                    )

                # Cache successful response
                if self.cache is not None:
                    try:
                        await self.cache.put(request_id, response)
                    except Exception as e:
                        logger.warning(f"Cache write error: {str(e)}")
                
                return wrapped

            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    if progress:
                        # Update progress even for failed requests
                        progress.update(0, 0, is_cache_hit=False)
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
            # Handle raw request list - compute request IDs and add order IDs
            requests = []
            for i, request in enumerate(batch):
                request = request.copy()  # Don't modify original request
                if "_request_id" not in request:
                    from fastllm.cache import compute_request_hash
                    request["_request_id"] = compute_request_hash(request)
                request["_order_id"] = i
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

    async def _make_provider_request(
        self,
        client: Optional[httpx.AsyncClient],
        request: LLMRequest,
    ) -> LLMResponse:
        """Make a single request to the provider."""
        try:
            response_dict = await self.provider.make_request(client, request, self.timeout)
            # Add required fields
            response_dict["request_id"] = id(request)
            response_dict["provider"] = request.provider
            response_dict["raw_response"] = {"provider_response": response_dict.copy()}
            return LLMResponse.from_dict(response_dict)
        except Exception as e:
            # Re-raise provider errors as is
            raise e

    async def _make_provider_request(
        self,
        client: Optional[httpx.AsyncClient],
        request: LLMRequest,
    ) -> LLMResponse:
        """Make a single request to the provider."""
        try:
            response_dict = await self.provider.make_request(client, request, self.timeout)
            # Add required fields
            response_dict["request_id"] = id(request)
            response_dict["provider"] = request.provider
            response_dict["raw_response"] = {"provider_response": response_dict.copy()}
            return LLMResponse.from_dict(response_dict)
        except Exception as e:
            # Re-raise provider errors as is
            raise e


class RequestBatch(AbstractContextManager):
    """A batch of requests to be processed together."""
    """A batch of requests to be processed together."""

    def __init__(self):
        self.requests = []
        self._next_order_id = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __len__(self):
        return len(self.requests)

    def _add_request(self, request: dict[str, Any]) -> str:
        """Add a request to the batch and return its request ID (cache key).
        
        Args:
            request: The request to add to the batch
            
        Returns:
            str: The request ID (cache key) for this request
        """
        # This method is deprecated and only kept for backward compatibility
        # All request creation should go through chat.completions.create
        if "_request_id" not in request:
            # Compute request ID for caching if not already present
            from fastllm.cache import compute_request_hash
            request["_request_id"] = compute_request_hash(request)
        
        # Add order ID for maintaining sequence
        request["_order_id"] = self._next_order_id
        self._next_order_id += 1
        
        # Add to batch
        self.requests.append(request)
        return request["_request_id"]

    @classmethod
    def merge(cls, batches: list["RequestBatch"]) -> "RequestBatch":
        """Merge multiple request batches into a single batch."""
        merged = cls()
        for batch in batches:
            merged.requests.extend(batch.requests)
        return merged

    @property
    def chat(self):
        """Access chat completion methods."""
        """Access chat completion methods."""
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
            ) -> str:
                """Add a chat completion request to the batch.
                
                Args:
                    model: The model to use for completion
                    messages: The messages to generate a completion for
                    temperature: Sampling temperature (0-2)
                    top_p: Nucleus sampling parameter (0-1)
                    n: Number of completions to generate
                    stop: Stop sequences to use
                    max_completion_tokens: Maximum tokens to generate
                    presence_penalty: Presence penalty (-2 to 2)
                    frequency_penalty: Frequency penalty (-2 to 2)
                    logit_bias: Token biases to use
                    user: User identifier
                    response_format: Format for the response
                    seed: Random seed for reproducibility
                    tools: List of tools available to the model
                    tool_choice: Tool choice configuration
                    **kwargs: Additional provider-specific parameters

                Returns:
                    str: The request ID (cache key) for this request
                """
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
                
                # Handle default values consistently:
                # If a parameter is None and has a non-None default, use the default
                defaults = {
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                }
                
                for key, default in defaults.items():
                    if key in request and request[key] is None:
                        request[key] = default
                
                # Remove None values to match OpenAI's behavior
                request = {k: v for k, v in request.items() if v is not None}
                
                # Compute request_id at creation time
                from fastllm.cache import compute_request_hash
                request["_request_id"] = compute_request_hash(request)
                
                # Add to batch and return the request ID
                self.batch.requests.append(request)
                
                # Set order ID
                request["_order_id"] = self.batch._next_order_id
                self.batch._next_order_id += 1
                
                return request["_request_id"]
