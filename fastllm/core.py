"""Core functionality for parallel LLM API requests."""

import asyncio
import time
import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
)

import httpx
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
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

from fastllm.cache import compute_request_hash

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
        elif isinstance(self.response, dict) and 'usage' in self.response:
            # Handle dict responses (like embeddings)
            usage = self.response['usage']
            # Convert to CompletionUsage if not already
            if not isinstance(usage, CompletionUsage):
                return CompletionUsage(
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    completion_tokens=usage.get('completion_tokens', 0),
                    total_tokens=usage.get('total_tokens', 0)
                )
            return usage
        return None
        
    @property
    def is_embedding_response(self) -> bool:
        """Check if this is an embedding response."""
        if isinstance(self.response, dict):
            return 'data' in self.response and all('embedding' in item for item in self.response.get('data', []))
        return False
        
    @property
    def embeddings(self) -> list:
        """Get embeddings from response if available."""
        if not self.is_embedding_response:
            return []
            
        if isinstance(self.response, dict) and 'data' in self.response:
            return [item.get('embedding', []) for item in self.response.get('data', [])]
        return []


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

    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on concurrency.
        
        The chunk size is calculated as 2 * concurrency to allow for some overlap
        and better resource utilization while still maintaining reasonable memory usage.
        This provides a balance between creating too many tasks at once and
        underutilizing the available concurrency.
        """
        return min(self.concurrency * 2, 25000)  # Cap at 25000 to prevent excessive memory usage

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
                    wrapped = ResponseWrapper(cached_response, request_id, order_id)
                    if progress and wrapped.usage:
                        # Update progress with cache hit
                        progress.update(
                            wrapped.usage.prompt_tokens,
                            wrapped.usage.completion_tokens or 0,  # Handle embeddings having no completion tokens
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

                # Create wrapper and update progress
                wrapped = ResponseWrapper(response, request_id, order_id)
                
                if progress:
                    # For embeddings, usage only has prompt_tokens
                    if isinstance(response, dict) and 'usage' in response:
                        usage = response['usage']
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        # Embeddings don't have completion tokens
                        completion_tokens = usage.get('completion_tokens', 0)
                        progress.update(
                            prompt_tokens,
                            completion_tokens,
                            is_cache_hit=False
                        )
                    elif wrapped.usage:
                        progress.update(
                            wrapped.usage.prompt_tokens,
                            wrapped.usage.completion_tokens or 0,  # Handle embeddings having no completion tokens
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
            # Extract original requests from batch format
            requests = []
            for batch_req in batch.requests:
                # Extract request_id and order_id from custom_id
                request_id, order_id_str = batch_req["custom_id"].split("#")
                order_id = int(order_id_str)
                
                # Determine type from URL
                req_type = "chat_completion" if batch_req["url"] == "/v1/chat/completions" else "embedding"
                
                # Extract the original request from the batch format
                request = {
                    **batch_req["body"],
                    "_request_id": request_id,
                    "_order_id": order_id,
                    "type": req_type
                }
                requests.append(request)
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
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a single request to the provider.
        
        This method handles both direct dictionaries and OpenAI Batch format requests.
        """
        try:
            # If request is in OpenAI Batch format, extract the body and metadata
            if "custom_id" in request and "url" in request and "body" in request:
                request_id, _ = request["custom_id"].split("#")
                provider_name = self.provider.__class__.__name__.replace("Provider", "").lower()
                request_body = request["body"]
                request_type = "chat_completion" if request["url"] == "/v1/chat/completions" else "embedding"
            else:
                # Legacy format
                request_id = id(request)
                provider_name = request.get("provider", self.provider.__class__.__name__.replace("Provider", "").lower())
                request_body = request
                request_type = request.get("type", "chat_completion")
            
            # Make the actual request
            response_dict = await self.provider.make_request(client, request_body, self.timeout)
            
            # Add required fields
            response_dict["request_id"] = request_id
            response_dict["provider"] = provider_name
            response_dict["raw_response"] = {"provider_response": response_dict.copy()}
            
            return response_dict
        except Exception as e:
            # Re-raise provider errors as is
            raise e


class RequestBatch(AbstractContextManager):
    """A batch of requests to be processed together in OpenAI Batch format."""

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
        
        # Compute request ID for caching if not already present
        request_id = compute_request_hash(request)
        order_id = self._next_order_id
        self._next_order_id += 1
        
        # Determine the endpoint URL based on request type
        url = "/v1/chat/completions"
        if request.get("type") == "embedding":
            url = "/v1/embeddings"
        
        # Create a custom_id from request_id and order_id
        custom_id = f"{request_id}#{order_id}"
        
        # Create batch format request directly
        batch_request = {
            "custom_id": custom_id,
            "url": url,
            "body": {k: v for k, v in request.items() if k not in ["type"]}
        }
        
        # Add to batch
        self.requests.append(batch_request)
        return request_id

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
        return self.Chat(self)
        
    @property
    def embeddings(self):
        """Access embeddings methods."""
        return self.Embeddings(self)

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
                # Create the request body
                body = {
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
                    if key in body and body[key] is None:
                        body[key] = default
                
                # Remove None values to match OpenAI's behavior
                body = {k: v for k, v in body.items() if v is not None}
                
                # Compute request_id at creation time
                
                request_id = compute_request_hash({"type": "chat_completion", **body})
                order_id = self.batch._next_order_id
                self.batch._next_order_id += 1
                
                # Create custom_id from request_id and order_id
                custom_id = f"{request_id}#{order_id}"
                
                # Create the batch request directly in OpenAI Batch format
                batch_request = {
                    "custom_id": custom_id,
                    "url": "/v1/chat/completions",
                    "body": body
                }
                
                # Add to batch
                self.batch.requests.append(batch_request)
                
                return request_id
    
    class Embeddings:
        """Embeddings API that mimics OpenAI's interface."""

        def __init__(self, batch):
            self.batch = batch

        def create(
            self,
            *,
            model: str,
            input: Union[str, list[str]],
            dimensions: Optional[int] = None,
            encoding_format: Optional[str] = None,
            user: Optional[str] = None,
            **kwargs: Any
        ) -> str:
            """Add an embedding request to the batch.
            
            Args:
                model: The model to use for embeddings (e.g., text-embedding-3-small)
                input: The text to embed (either a string or a list of strings)
                dimensions: The number of dimensions to return. Only supported with 
                            text-embedding-3 models. Defaults to the model's max dimensions.
                encoding_format: The format to return the embeddings in (float or base64)
                user: A unique identifier for the end-user
                **kwargs: Additional provider-specific parameters
                
            Returns:
                str: The request ID (cache key) for this request
            """
            # Create the request body
            body = {
                "model": model,
                "input": input,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
                "user": user,
                **kwargs,
            }
            
            # Remove None values to match OpenAI's behavior
            body = {k: v for k, v in body.items() if v is not None}
            
            request_id = compute_request_hash({"type": "embedding", **body})
            order_id = self.batch._next_order_id
            self.batch._next_order_id += 1
            
            # Create custom_id from request_id and order_id
            custom_id = f"{request_id}#{order_id}"
            
            # Create the batch request directly in OpenAI Batch format
            batch_request = {
                "custom_id": custom_id,
                "url": "/v1/embeddings",
                "body": body
            }
            
            # Add to batch
            self.batch.requests.append(batch_request)
            
            return request_id
