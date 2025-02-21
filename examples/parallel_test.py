"""Test script for parallel request handling."""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union, Literal
import asyncio

import typer
from openai.types.chat import ChatCompletion
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

from fastllm.core import RequestBatch, RequestManager, ResponseWrapper
from fastllm.providers.openai import OpenAIProvider
from fastllm.cache import InMemoryCache, DiskCache

# Default values for command options
DEFAULT_REPEATS = 10
DEFAULT_CONCURRENCY = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_OUTPUT = Path("results.json")

app = typer.Typer()

load_dotenv()


def process_response(
    response: ResponseWrapper[ChatCompletion], index: int
) -> dict[str, Any]:
    """Process a response into a serializable format."""
    return {
        "index": index,
        "type": "success",
        "request_id": response.request_id,
        "raw_response": response.response.model_dump(),
    }


def run_test(
    *,  # Force keyword arguments
    api_key: str,
    model: str,
    repeats: int,
    concurrency: int,
    output: Path,
    temperature: float,
    max_tokens: Optional[int],
    no_progress: bool = False,
    cache_type: Literal["memory", "disk"] = "memory",
    cache_ttl: Optional[int] = None,
) -> None:
    """Run the test with given parameters."""
    console = Console()

    # Create batch of requests using OpenAI-style API
    with RequestBatch() as batch:
        # Add single prompt requests
        for i in range(repeats):
            batch.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Print only number, number is {i}. Do not include any other text.",
                    }
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                include_reasoning=True,
            )

    # Show configuration
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Model: {model}",
                    f"Temperature: {temperature}",
                    f"Max Tokens: {max_tokens or 'default'}",
                    f"Requests: {len(batch)}",
                    f"Concurrency: {concurrency}",
                ]
            ),
            title="[bold blue]Test Configuration",
        )
    )

    # Create cache provider based on type
    if cache_type == "memory":
        cache_provider = InMemoryCache()
    else:
        cache_provider = DiskCache(
            directory="./cache",
            ttl=cache_ttl,
            size_limit=int(2e9),  # 2GB size limit
        )

    provider = OpenAIProvider(
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    manager = RequestManager(
        provider=provider,
        concurrency=concurrency,
        show_progress=not no_progress,
        caching_provider=cache_provider,
    )

    try:
        # First run: Process batch
        responses_first = manager.process_batch(batch)
        successful_first = 0
        results_data_first = []
        for i, response in enumerate(responses_first):
            result = process_response(response, i)
            successful_first += 1
            results_data_first.append(result)

        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"First Run - Successful: [green]{successful_first}[/green]",
                        f"Total: {len(responses_first)} (matches {len(batch)} requests)",
                    ]
                ),
                title="[bold green]Results - First Run",
            )
        )

        # Second run: Process the same batch, expecting cached results
        responses_second = manager.process_batch(batch)
        successful_second = 0
        results_data_second = []
        for i, response in enumerate(responses_second):
            result = process_response(response, i)
            successful_second += 1
            results_data_second.append(result)
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Second Run - Successful: [green]{successful_second}[/green]",
                        f"Total: {len(responses_second)} (matches {len(batch)} requests)",
                    ]
                ),
                title="[bold green]Results - Second Run (Cached)",
            )
        )

        # Save results from both runs
        output.write_text(
            json.dumps(
                {
                    "config": {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "repeats": repeats,
                        "concurrency": concurrency,
                        "cache_type": cache_type,
                        "cache_ttl": cache_ttl,
                    },
                    "first_run_results": results_data_first,
                    "second_run_results": results_data_second,
                    "first_run_summary": {
                        "successful": successful_first,
                        "total": len(responses_first),
                    },
                    "second_run_summary": {
                        "successful": successful_second,
                        "total": len(responses_second),
                    },
                },
                indent=2,
            )
        )
    finally:
        # Clean up disk cache if used
        if cache_type == "disk":
            # Run close in asyncio event loop
            asyncio.run(cache_provider.close())


@app.command()
def main(
    model: str = typer.Option(
        "meta-llama/llama-3.2-3b-instruct",
        "--model",
        "-m",
        help="Model to use",
    ),
    repeats: int = typer.Option(
        DEFAULT_REPEATS,
        "--repeats",
        "-n",
        help="Number of repeats",
    ),
    concurrency: int = typer.Option(
        DEFAULT_CONCURRENCY,
        "--concurrency",
        "-c",
        help="Concurrent requests",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Output file",
    ),
    temperature: float = typer.Option(
        DEFAULT_TEMPERATURE,
        "--temperature",
        "-t",
        help="Temperature for generation",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable progress tracking",
    ),
    cache_type: str = typer.Option(
        "memory",
        "--cache-type",
        help="Cache type to use (memory or disk)",
    ),
    cache_ttl: Optional[int] = typer.Option(
        None,
        "--cache-ttl",
        help="Time to live in seconds for cached items (disk cache only)",
    ),
) -> None:
    """Run parallel request test."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.echo("Error: OPENAI_API_KEY not set")
        raise typer.Exit(1)

    run_test(
        api_key=api_key,
        model=model,
        repeats=repeats,
        concurrency=concurrency,
        output=output,
        temperature=temperature,
        max_tokens=max_tokens,
        no_progress=no_progress,
        cache_type=cache_type,
        cache_ttl=cache_ttl,
    )


if __name__ == "__main__":
    app()
