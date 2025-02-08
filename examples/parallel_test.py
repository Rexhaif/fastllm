"""Test script for parallel request handling."""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import typer
from openai.types.chat import ChatCompletion
from rich.console import Console
from rich.panel import Panel

from fastllm.core import RequestBatch, RequestManager, ResponseWrapper
from fastllm.providers.openai import OpenAIProvider

AVAILABLE_MODELS = [
    "mistralai/mistral-small-24b-instruct-2501",
    "meta-llama/llama-3.2-3b-instruct",
]

# Default values for command options
DEFAULT_MODEL = AVAILABLE_MODELS[0]
DEFAULT_REPEATS = 10
DEFAULT_CONCURRENCY = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_OUTPUT = Path("results.json")

app = typer.Typer()


def process_response(
    response: Union[ResponseWrapper[ChatCompletion], Exception], index: int
) -> dict[str, Any]:
    """Process a response into a serializable format."""
    return {
        "index": index,
        **(
            {"type": "error", "error": str(response)}
            if isinstance(response, Exception)
            else {
                "type": "success",
                "request_id": response.request_id,
                "raw_response": response.response.model_dump(),
            }
        ),
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
                max_tokens=max_tokens,
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

    # Create provider and request manager
    provider = OpenAIProvider(
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )

    # Create request manager with provider
    manager = RequestManager(
        provider=provider,
        concurrency=concurrency,
        show_progress=not no_progress,
    )

    # Process batch
    responses = manager.process_batch(batch)

    # Show results
    successful = 0
    failed = 0

    # Process results while maintaining order
    results_data = []
    for i, response in enumerate(responses):
        result = process_response(response, i)
        if result["type"] == "error":
            failed += 1
        else:
            successful += 1
        results_data.append(result)

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Successful: [green]{successful}[/green]",
                    f"Failed: [red]{failed}[/red]",
                    f"Total: {len(responses)} (matches {len(batch)} requests)",
                ]
            ),
            title="[bold green]Results",
        )
    )

    # Save results
    output.write_text(
        json.dumps(
            {
                "config": {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "repeats": repeats,
                    "concurrency": concurrency,
                },
                "results": results_data,
                "summary": {
                    "successful": successful,
                    "failed": failed,
                    "total": len(responses),
                },
            },
            indent=2,
        )
    )


@app.command()
def main(
    model: str = typer.Option(
        DEFAULT_MODEL,
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
) -> None:
    """Run parallel request test."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("Error: OPENROUTER_API_KEY not set")
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
    )


if __name__ == "__main__":
    app()
