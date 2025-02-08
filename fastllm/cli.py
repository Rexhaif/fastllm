"""Command-line interface for FastLLM."""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional, Union

import typer
from rich.console import Console
from rich.table import Table

from fastllm.core import LLMRequest, RequestBatch, RequestManager
from fastllm.providers.openai import OpenAIProvider, OpenAIRequest

app = typer.Typer(help="FastLLM - High-performance parallel LLM API request tool")
console = Console()

# Default values for command options
DEFAULT_CONCURRENCY = 100
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-3.5-turbo"


def parse_conversation(
    conv: Union[str, dict[str, Any], list[dict[str, Any]]],
    provider: str,
    model: Optional[str] = None,
) -> LLMRequest:
    """Parse a conversation into a request object."""
    if provider == "openai":
        if isinstance(conv, str):
            return OpenAIRequest.from_prompt(conv, model=model)
        elif isinstance(conv, dict):
            if "messages" in conv:
                return OpenAIRequest.from_dict(
                    {
                        **conv,
                        "model": model or conv.get("model", DEFAULT_MODEL),
                    }
                )
            else:
                return OpenAIRequest.from_dict(
                    {
                        "messages": [conv],
                        "model": model or DEFAULT_MODEL,
                    }
                )
        else:  # List of messages
            return OpenAIRequest.from_dict(
                {
                    "messages": conv,
                    "model": model or DEFAULT_MODEL,
                }
            )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def process_requests(
    *,  # Force keyword arguments
    prompt_file: Path,
    provider: str,
    model: Optional[str],
    concurrency: int,
    output_file: Optional[Path],
    api_key: str,
    json_format: bool,
) -> None:
    """Process requests from the input file."""
    requests: list[LLMRequest] = []

    if json_format:
        try:
            data = json.loads(prompt_file.read_text())
            conversations = data if isinstance(data, list) else [data]

            for conv in conversations:
                try:
                    requests.append(parse_conversation(conv, provider, model))
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Skipping invalid conversation: {e}"
                    )
                    continue

        except json.JSONDecodeError as err:
            console.print("[red]Error: Invalid JSON format in prompt file")
            raise typer.Exit(1) from err
    else:
        # Load single prompts from text file
        prompts = prompt_file.read_text().splitlines()
        for prompt in prompts:
            try:
                requests.append(parse_conversation(prompt, provider, model))
            except Exception as e:
                console.print(f"[yellow]Warning: Skipping invalid prompt: {e}")
                continue

    if not requests:
        console.print("[red]Error: No valid requests found in input file")
        raise typer.Exit(1)

    console.print(f"[green]Loaded {len(requests)} requests from {prompt_file}")

    # Create provider and request manager
    if provider == "openai":
        provider_impl = OpenAIProvider(
            api_key=api_key,
        )
    else:
        console.print(f"[red]Error: Unsupported provider: {provider}")
        raise typer.Exit(1)

    # Create request manager with provider
    manager = RequestManager(
        provider=provider_impl,
        concurrency=concurrency,
    )

    # Run requests
    responses = asyncio.run(manager.process_batch(requests))

    # Show results
    table = Table("Request ID", "Provider", "Content")
    for response in responses:
        if isinstance(response, Exception):
            console.print(f"[red]Error: {response}")
            continue
        table.add_row(
            response.request_id, response.provider, response.content[:100] + "..."
        )

    console.print(table)

    # Save results if requested
    if output_file:
        results = [
            r.model_dump() if not isinstance(r, Exception) else str(r)
            for r in responses
        ]
        output_file.write_text(json.dumps(results, indent=2))
        console.print(f"[green]Results saved to {output_file}")


@app.command()
def run(
    prompt_file: Path = typer.Argument(
        ...,
        help="Path to file containing prompts (one per line) or conversations (JSON)",
        exists=True,
    ),
    provider: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    output_file: Optional[Path] = None,
    api_key: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """Run parallel LLM requests from a prompt file."""
    if not api_key:
        console.print("[red]Error: API key not provided")
        raise typer.Exit(1)

    process_requests(
        prompt_file=prompt_file,
        provider=provider,
        model=model,
        concurrency=concurrency,
        output_file=output_file,
        api_key=api_key,
        json_format=json_format,
    )


if __name__ == "__main__":
    app()
