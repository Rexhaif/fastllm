#!/usr/bin/env python
"""Test script for embeddings API with parallel processing and caching."""

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Union, Literal
import asyncio
import numpy as np

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

from fastllm import (
    RequestBatch, RequestManager,
    ResponseWrapper, InMemoryCache, DiskCache,
    OpenAIProvider
)

# Default values for command options
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 384
DEFAULT_CONCURRENCY = 10
DEFAULT_OUTPUT = "embedding_results.json"

app = typer.Typer()

load_dotenv()

# Sample texts for embedding
DEFAULT_TEXTS = [
    f"The quick brown fox jumps over the lazy dog {i}" for i in range(100)
]


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def normalize_l2(x):
    """L2 normalize a vector or batch of vectors."""
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def process_response(
    response: ResponseWrapper, index: int
) -> dict[str, Any]:
    """Process an embedding response into a serializable format."""
    if not response.is_embedding_response:
        return {
            "index": index,
            "type": "error",
            "error": "Not an embedding response"
        }
    
    return {
        "index": index,
        "type": "success",
        "request_id": response.request_id,
        "model": response.response.get("model"),
        "usage": response.response.get("usage"),
        "embedding_count": len(response.embeddings),
        "embedding_dimensions": [len(emb) for emb in response.embeddings],
    }


def create_embedding_batch(
    texts: List[str], 
    model: str, 
    dimensions: Optional[int] = None,
    encoding_format: Optional[str] = None
) -> RequestBatch:
    """Create a batch of embedding requests."""
    batch = RequestBatch()
    
    # Add individual text embedding requests
    for text in texts:
        batch.embeddings.create(
            model=model,
            input=text,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )
    
    # Add a batch request with multiple texts (last 3 texts)
    if len(texts) >= 3:
        batch.embeddings.create(
            model=model,
            input=texts[-3:],
            dimensions=dimensions,
            encoding_format=encoding_format,
        )
    
    return batch


def run_test(
    *,  # Force keyword arguments
    api_key: str,
    model: str,
    texts: List[str],
    dimensions: Optional[int],
    encoding_format: Optional[str],
    concurrency: int,
    output: Path | str,
    no_progress: bool = False,
    cache_type: Literal["memory", "disk"] = "memory",
    cache_ttl: Optional[int] = None,
) -> None:
    """Run the embedding test with given parameters."""
    console = Console()

    # Create batch of requests
    batch = create_embedding_batch(texts, model, dimensions, encoding_format)

    # Show configuration
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Model: {model}",
                    f"Dimensions: {dimensions or 'default'}",
                    f"Format: {encoding_format or 'default'}",
                    f"Texts: {len(texts)}",
                    f"Requests: {len(batch)}",
                    f"Concurrency: {concurrency}",
                ]
            ),
            title="[bold blue]Embedding Test Configuration",
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
    )
    manager = RequestManager(
        provider=provider,
        concurrency=concurrency,
        show_progress=not no_progress,
        caching_provider=cache_provider,
    )

    try:
        # First run: Process batch
        console.print("[bold blue]Starting first run (no cache)...[/bold blue]")
        responses_first = manager.process_batch(batch)
        
        # Extract and process results
        all_embeddings = []
        results_data_first = []
        
        for i, response in enumerate(responses_first):
            result = process_response(response, i)
            results_data_first.append(result)
            
            if response.is_embedding_response:
                embeddings = response.embeddings
                for embedding in embeddings:
                    all_embeddings.append(embedding)
        
        # Summary table for first run
        console.print("\n[bold green]First Run Results:[/bold green]")
        table = Table(show_header=True)
        table.add_column("Request #")
        table.add_column("Model")
        table.add_column("Embeddings")
        table.add_column("Dimensions")
        table.add_column("Tokens")
        
        for i, result in enumerate(results_data_first):
            if result["type"] == "success":
                dims = ", ".join([str(d) for d in result["embedding_dimensions"]])
                table.add_row(
                    f"{i+1}",
                    result["model"],
                    str(result["embedding_count"]),
                    dims,
                    str(result["usage"]["prompt_tokens"])
                )
        
        console.print(table)
        
        # Compare some embeddings
        if len(all_embeddings) >= 2:
            console.print("\n[bold blue]Embedding Similarities:[/bold blue]")
            similarity_table = Table(show_header=True)
            similarity_table.add_column("Embedding Pair")
            similarity_table.add_column("Cosine Similarity")
            
            # Compare a few pairs (not all combinations to keep output manageable)
            num_comparisons = min(5, len(all_embeddings) * (len(all_embeddings) - 1) // 2)
            compared = set()
            comparison_count = 0
            
            for i in range(len(all_embeddings)):
                for j in range(i+1, len(all_embeddings)):
                    if comparison_count >= num_comparisons:
                        break
                    if (i, j) not in compared:
                        sim = cosine_similarity(all_embeddings[i], all_embeddings[j])
                        similarity_table.add_row(f"{i+1} and {j+1}", f"{sim:.4f}")
                        compared.add((i, j))
                        comparison_count += 1
            
            console.print(similarity_table)

        # Second run: Process the same batch, expecting cached results
        console.print("\n[bold blue]Starting second run (cached)...[/bold blue]")
        responses_second = manager.process_batch(batch)
        results_data_second = []
        
        for i, response in enumerate(responses_second):
            result = process_response(response, i)
            results_data_second.append(result)

        # Compare cache performance
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"First Run - Successful: [green]{sum(1 for r in results_data_first if r['type'] == 'success')}[/green]",
                        f"Second Run - Successful: [green]{sum(1 for r in results_data_second if r['type'] == 'success')}[/green]",
                        f"Total Requests: {len(batch)}",
                    ]
                ),
                title="[bold green]Cache Performance",
            )
        )

        # Save results from both runs
        if output != "NO_OUTPUT":
            output_path = Path(output)
            output_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "model": model,
                            "dimensions": dimensions,
                            "encoding_format": encoding_format,
                            "concurrency": concurrency,
                            "cache_type": cache_type,
                            "cache_ttl": cache_ttl,
                        },
                        "first_run_results": results_data_first,
                        "second_run_results": results_data_second,
                    },
                    indent=2,
                )
            )
            console.print(f"\nResults saved to [bold]{output_path}[/bold]")
            
    finally:
        # Clean up disk cache if used
        if cache_type == "disk":
            # Run close in asyncio event loop
            asyncio.run(cache_provider.close())


@app.command()
def main(
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Embedding model to use",
    ),
    dimensions: Optional[int] = typer.Option(
        DEFAULT_DIMENSIONS,
        "--dimensions",
        "-d",
        help="Dimensions for the embeddings (None uses model default)",
    ),
    encoding_format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Encoding format (float or base64)",
    ),
    concurrency: int = typer.Option(
        DEFAULT_CONCURRENCY,
        "--concurrency",
        "-c",
        help="Concurrent requests",
    ),
    output: str = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Output file",
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
    input_file: Optional[str] = typer.Option(
        None,
        "--input-file",
        "-i",
        help="File with texts to embed (one per line)",
    ),
) -> None:
    """Run embeddings test with caching."""
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Load texts from file if provided, otherwise use default texts
    texts = DEFAULT_TEXTS
    if input_file:
        try:
            with open(input_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            typer.echo(f"Error reading input file: {e}")
            raise typer.Exit(code=1)

    run_test(
        api_key=api_key,
        model=model,
        texts=texts,
        dimensions=dimensions,
        encoding_format=encoding_format,
        concurrency=concurrency,
        output=output,
        no_progress=no_progress,
        cache_type=cache_type,
        cache_ttl=cache_ttl,
    )


if __name__ == "__main__":
    app() 