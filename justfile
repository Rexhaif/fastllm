set shell := ["bash", "-c"]

# List all available commands
default:
    @just --list

# Install all dependencies using uv
install:
    uv venv
    uv pip install -e .
    uv pip install -e ".[dev]"

# Run tests
test:
    uv run pytest tests/ -v --cov=fastllm

# Format code using ruff
format:
    uv run ruff format .
    uv run ruff check . --fix

# Run linting checks
lint:
    uv run ruff check .

# Clean up cache files
clean:
    rm -rf .pytest_cache
    rm -rf .coverage
    rm -rf .ruff_cache
    rm -rf dist
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +

live_test_completions:
    uv run python examples/parallel_test.py --model meta-llama/llama-3.2-3b-instruct --repeats 100 --concurrency 75 --cache-type memory --output NO_OUTPUT
