set shell := ["bash", "-c"]

# List all available commands
default:
    @just --list

# Install all dependencies using poetry and uv/pip
install:
    poetry install
    if command -v uv >/dev/null 2>&1; then \
        poetry run uv pip install --upgrade pip; \
    else \
        poetry run pip install --upgrade pip; \
    fi

# Run tests
test:
    poetry run pytest tests/ -v --cov=fastllm

# Format code using ruff
format:
    poetry run ruff format .
    poetry run ruff check . --fix

# Run linting checks
lint:
    poetry run ruff check .

# Clean up cache files
clean:
    rm -rf .pytest_cache
    rm -rf .coverage
    rm -rf .ruff_cache
    rm -rf dist
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +

# Run the CLI tool
run *ARGS:
    poetry run fastllm {{ARGS}}


# Run parallel request test with OpenRouter
openrouter-test repeats="20" cache_type="disk" concurrency="50" model="meta-llama/llama-3.2-3b-instruct" :
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" poetry run python examples/parallel_test.py \
        --model {{model}} \
        --repeats {{repeats}} \
        --concurrency {{concurrency}} \
        --cache-type {{cache_type}}