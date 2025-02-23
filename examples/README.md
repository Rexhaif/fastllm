# FastLLM Examples

This directory contains example notebooks and scripts demonstrating various use cases of FastLLM.

## Setup

1. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

2. Create a `.env` file in this directory with your API keys:
```env
OPENAI_API_KEY=your_openai_key_here  # If using OpenAI
OPENROUTER_API_KEY=your_openrouter_key_here  # If using OpenRouter
```

## Examples

### OpenRouter Example (`openrouter_example.ipynb`)
Demonstrates how to use FastLLM with OpenRouter to access Llama models:
- Setting up environment variables
- Creating and processing simple text prompts
- Creating and processing JSONL requests
- Using both CLI and direct API approaches

### Parallel Processing Test (`parallel_test.py`)
Shows how to use FastLLM for parallel processing of multiple requests:
- Batch processing configuration
- Concurrent request handling
- Progress tracking
- Error handling

### Notebook Integration Test (`notebook_test.ipynb`)
Demonstrates FastLLM integration in Jupyter notebooks:
- Interactive usage
- Progress bars
- Result visualization

## Notes

- Make sure to handle your API keys securely
- Adjust concurrency settings based on your API rate limits
- Check the provider's documentation for available models and their capabilities
- Consider using caching for development to avoid unnecessary API calls 