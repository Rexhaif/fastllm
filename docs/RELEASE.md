# Release Procedure

This document outlines the steps to release fastllm-kit to PyPI.

## Publishing to PyPI

The library is published as `fastllm-kit` on PyPI, but maintains the import name `fastllm` for user convenience.

### Prerequisites

- Ensure you have `uv` installed
- Have PyPI credentials set up
  - Set the PyPI token with `UV_PUBLISH_TOKEN` environment variable or use `--token` flag
  - For TestPyPI, you'll also need TestPyPI credentials

### Release Steps

1. Update the version in `pyproject.toml`
2. Update `CHANGELOG.md` with the latest changes
3. Commit all changes

4. Clean previous builds:
   ```
   just clean
   ```

5. Build the package:
   ```
   just build
   ```

6. Publish to TestPyPI first to verify everything works:
   ```
   just publish-test
   ```
   Or with explicit token:
   ```
   UV_PUBLISH_TOKEN=your_testpypi_token just publish-test
   ```

7. Verify installation from TestPyPI:
   ```
   pip uninstall -y fastllm-kit
   pip install --index-url https://test.pypi.org/simple/ fastllm-kit
   ```

8. Verify imports work correctly:
   ```python
   from fastllm import RequestBatch, RequestManager
   ```

9. If everything works, publish to PyPI:
   ```
   just publish
   ```
   Or with explicit token:
   ```
   UV_PUBLISH_TOKEN=your_pypi_token just publish
   ```

## Usage Instructions

Users will install the package with:
```
pip install fastllm-kit
```

But will import it in their code as:
```python
from fastllm import RequestBatch, RequestManager
```

This allows us to maintain a clean import interface despite the PyPI name being different from the import name. 