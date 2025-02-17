# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-03

### Added
- Initial release with core functionality
- Parallel request processing with configurable concurrency
- In-memory and disk-based caching support
- Multiple provider support (OpenAI, OpenRouter)
- Request batching with OpenAI-style API
- Progress tracking and statistics
- CLI interface for file-based processing
- Request deduplication and response ordering
- Configurable retry mechanism
- Rich progress bar with detailed statistics
- Support for existing asyncio event loops
- Jupyter notebook compatibility

### Changed
- Optimized request processing for better performance
- Improved error handling and reporting
- Enhanced request ID handling
- Added compatibility with existing asyncio event loops
- Fixed asyncio loop handling in Jupyter notebooks

### Fixed
- Request ID validation and string conversion
- Cache persistence issues
- Response ordering in parallel processing 