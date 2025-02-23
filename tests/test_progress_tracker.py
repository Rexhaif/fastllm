import time

from fastllm.core import ProgressTracker

# Constants for testing
INITIAL_PROMPT_TOKENS = 10
INITIAL_COMPLETION_TOKENS = 20
INITIAL_TOTAL_TOKENS = INITIAL_PROMPT_TOKENS + INITIAL_COMPLETION_TOKENS

UPDATE_PROMPT_TOKENS = 5
UPDATE_COMPLETION_TOKENS = 5
FINAL_PROMPT_TOKENS = INITIAL_PROMPT_TOKENS + UPDATE_PROMPT_TOKENS
FINAL_COMPLETION_TOKENS = INITIAL_COMPLETION_TOKENS + UPDATE_COMPLETION_TOKENS
FINAL_TOTAL_TOKENS = FINAL_PROMPT_TOKENS + FINAL_COMPLETION_TOKENS

REQUESTS_COMPLETED = 2


def test_progress_tracker():
    tracker = ProgressTracker(total_requests=1, show_progress=False)

    # Initial update
    tracker.update(INITIAL_PROMPT_TOKENS, INITIAL_COMPLETION_TOKENS, False)

    # Assert that the token stats are updated
    assert tracker.stats.prompt_tokens == INITIAL_PROMPT_TOKENS
    assert tracker.stats.completion_tokens == INITIAL_COMPLETION_TOKENS
    assert tracker.stats.total_tokens == INITIAL_TOTAL_TOKENS

    # Test cache hit scenario
    tracker.update(UPDATE_PROMPT_TOKENS, UPDATE_COMPLETION_TOKENS, True)
    assert tracker.stats.prompt_tokens == FINAL_PROMPT_TOKENS
    assert tracker.stats.completion_tokens == FINAL_COMPLETION_TOKENS
    assert tracker.stats.total_tokens == FINAL_TOTAL_TOKENS

    # The cache hit count should be incremented by 1
    assert tracker.stats.cache_hits == 1

    # Test that requests_completed is incremented correctly
    # Initially, it was 0, then two updates
    assert tracker.stats.requests_completed == REQUESTS_COMPLETED


def test_progress_tracker_update():
    # Create a ProgressTracker with a fixed total_requests and disable progress display
    tracker = ProgressTracker(total_requests=5, show_progress=False)
    # Simulate 1 second elapsed
    tracker.stats.start_time = time.time() - 1

    # Update tracker with some token counts
    tracker.update(INITIAL_PROMPT_TOKENS, INITIAL_COMPLETION_TOKENS, False)

    # Assert that the token stats are updated
    assert tracker.stats.prompt_tokens == INITIAL_PROMPT_TOKENS
    assert tracker.stats.completion_tokens == INITIAL_COMPLETION_TOKENS
    assert tracker.stats.total_tokens == INITIAL_TOTAL_TOKENS

    # Test cache hit scenario
    tracker.update(UPDATE_PROMPT_TOKENS, UPDATE_COMPLETION_TOKENS, True)
    assert tracker.stats.prompt_tokens == FINAL_PROMPT_TOKENS
    assert tracker.stats.completion_tokens == FINAL_COMPLETION_TOKENS
    assert tracker.stats.total_tokens == FINAL_TOTAL_TOKENS
    # The cache hit count should be incremented by 1
    assert tracker.stats.cache_hits == 1

    # Test that requests_completed is incremented correctly
    # Initially, it was 0, then two updates
    assert tracker.stats.requests_completed == REQUESTS_COMPLETED
