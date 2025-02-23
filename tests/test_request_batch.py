"""Tests for request batching functionality."""

import pytest

from fastllm.core import RequestBatch

# Constants for testing
EXPECTED_REQUESTS = 2
FIRST_REQUEST_ID = 0
SECOND_REQUEST_ID = 1

# Constants for batch addition testing
BATCH_SIZE_ONE = 1
BATCH_SIZE_TWO = 2
BATCH_SIZE_THREE = 3
FIRST_BATCH_START_ID = 0
SECOND_BATCH_START_ID = 1
THIRD_BATCH_START_ID = 2

# Constants for multiple additions testing
INITIAL_BATCH_SIZE = 1
FINAL_BATCH_SIZE = 3
FIRST_MULTIPLE_ID = 0
SECOND_MULTIPLE_ID = 1
THIRD_MULTIPLE_ID = 2


def test_request_batch():
    batch = RequestBatch()
    batch.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert len(batch.requests) == BATCH_SIZE_ONE


def test_request_batch_merge():
    """Test merging request batches."""
    batch1 = RequestBatch()
    batch1.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert len(batch1.requests) == BATCH_SIZE_ONE

    batch2 = RequestBatch()
    batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hey"}],
    )
    assert len(batch2.requests) == BATCH_SIZE_TWO

    # Test merging batches
    merged_batch = RequestBatch.merge([batch1, batch2])
    assert len(merged_batch.requests) == BATCH_SIZE_ONE + BATCH_SIZE_TWO


def test_request_batch_multiple_merges():
    """Test merging multiple request batches."""
    batch1 = RequestBatch()
    batch1.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert len(batch1.requests) == INITIAL_BATCH_SIZE

    batch2 = RequestBatch()
    batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    batch3 = RequestBatch()
    batch3.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hey"}],
    )

    # Test merging multiple batches
    final_batch = RequestBatch.merge([batch1, batch2, batch3])
    assert len(final_batch.requests) == 3
