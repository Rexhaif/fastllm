"""Tests for request batching functionality."""

import pytest
from fastllm.core import RequestBatch
from fastllm.cache import compute_request_hash

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
    """Test basic request batch functionality and request_id generation."""
    batch = RequestBatch()
    request_id = batch.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    
    # Verify request was added
    assert len(batch.requests) == 1
    
    # Verify request_id was properly set
    assert batch.requests[0]["_request_id"] == request_id
    
    # Verify order_id was properly set
    assert batch.requests[0]["_order_id"] == 0
    
    # Verify request_id is computed correctly
    # Include all fields that affect the hash
    expected_request = batch.requests[0].copy()
    expected_request.pop("_request_id", None)  # Remove request_id
    expected_request.pop("_order_id", None)  # Remove order_id
    assert request_id == compute_request_hash(expected_request)


def test_request_batch_merge():
    """Test merging request batches and request_id preservation."""
    # Create first batch
    batch1 = RequestBatch()
    request_id1 = batch1.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert len(batch1.requests) == 1
    assert batch1.requests[0]["_request_id"] == request_id1

    # Create second batch
    batch2 = RequestBatch()
    request_id2 = batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    request_id3 = batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hey"}],
    )
    assert len(batch2.requests) == 2
    assert batch2.requests[0]["_request_id"] == request_id2
    assert batch2.requests[1]["_request_id"] == request_id3

    # Test merging batches
    merged_batch = RequestBatch.merge([batch1, batch2])
    assert len(merged_batch.requests) == 3
    
    # Verify request_ids are preserved after merge
    assert merged_batch.requests[0]["_request_id"] == request_id1
    assert merged_batch.requests[1]["_request_id"] == request_id2
    assert merged_batch.requests[2]["_request_id"] == request_id3


def test_request_batch_multiple_merges():
    """Test merging multiple request batches and request_id preservation."""
    # Create first batch
    batch1 = RequestBatch()
    request_id1 = batch1.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert len(batch1.requests) == 1
    assert batch1.requests[0]["_request_id"] == request_id1

    # Create second batch
    batch2 = RequestBatch()
    request_id2 = batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert batch2.requests[0]["_request_id"] == request_id2

    # Create third batch
    batch3 = RequestBatch()
    request_id3 = batch3.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hey"}],
    )
    assert batch3.requests[0]["_request_id"] == request_id3

    # Test merging multiple batches
    final_batch = RequestBatch.merge([batch1, batch2, batch3])
    assert len(final_batch.requests) == 3
    
    # Verify request_ids are preserved after merge
    assert final_batch.requests[0]["_request_id"] == request_id1
    assert final_batch.requests[1]["_request_id"] == request_id2
    assert final_batch.requests[2]["_request_id"] == request_id3


def test_request_id_consistency():
    """Test that identical requests get the same request_id."""
    batch = RequestBatch()
    
    # Create two identical requests
    request_id1 = batch.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    
    # Create a new batch to avoid order_id interference
    batch2 = RequestBatch()
    request_id2 = batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    
    # Verify that identical requests get the same request_id
    assert request_id1 == request_id2
    
    # Create a different request
    request_id3 = batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Different"}],
    )
    
    # Verify that different requests get different request_ids
    assert request_id1 != request_id3


def test_request_id_with_none_values():
    """Test that None values are properly handled in request_id computation."""
    batch = RequestBatch()
    
    # Create request with some None values
    request_id = batch.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=None,  # Should be replaced with default
        top_p=None,  # Should be replaced with default
    )
    
    # Get the actual request that was created
    request1 = batch.requests[0].copy()
    request1.pop("_request_id", None)
    request1.pop("_order_id", None)
    
    # Create identical request without None values
    batch2 = RequestBatch()
    request_id2 = batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    
    # Get the second actual request
    request2 = batch2.requests[0].copy()
    request2.pop("_request_id", None)
    request2.pop("_order_id", None)
    
    # Both requests should have the same content
    assert request1 == request2
    
    # Both request IDs should be identical since None values are replaced with defaults
    assert request_id == request_id2
