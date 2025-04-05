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
    
    # Verify OpenAI Batch format
    assert "custom_id" in batch.requests[0]
    assert "url" in batch.requests[0]
    assert "body" in batch.requests[0]
    
    # Verify custom_id format and extract request_id and order_id
    custom_id_parts = batch.requests[0]["custom_id"].split("#")
    assert len(custom_id_parts) == 2
    extracted_request_id, order_id_str = custom_id_parts
    assert extracted_request_id == request_id
    assert order_id_str == "0"
    
    # Verify URL indicates chat completion
    assert batch.requests[0]["url"] == "/v1/chat/completions"
    
    # Verify request_id is computed correctly
    # Include all fields that affect the hash
    expected_request = {"type": "chat_completion", **batch.requests[0]["body"]}
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
    assert batch1.requests[0]["custom_id"].split("#")[0] == request_id1

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
    assert batch2.requests[0]["custom_id"].split("#")[0] == request_id2
    assert batch2.requests[1]["custom_id"].split("#")[0] == request_id3

    # Test merging batches
    merged_batch = RequestBatch.merge([batch1, batch2])
    assert len(merged_batch.requests) == 3
    
    # Verify request_ids are preserved after merge
    assert merged_batch.requests[0]["custom_id"].split("#")[0] == request_id1
    assert merged_batch.requests[1]["custom_id"].split("#")[0] == request_id2
    assert merged_batch.requests[2]["custom_id"].split("#")[0] == request_id3


def test_request_batch_multiple_merges():
    """Test merging multiple request batches and request_id preservation."""
    # Create first batch
    batch1 = RequestBatch()
    request_id1 = batch1.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert len(batch1.requests) == 1
    assert batch1.requests[0]["custom_id"].split("#")[0] == request_id1

    # Create second batch
    batch2 = RequestBatch()
    request_id2 = batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert batch2.requests[0]["custom_id"].split("#")[0] == request_id2

    # Create third batch
    batch3 = RequestBatch()
    request_id3 = batch3.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hey"}],
    )
    assert batch3.requests[0]["custom_id"].split("#")[0] == request_id3

    # Test merging multiple batches
    final_batch = RequestBatch.merge([batch1, batch2, batch3])
    assert len(final_batch.requests) == 3
    
    # Verify request_ids are preserved after merge
    assert final_batch.requests[0]["custom_id"].split("#")[0] == request_id1
    assert final_batch.requests[1]["custom_id"].split("#")[0] == request_id2
    assert final_batch.requests[2]["custom_id"].split("#")[0] == request_id3


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
    
    # Get the actual request body that was created
    body1 = batch.requests[0]["body"]
    
    # Create identical request without None values
    batch2 = RequestBatch()
    request_id2 = batch2.chat.completions.create(
        model="dummy-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    
    # Get the second actual request body
    body2 = batch2.requests[0]["body"]
    
    # Both requests should have the same content
    assert body1 == body2
    
    # Both request IDs should be identical since None values are replaced with defaults
    assert request_id == request_id2


def test_embeddings_request_format():
    """Test the format of embeddings requests."""
    batch = RequestBatch()
    request_id = batch.embeddings.create(
        model="text-embedding-ada-002",
        input="Hello world",
    )
    
    # Verify request was added
    assert len(batch.requests) == 1
    
    # Verify OpenAI Batch format
    assert "custom_id" in batch.requests[0]
    assert batch.requests[0]["url"] == "/v1/embeddings"
    assert "body" in batch.requests[0]
    
    # Verify custom_id format and extract request_id and order_id
    custom_id_parts = batch.requests[0]["custom_id"].split("#")
    assert len(custom_id_parts) == 2
    extracted_request_id, order_id_str = custom_id_parts
    assert extracted_request_id == request_id
    assert order_id_str == "0"
    
    # Verify body content
    assert batch.requests[0]["body"]["model"] == "text-embedding-ada-002"
    assert batch.requests[0]["body"]["input"] == "Hello world"
