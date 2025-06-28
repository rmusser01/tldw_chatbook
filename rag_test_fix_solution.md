# RAG Test Fix Solution

## Problem Summary
- 44 tests are failing because they expect 2-dimensional mock embeddings but get 384-dimensional real embeddings
- The EmbeddingsService auto-initializes a real SentenceTransformer provider
- Test mocks are not preventing this initialization

## Recommended Fix

### Option 1: Patch the Provider Creation (Best Solution)

Add this to the test file's imports and fixtures:

```python
# At the top of test files
from unittest.mock import patch, MagicMock

# Create a fixture that properly mocks the service
@pytest.fixture
def mock_embeddings_service(temp_dir):
    with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.SentenceTransformerProvider'):
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.EMBEDDINGS_AVAILABLE', False):
            service = EmbeddingsService(temp_dir)
            
            # Create mock provider
            mock_provider = MagicMock()
            mock_provider.dimension = 2
            mock_provider.create_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
            
            # Set as the current provider
            service.providers['mock'] = mock_provider
            service.current_provider_id = 'mock'
            service.embedding_model = mock_provider
            
            return service
```

### Option 2: Update All Tests to Accept Real Embeddings

Change all occurrences of:
```python
assert len(embedding) == 2
```

To:
```python
assert len(embedding) == 384
```

But this requires updating 44 tests.

### Option 3: Add Environment Variable to Disable Auto-Init

Set an environment variable in tests:
```python
import os
os.environ['TLDW_TEST_MODE'] = '1'
```

Then modify the service to check this before auto-initializing.

## Quick Workaround

For immediate testing, you can run tests with mocked sentence-transformers:

```bash
# Create a conftest.py in Tests/RAG/ with:
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    with patch('sentence_transformers.SentenceTransformer') as mock:
        model = MagicMock()
        model.encode.return_value = [[0.1, 0.2]] * 10  # Return 2D embeddings
        model.get_sentence_embedding_dimension.return_value = 2
        mock.return_value = model
        yield mock
```

## Root Cause Fix

The real fix would be to modify the EmbeddingsService to not auto-initialize providers when created without config, or to have a test mode flag.