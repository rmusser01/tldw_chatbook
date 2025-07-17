# Embeddings Pipeline Usage Guide

This guide provides comprehensive documentation for using the embedding pipeline in TLDW Chatbook, including model management, configuration, and usage.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Model Management](#model-management)
5. [Using Embeddings](#using-embeddings)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

## Overview

The TLDW Chatbook embedding system provides a flexible and powerful way to create vector embeddings for your content. It supports multiple embedding providers including HuggingFace and OpenAI, with features like:

- **Multiple Model Support**: Use various pre-configured models or add your own
- **Custom Cache Management**: Control where models are stored
- **Automatic Model Downloading**: Download HuggingFace models on-demand
- **Memory Management**: Load/unload models as needed
- **Batch Processing**: Efficient embedding generation for large datasets

## Installation

### Basic Installation
The core TLDW Chatbook installation includes basic embedding support:
```bash
pip install tldw_chatbook
```

### Full Embedding Support
For complete embedding functionality including RAG support:
```bash
pip install tldw_chatbook[embeddings_rag]
```

This installs additional dependencies:
- `transformers` - HuggingFace model support
- `torch` - PyTorch for model inference
- `numpy` - Numerical operations
- `huggingface_hub` - Model downloading
- `chromadb` - Vector database support

## Configuration

### Basic Configuration

The embedding system is configured in your `config.toml` file:

```toml
[embedding_config]
# Default model to use when creating embeddings
default_model_id = "e5-small-v2"

# Where to store downloaded HuggingFace models
model_cache_dir = "~/.local/share/tldw_cli/models/embeddings"

# Automatically download models when needed
auto_download = true

# Maximum cache size in GB
cache_size_limit_gb = 10.0

# Default LLM for contextualization (if using RAG)
default_llm_for_contextualization = "gpt-3.5-turbo"
```

### Pre-configured Models

The system comes with several pre-configured models:

#### HuggingFace Models

```toml
[embedding_config.models.e5-small-v2]
provider = "huggingface"
model_name_or_path = "intfloat/e5-small-v2"
dimension = 384
trust_remote_code = false
max_length = 512

[embedding_config.models.e5-base-v2]
provider = "huggingface"
model_name_or_path = "intfloat/e5-base-v2"
dimension = 768
trust_remote_code = false
max_length = 512

[embedding_config.models.all-MiniLM-L6-v2]
provider = "huggingface"
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
trust_remote_code = false
max_length = 256
```

#### OpenAI Models

```toml
[embedding_config.models.openai-ada-002]
provider = "openai"
model_name_or_path = "text-embedding-ada-002"
dimension = 1536
api_key = "YOUR_OPENAI_API_KEY"  # Or use environment variable

[embedding_config.models.openai-text-embedding-3-small]
provider = "openai"
model_name_or_path = "text-embedding-3-small"
dimension = 1536
api_key = "YOUR_OPENAI_API_KEY"
```

### Adding Custom Models

To add a new model, add its configuration to your `config.toml`:

```toml
[embedding_config.models.my-custom-model]
provider = "huggingface"
model_name_or_path = "organization/model-name"
dimension = 768  # Must match the model's output dimension
trust_remote_code = false  # Set to true if model requires custom code
max_length = 512  # Maximum input length
batch_size = 32  # Batch size for processing
device = "cuda"  # Or "cpu", "mps" for Apple Silicon
```

## Model Management

### Downloading Models

Models can be downloaded in several ways:

#### 1. Automatic Download
When `auto_download = true` in config, models are downloaded automatically when first used.

#### 2. Via Management UI
1. Go to the Embeddings tab
2. Click "Manage" to open the management window
3. Select a model from the list
4. Click "Download" to download the model

#### 3. Manual Pre-download
You can pre-download models using Python:

```python
from huggingface_hub import snapshot_download

# Download to specific directory
snapshot_download(
    repo_id="intfloat/e5-small-v2",
    cache_dir="~/.local/share/tldw_cli/models/embeddings"
)
```

### Model Storage Locations

#### Default Locations
- **HuggingFace Models**: `~/.cache/huggingface/hub/` (default)
- **Custom Cache**: Set via `model_cache_dir` in config
- **Per-Model Cache**: Can be set individually per model

#### Storage Structure
```
~/.local/share/tldw_cli/models/embeddings/
├── models--intfloat--e5-small-v2/
│   ├── blobs/
│   ├── refs/
│   └── snapshots/
├── models--sentence-transformers--all-MiniLM-L6-v2/
│   └── ...
└── models--BAAI--bge-small-en-v1.5/
    └── ...
```

### Memory Management

#### Loading Models
Models are loaded into memory when first used. The system uses an LRU cache to manage memory:

```python
# Models are automatically loaded when creating embeddings
embeddings = embedding_factory.embed(texts, model_id="e5-small-v2")
```

#### Unloading Models
To free memory, you can unload models:
1. Via UI: Select model → Click "Unload Model"
2. The LRU cache automatically evicts least-recently-used models when cache is full

#### Cache Size
Default cache size is 2 models. This can be configured when initializing the factory.

## Using Embeddings

### Creating Embeddings via UI

1. **Navigate to Embeddings Tab**
   - Click the Embeddings tab in the main interface

2. **Select Input Source**
   - **File Input**: For text files, documents, or custom content
   - **Database**: For existing content in your databases

3. **Configure Settings**
   - **Model**: Select from available embedding models
   - **Chunking**: Configure how text is split
     - Method: Character, word, sentence, or semantic
     - Size: Tokens per chunk (e.g., 512)
     - Overlap: Token overlap between chunks (e.g., 128)

4. **Generate Embeddings**
   - Click "Create Embeddings" to process your content
   - Monitor progress in the status area

### Creating Embeddings Programmatically

```python
from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory

# Initialize factory with config
config = {
    "default_model_id": "e5-small-v2",
    "models": {
        "e5-small-v2": {
            "provider": "huggingface",
            "model_name_or_path": "intfloat/e5-small-v2",
            "dimension": 384,
            "cache_dir": "~/.local/share/tldw_cli/models/embeddings"
        }
    }
}

factory = EmbeddingFactory(config, max_cached=2)

# Create embeddings
texts = ["Hello world", "This is a test"]
embeddings = factory.embed(texts, model_id="e5-small-v2")

# Async version
embeddings = await factory.async_embed(texts, model_id="e5-small-v2")
```

### Batch Processing

For large datasets, use batch processing:

```python
# Process in batches
batch_size = 100
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    batch_embeddings = factory.embed(batch, model_id="e5-small-v2")
    all_embeddings.extend(batch_embeddings)
```

## Advanced Features

### Custom Pooling Strategies

The system supports custom pooling for HuggingFace models:

```python
def custom_pooling(hidden_states, attention_mask):
    # Your custom pooling logic
    return pooled_output

# In config
model_config["pooling"] = custom_pooling
```

### Multi-GPU Support

For HuggingFace models, specify device in config:

```toml
[embedding_config.models.large-model]
provider = "huggingface"
model_name_or_path = "model-name"
device = "cuda:0"  # Or "cuda:1" for specific GPU
```

### Contextualized Embeddings

For RAG applications, you can add context to chunks:

```toml
[rag.contextualize_chunks]
enabled = true
llm_model = "gpt-3.5-turbo"
chunk_size = 512
overlap = 128
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
- **Solution**: Reduce batch size or use smaller models
- **Config**: Lower `batch_size` in model config
- **UI**: Unload unused models via Management window

#### 2. Model Download Failures
- **Check**: Internet connection and HuggingFace availability
- **Solution**: Use manual download or different mirror
- **Config**: Set custom `cache_dir` with write permissions

#### 3. Slow Performance
- **CPU vs GPU**: Ensure CUDA is available for GPU models
- **Batch Size**: Increase batch size for better throughput
- **Model Size**: Use smaller models for faster inference

#### 4. Dimension Mismatch
- **Cause**: Incorrect dimension in config
- **Solution**: Verify model output dimension matches config
- **Test**: Use the test embedding feature in Management UI

### Checking Model Status

Via Management UI:
1. Go to Embeddings → Manage
2. Select a model
3. Check:
   - Download Status
   - Memory Status
   - Cache Location
   - Model Size

### Debug Logging

Enable debug logging in config:

```toml
[logging]
level = "DEBUG"
```

Check logs at: `~/.local/share/tldw_cli/logs/`

## Model Recommendations

### For General Use
- **e5-small-v2**: Good balance of size and performance (384 dims)
- **all-MiniLM-L6-v2**: Fast and efficient (384 dims)

### For High Quality
- **e5-large-v2**: Best quality, larger size (1024 dims)
- **openai-text-embedding-3-small**: Commercial option (1536 dims)

### For Multilingual
- **multilingual-e5-large-instruct**: Supports 100+ languages (1024 dims)

### For Domain-Specific
- **bge-base-en-v1.5**: Good for retrieval tasks (768 dims)
- **gte-small**: Efficient general text embeddings (384 dims)

## Best Practices

1. **Model Selection**
   - Choose based on your use case (speed vs quality)
   - Consider dimension size for storage requirements
   - Test multiple models for your specific content

2. **Chunking Strategy**
   - Use semantic chunking for better context preservation
   - Adjust chunk size based on model max length
   - Include overlap to maintain context between chunks

3. **Performance Optimization**
   - Pre-download models during setup
   - Use GPU when available
   - Batch process large datasets
   - Monitor memory usage

4. **Storage Management**
   - Set appropriate cache limits
   - Periodically clean unused models
   - Use consistent cache directory across sessions

## Integration with RAG

The embedding system integrates seamlessly with the RAG (Retrieval-Augmented Generation) features:

1. **Create Embeddings**: Generate embeddings for your content
2. **Store in Vector DB**: Automatically stored in ChromaDB
3. **Search**: Use semantic search to find relevant content
4. **Generate**: Use retrieved context with LLMs

See the RAG documentation for detailed integration instructions.

## API Reference

For detailed API documentation, see:
- `Embeddings/Embeddings_Lib.py` - Core embedding factory
- `RAG_Search/simplified/embeddings_wrapper.py` - Simplified interface
- `UI/Embeddings_Window.py` - UI components

## Contributing

To add new models or features:
1. Add model config to `config.py` template
2. Test with various content types
3. Update this documentation
4. Submit PR with examples

## Support

For issues or questions:
1. Check logs in `~/.local/share/tldw_cli/logs/`
2. Enable debug logging for detailed information
3. Report issues on GitHub with model details and error messages