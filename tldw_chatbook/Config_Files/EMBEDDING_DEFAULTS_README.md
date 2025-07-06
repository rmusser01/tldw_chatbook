# Embedding Configuration Guide for tldw_chatbook

This guide explains how to configure embedding models for the `embeddings_rag` module in tldw_chatbook.

## Quick Start

When you install tldw_chatbook with the `embeddings_rag` feature, it comes with sensible defaults:

```bash
pip install tldw_chatbook[embeddings_rag]
```

**Default Configuration:**
- Model: `mxbai-embed-large-v1` (1024 dimensions)
- Provider: HuggingFace
- Device: Auto-detected (CUDA > MPS > CPU)
- Auto-download: Enabled
- Model size: ~335MB
- Special feature: Supports Matryoshka dimensions (512, 256)

No additional configuration is required to start using embeddings!

## Configuration Locations

Embedding settings can be configured in multiple places (in order of priority):

1. **Environment Variables** (highest priority)
   - `RAG_EMBEDDING_MODEL` - Override the embedding model
   - `RAG_DEVICE` - Force specific device (cuda/mps/cpu)
   - `OPENAI_API_KEY` - For OpenAI embeddings

2. **RAG Configuration** in `config.toml`:
   ```toml
   [rag.embedding]
   model = "mxbai-embed-large-v1"
   device = "auto"
   cache_size = 2
   batch_size = 16
   ```

3. **Embedding Configuration** in `config.toml`:
   ```toml
   [embedding_config]
   default_model_id = "mxbai-embed-large-v1"
   
   [embedding_config.models.mxbai-embed-large-v1]
   provider = "huggingface"
   model_name_or_path = "mixedbread-ai/mxbai-embed-large-v1"
   dimension = 1024
   ```

4. **Built-in Defaults** (lowest priority)

## Available Default Models

The system comes pre-configured with several embedding models:

### High-Quality Models (Default)
- **mxbai-embed-large-v1** (Default) - Best quality, supports Matryoshka dimensions
  - Full: 1024 dimensions
  - Reduced: 512 dimensions (93% performance)
  - Fast: 256 dimensions (still good quality)

### Small Models (Fast, ~100MB)
- **e5-small-v2** - Good balance of speed and quality
- **all-MiniLM-L6-v2** - Fastest, good for development
- **bge-small-en-v1.5** - Good multilingual support

### Medium Models (Balanced, ~400MB)
- **e5-base-v2** - Excellent general-purpose
- **all-mpnet-base-v2** - Great for semantic similarity
- **bge-base-en-v1.5** - Strong retrieval performance

### Large Models (Best Quality, ~1.3GB)
- **e5-large-v2** - Top quality embeddings
- **multilingual-e5-large-instruct** - Best for multiple languages

### State-of-the-Art Models (Require trust_remote_code)
- **stella_en_1.5B_v5** - Very high quality, supports 512-8192 dimensions
  - Security: Pinned to specific revision to prevent code injection
  - Matryoshka: Can use smaller dimensions with minimal quality loss
  - Size: ~1.5GB
- **qwen3-embedding-4b** - Top performance, 32k context
  - Supports 100+ languages
  - Flexible dimensions up to 4096
  - Size: ~4GB

### API Models (Requires API Key)
- **openai-ada-002** - OpenAI's legacy model
- **openai-3-small** - New generation, efficient
- **openai-3-large** - Highest quality

## Basic Configuration Examples

### 1. Minimal Configuration (Uses Defaults)
```toml
# No configuration needed! The system will use e5-small-v2 automatically
```

### 2. Change Default Model
```toml
[embedding_config]
default_model_id = "e5-base-v2"  # Use a different model

# Or use reduced dimensions with mxbai
[embedding_config.models.mxbai-embed-large-v1]
dimension = 512  # Use 512 instead of 1024 for speed
```

### 3. Configure for Production
```toml
[embedding_config]
default_model_id = "e5-base-v2"
model_cache_dir = "/opt/models/embeddings"
cache_size_limit_gb = 20.0

[embedding_config.models.e5-base-v2]
provider = "huggingface"
model_name_or_path = "intfloat/e5-base-v2"
dimension = 768
device = "cuda"  # Explicitly use GPU
batch_size = 16
```

### 4. Use OpenAI Embeddings
```toml
[embedding_config]
default_model_id = "openai-3-small"

[embedding_config.models.openai-3-small]
provider = "openai"
model_name_or_path = "text-embedding-3-small"
dimension = 1536
# api_key set via OPENAI_API_KEY environment variable
```

### 5. Local API Server
```toml
[embedding_config]
default_model_id = "local-server"

[embedding_config.models.local-server]
provider = "openai"  # Uses OpenAI-compatible API
model_name_or_path = "e5-base-v2"
base_url = "http://localhost:8080/v1"
dimension = 768
```

## RAG-Specific Configuration

For RAG functionality, configure embeddings in the RAG section:

```toml
[rag.embedding]
model = "e5-base-v2"     # Model ID from embedding_config
device = "auto"          # Auto-detect best device
cache_size = 2           # Models to keep in memory
batch_size = 16          # Batch size for processing
max_length = 512         # Max sequence length
```

## Advanced Usage

### Using the Helper Functions

The system provides helper functions for programmatic configuration:

```python
from tldw_chatbook.Embeddings.Embeddings_Lib import (
    get_default_embedding_config,
    get_common_embedding_models,
    create_embedding_factory_with_defaults
)

# Get default configuration
default_config = get_default_embedding_config()

# Get all pre-configured models
available_models = get_common_embedding_models()

# Create factory with defaults
factory = create_embedding_factory_with_defaults()
```

### Custom Model Configuration

To add a custom model:

```toml
[embedding_config.models.my-custom-model]
provider = "huggingface"
model_name_or_path = "organization/model-name"
dimension = 768  # Must match model output
trust_remote_code = false
max_length = 512
device = "auto"
batch_size = 32
```

## Performance Tips

1. **Device Selection**:
   - `"auto"` - Let the system choose (recommended)
   - `"cuda"` - Force GPU (NVIDIA)
   - `"mps"` - Force Metal (Apple Silicon)
   - `"cpu"` - Force CPU

2. **Batch Size**:
   - Larger = faster throughput
   - Smaller = less memory usage
   - Default: 32 for small models, 16 for base, 8 for large

3. **Cache Management**:
   - `cache_size` - Number of models to keep loaded
   - `cache_size_limit_gb` - Maximum disk space for downloaded models

4. **Model Selection**:
   - Development: Use `e5-small-v2` or `all-MiniLM-L6-v2`
   - Production: Use `e5-base-v2` or `bge-base-en-v1.5`
   - High Quality: Use `e5-large-v2` or OpenAI models

## Troubleshooting

### Common Issues

1. **"No embedding configuration found"**
   - The system will use built-in defaults
   - Not an error - just informational

2. **"Failed to download model"**
   - Check internet connection
   - Verify model name is correct
   - Set `auto_download = false` to prevent automatic downloads

3. **"CUDA out of memory"**
   - Reduce batch_size
   - Use a smaller model
   - Set device to "cpu"

4. **"Import Error: embeddings_rag not installed"**
   - Install with: `pip install tldw_chatbook[embeddings_rag]`

### Environment Variables

Override settings without changing config:

```bash
# Change model
export RAG_EMBEDDING_MODEL=e5-base-v2

# Force CPU
export RAG_DEVICE=cpu

# Set API key
export OPENAI_API_KEY=sk-...
```

## Example Configurations

See `embedding_configs_examples.toml` for complete examples including:
- Development/Testing setups
- Production configurations
- High-quality configurations
- OpenAI API usage
- Local API servers
- Hybrid configurations
- Platform-specific setups

## Security Considerations

### Models Requiring trust_remote_code

Some state-of-the-art models like `stella_en_1.5B_v5` and `qwen3-embedding-4b` require `trust_remote_code=true` because they use custom model architectures. This means they download and execute Python code from the model repository.

**Security Best Practices:**
1. **Revision Pinning**: Stella model is pinned to a specific git revision to prevent malicious updates
2. **Review Code**: For critical applications, review the custom code in the model repository
3. **Sandboxing**: Consider running in a containerized environment
4. **Updates**: Only update the revision after verifying the new code is safe

Example secure configuration:
```toml
[embedding_config.models.stella_en_1.5B_v5]
provider = "huggingface"
model_name_or_path = "NovaSearch/stella_en_1.5B_v5"
dimension = 1024
trust_remote_code = true
revision = "4bbc0f1e9df5b9563d418e9b5663e98070713eb8"  # Pinned for security
```

## Migration from Previous Versions

If upgrading from an older version:

1. The system now uses `mxbai-embed-large-v1` as default (was `e5-small-v2`)
2. Device selection now supports `"auto"`
3. RAG configuration moved to `[rag.embedding]` section
4. Models are defined in `[embedding_config.models]` section
5. New models support revision pinning for security

Old configurations will continue to work via backward compatibility.