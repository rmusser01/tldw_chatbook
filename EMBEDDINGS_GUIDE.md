# Complete Guide to Embeddings in tldw_chatbook

This guide covers everything you need to know about setting up and using embeddings for RAG (Retrieval-Augmented Generation) in tldw_chatbook.

## Table of Contents
- [What are Embeddings?](#what-are-embeddings)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setting Up Embeddings](#setting-up-embeddings)
- [Creating Embeddings](#creating-embeddings)
- [Using Embeddings for RAG Search](#using-embeddings-for-rag-search)
- [Managing Embedding Models](#managing-embedding-models)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## What are Embeddings?

Embeddings are numerical representations of text that capture semantic meaning. In tldw_chatbook, embeddings enable:
- **Semantic Search**: Find content by meaning, not just keywords
- **RAG (Retrieval-Augmented Generation)**: Enhance LLM responses with relevant context from your documents
- **Smart Document Retrieval**: Find related content across your entire knowledge base

## Prerequisites

Before using embeddings, ensure you have:
1. Python 3.11 or higher
2. tldw_chatbook installed
3. Sufficient disk space for models (varies by model: ~100MB to 2GB)
4. (Optional) OpenAI API key for OpenAI embeddings

## Installation

### Step 1: Install with Embeddings Support

```bash
# Install tldw-cli with embeddings support
pip install "tldw-cli[embeddings_rag]"

# Or if installing from source
pip install -e ".[embeddings_rag]"
```

This installs required dependencies:
- `sentence-transformers`
- `torch` (CPU version by default)
- `numpy`
- `scikit-learn`

### Step 2: Verify Installation

Run the application and check if the Embeddings tab appears:
```bash
tldw-cli
```

## Setting Up Embeddings

### Step 1: Access Embeddings Management

1. Launch tldw_chatbook
2. Click on the **"Embeddings"** tab
3. Click **"Manage Embeddings"** in the left panel

### Step 2: Choose an Embedding Model

#### Available Models:

**1. e5-small-v2 (Recommended for beginners)**
- Size: ~130MB
- Dimensions: 384
- Good balance of speed and quality
- Multilingual support

**2. multilingual-e5-large-instruct**
- Size: ~2GB
- Dimensions: 1024
- Better quality for complex queries
- Excellent multilingual support

**3. OpenAI Embeddings (Requires API key)**
- No download needed
- Dimensions: 1536 (text-embedding-ada-002)
- High quality but requires internet and costs money

### Step 3: Download a Model

1. Select a model from the list (e.g., "e5-small-v2")
2. Click the **"Download"** button
3. Wait for status to show "Downloaded"
4. The model is now ready to use

## Creating Embeddings

### Step 1: Prepare Your Content

First, ingest content into tldw_chatbook:
1. Go to the **"Media"** tab
2. Choose your content type:
   - Videos/Audio
   - Documents (Word, PowerPoint)
   - PDFs/EPUBs
   - Websites
   - Plain text

### Step 2: Create Embeddings for Your Content

1. Go to the **"Embeddings"** tab
2. You'll see a list of all ingested content
3. Select items to embed:
   - Use checkboxes for individual items
   - Use "Select All" for batch processing
4. Choose your embedding model from the dropdown
5. Click **"Create Embeddings"**
6. Monitor progress in the status area

### Step 3: Monitor Progress

- Progress bar shows completion percentage
- Status messages indicate current processing
- Large documents are processed in chunks
- You can continue using other tabs while embedding

## Using Embeddings for RAG Search

### Step 1: Access RAG Search

1. Go to the **"Search/RAG"** tab
2. Toggle **"Enable RAG Search"** to ON

### Step 2: Configure Search Settings

**Search Type Options:**
- **Hybrid** (Recommended): Combines keyword and semantic search
- **Semantic Only**: Pure embedding-based search
- **Keyword Only**: Traditional text search

**Top K Results:**
- Set how many results to retrieve (default: 10)
- More results = more context but slower

### Step 3: Perform a Search

1. Enter your query in the search box
2. Results appear instantly as you type
3. Each result shows:
   - Title and source
   - Relevant text snippet
   - Similarity score (for semantic search)

### Step 4: Use Results in Chat

1. Select relevant results using checkboxes
2. Go to the **"Chat"** tab
3. Your selected context is automatically included
4. The LLM will use this context to answer questions

## Managing Embedding Models

### View Model Information

In Embeddings Management:
- **Provider**: HuggingFace or OpenAI
- **Dimensions**: Vector size (affects storage and quality)
- **Status**: Not Downloaded, Downloaded, or Loaded
- **Cache Location**: Where model files are stored

### Model Operations

**Download Model:**
- Only needed for HuggingFace models
- Downloads to `~/.local/share/tldw_cli/models/embeddings/`
- Supports resume if interrupted

**Load Model:**
- Models load automatically when needed
- First use may take a few seconds

**Unload Model:**
- Frees memory by removing model from cache
- Useful when switching between models

## Configuration Options

### Edit Configuration File

Location: `~/.config/tldw_cli/config.toml`

```toml
[Embeddings]
# Default model to use
default_model = "e5-small-v2"

# Automatically download models when needed
auto_download = true

# Model cache directory
cache_dir = "~/.local/share/tldw_cli/models/embeddings"

# Chunk size for processing
chunk_size = 512

# Overlap between chunks
chunk_overlap = 128

[Embeddings.models.e5-small-v2]
provider = "huggingface"
model_name_or_path = "intfloat/e5-small-v2"
embedding_dims = 384
max_seq_length = 512

[Embeddings.models.openai]
provider = "openai"
model_name_or_path = "text-embedding-ada-002"
embedding_dims = 1536
api_key_env = "OPENAI_API_KEY"
```

### Environment Variables

```bash
# For OpenAI embeddings
export OPENAI_API_KEY="your-api-key-here"

# Custom cache directory
export TLDW_EMBEDDINGS_CACHE="/path/to/cache"
```

## Troubleshooting

### Common Issues

**1. "No module named 'sentence_transformers'"**
- Solution: Install with `pip install "tldw-cli[embeddings_rag]"`

**2. "Model download failed"**
- Check internet connection
- Verify sufficient disk space
- Try manual download: `huggingface-cli download intfloat/e5-small-v2`

**3. "Out of memory" errors**
- Use smaller model (e5-small-v2)
- Process fewer documents at once
- Increase system swap space

**4. "Embeddings tab not visible"**
- Verify embeddings dependencies installed
- Check logs for import errors
- Restart the application

### Performance Issues

**Slow Embedding Creation:**
- Use GPU if available: `pip install torch-cuda`
- Reduce chunk size in config
- Use smaller model
- Process in smaller batches

**High Memory Usage:**
- Unload models when not in use
- Use OpenAI embeddings (no local model)
- Reduce batch size

## Best Practices

### Content Preparation

1. **Clean Your Content**
   - Remove formatting artifacts
   - Fix encoding issues
   - Split very large documents

2. **Organize by Topic**
   - Group related content
   - Use consistent naming
   - Add metadata when ingesting

### Embedding Strategy

1. **Choose the Right Model**
   - Start with e5-small-v2
   - Upgrade to large model if needed
   - Use OpenAI for best quality

2. **Chunk Settings**
   - Default (512/128) works well
   - Increase for technical content
   - Decrease for conversational content

3. **Batch Processing**
   - Embed similar content together
   - Process during off-hours
   - Monitor system resources

### Search Optimization

1. **Query Formulation**
   - Be specific but natural
   - Include context in queries
   - Use questions for best results

2. **Result Selection**
   - Review similarity scores
   - Select diverse sources
   - Limit to relevant chunks

3. **Context Management**
   - Don't overload with context
   - Prioritize recent/relevant content
   - Clear context between topics

### Maintenance

1. **Regular Updates**
   - Re-embed updated content
   - Remove obsolete embeddings
   - Update models periodically

2. **Storage Management**
   - Monitor embedding database size
   - Archive old embeddings
   - Clean up unused models

3. **Performance Monitoring**
   - Track embedding creation time
   - Monitor search latency
   - Check memory usage

## Advanced Tips

### Custom Models

Add custom HuggingFace models to config:
```toml
[Embeddings.models.my-custom-model]
provider = "huggingface"
model_name_or_path = "username/model-name"
embedding_dims = 768
max_seq_length = 512
```

### GPU Acceleration

For NVIDIA GPUs:
```bash
pip install torch-cuda
```

For Apple Silicon:
```bash
# Automatically uses Metal Performance Shaders
```

### Embedding Export

Embeddings are stored in SQLite database:
- Location: `~/.share/tldw_cli/media_db.sqlite`
- Table: `embeddings`
- Can be exported for external use

## Conclusion

Embeddings transform tldw_chatbook into a powerful knowledge management system. Start with the basics:
1. Download e5-small-v2
2. Embed your most important documents
3. Use hybrid search for best results

As you become comfortable, explore advanced features like custom models and GPU acceleration. Remember that embeddings are a tool to enhance your workflow - focus on the content and queries that matter most to you.

For more help, check the application logs or file an issue on GitHub.