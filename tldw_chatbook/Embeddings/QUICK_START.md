# Embeddings Quick Start Guide

## Installation
```bash
pip install tldw_chatbook[embeddings_rag]
```

## Quick Setup

### 1. Configure Models (config.toml)
```toml
[embedding_config]
default_model_id = "e5-small-v2"
model_cache_dir = "~/.local/share/tldw_cli/models/embeddings"
auto_download = true
```

### 2. Download a Model
**Via UI:**
1. Go to Embeddings tab → Manage
2. Select a model (e.g., "e5-small-v2")
3. Click "Download"

**Via Command Line:**
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/e5-small-v2', cache_dir='~/.local/share/tldw_cli/models/embeddings')"
```

### 3. Create Embeddings
**Via UI:**
1. Embeddings tab → Create
2. Select model from dropdown
3. Choose input (File or Database)
4. Configure chunking
5. Click "Create Embeddings"

**Via Code:**
```python
from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory

# Initialize
factory = EmbeddingFactory(config)

# Create embeddings
texts = ["Your text here"]
embeddings = factory.embed(texts, model_id="e5-small-v2")
```

## Popular Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `e5-small-v2` | 384 | Fast | Good | General purpose |
| `all-MiniLM-L6-v2` | 384 | Very Fast | Good | Quick search |
| `e5-base-v2` | 768 | Medium | Better | Better quality |
| `e5-large-v2` | 1024 | Slow | Best | Highest quality |

## Model Locations

- **Default HF Cache**: `~/.cache/huggingface/hub/`
- **Custom Cache**: Set in `model_cache_dir`
- **Check Location**: Embeddings → Manage → Select Model → Cache Location

## Common Commands

### List Downloaded Models
```bash
ls ~/.local/share/tldw_cli/models/embeddings/
```

### Check Model Size
```bash
du -sh ~/.local/share/tldw_cli/models/embeddings/models--intfloat--e5-small-v2/
```

### Clear Model Cache
```bash
rm -rf ~/.local/share/tldw_cli/models/embeddings/models--intfloat--e5-small-v2/
```

## Troubleshooting

### Out of Memory
- Use smaller model (e.g., `e5-small-v2` instead of `e5-large-v2`)
- Reduce batch size in config
- Unload unused models in Management UI

### Download Issues
- Check internet connection
- Try manual download with `huggingface-cli`
- Verify write permissions on cache directory

### Slow Performance
- Enable GPU: `device = "cuda"` in model config
- Use smaller models for testing
- Increase batch size for better throughput

## Tips

1. **Start Small**: Begin with `e5-small-v2` for testing
2. **GPU Acceleration**: Add `device = "cuda"` to model config
3. **Batch Processing**: Process multiple texts at once
4. **Monitor Memory**: Check memory usage in Management UI
5. **Preload Models**: Download models before first use

## Need More?
See [EMBEDDINGS_GUIDE.md](EMBEDDINGS_GUIDE.md) for comprehensive documentation.