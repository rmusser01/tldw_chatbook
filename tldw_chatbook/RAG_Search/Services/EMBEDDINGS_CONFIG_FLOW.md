# Embeddings Service Configuration Flow

## Overview

Yes, the new embeddings service uses the existing `config.toml` file for loading its configuration. Here's how the configuration flows from config.toml to the embeddings service:

## Configuration Loading Flow

### 1. config.toml Structure

The embeddings configuration is defined in the `[embedding_config]` section of config.toml:

```toml
[embedding_config]
default_model_id = "e5-small-v2"
default_llm_for_contextualization = "gpt-3.5-turbo"

    [embedding_config.models.e5-small-v2]
    provider = "huggingface"
    model_name_or_path = "intfloat/e5-small-v2"
    dimension = 384
    trust_remote_code = false
    max_length = 512

    [embedding_config.models.openai-3-small]
    provider = "openai"
    model_name_or_path = "text-embedding-3-small"
    api_key = "${OPENAI_API_KEY}"  # Uses environment variable
    dimension = 1536
```

### 2. Config Loading in config.py

1. The `load_cli_config_and_ensure_existence()` function in `config.py` loads the TOML file from:
   - User config: `~/.config/tldw_cli/config.toml`
   - Or default location

2. The embedding_config section is extracted:
   ```python
   embedding_config_section = get_toml_section('embedding_config')
   ```

3. It's included in the comprehensive config:
   ```python
   "embedding_config": {
       # ... other settings ...
       'models': embedding_config_section.get('models', {})
   }
   ```

### 3. Config Flow to ChromaDBManager

When ChromaDBManager is initialized, it receives the config and looks for embedding_config in multiple locations:

```python
# Priority 1: Direct embedding_config
embeddings_config_dict = user_embedding_config.get("embedding_config")

# Priority 2: Inside COMPREHENSIVE_CONFIG_RAW
if not embeddings_config_dict:
    comprehensive_config = user_embedding_config.get("COMPREHENSIVE_CONFIG_RAW", {})
    embeddings_config_dict = comprehensive_config.get("embedding_config")
```

### 4. Initialization of Embeddings Service

When using the new embeddings service (controlled by environment variable or config flag):

```python
# Check if we should use new service
use_new_service = os.getenv("USE_NEW_EMBEDDINGS_SERVICE", "").lower() in ("true", "1", "yes")

if use_new_service:
    from tldw_chatbook.RAG_Search.Services.embeddings_compat import EmbeddingFactoryCompat
    self.embedding_factory = EmbeddingFactoryCompat(cfg=embeddings_config_dict)
```

### 5. EmbeddingFactoryCompat Initialization

The compatibility layer creates the new service and passes the config:

```python
class EmbeddingFactoryCompat:
    def __init__(self, cfg: Dict[str, Any], storage_path: Optional[str] = None):
        # Create new embeddings service
        self._service = EmbeddingsService(
            persist_directory=storage_path,
            embedding_config=cfg
        )
        
        # Initialize providers from config
        self._service.initialize_from_config({"embedding_config": cfg})
```

### 6. Provider Initialization from Config

The `initialize_from_config` method in EmbeddingsService parses the config and creates providers:

```python
def initialize_from_config(self, config: Dict[str, Any]) -> bool:
    embed_config = config.get("embedding_config", {})
    models = embed_config.get("models", {})
    
    for model_id, model_cfg in models.items():
        provider_type = model_cfg.get("provider", "").lower()
        
        if provider_type == "huggingface":
            provider = HuggingFaceProvider(
                model_name=model_cfg.get("model_name_or_path"),
                trust_remote_code=model_cfg.get("trust_remote_code", False),
                # ... other params from config
            )
        elif provider_type == "openai":
            provider = OpenAIProvider(
                model_name=model_cfg.get("model_name_or_path"),
                api_key=model_cfg.get("api_key"),
                # ... other params from config
            )
        # ... other provider types
        
        self.providers[model_id] = provider
```

## Configuration Compatibility

The new embeddings service maintains full compatibility with the existing config.toml structure:

1. **Same Config Format**: Uses the exact same `[embedding_config]` section
2. **Same Model Definitions**: Reads the same `models` subsections
3. **Same Parameters**: Supports all existing parameters (provider, model_name_or_path, dimension, etc.)
4. **Environment Variables**: Respects environment variable substitution (e.g., `${OPENAI_API_KEY}`)
5. **Backward Compatible**: When the new service is not enabled, falls back to legacy EmbeddingFactory

## Enabling the New Service

To use the new embeddings service with your existing config.toml:

1. Set environment variable:
   ```bash
   export USE_NEW_EMBEDDINGS_SERVICE=true
   ```

2. Or add to config:
   ```toml
   [embedding_config]
   use_new_embeddings_service = true
   ```

The service will automatically read your existing embedding configuration from config.toml and initialize the appropriate providers.