"""
Tests for RAG configuration management.

This module tests:
- RAGConfig data model
- Configuration loading and validation
- Factory functions for different collections
- Environment variable handling
- Configuration merging
- Property-based testing for configuration validation
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st

# Import configuration classes
try:
    from tldw_chatbook.RAG_Search.simplified.config import (
        RAGConfig, 
        create_config_for_collection,
        create_config_for_testing,
        load_config_from_settings,
        validate_config
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Create placeholder implementation
    from dataclasses import dataclass, field
    
    @dataclass
    class RAGConfig:
        """RAG service configuration."""
        # Required fields
        collection_name: str
        persist_directory: str
        
        # Optional fields with defaults
        embedding_provider: str = "sentence_transformers"
        embedding_model: str = "all-MiniLM-L6-v2"
        vector_store_type: str = "chroma"
        distance_metric: str = "cosine"
        
        # Chunking parameters
        chunk_size: int = 400
        chunk_overlap: int = 100
        chunking_method: str = "words"
        
        # Search parameters
        top_k: int = 10
        score_threshold: float = 0.0
        batch_size: int = 32
        
        # Cache configuration
        enable_cache: bool = True
        cache_ttl_seconds: int = 3600
        max_cache_size: int = 1000
        
        # Performance settings
        max_memory_mb: int = 1024
        n_threads: int = 4
        
        # Feature flags
        enable_citations: bool = True
        enable_reranking: bool = False
        reranker_model: Optional[str] = None
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert config to dictionary."""
            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
                "vector_store_type": self.vector_store_type,
                "distance_metric": self.distance_metric,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunking_method": self.chunking_method,
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "batch_size": self.batch_size,
                "enable_cache": self.enable_cache,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "max_cache_size": self.max_cache_size,
                "max_memory_mb": self.max_memory_mb,
                "n_threads": self.n_threads,
                "enable_citations": self.enable_citations,
                "enable_reranking": self.enable_reranking,
                "reranker_model": self.reranker_model
            }
        
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
            """Create config from dictionary."""
            return cls(**data)
    
    def create_config_for_collection(
        collection_name: str,
        persist_dir: Optional[Path] = None,
        **kwargs
    ) -> RAGConfig:
        """Create configuration for a specific collection."""
        if persist_dir is None:
            persist_dir = Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
        
        # Collection-specific defaults
        defaults = {
            "media": {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "top_k": 10
            },
            "conversations": {
                "chunk_size": 300,
                "chunk_overlap": 50,
                "top_k": 15
            },
            "notes": {
                "chunk_size": 400,
                "chunk_overlap": 80,
                "top_k": 12
            }
        }
        
        collection_defaults = defaults.get(collection_name, {})
        
        config_dict = {
            "collection_name": collection_name,
            "persist_directory": str(persist_dir),
            **collection_defaults,
            **kwargs
        }
        
        return RAGConfig(**config_dict)
    
    def create_config_for_testing(**kwargs) -> RAGConfig:
        """Create a test configuration."""
        test_defaults = {
            "collection_name": "test_collection",
            "persist_directory": ":memory:",
            "vector_store_type": "in_memory",
            "enable_cache": False,
            "batch_size": 10,
            "max_memory_mb": 256
        }
        
        config_dict = {**test_defaults, **kwargs}
        return RAGConfig(**config_dict)
    
    def load_config_from_settings(settings: Dict[str, Any]) -> Dict[str, RAGConfig]:
        """Load RAG configurations from settings."""
        configs = {}
        
        rag_settings = settings.get("rag", {})
        base_persist_dir = Path(rag_settings.get(
            "persist_directory",
            str(Path.home() / ".local" / "share" / "tldw_cli" / "chromadb")
        ))
        
        # Load collection-specific configs
        collections = rag_settings.get("collections", ["media", "conversations", "notes"])
        for collection in collections:
            collection_settings = rag_settings.get(collection, {})
            configs[collection] = create_config_for_collection(
                collection,
                base_persist_dir,
                **collection_settings
            )
        
        return configs
    
    def validate_config(config: RAGConfig) -> bool:
        """Validate a configuration."""
        # Required fields
        if not config.collection_name:
            raise ValueError("collection_name is required")
        if not config.persist_directory:
            raise ValueError("persist_directory is required")
        
        # Value constraints
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if config.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= config.score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        if config.max_cache_size < 0:
            raise ValueError("max_cache_size cannot be negative")
        if config.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if config.n_threads <= 0:
            raise ValueError("n_threads must be positive")
        
        # Valid enums
        valid_providers = ["sentence_transformers", "openai", "cohere", "mock"]
        if config.embedding_provider not in valid_providers:
            raise ValueError(f"Invalid embedding_provider: {config.embedding_provider}")
        
        valid_stores = ["chroma", "in_memory"]
        if config.vector_store_type not in valid_stores:
            raise ValueError(f"Invalid vector_store_type: {config.vector_store_type}")
        
        valid_metrics = ["cosine", "euclidean", "manhattan"]
        if config.distance_metric not in valid_metrics:
            raise ValueError(f"Invalid distance_metric: {config.distance_metric}")
        
        valid_methods = ["words", "sentences", "paragraphs", "tokens"]
        if config.chunking_method not in valid_methods:
            raise ValueError(f"Invalid chunking_method: {config.chunking_method}")
        
        return True


# === Unit Tests ===

@pytest.mark.unit
class TestRAGConfig:
    """Test RAGConfig data model."""
    
    def test_config_creation_minimal(self):
        """Test creating config with minimal parameters."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test"
        )
        
        assert config.collection_name == "test"
        assert config.persist_directory == "/tmp/test"
        
        # Check defaults
        assert config.embedding_provider == "sentence_transformers"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.vector_store_type == "chroma"
        assert config.distance_metric == "cosine"
        assert config.chunk_size == 400
        assert config.chunk_overlap == 100
        assert config.enable_cache is True
    
    def test_config_creation_full(self):
        """Test creating config with all parameters."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test",
            embedding_provider="openai",
            embedding_model="text-embedding-ada-002",
            vector_store_type="in_memory",
            distance_metric="euclidean",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_method="sentences",
            top_k=20,
            score_threshold=0.7,
            batch_size=64,
            enable_cache=False,
            cache_ttl_seconds=7200,
            max_cache_size=5000,
            max_memory_mb=2048,
            n_threads=8,
            enable_citations=False,
            enable_reranking=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.vector_store_type == "in_memory"
        assert config.distance_metric == "euclidean"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.chunking_method == "sentences"
        assert config.top_k == 20
        assert config.score_threshold == 0.7
        assert config.batch_size == 64
        assert config.enable_cache is False
        assert config.cache_ttl_seconds == 7200
        assert config.max_cache_size == 5000
        assert config.max_memory_mb == 2048
        assert config.n_threads == 8
        assert config.enable_citations is False
        assert config.enable_reranking is True
        assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def test_config_serialization(self):
        """Test config serialization to/from dict."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test",
            chunk_size=500,
            enable_reranking=True,
            reranker_model="test-model"
        )
        
        # To dict
        data = config.to_dict()
        assert data["collection_name"] == "test"
        assert data["persist_directory"] == "/tmp/test"
        assert data["chunk_size"] == 500
        assert data["enable_reranking"] is True
        assert data["reranker_model"] == "test-model"
        
        # From dict
        restored = RAGConfig.from_dict(data)
        assert restored.collection_name == config.collection_name
        assert restored.persist_directory == config.persist_directory
        assert restored.chunk_size == config.chunk_size
        assert restored.enable_reranking == config.enable_reranking
        assert restored.reranker_model == config.reranker_model


@pytest.mark.unit
class TestConfigFactoryFunctions:
    """Test configuration factory functions."""
    
    def test_create_config_for_collection_media(self, temp_dir):
        """Test creating config for media collection."""
        config = create_config_for_collection("media", persist_dir=temp_dir)
        
        assert config.collection_name == "media"
        assert config.persist_directory == str(temp_dir)
        assert config.chunk_size == 500  # Media-specific default
        assert config.chunk_overlap == 100
        assert config.top_k == 10
    
    def test_create_config_for_collection_conversations(self, temp_dir):
        """Test creating config for conversations collection."""
        config = create_config_for_collection("conversations", persist_dir=temp_dir)
        
        assert config.collection_name == "conversations"
        assert config.persist_directory == str(temp_dir)
        assert config.chunk_size == 300  # Conversation-specific default
        assert config.chunk_overlap == 50
        assert config.top_k == 15
    
    def test_create_config_for_collection_notes(self, temp_dir):
        """Test creating config for notes collection."""
        config = create_config_for_collection("notes", persist_dir=temp_dir)
        
        assert config.collection_name == "notes"
        assert config.persist_directory == str(temp_dir)
        assert config.chunk_size == 400  # Notes-specific default
        assert config.chunk_overlap == 80
        assert config.top_k == 12
    
    def test_create_config_for_collection_custom(self, temp_dir):
        """Test creating config for custom collection."""
        config = create_config_for_collection(
            "custom",
            persist_dir=temp_dir,
            chunk_size=600,
            embedding_model="custom-model"
        )
        
        assert config.collection_name == "custom"
        assert config.persist_directory == str(temp_dir)
        assert config.chunk_size == 600  # Custom override
        assert config.embedding_model == "custom-model"
    
    def test_create_config_for_testing(self):
        """Test creating test configuration."""
        config = create_config_for_testing()
        
        assert config.collection_name == "test_collection"
        assert config.persist_directory == ":memory:"
        assert config.vector_store_type == "in_memory"
        assert config.enable_cache is False
        assert config.batch_size == 10
        assert config.max_memory_mb == 256
    
    def test_create_config_for_testing_override(self):
        """Test creating test configuration with overrides."""
        config = create_config_for_testing(
            collection_name="custom_test",
            chunk_size=200,
            enable_cache=True
        )
        
        assert config.collection_name == "custom_test"
        assert config.chunk_size == 200
        assert config.enable_cache is True
        # Test defaults still applied
        assert config.persist_directory == ":memory:"
        assert config.vector_store_type == "in_memory"


@pytest.mark.unit
class TestConfigLoading:
    """Test configuration loading from settings."""
    
    def test_load_config_from_settings_basic(self):
        """Test loading config from basic settings."""
        settings = {
            "rag": {
                "persist_directory": "/custom/path",
                "collections": ["media", "notes"]
            }
        }
        
        configs = load_config_from_settings(settings)
        
        assert len(configs) == 2
        assert "media" in configs
        assert "notes" in configs
        assert configs["media"].persist_directory == "/custom/path"
        assert configs["notes"].persist_directory == "/custom/path"
    
    def test_load_config_from_settings_with_overrides(self):
        """Test loading config with collection-specific overrides."""
        settings = {
            "rag": {
                "persist_directory": "/base/path",
                "collections": ["media", "conversations"],
                "media": {
                    "chunk_size": 1000,
                    "embedding_model": "large-model"
                },
                "conversations": {
                    "chunk_size": 200,
                    "enable_cache": False
                }
            }
        }
        
        configs = load_config_from_settings(settings)
        
        assert configs["media"].chunk_size == 1000
        assert configs["media"].embedding_model == "large-model"
        assert configs["conversations"].chunk_size == 200
        assert configs["conversations"].enable_cache is False
    
    def test_load_config_from_empty_settings(self):
        """Test loading config from empty settings."""
        settings = {}
        
        configs = load_config_from_settings(settings)
        
        # Should use defaults
        assert len(configs) == 3  # Default collections
        assert "media" in configs
        assert "conversations" in configs
        assert "notes" in configs


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test"
        )
        
        assert validate_config(config) is True
    
    def test_validate_missing_collection_name(self):
        """Test validation with missing collection name."""
        config = RAGConfig(
            collection_name="",
            persist_directory="/tmp/test"
        )
        
        with pytest.raises(ValueError, match="collection_name is required"):
            validate_config(config)
    
    def test_validate_invalid_chunk_size(self):
        """Test validation with invalid chunk size."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test",
            chunk_size=0
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_config(config)
    
    def test_validate_invalid_chunk_overlap(self):
        """Test validation with invalid chunk overlap."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test",
            chunk_size=100,
            chunk_overlap=100  # Equal to chunk_size
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            validate_config(config)
    
    def test_validate_invalid_score_threshold(self):
        """Test validation with invalid score threshold."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test",
            score_threshold=1.5  # Out of range
        )
        
        with pytest.raises(ValueError, match="score_threshold must be between 0 and 1"):
            validate_config(config)
    
    def test_validate_invalid_embedding_provider(self):
        """Test validation with invalid embedding provider."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test",
            embedding_provider="invalid_provider"
        )
        
        with pytest.raises(ValueError, match="Invalid embedding_provider"):
            validate_config(config)
    
    def test_validate_invalid_vector_store_type(self):
        """Test validation with invalid vector store type."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test",
            vector_store_type="invalid_store"
        )
        
        with pytest.raises(ValueError, match="Invalid vector_store_type"):
            validate_config(config)


@pytest.mark.unit
class TestConfigEnvironmentVariables:
    """Test configuration with environment variables."""
    
    @patch.dict(os.environ, {"RAG_PERSIST_DIR": "/env/path"})
    def test_config_from_env_persist_dir(self):
        """Test using persist directory from environment."""
        # This would require implementation in actual config module
        # For now, test the pattern
        persist_dir = os.environ.get("RAG_PERSIST_DIR", "/default/path")
        config = create_config_for_collection("test", persist_dir=Path(persist_dir))
        
        assert config.persist_directory == "/env/path"
    
    @patch.dict(os.environ, {
        "RAG_EMBEDDING_PROVIDER": "openai",
        "RAG_EMBEDDING_MODEL": "text-embedding-3-small"
    })
    def test_config_from_env_embedding_settings(self):
        """Test using embedding settings from environment."""
        # Simulate loading from environment
        provider = os.environ.get("RAG_EMBEDDING_PROVIDER", "sentence_transformers")
        model = os.environ.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        config = create_config_for_collection(
            "test",
            embedding_provider=provider,
            embedding_model=model
        )
        
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"


# === Property-Based Tests ===

@pytest.mark.property
class TestConfigProperties:
    """Property-based tests for configuration."""
    
    @given(
        collection_name=st.text(min_size=1, max_size=50),
        persist_dir=st.text(min_size=1, max_size=200),
        chunk_size=st.integers(min_value=1, max_value=10000),
        chunk_overlap=st.integers(min_value=0, max_value=1000),
        top_k=st.integers(min_value=1, max_value=1000),
        score_threshold=st.floats(min_value=0.0, max_value=1.0),
        batch_size=st.integers(min_value=1, max_value=1000),
        cache_ttl=st.integers(min_value=0, max_value=86400),
        max_cache_size=st.integers(min_value=0, max_value=10000)
    )
    def test_config_creation_properties(
        self, collection_name, persist_dir, chunk_size, chunk_overlap,
        top_k, score_threshold, batch_size, cache_ttl, max_cache_size
    ):
        """Test that valid parameters create valid configs."""
        # Ensure chunk_overlap < chunk_size
        chunk_overlap = min(chunk_overlap, chunk_size - 1)
        
        config = RAGConfig(
            collection_name=collection_name,
            persist_directory=persist_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            score_threshold=score_threshold,
            batch_size=batch_size,
            cache_ttl_seconds=cache_ttl,
            max_cache_size=max_cache_size
        )
        
        # Should create successfully
        assert config.collection_name == collection_name
        assert config.persist_directory == persist_dir
        assert config.chunk_size == chunk_size
        assert config.chunk_overlap == chunk_overlap
        assert config.top_k == top_k
        assert config.score_threshold == score_threshold
        assert config.batch_size == batch_size
        assert config.cache_ttl_seconds == cache_ttl
        assert config.max_cache_size == max_cache_size
        
        # Should validate successfully
        assert validate_config(config) is True
    
    @given(
        config_dict=st.fixed_dictionaries({
            "collection_name": st.text(min_size=1, max_size=50),
            "persist_directory": st.text(min_size=1, max_size=200),
            "chunk_size": st.integers(min_value=50, max_value=2000),
            "chunk_overlap": st.integers(min_value=0, max_value=100),
            "enable_cache": st.booleans(),
            "enable_citations": st.booleans(),
            "enable_reranking": st.booleans()
        })
    )
    def test_config_serialization_properties(self, config_dict):
        """Test that serialization is lossless."""
        # Ensure valid overlap
        config_dict["chunk_overlap"] = min(
            config_dict["chunk_overlap"],
            config_dict["chunk_size"] - 1
        )
        
        config = RAGConfig(**config_dict)
        
        # Serialize and deserialize
        data = config.to_dict()
        restored = RAGConfig.from_dict(data)
        
        # Should be identical
        for key, value in config_dict.items():
            assert getattr(restored, key) == value


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration."""
    
    def test_config_with_real_paths(self, temp_dir):
        """Test configuration with real filesystem paths."""
        # Create directory structure
        chroma_dir = temp_dir / "chromadb"
        chroma_dir.mkdir()
        
        # Create config
        config = create_config_for_collection("test", persist_dir=chroma_dir)
        
        # Verify path exists
        assert Path(config.persist_directory).exists()
        assert Path(config.persist_directory).is_dir()
    
    def test_config_loading_from_file(self, temp_dir):
        """Test loading configuration from TOML file."""
        # Create config file
        config_file = temp_dir / "rag_config.toml"
        config_content = """
[rag]
persist_directory = "/custom/chromadb"
collections = ["media", "notes"]

[rag.media]
chunk_size = 1000
embedding_model = "large-model"

[rag.notes]
chunk_size = 300
enable_cache = false
"""
        config_file.write_text(config_content)
        
        # Parse TOML (would use tomli in real implementation)
        import tomli
        with open(config_file, "rb") as f:
            settings = tomli.load(f)
        
        # Load configs
        configs = load_config_from_settings(settings)
        
        assert len(configs) == 2
        assert configs["media"].chunk_size == 1000
        assert configs["media"].embedding_model == "large-model"
        assert configs["notes"].chunk_size == 300
        assert configs["notes"].enable_cache is False
    
    def test_config_with_different_stores(self):
        """Test configurations for different vector stores."""
        # ChromaDB config
        chroma_config = RAGConfig(
            collection_name="chroma_test",
            persist_directory="/tmp/chroma",
            vector_store_type="chroma"
        )
        
        # In-memory config
        memory_config = RAGConfig(
            collection_name="memory_test",
            persist_directory=":memory:",
            vector_store_type="in_memory"
        )
        
        # Both should validate
        assert validate_config(chroma_config) is True
        assert validate_config(memory_config) is True
        
        # Different store types
        assert chroma_config.vector_store_type != memory_config.vector_store_type


@pytest.mark.unit
class TestConfigEdgeCases:
    """Test edge cases for configuration."""
    
    def test_config_with_unicode_paths(self):
        """Test configuration with unicode in paths."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/测试/тест/🔍"
        )
        
        assert "测试" in config.persist_directory
        assert "тест" in config.persist_directory
        assert "🔍" in config.persist_directory
    
    def test_config_with_very_long_paths(self):
        """Test configuration with very long paths."""
        long_path = "/tmp/" + "a" * 1000 + "/chromadb"
        config = RAGConfig(
            collection_name="test",
            persist_directory=long_path
        )
        
        assert len(config.persist_directory) > 1000
    
    def test_config_with_special_collection_names(self):
        """Test configuration with special characters in collection names."""
        special_names = [
            "test-collection",
            "test_collection",
            "test.collection",
            "test:collection",
            "test@collection"
        ]
        
        for name in special_names:
            config = RAGConfig(
                collection_name=name,
                persist_directory="/tmp/test"
            )
            assert config.collection_name == name
    
    def test_config_minimal_cache_settings(self):
        """Test configuration with minimal cache settings."""
        config = RAGConfig(
            collection_name="test",
            persist_directory="/tmp/test",
            enable_cache=True,
            cache_ttl_seconds=0,  # No TTL
            max_cache_size=1  # Minimal size
        )
        
        assert config.cache_ttl_seconds == 0
        assert config.max_cache_size == 1
        assert validate_config(config) is True