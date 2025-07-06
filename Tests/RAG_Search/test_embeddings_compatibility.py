# test_embeddings_compatibility.py
# Compatibility tests to ensure the new embeddings service works with legacy code

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any

from tldw_chatbook.RAG_Search.simplified import (
    InMemoryVectorStore
)
from tldw_chatbook.RAG_Search.Services import (
    EmbeddingsService
)
# Note: Compatibility layer may need adjustment for simplified API
from tldw_chatbook.Embeddings.Chroma_Lib import ChromaDBManager
from tldw_chatbook.RAG_Search.Services import (
    EmbeddingFactoryCompat, 
    create_legacy_factory
)

# Import test utilities from conftest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import requires_numpy, MockEmbeddingProvider


@pytest.mark.compatibility
class TestLegacyAPICompatibility:
    """Test that the new service maintains compatibility with legacy API"""
    
    def test_embedding_factory_interface(self, legacy_config):
        """Test EmbeddingFactoryCompat provides the same interface as legacy"""
        factory = EmbeddingFactoryCompat(legacy_config)
        
        # Check required attributes
        assert hasattr(factory, 'config')
        assert hasattr(factory, 'embed')
        assert hasattr(factory, 'embed_one')
        assert hasattr(factory, 'close')
        assert hasattr(factory, 'allow_dynamic_hf')
        
        # Check config structure
        assert factory.config.default_model_id == legacy_config["default_model_id"]
        assert hasattr(factory.config, 'models')
    
    @requires_numpy
    def test_legacy_embed_methods(self, legacy_config):
        """Test legacy embed methods work correctly"""
        # Mock the provider creation
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.HuggingFaceProvider'):
            factory = EmbeddingFactoryCompat(legacy_config)
            
            # Add a mock provider
            mock_provider = MockEmbeddingProvider(dimension=384)
            factory._service.providers["test-model"] = mock_provider
            factory._service.current_provider_id = "test-model"
            
            # Test embed_one with different parameters
            # as_list=True
            embedding_list = factory.embed_one("test text", as_list=True)
            assert isinstance(embedding_list, list)
            assert len(embedding_list) == 384
            
            # as_list=False (numpy)
            embedding_np = factory.embed_one("test text", as_list=False)
            assert isinstance(embedding_np, np.ndarray)
            assert embedding_np.shape == (384,)
            
            # Test embed multiple
            texts = ["text1", "text2", "text3"]
            
            # as_list=True
            embeddings_list = factory.embed(texts, as_list=True)
            assert isinstance(embeddings_list, list)
            assert len(embeddings_list) == 3
            assert all(len(emb) == 384 for emb in embeddings_list)
            
            # as_list=False (numpy)
            embeddings_np = factory.embed(texts, as_list=False)
            assert isinstance(embeddings_np, np.ndarray)
            assert embeddings_np.shape == (3, 384)
    
    def test_model_id_parameter(self, legacy_config):
        """Test model_id parameter in embed methods"""
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.HuggingFaceProvider'):
            with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.OpenAIProvider'):
                factory = EmbeddingFactoryCompat(legacy_config)
                
                # Add mock providers
                mock_hf = MockEmbeddingProvider(dimension=384)
                mock_openai = MockEmbeddingProvider(dimension=1536)
                
                factory._service.providers["test-model"] = mock_hf
                factory._service.providers["openai-model"] = mock_openai
                
                # Use specific model
                embedding1 = factory.embed_one("test", model_id="test-model", as_list=True)
                assert len(embedding1) == 384
                
                embedding2 = factory.embed_one("test", model_id="openai-model", as_list=True)
                assert len(embedding2) == 1536
    
    def test_context_manager_compatibility(self):
        """Test context manager works like legacy"""
        factory = EmbeddingFactoryCompat({})
        
        # Should support context manager
        with factory as f:
            assert f is factory
        
        # close() should be idempotent
        factory.close()
        factory.close()  # Should not raise
    
    def test_create_legacy_factory_helper(self, legacy_config, temp_dir):
        """Test the create_legacy_factory helper function"""
        factory = create_legacy_factory(legacy_config, storage_path=temp_dir)
        
        assert isinstance(factory, EmbeddingFactoryCompat)
        assert factory._service.persist_directory == temp_dir


@pytest.mark.compatibility
class TestChromaDBManagerCompatibility:
    """Test compatibility with ChromaDBManager"""
    
    def test_chromadb_manager_with_new_service(self, temp_dir, legacy_config):
        """Test ChromaDBManager can use the new embeddings service"""
        # Enable new service
        os.environ["USE_NEW_EMBEDDINGS_SERVICE"] = "true"
        
        try:
            user_config = {
                "embedding_config": legacy_config,
                "USER_DB_BASE_DIR": str(temp_dir),
                "chroma_client_settings": {
                    "anonymized_telemetry": False,
                    "allow_reset": True
                }
            }
            
            # Mock dependencies
            with patch('tldw_chatbook.Embeddings.Chroma_Lib.DEPENDENCIES_AVAILABLE', {'embeddings_rag': True}):
                with patch('tldw_chatbook.Embeddings.Chroma_Lib.chromadb') as mock_chromadb:
                    # Mock ChromaDB client
                    mock_client = MagicMock()
                    mock_chromadb.PersistentClient.return_value = mock_client
                    
                    # Initialize ChromaDBManager
                    manager = ChromaDBManager("test_user", user_config)
                    
                    # Verify it's using the new service through compatibility layer
                    assert isinstance(manager.embedding_factory, EmbeddingFactoryCompat)
                    assert hasattr(manager.embedding_factory, '_service')
                    
                    # Test that embed methods work
                    with patch.object(manager.embedding_factory._service, 'create_embeddings') as mock_create:
                        mock_create.return_value = [[0.1] * 384]
                        
                        # Test embed_one
                        embedding = manager.embedding_factory.embed_one("test", as_list=True)
                        assert isinstance(embedding, list)
                        # Allow call with or without provider_id
                        args, kwargs = mock_create.call_args
                        assert args[0] == ["test"]
                        
                        # Test embed
                        mock_create.return_value = [[0.1] * 384, [0.2] * 384]
                        embeddings = manager.embedding_factory.embed(["text1", "text2"], as_list=True)
                        assert len(embeddings) == 2
                        
        finally:
            os.environ.pop("USE_NEW_EMBEDDINGS_SERVICE", None)
    
    def test_chromadb_manager_fallback_to_legacy(self, temp_dir, legacy_config):
        """Test ChromaDBManager falls back to legacy when new service unavailable"""
        # Disable new service
        os.environ.pop("USE_NEW_EMBEDDINGS_SERVICE", None)
        
        user_config = {
            "embedding_config": legacy_config,
            "USER_DB_BASE_DIR": str(temp_dir),
            "chroma_client_settings": {}
        }
        
        with patch('tldw_chatbook.Embeddings.Chroma_Lib.DEPENDENCIES_AVAILABLE', {'embeddings_rag': True}):
            with patch('tldw_chatbook.Embeddings.Chroma_Lib.chromadb'):
                with patch('tldw_chatbook.Embeddings.Chroma_Lib.EmbeddingFactory') as mock_legacy:
                    # Mock legacy factory
                    mock_factory_instance = MagicMock()
                    mock_legacy.return_value = mock_factory_instance
                    
                    # Initialize ChromaDBManager
                    manager = ChromaDBManager("test_user", user_config)
                    
                    # Should use legacy factory
                    mock_legacy.assert_called_once()
                    assert manager.embedding_factory == mock_factory_instance


@pytest.mark.compatibility
class TestConfigurationCompatibility:
    """Test configuration compatibility between old and new systems"""
    
    def test_legacy_config_format_support(self):
        """Test various legacy configuration formats are supported"""
        # Format 1: Simple config
        config1 = {
            "default_model_id": "minilm",
            "models": {
                "minilm": {
                    "provider": "huggingface",
                    "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        }
        
        service = EmbeddingsService()
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.HuggingFaceProvider'):
            success = service.initialize_from_config({"embedding_config": config1})
            assert success
            assert "minilm" in service.providers
        
        # Format 2: With additional fields
        config2 = {
            "default_model_id": "e5-small",
            "default_llm_for_contextualization": "gpt-3.5-turbo",
            "prompts": {
                "situate_context_template": "Template here"
            },
            "models": {
                "e5-small": {
                    "provider": "huggingface",
                    "model_name_or_path": "intfloat/e5-small-v2",
                    "dimension": 384,
                    "trust_remote_code": False,
                    "max_length": 512
                }
            }
        }
        
        service2 = EmbeddingsService()
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.HuggingFaceProvider'):
            success = service2.initialize_from_config({"embedding_config": config2})
            assert success
    
    def test_nested_config_compatibility(self, nested_config):
        """Test nested configuration format (as used in ChromaDBManager)"""
        service = EmbeddingsService()
        
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.HuggingFaceProvider') as mock_hf:
            with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.OpenAIProvider') as mock_openai:
                # Make the mocks return instances
                mock_hf.return_value = MockEmbeddingProvider(dimension=384)
                mock_openai.return_value = MockEmbeddingProvider(dimension=1536)
                
                success = service.initialize_from_config(nested_config)
                assert success
                assert len(service.providers) > 0
    
    def test_provider_type_mapping(self):
        """Test provider type mapping from legacy to new"""
        configs = [
            {
                "provider": "huggingface",
                "expected_class": "HuggingFaceProvider"
            },
            {
                "provider": "openai",
                "expected_class": "OpenAIProvider"
            },
            {
                "provider": "sentence_transformers",
                "expected_class": "SentenceTransformerProvider"
            }
        ]
        
        for config in configs:
            model_config = {
                "default_model_id": "test",
                "models": {
                    "test": {
                        "provider": config["provider"],
                        "model_name_or_path": "test-model"
                    }
                }
            }
            
            service = EmbeddingsService()
            
            # Mock all provider classes
            with patch(f'tldw_chatbook.RAG_Search.Services.embeddings_service.{config["expected_class"]}') as mock_provider:
                mock_provider.return_value = Mock(spec=MockEmbeddingProvider)
                
                success = service.initialize_from_config({"embedding_config": model_config})
                assert success
                mock_provider.assert_called_once()


@pytest.mark.compatibility
class TestBehaviorCompatibility:
    """Test that behavior matches legacy system"""
    
    def test_error_handling_compatibility(self):
        """Test error handling matches legacy behavior"""
        factory = EmbeddingFactoryCompat({})
        
        # No providers - should raise when trying to embed
        with pytest.raises(RuntimeError):
            factory.embed(["test"])
        
        # Invalid model_id - depends on implementation
        factory._service.add_provider("valid", MockEmbeddingProvider())
        factory._service.current_provider_id = "valid"
        
        # Should work with valid provider
        embedding = factory.embed_one("test", as_list=True)
        assert embedding is not None
    
    @requires_numpy
    def test_numpy_behavior_compatibility(self):
        """Test numpy array behavior matches legacy"""
        factory = EmbeddingFactoryCompat({})
        factory._service.add_provider("test", MockEmbeddingProvider(dimension=384))
        factory._service.current_provider_id = "test"
        
        # Single embedding shape
        single_np = factory.embed_one("test", as_list=False)
        assert single_np.shape == (384,)  # 1D array
        assert single_np.dtype == np.float64 or single_np.dtype == np.float32
        
        # Multiple embeddings shape
        multi_np = factory.embed(["text1", "text2", "text3"], as_list=False)
        assert multi_np.shape == (3, 384)  # 2D array
        assert multi_np.dtype == np.float64 or multi_np.dtype == np.float32
    
    def test_deterministic_embeddings(self):
        """Test embeddings are deterministic like legacy system"""
        factory = EmbeddingFactoryCompat({})
        factory._service.add_provider("test", MockEmbeddingProvider())
        factory._service.current_provider_id = "test"
        
        text = "This is a test document"
        
        # Generate embeddings multiple times
        embeddings1 = factory.embed_one(text, as_list=True)
        embeddings2 = factory.embed_one(text, as_list=True)
        embeddings3 = factory.embed_one(text, as_list=True)
        
        # Should be identical
        assert embeddings1 == embeddings2
        assert embeddings2 == embeddings3
    
    def test_empty_text_handling(self):
        """Test handling of empty texts matches legacy behavior"""
        factory = EmbeddingFactoryCompat({})
        factory._service.add_provider("test", MockEmbeddingProvider())
        factory._service.current_provider_id = "test"
        
        # Empty string
        embedding = factory.embed_one("", as_list=True)
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        
        # List with empty string
        embeddings = factory.embed(["text1", "", "text3"], as_list=True)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)


@pytest.mark.compatibility
class TestMigrationScenarios:
    """Test real-world migration scenarios"""
    
    def test_gradual_migration(self, legacy_config):
        """Test gradual migration from legacy to new system"""
        # Step 1: Create legacy factory
        with patch('tldw_chatbook.Embeddings.Embeddings_Lib.EmbeddingFactory') as mock_legacy:
            mock_legacy_instance = MagicMock()
            mock_legacy_instance.embed.return_value = [[0.1] * 384]
            mock_legacy.return_value = mock_legacy_instance
            
            # Step 2: Create new factory with compatibility
            new_factory = EmbeddingFactoryCompat(legacy_config)
            new_factory._service.add_provider("test", MockEmbeddingProvider())
            new_factory._service.current_provider_id = "test"
            
            # Step 3: Both should produce embeddings
            legacy_result = mock_legacy_instance.embed(["test"], as_list=True)
            new_result = new_factory.embed(["test"], as_list=True)
            
            assert len(legacy_result) == len(new_result)
            assert len(legacy_result[0]) == len(new_result[0])
    
    def test_config_migration_path(self):
        """Test configuration migration from legacy to new format"""
        # Legacy format
        legacy = {
            "default_model_id": "minilm",
            "models": {
                "minilm": {
                    "provider": "huggingface",
                    "model_name_or_path": "all-MiniLM-L6-v2",
                    "dimension": 384
                }
            }
        }
        
        # Can be used directly with new service
        service = EmbeddingsService()
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.HuggingFaceProvider'):
            success = service.initialize_from_config({"embedding_config": legacy})
            assert success
        
        # Can also work with compatibility layer
        factory = EmbeddingFactoryCompat(legacy)
        assert factory.config.default_model_id == "minilm"
    
    def test_feature_parity(self):
        """Test that new system supports all legacy features"""
        features_to_test = [
            "multi_provider_support",
            "dynamic_model_loading",
            "thread_safety",
            "deterministic_output",
            "numpy_support",
            "context_manager",
            "model_specific_embed"
        ]
        
        factory = EmbeddingFactoryCompat({})
        
        # Multi-provider support
        factory._service.add_provider("provider1", MockEmbeddingProvider(dimension=384))
        factory._service.add_provider("provider2", MockEmbeddingProvider(dimension=512))
        assert len(factory._service.providers) == 2
        
        # Thread safety - providers have locks
        assert hasattr(factory._service, '_provider_lock')
        
        # Context manager
        with factory:
            pass  # Should work
        
        # Model-specific embed
        embedding = factory.embed(["test"], model_id="provider1", as_list=True)
        assert embedding is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "compatibility"])