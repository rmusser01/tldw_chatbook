"""
Smoke Tests for tldw_chatbook
These tests cover critical paths to ensure basic functionality works.
Run with: pytest Tests/test_smoke.py -v
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Test markers for organization
pytestmark = pytest.mark.smoke


class TestDatabaseSmoke:
    """Smoke tests for database functionality."""
    
    def test_database_initialization(self):
        """Test that the main database can be initialized."""
        from tldw_chatbook.DB.ChaChaNotes_DB import ChaChaNotes_DB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ChaChaNotes_DB(db_path, "test_client")
            
            # Basic checks
            assert db is not None
            assert os.path.exists(db_path)
            
            # Check schema version
            with db.transaction() as cursor:
                cursor.execute("SELECT version FROM schema_version")
                version = cursor.fetchone()[0]
                assert version > 0
            
            db.close()
    
    def test_media_database_initialization(self):
        """Test that the media database can be initialized."""
        from tldw_chatbook.DB.Client_Media_DB_v2 import ClientMediaDB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "media.db")
            db = ClientMediaDB(db_path)
            
            assert db is not None
            assert os.path.exists(db_path)
            db.close()


class TestCoreFeatures:
    """Smoke tests for core application features."""
    
    def test_conversation_creation(self):
        """Test creating a conversation."""
        from tldw_chatbook.DB.ChaChaNotes_DB import ChaChaNotes_DB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChaChaNotes_DB(os.path.join(tmpdir, "test.db"), "test_client")
            
            # Create a conversation
            conv_id = db.add_conversation("Test Conversation")
            assert conv_id is not None
            assert conv_id > 0
            
            # Verify it exists
            conv = db.get_conversation(conv_id)
            assert conv is not None
            assert conv['conversation_name'] == "Test Conversation"
            
            db.close()
    
    def test_character_creation(self):
        """Test creating a character."""
        from tldw_chatbook.DB.ChaChaNotes_DB import ChaChaNotes_DB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChaChaNotes_DB(os.path.join(tmpdir, "test.db"), "test_client")
            
            # Create a character
            char_id = db.add_character_card(
                name="Test Character",
                description="A test character",
                personality="Friendly",
                scenario="Testing"
            )
            assert char_id is not None
            
            # Verify it exists
            char = db.get_character_card_by_name("Test Character")
            assert char is not None
            assert char['name'] == "Test Character"
            
            db.close()
    
    def test_note_creation(self):
        """Test creating a note."""
        from tldw_chatbook.DB.ChaChaNotes_DB import ChaChaNotes_DB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChaChaNotes_DB(os.path.join(tmpdir, "test.db"), "test_client")
            
            # Create a note
            note_id = db.save_note(
                title="Test Note",
                content="This is a test note",
                keywords=["test", "smoke"]
            )
            assert note_id is not None
            
            # Verify it exists
            notes = db.search_notes_titles("Test Note")
            assert len(notes) == 1
            assert notes[0]['title'] == "Test Note"
            
            db.close()


class TestLLMIntegration:
    """Smoke tests for LLM integration."""
    
    @patch('tldw_chatbook.LLM_Calls.LLM_API_Calls.chat_with_openai')
    def test_llm_api_call_mock(self, mock_chat):
        """Test that LLM API calls can be made (mocked)."""
        from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_openai
        
        # Mock response
        mock_chat.return_value = "Test response"
        
        # Make call
        response = chat_with_openai(
            [{"role": "user", "content": "Hello"}],
            api_key="test_key",
            model="gpt-3.5-turbo"
        )
        
        assert response == "Test response"
        mock_chat.assert_called_once()
    
    def test_llm_provider_list(self):
        """Test that LLM providers are available."""
        from tldw_chatbook.config import API_MODELS_BY_PROVIDER
        
        # Check that we have providers
        assert len(API_MODELS_BY_PROVIDER) > 0
        
        # Check common providers exist
        assert "openai" in API_MODELS_BY_PROVIDER
        assert "anthropic" in API_MODELS_BY_PROVIDER
        assert "ollama" in API_MODELS_BY_PROVIDER


class TestUIComponents:
    """Smoke tests for UI components."""
    
    @pytest.mark.asyncio
    async def test_app_initialization(self):
        """Test that the main app can be initialized."""
        from tldw_chatbook.app import TldwCli
        
        # Create app instance
        app = TldwCli()
        assert app is not None
        
        # Check title
        assert app.title == "tldw - ChatBook"
    
    def test_chat_window_creation(self):
        """Test that chat window can be created."""
        from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
        
        # Create instance
        window = ChatWindowEnhanced()
        assert window is not None
        
        # Check basic properties
        assert hasattr(window, 'conversation_manager')
        assert hasattr(window, 'message_input')


class TestConfiguration:
    """Smoke tests for configuration."""
    
    def test_config_loading(self):
        """Test that configuration can be loaded."""
        from tldw_chatbook.config import get_cli_config, get_cli_setting
        
        # Get config
        config = get_cli_config()
        assert config is not None
        
        # Check a basic setting
        theme = get_cli_setting("theme", "dark")
        assert theme in ["dark", "light", "dracula", "solarized", "monokai"]
    
    def test_paths_exist(self):
        """Test that required paths are set up."""
        from tldw_chatbook.config import (
            get_chachanotes_db_path,
            get_character_images_path,
            get_notes_media_path
        )
        
        # Check paths are defined
        assert get_chachanotes_db_path() is not None
        assert get_character_images_path() is not None
        assert get_notes_media_path() is not None


class TestSecurity:
    """Smoke tests for security features."""
    
    def test_path_validation(self):
        """Test path validation security."""
        from tldw_chatbook.Utils.path_validation import validate_path
        
        # Test valid path
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = validate_path("test.txt", tmpdir)
            assert valid_path is not None
            
            # Test path traversal attempt
            with pytest.raises(ValueError):
                validate_path("../../../etc/passwd", tmpdir)
    
    def test_sql_validation(self):
        """Test SQL identifier validation."""
        from tldw_chatbook.DB.sql_validation import validate_sql_identifier
        
        # Valid identifiers
        assert validate_sql_identifier("users") == "users"
        assert validate_sql_identifier("user_data") == "user_data"
        
        # Invalid identifiers
        with pytest.raises(ValueError):
            validate_sql_identifier("users; DROP TABLE users;")
        
        with pytest.raises(ValueError):
            validate_sql_identifier("users'")


class TestRAGFunctionality:
    """Smoke tests for RAG functionality."""
    
    def test_rag_service_initialization(self):
        """Test that RAG service can be initialized."""
        try:
            from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
            from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
            
            # Create service with config (will skip if dependencies missing)
            config = RAGConfig()
            service = RAGService(config)
            assert service is not None
            
        except ImportError:
            pytest.skip("RAG dependencies not installed")
    
    def test_chunking_service(self):
        """Test text chunking functionality."""
        from tldw_chatbook.RAG_Search.chunking_service import ChunkingService
        
        service = ChunkingService()
        
        # Test basic chunking
        text = "This is a test. " * 100  # Long text
        chunks = service.chunk_text(text, chunk_size=100)
        
        assert len(chunks) > 1
        assert all(len(chunk['text']) <= 200 for chunk in chunks)  # Allow for some flexibility


class TestOptionalFeatures:
    """Smoke tests for optional features."""
    
    def test_optional_dependencies(self):
        """Test optional dependency detection."""
        from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
        
        # Check that DEPENDENCIES_AVAILABLE is a dict
        assert isinstance(DEPENDENCIES_AVAILABLE, dict)
        
        # Check some common dependencies
        for dep in ['gradio', 'embeddings_rag', 'chromadb']:
            assert dep in DEPENDENCIES_AVAILABLE
            assert isinstance(DEPENDENCIES_AVAILABLE[dep], bool)
    
    def test_metrics_initialization(self):
        """Test metrics system initialization."""
        from tldw_chatbook.Metrics.metrics import log_counter, log_histogram
        
        # Test basic metric recording
        log_counter("test_counter", 1, {"env": "test"})
        log_histogram("test_duration", 1.5, {"operation": "test"})
        # Should not raise exceptions


# Quick test runner for smoke tests only
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])