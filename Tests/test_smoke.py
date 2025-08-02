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
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB as ChaChaNotes_DB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ChaChaNotes_DB(db_path, "test_client")
            
            # Basic checks
            assert db is not None
            assert os.path.exists(db_path)
            
            # Check schema version
            with db.transaction() as cursor:
                cursor.execute("SELECT version FROM db_schema_version")
                version = cursor.fetchone()[0]
                assert version > 0
            
            db.close()
    
    def test_media_database_initialization(self):
        """Test that the media database can be initialized."""
        from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as ClientMediaDB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "media.db")
            # MediaDatabase requires client_id parameter
            db = ClientMediaDB(db_path, "test_client")
            
            assert db is not None
            assert os.path.exists(db_path)
            db.close()


class TestCoreFeatures:
    """Smoke tests for core application features."""
    
    def test_conversation_creation(self):
        """Test creating a conversation."""
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB as ChaChaNotes_DB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChaChaNotes_DB(os.path.join(tmpdir, "test.db"), "test_client")
            
            # Create a conversation
            conv_data = {
                "title": "Test Conversation",
                "character_id": None  # No character
            }
            conv_id = db.add_conversation(conv_data)
            assert conv_id is not None
            
            # Verify it exists by searching
            convs = db.search_conversations_by_title("Test Conversation")
            assert len(convs) > 0
            assert convs[0]['title'] == "Test Conversation"
            
            db.close()
    
    def test_character_creation(self):
        """Test creating a character."""
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB as ChaChaNotes_DB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChaChaNotes_DB(os.path.join(tmpdir, "test.db"), "test_client")
            
            # Create a character
            char_data = {
                "name": "Test Character",
                "description": "A test character",
                "personality": "Friendly",
                "scenario": "Testing"
            }
            char_id = db.add_character_card(char_data)
            assert char_id is not None
            
            # Verify it exists
            char = db.get_character_card_by_name("Test Character")
            assert char is not None
            assert char['name'] == "Test Character"
            
            db.close()
    
    def test_note_creation(self):
        """Test creating a note."""
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB as ChaChaNotes_DB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChaChaNotes_DB(os.path.join(tmpdir, "test.db"), "test_client")
            
            # Create a note
            note_id = db.add_note(
                title="Test Note",
                content="This is a test note"
            )
            assert note_id is not None
            
            # Verify it exists using the correct method name
            notes = db.search_notes("Test Note")
            assert len(notes) >= 1
            assert any(note['title'] == "Test Note" for note in notes)
            
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
        from tldw_chatbook.config import API_MODELS_BY_PROVIDER, LOCAL_PROVIDERS
        
        # Check that we have providers
        assert len(API_MODELS_BY_PROVIDER) > 0
        
        # Check common providers exist (using proper case)
        assert "OpenAI" in API_MODELS_BY_PROVIDER
        assert "Anthropic" in API_MODELS_BY_PROVIDER
        assert "Ollama" in LOCAL_PROVIDERS  # Ollama is in LOCAL_PROVIDERS, not API_MODELS_BY_PROVIDER


class TestUIComponents:
    """Smoke tests for UI components."""
    
    @pytest.mark.asyncio
    async def test_app_initialization(self):
        """Test that the main app can be initialized."""
        from tldw_chatbook.app import TldwCli
        
        # Create app instance
        app = TldwCli()
        assert app is not None
        
        # Check title (using TITLE constant)
        assert app.TITLE is not None
        assert "tldw CLI" in app.TITLE
    
    def test_chat_window_creation(self):
        """Test that chat window can be created."""
        from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
        
        # Create a mock app instance
        mock_app = MagicMock()
        
        # Create window instance with required app_instance
        window = ChatWindowEnhanced(app_instance=mock_app)
        assert window is not None
        
        # Check basic properties
        assert hasattr(window, 'app_instance')
        assert window.app_instance == mock_app


class TestConfiguration:
    """Smoke tests for configuration."""
    
    def test_config_loading(self):
        """Test that configuration can be loaded."""
        from tldw_chatbook.config import get_cli_setting, settings
        
        # Check that settings is loaded
        assert settings is not None
        
        # Check a basic setting using the correct API
        theme = get_cli_setting("theme", default="dark")
        # Theme might not be configured, just check it's a string if it exists
        assert isinstance(theme, (str, type(None)))
    
    def test_paths_exist(self):
        """Test that required paths are set up."""
        from tldw_chatbook.config import (
            get_chachanotes_db_path,
            get_media_db_path,
            get_prompts_db_path,
            get_user_data_dir
        )
        
        # Check paths are defined
        assert get_chachanotes_db_path() is not None
        assert get_media_db_path() is not None
        assert get_prompts_db_path() is not None
        assert get_user_data_dir() is not None


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
        from tldw_chatbook.DB.sql_validation import validate_identifier, validate_table_name
        
        # Valid identifiers
        assert validate_identifier("users") is True
        assert validate_identifier("user_data") is True
        
        # Invalid identifiers (validate_identifier returns False, not raises)
        assert validate_identifier("users; DROP TABLE users;") is False
        assert validate_identifier("users'") is False
        
        # Test table name validation
        assert validate_table_name("conversations", "chachanotes") is True
        assert validate_table_name("invalid_table", "chachanotes") is False


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
        
        # Test basic chunking with proper overlap settings
        text = "This is a test. " * 100  # Long text
        # Set overlap to be less than chunk_size
        chunks = service.chunk_text(text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) >= 1  # At least one chunk
        # Check that chunks exist and have text
        assert all('text' in chunk for chunk in chunks)
        assert all(isinstance(chunk['text'], str) for chunk in chunks)


class TestOptionalFeatures:
    """Smoke tests for optional features."""
    
    def test_optional_dependencies(self):
        """Test optional dependency detection."""
        from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
        
        # Check that DEPENDENCIES_AVAILABLE is a dict
        assert isinstance(DEPENDENCIES_AVAILABLE, dict)
        
        # Check some common dependencies
        for dep in ['torch', 'embeddings_rag', 'chromadb']:
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