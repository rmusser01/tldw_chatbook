# conftest.py
# Description: Shared fixtures and utilities for chatbook tests
#
"""
Chatbook Test Configuration
---------------------------

Provides shared fixtures, utilities, and test data for chatbook testing.
"""

import pytest
import tempfile
import shutil
import sqlite3
import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Chatbooks import ChatbookCreator, ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest, ChatbookVersion, ContentType, ContentItem
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_db_paths(temp_dir):
    """Create paths for test databases."""
    return {
        "ChaChaNotes": str(temp_dir / "test_chachanotes.db"),
        "Prompts": str(temp_dir / "test_prompts.db"),
        "Media": str(temp_dir / "test_media.db")
    }


@pytest.fixture
def memory_db_paths():
    """Create in-memory database paths for faster tests."""
    return {
        "ChaChaNotes": ":memory:",
        "Prompts": ":memory:",
        "Media": ":memory:"
    }


@pytest.fixture
def populated_chachanotes_db(mock_db_paths):
    """Create a populated ChaChaNotes database."""
    db = CharactersRAGDB(mock_db_paths["ChaChaNotes"], "test")
    
    # Add test characters
    char1_id = db.add_character_card({
        "name": "Test Character 1",
        "description": "A test character",
        "personality": "Friendly and helpful",
        "scenario": "Testing chatbooks",
        "first_message": "Hello! I'm a test character.",
        "tags": ["test", "chatbook"],
        "creator": "Test Suite"
    })
    
    char2_id = db.add_character_card({
        "name": "Test Character 2",
        "description": "Another test character",
        "personality": "Curious and analytical",
        "scenario": "Testing imports",
        "first_message": "Greetings! Ready to test?",
        "tags": ["test", "import"],
        "creator": "Test Suite"
    })
    
    # Add test conversations
    conv1_id = db.add_conversation({
        "conversation_name": "Test Conversation 1",
        "character_id": char1_id
    })
    
    conv2_id = db.add_conversation({
        "conversation_name": "Test Conversation 2",
        "character_id": char2_id
    })
    
    # Add messages to conversations
    db.add_message({
        "conversation_id": conv1_id,
        "sender": "user",
        "content": "Hello, how are you?",
        "timestamp": datetime.now().isoformat()
    })
    
    db.add_message({
        "conversation_id": conv1_id,
        "sender": "assistant",
        "content": "I'm doing great! How can I help you test chatbooks?",
        "timestamp": datetime.now().isoformat()
    })
    
    # Add test notes
    note1_id = db.add_note(
        title="Test Note 1",
        content="This is a test note for chatbook testing.\n\nIt has multiple paragraphs."
    )
    
    note2_id = db.add_note(
        title="Test Note 2", 
        content="Another test note with **markdown** formatting."
    )
    
    return {
        "db": db,
        "character_ids": [char1_id, char2_id],
        "conversation_ids": [conv1_id, conv2_id],
        "note_ids": [note1_id, note2_id]
    }


@pytest.fixture
def populated_prompts_db(mock_db_paths):
    """Create a populated Prompts database."""
    db = PromptsDatabase(mock_db_paths["Prompts"], "test")
    
    # Add test prompts
    prompt1_id, _, _ = db.add_prompt(
        name="Test Prompt 1",
        author="Test Suite",
        details="A test prompt for chatbooks",
        system_prompt="You are a helpful test assistant.",
        user_prompt="Help me test chatbooks"
    )
    
    prompt2_id, _, _ = db.add_prompt(
        name="Test Prompt 2",
        author="Test Suite",
        details="Another test prompt",
        system_prompt="You are an analytical test bot.",
        user_prompt="Analyze this test"
    )
    
    return {
        "db": db,
        "prompt_ids": [prompt1_id, prompt2_id]
    }


@pytest.fixture
def populated_media_db(mock_db_paths):
    """Create a populated Media database."""
    db = MediaDatabase(mock_db_paths["Media"], "test")
    
    # Add test media
    media1_id = db.add_media_with_keywords(
        url="https://example.com/video1.mp4",
        title="Test Video 1",
        media_type="video",
        content="This is a transcript of test video 1.",
        media_keywords="test, video, chatbook",
        summary="A test video for chatbook testing",
        transcription_model="test_model",
        author="Test Suite",
        ingestion_date=datetime.now().isoformat()
    )
    
    media2_id = db.add_media_with_keywords(
        url="https://example.com/audio1.mp3",
        title="Test Audio 1",
        media_type="audio",
        content="This is a transcript of test audio 1.",
        media_keywords="test, audio, import",
        summary="A test audio file",
        transcription_model="test_model",
        author="Test Suite",
        ingestion_date=datetime.now().isoformat()
    )
    
    return {
        "db": db,
        "media_ids": [media1_id, media2_id]
    }


@pytest.fixture
def chatbook_creator(mock_db_paths):
    """Create a ChatbookCreator instance."""
    return ChatbookCreator(mock_db_paths)


@pytest.fixture
def chatbook_importer(mock_db_paths):
    """Create a ChatbookImporter instance."""
    return ChatbookImporter(mock_db_paths)


@pytest.fixture
def sample_manifest():
    """Create a sample chatbook manifest."""
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Test Chatbook",
        description="A test chatbook for testing",
        author="Test Suite",
        tags=["test", "sample"],
        categories=["testing"]
    )
    
    # Add sample content items
    manifest.content_items.extend([
        ContentItem(
            id="conv1",
            type=ContentType.CONVERSATION,
            title="Test Conversation",
            description="A test conversation",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_path="content/conversations/conversation_conv1.json"
        ),
        ContentItem(
            id="note1",
            type=ContentType.NOTE,
            title="Test Note",
            description="A test note",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_path="content/notes/Test Note.md"
        ),
        ContentItem(
            id="char1",
            type=ContentType.CHARACTER,
            title="Test Character",
            description="A test character",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_path="content/characters/character_char1.json"
        )
    ])
    
    # Update statistics
    manifest.total_conversations = 1
    manifest.total_notes = 1
    manifest.total_characters = 1
    
    return manifest


@pytest.fixture
def sample_chatbook_zip(temp_dir, sample_manifest):
    """Create a sample chatbook ZIP file."""
    chatbook_path = temp_dir / "test_chatbook.zip"
    
    with zipfile.ZipFile(chatbook_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Write manifest
        zf.writestr("manifest.json", json.dumps(sample_manifest.to_dict(), indent=2))
        
        # Write README
        readme_content = f"""# {sample_manifest.name}

{sample_manifest.description}

**Author:** {sample_manifest.author}
**Created:** {sample_manifest.created_at.strftime('%Y-%m-%d %H:%M')}

## Contents

- **Conversations:** 1
- **Notes:** 1
- **Characters:** 1
"""
        zf.writestr("README.md", readme_content)
        
        # Write sample conversation
        conv_data = {
            "id": "conv1",
            "name": "Test Conversation",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "character_id": 1,
            "messages": [
                {
                    "id": "msg1",
                    "role": "user",
                    "content": "Hello!",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "content": "Hi there!",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        zf.writestr("content/conversations/conversation_conv1.json", 
                   json.dumps(conv_data, indent=2))
        
        # Write sample note
        note_content = """---
id: note1
title: Test Note
created_at: 2024-01-01T00:00:00
updated_at: 2024-01-01T00:00:00
tags: test, sample
---

# Test Note

This is a test note for the chatbook."""
        zf.writestr("content/notes/Test Note.md", note_content)
        
        # Write sample character
        char_data = {
            "id": "char1",
            "name": "Test Character",
            "description": "A test character",
            "personality": "Helpful",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        zf.writestr("content/characters/character_char1.json",
                   json.dumps(char_data, indent=2))
    
    return chatbook_path


@pytest.fixture
def mock_app_config():
    """Create a mock app configuration."""
    return {
        "database": {
            "chachanotes_db_path": "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db",
            "prompts_db_path": "~/.local/share/tldw_cli/tldw_prompts.db", 
            "media_db_path": "~/.local/share/tldw_cli/media_db_v2.db"
        },
        "chatbooks": {
            "export_directory": "~/Documents/Chatbooks",
            "auto_include_dependencies": True,
            "default_media_quality": "thumbnail"
        }
    }


class MockWizardApp:
    """Mock wizard app for UI testing."""
    def __init__(self, config_data):
        self.config_data = config_data
        self.notifications = []
        
    def notify(self, message, severity="info"):
        """Mock notify method."""
        self.notifications.append({"message": message, "severity": severity})