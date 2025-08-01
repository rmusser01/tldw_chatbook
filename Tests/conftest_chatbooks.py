# conftest_chatbooks.py
# Additional fixtures specific to chatbook and database tools testing

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sqlite3

import sys
sys.path.append(str(Path(__file__).parent.parent))

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


@pytest.fixture
def test_databases(temp_dir):
    """Create a set of test databases with sample data."""
    db_paths = {
        'chachanotes': str(temp_dir / 'chachanotes.db'),
        'prompts': str(temp_dir / 'prompts.db'),
        'media': str(temp_dir / 'media.db')
    }
    
    # Initialize databases
    chachanotes_db = CharactersRAGDB(db_paths['chachanotes'], 'test_fixture')
    prompts_db = PromptsDatabase(db_paths['prompts'], 'test_fixture')
    media_db = MediaDatabase(db_paths['media'], 'test_fixture')
    
    # Add sample data to ChaChaNotes
    conv_id = chachanotes_db.add_conversation("Test Conversation", media_id=None)
    chachanotes_db.add_message(conv_id, "user", "Hello, world!")
    chachanotes_db.add_message(conv_id, "assistant", "Hello! How can I help you?")
    
    note_id = chachanotes_db.add_note(
        title="Test Note",
        content="This is a test note with some content.",
        keywords="test,sample"
    )
    
    char_id = chachanotes_db.create_character(
        name="Test Character",
        description="A character for testing",
        personality="Helpful and friendly",
        scenario="Testing environment",
        greeting_message="Hello!",
        example_messages="User: Hi\nAssistant: Hello!"
    )
    
    # Add sample prompts
    prompts_db.add_prompt(
        name="Test Prompt",
        author="Test Author",
        details="A prompt for testing",
        system_prompt="You are a helpful assistant.",
        user_prompt=None
    )
    
    # Return database instances and paths
    return {
        'paths': db_paths,
        'chachanotes': chachanotes_db,
        'prompts': prompts_db,
        'media': media_db,
        'sample_ids': {
            'conversation_id': conv_id,
            'note_id': note_id,
            'character_id': char_id
        }
    }


@pytest.fixture
def sample_chatbook_manifest():
    """Create a sample chatbook manifest for testing."""
    return {
        "version": "1.0",
        "name": "Test Chatbook",
        "description": "A chatbook for testing purposes",
        "author": "Test Suite",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "content_items": [
            {
                "id": "conv_1",
                "type": "conversation",
                "title": "Sample Conversation",
                "created_at": datetime.now().isoformat(),
                "file_path": "content/conversations/conversation_conv_1.json"
            },
            {
                "id": "note_1",
                "type": "note",
                "title": "Sample Note",
                "created_at": datetime.now().isoformat(),
                "tags": ["test", "sample"],
                "file_path": "content/notes/Sample Note.md"
            }
        ],
        "relationships": [
            {
                "source_id": "conv_1",
                "target_id": "char_1",
                "relationship_type": "uses_character",
                "metadata": {}
            }
        ],
        "include_media": False,
        "include_embeddings": False,
        "media_quality": "thumbnail",
        "statistics": {
            "total_conversations": 1,
            "total_notes": 1,
            "total_characters": 0,
            "total_media_items": 0,
            "total_size_bytes": 0
        },
        "tags": ["test", "sample"],
        "categories": ["testing"],
        "language": "en",
        "license": None
    }


@pytest.fixture
def create_test_database():
    """Factory fixture to create test databases with custom schema."""
    def _create_db(db_path: Path, schema_version: int = 1):
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create a simple test table
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Set schema version
        cursor.execute(f"PRAGMA user_version = {schema_version}")
        
        # Add some test data
        cursor.execute("INSERT INTO test_table (name, value) VALUES (?, ?)", 
                      ("test1", "value1"))
        cursor.execute("INSERT INTO test_table (name, value) VALUES (?, ?)", 
                      ("test2", "value2"))
        
        conn.commit()
        conn.close()
        
        return db_path
    
    return _create_db


# Test data generators
def generate_test_conversation(db: CharactersRAGDB, name: str = "Test Conv", 
                              message_count: int = 5) -> int:
    """Generate a test conversation with messages."""
    conv_id = db.add_conversation(name, media_id=None)
    
    for i in range(message_count):
        role = "user" if i % 2 == 0 else "assistant"
        message = f"Message {i} from {role}"
        db.add_message(conv_id, role, message)
    
    return conv_id


def generate_test_notes(db: CharactersRAGDB, count: int = 3) -> list:
    """Generate test notes."""
    note_ids = []
    
    for i in range(count):
        note_id = db.add_note(
            title=f"Test Note {i}",
            content=f"Content for test note {i}\n\nWith multiple lines.",
            keywords=f"test,note{i}"
        )
        note_ids.append(note_id)
    
    return note_ids


def generate_test_character(db: CharactersRAGDB, name: str = "Test Character") -> int:
    """Generate a test character."""
    return db.create_character(
        name=name,
        description=f"{name} description",
        personality="Test personality",
        scenario="Test scenario",
        greeting_message=f"Hello from {name}!",
        example_messages="User: Test\nAssistant: Response"
    )