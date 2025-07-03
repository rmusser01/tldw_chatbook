#!/usr/bin/env python3
"""Simple test to verify the Create Note feature works"""

import sys
import time
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService

def test_create_note():
    """Test creating a note"""
    # Initialize database
    db_path = Path.home() / ".local/share/tldw_cli/default_user/tldw_chatbook_ChaChaNotes.db"
    db = CharactersRAGDB(str(db_path), "test_client")
    
    # Create notes service
    notes_service = NotesInteropService(db, "test_client")
    
    # Create a test note
    test_title = f"Test Note - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    test_content = "This is a test note created to verify the Create Note feature works."
    
    try:
        note_id = notes_service.add_note(
            user_id="test_user",
            title=test_title,
            content=test_content
        )
        
        if note_id:
            print(f"✅ Successfully created note with ID: {note_id}")
            
            # Verify we can retrieve it
            note = notes_service.get_note(note_id)
            if note and note['title'] == test_title:
                print(f"✅ Successfully retrieved note: {note['title']}")
                return True
            else:
                print("❌ Failed to retrieve note")
                return False
        else:
            print("❌ Failed to create note")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("Testing Create Note functionality...\n")
    
    if test_create_note():
        print("\n✅ Create Note feature is working!")
    else:
        print("\n❌ Create Note feature has issues")