#!/usr/bin/env python3
"""Test the exact pattern used by character_cards."""

import sqlite3
import tempfile
from pathlib import Path

def test_character_cards_pattern():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        
        # Create main table
        conn.execute("""
        CREATE TABLE test_table(
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT
        )
        """)
        
        # Create FTS table with external content
        conn.execute("""
        CREATE VIRTUAL TABLE test_fts
        USING fts5(name, description, content='test_table', content_rowid='id')
        """)
        
        # Create trigger exactly like character_cards
        conn.execute("""
        CREATE TRIGGER test_au
        AFTER UPDATE ON test_table BEGIN
            INSERT INTO test_fts(test_fts, rowid, name, description)
            VALUES('delete', OLD.id, OLD.name, OLD.description);
            
            INSERT INTO test_fts(rowid, name, description)
            VALUES(NEW.id, NEW.name, NEW.description);
        END
        """)
        
        # Insert
        conn.execute("INSERT INTO test_table (name, description) VALUES (?, ?)",
                    ("Original", "Desc"))
        # Also insert into FTS
        conn.execute("INSERT INTO test_fts (rowid, name, description) VALUES (?, ?, ?)",
                    (1, "Original", "Desc"))
        conn.commit()
        print("Insert successful")
        
        # Update
        try:
            conn.execute("UPDATE test_table SET name = ?, description = ? WHERE id = ?",
                        ("Updated", "New Desc", 1))
            conn.commit()
            print("Update successful!")
        except sqlite3.DatabaseError as e:
            print(f"Update failed: {e}")
            
        conn.close()

if __name__ == "__main__":
    test_character_cards_pattern()