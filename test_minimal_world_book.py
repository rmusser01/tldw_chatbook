#!/usr/bin/env python3
"""Minimal test case to reproduce world book corruption issue."""

import sqlite3
import tempfile
import os
from pathlib import Path

# Create a minimal test database
def test_world_book_corruption():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Create database with the same schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        
        # Create simplified world_books table
        conn.execute("""
        CREATE TABLE world_books(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            enabled BOOLEAN DEFAULT 1,
            version INTEGER DEFAULT 1,
            deleted BOOLEAN DEFAULT 0
        )
        """)
        
        # Create FTS table
        conn.execute("""
        CREATE VIRTUAL TABLE world_books_fts
        USING fts5(name, description, content='world_books', content_rowid='id')
        """)
        
        # Test with simple UPDATE trigger (no WHEN clause)
        conn.execute("""
        CREATE TRIGGER world_books_au
        AFTER UPDATE ON world_books BEGIN
            UPDATE world_books_fts 
            SET name = NEW.name, description = NEW.description
            WHERE rowid = NEW.id;
        END
        """)
        
        # Insert a record
        conn.execute("INSERT INTO world_books (name, description) VALUES (?, ?)",
                    ("Original Name", "Original description"))
        conn.commit()
        
        print("Insert successful")
        
        # Try to update - this is where corruption happens
        try:
            conn.execute("UPDATE world_books SET name = ?, description = ?, version = version + 1 WHERE id = ?",
                        ("Updated Name", "New description", 1))
            conn.commit()
            print("Update successful!")
        except sqlite3.DatabaseError as e:
            print(f"Update failed with error: {e}")
            
        # Check if we can query
        try:
            rows = conn.execute("SELECT * FROM world_books").fetchall()
            print(f"Query successful, rows: {rows}")
        except sqlite3.DatabaseError as e:
            print(f"Query failed: {e}")
            
        conn.close()

if __name__ == "__main__":
    test_world_book_corruption()