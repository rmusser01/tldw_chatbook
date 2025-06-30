# Tests/DB/test_chat_image_db_compatibility.py
# Description: Database compatibility and performance tests for chat image support
#
# Imports
#
# Standard Library
import pytest
import tempfile
import time
from pathlib import Path
from io import BytesIO
import sqlite3

# 3rd-party Libraries
from PIL import Image as PILImage

# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler

#
#######################################################################################################################
#
# Test Fixtures

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    db = CharactersRAGDB(str(db_path), client_id="test_client")
    yield db
    
    # Cleanup - CharactersRAGDB doesn't have a close method, connections are thread-local
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_images():
    """Create sample images of various sizes."""
    images = []
    
    # Small image (100KB)
    small_img = PILImage.new('RGB', (200, 200), color='red')
    small_buffer = BytesIO()
    small_img.save(small_buffer, format='PNG')
    images.append(('small', small_buffer.getvalue(), 'image/png'))
    
    # Medium image (500KB)
    medium_img = PILImage.new('RGB', (800, 800), color='green')
    medium_buffer = BytesIO()
    medium_img.save(medium_buffer, format='JPEG', quality=85)
    images.append(('medium', medium_buffer.getvalue(), 'image/jpeg'))
    
    # Large image (2MB)
    large_img = PILImage.new('RGB', (2000, 2000), color='blue')
    large_buffer = BytesIO()
    large_img.save(large_buffer, format='PNG')
    images.append(('large', large_buffer.getvalue(), 'image/png'))
    
    return images


#
# Database Compatibility Tests
#

class TestDatabaseImageCompatibility:
    """Test database compatibility with image storage."""
    
    def test_database_schema_supports_images(self, temp_db):
        """Test that database schema includes image columns."""
        # Check table schema using transaction context
        with temp_db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(messages)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        # Verify image columns exist
        assert 'image_data' in columns
        assert 'image_mime_type' in columns
        
        # Verify column types
        assert columns['image_data'] == 'BLOB'
        assert columns['image_mime_type'] == 'TEXT'
    
    def test_store_and_retrieve_image(self, temp_db, sample_images):
        """Test storing and retrieving images from database."""
        _, image_data, mime_type = sample_images[0]
        
        # Create conversation
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Test Chat"})
        
        # Add message with image
        message_id = temp_db.add_message({
            "conversation_id": convo_id,
            "sender": "user",
            "content": "Check out this image",
            "image_data": image_data,
            "image_mime_type": mime_type
        })
        
        # Retrieve message
        with temp_db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content, image_data, image_mime_type FROM messages WHERE id = ?",
                (message_id,)
            )
            row = cursor.fetchone()
        
        assert row is not None
        content, retrieved_data, retrieved_mime = row
        
        assert content == "Check out this image"
        assert retrieved_data == image_data
        assert retrieved_mime == mime_type
    
    def test_messages_without_images(self, temp_db):
        """Test that messages without images work correctly."""
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Test Chat"})
        
        # Add message without image
        message_id = temp_db.add_message({
            "conversation_id": convo_id,
            "sender": "user",
            "content": "Text only message"
        })
        
        # Retrieve message
        with temp_db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content, image_data, image_mime_type FROM messages WHERE id = ?",
                (message_id,)
            )
            row = cursor.fetchone()
        
        assert row is not None
        content, image_data, mime_type = row
        
        assert content == "Text only message"
        assert image_data is None
        assert mime_type is None
    
    def test_mixed_messages_in_conversation(self, temp_db, sample_images):
        """Test conversation with mixed text and image messages."""
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Mixed Chat"})
        
        # Add various messages
        messages = []
        
        # Text only
        msg1 = temp_db.add_message({
            "conversation_id": convo_id,
            "sender": "user",
            "content": "Hello"
        })
        messages.append((msg1, False))
        
        # With image
        _, img_data, mime_type = sample_images[0]
        msg2 = temp_db.add_message({
            "conversation_id": convo_id,
            "sender": "user",
            "content": "Here's an image",
            "image_data": img_data,
            "image_mime_type": mime_type
        })
        messages.append((msg2, True))
        
        # Text only again
        msg3 = temp_db.add_message({
            "conversation_id": convo_id,
            "sender": "assistant",
            "content": "Nice image!"
        })
        messages.append((msg3, False))
        
        # Retrieve conversation
        chat_history = temp_db.get_messages_for_conversation(convo_id)
        
        assert len(chat_history) == 3
        
        for i, (msg_id, has_image) in enumerate(messages):
            msg = chat_history[i]
            if has_image:
                assert msg['image_data'] is not None
                assert msg['image_mime_type'] is not None
            else:
                assert msg['image_data'] is None
                assert msg['image_mime_type'] is None


class TestDatabaseImagePerformance:
    """Test performance with image storage."""
    
    def test_large_image_storage_performance(self, temp_db, sample_images):
        """Test performance of storing large images."""
        _, large_image_data, mime_type = sample_images[2]  # Large image
        
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Performance Test"})
        
        # Time the insertion
        start_time = time.time()
        
        message_id = temp_db.add_message({
            "conversation_id": convo_id,
            "sender": "user",
            "content": "Large image test",
            "image_data": large_image_data,
            "image_mime_type": mime_type
        })
        
        insert_time = time.time() - start_time
        
        # Should complete reasonably quickly (under 1 second)
        assert insert_time < 1.0
        assert message_id is not None
    
    def test_bulk_image_operations(self, temp_db, sample_images):
        """Test performance with multiple image messages."""
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Bulk Test"})
        
        # Insert multiple messages with images
        start_time = time.time()
        message_ids = []
        
        for i in range(10):
            _, img_data, mime_type = sample_images[i % len(sample_images)]
            msg_id = temp_db.add_message({
                "conversation_id": convo_id,
                "sender": "user",
                "content": f"Image {i}",
                "image_data": img_data,
                "image_mime_type": mime_type
            })
            message_ids.append(msg_id)
        
        bulk_insert_time = time.time() - start_time
        
        # Should handle bulk operations efficiently
        assert bulk_insert_time < 5.0  # 10 images in under 5 seconds
        assert len(message_ids) == 10
        
        # Test bulk retrieval
        start_time = time.time()
        chat_history = temp_db.get_messages_for_conversation(convo_id)
        retrieval_time = time.time() - start_time
        
        assert len(chat_history) == 10
        assert retrieval_time < 1.0  # Should retrieve quickly
    
    def test_conversation_with_many_images(self, temp_db):
        """Test conversation with many small images."""
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Many Images"})
        
        # Create many small images
        small_img = PILImage.new('RGB', (50, 50), color='yellow')
        buffer = BytesIO()
        small_img.save(buffer, format='PNG')
        small_data = buffer.getvalue()
        
        # Add 50 messages with small images
        start_time = time.time()
        
        for i in range(50):
            temp_db.add_message({
                "conversation_id": convo_id,
                "sender": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
                "image_data": small_data if i % 3 == 0 else None,
                "image_mime_type": "image/png" if i % 3 == 0 else None
            })
        
        insert_time = time.time() - start_time
        
        # Should handle many messages efficiently
        assert insert_time < 10.0
        
        # Test retrieval
        start_time = time.time()
        history = temp_db.get_messages_for_conversation(convo_id)
        retrieval_time = time.time() - start_time
        
        assert len(history) == 50
        assert retrieval_time < 2.0


class TestDatabaseImageIntegrity:
    """Test data integrity for image storage."""
    
    def test_image_data_integrity(self, temp_db):
        """Test that image data is stored and retrieved without corruption."""
        # Create image with specific pattern
        import numpy as np
        pixels = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        img = PILImage.fromarray(pixels, mode='RGB')
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        original_data = buffer.getvalue()
        
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Integrity Test"})
        
        # Store image
        msg_id = temp_db.add_message({
            "conversation_id": convo_id,
            "sender": "user",
            "content": "Test pattern",
            "image_data": original_data,
            "image_mime_type": "image/png"
        })
        
        # Retrieve and verify
        with temp_db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_data FROM messages WHERE id = ?", (msg_id,))
            retrieved_data = cursor.fetchone()[0]
        
        # Data should be identical
        assert retrieved_data == original_data
        
        # Verify image can be loaded
        retrieved_img = PILImage.open(BytesIO(retrieved_data))
        retrieved_pixels = np.array(retrieved_img)
        
        # Pixels should match
        np.testing.assert_array_equal(pixels, retrieved_pixels)
    
    def test_concurrent_image_operations(self, temp_db, sample_images):
        """Test concurrent read/write operations with images."""
        import threading
        import time
        
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Concurrent Test"})
        results = {'errors': [], 'success_count': 0}
        lock = threading.Lock()
        
        def write_messages(thread_id):
            for i in range(5):
                retry_count = 0
                max_retries = 10
                while retry_count < max_retries:
                    try:
                        _, img_data, mime_type = sample_images[i % len(sample_images)]
                        temp_db.add_message({
                            "conversation_id": convo_id,
                            "sender": f"thread_{thread_id}",
                            "content": f"Message from thread {thread_id}, msg {i}",
                            "image_data": img_data if i % 2 == 0 else None,
                            "image_mime_type": mime_type if i % 2 == 0 else None
                        })
                        with lock:
                            results['success_count'] += 1
                        break  # Success, exit retry loop
                    except Exception as e:
                        if "database is locked" in str(e):
                            retry_count += 1
                            if retry_count < max_retries:
                                time.sleep(0.1 * retry_count)  # Exponential backoff
                            else:
                                with lock:
                                    results['errors'].append(e)
                        else:
                            with lock:
                                results['errors'].append(e)
                            break  # Non-retryable error
        
        # Create multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=write_messages, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results - we should have successfully written all messages
        # even if some required retries due to database locking
        assert results['success_count'] == 15  # 3 threads * 5 messages
        
        # Verify all messages were stored
        history = temp_db.get_messages_for_conversation(convo_id)
        assert len(history) == 15  # 3 threads * 5 messages
    
    def test_image_null_handling(self, temp_db):
        """Test handling of NULL image data."""
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Null Test"})
        
        # Explicitly insert NULL values
        msg_id = temp_db._generate_uuid()
        with temp_db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, sender, content, image_data, image_mime_type, timestamp, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?, 1)
            """, (msg_id, convo_id, "user", "Null image test", temp_db.client_id))
        
        # Retrieve and verify
        with temp_db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content, image_data, image_mime_type FROM messages WHERE id = ?",
                (msg_id,)
            )
            row = cursor.fetchone()
        
        assert row[0] == "Null image test"
        assert row[1] is None
        assert row[2] is None


class TestDatabaseMigrationCompatibility:
    """Test compatibility with existing databases."""
    
    def test_existing_database_migration(self):
        """Test that new databases are created with image support."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        try:
            # Create a fresh database with CharactersRAGDB
            # This tests that the schema includes image columns from the start
            db = CharactersRAGDB(str(db_path), client_id="test_client")
            
            # Verify image columns exist in the schema
            with db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(messages)")
                columns = {row[1] for row in cursor.fetchall()}
            
            assert 'image_data' in columns
            assert 'image_mime_type' in columns
            
            # Verify we can store and retrieve images
            convo_id = db.add_conversation({"character_id": 1, "title": "Migration Test"})
            
            # Test storing a message with an image
            test_image = b'\x89PNG\r\n\x1a\n...'  # Minimal PNG header
            msg_id = db.add_message({
                "conversation_id": convo_id,
                "sender": "user",
                "content": "Test message with image",
                "image_data": test_image,
                "image_mime_type": "image/png"
            })
            
            # Retrieve and verify
            messages = db.get_messages_for_conversation(convo_id)
            assert len(messages) == 1
            assert messages[0]['image_data'] == test_image
            assert messages[0]['image_mime_type'] == "image/png"
        finally:
            if db_path.exists():
                db_path.unlink()
    
    def test_backward_compatibility(self, temp_db):
        """Test that new schema remains compatible with old queries."""
        convo_id = temp_db.add_conversation({"character_id": 1, "title": "Compat Test"})
        
        # Add messages both with and without images
        temp_db.add_message({
            "conversation_id": convo_id,
            "sender": "user",
            "content": "Old style message"
        })
        
        # Old-style query (without image columns)
        with temp_db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, conversation_id, sender, content, timestamp
                FROM messages
                WHERE conversation_id = ?
            """, (convo_id,))
        
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][3] == "Old style message"

#
#
#######################################################################################################################