# test_chatbook_performance.py
# Description: Performance tests for chatbook functionality
#
"""
Chatbook Performance Tests
--------------------------

Tests focused on performance characteristics and optimization.
"""

import pytest
import time
import json
import zipfile
import sqlite3
import tempfile
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch
import resource
import psutil
import os

from tldw_chatbook.Chatbooks import ChatbookCreator, ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, ChatbookManifest, ChatbookVersion
)
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from Tests.Chatbooks.factories import (
    CharacterFactory, ConversationFactory, NoteFactory,
    MediaFactory, PromptFactory
)


@contextmanager
def measure_time(description: str = "Operation"):
    """Context manager to measure execution time."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"{description} took {end_time - start_time:.3f} seconds")


@contextmanager
def measure_memory(description: str = "Operation"):
    """Context manager to measure memory usage."""
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    yield
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"{description} used {end_memory - start_memory:.2f} MB")


class PerformanceMetrics:
    """Helper class to collect performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'times': [],
            'memory': [],
            'sizes': []
        }
    
    def add_time(self, operation: str, duration: float):
        """Add time measurement."""
        self.metrics['times'].append({
            'operation': operation,
            'duration': duration
        })
    
    def add_memory(self, operation: str, memory_mb: float):
        """Add memory measurement."""
        self.metrics['memory'].append({
            'operation': operation,
            'memory_mb': memory_mb
        })
    
    def add_size(self, description: str, size_bytes: int):
        """Add size measurement."""
        self.metrics['sizes'].append({
            'description': description,
            'size_bytes': size_bytes,
            'size_mb': size_bytes / 1024 / 1024
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_time': sum(m['duration'] for m in self.metrics['times']),
            'max_memory': max((m['memory_mb'] for m in self.metrics['memory']), default=0),
            'total_size_mb': sum(m['size_mb'] for m in self.metrics['sizes'])
        }


@pytest.mark.performance
class TestChatbookPerformance:
    """Performance tests for chatbook operations."""
    
    @pytest.fixture
    def performance_db_setup(self, tmp_path):
        """Setup databases for performance testing."""
        db_dir = tmp_path / "perf_dbs"
        db_dir.mkdir()
        
        db_paths = {
            'ChaChaNotes': str(db_dir / "ChaChaNotes.db"),
            'Media': str(db_dir / "Media.db"),
            'Prompts': str(db_dir / "Prompts.db")
        }
        
        # Initialize databases
        chacha_db = CharactersRAGDB(db_paths['ChaChaNotes'], "perf_test")
        media_db = MediaDatabase(db_paths['Media'], "perf_test")
        prompts_db = PromptsDatabase(db_paths['Prompts'], "perf_test")
        
        return {
            'db_paths': db_paths,
            'chacha_db': chacha_db,
            'media_db': media_db,
            'prompts_db': prompts_db
        }
    
    def populate_large_dataset(self, db_setup: Dict[str, Any], size: str = "medium") -> Dict[str, List[int]]:
        """Populate databases with test data of various sizes."""
        sizes = {
            'small': {'conversations': 10, 'notes': 20, 'characters': 5},
            'medium': {'conversations': 100, 'notes': 200, 'characters': 20},
            'large': {'conversations': 1000, 'notes': 2000, 'characters': 100},
            'xlarge': {'conversations': 5000, 'notes': 10000, 'characters': 500}
        }
        
        config = sizes.get(size, sizes['medium'])
        chacha_db = db_setup['chacha_db']
        
        ids = {
            'character_ids': [],
            'conversation_ids': [],
            'note_ids': [],
            'media_ids': [],
            'prompt_ids': []
        }
        
        # Add characters
        for i in range(config['characters']):
            char = CharacterFactory.create(name=f"Character {i+1}")
            char_id = chacha_db.add_character_card(char)
            if char_id:
                ids['character_ids'].append(char_id)
        
        # Add conversations with messages
        for i in range(config['conversations']):
            char_id = ids['character_ids'][i % len(ids['character_ids'])]
            conv = ConversationFactory.create(
                name=f"Conversation {i+1}",
                character_id=char_id,
                message_count=20  # 20 messages per conversation
            )
            conv_id = chacha_db.add_conversation({
                'conversation_name': conv['conversation_name'],
                'character_id': char_id
            })
            if conv_id:
                ids['conversation_ids'].append(conv_id)
                # Add messages
                for msg in conv['messages']:
                    chacha_db.add_message({
                        'conversation_id': conv_id,
                        'sender': msg['role'],
                        'content': msg['content']
                    })
        
        # Add notes
        for i in range(config['notes']):
            note = NoteFactory.create(
                title=f"Note {i+1}",
                content=f"Content for note {i+1}\n" * 50  # Larger content
            )
            note_id = chacha_db.add_note(
                title=note['title'],
                content=note['content']
            )
            if note_id:
                ids['note_ids'].append(note_id)
        
        return ids
    
    def test_large_export_performance(self, performance_db_setup, tmp_path):
        """Test performance of exporting large chatbooks."""
        metrics = PerformanceMetrics()
        
        # Populate with large dataset
        with measure_time("Database population"):
            ids = self.populate_large_dataset(performance_db_setup, size="large")
        
        # Create chatbook
        creator = ChatbookCreator(performance_db_setup['db_paths'])
        output_path = tmp_path / "large_export.zip"
        
        content_selections = {
            ContentType.CONVERSATION: [str(id) for id in ids['conversation_ids'][:500]],
            ContentType.NOTE: [str(id) for id in ids['note_ids'][:1000]],
            ContentType.CHARACTER: [str(id) for id in ids['character_ids'][:50]]
        }
        
        start_time = time.perf_counter()
        success, message, dep_info = creator.create_chatbook(
            name="Large Performance Test",
            description="Testing export performance with large dataset",
            content_selections=content_selections,
            output_path=output_path,
            include_media=False  # Skip media for speed
        )
        export_time = time.perf_counter() - start_time
        
        assert success is True
        metrics.add_time("Large export", export_time)
        
        # Check file size
        file_size = output_path.stat().st_size
        metrics.add_size("Large chatbook", file_size)
        
        # Performance assertions
        assert export_time < 30.0  # Should complete within 30 seconds
        assert file_size < 100 * 1024 * 1024  # Should be less than 100MB
        
        print(f"Export metrics: {metrics.get_summary()}")
    
    def test_import_performance_with_conflicts(self, performance_db_setup, tmp_path):
        """Test import performance when handling conflicts."""
        # First create a chatbook
        ids = self.populate_large_dataset(performance_db_setup, size="small")
        
        creator = ChatbookCreator(performance_db_setup['db_paths'])
        export_path = tmp_path / "conflict_test.zip"
        
        content_selections = {
            ContentType.NOTE: [str(id) for id in ids['note_ids'][:10]]
        }
        
        success, _, _ = creator.create_chatbook(
            name="Conflict Test",
            description="Testing import conflicts",
            content_selections=content_selections,
            output_path=export_path
        )
        
        assert success is True
        
        # Import once to establish baseline
        importer = ChatbookImporter(performance_db_setup['db_paths'])
        success, _ = importer.import_chatbook(
            chatbook_path=export_path,
            conflict_resolution=ConflictResolution.SKIP
        )
        assert success is True
        
        # Import again with different conflict resolutions
        strategies = [
            ConflictResolution.SKIP,
            ConflictResolution.RENAME,
            ConflictResolution.REPLACE
        ]
        
        for strategy in strategies:
            start_time = time.perf_counter()
            success, _ = importer.import_chatbook(
                chatbook_path=export_path,
                conflict_resolution=strategy
            )
            import_time = time.perf_counter() - start_time
            
            assert success is True
            assert import_time < 5.0  # Should handle conflicts quickly
            print(f"{strategy.name} import took {import_time:.3f}s")
    
    def test_concurrent_operations(self, performance_db_setup, tmp_path):
        """Test concurrent chatbook operations."""
        # Populate small dataset
        ids = self.populate_large_dataset(performance_db_setup, size="small")
        
        def create_chatbook(index: int) -> Tuple[bool, float]:
            """Create a chatbook and return success status and time."""
            creator = ChatbookCreator(performance_db_setup['db_paths'])
            output_path = tmp_path / f"concurrent_{index}.zip"
            
            content_selections = {
                ContentType.NOTE: [str(ids['note_ids'][index % len(ids['note_ids'])])]
            }
            
            start_time = time.perf_counter()
            success, _, _ = creator.create_chatbook(
                name=f"Concurrent Test {index}",
                description="Testing concurrent operations",
                content_selections=content_selections,
                output_path=output_path
            )
            duration = time.perf_counter() - start_time
            
            return success, duration
        
        # Run concurrent exports
        num_concurrent = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            start_time = time.perf_counter()
            futures = [executor.submit(create_chatbook, i) for i in range(num_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            total_time = time.perf_counter() - start_time
        
        # All should succeed
        assert all(success for success, _ in results)
        
        # Should complete reasonably fast
        assert total_time < 10.0
        
        individual_times = [duration for _, duration in results]
        print(f"Concurrent exports: total={total_time:.3f}s, "
              f"avg individual={sum(individual_times)/len(individual_times):.3f}s")
    
    def test_memory_usage_large_export(self, performance_db_setup, tmp_path):
        """Test memory usage during large exports."""
        process = psutil.Process(os.getpid())
        
        # Populate medium dataset
        ids = self.populate_large_dataset(performance_db_setup, size="medium")
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        creator = ChatbookCreator(performance_db_setup['db_paths'])
        output_path = tmp_path / "memory_test.zip"
        
        content_selections = {
            ContentType.CONVERSATION: [str(id) for id in ids['conversation_ids']],
            ContentType.NOTE: [str(id) for id in ids['note_ids']]
        }
        
        # Monitor memory during export
        max_memory = baseline_memory
        
        def monitor_memory():
            nonlocal max_memory
            while getattr(monitor_memory, 'running', True):
                current_memory = process.memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                time.sleep(0.1)
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_memory.running = True
        monitor_thread.start()
        
        # Perform export
        success, _, _ = creator.create_chatbook(
            name="Memory Test",
            description="Testing memory usage",
            content_selections=content_selections,
            output_path=output_path
        )
        
        # Stop monitoring
        monitor_memory.running = False
        monitor_thread.join()
        
        assert success is True
        
        memory_increase = max_memory - baseline_memory
        print(f"Memory usage: baseline={baseline_memory:.2f}MB, "
              f"max={max_memory:.2f}MB, increase={memory_increase:.2f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
    
    def test_zip_compression_efficiency(self, performance_db_setup, tmp_path):
        """Test ZIP compression efficiency for different content types."""
        # Create different types of content
        chacha_db = performance_db_setup['chacha_db']
        
        # Add character
        char = CharacterFactory.create()
        char_id = chacha_db.add_character_card(char)
        
        # Add text-heavy content (compresses well)
        text_note_ids = []
        for i in range(10):
            note_id = chacha_db.add_note(
                title=f"Text Note {i}",
                content="Lorem ipsum dolor sit amet, " * 1000  # Repetitive text
            )
            if note_id:
                text_note_ids.append(str(note_id))
        
        # Add random content (compresses poorly)
        random_note_ids = []
        for i in range(10):
            import random
            import string
            random_content = ''.join(random.choices(string.ascii_letters + string.digits, k=10000))
            note_id = chacha_db.add_note(
                title=f"Random Note {i}",
                content=random_content
            )
            if note_id:
                random_note_ids.append(str(note_id))
        
        # Test compression for each type
        creator = ChatbookCreator(performance_db_setup['db_paths'])
        
        # Text-heavy chatbook
        text_output = tmp_path / "text_heavy.zip"
        success, _, _ = creator.create_chatbook(
            name="Text Heavy",
            description="Repetitive text content",
            content_selections={ContentType.NOTE: text_note_ids},
            output_path=text_output
        )
        assert success is True
        text_size = text_output.stat().st_size
        
        # Random content chatbook
        random_output = tmp_path / "random_content.zip"
        success, _, _ = creator.create_chatbook(
            name="Random Content",
            description="Random content",
            content_selections={ContentType.NOTE: random_note_ids},
            output_path=random_output
        )
        assert success is True
        random_size = random_output.stat().st_size
        
        # Text should compress much better
        compression_ratio = text_size / random_size
        print(f"Compression efficiency: text={text_size/1024:.2f}KB, "
              f"random={random_size/1024:.2f}KB, ratio={compression_ratio:.2f}")
        
        assert compression_ratio < 0.5  # Text should be less than half the size
    
    def test_incremental_import_performance(self, performance_db_setup, tmp_path):
        """Test performance of importing chatbooks incrementally."""
        # Create multiple small chatbooks
        chacha_db = performance_db_setup['chacha_db']
        creator = ChatbookCreator(performance_db_setup['db_paths'])
        
        chatbook_files = []
        for i in range(10):
            # Add a few notes
            note_ids = []
            for j in range(5):
                note_id = chacha_db.add_note(
                    title=f"Batch {i} Note {j}",
                    content=f"Content for batch {i} note {j}"
                )
                if note_id:
                    note_ids.append(str(note_id))
            
            output_path = tmp_path / f"batch_{i}.zip"
            success, _, _ = creator.create_chatbook(
                name=f"Batch {i}",
                description=f"Incremental test batch {i}",
                content_selections={ContentType.NOTE: note_ids},
                output_path=output_path
            )
            assert success is True
            chatbook_files.append(output_path)
        
        # Import incrementally and measure performance
        importer = ChatbookImporter(performance_db_setup['db_paths'])
        import_times = []
        
        for i, chatbook_file in enumerate(chatbook_files):
            start_time = time.perf_counter()
            success, _ = importer.import_chatbook(
                chatbook_path=chatbook_file,
                conflict_resolution=ConflictResolution.SKIP
            )
            import_time = time.perf_counter() - start_time
            
            assert success is True
            import_times.append(import_time)
        
        # Import times should remain consistent
        avg_time = sum(import_times) / len(import_times)
        max_deviation = max(abs(t - avg_time) for t in import_times)
        
        print(f"Incremental imports: times={[f'{t:.3f}' for t in import_times]}, "
              f"avg={avg_time:.3f}s, max_deviation={max_deviation:.3f}s")
        
        # Performance should not degrade significantly
        assert max_deviation < avg_time * 0.5  # Within 50% of average
    
    @pytest.mark.slow
    def test_stress_test_extreme_scale(self, performance_db_setup, tmp_path):
        """Stress test with extreme scale (marked as slow)."""
        # Only run if explicitly requested
        metrics = PerformanceMetrics()
        
        # Create extreme dataset
        chacha_db = performance_db_setup['chacha_db']
        
        # Add 10,000 notes in batches
        note_ids = []
        batch_size = 100
        
        for batch in range(100):
            with chacha_db.transaction():
                for i in range(batch_size):
                    note_num = batch * batch_size + i
                    note_id = chacha_db.add_note(
                        title=f"Stress Note {note_num}",
                        content=f"Content {note_num}\n" * 10
                    )
                    if note_id:
                        note_ids.append(str(note_id))
        
        # Export all
        creator = ChatbookCreator(performance_db_setup['db_paths'])
        output_path = tmp_path / "stress_test.zip"
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        success, _, _ = creator.create_chatbook(
            name="Stress Test",
            description="Extreme scale test",
            content_selections={ContentType.NOTE: note_ids},
            output_path=output_path
        )
        
        export_time = time.perf_counter() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        assert success is True
        
        file_size = output_path.stat().st_size
        print(f"Stress test: {len(note_ids)} notes, "
              f"time={export_time:.2f}s, "
              f"size={file_size/1024/1024:.2f}MB, "
              f"memory_used={end_memory-start_memory:.2f}MB")
        
        # Should complete even with extreme scale
        assert export_time < 120.0  # Within 2 minutes
        assert file_size < 500 * 1024 * 1024  # Less than 500MB