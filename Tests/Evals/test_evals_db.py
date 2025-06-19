# test_evals_db.py
# Description: Unit tests for Evals_DB module
#
"""
Unit Tests for Evaluation Database
==================================

Tests the EvalsDB class functionality including:
- Database initialization and schema creation
- CRUD operations for all entities
- Foreign key constraints and data integrity
- Full-text search capabilities
- Error handling and validation
- Concurrent access and thread safety
"""

import pytest
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tldw_chatbook.DB.Evals_DB import EvalsDB, EvalsDBError, SchemaError, InputError, ConflictError

class TestEvalsDBInitialization:
    """Test database initialization and schema creation."""
    
    def test_memory_db_initialization(self):
        """Test in-memory database initialization."""
        db = EvalsDB(db_path=":memory:", client_id="test")
        assert db.db_path == ":memory:"
        assert db.client_id == "test"
    
    def test_file_db_initialization(self, temp_db_path):
        """Test file-based database initialization."""
        db = EvalsDB(db_path=temp_db_path, client_id="test")
        # db_path is a Path object for file-based databases
        assert str(db.db_path) == temp_db_path
        assert Path(temp_db_path).exists()
    
    def test_schema_creation(self, in_memory_db):
        """Test that all required tables are created."""
        tables = in_memory_db.get_connection().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        
        table_names = [table[0] for table in tables]
        expected_tables = [
            'eval_tasks', 'eval_datasets', 'eval_models', 
            'eval_runs', 'eval_results', 'eval_run_metrics'
        ]
        
        for table in expected_tables:
            assert table in table_names
    
    def test_fts_tables_creation(self, in_memory_db):
        """Test that FTS5 virtual tables are created."""
        tables = in_memory_db.get_connection().execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'"
        ).fetchall()
        
        fts_tables = [table[0] for table in tables]
        assert 'eval_tasks_fts' in fts_tables
        assert 'eval_datasets_fts' in fts_tables

class TestTaskOperations:
    """Test CRUD operations for evaluation tasks."""
    
    def test_create_task_basic(self, in_memory_db):
        """Test basic task creation."""
        task_id = in_memory_db.create_task(
            name="test_task",
            description="A test task",
            task_type="question_answer",
            config_format="custom",
            config_data={"test": "data"}
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
    
    def test_create_task_with_dataset(self, in_memory_db):
        """Test task creation with dataset reference."""
        # First create a dataset
        dataset_id = in_memory_db.create_dataset(
            name="test_dataset",
            format="json",
            source_path="/path/to/data",
            metadata={"format": "json"}
        )
        
        # Then create task with dataset
        task_id = in_memory_db.create_task(
            name="test_task",
            description="A test task",
            task_type="question_answer",
            config_format="custom",
            config_data={"test": "data"},
            dataset_id=dataset_id
        )
        
        task = in_memory_db.get_task(task_id)
        assert task['dataset_id'] == dataset_id
    
    def test_get_task(self, in_memory_db):
        """Test task retrieval."""
        task_id = in_memory_db.create_task(
            name="test_task",
            description="A test task",
            task_type="question_answer",
            config_format="custom",
            config_data={"test": "data"}
        )
        
        task = in_memory_db.get_task(task_id)
        assert task['id'] == task_id
        assert task['name'] == "test_task"
        assert task['task_type'] == "question_answer"
    
    def test_get_nonexistent_task(self, in_memory_db):
        """Test retrieval of non-existent task."""
        task = in_memory_db.get_task("nonexistent_id")
        assert task is None
    
    def test_list_tasks(self, in_memory_db):
        """Test listing tasks."""
        # Create multiple tasks
        task_ids = []
        for i in range(3):
            task_id = in_memory_db.create_task(
                name=f"test_task_{i}",
                description=f"Test task {i}",
                task_type="question_answer",
                config_format="custom",
                config_data={"index": i}
            )
            task_ids.append(task_id)
        
        tasks = in_memory_db.list_tasks()
        assert len(tasks) == 3
        
        retrieved_ids = [task['id'] for task in tasks]
        for task_id in task_ids:
            assert task_id in retrieved_ids
    
    def test_update_task(self, in_memory_db):
        """Test task updates."""
        task_id = in_memory_db.create_task(
            name="test_task",
            description="Original description",
            task_type="question_answer",
            config_format="custom",
            config_data={"version": 1}
        )
        
        # Update task
        success = in_memory_db.update_task(
            task_id,
            description="Updated description",
            config_data={"version": 2}
        )
        
        assert success
        
        # Verify update
        task = in_memory_db.get_task(task_id)
        assert task['description'] == "Updated description"
        assert task['config_data']['version'] == 2
    
    def test_delete_task(self, in_memory_db):
        """Test task deletion (soft delete)."""
        task_id = in_memory_db.create_task(
            name="test_task",
            description="A test task",
            task_type="question_answer",
            config_format="custom",
            config_data={"test": "data"}
        )
        
        # Delete task
        success = in_memory_db.delete_task(task_id)
        assert success
        
        # Verify task is marked as deleted
        task = in_memory_db.get_task(task_id, include_deleted=True)
        assert task is not None
        assert task['deleted_at'] is not None
        
        # Verify task is not returned in normal queries
        task = in_memory_db.get_task(task_id)
        assert task is None

class TestDatasetOperations:
    """Test CRUD operations for datasets."""
    
    def test_create_dataset(self, in_memory_db):
        """Test dataset creation."""
        dataset_id = in_memory_db.create_dataset(
            name="test_dataset",
            format="huggingface",
            source_path="test/dataset",
            metadata={"split": "test", "config": "default"}
        )
        
        assert dataset_id is not None
        assert isinstance(dataset_id, str)
    
    def test_get_dataset(self, in_memory_db):
        """Test dataset retrieval."""
        dataset_id = in_memory_db.create_dataset(
            name="test_dataset",
            format="custom",
            source_path="/path/to/data",
            metadata={"format": "csv"}
        )
        
        dataset = in_memory_db.get_dataset(dataset_id)
        assert dataset['id'] == dataset_id
        assert dataset['name'] == "test_dataset"
        assert dataset['format'] == "local"
    
    def test_list_datasets(self, in_memory_db):
        """Test listing datasets."""
        dataset_ids = []
        for i in range(2):
            dataset_id = in_memory_db.create_dataset(
                name=f"dataset_{i}",
                format="custom",
                source_path=f"/path/to/data_{i}",
                metadata={"index": i}
            )
            dataset_ids.append(dataset_id)
        
        datasets = in_memory_db.list_datasets()
        assert len(datasets) == 2
        
        retrieved_ids = [dataset['id'] for dataset in datasets]
        for dataset_id in dataset_ids:
            assert dataset_id in retrieved_ids

class TestModelOperations:
    """Test CRUD operations for model configurations."""
    
    def test_create_model(self, in_memory_db):
        """Test model configuration creation."""
        model_id = in_memory_db.create_model(
            name="GPT-4",
            provider="openai",
            model_id="gpt-4",
            config={"temperature": 0.0, "max_tokens": 100}
        )
        
        assert model_id is not None
        assert isinstance(model_id, str)
    
    def test_get_model(self, in_memory_db):
        """Test model configuration retrieval."""
        model_id = in_memory_db.create_model(
            name="Claude-3",
            provider="anthropic",
            model_id="claude-3-sonnet",
            config={"temperature": 0.5}
        )
        
        model = in_memory_db.get_model(model_id)
        assert model['id'] == model_id
        assert model['name'] == "Claude-3"
        assert model['provider'] == "anthropic"

class TestRunOperations:
    """Test CRUD operations for evaluation runs."""
    
    def test_create_run(self, in_memory_db):
        """Test evaluation run creation."""
        # Create dependencies
        task_id = in_memory_db.create_task(
            name="test_task", description="Test", task_type="question_answer",
            config_format="custom", config_data={}
        )
        model_id = in_memory_db.create_model(
            name="Test Model", provider="test", model_id="test-1", config={}
        )
        
        run_id = in_memory_db.create_run(
            task_id=task_id,
            model_id=model_id,
            config={"max_samples": 100}
        )
        
        assert run_id is not None
        assert isinstance(run_id, str)
    
    def test_get_run(self, in_memory_db):
        """Test evaluation run retrieval."""
        # Create dependencies
        task_id = in_memory_db.create_task(
            name="test_task", description="Test", task_type="question_answer",
            config_format="custom", config_data={}
        )
        model_id = in_memory_db.create_model(
            name="Test Model", provider="test", model_id="test-1", config={}
        )
        
        run_id = in_memory_db.create_run(
            task_id=task_id,
            model_id=model_id,
            config={"max_samples": 100}
        )
        
        run = in_memory_db.get_run(run_id)
        assert run['id'] == run_id
        assert run['task_id'] == task_id
        assert run['model_id'] == model_id
    
    def test_update_run_status(self, in_memory_db):
        """Test updating run status."""
        # Create dependencies and run
        task_id = in_memory_db.create_task(
            name="test_task", description="Test", task_type="question_answer",
            config_format="custom", config_data={}
        )
        model_id = in_memory_db.create_model(
            name="Test Model", provider="test", model_id="test-1", config={}
        )
        run_id = in_memory_db.create_run(
            task_id=task_id, model_id=model_id, config={}
        )
        
        # Update status
        success = in_memory_db.update_run_status(run_id, "running", progress=0.5)
        assert success
        
        # Verify update
        run = in_memory_db.get_run(run_id)
        assert run['status'] == "running"
        assert run['progress'] == 0.5

class TestResultOperations:
    """Test operations for evaluation results."""
    
    def test_store_result(self, in_memory_db):
        """Test storing evaluation results."""
        # Create dependencies
        task_id = in_memory_db.create_task(
            name="test_task", description="Test", task_type="question_answer",
            config_format="custom", config_data={}
        )
        model_id = in_memory_db.create_model(
            name="Test Model", provider="test", model_id="test-1", config={}
        )
        run_id = in_memory_db.create_run(
            task_id=task_id, model_id=model_id, config={}
        )
        
        # Store result
        result_id = in_memory_db.store_result(
            run_id=run_id,
            sample_id="sample_1",
            input_text="What is 2+2?",
            expected_output="4",
            model_output="4",
            metrics={"exact_match": 1.0},
            metadata={"execution_time": 0.5}
        )
        
        assert result_id is not None
        assert isinstance(result_id, str)
    
    def test_get_results_for_run(self, in_memory_db):
        """Test retrieving results for a run."""
        # Create dependencies and run
        task_id = in_memory_db.create_task(
            name="test_task", description="Test", task_type="question_answer",
            config_format="custom", config_data={}
        )
        model_id = in_memory_db.create_model(
            name="Test Model", provider="test", model_id="test-1", config={}
        )
        run_id = in_memory_db.create_run(
            task_id=task_id, model_id=model_id, config={}
        )
        
        # Store multiple results
        for i in range(3):
            in_memory_db.store_result(
                run_id=run_id,
                sample_id=f"sample_{i}",
                input_text=f"Question {i}",
                expected_output=f"Answer {i}",
                model_output=f"Response {i}",
                metrics={"score": i * 0.3},
                metadata={"index": i}
            )
        
        # Retrieve results
        results = in_memory_db.get_results_for_run(run_id)
        assert len(results) == 3
        
        # Verify ordering and content
        for i, result in enumerate(results):
            assert result['sample_id'] == f"sample_{i}"
            assert result['metrics']['score'] == i * 0.3

class TestMetricsOperations:
    """Test operations for run metrics."""
    
    def test_store_run_metrics(self, in_memory_db):
        """Test storing run-level metrics."""
        # Create dependencies and run
        task_id = in_memory_db.create_task(
            name="test_task", description="Test", task_type="question_answer",
            config_format="custom", config_data={}
        )
        model_id = in_memory_db.create_model(
            name="Test Model", provider="test", model_id="test-1", config={}
        )
        run_id = in_memory_db.create_run(
            task_id=task_id, model_id=model_id, config={}
        )
        
        # Store metrics
        metrics_id = in_memory_db.store_run_metrics(
            run_id=run_id,
            metrics={
                "accuracy": 0.85,
                "total_samples": 100,
                "avg_response_time": 1.2
            },
            metadata={
                "model_version": "1.0",
                "evaluation_date": "2025-06-18"
            }
        )
        
        assert metrics_id is not None
        assert isinstance(metrics_id, str)
    
    def test_get_run_metrics(self, in_memory_db):
        """Test retrieving run metrics."""
        # Create dependencies and run
        task_id = in_memory_db.create_task(
            name="test_task", description="Test", task_type="question_answer",
            config_format="custom", config_data={}
        )
        model_id = in_memory_db.create_model(
            name="Test Model", provider="test", model_id="test-1", config={}
        )
        run_id = in_memory_db.create_run(
            task_id=task_id, model_id=model_id, config={}
        )
        
        # Store metrics
        metrics_id = in_memory_db.store_run_metrics(
            run_id=run_id,
            metrics={"accuracy": 0.92},
            metadata={"test": "data"}
        )
        
        # Retrieve metrics
        metrics = in_memory_db.get_run_metrics(run_id)
        assert metrics is not None
        assert metrics['id'] == metrics_id
        assert metrics['metrics']['accuracy'] == 0.92

class TestSearchOperations:
    """Test full-text search functionality."""
    
    def test_search_tasks(self, in_memory_db):
        """Test task search functionality."""
        # Create tasks with searchable content
        task_ids = []
        task_ids.append(in_memory_db.create_task(
            name="Math Problems",
            description="Arithmetic and algebra questions",
            task_type="question_answer", config_format="custom", config_data={}
        ))
        task_ids.append(in_memory_db.create_task(
            name="Science Quiz",
            description="Physics and chemistry evaluation",
            task_type="question_answer", config_format="custom", config_data={}
        ))
        
        # Search for math-related tasks
        results = in_memory_db.search_tasks("math")
        assert len(results) == 1
        assert results[0]['name'] == "Math Problems"
        
        # Search for evaluation-related tasks
        results = in_memory_db.search_tasks("evaluation")
        assert len(results) == 1
        assert results[0]['name'] == "Science Quiz"
    
    def test_search_datasets(self, in_memory_db):
        """Test dataset search functionality."""
        # Create datasets with searchable content
        dataset_ids = []
        dataset_ids.append(in_memory_db.create_dataset(
            name="MMLU Dataset",
            format="huggingface",
            source_path="cais/mmlu",
            metadata={"description": "Massive multitask language understanding"}
        ))
        dataset_ids.append(in_memory_db.create_dataset(
            name="GSM8K Dataset", 
            format="huggingface",
            source_path="gsm8k",
            metadata={"description": "Grade school math word problems"}
        ))
        
        # Search for multitask datasets
        results = in_memory_db.search_datasets("multitask")
        assert len(results) == 1
        assert results[0]['name'] == "MMLU Dataset"
        
        # Search for math datasets
        results = in_memory_db.search_datasets("math")
        assert len(results) == 1
        assert results[0]['name'] == "GSM8K Dataset"

class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_task_creation(self, in_memory_db):
        """Test error handling for invalid task creation."""
        with pytest.raises(InputError):
            in_memory_db.create_task(
                name="",  # Empty name should fail
                description="Test",
                task_type="question_answer",
                config_format="custom",
                config_data={}
            )
    
    def test_foreign_key_violation(self, in_memory_db):
        """Test foreign key constraint enforcement."""
        with pytest.raises(InputError):
            in_memory_db.create_run(
                task_id="nonexistent_task",
                model_id="nonexistent_model",
                config={}
            )
    
    def test_concurrent_modification(self, temp_db):
        """Test handling of concurrent modifications."""
        # Create a task
        task_id = temp_db.create_task(
            name="test_task", description="Test", task_type="question_answer",
            config_format="custom", config_data={}
        )
        
        # Simulate concurrent modification by updating version manually
        conn = temp_db.get_connection()
        conn.execute(
            "UPDATE eval_tasks SET version = version + 1 WHERE id = ?",
            (task_id,)
        )
        conn.commit()
        
        # Now try to update - should detect version mismatch
        success = temp_db.update_task(task_id, description="Updated")
        assert not success  # Should fail due to version mismatch

class TestThreadSafety:
    """Test thread safety of database operations."""
    
    def test_concurrent_task_creation(self, temp_db):
        """Test concurrent task creation from multiple threads."""
        task_ids = []
        errors = []
        
        def create_task(index):
            try:
                task_id = temp_db.create_task(
                    name=f"task_{index}",
                    description=f"Task {index}",
                    task_type="question_answer",
                    config_format="custom",
                    config_data={"index": index}
                )
                task_ids.append(task_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0
        assert len(task_ids) == 10
        assert len(set(task_ids)) == 10  # All IDs should be unique
    
    def test_concurrent_read_write(self, temp_db):
        """Test concurrent read and write operations."""
        # Create initial task
        task_id = temp_db.create_task(
            name="concurrent_test",
            description="Initial description",
            task_type="question_answer",
            config_format="custom", 
            config_data={}
        )
        
        read_results = []
        write_results = []
        
        def read_task():
            for _ in range(5):
                task = temp_db.get_task(task_id)
                read_results.append(task['description'] if task else None)
                time.sleep(0.01)
        
        def write_task():
            for i in range(5):
                success = temp_db.update_task(
                    task_id, description=f"Updated description {i}"
                )
                write_results.append(success)
                time.sleep(0.01)
        
        # Start concurrent operations
        read_thread = threading.Thread(target=read_task)
        write_thread = threading.Thread(target=write_task)
        
        read_thread.start()
        write_thread.start()
        
        read_thread.join()
        write_thread.join()
        
        # Verify no errors occurred
        assert len(read_results) == 5
        assert len(write_results) == 5
        assert all(result is not None for result in read_results)

class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_batch_insert(self, temp_db):
        """Test performance with large batch operations."""
        import time
        
        start_time = time.time()
        
        # Create many tasks
        task_ids = []
        for i in range(100):
            task_id = temp_db.create_task(
                name=f"perf_task_{i}",
                description=f"Performance test task {i}",
                task_type="question_answer",
                config_format="custom",
                config_data={"index": i}
            )
            task_ids.append(task_id)
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
        assert len(task_ids) == 100
    
    def test_search_performance(self, temp_db):
        """Test search performance with many records."""
        # Create tasks with varied content
        for i in range(50):
            temp_db.create_task(
                name=f"task_{i}",
                description=f"Task {i} with math problems and evaluation content",
                task_type="question_answer",
                config_format="custom", 
                config_data={}
            )
        
        import time
        start_time = time.time()
        
        # Perform search
        results = temp_db.search_tasks("math")
        
        end_time = time.time()
        
        # Search should be fast (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(results) == 50  # All tasks should match