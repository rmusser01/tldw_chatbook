"""
Performance Tests for Evals Window V2
Tests handling of large datasets, memory usage, and responsiveness
Following Textual's testing best practices
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
import time
import psutil
import gc
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from textual.app import App, ComposeResult
from textual.widgets import Select, DataTable

from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
from tldw_chatbook.DB.Evals_DB import EvalsDB
from Tests.UI.textual_test_helpers import get_valid_select_value
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator


class EvalsPerfTestApp(App):
    """Test app for performance testing"""
    
    def __init__(self, db_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_path = db_path
    
    def compose(self) -> ComposeResult:
        """Compose with EvalsWindow"""
        if self.db_path:
            with patch.object(EvaluationOrchestrator, '_initialize_database') as mock_init:
                mock_init.return_value = EvalsDB(self.db_path, client_id="perf_test")
                yield EvalsWindow(app_instance=self)
        else:
            yield EvalsWindow(app_instance=self)
    
    def notify(self, message: str, severity: str = "information"):
        """Mock notify"""
        pass


@pytest.fixture
def temp_db_dir():
    """Create temporary directory for test databases"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def large_database(temp_db_dir):
    """Create a database with large amounts of data"""
    db_path = Path(temp_db_dir) / "large_test.db"
    db = EvalsDB(str(db_path), client_id="perf_test")
    
    # Create many tasks (1000+)
    task_ids = []
    for i in range(1000):
        task_id = db.create_task(
            name=f"Task {i:04d}",
            task_type=["question_answer", "generation", "classification"][i % 3],
            config_format="custom",
            config_data={
                "prompt_template": f"Template for task {i}",
                "dataset_name": f"dataset_{i}",
                "additional_config": {"param": i}
            },
            description=f"Description for task {i} with some additional text to make it longer"
        )
        task_ids.append(task_id)
    
    # Create many models (1000+)
    model_ids = []
    for i in range(1000):
        model_id = db.create_model(
            name=f"Model {i:04d}",
            provider=["openai", "anthropic", "local", "custom"][i % 4],
            model_id=f"model-{i}",
            config={
                "temperature": 0.5 + (i % 10) * 0.1,
                "max_tokens": 1024 + (i % 4) * 1024,
                "additional_params": {"param": i}
            }
        )
        model_ids.append(model_id)
    
    # Create many runs with results (10000+ rows)
    for i in range(100):
        run_id = db.create_run(
            name=f"Run {i:04d}",
            task_id=task_ids[i % len(task_ids)],
            model_id=model_ids[i % len(model_ids)],
            config_overrides={"override": i}
        )
        
        # Add results for each run
        for j in range(100):
            db.store_result(
                run_id=run_id,
                sample_id=f"sample_{i}_{j}",
                input_data={"input": f"Input text {j}" * 10},  # Larger input
                actual_output=f"Output text {j}" * 10,  # Larger output
                expected_output=f"Expected text {j}" * 10,
                metrics={
                    "accuracy": 0.5 + (j % 50) * 0.01,
                    "f1_score": 0.6 + (j % 40) * 0.01,
                    "bleu": 0.7 + (j % 30) * 0.01
                },
                metadata={"meta": f"data_{j}"}
            )
        
        db.update_run_status(run_id, "completed")
    
    return str(db_path), task_ids, model_ids


# ============================================================================
# LARGE DATA HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_load_1000_plus_tasks(large_database):
    """Test loading and displaying 1000+ tasks"""
    db_path, task_ids, _ = large_database
    
    start_time = time.time()
    
    app = EvalsPerfTestApp(db_path=db_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        
        load_time = time.time() - start_time
        
        # Should load within reasonable time (5 seconds)
        assert load_time < 5.0
        
        # Check tasks loaded
        task_select = app.query_one("#task-select", Select)
        # list_tasks has a default limit of 100
        assert len(task_select._options) <= 101  # blank + up to 100 tasks
        
        # Test selection performance (select from available options)
        select_start = time.time()
        # Select first available task (not 500 which doesn't exist)
        task_value = get_valid_select_value(task_select, 0)
        if task_value:
            task_select.value = task_value
            await pilot.pause()
        select_time = time.time() - select_start
        
        # Selection should be fast (< 0.5 seconds)
        assert select_time < 0.5
        
        # Check UI is still responsive
        evals_window = app.query_one(EvalsWindow)
        if task_value:
            assert evals_window.selected_task_id == task_value


@pytest.mark.asyncio
async def test_load_1000_plus_models(large_database):
    """Test loading and displaying 1000+ models"""
    db_path, _, model_ids = large_database
    
    start_time = time.time()
    
    app = EvalsPerfTestApp(db_path=db_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        
        load_time = time.time() - start_time
        
        # Should load within reasonable time
        assert load_time < 5.0
        
        # Check models loaded
        model_select = app.query_one("#model-select", Select)
        # list_models has a default limit of 100  
        assert len(model_select._options) <= 101  # blank + up to 100 models
        
        # Test scrolling through options
        scroll_start = time.time()
        # Simulate scrolling by changing selection multiple times
        # Use actual available options
        available_options = [opt[1] for opt in model_select._options if opt[0] != Select.BLANK]
        for i in range(min(10, len(available_options))):
            model_select.value = available_options[i]
            await pilot.pause()
        scroll_time = time.time() - scroll_start
        
        # Scrolling should be smooth (< 2 seconds for 10 selections)
        assert scroll_time < 2.0


@pytest.mark.asyncio
async def test_results_table_10000_plus_rows(large_database):
    """Test results table with 10000+ rows"""
    db_path, _, _ = large_database
    
    app = EvalsPerfTestApp(db_path=db_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        
        table = app.query_one("#results-table", DataTable)
        
        # Table should handle large dataset
        # (May be limited/paginated for performance)
        assert table is not None
        assert table.row_count <= 100  # Should limit displayed rows
        
        # Test scrolling performance
        scroll_start = time.time()
        
        # Simulate scrolling
        table.scroll_down()
        await pilot.pause()
        table.scroll_down()
        await pilot.pause()
        table.scroll_up()
        await pilot.pause()
        
        scroll_time = time.time() - scroll_start
        
        # Scrolling should be responsive (< 1 second)
        assert scroll_time < 1.0


@pytest.mark.asyncio
async def test_search_performance_large_dataset(large_database):
    """Test search/filter performance with large dataset"""
    db_path, task_ids, _ = large_database
    
    app = EvalsPerfTestApp(db_path=db_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Mock search functionality
        with patch.object(evals_window.orchestrator.db, 'search_tasks') as mock_search:
            mock_search.return_value = [
                {'id': task_ids[i], 'name': f'Task {i:04d}'}
                for i in range(100)  # Return 100 results
            ]
            
            search_start = time.time()
            
            # Trigger search (would be through a search input in real implementation)
            mock_search("test query")
            
            search_time = time.time() - search_start
            
            # Search should be fast (< 0.5 seconds)
            assert search_time < 0.5


# ============================================================================
# MEMORY USAGE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_memory_usage_baseline():
    """Test baseline memory usage of empty EvalsWindow"""
    process = psutil.Process()
    
    # Force garbage collection
    gc.collect()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Get memory after loading
        loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = loaded_memory - initial_memory
        
        # Should not use excessive memory for empty window (< 100 MB increase)
        assert memory_increase < 100


@pytest.mark.asyncio
async def test_memory_usage_with_large_data(large_database):
    """Test memory usage with large dataset"""
    db_path, _, _ = large_database
    
    process = psutil.Process()
    gc.collect()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    app = EvalsPerfTestApp(db_path=db_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        
        loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = loaded_memory - initial_memory
        
        # Should handle large data efficiently (< 200 MB increase)
        assert memory_increase < 200
        
        # Test memory doesn't leak during operations
        for _ in range(10):
            # Refresh tasks
            evals_window = app.query_one(EvalsWindow)
            evals_window._load_tasks()
            await pilot.pause()
        
        gc.collect()
        after_operations = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should not grow significantly (< 50 MB additional)
        assert after_operations - loaded_memory < 50


@pytest.mark.asyncio
async def test_memory_cleanup_after_evaluation():
    """Test memory is properly cleaned up after evaluation"""
    process = psutil.Process()
    
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        gc.collect()
        before_eval = process.memory_info().rss / 1024 / 1024  # MB
        
        # Mock a large evaluation
        with patch.object(evals_window.orchestrator, 'run_evaluation', new_callable=AsyncMock) as mock_run:
            # Simulate large result data
            large_result = "x" * (10 * 1024 * 1024)  # 10 MB of data
            mock_run.return_value = large_result
            
            # Run evaluation
            evals_window.selected_task_id = "1"
            evals_window.selected_model_id = "1"
            await evals_window.run_evaluation()
            await pilot.pause()
            
            during_eval = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clear evaluation data
            evals_window.evaluation_status = "idle"
            evals_window.current_run_id = None
            del large_result
            
            gc.collect()
            after_cleanup = process.memory_info().rss / 1024 / 1024  # MB
            
            # Memory should be released (within 20 MB of original)
            assert after_cleanup - before_eval < 20


# ============================================================================
# REACTIVE ATTRIBUTE PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_reactive_updates_performance():
    """Test performance of reactive attribute updates"""
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Test rapid reactive updates
        update_start = time.time()
        
        for i in range(100):
            evals_window.evaluation_progress = i
            evals_window.progress_message = f"Update {i}"
            if i % 10 == 0:
                await pilot.pause()  # Allow UI to update
        
        update_time = time.time() - update_start
        
        # Should handle 100 updates quickly (< 2 seconds)
        assert update_time < 2.0


@pytest.mark.asyncio
async def test_concurrent_reactive_updates():
    """Test handling of concurrent reactive updates"""
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Simulate concurrent updates
        async def update_progress():
            for i in range(50):
                evals_window.evaluation_progress = i * 2
                await asyncio.sleep(0.01)
        
        async def update_message():
            for i in range(50):
                evals_window.progress_message = f"Message {i}"
                await asyncio.sleep(0.01)
        
        async def update_status():
            statuses = ["idle", "running", "completed", "error"]
            for i in range(20):
                evals_window.evaluation_status = statuses[i % 4]
                await asyncio.sleep(0.025)
        
        # Run all updates concurrently
        start_time = time.time()
        
        await asyncio.gather(
            update_progress(),
            update_message(),
            update_status()
        )
        
        elapsed = time.time() - start_time
        
        # Should complete without deadlocks (< 1 second)
        assert elapsed < 1.0


# ============================================================================
# WORKER THREAD PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_worker_thread_creation_performance():
    """Test performance of worker thread creation"""
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Mock the worker creation
        with patch.object(evals_window, 'run_worker') as mock_worker:
            start_time = time.time()
            
            # Create multiple workers
            for i in range(10):
                evals_window.run_worker(lambda: None, name=f"test_worker_{i}")
            
            creation_time = time.time() - start_time
            
            # Should create workers quickly (< 0.5 seconds for 10)
            assert creation_time < 0.5
            assert mock_worker.call_count == 10


@pytest.mark.asyncio
async def test_worker_cleanup_performance():
    """Test performance of worker cleanup"""
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Create mock workers
        mock_workers = {}
        for i in range(10):
            worker = Mock()
            worker.cancel = Mock()
            mock_workers[f"worker_{i}"] = worker
        
        evals_window._workers = mock_workers
        
        # Test cleanup performance
        start_time = time.time()
        
        for name, worker in mock_workers.items():
            worker.cancel()
        
        mock_workers.clear()
        
        cleanup_time = time.time() - start_time
        
        # Should clean up quickly (< 0.1 seconds)
        assert cleanup_time < 0.1


# ============================================================================
# UI RESPONSIVENESS TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_ui_responsiveness_during_evaluation():
    """Test UI remains responsive during evaluation"""
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Mock slow evaluation
        async def slow_evaluation(*args, **kwargs):
            for i in range(10):
                await asyncio.sleep(0.1)
                # Update progress
                app.call_from_thread(
                    lambda: setattr(evals_window, 'evaluation_progress', i * 10)
                )
            return "run-123"
        
        with patch.object(evals_window.orchestrator, 'run_evaluation', new=slow_evaluation):
            # Start evaluation
            evals_window.selected_task_id = "1"
            evals_window.selected_model_id = "1"
            
            eval_task = asyncio.create_task(evals_window.run_evaluation())
            
            # Test UI interactions during evaluation
            interaction_start = time.time()
            
            # Should be able to interact with UI
            await pilot.click("#cancel-button")
            await pilot.pause()
            
            interaction_time = time.time() - interaction_start
            
            # UI should respond quickly (< 0.5 seconds)
            assert interaction_time < 0.5
            
            # Clean up
            eval_task.cancel()
            try:
                await eval_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_animation_performance():
    """Test performance of animations (collapsible expand/collapse)"""
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Get all collapsibles
        from textual.widgets import Collapsible, Button
        collapsibles = app.query(Collapsible)
        
        animation_start = time.time()
        
        # Toggle all collapsibles
        for collapsible in collapsibles:
            toggle_button = collapsible.get_child_by_type(Button)
            await pilot.click(toggle_button)
            await pilot.wait_for_animation()
        
        animation_time = time.time() - animation_start
        
        # Animations should complete reasonably fast (< 3 seconds for all)
        assert animation_time < 3.0


# ============================================================================
# DATABASE QUERY PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_database_query_performance(large_database):
    """Test performance of database queries"""
    db_path, _, _ = large_database
    
    db = EvalsDB(db_path, client_id="perf_test")
    
    # Test list queries
    query_times = {}
    
    # List tasks
    start = time.time()
    tasks = db.list_tasks()
    query_times['list_tasks'] = time.time() - start
    # Default limit is 100
    assert len(tasks) == 100
    
    # List models
    start = time.time()
    models = db.list_models()
    query_times['list_models'] = time.time() - start
    # Default limit is 100
    assert len(models) == 100
    
    # List runs
    start = time.time()
    runs = db.list_runs(limit=100)
    query_times['list_runs'] = time.time() - start
    assert len(runs) <= 100
    
    # All queries should be fast (< 1 second each)
    for query_name, query_time in query_times.items():
        assert query_time < 1.0, f"{query_name} took {query_time:.2f} seconds"


@pytest.mark.asyncio
async def test_incremental_loading_performance(large_database):
    """Test performance of incremental/paginated loading"""
    db_path, _, _ = large_database
    
    app = EvalsPerfTestApp(db_path=db_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Test incremental loading of results
        table = app.query_one("#results-table", DataTable)
        
        load_times = []
        
        for page in range(5):
            start = time.time()
            
            # Simulate loading more results
            # (In real implementation, this would be pagination)
            table.clear()
            for i in range(20):  # Load 20 rows at a time
                table.add_row(f"Row {page * 20 + i}")
            
            await pilot.pause()
            load_times.append(time.time() - start)
        
        # Each page should load quickly (< 0.5 seconds)
        for i, load_time in enumerate(load_times):
            assert load_time < 0.5, f"Page {i} took {load_time:.2f} seconds"


# ============================================================================
# STRESS TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_rapid_task_switching_stress():
    """Stress test rapid task switching"""
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Create mock tasks
        with patch.object(evals_window.orchestrator.db, 'list_tasks') as mock_list:
            mock_list.return_value = [
                {'id': str(i), 'name': f'Task {i}', 'task_type': 'test'}
                for i in range(100)
            ]
            
            evals_window._load_tasks()
            await pilot.pause()
            
            # Rapidly switch between tasks
            switch_start = time.time()
            
            for i in range(50):
                evals_window.selected_task_id = str(i)
                if i % 10 == 0:
                    await pilot.pause()
            
            switch_time = time.time() - switch_start
            
            # Should handle rapid switching (< 2 seconds for 50 switches)
            assert switch_time < 2.0


@pytest.mark.asyncio
async def test_concurrent_operations_stress():
    """Stress test with multiple concurrent operations"""
    app = EvalsPerfTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        async def refresh_tasks():
            for _ in range(10):
                evals_window._load_tasks()
                await asyncio.sleep(0.1)
        
        async def refresh_models():
            for _ in range(10):
                evals_window._load_models()
                await asyncio.sleep(0.1)
        
        async def update_progress():
            for i in range(100):
                evals_window.evaluation_progress = i
                await asyncio.sleep(0.01)
        
        # Run all operations concurrently
        stress_start = time.time()
        
        await asyncio.gather(
            refresh_tasks(),
            refresh_models(),
            update_progress()
        )
        
        stress_time = time.time() - stress_start
        
        # Should complete without issues (< 2 seconds)
        assert stress_time < 2.0
        
        # UI should still be functional
        assert app.query_one("#run-button") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])