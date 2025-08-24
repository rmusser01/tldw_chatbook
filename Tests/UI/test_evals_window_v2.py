"""
Tests for the pragmatic Evals Window V2 implementation
Following Textual's official testing documentation
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, MagicMock
from textual.app import App, ComposeResult
from textual.widgets import Button, Input

from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
from Tests.UI.textual_test_helpers import safe_click, focus_and_type, get_valid_select_value


class EvalsTestApp(App):
    """Test app for mounting the EvalsWindow"""
    
    def compose(self) -> ComposeResult:
        """Compose the test app with EvalsWindow"""
        yield EvalsWindow(app_instance=self)


@pytest.fixture
def mock_orchestrator():
    """Mock the evaluation orchestrator"""
    with patch('tldw_chatbook.UI.evals_window_v2.EvaluationOrchestrator') as mock:
        orchestrator = Mock()
        orchestrator.db = Mock()
        orchestrator.db.list_tasks = Mock(return_value=[
            {'id': '1', 'name': 'Test Task 1', 'task_type': 'multiple_choice', 'description': 'Test MC task'},
            {'id': '2', 'name': 'Test Task 2', 'task_type': 'generation', 'description': 'Test gen task'}
        ])
        orchestrator.db.list_models = Mock(return_value=[
            {'id': '1', 'name': 'GPT-4', 'provider': 'openai', 'model_id': 'gpt-4'},
            {'id': '2', 'name': 'Claude', 'provider': 'anthropic', 'model_id': 'claude-3'}
        ])
        orchestrator.db.list_runs = Mock(return_value=[])
        orchestrator.db.get_run_details = Mock(return_value={
            'id': 'run-123', 'task_id': '1', 'model_id': '1', 
            'name': 'Test Run', 'status': 'completed',
            'created_at': '2024-01-01', 'completed_at': '2024-01-02', 
            'total_samples': 100, 'completed_samples': 95,
            'metrics': {'accuracy': 0.95}, 'errors': []
        })
        orchestrator.db.create_task = Mock(return_value="task-123")
        orchestrator.db.create_model = Mock(return_value="model-123")
        orchestrator.create_model_config = Mock(return_value="model-123")
        orchestrator.run_evaluation = Mock()
        
        mock.return_value = orchestrator
        yield mock


@pytest.mark.asyncio
async def test_evals_window_initialization(mock_orchestrator):
    """Test that EvalsWindow initializes correctly"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        # Check that the window exists
        evals_window = app.query_one(EvalsWindow)
        assert evals_window is not None
        
        # Check that the header is present
        header = app.query_one(".header-title")
        assert header is not None
        assert "Evaluation Lab V2" in header.renderable
        
        # Check that orchestrator was initialized
        assert evals_window.orchestrator is not None


@pytest.mark.asyncio
async def test_task_selection(mock_orchestrator):
    """Test task selection functionality"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Check that tasks were loaded
        task_select = app.query_one("#task-select")
        assert task_select is not None
        
        # Simulate selecting a task
        task_select.value = "1"
        await pilot.pause()  # Let message propagate
        
        # Check that the task was selected
        assert evals_window.selected_task_id == "1"


@pytest.mark.asyncio
async def test_model_selection(mock_orchestrator):
    """Test model selection functionality"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Check that models were loaded
        model_select = app.query_one("#model-select")
        assert model_select is not None
        
        # Simulate selecting a model
        model_select.value = "1"
        await pilot.pause()  # Let message propagate
        
        # Check that the model was selected
        assert evals_window.selected_model_id == "1"


@pytest.mark.asyncio
async def test_temperature_input(mock_orchestrator):
    """Test temperature input functionality"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Get temperature input
        temp_input = app.query_one("#temperature-input", Input)
        assert temp_input is not None
        
        # Focus and type new temperature
        await focus_and_type(pilot, temp_input, "1.5")
        await pilot.pause()  # Let message propagate
        
        # Check that temperature was updated
        assert evals_window.temperature == 1.5


@pytest.mark.asyncio
async def test_max_samples_input(mock_orchestrator):
    """Test max samples input functionality"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Get max samples input
        samples_input = app.query_one("#max-samples-input", Input)
        assert samples_input is not None
        
        # Focus and type new value
        await focus_and_type(pilot, samples_input, "500")
        await pilot.pause()  # Let message propagate
        
        # Check that max samples was updated
        assert evals_window.max_samples == 500


@pytest.mark.asyncio
async def test_cost_estimation_updates(mock_orchestrator):
    """Test that cost estimation updates when configuration changes"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Select a model
        model_select = app.query_one("#model-select")
        model_select.value = "1"
        await pilot.pause()
        
        # Update max samples
        samples_input = app.query_one("#max-samples-input")
        samples_input.clear()
        await pilot.press("1", "0", "0", "0")
        await pilot.pause()
        
        # Check that cost was estimated
        cost_display = app.query_one("#cost-estimate")
        assert cost_display is not None
        assert "$" in cost_display.renderable


@pytest.mark.asyncio
async def test_run_button_validation(mock_orchestrator):
    """Test that run button validates configuration"""
    app = EvalsTestApp()
    async with app.run_test(size=(120, 50)) as pilot:  # Larger screen
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Try to run without configuration
        run_button = app.query_one("#run-button", Button)
        await safe_click(pilot, run_button)
        await pilot.pause()
        
        # Should still be idle (validation failed)
        assert evals_window.evaluation_status == "idle"
        
        # Now configure properly
        from textual.widgets import Select
        task_select = app.query_one("#task-select", Select)
        task_value = get_valid_select_value(task_select, 0)
        if task_value:
            task_select.value = task_value
            await pilot.pause()  # Wait for change event
            # Verify it was set
            assert evals_window.selected_task_id is not None, "Task ID was not set"
        
        model_select = app.query_one("#model-select", Select)
        model_value = get_valid_select_value(model_select, 0)
        if model_value:
            model_select.value = model_value
            await pilot.pause()  # Wait for change event
            # Verify it was set
            assert evals_window.selected_model_id is not None, "Model ID was not set"
        
        # Try to run again - should start evaluation
        # Note: Actual evaluation won't complete in test, but status should change
        with patch.object(evals_window, 'run_worker') as mock_worker:
            await safe_click(pilot, run_button)
            await pilot.pause()
            # Check that run_worker was called with run_evaluation
            mock_worker.assert_called_once()
            # Verify the first argument is the run_evaluation method
            assert mock_worker.call_args[0][0].__name__ == 'run_evaluation'


@pytest.mark.asyncio
async def test_add_model_button(mock_orchestrator):
    """Test adding a new model"""
    app = EvalsTestApp()
    async with app.run_test(size=(120, 50)) as pilot:  # Larger screen
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Ensure orchestrator is set up
        assert evals_window.orchestrator is not None, "Orchestrator not initialized"
        
        # Click add model button
        add_btn = app.query_one("#add-model-btn", Button)
        click_result = await safe_click(pilot, add_btn)
        assert click_result, "Failed to click add model button"
        await pilot.pause()
        
        # Check that create_model was called on the database
        mock_orchestrator.return_value.db.create_model.assert_called_once()


@pytest.mark.asyncio
async def test_create_task_button(mock_orchestrator):
    """Test creating a new task"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        # Click create task button
        create_btn = app.query_one("#create-task-btn", Button)
        await safe_click(pilot, create_btn)
        await pilot.pause()
        
        # Check that task was created
        mock_orchestrator.return_value.db.create_task.assert_called_once()


@pytest.mark.asyncio
async def test_progress_display(mock_orchestrator):
    """Test that progress section shows/hides correctly"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Progress collapsible should be collapsed initially
        from textual.widgets import Collapsible
        progress_collapsible = app.query_one("#progress-collapsible", Collapsible)
        assert progress_collapsible.collapsed == True
        
        # When status changes to running, progress should expand
        evals_window.evaluation_status = "running"
        await pilot.pause()
        assert progress_collapsible.collapsed == False
        
        # When status changes to completed, it should remain visible briefly
        evals_window.evaluation_status = "completed"
        await pilot.pause()
        assert evals_window.evaluation_status == "completed"


@pytest.mark.asyncio
async def test_reactive_state_updates(mock_orchestrator):
    """Test that reactive attributes trigger UI updates"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Update progress
        from textual.widgets import ProgressBar
        evals_window.evaluation_progress = 50.0
        await pilot.pause()
        
        progress_bar = app.query_one("#progress-bar", ProgressBar)
        assert progress_bar.percentage == 0.5  # ProgressBar uses 0-1 scale
        
        # Update progress message
        evals_window.progress_message = "Processing sample 50/100"
        await pilot.pause()
        
        progress_message = app.query_one("#progress-message")
        assert "50/100" in progress_message.renderable


@pytest.mark.asyncio
async def test_results_table_structure(mock_orchestrator):
    """Test that results table is properly structured"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        # Get the results table
        results_table = app.query_one("#results-table")
        assert results_table is not None
        
        # Check that columns were added
        assert len(results_table.columns) == 7  # Time, Task, Model, Samples, Success Rate, Duration, Status
        assert "time" in results_table.columns
        assert "task" in results_table.columns
        assert "model" in results_table.columns
        assert "samples" in results_table.columns
        assert "success" in results_table.columns
        assert "duration" in results_table.columns
        assert "status" in results_table.columns


@pytest.mark.asyncio
async def test_keyboard_shortcuts(mock_orchestrator):
    """Test keyboard shortcuts work"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        # Since EvalsWindow is a Container, not a Screen, 
        # it doesn't have direct bindings, but we can test
        # that the UI responds to button clicks which would
        # be triggered by parent app bindings
        
        # Verify UI elements are accessible
        run_button = app.query_one("#run-button")
        assert run_button is not None
        assert run_button.disabled == False


@pytest.mark.asyncio 
async def test_status_updates(mock_orchestrator):
    """Test status bar updates correctly"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Get status element
        from textual.widgets import Static
        status = app.query_one("#status-text", Static)
        assert status is not None
        # Status could be "Ready" or "Data loaded successfully"
        assert "Ready" in status.renderable or "loaded" in status.renderable
        
        # Update status
        evals_window._update_status("Testing", error=True)
        await pilot.pause()
        
        # Check error class was added
        assert "error" in status.classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])