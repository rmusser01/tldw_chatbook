"""
Tests for the pragmatic Evals Window V2 implementation
Following Textual's official testing documentation
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, MagicMock
from textual.app import App, ComposeResult

from tldw_chatbook.UI.evals_window_v2 import EvalsWindow


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
        orchestrator.db.get_tasks = Mock(return_value=[
            (1, "Test Task 1", "multiple_choice", None),
            (2, "Test Task 2", "generation", None)
        ])
        orchestrator.db.get_model_configs = Mock(return_value=[
            (1, "GPT-4", "openai", "gpt-4", None),
            (2, "Claude", "anthropic", "claude-3", None)
        ])
        orchestrator.db.get_run_details = Mock(return_value=(
            "run-123", 1, 1, "Test Run", "completed",
            "2024-01-01", "2024-01-02", 100, 95,
            '{"accuracy": 0.95}', '[]'
        ))
        orchestrator.db.create_task = Mock(return_value="task-123")
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
        temp_input = app.query_one("#temperature-input")
        assert temp_input is not None
        
        # Clear and type new temperature
        temp_input.clear()
        await pilot.press("1", ".", "5")
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
        samples_input = app.query_one("#max-samples-input")
        assert samples_input is not None
        
        # Clear and type new value
        samples_input.clear()
        await pilot.press("5", "0", "0")
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
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Try to run without configuration
        run_button = app.query_one("#run-button")
        await pilot.click("#run-button")
        await pilot.pause()
        
        # Should still be idle (validation failed)
        assert evals_window.evaluation_status == "idle"
        
        # Now configure properly
        task_select = app.query_one("#task-select")
        task_select.value = "1"
        model_select = app.query_one("#model-select")
        model_select.value = "1"
        await pilot.pause()
        
        # Try to run again - should start evaluation
        # Note: Actual evaluation won't complete in test, but status should change
        with patch.object(evals_window, 'run_evaluation') as mock_run:
            await pilot.click("#run-button")
            await pilot.pause()
            mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_add_model_button(mock_orchestrator):
    """Test adding a new model"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        evals_window = app.query_one(EvalsWindow)
        
        # Click add model button
        await pilot.click("#add-model-btn")
        await pilot.pause()
        
        # Check that create_model_config was called
        mock_orchestrator.return_value.create_model_config.assert_called_once()


@pytest.mark.asyncio
async def test_create_task_button(mock_orchestrator):
    """Test creating a new task"""
    app = EvalsTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize
        
        # Click create task button
        await pilot.click("#create-task-btn")
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
        
        # Progress should be hidden initially
        progress_section = app.query_one("#progress-section")
        assert progress_section.display == False
        
        # When status changes to running, progress should show
        evals_window.evaluation_status = "running"
        await pilot.pause()
        assert progress_section.display == True
        
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
        evals_window.evaluation_progress = 50.0
        await pilot.pause()
        
        progress_label = app.query_one("#progress-label")
        assert "50.0%" in progress_label.renderable
        
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
        status = app.query_one("#status-text")
        assert status is not None
        assert "Ready" in status.renderable
        
        # Update status
        evals_window._update_status("Testing", error=True)
        await pilot.pause()
        
        # Check error class was added
        assert "error" in status.classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])