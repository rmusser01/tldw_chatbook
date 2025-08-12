"""
Integration tests for Evals Window V2
Tests actual functionality with real database interface
Following Textual's testing best practices
"""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Input, DataTable, Static, ProgressBar

from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator


class EvalsIntegrationTestApp(App):
    """Test app for integration testing"""
    
    def __init__(self, db_path: str = None):
        super().__init__()
        self.db_path = db_path
        self.evals_window = None
    
    def compose(self) -> ComposeResult:
        """Compose with real EvalsWindow"""
        self.evals_window = EvalsWindow(app_instance=self)
        # Override the DB path if provided
        if self.db_path:
            with patch.object(EvalsDB, '__init__', lambda self, *args, **kwargs: setattr(self, 'db_path', self.db_path) or EvalsDB.__init__(self, self.db_path)):
                yield self.evals_window
        else:
            yield self.evals_window
    
    def notify(self, message: str, severity: str = "information"):
        """Mock notify for testing"""
        pass


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_db(temp_db_dir):
    """Create a test database instance"""
    db_path = Path(temp_db_dir) / "test_evals.db"
    db = EvalsDB(str(db_path))
    
    # Add some test data
    task_id = db.create_task(
        name="Test Task",
        task_type="question_answer",
        config_format="custom",
        config_data={"prompt_template": "Test: {question}"},
        description="Test description"
    )
    
    model_id = db.create_model(
        name="Test Model",
        provider="openai",
        model_id="gpt-3.5-turbo",
        config={"temperature": 0.7}
    )
    
    return db, task_id, model_id


@pytest.mark.asyncio
async def test_ui_loads_without_errors(temp_db_dir):
    """Test that the UI loads without any errors"""
    db_path = str(Path(temp_db_dir) / "test_evals.db")
    app = EvalsIntegrationTestApp(db_path=db_path)
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check that the window mounted
        evals_window = app.query_one(EvalsWindow)
        assert evals_window is not None
        
        # Check that key elements exist
        assert app.query_one("#task-select") is not None
        assert app.query_one("#model-select") is not None
        assert app.query_one("#run-button") is not None
        assert app.query_one("#results-table") is not None
        
        # Check that orchestrator was initialized
        assert evals_window.orchestrator is not None
        assert isinstance(evals_window.orchestrator, EvaluationOrchestrator)


@pytest.mark.asyncio
async def test_database_data_loads_correctly(test_db, temp_db_dir):
    """Test that data from the database loads into the UI"""
    db, task_id, model_id = test_db
    
    # Create app with the test database
    with patch('tldw_chatbook.UI.evals_window_v2.EvalsDB', return_value=db):
        app = EvalsIntegrationTestApp()
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            evals_window = app.query_one(EvalsWindow)
            
            # Check that tasks were loaded
            task_select = app.query_one("#task-select", Select)
            # Should have at least BLANK option + our test task
            assert len(task_select._options) >= 2
            
            # Check that models were loaded  
            model_select = app.query_one("#model-select", Select)
            # Should have at least BLANK option + our test model
            assert len(model_select._options) >= 2


@pytest.mark.asyncio
async def test_task_selection_updates_state(test_db, temp_db_dir):
    """Test that selecting a task updates the window state"""
    db, task_id, model_id = test_db
    
    with patch('tldw_chatbook.UI.evals_window_v2.EvalsDB', return_value=db):
        app = EvalsIntegrationTestApp()
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            evals_window = app.query_one(EvalsWindow)
            task_select = app.query_one("#task-select", Select)
            
            # Select the task
            task_select.value = str(task_id)
            await pilot.pause()
            
            # Check that the state was updated
            assert evals_window.selected_task_id == str(task_id)
            
            # Check that task info was stored
            assert str(task_id) in evals_window.available_tasks


@pytest.mark.asyncio
async def test_model_selection_updates_state(test_db, temp_db_dir):
    """Test that selecting a model updates the window state"""
    db, task_id, model_id = test_db
    
    with patch('tldw_chatbook.UI.evals_window_v2.EvalsDB', return_value=db):
        app = EvalsIntegrationTestApp()
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            evals_window = app.query_one(EvalsWindow)
            model_select = app.query_one("#model-select", Select)
            
            # Select the model
            model_select.value = str(model_id)
            await pilot.pause()
            
            # Check that the state was updated
            assert evals_window.selected_model_id == str(model_id)
            
            # Check that model info was stored
            assert str(model_id) in evals_window.available_models


@pytest.mark.asyncio
async def test_form_inputs_update_state():
    """Test that form inputs correctly update the window state"""
    app = EvalsIntegrationTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Test temperature input
        temp_input = app.query_one("#temperature-input", Input)
        temp_input.value = "1.5"
        await pilot.pause()
        # Trigger the change event
        temp_input.post_message(Input.Changed(temp_input, "1.5"))
        await pilot.pause()
        assert evals_window.temperature == 1.5
        
        # Test max tokens input
        tokens_input = app.query_one("#max-tokens-input", Input)
        tokens_input.value = "4096"
        await pilot.pause()
        tokens_input.post_message(Input.Changed(tokens_input, "4096"))
        await pilot.pause()
        assert evals_window.max_tokens == 4096
        
        # Test max samples input
        samples_input = app.query_one("#max-samples-input", Input)
        samples_input.value = "500"
        await pilot.pause()
        samples_input.post_message(Input.Changed(samples_input, "500"))
        await pilot.pause()
        assert evals_window.max_samples == 500


@pytest.mark.asyncio
async def test_run_button_requires_configuration():
    """Test that run button validates configuration before running"""
    app = EvalsIntegrationTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Try to run without configuration
        initial_status = evals_window.evaluation_status
        await pilot.click("#run-button")
        await pilot.pause()
        
        # Should still be idle because validation failed
        assert evals_window.evaluation_status == "idle"
        
        # Status should show error
        status_text = app.query_one("#status-text", Static)
        assert "No task selected" in status_text.renderable or "No model selected" in status_text.renderable


@pytest.mark.asyncio
async def test_progress_section_visibility():
    """Test that progress section shows/hides based on evaluation status"""
    app = EvalsIntegrationTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        progress_section = app.query_one("#progress-section")
        
        # Should be hidden initially
        assert progress_section.display == False
        
        # When status changes to running, should show
        evals_window.evaluation_status = "running"
        await pilot.pause()
        assert progress_section.display == True
        
        # When completed, should stay visible briefly
        evals_window.evaluation_status = "completed"
        await pilot.pause()
        # Still visible immediately after completion
        assert evals_window.evaluation_status == "completed"


@pytest.mark.asyncio
async def test_results_table_structure():
    """Test that results table has correct structure"""
    app = EvalsIntegrationTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        results_table = app.query_one("#results-table", DataTable)
        assert results_table is not None
        
        # Check correct number of columns
        assert len(results_table.columns) == 7
        
        # Check column keys
        expected_columns = ["time", "task", "model", "samples", "success", "duration", "status"]
        for col in expected_columns:
            assert col in results_table.columns


@pytest.mark.asyncio
async def test_cost_estimation_updates():
    """Test that cost estimation updates based on selections"""
    app = EvalsIntegrationTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Set up a model selection (mock)
        evals_window.selected_model_id = "1"
        evals_window.available_models["1"] = {
            "name": "GPT-4",
            "provider": "openai",
            "model_id": "gpt-4"
        }
        
        # Update samples
        evals_window.max_samples = 1000
        evals_window._update_cost_estimate()
        await pilot.pause()
        
        # Check that cost was calculated
        cost_display = app.query_one("#cost-estimate", Static)
        assert "$" in cost_display.renderable
        
        # For high cost, warning should appear
        evals_window.max_samples = 10000
        evals_window._update_cost_estimate()
        await pilot.pause()
        
        warning_widget = app.query_one("#cost-warning", Static)
        assert "High cost" in warning_widget.renderable or warning_widget.renderable == ""


@pytest.mark.asyncio
async def test_create_task_button_functionality(test_db):
    """Test that create task button actually creates a task"""
    db, _, _ = test_db
    
    with patch('tldw_chatbook.UI.evals_window_v2.EvalsDB', return_value=db):
        app = EvalsIntegrationTestApp()
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            initial_task_count = len(db.list_tasks())
            
            # Click create task button
            await pilot.click("#create-task-btn")
            await pilot.pause()
            
            # Should have created a new task
            new_task_count = len(db.list_tasks())
            assert new_task_count > initial_task_count


@pytest.mark.asyncio
async def test_add_model_button_functionality(test_db):
    """Test that add model button actually adds a model"""
    db, _, _ = test_db
    
    with patch('tldw_chatbook.UI.evals_window_v2.EvalsDB', return_value=db):
        app = EvalsIntegrationTestApp()
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            initial_model_count = len(db.list_models())
            
            # Click add model button
            await pilot.click("#add-model-btn")
            await pilot.pause()
            
            # Should have added a new model
            new_model_count = len(db.list_models())
            assert new_model_count > initial_model_count


@pytest.mark.asyncio
async def test_refresh_buttons_reload_data(test_db):
    """Test that refresh buttons actually reload data from database"""
    db, task_id, model_id = test_db
    
    with patch('tldw_chatbook.UI.evals_window_v2.EvalsDB', return_value=db):
        app = EvalsIntegrationTestApp()
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add a new task to the database
            new_task_id = db.create_task(
                name="New Task After Load",
                task_type="generation",
                config_format="custom",
                config_data={},
                description="Added after initial load"
            )
            
            # Click refresh tasks
            await pilot.click("#refresh-tasks-btn")
            await pilot.pause()
            
            # Check that new task appears in selector
            task_select = app.query_one("#task-select", Select)
            task_values = [opt[1] for opt in task_select._options if opt[1]]
            assert str(new_task_id) in task_values


@pytest.mark.asyncio
async def test_status_bar_updates():
    """Test that status bar updates with different states"""
    app = EvalsIntegrationTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        status_text = app.query_one("#status-text", Static)
        
        # Test error status
        evals_window._update_status("Test error", error=True)
        await pilot.pause()
        assert "Test error" in status_text.renderable
        assert "error" in status_text.classes
        
        # Test success status
        evals_window._update_status("Test success", success=True)
        await pilot.pause()
        assert "Test success" in status_text.renderable
        assert "success" in status_text.classes


@pytest.mark.asyncio
async def test_evaluation_status_reactive_updates():
    """Test that reactive status changes trigger UI updates"""
    app = EvalsIntegrationTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        run_button = app.query_one("#run-button", Button)
        
        # Change to running status
        evals_window.evaluation_status = "running"
        await pilot.pause()
        
        # Button should show running state
        assert "Running" in run_button.label
        assert "--running" in run_button.classes


@pytest.mark.asyncio
async def test_progress_updates():
    """Test that progress bar updates correctly"""
    app = EvalsIntegrationTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Set to running to show progress
        evals_window.evaluation_status = "running"
        await pilot.pause()
        
        # Update progress
        evals_window.evaluation_progress = 50.0
        await pilot.pause()
        
        progress_label = app.query_one("#progress-label", Static)
        assert "50.0%" in progress_label.renderable
        
        # Update progress message
        evals_window.progress_message = "Processing sample 50/100"
        await pilot.pause()
        
        progress_message = app.query_one("#progress-message", Static)
        assert "50/100" in progress_message.renderable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])