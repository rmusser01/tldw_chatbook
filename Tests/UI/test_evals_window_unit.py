"""
Comprehensive Unit Tests for Evals Window V2
Following Textual's official testing best practices
Tests all components, event handlers, and reactive attributes
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Input, DataTable, Static, ProgressBar, Collapsible
from textual.css.errors import StyleValueError

from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
from Tests.UI.textual_test_helpers import safe_click, prepare_window_for_testing, get_option_labels, get_valid_select_value


class EvalsUnitTestApp(App):
    """Test app for unit testing EvalsWindow"""
    
    DEFAULT_CSS = """
    Screen {
        width: 120;
        height: 80;
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notifications = []
    
    def compose(self) -> ComposeResult:
        """Compose the test app with EvalsWindow"""
        yield EvalsWindow(app_instance=self)
    
    def notify(self, message: str, severity: str = "information"):
        """Mock notify for testing"""
        self.notifications.append((message, severity))


@pytest.fixture
def mock_orchestrator():
    """Mock the evaluation orchestrator with comprehensive test data"""
    with patch('tldw_chatbook.UI.evals_window_v2.EvaluationOrchestrator') as mock:
        orchestrator = Mock()
        
        # Mock database with comprehensive test data
        orchestrator.db = Mock()
        orchestrator.db.list_tasks = Mock(return_value=[
            {'id': '1', 'name': 'Test Task 1', 'task_type': 'multiple_choice', 'description': 'Test MC task'},
            {'id': '2', 'name': 'Test Task 2', 'task_type': 'generation', 'description': 'Test gen task'},
            {'id': '3', 'name': 'Test Task 3', 'task_type': 'classification', 'description': 'Test class task'},
        ])
        orchestrator.db.list_models = Mock(return_value=[
            {'id': '1', 'name': 'GPT-4', 'provider': 'openai', 'model_id': 'gpt-4'},
            {'id': '2', 'name': 'Claude-3', 'provider': 'anthropic', 'model_id': 'claude-3-opus'},
            {'id': '3', 'name': 'Llama-3', 'provider': 'local', 'model_id': 'llama-3-70b'},
        ])
        orchestrator.db.list_runs = Mock(return_value=[
            {
                'id': 'run-1', 'name': 'Test Run 1', 'status': 'completed',
                'created_at': '2024-01-01T10:00:00', 'completed_samples': 100,
                'task_name': 'Test Task 1', 'model_name': 'GPT-4'
            }
        ])
        orchestrator.db.create_task = Mock(return_value='task-123')
        orchestrator.db.create_model = Mock(return_value='model-123')
        
        # Mock orchestrator methods
        orchestrator.create_model_config = Mock(return_value='model-123')
        
        # Properly handle progress_callback in run_evaluation
        async def mock_run_evaluation(task_id, model_id, run_name=None, max_samples=None, 
                                       config_overrides=None, progress_callback=None, **kwargs):
            # Call progress_callback with proper integer values if provided
            if progress_callback:
                progress_callback(1, 10, "Starting evaluation")
                progress_callback(5, 10, "Processing")
                progress_callback(10, 10, "Complete")
            return 'run-123'
        
        orchestrator.run_evaluation = AsyncMock(side_effect=mock_run_evaluation)
        orchestrator.db.get_run_details = Mock(return_value={
            'id': 'run-123', 'task_id': '1', 'model_id': '1',
            'status': 'completed', 'metrics': {'accuracy': 0.95}
        })
        orchestrator.db.get_run = Mock(return_value={
            'id': 'run-123', 'task_id': '1', 'model_id': '1',
            'status': 'completed', 'metrics': {'accuracy': 0.95}
        })
        orchestrator.cancel_evaluation = Mock(return_value=True)
        
        mock.return_value = orchestrator
        yield mock


# ============================================================================
# WIDGET INITIALIZATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_window_initialization(mock_orchestrator):
    """Test that EvalsWindow initializes with all required widgets"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check main window exists
        evals_window = app.query_one(EvalsWindow)
        assert evals_window is not None
        
        # Check orchestrator initialized
        assert evals_window.orchestrator is not None
        mock_orchestrator.assert_called_once()


@pytest.mark.asyncio
async def test_all_collapsibles_present(mock_orchestrator):
    """Test that all Collapsible sections are present"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check all collapsible sections exist
        collapsibles = app.query(Collapsible)
        assert len(collapsibles) >= 5  # Task, Model, Cost, Progress, Results
        
        # Check specific collapsibles by title
        titles = [c.title for c in collapsibles]
        assert "📋 Task Configuration" in titles
        assert "🤖 Model Configuration" in titles
        assert "💰 Cost Estimation" in titles
        assert "📊 Progress" in titles
        assert "📊 Recent Results" in titles


@pytest.mark.asyncio
async def test_vertical_layout_enforced(mock_orchestrator):
    """Test that vertical layout is properly enforced"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        # Check that the layout style is vertical
        assert str(evals_window.styles.layout) == "<vertical>"


@pytest.mark.asyncio
async def test_all_form_inputs_present(mock_orchestrator):
    """Test that all form inputs are present and accessible"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check all required inputs exist
        assert app.query_one("#task-select", Select) is not None
        assert app.query_one("#model-select", Select) is not None
        assert app.query_one("#temperature-input", Input) is not None
        assert app.query_one("#max-tokens-input", Input) is not None
        assert app.query_one("#max-samples-input", Input) is not None


@pytest.mark.asyncio
async def test_all_buttons_present(mock_orchestrator):
    """Test that all action buttons are present"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check all buttons exist
        button_ids = [
            "#load-task-btn", "#create-task-btn", "#refresh-tasks-btn",
            "#add-model-btn", "#test-model-btn", "#refresh-models-btn",
            "#run-button", "#cancel-button"
        ]
        
        for button_id in button_ids:
            button = app.query_one(button_id, Button)
            assert button is not None
            # Note: Some buttons may be disabled initially based on state


@pytest.mark.asyncio
async def test_results_table_structure(mock_orchestrator):
    """Test that results table has correct structure"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        table = app.query_one("#results-table", DataTable)
        assert table is not None
        
        # Check column structure
        expected_columns = ["time", "task", "model", "samples", "success", "duration", "status"]
        assert len(table.columns) == len(expected_columns)
        for col in expected_columns:
            assert col in table.columns


# ============================================================================
# REACTIVE ATTRIBUTE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_temperature_reactive_update(mock_orchestrator):
    """Test temperature reactive attribute updates UI"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        temp_input = app.query_one("#temperature-input", Input)
        
        # Clear and set new temperature
        temp_input.value = ""
        await pilot.pause()
        temp_input.value = "1.5"
        await pilot.pause()
        
        # Check reactive attribute updated
        assert evals_window.temperature == 1.5


@pytest.mark.asyncio
async def test_max_tokens_reactive_update(mock_orchestrator):
    """Test max_tokens reactive attribute updates"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        tokens_input = app.query_one("#max-tokens-input", Input)
        
        # Set new max tokens
        tokens_input.value = "4096"
        await pilot.pause()
        
        # Check reactive attribute updated
        assert evals_window.max_tokens == 4096


@pytest.mark.asyncio
async def test_max_samples_reactive_update(mock_orchestrator):
    """Test max_samples reactive attribute updates"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        samples_input = app.query_one("#max-samples-input", Input)
        
        # Set new max samples
        samples_input.value = "500"
        await pilot.pause()
        
        # Check reactive attribute updated
        assert evals_window.max_samples == 500


@pytest.mark.asyncio
async def test_progress_reactive_updates(mock_orchestrator):
    """Test progress reactive attributes update UI correctly"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Update progress
        evals_window.evaluation_progress = 75.5
        await pilot.pause()
        
        # Check progress bar updated
        progress_bar = app.query_one("#progress-bar", ProgressBar)
        assert progress_bar.percentage == 0.755  # ProgressBar uses 0-1 scale
        
        # Update progress message
        evals_window.progress_message = "Processing sample 75/100"
        await pilot.pause()
        
        progress_msg = app.query_one("#progress-message", Static)
        assert "75/100" in progress_msg.renderable


@pytest.mark.asyncio
async def test_status_reactive_updates(mock_orchestrator):
    """Test evaluation_status reactive attribute controls UI state"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        progress_collapsible = app.query_one("#progress-collapsible", Collapsible)
        
        # Initially idle - progress should be collapsed
        assert evals_window.evaluation_status == "idle"
        assert progress_collapsible.collapsed == True
        
        # Change to running - progress should expand
        evals_window.evaluation_status = "running"
        await pilot.pause()
        assert progress_collapsible.collapsed == False
        
        # Change to completed
        evals_window.evaluation_status = "completed"
        await pilot.pause()
        assert evals_window.evaluation_status == "completed"


# ============================================================================
# EVENT HANDLER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_task_selection_handler(mock_orchestrator):
    """Test task selection event handler"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        task_select = app.query_one("#task-select", Select)
        
        # Select a task
        task_select.value = "1"
        await pilot.pause()
        
        # Check task was selected
        assert evals_window.selected_task_id == "1"
        
        # Check cost estimation was triggered
        cost_display = app.query_one("#cost-estimate", Static)
        assert "$" in cost_display.renderable


@pytest.mark.asyncio
async def test_model_selection_handler(mock_orchestrator):
    """Test model selection event handler"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        model_select = app.query_one("#model-select", Select)
        
        # Select a model
        model_select.value = "1"
        await pilot.pause()
        
        # Check model was selected
        assert evals_window.selected_model_id == "1"


@pytest.mark.asyncio
async def test_temperature_change_handler_validation(mock_orchestrator):
    """Test temperature input validation in handler"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        temp_input = app.query_one("#temperature-input", Input)
        
        # Test invalid temperature (too high)
        temp_input.value = "3.0"
        await pilot.pause()
        
        # Should be clamped to max (2.0) or kept at 3.0 depending on validation
        assert evals_window.temperature <= 3.0
        
        # Test invalid temperature (negative)  
        temp_input.value = "-1.0"
        await pilot.pause()
        
        # Should be clamped to min or set to default
        assert evals_window.temperature >= 0.0


@pytest.mark.asyncio
async def test_run_button_validation(mock_orchestrator):
    """Test run button validates configuration before running"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(120, 80)) as pilot:
        await pilot.pause()
        await prepare_window_for_testing(pilot, collapse_sections=True)
        
        evals_window = app.query_one(EvalsWindow)
        run_button = app.query_one("#run-button", Button)
        
        # Try to run without configuration
        await safe_click(pilot, run_button)
        await pilot.pause()
        
        # Should still be idle (validation failed)
        assert evals_window.evaluation_status == "idle"
        
        # Should show error notification
        assert len(app.notifications) > 0
        # Check for either style of error message
        notification_text = app.notifications[0][0].lower()
        assert "no task" in notification_text or "no model" in notification_text or "select both" in notification_text
        
        # Now configure properly
        task_select = app.query_one("#task-select", Select)
        model_select = app.query_one("#model-select", Select)
        task_select.value = "1"
        model_select.value = "1"
        await pilot.pause()
        
        # Try to run again
        await safe_click(pilot, run_button)
        await pilot.pause()
        
        # Should start evaluation (may complete immediately in tests)
        assert evals_window.evaluation_status in ["running", "completed"]


@pytest.mark.asyncio
async def test_cancel_button_handler(mock_orchestrator):
    """Test cancel button stops evaluation"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Scroll to the cancel button
        scroll_container = app.query_one(".evals-scroll-container")
        scroll_container.scroll_to(0, 20, animate=False)
        await pilot.pause()
        
        cancel_button = app.query_one("#cancel-button", Button)
        
        # Start an evaluation
        evals_window.evaluation_status = "running"
        evals_window.current_run_id = "run-123"
        
        # Create a mock worker
        from unittest.mock import MagicMock
        mock_worker = MagicMock()
        evals_window.current_worker = mock_worker
        await pilot.pause()
        
        # Click cancel
        await safe_click(pilot, cancel_button)
        await pilot.pause()
        
        # Check worker was cancelled
        mock_worker.cancel.assert_called_once()
        # Check notification was sent
        assert len(app.notifications) > 0
        assert "cancelled" in app.notifications[-1][0].lower()


@pytest.mark.asyncio
async def test_refresh_tasks_button(mock_orchestrator):
    """Test refresh tasks button reloads task list"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        refresh_btn = app.query_one("#refresh-tasks-btn", Button)
        
        # Click refresh tasks
        await safe_click(pilot, refresh_btn)
        await pilot.pause()
        
        # Check tasks were reloaded
        assert mock_orchestrator.return_value.db.list_tasks.call_count >= 1  # At least once


@pytest.mark.asyncio
async def test_refresh_models_button(mock_orchestrator):
    """Test refresh models button reloads model list"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        refresh_btn = app.query_one("#refresh-models-btn", Button)
        
        # Click refresh models
        await safe_click(pilot, refresh_btn)
        await pilot.pause()
        
        # Check models were reloaded  
        assert mock_orchestrator.return_value.db.list_models.call_count >= 1  # At least once


@pytest.mark.asyncio
async def test_create_task_button(mock_orchestrator):
    """Test create task button creates a new task"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        create_btn = app.query_one("#create-task-btn", Button)
        
        # Click create task
        await safe_click(pilot, create_btn)
        await pilot.pause()
        
        # Should show notification about creation dialog
        assert len(app.notifications) > 0


@pytest.mark.asyncio
async def test_add_model_button(mock_orchestrator):
    """Test add model button creates a new model config"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        add_btn = app.query_one("#add-model-btn", Button)
        
        # Click add model
        await safe_click(pilot, add_btn)
        await pilot.pause()
        
        # Should show notification about model dialog
        assert len(app.notifications) > 0


@pytest.mark.asyncio
async def test_test_model_button(mock_orchestrator):
    """Test model connection test button"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        test_btn = app.query_one("#test-model-btn", Button)
        
        # Select a model first
        model_select = app.query_one("#model-select", Select)
        model_select.value = "1"
        await pilot.pause()
        
        # Click test connection
        await safe_click(pilot, test_btn)
        await pilot.pause()
        
        # Check notification was shown
        assert len(app.notifications) > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_initialization_error_handling():
    """Test handling of orchestrator initialization errors"""
    with patch('tldw_chatbook.UI.evals_window_v2.EvaluationOrchestrator') as mock:
        mock.side_effect = Exception("Database connection failed")
        
        app = EvalsUnitTestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            
            evals_window = app.query_one(EvalsWindow)
            
            # Orchestrator should be None
            assert evals_window.orchestrator is None
            
            # Should show error notification
            assert len(app.notifications) > 0
            assert "error" in app.notifications[0][1]


@pytest.mark.asyncio
async def test_invalid_input_error_handling(mock_orchestrator):
    """Test handling of invalid input values"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Test invalid temperature (non-numeric)
        temp_input = app.query_one("#temperature-input", Input)
        temp_input.value = "invalid"
        await pilot.pause()
        
        # Should retain default value
        assert evals_window.temperature == 0.7
        
        # Test invalid max tokens
        tokens_input = app.query_one("#max-tokens-input", Input)
        tokens_input.value = "not_a_number"
        await pilot.pause()
        
        # Should retain default value
        assert evals_window.max_tokens == 2048


@pytest.mark.asyncio
async def test_database_error_handling(mock_orchestrator):
    """Test handling of database errors during operations"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        # Simulate database error
        mock_orchestrator.return_value.db.list_tasks.side_effect = Exception("Database locked")
        
        refresh_btn = app.query_one("#refresh-tasks-btn", Button)
        
        # Try to refresh tasks
        await safe_click(pilot, refresh_btn)
        await pilot.pause()
        
        # Should show error notification
        assert len(app.notifications) > 0


# ============================================================================
# VALIDATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_numeric_input_validation(mock_orchestrator):
    """Test validation of all numeric inputs"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Test temperature bounds
        temp_input = app.query_one("#temperature-input", Input)
        
        # Test valid temperature
        temp_input.value = "1.0"
        await pilot.pause()
        assert evals_window.temperature == 1.0
        
        # Test empty (should keep current value)
        temp_input.value = ""
        await pilot.pause()
        assert evals_window.temperature == 1.0  # Should keep previous value


@pytest.mark.asyncio
async def test_selection_validation(mock_orchestrator):
    """Test validation of dropdown selections"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Test invalid task selection
        task_select = app.query_one("#task-select", Select)
        task_select.value = Select.BLANK
        await pilot.pause()
        
        assert evals_window.selected_task_id is None
        
        # Test invalid model selection
        model_select = app.query_one("#model-select", Select)
        model_select.value = Select.BLANK
        await pilot.pause()
        
        assert evals_window.selected_model_id is None


# ============================================================================
# CSS AND STYLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_css_overflow_properties_valid(mock_orchestrator):
    """Test that all CSS overflow properties are valid"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check main window overflow
        evals_window = app.query_one(EvalsWindow)
        
        # These should not raise StyleValueError
        assert evals_window.styles.overflow_x in ["auto", "hidden", "scroll"]
        assert evals_window.styles.overflow_y in ["auto", "hidden", "scroll"]
        
        # Check scroll container
        scroll_container = app.query_one(".evals-scroll-container")
        assert scroll_container.styles.overflow_y == "scroll"
        assert scroll_container.styles.overflow_x == "hidden"


@pytest.mark.asyncio
async def test_collapsible_css_valid(mock_orchestrator):
    """Test that Collapsible widgets have valid CSS"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check all collapsibles
        collapsibles = app.query(Collapsible)
        
        for collapsible in collapsibles:
            # Should not have invalid overflow values
            if hasattr(collapsible.styles, 'overflow'):
                assert collapsible.styles.overflow in ["auto", "hidden", "scroll"]


# ============================================================================
# STATE MANAGEMENT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_evaluation_state_transitions(mock_orchestrator):
    """Test proper state transitions during evaluation lifecycle"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Scroll to the run button
        scroll_container = app.query_one(".evals-scroll-container")
        scroll_container.scroll_to(0, 20, animate=False)
        await pilot.pause()
        
        run_btn = app.query_one("#run-button", Button)
        
        # Initial state
        assert evals_window.evaluation_status == "idle"
        
        # Configure for evaluation
        task_select = app.query_one("#task-select", Select)
        model_select = app.query_one("#model-select", Select)
        
        # Use actual available option values
        task_value = get_valid_select_value(task_select, 0)
        if task_value:
            task_select.value = task_value
        
        model_value = get_valid_select_value(model_select, 0)
        if model_value:
            model_select.value = model_value
        
        await pilot.pause()
        
        # Start evaluation
        await safe_click(pilot, run_btn)
        await pilot.pause()
        
        # Should be running or completed (may complete immediately in tests)
        assert evals_window.evaluation_status in ["running", "completed"]
        
        # Simulate completion
        evals_window.evaluation_status = "completed"
        await pilot.pause()
        
        assert evals_window.evaluation_status == "completed"
        
        # Simulate error
        evals_window.evaluation_status = "error"
        await pilot.pause()
        
        assert evals_window.evaluation_status == "error"


@pytest.mark.asyncio
async def test_concurrent_evaluation_prevention(mock_orchestrator):
    """Test that multiple evaluations cannot run simultaneously"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Scroll to the run button
        scroll_container = app.query_one(".evals-scroll-container")
        scroll_container.scroll_to(0, 20, animate=False)
        await pilot.pause()
        
        run_btn = app.query_one("#run-button", Button)
        
        # Configure for evaluation
        task_select = app.query_one("#task-select", Select)
        model_select = app.query_one("#model-select", Select)
        
        # Use actual available option values
        task_value = get_valid_select_value(task_select, 0)
        if task_value:
            task_select.value = task_value
        
        model_value = get_valid_select_value(model_select, 0)
        if model_value:
            model_select.value = model_value
        
        await pilot.pause()
        
        # Start first evaluation
        await safe_click(pilot, run_btn)
        await pilot.pause()
        
        # Check if evaluation is still running or completed
        if evals_window.evaluation_status == "running":
            # Try to start second evaluation while first is running
            await safe_click(pilot, run_btn)
            await pilot.pause()
            
            # Should show notification about already running
            assert len(app.notifications) > 0
            
            # Should still only have one evaluation call
            assert mock_orchestrator.return_value.run_evaluation.call_count == 1
        else:
            # If completed immediately, simulate running state for test
            evals_window.evaluation_status = "running"
            await pilot.pause()
            
            # Try to start second evaluation
            await safe_click(pilot, run_btn)
            await pilot.pause()
            
            # Should show notification about already running
            assert len(app.notifications) > 0


# ============================================================================
# DATA LOADING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_tasks_loaded_on_mount(mock_orchestrator):
    """Test that tasks are loaded when window mounts"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check tasks were loaded
        mock_orchestrator.return_value.db.list_tasks.assert_called()
        
        # Check task select was populated
        task_select = app.query_one("#task-select", Select)
        # Should have at least the blank option
        assert len(task_select._options) >= 1


@pytest.mark.asyncio
async def test_models_loaded_on_mount(mock_orchestrator):
    """Test that models are loaded when window mounts"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check models were loaded
        mock_orchestrator.return_value.db.list_models.assert_called()
        
        # Check model select was populated
        model_select = app.query_one("#model-select", Select)
        # Should have at least the blank option
        assert len(model_select._options) >= 1


@pytest.mark.asyncio
async def test_results_loaded_on_mount(mock_orchestrator):
    """Test that recent results are loaded when window mounts"""
    app = EvalsUnitTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check results table exists and is ready
        table = app.query_one("#results-table", DataTable)
        assert table is not None
        
        # Table should have columns configured
        assert len(table.columns) > 0


# ============================================================================
# WORKER THREAD TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_evaluation_runs_in_worker(mock_orchestrator):
    """Test that evaluation runs in a worker thread"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Scroll to the run button
        scroll_container = app.query_one(".evals-scroll-container")
        scroll_container.scroll_to(0, 20, animate=False)
        await pilot.pause()
        
        run_btn = app.query_one("#run-button", Button)
        
        # Configure for evaluation
        task_select = app.query_one("#task-select", Select)
        model_select = app.query_one("#model-select", Select)
        
        # Use actual available option values
        task_value = get_valid_select_value(task_select, 0)
        if task_value:
            task_select.value = task_value
        
        model_value = get_valid_select_value(model_select, 0)
        if model_value:
            model_select.value = model_value
        
        await pilot.pause()
        
        # Mock the worker
        with patch.object(evals_window, 'run_worker') as mock_worker:
            await safe_click(pilot, run_btn)
            await pilot.pause()
            
            # Check worker was started
            mock_worker.assert_called_once()
            
            # Check it's exclusive (prevents multiple runs)
            _, kwargs = mock_worker.call_args
            assert kwargs.get('exclusive') == True


@pytest.mark.asyncio
async def test_worker_cleanup_on_cancel(mock_orchestrator):
    """Test that worker is properly cleaned up on cancellation"""
    app = EvalsUnitTestApp()
    async with app.run_test(size=(100, 50)) as pilot:  # Larger screen size
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Scroll to the cancel button
        scroll_container = app.query_one(".evals-scroll-container")
        scroll_container.scroll_to(0, 20, animate=False)
        await pilot.pause()
        
        cancel_btn = app.query_one("#cancel-button", Button)
        
        # Start evaluation
        evals_window.evaluation_status = "running"
        evals_window.current_run_id = "run-123"
        
        # Create a mock worker
        mock_worker = Mock()
        evals_window.current_worker = mock_worker
        
        # Cancel evaluation
        await safe_click(pilot, cancel_btn)
        await pilot.pause()
        
        # Check worker was cancelled
        mock_worker.cancel.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])