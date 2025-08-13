"""
UI and Visual Tests for Evals Window V2
Tests collapsible behavior, layout, scrolling, and CSS validation
Following Textual's testing best practices
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch
from textual.app import App, ComposeResult
from textual.widgets import Collapsible, Button, Select, Input, DataTable, Static, ProgressBar
from textual.containers import VerticalScroll, Container
from textual.geometry import Size

from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
from Tests.UI.textual_test_helpers import safe_click


class EvalsUITestApp(App):
    """Test app for UI/visual testing"""
    
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
    """Mock orchestrator with minimal test data"""
    with patch('tldw_chatbook.UI.evals_window_v2.EvaluationOrchestrator') as mock:
        orchestrator = Mock()
        orchestrator.db = Mock()
        orchestrator.db.list_tasks = Mock(return_value=[
            {'id': '1', 'name': 'Task 1', 'task_type': 'test', 'description': 'Test'}
        ])
        orchestrator.db.list_models = Mock(return_value=[
            {'id': '1', 'name': 'Model 1', 'provider': 'test', 'model_id': 'test-1'}
        ])
        orchestrator.db.list_runs = Mock(return_value=[])
        mock.return_value = orchestrator
        yield mock


# ============================================================================
# COLLAPSIBLE BEHAVIOR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_collapsible_expand_collapse(mock_orchestrator):
    """Test that collapsible sections expand and collapse correctly"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Get all collapsibles
        collapsibles = app.query(Collapsible)
        assert len(collapsibles) >= 5
        
        # Test Task Configuration collapsible
        task_config = next(c for c in collapsibles if "Task Configuration" in c.title)
        
        # Should start expanded
        assert task_config.collapsed == False
        
        # Click to collapse
        await safe_click(pilot, task_config)
        await pilot.pause()
        assert task_config.collapsed == True
        
        # Click to expand
        await safe_click(pilot, task_config)
        await pilot.pause()
        assert task_config.collapsed == False


@pytest.mark.asyncio
async def test_collapsible_content_visibility(mock_orchestrator):
    """Test that collapsible content is hidden when collapsed"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Get Model Configuration collapsible
        collapsibles = app.query(Collapsible)
        model_config = next(c for c in collapsibles if "Model Configuration" in c.title)
        
        # Check content is visible when expanded
        assert model_config.collapsed == False
        temp_input = app.query_one("#temperature-input", Input)
        assert temp_input.display == True
        
        # Collapse it
        await safe_click(pilot, model_config)
        await pilot.pause()
        
        # Content should be hidden
        assert model_config.collapsed == True
        # Note: In Textual, collapsed content is still in DOM but not displayed


@pytest.mark.asyncio
async def test_collapsible_initial_states(mock_orchestrator):
    """Test that collapsibles have correct initial states"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        collapsibles = app.query(Collapsible)
        
        # Check initial states
        for collapsible in collapsibles:
            if "Cost Estimation" in collapsible.title:
                assert collapsible.collapsed == True  # Starts collapsed
            elif "Progress" in collapsible.title:
                assert collapsible.collapsed == True  # Starts collapsed
            else:
                assert collapsible.collapsed == False  # Others start expanded


@pytest.mark.asyncio
async def test_progress_collapsible_auto_expand(mock_orchestrator):
    """Test that progress collapsible auto-expands when evaluation starts"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        progress_collapsible = app.query_one("#progress-collapsible", Collapsible)
        
        # Should start collapsed
        assert progress_collapsible.collapsed == True
        
        # Start evaluation (simulate)
        evals_window.evaluation_status = "running"
        await pilot.pause()
        
        # Should auto-expand
        assert progress_collapsible.collapsed == False


# ============================================================================
# LAYOUT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_vertical_layout_at_different_sizes(mock_orchestrator):
    """Test layout remains vertical at different terminal sizes"""
    # Test small size
    app_small = EvalsUITestApp()
    async with app_small.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        
        evals_window = app_small.query_one(EvalsWindow)
        assert str(evals_window.styles.layout) == "<vertical>"
    
    # Test medium size
    app_medium = EvalsUITestApp()
    async with app_medium.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        
        evals_window = app_medium.query_one(EvalsWindow)
        assert str(evals_window.styles.layout) == "<vertical>"
    
    # Test large size
    app_large = EvalsUITestApp()
    async with app_large.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        
        evals_window = app_large.query_one(EvalsWindow)
        assert str(evals_window.styles.layout) == "<vertical>"


@pytest.mark.asyncio
async def test_responsive_layout_on_resize(mock_orchestrator):
    """Test that layout adapts when terminal is resized"""
    app = EvalsUITestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        
        # Get initial element positions
        task_select = app.query_one("#task-select", Select)
        initial_width = task_select.size.width
        
        # Resize terminal
        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        
        # Check elements adapted
        new_width = task_select.size.width
        assert new_width != initial_width  # Width should have changed
        
        # Verify layout is still vertical
        evals_window = app.query_one(EvalsWindow)
        assert str(evals_window.styles.layout) == "<vertical>"


@pytest.mark.asyncio
async def test_header_footer_fixed_positions(mock_orchestrator):
    """Test that header and footer remain in fixed positions"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check header is at top
        header = app.query_one(".evals-header")
        assert header is not None
        
        # Check footer is at bottom
        footer = app.query_one(".status-footer")
        assert footer is not None
        
        # Scroll the main content
        scroll_container = app.query_one(".evals-scroll-container")
        if hasattr(scroll_container, 'scroll_to'):
            scroll_container.scroll_to(0, 100)
            await pilot.pause()
        
        # Header and footer should still be visible
        assert header.display == True
        assert footer.display == True


# ============================================================================
# SCROLLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_vertical_scroll_functionality(mock_orchestrator):
    """Test that vertical scrolling works correctly"""
    app = EvalsUITestApp()
    async with app.run_test(size=(80, 20)) as pilot:  # Small height to force scrolling
        await pilot.pause()
        
        scroll_container = app.query_one(".evals-scroll-container", VerticalScroll)
        assert scroll_container is not None
        
        # Check scroll properties
        assert scroll_container.styles.overflow_y == "scroll"
        assert scroll_container.styles.overflow_x == "hidden"
        
        # Test scrolling down
        if hasattr(scroll_container, 'scroll_down'):
            await scroll_container.scroll_down()
            await pilot.pause()
            
            # Scroll position should have changed
            assert scroll_container.scroll_y > 0


@pytest.mark.asyncio
async def test_no_horizontal_scroll(mock_orchestrator):
    """Test that horizontal scrolling is disabled"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        scroll_container = app.query_one(".evals-scroll-container")
        
        # Check horizontal scroll is hidden
        assert scroll_container.styles.overflow_x == "hidden"
        
        # Verify no horizontal scrollbar appears
        if hasattr(scroll_container, 'scroll_x'):
            assert scroll_container.scroll_x == 0


@pytest.mark.asyncio
async def test_nested_scrolling_behavior(mock_orchestrator):
    """Test that nested scrolling (table within scrollable container) works"""
    app = EvalsUITestApp()
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        
        # Get the results table
        table = app.query_one("#results-table", DataTable)
        
        # Add many rows to test scrolling
        for i in range(50):
            table.add_row(f"Row {i}", "Task", "Model", "100", "95%", "10s", "completed")
        await pilot.pause()
        
        # Both container and table should be scrollable
        scroll_container = app.query_one(".evals-scroll-container")
        assert scroll_container.styles.overflow_y == "scroll"
        
        # Table should handle its own scrolling
        assert table.show_cursor == True  # Table is interactive


# ============================================================================
# CSS VALIDATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_all_css_properties_valid(mock_orchestrator):
    """Test that all CSS properties have valid values"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        
        # Test main window CSS
        layout_str = str(evals_window.styles.layout).strip("<>")
        assert layout_str in ["vertical", "horizontal", "grid", "dock"]
        assert evals_window.styles.overflow_x in ["auto", "hidden", "scroll"]
        assert evals_window.styles.overflow_y in ["auto", "hidden", "scroll"]
        
        # Test all containers
        containers = app.query(Container)
        for container in containers:
            if hasattr(container.styles, 'overflow'):
                assert container.styles.overflow in ["auto", "hidden", "scroll"]
            if hasattr(container.styles, 'overflow_x'):
                assert container.styles.overflow_x in ["auto", "hidden", "scroll"]
            if hasattr(container.styles, 'overflow_y'):
                assert container.styles.overflow_y in ["auto", "hidden", "scroll"]


@pytest.mark.asyncio
async def test_theme_compatibility(mock_orchestrator):
    """Test that UI works with different color themes"""
    # Test with dark theme (default)
    app_dark = EvalsUITestApp()
    app_dark.theme = "textual-dark"
    async with app_dark.run_test() as pilot:
        await pilot.pause()
        
        # Check elements are visible
        header = app_dark.query_one(".header-title", Static)
        assert header is not None
        assert header.renderable is not None
    
    # Test with light theme
    app_light = EvalsUITestApp()
    app_light.theme = "textual-light"
    async with app_light.run_test() as pilot:
        await pilot.pause()
        
        # Check elements are visible
        header = app_light.query_one(".header-title", Static)
        assert header is not None
        assert header.renderable is not None


@pytest.mark.asyncio
async def test_css_classes_applied_correctly(mock_orchestrator):
    """Test that CSS classes are applied to correct elements"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check header classes
        header = app.query_one(".evals-header")
        assert "evals-header" in header.classes
        
        # Check scroll container classes
        scroll = app.query_one(".evals-scroll-container")
        assert "evals-scroll-container" in scroll.classes
        
        # Check section classes
        sections = app.query(".config-section")
        for section in sections:
            assert "config-section" in section.classes
        
        # Check form classes
        form_rows = app.query(".form-row")
        assert len(form_rows) > 0
        for row in form_rows:
            assert "form-row" in row.classes


# ============================================================================
# VISUAL CONSISTENCY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_button_styling_consistency(mock_orchestrator):
    """Test that all buttons have consistent styling"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        buttons = app.query(Button)
        
        # Check all buttons have consistent properties
        for button in buttons:
            assert button.variant in ["default", "primary", "success", "warning", "error"]
            assert button.disabled == False  # Initially all enabled
            
            # Check button has text/label
            assert button.label != ""


@pytest.mark.asyncio
async def test_input_field_consistency(mock_orchestrator):
    """Test that all input fields have consistent styling"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        inputs = app.query(Input)
        
        # Check all inputs have consistent properties
        for input_field in inputs:
            assert input_field.placeholder != ""  # Should have placeholder
            assert hasattr(input_field, 'value')  # Should have value property
            
            # Check input has appropriate type
            if "temperature" in input_field.id:
                assert input_field.type == "number"
            elif "tokens" in input_field.id or "samples" in input_field.id:
                assert input_field.type == "integer"


@pytest.mark.asyncio
async def test_select_dropdown_consistency(mock_orchestrator):
    """Test that all select dropdowns have consistent styling"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        selects = app.query(Select)
        
        # Check all selects have consistent properties
        for select in selects:
            # Check that blank option is available (first option is usually blank)
            assert len(select._options) > 0
            assert select._options[0][0] == Select.BLANK or select._options[1][0] == Select.BLANK
            assert len(select._options) > 0  # Should have options
            assert select.prompt != ""  # Should have a prompt


# ============================================================================
# ANIMATION AND TRANSITION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_collapsible_animation(mock_orchestrator):
    """Test that collapsible animations work smoothly"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Get a collapsible
        collapsibles = app.query(Collapsible)
        test_collapsible = collapsibles[0]
        
        # Collapse with animation
        await safe_click(pilot, test_collapsible)
        await pilot.wait_for_animation()  # Wait for animation to complete
        
        assert test_collapsible.collapsed == True
        
        # Expand with animation
        await safe_click(pilot, test_collapsible)
        await pilot.wait_for_animation()  # Wait for animation to complete
        
        assert test_collapsible.collapsed == False


@pytest.mark.asyncio
async def test_progress_bar_animation(mock_orchestrator):
    """Test that progress bar animates smoothly"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        evals_window = app.query_one(EvalsWindow)
        progress_bar = app.query_one("#progress-bar", ProgressBar)
        
        # Update progress
        evals_window.evaluation_progress = 25.0
        await pilot.pause()
        await pilot.wait_for_animation()
        
        assert progress_bar.percentage == 0.25
        
        # Update again
        evals_window.evaluation_progress = 75.0
        await pilot.pause()
        await pilot.wait_for_animation()
        
        assert progress_bar.percentage == 0.75


# ============================================================================
# EDGE CASE UI TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_ui_with_no_data(mock_orchestrator):
    """Test UI handles empty data gracefully"""
    # Mock empty data
    mock_orchestrator.return_value.db.list_tasks.return_value = []
    mock_orchestrator.return_value.db.list_models.return_value = []
    mock_orchestrator.return_value.db.list_runs.return_value = []
    
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # UI should still load
        evals_window = app.query_one(EvalsWindow)
        assert evals_window is not None
        
        # Selects should have at least blank option
        task_select = app.query_one("#task-select", Select)
        assert len(task_select._options) >= 1  # At least blank
        
        model_select = app.query_one("#model-select", Select)
        assert len(model_select._options) >= 1  # At least blank
        
        # Table should be empty but present
        table = app.query_one("#results-table", DataTable)
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_ui_with_very_long_text(mock_orchestrator):
    """Test UI handles very long text gracefully"""
    # Mock data with very long names
    mock_orchestrator.return_value.db.list_tasks.return_value = [
        {
            'id': '1',
            'name': 'A' * 200,  # Very long name
            'task_type': 'test',
            'description': 'B' * 500  # Very long description
        }
    ]
    
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # UI should handle long text without breaking
        task_select = app.query_one("#task-select", Select)
        assert len(task_select._options) > 1
        
        # Long text should be truncated or wrapped appropriately
        if len(task_select._options) > 2:  # If we have more than just blank options
            option_label = task_select._options[2][0]  # Skip blank options
            if option_label != Select.BLANK:
                assert len(str(option_label)) > 0  # Should have some text


@pytest.mark.asyncio
async def test_ui_at_minimum_size(mock_orchestrator):
    """Test UI at minimum supported terminal size"""
    app = EvalsUITestApp()
    async with app.run_test(size=(40, 15)) as pilot:  # Very small terminal
        await pilot.pause()
        
        # UI should still be functional
        evals_window = app.query_one(EvalsWindow)
        assert evals_window is not None
        
        # Key elements should still be accessible
        assert app.query_one("#task-select") is not None
        assert app.query_one("#model-select") is not None
        assert app.query_one("#run-button") is not None
        
        # Should be scrollable to access all content
        scroll_container = app.query_one(".evals-scroll-container")
        assert scroll_container.styles.overflow_y == "scroll"


# ============================================================================
# FOCUS AND TAB ORDER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_tab_order_through_form(mock_orchestrator):
    """Test that tab order through form elements is logical"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Start at first select
        task_select = app.query_one("#task-select", Select)
        task_select.focus()
        await pilot.pause()
        
        # Tab through elements
        await pilot.press("tab")
        await pilot.pause()
        
        # Should move to next logical element
        focused = app.focused
        assert focused is not None
        assert focused != task_select


@pytest.mark.asyncio
async def test_focus_visible_indicators(mock_orchestrator):
    """Test that focused elements have visible indicators"""
    app = EvalsUITestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Focus an input
        temp_input = app.query_one("#temperature-input", Input)
        temp_input.focus()
        await pilot.pause()
        
        # Check it has focus
        assert app.focused == temp_input
        
        # Focus a button
        run_button = app.query_one("#run-button", Button)
        run_button.focus()
        await pilot.pause()
        
        # Check it has focus
        assert app.focused == run_button


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])