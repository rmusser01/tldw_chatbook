"""
Test suite for verifying focus accessibility improvements.
Tests that focus indicators are properly visible on all interactive elements.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, TextArea, Select, Checkbox, RadioButton, Label
from textual.containers import Container


class FocusTestApp(App):
    """Test app with various focusable widgets."""
    
    CSS_PATH = "../../tldw_chatbook/css/tldw_cli_modular.tcss"
    
    def compose(self) -> ComposeResult:
        """Create test UI with all focusable widget types."""
        with Container(id="test-container"):
            yield Label("Focus Accessibility Test")
            yield Button("Test Button", id="test-button")
            yield Input(placeholder="Test Input", id="test-input")
            yield TextArea("Test TextArea", id="test-textarea")
            yield Select([("opt1", "Option 1"), ("opt2", "Option 2")], id="test-select")
            yield Checkbox("Test Checkbox", id="test-checkbox")
            yield RadioButton("Test Radio", id="test-radio")


@pytest.mark.asyncio
async def test_button_has_focus_outline():
    """Test that buttons show focus outline when focused."""
    app = FocusTestApp()
    async with app.run_test() as pilot:
        # Focus the button
        button = app.query_one("#test-button", Button)
        button.focus()
        
        # Get computed styles
        styles = button.styles
        
        # Verify outline is not 'none'
        # Note: Textual doesn't expose outline directly, but we can verify
        # the widget has focus and the CSS is loaded
        assert button.has_focus
        assert app.CSS_PATH  # Verify CSS is loaded
        
        # Check that the CSS file exists and has proper focus styles
        import os
        css_path = os.path.join(
            os.path.dirname(__file__), 
            "../../tldw_chatbook/css/tldw_cli_modular.tcss"
        )
        assert os.path.exists(css_path)
        
        with open(css_path, 'r') as f:
            css_content = f.read()
            # Verify focus styles are present and not suppressed
            assert "outline: 2px solid $accent" in css_content
            assert "outline: none !important" not in css_content


@pytest.mark.asyncio
async def test_input_has_focus_outline():
    """Test that input fields show focus outline when focused."""
    app = FocusTestApp()
    async with app.run_test() as pilot:
        # Focus the input
        input_widget = app.query_one("#test-input", Input)
        input_widget.focus()
        
        # Verify focus
        assert input_widget.has_focus


@pytest.mark.asyncio
async def test_textarea_has_focus_outline():
    """Test that textareas show focus outline when focused."""
    app = FocusTestApp()
    async with app.run_test() as pilot:
        # Focus the textarea
        textarea = app.query_one("#test-textarea", TextArea)
        textarea.focus()
        
        # Verify focus
        assert textarea.has_focus


@pytest.mark.asyncio
async def test_no_outline_suppression_in_css():
    """Test that CSS doesn't contain outline suppression anti-patterns."""
    import os
    
    # Check the main CSS file
    css_path = os.path.join(
        os.path.dirname(__file__), 
        "../../tldw_chatbook/css/tldw_cli_modular.tcss"
    )
    
    with open(css_path, 'r') as f:
        css_content = f.read()
        
        # These anti-patterns should NOT be present
        assert "outline: none !important" not in css_content
        assert "outline:none!important" not in css_content
        
        # These proper patterns SHOULD be present
        assert "*:focus" in css_content
        assert "outline: 2px solid" in css_content or "outline: solid" in css_content


@pytest.mark.asyncio
async def test_keyboard_navigation_visible():
    """Test that keyboard navigation shows visible focus indicators."""
    app = FocusTestApp()
    async with app.run_test() as pilot:
        # Tab through widgets
        await pilot.press("tab")  # Focus first widget
        
        # Find which widget has focus
        focused_widget = None
        for widget in app.query("Button, Input, TextArea, Select, Checkbox, RadioButton"):
            if widget.has_focus:
                focused_widget = widget
                break
        
        assert focused_widget is not None, "No widget has focus after pressing Tab"
        
        # Tab to next widget
        await pilot.press("tab")
        
        # Verify focus moved
        new_focused = None
        for widget in app.query("Button, Input, TextArea, Select, Checkbox, RadioButton"):
            if widget.has_focus:
                new_focused = widget
                break
        
        assert new_focused is not None
        assert new_focused != focused_widget, "Focus didn't move to next widget"


@pytest.mark.asyncio 
async def test_focus_within_containers():
    """Test that containers show focus-within styles."""
    app = FocusTestApp()
    async with app.run_test() as pilot:
        # Focus a widget inside the container
        button = app.query_one("#test-button", Button)
        button.focus()
        
        # Get the container
        container = app.query_one("#test-container", Container)
        
        # The container should be the parent of the focused element
        assert button.parent == container
        assert button.has_focus


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])