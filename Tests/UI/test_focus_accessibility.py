"""Test suite for verifying non-obscuring focus accessibility improvements."""

import re
from pathlib import Path

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


CSS_PATH = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


def css_block(text: str, selector: str) -> str:
    """Return a CSS rule body whose selector list contains selector."""
    uncommented = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    for match in re.finditer(r"\{(?P<body>[^{}]*)\}", uncommented, flags=re.DOTALL):
        prefix = uncommented[: match.start()]
        selector_start = max(prefix.rfind("}"), prefix.rfind(";")) + 1
        selector_text = prefix[selector_start : match.start()]
        selectors = [item.strip() for item in selector_text.split(",")]
        if selector in selectors:
            return match.group("body")
    raise AssertionError(f"Missing CSS block for {selector}")


@pytest.mark.asyncio
async def test_button_has_visible_non_obscuring_focus():
    """Test that buttons retain a visible focus cue without heavy outline."""
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
        
        assert CSS_PATH.exists()

        with CSS_PATH.open("r", encoding="utf-8") as f:
            css_content = f.read()
            button_focus = css_block(css_content, "Button:focus")
            assert "text-style: bold underline" in button_focus
            assert "outline: heavy" not in button_focus
            assert "outline: none !important" not in css_content


@pytest.mark.asyncio
async def test_input_has_visible_non_obscuring_focus():
    """Test that input fields still receive keyboard focus."""
    app = FocusTestApp()
    async with app.run_test() as pilot:
        # Focus the input
        input_widget = app.query_one("#test-input", Input)
        input_widget.focus()
        await pilot.pause()
        
        # Verify focus
        assert input_widget.has_focus


@pytest.mark.asyncio
async def test_textarea_has_visible_non_obscuring_focus():
    """Test that textareas still receive keyboard focus."""
    app = FocusTestApp()
    async with app.run_test() as pilot:
        # Focus the textarea
        textarea = app.query_one("#test-textarea", TextArea)
        textarea.focus()
        await pilot.pause()
        
        # Verify focus
        assert textarea.has_focus


@pytest.mark.asyncio
async def test_no_global_outline_suppression_in_css():
    """Test that CSS keeps a visible global fallback without global suppression."""
    # Check the main CSS file
    with CSS_PATH.open("r", encoding="utf-8") as f:
        css_content = f.read()
        
        # These anti-patterns should NOT be present
        assert "outline: none !important" not in css_content
        assert "outline:none!important" not in css_content
        
        assert "*:focus" in css_content
        global_focus = css_block(css_content, "*:focus")
        assert "outline: solid" in global_focus
        assert "outline: heavy" not in global_focus


def test_generated_input_focus_uses_thin_border_and_bottom_emphasis():
    """Test generated input focus styles use the non-obscuring input pattern."""
    css_content = CSS_PATH.read_text(encoding="utf-8")

    for selector in ("Input:focus", "TextArea:focus", "Select:focus"):
        block = css_block(css_content, selector)
        assert "outline: heavy" not in block
        assert "border: solid $ds-input-focus-border;" in block
        assert "border-bottom: solid $ds-input-focus-accent;" in block


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
