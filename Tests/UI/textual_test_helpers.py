"""
Textual Test Helpers - Following Official Best Practices
Comprehensive utilities for testing Textual apps
"""

from typing import Optional, Any, Union, List
from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Select, Button, Collapsible, Input
from textual.widget import Widget
import asyncio


async def safe_click(pilot: Pilot, selector_or_widget: Union[str, Widget], force_visible: bool = True) -> bool:
    """
    Safely click a widget following Textual best practices.
    NEVER throws OutOfBounds exceptions.
    
    Args:
        pilot: The test pilot
        selector_or_widget: CSS selector string or widget instance
        force_visible: Whether to force widget into view
        
    Returns:
        True if click succeeded, False otherwise
    """
    app = pilot.app
    
    try:
        # Get the widget
        if isinstance(selector_or_widget, str):
            widget = app.query_one(selector_or_widget)
        else:
            widget = selector_or_widget
            
        if force_visible:
            # Try multiple approaches to make widget visible
            try:
                # First try screen-level scrolling
                app.screen.scroll_to_widget(widget, animate=False)
                await pilot.pause()
            except:
                pass
                
            try:
                # Then try container-level scrolling
                for container in app.query(".evals-scroll-container, VerticalScroll, ScrollableContainer"):
                    try:
                        container.scroll_to_widget(widget, animate=False)
                        await pilot.pause()
                        break
                    except:
                        continue
            except:
                pass
                
        # Now try to click
        await pilot.click(selector_or_widget)
        await pilot.pause()
        return True
        
    except Exception as e:
        # NEVER re-raise OutOfBounds - just return False
        return False


async def prepare_window_for_testing(pilot: Pilot, collapse_sections: bool = True) -> None:
    """
    Prepare the EvalsWindow for testing by setting initial state.
    
    Args:
        pilot: The test pilot
        collapse_sections: Whether to collapse all collapsibles to save space
    """
    app = pilot.app
    await pilot.pause()
    
    if collapse_sections:
        # Collapse all collapsibles to make more widgets visible
        for collapsible in app.query(Collapsible):
            if not collapsible.collapsed:
                collapsible.collapsed = True
        await pilot.pause()


def filter_select_options(options: List[tuple]) -> List[tuple]:
    """
    Filter out Select.BLANK from options list.
    
    Args:
        options: List of option tuples from Select widget
        
    Returns:
        Filtered list without blank options
    """
    # Filter out all forms of blank options
    return [opt for opt in options 
            if opt[0] != Select.BLANK 
            and opt[1] != Select.BLANK 
            and opt[0] != '' 
            and opt[1] is not None]


def get_option_labels(select_widget: Select) -> List[str]:
    """
    Get string labels from a Select widget, filtering out blanks.
    
    Args:
        select_widget: The Select widget
        
    Returns:
        List of string labels
    """
    options = filter_select_options(select_widget._options)
    return [str(opt[0]) for opt in options]


def layout_to_string(layout) -> str:
    """
    Convert a layout object to string for comparison.
    
    Args:
        layout: Layout object from styles
        
    Returns:
        String representation
    """
    layout_str = str(layout)
    # Handle both <vertical> and vertical formats
    if layout_str.startswith("<") and layout_str.endswith(">"):
        return layout_str
    return f"<{layout_str}>"


async def expand_collapsible(pilot: Pilot, title_or_id: str) -> bool:
    """
    Expand a specific collapsible section.
    
    Args:
        pilot: The test pilot
        title_or_id: Title text or ID of the collapsible
        
    Returns:
        True if expanded successfully
    """
    app = pilot.app
    
    try:
        # Find collapsible by ID first
        if title_or_id.startswith("#"):
            collapsible = app.query_one(title_or_id, Collapsible)
        else:
            # Find by title
            for collapsible in app.query(Collapsible):
                if title_or_id in collapsible.title:
                    break
            else:
                return False
                
        if collapsible.collapsed:
            collapsible.collapsed = False
            await pilot.pause()
            
        return True
    except:
        return False


async def set_select_value(pilot: Pilot, select_id: str, value: Any) -> bool:
    """
    Set a Select widget's value safely.
    
    Args:
        pilot: The test pilot
        select_id: ID of the select widget
        value: Value to set
        
    Returns:
        True if successful
    """
    try:
        select = pilot.app.query_one(select_id, Select)
        select.value = value
        await pilot.pause()
        return True
    except:
        return False


async def get_visible_buttons(app: App) -> List[Button]:
    """
    Get all buttons that are currently visible in the viewport.
    
    Args:
        app: The Textual app
        
    Returns:
        List of visible Button widgets
    """
    visible_buttons = []
    screen_region = app.screen.region
    
    for button in app.query(Button):
        try:
            if button.visible and button.region.overlaps(screen_region):
                visible_buttons.append(button)
        except:
            pass
            
    return visible_buttons


class TestAppWithLargeScreen(App):
    """Base test app with larger default screen size."""
    
    DEFAULT_CSS = """
    Screen {
        width: 120;
        height: 80;
    }
    """


async def wait_for_worker(pilot: Pilot, timeout: float = 1.0) -> bool:
    """
    Wait for any running workers to complete.
    
    Args:
        pilot: The test pilot
        timeout: Maximum time to wait
        
    Returns:
        True if workers completed
    """
    app = pilot.app
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        if not app._workers:
            return True
        await pilot.pause()
        
    return False


async def assert_click_succeeded(pilot: Pilot, selector_or_widget: Union[str, Widget], message: str = None) -> None:
    """
    Assert that a click succeeded, with helpful error message.
    
    Args:
        pilot: The test pilot
        selector_or_widget: Widget to click
        message: Optional custom error message
    """
    result = await safe_click(pilot, selector_or_widget)
    if not result:
        if message:
            assert False, message
        else:
            widget_str = selector_or_widget if isinstance(selector_or_widget, str) else str(selector_or_widget)
            assert False, f"Failed to click {widget_str} - widget may be outside viewport"


def get_valid_select_value(select_widget: Select, index: int = 0) -> Optional[Any]:
    """
    Get a valid value from Select widget options.
    
    Args:
        select_widget: The Select widget
        index: Index of the option to get (0 = first non-blank option)
        
    Returns:
        The value of the option at the given index, or None if not available
    """
    options = filter_select_options(select_widget._options)
    if options and index < len(options):
        return options[index][1]
    return None


async def focus_and_type(pilot: Pilot, input_widget: Input, text: str) -> None:
    """
    Focus an input widget and type text into it.
    
    Args:
        pilot: The test pilot
        input_widget: The Input widget to type into
        text: The text to type
    """
    input_widget.focus()
    await pilot.pause()
    
    # Clear existing text first
    input_widget.clear()
    await pilot.pause()
    
    # Type each character
    for char in text:
        await pilot.press(char)
    await pilot.pause()


async def set_select_by_index(pilot: Pilot, select_widget: Select, index: int = 0) -> bool:
    """
    Set a Select widget's value by option index.
    
    Args:
        pilot: The test pilot
        select_widget: The Select widget
        index: Index of the option to select (0 = first non-blank)
        
    Returns:
        True if successful, False otherwise
    """
    value = get_valid_select_value(select_widget, index)
    if value is not None:
        select_widget.value = value
        await pilot.pause()
        return True
    return False


# Performance test helpers use existing mock fixtures, not new mocks