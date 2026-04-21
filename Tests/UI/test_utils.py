"""
Test utilities for Evals Window tests
"""

from textual.widgets import Button
from textual.pilot import Pilot


async def safe_click(pilot: Pilot, widget_or_selector, max_attempts: int = 3) -> bool:
    """
    Safely click a widget, scrolling to it if needed.
    
    Args:
        pilot: The test pilot
        widget_or_selector: Widget instance or CSS selector
        max_attempts: Maximum scroll attempts
        
    Returns:
        True if click was successful, False otherwise
    """
    app = pilot.app
    
    for attempt in range(max_attempts):
        try:
            await pilot.click(widget_or_selector)
            return True
        except Exception as e:
            if "OutOfBounds" in str(e):
                # Try to scroll to the widget
                try:
                    if isinstance(widget_or_selector, str):
                        widget = app.query_one(widget_or_selector)
                    else:
                        widget = widget_or_selector
                    
                    # Find scrollable container
                    scroll_container = app.query_one(".evals-scroll-container")
                    if scroll_container:
                        # Scroll to widget position
                        scroll_container.scroll_to_widget(widget, animate=False)
                        await pilot.pause()
                    else:
                        # Try scrolling the screen
                        app.screen.scroll_to_widget(widget, animate=False)
                        await pilot.pause()
                except:
                    pass
            else:
                # Some other error
                return False
    
    return False