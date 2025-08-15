"""
Debug version of EvalsWindow with explicit visibility
"""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

class EvalsWindowDebug(Container):
    """Super simple debug version"""
    
    def __init__(self, app_instance=None, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
    
    def compose(self) -> ComposeResult:
        # Create properly styled widgets
        header = Static("🔴 EVALS WINDOW IS VISIBLE 🔴")
        header.styles.color = "red"
        header.styles.background = "yellow"
        header.styles.padding = 2
        yield header
        
        line2 = Static("If you can see this, the window is rendering correctly!")
        line2.styles.color = "white"
        line2.styles.background = "blue"
        line2.styles.padding = 1
        yield line2
        
        line3 = Static("This is line 3 of the debug content")
        line3.styles.color = "green"
        line3.styles.background = "black"
        line3.styles.padding = 1
        yield line3