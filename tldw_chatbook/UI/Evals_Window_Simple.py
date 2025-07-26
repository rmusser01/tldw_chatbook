# Evals_Window_Simple.py
# Description: Simple test version of the Evals window to debug display issues
#
from typing import TYPE_CHECKING, Dict, Any
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static, Label
from textual import on
from loguru import logger
from ..Widgets.base_components import NavigationButton, SectionContainer, ActionButtonRow, ButtonConfig
from ..Models.evaluation_state import EvaluationState, ViewType

if TYPE_CHECKING:
    from ..app import TldwCli

logger = logger.bind(module="EvalsWindow_Simple")


class EvalsWindow(Container):
    """Simple evaluation window for testing."""
    
    # Add state
    active_view = "setup"
    
    DEFAULT_CSS = """
    EvalsWindow {
        height: 100%;
        width: 100%;
    }
    
    .evals-window {
        layout: grid;
        grid-size: 2;
        grid-columns: 250 1fr;
        height: 100%;
        width: 100%;
    }
    
    .sidebar {
        background: $surface;
        border-right: solid $primary;
        padding: 1;
        height: 100%;
    }
    
    .content-area {
        layout: vertical;
        height: 100%;
        padding: 1;
    }
    
    .content-header {
        height: 3;
        padding: 0 2;
        background: $surface;
        border-bottom: solid $primary-background;
    }
    
    .view-container {
        height: 1fr;
        padding: 1;
    }
    
    .sidebar-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $primary;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the evaluation window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.info("EvalsWindow Simple initialized")
    
    def compose(self) -> ComposeResult:
        """Build the window structure."""
        logger.info("Composing EvalsWindow Simple")
        
        # Try the same structure as v2
        with Container(classes="evals-window", id="evals-main-container"):
            # Create sidebar
            yield self._create_sidebar()
            
            # Content area
            with Container(classes="content-area", id="evals-content-area"):
                yield self._create_header()
                yield self._create_content()
    
    def _create_sidebar(self) -> Vertical:
        """Create the navigation sidebar."""
        return Vertical(
            Static("Evaluation Tools", classes="sidebar-title"),
            NavigationButton("Setup", icon="⚙️", id="nav-setup", active=True),
            NavigationButton("Results", icon="📊", id="nav-results"),
            NavigationButton("Models", icon="🤖", id="nav-models"),
            NavigationButton("Datasets", icon="📚", id="nav-datasets"),
            classes="sidebar",
            id="sidebar"
        )
    
    def _create_header(self) -> Horizontal:
        """Create the content header."""
        return Horizontal(
            Button("☰", id="sidebar-toggle", classes="sidebar-toggle"),
            Static("Evaluation Setup", id="view-title", classes="view-title"),
            classes="content-header"
        )
    
    def _create_content(self) -> Container:
        """Create simple content."""
        return Container(
            SectionContainer(
                "Test Section",
                Label("This is working!"),
                Button("Test Button"),
                id="test-section"
            ),
            classes="view-container",
            id="view-container"
        )
    
    def on_mount(self) -> None:
        """Called when mounted."""
        logger.info("EvalsWindow Simple mounted successfully")
    
    @on(Button.Pressed, ".navigation-button")
    def handle_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation button clicks."""
        button_id = event.button.id
        if not button_id:
            return
        
        logger.info(f"Navigation button pressed: {button_id}")
        
        # Update active states
        for button in self.query(".navigation-button"):
            if button.id == button_id:
                button.activate()
            else:
                button.deactivate()
        
        # Update view title
        view_names = {
            "nav-setup": "Evaluation Setup",
            "nav-results": "Results Dashboard",
            "nav-models": "Model Management",
            "nav-datasets": "Dataset Management"
        }
        
        title = self.query_one("#view-title", Static)
        title.update(view_names.get(button_id, "Unknown"))
        
        # Update active view
        self.active_view = button_id.replace("nav-", "")