# tldw_chatbook/UI/Coding_Window.py
#
# Imports
#
# 3rd-Party Libraries
from typing import TYPE_CHECKING, Optional
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Static, Button, Label
#
# Local Imports
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
from .CodeRepoCopyPasteWindow import CodeRepoCopyPasteWindow
if TYPE_CHECKING:
    from ..app import TldwCli
#
# Configure logger with context
logger = logger.bind(module="CodingWindow")
#
# #######################################################################################################################
#
# Functions:

# Constants for clarity
CODING_VIEW_CODE_MAP = "coding-view-code-map"
CODING_VIEW_AGENTIC_CODER = "coding-view-agentic-coder"
CODING_VIEW_STEP_BY_STEP = "coding-view-step-by-step"
CODING_VIEW_REPO_COPY_PASTE = "coding-view-repo-copy-paste"

CODING_NAV_CODE_MAP = "coding-nav-code-map"
CODING_NAV_AGENTIC_CODER = "coding-nav-agentic-coder"
CODING_NAV_STEP_BY_STEP = "coding-nav-step-by-step"
CODING_NAV_REPO_COPY_PASTE = "coding-nav-repo-copy-paste"

class CodingWindow(Container):
    """
    A fully self-contained component for the Coding Tab, featuring a collapsible
    sidebar and content areas for code-related functionality.
    """
    # --- STATE LIVES HERE NOW ---
    coding_sidebar_collapsed: reactive[bool] = reactive(False)
    coding_active_view: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    # --- WATCHERS LIVE HERE NOW ---
    def watch_coding_sidebar_collapsed(self, collapsed: bool) -> None:
        """Dynamically adjusts the coding browser panes when the sidebar is collapsed or expanded."""
        try:
            nav_pane = self.query_one("#coding-nav-pane")
            toggle_button = self.query_one("#coding-sidebar-toggle-button")
            nav_pane.set_class(collapsed, "collapsed")
            toggle_button.set_class(collapsed, "collapsed")
        except QueryError as e:
            logger.warning(f"UI component not found during coding sidebar collapse: {e}")

    def watch_coding_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Shows/hides the relevant content view when the active view slug changes."""
        if old_view:
            try:
                self.query_one(f"#{old_view}").styles.display = "none"
            except QueryError: pass
        if new_view:
            try:
                view_to_show = self.query_one(f"#{new_view}")
                view_to_show.styles.display = "block"
            except QueryError:
                logger.error(f"Could not find new coding view to display: #{new_view}")

    # --- EVENT HANDLERS LIVE HERE NOW ---
    @on(Button.Pressed, "#coding-sidebar-toggle-button")
    def handle_sidebar_toggle(self) -> None:
        """Toggles the sidebar's collapsed state."""
        self.coding_sidebar_collapsed = not self.coding_sidebar_collapsed

    @on(Button.Pressed, ".coding-nav-button")
    def handle_nav_button_press(self, event: Button.Pressed) -> None:
        """Handles a click on a coding navigation button."""
        if event.button.id:
            type_slug = event.button.id.replace("coding-nav-", "")
            self.coding_active_view = f"coding-view-{type_slug}"

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        logger.info(f"CodingWindow on_mount: UI composed")
        
        # Hide all views initially
        for view_area in self.query(".coding-view-area"):
            view_area.styles.display = "none"
        
        # Set initial active view
        if not self.coding_active_view:
            self.coding_active_view = CODING_VIEW_CODE_MAP

    def compose(self) -> ComposeResult:
        # Left Navigation Pane
        with VerticalScroll(classes="coding-nav-pane", id="coding-nav-pane"):
            yield Static("Coding Tools", classes="sidebar-title")
            yield Button("Code Map", id=CODING_NAV_CODE_MAP, classes="coding-nav-button")
            yield Button("Agentic Coder", id=CODING_NAV_AGENTIC_CODER, classes="coding-nav-button")
            yield Button("Step-by-Step", id=CODING_NAV_STEP_BY_STEP, classes="coding-nav-button")
            yield Button("Code Repo Copy/Paste", id=CODING_NAV_REPO_COPY_PASTE, classes="coding-nav-button")

        # Main Content Pane
        with Container(classes="coding-content-pane", id="coding-content-pane"):
            yield Button(
                get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                id="coding-sidebar-toggle-button",
                classes="sidebar-toggle",
                tooltip="Toggle sidebar"
            )

            # Create a view for Code Map
            with Container(id=CODING_VIEW_CODE_MAP, classes="coding-view-area"):
                yield Static("Code Map Content Area", classes="pane-title")
                yield Label("This is the Code Map view where you can visualize code structure.")

            # Create a view for Agentic Coder
            with Container(id=CODING_VIEW_AGENTIC_CODER, classes="coding-view-area"):
                yield Static("Agentic Coder Content Area", classes="pane-title")
                yield Label("This is the Agentic Coder view where you can use AI to help with coding tasks.")

            # Create a view for Step-by-Step
            with Container(id=CODING_VIEW_STEP_BY_STEP, classes="coding-view-area"):
                yield Static("Step-by-Step Content Area", classes="pane-title")
                yield Label("This is the Step-by-Step view where you can follow guided coding tutorials.")

            # Create a view for Code Repo Copy/Paste
            with Container(id=CODING_VIEW_REPO_COPY_PASTE, classes="coding-view-area"):
                # Embed the repository selector directly instead of using a modal
                repo_window = CodeRepoCopyPasteWindow(self.app_instance)
                yield from repo_window.compose()


#
# End of Coding_Window.py
# #######################################################################################################################
