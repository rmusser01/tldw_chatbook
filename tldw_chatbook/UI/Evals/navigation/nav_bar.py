"""Navigation bar for evaluation screens."""

from typing import TYPE_CHECKING, Optional
from enum import Enum

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static
from textual.message import Message
from textual.reactive import reactive

from loguru import logger

from .breadcrumbs import BreadcrumbTrail, BreadcrumbClicked

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvalStatus(Enum):
    """Evaluation status states."""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    SUCCESS = "success"


class QuickAction(Message):
    """Message for quick action buttons."""
    
    def __init__(self, action: str):
        super().__init__()
        self.action = action


class EvalNavigationBar(Container):
    """
    Navigation bar for evaluation screens.
    
    Includes:
    - Breadcrumb trail
    - Quick action buttons
    - Status indicator
    """
    
    DEFAULT_CSS = """
    EvalNavigationBar {
        height: 6;
        width: 100%;
        dock: top;
        layout: vertical;
        background: $panel;
        border-bottom: double $primary;
    }
    
    .nav-top-row {
        height: 3;
        width: 100%;
        layout: horizontal;
        padding: 0 2;
        align: left middle;
    }
    
    .nav-actions-row {
        height: 3;
        width: 100%;
        layout: horizontal;
        padding: 0 2;
        background: $surface;
        border-top: solid $primary-background;
    }
    
    .quick-actions {
        width: auto;
        layout: horizontal;
        align: left middle;
    }
    
    .quick-action-btn {
        margin: 0 1;
        min-width: 12;
        height: 1;
    }
    
    .quick-action-btn.run {
        background: $success;
    }
    
    .quick-action-btn.stop {
        background: $error;
    }
    
    .quick-action-btn:disabled {
        opacity: 0.5;
    }
    
    .nav-status {
        width: 1fr;
        content-align: right middle;
        padding-right: 2;
    }
    
    .status-indicator {
        width: auto;
        padding: 0 2;
        border: round $primary;
    }
    
    .status-indicator.idle {
        color: $text-muted;
        border-color: $primary-background;
    }
    
    .status-indicator.running {
        color: $warning;
        border-color: $warning;
        text-style: bold blink;
    }
    
    .status-indicator.error {
        color: $error;
        border-color: $error;
        text-style: bold;
    }
    
    .status-indicator.success {
        color: $success;
        border-color: $success;
        text-style: bold;
    }
    """
    
    # Reactive properties
    status = reactive(EvalStatus.IDLE)
    can_run = reactive(True)
    can_stop = reactive(False)
    can_export = reactive(False)
    
    def __init__(self, app_instance: Optional['TldwCli'] = None, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.breadcrumbs: Optional[BreadcrumbTrail] = None
    
    def compose(self) -> ComposeResult:
        """Compose the navigation bar."""
        # Top row with breadcrumbs
        with Container(classes="nav-top-row"):
            self.breadcrumbs = BreadcrumbTrail()
            yield self.breadcrumbs
        
        # Actions row with quick buttons and status
        with Container(classes="nav-actions-row"):
            with Horizontal(classes="quick-actions"):
                yield Button(
                    "▶️ Run",
                    id="quick-run",
                    classes="quick-action-btn run",
                    variant="success",
                    disabled=not self.can_run
                )
                yield Button(
                    "⏹️ Stop",
                    id="quick-stop",
                    classes="quick-action-btn stop",
                    variant="error",
                    disabled=not self.can_stop
                )
                yield Button(
                    "💾 Export",
                    id="quick-export",
                    classes="quick-action-btn",
                    variant="default",
                    disabled=not self.can_export
                )
                yield Button(
                    "🔄 Refresh",
                    id="quick-refresh",
                    classes="quick-action-btn",
                    variant="default"
                )
            
            # Status indicator
            with Container(classes="nav-status"):
                yield Static(
                    self._get_status_text(),
                    id="status-indicator",
                    classes=f"status-indicator {self.status.value}"
                )
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        logger.debug("Navigation bar mounted")
    
    def _get_status_text(self) -> str:
        """Get status display text."""
        status_map = {
            EvalStatus.IDLE: "⭘ Ready",
            EvalStatus.RUNNING: "⚡ Running",
            EvalStatus.ERROR: "✗ Error",
            EvalStatus.SUCCESS: "✓ Complete"
        }
        return status_map.get(self.status, "Unknown")
    
    def watch_status(self, old: EvalStatus, new: EvalStatus) -> None:
        """React to status changes."""
        # Update indicator
        try:
            indicator = self.query_one("#status-indicator", Static)
            indicator.update(self._get_status_text())
            
            # Update classes
            for status in EvalStatus:
                indicator.remove_class(status.value)
            indicator.add_class(new.value)
            
            # Update button states based on status
            if new == EvalStatus.RUNNING:
                self.can_run = False
                self.can_stop = True
                self.can_export = False
            elif new in [EvalStatus.SUCCESS, EvalStatus.ERROR]:
                self.can_run = True
                self.can_stop = False
                self.can_export = True
            else:  # IDLE
                self.can_run = True
                self.can_stop = False
                self.can_export = False
            
        except Exception as e:
            logger.warning(f"Failed to update status indicator: {e}")
    
    def watch_can_run(self, old: bool, new: bool) -> None:
        """Update run button state."""
        self._update_button_state("quick-run", not new)
    
    def watch_can_stop(self, old: bool, new: bool) -> None:
        """Update stop button state."""
        self._update_button_state("quick-stop", not new)
    
    def watch_can_export(self, old: bool, new: bool) -> None:
        """Update export button state."""
        self._update_button_state("quick-export", not new)
    
    def _update_button_state(self, button_id: str, disabled: bool) -> None:
        """Update button disabled state."""
        try:
            button = self.query_one(f"#{button_id}", Button)
            button.disabled = disabled
        except Exception as e:
            logger.warning(f"Failed to update button {button_id}: {e}")
    
    @on(Button.Pressed, "#quick-run")
    def handle_run(self) -> None:
        """Handle run button."""
        if self.can_run:
            self.post_message(QuickAction("run"))
            logger.info("Quick run action triggered")
    
    @on(Button.Pressed, "#quick-stop")
    def handle_stop(self) -> None:
        """Handle stop button."""
        if self.can_stop:
            self.post_message(QuickAction("stop"))
            logger.info("Quick stop action triggered")
    
    @on(Button.Pressed, "#quick-export")
    def handle_export(self) -> None:
        """Handle export button."""
        if self.can_export:
            self.post_message(QuickAction("export"))
            logger.info("Quick export action triggered")
    
    @on(Button.Pressed, "#quick-refresh")
    def handle_refresh(self) -> None:
        """Handle refresh button."""
        self.post_message(QuickAction("refresh"))
        logger.info("Quick refresh action triggered")
    
    @on(BreadcrumbClicked)
    def handle_breadcrumb_navigation(self, message: BreadcrumbClicked) -> None:
        """Handle breadcrumb navigation."""
        logger.info(f"Breadcrumb navigation to: {message.screen_id}")
        # Forward the navigation request
        from .eval_nav_screen import NavigateToEvalScreen
        self.post_message(NavigateToEvalScreen(message.screen_id))
    
    def push_breadcrumb(self, label: str, screen_id: str) -> None:
        """Add a breadcrumb to the trail."""
        if self.breadcrumbs:
            self.breadcrumbs.push(label, screen_id)
    
    def pop_breadcrumb(self) -> None:
        """Remove the last breadcrumb."""
        if self.breadcrumbs:
            self.breadcrumbs.pop()
    
    def set_status(self, status: EvalStatus) -> None:
        """Set the current status."""
        self.status = status