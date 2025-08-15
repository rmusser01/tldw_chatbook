"""Main evaluation navigation hub screen."""

from typing import TYPE_CHECKING
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Grid, Vertical
from textual.widgets import Button, Static, Header, Footer
from textual.binding import Binding
from textual.message import Message

from loguru import logger

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


@dataclass
class NavigationCard:
    """Data for a navigation card."""
    id: str
    title: str
    icon: str
    description: str
    shortcut: str
    color: str = "primary"


class NavigateToEvalScreen(Message):
    """Message to navigate to a specific eval screen."""
    
    def __init__(self, screen_id: str):
        super().__init__()
        self.screen_id = screen_id


class EvalNavigationScreen(Screen):
    """
    Main navigation hub for evaluation workflows.
    
    Provides card-based navigation to different evaluation modes
    with keyboard shortcuts and clear visual hierarchy.
    """
    
    BINDINGS = [
        Binding("1", "quick_test", "Quick Test", show=True),
        Binding("2", "comparison", "Comparison", show=True),
        Binding("3", "batch_eval", "Batch Eval", show=True),
        Binding("4", "results", "Results", show=True),
        Binding("5", "tasks", "Tasks", show=True),
        Binding("6", "models", "Models", show=True),
        Binding("escape", "app.pop_screen", "Back", show=True),
        Binding("ctrl+/", "show_shortcuts", "Shortcuts", show=True),
        Binding("ctrl+r", "run_last", "Run Last", show=False),
    ]
    
    DEFAULT_CSS = """
    EvalNavigationScreen {
        background: $background;
    }
    
    .nav-header {
        height: 5;
        background: $panel;
        border-bottom: solid $primary;
        padding: 1 2;
    }
    
    .nav-title {
        text-style: bold;
        color: $primary;
        text-align: center;
    }
    
    .nav-subtitle {
        color: $text-muted;
        text-align: center;
        text-style: italic;
    }
    
    .cards-container {
        padding: 2;
        align: center middle;
    }
    
    .cards-grid {
        grid-size: 3 2;
        grid-gutter: 2;
        width: auto;
        height: auto;
        margin: 0 1;
    }
    
    .nav-card {
        width: 30;
        height: 12;
        border: round $primary;
        background: $panel;
        padding: 1;
        text-align: center;
        content-align: center middle;
    }
    
    .nav-card:hover {
        background: $boost;
        border: round $accent;
    }
    
    .nav-card:focus {
        border: thick $warning;
    }
    
    .nav-card.quick-test {
        border: round $success;
    }
    
    .nav-card.comparison {
        border: round $warning;
    }
    
    .nav-card.batch {
        border: round $error;
    }
    
    .nav-card.results {
        border: round $primary;
    }
    
    .nav-card.tasks {
        border: round $secondary;
    }
    
    .nav-card.models {
        border: round $accent;
    }
    
    .card-icon {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .card-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .card-description {
        text-align: center;
        color: $text-muted;
    }
    
    .card-shortcut {
        text-align: center;
        color: $text-disabled;
        margin-top: 1;
    }
    
    .status-bar {
        height: 3;
        dock: bottom;
        background: $panel;
        border-top: solid $primary;
        padding: 0 2;
        layout: horizontal;
    }
    
    .status-text {
        width: 1fr;
        content-align: left middle;
    }
    
    .quick-actions {
        width: auto;
        layout: horizontal;
        align: right middle;
    }
    
    .quick-action {
        margin: 0 1;
        min-width: 10;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.last_evaluation = None
        
        # Define navigation cards
        self.cards = [
            NavigationCard(
                id="quick_test",
                title="Quick Test",
                icon="⚡",
                description="Run a single evaluation\nwith one model and task",
                shortcut="Press [1]",
                color="success"
            ),
            NavigationCard(
                id="comparison",
                title="Comparison Mode",
                icon="⚖️",
                description="Compare multiple models\non the same task",
                shortcut="Press [2]",
                color="warning"
            ),
            NavigationCard(
                id="batch_eval",
                title="Batch Evaluation",
                icon="📦",
                description="Queue and run multiple\nevaluations in sequence",
                shortcut="Press [3]",
                color="error"
            ),
            NavigationCard(
                id="results",
                title="Results Browser",
                icon="📊",
                description="Browse, search and export\nevaluation results",
                shortcut="Press [4]",
                color="primary"
            ),
            NavigationCard(
                id="tasks",
                title="Task Manager",
                icon="📋",
                description="Create, edit and manage\nevaluation tasks",
                shortcut="Press [5]",
                color="secondary"
            ),
            NavigationCard(
                id="models",
                title="Model Manager",
                icon="🤖",
                description="Configure and test\nmodel connections",
                shortcut="Press [6]",
                color="accent"
            ),
        ]
    
    def compose(self) -> ComposeResult:
        """Compose the navigation screen."""
        # Header with title
        with Container(classes="nav-header"):
            yield Static("🧪 Evaluation Lab", classes="nav-title")
            yield Static("Choose an evaluation workflow", classes="nav-subtitle")
        
        # Main content with cards
        with Container(classes="cards-container"):
            with Grid(classes="cards-grid"):
                for card in self.cards:
                    yield self._create_card(card)
        
        # Status bar with quick actions
        with Container(classes="status-bar"):
            yield Static("Ready", id="status-text", classes="status-text")
            with Container(classes="quick-actions"):
                yield Button("⚙️ Settings", id="settings-btn", classes="quick-action", variant="default")
                yield Button("❓ Help", id="help-btn", classes="quick-action", variant="default")
    
    def _create_card(self, card: NavigationCard) -> Button:
        """Create a navigation card widget."""
        # Create button with card content
        card_content = f"{card.icon}\n\n{card.title}\n\n{card.description}\n\n{card.shortcut}"
        button = Button(
            card_content,
            id=f"card-{card.id}",
            classes=f"nav-card nav-card-button {card.id}"
        )
        return button
    
    def on_mount(self) -> None:
        """Initialize when screen mounts."""
        logger.info("Evaluation navigation screen mounted")
        self._update_status("Ready - Choose a workflow or press a number key")
        
        # Focus first card
        cards = self.query(".nav-card")
        if cards:
            cards.first().focus()
    
    @on(Button.Pressed, ".nav-card-button")
    def handle_card_click(self, event: Button.Pressed) -> None:
        """Handle card selection via click."""
        # Find which card was clicked
        button_id = event.button.id
        if button_id and button_id.startswith("card-"):
            card_id = button_id.replace("card-", "")
            self._navigate_to(card_id)
    
    def action_quick_test(self) -> None:
        """Navigate to quick test screen."""
        self._navigate_to("quick_test")
    
    def action_comparison(self) -> None:
        """Navigate to comparison screen."""
        self._navigate_to("comparison")
    
    def action_batch_eval(self) -> None:
        """Navigate to batch evaluation screen."""
        self._navigate_to("batch_eval")
    
    def action_results(self) -> None:
        """Navigate to results browser."""
        self._navigate_to("results")
    
    def action_tasks(self) -> None:
        """Navigate to task manager."""
        self._navigate_to("tasks")
    
    def action_models(self) -> None:
        """Navigate to model manager."""
        self._navigate_to("models")
    
    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcuts help."""
        shortcuts = [
            "Keyboard Shortcuts:",
            "",
            "1-6: Quick navigation to sections",
            "Tab/Shift+Tab: Focus navigation",
            "Enter: Activate focused card",
            "Escape: Go back",
            "Ctrl+R: Run last evaluation",
            "Ctrl+/: Show this help",
        ]
        
        if self.app_instance:
            self.app_instance.notify("\n".join(shortcuts), title="Shortcuts", timeout=10)
        
        self._update_status("Shortcuts displayed")
    
    def action_run_last(self) -> None:
        """Re-run the last evaluation."""
        if self.last_evaluation:
            self._update_status("Re-running last evaluation...")
            # TODO: Implement re-run logic
            if self.app_instance:
                self.app_instance.notify("Re-running last evaluation", severity="information")
        else:
            self._update_status("No previous evaluation to run")
            if self.app_instance:
                self.app_instance.notify("No previous evaluation to run", severity="warning")
    
    def _navigate_to(self, screen_id: str) -> None:
        """Navigate to a specific evaluation screen."""
        logger.info(f"Navigating to: {screen_id}")
        self._update_status(f"Opening {screen_id.replace('_', ' ').title()}...")
        
        # Post navigation message
        self.post_message(NavigateToEvalScreen(screen_id))
        
        # For now, show notification
        if self.app_instance:
            self.app_instance.notify(
                f"Opening {screen_id.replace('_', ' ').title()} screen",
                severity="information"
            )
    
    def _update_status(self, message: str) -> None:
        """Update the status text."""
        try:
            status = self.query_one("#status-text", Static)
            status.update(message)
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")
    
    @on(Button.Pressed, "#settings-btn")
    def handle_settings(self) -> None:
        """Handle settings button."""
        self._update_status("Opening settings...")
        if self.app_instance:
            self.app_instance.notify("Settings coming soon", severity="information")
    
    @on(Button.Pressed, "#help-btn")
    def handle_help(self) -> None:
        """Handle help button."""
        self.action_show_shortcuts()