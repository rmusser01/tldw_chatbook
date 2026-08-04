"""Main evaluation navigation hub screen."""

from typing import TYPE_CHECKING
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Grid
from textual.widgets import Button, Static
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
    planned: bool = False
    demo: bool = False


#: Hub card definitions, shared with the EvalsScreen status chip so the
#: "available" count can never rot out of sync with the cards (UX-058).
EVAL_NAV_CARDS: tuple[NavigationCard, ...] = (
    NavigationCard(
        id="quick_test",
        title="Quick Test",
        icon="⚡",
        description="Run a single evaluation\nwith one model and task",
        shortcut="Press [1]",
        color="success",
    ),
    NavigationCard(
        id="comparison",
        title="Comparison Mode",
        icon="⇄",
        description="Compare multiple models\non the same task",
        shortcut="Planned",
        color="warning",
        planned=True,
    ),
    NavigationCard(
        id="batch_eval",
        title="Batch Evaluation",
        icon="📦",
        description="Queue and run multiple\nevaluations in sequence",
        shortcut="Planned",
        color="error",
        planned=True,
    ),
    NavigationCard(
        id="results",
        title="Results Browser",
        icon="📊",
        description="Browse, search and export\nevaluation results",
        shortcut="Press [4]",
        color="primary",
    ),
    NavigationCard(
        id="tasks",
        title="Evaluations",
        icon="📋",
        description="Browse evaluation definitions,\ndatasets and recent runs",
        shortcut="Press [5]",
        color="secondary",
    ),
    NavigationCard(
        id="models",
        title="Model Manager",
        icon="🤖",
        description="Configure and test\nmodel connections",
        shortcut="Planned",
        color="accent",
        planned=True,
    ),
)


def evals_workflows_chip_label() -> str:
    """Truthful chip text derived from the hub cards (live/demo/planned)."""
    live = sum(1 for c in EVAL_NAV_CARDS if not c.planned and not c.demo)
    demo = sum(1 for c in EVAL_NAV_CARDS if c.demo and not c.planned)
    planned = sum(1 for c in EVAL_NAV_CARDS if c.planned)
    parts = [f"{live} live"]
    if demo:
        parts.append(f"{demo} demo")
    parts.append(f"{planned} planned")
    return " · ".join(parts)


class NavigateToEvalScreen(Message):
    """Message to navigate to a specific eval screen."""

    def __init__(self, screen_id: str):
        super().__init__()
        self.screen_id = screen_id


class EvalNavigationScreen(Container):
    """
    Main navigation hub for evaluation workflows.

    Provides card-based navigation to different evaluation modes
    with keyboard shortcuts and clear visual hierarchy.

    Rendered inline inside EvalsWindowV3 (a Container), so it must be a
    widget, not a Screen -- nested Screens receive no layout geometry.
    Number-key and Escape navigation are handled by the parent EvalsScreen
    (see its BINDINGS); per ADR-031 this hub binds no ctrl-chords.
    """

    DEFAULT_CSS = """
    /* Local fallbacks so DEFAULT_CSS parses without the app bundle. */
    $ds-focus-accent: $primary;
    $ds-focus-bg: $surface;
    $ds-focus-fg: $text;

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
        grid-columns: 30 30 30;
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

    .nav-card.quick_test {
        border: round $success;
    }

    .nav-card.comparison {
        border: round $warning;
    }

    .nav-card.batch_eval {
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

    .nav-card.planned {
        border: round $surface-darken-2;
        color: $text-muted;
    }

    .nav-card:focus {
        background: $ds-focus-bg;
        border: round $ds-focus-accent;
        color: $ds-focus-fg;
        text-style: bold underline;
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

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.last_evaluation = None

        # Hub cards come from the module constant so the EvalsScreen chip
        # count stays in sync with what the hub actually offers.
        self.cards = list(EVAL_NAV_CARDS)

    def compose(self) -> ComposeResult:
        """Compose the navigation screen.

        No internal title block: the shell's DestinationHeader already carries
        the destination identity, so the hub goes straight to the workflow
        cards (de-duplicated, emoji marketing header removed per DESIGN.md).
        """
        # Main content with cards
        with Container(classes="cards-container"):
            with Grid(classes="cards-grid"):
                for card in self.cards:
                    yield self._create_card(card)

        # Status bar with quick actions
        with Container(classes="status-bar"):
            yield Static(
                "Ready — choose a workflow or press 1, 4, or 5",
                id="status-text",
                classes="status-text",
            )
            with Container(classes="quick-actions"):
                yield Button(
                    "Settings",
                    id="settings-btn",
                    classes="quick-action",
                    variant="default",
                )
                yield Button(
                    "Help", id="help-btn", classes="quick-action", variant="default"
                )

    def _create_card(self, card: NavigationCard) -> Button:
        """Create a navigation card widget."""
        # Create button with card content
        card_content = (
            f"{card.icon}\n\n{card.title}\n\n{card.description}\n\n{card.shortcut}"
        )
        classes = f"nav-card nav-card-button {card.id}"
        if card.planned:
            classes = f"{classes} planned"
        button = Button(
            card_content,
            id=f"card-{card.id}",
            classes=classes,
            disabled=card.planned,
        )
        if card.planned:
            button.tooltip = f"{card.title} is planned — not available yet."
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
            "1, 4, 5: Quick navigation (Quick Test, Results, Evaluations)",
            "Tab/Shift+Tab: Focus navigation",
            "Enter: Activate focused card",
            "Escape: Go back",
        ]

        if self.app_instance:
            self.app_instance.notify(
                "\n".join(shortcuts), title="Shortcuts", timeout=10
            )

        self._update_status("Shortcuts displayed")

    def _navigate_to(self, screen_id: str) -> None:
        """Navigate to a specific evaluation screen."""
        logger.info(f"Navigating to: {screen_id}")
        self._update_status(f"Opening {screen_id.replace('_', ' ').title()}...")

        # Post navigation message; the visible transition is the feedback,
        # so no redundant toast on top of it.
        self.post_message(NavigateToEvalScreen(screen_id))

    def _update_status(self, message: str) -> None:
        """Update the status text."""
        try:
            status = self.query_one("#status-text", Static)
            status.update(message)
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")

    @on(Button.Pressed, "#settings-btn")
    def handle_settings(self) -> None:
        """Open the app's Settings destination."""
        from ...Navigation.main_navigation import NavigateToScreen

        self.post_message(NavigateToScreen("settings"))

    @on(Button.Pressed, "#help-btn")
    def handle_help(self) -> None:
        """Handle help button."""
        self.action_show_shortcuts()
