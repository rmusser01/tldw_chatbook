from textual.app import ComposeResult
from textual.containers import VerticalScroll # Correct base class
from textual.widgets import Static, Collapsible, Label # Added Label for placeholders

class EvalsSidebar(VerticalScroll):
    """The sidebar for the Evals tab."""

    DEFAULT_CSS = """
    EvalsSidebar {
        width: 25%; /* Example width, adjust as needed */
        min-width: 20;
        max-width: 40;
        height: 100%;
        background: $boost; /* Using a common sidebar background color */
        padding: 1;
        border-right: thick $background-darken-1; /* Standard sidebar border */
        overflow-y: auto;
        overflow-x: hidden;
    }
    EvalsSidebar .sidebar-title { /* Styling for the title */
        text-style: bold underline;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    EvalsSidebar Collapsible { /* Styling for Collapsible headers */
        margin-bottom: 1;
    }
    EvalsSidebar .collapsible-content-placeholder {
        padding: 1;
        background: $panel-lighten-1; /* Slight background for placeholder content */
        border: round $surface;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the EvalsSidebar."""
        yield Static("LLM Evals", classes="sidebar-title")

        with Collapsible(title="Evaluation Setup"):
            yield Label("Configure evaluation runs, datasets, models, etc.", classes="collapsible-content-placeholder")
            # Placeholder for more specific controls later

        with Collapsible(title="Results Dashboard"):
            yield Label("View evaluation metrics, comparisons, and reports.", classes="collapsible-content-placeholder")
            # Placeholder for results display later

        with Collapsible(title="Model Management"):
            yield Label("Manage models under evaluation.", classes="collapsible-content-placeholder")
            # Placeholder

        with Collapsible(title="Dataset Management"):
            yield Label("Manage datasets used for evaluations.", classes="collapsible-content-placeholder")
            # Placeholder
