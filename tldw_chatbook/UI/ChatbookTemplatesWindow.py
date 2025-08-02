# ChatbookTemplatesWindow.py
# Description: Window for browsing and selecting chatbook templates
#
"""
Chatbook Templates Window
-------------------------

Provides pre-configured chatbook templates for common use cases.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Container, Horizontal, VerticalScroll, Grid
from textual.widgets import Static, Button, RadioSet, RadioButton
from textual.reactive import reactive
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


class ChatbookTemplate:
    """Represents a chatbook template."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        icon: str,
        content_types: List[str],
        tags: List[str],
        use_cases: List[str]
    ):
        self.id = id
        self.name = name
        self.description = description
        self.icon = icon
        self.content_types = content_types
        self.tags = tags
        self.use_cases = use_cases


# Pre-defined templates
CHATBOOK_TEMPLATES = [
    ChatbookTemplate(
        id="research_project",
        name="Research Project",
        description="Organize research conversations, notes, and references for academic or professional projects",
        icon="🔬",
        content_types=["conversations", "notes", "media", "prompts"],
        tags=["research", "academic", "project", "references"],
        use_cases=[
            "Academic research papers",
            "Market research projects",
            "Technical investigations",
            "Literature reviews"
        ]
    ),
    ChatbookTemplate(
        id="creative_writing",
        name="Creative Writing",
        description="Bundle character profiles, story notes, and world-building conversations",
        icon="✍️",
        content_types=["characters", "notes", "conversations"],
        tags=["writing", "creative", "storytelling", "characters"],
        use_cases=[
            "Novel writing",
            "Short story collections",
            "Screenplay development",
            "World-building projects"
        ]
    ),
    ChatbookTemplate(
        id="learning_journey",
        name="Learning Journey",
        description="Track your learning progress with conversations, notes, and study materials",
        icon="📚",
        content_types=["conversations", "notes", "media"],
        tags=["education", "learning", "study", "knowledge"],
        use_cases=[
            "Online course notes",
            "Self-study projects",
            "Skill development",
            "Tutorial collections"
        ]
    ),
    ChatbookTemplate(
        id="project_documentation",
        name="Project Documentation",
        description="Document project conversations, decisions, and technical notes",
        icon="📋",
        content_types=["conversations", "notes", "media"],
        tags=["documentation", "project", "technical", "decisions"],
        use_cases=[
            "Software documentation",
            "Project retrospectives",
            "Decision logs",
            "Technical specifications"
        ]
    ),
    ChatbookTemplate(
        id="personal_assistant",
        name="Personal Assistant",
        description="Export your AI assistant conversations and custom prompts",
        icon="🤖",
        content_types=["conversations", "prompts", "characters"],
        tags=["AI", "assistant", "productivity", "prompts"],
        use_cases=[
            "AI workflow templates",
            "Custom assistant personas",
            "Prompt libraries",
            "Productivity systems"
        ]
    ),
    ChatbookTemplate(
        id="knowledge_base",
        name="Knowledge Base",
        description="Create a comprehensive knowledge repository with all content types",
        icon="🧠",
        content_types=["conversations", "notes", "characters", "media", "prompts"],
        tags=["knowledge", "repository", "comprehensive", "archive"],
        use_cases=[
            "Team knowledge sharing",
            "Personal wikis",
            "Reference collections",
            "Archive backups"
        ]
    )
]


class ChatbookTemplatesWindow(ModalScreen):
    """Window for browsing chatbook templates."""
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("enter", "select", "Select Template")
    ]
    
    DEFAULT_CSS = """
    ChatbookTemplatesWindow {
        align: center middle;
    }
    
    ChatbookTemplatesWindow > Container {
        width: 80%;
        height: 80%;
        max-width: 100;
        max-height: 40;
        background: $surface;
        border: thick $primary;
    }
    
    .window-header {
        height: 5;
        padding: 1;
        background: $boost;
        border-bottom: solid $background-darken-1;
    }
    
    .window-title {
        text-style: bold;
        text-align: center;
        color: $primary;
        margin-bottom: 1;
    }
    
    .window-subtitle {
        text-align: center;
        color: $text-muted;
        margin-top: 0;
    }
    
    .templates-container {
        height: 1fr;
        padding: 2;
        overflow-y: auto;
    }
    
    .template-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 2;
        margin-bottom: 2;
    }
    
    .template-card {
        padding: 2;
        background: $boost;
        border: round $background-darken-1;
        height: auto;
    }
    
    .template-card:hover {
        background: $primary 10%;
        border-color: $primary;
    }
    
    .template-card.selected {
        background: $primary 20%;
        border: thick $primary;
    }
    
    .template-icon {
        font-size: 200%;
        text-align: center;
        margin-bottom: 1;
    }
    
    .template-name {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $primary;
    }
    
    .template-description {
        text-align: center;
        color: $text-muted;
        font-size: 90%;
        margin-bottom: 1;
    }
    
    .template-tags {
        text-align: center;
        font-size: 85%;
        color: $text-disabled;
    }
    
    .details-panel {
        height: 12;
        padding: 1 2;
        background: $panel;
        border-top: solid $background-darken-1;
    }
    
    .details-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .details-section {
        margin-bottom: 1;
    }
    
    .details-label {
        color: $text-muted;
        margin-right: 1;
    }
    
    .use-case-list {
        margin-left: 2;
        color: $text-muted;
        font-size: 90%;
    }
    
    .action-buttons {
        dock: bottom;
        height: 4;
        padding: 1 2;
        background: $panel;
        border-top: solid $background-darken-1;
        align: center middle;
    }
    
    .action-buttons Button {
        margin: 0 1;
        min-width: 20;
    }
    """
    
    selected_template = reactive(None)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.templates = CHATBOOK_TEMPLATES
        
    def compose(self) -> ComposeResult:
        """Compose the templates UI."""
        with Container():
            # Header
            with Container(classes="window-header"):
                yield Static("📚 Chatbook Templates", classes="window-title")
                yield Static("Choose a template to get started quickly", classes="window-subtitle")
            
            # Templates grid
            with VerticalScroll(classes="templates-container"):
                grid = Grid(classes="template-grid", id="template-grid")
                yield grid
            
            # Details panel
            with Container(classes="details-panel"):
                yield Static("Template Details", id="details-title", classes="details-title")
                yield Container(id="details-content")
            
            # Action buttons
            with Container(classes="action-buttons"):
                yield Button("Use Template", id="use-template", variant="primary", disabled=True)
                yield Button("Cancel", id="cancel", variant="default")
    
    async def on_mount(self) -> None:
        """Initialize when mounted."""
        # Create template cards
        grid = self.query_one("#template-grid", Grid)
        
        for template in self.templates:
            card = Container(classes="template-card", id=f"template-{template.id}")
            
            card.mount(Static(template.icon, classes="template-icon"))
            card.mount(Static(template.name, classes="template-name"))
            card.mount(Static(template.description, classes="template-description"))
            card.mount(Static(f"Tags: {', '.join(template.tags[:3])}", classes="template-tags"))
            
            grid.mount(card)
        
        # Initialize details
        self._update_details(None)
    
    async def on_click(self, event) -> None:
        """Handle clicks on template cards."""
        # Find clicked template card
        for node in event.target.ancestors:
            if hasattr(node, 'id') and node.id and node.id.startswith("template-"):
                template_id = node.id.replace("template-", "")
                await self._select_template(template_id)
                break
    
    async def _select_template(self, template_id: str) -> None:
        """Select a template."""
        # Find template
        template = next((t for t in self.templates if t.id == template_id), None)
        if not template:
            return
        
        self.selected_template = template
        
        # Update visual selection
        grid = self.query_one("#template-grid", Grid)
        for card in grid.children:
            if isinstance(card, Container) and card.has_class("template-card"):
                if card.id == f"template-{template_id}":
                    card.add_class("selected")
                else:
                    card.remove_class("selected")
        
        # Update details
        self._update_details(template)
        
        # Enable use button
        self.query_one("#use-template", Button).disabled = False
    
    def _update_details(self, template: Optional[ChatbookTemplate]) -> None:
        """Update the details panel."""
        content = self.query_one("#details-content", Container)
        content.remove_children()
        
        if not template:
            content.mount(Static("Select a template to view details", classes="details-section"))
            return
        
        # Content types
        types_section = Container(classes="details-section")
        types_section.mount(Static("Includes: ", classes="details-label"))
        types_text = ", ".join(t.title() for t in template.content_types)
        types_section.mount(Static(types_text))
        content.mount(types_section)
        
        # Use cases
        if template.use_cases:
            use_cases_section = Container(classes="details-section")
            use_cases_section.mount(Static("Common use cases:", classes="details-label"))
            
            use_list = Container(classes="use-case-list")
            for use_case in template.use_cases:
                use_list.mount(Static(f"• {use_case}"))
            use_cases_section.mount(use_list)
            content.mount(use_cases_section)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "use-template" and self.selected_template:
            # Return the selected template
            self.dismiss(self.selected_template)
        elif event.button.id == "cancel":
            self.dismiss(None)
    
    def action_close(self) -> None:
        """Close action."""
        self.dismiss(None)
    
    def action_select(self) -> None:
        """Select action."""
        if self.selected_template:
            self.dismiss(self.selected_template)