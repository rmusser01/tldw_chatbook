"""Skills destination shell for Agent Skills packs."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class SkillsScreen(BaseAppScreen):
    """Agent Skills packs, discovery, validation, and attachments."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "skills", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="skills-shell"):
            yield Static("Skills", id="skills-title", classes="ds-destination-header")
            yield Static(
                "Agent Skills packs, discovery, validation, and attachments.",
                id="skills-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="skills-sections", classes="ds-panel"):
                yield Static("Installed skills | Import | Validate | Attach to Console")
