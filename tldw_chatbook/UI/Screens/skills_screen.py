"""Skills destination shell for Agent Skills packs."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ..Navigation.base_app_screen import BaseAppScreen


class SkillsScreen(BaseAppScreen):
    """Agent Skills packs, discovery, validation, and attachments."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "skills", **kwargs)

    def compose_content(self) -> ComposeResult:
        local_skills_service = getattr(self.app_instance, "local_skills_service", None)
        skills_dir = getattr(local_skills_service, "skills_dir", None)
        skills_dir_label = str(skills_dir) if skills_dir is not None else "Local skills directory unavailable."

        with Vertical(id="skills-shell"):
            yield Static("Skills", id="skills-title", classes="ds-destination-header")
            yield Static(
                "Skills owns Agent Skills packs. Each skill is a directory with a required SKILL.md file.",
                id="skills-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="skills-sections", classes="ds-panel"):
                yield Static("Installed", classes="destination-section")
                yield Static("Discover/Import", classes="destination-section")
                yield Static("Validate", classes="destination-section")
                yield Static("Scripts", classes="destination-section")
                yield Static("References", classes="destination-section")
                yield Static("Assets", classes="destination-section")
                yield Static("Attachments", classes="destination-section")
                yield Static(f"Local skills directory: {skills_dir_label}", id="skills-local-directory")
                yield Static(
                    "Skill import is not wired in this shell yet.",
                    id="skills-import-unavailable",
                )
                yield Button(
                    "Import Skill",
                    id="skills-import-skill",
                    disabled=True,
                    tooltip="Unavailable until skill import is wired in this shell.",
                )
                yield Button(
                    "Attach to Console",
                    id="skills-attach-to-console",
                    tooltip="Stage Agent Skills context in Console.",
                )

    @on(Button.Pressed, "#skills-attach-to-console")
    def attach_to_console(self) -> None:
        self.app_instance.open_chat_with_handoff(
            ChatHandoffPayload(
                source="skills",
                item_type="skill-context",
                title="Skills context",
                body="Stage installed skills, SKILL.md instructions, scripts, references, assets, or attachments for Console.",
            )
        )
