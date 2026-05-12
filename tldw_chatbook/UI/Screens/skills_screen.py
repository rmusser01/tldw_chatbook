"""Skills destination shell for Agent Skills packs."""

from __future__ import annotations

import inspect
import asyncio
from collections.abc import Mapping
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Rule, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...runtime_policy.types import PolicyDeniedError
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.base_app_screen import BaseAppScreen
from .destination_recovery import DestinationRecoveryState, policy_denied_recovery_state


logger = logger.bind(module="SkillsScreen")
SKILLS_LOCAL_PAGE_SIZE = 25
SKILLS_SERVICE_ERROR_COPY = "Skills service unavailable; retry Skills later."
SKILLS_SERVICE_UNAVAILABLE_COPY = "Skills service is unavailable in this runtime."
SKILLS_POLICY_DENIED_FALLBACK_COPY = "Local Skills are blocked by the current runtime policy."
SKILL_TEXT_LIMITS = {
    "name": 64,
    "skill_name": 64,
    "description": 1024,
    "argument_hint": 500,
    "record_id": 256,
    "backend": 32,
    "policy_message": 500,
}


class SkillsScreen(BaseAppScreen):
    """Agent Skills packs, discovery, validation, and attachments."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "skills", **kwargs)
        self._local_skill_records: tuple[Mapping[str, Any], ...] = ()
        self._skills_lookup_error: str | None = None
        self._skills_lookup_recovery_state: DestinationRecoveryState | None = None
        self._skills_loaded = False

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_local_skills_context()

    @work(exclusive=True, thread=True)
    def _refresh_local_skills_context(self) -> None:
        records, lookup_error, recovery_state = asyncio.run(self._list_local_skills())
        self.app.call_from_thread(
            self._apply_local_skills_context,
            records,
            lookup_error,
            recovery_state,
        )

    def _apply_local_skills_context(
        self,
        records: tuple[Mapping[str, Any], ...],
        lookup_error: str | None = None,
        recovery_state: DestinationRecoveryState | None = None,
    ) -> None:
        self._local_skill_records = records
        self._skills_lookup_error = lookup_error
        self._skills_lookup_recovery_state = recovery_state
        self._skills_loaded = True
        if self.is_mounted:
            self.refresh(recompose=True)

    async def _list_local_skills(
        self,
    ) -> tuple[tuple[Mapping[str, Any], ...], str | None, DestinationRecoveryState | None]:
        service = getattr(self.app_instance, "skills_scope_service", None)
        list_skills = getattr(service, "list_skills", None)
        if not callable(list_skills):
            return (), SKILLS_SERVICE_UNAVAILABLE_COPY, None
        try:
            result = list_skills(
                mode="local",
                limit=SKILLS_LOCAL_PAGE_SIZE,
                offset=0,
                include_hidden=False,
            )
            if inspect.isawaitable(result):
                result = await result
        except PolicyDeniedError as exc:
            logger.info(
                "Runtime policy denied local Agent Skills listing.",
                action_id=exc.action_id,
                reason_code=exc.reason_code,
            )
            policy_message = self._safe_skill_text(
                exc.user_message,
                fallback=SKILLS_POLICY_DENIED_FALLBACK_COPY,
                max_length=SKILL_TEXT_LIMITS["policy_message"],
            )
            recovery_state = policy_denied_recovery_state(
                exc,
                unavailable_what="Attach local Skills to Console",
                stable_selector="skills-service-error",
                policy_message=policy_message,
            )
            return (), recovery_state.visible_copy, recovery_state
        except Exception:
            logger.warning(
                "Failed to load local Agent Skills for Skills destination.",
                exc_info=True,
            )
            return (), SKILLS_SERVICE_ERROR_COPY, None

        skills = result.get("skills") if isinstance(result, Mapping) else None
        records = tuple(record for record in tuple(skills or ()) if isinstance(record, Mapping))
        return records, None, None

    @staticmethod
    def _safe_skill_text(value: Any, fallback: str = "", *, max_length: int = 1000) -> str:
        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        if not text:
            return fallback
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        return fallback

    @classmethod
    def _skill_field(cls, record: Mapping[str, Any], key: str, fallback: str = "") -> str:
        value = record.get(key)
        max_length = SKILL_TEXT_LIMITS.get(key, 1000)
        return cls._safe_skill_text(value, fallback=fallback, max_length=max_length)

    def _skill_body(self) -> str:
        lines = [
            "Local Agent Skills available to stage into Console:",
            "",
        ]
        for index, record in enumerate(self._local_skill_records, start=1):
            name = self._skill_field(record, "name") or self._skill_field(record, "skill_name", "Untitled skill")
            description = self._skill_field(record, "description")
            argument_hint = self._skill_field(record, "argument_hint")
            record_id = self._skill_field(record, "record_id")
            backend = self._skill_field(record, "backend", "local")

            lines.append(f"{index}. {name}")
            if description:
                lines.append(f"   description: {description}")
            if argument_hint:
                lines.append(f"   argument hint: {argument_hint}")
            if record_id:
                lines.append(f"   record id: {record_id}")
            lines.append(f"   backend: {backend}")
            lines.append("")
        return "\n".join(lines).strip()

    def _skill_names(self) -> list[str]:
        return [
            self._skill_field(record, "name") or self._skill_field(record, "skill_name", "Untitled skill")
            for record in self._local_skill_records
        ]

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    def compose_content(self) -> ComposeResult:
        local_skills_service = getattr(self.app_instance, "local_skills_service", None)
        skills_dir = getattr(local_skills_service, "skills_dir", None)
        skills_dir_label = str(skills_dir) if skills_dir is not None else "Local skills directory unavailable."

        with Vertical(id="skills-shell"):
            yield Static(
                "Skills | Agent Skills packs, validation, Console attachments | Local",
                id="skills-title",
                classes="ds-destination-header",
            )
            with DestinationModeStrip(id="skills-mode-strip", classes="destination-mode-strip"):
                yield Static(
                    "Mode: Installed / Validate / Attach | Source: local SKILL.md directories",
                    id="skills-mode-label",
                    classes="destination-section",
                )
            with Horizontal(id="skills-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="skills-list-pane", classes="destination-workbench-pane"):
                    yield Static("Column 1: Skill Library", classes="destination-section skills-column-title")
                    yield Static("Installed", classes="destination-section")
                    yield Static("Discover/Import", classes="destination-section")
                    yield Static("Validate", classes="destination-section")
                    yield Static("Scripts", classes="destination-section")
                    yield Static("References", classes="destination-section")
                    yield Static("Assets", classes="destination-section")
                    yield Static("Attachments", classes="destination-section")
                    yield Static(f"Local skills directory: {skills_dir_label}", id="skills-local-directory")
                yield self._column_divider("skills-list-detail-divider")
                with Vertical(id="skills-detail-pane", classes="destination-workbench-pane"):
                    yield Static(
                        "Column 2: Skill Detail / SKILL.md",
                        classes="destination-section skills-column-title",
                    )
                    if not self._skills_loaded:
                        yield Static(
                            "Loading local Agent Skills...",
                            id="skills-loading-state",
                        )
                        attach_label = "Attach local Skills to Console"
                        attach_disabled = True
                        attach_tooltip = "Stage local skill context after Skills finishes loading."
                    elif self._skills_lookup_error:
                        recovery_state = self._skills_lookup_recovery_state
                        yield Static(
                            self._skills_lookup_error,
                            id=(
                                recovery_state.stable_selector
                                if recovery_state is not None
                                else "skills-service-error"
                            ),
                        )
                        attach_label = "Attach local Skills to Console"
                        attach_disabled = True
                        attach_tooltip = (
                            recovery_state.disabled_tooltip
                            if recovery_state is not None
                            else "Skills service is unavailable; retry Skills later."
                        )
                    elif not self._local_skill_records:
                        yield Static(
                            "No local Agent Skills are installed yet.",
                            id="skills-empty-state",
                        )
                        attach_label = "Attach local Skills to Console"
                        attach_disabled = True
                        attach_tooltip = "Stage local skill context after installing a local Agent Skill."
                    else:
                        yield Static(
                            f"Installed local skills: {len(self._local_skill_records)}",
                            id="skills-local-summary",
                        )
                        for index, record in enumerate(self._local_skill_records):
                            name = self._skill_field(record, "name") or self._skill_field(
                                record,
                                "skill_name",
                                "Untitled skill",
                            )
                            description = self._skill_field(record, "description", "No description provided.")
                            yield Static(
                                Text.from_markup(
                                    f"{escape_markup(name)} - {escape_markup(description)}"
                                ),
                                id=f"skills-local-skill-{index}",
                        )
                        attach_label = "Attach local Skills to Console"
                        attach_disabled = False
                        attach_tooltip = "Stage local skill context in Console."
                yield self._column_divider("skills-detail-inspector-divider")
                with Vertical(id="skills-inspector-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static("Column 3: Skill Inspector", classes="destination-section skills-column-title")
                    yield Static("Actions", classes="destination-section")
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
                        attach_label,
                        id="skills-attach-to-console",
                        disabled=attach_disabled,
                        tooltip=attach_tooltip,
                    )

    @on(Button.Pressed, "#skills-attach-to-console")
    def attach_to_console(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._local_skill_records:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "No local Agent Skills are available to stage in Console.",
                    severity="warning",
                )
            return
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Console handoff is unavailable for Skills in this runtime.",
                    severity="warning",
                )
            return
        skill_count = len(self._local_skill_records)
        skill_names = self._skill_names()
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="skills",
                item_type="skills-context",
                title=f"Local Agent Skills ({skill_count})",
                body=self._skill_body(),
                display_summary=f"{skill_count} local Agent Skill{'s' if skill_count != 1 else ''} staged.",
                suggested_prompt="Help me choose and apply the relevant local Agent Skill.",
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata={
                    "skill_count": skill_count,
                    "skill_names": skill_names,
                    "backend": "local",
                    "skills_dir": str(
                        getattr(
                            getattr(self.app_instance, "local_skills_service", None),
                            "skills_dir",
                            "",
                        )
                        or ""
                    ),
                },
            )
        )
