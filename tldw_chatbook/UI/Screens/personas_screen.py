"""Personas destination shell for behavior profiles and prompt context."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...runtime_policy.types import PolicyDeniedError
from ...Utils.input_validation import sanitize_string, validate_text_input
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


logger = logger.bind(module="PersonasScreen")
PERSONAS_LOCAL_PAGE_SIZE = 5
PERSONAS_SERVICE_ERROR_COPY = "Personas service unavailable; retry Personas later."
PERSONAS_SERVICE_UNAVAILABLE_COPY = "Personas service is unavailable in this runtime."
PERSONAS_EMPTY_COPY = "No local characters or persona profiles are available yet."


class PersonasScreen(BaseAppScreen):
    """Characters, personas, prompts, dictionaries, and behavior profiles."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "personas", **kwargs)
        self._local_behavior_records: dict[str, tuple[Mapping[str, Any], ...]] = {
            "characters": (),
            "profiles": (),
        }
        self._local_behavior_counts: dict[str, int] = {
            "characters": 0,
            "profiles": 0,
        }
        self._personas_lookup_error: str | None = None
        self._personas_loaded = False

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_local_behavior_snapshot()

    @work(exclusive=True, thread=True)
    def _refresh_local_behavior_snapshot(self) -> None:
        records, counts, lookup_error = self._list_local_behavior_snapshot()
        self.app.call_from_thread(self._apply_local_behavior_snapshot, records, counts, lookup_error)

    def _apply_local_behavior_snapshot(
        self,
        records: dict[str, tuple[Mapping[str, Any], ...]],
        counts: dict[str, int],
        lookup_error: str | None = None,
    ) -> None:
        self._local_behavior_records = records
        self._local_behavior_counts = counts
        self._personas_lookup_error = lookup_error
        self._personas_loaded = True
        if self.is_mounted:
            self.refresh(recompose=True)

    @staticmethod
    def _run_maybe_awaitable(value: Any) -> Any:
        if inspect.isawaitable(value):
            return asyncio.run(value)
        return value

    @staticmethod
    def _safe_text(value: Any, fallback: str = "", *, max_length: int = 500) -> str:
        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        if not text:
            return fallback
        text = text.replace("<", "").replace(">", "")
        for pattern in ("javascript:", "onclick=", "onerror="):
            text = text.replace(pattern, "")
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        return fallback

    @classmethod
    def _record_name(cls, record_type: str, record: Mapping[str, Any]) -> str:
        keys = {
            "characters": ("name", "character_name", "title", "card_name"),
            "profiles": ("name", "profile_name", "title", "label"),
        }[record_type]
        for key in keys:
            name = cls._safe_text(record.get(key))
            if name:
                return name
        return "Untitled persona"

    @classmethod
    def _record_description(cls, record: Mapping[str, Any]) -> str:
        for key in ("description", "personality", "system_prompt", "summary"):
            description = cls._safe_text(record.get(key), max_length=600)
            if description:
                return description
        return ""

    @staticmethod
    def _response_records_and_count(result: Any) -> tuple[tuple[Mapping[str, Any], ...], int]:
        if isinstance(result, Mapping):
            raw_items = result.get("items") or result.get("profiles") or result.get("characters")
            total = result.get("total")
            pagination = result.get("pagination")
            if isinstance(pagination, Mapping):
                total = pagination.get("total", total)
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            raw_items = result
            total = None
        else:
            raw_items = ()
            total = None

        records = tuple(record for record in tuple(raw_items or ()) if isinstance(record, Mapping))
        try:
            count = int(total) if total is not None else len(records)
        except (TypeError, ValueError):
            count = len(records)
        return records, count

    def _list_local_behavior_snapshot(
        self,
    ) -> tuple[dict[str, tuple[Mapping[str, Any], ...]], dict[str, int], str | None]:
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        list_characters = getattr(service, "list_characters", None)
        list_profiles = getattr(service, "list_persona_profiles", None)
        empty_records: dict[str, tuple[Mapping[str, Any], ...]] = {
            "characters": (),
            "profiles": (),
        }
        empty_counts = {"characters": 0, "profiles": 0}
        if not callable(list_characters) or not callable(list_profiles):
            return empty_records, empty_counts, PERSONAS_SERVICE_UNAVAILABLE_COPY

        try:
            characters_result = self._run_maybe_awaitable(
                list_characters(
                    mode="local",
                    limit=PERSONAS_LOCAL_PAGE_SIZE,
                    offset=0,
                )
            )
            profiles_result = self._run_maybe_awaitable(
                list_profiles(
                    mode="local",
                    active_only=True,
                    include_deleted=False,
                    limit=PERSONAS_LOCAL_PAGE_SIZE,
                    offset=0,
                )
            )
        except PolicyDeniedError as exc:
            policy_message = self._safe_text(exc.user_message, PERSONAS_SERVICE_ERROR_COPY)
            return empty_records, empty_counts, policy_message
        except Exception:
            logger.warning(
                "Failed to load local Personas behavior snapshot.",
                exc_info=True,
            )
            return empty_records, empty_counts, PERSONAS_SERVICE_ERROR_COPY

        characters, character_count = self._response_records_and_count(characters_result)
        profiles, profile_count = self._response_records_and_count(profiles_result)
        return (
            {
                "characters": characters,
                "profiles": profiles,
            },
            {
                "characters": character_count,
                "profiles": profile_count,
            },
            None,
        )

    def _has_local_behavior_context(self) -> bool:
        return any(count > 0 for count in self._local_behavior_counts.values())

    def _snapshot_body(self) -> str:
        lines = ["Local Personas behavior context staged for Console:", ""]
        for record_type, label in (
            ("characters", "Characters"),
            ("profiles", "Persona profiles"),
        ):
            lines.append(f"{label}: {self._local_behavior_counts[record_type]}")
            for index, record in enumerate(self._local_behavior_records[record_type], start=1):
                name = self._record_name(record_type, record)
                description = self._record_description(record)
                lines.append(f"  {index}. {name}")
                if description:
                    lines.append(f"     description: {description}")
            lines.append("")
        return "\n".join(lines).strip()

    def _record_names(self, record_type: str) -> list[str]:
        return [
            self._record_name(record_type, record)
            for record in self._local_behavior_records[record_type]
        ]

    def compose_content(self) -> ComposeResult:
        has_context = self._has_local_behavior_context()
        with Vertical(id="personas-shell"):
            yield Static("Personas", id="personas-title", classes="ds-destination-header")
            yield Static(
                "Characters, personas, prompts, dictionaries, lore, and behavior profiles.",
                id="personas-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="personas-sections", classes="ds-panel"):
                yield Button(
                    "Open Personas",
                    id="personas-open-profiles",
                    tooltip="Open character, prompt, dictionary, and lore management.",
                )
                yield Static(
                    "Characters, prompts, dictionaries, and lore stay here; Library owns saved conversation browsing.",
                    id="personas-boundary",
                    classes="destination-purpose",
                )
                yield Static("Local Personas snapshot", classes="destination-section")
                if not self._personas_loaded:
                    yield Static(
                        "Loading local Personas behavior context...",
                        id="personas-loading-state",
                    )
                    attach_disabled = True
                    attach_tooltip = "Stage local persona context after Personas finishes loading."
                elif self._personas_lookup_error:
                    yield Static(
                        self._personas_lookup_error,
                        id="personas-service-error",
                    )
                    attach_disabled = True
                    attach_tooltip = "Personas service is unavailable; retry Personas later."
                elif not has_context:
                    yield Static(
                        PERSONAS_EMPTY_COPY,
                        id="personas-empty-state",
                    )
                    attach_disabled = True
                    attach_tooltip = "Stage local persona context after adding characters or persona profiles."
                else:
                    for record_type, label, widget_id in (
                        ("characters", "Characters", "personas-characters-summary"),
                        ("profiles", "Persona profiles", "personas-profiles-summary"),
                    ):
                        yield Static(
                            f"{label}: {self._local_behavior_counts[record_type]}",
                            id=widget_id,
                        )
                        for index, record in enumerate(self._local_behavior_records[record_type]):
                            yield Static(
                                Text.from_markup(
                                    escape_markup(self._record_name(record_type, record))
                                ),
                                id=f"personas-{record_type}-item-{index}",
                            )
                    attach_disabled = False
                    attach_tooltip = "Stage local persona context in Console."
                yield Button(
                    "Attach to Console",
                    id="personas-attach-to-console",
                    disabled=attach_disabled,
                    tooltip=attach_tooltip,
                )

    @on(Button.Pressed, "#personas-open-profiles")
    def open_profiles(self) -> None:
        self.post_message(NavigateToScreen("ccp"))

    @on(Button.Pressed, "#personas-attach-to-console")
    def attach_to_console(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._has_local_behavior_context():
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    self._personas_lookup_error or PERSONAS_EMPTY_COPY,
                    severity="warning",
                )
            return
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Console handoff is unavailable for Personas in this runtime.",
                    severity="warning",
                )
            return
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="personas",
                item_type="personas-context",
                title="Local Personas Context",
                body=self._snapshot_body(),
                display_summary="Local Personas context staged.",
                suggested_prompt="Use these local characters and persona profiles to guide the next response.",
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata={
                    "character_count": self._local_behavior_counts["characters"],
                    "persona_profile_count": self._local_behavior_counts["profiles"],
                    "character_names": self._record_names("characters"),
                    "persona_profile_names": self._record_names("profiles"),
                    "backend": "local",
                },
            )
        )
