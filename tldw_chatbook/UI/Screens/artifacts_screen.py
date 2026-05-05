"""Artifacts destination shell for generated outputs and Chatbooks."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html import escape as html_escape
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Utils.input_validation import sanitize_string, validate_text_input
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from .destination_recovery import DestinationRecoveryState


logger = logger.bind(module="ArtifactsScreen")
CHATBOOK_SERVICE_ERROR_COPY = "Chatbook service unavailable; retry Artifacts later."
DANGEROUS_TEXT_PATTERNS = ("javascript:", "onclick=", "onerror=")
ARTIFACTS_EMPTY_CHATBOOK_RECOVERY = DestinationRecoveryState(
    status_label="Select an artifact",
    unavailable_what="Console launch for Chatbook artifacts",
    why="no local Chatbook artifact exists",
    next_action="Create or import a Chatbook artifact before opening it in Console.",
    recovery_action="Artifacts",
    authority_owner="local Chatbook service",
    stable_selector="artifacts-console-unavailable",
    disabled_tooltip="Create or import a Chatbook artifact before opening it in Console.",
)
ARTIFACTS_CHATBOOK_SERVICE_UNAVAILABLE_RECOVERY = DestinationRecoveryState(
    status_label="Service unavailable",
    unavailable_what="Console launch for Chatbook artifacts",
    why="the local Chatbook service is unavailable",
    next_action="Retry Artifacts after the local Chatbook service is available.",
    recovery_action="Retry Artifacts",
    authority_owner="local Chatbook service",
    stable_selector="artifacts-console-unavailable",
    disabled_tooltip="Retry Artifacts after the local Chatbook service is available.",
)


class ArtifactsScreen(BaseAppScreen):
    """Generated outputs, portable bundles, reports, datasets, and Chatbooks."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "artifacts", **kwargs)
        self._latest_chatbook_console_launch: dict[str, Any] | None = None
        self._chatbook_lookup_error: str | None = None

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_latest_chatbook_context()

    @work(exclusive=True, thread=True)
    def _refresh_latest_chatbook_context(self) -> None:
        launch_kwargs, lookup_error = self._latest_local_chatbook_console_launch()
        self.app.call_from_thread(self._apply_latest_chatbook_context, launch_kwargs, lookup_error)

    def _apply_latest_chatbook_context(
        self,
        launch_kwargs: dict[str, Any] | None,
        lookup_error: str | None = None,
    ) -> None:
        self._latest_chatbook_console_launch = launch_kwargs
        self._chatbook_lookup_error = lookup_error
        if self.is_mounted:
            self.refresh(recompose=True)

    @property
    def _blocked_chatbook_recovery_state(self) -> DestinationRecoveryState:
        return (
            ARTIFACTS_CHATBOOK_SERVICE_UNAVAILABLE_RECOVERY
            if self._chatbook_lookup_error
            else ARTIFACTS_EMPTY_CHATBOOK_RECOVERY
        )

    @staticmethod
    def _text(value: Any, fallback: str = "") -> str:
        text = str(value or "").strip()
        return text or fallback

    @classmethod
    def _safe_text(cls, value: Any, fallback: str = "", *, max_length: int = 1000) -> str:
        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        if not text:
            return fallback
        text = html_escape(text, quote=False)
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        for pattern in DANGEROUS_TEXT_PATTERNS:
            text = re.sub(re.escape(pattern), pattern.rstrip(":=").replace("=", ""), text, flags=re.IGNORECASE)
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        return fallback

    @classmethod
    def _csv(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return cls._safe_text(value) or None
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            safe_items = [cls._safe_text(item) for item in value]
            text = ", ".join(item for item in safe_items if item)
            return text or None
        return cls._safe_text(value) or None

    @classmethod
    def _safe_identifier(cls, value: Any) -> int | str | None:
        if isinstance(value, int):
            return value
        text = cls._safe_text(value, max_length=128)
        return text or None

    @classmethod
    def _datetime_sort_key(cls, value: Any) -> float:
        text = cls._text(value)
        if not text:
            return 0.0
        try:
            normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _chatbook_id_sort_key(cls, value: Any) -> tuple[int, int, str]:
        text = cls._text(value)
        if text.isdigit():
            return (1, int(text), "")
        return (0, 0, text)

    @classmethod
    def _chatbook_sort_key(cls, record: Mapping[str, Any]) -> tuple[float, int, int, str]:
        updated_at = cls._datetime_sort_key(record.get("updated_at") or record.get("created_at"))
        id_kind, id_number, id_text = cls._chatbook_id_sort_key(record.get("chatbook_id") or record.get("id"))
        return (updated_at, id_kind, id_number, id_text)

    @classmethod
    def _build_chatbook_console_launch(cls, record: Mapping[str, Any]) -> dict[str, Any] | None:
        chatbook_id = cls._safe_identifier(record.get("chatbook_id") or record.get("id"))
        if chatbook_id in (None, ""):
            return None
        title = cls._safe_text(record.get("name") or record.get("title"), "Untitled Chatbook")
        description = cls._safe_text(record.get("description"))
        payload = {
            "target_id": f"local:chatbook:{chatbook_id}",
            "chatbook_id": chatbook_id,
            "record_id": cls._safe_text(record.get("id")),
            "file_path": cls._safe_text(record.get("file_path"), max_length=2000),
            "description": description,
            "tags": cls._csv(record.get("tags")),
            "categories": cls._csv(record.get("categories")),
            "updated_at": cls._safe_text(record.get("updated_at")),
        }
        return {
            "source": "artifacts",
            "title": title,
            "payload": payload,
            "status": "ready",
            "recovery": "Review this Chatbook artifact in Console or return to Artifacts.",
            "action_label": "Open Chatbook artifact",
        }

    def _latest_local_chatbook_console_launch(self) -> tuple[dict[str, Any] | None, str | None]:
        service = getattr(self.app_instance, "local_chatbook_service", None)
        list_chatbooks = getattr(service, "list_chatbooks", None)
        if not callable(list_chatbooks):
            return None, None
        try:
            result = list_chatbooks(q=None, limit=25, offset=0)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
        except Exception:
            logger.warning(
                "Failed to load latest local Chatbook artifact for Console launch.",
                exc_info=True,
            )
            return None, CHATBOOK_SERVICE_ERROR_COPY
        records = [record for record in tuple(result or ()) if isinstance(record, Mapping)]
        if not records:
            return None, None
        latest_record = max(records, key=self._chatbook_sort_key)
        return self._build_chatbook_console_launch(latest_record), None

    def compose_content(self) -> ComposeResult:
        launch_kwargs = self._latest_chatbook_console_launch
        with Vertical(id="artifacts-shell"):
            yield Static("Artifacts", id="artifacts-title", classes="ds-destination-header")
            yield Static(
                "Generated outputs, bundles, reports, datasets, and Chatbooks.",
                id="artifacts-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="artifacts-sections", classes="ds-panel"):
                yield Button(
                    "Open Chatbooks",
                    id="artifacts-open-chatbooks",
                    tooltip="Open portable Chatbook bundles.",
                )
                yield Static(
                    "Generated outputs from local and server output services will appear here.",
                    id="artifacts-output-status",
                    classes="destination-purpose",
                )
                if launch_kwargs is not None:
                    title = str(launch_kwargs["title"])
                    payload = launch_kwargs.get("payload") or {}
                    description = str(payload.get("description") or "").strip()
                    yield Static("Console launch available", classes="destination-section")
                    yield Static(
                        Text.from_markup(
                            "Console can launch latest Chatbook artifact: "
                            f"{escape_markup(title)}."
                        ),
                        id="artifacts-console-available",
                    )
                    if description:
                        yield Static(
                            Text.from_markup(escape_markup(description)),
                            id="artifacts-chatbook-description",
                        )
                    yield Button(
                        Text.from_markup(f"Launch {escape_markup(title)} in Console"),
                        id="artifacts-use-in-console",
                        tooltip="Open the latest local Chatbook artifact in Console.",
                    )
                else:
                    yield Static("Console launch unavailable", classes="destination-section")
                    recovery_state = self._blocked_chatbook_recovery_state
                    yield Static(
                        recovery_state.visible_copy,
                        id=recovery_state.stable_selector,
                    )
                    yield Button(
                        "Console launch unavailable",
                        id="artifacts-use-in-console",
                        disabled=True,
                        tooltip=recovery_state.disabled_tooltip,
                    )

    @on(Button.Pressed, "#artifacts-open-chatbooks")
    def open_chatbooks(self) -> None:
        self.post_message(NavigateToScreen("chatbooks"))

    @on(Button.Pressed, "#artifacts-use-in-console")
    def use_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        launch_kwargs = self._latest_chatbook_console_launch
        if launch_kwargs is None:
            self.app_instance.notify(
                self._blocked_chatbook_recovery_state.disabled_tooltip,
                severity="warning",
            )
            return
        open_in_console = getattr(self.app_instance, "open_console_for_live_work", None)
        if not callable(open_in_console):
            self.app_instance.notify(
                "Console launch is unavailable for Artifacts in this runtime.",
                severity="warning",
            )
            return
        open_in_console(**launch_kwargs)
