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
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Widgets.destination_workbench import DestinationModeStrip
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
        self._requested_chatbook_target_id: str | None = None
        self._chatbook_context_loaded = False

    def on_mount(self) -> None:
        super().on_mount()
        self._consume_pending_chatbook_target_id()
        self._refresh_latest_chatbook_context()

    def on_screen_resume(self) -> None:
        """Refresh one-shot Chatbook handoffs when returning to Artifacts."""
        if not str(
            getattr(self.app_instance, "pending_artifacts_chatbook_target_id", "") or ""
        ).strip():
            return
        self._consume_pending_chatbook_target_id()
        self._refresh_latest_chatbook_context()

    def _consume_pending_chatbook_target_id(self) -> None:
        target_id = str(
            getattr(self.app_instance, "pending_artifacts_chatbook_target_id", "") or ""
        ).strip()
        if not target_id:
            return
        self._requested_chatbook_target_id = target_id
        self.app_instance.pending_artifacts_chatbook_target_id = None

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
        self._chatbook_context_loaded = True
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
    def _safe_metadata_value(cls, value: Any, *, max_length: int = 1000) -> str | int | float | bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        text = cls._safe_text(value, max_length=max_length)
        return text or None

    @classmethod
    def _console_saved_artifact_payload(cls, metadata: Any) -> dict[str, Any]:
        if not isinstance(metadata, Mapping):
            return {}

        artifact_source = cls._safe_metadata_value(metadata.get("artifact_source"), max_length=128)
        artifact_kind = cls._safe_metadata_value(metadata.get("artifact_kind"), max_length=128)
        if str(artifact_source or "").strip().lower() != "console":
            return {}
        if str(artifact_kind or "").strip().lower() != "assistant-response":
            return {}

        payload: dict[str, Any] = {
            "artifact_source": artifact_source,
            "artifact_kind": artifact_kind,
        }
        for key in ("conversation_id", "message_id", "message_role", "provider", "model"):
            if (safe_value := cls._safe_metadata_value(metadata.get(key), max_length=256)) is not None:
                payload[key] = safe_value

        if (content_preview := cls._safe_metadata_value(metadata.get("content"), max_length=1000)) is not None:
            payload["content_preview"] = content_preview
        content_truncated = metadata.get("content_truncated")
        if isinstance(content_truncated, bool):
            payload["content_truncated"] = content_truncated
        elif "content_preview" in payload:
            payload["content_truncated"] = False
        return payload

    @staticmethod
    def _console_saved_artifact_provenance(payload: Mapping[str, Any]) -> str | None:
        if str(payload.get("artifact_source") or "").strip().lower() != "console":
            return None
        provider = str(payload.get("provider") or "").strip()
        model = str(payload.get("model") or "").strip()
        if provider and model:
            return f"Saved from Console assistant response via {provider} / {model}."
        if provider:
            return f"Saved from Console assistant response via {provider}."
        if model:
            return f"Saved from Console assistant response using {model}."
        return "Saved from Console assistant response."

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
    def _chatbook_identifier(cls, record: Mapping[str, Any]) -> int | str | None:
        return cls._safe_identifier(record.get("chatbook_id") or record.get("id"))

    @classmethod
    def _chatbook_target_id(cls, record: Mapping[str, Any]) -> str:
        chatbook_id = cls._chatbook_identifier(record)
        if chatbook_id in (None, ""):
            return ""
        return f"local:chatbook:{chatbook_id}"

    @classmethod
    def _build_chatbook_console_launch(cls, record: Mapping[str, Any]) -> dict[str, Any] | None:
        chatbook_id = cls._chatbook_identifier(record)
        if chatbook_id in (None, ""):
            return None
        target_id = cls._chatbook_target_id(record)
        if not target_id:
            return None
        title = cls._safe_text(record.get("name") or record.get("title"), "Untitled Chatbook")
        description = cls._safe_text(record.get("description"))
        payload = {
            "target_id": target_id,
            "chatbook_id": chatbook_id,
            "record_id": cls._safe_text(record.get("id")),
            "file_path": cls._safe_text(record.get("file_path"), max_length=2000),
            "description": description,
            "tags": cls._csv(record.get("tags")),
            "categories": cls._csv(record.get("categories")),
            "updated_at": cls._safe_text(record.get("updated_at")),
        }
        payload.update(cls._console_saved_artifact_payload(record.get("metadata")))
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
        if self._requested_chatbook_target_id:
            requested_record = next(
                (
                    record
                    for record in records
                    if self._chatbook_target_id(record) == self._requested_chatbook_target_id
                ),
                None,
            )
            if requested_record is not None:
                return self._build_chatbook_console_launch(requested_record), None
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
            with DestinationModeStrip(id="artifacts-mode-strip", classes="destination-mode-strip"):
                yield Static(
                    "Mode: Chatbooks | Outputs | Reports | Imports",
                    id="artifacts-mode-label",
                    classes="destination-section",
                )
            with Horizontal(id="artifacts-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="artifacts-list-pane", classes="destination-workbench-pane"):
                    yield Static("Artifact Sources", classes="destination-section")
                    yield Button(
                        "Open Chatbooks",
                        id="artifacts-open-chatbooks",
                        tooltip="Open portable Chatbook bundles.",
                    )
                    yield Button(
                        "Open Library",
                        id="artifacts-open-library",
                        tooltip="Open Library source material that can produce or contextualize artifacts.",
                    )
                    yield Button(
                        "Import Artifact",
                        id="artifacts-import-artifact",
                        disabled=True,
                        tooltip="Artifact import is a later-stage path for this shell.",
                    )
                    yield Static(
                        "Generated outputs from local and server output services will appear here.",
                        id="artifacts-output-status",
                        classes="destination-purpose",
                    )
                with Vertical(id="artifacts-detail-pane", classes="destination-workbench-pane"):
                    yield Static("Artifact Detail", classes="destination-section")
                    if not self._chatbook_context_loaded:
                        yield Static(
                            "Loading latest local Chatbook artifact...",
                            id="artifacts-loading-state",
                        )
                    elif launch_kwargs is not None:
                        yield Static("Latest Chatbook artifact is available.", id="artifacts-detail-ready")
                    else:
                        yield Static("No local Chatbook artifact is selected.", id="artifacts-detail-empty")
                with Vertical(id="artifacts-inspector-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static("Console Inspector", classes="destination-section")
                    if launch_kwargs is not None:
                        title = str(launch_kwargs["title"])
                        payload = launch_kwargs.get("payload") or {}
                        target_id = str(payload.get("target_id") or "").strip()
                        is_requested = bool(
                            self._requested_chatbook_target_id
                            and target_id == self._requested_chatbook_target_id
                        )
                        launch_scope = "requested" if is_requested else "latest"
                        description = str(payload.get("description") or "").strip()
                        provenance = self._console_saved_artifact_provenance(payload)
                        content_preview = str(payload.get("content_preview") or "").strip()
                        yield Static("Console launch available", classes="destination-section")
                        yield Static(
                            Text.from_markup(
                                f"Console can launch {launch_scope} Chatbook artifact: "
                                f"{escape_markup(title)}."
                            ),
                            id="artifacts-console-available",
                        )
                        if description:
                            yield Static(
                                Text.from_markup(escape_markup(description)),
                                id="artifacts-chatbook-description",
                            )
                        if provenance:
                            yield Static(
                                Text.from_markup(escape_markup(provenance)),
                                id="artifacts-chatbook-provenance",
                            )
                        if content_preview:
                            yield Static(
                                Text.from_markup(f"Preview: {escape_markup(content_preview)}"),
                                id="artifacts-chatbook-content-preview",
                            )
                        yield Button(
                            Text.from_markup(f"Launch {escape_markup(title)} in Console"),
                            id="artifacts-use-in-console",
                            tooltip=f"Open the {launch_scope} local Chatbook artifact in Console.",
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

    @on(Button.Pressed, "#artifacts-open-library")
    def open_library(self) -> None:
        self.post_message(NavigateToScreen("library"))

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
