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
from textual.worker import Worker, WorkerState

from ...Chat.answer_citations import summarize_citation_artifact_metadata
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Navigation.pending_handoff_store import (
    ARTIFACT_CHATBOOK_RECORD_PREFIX,
    HandoffChannel,
    HandoffClaim,
)
from .destination_recovery import DestinationRecoveryState


logger = logger.bind(module="ArtifactsScreen")
CHATBOOK_SERVICE_ERROR_COPY = "Chatbook service unavailable; retry Artifacts later."
CHATBOOK_TARGET_MISSING_COPY = "The requested local Chatbook artifact no longer exists."
DANGEROUS_TEXT_PATTERNS = ("javascript:", "onclick=", "onerror=")
CHATBOOK_OUTCOME_SUCCESS = "success"
CHATBOOK_OUTCOME_EMPTY = "empty"
CHATBOOK_OUTCOME_MISSING = "missing"
CHATBOOK_OUTCOME_TRANSIENT = "transient"
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
ARTIFACTS_CHATBOOK_TARGET_MISSING_RECOVERY = DestinationRecoveryState(
    status_label="Artifact not found",
    unavailable_what="The requested local Chatbook artifact",
    why="the requested local Chatbook artifact no longer exists",
    next_action="Return to Console and choose an available Chatbook artifact.",
    recovery_action="Console",
    authority_owner="local Chatbook service",
    stable_selector="artifacts-chatbook-target-missing",
    disabled_tooltip=CHATBOOK_TARGET_MISSING_COPY,
)


class ArtifactsScreen(BaseAppScreen):
    """Generated outputs, portable bundles, reports, datasets, and Chatbooks."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "artifacts", **kwargs)
        self._latest_chatbook_console_launch: dict[str, Any] | None = None
        self._chatbook_lookup_error: str | None = None
        self._chatbook_context_requested = False
        self._chatbook_missing_target = False
        self._chatbook_context_loaded = False
        self._active_chatbook_claim: HandoffClaim[str] | None = None
        self._chatbook_refresh_worker: Worker[Any] | None = None
        self._chatbook_refresh_generation = 0
        self._chatbook_unmounted = True

    def on_mount(self) -> None:
        super().on_mount()
        self._chatbook_unmounted = False
        self._start_chatbook_refresh()

    def on_screen_resume(self) -> None:
        """Refresh one-shot Chatbook handoffs when returning to Artifacts."""
        if self._chatbook_refresh_worker is None or (
            self._active_chatbook_claim is None
            and self.app_instance.pending_handoffs.has_pending(
                HandoffChannel.ARTIFACT_CHATBOOK_TARGET
            )
        ):
            self._start_chatbook_refresh()

    def on_unmount(self) -> None:
        self._chatbook_unmounted = True
        self._chatbook_refresh_generation += 1
        self._release_active_chatbook_claim()
        worker = self._chatbook_refresh_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        super().on_unmount()

    def _release_active_chatbook_claim(self) -> None:
        claim = self._active_chatbook_claim
        if claim is None:
            return
        self.app_instance.pending_handoffs.release(claim)
        if self._active_chatbook_claim is claim:
            self._active_chatbook_claim = None

    def _start_chatbook_refresh(self) -> None:
        """Start one generation after releasing any superseded exact claim."""
        if self._chatbook_unmounted:
            return

        previous_worker = self._chatbook_refresh_worker
        self._release_active_chatbook_claim()
        self._chatbook_refresh_generation += 1
        generation = self._chatbook_refresh_generation
        claim = self.app_instance.pending_handoffs.claim(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET
        )
        self._active_chatbook_claim = claim
        self._chatbook_context_loaded = False
        self._chatbook_context_requested = False
        self._chatbook_missing_target = False
        self._chatbook_lookup_error = None
        self._latest_chatbook_console_launch = None
        if self.is_mounted:
            self.refresh(recompose=True)

        if previous_worker is not None and not previous_worker.is_finished:
            previous_worker.cancel()
        try:
            self._chatbook_refresh_worker = self._refresh_chatbook_context(
                generation,
                claim,
                claim.value if claim is not None else None,
            )
        except Exception as exc:
            self._release_active_chatbook_claim()
            logger.warning(
                "Chatbook refresh could not start (exception_category={}).",
                type(exc).__name__,
            )

    @work(exclusive=True, thread=True)
    def _refresh_chatbook_context(
        self,
        generation: int,
        claim: HandoffClaim[str] | None,
        requested_target: str | None,
    ) -> None:
        if claim is None:
            launch_kwargs, lookup_error = self._latest_local_chatbook_console_launch()
            outcome = (
                CHATBOOK_OUTCOME_TRANSIENT
                if lookup_error
                else (
                    CHATBOOK_OUTCOME_SUCCESS
                    if launch_kwargs is not None
                    else CHATBOOK_OUTCOME_EMPTY
                )
            )
        else:
            outcome, launch_kwargs = self._exact_local_chatbook_console_launch(
                requested_target
            )
        self.app.call_from_thread(
            self._apply_chatbook_refresh_outcome,
            generation,
            claim,
            outcome,
            launch_kwargs,
        )

    def _apply_chatbook_refresh_outcome(
        self,
        generation: int,
        claim: HandoffClaim[str] | None,
        outcome: str,
        launch_kwargs: dict[str, Any] | None,
    ) -> None:
        store = self.app_instance.pending_handoffs
        if (
            self._chatbook_unmounted
            or generation != self._chatbook_refresh_generation
            or claim is not self._active_chatbook_claim
        ):
            if claim is not None:
                store.release(claim)
            return

        try:
            self._latest_chatbook_console_launch = launch_kwargs
            self._chatbook_context_requested = (
                claim is not None and outcome == CHATBOOK_OUTCOME_SUCCESS
            )
            self._chatbook_missing_target = outcome == CHATBOOK_OUTCOME_MISSING
            self._chatbook_lookup_error = (
                CHATBOOK_SERVICE_ERROR_COPY
                if outcome == CHATBOOK_OUTCOME_TRANSIENT
                else None
            )
            self._chatbook_context_loaded = True
            if self.is_mounted:
                self.refresh(recompose=True)

            if claim is None:
                return
            if outcome == CHATBOOK_OUTCOME_MISSING:
                self.app_instance.notify(
                    CHATBOOK_TARGET_MISSING_COPY,
                    severity="warning",
                )
                store.acknowledge(claim)
            elif outcome == CHATBOOK_OUTCOME_SUCCESS:
                store.acknowledge(claim)
            else:
                store.release(claim)
        except Exception as exc:
            if claim is not None:
                store.release(claim)
            logger.warning(
                "Chatbook refresh callback failed (exception_category={}).",
                type(exc).__name__,
            )
        finally:
            if self._active_chatbook_claim is claim:
                self._active_chatbook_claim = None

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker is not self._chatbook_refresh_worker:
            return
        if event.state not in {
            WorkerState.CANCELLED,
            WorkerState.ERROR,
            WorkerState.SUCCESS,
        }:
            return
        if self._active_chatbook_claim is not None:
            self._release_active_chatbook_claim()
        if (
            not self._chatbook_unmounted
            and not self._chatbook_context_loaded
            and event.state
            in {WorkerState.CANCELLED, WorkerState.ERROR, WorkerState.SUCCESS}
        ):
            self._latest_chatbook_console_launch = None
            self._chatbook_context_requested = False
            self._chatbook_missing_target = False
            self._chatbook_lookup_error = CHATBOOK_SERVICE_ERROR_COPY
            self._chatbook_context_loaded = True
            if self.is_mounted:
                self.refresh(recompose=True)
        if event.state is WorkerState.ERROR:
            logger.warning(
                "Chatbook refresh worker failed (exception_category=worker)."
            )

    @property
    def _blocked_chatbook_recovery_state(self) -> DestinationRecoveryState:
        if self._chatbook_missing_target:
            return ARTIFACTS_CHATBOOK_TARGET_MISSING_RECOVERY
        return (
            ARTIFACTS_CHATBOOK_SERVICE_UNAVAILABLE_RECOVERY
            if self._chatbook_lookup_error
            else ARTIFACTS_EMPTY_CHATBOOK_RECOVERY
        )

    @staticmethod
    def _text(value: Any, fallback: str = "") -> str:
        text = str(value or "").strip()
        return text or fallback

    @staticmethod
    def _literal_text(value: Any) -> Text:
        return Text.from_markup(escape_markup(str(value)))

    @classmethod
    def _safe_text(
        cls, value: Any, fallback: str = "", *, max_length: int = 1000
    ) -> str:
        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        if not text:
            return fallback
        text = html_escape(text, quote=False)
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        for pattern in DANGEROUS_TEXT_PATTERNS:
            text = re.sub(
                re.escape(pattern),
                pattern.rstrip(":=").replace("=", ""),
                text,
                flags=re.IGNORECASE,
            )
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
    def _safe_metadata_value(
        cls, value: Any, *, max_length: int = 1000
    ) -> str | int | float | bool | None:
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

        artifact_source = cls._safe_metadata_value(
            metadata.get("artifact_source"), max_length=128
        )
        artifact_kind = cls._safe_metadata_value(
            metadata.get("artifact_kind"), max_length=128
        )
        if str(artifact_source or "").strip().lower() != "console":
            return {}
        if str(artifact_kind or "").strip().lower() != "assistant-response":
            return {}

        payload: dict[str, Any] = {
            "artifact_source": artifact_source,
            "artifact_kind": artifact_kind,
        }
        for key in (
            "conversation_id",
            "message_id",
            "message_role",
            "provider",
            "model",
        ):
            if (
                safe_value := cls._safe_metadata_value(
                    metadata.get(key), max_length=256
                )
            ) is not None:
                payload[key] = safe_value

        if (
            content_preview := cls._safe_metadata_value(
                metadata.get("content"), max_length=1000
            )
        ) is not None:
            payload["content_preview"] = content_preview
        content_truncated = metadata.get("content_truncated")
        if isinstance(content_truncated, bool):
            payload["content_truncated"] = content_truncated
        elif "content_preview" in payload:
            payload["content_truncated"] = False
        payload.update(cls._safe_console_summary_payload(metadata))
        return payload

    @classmethod
    def _safe_console_summary_payload(cls, metadata: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in summarize_citation_artifact_metadata(metadata).items():
            if isinstance(value, bool) or isinstance(value, int):
                payload[key] = value
                continue
            if (
                safe_value := cls._safe_metadata_value(value, max_length=256)
            ) is not None:
                payload[key] = safe_value
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
    def _chatbook_sort_key(
        cls, record: Mapping[str, Any]
    ) -> tuple[float, int, int, str]:
        updated_at = cls._datetime_sort_key(
            record.get("updated_at") or record.get("created_at")
        )
        id_kind, id_number, id_text = cls._chatbook_id_sort_key(
            record.get("chatbook_id") or record.get("id")
        )
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
    def _build_chatbook_console_launch(
        cls, record: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        chatbook_id = cls._chatbook_identifier(record)
        if chatbook_id in (None, ""):
            return None
        target_id = cls._chatbook_target_id(record)
        if not target_id:
            return None
        title = cls._safe_text(
            record.get("name") or record.get("title"), "Untitled Chatbook"
        )
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

    def _latest_local_chatbook_console_launch(
        self,
    ) -> tuple[dict[str, Any] | None, str | None]:
        service = getattr(self.app_instance, "local_chatbook_service", None)
        list_chatbooks = getattr(service, "list_chatbooks", None)
        if not callable(list_chatbooks):
            return None, None
        try:
            result = list_chatbooks(q=None, limit=25, offset=0)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
        except Exception as exc:
            logger.warning(
                "Failed to load latest local Chatbook artifact "
                "(exception_category={}).",
                type(exc).__name__,
            )
            return None, CHATBOOK_SERVICE_ERROR_COPY
        records = [
            record for record in tuple(result or ()) if isinstance(record, Mapping)
        ]
        if not records:
            return None, None
        latest_record = max(records, key=self._chatbook_sort_key)
        return self._build_chatbook_console_launch(latest_record), None

    def _exact_local_chatbook_console_launch(
        self,
        requested_target: str | None,
    ) -> tuple[str, dict[str, Any] | None]:
        if not isinstance(requested_target, str) or not requested_target.startswith(
            ARTIFACT_CHATBOOK_RECORD_PREFIX
        ):
            return CHATBOOK_OUTCOME_TRANSIENT, None
        chatbook_id = requested_target.removeprefix(
            ARTIFACT_CHATBOOK_RECORD_PREFIX
        ).strip()
        if not chatbook_id:
            return CHATBOOK_OUTCOME_TRANSIENT, None

        service = getattr(self.app_instance, "local_chatbook_service", None)
        get_chatbook = getattr(service, "get_chatbook", None)
        if not callable(get_chatbook):
            return CHATBOOK_OUTCOME_TRANSIENT, None
        try:
            record = get_chatbook(chatbook_id)
            if inspect.isawaitable(record):
                record = asyncio.run(record)
        except KeyError:
            return CHATBOOK_OUTCOME_MISSING, None
        except Exception as exc:
            logger.warning(
                "Exact Chatbook lookup failed (exception_category={}).",
                type(exc).__name__,
            )
            return CHATBOOK_OUTCOME_TRANSIENT, None

        if not isinstance(record, Mapping):
            logger.warning(
                "Exact Chatbook lookup returned an invalid service response."
            )
            return CHATBOOK_OUTCOME_TRANSIENT, None
        if self._chatbook_target_id(record) != requested_target:
            logger.warning(
                "Exact Chatbook lookup returned a mismatched service response."
            )
            return CHATBOOK_OUTCOME_TRANSIENT, None
        launch_kwargs = self._build_chatbook_console_launch(record)
        if launch_kwargs is None:
            logger.warning(
                "Exact Chatbook lookup returned an unusable service response."
            )
            return CHATBOOK_OUTCOME_TRANSIENT, None
        return CHATBOOK_OUTCOME_SUCCESS, launch_kwargs

    def compose_content(self) -> ComposeResult:
        launch_kwargs = self._latest_chatbook_console_launch
        with Vertical(id="artifacts-shell"):
            yield Static(
                "Artifacts", id="artifacts-title", classes="ds-destination-header"
            )
            yield Static(
                "Generated outputs, bundles, reports, datasets, and Chatbooks.",
                id="artifacts-purpose",
                classes="destination-purpose",
            )
            with DestinationModeStrip(
                id="artifacts-mode-strip", classes="destination-mode-strip"
            ):
                yield Static(
                    "Types: All | Chatbooks | Reports | Datasets | Drafts | Exports | Sort: Recent",
                    id="artifacts-mode-label",
                    classes="destination-section",
                )
            with Horizontal(
                id="artifacts-workbench", classes="ds-panel destination-workbench"
            ):
                with Vertical(
                    id="artifacts-list-pane", classes="destination-workbench-pane"
                ):
                    yield Static(
                        "Artifact List",
                        id="artifacts-list-title",
                        classes="destination-section artifacts-column-title",
                    )
                    if launch_kwargs is not None:
                        yield Static(
                            self._literal_text(f"> Chatbook: {launch_kwargs['title']}"),
                            id="artifacts-list-chatbooks",
                        )
                    else:
                        yield Static(
                            "> Chatbooks: none selected", id="artifacts-list-chatbooks"
                        )
                    yield Static(
                        "  Reports: none available", id="artifacts-list-reports"
                    )
                    yield Static(
                        "  Datasets: none available", id="artifacts-list-datasets"
                    )
                    yield Static("  Drafts: none available", id="artifacts-list-drafts")
                    yield Static(
                        "  Exports: none available", id="artifacts-list-exports"
                    )
                    yield Button(
                        "Open Chatbooks",
                        id="artifacts-open-chatbooks",
                        tooltip="Open portable Chatbook bundles.",
                    )
                    yield Button(
                        "Open Console",
                        id="artifacts-open-console",
                        tooltip="Open Console to create, review, or save Chatbook artifacts.",
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
                with Vertical(
                    id="artifacts-detail-pane", classes="destination-workbench-pane"
                ):
                    yield Static(
                        "Artifact Preview",
                        id="artifacts-preview-title",
                        classes="destination-section artifacts-column-title",
                    )
                    if not self._chatbook_context_loaded:
                        yield Static(
                            "Loading latest local Chatbook artifact...",
                            id="artifacts-loading-state",
                        )
                    elif launch_kwargs is not None:
                        payload = launch_kwargs.get("payload") or {}
                        description = str(payload.get("description") or "").strip()
                        content_preview = str(
                            payload.get("content_preview") or ""
                        ).strip()
                        yield Static(
                            self._literal_text(f"Title: {launch_kwargs['title']}"),
                            id="artifacts-detail-ready",
                        )
                        yield Static(
                            self._literal_text(
                                description
                                or "Summary: Console-saved Chatbook artifact."
                            ),
                            id="artifacts-detail-summary",
                        )
                        yield Static(
                            self._literal_text(
                                f"Transcript preview: {content_preview or 'No preview text available.'}"
                            ),
                            id="artifacts-detail-preview",
                        )
                    else:
                        yield Static(
                            "No artifact selected. Create a Chatbook in Console, import an artifact, "
                            "or use Library sources to generate outputs.",
                            id="artifacts-detail-empty",
                        )
                with Vertical(
                    id="artifacts-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                ):
                    yield Static(
                        "Provenance",
                        id="artifacts-provenance-title",
                        classes="destination-section artifacts-column-title",
                    )
                    if launch_kwargs is not None:
                        title = str(launch_kwargs["title"])
                        payload = launch_kwargs.get("payload") or {}
                        launch_scope = (
                            "requested"
                            if self._chatbook_context_requested
                            else "latest"
                        )
                        description = str(payload.get("description") or "").strip()
                        provenance = self._console_saved_artifact_provenance(payload)
                        content_preview = str(
                            payload.get("content_preview") or ""
                        ).strip()
                        yield Static("Created: Console", classes="destination-section")
                        yield Static(
                            Text.from_markup(
                                f"Open Console for {launch_scope} Chatbook artifact: "
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
                                Text.from_markup(
                                    f"Preview: {escape_markup(content_preview)}"
                                ),
                                id="artifacts-chatbook-content-preview",
                            )
                        yield Button(
                            Text.from_markup(f"Open {escape_markup(title)} in Console"),
                            id="artifacts-use-in-console",
                            tooltip=f"Open the {launch_scope} local Chatbook artifact in Console.",
                        )
                    else:
                        yield Static("Created: none", classes="destination-section")
                        recovery_state = self._blocked_chatbook_recovery_state
                        yield Static(
                            recovery_state.visible_copy,
                            id=recovery_state.stable_selector,
                        )
                        yield Button(
                            "Open selected in Console",
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

    @on(Button.Pressed, "#artifacts-open-console")
    def open_console(self) -> None:
        self.post_message(NavigateToScreen("chat"))

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
