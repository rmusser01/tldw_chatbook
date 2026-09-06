"""Artifacts destination shell for generated outputs and Chatbooks."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static
from textual.worker import Worker, WorkerState

from ...Chat.answer_citations import summarize_citation_artifact_metadata
from ...Constants import (
    TAB_WATCHLISTS_COLLECTIONS,
    WATCHLISTS_NAV_CONTEXT_BACKEND,
    WATCHLISTS_NAV_CONTEXT_BRIEFING_ID,
    WATCHLISTS_NAV_CONTEXT_SECTION,
)
from ...Subscriptions.briefing_export import (
    BriefingExportError,
    briefing_markdown_document,
    default_briefing_filename,
)
from ...Subscriptions.briefing_keep import KeepRefused, keep_briefing
from ...Subscriptions.daily_reports_view import format_report_timestamp
from ...Subscriptions.briefing_service import (
    STATUS_COMPLETE,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_GENERATING,
)
from ...Subscriptions.html_text import strip_control_characters
from ...Third_Party.textual_fspicker import FileSave
from ...TTS.audio_player import play_audio_file
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Utils.path_validation import validate_path_simple
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
#: How many Daily Report rows the Artifacts list paints (Qodo #6: the one
#: named constant driving the slice, the overflow check, and the "+ N more"
#: count -- previously three separate `5` literals that could drift).
REPORT_DISPLAY_LIMIT = 5
#: Shared tooltip for the Daily Report demo control -- rendered both as the
#: empty-state "Create Your First Daily Report" CTA and, after a failed brief
#: (TASK-31801), as the "Run the Daily Report demo again" retry affordance
#: that the failure toast ("...then run the demo again") points users to.
DAILY_REPORT_DEMO_TOOLTIP = (
    "Seeds a 'Daily Brief' watchlist from live RSS, drafts a text brief with "
    "your configured LLM provider, and records audio when a TTS voice profile "
    "exists. Uses live sources and your provider's API quota."
)
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
        self._daily_reports: list[dict[str, Any]] = []
        self._daily_reports_generation = 0
        self._daily_reports_worker: Worker[Any] | None = None
        # TASK-21514: the previewed Daily Report (a full `briefings` row via
        # `SubscriptionsDB.get_briefing`, with `watchlist_name`/`kept` merged
        # in from the list row), or None when no report is previewed.
        self._previewed_report: dict[str, Any] | None = None
        self._report_preview_generation = 0
        self._report_preview_worker: Worker[Any] | None = None
        self._keep_in_flight = False
        self._report_export_in_flight = False

    def on_mount(self) -> None:
        # No super().on_mount(): the dispatcher already invokes
        # BaseAppScreen.on_mount separately for this Mount event.
        self._chatbook_unmounted = False
        self._start_chatbook_refresh()
        self._start_daily_reports_refresh()

    def on_screen_resume(self) -> None:
        """Refresh daily reports, and one-shot Chatbook handoffs, on resume."""
        self._start_daily_reports_refresh()
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
        # Qodo #15: the daily-reports refresh and the report preview are
        # this screen's workers too, and they used to outlive it -- unmount
        # tore down only the Chatbook worker, so a navigation mid-refresh
        # left the apply callbacks firing into an unmounted screen. Same
        # teardown shape: cancel the worker AND bump the generation, so an
        # in-flight `call_from_thread` apply is invalidated even if the
        # cancellation lands too late to stop the worker body.
        self._daily_reports_generation += 1
        worker = self._daily_reports_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        self._report_preview_generation += 1
        worker = self._report_preview_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        # No super().on_unmount(): the dispatcher already invokes
        # BaseAppScreen.on_unmount separately for this Unmount event (TASK-31418).

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

    def _start_daily_reports_refresh(self) -> None:
        """Re-read recent briefings off the UI thread, then repaint."""
        self._daily_reports_generation += 1
        self.refresh(recompose=True)
        self._daily_reports_worker = self._refresh_daily_reports(
            self._daily_reports_generation
        )

    @work(exclusive=True, thread=True, group="artifacts-daily-reports")
    def _refresh_daily_reports(self, generation: int) -> None:
        from ...Subscriptions.daily_reports_view import list_recent_reports

        db = getattr(self.app_instance, "subscriptions_db", None)
        reports: list[dict[str, Any]] = []
        if db is not None:
            try:
                reports = list_recent_reports(db, limit=20)
            except Exception:  # noqa: BLE001 - an Artifacts refresh must never crash the app
                reports = []
        # Kept state lives in ChaChaNotes (kept_briefings), not in
        # SubscriptionsDB, so the badge needs a cross-DB lookup. Qodo #14:
        # kept-ness is resolved PER DISPLAYED ROW via the indexed
        # `get_kept_briefing_by_source` SELECT (≤20 single-row lookups) --
        # the previous `list_kept_briefings(limit=200)` id-set silently
        # dropped the badge from any keep older than the newest 200 rows. A
        # missing or failing handle degrades to "no badges", never to a
        # crash.
        chacha_db = getattr(self.app_instance, "chachanotes_db", None)
        for report in reports:
            kept = None
            if chacha_db is not None:
                try:
                    kept = chacha_db.get_kept_briefing_by_source(report["id"])
                except Exception:  # noqa: BLE001 - badge lookup must never break the refresh
                    kept = None
            report["kept"] = kept is not None
        self.app.call_from_thread(self._apply_daily_reports, generation, reports)

    def _apply_daily_reports(
        self, generation: int, reports: list[dict[str, Any]]
    ) -> None:
        if not self.is_attached or generation != self._daily_reports_generation:
            # Qodo #15: superseded by a newer refresh, or the screen went
            # away -- an unmounted screen must not recompose.
            return
        self._daily_reports = reports
        # Keep the preview's kept flag honest across refreshes (a keep that
        # just landed flips the badge; a preview opened before it must not
        # lag behind the row beside it).
        previewed = self._previewed_report
        if previewed is not None:
            summary = next(
                (r for r in reports if r.get("id") == previewed.get("id")), None
            )
            if summary is not None:
                previewed["kept"] = bool(summary.get("kept"))
        self.refresh(recompose=True)

    # --- TASK-21514: previewing one Daily Report in the detail pane ---------

    @property
    def _latest_report_failed(self) -> bool:
        """True when the most recent Daily Report ended in failure.

        Drives the retry-CTA gate (TASK-31801). The demo failure toast points
        at "run the demo again", so the retry affordance must track the
        RELEVANT state -- the newest report -- not "any report ever
        completed" (Qodo #4 on PR #2460: an older success followed by a fresh
        failure kept the toast but lost the CTA). `_daily_reports` is ordered
        newest-first (`list_recent_briefings` ORDER BY created_at DESC, id
        DESC), so index 0 is the latest run: the CTA shows exactly when that
        run is `failed`, and a later success (or an empty/generating run)
        removes it.
        """
        if not self._daily_reports:
            return False
        newest = self._daily_reports[0]
        return str(newest.get("status") or "").strip().lower() == STATUS_FAILED

    @property
    def _previewed_report_complete(self) -> bool:
        """True when the previewed report's status is `complete`."""
        report = self._previewed_report
        return report is not None and (
            str(report.get("status") or "").strip().lower() == STATUS_COMPLETE
        )

    @property
    def _keep_target_ready(self) -> bool:
        """Keep needs a complete preview AND a ChaChaNotes handle."""
        return self._previewed_report_complete and (
            getattr(self.app_instance, "chachanotes_db", None) is not None
        )

    def _notify(
        self, message: str, severity: str = "information", *, markup: bool = False
    ) -> None:
        """Notify through the app instance, degrading when it has none.

        Same idiom as the Watchlists screen's `_notify_watchlists`: the app
        instance is a stub in several harnesses. `markup` defaults to False
        here because several of this screen's action toasts embed text the
        app did not author (a KeepRefused message, a path-validation error).
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity=severity, markup=markup)

    def _start_report_preview(self, briefing_id: int) -> None:
        """Reset the preview slot, then fetch the full row off-thread."""
        self._report_preview_generation += 1
        generation = self._report_preview_generation
        self._previewed_report = None
        self.refresh(recompose=True)
        try:
            self._report_preview_worker = self._load_report_preview(
                generation, briefing_id
            )
        except Exception as exc:  # noqa: BLE001 - preview must never crash the screen
            logger.warning(
                "Report preview could not start (exception_category={}).",
                type(exc).__name__,
            )

    @work(exclusive=True, thread=True, group="artifacts-report-preview")
    def _load_report_preview(self, generation: int, briefing_id: int) -> None:
        db = getattr(self.app_instance, "subscriptions_db", None)
        row: dict[str, Any] | None = None
        if db is not None:
            try:
                row = db.get_briefing(briefing_id)
            except Exception:  # noqa: BLE001 - a missing row degrades to a notify
                row = None
        if row is not None:
            # `get_briefing` returns the bare `briefings` row; the list row
            # already resolved the display name and kept flag for this id,
            # so merge them in for the preview header and the export path.
            summary = next(
                (r for r in self._daily_reports if r.get("id") == briefing_id), None
            )
            if summary is not None:
                row["watchlist_name"] = summary.get("watchlist_name")
                row["kept"] = bool(summary.get("kept"))
        self.app.call_from_thread(
            self._apply_report_preview, generation, briefing_id, row
        )

    def _apply_report_preview(
        self, generation: int, briefing_id: int, row: dict[str, Any] | None
    ) -> None:
        if (
            not self.is_attached
            or generation != self._report_preview_generation
        ):
            # Qodo #15: a newer preview (or a clear, or the screen going
            # away) superseded this one.
            return
        if row is None:
            self._previewed_report = None
            self._notify(
                "This report no longer exists; refresh or reopen Artifacts.",
                severity="warning",
            )
        else:
            self._previewed_report = row
        self.refresh(recompose=True)

    def _report_preview_renderable(self) -> Group:
        """The previewed report's detail-pane body.

        Follows the Watchlists artifacts pane's `_detail_renderable`
        convention for untrusted LLM bodies: a literal header `Text` (never
        markup-parsed) grouped with a `rich.markdown.Markdown` body rendered
        with `hyperlinks=False` -- never Textual's Markdown widget. Every
        status gets a body of its own; none renders as a blank pane.
        """
        row = self._previewed_report or {}
        status = str(row.get("status") or "").strip().lower()
        header = Text()
        watchlist_name = str(
            row.get("watchlist_name") or f"Watchlist {row.get('watchlist_id')}"
        )
        header.append(strip_control_characters(watchlist_name), style="bold")
        header.append(" · ")
        header.append(
            strip_control_characters(format_report_timestamp(row.get("created_at")))
        )
        header.append(" · ")
        header.append(strip_control_characters(status or "unknown status"))
        item_count = row.get("item_count") or 0
        header.append(f" · {item_count} item{'s' if item_count != 1 else ''}")
        if row.get("kept"):
            header.append(" · kept")
        header.append("\n")

        body = str(row.get("body_markdown") or "").strip()
        if status == STATUS_COMPLETE:
            if not body:
                return Group(header, Text("This briefing recorded no body."))
            return Group(header, Markdown(body, hyperlinks=False))
        if status == STATUS_FAILED:
            return Group(
                header,
                Text(
                    str(
                        row.get("error")
                        or "This report failed without a recorded error."
                    )
                ),
            )
        if status == STATUS_EMPTY:
            return Group(header, Text("This briefing's window held no stories."))
        if status == STATUS_GENERATING:
            return Group(header, Text("This briefing is still being written."))
        return Group(header, Text(f"Unrecognised report status: {status or '—'}"))

    @work(exclusive=True, group="artifacts-refresh-chatbook-context", thread=True)
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
                    if self._daily_reports:
                        for report in self._daily_reports[:REPORT_DISPLAY_LIMIT]:
                            label = f"> Report: {report['label']}"
                            if report.get("kept"):
                                label += " · kept"
                            yield Static(
                                self._literal_text(label),
                                id=f"artifacts-report-row-{report['id']}",
                            )
                            if report.get("has_audio"):
                                yield Button(
                                    "Play",
                                    id=f"artifacts-report-play-{report['id']}",
                                    tooltip="Play this report's audio brief.",
                                )
                            yield Button(
                                "View",
                                id=f"artifacts-report-view-{report['id']}",
                                tooltip=(
                                    "Preview this report's briefing body in "
                                    "the detail pane."
                                ),
                            )
                            yield Button(
                                "Open",
                                id=f"artifacts-report-open-{report['id']}",
                                tooltip=(
                                    "Open this report in its watchlist's "
                                    "artifacts pane in Watchlists."
                                ),
                            )
                        if len(self._daily_reports) > REPORT_DISPLAY_LIMIT:
                            yield Static(
                                self._literal_text(
                                    f"  + {len(self._daily_reports) - REPORT_DISPLAY_LIMIT} more in Watchlists"
                                ),
                                id="artifacts-reports-more",
                            )
                        yield Button(
                            "Open Watchlists",
                            id="artifacts-open-watchlists",
                            tooltip="Read, play, keep, or export daily reports.",
                        )
                        keep_tooltip = (
                            "Keep this report in your library so it survives "
                            "watchlist deletion."
                        )
                        if not self._keep_target_ready:
                            keep_tooltip += " View a completed report first."
                        yield Button(
                            "Keep",
                            id="artifacts-report-keep",
                            disabled=not self._keep_target_ready,
                            tooltip=keep_tooltip,
                        )
                        export_tooltip = (
                            "Save the previewed report's briefing as a "
                            "Markdown file."
                        )
                        if not self._previewed_report_complete:
                            export_tooltip += " View a completed report first."
                        yield Button(
                            "Export",
                            id="artifacts-report-export",
                            disabled=not self._previewed_report_complete,
                            tooltip=export_tooltip,
                        )
                        # TASK-31801: a failed run leaves a report row (so the
                        # empty-state branch below is gone), yet the failure
                        # toast tells the user to "run the demo again". Keep
                        # that retry affordance reachable while the NEWEST run
                        # is failed (Qodo #4: not "any report ever completed").
                        if self._latest_report_failed:
                            yield Button(
                                "Run the Daily Report demo again",
                                id="artifacts-daily-report-demo",
                                tooltip=DAILY_REPORT_DEMO_TOOLTIP,
                            )
                    else:
                        yield Static(
                            "  Reports: none yet", id="artifacts-list-reports"
                        )
                        yield Button(
                            "Create Your First Daily Report",
                            id="artifacts-daily-report-demo",
                            tooltip=DAILY_REPORT_DEMO_TOOLTIP,
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
                    # TASK-31802: the Import control is permanently disabled,
                    # so state its precondition inline rather than leaving the
                    # empty-state copy to advertise an action the user cannot
                    # take (the copy below no longer says "import an artifact").
                    yield Static(
                        "Import Artifact is not yet available in this shell.",
                        id="artifacts-import-note",
                        classes="destination-purpose",
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
                    if self._previewed_report is not None:
                        # A previewed Daily Report takes precedence over the
                        # Chatbook content while set; the clear button
                        # restores the previous pane content.
                        yield Static(
                            self._report_preview_renderable(),
                            id="artifacts-report-preview",
                        )
                        yield Button(
                            "Clear preview",
                            id="artifacts-report-preview-clear",
                            tooltip=(
                                "Return the preview pane to the latest "
                                "Chatbook artifact."
                            ),
                        )
                    elif not self._chatbook_context_loaded:
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
                            "No artifact selected. Create a Chatbook in Console, "
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

    @on(Button.Pressed, "#artifacts-open-watchlists")
    def open_watchlists(self) -> None:
        """Navigate to the Watchlists destination screen."""
        self.post_message(NavigateToScreen("watchlists_collections"))

    @on(Button.Pressed, "#artifacts-report-preview-clear")
    def clear_report_preview(self, event: Button.Pressed) -> None:
        """Clear the previewed report and restore the previous pane content.

        Args:
            event: The clear-button press.
        """
        event.stop()
        self._report_preview_generation += 1  # invalidate any in-flight load
        self._previewed_report = None
        self.refresh(recompose=True)

    # --- TASK-21514: keeping a previewed report into ChaChaNotes ------------
    #
    # `briefing_keep.keep_briefing` is the one writer for
    # `kept_briefings`/`kept_scripts`; this handler is mount wiring around
    # it, copied from the Watchlists screen's own Keep trio (guard claimed
    # in the sync handler BEFORE `run_worker`; `asyncio.to_thread` for the
    # blocking cross-DB call; `KeepRefused` surfaces verbatim as a warning;
    # success distinguishes created vs re-keep).

    @on(Button.Pressed, "#artifacts-report-keep")
    def keep_previewed_report(self, event: Button.Pressed) -> None:
        """Keep the previewed report into the library (ChaChaNotes).

        Args:
            event: The Keep-button press.
        """
        event.stop()
        report = self._previewed_report
        subs_db = getattr(self.app_instance, "subscriptions_db", None)
        chacha_db = getattr(self.app_instance, "chachanotes_db", None)
        if report is None or not self._previewed_report_complete:
            self._notify("View a completed report to keep it.", severity="warning")
            return
        if subs_db is None or chacha_db is None:
            self._notify(
                "Keeping is unavailable in this runtime: no library database.",
                severity="warning",
            )
            return
        if self._keep_in_flight:
            self._notify(
                "A keep is already in progress. Nothing else was started.",
                severity="warning",
            )
            return
        self._keep_in_flight = True
        self.run_worker(
            self._keep_report(subs_db, chacha_db, report["id"]),
            group="artifacts-report-keep",
        )

    async def _keep_report(
        self, subs_db: Any, chacha_db: Any, briefing_id: int
    ) -> None:
        """Worker body: keep, toast honestly, then refresh the badges."""
        try:
            try:
                result = await asyncio.to_thread(
                    keep_briefing, subs_db, chacha_db, briefing_id, origin="manual"
                )
            except KeepRefused as exc:
                self._notify(str(exc), severity="warning")
                return
            except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
                logger.warning(
                    "Keep failed for report {} (exception_category={}).",
                    briefing_id,
                    type(exc).__name__,
                )
                self._notify(
                    "Could not keep this report: the database could not be "
                    "reached. Nothing was recorded.",
                    severity="error",
                )
                return
            scripts_added = result["scripts_added"]
            if result["created"]:
                message = f"Kept with {scripts_added} scripts"
            else:
                message = f"Already kept — added {scripts_added} new scripts"
            self._notify(message)
            previewed = self._previewed_report
            if previewed is not None and previewed.get("id") == briefing_id:
                previewed["kept"] = True
        finally:
            self._keep_in_flight = False
            if self.is_attached:
                # Refresh so the row badge flips in place beside the toast.
                self._start_daily_reports_refresh()

    # --- TASK-21514: exporting the previewed report as Markdown -------------

    @on(Button.Pressed, "#artifacts-report-export")
    def export_previewed_report(self, event: Button.Pressed) -> None:
        """Export the previewed report as a Markdown file.

        Args:
            event: The Export-button press.
        """
        event.stop()
        report = self._previewed_report
        if report is None or not self._previewed_report_complete:
            self._notify("View a completed report to export it.", severity="warning")
            return
        if self._report_export_in_flight:
            self._notify(
                "An export is already in progress. Nothing else was started.",
                severity="warning",
            )
            return
        self._report_export_in_flight = True
        self.run_worker(
            self._push_report_export_dialog(dict(report)),
            group="artifacts-report-export",
        )

    async def _push_report_export_dialog(self, report: dict[str, Any]) -> None:
        """Push the vendored `FileSave` picker seeded with a safe filename.

        Copy of the Watchlists screen's `_push_export_briefing_dialog`
        shape: the `pushed` sentinel distinguishes "never reached the
        callback" from "did", so the in-flight guard is re-armed on every
        path that did not hand control to the dialog's callback.
        """
        pushed = False
        try:
            watchlist_name = str(
                report.get("watchlist_name")
                or f"Watchlist {report.get('watchlist_id')}"
            )
            enriched = {**report, "watchlist_name": watchlist_name}
            default_filename = default_briefing_filename(
                enriched, watchlist_name=watchlist_name
            )
            await self.app.push_screen(
                FileSave(
                    location=str(Path.home()),
                    title="Export Daily Report as Markdown",
                    default_file=default_filename,
                ),
                callback=lambda path: self._write_report_export_file(
                    path, enriched
                ),
            )
            pushed = True
        except Exception as exc:  # noqa: BLE001 - a worker crash exits the app
            logger.warning(
                "Failed to open the report export dialog "
                "(exception_category={}).",
                type(exc).__name__,
            )
            self._notify("Could not open the export dialog.", severity="error")
        finally:
            if not pushed:
                self._report_export_in_flight = False

    async def _write_report_export_file(
        self, selected_path: Path | None, report: Mapping[str, Any]
    ) -> None:
        """Validate the chosen path, build the document, write it off-loop.

        Copy of the Watchlists screen's `_write_briefing_export_file`
        shape: validate (`validate_path_simple`, never the private-path
        helpers -- the destination is the user's own folder), build
        (`briefing_markdown_document`, which refuses a blank body), write in
        `asyncio.to_thread`, honest toasts each way, guard cleared in
        `finally` on every exit path.
        """
        try:
            if not selected_path:
                self._notify("Report export cancelled.")
                return
            try:
                validated_path = validate_path_simple(
                    Path(selected_path), require_exists=False
                )
            except ValueError as exc:
                logger.warning(
                    "Rejected report export path (exception_category={}).",
                    type(exc).__name__,
                )
                self._notify(f"Rejected export path: {exc}", severity="warning")
                return
            try:
                document = briefing_markdown_document(report)
            except BriefingExportError as exc:
                self._notify(str(exc), severity="warning")
                return
            try:
                await asyncio.to_thread(
                    validated_path.write_text, document, encoding="utf-8"
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - encode errors are plausible here
                logger.warning(
                    "Report export write failed (exception_category={}).",
                    type(exc).__name__,
                )
                self._notify(
                    f"Error exporting report: {type(exc).__name__}",
                    severity="error",
                )
                return
            self._notify(
                f"Report exported successfully to {validated_path.name}"
            )
        finally:
            self._report_export_in_flight = False

    @on(Button.Pressed, "#artifacts-daily-report-demo")
    def start_daily_report_demo(self) -> None:
        """Start the wired Daily Report demo from the empty-state CTA."""
        service = getattr(self.app_instance, "daily_report_demo_service", None)
        if service is None:
            self.app_instance.notify(
                "The Daily Report demo is unavailable in this runtime.",
                severity="warning",
            )
            return
        self.run_worker(
            self._run_daily_report_demo(service),
            exclusive=True,
            group="artifacts-daily-report-demo",
        )

    async def _run_daily_report_demo(self, service: Any) -> None:
        """Start the app-owned demo task, then refresh on start AND on finish.

        Qodo #10: the demo itself runs as a SERVICE-owned task
        (`run_demo_detached`), never inside this screen worker -- Textual
        cancels a widget's workers on unmount, which used to kill the
        orchestration mid-flight after its persistent seed state had already
        committed. The worker only starts the task and refreshes; stage and
        completion notifications arrive through the dispatch service.

        Qodo #5 (PR #2460 review): the detached task runs for minutes, so the
        immediate refresh below only shows the freshly-seeded `generating`
        row. Without a completion refresh, a retry that SUCCEEDS while
        Artifacts stays open left the stale `failed` row and its retry CTA on
        screen until an unrelated refresh or a screen resume fired. Attaching
        a done-callback to the returned task closes that: it re-reads the rows
        on the event loop when the demo terminates (guarded by `is_attached`).
        """
        task = None
        try:
            task = service.run_demo_detached()
        except Exception:  # noqa: BLE001 - a worker crash exits the app
            logger.warning("Daily report demo failed to start")
            self.app_instance.notify(
                "The Daily Report demo failed unexpectedly.",
                severity="error",
            )
        finally:
            # `run_demo_detached` returns None when a demo is already running;
            # only a real task carries a completion to refresh on.
            if task is not None:
                task.add_done_callback(self._on_demo_task_done)
            if self.is_attached:
                self._start_daily_reports_refresh()

    def _on_demo_task_done(self, _task: Any) -> None:
        """Refresh the Reports rows when the detached demo task terminates.

        Runs on the event loop (asyncio done-callback), the same thread
        Textual drives, so the refresh is dispatched directly. A screen that
        unmounted mid-demo must not recompose, hence the `is_attached` guard.
        """
        if not self.is_attached:
            return
        self._start_daily_reports_refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dynamic-id dispatch for per-report Play/View/Open buttons.

        `@on` selectors cannot express the `artifacts-report-*-{id}` family,
        so prefix-match here; unrelated buttons fall through untouched.

        Args:
            event: The button press; stopped only on a recognized prefix,
                so unrelated buttons keep bubbling untouched.
        """
        button_id = event.button.id or ""
        if button_id.startswith("artifacts-report-open-"):
            # Deep-link to the owning watchlist's artifacts pane in
            # Watchlists. Post-only: the Watchlists screen's
            # `apply_navigation_context` resolves the owning watchlist and
            # selects the briefing from these context keys (the same
            # receipt the Console's watchlists operation cards post).
            event.stop()
            try:
                briefing_id = int(button_id.rsplit("-", 1)[-1])
            except ValueError:
                return
            self.post_message(
                NavigateToScreen(
                    TAB_WATCHLISTS_COLLECTIONS,
                    screen_context={
                        WATCHLISTS_NAV_CONTEXT_SECTION: "artifacts",
                        WATCHLISTS_NAV_CONTEXT_BACKEND: "local",
                        WATCHLISTS_NAV_CONTEXT_BRIEFING_ID: (
                            f"local:briefing:{briefing_id}"
                        ),
                    },
                )
            )
            return
        if button_id.startswith("artifacts-report-view-"):
            event.stop()
            try:
                briefing_id = int(button_id.rsplit("-", 1)[-1])
            except ValueError:
                return
            self._start_report_preview(briefing_id)
            return
        if not button_id.startswith("artifacts-report-play-"):
            return
        event.stop()
        try:
            briefing_id = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        report = next(
            (r for r in self._daily_reports if r.get("id") == briefing_id), None
        )
        if report is None or not report.get("has_audio"):
            return
        # Qodo #1: the centralized path validator is the authority here, not
        # a boolean `.exists()` plus a rebuilt Path -- its normalized result
        # is what reaches the player, and its `ValueError` (a missing file,
        # a dangerous pattern) is the existing "no longer exists" warning.
        try:
            path = validate_path_simple(
                Path(str(report["audio_file_path"])), require_exists=True
            )
        except ValueError:
            self.app_instance.notify(
                "This audio file no longer exists on disk.", severity="warning"
            )
            return
        play_audio_file(path)
