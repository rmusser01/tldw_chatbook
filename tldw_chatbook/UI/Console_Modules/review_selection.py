"""Console review, selection-feedback, annotation, and trajectory policy.

TASK-3070.13 moves seven policy methods out of the screen class while the three
Textual event/action boundaries remain small screen delegates and six
presentation/DOM methods remain on the screen. Services keep persistence,
Git, note, and annotation authority; this owner only sequences them through
explicit late-bound callables.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from tldw_chatbook.Utils.input_validation import escape_markup

from tldw_chatbook.Chat.citation_trace_repository import ActiveCitationTraceState
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trajectory import TrajectorySnapshot, derive_trajectory
from tldw_chatbook.Widgets.Console.console_selection import SELECTION_QUOTE_CAP


_FEEDBACK_MESSAGE_HEADERS = {
    "request-changes": "[Request changes]",
    "lgm": "[LGTM]",
    "comment": "[Comment]",
}


class _SelectionFeedbackRequest(BaseModel):
    """Validated boundary values for one selection-feedback request."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: str
    quote: str = Field(min_length=1, max_length=SELECTION_QUOTE_CAP)
    anchor_message_id: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("action")
    @classmethod
    def _validate_action(cls, value: str) -> str:
        if value not in _FEEDBACK_MESSAGE_HEADERS:
            raise ValueError("unsupported feedback action")
        return value

    @field_validator("quote")
    @classmethod
    def _validate_quote(cls, value: str) -> str:
        from tldw_chatbook.Utils.input_validation import validate_text_input

        if not value.strip() or not validate_text_input(
            value, max_length=SELECTION_QUOTE_CAP, allow_html=True
        ):
            raise ValueError("invalid feedback quote")
        return value

    @field_validator("anchor_message_id")
    @classmethod
    def _validate_anchor_message_id(cls, value: str | None) -> str | None:
        from tldw_chatbook.Utils.input_validation import validate_text_input

        if value is not None and not validate_text_input(value, max_length=255):
            raise ValueError("invalid feedback anchor")
        return value


class _SelectionNoteRequest(BaseModel):
    """Strict selection capture, independent of the prompt quote limit."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    quote: str = Field(min_length=1)
    source_session_id: str | None = Field(default=None, min_length=1, max_length=255)
    open_note: bool = False

    @field_validator("quote")
    @classmethod
    def _validate_quote(cls, value: str) -> str:
        from ...Utils.input_validation import validate_text_input

        if not value.strip() or not validate_text_input(
            value,
            max_length=len(value),
            allow_html=True,
        ):
            raise ValueError("invalid selected text")
        return value


@dataclass(frozen=True, slots=True)
class ConsoleTrajectoryLaunch:
    """Presentation data for one lazily opened trajectory view."""

    snapshot: TrajectorySnapshot
    screen_title: str
    conversation_id: str
    revision_provider: Callable[[], int]
    snapshot_builder: Callable[[], TrajectorySnapshot]
    capture_policy_bindings: Any | None


def _build_trajectory_snapshot(
    store: Any,
    conversation_id: str,
    *,
    agent_runs_db: Any | None = None,
) -> TrajectorySnapshot:
    """Assemble the ``derive_trajectory`` inputs for one conversation."""
    messages: list[Any] = []
    traj_rows: list[Any] = []
    variant_sets: list[Any] = []
    compaction_records: list[Any] = []
    agent_runs: list[Any] = []
    agent_steps: list[Any] = []
    retrieval_runs: list[Any] = []
    diagnostic_events: list[Any] = []
    active_leaf: str | None = None

    def capture_failed(
        source: str, error: Exception, *, message_id: str | None = None
    ) -> None:
        logger.opt(exception=error).error(
            "Trace source read failed: source={} conversation_id={}",
            source,
            conversation_id,
        )
        diagnostic_events.append(
            {
                "event_id": (
                    f"capture-failed:{source}:{conversation_id}"
                    f"{f':{message_id}' if message_id else ''}"
                ),
                "conversation_id": conversation_id,
                "message_id": message_id,
                "event_kind": "capture_failed",
                "status": "capture_failed",
                "summary": f"{source} capture failed",
                "field_states": {
                    "source": "capture_failed",
                    **({"message_id": "observed"} if message_id else {}),
                },
                "sensitivity": "diagnostic",
            }
        )

    persistence = getattr(store, "persistence", None)
    db = getattr(persistence, "db", None)
    if db is not None:
        try:
            messages = list(
                db.get_messages_for_conversation(
                    conversation_id,
                    limit=1_000_000,
                    include_image_data=False,
                )
            )
        except Exception as error:  # noqa: BLE001 - launch degrades, never fails
            capture_failed("messages", error)
            messages = []
        try:
            traj_rows = list(db.get_trajectory_rows(conversation_id))
        except Exception as error:  # noqa: BLE001
            capture_failed("trajectory", error)
            traj_rows = []
        try:
            active_leaf = db.get_conversation_active_leaf(conversation_id)
        except Exception as error:  # noqa: BLE001
            capture_failed("active_leaf", error)
            active_leaf = None
    usage_by_id: dict[str, ProviderUsage] = {}
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        usage = ProviderUsage.from_json(message.get("usage_json"))
        if usage is not None:
            usage_by_id[str(message.get("id"))] = usage
    try:
        variant_sets = list(store.variant_sets_for_conversation(conversation_id))
    except Exception as error:  # noqa: BLE001
        capture_failed("variants", error)
        variant_sets = []
    context_repository = getattr(persistence, "context_repository", None)
    if context_repository is not None:
        try:
            offset = 0
            while True:
                page = list(
                    context_repository.list_auxiliary_attempts(
                        conversation_id, limit=500, offset=offset
                    )
                )
                compaction_records.extend(
                    {**record, "trace_lifecycle": True}
                    if isinstance(record, Mapping)
                    else record
                    for record in page
                )
                if len(page) < 500:
                    break
                offset += len(page)
        except Exception as error:  # noqa: BLE001
            capture_failed("context", error)
    turn_by_message: dict[str, str] = {}
    for trajectory_row in traj_rows:
        if isinstance(trajectory_row, Mapping):
            row_message_id = trajectory_row.get("message_id")
            row_turn_id = trajectory_row.get("turn_id")
        else:
            row_message_id = getattr(trajectory_row, "message_id", None)
            row_turn_id = getattr(trajectory_row, "turn_id", None)
        if row_message_id and row_turn_id:
            turn_by_message[str(row_message_id)] = str(row_turn_id)
    if agent_runs_db is not None:
        try:
            raw_runs = (
                raw_run
                for agent_kind in ("primary", "subagent")
                for raw_run in agent_runs_db.list_runs(
                    conversation_id,
                    agent_kind=agent_kind,
                )
            )
            for raw_run in raw_runs:
                try:
                    run = dict(raw_run) if isinstance(raw_run, Mapping) else {}
                    run_id = str(run.get("id") or "")
                    if not run_id:
                        continue
                    assistant_message_id = str(run.get("assistant_message_id") or "")
                    if assistant_message_id in turn_by_message:
                        run["turn_id"] = turn_by_message[assistant_message_id]
                    steps = list(run.get("steps", ()) or ())
                    converted_steps = [
                        {
                            **step,
                            "run_id": run_id,
                            "conversation_id": conversation_id,
                            "turn_id": run.get("turn_id"),
                        }
                        for step in steps
                        if isinstance(step, Mapping)
                    ]
                    agent_runs.append(run)
                    agent_steps.extend(converted_steps)
                except Exception as error:  # noqa: BLE001
                    capture_failed("agent", error)
        except Exception as error:  # noqa: BLE001
            capture_failed("agent", error)
    citation_repository = getattr(persistence, "citation_repository", None)
    if citation_repository is not None:
        assistant_ids = [
            str(message.get("id") or "")
            for message in messages
            if isinstance(message, Mapping)
            and str(message.get("sender") or "").lower() == "assistant"
            and message.get("id")
        ]
        try:
            candidates = citation_repository.active_owner_candidate_message_ids(
                assistant_ids
            )
        except Exception as error:  # noqa: BLE001
            capture_failed("retrieval_candidates", error)
            candidates = set()
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            message_id = str(message.get("id") or "")
            if (
                not message_id
                or str(message.get("sender") or "").lower() != "assistant"
                or message_id not in candidates
            ):
                continue
            try:
                result = citation_repository.get_active_trace_for_current_message(
                    message_id,
                    str(message.get("content") or ""),
                )
                if (
                    result.state is not ActiveCitationTraceState.ACTIVE
                    or result.summary is None
                    or not citation_repository.verify_active_trace_result(result)
                ):
                    continue
                for run in result.summary.trace.evidence_runs:
                    row = run.model_dump(mode="python")
                    retrieval_runs.append(
                        {
                            **row,
                            "conversation_id": conversation_id,
                            "message_id": message_id,
                            "turn_id": turn_by_message.get(message_id),
                            "field_states": {"payload": "omitted"},
                            "sensitivity": "retrieval_metadata",
                            "trace_lifecycle": True,
                        }
                    )
            except Exception as error:  # noqa: BLE001
                capture_failed("retrieval", error, message_id=message_id)
    return derive_trajectory(
        messages,
        usage_by_id,
        traj_rows,
        variant_sets,
        compaction_records,
        active_leaf_message_id=active_leaf,
        agent_runs=agent_runs,
        agent_steps=agent_steps,
        retrieval_runs=retrieval_runs,
        diagnostic_events=diagnostic_events,
    )


class ConsoleReviewSelectionController:
    """Own Console review/selection policy without DOM or sibling authority."""

    def __init__(
        self,
        *,
        store_accessor: Callable[[], Any],
        agent_conversation_id_accessor: Callable[[], str | None],
        change_review_provider_accessor: Callable[[str], Any | None],
        run_active_accessor: Callable[[], bool],
        run_active_for_root: Callable[[str], bool],
        workspace_roots_accessor: Callable[[], tuple[str, ...] | None],
        agent_runs_db_accessor: Callable[[], Any | None],
        capture_policy_bindings_accessor: Callable[[str, str], Any | None],
        native_messages_accessor: Callable[[], list[Any]],
        run_worker: Callable[..., Any],
        show_feedback_comment: Callable[[str, str], Awaitable[str | None]],
        dispatch_prompt: Callable[[str], Awaitable[Any]],
        marshal_to_ui: Callable[..., None],
        present_trajectory: Callable[..., None],
        notify: Callable[..., None],
        copy_to_clipboard: Callable[[str], None] | None = None,
        save_selection_note: Callable[[str, str], Awaitable[str | None]] | None = None,
        open_selection_note: Callable[[str], bool] | None = None,
        offer_selection_note_open: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the controller from explicit, late-bound collaborators.

        Args:
            store_accessor: Return the active Console chat store.
            agent_conversation_id_accessor: Return the active run conversation ID.
            change_review_provider_accessor: Build review operations for a conversation.
            run_active_accessor: Report whether any Console run is active.
            run_active_for_root: Report whether a workspace root has an active run.
            workspace_roots_accessor: Return the active turn's workspace roots.
            agent_runs_db_accessor: Return the optional agent-run database.
            capture_policy_bindings_accessor: Resolve trajectory capture bindings.
            native_messages_accessor: Return current native Console messages.
            run_worker: Schedule a Textual worker.
            show_feedback_comment: Present the feedback-comment modal.
            dispatch_prompt: Dispatch composed feedback through the prompt queue.
            marshal_to_ui: Marshal a callback onto the UI thread.
            present_trajectory: Present a completed trajectory launch.
            notify: Show a user-facing Console notification.
            copy_to_clipboard: Send the completed trace on the UI thread.
            save_selection_note: Persist the full selection through Notes authority.
            open_selection_note: Hand off the exact saved note to Library.
            offer_selection_note_open: Offer retry of a failed handoff without saving again.
        """
        self._store_accessor = store_accessor
        self._agent_conversation_id_accessor = agent_conversation_id_accessor
        self._change_review_provider_accessor = change_review_provider_accessor
        self._run_active_accessor = run_active_accessor
        self._run_active_for_root = run_active_for_root
        self._workspace_roots_accessor = workspace_roots_accessor
        self._agent_runs_db_accessor = agent_runs_db_accessor
        self._capture_policy_bindings_accessor = capture_policy_bindings_accessor
        self._native_messages_accessor = native_messages_accessor
        self._run_worker = run_worker
        self._show_feedback_comment = show_feedback_comment
        self._dispatch_prompt = dispatch_prompt
        self._marshal_to_ui = marshal_to_ui
        self._present_trajectory = present_trajectory
        self._notify = notify
        self._copy_to_clipboard = copy_to_clipboard
        self._trace_copy_serial = 0
        self._save_selection_note = save_selection_note
        self._open_selection_note = open_selection_note
        self._offer_selection_note_open = offer_selection_note_open
        self._selection_note_inflight: set[str] = set()
        self.annotation_loaded_conversation: str | None = None
        self.annotation_previews: dict[str, tuple[str, ...]] = {}
        self.selection_feedback_inflight = False

    def request_selection_action(self, request: Any) -> None:
        """Adapt a selection request's value payload to its policy entry point.

        Args:
            request: Trace-copy or note-creation message, without DOM authority.
        """
        from ...Widgets.Console.console_selection_menu import ConsoleTraceCopyRequested

        if isinstance(request, ConsoleTraceCopyRequested):
            self.request_trace_copy(
                request.session_id,
                message_id=request.message_id,
                source_message_ids=request.source_message_ids,
            )
        else:
            self.request_selection_note(
                request.quote,
                source_session_id=request.source_session_id,
                open_note=request.open_note,
            )

    def request_trace_copy(
        self,
        session_id: str,
        *,
        message_id: str | None = None,
        source_message_ids: tuple[str, ...] | None = None,
    ) -> None:
        """Capture one explicit source before scheduling trace formatting.

        Args:
            session_id: Source chat captured by the invoking menu.
            message_id: Response/owned activity ID, or None for the whole chat.
            source_message_ids: Menu's captured path; refuse a replacement branch.
        """
        self._trace_copy_serial += 1
        serial = self._trace_copy_serial
        if self._copy_to_clipboard is None:
            self._notify("Clipboard is unavailable.", severity="warning")
            return
        try:
            store = self._store_accessor()
            messages = tuple(store.read_only_messages_for_session(session_id))
            if source_message_ids is not None:
                from ...Chat.console_chat_models import ConsoleMessageRole

                current_ids = tuple(
                    message.id
                    for message in messages
                    if message.role
                    in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}
                )
                if current_ids != source_message_ids:
                    self._notify(
                        "The chat branch changed. Select text again to copy it.",
                        severity="warning",
                    )
                    return
            title = next(
                (
                    session.title
                    for session in store.sessions()
                    if session.id == session_id
                ),
                "Console",
            )
        except Exception:
            self._notify("The source chat is no longer available.", severity="warning")
            return
        self._run_worker(
            self._copy_trace_snapshot(messages, title, message_id, serial),
            group="console-trace-copy",
            exclusive=True,
            exit_on_error=False,
        )

    async def _copy_trace_snapshot(self, messages, title, message_id, serial) -> None:
        """Format detached data off-thread and reject superseded publication."""

        def render():
            from ...Chat.console_response_trace import (
                find_response_turn,
                render_chat_trace_markdown,
                render_response_trace_markdown,
            )

            if message_id is None:
                return render_chat_trace_markdown(messages, title=title)
            turn = find_response_turn(messages, message_id)
            return render_response_trace_markdown(turn) if turn is not None else None

        try:
            text = await asyncio.to_thread(render)
            if serial != self._trace_copy_serial:
                return
            if text is None:
                self._notify(
                    "There is no response or chat content to copy.", severity="warning"
                )
                return
            size_kib = max(1, round(len(text.encode("utf-8")) / 1024))
            self._copy_to_clipboard(text)
        except Exception as error:
            if serial == self._trace_copy_serial:
                logger.warning("Console trace copy failed: {}", type(error).__name__)
                self._notify(
                    "Could not send the trace to the clipboard.", severity="warning"
                )
            return
        if message_id is None:
            from ...Chat.console_chat_models import ConsoleMessageRole

            count = sum(
                message.role in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}
                for message in messages
            )
            label = f"{count} messages"
        else:
            label = "Response trace"
        self._notify(f"{label} sent to clipboard ({size_kib} KiB).")

    def _console_change_review_provider(self, conversation_id=None) -> Any | None:
        """Return the live provider recipe, degrading on missing collaborators."""
        try:
            conversation_id = conversation_id or self._agent_conversation_id_accessor()
            if not conversation_id:
                return None
            provider = self._change_review_provider_accessor(conversation_id)
        except Exception:  # noqa: BLE001 -- opener must degrade, not raise
            return None
        if provider is None:
            return None
        provider.run_active = self._run_active_accessor
        provider.run_active_for_root = self._run_active_for_root
        return provider

    def _console_change_review_workspace_roots(self) -> tuple[str, ...] | None:
        """Return the current turn workspace roots, or ``None`` on failure."""
        try:
            return self._workspace_roots_accessor()
        except Exception:  # noqa: BLE001 -- opener must degrade, not raise
            return None

    def _sync_console_annotation_discovery(self, store: Any) -> None:
        """Load persisted annotations when the active conversation changes."""
        session = getattr(store, "_sessions", {}).get(
            getattr(store, "active_session_id", None)
        )
        conversation_id = getattr(session, "persisted_conversation_id", None)
        if not conversation_id:
            if self.annotation_loaded_conversation is not None:
                self.annotation_loaded_conversation = None
                self.annotation_previews = {}
            return
        conversation_id = str(conversation_id)
        if conversation_id == self.annotation_loaded_conversation:
            return
        self.annotation_loaded_conversation = conversation_id
        self.annotation_previews = {}
        database = getattr(getattr(store, "persistence", None), "db", None)
        if database is None:
            return
        self._run_worker(
            self._load_console_annotation_previews(database, store, conversation_id),
            exclusive=True,
            group="console-annotation-previews",
            exit_on_error=False,
        )

    async def _load_console_annotation_previews(
        self, database: Any, store: Any, conversation_id: str
    ) -> None:
        """Read annotations off-thread and re-key them on the event loop."""
        try:
            from ...DB.base_db import run_owned_db_call

            rows = await run_owned_db_call(
                database, database.get_transcript_annotations, conversation_id
            )
        except Exception:
            logger.warning(
                f"Console annotations: load failed for {conversation_id!r}",
                exc_info=True,
            )
            return
        if self.annotation_loaded_conversation != conversation_id:
            return
        native_by_persisted = {
            message.persisted_message_id: message.id
            for message in self._native_messages_accessor()
            if message.persisted_message_id is not None
        }
        previews: dict[str, tuple[str, ...]] = {}
        for row in rows:
            native_id = native_by_persisted.get(row.get("message_id"))
            if native_id is None:
                continue
            previews[native_id] = previews.get(native_id, ()) + (row["comment"],)
        self.annotation_previews = previews

    def request_selection_note(
        self,
        quote: str,
        *,
        source_session_id: str | None = None,
        open_note: bool = False,
    ) -> None:
        """Admit one full selection against its captured source chat.

        Args:
            quote: Complete selected text, without the prompt quote cap.
            source_session_id: Original chat, or the current chat for legacy callers.
            open_note: Open the created note after a successful save.
        """
        try:
            request = _SelectionNoteRequest(
                quote=quote,
                source_session_id=source_session_id,
                open_note=open_note,
            )
        except ValidationError:
            return
        source = self._selection_note_source(request.source_session_id)
        if source is None:
            return
        session_id, title = source
        if session_id in self._selection_note_inflight:
            self._notify("A note is already being created for this chat.")
            return
        self._selection_note_inflight.add(session_id)
        self._run_worker(
            self._create_console_selection_note(
                request.quote,
                source_session_id=session_id,
                source_title=title,
                open_note=request.open_note,
            ),
            group=f"console-selection-note-{session_id}",
            exclusive=True,
            exit_on_error=False,
        )

    def _selection_note_source(self, session_id):
        from ...Chat.console_ephemeral import blocked_reason

        try:
            store = self._store_accessor()
            session_id = session_id or store.active_session_id
            reader = getattr(store, "sessions", None)
            sessions = (
                reader()
                if callable(reader)
                else getattr(store, "_sessions", {}).values()
            )
            session = next((item for item in sessions if item.id == session_id), None)
            if session is None:
                raise KeyError(session_id)
            reason = blocked_reason(
                "save-as-note", ephemeral=bool(getattr(session, "ephemeral", False))
            )
            if reason:
                self._notify(reason, severity="warning")
                return None
            return session_id, str(session.title or "Console")
        except Exception:
            self._notify(
                "Notes are unavailable (source chat or notes database unavailable).",
                severity="warning",
            )
            return None

    async def _create_console_selection_note(
        self,
        quote: str,
        *,
        source_session_id: str | None = None,
        source_title: str | None = None,
        open_note: bool = False,
    ) -> None:
        """Retain issued save ownership until it settles, then offer navigation."""
        if source_title is None:
            source = self._selection_note_source(source_session_id)
            if source is None:
                return
            source_session_id, source_title = source
        note_id = None
        try:
            if self._save_selection_note is None:
                self._notify(
                    "Notes are unavailable (no notes database service).",
                    severity="warning",
                )
                return
            request = _SelectionNoteRequest(quote=quote, open_note=open_note)
            first_line = request.quote.strip().splitlines()[0]
            title = first_line if len(first_line) <= 48 else first_line[:47] + "…"
            stamp = datetime.now().strftime("%Y-%m-%d")
            content = f"{request.quote}\n\n— Console selection, {source_title}, {stamp}"
            save = asyncio.create_task(self._save_selection_note(title, content))
            cancelled = False
            while not save.done():
                try:
                    await asyncio.shield(save)
                except asyncio.CancelledError:
                    if save.cancelled():
                        raise
                    cancelled = True
            note_id = save.result()
            if not note_id:
                self._notify("Could not create the note.", severity="warning")
                return
            self._notify(f"Note created: {escape_markup(title)}")
            if not open_note:
                return
            if (
                cancelled
                or self._store_accessor().active_session_id != source_session_id
            ):
                self._notify(
                    f"Note saved ({escape_markup(note_id)}). Open it in Library > Notes."
                )
                return
            try:
                opened = (
                    self._open_selection_note is not None
                    and self._open_selection_note(note_id)
                )
            except Exception:
                opened = False
            if not opened:
                self._notify(
                    f"Note saved ({escape_markup(note_id)}), but it could not be opened.",
                    severity="warning",
                )
                if self._offer_selection_note_open is not None:
                    await self._offer_selection_note_open(note_id)
        except Exception as error:
            logger.warning("Console selection note failed: {}", type(error).__name__)
            if note_id:
                self._notify(
                    f"Note saved ({escape_markup(note_id)}). Open it in Library > Notes.",
                    severity="warning",
                )
            else:
                self._notify("Could not create the note.", severity="warning")
        finally:
            self._selection_note_inflight.discard(source_session_id)

    def request_selection_feedback(
        self, action: str, quote: str, anchor_message_id: str | None
    ) -> None:
        """Validate and schedule one transcript-selection feedback flow.

        Args:
            action: Supported feedback action identifier.
            quote: Capped transcript selection included in the feedback.
            anchor_message_id: Optional native message ID for durable attribution.
        """
        if self.selection_feedback_inflight:
            return
        if isinstance(quote, str) and not quote.strip():
            return
        try:
            request = _SelectionFeedbackRequest(
                action=action,
                quote=quote,
                anchor_message_id=anchor_message_id,
            )
        except ValidationError:
            self._notify(
                "Selection feedback is invalid or too large.", severity="warning"
            )
            return
        self.selection_feedback_inflight = True
        self._run_worker(
            self._console_selection_feedback_flow(
                request.action,
                request.quote,
                request.anchor_message_id,
            ),
            group="console-selection-feedback",
        )

    def _record_console_feedback_event(
        self,
        store: Any | None,
        session_id: str | None,
        action: str,
        quote: str,
        comment: str,
        anchor_message_id: str | None,
    ) -> bool:
        """Write feedback persistence and report whether an annotation was made."""
        if store is None or not session_id or not anchor_message_id:
            return False
        try:
            store.record_feedback_event(
                session_id,
                anchor_message_id=anchor_message_id,
                action=action,
                quote=quote,
                comment=comment or None,
            )
            if action != "comment" or not comment:
                return False
            return bool(
                store.record_feedback_annotation(
                    session_id,
                    anchor_message_id=anchor_message_id,
                    quote=quote,
                    comment=comment,
                )
            )
        except Exception:
            logger.warning(
                "Console selection feedback: audit record failed for anchor "
                f"{anchor_message_id!r}; the feedback itself was dispatched.",
                exc_info=True,
            )
            return False

    async def _console_selection_feedback_flow(
        self, action: str, quote: str, anchor_message_id: str | None = None
    ) -> None:
        """Collect a comment, persist its audit, and dispatch exact feedback."""
        try:
            comment = await self._show_feedback_comment(action, quote)
            if comment is None:
                return
            lines = [_FEEDBACK_MESSAGE_HEADERS.get(action, "[Comment]")]
            lines.extend(
                f"> {line}" if line.strip() else ">" for line in quote.splitlines()
            )
            if comment:
                lines.append(comment)
            try:
                store = self._store_accessor()
                session_id = getattr(store, "active_session_id", None)
            except Exception:  # noqa: BLE001 -- audit loss cannot cost feedback
                store = None
                session_id = None
            annotation_created = await asyncio.to_thread(
                self._record_console_feedback_event,
                store,
                session_id,
                action,
                quote,
                comment,
                anchor_message_id,
            )
            if annotation_created and anchor_message_id:
                existing = self.annotation_previews.get(anchor_message_id, ())
                self.annotation_previews[anchor_message_id] = existing + (comment,)
            await self._dispatch_prompt("\n".join(lines))
        finally:
            self.selection_feedback_inflight = False

    def open_trajectory_view(self) -> None:
        """Build and present a trace for the active persisted conversation."""
        store = self._store_accessor()
        session = getattr(store, "_sessions", {}).get(
            getattr(store, "active_session_id", None)
        )
        conversation_id = getattr(session, "persisted_conversation_id", None)
        if not conversation_id:
            self._notify("The active conversation has no persisted trace yet.")
            return
        conv_id = str(conversation_id)
        target_session_id = str(session.id)
        capture_policy_bindings = self._capture_policy_bindings_accessor(
            target_session_id, conv_id
        )
        screen_title = str(getattr(session, "title", "") or "Console")
        agent_runs_db = self._agent_runs_db_accessor()

        def build() -> TrajectorySnapshot:
            return _build_trajectory_snapshot(
                store,
                conv_id,
                agent_runs_db=agent_runs_db,
            )

        def present(snapshot: TrajectorySnapshot) -> None:
            self._present_trajectory(
                ConsoleTrajectoryLaunch(
                    snapshot=snapshot,
                    screen_title=screen_title,
                    conversation_id=conv_id,
                    revision_provider=lambda: store.get_payload_revision(conv_id),
                    snapshot_builder=build,
                    capture_policy_bindings=capture_policy_bindings,
                )
            )

        def build_worker() -> None:
            self._marshal_to_ui(present, build())

        self._notify("Building trace…")
        self._run_worker(
            build_worker, thread=True, exclusive=True, group="trajectory-launch"
        )

    def open_change_review(
        self,
        run_id=None,
        *,
        push_screen,
        conversation_id=None,
        review_workspace_roots=None,
        initial_path=None,
        initial_snapshot_id=None,
        initial_current_mode=False,
    ):
        """Open the exact conversation and workspace through the shared review owner."""
        provider = self._console_change_review_provider(conversation_id=conversation_id)
        if provider is None:
            self._notify(
                "Change review needs git and a saved conversation.",
                severity="warning",
            )
            return
        from tldw_chatbook.UI.Screens.change_review_screen import (
            ChangeReviewScreen,
        )

        # TASK-16801 arc B (Task 9): the conversation's LIVE workspace
        # roots -- see `_console_change_review_workspace_roots`'s docstring
        # for why this matters (it is what makes `current` mode reachable
        # from a fresh conversation with no recorded turns).
        workspace_roots = (
            review_workspace_roots
            if conversation_id is not None
            else self._console_change_review_workspace_roots()
        )

        # initial_run_id/initial_path/initial_snapshot_id all ride the
        # constructor: a post-push select_turn/select_file call raced the
        # screen's own compose (NoMatches) -- caught by the opener wiring
        # test (TASK-1972 / TASK-18060 Task 3).
        push_screen(
            ChangeReviewScreen(
                provider,
                initial_run_id=run_id,
                initial_path=initial_path,
                initial_snapshot_id=initial_snapshot_id,
                workspace_roots=workspace_roots,
                initial_current_mode=initial_current_mode,
            )
        )


__all__ = [
    "ConsoleReviewSelectionController",
    "ConsoleTrajectoryLaunch",
    "_build_trajectory_snapshot",
]
