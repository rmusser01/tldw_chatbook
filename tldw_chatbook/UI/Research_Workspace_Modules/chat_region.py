"""Research's mounted conversation, composer, and explicit consent controls."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    ListView,
    Select,
    Static,
    TextArea,
)

from ...Research_Workspace.chat_models import (
    ResearchChatMode,
    ResearchChatSessionRequest,
    ResearchChatTurnRequest,
    ResearchRetrievalSettings,
)
from ...Research_Workspace.contracts import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    WorkspaceDataSource,
)
from ...Research_Workspace.egress_consent import ResearchEgressConsentService
from ...Research_Workspace.layout_state import ResearchPanePreferences
from ...Research_Workspace.overlay_store import (
    OverlayConflictError,
    ResearchPresentationOverlay,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog
from .chat_message import ResearchChatMessage, ResearchChatMessageEditor
from .citation_detail import ResearchCitationDetail


class ResearchChatRegion(Vertical):
    """Publish conversation results only to their still-current workspace."""

    can_focus = True

    class OverlayUpdated(Message):
        def __init__(self, overlay: ResearchPresentationOverlay) -> None:
            super().__init__()
            self.overlay = overlay

    class CaptureRequested(Message):
        def __init__(self, ref: QualifiedWorkspaceRef, row: Mapping[str, Any]) -> None:
            super().__init__()
            self.ref = ref
            self.row = row

    class MessageSelectionChanged(Message):
        def __init__(self, ref: QualifiedWorkspaceRef | None, available: bool) -> None:
            super().__init__()
            self.ref = ref
            self.available = available

    def __init__(self, *, controller=None, overlay_store=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.controller = controller
        self.overlay_store = overlay_store
        self.ref = None
        self._generation = 0
        self._session = ""
        self._sessions = ()
        self._ready = False
        self._busy = False
        self._cancel = None
        self._retry = None
        self._resume = None
        self._saved_draft = ""
        self._overlay_lock = asyncio.Lock()
        self._undo = None
        self._speech_destination = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Grounded Chat", id="research-chat-heading", classes="research-pane-title"
        )
        yield Static(
            "Select a Research workspace to start a conversation.",
            id="research-chat-status",
            markup=False,
        )
        yield Static(
            "Processing route appears before sending.",
            id="research-chat-route",
            markup=False,
        )
        with (
            VerticalScroll(id="research-chat-scroll"),
            Vertical(id="research-chat-controls"),
        ):
            with Horizontal(classes="research-chat-row"):
                yield Select([], prompt="Conversation", id="research-chat-session")
                yield Button("New", id="research-chat-new")
            with Horizontal(classes="research-chat-row"):
                yield Select(
                    [
                        ("Sources off", "general"),
                        ("Grounded", "rag"),
                        ("Auto retrieval", "auto"),
                    ],
                    value="auto",
                    allow_blank=False,
                    id="research-chat-mode",
                )
                yield Input(placeholder="Provider", id="research-chat-provider")
                yield Input(placeholder="Model", id="research-chat-model")
            yield Static(
                "Sources: checking selection", id="research-chat-sources", markup=False
            )
            with Collapsible(title="Retrieval options", collapsed=True):
                with Horizontal(classes="research-chat-row"):
                    yield Select(
                        [("FTS", "fts"), ("Vector", "semantic"), ("Hybrid", "hybrid")],
                        value="fts",
                        allow_blank=False,
                        id="research-chat-retrieval",
                    )
                    yield Input(
                        "5",
                        placeholder="Results",
                        type="integer",
                        id="research-chat-top-k",
                        tooltip="Maximum results between 1 and 50",
                    )
                    yield Input(
                        placeholder="Score threshold",
                        type="number",
                        id="research-chat-threshold",
                    )
                with Horizontal(classes="research-chat-row"):
                    yield Checkbox("Rerank", id="research-chat-rerank")
                    yield Checkbox(
                        "Citations", value=True, id="research-chat-include-citations"
                    )
            yield ListView(id="research-chat-transcript")
            yield Static("", id="research-chat-stream", markup=False)
            with Horizontal(
                id="research-chat-message-actions", classes="research-chat-row"
            ):
                yield Button("Copy", id="research-chat-copy", disabled=True)
                yield Button("Citations", id="research-chat-citations", disabled=True)
                yield Button("Save to Note", id="research-chat-note", disabled=True)
                yield Button("Read aloud", id="research-chat-speak", compact=True)
            with Horizontal(
                id="research-chat-revision-actions", classes="research-chat-row"
            ):
                yield Button("Edit", id="research-chat-edit", compact=True)
                yield Button("Delete", id="research-chat-delete", compact=True)
                yield Button("Undo", id="research-chat-undo", compact=True)
                yield Button("Branch", id="research-chat-branch", compact=True)
                yield Button("Regenerate", id="research-chat-regenerate", compact=True)
            yield TextArea(id="research-chat-composer", disabled=True)
            with Horizontal(classes="research-chat-row"):
                yield Button(
                    "Send", id="research-chat-send", variant="primary", disabled=True
                )
                yield Button("Stop", id="research-chat-stop", disabled=True)
                yield Button("Clear draft", id="research-chat-clear", disabled=True)
                yield Button("Retry", id="research-chat-retry", disabled=True)

    def on_mount(self) -> None:
        self._sync_controls()

    def set_workspace(self, ref: QualifiedWorkspaceRef | None) -> None:
        if ref == self.ref:
            return
        self.ref = ref
        self._generation += 1
        self._ready = False
        self._session = ""
        self._sessions = ()
        self._retry = self._resume = None
        self._undo = None
        self._speech_destination = None
        self._saved_draft = ""
        self._composer.load_text("")
        self.query_one("#research-chat-route", Static).update(
            "Processing route appears before sending."
        )
        self.post_message(self.MessageSelectionChanged(ref, False))
        self._sync_sessions()
        self._sync_controls()
        self._status(
            "Loading workspace conversation…"
            if ref
            else "Select a Research workspace to start a conversation."
        )
        self.run_worker(
            self._load_workspace(ref, self._generation),
            group="research-chat-view",
            exclusive=True,
            exit_on_error=False,
        )

    @property
    def _composer(self) -> TextArea:
        return self.query_one("#research-chat-composer", TextArea)

    @property
    def _selected_row(self) -> Mapping[str, Any] | None:
        item = self.query_one(ListView).highlighted_child
        return item.row if isinstance(item, ResearchChatMessage) else None

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def has_dirty_draft(self) -> bool:
        return self.ref is not None and self._composer.text != self._saved_draft

    def _port(self, ref: QualifiedWorkspaceRef):
        return (
            self.controller.port_for_data_source(ref.data_source)
            if self.controller
            else None
        )

    def _current(self, ref, generation: int) -> bool:
        return self.is_mounted and self.ref == ref and self._generation == generation

    def _status(self, text: str) -> None:
        self.query_one("#research-chat-status", Static).update(text)

    def _stream(self, text: str) -> None:
        stream = self.query_one("#research-chat-stream", Static)
        stream.update(text)
        stream.display = bool(text)

    async def _load_workspace(self, ref, generation: int) -> None:
        await self.query_one(ListView).clear()
        self._stream("")
        port = self._port(ref) if ref else None
        if port is None or not hasattr(port, "chat_capability"):
            if ref and self._current(ref, generation):
                self._status("Chat is unavailable for this workspace.")
            return
        try:
            capability = await port.chat_capability(ref)
            if not self._current(ref, generation):
                return
            if not capability.available:
                self._status(
                    f"{capability.user_message} {capability.recovery_action}".strip()
                )
                return
            overlay = (
                await asyncio.to_thread(self.overlay_store.load, ref)
                if self.overlay_store
                else None
            )
            if not self._current(ref, generation):
                return
            if overlay:
                self._saved_draft = overlay.chat_draft
                self._composer.load_text(overlay.chat_draft)
            page = await port.list_chat_sessions(ref)
            if not self._current(ref, generation):
                return
            self._sessions = tuple(row for row in page.items if row.ref == ref)
            ids = {row.conversation_id for row in self._sessions}
            saved_id = overlay.chat_conversation_id if overlay else ""
            self._session = (
                saved_id
                if saved_id in ids
                else (self._sessions[0].conversation_id if self._sessions else "")
            )
            self._sync_sessions()
            await self._load_transcript(ref, generation, self._session, overlay=overlay)
            if self._current(ref, generation):
                self._ready = True
                self._status("Write a question, or select a saved conversation.")
                self._sync_controls()
                self.run_worker(
                    self._resolve_speech(ref, generation),
                    group="research-chat-speech",
                    exclusive=True,
                    exit_on_error=False,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._current(ref, generation):
                self._show_error(exc, "Chat could not load for this workspace.")

    def _sync_sessions(self) -> None:
        chooser = self.query_one("#research-chat-session", Select)
        with chooser.prevent(Select.Changed):
            chooser.set_options(
                [(row.title, row.conversation_id) for row in self._sessions]
            )
            chooser.value = self._session or Select.NULL

    def _sync_controls(self) -> None:
        controls = self.query_one("#research-chat-controls")
        controls.display = self._ready or bool(self._saved_draft)
        self.query_one("#research-chat-route").display = self._ready
        disabled = not self._ready or self._busy
        for name in (
            "session",
            "new",
            "mode",
            "provider",
            "model",
            "composer",
            "send",
            "clear",
            "retrieval",
            "top-k",
            "threshold",
            "rerank",
            "include-citations",
        ):
            self.query_one(f"#research-chat-{name}").disabled = disabled
        self.query_one("#research-chat-stop", Button).disabled = (
            not self._busy or self._cancel is None
        )
        retry = self.query_one("#research-chat-retry", Button)
        retry.disabled = disabled or (self._retry is None and self._resume is None)
        retry.label = "Regenerate" if self._resume is not None else "Retry"
        for name in ("copy", "note", "citations", "speak"):
            self.query_one(f"#research-chat-{name}", Button).disabled = (
                disabled or self._selected_row is None
            )
        self.query_one("#research-chat-speak", Button).display = (
            self._speech_destination is not None
        )
        port = self._port(self.ref) if self.ref else None
        row = self._selected_row or {}
        actions = {
            "edit": "edit_chat_message",
            "delete": "delete_chat_message",
            "undo": "restore_chat_messages",
            "branch": "branch_chat_session",
            "regenerate": "branch_chat_session",
        }
        visible = False
        for name, method in actions.items():
            button = self.query_one(f"#research-chat-{name}", Button)
            button.display = callable(getattr(port, method, None))
            visible |= button.display
            available = (
                self._undo and self._undo[:2] == (self.ref, self._session)
                if name == "undo"
                else row
            )
            if self.ref and self.ref.data_source is WorkspaceDataSource.SERVER:
                available = type(row.get("version")) is int and (
                    name != "edit" or row.get("sender") == "user"
                )
            button.disabled = disabled or not available
        self.query_one("#research-chat-revision-actions").display = visible

    def sync_sources(self, page, readiness) -> None:
        source_count = len(page.desired_source_ids)
        ready_ids = {
            item.source_id
            for item in readiness
            if not item.stale and (item.fts_ready or item.vector_ready)
        }
        visible = tuple(source for source in page.items if source.selected)
        count = sum(source.source_id in ready_ids for source in visible)
        detail = f"Sources: {source_count} selected"
        if len(visible) == source_count:
            detail += f" · {count} ready"
        titles = ", ".join(source.title for source in visible[:3])
        self.query_one("#research-chat-sources", Static).update(
            f"{detail} · {titles}" if titles else detail
        )

    async def _load_transcript(
        self, ref, generation: int, conversation_id: str, *, overlay=None
    ) -> None:
        if not conversation_id:
            rows = ()
        else:
            rows = await self._port(ref).load_chat_messages(ref, conversation_id)
        if not self._current(ref, generation) or conversation_id != self._session:
            return
        transcript = self.query_one(ListView)
        await transcript.clear()
        if rows:
            await transcript.extend(ResearchChatMessage(row) for row in rows)
            transcript.index = len(rows) - 1
        self._resume = None
        if overlay and rows:
            tail = rows[-1]
            for receipt in reversed(overlay.chat_append_receipts):
                if (
                    receipt.conversation_id == conversation_id
                    and receipt.stage == "user_committed"
                    and receipt.user_message_id == tail["id"]
                ):
                    self._resume = tail
                    break

    def _store_state(self, ref, old_draft, new_draft, conversation, fingerprint):
        store = self.overlay_store
        if store is None:
            raise ValueError("Device workspace storage is unavailable.")
        for _ in range(3):
            current = store.load(ref)
            current_draft = current.chat_draft if current else ""
            if current_draft not in {old_draft, new_draft}:
                raise OverlayConflictError("The chat draft changed in another session.")
            approvals = current.chat_egress_consents if current else ()
            if fingerprint is not None:
                approvals = ResearchEgressConsentService().approve(
                    fingerprint, approvals
                )
            if (
                current
                and current.chat_conversation_id == conversation
                and current_draft == new_draft
                and approvals == current.chat_egress_consents
            ):
                return current
            try:
                return store.save(
                    ref,
                    current.preferences if current else ResearchPanePreferences(),
                    expected_revision=current.revision if current else 0,
                    chat_draft=new_draft,
                    chat_conversation_id=conversation,
                    chat_egress_consents=approvals,
                )
            except OverlayConflictError:
                continue
        raise OverlayConflictError(
            "The chat draft could not be saved while the workspace was changing."
        )

    async def _save_draft(self, *, fingerprint=None) -> ResearchPresentationOverlay:
        ref, generation, conversation = self.ref, self._generation, self._session
        draft = self._composer.text
        if ref is None:
            raise ValueError("Select a Research workspace first.")
        async with self._overlay_lock:
            if not self._current(ref, generation):
                raise ValueError(
                    "The Research workspace changed before the draft could be saved."
                )
            saved = await asyncio.to_thread(
                self._store_state,
                ref,
                self._saved_draft,
                draft,
                conversation,
                fingerprint,
            )
            if self._current(ref, generation):
                self._saved_draft = saved.chat_draft
                self.post_message(self.OverlayUpdated(saved))
            return saved

    async def flush_pending_chat(self) -> bool:
        """Keep a draft in its visible workspace and veto during an active send."""
        if self._busy:
            message = (
                "Stop the active answer before leaving this workspace."
                if self._cancel is not None
                else "Wait for the Chat action to finish before leaving this workspace."
            )
            self._status(message)
            return False
        if self.has_dirty_draft:
            try:
                await self._save_draft()
            except (OSError, ValueError, OverlayConflictError) as exc:
                self._show_error(exc, "The Chat draft could not be saved.")
                return False
        return True

    async def add_output_to_draft(self, ref, title: str, content: str) -> None:
        from ...Research_Workspace.chat_models import validate_chat_draft

        if self.ref != ref or self._busy or not self._ready:
            raise ValueError("Open the workspace conversation before discussing the output.")
        prompt = f'Discuss the Studio output "{title}".\n\n{content}'
        draft = validate_chat_draft(f"{self._composer.text.rstrip()}\n\n{prompt}".lstrip())
        self._composer.load_text(draft)
        self.query_one("#research-chat-mode", Select).value = ResearchChatMode.RAG
        self._retry = None
        await self._save_draft()
        self._status("Studio output added to the draft.")

    @on(Select.Changed, "#research-chat-session")
    def session_changed(self, event: Select.Changed) -> None:
        if self._busy or event.value is Select.NULL or event.value == self._session:
            return
        self._session = str(event.value)
        self._retry = self._resume = None
        self._run(self._switch_session())

    async def _switch_session(self) -> None:
        saved = await self._save_draft()
        await self._load_transcript(
            self.ref, self._generation, self._session, overlay=saved
        )
        self._status("Conversation loaded.")

    @on(Button.Pressed, "#research-chat-new")
    def new_session(self) -> None:
        self._run(self._new_session())

    async def _new_session(self) -> None:
        await self._create_session()
        await self._switch_session()
        self._composer.focus()

    async def _create_session(self) -> None:
        session = await self._port(self.ref).create_chat_session(
            self.ref, ResearchChatSessionRequest()
        )
        if session.ref != self.ref:
            raise ValueError("New conversation belongs to a different workspace.")
        self._session = session.conversation_id
        self._sessions = (session, *self._sessions)
        self._sync_sessions()

    @on(Button.Pressed, "#research-chat-clear")
    def clear(self) -> None:
        self._composer.load_text("")
        self._retry = None
        self._run(self._save_draft())

    @on(Button.Pressed, "#research-chat-send")
    def send(self) -> None:
        self._retry = self._resume = None
        self._run(self._send(), answer=True)

    @on(Button.Pressed, "#research-chat-stop")
    def stop(self) -> None:
        if self._cancel is not None:
            self._cancel.set()

    @on(Button.Pressed, "#research-chat-retry")
    def retry(self) -> None:
        if self._resume is not None:
            self._run(self._send(retry_user=self._resume), answer=True)
        elif self._retry is not None and self._retry.user_text == self._composer.text:
            self._run(self._send(previous=self._retry), answer=True)
        else:
            self.send()

    def _run(self, operation, *, answer: bool = False) -> None:
        if self._busy or not self._ready:
            operation.close()
            return
        self._busy = True
        self._cancel = asyncio.Event() if answer else None
        self._sync_controls()
        self.run_worker(
            self._run_guarded(operation),
            group="research-chat-action",
            exit_on_error=False,
        )

    async def _run_guarded(self, operation) -> None:
        try:
            await operation
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._show_error(
                exc, "Chat could not complete the action. Your draft remains available."
            )
        finally:
            self._cancel = None
            self._busy = False
            if self.is_mounted:
                self._stream("")
                self._sync_controls()

    def _show_error(self, exc, fallback: str) -> None:
        logger.warning(
            "Research Chat action failed (error_type={})", type(exc).__name__
        )
        if isinstance(exc, CapabilityUnavailableError):
            capability = exc.capability
            self._status(
                f"{capability.user_message} {capability.recovery_action}".strip()
            )
        elif type(exc) in {ValueError, OverlayConflictError}:
            self._status(str(exc))
        else:
            self._status(fallback)

    def _retrieval_options(self) -> ResearchRetrievalSettings:
        threshold = self.query_one("#research-chat-threshold", Input).value
        return ResearchRetrievalSettings(
            mode=self.query_one("#research-chat-retrieval", Select).value,
            top_k=int(self.query_one("#research-chat-top-k", Input).value),
            threshold=float(threshold) if threshold else None,
            rerank=self.query_one("#research-chat-rerank", Checkbox).value,
            citations=self.query_one(
                "#research-chat-include-citations", Checkbox
            ).value,
        )

    async def _send(self, *, previous=None, retry_user=None) -> None:
        ref, generation = self.ref, self._generation
        capture = self.controller.capture_request()
        port = self._port(ref)
        question = retry_user["content"] if retry_user else self._composer.text
        if not question.strip():
            self._status("Write a question before sending.")
            return
        if previous is None:
            mode = ResearchChatMode(self.query_one("#research-chat-mode", Select).value)
            settings = self._retrieval_options()
            sources = (
                ()
                if mode is ResearchChatMode.GENERAL
                else await port.capture_chat_sources(ref)
            )
            if mode is ResearchChatMode.RAG and not any(
                source.ready_modes for source in sources
            ):
                raise ValueError("Grounded requires a selected ready source")
            route = await port.resolve_chat_route(
                ref,
                self.query_one("#research-chat-provider", Input).value.strip(),
                self.query_one("#research-chat-model", Input).value.strip(),
            )
            if self._cancel.is_set() or not self.controller.is_current_request(capture):
                return
            if not self._session:
                await self._create_session()
            request = ResearchChatTurnRequest(
                ref,
                self._session,
                question,
                mode,
                sources,
                settings,
                route,
                capture.context_revision,
                retry_user_message_id=retry_user["id"] if retry_user else "",
            )
        else:
            request = previous
            route = request.processing_route
        self.query_one("#research-chat-route", Static).update(
            f"Processing: {route.endpoint_class.replace('_', ' ') or route.processor} · "
            + f"{route.provider or 'server default'} / {route.model}"
        )
        saved = await self._save_draft()
        preflight = ResearchEgressConsentService().preflight(
            request, saved.chat_egress_consents
        )
        if preflight is not None:
            source_types = ", ".join(preflight.source_types)
            prompt = ConfirmationDialog(
                title="Send Local source evidence?",
                message=f"Send {preflight.source_body_mode} from {preflight.source_count} selected source(s) ({source_types}) to {preflight.provider}. Destination is on the {preflight.endpoint_class.replace('_', ' ')}. Redaction policy is {preflight.redaction_policy}.",
                confirm_label="Continue",
                cancel_label="Cancel",
            )
            if await self.app.push_screen_wait(prompt) is not True:
                self._status("Send cancelled. Draft retained.")
                return
            await self._save_draft(fingerprint=preflight)
        if self._cancel.is_set() or not self.controller.is_current_request(capture):
            return
        self._retry = request
        chunks = []
        self._status("Waiting for the answer…")

        def show(chunk: str) -> None:
            if self._current(ref, generation) and self.controller.is_current_request(
                capture
            ):
                chunks.append(chunk)
                self._stream("".join(chunks))

        result = await port.send_chat_turn(request, show, self._cancel)
        if not self._current(ref, generation) or not self.controller.is_current_request(
            capture
        ):
            return
        if result.assistant_message_id:
            if retry_user is None and self._composer.text == question:
                self._composer.load_text("")
            await self._save_draft()
            self._retry = self._resume = None
            await self._load_transcript(ref, generation, self._session)
        status = {
            "complete": "Answer saved",
            "cancelled": "Stopped",
            "failed": "Answer failed. Retry or edit the draft.",
        }[result.status]
        if result.status == "failed" and result.assistant_message_id:
            status = "Answer interrupted. Partial answer saved."
        disclosure = result.retrieval_disclosure
        if result.effective_retrieval is not None:
            methods = {"fts": "FTS", "semantic": "Vector", "hybrid": "Hybrid"}
            effective = methods[result.effective_retrieval.value]
            requested = methods[result.requested_retrieval.value]
            disclosure += f" · {effective}"
            if effective != requested:
                disclosure += f" ({requested} requested)"
        self._status(f"{status} · {disclosure}")

    @on(ListView.Highlighted, "#research-chat-transcript")
    def message_highlighted(self) -> None:
        self._sync_controls()
        self.post_message(
            self.MessageSelectionChanged(self.ref, self._selected_row is not None)
        )

    @on(Button.Pressed, "#research-chat-copy")
    def copy_message(self) -> None:
        if self._selected_row:
            self.app.copy_to_clipboard(str(self._selected_row.get("content") or ""))
            self._status("Message copied.")

    @on(Button.Pressed, "#research-chat-citations")
    def show_citations(self) -> None:
        if self._selected_row:
            self.app.push_screen(ResearchCitationDetail(self._selected_row))

    @on(Button.Pressed, "#research-chat-note")
    def capture(self) -> None:
        if self.ref and self._selected_row and self._ready and not self._busy:
            self.post_message(self.CaptureRequested(self.ref, self._selected_row))

    async def _resolve_speech(self, ref, generation) -> None:
        if ref.data_source is not WorkspaceDataSource.LOCAL:
            return
        handler = await self.app._ensure_tts_handler()
        if handler is None:
            return
        destination = await handler.resolve_console_speech_destination(None, None)
        if self._current(ref, generation):
            self._speech_destination = destination
            self._sync_controls()

    @on(Button.Pressed, "#research-chat-speak")
    def speak_message(self) -> None:
        if self._selected_row and self._speech_destination is not None:
            self._run(self._speak(self._selected_row))

    async def _speak(self, row) -> None:
        ref, generation, session = self.ref, self._generation, self._session
        handler = await self.app._ensure_tts_handler()
        if handler is None:
            return
        destination = await handler.resolve_console_speech_destination(None, None)
        if destination is None:
            self._speech_destination = None
            self._status("Speech is unavailable.")
            return
        if (
            await self.app.push_screen_wait(
                ConfirmationDialog(
                    title="Read message aloud?",
                    message=f"Process this message with {destination.provider_label} at {destination.sanitized_destination}.",
                    confirm_label="Speak",
                )
            )
            is not True
        ):
            return
        rows = await self._port(ref).load_chat_messages(ref, session)
        if not any(
            item["id"] == row["id"] and item["version"] == row["version"]
            for item in rows
        ):
            raise ValueError(
                "The selected message changed. Reload the conversation and try again"
            )
        played = await handler.speak_guarded_utterance(
            row["content"],
            assistant_kind=None,
            character_ref=None,
            expected_destination_fingerprint=destination.fingerprint,
            validator=lambda: (
                self._current(ref, generation) and self._session == session
            ),
        )
        self._status("Reading complete." if played else "Speech could not be played.")

    @on(Button.Pressed, "#research-chat-edit")
    def edit_selected(self) -> None:
        if self._selected_row is not None:
            self._run(self._edit_message(self._selected_row))

    async def _edit_message(self, row) -> None:
        ref, generation, session = self.ref, self._generation, self._session
        body = await self.app.push_screen_wait(
            ResearchChatMessageEditor(row["content"])
        )
        if body is not None and self._current(ref, generation):
            await self._port(ref).edit_chat_message(ref, session, row, body)
            await self._load_transcript(ref, generation, session)
            self._status("Message saved.")

    @on(Button.Pressed, "#research-chat-delete")
    def delete_selected(self) -> None:
        if self._selected_row is not None:
            self._run(self._delete_message(self._selected_row))

    async def _delete_message(self, row) -> None:
        ref, generation, session = self.ref, self._generation, self._session
        port = self._port(ref)
        restorable = callable(getattr(port, "restore_chat_messages", None))
        if await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Delete message?",
                message="Delete this message and its replies?"
                if restorable
                else "Delete this message from the server conversation?",
                confirm_label="Delete",
            )
        ) is not True or not self._current(ref, generation):
            return
        tombstones = await port.delete_chat_message(ref, session, row)
        self._undo = (ref, session, tombstones) if tombstones and restorable else None
        await self._load_transcript(ref, generation, session)
        self._status("Message deleted.")

    @on(Button.Pressed, "#research-chat-undo")
    def undo_delete(self) -> None:
        if self._undo and self._undo[:2] == (self.ref, self._session):
            self._run(self._restore_messages(*self._undo))

    async def _restore_messages(self, ref, session, tombstones) -> None:
        generation = self._generation
        await self._port(ref).restore_chat_messages(ref, session, tombstones)
        self._undo = None
        await self._load_transcript(ref, generation, session)
        self._status("Messages restored.")

    @on(Button.Pressed, "#research-chat-branch")
    @on(Button.Pressed, "#research-chat-regenerate")
    def branch_selected(self, event: Button.Pressed) -> None:
        if self._selected_row is not None:
            regenerate = event.button.id == "research-chat-regenerate"
            self._run(self._branch(self._selected_row, regenerate), answer=regenerate)

    async def _branch(self, row, regenerate: bool) -> None:
        ref, generation = self.ref, self._generation
        port = self._port(ref)
        branch = await port.branch_chat_session(
            ref, self._session, row, at_question=regenerate
        )
        if not self._current(ref, generation):
            return
        if branch.ref != ref:
            raise ValueError("New conversation belongs to a different workspace.")
        self._session = branch.conversation_id
        self._sessions = (branch, *self._sessions)
        self._retry = self._resume = None
        self._sync_sessions()
        await self._switch_session()
        if regenerate:
            rows = await port.load_chat_messages(ref, self._session)
            if rows and not self._cancel.is_set():
                self._resume = rows[-1]
                await self._send(retry_user=self._resume)
        else:
            self._status("Branch created.")
