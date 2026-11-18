"""Contained Console transport and parent receipt custody.

This module does not select the process route on import. A channel receipt is
neither permission to perform an effect nor proof that its native owners ended.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Mapping

from tldw_chatbook.MCP.permission_process import read_message, write_message

from .console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
    decode_message,
    encode_message,
)


class ConsoleChannelLost(RuntimeError):
    """An outcome may remain unknown after the channel stops."""


class ConsoleReadinessExpired(ConsoleChannelLost):
    """The checked display aged out before the client could adopt it."""


class ConsoleRemoteFailure(RuntimeError):
    """A fixed backend operation refused at its reported original stage."""

    def __init__(self, code: str, stage: str) -> None:
        self.code = code
        self.stage = stage
        super().__init__(f"Console operation failed: {code} ({stage})")


@dataclass(slots=True)
class ConsoleProcessReceipt:
    """Retain each acknowledgement independently of an observing caller."""

    identity: WireIdentity
    operation: str
    received: asyncio.Future[None] | Future[None]
    applied: asyncio.Future[Mapping[str, Any]] | Future[Mapping[str, Any]]
    retired: asyncio.Future[bool] | Future[bool]
    potentially_committed: bool = False
    turn_id: str | None = None
    events: list[ConsoleWireMessage] = field(default_factory=list, repr=False)
    projection_message: ConsoleWireMessage | None = field(default=None, repr=False)
    changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


@dataclass(eq=False, slots=True)
class _ConsoleInputPress:
    view: Any
    visit: int
    key: tuple[str, str]
    entry: tuple
    inputs: Mapping[str, Any]
    source_revision: int
    task: asyncio.Task | None = None
    receipt: ConsoleProcessReceipt | None = None


class _ConsoleFrameInput:
    """Distinguish EOF before any byte of one original framed message."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.started = False

    def read(self, count: int) -> bytes:
        part = self.stream.read(count)
        if part == b"" and not self.started:
            raise EOFError("Console peer closed between frames")
        self.started = True
        return part


class _ConsoleFramedChannel:
    """Keep framing off the owner loop without waiting for admitted work."""

    def __init__(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        loop: asyncio.AbstractEventLoop | None,
        receive: Callable[[ConsoleWireMessage], None],
        failed: Callable[[BaseException], None],
    ) -> None:
        self.input = input_stream
        self.output = output_stream
        self.loop = loop
        self.receive = receive
        self.failed = failed
        self.outbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.reader = threading.Thread(
            target=self._read, name="console-service-reader", daemon=True
        )
        self.writer = threading.Thread(
            target=self._write, name="console-service-writer", daemon=True
        )
        self._closing = threading.Event()
        self._started: list[threading.Thread] = []

    def start(self) -> None:
        """Start only after the lifecycle owner has retained this channel."""
        if self._started or self.loop is None:
            raise RuntimeError("Console channel already started")
        for thread in (self.reader, self.writer):
            try:
                thread.start()
            finally:
                if thread.ident is not None:
                    self._started.append(thread)

    def send(self, message: ConsoleWireMessage) -> None:
        # Validate/detach before acceptance into the writer queue. Neither the
        # queue nor the pipe owns native/source authority.
        value = encode_message(message)
        if self._closing.is_set():
            raise ConsoleChannelLost("Console channel is closed")
        self.outbox.put_nowait(value)

    def _report_failure(self, error: BaseException) -> None:
        try:
            self.loop.call_soon_threadsafe(self.failed, error)
        except RuntimeError:
            # Loop shutdown is not operation retirement. The lifecycle owner
            # still retains this channel and its actual thread handles.
            pass

    def _read(self) -> None:
        try:
            while not self._closing.is_set():
                message = decode_message(read_message(_ConsoleFrameInput(self.input)))
                self.loop.call_soon_threadsafe(self.receive, message)
        except BaseException as error:
            if not self._closing.is_set():
                self._report_failure(error)

    def _write(self) -> None:
        try:
            while True:
                value = self.outbox.get()
                if value is None:
                    return
                write_message(self.output, value)
        except BaseException as error:
            self._report_failure(error)

    def close_after_exit(self, deadline: float | None) -> bool:
        """Join pipes only after the actual peer process has exited."""
        self._closing.set()
        self.outbox.put_nowait(None)
        for thread in self._started:
            thread.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
        return all(not thread.is_alive() for thread in self._started)


@dataclass(frozen=True, slots=True)
class ConsoleProcessProfile:
    """Expected original profile selection, checked again by the backend."""

    configuration_path: str
    data_directory: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not Path(value).is_absolute()
            for value in (self.configuration_path, self.data_directory)
        ):
            raise ValueError("Console process profile paths must be absolute")


def _projection_readiness_deadline(requested_at, captured_at, remaining, now) -> float:
    """Keep the backend observation's original expiry on the same host clock."""
    if not requested_at <= captured_at <= now or not 0.0 <= remaining <= 1.0:
        raise ConsoleChannelLost("Console readiness clock is outside the request")
    deadline = captured_at + remaining
    if now >= deadline:
        raise ConsoleReadinessExpired("Console readiness expired before adoption")
    return deadline


class ConsoleExecutionClient:
    """Own one process generation; never create a parent Console store."""

    def __init__(self, app: Any, *, profile_epoch: str | None = None) -> None:
        self._app = app
        self.profile_epoch = profile_epoch or secrets.token_hex(16)
        self.service_epoch = secrets.token_hex(16)
        self._ready_request_id = secrets.token_hex(16)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._profile: ConsoleProcessProfile | None = None
        self._start_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | Future[None] | None = None
        self._permission_bootstrap = None
        self._channel: _ConsoleFramedChannel | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._tree: Any = None
        self._monitor: threading.Thread | None = None
        self._initialization_attempted = False
        self._shutdown_acknowledged = False
        self._requests: dict[str, ConsoleProcessReceipt] = {}
        self._plans: dict[tuple[str, int], ConsoleWireMessage] = {}
        self._sent_sequence = 0
        self._received_sequence = 0
        self._source_revision = 0
        self._intake_open = True
        self._channel_error: BaseException | None = None
        self._cleanup_errors: list[BaseException] = []
        self._shutdown_task: asyncio.Task[bool] | None = None
        self._physically_closed = False
        self._disposed = False
        self._attachment_id = secrets.token_hex(16)
        self._view_checkpoints: set[asyncio.Task] = set()
        self._input_press = None
        self._deferred_inputs = []
        self._authored_inputs = {}
        self._input_tasks: set[asyncio.Task] = set()
        self._view: Any = None
        self._visit = 0
        self._projection = None
        self.projection_view = None
        from .console_execution_readers import ConsoleControllerReader, ConsoleStoreReader

        self.store_reader = ConsoleStoreReader(self)
        self.controller_reader = ConsoleControllerReader(self)
        self.pressed_inputs = None
        self.settings_readiness = None
        self.context_summary = None
        self.agent_rail = None
        self.workspace_surface = None
        self._workspace_count_ids = ()
        self.session_surface = None
        from .console_execution_prompt_history import ConsolePromptHistoryView

        self.prompt_history = ConsolePromptHistoryView(self)
        self.provider_model_display = None
        self.control_provider_label = None
        self._readiness_deadline = 0.0
        self.session_composition = None
        self.send_selection = None
        self.project_head = None
        self._project_modal = None
        self._decision_hint = None
        self._decision_rendered_hint = None
        self._projection_sequence = 0
        self._projection_lock = asyncio.Lock()
        self._projection_reads: set[asyncio.Task] = set()
        self._initial_session_task: asyncio.Task | None = None
        self._initial_conversation_id: str | None = None
        self._resume_view_request = None
        self._view_preparation = None
        self._opening_sessions: set[asyncio.Task] = set()
        self._workspace_operations: set[asyncio.Task] = set()

    def attach_view(self, view: Any) -> int:
        """Bind the actual screen visit without constructing a parent graph."""
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        if (
            type(view) is not ChatScreen
            or view.app_instance is not self._app
            or view.app is not self._app
            or self._app.screen is not view
            or getattr(view, "_console_chat_tearing_down", False)
            or getattr(view, "_closing", False)
            or (self._view is not view and view._task is not asyncio.current_task())
            or getattr(self._app, "console_execution_client", None) is not self
            or getattr(self._app, "console_runtime", None) is not None
            or self._disposed
            or self._shutdown_task is not None
        ):
            raise ConsoleChannelLost("Console attachment owner changed")
        if self._view is not view:
            self._visit += 1
            self._view = view
        return self._visit

    def detach_view(self, view: Any, visit: int) -> bool:
        """End only this screen visit; the process and turns remain app-owned."""
        if self._view is not view or self._visit != visit:
            return False
        # Revoke before any row lookup, channel submission or task creation.
        # The request carries the captured old visit, never the new generation.
        self._view = None
        self._visit += 1
        if self.projection_view is not None and self._channel_error is None and self._shutdown_task is None:
            row = self.projection_view.session(self.projection_view.active_session_id)
            receipt = self.submit("ReadProjection", {
                "known_display_revision": None,
                "view_checkpoint": {
                    "attachment_id": self._attachment_id, "visit": visit, "action": "hidden",
                },
            }, session_id=row.id, incarnation_id=row.incarnation_id)
            task = asyncio.create_task(self._join_hidden_checkpoint(receipt))
            self._view_checkpoints.add(task)
            task.add_done_callback(self._view_checkpoint_finished)
        return True

    async def _join_hidden_checkpoint(self, receipt) -> None:
        try:
            await asyncio.shield(receipt.applied)
        except ConsoleRemoteFailure as error:
            if error.code != "view_checkpoint_superseded" or error.stage != "applied":
                raise
            # This exact old hidden command lost presentation currentness.
            # The finally still requires its original physical retirement.
        finally:
            if await asyncio.shield(receipt.retired) is not True:
                raise ConsoleChannelLost("Console hidden checkpoint did not retire")

    def _view_checkpoint_finished(self, task: asyncio.Task) -> None:
        self._view_checkpoints.discard(task)
        if task.cancelled():
            self._fail(ConsoleChannelLost("Console hidden checkpoint observer cancelled"))
        elif task.exception() is not None:
            self._fail(task.exception())

    def require_view(self, view: Any, visit: int) -> None:
        """Recheck this actual App and visit after an asynchronous boundary."""
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        owned_modal = False
        if self._project_modal is not None:
            from tldw_chatbook.UI.Console_Modules.execution_decisions import is_owned_project_modal

            owned_modal = is_owned_project_modal(self, view, visit)
        if (
            type(view) is not ChatScreen
            or self._view is not view
            or self._visit != visit
            or view.app_instance is not self._app
            or view.app is not self._app
            or (self._app.screen is not view and not owned_modal)
            or getattr(self._app, "console_execution_client", None) is not self
            or getattr(self._app, "console_runtime", None) is not None
            or getattr(view, "_console_chat_tearing_down", False)
            or self._disposed
            or self._shutdown_task is not None
            or self._channel_error is not None
        ):
            raise ConsoleChannelLost("Console attachment visit changed")

    async def prepare_view(
        self, view: Any, *, conversation_id: str | None = None
    ) -> tuple[int, Any]:
        """Join the original target before this actual visit derives content."""
        if self._view is not view and view._task is None:
            # Stock eager startup publishes the actual message-pump Task on
            # return from create_task. No session/native entry precedes this.
            await asyncio.sleep(0)
        visit = self.attach_view(view)
        request = object()
        self._view_preparation = request
        try:
            profile = getattr(self._app, "_console_execution_profile", None)
            await self.start(profile)
            self._require_view_preparation(view, visit, request)
            if self.projection_view is None:
                if self._initial_session_task is None:
                    self._initial_conversation_id = conversation_id
                    self._initial_session_task = self._retain_open_session(conversation_id)
                # An absent target means resume the existing first-session
                # command, including a saved target whose observer was cancelled.
                # It is not an instruction to create another blank session.
                if conversation_id is None or self._initial_conversation_id == conversation_id:
                    session_id, incarnation = await asyncio.shield(self._initial_session_task)
                else:
                    await asyncio.shield(self._initial_session_task)
                    self._require_view_preparation(view, visit, request)
                    session_id, incarnation = await asyncio.shield(
                        self._retain_open_session(conversation_id)
                    )
            elif conversation_id is not None:
                current = self._resume_view_request
                if (
                    current is None or current[0] is not view
                    or current[1] != visit or current[2] != conversation_id
                ):
                    current = (view, visit, conversation_id, self._retain_open_session(conversation_id))
                    self._resume_view_request = current
                session_id, incarnation = await asyncio.shield(current[3])
                if self._resume_view_request is not current:
                    raise ConsoleChannelLost("Console saved target was superseded")
            else:
                row = self.projection_view.session(self.projection_view.active_session_id)
                session_id, incarnation = row.id, row.incarnation_id
            self._require_view_preparation(view, visit, request)
            projection = await self.read_view_display(
                view, visit, session_id, incarnation, preparation=request
            )
            self._require_view_preparation(view, visit, request)
            return visit, projection
        finally:
            if self._view_preparation is request:
                self._view_preparation = None

    def _require_view_preparation(self, view: Any, visit: int, request: object) -> None:
        self.require_view(view, visit)
        if self._view_preparation is not request:
            raise ConsoleChannelLost("Console target preparation was superseded")

    def _retain_open_session(self, conversation_id: str | None) -> asyncio.Task:
        task = asyncio.create_task(
            self._open_initial_session(conversation_id),
            name="console_execution_open_session",
        )
        self._opening_sessions.add(task)
        task.add_done_callback(self._open_session_finished)
        return task

    def _open_session_finished(self, task: asyncio.Task) -> None:
        self._opening_sessions.discard(task)
        # A superseded/cancelled screen may no longer await this command.
        # Retrieve its exception without changing the original receipt outcome.
        if not task.cancelled():
            task.exception()

    async def _open_initial_session(
        self, conversation_id: str | None
    ) -> tuple[str, str]:
        receipt = self.submit("OpenSession", {
            "title": "New chat", "workspace_id": None,
            "conversation_id": conversation_id,
        })
        try:
            value = await asyncio.shield(receipt.applied)
        finally:
            if await asyncio.shield(receipt.retired) is not True:
                error = ConsoleChannelLost("Console initial session did not retire")
                self._fail(error)
                raise error
        return value["session_id"], value["incarnation_id"]

    async def open_new_chat(
        self, view: Any, *, workspace_id: str, title: str = "",
        ephemeral: bool = False, new_chat_default_generation: int = 0,
    ):
        """Retain the original new-session command and its exact publication."""
        visit, source = self._visit, self._source_revision
        self.require_view(view, visit)
        if self._view_preparation is not None or self._opening_sessions:
            raise ConsoleChannelLost("Console session opening is already pending")
        row = self.projection_view.session(self.projection_view.active_session_id)

        async def open_chat():
            # Publish the actual retained task before any command can enter.
            await asyncio.sleep(0)
            async with self._projection_lock:
                self.require_view(view, visit)
                if (
                    self._source_revision != source
                    or self._view_preparation is not None
                    or self.projection_view.active_session_id != row.id
                    or self.projection_view.session(row.id).incarnation_id != row.incarnation_id
                ):
                    raise ConsoleChannelLost("Console New Chat origin changed")
                receipt = self.submit("OpenSession", {
                    "title": title, "workspace_id": workspace_id,
                    "conversation_id": None, "new_chat": True,
                    "ephemeral": ephemeral,
                    "new_chat_default_generation": new_chat_default_generation,
                }, session_id=row.id, incarnation_id=row.incarnation_id)
                try:
                    value = await asyncio.shield(receipt.applied)
                finally:
                    if await asyncio.shield(receipt.retired) is not True:
                        error = ConsoleChannelLost("Console New Chat did not retire")
                        self._fail(error)
                        raise error
                self.require_view(view, visit)
                if self._source_revision != source or self._view_preparation is not None:
                    raise ConsoleChannelLost("Console New Chat source changed")
                return await self._read_projection_locked(
                    value["session_id"], value["incarnation_id"],
                )

        task = asyncio.create_task(open_chat(), name="console_execution_new_chat")
        self._opening_sessions.add(task)
        task.add_done_callback(self._open_session_finished)
        return await asyncio.shield(task)

    def acknowledge_plan(self, receipt: ConsoleProcessReceipt, plan: ConsoleWireMessage) -> ConsoleProcessReceipt:
        """Acknowledge only the exact retained plan after parent presentation."""
        if (
            type(receipt) is not ConsoleProcessReceipt
            or type(plan) is not ConsoleWireMessage
            or self._requests.get(receipt.identity.request_id) is not receipt
            or receipt.operation != "ReceiveTurn"
            or plan.kind != "commit_planned"
            or self._plans.get((receipt.identity.request_id, plan.payload["ordinal"])) is not plan
            or plan.identity.session_id != receipt.identity.session_id
            or plan.identity.incarnation_id != receipt.identity.incarnation_id
        ):
            raise ConsoleProtocolError("unknown_commit_request")
        values = encode_message(plan)["payload"]
        return self.submit(
            "CommitPlanAck",
            {name: values[name] for name in (
                "received_request_id", "turn_id", "ordinal", "fingerprint"
            )},
            session_id=receipt.identity.session_id,
            incarnation_id=receipt.identity.incarnation_id,
        )

    async def next_turn_event(self, receipt: ConsoleProcessReceipt, index: int) -> ConsoleWireMessage | None:
        """Observe the next original event without replay or receipt cancellation."""
        if (type(receipt) is not ConsoleProcessReceipt
            or receipt.operation != "ReceiveTurn"
            or type(index) is not int or index < 0
        ):
            raise ConsoleProtocolError("turn_receipt_required")
        while True:
            receipt.changed.clear()
            if index < len(receipt.events):
                return receipt.events[index]
            if receipt.retired.done():
                receipt.retired.result()
                return None
            if self._channel_error is not None:
                raise ConsoleChannelLost("Console outcome is unknown")
            if self._requests.get(receipt.identity.request_id) is not receipt:
                raise ConsoleProtocolError("turn_receipt_changed")
            await receipt.changed.wait()

    def publish_authored_inputs(self, view, visit: int, draft: str, token: tuple[int, int], *, _switch=None, _session_id=None) -> bool:
        """Queue the actual widget edit; Applied alone publishes its revision."""
        from .console_execution_projection_values import _settings

        self.require_view(view, visit)
        if (type(draft) is not str or type(token) is not tuple or len(token) != 2
            or any(type(item) is not int or item < 0 for item in token)):
            raise ConsoleProtocolError("invalid_authored_input_identity")
        transition = getattr(view, "_console_process_draft_switch", None)
        if transition is not None and _switch is None:
            # The original switch snapshot assigns edits made in this window
            # only when the destination's actual command has published its row.
            if transition[0] is not self or transition[1] != visit:
                raise ConsoleProtocolError("authored_switch_owner_changed")
            return True
        if _switch is not None and transition is not _switch:
            raise ConsoleProtocolError("authored_switch_owner_changed")
        row = self.projection_view.session(_session_id or self.projection_view.active_session_id)
        if _switch is None and view._console_visible_draft_session_id != row.id:
            return False
        key = (row.id, row.incarnation_id)
        if self._input_press is not None:
            self._deferred_inputs.append((view, visit, key, self._source_revision, draft, token))
            return True
        previous = self._authored_inputs.get(key)
        if previous is not None and previous[:2] == (draft, token):
            return True
        receipt = self.submit("UpdateInputs", {
            "draft": draft, "authored_token": list(token),
            "expected_settings_revision": row.settings_revision,
            "expected_settings": _settings(row.settings, False),
            "expected_attachment_ids": None if _switch is not None else list(self.pressed_inputs["attachment_ids"]),
            "remove_attachment_ids": [], "prefill": None,
        }, session_id=row.id, incarnation_id=row.incarnation_id)
        entry = (draft, token, receipt)
        self._authored_inputs[key] = entry
        work = self._join_authored_inputs(view, visit, key, entry)
        try:
            task = asyncio.create_task(work, name="console_execution_authored_input")
        except BaseException as error:
            work.close()
            # The actual issued receipt remains retained; never invent rollback.
            self._fail(error)
            raise
        self._input_tasks.add(task)
        task.add_done_callback(self._authored_inputs_finished)
        return True

    async def _join_authored_inputs(self, view, visit, key, entry) -> None:
        receipt = entry[2]
        try:
            applied = await asyncio.shield(receipt.applied)
        except ConsoleRemoteFailure as error:
            if error.stage == "admission":
                # No original command receipt was returned. Keep the unresolved
                # physical receipt retained, without waiting forever or claiming
                # a positive retirement for this refused edit.
                if receipt.received.done() and not receipt.received.cancelled():
                    receipt.received.exception()
                raise
            if await asyncio.shield(receipt.retired) is not True:
                raise ConsoleChannelLost("Console authored input retirement is unproven") from error
            raise
        else:
            if await asyncio.shield(receipt.retired) is not True:
                raise ConsoleChannelLost("Console authored input retirement is unproven")
        if (self._view is view and self._visit == visit
            and self._authored_inputs.get(key) is entry
            and view._console_visible_draft_session_id == key[0]
            and self.projection_view.active_session_id == key[0]
            and self.projection_view.session(key[0]).incarnation_id == key[1]):
            from tldw_chatbook.UI.Console_Modules.execution_state import require_authored_owner

            require_authored_owner(view, self, visit)
            view._session._console_visible_draft_revision = applied["draft_revision"]

    def _authored_inputs_finished(self, task: asyncio.Task) -> None:
        self._input_tasks.discard(task)
        if task.cancelled():
            self._fail(ConsoleChannelLost("Console input observer cancelled"))
        else:
            # A normal stale-settings refusal stays on its exact receipt;
            # the next press must join that receipt and preserve its draft.
            error = task.exception()
            if isinstance(error, ConsoleChannelLost):
                self._fail(error)

    def retain_authored_press(self, view, visit: int, draft: str, token: tuple[int, int], *, inputs=None):
        """Retain one actual press before later edits enter the command channel."""
        from .console_execution_protocol import _PROJECTED_INPUTS, _detach

        self.require_view(view, visit)
        if self._input_press is not None:
            raise ConsoleProtocolError("authored_press_already_pending")
        row = self.projection_view.session(self.projection_view.active_session_id)
        key = (row.id, row.incarnation_id)
        retained = _detach(self.pressed_inputs if inputs is None else inputs, "pressed_inputs", 0, set())
        _PROJECTED_INPUTS(retained, "pressed_inputs")
        if (retained["session_id"], retained["incarnation_id"]) != key:
            raise ConsoleProtocolError("authored_press_session_changed")
        if not self.publish_authored_inputs(view, visit, draft, token):
            raise ConsoleProtocolError("authored_press_session_changed")
        press = _ConsoleInputPress(
            view, visit, key, self._authored_inputs[key],
            retained, self._source_revision,
        )
        self._input_press = press
        return press

    def release_authored_press(self, press) -> None:
        """Release only this press; original later widget edits keep FIFO order."""
        if type(press) is not _ConsoleInputPress or self._input_press is not press:
            raise ConsoleProtocolError("authored_press_changed")
        self._input_press = None
        queued, self._deferred_inputs = self._deferred_inputs, []
        for view, visit, key, revision, draft, token in queued:
            # Detached edits have no authority to change the new visit. Their
            # actual composer still owns the text; never replay into another chat.
            if (
                self._view is view and self._visit == visit
                and self._source_revision == revision
                and self.projection_view.active_session_id == key[0]
                and self.projection_view.session(key[0]).incarnation_id == key[1]
            ):
                self.publish_authored_inputs(view, visit, draft, token)

    def submit_authored_turn(self, press, selection: dict, turn_id: str) -> asyncio.Task:
        """Retain admission independently of the UI task observing this press."""
        from .console_execution_protocol import _detach

        if type(press) is not _ConsoleInputPress or self._input_press is not press:
            raise ConsoleProtocolError("authored_press_changed")
        self.require_view(press.view, press.visit)
        if press.task is not None:
            raise ConsoleProtocolError("authored_press_already_dispatched")
        # Caller values are detached now, before a later UI callback can edit them.
        selection = _detach(_detach(selection, "selection", 0, set(), thaw=False), "selection", 0, set(), thaw=True)
        work = self._submit_authored_turn(press, selection, turn_id)
        try:
            task = asyncio.create_task(work, name="console_execution_pressed_admission")
        except BaseException:
            work.close()
            self.release_authored_press(press)
            raise
        press.task = task
        self._input_tasks.add(task)
        task.add_done_callback(self._authored_inputs_finished)
        return task

    async def _submit_authored_turn(self, press, selection, turn_id):
        from .console_execution_protocol import _detach

        try:
            revision = await asyncio.shield(press.entry[2].applied)
            if await asyncio.shield(press.entry[2].retired) is not True:
                raise ConsoleChannelLost("Console pressed input retirement is unproven")
            from tldw_chatbook.UI.Console_Modules.execution_state import require_authored_owner

            require_authored_owner(press.view, self, press.visit)
            if self._input_press is not press or self._source_revision != press.source_revision:
                raise ConsoleChannelLost("Console pressed input source changed")
            row = self.projection_view.session(self.projection_view.active_session_id)
            if (row.id, row.incarnation_id) != press.key:
                raise ConsoleChannelLost("Console pressed input session changed")
            inputs = _detach(press.inputs, "pressed_inputs", 0, set(), thaw=True)
            inputs["draft"], token = press.entry[:2]
            inputs["authored_token"] = list(token)
            receipt = self.submit("ReceiveTurn", {
                "turn_id": turn_id,
                "draft_revision": revision["draft_revision"],
                "selection": selection,
                "expected_settings_revision": inputs["settings_revision"],
                "expected_attachment_revision": inputs["attachment_revision"],
                "expected_prefill_revision": inputs["prefill_revision"],
                "pressed_inputs": inputs,
            }, session_id=press.key[0], incarnation_id=press.key[1])
            press.receipt = receipt
            # Received follows the original claim, not merely transport parsing.
            # Later edits may enter only after this claim or its known refusal.
            await asyncio.shield(receipt.received)
            return receipt
        finally:
            if self._input_press is press:
                self.release_authored_press(press)


    async def read_projection(
        self, session_id: str, incarnation_id: str, *, _preparation=None,
        _checkpoint: str | None = None, _run_history=None, _inspector_refresh: str | None = None,
    ):
        """Serialize actual reads through their positive physical retirement."""
        view, visit, source = self._view, self._visit, self._source_revision
        active = None if self.projection_view is None else self.projection_view.active_session_id

        async def read():
            await asyncio.sleep(0)
            async with self._projection_lock:
                if (
                    self._view is not view or self._visit != visit
                    or self._source_revision != source
                    or self._view_preparation is not _preparation
                    or self._shutdown_task is not None
                    or (self.projection_view is not None
                        and self.projection_view.active_session_id != active)
                ):
                    raise ConsoleChannelLost("Console queued projection changed")
                if view is not None:
                    self.require_view(view, visit)
                return await self._read_projection_locked(
                    session_id, incarnation_id, _preparation=_preparation,
                    _checkpoint=_checkpoint,
                    _run_history=_run_history,
                    _inspector_refresh=_inspector_refresh,
                )

        task = asyncio.create_task(read(), name="console_execution_projection")
        self._projection_reads.add(task)
        task.add_done_callback(self._projection_read_finished)
        return await asyncio.shield(task)

    async def read_view_display(
        self, view, visit: int, session_id: str, incarnation_id: str, *,
        preparation=None, checkpoint: str | None = None,
    ):
        """Wait for fresh display data on this visit; issue its lifecycle checkpoint once."""
        source = self._source_revision
        active = self.projection_view.active_session_id if self.projection_view is not None else None
        while True:
            self.require_view(view, visit)
            current = self.projection_view.active_session_id if self.projection_view is not None else None
            if self._source_revision != source or self._view_preparation is not preparation or current != active:
                raise ConsoleChannelLost("Console display request was superseded")
            try:
                return await self.read_projection(
                    session_id, incarnation_id, _preparation=preparation, _checkpoint=checkpoint,
                )
            except ConsoleReadinessExpired:
                # The original request already applied and physically retired.
                # Only refresh its display. Replaying a mount/visible action
                # or publishing its expired readiness would each be incorrect.
                checkpoint = None
                await asyncio.sleep(0.25)

    def _projection_read_finished(self, task: asyncio.Task) -> None:
        self._projection_reads.discard(task)
        if not task.cancelled():
            task.exception()

    async def _read_projection_locked(
        self, session_id: str, incarnation_id: str, *, _preparation=None,
        _checkpoint: str | None = None, _run_history=None, _inspector_refresh: str | None = None,
    ):
        """Adopt one exact retired projection without constructing a local store."""
        requested_at = time.monotonic()
        from .console_execution_projection_result import (
            decode_projection_payload,
            decode_projection_view,
            decode_projection_readiness,
            projection_key,
        )
        from .console_execution_composition_values import decode_session_composition

        previous = self._projection
        view, visit = self._view, self._visit
        attachment, app = self._attachment_id, self._app
        if self._view_preparation is not _preparation:
            raise ConsoleChannelLost("Console target preparation is pending")
        if view is not None:
            self.require_view(view, visit)
        binding = (self._source_revision, session_id, incarnation_id)
        known = (
            previous.display_revision
            if previous is not None and projection_key(previous)[:3] == binding
            else None
        )
        payload = {"known_display_revision": known}
        payload["workspace_count_ids"] = list(self._workspace_count_ids)
        payload["known_marker_revision"] = previous.marker_revision if known is not None else None
        if _run_history is not None:
            payload["run_history"] = _run_history
        if _inspector_refresh is not None:
            payload["inspector_refresh"] = _inspector_refresh
        if _checkpoint is not None:
            if view is None or _checkpoint not in {"mount", "visible"}:
                raise ConsoleChannelLost("Console visible checkpoint requires its view")
            self.require_view(view, visit)
            payload["view_checkpoint"] = {
                "attachment_id": self._attachment_id, "visit": visit, "action": _checkpoint,
            }
        receipt = self.submit(
            "ReadProjection",
            payload,
            session_id=session_id,
            incarnation_id=incarnation_id,
        )
        failure = None
        try:
            await asyncio.shield(receipt.applied)
        except ConsoleRemoteFailure as error:
            failure = error
        finally:
            admission_refused = False
            try:
                # Cancellation can arrive before the admission verdict. An
                # actual Received still owes retirement, even when Applied fails.
                await asyncio.shield(receipt.received)
            except ConsoleRemoteFailure as error:
                admission_refused = error.stage == "admission"
                if admission_refused:
                    # The worker returned no local receipt and emits no Retired
                    # frame. Retain the unresolved physical receipt; consume its
                    # paired error if cancellation interrupted the Applied wait.
                    receipt.applied.exception()
            if not admission_refused:
                if await asyncio.shield(receipt.retired) is not True:
                    error = ConsoleChannelLost("Console projection retirement is unproven")
                    self._fail(error)
                    raise error
        if failure is not None:
            if (
                type(failure) is ConsoleRemoteFailure
                and failure.code == "view_checkpoint_superseded"
                and failure.stage == "applied"
                and not admission_refused
                and receipt.received.exception() is None
                and receipt.retired.result() is True
                and view is not None and self._visit > visit
                and self._attachment_id == attachment
                and self._source_revision == binding[0]
                and self._view_preparation is _preparation
                and self._app is app
                and getattr(app, "console_execution_client", None) is self
                and getattr(app, "console_runtime", None) is None
                and self._channel_error is None and not self._disposed
                and self._shutdown_task is None
                and self.projection_view is not None
                and self.projection_view.active_session_id == session_id
                and self.projection_view.session(session_id).incarnation_id == incarnation_id
            ):
                # The backend proved this original FULL's native retirement;
                # only the actual ended parent visit may defer its display.
                raise ConsoleChannelLost("Console display visit was superseded") from failure
            raise failure
        if (
            self._channel_error is not None
            or self._source_revision != binding[0]
            or self._view is not view
            or self._visit != visit
            or self._view_preparation is not _preparation
            or receipt.identity.sequence <= self._projection_sequence
        ):
            raise ConsoleChannelLost("Console projection source changed")
        # Thaw only this validated message using the canonical codec. The
        # reference resolves against the snapshot captured for this request.
        message = receipt.projection_message
        if message is None:
            raise ConsoleProtocolError("projection_message_missing")
        payload = encode_message(message)["payload"]
        value = decode_projection_payload(payload, previous=previous)
        projection_view = decode_projection_view(payload, value)
        from .console_execution_workspace_values import decode_workspace_surface
        from .console_execution_session_surface import decode_session_surface_projection

        workspace_surface = decode_workspace_surface(payload["session_view"]["workspace"])
        session_surface = decode_session_surface_projection(
            projection_view, payload["session_view"]["session_surface"],
        )
        decoded_readiness = decode_projection_readiness(
            payload, projection_view
        )
        if len(decoded_readiness) == 11:
            decoded_readiness = (*decoded_readiness, None)
        effective, readiness, provider, model, remaining, captured_at, provider_chip, context_summary, agent_rail, prompt_history, send_selection, project_head = decoded_readiness
        composition = (
            decode_session_composition(
                payload["session_view"]["composition"],
                projection_view.session(projection_view.active_session_id),
            ) if projection_view.active_session_id is not None else None
        )
        deadline = _projection_readiness_deadline(
            requested_at, captured_at, remaining, time.monotonic(),
        )
        if view is not None:
            self.require_view(view, visit)
        self._projection = value
        self.projection_view = projection_view
        self.settings_readiness = effective, readiness
        self.context_summary = context_summary
        self.agent_rail = agent_rail
        self.workspace_surface = workspace_surface
        self.session_surface = session_surface
        self.prompt_history.publish(prompt_history)
        self.control_provider_label = provider_chip
        self.provider_model_display = provider, model, projection_view.session(session_id).settings
        self._readiness_deadline = deadline
        self.session_composition = composition
        self.send_selection = send_selection
        self.project_head = project_head
        self.pressed_inputs = payload["session_view"]["pressed_inputs"]
        self._projection_sequence = receipt.identity.sequence
        return value

    async def read_run_history(self, request):
        """Join one original history read for the currently published chat."""
        from .console_backend_run_history import _REQUEST, decode_run_history
        from .console_execution_protocol import _detach

        frozen = _detach(request, "run_history", 0, set(), thaw=False)
        _REQUEST(frozen, "run_history")
        request = _detach(frozen, "run_history", 0, set(), thaw=True)
        view = self.store_reader._current()
        row = view.session(view.active_session_id)
        if row.persisted_conversation_id != request["conversation_id"]:
            raise ConsoleChannelLost("Console run history conversation changed")
        frame = await self.read_projection(row.id, row.incarnation_id, _run_history=request)
        if frame.run_history is None:
            raise ConsoleChannelLost("Console original run history is unavailable")
        values = decode_run_history(list(frame.run_history))
        if any(value["conversation_id"] != request["conversation_id"] for value in values):
            raise ConsoleChannelLost("Console run history ownership changed")
        return values

    async def refresh_prompt_history(self, history) -> None:
        """Retry the original backend load through this view's named projection."""
        from .console_execution_prompt_history import ConsolePromptHistoryView

        if type(history) is not ConsolePromptHistoryView or history is not self.prompt_history:
            raise ConsoleChannelLost("Console prompt history view changed")
        view, visit = self._view, self._visit
        self.require_view(view, visit)
        row = self.projection_view.session(self.projection_view.active_session_id)
        await self.read_projection(row.id, row.incarnation_id)
        self.require_view(view, visit)

    def set_global_default_before_mount(
        self, profile: ConsoleProcessProfile, *, state: str,
        profile_id: str = "default", expected_profile_digest: str | None = None,
        expected_revision: int | None = None,
    ) -> None:
        """Complete the original fixed setup mutation before async UI startup."""
        from .console_execution_bootstrap import ConsolePermissionBootstrap

        if type(profile) is not ConsoleProcessProfile:
            raise TypeError("Original Console profile selection is required")
        if (self._loop is not None or self._start_task is not None
            or self._shutdown_task is not None or self._channel_error is not None):
            raise ConsoleChannelLost("Console synchronous setup is no longer available")
        operation = {
            "kind": "SetGlobalDefault", "state": state, "profile_id": profile_id,
            "expected_profile_digest": expected_profile_digest,
            "expected_revision": expected_revision,
        }
        # Validate the existing closed operation before launching any backend.
        ConsoleWireMessage(
            WireIdentity(self.profile_epoch, self.service_epoch, self._sent_sequence + 1,
                         secrets.token_hex(16), "", "", self._source_revision),
            "command", "PermissionAction", {"operation": operation},
        )
        if self._permission_bootstrap is None:
            self._permission_bootstrap = ConsolePermissionBootstrap(self)
        self._permission_bootstrap.invoke(profile, operation)

    async def start(self, profile: ConsoleProcessProfile) -> None:
        """Join the exact retained startup, including a Send made before Ready."""
        if type(profile) is not ConsoleProcessProfile:
            raise TypeError("Original Console profile selection is required")
        loop = asyncio.get_running_loop()
        if self._profile is not None and self._profile != profile:
            raise ConsoleChannelLost("Console profile changed within its lifetime")
        if self._loop is not None and self._loop is not loop:
            raise ConsoleChannelLost("Console owner loop changed")
        if self._start_task is None:
            bootstrap = self._permission_bootstrap
            if bootstrap is not None:
                bootstrap.require_adoptable(profile)
            self._profile = profile
            self._loop = loop
            if bootstrap is None:
                self._ready = loop.create_future()
            self._start_task = loop.create_task(self._start(bootstrap=bootstrap))
        await asyncio.shield(self._start_task)

    async def _start(self, *, bootstrap=None) -> None:
        try:
            # Creation is retained independently of observer cancellation. No
            # native/backend work starts until containment is established and
            # the fixed initialization frame is transmitted.
            if bootstrap is None:
                await asyncio.to_thread(self._launch)
            else:
                if self._permission_bootstrap is not bootstrap:
                    raise ConsoleChannelLost("Console pre-mount startup owner changed")
                bootstrap.require_adoptable(self._profile)
                self._ready = self._loop.create_future()
                self._ready.set_result(None)  # Already validated original Ready.
                self._channel.loop = self._loop
                self._channel.start()
            assert self._ready is not None
            await self._ready
        except BaseException as error:
            self._fail(error)
            raise

    def _launch(self, *, start_channel: bool = True) -> None:
        from tldw_chatbook.STT.executor_process_tree import (
            ExecutorProcessTree,
            WorkerContainmentIdentity,
        )
        from tldw_chatbook.Tools.workspace_tool_executor import _PopenProcessAdapter

        assert self._profile is not None and (not start_channel or self._loop is not None)
        root = Path(__file__).resolve().parents[2]
        application_paths = [str(root)]
        profile_core = root / "packages" / "tldw_profile_core" / "src"
        if (profile_core / "tldw_profile_core" / "__init__.py").is_file():
            # The source distribution declares this second package root;
            # installed wheels already expose it through site-packages.
            application_paths.append(str(profile_core))
        package_roots = tuple(
            Path(path).resolve()
            for path in sys.path
            if path and Path(path).name == "site-packages"
        )
        dependencies = [
            str(Path(path).resolve())
            for path in sys.path
            if path
            and any(Path(path).resolve().is_relative_to(item) for item in package_roots)
        ]
        executable = str(
            Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        )
        code = (
            "import runpy,sys;sys.path[:0]="
            + repr([*application_paths, *dependencies])
            + ";runpy.run_module('tldw_chatbook.Chat.console_execution_worker',run_name='__main__')"
        )
        process = subprocess.Popen(
            [executable, "-I", "-S", "-B", "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name == "posix",
        )
        self._process = process
        self._tree = ExecutorProcessTree(
            _PopenProcessAdapter(process),
            threading.Event(),
            WorkerContainmentIdentity(
                process.pid, process.pid if os.name == "posix" else None
            ),
        )
        self._tree.admit()
        assert process.stdin is not None and process.stdout is not None
        self._initialization_attempted = True
        write_message(
            process.stdin,
            {
                "initialize": 1,
                "profile_epoch": self.profile_epoch,
                "service_epoch": self.service_epoch,
                "request_id": self._ready_request_id,
                "configuration_path": self._profile.configuration_path,
                "data_directory": self._profile.data_directory,
            },
        )
        self._channel = _ConsoleFramedChannel(
            process.stdout,
            process.stdin,
            self._loop,
            self._receive,
            self._channel_failed,
        )
        if start_channel:
            self._channel.start()

        def monitor() -> None:
            # Drain diagnostics continuously so stderr capacity cannot stop the
            # original backend. Do not expose sensitive payloads in the client.
            assert process.stderr is not None
            while process.stderr.read(65536):
                pass
            process.wait()
            loop = self._loop
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(self._exited, process.returncode)
                except RuntimeError:
                    pass

        self._monitor = threading.Thread(
            target=monitor, name="console-service-monitor", daemon=True
        )
        self._monitor.start()

    def submit(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        session_id: str = "",
        incarnation_id: str = "",
    ) -> ConsoleProcessReceipt:
        """Retain a fixed command before handing bytes to the serialized writer."""
        if self._loop is not asyncio.get_running_loop():
            raise ConsoleChannelLost("Console command came from another owner loop")
        return self._submit_command(
            operation, payload, session_id=session_id, incarnation_id=incarnation_id,
        )

    def _submit_command(
        self, operation: str, payload: Mapping[str, Any], *,
        session_id: str = "", incarnation_id: str = "",
    ) -> ConsoleProcessReceipt:
        """Issue through the same sequencer on one actual transport owner."""
        synchronous = self._loop is None
        if synchronous:
            from .console_execution_bootstrap import ConsolePermissionBootstrap

            owner = self._permission_bootstrap
            if (type(owner) is not ConsolePermissionBootstrap or owner.client is not self
                or owner.thread is not threading.current_thread()
                or self._start_task is not None or operation != "PermissionAction"
                or payload.get("operation", {}).get("kind") != "SetGlobalDefault"
                or session_id or incarnation_id or self._channel is None
                or self._channel._started or owner.aborted.is_set()):
                raise ConsoleChannelLost("Console pre-mount command owner changed")
            future = Future
        else:
            if self._loop is not asyncio.get_running_loop():
                raise ConsoleChannelLost("Console command came from another owner loop")
            future = self._loop.create_future
        if self._channel_error is not None:
            raise ConsoleChannelLost(
                "Console channel is unavailable"
            ) from self._channel_error
        if self._ready is None or not self._ready.done() or self._ready.cancelled():
            raise ConsoleChannelLost("Console service is not ready")
        self._ready.result()
        if not self._intake_open and operation not in {
            "CommitPlanAck",
            "CancelTurn",
            "CloseIntake",
            "Drain",
            "Shutdown",
            "FenceSource",
            "ApplySourceRevision",
            "ReplaceConfiguration",
            "RefreshConfiguration",
            "SetWorkspaceBindingAccess",
            "ProjectDecisionPresentation",
        }:
            raise ConsoleChannelLost("Console intake is closed")
        identity = WireIdentity(
            self.profile_epoch,
            self.service_epoch,
            self._sent_sequence + 1,
            secrets.token_hex(16),
            session_id,
            incarnation_id,
            self._source_revision,
        )
        message = ConsoleWireMessage(identity, "command", operation, payload)
        receipt = ConsoleProcessReceipt(
            identity,
            operation,
            future(),
            future(),
            future(),
        )
        receipt.turn_id = message.payload["turn_id"] if operation == "ReceiveTurn" else None
        self._requests[identity.request_id] = receipt
        self._sent_sequence = identity.sequence
        assert self._channel is not None
        if operation == "CommitPlanAck":
            # Queueing transmission is already potentially committed. A writer
            # failure cannot prove which bytes the peer received.
            original = self._requests.get(payload["received_request_id"])
            plan = self._plans.get((payload["received_request_id"], payload["ordinal"]))
            if (
                original is None
                or original.operation != "ReceiveTurn"
                or plan is None
                or any(
                    payload[name] != plan.payload[name]
                    for name in (
                        "received_request_id",
                        "turn_id",
                        "ordinal",
                        "fingerprint",
                    )
                )
            ):
                self._requests.pop(identity.request_id)
                self._sent_sequence -= 1
                raise ConsoleProtocolError("unknown_commit_request")
            original.potentially_committed = True
        if operation in {"FenceSource", "CloseIntake", "Drain", "Shutdown"}:
            self._intake_open = False
        try:
            if synchronous:
                write_message(self._channel.output, encode_message(message))
            else:
                self._channel.send(message)
        except BaseException as error:
            self._fail(error)
            raise
        return receipt

    def _receive(self, message: ConsoleWireMessage) -> None:
        try:
            identity = message.identity
            if (
                identity.profile_epoch != self.profile_epoch
                or identity.service_epoch != self.service_epoch
                or identity.sequence != self._received_sequence + 1
            ):
                raise ConsoleProtocolError("channel_identity_or_sequence_changed")
            self._received_sequence = identity.sequence
            if message.kind == "ready":
                if (
                    self._ready is None
                    or self._ready.done()
                    or self._process is None
                    or message.payload["pid"] != self._process.pid
                    or identity.request_id != self._ready_request_id
                    or identity.session_id
                    or identity.incarnation_id
                    or identity.source_revision != message.payload["source_revision"]
                ):
                    raise ConsoleProtocolError("unexpected_ready")
                self._source_revision = message.payload["source_revision"]
                self._ready.set_result(None)
                return
            if message.kind == "decision_changed":
                from tldw_chatbook.UI.Console_Modules.execution_state import request_selected_decision_refresh

                request_selected_decision_refresh(self, (
                    identity.source_revision, identity.session_id,
                    identity.incarnation_id, message.payload["attachment_id"],
                    message.payload["visit"], message.payload["decision_revision"],
                ))
                return
            receipt = self._requests.get(identity.request_id)
            if receipt is None or receipt.operation != message.operation:
                raise ConsoleProtocolError("unknown_operation_result")
            if (
                identity.session_id != receipt.identity.session_id
                or identity.incarnation_id != receipt.identity.incarnation_id
                or identity.source_revision != receipt.identity.source_revision
            ):
                raise ConsoleProtocolError("operation_session_changed")
            if message.kind == "received":
                if (
                    receipt.received.done()
                    or message.payload["sequence"] != receipt.identity.sequence
                ):
                    raise ConsoleProtocolError("duplicate_or_conflicting_received")
                receipt.received.set_result(None)
            elif message.kind == "applied":
                if receipt.applied.done() or not receipt.received.done():
                    raise ConsoleProtocolError("unexpected_applied")
                if receipt.operation == "FenceSource":
                    revision = message.payload["revision"]
                    if revision != self._source_revision + 1:
                        raise ConsoleProtocolError("unexpected_fence_revision")
                    self._source_revision = revision
                elif receipt.operation == "ApplySourceRevision":
                    if message.payload["source_revision"] != self._source_revision:
                        raise ConsoleProtocolError("unexpected_publication_revision")
                    self._intake_open = True
                elif receipt.operation == "ReadProjection":
                    receipt.projection_message = message
                receipt.applied.set_result(message.payload)
            elif message.kind == "retired":
                if (
                    receipt.retired.done()
                    or not receipt.applied.done()
                    or message.payload["operation_id"] != identity.request_id
                ):
                    raise ConsoleProtocolError("unexpected_retired")
                receipt.retired.set_result(message.payload["retired"])
                receipt.changed.set()
                if (
                    receipt.operation == "Shutdown"
                    and message.payload["retired"] is True
                    and receipt.applied.exception() is None
                    and receipt.applied.result().get("applied") is True
                ):
                    self._shutdown_acknowledged = True
                if message.payload["retired"] is True:
                    self._requests.pop(identity.request_id)
                    if receipt.operation == "ReceiveTurn":
                        for key in tuple(self._plans):
                            if key[0] == identity.request_id:
                                self._plans.pop(key)
            elif message.kind == "failed":
                if receipt.applied.done():
                    if (
                        receipt.operation != "ReceiveTurn"
                        or receipt.applied.exception() is not None
                    ):
                        raise ConsoleProtocolError("conflicting_operation_failure")
                    # Claim adoption and the retained turn's terminal outcome
                    # are separate. Do not retract the original Applied claim.
                    receipt.events.append(message)
                    self._project_event(receipt, message)
                    return
                error = ConsoleRemoteFailure(
                    message.payload["code"], message.payload["stage"]
                )
                if not receipt.received.done():
                    receipt.received.set_exception(error)
                receipt.applied.set_exception(error)
            else:
                if message.kind in {"input_projected", "terminal"}:
                    if (
                        receipt.operation != "ReceiveTurn"
                        or not receipt.received.done()
                        or message.payload["turn_id"] != receipt.turn_id
                        or any(item.kind == message.kind for item in receipt.events)
                    ):
                        raise ConsoleProtocolError("unexpected_turn_checkpoint")
                if message.kind == "commit_planned":
                    if receipt.operation != "ReceiveTurn":
                        raise ConsoleProtocolError("plan_without_received_turn")
                    key = (identity.request_id, message.payload["ordinal"])
                    if key in self._plans:
                        raise ConsoleProtocolError("duplicate_commit_plan")
                    self._plans[key] = message
                receipt.events.append(message)
                self._project_event(receipt, message)
        except BaseException as error:
            self._fail(error)

    def _project_event(
        self, receipt: ConsoleProcessReceipt, message: ConsoleWireMessage
    ) -> None:
        """Keep retained events until the exact attachment consumes them."""
        # Events wake the exact original receipt's UI observer; they grant no
        # command/source/native authority and never trigger automatic ACK.
        receipt.changed.set()

    def _fail(self, error: BaseException) -> None:
        if self._channel_error is not None:
            return
        self._channel_error = error
        self._intake_open = False
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(ConsoleChannelLost("Console startup failed"))
        for receipt in self._requests.values():
            receipt.changed.set()
            for future in (receipt.received, receipt.applied, receipt.retired):
                if not future.done():
                    future.set_exception(
                        ConsoleChannelLost("Console outcome is unknown")
                    )

    def _channel_failed(self, error: BaseException) -> None:
        # The reader queues decoded frames and EOF on the same loop in order.
        # Only the original positive Shutdown receipt permits expected EOF;
        # process absence, task completion and an earlier Drain are insufficient.
        if isinstance(error, EOFError) and self._shutdown_acknowledged:
            return
        self._fail(error)

    def _exited(self, returncode: int | None) -> None:
        if self._shutdown_task is None:
            self._fail(ConsoleChannelLost(f"Console service exited ({returncode})"))

    @property
    def physically_closed(self) -> bool:
        """Report lifecycle cleanup, independently of command retirement."""
        return self._physically_closed

    async def shutdown(self, *, deadline: float | None) -> bool:
        """Retain original drain and process cleanup across observer cancellation.

        Args:
            deadline: The initiating lifecycle owner's existing monotonic bound, or None
                for the original App quit path, which has no deadline.

        Returns:
            Whether backend drain, process exit, pipes, monitor and containment
            all positively settled. Forced exit alone never returns success.
        """
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown(deadline))
        return await asyncio.shield(self._shutdown_task)

    async def _shutdown(self, deadline: float | None) -> bool:
        drained = False
        cancellation = None
        try:
            async with asyncio.timeout(None if deadline is None else max(0.0, deadline - time.monotonic())):
                if self._permission_bootstrap is not None and self._start_task is None:
                    await self.start(self._profile)
                if self._start_task is not None:
                    await asyncio.shield(self._start_task)
                # Retain issued session commands through their exact retirement;
                # queued openings recheck this shutdown before issuing.
                opening = tuple(self._opening_sessions)
                if opening:
                    await asyncio.gather(
                        *(asyncio.shield(task) for task in opening),
                        return_exceptions=True,
                    )
                reads = tuple(self._projection_reads)
                if reads:
                    await asyncio.gather(
                        *(asyncio.shield(task) for task in reads), return_exceptions=True,
                    )
                while self._input_tasks:
                    input_tasks = tuple(self._input_tasks)
                    await asyncio.gather(
                        *(asyncio.shield(task) for task in input_tasks), return_exceptions=True,
                    )
                checkpoints = tuple(self._view_checkpoints)
                if checkpoints:
                    await asyncio.gather(*(asyncio.shield(task) for task in checkpoints))
                workspace_operations = tuple(self._workspace_operations)
                if workspace_operations:
                    await asyncio.gather(
                        *(asyncio.shield(task) for task in workspace_operations),
                        return_exceptions=True,
                    )
                for operation in ("CloseIntake", "Drain", "Shutdown"):
                    receipt = self.submit(operation, {})
                    await asyncio.shield(receipt.received)
                    applied = await asyncio.shield(receipt.applied)
                    retired = await asyncio.shield(receipt.retired)
                    if applied.get("applied") is not True or retired is not True:
                        raise ConsoleChannelLost("Console lifecycle did not retire")
                if self._process is not None:
                    returncode = await asyncio.to_thread(
                        self._process.wait,
                        None if deadline is None else max(0.0, deadline - time.monotonic()),
                    )
                    if returncode != 0:
                        raise ConsoleChannelLost(
                            "Console service did not exit normally"
                        )
                drained = True
        except (Exception, asyncio.CancelledError) as error:
            # Preserve the negative result even if containment subsequently
            # succeeds. Termination is not native retirement or rollback.
            drained = False
            self._fail(error)
            if isinstance(error, asyncio.CancelledError):
                cancellation = error
        finally:
            self._intake_open = False
            if self._permission_bootstrap is not None:
                if self._permission_bootstrap.thread is not None and self._permission_bootstrap.thread.is_alive():
                    self._permission_bootstrap.aborted.set()
                    await asyncio.to_thread(self._permission_bootstrap.stop)
                await asyncio.to_thread(self._permission_bootstrap.join)
            # A timeout of the observer does not cancel the off-loop creator.
            # Keep this cleanup Task until it can see every resource that the
            # exact startup actually published, including partial construction.
            if self._start_task is not None and not self._start_task.done():
                try:
                    await asyncio.shield(self._start_task)
                except BaseException as error:
                    self._fail(error)
            opening = tuple(self._opening_sessions)
            if opening:
                await asyncio.gather(
                    *(asyncio.shield(task) for task in opening),
                    return_exceptions=True,
                )
            reads = tuple(self._projection_reads)
            if reads:
                await asyncio.gather(
                    *(asyncio.shield(task) for task in reads), return_exceptions=True,
                )
            workspace_operations = tuple(self._workspace_operations)
            if workspace_operations:
                await asyncio.gather(
                    *(asyncio.shield(task) for task in workspace_operations),
                    return_exceptions=True,
                )
            closed = False
            if self._tree is not None:
                try:
                    closed = await asyncio.to_thread(self._tree.close)
                except BaseException as error:
                    closed = False
                    self._fail(error)
            elif self._process is not None and not self._initialization_attempted:
                # Popen succeeded but containment construction did not. No
                # initialization bytes were sent, so this exact child has not
                # been allowed to construct the backend or spawn its work.
                closed = await asyncio.to_thread(self._close_uninitialized_process)
            elif self._process is None and (
                self._start_task is None or self._start_task.done()
            ):
                self._physically_closed = True
            if (
                closed
                and self._process is not None
                and self._process.poll() is not None
            ):
                joined = True
                if self._channel is not None:
                    try:
                        joined = await asyncio.to_thread(
                            self._channel.close_after_exit, deadline
                        )
                    except BaseException as error:
                        joined = False
                        self._cleanup_errors.append(error)
                        self._fail(error)
                if self._monitor is not None and self._monitor.ident is not None:
                    try:
                        await asyncio.to_thread(
                            self._monitor.join,
                            None if deadline is None else max(0.0, deadline - time.monotonic()),
                        )
                        joined = joined and not self._monitor.is_alive()
                    except BaseException as error:
                        joined = False
                        self._cleanup_errors.append(error)
                        self._fail(error)
                pipes_closed = await asyncio.to_thread(self._close_pipes)
                self._physically_closed = joined and pipes_closed
        self._disposed = (
            drained
            and self._physically_closed
            and self._channel_error is None
            and not self._cleanup_errors
        )
        if cancellation is not None:
            raise cancellation
        return self._disposed

    def _close_uninitialized_process(self) -> bool:
        """Retain partial startup through the original containment stop bounds."""
        process = self._process
        if process is None or self._initialization_attempted:
            return False
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(2.0)
            return process.poll() is not None
        except BaseException as error:
            self._cleanup_errors.append(error)
            return False

    def _close_pipes(self) -> bool:
        """Close independent pipes even after a partial channel/monitor start."""
        process = self._process
        if process is None:
            return True
        all_closed = True
        channel = self._channel
        for stream, owner in (
            (process.stdin, channel.writer if channel is not None else None),
            (process.stdout, channel.reader if channel is not None else None),
            (process.stderr, self._monitor),
        ):
            if stream is None:
                continue
            # A still-running stream owner remains registered. Closing a
            # buffered file under its lock could itself wait without a bound.
            if owner is not None and owner.is_alive():
                all_closed = all_closed and stream.closed
                continue
            try:
                stream.close()
            except BaseException as error:
                all_closed = False
                self._cleanup_errors.append(error)
                if self._loop is not None:
                    try:
                        self._loop.call_soon_threadsafe(self._fail, error)
                    except RuntimeError:
                        pass
            all_closed = all_closed and stream.closed
        return all_closed
