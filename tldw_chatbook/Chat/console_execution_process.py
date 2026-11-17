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
    received: asyncio.Future[None]
    applied: asyncio.Future[Mapping[str, Any]]
    retired: asyncio.Future[bool]
    potentially_committed: bool = False
    events: list[ConsoleWireMessage] = field(default_factory=list, repr=False)
    projection_message: ConsoleWireMessage | None = field(default=None, repr=False)


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
        loop: asyncio.AbstractEventLoop,
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
        if self._started:
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

    def close_after_exit(self, deadline: float) -> bool:
        """Join pipes only after the actual peer process has exited."""
        self._closing.set()
        self.outbox.put_nowait(None)
        for thread in self._started:
            thread.join(max(0.0, deadline - time.monotonic()))
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
        self._ready: asyncio.Future[None] | None = None
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
        self._view: Any = None
        self._visit = 0
        self._projection = None
        self.projection_view = None
        self.pressed_inputs = None
        self._projection_sequence = 0

    async def read_projection(self, session_id: str, incarnation_id: str):
        """Adopt one exact retired projection without constructing a local store."""
        from .console_execution_projection_result import (
            decode_projection_payload,
            decode_projection_view,
            projection_key,
        )

        previous = self._projection
        view, visit = self._view, self._visit
        binding = (self._source_revision, session_id, incarnation_id)
        known = (
            previous.display_revision
            if previous is not None and projection_key(previous)[:3] == binding
            else None
        )
        receipt = self.submit(
            "ReadProjection",
            {"known_display_revision": known},
            session_id=session_id,
            incarnation_id=incarnation_id,
        )
        await asyncio.shield(receipt.applied)
        if await asyncio.shield(receipt.retired) is not True:
            raise ConsoleChannelLost("Console projection retirement is unproven")
        if (
            self._channel_error is not None
            or self._source_revision != binding[0]
            or self._view is not view
            or self._visit != visit
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
        self._projection = value
        self.projection_view = projection_view
        self.pressed_inputs = payload["session_view"]["pressed_inputs"]
        self._projection_sequence = receipt.identity.sequence
        return value

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
            self._profile = profile
            self._loop = loop
            self._ready = loop.create_future()
            self._start_task = loop.create_task(self._start())
        await asyncio.shield(self._start_task)

    async def _start(self) -> None:
        try:
            # Creation is retained independently of observer cancellation. No
            # native/backend work starts until containment is established and
            # the fixed initialization frame is transmitted.
            await asyncio.to_thread(self._launch)
            assert self._ready is not None
            await self._ready
        except BaseException as error:
            self._fail(error)
            raise

    def _launch(self) -> None:
        from tldw_chatbook.STT.executor_process_tree import (
            ExecutorProcessTree,
            WorkerContainmentIdentity,
        )
        from tldw_chatbook.Tools.workspace_tool_executor import _PopenProcessAdapter

        assert self._profile is not None and self._loop is not None
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
        self._channel.start()

        def monitor() -> None:
            # Drain diagnostics continuously so stderr capacity cannot stop the
            # original backend. Do not expose sensitive payloads in the client.
            assert process.stderr is not None
            while process.stderr.read(65536):
                pass
            process.wait()
            try:
                self._loop.call_soon_threadsafe(self._exited, process.returncode)
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
            self._loop.create_future(),
            self._loop.create_future(),
            self._loop.create_future(),
        )
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
        # The actual projection adapter is installed with the selected UI route;
        # retention here never creates a parent runtime, store or command proof.

    def _fail(self, error: BaseException) -> None:
        if self._channel_error is not None:
            return
        self._channel_error = error
        self._intake_open = False
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(ConsoleChannelLost("Console startup failed"))
        for receipt in self._requests.values():
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

    async def shutdown(self, *, deadline: float) -> bool:
        """Retain original drain and process cleanup across observer cancellation.

        Args:
            deadline: The initiating lifecycle owner's existing monotonic bound.

        Returns:
            Whether backend drain, process exit, pipes, monitor and containment
            all positively settled. Forced exit alone never returns success.
        """
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown(deadline))
        return await asyncio.shield(self._shutdown_task)

    async def _shutdown(self, deadline: float) -> bool:
        drained = False
        cancellation = None
        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                if self._start_task is not None:
                    await asyncio.shield(self._start_task)
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
                        max(0.0, deadline - time.monotonic()),
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
            # A timeout of the observer does not cancel the off-loop creator.
            # Keep this cleanup Task until it can see every resource that the
            # exact startup actually published, including partial construction.
            if self._start_task is not None and not self._start_task.done():
                try:
                    await asyncio.shield(self._start_task)
                except BaseException as error:
                    self._fail(error)
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
                            max(0.0, deadline - time.monotonic()),
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
                try:
                    self._loop.call_soon_threadsafe(self._fail, error)
                except RuntimeError:
                    pass
            all_closed = all_closed and stream.closed
        return all_closed
