"""Ordered local controls over one existing Console graph.

This module has no transport or default application route. Original runtime
custody owns every submitted turn; a control receipt is not native retirement.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from contextlib import contextmanager
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from .console_configuration_preparation import ConsoleTurnCaptureSelection
from .console_received_intent import ConsoleReceivedTurnIntent
from .console_session_settings import ConsoleSessionSettings
from .console_settings_apply import ConsoleSettingsSubmission
from .console_control_decisions import Decision, resolve_original_decision
from .console_control_permissions import (
    PERMISSION_OPERATION_TYPES,
    PermissionOperation,
    apply_permission_operation,
)
from .console_control_publication import (
    ApplySourceRevision,
    ConsoleConfigurationPublication,
    ConsoleWorkspacePublication,
    RefreshConfiguration,
    ReplaceConfiguration,
    SetWorkspaceBindingAccess,
    publish_original_source,
    require_publication_definitions,
)


_command_source = contextvars.ContextVar("console_control_command_source", default=None)


def require_control_source(runtime):
    """Retain the received command's actual source across original workers."""
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    source = _command_source.get()
    if (
        type(source) is not _CommandSource
        or source.controls is not controls
        or controls._runtime is not runtime
        or controls._controller is not runtime._chat_controller
        or controls._store is not runtime._chat_store
    ):
        raise RuntimeError("Selected Console effect lost its original command source")
    return source


@dataclass(frozen=True, slots=True)
class _CommandSource:
    controls: Any
    identity: Any
    sources: Any
    effects: list = field(default_factory=list)
    turn_retirement: Any = None


@contextmanager
def bind_received_control(runtime, record):
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        yield
        return
    entry = controls._turns.get(record.turn_id)
    if (
        entry is None
        or entry[1] is not record
        or entry[2] is not asyncio.current_task()
    ):
        raise RuntimeError("Received control custody changed")
    retirement = controls._turn_retirements.get(record.turn_id)
    source = _CommandSource(
        controls, entry[0], entry[3], turn_retirement=retirement
    )
    if retirement is not None:
        retirement.source = source
    token = _command_source.set(source)
    try:
        yield
    finally:
        if retirement is not None:
            retirement.source_closed = True
        _command_source.reset(token)


@dataclass(frozen=True, slots=True)
class ControlIdentity:
    profile_epoch: str
    service_epoch: str
    sequence: int
    request_id: str
    session_id: str = ""
    incarnation_id: str = ""
    source_revision: int = 0

    def __post_init__(self) -> None:
        for name in (
            "profile_epoch",
            "service_epoch",
            "request_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value) > 256:
                raise ValueError(f"Invalid control {name}")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("Control sequence must be positive")
        if type(self.source_revision) is not int or self.source_revision < 0:
            raise ValueError("Source revision must be non-negative")
        if (
            type(self.session_id) is not str
            or type(self.incarnation_id) is not str
            or bool(self.session_id) != bool(self.incarnation_id)
            or len(self.session_id) > 256
            or len(self.incarnation_id) > 256
        ):
            raise ValueError("Session and incarnation must be supplied together")


@dataclass(frozen=True, slots=True)
class UpdateInputs:
    identity: ControlIdentity
    draft: str = field(repr=False)
    authored_token: tuple[int, int]
    expected_settings_revision: int | None = None
    expected_settings: ConsoleSessionSettings | None = field(default=None, repr=False)
    expected_attachment_ids: tuple[str, ...] | None = None
    remove_attachment_ids: tuple[str, ...] = ()
    prefill: PrefillUpdate | None = field(default=None, repr=False)

    def __post_init__(self):
        if (
            type(self.draft) is not str
            or type(self.authored_token) is not tuple
            or len(self.authored_token) != 2
            or any(type(value) is not int or value < 0 for value in self.authored_token)
        ):
            raise ValueError("Invalid authored Console draft")
        if self.expected_settings_revision is not None and (
            type(self.expected_settings_revision) is not int
            or self.expected_settings_revision < 0
        ):
            raise ValueError("Invalid expected settings revision")
        if (
            self.expected_settings is not None
            and type(self.expected_settings) is not ConsoleSessionSettings
        ):
            raise TypeError("Expected settings must be the original settings value")
        for values in (self.expected_attachment_ids, self.remove_attachment_ids):
            if values is not None and (
                type(values) is not tuple
                or any(type(value) is not str or not value for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("Invalid attachment identity selection")
        if self.prefill is not None and type(self.prefill) is not PrefillUpdate:
            raise TypeError("Invalid prefill update")


@dataclass(frozen=True, slots=True)
class PrefillUpdate:
    expected_revision: int
    value: str | None = field(repr=False)

    def __post_init__(self):
        if (
            type(self.expected_revision) is not int
            or self.expected_revision < 0
            or self.value is not None
            and type(self.value) is not str
        ):
            raise ValueError("Invalid Console prefill revision or value")


@dataclass(frozen=True, slots=True)
class ReceiveTurn:
    identity: ControlIdentity
    turn_id: str
    draft_revision: int
    selection: ConsoleTurnCaptureSelection = field(repr=False)
    expected_settings_revision: int | None = None
    expected_attachment_revision: int | None = None
    expected_prefill_revision: int | None = None
    # Detached transport values add equality checks at the original locked
    # capture. They are not a substitute snapshot, stash or source capability.
    wire_inputs: object | None = field(default=None, repr=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class ApplySettings:
    """Apply the exact accepted/rebased submission through its original phases."""

    identity: ControlIdentity
    submission: ConsoleSettingsSubmission = field(repr=False)


@dataclass(frozen=True, slots=True)
class OpenSession:
    identity: ControlIdentity
    title: str = "New chat"
    workspace_id: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True, slots=True)
class CloseSession:
    identity: ControlIdentity
    expected_revision: int


@dataclass(frozen=True, slots=True)
class ResolveDecision:
    identity: ControlIdentity
    decision: Decision


@dataclass(frozen=True, slots=True)
class ReadProjection:
    identity: ControlIdentity


@dataclass(frozen=True, slots=True)
class ConsoleProjection:
    source_revision: int
    session_id: str
    incarnation_id: str
    display_revision: int
    messages: tuple = field(repr=False)
    run_state: Any
    decision: Any = field(repr=False)
    marker_revision: int | None = None


@dataclass(frozen=True, slots=True)
class CancelTurn:
    identity: ControlIdentity
    turn_id: str


@dataclass(frozen=True, slots=True)
class PermissionAction:
    identity: ControlIdentity
    operation: PermissionOperation


@dataclass(frozen=True, slots=True)
class FenceSource:
    identity: ControlIdentity
    source: str


@dataclass(frozen=True, slots=True, eq=False)
class SourceFence:
    source: str
    revision: int
    previous_sources: Any = field(repr=False)
    _issuer: Any = field(repr=False)


@dataclass(frozen=True, slots=True)
class CloseIntake:
    identity: ControlIdentity


@dataclass(frozen=True, slots=True)
class Drain:
    identity: ControlIdentity


@dataclass(frozen=True, slots=True)
class Shutdown:
    identity: ControlIdentity


@dataclass(frozen=True, slots=True)
class CommitPlanAck:
    identity: ControlIdentity
    received_request_id: str
    turn_id: str
    ordinal: int
    fingerprint: Any


@dataclass(frozen=True, slots=True)
class CommitPlanned:
    received_request_id: str
    turn_id: str
    ordinal: int
    fingerprint: Any
    recovery: Any


@dataclass(slots=True)
class _PendingPlan:
    projection: CommitPlanned
    plan: Any
    task: asyncio.Task | None
    acknowledged: asyncio.Future | None
    source_revision: int
    publication_order: int
    state: str = "waiting"


@dataclass(slots=True, eq=False)
class _ControlReservation:
    """Exact reservation outcome retained until its original owner releases."""

    controls: Any
    owner: Any
    result: list
    effect: Any = None
    pending: Any = None
    writer_started: bool = False
    native_retired: bool = False
    cancel_requested: bool = False


@dataclass(slots=True, eq=False)
class _FiniteEffect:
    kind: str
    source_revision: int
    owner: object
    completed: Future[bool]
    physically_retired: bool = False


@dataclass(slots=True)
class ControlReceipt:
    """Applied is a control transition; work retains its original Task."""

    identity: ControlIdentity
    applied: asyncio.Future[Any]
    work: asyncio.Future[Any] | None = None
    retired: asyncio.Future[bool] | None = None
    turn_retirement: Any = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _DisplayEffectOwner:
    full: object
    selection: object


class ConsoleLocalControls:
    """Dispatch finite commands without waiting for a turn's custody Task."""

    def __init__(self, service, *, profile_epoch: str, service_epoch: str) -> None:
        from .console_chat_controller import ConsoleChatController
        from .console_chat_store import ConsoleChatStore
        from .console_runtime import ConsoleRuntime

        runtime = service._runtime
        domain = service.domain_services
        if (
            type(runtime) is not ConsoleRuntime
            or domain is None
            or runtime.domain_services is not domain
            or runtime._execution_service is not service
            or type(runtime._chat_controller) is not ConsoleChatController
            or type(runtime._chat_store) is not ConsoleChatStore
            or runtime._chat_controller.store is not runtime._chat_store
        ):
            raise TypeError(
                "Local controls require the existing complete Console graph"
            )
        if any(
            type(epoch) is not str or not epoch
            for epoch in (profile_epoch, service_epoch)
        ):
            raise ValueError("Local control epochs must be nonempty strings")
        domain.require_current()
        self._service = service
        self._runtime = runtime
        self._domain = domain
        self._store = runtime._chat_store
        self._controller = runtime._chat_controller
        from .console_provider_gateway import ConsoleProviderGateway

        self._gateway = self._controller.provider_gateway
        if (
            type(self._gateway) is not ConsoleProviderGateway
            or getattr(self._gateway, "_console_local_controls", None) is not None
        ):
            raise TypeError(
                "Local controls require the original unbound provider gateway"
            )
        self._host = self._controller._interrupt_host
        self._permissions = service.ensure_permission_operations()
        if (
            self._permissions._controller is not self._controller
            or self._permissions._host is not self._host
        ):
            raise RuntimeError(
                "Permission operations were constructed before the controller"
            )
        self.profile_epoch = profile_epoch
        self.service_epoch = service_epoch
        self._loop = asyncio.get_running_loop()
        self._thread = threading.get_ident()
        self._lock = threading.Lock()
        self._sequence = 0
        # Exact seen identities remain retained for this service epoch; cleanup
        # controls must not acquire an arbitrary lifetime command ceiling.
        self._requests: set[str] = set()
        self._source_revision = 0
        self._intake_open = True
        self._effects_open = True
        self._turns: dict[str, tuple] = {}
        self._turn_retirements = {}
        self._turn_retirement_tasks = set()
        self._plans: dict[tuple[str, int], _PendingPlan] = {}
        self._fenced_turns: set[str] = set()
        self._effects: set[_FiniteEffect] = set()
        self._control_tasks: set[asyncio.Task] = set()
        self._settings_command_tasks = {}
        self._settings_readiness_waits = {}
        self._settings_readiness_captures = {}
        self._due_settings_reads = {}
        self._capture_predecessors = {}
        self._capture_tasks = {}
        self._session_capture_tails: dict[str, asyncio.Future] = {}
        self._parent_settings_captures = {}
        self._lifecycle_tasks: set[asyncio.Task] = set()
        self._mcp_submissions = set()
        self._tool_threads = set()
        self._trace_coordinators = set()
        self._intake_closed = False
        self._fence: SourceFence | None = None
        self._fence_drained = False
        self._publication_started = False
        self._publication_reads = set()
        self._publications = set()
        self._closing_sessions = set()
        from .console_control_index import bind_index_owner

        bind_index_owner(self, self._domain._project_index_owner)
        self._gateway._console_local_controls = self

    async def _read_projection(self, identity):
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
        from tldw_chatbook.UI.Console_Modules import composer_run_controls

        sources, revision = self._domain.sources, self.source_revision
        self._domain.require_current(sources)
        view = self._runtime.view
        association = self._domain.presentation.association
        self._domain.presentation.require_association(association)
        prepared_transcript = None
        if view is None:
            from .console_backend_presentation import ConsoleBackendPresentation

            if type(self._domain.presentation) is ConsoleBackendPresentation:
                await self._domain.presentation.prepare_view_checkpoint(identity)
            owner = self._runtime._execution_service.prepare_display_owner()
            owner.bind_original_graph()
            if owner._full is not None or owner.context.pending_key is not None:
                from .console_domain_display import ConsoleDisplayUnsupported

                raise ConsoleDisplayUnsupported("prior-display-still-owned")
            context = owner.context
            context.screen = context.publish = context.schedule = context.refresh = None
            context.config_projection = None
            context._display_view_stamp = None
            await owner.read()
            from .console_backend_presentation import ConsoleBackendPresentation

            if type(self._domain.presentation) is ConsoleBackendPresentation:
                await self._domain.presentation.prepare_readiness(identity)
                prepared_transcript = await self._domain.presentation.prepare_transcript(identity)
        else:
            if (
                type(view) is not ChatScreen
                or getattr(view, "_console_chat_store", None) is not self._store
                or getattr(view, "_console_chat_controller", None)
                is not self._controller
                or not composer_run_controls._preparing_methods_current(
                    view, (composer_run_controls._PREPARING_FULL_SYNC,)
                )
            ):
                raise RuntimeError(
                    "Native Console display requires its actual attached view"
                )
            association = self._domain.presentation.association
            self._domain.presentation.require_association(association)
            if not await view._sync_native_console_chat_ui():
                raise RuntimeError("Original Console FULL publication was deferred")
        self._domain.presentation.require_association(association)
        self._domain.require_current(sources)
        if self._runtime.view is not view or self.source_revision != revision:
            raise RuntimeError("Console projection attachment changed")
        with self._store._preparation_lock:
            session = self._store._sessions.get(identity.session_id)
            if session is None or session.incarnation_id != identity.incarnation_id:
                raise RuntimeError("Console projection session changed")
            messages = tuple(self._store.read_only_messages_for_session(session.id))
            marker_revision = None
            if prepared_transcript is not None:
                messages, marker_revision = self._domain.presentation.project_transcript(
                    identity, prepared_transcript, messages
                )
            projection = ConsoleProjection(
                revision,
                session.id,
                session.incarnation_id,
                self._store.display_projection_revision(session.id),
                messages,
                self._controller.run_state_for(session.id),
                self._controller.pending_decision_projection(session.id),
                marker_revision,
            )
            from .console_backend_presentation import ConsoleBackendPresentation

            if type(self._domain.presentation) is ConsoleBackendPresentation:
                self._domain.presentation.capture_projection_view(
                    self, identity, projection
                )
            return projection

    def _require_source_publication(self, fence):
        with self._lock:
            if (
                type(fence) is not SourceFence
                or self._fence is not fence
                or fence._issuer is not self
                or not self._fence_drained
                or self._intake_open
                or self._intake_closed
                or fence.revision != self._source_revision
                or self._domain.sources is not fence.previous_sources
            ):
                raise RuntimeError("Console source publication fence changed")
        self._domain.require_current(fence.previous_sources)

    def _begin_publication_effect(self, fence):
        self._require_source_publication(fence)
        with self._lock:
            if self._effects:
                raise RuntimeError("Prior finite effects have not retired")
            effect = _FiniteEffect(
                "source_publication", fence.revision, fence, Future()
            )
            self._effects.add(effect)
        return effect

    def _require_publication(self, publication):
        if (
            type(publication)
            not in (ConsoleConfigurationPublication, ConsoleWorkspacePublication)
            or publication._issuer is not self
            or publication not in self._publications
        ):
            raise RuntimeError("Console source publication was not issued here")
        self._require_source_publication(publication.fence)
        require_publication_definitions()
        if self._effects:
            raise RuntimeError("Source publication has not physically retired")
        if type(publication) is ConsoleConfigurationPublication:
            from tldw_chatbook import config

            if config.current_config_identity() != publication.config_identity:
                raise RuntimeError("Configuration publication identity changed")

    @property
    def source_revision(self) -> int:
        with self._lock:
            return self._source_revision

    def _begin_effect(self, kind, owner, source):
        if type(source) is not _CommandSource or source.controls is not self:
            raise RuntimeError("Finite effect requires its original control source")
        with self._lock:
            if (
                not self._effects_open
                or source.identity.source_revision != self._source_revision
                or self._domain.sources is not source.sources
            ):
                raise RuntimeError("Console finite effect source is fenced")
            effect = _FiniteEffect(kind, self._source_revision, owner, Future())
            self._effects.add(effect)
            source.effects.append(effect)
        try:
            self._domain.require_current(source.sources)
        except BaseException:
            self._finish_effect(effect, retired=True)
            raise
        return effect

    def _finish_effect(self, effect, *, retired):
        with self._lock:
            if effect not in self._effects or effect.completed.done():
                raise RuntimeError("Finite effect retirement owner changed")
            effect.physically_retired = retired
            if retired:
                self._effects.remove(effect)
            else:
                self._intake_open = False
                self._effects_open = False
        effect.completed.set_result(retired)

    def _begin_parent_settings_effect(self, capture):
        from .console_parent_settings import (
            ParentSettingsDeferred,
            require_parent_settings_capture,
        )

        with self._lock:
            if (
                not self._intake_open
                or not self._effects_open
                or capture.source_revision != self._source_revision
                or capture.sources is not self._domain.sources
            ):
                raise ParentSettingsDeferred("Parent settings source is fenced")
            if capture in self._parent_settings_captures:
                raise RuntimeError("Parent settings capture was already admitted")
            self._parent_settings_captures[capture] = (
                capture.port,
                capture.stamp,
                capture.sources,
                capture.session,
                capture.checked_config,
                capture.source_revision,
                capture.session_id,
                capture.incarnation_id,
                capture.purpose,
            )
            try:
                require_parent_settings_capture(capture, self)
                effect = _FiniteEffect(
                    "parent_settings", self._source_revision, capture, Future()
                )
                self._effects.add(effect)
                capture.effects.append(effect)
            except BaseException:
                del self._parent_settings_captures[capture]
                raise
        return effect

    @contextmanager
    def permission_effect(self, operations, operation, *, _source=None):
        if (
            operations is not self._permissions
            or operation.source is not operations._store
        ):
            raise RuntimeError("Permission effect owner changed")
        source = _command_source.get() if _source is None else _source
        effect = self._begin_effect("permission", operation, source)
        try:
            yield
        finally:
            self._finish_effect(effect, retired=operation.retired)

    def _retain_control_work(self, receipt, work, *, source=None, before_entry=None):
        entered = False

        async def execute():
            nonlocal entered
            entered = True
            task = asyncio.current_task()
            if type(source) is _CommandSource:
                self._settings_command_tasks[task] = source
            try:
                value = await work
            except BaseException as error:
                if not receipt.applied.done():
                    receipt.applied.set_exception(error)
            else:
                if not receipt.applied.done():
                    receipt.applied.set_result(value)
            finally:
                self._settings_command_tasks.pop(task, None)
                if receipt.retired is not None and (
                    type(receipt) is not ControlReceipt or receipt.turn_retirement is None
                ):
                    receipt.retired.set_result(
                        all(
                            effect.completed.done() and effect.physically_retired
                            for effect in source.effects
                        )
                    )

        if source is None:
            source = _CommandSource(self, receipt.identity, self._domain.sources)
        from .console_parent_settings import ParentSettingsCapture

        token = _command_source.set(
            None if type(source) is ParentSettingsCapture else source
        )
        execution = execute()
        try:
            task = self._loop.create_task(execution)
        except BaseException:
            execution.close()
            work.close()
            if before_entry is not None:
                before_entry()
                receipt.applied.cancel()
                if receipt.retired is not None:
                    receipt.retired.set_result(True)
            raise
        finally:
            if token is not None:
                _command_source.reset(token)
        self._control_tasks.add(task)

        def completed(finished):
            self._control_tasks.discard(finished)
            if not entered and finished.cancelled() and before_entry is not None:
                work.close()
                before_entry()
                if not receipt.applied.done():
                    receipt.applied.cancel()
                if receipt.retired is not None and not receipt.retired.done():
                    receipt.retired.set_result(True)

        task.add_done_callback(completed)
        receipt.work = task

    def _queue_session_capture(self, command, receipt):
        """Retain each ordered capture while leaving control intake available."""
        from .console_control_settings import PendingSettingsApply
        from .console_parent_settings import (
            ParentSettingsCapture,
            finish_parent_settings_capture,
            require_parent_settings_capture,
            run_parent_settings_capture,
        )

        parent = type(command) is ParentSettingsCapture
        if parent:
            require_parent_settings_capture(command, self)
        identity = command if parent else command.identity
        previous = self._session_capture_tails.get(identity.session_id)
        completion = self._loop.create_future()
        source = (
            command if parent else _CommandSource(self, identity, self._domain.sources)
        )
        operation = None
        if type(command) is ApplySettings:
            if (
                type(command.submission) is not ConsoleSettingsSubmission
                or command.submission.origin.session_id != identity.session_id
            ):
                raise TypeError("Settings require the exact session submission")
            operation = PendingSettingsApply(self, command, source)
            receipt.retired = self._loop.create_future()
        self._session_capture_tails[identity.session_id] = completion
        self._capture_predecessors[completion] = previous

        def finish_tail():
            if not completion.done():
                completion.set_result(None)
            self._capture_predecessors.pop(completion, None)
            if self._session_capture_tails.get(identity.session_id) is completion:
                del self._session_capture_tails[identity.session_id]

        def finish_unentered_tail():
            # A canceled capture never overtakes its predecessor. Keep its
            # existing completion in the chain without starting another Task.
            if previous is not None and not previous.done():
                previous.add_done_callback(lambda _finished: finish_tail())
            else:
                finish_tail()

        def finish_operation():
            if parent:
                finish_parent_settings_capture(command)
                return
            if operation is None:
                return
            operation.admission.release()
            operation.lifecycle.pending_live.discard(operation.finished)
            if not operation.finished.done():
                operation.finished.set_result(None)
            self._finish_effect(operation.effect, retired=operation.retired)

        def before_entry():
            finish_unentered_tail()
            finish_operation()

        capture_entered = False
        parent_started = self._loop.create_future() if parent else None

        async def capture():
            nonlocal capture_entered
            if parent_started is not None:
                await parent_started
            capture_entered = True
            task = asyncio.current_task()
            self._capture_tasks[task] = completion
            try:
                if previous is not None:
                    if parent and command.purpose == "readiness-defaults":
                        from .console_parent_settings import wait_readiness_predecessor

                        await wait_readiness_predecessor(command, previous)
                    else:
                        await asyncio.shield(previous)
                self._require_graph()
                self._domain.require_current(source.sources)
                with self._store._preparation_lock:
                    session = self._store._sessions.get(identity.session_id)
                    if (
                        session is None
                        or session.incarnation_id != identity.incarnation_id
                    ):
                        raise RuntimeError("Console control session changed")
                if operation is not None:
                    return await operation.apply()
                if identity.source_revision != self.source_revision:
                    if parent:
                        from .console_parent_settings import ParentSettingsDeferred

                        raise ParentSettingsDeferred(
                            "Parent settings source is fenced before entry"
                        )
                    raise RuntimeError("Console control intake is fenced")
                if parent:
                    return await run_parent_settings_capture(command)
                if type(command) is ReadProjection:
                    return await self._read_projection(identity)
                self._apply_input_command(command, receipt)
                return receipt.applied.result()
            finally:
                self._capture_tasks.pop(task, None)
                finish_unentered_tail()

        async def retained_capture():
            if parent:
                task = None
                work = capture()
                try:
                    task = self._loop.create_task(work)
                    command.operation_task = task
                    self._parent_settings_captures[command] += (task,)
                    self._control_tasks.add(task)
                    if not parent_started.done():
                        parent_started.set_result(None)
                    cancelled = False
                    while not task.done():
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            cancelled = True
                        except BaseException:
                            break
                    value = task.result()
                    if cancelled:
                        raise asyncio.CancelledError
                    return value
                finally:
                    if task is None:
                        work.close()
                        finish_unentered_tail()
                    else:
                        self._control_tasks.discard(task)
                        if not capture_entered:
                            finish_unentered_tail()
                    finish_operation()
            if operation is None:
                return await capture()
            try:
                task = operation.durability.launch(
                    operation.admission,
                    capture(),
                    name="console-settings-live-apply",
                )
                operation.lifecycle.tasks.add(task)
                cancelled = False
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        cancelled = True
                    except BaseException:
                        break
                value = task.result()
                if cancelled:
                    raise asyncio.CancelledError
                return value
            finally:
                if "task" in locals():
                    operation.lifecycle.tasks.discard(task)
                    if not capture_entered:
                        finish_unentered_tail()
                else:
                    finish_unentered_tail()
                finish_operation()

        self._retain_control_work(
            receipt,
            retained_capture(),
            source=source,
            before_entry=before_entry,
        )

    async def _drain_finite_effects(self):
        with self._lock:
            effects = tuple(self._effects)
        for effect in effects:
            if not await asyncio.shield(asyncio.wrap_future(effect.completed)):
                raise RuntimeError("Console native effect retirement is unproven")

    async def _finish_source_fence(self, fence):
        await self._drain_finite_effects()
        with self._lock:
            if self._fence is not fence:
                raise RuntimeError("Console source fence changed")
            self._fence_drained = True
        return fence

    async def _drain(self, *, shutdown=False):
        from .console_control_trace import drain_trace_worker, require_trace_outcomes

        try:
            await self._drain_owned()
        finally:
            if shutdown:
                # An unresolved outcome must not bypass original store teardown.
                await self._runtime.dispose()
                await drain_trace_worker(self._store)
        require_trace_outcomes(self)
        return True

    async def _close_session(self, command, session):
        identity = command.identity
        try:
            return await self._runtime.close_session(
                identity.session_id, expected_revision=command.expected_revision
            )
        finally:
            with self._store._preparation_lock:
                current = self._store._sessions.get(identity.session_id)
                same_incarnation = (
                    current is session
                    and current.incarnation_id == identity.incarnation_id
                )
                fenced = (
                    identity.session_id in self._runtime._admission_fenced_sessions
                    or identity.session_id
                    in self._controller._session_close_generations
                    or identity.session_id
                    in self._controller._failed_session_close_generations
                )
                # Only original irreversible/recovery fences survive refusal.
                # A deleted/replaced row cannot clear those original fences.
                if not fenced and (same_incarnation or current is not session):
                    self._closing_sessions.discard(identity.session_id)

    async def _drain_owned(self):
        self._runtime.cancel_received_intents()
        for session in self._store.sessions():
            self._controller._signal_stop(session_id=session.id, reason="shutdown")
        with self._lock:
            tasks = tuple(
                retirement.task
                for retirement in self._turn_retirements.values()
                if retirement.task is not None
            )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(
                task
                for task in tuple(self._control_tasks)
                if task is not asyncio.current_task()
                and task not in self._lifecycle_tasks
            ),
            return_exceptions=True,
        )
        from .console_backend_presentation import ConsoleBackendPresentation

        if type(self._domain.presentation) is ConsoleBackendPresentation:
            await self._domain.presentation.drain_view_lifecycle()
        await self._runtime._drain_ordinary_native_commits(self._controller)
        await self._runtime._drain_ordinary_turns()
        from .console_preparation_reads import drain_preparation_reads

        await drain_preparation_reads(self._runtime._preparation_reads)
        from .console_control_tools import drain_tool_threads

        await drain_tool_threads(self)
        owner = self._domain._project_index_owner
        if owner is not None:
            await owner.aclose()
        with self._lock:
            submissions = tuple(self._mcp_submissions)
        for submission in submissions:
            if not await asyncio.shield(asyncio.wrap_future(submission.retired)):
                raise RuntimeError("Console MCP native retirement is unproven")
        from .console_control_trace import drain_trace_worker

        await drain_trace_worker(self._store)
        await self._drain_finite_effects()
        await asyncio.gather(
            *tuple(self._turn_retirement_tasks), return_exceptions=True
        )
        if self._turn_retirements:
            raise RuntimeError("Console turn native retirement is unproven")
        if self._permissions._unresolved:
            raise RuntimeError("Permission retirement remains unresolved")

    @property
    def commit_plans(self) -> tuple[CommitPlanned, ...]:
        """Return only bounded body-free projections of original sealed plans."""
        with self._lock:
            waiting = (
                self._plans[key]
                for retirement in self._turn_retirements.values()
                for key in retirement.plan_keys
                if self._plans[key].state == "waiting"
            )
            return tuple(
                item.projection
                for item in sorted(waiting, key=lambda item: item.publication_order)
            )

    @property
    def admitted_effects(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(item.kind for item in self._effects)

    async def commit_received_turn(self, owner, acceptance):
        """Pause the original submit Task after sealing and before native entry."""
        from .console_chat_store import ConsoleDurableTurnCommit
        from .console_durable_commit_plan import capture_durable_plan_source
        from .console_native_commit import (
            ConsoleNativeCommitCompletion,
            commit_reserved_durable_turn_owned,
            reserve_durable_commit_plan_owned,
        )

        self._require_graph()
        source = capture_durable_plan_source(self._store)
        turn_id = source.record.turn_id
        entry = self._turns.get(turn_id)
        if (
            entry is None
            or entry[1] is not source.record
            or source.task is not asyncio.current_task()
            or owner.task is not source.task
            or owner.acceptance is not acceptance
        ):
            raise RuntimeError("Commit does not belong to the received control request")
        identity = entry[0]
        invocation = self._turn_retirements.get(turn_id)
        if (
            invocation is None
            or invocation.identity is not identity
            or invocation.record is not source.record
            or invocation.task is not source.task
        ):
            raise RuntimeError("Console commit retirement owner changed")
        reservation = _ControlReservation(self, owner, [])
        reservation.effect = self._begin_effect(
            "commit_reservation", reservation, _command_source.get()
        )
        owner.control_reservation = reservation
        plan = await reserve_durable_commit_plan_owned(
            self._store, acceptance, _retained_result=reservation.result
        )
        if type(plan) is ConsoleDurableTurnCommit:
            return ConsoleNativeCommitCompletion(plan, None, False, True)
        pending = None
        with self._lock:
            if (
                not self._effects_open
                or identity.source_revision != self._source_revision
                or turn_id in self._fenced_turns
                or plan.recovery is None
            ):
                raise RuntimeError("Console commit source is fenced")
            ordinal = len(invocation.plan_keys) + 1
            projection = CommitPlanned(
                identity.request_id,
                turn_id,
                ordinal,
                plan.fingerprint,
                plan.recovery,
            )
            pending = _PendingPlan(
                projection,
                plan,
                source.task,
                self._loop.create_future(),
                identity.source_revision,
                len(self._plans),
            )
            self._plans[turn_id, ordinal] = pending
            invocation.plan_keys.append((turn_id, ordinal))
            reservation.pending = pending
        from .console_backend_presentation import ConsoleBackendPresentation

        presentation = self._domain.presentation
        if type(presentation) is ConsoleBackendPresentation:
            presentation.publish_commit_plan(self, pending)
        await pending.acknowledged
        plan.source.require_current(plan.preparation)
        with self._lock:
            if pending.state != "acknowledged" or turn_id in self._fenced_turns:
                raise RuntimeError("Console commit plan is no longer pending")
            pending.state = "started"
            reservation.writer_started = True
        completion = await commit_reserved_durable_turn_owned(plan, _native_owner=owner)
        with self._lock:
            pending.state = "finished" if completion.native_retired else "uncertain"
            reservation.native_retired = completion.native_retired
        return completion

    def _settle_released_reservation(self, owner) -> None:
        """Finish only the retained plan after original owner exclusion ends."""
        from .console_chat_store import ConsoleDurableTurnCommit
        from .console_durable_commit_plan import (
            _ConsoleDurableCommitPlan,
            require_plan_methods,
        )

        reservation = owner.control_reservation
        retired = False
        try:
            if (
                type(reservation) is not _ControlReservation
                or reservation.controls is not self
                or reservation.owner is not owner
                or owner._issuer is not self._controller
                or owner.store is not self._store
                or owner.task is not asyncio.current_task()
                or self._controller._ordinary_native_commits.get(
                    owner.acceptance.preparation_id
                )
                is owner
                or self._store._native_commit_owners_by_preparation.get(
                    owner.acceptance.preparation_id
                )
                is owner
            ):
                raise RuntimeError("Console reservation release owner changed")
            if reservation.writer_started:
                retired = reservation.native_retired
                return
            if not reservation.result:
                retired = True  # Original reservation read joined without a plan.
                return
            if len(reservation.result) != 1:
                raise RuntimeError("Console reservation outcome changed")
            plan = reservation.result[0]
            if type(plan) is ConsoleDurableTurnCommit:
                retired = True
                return
            if (
                type(plan) is not _ConsoleDurableCommitPlan
                or plan.store is not owner.store
                or plan.session is not owner.session
                or plan.persistence is not owner.persistence
                or plan.database is not owner.database
                or plan.received_acceptance is not owner.acceptance
                or plan.source.task is not owner.task
                or (
                    reservation.pending is not None
                    and (
                        type(reservation.pending) is not _PendingPlan
                        or self._plans.get(
                            (
                                reservation.pending.projection.turn_id,
                                reservation.pending.projection.ordinal,
                            )
                        )
                        is not reservation.pending
                        or reservation.pending.plan is not plan
                        or reservation.pending.task is not owner.task
                        or reservation.pending.state
                        not in {"waiting", "acknowledged", "conflict"}
                    )
                )
            ):
                raise RuntimeError("Console reservation outcome owner changed")
            require_plan_methods(self._store, owner.persistence)
            if not self._store.abandon_reserved_durable_commit(
                plan, "control_wait_abandoned"
            ):
                raise RuntimeError("Console reservation abandonment is unconfirmed")
            if reservation.cancel_requested:
                from tldw_chatbook.Agents.activation import callable_source_current
                from .console_chat_controller import _CONSOLE_PRECOMMIT_CANCEL_METHODS

                for is_store, name, definition in _CONSOLE_PRECOMMIT_CANCEL_METHODS:
                    receiver = self._store if is_store else self._controller
                    method = getattr(receiver, name, None)
                    if (
                        getattr(method, "__self__", None) is not receiver
                        or getattr(method, "__func__", None) is not definition[0]
                        or not callable_source_current(definition)
                    ):
                        raise RuntimeError("Console cancellation source changed")
                self._controller.cancel_library_preparation(
                    plan.acceptance.preparation_id
                )
                if (
                    self._controller._preparation_by_id(plan.acceptance.preparation_id)
                    is not None
                ):
                    raise RuntimeError("Console explicit cancellation is unconfirmed")
            if reservation.pending is not None:
                reservation.pending.state = "abandoned"
            retired = True
        except Exception as error:
            # Keep the original caller cancellation/error; negative effect
            # settlement prevents any positive retirement receipt.
            owner.commit_error = error
        finally:
            self._finish_effect(reservation.effect, retired=retired)
            if retired:
                owner.control_reservation = None

    @contextmanager
    def native_commit_effect(self, plan, retirement):
        """Linearize the actual worker immediately before original native preflight."""
        with self._lock:
            owner = self._controller._ordinary_native_commits.get(
                plan.acceptance.preparation_id
            )
            reservation = getattr(owner, "control_reservation", None)
            pending = getattr(reservation, "pending", None)
            if (
                type(reservation) is not _ControlReservation
                or reservation.controls is not self
                or reservation.owner is not owner
                or type(pending) is not _PendingPlan
                or self._plans.get(
                    (pending.projection.turn_id, pending.projection.ordinal)
                ) is not pending
                or pending.plan is not plan
                or pending.state != "started"
                or pending.task is not plan.source.task
                or pending.source_revision != self._source_revision
                or not self._effects_open
                or pending.projection.turn_id in self._fenced_turns
            ):
                # This worker has not entered the original native owner at all.
                retirement.confirmed = True
                raise RuntimeError("Console native effect source is fenced")
        try:
            effect = self._begin_effect("durable_commit", plan, _command_source.get())
        except BaseException:
            retirement.confirmed = True
            raise
        try:
            yield
        finally:
            self._finish_effect(effect, retired=retirement.confirmed)

    @contextmanager
    def native_write_effect(self, binding):
        source, kind = binding.control_phase
        if (
            kind not in {"terminal", "postcommit", "trace"}
            or source.controls is not self
            or binding.store is not self._store
            or binding.persistence is not self._store.persistence
            or binding.database is not self._domain.sources.chat_database
        ):
            raise RuntimeError("Console native write owner changed")
        try:
            effect = self._begin_effect(kind, binding, source)
        except BaseException:
            binding.retirement.confirmed = True  # Refused before original native entry.
            raise
        try:
            yield
        finally:
            self._finish_effect(effect, retired=binding.retirement.confirmed)

    def _require_graph(self) -> None:
        if (
            threading.get_ident() != self._thread
            or asyncio.get_running_loop() is not self._loop
        ):
            raise RuntimeError("Local control graph changed")
        self._require_graph_identity()

    def _require_graph_identity(self) -> None:
        """Check the same retained collaborators from an original owned worker."""
        if (
            self._runtime._execution_service is not self._service
            or self._service.domain_services is not self._domain
            or self._runtime.domain_services is not self._domain
            or self._runtime._chat_controller is not self._controller
            or self._runtime._chat_store is not self._store
            or self._controller.store is not self._store
            or self._controller.provider_gateway is not self._gateway
            or self._gateway._console_local_controls is not self
            or self._controller._interrupt_host is not self._host
            or self._runtime._disposed
        ):
            raise RuntimeError("Local control graph changed")

    def dispatch(self, command) -> ControlReceipt:
        """Apply an ordered input/control transition and retain original work."""
        if type(command) not in (
            UpdateInputs,
            ReceiveTurn,
            ApplySettings,
            CancelTurn,
            CommitPlanAck,
            PermissionAction,
            FenceSource,
            CloseIntake,
            Drain,
            Shutdown,
            ReplaceConfiguration,
            RefreshConfiguration,
            SetWorkspaceBindingAccess,
            ApplySourceRevision,
            OpenSession,
            CloseSession,
            ResolveDecision,
            ReadProjection,
        ):
            raise TypeError("Unsupported local Console command")
        self._require_graph()
        identity = command.identity
        if type(identity) is not ControlIdentity:
            raise TypeError("Control identity must be exact")
        with self._lock:
            if (
                identity.profile_epoch != self.profile_epoch
                or identity.service_epoch != self.service_epoch
                or identity.sequence <= self._sequence
                or identity.request_id in self._requests
            ):
                self._intake_open = False
                self._effects_open = False
                raise RuntimeError("Conflicting or out-of-order Console command")
            self._sequence = identity.sequence
            self._requests.add(identity.request_id)
            if type(command) not in (
                CancelTurn,
                CommitPlanAck,
                CloseIntake,
                Drain,
                Shutdown,
                CloseSession,
                ResolveDecision,
                ReplaceConfiguration,
                RefreshConfiguration,
                SetWorkspaceBindingAccess,
                ApplySourceRevision,
            ) and (
                not self._intake_open
                or identity.source_revision != self._source_revision
            ):
                raise RuntimeError("Console control intake is fenced")
        receipt = ControlReceipt(identity, self._loop.create_future())
        if type(command) is OpenSession:
            if identity.session_id:
                raise ValueError(
                    "OpenSession does not adopt a caller-provided incarnation"
                )
            self._domain.require_current()
            if command.conversation_id is not None:
                from .console_control_sessions import open_saved_session

                self._domain.conversation_scope_service()
                self._retain_control_work(receipt, open_saved_session(self, command))
                return receipt
            session = self._store.create_session(
                title=command.title,
                workspace_id=command.workspace_id,
                settings=self._domain.default_session_settings(),
            )
            receipt.applied.set_result((session.id, session.incarnation_id))
            return receipt
        if type(command) in (
            ReplaceConfiguration,
            RefreshConfiguration,
            SetWorkspaceBindingAccess,
        ):
            self._require_source_publication(command.fence)
            expected_kind = (
                "workspace"
                if type(command) is SetWorkspaceBindingAccess
                else "configuration"
            )
            if (
                command.fence.source != expected_kind
                or identity.source_revision != command.fence.revision
                or self._publication_started
            ):
                raise RuntimeError(
                    "Console publication command does not match its fence"
                )
            self._publication_started = True
            self._retain_control_work(receipt, publish_original_source(self, command))
            return receipt
        if type(command) is ApplySourceRevision:
            self._require_publication(command.publication)
            if identity.source_revision != self._source_revision:
                raise RuntimeError("Console publication revision changed")
            if type(command.publication) is ConsoleConfigurationPublication:
                self._domain.apply_configuration_publication(command.publication)
            with self._lock:
                self._publications.remove(command.publication)
                self._fence = None
                self._fence_drained = False
                self._publication_started = False
                self._intake_open = not self._intake_closed
                self._effects_open = True
            receipt.applied.set_result(self._source_revision)
            return receipt
        if type(command) is FenceSource:
            if command.source not in {"configuration", "workspace"}:
                raise ValueError("Unsupported Console source fence")
            with self._lock:
                self._intake_open = False
                self._effects_open = False
                self._source_revision += 1
                fence = SourceFence(
                    command.source, self._source_revision, self._domain.sources, self
                )
                self._fence = fence
                self._fence_drained = False
                self._publication_started = False
                for retirement in self._turn_retirements.values():
                    for key in retirement.plan_keys:
                        pending = self._plans[key]
                        if pending.state in {"waiting", "acknowledged"}:
                            pending.state = "conflict"
                            self._fenced_turns.add(pending.projection.turn_id)
                            pending.task.cancel()
            self._retain_control_work(receipt, self._finish_source_fence(fence))
            return receipt
        if type(command) in (CloseIntake, Drain, Shutdown):
            with self._lock:
                self._intake_open = False
                self._intake_closed = True
            if type(command) is CloseIntake:
                receipt.applied.set_result(True)
            else:
                self._retain_control_work(
                    receipt, self._drain(shutdown=type(command) is Shutdown)
                )
                self._lifecycle_tasks.add(receipt.work)
                receipt.work.add_done_callback(self._lifecycle_tasks.discard)
            return receipt
        if type(command) is PermissionAction:
            if type(command.operation) not in PERMISSION_OPERATION_TYPES:
                raise TypeError("Unsupported permission command")
            receipt.retired = self._loop.create_future()
            self._retain_control_work(
                receipt,
                apply_permission_operation(self._permissions, command.operation),
            )
            return receipt
        with self._store._preparation_lock:
            session = self._store._sessions.get(identity.session_id)
            if session is None or session.incarnation_id != identity.incarnation_id:
                raise RuntimeError("Console control session changed")
        if type(command) is CloseSession:
            if identity.session_id in self._closing_sessions:
                raise RuntimeError("Console session control intake is closing")
            self._closing_sessions.add(identity.session_id)
            self._retain_control_work(
                receipt,
                self._close_session(command, session),
            )
            return receipt
        if type(command) is ResolveDecision:
            receipt.applied.set_result(
                resolve_original_decision(self, identity, command.decision)
            )
            return receipt
        if type(command) is ReadProjection:
            receipt.retired = self._loop.create_future()
            if identity.session_id in self._session_capture_tails:
                self._queue_session_capture(command, receipt)
            else:
                self._retain_control_work(
                    receipt, self._read_projection(identity), before_entry=lambda: None
                )
            return receipt
        if type(command) is CommitPlanAck:
            with self._lock:
                pending = self._plans.get((command.turn_id, command.ordinal))
                if pending is None or pending.state in {
                    "abandoned",
                    "conflict",
                    "uncertain",
                }:
                    raise RuntimeError("Console commit plan is no longer pending")
                entry = self._turns.get(command.turn_id)
                if (
                    entry is None
                    or entry[0].session_id != identity.session_id
                    or entry[0].incarnation_id != identity.incarnation_id
                    or command.received_request_id
                    != pending.projection.received_request_id
                    or command.fingerprint != pending.projection.fingerprint
                ):
                    self._fenced_turns.add(command.turn_id)
                    pending.state = "conflict"
                    if pending.task is not None:
                        pending.task.cancel()
                    raise RuntimeError("Conflicting Console plan acknowledgement")
                if pending.state == "waiting":
                    pending.state = "acknowledged"
                    pending.acknowledged.set_result(None)
            receipt.applied.set_result(True)
            return receipt
        if type(command) is CancelTurn:
            entry = self._turns.get(command.turn_id)
            record = None if entry is None else entry[1]
            if (
                record is None
                or entry[0].session_id != identity.session_id
                or entry[0].incarnation_id != identity.incarnation_id
                or self._runtime._turn_custody.get(command.turn_id) is not record
            ):
                raise RuntimeError("Console cancellation owner changed")
            if record.received_claim is not None:
                self._store.seal_received_turn(record.received_claim)
            for owner in self._controller._ordinary_native_commits.values():
                reservation = owner.control_reservation
                if (
                    type(reservation) is _ControlReservation
                    and reservation.controls is self
                    and reservation.owner is owner
                    and owner.task is entry[2]
                    and owner.session.id == identity.session_id
                ):
                    reservation.cancel_requested = True
            self._controller._signal_stop(session_id=identity.session_id)
            entry[2].cancel()
            receipt.work = entry[2]
            retirement = self._turn_retirements.get(command.turn_id)
            if retirement is not None:
                receipt.turn_retirement = retirement
                receipt.retired = retirement.retired
            receipt.applied.set_result(True)
            return receipt
        self._domain.require_current()
        if identity.session_id in self._closing_sessions:
            raise RuntimeError("Console session control intake is closing")
        if (
            type(command) is ApplySettings
            or identity.session_id in self._session_capture_tails
        ):
            self._queue_session_capture(command, receipt)
            return receipt
        return self._apply_input_command(command, receipt)

    def _apply_input_command(self, command, receipt):
        identity = command.identity
        if type(command) is UpdateInputs:
            if type(command.draft) is not str:
                raise TypeError("Draft must be text")
            with self._store._preparation_lock:
                inputs = self._store.session_input_snapshot(identity.session_id)
                if (
                    command.expected_settings_revision is not None
                    and self._store.session_settings_revision(identity.session_id)
                    != command.expected_settings_revision
                    or command.expected_settings is not None
                    and inputs._settings != command.expected_settings
                    or command.expected_attachment_ids is not None
                    and inputs._attachment_ids != command.expected_attachment_ids
                    or not set(command.remove_attachment_ids).issubset(
                        inputs._attachment_ids
                    )
                    or command.prefill is not None
                    and inputs.prefill_revision != command.prefill.expected_revision
                ):
                    raise RuntimeError("Console input revision changed")
                self._store.set_session_draft(
                    identity.session_id,
                    command.draft,
                    authored_token=command.authored_token,
                )
                for attachment_id in command.remove_attachment_ids:
                    self._store.consume_pending_attachment(
                        identity.session_id, attachment_id
                    )
                if command.prefill is not None:
                    self._store.set_session_one_shot_prefill(
                        identity.session_id, command.prefill.value
                    )
            receipt.applied.set_result(
                self._store.session_input_snapshot(identity.session_id).draft_revision
            )
            return receipt
        with self._store._preparation_lock:
            inputs = self._store.session_input_snapshot(identity.session_id)
            if command.wire_inputs is not None:
                from .console_execution_protocol import ConsoleWireMessage
                from dataclasses import fields
                import base64

                wire = command.wire_inputs
                if (
                    type(wire) is not ConsoleWireMessage
                    or wire.kind != "command" or wire.operation != "ReceiveTurn"
                    or any(
                        getattr(wire.identity, item.name) != getattr(identity, item.name)
                        for item in fields(ControlIdentity)
                    )
                ):
                    raise RuntimeError("Remote pressed command association changed")
                pressed = wire.payload["pressed_inputs"]
                session = self._store._sessions[identity.session_id]
                settings = (
                    None if inputs._settings is None else {
                        item.name: getattr(inputs._settings, item.name)
                        for item in fields(ConsoleSessionSettings)
                    }
                )
                if (
                    pressed["session_id"] != inputs.session_id
                    or pressed["incarnation_id"] != inputs.incarnation_id
                    or pressed["draft"] != inputs.draft
                    or tuple(pressed["authored_token"]) != session._draft_authored_token
                    or pressed["settings"] != settings
                    or pressed["settings_revision"] != inputs.settings_revision
                    or pressed["attachment_revision"] != inputs.attachment_revision
                    or tuple(pressed["attachment_ids"]) != inputs._attachment_ids
                    or pressed["one_shot_prefill"] != inputs.one_shot_prefill
                    or pressed["prefill_revision"] != inputs.prefill_revision
                    or len(pressed["attachments"]) != len(inputs.attachments)
                ):
                    raise RuntimeError("Remote pressed values changed")
                for expected, actual in zip(pressed["attachments"], inputs.attachments):
                    for name, value in expected.items():
                        original = getattr(actual, name)
                        if name == "data" and original is not None:
                            original = base64.b64encode(original).decode("ascii")
                        if value != original:
                            raise RuntimeError("Remote pressed attachment changed")
            if inputs.draft_revision != command.draft_revision:
                raise RuntimeError("Pressed Console input changed")
            if (
                command.expected_settings_revision is not None
                and self._store.session_settings_revision(identity.session_id)
                != command.expected_settings_revision
                or command.expected_attachment_revision is not None
                and inputs.attachment_revision != command.expected_attachment_revision
                or command.expected_prefill_revision is not None
                and inputs.prefill_revision != command.expected_prefill_revision
            ):
                raise RuntimeError("Pressed Console input revision changed")
            if command.turn_id in self._turns:
                raise RuntimeError("Console turn identity was already received")
            launch, revision, _notice = self._runtime.snapshot_console_staged_evidence(
                identity.session_id
            )
            from .console_configuration_preparation import (
                capture_console_received_configuration_preparation,
            )
            from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

            preparation = capture_console_received_configuration_preparation(
                None, self._store, self._controller, session_id=identity.session_id
            )
            if preparation is None:
                raise RecoveryRequired("console_capture_source_not_ready")
            intent = ConsoleReceivedTurnIntent(
                turn_id=command.turn_id,
                session_id=identity.session_id,
                inputs=inputs,
                selection=command.selection,
                staged_evidence_launch=launch,
                staged_evidence_revision=revision,
                view_attachment_generation=self._runtime._attached_generation or 0,
                _pressed_inputs=inputs,
                recovery_user_origin=self._runtime.ensure_archive_recovery().issue_user_proof(
                    identity.session_id
                ),
            )
            from .console_execution_retirement import ConsoleTurnRetirement

            retirement = ConsoleTurnRetirement(self, identity, command.turn_id)
            self._turn_retirements[command.turn_id] = retirement
            try:
                turn_id = self._runtime.accept_received_intent(
                    intent,
                    _configuration_preparation=preparation,
                    terminal_callback=retirement.terminal,
                )
            except BaseException:
                # Original admission did not return a received claim. Do not
                # invent its positive receipt or physical outcome.
                self._turn_retirements.pop(command.turn_id, None)
                raise
            record = self._runtime._turn_custody[turn_id]
            self._turns[turn_id] = (identity, record, record.task, self._domain.sources)
            retirement.bind(record)
        receipt.work = record.task
        receipt.turn_retirement = retirement
        receipt.retired = retirement.retired
        receipt.applied.set_result(turn_id)
        return receipt

    def _begin_display_effect(self, full, selection):
        """Admit only this exact finite display, before producer exposure."""
        from .console_domain_display import ConsoleDisplayFull, _DisplaySelection
        from .console_display_phases import ConsoleDisplayPhaseRefused

        self._require_graph()
        if (
            type(self) is not ConsoleLocalControls
            or type(full) is not ConsoleDisplayFull
            or type(selection) is not _DisplaySelection
            or full.selection is not selection
            or full.owner is not selection.owner
            or asyncio.current_task() is not selection.task
        ):
            raise ConsoleDisplayPhaseRefused("Display admission owner changed")
        owner = selection.owner
        owner.require_current()
        source = selection.command_source
        with self._lock:
            if (
                not self._intake_open
                or not self._effects_open
                or selection.controls is not self
                or selection.source_revision != self._source_revision
                or selection.sources is not self._domain.sources
                or owner.domain is not self._domain
                or owner.runtime is not self._runtime
                or owner.store is not self._store
                or owner.controller is not self._controller
                or owner._full is not None
                or full.effect is not None
                or self._store._sessions.get(selection.session.id)
                is not selection.session
                or selection.session.incarnation_id != selection.incarnation
                or self._store.active_session_id != selection.session.id
                or selection.port is not self._domain.presentation
            ):
                raise ConsoleDisplayPhaseRefused("New display admission is fenced")
            if source is not None:
                if (
                    type(source) is not _CommandSource
                    or source.controls is not self
                    or self._settings_command_tasks.get(selection.task) is not source
                    or _command_source.get() is not source
                    or source.sources is not selection.sources
                    or source.identity.source_revision != selection.source_revision
                    or source.identity.session_id != selection.session.id
                    or source.identity.incarnation_id != selection.incarnation
                ):
                    raise ConsoleDisplayPhaseRefused("Display command custody changed")
            elif selection.stamp is None:
                raise ConsoleDisplayPhaseRefused(
                    "Detached display has no actual command"
                )
            if selection.stamp is not None:
                # Fixed parent memory/definition checks only; no await/native work.
                if not selection.port._current_visit(selection.stamp):
                    raise ConsoleDisplayPhaseRefused("Display selected visit changed")
                selection.port.require_display_full_entry(
                    selection.parent_entry,
                    owner,
                    selection.task,
                )
            elif (
                selection.parent_entry is not None
                or selection.port._view is not None
                or self._runtime.view is not None
            ):
                raise ConsoleDisplayPhaseRefused(
                    "Detached display has a false view entry"
                )
            effect = _FiniteEffect(
                "display",
                selection.source_revision,
                _DisplayEffectOwner(full, selection),
                Future(),
            )
            try:
                self._effects.add(effect)
                if source is not None:
                    source.effects.append(effect)
                owner._full = full
                full.effect = effect
                if selection.parent_entry is not None:
                    selection.parent_entry.full = full
            except BaseException:
                self._effects.discard(effect)
                if source is not None and effect in source.effects:
                    source.effects.remove(effect)
                if owner._full is full:
                    owner._full = None
                full.effect = None
                if (
                    selection.parent_entry is not None
                    and selection.parent_entry.full is full
                ):
                    selection.parent_entry.full = None
                raise
        return effect

    def _require_display_effect(self, full):
        """Keep the admitted finite display alive through source-fence drain."""
        from .console_display_phases import ConsoleDisplayPhaseRefused

        self._require_graph_identity()
        effect = full.effect
        with self._lock:
            if (
                type(effect) is not _FiniteEffect
                or effect not in self._effects
                or effect.completed.done()
                or effect.kind != "display"
                or type(effect.owner) is not _DisplayEffectOwner
                or effect.owner.full is not full
                or effect.owner.selection is not full.selection
                or full.owner._full is not full
            ):
                raise ConsoleDisplayPhaseRefused("Display effect registration changed")
            selection = effect.owner.selection
            if (
                selection.controls is not self
                or effect.source_revision != selection.source_revision
                or selection.sources is not self._domain.sources
                or full.sources is not selection.sources
                or full.session is not selection.session
                or full.incarnation != selection.incarnation
                or full.port is not selection.port
                or full.stamp is not selection.stamp
                or selection.task.done()
            ):
                raise ConsoleDisplayPhaseRefused("Display admitted source changed")
            source = selection.command_source
            if source is not None:
                if (
                    self._settings_command_tasks.get(selection.task) is not source
                    or source.controls is not self
                    or source.sources is not selection.sources
                    or effect not in source.effects
                ):
                    raise ConsoleDisplayPhaseRefused(
                        "Display command registration changed"
                    )
            if selection.parent_entry is not None:
                # Native phases check registry identity only. Real view/teardown
                # checks remain on the presentation loop at fixed checkpoints.
                entry = selection.parent_entry
                if (
                    selection.port._display_full_entries.get(selection.task)
                    is not entry
                    or entry.task is not selection.task
                    or entry.full is not full
                    or entry.body_returned
                ):
                    raise ConsoleDisplayPhaseRefused("Display parent Task changed")

    def _finish_display_effect(self, full):
        """Complete from exact native records; Task completion proves nothing."""
        from .console_native_commit import ConsoleNativeRetirementUncertain
        from .console_preparation_reads import _retire

        effect = full.effect
        if (
            effect is None
            or type(effect.owner) is not _DisplayEffectOwner
            or effect.owner.full is not full
            or effect.owner.selection is not full.selection
            or asyncio.current_task() is not effect.owner.selection.task
        ):
            raise RuntimeError("Display settlement owner changed")
        # No live source validation here: canceled/fenced/displaced work must
        # still settle its own exact resource records, not a replacement view.
        from .console_display_phases import require_display_phase_definitions

        try:
            require_display_phase_definitions(full)
            retired = (
                full.phase_issuance_closed is True
                and all(task.done() for task in full.tasks.values())
                and full.native_retirement_confirmed()
            )
            require_display_phase_definitions(full)
        except BaseException:
            # A changed observer definition cannot manufacture a positive close.
            retired = False
        logical = full.logical
        if retired:
            if logical is not None:
                _retire(logical)
        elif logical is not None:
            # Leave the SAME handle in its original observer sets. Original
            # drain_preparation_reads calls retired.result() and receives this
            # refusal. It cannot mistake _retire's successful None for closure.
            if not logical.retired.done():
                logical.retired.set_exception(
                    ConsoleNativeRetirementUncertain(
                        "Display native resources did not positively retire"
                    )
                )
                logical.retired.exception()  # Consume warning, not the retained outcome.
        self._finish_effect(effect, retired=retired)
        # _finish_effect(False) retains the effect and closes intake/effects.
        # The source fence then fails through its original finite-effect drain.
        if retired:
            with self._lock:
                if full.owner._full is full:
                    full.owner._full = None
            # Deliberately do NOT require _effects_open or the new current revision.
            # FenceSource has closed new admission and is awaiting THIS effect.
            # Actual source replacement, participant/DB pause and original guards
            # remain effective. This grants nothing to another native operation.


# SOURCE-RECORD HUNK: console_control_dispatch.py, after ConsoleLocalControls.
# Same full defining-module/receiver/closure record shape as character readers.
# It covers the actual source of command Task registration as well as the new
# display-only methods. Existing settings/provider/permission authority is not
# changed by adding this qualification.

_DISPLAY_CONTROL_READERS = tuple(
    (
        ConsoleLocalControls,
        name,
        function,
        function,
        function.__code__,
        function.__globals__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
        globals().get("__file__"),
        globals().get("__spec__"),
        getattr(globals().get("__spec__"), "origin", None),
    )
    for name in (
        "_begin_display_effect",
        "_require_display_effect",
        "_finish_display_effect",
        "_finish_effect",
        "_drain_finite_effects",
        "_finish_source_fence",
        "_require_graph",
        "_require_graph_identity",
        "_retain_control_work",
        "_queue_session_capture",
        "_read_projection",
        "dispatch",
    )
    for function in (vars(ConsoleLocalControls)[name],)
)


def _display_control_sources_current(controls):
    from tldw_chatbook.UI.Console_Modules.character_context import (
        _character_reader_current,
    )

    return type(controls) is ConsoleLocalControls and all(
        _character_reader_current(record, controls)
        for record in _DISPLAY_CONTROL_READERS
    )


# In preselect_display, BEFORE invoking controls._require_graph:
#     if not _display_control_sources_current(controls):
#         return _DisplayDeclined("custom-display-control-definitions")
# In require_selected_display_sources, BEFORE invoking _require_display_effect:
#     if not _display_control_sources_current(selected.controls):
#         raise ConsoleDisplayPhaseRefused("Admitted display control source changed")
# Recheck this exact qualification at admission and terminal observation too.
# A changed finisher cannot claim retirement: use the retained original effect
# and original still-qualified _finish_effect(False) to record uncertainty;
# never invoke an overridden finisher or replace it with task.done().

# Install _display_control_sources_current and the new display-selector helper
# functions in the domain's existing _source_modules/_source_definitions scheme
# with _DOMAIN_SOURCE_DEFINITIONS tuples using console_domain_services._definition.
# The control method inventory above and the phase method inventory use actual
# installed class bodies after integration, not v1 preimage code as authorization.
# These additions must be part of the same integration; omission is not a reason
# to relax any source predicate. Existing port _PORT_DEFINITIONS automatically
# includes its new fixed methods when defined before that tuple is built.
# Fixed module helper, called only from the admitted owner's finally. Register
# its actual definition in _DOMAIN_SOURCE_DEFINITIONS along with the predicate.


def finish_admitted_display(full):
    from tldw_chatbook.UI.Console_Modules.character_context import (
        _character_reader_current,
    )
    from .console_display_phases import ConsoleDisplayPhaseRefused
    from .console_native_commit import ConsoleNativeRetirementUncertain

    controls = full.controls
    if _display_control_sources_current(controls):
        return controls._finish_display_effect(full)
    # Defining-source drift is never permission to call a custom cleanup hook.
    # Preserve the exact retained observer and record a NEGATIVE outcome through
    # the original still-qualified effect primitive on its original receiver.
    logical = full.logical
    if logical is not None and not logical.retired.done():
        logical.retired.set_exception(
            ConsoleNativeRetirementUncertain(
                "Display retirement defining source changed"
            )
        )
        logical.retired.exception()
    record = next(
        item for item in _DISPLAY_CONTROL_READERS if item[1] == "_finish_effect"
    )
    if not _character_reader_current(record):
        raise ConsoleDisplayPhaseRefused(
            "Original effect retirement source is unavailable"
        )
    effect = full.effect
    if (
        effect is None
        or type(effect.owner) is not _DisplayEffectOwner
        or effect.owner.full is not full
        or effect.owner.selection is not full.selection
        or asyncio.current_task() is not effect.owner.selection.task
    ):
        raise ConsoleDisplayPhaseRefused("Original display 