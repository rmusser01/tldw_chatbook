"""Retain original fleet mount work behind an exact inert parent visit."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from inspect import getattr_static
from types import FunctionType

from .console_agent_bridge import ConsoleAgentBridge
from .console_backend_presentation import ConsoleBackendPresentation
from .console_chat_controller import ConsoleChatController
from .console_chat_store import ConsoleChatStore
from .console_domain_services import _definition, _definition_current
from .console_execution_backend import ConsoleExecutionBackend
from .console_fleet_wake import ConsoleFleetWakeCoordinator
from .console_interrupt_rounds import InterruptRoundHost


class ConsoleBackendViewSuperseded(RuntimeError):
    """An authentic presentation checkpoint was replaced by a later action."""


@dataclass(frozen=True, slots=True, eq=False)
class ConsoleBackendViewCheckpoint:
    """Opaque retained checkpoint; its primitive fields grant no authority."""

    attachment_id: str
    visit: int
    action: str
    _registration: tuple | None = field(default=None, init=False, repr=False)


@dataclass(slots=True)
class _MountSeed:
    owners: tuple
    attempted: bool = False
    task: asyncio.Task[int] | None = None


async def _retire_seed(task: asyncio.Task[int]) -> bool:
    """Join the actual seed task without forwarding observer cancellation."""
    cancelled = False
    if task.get_loop() is not asyncio.get_running_loop():
        raise RuntimeError("Backend fleet seed owner loop changed")
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    if task.cancelled():
        raise RuntimeError("Backend fleet seed retirement is unproven")
    task.result()
    return cancelled


class ConsoleBackendViewLifecycle:
    """Run the original mount seed and visibility calls on their native owner."""

    def __init__(self, presentation: ConsoleBackendPresentation) -> None:
        if type(presentation) is not ConsoleBackendPresentation:
            raise TypeError("Original backend presentation is required")
        self.presentation = presentation
        self._loop = asyncio.get_running_loop()
        self._association = presentation.association
        self._attachment: tuple[str, int] | None = None
        self._current: ConsoleBackendViewCheckpoint | None = None
        self._current_values: tuple[str, int, str] | None = None
        self._current_owners: tuple | None = None
        self._seeds: dict[tuple[str, int], _MountSeed] = {}
        self._tasks: list[asyncio.Task[int]] = []
        self._draining = False
        self._drained = False
        self._drain_error: BaseException | None = None

    def _require_owner(self) -> tuple:
        if (
            type(self) is not ConsoleBackendViewLifecycle
            or self._draining
            or asyncio.get_running_loop() is not self._loop
            or any(
                getattr_static(self, name, None) is not record[0]
                or not _definition_current(record)
                for name, record in _METHODS
            )
            or _retire_seed is not _RETIRE[0]
            or not _definition_current(_RETIRE)
            or ConsoleBackendViewSuperseded is not _SUPERSEDED
        ):
            raise RuntimeError("Backend view lifecycle definitions changed")
        presentation = self.presentation
        if (
            type(presentation) is not ConsoleBackendPresentation
            or getattr(presentation, "view_lifecycle", None) is not self
        ):
            raise RuntimeError("Backend view lifecycle endpoint changed")
        backend = presentation.backend
        if type(backend) is not ConsoleExecutionBackend:
            raise RuntimeError("Original Console backend is required")
        backend._require_owner_loop()
        presentation.require_association(self._association)
        controller, store = backend.controller, backend.store
        if (
            type(controller) is not ConsoleChatController
            or type(store) is not ConsoleChatStore
            or backend.runtime._chat_controller is not controller
            or backend.runtime._chat_store is not store
            or controller.store is not store
            or controller.app is not None
            or controller.domain_services is not backend.domain
            or controller._agent_bridge is not backend.runtime._agent_bridge
        ):
            raise RuntimeError("Backend view lifecycle owner changed")
        wake = controller._fleet_wake
        host = controller._interrupt_host
        bridge = controller._agent_bridge
        if (
            type(wake) is not ConsoleFleetWakeCoordinator
            or wake._controller is not controller
            or wake._app is not None
            or wake._disposed
            or type(host) is not InterruptRoundHost
            or (bridge is not None and type(bridge) is not ConsoleAgentBridge)
            or any(
                getattr_static(owner, name, None) is not record[0]
                or not _definition_current(record)
                for owner, definitions in (
                    (presentation, _PRESENTATION_METHODS),
                    (controller, _CONTROLLER_METHODS),
                    (store, _STORE_METHODS),
                    (wake, _WAKE_METHODS),
                    (host, _HOST_METHODS),
                )
                for name, record in definitions
            )
            or presentation.parent_attachment != self._attachment
        ):
            raise RuntimeError("Backend view lifecycle source changed")
        return controller, store, wake, bridge

    def _require_checkpoint(self, checkpoint: ConsoleBackendViewCheckpoint) -> tuple:
        owners = _REQUIRE[0](self)
        if (
            type(checkpoint) is not ConsoleBackendViewCheckpoint
            or type(checkpoint._registration) is not tuple
            or len(checkpoint._registration) != 4
            or checkpoint._registration[0] is not self
            or checkpoint._registration[1] is not checkpoint
            or (checkpoint.attachment_id, checkpoint.visit, checkpoint.action)
            != checkpoint._registration[2]
        ):
            raise RuntimeError("Backend parent view checkpoint changed")
        if any(
            current is not captured
            for current, captured in zip(
                owners, checkpoint._registration[3], strict=True
            )
        ):
            raise RuntimeError("Backend parent view seed owner changed")
        if checkpoint is not self._current:
            raise ConsoleBackendViewSuperseded("Backend parent view was superseded")
        if (
            checkpoint._registration[2] != self._current_values
            or (checkpoint.attachment_id, checkpoint.visit) != self._attachment
            or self._current_owners != checkpoint._registration[3]
        ):
            raise RuntimeError("Backend parent view checkpoint changed")
        seed = self._seeds.get(self._attachment)
        if seed is not None and any(
            current is not captured
            for current, captured in zip(owners, seed.owners, strict=True)
        ):
            raise RuntimeError("Backend parent view seed owner changed")
        return owners

    def admit(
        self, attachment_id: str, visit: int, action: str
    ) -> ConsoleBackendViewCheckpoint:
        """Synchronously supersede older visibility before command tasks can await.

        Args:
            attachment_id: The fixed client identity admitted by its real channel.
            visit: The positive parent visit from that channel.
            action: Exactly mount, visible, or hidden.

        Returns:
            The actual retained checkpoint token, never a reconstructed witness.
        """
        owners = _REQUIRE[0](self)
        if (
            type(attachment_id) is not str
            or not attachment_id
            or len(attachment_id) > 256
            or type(visit) is not int
            or visit < 1
            or type(action) is not str
            or action not in {"mount", "visible", "hidden"}
        ):
            raise ValueError("Invalid Console parent view checkpoint")
        previous = self._attachment
        if previous is None:
            if action not in {"mount", "hidden"}:
                raise RuntimeError("Console parent visit has not mounted")
        elif (
            attachment_id != previous[0]
            or visit < previous[1]
            or (visit > previous[1] and action not in {"mount", "hidden"})
        ):
            raise RuntimeError("Console parent visit order changed")
        key = (attachment_id, visit)
        if action == "visible" and key not in self._seeds:
            raise RuntimeError("Console parent visit has not mounted")
        if action == "mount" and key not in self._seeds:
            self._seeds[key] = _MountSeed(owners)
        checkpoint = ConsoleBackendViewCheckpoint(attachment_id, visit, action)
        object.__setattr__(
            checkpoint,
            "_registration",
            (self, checkpoint, (attachment_id, visit, action), owners),
        )
        self._attachment = key
        self.presentation.parent_attachment = key
        self._current = checkpoint
        self._current_values = (attachment_id, visit, action)
        self._current_owners = owners
        # A completed original task has already returned through its owned DB
        # callback. Keep pending, cancelled, or failed tasks until drain proves
        # their outcome; do not accumulate settled historical visit owners.
        settled = {
            task
            for task in self._tasks
            if task.done() and not task.cancelled() and task.exception() is None
        }
        self._tasks = [task for task in self._tasks if task not in settled]
        self._seeds = {
            seed_key: seed
            for seed_key, seed in self._seeds.items()
            if seed_key == key or (seed.task is not None and seed.task not in settled)
        }
        return checkpoint

    async def prepare(self, checkpoint: ConsoleBackendViewCheckpoint) -> None:
        """Retire the original visit seed before publishing current visibility.

        A hidden checkpoint does not wait for or cancel a superseded seed. A
        visible checkpoint that overtakes mount still joins that visit's seed.
        No recovery audit is started or awaited beyond the original mount path.
        """
        controller, store, wake, bridge = _CHECK[0](self, checkpoint)
        if checkpoint.action == "hidden":
            controller.on_console_view_visibility_changed(False)
            _CHECK[0](self, checkpoint)
            return
        seed = self._seeds[self._attachment]
        if not seed.attempted:
            # The original mount claim first requires an actual bridge, then
            # wires the original coordinator and schedules only saved sessions.
            if bridge is not None:
                self.presentation.wire_wake(wake, wake._recovery_projection)
                if any(
                    session.persisted_conversation_id for session in store.sessions()
                ):
                    # Incremental fresh-mount boundary: the original producer
                    # swallows optional errors, including native-close failure.
                    # Until its exact retirement receipt is integrated, do not
                    # issue it or infer native success from its returned zero.
                    raise RuntimeError("Saved Console mount seed retirement is unqualified")
            seed.attempted = True
        if seed.task is not None:
            try:
                await asyncio.shield(seed.task)
            except asyncio.CancelledError:
                await _RETIRE[0](seed.task)
                raise
            _CHECK[0](self, checkpoint)
        # Stale seed work may finish and update the original coordinator, but
        # only the exact current visit/action can publish visibility afterward.
        _CHECK[0](self, checkpoint)
        controller.on_console_view_visibility_changed(True)
        _CHECK[0](self, checkpoint)

    async def drain(self) -> None:
        """Seal this producer and retire every issued seed before backend close."""
        # Repeated Drain/Shutdown may arrive after native disposal. Revalidate
        # this exact producer and loop without demanding an open backend graph.
        if (
            type(self) is not ConsoleBackendViewLifecycle
            or asyncio.get_running_loop() is not self._loop
            or any(
                getattr_static(self, name, None) is not record[0]
                or not _definition_current(record)
                for name, record in _METHODS
            )
            or _retire_seed is not _RETIRE[0]
            or not _definition_current(_RETIRE)
            or ConsoleBackendViewSuperseded is not _SUPERSEDED
        ):
            raise RuntimeError("Backend view lifecycle drain definitions changed")
        if self._drain_error is not None:
            raise self._drain_error
        if self._drained:
            return
        self._draining = True
        cancelled = False
        error = None
        for task in self._tasks:
            try:
                cancelled |= await _RETIRE[0](task)
            except BaseException as failure:
                if error is None:
                    error = failure
        if error is not None:
            self._drain_error = error
            raise error
        self._drained = True
        self._tasks.clear()
        self._seeds.clear()
        if cancelled:
            raise asyncio.CancelledError


_METHODS = tuple(
    (name, _definition(value))
    for name, value in vars(ConsoleBackendViewLifecycle).items()
    if type(value) is FunctionType
)
_REQUIRE = dict(_METHODS)["_require_owner"]
_CHECK = dict(_METHODS)["_require_checkpoint"]
_RETIRE = _definition(_retire_seed)
_SUPERSEDED = ConsoleBackendViewSuperseded
_PRESENTATION_METHODS = tuple(
    (name, _definition(getattr(ConsoleBackendPresentation, name)))
    for name in ("require_association", "definitions_current", "wire_wake")
)
_CONTROLLER_METHODS = (
    (
        "on_console_view_visibility_changed",
        _definition(ConsoleChatController.on_console_view_visibility_changed),
    ),
)
_STORE_METHODS = (("sessions", _definition(ConsoleChatStore.sessions)),)
_WAKE_METHODS = tuple(
    (name, _definition(getattr(ConsoleFleetWakeCoordinator, name)))
    for name in (
        "wire",
        "capture_loop_if_running",
        "_seed_owned_history",
        "seed_from_marks",
        "retry_soon",
        "_runs_db",
    )
)
_SEED = dict(_WAKE_METHODS)["_seed_owned_history"]
_HOST_METHODS = tuple(
    (name, _definition(getattr(InterruptRoundHost, name)))
    for name in (
        "on_console_view_visibility_changed",
        "set_view_visible",
        "remount_pending_approval_for_active_session",
        "_remount_session_kinds",
    )
)
