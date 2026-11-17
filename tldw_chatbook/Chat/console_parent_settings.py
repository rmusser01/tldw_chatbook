"""Registered custody for fixed Console settings continuations.

These local presentation operations are not user command identities and never
authorize provider, tool, MCP, or arbitrary callback work.
"""

from __future__ import annotations

import asyncio
from inspect import getattr_static
from dataclasses import dataclass, field, replace
from contextlib import ExitStack, asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from .console_display_state import ConsoleControlState
    from .console_live_work import ConsoleLiveWorkLaunch
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )
    from .console_control_dispatch import ConsoleLocalControls, _CommandSource
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

_READ_ACTION_PURPOSES = frozenset(
    {
        "recovery-prepare",
        "settings-open-prepare",
        "settings-context-current",
        "settings-display-prepare",
        "settings-modal-prepare",
        "tabs-settings",
    }
)

_ACTION_PURPOSES = _READ_ACTION_PURPOSES | frozenset(
    {
        "compact-controls",
        "prompt-modal",
        "prompt-picker-result",
        "prompt-editor-result",
        "handsfree-entry",
        "sidechat-open",
        "realtime-ready",
        "prompt-editor-open",
        "prefill-command",
        "workbench-help",
    }
)
_PURPOSES = _ACTION_PURPOSES | {
    "initial-compose",
    "early-full-defaults",
    "readiness-defaults",
    "settings-render-replay",
    "system-prompt",
    "prompt-component",
    "prompt-resolution",
    "realtime-instructions",
    "attachment-completion",
    "guarded-prompt",
    "prompt-target",
    "identity-surfaces",
    "visible-send",
}


class ParentSettingsDeferred(RuntimeError):
    """The existing presentation replay must wait for source intake to reopen."""


class ParentSettingsReadExpired(ParentSettingsDeferred):
    """The exact checked result reached its original validity bound."""


class ParentSettingsOriginChanged(RuntimeError):
    """The waiting action no longer owns the same session or presentation visit."""


class ParentSettingsSourceChanged(RuntimeError):
    """A selected checked-read authority failure, never a UI fallback result."""


@dataclass(eq=False, slots=True)
class ParentSettingsCapture:
    controls: Any
    port: Any
    stamp: tuple
    sources: Any
    source_revision: int
    session: Any
    session_id: str
    incarnation_id: str
    purpose: str
    checked_config: Any
    applied: asyncio.Future
    retired: asyncio.Future
    work: asyncio.Future | None = None
    operation_task: asyncio.Task | None = None
    effects: list = field(default_factory=list)
    effect: Any = None
    pending_operation: Any = None
    guarded_transaction: Any = None
    guarded_claim_settled: bool = False
    action_definition: Any = None
    started: bool = False


@dataclass(eq=False, slots=True)
class _InitialComposePhase:
    screen: Any
    controls: Any
    task: asyncio.Task
    sources: Any
    revision: int
    original_session: Any
    original_incarnation: str | None
    scope: ExitStack = field(default_factory=ExitStack)
    launch: Any = None
    control: Any = None
    error: Exception | None = None
    capture: ParentSettingsCapture | None = None
    previous: asyncio.Future | None = None
    completion: asyncio.Future | None = None
    consumed: bool = False
    content_completed: bool = False
    closed: bool = False


def _require_initial_compose_helpers(screen):
    from tldw_chatbook.UI.Screens import chat_screen
    from .console_guarded_prompt import _require_target_ensure

    if type(screen) is not chat_screen.ChatScreen:
        raise ParentSettingsSourceChanged("Initial Compose receiver changed")
    for name, function, code in chat_screen._selected_compose_definitions:
        if (
            getattr_static(screen, name) is not function
            or function.__code__ is not code
        ):
            raise ParentSettingsSourceChanged("Initial Compose definition changed")
    for owner, name, function, code in chat_screen._selected_compose_framework:
        if getattr_static(owner, name) is not function or function.__code__ is not code:
            raise ParentSettingsSourceChanged("Initial Compose dispatch changed")
    _require_target_ensure(screen)


def _require_initial_compose_phase(phase):
    if (
        type(phase) is not _InitialComposePhase
        or phase.closed
        or phase.task is not asyncio.current_task()
        or phase.screen._task is not phase.task
        or getattr(phase.screen, "_console_initial_compose", None) is not phase
        or phase.screen.app_instance.screen is not phase.screen
        or phase.controls._service._local_controls is not phase.controls
    ):
        raise ParentSettingsOriginChanged("Initial Compose invocation changed")
    _require_initial_compose_helpers(phase.screen)
    phase.controls._require_graph()
    phase.controls._domain.require_current(phase.sources)
    if phase.controls.source_revision != phase.revision:
        raise ParentSettingsDeferred("Initial Compose source was fenced")


def _finish_initial_compose_tail(phase):
    controls, completion = phase.controls, phase.completion
    if completion is None:
        return
    if not completion.done():
        completion.set_result(None)
    controls._capture_predecessors.pop(completion, None)
    session_id = phase.capture.session_id
    if controls._session_capture_tails.get(session_id) is completion:
        del controls._session_capture_tails[session_id]


def finish_initial_console_compose(phase: _InitialComposePhase | None) -> None:
    """Close the synchronous generator phase before Textual mounts children."""
    if phase is None or phase.closed:
        return
    if phase.task is not asyncio.current_task():
        raise ParentSettingsSourceChanged("Initial Compose cleanup Task changed")
    try:
        phase.scope.close()
    finally:
        phase.closed = True
        capture = phase.capture
        if capture is not None:
            controls = phase.controls
            controls._capture_tasks.pop(phase.task, None)
            finish_parent_settings_capture(capture)
            if not capture.work.done():
                capture.work.set_result(None)
            if not capture.applied.done():
                capture.applied.set_result(
                    phase.content_completed and phase.error is None
                )
            if not capture.retired.done():
                capture.retired.set_result(
                    capture.pending_operation is None
                    or capture.pending_operation.retired
                )
            if phase.previous is not None and not phase.previous.done():
                phase.previous.add_done_callback(
                    lambda _done: _finish_initial_compose_tail(phase)
                )
            else:
                _finish_initial_compose_tail(phase)


def consume_initial_console_compose(
    screen: ChatScreen,
) -> tuple[ConsoleLiveWorkLaunch | None, ConsoleControlState] | None:
    """Consume only this uninterrupted prelude's original launch and control."""
    phase = getattr(screen, "_console_initial_compose", None)
    if phase is None:
        return None
    if phase.error is not None:
        raise phase.error
    _require_initial_compose_phase(phase)
    if phase.consumed:
        raise ParentSettingsSourceChanged("Initial Compose was already consumed")
    if phase.capture is not None:
        require_parent_settings_capture(phase.capture)
    phase.consumed = True
    return phase.launch, phase.control


async def prepare_initial_console_compose(
    screen: ChatScreen,
) -> _InitialComposePhase | None:
    """Prepare the original control prefix only after its actual due boundary.

    Args:
        screen: The actual Console whose Compose message is executing.

    Returns:
        A finite synchronous phase, or None for the original legacy graph.
        Ordinary errors are carried into the original compose error boundary.
    """
    from .console_control_settings import (
        ConsoleDefaultConvergenceRequired,
        ConsoleSettingsPending,
        wait_settings_settled,
    )

    runtime = getattr_static(screen.app_instance, "console_runtime", None)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    task = asyncio.current_task()
    if screen._task is None:
        # Stock eager create_task publishes the message-pump Task on return.
        # No launch, derivation, effect, or checked result exists at this gate.
        await asyncio.sleep(0)
    store = controls._store
    original = store._sessions.get(store.active_session_id)
    phase = _InitialComposePhase(
        screen,
        controls,
        task,
        controls._domain.sources,
        controls.source_revision,
        original,
        original.incarnation_id if original is not None else None,
    )
    screen._console_initial_compose = phase
    try:
        _require_initial_compose_phase(phase)
        phase.scope.enter_context(screen._console_derivation_scope())
        phase.launch = screen._consume_pending_console_launch()
        try:
            phase.control = screen._build_console_control_state(phase.launch)
            return phase
        except (ConsoleDefaultConvergenceRequired, ConsoleSettingsPending) as due:
            phase.scope.close()
            _require_initial_compose_phase(phase)
            port = controls._domain.presentation
            stamp = port._stamp()
            if port._view is not screen or not port._current_visit(stamp):
                raise ParentSettingsOriginChanged(
                    "Initial Compose attachment changed"
                ) from due
            if original is None:
                witness = getattr(due, "created_session", None)
                if type(witness) is not tuple or len(witness) != 8:
                    raise
                (
                    controller,
                    actual_store,
                    target,
                    incarnation,
                    settings,
                    revision,
                    binding,
                    task,
                ) = witness
                if (
                    controller is not screen._session
                    or actual_store is not store
                    or task is not phase.task
                    or target.incarnation_id != incarnation
                    or target.settings is not settings
                    or target.settings_revision != revision
                    or target.conversation_binding_revision != binding
                ):
                    raise ParentSettingsOriginChanged(
                        "Initial Compose creation changed"
                    ) from due
            else:
                target = original
                if target.incarnation_id != phase.original_incarnation:
                    raise ParentSettingsOriginChanged(
                        "Initial Compose target changed"
                    ) from due
            if (
                store._sessions.get(target.id) is not target
                or store.active_session_id != target.id
            ):
                raise ParentSettingsOriginChanged(
                    "Initial Compose target changed"
                ) from due
            capture = issue_parent_settings_capture(controls, "initial-compose", phase)
            phase.capture = capture
            capture.operation_task = phase.task
            controls._parent_settings_captures[capture] += (phase.task,)
            capture.work = controls._loop.create_future()
            phase.previous = controls._session_capture_tails.get(target.id)
            phase.completion = controls._loop.create_future()
            controls._session_capture_tails[target.id] = phase.completion
            controls._capture_predecessors[phase.completion] = phase.previous
            controls._capture_tasks[phase.task] = phase.completion
            if phase.previous is not None:
                await asyncio.shield(phase.previous)
            _require_initial_compose_phase(phase)
            require_parent_settings_capture(capture)
            await wait_settings_settled(store, target.id)
            _require_initial_compose_phase(phase)
            require_parent_settings_capture(capture)
            capture.started = True
            if isinstance(due, ConsoleSettingsPending):
                phase.scope.enter_context(screen._console_derivation_scope())
                try:
                    phase.control = screen._build_console_control_state(phase.launch)
                    return phase
                except ConsoleDefaultConvergenceRequired:
                    phase.scope.close()
            request = await settle_parent_action_defaults(capture)
            phase.scope.enter_context(due_settings_phase(request))
            _require_initial_compose_phase(phase)
            phase.control = screen._build_console_control_state(phase.launch)
            return phase
    except Exception as error:
        phase.error = error
        finish_initial_console_compose(phase)
        return phase
    except BaseException:
        finish_initial_console_compose(phase)
        raise


@dataclass(eq=False, slots=True)
class _ReadinessSettingsDependency:
    capture: ParentSettingsCapture
    producer: asyncio.Task
    projection: Any
    handoff: Any
    key: tuple
    changed: asyncio.Future
    previous: asyncio.Future | None = None
    waiting: bool = False


def _readiness_dependency_current(dependency):
    """A proven waiter permits only this pre-entry fixed dependency to advance."""
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _handoff_projection_is_stock,
    )

    capture = dependency.capture
    controls = capture.controls
    entry = controls._settings_readiness_waits.get(dependency.producer)
    if entry is None or not dependency.waiting:
        return False
    source, owner, projection, handoff, key = entry
    if (
        owner.done()
        or dependency.producer.done()
        or projection is not dependency.projection
        or handoff is not dependency.handoff
        or key is not dependency.key
    ):
        return False
    if type(source) is ParentSettingsCapture:
        if source.operation_task is not owner or not source.started:
            return False
        require_parent_settings_capture(source, controls)
        session_id, incarnation = source.session_id, source.incarnation_id
    else:
        if controls._settings_command_tasks.get(owner) is not source:
            return False
        session_id, incarnation = (
            source.identity.session_id,
            source.identity.incarnation_id,
        )
    ancestor = controls._capture_tasks.get(owner)
    previous = dependency.previous
    if ancestor is None or ancestor.done():
        return False
    while previous is not None and previous is not ancestor:
        if previous.done():
            return False
        previous = controls._capture_predecessors.get(previous)
    if previous is not ancestor:
        return False
    require_parent_settings_capture(capture, controls)
    if (
        controls._settings_readiness_captures.get(dependency.producer) is not dependency
        or capture.purpose != "readiness-defaults"
        or capture.started
        or capture.pending_operation is not None
        or capture.operation_task is not asyncio.current_task()
        or capture.sources is not source.sources
        or capture.session_id != session_id
        or capture.incarnation_id != incarnation
        or projection.screen is not capture.port._view
        or projection._handoff_read is not handoff
        or handoff[0] is not key
        or key[5] != session_id
        or handoff[2]._task is not dependency.producer
        or not _handoff_projection_is_stock(projection)
    ):
        raise RuntimeError("Readiness settings dependency owner changed")
    return True


def require_readiness_capture_current(dependency):
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _handoff_projection_is_stock,
        _same_readiness_key,
    )

    capture, projection, handoff = (
        dependency.capture,
        dependency.projection,
        dependency.handoff,
    )
    require_parent_settings_capture(capture)
    if (
        capture.controls._settings_readiness_captures.get(dependency.producer)
        is not dependency
        or projection.screen is not capture.port._view
        or projection._handoff_read is not handoff
        or handoff[0] is not dependency.key
        or handoff[2]._task is not dependency.producer
        or not _handoff_projection_is_stock(projection)
        or not _same_readiness_key(dependency.key, projection._key())
    ):
        raise RuntimeError("Readiness defaults source changed before entry")


async def wait_readiness_predecessor(capture, previous):
    """Keep the exact queued capture; only its fixed awaited dependency may enter."""
    controls = capture.controls
    dependency = next(
        (
            item
            for item in controls._settings_readiness_captures.values()
            if item.capture is capture
        ),
        None,
    )
    if dependency is None:
        await asyncio.shield(previous)
        return
    dependency.previous = previous
    dependency.waiting = True
    try:
        while not previous.done():
            if _readiness_dependency_current(dependency):
                require_readiness_capture_current(dependency)
                return
            changed = dependency.changed
            await asyncio.wait((previous, changed), return_when=asyncio.FIRST_COMPLETED)
            if changed.done() and dependency.changed is changed:
                dependency.changed = controls._loop.create_future()
        previous.result()
        require_readiness_capture_current(dependency)
    finally:
        dependency.waiting = False


def require_parent_settings_capture(
    capture: ParentSettingsCapture,
    controls: ConsoleLocalControls | None = None,
    *,
    presentation: bool = True,
) -> ParentSettingsCapture:
    """Require the actual registered owner, not structurally similar data."""
    _require_definitions()
    if type(capture) is not ParentSettingsCapture:
        raise RuntimeError("Parent settings source is not an issued capture")
    owner = capture.controls
    issued = owner._parent_settings_captures.get(capture)
    if (
        (controls is not None and owner is not controls)
        or issued is None
        or any(
            issued[index] is not value
            for index, value in enumerate(
                (
                    capture.port,
                    capture.stamp,
                    capture.sources,
                    capture.session,
                    capture.checked_config,
                )
            )
        )
        or issued[5:9]
        != (
            capture.source_revision,
            capture.session_id,
            capture.incarnation_id,
            capture.purpose,
        )
        or (len(issued) == 10 and issued[9] is not capture.operation_task)
        or owner._domain.presentation is not capture.port
        or owner._runtime._execution_service._local_controls is not owner
        or owner._store._sessions.get(capture.session_id) is not capture.session
        or capture.session.incarnation_id != capture.incarnation_id
        or capture.purpose not in _PURPOSES
    ):
        raise RuntimeError("Parent settings capture owner changed")
    owner._domain.require_current(capture.sources)
    if presentation:
        if not capture.port._current_visit(capture.stamp):
            raise ParentSettingsOriginChanged("Parent settings presentation changed")
        if capture.purpose == "visible-send":
            _require_visible_send_press(capture)
        elif owner._store.active_session_id != capture.session_id:
            raise ParentSettingsOriginChanged("Parent settings presentation changed")
    return capture


def issue_parent_settings_capture(
    controls: ConsoleLocalControls, purpose: str, checked_config: Any
) -> ParentSettingsCapture:
    """Admit one named original parent action before any await or exposure.

    Args:
        controls: The actual controls from the selected graph.
        purpose: A registered fixed action purpose.
        checked_config: That purpose's original captured inputs.

    Returns:
        The registered capture and its admitted finite effect.

    Raises:
        RuntimeError: The selected source, graph, or active visit changed.
        ValueError: The fixed purpose is not registered.
    """
    _require_definitions()
    if purpose not in _PURPOSES:
        raise ValueError("Unknown parent settings purpose")
    controls._require_graph()
    port = controls._domain.presentation
    stamp = port._stamp()
    store = controls._store
    if purpose == "visible-send":
        from tldw_chatbook.UI.Console_Modules.send_acknowledgement import (
            _VisibleSettingsPress,
        )

        if type(checked_config) is not _VisibleSettingsPress:
            raise TypeError("Visible Send requires its actual press")
        session = store._sessions.get(checked_config.session_id)
    else:
        session = store._sessions.get(store.active_session_id)
    if session is None or not port._current_visit(stamp):
        raise RuntimeError("Parent settings requires its actual active session view")
    capture = ParentSettingsCapture(
        controls,
        port,
        stamp,
        controls._domain.sources,
        controls.source_revision,
        session,
        session.id,
        session.incarnation_id,
        purpose,
        checked_config,
        controls._loop.create_future(),
        controls._loop.create_future(),
    )
    capture.effect = controls._begin_parent_settings_effect(capture)
    return capture


def finish_parent_settings_capture(capture: ParentSettingsCapture) -> None:
    """Retire only this capture's actual settings/native operation."""
    operation = capture.pending_operation
    if capture.operation_task is not None and not capture.operation_task.done():
        phase = capture.checked_config
        if not (
            capture.purpose == "initial-compose"
            and type(phase) is _InitialComposePhase
            and phase.capture is capture
            and phase.closed
            and phase.task is capture.operation_task is asyncio.current_task()
            and getattr(phase.screen, "_console_initial_compose", None) is phase
        ):
            raise RuntimeError("Parent settings continuation has not settled")
    if capture.purpose == "guarded-prompt":
        from .console_guarded_prompt import finish_guarded_capture

        finish_guarded_capture(capture)
    if capture.purpose == "identity-surfaces":
        _origin, request, token = capture.checked_config
        _retire_identity_request(request, token)
    retired = operation is None or operation.retired
    capture.controls._finish_effect(capture.effect, retired=retired)
    if retired:
        del capture.controls._parent_settings_captures[capture]


async def run_parent_settings_capture(capture: ParentSettingsCapture) -> Any:
    from .console_control_settings import (
        ConsoleDefaultConvergenceRequired,
        converge_default_settings,
    )

    require_parent_settings_capture(capture)
    _require_action_helpers(capture)
    if capture.operation_task is not asyncio.current_task():
        raise RuntimeError("Parent settings continuation Task changed")
    if capture.started:
        raise RuntimeError("Parent settings capture was already consumed")
    capture.started = True
    if capture.purpose == "visible-send":
        from tldw_chatbook.UI.Console_Modules import send_acknowledgement as sending

        _require_visible_send_press(capture)
        try:
            return sending._finish_settings_press(
                capture.stamp[1], capture.checked_config
            )
        except ConsoleDefaultConvergenceRequired:
            # Only an original unbound target can reach ensure here. Never
            # converge an active replacement for a still-painted old chat.
            if capture.checked_config.bound is not None:
                raise
            request = await settle_parent_action_defaults(capture)
            with due_settings_phase(request):
                _require_visible_send_press(capture)
                return sending._finish_settings_press(
                    capture.stamp[1], capture.checked_config
                )
    if capture.purpose == "early-full-defaults":
        from tldw_chatbook.UI.Console_Modules.config_sync import (
            sync_selected_console_core_state,
        )

        await sync_selected_console_core_state(capture.port._view, capture)
        return
    if capture.purpose == "identity-surfaces":
        await _run_parent_identity(capture)
        return
    if capture.purpose == "prompt-target":
        from .console_guarded_prompt import run_prompt_target

        return await run_prompt_target(capture)
    if capture.purpose == "guarded-prompt":
        from .console_guarded_prompt import run_guarded_prompt

        return await run_guarded_prompt(capture)
    if capture.purpose == "attachment-completion":
        origin, phase, data = capture.checked_config
        screen = capture.port._view
        _require_attachment_origin(screen, origin)
        try:
            with screen._console_derivation_scope():
                return _finish_parent_attachment(screen, origin, phase, data)
        except ConsoleDefaultConvergenceRequired:
            request = await settle_parent_action_defaults(capture)
            with due_settings_phase(request):
                _require_attachment_origin(screen, origin)
                return _finish_parent_attachment(screen, origin, phase, data)
    if capture.purpose == "realtime-instructions":
        owner = _parent_realtime_owner(capture)
        try:
            return owner._console_realtime_instructions()
        except ConsoleDefaultConvergenceRequired:
            request = await settle_parent_action_defaults(capture)
            with due_settings_phase(request):
                return _parent_realtime_owner(capture)._console_realtime_instructions()
    if capture.purpose == "prompt-resolution":
        try:
            return _finish_parent_prompt_resolution(capture)
        except ConsoleDefaultConvergenceRequired:
            request = await settle_parent_action_defaults(capture)
            with due_settings_phase(request):
                return _finish_parent_prompt_resolution(capture)
    if capture.purpose == "prompt-component":
        from .console_control_settings import (
            PendingSettingsApply,
            retain_settings_operation,
        )

        flow, value, retry, reviewed = capture.checked_config
        _require_prompt_flow(capture.controls, flow)
        try:
            _prepare_parent_prompt_component(capture)
        except ConsoleDefaultConvergenceRequired:
            request = await settle_parent_action_defaults(capture)
            with due_settings_phase(request):
                _prepare_parent_prompt_component(capture)
        if reviewed is not None:
            flow._apply_reviewed_user_prompt(reviewed, revalidate=True)
        operation = PendingSettingsApply(
            capture.controls,
            capture,
            capture,
            mutation=("system_prompt", value),
        )
        post_draft = _capture_prompt_post_draft(capture)
        _session, persisted = await retain_settings_operation(operation)
        require_parent_settings_capture(capture)
        _require_prompt_flow(capture.controls, flow)
        await _finish_parent_prompt_posts(capture, operation, post_draft, flow=flow)
        return persisted
    if capture.purpose == "system-prompt":
        await _apply_parent_system_prompt(capture, capture.checked_config)
        return
    if capture.purpose == "settings-render-replay":
        from tldw_chatbook.UI.Console_Modules.config_sync import (
            replay_whole_console_sync,
        )

        replay_whole_console_sync(capture.port._view)
        return
    if capture.purpose in _ACTION_PURPOSES:
        from .console_domain_services import _definition_current

        method, original = _parent_action(capture.port._view, capture.purpose)
        if (
            method.__func__ is not original
            or capture.action_definition[0] is not original
            or not _definition_current(capture.action_definition)
        ):
            raise RuntimeError("Parent settings action definition changed")
        args = capture.checked_config
        if capture.purpose in _READ_ACTION_PURPOSES:
            _require_settings_read_target(capture, capture.purpose, args[0])
            try:
                return method(*args[0])
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(capture)
                with due_settings_phase(request):
                    _require_settings_read_target(capture, capture.purpose, args[0])
                    return method(*args[0])
        if capture.purpose == "compact-controls":
            from .console_control_settings import (
                PendingSettingsApply,
                retain_settings_operation,
            )

            screen = capture.port._view
            try:
                updates, settings = _prepare_compact_action(capture, args)
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(capture)
                with due_settings_phase(request):
                    updates, settings = _prepare_compact_action(capture, args)
            if not updates:
                return
            if settings is not None:
                operation = PendingSettingsApply(
                    capture.controls,
                    capture,
                    capture,
                    mutation=("replacement", settings),
                )
                await retain_settings_operation(operation)
                require_parent_settings_capture(capture)
                _require_action_helpers(capture)
                screen._session._finish_active_console_session_replacement()
            screen._finish_compact_shell_controls(updates)
        elif capture.purpose == "workbench-help":
            try:
                method()
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(capture)
                with due_settings_phase(request):
                    method()
        elif capture.purpose in {"prompt-editor-open", "prefill-command"}:
            await method(*args[0])
        elif capture.purpose == "prompt-modal":
            try:
                method(initial_mode=args[0], initial_source=args[1])
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(capture)
                with due_settings_phase(request):
                    method(initial_mode=args[0], initial_source=args[1])
        elif capture.purpose == "handsfree-entry":
            try:
                method(capture_live=args[0])
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(capture)
                with due_settings_phase(request):
                    method(capture_live=args[0])
        elif capture.purpose == "sidechat-open":
            try:
                method(args[0])
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(capture)
                with due_settings_phase(request):
                    method(args[0])
        elif capture.purpose == "realtime-ready":
            _parent_realtime_owner(capture)
            try:
                method(args[0])
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(capture)
                with due_settings_phase(request):
                    _parent_realtime_owner(capture)
                    method(args[0])
        elif capture.purpose in {"prompt-picker-result", "prompt-editor-result"}:
            prompts = capture.port._view._prompts
            prepare = (
                prompts._prepare_console_system_prompt_picker_result
                if capture.purpose == "prompt-picker-result"
                else prompts._prepare_console_system_prompt_editor_result
            )
            applies, value = prepare(args[0])
            if applies:
                await _apply_parent_system_prompt(capture, value)
        return
    await converge_default_settings(
        capture.controls,
        capture.session_id,
        capture.checked_config,
        _parent_capture=capture,
    )


async def prepare_parent_default_settings(
    screen: ChatScreen, *, purpose: str, checked_config: Any = None
) -> None:
    """Join existing settings order, or stay inline under the exact command."""
    from .console_control_settings import converge_default_settings

    runtime = screen._console_runtime()
    controls = runtime._execution_service._local_controls
    if controls is None or controls._store.active_session_id is None:
        if purpose == "early-full-defaults":
            from tldw_chatbook.UI.Console_Modules.config_sync import (
                sync_console_core_state,
            )

            sync_console_core_state(screen)
        return
    session_id = controls._store.active_session_id
    if purpose == "early-full-defaults":
        from tldw_chatbook.UI.Console_Modules.config_sync import (
            sync_selected_console_core_state,
        )

        owner = settings_inline_source(runtime) or settings_inline_parent(runtime)
        if owner is not None:
            await sync_selected_console_core_state(screen, owner)
            return
    parent = settings_inline_parent(runtime)
    if parent is not None:
        if parent.session_id != session_id:
            raise ParentSettingsOriginChanged("Default settings parent session changed")
        operation = await converge_default_settings(
            controls, session_id, checked_config, _parent_capture=parent
        )
        _record_due_settings_convergence(controls, checked_config, operation)
        return
    source = settings_inline_source(runtime)
    if source is not None:
        if source.identity.session_id != session_id:
            raise RuntimeError("Default settings command session changed")
        operation = await converge_default_settings(
            controls, session_id, checked_config
        )
        _record_due_settings_convergence(controls, checked_config, operation)
        return
    capture = issue_parent_settings_capture(controls, purpose, checked_config)
    dependency = None
    if purpose == "readiness-defaults":
        from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
            _handoff_projection_is_stock,
        )

        projection = getattr(screen, "_console_readiness_config_projection", None)
        handoff = getattr(projection, "_handoff_read", None)
        producer = asyncio.current_task()
        if (
            projection is None
            or not _handoff_projection_is_stock(projection)
            or handoff is None
            or handoff[2]._task is not producer
            or handoff[0][5] != session_id
            or producer in controls._settings_readiness_captures
        ):
            finish_parent_settings_capture(capture)
            raise RuntimeError("Readiness default capture requires its actual producer")
        dependency = _ReadinessSettingsDependency(
            capture,
            producer,
            projection,
            handoff,
            handoff[0],
            controls._loop.create_future(),
        )
        controls._settings_readiness_captures[producer] = dependency
    try:
        controls._queue_session_capture(capture, capture)
        cancelled = False
        while not capture.applied.done():
            try:
                await asyncio.shield(capture.applied)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                break
        capture.applied.result()
        if cancelled:
            raise asyncio.CancelledError
        _record_due_settings_convergence(
            controls, checked_config, capture.pending_operation, dependency
        )
    finally:
        if dependency is not None:
            if (
                controls._settings_readiness_captures.get(dependency.producer)
                is dependency
            ):
                del controls._settings_readiness_captures[dependency.producer]
            if not dependency.changed.done():
                dependency.changed.cancel()


@dataclass(frozen=True, slots=True)
class _IdentitySettingsOrigin:
    controls: Any
    sources: Any
    revision: int
    stamp: tuple
    session: Any
    incarnation: str | None
    binding: int | None
    task: asyncio.Task


def _require_identity_body(screen):
    from tldw_chatbook.UI.Screens import chat_screen
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    wrapper, body, wrapper_code, body_code = (
        chat_screen._selected_identity_settings_action
    )
    if (
        type(screen) is not chat_screen.ChatScreen
        or getattr_static(screen, "_sync_console_identity_surfaces") is not wrapper
        or wrapper.__wrapped__ is not body
        or wrapper.__code__ is not wrapper_code
        or body.__code__ is not body_code
        or wrapper.__globals__ is not vars(spend)
        or dict(
            zip(
                wrapper.__code__.co_freevars,
                (cell.cell_contents for cell in wrapper.__closure__),
                strict=True,
            )
        )
        != {"function": body, "identity": True, "wrapped": wrapper}
    ):
        raise ParentSettingsSourceChanged("Identity refresh body changed")
    for owner, name, function, code in chat_screen._selected_identity_settings_sources:
        if getattr_static(owner, name) is not function or function.__code__ is not code:
            raise ParentSettingsSourceChanged("Identity refresh source changed")
    return body


def capture_parent_identity_origin(
    screen: ChatScreen,
) -> _IdentitySettingsOrigin | None:
    """Capture the autonomous original identity body before its due attempt.

    Args:
        screen: The original identity receiver.

    Returns:
        Its exact origin, or None for legacy or an already retained owner.
    """
    app = getattr_static(screen, "app_instance", None)
    runtime = getattr_static(app, "console_runtime", None)
    if runtime is None:
        runtime = getattr_static(screen, "_console_runtime_fallback", None)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    _require_definitions()
    _require_identity_body(screen)
    if (
        settings_inline_source(runtime) is not None
        or settings_inline_parent(runtime) is not None
    ):
        return None
    controls._require_graph()
    controls._domain.require_current(controls._domain.sources)
    session = controls._store._sessions.get(controls._store.active_session_id)
    return _IdentitySettingsOrigin(
        controls,
        controls._domain.sources,
        controls.source_revision,
        controls._domain.presentation._stamp(),
        session,
        session.incarnation_id if session is not None else None,
        session.conversation_binding_revision if session is not None else None,
        asyncio.current_task(),
    )


def _require_identity_origin(origin):
    controls = origin.controls
    controls._require_graph()
    controls._domain.require_current(origin.sources)
    _require_identity_body(origin.stamp[1])
    if controls.source_revision != origin.revision:
        raise ParentSettingsDeferred("Identity refresh source was fenced")
    if (
        not controls._domain.presentation._current_visit(origin.stamp)
        or controls._store._sessions.get(controls._store.active_session_id)
        is not origin.session
        or (
            origin.session is not None
            and (
                origin.session.incarnation_id != origin.incarnation
                or origin.session.conversation_binding_revision != origin.binding
            )
        )
    ):
        raise ParentSettingsOriginChanged("Identity refresh origin changed")


def _retire_identity_request(request, token):
    if request is not None and request.token is token:
        _require_identity_body(request.screen)
        request.retire()


def defer_parent_identity(
    origin: _IdentitySettingsOrigin, request: Any, token: Any, due: Exception
) -> None:
    """Queue only the fixed due identity body after the original locks exit.

    Args:
        origin: The original receiver/source captured before its body.
        request: Its original presentation handoff, when issued.
        token: Exact token observed by that original callback.
        due: The typed request raised by the original body.

    Raises:
        ParentSettingsDeferred: Admission fenced before any capture was issued.
        ParentSettingsOriginChanged: The original session or visit changed.
        ParentSettingsSourceChanged: An original defining source changed.
    """
    _require_definitions()
    if origin.session is None:
        witness = getattr(due, "created_session", None)
        if type(witness) is not tuple or len(witness) != 8:
            raise due
        controller, store, session, incarnation, settings, revision, binding, task = (
            witness
        )
        if (
            controller is not origin.stamp[1]._session
            or store is not origin.controls._store
            or task is not origin.task
            or task is not asyncio.current_task()
            or store._sessions.get(session.id) is not session
            or session.settings is not settings
            or session.settings_revision != revision
        ):
            raise ParentSettingsOriginChanged(
                "Identity created-session witness changed"
            )
        origin = replace(
            origin, session=session, incarnation=incarnation, binding=binding
        )
    _require_identity_origin(origin)
    controls = origin.controls
    for active in controls._parent_settings_captures:
        if active.purpose != "identity-surfaces":
            continue
        prior, prior_request, prior_token = active.checked_config
        if (
            prior.sources is origin.sources
            and prior.revision == origin.revision
            and prior.session is origin.session
            and prior.incarnation == origin.incarnation
            and prior.binding == origin.binding
            and controls._domain.presentation._current_visit(prior.stamp)
        ):
            # No arguments on the actual identity method: latest call equals
            # the same original body. Retire only this coalesced call's token.
            if prior_request is not request or prior_token is not token:
                _retire_identity_request(request, token)
            return
    capture = issue_parent_settings_capture(
        controls, "identity-surfaces", (origin, request, token)
    )
    controls._queue_session_capture(capture, capture)
    capture.applied.add_done_callback(
        lambda value: None if value.cancelled() else value.exception()
    )


async def _run_parent_identity(capture):
    origin, request, token = capture.checked_config
    _require_identity_origin(origin)
    due = await settle_parent_action_defaults(capture)
    with due_settings_phase(due):
        _require_identity_origin(origin)
        _require_identity_body(capture.port._view)(capture.port._view)


def defer_parent_settings_render(screen):
    """Retain one original coalesced replay without deriving mixed settings."""
    from .console_control_settings import settings_pending

    store = getattr(screen, "_console_chat_store", None)
    if store is None:
        return False
    runtime = screen._console_runtime()
    controls = runtime._execution_service._local_controls
    if controls is None:
        return False
    pending = settings_pending(store, store.active_session_id)
    tail = controls._session_capture_tails.get(store.active_session_id)
    if pending is None:
        if tail is None or tail.done():
            return False
        if (
            settings_inline_source(runtime) is not None
            or settings_inline_parent(runtime) is not None
        ):
            return False
    for capture in controls._parent_settings_captures:
        if capture.purpose == "settings-render-replay" and capture.stamp[1] is screen:
            return True
    try:
        capture = issue_parent_settings_capture(
            controls, "settings-render-replay", None
        )
    except ParentSettingsDeferred:
        from tldw_chatbook.UI.Console_Modules.config_sync import (
            replay_whole_console_sync,
        )

        replay_whole_console_sync(screen)
        return True
    controls._queue_session_capture(capture, capture)
    # Render has no caller awaiting a value. Consume only its final error;
    # original Task/effect custody still owns cancellation and retirement.
    capture.applied.add_done_callback(
        lambda result: None if result.cancelled() else result.exception()
    )
    return True


@contextmanager
def parent_settings_render(screen):
    """Retain the existing replay when a fixed pure renderer reaches due settings."""
    from .console_control_settings import (
        ConsoleDefaultConvergenceRequired,
        ConsoleSettingsPending,
    )

    try:
        yield
    except (ConsoleDefaultConvergenceRequired, ConsoleSettingsPending):
        if not defer_parent_settings_render(screen):
            from tldw_chatbook.UI.Console_Modules.config_sync import (
                replay_whole_console_sync,
            )

            replay_whole_console_sync(screen)


def _parent_action(screen, purpose):
    """Resolve only the five named original settings action bodies."""
    from tldw_chatbook.UI.Screens import chat_screen
    from tldw_chatbook.UI.Console_Modules import prompts, hands_free

    if type(screen) is not chat_screen.ChatScreen:
        raise RuntimeError("Parent settings action requires its actual Console")
    _require_action_codes(chat_screen)
    _require_action_codes(prompts)
    _require_action_codes(hands_free)
    if purpose == "compact-controls":
        return (
            screen._sync_compact_shell_controls,
            chat_screen._selected_compact_settings_action,
        )
    if purpose == "handsfree-entry":
        return (
            screen._hands_free._enter_console_hands_free_loop,
            hands_free._selected_handsfree_settings_action,
        )
    if purpose == "sidechat-open":
        return (
            screen._console_side_chat_requested,
            chat_screen._selected_sidechat_settings_action,
        )
    if purpose == "realtime-ready":
        from tldw_chatbook.UI.Console_Modules import realtime

        _require_action_codes(realtime)
        return (
            screen._realtime._on_console_realtime_ready,
            realtime._selected_realtime_settings_action,
        )
    if purpose in _READ_ACTION_PURPOSES:
        name = {
            "recovery-prepare": "_prepare_console_provider_recovery",
            "settings-open-prepare": "_prepare_console_settings_open",
            "settings-context-current": "_console_settings_context_current",
            "settings-display-prepare": "_prepare_console_settings_display",
            "settings-modal-prepare": "_prepare_console_settings_modal",
            "tabs-settings": "_prepare_console_tabs_settings",
        }[purpose]
        return getattr(screen, name), chat_screen._selected_settings_read_actions[name]
    if purpose == "workbench-help":
        return (
            screen._show_console_workbench_help,
            chat_screen._selected_workbench_help_action,
        )
    if purpose == "prefill-command":
        return (
            screen._run_console_prefill_command,
            chat_screen._selected_prefill_settings_action,
        )
    methods = prompts._selected_prompt_settings_actions
    name = {
        "prompt-modal": "_open_console_prompts_modal",
        "prompt-picker-result": "_apply_console_system_prompt_picker_result",
        "prompt-editor-result": "_apply_console_system_prompt_editor_result",
        "prompt-editor-open": "_run_console_system_prompt_editor",
    }[purpose]
    return getattr(screen._prompts, name), methods[name]


def _require_action_codes(module):
    if any(
        function.__code__ is not code
        for function, code in module._selected_parent_settings_action_codes
    ):
        raise RuntimeError("Parent settings action defining body changed")


def _require_visible_send_helpers(screen):
    from tldw_chatbook.UI.Screens import chat_screen
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement as sending
    from .console_guarded_prompt import _require_target_ensure

    if type(screen) is not chat_screen.ChatScreen:
        raise ParentSettingsSourceChanged("Visible Send receiver changed")
    _require_target_ensure(screen)
    from tldw_chatbook.UI.Console_Modules import session
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    for name, function, code in session._selected_visible_send_helpers:
        if (
            getattr_static(screen._session, name) is not function
            or function.__code__ is not code
        ):
            raise ParentSettingsSourceChanged("Visible Send session helper changed")
    composer = screen._console_composer_or_none()
    if composer is not None:
        if type(composer) is not ConsoleComposerBar:
            raise ParentSettingsSourceChanged("Visible Send composer changed")
        for name, record in _VISIBLE_SEND_COMPOSER_DEFINITIONS:
            from .console_domain_services import _definition_current

            if getattr_static(composer, name) is not record[
                0
            ] or not _definition_current(record):
                raise ParentSettingsSourceChanged(
                    "Visible Send capture definition changed"
                )
    for name, function, code in chat_screen._selected_visible_send_helpers:
        if (
            getattr_static(screen, name) is not function
            or function.__code__ is not code
        ):
            raise ParentSettingsSourceChanged("Visible Send helper changed")
    for name, function, code in sending._selected_settings_send_helpers:
        if getattr(sending, name) is not function or function.__code__ is not code:
            raise ParentSettingsSourceChanged("Visible Send scheduling helper changed")


def _require_visible_send_press(capture):
    from tldw_chatbook.UI.Console_Modules.send_acknowledgement import (
        _VisibleSettingsPress,
    )
    from .console_guarded_prompt import _require_composer

    press, screen = capture.checked_config, capture.stamp[1]
    _require_visible_send_helpers(screen)
    if (
        type(press) is not _VisibleSettingsPress
        or press.session_id != capture.session_id
    ):
        raise ParentSettingsSourceChanged("Visible Send press changed")
    if screen._console_composer_or_none() is not press.composer:
        raise ParentSettingsOriginChanged("Visible Send composer changed")
    if press.composer is not None:
        _require_composer(press.composer, screen)
    bound = screen._console_visible_draft_session_id
    active = capture.controls._store.active_session_id
    if (
        bound != capture.session_id
        and not (bound is None and press.bound is None and active == capture.session_id)
    ) or (
        bound is not None
        and active is not None
        and bound != active
        and screen._session._composer_chat_left_the_screen(bound, active)
    ):
        raise ParentSettingsOriginChanged("Visible Send chat changed")


def retain_parent_visible_send(
    screen: ChatScreen,
    *,
    guard=None,
    schedule: bool = True,
    expected_session: str | None = None,
    capture_supplied: bool = False,
    supplied_stash: Any = None,
    check_only: bool = False,
) -> tuple[bool, ParentSettingsCapture | None]:
    """Retain the exact original press through its fixed settings checkpoint.

    Args:
        screen: The original attached Console.
        guard: The original Enter scheduling guard.
        schedule: Schedule the original press, or return its async target capture.
        expected_session: Refuse a target different from this captured session.
        capture_supplied: Use the existing press capture without recapturing it.
        supplied_stash: The exact existing press capture, including None.
        check_only: Retain only the original getter, with no Send side effect.

    Returns:
        The selected flag and issued capture, or None for no original target.
    """
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement as sending
    from .console_control_settings import ConsoleDefaultConvergenceRequired, ConsoleSettingsPending

    controls = _async_action_controls(screen)
    if controls is None:
        return False, None
    _require_visible_send_helpers(screen)
    store, port = controls._store, controls._domain.presentation
    sources, revision, stamp = (
        controls._domain.sources,
        controls.source_revision,
        port._stamp(),
    )
    original_active = store.active_session_id
    session_id = screen._console_visible_draft_session_id
    if session_id is None:
        try:
            session_id = screen._console_visible_send_session_id()
        except (ConsoleDefaultConvergenceRequired, ConsoleSettingsPending) as due:
            if original_active is None:
                witness = getattr(due, "created_session", None)
                if type(witness) is not tuple or len(witness) != 8:
                    raise
                (
                    controller,
                    actual_store,
                    session,
                    incarnation,
                    settings,
                    settings_revision,
                    binding,
                    task,
                ) = witness
                if (
                    controller is not screen._session
                    or actual_store is not store
                    or task is not asyncio.current_task()
                    or store._sessions.get(session.id) is not session
                    or store.active_session_id != session.id
                    or session.incarnation_id != incarnation
                    or session.settings is not settings
                    or session.settings_revision != settings_revision
                    or session.conversation_binding_revision != binding
                ):
                    raise ParentSettingsOriginChanged(
                        "Visible Send creation changed"
                    ) from due
                session_id = session.id
            else:
                if store.active_session_id != original_active:
                    raise ParentSettingsOriginChanged(
                        "Visible Send target changed"
                    ) from due
                session_id = original_active
    controls._domain.require_current(sources)
    if controls.source_revision != revision or not port._current_visit(stamp):
        raise ParentSettingsOriginChanged("Visible Send origin changed")
    if session_id is None:
        return True, None
    if expected_session is not None and session_id != expected_session:
        raise ParentSettingsOriginChanged("Visible Send captured target changed")
    press = sending._capture_settings_press(
        screen,
        session_id,
        guard,
        schedule,
        capture_supplied,
        supplied_stash,
        check_only,
    )
    capture = issue_parent_settings_capture(controls, "visible-send", press)
    controls._queue_session_capture(capture, capture)
    if schedule:

        def completed(result):
            if result.cancelled():
                return
            error = result.exception()
            if isinstance(error, ParentSettingsOriginChanged):
                screen.app_instance.notify(
                    sending.CHAT_CHANGED_COPY, severity="warning"
                )
            elif error is not None:
                screen.app.call_later(sending._raise, error)

        capture.applied.add_done_callback(completed)
    return True, capture


async def prepare_parent_visible_send_target(
    screen: ChatScreen,
    *,
    expected_session: str | None = None,
    capture_supplied: bool = False,
    supplied_stash: Any = None,
) -> tuple[bool, Any]:
    """Keep target/stash capture through its original due checkpoint.

    Args:
        screen: The original attached Console.
        expected_session: Refuse a target different from this captured session.
        capture_supplied: Use the existing press capture without recapturing it.
        supplied_stash: The exact existing press capture, including None.

    Returns:
        Selection and the settled original target/stash/input tuple, or None.

    Raises:
        ParentSettingsOriginChanged: The captured session or visit changed.
        ParentSettingsSourceChanged: The original defining source changed.
    """
    from .console_guarded_prompt import join_guarded_prompt

    selected, capture = retain_parent_visible_send(
        screen,
        schedule=False,
        expected_session=expected_session,
        capture_supplied=capture_supplied,
        supplied_stash=supplied_stash,
    )
    if not selected:
        return False, None
    return True, None if capture is None else await join_guarded_prompt(capture)


def begin_parent_visible_target_check(
    screen: ChatScreen,
    expected_session: str,
) -> tuple[bool, str | None, ParentSettingsCapture | None]:
    """Run the original getter, retaining only its actual due boundary.

    Args:
        screen: The original attached Console.
        expected_session: The already captured action target.

    Returns:
        Selection, immediate target and any retained due capture.

    Raises:
        ParentSettingsSourceChanged: A defining owner or callback changed.
    """
    from .console_control_settings import ConsoleDefaultConvergenceRequired, ConsoleSettingsPending

    if _async_action_controls(screen) is None:
        return False, None, None
    _require_visible_send_helpers(screen)
    try:
        return True, screen._console_visible_send_session_id(), None
    except (ConsoleDefaultConvergenceRequired, ConsoleSettingsPending):
        try:
            _selected, capture = retain_parent_visible_send(
                screen,
                schedule=False,
                expected_session=expected_session,
                check_only=True,
            )
        except ParentSettingsOriginChanged:
            return True, None, None
        return True, None, capture


async def prepare_parent_visible_target_check(
    owner: Any,
    expected_session: str,
) -> tuple[bool, str | None]:
    """Settle a registered Hooks or painted-visit getter at its original check.

    Args:
        owner: The originally constructed Hooks or session controller.
        expected_session: The action's original captured target.

    Returns:
        Whether this is selected and the original getter's current result.

    Raises:
        ParentSettingsSourceChanged: The registered callback or source changed.
        ParentSettingsOriginChanged: The registered owner retired.
    """
    from tldw_chatbook.UI.Console_Modules import wiring
    from .console_guarded_prompt import join_guarded_prompt

    binding = wiring._selected_visible_check_bindings.get(owner)
    if binding is None:
        return False, None
    screen_ref, name, callback_ref = binding
    callback = callback_ref()
    screen = screen_ref()
    if screen is None:
        raise ParentSettingsOriginChanged("Visible target owner retired")
    if _async_action_controls(screen) is None:
        return False, None
    _require_visible_send_helpers(screen)
    _require_visible_check_callback(owner, screen, name, callback)
    # Preserve the original painted accessor's is_current short circuit.
    if name == "_painted_session_accessor" and not screen.is_current:
        return True, None
    selected, target, capture = begin_parent_visible_target_check(
        screen, expected_session
    )
    if capture is not None:
        try:
            target = await join_guarded_prompt(capture)
        except ParentSettingsOriginChanged:
            return True, None
        _require_visible_check_callback(owner, screen, name, callback)
    if name == "_painted_session_accessor":
        return True, callback() if target == expected_session else None
    return selected, target


def _require_visible_check_callback(owner, screen, name, callback):
    from tldw_chatbook.UI.Console_Modules import wiring
    from types import FunctionType

    original, code = wiring._selected_prompt_wiring_definition
    if (
        wiring.build_console_controllers is not original
        or original.__code__ is not code
        or owner
        is not (
            screen._session if name == "_painted_session_accessor" else screen._hooks
        )
        or getattr(owner, name) is not callback
        or type(callback) is not FunctionType
        or callback.__code__ not in wiring._selected_visible_check_codes
        or callback.__globals__ is not vars(wiring)
        or callback.__closure__ is None
        or len(callback.__closure__) != 1
        or callback.__closure__[0].cell_contents is not screen
    ):
        raise ParentSettingsSourceChanged("Visible target callback changed")
    from tldw_chatbook.UI.Console_Modules import hooks, session

    module = session if name == "_painted_session_accessor" else hooks
    for method, function, code in module._selected_visible_check_helpers:
        if (
            getattr_static(owner, method) is not function
            or function.__code__ is not code
        ):
            raise ParentSettingsSourceChanged("Visible target continuation changed")


def _require_settings_read_target(capture, purpose, args):
    """Refuse old settings-open facts before using any replacement target."""
    if purpose == "settings-context-current":
        controller, store, session_id = args[:3]
    elif purpose == "settings-display-prepare":
        controller, store, _draft, session_id = args[0][:4]
    elif purpose == "settings-modal-prepare":
        controller, store, session_id = args[1][:3]
    else:
        return
    if (
        controller is not capture.controls._controller
        or store is not capture.controls._store
        or session_id
        != (
            capture.session_id
            if type(capture) is ParentSettingsCapture
            else capture.identity.session_id
        )
    ):
        raise ParentSettingsOriginChanged("Settings preparation target changed")


def _require_settings_read_helpers(screen):
    from tldw_chatbook.UI.Screens import chat_screen

    if type(screen) is not chat_screen.ChatScreen:
        raise ParentSettingsSourceChanged("Settings read receiver changed")
    for name, descriptor, function, code in chat_screen._selected_settings_read_helpers:
        if (
            getattr_static(screen, name) is not descriptor
            or function.__code__ is not code
        ):
            raise ParentSettingsSourceChanged("Settings read helper changed")
    from tldw_chatbook.UI.Console_Modules import (
        connection_probe,
        model_switcher,
        console_spend_projection,
    )

    for module in (connection_probe, model_switcher, console_spend_projection):
        for name, function, code in module._selected_settings_read_helpers:
            if getattr(module, name) is not function or function.__code__ is not code:
                raise ParentSettingsSourceChanged("Settings route helper changed")


def _require_action_helpers(capture):
    from tldw_chatbook.UI.Screens import chat_screen
    from tldw_chatbook.UI.Console_Modules import prompts, session

    screen = capture.port._view
    if capture.purpose == "initial-compose":
        _require_initial_compose_phase(capture.checked_config)
    if type(screen) is not chat_screen.ChatScreen:
        raise RuntimeError("Parent settings requires its original Console")
    receivers = [(screen, chat_screen)]
    if capture.purpose in {
        "prompt-component",
        "system-prompt",
        "prompt-picker-result",
        "prompt-editor-result",
    }:
        for name, function, code in chat_screen._selected_prompt_post_helpers:
            if (
                getattr_static(screen, name) is not function
                or function.__code__ is not code
            ):
                raise ParentSettingsSourceChanged("Prompt post helper changed")
    if capture.purpose in _READ_ACTION_PURPOSES:
        _require_settings_read_helpers(screen)
    if capture.purpose == "workbench-help":
        for name, function, code in chat_screen._selected_workbench_help_helpers:
            if (
                getattr_static(screen, name) is not function
                or function.__code__ is not code
            ):
                raise ParentSettingsSourceChanged("Help helper definition changed")
    if capture.purpose == "prefill-command":
        _require_prefill_message(screen)
        for name, function, code in chat_screen._selected_prefill_settings_helpers:
            if (
                getattr_static(screen, name) is not function
                or function.__code__ is not code
            ):
                raise ParentSettingsSourceChanged("Prefill helper definition changed")
    if capture.purpose == "early-full-defaults":
        from tldw_chatbook.UI.Console_Modules import config_sync

        for name, function, code in config_sync._selected_core_phase_definitions:
            if (
                getattr(config_sync, name) is not function
                or function.__code__ is not code
            ):
                raise RuntimeError("Console core phase definition changed")
    if capture.purpose in _ACTION_PURPOSES or capture.purpose in {
        "system-prompt",
        "realtime-instructions",
        "attachment-completion",
        "guarded-prompt",
        "prompt-target",
        "initial-compose",
    }:
        if (
            type(screen._session) is not session.ConsoleSessionController
            or screen._session._screen is not screen
        ):
            raise RuntimeError("Parent settings session controller changed")
        receivers.append((screen._session, session))
    if capture.purpose in {
        "prompt-modal",
        "prompt-picker-result",
        "prompt-editor-result",
        "prompt-editor-open",
    }:
        if (
            type(screen._prompts) is not prompts.ConsolePromptsController
            or screen._prompts._screen is not screen
        ):
            raise RuntimeError("Parent settings prompt controller changed")
        receivers.append((screen._prompts, prompts))
    for receiver, module in receivers:
        for name, function, code in module._selected_parent_settings_helpers:
            if (
                getattr_static(receiver, name) is not function
                or function.__code__ is not code
            ):
                raise RuntimeError("Parent settings helper definition changed")


@contextmanager
def initial_parent_settings_action(
    screen: ChatScreen, purpose: str, args: tuple[Any, ...]
) -> Iterator[None]:
    """Retain only an original modal/sidechat ensure-created target."""
    from .console_control_settings import ConsoleDefaultConvergenceRequired
    from .console_domain_services import _definition

    if purpose not in {"prompt-modal", "sidechat-open"}:
        raise ValueError("Unknown initial parent action")
    controls = _async_action_controls(screen)
    if controls is None or controls._store.active_session_id is not None:
        yield
        return
    from .console_guarded_prompt import _require_target_ensure

    _require_target_ensure(screen)
    method, original = _parent_action(screen, purpose)
    if method.__func__ is not original:
        raise ParentSettingsSourceChanged("Initial settings action changed")
    store, port = controls._store, controls._domain.presentation
    sources, revision, stamp, task = (
        controls._domain.sources,
        controls.source_revision,
        port._stamp(),
        asyncio.current_task(),
    )
    composer = screen._console_composer_or_none()
    try:
        yield
    except ConsoleDefaultConvergenceRequired as due:
        witness = getattr(due, "created_session", None)
        if type(witness) is not tuple or len(witness) != 8:
            raise
        (
            controller,
            actual_store,
            session,
            incarnation,
            settings,
            setting_revision,
            binding,
            creator,
        ) = witness
        controls._require_graph()
        controls._domain.require_current(sources)
        if controls.source_revision != revision:
            raise ParentSettingsDeferred("Initial settings action was fenced") from due
        if (
            controller is not screen._session
            or actual_store is not store
            or creator is not task
            or asyncio.current_task() is not task
            or store._sessions.get(session.id) is not session
            or store.active_session_id != session.id
            or session.incarnation_id != incarnation
            or session.settings is not settings
            or session.settings_revision != setting_revision
            or session.conversation_binding_revision != binding
            or not port._current_visit(stamp)
            or screen._console_composer_or_none() is not composer
        ):
            raise ParentSettingsOriginChanged(
                "Initial settings action target changed"
            ) from due
        capture = issue_parent_settings_capture(controls, purpose, args)
        capture.action_definition = _definition(original)
        controls._queue_session_capture(capture, capture)
        capture.applied.add_done_callback(
            lambda result: None if result.cancelled() else result.exception()
        )


def defer_parent_settings_action(screen, purpose, args):
    """Queue an existing session's complete selected action in original order."""
    from .console_domain_services import _definition
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _readiness_runtime,
    )

    runtime = _readiness_runtime(screen)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return False
    if purpose not in _ACTION_PURPOSES:
        raise ValueError("Unknown parent settings action")
    for active in controls._parent_settings_captures:
        if (
            active.operation_task is asyncio.current_task()
            and active.started
            and active.purpose == purpose
        ):
            require_parent_settings_capture(active, controls)
            return False
    session_id = controls._store.active_session_id
    if session_id is None:
        # The original creation entrance has no existing durable target or
        # pending snapshot to observe; let it create the actual session.
        return False
    method, original = _parent_action(screen, purpose)
    if method.__func__ is not original:
        raise RuntimeError("Parent settings action definition changed")
    capture = issue_parent_settings_capture(controls, purpose, args)
    capture.action_definition = _definition(original)
    controls._queue_session_capture(capture, capture)
    capture.applied.add_done_callback(
        lambda result: None if result.cancelled() else result.exception()
    )
    return True


def _require_prefill_message(screen):
    from types import FunctionType
    from .console_domain_services import _definition_current
    from tldw_chatbook.UI.Console_Modules import wiring
    from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController

    binding = wiring._selected_guarded_prompt_bindings.get(screen._prompts)
    owner = getattr_static(screen, "_message", None)
    if (
        binding is None
        or len(binding) != 6
        or binding[4]() is not owner
        or type(owner) is not ConsoleMessageController
        or owner._screen is not screen
        or owner.app_instance is not screen.app_instance
    ):
        raise ParentSettingsSourceChanged("Prefill message construction changed")
    for name, original, record in _PREFILL_MESSAGE_DEFINITIONS:
        if getattr_static(owner, name) is not original or not _definition_current(
            record
        ):
            raise ParentSettingsSourceChanged("Prefill message definition changed")
    for name, codes in wiring._selected_prefill_message_accessor_codes:
        callback = getattr_static(owner, name)
        registered = next((ref() for edge, ref in binding[5] if edge == name), None)
        if (
            callback is not registered
            or type(callback) is not FunctionType
            or callback.__code__ not in codes
            or callback.__globals__ is not vars(wiring)
            or callback.__defaults__ is not None
            or callback.__kwdefaults__ is not None
            or callback.__closure__ is None
            or len(callback.__closure__) != 1
            or callback.__closure__[0].cell_contents is not screen
        ):
            raise ParentSettingsSourceChanged("Prefill message callback changed")


def _async_action_controls(screen):
    from . import console_guarded_prompt as guarded

    app = getattr_static(screen, "app_instance", None)
    runtime = getattr_static(app, "console_runtime", None) if app is not None else None
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    return None if controls is None else guarded._controls(screen._prompts)


async def run_parent_async_settings_action(
    screen: ChatScreen, purpose: str, args: tuple[Any, ...]
) -> bool | tuple[bool, Any]:
    """Retain a fixed editor, prefill, or help action through completion.

    Args:
        screen: The original attached Console.
        purpose: Editor opening, the prefill command, or workbench help.
        args: Original positional action arguments.

    Returns:
        Whether the selected parent completed the action; fixed read phases
        also return their original result in a pair with that selected flag.
    """
    from . import console_guarded_prompt as guarded
    from .console_control_settings import ConsoleDefaultConvergenceRequired
    from .console_domain_services import _definition

    read_action = purpose in _READ_ACTION_PURPOSES
    if (
        purpose
        not in {"prompt-editor-open", "prefill-command", "workbench-help"}
        | _READ_ACTION_PURPOSES
    ):
        raise ValueError("Unknown asynchronous settings action")
    controls = _async_action_controls(screen)
    if controls is None:
        return (False, None) if read_action else False
    method, original = _parent_action(screen, purpose)
    if method.__func__ is not original:
        raise ParentSettingsSourceChanged("Async settings action changed")
    guarded._require_target_ensure(screen)
    if read_action:
        owner = settings_inline_source(controls._runtime) or settings_inline_parent(
            controls._runtime
        )
        actual_task = asyncio.current_task()
        direct_owner = owner is not None and (
            (
                type(owner) is ParentSettingsCapture
                and owner.operation_task is actual_task
            )
            or controls._settings_command_tasks.get(actual_task) is owner
        )
        if direct_owner:
            _require_settings_read_helpers(screen)
            _require_settings_read_target(owner, purpose, args)
            try:
                return True, method(*args)
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(owner)
                with due_settings_phase(request):
                    _require_settings_read_helpers(screen)
                    _require_settings_read_target(owner, purpose, args)
                    return True, method(*args)
    store, port = controls._store, controls._domain.presentation
    sources, revision, stamp = (
        controls._domain.sources,
        controls.source_revision,
        port._stamp(),
    )
    composer = (
        screen._console_composer_or_none() if purpose == "prefill-command" else None
    )
    if composer is not None:
        guarded._require_composer(composer, screen)
    draft = (
        (
            composer.capture_draft_snapshot().generation,
            composer.edit_serial,
            composer.draft_text(),
        )
        if composer is not None
        else None
    )
    if store.active_session_id is None:
        # Only the original body may create its initial session. A cold due
        # handoff names that exact creation; ordered-resume remains original.
        try:
            if purpose == "workbench-help" or read_action:
                value = method(*args)
            else:
                value = await method(*args)
            return (True, value) if read_action else True
        except ConsoleDefaultConvergenceRequired as due:
            witness = getattr(due, "created_session", None)
            if type(witness) is not tuple or len(witness) != 8:
                raise
            (
                controller,
                actual_store,
                session,
                incarnation,
                settings,
                settings_revision,
                binding,
                task,
            ) = witness
            controls._require_graph()
            controls._domain.require_current(sources)
            guarded._require_target_ensure(screen)
            if controls.source_revision != revision:
                raise ParentSettingsDeferred(
                    "Async settings creation was fenced"
                ) from due
            if (
                controller is not screen._session
                or actual_store is not store
                or task is not asyncio.current_task()
                or store._sessions.get(session.id) is not session
                or store.active_session_id != session.id
                or session.incarnation_id != incarnation
                or session.settings is not settings
                or session.settings_revision != settings_revision
                or session.conversation_binding_revision != binding
                or not port._current_visit(stamp)
            ):
                raise ParentSettingsOriginChanged(
                    "Async settings creation changed"
                ) from due
    capture = issue_parent_settings_capture(controls, purpose, (args, composer, draft))
    capture.action_definition = _definition(original)
    controls._queue_session_capture(capture, capture)
    value = await guarded.join_guarded_prompt(capture)
    return (True, value) if read_action else True


def _async_settings_action(screen):
    """Resolve only the exact retained Task of the two fixed async actions."""
    from . import console_guarded_prompt as guarded

    controls = _async_action_controls(screen)
    if controls is None:
        return None
    capture = settings_inline_parent(controls._runtime)
    if capture is None or capture.purpose not in {
        "prompt-editor-open",
        "prefill-command",
    }:
        return None
    require_parent_settings_capture(capture)
    _require_action_helpers(capture)
    guarded._require_target_ensure(screen)
    return capture


async def ensure_parent_async_action_settings(screen: ChatScreen) -> Any:
    """Continue initial original ensure after its actual due read.

    Args:
        screen: The original retained action receiver.

    Returns:
        The original effective settings result.

    Raises:
        ConsoleDefaultConvergenceRequired: An unowned legacy ensure needs a read.
        ParentSettingsSourceChanged: The selected source or defining edge changed.
    """
    from .console_control_settings import ConsoleDefaultConvergenceRequired

    capture = _async_settings_action(screen)
    try:
        return screen._session._ensure_active_console_session_settings()
    except ConsoleDefaultConvergenceRequired:
        if capture is None:
            raise
        due = await settle_parent_action_defaults(capture)
        with due_settings_phase(due):
            _async_settings_action(screen)
            return screen._session._ensure_active_console_session_settings()


async def mutate_parent_prefill(
    screen: ChatScreen, value: Any, *, seed: bool = False
) -> Any:
    """Run the fixed prefill or original settings-less seed mutation.

    Args:
        screen: The retained prefill receiver.
        value: Original pinned text, or the original seed settings.
        seed: Select the fixed settings-less seed phase.

    Returns:
        The original mutation result, or None for legacy composition.

    Raises:
        ParentSettingsSourceChanged: The mutation lacks its exact prefill owner.
    """
    from .console_control_settings import (
        PendingSettingsApply,
        retain_settings_operation,
    )

    capture = _async_settings_action(screen)
    if capture is None:
        return None
    if capture.purpose != "prefill-command":
        raise ParentSettingsSourceChanged("Only prefill owns this mutation")
    operation = PendingSettingsApply(
        capture.controls,
        capture,
        capture,
        mutation=("replacement" if seed else "pinned_prefill", value),
    )
    result = await retain_settings_operation(operation)
    _async_settings_action(screen)
    return result


async def finish_parent_prefill_core(screen: ChatScreen) -> bool:
    """Complete original core and summary posts without replaying mutation.

    Args:
        screen: The retained prefill receiver.

    Returns:
        True after selected posts; False for original legacy composition.

    Raises:
        ParentSettingsSourceChanged: The post phase lost its exact owner.
    """
    from .console_control_settings import ConsoleDefaultConvergenceRequired

    capture = _async_settings_action(screen)
    if capture is None:
        return False
    if capture.purpose != "prefill-command":
        raise ParentSettingsSourceChanged("Only prefill owns this post phase")
    try:
        screen._sync_console_chat_core_state()
        screen._sync_console_settings_summary()
    except ConsoleDefaultConvergenceRequired:
        due = await settle_parent_action_defaults(capture)
        with due_settings_phase(due):
            _async_settings_action(screen)
            screen._sync_console_chat_core_state()
            screen._sync_console_settings_summary()
    return True


async def finish_parent_prefill_output(
    screen: ChatScreen, message: str, *, clear: bool = True
) -> bool:
    """Keep original clear and output order on the admitted chat and draft.

    Args:
        screen: The retained prefill receiver.
        message: Original confirmation or status text.
        clear: Clear only the unchanged original composer draft.

    Returns:
        True after selected output; False for original legacy composition.

    Raises:
        ParentSettingsOriginChanged: The original chat or attachment changed.
        ParentSettingsSourceChanged: A selected message or composer edge changed.
    """
    from . import console_guarded_prompt as guarded

    capture = _async_settings_action(screen)
    if capture is None:
        return False
    if capture.purpose != "prefill-command":
        raise ParentSettingsSourceChanged("Only prefill owns this output")
    _, composer, draft = capture.checked_config
    if clear and screen._console_composer_or_none() is composer:
        if composer is not None:
            guarded._require_composer(composer, screen)
            from .console_domain_services import _definition_current

            if getattr_static(composer, "clear_draft") is not _PREFILL_CLEAR_DEFINITION[
                0
            ] or not _definition_current(_PREFILL_CLEAR_DEFINITION):
                raise ParentSettingsSourceChanged("Prefill composer clear changed")
        if (
            composer is None
            or (
                composer.capture_draft_snapshot().generation,
                composer.edit_serial,
                composer.draft_text(),
            )
            == draft
        ):
            screen._clear_console_composer_draft()
    require_parent_settings_capture(capture)
    await screen._append_native_console_system_message(
        message, session_id=capture.session_id
    )
    return True


@dataclass(slots=True)
class _AttachmentSettingsOrigin:
    controls: Any
    sources: Any
    revision: int
    stamp: tuple
    session: Any
    incarnation: str | None
    composer: Any


def capture_parent_attachment_origin(
    screen: ChatScreen,
) -> _AttachmentSettingsOrigin | None:
    """Capture the actual attachment target before its external I/O.

    Args:
        screen: The original attachment receiver.

    Returns:
        Its checked origin, or None for legacy composition.

    Raises:
        ParentSettingsSourceChanged: An original selected attachment edge changed.
    """
    runtime = screen._console_runtime()
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    controls._require_graph()
    _require_attachment_helpers(screen)
    session = controls._store._sessions.get(controls._store.active_session_id)
    origin = _AttachmentSettingsOrigin(
        controls,
        controls._domain.sources,
        controls.source_revision,
        controls._domain.presentation._stamp(),
        session,
        session.incarnation_id if session is not None else None,
        screen._console_composer_or_none(),
    )
    _require_attachment_origin(screen, origin)
    return origin


def _require_attachment_helpers(screen):
    from tldw_chatbook.UI.Screens import chat_screen

    if type(screen) is not chat_screen.ChatScreen:
        raise ParentSettingsSourceChanged("Console attachment receiver changed")
    for name, function, code in chat_screen._selected_attachment_completion_helpers:
        if (
            getattr_static(screen, name) is not function
            or function.__code__ is not code
        ):
            raise ParentSettingsSourceChanged(
                "Console attachment completion definition changed"
            )


def _require_attachment_origin(screen, origin, *, created=False):
    controls = origin.controls
    controls._require_graph()
    controls._domain.require_current(origin.sources)
    port, store = controls._domain.presentation, controls._store
    if controls.source_revision != origin.revision:
        raise ParentSettingsDeferred("Console attachment source fenced")
    if port._view is not screen or not port._current_visit(origin.stamp):
        raise ParentSettingsOriginChanged("Console attachment origin changed")
    _require_attachment_helpers(screen)
    if screen._console_composer_or_none() is not origin.composer:
        raise ParentSettingsOriginChanged("Console attachment composer changed")
    session = store._sessions.get(store.active_session_id)
    if created:
        if origin.session is not None:
            raise ParentSettingsSourceChanged(
                "Console attachment creation already bound"
            )
        # Only the synchronous original no-target creation checkpoint calls
        # this branch, with no wait between its creation and this binding.
        origin.session = session
        origin.incarnation = session.incarnation_id if session is not None else None
    if session is not origin.session or (
        session is not None and session.incarnation_id != origin.incarnation
    ):
        raise ParentSettingsOriginChanged("Console attachment session changed")
    return controls


def _finish_parent_attachment(screen, origin, phase, data):
    from .console_chat_store import MAX_PENDING_ATTACHMENTS

    if phase == "validate":
        return None
    if phase == "paths-target":
        return screen._prepare_console_attachment_target()
    if phase == "paths-cap":
        return (
            len(origin.controls._store.pending_attachments(origin.session.id))
            >= MAX_PENDING_ATTACHMENTS
        )
    if phase == "clipboard-image":
        return screen._finish_console_clipboard_attachment(data)
    if phase == "file":
        return screen._finish_console_attachment(data)
    raise ValueError("Unknown attachment completion phase")


async def complete_parent_attachment(
    screen: ChatScreen, origin: _AttachmentSettingsOrigin | None, phase: str, data: Any
) -> tuple[bool, Any]:
    """Retain a fixed attachment completion against its captured origin.

    Args:
        screen: The original attachment receiver.
        origin: The pre-I/O target witness, or None for legacy composition.
        phase: One registered attachment completion phase.
        data: The original result needed by that phase.

    Returns:
        A selected-path flag and the original completion result.

    Raises:
        ValueError: The phase is not a fixed attachment completion.
        ParentSettingsOriginChanged: The original target no longer owns completion.
    """
    if origin is None:
        return False, None
    from .console_control_settings import ConsoleDefaultConvergenceRequired

    controls = _require_attachment_origin(screen, origin)
    if phase not in {
        "validate",
        "paths-target",
        "paths-cap",
        "clipboard-image",
        "file",
    }:
        raise ValueError("Unknown attachment completion phase")
    if origin.session is None:
        try:
            value = _finish_parent_attachment(screen, origin, phase, data)
        except ConsoleDefaultConvergenceRequired:
            _require_attachment_origin(screen, origin, created=True)
            if origin.session is None:
                raise
        else:
            _require_attachment_origin(screen, origin, created=True)
            return True, value
    capture = issue_parent_settings_capture(
        controls, "attachment-completion", (origin, phase, data)
    )
    controls._queue_session_capture(capture, capture)
    cancelled = False
    while not capture.applied.done():
        try:
            await asyncio.shield(capture.applied)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    value = capture.applied.result()
    if cancelled:
        raise asyncio.CancelledError
    return True, value


def _require_realtime_callbacks(owner, screen):
    from types import FunctionType
    from tldw_chatbook.UI.Console_Modules import realtime, wiring

    binding = wiring._selected_realtime_bindings.get(owner)
    if (
        type(owner) is not realtime.ConsoleRealtimeController
        or binding is None
        or binding[0]() is not screen
        or screen._realtime is not owner
    ):
        raise ParentSettingsSourceChanged("Console realtime wiring receiver changed")
    original, code = wiring._selected_realtime_wiring_definition
    if (
        wiring.build_console_controllers is not original
        or original.__code__ is not code
    ):
        raise ParentSettingsSourceChanged("Console realtime wiring definition changed")
    for name, codes in wiring._selected_realtime_accessor_codes:
        callback = getattr_static(owner, name, None)
        registered = next(
            (reference() for edge, reference in binding[1] if edge == name), None
        )
        if (
            callback is not registered
            or type(callback) is not FunctionType
            or callback.__code__ not in codes
            or callback.__globals__ is not vars(wiring)
            or callback.__defaults__ is not None
            or callback.__kwdefaults__ is not None
            or callback.__closure__ is None
            or len(callback.__closure__) != 1
            or callback.__closure__[0].cell_contents is not screen
        ):
            raise ParentSettingsSourceChanged(
                "Console realtime settings callback changed"
            )


def realtime_settings_runtime(owner: Any) -> Any:
    """Discover custody before invoking selected settings edges.

    Args:
        owner: The original realtime controller.

    Returns:
        Its original runtime after selected wiring validation.

    Raises:
        ParentSettingsSourceChanged: A registered receiver or callback changed.
    """
    from tldw_chatbook.UI.Console_Modules import wiring

    binding = wiring._selected_realtime_bindings.get(owner)
    screen = binding[0]() if binding is not None else None
    runtime = (
        getattr_static(screen.app_instance, "console_runtime", None)
        if screen is not None
        else None
    )
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        runtime = owner._runtime_accessor()
        if (
            getattr(
                getattr(runtime, "_execution_service", None), "_local_controls", None
            )
            is not None
        ):
            raise ParentSettingsSourceChanged(
                "Console realtime selected association missing"
            )
        return runtime
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    if (
        type(screen) is not ChatScreen
        or runtime.view is not screen
        or controls._domain.presentation._view is not screen
        or controls._domain.presentation._app is not screen.app_instance
    ):
        raise ParentSettingsOriginChanged("Console realtime view changed")
    _require_realtime_callbacks(owner, screen)
    return runtime


def _parent_realtime_owner(capture):
    from tldw_chatbook.UI.Console_Modules import realtime

    require_parent_settings_capture(capture)
    owner = capture.port._view._realtime
    session, attempt = capture.checked_config
    _require_realtime_callbacks(owner, capture.port._view)
    if (
        type(session) is not realtime.ConsoleRealtimeSession
        or owner._runtime_accessor() is not capture.controls._runtime
        or owner._chat_store_accessor() is not capture.controls._store
    ):
        raise ParentSettingsSourceChanged("Console realtime settings owner changed")
    for name, function, code in realtime._selected_parent_settings_helpers:
        if getattr_static(owner, name) is not function or function.__code__ is not code:
            raise ParentSettingsSourceChanged(
                "Console realtime settings definition changed"
            )
    if (
        owner.session is not session
        or session.connect_attempt != attempt
        or session.console_session_id != capture.session_id
    ):
        raise ParentSettingsOriginChanged("Console realtime attempt changed")
    return owner


async def read_parent_realtime_instructions(
    owner: Any, session: Any, attempt: Any
) -> str | None:
    """Read instructions while the original worker retains provider ownership.

    Args:
        owner: The original realtime controller.
        session: The actual current realtime session.
        attempt: Its original connection-attempt witness.

    Returns:
        The original instructions, including the original None result.

    Raises:
        ParentSettingsOriginChanged: The current view or realtime attempt changed.
        ParentSettingsSourceChanged: A selected settings edge changed.
    """
    runtime = realtime_settings_runtime(owner)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return owner._console_realtime_instructions()
    if runtime.view is None or runtime.view._realtime is not owner:
        raise ParentSettingsOriginChanged("Console realtime view changed")
    capture = issue_parent_settings_capture(
        controls, "realtime-instructions", (session, attempt)
    )
    controls._queue_session_capture(capture, capture)
    cancelled = False
    while not capture.applied.done():
        try:
            await asyncio.shield(capture.applied)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    value = capture.applied.result()
    if cancelled:
        raise asyncio.CancelledError
    return value


def _prepare_compact_action(capture, args):
    screen = capture.port._view
    with screen._console_derivation_scope():
        updates, settings = screen._prepare_compact_shell_controls(
            provider=args[0],
            model=args[1],
            temperature=args[2],
        )
        if updates and settings is not None:
            store, session = (
                screen._session._prepare_active_console_session_replacement()
            )
            if store is not capture.controls._store or session is not capture.session:
                raise RuntimeError("Compact settings target changed")
        return updates, settings


def _parent_post_settings_state(session):
    return (
        session.settings,
        session.generation_settings_revision,
        session.settings_revision,
        session.context_policy_revision,
    )


def _advance_parent_post_settings(capture, before, request):
    """Validate only this fixed post's exact completed convergence chain."""
    _due_settings_read_current(request)
    require_parent_settings_capture(capture)
    if (
        request.source is not capture
        or request.task is not asyncio.current_task()
        or not request.active
        or request.consumed
    ):
        raise ParentSettingsSourceChanged("Prompt post due phase changed")
    expected = before
    seen = set()
    for issued, operation, entry, dependency, checked in request.convergences:
        source, task, projection, handoff, key = entry
        prepared = operation.convergence.prepared
        native_settled = (
            operation.native_required is True
            and operation.native_entered
            and operation.retirement is not None
            and operation.retirement.confirmed
        ) or (
            operation.native_required is False
            and not operation.native_entered
            and operation.retirement is None
        )
        if (
            id(operation) in seen
            or source is not capture
            or task is not request.task
            or issued.source is not capture
            or issued.task is not task
            or issued.projection is not projection
            or issued.producer is not handoff[2]._task
            or issued.handoff is not handoff
            or handoff[0] is not key
            or issued.result is None
            or issued.result.value is not checked
            or prepared.config is not checked
            or operation.session is not capture.session
            or operation.incarnation != capture.incarnation_id
            or operation.binding != capture.session.conversation_binding_revision
            or operation.conversation != capture.session.persisted_conversation_id
            or operation.source.sources is not capture.sources
            or operation.source.source_revision != capture.source_revision
            or operation.live_before_state is None
            or operation.live_before_state[0] is not expected[0]
            or operation.live_before_state[1:] != expected[1:]
            or operation.owned_state is None
            or operation.commit is not operation.owned_state[0]
            or not native_settled
            or operation.error is not None
            or not operation.finished.done()
            or operation.actual_task is None
            or not operation.actual_task.done()
            or operation.actual_task.cancelled()
            or operation.actual_task.exception() is not None
        ):
            raise ParentSettingsSourceChanged("Guarded post convergence chain changed")
        if dependency is None:
            if operation.source is not capture:
                raise ParentSettingsSourceChanged("Guarded post inline source changed")
        elif (
            dependency.capture is not operation.source
            or dependency.producer is not issued.producer
            or dependency.projection is not projection
            or dependency.handoff is not handoff
            or dependency.key is not key
        ):
            raise ParentSettingsSourceChanged("Guarded post readiness lineage changed")
        seen.add(id(operation))
        expected = operation.owned_state
    if (
        expected[0] is not capture.session.settings
        or expected[1:] != _parent_post_settings_state(capture.session)[1:]
    ):
        raise ParentSettingsOriginChanged("Guarded post has an unowned settings change")
    return expected


def _capture_prompt_post_draft(capture):
    """Keep original draft/composer ownership before the admitted write wait."""
    from .console_guarded_prompt import _require_composer

    screen = capture.stamp[1]
    composer = screen._console_composer_or_none()
    if composer is not None:
        _require_composer(composer, screen)
    session = capture.session
    return (
        session.draft,
        session.draft_revision,
        session._draft_authored_token,
        composer,
        None if composer is None else composer.capture_draft_snapshot(),
        None if composer is None else composer.edit_serial,
    )


async def _finish_parent_prompt_posts(capture, operation, draft, *, flow=None):
    """Complete fixed posts once, without repeating warning or mutation."""
    from .console_control_settings import ConsoleDefaultConvergenceRequired

    state = operation.owned_state
    settled = (operation.native_required is False and not operation.native_entered) or (
        operation.native_required is True
        and operation.native_entered
        and operation.retirement is not None
        and operation.retirement.confirmed
    )
    if not settled or operation.commit is None or not operation.retired:
        raise ParentSettingsSourceChanged("Prompt write has not settled")
    screen = capture.stamp[1]
    names = (
        (
            "_sync_console_chat_core_state",
            "_sync_console_rail_system_line",
            "_sync_console_settings_summary",
            "_sync_console_control_bar",
        )
        if flow is not None
        else (
            "_sync_console_chat_core_state",
            "_sync_console_settings_summary",
            "_sync_console_control_bar",
        )
    )
    for name in names:
        require_parent_settings_capture(capture)
        _require_action_helpers(capture)
        if flow is not None:
            _require_prompt_flow(capture.controls, flow)
        if (
            state is None
            or state[0] is not capture.session.settings
            or state[1:] != _parent_post_settings_state(capture.session)[1:]
            or _capture_prompt_post_draft(capture) != draft
        ):
            raise ParentSettingsOriginChanged("Prompt post target changed")
        try:
            getattr(screen, name)()
        except ConsoleDefaultConvergenceRequired:
            request = await settle_parent_action_defaults(capture)
            with due_settings_phase(request):
                state = _advance_parent_post_settings(capture, state, request)
                if _capture_prompt_post_draft(capture) != draft:
                    raise ParentSettingsOriginChanged(
                        "Prompt post draft changed"
                    ) from None
                _require_action_helpers(capture)
                if flow is not None:
                    _require_prompt_flow(capture.controls, flow)
                getattr(screen, name)()


async def _apply_parent_system_prompt(capture, value):
    from .console_control_settings import (
        ConsoleDefaultConvergenceRequired,
        PendingSettingsApply,
        retain_settings_operation,
    )

    require_parent_settings_capture(capture)
    _require_action_helpers(capture)
    controller = capture.port._view._session
    try:
        store, session_id = controller._prepare_console_session_system_prompt()
    except ConsoleDefaultConvergenceRequired:
        request = await settle_parent_action_defaults(capture)
        with due_settings_phase(request):
            store, session_id = controller._prepare_console_session_system_prompt()
    if store is not capture.controls._store or session_id != capture.session_id:
        raise RuntimeError("System prompt action target changed")
    operation = PendingSettingsApply(
        capture.controls,
        capture,
        capture,
        mutation=("system_prompt", value),
    )
    post_draft = _capture_prompt_post_draft(capture)
    _session, persisted = await retain_settings_operation(operation)
    require_parent_settings_capture(capture)
    _require_action_helpers(capture)
    controller._warn_console_system_prompt_persistence(persisted)
    await _finish_parent_prompt_posts(capture, operation, post_draft)


async def apply_parent_system_prompt(screen, value):
    """Await the selected prompt write before its caller consumes the draft."""
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _readiness_runtime,
    )

    runtime = _readiness_runtime(screen)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return False
    capture = issue_parent_settings_capture(controls, "system-prompt", value)
    controls._queue_session_capture(capture, capture)
    cancelled = False
    while not capture.applied.done():
        try:
            await asyncio.shield(capture.applied)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    capture.applied.result()
    if cancelled:
        raise asyncio.CancelledError
    return True


def _require_prompt_flow(controls, flow):
    from types import FunctionType
    from tldw_chatbook.UI.Console_Modules import prompts, wiring

    screen = controls._domain.presentation._view
    if (
        type(flow) is not prompts._ConsolePromptImprovementFlow
        or flow._store is not controls._store
        or flow._app_instance is not controls._domain.presentation._app
    ):
        raise RuntimeError("Prompt settings flow owner changed")
    if flow._session_id != controls._store.active_session_id:
        raise ParentSettingsOriginChanged("The active Console session changed.")
    original, code = wiring._selected_prompt_wiring_definition
    callback = flow._sync_system_prompt_surfaces
    if (
        wiring.build_console_controllers is not original
        or original.__code__ is not code
        or type(callback) is not FunctionType
        or callback.__code__ is not wiring._selected_prompt_surfaces_code
        or callback.__closure__ is None
        or len(callback.__closure__) != 1
    ):
        raise RuntimeError("Prompt settings presentation source changed")
    if callback.__closure__[0].cell_contents is not screen:
        raise ParentSettingsOriginChanged("Prompt settings presentation changed")
    for name, function, code in prompts._selected_prompt_flow_definitions:
        if getattr_static(flow, name) is not function or function.__code__ is not code:
            raise RuntimeError("Prompt settings flow definition changed")


async def wait_prompt_flow_settings(flow):
    runtime = getattr(flow._app_instance, "console_runtime", None)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return True
    try:
        _require_prompt_flow(controls, flow)
        await wait_parent_settings(controls._domain.presentation._view)
    except ParentSettingsOriginChanged:
        return False
    return True


def capture_prompt_resolution_origin(flow: Any) -> tuple[Any, ...] | None:
    """Capture a prompt flow before provider I/O without granting authority.

    Args:
        flow: The original prompt-improvement flow.

    Returns:
        Its checked origin, or None for legacy composition.

    Raises:
        ParentSettingsOriginChanged: The flow no longer belongs to the active chat.
        ParentSettingsSourceChanged: A selected flow edge changed.
    """
    runtime = getattr(flow._app_instance, "console_runtime", None)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    _require_prompt_flow(controls, flow)
    port = controls._domain.presentation
    session = controls._store._sessions[flow._session_id]
    return (
        controls,
        controls._domain.sources,
        controls.source_revision,
        port._stamp(),
        session,
        session.incarnation_id,
    )


def _require_prompt_resolution_origin(flow, origin):
    controls, sources, revision, stamp, session, incarnation = origin
    _require_prompt_flow(controls, flow)
    controls._domain.require_current(sources)
    if controls.source_revision != revision:
        raise ParentSettingsSourceChanged("Prompt resolution source changed")
    if (
        controls._store._sessions.get(flow._session_id) is not session
        or session.incarnation_id != incarnation
        or not controls._domain.presentation._current_visit(stamp)
    ):
        raise ParentSettingsOriginChanged("The active Console session changed.")
    return controls


def _finish_parent_prompt_resolution(capture):
    flow, origin, phase, args = capture.checked_config
    if _require_prompt_resolution_origin(flow, origin) is not capture.controls:
        raise RuntimeError("Prompt resolution controls changed")
    if phase == "prepare-retry":
        live = flow._active_session_settings()
        return str(live.system_prompt or "") == str(args[0] or "")
    if stale := flow._stale_reason():
        raise ValueError(stale)
    if phase == "prepare-apply":
        return flow._build_provider_selection() if args[0] else None
    if phase == "apply-user":
        flow._apply_reviewed_user_prompt(args[0], revalidate=True)
        return None
    if phase == "prepare-activation":
        projection, blocker = flow._prepare_improvement_activation()
        return projection, blocker, flow._build_provider_selection()
    if phase == "prepare-manual":
        pinned = flow._pinned_resolution
        return pinned, None if pinned is not None else flow._build_provider_selection()
    if phase == "prepare-snapshot":
        projection, pinned = flow._prepare_improvement_snapshot(*args)
        return projection, pinned, flow._build_provider_selection()
    if phase == "activation":
        return flow._finish_improvement_activation(*args)
    if phase == "manual":
        return flow._finish_manual_resolution(*args)
    if phase == "snapshot":
        return flow._finish_improvement_snapshot(*args)
    raise RuntimeError("Unknown prompt resolution completion")


async def complete_parent_prompt_resolution(
    flow: Any, origin: tuple[Any, ...] | None, phase: str, args: tuple[Any, ...]
) -> tuple[bool, Any]:
    """Retain the fixed pin, disclosure or snapshot phase after provider work.

    Args:
        flow: The original prompt-improvement flow.
        origin: Its pre-I/O origin, or None for legacy composition.
        phase: One registered prompt resolution completion.
        args: Original arguments for that fixed completion.

    Returns:
        A selected-path flag and the original completion result.

    Raises:
        ValueError: The phase or original flow target is no longer valid.
        ParentSettingsSourceChanged: A selected source or flow edge changed.
    """
    if origin is None:
        return False, None
    if phase not in {
        "activation",
        "manual",
        "snapshot",
        "prepare-activation",
        "prepare-manual",
        "prepare-snapshot",
        "prepare-apply",
        "prepare-retry",
        "apply-user",
    }:
        raise ValueError("Unknown prompt resolution completion")
    try:
        controls = _require_prompt_resolution_origin(flow, origin)
        capture = issue_parent_settings_capture(
            controls, "prompt-resolution", (flow, origin, phase, args)
        )
        controls._queue_session_capture(capture, capture)
        cancelled = False
        while not capture.applied.done():
            try:
                await asyncio.shield(capture.applied)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                break
        value = capture.applied.result()
        if cancelled:
            raise asyncio.CancelledError
        return True, value
    except ParentSettingsOriginChanged as error:
        raise ValueError(str(error)) from error


def _prepare_parent_prompt_component(capture):
    """Run original final checks and User mutation immediately before System."""
    flow, value, retry, reviewed = capture.checked_config
    require_parent_settings_capture(capture)
    _require_prompt_flow(capture.controls, flow)
    if retry:
        live = flow._active_session_settings()
        if str(live.system_prompt or "") != str(value or ""):
            raise ParentSettingsOriginChanged("The live System prompt changed.")
    elif stale := flow._stale_reason():
        raise ParentSettingsOriginChanged(stale)


async def apply_parent_prompt_component(flow, value, *, retry=False, reviewed=None):
    """Return None for legacy, otherwise the original selected persisted flag."""
    runtime = getattr(flow._app_instance, "console_runtime", None)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    _require_prompt_flow(controls, flow)
    capture = issue_parent_settings_capture(
        controls, "prompt-component", (flow, value, retry, reviewed)
    )
    controls._queue_session_capture(capture, capture)
    cancelled = False
    while not capture.applied.done():
        try:
            await asyncio.shield(capture.applied)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    value = capture.applied.result()
    if cancelled:
        raise asyncio.CancelledError
    return value


async def wait_parent_settings(screen):
    """Wait before a named async action's coherent capture, retaining origin."""
    from .console_control_settings import wait_settings_settled
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _readiness_runtime,
    )

    runtime = _readiness_runtime(screen)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return
    store = controls._store
    session = store._sessions.get(store.active_session_id)
    if session is None:
        return
    port, sources = controls._domain.presentation, controls._domain.sources
    stamp = port._stamp()
    incarnation = session.incarnation_id
    previous = controls._session_capture_tails.get(session.id)
    source = settings_inline_source(runtime)
    parent = settings_inline_parent(runtime)
    if (source is not None and source.identity.session_id == session.id) or (
        parent is not None and parent.session_id == session.id
    ):
        previous = None  # The enclosing selected capture owns this order.
    task = asyncio.current_task()
    already_retained = task in controls._control_tasks
    controls._control_tasks.add(task)
    try:
        if previous is not None:
            await asyncio.shield(previous)
        await wait_settings_settled(store, session.id)
        controls._domain.require_current(sources)
        if (
            store._sessions.get(session.id) is not session
            or store.active_session_id != session.id
            or session.incarnation_id != incarnation
            or not port._current_visit(stamp)
        ):
            raise ParentSettingsOriginChanged("Console settings action origin changed")
    finally:
        if not already_retained:
            controls._control_tasks.discard(task)


def settings_inline_source(runtime):
    """Recognize only the live command or its actually awaited readiness worker."""
    from .console_control_dispatch import _command_source, require_control_source

    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    task = asyncio.current_task()
    source = _command_source.get()
    if (
        source is not None
        and controls._settings_command_tasks.get(task) is source
        and not task.done()
    ):
        return require_control_source(runtime)
    entry = controls._settings_readiness_waits.get(task)
    if entry is not None:
        original, owner, projection, handoff, key = entry
        if (
            type(original) is not ParentSettingsCapture
            and not owner.done()
            and not task.done()
            and controls._settings_command_tasks.get(owner) is original
            and original.controls is controls
            and original.sources is controls._domain.sources
            and original.identity.session_id == key[5]
            and projection._handoff_read is handoff
            and handoff[0] is key
            and handoff[2]._task is task
        ):
            controls._domain.require_current(original.sources)
            return original
    return None


def settings_inline_parent(runtime):
    """Recognize only an issued parent operation or its actually awaited producer."""
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    task = asyncio.current_task()
    for capture in controls._parent_settings_captures:
        if capture.operation_task is task and capture.started and not task.done():
            return require_parent_settings_capture(capture, controls)
    entry = controls._settings_readiness_waits.get(task)
    if entry is not None:
        capture, owner, projection, handoff, key = entry
        if (
            type(capture) is ParentSettingsCapture
            and capture.operation_task is owner
            and not owner.done()
            and not task.done()
            and capture.started
            and capture.session_id == key[5]
            and projection._handoff_read is handoff
            and handoff[0] is key
            and handoff[2]._task is task
        ):
            return require_parent_settings_capture(capture, controls)
    return None


@dataclass(eq=False, slots=True)
class _DueSettingsRead:
    source: Any
    task: asyncio.Task
    screen: Any
    projection: Any
    stamp: tuple
    session: Any
    incarnation: str
    producer: asyncio.Task | None = None
    handoff: Any = None
    result: Any = None
    key: tuple | None = None
    settings: Any = None
    active: bool = False
    consumed: bool = False
    convergences: list = field(default_factory=list)


def _record_due_settings_convergence(
    controls, checked_config, operation, dependency=None
):
    """Keep only actual completed convergence under its retained readiness waiter."""
    producer = asyncio.current_task()
    entry = controls._settings_readiness_waits.get(producer)
    if entry is None or operation is None:
        return
    source, task, projection, handoff, original_key = entry
    request = controls._due_settings_reads.get(task)
    if request is None:
        return
    if (
        request.source is not source
        or request.projection is not projection
        or handoff[2]._task is not producer
        or handoff[0] is not original_key
        or operation.convergence is None
        or operation.convergence.prepared.config is not checked_config
        or operation.session is not request.session
        or operation.controls is not controls
        or operation.error is not None
        or operation.commit is None
        or not operation.finished.done()
        or operation.actual_task is None
        or not operation.actual_task.done()
        or operation.actual_task.cancelled()
        or operation.actual_task.exception() is not None
    ):
        raise ParentSettingsSourceChanged(
            "Due convergence did not complete its original operation"
        )
    if dependency is None:
        if operation.source is not source:
            raise ParentSettingsSourceChanged("Due convergence inline source changed")
    elif (
        controls._settings_readiness_captures.get(producer) is not dependency
        or dependency.producer is not producer
        or dependency.projection is not projection
        or dependency.handoff is not handoff
        or dependency.key is not original_key
        or dependency.capture.pending_operation is not operation
        or operation.source is not dependency.capture
    ):
        raise ParentSettingsSourceChanged("Due convergence dependency changed")
    request.convergences.append((request, operation, entry, dependency, checked_config))


def _due_settings_read_current(request):
    try:
        return _validate_due_settings_read(request)
    except (ParentSettingsDeferred, ParentSettingsOriginChanged):
        raise
    except RuntimeError as error:
        raise ParentSettingsSourceChanged(str(error)) from error


def _validate_due_settings_read(request):
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _checked_config_proof_definitions_current,
        _checked_config_proof_participant_current,
        _checked_display_proofs,
        _handoff_projection_is_stock,
        _same_readiness_key,
    )
    import time
    import threading
    from tldw_chatbook.UI.Console_Modules import (
        console_spend_projection,
        session as session_module,
    )

    _require_definitions()
    for (
        name,
        function,
        code,
    ) in console_spend_projection._selected_due_read_source_checks:
        if (
            getattr(console_spend_projection, name) is not function
            or function.__code__ is not code
        ):
            raise RuntimeError("Console due read source checks changed")
    source, task = request.source, request.task
    controls = source.controls
    if (
        controls._due_settings_reads.get(task) is not request
        or task is not asyncio.current_task()
        or task.done()
        or controls._domain.presentation._view is not request.screen
        or not controls._domain.presentation._current_visit(request.stamp)
        or controls._store._sessions.get(request.session.id) is not request.session
        or controls._store.active_session_id != request.session.id
        or request.session.incarnation_id != request.incarnation
    ):
        raise ParentSettingsOriginChanged("Console due settings origin changed")
    if type(source) is ParentSettingsCapture:
        require_parent_settings_capture(source)
        if source.operation_task is not task:
            raise RuntimeError("Console due settings Task changed")
    elif controls._settings_command_tasks.get(task) is not source:
        raise RuntimeError("Console due settings command ended")
    controls._domain.require_current(source.sources)
    projection, result = request.projection, request.result
    if (
        result is None
        or request.producer is None
        or request.handoff is None
        or request.handoff[2]._task is not request.producer
        or not _handoff_projection_is_stock(projection)
        or projection.screen is not request.screen
        or projection._reads
        or not _same_readiness_key(request.key, projection._key())
        or request.session.settings is not request.settings
    ):
        raise ParentSettingsDeferred("Console due settings read did not settle current")
    proof = result._display_proof
    if (
        proof is None
        or proof not in _checked_display_proofs
        or proof.projection is not projection
        or proof.reader is not projection.read_current
        or proof.screen is not request.screen
        or proof.value is not result.value
        or result.source_before != request.key[0]
        or result.source_after != request.key[0]
        or not _same_readiness_key(proof.owner, request.key, ignore_settings=True)
        or proof.thread is not threading.current_thread()
        or proof.loop is not asyncio.get_running_loop()
    ):
        raise ParentSettingsDeferred(
            "Console due settings checked result owner changed"
        )
    if not _checked_config_proof_definitions_current(proof):
        raise ParentSettingsDeferred("Console due settings config definitions changed")
    if not _checked_config_proof_participant_current(proof):
        raise ParentSettingsDeferred("Console due settings participant refused")
    age = time.monotonic() - proof.at
    if age >= min(1.0, projection.max_age):
        raise ParentSettingsReadExpired(
            f"Console due settings checked result expired after {age:.6f}s"
        )
    from tldw_chatbook.UI.Screens import chat_screen

    for name, function, code in chat_screen._selected_due_core_definitions:
        if (
            getattr_static(request.screen, name) is not function
            or function.__code__ is not code
        ):
            raise RuntimeError("Console due core definition changed")
    controller = request.screen._session
    if (
        type(controller) is not session_module.ConsoleSessionController
        or controller._screen is not request.screen
    ):
        raise RuntimeError("Console due settings controller changed")
    for name, function, code in session_module._selected_due_ensure_definitions:
        if (
            getattr_static(controller, name) is not function
            or function.__code__ is not code
        ):
            raise RuntimeError("Console due settings ensure changed")
    if type(source) is ParentSettingsCapture:
        _require_action_helpers(source)
    return request


def complete_due_settings_read(projection, handoff, result, key):
    """Hand the exact retired read to its already registered actual waiter only."""
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _readiness_runtime,
    )

    runtime = _readiness_runtime(projection.screen)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return
    _require_definitions()
    producer = asyncio.current_task()
    entry = controls._settings_readiness_waits.get(producer)
    if entry is None:
        return
    source, task, actual_projection, actual_handoff, original_key = entry
    request = controls._due_settings_reads.get(task)
    if request is None:
        return
    if (
        request.source is not source
        or request.projection is not projection
        or actual_projection is not projection
        or actual_handoff is not handoff
        or handoff[0] is not original_key
        or handoff[2]._task is not producer
        or request.result is not None
        or projection._reads
    ):
        raise RuntimeError("Console due settings producer changed")
    request.producer, request.handoff = producer, handoff
    request.result, request.key = result, key
    request.settings = request.session.settings


async def settle_parent_action_defaults(
    source: ParentSettingsCapture | _CommandSource, *, retry_expired: bool = True
) -> _DueSettingsRead:
    """Read only after the original action/core gate reaches its due point.

    Args:
        source: The actual registered parent or retained command owner.
        retry_expired: Whether original freshness expiry permits a fresh read.

    Returns:
        The current registered one-use read and same-call convergence lineage.

    Raises:
        ParentSettingsDeferred: The original read did not settle current.
        ParentSettingsOriginChanged: The original session or visit changed.
        ParentSettingsSourceChanged: The selected source lost authority.
    """
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
        _handoff_projection_is_stock,
    )

    controls = source.controls
    screen = controls._domain.presentation._view
    task = asyncio.current_task()
    if type(source) is ParentSettingsCapture:
        require_parent_settings_capture(source)
        if source.operation_task is not task:
            raise RuntimeError("Default settings action Task changed")
        session = source.session
    else:
        if controls._settings_command_tasks.get(task) is not source:
            raise RuntimeError("Default settings command Task changed")
        session = controls._store._sessions[source.identity.session_id]
    projection = ConsoleReadinessConfigProjection.for_screen(screen)
    if not _handoff_projection_is_stock(projection):
        raise RuntimeError("Default settings readiness source changed")
    convergences = []
    while True:
        if type(source) is ParentSettingsCapture:
            require_parent_settings_capture(source)
        controls._domain.require_current(source.sources)
        if task in controls._due_settings_reads:
            raise RuntimeError("Console due settings continuation already issued")
        request = _DueSettingsRead(
            source,
            task,
            screen,
            projection,
            controls._domain.presentation._stamp(),
            session,
            session.incarnation_id,
        )
        request.convergences = convergences
        controls._due_settings_reads[task] = request
        try:
            # Initially the live read is due; on retry its exact prior result
            # expired. Join the original producer, never a second live probe.
            projection.run(lambda: None, life=0.0)
            await projection.warm()
            return _due_settings_read_current(request)
        except ParentSettingsReadExpired:
            controls._due_settings_reads.pop(task, None)
            if not retry_expired:
                raise
        except BaseException:
            controls._due_settings_reads.pop(task, None)
            raise


@contextmanager
def due_settings_phase(request: _DueSettingsRead) -> Iterator[None]:
    """One uninterrupted original phase; never carry its memo across a wait."""
    try:
        _due_settings_read_current(request)
        if request.active or request.consumed:
            raise RuntimeError("Console due settings continuation already used")
        request.active = True
        with request.screen._console_derivation_scope():
            yield
    finally:
        request.active = False
        request.source.controls._due_settings_reads.pop(request.task, None)


def due_settings_config(screen):
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _readiness_runtime,
    )

    runtime = _readiness_runtime(screen)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    request = (
        controls._due_settings_reads.get(asyncio.current_task())
        if controls is not None
        else None
    )
    if request is None or not request.active:
        return None
    return _due_settings_read_current(request).result.value


def consume_due_settings_result(controller, store, session):
    runtime = getattr(controller.app_instance, "console_runtime", None)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    request = (
        controls._due_settings_reads.get(asyncio.current_task())
        if controls is not None
        else None
    )
    if request is None or not request.active:
        return False, None
    _due_settings_read_current(request)
    if (
        request.consumed
        or request.screen._session is not controller
        or store is not controls._store
        or session is not request.session
    ):
        raise ParentSettingsSourceChanged("Console due settings ensure owner changed")
    request.consumed = True
    return True, request.settings


@asynccontextmanager
async def await_readiness_settings(
    projection: ConsoleReadinessConfigProjection,
) -> AsyncIterator[None]:
    """Bind the exact original handoff only while its command awaits settlement."""
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        _handoff_projection_is_stock,
        _readiness_runtime,
    )

    runtime = _readiness_runtime(projection.screen)
    source = settings_inline_source(runtime) or settings_inline_parent(runtime)
    session_id = (
        source.session_id
        if type(source) is ParentSettingsCapture
        else source.identity.session_id
        if source is not None
        else None
    )
    handoff = projection._handoff_read
    worker = handoff[2] if handoff is not None else None
    child = getattr(worker, "_task", None)
    if (
        source is None
        or child is None
        or child.done()
        or not _handoff_projection_is_stock(projection)
        or handoff[0][5] != session_id
    ):
        yield
        return
    controls = source.controls
    entry = (source, asyncio.current_task(), projection, handoff, handoff[0])
    previous = controls._settings_readiness_waits.get(child)
    if previous is not None:
        raise RuntimeError("Readiness settings await already has an owner")
    controls._settings_readiness_waits[child] = entry
    dependency = controls._settings_readiness_captures.get(child)
    if dependency is not None and not dependency.changed.done():
        dependency.changed.set_result(None)
    try:
        yield
    except asyncio.CancelledError:
        if (
            type(source) is ParentSettingsCapture
            and source.purpose == "initial-compose"
        ):
            # This source is the actual message-pump Task, not a retained child.
            # Defer leaving its wait scope until the exact original producer's
            # raw read/default continuation settles; preserve its live waiter.
            while not child.done():
                try:
                    await asyncio.shield(child)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if not child.cancelled():
                child.exception()
        raise
    finally:
        if controls._settings_readiness_waits.get(child) is entry:
            del controls._settings_readiness_waits[child]
        dependency = controls._settings_readiness_captures.get(child)
        if dependency is not None and not dependency.changed.done():
            dependency.changed.set_result(None)
        # Once a required settings phase entered, releasing this waiter's tail
        # early would let the next user capture observe its unfinished state.
        from .console_control_settings import PendingSettingsApply

        admitted = any(
            type(effect.owner) is PendingSettingsApply
            and not effect.owner.finished.done()
            for effect in source.effects
        )
        if type(source) is ParentSettingsCapture:
            operation = source.pending_operation
            admitted = admitted or (
                operation is not None and not operation.finished.done()
            )
        entered = dependency is not None and dependency.capture.started
        if admitted or entered:
            while not child.done():
                try:
                    await asyncio.shield(child)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break  # Original producer records its own failure/retirement.


def _require_definitions():
    from .console_domain_services import _definition_current

    if any(
        globals().get(name) is not record[0] or not _definition_current(record)
        for name, record in _PARENT_SETTINGS_DEFINITIONS
    ):
        raise RuntimeError("Parent settings defining source changed")


from .console_domain_services import _definition

_PARENT_SETTINGS_DEFINITIONS = tuple(
    (name, _definition(globals()[name]))
    for name in (
        "_require_initial_compose_helpers",
        "_require_initial_compose_phase",
        "_finish_initial_compose_tail",
        "finish_initial_console_compose",
        "consume_initial_console_compose",
        "prepare_initial_console_compose",
        "require_parent_settings_capture",
        "issue_parent_settings_capture",
        "finish_parent_settings_capture",
        "run_parent_settings_capture",
        "prepare_parent_default_settings",
        "_require_definitions",
        "defer_parent_settings_render",
        "parent_settings_render",
        "wait_parent_settings",
        "settings_inline_source",
        "settings_inline_parent",
        "await_readiness_settings",
        "settle_parent_action_defaults",
        "wait_readiness_predecessor",
        "_due_settings_read_current",
        "_validate_due_settings_read",
        "capture_parent_identity_origin",
        "_require_identity_body",
        "_require_identity_origin",
        "_retire_identity_request",
        "defer_parent_identity",
        "_run_parent_identity",
        "complete_due_settings_read",
        "_record_due_settings_convergence",
        "due_settings_phase",
        "due_settings_config",
        "consume_due_settings_result",
        "_readiness_dependency_current",
        "require_readiness_capture_current",
        "_parent_action",
        "defer_parent_settings_action",
        "initial_parent_settings_action",
        "_require_visible_send_helpers",
        "_require_visible_send_press",
        "retain_parent_visible_send",
        "prepare_parent_visible_send_target",
        "begin_parent_visible_target_check",
        "prepare_parent_visible_target_check",
        "_require_visible_check_callback",
        "_require_prefill_message",
        "_async_action_controls",
        "run_parent_async_settings_action",
        "_async_settings_action",
        "ensure_parent_async_action_settings",
        "mutate_parent_prefill",
        "finish_parent_prefill_core",
        "finish_parent_prefill_output",
        "_parent_post_settings_state",
        "_advance_parent_post_settings",
        "_capture_prompt_post_draft",
        "_finish_parent_prompt_posts",
        "_prepare_parent_prompt_component",
        "_apply_parent_system_prompt",
        "apply_parent_system_prompt",
        "_prepare_compact_action",
        "_require_action_helpers",
        "_require_settings_read_helpers",
        "_require_settings_read_target",
        "_require_action_codes",
        "_require_prompt_flow",
        "wait_prompt_flow_settings",
        "apply_parent_prompt_component",
        "capture_prompt_resolution_origin",
        "_require_prompt_resolution_origin",
        "_finish_parent_prompt_resolution",
        "complete_parent_prompt_resolution",
        "_parent_realtime_owner",
        "read_parent_realtime_instructions",
        "_require_realtime_callbacks",
        "realtime_settings_runtime",
        "capture_parent_attachment_origin",
        "_require_attachment_origin",
        "_require_attachment_helpers",
        "_finish_parent_attachment",
        "complete_parent_attachment",
    )
)


from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

_PREFILL_MESSAGE_DEFINITIONS = tuple(
    (
        name,
        original,
        _definition(original.fget if isinstance(original, property) else original),
    )
    for name in (
        "_append_native_console_system_message",
        "_ensure_console_chat_store",
        "_sync_native_console_chat_ui",
    )
    for original in (getattr_static(ConsoleMessageController, name),)
)
_PREFILL_CLEAR_DEFINITION = _definition(ConsoleComposerBar.clear_draft)

_VISIBLE_SEND_COMPOSER_DEFINITIONS = tuple(
    (name, _definition(getattr(ConsoleComposerBar, name)))
    for name in (
        "capture_draft_for_send",
        "raw_cli_draft_authority",
        "capture_draft_snapshot",
        "draft_text",
    )
)
