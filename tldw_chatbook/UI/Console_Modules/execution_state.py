"""Read the selected client's already-retired values on its actual screen visit.

Import lazily after ChatScreen is defined. No function constructs a runtime,
reads native storage, changes a session, or grants command authority.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable
from dataclasses import replace
from functools import partial

from ...Chat.console_control_dispatch import ConsoleProjection
from ...Chat.console_execution_process import ConsoleExecutionClient
from ...Chat.console_execution_projection_view import ConsoleProjectionView
from ...Chat.console_execution_composition_values import ConsoleSessionComposition
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
)
from ..Screens.chat_screen import ChatScreen, _workspace_build_scope
from .execution_projection import _definition, _require_methods, _require_sources

_CLIENT_METHODS = (("require_view", _definition(ConsoleExecutionClient.require_view)),)
_PASSIVE_FAILURE_METHODS = tuple(
    (name, _definition(getattr(ConsoleExecutionClient, name)))
    for name in ("read_projection", "_read_projection_locked", "require_passive_projection_failure")
)
_WARNING_METHOD = (
    (
        "_console_model_capability_warning",
        _definition(ChatScreen._console_model_capability_warning),
    ),
)


def selected_client(screen: ChatScreen) -> ConsoleExecutionClient | None:
    """Return the exact selected owner, refusing changed or unpublished values."""
    app = getattr(screen, "app_instance", None)
    client = getattr(app, "console_execution_client", None)
    if client is None:
        return None
    _require_sources()
    if type(screen) is not ChatScreen or type(client) is not ConsoleExecutionClient:
        raise TypeError("Original Console screen and client are required")
    _require_methods(client, _CLIENT_METHODS)
    from ...Chat.console_execution_readers import require_reader_sources

    require_reader_sources()
    client.require_view(screen, client._visit)
    _require_selected_publication(client, client._source_revision)
    return client


def _require_selected_publication(client, revision: int) -> None:
    """Check one internally consistent publication without rendering its values."""
    view, frame = client.projection_view, client._projection
    composition = client.session_composition
    if (
        type(composition) is not ConsoleSessionComposition
        or type(view) is not ConsoleProjectionView
        or type(frame) is not ConsoleProjection
        or view.source_revision != revision
        or frame.source_revision != view.source_revision
        or frame.session_id != view.active_session_id
        or not any(item is frame for item in view.projections)
        or view.session(frame.session_id).incarnation_id != frame.incarnation_id
        or (
            composition.session_id,
            composition.incarnation_id,
            composition.settings_revision,
        )
        != (
            frame.session_id,
            frame.incarnation_id,
            view.session(frame.session_id).settings_revision,
        )
    ):
        raise RuntimeError("Console selected projection is unavailable")


def _resumed_source_refresh(screen, client) -> bool:
    """Allow only a resumed visit to replace its older, coherent publication.

    The old row supplies identifiers to the ordinary checked ReadProjection;
    it is never accepted as current display or command authority.
    """
    view = getattr(client, "projection_view", None)
    if (type(view) is not ConsoleProjectionView
        or view.source_revision >= client._source_revision):
        return False
    _require_sources()
    if type(screen) is not ChatScreen or type(client) is not ConsoleExecutionClient:
        raise TypeError("Original Console process refresh owner is required")
    _require_methods(client, _CLIENT_METHODS)
    from ...Chat.console_execution_readers import require_reader_sources

    require_reader_sources()
    client.require_view(screen, client._visit)
    _require_selected_publication(client, view.source_revision)
    mounted = getattr(screen, "_console_process_mounted_visit", None)
    return (
        type(mounted) is int and mounted < client._visit
        and getattr(screen, "_console_process_visit", None) == client._visit
        and client._intake_open
    )


def selected_rail_publication(screen, client=None):
    """Observe only the original retained configuration action for this visit."""
    if client is None:
        client = getattr(screen.app_instance, "console_execution_client", None)
    if client is None or getattr(client, "_rail_publication", None) is None:
        return None
    from ...Chat import console_execution_rail_preferences as rail
    from ...Chat.console_domain_services import _definition_current

    if (rail.rail_publication_for is not rail._RAIL_LOOKUP[0]
        or not _definition_current(rail._RAIL_LOOKUP)):
        raise RuntimeError("Console rail publication observer changed")
    return rail.rail_publication_for(client, screen, client._visit)


def _require_applied_rail_publication(scope):
    from ...Chat import console_execution_rail_preferences as rail
    from ...Chat.console_domain_services import _definition_current

    if (rail.require_rail_publication is not rail._RAIL_REQUIRE[0]
        or not _definition_current(rail._RAIL_REQUIRE)):
        raise RuntimeError("Console rail publication verifier changed")
    rail.require_rail_publication(scope, applied=True)


def _rail_source_refresh(screen, client) -> bool:
    """The exact completed writer may replace its old same-visit publication."""
    import asyncio

    scope = selected_rail_publication(screen, client)
    if scope is None or scope.task is not asyncio.current_task():
        return False
    _require_applied_rail_publication(scope)
    client.require_view(screen, scope.visit)
    if client.projection_view.source_revision == client._source_revision:
        return False
    if (client.projection_view is not scope.projection_view
        or client._attachment_id != scope.attachment_id
        or client._source_revision != scope.revision
        or scope.revision != scope.old_revision + 1):
        raise RuntimeError("Console rail publication selection changed")
    _require_sources()
    _require_methods(client, _PREPARE_METHODS)
    _require_selected_publication(client, scope.old_revision)
    return True


async def finish_selected_rail_publication(scope) -> None:
    """Adopt original visible data only after the exact config action retired."""
    _require_applied_rail_publication(scope)
    screen, client = scope.view, scope.client
    if (screen is None or client._view is not screen or client._visit != scope.visit
        or not screen.is_current or getattr(screen, "_console_chat_tearing_down", False)):
        return  # Original resume owns a later visible read; never read while hidden.
    client.require_view(screen, scope.visit)
    if not await refresh_selected_view(screen, checkpoint="visible"):
        # Navigation owns its later visible mount. An interrupted widget render
        # may defer normally only if the checked new publication was adopted.
        if (client._view is not screen or client._visit != scope.visit
            or not screen.is_current or getattr(screen, "_console_chat_tearing_down", False)):
            return
        if selected_client(screen) is not client:
            raise RuntimeError("Console rail publication display did not complete")
        screen._console_process_surface_reconciled = False


def rail_publication_finished(scope) -> None:
    """Resume actual queued gestures after the action releases its sole record."""
    screen, client = scope.view, scope.client
    if (screen is None or getattr(client, "_rail_publication", None) is not None
        or client._view is not screen or client._visit != scope.visit
        or not screen.is_current or getattr(screen, "_console_chat_tearing_down", False)):
        return
    client.require_view(screen, scope.visit)
    if selected_client(screen) is not client:
        raise RuntimeError("Console rail publication completion changed")
    from .execution_rail import replay_selected_rail_toggles

    replay_selected_rail_toggles(screen, client, scope.visit)


_RAIL_FINISH = _definition(finish_selected_rail_publication)
_RAIL_FINISHED = _definition(rail_publication_finished)


class ConsoleDisplayExpired(RuntimeError):
    """The current visit needs a new checked backend display publication."""


def provider_model_display(
    screen: ChatScreen,
    client: ConsoleExecutionClient,
) -> tuple[str, str | None, ConsoleSessionSettings]:
    """Consume this exact projection's effective provider/model display facts."""
    if selected_client(screen) is not client:
        raise RuntimeError("Console provider presentation owner changed")
    if time.monotonic() >= client._readiness_deadline:
        raise ConsoleDisplayExpired("Console readiness presentation expired")
    return client.provider_model_display


def settings_readiness(
    screen: ChatScreen,
    client: ConsoleExecutionClient,
) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness]:
    """Apply only the original screen-local model warning to backend readiness."""
    if selected_client(screen) is not client:
        raise RuntimeError("Console readiness presentation owner changed")
    if time.monotonic() >= client._readiness_deadline:
        raise ConsoleDisplayExpired("Console readiness presentation expired")
    effective, readiness = client.settings_readiness
    _require_methods(screen, _WARNING_METHOD)
    warning = screen._console_model_capability_warning(
        effective.provider, effective.model
    )
    if warning and readiness.native_send_supported:
        readiness = replace(
            readiness,
            label="Capabilities unknown",
            detail=f"{readiness.detail}\n{warning}",
            native_send_supported=True,
        )
    return effective, readiness


from ..Navigation.pending_handoff_store import HandoffChannel, PendingHandoffStore

_HANDOFF_METHODS = (("has_pending", _definition(PendingHandoffStore.has_pending)),)


def empty_pending_launch(screen: ChatScreen, client: ConsoleExecutionClient) -> None:
    """Read proven empty staging without claiming an unimplemented transfer."""
    if selected_client(screen) is not client:
        raise RuntimeError("Console staged presentation owner changed")
    store = screen.app_instance.pending_handoffs
    if type(store) is not PendingHandoffStore:
        raise TypeError("Original Console handoff store is required")
    _require_methods(store, _HANDOFF_METHODS)
    if store.has_pending(HandoffChannel.CONSOLE_FIRST_CHAT):
        raise RuntimeError("Console first-chat handoff requires its process transfer")
    if store.has_pending(HandoffChannel.CONSOLE_LIVE_WORK):
        raise RuntimeError("Console staged handoff requires its process transfer")
    if client.session_composition.staged_evidence_present:
        raise RuntimeError("Console staged evidence requires its presentation")
    return None


_PREPARE_METHODS = tuple(
    (name, _definition(getattr(ConsoleExecutionClient, name)))
    for name in (
        "prepare_view", "attach_view", "detach_view", "require_view",
        "_require_view_preparation", "_retain_open_session", "_open_initial_session",
        "read_projection", "read_view_display", "_join_hidden_checkpoint", "_view_checkpoint_finished",
        "open_new_chat", "_read_projection_locked", "_projection_read_finished",
        "read_run_history",
    )
)


async def prepare_first_compose(screen: ChatScreen) -> None:
    """Join one selected view before any original synchronous builder executes."""
    _require_sources()
    client = screen.app_instance.console_execution_client
    if type(screen) is not ChatScreen or type(client) is not ConsoleExecutionClient:
        raise TypeError("Original Console process attachment is required")
    _require_methods(client, _PREPARE_METHODS)
    if screen._pending_character_conversation_target is not None:
        raise RuntimeError("Console character target requires its process opener")
    try:
        visit, _projection = await client.prepare_view(
            screen, conversation_id=screen._pending_resume_local_conversation_id
        )
        _require_sources()
        _require_methods(client, _PREPARE_METHODS)
        client.require_view(screen, visit)
        if selected_client(screen) is not client:
            raise RuntimeError("Console initial projection owner changed")
        screen._console_process_visit = visit
        screen._console_process_projection_view = client.projection_view
    except BaseException:
        if client._view is screen:
            client.detach_view(screen, client._visit)
        raise


from ...Chat.console_execution_context_summary_values import ConsoleContextSummaryProjection


def selected_context_summary(screen: ChatScreen) -> ConsoleContextSummaryProjection | None:
    """Return only this published session's original context display result."""
    client = selected_client(screen)
    if client is None:
        return None
    if time.monotonic() >= client._readiness_deadline:
        raise ConsoleDisplayExpired("Console context presentation expired")
    if type(client.context_summary) is not ConsoleContextSummaryProjection:
        raise RuntimeError("Console context presentation is unavailable")
    return client.context_summary


def selected_agent_rail(screen: ChatScreen):
    """Consume the original rail display captured for this published session."""
    from ...Chat.console_execution_agent_rail_values import ConsoleAgentRailProjection

    client = selected_client(screen)
    if client is None:
        return None
    row = client.projection_view.session(client.projection_view.active_session_id)
    conversation_id = str(row.persisted_conversation_id or "")
    if (
        screen._console_agent_drilldown_run_id
        and screen._console_agent_drilldown_conversation_id == conversation_id
    ):
        raise RuntimeError("Console Agent drill requires its process checkpoint")
    if type(client.agent_rail) is not ConsoleAgentRailProjection:
        raise RuntimeError("Console Agent rail presentation is unavailable")
    return client.agent_rail


def detach_displaced_view(app, screen: ChatScreen, target) -> None:
    """End a published visit when the routed target replaces it before suspend."""
    client = app.console_execution_client
    if (
        type(screen) is ChatScreen
        and type(client) is ConsoleExecutionClient
        and screen.app_instance is app and client._app is app
        and client._view is screen
        and getattr(screen, "_console_process_visit", None) == client._visit
        and type(getattr(screen, "_console_process_projection_view", None)) is ConsoleProjectionView
        and not getattr(screen, "_console_process_prepare_failed", False)
        and app.screen is target
        and all(item is not screen for item in app.screen_stack)
    ):
        detach_selected_view(screen)


def detach_selected_view(screen: ChatScreen) -> ConsoleProjectionView:
    """Revoke the actual visit before teardown awaits; retain read-only rows."""
    _require_sources()
    client = screen.app_instance.console_execution_client
    if type(screen) is not ChatScreen or type(client) is not ConsoleExecutionClient:
        raise TypeError("Original Console process attachment is required")
    if client._app is not screen.app_instance:
        raise RuntimeError("Console teardown App owner changed")
    _require_methods(client, _PREPARE_METHODS)
    view = getattr(screen, "_console_process_projection_view", None)
    visit = getattr(screen, "_console_process_visit", None)
    if type(view) is not ConsoleProjectionView:
        raise RuntimeError("Console teardown projection is unavailable")
    if client._view is screen:
        if client._visit != visit:
            raise RuntimeError("Console teardown visit changed")
        from .execution_decisions import is_owned_project_modal

        if is_owned_project_modal(client, screen, visit):
            return view
        client.detach_view(screen, visit)
    return view


def selected_workspace_surface(screen: ChatScreen):
    """Return the exact workspace/session display facts in this publication."""
    from ...Chat.console_execution_workspace_values import ConsoleWorkspaceSurface

    client = selected_client(screen)
    if client is None:
        return None
    surface = client.workspace_surface
    if (
        type(surface) is not ConsoleWorkspaceSurface
        or tuple(item.session_id for item in surface.sessions)
        != tuple(row.id for row in client.projection_view.session_rows)
    ):
        raise RuntimeError("Console workspace publication changed")
    return surface


def defer_selected_resize(screen: ChatScreen, event) -> bool:
    """Keep the latest resize until this visit has a current display snapshot."""
    if _awaiting_resume_publication(screen):
        client = screen.app_instance.console_execution_client
        screen._console_pending_process_resize = (client, client._visit, event)
        return True
    publication = selected_rail_publication(screen)
    if publication is not None:
        client = publication.client
        client.require_view(screen, publication.visit)
        screen._console_pending_process_resize = (client, publication.visit, event)
        return True
    client = selected_client(screen)
    if client is None:
        return False
    if time.monotonic() < client._readiness_deadline:
        screen._console_pending_process_resize = None
        return False
    screen._console_pending_process_resize = (client, client._visit, event)
    refresh_selected_when_due(screen)
    return True


def _defer_selected_refresh(screen, client, visit: int) -> bool:
    """Recognize presentation suspension without granting a covered view access."""
    from textual.screen import ModalScreen
    from .execution_decisions import is_owned_project_modal

    _require_sources()
    if type(screen) is not ChatScreen or type(client) is not ConsoleExecutionClient:
        raise TypeError("Original Console process refresh owner is required")
    _require_methods(client, _CLIENT_METHODS)
    app = screen.app_instance
    if (
        client._app is not app or app.console_execution_client is not client
        or getattr(app, "console_runtime", None) is not None
        or client._channel_error is not None
    ):
        return False
    if _awaiting_resume_publication(screen):
        return True
    scope = selected_rail_publication(screen, client)
    if scope is not None:
        import asyncio

        if scope.task is not asyncio.current_task():
            return True
        _require_applied_rail_publication(scope)
    deferred = (
        client._visit > visit or client._disposed or client._shutdown_task is not None
        or getattr(screen, "_console_chat_tearing_down", False)
        or getattr(screen, "_closing", False)
    )
    if not deferred and client._view is screen and client._visit == visit:
        stack = app.screen_stack
        index = next((index for index, item in enumerate(stack) if item is screen), None)
        deferred = bool(
            screen.is_attached and screen.is_mounted and screen.app is app
            and index is not None and index < len(stack) - 1
            and app.screen is stack[-1]
            and all(isinstance(item, ModalScreen) for item in stack[index + 1:])
            and not is_owned_project_modal(client, screen, visit)
        )
    if deferred:
        # Keep the last retired values; resume must fetch and reconcile again.
        screen._console_process_surface_reconciled = False
    return deferred


async def refresh_selected_view(screen: ChatScreen, *, checkpoint: str | None = None) -> bool:
    """Join a projection, deferring only an obsolete or genuinely covered visit."""
    from ...Chat.console_execution_process import ConsoleChannelLost, ConsoleDisplaySuperseded

    if getattr(screen, "_console_process_content_failed", False):
        return False
    client = screen.app_instance.console_execution_client
    if client is None:
        raise RuntimeError("Console process refresh requires its selected owner")
    visit = getattr(screen, "_console_process_visit", client._visit)
    if _defer_selected_refresh(screen, client, visit):
        return False
    if (not _resumed_source_refresh(screen, client)
        and not _rail_source_refresh(screen, client)
        and selected_client(screen) is not client):
        raise RuntimeError("Console process refresh requires its selected owner")
    source = client._source_revision
    screen._console_selected_refresh_count = getattr(screen, "_console_selected_refresh_count", 0) + 1
    refreshed = False
    try:
        try:
            refreshed = await _refresh_current_selected_view(screen, client, visit, checkpoint)
        except ConsoleDisplaySuperseded:
            if client._source_revision != source:
                raise
            client.require_view(screen, visit)
            screen._console_process_surface_reconciled = False
            return False
        except ConsoleChannelLost:
            if client._source_revision == source and _defer_selected_refresh(screen, client, visit):
                return False
            raise
        if refreshed:
            screen._console_passive_projection_failure = None
        return refreshed
    finally:
        screen._console_selected_refresh_count -= 1
        if refreshed and not screen._console_selected_refresh_count:
            _require_methods(screen, _DECISION_WAKE_METHODS)
            # FULL never acknowledges a decision hint or overwrites its head.
            # Keep the original periodic reconciliation as the missed-hint hedge.
            screen.call_later(partial(start_selected_decision_refresh, screen, force=True))


async def _refresh_current_selected_view(screen, client, visit, checkpoint) -> bool:
    from .execution_projection import apply_console_process_projection
    from .execution_sessions import apply_console_process_sessions
    from .execution_decisions import is_owned_project_modal

    if is_owned_project_modal(client, screen, visit):
        return False  # Only the finite decision head is eligible under this modal.

    row = client.projection_view.session(client.projection_view.active_session_id)
    prepared = None
    if checkpoint == "mount":
        prepared = getattr(screen, "_console_resumed_publication", None)
        screen._console_resumed_publication = None
        if prepared is not None and (prepared[0] is not client or prepared[1] != visit):
            prepared = None
    if (prepared is not None and prepared[2] is client.projection_view
        and prepared[3] is client._projection and time.monotonic() < client._readiness_deadline):
        projection = client._projection
    else:
        if prepared is not None:
            checkpoint = None  # This visit's original mount already applied and retired.
        projection = await client.read_view_display(
            screen, visit, row.id, row.incarnation_id, checkpoint=checkpoint,
        )
    if _defer_selected_refresh(screen, client, visit):
        return False
    client.require_view(screen, visit)
    if getattr(screen, "_console_process_content_failed", False):
        return False
    if selected_client(screen) is not client:
        raise RuntimeError("Console refreshed publication changed")
    screen._console_process_projection_view = client.projection_view
    if is_owned_project_modal(client, screen, visit):
        return False
    from .execution_rail import replay_selected_rail_toggles

    replay_selected_rail_toggles(screen, client, visit)
    pending_resize = getattr(screen, "_console_pending_process_resize", None)
    if pending_resize is not None:
        screen._console_pending_process_resize = None
        if pending_resize[0] is client and pending_resize[1] == visit and screen.is_current:
            _require_methods(screen, _RESIZE_METHOD)
            screen._adapt_console_workspace_to_width(pending_resize[2])
    if not await apply_console_process_sessions(
        screen, client, client.session_surface, visit=visit,
    ):
        return False
    if _defer_selected_refresh(screen, client, visit):
        return False
    client.require_view(screen, visit)
    if not await apply_console_process_projection(screen, client, projection, visit=visit):
        return False
    if not sync_selected_controls(screen):
        return False
    if _defer_selected_refresh(screen, client, visit):
        return False
    client.require_view(screen, visit)
    return True


def sync_selected_controls(screen: ChatScreen, *, draft_only: bool = False) -> bool:
    """Render the attached visit's controls from its current checked display."""
    if (
        not screen.is_attached or not screen.is_current
        or getattr(screen, "_console_sync_maintenance_paused", False)
        or getattr(screen, "_console_process_content_failed", False)
    ):
        return False
    client = selected_client(screen)
    if client is None:
        raise RuntimeError("Console control presentation owner changed")
    if time.monotonic() < client._readiness_deadline:
        _require_methods(screen, _REFRESH_METHODS)
        try:
            with screen._console_derivation_scope(), _workspace_build_scope(screen):
                control = screen._build_console_control_state(screen._pending_console_launch_context)
                workbench = screen._build_console_workbench_state(control)
                if time.monotonic() >= client._readiness_deadline:
                    raise ConsoleDisplayExpired("Console control presentation expired")
                screen._push_console_control_state_if_changed(control, workbench)
                if not draft_only:
                    screen._sync_console_settings_summary()
                    screen._sync_console_selected_recovery()
                    readiness = screen._active_console_settings_readiness()
                    screen._apply_console_transcript_guidance(readiness)
                    screen._sync_console_pending_delete_confirmation()
                    screen._sync_console_inspector_and_rail()
                    screen._sync_console_cost_chip()
            if time.monotonic() >= client._readiness_deadline:
                raise ConsoleDisplayExpired("Console control presentation expired")
            return True
        except ConsoleDisplayExpired:
            pass
    refresh_selected_when_due(screen)
    return False


def sync_selected_draft_spend(screen: ChatScreen) -> bool:
    """Render the original spend pair; coalesce shared refresh only on expiry."""
    if getattr(screen.app_instance, "console_execution_client", None) is None:
        return False
    if defer_selected_passive_render(screen):
        return True
    client = selected_client(screen)
    if client is None:
        raise RuntimeError("Console spend presentation owner changed")
    visit = client._visit
    if time.monotonic() < client._readiness_deadline:
        _require_methods(screen, _SPEND_REFRESH_METHODS)
        try:
            with screen._console_derivation_scope():
                screen._sync_console_settings_summary()
                if time.monotonic() >= client._readiness_deadline:
                    raise ConsoleDisplayExpired("Console spend presentation expired")
                screen._sync_console_cost_chip()
            if selected_client(screen) is not client or client._visit != visit:
                raise RuntimeError("Console spend presentation owner changed")
            if time.monotonic() >= client._readiness_deadline:
                raise ConsoleDisplayExpired("Console spend presentation expired")
            return True
        except ConsoleDisplayExpired:
            pass
    refresh_selected_when_due(screen)
    return True


_SPEND_REFRESH_METHODS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in (
        "_console_derivation_scope", "_sync_console_settings_summary", "_sync_console_cost_chip",
    )
)


def sync_selected_guidance(screen: ChatScreen) -> None:
    """Refresh the inline guidance and blocking modal together from this visit."""
    if (
        not screen.is_attached or not screen.is_current
        or getattr(screen, "_console_sync_maintenance_paused", False)
    ):
        return
    client = selected_client(screen)
    if client is None:
        raise RuntimeError("Console guidance presentation owner changed")
    _require_methods(screen, _GUIDANCE_METHODS)
    try:
        screen._apply_console_transcript_guidance(settings_readiness(screen, client))
    except ConsoleDisplayExpired:
        refresh_selected_when_due(screen)


_RESIZE_METHOD = (("_adapt_console_workspace_to_width", _definition(ChatScreen._adapt_console_workspace_to_width)),)

_GUIDANCE_METHODS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in ("_apply_console_transcript_guidance", "_active_console_settings_readiness")
)

_REFRESH_METHODS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in (
        "_console_derivation_scope", "_build_console_control_state",
        "_build_console_workbench_state", "_push_console_control_state_if_changed",
        "_sync_console_workbench_state", "_apply_console_workbench_state",
        "_sync_console_pending_delete_confirmation", "_sync_console_inspector_and_rail",
        "_sync_console_cost_chip",
        "_sync_console_settings_summary", "_sync_console_rail_system_line",
        "_sync_console_agent_section",
        "_sync_console_selected_recovery", "_console_settings_failure_display",
        "_console_selected_continuation_recovery", "_sync_console_settings_recovery_surfaces",
    )
) + _GUIDANCE_METHODS


async def wait_selected_navigation_resume(screen: ChatScreen) -> None:
    """Join the queued ScreenResume before navigation invokes Chat readers.

    Textual's switch awaitable joins mount/removal, not ScreenResume. Its
    actual screen pump owns the visible publication; this barrier does not
    issue another read or block the App's input pump.
    """
    import asyncio

    _require_sources()
    app = screen.app_instance
    client = app.console_execution_client
    if type(screen) is not ChatScreen or type(client) is not ConsoleExecutionClient:
        raise TypeError("Original Console navigation owner is required")
    pump = screen._task
    if pump is None or pump is asyncio.current_task():
        raise RuntimeError("Console navigation must join its independent screen pump")
    ready = asyncio.get_running_loop().create_future()

    def completed() -> None:
        if ready.done():
            return
        try:
            if app.console_execution_client is not client or selected_client(screen) is not client:
                raise RuntimeError("Console navigation publication owner changed")
            if _awaiting_resume_publication(screen):
                raise RuntimeError("Console resume publication is still pending")
        except Exception as error:
            ready.set_exception(error)
        else:
            ready.set_result(None)

    if not screen.call_later(completed):
        raise RuntimeError("Console resume message pump is unavailable")
    try:
        await asyncio.wait((ready, pump), return_when=asyncio.FIRST_COMPLETED)
        if not ready.done():
            raise RuntimeError("Console resume message pump ended before publication")
        ready.result()
        if app.console_execution_client is not client or selected_client(screen) is not client:
            raise RuntimeError("Console navigation publication owner changed")
    finally:
        if not ready.done():
            ready.cancel()


def resume_selected_view(screen: ChatScreen) -> Awaitable[None] | None:
    """Publish the resumed source before synchronous screen readers run."""
    _require_sources()
    client = screen.app_instance.console_execution_client
    if type(client) is not ConsoleExecutionClient or type(screen) is not ChatScreen:
        raise TypeError("Original Console process view is required")
    _require_methods(client, _PREPARE_METHODS)
    visit = client.attach_view(screen)
    if getattr(screen, "_console_process_visit", None) != visit:
        screen._console_process_visit = visit
        screen._console_process_surface_reconciled = False
    mounted = getattr(screen, "_console_process_mounted_visit", None)
    operation = getattr(client, "_workspace_binding_task", None)
    if (type(mounted) is int and mounted < visit
        and (operation is not None or (
            client.projection_view is not None
            and client.projection_view.source_revision < client._source_revision
        ))):
        return _publish_resumed_source(screen, client, visit, operation)
    if not getattr(screen, "_console_process_surface_reconciled", False):
        screen.call_after_refresh(screen._start_console_attach_reconciliation)


def _resumed_view_displaced(screen, client, visit: int) -> bool:
    """Recognize an actual stack transfer without granting the old view access."""
    from textual.screen import ModalScreen, Screen

    _require_sources()
    if type(screen) is not ChatScreen or type(client) is not ConsoleExecutionClient:
        return False
    _require_methods(client, _CLIENT_METHODS)
    app = screen.app_instance
    if (client._app is not app or app.console_execution_client is not client
        or screen.app is not app
        or getattr(app, "console_runtime", None) is not None
        or client._channel_error is not None or client._disposed
        or client._visit < visit
        or (client._visit == visit and client._view is not screen)):
        return False
    stack = app.screen_stack
    if not stack or app.screen is not stack[-1]:
        return False
    if app.screen is screen:
        return client._view is screen and client._visit > visit
    destination = stack[-1]
    if (not isinstance(destination, Screen) or destination.app is not app
        or not destination.is_attached or not destination.is_mounted):
        return False
    index = next((i for i, item in enumerate(stack) if item is screen), None)
    if index is not None:
        return all(isinstance(item, ModalScreen) for item in stack[index + 1:])
    return client._view is screen or client._visit > visit


def _awaiting_resume_publication(screen) -> bool:
    client = getattr(screen.app_instance, "console_execution_client", None)
    pending = getattr(screen, "_console_resuming_source", None)
    if pending is None or client is None or pending[0] is not client or pending[1] != client._visit:
        return False
    if _resumed_view_displaced(screen, client, pending[1]):
        return True
    client.require_view(screen, client._visit)
    return True


async def _publish_resumed_source(screen, client, visit, operation):
    import asyncio

    # Textual awaits this ScreenResume handler. Only the original visible
    # backend read belongs here. Widget mounting continues in its normal worker.
    pending = client, visit
    screen._console_resuming_source = pending
    try:
        if _resumed_view_displaced(screen, client, visit):
            return
        client.require_view(screen, visit)
        if operation is not None:
            if (operation is not client._workspace_binding_task
                or operation not in client._workspace_operations):
                raise RuntimeError("Console workspace publication owner changed")
            await asyncio.shield(operation)
            if _resumed_view_displaced(screen, client, visit):
                return
            client.require_view(screen, visit)
        if not _resumed_source_refresh(screen, client):
            selected_client(screen)
        else:
            row = client.projection_view.session(client.projection_view.active_session_id)
            source, attachment = client._source_revision, client._attachment_id
            from ...Chat.console_execution_process import ConsoleChannelLost

            try:
                frame = await client.read_view_display(
                    screen, visit, row.id, row.incarnation_id, checkpoint="mount",
                )
            except ConsoleChannelLost:
                # The retained transport still owns physical settlement. Only
                # an unchanged healthy source displaced by actual navigation
                # may end this screen observer without a fatal app exception.
                if (client._source_revision == source
                    and client._attachment_id == attachment
                    and _resumed_view_displaced(screen, client, visit)):
                    return
                raise
            if client._source_revision != source or client._attachment_id != attachment:
                raise ConsoleChannelLost("Console resumed publication source changed")
            if _resumed_view_displaced(screen, client, visit):
                return
            client.require_view(screen, visit)
            if selected_client(screen) is not client or frame is not client._projection:
                raise RuntimeError("Console resumed publication changed")
            screen._console_resumed_publication = client, visit, client.projection_view, frame
        screen.call_after_refresh(screen._start_console_attach_reconciliation)
    finally:
        if screen._console_resuming_source is pending:
            screen._console_resuming_source = None


async def reconcile_selected_view(screen: ChatScreen, *, passive: bool = False) -> None:
    """Join the real remote visit and publish its original mounted values."""
    from ...Chat.console_execution_process import ConsoleChannelLost, ConsoleRemoteFailure

    if getattr(screen, "_console_process_content_failed", False):
        return
    client = screen.app_instance.console_execution_client
    if (client is None or screen._console_attach_reconcile_running
        or getattr(screen, "_console_process_draft_switch", None) is not None):
        return
    visit = getattr(screen, "_console_process_visit", client._visit)
    if _defer_selected_refresh(screen, client, visit):
        return
    if not _resumed_source_refresh(screen, client) and selected_client(screen) is not client:
        raise RuntimeError("Console reconciliation owner changed")
    from .execution_decisions import pending_project_visibility

    if pending_project_visibility(client, screen, visit) is not None:
        return  # The actual dismissal owner must restore visibility first.
    source = client._source_revision
    screen._console_attach_reconcile_running = True
    try:
        action = "mount" if getattr(screen, "_console_process_mounted_visit", None) != visit else None
        if not await refresh_selected_view(screen, checkpoint=action):
            # The first mount has no periodic refresh yet. A snapshot may
            # expire while Textual settles the newly mounted session widgets.
            if (client._view is screen and client._visit == visit
                and screen.is_current and screen._console_credential_poll_timer is None
                and time.monotonic() >= client._readiness_deadline):
                screen.set_timer(0.25, partial(refresh_selected_when_due, screen))
            return
        client.require_view(screen, visit)
        screen._console_process_mounted_visit = visit
        screen._console_process_surface_reconciled = True
        after_mount = getattr(screen, "_console_pending_manual_return", None)
        if after_mount is not None:
            screen._console_pending_manual_return = None
            screen.run_worker(after_mount, group="console-manual-read-return", exclusive=True)
        _require_methods(screen, _MOUNT_UI_METHODS)
        if not getattr(screen, "_console_process_view_started", False):
            # Existing UI restoration does not own backend execution. The
            # selected command/discovery adapters remain separate consumers.
            screen.set_timer(0.1, screen._restore_collapsible_states)
            screen.set_timer(0.05, screen.sync_task_resume_state)
            screen.call_after_refresh(screen._restore_console_workbench_focus)
            screen.set_timer(0.2, screen._restore_console_workbench_focus)
            screen.call_after_refresh(screen._sync_console_dictation_availability)
            screen.set_timer(0.15, screen._sync_console_dictation_availability)
            screen._console_process_view_started = True
        if screen._console_credential_poll_timer is None:
            screen._console_credential_poll_timer = screen.set_interval(
                0.25, lambda: refresh_selected_when_due(screen),
            )
            screen._record_ui_timer_created("console-credential-poll")
    except ConsoleRemoteFailure as error:
        if not passive or action is not None or client._source_revision != source:
            raise
        _require_sources()
        _require_methods(client, _PASSIVE_FAILURE_METHODS)
        if selected_client(screen) is not client:
            raise
        client.require_passive_projection_failure(error, screen, visit)
        row = client.projection_view.session(client.projection_view.active_session_id)
        scope = (client, visit, source, row.id, row.incarnation_id)
        previous = getattr(screen, "_console_passive_projection_failure", None)
        # A traceback retains the failed read's entire projection and receipt.
        # The UI needs only the scope and the safe, fixed diagnostic values.
        screen._console_passive_projection_failure = (*scope, error.code, error.stage)
        screen._console_process_surface_reconciled = False
        client._readiness_deadline = 0.0
        if previous is None or previous[:5] != scope:
            from loguru import logger

            logger.warning(
                "Console passive display refresh refused (code={}, stage={}); will retry",
                error.code, error.stage,
            )
    except ConsoleChannelLost:
        if client._source_revision != source or not _defer_selected_refresh(screen, client, visit):
            raise
        # Superseded projection was not adopted; its command retains its
        # original physical owners independently of this removed screen.
    finally:
        screen._console_attach_reconcile_running = False
        if (
            client._view is screen and client._visit > visit and screen.is_current
            and client._channel_error is None and client._shutdown_task is None
            and not client._disposed and not getattr(screen, "_console_process_content_failed", False)
        ):
            # Resume may have run while this older worker still owned the flag.
            screen.call_after_refresh(screen._start_console_attach_reconciliation)


def _decision_hint_scope(client) -> tuple | None:
    """Return only an actual mounted selected visit's published scope."""
    screen, app, view = client._view, client._app, client.projection_view
    if (
        type(screen) is not ChatScreen or view is None
        or view.source_revision != client._source_revision
        or view.active_session_id is None
        or screen.app_instance is not app
        or getattr(app, "console_execution_client", None) is not client
        or getattr(app, "console_runtime", None) is not None
        or not screen.is_attached or not screen.is_mounted
        or getattr(screen, "_console_process_visit", None) != client._visit
        or getattr(screen, "_console_chat_tearing_down", False)
        or getattr(screen, "_closing", False)
        or client._disposed or client._shutdown_task is not None
        or client._channel_error is not None
        or getattr(client, "_view_preparation", None) is not None
    ):
        return None
    row = view.session(view.active_session_id)
    return (
        client._source_revision, row.id, row.incarnation_id,
        client._attachment_id, client._visit,
    )


def _decision_refresh_due(client) -> bool:
    hint = getattr(client, "_decision_hint", None)
    return bool(
        hint is not None and hint[:-1] == _decision_hint_scope(client)
        and hint != getattr(client, "_decision_rendered_hint", None)
    )


def _decision_refresh_rendered(client, hint) -> None:
    """Consume only the hint whose checked projection was actually rendered."""
    if hint is None or hint[:-1] != _decision_hint_scope(client):
        return
    previous = getattr(client, "_decision_rendered_hint", None)
    if previous is None or previous[:-1] != hint[:-1] or previous[-1] < hint[-1]:
        client._decision_rendered_hint = hint


_DECISION_WAKE_METHODS = (("call_later", _definition(ChatScreen.call_later)),)
_DECISION_READ_METHODS = (
    ("read_decision_head", _definition(ConsoleExecutionClient.read_decision_head)),
)


def _retired_decision_switched_away(screen, client, visit, captured) -> bool:
    """Defer a completed passive read while ordinary navigation suspends Chat."""
    from textual.screen import ModalScreen
    from ...Constants import TAB_CHAT
    from ..Navigation.base_app_screen import BaseAppScreen

    _require_sources()
    _require_methods(client, _CLIENT_METHODS + _DECISION_READ_METHODS)
    source, attachment, session_id, incarnation_id, reusable = captured
    app = screen.app_instance
    if (
        type(screen) is not ChatScreen or type(client) is not ConsoleExecutionClient
        or type(reusable) is not tuple or len(reusable) != 2 or reusable[1] is not screen
        or getattr(app, "_reusable_screen_instances", {}).get(TAB_CHAT) is not reusable
        or client._app is not app or app.console_execution_client is not client
        or getattr(app, "console_runtime", None) is not None
        or client._view is not screen or client._visit != visit
        or client._source_revision != source or client._attachment_id != attachment
        or client._view_preparation is not None or client._project_modal is not None
        or client._disposed or client._shutdown_task is not None or client._channel_error is not None
        or not screen.is_attached or not screen.is_mounted or screen.app is not app
        or getattr(screen, "_console_chat_tearing_down", False) or getattr(screen, "_closing", False)
    ):
        return False
    current, stack = app.screen, app.screen_stack
    if (
        screen.is_current or screen in stack or not stack or current is not stack[-1]
        or not isinstance(current, BaseAppScreen) or isinstance(current, (ChatScreen, ModalScreen))
        or not current.is_attached or not current.is_mounted or current.app is not app
    ):
        return False
    _require_selected_publication(client, source)
    published = client.projection_view
    if (published.active_session_id != session_id
        or published.session(session_id).incarnation_id != incarnation_id):
        return False
    screen._console_process_surface_reconciled = False
    return True


def request_selected_decision_refresh(client, hint: tuple) -> bool:
    """Wake the existing checked display; the stream hint grants no authority."""
    _require_sources()
    if type(client) is not ConsoleExecutionClient:
        raise TypeError("Original Console decision display client is required")
    if hint[:-1] != _decision_hint_scope(client):
        return False
    previous = getattr(client, "_decision_hint", None)
    if previous is not None and previous[:-1] == hint[:-1] and previous[-1] >= hint[-1]:
        return False
    client._decision_hint = hint
    _require_methods(client._view, _DECISION_WAKE_METHODS)
    # Enter the real screen pump; queued callbacks coalesce at its existing
    # reconcile worker and recheck a visit which may have ended meanwhile.
    client._view.call_later(refresh_selected_when_due, client._view)
    return True


def refresh_selected_when_due(screen: ChatScreen) -> None:
    """Request the original readiness refresh after its published expiry."""
    if defer_selected_passive_render(screen, for_refresh=True):
        return
    # A genuine decision change need not wait for an unrelated FULL read. It
    # still uses the same source-qualified channel and original UI owners.
    from .execution_decisions import is_owned_project_modal, pending_project_visibility

    owner = screen.app_instance.console_execution_client
    resumed = owner is not None and _resumed_source_refresh(screen, owner)
    modal_visible = owner is not None and is_owned_project_modal(owner, screen, owner._visit)
    if not resumed:
        start_selected_decision_refresh(screen, force=modal_visible)
    if modal_visible:
        return  # The decision refresh above remains eligible; FULL is hidden.
    if owner is not None and pending_project_visibility(owner, screen, owner._visit) is not None:
        return  # Its original explicit visible FULL owns this transition.
    if (
        not screen.is_attached or (not screen.is_current and not modal_visible)
        or getattr(screen, "_console_sync_maintenance_paused", False)
        or screen._console_attach_reconcile_running
        or getattr(screen, "_console_selected_refresh_count", 0)
        or (getattr(screen, "_console_attach_reconcile_worker", None) is not None
            and not screen._console_attach_reconcile_worker.is_finished)
        or getattr(screen, "_console_process_draft_switch", None) is not None
    ):
        return
    client = owner if resumed else selected_client(screen)
    if client is None or (not resumed and time.monotonic() < client._readiness_deadline):
        return
    screen._console_attach_reconcile_worker = screen.run_worker(
        partial(reconcile_selected_view, screen, passive=True),
        group="console-attach-reconciliation", exclusive=False,
    )


def start_selected_decision_refresh(screen: ChatScreen, *, force: bool = False) -> None:
    """Coalesce the existing hint into one finite read, without a second queue."""
    if defer_selected_passive_render(screen):
        return
    client = selected_client(screen)
    if (client is None or getattr(screen, "_console_process_mounted_visit", None) != client._visit
        or getattr(screen, "_console_process_draft_switch", None) is not None
        or getattr(screen, "_console_sync_maintenance_paused", False)):
        return
    from .execution_decisions import pending_project_visibility

    if pending_project_visibility(client, screen, client._visit) is not None:
        return  # Keep the hint owed until the actual visible checkpoint settles.
    if not force and not _decision_refresh_due(client):
        return
    worker = getattr(screen, "_console_decision_refresh_worker", None)
    if worker is not None and not worker.is_finished:
        return
    screen._console_decision_refresh_worker = screen.run_worker(
        partial(_refresh_selected_decisions, screen, client, client._visit),
        group="console-process-decision-read", exclusive=False,
    )


async def _refresh_selected_decisions(screen, client, visit: int) -> None:
    """Paint only a positively retired head; leave newer invalidations owed."""
    from ...Chat.console_execution_process import ConsoleChannelLost
    from ...Constants import TAB_CHAT
    from .execution_decisions import pending_project_visibility, present_selected_decision

    source = client._source_revision
    hint = getattr(client, "_decision_hint", None)
    rendered = False
    try:
        if _defer_selected_refresh(screen, client, visit):
            return
        if selected_client(screen) is not client:
            raise ConsoleChannelLost("Console decision owner changed")
        row = client.projection_view.session(client.projection_view.active_session_id)
        captured = (
            source, client._attachment_id, row.id, row.incarnation_id,
            getattr(screen.app_instance, "_reusable_screen_instances", {}).get(TAB_CHAT),
        )
        _require_methods(client, _DECISION_READ_METHODS)
        await client.read_decision_head(screen, visit, row.id, row.incarnation_id)
        # The original method joins positive physical retirement before return.
        # Textual may switch its stack before queued ScreenSuspend revokes visit.
        if not screen.is_current and _retired_decision_switched_away(screen, client, visit, captured):
            return
        if pending_project_visibility(client, screen, visit) is not None:
            return
        if _defer_selected_refresh(screen, client, visit):
            return
        client.require_view(screen, visit)
        if await present_selected_decision(screen, client, visit) is None:
            return  # An already registered rail writer precedes a new modal.
        if pending_project_visibility(client, screen, visit) is not None:
            return
        if _defer_selected_refresh(screen, client, visit):
            return
        client.require_view(screen, visit)
        _decision_refresh_rendered(client, hint)
        rendered = True
    except ConsoleChannelLost:
        if client._source_revision != source or not _defer_selected_refresh(screen, client, visit):
            raise
    finally:
        if rendered and _decision_refresh_due(client):
            _require_methods(screen, _DECISION_WAKE_METHODS)
            screen.call_later(refresh_selected_when_due, screen)


def defer_selected_passive_render(screen: ChatScreen, *, for_refresh: bool = False) -> bool:
    """Skip obsolete timer work without substituting display values or authority."""
    app = getattr(screen, "app_instance", None)
    client = getattr(app, "console_execution_client", None)
    if client is None:
        return False
    if (
        not screen.is_attached or not screen.is_mounted
        or getattr(screen, "_console_chat_tearing_down", False)
        or getattr(screen, "_closing", False)
        or getattr(screen, "_console_process_content_failed", False)
    ):
        return True
    visit = getattr(screen, "_console_process_visit", None)
    if visit is None:
        return True  # No visit has published yet; this timer cannot render it.
    if (type(client) is ConsoleExecutionClient and client._app is app
        and client._visit > visit):
        screen._console_process_surface_reconciled = False
        return True
    if _defer_selected_refresh(screen, client, visit):
        return True
    from .execution_decisions import is_owned_project_modal

    # A passive timer has nothing to paint on a covered/replaced screen. The
    # original owned disclosure remains eligible for its checked refresh path.
    # Eligible callers still use selected_client and propagate real failures.
    if not screen.is_current and not is_owned_project_modal(client, screen, visit):
        return True
    # Only a refresh may replace the older publication; ordinary consumers
    # must wait rather than paint it. All visit/teardown guards above still apply.
    return not for_refresh and _resumed_source_refresh(screen, client)


_MOUNT_UI_METHODS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in (
        "_restore_collapsible_states", "sync_task_resume_state",
        "_restore_console_workbench_focus", "_sync_console_dictation_availability",
        "_record_ui_timer_created",
    )
)


_AUTHORED_INPUT_METHODS = tuple(
    (name, _definition(getattr(ConsoleExecutionClient, name)))
    for name in (
        "publish_authored_inputs", "_join_authored_inputs", "_authored_inputs_finished",
        "retain_authored_press", "release_authored_press", "submit_authored_turn",
        "_submit_authored_turn", "submit",
    )
)


def require_authored_owner(screen: ChatScreen, client: ConsoleExecutionClient, visit: int) -> None:
    """Recheck exact input methods and the actual visit after an observer wait."""
    if selected_client(screen) is not client or client is None or client._visit != visit:
        raise RuntimeError("Console process input owner is unavailable")
    _require_methods(client, _AUTHORED_INPUT_METHODS)


def publish_authored_draft(screen: ChatScreen, draft: str, token: tuple[int, int]) -> bool:
    """Send the actual widget edit to the selected original input command."""
    client = selected_client(screen)
    require_authored_owner(screen, client, client._visit)
    return client.publish_authored_inputs(screen, client._visit, draft, token)


def begin_selected_session_switch(screen: ChatScreen) -> None:
    """Retain the original switch snapshot before the new-session await."""
    client = selected_client(screen)
    require_authored_owner(screen, client, client._visit)
    if getattr(screen, "_console_process_draft_switch", None) is not None:
        raise RuntimeError("Console draft switch is already pending")
    if client._input_press is not None:
        raise RuntimeError("Console pressed input still owns its draft")
    owner = screen._session
    composer = screen._console_composer_or_none()
    if composer is None:
        raise RuntimeError("Console switch composer is unavailable")
    snapshot = owner._console_draft_switch_snapshot
    if snapshot is None or snapshot[0] != owner._console_visible_draft_session_id:
        raise RuntimeError("Console original switch snapshot is unavailable")
    screen._console_process_draft_switch = (
        client, client._visit, snapshot, composer, client._source_revision,
        composer.capture_draft_snapshot(),
    )


async def _join_selected_switch_entries(client, entries) -> None:
    import asyncio

    for receipt in entries:
        await asyncio.shield(receipt.applied)
        if await asyncio.shield(receipt.retired) is not True:
            raise RuntimeError("Console switched draft did not retire")


async def cancel_selected_session_switch(screen: ChatScreen) -> None:
    """Preserve actual typing if the original session command was refused."""
    transition = getattr(screen, "_console_process_draft_switch", None)
    if transition is None:
        return
    client, visit, snapshot, composer, revision, _original_draft = transition
    try:
        require_authored_owner(screen, client, visit)
        if (client._source_revision != revision
            or client.projection_view.active_session_id != snapshot[0]):
            raise RuntimeError("Console cancelled switch destination changed")
        captured = composer.capture_draft_snapshot()
        client.publish_authored_inputs(
            screen, visit, composer.draft_text(),
            (captured.generation, captured.edit_serial), _switch=transition,
        )
        row = client.projection_view.session(snapshot[0])
        receipt = client._authored_inputs[(row.id, row.incarnation_id)][2]
    finally:
        if getattr(screen, "_console_process_draft_switch", None) is transition:
            screen._console_process_draft_switch = None
    await _join_selected_switch_entries(client, (receipt,))


async def bind_selected_session_draft(screen: ChatScreen) -> None:
    """Apply the original draft/undo/suffix transition to retired row values."""
    client = selected_client(screen)
    owner = screen._session
    transition = getattr(screen, "_console_process_draft_switch", None)
    if transition is None:
        return
    retained, visit, snapshot, composer, revision, original_draft = transition
    require_authored_owner(screen, client, visit)
    if retained is not client or client._source_revision != revision:
        raise RuntimeError("Console draft switch source changed")
    row = client.projection_view.session(client.projection_view.active_session_id)
    visible = owner._console_visible_draft_session_id
    if snapshot[0] != visible or composer is not screen._console_composer_or_none():
        raise RuntimeError("Console draft switch composer changed")
    if row.id == visible:
        await cancel_selected_session_switch(screen)
        return
    live_text = composer.draft_text()
    save_text, suffix, resident_changed = live_text, "", False
    snap_text, snap_serial = snapshot[1:]
    if composer.edit_serial != snap_serial and live_text.startswith(snap_text):
        save_text, suffix = snap_text, live_text[len(snap_text):]
    try:
        old = client.projection_view.session(visible)
    except KeyError:
        old = None
    if (old is not None and live_text.startswith(snap_text)
        and owner._console_visible_draft_revision is not None
        and old.draft_revision != owner._console_visible_draft_revision):
        save_text, resident_changed = old.draft, True
    receipts = []
    captured = composer.capture_draft_snapshot()
    if old is not None:
        if save_text != old.draft:
            token = ((original_draft.generation, snap_serial) if suffix
                     else (captured.generation, captured.edit_serial))
            client.publish_authored_inputs(
                screen, visit, save_text, token,
                _switch=transition, _session_id=old.id,
            )
            receipts.append(client._authored_inputs[(old.id, old.incarnation_id)][2])
        if resident_changed:
            owner._console_undo_histories.pop(old.id, None)
        else:
            owner._console_undo_histories[old.id] = composer.export_undo_history()
    observer = getattr(composer, "_authored_draft_observer", None)
    composer._authored_draft_observer = None
    try:
        composer.load_draft(row.draft)
        composer.restore_undo_history(owner._console_undo_histories.get(row.id))
        owner._restore_banked_raw_cli_stashes_fn(row.id, composer)
        if suffix:
            composer.insert_text(suffix)
        owner._sync_console_command_popup()
        if owner._on_draft_session_changed is not None:
            owner._on_draft_session_changed()
        owner._console_visible_draft_session_id = row.id
        owner._console_visible_draft_revision = row.draft_revision
        owner._console_draft_switch_snapshot = None
        owner._initial_composer_sync_pending = False
        composition = client.session_composition
        owner._visible_agent_handoff_draft = (
            (row.id, row.incarnation_id, composition.agent_handoff_revision,
             composer.capture_draft_snapshot())
            if composition.agent_handoff_state == "pending" else None
        )
        if suffix:
            captured = composer.capture_draft_snapshot()
            client.publish_authored_inputs(
                screen, visit, composer.draft_text(),
                (captured.generation, captured.edit_serial), _switch=transition,
            )
            receipts.append(client._authored_inputs[(row.id, row.incarnation_id)][2])
    finally:
        composer._authored_draft_observer = observer
        if getattr(screen, "_console_process_draft_switch", None) is transition:
            screen._console_process_draft_switch = None
    await _join_selected_switch_entries(client, receipts)


from ...Chat import console_execution_queue_values as _queue_values
from . import prompt_queue as _queue_display

_QUEUE_PRESENTATION = _definition(_queue_values.queue_presentation)
_QUEUE_DISPLAY_SOURCES = tuple(
    (name, _definition(getattr(_queue_display, name)))
    for name in ("derive_prompt_queue_presentation", "with_prompt_queue_start_state")
)


def selected_queue_presentation(screen: ChatScreen, *, sending: bool, collapsed: bool):
    """Combine captured original queue facts with this view's ACK and collapse."""
    from ...Chat.console_domain_services import _definition_current

    client = selected_client(screen)
    if client is None:
        raise RuntimeError("Console queue presentation requires its selected owner")
    if time.monotonic() >= client._readiness_deadline:
        raise ConsoleDisplayExpired("Console queue presentation expired")
    value = client.session_composition.queue
    if (
        type(value) is not _queue_values.ConsoleQueueDisplay
        or value.snapshot.session_id != client.session_composition.session_id
        or value.activity.session_id != value.snapshot.session_id
        or _queue_values.queue_presentation is not _QUEUE_PRESENTATION[0]
        or not _definition_current(_QUEUE_PRESENTATION)
        or any(getattr(_queue_display, name) is not record[0] or not _definition_current(record)
               for name, record in _QUEUE_DISPLAY_SOURCES)
    ):
        raise RuntimeError("Console queue presentation changed")
    return value, _QUEUE_PRESENTATION[0](value, sending=sending, collapsed=collapsed)


from ...Chat import attachment_core as _attachment_gate
from ... import model_capabilities as _capabilities

_VISION_CHECK = _definition(_capabilities.is_vision_capable)
_VISION_LOOKUP = _definition(_capabilities.get_model_capabilities)
_VISION_METHOD = _capabilities.ModelCapabilities.is_vision_capable
_VISION_BODY = _definition(_VISION_METHOD.__wrapped__)
_VISION_CAPS = _definition(_capabilities.ModelCapabilities.get_model_capabilities)


def selected_vision_capability(screen: ChatScreen) -> bool | None:
    """Reuse the matched native fact only for the unchanged original predicate."""
    from ...Chat.console_domain_services import _definition_current

    client = selected_client(screen)
    if client is None:
        return None
    if time.monotonic() >= client._readiness_deadline:
        raise ConsoleDisplayExpired("Console attachment presentation expired")
    if (
        _attachment_gate.is_vision_capable is not _VISION_CHECK[0]
        or _capabilities.is_vision_capable is not _VISION_CHECK[0]
        or _capabilities.get_model_capabilities is not _VISION_LOOKUP[0]
        or _capabilities.ModelCapabilities.is_vision_capable is not _VISION_METHOD
        or _VISION_METHOD.__wrapped__ is not _VISION_BODY[0]
        or _capabilities.ModelCapabilities.get_model_capabilities is not _VISION_CAPS[0]
        or any(not _definition_current(record) for record in (
            _VISION_CHECK, _VISION_LOOKUP, _VISION_BODY, _VISION_CAPS,
        ))
    ):
        return None
    return client.session_composition.vision_capable
