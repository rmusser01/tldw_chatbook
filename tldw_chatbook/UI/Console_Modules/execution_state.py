"""Read the selected client's already-retired values on its actual screen visit.

Import lazily after ChatScreen is defined. No function constructs a runtime,
reads native storage, changes a session, or grants command authority.
"""

from __future__ import annotations

import time
from dataclasses import replace

from ...Chat.console_control_dispatch import ConsoleProjection
from ...Chat.console_execution_process import ConsoleExecutionClient
from ...Chat.console_execution_projection_view import ConsoleProjectionView
from ...Chat.console_execution_composition_values import ConsoleSessionComposition
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
)
from ..Screens.chat_screen import ChatScreen
from .execution_projection import _definition, _require_methods, _require_sources

_CLIENT_METHODS = (("require_view", _definition(ConsoleExecutionClient.require_view)),)
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
    client.require_view(screen, client._visit)
    view, frame = client.projection_view, client._projection
    composition = client.session_composition
    if (
        type(composition) is not ConsoleSessionComposition
        or type(view) is not ConsoleProjectionView
        or type(frame) is not ConsoleProjection
        or view.source_revision != client._source_revision
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
    return client


def provider_model_display(
    screen: ChatScreen,
    client: ConsoleExecutionClient,
) -> tuple[str, str | None, ConsoleSessionSettings]:
    """Consume this exact projection's effective provider/model display facts."""
    if selected_client(screen) is not client:
        raise RuntimeError("Console provider presentation owner changed")
    if time.monotonic() >= client._readiness_deadline:
        raise RuntimeError("Console readiness presentation expired")
    return client.provider_model_display


def settings_readiness(
    screen: ChatScreen,
    client: ConsoleExecutionClient,
) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness]:
    """Apply only the original screen-local model warning to backend readiness."""
    if selected_client(screen) is not client:
        raise RuntimeError("Console readiness presentation owner changed")
    if time.monotonic() >= client._readiness_deadline:
        raise RuntimeError("Console readiness presentation expired")
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
        "read_projection", "_join_hidden_checkpoint", "_view_checkpoint_finished",
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
        raise RuntimeError("Console context presentation expired")
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


async def refresh_selected_view(screen: ChatScreen, *, checkpoint: str | None = None) -> bool:
    """Join the named projection and apply its original mounted surfaces."""
    from .execution_projection import apply_console_process_projection
    from .execution_sessions import apply_console_process_sessions

    client = selected_client(screen)
    if client is None:
        raise RuntimeError("Console process refresh requires its selected owner")
    visit = client._visit
    row = client.projection_view.session(client.projection_view.active_session_id)
    projection = await client.read_projection(
        row.id, row.incarnation_id, _checkpoint=checkpoint,
    )
    client.require_view(screen, visit)
    if selected_client(screen) is not client:
        raise RuntimeError("Console refreshed publication changed")
    screen._console_process_projection_view = client.projection_view
    if not await apply_console_process_sessions(
        screen, client, client.session_surface, visit=visit,
    ):
        return False
    client.require_view(screen, visit)
    if not apply_console_process_projection(screen, client, projection, visit=visit):
        return False
    _require_methods(screen, _REFRESH_METHODS)
    with screen._console_derivation_scope():
        control = screen._build_console_control_state(screen._pending_console_launch_context)
        screen._push_console_control_state_if_changed(
            control, screen._build_console_workbench_state(control),
        )
        screen._sync_console_settings_summary()
    client.require_view(screen, visit)
    return True


_REFRESH_METHODS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in (
        "_console_derivation_scope", "_build_console_control_state",
        "_build_console_workbench_state", "_push_console_control_state_if_changed",
        "_sync_console_settings_summary", "_sync_console_rail_system_line",
        "_sync_console_agent_section",
    )
)


def resume_selected_view(screen: ChatScreen) -> None:
    """Issue reconciliation from the actual screen message-pump visit."""
    _require_sources()
    client = screen.app_instance.console_execution_client
    if type(client) is not ConsoleExecutionClient or type(screen) is not ChatScreen:
        raise TypeError("Original Console process view is required")
    _require_methods(client, _PREPARE_METHODS)
    visit = client.attach_view(screen)
    if getattr(screen, "_console_process_visit", None) != visit:
        screen._console_process_visit = visit
        screen._console_process_surface_reconciled = False
    if not getattr(screen, "_console_process_surface_reconciled", False):
        screen.call_after_refresh(screen._start_console_attach_reconciliation)


async def reconcile_selected_view(screen: ChatScreen) -> None:
    """Join the real remote visit and publish its original mounted values."""
    client = selected_client(screen)
    if client is None or screen._console_attach_reconcile_running:
        return
    visit = client._visit
    screen._console_attach_reconcile_running = True
    try:
        action = "mount" if getattr(screen, "_console_process_mounted_visit", None) != visit else None
        if not await refresh_selected_view(screen, checkpoint=action):
            return
        client.require_view(screen, visit)
        screen._console_process_mounted_visit = visit
        screen._console_process_surface_reconciled = True
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
    except Exception:
        if client._view is screen and client._visit == visit:
            raise
        # Superseded projection was not adopted; its command retains its
        # original physical owners independently of this removed screen.
    finally:
        screen._console_attach_reconcile_running = False


def refresh_selected_when_due(screen: ChatScreen) -> None:
    """Request the original readiness refresh after its published expiry."""
    if (
        not screen.is_attached or not screen.is_current
        or getattr(screen, "_console_sync_maintenance_paused", False)
        or screen._console_attach_reconcile_running
    ):
        return
    client = selected_client(screen)
    if client is None or time.monotonic() < client._readiness_deadline:
        return
    screen._console_attach_reconcile_worker = screen.run_worker(
        lambda: reconcile_selected_view(screen),
        group="console-attach-reconciliation", exclusive=False,
    )


_MOUNT_UI_METHODS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in (
        "_restore_collapsible_states", "sync_task_resume_state",
        "_restore_console_workbench_focus", "_sync_console_dictation_availability",
        "_record_ui_timer_created",
    )
)
