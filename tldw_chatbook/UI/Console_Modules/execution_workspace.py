"""Selected Console New Chat entry through the original controller command."""

from __future__ import annotations

import asyncio

from .execution_state import (
    begin_selected_session_switch,
    bind_selected_session_draft,
    cancel_selected_session_switch,
    refresh_selected_view,
    selected_client,
)


async def create_selected_chat(
    screen, *, workspace_id: str | None = None, title: str = "",
    ephemeral: bool = False,
) -> bool:
    """Retain the whole original switch when only its observer is cancelled."""
    client = selected_client(screen)
    if client is None:
        raise RuntimeError("Console New Chat requires its selected owner")
    previous = client._workspace_chat_tail
    if client._workspace_operations and previous is None:
        raise RuntimeError("Console workspace operation is already pending")
    if previous is not None:
        visit, source = client._visit, client._source_revision
        origin = client.projection_view.session(client.projection_view.active_session_id)

    async def open_chat():
        if previous is not None:
            if not await asyncio.shield(previous):
                return False
            client.require_view(screen, visit)
            row = client.projection_view.session(client.projection_view.active_session_id)
            if (
                selected_client(screen) is not client or client._source_revision != source
                or (row.id, row.incarnation_id, row.workspace_id)
                != (origin.id, origin.incarnation_id, origin.workspace_id)
            ):
                return False
        return await _create_selected_chat(
            screen, client, workspace_id=workspace_id, title=title, ephemeral=ephemeral,
        )

    def finished(task):
        client._workspace_operations.discard(task)
        if client._workspace_chat_tail is task:
            client._workspace_chat_tail = None
        if not task.cancelled():
            task.exception()

    task = asyncio.create_task(
        open_chat(), name="console-workspace-new-chat",
    )
    client._workspace_chat_tail = task
    client._workspace_operations.add(task)
    task.add_done_callback(finished)
    return await asyncio.shield(task)


async def _create_selected_chat(screen, client, *, workspace_id, title, ephemeral):
    """Join the actual attach, then preserve the original composer transition."""
    from .execution_sessions import apply_console_process_sessions

    await asyncio.sleep(0)
    if selected_client(screen) is not client:
        raise RuntimeError("Console New Chat owner changed")
    visit = client._visit
    worker = getattr(screen, "_console_attach_reconcile_worker", None)
    if worker is not None and not worker.is_finished:
        await worker.wait()
        client.require_view(screen, visit)
    if selected_client(screen) is not client:
        raise RuntimeError("Console New Chat owner changed")
    old = client.projection_view.session(client.projection_view.active_session_id)
    owner = screen._session
    owner._capture_console_draft_switch_snapshot()
    begin_selected_session_switch(screen)
    try:
        await client.open_new_chat(
            screen, workspace_id=workspace_id or old.workspace_id, title=title,
            ephemeral=ephemeral,
            new_chat_default_generation=owner._console_new_chat_default_generation(),
        )
        client.require_view(screen, visit)
        # OpenSession has already produced the new tab rows. Install them
        # before publishing the new composer, so the next click targets the
        # workspace's current strip while the full refresh is in flight.
        await apply_console_process_sessions(
            screen, client, client.session_surface, visit=visit,
        )
        client.require_view(screen, visit)
        await bind_selected_session_draft(screen)
        client.require_view(screen, visit)
        if not await refresh_selected_view(screen):
            return False
        client.require_view(screen, visit)
        owner._invalidate_console_persisted_rows_cache()
        owner._sync_console_temporary_chip()
        owner._focus_console_composer_if_needed(force=True)
        return True
    except BaseException:
        if (client._view is screen and client._visit == visit
            and client.projection_view.active_session_id == old.id):
            await cancel_selected_session_switch(screen)
        raise
