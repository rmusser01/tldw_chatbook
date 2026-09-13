"""Shared, loss-aware actions for the local conversation archive."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any


def local_conversation_service(app: Any) -> Any:
    """Resolve the local service that owns durable conversation archive state.

    Args:
        app: Application exposing a local service or a scoped service wrapper.

    Returns:
        The local conversation service, never its remote counterpart.

    Raises:
        RuntimeError: If local conversation storage is unavailable.
    """
    service = getattr(app, "local_chat_conversation_service", None)
    if service is None:
        service = getattr(
            getattr(app, "chat_conversation_scope_service", None), "local_service", None
        )
    if service is None:
        raise RuntimeError("Local conversation storage is unavailable.")
    return service


def archive_failure_copy(reason: str) -> str:
    """Translate storage outcomes into recoverable user-facing explanations.

    Args:
        reason: Storage failure code or an existing user-facing refusal.

    Returns:
        Actionable copy for known codes, otherwise the supplied reason.
    """
    return {
        "already_archived": "Already archived.",
        "already_active": "Already active.",
        "missing_version": "Refresh this conversation before trying again.",
        "stale_version": "Changed since selection; refresh and try again.",
        "not_found": "No longer available.",
    }.get(reason, reason)


async def storage_call(service: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Run storage off-loop, retaining thread-local memory database ownership.

    Args:
        service: Service owning the storage operation and optional database.
        method: Name of the service method to invoke.
        *args: Positional arguments forwarded to that method.
        **kwargs: Keyword arguments forwarded to that method.

    Returns:
        The service method's result.

    Raises:
        AttributeError: If the requested method is unavailable. Exceptions from
            the storage method propagate to the caller's recovery boundary.
    """
    call = getattr(service, method)
    if getattr(getattr(service, "db", None), "is_memory_db", False):
        return call(*args, **kwargs)
    return await asyncio.to_thread(call, *args, **kwargs)


def capture_console_archive_draft(app: Any, *, screen: Any = None) -> None:
    """Capture unsent composer text before checking archive eligibility.

    Args:
        app: Application owning the Console runtime and reusable screens.
        screen: Optional caller's Console screen, including embedded hosts.
    """
    store = getattr(getattr(app, "console_runtime", None), "chat_store", None)
    if store is None:
        return
    candidates = [screen] if screen is not None else []
    candidates.extend(reversed(tuple(getattr(app, "screen_stack", ()))))
    candidates.extend(
        entry[1] for entry in getattr(app, "_reusable_screen_instances", {}).values()
    )
    seen = set()
    session_ids = {session.id for session in store.sessions()}
    for console in candidates:
        if (
            console is None
            or id(console) in seen
            or not getattr(console, "is_mounted", False)
        ):
            continue
        seen.add(id(console))
        owner_id = getattr(console, "_console_visible_draft_session_id", None)
        composer_lookup = getattr(console, "_console_composer_or_none", None)
        if owner_id not in session_ids or not callable(composer_lookup):
            continue
        composer = composer_lookup()
        if composer is not None:
            draft = composer.draft_text()
            if draft or owner_id == getattr(store, "active_session_id", None):
                # An empty settled composer clears its stale stored draft;
                # during a switch it must not clear another session's draft.
                # Never assign the visible text to a newer active session.
                store.set_session_draft(owner_id, draft)


def _session_refusal(app: Any, session: Any) -> str | None:
    if getattr(app, "_conversation_send_inflight", {}).get(
        session.persisted_conversation_id, 0
    ):
        return "Wait for the current send to finish before archiving."
    runtime = getattr(app, "console_runtime", None)
    voice = getattr(runtime, "_voice_process_supervisor", None)
    if voice is not None and voice.has_live_session(session.id):
        return "Stop Hands-free and wait for its work to finish before archiving."
    controller = getattr(runtime, "chat_controller", None)
    if (
        controller is not None
        and controller.lifecycle_impact(session_id=session.id).has_loss_risk
    ):
        return "Finish or cancel running or queued work before archiving."
    if getattr(session, "draft", "") or getattr(session, "pending_attachments", ()):
        return "Send or clear the draft and attachments before archiving."
    store = getattr(runtime, "chat_store", None)
    if store is None:
        return None
    # Inactive branches can still contain unsaved or pending tree nodes.
    messages_for_session = (
        getattr(store, "all_messages_for_session", None) or store.messages_for_session
    )
    pending = getattr(store, "_pending_persistence_message_ids", set())
    if any(
        message.id in pending or not getattr(message, "persisted_message_id", None)
        for message in messages_for_session(session.id)
    ):
        return "Wait for the transcript to finish saving before archiving."
    return None


def conversation_archive_refusal(app: Any, conversation_id: str) -> str | None:
    """Explain why archiving would put this open conversation at risk.

    Args:
        app: Application owning the Console runtime and archive reservations.
        conversation_id: Persisted local conversation identity to check.

    Returns:
        A refusal for running, queued or unsaved work, otherwise None.
    """
    capture_console_archive_draft(app)
    store = getattr(getattr(app, "console_runtime", None), "chat_store", None)
    for session in store.sessions() if store is not None else ():
        if session.persisted_conversation_id == conversation_id:
            reason = _session_refusal(app, session)
            if reason:
                return reason
    return None


def workspace_archive_refusal(app: Any, workspace_id: str) -> str | None:
    """Apply the same loss checks to every open session in a workspace.

    Args:
        app: Application owning the Console runtime and retained composers.
        workspace_id: Workspace whose open sessions must be safe to archive.

    Returns:
        The first unsafe-session refusal, otherwise None.
    """
    capture_console_archive_draft(app)
    store = getattr(getattr(app, "console_runtime", None), "chat_store", None)
    for session in store.sessions() if store is not None else ():
        if getattr(session, "workspace_id", None) == workspace_id:
            reason = _session_refusal(app, session)
            if reason:
                return reason
    return None


async def change_conversation_archive(
    app: Any,
    conversation_ids: Iterable[str],
    *,
    archived: bool,
    expected_versions: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Change safe targets, retaining exact result versions for a later Undo.

    Args:
        app: Application owning local storage, Console state and reservations.
        conversation_ids: Persisted identities; duplicate targets are coalesced.
        archived: True to archive, False to restore.
        expected_versions: Observed version for each target. Missing or stale
            versions are reported as per-target failures instead of overwritten.

    Returns:
        A mapping with ``changed`` (identity to committed optimistic version)
        and ``failures`` (identity to refusal or storage failure code). Undo must
        use exactly the returned successful identities and versions.

    Raises:
        RuntimeError: If local storage is unavailable. Storage exceptions
            propagate; reservations remain held until the actual operation ends.
        asyncio.CancelledError: If the caller is cancelled. An already started
            storage mutation still finishes and publishes its result safely.
    """
    inflight = getattr(app, "_conversation_archive_inflight", None)
    if inflight is None:
        inflight = app._conversation_archive_inflight = set()
    failures = {}
    safe = []
    for conversation_id in dict.fromkeys(conversation_ids):
        reason = (
            "An archive change is already in progress."
            if conversation_id in inflight
            else conversation_archive_refusal(app, conversation_id)
            if archived
            else None
        )
        if reason:
            failures[conversation_id] = reason
        else:
            safe.append(conversation_id)
    inflight.update(safe)

    async def complete_change() -> dict[str, dict[str, Any]]:
        # Cancellation of a Textual worker cannot cancel a running SQLite
        # thread. This task retains the reservation through the actual commit
        # and publishes the result before accepting any later send.
        try:
            result = (
                await storage_call(
                    local_conversation_service(app),
                    "set_conversations_archived",
                    safe,
                    archived=archived,
                    expected_versions={
                        key: expected_versions[key]
                        for key in safe
                        if key in expected_versions
                    },
                )
                if safe
                else {"changed": {}, "failures": {}}
            )
            states = getattr(app, "_conversation_archive_states", None)
            if states is None:
                states = app._conversation_archive_states = {}
            states.update({key: archived for key in result["changed"]})
            if result["changed"]:
                app._conversation_archive_generation = (
                    getattr(app, "_conversation_archive_generation", 0) + 1
                )
            return {
                "changed": result["changed"],
                "failures": {**failures, **result["failures"]},
            }
        finally:
            inflight.difference_update(safe)

    operation = asyncio.create_task(complete_change())
    operation.add_done_callback(
        lambda task: task.exception() if not task.cancelled() else None
    )
    return await asyncio.shield(operation)


async def conversation_send_refusal(
    app: Any, conversation_id: str | None
) -> str | None:
    """Reconcile durable state before accepting a send from an already open tab.

    Args:
        app: Application owning local storage and archive reservations/cache.
        conversation_id: Persisted identity, or None for a new unsaved chat.

    Returns:
        A refusal if archived or if an archive mutation overlaps the read,
        otherwise None. A current read replaces stale cached lifecycle state.

    Raises:
        RuntimeError: If local storage is unavailable. Storage exceptions
            propagate so the send boundary can preserve the draft and report it.
    """
    if not conversation_id:
        return None
    if conversation_id in getattr(app, "_conversation_archive_inflight", ()):
        return "Wait for the archive change to finish; your draft is preserved."
    service = local_conversation_service(app)
    generation = getattr(app, "_conversation_archive_generation", 0)
    states = await storage_call(
        service, "get_conversation_archive_states", [conversation_id]
    )
    if conversation_id in getattr(app, "_conversation_archive_inflight", ()) or (
        generation != getattr(app, "_conversation_archive_generation", 0)
    ):
        return "Wait for the archive change to finish; your draft is preserved."
    cached = getattr(app, "_conversation_archive_states", None)
    if cached is None:
        cached = app._conversation_archive_states = {}
    if conversation_id in states:
        cached[conversation_id] = states[conversation_id]
    else:
        cached.pop(conversation_id, None)
    if states.get(conversation_id):
        return "This conversation is archived. Open Archived chats and choose Restore & resume."
    return None
