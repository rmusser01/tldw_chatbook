"""Shared, loss-aware actions for the local conversation archive."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any


def local_conversation_service(app: Any) -> Any:
    service = getattr(app, "local_chat_conversation_service", None)
    if service is None:
        service = getattr(
            getattr(app, "chat_conversation_scope_service", None), "local_service", None
        )
    if service is None:
        raise RuntimeError("Local conversation storage is unavailable.")
    return service


def archive_failure_copy(reason: str) -> str:
    """Translate storage outcomes into recoverable user-facing explanations."""
    return {
        "already_archived": "Already archived.",
        "already_active": "Already active.",
        "missing_version": "Refresh this conversation before trying again.",
        "stale_version": "Changed since selection; refresh and try again.",
        "not_found": "No longer available.",
    }.get(reason, reason)


async def storage_call(service: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Keep thread-local in-memory test databases on their owning thread."""
    call = getattr(service, method)
    if getattr(getattr(service, "db", None), "is_memory_db", False):
        return call(*args, **kwargs)
    return await asyncio.to_thread(call, *args, **kwargs)


def _session_refusal(app: Any, session: Any) -> str | None:
    if getattr(app, "_conversation_send_inflight", {}).get(
        session.persisted_conversation_id, 0
    ):
        return "Wait for the current send to finish before archiving."
    runtime = getattr(app, "console_runtime", None)
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
    """Explain why archiving would put this open conversation at risk."""
    store = getattr(getattr(app, "console_runtime", None), "chat_store", None)
    for session in store.sessions() if store is not None else ():
        if session.persisted_conversation_id == conversation_id:
            reason = _session_refusal(app, session)
            if reason:
                return reason
    return None


def workspace_archive_refusal(app: Any, workspace_id: str) -> str | None:
    """Apply the same loss checks to every open session in a workspace."""
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
) -> dict[str, dict]:
    """Change safe targets, retaining exact result versions for a later Undo."""
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

    async def complete_change() -> dict[str, dict]:
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
    """Recheck durable state before accepting a send from an already open tab."""
    if not conversation_id:
        return None
    if conversation_id in getattr(app, "_conversation_archive_inflight", ()):
        return "Wait for the archive change to finish; your draft is preserved."
    service = local_conversation_service(app)
    states = await storage_call(
        service, "get_conversation_archive_states", [conversation_id]
    )
    if conversation_id in getattr(app, "_conversation_archive_inflight", ()):
        return "Wait for the archive change to finish; your draft is preserved."
    if states.get(conversation_id) or getattr(
        app, "_conversation_archive_states", {}
    ).get(conversation_id):
        return "This conversation is archived. Open Archived chats and choose Restore & resume."
    return None
