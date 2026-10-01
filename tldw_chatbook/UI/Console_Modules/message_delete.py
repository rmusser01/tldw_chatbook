"""Console message Delete flow: scoped in-row confirmation, receipt, Undo.

TASK-33628.2 (Console UX review G1-02). The flow the message controller
delegates its ``delete``/``delete-confirm``/``delete-cancel`` actions to:

1. **Arm.** The first Delete measures the real scope (the message plus every
   later message beneath it) and turns the selected row's action bar into
   ``[Delete N messages] [Cancel]`` with the scoped question as its legend
   (the transcript renders it from :func:`pending_delete_scope`).
2. **Confirm.** Deletes only the scope the user saw -- if the subtree
   changed meanwhile the confirmation re-arms with the new count instead.
   Recovered-media reference release is held back while Undo is possible.
3. **Receipt.** A counted receipt (the Archive receipt pattern) offers Undo;
   Undo restores exactly the tombstoned rows (version-checked) and the
   previous active branch. Done makes the delete final and releases held
   references, warning only if cleanup is genuinely pending.

``host`` is the ``ConsoleMessageController``; everything is reached through
the attributes its delete branch already used, plus ``push_screen``.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ...Chat.console_message_actions import ConsoleActionResult
from ...Chat.console_message_delete import (
    ConsoleDeletedSubtree,
    ConsoleDeleteScope,
    ConsoleDeleteUndoError,
    console_delete_receipt_copy,
    console_delete_scope,
    delete_subtree_for_undo,
    restore_deleted_subtree,
)

#: Actions the delete flow owns end to end (the confirm/cancel pair only
#: exists on a row whose delete is pending).
CONSOLE_DELETE_ACTION_IDS = frozenset({"delete", "delete-confirm", "delete-cancel"})


def pending_delete_scope(host: Any) -> ConsoleDeleteScope | None:
    """Return the armed scope while its message is still the pending one.

    Args:
        host: The Console message controller.

    Returns:
        The scope the pending confirmation shows, or ``None``.
    """
    scope = getattr(host, "_console_delete_scope", None)
    pending = getattr(host, "_pending_console_delete_message_id", None)
    if scope is None or pending is None or scope.message_id != pending:
        return None
    return scope


def pending_delete_copy(host: Any) -> str:
    """Return the Inspector copy for a pending delete.

    Args:
        host: The Console message controller.

    Returns:
        The scoped question plus where to answer it.
    """
    scope = pending_delete_scope(host)
    if scope is None:
        return "Confirm or cancel the delete on the selected message."
    return f"{scope.prompt} Confirm or cancel on the selected message."


async def handle_console_delete_action(
    host: Any, action_id: str, message_id: str
) -> bool:
    """Arm, confirm or cancel one scoped message delete.

    Args:
        host: The Console message controller.
        action_id: ``delete``, ``delete-confirm`` or ``delete-cancel``.
        message_id: Native id of the selected message.

    Returns:
        Always True: the delete flow owns these actions.
    """
    if action_id == "delete-cancel":
        host._pending_console_delete_message_id = None
        host._console_delete_scope = None
        host._last_console_action = ConsoleActionResult(
            action_id="delete",
            status="completed",
            visible_copy="Delete cancelled; nothing was removed.",
            target_message_id=message_id,
        )
        await host._sync_native_console_chat_ui()
        return True
    store = host._ensure_console_chat_store()
    try:
        message = store.get_message(message_id)
        scope = console_delete_scope(store, message_id)
    except KeyError:
        host.app_instance.notify(
            "Console message action target no longer exists.", severity="warning"
        )
        return True
    result = host._console_message_action_service.dispatch("delete", message)
    if result.status != "completed":
        host.app_instance.notify(result.visible_copy, severity="warning")
        return True
    armed = pending_delete_scope(host)
    if host._pending_console_delete_message_id != message_id:
        await _arm(host, scope)
        return True
    if armed is not None and set(armed.subtree_ids) != set(scope.subtree_ids):
        await _arm(host, scope)
        host.app_instance.notify(
            "The messages under this one changed. Check the new count, then "
            "confirm again.",
            severity="warning",
        )
        return True
    await _delete(host, store, scope)
    return True


async def _arm(host: Any, scope: ConsoleDeleteScope) -> None:
    host._pending_console_delete_message_id = scope.message_id
    host._console_delete_scope = scope
    host._last_console_action = ConsoleActionResult(
        action_id="delete",
        status="blocked",
        visible_copy=scope.prompt,
        target_message_id=scope.message_id,
    )
    await host._sync_native_console_chat_ui()


async def _delete(host: Any, store: Any, scope: ConsoleDeleteScope) -> None:
    message_id = scope.message_id
    host._pending_console_delete_message_id = None
    host._console_delete_scope = None
    # Original-attempt previews are in-memory only, so they are cleared when
    # the delete becomes final (_finalize), never before: Undo restores the
    # same node objects and nothing else could rebuild those previews.
    try:
        deleted, held_ids = delete_subtree_for_undo(store, message_id)
    except ValueError as exc:  # a pending dispatch or live reply owns it
        host.app_instance.notify(str(exc), severity="warning")
        await host._sync_native_console_chat_ui()
        return
    except Exception as exc:  # noqa: BLE001 - report at the UI boundary
        logger.warning("Console message delete failed: {}", type(exc).__name__)
        host.app_instance.notify(
            "Delete could not complete. Reopen this chat to see what is saved.",
            severity="error",
        )
        await host._sync_native_console_chat_ui()
        return
    host._invalidate_console_fork_image_selections(scope.subtree_ids)
    # TASK-251: a deleted message can change what the browser row shows for
    # this conversation (title/updated_at) -- invalidate so the next sync
    # reflects it immediately.
    host._invalidate_console_persisted_rows_cache()
    host._last_console_action = ConsoleActionResult(
        action_id="delete",
        status="completed",
        visible_copy=console_delete_receipt_copy(deleted.count),
        target_message_id=message_id,
    )
    await host._sync_native_console_chat_ui()
    await _offer_receipt(host, store, deleted, held_ids)


async def _offer_receipt(
    host: Any,
    store: Any,
    deleted: ConsoleDeletedSubtree,
    held_ids: tuple[str, ...],
) -> None:
    from ...Widgets.Console.console_message_delete_receipt import (
        ConsoleMessageDeleteReceiptModal,
    )

    async def settle(choice: str | None) -> None:
        if choice != "undo":
            await _finalize(host, store, deleted, held_ids)
            return
        try:
            restore_deleted_subtree(store, deleted)
        except ConsoleDeleteUndoError as exc:
            host.app_instance.notify(str(exc), severity="warning")
            if exc.retryable:  # nothing changed; keep Undo on offer
                await _offer_receipt(host, store, deleted, held_ids)
            else:
                await _finalize(host, store, deleted, held_ids)
            return
        noun = "message" if deleted.count == 1 else "messages"
        host._last_console_action = ConsoleActionResult(
            action_id="delete",
            status="completed",
            visible_copy=f"Restored {deleted.count} {noun}.",
            target_message_id=deleted.root_id,
        )
        host._invalidate_console_persisted_rows_cache()
        # Land the reader on what came back (applied when the transcript
        # ingests the restored rows).
        host._pending_console_swipe_selection = deleted.root_id
        await host._sync_native_console_chat_ui()
        host.app_instance.notify(
            f"Restored {deleted.count} {noun}.", severity="information"
        )

    await host.push_screen(
        ConsoleMessageDeleteReceiptModal(count=deleted.count), callback=settle
    )


async def _finalize(
    host: Any,
    store: Any,
    deleted: ConsoleDeletedSubtree,
    held_ids: tuple[str, ...],
) -> None:
    """Make the delete final: drop its previews and release held references.

    Args:
        host: The Console message controller.
        store: The Console store the delete ran on.
        deleted: The delete that can no longer be undone.
        held_ids: Persisted ids whose recovered-media release Undo held back.
    """
    # The session-wide clear the delete always made (the deleted ids are no
    # longer in the store, so the controller drops them as unknown).
    host._ensure_console_chat_controller().clear_original_attempts_for_session(
        deleted.session_id
    )
    host._console_original_attempt_previews.clear()
    persistence = store.persistence
    release = getattr(persistence, "release_recovered_media_references", None)
    if held_ids and callable(release) and release(held_ids):
        warning = getattr(persistence, "recovered_media_cleanup_warning", None)
        if warning:
            host.app_instance.notify(warning, severity="warning")
    await host._sync_native_console_chat_ui()
