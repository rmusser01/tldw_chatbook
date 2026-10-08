"""A Console subtree delete in three phases: plan, durable write, apply.

TASK-33628.5. ``ConsoleChatStore._delete_message`` read the in-memory tree,
tombstoned the subtree in the database and purged the tree in one call, so
the Console's Delete ran its whole durable write on the UI loop: 0.45-1.1 s
for a 3,000-message subtree on a file-backed database. The phases are split
so the durable one can run on another thread:

* :func:`plan_subtree_delete` reads the store (UI thread only): validation,
  the subtree, its saved ids, where the active cursor moves.
* :func:`write_subtree_delete` reads **no store state**, so it may run on any
  thread: the dispatch-branch transaction, the subtree tombstone, the hidden
  flat-root lookup and the cursor move, all in one write transaction.
* :func:`apply_subtree_delete` purges the tree (UI thread only).

``store.delete_message`` runs the three in a row, as before; the Console's
Delete flow (``console_message_delete.delete_subtree_off_loop``) runs the
middle one off the event loop under the store's fork-source and
voice-promotion fences. Kept out of ``console_chat_store.py``, which is held
by a size ratchet (``Tests/Architecture/test_module_size_ratchet.py``), and
imported at first delete, never before first paint (ADR-097).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat import console_legacy_flat_roots as flat_roots
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage


@dataclass(frozen=True)
class SubtreeDeletePlan:
    """Everything the durable write needs, read from the store up front.

    Attributes:
        session_id: The session holding the subtree.
        message_id: Native id of the selected message.
        parent_native_id: Its native parent; the active leaf moves there.
        on_active_path: Whether the selected message is on the active path.
        subtree_ids: Native ids removed, selected message first.
        seeds: Saved ids of the subtree (``None`` for unsaved nodes).
        anchor: The saved id the durable delete starts from, if any.
        scan_conversation_id: Conversation whose parentless rows must be
            read for hidden flat roots (``None``: none are read).
        conversation_id: The session's saved conversation, if any.
        leaf_persisted_id: Saved id of the parent the cursor moves to.
    """

    session_id: str
    message_id: str
    parent_native_id: str | None
    on_active_path: bool
    subtree_ids: tuple[str, ...]
    seeds: tuple[str | None, ...]
    anchor: str | None
    scan_conversation_id: str | None
    conversation_id: str | None
    leaf_persisted_id: str | None


def plan_subtree_delete(store: Any, message_id: str) -> SubtreeDeletePlan:
    """Validate a delete and read what its durable write needs (UI thread).

    Args:
        store: The Console store.
        message_id: Native id of the message to delete with its subtree.

    Returns:
        The plan :func:`write_subtree_delete` and
        :func:`apply_subtree_delete` carry out.

    Raises:
        KeyError: The message is not in the store.
        ValueError: The message is still being generated.
    """
    message = store._message_or_raise(message_id)
    store._materialize_stream_buffer(message)
    if message.status in {"pending", "streaming"}:
        raise ValueError("Wait for response to finish before deleting this message.")
    session_id = store._message_session_index[message_id]
    parent_native_id = store._native_parent_by_message.get(message_id)
    subtree_ids = tuple(store._subtree_ids(session_id, message_id))
    # TASK-33628.6/.7/.9: every saved row, even under an unsaved node.
    seeds, scan_conversation_id = flat_roots.delete_seed_plan(
        store, session_id, subtree_ids
    )
    session = store._sessions.get(session_id)
    parent = store._nodes_by_session.get(session_id, {}).get(parent_native_id)
    return SubtreeDeletePlan(
        session_id=session_id,
        message_id=message_id,
        parent_native_id=parent_native_id,
        on_active_path=message_id in store.active_path_message_ids(session_id),
        subtree_ids=subtree_ids,
        seeds=tuple(seeds),
        anchor=message.persisted_message_id or next(filter(None, seeds), None),
        scan_conversation_id=scan_conversation_id,
        conversation_id=getattr(session, "persisted_conversation_id", None),
        leaf_persisted_id=getattr(parent, "persisted_message_id", None),
    )


def write_subtree_delete(
    store_type: type, persistence: Any, plan: SubtreeDeletePlan
) -> list[dict[str, Any]]:
    """Tombstone the planned subtree durably; reads no store state.

    Args:
        store_type: The store's class, for its stateless write helpers.
        persistence: The store's persistence adapter (``store.persistence``).
        plan: What :func:`plan_subtree_delete` returned.

    Returns:
        The committed tombstones (empty when nothing was saved).

    Raises:
        ValueError: A pending dispatch owns the conversation's branches.
        RuntimeError: The adapter cannot delete messages.
    """
    database = getattr(persistence, "db", None) if persistence is not None else None
    tombstones: list[dict[str, Any]] = []
    with store_type._dispatch_branch_transaction(database, plan.conversation_id):
        if plan.anchor is not None and persistence is not None:
            deleter = getattr(persistence, "delete_message_subtree", None)
            if not callable(deleter):
                raise RuntimeError("Message deletion could not be persisted.")
            saved = [
                *plan.seeds,
                *flat_roots.hidden_seeds(
                    database, plan.scan_conversation_id, plan.seeds
                ),
            ]
            tombstones = deleter(message_id=plan.anchor, subtree_message_ids=saved)
        if (
            plan.on_active_path
            and plan.conversation_id is not None
            and database is not None
            and not store_type._write_active_leaf(
                database,
                plan.conversation_id,
                plan.leaf_persisted_id,
                session_id=plan.session_id,
            )
        ):
            raise ValueError("Resolve pending dispatch before deleting this message.")
    return tombstones


def apply_subtree_delete(
    store: Any, plan: SubtreeDeletePlan, tombstones: list[dict[str, Any]]
) -> ConsoleChatMessage:
    """Purge the deleted subtree from the store once it is durable (UI thread).

    Args:
        store: The Console store the plan was read from.
        plan: The plan :func:`write_subtree_delete` carried out.
        tombstones: What that write committed.

    Returns:
        A snapshot of the deleted selected message.
    """
    session_id, message_id = plan.session_id, plan.message_id
    parent_native_id = plan.parent_native_id
    message = store._message_or_raise(message_id)
    nodes = store._nodes_by_session.get(session_id, {})
    store._project_sync_v2_message_deletes(tombstones)
    for node_id in plan.subtree_ids:
        store._invalidate_generation_attempt(node_id)
    children_map = store._children_by_parent.get(session_id, {})
    # Detach the deleted node from its parent's ordered child list.
    siblings = children_map.get(parent_native_id)
    if siblings is not None and message_id in siblings:
        siblings.remove(message_id)
        if not siblings:
            children_map.pop(parent_native_id, None)
    # Purge the deleted node AND its whole subtree from every structure --
    # deleting a mid-conversation node drops the branch beneath it.
    for node_id in plan.subtree_ids:
        store.clear_terminal_citation_state(node_id)
        nodes.pop(node_id, None)
        children_map.pop(node_id, None)
        store._native_parent_by_message.pop(node_id, None)
        store._restored_tree_message_ids.discard(node_id)
        store._message_session_index.pop(node_id, None)
        store._stream_chunks_by_message.pop(node_id, None)
        store._stream_materialized_counts.pop(node_id, None)
        store._pending_persistence_message_ids.discard(node_id)
        store._variant_stream_bases.pop(node_id, None)
        store._variant_restored_message_ids.discard(node_id)
        store._failed_retry_message_ids.discard(node_id)
        store._message_speech_revisions.pop(node_id, None)
        store._message_completion_generations.pop(node_id, None)
        store._exchange_blob_cache.pop(node_id, None)
    # Only when the deleted branch was on the active path does the leaf move
    # (up to the deleted node's parent); an off-path delete leaves it alone.
    store._purge_tool_markers(session_id, set(plan.subtree_ids))
    if plan.on_active_path:
        store._active_leaf_by_session[session_id] = parent_native_id
    store._recompute_active_path(session_id)
    store._bump_payload_revision(session_id)
    if plan.on_active_path:
        store._bump_conversation_context_epoch(session_id)
    return store._snapshot(message)
