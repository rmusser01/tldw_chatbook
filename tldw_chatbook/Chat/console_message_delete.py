"""Scoped Console message deletion: confirmation scope and exact subtree Undo.

TASK-33628.2 (Console UX review G1-02). Deleting a Console message removes
the message AND every message beneath it in the conversation tree. This
module owns the two facts the UI needs to make that safe:

* :class:`ConsoleDeleteScope` -- what one confirmed Delete would remove, and
  the copy that states it (the selected row's confirmation, the confirm
  button, and the counted receipt).
* :class:`ConsoleDeletedSubtree` -- everything needed to put exactly those
  messages back: the live node objects (same native ids, so selection,
  speech state and caches keyed by id stay valid), their tree position, the
  previous active leaf and durable cursor, and the committed tombstone
  versions that a version-checked undelete must still match.

The capture/restore pair works directly on the store's tree-registration
primitives -- the ones ``ConsoleChatStore._delete_message`` itself tears
down -- because ``console_chat_store.py`` is held by a size ratchet
(``Tests/Architecture/test_module_size_ratchet.py``) and must not grow.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Mapping

from loguru import logger

from ..DB.ChaChaNotes_DB import ConflictError
from .console_chat_models import ConsoleChatMessage


class ConsoleDeleteUndoError(RuntimeError):
    """Undo was refused; the message is user-facing copy.

    Attributes:
        retryable: Nothing changed and the refusal may be transient (a busy
            database, a pending dispatch), so Undo can be offered again.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def _messages(count: int) -> str:
    return "message" if count == 1 else "messages"


def console_delete_receipt_copy(count: int) -> str:
    """Return the receipt for ``count`` removed messages.

    Args:
        count: How many transcript messages the delete removed.

    Returns:
        The receipt copy, singular for one message.
    """
    if count == 1:
        return "Deleted message from transcript."
    return f"Deleted {count} messages from transcript."


@dataclass(frozen=True)
class ConsoleDeleteScope:
    """What confirming Delete on ``message_id`` removes.

    Attributes:
        message_id: Native id of the selected message.
        subtree_ids: The selected message plus every descendant (all
            branches), as the store would tombstone them.
        off_branch_count: How many of those sit on other branches, out of
            view on the active path.
    """

    message_id: str
    subtree_ids: tuple[str, ...]
    off_branch_count: int = 0

    @property
    def removed_count(self) -> int:
        """Return how many messages the confirmed Delete removes."""
        return len(self.subtree_ids)

    @property
    def prompt(self) -> str:
        """Return the scoped question shown on the selected row."""
        later = self.removed_count - 1
        if later <= 0:
            return "Delete this message?"
        branches = (
            f" ({self.off_branch_count} on other branches)"
            if self.off_branch_count
            else ""
        )
        return f"Delete this message and {later} later {_messages(later)}{branches}?"

    @property
    def guide(self) -> str:
        """Return the row legend while the confirmation is pending."""
        kept = "it" if self.removed_count == 1 else "them"
        return (
            f"{self.prompt} {self.confirm_label} to confirm, Cancel to keep "
            f"{kept}. Undo is offered right after."
        )

    @property
    def confirm_label(self) -> str:
        """Return the confirm button label, naming the real count."""
        return f"Delete {self.removed_count} {_messages(self.removed_count)}"


def console_delete_scope(store: Any, message_id: str) -> ConsoleDeleteScope:
    """Measure what deleting ``message_id`` would remove right now.

    Args:
        store: The Console store holding the message.
        message_id: Native id of the selected message.

    Returns:
        The subtree the delete would tombstone, and how much of it sits off
        the active branch.

    Raises:
        KeyError: The message is no longer in the store.
    """
    subtree = tuple(store.subtree_message_ids(message_id))
    active = set(
        store.active_path_message_ids(store.session_id_for_message(message_id))
    )
    return ConsoleDeleteScope(
        message_id=message_id,
        subtree_ids=subtree,
        off_branch_count=sum(1 for node_id in subtree if node_id not in active),
    )


@dataclass(frozen=True)
class ConsoleDeletedSubtree:
    """Evidence needed to undo one confirmed subtree delete exactly."""

    session_id: str
    root_id: str
    parent_id: str | None
    sibling_index: int
    nodes: tuple[ConsoleChatMessage, ...]
    parents: Mapping[str, str | None]
    on_active_path: bool
    previous_active_leaf: str | None
    conversation_id: str | None
    previous_cursor: tuple[str | None, str | None] | None
    restored_tree_ids: frozenset[str]
    pending_persistence_ids: frozenset[str]
    failed_retry_ids: frozenset[str]
    variant_restored_ids: frozenset[str]
    speech_revisions: Mapping[str, int]
    completion_generations: Mapping[str, int]
    tool_markers: tuple[tuple[str | None, ConsoleChatMessage], ...]
    tombstones: tuple[tuple[str, int], ...] = ()

    @property
    def count(self) -> int:
        """Return how many messages the delete removed."""
        return len(self.nodes)

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return the removed native ids, parent first."""
        return tuple(node.id for node in self.nodes)


def capture_deleted_subtree(store: Any, message_id: str) -> ConsoleDeletedSubtree:
    """Snapshot the subtree rooted at ``message_id`` BEFORE it is deleted.

    Args:
        store: The Console store about to delete the subtree.
        message_id: Native id of the subtree's root message.

    Returns:
        Everything Undo needs to re-register the subtree exactly; its
        ``tombstones`` stay empty until :func:`with_committed_tombstones`.

    Raises:
        KeyError: The message is no longer in the store.
    """
    session_id = store.session_id_for_message(message_id)
    nodes_by_id = store._nodes_by_session.get(session_id, {})
    children = store._children_by_parent.get(session_id, {})
    parent_id = store._native_parent_by_message.get(message_id)
    siblings = children.get(parent_id, [])
    # Pre-order that keeps each parent's child order, so re-registration
    # rebuilds every child list exactly as it was.
    ordered: list[ConsoleChatMessage] = []
    parents: dict[str, str | None] = {}
    stack: list[tuple[str, str | None]] = [(message_id, parent_id)]
    while stack:
        node_id, node_parent = stack.pop()
        ordered.append(nodes_by_id[node_id])
        parents[node_id] = node_parent
        stack.extend((child, node_id) for child in reversed(children.get(node_id, [])))
    ids = set(parents)
    session = store._sessions.get(session_id)
    conversation_id = getattr(session, "persisted_conversation_id", None)
    database = getattr(store.persistence, "db", None) if store.persistence else None
    cursor_reader = getattr(database, "get_conversation_active_cursor", None)
    previous_cursor = (
        tuple(cursor_reader(conversation_id))
        if conversation_id is not None and callable(cursor_reader)
        else None
    )
    return ConsoleDeletedSubtree(
        session_id=session_id,
        root_id=message_id,
        parent_id=parent_id,
        sibling_index=(
            siblings.index(message_id) if message_id in siblings else len(siblings)
        ),
        nodes=tuple(ordered),
        parents=parents,
        on_active_path=message_id in store.active_path_message_ids(session_id),
        previous_active_leaf=store._active_leaf_by_session.get(session_id),
        conversation_id=conversation_id,
        previous_cursor=previous_cursor,
        restored_tree_ids=frozenset(ids & store._restored_tree_message_ids),
        pending_persistence_ids=frozenset(ids & store._pending_persistence_message_ids),
        failed_retry_ids=frozenset(ids & store._failed_retry_message_ids),
        variant_restored_ids=frozenset(ids & store._variant_restored_message_ids),
        speech_revisions={
            node_id: store._message_speech_revisions.get(node_id, 0) for node_id in ids
        },
        completion_generations={
            node_id: store._message_completion_generations.get(node_id, 0)
            for node_id in ids
        },
        tool_markers=tuple(store._tool_markers_by_session.get(session_id, ())),
    )


def with_committed_tombstones(
    store: Any,
    deleted: ConsoleDeletedSubtree,
    *,
    committed_ids: tuple[str, ...] | None = None,
) -> ConsoleDeletedSubtree:
    """Record the tombstone versions the delete just committed.

    Args:
        store: The Console store the delete ran on.
        deleted: The pre-delete capture.
        committed_ids: The persisted ids the durable delete tombstoned. The
            DB deletes its whole ``parent_message_id`` subtree, which can hold
            rows that are never store nodes (tool-role rows, empty rows), so
            these -- not the captured nodes -- are what Undo must restore.
            ``None`` falls back to the captured nodes' persisted ids.

    Returns:
        ``deleted`` carrying the committed ``(message_id, version)``
        tombstones, or ``deleted`` unchanged when nothing was persisted.
    """
    persisted = (
        list(committed_ids)
        if committed_ids is not None
        else [
            node.persisted_message_id
            for node in deleted.nodes
            if node.persisted_message_id
        ]
    )
    database = getattr(store.persistence, "db", None) if store.persistence else None
    reader = getattr(database, "get_message_tombstones", None)
    if not persisted or not callable(reader):
        return deleted
    rows = reader(persisted)
    return replace(
        deleted,
        tombstones=tuple((str(row["message_id"]), int(row["version"])) for row in rows),
    )


def delete_subtree_for_undo(
    store: Any, message_id: str
) -> tuple[ConsoleDeletedSubtree, tuple[str, ...]]:
    """Delete ``message_id`` and its subtree, keeping what Undo needs.

    Recovered-media reference release is held back while Undo is possible;
    the held message ids are returned so the caller can release them once
    the delete is final.

    Args:
        store: The Console store to delete from.
        message_id: Native id of the subtree's root message.

    Returns:
        The Undo snapshot, and the persisted ids whose reference release
        was held back.

    Raises:
        Exception: Whatever ``store.delete_message`` raised.
    """
    deleted = capture_deleted_subtree(store, message_id)
    hold = getattr(store.persistence, "hold_recovered_media_release", None)
    with hold() if callable(hold) else nullcontext(None) as held:
        store.delete_message(message_id)
    # The hold collects exactly the ids this thread's delete tombstoned.
    committed = tuple(held) if held is not None else None
    return (
        with_committed_tombstones(store, deleted, committed_ids=committed),
        committed or (),
    )


def restore_deleted_subtree(store: Any, deleted: ConsoleDeletedSubtree) -> None:
    """Put exactly the deleted messages back, durably and in memory.

    The durable half is one version-checked transaction: every tombstone
    must still be at the version this delete wrote, or nothing changes.

    Args:
        store: The Console store the delete ran on.
        deleted: The snapshot :func:`delete_subtree_for_undo` returned.

    Raises:
        ConsoleDeleteUndoError: The conversation changed so Undo is unsafe.
    """
    session_id = deleted.session_id
    if session_id not in store._sessions:
        raise ConsoleDeleteUndoError("This conversation is no longer open.")
    live_nodes = store._nodes_by_session.get(session_id, {})
    if deleted.parent_id is not None and deleted.parent_id not in live_nodes:
        raise ConsoleDeleteUndoError(
            "The message these belonged under is gone, so they can't be restored."
        )
    if any(node_id in store._message_session_index for node_id in deleted.node_ids):
        raise ConsoleDeleteUndoError("These messages are already back.")
    with store._fork_source_transition(session_id):
        if deleted.tombstones:
            restorer = getattr(store.persistence, "restore_message_subtree", None)
            if not callable(restorer):
                raise ConsoleDeleteUndoError("Saved messages can't be restored here.")
            try:
                with store._dispatch_branch_mutation(session_id):
                    restored_rows = restorer(
                        tombstones=deleted.tombstones,
                        conversation_id=deleted.conversation_id,
                        active_cursor=(
                            deleted.previous_cursor if deleted.on_active_path else None
                        ),
                    )
            except ConflictError as exc:  # a tombstone moved since the delete
                raise ConsoleDeleteUndoError(
                    "These messages changed after they were deleted, so Undo "
                    "can't restore them."
                ) from exc
            except ValueError as exc:  # pending dispatch owns the branch
                raise ConsoleDeleteUndoError(str(exc), retryable=True) from exc
            except Exception as exc:  # noqa: BLE001 - storage refusal, nothing changed
                logger.bind(conversation_id=deleted.conversation_id).warning(
                    "Console delete Undo failed in storage: {}", type(exc).__name__
                )
                raise ConsoleDeleteUndoError(
                    "Undo couldn't finish; the messages are still deleted. Try "
                    "Undo again, or choose Done to keep the delete.",
                    retryable=True,
                ) from exc
            _rebind_versions(deleted, restored_rows)
        _reinsert(store, deleted)
        if deleted.on_active_path and not deleted.tombstones:
            store._persist_active_leaf(session_id, deleted.previous_active_leaf)
    if deleted.tombstones and deleted.conversation_id is not None:
        try:
            store._reconcile_restored_chat_sync_intents(
                session_id, deleted.conversation_id
            )
        except Exception:  # noqa: BLE001 - projection is best-effort, as for deletes
            logger.warning("Failed to project restored Console messages to Sync v2")


def _rebind_versions(deleted: ConsoleDeletedSubtree, rows: Any) -> None:
    """Teach restored nodes the versions the undelete committed.

    Delete and undelete each bump a row's version, so a node that cached its
    pre-delete version would fail its next version-checked write (Regenerate,
    a manual variant) with ConflictError and lose that generation.
    """
    versions = {
        str(row["message_id"]): int(row["version"]) for row in rows or () if row
    }
    for node in deleted.nodes:
        version = versions.get(node.persisted_message_id or "")
        if (
            version is not None
            and type(node.provider_continuation_message_version) is int
        ):
            node.provider_continuation_message_version = version


def _reinsert(store: Any, deleted: ConsoleDeletedSubtree) -> None:
    session_id = deleted.session_id
    for node in deleted.nodes:
        store._register_tree_node(
            session_id, node, parent_native_id=deleted.parents[node.id]
        )
    siblings = store._children_by_parent[session_id][deleted.parent_id]
    siblings.remove(deleted.root_id)
    siblings.insert(min(deleted.sibling_index, len(siblings)), deleted.root_id)
    store._restored_tree_message_ids.update(deleted.restored_tree_ids)
    store._pending_persistence_message_ids.update(deleted.pending_persistence_ids)
    store._failed_retry_message_ids.update(deleted.failed_retry_ids)
    store._variant_restored_message_ids.update(deleted.variant_restored_ids)
    store._message_speech_revisions.update(deleted.speech_revisions)
    store._message_completion_generations.update(deleted.completion_generations)
    restored = set(deleted.node_ids)
    current = store._tool_markers_by_session.get(session_id, [])
    live = {id(marker) for _anchor, marker in current}
    captured = {id(marker) for _anchor, marker in deleted.tool_markers}
    markers = [
        (anchor, marker)
        for anchor, marker in deleted.tool_markers
        if anchor in restored or id(marker) in live
    ] + [(anchor, marker) for anchor, marker in current if id(marker) not in captured]
    for anchor, marker in markers:
        if anchor in restored:
            store._message_session_index[marker.id] = session_id
    if markers:
        store._tool_markers_by_session[session_id] = markers
    if deleted.on_active_path:
        store._active_leaf_by_session[session_id] = deleted.previous_active_leaf
    store._recompute_active_path(session_id)
    store._bump_payload_revision(session_id)
    if deleted.on_active_path:
        store._bump_conversation_context_epoch(session_id)
