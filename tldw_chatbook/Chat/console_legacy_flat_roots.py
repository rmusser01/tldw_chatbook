"""Which root-level Console messages are legacy flat rows to chain.

Pre-feature Console persistence wrote EVERY message with
``parent_message_id=NULL`` (the base ``_persist_new_message`` hardcoded
``None``), so an existing conversation ``[U1, A1, U2, A2]`` is stored as four
separate roots -- all siblings under ``None``, none with children. On resume
the active-leaf fallback then walks only the LAST root, collapsing the
transcript to its final message. ``ConsoleChatStore._chain_legacy_flat_roots``
(the C1 repair) chains such roots into one linear in-memory spine;
:func:`legacy_flat_chain` decides WHICH roots belong in it, and the store
applies the answer. This decision lives here because the store is under a size
ratchet (``Tests/Architecture/test_module_size_ratchet.py``).

A genuine root-level branch exists too. Edit and resend of a conversation's
FIRST message forks a new USER row beside it, parented at the first message's
own parent -- none -- with the resend's reply under it; so does a prompt sent
after /rewind placed the cursor before the first message. The saved tree alone
cannot tell that fork from legacy flat data: a flat run whose last USER row was
answered later (Resend on a broken last turn appends the reply directly under
that flat row) has the same shape, a parentless USER row with a reply subtree
after the other flat rows. Guessing from shape hides one or the other: chain
the fork and it reads as later flat messages; split the answered flat row and
every reopen shows -- and sends the model -- only the rows from it down.

So the store does not guess. When the Console creates a row at root level
beside an existing root -- a sibling of a ROOT message
(:func:`root_fork_metadata`), or a USER row appended while the cursor sits
before the first message (:func:`appended_metadata`) -- it records
``MessageMetadata.root_fork`` (stored as ``"root_fork": true`` in the row's
local ``metadata_json``) on the new row. A send in a saved conversation writes
its prompt in the turn's own commit, so the marker travels with that commit
(:func:`durable_acceptance`); a completed voice exchange is saved by its own
commit, so its store-resolved destination carries it instead
(:func:`voice_user_root_fork`); and a writer that replaces a row's whole record
carries it over (:func:`keep_root_fork`). And :func:`legacy_flat_chain`:

* leaves every MARKED root out of the chain: it stays an independent root, a
  sibling of the first root, navigable via ``siblings_at``/``set_active_leaf``;
* decides about the UNMARKED roots exactly as the repair did before the marker
  existed. A role-MIXED set (USER and ASSISTANT roots) is legacy flat data and
  chains: the Console never creates an ASSISTANT root beside another root (a
  greeting root is the only root of its chat, and replies always have parents).
  An all-USER set chains only when every root is childless (task-572: repeated
  failed or blocked sends in the flat era); an all-USER set with a reply
  subtree is a genuine first-message fork of a post-branching conversation and
  is left alone.

A fork projection (a forked chat, saved or temporary) is never marked: its rows
are always parent-linked, so it holds no flat rows to tell a fork from, and its
save path stores fork-only facts in the same column instead.

RESIDUAL EDGES. None loses data: the transcript, the Delete prompt's count, its
receipt and the durable delete all follow the same in-memory tree, so whatever
chains is shown, counted, deleted and restored by Undo together.

* A root fork saved before the marker existed is unmarked and keeps the
  unmarked reading: in a legacy flat conversation it chains after the flat
  rows, and Delete on an earlier flat row also deletes it -- the prompt counts
  it and Undo restores it. The same holds for a fork whose row arrives without
  the marker: ``metadata_json`` is local-only, so another synced device, an
  export/import, or a rewrite by an older build does not carry it.
* A generated image or video reply created at a before-first cursor is not
  marked and keeps that reading too (only USER rows are marked on append; a
  video row stores a different record in the same column).
* An unmarked all-USER set whose fork has no reply on either branch chains.

DELETE (:func:`delete_seeds`). The durable subtree delete follows parent links,
so the store passes it the saved ids of its whole in-memory subtree: the
chained flat roots reach the database only that way (TASK-33628.6). A flat
conversation can also hold rows the Console never shows -- a tool-role row
(never a store node) or an empty row (dropped at hydration) -- saved as
parentless roots between the flat rows. They are later rows of the same
chain, so a Delete that removes chained roots also removes every such hidden
root after the first deleted root, in the database's root order
(TASK-33628.7). Undo restores them with the rest. They are not counted in the
prompt, which counts only what the transcript shows. Finding them reads only
the conversation's parentless rows, a page at a time
(``CharactersRAGDB.get_root_message_rows_page``), and only when the delete
removes a chained root.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleDurableTurnAcceptance,
    )
    from tldw_chatbook.Chat.console_voice_promotion import VoicePromotionContext


#: Parentless rows read per page while looking for hidden flat roots.
ROOT_ROWS_PAGE_SIZE = 500


class _ForkProjectionFlag(Protocol):
    fork_projection: bool


def root_fork_metadata(
    parent_native_id: str | None, session: _ForkProjectionFlag
) -> MessageMetadata | None:
    """Return the metadata a new sibling starts with.

    Args:
        parent_native_id: The native parent the sibling is created under --
            its anchor's own parent; ``None`` when the anchor is a root.
        session: The session the sibling is created in.

    Returns:
        ``MessageMetadata(root_fork=True)`` for a new root outside a fork
        projection, otherwise ``None`` (no metadata, as before).
    """
    if parent_native_id is not None or session.fork_projection:
        return None
    return MessageMetadata(root_fork=True)


def appended_metadata(
    metadata: MessageMetadata | None,
    role: ConsoleMessageRole,
    parent_native_id: str | None,
    children: Mapping[str | None, Sequence[str]] | None,
    session: _ForkProjectionFlag,
) -> MessageMetadata | None:
    """Return the metadata an appended row starts with.

    A USER row appended with no parent while the session already holds a root
    (a prompt sent after /rewind placed the cursor before the first message)
    is a new root-level branch and is marked like a root sibling.

    Args:
        metadata: The record the caller supplied, if any.
        role: The appended row's role.
        parent_native_id: The native parent it is appended under -- the
            active leaf, ``None`` at a before-first cursor.
        children: The session's ordered child lists by native parent id.
        session: The session the row is appended to.

    Returns:
        ``metadata`` with ``root_fork`` set for such a row, otherwise
        ``metadata`` unchanged.
    """
    if role is not ConsoleMessageRole.USER or not (children or {}).get(None):
        return metadata
    marker = root_fork_metadata(parent_native_id, session)
    if marker is None:
        return metadata
    return marker if metadata is None else replace(metadata, root_fork=True)


def durable_acceptance(
    acceptance: ConsoleDurableTurnAcceptance,
    nodes: Mapping[str, ConsoleChatMessage],
) -> ConsoleDurableTurnAcceptance:
    """Carry a marked prompt's marker into the durable turn that saves it.

    A send in a saved conversation writes its USER row in the turn's own
    commit, not from the in-memory record, so the marker has to travel with
    the acceptance.

    Args:
        acceptance: The turn about to be committed.
        nodes: The session's tree nodes by native id; the prompt is
            ``acceptance.user_message_id``.

    Returns:
        ``acceptance`` with ``user_root_fork`` set when its prompt is marked
        and saved with no parent, otherwise ``acceptance`` unchanged.
    """
    if acceptance.parent_message_id is not None or not _is_marked(
        nodes.get(acceptance.user_message_id)
    ):
        return acceptance
    return replace(acceptance, user_root_fork=True)


def voice_user_root_fork(
    store: Any, session: _ForkProjectionFlag, context: VoicePromotionContext
) -> bool:
    """Whether a completed voice exchange's prompt is a new root-level branch.

    Its prompt is appended under the claimed native leaf, so the typed
    prompt's rule decides (:func:`appended_metadata`): a USER row appended with
    no parent while the session already holds a root, outside a fork
    projection (TASK-33628.12).

    Args:
        store: The Console store resolving the exchange's destination.
        session: The session the exchange is published to.
        context: The sealed exchange; ``expected_native_leaf_id`` is the
            native parent its prompt is appended under.

    Returns:
        True when the prompt is saved with the ``root_fork`` marker.
    """
    children = store._children_by_parent.get(context.origin.session_id)
    marker = appended_metadata(
        None,
        ConsoleMessageRole.USER,
        context.expected_native_leaf_id,
        children,
        session,
    )
    return marker is not None


def keep_root_fork(
    previous: MessageMetadata | None, metadata: MessageMetadata
) -> MessageMetadata:
    """Carry the marker through a write that replaces a row's whole record.

    Args:
        previous: The row's current record.
        metadata: The record replacing it, composed by a caller that knows
            nothing of the marker (a realtime transcript-status update).

    Returns:
        ``metadata``, with ``root_fork`` set when ``previous`` had it.
    """
    if previous is None or not previous.root_fork or metadata.root_fork:
        return metadata
    return replace(metadata, root_fork=True)


def _is_marked(message: ConsoleChatMessage | None) -> bool:
    return (
        message is not None
        and message.metadata is not None
        and message.metadata.root_fork
    )


def legacy_flat_chain(
    roots: Sequence[str],
    children: Mapping[str | None, Sequence[str]],
    nodes: Mapping[str, ConsoleChatMessage],
) -> list[str]:
    """Return the roots to chain into one spine, in order.

    Args:
        roots: Native root ids, oldest first (the DB's timestamp order).
        children: The session's ordered child lists by native parent id.
        nodes: The session's tree nodes by native id.

    Returns:
        The unmarked roots to link, each onto the one before it, or an empty
        list when they do not chain. Every root NOT returned stays an
        independent root beside the first.
    """
    flat = [root for root in roots if not _is_marked(nodes.get(root))]
    if len(flat) <= 1:
        return []
    has_assistant_root = any(
        nodes[root].role is ConsoleMessageRole.ASSISTANT
        for root in flat
        if root in nodes
    )
    if not has_assistant_root and any(children.get(root) for root in flat):
        return []
    return flat


def delete_seeds(
    store: Any, session_id: str, subtree_ids: Sequence[str]
) -> list[str | None]:
    """Return the ids a Delete of ``subtree_ids`` passes to the durable delete.

    Args:
        store: The Console store about to delete the subtree.
        session_id: The session holding it.
        subtree_ids: The native ids the delete removes, selected row first.

    Returns:
        The subtree's saved ids (``None`` for unsaved nodes, which the
        database ignores), then the hidden flat rows later in its chain.
    """
    seeds, scan_conversation_id = delete_seed_plan(store, session_id, subtree_ids)
    database = getattr(store.persistence, "db", None) if store.persistence else None
    return seeds + hidden_seeds(database, scan_conversation_id, seeds)


def delete_seed_plan(
    store: Any, session_id: str, subtree_ids: Sequence[str]
) -> tuple[list[str | None], str | None]:
    """The in-memory half of :func:`delete_seeds`: no database read.

    TASK-33628.5: the Console reads its tree on the UI thread and runs the
    durable delete -- this lookup's database half included, inside that
    delete's transaction -- off the event loop.

    Args:
        store: The Console store about to delete the subtree.
        session_id: The session holding it.
        subtree_ids: The native ids the delete removes, selected row first.

    Returns:
        The subtree's saved ids, and the conversation whose parentless rows
        :func:`hidden_seeds` must read -- ``None`` when the delete removes no
        chained flat root, so nothing is read.
    """
    nodes = store._nodes_by_session.get(session_id, {})
    seeds = [nodes[n].persisted_message_id for n in subtree_ids if n in nodes]
    if not any(_was_chained(store, nodes, node_id) for node_id in subtree_ids):
        return seeds, None
    session = store._sessions.get(session_id)
    return seeds, getattr(session, "persisted_conversation_id", None)


def hidden_seeds(
    database: Any, conversation_id: str | None, seeds: Sequence[str | None]
) -> list[str]:
    """The database half of :func:`delete_seeds`; reads no store state.

    Args:
        database: The ChaChaNotes database the delete writes.
        conversation_id: What :func:`delete_seed_plan` returned; ``None``
            reads nothing.
        seeds: The subtree's saved ids.

    Returns:
        The hidden flat rows later in the deleted chain.
    """
    reader = getattr(database, "get_root_message_rows_page", None)
    if conversation_id is None or not callable(reader):
        return []
    roots = _root_rows(reader, conversation_id)
    return hidden_rows_after(roots, {seed for seed in seeds if seed})


def _root_rows(
    reader: Callable[..., Sequence[Mapping[str, Any]]], conversation_id: str
) -> Iterator[Mapping[str, Any]]:
    """Yield the conversation's live parentless rows, one page at a time."""
    after: str | None = None
    while True:
        page = reader(
            conversation_id, after_message_id=after, limit=ROOT_ROWS_PAGE_SIZE
        )
        yield from page
        if len(page) < ROOT_ROWS_PAGE_SIZE:
            return
        after = str(page[-1]["id"])


def hidden_rows_after(
    root_rows: Iterable[Mapping[str, Any]], deleted: set[str]
) -> list[str]:
    """Return the never-shown root rows after the first deleted root.

    Args:
        root_rows: The conversation's live parentless rows in the database's
            root order (timestamp order, as resume reads them), shaped as
            ``CharactersRAGDB.get_root_message_rows_page`` returns them.
        deleted: Saved ids the delete already removes.

    Returns:
        Ids of the unmarked tool-role or empty roots positioned after the
        first root in ``deleted``.
    """
    hidden: list[str] = []
    later = False
    for row in root_rows:
        if row["id"] in deleted:
            later = True
        elif later and _never_shown(row):
            hidden.append(str(row["id"]))
    return hidden


def _was_chained(
    store: Any, nodes: Mapping[str, ConsoleChatMessage], node_id: str
) -> bool:
    """Whether the flat repair hung this saved row under another node.

    A restored node's ``parent_message_id`` is its saved parent; a chained
    root's native parent is the previous flat root instead, which is saved.
    A saved row under an unsaved node (an interstitial note) is saved under
    its nearest saved ancestor; that is not a chain.
    """
    node = nodes.get(node_id)
    parent = nodes.get(store._native_parent_by_message.get(node_id))
    return bool(
        node is not None
        and node.persisted_message_id
        and parent is not None
        and parent.persisted_message_id
        and node.parent_message_id != parent.persisted_message_id
    )


def _never_shown(row: Mapping[str, Any]) -> bool:
    """Whether resume leaves this row out of the transcript, and it is unmarked.

    Mirrors ``console_messages_from_conversation_tree`` (a row with no text,
    image, generation state or provider continuation is dropped) and
    ``ConsoleChatStore._ingest_full_tree`` (a tool row is never a node), over
    the presence flags the root-row page carries instead of the columns.
    """
    from tldw_chatbook.Chat.console_conversation_hydration import (
        _console_message_role_from_persisted,
    )

    marker = MessageMetadata.from_json(row.get("metadata_json"))
    if marker is not None and marker.root_fork:
        return False
    shown = (
        bool(row.get("has_content"))
        or bool(row.get("has_image"))
        or bool(row.get("has_generation_state"))
        or bool(row.get("has_provider_continuation"))
    )
    return not shown or (
        _console_message_role_from_persisted(row) is ConsoleMessageRole.TOOL
    )
