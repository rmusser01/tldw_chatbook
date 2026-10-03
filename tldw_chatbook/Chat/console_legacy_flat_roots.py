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
(:func:`durable_acceptance`), and a writer that replaces a row's whole record
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
* Other rows created at a before-first cursor are not marked and keep that
  reading too: a completed voice exchange's prompt (its commit proof requires
  the user row to carry no metadata), and a generated image or video reply
  (only USER rows are marked on append; a video row stores a different record
  in the same column).
* An unmarked all-USER set whose fork has no reply on either branch chains.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleDurableTurnAcceptance,
    )


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
