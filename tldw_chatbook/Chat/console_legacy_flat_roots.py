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

A genuine root-level branch exists too: Edit and resend of a conversation's
FIRST message forks a new USER row beside it, parented at the first message's
own parent -- none -- with the resend's reply under it. The saved tree alone
cannot tell that fork from legacy flat data: a flat run whose last USER row was
answered later (Resend on a broken last turn appends the reply directly under
that flat row) has the same shape, a parentless USER row with a reply subtree
after the other flat rows. Guessing from shape hides one or the other: chain
the fork and it reads as later flat messages; split the answered flat row and
every reopen shows -- and sends the model -- only the rows from it down.

So the store does not guess. When the Console creates a sibling of a ROOT
message it records ``MessageMetadata.root_fork`` (stored as
``"root_fork": true`` in the row's local ``metadata_json``) on the new row
(:func:`root_fork_metadata`), and :func:`legacy_flat_chain`:

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
* An unmarked all-USER set whose fork has no reply on either branch chains.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata


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
