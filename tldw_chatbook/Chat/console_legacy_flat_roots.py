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

Historically a GENUINE Console branch was ALWAYS a set of siblings under a
shared *non-None* parent (regenerate / create-sibling parent the new node at
the anchor's parent), NEVER two separate root threads. Phase B's
``edit_and_resend_message`` broke that on purpose: editing-and-resending the
conversation's very FIRST user message forks a NEW root-level USER sibling
(``create_sibling`` parents the fork at the anchor's own parent, ``None``),
with the resend's ASSISTANT reply appended directly under it. An ASSISTANT
node's native parent is never ``None``, so an all-USER root set with a reply
subtree is a genuine Phase-B branch and is left alone (chaining it would splice
the newer branch onto the older as a fake parent-child link).

task-572: a DEGENERATE legacy conversation whose 2+ user turns each got NO
assistant reply (repeated failed/blocked sends in the flat era) also loads as
all-USER roots, but ALL CHILDLESS, so an all-USER root set is chained when
every root is childless. RESIDUAL EDGE: a genuine first-message fork whose
both branches ended up childless is indistinguishable from that and chains.
Non-data-loss (both user rows stay visible, linearly).

A role-MIXED root set (USER and ASSISTANT roots) is legacy flat data -- but it
can ALSO hold genuine forks (TASK-33628.6 review). Editing-and-resending a
legacy flat conversation's first message (the chained spine's root) saves the
fork as one more NULL-parent USER row after every flat row. Chaining it showed
the fork as later messages of the flat transcript, and since Delete follows
the in-memory subtree, deleting an earlier flat row durably tombstoned the
fork and its replies. So in a mixed set a USER root that comes AFTER the last
ASSISTANT root (every flat row predates the fork) and has an ASSISTANT child
of its own stays an independent root; every other root chains. The position
test keeps a flat USER row that gained a regenerated reply mid-run chained (a
flat ASSISTANT row still follows it); the ASSISTANT-child test keeps a trailing
unanswered flat turn chained when it only gained an edited USER turn.
RESIDUAL EDGE, also indistinguishable from the persisted tree alone: a flat run
whose LAST flat row is a USER turn that later gained an ASSISTANT child (its
flat reply was regenerated and the old one deleted) loads as such a fork --
its own root branch, navigable beside the first message. Non-data-loss:
Delete, its prompt and its receipt all follow the same in-memory tree, so
nothing is removed that was not shown and counted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
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
        The roots to link, each onto the one before it. Every root NOT
        returned stays an independent root beside the first. Fewer than two
        ids means nothing chains.
    """
    if len(roots) <= 1:
        return []
    roles = [nodes[root].role if root in nodes else None for root in roots]
    if ConsoleMessageRole.ASSISTANT not in roles:
        # All-USER roots: chain only the degenerate all-childless shape.
        return [] if any(children.get(root) for root in roots) else list(roots)
    last_assistant = max(
        index
        for index, role in enumerate(roles)
        if role is ConsoleMessageRole.ASSISTANT
    )
    return [
        root
        for index, root in enumerate(roots)
        if index <= last_assistant
        or roles[index] is not ConsoleMessageRole.USER
        or not any(
            child in nodes and nodes[child].role is ConsoleMessageRole.ASSISTANT
            for child in children.get(root, ())
        )
    ]
