"""The rewritten-tail proof admits only the current saved branch.

TASK-33621.29 / TASK-33621.7 (ADR-097 amendment 2026-10-27). The planner may
replace a recorded surface tail after Compact now or Edit & resend, but only
when every replacement row is a saved message on the current branch, linked
parent to child, ending at the current user, and the oldest row hangs off a
proven anchor. These cases pin each refusal over a real SQLite database with
the two tables the proof reads; ``Tests/Chat/test_console_trace_rewritten_
history.py`` drives the admitted shapes end to end.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_chatbook.Chat.console_trace_final_values import (
    CompletedToolTurnWitness,
    RewrittenTailWitness,
    SurfaceDeltaAdmission,
    VerifiedSurfaceDelta,
    VerifiedSurfaceDeltaItem,
    VerifiedSurfaceReplacement,
    VerifiedSurfaceReplacementRange,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy
from tldw_chatbook.Chat.console_trace_provenance import (
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_rewritten_tail import (
    build_rewritten_tail_witness,
    validate_rewritten_tail,
)

CONVERSATION = str(uuid4())
OWNER = str(uuid4())


class _Branch:
    """A conversation tree with one revision per message and a fake ledger."""

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.executescript(
            """
            CREATE TABLE messages(
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                parent_message_id TEXT, deleted INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE console_conversation_memories(
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                boundary_message_id TEXT, active INTEGER NOT NULL DEFAULT 1);
            """
        )
        self.revisions: dict[str, SimpleNamespace] = {}

    def message(self, parent: str | None, conversation: str = CONVERSATION) -> str:
        message_id = str(uuid4())
        self.db.execute(
            "INSERT INTO messages(id, conversation_id, parent_message_id) "
            "VALUES (?, ?, ?)",
            (message_id, conversation, parent),
        )
        revision_id = str(uuid4())
        self.revisions[message_id] = SimpleNamespace(
            revision_id=revision_id,
            source_conversation_id=conversation,
            source_message_id=message_id,
        )
        return message_id

    def saved(self, message_id: str) -> SavedRevisionTraceProvenance:
        return SavedRevisionTraceProvenance(self.revisions[message_id].revision_id)

    def remember(self, boundary: str, *, active: bool = True) -> None:
        self.db.execute(
            "INSERT INTO console_conversation_memories VALUES (?, ?, ?, ?)",
            (str(uuid4()), CONVERSATION, boundary, int(active)),
        )

    # The two ConsoleTraceRepository reads the proof makes.
    def get_owner(self, _cursor, owner_id):
        return (
            SimpleNamespace(conversation_id=CONVERSATION) if owner_id == OWNER else None
        )

    def get_semantic_revision(self, _cursor, revision_id):
        return next(
            (
                item
                for item in self.revisions.values()
                if item.revision_id == revision_id
            ),
            None,
        )

    def build(self, retained, incoming, current):
        return build_rewritten_tail_witness(
            self.db.cursor(),
            self,
            owner_id=OWNER,
            retained=tuple(retained),
            incoming=tuple(incoming),
            current_turn_id=current,
        )


@pytest.fixture
def branch():
    tree = _Branch()
    tree.q1 = tree.message(None)
    tree.a1 = tree.message(tree.q1)
    tree.q2 = tree.message(tree.a1)  # the original turn
    tree.q2_edit = tree.message(tree.a1)  # Edit & resend's sibling
    tree.a2_edit = tree.message(tree.q2_edit)
    tree.q3 = tree.message(tree.a2_edit)  # the current user
    yield tree
    tree.db.close()


def _edited_tail(branch):
    return [branch.saved(m) for m in (branch.q2_edit, branch.a2_edit, branch.q3)]


def _system_row():
    return ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RENDERED_SYSTEM,
        FrozenTracePolicy(str(uuid4()), "credentials-v1", False, None),
    )


def test_an_edited_branch_hangs_off_the_last_retained_row(branch):
    retained = [branch.saved(branch.q1), branch.saved(branch.a1)]

    witness = branch.build(retained, _edited_tail(branch), branch.q3)

    assert witness == RewrittenTailWitness(
        tuple(_edited_tail(branch)[:-1]),
        branch.revisions[branch.q3].revision_id,
        anchor_message_id=branch.a1,
    )


def test_a_branch_from_the_conversation_root_needs_no_retained_saved_row(branch):
    incoming = [branch.saved(m) for m in (branch.q1, branch.a1, branch.q2_edit)]
    current = branch.message(branch.q2_edit)
    incoming.append(branch.saved(current))

    witness = branch.build([_system_row()], incoming, current)

    assert witness is not None and witness.anchor_message_id is None


@pytest.mark.parametrize(
    "case",
    [
        "gap_in_chain",
        "row_from_other_conversation",
        "deleted_row",
        "provider_artifact_in_history",
        "last_row_is_not_the_current_user",
        "only_the_current_user",
        "unattached_owner",
        "anchor_is_not_the_retained_row",
        "root_after_a_retained_row",
    ],
)
def test_an_unproven_edited_branch_stays_refused(branch, case):
    """Each case breaks one link of an otherwise admitted edited branch."""

    retained = [branch.saved(branch.q1), branch.saved(branch.a1)]
    incoming = _edited_tail(branch)
    current = branch.q3
    owner = OWNER
    if case == "gap_in_chain":
        incoming = [branch.saved(branch.q2_edit), branch.saved(branch.q3)]
    elif case == "row_from_other_conversation":
        foreign = branch.message(branch.a1, conversation=str(uuid4()))
        incoming = [branch.saved(foreign), *incoming[1:]]
    elif case == "deleted_row":
        branch.db.execute(
            "UPDATE messages SET deleted = 1 WHERE id = ?", (branch.a2_edit,)
        )
    elif case == "provider_artifact_in_history":
        incoming = [
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.CONTEXT_SUMMARY,
                FrozenTracePolicy(str(uuid4()), "credentials-v1", False, None),
            ),
            *incoming,
        ]
    elif case == "last_row_is_not_the_current_user":
        current = branch.q2_edit
    elif case == "only_the_current_user":
        # Even with its parent retained, a lone user is the one-item
        # replacement the planner already owns, never a rewritten tail.
        retained = [
            *retained,
            branch.saved(branch.q2_edit),
            branch.saved(branch.a2_edit),
        ]
        incoming = incoming[-1:]
    elif case == "unattached_owner":
        owner = str(uuid4())
    elif case == "anchor_is_not_the_retained_row":
        retained = [branch.saved(branch.q1)]
    elif case == "root_after_a_retained_row":
        retained = [branch.saved(branch.q2)]
        incoming = [branch.saved(m) for m in (branch.q1, branch.a1, branch.q2_edit)]
        current = branch.message(branch.q2_edit)
        incoming.append(branch.saved(current))

    witness = build_rewritten_tail_witness(
        branch.db.cursor(),
        branch,
        owner_id=owner,
        retained=tuple(retained),
        incoming=tuple(incoming),
        current_turn_id=current,
    )

    assert witness is None


def test_a_compacted_history_hangs_off_the_active_memory_boundary(branch):
    branch.remember(branch.a1)
    incoming = _edited_tail(branch)

    witness = branch.build([_system_row()], incoming, branch.q3)

    assert witness == RewrittenTailWitness(
        tuple(incoming[:-1]),
        branch.revisions[branch.q3].revision_id,
        anchor_message_id=branch.a1,
        memory_anchor=True,
    )


def test_a_memory_anchor_may_follow_retained_saved_rows(branch):
    """The shape a manual range memory sends: kept turns, a gap, later turns.

    Pinned as today's behaviour: no link from the retained tail to the
    memory's range is proven, and the anchor is any active memory of the
    conversation. Tightening both is TASK-37601 (ADR-097, 2026-10-29).
    """

    branch.remember(branch.a1)

    witness = branch.build([branch.saved(branch.q1)], _edited_tail(branch), branch.q3)

    assert witness is not None
    assert witness.memory_anchor is True
    assert witness.anchor_message_id == branch.a1


@pytest.mark.parametrize(
    "case", ["no_memory", "inactive_memory", "memory_boundary_elsewhere"]
)
def test_an_unproven_compacted_history_stays_refused(branch, case):
    """Dropped rows are admitted only behind an active memory's boundary."""

    if case == "inactive_memory":
        branch.remember(branch.a1, active=False)
    elif case == "memory_boundary_elsewhere":
        branch.remember(branch.q1)

    assert branch.build([_system_row()], _edited_tail(branch), branch.q3) is None


@pytest.mark.parametrize(
    ("anchor", "change"),
    [
        ("branch", "none"),
        ("branch", "row_deleted"),
        ("branch", "reparented"),
        ("branch", "descriptors"),
        ("branch", "other_user"),
        ("memory", "none"),
        ("memory", "memory_reset"),
        ("memory", "row_deleted"),
    ],
)
def test_final_binding_rechecks_the_proof(branch, anchor, change):
    if anchor == "memory":
        branch.remember(branch.a1)
        retained = []
    else:
        retained = [branch.saved(branch.q1), branch.saved(branch.a1)]
    incoming = _edited_tail(branch)
    witness = branch.build(retained, incoming, branch.q3)
    assert witness is not None and witness.memory_anchor is (anchor == "memory")
    current = branch.q3
    if change == "memory_reset":
        branch.db.execute("UPDATE console_conversation_memories SET active = 0")
    elif change == "row_deleted":
        branch.db.execute(
            "UPDATE messages SET deleted = 1 WHERE id = ?", (branch.q2_edit,)
        )
    elif change == "reparented":
        branch.db.execute(
            "UPDATE messages SET parent_message_id = ? WHERE id = ?",
            (branch.q1, branch.q2_edit),
        )
    elif change == "descriptors":
        incoming = incoming[1:]
    elif change == "other_user":
        current = branch.q2

    def recheck():
        validate_rewritten_tail(
            branch.db.cursor(),
            branch,
            owner_id=OWNER,
            witness=witness,
            descriptors=tuple(incoming),
            current_turn_id=current,
        )

    if change == "none":
        recheck()
    else:
        with pytest.raises(ValueError, match="^rewritten_tail_unavailable$"):
            recheck()


@pytest.mark.parametrize("field_name", ["completed_tool_turn", "rewritten_tail"])
def test_each_witness_field_accepts_only_its_own_witness_type(branch, field_name):
    """A witness in the other kind's field never passes as that kind."""

    incoming = tuple(_edited_tail(branch))
    rewritten = RewrittenTailWitness(
        incoming[:-1], branch.revisions[branch.q3].revision_id
    )
    tool_turn = CompletedToolTurnWitness(
        str(uuid4()),
        str(uuid4()),
        incoming[1].revision_id,
        incoming[2].revision_id,
        incoming[0].revision_id,
    )
    # Both witnesses match these exact descriptors, so only the type differs.
    assert rewritten.matches_descriptors(incoming)
    assert tool_turn.matches_descriptors(incoming)
    misplaced = {field_name: rewritten if field_name != "rewritten_tail" else tool_turn}
    replacement_range = VerifiedSurfaceReplacementRange(
        str(uuid4()), str(uuid4()), str(uuid4()), 0, 1
    )
    items = tuple(
        VerifiedSurfaceDeltaItem("messages_payload", ordinal, descriptor)
        for ordinal, descriptor in enumerate(incoming)
    )

    with pytest.raises(ValueError, match="^surface_delta_shape$"):
        SurfaceDeltaAdmission(
            OWNER,
            str(uuid4()),
            str(uuid4()),
            "fresh",
            "preparation",
            incoming,
            replacement_range=replacement_range,
            **misplaced,
        )
    with pytest.raises(ValueError, match="^surface_delta_shape$"):
        VerifiedSurfaceDelta(
            OWNER,
            str(uuid4()),
            str(uuid4()),
            "fresh",
            "preparation",
            items=items[1:],
            replacement=VerifiedSurfaceReplacement(
                str(uuid4()), str(uuid4()), str(uuid4()), 0, 1, 0, items[0]
            ),
            **misplaced,
        )
