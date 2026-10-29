"""The trace planner keeps the rewritten tail narrow and bounded.

TASK-33621.29 / TASK-33621.7 review (ADR-097 amendment 2026-10-27).
``Tests/Chat/test_console_trace_rewritten_tail.py`` pins the proof itself over
a fake ledger. These cases drive ``prepare_current_surface_delta`` over a real
database and trace ledger, so each planner gate around the proof is exercised:
removing a gate turns its case red. Each case records earlier sends with one
warm service and plans the next send with a fresh one, as after a restart.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from Tests.Chat.test_console_trace_service import (
    SurfaceReplacementPlan,
    _available_bundle,
    _continuation_value,
    _owned_segment,
    _persist,
    _policy,
    _provenance,
)
from tldw_chatbook.Chat import console_send_diagnostics
from tldw_chatbook.Chat.console_trace_final_values import CompletedToolTurnWitness
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class _Chat:
    """One conversation with a trace owner and the surface its sends recorded."""

    def __init__(self, db: CharactersRAGDB, refusals: list) -> None:
        self.db = db
        self.repository = ConsoleTraceRepository()
        self.owner_id, self.segment_id = _owned_segment(db, self.repository)
        with db.transaction() as cursor:
            owner = self.repository.get_owner(cursor, self.owner_id)
            self.conversation_id = owner.conversation_id
        self.policy = _policy()
        self.refusals = refusals
        # Earlier sends share one warm service, as one app session does.
        self.recorder = ConsoleTraceService(self.repository)
        self.head = None

    def add(self, parent, label, role="user"):
        message_id = self.db.add_message(
            {
                "conversation_id": self.conversation_id,
                "sender": role,
                "content": label,
                "parent_message_id": None if parent is None else parent.message_id,
            }
        )
        with self.db.transaction() as cursor:
            (revision_id,) = cursor.execute(
                "SELECT revision_id FROM console_trace_semantic_revisions "
                "WHERE source_message_id = ? "
                "ORDER BY revision_sequence DESC LIMIT 1",
                (message_id,),
            ).fetchone()
        return SimpleNamespace(
            message_id=message_id,
            descriptor=SavedRevisionTraceProvenance(revision_id),
            value={"role": role, "content": label},
        )

    def chain(self, count, parent=None, start=0):
        rows = []
        for index in range(start, start + count):
            parent = self.add(
                parent, f"m{index}", "user" if index % 2 == 0 else "assistant"
            )
            rows.append(parent)
        return rows

    def remember(self, boundary) -> None:
        """Commit an active conversation memory, as Compact now does."""

        with self.db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO console_conversation_memories("
                "id, conversation_id, boundary_message_id, summary_text) "
                "VALUES (?, ?, ?, 'memory')",
                (str(uuid4()), self.conversation_id, boundary.message_id),
            )

    def record(self, rows, *, continuations=(), replacement=None, append_from=0):
        """Persist one captured call's surface, as an earlier send did."""

        provenance = replace(
            _provenance(tuple(row.descriptor for row in rows)),
            continuations=tuple(descriptor for descriptor, _ in continuations),
        )
        bundle = _available_bundle(
            provenance,
            messages=[row.value for row in rows],
            continuations=[value for _, value in continuations] or None,
        )
        with self.db.transaction() as cursor:
            self.repository.ensure_policy(cursor, self.policy)
            persisted = _persist(
                self.recorder,
                cursor,
                owner_id=self.owner_id,
                segment_id=self.segment_id,
                provenance=provenance,
                bundle=bundle,
                replacement=replacement,
                append_from_index=append_from,
                previous_surface_head_id=self.head,
            )
        self.head = persisted.surface_head_id
        return persisted

    def prepare(
        self,
        rows,
        current,
        *,
        route=ConsoleRequestRoute.FRESH,
        policy=True,
        completed_tool_turn=None,
    ):
        """Plan the next captured send with a fresh (restarted) service."""

        with self.db.transaction() as cursor:
            admission, _boundary = ConsoleTraceService(
                self.repository
            ).prepare_current_surface_delta(
                cursor,
                owner_id=self.owner_id,
                segment_id=self.segment_id,
                route_identity=route.value,
                preparation_identity=str(uuid4()),
                provenance=_provenance(tuple(row.descriptor for row in rows)),
                values=tuple(row.value for row in rows),
                completed_tool_turn=completed_tool_turn,
                current_turn_id=current.message_id,
                current_policy_id=self.policy.policy_id if policy else None,
            )
        return admission

    def refusal(self, rows, current, **kwargs):
        """Return the recorded refusal kinds of a send the planner refuses."""

        with pytest.raises(ValueError, match="^unsupported_surface_change$"):
            self.prepare(rows, current, **kwargs)
        return [fields.get("surface_refusal_kind") for fields in self.refusals]


@pytest.fixture
def chat_factory(tmp_path, monkeypatch):
    """Fresh chats in one database; each records its own refusals."""

    database = CharactersRAGDB(str(tmp_path / "planner.sqlite"), "task33621")
    refusals: list = []
    monkeypatch.setattr(
        console_send_diagnostics,
        "record_send_stage",
        lambda *_args, **fields: refusals.append(fields),
    )

    def make() -> _Chat:
        refusals.clear()
        return _Chat(database, refusals)

    yield make
    database.close_connection()


def _edited_chat(chat: _Chat, *, record_current: bool = False, continuation=None):
    """send q1, send q2, Edit & resend q2, then plan a send of ``next``."""

    q1, a1, q2, a2 = chat.chain(4)
    q2_edit = chat.add(a1, "edited")
    a2_edit = chat.add(q2_edit, "answer", "assistant")
    current = chat.add(a2_edit, "next")
    chat.record(
        [q1, a1, q2, a2, *([current] if record_current else [])],
        continuations=() if continuation is None else (continuation,),
    )
    return [q1, a1, q2_edit, a2_edit, current], current


def test_planner_gates_keep_the_rewritten_tail_narrow(chat_factory):
    """Only a FRESH/AGENT_FIRST, policy-bound, messages-only whole tail.

    One database holds every case, each in its own chat. The admitted edit is
    the control: each refused case changes exactly one gate's input.
    """

    chat = chat_factory()
    incoming, current = _edited_chat(chat)
    admission = chat.prepare(incoming, current)
    assert admission.rewritten_tail is not None
    assert admission.completed_tool_turn is None
    assert admission.descriptors == tuple(row.descriptor for row in incoming[2:])

    # Edit & resend after a tool run still offers that run's witness. Its range
    # does not match the edited branch, so the branch is proven on its own and
    # no tool-turn claim is recorded.
    chat = chat_factory()
    incoming, current = _edited_chat(chat)
    witness = CompletedToolTurnWitness(
        str(uuid4()), str(uuid4()), str(uuid4()), current.descriptor.revision_id
    )
    assert witness.descriptor_count != len(incoming[2:])
    admission = chat.prepare(incoming, current, completed_tool_turn=witness)
    assert admission.completed_tool_turn is None
    assert admission.rewritten_tail is not None

    refused = {}
    chat = chat_factory()
    incoming, current = _edited_chat(chat)
    refused["route"] = chat.refusal(
        incoming, current, route=ConsoleRequestRoute.REGENERATE
    )
    chat = chat_factory()
    incoming, current = _edited_chat(chat)
    refused["policy"] = chat.refusal(incoming, current, policy=False)
    # The current user is already recorded after the replaced rows, so the
    # change would leave a retained suffix behind it.
    chat = chat_factory()
    incoming, current = _edited_chat(chat, record_current=True)
    refused["suffix"] = chat.refusal(incoming, current)
    # A recorded provider continuation is never part of the replacement.
    chat = chat_factory()
    continuation = (
        ProviderArtifactTraceProvenance(
            TraceProvenanceSource.CONTINUATION, chat.policy
        ),
        _continuation_value("recorded"),
    )
    incoming, current = _edited_chat(chat, continuation=continuation)
    refused["continuation"] = chat.refusal(incoming, current)

    assert refused == {
        "route": ["count_or_span"],
        "policy": ["count_or_span"],
        "suffix": ["count_or_span"],
        "continuation": ["count_or_span"],
    }


def test_a_compacted_history_is_bounded_by_the_replacement_range(chat_factory):
    """ADR-097 keeps the 256-node range: a longer history stays refused.

    Compact now on a chat whose recorded history holds more than 256 rows
    still stops at the card (TASK-33621.29 review). Lifting that needs a
    durable-format decision, tracked separately.
    """

    outcomes = {}
    for recorded in (256, 257):
        chat = chat_factory()
        rows = chat.chain(recorded)
        chat.record(rows)
        chat.remember(rows[-3])
        current = chat.add(rows[-1], "after compaction")
        incoming = [rows[-2], rows[-1], current]
        if recorded == 256:
            admission = chat.prepare(incoming, current)
            assert admission.rewritten_tail.memory_anchor is True
            outcomes[recorded] = (
                admission.replacement_range.start_sequence,
                admission.replacement_range.end_sequence,
            )
        else:
            outcomes[recorded] = chat.refusal(incoming, current)

    assert outcomes == {256: (0, 255), 257: ["count_or_span"]}


def test_a_rewritten_tail_is_bounded_by_its_sequence_span_not_its_row_count(
    chat_factory,
):
    """Dead sequences inside the range count toward the 256-node bound.

    An earlier replacement leaves its replaced rows' sequences inside the next
    range. Here 131 live rows span 261 sequences, which one replacement cannot
    store. Admitting the rewrite reached the over-span fallback: it recorded
    an omission in place of the first new row and kept every old row in the
    surface, so the trace no longer matched what the provider received.
    """

    chat = chat_factory()
    rows = chat.chain(140)
    first = chat.record(rows)
    edit = chat.add(rows[0], "edited", "assistant")
    surface = [rows[0], edit, *rows[131:]]
    chat.record(
        surface,
        replacement=SurfaceReplacementPlan(
            predecessor_head_id=first.surface_head_id,
            start_node_id=first.appended_nodes[1].node_id,
            end_node_id=first.appended_nodes[130].node_id,
            start_sequence=1,
            end_sequence=130,
            replacement_index=1,
        ),
    )
    later = chat.chain(120, parent=rows[-1], start=140)
    chat.record([*surface, *later], append_from=len(surface))
    chat.remember(later[-3])
    current = chat.add(later[-1], "after compaction")

    assert chat.refusal([later[-2], later[-1], current], current) == [
        "count_or_span"
    ]
