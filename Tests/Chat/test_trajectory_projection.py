"""Unit tests for the pure trajectory projection (``Chat/trajectory.py``).

The projection is deliberately stdlib-pure: these tests feed it plain
dicts / local dataclass stand-ins (NOT ``TrajectoryRowRead`` from the DB
module) to prove it duck-types its inputs instead of importing the DB.
"""

from __future__ import annotations

from dataclasses import dataclass

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    derive_trajectory,
)


# ---------------------------------------------------------------------------
# Input stand-ins (duck-typed mirrors of the DB / store shapes)
# ---------------------------------------------------------------------------


def msg(
    mid: str,
    sender: str,
    *,
    content: str = "content",
    ts: object = "2026-08-14 10:00:00",
    parent: str | None = None,
    deleted: bool = False,
) -> dict:
    """A ``messages``-table-shaped row (mapping form)."""
    return {
        "id": mid,
        "sender": sender,
        "content": content,
        "timestamp": ts,
        "parent_message_id": parent,
        "deleted": deleted,
    }


@dataclass(frozen=True)
class TrajRow:
    """Duck-typed stand-in for ``TrajectoryRowRead`` (proves no DB import)."""

    message_id: str
    conversation_id: str = "conv-1"
    turn_id: str = "t1"
    seq: int = 0
    event_kind: str = "assistant"
    step_started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None
    model: str | None = None
    provider: str | None = None
    payload_json: str | None = None


@dataclass(frozen=True)
class VariantLike:
    """One variant content (``ConsoleVariant`` stand-in)."""

    content: str


@dataclass(frozen=True)
class VariantSetLike:
    """Duck-typed stand-in for ``ConsoleVariantSet``."""

    turn_id: str
    variants: tuple[VariantLike, ...]
    selected_index: int = 0


def compaction_record(
    *,
    started_at: str = "2026-08-14T10:00:30+00:00",
    finished_at: str | None = "2026-08-14T10:00:35+00:00",
    status: str = "succeeded",
    provider: str | None = "openai",
    model: str | None = "gpt-test",
    usage_json: str | None = None,
) -> dict:
    """A ``list_auxiliary_attempts``-shaped mapping."""
    return {
        "operation_id": "op-1",
        "conversation_id": "conv-1",
        "purpose": "conversation_compaction",
        "provider": provider,
        "model": model,
        "requested_output_cap": 2048,
        "estimated_input_tokens": 12345,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": 5000,
        "provider_usage_json": usage_json,
        "pricing_provenance_json": None,
    }


def linear_chain(*sends: tuple[str, str]) -> tuple[list[dict], str]:
    """Build a linear parent chain of ``(id, sender)`` message rows.

    Returns ``(rows, leaf_id)`` with each node parented on the previous one.
    """
    rows: list[dict] = []
    previous: str | None = None
    for index, (mid, sender) in enumerate(sends):
        rows.append(msg(mid, sender, ts=f"2026-08-14 10:00:0{index}", parent=previous))
        previous = mid
    return rows, previous


def record_kinds(snapshot: TrajectorySnapshot) -> list[list[str]]:
    return [[r.kind for r in turn.records] for turn in snapshot.turns]


# ---------------------------------------------------------------------------
# Grouping: a turn starts at each user record
# ---------------------------------------------------------------------------


def test_turn_starts_at_each_user_record() -> None:
    rows, leaf = linear_chain(
        ("u1", "user"), ("a1", "assistant"), ("u2", "user"), ("a2", "assistant")
    )
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("u2", turn_id="t2", seq=3, event_kind="user"),
        TrajRow("a2", turn_id="t2", seq=4, event_kind="assistant"),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    assert [t.turn_id for t in snapshot.turns] == ["t1", "t2"]
    assert record_kinds(snapshot) == [
        ["user", "assistant"],
        ["user", "assistant"],
    ]
    # The user record is the FIRST record of its turn.
    assert snapshot.turns[0].records[0].message_id == "u1"
    assert snapshot.turns[1].records[0].message_id == "u2"


# ---------------------------------------------------------------------------
# Tool nesting
# ---------------------------------------------------------------------------


def test_tool_records_nest_under_owning_assistant_at_depth_one() -> None:
    rows, leaf = linear_chain(("u1", "user"), ("a1", "assistant"))
    payload = '{"name": "fs_read", "args": null, "result": "file body"}'
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow(
            "a1", turn_id="t1", seq=3, event_kind="tool_call", payload_json=payload
        ),
        TrajRow(
            "a1", turn_id="t1", seq=4, event_kind="tool_result", payload_json=payload
        ),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    (turn,) = snapshot.turns
    assert [(r.kind, r.depth) for r in turn.records] == [
        ("user", 0),
        ("assistant", 0),
        ("tool_call", 1),
        ("tool_result", 1),
    ]
    tool_call = turn.records[2]
    assert tool_call.payload == {"name": "fs_read", "args": None, "result": "file body"}
    assert tool_call.message_id == "a1"  # keyed on the parent assistant row
    assert tool_call.usage is None  # tool records never carry message usage
    assert tool_call.content_preview == "fs_read -> file body"


def test_tool_records_ordered_by_seq_under_owner() -> None:
    rows, leaf = linear_chain(("u1", "user"), ("a1", "assistant"))
    # Two markers = four tool rows; input deliberately NOT seq-ordered.
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("a1", turn_id="t1", seq=6, event_kind="tool_result", payload_json='{"result": "b"}'),
        TrajRow("a1", turn_id="t1", seq=5, event_kind="tool_call", payload_json='{"result": "b"}'),
        TrajRow("a1", turn_id="t1", seq=3, event_kind="tool_call", payload_json='{"result": "a"}'),
        TrajRow("a1", turn_id="t1", seq=4, event_kind="tool_result", payload_json='{"result": "a"}'),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    (turn,) = snapshot.turns
    tool_records = [r for r in turn.records if r.depth == 1]
    assert [(r.kind, r.payload["result"]) for r in tool_records] == [
        ("tool_call", "a"),
        ("tool_result", "a"),
        ("tool_call", "b"),
        ("tool_result", "b"),
    ]


def test_orphaned_tool_rows_are_dropped() -> None:
    # Tool rows keyed on an assistant that was soft-deleted must vanish.
    rows, leaf = linear_chain(
        ("u1", "user"), ("a1", "assistant", ), ("u2", "user")
    )
    rows[1]["deleted"] = True
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("a1", turn_id="t1", seq=3, event_kind="tool_call", payload_json='{"result": "x"}'),
        TrajRow("a1", turn_id="t1", seq=4, event_kind="tool_result", payload_json='{"result": "x"}'),
        TrajRow("u2", turn_id="t2", seq=5, event_kind="user"),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    kinds = [r.kind for t in snapshot.turns for r in t.records]
    assert kinds == ["user", "user"]  # no assistant, no tool rows


# ---------------------------------------------------------------------------
# Timing: NULL stays None; append-time timing surfaces as-is
# ---------------------------------------------------------------------------


def test_null_timing_renders_none_fields() -> None:
    rows, leaf = linear_chain(("u1", "user"), ("a1", "assistant"))
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),  # all timing NULL
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    assistant = snapshot.turns[0].records[1]
    assert assistant.step_started_at is None
    assert assistant.first_token_at is None
    assert assistant.completed_at is None


def test_tool_append_time_timing_surfaces_as_is() -> None:
    # Marker-append rows carry zero-duration append-time stamps; the
    # projection must surface them verbatim -- never derive, never drop.
    rows, leaf = linear_chain(("u1", "user"), ("a1", "assistant"))
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user", step_started_at=10.0),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=2,
            event_kind="assistant",
            step_started_at=11.0,
            first_token_at=12.0,
            completed_at=13.0,
        ),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=3,
            event_kind="tool_result",
            payload_json='{"result": "x"}',
            step_started_at=99.5,
            completed_at=99.5,  # append-time zero duration
        ),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    records = snapshot.turns[0].records
    assert (records[1].step_started_at, records[1].first_token_at, records[1].completed_at) == (
        11.0,
        12.0,
        13.0,
    )
    tool = records[2]
    assert (tool.step_started_at, tool.first_token_at, tool.completed_at) == (
        99.5,
        None,
        99.5,
    )


# ---------------------------------------------------------------------------
# Ordering: seq breaks timestamp ties
# ---------------------------------------------------------------------------


def test_seq_breaks_timestamp_ties() -> None:
    rows = [
        msg("a1", "assistant", ts="2026-08-14 10:00:05", parent="u1"),
        msg("a2", "assistant", ts="2026-08-14 10:00:05", parent="a1"),
        msg("u1", "user", ts="2026-08-14 10:00:04"),
    ]
    # Same timestamp for a1/a2; seq says a2 (4) precedes a1 (5).
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=5, event_kind="assistant"),
        TrajRow("a2", turn_id="t1", seq=4, event_kind="assistant"),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id="a2")
    ids = [r.message_id for r in snapshot.turns[0].records]
    assert ids == ["u1", "a2", "a1"]


def test_records_carry_running_ledger_seq() -> None:
    rows, leaf = linear_chain(("u1", "user"), ("a1", "assistant"))
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("a1", turn_id="t1", seq=3, event_kind="tool_call", payload_json='{"result": "x"}'),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    records = [r for t in snapshot.turns for r in t.records]
    assert [r.seq for r in records] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Active path + soft deletion + variants
# ---------------------------------------------------------------------------


def test_soft_deleted_messages_excluded_but_chain_traversed_through() -> None:
    # a1 is soft-deleted MID-CHAIN: the walk must pass through it to reach
    # u1, or every ancestor of a deleted node would vanish too.
    rows, leaf = linear_chain(
        ("u1", "user"), ("a1", "assistant"), ("u2", "user"), ("a2", "assistant")
    )
    rows[1]["deleted"] = True
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("u2", turn_id="t2", seq=3, event_kind="user"),
        TrajRow("a2", turn_id="t2", seq=4, event_kind="assistant"),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    ids = [r.message_id for t in snapshot.turns for r in t.records]
    assert ids == ["u1", "u2", "a2"]


def test_mid_chain_gap_in_inputs_renders_all_not_partial_path() -> None:
    # a1 is ABSENT from inputs mid-chain -- what the real DB seam produces
    # for a hard-deleted row (get_messages_for_conversation filters
    # deleted=0), unlike the soft-deleted node above which stays in the
    # map. The walk cannot know which pre-gap messages are on the path,
    # so it must yield "unknown" and the snapshot must degrade to
    # render-all instead of silently rendering only post-gap messages.
    rows, leaf = linear_chain(
        ("u1", "user"), ("a1", "assistant"), ("u2", "user"), ("a2", "assistant")
    )
    rows = [row for row in rows if row["id"] != "a1"]
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("u2", turn_id="t2", seq=3, event_kind="user"),
        TrajRow("a2", turn_id="t2", seq=4, event_kind="assistant"),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    # Render-all: u1 still appears (a partial-path walk would drop it).
    ids = [r.message_id for t in snapshot.turns for r in t.records]
    assert ids == ["u1", "u2", "a2"]


def test_off_path_tree_siblings_surface_as_variants_not_rows() -> None:
    # u1 has two assistant children: a1 (active) and a1b (superseded fork).
    rows = [
        msg("u1", "user", ts="2026-08-14 10:00:00"),
        msg("a1", "assistant", content="active reply", ts="2026-08-14 10:00:01", parent="u1"),
        msg("a1b", "assistant", content="superseded fork", ts="2026-08-14 10:00:02", parent="u1"),
        msg("u2", "user", ts="2026-08-14 10:00:03", parent="a1"),
    ]
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("u2", turn_id="t2", seq=3, event_kind="user"),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id="u2")
    ids = [r.message_id for t in snapshot.turns for r in t.records]
    assert ids == ["u1", "a1", "u2"]  # a1b is NOT a row
    a1_record = snapshot.turns[0].records[1]
    assert a1_record.variants == ("superseded fork",)
    # The user record owns no variants here.
    assert snapshot.turns[0].records[0].variants == ()


def test_none_leaf_renders_all_undeleted_messages() -> None:
    rows = [
        msg("u1", "user", ts="2026-08-14 10:00:00"),
        msg("a1", "assistant", ts="2026-08-14 10:00:01", parent="u1"),
        msg("a1b", "assistant", content="fork", ts="2026-08-14 10:00:02", parent="u1"),
    ]
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("a1b", turn_id="t1b", seq=3, event_kind="assistant"),
    ]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=None)
    ids = [r.message_id for t in snapshot.turns for r in t.records]
    assert ids == ["u1", "a1", "a1b"]


def test_variant_set_superseded_contents_attach_to_assistant_record() -> None:
    rows, leaf = linear_chain(("u1", "user"), ("a1", "assistant"))
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
    ]
    variant_set = VariantSetLike(
        turn_id="t1",
        variants=(VariantLike("base reply"), VariantLike("regenerated reply")),
        selected_index=1,  # regenerated is current; base is superseded
    )
    snapshot = derive_trajectory(
        rows, {}, traj_rows, [variant_set], [], active_leaf_message_id=leaf
    )
    user_rec, assistant_rec = snapshot.turns[0].records
    assert assistant_rec.variants == ("base reply",)
    assert user_rec.variants == ()


# ---------------------------------------------------------------------------
# Compaction markers
# ---------------------------------------------------------------------------


def test_compaction_renders_between_turns_with_null_message_id() -> None:
    # Message timestamps (not sidecar timing) place the marker: turn t1
    # spans 10:00, turn t2 spans 11:00, compaction ran at 10:30.
    rows = [
        msg("u1", "user", ts="2026-08-14 10:00:00"),
        msg("a1", "assistant", ts="2026-08-14 10:00:01", parent="u1"),
        msg("u2", "user", ts="2026-08-14 11:00:00", parent="a1"),
        msg("a2", "assistant", ts="2026-08-14 11:00:01", parent="u2"),
    ]
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
        TrajRow("u2", turn_id="t2", seq=3, event_kind="user"),
        TrajRow("a2", turn_id="t2", seq=4, event_kind="assistant"),
    ]
    compactions = [
        compaction_record(
            started_at="2026-08-14T10:30:00+00:00",
            finished_at="2026-08-14T10:30:05+00:00",
            usage_json='{"uncached_input": 10, "output": 2, "provider": "openai"}',
        )
    ]
    snapshot = derive_trajectory(
        rows, {}, traj_rows, [], compactions, active_leaf_message_id="a2"
    )
    assert len(snapshot.turns) == 2
    # The marker trails turn t1 -- i.e. between the two turns.
    t1_kinds = [r.kind for r in snapshot.turns[0].records]
    assert t1_kinds == ["user", "assistant", "compaction"]
    marker = snapshot.turns[0].records[-1]
    assert isinstance(marker, TrajectoryRecord)
    assert marker.message_id is None
    assert marker.depth == 0
    assert marker.model == "gpt-test"
    assert marker.provider == "openai"
    assert marker.usage is not None
    assert marker.usage.uncached_input == 10
    assert marker.usage.output == 2
    assert "succeeded" in marker.content_preview
    assert snapshot.turns[1].records[0].kind == "user"


def test_compaction_before_first_turn_leads_the_ledger() -> None:
    rows = [
        msg("u1", "user", ts="2026-08-14 10:00:00"),
        msg("a1", "assistant", ts="2026-08-14 10:00:01", parent="u1"),
    ]
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
    ]
    compactions = [
        compaction_record(started_at="2026-08-13T09:00:00+00:00")  # before everything
    ]
    snapshot = derive_trajectory(
        rows, {}, traj_rows, [], compactions, active_leaf_message_id="a1"
    )
    (turn,) = snapshot.turns
    assert [r.kind for r in turn.records] == ["compaction", "user", "assistant"]


# ---------------------------------------------------------------------------
# Usage + preview
# ---------------------------------------------------------------------------


def test_usage_by_id_attaches_to_records() -> None:
    rows, leaf = linear_chain(("u1", "user"), ("a1", "assistant"))
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
    ]
    usage = ProviderUsage(uncached_input=7, cache_read=3, output=5, provider="anthropic")
    snapshot = derive_trajectory(
        rows, {"a1": usage}, traj_rows, [], [], active_leaf_message_id=leaf
    )
    user_rec, assistant_rec = snapshot.turns[0].records
    assert assistant_rec.usage is usage
    assert user_rec.usage is None


def test_content_preview_single_line_capped_at_120() -> None:
    rows, leaf = linear_chain(("u1", "user"))
    rows[0]["content"] = "line one\nline two " + "x" * 300
    traj_rows = [TrajRow("u1", turn_id="t1", seq=1, event_kind="user")]
    snapshot = derive_trajectory(rows, {}, traj_rows, [], [], active_leaf_message_id=leaf)
    preview = snapshot.turns[0].records[0].content_preview
    assert "\n" not in preview
    assert len(preview) == 120
    assert preview.startswith("line one line two")


# ---------------------------------------------------------------------------
# Legacy fallback (no sidecar rows)
# ---------------------------------------------------------------------------


def test_legacy_grouping_by_timestamp_adjacency() -> None:
    # Same-calendar-second reply joins the preceding user message's turn;
    # the later reply (next second) still joins the open turn.
    rows = [
        msg("a1", "assistant", content="fast reply", ts="2026-08-14 10:00:01", parent="u1"),
        msg("u1", "user", content="question", ts="2026-08-14 10:00:01"),
        msg("a1b", "assistant", content="slow followup", ts="2026-08-14 10:00:07", parent="a1"),
        msg("u2", "user", content="next question", ts="2026-08-14 10:00:20", parent="a1b"),
        msg("a2", "assistant", content="answer two", ts="2026-08-14 10:00:22", parent="u2"),
    ]
    snapshot = derive_trajectory(rows, {}, [], [], [], active_leaf_message_id="a2")
    assert [t.turn_id for t in snapshot.turns] == ["u1", "u2"]
    assert record_kinds(snapshot) == [
        ["user", "assistant", "assistant"],
        ["user", "assistant"],
    ]
    # No sidecar facts: timing and model stay None.
    assert all(
        r.step_started_at is None and r.model is None
        for t in snapshot.turns
        for r in t.records
    )


def test_legacy_assistant_first_opens_own_turn() -> None:
    rows = [
        msg("a0", "assistant", ts="2026-08-14 10:00:00"),
        msg("u1", "user", ts="2026-08-14 10:00:05", parent="a0"),
        msg("a1", "assistant", ts="2026-08-14 10:00:06", parent="u1"),
    ]
    snapshot = derive_trajectory(rows, {}, [], [], [], active_leaf_message_id="a1")
    assert [t.turn_id for t in snapshot.turns] == ["a0", "u1"]
    assert record_kinds(snapshot) == [["assistant"], ["user", "assistant"]]


def test_legacy_messages_take_usage_and_variants() -> None:
    rows = [
        msg("u1", "user", ts="2026-08-14 10:00:00"),
        msg("a1", "assistant", content="kept", ts="2026-08-14 10:00:01", parent="u1"),
        msg("a1b", "assistant", content="dropped", ts="2026-08-14 10:00:02", parent="u1"),
    ]
    usage = ProviderUsage(output=4)
    snapshot = derive_trajectory(
        rows, {"a1": usage}, [], [], [], active_leaf_message_id="a1"
    )
    assistant_rec = snapshot.turns[0].records[1]
    assert assistant_rec.usage is usage
    assert assistant_rec.variants == ("dropped",)


# ---------------------------------------------------------------------------
# ConsoleChatMessage inputs (in-memory turn ids)
# ---------------------------------------------------------------------------


def test_console_chat_message_inputs_group_by_turn_id() -> None:
    messages = [
        ConsoleChatMessage(role="user", content="hello", id="m1", turn_id="tA", persisted_message_id="p1"),
        ConsoleChatMessage(role="assistant", content="hi", id="m2", turn_id="tA", persisted_message_id="p2"),
        ConsoleChatMessage(role="user", content="again", id="m3", turn_id="tB", persisted_message_id="p3"),
    ]
    usage = ProviderUsage(output=9)
    snapshot = derive_trajectory(
        messages, {"p2": usage}, [], [], [], active_leaf_message_id=None
    )
    assert [t.turn_id for t in snapshot.turns] == ["tA", "tB"]
    assistant_rec = snapshot.turns[0].records[1]
    # Usage join happens on the PERSISTED id.
    assert assistant_rec.usage is usage
    assert assistant_rec.message_id == "p2"


# ---------------------------------------------------------------------------
# Empty + purity
# ---------------------------------------------------------------------------


def test_empty_conversation_yields_empty_snapshot() -> None:
    snapshot = derive_trajectory([], {}, [], [], [], active_leaf_message_id=None)
    assert snapshot.turns == ()


def test_module_is_pure_stdlib() -> None:
    import inspect
    import tldw_chatbook.Chat.trajectory as trajectory_module

    source = inspect.getsource(trajectory_module)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            assert "textual" not in stripped, f"forbidden import: {stripped}"
            assert "tldw_chatbook.DB" not in stripped, f"forbidden import: {stripped}"
            assert "ConsoleChatStore" not in stripped, f"forbidden import: {stripped}"
