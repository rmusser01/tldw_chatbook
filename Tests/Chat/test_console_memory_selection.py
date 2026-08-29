from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_context_compaction import (
    NO_LEGACY_MEMORY,
    DurableMessageSnapshot,
    EffectiveMemoryKind,
    LegacyMemorySnapshot,
    prefix_digest,
    select_effective_memory,
)
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleMemoryRecord,
    ConsoleMemoryScopeRecord,
    ConsoleMemorySelectionRecord,
    MemoryCoverageKind,
    MemoryOriginKind,
    MemorySelectionKind,
)


def _message(message_id: str, content: str) -> DurableMessageSnapshot:
    return DurableMessageSnapshot(
        message_id=message_id,
        version=1,
        role="user" if message_id.startswith("u") else "assistant",
        content=content,
    )


ACTIVE = (
    _message("u1", "one"),
    _message("a1", "two"),
    _message("u2", "three"),
    _message("a2", "four"),
)


def _memory(
    memory_id: str,
    *,
    boundary: str = "a1",
    digest_through: int = 2,
) -> ConsoleMemoryRecord:
    return ConsoleMemoryRecord(
        memory_id=memory_id,
        conversation_id="conversation-1",
        boundary_message_id=boundary,
        captured_leaf_message_id="a2",
        lineage_json='["u1", "a1", "u2", "a2"]',
        summary_text=f"Summary for {memory_id}.",
        provider="openai",
        model="gpt-test",
        prompt_id="console.rewind_summarize",
        prompt_revision=1,
        prompt_digest="p" * 64,
        selected_units_json="[]",
        summarized_prefix_digest=prefix_digest(ACTIVE[:digest_through]),
        input_tokens=40,
        output_tokens=10,
        before_tokens=100,
        after_tokens=50,
        created_at="2026-08-28T00:00:00Z",
    )


def _scope(
    memory_id: str,
    *,
    coverage: MemoryCoverageKind = MemoryCoverageKind.PREFIX,
    origin: MemoryOriginKind = MemoryOriginKind.AUTOMATIC,
    anchor: str | None = None,
) -> ConsoleMemoryScopeRecord:
    return ConsoleMemoryScopeRecord(
        memory_id=memory_id,
        conversation_id="conversation-1",
        coverage_kind=coverage,
        origin_kind=origin,
        selection_anchor_message_id=anchor,
    )


def _selection(
    sequence: int,
    memory_id: str | None,
    *,
    activation: str = "a2",
    suppresses_legacy: bool = False,
) -> ConsoleMemorySelectionRecord:
    kind = (
        MemorySelectionKind.SELECT
        if memory_id is not None
        else MemorySelectionKind.RESET
    )
    return ConsoleMemorySelectionRecord(
        sequence=sequence,
        selection_id=f"selection-{sequence}",
        conversation_id="conversation-1",
        activation_message_id=activation,
        selected_memory_id=memory_id,
        event_kind=kind,
        suppresses_legacy=suppresses_legacy,
        created_at="2026-08-28T00:00:00Z",
    )


def _select(
    *,
    memories: tuple[ConsoleMemoryRecord, ...],
    scopes: tuple[ConsoleMemoryScopeRecord, ...],
    selections: tuple[ConsoleMemorySelectionRecord, ...],
    active=ACTIVE,
    legacy=NO_LEGACY_MEMORY,
):
    return select_effective_memory(
        "conversation-1",
        active,
        memories=memories,
        scopes=scopes,
        selection_candidates=selections,
        legacy=legacy,
    )


def test_generated_prefix_requires_a_live_boundary_and_matching_digest() -> None:
    valid = _memory("prefix")
    head = _selection(1, "prefix")

    assert _select(
        memories=(valid,), scopes=(_scope("prefix"),), selections=(head,)
    ).kind is EffectiveMemoryKind.GENERATED_PREFIX
    assert _select(
        memories=(replace(valid, boundary_message_id="sibling"),),
        scopes=(_scope("prefix"),),
        selections=(head,),
    ).kind is EffectiveMemoryKind.RAW
    assert _select(
        memories=(replace(valid, summarized_prefix_digest="0" * 64),),
        scopes=(_scope("prefix"),),
        selections=(head,),
    ).kind is EffectiveMemoryKind.RAW


def test_generated_range_requires_both_ordered_anchors_and_prefix_digest() -> None:
    memory = _memory("range", boundary="a2", digest_through=4)
    head = _selection(1, "range", suppresses_legacy=True)
    valid_scope = _scope(
        "range",
        coverage=MemoryCoverageKind.RANGE,
        origin=MemoryOriginKind.MANUAL_REWIND,
        anchor="u2",
    )

    result = _select(
        memories=(memory,), scopes=(valid_scope,), selections=(head,)
    )
    assert result.kind is EffectiveMemoryKind.GENERATED_RANGE
    assert result.memory == memory
    assert result.branch_head == head

    for bad_anchor in ("sibling", "a2"):
        result = _select(
            memories=(memory,),
            scopes=(replace(valid_scope, selection_anchor_message_id=bad_anchor),),
            selections=(head,),
        )
        assert result.kind is EffectiveMemoryKind.RAW
        assert result.branch_head == head


def test_newest_branch_valid_event_uses_database_sequence_and_skips_siblings() -> None:
    older = _memory("older")
    newest = _memory("newest")
    sibling = _selection(30, None, activation="sibling", suppresses_legacy=True)
    newest_on_branch = _selection(20, "newest")
    older_on_branch = _selection(10, "older")

    result = _select(
        memories=(older, newest),
        scopes=(_scope("older"), _scope("newest")),
        selections=(older_on_branch, sibling, newest_on_branch),
    )

    assert result.kind is EffectiveMemoryKind.GENERATED_PREFIX
    assert result.memory == newest
    assert result.branch_head == newest_on_branch


def test_reset_is_a_terminal_branch_head() -> None:
    older = _memory("older")
    reset = _selection(2, None, suppresses_legacy=True)

    result = _select(
        memories=(older,),
        scopes=(_scope("older"),),
        selections=(_selection(1, "older"), reset),
    )

    assert result.kind is EffectiveMemoryKind.RAW
    assert result.memory is None
    assert result.branch_head == reset


def test_invalid_selected_memory_fails_raw_without_older_generated_fallback() -> None:
    invalid_newest = replace(
        _memory("invalid-newest"), summarized_prefix_digest="0" * 64
    )
    older = _memory("older")
    newest_head = _selection(2, "invalid-newest")

    result = _select(
        memories=(invalid_newest, older),
        scopes=(_scope("invalid-newest"), _scope("older")),
        selections=(_selection(1, "older"), newest_head),
    )

    assert result.kind is EffectiveMemoryKind.RAW
    assert result.memory is None
    assert result.branch_head == newest_head


@pytest.mark.parametrize("with_head", [False, True])
def test_valid_legacy_wins_without_a_suppressing_branch_head(with_head: bool) -> None:
    generated = _memory("generated")
    head = _selection(1, "generated")
    legacy = LegacyMemorySnapshot(
        conversation_id="conversation-1",
        summary_text="Legacy facts.",
        boundary_message_id="a1",
    )

    result = _select(
        memories=(generated,),
        scopes=(_scope("generated"),),
        selections=(head,) if with_head else (),
        legacy=legacy,
    )

    assert result.kind is EffectiveMemoryKind.LEGACY_PREFIX
    assert result.legacy == legacy
    assert result.branch_head == (head if with_head else None)


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (_selection(1, "generated", suppresses_legacy=True), EffectiveMemoryKind.GENERATED_PREFIX),
        (_selection(1, None, suppresses_legacy=True), EffectiveMemoryKind.RAW),
    ],
)
def test_suppressing_manual_or_reset_head_overrides_valid_legacy(
    head: ConsoleMemorySelectionRecord,
    expected: EffectiveMemoryKind,
) -> None:
    generated = _memory("generated")
    legacy = LegacyMemorySnapshot(
        conversation_id="conversation-1",
        summary_text="Legacy facts.",
        boundary_message_id="a1",
    )

    result = _select(
        memories=(generated,),
        scopes=(_scope("generated"),),
        selections=(head,),
        legacy=legacy,
    )

    assert result.kind is expected
    assert result.branch_head == head


@pytest.mark.parametrize(
    "legacy",
    [
        LegacyMemorySnapshot(
            conversation_id="conversation-2",
            summary_text="Foreign facts.",
            boundary_message_id="a1",
        ),
        LegacyMemorySnapshot(
            conversation_id="conversation-1",
            summary_text="Sibling facts.",
            boundary_message_id="sibling",
        ),
    ],
)
def test_invalid_or_off_lineage_legacy_falls_through_to_generated_head(
    legacy: LegacyMemorySnapshot,
) -> None:
    generated = _memory("generated")
    head = _selection(1, "generated")

    result = _select(
        memories=(generated,),
        scopes=(_scope("generated"),),
        selections=(head,),
        legacy=legacy,
    )

    assert result.kind is EffectiveMemoryKind.GENERATED_PREFIX
    assert result.memory == generated


def test_scope_and_selection_records_reject_contradictory_combinations() -> None:
    with pytest.raises(ValueError, match="automatic"):
        _scope(
            "bad-auto-range",
            coverage=MemoryCoverageKind.RANGE,
            origin=MemoryOriginKind.AUTOMATIC,
        )
    with pytest.raises(ValueError, match="selection anchor"):
        _scope(
            "bad-manual",
            origin=MemoryOriginKind.MANUAL_REWIND,
            anchor=None,
        )
    with pytest.raises(ValueError, match="selected memory"):
        ConsoleMemorySelectionRecord(
            sequence=1,
            selection_id="bad-select",
            conversation_id="conversation-1",
            activation_message_id="a2",
            selected_memory_id=None,
            event_kind=MemorySelectionKind.SELECT,
            suppresses_legacy=True,
            created_at="2026-08-28T00:00:00Z",
        )
    with pytest.raises(ValueError, match="positive integer"):
        replace(_selection(1, "generated"), sequence=0)
    with pytest.raises(ValueError, match="reset event must suppress legacy"):
        _selection(1, None, suppresses_legacy=False)


def test_manual_scope_with_non_suppressing_head_fails_raw() -> None:
    manual = _memory("manual", boundary="a2", digest_through=4)
    result = _select(
        memories=(manual,),
        scopes=(
            _scope(
                "manual",
                coverage=MemoryCoverageKind.RANGE,
                origin=MemoryOriginKind.MANUAL_REWIND,
                anchor="u2",
            ),
        ),
        selections=(_selection(1, "manual", suppresses_legacy=False),),
    )

    assert result.kind is EffectiveMemoryKind.RAW
