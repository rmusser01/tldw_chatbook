from __future__ import annotations

from dataclasses import replace
import hashlib
import json

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
    BranchMemoryCommit,
    ConsoleContextRepository,
    ConsoleMemoryRecord,
    ConsoleMemoryScopeRecord,
    ConsoleMemorySelectionRecord,
    MemorySelectionFence,
    MemoryCoverageKind,
    MemoryOriginKind,
    MemorySelectionKind,
    PersistedLineageFenceRow,
    persisted_attachment_digest,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


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


def _digest_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prefix_digest(
    root_id: str,
    leaf_id: str,
    *,
    through_leaf: bool,
    root_content: str = "root content",
    root_variant_id: str | None = None,
    root_variant_index: int | None = None,
    root_attachments: tuple[str, ...] = (),
) -> str:
    rows = [
        {
            "message_id": root_id,
            "version": 1,
            "role": "user",
            "content": root_content,
            "selected_variant_id": root_variant_id,
            "selected_variant_index": root_variant_index,
            "attachment_digests": list(root_attachments),
        },
        {
            "message_id": leaf_id,
            "version": 1,
            "role": "assistant",
            "content": "leaf content",
            "selected_variant_id": None,
            "selected_variant_index": None,
            "attachment_digests": [],
        },
    ]
    return _digest_json(rows if through_leaf else rows[:1])


def _repository_database(_tmp_path, name: str):
    db = CharactersRAGDB(":memory:", client_id=name)
    conversation_id = db.add_conversation({"title": name})
    root_id = db.add_message(
        {
            "id": f"{name}-root",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "root content",
        }
    )
    leaf_id = db.add_message(
        {
            "id": f"{name}-leaf",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "leaf content",
            "parent_message_id": root_id,
        }
    )
    assert root_id is not None
    assert leaf_id is not None
    assert db.set_conversation_active_cursor(
        conversation_id,
        active_leaf_message_id=leaf_id,
        before_message_id=None,
    )
    return db, ConsoleContextRepository(db), conversation_id, root_id, leaf_id


def _persisted_lineage(
    root_id: str,
    leaf_id: str,
    *,
    root_attachments: tuple[str, ...] = (),
) -> tuple[PersistedLineageFenceRow, ...]:
    return (
        PersistedLineageFenceRow(
            message_id=root_id,
            parent_message_id=None,
            version=1,
            deleted=False,
            content_digest=_digest_json("root content"),
            selected_variant_id=None,
            selected_variant_index=None,
            attachment_digests=root_attachments,
        ),
        PersistedLineageFenceRow(
            message_id=leaf_id,
            parent_message_id=root_id,
            version=1,
            deleted=False,
            content_digest=_digest_json("leaf content"),
            selected_variant_id=None,
            selected_variant_index=None,
            attachment_digests=(),
        ),
    )


def _no_memory_fence() -> MemorySelectionFence:
    return MemorySelectionFence(
        effective_kind="raw",
        legacy_boundary_message_id=None,
        legacy_summary_digest=None,
        selection_sequence=None,
        selection_id=None,
        selection_revision=None,
        memory_id=None,
        memory_revision=None,
    )


def _no_head_fence() -> MemorySelectionFence:
    return replace(_no_memory_fence(), effective_kind="no_head")


def _branch_commit(
    conversation_id: str,
    root_id: str,
    leaf_id: str,
    *,
    memory_id: str,
    selection_id: str,
    expected_effective: MemorySelectionFence | None = None,
    expected_branch_head: MemorySelectionFence | None = None,
    durable_lineage: tuple[PersistedLineageFenceRow, ...] | None = None,
    origin: MemoryOriginKind = MemoryOriginKind.AUTOMATIC,
) -> BranchMemoryCommit:
    manual = origin is MemoryOriginKind.MANUAL_REWIND
    boundary_id = leaf_id if manual else root_id
    memory = ConsoleMemoryRecord(
        memory_id=memory_id,
        conversation_id=conversation_id,
        boundary_message_id=boundary_id,
        captured_leaf_message_id=leaf_id,
        lineage_json=json.dumps([root_id, leaf_id]),
        summary_text=f"Summary for {memory_id}.",
        provider="openai",
        model="gpt-test",
        prompt_id="console.rewind_summarize",
        prompt_revision=1,
        prompt_digest="p" * 64,
        selected_units_json="[]",
        summarized_prefix_digest=_prefix_digest(
            root_id, leaf_id, through_leaf=manual
        ),
        input_tokens=20,
        output_tokens=5,
        before_tokens=100,
        after_tokens=50,
        created_at="2026-08-28T00:00:00Z",
    )
    scope = ConsoleMemoryScopeRecord(
        memory_id=memory_id,
        conversation_id=conversation_id,
        coverage_kind=(
            MemoryCoverageKind.RANGE if manual else MemoryCoverageKind.PREFIX
        ),
        origin_kind=origin,
        selection_anchor_message_id=root_id if manual else None,
    )
    selection = ConsoleMemorySelectionRecord(
        sequence=1,
        selection_id=selection_id,
        conversation_id=conversation_id,
        activation_message_id=leaf_id,
        selected_memory_id=memory_id,
        event_kind=MemorySelectionKind.SELECT,
        # The transaction, not a caller-supplied bit, owns inheritance/forcing.
        suppresses_legacy=False,
        created_at="2026-08-28T00:00:01Z",
    )
    return BranchMemoryCommit(
        memory=memory,
        scope=scope,
        selection=selection,
        expected_effective=expected_effective or _no_memory_fence(),
        expected_branch_head=expected_branch_head or _no_head_fence(),
        expected_cursor=(leaf_id, None),
        durable_lineage=durable_lineage or _persisted_lineage(root_id, leaf_id),
    )


def _generated_fences(
    selection: ConsoleMemorySelectionRecord,
    *,
    memory_revision: int = 1,
    effective_kind: str = "generated_prefix",
) -> tuple[MemorySelectionFence, MemorySelectionFence]:
    exact = dict(
        legacy_boundary_message_id=None,
        legacy_summary_digest=None,
        selection_sequence=selection.sequence,
        selection_id=selection.selection_id,
        selection_revision=selection.revision,
        memory_id=selection.selected_memory_id,
        memory_revision=memory_revision,
    )
    return (
        MemorySelectionFence(effective_kind=effective_kind, **exact),
        MemorySelectionFence(effective_kind="select", **exact),
    )


def _derived_row_snapshot(db: CharactersRAGDB, conversation_id: str):
    connection = db.get_connection()
    memories = connection.execute(
        "SELECT id, revision, active FROM console_conversation_memories "
        "WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    scopes = connection.execute(
        "SELECT memory_id, coverage_kind, origin_kind "
        "FROM console_conversation_memory_scopes WHERE conversation_id = ? "
        "ORDER BY memory_id",
        (conversation_id,),
    ).fetchall()
    selections = connection.execute(
        "SELECT selection_id, revision, active FROM "
        "console_conversation_memory_selections WHERE conversation_id = ? "
        "ORDER BY sequence",
        (conversation_id,),
    ).fetchall()
    return (
        tuple(tuple(row) for row in memories),
        tuple(tuple(row) for row in scopes),
        tuple(tuple(row) for row in selections),
    )


def test_two_jobs_admitted_without_memory_allow_only_one_atomic_commit(tmp_path) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, "no-memory-race"
    )
    first = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="memory-first",
        selection_id="selection-first",
    )
    second = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="memory-second",
        selection_id="selection-second",
    )

    assert repository.commit_memory_selection_if_current(first)
    assert not repository.commit_memory_selection_if_current(second)

    assert _derived_row_snapshot(db, conversation_id) == (
        (("memory-first", 1, 1),),
        (("memory-first", "prefix", "automatic"),),
        (("selection-first", 1, 1),),
    )
    assert not repository.list_active_memory_selections(
        conversation_id
    )[0].suppresses_legacy


def test_manual_selection_forces_legacy_suppression_without_clearing_legacy(
    tmp_path,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, "manual-suppression"
    )
    db.set_conversation_context_summary(conversation_id, "Legacy facts.", root_id)
    legacy = MemorySelectionFence(
        effective_kind="legacy_prefix",
        legacy_boundary_message_id=root_id,
        legacy_summary_digest=_digest_json("Legacy facts."),
        selection_sequence=None,
        selection_id=None,
        selection_revision=None,
        memory_id=None,
        memory_revision=None,
    )
    commit = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="manual-memory",
        selection_id="manual-selection",
        expected_effective=legacy,
        origin=MemoryOriginKind.MANUAL_REWIND,
    )

    assert repository.commit_memory_selection_if_current(commit)

    assert db.get_conversation_context_summary(conversation_id) == (
        "Legacy facts.",
        root_id,
    )
    persisted = repository.list_active_memory_selections(conversation_id)
    assert len(persisted) == 1
    assert persisted[0].suppresses_legacy


def test_automatic_replacement_inherits_suppression_and_ignores_hidden_legacy(
    tmp_path,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, "automatic-inheritance"
    )
    db.set_conversation_context_summary(conversation_id, "Legacy facts.", root_id)
    legacy = MemorySelectionFence(
        effective_kind="legacy_prefix",
        legacy_boundary_message_id=root_id,
        legacy_summary_digest=_digest_json("Legacy facts."),
        selection_sequence=None,
        selection_id=None,
        selection_revision=None,
        memory_id=None,
        memory_revision=None,
    )
    assert repository.commit_memory_selection_if_current(
        _branch_commit(
            conversation_id,
            root_id,
            leaf_id,
            memory_id="suppressing-memory",
            selection_id="suppressing-selection",
            expected_effective=legacy,
            origin=MemoryOriginKind.MANUAL_REWIND,
        )
    )
    current = repository.list_active_memory_selections(conversation_id)[0]
    effective, head = _generated_fences(
        current, effective_kind="generated_range"
    )
    replacement = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="automatic-replacement-memory",
        selection_id="automatic-replacement-selection",
        expected_effective=effective,
        expected_branch_head=head,
    )

    # This compatibility state is hidden by the applicable suppressing head, so
    # it did not participate in admission and is deliberately outside the CAS.
    db.set_conversation_context_summary(
        conversation_id, "Changed hidden legacy facts.", root_id
    )
    assert repository.commit_memory_selection_if_current(replacement)

    selections = repository.list_active_memory_selections(conversation_id)
    assert [selection.selection_id for selection in selections] == [
        "automatic-replacement-selection",
        "suppressing-selection",
    ]
    assert selections[0].suppresses_legacy


@pytest.mark.parametrize("changed_revision", ["selection", "memory"])
def test_exact_generated_head_revision_mismatch_preserves_prior_rows(
    tmp_path,
    changed_revision: str,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, f"generated-revision-{changed_revision}"
    )
    assert repository.commit_memory_selection_if_current(
        _branch_commit(
            conversation_id,
            root_id,
            leaf_id,
            memory_id="existing-memory",
            selection_id="existing-selection",
        )
    )
    selection = repository.list_active_memory_selections(conversation_id)[0]
    effective, head = _generated_fences(selection)
    stale = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="stale-memory",
        selection_id="stale-selection",
        expected_effective=effective,
        expected_branch_head=head,
    )
    table = (
        "console_conversation_memory_selections"
        if changed_revision == "selection"
        else "console_conversation_memories"
    )
    db.get_connection().execute(
        f"UPDATE {table} SET revision = revision + 1 WHERE conversation_id = ?",
        (conversation_id,),
    )
    db.get_connection().commit()
    before = _derived_row_snapshot(db, conversation_id)

    assert not repository.commit_memory_selection_if_current(stale)
    assert _derived_row_snapshot(db, conversation_id) == before


@pytest.mark.parametrize("legacy_mutation", ["boundary", "summary"])
def test_exact_legacy_boundary_and_summary_digest_mismatch_is_stale(
    tmp_path,
    legacy_mutation: str,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, f"legacy-{legacy_mutation}"
    )
    db.set_conversation_context_summary(conversation_id, "Legacy facts.", root_id)
    legacy = MemorySelectionFence(
        effective_kind="legacy_prefix",
        legacy_boundary_message_id=root_id,
        legacy_summary_digest=_digest_json("Legacy facts."),
        selection_sequence=None,
        selection_id=None,
        selection_revision=None,
        memory_id=None,
        memory_revision=None,
    )
    stale = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="legacy-stale-memory",
        selection_id="legacy-stale-selection",
        expected_effective=legacy,
    )
    if legacy_mutation == "boundary":
        db.set_conversation_context_summary(
            conversation_id, "Legacy facts.", leaf_id
        )
    else:
        db.set_conversation_context_summary(
            conversation_id, "Changed legacy facts.", root_id
        )

    assert not repository.commit_memory_selection_if_current(stale)
    assert _derived_row_snapshot(db, conversation_id) == ((), (), ())


@pytest.mark.parametrize("lineage_mutation", ["active_leaf", "before_marker", "parent"])
def test_changed_persisted_cursor_leaf_or_parent_rejects_without_writes(
    tmp_path,
    lineage_mutation: str,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, f"cursor-{lineage_mutation}"
    )
    stale = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="cursor-stale-memory",
        selection_id="cursor-stale-selection",
    )
    if lineage_mutation == "active_leaf":
        db.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id=root_id,
            before_message_id=None,
        )
    elif lineage_mutation == "before_marker":
        db.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id=leaf_id,
            before_message_id=root_id,
        )
    else:
        db.get_connection().execute(
            "UPDATE messages SET parent_message_id = NULL WHERE id = ?",
            (leaf_id,),
        )
        db.get_connection().commit()

    assert not repository.commit_memory_selection_if_current(stale)
    assert _derived_row_snapshot(db, conversation_id) == ((), (), ())


def test_commit_rejects_a_new_memory_with_the_wrong_persisted_prefix_digest(
    tmp_path,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, "new-memory-prefix-digest"
    )
    commit = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="wrong-digest-memory",
        selection_id="wrong-digest-selection",
    )
    commit = replace(
        commit,
        memory=replace(commit.memory, summarized_prefix_digest="f" * 64),
    )

    assert not repository.commit_memory_selection_if_current(commit)
    assert _derived_row_snapshot(db, conversation_id) == ((), (), ())


def test_selected_variant_image_is_fenced_from_the_persisted_variant_row(
    tmp_path,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, "selected-variant-image"
    )
    variant_id = db.create_message_variant(
        root_id, "selected variant content", is_selected=True
    )
    assert variant_id is not None
    connection = db.get_connection()
    connection.execute(
        "UPDATE messages SET image_data = ?, image_mime_type = ? WHERE id = ?",
        (b"variant image", "image/png", variant_id),
    )
    connection.commit()
    image_digest = persisted_attachment_digest(
        position=0,
        mime_type="image/png",
        display_name="variant.png",
        data=b"variant image",
    )
    lineage = _persisted_lineage(root_id, leaf_id)
    lineage = (
        replace(
            lineage[0],
            content_digest=_digest_json("selected variant content"),
            selected_variant_id=variant_id,
            selected_variant_index=1,
            attachment_digests=(image_digest,),
        ),
        lineage[1],
    )
    commit = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="variant-image-memory",
        selection_id="variant-image-selection",
        durable_lineage=lineage,
    )
    commit = replace(
        commit,
        memory=replace(
            commit.memory,
            summarized_prefix_digest=_prefix_digest(
                root_id,
                leaf_id,
                through_leaf=False,
                root_content="selected variant content",
                root_variant_id=variant_id,
                root_variant_index=1,
                root_attachments=(image_digest,),
            ),
        ),
    )

    assert repository.commit_memory_selection_if_current(commit)


def test_non_fork_position_zero_runtime_label_normalizes_to_persisted_facts(
    tmp_path,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, "ordinary-position-zero-label"
    )
    connection = db.get_connection()
    connection.execute(
        "UPDATE messages SET image_data = ?, image_mime_type = ? WHERE id = ?",
        (b"ordinary image", "image/png", root_id),
    )
    connection.commit()
    assert connection.execute(
        "SELECT metadata_json FROM messages WHERE id = ?", (root_id,)
    ).fetchone()["metadata_json"] is None

    runtime_digest = persisted_attachment_digest(
        position=0,
        mime_type="image/png",
        display_name="facts.png",
        data=b"ordinary image",
    )
    lineage = _persisted_lineage(
        root_id,
        leaf_id,
        root_attachments=(runtime_digest,),
    )
    commit = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="ordinary-image-memory",
        selection_id="ordinary-image-selection",
        durable_lineage=lineage,
    )
    commit = replace(
        commit,
        memory=replace(
            commit.memory,
            summarized_prefix_digest=_prefix_digest(
                root_id,
                leaf_id,
                through_leaf=False,
                root_attachments=(runtime_digest,),
            ),
        ),
    )

    assert repository.commit_memory_selection_if_current(commit)


@pytest.mark.parametrize(
    "durable_mutation",
    ["content", "version", "deletion", "variant", "attachment"],
)
def test_changed_durable_message_fact_rejects_without_partial_insert(
    tmp_path,
    durable_mutation: str,
) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, f"durable-{durable_mutation}"
    )
    durable_lineage = None
    if durable_mutation == "attachment":
        db.set_message_attachments(
            root_id,
            [
                {
                    "position": 1,
                    "data": b"original attachment",
                    "mime_type": "text/plain",
                    "display_name": "facts.txt",
                }
            ],
        )
        durable_lineage = _persisted_lineage(
            root_id,
            leaf_id,
            root_attachments=(
                persisted_attachment_digest(
                    position=1,
                    mime_type="text/plain",
                    display_name="facts.txt",
                    data=b"original attachment",
                ),
            ),
        )
    stale = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="durable-stale-memory",
        selection_id="durable-stale-selection",
        durable_lineage=durable_lineage,
    )
    if durable_mutation == "content":
        db.get_connection().execute(
            "UPDATE messages SET content = 'changed content' WHERE id = ?",
            (root_id,),
        )
        db.get_connection().commit()
    elif durable_mutation == "version":
        db.get_connection().execute(
            "UPDATE messages SET version = 2 WHERE id = ?",
            (root_id,),
        )
        db.get_connection().commit()
    elif durable_mutation == "deletion":
        db.get_connection().execute(
            "UPDATE messages SET deleted = 1 WHERE id = ?",
            (root_id,),
        )
        db.get_connection().commit()
    elif durable_mutation == "variant":
        assert db.create_message_variant(
            root_id, "selected variant", is_selected=True
        )
    else:
        db.set_message_attachments(
            root_id,
            [
                {
                    "position": 1,
                    "data": b"changed attachment",
                    "mime_type": "text/plain",
                    "display_name": "facts.txt",
                }
            ],
        )

    assert not repository.commit_memory_selection_if_current(stale)
    assert _derived_row_snapshot(db, conversation_id) == ((), (), ())


def test_unrelated_newer_sibling_event_does_not_stale_captured_branch(tmp_path) -> None:
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, "sibling-event"
    )
    sibling_id = db.add_message(
        {
            "id": "sibling-event-other-leaf",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "sibling",
            "parent_message_id": root_id,
        }
    )
    assert sibling_id is not None
    repository.insert_memory_selection(
        ConsoleMemorySelectionRecord(
            sequence=1,
            selection_id="sibling-reset",
            conversation_id=conversation_id,
            activation_message_id=sibling_id,
            selected_memory_id=None,
            event_kind=MemorySelectionKind.RESET,
            suppresses_legacy=True,
            created_at="2026-08-28T00:00:00Z",
        )
    )
    commit = _branch_commit(
        conversation_id,
        root_id,
        leaf_id,
        memory_id="branch-memory",
        selection_id="branch-selection",
    )

    assert repository.commit_memory_selection_if_current(commit)

    selections = repository.list_active_memory_selections(conversation_id)
    assert [item.selection_id for item in selections] == [
        "branch-selection",
        "sibling-reset",
    ]
    assert not selections[0].suppresses_legacy


def _reset_selection(
    conversation_id: str,
    leaf_id: str,
    *,
    selection_id: str,
) -> ConsoleMemorySelectionRecord:
    return ConsoleMemorySelectionRecord(
        sequence=1,
        selection_id=selection_id,
        conversation_id=conversation_id,
        activation_message_id=leaf_id,
        selected_memory_id=None,
        event_kind=MemorySelectionKind.RESET,
        suppresses_legacy=True,
        created_at="2026-08-28T00:00:02Z",
    )


def _manual_memory_with_legacy(tmp_path, name: str):
    db, repository, conversation_id, root_id, leaf_id = _repository_database(
        tmp_path, name
    )
    db.set_conversation_context_summary(conversation_id, "Legacy facts.", root_id)
    legacy = MemorySelectionFence(
        effective_kind="legacy_prefix",
        legacy_boundary_message_id=root_id,
        legacy_summary_digest=_digest_json("Legacy facts."),
        selection_sequence=None,
        selection_id=None,
        selection_revision=None,
        memory_id=None,
        memory_revision=None,
    )
    assert repository.commit_memory_selection_if_current(
        _branch_commit(
            conversation_id,
            root_id,
            leaf_id,
            memory_id="manual-current-memory",
            selection_id="manual-current-selection",
            expected_effective=legacy,
            origin=MemoryOriginKind.MANUAL_REWIND,
        )
    )
    current = repository.list_active_memory_selections(conversation_id)[0]
    effective, head = _generated_fences(
        current, effective_kind="generated_range"
    )
    return (
        db,
        repository,
        conversation_id,
        root_id,
        leaf_id,
        effective,
        head,
    )


def test_current_reset_appends_exact_tombstone_without_mutating_memory_or_legacy(
    tmp_path,
) -> None:
    (
        db,
        repository,
        conversation_id,
        root_id,
        leaf_id,
        effective,
        head,
    ) = _manual_memory_with_legacy(tmp_path, "current-reset")
    before_memory = tuple(
        db.get_connection()
        .execute(
            "SELECT id, active, revision, reset_at FROM "
            "console_conversation_memories WHERE conversation_id = ?",
            (conversation_id,),
        )
        .fetchone()
    )

    token = repository.append_current_branch_reset_if_current(
        _reset_selection(
            conversation_id, leaf_id, selection_id="current-reset-tombstone"
        ),
        expected_effective=effective,
        expected_branch_head=head,
        expected_cursor=(leaf_id, None),
        durable_lineage=_persisted_lineage(root_id, leaf_id),
    )

    assert token == ("current-reset-tombstone", 1)
    assert tuple(
        db.get_connection()
        .execute(
            "SELECT id, active, revision, reset_at FROM "
            "console_conversation_memories WHERE conversation_id = ?",
            (conversation_id,),
        )
        .fetchone()
    ) == before_memory
    assert db.get_conversation_context_summary(conversation_id) == (
        "Legacy facts.",
        root_id,
    )
    selections = repository.list_active_memory_selections(conversation_id)
    assert [item.selection_id for item in selections] == [
        "current-reset-tombstone",
        "manual-current-selection",
    ]
    assert selections[0].event_kind is MemorySelectionKind.RESET
    assert selections[0].suppresses_legacy


def test_undo_deactivates_only_exact_current_tombstone_at_expected_revision(
    tmp_path,
) -> None:
    (
        db,
        repository,
        conversation_id,
        root_id,
        leaf_id,
        effective,
        head,
    ) = _manual_memory_with_legacy(tmp_path, "current-undo")
    token = repository.append_current_branch_reset_if_current(
        _reset_selection(conversation_id, leaf_id, selection_id="undo-tombstone"),
        expected_effective=effective,
        expected_branch_head=head,
        expected_cursor=(leaf_id, None),
        durable_lineage=_persisted_lineage(root_id, leaf_id),
    )
    assert token == ("undo-tombstone", 1)

    assert not repository.undo_current_branch_reset_if_current(
        conversation_id,
        selection_id=token[0],
        expected_revision=2,
    )
    assert repository.undo_current_branch_reset_if_current(
        conversation_id,
        selection_id=token[0],
        expected_revision=token[1],
    )

    rows = db.get_connection().execute(
        "SELECT selection_id, active, revision FROM "
        "console_conversation_memory_selections WHERE conversation_id = ? "
        "ORDER BY sequence",
        (conversation_id,),
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("manual-current-selection", 1, 1),
        ("undo-tombstone", 0, 2),
    ]
    memory = db.get_connection().execute(
        "SELECT active, revision FROM console_conversation_memories "
        "WHERE id = 'manual-current-memory'"
    ).fetchone()
    assert tuple(memory) == (1, 1)


@pytest.mark.parametrize("later_event_kind", ["select", "reset"])
def test_later_applicable_select_or_reset_expires_undo_token(
    tmp_path,
    later_event_kind: str,
) -> None:
    (
        db,
        repository,
        conversation_id,
        root_id,
        leaf_id,
        effective,
        head,
    ) = _manual_memory_with_legacy(tmp_path, f"undo-expiry-{later_event_kind}")
    token = repository.append_current_branch_reset_if_current(
        _reset_selection(
            conversation_id, leaf_id, selection_id="expiring-tombstone"
        ),
        expected_effective=effective,
        expected_branch_head=head,
        expected_cursor=(leaf_id, None),
        durable_lineage=_persisted_lineage(root_id, leaf_id),
    )
    assert token == ("expiring-tombstone", 1)
    repository.insert_memory_selection(
        (
            ConsoleMemorySelectionRecord(
                sequence=1,
                selection_id="later-select",
                conversation_id=conversation_id,
                activation_message_id=leaf_id,
                selected_memory_id="manual-current-memory",
                event_kind=MemorySelectionKind.SELECT,
                suppresses_legacy=True,
                created_at="2026-08-28T00:00:03Z",
            )
            if later_event_kind == "select"
            else _reset_selection(
                conversation_id, leaf_id, selection_id="later-reset"
            )
        )
    )

    assert not repository.undo_current_branch_reset_if_current(
        conversation_id,
        selection_id=token[0],
        expected_revision=token[1],
    )
    expired = db.get_connection().execute(
        "SELECT active, revision FROM console_conversation_memory_selections "
        "WHERE selection_id = ?",
        (token[0],),
    ).fetchone()
    assert tuple(expired) == (1, 1)
