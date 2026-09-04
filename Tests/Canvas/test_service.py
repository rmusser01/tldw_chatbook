"""Branch-aware service contract for durable Canvas revisions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from uuid import uuid4

import pytest

from tldw_chatbook.Canvas.limits import CanvasRepositoryLimits
from tldw_chatbook.Canvas.models import CanvasConflictResult, CanvasScope
from tldw_chatbook.Canvas.repository import CanvasRepository
from tldw_chatbook.Canvas.service import CanvasService, CanvasServiceError
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path: Path):
    database = CharactersRAGDB(
        tmp_path / "canvas-service.sqlite", client_id="canvas-service"
    )
    try:
        yield database
    finally:
        database.close_connection()


def _conversation(db: CharactersRAGDB, title: str = "Canvas chat") -> str:
    conversation_id = db.add_conversation({"title": title})
    assert conversation_id is not None
    return conversation_id


def _message(
    db: CharactersRAGDB,
    conversation_id: str,
    content: str,
    *,
    parent_message_id: str | None = None,
) -> str:
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "sender": "assistant",
            "role": "assistant",
            "content": content,
        }
    )
    assert message_id is not None
    return message_id


def _scope(
    conversation_id: str,
    *active_message_ids: str,
    selected_canvas_id: str | None = None,
    selected_revision_id: str | None = None,
    run_id: str = "run-current",
) -> CanvasScope:
    return CanvasScope(
        session_id="session-current",
        conversation_id=conversation_id,
        active_message_ids=tuple(active_message_ids),
        selected_canvas_id=selected_canvas_id,
        selected_revision_id=selected_revision_id,
        run_id=run_id,
    )


def _html(text: str) -> str:
    return (
        "<!doctype html><html><head><title>Canvas</title></head>"
        f"<body><main>{text}</main></body></html>"
    )


def _create_canvas(
    repository: CanvasRepository,
    conversation_id: str,
    origin_message_id: str,
    *,
    title: str,
    source: str,
    run_id: str,
    canvas_id: str | None = None,
    revision_id: str | None = None,
    created_at: str | None = None,
):
    return repository.create_canvas(
        conversation_id,
        title=title,
        source=source,
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=origin_message_id,
        origin_turn_id=run_id,
        canvas_id=canvas_id,
        revision_id=revision_id,
        created_at=created_at,
    )


def _append_revision(
    repository: CanvasRepository,
    conversation_id: str,
    canvas_id: str,
    parent_revision_id: str,
    origin_message_id: str,
    *,
    title: str,
    source: str,
    run_id: str,
    created_at: str | None = None,
):
    return repository.append_revision(
        conversation_id,
        canvas_id,
        parent_revision_id=parent_revision_id,
        title=title,
        source=source,
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=origin_message_id,
        origin_turn_id=run_id,
        created_at=created_at,
    )


def _revision_count(db: CharactersRAGDB, canvas_id: str | None = None) -> int:
    if canvas_id is None:
        row = (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM canvas_revisions")
            .fetchone()
        )
    else:
        row = (
            db.get_connection()
            .execute(
                "SELECT COUNT(*) FROM canvas_revisions WHERE canvas_id = ?",
                (canvas_id,),
            )
            .fetchone()
        )
    assert row is not None
    return int(row[0])


def test_list_resolves_only_revisions_on_the_supplied_active_message_branch(db) -> None:
    """Removing the origin-path filter must expose the newer sibling revision."""

    conversation_id = _conversation(db)
    root_message_id = _message(db, conversation_id, "root")
    left_message_id = _message(
        db,
        conversation_id,
        "left branch",
        parent_message_id=root_message_id,
    )
    right_message_id = _message(
        db,
        conversation_id,
        "right branch",
        parent_message_id=root_message_id,
    )
    repository = CanvasRepository(db)
    created = repository.create_canvas(
        conversation_id,
        title="Shared map",
        source="<main>root</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=root_message_id,
        origin_turn_id="run-root",
        created_at="2026-09-03T12:00:00+00:00",
    )
    left_revision = repository.append_revision(
        conversation_id,
        created.identity.canvas_id,
        parent_revision_id=created.revision.revision_id,
        title="Shared map — left",
        source="<main>left</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=left_message_id,
        origin_turn_id="run-left",
        created_at="2026-09-03T23:00:00+00:00",
    )
    sibling_revision = repository.append_revision(
        conversation_id,
        created.identity.canvas_id,
        parent_revision_id=created.revision.revision_id,
        title="Shared map — right",
        source="<main>right</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=right_message_id,
        origin_turn_id="run-right",
        created_at="2026-09-03T01:00:00+00:00",
    )
    right_only = repository.create_canvas(
        conversation_id,
        title="Right-only notes",
        source="<main>hidden</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=right_message_id,
        origin_turn_id="run-right-only",
    )

    service = CanvasService(db, repository=repository)
    items = service.list_canvases(
        _scope(conversation_id, root_message_id, left_message_id)
    )

    assert len(items) == 1
    assert items[0].canvas_id == created.identity.canvas_id
    assert items[0].revision_id == left_revision.revision_id
    assert items[0].title == "Shared map — left"
    assert items[0].origin.message_id == left_message_id
    assert not hasattr(items[0], "source")
    assert sibling_revision.revision_id not in repr(items)
    assert right_only.identity.canvas_id not in repr(items)

    switched = service.list_canvases(
        _scope(conversation_id, root_message_id, right_message_id)
    )
    switched_by_canvas = {item.canvas_id: item for item in switched}
    assert set(switched_by_canvas) == {
        created.identity.canvas_id,
        right_only.identity.canvas_id,
    }
    assert (
        switched_by_canvas[created.identity.canvas_id].revision_id
        == sibling_revision.revision_id
    )
    assert left_revision.revision_id not in repr(switched)


def test_scope_shape_is_exact_immutable_and_list_resolution_ignores_clocks(db) -> None:
    """Wall clocks or unstable insertion order must not replace path/sequence order."""

    assert [field.name for field in fields(CanvasScope)] == [
        "session_id",
        "conversation_id",
        "active_message_ids",
        "selected_canvas_id",
        "selected_revision_id",
        "run_id",
    ]
    conversation_id = _conversation(db)
    root_message_id = _message(db, conversation_id, "root")
    leaf_message_id = _message(
        db, conversation_id, "leaf", parent_message_id=root_message_id
    )
    scope = _scope(conversation_id, root_message_id, leaf_message_id)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        scope.run_id = "changed"  # type: ignore[misc]

    repository = CanvasRepository(db)
    branched = _create_canvas(
        repository,
        conversation_id,
        root_message_id,
        title="Branched",
        source=_html("root"),
        run_id="run-root",
        created_at="2026-09-03T12:00:00+00:00",
    )
    newer_clock = _append_revision(
        repository,
        conversation_id,
        branched.identity.canvas_id,
        branched.revision.revision_id,
        leaf_message_id,
        title="Branch A",
        source=_html("branch-a"),
        run_id="run-a",
        created_at="2026-09-03T23:59:00+00:00",
    )
    later_sequence = _append_revision(
        repository,
        conversation_id,
        branched.identity.canvas_id,
        branched.revision.revision_id,
        leaf_message_id,
        title="Branch B",
        source=_html("branch-b"),
        run_id="run-b",
        created_at="2026-09-03T00:01:00+00:00",
    )
    assert later_sequence.parent_revision_id == newer_clock.parent_revision_id

    low_canvas_id = "00000000-0000-0000-0000-000000000010"
    high_canvas_id = "00000000-0000-0000-0000-000000000020"
    for canvas_id, label in ((high_canvas_id, "high"), (low_canvas_id, "low")):
        _create_canvas(
            repository,
            conversation_id,
            leaf_message_id,
            title=label,
            source=_html(label),
            run_id=f"run-{label}",
            canvas_id=canvas_id,
            revision_id=str(uuid4()),
            created_at="2026-09-03T12:00:00+00:00",
        )

    service = CanvasService(db, repository=repository)
    first = service.list_canvases(scope)
    second = service.list_canvases(scope)

    assert first == second
    assert first[0].revision_id == later_sequence.revision_id
    assert [item.canvas_id for item in first[1:]] == [low_canvas_id, high_canvas_id]
    repository.set_reopen_hint(conversation_id, high_canvas_id)
    assert service.list_canvases(scope)[0].canvas_id == high_canvas_id


def test_historical_selection_controls_exact_read_rename_and_next_update_parent(
    db,
) -> None:
    """Resolving from the default instead of the captured selection breaks this graph."""

    conversation_id = _conversation(db)
    root_message_id = _message(db, conversation_id, "root")
    leaf_message_id = _message(
        db, conversation_id, "leaf", parent_message_id=root_message_id
    )
    repository = CanvasRepository(db)
    root = _create_canvas(
        repository,
        conversation_id,
        root_message_id,
        title="Original title",
        source=_html("root source"),
        run_id="run-root",
    )
    newer = _append_revision(
        repository,
        conversation_id,
        root.identity.canvas_id,
        root.revision.revision_id,
        leaf_message_id,
        title="Newer title",
        source=_html("newer source"),
        run_id="run-newer",
    )
    historical_scope = _scope(
        conversation_id,
        root_message_id,
        leaf_message_id,
        selected_canvas_id=root.identity.canvas_id,
        selected_revision_id=root.revision.revision_id,
        run_id="run-historical",
    )
    newer_scope = replace(
        historical_scope,
        selected_revision_id=newer.revision_id,
        run_id="run-newer-selection",
    )
    service = CanvasService(db, repository=repository)

    listed = service.list_canvases(historical_scope)[0]
    exact = service.read_canvas(historical_scope, root.identity.canvas_id)
    assert listed.revision_id == root.revision.revision_id
    assert listed.is_selected is True
    assert listed.is_historical_selection is True
    assert exact.revision.revision_id == root.revision.revision_id
    assert exact.source == _html("root source")
    assert service.read_canvas(
        newer_scope, root.identity.canvas_id
    ).revision.revision_id == (newer.revision_id)

    renamed = service.rename_canvas(
        historical_scope,
        root.identity.canvas_id,
        expected_parent_revision_id=root.revision.revision_id,
        title="Historical rename",
    )
    updated = service.update_canvas(
        historical_scope,
        root.identity.canvas_id,
        expected_parent_revision_id=root.revision.revision_id,
        source=_html("historical update"),
    )

    assert renamed.revision.parent_revision_id == root.revision.revision_id
    assert updated.revision.parent_revision_id == root.revision.revision_id
    assert renamed.revision.revision_id != updated.revision.revision_id
    renamed_exact = repository.read_revision(
        conversation_id, renamed.revision.revision_id
    )
    updated_exact = repository.read_revision(
        conversation_id, updated.revision.revision_id
    )
    assert renamed_exact.title == "Historical rename"
    assert renamed_exact.source == root.revision.source
    assert renamed_exact.actor_kind == "user_rename"
    assert updated_exact.title == root.revision.title
    assert updated_exact.source == _html("historical update")
    assert updated_exact.actor_kind == "assistant"
    assert not hasattr(renamed, "source")
    assert not hasattr(updated, "source")
    assert service.list_canvases(historical_scope)[0].revision_id == (
        root.revision.revision_id
    )


def test_invalid_selection_is_logically_cleared_without_existence_disclosure(
    db,
) -> None:
    """Foreign, deleted, mismatched, and sibling selections must all degrade alike."""

    conversation_id = _conversation(db)
    root_message_id = _message(db, conversation_id, "root")
    left_message_id = _message(
        db, conversation_id, "left", parent_message_id=root_message_id
    )
    right_message_id = _message(
        db, conversation_id, "right", parent_message_id=root_message_id
    )
    repository = CanvasRepository(db)
    main = _create_canvas(
        repository,
        conversation_id,
        root_message_id,
        title="Main",
        source=_html("root"),
        run_id="run-root",
    )
    current = _append_revision(
        repository,
        conversation_id,
        main.identity.canvas_id,
        main.revision.revision_id,
        left_message_id,
        title="Main left",
        source=_html("left"),
        run_id="run-left",
    )
    sibling = _append_revision(
        repository,
        conversation_id,
        main.identity.canvas_id,
        main.revision.revision_id,
        right_message_id,
        title="Main right",
        source=_html("right"),
        run_id="run-right",
    )
    second = _create_canvas(
        repository,
        conversation_id,
        left_message_id,
        title="Second",
        source=_html("second"),
        run_id="run-second",
    )
    deleted = _create_canvas(
        repository,
        conversation_id,
        left_message_id,
        title="Deleted",
        source=_html("deleted"),
        run_id="run-deleted",
    )
    repository.soft_delete_canvas(conversation_id, deleted.identity.canvas_id)
    foreign_conversation_id = _conversation(db, "foreign")
    foreign_message_id = _message(db, foreign_conversation_id, "foreign")
    foreign = _create_canvas(
        repository,
        foreign_conversation_id,
        foreign_message_id,
        title="Foreign",
        source=_html("foreign"),
        run_id="run-foreign",
    )
    base_scope = _scope(conversation_id, root_message_id, left_message_id)
    service = CanvasService(db, repository=repository)
    baseline = service.list_canvases(base_scope)

    invalid_selections = (
        (main.identity.canvas_id, sibling.revision_id),
        (foreign.identity.canvas_id, foreign.revision.revision_id),
        (deleted.identity.canvas_id, deleted.revision.revision_id),
        (main.identity.canvas_id, second.revision.revision_id),
    )
    for selected_canvas_id, selected_revision_id in invalid_selections:
        invalid_scope = replace(
            base_scope,
            selected_canvas_id=selected_canvas_id,
            selected_revision_id=selected_revision_id,
        )
        assert service.list_canvases(invalid_scope) == baseline
        read = service.read_canvas(invalid_scope, main.identity.canvas_id)
        assert read.revision.revision_id == current.revision_id

    right_only = _create_canvas(
        repository,
        conversation_id,
        right_message_id,
        title="Right only",
        source=_html("right only"),
        run_id="run-right-only",
    )
    failures: list[tuple[str, str]] = []
    for unavailable_canvas_id in (
        right_only.identity.canvas_id,
        foreign.identity.canvas_id,
        deleted.identity.canvas_id,
    ):
        with pytest.raises(CanvasServiceError) as unavailable:
            service.read_canvas(base_scope, unavailable_canvas_id)
        failures.append((unavailable.value.code, str(unavailable.value)))
    assert len(set(failures)) == 1
    assert failures[0][0] == "canvas_not_found"


def test_stale_update_and_rename_return_only_selected_current_metadata_without_calls(
    db, monkeypatch
) -> None:
    """A stale check after compile/read/write would disclose source or create a child."""

    conversation_id = _conversation(db)
    root_message_id = _message(db, conversation_id, "root")
    left_message_id = _message(
        db, conversation_id, "left", parent_message_id=root_message_id
    )
    right_message_id = _message(
        db, conversation_id, "right", parent_message_id=root_message_id
    )
    repository = CanvasRepository(db)
    created = _create_canvas(
        repository,
        conversation_id,
        root_message_id,
        title="Root title",
        source=_html("root private source sentinel"),
        run_id="run-root",
    )
    selected = _append_revision(
        repository,
        conversation_id,
        created.identity.canvas_id,
        created.revision.revision_id,
        left_message_id,
        title="Selected title",
        source=_html("selected private source sentinel"),
        run_id="run-left",
    )
    sibling = _append_revision(
        repository,
        conversation_id,
        created.identity.canvas_id,
        created.revision.revision_id,
        right_message_id,
        title="Sibling title sentinel",
        source=_html("sibling private source sentinel"),
        run_id="run-right",
    )
    scope = _scope(
        conversation_id,
        root_message_id,
        left_message_id,
        selected_canvas_id=created.identity.canvas_id,
        selected_revision_id=selected.revision_id,
    )
    compile_calls: list[str] = []
    source_read_calls: list[tuple[object, ...]] = []

    def forbidden_compile(source: str):
        compile_calls.append(source)
        raise AssertionError("stale update compiled")

    def forbidden_source_read(*args: object):
        source_read_calls.append(args)
        raise AssertionError("stale rename read source")

    monkeypatch.setattr(repository, "read_revision", forbidden_source_read)
    service = CanvasService(db, repository=repository, compiler=forbidden_compile)
    stale_parent = str(uuid4())
    before = _revision_count(db, created.identity.canvas_id)

    update_conflict = service.update_canvas(
        scope,
        created.identity.canvas_id,
        expected_parent_revision_id=stale_parent,
        source=_html("must never compile"),
    )
    rename_conflict = service.rename_canvas(
        scope,
        created.identity.canvas_id,
        expected_parent_revision_id=stale_parent,
        title="must never write",
    )
    with pytest.raises(CanvasServiceError) as malformed_parent:
        service.update_canvas(
            scope,
            created.identity.canvas_id,
            expected_parent_revision_id="not-a-revision-id",
            source=_html("must never compile either"),
        )

    assert isinstance(update_conflict, CanvasConflictResult)
    assert update_conflict == rename_conflict
    assert [field.name for field in fields(CanvasConflictResult)] == [
        "code",
        "canvas_id",
        "current_revision_id",
        "content_sha256",
        "title",
        "sequence",
        "origin",
    ]
    assert update_conflict.code == "stale_parent"
    assert update_conflict.current_revision_id == selected.revision_id
    assert update_conflict.title == "Selected title"
    assert update_conflict.origin.message_id == left_message_id
    assert not hasattr(update_conflict, "source")
    assert "private source sentinel" not in repr(update_conflict)
    assert sibling.revision_id not in repr(update_conflict)
    assert "Sibling title sentinel" not in repr(update_conflict)
    assert malformed_parent.value.code == "invalid_expected_parent"
    assert compile_calls == []
    assert source_read_calls == []
    assert _revision_count(db, created.identity.canvas_id) == before


def test_scope_validation_rejects_foreign_deleted_duplicate_and_non_durable_paths(
    db,
) -> None:
    """Accepting any same-conversation set would let sibling origins become authority."""

    conversation_id = _conversation(db)
    root_message_id = _message(db, conversation_id, "root")
    left_message_id = _message(
        db, conversation_id, "left", parent_message_id=root_message_id
    )
    right_message_id = _message(
        db, conversation_id, "right", parent_message_id=root_message_id
    )
    foreign_conversation_id = _conversation(db, "foreign")
    foreign_message_id = _message(db, foreign_conversation_id, "foreign")
    deleted_message_id = _message(
        db,
        conversation_id,
        "deleted",
        parent_message_id=root_message_id,
    )
    assert db.soft_delete_message(deleted_message_id, expected_version=1)
    deleted_conversation_id = _conversation(db, "deleted conversation")
    deleted_conversation_message_id = _message(
        db, deleted_conversation_id, "deleted owner"
    )
    assert db.soft_delete_conversation(deleted_conversation_id, expected_version=1)
    service = CanvasService(db)

    invalid_scopes = (
        _scope(conversation_id, root_message_id, root_message_id),
        _scope(conversation_id, root_message_id, foreign_message_id),
        _scope(conversation_id, root_message_id, deleted_message_id),
        _scope(deleted_conversation_id, deleted_conversation_message_id),
        _scope(conversation_id, root_message_id, "not-a-durable-message"),
        _scope(conversation_id, root_message_id, left_message_id, right_message_id),
        replace(
            _scope(conversation_id, root_message_id),
            selected_canvas_id=str(uuid4()),
            selected_revision_id=None,
        ),
        replace(_scope(conversation_id, root_message_id), run_id=""),
    )
    for invalid_scope in invalid_scopes:
        with pytest.raises(CanvasServiceError) as invalid:
            service.list_canvases(invalid_scope)
        assert invalid.value.code == "invalid_scope"

    assert service.list_canvases(_scope(conversation_id)) == ()
    compile_calls: list[str] = []
    empty_service = CanvasService(
        db,
        compiler=lambda source: compile_calls.append(source),
    )
    with pytest.raises(CanvasServiceError) as empty_mutation:
        empty_service.create_canvas(
            _scope(conversation_id), title="No leaf", source=_html("unused")
        )
    assert empty_mutation.value.code == "invalid_scope"
    assert compile_calls == []


def test_create_compiles_before_write_and_returns_exact_source_with_safe_diagnostics(
    db,
) -> None:
    """Skipping compilation or changing the origin from the captured leaf fails here."""

    conversation_id = _conversation(db)
    root_message_id = _message(db, conversation_id, "root")
    leaf_message_id = _message(
        db, conversation_id, "leaf", parent_message_id=root_message_id
    )
    source = "<main>created fragment</main>"
    scope = _scope(
        conversation_id,
        root_message_id,
        leaf_message_id,
        run_id="run-create",
    )
    repository = CanvasRepository(db)
    result = CanvasService(db, repository=repository).create_canvas(
        scope, title="Created title", source=source
    )

    assert result.source == source
    assert result.revision.title == "Created title"
    assert {issue.code for issue in result.compatibility_issues} == {"fragment-wrapped"}
    exact = repository.read_revision(conversation_id, result.revision.revision_id)
    assert exact.origin_message_id == leaf_message_id
    assert exact.origin_turn_id == "run-create"
    assert exact.actor_kind == "assistant"
    assert exact.runtime_profile == "canvas-v1"
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.source = "changed"  # type: ignore[misc]


def test_compiler_and_repository_refusals_are_bounded_and_make_no_write(db) -> None:
    """Refusal paths must not leak source or leave a partial Canvas graph."""

    conversation_id = _conversation(db)
    message_id = _message(db, conversation_id, "root")
    scope = _scope(conversation_id, message_id)
    sentinel = "source-secret-4d7c"
    repository = CanvasRepository(db)
    service = CanvasService(db, repository=repository)

    with pytest.raises(CanvasServiceError) as incompatible:
        service.create_canvas(
            scope,
            title="Invalid",
            source=f"<script src='https://example.invalid/{sentinel}'></script>",
        )
    assert incompatible.value.code == "document_incompatible"
    assert sentinel not in str(incompatible.value)
    assert sentinel not in repr(incompatible.value.issues)
    assert repository.list_identities(conversation_id) == ()

    limited_repository = CanvasRepository(
        db,
        limits=CanvasRepositoryLimits(
            max_canvases_per_conversation=1,
            max_revisions_per_canvas=1,
            max_source_bytes_per_conversation=16 * 1024,
            max_source_bytes_per_revision=8 * 1024,
            max_title_bytes=128,
            max_origin_turn_id_bytes=128,
        ),
    )
    limited_service = CanvasService(db, repository=limited_repository)
    created = limited_service.create_canvas(scope, title="First", source=_html("first"))
    before = _revision_count(db)
    with pytest.raises(CanvasServiceError) as canvas_quota:
        limited_service.create_canvas(scope, title="Second", source=_html("second"))
    assert canvas_quota.value.code == "canvas_count_limit"
    assert _revision_count(db) == before
    with pytest.raises(CanvasServiceError) as revision_quota:
        limited_service.update_canvas(
            replace(
                scope,
                selected_canvas_id=created.revision.canvas_id,
                selected_revision_id=created.revision.revision_id,
            ),
            created.revision.canvas_id,
            expected_parent_revision_id=created.revision.revision_id,
            source=_html("overflow"),
        )
    assert revision_quota.value.code == "revision_count_limit"
    assert _revision_count(db) == before


def test_list_uses_one_metadata_snapshot_and_only_exact_read_exposes_source(
    db, monkeypatch
) -> None:
    """Per-Canvas metadata reads would regress into N+1 queries and source exposure."""

    conversation_id = _conversation(db)
    message_id = _message(db, conversation_id, "root")
    repository = CanvasRepository(db)
    first = _create_canvas(
        repository,
        conversation_id,
        message_id,
        title="First",
        source=_html("first private"),
        run_id="run-first",
    )
    _create_canvas(
        repository,
        conversation_id,
        message_id,
        title="Second",
        source=_html("second private"),
        run_id="run-second",
    )
    metadata_calls = 0
    real_list = repository.list_revision_metadata

    def counted_list(owner_id: str):
        nonlocal metadata_calls
        metadata_calls += 1
        return real_list(owner_id)

    monkeypatch.setattr(repository, "list_revision_metadata", counted_list)
    service = CanvasService(db, repository=repository)
    scope = _scope(conversation_id, message_id)

    listed = service.list_canvases(scope)
    assert len(listed) == 2
    assert metadata_calls == 1
    assert all(not hasattr(item, "source") for item in listed)
    read = service.read_canvas(scope, first.identity.canvas_id)
    assert read.source == _html("first private")
    assert read.revision.canvas_id == first.identity.canvas_id
    assert metadata_calls == 2
