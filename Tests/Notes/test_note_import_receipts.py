"""Contracts for the device-private one-time Notes import receipt ledger."""

from __future__ import annotations

import inspect
import os
import sqlite3
import stat
from dataclasses import FrozenInstanceError, replace
from itertools import islice, repeat
from pathlib import Path

import pytest

from tldw_chatbook import config
from tldw_chatbook.Notes import note_import_receipts as receipt_module
from tldw_chatbook.Notes.note_import_execution_models import (
    ImportEffectState,
    ImportItemOutcome,
    ImportSessionState,
    approve_note_import_plan,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportBounds,
    ImportClassification,
    ImportMatch,
    ImportMatchKind,
    ImportPreviewItem,
    ImportSource,
    ImportSourceKind,
    NoteImportPlan,
    ParsedNotePayload,
    ProposedFolderMembership,
    RootCollisionChoice,
    RootCollisionState,
)
from tldw_chatbook.Notes.note_import_receipts import (
    EFFECT_STATE_TRANSITIONS,
    ITEM_OUTCOME_TRANSITIONS,
    SESSION_STATE_TRANSITIONS,
    EffectTransition,
    ImportBatchResult,
    ImportEffectCategory,
    ImportReceiptConflictError,
    ImportReceiptError,
    ImportReceiptTransitionError,
    ItemTransition,
    NoteImportReceiptRepository,
)

_APPROVAL_ID = "00000000-0000-4000-8000-000000000011"
_PRIVATE_SOURCE = Path("/private/alice/Project/notes.json")
_PRIVATE_TITLE = "Private quarterly title"
_PRIVATE_BODY = "Body secret which must never be persisted"
_PRIVATE_KEYWORD = "confidential-keyword"
_PRIVATE_TEMPLATE = "Private journal template"
_RAW_EXCEPTION = "raw exception /private/alice/Project/notes.json"


def test_public_effect_api_exposes_semantic_categories_and_lightweight_batch_result() -> (
    None
):
    assert receipt_module.ImportEffectCategory.PAYLOAD.value == "payload"
    assert receipt_module.ImportEffectCategory.FOLDER.value == "folder"
    assert receipt_module.ImportEffectCategory.MEMBERSHIP.value == "membership"
    assert "ImportEffectCategory" in receipt_module.__all__
    assert "ImportBatchResult" in receipt_module.__all__


def test_begin_uses_the_repository_transaction_context() -> None:
    source = inspect.getsource(NoteImportReceiptRepository.begin)

    assert "with self.transaction(immediate=True) as connection:" in source
    assert ".commit()" not in source
    assert ".rollback()" not in source


def test_effect_transition_public_signature_uses_category_not_table_name() -> None:
    parameters = inspect.signature(EffectTransition).parameters

    assert "category" in parameters
    assert "table" not in parameters


def test_effect_category_requires_the_exact_public_enum_type(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]

    with pytest.raises(TypeError, match="ImportEffectCategory"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category="payload",  # type: ignore[arg-type]
                    effect_id=payload.effect_id,
                    state=ImportEffectState.FAILED,
                    reason_code="database_busy",
                ),
            ),
        )
    with pytest.raises(TypeError, match="ImportEffectCategory"):
        repository.reset_retryable_effect(
            _APPROVAL_ID,
            category="payload",  # type: ignore[arg-type]
            effect_id=payload.effect_id,
        )


def test_transition_batch_returns_only_changed_semantic_records(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]
    assert payload.category is ImportEffectCategory.PAYLOAD
    assert not hasattr(payload, "table")

    result = repository.transition_batch(
        _APPROVAL_ID,
        effect_transitions=(
            EffectTransition(
                category=ImportEffectCategory.PAYLOAD,
                effect_id=payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                observed_version=7,
            ),
        ),
    )

    assert isinstance(result, ImportBatchResult)
    assert result.items == ()
    assert len(result.effects) == 1
    assert result.effects[0].category is ImportEffectCategory.PAYLOAD
    assert not hasattr(result, "session_id")
    assert payload.effect_id not in repr(result)


def test_effect_transition_and_reset_do_not_load_a_full_session_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]

    def reject_full_scan(*_args, **_kwargs):
        raise AssertionError("per-effect mutation must not load the full session")

    monkeypatch.setattr(repository, "_load_snapshot", reject_full_scan)
    changed = repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=ImportEffectCategory.PAYLOAD,
                effect_id=payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
            ),
        ),
    )
    assert changed[0].effect_id == payload.effect_id

    monkeypatch.undo()
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    monkeypatch.setattr(repository, "_load_snapshot", reject_full_scan)
    reset = repository.reset_retryable_effect(
        _APPROVAL_ID,
        category=ImportEffectCategory.PAYLOAD,
        effect_id=payload.effect_id,
    )
    assert reset.state is ImportEffectState.PENDING


def test_effect_transition_loads_only_the_affected_dependency_subgraph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = tuple(
        replace(
            _item(item_id=f"create-{index}", selected_action=ImportAction.CREATE_NEW),
            payloads=(
                replace(
                    _item(selected_action=ImportAction.CREATE_NEW).payloads[0],
                    content=f"private body {index}",
                ),
            ),
        )
        for index in range(40)
    )
    repository = _repository(tmp_path)
    repository.begin(
        approve_note_import_plan(
            replace(_plan(), items=items),
            approval_id=_APPROVAL_ID,
        ),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    target = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[17]
    original = repository._load_dependency_snapshot
    observed: list[tuple[int, int, int, int]] = []
    statements: list[str] = []
    original_connect = repository._connect

    def capture(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        observed.append(
            (
                len(snapshot.items),
                len(snapshot.payload_effects),
                len(snapshot.membership_effects),
                len(snapshot.folder_effects),
            )
        )
        return snapshot

    monkeypatch.setattr(repository, "_load_dependency_snapshot", capture)

    def traced_connect():
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(repository, "_connect", traced_connect)
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=ImportEffectCategory.PAYLOAD,
                effect_id=target.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                target_note_id="opaque-created-note",
                observed_version=1,
            ),
        ),
    )

    assert observed == [(1, 1, 1, 2)]
    assert _repository(tmp_path).aggregate_receipt(_APPROVAL_ID).failed == 1
    large_selects = [
        sql for sql in statements if sql.lstrip().upper().startswith("SELECT")
    ]
    assert all("GROUP BY" not in sql.upper() for sql in large_selects)
    assert any("target_note_id =" in sql for sql in large_selects)

    small_id = "00000000-0000-4000-8000-000000000012"
    small = NoteImportReceiptRepository(tmp_path / "small.sqlite3")
    small.begin(
        _approved_for_item(
            _item(item_id="small", selected_action=ImportAction.CREATE_NEW),
            approval_id=small_id,
        ),
        batch_size=25,
    )
    small.transition_session(small_id, ImportSessionState.RUNNING)
    small_target = small.load_session_snapshot(small_id).payload_effects[0]
    small_statements: list[str] = []
    small_connect = small._connect

    def traced_small_connect():
        connection = small_connect()
        connection.set_trace_callback(small_statements.append)
        return connection

    monkeypatch.setattr(small, "_connect", traced_small_connect)
    small.transition_effects(
        small_id,
        (
            EffectTransition(
                category=ImportEffectCategory.PAYLOAD,
                effect_id=small_target.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                target_note_id="opaque-small-note",
                observed_version=1,
            ),
        ),
    )
    small_selects = [
        sql for sql in small_statements if sql.lstrip().upper().startswith("SELECT")
    ]
    assert len(large_selects) == len(small_selects)


def _item(
    *,
    item_id: str = "item-1",
    content: str = _PRIVATE_BODY,
    selected_action: ImportAction = ImportAction.UPDATE_EXISTING,
) -> ImportPreviewItem:
    payload = ParsedNotePayload(
        title=_PRIVATE_TITLE,
        content=content,
        keywords=(_PRIVATE_KEYWORD,),
        template_name=_PRIVATE_TEMPLATE,
    )
    if selected_action is ImportAction.UPDATE_EXISTING:
        classification = ImportClassification.CHANGED_REPEAT
        match = ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id="opaque-note-7",
            note_version=7,
        )
        allowed_actions = (
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        )
        replace_content = True
    else:
        classification = ImportClassification.NEW
        match = None
        allowed_actions = (ImportAction.SKIP, ImportAction.CREATE_NEW)
        replace_content = False
    return ImportPreviewItem(
        item_id=item_id,
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="Project/notes.json",
            source_path=_PRIVATE_SOURCE,
        ),
        payloads=(payload,),
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Imported Project", "Meetings"),
            ),
        ),
        classification=classification,
        reason="Ready after explicit review.",
        default_action=ImportAction.CREATE_NEW,
        selected_action=selected_action,
        allowed_actions=allowed_actions,
        match=match,
        replace_content=replace_content,
        add_membership=True,
    )


def _plan(*, content: str = _PRIVATE_BODY) -> NoteImportPlan:
    return NoteImportPlan(
        bounds=ImportBounds(
            max_files=50,
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
            max_depth=8,
            max_entries=1_000,
            max_notes_per_file=100,
            max_keywords_per_note=50,
        ),
        items=(_item(content=content),),
        proposed_folder_paths=(
            ("Imported Project",),
            ("Imported Project", "Meetings"),
        ),
        root_collision=RootCollisionState(
            proposed_label="Project",
            collides=True,
            choice=RootCollisionChoice.RENAMED_ROOT,
            resolved_label="Imported Project",
        ),
    )


def _approved(*, content: str = _PRIVATE_BODY):
    return approve_note_import_plan(
        _plan(content=content),
        approval_id=_APPROVAL_ID,
    )


def _create_item_with_payloads(*, payload_count: int = 2) -> ImportPreviewItem:
    payloads = tuple(
        ParsedNotePayload(
            title=f"Private title {index}",
            content=f"Private body {index}",
            keywords=(f"private-keyword-{index}",),
        )
        for index in range(payload_count)
    )
    return ImportPreviewItem(
        item_id="multi-create",
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="Project/multi.json",
            source_path=Path("/private/alice/Project/multi.json"),
        ),
        payloads=payloads,
        memberships=tuple(
            ProposedFolderMembership(
                payload_index=index,
                folder_segments=("Imported Project", "Meetings"),
            )
            for index in range(payload_count)
        ),
        classification=ImportClassification.NEW,
        reason="Ready to create.",
        default_action=ImportAction.CREATE_NEW,
        selected_action=ImportAction.CREATE_NEW,
        allowed_actions=(ImportAction.SKIP, ImportAction.CREATE_NEW),
        match=None,
        replace_content=False,
        add_membership=True,
    )


def _create_item_in_folder(*, item_id: str, leaf: str) -> ImportPreviewItem:
    item = _item(item_id=item_id, selected_action=ImportAction.CREATE_NEW)
    return replace(
        item,
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Imported Project", leaf),
            ),
        ),
    )


def _approved_for_item(
    item: ImportPreviewItem,
    *,
    approval_id: str = _APPROVAL_ID,
    proposed_folder_paths: tuple[tuple[str, ...], ...] = (
        ("Imported Project",),
        ("Imported Project", "Meetings"),
    ),
):
    return approve_note_import_plan(
        replace(
            _plan(),
            items=(item,),
            proposed_folder_paths=proposed_folder_paths,
        ),
        approval_id=approval_id,
    )


def _folder_id_for_effect(effect) -> str:
    assert effect.folder_path_digest is not None
    return f"opaque-folder-{effect.folder_path_digest[:12]}"


def _applied_transition(effect, *, note_id: str = "opaque-note-1"):
    folder_id = (
        _folder_id_for_effect(effect)
        if effect.folder_path_digest is not None
        else "opaque-folder-1"
    )
    if effect.category is ImportEffectCategory.PAYLOAD:
        return EffectTransition(
            category=effect.category,
            effect_id=effect.effect_id,
            state=ImportEffectState.APPLIED,
            target_note_id=note_id,
            observed_version=1,
        )
    if effect.category is ImportEffectCategory.FOLDER:
        return EffectTransition(
            category=effect.category,
            effect_id=effect.effect_id,
            state=ImportEffectState.APPLIED,
            target_folder_id=folder_id,
        )
    return EffectTransition(
        category=effect.category,
        effect_id=effect.effect_id,
        state=ImportEffectState.APPLIED,
        target_note_id=note_id,
        target_folder_id=folder_id,
    )


def _repository(tmp_path: Path) -> NoteImportReceiptRepository:
    return NoteImportReceiptRepository(tmp_path / "notes-sync.sqlite3")


def test_notes_sync_state_path_is_profile_local_and_not_a_generic_database_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    assert config.get_notes_sync_state_db_path() == (
        tmp_path / "tldw_chatbook_notes_sync_state.db"
    )
    generic_paths = (
        Path(__file__).parents[2] / "tldw_chatbook/Chatbooks/database_paths.py"
    ).read_text(encoding="utf-8")
    settings_backup = (
        Path(__file__).parents[2] / "tldw_chatbook/UI/Tools_Settings_Window.py"
    ).read_text(encoding="utf-8")
    for source in (generic_paths, settings_backup):
        assert "get_notes_sync_state_db_path" not in source
        assert "tldw_chatbook_notes_sync_state.db" not in source


def test_receipt_repository_creates_v1_normalized_schema_without_private_text(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    snapshot = repository.begin(_approved(), batch_size=25)
    schema = repository._test_schema_snapshot()

    assert snapshot.batch_size == 25
    assert schema.user_version == 1
    assert set(schema.tables) == {
        "import_sessions",
        "import_items",
        "import_payload_effects",
        "import_folder_effects",
        "import_membership_effects",
    }
    database_text = (
        (tmp_path / "notes-sync.sqlite3").read_bytes().decode("utf-8", errors="ignore")
    )
    for forbidden in (
        str(_PRIVATE_SOURCE),
        "Project/notes.json",
        _PRIVATE_TITLE,
        _PRIVATE_BODY,
        _PRIVATE_KEYWORD,
        _PRIVATE_TEMPLATE,
        _RAW_EXCEPTION,
    ):
        assert forbidden not in database_text


def test_schema_column_census_excludes_content_and_exception_fields(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)

    columns = {
        column
        for table_columns in repository._test_schema_snapshot().columns.values()
        for column in table_columns
    }
    assert {
        "approval_id",
        "session_id",
        "item_id",
        "plan_digest",
        "source_locator_digest",
        "payload_digest",
        "selected_action",
        "effect_kind",
        "state",
        "target_note_id",
        "target_folder_id",
        "expected_version",
        "observed_version",
        "reason_code",
        "retryable",
    } <= columns
    forbidden_column_fragments = {
        "absolute_path",
        "display_path",
        "source_path",
        "title",
        "content",
        "keyword",
        "template",
        "exception",
        "raw_error",
    }
    assert not {
        column
        for column in columns
        if any(fragment in column for fragment in forbidden_column_fragments)
    }


def test_schema_v1_indexes_targeted_dependency_and_identity_queries(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    NoteImportReceiptRepository(database).begin(_approved(), batch_size=25)

    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX idx_import_payload_target")
        connection.execute("DROP INDEX idx_import_folder_target")
        connection.execute("DROP INDEX idx_import_membership_path")
        connection.execute("DROP INDEX idx_import_folder_parent")
        connection.execute("DROP INDEX IF EXISTS idx_import_items_source_session")
        connection.commit()

    NoteImportReceiptRepository(database).load_session_snapshot(_APPROVAL_ID)

    with sqlite3.connect(database) as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        detail = " ".join(
            str(row[3])
            for row in connection.execute(
                """EXPLAIN QUERY PLAN SELECT item_id
                FROM import_membership_effects
                WHERE session_id = ? AND folder_path_digest = ?""",
                ("opaque-session", "0" * 64),
            ).fetchall()
        )
        payload_detail = " ".join(
            str(row[3])
            for row in connection.execute(
                """EXPLAIN QUERY PLAN SELECT effect_id
                FROM import_payload_effects
                WHERE session_id = ? AND target_note_id = ?""",
                ("opaque-session", "opaque-note"),
            ).fetchall()
        )
        folder_detail = " ".join(
            str(row[3])
            for row in connection.execute(
                """EXPLAIN QUERY PLAN SELECT effect_id
                FROM import_folder_effects
                WHERE session_id = ? AND target_folder_id = ?""",
                ("opaque-session", "opaque-folder"),
            ).fetchall()
        )
        source_detail = " ".join(
            str(row[3])
            for row in connection.execute(
                """EXPLAIN QUERY PLAN
                SELECT item.item_id
                FROM import_items AS item
                JOIN import_sessions AS session
                  ON session.session_id = item.session_id
                WHERE item.source_locator_digest IN (?)
                  AND session.state = ?""",
                ("0" * 64, ImportSessionState.COMPLETED.value),
            ).fetchall()
        )

    assert {
        "idx_import_payload_target",
        "idx_import_folder_target",
        "idx_import_membership_path",
        "idx_import_folder_parent",
        "idx_import_items_source_session",
    } <= indexes
    assert "idx_import_membership_path" in detail
    assert "idx_import_payload_target" in payload_detail
    assert "idx_import_folder_target" in folder_detail
    assert "idx_import_items_source_session" in source_detail
    assert "SCAN item" not in source_detail
    assert "USE TEMP B-TREE" not in source_detail


def test_prior_observation_lookup_uses_bounded_large_input_query_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item_count = 1_025
    base = replace(
        _item(selected_action=ImportAction.CREATE_NEW),
        selected_action=ImportAction.SKIP,
        memberships=(),
        add_membership=False,
    )
    items = tuple(
        replace(
            base,
            item_id=f"bounded-source-{index}",
            source=ImportSource(
                kind=ImportSourceKind.SELECTED_FILE,
                display_path=f"Selected/private-{index}.md",
                source_path=Path(f"/private/bounded/private-{index}.md"),
            ),
            memberships=(),
            add_membership=False,
        )
        for index in range(item_count)
    )
    plan = NoteImportPlan(
        bounds=ImportBounds(
            max_files=item_count,
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
            max_depth=8,
            max_entries=item_count,
            max_notes_per_file=100,
            max_keywords_per_note=50,
        ),
        items=items,
        proposed_folder_paths=(),
    )
    repository = _repository(tmp_path)
    statements: list[str] = []
    original_connect = repository._connect

    def traced_connect():
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(repository, "_connect", traced_connect)

    assert repository.prior_observations_for_plan(plan) == ()

    lookup_statements = tuple(
        statement
        for statement in statements
        if "FROM import_items AS item" in statement
    )
    assert 1 <= len(lookup_statements) <= 2
    assert "/private/bounded" not in repr(repository)
    assert all("/private/bounded" not in statement for statement in lookup_statements)


def test_prior_observation_lookup_respects_a_lowered_sqlite_variable_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item_count = 41
    base = replace(
        _item(selected_action=ImportAction.CREATE_NEW),
        selected_action=ImportAction.SKIP,
        memberships=(),
        add_membership=False,
    )
    items = tuple(
        replace(
            base,
            item_id=f"limited-source-{index}",
            source=ImportSource(
                kind=ImportSourceKind.SELECTED_FILE,
                display_path=f"Selected/limited-{index}.md",
                source_path=Path(f"/private/limited/limited-{index}.md"),
            ),
        )
        for index in range(item_count)
    )
    plan = NoteImportPlan(
        bounds=ImportBounds(
            max_files=item_count,
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
            max_depth=8,
            max_entries=item_count,
            max_notes_per_file=100,
            max_keywords_per_note=50,
        ),
        items=items,
        proposed_folder_paths=(),
    )
    repository = _repository(tmp_path)
    statements: list[str] = []
    original_connect = repository._connect

    def limited_connect():
        connection = original_connect()
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 10)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(repository, "_connect", limited_connect)

    assert repository.prior_observations_for_plan(plan) == ()
    lookup_statements = tuple(
        statement
        for statement in statements
        if "FROM import_items AS item" in statement
    )
    assert len(lookup_statements) == 5
    assert all("/private/limited" not in statement for statement in lookup_statements)


def test_read_only_prior_observation_lookup_does_not_create_a_missing_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    repository = NoteImportReceiptRepository(database)

    assert repository.prior_observations_for_plan_read_only(_plan()) == ()
    assert not database.exists()


def test_read_only_prior_observation_lookup_leaves_schema_less_database_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated (value INTEGER)")
        connection.execute("INSERT INTO unrelated VALUES (42)")
        connection.commit()
    database.chmod(0o600)
    before = database.read_bytes()
    repository = NoteImportReceiptRepository(database)

    def reject_mutating_path(*_args, **_kwargs):
        raise AssertionError("read-only lookup entered a mutating schema path")

    monkeypatch.setattr(repository, "transaction", reject_mutating_path)
    monkeypatch.setattr(repository, "_initialize_schema", reject_mutating_path)

    assert repository.prior_observations_for_plan_read_only(_plan()) == ()
    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert connection.execute("SELECT value FROM unrelated").fetchone() == (42,)
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_schema WHERE name LIKE 'import_%'"
        ).fetchone() == (0,)


def test_read_only_prior_observation_lookup_uses_sqlite_enforced_read_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    calls: list[tuple[bool, bool]] = []
    original_connect = repository._connect

    def traced_connect(*, read_only: bool = False, must_exist: bool = False):
        calls.append((read_only, must_exist))
        connection = original_connect(
            read_only=read_only,
            must_exist=must_exist,
        )
        if read_only:
            with pytest.raises(sqlite3.OperationalError):
                connection.execute("DELETE FROM import_sessions")
        return connection

    monkeypatch.setattr(repository, "_connect", traced_connect)

    assert repository.prior_observations_for_plan_read_only(_plan()) == ()
    assert calls == [(True, True)]


def test_read_only_prior_observation_lookup_returns_completed_exact_match(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            EffectTransition(
                category=membership.category,
                effect_id=membership.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                target_folder_id=_folder_id_for_effect(membership),
            ),
        ),
    )
    repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.UPDATED,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    expected = repository.prior_observations_for_plan(_plan())
    database = tmp_path / "notes-sync.sqlite3"
    before = database.read_bytes()

    observations = repository.prior_observations_for_plan_read_only(_plan())

    assert observations == expected
    assert len(observations) == 1
    assert observations[0].note_id == "opaque-note-7"
    assert observations[0].note_version == 8
    assert database.read_bytes() == before


@pytest.mark.parametrize("database_kind", ["corrupt", "newer"])
def test_read_only_prior_observation_lookup_bounds_invalid_database_failures(
    tmp_path: Path,
    database_kind: str,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    if database_kind == "corrupt":
        database.write_bytes(b"private corrupt sqlite payload")
    else:
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA user_version = 2")
    database.chmod(0o600)

    with pytest.raises(ImportReceiptError) as caught:
        NoteImportReceiptRepository(database).prior_observations_for_plan_read_only(
            _plan()
        )

    assert str(database) not in str(caught.value)
    assert caught.value.__cause__ is None


def test_existing_v1_without_folder_parent_authority_fails_safe(tmp_path: Path) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE import_items (
                session_id TEXT, outcome TEXT, target_note_id TEXT,
                selected_action TEXT
            );
            CREATE TABLE import_payload_effects (
                session_id TEXT, state TEXT, target_note_id TEXT
            );
            CREATE TABLE import_folder_effects (
                session_id TEXT, state TEXT, target_folder_id TEXT
            );
            CREATE TABLE import_membership_effects (
                session_id TEXT, state TEXT, folder_path_digest TEXT,
                item_id TEXT
            );
            PRAGMA user_version = 1;
            """
        )

    with pytest.raises(ImportReceiptError, match="incompatible"):
        NoteImportReceiptRepository(database).load_session_snapshot(_APPROVAL_ID)


def test_begin_is_idempotent_durable_and_rejects_digest_or_batch_substitution(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    first = repository.begin(_approved(), batch_size=10)

    assert repository.begin(_approved(), batch_size=10) == first
    assert _repository(tmp_path).get_session(_APPROVAL_ID) == first
    with pytest.raises(ImportReceiptConflictError) as digest_conflict:
        repository.begin(_approved(content="substituted private body"), batch_size=10)
    with pytest.raises(ImportReceiptConflictError) as batch_conflict:
        repository.begin(_approved(), batch_size=11)
    for error in (digest_conflict.value, batch_conflict.value):
        assert _APPROVAL_ID not in str(error)
        assert "substituted private body" not in str(error)


def test_begin_validates_bounded_batch_size_without_creating_a_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    repository = NoteImportReceiptRepository(database)

    for invalid in (0, 101, True, 1.5, "25"):
        with pytest.raises((TypeError, ValueError)):
            repository.begin(_approved(), batch_size=invalid)  # type: ignore[arg-type]
    assert not database.exists()


def test_begin_rejects_missing_required_folder_prefix_before_database_creation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    plan = replace(
        _plan(),
        proposed_folder_paths=(("Imported Project", "Meetings"),),
    )
    repository = NoteImportReceiptRepository(database)

    with pytest.raises(ImportReceiptError, match="folder"):
        repository.begin(
            approve_note_import_plan(plan, approval_id=_APPROVAL_ID),
            batch_size=25,
        )

    assert not database.exists()


def test_begin_seeds_required_nested_folders_in_depth_then_plan_order(
    tmp_path: Path,
) -> None:
    plan = replace(
        _plan(),
        proposed_folder_paths=(
            ("Unused",),
            ("Imported Project", "Meetings"),
            ("Imported Project",),
        ),
    )
    repository = _repository(tmp_path)

    repository.begin(
        approve_note_import_plan(plan, approval_id=_APPROVAL_ID),
        batch_size=25,
    )
    folders = repository.load_session_snapshot(_APPROVAL_ID).folder_effects

    assert [effect.folder_path_digest for effect in folders] == [
        receipt_module._folder_path_digest(("Imported Project",)),
        receipt_module._folder_path_digest(("Imported Project", "Meetings")),
    ]


def test_begin_enforces_plan_bounds_before_database_creation(tmp_path: Path) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    payload = _item(selected_action=ImportAction.CREATE_NEW).payloads[0]
    oversized_payload = replace(payload, keywords=("one", "two"))
    oversized_item = replace(
        _item(selected_action=ImportAction.CREATE_NEW),
        payloads=(oversized_payload,),
    )
    plan = replace(
        _plan(),
        bounds=replace(_plan().bounds, max_keywords_per_note=1),
        items=(oversized_item,),
    )

    with pytest.raises(ImportReceiptError, match="bounds"):
        NoteImportReceiptRepository(database).begin(
            approve_note_import_plan(plan, approval_id=_APPROVAL_ID),
            batch_size=25,
        )

    assert not database.exists()


def test_begin_enforces_file_and_payload_bounds_before_database_creation(
    tmp_path: Path,
) -> None:
    first = _item(item_id="first", selected_action=ImportAction.CREATE_NEW)
    second = _item(item_id="second", selected_action=ImportAction.CREATE_NEW)
    file_plan = replace(
        _plan(),
        bounds=replace(_plan().bounds, max_files=1),
        items=(first, second),
    )
    payload_plan = replace(
        _plan(),
        bounds=replace(_plan().bounds, max_notes_per_file=1),
        items=(_create_item_with_payloads(payload_count=2),),
    )

    for name, plan in (
        ("files.sqlite3", file_plan),
        ("payloads.sqlite3", payload_plan),
    ):
        database = tmp_path / name
        with pytest.raises(ImportReceiptError, match="bounds"):
            NoteImportReceiptRepository(database).begin(
                approve_note_import_plan(plan, approval_id=_APPROVAL_ID),
                batch_size=25,
            )
        assert not database.exists()


def test_begin_enforces_absolute_ledger_row_ceiling_before_database_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    monkeypatch.setattr(receipt_module, "MAX_RECEIPT_LEDGER_ROWS", 5)

    with pytest.raises(ImportReceiptError, match="ledger"):
        NoteImportReceiptRepository(database).begin(_approved(), batch_size=25)

    assert not database.exists()


def test_begin_seeds_only_approved_effects_and_skip_has_no_mutation_effects(
    tmp_path: Path,
) -> None:
    plan = _plan()
    create_item = _item(item_id="item-create", selected_action=ImportAction.CREATE_NEW)
    skip_item = replace(
        create_item,
        item_id="item-skip",
        selected_action=ImportAction.SKIP,
        add_membership=False,
    )
    approved = approve_note_import_plan(
        replace(plan, items=(plan.items[0], create_item, skip_item)),
        approval_id=_APPROVAL_ID,
    )
    repository = _repository(tmp_path)

    session = repository.begin(approved, batch_size=25)
    durable = repository.load_session_snapshot(session.approval_id)

    assert len(durable.items) == 3
    assert len(durable.payload_effects) == 2
    assert {effect.effect_kind for effect in durable.payload_effects} == {
        "create_note",
        "replace_content",
    }
    assert len(durable.folder_effects) == 2
    assert len(durable.membership_effects) == 2
    assert all(effect.item_id != "item-skip" for effect in durable.payload_effects)
    assert all(effect.item_id != "item-skip" for effect in durable.membership_effects)


def test_all_skip_and_zero_work_sessions_complete_without_folder_effects(
    tmp_path: Path,
) -> None:
    create_item = _item(selected_action=ImportAction.CREATE_NEW)
    skip_item = replace(
        create_item,
        selected_action=ImportAction.SKIP,
        add_membership=False,
    )
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(skip_item), batch_size=25)

    skipped = repository.load_session_snapshot(_APPROVAL_ID)
    assert skipped.payload_effects == ()
    assert skipped.folder_effects == ()
    assert skipped.membership_effects == ()
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    repository.transition_item(
        _APPROVAL_ID,
        skip_item.item_id,
        ImportItemOutcome.SKIPPED,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    skip_receipt = repository.aggregate_receipt(_APPROVAL_ID)
    assert (skip_receipt.total, skip_receipt.skipped) == (1, 1)

    empty_id = "00000000-0000-4000-8000-000000000012"
    empty = approve_note_import_plan(
        replace(_plan(), items=()),
        approval_id=empty_id,
    )
    repository.begin(empty, batch_size=25)
    assert repository.load_session_snapshot(empty_id).folder_effects == ()
    repository.transition_session(empty_id, ImportSessionState.RUNNING)
    repository.transition_session(empty_id, ImportSessionState.COMPLETED)
    empty_receipt = repository.aggregate_receipt(empty_id)
    assert (empty_receipt.total, empty_receipt.completed) == (0, 0)


def test_multi_payload_create_counts_notes_and_reopens_all_target_ids(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    session = repository.begin(
        _approved_for_item(_create_item_with_payloads()),
        batch_size=25,
    )
    assert session.total == 2
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    durable = repository.load_session_snapshot(_APPROVAL_ID)
    note_ids = ("opaque-created-note-1", "opaque-created-note-2")
    transitions = [
        EffectTransition(
            category=effect.category,
            effect_id=effect.effect_id,
            state=ImportEffectState.APPLIED,
            target_note_id=note_ids[effect.payload_index],
            observed_version=1,
        )
        for effect in durable.payload_effects
    ]
    transitions.extend(_applied_transition(effect) for effect in durable.folder_effects)
    transitions.extend(
        _applied_transition(
            effect,
            note_id=note_ids[effect.payload_index],
        )
        for effect in durable.membership_effects
    )
    repository.transition_effects(_APPROVAL_ID, transitions)
    repository.transition_item(
        _APPROVAL_ID,
        "multi-create",
        ImportItemOutcome.IMPORTED,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)

    receipt = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (receipt.total, receipt.completed, receipt.imported) == (2, 2, 2)
    assert receipt._note_ids == note_ids


def test_multi_payload_receipt_preserves_partial_success_failure_and_note_ids(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(
        _approved_for_item(_create_item_with_payloads()),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    first_payload, second_payload = snapshot.payload_effects
    first_membership, _second_membership = snapshot.membership_effects
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=first_payload.category,
                effect_id=first_payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-created-note-a",
                observed_version=1,
            ),
            _applied_transition(
                first_membership,
                note_id="opaque-created-note-a",
            ),
            EffectTransition(
                category=second_payload.category,
                effect_id=second_payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-created-note-b",
                observed_version=8,
            ),
        ),
    )
    repository.transition_item(
        _APPROVAL_ID,
        "multi-create",
        ImportItemOutcome.FAILED,
        reason_code="database_busy",
        retryable=True,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)

    receipt = repository.aggregate_receipt(_APPROVAL_ID)
    reopened = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)

    assert receipt == reopened
    assert (
        receipt.total,
        receipt.completed,
        receipt.imported,
        receipt.failed,
        receipt.retryable,
    ) == (
        2,
        2,
        1,
        1,
        1,
    )
    assert receipt._note_ids == (
        "opaque-created-note-a",
        "opaque-created-note-b",
    )


@pytest.mark.parametrize(
    ("outcome", "reason_code", "retryable", "target_note_id", "observed_version"),
    [
        (ImportItemOutcome.IMPORTED, None, False, "spurious-note", None),
        (ImportItemOutcome.IMPORTED, None, False, None, 1),
        (ImportItemOutcome.FAILED, "database_busy", True, "spurious-note", 1),
    ],
)
def test_create_item_summary_rejects_note_reconciliation_metadata_atomically(
    tmp_path: Path,
    outcome: ImportItemOutcome,
    reason_code: str | None,
    retryable: bool,
    target_note_id: str | None,
    observed_version: int | None,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ValueError, match="Create item summaries"):
        repository.transition_item(
            _APPROVAL_ID,
            item.item_id,
            outcome,
            reason_code=reason_code,
            retryable=retryable,
            target_note_id=target_note_id,
            observed_version=observed_version,
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert reopened.outcome is ImportItemOutcome.PENDING
    assert reopened.target_note_id is None
    assert reopened.observed_version is None


def test_create_item_metadata_rejection_rolls_back_mixed_effect_batch(
    tmp_path: Path,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]

    with pytest.raises(ValueError, match="Create item summaries"):
        repository.transition_batch(
            _APPROVAL_ID,
            item_transitions=(
                ItemTransition(
                    item_id=item.item_id,
                    outcome=ImportItemOutcome.IMPORTED,
                    target_note_id="spurious-item-note",
                    observed_version=1,
                ),
            ),
            effect_transitions=(
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="durable-payload-note",
                    observed_version=1,
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.items[0].target_note_id is None
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert reopened.payload_effects[0].target_note_id is None


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("item_reconciliation", "Create item summary"),
        ("item_expected", "Create item summary"),
        ("payload_expected", "Create payload"),
    ],
)
def test_reducer_rejects_corrupt_create_metadata_after_reopen(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_batch(
        _APPROVAL_ID,
        item_transitions=(
            ItemTransition(
                item_id=item.item_id,
                outcome=ImportItemOutcome.IMPORTED,
            ),
        ),
        effect_transitions=tuple(
            _applied_transition(effect) for effect in snapshot.folder_effects
        )
        + (
            EffectTransition(
                category=snapshot.payload_effects[0].category,
                effect_id=snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="durable-payload-note",
                observed_version=1,
            ),
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="durable-payload-note",
            ),
        ),
    )
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        if corruption == "item_reconciliation":
            connection.execute(
                """
                UPDATE import_items SET target_note_id = ?, observed_version = ?
                WHERE session_id = ?
                """,
                ("spurious-item-note", 99, snapshot.session_id),
            )
        elif corruption == "item_expected":
            connection.execute(
                "UPDATE import_items SET expected_version = 7 WHERE session_id = ?",
                (snapshot.session_id,),
            )
        else:
            connection.execute(
                """
                UPDATE import_payload_effects SET expected_version = 7
                WHERE session_id = ?
                """,
                (snapshot.session_id,),
            )

    reopened_repository = _repository(tmp_path)
    with pytest.raises(ImportReceiptConflictError, match=message):
        reopened_repository.aggregate_receipt(_APPROVAL_ID)
    with pytest.raises(ImportReceiptConflictError, match=message):
        reopened_repository.transition_session(
            _APPROVAL_ID,
            ImportSessionState.COMPLETED,
        )
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).state
        is ImportSessionState.RUNNING
    )


@pytest.mark.parametrize(
    ("classification", "match_kind"),
    [
        (ImportClassification.CHANGED_REPEAT, ImportMatchKind.EXACT),
        (ImportClassification.UNCERTAIN_MATCH, ImportMatchKind.UNCERTAIN),
    ],
)
def test_matched_create_plan_seeds_unbound_authority_and_imports_new_note(
    tmp_path: Path,
    classification: ImportClassification,
    match_kind: ImportMatchKind,
) -> None:
    matched_create = replace(
        _item(),
        classification=classification,
        selected_action=ImportAction.CREATE_NEW,
        allowed_actions=(
            (
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            )
            if classification is ImportClassification.CHANGED_REPEAT
            else (ImportAction.SKIP, ImportAction.CREATE_NEW)
        ),
        match=ImportMatch(
            kind=match_kind,
            note_id="matched-existing-note",
            note_version=7,
        ),
        replace_content=False,
    )
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(matched_create), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)

    assert (snapshot.items[0].target_note_id, snapshot.items[0].expected_version) == (
        None,
        None,
    )
    assert (
        snapshot.payload_effects[0].target_note_id,
        snapshot.payload_effects[0].expected_version,
    ) == (None, None)
    assert snapshot.membership_effects[0].target_note_id is None

    repository.transition_batch(
        _APPROVAL_ID,
        item_transitions=(
            ItemTransition(
                item_id=matched_create.item_id,
                outcome=ImportItemOutcome.IMPORTED,
            ),
        ),
        effect_transitions=tuple(
            _applied_transition(effect) for effect in snapshot.folder_effects
        )
        + (
            EffectTransition(
                category=snapshot.payload_effects[0].category,
                effect_id=snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="newly-created-note",
                observed_version=1,
            ),
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="newly-created-note",
            ),
        ),
    )

    receipt = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (receipt.imported, receipt._note_ids) == (1, ("newly-created-note",))


def test_transition_batch_rejects_terminal_create_with_pending_membership(
    tmp_path: Path,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]

    with pytest.raises(ImportReceiptError, match="terminal item summary"):
        repository.transition_batch(
            _APPROVAL_ID,
            item_transitions=(
                ItemTransition(
                    item_id=item.item_id,
                    outcome=ImportItemOutcome.IMPORTED,
                ),
            ),
            effect_transitions=(
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="durable-payload-note",
                    observed_version=1,
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert reopened.membership_effects[0].state is ImportEffectState.PENDING


@pytest.mark.parametrize(
    "child_state", [ImportEffectState.APPLIED, ImportEffectState.FAILED]
)
def test_transition_batch_allows_pending_parent_with_terminal_child_effect(
    tmp_path: Path,
    child_state: ImportEffectState,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]
    transition = EffectTransition(
        category=payload.category,
        effect_id=payload.effect_id,
        state=child_state,
        reason_code="database_busy"
        if child_state is ImportEffectState.FAILED
        else None,
        retryable=child_state is ImportEffectState.FAILED,
        target_note_id="durable-payload-note",
        observed_version=1,
    )

    repository.transition_effects(_APPROVAL_ID, (transition,))

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.payload_effects[0].state is child_state


def test_multi_payload_retryable_count_is_derived_per_failed_note_unit(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(
        _approved_for_item(_create_item_with_payloads()),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payloads = repository.load_session_snapshot(_APPROVAL_ID).payload_effects
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=payloads[0].category,
                effect_id=payloads[0].effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-created-note-a",
                observed_version=1,
            ),
            EffectTransition(
                category=payloads[1].category,
                effect_id=payloads[1].effect_id,
                state=ImportEffectState.FAILED,
                reason_code="invalid_payload",
                retryable=False,
                target_note_id="opaque-created-note-b",
                observed_version=1,
            ),
        ),
    )
    repository.transition_item(
        _APPROVAL_ID,
        "multi-create",
        ImportItemOutcome.FAILED,
        reason_code="partial_failure",
        retryable=True,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)

    receipt = repository.aggregate_receipt(_APPROVAL_ID)

    assert (receipt.failed, receipt.retryable) == (2, 1)


def test_create_membership_note_identity_must_match_its_payload(
    tmp_path: Path,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    with pytest.raises(ImportReceiptConflictError, match="identity"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=snapshot.payload_effects[0].category,
                    effect_id=snapshot.payload_effects[0].effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-created-note-a",
                    observed_version=1,
                ),
                EffectTransition(
                    category=snapshot.membership_effects[0].category,
                    effect_id=snapshot.membership_effects[0].effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-created-note-b",
                    target_folder_id="opaque-folder-1",
                ),
            ),
        )
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert reopened.membership_effects[0].state is ImportEffectState.PENDING


def test_conflicting_failed_membership_identity_rolls_back_at_transition(
    tmp_path: Path,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-created-note-a",
                observed_version=1,
            ),
        ),
    )

    with pytest.raises(ImportReceiptConflictError, match="identity"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=membership.category,
                    effect_id=membership.effect_id,
                    state=ImportEffectState.FAILED,
                    reason_code="database_busy",
                    retryable=True,
                    target_note_id="opaque-created-note-b",
                    target_folder_id="opaque-folder-1",
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.payload_effects[0].state is ImportEffectState.APPLIED
    assert reopened.membership_effects[0].state is ImportEffectState.PENDING
    assert reopened.membership_effects[0].target_note_id is None


def test_create_payload_note_identities_must_be_unique_per_note_unit(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(
        _approved_for_item(_create_item_with_payloads()),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    with pytest.raises(ImportReceiptConflictError, match="unique"):
        repository.transition_effects(
            _APPROVAL_ID,
            tuple(
                _applied_transition(effect, note_id="duplicate-created-note")
                for effect in (
                    *snapshot.payload_effects,
                    *snapshot.membership_effects,
                )
            ),
        )
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert all(
        effect.state is ImportEffectState.PENDING
        for effect in (*reopened.payload_effects, *reopened.membership_effects)
    )


def test_create_payload_note_identities_must_be_unique_across_the_session(
    tmp_path: Path,
) -> None:
    base_item = _create_item_with_payloads(payload_count=1)
    first = replace(
        base_item,
        item_id="create-a",
        source=replace(
            base_item.source,
            display_path="Project/a.json",
            source_path=Path("/private/alice/Project/a.json"),
        ),
    )
    second = replace(
        base_item,
        item_id="create-b",
        source=replace(
            base_item.source,
            display_path="Project/b.json",
            source_path=Path("/private/alice/Project/b.json"),
        ),
    )
    approved = approve_note_import_plan(
        replace(_plan(), items=(first, second)),
        approval_id=_APPROVAL_ID,
    )
    repository = _repository(tmp_path)
    repository.begin(approved, batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)

    with pytest.raises(ImportReceiptConflictError, match="unique"):
        repository.transition_effects(
            _APPROVAL_ID,
            tuple(
                _applied_transition(effect, note_id="duplicate-session-note")
                for effect in (
                    *snapshot.payload_effects,
                    *snapshot.membership_effects,
                )
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert all(
        effect.state is ImportEffectState.PENDING
        for effect in (*reopened.payload_effects, *reopened.membership_effects)
    )


def test_create_note_identity_cannot_alias_an_update_unit_in_the_same_session(
    tmp_path: Path,
) -> None:
    create_item = _item(
        item_id="create-item",
        selected_action=ImportAction.CREATE_NEW,
    )
    update_item = replace(
        _item(item_id="update-item"),
        source=replace(
            _item().source,
            display_path="Project/update.json",
            source_path=Path("/private/alice/Project/update.json"),
        ),
    )
    approved = approve_note_import_plan(
        replace(_plan(), items=(create_item, update_item)),
        approval_id=_APPROVAL_ID,
    )
    repository = _repository(tmp_path)
    repository.begin(approved, batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    create_payload = next(
        effect for effect in snapshot.payload_effects if effect.item_id == "create-item"
    )
    create_membership = next(
        effect
        for effect in snapshot.membership_effects
        if effect.item_id == "create-item"
    )

    with pytest.raises(ImportReceiptConflictError, match="unique"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                _applied_transition(create_payload, note_id="opaque-note-7"),
                _applied_transition(create_membership, note_id="opaque-note-7"),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert (
        next(
            effect
            for effect in reopened.payload_effects
            if effect.item_id == "create-item"
        ).state
        is ImportEffectState.PENDING
    )


@pytest.mark.parametrize(
    "state",
    [ImportEffectState.APPLIED, ImportEffectState.FAILED],
)
def test_create_membership_cannot_bind_before_its_payload_target(
    tmp_path: Path,
    state: ImportEffectState,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    membership = repository.load_session_snapshot(_APPROVAL_ID).membership_effects[0]
    transition = EffectTransition(
        category=membership.category,
        effect_id=membership.effect_id,
        state=state,
        reason_code="database_busy" if state is ImportEffectState.FAILED else None,
        retryable=state is ImportEffectState.FAILED,
        target_note_id="premature-note-id",
        target_folder_id="opaque-folder-1",
    )

    with pytest.raises(ImportReceiptConflictError, match="payload"):
        repository.transition_effects(_APPROVAL_ID, (transition,))

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.membership_effects[0].state is ImportEffectState.PENDING
    assert reopened.membership_effects[0].target_note_id is None


def test_failed_create_membership_may_omit_identities_before_authority_exists(
    tmp_path: Path,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    membership = repository.load_session_snapshot(_APPROVAL_ID).membership_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=membership.category,
                effect_id=membership.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
            ),
        ),
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)

    receipt = repository.aggregate_receipt(_APPROVAL_ID)

    assert (receipt.completed, receipt.failed, receipt.retryable) == (1, 1, 1)


def test_applied_membership_requires_applied_folder_authority(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]

    with pytest.raises(ImportReceiptConflictError, match="applied folder"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                ),
                EffectTransition(
                    category=membership.category,
                    effect_id=membership.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    target_folder_id="premature-folder",
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert reopened.membership_effects[0].state is ImportEffectState.PENDING


def test_failed_membership_folder_binding_requires_known_folder_authority(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    membership = repository.load_session_snapshot(_APPROVAL_ID).membership_effects[0]

    with pytest.raises(ImportReceiptConflictError, match="known folder"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=membership.category,
                    effect_id=membership.effect_id,
                    state=ImportEffectState.FAILED,
                    reason_code="database_busy",
                    retryable=True,
                    target_note_id="opaque-note-7",
                    target_folder_id="premature-folder",
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.membership_effects[0].state is ImportEffectState.PENDING
    assert reopened.membership_effects[0].target_folder_id is None


def test_applied_create_membership_requires_an_applied_payload(
    tmp_path: Path,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-created-note",
                observed_version=1,
            ),
        ),
    )

    with pytest.raises(ImportReceiptConflictError, match="applied payload"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=membership.category,
                    effect_id=membership.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-created-note",
                    target_folder_id="opaque-folder-1",
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.payload_effects[0].state is ImportEffectState.FAILED
    assert reopened.membership_effects[0].state is ImportEffectState.PENDING


@pytest.mark.parametrize("corruption", ["membership_mismatch", "duplicate_payload"])
def test_completion_reducer_rejects_corrupt_note_identity_atomically(
    tmp_path: Path,
    corruption: str,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(
        _approved_for_item(_create_item_with_payloads()),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    note_ids = ("opaque-created-note-a", "opaque-created-note-b")
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + tuple(
            _applied_transition(effect, note_id=note_ids[effect.payload_index])
            for effect in (
                *snapshot.payload_effects,
                *snapshot.membership_effects,
            )
        ),
    )
    repository.transition_item(
        _APPROVAL_ID,
        "multi-create",
        ImportItemOutcome.IMPORTED,
    )
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        if corruption == "membership_mismatch":
            connection.execute(
                """
                UPDATE import_membership_effects SET target_note_id = ?
                WHERE session_id = ? AND payload_index = 0
                """,
                ("corrupt-note-id", snapshot.session_id),
            )
        else:
            connection.execute(
                """
                UPDATE import_payload_effects SET target_note_id = ?
                WHERE session_id = ? AND payload_index = 1
                """,
                (note_ids[0], snapshot.session_id),
            )
            connection.execute(
                """
                UPDATE import_membership_effects SET target_note_id = ?
                WHERE session_id = ? AND payload_index = 1
                """,
                (note_ids[0], snapshot.session_id),
            )

    with pytest.raises(ImportReceiptError):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.RUNNING
    with pytest.raises(ImportReceiptError):
        repository.aggregate_receipt(_APPROVAL_ID)


def test_distinct_folder_paths_cannot_bind_the_same_folder_identity(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)

    with pytest.raises(ImportReceiptConflictError, match="folder identities"):
        repository.transition_effects(
            _APPROVAL_ID,
            tuple(
                EffectTransition(
                    category=effect.category,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_folder_id="aliased-folder",
                )
                for effect in snapshot.folder_effects
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert all(
        effect.state is ImportEffectState.PENDING for effect in reopened.folder_effects
    )
    assert all(effect.target_folder_id is None for effect in reopened.folder_effects)


def test_membership_folder_identity_must_match_its_approved_path(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]

    with pytest.raises(ImportReceiptConflictError, match="folder identity"):
        repository.transition_effects(
            _APPROVAL_ID,
            tuple(
                EffectTransition(
                    category=effect.category,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_folder_id=f"approved-folder-{index}",
                )
                for index, effect in enumerate(snapshot.folder_effects)
            )
            + (
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                ),
                EffectTransition(
                    category=membership.category,
                    effect_id=membership.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    target_folder_id="unapproved-folder",
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert all(
        effect.state is ImportEffectState.PENDING for effect in reopened.folder_effects
    )
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert reopened.membership_effects[0].state is ImportEffectState.PENDING


def test_completion_reducer_rejects_corrupt_folder_identity_atomically(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]
    folder_transitions = tuple(
        EffectTransition(
            category=effect.category,
            effect_id=effect.effect_id,
            state=ImportEffectState.APPLIED,
            target_folder_id=f"approved-folder-{index}",
        )
        for index, effect in enumerate(snapshot.folder_effects)
    )
    repository.transition_effects(
        _APPROVAL_ID,
        folder_transitions
        + (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            EffectTransition(
                category=membership.category,
                effect_id=membership.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                target_folder_id="approved-folder-1",
            ),
        ),
    )
    repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.UPDATED,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE import_membership_effects SET target_folder_id = ? WHERE session_id = ?",
            ("corrupt-folder-id", snapshot.session_id),
        )

    with pytest.raises(ImportReceiptConflictError, match="folder identity"):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.RUNNING
    with pytest.raises(ImportReceiptConflictError, match="folder identity"):
        repository.aggregate_receipt(_APPROVAL_ID)


def test_membership_failure_makes_its_create_note_unit_retryable_failed(
    tmp_path: Path,
) -> None:
    item = _create_item_with_payloads(payload_count=1)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-created-note",
                observed_version=1,
            ),
            EffectTransition(
                category=membership.category,
                effect_id=membership.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="folder_conflict",
                retryable=True,
                target_note_id="opaque-created-note",
                target_folder_id=_folder_id_for_effect(membership),
            ),
        ),
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)

    receipt = repository.aggregate_receipt(_APPROVAL_ID)

    assert (
        receipt.total,
        receipt.completed,
        receipt.imported,
        receipt.failed,
        receipt.retryable,
    ) == (1, 1, 0, 1, 1)
    assert receipt.reason_code == "folder_conflict"


def test_membership_only_update_counts_one_updated_note_unit(tmp_path: Path) -> None:
    item = replace(_item(), replace_content=False)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    assert snapshot.payload_effects == ()
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )
    before_item_summary = repository.aggregate_receipt(_APPROVAL_ID)
    repository.transition_item(
        _APPROVAL_ID,
        item.item_id,
        ImportItemOutcome.UPDATED,
        target_note_id="opaque-note-7",
        observed_version=7,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)

    receipt = repository.aggregate_receipt(_APPROVAL_ID)

    assert (
        before_item_summary.completed,
        before_item_summary.updated,
    ) == (1, 1)
    assert (receipt.total, receipt.completed, receipt.updated) == (1, 1, 1)
    assert receipt._note_ids == ("opaque-note-7",)


def test_replace_content_update_waits_for_membership_then_counts_one_unit(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
        ),
    )
    payload_only = repository.aggregate_receipt(_APPROVAL_ID)
    repository.transition_effects(
        _APPROVAL_ID,
        (
            _applied_transition(
                membership,
                note_id="opaque-note-7",
            ),
        ),
    )

    before_item_summary = repository.aggregate_receipt(_APPROVAL_ID)
    repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.UPDATED,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    after_item_summary = repository.aggregate_receipt(_APPROVAL_ID)

    assert (payload_only.completed, payload_only.updated) == (0, 0)
    assert (before_item_summary.completed, before_item_summary.updated) == (1, 1)
    assert (after_item_summary.completed, after_item_summary.updated) == (1, 1)


def test_terminal_update_requires_a_final_item_observation(tmp_path: Path) -> None:
    item = replace(_item(), replace_content=False)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptTransitionError, match="observation"):
        repository.transition_item(
            _APPROVAL_ID,
            item.item_id,
            ImportItemOutcome.UPDATED,
            target_note_id="opaque-note-7",
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert reopened.outcome is ImportItemOutcome.PENDING
    assert reopened.observed_version is None


@pytest.mark.parametrize("observed_version", [6, 8])
def test_membership_only_update_observation_must_equal_expected_version(
    tmp_path: Path,
    observed_version: int,
) -> None:
    item = replace(_item(), replace_content=False)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )

    with pytest.raises(ImportReceiptConflictError, match="exactly match"):
        repository.transition_item(
            _APPROVAL_ID,
            item.item_id,
            ImportItemOutcome.UPDATED,
            target_note_id="opaque-note-7",
            observed_version=observed_version,
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert reopened.outcome is ImportItemOutcome.PENDING
    assert reopened.observed_version is None


@pytest.mark.parametrize("observed_version", [7, 9])
def test_replace_update_observation_must_be_exactly_expected_plus_one(
    tmp_path: Path,
    observed_version: int,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]

    with pytest.raises(ImportReceiptConflictError, match="exactly one version"):
        repository.transition_batch(
            _APPROVAL_ID,
            item_transitions=(
                ItemTransition(
                    item_id="item-1",
                    outcome=ImportItemOutcome.UPDATED,
                    target_note_id="opaque-note-7",
                    observed_version=observed_version,
                ),
            ),
            effect_transitions=tuple(
                _applied_transition(effect) for effect in snapshot.folder_effects
            )
            + (
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=observed_version,
                ),
                _applied_transition(membership, note_id="opaque-note-7"),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.items[0].observed_version is None
    assert all(
        effect.state is ImportEffectState.PENDING for effect in reopened.payload_effects
    )
    assert all(
        effect.state is ImportEffectState.PENDING
        for effect in reopened.membership_effects
    )
    assert all(
        effect.state is ImportEffectState.PENDING for effect in reopened.folder_effects
    )


@pytest.mark.parametrize("observed_version", [7, 9])
def test_replace_payload_first_rejects_non_successor_version_atomically(
    tmp_path: Path,
    observed_version: int,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]

    with pytest.raises(ImportReceiptConflictError, match="exactly one version"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=observed_version,
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert reopened.payload_effects[0].observed_version is None


def test_replace_payload_first_success_reopens_and_later_finalizes_item(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=snapshot.payload_effects[0].category,
                effect_id=snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
        ),
    )

    reopened_repository = _repository(tmp_path)
    pending_receipt = reopened_repository.aggregate_receipt(_APPROVAL_ID)
    reopened = reopened_repository.load_session_snapshot(_APPROVAL_ID)
    assert (pending_receipt.completed, pending_receipt.updated) == (0, 0)
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.payload_effects[0].observed_version == 8

    reopened_repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in reopened.folder_effects)
        + (
            _applied_transition(
                reopened.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )
    finalized = reopened_repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.UPDATED,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    assert (finalized.outcome, finalized.observed_version) == (
        ImportItemOutcome.UPDATED,
        8,
    )


def test_reducer_rejects_corrupt_pending_parent_replace_payload_version(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=snapshot.payload_effects[0].category,
                effect_id=snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
        ),
    )
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE import_payload_effects SET observed_version = 9 WHERE session_id = ?",
            (snapshot.session_id,),
        )

    reopened_repository = _repository(tmp_path)
    with pytest.raises(ImportReceiptConflictError, match="exactly one version"):
        reopened_repository.aggregate_receipt(_APPROVAL_ID)
    reopened = reopened_repository.load_session_snapshot(_APPROVAL_ID)
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.payload_effects[0].observed_version == 9


def test_replace_payload_rejects_missing_expected_version_authority(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE import_items SET expected_version = NULL WHERE session_id = ?",
            (snapshot.session_id,),
        )

    with pytest.raises(ImportReceiptConflictError, match="expected version"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=snapshot.payload_effects[0].category,
                    effect_id=snapshot.payload_effects[0].effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert reopened.payload_effects[0].observed_version is None


@pytest.mark.parametrize("payload_expected_version", [6, 8])
def test_replace_payload_rejects_divergent_expected_authority_atomically(
    tmp_path: Path,
    payload_expected_version: int,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE import_payload_effects SET expected_version = ? WHERE session_id = ?",
            (payload_expected_version, snapshot.session_id),
        )

    with pytest.raises(ImportReceiptConflictError, match="expected version authority"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=snapshot.payload_effects[0].category,
                    effect_id=snapshot.payload_effects[0].effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert reopened.payload_effects[0].observed_version is None
    assert reopened.payload_effects[0].expected_version == payload_expected_version


def test_completion_reducer_rejects_corrupt_payload_expected_authority(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_batch(
        _APPROVAL_ID,
        item_transitions=(
            ItemTransition(
                item_id="item-1",
                outcome=ImportItemOutcome.UPDATED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
        ),
        effect_transitions=tuple(
            _applied_transition(effect) for effect in snapshot.folder_effects
        )
        + (
            EffectTransition(
                category=snapshot.payload_effects[0].category,
                effect_id=snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE import_payload_effects SET expected_version = 6 WHERE session_id = ?",
            (snapshot.session_id,),
        )

    reopened_repository = _repository(tmp_path)
    with pytest.raises(ImportReceiptConflictError, match="expected version authority"):
        reopened_repository.aggregate_receipt(_APPROVAL_ID)
    with pytest.raises(ImportReceiptConflictError, match="expected version authority"):
        reopened_repository.transition_session(
            _APPROVAL_ID,
            ImportSessionState.COMPLETED,
        )
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).state
        is ImportSessionState.RUNNING
    )


def test_replace_update_item_observation_must_match_applied_payload(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    membership = snapshot.membership_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            _applied_transition(
                membership,
                note_id="opaque-note-7",
            ),
        ),
    )

    with pytest.raises(ImportReceiptConflictError, match="exactly one version"):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.UPDATED,
            target_note_id="opaque-note-7",
            observed_version=9,
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert reopened.outcome is ImportItemOutcome.PENDING
    assert reopened.observed_version is None


def test_completion_reducer_rejects_corrupt_update_observation_atomically(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=snapshot.payload_effects[0].category,
                effect_id=snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )
    repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.UPDATED,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE import_items SET observed_version = ? WHERE session_id = ?",
            (9, snapshot.session_id),
        )

    with pytest.raises(ImportReceiptConflictError, match="exactly one version"):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.RUNNING
    assert reopened.items[0].observed_version == 9
    with pytest.raises(ImportReceiptConflictError, match="exactly one version"):
        repository.aggregate_receipt(_APPROVAL_ID)


@pytest.mark.parametrize("corrupt_version", [7, 9])
def test_replace_update_completion_rejects_non_successor_version_after_reopen(
    tmp_path: Path,
    corrupt_version: int,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_batch(
        _APPROVAL_ID,
        item_transitions=(
            ItemTransition(
                item_id="item-1",
                outcome=ImportItemOutcome.UPDATED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
        ),
        effect_transitions=tuple(
            _applied_transition(effect) for effect in snapshot.folder_effects
        )
        + (
            EffectTransition(
                category=snapshot.payload_effects[0].category,
                effect_id=snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE import_items SET observed_version = ? WHERE session_id = ?",
            (corrupt_version, snapshot.session_id),
        )
        connection.execute(
            "UPDATE import_payload_effects SET observed_version = ? WHERE session_id = ?",
            (corrupt_version, snapshot.session_id),
        )

    reopened_repository = _repository(tmp_path)
    with pytest.raises(ImportReceiptConflictError, match="exactly one version"):
        reopened_repository.aggregate_receipt(_APPROVAL_ID)
    with pytest.raises(ImportReceiptConflictError, match="exactly one version"):
        reopened_repository.transition_session(
            _APPROVAL_ID,
            ImportSessionState.COMPLETED,
        )
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).state
        is ImportSessionState.RUNNING
    )


def test_membership_only_completion_rejects_later_version_after_reopen(
    tmp_path: Path,
) -> None:
    item = replace(_item(), replace_content=False)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_batch(
        _APPROVAL_ID,
        item_transitions=(
            ItemTransition(
                item_id=item.item_id,
                outcome=ImportItemOutcome.UPDATED,
                target_note_id="opaque-note-7",
                observed_version=7,
            ),
        ),
        effect_transitions=tuple(
            _applied_transition(effect) for effect in snapshot.folder_effects
        )
        + (
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )
    database = tmp_path / "notes-sync.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE import_items SET observed_version = 8 WHERE session_id = ?",
            (snapshot.session_id,),
        )

    reopened_repository = _repository(tmp_path)
    with pytest.raises(ImportReceiptConflictError, match="exactly match"):
        reopened_repository.aggregate_receipt(_APPROVAL_ID)
    with pytest.raises(ImportReceiptConflictError, match="exactly match"):
        reopened_repository.transition_session(
            _APPROVAL_ID,
            ImportSessionState.COMPLETED,
        )
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).state
        is ImportSessionState.RUNNING
    )


def test_receipt_keeps_note_unit_pending_until_all_required_effects_apply(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = snapshot.payload_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
        ),
    )

    receipt = repository.aggregate_receipt(_APPROVAL_ID)

    assert (
        receipt.total,
        receipt.completed,
        receipt.imported,
        receipt.updated,
        receipt.failed,
    ) == (1, 0, 0, 0, 0)


def test_terminal_item_summary_inconsistent_with_effects_fails_closed(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(
        _approved_for_item(_create_item_with_payloads()),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    with pytest.raises(ImportReceiptError, match="inconsistent"):
        repository.transition_item(
            _APPROVAL_ID,
            "multi-create",
            ImportItemOutcome.FAILED,
            reason_code="database_busy",
            retryable=True,
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.RUNNING
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING


def _apply_complete_update_effects(
    repository: NoteImportReceiptRepository,
) -> None:
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in snapshot.folder_effects)
        + (
            EffectTransition(
                category=snapshot.payload_effects[0].category,
                effect_id=snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            _applied_transition(
                snapshot.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )


def test_applied_update_can_end_in_permanent_item_reconciliation_conflict(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    _apply_complete_update_effects(repository)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]
    annotated = repository.annotate_applied_payload_reconciliation_conflict(
        _APPROVAL_ID,
        effect_id=payload.effect_id,
    )

    item = repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.FAILED,
        reason_code="note_conflict",
        retryable=False,
    )

    receipt = repository.aggregate_receipt(_APPROVAL_ID)
    durable = repository.load_session_snapshot(_APPROVAL_ID)
    assert item.outcome is ImportItemOutcome.FAILED
    assert (receipt.updated, receipt.failed, receipt.retryable) == (0, 1, 0)
    assert annotated.state is ImportEffectState.APPLIED
    assert annotated.reason_code == "note_conflict"
    assert annotated.retryable is False
    assert durable.payload_effects[0].state is ImportEffectState.APPLIED
    assert durable.payload_effects[0].reason_code == "note_conflict"


def test_ordinary_effect_transition_cannot_annotate_applied_payload(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]

    with pytest.raises(ValueError, match="Only failed rows"):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                    reason_code="note_conflict",
                ),
            ),
        )
    with pytest.raises(ImportReceiptTransitionError, match="exact applied payload"):
        repository.annotate_applied_payload_reconciliation_conflict(
            _APPROVAL_ID,
            effect_id=payload.effect_id,
        )


@pytest.mark.parametrize(
    ("reason_code", "retryable"),
    [
        ("database_busy", False),
        ("note_conflict", True),
    ],
)
def test_applied_update_rejects_other_item_failure_shapes(
    tmp_path: Path,
    reason_code: str,
    retryable: bool,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    _apply_complete_update_effects(repository)

    with pytest.raises(ImportReceiptError, match="inconsistent"):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.FAILED,
            reason_code=reason_code,
            retryable=retryable,
        )

    durable = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert durable.items[0].outcome is ImportItemOutcome.PENDING
    assert durable.payload_effects[0].state is ImportEffectState.APPLIED


def test_exact_state_transition_allowlists_are_closed_and_immutable() -> None:
    assert SESSION_STATE_TRANSITIONS == {
        ImportSessionState.PENDING: frozenset(
            {ImportSessionState.RUNNING, ImportSessionState.CANCELLED}
        ),
        ImportSessionState.RUNNING: frozenset(
            {
                ImportSessionState.CANCELLED,
                ImportSessionState.COMPLETED,
                ImportSessionState.NEEDS_ATTENTION,
            }
        ),
        ImportSessionState.NEEDS_ATTENTION: frozenset(
            {ImportSessionState.RUNNING, ImportSessionState.CANCELLED}
        ),
        ImportSessionState.COMPLETED: frozenset(),
        ImportSessionState.CANCELLED: frozenset(),
    }
    assert ITEM_OUTCOME_TRANSITIONS == {
        ImportItemOutcome.PENDING: frozenset(
            {
                ImportItemOutcome.IMPORTED,
                ImportItemOutcome.UPDATED,
                ImportItemOutcome.SKIPPED,
                ImportItemOutcome.FAILED,
            }
        ),
        ImportItemOutcome.IMPORTED: frozenset(),
        ImportItemOutcome.UPDATED: frozenset(),
        ImportItemOutcome.SKIPPED: frozenset(),
        ImportItemOutcome.FAILED: frozenset(),
    }
    assert EFFECT_STATE_TRANSITIONS == {
        ImportEffectState.PENDING: frozenset(
            {ImportEffectState.APPLIED, ImportEffectState.FAILED}
        ),
        ImportEffectState.APPLIED: frozenset(),
        ImportEffectState.FAILED: frozenset(),
    }
    with pytest.raises(TypeError):
        SESSION_STATE_TRANSITIONS[ImportSessionState.PENDING] = frozenset()  # type: ignore[index]


def test_session_item_and_effect_illegal_transitions_fail_without_mutation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    session = repository.begin(_approved(), batch_size=25)
    durable = repository.load_session_snapshot(session.approval_id)
    effect = durable.payload_effects[0]

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.PENDING,
        )
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=effect.category,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.PENDING,
                ),
            ),
        )
    after = repository.load_session_snapshot(_APPROVAL_ID)
    assert after.state is ImportSessionState.PENDING
    assert after.items[0].outcome is ImportItemOutcome.PENDING
    assert after.payload_effects[0].state is ImportEffectState.PENDING


def test_completion_rejects_pending_failed_or_unapplied_work_and_rolls_back(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.RUNNING
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.UPDATED,
            target_note_id="opaque-note-7",
            observed_version=8,
        )
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.RUNNING
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert any(
        effect.state is ImportEffectState.PENDING
        for effect in (
            *reopened.payload_effects,
            *reopened.folder_effects,
            *reopened.membership_effects,
        )
    )


def test_cancelled_session_rejects_item_effect_and_retry_mutations_durably(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.CANCELLED)
    effect = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.FAILED,
            retryable=True,
            reason_code="cancelled_work",
        )
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=effect.category,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.FAILED,
                    retryable=True,
                    reason_code="cancelled_work",
                ),
            ),
        )
    with pytest.raises(ImportReceiptTransitionError):
        repository.reset_retryable_effect(
            _APPROVAL_ID,
            category=effect.category,
            effect_id=effect.effect_id,
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    receipt = repository.aggregate_receipt(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.CANCELLED
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING
    assert (receipt.state, receipt.total, receipt.completed) == (
        ImportSessionState.CANCELLED,
        1,
        0,
    )


def test_cancelled_resume_requires_exact_authority_and_preserves_the_session(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    approved = _approved()
    original = repository.begin(approved, batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.CANCELLED)
    changed = approve_note_import_plan(
        _plan(content="Different private authority"),
        approval_id=_APPROVAL_ID,
    )

    with pytest.raises(ImportReceiptConflictError):
        repository.resume_cancelled(changed, batch_size=25)
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).state
        is ImportSessionState.CANCELLED
    )

    resumed = _repository(tmp_path).resume_cancelled(approved, batch_size=25)

    assert resumed.session_id == original.session_id
    assert resumed.state is ImportSessionState.RUNNING
    assert resumed.items[0].outcome is ImportItemOutcome.PENDING
    assert resumed.payload_effects[0].state is ImportEffectState.PENDING


@pytest.mark.parametrize(
    ("action", "invalid_outcome"),
    [
        (ImportAction.SKIP, ImportItemOutcome.IMPORTED),
        (ImportAction.SKIP, ImportItemOutcome.UPDATED),
        (ImportAction.SKIP, ImportItemOutcome.FAILED),
        (ImportAction.CREATE_NEW, ImportItemOutcome.UPDATED),
        (ImportAction.UPDATE_EXISTING, ImportItemOutcome.IMPORTED),
    ],
)
def test_item_outcomes_must_match_the_approved_action(
    tmp_path: Path,
    action: ImportAction,
    invalid_outcome: ImportItemOutcome,
) -> None:
    item = _item(selected_action=ImportAction.CREATE_NEW)
    if action is ImportAction.SKIP:
        item = replace(item, selected_action=action, add_membership=False)
    elif action is ImportAction.UPDATE_EXISTING:
        item = _item(selected_action=action)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_item(
            _APPROVAL_ID,
            item.item_id,
            invalid_outcome,
        )
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0].outcome
        is ImportItemOutcome.PENDING
    )


@pytest.mark.parametrize(
    ("target_note_id", "observed_version"),
    [("forbidden-note-id", None), (None, 1)],
)
def test_skipped_transition_metadata_rejects_the_entire_mixed_batch(
    tmp_path: Path,
    target_note_id: str | None,
    observed_version: int | None,
) -> None:
    create_item = _item(
        item_id="create-item",
        selected_action=ImportAction.CREATE_NEW,
    )
    skip_item = replace(
        _item(item_id="skip-item", selected_action=ImportAction.CREATE_NEW),
        selected_action=ImportAction.SKIP,
        add_membership=False,
    )
    approved = approve_note_import_plan(
        replace(_plan(), items=(skip_item, create_item)),
        approval_id=_APPROVAL_ID,
    )
    repository = _repository(tmp_path)
    repository.begin(approved, batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]

    with pytest.raises(ValueError, match="Skip"):
        repository.transition_batch(
            _APPROVAL_ID,
            item_transitions=(
                ItemTransition(
                    item_id=skip_item.item_id,
                    outcome=ImportItemOutcome.SKIPPED,
                    target_note_id=target_note_id,
                    observed_version=observed_version,
                ),
            ),
            effect_transitions=(
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-created-note",
                    observed_version=1,
                ),
            ),
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.items[0].target_note_id is None
    assert reopened.items[0].observed_version is None
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING


@pytest.mark.parametrize(
    "classification",
    [ImportClassification.UNSUPPORTED, ImportClassification.FAILED],
)
def test_non_importable_preview_items_persist_as_mutation_free_skips(
    tmp_path: Path,
    classification: ImportClassification,
) -> None:
    item = ImportPreviewItem(
        item_id=f"{classification.value}-item",
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path=f"Project/{classification.value}.bin",
            source_path=Path(f"/private/alice/Project/{classification.value}.bin"),
        ),
        payloads=(),
        memberships=(),
        classification=classification,
        reason="Not importable and approved to skip.",
        default_action=ImportAction.SKIP,
        selected_action=ImportAction.SKIP,
        allowed_actions=(ImportAction.SKIP,),
        match=None,
        replace_content=False,
        add_membership=False,
    )
    repository = _repository(tmp_path)
    repository.begin(
        _approved_for_item(item, proposed_folder_paths=()),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    repository.transition_item(
        _APPROVAL_ID,
        item.item_id,
        ImportItemOutcome.SKIPPED,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)

    snapshot = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    receipt = repository.aggregate_receipt(_APPROVAL_ID)

    assert snapshot.payload_effects == ()
    assert snapshot.folder_effects == ()
    assert snapshot.membership_effects == ()
    assert (receipt.total, receipt.completed, receipt.skipped) == (1, 1, 1)


def test_multi_item_effect_transition_is_atomic_when_one_row_is_invalid(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    effects = repository.load_session_snapshot(_APPROVAL_ID).folder_effects
    assert len(effects) == 2

    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=effects[1].category,
                effect_id=effects[1].effect_id,
                state=ImportEffectState.APPLIED,
                target_folder_id="opaque-folder-2",
            ),
        ),
    )
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_batch(
            _APPROVAL_ID,
            item_transitions=(
                ItemTransition(
                    item_id="item-1",
                    outcome=ImportItemOutcome.UPDATED,
                ),
            ),
            effect_transitions=(
                EffectTransition(
                    category=effects[0].category,
                    effect_id=effects[0].effect_id,
                    state=ImportEffectState.APPLIED,
                    target_folder_id="opaque-folder-1",
                ),
                EffectTransition(
                    category=effects[1].category,
                    effect_id=effects[1].effect_id,
                    state=ImportEffectState.FAILED,
                    reason_code="folder_conflict",
                ),
            ),
        )

    after = repository.load_session_snapshot(_APPROVAL_ID)
    assert after.items[0].outcome is ImportItemOutcome.PENDING
    assert [effect.state for effect in after.folder_effects] == [
        ImportEffectState.PENDING,
        ImportEffectState.APPLIED,
    ]


def test_failed_effect_retry_requires_explicit_reset_and_safe_reason_codes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    effect = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=effect.category,
                effect_id=effect.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
            ),
        ),
    )

    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    category=effect.category,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.APPLIED,
                ),
            ),
        )
    reset = repository.reset_retryable_effect(
        _APPROVAL_ID,
        category=effect.category,
        effect_id=effect.effect_id,
    )
    assert reset.state is ImportEffectState.PENDING
    assert reset.reason_code is None
    assert reset.retryable is False
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    with pytest.raises(ValueError) as caught:
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.FAILED,
            reason_code=_RAW_EXCEPTION,
            retryable=True,
        )
    assert _RAW_EXCEPTION not in str(caught.value)


def test_shared_failed_folder_marks_each_dependent_note_unit_and_reset_restores_pending(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(
        _approved_for_item(_create_item_with_payloads(payload_count=2)),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    folder = repository.load_session_snapshot(_APPROVAL_ID).folder_effects[-1]
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=ImportEffectCategory.FOLDER,
                effect_id=folder.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
            ),
        ),
    )

    failed = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (failed.total, failed.failed, failed.completed, failed.retryable) == (
        2,
        2,
        2,
        2,
    )

    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    repository.reset_retryable_effect(
        _APPROVAL_ID,
        category=ImportEffectCategory.FOLDER,
        effect_id=folder.effect_id,
    )
    pending = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (pending.total, pending.failed, pending.completed, pending.retryable) == (
        2,
        0,
        0,
        0,
    )


def test_failed_root_folder_propagates_across_descendants_and_resets_items_first(
    tmp_path: Path,
) -> None:
    items = (
        _create_item_in_folder(item_id="alpha", leaf="Alpha"),
        _create_item_in_folder(item_id="beta", leaf="Beta"),
    )
    plan = replace(
        _plan(),
        items=items,
        proposed_folder_paths=(
            ("Imported Project",),
            ("Imported Project", "Alpha"),
            ("Imported Project", "Beta"),
        ),
    )
    repository = _repository(tmp_path)
    repository.begin(
        approve_note_import_plan(plan, approval_id=_APPROVAL_ID),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    root = repository.load_session_snapshot(_APPROVAL_ID).folder_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=ImportEffectCategory.FOLDER,
                effect_id=root.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
            ),
        ),
    )
    failed = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (failed.total, failed.failed, failed.retryable) == (2, 2, 2)

    for item in items:
        repository.transition_item(
            _APPROVAL_ID,
            item.item_id,
            ImportItemOutcome.FAILED,
            reason_code="database_busy",
            retryable=True,
        )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    with pytest.raises(ImportReceiptTransitionError, match="dependent item"):
        repository.reset_retryable_effect(
            _APPROVAL_ID,
            category=ImportEffectCategory.FOLDER,
            effect_id=root.effect_id,
        )

    repository.reset_retryable_item(_APPROVAL_ID, item_id="alpha")
    interrupted = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (interrupted.failed, interrupted.retryable) == (2, 2)
    _repository(tmp_path).reset_retryable_item(_APPROVAL_ID, item_id="beta")
    before_folder_reset = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (before_folder_reset.failed, before_folder_reset.retryable) == (2, 2)

    _repository(tmp_path).reset_retryable_effect(
        _APPROVAL_ID,
        category=ImportEffectCategory.FOLDER,
        effect_id=root.effect_id,
    )
    pending = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (pending.failed, pending.completed, pending.retryable) == (0, 0, 0)


def test_shared_folder_transition_respects_a_lowered_sqlite_variable_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = tuple(
        _create_item_in_folder(item_id=f"item-{index}", leaf="Shared")
        for index in range(45)
    )
    plan = replace(
        _plan(),
        items=items,
        proposed_folder_paths=(
            ("Imported Project",),
            ("Imported Project", "Shared"),
        ),
    )
    repository = _repository(tmp_path)
    repository.begin(
        approve_note_import_plan(plan, approval_id=_APPROVAL_ID), batch_size=25
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    root = repository.load_session_snapshot(_APPROVAL_ID).folder_effects[0]
    original_connect = repository._connect

    def limited_connect():
        connection = original_connect()
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 40)
        return connection

    monkeypatch.setattr(repository, "_connect", limited_connect)
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=ImportEffectCategory.FOLDER,
                effect_id=root.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
            ),
        ),
    )

    assert _repository(tmp_path).aggregate_receipt(_APPROVAL_ID).failed == 45


def test_item_retry_preserves_target_but_accepts_a_fresh_observed_version(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    initial_snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    payload = initial_snapshot.payload_effects[0]

    with pytest.raises(ImportReceiptConflictError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.FAILED,
            reason_code="version_conflict",
            retryable=True,
            target_note_id="substituted-note",
            observed_version=7,
        )
    original = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert original.target_note_id == "opaque-note-7"
    assert original.observed_version is None
    assert original.outcome is ImportItemOutcome.PENDING

    failed_snapshot = repository.transition_batch(
        _APPROVAL_ID,
        item_transitions=(
            ItemTransition(
                item_id="item-1",
                outcome=ImportItemOutcome.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                observed_version=7,
            ),
        ),
        effect_transitions=(
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                observed_version=7,
            ),
        ),
    )
    failed = failed_snapshot.items[0]
    assert (failed.target_note_id, failed.observed_version) == ("opaque-note-7", 7)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    reset = repository.reset_retryable_item(_APPROVAL_ID, item_id="item-1")
    assert (reset.target_note_id, reset.observed_version) == ("opaque-note-7", None)
    reset_payload = repository.reset_retryable_effect(
        _APPROVAL_ID,
        category=payload.category,
        effect_id=payload.effect_id,
    )
    assert (reset_payload.target_note_id, reset_payload.observed_version) == (
        "opaque-note-7",
        None,
    )
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert (reopened.target_note_id, reopened.observed_version) == (
        "opaque-note-7",
        None,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptConflictError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.UPDATED,
            target_note_id="substituted-note",
            observed_version=8,
        )
    unchanged = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert unchanged.outcome is ImportItemOutcome.PENDING
    assert (unchanged.target_note_id, unchanged.observed_version) == (
        "opaque-note-7",
        None,
    )
    retry_snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    repository.transition_effects(
        _APPROVAL_ID,
        tuple(_applied_transition(effect) for effect in retry_snapshot.folder_effects)
        + (
            EffectTransition(
                category=retry_snapshot.payload_effects[0].category,
                effect_id=retry_snapshot.payload_effects[0].effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            _applied_transition(
                retry_snapshot.membership_effects[0],
                note_id="opaque-note-7",
            ),
        ),
    )
    applied = repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.UPDATED,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    assert (applied.target_note_id, applied.observed_version) == (
        "opaque-note-7",
        8,
    )


def test_item_owned_effect_retry_requires_parent_reset_first(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]
    repository.transition_batch(
        _APPROVAL_ID,
        item_transitions=(
            ItemTransition(
                item_id="item-1",
                outcome=ImportItemOutcome.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                observed_version=7,
            ),
        ),
        effect_transitions=(
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
        ),
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    before_reset = repository.aggregate_receipt(_APPROVAL_ID)

    with pytest.raises(ImportReceiptTransitionError, match="parent"):
        repository.reset_retryable_effect(
            _APPROVAL_ID,
            category=payload.category,
            effect_id=payload.effect_id,
        )
    unchanged = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert unchanged.items[0].outcome is ImportItemOutcome.FAILED
    assert unchanged.payload_effects[0].state is ImportEffectState.FAILED
    assert _repository(tmp_path).aggregate_receipt(_APPROVAL_ID) == before_reset

    parent_reset = repository.reset_retryable_item(
        _APPROVAL_ID,
        item_id="item-1",
    )
    parent_pending_receipt = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert parent_reset.observed_version is None
    assert (
        parent_pending_receipt.completed,
        parent_pending_receipt.failed,
        parent_pending_receipt.retryable,
    ) == (1, 1, 1)

    child_reset = repository.reset_retryable_effect(
        _APPROVAL_ID,
        category=payload.category,
        effect_id=payload.effect_id,
    )
    both_pending_receipt = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert child_reset.target_note_id == "opaque-note-7"
    assert child_reset.observed_version is None
    assert (
        both_pending_receipt.completed,
        both_pending_receipt.failed,
        both_pending_receipt.retryable,
    ) == (0, 0, 0)


def test_effect_bindings_survive_retry_and_mixed_conflict_rolls_back_after_reopen(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    durable = repository.load_session_snapshot(_APPROVAL_ID)
    payload = durable.payload_effects[0]
    membership = durable.membership_effects[0]
    folder = next(
        effect
        for effect in durable.folder_effects
        if effect.folder_path_digest == membership.folder_path_digest
    )
    ancestor = next(
        effect
        for effect in durable.folder_effects
        if effect.effect_id == folder.parent_effect_id
    )
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                observed_version=7,
            ),
            EffectTransition(
                category=ancestor.category,
                effect_id=ancestor.effect_id,
                state=ImportEffectState.APPLIED,
                target_folder_id="opaque-root-folder",
            ),
            EffectTransition(
                category=folder.category,
                effect_id=folder.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_folder_id="opaque-effect-folder",
            ),
            EffectTransition(
                category=membership.category,
                effect_id=membership.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                target_folder_id="opaque-effect-folder",
            ),
        ),
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    for effect in (payload, folder, membership):
        repository.reset_retryable_effect(
            _APPROVAL_ID,
            category=effect.category,
            effect_id=effect.effect_id,
        )
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    rebound = {
        effect.category: effect
        for effect in (
            reopened.payload_effects[0],
            next(
                candidate
                for candidate in reopened.folder_effects
                if candidate.effect_id == folder.effect_id
            ),
            reopened.membership_effects[0],
        )
    }
    assert (
        rebound[payload.category].target_note_id,
        rebound[payload.category].observed_version,
    ) == ("opaque-note-7", None)
    assert rebound[folder.category].target_folder_id == "opaque-effect-folder"
    assert (
        rebound[membership.category].target_note_id,
        rebound[membership.category].target_folder_id,
    ) == ("opaque-note-7", "opaque-effect-folder")
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptConflictError):
        repository.transition_batch(
            _APPROVAL_ID,
            item_transitions=(
                ItemTransition(
                    item_id="item-1",
                    outcome=ImportItemOutcome.UPDATED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                ),
            ),
            effect_transitions=(
                EffectTransition(
                    category=payload.category,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                ),
                EffectTransition(
                    category=folder.category,
                    effect_id=folder.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_folder_id="substituted-folder",
                ),
            ),
        )
    rolled_back = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert rolled_back.items[0].outcome is ImportItemOutcome.PENDING
    assert rolled_back.payload_effects[0].state is ImportEffectState.PENDING
    assert (
        next(
            effect
            for effect in rolled_back.folder_effects
            if effect.effect_id == folder.effect_id
        ).state
        is ImportEffectState.PENDING
    )
    applied = repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            EffectTransition(
                category=folder.category,
                effect_id=folder.effect_id,
                state=ImportEffectState.APPLIED,
                target_folder_id="opaque-effect-folder",
            ),
            EffectTransition(
                category=membership.category,
                effect_id=membership.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                target_folder_id="opaque-effect-folder",
            ),
        ),
    )
    assert all(effect.state is ImportEffectState.APPLIED for effect in applied)
    assert applied[0].observed_version == 8


@pytest.mark.parametrize(
    "transition",
    [
        EffectTransition(
            category=ImportEffectCategory.PAYLOAD,
            effect_id="opaque-effect",
            state=ImportEffectState.APPLIED,
        ),
        EffectTransition(
            category=ImportEffectCategory.FOLDER,
            effect_id="opaque-effect",
            state=ImportEffectState.APPLIED,
        ),
        EffectTransition(
            category=ImportEffectCategory.MEMBERSHIP,
            effect_id="opaque-effect",
            state=ImportEffectState.APPLIED,
        ),
    ],
)
def test_applied_effects_require_their_reconciliation_identities(
    tmp_path: Path,
    transition: EffectTransition,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    effects = {
        ImportEffectCategory.PAYLOAD: snapshot.payload_effects[0],
        ImportEffectCategory.FOLDER: snapshot.folder_effects[0],
        ImportEffectCategory.MEMBERSHIP: snapshot.membership_effects[0],
    }
    seeded = effects[transition.category]

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_effects(
            _APPROVAL_ID,
            (replace(transition, effect_id=seeded.effect_id),),
        )
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).state
        is ImportSessionState.RUNNING
    )


def test_transition_collection_ceiling_fails_before_database_open(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    repository = NoteImportReceiptRepository(database)
    transition = EffectTransition(
        category=ImportEffectCategory.PAYLOAD,
        effect_id="opaque-effect",
        state=ImportEffectState.FAILED,
        reason_code="database_busy",
        retryable=True,
    )

    with pytest.raises(ValueError, match="ceiling"):
        repository.transition_effects(
            _APPROVAL_ID,
            islice(repeat(transition), 100_001),
        )
    assert not database.exists()


def test_hostile_transition_iterator_is_sanitized_before_database_open(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    repository = NoteImportReceiptRepository(database)
    secret = "hostile iterator detail must not leak"
    transition = EffectTransition(
        category=ImportEffectCategory.PAYLOAD,
        effect_id="opaque-effect",
        state=ImportEffectState.FAILED,
        reason_code="database_busy",
        retryable=True,
    )

    def hostile_transitions():
        yield transition
        raise RuntimeError(secret)

    with pytest.raises(ValueError, match="ceiling") as caught:
        repository.transition_effects(_APPROVAL_ID, hostile_transitions())
    assert secret not in str(caught.value)
    assert caught.value.__context__ is None
    assert not database.exists()


def test_begin_sanitizes_stateful_private_canonicalization_failure(
    tmp_path: Path,
) -> None:
    secret = "hostile source and content must not leak"
    approved = _approved()
    concrete_path_type = type(Path("/"))

    class StatefulHostilePath(concrete_path_type):  # type: ignore[misc,valid-type]
        def __str__(self) -> str:
            raise RuntimeError(secret)

    object.__setattr__(
        approved.plan.items[0].source,
        "source_path",
        StatefulHostilePath("/private/alice/hostile.json"),
    )

    with pytest.raises(ImportReceiptError) as caught:
        _repository(tmp_path).begin(approved, batch_size=25)
    assert secret not in str(caught.value)
    assert caught.value.__context__ is None
    database = tmp_path / "notes-sync.sqlite3"
    assert not database.exists()


def test_aggregate_receipt_uses_frozen_projection_and_hides_private_values(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    payload = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                category=payload.category,
                effect_id=payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="version_conflict",
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
        ),
    )
    repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.FAILED,
        reason_code="version_conflict",
        retryable=False,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)

    receipt = repository.aggregate_receipt(_APPROVAL_ID)

    assert receipt.failed == 1
    assert receipt.reason_code == "version_conflict"
    assert receipt.to_diagnostic().failed == 1
    for rendered in (repr(repository), repr(receipt), repr(receipt.to_diagnostic())):
        assert _APPROVAL_ID not in rendered
        assert "opaque-note-7" not in rendered
        assert _PRIVATE_BODY not in rendered
        assert _PRIVATE_SOURCE.as_posix() not in rendered


def test_public_repository_methods_return_frozen_models_not_sqlite_rows(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    session = repository.begin(_approved(), batch_size=25)
    durable = repository.load_session_snapshot(_APPROVAL_ID)
    transitioned = repository.transition_session(
        _APPROVAL_ID, ImportSessionState.RUNNING
    )

    assert not isinstance(session, sqlite3.Row)
    assert not isinstance(durable, sqlite3.Row)
    assert not isinstance(transitioned, sqlite3.Row)
    with pytest.raises(FrozenInstanceError):
        session.batch_size = 50  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        durable.items[0].outcome = ImportItemOutcome.SKIPPED  # type: ignore[misc]
    public_methods = {
        name
        for name, _method in inspect.getmembers(
            NoteImportReceiptRepository, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert {
        "begin",
        "get_session",
        "load_session_snapshot",
        "transition_session",
        "transition_item",
        "transition_effects",
        "reset_retryable_effect",
        "aggregate_receipt",
    } <= public_methods


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-file contract")
def test_receipt_database_and_parent_have_private_modes(tmp_path: Path) -> None:
    database = tmp_path / "private-parent" / "notes-sync.sqlite3"
    database.parent.mkdir(mode=0o700)
    database.parent.chmod(0o700)

    NoteImportReceiptRepository(database).begin(_approved(), batch_size=25)

    assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(database.stat().st_mode) == 0o600
