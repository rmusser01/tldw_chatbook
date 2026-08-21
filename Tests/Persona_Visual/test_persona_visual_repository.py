"""Real-SQLite tests for the immutable Persona Visual graph repository."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.contracts import ALLOWED_ASSET_ROLES
from tldw_chatbook.Persona_Visual.repository import (
    PersonaVisualGraph,
    PersonaVisualIdentity,
    PersonaVisualRepository,
)


def _manifest(asset_key: str = "idle", *, marker: str = "v1") -> dict[str, Any]:
    return {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {
            "idle": {"animation_id": "idle"},
            "listening": {"animation_id": "idle"},
            "thinking": {"animation_id": "idle"},
            "speaking": {"animation_id": "idle"},
            "error": {"animation_id": "idle"},
        },
        "animations": {
            "idle": {
                "frames": [{"asset_id": asset_key}],
                "preview_asset_id": asset_key,
            }
        },
        "state_catalog": {},
        "fallbacks": {},
        "authored_triggers": [],
        "marker": marker,
    }


def _valid_manifest(asset_key: str = "idle", *, frame_rate: int = 1) -> dict[str, Any]:
    manifest = _manifest(asset_key)
    manifest.pop("marker")
    manifest["animations"]["idle"]["frame_rate"] = frame_rate
    return manifest


def _asset(
    asset_key: str = "idle",
    *,
    role: str = "frame",
    sha256: str = "b" * 64,
    storage_relpath: str = "persona_visual/pack/v1/idle.png",
) -> dict[str, Any]:
    return {
        "asset_key": asset_key,
        "role": role,
        "storage_relpath": storage_relpath,
        "mime_type": "image/png",
        "bytes": 12,
        "sha256": sha256,
        "width": 4,
        "height": 5,
        "frame_count": 1,
        "duration_ms": None,
    }


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "persona-visual.db", client_id="repository")
    yield database
    database.close_connection()


@pytest.fixture
def repository(db: CharactersRAGDB) -> PersonaVisualRepository:
    return PersonaVisualRepository(db)


def _activate(
    repository: PersonaVisualRepository,
    *,
    persona_id: str = "persona-local-1",
    persona_revision: int = 7,
    role: str = "frame",
) -> PersonaVisualGraph:
    return repository.activate_new_pack(
        persona_id=persona_id,
        title="Operator states",
        description="Local-only runtime art",
        source_kind="manual",
        source_context={"provenance": "workbench"},
        manifest=_valid_manifest(),
        manifest_storage_relpath="persona_visual/pack/v1/manifest.json",
        assets=[_asset(role=role)],
        expected_persona_revision=persona_revision,
        authority_guard=lambda: True,
    )


@pytest.mark.parametrize("role", ALLOWED_ASSET_ROLES)
def test_each_pinned_asset_role_can_activate_and_read(
    repository: PersonaVisualRepository,
    role: str,
) -> None:
    graph = _activate(repository, role=role)

    assert graph.assets[0].role == role
    assert repository.get_active_persona_pack(graph.identity.persona_id) == graph


@pytest.mark.parametrize("role", ALLOWED_ASSET_ROLES)
def test_each_pinned_asset_role_can_publish_and_read(
    repository: PersonaVisualRepository,
    role: str,
) -> None:
    active = _activate(repository)
    published = repository.publish_version(
        persona_id=active.identity.persona_id,
        manifest=_valid_manifest(frame_rate=2),
        manifest_storage_relpath="persona_visual/roles/v2/manifest.json",
        assets=[_asset(role=role, storage_relpath="persona_visual/roles/v2/idle.png")],
        expected_identity=active.identity,
        expected_persona_revision=active.identity.persona_revision,
        authority_guard=lambda: True,
    )

    assert published.assets[0].role == role
    assert repository.get_active_persona_pack(active.identity.persona_id) == published


@pytest.mark.parametrize("role", ("sprite", "unknown"))
def test_unpinned_asset_roles_are_rejected_on_write(
    repository: PersonaVisualRepository,
    role: str,
) -> None:
    with pytest.raises(ValueError, match="^persona_visual_asset_invalid$"):
        _activate(repository, role=role)


@pytest.mark.parametrize(
    "persona_id",
    (
        "Ada Lovelace / local",
        "ペルソナ-éclair",
        "actor?! @local#1",
        "x" * 200,
    ),
)
def test_persona_id_matches_authoritative_local_profile_boundary(
    repository: PersonaVisualRepository,
    persona_id: str,
) -> None:
    graph = _activate(repository, persona_id=persona_id)

    assert graph.identity.persona_id == persona_id
    assert repository.get_active_persona_pack(persona_id) == graph


def test_persona_id_rejects_more_than_200_characters(
    repository: PersonaVisualRepository,
) -> None:
    with pytest.raises(ValueError, match="^persona_visual_persona_id_invalid$"):
        _activate(repository, persona_id="x" * 201)


def test_activate_new_pack_returns_immutable_path_safe_graph(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    shared_counts_before = tuple(
        db.get_connection()
        .execute(
            """
            SELECT
                (SELECT COUNT(*) FROM visual_identity_packs),
                (SELECT COUNT(*) FROM visual_identity_pack_versions),
                (SELECT COUNT(*) FROM visual_identity_assets),
                (SELECT COUNT(*) FROM visual_identity_bindings)
            """
        )
        .fetchone()
    )

    graph = _activate(repository)

    assert graph == repository.get_active_persona_pack("persona-local-1")
    assert graph.identity == PersonaVisualIdentity(
        persona_id="persona-local-1",
        persona_revision=7,
        binding_id=graph.binding.id,
        binding_version=1,
        pack_id=graph.pack.id,
        pack_revision=1,
        pack_version_id=graph.version.id,
        version_number=1,
        manifest_sha256=_canonical_digest(_valid_manifest()),
    )
    assert tuple(field.name for field in fields(PersonaVisualIdentity)) == (
        "persona_id",
        "persona_revision",
        "binding_id",
        "binding_version",
        "pack_id",
        "pack_revision",
        "pack_version_id",
        "version_number",
        "manifest_sha256",
    )
    assert not any("path" in field.name for field in fields(PersonaVisualIdentity))
    assert not any(
        "path" in field.name or "storage" in field.name
        for record in (graph.pack, graph.version, graph.assets[0], graph.binding)
        for field in fields(record)
    )
    assert not hasattr(graph.pack, "source_context")

    shared_counts_after = tuple(
        db.get_connection()
        .execute(
            """
            SELECT
                (SELECT COUNT(*) FROM visual_identity_packs),
                (SELECT COUNT(*) FROM visual_identity_pack_versions),
                (SELECT COUNT(*) FROM visual_identity_assets),
                (SELECT COUNT(*) FROM visual_identity_bindings)
            """
        )
        .fetchone()
    )
    assert shared_counts_after == shared_counts_before


def test_private_storage_lookup_requires_exact_active_graph_and_asset(
    repository: PersonaVisualRepository,
) -> None:
    graph = _activate(repository)

    storage_key = repository._get_active_asset_storage_key(
        graph.identity, graph.assets[0]
    )

    assert storage_key == "persona_visual/pack/v1/idle.png"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("persona_revision", 8),
        ("binding_version", 2),
        ("pack_revision", 2),
        ("pack_version_id", 999),
        ("version_number", 2),
        ("manifest_sha256", "c" * 64),
    ],
)
def test_private_storage_lookup_refuses_stale_full_identity(
    repository: PersonaVisualRepository,
    field: str,
    value: object,
) -> None:
    graph = _activate(repository)

    with pytest.raises(ValueError, match="^persona_visual_asset_storage_unavailable$"):
        repository._get_active_asset_storage_key(
            replace(graph.identity, **{field: value}), graph.assets[0]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", 999),
        ("asset_key", "changed"),
        ("role", "preview"),
        ("mime_type", "image/gif"),
        ("byte_count", 13),
        ("sha256", "c" * 64),
        ("width", 6),
        ("height", 7),
        ("frame_count", 2),
        ("duration_ms", 100),
        ("created_at", "2026-08-20 20:00:00"),
    ],
)
def test_private_storage_lookup_refuses_changed_asset_metadata(
    repository: PersonaVisualRepository,
    field: str,
    value: object,
) -> None:
    graph = _activate(repository)

    with pytest.raises(ValueError, match="^persona_visual_asset_storage_unavailable$"):
        repository._get_active_asset_storage_key(
            graph.identity, replace(graph.assets[0], **{field: value})
        )


def test_private_storage_lookup_uses_stable_repository_read_failure(
    repository: PersonaVisualRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _activate(repository)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("private database detail")

    monkeypatch.setattr(repository.db, "execute_query", fail)
    with pytest.raises(ValueError, match="^persona_visual_repository_read_failed$"):
        repository._get_active_asset_storage_key(graph.identity, graph.assets[0])


def test_publish_version_preserves_old_rows_and_advances_full_identity(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    first = _activate(repository)
    old_version_row = tuple(
        db.get_connection()
        .execute(
            "SELECT * FROM persona_visual_pack_versions WHERE id = ?",
            (first.version.id,),
        )
        .fetchone()
    )
    old_asset_row = tuple(
        db.get_connection()
        .execute(
            "SELECT * FROM persona_visual_assets WHERE id = ?",
            (first.assets[0].id,),
        )
        .fetchone()
    )
    next_manifest = _valid_manifest(frame_rate=2)

    second = repository.publish_version(
        persona_id="persona-local-1",
        manifest=next_manifest,
        manifest_storage_relpath="persona_visual/pack/v2/manifest.json",
        assets=[
            _asset(
                sha256="c" * 64,
                storage_relpath="persona_visual/pack/v2/idle.png",
            )
        ],
        expected_identity=first.identity,
        expected_persona_revision=7,
        authority_guard=lambda: True,
    )

    assert second.identity == PersonaVisualIdentity(
        persona_id="persona-local-1",
        persona_revision=7,
        binding_id=first.identity.binding_id,
        binding_version=2,
        pack_id=first.identity.pack_id,
        pack_revision=2,
        pack_version_id=second.version.id,
        version_number=2,
        manifest_sha256=_canonical_digest(next_manifest),
    )
    assert (
        tuple(
            db.get_connection()
            .execute(
                "SELECT * FROM persona_visual_pack_versions WHERE id = ?",
                (first.version.id,),
            )
            .fetchone()
        )
        == old_version_row
    )
    assert (
        tuple(
            db.get_connection()
            .execute(
                "SELECT * FROM persona_visual_assets WHERE id = ?",
                (first.assets[0].id,),
            )
            .fetchone()
        )
        == old_asset_row
    )
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM persona_visual_pack_versions WHERE pack_id = ?",
            (first.pack.id,),
        )
        .fetchone()[0]
        == 2
    )


def test_only_one_active_binding_and_archived_or_deleted_bindings_are_ignored(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    with pytest.raises(ValueError, match="^persona_visual_binding_changed$"):
        _activate(repository)

    repository.archive_binding(
        persona_id="persona-local-1", expected_identity=active.identity
    )
    assert repository.get_active_persona_pack("persona-local-1") is None
    replacement = _activate(repository, persona_revision=8)
    assert replacement.identity.persona_revision == 8

    db.get_connection().execute(
        "UPDATE persona_visual_bindings SET status = 'deleted' WHERE id = ?",
        (replacement.binding.id,),
    )
    db.get_connection().commit()
    assert repository.get_active_persona_pack("persona-local-1") is None


@pytest.mark.parametrize(
    "field",
    (
        "persona_id",
        "persona_revision",
        "binding_id",
        "binding_version",
        "pack_id",
        "pack_revision",
        "pack_version_id",
        "version_number",
        "manifest_sha256",
    ),
)
def test_publish_rejects_every_stale_identity_component(
    repository: PersonaVisualRepository,
    field: str,
) -> None:
    active = _activate(repository)
    value = getattr(active.identity, field)
    stale_value: object = f"{value}-stale" if isinstance(value, str) else value + 1
    stale = replace(active.identity, **{field: stale_value})

    with pytest.raises(ValueError, match="^persona_visual_identity_changed$"):
        repository.publish_version(
            persona_id="persona-local-1",
            manifest=_valid_manifest(frame_rate=2),
            manifest_storage_relpath="persona_visual/pack/v2/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/pack/v2/idle.png")],
            expected_identity=stale,
            expected_persona_revision=7,
            authority_guard=lambda: True,
        )


def test_publish_rejects_stale_expected_persona_revision(
    repository: PersonaVisualRepository,
) -> None:
    active = _activate(repository)
    with pytest.raises(ValueError, match="^persona_visual_persona_revision_changed$"):
        repository.publish_version(
            persona_id="persona-local-1",
            manifest=_valid_manifest(frame_rate=2),
            manifest_storage_relpath="persona_visual/pack/v2/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/pack/v2/idle.png")],
            expected_identity=active.identity,
            expected_persona_revision=8,
            authority_guard=lambda: True,
        )


def test_source_pack_and_version_must_still_be_active_and_same_pack(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    connection = db.get_connection()
    other_pack = int(
        connection.execute(
            "INSERT INTO persona_visual_packs(title) VALUES ('Other')"
        ).lastrowid
    )
    other_version = int(
        connection.execute(
            """
            INSERT INTO persona_visual_pack_versions(
                pack_id, version_number, renderer_type, manifest_version,
                manifest_json, manifest_sha256, storage_relpath
            ) VALUES (?, 1, 'sprite_frames', 1, ?, ?, 'other/manifest.json')
            """,
            (other_pack, json.dumps(_valid_manifest()), "d" * 64),
        ).lastrowid
    )
    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute(
        "UPDATE persona_visual_packs SET active_version_id = ? WHERE id = ?",
        (other_version, active.pack.id),
    )
    connection.commit()
    connection.execute("PRAGMA foreign_keys = ON")

    with pytest.raises(ValueError, match="^persona_visual_pack_relationship_invalid$"):
        repository.get_active_persona_pack("persona-local-1")


def test_guard_runs_under_write_reservation_immediately_before_activation(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    observed: dict[str, object] = {}

    def guard() -> bool:
        connection = db.get_connection()
        observed["in_transaction"] = connection.in_transaction
        observed["version_rows"] = connection.execute(
            "SELECT COUNT(*) FROM persona_visual_pack_versions WHERE pack_id = ?",
            (active.pack.id,),
        ).fetchone()[0]
        observed["active_version"] = connection.execute(
            "SELECT active_version_id FROM persona_visual_packs WHERE id = ?",
            (active.pack.id,),
        ).fetchone()[0]
        with sqlite3.connect(db.db_path_str, timeout=0.01) as contender:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                contender.execute("BEGIN IMMEDIATE")
        return True

    published = repository.publish_version(
        persona_id="persona-local-1",
        manifest=_valid_manifest(frame_rate=2),
        manifest_storage_relpath="persona_visual/pack/v2/manifest.json",
        assets=[_asset(storage_relpath="persona_visual/pack/v2/idle.png")],
        expected_identity=active.identity,
        expected_persona_revision=7,
        authority_guard=guard,
    )

    assert observed == {
        "in_transaction": True,
        "version_rows": 1,
        "active_version": active.version.id,
    }
    assert published.version.version_number == 2


def test_active_lookup_ignores_malformed_inactive_binding_history(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    connection = db.get_connection()
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(
        """
        INSERT INTO persona_visual_bindings(
            persona_id, persona_revision, pack_id, active_version_id, status
        ) VALUES (?, 'malformed', ?, ?, 'archived')
        """,
        (active.identity.persona_id, active.pack.id, active.version.id),
    )
    connection.commit()

    assert repository.get_active_persona_pack(active.identity.persona_id) == active


def test_locked_write_uses_fixed_repository_category(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    db.get_connection().execute("PRAGMA busy_timeout = 1")
    locker = sqlite3.connect(db.db_path_str)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(
            ValueError, match="^persona_visual_repository_write_failed$"
        ):
            _activate(repository, persona_id="locked / actor")
    finally:
        locker.rollback()
        locker.close()


def test_read_operational_error_uses_fixed_repository_category(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_query(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("private database path and failure detail")

    monkeypatch.setattr(db, "execute_query", fail_query)

    with pytest.raises(ValueError, match="^persona_visual_repository_read_failed$"):
        repository.get_active_persona_pack("Ada Lovelace / local")


class _FailingFetchCursor:
    def __init__(self, method: str, error: sqlite3.Error) -> None:
        self.method = method
        self.error = error

    def fetchone(self) -> None:
        if self.method == "fetchone":
            raise self.error
        raise AssertionError("unexpected fetchone")

    def fetchall(self) -> None:
        if self.method == "fetchall":
            raise self.error
        raise AssertionError("unexpected fetchall")


@pytest.mark.parametrize(
    ("operation", "fetch_method", "query_marker", "failure_detail", "error_code"),
    (
        (
            "read",
            "fetchone",
            "FROM persona_visual_bindings",
            "database is locked",
            sqlite3.SQLITE_LOCKED,
        ),
        (
            "read",
            "fetchall",
            "FROM persona_visual_assets",
            "interrupted",
            sqlite3.SQLITE_INTERRUPT,
        ),
        (
            "write",
            "fetchone",
            "SELECT COALESCE(MAX",
            "interrupted",
            sqlite3.SQLITE_INTERRUPT,
        ),
        (
            "write",
            "fetchall",
            "FROM persona_visual_assets",
            "database is locked",
            sqlite3.SQLITE_LOCKED,
        ),
    ),
)
def test_fetch_operational_errors_use_repository_category(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    fetch_method: str,
    query_marker: str,
    failure_detail: str,
    error_code: int,
) -> None:
    active = (
        _activate(repository)
        if query_marker != "FROM persona_visual_bindings"
        else None
    )
    original_execute = db.execute_query

    def execute_with_failing_fetch(
        query: str, *args: object, **kwargs: object
    ) -> object:
        cursor = original_execute(query, *args, **kwargs)  # type: ignore[arg-type]
        if query_marker in query:
            error = sqlite3.OperationalError(failure_detail)
            error.sqlite_errorcode = error_code
            return _FailingFetchCursor(fetch_method, error)
        return cursor

    monkeypatch.setattr(db, "execute_query", execute_with_failing_fetch)
    category = f"persona_visual_repository_{operation}_failed"
    with pytest.raises(ValueError, match=f"^{category}$"):
        if operation == "read":
            repository.get_active_persona_pack(
                "persona-local-1" if active is not None else "missing"
            )
        else:
            assert active is not None
            repository.publish_version(
                persona_id=active.identity.persona_id,
                manifest=_valid_manifest(frame_rate=2),
                manifest_storage_relpath="persona_visual/fetch/v2/manifest.json",
                assets=[_asset(storage_relpath="persona_visual/fetch/v2/idle.png")],
                expected_identity=active.identity,
                expected_persona_revision=active.identity.persona_revision,
                authority_guard=lambda: True,
            )


@pytest.mark.parametrize("error_code", (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB))
def test_fetch_database_corruption_uses_graph_category(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    error_code: int,
) -> None:
    original_execute = db.execute_query
    corruption = sqlite3.DatabaseError("private malformed database detail")
    corruption.sqlite_errorcode = error_code

    def execute_with_corrupt_fetch(
        query: str, *args: object, **kwargs: object
    ) -> object:
        cursor = original_execute(query, *args, **kwargs)  # type: ignore[arg-type]
        if "FROM persona_visual_bindings" in query:
            return _FailingFetchCursor("fetchone", corruption)
        return cursor

    monkeypatch.setattr(db, "execute_query", execute_with_corrupt_fetch)

    with pytest.raises(ValueError, match="^persona_visual_graph_invalid$"):
        repository.get_active_persona_pack("persona-local-1")


@pytest.mark.parametrize(
    "guard_behavior",
    (
        "false",
        "raise",
        "commit_false",
        "commit_raise",
        "commit_true",
        "rollback_false",
        "rollback_raise",
        "rollback_true",
    ),
)
def test_activate_guard_cannot_escape_repository_transaction(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    guard_behavior: str,
) -> None:
    connection = db.get_connection()

    def guard() -> bool:
        if guard_behavior.startswith("commit"):
            connection.commit()
        elif guard_behavior.startswith("rollback"):
            connection.rollback()
        if guard_behavior.endswith("raise"):
            raise RuntimeError("private guard detail")
        return guard_behavior.endswith("true")

    with pytest.raises(ValueError, match="^persona_visual_authority_changed$"):
        repository.activate_new_pack(
            persona_id="persona-guard-escape",
            title="Guard escape",
            manifest=_valid_manifest(),
            manifest_storage_relpath="persona_visual/guard/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/guard/idle.png")],
            expected_persona_revision=1,
            authority_guard=guard,
        )

    assert tuple(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "persona_visual_packs",
            "persona_visual_pack_versions",
            "persona_visual_assets",
            "persona_visual_bindings",
        )
    ) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "guard_behavior",
    (
        "false",
        "raise",
        "commit_false",
        "commit_raise",
        "commit_true",
        "rollback_false",
        "rollback_raise",
        "rollback_true",
    ),
)
def test_publish_guard_cannot_escape_repository_transaction(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    guard_behavior: str,
) -> None:
    active = _activate(repository)
    connection = db.get_connection()

    def guard() -> bool:
        if guard_behavior.startswith("commit"):
            connection.commit()
        elif guard_behavior.startswith("rollback"):
            connection.rollback()
        if guard_behavior.endswith("raise"):
            raise RuntimeError("private guard detail")
        return guard_behavior.endswith("true")

    with pytest.raises(ValueError, match="^persona_visual_authority_changed$"):
        repository.publish_version(
            persona_id="persona-local-1",
            manifest=_valid_manifest(frame_rate=2),
            manifest_storage_relpath="persona_visual/guard/v2/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/guard/v2/idle.png")],
            expected_identity=active.identity,
            expected_persona_revision=7,
            authority_guard=guard,
        )

    assert (
        connection.execute(
            "SELECT COUNT(*) FROM persona_visual_pack_versions WHERE pack_id = ?",
            (active.pack.id,),
        ).fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM persona_visual_assets WHERE pack_id = ?",
            (active.pack.id,),
        ).fetchone()[0]
        == 1
    )
    assert repository.get_active_persona_pack("persona-local-1") == active


@pytest.mark.parametrize("transaction_sql", ("COMMIT", "ROLLBACK"))
def test_guard_cannot_catch_transaction_denial_and_rebegin(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    transaction_sql: str,
) -> None:
    active = _activate(repository)
    connection = db.get_connection()

    def guard() -> bool:
        for statement in (transaction_sql, "BEGIN IMMEDIATE"):
            try:
                connection.execute(statement)
            except sqlite3.DatabaseError:
                pass
        return True

    with pytest.raises(ValueError, match="^persona_visual_authority_changed$"):
        repository.publish_version(
            persona_id="persona-local-1",
            manifest=_valid_manifest(frame_rate=2),
            manifest_storage_relpath="persona_visual/guard/v2/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/guard/v2/idle.png")],
            expected_identity=active.identity,
            expected_persona_revision=7,
            authority_guard=guard,
        )

    assert repository.get_active_persona_pack("persona-local-1") == active


@pytest.mark.parametrize(
    ("statement", "params"),
    (
        (
            "DELETE FROM persona_visual_assets WHERE id = ?",
            lambda graph: (graph.assets[0].id,),
        ),
        (
            "UPDATE persona_visual_assets SET sha256 = ? WHERE id = ?",
            lambda graph: ("d" * 64, graph.assets[0].id),
        ),
        (
            "UPDATE persona_visual_packs SET title = 'changed' WHERE id = ?",
            lambda graph: (graph.pack.id,),
        ),
        (
            "UPDATE persona_visual_packs SET status = 'archived' WHERE id = ?",
            lambda graph: (graph.pack.id,),
        ),
        (
            "UPDATE persona_visual_packs SET source_context_json = '{}' WHERE id = ?",
            lambda graph: (graph.pack.id,),
        ),
    ),
)
def test_guard_is_read_only_even_when_it_catches_denial(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    statement: str,
    params: Callable[[PersonaVisualGraph], tuple[object, ...]],
) -> None:
    active = _activate(repository)
    connection = db.get_connection()
    before = {
        table: tuple(connection.execute(f"SELECT * FROM {table}").fetchall())
        for table in (
            "persona_visual_packs",
            "persona_visual_pack_versions",
            "persona_visual_assets",
            "persona_visual_bindings",
        )
    }

    def guard() -> bool:
        try:
            connection.execute(statement, params(active))
        except sqlite3.DatabaseError:
            pass
        return True

    with pytest.raises(ValueError, match="^persona_visual_authority_changed$"):
        repository.publish_version(
            persona_id="persona-local-1",
            manifest=_valid_manifest(frame_rate=2),
            manifest_storage_relpath="persona_visual/guard/v2/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/guard/v2/idle.png")],
            expected_identity=active.identity,
            expected_persona_revision=7,
            authority_guard=guard,
        )

    assert {
        table: tuple(connection.execute(f"SELECT * FROM {table}").fetchall())
        for table in before
    } == before


def test_read_only_guard_can_select_and_use_sql_functions(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    connection = db.get_connection()

    def guard() -> bool:
        row = connection.execute(
            "SELECT COUNT(*), LENGTH(title) FROM persona_visual_packs WHERE id = ?",
            (active.pack.id,),
        ).fetchone()
        return tuple(row) == (1, len(active.pack.title))

    published = repository.publish_version(
        persona_id="persona-local-1",
        manifest=_valid_manifest(frame_rate=2),
        manifest_storage_relpath="persona_visual/guard/v2/manifest.json",
        assets=[_asset(storage_relpath="persona_visual/guard/v2/idle.png")],
        expected_identity=active.identity,
        expected_persona_revision=7,
        authority_guard=guard,
    )
    assert published.version.version_number == 2


@pytest.mark.parametrize("guard_behavior", ["false", "raise"])
def test_guard_failure_is_fixed_and_rolls_back_inserted_graph_rows(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    guard_behavior: str,
) -> None:
    active = _activate(repository)
    counts_before = tuple(
        db.get_connection()
        .execute(
            """
            SELECT
                (SELECT COUNT(*) FROM persona_visual_pack_versions),
                (SELECT COUNT(*) FROM persona_visual_assets)
            """
        )
        .fetchone()
    )

    def guard() -> bool:
        if guard_behavior == "raise":
            raise RuntimeError("private path and exception detail")
        return False

    with pytest.raises(ValueError, match="^persona_visual_authority_changed$"):
        repository.publish_version(
            persona_id="persona-local-1",
            manifest=_valid_manifest(frame_rate=2),
            manifest_storage_relpath="persona_visual/private/v2/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/private/v2/idle.png")],
            expected_identity=active.identity,
            expected_persona_revision=7,
            authority_guard=guard,
        )

    counts_after = tuple(
        db.get_connection()
        .execute(
            """
            SELECT
                (SELECT COUNT(*) FROM persona_visual_pack_versions),
                (SELECT COUNT(*) FROM persona_visual_assets)
            """
        )
        .fetchone()
    )
    assert counts_after == counts_before
    assert repository.get_active_persona_pack("persona-local-1") == active


@pytest.mark.parametrize(
    ("column", "changed_value"),
    (
        ("manifest_sha256", "d" * 64),
        ("manifest_json", json.dumps(_valid_manifest(frame_rate=9))),
        ("renderer_type", "live2d"),
        ("manifest_version", 99),
    ),
)
def test_guard_cannot_mutate_the_source_identity_before_activation(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    column: str,
    changed_value: object,
) -> None:
    active = _activate(repository)
    original_value = (
        db.get_connection()
        .execute(
            f"SELECT {column} FROM persona_visual_pack_versions WHERE id = ?",
            (active.version.id,),
        )
        .fetchone()[0]
    )

    def guard() -> bool:
        db.get_connection().execute(
            f"UPDATE persona_visual_pack_versions SET {column} = ? WHERE id = ?",
            (changed_value, active.version.id),
        )
        return True

    with pytest.raises(ValueError, match="^persona_visual_authority_changed$"):
        repository.publish_version(
            persona_id="persona-local-1",
            manifest=_valid_manifest(frame_rate=2),
            manifest_storage_relpath="persona_visual/pack/v2/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/pack/v2/idle.png")],
            expected_identity=active.identity,
            expected_persona_revision=7,
            authority_guard=guard,
        )

    connection = db.get_connection()
    assert (
        connection.execute(
            f"SELECT {column} FROM persona_visual_pack_versions WHERE id = ?",
            (active.version.id,),
        ).fetchone()[0]
        == original_value
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM persona_visual_pack_versions WHERE pack_id = ?",
            (active.pack.id,),
        ).fetchone()[0]
        == 1
    )


def test_late_asset_insert_failure_rolls_back_pack_and_version(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    with pytest.raises(ValueError, match="^persona_visual_repository_write_failed$"):
        repository.activate_new_pack(
            persona_id="persona-local-rollback",
            title="Rollback",
            manifest=_valid_manifest(),
            manifest_storage_relpath="persona_visual/rollback/manifest.json",
            assets=[_asset(), _asset(storage_relpath="different/idle.png")],
            expected_persona_revision=1,
            authority_guard=lambda: True,
        )

    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM persona_visual_packs WHERE title = 'Rollback'"
        ).fetchone()[0]
        == 0
    )
    assert repository.get_active_persona_pack("persona-local-rollback") is None


def test_write_rejects_a_caller_owned_transaction_before_any_insert(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    connection = db.get_connection()
    connection.execute("BEGIN")

    with pytest.raises(ValueError, match="^persona_visual_transaction_active$"):
        repository.activate_new_pack(
            persona_id="persona-borrowed-transaction",
            title="Must not borrow",
            manifest=_valid_manifest(),
            manifest_storage_relpath="persona_visual/borrowed/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/borrowed/idle.png")],
            expected_persona_revision=1,
            authority_guard=lambda: True,
        )

    assert connection.in_transaction is True
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM persona_visual_packs WHERE title = 'Must not borrow'"
        ).fetchone()[0]
        == 0
    )
    connection.rollback()


def test_write_rejects_managed_transaction_after_native_transaction_ended(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    connection = db.get_connection()
    with db.transaction():
        connection.executescript("SELECT 1;")
        assert connection.in_transaction is False

        with pytest.raises(ValueError, match="^persona_visual_transaction_active$"):
            repository.activate_new_pack(
                persona_id="persona-managed-transaction",
                title="Must not nest",
                manifest=_valid_manifest(),
                manifest_storage_relpath="persona_visual/nested/manifest.json",
                assets=[_asset(storage_relpath="persona_visual/nested/idle.png")],
                expected_persona_revision=1,
                authority_guard=lambda: True,
            )

        assert (
            connection.execute(
                "SELECT COUNT(*) FROM persona_visual_packs WHERE title = 'Must not nest'"
            ).fetchone()[0]
            == 0
        )


def test_source_context_rejects_private_data_and_is_redacted_from_sql_logs(
    repository: PersonaVisualRepository,
) -> None:
    for source_context in (
        {"provenance": "imported from /Users/alice/private.png"},
        {"provenance": r"C:\Users\alice\private.png"},
        {"provenance": '{"personas":[{"name":"Alice"}]}'},
        {"provenance": {"persona": "Alice"}},
        {"unknown_neutral_key": "neutral"},
        {"provenance": "\ud800"},
    ):
        with pytest.raises(ValueError, match="^persona_visual_source_context_invalid$"):
            repository.activate_new_pack(
                persona_id="persona-private-context",
                title="Private",
                source_context=source_context,
                manifest=_valid_manifest(),
                manifest_storage_relpath="persona_visual/private/manifest.json",
                assets=[_asset(storage_relpath="persona_visual/private/idle.png")],
                expected_persona_revision=1,
                authority_guard=lambda: True,
            )

    messages: list[str] = []
    sink = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        repository.activate_new_pack(
            persona_id="persona-redacted-context",
            title="Redacted",
            source_context={"provenance": "private-context-marker"},
            manifest=_valid_manifest(),
            manifest_storage_relpath="persona_visual/redacted/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/redacted/idle.png")],
            expected_persona_revision=1,
            authority_guard=lambda: True,
        )
    finally:
        logger.remove(sink)
    assert "private-context-marker" not in "".join(messages)


def test_source_context_accepts_only_bounded_scalar_provenance(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    graph = repository.activate_new_pack(
        persona_id="persona-provenance",
        title="Provenance",
        source_context={
            "source_id": "upstream-42",
            "provenance": "local-import",
            "license": "CC0-1.0",
            "source_server_commit": "abcdef1234",
        },
        manifest=_valid_manifest(),
        manifest_storage_relpath="persona_visual/provenance/manifest.json",
        assets=[_asset(storage_relpath="persona_visual/provenance/idle.png")],
        expected_persona_revision=1,
        authority_guard=lambda: True,
    )
    stored = (
        db.get_connection()
        .execute(
            "SELECT source_context_json FROM persona_visual_packs WHERE id = ?",
            (graph.pack.id,),
        )
        .fetchone()[0]
    )
    assert json.loads(stored) == {
        "source_id": "upstream-42",
        "provenance": "local-import",
        "license": "CC0-1.0",
        "source_server_commit": "abcdef1234",
    }


@pytest.mark.parametrize(
    "asset_key",
    (
        "sprite/idle",
        r"sprite\idle",
        "/absolute",
        r"C:\device",
        "..",
        ".hidden",
        "idle\x00private",
        "idle\ud800",
        "idé",
        "a" * 129,
    ),
)
def test_activate_rejects_pathlike_asset_keys(
    repository: PersonaVisualRepository,
    asset_key: str,
) -> None:
    with pytest.raises(ValueError, match="^persona_visual_asset_invalid$"):
        repository.activate_new_pack(
            persona_id="persona-invalid-asset-key",
            title="Invalid asset key",
            manifest=_valid_manifest(asset_key),
            manifest_storage_relpath="persona_visual/invalid/manifest.json",
            assets=[_asset(asset_key)],
            expected_persona_revision=1,
            authority_guard=lambda: True,
        )


def test_publish_rejects_pathlike_asset_key(
    repository: PersonaVisualRepository,
) -> None:
    active = _activate(repository)
    with pytest.raises(ValueError, match="^persona_visual_asset_invalid$"):
        repository.publish_version(
            persona_id="persona-local-1",
            manifest=_valid_manifest("private/idle"),
            manifest_storage_relpath="persona_visual/invalid/v2/manifest.json",
            assets=[_asset("private/idle")],
            expected_identity=active.identity,
            expected_persona_revision=7,
            authority_guard=lambda: True,
        )


def test_corrupt_stored_pathlike_asset_key_is_never_returned(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    db.get_connection().execute(
        "UPDATE persona_visual_assets SET asset_key = ? WHERE id = ?",
        ("/Users/alice/private.png", active.assets[0].id),
    )
    db.get_connection().commit()

    with pytest.raises(ValueError, match="^persona_visual_graph_invalid$"):
        repository.get_active_persona_pack("persona-local-1")


def test_deep_manifest_mapping_uses_fixed_category(
    repository: PersonaVisualRepository,
) -> None:
    deep: dict[str, object] = {}
    current = deep
    for _ in range(20_000):
        child: dict[str, object] = {}
        current["nested"] = child
        current = child
    manifest = _valid_manifest()
    manifest["deep"] = deep

    with pytest.raises(ValueError, match="^persona_visual_manifest_invalid$"):
        repository.activate_new_pack(
            persona_id="persona-deep-manifest",
            title="Deep manifest",
            manifest=manifest,
            manifest_storage_relpath="persona_visual/deep/manifest.json",
            assets=[_asset(storage_relpath="persona_visual/deep/idle.png")],
            expected_persona_revision=1,
            authority_guard=lambda: True,
        )


def test_deep_stored_source_context_json_uses_fixed_category(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    deep_json = '{"nested":' * 1_200 + "null" + "}" * 1_200
    db.get_connection().execute(
        "UPDATE persona_visual_packs SET source_context_json = ? WHERE id = ?",
        (deep_json, active.pack.id),
    )
    db.get_connection().commit()

    with pytest.raises(ValueError, match="^persona_visual_source_context_invalid$"):
        repository.get_active_persona_pack("persona-local-1")


@pytest.mark.parametrize(
    ("table", "column", "record_name"),
    (
        ("persona_visual_packs", "version", "pack"),
        ("persona_visual_pack_versions", "version_number", "version"),
        ("persona_visual_assets", "width", "asset"),
        ("persona_visual_bindings", "persona_revision", "binding"),
    ),
)
def test_corrupt_stored_numeric_fields_use_fixed_graph_category(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    table: str,
    column: str,
    record_name: str,
) -> None:
    active = _activate(repository)
    record = (
        active.assets[0] if record_name == "asset" else getattr(active, record_name)
    )
    connection = db.get_connection()
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(
        f"UPDATE {table} SET {column} = 'not-an-integer' WHERE id = ?",
        (record.id,),
    )
    connection.commit()

    with pytest.raises(ValueError, match="^persona_visual_graph_invalid$"):
        repository.get_active_persona_pack("persona-local-1")


@pytest.mark.parametrize(
    ("table", "column", "value", "record_name"),
    (
        ("persona_visual_packs", "title", "", "pack"),
        ("persona_visual_packs", "status", "invalid", "pack"),
        ("persona_visual_packs", "source_kind", "", "pack"),
        ("persona_visual_packs", "created_at", "not-a-timestamp", "pack"),
        ("persona_visual_pack_versions", "renderer_type", "live2d", "version"),
        ("persona_visual_pack_versions", "manifest_version", 2, "version"),
        ("persona_visual_pack_versions", "manifest_sha256", "A" * 64, "version"),
        ("persona_visual_assets", "role", "", "asset"),
        ("persona_visual_assets", "role", "sprite", "asset"),
        ("persona_visual_assets", "role", "unknown", "asset"),
        ("persona_visual_assets", "mime_type", "text/plain", "asset"),
        ("persona_visual_assets", "sha256", "A" * 64, "asset"),
        ("persona_visual_assets", "bytes", 0, "asset"),
        ("persona_visual_assets", "width", 4_097, "asset"),
        ("persona_visual_assets", "frame_count", 241, "asset"),
        ("persona_visual_assets", "duration_ms", 30_001, "asset"),
        ("persona_visual_assets", "created_at", "invalid", "asset"),
        ("persona_visual_bindings", "created_at", "invalid", "binding"),
    ),
)
def test_corrupt_stored_domain_fields_use_fixed_graph_category(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    table: str,
    column: str,
    value: object,
    record_name: str,
) -> None:
    active = _activate(repository)
    record = (
        active.assets[0] if record_name == "asset" else getattr(active, record_name)
    )
    connection = db.get_connection()
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(
        f"UPDATE {table} SET {column} = ? WHERE id = ?",
        (value, record.id),
    )
    connection.commit()

    with pytest.raises(ValueError, match="^persona_visual_graph_invalid$"):
        repository.get_active_persona_pack("persona-local-1")


def test_invalid_utf8_text_fetch_uses_fixed_graph_category(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    db.get_connection().execute(
        "UPDATE persona_visual_packs SET title = CAST(x'80' AS TEXT) WHERE id = ?",
        (active.pack.id,),
    )
    db.get_connection().commit()

    with pytest.raises(ValueError, match="^persona_visual_graph_invalid$"):
        repository.get_active_persona_pack("persona-local-1")


@pytest.mark.parametrize(
    ("overrides", "category"),
    (
        ({"manifest": ["not", "an", "object"]}, "persona_visual_manifest_invalid"),
        ({"manifest": {"value": float("nan")}}, "persona_visual_manifest_invalid"),
        (
            {"source_context": ["not", "an", "object"]},
            "persona_visual_source_context_invalid",
        ),
        ({"source_context": []}, "persona_visual_source_context_invalid"),
        (
            {"source_context": {"value": "\ud800"}},
            "persona_visual_source_context_invalid",
        ),
        (
            {"source_context": {"value": float("inf")}},
            "persona_visual_source_context_invalid",
        ),
        (
            {"assets": [{"asset_key": "idle"}]},
            "persona_visual_asset_invalid",
        ),
    ),
)
def test_invalid_json_and_asset_shapes_use_fixed_categories(
    repository: PersonaVisualRepository,
    overrides: dict[str, object],
    category: str,
) -> None:
    kwargs: dict[str, object] = {
        "persona_id": "persona-invalid",
        "title": "Invalid",
        "source_context": {},
        "manifest": _valid_manifest(),
        "manifest_storage_relpath": "persona_visual/invalid/manifest.json",
        "assets": [_asset()],
        "expected_persona_revision": 1,
        "authority_guard": lambda: True,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=f"^{category}$"):
        repository.activate_new_pack(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("column", "value", "category"),
    (
        ("source_context_json", "[]", "persona_visual_source_context_invalid"),
        ("manifest_json", "NaN", "persona_visual_manifest_invalid"),
    ),
)
def test_corrupt_stored_json_is_rejected_without_private_detail(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
    column: str,
    value: str,
    category: str,
) -> None:
    active = _activate(repository)
    table = (
        "persona_visual_packs"
        if column == "source_context_json"
        else "persona_visual_pack_versions"
    )
    row_id = active.pack.id if table.endswith("packs") else active.version.id
    db.get_connection().execute(
        f"UPDATE {table} SET {column} = ? WHERE id = ?", (value, row_id)
    )
    db.get_connection().commit()

    with pytest.raises(ValueError, match=f"^{category}$"):
        repository.get_active_persona_pack("persona-local-1")


def test_stored_manifest_must_match_its_full_identity_digest(
    repository: PersonaVisualRepository,
    db: CharactersRAGDB,
) -> None:
    active = _activate(repository)
    db.get_connection().execute(
        "UPDATE persona_visual_pack_versions SET manifest_json = ? WHERE id = ?",
        (json.dumps(_valid_manifest(frame_rate=2)), active.version.id),
    )
    db.get_connection().commit()

    with pytest.raises(ValueError, match="^persona_visual_manifest_invalid$"):
        repository.get_active_persona_pack("persona-local-1")


def test_archive_binding_uses_full_identity_cas(
    repository: PersonaVisualRepository,
) -> None:
    active = _activate(repository)
    stale = replace(active.identity, pack_revision=active.identity.pack_revision + 1)
    with pytest.raises(ValueError, match="^persona_visual_identity_changed$"):
        repository.archive_binding(
            persona_id="persona-local-1", expected_identity=stale
        )
    assert asdict(
        repository.get_active_persona_pack("persona-local-1").identity
    ) == asdict(active.identity)
