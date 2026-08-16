"""Behavioral coverage for the local Visual Identity repository."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from tldw_chatbook.DB import VisualIdentity_DB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.DB.VisualIdentity_DB import (
    LOCAL_OWNER_ID,
    VisualIdentityRepository,
)


@pytest.fixture
def repository(tmp_path: Path) -> Iterator[VisualIdentityRepository]:
    db = CharactersRAGDB(tmp_path / "visual-identity.db", client_id="repository-test")
    try:
        yield VisualIdentityRepository(db)
    finally:
        db.close_connection()


def _asset(
    original_expression_key: str,
    *,
    expression_key: str | None = None,
    bytes_: int = 10,
    pack_id: int | None = None,
) -> dict[str, Any]:
    filename = f"{original_expression_key}.webp"
    asset = {
        "expression_key": expression_key or original_expression_key,
        "original_expression_key": original_expression_key,
        "display_label": original_expression_key.title(),
        "source_filename": filename,
        "storage_relpath": f"characters/test/{filename}",
        "content_type": "image/webp",
        "bytes": bytes_,
        "sha256": f"sha-{original_expression_key}",
        "width": 1024,
        "height": 1024,
        "source_context": {"fixture": True},
        "is_animated": False,
    }
    if pack_id is not None:
        asset["pack_id"] = pack_id
    return asset


def _activate(
    repository: VisualIdentityRepository,
    *,
    actor_id: str = "42",
    source_kind: str = "user",
    source_id: str = "fixture.pack",
    manifest: Mapping[str, Any] | None = None,
    assets: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return repository.activate_pack(
        pack={
            "owner_user_id": LOCAL_OWNER_ID,
            "title": "Fixture pack",
            "description": "Repository fixture",
            "default_expression_key": "neutral",
            "source_kind": source_kind,
            "source_context": {"source_id": source_id},
        },
        manifest=manifest or {"schema": "test/v1", "source_id": source_id},
        assets=assets or [_asset("neutral")],
        actor_kind="character",
        actor_id=actor_id,
    )


def _seed_pack_version(
    db: CharactersRAGDB, *, title: str, source_kind: str = "user"
) -> tuple[int, int]:
    with db.transaction():
        pack_id = int(
            db.execute_query(
                """
                INSERT INTO visual_identity_packs(
                    owner_user_id, title, source_kind
                ) VALUES (?, ?, ?)
                """,
                (LOCAL_OWNER_ID, title, source_kind),
            ).lastrowid
        )
        version_id = int(
            db.execute_query(
                """
                INSERT INTO visual_identity_pack_versions(
                    pack_id, owner_user_id, version_number, manifest_json
                ) VALUES (?, ?, ?, ?)
                """,
                (pack_id, LOCAL_OWNER_ID, 1, "{}"),
            ).lastrowid
        )
        db.execute_query(
            "UPDATE visual_identity_packs SET active_version_id = ? WHERE id = ?",
            (version_id, pack_id),
        )
    return pack_id, version_id


def _counts(db: CharactersRAGDB) -> dict[str, int]:
    return {
        table: int(db.execute_query(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in (
            "visual_identity_packs",
            "visual_identity_pack_versions",
            "visual_identity_assets",
            "visual_identity_bindings",
        )
    }


def test_local_owner_id_is_named_and_documented_as_non_server_identity() -> None:
    assert LOCAL_OWNER_ID == 0
    documentation = (VisualIdentity_DB.__doc__ or "").lower()
    assert "local-only" in documentation
    assert "server" in documentation


def test_find_pack_by_stable_source_id_respects_deleted_tombstones(
    repository: VisualIdentityRepository,
) -> None:
    activated = _activate(
        repository,
        source_kind="builtin",
        source_id="tldw.builtin.samira.reactions",
    )
    pack_id = activated["pack"]["id"]

    found = repository.find_pack_by_source_id("tldw.builtin.samira.reactions")
    assert found is not None
    assert found["id"] == pack_id

    repository.mark_pack_deleted(pack_id)

    assert repository.find_pack_by_source_id("tldw.builtin.samira.reactions") is None
    tombstone = repository.find_pack_by_source_id(
        "tldw.builtin.samira.reactions", include_deleted=True
    )
    assert tombstone is not None
    assert tombstone["id"] == pack_id
    assert tombstone["status"] == "deleted"


def test_find_pack_by_source_id_reads_one_consistent_transaction(
    repository: VisualIdentityRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    _activate(repository, source_kind="builtin", source_id="fixture.pack")
    original_transaction = repository.db.transaction
    transaction_calls = 0

    def tracked_transaction(*args, **kwargs):
        nonlocal transaction_calls
        transaction_calls += 1
        return original_transaction(*args, **kwargs)

    monkeypatch.setattr(repository.db, "transaction", tracked_transaction)

    assert repository.find_pack_by_source_id("fixture.pack") is not None
    assert transaction_calls == 1


@pytest.mark.parametrize(
    "source_context_json",
    [
        "{bad json",
        "[]",
        '{"source_id":"fixture.pack","value":NaN}',
        '{"source_id":"fixture.pack","value":Infinity}',
    ],
)
def test_find_pack_rejects_invalid_source_context(
    repository: VisualIdentityRepository, source_context_json: str
) -> None:
    activated = _activate(repository, source_kind="builtin")
    with repository.db.transaction():
        repository.db.execute_query(
            "UPDATE visual_identity_packs SET source_context_json = ? WHERE id = ?",
            (source_context_json, activated["pack"]["id"]),
        )

    with pytest.raises(ValueError, match="visual_identity_source_context_invalid"):
        repository.find_pack_by_source_id("fixture.pack")


@pytest.mark.parametrize("corruption", ["null", "missing", "non_owned", "cross_pack"])
def test_find_pack_rejects_matching_pack_with_invalid_active_version(
    repository: VisualIdentityRepository, corruption: str
) -> None:
    activated = _activate(repository, source_kind="builtin")
    pack_id = activated["pack"]["id"]
    bad_version_id = None
    if corruption == "cross_pack":
        _, bad_version_id = _seed_pack_version(repository.db, title="Other")
    elif corruption == "missing":
        bad_version_id = 999_999
    elif corruption == "non_owned":
        with repository.db.transaction():
            other_pack_id = int(
                repository.db.execute_query(
                    """
                    INSERT INTO visual_identity_packs(owner_user_id, title)
                    VALUES (?, ?)
                    """,
                    (99, "Non-owned"),
                ).lastrowid
            )
            bad_version_id = int(
                repository.db.execute_query(
                    """
                    INSERT INTO visual_identity_pack_versions(
                        pack_id, owner_user_id, version_number, manifest_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (other_pack_id, 99, 1, "{}"),
                ).lastrowid
            )

    connection = repository.db.get_connection()
    if corruption == "missing":
        connection.execute("PRAGMA foreign_keys = OFF")
    try:
        with repository.db.transaction():
            repository.db.execute_query(
                "UPDATE visual_identity_packs SET active_version_id = ? WHERE id = ?",
                (bad_version_id, pack_id),
            )
    finally:
        if corruption == "missing":
            connection.execute("PRAGMA foreign_keys = ON")

    with pytest.raises(
        ValueError, match="visual_identity_pack_active_version_mismatch"
    ):
        repository.find_pack_by_source_id("fixture.pack")


def test_get_active_actor_pack_returns_pack_version_and_sorted_live_assets(
    repository: VisualIdentityRepository,
) -> None:
    activated = _activate(
        repository,
        assets=[
            _asset("zeta"),
            _asset("alpha"),
            _asset("alpha", expression_key="custom:alpha"),
        ],
    )
    hidden_asset_id = next(
        asset["id"]
        for asset in activated["assets"]
        if asset["expression_key"] == "custom:alpha"
    )
    with repository.db.transaction():
        repository.db.execute_query(
            "UPDATE visual_identity_assets SET deleted = 1 WHERE id = ?",
            (hidden_asset_id,),
        )

    active = repository.get_active_actor_pack("character", "42")

    assert active is not None
    assert active["binding"]["actor_id"] == "42"
    assert active["pack"]["active_version_id"] == active["version"]["id"]
    assert active["version"]["pack_id"] == active["pack"]["id"]
    assert [asset["original_expression_key"] for asset in active["assets"]] == [
        "alpha",
        "zeta",
    ]
    assert active["assets"] == repository.list_version_assets(active["version"]["id"])


def test_get_active_actor_pack_uses_one_database_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "snapshot.db"
    writer_db = CharactersRAGDB(path, client_id="snapshot-writer")
    reader_db = CharactersRAGDB(path, client_id="snapshot-reader")
    writer = VisualIdentityRepository(writer_db)
    reader = VisualIdentityRepository(reader_db)
    first = _activate(writer)
    old_version_id = first["version"]["id"]
    reached_pack_read = Event()
    continue_pack_read = Event()
    original_execute = reader_db.execute_query
    blocked = False

    def interleaved_execute(query: str, params: Any = None, **kwargs: Any) -> Any:
        nonlocal blocked
        if not blocked and "FROM visual_identity_packs" in query:
            blocked = True
            reached_pack_read.set()
            assert continue_pack_read.wait(5)
        return original_execute(query, params, **kwargs)

    monkeypatch.setattr(reader_db, "execute_query", interleaved_execute)

    def read_active() -> dict[str, Any] | None:
        try:
            return reader.get_active_actor_pack("character", "42")
        finally:
            reader_db.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(read_active)
            assert reached_pack_read.wait(5)
            try:
                published = writer.publish_version(
                    first["pack"]["id"],
                    manifest={"revision": 2},
                    assets=[_asset("neutral")],
                    actor_kind="character",
                    actor_id="42",
                )
            finally:
                continue_pack_read.set()
            observed = future.result(timeout=5)

        assert observed is not None
        assert observed["binding"]["active_version_id"] == old_version_id
        assert observed["pack"]["active_version_id"] == old_version_id
        assert observed["version"]["id"] == old_version_id
        assert published["version"]["id"] != old_version_id
    finally:
        reader_db.close_connection()
        writer_db.close_connection()


def test_get_active_actor_pack_rejects_null_pack_active_version(
    repository: VisualIdentityRepository,
) -> None:
    activated = _activate(repository)
    with repository.db.transaction():
        repository.db.execute_query(
            "UPDATE visual_identity_packs SET active_version_id = ? WHERE id = ?",
            (None, activated["pack"]["id"]),
        )

    with pytest.raises(
        ValueError, match="visual_identity_pack_active_version_mismatch"
    ):
        repository.get_active_actor_pack("character", "42")


def test_activate_pack_creates_the_complete_active_graph_atomically(
    repository: VisualIdentityRepository,
) -> None:
    result = _activate(
        repository,
        assets=[_asset("neutral"), _asset("thinking")],
    )

    assert result["pack"]["active_version_id"] == result["version"]["id"]
    assert result["binding"]["pack_id"] == result["pack"]["id"]
    assert result["binding"]["active_version_id"] == result["version"]["id"]
    assert result["version"]["version_number"] == 1
    assert [asset["original_expression_key"] for asset in result["assets"]] == [
        "neutral",
        "thinking",
    ]


def test_activate_pack_captures_its_result_before_commit(
    repository: VisualIdentityRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_get = repository.get_active_actor_pack

    def guarded_get(actor_kind: str, actor_id: int | str) -> dict[str, Any] | None:
        assert repository.db.get_connection().in_transaction
        return original_get(actor_kind, actor_id)

    monkeypatch.setattr(repository, "get_active_actor_pack", guarded_get)

    result = _activate(repository)

    assert result["version"]["version_number"] == 1
    assert result["pack"]["active_version_id"] == result["version"]["id"]


def test_activate_pack_rolls_back_when_the_final_asset_insert_fails(
    repository: VisualIdentityRepository,
) -> None:
    assets = [_asset(f"reaction-{index:02d}") for index in range(30)]
    assets.append(_asset("reaction-31", bytes_=0))

    with pytest.raises(CharactersRAGDBError, match="constraint violation"):
        _activate(repository, assets=assets)

    assert _counts(repository.db) == {
        "visual_identity_packs": 0,
        "visual_identity_pack_versions": 0,
        "visual_identity_assets": 0,
        "visual_identity_bindings": 0,
    }


@pytest.mark.parametrize("nonstandard", [float("nan"), float("inf")])
def test_activate_pack_rejects_nonstandard_json_without_writing(
    repository: VisualIdentityRepository, nonstandard: float
) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        _activate(repository, manifest={"value": nonstandard})

    assert _counts(repository.db) == {
        "visual_identity_packs": 0,
        "visual_identity_pack_versions": 0,
        "visual_identity_assets": 0,
        "visual_identity_bindings": 0,
    }


def test_publish_version_keeps_versions_immutable_and_activates_the_next_number(
    repository: VisualIdentityRepository,
) -> None:
    first = _activate(repository, assets=[_asset("neutral")])
    pack_id = first["pack"]["id"]
    old_version_id = first["version"]["id"]
    old_asset_id = first["assets"][0]["id"]

    second = repository.publish_version(
        pack_id,
        manifest={"schema": "test/v1", "revision": 2},
        assets=[_asset("neutral"), _asset("thinking")],
        actor_kind="character",
        actor_id="42",
    )

    assert second["version"]["version_number"] == 2
    assert second["version"]["id"] != old_version_id
    assert second["pack"]["active_version_id"] == second["version"]["id"]
    assert second["binding"]["active_version_id"] == second["version"]["id"]
    old_asset = repository.db.execute_query(
        "SELECT * FROM visual_identity_assets WHERE id = ?", (old_asset_id,)
    ).fetchone()
    assert old_asset is not None
    version_numbers = [
        int(row[0])
        for row in repository.db.execute_query(
            """
            SELECT version_number
              FROM visual_identity_pack_versions
             WHERE pack_id = ?
             ORDER BY version_number
            """,
            (pack_id,),
        ).fetchall()
    ]
    assert version_numbers == [1, 2]


def test_publish_version_captures_its_result_before_commit(
    repository: VisualIdentityRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _activate(repository)
    original_get = repository.get_active_actor_pack

    def guarded_get(actor_kind: str, actor_id: int | str) -> dict[str, Any] | None:
        assert repository.db.get_connection().in_transaction
        return original_get(actor_kind, actor_id)

    monkeypatch.setattr(repository, "get_active_actor_pack", guarded_get)

    result = repository.publish_version(
        first["pack"]["id"],
        manifest={"revision": 2},
        assets=[_asset("neutral")],
        actor_kind="character",
        actor_id="42",
    )

    assert result["version"]["version_number"] == 2
    assert result["pack"]["active_version_id"] == result["version"]["id"]


def test_publish_version_rolls_back_late_asset_failure_to_prior_graph(
    repository: VisualIdentityRepository,
) -> None:
    first = _activate(repository, assets=[_asset("neutral"), _asset("thinking")])
    pack_id = first["pack"]["id"]
    binding_id = first["binding"]["id"]
    pack_before = dict(
        repository.db.execute_query(
            "SELECT * FROM visual_identity_packs WHERE id = ?", (pack_id,)
        ).fetchone()
    )
    binding_before = dict(
        repository.db.execute_query(
            "SELECT * FROM visual_identity_bindings WHERE id = ?", (binding_id,)
        ).fetchone()
    )
    versions_before = [
        dict(row)
        for row in repository.db.execute_query(
            "SELECT * FROM visual_identity_pack_versions WHERE pack_id = ? ORDER BY id",
            (pack_id,),
        ).fetchall()
    ]
    assets_before = [
        dict(row)
        for row in repository.db.execute_query(
            "SELECT * FROM visual_identity_assets WHERE pack_id = ? ORDER BY id",
            (pack_id,),
        ).fetchall()
    ]
    failing_assets = [_asset(f"reaction-{index:02d}") for index in range(30)]
    failing_assets.append(_asset("reaction-31", bytes_=0))

    with pytest.raises(CharactersRAGDBError, match="constraint violation"):
        repository.publish_version(
            pack_id,
            manifest={"revision": 2},
            assets=failing_assets,
            actor_kind="character",
            actor_id="42",
        )

    assert (
        dict(
            repository.db.execute_query(
                "SELECT * FROM visual_identity_packs WHERE id = ?", (pack_id,)
            ).fetchone()
        )
        == pack_before
    )
    assert (
        dict(
            repository.db.execute_query(
                "SELECT * FROM visual_identity_bindings WHERE id = ?", (binding_id,)
            ).fetchone()
        )
        == binding_before
    )
    assert [
        dict(row)
        for row in repository.db.execute_query(
            "SELECT * FROM visual_identity_pack_versions WHERE pack_id = ? ORDER BY id",
            (pack_id,),
        ).fetchall()
    ] == versions_before
    assert [
        dict(row)
        for row in repository.db.execute_query(
            "SELECT * FROM visual_identity_assets WHERE pack_id = ? ORDER BY id",
            (pack_id,),
        ).fetchall()
    ] == assets_before


def test_publish_rejects_null_pack_active_version_without_writing(
    repository: VisualIdentityRepository,
) -> None:
    activated = _activate(repository)
    pack_id = activated["pack"]["id"]
    with repository.db.transaction():
        repository.db.execute_query(
            "UPDATE visual_identity_packs SET active_version_id = ? WHERE id = ?",
            (None, pack_id),
        )
    before = _counts(repository.db)

    with pytest.raises(
        ValueError, match="visual_identity_pack_active_version_mismatch"
    ):
        repository.publish_version(
            pack_id,
            manifest={"revision": 2},
            assets=[_asset("neutral")],
            actor_kind="character",
            actor_id="42",
        )

    assert _counts(repository.db) == before
    assert (
        repository.db.execute_query(
            "SELECT active_version_id FROM visual_identity_packs WHERE id = ?",
            (pack_id,),
        ).fetchone()[0]
        is None
    )


def test_archive_delete_and_binding_tombstone_never_hard_delete_rows(
    repository: VisualIdentityRepository,
) -> None:
    activated = _activate(repository)
    pack_id = activated["pack"]["id"]
    binding_id = activated["binding"]["id"]
    with repository.db.transaction():
        repository.db.execute_query(
            """
            INSERT INTO visual_identity_bindings(
                owner_user_id, actor_kind, actor_id, pack_id,
                active_version_id, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                LOCAL_OWNER_ID,
                "character",
                "42",
                pack_id,
                activated["version"]["id"],
                "deleted",
            ),
        )

    assert repository.archive_pack(pack_id)["status"] == "archived"
    assert repository.mark_pack_deleted(pack_id)["status"] == "deleted"
    deleted_binding = repository.mark_binding_deleted("character", "42")
    assert deleted_binding["id"] == binding_id
    assert deleted_binding["status"] == "deleted"

    assert (
        repository.db.execute_query(
            "SELECT id FROM visual_identity_packs WHERE id = ?", (pack_id,)
        ).fetchone()
        is not None
    )
    assert (
        repository.db.execute_query(
            "SELECT id FROM visual_identity_bindings WHERE id = ?", (binding_id,)
        ).fetchone()
        is not None
    )


@pytest.mark.parametrize("operation", ["archive_pack", "mark_pack_deleted"])
def test_repeated_pack_status_change_is_idempotent(
    repository: VisualIdentityRepository, operation: str
) -> None:
    activated = _activate(repository)
    method = getattr(repository, operation)

    first = method(activated["pack"]["id"])
    second = method(activated["pack"]["id"])

    assert second == first


@pytest.mark.parametrize(
    ("operation", "requested_status", "concurrent_status"),
    [
        ("archive_pack", "archived", "deleted"),
        ("mark_pack_deleted", "deleted", "archived"),
    ],
)
def test_pack_status_change_returns_its_own_transaction_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    requested_status: str,
    concurrent_status: str,
) -> None:
    path = tmp_path / f"{operation}-ownership.db"
    primary_db = CharactersRAGDB(path, client_id="status-primary")
    concurrent_db = CharactersRAGDB(path, client_id="status-concurrent")
    primary = VisualIdentityRepository(primary_db)
    pack_id = _activate(primary)["pack"]["id"]
    result_read_reached = Event()
    concurrent_write_attempting = Event()
    concurrent_write_done = Event()
    original_execute = primary_db.execute_query
    intercepted = False

    def interleaved_execute(query: str, params: Any = None, **kwargs: Any) -> Any:
        nonlocal intercepted
        result_query = (
            "SELECT * FROM visual_identity_packs WHERE id = ? AND owner_user_id = ?"
        )
        if not intercepted and " ".join(query.split()) == result_query:
            intercepted = True
            result_read_reached.set()
            assert concurrent_write_attempting.wait(5)
            if not primary_db.get_connection().in_transaction:
                assert concurrent_write_done.wait(5)
        return original_execute(query, params, **kwargs)

    monkeypatch.setattr(primary_db, "execute_query", interleaved_execute)

    def write_concurrently() -> None:
        try:
            assert result_read_reached.wait(5)
            with concurrent_db.transaction():
                concurrent_write_attempting.set()
                concurrent_db.execute_query(
                    """
                    UPDATE visual_identity_packs
                       SET status = ?, version = version + 1
                     WHERE id = ?
                    """,
                    (concurrent_status, pack_id),
                )
        finally:
            concurrent_write_done.set()
            concurrent_db.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write_concurrently)
            result = getattr(primary, operation)(pack_id)
            future.result(timeout=5)

        stored = dict(
            primary_db.execute_query(
                "SELECT status, version FROM visual_identity_packs WHERE id = ?",
                (pack_id,),
            ).fetchone()
        )
        assert result["status"] == requested_status
        assert result["version"] == 2
        assert stored == {"status": concurrent_status, "version": 3}
    finally:
        concurrent_db.close_connection()
        primary_db.close_connection()


@pytest.mark.parametrize("operation", ["archive_pack", "mark_pack_deleted"])
@pytest.mark.parametrize("corruption", ["null", "cross_pack"])
def test_pack_status_changes_reject_invalid_active_version_without_writing(
    repository: VisualIdentityRepository, operation: str, corruption: str
) -> None:
    activated = _activate(repository)
    pack_id = activated["pack"]["id"]
    bad_version_id = None
    if corruption == "cross_pack":
        _, bad_version_id = _seed_pack_version(repository.db, title="Other")
    with repository.db.transaction():
        repository.db.execute_query(
            "UPDATE visual_identity_packs SET active_version_id = ? WHERE id = ?",
            (bad_version_id, pack_id),
        )
    before = dict(
        repository.db.execute_query(
            "SELECT status, version FROM visual_identity_packs WHERE id = ?",
            (pack_id,),
        ).fetchone()
    )

    with pytest.raises(
        ValueError, match="visual_identity_pack_active_version_mismatch"
    ):
        getattr(repository, operation)(pack_id)

    after = dict(
        repository.db.execute_query(
            "SELECT status, version FROM visual_identity_packs WHERE id = ?",
            (pack_id,),
        ).fetchone()
    )
    assert after == before


def test_candidate_asset_pack_mismatch_rejects_and_rolls_back_activation(
    repository: VisualIdentityRepository,
) -> None:
    other_pack_id, _ = _seed_pack_version(repository.db, title="Other")
    before = _counts(repository.db)

    with pytest.raises(ValueError, match="visual_identity_asset_pack_mismatch"):
        _activate(repository, assets=[_asset("neutral", pack_id=other_pack_id)])

    assert _counts(repository.db) == before


def test_list_version_assets_rejects_cross_pack_rows_without_writing(
    repository: VisualIdentityRepository,
) -> None:
    first_pack_id, _ = _seed_pack_version(repository.db, title="First")
    _, second_version_id = _seed_pack_version(repository.db, title="Second")
    with repository.db.transaction():
        repository.db.execute_query(
            """
            INSERT INTO visual_identity_assets(
                owner_user_id, pack_id, pack_version_id, expression_key,
                original_expression_key, source_filename, storage_relpath,
                content_type, bytes, sha256, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                LOCAL_OWNER_ID,
                first_pack_id,
                second_version_id,
                "neutral",
                "neutral",
                "neutral.webp",
                "characters/test/neutral.webp",
                "image/webp",
                10,
                "sha-neutral",
                1,
                1,
            ),
        )
    before = _counts(repository.db)

    with pytest.raises(ValueError, match="visual_identity_asset_pack_mismatch"):
        repository.list_version_assets(second_version_id)

    assert _counts(repository.db) == before


def test_publish_rejects_cross_pack_active_version_before_inserting(
    repository: VisualIdentityRepository,
) -> None:
    first_pack_id, _ = _seed_pack_version(repository.db, title="First")
    _, second_version_id = _seed_pack_version(repository.db, title="Second")
    with repository.db.transaction():
        repository.db.execute_query(
            "UPDATE visual_identity_packs SET active_version_id = ? WHERE id = ?",
            (second_version_id, first_pack_id),
        )
    before = _counts(repository.db)

    with pytest.raises(
        ValueError, match="visual_identity_pack_active_version_mismatch"
    ):
        repository.publish_version(
            first_pack_id,
            manifest={"revision": 2},
            assets=[_asset("neutral")],
            actor_kind="character",
            actor_id="42",
        )

    assert _counts(repository.db) == before


def test_active_binding_rejects_cross_pack_version_and_publish_writes_nothing(
    repository: VisualIdentityRepository,
) -> None:
    first_pack_id, first_version_id = _seed_pack_version(repository.db, title="First")
    _, second_version_id = _seed_pack_version(repository.db, title="Second")
    with repository.db.transaction():
        repository.db.execute_query(
            """
            INSERT INTO visual_identity_bindings(
                owner_user_id, actor_kind, actor_id, pack_id, active_version_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                LOCAL_OWNER_ID,
                "character",
                "42",
                first_pack_id,
                second_version_id,
            ),
        )
    before = _counts(repository.db)

    with pytest.raises(
        ValueError, match="visual_identity_binding_active_version_mismatch"
    ):
        repository.get_active_actor_pack("character", "42")
    with pytest.raises(
        ValueError, match="visual_identity_binding_active_version_mismatch"
    ):
        repository.publish_version(
            first_pack_id,
            manifest={"revision": 2},
            assets=[_asset("neutral")],
            actor_kind="character",
            actor_id="42",
        )

    assert _counts(repository.db) == before
    assert (
        repository.db.execute_query(
            "SELECT active_version_id FROM visual_identity_packs WHERE id = ?",
            (first_pack_id,),
        ).fetchone()[0]
        == first_version_id
    )


def test_mark_binding_deleted_rejects_cross_pack_version_without_writing(
    repository: VisualIdentityRepository,
) -> None:
    activated = _activate(repository)
    binding_id = activated["binding"]["id"]
    _, bad_version_id = _seed_pack_version(repository.db, title="Other")
    with repository.db.transaction():
        repository.db.execute_query(
            """
            UPDATE visual_identity_bindings
               SET active_version_id = ?
             WHERE id = ?
            """,
            (bad_version_id, binding_id),
        )
    before = dict(
        repository.db.execute_query(
            "SELECT status, version FROM visual_identity_bindings WHERE id = ?",
            (binding_id,),
        ).fetchone()
    )

    with pytest.raises(
        ValueError, match="visual_identity_binding_active_version_mismatch"
    ):
        repository.mark_binding_deleted("character", "42")

    after = dict(
        repository.db.execute_query(
            "SELECT status, version FROM visual_identity_bindings WHERE id = ?",
            (binding_id,),
        ).fetchone()
    )
    assert after == before


def test_repository_does_not_bootstrap_its_own_schema(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "current.db", client_id="schema-owner")
    try:
        with db.transaction():
            db.execute_query("DROP TABLE visual_identity_bindings")
        repo = VisualIdentityRepository(db)

        with pytest.raises(CharactersRAGDBError, match="no such table"):
            repo.get_active_actor_pack("character", "42")
    finally:
        db.close_connection()
