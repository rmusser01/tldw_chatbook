"""Create-only lifecycle coverage for the bundled Samira identity pack."""

from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Character_Chat import visual_identity
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
SAMIRA_NAME = "Samira “Sammy” Vadem"
BUILTIN_NAME = f"{SAMIRA_NAME} (Built-in)"


def _ensure(db: CharactersRAGDB, **kwargs) -> None:
    visual_identity.ensure_builtin_samira(db, package_root=PACKAGE_ROOT, **kwargs)


def _rows(db: CharactersRAGDB, table: str) -> list[dict]:
    return [dict(row) for row in db.execute_query(f"SELECT * FROM {table}").fetchall()]


def _samira_row(db: CharactersRAGDB) -> dict | None:
    for row in _rows(db, "character_cards"):
        extensions = json.loads(row["extensions"] or "{}")
        if extensions.get("tldw/builtin_id") == "samira":
            return row
    return None


def _assert_complete_seed(db: CharactersRAGDB) -> tuple[dict, dict]:
    card = _samira_row(db)
    assert card is not None
    packs = _rows(db, "visual_identity_packs")
    versions = _rows(db, "visual_identity_pack_versions")
    assets = _rows(db, "visual_identity_assets")
    bindings = _rows(db, "visual_identity_bindings")
    assert len(packs) == len(versions) == len(bindings) == 1
    assert len(assets) == 31
    assert bindings[0]["actor_id"] == str(card["id"])
    assert bindings[0]["active_version_id"] == versions[0]["id"]
    assert packs[0]["active_version_id"] == versions[0]["id"]
    context = json.loads(packs[0]["source_context_json"])
    assert context == {
        "pack_content_sha256": "5993ec841ca635d99ca83691c3ac284db1b14bff35978c72edad12df04a917c8",
        "source_id": "tldw.builtin.samira.reactions",
    }
    return card, packs[0]


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "lifecycle.sqlite", "lifecycle-test")
    yield database
    database.close_connection()


def test_fresh_install_seeds_searchable_card_and_complete_pack(
    db: CharactersRAGDB,
) -> None:
    _ensure(db)

    card, _pack = _assert_complete_seed(db)
    assert card["name"] == SAMIRA_NAME
    assert card["image"]
    assert [row["id"] for row in db.search_character_cards("Samira")] == [card["id"]]


def test_v37_upgrade_preserves_existing_card_then_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "upgrade.sqlite"
    with monkeypatch.context() as old:
        old.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 37)
        seeded = CharactersRAGDB(path, "v37")
        existing_id = seeded.add_character_card(
            {"name": "Existing", "description": "keep"}
        )
        seeded.close_connection()

    upgraded = CharactersRAGDB(path, "v38")
    try:
        _ensure(upgraded)
        _assert_complete_seed(upgraded)
        assert upgraded.get_character_card_by_id(existing_id)["description"] == "keep"
    finally:
        upgraded.close_connection()


def test_repeat_startup_is_idempotent_and_does_not_read_package(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure(db)
    before = {
        table: _rows(db, table)
        for table in (
            "character_cards",
            "visual_identity_packs",
            "visual_identity_pack_versions",
            "visual_identity_assets",
            "visual_identity_bindings",
        )
    }

    monkeypatch.setattr(
        visual_identity,
        "_read_samira_resource",
        lambda *args, **kwargs: pytest.fail("healthy preflight opened package data"),
    )
    _ensure(db)

    assert {table: _rows(db, table) for table in before} == before


@pytest.mark.parametrize(
    ("occupied", "expected"),
    [
        ([SAMIRA_NAME], BUILTIN_NAME),
        ([SAMIRA_NAME, BUILTIN_NAME], f"{BUILTIN_NAME} 2"),
        ([SAMIRA_NAME, BUILTIN_NAME, f"{BUILTIN_NAME} 2"], f"{BUILTIN_NAME} 3"),
    ],
)
def test_name_collision_uses_lowest_deterministic_suffix(
    db: CharactersRAGDB, occupied: list[str], expected: str
) -> None:
    for name in occupied:
        db.add_character_card({"name": name})

    _ensure(db)

    assert _samira_row(db)["name"] == expected


def test_renamed_and_edited_builtin_is_preserved_without_resource_reads(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure(db)
    card = _samira_row(db)
    original_id = card["id"]
    db.update_character_card(
        original_id,
        {"name": "My Sammy", "description": "User-authored description"},
        expected_version=card["version"],
    )
    monkeypatch.setattr(
        visual_identity,
        "_read_samira_resource",
        lambda *args, **kwargs: pytest.fail("customized card triggered package reads"),
    )

    _ensure(db)

    preserved = db.get_character_card_by_id(original_id)
    assert preserved["name"] == "My Sammy"
    assert preserved["description"] == "User-authored description"
    assert [row["id"] for row in db.search_character_cards("Sammy")] == [original_id]


def test_soft_delete_keeps_dormant_binding_and_restore_reuses_it(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure(db)
    card = _samira_row(db)
    binding = _rows(db, "visual_identity_bindings")[0]
    db.soft_delete_character_card(card["id"], card["version"])
    monkeypatch.setattr(
        visual_identity,
        "_read_samira_resource",
        lambda *args, **kwargs: pytest.fail("deleted card triggered package reads"),
    )

    _ensure(db)
    assert _rows(db, "visual_identity_bindings") == [binding]
    deleted = _samira_row(db)
    assert deleted["deleted"] == 1

    db.restore_character_card(deleted["id"], deleted["version"])
    _ensure(db)
    assert _rows(db, "visual_identity_bindings") == [binding]


def test_explicit_binding_tombstone_is_not_recreated(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure(db)
    card = _samira_row(db)
    VisualIdentityRepository(db).mark_binding_deleted("character", card["id"])
    monkeypatch.setattr(
        visual_identity,
        "_read_samira_resource",
        lambda *args, **kwargs: pytest.fail(
            "binding tombstone triggered package reads"
        ),
    )

    _ensure(db)

    bindings = _rows(db, "visual_identity_bindings")
    assert len(bindings) == 1
    assert bindings[0]["status"] == "deleted"


@pytest.mark.parametrize("status", ["archived", "deleted"])
def test_pack_tombstone_is_not_resurrected(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    _ensure(db)
    repository = VisualIdentityRepository(db)
    pack = _rows(db, "visual_identity_packs")[0]
    (repository.archive_pack if status == "archived" else repository.mark_pack_deleted)(
        pack["id"]
    )
    monkeypatch.setattr(
        visual_identity,
        "_read_samira_resource",
        lambda *args, **kwargs: pytest.fail("pack tombstone triggered package reads"),
    )

    _ensure(db)

    assert _rows(db, "visual_identity_packs")[0]["status"] == status
    assert len(_rows(db, "visual_identity_pack_versions")) == 1


def test_pack_failure_keeps_card_warns_once_and_retries_only_absent_pack(
    db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken_root = tmp_path / "broken" / "tldw_chatbook"
    shutil.copytree(PACKAGE_ROOT / "assets", broken_root / "assets")
    (broken_root / "assets/characters/samira/expressions/anger.webp").unlink()
    warnings: list[tuple] = []
    monkeypatch.setattr(
        visual_identity.logger, "warning", lambda *args: warnings.append(args)
    )

    visual_identity.ensure_builtin_samira(db, package_root=broken_root)

    card = _samira_row(db)
    assert card is not None and card["image"]
    assert not _rows(db, "visual_identity_packs")
    assert len(warnings) == 1
    assert "samira_pack_activation_failed" in warnings[0][0]

    preserved = dict(card)
    _ensure(db)
    after = _samira_row(db)
    assert after == preserved
    _assert_complete_seed(db)


def test_concurrent_eligible_repair_claims_one_pack_graph_and_one_reaction_set_read(
    db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken_root = tmp_path / "broken-concurrent" / "tldw_chatbook"
    shutil.copytree(PACKAGE_ROOT / "assets", broken_root / "assets")
    (broken_root / "assets/characters/samira/expressions/anger.webp").unlink()
    visual_identity.ensure_builtin_samira(db, package_root=broken_root)
    assert _samira_row(db) is not None
    assert not _rows(db, "visual_identity_packs")

    handles = [
        CharactersRAGDB(db.db_path_str, f"concurrent-{index}") for index in range(2)
    ]
    original_preflight = visual_identity._samira_seed_preflight
    original_read = visual_identity._read_samira_resource
    ready = threading.Barrier(2)
    first_preflight_threads: set[int] = set()
    guard = threading.Lock()
    reaction_reads = 0

    def synchronized_initial_preflight(handle):
        state = original_preflight(handle)
        thread_id = threading.get_ident()
        with guard:
            first = thread_id not in first_preflight_threads
            first_preflight_threads.add(thread_id)
        if first:
            ready.wait(timeout=5)
        return state

    def count_reads(package_root, relative_path, *, max_bytes):
        nonlocal reaction_reads
        if str(relative_path).startswith("expressions/"):
            with guard:
                reaction_reads += 1
        return original_read(package_root, relative_path, max_bytes=max_bytes)

    monkeypatch.setattr(
        visual_identity, "_samira_seed_preflight", synchronized_initial_preflight
    )
    monkeypatch.setattr(visual_identity, "_read_samira_resource", count_reads)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    visual_identity.ensure_builtin_samira,
                    handle,
                    package_root=PACKAGE_ROOT,
                )
                for handle in handles
            ]
            for future in futures:
                future.result(timeout=10)
    finally:
        for handle in handles:
            handle.close_connection()

    _assert_complete_seed(db)
    assert reaction_reads == 31


def test_oversized_samira_manifest_is_rejected_before_reaction_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed_root = tmp_path / "oversized" / "tldw_chatbook"
    shutil.copytree(PACKAGE_ROOT / "assets", malformed_root / "assets")
    manifest_path = (
        malformed_root / "assets/characters/samira/visual_identity_pack.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"] = [manifest["assets"][0] for _ in range(129)]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reaction_reads: list[str] = []
    original_read = visual_identity._read_samira_resource

    def count_reads(package_root, relative_path, *, max_bytes):
        if str(relative_path).startswith("expressions/"):
            reaction_reads.append(str(relative_path))
        return original_read(package_root, relative_path, max_bytes=max_bytes)

    monkeypatch.setattr(visual_identity, "_read_samira_resource", count_reads)

    with pytest.raises(ValueError, match="visual_identity_budget_exceeded"):
        visual_identity._load_samira_pack(
            malformed_root,
            card_bytes=1,
            portrait_bytes=1,
        )

    assert reaction_reads == []


def test_valid_profile_owned_fork_is_terminal_without_package_reads_or_warning(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ensure(db)
    card = _samira_row(db)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", card["id"])
    asset = graph["assets"][0]
    with db.transaction() as connection:
        pack_id = connection.execute(
            """INSERT INTO visual_identity_packs(
                   owner_user_id,title,status,default_expression_key,source_kind,source_context_json
               ) VALUES (0,'User fork','active','neutral','manual','{}')"""
        ).lastrowid
        version_id = connection.execute(
            """INSERT INTO visual_identity_pack_versions(
                   pack_id,owner_user_id,version_number,default_expression_key,manifest_json
               ) VALUES (?,0,1,'neutral','{}')""",
            (pack_id,),
        ).lastrowid
        connection.execute(
            """INSERT INTO visual_identity_assets(
                   owner_user_id,pack_id,pack_version_id,expression_key,
                   original_expression_key,display_label,source_filename,storage_relpath,
                   content_type,bytes,sha256,width,height,source_context_json,
                   is_animated,frame_count,deleted
               ) VALUES (0,?,?, 'neutral','neutral','Neutral','neutral.webp',
                   'fork/neutral.webp',?,?,?,?,?,'{}',0,1,0)""",
            (
                pack_id,
                version_id,
                asset["content_type"],
                asset["bytes"],
                asset["sha256"],
                asset["width"],
                asset["height"],
            ),
        )
        connection.execute(
            "UPDATE visual_identity_packs SET active_version_id=? WHERE id=?",
            (version_id, pack_id),
        )
        connection.execute(
            "UPDATE visual_identity_bindings SET pack_id=?,active_version_id=? WHERE actor_kind='character' AND actor_id=?",
            (pack_id, version_id, str(card["id"])),
        )
    monkeypatch.setattr(
        visual_identity,
        "_read_samira_resource",
        lambda *args, **kwargs: pytest.fail("user fork triggered package reads"),
    )
    warnings: list[tuple] = []
    monkeypatch.setattr(
        visual_identity.logger, "warning", lambda *args: warnings.append(args)
    )

    _ensure(db)

    active = VisualIdentityRepository(db).get_active_actor_pack("character", card["id"])
    assert active["pack"]["id"] == pack_id
    assert active["pack"]["source_kind"] == "manual"
    assert warnings == []


def test_config_eager_and_lazy_paths_seed_the_exact_constructed_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config

    created: list[object] = []
    seeded: list[object] = []

    class FakeCharactersDB:
        def __init__(self, *args, **kwargs):
            created.append(self)

    monkeypatch.setattr(config, "CharactersRAGDB", FakeCharactersDB)
    monkeypatch.setattr(config, "PromptsDatabase", lambda *args, **kwargs: object())
    monkeypatch.setattr(config, "MediaDatabase", lambda *args, **kwargs: object())
    monkeypatch.setattr(config, "seed_builtin_content", seeded.append)
    monkeypatch.setattr(config, "chachanotes_db", None)

    config.initialize_all_databases()
    assert seeded == [created[0]]

    created.clear()
    seeded.clear()
    monkeypatch.setattr(config, "chachanotes_db", None)
    config.get_chachanotes_db_lazy()
    assert seeded == [created[0]]


def test_app_injected_notes_database_uses_shared_seed_helper_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.app as app_module

    db = object()
    seeded: list[object] = []

    def record_seed(candidate):
        seeded.append(candidate)
        return candidate

    monkeypatch.setattr(app_module, "seed_builtin_content", record_seed)
    monkeypatch.setattr(
        app_module,
        "get_chachanotes_db_lazy",
        lambda: pytest.fail("injected DB should not use lazy construction"),
    )

    assert app_module._select_profile_database(SimpleNamespace(db=db)) is db
    assert seeded == [db]
