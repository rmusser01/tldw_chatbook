"""Fresh-profile and preservation contracts for the bundled pixel-migu character."""

from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from io import BytesIO

import pytest

from tldw_chatbook.Character_Chat import visual_identity
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

TABLES = (
    "character_cards",
    "visual_identity_packs",
    "visual_identity_pack_versions",
    "visual_identity_assets",
    "visual_identity_bindings",
)
EXPRESSIONS = {
    "angry",
    "custom:celebrate",
    "confused",
    "custom:error",
    "excited",
    "happy",
    "custom:listening",
    "custom:love",
    "neutral",
    "sad",
    "custom:skeptical",
    "custom:sleepy",
    "custom:speaking",
    "surprised",
    "thinking",
    "custom:thumbs_up",
    "custom:type",
    "custom:wave",
}


def _ensure(db):
    from tldw_chatbook.Character_Chat.builtin_pixel_migu import (
        ensure_builtin_pixel_migu,
    )

    ensure_builtin_pixel_migu(db)


def _snapshot(db):
    return {
        table: [
            dict(row) for row in db.execute_query(f"SELECT * FROM {table} ORDER BY id")
        ]
        for table in TABLES
    }


def _character(db):
    rows = db.execute_query("SELECT * FROM character_cards ORDER BY id").fetchall()
    matches = [
        dict(row)
        for row in rows
        if json.loads(row["extensions"] or "{}").get("tldw/builtin_id") == "pixel-migu"
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "pixel-migu.db", "pixel-migu-test")
    yield database
    database.close_connection()


def test_fresh_character_is_discoverable_with_all_manual_and_operational_images(db):
    existing_id = db.add_character_card({"name": "Existing assistant"})
    existing = db.get_character_card_by_id(existing_id)
    _ensure(db)

    card = _character(db)
    assert card["name"] == "pixel-migu"
    assert db.get_character_card_by_id(existing_id) == existing
    assert [row["id"] for row in db.search_character_cards("pixel-migu")] == [
        card["id"]
    ]
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", card["id"])
    assert graph["pack"]["source_kind"] == "builtin"
    assert (
        json.loads(graph["pack"]["source_context_json"])["source_id"]
        == "pixel-migu.expressions"
    )
    assert {asset["expression_key"] for asset in graph["assets"]} == EXPRESSIONS
    assert len(graph["assets"]) == 18
    manifest = json.loads(graph["version"]["manifest_json"])
    assert manifest["license"] == "LicenseRef-User-Supplied"

    for key in EXPRESSIONS:
        result = visual_identity.resolve_visual_identity(
            db,
            actor_kind="character",
            actor_id=card["id"],
            requested_state="idle",
            manual_expression_key=key,
        )
        assert result.resolved_expression_key == key
        assert result.resolution_source == "pack_manual"
        assert result.storage_source == "builtin"
        assert result.image_bytes
    for state, key in {
        "idle": "neutral",
        "thinking": "thinking",
        "speaking": "custom:speaking",
        "error": "custom:error",
    }.items():
        result = visual_identity.resolve_visual_identity(
            db,
            actor_kind="character",
            actor_id=card["id"],
            requested_state=state,
        )
        assert result.resolved_expression_key == key
        assert result.resolution_source == "pack_operational"
        assert result.image_bytes


def test_packaged_png_and_json_cards_have_matching_builtin_provenance(db):
    _ensure(db)
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        extract_json_from_image_file,
    )

    root = resources.files("tldw_chatbook").joinpath(
        "assets", "characters", "pixel_migu"
    )
    card = json.loads(root.joinpath("pixel-migu.character.json").read_bytes())
    embedded = extract_json_from_image_file(BytesIO(_character(db)["image"]))
    assert json.loads(embedded) == card
    assert card["data"]["extensions"]["tldw/builtin_id"] == "pixel-migu"


@pytest.mark.parametrize(
    "changes",
    [
        "none",
        "rename",
        "character_deleted",
        "pack_deleted",
        "pack_archived",
        "binding_deleted",
        "fork",
    ],
)
def test_reopened_profile_preserves_all_existing_state(
    db, monkeypatch, tmp_path, changes
):
    _ensure(db)
    card = _character(db)
    repo = VisualIdentityRepository(db)
    graph = repo.get_active_actor_pack("character", card["id"])
    if changes == "rename":
        db.update_character_card(
            card["id"],
            {"name": "My companion", "description": "My words"},
            card["version"],
        )
    elif changes == "character_deleted":
        db.soft_delete_character_card(card["id"], card["version"])
    elif changes == "pack_deleted":
        repo.mark_pack_deleted(graph["pack"]["id"])
    elif changes == "pack_archived":
        repo.archive_pack(graph["pack"]["id"])
    elif changes == "binding_deleted":
        repo.mark_binding_deleted("character", card["id"])
    elif changes == "fork":
        candidate = visual_identity.create_visual_identity_candidate(
            db,
            actor_kind="character",
            actor_id=card["id"],
        )
        candidate.stage_clear("angry")
        visual_identity.publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=tmp_path / "profile",
        )
        fork = repo.get_active_actor_pack("character", card["id"])
        assert fork["pack"]["source_kind"] == "manual"
        assert len(fork["assets"]) == 17
        assert "angry" not in {asset["expression_key"] for asset in fork["assets"]}
    before = _snapshot(db)
    path = db.db_path_str
    db.close_connection()
    reopened = CharactersRAGDB(path, "restart")
    try:
        from tldw_chatbook.Character_Chat import builtin_pixel_migu

        monkeypatch.setattr(
            builtin_pixel_migu,
            "_load_bundle",
            lambda: pytest.fail("restart read immutable resources"),
        )
        for _ in range(3):
            _ensure(reopened)
        assert _snapshot(reopened) == before
    finally:
        reopened.close_connection()


@pytest.mark.parametrize(
    "occupied, expected",
    [
        (["pixel-migu"], "pixel-migu (Built-in)"),
        (["pixel-migu", "pixel-migu (Built-in)"], "pixel-migu (Built-in) 2"),
    ],
)
def test_name_collision_preserves_user_cards_and_uses_available_suffix(
    db, occupied, expected
):
    for name in occupied:
        db.add_character_card({"name": name, "description": "User content"})
    before = _snapshot(db)["character_cards"]
    _ensure(db)
    assert _character(db)["name"] == expected
    assert _snapshot(db)["character_cards"][: len(before)] == before


def test_binding_failure_rolls_back_entire_seed_and_next_startup_retries(db):
    before = _snapshot(db)
    db.execute_query("""CREATE TEMP TRIGGER reject_pixel_binding
        BEFORE INSERT ON visual_identity_bindings
        BEGIN SELECT RAISE(ABORT, 'injected binding failure'); END""")
    with pytest.raises(Exception, match="injected binding failure"):
        _ensure(db)
    assert _snapshot(db) == before
    assert not db.get_connection().in_transaction
    db.execute_query("DROP TRIGGER reject_pixel_binding")
    _ensure(db)
    assert len(_snapshot(db)["visual_identity_assets"]) == 18


def test_deleted_pack_without_character_is_not_recreated(db):
    _ensure(db)
    card = _character(db)
    pack = VisualIdentityRepository(db).get_active_actor_pack("character", card["id"])[
        "pack"
    ]
    VisualIdentityRepository(db).mark_pack_deleted(pack["id"])
    db.execute_query("DELETE FROM character_cards WHERE id = ?", (card["id"],))
    before = _snapshot(db)
    _ensure(db)
    assert _snapshot(db) == before


@pytest.mark.parametrize(
    "broken_file", ["expressions/happy.png", "pixel-migu.character.json"]
)
def test_corrupt_resources_leave_profile_unchanged(
    db, tmp_path, monkeypatch, broken_file
):
    package_root = tmp_path / "package"
    source = resources.files("tldw_chatbook").joinpath(
        "assets", "characters", "pixel_migu"
    )
    target = package_root / "assets" / "characters" / "pixel_migu"
    shutil.copytree(source, target)
    (target / broken_file).write_bytes(b"corrupt")
    before = _snapshot(db)
    monkeypatch.setattr(resources, "files", lambda _package: package_root)
    with pytest.raises(ValueError):
        _ensure(db)
    assert _snapshot(db) == before


def test_concurrent_initialization_creates_one_complete_character_graph(
    db, monkeypatch
):
    from tldw_chatbook.Character_Chat import builtin_pixel_migu

    loaded = threading.Barrier(2)
    original_load = builtin_pixel_migu._load_bundle

    def synchronized_load():
        bundle = original_load()
        loaded.wait(timeout=10)
        return bundle

    def seed_handle():
        handle = CharactersRAGDB(db.db_path_str, "parallel-seed")
        try:
            _ensure(handle)
        finally:
            handle.close_connection()

    monkeypatch.setattr(builtin_pixel_migu, "_load_bundle", synchronized_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(seed_handle) for _ in range(2)]
        for future in futures:
            future.result(timeout=20)
    assert _character(db)["name"] == "pixel-migu"
    state = _snapshot(db)
    assert len(state["visual_identity_packs"]) == 1
    assert len(state["visual_identity_pack_versions"]) == 1
    assert len(state["visual_identity_bindings"]) == 1
    assert len(state["visual_identity_assets"]) == 18
