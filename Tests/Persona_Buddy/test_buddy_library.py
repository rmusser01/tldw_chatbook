"""Independent Buddy publication uses real SQLite and decoded image bytes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository


@pytest.fixture
def environment(tmp_path: Path):
    from tldw_chatbook.Persona_Buddy.library import BuddyLibrary

    profile = tmp_path / "profile"
    source = tmp_path / "source"
    profile.mkdir(mode=0o700)
    source.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "buddy.db", "independent-buddy")
    repository = PersonaVisualRepository(db)
    snapshot = _snapshot(source)
    publish_persona_visual(
        repository,
        snapshot,
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
    )
    library = BuddyLibrary(
        db,
        profile,
        persona_reader=lambda _: {
            "id": "persona-local-1",
            "version": 7,
            "is_active": True,
            "deleted": False,
        },
    )
    try:
        yield library, repository, source, profile, snapshot
    finally:
        db.close_connection()


def test_copy_has_real_buddy_owner_and_survives_persona_changes(environment):
    from tldw_chatbook.Persona_Visual.runtime import resolve_active_buddy_visual

    library, repo, source, profile, _ = environment
    original = repo.get_active_persona_pack("persona-local-1")
    buddy = library.copy_persona("persona-local-1", source_key="legacy:one")
    graph = library.get_graph(buddy.id)
    assert graph.identity.persona_id is None
    assert graph.identity.buddy_id == buddy.id
    assert graph.pack.id != original.pack.id
    assert (
        library.copy_persona("persona-local-1", source_key="legacy:one").id == buddy.id
    )
    updated = _snapshot(source, expected_identity=original.identity, color=(250, 0, 0))
    publish_persona_visual(
        repo,
        updated,
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
    )
    repo.archive_binding(
        persona_id="persona-local-1",
        expected_identity=repo.get_active_persona_pack("persona-local-1").identity,
    )
    assert library.get_graph(buddy.id) == graph
    visual = resolve_active_buddy_visual(repo, buddy.id, profile, "idle")
    assert visual.cache_identity.graph == graph.identity
    assert visual.reason is None
    assert len(library.list_buddies()) == 1


@pytest.mark.asyncio
async def test_independent_controller_needs_no_persona_service(environment):
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Persona_Buddy.preferences import (
        BuddySelection,
        PersonaBuddyPreferences,
    )

    library, repo, _, profile, _ = environment
    buddy = library.copy_persona("persona-local-1")
    controller = PersonaBuddyController(
        preferences=PersonaBuddyPreferences(
            enabled=True, selection=BuddySelection(buddy.id)
        ),
        profile_db=repo.db,
        profile_root=profile,
    )
    try:
        visual = await controller.resolve_current_visual(cols=12, lines=6)
        assert visual.available
        assert visual.graph_identity.buddy_id == buddy.id
        assert visual.persona_id is None
    finally:
        await controller.shutdown()


def test_builtin_is_independent_and_idempotent(environment):
    library, repo, _, _, _ = environment
    first = library.ensure_builtin()
    second = library.ensure_builtin()
    assert first.id == second.id
    assert library.get_graph(first.id).identity.persona_id is None
    assert repo.get_active_persona_pack("local-persona-builtin-pixel-migu") is None


def test_native_archive_review_is_read_only_and_preserves_notices(
    environment, tmp_path
):
    import json

    from Tests.Persona_Visual.test_persona_visual_importer import (
        _archive_payloads,
        _canonical,
        _replace_declared_payload,
        _write_archive,
    )
    from tldw_chatbook.Persona_Visual.artwork import decode_native_artwork

    library, repo, _, _, _ = environment
    payloads = _archive_payloads()
    pack = json.loads(payloads["metadata/pack.json"])
    pack["pack"].update(
        creator="Original Artist",
        license=None,
        source_url="https://example.com/pet",
        notices="Original notice\nSecond line",
    )
    _replace_declared_payload(payloads, "metadata/pack.json", _canonical(pack))
    path = _write_archive(tmp_path / "native.tldw-persona-vpack", payloads)
    review = library.review_archive(path)
    assert library.list_buddies() == ()
    assert review.artwork["license"] is None
    buddy = library.publish_review(review)
    saved = repo.get_active_buddy_pack_for_export(buddy.id)
    assert decode_native_artwork(dict(saved.source_context)["artwork"]) == dict(
        review.artwork
    )
    assert saved.graph.identity.persona_id is None
    assert library.resolve_preview(buddy.id).reason is None


def test_migration_keeps_geometry_and_reuses_committed_copy_after_write_failure(
    environment,
):
    from tldw_chatbook.Persona_Buddy.library import BuddyLibrary
    from tldw_chatbook.Persona_Buddy.preferences import (
        BuddySelection,
        PersonaBuddyGeometry,
        PersonaBuddyPreferences,
        PersonaBuddySelection,
        parse_persona_buddy_preferences,
        serialize_persona_buddy_preferences,
    )

    library, repo, _, profile, _ = environment
    old = PersonaBuddyPreferences(
        enabled=False,
        open=False,
        collapsed=True,
        geometry=PersonaBuddyGeometry(2, 3, 19, 8),
        selection=PersonaBuddySelection("local", "persona-local-1"),
    )
    assert library.migrate_legacy_selection(old, writer=lambda _: False) == old
    first = library.list_buddies()
    assert len(first) == 1
    # Restart with no source service; the committed migration copy is sufficient.
    reopened = BuddyLibrary(repo.db, profile)
    writes = []
    new = reopened.migrate_legacy_selection(
        old, writer=lambda value: writes.append(value) is None
    )
    assert new == replace(old, selection=BuddySelection(first[0].id))
    assert (
        parse_persona_buddy_preferences(serialize_persona_buddy_preferences(new)) == new
    )
    assert len(reopened.list_buddies()) == 1
    assert writes == [new]


def test_failed_copy_and_stale_review_do_not_change_library_or_selection(
    environment, tmp_path
):
    from Tests.Persona_Visual.test_persona_visual_importer import _write_archive
    from tldw_chatbook.Persona_Buddy.preferences import (
        PersonaBuddyPreferences,
        PersonaBuddySelection,
    )

    library, repo, _, _, _ = environment
    old = PersonaBuddyPreferences(
        enabled=True, selection=PersonaBuddySelection("local", "missing")
    )
    assert library.migrate_legacy_selection(old, writer=lambda _: True) == old
    path = _write_archive(tmp_path / "native.tldw-persona-vpack")
    review = library.review_archive(path)
    path.unlink()
    with pytest.raises(ValueError, match="buddy_source_changed"):
        library.publish_review(review)
    assert library.list_buddies() == ()
    assert (
        repo.db.execute_query("SELECT COUNT(*) FROM buddy_profiles").fetchone()[0] == 0
    )


def test_publication_rejection_leaves_no_owned_graph(environment, tmp_path):
    from Tests.Persona_Visual.test_persona_visual_importer import _write_archive

    library, repo, _, _, _ = environment
    review = library.review_archive(
        _write_archive(tmp_path / "native.tldw-persona-vpack")
    )
    corrupt = replace(review, assets=(replace(review.assets[0], data=b"invalid"),))
    with pytest.raises(ValueError):
        library.publish_review(corrupt)
    assert library.list_buddies() == ()
    assert (
        repo.db.execute_query("SELECT COUNT(*) FROM buddy_profiles").fetchone()[0] == 0
    )


def test_schema69_upgrade_and_restart_preserve_legacy_binding(tmp_path, monkeypatch):
    from Tests.Persona_Visual.test_persona_visual_repository import _activate
    from tldw_chatbook.Persona_Buddy.library import BuddyLibrary

    path = tmp_path / "upgrade.db"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 69)
    old = CharactersRAGDB(path, "upgrade")
    legacy = _activate(PersonaVisualRepository(old))
    old.close_connection()
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 70)
    upgraded = CharactersRAGDB(path, "upgrade")
    try:
        assert (
            upgraded.execute_query(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 70
        )
        assert upgraded.execute_query("PRAGMA foreign_key_check").fetchall() == []
        assert BuddyLibrary(upgraded, tmp_path).list_buddies() == ()
        assert (
            PersonaVisualRepository(upgraded).get_active_persona_pack(
                legacy.identity.persona_id
            )
            == legacy
        )
    finally:
        upgraded.close_connection()


@pytest.mark.asyncio
async def test_management_batch_save_and_revision_guard(monkeypatch):
    import tldw_chatbook.Persona_Buddy.preferences as preference_module
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Persona_Buddy.preferences import BuddySelection

    writes = []
    monkeypatch.setattr(
        preference_module,
        "save_settings_to_cli_config",
        lambda sections: writes.append(sections) is None,
    )
    controller = PersonaBuddyController()
    try:
        old = controller.current_preferences()
        revision = controller.apply_preferences_patch(
            enabled=True, selection=BuddySelection("buddy-one")
        )
        assert await controller.persist_preferences_revision(
            revision,
            extra_sections={
                "buddy_interaction": {
                    "scope_kind": "conversation",
                    "scope_id": "exact-conversation",
                },
            },
        )
        assert len(writes) == 1
        assert writes[0]["persona_buddy"]["buddy_id"] == "buddy-one"
        assert writes[0]["buddy_interaction"]["scope_id"] == "exact-conversation"
        newer = controller.apply_preferences_patch(open=False)
        assert not controller.rollback_preferences_revision(revision, old)
        assert controller.current_preferences().open is False
        assert controller.rollback_preferences_revision(newer, old)
        assert controller.current_preferences() == old
        assert not await controller.persist_preferences_revision(
            revision, extra_sections={"buddy_interaction": {}}
        )
        assert len(writes) == 1
    finally:
        await controller.shutdown()


@pytest.mark.asyncio
async def test_controller_migrates_only_after_persisted_copy(environment):
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Persona_Buddy.preferences import (
        BuddySelection,
        PersonaBuddyPreferences,
        PersonaBuddySelection,
    )

    library, repo, _, profile, _ = environment
    old = PersonaBuddyPreferences(
        enabled=True, selection=PersonaBuddySelection("local", "persona-local-1")
    )
    controller = PersonaBuddyController(
        preferences=old,
        profile_db=repo.db,
        profile_root=profile,
        preference_writer=lambda _: False,
    )
    try:
        assert not await controller.migrate_legacy_selection(library)
        assert controller.current_preferences() == old
        controller._preference_writer = lambda _: True
        assert await controller.migrate_legacy_selection(library)
        assert type(controller.current_preferences().selection) is BuddySelection
        assert len(library.list_buddies()) == 1
    finally:
        await controller.shutdown()


def test_buddy_version_publication_uses_exact_owner_and_preserves_notices(environment):
    import json

    from tldw_chatbook.Persona_Visual.artwork import encode_native_artwork

    library, repo, source, profile, _ = environment
    buddy = library.copy_persona("persona-local-1")
    graph = library.get_graph(buddy.id)
    artwork = {
        "version": 1,
        "creator": "Artist",
        "license": None,
        "source_url": None,
        "notices": "Keep this notice",
    }
    updated = replace(
        _snapshot(source),
        persona_id=None,
        persona_revision=0,
        buddy_id=buddy.id,
        buddy_revision=buddy.revision,
        expected_identity=graph.identity,
        source_context=(("artwork", encode_native_artwork(artwork)),),
    )
    result = publish_persona_visual(
        repo,
        updated,
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
    )
    assert result.new_identity.version_number == 2
    assert result.new_identity.buddy_id == buddy.id
    stored = repo.get_active_buddy_pack_for_export(buddy.id)
    assert json.loads(dict(stored.source_context)["artwork"]) == artwork
    assert library.resolve_preview(buddy.id).cache_identity.graph == result.new_identity
    with pytest.raises(ValueError, match="identity_changed"):
        publish_persona_visual(
            repo,
            updated,
            source_root=source,
            profile_root=profile,
            authority_guard=lambda: True,
        )


def test_builtin_retains_legacy_and_independent_tombstones(environment):
    library, repo, _, _, _ = environment
    assert library.ensure_builtin(legacy_retired=True) is None
    assert library.list_buddies() == ()
    buddy = library.ensure_builtin()
    assert library.ensure_builtin(legacy_retired=True) == buddy
    with repo.db.transaction():
        repo.db.execute_query(
            "UPDATE buddy_profiles SET status='deleted' WHERE id=?", (buddy.id,)
        )
    assert library.ensure_builtin() is None
    assert library.get_graph(buddy.id) is None


def test_published_buddy_reopens_without_source_service(environment):
    from tldw_chatbook.Persona_Buddy.library import BuddyLibrary

    library, repo, _, profile, _ = environment
    buddy = library.copy_persona("persona-local-1")
    expected = library.get_graph(buddy.id)
    reopened_db = CharactersRAGDB(repo.db.db_path, "reopened-independent-buddy")
    try:
        reopened = BuddyLibrary(reopened_db, profile)
        assert reopened.get_buddy(buddy.id) == buddy
        assert reopened.get_graph(buddy.id) == expected
        assert (
            reopened.resolve_preview(buddy.id).cache_identity.graph == expected.identity
        )
    finally:
        reopened_db.close_connection()


def test_changed_persona_authority_rejects_the_copy(environment, monkeypatch):
    import tldw_chatbook.Persona_Buddy.library as library_module

    library, repo, _, _, _ = environment
    record = {
        "id": "persona-local-1",
        "version": 7,
        "is_active": True,
        "deleted": False,
    }
    library.persona_reader = lambda _: dict(record)
    original_publish = library_module.publish_persona_visual

    def publish_after_persona_change(*args, **kwargs):
        record["version"] = 8
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        library_module, "publish_persona_visual", publish_after_persona_change
    )
    with pytest.raises(ValueError, match="authority_changed"):
        library.copy_persona("persona-local-1")
    assert library.list_buddies() == ()
    assert (
        repo.db.execute_query("SELECT COUNT(*) FROM buddy_profiles").fetchone()[0] == 0
    )


def test_active_buddy_lookup_uses_partial_index_without_statistics(environment):
    library, repository, *_ = environment
    buddy = library.copy_persona("persona-local-1")
    connection = library.db.get_connection()
    # Pin the actual union-view lookup, without ANALYZE influencing the planner.
    assert (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'sqlite_stat1'"
        ).fetchone()
        is None
    )
    plan = connection.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM visual_owner_bindings "
        "WHERE persona_id IS ? AND buddy_id IS ? AND status = 'active'",
        (None, buddy.id),
    ).fetchall()
    assert any("idx_buddy_visual_bindings_active" in str(row[3]) for row in plan)
    binding = repository._active_binding_record(None, buddy_id=buddy.id)
    assert binding is not None and binding.buddy_id == buddy.id


def test_archive_path_uses_shared_validation_before_opening(tmp_path, monkeypatch):
    from tldw_chatbook.Persona_Visual import snapshot
    from tldw_chatbook.Persona_Visual.importer import PersonaVisualImportError

    opened = []
    monkeypatch.setattr(
        snapshot.importer, "_pin_source", lambda path: opened.append(path)
    )
    with pytest.raises(PersonaVisualImportError):
        snapshot.read_buddy_archive(str(tmp_path / "../../forbidden.zip"))
    assert opened == []


def test_archive_path_keeps_no_follow_symlink_boundary(tmp_path):
    from Tests.Persona_Visual.test_persona_visual_importer import _write_archive
    from tldw_chatbook.Persona_Visual.importer import PersonaVisualImportError
    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive

    target = _write_archive(tmp_path / "target.zip")
    link = tmp_path / "link.zip"
    link.symlink_to(target)
    with pytest.raises(PersonaVisualImportError):
        read_buddy_archive(link)


def test_listing_pages_are_bounded_and_lookup_does_not_scan_collection(
    environment, monkeypatch
):
    library, repo, *_ = environment
    original = library.copy_persona("persona-local-1")
    graph = library.get_graph(original.id)
    with repo.db.transaction() as cursor:
        for index in range(205):
            buddy_id = f"page-{index:03d}"
            cursor.execute(
                "INSERT INTO buddy_profiles(id,name) VALUES (?,?)",
                (buddy_id, "Same name"),
            )
            cursor.execute(
                "INSERT INTO buddy_visual_bindings(buddy_id,buddy_revision,pack_id,active_version_id) VALUES (?,1,?,?)",
                (buddy_id, graph.identity.pack_id, graph.identity.pack_version_id),
            )
    page1 = library.list_buddies(limit=100, offset=0)
    page2 = library.list_buddies(limit=100, offset=100)
    page3 = library.list_buddies(limit=100, offset=200)
    assert [len(page) for page in (page1, page2, page3)] == [100, 100, 6]
    assert len({row.id for page in (page1, page2, page3) for row in page}) == 206
    assert len(library.list_buddies()) == 100
    monkeypatch.setattr(
        library,
        "list_buddies",
        lambda **_: pytest.fail("single lookup scanned library"),
    )
    assert library.get_buddy("page-204").id == "page-204"
    assert library.get_buddy("missing") is None


def test_library_reads_hold_transaction(environment):
    library, repo, *_ = environment
    buddy = library.copy_persona("persona-local-1", source_key="read-guard")
    connection = repo.db.get_connection()
    reads = []
    connection.set_trace_callback(
        lambda statement: (
            reads.append(connection.in_transaction)
            if statement.lstrip().upper().startswith("SELECT")
            and "buddy_profiles" in statement
            else None
        )
    )
    try:
        assert library.list_buddies()
        assert library._source_record("read-guard") == buddy
    finally:
        connection.set_trace_callback(None)
    assert reads and all(reads)


def test_buddy_migration_verification_failure_rolls_back_schema(tmp_path, monkeypatch):
    from tldw_chatbook.DB.ChaChaNotes_DB import SchemaError

    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 69)
    db = CharactersRAGDB(tmp_path / "rollback.db", "rollback")
    original = db._get_db_version
    monkeypatch.setattr(
        db,
        "_get_db_version",
        lambda connection: 71 if original(connection) == 70 else original(connection),
    )
    try:
        with pytest.raises(SchemaError, match="version check"):
            db._migrate_from_v69_to_v70(db.get_connection())
        assert original(db.get_connection()) == 69
        assert (
            db.execute_query(
                "SELECT name FROM sqlite_master WHERE name='buddy_profiles'"
            ).fetchone()
            is None
        )
    finally:
        db.close_connection()


def test_buddy_archive_publication_cannot_spoof_import_provenance(
    environment, tmp_path
):
    import json

    from Tests.Persona_Visual.test_persona_visual_importer import (
        _archive_payloads,
        _canonical,
        _replace_declared_payload,
        _write_archive,
    )

    library, repo, *_ = environment
    payloads = _archive_payloads()
    pack = json.loads(payloads["metadata/pack.json"])
    pack["pack"]["source_context"] = {
        "provenance": "trusted-builtin",
        "license": "Keep notice",
    }
    _replace_declared_payload(payloads, "metadata/pack.json", _canonical(pack))
    buddy = library.import_archive(_write_archive(tmp_path / "forged.zip", payloads))
    context = dict(repo.get_active_buddy_pack_for_export(buddy.id).source_context)
    assert context["provenance"] == "untrusted-import"
    assert context["license"] == "Keep notice"
