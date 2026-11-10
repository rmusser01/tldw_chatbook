"""Fresh-profile built-in Buddy seeding through real Buddy and SQLite stores.

TASK-39057: these cases drive ``BuddyLibrary.ensure_builtin``, the seeding the
app's readiness pass runs, since the Persona_Visual installer they used to
drive had no production caller and was deleted.
"""

import pytest

import tldw_chatbook.Persona_Buddy.library as library_module
from Tests.app_module_patches import set_app_global
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Buddy.library import BuddyLibrary
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

#: The Persona the retired installer created; app readiness still reads it to
#: decide whether a legacy profile retired the built-in Buddy.
LEGACY_PERSONA_ID = "local-persona-builtin-pixel-migu"
BUNDLE = "assets/persona_visual/pixel_migu"


@pytest.fixture
def components(tmp_path):
    db = CharactersRAGDB(tmp_path / "actors.db", client_id="pixel-migu")
    service = LocalCharacterPersonaService(
        db, persona_store_path=tmp_path / "personas.json"
    )
    coordinator = PersonaActorPackCoordinator(ActorPackRepository(db), service)
    yield db, service, coordinator, tmp_path
    db.close_connection()


def install(components):
    db, _service, _coordinator, root = components
    return BuddyLibrary(db, root).ensure_builtin()


def _copied_bundle(root, monkeypatch):
    """Point the library's resource reader at an editable copy of the bundle."""
    import shutil

    package_root = root / "modified-package"
    resource_root = package_root / BUNDLE
    shutil.copytree(library_module.files("tldw_chatbook").joinpath(BUNDLE), resource_root)
    monkeypatch.setattr(library_module, "files", lambda _package: package_root)
    return resource_root


def _buddy_rows(db) -> int:
    return db.execute_query("SELECT COUNT(*) FROM buddy_profiles").fetchone()[0]


def test_fresh_install_publishes_an_independent_buddy_with_every_state(components):
    db, service, _coordinator, root = components
    buddy = install(components)
    assert buddy.name == "pixel-migu"
    assert buddy.source_key == "builtin:pixel-migu"
    library = BuddyLibrary(db, root)
    graph = library.get_graph(buddy.id)
    assert graph.identity.persona_id is None
    assert graph.identity.buddy_id == buddy.id
    assert len(graph.assets) == 64
    assert (
        "LicenseRef-User-Supplied"
        in db.execute_query(
            "SELECT source_context_json FROM persona_visual_packs WHERE id = ?",
            (graph.pack.id,),
        ).fetchone()[0]
    )
    for state in graph.version.manifest.states:
        for reduced in (False, True):
            resolution = library.resolve_preview(
                buddy.id, state=state, reduced_motion=reduced
            )
            assert resolution.frames, state
            assert resolution.resolved_state == state
    # Seeding creates no Persona, legacy or otherwise.
    assert service.list_persona_profiles(include_deleted=True) == []
    assert PersonaVisualRepository(db).get_active_persona_pack(LEGACY_PERSONA_ID) is None


def test_restart_reuses_the_buddy_without_reading_resources(components, monkeypatch):
    db, _service, _coordinator, root = components
    first = install(components)
    graph = BuddyLibrary(db, root).get_graph(first.id)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Repeated readiness reopened bundled resources")

    monkeypatch.setattr(library_module, "files", unexpected)
    restarted = BuddyLibrary(db, root)
    assert restarted.ensure_builtin() == first
    assert restarted.list_buddies() == (first,)
    assert restarted.get_graph(first.id).identity == graph.identity


def test_failed_publication_leaves_no_buddy_and_retries(components, monkeypatch):
    db, _service, _coordinator, root = components

    def fail(*_args, **_kwargs):
        raise ValueError("injected publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(library_module, "publish_persona_visual", fail)
        with pytest.raises(ValueError):
            install(components)
    assert BuddyLibrary(db, root).list_buddies() == ()
    assert _buddy_rows(db) == 0
    assert install(components).name == "pixel-migu"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_postcommit_interruption_keeps_committed_frames(
    components, monkeypatch, interruption
):
    db, _service, _coordinator, root = components
    publish = library_module.publish_persona_visual
    committed = []

    def interrupt_after_commit(repository, snapshot, **kwargs):
        publish(repository, snapshot, **kwargs)
        committed.append(snapshot.buddy_id)
        raise interruption

    with monkeypatch.context() as patch:
        patch.setattr(library_module, "publish_persona_visual", interrupt_after_commit)
        with pytest.raises(interruption):
            install(components)
    restarted = BuddyLibrary(db, root)
    buddy = restarted.ensure_builtin()
    # The committed owner survives; the retry neither removes nor replaces it.
    assert buddy.id == committed[0]
    assert restarted.list_buddies() == (buddy,)
    result = restarted.resolve_preview(buddy.id, state="idle")
    assert result.frames
    assert result.resolved_state == "idle"


def test_app_readiness_installs_after_recovery_and_before_return(
    components, monkeypatch
):
    from types import SimpleNamespace

    from loguru import logger

    import tldw_chatbook.app as app_module

    db, service, coordinator, root = components
    set_app_global(monkeypatch, "get_user_data_dir", lambda: root)
    app = SimpleNamespace(
        chachanotes_db=db,
        persona_actor_pack_coordinator=coordinator,
        local_character_persona_service=service,
        actor_pack_recovery_error=None,
        loguru_logger=logger,
    )
    app_module.TldwCli.ensure_actor_pack_recovery(app)
    assert coordinator.recovery_attempted
    assert BuddyLibrary(db, root).list_buddies()[0].name == "pixel-migu"
    assert (
        PersonaVisualRepository(db).get_active_persona_pack(LEGACY_PERSONA_ID)
        is None
    )
    with pytest.raises(ValueError):
        service.get_persona_profile(LEGACY_PERSONA_ID)


@pytest.mark.parametrize("retirement", ["inactive", "deleted", "archived"])
def test_app_readiness_preserves_retired_legacy_builtin(
    components, monkeypatch, retirement
):
    """A profile whose user retired the legacy built-in Persona gets no Buddy."""
    from dataclasses import replace
    from types import SimpleNamespace

    from loguru import logger

    import tldw_chatbook.app as app_module
    from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
    from tldw_chatbook.Persona_Visual.publication import publish_persona_visual

    db, service, coordinator, root = components
    # The state an older release's installer left: the Persona and its pack.
    service.create_persona_profile(
        {"id": LEGACY_PERSONA_ID, "name": "pixel-migu", "is_active": True}
    )
    repository = PersonaVisualRepository(db)
    source = root / "legacy-source"
    profile = root / "legacy-profile"
    source.mkdir(mode=0o700)
    profile.mkdir(mode=0o700)
    publish_persona_visual(
        repository,
        replace(_snapshot(source), persona_id=LEGACY_PERSONA_ID),
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
    )
    if retirement == "inactive":
        service.update_persona_profile(LEGACY_PERSONA_ID, {"is_active": False})
    elif retirement == "deleted":
        service.delete_persona_profile(LEGACY_PERSONA_ID)
    else:
        graph = repository.get_active_persona_pack(LEGACY_PERSONA_ID)
        repository.archive_binding(
            persona_id=LEGACY_PERSONA_ID, expected_identity=graph.identity
        )
    set_app_global(monkeypatch, "get_user_data_dir", lambda: root)
    app = SimpleNamespace(
        chachanotes_db=db,
        persona_actor_pack_coordinator=coordinator,
        local_character_persona_service=service,
        actor_pack_recovery_error=None,
        loguru_logger=logger,
    )
    app_module.TldwCli.ensure_actor_pack_recovery(app)
    assert BuddyLibrary(db, root).list_buddies() == ()


def test_buddy_seed_reads_resources_from_zip_traversable(components, monkeypatch):
    """A ZIP-backed package must expose the same installed Buddy states."""
    import zipfile

    db, _service, _coordinator, root = components
    source = library_module.files("tldw_chatbook").joinpath(BUNDLE)
    archive_path = root / "resources.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for resource in source.iterdir():
            archive.writestr(f"{BUNDLE}/{resource.name}", resource.read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        monkeypatch.setattr(
            library_module, "files", lambda _package: zipfile.Path(archive)
        )
        buddy = install(components)
    library = BuddyLibrary(db, root)
    graph = library.get_graph(buddy.id)
    assert len(graph.assets) == 64
    for state in graph.version.manifest.states:
        result = library.resolve_preview(buddy.id, state=state)
        assert result.frames, state
        assert result.resolved_state == state


@pytest.mark.parametrize(
    "filename", ["../escape.png", "/escape.png", "nested/escape.png"]
)
def test_buddy_seed_rejects_non_flat_asset_paths(components, monkeypatch, filename):
    """TASK-39124: a bundled filename that is not one flat name is refused
    before it is read, and nothing is published."""
    import json

    db, _service, _coordinator, root = components
    resource_root = _copied_bundle(root, monkeypatch)
    (resource_root.parent / "escape.png").write_bytes(
        (resource_root / "alert-idle-1.png").read_bytes()
    )
    (resource_root / "nested").mkdir()
    (resource_root / "nested/escape.png").write_bytes(b"x")
    declarations = json.loads((resource_root / "assets.json").read_bytes())
    declarations[0]["filename"] = (
        str(resource_root.parent / "escape.png") if filename == "/escape.png" else filename
    )
    (resource_root / "assets.json").write_text(json.dumps(declarations))
    monkeypatch.setattr(
        library_module,
        "publish_persona_visual",
        lambda *args, **kwargs: pytest.fail("invalid asset reached publication"),
    )
    with pytest.raises(ValueError):
        install(components)
    assert _buddy_rows(db) == 0


def test_buddy_seed_rejects_symlinked_resource_leaf(components, monkeypatch):
    """TASK-39124: the seed reads each bundled leaf without following a link."""
    db, _service, _coordinator, root = components
    resource_root = _copied_bundle(root, monkeypatch)
    resource = resource_root / "alert-idle-1.png"
    external = root / "external.png"
    resource.replace(external)
    try:
        resource.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation unavailable")
    monkeypatch.setattr(
        library_module,
        "publish_persona_visual",
        lambda *args, **kwargs: pytest.fail("linked asset reached publication"),
    )
    with pytest.raises(ValueError):
        install(components)
    assert _buddy_rows(db) == 0


@pytest.mark.parametrize(
    "field,value",
    [("renderer_type", "html"), ("animations", {}), ("unknown_field", True)],
)
def test_buddy_seed_uses_canonical_manifest_validation(
    components, monkeypatch, field, value
):
    """A canonical schema failure installs nothing and leaves no owner row."""
    import json

    db, _service, _coordinator, root = components
    resource_root = _copied_bundle(root, monkeypatch)
    manifest = json.loads((resource_root / "manifest.json").read_bytes())
    manifest[field] = value
    (resource_root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        install(components)
    assert BuddyLibrary(db, root).list_buddies() == ()
    assert _buddy_rows(db) == 0
