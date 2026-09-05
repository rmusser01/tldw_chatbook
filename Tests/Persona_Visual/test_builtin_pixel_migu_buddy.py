"""Fresh-profile Buddy installation through real Persona and SQLite stores."""

import pytest

from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.builtin_pixel_migu import (
    PIXEL_MIGU_PERSONA_ID,
    ensure_builtin_pixel_migu_buddy,
)
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
from tldw_chatbook.Persona_Visual.runtime import resolve_active_persona_visual


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
    _db, service, coordinator, root = components
    return ensure_builtin_pixel_migu_buddy(service, coordinator, profile_root=root)


def test_fresh_install_links_independent_character_and_resolves_baseline_states(
    components,
):
    db, _service, coordinator, root = components
    persona = install(components)
    assert persona["id"] == PIXEL_MIGU_PERSONA_ID
    assert persona["name"] == "pixel-migu"
    character = db.get_character_card_by_id(persona["character_card_id"])
    assert character["extensions"]["tldw/builtin_id"] == "pixel-migu"
    graph = PersonaVisualRepository(db).get_active_persona_pack(persona["id"])
    assert len(graph.assets) == 64
    assert (
        "LicenseRef-User-Supplied"
        in db.execute_query(
            "SELECT source_context_json FROM persona_visual_packs WHERE id = ?",
            (graph.pack.id,),
        ).fetchone()[0]
    )
    for state in (
        "idle",
        "listening",
        "thinking",
        "speaking",
        "approval_needed",
        "tool_running",
        "error",
        "wake_armed",
        "offline",
    ):
        resolution = resolve_active_persona_visual(
            PersonaVisualRepository(db),
            profile_root=root,
            persona_id=persona["id"],
            requested_state=state,
        )
        assert resolution.frames, state
    assert coordinator.repository.list_persona_intents() == ()


def test_restart_preserves_edits_tombstone_and_existing_selection(components):
    db, service, coordinator, root = components
    service.create_persona_profile(
        {"id": "chosen", "name": "Chosen", "is_active": True}
    )
    install(components)
    service.update_persona_profile(
        PIXEL_MIGU_PERSONA_ID, {"name": "My custom name", "is_active": False}
    )
    service.delete_persona_profile(PIXEL_MIGU_PERSONA_ID)
    before = service.persona_store_path.read_bytes()
    service = LocalCharacterPersonaService(
        db, persona_store_path=root / "personas.json"
    )
    coordinator = PersonaActorPackCoordinator(ActorPackRepository(db), service)
    ensure_builtin_pixel_migu_buddy(service, coordinator, profile_root=root)
    assert service.persona_store_path.read_bytes() == before
    assert service.get_persona_profile("chosen")["is_active"] is True
    assert len(service.list_persona_profiles(include_deleted=True)) == 2


def test_failed_graph_activation_rolls_back_persona_and_retries(
    components, monkeypatch
):
    db, service, coordinator, _root = components
    original = PersonaVisualRepository._activate_new_pack_in_transaction

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("injected activation failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            PersonaVisualRepository, "_activate_new_pack_in_transaction", fail
        )
        with pytest.raises(ValueError):
            install(components)
    assert service.list_persona_profiles() == []
    assert coordinator.repository.get_identity("persona", PIXEL_MIGU_PERSONA_ID) is None
    assert coordinator.repository.list_persona_intents() == ()
    assert (
        PersonaVisualRepository(db).get_active_persona_pack(PIXEL_MIGU_PERSONA_ID)
        is None
    )
    assert install(components)["name"] == "pixel-migu"


def test_restart_preserves_archived_visual_binding(components):
    db, service, coordinator, root = components
    install(components)
    repository = PersonaVisualRepository(db)
    graph = repository.get_active_persona_pack(PIXEL_MIGU_PERSONA_ID)
    repository.archive_binding(
        persona_id=PIXEL_MIGU_PERSONA_ID, expected_identity=graph.identity
    )
    service = LocalCharacterPersonaService(
        db, persona_store_path=root / "personas.json"
    )
    coordinator = PersonaActorPackCoordinator(ActorPackRepository(db), service)
    ensure_builtin_pixel_migu_buddy(service, coordinator, profile_root=root)
    assert repository.get_active_persona_pack(PIXEL_MIGU_PERSONA_ID) is None


def test_repeated_readiness_avoids_resource_io_and_duplicates(components, monkeypatch):
    from tldw_chatbook.Persona_Visual import builtin_pixel_migu

    db, service, _coordinator, _root = components
    first = install(components)
    graph = PersonaVisualRepository(db).get_active_persona_pack(PIXEL_MIGU_PERSONA_ID)

    def unexpected(*args, **kwargs):
        raise AssertionError("Repeated readiness reopened bundled resources")

    monkeypatch.setattr(builtin_pixel_migu, "files", unexpected)
    assert install(components) == first
    assert len(service.list_persona_profiles()) == 1
    assert (
        PersonaVisualRepository(db).get_active_persona_pack(PIXEL_MIGU_PERSONA_ID)
        == graph
    )


def test_app_readiness_installs_after_recovery_and_before_return(
    components, monkeypatch
):
    from types import SimpleNamespace

    from loguru import logger

    import tldw_chatbook.app as app_module

    db, service, coordinator, root = components
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: root)
    app = SimpleNamespace(
        chachanotes_db=db,
        persona_actor_pack_coordinator=coordinator,
        local_character_persona_service=service,
        actor_pack_recovery_error=None,
        loguru_logger=logger,
    )
    app_module.TldwCli.ensure_actor_pack_recovery(app)
    assert coordinator.recovery_attempted
    assert service.get_persona_profile(PIXEL_MIGU_PERSONA_ID)["name"] == "pixel-migu"
    assert (
        PersonaVisualRepository(db).get_active_persona_pack(PIXEL_MIGU_PERSONA_ID)
        is not None
    )


def test_racing_service_loser_cleanup_preserves_winners_frames(components, monkeypatch):
    db, service, coordinator, root = components
    other_service = LocalCharacterPersonaService(
        db, persona_store_path=root / "personas.json"
    )
    other_coordinator = PersonaActorPackCoordinator(
        ActorPackRepository(db), other_service
    )
    create = coordinator.create_persona

    def competing_create(*args, **kwargs):
        # Both services observed no Persona. The second commits after the first
        # copies assets but before the first prepares its JSON CAS operation.
        ensure_builtin_pixel_migu_buddy(
            other_service, other_coordinator, profile_root=root
        )
        return create(*args, **kwargs)

    monkeypatch.setattr(coordinator, "create_persona", competing_create)
    with pytest.raises(ValueError):
        install(components)
    repository = PersonaVisualRepository(db)
    assert repository.get_active_persona_pack(PIXEL_MIGU_PERSONA_ID) is not None
    for state in ("idle", "thinking", "tool_running"):
        result = resolve_active_persona_visual(
            repository, PIXEL_MIGU_PERSONA_ID, root, state
        )
        assert result.frames
        assert result.resolved_state == state
    assert len(other_service.list_persona_profiles()) == 1
    # The losing instance must discover the winning profile on its next read.
    assert install(components)["id"] == PIXEL_MIGU_PERSONA_ID
    assert len(service.list_persona_profiles()) == 1


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_postcommit_interruption_keeps_committed_frames(
    components, monkeypatch, interruption
):
    db, _service, coordinator, root = components
    create = coordinator.create_persona

    def interrupt_after_commit(*args, **kwargs):
        create(*args, **kwargs)
        raise interruption

    monkeypatch.setattr(coordinator, "create_persona", interrupt_after_commit)
    with pytest.raises(interruption):
        install(components)
    restarted = LocalCharacterPersonaService(
        db, persona_store_path=root / "personas.json"
    )
    ensure_builtin_pixel_migu_buddy(
        restarted,
        PersonaActorPackCoordinator(ActorPackRepository(db), restarted),
        profile_root=root,
    )
    result = resolve_active_persona_visual(
        PersonaVisualRepository(db), PIXEL_MIGU_PERSONA_ID, root, "idle"
    )
    assert result.frames
    assert result.resolved_state == "idle"
