"""Persona Shared Visual Identity candidate and publication contracts."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import pytest
from PIL import Image

from Tests.Character_Chat.test_persona_visual_identity_resolution import (
    _LocalPersonaService,
    _activate,
)
from tldw_chatbook.Character_Chat.persona_visual_identity import (
    capture_local_persona_visual_identity,
    local_persona_visual_identity_is_current,
)
from tldw_chatbook.Character_Chat.visual_identity import (
    CANONICAL_EXPRESSION_SLOTS,
    VisualIdentityPublicationError,
    create_visual_identity_candidate,
    publish_visual_identity_candidate,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        tmp_path / "persona-svi-publication.db", "persona-svi-publication"
    )
    yield database
    database.close_connection()


@pytest.fixture
def service(db: CharactersRAGDB) -> _LocalPersonaService:
    local = _LocalPersonaService()
    portrait = _png((4, 5, 6))
    character_id = db.add_character_card(
        {"name": "Persona portrait", "image": portrait}
    )
    assert character_id is not None
    local.personas["p-1"]["character_card_id"] = int(character_id)
    local.characters = {
        int(character_id): {
            "id": int(character_id),
            "version": 3,
            "deleted": False,
            "image": portrait,
        }
    }
    return local


def _candidate(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
):
    authority = capture_local_persona_visual_identity(service, "p-1")
    assert authority is not None
    return create_visual_identity_candidate(
        db,
        actor_kind="persona",
        actor_id="p-1",
        actor_authority=authority.cache_identity,
        actor_guard=lambda: local_persona_visual_identity_is_current(
            service, authority
        ),
    )


def _png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def test_unbound_local_persona_can_create_empty_canonical_candidate(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
) -> None:
    candidate = _candidate(db, service)

    assert candidate.actor_kind == "persona"
    assert candidate.actor_id == "p-1"
    assert candidate.old_pack_id is None
    assert candidate.old_version_id is None
    assert candidate.old_binding_id is None
    assert candidate.actor_authority
    assert (
        tuple(asset["expression_key"] for asset in candidate.assets)
        == CANONICAL_EXPRESSION_SLOTS
    )
    assert all(asset["bytes"] == 0 for asset in candidate.assets)
    assert VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1") is None


def test_bound_persona_candidate_snapshots_exact_binding_and_actor_authority(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    graph = _activate(db, tmp_path, ("neutral", "thinking"))

    candidate = _candidate(db, service)

    assert candidate.old_pack_id == graph["pack"]["id"]
    assert candidate.old_pack_version == graph["pack"]["version"]
    assert candidate.old_version_id == graph["version"]["id"]
    assert candidate.old_binding_id == graph["binding"]["id"]
    assert candidate.old_binding_version == graph["binding"]["version"]
    assert (
        candidate.actor_authority
        == capture_local_persona_visual_identity(service, "p-1").cache_identity
    )


@pytest.mark.parametrize(
    "mutation",
    (
        {"backend": "server"},
        {"deleted": True},
        {"is_active": False},
    ),
)
def test_persona_candidate_rejects_server_deleted_disabled_missing_and_stale_actor(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    mutation: dict[str, object],
) -> None:
    authority = capture_local_persona_visual_identity(service, "p-1")
    assert authority is not None
    service.personas["p-1"].update(mutation)

    with pytest.raises(ValueError, match="^visual_identity_actor_changed$"):
        create_visual_identity_candidate(
            db,
            actor_kind="persona",
            actor_id="p-1",
            actor_authority=authority.cache_identity,
            actor_guard=lambda: local_persona_visual_identity_is_current(
                service, authority
            ),
        )


def test_candidate_replace_clear_and_cancel_leave_active_version_unchanged(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    graph = _activate(db, tmp_path, ("neutral", "thinking"))
    candidate = _candidate(db, service)

    candidate.stage_replacement("neutral", b"replacement")
    candidate.stage_clear("thinking")
    candidate.cancel()

    current = VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1")
    assert current is not None
    assert current["version"]["id"] == graph["version"]["id"]
    assert current["binding"]["version"] == graph["binding"]["version"]
    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_candidate_cancelled$"
    ):
        candidate.stage_replacement("neutral", b"later")


def test_unbound_persona_publish_creates_one_pack_version_assets_and_binding(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    candidate = _candidate(db, service)
    candidate.stage_replacement("neutral", _png((10, 20, 30)))

    result = publish_visual_identity_candidate(db, candidate, user_data_dir=tmp_path)

    graph = VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1")
    assert graph is not None
    assert result.old_pack_id is None
    assert result.old_version_id is None
    assert result.new_pack_id == graph["pack"]["id"]
    assert result.new_version_id == graph["version"]["id"]
    assert graph["version"]["version_number"] == 1
    assert [asset["expression_key"] for asset in graph["assets"]] == ["neutral"]
    assert result.version_directory.is_dir()


def test_bound_persona_publish_appends_immutable_version_and_preserves_old_rows(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    first = _candidate(db, service)
    first.stage_replacement("neutral", _png((10, 20, 30)))
    first_result = publish_visual_identity_candidate(db, first, user_data_dir=tmp_path)
    old_graph = VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1")
    assert old_graph is not None

    second = _candidate(db, service)
    second.stage_replacement("neutral", _png((30, 20, 10)))
    second_result = publish_visual_identity_candidate(
        db, second, user_data_dir=tmp_path
    )

    new_graph = VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1")
    assert new_graph is not None
    assert second_result.old_version_id == first_result.new_version_id
    assert second_result.new_version_id != first_result.new_version_id
    assert new_graph["version"]["version_number"] == 2
    old_row = db.execute_query(
        "SELECT id FROM visual_identity_pack_versions WHERE id = ?",
        (first_result.new_version_id,),
    ).fetchone()
    assert old_row is not None
    assert first_result.version_directory.is_dir()
    assert second_result.version_directory.is_dir()


@pytest.mark.parametrize("change", ("revision", "source", "portrait"))
def test_persona_revision_source_binding_version_and_portrait_change_fail_closed(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
    change: str,
) -> None:
    candidate = _candidate(db, service)
    candidate.stage_replacement("neutral", _png((1, 2, 3)))
    if change == "revision":
        service.personas["p-1"]["version"] = 5
    elif change == "source":
        service.personas["p-1"]["backend"] = "server"
    else:
        character_id = service.personas["p-1"]["character_card_id"]
        service.characters[character_id]["version"] = 4

    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_actor_changed$"
    ):
        publish_visual_identity_candidate(db, candidate, user_data_dir=tmp_path)

    assert VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1") is None


def test_persona_binding_version_change_fails_closed(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    first = _candidate(db, service)
    first.stage_replacement("neutral", _png((5, 6, 7)))
    publish_visual_identity_candidate(db, first, user_data_dir=tmp_path)
    candidate = _candidate(db, service)
    candidate.stage_replacement("neutral", _png((7, 6, 5)))
    old_version_id = candidate.old_version_id
    with db.transaction():
        db.execute_query(
            "UPDATE visual_identity_bindings SET version = version + 1 WHERE id = ?",
            (candidate.old_binding_id,),
        )

    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_binding_changed$"
    ):
        publish_visual_identity_candidate(db, candidate, user_data_dir=tmp_path)

    current = VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1")
    assert current is not None
    assert current["version"]["id"] == old_version_id


def test_persona_authority_is_rechecked_inside_reserved_sqlite_transaction(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    authority = capture_local_persona_visual_identity(service, "p-1")
    assert authority is not None
    observations: list[bool] = []

    def guard() -> bool:
        in_transaction = db.get_connection().in_transaction
        observations.append(in_transaction)
        return not in_transaction and local_persona_visual_identity_is_current(
            service, authority
        )

    candidate = create_visual_identity_candidate(
        db,
        actor_kind="persona",
        actor_id="p-1",
        actor_authority=authority.cache_identity,
        actor_guard=guard,
    )
    candidate.stage_replacement("neutral", _png((1, 2, 3)))

    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_actor_changed$"
    ):
        publish_visual_identity_candidate(db, candidate, user_data_dir=tmp_path)

    assert observations == [False, False, True]
    assert VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1") is None


def test_failed_cancelled_or_stale_publish_keeps_prior_binding_and_cleans_owned_staging(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    cancelled = _candidate(db, service)
    cancelled.stage_replacement("neutral", _png((1, 2, 3)))
    cancelled.cancel()
    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_candidate_cancelled$"
    ):
        publish_visual_identity_candidate(db, cancelled, user_data_dir=tmp_path)

    stale = _candidate(db, service)
    stale.stage_replacement("neutral", _png((3, 2, 1)))
    service.personas["p-1"]["version"] = 5
    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_actor_changed$"
    ):
        publish_visual_identity_candidate(db, stale, user_data_dir=tmp_path)

    assert VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1") is None
    publication_root = tmp_path / "visual_identities"
    assert not publication_root.exists() or not any(
        publication_root.rglob(".staging-*")
    )


def test_unbound_persona_publish_rejects_concurrent_binding_creation(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(db, service)
    candidate.stage_replacement("neutral", _png((8, 9, 10)))
    original_activate = VisualIdentityRepository.activate_pack
    competitor: dict[str, object] = {}

    def race(self, **kwargs):
        if not competitor:
            competitor.update(
                original_activate(
                    self,
                    pack={
                        "title": "Concurrent Persona reactions",
                        "description": "",
                        "default_expression_key": "neutral",
                        "source_kind": "manual",
                        "source_context": {"source_id": "concurrent.fixture"},
                    },
                    manifest=kwargs["manifest"],
                    assets=kwargs["assets"],
                    actor_kind="persona",
                    actor_id="p-1",
                )
            )
        return original_activate(self, **kwargs)

    monkeypatch.setattr(VisualIdentityRepository, "activate_pack", race)

    with pytest.raises(
        VisualIdentityPublicationError, match="^visual_identity_binding_changed$"
    ):
        publish_visual_identity_candidate(db, candidate, user_data_dir=tmp_path)

    current = VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1")
    assert current is not None
    assert current["pack"]["id"] == competitor["pack"]["id"]
