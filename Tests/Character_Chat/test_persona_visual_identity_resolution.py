"""Local Persona authority and Shared Visual Identity resolution contracts."""

from __future__ import annotations

import dataclasses
import hashlib
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat import persona_visual_identity as persona_svi
from tldw_chatbook.Character_Chat.persona_visual_identity import (
    LocalPersonaVisualIdentityAuthority,
    capture_local_persona_visual_identity,
    local_persona_visual_identity_is_current,
    resolve_persona_visual_identity,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository


_PNG = b"\x89PNG\r\n\x1a\n" + b"persona-portrait"


class _LocalPersonaService:
    def __init__(self) -> None:
        self.personas = {
            "p-1": {
                "backend": "local",
                "id": "p-1",
                "version": 4,
                "deleted": False,
                "is_active": True,
                "character_card_id": 7,
            }
        }
        self.characters = {
            7: {
                "id": 7,
                "version": 3,
                "deleted": False,
                "image": _PNG,
            }
        }

    def get_persona_profile(self, persona_id: str):
        record = self.personas.get(persona_id)
        if record is None:
            raise ValueError("local_persona_profile_not_found")
        return deepcopy(record)

    def get_character(self, character_id: int):
        record = self.characters.get(character_id)
        if record is None:
            raise ValueError("character_not_found")
        return deepcopy(record)


def test_capture_requires_exact_local_persona_id_revision_and_active_state() -> None:
    service = _LocalPersonaService()

    authority = capture_local_persona_visual_identity(service, "p-1")

    assert type(authority) is LocalPersonaVisualIdentityAuthority
    assert authority.source == "local"
    assert authority.persona_id == "p-1"
    assert authority.persona_revision == 4
    assert authority.portrait is not None
    assert authority.portrait.portrait_id == "local-character:7"
    assert authority.portrait.revision == 3
    assert authority.portrait.content_type == "image/png"
    assert authority.portrait.sha256 == hashlib.sha256(_PNG).hexdigest()
    assert authority.portrait.data == _PNG
    assert dataclasses.is_dataclass(authority)
    with pytest.raises(dataclasses.FrozenInstanceError):
        authority.persona_revision = 5  # type: ignore[misc]
    assert not hasattr(authority, "__dict__")
    assert "persona-portrait" not in repr(authority)


@pytest.mark.parametrize(
    ("mutation", "persona_id"),
    (
        ({"backend": "server"}, "p-1"),
        ({"id": "p-2"}, "p-1"),
        ({"version": True}, "p-1"),
        ({"version": 0}, "p-1"),
        ({"deleted": True}, "p-1"),
        ({"is_active": False}, "p-1"),
        ({}, "missing"),
    ),
)
def test_capture_rejects_deleted_disabled_missing_and_server_records(
    mutation: dict[str, object], persona_id: str
) -> None:
    service = _LocalPersonaService()
    if persona_id in service.personas:
        service.personas[persona_id].update(mutation)

    assert capture_local_persona_visual_identity(service, persona_id) is None


@pytest.mark.parametrize(
    "character_mutation",
    (
        {"id": 8},
        {"version": True},
        {"version": -1},
        {"deleted": True},
        {"image": b"not-an-image"},
        {"image": b"\x89PNG\r\n\x1a\n" + b"x" * (25 * 1024 * 1024)},
    ),
)
def test_linked_character_portrait_is_bounded_and_path_free(
    character_mutation: dict[str, object],
) -> None:
    service = _LocalPersonaService()
    service.characters[7].update(character_mutation)

    authority = capture_local_persona_visual_identity(service, "p-1")

    assert authority is not None
    assert authority.portrait is None
    assert "/Users/" not in repr(authority)


def test_authority_revalidation_detects_revision_aba_and_linked_portrait_change() -> (
    None
):
    service = _LocalPersonaService()
    authority = capture_local_persona_visual_identity(service, "p-1")
    assert authority is not None
    assert local_persona_visual_identity_is_current(service, authority) is True

    service.personas["p-1"]["version"] = 5
    assert local_persona_visual_identity_is_current(service, authority) is False

    service.personas["p-1"]["version"] = 4
    service.characters[7]["image"] = b"\x89PNG\r\n\x1a\nchanged"
    assert local_persona_visual_identity_is_current(service, authority) is False

    service.characters[7]["image"] = _PNG
    service.characters[7]["version"] = 4
    assert local_persona_visual_identity_is_current(service, authority) is False


def test_capture_rejects_stateful_mapping_subclasses() -> None:
    service = _LocalPersonaService()
    record = service.personas["p-1"]

    class _StatefulRecord(dict):
        pass

    service.get_persona_profile = lambda _persona_id: _StatefulRecord(record)  # type: ignore[method-assign]

    assert capture_local_persona_visual_identity(service, "p-1") is None


def _image_bytes(image_format: str, color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format=image_format)
    return output.getvalue()


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "persona-svi.db", "persona-svi-test")
    yield database
    database.close_connection()


@pytest.fixture
def service(db: CharactersRAGDB) -> _LocalPersonaService:
    local = _LocalPersonaService()
    portrait = _image_bytes("PNG", (2, 4, 8))
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


def _asset(
    profile_root: Path,
    expression_key: str,
    color: tuple[int, int, int],
) -> dict[str, Any]:
    data = _image_bytes("WEBP", color)
    filename = expression_key.replace("custom:", "") + ".webp"
    relative_path = f"personas/p-1/{filename}"
    path = profile_root / "visual_identities" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "expression_key": expression_key,
        "original_expression_key": expression_key,
        "display_label": expression_key.title(),
        "source_filename": filename,
        "storage_relpath": relative_path,
        "content_type": "image/webp",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": 8,
        "height": 8,
        "source_context": {"fixture": True},
        "is_animated": False,
        "frame_count": 1,
    }


def _activate(
    db: CharactersRAGDB,
    profile_root: Path,
    keys: tuple[str, ...],
    *,
    default: str = "neutral",
) -> dict[str, Any]:
    return VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Persona reactions",
            "default_expression_key": default,
            "source_kind": "manual",
            "source_context": {"source_id": "persona.fixture"},
        },
        manifest={"schema_id": "resolver/v1", "fixture": "persona"},
        assets=[
            _asset(profile_root, key, (index * 40, 70, 120))
            for index, key in enumerate(keys, 1)
        ],
        actor_kind="persona",
        actor_id="p-1",
    )


@pytest.mark.parametrize(
    ("manual", "state", "keys", "default", "expected", "source"),
    (
        (
            "admiration",
            "thinking",
            ("custom:admiration", "thinking", "neutral"),
            "neutral",
            "custom:admiration",
            "pack_manual",
        ),
        (
            None,
            "thinking",
            ("thinking", "neutral"),
            "neutral",
            "thinking",
            "pack_operational",
        ),
        (None, "error", ("happy", "neutral"), "happy", "happy", "pack_default"),
        (None, "error", ("neutral",), "missing", "neutral", "pack_neutral"),
    ),
)
def test_persona_bound_pack_resolves_manual_requested_default_and_neutral_order(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
    manual: str | None,
    state: str,
    keys: tuple[str, ...],
    default: str,
    expected: str,
    source: str,
) -> None:
    _activate(db, tmp_path, keys, default=default)

    result = resolve_persona_visual_identity(
        db,
        service,
        persona_id="p-1",
        requested_state=state,
        manual_expression_key=manual,
        user_data_dir=tmp_path,
    )

    assert result.actor_kind == "persona"
    assert result.actor_id == "p-1"
    assert result.resolved_expression_key == expected
    assert result.resolution_source == source
    assert result.image_bytes


def test_persona_missing_pack_asset_falls_back_to_linked_portrait(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    graph = _activate(db, tmp_path, ("thinking", "neutral"))
    for asset in graph["assets"]:
        (tmp_path / "visual_identities" / asset["storage_relpath"]).unlink()

    result = resolve_persona_visual_identity(
        db,
        service,
        persona_id="p-1",
        requested_state="thinking",
        user_data_dir=tmp_path,
    )

    portrait = next(iter(service.characters.values()))["image"]
    assert result.resolution_source == "persona_portrait"
    assert result.fallback_reason == "pack_assets_unavailable"
    assert result.image_bytes == portrait
    assert result.storage_relpath is None


def test_persona_unavailable_or_changed_authority_returns_actor_unavailable(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _activate(db, tmp_path, ("neutral",))
    service.personas["p-1"]["deleted"] = True
    unavailable = resolve_persona_visual_identity(
        db, service, persona_id="p-1", requested_state="idle", user_data_dir=tmp_path
    )
    assert unavailable.fallback_reason == "actor_unavailable"

    service.personas["p-1"]["deleted"] = False
    original_query = db.execute_query
    changed = False

    def mutate_after_query(query: str, params=None):
        nonlocal changed
        cursor = original_query(query, params)
        if "visual_identity_resolver" in query and not changed:
            service.personas["p-1"]["version"] = 5
            changed = True
        return cursor

    monkeypatch.setattr(db, "execute_query", mutate_after_query)
    stale = resolve_persona_visual_identity(
        db, service, persona_id="p-1", requested_state="idle", user_data_dir=tmp_path
    )
    assert stale.resolution_source == "placeholder"
    assert stale.fallback_reason == "actor_unavailable"
    assert stale.image_bytes is None


def test_persona_delete_source_change_and_restore_reuse_exact_binding(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    graph = _activate(db, tmp_path, ("neutral",))
    binding_id = graph["binding"]["id"]

    service.personas["p-1"]["deleted"] = True
    deleted = resolve_persona_visual_identity(
        db, service, persona_id="p-1", requested_state="idle", user_data_dir=tmp_path
    )
    service.personas["p-1"].update(
        {"deleted": False, "backend": "server", "version": 5}
    )
    server = resolve_persona_visual_identity(
        db, service, persona_id="p-1", requested_state="idle", user_data_dir=tmp_path
    )

    dormant = VisualIdentityRepository(db).get_active_actor_pack("persona", "p-1")
    assert deleted.fallback_reason == "actor_unavailable"
    assert server.fallback_reason == "actor_unavailable"
    assert dormant is not None
    assert dormant["binding"]["id"] == binding_id

    service.personas["p-1"].update({"backend": "local", "version": 6})
    restored = resolve_persona_visual_identity(
        db, service, persona_id="p-1", requested_state="idle", user_data_dir=tmp_path
    )

    assert restored.resolved_expression_key == "neutral"
    assert f"binding_id={binding_id}" in restored.cache_identity
    assert "persona_revision=6" in restored.cache_identity


def test_persona_authority_change_during_portrait_decode_fails_closed(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_validate = persona_svi._shared._validate_fallback_image

    def mutate_during_decode(value):
        result = original_validate(value)
        service.personas["p-1"]["version"] = 5
        return result

    monkeypatch.setattr(
        persona_svi._shared, "_validate_fallback_image", mutate_during_decode
    )

    result = resolve_persona_visual_identity(
        db, service, persona_id="p-1", requested_state="idle", user_data_dir=tmp_path
    )

    assert result.resolution_source == "placeholder"
    assert result.fallback_reason == "actor_unavailable"
    assert result.image_bytes is None


def test_persona_cache_identity_contains_full_actor_binding_version_asset_and_portrait_identity(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
) -> None:
    graph = _activate(db, tmp_path, ("neutral",))
    result = resolve_persona_visual_identity(
        db, service, persona_id="p-1", requested_state="idle", user_data_dir=tmp_path
    )
    cache = set(result.cache_identity)
    portrait = next(iter(service.characters.values()))
    assert {
        "actor_kind=persona",
        "actor_id=p-1",
        "actor_source=local",
        "persona_revision=4",
        f"portrait_id=local-character:{portrait['id']}",
        "portrait_revision=3",
        f"binding_id={graph['binding']['id']}",
        f"binding_version={graph['binding']['version']}",
        f"pack_id={graph['pack']['id']}",
        f"pack_revision={graph['pack']['version']}",
        f"pack_version_id={graph['version']['id']}",
        f"pack_version_number={graph['version']['version_number']}",
        f"asset_id={graph['assets'][0]['id']}",
        f"sha256={graph['assets'][0]['sha256']}",
    } <= cache
    assert any(token.startswith("manifest_sha256=") for token in cache)
    assert any(token.startswith("portrait_sha256=") for token in cache)


def test_persona_resolution_does_not_read_character_legacy_expression_rows(
    db: CharactersRAGDB,
    service: _LocalPersonaService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _activate(db, tmp_path, ("neutral",))
    queries: list[str] = []
    original_query = db.execute_query

    def record_query(query: str, params=None):
        queries.append(query)
        return original_query(query, params)

    monkeypatch.setattr(db, "execute_query", record_query)
    result = resolve_persona_visual_identity(
        db, service, persona_id="p-1", requested_state="idle", user_data_dir=tmp_path
    )

    assert result.image_bytes
    assert not any("character_expression_images" in query for query in queries)
