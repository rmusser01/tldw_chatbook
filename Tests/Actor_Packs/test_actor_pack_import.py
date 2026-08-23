from __future__ import annotations

import copy
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.importer import (
    ActorPackImportError,
    ActorPackImportService,
)
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


FIXTURES = Path(__file__).parent / "fixtures" / "export-golden"


@pytest.fixture
def import_service(tmp_path: Path) -> ActorPackImportService:
    db = CharactersRAGDB(str(tmp_path / "profile.db"), client_id="actor-pack-import")
    return ActorPackImportService(
        ActorPackRepository(db),
        staging_root=tmp_path / "staging",
        profile_root=tmp_path,
    )


def test_inspects_independent_character_golden_path_free(
    import_service: ActorPackImportService,
) -> None:
    review = import_service.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )

    assert review.actor_kind == "character"
    assert review.portable_uuid == "123e4567-e89b-42d3-a456-426614174000"
    assert review.actor_fields == (("name", "Golden"),)
    assert review.allowed_actions == ("create_new", "create_copy")
    assert review.section_effects == ()
    assert review.portrait.mime_type == "image/png"
    assert review.portrait.width == 1
    assert review.portrait.height == 1
    assert "actor/" not in repr(review)
    assert ".import-" not in repr(review)
    assert "staging" not in repr(review)

    preview = import_service.read_portrait_preview(review)
    assert preview.mime_type == "image/png"
    assert preview.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert "PNG" not in repr(preview)
    assert import_service.cleanup_review(review) is True


def test_inspects_independent_persona_golden(
    import_service: ActorPackImportService,
) -> None:
    review = import_service.inspect_archive(
        (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
    )

    assert review.actor_kind == "persona"
    assert dict(review.actor_fields)["name"] == "Golden"
    assert review.allowed_actions == ("create_new", "create_copy")


def test_rejects_undeclared_member_without_leaving_staging(
    import_service: ActorPackImportService,
    tmp_path: Path,
) -> None:
    source = FIXTURES / "minimal-character.tldw-actor-pack"
    archive = tmp_path / "undeclared.tldw-actor-pack"
    with zipfile.ZipFile(source) as reader, zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_STORED
    ) as writer:
        for info in reader.infolist():
            writer.writestr(info, reader.read(info))
        writer.writestr("actor/extra.txt", b"undeclared")

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(archive.resolve())

    assert raised.value.category == "actor_pack_import_invalid"
    assert str(tmp_path) not in str(raised.value)
    staging = tmp_path / "staging"
    assert not staging.exists() or list(staging.iterdir()) == []


@pytest.mark.parametrize(
    ("name", "external_attr", "flag_bits"),
    [
        ("actor/linked.png", stat.S_IFLNK << 16, 0),
        ("actor/device.png", stat.S_IFCHR << 16, 0),
        ("actor/encrypted.png", stat.S_IFREG << 16, 1),
    ],
)
def test_rejects_link_device_and_encrypted_entries(
    import_service: ActorPackImportService,
    tmp_path: Path,
    name: str,
    external_attr: int,
    flag_bits: int,
) -> None:
    archive = tmp_path / "hostile.tldw-actor-pack"
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = external_attr
    info.flag_bits = flag_bits
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(info, b"x")

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(archive.resolve())

    assert raised.value.category == "actor_pack_import_invalid"


def test_rejects_casefold_alias_collision(
    import_service: ActorPackImportService,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "collision.tldw-actor-pack"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("actor-pack.json", b"{}")
        writer.writestr("actor/actor.json", b"{}")
        writer.writestr("Actor/actor.json", b"{}")

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(archive.resolve())

    assert raised.value.category == "actor_pack_import_invalid"


def test_rejects_unknown_required_feature(
    import_service: ActorPackImportService,
    tmp_path: Path,
) -> None:
    source = FIXTURES / "minimal-character.tldw-actor-pack"
    archive = tmp_path / "feature.tldw-actor-pack"
    with zipfile.ZipFile(source) as reader:
        manifest = json.loads(reader.read("actor-pack.json"))
        files = {
            info.filename: reader.read(info)
            for info in reader.infolist()
            if info.filename != "actor-pack.json"
        }
    manifest["required_features"] = ["future-runtime/v9"]
    material = copy.deepcopy(manifest)
    material.pop("content_digest")
    import hashlib

    manifest["content_digest"] = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as writer:
        writer.writestr(
            "actor-pack.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode(),
        )
        for name, data in files.items():
            writer.writestr(name, data)

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(archive.resolve())

    assert raised.value.category == "actor_pack_import_unsupported"


def test_portrait_preview_rejects_staged_inode_substitution(
    import_service: ActorPackImportService,
) -> None:
    review = import_service.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    candidate = import_service._staging_root / review._candidate_name
    portrait = candidate / "actor" / "portrait.png"
    replacement = candidate / "actor" / "replacement.png"
    replacement.write_bytes(portrait.read_bytes())
    os.replace(replacement, portrait)

    with pytest.raises(ActorPackImportError) as raised:
        import_service.read_portrait_preview(review)

    assert raised.value.category == "actor_pack_import_review_stale"


def test_same_kind_uuid_match_offers_copy_or_explicit_update(
    import_service: ActorPackImportService,
) -> None:
    db = import_service.repository.db
    character_id = db.add_character_card({"name": "Existing Golden"})
    assert character_id is not None
    with db.transaction(immediate=True):
        import_service.repository._assign_identity_in_transaction(
            "character",
            character_id,
            portable_uuid="123e4567-e89b-42d3-a456-426614174000",
        )

    review = import_service.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )

    assert review.uuid_match == "same_kind"
    assert review.allowed_actions == ("create_copy", "update_existing")


def test_cross_kind_uuid_match_is_rejected(
    import_service: ActorPackImportService,
) -> None:
    db = import_service.repository.db
    with db.transaction(immediate=True):
        import_service.repository._assign_identity_in_transaction(
            "persona",
            "existing-persona",
            portable_uuid="123e4567-e89b-42d3-a456-426614174000",
        )

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(
            (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
        )

    assert raised.value.category == "actor_pack_import_identity_conflict"
