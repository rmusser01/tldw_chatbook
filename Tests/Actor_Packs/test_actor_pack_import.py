from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs import importer as importer_module
from tldw_chatbook.Actor_Packs.contracts import (
    ActorPackValidationError,
    canonical_json_bytes,
    validate_actor_payload,
)
from tldw_chatbook.Actor_Packs.importer import (
    ActorPackImportError,
    ActorPackImportService,
)
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Utils.private_paths import PrivatePathResult, PrivatePathStatus
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


def test_inspect_archive_uses_central_path_validation(
    import_service: ActorPackImportService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    calls: list[tuple[Path, Path]] = []

    def validate(selected: Path, base_directory: Path, **_kwargs: object) -> Path:
        calls.append((selected, base_directory))
        return selected

    monkeypatch.setattr(importer_module, "validate_path", validate)

    review = import_service.inspect_archive(archive)

    assert calls == [(archive, archive.parent)]
    import_service.cleanup_review(review)


def test_inspect_archive_documents_public_contract() -> None:
    docstring = inspect.getdoc(ActorPackImportService.inspect_archive) or ""

    assert "Args:" in docstring
    assert "Returns:" in docstring
    assert "Raises:" in docstring


def test_candidate_creation_failure_removes_private_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    real_open = importer_module.os.open

    def fail_marker(path: object, *args: object, **kwargs: object) -> int:
        if Path(path).name == ".actor-pack-import":
            raise OSError("injected marker failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(importer_module.os, "open", fail_marker)

    with pytest.raises(OSError, match="injected marker failure"):
        importer_module._create_candidate(root)

    assert list(root.iterdir()) == []


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
    assert review.section_effects == (
        (
            "shared-visual-identity",
            "Create New: Not included — no visual binding will be created; "
            "Create Copy: Not included — no visual binding will be created",
        ),
    )
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
    with (
        zipfile.ZipFile(source) as reader,
        zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as writer,
    ):
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
    "external_attr",
    [
        stat.S_IFLNK << 16,
        stat.S_IFCHR << 16,
    ],
)
def test_rejects_link_and_device_entries_in_an_otherwise_valid_archive(
    import_service: ActorPackImportService,
    tmp_path: Path,
    external_attr: int,
) -> None:
    archive = tmp_path / "hostile.tldw-actor-pack"
    with zipfile.ZipFile(FIXTURES / "minimal-character.tldw-actor-pack") as reader:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as writer:
            for info in reader.infolist():
                copied = copy.copy(info)
                if info.filename == "actor/portrait.png":
                    copied.create_system = 3
                    copied.external_attr = external_attr
                writer.writestr(copied, reader.read(info))

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(archive.resolve())

    assert raised.value.category == "actor_pack_import_invalid"


def test_rejects_encrypted_flag_in_an_otherwise_valid_archive(
    import_service: ActorPackImportService,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "encrypted.tldw-actor-pack"
    payload = bytearray((FIXTURES / "minimal-character.tldw-actor-pack").read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        cursor = 0
        while (cursor := payload.find(signature, cursor)) >= 0:
            flags = int.from_bytes(
                payload[cursor + flag_offset : cursor + flag_offset + 2]
            )
            payload[cursor + flag_offset : cursor + flag_offset + 2] = (
                flags | 1
            ).to_bytes(2, "little")
            cursor += 4
    archive.write_bytes(payload)

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(archive.resolve())

    assert raised.value.category == "actor_pack_import_invalid"


def test_rejects_truncated_archive_without_leaving_staging(
    import_service: ActorPackImportService,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "truncated.tldw-actor-pack"
    payload = (FIXTURES / "minimal-character.tldw-actor-pack").read_bytes()
    archive.write_bytes(payload[:-12])

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(archive.resolve())

    assert raised.value.category == "actor_pack_import_invalid"
    assert list(import_service._staging_root.iterdir()) == []


def test_rejects_oversized_central_directory_count_before_zipfile_parsing(
    import_service: ActorPackImportService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "many-members.tldw-actor-pack"
    count = 4098
    archive.write_bytes(
        b"PK\x05\x06"
        + (0).to_bytes(2, "little")
        + (0).to_bytes(2, "little")
        + count.to_bytes(2, "little")
        + count.to_bytes(2, "little")
        + (0).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + (0).to_bytes(2, "little")
    )
    parsed = False

    def forbidden_zipfile(*_args, **_kwargs):
        nonlocal parsed
        parsed = True
        raise AssertionError("ZipFile must not parse an over-budget directory")

    monkeypatch.setattr(importer_module.zipfile, "ZipFile", forbidden_zipfile)

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(archive.resolve())

    assert raised.value.category == "actor_pack_import_invalid"
    assert parsed is False


def test_section_image_rejects_suffix_mime_mismatch() -> None:
    from .conftest import PNG_1X1

    with pytest.raises(ValueError):
        importer_module._section_image("persona-runtime/assets/frame.jpg", PNG_1X1)


def test_section_image_rejects_decode_budget_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    loaded = False

    class OversizedImage:
        format = "PNG"
        size = (4097, 4096)
        n_frames = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def load(self) -> None:
            nonlocal loaded
            loaded = True

    monkeypatch.setattr(Image, "open", lambda _stream: OversizedImage())

    with pytest.raises(ValueError):
        importer_module._section_image("persona-runtime/assets/frame.png", b"image")

    assert loaded is False


@pytest.mark.parametrize(
    "actor_fields",
    [
        {"name": "Guide", "voice_defaults": {"vad_threshold": {"bad": True}}},
        {"name": "Guide", "setup": {"status": "impossible", "version": 0}},
    ],
)
def test_import_rejects_invalid_nested_persona_mutation_fields(
    actor_fields: dict[str, object],
) -> None:
    payload = canonical_json_bytes(
        {
            "schema": "tldw.actor/v1",
            "actor_kind": "persona",
            "portable_uuid": "123e4567-e89b-42d3-a456-426614174000",
            "data": actor_fields,
        }
    )
    with pytest.raises(ActorPackValidationError) as raised:
        validate_actor_payload(
            payload,
            actor_kind="persona",
            portable_uuid="123e4567-e89b-42d3-a456-426614174000",
        )

    assert raised.value.category == "actor_pack_actor_invalid"


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
    assert review.section_effects == (
        (
            "shared-visual-identity",
            "Create Copy: Not included — no visual binding will be created; "
            "Update Existing: Not included — existing visuals will be preserved",
        ),
    )


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


def test_revalidation_detects_same_inode_source_digest_change(
    import_service: ActorPackImportService, tmp_path: Path
) -> None:
    archive = (tmp_path / "mutable.tldw-actor-pack").resolve()
    shutil.copyfile(FIXTURES / "minimal-character.tldw-actor-pack", archive)
    review = import_service.inspect_archive(archive)
    before = archive.stat()
    with archive.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        original = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([original[0] ^ 1]))
    os.utime(archive, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(ActorPackImportError) as raised:
        import_service.revalidate_review(review)

    assert raised.value.category == "actor_pack_import_review_stale"
    import_service.cleanup_review(review)


def test_revalidation_rechecks_publication_free_space(
    import_service: ActorPackImportService, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = import_service.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    current = shutil.disk_usage(import_service._profile_root)
    monkeypatch.setattr(
        importer_module.shutil,
        "disk_usage",
        lambda _path: current._replace(free=0),
    )

    with pytest.raises(ActorPackImportError) as raised:
        import_service.revalidate_review(review)

    assert raised.value.category == "actor_pack_import_disk_unavailable"


def test_inspect_reports_disk_unavailable_when_staging_is_unverified(
    import_service: ActorPackImportService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = import_service._staging_root
    privacy = PrivatePathResult(
        staging_root,
        PrivatePathStatus.UNVERIFIED_PLATFORM,
        reason="native_acl_not_verified",
    )
    monkeypatch.setattr(
        importer_module,
        "secure_private_directory",
        lambda *_args, **_kwargs: privacy,
    )

    with pytest.raises(ActorPackImportError) as raised:
        import_service.inspect_archive(
            (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
        )

    assert raised.value.category == "actor_pack_import_disk_unavailable"
    assert list(staging_root.iterdir()) == []


def test_startup_sweep_removes_only_authenticated_bounded_candidates(
    import_service: ActorPackImportService,
) -> None:
    review = import_service.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    authenticated = import_service._staging_root / review._candidate_name
    malformed = import_service._staging_root / f".import-{'0' * 32}"
    malformed.mkdir(mode=0o700)
    (malformed / ".actor-pack-import").write_text("not-authority", encoding="ascii")

    replacement = ActorPackImportService(
        import_service.repository,
        staging_root=import_service._staging_root,
        profile_root=import_service._profile_root,
    )

    assert not authenticated.exists()
    assert malformed.exists()
    assert replacement.sweep_staging(max_candidates=1) == 0


def test_startup_sweep_skips_usable_unverified_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "profile.db"), client_id="actor-pack-import-unverified"
    )
    repository = ActorPackRepository(db)
    staging_root = tmp_path / "staging"
    candidate = staging_root / f".import-{'0' * 32}"
    candidate.mkdir(parents=True)
    privacy = PrivatePathResult(
        staging_root,
        PrivatePathStatus.UNVERIFIED_PLATFORM,
        reason="native_acl_not_verified",
    )

    monkeypatch.setattr(
        importer_module,
        "secure_private_directory",
        lambda *_args, **_kwargs: privacy,
    )

    def unexpected_access(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unverified staging contents must not be examined")

    monkeypatch.setattr(importer_module.os, "scandir", unexpected_access)
    monkeypatch.setattr(importer_module, "_read_candidate_authority", unexpected_access)

    service = ActorPackImportService(
        repository,
        staging_root=staging_root,
        profile_root=tmp_path,
    )

    assert service.sweep_staging() == 0
    assert candidate.exists()
