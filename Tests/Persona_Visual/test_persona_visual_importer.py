"""Tests for review-first Persona Visual pack archive import."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import threading
import unicodedata
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import tldw_chatbook.Persona_Visual.importer as importer_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.authoring import (
    inspect_persona_visual_draft,
    persona_visual_draft_publication_snapshot,
)
from tldw_chatbook.Persona_Visual.importer import (
    PersonaVisualImportError,
    cleanup_persona_visual_import_review,
    import_persona_visual_pack,
    persona_visual_import_source_root,
)
from tldw_chatbook.Persona_Visual.repository import PersonaVisualIdentity
from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _png_bytes(color: tuple[int, int, int] = (5, 10, 15)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 5), color).save(output, format="PNG")
    return output.getvalue()


def _visual_manifest() -> dict[str, object]:
    return {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {
            state: {"animation_id": "idle-loop"}
            for state in ("idle", "listening", "thinking", "speaking", "error")
        },
        "animations": {
            "idle-loop": {
                "frames": [{"asset_id": "idle"}],
                "preview_asset_id": "idle",
                "frame_rate": 2,
            }
        },
        "fallbacks": {},
        "state_catalog": {},
        "authored_triggers": [],
    }


def _archive_payloads(
    *,
    asset_data: bytes | None = None,
    asset_path: str = "assets/persona_visuals/idle.png",
    asset_mime: str = "image/png",
    asset_sha256: str | None = None,
    visual_manifest: dict[str, object] | None = None,
) -> dict[str, bytes]:
    data = asset_data or _png_bytes()
    digest = asset_sha256 or hashlib.sha256(data).hexdigest()
    pack_payload = _canonical(
        {
            "pack": {
                "source_pack_id": "server-pack-1",
                "source_persona_id": "server-persona-1",
                "title": "Imported operator states",
                "renderer_type": "sprite_frames",
                "manifest_version": 1,
                "visual_manifest": visual_manifest or _visual_manifest(),
                "provenance": "imported",
            }
        }
    )
    assets_payload = _canonical(
        {
            "assets": [
                {
                    "source_asset_id": "idle",
                    "source_pack_id": "server-pack-1",
                    "source_persona_id": "server-persona-1",
                    "asset_role": "frame",
                    "mime_type": asset_mime,
                    "byte_size": len(data),
                    "checksum_sha256": digest,
                    "width": 4,
                    "height": 5,
                    "duration_ms": None,
                    "asset_bytes_status": "present",
                    "asset_path": asset_path,
                    "asset_sha256": digest,
                    "asset_size_bytes": len(data),
                }
            ]
        }
    )
    checksums = {
        "metadata/pack.json": hashlib.sha256(pack_payload).hexdigest(),
        "metadata/assets.json": hashlib.sha256(assets_payload).hexdigest(),
        asset_path: hashlib.sha256(data).hexdigest(),
    }
    outer_manifest = _canonical(
        {
            "schema_version": "tldw.persona_visual_pack.v1",
            "exported_by": {"app": "tldw_server"},
            "pack_title": "Imported operator states",
            "renderer_type": "sprite_frames",
            "counts": {"assets": 1, "assets_with_bytes": 1, "missing_assets": 0},
            "encryption": {"encrypted": False, "scheme": None},
            "sections": [
                {"path": path, "sha256": checksum}
                for path, checksum in sorted(checksums.items())
            ],
        }
    )
    checksums["manifest.json"] = hashlib.sha256(outer_manifest).hexdigest()
    return {
        "manifest.json": outer_manifest,
        "metadata/pack.json": pack_payload,
        "metadata/assets.json": assets_payload,
        "checksums/sha256.json": _canonical(checksums),
        asset_path: data,
        "README.md": b"# Imported title\n\nUntrusted review text.\n",
        "signatures/README.md": b"Reserved.\n",
    }


def _write_archive(
    path: Path,
    payloads: dict[str, bytes] | None = None,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> Path:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, data in (payloads or _archive_payloads()).items():
            archive.writestr(name, data)
    return path


def _replace_declared_payload(
    payloads: dict[str, bytes],
    name: str,
    data: bytes,
) -> None:
    payloads[name] = data
    checksums = json.loads(payloads["checksums/sha256.json"])
    checksums[name] = hashlib.sha256(data).hexdigest()
    outer = json.loads(payloads["manifest.json"])
    for section in outer["sections"]:
        if section["path"] == name:
            section["sha256"] = checksums[name]
    outer_data = _canonical(outer)
    payloads["manifest.json"] = outer_data
    checksums["manifest.json"] = hashlib.sha256(outer_data).hexdigest()
    payloads["checksums/sha256.json"] = _canonical(checksums)


def _identity() -> PersonaVisualIdentity:
    return PersonaVisualIdentity(
        persona_id="local-persona-1",
        persona_revision=7,
        binding_id=2,
        binding_version=3,
        pack_id=5,
        pack_revision=7,
        pack_version_id=11,
        version_number=2,
        manifest_sha256="a" * 64,
    )


def _import(archive: Path, staging_root: Path, **kwargs: Any):
    staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return import_persona_visual_pack(
        archive,
        staging_root=staging_root,
        persona_id="local-persona-1",
        persona_revision=7,
        expected_identity=_identity(),
        **kwargs,
    )


def test_valid_server_v1_archive_becomes_inactive_path_free_review(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "valid.tldw-persona-vpack")
    staging_root = tmp_path / "private-staging"

    review = _import(archive, staging_root)
    inventory = inspect_persona_visual_draft(review.draft)

    assert review.schema_version == "tldw.persona_visual_pack.v1"
    assert review.archive_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert review.pack_title == "Imported operator states"
    assert review.asset_count == 1
    assert review.state_count == 5
    assert inventory.activatable is True
    assert review.draft.expected_identity == _identity()
    assert "private-staging" not in repr(review)
    source_root = persona_visual_import_source_root(review, staging_root=staging_root)
    assert source_root.is_dir()
    assert (source_root / review.draft.assets[0].source_storage_key).is_file()
    assert not hasattr(review, "activate")


def test_import_cleanup_removes_only_the_exact_issued_staging_tree(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "valid.tldw-persona-vpack")
    staging_root = tmp_path / "private-staging"
    review = _import(archive, staging_root)
    source_root = persona_visual_import_source_root(review, staging_root=staging_root)
    unrelated = staging_root / "unrelated"
    unrelated.mkdir(mode=0o700)
    (unrelated / "keep.txt").write_text("keep")

    assert (
        cleanup_persona_visual_import_review(review, staging_root=staging_root) is True
    )

    assert not source_root.exists()
    assert (unrelated / "keep.txt").read_text() == "keep"


def test_review_draft_publishes_once_through_the_existing_sqlite_boundary(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "valid.tldw-persona-vpack")
    staging_root = tmp_path / "private-staging"
    staging_root.mkdir(mode=0o700)
    review = import_persona_visual_pack(
        archive,
        staging_root=staging_root,
        persona_id="first-local-persona",
        persona_revision=1,
        expected_identity=None,
    )
    source_root = persona_visual_import_source_root(
        review,
        staging_root=staging_root,
    )
    profile_root = tmp_path / "profile"
    profile_root.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "import-publication.db", "import-publication")
    repository = PersonaVisualRepository(db)
    try:
        result = publish_persona_visual(
            repository,
            persona_visual_draft_publication_snapshot(review.draft),
            source_root=source_root,
            profile_root=profile_root,
            authority_guard=lambda: True,
        )

        assert result.old_identity is None
        assert (
            repository.get_active_persona_pack("first-local-persona").identity
            == result.new_identity
        )
        assert (
            repository.get_active_persona_pack("first-local-persona").pack.source_kind
            == "imported"
        )
        assert (
            cleanup_persona_visual_import_review(
                review,
                staging_root=staging_root,
            )
            is True
        )
    finally:
        db.close_connection()


def test_cleanup_refuses_same_path_replacement_even_with_copied_marker(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "valid.tldw-persona-vpack")
    staging_root = tmp_path / "private-staging"
    review = _import(archive, staging_root)
    issued = persona_visual_import_source_root(review, staging_root=staging_root)
    backup = staging_root / "issued-backup"
    issued.rename(backup)
    issued.mkdir(mode=0o700)
    marker = next(path for path in backup.iterdir() if path.name.startswith("."))
    shutil.copyfile(marker, issued / marker.name)
    (issued / "unrelated.txt").write_text("must survive")

    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_cleanup_denied$",
    ):
        cleanup_persona_visual_import_review(review, staging_root=staging_root)

    assert (issued / "unrelated.txt").read_text() == "must survive"
    assert backup.is_dir()


def test_cleanup_refuses_exact_same_path_copy_of_issued_candidate(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "valid.tldw-persona-vpack")
    staging_root = tmp_path / "private-staging"
    review = _import(archive, staging_root)
    issued = persona_visual_import_source_root(review, staging_root=staging_root)
    backup = staging_root / "issued-backup"
    issued.rename(backup)
    shutil.copytree(backup, issued)

    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_cleanup_denied$",
    ):
        cleanup_persona_visual_import_review(review, staging_root=staging_root)

    assert issued.is_dir()
    assert backup.is_dir()


def test_source_and_cleanup_refuse_a_marker_symlink(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "valid.tldw-persona-vpack")
    staging_root = tmp_path / "private-staging"
    review = _import(archive, staging_root)
    issued = persona_visual_import_source_root(review, staging_root=staging_root)
    marker = next(path for path in issued.iterdir() if path.name.startswith("."))
    backup = issued / "marker-backup"
    marker.rename(backup)
    marker.symlink_to(backup.name)

    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_cleanup_denied$",
    ):
        persona_visual_import_source_root(review, staging_root=staging_root)
    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_cleanup_denied$",
    ):
        cleanup_persona_visual_import_review(review, staging_root=staging_root)

    assert backup.is_file()


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../escape.png",
        "/absolute.png",
        "assets\\backslash.png",
        "assets/CON.png",
        "assets/nested.zip",
        "other/file.png",
    ),
)
def test_import_rejects_unsafe_or_unexpected_members_before_staging(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    payloads = _archive_payloads()
    payloads[unsafe_name] = b"unsafe"
    archive = _write_archive(tmp_path / "unsafe.tldw-persona-vpack", payloads)
    staging_root = tmp_path / "private-staging"

    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_invalid$",
    ):
        _import(archive, staging_root)

    assert not staging_root.exists() or list(staging_root.iterdir()) == []


def test_import_rejects_unicode_and_case_colliding_members(tmp_path: Path) -> None:
    payloads = _archive_payloads()
    payloads["assets/persona_visuals/IDLE.png"] = b"case"
    payloads[
        "assets/persona_visuals/" + unicodedata.normalize("NFD", "café") + ".png"
    ] = b"unicode-a"
    payloads["assets/persona_visuals/café.png"] = b"unicode-b"
    archive = _write_archive(tmp_path / "collision.tldw-persona-vpack", payloads)

    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_invalid$",
    ):
        _import(archive, tmp_path / "private-staging")


def test_import_rejects_duplicate_members_and_json_keys(tmp_path: Path) -> None:
    duplicate_archive = tmp_path / "duplicate.tldw-persona-vpack"
    payloads = _archive_payloads()
    with zipfile.ZipFile(duplicate_archive, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("metadata/pack.json", payloads["metadata/pack.json"])
    with pytest.raises(PersonaVisualImportError):
        _import(duplicate_archive, tmp_path / "duplicate-staging")

    payloads = _archive_payloads()
    pack = payloads["metadata/pack.json"]
    duplicate_pack = b'{"pack":' + pack[len(b'{"pack":') : -1] + b',"pack":{}}'
    _replace_declared_payload(payloads, "metadata/pack.json", duplicate_pack)
    duplicate_json = _write_archive(
        tmp_path / "duplicate-json.tldw-persona-vpack",
        payloads,
    )
    with pytest.raises(PersonaVisualImportError):
        _import(duplicate_json, tmp_path / "json-staging")


@pytest.mark.parametrize("encrypted", (False, True))
def test_import_rejects_link_and_encrypted_entries(
    tmp_path: Path,
    encrypted: bool,
) -> None:
    archive_path = tmp_path / "linked.tldw-persona-vpack"
    payloads = _archive_payloads()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        info = zipfile.ZipInfo("assets/persona_visuals/link.png")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        if encrypted:
            info.flag_bits |= 0x1
        archive.writestr(info, b"idle.png")

    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_invalid$",
    ):
        _import(archive_path, tmp_path / "private-staging")


@pytest.mark.parametrize(
    "mutation",
    ("checksum", "mime", "missing", "undeclared", "unsupported", "encrypted"),
)
def test_import_rejects_invalid_declarations_without_residue(
    tmp_path: Path,
    mutation: str,
) -> None:
    payloads = _archive_payloads(
        asset_sha256=("f" * 64 if mutation == "checksum" else None),
        asset_mime=("image/jpeg" if mutation == "mime" else "image/png"),
        visual_manifest=(
            {**_visual_manifest(), "manifest_version": 2}
            if mutation == "unsupported"
            else None
        ),
    )
    if mutation == "missing":
        payloads.pop("assets/persona_visuals/idle.png")
    elif mutation == "undeclared":
        checksums = json.loads(payloads["checksums/sha256.json"])
        checksums.pop("assets/persona_visuals/idle.png")
        payloads["checksums/sha256.json"] = _canonical(checksums)
    elif mutation == "encrypted":
        outer = json.loads(payloads["manifest.json"])
        outer["encryption"] = {"encrypted": True, "scheme": "secret"}
        payloads["manifest.json"] = _canonical(outer)
    archive = _write_archive(tmp_path / f"{mutation}.tldw-persona-vpack", payloads)
    staging_root = tmp_path / "private-staging"

    with pytest.raises(PersonaVisualImportError):
        _import(archive, staging_root)

    assert not staging_root.exists() or list(staging_root.iterdir()) == []


def test_import_cancellation_discards_owned_staging(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "cancel.tldw-persona-vpack")
    staging_root = tmp_path / "private-staging"
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_cancelled$",
    ):
        _import(archive, staging_root, cancel_event=cancelled)

    assert not staging_root.exists() or list(staging_root.iterdir()) == []


def test_import_cancellation_is_checked_while_streaming_checksums(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "cancel-stream.tldw-persona-vpack")
    staging_root = tmp_path / "private-staging"

    class CancelDuringChecksum:
        calls = 0

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls >= 4

    cancelled = CancelDuringChecksum()
    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_cancelled$",
    ):
        _import(archive, staging_root, cancel_event=cancelled)

    assert cancelled.calls == 4
    assert list(staging_root.iterdir()) == []


def test_import_rejects_source_symlink_and_insufficient_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "valid.tldw-persona-vpack")
    linked = tmp_path / "linked.tldw-persona-vpack"
    linked.symlink_to(archive.name)
    with pytest.raises(PersonaVisualImportError):
        _import(linked, tmp_path / "linked-staging")

    disk_usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        importer_module.shutil,
        "disk_usage",
        lambda _path: type(disk_usage)(disk_usage.total, disk_usage.used, 0),
    )
    staging_root = tmp_path / "space-staging"
    with pytest.raises(PersonaVisualImportError):
        _import(archive, staging_root)
    assert list(staging_root.iterdir()) == []


def test_archive_replacement_before_final_revalidation_discards_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "replace.tldw-persona-vpack")
    replacement = _write_archive(
        tmp_path / "replacement.tldw-persona-vpack",
        _archive_payloads(asset_data=_png_bytes((90, 80, 70))),
    )
    staging_root = tmp_path / "private-staging"
    original = importer_module._source_identity_current
    replaced = False

    def replace_before_check(*args: object, **kwargs: object) -> bool:
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, archive)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        importer_module,
        "_source_identity_current",
        replace_before_check,
    )

    with pytest.raises(
        PersonaVisualImportError,
        match="^persona_visual_import_stale$",
    ):
        _import(archive, staging_root)

    assert list(staging_root.iterdir()) == []


def test_import_errors_and_repr_do_not_expose_private_paths(tmp_path: Path) -> None:
    private_marker = "private-user-secret-archive"
    archive = tmp_path / f"{private_marker}.tldw-persona-vpack"
    archive.write_bytes(b"not a zip")

    with pytest.raises(PersonaVisualImportError) as caught:
        _import(archive, tmp_path / "private-staging")

    assert private_marker not in str(caught.value)
    assert private_marker not in repr(caught.value)
