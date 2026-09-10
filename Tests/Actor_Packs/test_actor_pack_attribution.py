"""Independent attribution archive exercised through real publication services."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.activation import ActorPackActivationService
from tldw_chatbook.Actor_Packs.contracts import validate_actor_pack_document
from tldw_chatbook.Actor_Packs.export import (
    ActorPackExportError,
    ActorPackExportService,
    ActorPackExportSnapshot,
    write_actor_pack_archive,
)
from tldw_chatbook.Actor_Packs.importer import (
    ActorPackImportError,
    ActorPackImportService,
)
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Character_Chat.visual_identity import (
    compute_pack_content_sha256,
    create_visual_identity_candidate,
    publish_visual_identity_candidate,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

from .conftest import (
    PNG_1X1,
    PORTABLE_UUID,
    canonical_json,
    file_descriptor,
    with_content_digest,
)
from .test_actor_pack_import_roundtrip import _write_independent_archive

MEMBER = "shared-visual-identity/attribution.json"
FEATURE = "visual-artwork-attribution/v1"
NOTICE = "Source copyright and redistribution notice.\n" * 200
RECORD = {
    "version": 1,
    "creator": "Original artist",
    "license": "MIT",
    "source_url": "https://example.com/art",
    "notices": NOTICE,
}
ASSET_RECORD = {**RECORD, "output_sha256": hashlib.sha256(PNG_1X1).hexdigest()}
CARRIER = {"version": 1, "pack": RECORD, "assets": {"neutral": ASSET_RECORD}}


def archive_with_attribution(path, *, carrier=CARRIER, feature=FEATURE, tamper=False):
    _write_independent_archive(path, "character", ("shared",))
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files.pop("actor-pack.json"))
    visual = json.loads(files["shared-visual-identity/manifest.json"])
    second_path = "shared-visual-identity/assets/asset-0002.png"
    visual["assets"].append(
        {
            **visual["assets"][0],
            "expression_key": "thinking",
            "original_label": "thinking",
            "display_label": "Thinking",
            "storage_relpath": second_path,
        }
    )
    visual["pack_content_sha256"] = compute_pack_content_sha256(visual)
    files["shared-visual-identity/manifest.json"] = canonical_json(visual)
    files[second_path] = PNG_1X1
    if carrier is not None:
        files[MEMBER] = canonical_json(carrier)
    if feature is not None:
        manifest["required_features"].append(feature)
    manifest["files"] = [
        file_descriptor(name, data) for name, data in sorted(files.items())
    ]
    manifest = with_content_digest(manifest)
    if tamper:
        files[MEMBER] = files[MEMBER].replace(b"Original artist", b"Different artist")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("actor-pack.json", canonical_json(manifest))
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    return manifest, files


def services(root):
    root.mkdir(parents=True, exist_ok=True)
    db = CharactersRAGDB(root / "actors.db", "attribution-test")
    repo = ActorPackRepository(db)
    local = LocalCharacterPersonaService(db, persona_store_path=root / "personas.json")
    importer = ActorPackImportService(
        repo, staging_root=root / "staging", profile_root=root, local_service=local
    )
    activation = ActorPackActivationService(
        db, local, repo, PersonaActorPackCoordinator(repo, local), importer
    )
    export = ActorPackExportService(
        db,
        local,
        repo,
        visual_identity_repository=VisualIdentityRepository(db),
        profile_root=root,
    )
    return db, importer, activation, export


def test_attribution_survives_review_activation_reopen_edit_and_export(tmp_path):
    path = tmp_path / "incoming.tldw-actor-pack"
    manifest, files = archive_with_attribution(path)
    validate_actor_pack_document(manifest, files)
    db, importer, activation, export = services(tmp_path / "profile")
    try:
        review = importer.inspect_archive(path)
        assert json.loads(review.artwork_attribution) == CARRIER
        from tldw_chatbook.Widgets.Persona_Widgets.actor_pack_import_review import (
            _metadata_copy,
        )

        assert NOTICE in _metadata_copy(review)
        result = activation.activate(review, "create_new")
        actor_id = result.local_actor_id
    finally:
        db.close_connection()

    db, importer, activation, export = services(tmp_path / "profile")
    try:
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", actor_id
        )
        assert (
            json.loads(graph["assets"][0]["source_context_json"])["tldw/artwork"]
            == ASSET_RECORD
        )
        second_review = importer.inspect_archive(path)
        assert second_review.uuid_match == "same_kind"
        assert json.loads(second_review.artwork_attribution) == CARRIER
        importer.cleanup_review(second_review)
        candidate = create_visual_identity_candidate(
            db, actor_kind="character", actor_id=actor_id
        )
        candidate.stage_clear("thinking")
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=tmp_path / "profile"
        )
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", actor_id
        )
        assert json.loads(graph["version"]["manifest_json"])["license"] == "MIT"
        assert (
            json.loads(graph["pack"]["source_context_json"])["tldw/artwork"] == RECORD
        )
        output = io.BytesIO()
        write_actor_pack_archive(
            export.capture_snapshot("character", str(actor_id), source="local"), output
        )
        with zipfile.ZipFile(output) as archive:
            assert json.loads(archive.read(MEMBER)) == CARRIER
            assert (
                FEATURE
                in json.loads(archive.read("actor-pack.json"))["required_features"]
            )
            assert str(tmp_path).encode() not in archive.read(MEMBER)
        path2 = tmp_path / "outgoing.tldw-actor-pack"
        path2.write_bytes(output.getvalue())
    finally:
        db.close_connection()
    db2, importer2, activation2, _ = services(tmp_path / "second-profile")
    try:
        created = activation2.activate(importer2.inspect_archive(path2), "create_new")
        graph = VisualIdentityRepository(db2).get_active_actor_pack(
            "character", created.local_actor_id
        )
        assert (
            json.loads(graph["assets"][0]["source_context_json"])["tldw/artwork"]
            == ASSET_RECORD
        )
    finally:
        db2.close_connection()


@pytest.mark.parametrize(
    "changes",
    [
        {"feature": None},
        {"carrier": None},
        {"feature": "visual-artwork-attribution/v2"},
        {"tamper": True},
        {"carrier": {**CARRIER, "version": 2}},
        {
            "carrier": {
                **CARRIER,
                "assets": {"neutral": {**ASSET_RECORD, "output_sha256": "0" * 64}},
            }
        },
    ],
)
def test_invalid_attribution_never_activates(tmp_path, changes):
    path = tmp_path / "invalid.tldw-actor-pack"
    archive_with_attribution(path, **changes)
    db, importer, _, _ = services(tmp_path / "profile")
    try:
        with pytest.raises(ActorPackImportError):
            importer.inspect_archive(path)
        assert (
            db.execute_query(
                "SELECT count(*) FROM actor_portable_identities"
            ).fetchone()[0]
            == 0
        )
    finally:
        db.close_connection()


def test_notice_edit_invalidates_existing_review_without_mutating_actor(tmp_path):
    path = tmp_path / "incoming.tldw-actor-pack"
    archive_with_attribution(path)
    db, importer, activation, _ = services(tmp_path / "profile")
    try:
        actor = activation.activate(importer.inspect_archive(path), "create_new")
        review = importer.inspect_archive(path)
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", actor.local_actor_id
        )
        context = json.loads(graph["assets"][0]["source_context_json"])
        context["tldw/artwork"]["notices"] += "\nNew notice"
        with db.transaction():
            db.execute_query(
                "UPDATE visual_identity_assets SET source_context_json=? WHERE id=?",
                (json.dumps(context), graph["assets"][0]["id"]),
            )
        with pytest.raises(ActorPackImportError, match="review_stale"):
            importer.revalidate_review(review)
        importer.cleanup_review(review)
        assert (
            db.execute_query(
                "SELECT count(*) FROM actor_portable_identities"
            ).fetchone()[0]
            == 1
        )
    finally:
        db.close_connection()


@pytest.mark.parametrize("actor_kind", ["character", "persona"])
def test_legacy_archive_bytes_match_golden_at_its_declared_producer_version(
    actor_kind, monkeypatch
):
    """The shipped fixture pins an older product version, not today's version."""
    import tldw_chatbook.Actor_Packs.export as export_module

    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "export-golden"
        / f"minimal-{actor_kind}.tldw-actor-pack"
    )
    with zipfile.ZipFile(fixture) as archive:
        root = json.loads(archive.read("actor-pack.json"))
        actor_payload = archive.read("actor/actor.json")
    monkeypatch.setattr(export_module, "__version__", root["producer"]["version"])
    snapshot = ActorPackExportSnapshot(
        actor_kind=actor_kind,
        actor_revision=1,
        portable_uuid=PORTABLE_UUID,
        identity_version=1,
        portrait_name="portrait.png",
        local_actor_id="private-id",
        portrait_sha256=hashlib.sha256(PNG_1X1).hexdigest(),
        actor_payload=actor_payload,
        portrait_bytes=PNG_1X1,
    )
    output = io.BytesIO()
    write_actor_pack_archive(snapshot, output)
    assert output.getvalue() == fixture.read_bytes()


def test_export_rejects_notice_change_between_snapshot_reads(tmp_path):
    path = tmp_path / "incoming.tldw-actor-pack"
    archive_with_attribution(path)
    db, importer, activation, export = services(tmp_path / "profile")
    try:
        actor = activation.activate(importer.inspect_archive(path), "create_new")
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", actor.local_actor_id
        )

        def change_notice(phase):
            if phase != "visuals_loaded":
                return
            context = json.loads(graph["pack"]["source_context_json"])
            context["tldw/artwork"]["notices"] += "\nChanged"
            with db.transaction():
                db.execute_query(
                    "UPDATE visual_identity_packs SET source_context_json=? WHERE id=?",
                    (json.dumps(context), graph["pack"]["id"]),
                )

        with pytest.raises(ActorPackExportError, match="authority_changed"):
            export.capture_snapshot(
                "character",
                str(actor.local_actor_id),
                source="local",
                phase_hook=change_notice,
            )
    finally:
        db.close_connection()
