"""Real lineage import, reopen, editing and export preserve exact public bytes."""

import io
import json
import zipfile

import pytest

from tldw_chatbook.Actor_Packs.contracts import (
    ActorPackValidationError,
    validate_actor_pack_document,
)
from tldw_chatbook.Actor_Packs.export import write_actor_pack_archive
from tldw_chatbook.Actor_Packs.importer import ActorPackImportError
from tldw_chatbook.Character_Chat.visual_identity import (
    create_visual_identity_candidate,
    publish_visual_identity_candidate,
)
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

from .conftest import PNG_1X1, canonical_json, with_content_digest
from .test_actor_pack_attribution import (
    ASSET_RECORD,
    CARRIER,
    FEATURE,
    MEMBER,
    archive_with_attribution,
    services,
)

NAMESPACE = "tldw/buddy_conversion"
CONVERSION_FEATURE = "visual-buddy-conversion/v1"
LINEAGE = {
    "version": 1,
    "source_sha256": "b" * 64,
    "source_state": "idle",
    "source_asset_sha256": ["c" * 64],
    "converted_at": "2026-09-07T12:30:00+00:00",
    "fallback": False,
    "output_sha256": ASSET_RECORD["output_sha256"],
}
V2 = {**CARRIER, "version": 2, "conversions": {"neutral": LINEAGE}}


def archive_v2(path, *, carrier=V2, features=(FEATURE, CONVERSION_FEATURE)):
    manifest, files = archive_with_attribution(path, carrier=carrier, feature=None)
    manifest["required_features"].extend(features)
    manifest = with_content_digest(manifest)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("actor-pack.json", canonical_json(manifest))
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    return manifest, files


@pytest.mark.parametrize(
    "carrier,features",
    [
        (V2, (FEATURE,)),
        (V2, (CONVERSION_FEATURE,)),
        (CARRIER, (FEATURE, CONVERSION_FEATURE)),
        (V2, (FEATURE, "visual-buddy-conversion/v2")),
        (
            {**V2, "conversions": {"neutral": {**LINEAGE, "output_sha256": "e" * 64}}},
            (FEATURE, CONVERSION_FEATURE),
        ),
    ],
)
def test_lineage_carrier_feature_pairing_and_stale_hashes_rejected(
    tmp_path, carrier, features
):
    path = tmp_path / "bad.tldw-actor-pack"
    manifest, files = archive_v2(path, carrier=carrier, features=features)
    with pytest.raises(ActorPackValidationError):
        validate_actor_pack_document(manifest, files)
    db, importer, _, _ = services(tmp_path / "profile")
    try:
        with pytest.raises(ActorPackImportError):
            importer.inspect_archive(path)
    finally:
        db.close_connection()


def test_lineage_roundtrip_reopen_edit_and_replacement(tmp_path):
    path = tmp_path / "incoming.tldw-actor-pack"
    manifest, files = archive_v2(path)
    validate_actor_pack_document(manifest, files)
    db, importer, activation, _ = services(tmp_path / "profile")
    try:
        actor_id = activation.activate(
            importer.inspect_archive(path), "create_new"
        ).local_actor_id
    finally:
        db.close_connection()
    db, importer, _, export = services(tmp_path / "profile")
    try:
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", actor_id
        )
        assert (
            json.loads(graph["assets"][0]["source_context_json"])[NAMESPACE] == LINEAGE
        )
        candidate = create_visual_identity_candidate(
            db, actor_kind="character", actor_id=actor_id
        )
        candidate.stage_clear("thinking")
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=tmp_path / "profile"
        )
        output = io.BytesIO()
        write_actor_pack_archive(
            export.capture_snapshot("character", str(actor_id), source="local"), output
        )
        with zipfile.ZipFile(output) as archive:
            assert archive.read(MEMBER) == canonical_json(V2)
            assert (
                CONVERSION_FEATURE
                in json.loads(archive.read("actor-pack.json"))["required_features"]
            )
            visual = json.loads(archive.read("shared-visual-identity/manifest.json"))
            assert archive.read(visual["assets"][0]["storage_relpath"]) == PNG_1X1
        outgoing = tmp_path / "outgoing.tldw-actor-pack"
        outgoing.write_bytes(output.getvalue())
        candidate = create_visual_identity_candidate(
            db, actor_kind="character", actor_id=actor_id
        )
        candidate.stage_replacement("neutral", PNG_1X1)
        publish_visual_identity_candidate(
            db, candidate, user_data_dir=tmp_path / "profile"
        )
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", actor_id
        )
        context = json.loads(graph["assets"][0]["source_context_json"])
        assert NAMESPACE not in context and "tldw/artwork" not in context
        replaced = io.BytesIO()
        write_actor_pack_archive(
            export.capture_snapshot("character", str(actor_id), source="local"),
            replaced,
        )
        with zipfile.ZipFile(replaced) as archive:
            assert json.loads(archive.read(MEMBER))["version"] == 1
            assert (
                CONVERSION_FEATURE
                not in json.loads(archive.read("actor-pack.json"))["required_features"]
            )
    finally:
        db.close_connection()
    db, importer, activation, _ = services(tmp_path / "second-profile")
    try:
        actor_id = activation.activate(
            importer.inspect_archive(outgoing), "create_new"
        ).local_actor_id
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", actor_id
        )
        assert (
            json.loads(graph["assets"][0]["source_context_json"])[NAMESPACE] == LINEAGE
        )
    finally:
        db.close_connection()
