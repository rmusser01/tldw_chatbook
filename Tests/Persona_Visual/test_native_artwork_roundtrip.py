"""Native Buddy publication, editing and portability retain original artwork terms."""

import json
from dataclasses import replace

import pytest

from Tests.Persona_Visual.test_persona_visual_importer import (
    _archive_payloads,
    _canonical,
    _replace_declared_payload,
    _write_archive,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.authoring import (
    persona_visual_draft_from_graph,
    persona_visual_draft_publication_snapshot,
)
from tldw_chatbook.Persona_Visual.importer import (
    import_persona_visual_pack,
    persona_visual_import_source_root,
)
from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive, read_saved_buddy

ARTWORK = {
    "version": 1,
    "creator": "Original Artist",
    "license": "Original / Terms",
    "source_url": "https://example.org/pets/original",
    "notices": "Copyright original artist\nPreserve this notice.\n" * 200,
}


def _credited_archive(tmp_path):
    payloads = _archive_payloads()
    pack = json.loads(payloads["metadata/pack.json"])
    pack["pack"]["description"] = "Saved Buddy description"
    pack["pack"]["source_context"] = {
        "artwork": _canonical(ARTWORK).decode(),
        "mapping_source": "declared",
        "source_id": "a" * 64,
    }
    _replace_declared_payload(payloads, "metadata/pack.json", _canonical(pack))
    return _write_archive(tmp_path / "credited.zip", payloads)


def test_native_import_save_edit_offline_export_reimport_preserves_terms(tmp_path):
    path = _credited_archive(tmp_path)
    assert dict(read_buddy_archive(path).artwork) == ARTWORK
    staging, profile = tmp_path / "staging", tmp_path / "profile"
    staging.mkdir(mode=0o700)
    profile.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "credits.db", client_id="native-credits")
    repository = PersonaVisualRepository(db)
    try:
        review = import_persona_visual_pack(
            path,
            staging_root=staging,
            persona_id="buddy",
            persona_revision=1,
            expected_identity=None,
        )
        assert json.loads(dict(review.draft.source_context)["artwork"]) == ARTWORK
        publish_persona_visual(
            repository,
            persona_visual_draft_publication_snapshot(review.draft),
            source_root=persona_visual_import_source_root(review, staging_root=staging),
            profile_root=profile,
            authority_guard=lambda: True,
        )
        exported = repository.get_active_persona_pack_for_export("buddy")
        draft = persona_visual_draft_from_graph(
            exported.graph,
            source_context=dict(exported.source_context),
            source_storage_keys={
                x.record.asset_key: x.storage_key for x in exported.assets
            },
        )
        assert json.loads(dict(draft.source_context)["artwork"]) == ARTWORK
        publish_persona_visual(
            repository,
            persona_visual_draft_publication_snapshot(replace(draft, revision=1)),
            source_root=profile,
            profile_root=profile,
            authority_guard=lambda: True,
        )
        path.unlink()
        saved = read_saved_buddy(repository, "buddy", profile)
        assert saved.artwork == ARTWORK
        # Legacy edit callers omit context; publishing their next version must
        # retain pack-level terms already attached to the saved Buddy.
        exported = repository.get_active_persona_pack_for_export("buddy")
        legacy_draft = persona_visual_draft_from_graph(
            exported.graph,
            source_storage_keys={
                x.record.asset_key: x.storage_key for x in exported.assets
            },
        )
        publish_persona_visual(
            repository,
            persona_visual_draft_publication_snapshot(legacy_draft),
            source_root=profile,
            profile_root=profile,
            authority_guard=lambda: True,
        )
        saved = read_saved_buddy(repository, "buddy", profile)
        assert saved.artwork == ARTWORK
        from tldw_chatbook.Actor_Packs.export import (
            ActorPackExportError,
            ActorPackExportService,
        )
        from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
        from tldw_chatbook.Character_Chat.local_character_persona_service import (
            LocalCharacterPersonaService,
        )

        actor_exporter = ActorPackExportService(
            db,
            LocalCharacterPersonaService(
                db, persona_store_path=profile / "personas.json"
            ),
            ActorPackRepository(db),
            persona_visual_repository=repository,
            profile_root=profile,
        )
        with pytest.raises(ActorPackExportError) as refused:
            actor_exporter._capture_persona_visual("persona", "buddy")
        assert "Export Buddy" in refused.value.user_message
        from tldw_chatbook.Character_Chat.buddy_conversion import convert_buddy

        converted = convert_buddy(saved)
        from Tests.Actor_Packs.test_actor_pack_attribution import services
        from tldw_chatbook.Character_Chat.buddy_conversion import (
            publish_buddy_character,
        )
        from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

        character_root = tmp_path / "character-profile"
        character_db, character_importer, _, _ = services(character_root)
        try:
            result = publish_buddy_character(
                converted,
                name="Independent native Buddy",
                db=character_db,
                local_service=character_importer._local_service,
                profile_root=character_root,
                authority_guard=lambda: True,
            )
            character = VisualIdentityRepository(character_db).get_active_actor_pack(
                "character",
                result.local_actor_id,
            )
            assert (
                json.loads(character["pack"]["source_context_json"])["tldw/artwork"]
                == ARTWORK
            )
            for asset in character["assets"]:
                record = json.loads(asset["source_context_json"])["tldw/artwork"]
                assert {key: record[key] for key in ARTWORK} == ARTWORK
        finally:
            character_db.close_connection()
        from tldw_chatbook.Persona_Visual.export import export_persona_visual_archive

        output = tmp_path / "export.tldw-persona-vpack"
        output.write_bytes(export_persona_visual_archive(repository, "buddy", profile))
        reopened = read_buddy_archive(output)
        assert reopened.artwork == ARTWORK
        assert reopened.assets == saved.assets
        assert reopened.description == saved.description == "Saved Buddy description"
        next_review = import_persona_visual_pack(
            output,
            staging_root=staging,
            persona_id="second",
            persona_revision=1,
            expected_identity=None,
        )
        assert json.loads(dict(next_review.draft.source_context)["artwork"]) == ARTWORK
        assert next_review.draft.description == "Saved Buddy description"
        assert dict(next_review.draft.source_context)["mapping_source"] == "declared"
        assert dict(next_review.draft.source_context)["source_id"] == "a" * 64
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "context",
    [
        {"artwork": "{}"},
        {"artwork": json.dumps(ARTWORK)},
        {"source_id": "/private/path"},
        {"mapping_source": "../guess"},
    ],
)
def test_context_rejects_invalid_artwork_and_keeps_generic_path_restrictions(context):
    from tldw_chatbook.Persona_Visual.repository import _source_context_json

    with pytest.raises(ValueError, match="persona_visual_source_context_invalid"):
        _source_context_json(context)


@pytest.mark.parametrize("damage", ["stale", "bytes", "conflicting_credits"])
def test_native_writer_rejects_changed_bytes_or_authority(tmp_path, damage):
    from tldw_chatbook.Persona_Visual.export import build_native_buddy_archive
    from tldw_chatbook.Persona_Visual.importer import PersonaVisualImportError

    source = read_buddy_archive(_credited_archive(tmp_path))
    context = None
    if damage == "stale":
        source = replace(source, _guard=lambda: False)
    elif damage == "bytes":
        source = replace(source, assets=(replace(source.assets[0], data=b"changed"),))
    else:
        context = {
            "artwork": _canonical({**ARTWORK, "creator": "Another artist"}).decode()
        }
    with pytest.raises(PersonaVisualImportError):
        build_native_buddy_archive(source, source_context=context)


@pytest.mark.parametrize("carrier", ["legacy_fields", "namespace", "absent"])
def test_native_import_retains_existing_credit_shapes_and_unspecified_terms(
    tmp_path, carrier
):
    payloads = _archive_payloads()
    pack = json.loads(payloads["metadata/pack.json"])
    if carrier == "legacy_fields":
        pack["pack"].update(
            {key: value for key, value in ARTWORK.items() if key != "version"}
        )
    elif carrier == "namespace":
        pack["pack"]["source_context"] = {"tldw/artwork": ARTWORK}
    _replace_declared_payload(payloads, "metadata/pack.json", _canonical(pack))
    path = _write_archive(tmp_path / "old.zip", payloads)
    staging = tmp_path / "old-staging"
    staging.mkdir(mode=0o700)
    review = import_persona_visual_pack(
        path,
        staging_root=staging,
        persona_id="old",
        persona_revision=1,
        expected_identity=None,
    )
    if carrier == "absent":
        # Current native imports preserve an explicit record of unspecified terms.
        assert json.loads(dict(review.draft.source_context)["artwork"]) == {
            "version": 1,
            "creator": None,
            "license": None,
            "source_url": None,
            "notices": "",
        }
        assert read_buddy_archive(path).artwork["license"] is None
    else:
        assert json.loads(dict(review.draft.source_context)["artwork"]) == ARTWORK
        assert read_buddy_archive(path).artwork == ARTWORK


def test_unknown_credits_native_export_reimport_remains_actor_pack_exportable(tmp_path):
    from tldw_chatbook.Actor_Packs.export import ActorPackExportService
    from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
    from tldw_chatbook.Character_Chat.local_character_persona_service import (
        LocalCharacterPersonaService,
    )
    from tldw_chatbook.Persona_Visual.export import build_native_buddy_archive

    original = read_buddy_archive(_write_archive(tmp_path / "unknown.zip"))
    exported = tmp_path / "unknown.tldw-persona-vpack"
    exported.write_bytes(build_native_buddy_archive(original))
    staging, profile = tmp_path / "staging", tmp_path / "profile"
    staging.mkdir(mode=0o700)
    profile.mkdir(mode=0o700)
    review = import_persona_visual_pack(
        exported,
        staging_root=staging,
        persona_id="unknown",
        persona_revision=1,
        expected_identity=None,
    )
    assert json.loads(dict(review.draft.source_context)["artwork"]) == dict(
        original.artwork
    )
    db = CharactersRAGDB(tmp_path / "unknown.db", client_id="unknown-artwork")
    repository = PersonaVisualRepository(db)
    try:
        publish_persona_visual(
            repository,
            persona_visual_draft_publication_snapshot(review.draft),
            source_root=persona_visual_import_source_root(review, staging_root=staging),
            profile_root=profile,
            authority_guard=lambda: True,
        )
        exporter = ActorPackExportService(
            db,
            LocalCharacterPersonaService(
                db, persona_store_path=profile / "personas.json"
            ),
            ActorPackRepository(db),
            persona_visual_repository=repository,
            profile_root=profile,
        )
        section = exporter._capture_persona_visual("persona", "unknown")
        assert section.kind == "persona-runtime"
        assert section.license is None
        assert section.assets[0].data == original.assets[0].data
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "override", [None, {}, {"mapping_source": "manual", "source_id": "replacement"}]
)
def test_native_export_roundtrip_preserves_description_and_context_defaults(
    tmp_path, override
):
    from tldw_chatbook.Persona_Visual.export import build_native_buddy_archive

    original = read_buddy_archive(_credited_archive(tmp_path))
    assert original.description == "Saved Buddy description"
    output = tmp_path / "roundtrip.tldw-persona-vpack"
    output.write_bytes(build_native_buddy_archive(original, source_context=override))
    reopened = read_buddy_archive(output)
    assert reopened.description == original.description
    assert reopened.artwork == original.artwork
    expected = dict(original.source_context) if override is None else override
    context = dict(reopened.source_context)
    for key in ("mapping_source", "source_id"):
        assert context.get(key) == expected.get(key)
    staging = tmp_path / "roundtrip-staging"
    staging.mkdir(mode=0o700)
    review = import_persona_visual_pack(
        output,
        staging_root=staging,
        persona_id="roundtrip",
        persona_revision=1,
        expected_identity=None,
    )
    assert review.draft.description == "Saved Buddy description"
    assert dict(review.draft.source_context) == context


@pytest.mark.parametrize(
    "description", [None, 42, "x" * 4097], ids=["null", "number", "oversized"]
)
def test_native_import_rejects_invalid_description_in_snapshot_and_review(
    tmp_path, description
):
    from tldw_chatbook.Persona_Visual.importer import PersonaVisualImportError

    payloads = _archive_payloads()
    pack = json.loads(payloads["metadata/pack.json"])
    pack["pack"]["description"] = description
    _replace_declared_payload(payloads, "metadata/pack.json", _canonical(pack))
    path = _write_archive(tmp_path / "bad-description.zip", payloads)
    with pytest.raises(PersonaVisualImportError):
        read_buddy_archive(path)
    staging = tmp_path / "bad-description-staging"
    staging.mkdir(mode=0o700)
    with pytest.raises(PersonaVisualImportError):
        import_persona_visual_pack(
            path,
            staging_root=staging,
            persona_id="invalid",
            persona_revision=1,
            expected_identity=None,
        )


def test_native_import_missing_description_keeps_legacy_defaults(tmp_path):
    path = _write_archive(tmp_path / "legacy.zip", _archive_payloads())
    assert read_buddy_archive(path).description == ""
    staging = tmp_path / "legacy-staging"
    staging.mkdir(mode=0o700)
    review = import_persona_visual_pack(
        path,
        staging_root=staging,
        persona_id="legacy",
        persona_revision=1,
        expected_identity=None,
    )
    assert review.draft.description == "Imported Persona Visual pack"
