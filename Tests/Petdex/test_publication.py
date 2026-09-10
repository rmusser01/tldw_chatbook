"""Petdex output crosses real native and independent-character boundaries."""

import json

from Tests.Petdex.test_conversion import atlas_source
from tldw_chatbook.Character_Chat.buddy_conversion import convert_buddy
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.authoring import (
    persona_visual_draft_publication_snapshot,
)
from tldw_chatbook.Persona_Visual.export import export_persona_visual_archive
from tldw_chatbook.Persona_Visual.importer import (
    import_persona_visual_pack,
    persona_visual_import_source_root,
)
from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive, read_saved_buddy
from tldw_chatbook.Petdex.conversion import build_petdex_archive


def test_petdex_saved_offline_export_and_character_pixels(tmp_path):
    from io import BytesIO

    from PIL import Image

    source = atlas_source()
    archive = tmp_path / "petdex.zip"
    archive.write_bytes(build_petdex_archive(source))
    staging, profile = tmp_path / "staging", tmp_path / "profile"
    staging.mkdir(mode=0o700)
    profile.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "native.db", client_id="petdex-publication")
    repository = PersonaVisualRepository(db)
    try:
        review = import_persona_visual_pack(
            archive,
            staging_root=staging,
            persona_id="pet",
            persona_revision=1,
            expected_identity=None,
        )
        publish_persona_visual(
            repository,
            persona_visual_draft_publication_snapshot(review.draft),
            source_root=persona_visual_import_source_root(review, staging_root=staging),
            profile_root=profile,
            authority_guard=lambda: True,
        )
        archive.unlink()
        saved = read_saved_buddy(repository, "pet", profile)
        assert dict(saved.artwork) == dict(source.artwork)
        converted = convert_buddy(saved)
        thinking = next(
            e for e in converted.expressions if e.source_state == "thinking"
        )
        # Row 8, six frames: actual decoded color and duration, not metadata alone.
        with Image.open(BytesIO(thinking.data)) as image:
            assert image.n_frames == 6
            duration = 0
            for index in range(image.n_frames):
                image.seek(index)
                image.load()
                assert image.convert("RGBA").getpixel((6, 6)) == (
                    160,
                    index * 30,
                    80,
                    255,
                )
                duration += image.info["duration"]
            assert duration == 1030
        output = tmp_path / "export.tldw-persona-vpack"
        output.write_bytes(export_persona_visual_archive(repository, "pet", profile))
        reopened = read_buddy_archive(output)
        assert reopened.assets == saved.assets
        assert reopened.artwork == saved.artwork
        imported = import_persona_visual_pack(
            output,
            staging_root=staging,
            persona_id="copy",
            persona_revision=1,
            expected_identity=None,
        )
        assert json.loads(dict(imported.draft.source_context)["artwork"]) == dict(
            source.artwork
        )
    finally:
        db.close_connection()
