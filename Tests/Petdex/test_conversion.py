"""Petdex atlas semantics verified with distinctive row and cell pixels."""

import io
import json
from dataclasses import asdict

import pytest
from PIL import Image, ImageDraw

from tldw_chatbook.Petdex.conversion import (
    PetdexState,
    build_petdex_archive,
    inspect_petdex,
)
from tldw_chatbook.Petdex.sources import source_from_bytes


def atlas_source(version=1, *, states=None, size=None):
    dims = size or (96, 13 * (9 if version == 1 else 11))
    output = io.BytesIO()
    with Image.new("RGBA", dims) as image:
        draw = ImageDraw.Draw(image)
        for row in range(9 if version == 1 else 11):
            for col in range(8):
                draw.rectangle(
                    (col * 12, row * 13, (col + 1) * 12 - 1, (row + 1) * 13 - 1),
                    fill=(row * 20, col * 30, 80, 255),
                )
        image.save(output, format="PNG")
    metadata = {
        "name": "Sentinel",
        "spriteVersionNumber": version,
        "creator": "Original Artist",
        "license": "MIT",
    }
    if states is not None:
        metadata["states"] = states
    return source_from_bytes(
        json.dumps(metadata).encode(), output.getvalue(), "spritesheet.png"
    )


def test_classic_mapping_preserves_regions_frame_count_and_exact_total(tmp_path):
    from tldw_chatbook.Persona_Visual.contracts import resolve_manifest_state
    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive
    from tldw_chatbook.Persona_Visual.validation import validate_persona_visual_manifest

    source = atlas_source()
    inspected = inspect_petdex(source)
    assert inspected.cell_width == 12 and inspected.cell_height == 13
    assert inspected.mapping_source == "pinned-classic"
    assert [(s.name, s.frames) for s in inspected.states][:2] == [
        ("idle", 6),
        ("running-right", 8),
    ]
    path = tmp_path / "out.tldw-persona-vpack"
    path.write_bytes(build_petdex_archive(source))
    snapshot = read_buddy_archive(path)
    manifest = validate_persona_visual_manifest(
        snapshot.manifest_json, {"sheet": (96, 117)}
    )
    selection = resolve_manifest_state(manifest, "thinking")
    assert selection.animation.frames[0].region.y == 8 * 13
    assert sum(f.duration_ms for f in selection.animation.frames) == 1030
    speaking = resolve_manifest_state(manifest, "speaking")
    assert speaking.resolved_state == "idle"
    assert len(manifest.state_catalog) == 9
    assert snapshot.artwork["creator"] == "Original Artist"
    assert snapshot.assets[0].data == source.image_bytes


def test_undeclared_v2_requires_explicit_manual_rows():
    source = atlas_source(2)
    inspected = inspect_petdex(source)
    assert inspected.states == ()
    assert inspected.mapping_source == "manual-required"
    with pytest.raises(ValueError, match="mapping_required"):
        build_petdex_archive(source)
    states = (PetdexState("idle", 10, 3, 500), PetdexState("review", 1, 2, 100))
    data = build_petdex_archive(source, states=states)
    assert data.startswith(b"PK")


def test_declared_v2_and_conflicting_geometry():
    source = atlas_source(2, states=[asdict(PetdexState("idle", 10, 3, 501))])
    assert inspect_petdex(source).mapping_source == "declared"
    assert inspect_petdex(source).states[0].row == 10
    with pytest.raises(ValueError):
        inspect_petdex(atlas_source(2, size=(96, 117)))


@pytest.mark.parametrize(
    "state",
    [
        PetdexState("idle", 11, 3, 500),
        PetdexState("idle", 0, 9, 500),
        PetdexState("idle", 0, 8, 30),
        PetdexState("idle", 0, 3, True),
    ],
)
def test_invalid_state_cannot_publish(state):
    with pytest.raises(ValueError):
        build_petdex_archive(atlas_source(2), states=(state,))


def test_duplicate_states_and_bad_native_mapping_reject():
    with pytest.raises(ValueError):
        build_petdex_archive(
            atlas_source(2), states=(PetdexState("idle", 0, 3, 500),) * 2
        )
    with pytest.raises(ValueError):
        build_petdex_archive(atlas_source(), mappings={"idle": "missing"})
