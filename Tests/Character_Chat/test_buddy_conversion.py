"""Faithful bounded native Buddy timelines and independent publication."""

import hashlib
import io
import json

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat.buddy_conversion import (
    convert_buddy,
    suggest_buddy_mappings,
)
from tldw_chatbook.Persona_Visual.assets import PersonaVisualAssetMetadata
from tldw_chatbook.Persona_Visual.snapshot import BuddyAssetSnapshot, BuddySnapshot


def snapshot(*, same=False, loop=True, regions=False, current=lambda: True):
    assets = []
    for key, color in (("red", "red"), ("blue", "red" if same else "blue")):
        out = io.BytesIO()
        with Image.new("RGBA", (4, 4), color) as image:
            image.save(out, format="PNG")
        data = out.getvalue()
        assets.append(
            BuddyAssetSnapshot(
                PersonaVisualAssetMetadata(
                    key,
                    "frame",
                    "image/png",
                    len(data),
                    hashlib.sha256(data).hexdigest(),
                    4,
                    4,
                    1,
                    None,
                ),
                data,
            )
        )
    frames = [
        {"asset_id": "red", "duration_ms": 100},
        {"asset_id": "red", "duration_ms": 200},
        {"asset_id": "blue", "duration_ms": 300},
    ]
    if regions:
        frames[0]["region"] = {"x": 0, "y": 0, "width": 2, "height": 2}
    manifest = {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": frames,
                "loop": loop,
                "alignment": {"x": 1, "y": 1},
                "preview_frame": 2,
            }
        },
        "fallbacks": {
            key: ["idle"] for key in ("thinking", "speaking", "listening", "error")
        },
        "state_catalog": {},
        "authored_triggers": [],
    }
    return BuddySnapshot(
        title="Source Buddy",
        manifest_json=json.dumps(manifest),
        assets=tuple(assets),
        artwork={
            "version": 1,
            "creator": "Artist",
            "license": "MIT",
            "source_url": None,
            "notices": "Original notice",
        },
        source_sha256="a" * 64,
        _guard=current,
    )


def only_idle(source, **kwargs):
    mappings = {
        row.source_state: ("neutral" if row.source_state == "idle" else None)
        for row in suggest_buddy_mappings(source)
    }
    return convert_buddy(source, mappings, **kwargs)


@pytest.mark.parametrize("loop", [False, True])
def test_coalesced_animation_preserves_pixels_duration_loop_and_independent_portrait(
    loop,
):
    result = only_idle(snapshot(loop=loop))
    expression = result.expressions[0]
    assert expression.metadata["frame_count"] == 2
    assert expression.metadata["duration_ms"] == 600
    with Image.open(io.BytesIO(expression.data)) as image:
        assert image.info["loop"] == (0 if loop else 1)
        image.seek(0)
        image.load()
        assert image.getpixel((0, 0))[:3] == (255, 0, 0)
        assert image.info["duration"] == 300
        image.seek(1)
        image.load()
        assert image.getpixel((0, 0))[:3] == (0, 0, 255)
    with Image.open(io.BytesIO(result.portrait)) as portrait:
        assert portrait.getpixel((0, 0))[:3] == (0, 0, 255)


def test_identical_sequence_is_published_as_png():
    expression = only_idle(snapshot(same=True)).expressions[0]
    assert expression.metadata["content_type"] == "image/png"
    assert expression.metadata["frame_count"] == 1


def test_static_uses_first_composited_frame_and_alignment():
    expression = only_idle(snapshot(regions=True), animate=False).expressions[0]
    with Image.open(io.BytesIO(expression.data)) as image:
        assert image.size == (4, 4)
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((3, 3)) == (255, 0, 0, 255)


def test_missing_states_are_labelled_as_fallback_and_collisions_need_review():
    source = snapshot()
    rows = suggest_buddy_mappings(source)
    assert next(row for row in rows if row.source_state == "speaking").fallback
    with pytest.raises(ValueError, match="collision"):
        convert_buddy(source, {row.source_state: "neutral" for row in rows})
    with pytest.raises(ValueError, match="stale"):
        only_idle(snapshot(current=lambda: False))


def test_memory_budget_refuses_before_decoding(monkeypatch):
    import tldw_chatbook.Character_Chat.buddy_conversion as module

    monkeypatch.setattr(module, "MAX_CONVERSION_RGBA_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        only_idle(snapshot())


def test_unused_animation_keeps_preview_asset_and_cannot_replace_authored_state():
    from dataclasses import replace

    source = snapshot()
    manifest = json.loads(source.manifest_json)
    manifest["states"]["animation:extra"] = {"animation_id": "idle"}
    manifest["state_catalog"]["animation:extra"] = {
        "label": "Extra",
        "kind": "reaction",
    }
    manifest["animations"]["extra"] = dict(manifest["animations"]["idle"])
    manifest["animations"]["extra"].pop("preview_frame")
    manifest["animations"]["extra"]["preview_asset_id"] = "blue"
    source = replace(source, manifest_json=json.dumps(manifest))
    rows = suggest_buddy_mappings(source)
    assert {"animation:extra", "animation:animation:extra"} <= {
        r.source_state for r in rows
    }
    result = only_idle(source, portrait_state="animation:animation:extra")
    with Image.open(io.BytesIO(result.portrait)) as image:
        assert image.getpixel((0, 0))[:3] == (0, 0, 255)


def test_codec_cannot_silently_drop_distinct_frames():
    from tldw_chatbook.Character_Chat.buddy_conversion import _verified_timeline

    with (
        Image.new("RGBA", (4, 4), "red") as red,
        Image.new("RGBA", (4, 4), "blue") as blue,
    ):
        data = io.BytesIO()
        red.save(data, format="WEBP", lossless=True)
        with pytest.raises(ValueError, match="unfaithful"):
            _verified_timeline(data.getvalue(), [red, blue], [100, 100], True)


def test_unavailable_animation_codec_reports_static_fallback(monkeypatch):
    from PIL import features

    monkeypatch.setattr(features, "check", lambda feature: False)
    result = only_idle(snapshot())
    assert not result.expressions[0].metadata["is_animated"]
    assert "unavailable" in result.warnings[0]


def test_encoded_output_enforces_limit_during_write(monkeypatch):
    from tldw_chatbook.Character_Chat import visual_identity
    from tldw_chatbook.Character_Chat.buddy_conversion import _BoundedOutput

    monkeypatch.setattr(visual_identity, "MAX_EXPRESSION_ASSET_BYTES", 4)
    output = _BoundedOutput()
    output.write(b"1234")
    with pytest.raises(ValueError, match="budget"):
        output.write(b"5")
    assert output.getvalue() == b"1234"


def test_publication_is_independent_and_roundtrips_lineage(tmp_path):
    from Tests.Actor_Packs.test_actor_pack_attribution import services
    from tldw_chatbook.Actor_Packs.export import write_actor_pack_archive
    from tldw_chatbook.Character_Chat.buddy_conversion import publish_buddy_character
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

    current = [True]
    conversion = only_idle(snapshot(current=lambda: current[0]))
    root = tmp_path / "profile"
    db, importer, _, exporter = services(root)
    try:
        result = publish_buddy_character(
            conversion,
            name="Independent Buddy",
            db=db,
            local_service=importer._local_service,
            profile_root=root,
            authority_guard=lambda: True,
        )
        assert (
            db.get_character_card_by_id(int(result.local_actor_id))["name"]
            == "Independent Buddy"
        )
        current[0] = False
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", result.local_actor_id
        )
        context = json.loads(graph["assets"][0]["source_context_json"])
        assert context["tldw/artwork"]["creator"] == "Artist"
        assert context["tldw/buddy_conversion"]["source_state"] == "idle"
        output = tmp_path / "independent.tldw-actor-pack"
        with output.open("w+b") as stream:
            write_actor_pack_archive(
                exporter.capture_snapshot(
                    "character", result.local_actor_id, source="local"
                ),
                stream,
            )
        assert not list((root / "buddy-conversion").iterdir())
    finally:
        db.close_connection()
    db, importer, activation, _ = services(tmp_path / "second")
    try:
        result = activation.activate(importer.inspect_archive(output), "create_new")
        graph = VisualIdentityRepository(db).get_active_actor_pack(
            "character", result.local_actor_id
        )
        assert json.loads(graph["assets"][0]["source_context_json"]) == context
    finally:
        db.close_connection()


@pytest.mark.parametrize("failure", ["source_changed", "binding_failure"])
def test_publication_rolls_back_character_identity_and_files_after_writes(
    tmp_path, monkeypatch, failure
):
    from Tests.Actor_Packs.test_actor_pack_attribution import services
    from tldw_chatbook.Actor_Packs.activation import (
        ActorPackActivationError,
        ActorPackActivationService,
    )
    from tldw_chatbook.Character_Chat.buddy_conversion import publish_buddy_character

    root = tmp_path / "profile"
    db, importer, _, _ = services(root)
    current = [True]
    conversion = only_idle(snapshot(current=lambda: current[0]))
    original = ActorPackActivationService._activate_shared_visual

    def fail_after_write(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if failure == "source_changed":
            current[0] = False
        else:
            raise ValueError("injected binding failure")

    monkeypatch.setattr(
        ActorPackActivationService, "_activate_shared_visual", fail_after_write
    )
    before = {p for p in root.rglob("*") if p.is_file()}
    try:
        with pytest.raises(ActorPackActivationError):
            publish_buddy_character(
                conversion,
                name="Must not exist",
                db=db,
                local_service=importer._local_service,
                profile_root=root,
                authority_guard=lambda: True,
            )
        assert db.get_character_card_by_name("Must not exist") is None
        assert not db.execute_query("SELECT * FROM visual_identity_bindings").fetchall()
        assert not list((root / "buddy-conversion").iterdir())
        assert {p for p in root.rglob("*") if p.is_file()} == before
    finally:
        db.close_connection()


def test_cleanup_failure_after_commit_reports_created_character(tmp_path, monkeypatch):
    from Tests.Actor_Packs.test_actor_pack_attribution import services
    from tldw_chatbook.Character_Chat import buddy_conversion as module

    root = tmp_path / "profile"
    db, importer, _, _ = services(root)
    original = module.shutil.rmtree

    def fail_private_cleanup(path, *args, **kwargs):
        if str(path).split("/")[-1].startswith("convert-"):
            raise OSError("injected cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", fail_private_cleanup)
    try:
        result = module.publish_buddy_character(
            only_idle(snapshot()),
            name="Created once",
            db=db,
            local_service=importer._local_service,
            profile_root=root,
            authority_guard=lambda: True,
        )
        assert result.cleanup_pending
        assert (
            db.get_character_card_by_id(int(result.local_actor_id))["name"]
            == "Created once"
        )
    finally:
        db.close_connection()


def test_timeline_ignores_only_invisible_rgb_and_preserves_alpha():
    from tldw_chatbook.Character_Chat.buddy_conversion import _verified_timeline

    with (
        Image.new("RGBA", (4, 4), (255, 0, 0, 0)) as transparent,
        Image.new("RGBA", (4, 4), "blue") as blue,
    ):
        output = io.BytesIO()
        transparent.save(
            output,
            format="WEBP",
            save_all=True,
            append_images=[blue],
            duration=[100, 100],
            loop=0,
            lossless=True,
        )
        assert _verified_timeline(
            output.getvalue(), [transparent, blue], [100, 100], True
        )
        transparent.putpixel((0, 0), (255, 0, 0, 1))
        with pytest.raises(ValueError, match="unfaithful"):
            _verified_timeline(output.getvalue(), [transparent, blue], [100, 100], True)
