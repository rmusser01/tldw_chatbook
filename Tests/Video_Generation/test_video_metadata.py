"""VideoGenerationMetadata payload round-trips and degradation (task-3401.4)."""

import pytest

from tldw_chatbook.Video_Generation.video_metadata import (
    VIDEO_METADATA_TOP_KEY,
    VideoGenerationMetadata,
)


def _meta(**overrides):
    base = {
        "name": "dusk-over-neon-tokyo",
        "prompt": "dusk over neon tokyo, cinematic",
        "backend": "minimax",
        "container": "mp4",
        "negative_prompt": "blurry",
        "model": "MiniMax-H3",
        "seed": 42,
        "duration_seconds": 6.0,
        "fps": 24.0,
        "width": 1920,
        "height": 1080,
        "ratio": "16:9",
        "source_image_message_id": None,
    }
    base.update(overrides)
    return VideoGenerationMetadata(**base)


@pytest.mark.parametrize("container", ["mp4", "webm"])
def test_round_trip_preserves_every_field(container):
    meta = _meta(container=container, source_image_message_id="msg-123")
    rebuilt = VideoGenerationMetadata.from_json(meta.to_json())
    assert rebuilt == meta


@pytest.mark.parametrize("container", ["mp4", "webm"])
def test_construction_accepts_supported_container(container):
    assert _meta(container=container).container == container


def test_payload_is_namespaced_under_top_key():
    import json

    payload = json.loads(_meta().to_json())
    assert set(payload) == {VIDEO_METADATA_TOP_KEY}
    # No path and no URL anywhere in the persisted facts (ADR-044).
    for forbidden in ("path", "url", "file_path", "download_url"):
        assert forbidden not in payload[VIDEO_METADATA_TOP_KEY]


def test_from_json_degrades_missing_and_foreign_payloads():
    assert VideoGenerationMetadata.from_json(None) is None
    assert VideoGenerationMetadata.from_json("") is None
    assert VideoGenerationMetadata.from_json("not-json") is None
    assert VideoGenerationMetadata.from_json("[]") is None
    assert VideoGenerationMetadata.from_json("{}") is None
    # A turn-provenance payload (MessageMetadata shape) is NOT video metadata.
    provenance = '{"engine": "realtime", "interrupted": true, "model": "", "provider": "", "transcript_status": ""}'
    assert VideoGenerationMetadata.from_json(provenance) is None


def test_from_json_degrades_unconstructable_payload():
    # Valid JSON, right top key, but empty name/backend would fail the
    # constructor's call-site rule -- degrade rather than raise on resume.
    bad = '{"video_generation": {"name": "", "backend": "minimax"}}'
    assert VideoGenerationMetadata.from_json(bad) is None


def test_construction_refuses_empty_name_and_backend():
    with pytest.raises(ValueError, match="name"):
        _meta(name="")
    with pytest.raises(ValueError, match="backend"):
        _meta(backend="  ")


@pytest.mark.parametrize("container", ["", "mov", "MP4", ".webm", None])
def test_construction_refuses_invalid_container(container):
    with pytest.raises(ValueError, match="container"):
        _meta(container=container)


def test_historical_payload_without_container_defaults_to_mp4():
    raw = '{"video_generation": {"name": "clip", "prompt": "p", "backend": "comfyui"}}'

    rebuilt = VideoGenerationMetadata.from_json(raw)

    assert rebuilt is not None
    assert rebuilt.container == "mp4"


@pytest.mark.parametrize("container", ["", "mov", "MP4", None, 7])
def test_present_invalid_container_never_receives_historical_fallback(container):
    import json

    raw = json.dumps(
        {
            VIDEO_METADATA_TOP_KEY: {
                "name": "clip",
                "prompt": "p",
                "backend": "comfyui",
                "container": container,
            }
        }
    )

    assert VideoGenerationMetadata.from_json(raw) is None


def test_numeric_coercion_on_resume():
    import json

    raw = json.dumps(
        {
            VIDEO_METADATA_TOP_KEY: {
                "name": "clip",
                "prompt": "p",
                "backend": "comfyui",
                "seed": "42",          # hand-edited payload coerces
                "width": 1280.0,       # whole float coerces to int
                "fps": True,           # bools never coerce
                "height": "not-a-number",
            }
        }
    )
    rebuilt = VideoGenerationMetadata.from_json(raw)
    assert rebuilt is not None
    assert rebuilt.seed == 42
    assert rebuilt.width == 1280
    assert rebuilt.fps is None
    assert rebuilt.height is None


def test_cohabitation_with_message_metadata():
    """The two metadata_json shapes read the same column and never confuse
    each other: provenance readers see "nothing recorded" on a video row,
    and video readers see None on a provenance row."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    video_payload = _meta().to_json()
    provenance = MessageMetadata.from_json(video_payload)
    assert provenance is not None and provenance.is_empty

    provenance_payload = MessageMetadata(interrupted=True).to_json()
    assert VideoGenerationMetadata.from_json(provenance_payload) is None
