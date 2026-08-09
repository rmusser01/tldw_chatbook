"""ConsoleVideoCard spec/signature/details (task-3401.5)."""

from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Widgets.Console.console_video_card import (
    ConsoleVideoCardSpec,
    video_card_details_text,
    video_card_signature,
    video_card_status_line,
)


def _meta(**overrides):
    base = {
        "name": "dusk-over-neon-tokyo",
        "prompt": "dusk over neon tokyo",
        "backend": "minimax",
        "model": "MiniMax-H3",
        "seed": 42,
        "duration_seconds": 6.0,
        "fps": 24.0,
        "width": 1920,
        "height": 1080,
        "ratio": "16:9",
    }
    base.update(overrides)
    return VideoGenerationMetadata(**base)


def _spec(status="ready", **meta_overrides):
    return ConsoleVideoCardSpec(
        message_id="m1",
        meta=_meta(**meta_overrides),
        status=status,
        file_path="/tmp/x.mp4" if status == "ready" else None,
    )


def test_signature_flips_on_status_change():
    assert video_card_signature(_spec("ready")) != video_card_signature(_spec("expired"))


def test_signature_stable_across_path_moves_same_status():
    # A path move with the same status does not alter the render, so it must
    # not alter the signature (keeps refresh coalescing intact).
    a = _spec("ready")
    b = ConsoleVideoCardSpec(
        message_id="m1", meta=a.meta, status="ready", file_path="/elsewhere/y.mp4"
    )
    assert video_card_signature(a) == video_card_signature(b)


def test_details_text_covers_facts():
    text = video_card_details_text(_spec())
    for fragment in (
        "Name: dusk-over-neon-tokyo",
        "Source: minimax",
        "Seed: 42",
        "Duration: 6s",
        "Resolution: 1920x1080",
        "FPS: 24",
        "Model: MiniMax-H3",
        "Prompt: dusk over neon tokyo",
    ):
        assert fragment in text


def test_details_omit_unknown_model_and_add_negative():
    text = video_card_details_text(_spec(model=None, negative_prompt="blurry"))
    assert "Model:" not in text
    assert "Negative: blurry" in text


def test_status_lines():
    assert "Ready" in video_card_status_line(_spec("ready"))
    assert "Expired" in video_card_status_line(_spec("expired"))


def test_random_seed_and_unknown_duration_format():
    text = video_card_details_text(_spec(seed=-1, duration_seconds=None))
    assert "Seed: random" in text
    assert "Duration: unknown" in text
