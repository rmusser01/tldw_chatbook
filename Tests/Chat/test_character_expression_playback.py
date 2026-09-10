from io import BytesIO

import pytest
from PIL import Image

from tldw_chatbook.Chat.character_expression_playback import (
    expression_motion_enabled,
    preparation_bytes,
    prepare_expression,
)


def encoded_animation(fmt="GIF", *, loop=0, durations=(100, 200)):
    frames = [Image.new("RGBA", (12, 12), color) for color in ("red", "blue")]
    out = BytesIO()
    frames[0].save(
        out,
        format=fmt,
        save_all=True,
        append_images=frames[1:],
        duration=list(durations),
        loop=loop,
        lossless=True,
    )
    return out.getvalue()


@pytest.mark.parametrize(
    "appearance,react,manual,expected",
    [
        ({}, True, False, True),
        ({}, False, False, False),
        ({}, False, True, True),
        ({"character_expression_mode": "static"}, True, True, False),
        ({"character_expression_mode": "invalid"}, True, False, True),
        ({"animations_enabled": False}, True, True, False),
        ({"animations_enabled": "false"}, True, True, False),
        ({"reduce_motion": True}, True, True, False),
    ],
)
def test_motion_policy(appearance, react, manual, expected):
    assert (
        expression_motion_enabled(
            {"appearance": appearance}, react=react, manual=manual
        )
        is expected
    )


@pytest.mark.parametrize(
    "fmt,loop,plays", [("GIF", 0, 0), ("GIF", 1, 2), ("WEBP", 1, 1)]
)
def test_encoded_timeline_pixels_and_loops(fmt, loop, plays):
    data = encoded_animation(fmt, loop=loop)
    before = preparation_bytes()
    result = prepare_expression(data, (8, 8), animate=True)
    try:
        assert result.frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
        assert result.frames[1].getpixel((0, 0)) == (0, 0, 255, 255)
        assert result.durations_ms == (100, 200)
        assert result.plays == plays
        assert result.frame_at(99) == (0, False)
        assert result.frame_at(100) == (1, False)
        assert result.frame_at(300 * (plays or 50)) == (
            (1, True) if plays else (0, False)
        )
        assert preparation_bytes() > before
    finally:
        result.close()
    assert preparation_bytes() == before


def test_static_preserves_source_and_uses_encoded_frame_zero():
    data = encoded_animation()
    original = bytes(data)
    result = prepare_expression(data, (8, 8), animate=False)
    try:
        assert len(result.frames) == 1
        assert result.frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
        assert result.frame_at(99999) == (0, True)
        assert data == original
    finally:
        result.close()


def test_corrupt_bytes_release_budget():
    before = preparation_bytes()
    with pytest.raises(ValueError):
        prepare_expression(b"bad image", (8, 8), animate=True)
    assert preparation_bytes() == before


def test_invalid_timing_falls_back_to_static():
    result = prepare_expression(
        encoded_animation(durations=(0, 200)), (8, 8), animate=True
    )
    try:
        assert len(result.frames) == 1
        assert result.fallback_reason
    finally:
        result.close()


def test_budget_rejected_before_loading_pixels(monkeypatch):
    import tldw_chatbook.Chat.character_expression_playback as module

    data = encoded_animation()
    monkeypatch.setattr(module, "MAX_PREPARATION_BYTES", 32)
    monkeypatch.setattr(
        Image.Image, "load", lambda self: pytest.fail("must preflight before load")
    )
    with pytest.raises(ValueError, match="budget"):
        prepare_expression(data, (8, 8), animate=True)


def test_apng_default_portrait_is_not_an_animation_frame():
    output = BytesIO()
    Image.new("RGBA", (8, 8), "red").save(
        output,
        format="PNG",
        save_all=True,
        default_image=True,
        append_images=[
            Image.new("RGBA", (8, 8), "green"),
            Image.new("RGBA", (8, 8), "blue"),
        ],
        duration=[100, 200],
        loop=1,
    )
    dynamic = prepare_expression(output.getvalue(), (8, 8), animate=True)
    static = prepare_expression(output.getvalue(), (8, 8), animate=False)
    try:
        assert len(dynamic.frames) == 2
        assert dynamic.frames[0].getpixel((0, 0)) == (0, 128, 0, 255)
        assert static.frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
        assert dynamic.frame_at(300) == (1, True)
    finally:
        dynamic.close()
        static.close()


def test_apng_alpha_and_background_disposal():
    first = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    second = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
    third = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    third.putpixel((0, 0), (0, 255, 0, 255))
    output = BytesIO()
    first.save(
        output,
        format="PNG",
        save_all=True,
        append_images=[second, third],
        duration=[100, 100, 100],
        disposal=[0, 1, 0],
        blend=[0, 0, 1],
        loop=0,
    )
    result = prepare_expression(output.getvalue(), (8, 8), animate=True)
    try:
        assert result.frames[0].getpixel((7, 7)) == (255, 0, 0, 255)
        assert result.frames[1].getpixel((7, 7)) == (0, 0, 255, 255)
        assert result.frames[2].getpixel((7, 7)) == (0, 0, 0, 0)
        assert result.frames[2].getpixel((0, 0)) == (0, 255, 0, 255)
    finally:
        result.close()


def test_active_buffers_count_against_new_preparation(monkeypatch):
    import tldw_chatbook.Chat.character_expression_playback as module

    first = prepare_expression(encoded_animation(), (12, 12), animate=True)
    retained = preparation_bytes()
    monkeypatch.setattr(module, "MAX_PREPARATION_BYTES", retained + 12 * 12 * 4 * 3)
    try:
        with pytest.raises(ValueError, match="budget"):
            prepare_expression(encoded_animation(), (12, 12), animate=True)
        assert preparation_bytes() == retained
    finally:
        first.close()


def test_many_overdue_frames_coalesce_without_extending_duration():
    result = prepare_expression(encoded_animation("WEBP", loop=2), (8, 8), animate=True)
    try:
        assert result.frame_at(550) == (1, False)
        assert result.frame_at(600) == (1, True)
        assert result.frame_at(100000) == (1, True)
    finally:
        result.close()


def test_codec_memory_probe_in_disposable_process(tmp_path):
    import json
    import os
    import subprocess
    import sys

    probe = r"""
import gc, json, resource, sys
from io import BytesIO
from PIL import Image
from tldw_chatbook.Chat.character_expression_playback import prepare_expression, preparation_bytes
frames = [Image.new("RGBA", (1024, 1024), color) for color in ("red", "blue")]
out = BytesIO()
frames[0].save(out, format="WEBP", save_all=True, append_images=frames[1:], duration=[100, 200], loop=0, lossless=True)
data = out.getvalue()
for frame in frames: frame.close()
del frames
gc.collect()
scale = 1 if sys.platform == "darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
for _ in range(8):
    prepared = prepare_expression(data, (128, 128), animate=True)
    assert preparation_bytes() == 128 * 128 * 4 * 2
    prepared.close()
    assert preparation_bytes() == 0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
print(json.dumps({"peak_growth_bytes": max(0, peak-before), "peak_rss_bytes": peak, "iterations": 8}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=30,
        check=True,
    )
    measurement = json.loads(result.stdout.strip().splitlines()[-1])
    assert measurement["peak_growth_bytes"] < 96 * 1024 * 1024
    (tmp_path / "codec-memory.json").write_text(json.dumps(measurement))
    print("Codec memory probe:", measurement)
