"""Pure motion and shared-budget performance preparation."""

from io import BytesIO

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat.portrait_performance import (
    FRAME_ROLES,
    PerformanceSource,
    PortraitPerformanceRecord,
    normalize_performance_frames,
)
from tldw_chatbook.Chat.character_expression_playback import (
    preparation_bytes,
    prepare_performance,
)
from tldw_chatbook.Chat.portrait_performance_state import (
    PerformanceMotionValues,
    PortraitPerformanceClock,
    load_performance_motion_values,
)


def performance_record(*, revision=1):
    images = {}
    for role, color in zip(
        FRAME_ROLES, ["red", "green", "blue", "yellow", "cyan", "magenta"], strict=True
    ):
        output = BytesIO()
        with Image.new("RGBA", (12, 12), color) as image:
            image.save(output, format="PNG")
        images[role] = output.getvalue()
    return PortraitPerformanceRecord(
        PerformanceSource(7, "neutral", "a" * 64, "portrait", ()),
        revision,
        True,
        normalize_performance_frames(images),
    )


def test_blink_uses_current_mouth_without_restarting_clock():
    clock = PortraitPerformanceClock(
        motion=PerformanceMotionValues(3, 7, 0.2, 6, 1),
        interval=lambda: 3,
    )
    assert clock.sample("rest", now=0).role == "rest"
    assert clock.sample("small", now=3).role == "blink_small"
    assert clock.sample("open", now=3.05).role == "blink_open"
    assert clock.sample("open", now=3.21).role == "open"
    assert clock.sample("rest", now=6.21).role == "blink_rest"


def test_idle_stays_within_fixed_canvas_offset():
    clock = PortraitPerformanceClock(
        motion=PerformanceMotionValues(3, 7, 0.2, 6, 1),
        interval=lambda: 7,
    )
    offsets = [clock.sample("rest", now=t / 10).offset_px for t in range(61)]
    assert max(offsets) <= 1 and min(offsets) >= -1
    assert max(offsets) > 0.9 and min(offsets) < -0.9


def test_runtime_motion_matches_packaged_design_tokens():
    values = load_performance_motion_values()
    assert values == PerformanceMotionValues(3, 7, 0.2, 6, 1)


def test_missing_motion_token_fails_closed(monkeypatch, tmp_path):
    import tldw_chatbook.Chat.portrait_performance_state as module

    (tmp_path / "css/core").mkdir(parents=True)
    (tmp_path / "css/core/_variables.tcss").write_text("$ds-duration-fast: 0.2s;")
    module.load_performance_motion_values.cache_clear()
    monkeypatch.setattr(module, "files", lambda _package: tmp_path)
    try:
        with pytest.raises(ValueError, match="motion"):
            module.load_performance_motion_values()
    finally:
        module.load_performance_motion_values.cache_clear()


def test_preparation_fits_six_roles_and_releases_shared_budget():
    record = performance_record()
    before = preparation_bytes()
    prepared = prepare_performance(record, (10, 8))
    assert len(prepared.frames) == 6
    assert {f.size for f in prepared.frames} == {(8, 8)}
    assert preparation_bytes() == before + 6 * 8 * 8 * 4
    pose = prepared.render_pose(2, 0.5)
    assert pose.frames[0].size == (8, 8)
    pose.close()
    prepared.close()
    prepared.close()
    assert preparation_bytes() == before


def test_partial_preparation_failure_releases_previous_frames(monkeypatch):
    import tldw_chatbook.Chat.character_expression_playback as module

    record = performance_record()
    before = preparation_bytes()
    original = module.prepare_expression
    calls = []

    def fail_third(*args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise ValueError("bad frame")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_expression", fail_third)
    with pytest.raises(ValueError, match="bad frame"):
        prepare_performance(record, (8, 8))
    assert preparation_bytes() == before


def test_small_art_is_prepared_at_fitted_resolution_for_one_pixel_idle_motion():
    record = performance_record()
    before = preparation_bytes()
    prepared = prepare_performance(record, (96, 64))
    try:
        assert {frame.size for frame in prepared.frames} == {(64, 64)}
        pose = prepared.render_pose(0, 1)
        try:
            assert pose.frames[0].getpixel((32, 1))[3] == 255
            assert pose.frames[0].getpixel((32, 0))[3] == 0
        finally:
            pose.close()
    finally:
        prepared.close()
    assert preparation_bytes() == before


def test_motion_import_loads_no_image_ui_or_audio_backend_in_fresh_interpreter():
    import subprocess
    import sys

    script = """
import sys
import tldw_chatbook.Chat.portrait_performance_state
assert not any(n == 'sounddevice' or n == 'PIL.Image' or n.startswith('textual') or n.startswith('tldw_chatbook.TTS') for n in sys.modules)
print('PURE_PORTRAIT_IMPORT_OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0 and "PURE_PORTRAIT_IMPORT_OK" in result.stdout, (
        result.stderr
    )
