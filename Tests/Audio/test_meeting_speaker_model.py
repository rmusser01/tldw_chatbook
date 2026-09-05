"""Task 2: segment speaker id, per-meeting name map, versioned meeting files."""
from __future__ import annotations

import pytest

from tldw_chatbook.Audio.meeting_session import (
    MeetingSegment, render_label, write_meeting_json, read_meeting_json,
)

pytestmark = pytest.mark.unit


def _seg(**kw):
    base = dict(seq=0, t_audio_start=0.0, t_audio_end=1.0, t_wall_start=0.0,
                t_wall_end=1.0, label="others", text="hi")
    base.update(kw); return MeetingSegment(**base)

def test_render_uses_the_name_map_by_cluster_id():
    seg = _seg(label="others", speaker_id="S2")
    assert render_label(seg, {"S2": "Alice"}, "Me") == "Alice"

def test_render_falls_back_to_generic_speaker_when_unnamed():
    seg = _seg(label="others", speaker_id="S2")
    assert render_label(seg, {}, "Me") == "Speaker 2"

def test_you_channel_renders_the_user_display_name():
    seg = _seg(label="you", speaker_id=None)
    assert render_label(seg, {}, "Me") == "Me"

def test_old_meeting_json_backfills_speaker_fields(tmp_path):
    write_meeting_json(tmp_path, {"mode": "call", "format_version": 1})
    payload = read_meeting_json(tmp_path)
    assert payload["format_version"] == 1
    assert payload.get("speaker_names", {}) == {}
