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

def test_render_never_shows_a_raw_final_cluster_id():
    """A legacy recording whose jsonl still holds an unmatched final-cluster
    id ('F0') must never render as 'Speaker F0' (final whole-branch review I2;
    new recordings mint an 'S' id in the worker)."""
    seg = _seg(label="others", speaker_id="F0")
    assert render_label(seg, {}, "Me") == "Speaker 0"

def test_you_channel_renders_the_user_display_name():
    seg = _seg(label="you", speaker_id=None)
    assert render_label(seg, {}, "Me") == "Me"

def test_others_without_cluster_renders_others():
    seg = _seg(label="others", speaker_id=None)
    assert render_label(seg, {}, "Me") == "Others"

def test_both_channel_without_cluster_renders_the_display_name_plus_others():
    """task 31746 review (spec gap): an overlap segment names the mic channel
    too -- bare "Others" would silently disagree with the partial preview and
    `render_markdown`, which already say "<name> + Others"."""
    seg = _seg(label="both", speaker_id=None)
    assert render_label(seg, {}, "You") == "You + Others"
    assert render_label(seg, {}, "Alice") == "Alice + Others"

def test_room_mode_segment_without_speaker_renders_none():
    seg = _seg(label=None, speaker_id=None)
    assert render_label(seg, {}, "Me") is None

def test_old_meeting_json_backfills_speaker_fields(tmp_path):
    write_meeting_json(tmp_path, {"mode": "call", "format_version": 1})
    payload = read_meeting_json(tmp_path)
    assert payload["format_version"] == 1
    assert payload.get("speaker_names", {}) == {}

def test_old_meeting_json_backfills_diarize_mic_channel_false(tmp_path):
    """task 31743: old recordings predate hybrid-room mic diarization and
    must read back as flag-off."""
    write_meeting_json(tmp_path, {"mode": "call", "format_version": 1})
    payload = read_meeting_json(tmp_path)
    assert payload["diarize_mic_channel"] is False

def test_you_channel_with_diarized_speaker_renders_the_speaker_when_flag_on():
    """task 31743: `diarize_mic=True` yields the "you" pre-naming to a
    diarized mic segment's own speaker id."""
    seg = _seg(label="you", speaker_id="S2")
    assert render_label(seg, {"S2": "Bob"}, "You", diarize_mic=True) == "Bob"

def test_you_channel_with_diarized_speaker_keeps_display_name_when_flag_off():
    """Default `diarize_mic=False` keeps every existing caller's behaviour:
    a "you" segment always renders the display name, speaker_id or not."""
    seg = _seg(label="you", speaker_id="S2")
    assert render_label(seg, {"S2": "Bob"}, "You", diarize_mic=False) == "You"
    assert render_label(seg, {"S2": "Bob"}, "You") == "You"

def test_you_channel_without_a_speaker_id_still_renders_display_name_when_flag_on():
    """Best-effort: a failed/absent assignment falls back to the mic name
    even with the flag on."""
    seg = _seg(label="you", speaker_id=None)
    assert render_label(seg, {}, "You", diarize_mic=True) == "You"
