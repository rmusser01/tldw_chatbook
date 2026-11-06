"""Source-inspected evidence for the amended Console Wave 6 closeout.

TASK-39055 retired the frozen-revision evidence (owner decision, 2026-11-06).
The realtime method sets stay: the realtime boundary test reads them.
"""

from __future__ import annotations

import ast

import pytest


FINAL_CLOSEOUT_COUNTS = (17_037, 564)
REALTIME_METHODS = frozenset(
    {
        "_append_console_realtime_row",
        "_begin_console_realtime_reply_audio",
        "_build_console_realtime_callbacks",
        "_build_console_realtime_session",
        "_build_console_realtime_sink",
        "_close_console_realtime_resources",
        "_close_console_realtime_session",
        "_connect_console_realtime",
        "_console_realtime_adopt_transcript",
        "_console_realtime_api_key",
        "_console_realtime_begin_reconnect",
        "_console_realtime_connect_failed",
        "_console_realtime_exit_loop",
        "_console_realtime_exit_message",
        "_console_realtime_failure_token",
        "_console_realtime_fallback_to_pipeline",
        "_console_realtime_instructions",
        "_console_realtime_marshal",
        "_console_realtime_mode_changed",
        "_console_realtime_playback_finished",
        "_console_realtime_played_ms",
        "_console_realtime_row_metadata",
        "_console_realtime_seed_items",
        "_console_realtime_seed_text",
        "_console_realtime_silence_speech",
        "_end_console_realtime_reply_audio",
        "_enter_console_realtime_loop",
        "_finish_console_realtime_reply_row",
        "_handle_console_realtime_intent",
        "_mark_console_realtime_transcript_empty",
        "_note_console_realtime_audio_unavailable",
        "_on_console_realtime_audio_delta",
        "_on_console_realtime_closed",
        "_on_console_realtime_error",
        "_on_console_realtime_first_audio",
        "_on_console_realtime_frames",
        "_on_console_realtime_input_transcript",
        "_on_console_realtime_output_transcript_delta",
        "_on_console_realtime_ready",
        "_on_console_realtime_reply_done",
        "_on_console_realtime_reply_started",
        "_on_console_realtime_sink_event",
        "_on_console_realtime_speech_started",
        "_on_console_realtime_transcription_usage",
        "_on_console_realtime_turn_committed",
        "_on_console_realtime_usage",
        "_persist_console_realtime_event",
        "_pump_console_realtime_audio",
        "_release_console_realtime_state",
        "_repaint_console_realtime_chip",
        "_sanitize_console_realtime_failure",
        "_send_console_realtime_text_turn",
        "_set_console_realtime_transcript_status",
        "_start_console_realtime_connect",
        "_start_console_realtime_tap",
        "_teardown_console_realtime_loop",
        "_tick_console_realtime",
    }
)

REALTIME_DELEGATE_METHODS: frozenset[str] = frozenset()
REALTIME_STAY_METHODS = frozenset(
    {
        "_repaint_console_realtime_chip",
    }
)
REALTIME_MOVE_METHODS = (
    REALTIME_METHODS - REALTIME_DELEGATE_METHODS - REALTIME_STAY_METHODS
)


def _source_references_method(source: str, method_name: str) -> bool:
    """Return whether executable syntax references ``method_name`` externally."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == method_name:
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == method_name
        ):
            return True
    return False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("callback = screen._target", True),
        ('callback = getattr(screen, "_target")', True),
        ('available = hasattr(screen, "_target")', True),
        ('"""Documentation mentioning _target."""', False),
        ("screen._other()", False),
    ],
)
def test_source_method_reference_detection(source: str, expected: bool) -> None:
    """Detect executable external references without matching documentation."""
    assert _source_references_method(source, "_target") is expected
