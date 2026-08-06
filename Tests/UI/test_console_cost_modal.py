"""Console cost-breakdown modal (task-5, PR3): pure row/totals formatting.

task-2390 extends ``_format_row`` to surface realtime audio-token and
transcription-duration costs -- ``ConsoleCostRow`` already folds them into
``cost_usd`` (a single dollar figure), and this task's AC requires that the
modal not silently hide them inside that undecomposable total.

Only the ``@staticmethod`` formatters are exercised here (no Textual app
needed): they are pure functions over already-computed dataclasses.
"""

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRow
from tldw_chatbook.Widgets.Console.console_cost_modal import ConsoleCostModal


def test_format_row_shows_audio_and_transcription_when_present():
    row = ConsoleCostRow(
        index=0,
        role="assistant",
        model="gpt-realtime",
        uncached_input=15,
        cache_read=0,
        cache_write=0,
        output=28,
        cost_usd=0.006844,
        estimated=False,
        audio_input=18,
        audio_output=90,
        transcription_seconds=2.5,
    )
    text = ConsoleCostModal._format_row(row)
    assert "audio_in:18" in text
    assert "audio_out:90" in text
    assert "transcribe:2.5s" in text


def test_format_row_omits_audio_fields_for_a_non_realtime_row():
    row = ConsoleCostRow(
        index=0,
        role="user",
        model="claude-sonnet-4-6",
        uncached_input=100,
        cache_read=0,
        cache_write=0,
        output=0,
        cost_usd=0.10,
        estimated=False,
    )
    text = ConsoleCostModal._format_row(row)
    assert "audio_in" not in text
    assert "audio_out" not in text
    assert "transcribe" not in text
