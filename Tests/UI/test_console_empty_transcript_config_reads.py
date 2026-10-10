"""An empty transcript must not read the unused file-card preference."""

import pytest

from tldw_chatbook.Widgets.Console import console_transcript as source

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("compose", "message_widgets", "reconcile"))
async def test_empty_transcript_does_not_read_file_card_setting(monkeypatch, route):
    transcript = source.ConsoleTranscript()
    rows = transcript._transcript_rows()
    assert [row.kind for row in rows] == ["empty"]

    def unexpected_read(section, key, default=None):
        raise AssertionError(f"unused empty-transcript setting read: {section}.{key}")

    monkeypatch.setattr(source, "get_cli_setting", unexpected_read)
    if route == "compose":
        widgets = list(transcript.compose())
        assert widgets and transcript._row_widgets["empty"] is widgets[0]
    elif route == "message_widgets":
        assert len(transcript._message_widgets()) == 1
    else:
        await transcript._reconcile_rows(rows)
