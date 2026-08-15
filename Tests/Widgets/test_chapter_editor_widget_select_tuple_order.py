"""task-15772 review follow-up: `#chapter-voice-select` composed backwards.

Same bug class as the audiobook Selects task-15772 fixed in
`STTS_Window.py`: Textual's `Select` interprets each options tuple as
`(label, value)`, but `ChapterEditorWidget.compose` built
`#chapter-voice-select` as `options=[("narrator", "Use Narrator Voice"),
("custom", "Custom Voice")]` -- `(id, label)`, the reverse.

Grepped the whole repo for `chapter-voice-select`/`chapter_voice_select`:
zero consumers anywhere (no `.query_one`, no `Select.Changed` handler). This
Select is genuinely dead UI today, so the backwards order never broke
anything live -- but it's exactly the bug the task swept for, so it's fixed
here for uniformity and to guard the moment a consumer is added.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield ChapterEditorWidget()


def _options_as_pairs(select: Select) -> list[tuple[str, object]]:
    return [
        (str(label), value)
        for label, value in select._options
        if value is not Select.NULL
    ]


@pytest.mark.asyncio
async def test_chapter_voice_select_composes_label_value_order():
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        select = app.query_one("#chapter-voice-select", Select)
        assert _options_as_pairs(select) == [
            ("Use Narrator Voice", "narrator"),
            ("Custom Voice", "custom"),
        ]

        # The real ids must be legal Select values -- the exact assertion
        # that fails with `InvalidSelectValueError` against the pre-fix
        # (id, label) order.
        select.value = "custom"
        assert select.value == "custom"
        select.value = "narrator"
        assert select.value == "narrator"
