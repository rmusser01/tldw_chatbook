"""TASK-16841: `StudyGuideWidget`'s `#guide-topic-select` is backwards.

Found by the repo-wide AST sweep for TASK-16841. `UI/Study_Window.py`
composed the topic selector as:

    yield Select(options=[("new", "New Topic")], id="guide-topic-select")

-- `(value, label)` order, backwards against Textual's `(label, value)`
contract. `#guide-topic-select`'s `.value` has no consumer anywhere in the
repo (grep confirms), so this never raised `InvalidSelectValueError` --
but Textual always renders a Select's option using element 0 as the
display text, so this was a real, user-visible defect independent of any
`.value` read: opening the Study Guide tab showed the dropdown's only
choice as the literal text "new", not "New Topic".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from tldw_chatbook.UI.Study_Window import StudyGuideWidget

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = str(BUNDLE)

    def compose(self) -> ComposeResult:
        yield StudyGuideWidget()


@pytest.mark.asyncio
async def test_topic_select_renders_the_human_label_not_the_token() -> None:
    """AC born-red: the dropdown must show 'New Topic', not 'new'."""
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()

        select = app.query_one("#guide-topic-select", Select)
        # Index 0 is the blank placeholder Select injects when allow_blank
        # (the default, and true here -- no explicit allow_blank=False).
        rendered_label, value = select._options[1]
        assert str(rendered_label) == "New Topic"
        assert value == "new"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
