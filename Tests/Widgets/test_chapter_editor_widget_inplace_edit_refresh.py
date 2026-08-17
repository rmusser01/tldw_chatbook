"""task-16849: the chapter editor's own edit buttons never refreshed the
now-truthful table (task-15773 residual 4).

`_add_chapter`/`_split_chapter`/`_merge_chapters`/`_delete_chapter` all
mutate `self.chapters` **in place** (`list.insert`/`list.pop`) and then
`post_message(ChapterEditEvent(...))` -- but a plain reactive assignment
check (`chapters == chapters`, same list object) never fires
`watch_chapters`, so `_refresh_chapter_table` never runs. Only
`set_chapters` (the detection/population path, task-15773) reassigns the
reactive and gets a table refresh for free.

Pre-fix, the review measured this directly: after `_add_chapter`,
`len(chapters) == 6` but the table still had 5 rows; after
`_delete_chapter`, counts happened to match (5/5) but the wrong titles
were shown because the preview/selection had drifted too.

Each test below drives one of the four edit buttons via the SAME
`on_button_pressed` dispatch the UI uses, then asserts the DataTable row
count matches `len(editor.chapters)` and that the edited chapter's own
content is what the table/preview actually show -- not just that the
counts happen to coincide.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, DataTable, TextArea

from tldw_chatbook.TTS.audiobook_generator import Chapter
from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Container(id="slot")


def _chapters(n: int, *, words_per_chapter: int = 10) -> list[Chapter]:
    return [
        Chapter(
            number=i + 1,
            title=f"Chapter {i + 1}",
            content=" ".join(f"w{i}-{j}" for j in range(words_per_chapter)),
            start_position=0,
            end_position=0,
        )
        for i in range(n)
    ]


async def _mount_editor(pilot, app: App, chapters: list[Chapter]) -> ChapterEditorWidget:
    slot = app.query_one("#slot", Container)
    editor = ChapterEditorWidget(id="chapter-editor-widget")
    await slot.mount(editor)
    await pilot.pause()
    editor.set_chapters(chapters)
    await pilot.pause()
    return editor


def _press(editor: ChapterEditorWidget, button_id: str) -> None:
    """Drive an edit the same way a real click does: through
    `on_button_pressed`, using the actual mounted `Button` instance."""
    button = editor.query_one(f"#{button_id}", Button)
    editor.on_button_pressed(Button.Pressed(button))


@pytest.mark.asyncio
async def test_add_chapter_refreshes_table_and_shows_new_chapter():
    app = _Host()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        editor = await _mount_editor(pilot, app, _chapters(3))

        # Select the first chapter, then add -- new chapter lands at index 1.
        editor.selected_chapter_index = 0
        await pilot.pause()

        _press(editor, "add-chapter-btn")
        await pilot.pause()

        table = editor.query_one("#chapter-table", DataTable)
        assert table.row_count == len(editor.chapters) == 4, (
            f"table has {table.row_count} rows for {len(editor.chapters)} "
            "chapters after add -- the table never refreshed"
        )
        # The new chapter is what the editor should now be showing.
        assert editor.selected_chapter_index == 1
        new_chapter = editor.chapters[1]
        assert new_chapter.title == "New Chapter 2"
        preview = editor.query_one("#chapter-preview", TextArea)
        assert preview.text == new_chapter.content == ""
        assert table.get_row_at(1)[2] == "New Chapter 2"


@pytest.mark.asyncio
async def test_split_chapter_refreshes_table_and_shows_truncated_content():
    app = _Host()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        editor = await _mount_editor(pilot, app, _chapters(2, words_per_chapter=1))
        # Give chapter 0 multiple lines so a cursor at line 1 has something
        # real to split.
        editor.chapters[0].content = "line-a\nline-b\nline-c"
        editor.selected_chapter_index = 0
        await pilot.pause()

        preview = editor.query_one("#chapter-preview", TextArea)
        preview.text = editor.chapters[0].content
        preview.move_cursor((1, 0))

        _press(editor, "split-chapter-btn")
        await pilot.pause()

        table = editor.query_one("#chapter-table", DataTable)
        assert table.row_count == len(editor.chapters) == 3, (
            f"table has {table.row_count} rows for {len(editor.chapters)} "
            "chapters after split -- the table never refreshed"
        )
        # Selection stays on the original, now-truncated chapter.
        assert editor.selected_chapter_index == 0
        first_half = editor.chapters[0]
        assert first_half.content == "line-a"
        assert preview.text == first_half.content, (
            "preview still shows the pre-split content"
        )
        assert table.get_row_at(1)[2] == f"{first_half.title} (Part 2)"


@pytest.mark.asyncio
async def test_merge_chapters_refreshes_table_and_shows_merged_content():
    app = _Host()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        editor = await _mount_editor(pilot, app, _chapters(3))
        editor.selected_chapter_index = 0
        await pilot.pause()

        first_content = editor.chapters[0].content
        second_content = editor.chapters[1].content

        _press(editor, "merge-chapter-btn")
        await pilot.pause()

        table = editor.query_one("#chapter-table", DataTable)
        assert table.row_count == len(editor.chapters) == 2, (
            f"table has {table.row_count} rows for {len(editor.chapters)} "
            "chapters after merge -- the table never refreshed"
        )
        assert editor.selected_chapter_index == 0
        merged = editor.chapters[0]
        assert merged.content == f"{first_content}\n\n{second_content}"
        preview = editor.query_one("#chapter-preview", TextArea)
        assert preview.text == merged.content, (
            "preview still shows the pre-merge content"
        )


@pytest.mark.asyncio
async def test_delete_chapter_refreshes_table_and_selects_neighbor():
    app = _Host()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        editor = await _mount_editor(pilot, app, _chapters(3))
        editor.selected_chapter_index = 1
        await pilot.pause()

        neighbor_title = editor.chapters[2].title
        neighbor_content = editor.chapters[2].content

        _press(editor, "delete-chapter-btn")
        await pilot.pause()

        table = editor.query_one("#chapter-table", DataTable)
        assert table.row_count == len(editor.chapters) == 2, (
            f"table has {table.row_count} rows for {len(editor.chapters)} "
            "chapters after delete -- the table never refreshed"
        )
        # Deleting index 1 clamps selection to the shifted-in neighbor,
        # now at index 1 (the old index 2 chapter).
        assert editor.selected_chapter_index == 1
        assert editor.chapters[1].title == neighbor_title
        preview = editor.query_one("#chapter-preview", TextArea)
        assert preview.text == neighbor_content, (
            "preview still shows the deleted chapter's old neighbor content, "
            "not the newly-selected one"
        )
        assert table.get_row_at(1)[2] == neighbor_title
