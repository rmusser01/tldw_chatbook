"""Mounted-widget qualification of formatting as a native history unit."""

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.document._document import Selection
from textual.widgets import Button, TextArea

from tldw_chatbook.Document_Editing.models import (
    CheckResult,
    DocumentRequest,
    FormatResult,
    RequestKey,
)
from tldw_chatbook.Widgets.document_editing_bridge import TextAreaFormatBridge

KEY = RequestKey("local", "note", 1, 1, 1, 1)
FORMATTED = '{\n  "a": 1\n}'


def result(request, replacement=FORMATTED):
    check = CheckResult.for_request(request, "checked", True, (), False, True)
    return FormatResult(request.key, "formatted", replacement, check)


class Harness(App):
    def compose(self) -> ComposeResult:
        yield Button("Format document", id="format")
        yield TextArea('{"a":1}', id="editor")

    def on_mount(self):
        self.editor = self.query_one(TextArea)
        self.bridge = TextAreaFormatBridge(self.editor)
        self.editor.focus()

    def on_button_pressed(self, event):
        request = DocumentRequest(KEY, self.editor.text, "json")
        self.bridge.capture_request(request)
        assert self.bridge.apply_format(result(request), KEY, True)
        self.editor.focus()


@pytest.mark.asyncio
async def test_native_typing_format_typing_undo_redo():
    app = Harness()
    async with app.run_test() as pilot:
        await pilot.press("end", "space")
        before = app.editor.text
        assert before == '{"a":1} '
        await pilot.click("#format")
        assert app.editor.text == FORMATTED
        assert app.editor.document.end == (2, 1)
        await pilot.press("ctrl+end", "space")
        assert app.editor.text != FORMATTED
        await pilot.press("ctrl+z")
        assert app.editor.text == FORMATTED
        await pilot.press("ctrl+z")
        assert app.editor.text == before
        await pilot.press("ctrl+y")
        assert app.editor.text == FORMATTED
        await pilot.press("ctrl+z", "ctrl+z")
        assert app.editor.text == '{"a":1}'


@pytest.mark.asyncio
async def test_guards_pending_typing_new_capture_readonly_and_noop():
    app = Harness()
    async with app.run_test():
        request = DocumentRequest(KEY, app.editor.text, "json")
        app.bridge.capture_request(request)
        app.editor.insert(" ")  # Textual Changed is queued; supplied key is still old.
        assert not app.bridge.apply_format(result(request), KEY, True)
        app.editor.load_text(request.text)
        for field in KEY.__dataclass_fields__:
            changed = replace(
                KEY,
                **{
                    field: "other"
                    if field.endswith("id") and field != "request_id"
                    else 2
                },
            )
            assert not app.bridge.apply_format(result(request), changed, True)
        newer = replace(request, key=replace(KEY, request_id=2))
        app.bridge.capture_request(newer)
        assert not app.bridge.apply_format(result(request), KEY, True)
        assert not app.bridge.apply_format(result(newer), newer.key, False)
        app.editor.read_only = True
        assert not app.bridge.apply_format(result(newer), newer.key, True)
        app.editor.read_only = False
        app.editor.disabled = True
        assert not app.bridge.apply_format(result(newer), newer.key, True)
        app.editor.disabled = False
        app.screen.disabled = True
        assert not app.bridge.apply_format(result(newer), newer.key, True)
        app.screen.disabled = False
        assert not app.bridge.apply_format(result(newer, request.text), newer.key, True)
        assert app.bridge.apply_format(result(newer), newer.key, True)


@pytest.mark.asyncio
async def test_maps_backwards_selection_to_same_token():
    app = Harness()
    async with app.run_test() as pilot:
        app.editor.selection = Selection((0, 4), (0, 1))
        request = DocumentRequest(KEY, app.editor.text, "json")
        app.bridge.capture_request(request)
        assert app.bridge.apply_format(result(request), KEY, True)
        assert app.editor.selected_text == '"a"'
        assert app.editor.selection.start > app.editor.selection.end
        await pilot.press("ctrl+z")
        assert app.editor.selection == Selection((0, 4), (0, 1))


@pytest.mark.asyncio
async def test_noop_does_not_add_history_and_disabled_result_preserves_typing():
    app = Harness()
    async with app.run_test() as pilot:
        await pilot.press("end", "space")
        request = DocumentRequest(KEY, app.editor.text, "json")
        app.bridge.capture_request(request)
        assert not app.bridge.apply_format(result(request, request.text), KEY, True)
        assert not app.bridge.apply_format(result(request), KEY, False)
        await pilot.press("ctrl+z")
        assert app.editor.text == '{"a":1}'


@pytest.mark.asyncio
async def test_scroll_and_focus_survive_format():
    app = Harness()
    async with app.run_test(size=(80, 20)) as pilot:
        source = "\n".join('{"a":1}' for _ in range(100))
        app.editor.load_text(source)
        await pilot.pause()
        app.editor.scroll_to(y=30, animate=False, force=True)
        app.editor.selection = Selection.cursor((40, 5))
        await pilot.pause()
        scroll = app.editor.scroll_offset
        request = DocumentRequest(KEY, source, "jsonl")
        app.bridge.capture_request(request)
        assert app.bridge.apply_format(
            result(request, source.replace(":", ": ")), KEY, True
        )
        await pilot.pause()
        assert app.editor.scroll_offset == scroll
        assert app.editor.has_focus
        assert app.editor.cursor_location == (40, 6)
