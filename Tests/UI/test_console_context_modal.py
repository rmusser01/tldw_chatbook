import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Label, Static, TextArea

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console import console_context_modal
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal


SNAPSHOT = ConsoleContextSnapshot(
    current_messages=[
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="Hello"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="Hi"),
    ],
    next_send_payload={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    },
)

EMPTY_SNAPSHOT = ConsoleContextSnapshot(current_messages=[], next_send_payload={})


async def _snapshot_factory() -> ConsoleContextSnapshot:
    return SNAPSHOT


async def _empty_factory() -> ConsoleContextSnapshot:
    return EMPTY_SNAPSHOT


class ModalHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")

    def on_mount(self) -> None:
        self.push_screen(ConsoleContextModal(_snapshot_factory, token_estimate=42))


@pytest.mark.asyncio
async def test_context_modal_renders_tabs():
    app = ModalHarness()

    async with app.run_test(size=(100, 40)) as _pilot:
        modal = app.screen
        header = modal.query_one("#console-context-header", Static)
        header_text = str(header.renderable)
        assert "Chat Context" in header_text
        assert "42 tokens" in header_text

        current_container = modal.query_one(
            "#console-context-current-body", Vertical
        )
        text_areas = current_container.query(TextArea)
        assert any("Hello" in ta.text for ta in text_areas)

        next_container = modal.query_one(
            "#console-context-next-send-body", Vertical
        )
        labels = list(next_container.query(Label))
        assert any("gpt-4" in str(label.renderable) for label in labels)


@pytest.mark.asyncio
async def test_context_modal_empty_state():
    app = ModalHarness()
    app._push_empty = lambda: app.push_screen(
        ConsoleContextModal(_empty_factory)
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app._push_empty()
        await pilot.pause()
        modal = app.screen
        current_container = modal.query_one(
            "#console-context-current-body", Vertical
        )
        labels = list(current_container.query(Label))
        assert any(
            "No conversation context" in str(label.renderable)
            for label in labels
        )


@pytest.mark.asyncio
async def test_context_modal_in_progress_warning():
    app = ModalHarness()
    app._push_in_progress = lambda: app.push_screen(
        ConsoleContextModal(_snapshot_factory, in_progress=True)
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app._push_in_progress()
        await pilot.pause()
        modal = app.screen
        warning = modal.query_one("#console-context-warning", Static)
        assert "in progress" in str(warning.renderable)
        refresh_button = modal.query_one("#console-context-refresh", Button)
        assert refresh_button.disabled


class ActionHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")


@pytest.mark.asyncio
async def test_context_modal_toggle_raw_json():
    app = ActionHarness()
    expected_raw = json.dumps(SNAPSHOT.next_send_payload, indent=2, default=str)

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleContextModal(_snapshot_factory))
        await pilot.pause()

        await pilot.click("#console-context-raw")
        await pilot.pause()

        modal = app.screen
        next_container = modal.query_one(
            "#console-context-next-send-body", Vertical
        )
        text_areas = list(next_container.query(TextArea))
        assert any(ta.text == expected_raw for ta in text_areas)


@pytest.mark.asyncio
async def test_context_modal_refresh_invokes_factory():
    calls = 0

    async def counting_factory() -> ConsoleContextSnapshot:
        nonlocal calls
        calls += 1
        return SNAPSHOT

    app = ActionHarness()

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleContextModal(counting_factory))
        await pilot.pause()
        assert calls == 1

        await pilot.click("#console-context-refresh")
        await pilot.pause()
        assert calls == 2


@pytest.mark.asyncio
async def test_context_modal_close_dismisses():
    app = ActionHarness()

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleContextModal(_snapshot_factory))
        await pilot.pause()
        assert isinstance(app.screen, ConsoleContextModal)

        await pilot.click("#console-context-close")
        await pilot.pause()
        assert not isinstance(app.screen, ConsoleContextModal)


@pytest.mark.asyncio
async def test_context_modal_copy_json(monkeypatch):
    app = ActionHarness()
    expected_text = json.dumps(SNAPSHOT.next_send_payload, indent=2, default=str)
    fake_copy = types.SimpleNamespace(copy=Mock())
    monkeypatch.setitem(sys.modules, "pyperclip", fake_copy)

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleContextModal(_snapshot_factory))
        await pilot.pause()

        await pilot.click("#console-context-copy")
        await pilot.pause()

        fake_copy.copy.assert_called_once_with(expected_text)


@pytest.mark.asyncio
async def test_context_modal_save_to_file(tmp_path, monkeypatch):
    app = ActionHarness()
    expected_text = json.dumps(SNAPSHOT.next_send_payload, indent=2, default=str)

    class FakePath:
        """Redirect filesystem operations under ``tmp_path`` for hermetic tests."""

        def __init__(self, *parts: str | Path) -> None:
            self._path = tmp_path.joinpath(*parts)

        @classmethod
        def home(cls):
            return cls(tmp_path)

        def __truediv__(self, other: str) -> "FakePath":
            return FakePath(self._path, other)

        def __getattr__(self, name: str):
            return getattr(self._path, name)

    monkeypatch.setattr(
        console_context_modal,
        "Path",
        FakePath,
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleContextModal(_snapshot_factory))
        await pilot.pause()

        await pilot.click("#console-context-save")
        await pilot.pause()

        saved_files = list((tmp_path / "Downloads").glob("*.json"))
        assert len(saved_files) == 1
        assert saved_files[0].read_text(encoding="utf-8") == expected_text


@pytest.mark.asyncio
async def test_context_modal_save_to_file_failure(monkeypatch):
    app = ActionHarness()

    class FailingPath:
        """Path stand-in whose ``write_text`` always raises ``OSError``."""

        @classmethod
        def home(cls):
            return cls()

        def __truediv__(self, other: str) -> "FailingPath":
            return self

        def mkdir(self, **kwargs: object) -> None:
            return None

        def write_text(self, *args: object, **kwargs: object) -> None:
            raise OSError("disk full")

    monkeypatch.setattr(
        console_context_modal,
        "Path",
        FailingPath,
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleContextModal(_snapshot_factory))
        await pilot.pause()

        # Should not crash; notification severity is checked best-effort.
        await pilot.click("#console-context-save")
        await pilot.pause()
