from __future__ import annotations

from pathlib import Path

import pytest
from textual.css.query import NoMatches
from textual.widgets import Input, Select, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Terminal.launch import discover_shell_choices
from tldw_chatbook.Widgets.Console.console_terminal_session_modal import (
    ConsoleTerminalSessionModal,
    TerminalSessionFormResult,
    build_default_terminal_name,
)


def _shell_choices():
    paths = {
        "bash": "/bin/bash",
        "zsh": "/bin/zsh",
        "sh": "/bin/sh",
    }
    return discover_shell_choices(
        platform_name="posix",
        account_shell="/bin/zsh",
        executable_lookup=paths.get,
        executable_is_file=lambda _path: True,
    )


def test_default_name_uses_first_normalized_unique_terminal_number() -> None:
    assert build_default_terminal_name(()) == "Terminal 1"
    assert build_default_terminal_name(("terminal 1", "TERMINAL 2")) == "Terminal 3"
    assert build_default_terminal_name(("Build", "Terminal 2")) == "Terminal 1"


@pytest.mark.asyncio
async def test_new_modal_defaults_to_name_default_shell_and_selected_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    modal = ConsoleTerminalSessionModal(
        mode="new",
        name="Terminal 1",
        shell_choices=_shell_choices(),
        start_directory=root,
        existing_names=(),
    )
    app = ConsolidatedCSSApp()
    results: list[TerminalSessionFormResult | None] = []

    async with app.run_test(size=(100, 34)) as pilot:
        app.push_screen(modal, callback=results.append)
        await pilot.pause()
        assert modal.query_one("#console-terminal-session-name", Input).value == (
            "Terminal 1"
        )
        assert modal.query_one("#console-terminal-session-shell", Select).value == (
            "default"
        )
        assert modal.query_one(
            "#console-terminal-session-directory", Input
        ).value == str(root)

        await pilot.click("#console-terminal-session-save")
        await pilot.pause()

    assert results == [
        TerminalSessionFormResult(
            name="Terminal 1",
            shell="default",
            start_directory=root,
        )
    ]


@pytest.mark.asyncio
async def test_new_modal_keeps_values_and_explains_name_and_directory_errors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    modal = ConsoleTerminalSessionModal(
        mode="new",
        name="Terminal 2",
        shell_choices=_shell_choices(),
        start_directory=root,
        existing_names=("Terminal 1",),
    )
    app = ConsolidatedCSSApp()
    results: list[TerminalSessionFormResult | None] = []

    async with app.run_test(size=(100, 34)) as pilot:
        app.push_screen(modal, callback=results.append)
        await pilot.pause()
        name = modal.query_one("#console-terminal-session-name", Input)
        directory = modal.query_one("#console-terminal-session-directory", Input)
        error = modal.query_one("#console-terminal-session-error", Static)

        name.value = "terminal 1"
        await pilot.click("#console-terminal-session-save")
        await pilot.pause()
        assert "unique" in str(error.renderable).lower()
        assert name.value == "terminal 1"
        assert results == []

        name.value = "Terminal 2"
        directory.value = "relative/path"
        # Avoid Textual classifying the second synthetic click as the tail of
        # the preceding click chain; a real user edit naturally exceeds it.
        await pilot.pause(0.6)
        await pilot.click("#console-terminal-session-save")
        await pilot.pause()
        assert "absolute existing directory" in str(error.renderable).lower()
        assert directory.value == "relative/path"
        assert results == []


@pytest.mark.asyncio
async def test_rename_modal_exposes_only_name_and_normalizes_submission(
    tmp_path: Path,
) -> None:
    modal = ConsoleTerminalSessionModal(
        mode="rename",
        name="Current",
        shell_choices=_shell_choices(),
        start_directory=tmp_path,
        existing_names=("Other",),
    )
    app = ConsolidatedCSSApp()
    results: list[TerminalSessionFormResult | None] = []

    async with app.run_test(size=(90, 26)) as pilot:
        app.push_screen(modal, callback=results.append)
        await pilot.pause()
        with pytest.raises(NoMatches):
            modal.query_one("#console-terminal-session-shell", Select)
        with pytest.raises(NoMatches):
            modal.query_one("#console-terminal-session-directory", Input)

        name = modal.query_one("#console-terminal-session-name", Input)
        name.value = "  Renamed  "
        await pilot.click("#console-terminal-session-save")
        await pilot.pause()

    assert results == [
        TerminalSessionFormResult(
            name="Renamed",
            shell=None,
            start_directory=None,
        )
    ]


def test_modal_shell_choices_are_exactly_the_discovered_allowlist(
    tmp_path: Path,
) -> None:
    choices = _shell_choices()
    modal = ConsoleTerminalSessionModal(
        mode="new",
        name="Terminal 1",
        shell_choices=choices,
        start_directory=tmp_path,
        existing_names=(),
    )

    assert modal.shell_options == tuple(
        (choice.label, choice.key) for choice in choices
    )
    assert all(" " not in choice.key for choice in choices)
