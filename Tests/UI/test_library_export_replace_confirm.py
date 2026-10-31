"""TASK-34000.3 (review N-11): an export onto an existing file asks first.

The PR-gated core (``scripts/ui_pr_gate_census.txt``): a note export and a
prompt export, each onto a pre-existing file, driven through the real
``FileSave`` picker at the review's wide size. The file on disk is the
witness: its bytes must be unchanged until Replace is chosen, and unchanged
after Escape. ``test_library_export_replace_confirm_extended.py`` holds the
Cancel-button, compact-size and remembered-folder variants, off the UI lane.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from textual.widgets import Button, Label, Static

from Tests.UI.test_library_prompt_export_journeys import _open_export, _save_to
from Tests.UI.test_library_prompts_canvas import (
    _build_test_app as _build_prompts_test_app,
    _open_prompt_editor,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _open_note_editor,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileNameInput
from tldw_chatbook.Third_Party.textual_fspicker.parts.directory_navigation import (
    DirectoryNavigation,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

#: ``bootstrap_profile`` as in ``test_library_quit_guard.py``: ``_build_test_app``
#: reloads app config, which trips the per-test sandbox's profile selection
#: (``RecoveryRequired: raw_source_selection_changed``, lessons-testing-evidence).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: The review's wide repro size (verify captures 09 and 15).
SIZE = (160, 45)

PRECIOUS = b"PRECIOUS USER FILE\n"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _precious_file(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / "exp"
    folder.mkdir(exist_ok=True)
    destination = folder / name
    destination.write_bytes(PRECIOUS)
    return destination


async def _await_replace_prompt(host, pilot, destination, before) -> ConfirmationDialog:
    """Wait for the replace prompt; fail first on the file having changed."""
    await _wait_for_condition(
        pilot,
        lambda: (
            isinstance(host.screen, ConfirmationDialog) or _md5(destination) != before
        ),
        message="neither a replace prompt nor a write followed Save",
    )
    assert _md5(destination) == before, (
        f"the export replaced {destination} before asking"
    )
    prompt = host.screen
    assert isinstance(prompt, ConfirmationDialog), (
        f"expected a replace prompt on top, found {type(prompt).__name__}"
    )
    await _wait_for_condition(
        pilot, lambda: prompt.focused is not None, message="the prompt's focus"
    )
    focused = prompt.focused
    assert isinstance(focused, Button) and focused.id == "cancel-button", (
        f"Cancel must be the focused default, not {focused!r}"
    )
    assert str(focused.label) == "Cancel"
    replace = prompt.query_one("#confirm-button", Button)
    assert str(replace.label) == "Replace"
    title = str(prompt.query_one(".dialog-title", Static).renderable)
    message = str(prompt.query_one(".dialog-message", Label).renderable)
    assert destination.name in message, (title, message)
    assert destination.parent.name in message, (title, message)
    return prompt


async def _returned(host, pilot, screen) -> None:
    await _wait_for_condition(
        pilot, lambda: host.screen is screen, message="the prompt to close"
    )
    await pilot.pause()


def _notice_messages(host) -> list[str]:
    return [notice.message for notice in host._notifications]


def _prompt_host(tmp_path):
    """One real prompt behind the real scope service, on the bundle-CSS harness.

    The same object graph as ``test_library_prompt_export_journeys._export_host``
    minus its production-CSS harness and theme sweep: this file sits in the
    PR gate, and the bundle harness mounts about ten seconds faster.
    """
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _, _ = db.add_prompt(
        name="Export [bold] café",
        author="Zoë",
        details="A reusable message.",
        system_prompt="",
        user_prompt="Keep [bold] café.",
        keywords=["alpha", "beta"],
    )
    app = _build_prompts_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    app.notify = host.notify
    app.copy_to_clipboard = host.copy_to_clipboard
    return db, prompt_id, host


async def _picker_location(host, pilot) -> Path:
    await _wait_for_condition(
        pilot,
        lambda: (
            isinstance(host.screen, FileSave)
            and isinstance(host.screen.focused, FileNameInput)
        ),
        message="the next export picker did not open",
    )
    return Path(host.screen.query_one(DirectoryNavigation).location)


# --- AC#1 / AC#2 / AC#4 / AC#5: prompt export -----------------------------


@pytest.mark.asyncio
async def test_prompt_export_onto_an_existing_file_asks_escape_keeps_it_replace_writes(
    tmp_path,
):
    """Verify capture 15: the prompt export must not overwrite silently."""
    db, prompt_id, host = _prompt_host(tmp_path)
    original = db.fetch_prompt_details(prompt_id)
    destination = _precious_file(tmp_path, "Weekly review coach.md")
    before = _md5(destination)

    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        dialog, filename = await _open_export(host, screen, pilot)
        await _save_to(host, dialog, filename, pilot, destination)
        await _await_replace_prompt(host, pilot, destination, before)

        await pilot.press("escape")
        await _returned(host, pilot, screen)
        assert _md5(destination) == before, "Escape must leave the file alone"
        assert destination.read_bytes() == PRECIOUS
        status = screen._prompts_state.status
        assert "exported" not in status.lower(), status
        assert "cancelled" in status.lower(), status
        assert not [
            message for message in _notice_messages(host) if "exported" in message
        ]

        dialog, filename = await _open_export(host, screen, pilot)
        await _save_to(host, dialog, filename, pilot, destination)
        prompt = await _await_replace_prompt(host, pilot, destination, before)
        prompt.query_one("#confirm-button", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: _md5(destination) != before,
            message="Replace did not write the file",
        )
        await _returned(host, pilot, screen)
        written = destination.read_text(encoding="utf-8")
        assert original["name"] in written
        assert PRECIOUS.decode() not in written
        assert not list(destination.parent.glob(".*.tmp")), "temp file left behind"
        await _wait_for_condition(
            pilot,
            lambda: str(destination) in screen._prompts_state.status,
            message="the success status must name the full destination path",
        )
    assert db.fetch_prompt_details(prompt_id) == original


# --- AC#1 / AC#2 / AC#4 / AC#5: note export -------------------------------


async def _open_note_export(host, screen, pilot):
    screen.query_one("#library-note-export-md", Button).press()
    await _wait_for_condition(
        pilot,
        lambda: (
            isinstance(host.screen, FileSave)
            and isinstance(host.screen.focused, FileNameInput)
        ),
        message="Export Markdown did not open the picker with filename focus",
    )
    dialog = host.screen
    return dialog, dialog.query_one(FileNameInput)


def _note_status(screen) -> str:
    return str(screen.query_one("#library-note-transfer-status", Static).renderable)


@pytest.mark.asyncio
async def test_note_export_onto_an_existing_file_asks_escape_keeps_it_replace_writes(
    tmp_path,
):
    """Verify capture 09: the note export must not overwrite silently."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    app.notify = host.notify
    destination = _precious_file(tmp_path, "precious.md")
    before = _md5(destination)

    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        dialog, filename = await _open_note_export(host, screen, pilot)
        await _save_to(host, dialog, filename, pilot, destination)
        await _await_replace_prompt(host, pilot, destination, before)

        await pilot.press("escape")
        await _returned(host, pilot, screen)
        assert _md5(destination) == before, "Escape must leave the file alone"
        assert destination.read_bytes() == PRECIOUS
        assert _note_status(screen) != "Export complete."
        assert not [
            message for message in _notice_messages(host) if "exported" in message
        ]

        dialog, filename = await _open_note_export(host, screen, pilot)
        await _save_to(host, dialog, filename, pilot, destination)
        prompt = await _await_replace_prompt(host, pilot, destination, before)
        prompt.query_one("#confirm-button", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: _md5(destination) != before,
            message="Replace did not write the file",
        )
        await _returned(host, pilot, screen)
        written = destination.read_text(encoding="utf-8")
        assert "title: Q3 retro" in written
        assert written.endswith("alpha budget line")
        assert not list(destination.parent.glob(".*.tmp")), "temp file left behind"
        await _wait_for_condition(
            pilot,
            lambda: _note_status(screen) == "Export complete.",
            message="the note status did not reach Export complete.",
        )
        success = [m for m in _notice_messages(host) if "exported successfully" in m]
        assert success and str(destination) in success[-1], _notice_messages(host)
