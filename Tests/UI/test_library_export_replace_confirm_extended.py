"""TASK-34000.3 (N-11): replace-confirmation variants kept off the UI lane.

The PR-gated core is ``test_library_export_replace_confirm.py``. These run
with the full ``Tests/UI`` suite: the Cancel button (not Escape) at the
review's compact size, for a note and a prompt, and AC#4's remembered
folder -- the next export picker opens where the last export went.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_export_replace_confirm import (
    PRECIOUS,
    _await_replace_prompt,
    _md5,
    _note_status,
    _notice_messages,
    _open_note_export,
    _picker_location,
    _precious_file,
    _returned,
)
from Tests.UI.test_library_prompt_export_journeys import (
    _export_host,
    _open_export,
    _save_to,
)
from Tests.UI.test_library_prompts_canvas import _open_prompt_editor
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _open_note_editor,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
)

#: ``bootstrap_profile`` as in ``test_library_quit_guard.py``: ``_build_test_app``
#: reloads app config, which trips the per-test sandbox's profile selection
#: (``RecoveryRequired: raw_source_selection_changed``, lessons-testing-evidence).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: The review's compact repro size.
COMPACT = (120, 36)


@pytest.mark.asyncio
async def test_prompt_export_cancel_button_keeps_the_existing_file_compact(tmp_path):
    db, prompt_id, _, host = _export_host(tmp_path, "textual-dark")
    original = db.fetch_prompt_details(prompt_id)
    destination = _precious_file(tmp_path, "Weekly review coach.md")
    before = _md5(destination)

    async with host.run_test(size=COMPACT) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        dialog, filename = await _open_export(host, screen, pilot)
        await _save_to(host, dialog, filename, pilot, destination)
        prompt = await _await_replace_prompt(host, pilot, destination, before)
        prompt.query_one("#cancel-button", Button).press()
        await _returned(host, pilot, screen)

        assert destination.read_bytes() == PRECIOUS
        assert "exported" not in screen._prompts_state.status.lower()
        assert not [m for m in _notice_messages(host) if "exported" in m]
    assert db.fetch_prompt_details(prompt_id) == original


@pytest.mark.asyncio
async def test_note_export_cancel_button_keeps_the_existing_file_compact(tmp_path):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    app.notify = host.notify
    destination = _precious_file(tmp_path, "precious.md")
    before = _md5(destination)

    async with host.run_test(size=COMPACT) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        dialog, filename = await _open_note_export(host, screen, pilot)
        await _save_to(host, dialog, filename, pilot, destination)
        prompt = await _await_replace_prompt(host, pilot, destination, before)
        prompt.query_one("#cancel-button", Button).press()
        await _returned(host, pilot, screen)

        assert destination.read_bytes() == PRECIOUS
        assert _note_status(screen) != "Export complete."
        assert not [m for m in _notice_messages(host) if "exported" in m]


@pytest.mark.asyncio
async def test_the_next_export_picker_opens_in_the_last_export_folder(tmp_path):
    """AC#4: after a Replace into ``exp/``, the next picker starts in ``exp/``."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    app.notify = host.notify
    destination = _precious_file(tmp_path, "precious.md")
    before = _md5(destination)

    async with host.run_test(size=COMPACT) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        dialog, filename = await _open_note_export(host, screen, pilot)
        assert await _picker_location(host, pilot) != destination.parent, (
            "the first picker of a session must not already sit in exp/"
        )
        await _save_to(host, dialog, filename, pilot, destination)
        prompt = await _await_replace_prompt(host, pilot, destination, before)
        prompt.query_one("#confirm-button", Button).press()
        await _returned(host, pilot, screen)
        assert _md5(destination) != before

        await _open_note_export(host, screen, pilot)
        assert await _picker_location(host, pilot) == destination.parent
        await pilot.press("escape")
        await _returned(host, pilot, screen)
