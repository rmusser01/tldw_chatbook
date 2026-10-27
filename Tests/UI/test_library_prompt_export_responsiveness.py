"""Saved-Prompt destination writes must leave the actual App pump responsive."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
import time

import pytest
from textual import events
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_library_export_replace_confirm import (
    PRECIOUS,
    _await_replace_prompt,
    _md5,
    _prompt_host,
)
from Tests.UI.test_library_prompt_export_journeys import (
    _assert_round_trip,
    _open_export,
    _save_to,
)
from Tests.UI.test_library_prompts_canvas import _open_prompt_editor
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.UI.Library_Modules import library_file_export
from tldw_chatbook.UI.Screens import library_screen

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held-writer"])
@pytest.mark.parametrize("replace", [False, True], ids=["new", "replace"])
@private_profile_test
async def test_prompt_export_dispatches_app_key_before_destination_release(
    tmp_path, monkeypatch, request, held, replace
):
    # A synchronous writer, or awaiting it from the App picker callback, breaks
    # this: the actual App event cannot dispatch until the gate is released.
    assert Path(library_file_export.__file__).resolve() == (
        ROOT / "tldw_chatbook/UI/Library_Modules/library_file_export.py"
    )
    assert Path(library_screen.__file__).resolve() == (
        ROOT / "tldw_chatbook/UI/Screens/library_screen.py"
    )
    db, prompt_id, host = _prompt_host(tmp_path)
    original = db.fetch_prompt_details(prompt_id)
    destination = tmp_path / "exported.md"
    if replace:
        destination.write_bytes(PRECIOUS)
    entered, release = threading.Event(), threading.Event()
    dispatched_before_release = []
    writer_threads = []
    timeline = []
    real_write = library_file_export.write_export_text
    real_on_event = host.on_event
    loop = None

    def mark(event, **detail):
        timeline.append({"event": event, "time": time.monotonic(), **detail})

    async def observe_event(event):
        is_witness = isinstance(event, events.Key) and event.key == "f12"
        if is_witness:
            mark("app_key_handler_entered")
        result = await real_on_event(event)
        # Textual can forward/bubble this one key through App again.
        if is_witness and not dispatched_before_release:
            before_release = entered.is_set() and not release.is_set()
            dispatched_before_release.append(before_release)
            mark("app_key_acknowledged", before_release=before_release)
            release.set()
        return result

    def destination_writer(*args, **kwargs):
        writer_threads.append(threading.get_ident())
        mark("writer_entered", thread=threading.get_ident())
        entered.set()
        if held:
            assert loop is not None
            mark("key_posted")
            loop.call_soon_threadsafe(host.post_message, events.Key("f12", None))
            # The actual App handler releases the write. The original 3s write
            # bound is only a watchdog, not an unrelated 0.5s dispatch SLA.
            if not release.wait(3):
                mark("watchdog_released_write")
                release.set()
            mark("write_released")
        result = real_write(*args, **kwargs)
        mark("write_completed")
        return result

    monkeypatch.setattr(host, "on_event", observe_event)
    monkeypatch.setattr(library_file_export, "write_export_text", destination_writer)
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            dialog, filename = await _open_export(host, screen, pilot)
            if replace:
                before = _md5(destination)
                await _save_to(host, dialog, filename, pilot, destination)
                prompt = await _await_replace_prompt(host, pilot, destination, before)
                assert not entered.is_set(), "writing started before Replace"
                prompt.query_one("#confirm-button", Button).focus()
                await pilot.pause()
            loop = asyncio.get_running_loop()
            mark("picker_or_confirmation_triggered")
            if replace:
                await pilot.press("enter")
            else:
                await _save_to(host, dialog, filename, pilot, destination)
            mark("picker_or_confirmation_returned")
            await _wait_for_condition(
                pilot,
                lambda: (
                    destination.exists()
                    and host.screen is screen
                    and "exported successfully" in screen._prompts_state.status
                ),
                message="real export did not complete",
            )
            await host.workers.wait_for_complete()
            await pilot.pause()
            written = destination.read_text(encoding="utf-8")
            _assert_round_trip(written, original)
            assert "Keep [bold] caf\u00e9." in written
            assert db.fetch_prompt_details(prompt_id) == original
            assert screen.focused.id == "library-prompt-export"
            assert "exported successfully" in screen._prompts_state.status
    finally:
        release.set()
        db.close()
        (tmp_path / "dispatch-timeline.json").write_text(
            json.dumps(timeline, indent=2), encoding="utf-8"
        )
    if held:
        assert entered.is_set(), "actual destination writer never entered"
        assert dispatched_before_release == [True], (
            "Library Prompt export blocked UI key dispatch until destination "
            "writer was released: " + json.dumps(timeline)
        )
        assert writer_threads[0] != threading.get_ident()
        event_order = [item["event"] for item in timeline]
        assert event_order.index("app_key_acknowledged") < event_order.index(
            "write_completed"
        )
    assert len(writer_threads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("replace", [False, True], ids=["cancel", "replace"])
@private_profile_test
async def test_prompt_export_racing_file_requires_an_explicit_answer(
    tmp_path, monkeypatch, request, replace
):
    db, prompt_id, host = _prompt_host(tmp_path)
    original = db.fetch_prompt_details(prompt_id)
    destination = tmp_path / "raced.md"
    real_write = library_file_export.write_export_text
    attempts = []

    def racing_write(path, content, *, overwrite=True):
        attempts.append(overwrite)
        if not overwrite:
            path.write_bytes(PRECIOUS)
        # Exercise the real atomic no-clobber refusal, not an injected exception.
        return real_write(path, content, overwrite=overwrite)

    monkeypatch.setattr(library_file_export, "write_export_text", racing_write)
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            dialog, filename = await _open_export(host, screen, pilot)
            await _save_to(host, dialog, filename, pilot, destination)
            await _wait_for_condition(
                pilot, destination.exists, message="racing file was not created"
            )
            prompt = await _await_replace_prompt(
                host, pilot, destination, _md5(destination)
            )
            await host.workers.wait_for_complete()
            assert attempts == [False]
            assert destination.read_bytes() == PRECIOUS
            assert "exported successfully" not in screen._prompts_state.status
            button = "#confirm-button" if replace else "#cancel-button"
            prompt.query_one(button, Button).focus()
            await pilot.pause()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: host.screen is screen
                and ("exported successfully" if replace else "cancelled")
                in screen._prompts_state.status,
                message="race answer did not settle export feedback",
            )
            await host.workers.wait_for_complete()
            assert screen.focused.id == "library-prompt-export"
            if replace:
                assert attempts == [False, True]
                _assert_round_trip(destination.read_text(encoding="utf-8"), original)
            else:
                assert attempts == [False]
                assert destination.read_bytes() == PRECIOUS
            assert db.fetch_prompt_details(prompt_id) == original
            assert not list(tmp_path.glob(".*")), "atomic temporary file leaked"
    finally:
        db.close()
