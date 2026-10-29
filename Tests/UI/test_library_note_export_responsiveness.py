"""Notes export keeps the native App pump available during destination I/O."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime
import json
import os
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
    _note_status,
    _notice_messages,
    _open_note_export,
)
from Tests.UI.test_library_prompt_export_journeys import _save_to
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
from tldw_chatbook.Library import library_notes_state
from tldw_chatbook.UI.Library_Modules import library_file_export

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]
ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MARKDOWN = (
    b"---\ntitle: Q3 retro\ndate: 2026-10-28 12:34:56\nkeywords: \n"
    b"note_id: n-1\n---\n\n# Q3 retro\n\nalpha budget line"
)
# This task preserves the existing atomic writer's native newline policy;
# adjacent source/newline-fidelity work owns any change to that policy.
EXPECTED_MARKDOWN = EXPECTED_MARKDOWN.replace(b"\n", os.linesep.encode("ascii"))


def _note_host(monkeypatch):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    # Established seeded Notes stand-ins, not a real Notes SQLite database.
    # Optional unrelated backend counts must not obscure source readiness.
    app.study_scope_service = None
    app.study_quiz_scope_service = None
    app.prompt_scope_service = None
    app.skills_scope_service = None
    host = LibraryHarness(app)
    app.notify = host.notify

    class ExportClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 28, 12, 34, 56, tzinfo=tz)

    monkeypatch.setattr(library_notes_state, "datetime", ExportClock)
    return app, host


async def _ready_note(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await host.workers.wait_for_complete()
    for _ in range(2):
        if screen._library_lookup_error is None:
            break
        # Worker completion can precede the native recovery callout's recompose.
        # Wait for its actual Retry, then deliver the real Button event.
        await _wait_for_condition(
            pilot,
            lambda: screen._library_lookup_error is None
            or bool(screen.query("#library-source-retry")),
            message=lambda: f"Source recovery never mounted: {screen._library_lookup_error}",
        )
        if screen._library_lookup_error is None:
            break
        retries = list(screen.query("#library-source-retry"))
        assert retries, "Source setup failed without its real Retry"
        retries[0].press()
        await pilot.pause()
        await host.workers.wait_for_complete()
    assert screen._library_lookup_error is None, "Notes source never became ready"
    await _open_note_editor(screen, pilot)
    assert screen._notes_controller._selected_note_id == "n-1"
    return screen


@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "held-writer"])
@pytest.mark.parametrize("replace", [False, True], ids=["new", "replace"])
@private_profile_test
async def test_note_export_dispatches_app_key_before_destination_release(
    tmp_path, monkeypatch, request, held, replace
):
    # Reverting only Notes worker admission makes the held cases fail: the
    # actual App key is acknowledged after the writer's watchdog releases it.
    assert Path(library_file_export.__file__).resolve() == (
        ROOT / "tldw_chatbook/UI/Library_Modules/library_file_export.py"
    )
    app, host = _note_host(monkeypatch)
    source = deepcopy(app.notes_scope_service.notes)
    destination = tmp_path / "exported.md"
    if replace:
        destination.write_bytes(PRECIOUS)
    entered, release = threading.Event(), threading.Event()
    dispatch, writer_threads, publications, timeline = [], [], [], []
    real_write = library_file_export.write_export_text
    real_on_event = host.on_event
    main_thread = threading.get_ident()
    loop = None

    def mark(event, **detail):
        timeline.append({"event": event, "time": time.monotonic(), **detail})

    async def observe_event(event):
        result = await real_on_event(event)
        if isinstance(event, events.Key) and event.key == "f12" and not dispatch:
            before = entered.is_set() and not release.is_set()
            dispatch.append(before)
            mark("app_key_acknowledged", before_release=before)
            release.set()
        return result

    def destination_writer(*args, **kwargs):
        writer_threads.append(threading.get_ident())
        mark("writer_entered", thread=threading.get_ident(), main_thread=main_thread)
        entered.set()
        if held:
            assert loop is not None
            mark("key_posted")
            loop.call_soon_threadsafe(host.post_message, events.Key("f12", None))
            if not release.wait(3):
                mark("watchdog_released_write")
                release.set()
        result = real_write(*args, **kwargs)
        publications.append(destination.read_bytes())
        mark("write_completed")
        return result

    monkeypatch.setattr(host, "on_event", observe_event)
    monkeypatch.setattr(library_file_export, "write_export_text", destination_writer)
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            try:
                screen = await _ready_note(host, pilot)
                dialog, filename = await _open_note_export(host, screen, pilot)
                operation = screen._notes_controller._library_notes_operation
                assert operation.running
                if replace:
                    await _save_to(host, dialog, filename, pilot, destination)
                    prompt = await _await_replace_prompt(
                        host, pilot, destination, _md5(destination)
                    )
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
                    lambda: destination.exists()
                    and host.screen is screen
                    and _note_status(screen) == "Export complete.",
                    message="real Notes export did not complete",
                )
                await host.workers.wait_for_complete()
                await pilot.pause()
                if held:
                    await _wait_for_condition(
                        pilot, lambda: bool(dispatch), message="App key never arrived"
                    )
                    assert dispatch == [True], (
                        "Notes writer blocked App dispatch: " + json.dumps(timeline)
                    )
                assert destination.read_bytes() == EXPECTED_MARKDOWN
                assert app.notes_scope_service.notes == source
                assert (
                    screen._notes_controller._library_notes_operation.token
                    == operation.token
                )
                assert (
                    screen._notes_controller._library_notes_operation.phase
                    == "complete"
                )
                assert screen.focused.id == "library-note-body"
                success = [
                    m for m in _notice_messages(host) if "exported successfully" in m
                ]
                assert len(success) == 1
                assert str(destination).replace("\\", "/") in success[0].replace(
                    "\\", "/"
                )
                assert not list(tmp_path.glob(".*.tmp"))
            finally:
                release.set()
                await host.workers.wait_for_complete()
    finally:
        release.set()
        (tmp_path / "dispatch-timeline.json").write_text(
            json.dumps(timeline, indent=2), encoding="utf-8"
        )
    assert publications == [EXPECTED_MARKDOWN]
    assert len(writer_threads) == 1
    if held:
        assert dispatch == [True], "Notes writer blocked App dispatch: " + json.dumps(
            timeline
        )
        assert writer_threads[0] != main_thread
        event_order = [item["event"] for item in timeline]
        assert event_order.index("app_key_acknowledged") < event_order.index(
            "write_completed"
        )


@pytest.mark.parametrize("answer", ["cancel", "replace", "recheck-error"])
@private_profile_test
async def test_note_export_late_collision_keeps_precious_file_until_answer(
    tmp_path, monkeypatch, request, answer
):
    app, host = _note_host(monkeypatch)
    source = deepcopy(app.notes_scope_service.notes)
    destination = tmp_path / "raced.md"
    attempts = []
    real_write = library_file_export.write_export_text
    real_exists = library_file_export.export_destination_exists

    def racing_write(path, content, *, overwrite=True):
        attempts.append(overwrite)
        if not overwrite:
            path.write_bytes(PRECIOUS)
        return real_write(path, content, overwrite=overwrite)

    def destination_exists(path):
        if answer == "recheck-error" and attempts:
            raise PermissionError("test destination metadata unavailable")
        return real_exists(path)

    monkeypatch.setattr(library_file_export, "write_export_text", racing_write)
    monkeypatch.setattr(
        library_file_export, "export_destination_exists", destination_exists
    )
    async with host.run_test(size=(160, 45)) as pilot:
        try:
            screen = await _ready_note(host, pilot)
            dialog, filename = await _open_note_export(host, screen, pilot)
            await _save_to(host, dialog, filename, pilot, destination)
            await _wait_for_condition(
                pilot, destination.exists, message="racing file never appeared"
            )
            if answer == "recheck-error":
                await _wait_for_condition(
                    pilot,
                    lambda: "Export failed" in _note_status(screen),
                    message="late metadata failure escaped recoverable feedback",
                )
                assert "PermissionError" in " ".join(_notice_messages(host))
            else:
                prompt = await _await_replace_prompt(
                    host, pilot, destination, _md5(destination)
                )
                await host.workers.wait_for_complete()
                assert attempts == [False]
                assert destination.read_bytes() == PRECIOUS
                button = "#confirm-button" if answer == "replace" else "#cancel-button"
                prompt.query_one(button, Button).focus()
                await pilot.pause()
                await pilot.press("enter")
                await _wait_for_condition(
                    pilot,
                    lambda: host.screen is screen
                    and ("Export complete." if answer == "replace" else "cancelled")
                    in _note_status(screen),
                    message="collision answer did not settle Notes feedback",
                )
            await host.workers.wait_for_complete()
            assert destination.read_bytes() == (
                EXPECTED_MARKDOWN if answer == "replace" else PRECIOUS
            )
            assert attempts == ([False, True] if answer == "replace" else [False])
            assert app.notes_scope_service.notes == source
            assert not list(tmp_path.glob(".*.tmp"))
        finally:
            await host.workers.wait_for_complete()


@pytest.mark.parametrize(
    "retired", [False, True], ids=["context-region", "retired-screen"]
)
@private_profile_test
async def test_note_export_accepted_write_preserves_changed_result_owner(
    tmp_path, monkeypatch, request, retired
):
    app, host = _note_host(monkeypatch)
    source = deepcopy(app.notes_scope_service.notes)
    destination = tmp_path / "captured.md"
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    publications = []
    real_write = library_file_export.write_export_text

    def held_write(*args, **kwargs):
        entered.set()
        try:
            release.wait()
            result = real_write(*args, **kwargs)
            publications.append(destination.read_bytes())
            return result
        finally:
            finished.set()

    monkeypatch.setattr(library_file_export, "write_export_text", held_write)
    async with host.run_test(size=(160, 45)) as pilot:
        try:
            screen = await _ready_note(host, pilot)
            dialog, filename = await _open_note_export(host, screen, pilot)
            await _save_to(host, dialog, filename, pilot, destination)
            await _wait_for_condition(
                pilot, entered.is_set, message="accepted writer never started"
            )
            operation = screen._notes_controller._library_notes_operation
            assert operation.running
            # Existing Notes admission refuses another export while this token
            # owns the write, rather than cancelling the accepted worker.
            assert screen._begin_library_notes_operation("export") is None
            if retired:
                await host.pop_screen()
                await pilot.pause()
            else:
                screen.query_one("#library-note-context", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: screen._notes_controller._library_notes_active_region()
                    == "context",
                    message="the real Notes context action did not change region",
                )
                await pilot.pause()
            focus = host.screen.focused
            release.set()
            await asyncio.to_thread(finished.wait)
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert publications == [EXPECTED_MARKDOWN]
            assert app.notes_scope_service.notes == source
            assert host.screen.focused is focus
            if retired:
                assert host.screen is not screen
                assert not [
                    m for m in _notice_messages(host) if "exported successfully" in m
                ]
            else:
                assert screen._notes_controller._library_notes_operation is None
                assert "Export complete" not in str(
                    screen.query_one("#library-note-context-transfer-status").renderable
                )
        finally:
            release.set()
            if entered.is_set():
                await asyncio.to_thread(finished.wait)
            await host.workers.wait_for_complete()


@private_profile_test
async def test_note_export_failed_replace_keeps_file_and_allows_text_retry(
    tmp_path, monkeypatch, request
):
    # Inject at the atomic publication boundary: failure must preserve the
    # precious file and release Notes admission so the actual action can retry.
    from tldw_chatbook.Third_Party.textual_fspicker import FileSave
    from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileNameInput

    app, host = _note_host(monkeypatch)
    source = deepcopy(app.notes_scope_service.notes)
    destination = tmp_path / "retry.txt"
    destination.write_bytes(PRECIOUS)
    real_replace = os.replace

    def refuse_export(source_path, target_path, *args, **kwargs):
        if Path(target_path) == destination:
            raise PermissionError("test atomic publication refused")
        return real_replace(source_path, target_path, *args, **kwargs)

    async with host.run_test(size=(160, 45)) as pilot:
        try:
            screen = await _ready_note(host, pilot)
            for failing in (True, False):
                monkeypatch.setattr(
                    os, "replace", refuse_export if failing else real_replace
                )
                screen.query_one("#library-note-export-txt", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: isinstance(host.screen, FileSave)
                    and isinstance(host.screen.focused, FileNameInput),
                    message="Text export did not open the actual picker",
                )
                dialog = host.screen
                await _save_to(
                    host, dialog, dialog.query_one(FileNameInput), pilot, destination
                )
                prompt = await _await_replace_prompt(
                    host, pilot, destination, _md5(destination)
                )
                prompt.query_one("#confirm-button", Button).focus()
                await pilot.pause()
                await pilot.press("enter")
                await _wait_for_condition(
                    pilot,
                    lambda: ("Export failed" if failing else "Export complete.")
                    in _note_status(screen),
                    message="Text export did not settle failure/retry feedback",
                )
                await host.workers.wait_for_complete()
                if failing:
                    assert destination.read_bytes() == PRECIOUS
                    assert "PermissionError" in " ".join(_notice_messages(host))
                assert not list(tmp_path.glob(".*.tmp"))
            expected_text = (
                b"Title: Q3 retro\nDate: 2026-10-28 12:34:56\nKeywords: \n"
                b"Note ID: n-1\n\n==================================================\n\n"
                b"alpha budget line"
            )
            assert destination.read_bytes() == expected_text.replace(
                b"\n", os.linesep.encode("ascii")
            )
            assert app.notes_scope_service.notes == source
            assert screen.focused.id == "library-note-body"
            assert (
                len([m for m in _notice_messages(host) if "exported successfully" in m])
                == 1
            )
        finally:
            await host.workers.wait_for_complete()
