import asyncio
import functools
import json
import os
import time
from pathlib import Path

import pytest

MODE = os.environ.get("BUDDY_FOLDER_DIAG_MODE", "observe")
OUTPUT = Path(f"/private/tmp/buddy-folder-diagnostic-{MODE}.json")
events = []
started = time.monotonic()


def record(kind, **values):
    events.append({"t": round(time.monotonic() - started, 6), "event": kind, **values})


def state(screen):
    screen = getattr(screen, "_screen", screen)
    notes = screen._notes_state
    branches = []
    for key, value in notes.tree_branches.items():
        branches.append({"key": repr(key), **{
            field: getattr(value, field, None) for field in
            ("loading", "freshness", "generation", "topology_epoch", "item_ids", "error")
        }})
    workers = getattr(screen, "workers", ())
    return {"expanded": sorted(notes.tree_expanded_ids),
            "epoch": notes.tree_topology_epoch,
            "lifecycle": notes.tree_lifecycle_generation,
            "branches": branches,
            "workers": [{"name": w.name, "group": w.group, "state": str(w.state)}
                        for w in workers]}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    if item.module.__name__ != "Tests.UI.test_library_notes_sync_delete_restore":
        return
    mp = item._request.getfixturevalue("monkeypatch")
    module = item.module
    from textual.widgets import Button
    from textual.widget import Widget
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.UI.Library_Modules.library_notes_controller import LibraryNotesController
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback

    released = asyncio.Event()
    press_original = Button.press
    post_original = Button.post_message

    @functools.wraps(press_original)
    def press(button):
        if getattr(button, "folder_id", None) == "folder-1":
            record("folder_press", mounted=button.is_mounted, disabled=button.disabled,
                   display=button.display, parent=type(button.parent).__name__,
                   closing=button._closing, closed=button._closed,
                   pruning=button._pruning,
                   button_id=id(button), ancestors=[
                       {"type": type(a).__name__, "id": a.id, "display": a.display,
                        "mounted": a.is_mounted, "closing": a._closing,
                        "pruning": a._pruning,
                        "children_ready": getattr(a, "_children_ready", None)}
                       for a in button.ancestors])
            if MODE == "release_root_placements_on_press":
                released.set()
        return press_original(button)

    @functools.wraps(post_original)
    def post(button, message):
        accepted = post_original(button, message)
        if isinstance(message, Button.Pressed) and getattr(button, "folder_id", None) == "folder-1":
            record("folder_message_queued", accepted=accepted, button_id=id(button))
        return accepted

    mp.setattr(Button, "press", press)
    mp.setattr(Button, "post_message", post)
    original = LibraryNotesController.handle_library_notes_folder_row

    @functools.wraps(original)
    def processed(controller, event, _original=original):
        record("folder_handler_before", state=state(controller))
        result = _original(controller, event)
        record("folder_handler_after", state=state(controller))
        return result

    mp.setattr(LibraryNotesController, "handle_library_notes_folder_row", processed)
    for name in ("_request_library_notes_tree_slice", "_begin_library_notes_tree_visit"):
        original = getattr(LibraryScreen, name)

        def wrap(original, name):
            @functools.wraps(original)
            def called(screen, *args, **kwargs):
                record(name + "_before", args=repr(args), kwargs=kwargs, state=state(screen))
                result = original(screen, *args, **kwargs)
                record(name + "_after", state=state(screen))
                return result
            return called
        mp.setattr(LibraryScreen, name, wrap(original, name))

    original = LibraryScreen._apply_library_notes_tree_slice_page

    @functools.wraps(original)
    async def apply(screen, key, incoming, _original=original, **kwargs):
        record("apply_before", key=repr(key), authority=kwargs, state=state(screen))
        result = await _original(screen, key, incoming, **kwargs)
        record("apply_after", key=repr(key), state=state(screen))
        return result

    mp.setattr(LibraryScreen, "_apply_library_notes_tree_slice_page", apply)
    original = NotesScopeService.page_note_placements

    @functools.wraps(original)
    async def placements(service, _original=original, **kwargs):
        record("placements_enter", request=kwargs)
        if MODE == "release_root_placements_on_press" and kwargs.get("parent_id") is None:
            await released.wait()
        result = await _original(service, **kwargs)
        record("placements_return", request=kwargs)
        return result

    mp.setattr(NotesScopeService, "page_note_placements", placements)
    original = module._until

    @functools.wraps(original)
    async def until(pilot, predicate, what, *args, _original=original, **kwargs):
        if what == "the folder to list its note":
            record("note_wait_start", state=state(pilot.app.screen))
        try:
            return await _original(pilot, predicate, what, *args, **kwargs)
        finally:
            if what == "the folder to list its note":
                screen = pilot.app.screen
                record("note_wait_end", state=state(screen),
                       rows=[{"id": row.id, "label": str(row.label), "mounted": row.is_mounted}
                             for row in screen.query(".library-notes-folder-row, .library-notes-tree-note-row")])

    mp.setattr(module, "_until", until)

    if MODE == "real_prune_handoff":
        entered = asyncio.Event()
        release = asyncio.Event()
        armed = False
        consumed = False
        prune_original = Widget.on_prune

        @functools.wraps(prune_original)
        async def on_prune(widget, event):
            nonlocal consumed
            if armed and not consumed and widget.id == "library-notes-list":
                consumed = True
                record("real_prune_handoff", pruning=widget._pruning,
                       rows=[{"folder": getattr(b, "folder_id", None),
                              "mounted": b.is_mounted, "pruning": b._pruning}
                             for b in widget.query(Button)])
                entered.set()
                await release.wait()
                record("real_prune_release")
            return await prune_original(widget, event)

        mp.setattr(Widget, "on_prune", on_prune)
        open_original = module._open_synced_folder

        @functools.wraps(open_original)
        async def open_folder(screen, pilot):
            nonlocal armed
            await module._until(pilot, lambda: any(
                getattr(row, "folder_id", "") == "folder-1"
                for row in screen.query(".library-notes-folder-row")),
                "the diagnostic's actual synced folder")
            canvas = screen.query_one("#library-notes-canvas")
            armed = True
            recompose = asyncio.create_task(canvas.recompose())
            await asyncio.wait_for(entered.wait(), 2)
            asyncio.get_running_loop().call_later(0.1, release.set)
            try:
                await open_original(screen, pilot)
            finally:
                await recompose

        mp.setattr(module, "_open_synced_folder", open_folder)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if report.when == "call":
        OUTPUT.write_text(json.dumps({"mode": MODE, "outcome": report.outcome,
                                      "events": events}, indent=2, default=str))
