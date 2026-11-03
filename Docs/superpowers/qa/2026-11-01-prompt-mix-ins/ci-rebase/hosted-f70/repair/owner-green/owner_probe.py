"""One explicit onboarding system event at the real landing remount boundary."""

import functools
import json
from pathlib import Path

import pytest

OUT = Path(__file__).with_name("owner-events.jsonl")


@pytest.fixture(autouse=True)
def schedule_one_onboarding_sync(monkeypatch):
    from tldw_chatbook.Widgets.Library.library_entry_canvases import (
        LibraryLandingCanvas,
    )

    original_mount_all = LibraryLandingCanvas.mount_all
    fired = False

    def snapshot(canvas, phase):
        screen = canvas.screen
        callback = canvas._post_recompose_callback
        record = {
            "phase": phase,
            "callback_present": callback is not None,
            "callback_identity": id(callback) if callback is not None else None,
            "callback_func": (
                getattr(callback.func, "__name__", None)
                if isinstance(callback, functools.partial)
                else type(callback).__name__
            ),
            "canvas_attached": canvas.is_attached,
            "canvas_mounted": canvas.is_mounted,
            "canvas_queryable": screen.query_one("#library-landing-canvas") is canvas,
            "canvas_child_count": len(canvas.children),
            "landing_purpose_count": len(canvas.query("#library-canvas-landing")),
            "canvas_recompose_required": canvas._recompose_required,
            "rendered_generation": screen._library_snapshot_rendered_generation,
            "state_generation": screen._library_snapshot_state_generation,
            "entry_dirty": screen._library_entry_reconcile_dirty,
            "entry_pending_matches": screen._library_entry_reconcile_pending
            == (
                screen._library_snapshot_state_generation,
                screen._library_entry_route_key(),
            ),
            "screen_current": screen.app.screen is screen,
        }
        with OUT.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def mount_at_system_event(canvas, *args, **kwargs):
        nonlocal fired
        callback = canvas._post_recompose_callback
        if (
            not fired
            and canvas.is_attached
            and not canvas.children
            and isinstance(callback, functools.partial)
            and getattr(callback.func, "__name__", "")
            == "_finish_library_entry_canvas_sync"
            and canvas.screen._library_entry_reconcile_dirty
            and callback.keywords.get("generation")
            == canvas.screen._library_snapshot_state_generation
            and callback.keywords.get("route_key")
            == canvas.screen._library_entry_route_key()
        ):
            fired = True
            snapshot(canvas, "before-original-onboarding-sync")
            canvas.screen._sync_library_landing_lifecycle_presentation()
            snapshot(canvas, "after-original-onboarding-sync")
        return original_mount_all(canvas, *args, **kwargs)

    monkeypatch.setattr(LibraryLandingCanvas, "mount_all", mount_at_system_event)
    yield
    with OUT.open("a") as stream:
        stream.write(json.dumps({"phase": "probe-retired", "fired": fired}) + "\n")
