"""SCRATCH PROBE - not for commit. Does an expanded lifecycle restore the browse rows?"""
import pytest
from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE, LibraryHarness, _active_library_screen,
    _seed_conversations, _two_conversations, _two_media_items,
    _wait_for_library_shell,
)


@pytest.mark.asyncio
async def test_probe_expanded_lifecycle_restores_browse_rows():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    # the exact knob test_library_shell.py:1085 uses for a returning user
    rail = app.app_config.setdefault("library", {}).setdefault("rail_state", {})
    rail["lifecycle"] = LibraryLifecycle.EXPANDED.value
    app.library_new_profile_admission = False
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        ids = sorted(w.id for w in screen.query("*") if w.id and "library-row" in w.id)
        print("\n=== WITH lifecycle=EXPANDED ===")
        print("lifecycle:", screen._library_lifecycle)
        print("library-row-* ids:", ids)
        print("browse-media present:", bool(screen.query("#library-row-browse-media")))
