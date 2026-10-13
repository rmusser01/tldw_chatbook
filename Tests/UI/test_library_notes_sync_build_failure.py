"""A Notes-sync construction failure cannot abort the native Notes surface."""

import pytest

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _TITLE,
    SIZE,
    _ctrl_q,
    _library,
    _library_app,
    _open_note,
    _settings_without_splash,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


async def test_library_mount_keeps_failed_sync_state_without_rebuilding(
    tmp_path, monkeypatch
):
    events = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    attempts = []

    def fail_construction():
        attempts.append("build")
        raise RuntimeError("Synthetic Notes-sync construction failure")

    assert app._notes_sync_runtime_owner is None
    monkeypatch.setattr(app, "_construct_notes_sync_runtime_owner", fail_construction)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                controller = screen._library_notes_sync_controller
                snapshot = controller._runtime.snapshot()
                assert (snapshot.status, snapshot.next_action) == (
                    "failed",
                    "review_settings",
                )
                assert controller.snapshot.lasting_available is False
                assert attempts == ["build"]
                assert isinstance(app._notes_sync_runtime_build_failure, RuntimeError)
                assert screen._library_note_session._recovery is None
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                assert body.text == _BODY
                assert profile.note(profile.note_id)["content"] == _BODY
                assert await _ctrl_q(pilot, app, events) is None
                assert events == [("quit", None)]
                assert attempts == ["build"]
    finally:
        profile.db.close_connection()
