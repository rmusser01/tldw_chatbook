"""The OS video fallback must actually call the launcher, and log if it fails.

TASK-32811.2. When ffmpeg/ffplay are absent, `_play_console_video` falls
back to opening the file with the host OS player through the injected
`open_video_with_os` seam. The review flagged the launcher call and its
error handling; this pins the call itself (not a stub of the helper) and
that a raising launcher is logged and surfaced, never swallowed silently.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Console_Modules.video import ConsoleVideoController


class _App:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, str]] = []

    def notify(self, message, severity="information"):
        self.notifications.append((message, severity))

    def push_screen(self, *_a, **_k):  # pragma: no cover - not hit in fallback
        raise AssertionError("fallback must not push the in-app player")


def _controller_with_video(app, opened, video_path):
    message = SimpleNamespace(
        id="m1",
        video_metadata=SimpleNamespace(name="clip", container="mp4"),
    )
    store = SimpleNamespace(get_message=lambda _mid: message)

    controller = ConsoleVideoController(
        app_instance=app,
        sync_native_console_chat_ui=lambda: None,
        ensure_console_chat_store=lambda: store,
        wait_for_console_screen_result=lambda _s: None,
        open_video_with_os=opened,
        append_native_console_system_message=lambda *a, **k: None,
        default_console_session_settings=lambda: None,
        console_composer_or_none=lambda: None,
        clear_console_composer_draft=lambda: None,
    )
    # Resolve to a concrete existing path, and force the ffmpeg-absent branch.
    controller._ensure_console_video_store = lambda: SimpleNamespace(
        resolve_state=lambda *_a, **_k: ("ready", video_path)
    )
    controller._video_storage_message_id = lambda _m: "m1"
    return controller


@pytest.mark.asyncio
async def test_the_fallback_calls_the_os_launcher(tmp_path, monkeypatch):
    import tldw_chatbook.Media_Playback.player_pipeline as pipeline
    monkeypatch.setattr(
        pipeline, "playback_tools_available", lambda: (False, "install ffmpeg")
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00")
    opened: list[Path] = []
    app = _App()
    controller = _controller_with_video(app, opened.append, video_path)

    await controller._play_console_video("m1")

    assert opened == [video_path], "the OS launcher was not called"


@pytest.mark.asyncio
async def test_a_failing_launcher_is_logged_and_surfaced_not_swallowed(
    tmp_path, monkeypatch
):
    import tldw_chatbook.Media_Playback.player_pipeline as pipeline
    monkeypatch.setattr(
        pipeline, "playback_tools_available", lambda: (False, "install ffmpeg")
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00")

    def boom(_path):
        raise OSError("no player")

    app = _App()
    controller = _controller_with_video(app, boom, video_path)

    # Must not raise out of the handler.
    await controller._play_console_video("m1")

    # The failure is surfaced to the user, not silently dropped.
    assert any(sev == "error" for _msg, sev in app.notifications), (
        f"a failed launch was swallowed: {app.notifications}"
    )
