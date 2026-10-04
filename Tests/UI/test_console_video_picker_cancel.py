"""Cancel in the generated video's Save-to-disk picker returns to the choice.

TASK-33622.17 (owner decision 2026-10-03). A generated video that missed
managed storage waits in the storage choice (Keep here / Retry, Save to disk,
Discard). **Save to disk** closes that choice and opens a file picker. Until
this task, Escape or **Cancel** in the picker discarded the video without a
word -- a paid generation lost to a dismissed file dialog. Now the picker's
Cancel and Escape return to the storage choice with the same video, and only
an explicit **Discard** throws it away.

Every test drives the real ``TldwCli``, the real pending-video resolver and the
real picker and choice screens, and presses the real keys and buttons. The
irreversible shutdown is replaced by a recorder, and the OS opener (called
after a successful save) by a list.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_app_quit_in_flight_modals import (
    _recording_cleanup,
    _video_waiting_in_the_save_picker,
)
from Tests.UI.test_app_quit_under_modal import (
    _dialogs_titled,
    _mounted_console,
    _until,
)
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
    ConsoleVideoCapacityModal,
)
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: The storage choice's buttons for a video over the configured capacity.
_OVER_CAPACITY_CHOICES = ["Keep here (remove other videos)", "Save to disk", "Discard"]
_QUIT_TITLE = "Discard generated video and quit?"


def _choice_labels(choice: ConsoleVideoCapacityModal) -> list[str]:
    return [
        button.label.plain for button in choice.query("#video-capacity-actions Button")
    ]


async def _back_at_the_storage_choice(app, pilot, picker, artifact, how: str):
    """Wait for the picker's cancel to settle; fail if it lost the video.

    Returns:
        The storage choice the cancel returned to.
    """
    await _until(
        pilot,
        lambda: (
            isinstance(app.screen, ConsoleVideoCapacityModal) or artifact.stream.closed
        ),
        f"{how} in the Save-to-disk picker to settle",
        timeout=5.0,
    )
    assert not artifact.stream.closed, (
        f"{how} in the Save-to-disk picker discarded the generated video "
        "instead of returning to the storage choice"
    )
    assert picker not in app.screen_stack
    await pilot.pause(0.2)
    return app.screen


def _assert_the_same_video_waits_alone(console, artifact) -> None:
    """The one staged video is still owned, open and unduplicated."""
    video = console._video
    assert video._owns_pending_console_video(artifact)
    assert not artifact.stream.closed
    assert artifact.stream.close_calls == 0
    # The same staged payload, not a second copy: one registry entry, and no
    # operation, publication gate or deferred close left behind by the picker.
    assert video._pending_console_video_artifacts() == {artifact.message_id: artifact}
    assert video._pending_video_active_operations == {}
    assert video._pending_video_operation_cancels == {}
    assert video._pending_video_deferred_closes == {}
    artifact.rewind()
    assert artifact.stream.read() == b"paid generation"
    artifact.rewind()


async def _open_the_picker_from(app, pilot, choice) -> EnhancedFileSave:
    choice.query_one("#video-capacity-save", Button).press()
    await _until(
        pilot,
        lambda: isinstance(app.screen, EnhancedFileSave),
        "Save to disk to open the picker again",
    )
    await pilot.pause(0.2)
    return cast(EnhancedFileSave, app.screen)


async def test_cancelling_the_save_picker_returns_to_the_storage_choice(
    monkeypatch, tmp_path: Path
):
    """Escape, then Cancel, each return to the choice; a later save completes."""
    from Tests.Chat.test_console_video_capacity import _artifact

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    artifact = _artifact(b"paid generation", message_id="cancel-returns-to-choice")
    destination = tmp_path / "saved"
    destination.mkdir()
    target = destination / "kept.mp4"
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        opened: list[Path] = []
        monkeypatch.setattr(console, "_open_video_with_os", opened.append)
        first_picker = await _video_waiting_in_the_save_picker(
            app, pilot, console, artifact
        )

        # 1. Escape in the picker: back to the choice, the video still there.
        await pilot.press("escape")
        choice = await _back_at_the_storage_choice(
            app, pilot, first_picker, artifact, "Escape"
        )
        assert isinstance(choice, ConsoleVideoCapacityModal)
        assert _choice_labels(choice) == _OVER_CAPACITY_CHOICES
        _assert_the_same_video_waits_alone(console, artifact)

        # The choice it returned to still asks before Ctrl+Q discards the video.
        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, _QUIT_TITLE)) or bool(cleanups),
            "Ctrl+Q over the re-opened storage choice to ask first",
            timeout=5.0,
        )
        assert cleanups == []
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is choice, "Stay to restore the choice")
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Stay",
        )
        _assert_the_same_video_waits_alone(console, artifact)

        # 2. A second round, through the visible Cancel button this time.
        second_picker = await _open_the_picker_from(app, pilot, choice)
        assert second_picker is not first_picker
        await pilot.click("#cancel")
        choice = await _back_at_the_storage_choice(
            app, pilot, second_picker, artifact, "Cancel"
        )
        assert _choice_labels(choice) == _OVER_CAPACITY_CHOICES
        _assert_the_same_video_waits_alone(console, artifact)

        # 3. The video survived both cancels, so a third try still saves it.
        third_picker = await _open_the_picker_from(app, pilot, choice)
        third_picker.query_one("#filename-input", Input).value = str(target)
        await pilot.click("#select")
        await _until(
            pilot,
            lambda: artifact.stream.closed,
            "the save to finish and release the staged video",
            timeout=5.0,
        )
        await pilot.pause(0.2)

        assert target.read_bytes() == b"paid generation"
        # Exactly one file: no staging sibling left behind by any round.
        assert sorted(destination.iterdir()) == [target]
        assert [path.resolve() for path in opened] == [target.resolve()]
        assert artifact.stream.close_calls == 1
        assert console._video._pending_console_video_artifacts() == {}
        assert console._video._pending_video_operation_cancels == {}
        assert not [
            screen
            for screen in app.screen_stack
            if isinstance(screen, (ConsoleVideoCapacityModal, EnhancedFileSave))
        ]
        assert cleanups == []


async def test_only_an_explicit_discard_throws_the_video_away(monkeypatch):
    """After a cancelled picker, the choice's Discard is what ends the video."""
    from Tests.Chat.test_console_video_capacity import _artifact

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    artifact = _artifact(b"paid generation", message_id="discard-after-cancel")
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        appended: list[tuple] = []
        store = console._ensure_console_chat_store()
        monkeypatch.setattr(
            store,
            "append_video_message",
            lambda *args, **kwargs: appended.append((args, kwargs)),
        )
        picker = await _video_waiting_in_the_save_picker(app, pilot, console, artifact)
        await pilot.press("escape")
        choice = await _back_at_the_storage_choice(
            app, pilot, picker, artifact, "Escape"
        )
        _assert_the_same_video_waits_alone(console, artifact)

        choice.query_one("#video-capacity-discard", Button).press()
        await _until(
            pilot,
            lambda: artifact.stream.closed,
            "Discard to release the staged video",
            timeout=5.0,
        )
        await pilot.pause(0.2)
        assert artifact.stream.close_calls == 1
        assert console._video._pending_console_video_artifacts() == {}
        assert appended == []
        assert not isinstance(app.screen, ConsoleVideoCapacityModal)
        assert cleanups == []
