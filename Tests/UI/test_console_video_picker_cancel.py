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
irreversible shutdown is replaced by a recorder (in the Discard-and-quit test,
one that then calls the real ``App.exit``), and the OS opener (called after a
successful save) by a list.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_app_quit_in_flight_modals import (
    _click_when_shown,
    _recording_cleanup,
    _video_waiting_in_the_save_picker,
)
from Tests.UI.test_app_quit_under_modal import (
    _dialogs_titled,
    _mounted_console,
    _record_cleanup_then_exit,
    _until,
    _until_exited,
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


def _assert_the_same_video_waits_alone(
    console, artifact, *, operation_attempted: bool = False
) -> None:
    """The one staged video is still owned, open and unduplicated."""
    video = console._video
    assert video._owns_pending_console_video(artifact)
    assert not artifact.stream.closed
    assert artifact.stream.close_calls == 0
    # The same staged payload, not a second copy: one registry entry, and no
    # active operation or deferred close left behind by the picker/copy.
    assert video._pending_console_video_artifacts() == {artifact.message_id: artifact}
    assert video._pending_video_active_operations == {}
    if operation_attempted:
        from tldw_chatbook.Video_Generation.video_store import VideoPublicationGate

        # A real attempted copy retains this artifact's cancellation gate until
        # its final disposition, even though the native operation has retired.
        gates = video._pending_video_operation_cancels
        assert set(gates) == {artifact.message_id}
        gate = gates[artifact.message_id]
        assert isinstance(gate, VideoPublicationGate)
        with gate.claim_publication() as allowed:
            assert allowed
    else:
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
    """Cancels retain the video; a later save honors native capabilities."""
    from Tests.Chat.test_console_video_capacity import _artifact

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    artifact = _artifact(b"paid generation", message_id="cancel-returns-to-choice")
    destination = tmp_path / "saved"
    destination.mkdir()
    target = destination / "kept.mp4"
    async with app.run_test(size=(140, 44), notifications=True) as pilot:
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

        # 3. The original gate determines whether native external save is
        # supported. Do not replace it or the copy: unsupported platforms must
        # retain the paid video and offer another explicit storage decision.
        try:
            console._video._require_external_video_pinned_capabilities()
        except OSError as exc:
            assert str(exc) == "pinned external save unsupported"
            external_save_supported = False
        else:
            external_save_supported = True

        third_picker = await _open_the_picker_from(app, pilot, choice)
        third_picker.query_one("#filename-input", Input).value = str(target)
        previous_toasts = tuple(app.query("Toast"))
        assert await pilot.click("#select"), "the final Save click missed its button"
        if not external_save_supported:
            try:
                await _until(
                    pilot,
                    lambda: isinstance(app.screen, ConsoleVideoCapacityModal)
                    and app.screen is not choice
                    and any(
                        toast not in previous_toasts
                        and toast.is_on_screen
                        and toast.has_class("-error")
                        and "Could not save the generated video to "
                        in toast.render().plain
                        and str(target) in toast.render().plain
                        for toast in app.screen.query("Toast")
                    ),
                    "the unsupported save to show its error and return to the choice",
                    timeout=5.0,
                )
            except AssertionError as exc:
                # Diagnose the original predicate without another wait or any
                # replacement of notification, capability or copy behavior.
                try:
                    toast_rows = []
                    for screen in app.screen_stack[-8:]:
                        for toast in list(screen.query("Toast"))[:8]:
                            rendered = toast.render().plain
                            toast_rows.append(
                                {
                                    "screen_class": type(screen).__name__,
                                    "on_current_screen": screen is app.screen,
                                    "previous_toast": toast in previous_toasts,
                                    "is_on_screen": toast.is_on_screen,
                                    "error_class": toast.has_class("-error"),
                                    "save_prefix_matches": (
                                        "Could not save the generated video to "
                                        in rendered
                                    ),
                                    "target_matches": str(target) in rendered,
                                    "text": rendered[:512],
                                }
                            )
                    exc.add_note(
                        "Unsupported-save UI state: "
                        + repr(
                            {
                                "screen_class": type(app.screen).__name__,
                                "new_capacity_choice": (
                                    isinstance(app.screen, ConsoleVideoCapacityModal)
                                    and app.screen is not choice
                                ),
                                "is_previous_choice": app.screen is choice,
                                "picker_on_stack": third_picker in app.screen_stack,
                                "notifications_disabled": app._disable_notifications,
                                "notifications": [
                                    (note.severity, note.message[:512])
                                    for note in list(app._notifications)[:8]
                                ],
                                "toasts": toast_rows,
                            }
                        )
                    )
                except Exception as diagnostic_error:
                    exc.add_note(
                        "Unsupported-save UI diagnostic failed: "
                        f"{type(diagnostic_error).__name__}"
                    )
                raise
            assert third_picker not in app.screen_stack
            assert _choice_labels(app.screen) == _OVER_CAPACITY_CHOICES
            _assert_the_same_video_waits_alone(
                console, artifact, operation_attempted=True
            )
            assert not target.exists()
            assert list(destination.iterdir()) == []
            assert opened == []
            assert await pilot.click("#video-capacity-discard")

        await _until(
            pilot,
            lambda: artifact.stream.closed,
            "the save or explicit discard to release the staged video",
            timeout=5.0,
        )
        await pilot.pause(0.2)

        if external_save_supported:
            assert target.read_bytes() == b"paid generation"
            # Exactly one file: no staging sibling left behind by any round.
            assert sorted(destination.iterdir()) == [target]
            assert [path.resolve() for path in opened] == [target.resolve()]
        else:
            assert list(destination.iterdir()) == []
            assert opened == []
        assert artifact.stream.close_calls == 1
        assert console._video._pending_console_video_artifacts() == {}
        assert console._video._pending_video_operation_cancels == {}
        assert console._video._pending_video_active_operations == {}
        assert console._video._pending_video_deferred_closes == {}
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


async def test_discard_and_quit_over_the_picker_exits_without_reopening_the_choice(
    monkeypatch,
):
    """Ctrl+Q's Discard and quit over the picker ends the app, not a new choice.

    A cancelled picker now re-opens the storage choice, so the shutdown that
    takes the picker down must never read as a cancel: the app exits, the video
    is released exactly once, and the resolver never asks for the storage choice
    again once the user has approved the quit. Only the persistence steps of the
    shutdown are stood in for; the ``App.exit`` that closes every screen is the
    real one.
    """
    from Tests.Chat.test_console_video_capacity import _artifact

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    artifact = _artifact(b"paid generation", message_id="discard-and-quit")
    quit_approved: list[bool] = []
    choices_after_approval: list[ConsoleVideoCapacityModal] = []
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        picker = await _video_waiting_in_the_save_picker(app, pilot, console, artifact)
        # Record the resolver's REQUEST for the choice, not the push: once the
        # app is exiting a requested screen may never reach the stack, and a
        # check on pushes alone stays green while the resolver loops back.
        wait_for_screen = console._video._wait_for_console_screen_result

        async def recording_wait_for_screen(screen):
            if quit_approved and isinstance(screen, ConsoleVideoCapacityModal):
                choices_after_approval.append(screen)
            return await wait_for_screen(screen)

        monkeypatch.setattr(
            console._video, "_wait_for_console_screen_result", recording_wait_for_screen
        )

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, _QUIT_TITLE)) or bool(cleanups),
            "Ctrl+Q over the Save-to-disk picker to ask first",
            timeout=5.0,
        )
        assert cleanups == []
        assert picker in app.screen_stack
        quit_approved.append(True)
        assert await _click_when_shown(app, pilot, "#confirm-button")
        await _until_exited(
            app, cleanups, "Discard and quit to reach the approved shutdown"
        )
    assert cleanups == [True]
    assert app.return_code == 0
    assert choices_after_approval == [], (
        "the shutdown that closed the Save-to-disk picker read as a cancel and "
        "asked for the storage choice again"
    )
    assert artifact.stream.closed
    assert artifact.stream.close_calls == 1
    assert console._video._pending_console_video_artifacts() == {}
