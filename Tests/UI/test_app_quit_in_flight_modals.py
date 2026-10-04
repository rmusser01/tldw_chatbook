"""Ctrl+Q under a modal that is mid-operation, or holding a generated video.

TASK-33622.15, the residue of TASK-33622.10's review. Ctrl+Q is a priority
binding, so the quit flow runs while any modal is open and asks each modal's
``confirm_quit`` first. Three kinds of modal still lost work that way:

* Modals that REFUSE to close while an operation runs -- the fork dialog while
  it commits, the capture-policy and trace-privacy dialogs while they apply,
  the project-skills offer mid-import. Escape is refused there, but they had
  no ``confirm_quit``, so Ctrl+Q quit straight past the operation. Now Ctrl+Q
  says the dialog is still working and stays; once the operation settles, the
  next Ctrl+Q quits as usual. Pressed again while it still runs, Ctrl+Q asks
  whether to quit anyway, so an operation that never settles (a stuck Watchlists
  bulk-sources batch, driven here) cannot make the app impossible to quit.
* The generated video's Save-to-disk picker and its Replace confirmation.
  The capacity choice that asks "Discard generated video and quit?" is
  already closed by then, so the quit walk found no hook and the video was
  lost without a word. Both now ask the same question.
* The profile interview's Leave > Discard. While the discard ran, Ctrl+Q
  still asked "Discard interview and quit?"; a discard finishing under that
  prompt had its close refused (the interview was covered) and never
  retried, leaving the screen open and busy over a discarded draft.

Every test drives the real ``TldwCli`` and presses the real key. The
irreversible shutdown is replaced by a recorder, as in
``test_app_quit_under_modal.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import threading

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_app_quit_under_modal import (
    _dialogs_titled,
    _HooklessModal,
    _mounted_console,
    _quit_dialogs,
    _until,
)
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: What the walk says when a modal is still working (TASK-33622.15).
_STILL_WORKING = "Quit again once it finishes."


def _recording_cleanup(app, monkeypatch) -> list[bool]:
    """Replace the irreversible shutdown with a recorder; return its record."""
    cleanups: list[bool] = []

    async def _record_cleanup() -> None:
        cleanups.append(True)

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_cleanup)
    return cleanups


def _still_working_notices(app) -> list[str]:
    return [
        note.message for note in app._notifications if _STILL_WORKING in note.message
    ]


async def _click_when_shown(app, pilot, selector: str) -> bool:
    """Click ``selector`` on the top screen once it is laid out.

    A just-pushed prompt is on the stack before its buttons are placed, and
    Pilot aims at the widget's region, which is empty until then: a click
    made that early lands on the backdrop and answers nothing.

    Args:
        app: The running app.
        pilot: Its pilot.
        selector: The button to click on the top screen.

    Returns:
        Pilot's answer: whether the click landed on that widget.
    """

    def shown() -> bool:
        found = app.screen.query(selector)
        return bool(found) and found.first().region.area > 0

    await _until(pilot, shown, f"{selector} to be shown", timeout=5.0)
    return await pilot.click(selector)


async def _assert_ctrl_q_waits_for(app, pilot, modal, cleanups, what: str) -> None:
    """Ctrl+Q over ``modal`` mid-operation stays, and says it is still working.

    Args:
        app: The running app.
        pilot: Its pilot.
        modal: The modal whose operation is in flight.
        cleanups: Filled by the stand-in for the irreversible shutdown.
        what: Words the still-working notice must name.
    """
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: bool(_still_working_notices(app)) or bool(cleanups),
        f"Ctrl+Q over {type(modal).__name__} mid-operation to answer",
        timeout=5.0,
    )
    assert cleanups == [], (
        f"Ctrl+Q quit straight past {type(modal).__name__} while its "
        "operation was still in flight"
    )
    notices = _still_working_notices(app)
    assert len(notices) == 1 and what in notices[0].lower(), notices
    await _until(
        pilot,
        lambda: app._quit_in_progress is False,
        "the quit guard to clear after the still-working answer",
        timeout=5.0,
    )
    assert app.screen is modal
    assert not _quit_dialogs(app)
    assert not [
        screen for screen in app.screen_stack if isinstance(screen, ConfirmationDialog)
    ], "a still-working modal must not raise a quit prompt"
    assert app._shutting_down is False
    assert app.is_running


async def _assert_ctrl_q_quits(app, pilot, cleanups, what: str) -> None:
    await pilot.press("ctrl+q")
    await _until(pilot, lambda: cleanups == [True], what, timeout=5.0)


# --- Modals that refuse to close mid-operation (AC #1) -----------------------


async def test_ctrl_q_waits_for_a_fork_that_is_committing(monkeypatch):
    """Escape is refused while a fork commits; Ctrl+Q must not quit past it."""
    from Tests.UI.test_console_fork_chat_modal import _summary
    from tldw_chatbook.Widgets.Console.console_fork_chat_modal import (
        ConsoleForkChatModal,
    )

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        modal = ConsoleForkChatModal(_summary(), on_submit=lambda _result: None)
        await app.push_screen(modal)
        await _until(pilot, lambda: app.screen is modal, "the fork dialog")
        modal.show_validating()
        modal.show_committing()
        await pilot.pause()
        assert modal.state == "committing"

        await _assert_ctrl_q_waits_for(app, pilot, modal, cleanups, "fork")

        # The commit lands: the fork exists durably, so quitting loses nothing.
        modal.show_created_not_opened(
            title="Forked chat", identity="conversation-1", detail="Fork created."
        )
        await pilot.pause()
        await _assert_ctrl_q_quits(
            app, pilot, cleanups, "Ctrl+Q to quit once the fork was created"
        )


async def test_ctrl_q_waits_for_a_capture_policy_change_being_applied(monkeypatch):
    """The capture-policy dialog refuses Escape while it applies."""
    from Tests.UI.test_console_capture_policy_dialog import _PolicyHost, _snapshot
    from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
        ConsoleCapturePolicyDialog,
    )

    host = _PolicyHost(_snapshot())
    bindings = host.bindings()
    release = asyncio.Event()
    applying = asyncio.Event()

    async def held_apply(detail, revision):
        applying.set()
        await release.wait()
        return await bindings.apply_conversation(detail, revision)

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        dialog = ConsoleCapturePolicyDialog(
            replace(bindings, apply_conversation=held_apply)
        )
        await app.push_screen(dialog)
        await _until(pilot, lambda: app.screen is dialog, "the capture dialog")
        await pilot.pause(0.2)
        await pilot.click("#capture-policy-apply")
        await _until(pilot, applying.is_set, "the apply to start")
        assert dialog._applying

        await _assert_ctrl_q_waits_for(app, pilot, dialog, cleanups, "capture policy")

        release.set()
        await _until(pilot, lambda: not dialog._applying, "the apply to settle")
        assert dialog.status_text == "Saved and active"
        await _assert_ctrl_q_quits(
            app, pilot, cleanups, "Ctrl+Q to quit once the change was applied"
        )


async def test_ctrl_q_waits_for_a_trace_privacy_change_being_applied(monkeypatch):
    """The trace-privacy dialog refuses Escape while it applies."""
    from Tests.UI.test_console_capture_policy_dialog import _PolicyHost, _snapshot
    from tldw_chatbook.Chat.console_chat_controller import (
        CapturePolicyMutationResult,
        CapturePolicyMutationStatus,
    )
    from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
        ConsoleTracePrivacyDialog,
    )

    snapshot = _snapshot()
    release = asyncio.Event()
    applying = asyncio.Event()

    async def held_privacy(capture, pii, revision):
        applying.set()
        await release.wait()
        return CapturePolicyMutationResult(
            CapturePolicyMutationStatus.APPLIED,
            replace(snapshot, policy_revision=revision + 1),
            False,
            None,
        )

    bindings = replace(
        _PolicyHost(snapshot).bindings(), apply_conversation_privacy=held_privacy
    )
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        dialog = ConsoleTracePrivacyDialog(bindings)
        await app.push_screen(dialog)
        await _until(pilot, lambda: app.screen is dialog, "the trace dialog")
        await pilot.pause(0.2)
        await pilot.click("#trace-privacy-apply")
        await _until(pilot, applying.is_set, "the apply to start")
        assert dialog._applying

        await _assert_ctrl_q_waits_for(app, pilot, dialog, cleanups, "trace privacy")

        release.set()
        await _until(pilot, lambda: not dialog._applying, "the apply to settle")
        assert dialog.status_text == "Saved and active"
        await _assert_ctrl_q_quits(
            app, pilot, cleanups, "Ctrl+Q to quit once the change was applied"
        )


async def test_ctrl_q_waits_for_a_project_skills_import(monkeypatch, tmp_path):
    """The project-skills offer refuses every close while an import runs.

    Quitting mid-import would leave a half-imported skill with no way for
    the user to know (the modal's own TASK-17964 rationale).
    """
    from Tests.Skills.test_project_skills_import_modal import _discovery
    from tldw_chatbook.Widgets.project_skills_import_modal import (
        ProjectSkillsImportModal,
    )

    release = asyncio.Event()
    importing = asyncio.Event()
    imported: list[str] = []

    async def held_importer(entry) -> None:
        importing.set()
        await release.wait()
        imported.append(entry.name)

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        modal = ProjectSkillsImportModal(
            discovery=_discovery(tmp_path),
            installed_names=frozenset(),
            importer=held_importer,
        )
        await app.push_screen(modal)
        await _until(pilot, lambda: app.screen is modal, "the skills offer")
        await pilot.pause(0.2)
        await pilot.click("#project-skills-import")
        await _until(pilot, importing.is_set, "the import to start")
        assert modal._import_in_flight()

        await _assert_ctrl_q_waits_for(app, pilot, modal, cleanups, "skills")
        assert imported == []

        release.set()
        await _until(
            pilot, lambda: not modal._import_in_flight(), "the import to settle"
        )
        assert imported == ["alpha-skill", "beta-skill"]
        await _assert_ctrl_q_quits(
            app, pilot, cleanups, "Ctrl+Q to quit once the import finished"
        )


async def test_a_stuck_operation_still_lets_a_repeated_ctrl_q_quit(monkeypatch):
    """A refusal whose flag never clears must not make the app unquittable.

    BulkSourcesModal refuses Cancel and Escape until its owner answers the
    batch, and its owner skips the answer when the modal is covered, so the
    flag can stay set for good. Ctrl+Q was then the only way out; the
    still-working refusal must not close it. Here nothing answers the batch
    at all. The first Ctrl+Q says it is still working; the next asks, Wait
    keeps the dialog, and Quit anyway quits.
    """
    from textual.widgets import TextArea

    from tldw_chatbook.UI.Watchlists_Modules.bulk_sources_modal import (
        BulkSourcesModal,
    )

    title = "Quit while still working?"
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        # No owner: the batch request reaches the app, which never answers.
        modal = BulkSourcesModal()
        await app.push_screen(modal)
        await _until(pilot, lambda: app.screen is modal, "the bulk sources dialog")
        await pilot.pause(0.2)
        modal.query_one(
            "#bulk-sources-draft", TextArea
        ).text = "https://example.com/feed.xml"
        modal.query_one("#bulk-sources-create", Button).press()
        await _until(pilot, lambda: modal._batch_posted, "the batch to be posted")

        await _assert_ctrl_q_waits_for(app, pilot, modal, cleanups, "sources")
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert app.screen is modal, "the stuck dialog refuses Escape too"

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, title)) or bool(cleanups),
            "the repeated Ctrl+Q to ask before quitting past the operation",
            timeout=5.0,
        )
        assert cleanups == [], "a repeated Ctrl+Q quit without asking"
        assert app.screen is _dialogs_titled(app, title)[0]
        assert len(_still_working_notices(app)) == 1

        # Wait: back to the dialog, nothing quit.
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is modal, "Wait to restore the dialog")
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Wait",
        )
        assert cleanups == []

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, title)),
            "Ctrl+Q to ask again",
            timeout=5.0,
        )
        await _click_when_shown(app, pilot, "#confirm-button")
        await _until(
            pilot,
            lambda: cleanups == [True],
            "Quit anyway to reach the approved shutdown",
            timeout=5.0,
        )


async def test_a_fork_that_finishes_under_the_quit_question_closes_after_wait(
    monkeypatch,
):
    """Wait after the fork finished under the question returns to the chat.

    From the second Ctrl+Q on, "Quit while still working?" covers the fork
    dialog. The controller closes the dialog once the fork is open
    (``close_after_success``), but only the top screen may be dismissed
    (ADR-031), so that close was refused while the question was up and never
    tried again: after Wait the dialog sat on "Forking..." with every button
    disabled and Escape refused, and quitting was the only way out.
    """
    from Tests.UI.test_console_fork_chat_modal import _summary
    from tldw_chatbook.Widgets.Console.console_fork_chat_modal import (
        ConsoleForkChatModal,
    )

    title = "Quit while still working?"
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        modal = ConsoleForkChatModal(_summary(), on_submit=lambda _result: None)
        await app.push_screen(modal)
        await _until(pilot, lambda: app.screen is modal, "the fork dialog")
        modal.show_validating()
        modal.show_committing()
        await pilot.pause()

        await _assert_ctrl_q_waits_for(app, pilot, modal, cleanups, "fork")
        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, title)),
            "the repeated Ctrl+Q to ask before quitting past the fork",
            timeout=5.0,
        )
        question = _dialogs_titled(app, title)[0]
        assert app.screen is question

        # The fork opens while the question is up: the controller's close.
        modal.close_after_success()
        await pilot.pause(0.2)
        assert app.screen is question, "the dialog's close must not pop the question"
        assert modal in app.screen_stack, "covered, the dialog cannot close yet"

        await _click_when_shown(app, pilot, "#cancel-button")  # Wait
        await _until(
            pilot,
            lambda: modal not in app.screen_stack,
            "the finished fork dialog to close once Wait uncovered it",
        )
        assert app.screen is console
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Wait",
        )
        assert cleanups == []
        assert app._exception is None


# --- The generated video's Save-to-disk picker (AC #2) ------------------------


async def _video_waiting_in_the_save_picker(app, pilot, console, artifact):
    """Run the real pending-video resolver up to its Save-to-disk picker."""
    from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
        ConsoleVideoCapacityModal,
    )
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

    console.run_worker(
        console._video._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        ),
        exclusive=False,
        exit_on_error=False,
    )
    await _until(
        pilot,
        lambda: isinstance(app.screen, ConsoleVideoCapacityModal),
        "the generated-video choice",
    )
    capacity = app.screen
    await pilot.pause(0.2)
    capacity.query_one("#video-capacity-save", Button).press()
    await _until(
        pilot,
        lambda: isinstance(app.screen, EnhancedFileSave),
        "Save to disk to open the picker",
    )
    picker = app.screen
    await pilot.pause(0.2)
    return picker


async def _assert_ctrl_q_asks_before_discarding_the_video(
    app, pilot, screen, console, artifact, cleanups
) -> None:
    title = "Discard generated video and quit?"
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: bool(_dialogs_titled(app, title)) or bool(cleanups),
        f"Ctrl+Q over {type(screen).__name__} to ask {title!r}",
        timeout=5.0,
    )
    assert cleanups == [], (
        f"Ctrl+Q quit past {type(screen).__name__} and lost the generated video"
    )
    assert app.screen is _dialogs_titled(app, title)[0]

    # Stay: the picker and the video are both still there.
    await pilot.press("escape")
    await _until(pilot, lambda: app.screen is screen, "Stay to restore the picker")
    await _until(
        pilot,
        lambda: app._quit_in_progress is False,
        "the quit guard to clear after Stay",
    )
    assert console._video._owns_pending_console_video(artifact)
    assert not artifact.stream.closed
    assert cleanups == []

    # The confirm: the user chose to lose it, so the quit runs.
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: bool(_dialogs_titled(app, title)),
        "the second Ctrl+Q to ask again",
        timeout=5.0,
    )
    await _click_when_shown(app, pilot, "#confirm-button")
    await _until(
        pilot,
        lambda: cleanups == [True],
        "the confirm to reach the approved shutdown",
        timeout=5.0,
    )


async def test_ctrl_q_over_the_video_save_picker_asks_before_discarding_it(
    monkeypatch,
):
    from Tests.Chat.test_console_video_capacity import _artifact

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    artifact = _artifact(message_id="quit-under-picker")
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        picker = await _video_waiting_in_the_save_picker(app, pilot, console, artifact)
        await _assert_ctrl_q_asks_before_discarding_the_video(
            app, pilot, picker, console, artifact, cleanups
        )


async def test_ctrl_q_over_the_video_replace_confirmation_asks_first(
    monkeypatch, tmp_path
):
    from Tests.Chat.test_console_video_capacity import _artifact

    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"keep me")
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    artifact = _artifact(message_id="quit-under-replace")
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        picker = await _video_waiting_in_the_save_picker(app, pilot, console, artifact)
        picker.dismiss(existing)
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, "Replace existing file?")),
            "an existing destination to ask before replacing it",
        )
        confirmation = _dialogs_titled(app, "Replace existing file?")[0]
        await pilot.pause(0.2)
        await _assert_ctrl_q_asks_before_discarding_the_video(
            app, pilot, confirmation, console, artifact, cleanups
        )
        assert existing.read_bytes() == b"keep me"


# --- The interview discard race (AC #3) ---------------------------------------


@pytest.fixture
def held_discard_coordinator():
    """A memory-only interview whose discard blocks until the test releases it.

    Released at teardown too, so a failing test never strands the thread.
    """
    from Tests.UI.test_profile_interview_screen import _Coordinator, _session_base

    class _HeldDiscardCoordinator(_Coordinator):
        def __init__(self, session) -> None:
            super().__init__(session)
            self.discard_started = threading.Event()
            self.release_discard = threading.Event()
            #: Raised by the held discard once released, when set.
            self.discard_error: Exception | None = None

        def discard(self, session_id):
            self.discard_started.set()
            self.release_discard.wait(10.0)
            if self.discard_error is not None:
                raise self.discard_error
            super().discard(session_id)

    coordinator = _HeldDiscardCoordinator(
        replace(_session_base(), draft_is_memory_only=True)
    )
    yield coordinator
    coordinator.release_discard.set()


async def _interview_discarding(app, pilot, coordinator, results):
    """Open a memory-only interview and choose Leave > Discard, held mid-flight."""
    from tldw_chatbook.UI.Screens.profile_interview_screen import (
        ProfileInterviewCancelModal,
        ProfileInterviewScreen,
    )

    screen = ProfileInterviewScreen(
        coordinator, kind="personal", scope_id="scope-global", mode="fixed"
    )
    await app.push_screen(screen, callback=results.append)
    await _until(
        pilot,
        lambda: (
            app.screen is screen and screen._session is not None and not screen._busy
        ),
        "the interview to load",
    )
    await pilot.click("#profile-interview-cancel")
    await _until(
        pilot,
        lambda: isinstance(app.screen, ProfileInterviewCancelModal),
        "the Leave interview prompt",
    )
    await pilot.pause(0.2)
    await pilot.click("#profile-interview-cancel-discard")
    await _until(
        pilot,
        lambda: coordinator.discard_started.is_set(),
        "the chosen discard to start",
    )
    assert app.screen is screen
    return screen


async def test_ctrl_q_while_the_interview_discard_runs_waits_then_quits(
    monkeypatch, held_discard_coordinator
):
    """One end state: the chosen discard finishes, the screen closes, quit runs.

    Before the fix Ctrl+Q asked "Discard interview and quit?" about an
    interview the user had already chosen to discard.
    """
    from tldw_chatbook.UI.Screens.profile_interview_screen import (
        ProfileInterviewResult,
    )

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    coordinator = held_discard_coordinator
    results: list[object] = []
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        screen = await _interview_discarding(app, pilot, coordinator, results)

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: (
                bool(_still_working_notices(app))
                or bool(_dialogs_titled(app, "Discard interview and quit?"))
                or bool(cleanups)
            ),
            "Ctrl+Q during the discard to answer",
            timeout=5.0,
        )
        assert not _dialogs_titled(app, "Discard interview and quit?"), (
            "Ctrl+Q asked to discard an interview that is already being discarded"
        )
        assert cleanups == []
        notices = _still_working_notices(app)
        assert len(notices) == 1 and "interview" in notices[0], notices
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear",
            timeout=5.0,
        )

        coordinator.release_discard.set()
        await _until(
            pilot,
            lambda: screen not in app.screen_stack and bool(results),
            "the finished discard to close the interview",
        )
        assert results == [ProfileInterviewResult("discarded", (), None)]
        await _assert_ctrl_q_quits(
            app, pilot, cleanups, "Ctrl+Q to quit once the interview closed"
        )


async def test_a_failed_interview_discard_lets_ctrl_q_ask_again(
    monkeypatch, held_discard_coordinator
):
    """A discard that fails leaves the interview open, so quitting asks again.

    While the discard runs Ctrl+Q waits for it. When it fails the interview
    is back in front of the user with its answers, nothing is running, and
    quitting would lose a memory-only interview -- so Ctrl+Q must ask
    "Discard interview and quit?" again, not keep saying it is still being
    discarded.
    """
    title = "Discard interview and quit?"
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    coordinator = held_discard_coordinator
    coordinator.discard_error = RuntimeError("discard failed")
    results: list[object] = []
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        screen = await _interview_discarding(app, pilot, coordinator, results)
        await _assert_ctrl_q_waits_for(app, pilot, screen, cleanups, "interview")

        coordinator.release_discard.set()
        await _until(
            pilot,
            lambda: not screen._busy,
            "the failed discard to hand the interview back",
        )
        assert app.screen is screen and results == []

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: (
                bool(_dialogs_titled(app, title))
                or bool(_dialogs_titled(app, "Quit while still working?"))
                or len(_still_working_notices(app)) > 1
                or bool(cleanups)
            ),
            "Ctrl+Q after the failed discard to answer",
            timeout=5.0,
        )
        assert _dialogs_titled(app, title), (
            "Ctrl+Q still treats a failed discard as running"
        )
        assert cleanups == []

        # Continue interview: the interview and its answers stay.
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is screen, "Continue interview")
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Continue interview",
        )
        assert cleanups == [] and results == []


async def test_an_interview_discard_finishing_while_covered_still_closes_it(
    monkeypatch, held_discard_coordinator
):
    """A refused close is kept and finished once the interview is on top again.

    The discard completes while another screen covers the interview, so its
    close is refused (only the top screen may be dismissed, ADR-031). Before
    the fix that close was never retried: the interview stayed open and busy
    over a discarded draft.
    """
    from tldw_chatbook.UI.Screens.profile_interview_screen import (
        ProfileInterviewResult,
    )

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    _recording_cleanup(app, monkeypatch)
    coordinator = held_discard_coordinator
    results: list[object] = []
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        screen = await _interview_discarding(app, pilot, coordinator, results)

        cover = _HooklessModal()
        await app.push_screen(cover)
        await _until(pilot, lambda: app.screen is cover, "the covering modal")
        coordinator.release_discard.set()
        await _until(
            pilot,
            lambda: any(call[0] == "discard" for call in coordinator.calls),
            "the discard to finish under the cover",
        )
        await pilot.pause(0.3)
        assert screen in app.screen_stack, "covered, the interview cannot close yet"
        assert results == []

        cover.dismiss(None)
        await _until(
            pilot,
            lambda: screen not in app.screen_stack and bool(results),
            "the interview to finish its close once uncovered",
        )
        assert results == [ProfileInterviewResult("discarded", (), None)]
        assert app.screen is console
        assert app._exception is None


async def test_an_interview_discard_finishing_under_the_quit_question_closes_after_wait(
    monkeypatch, held_discard_coordinator
):
    """The cover is Ctrl+Q's own "Quit while still working?" this time.

    The interview keeps a close refused while covered (``_close_with``) and
    finishes it on ScreenResume, so Wait returns to the Console, not to an
    interview left open and busy over a discarded draft.
    """
    from tldw_chatbook.UI.Screens.profile_interview_screen import (
        ProfileInterviewResult,
    )

    title = "Quit while still working?"
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups = _recording_cleanup(app, monkeypatch)
    coordinator = held_discard_coordinator
    results: list[object] = []
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        screen = await _interview_discarding(app, pilot, coordinator, results)
        await _assert_ctrl_q_waits_for(app, pilot, screen, cleanups, "interview")

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, title)) or bool(cleanups),
            "the repeated Ctrl+Q to ask before quitting past the discard",
            timeout=5.0,
        )
        assert cleanups == []
        assert app.screen is _dialogs_titled(app, title)[0]

        coordinator.release_discard.set()
        await _until(
            pilot,
            lambda: any(call[0] == "discard" for call in coordinator.calls),
            "the discard to finish under the question",
        )
        await pilot.pause(0.3)
        assert screen in app.screen_stack, "covered, the interview cannot close yet"
        assert results == []

        await _click_when_shown(app, pilot, "#cancel-button")  # Wait
        await _until(
            pilot,
            lambda: screen not in app.screen_stack and bool(results),
            "the interview to close once Wait uncovered it",
        )
        assert results == [ProfileInterviewResult("discarded", (), None)]
        assert app.screen is console
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Wait",
        )
        assert cleanups == []
        assert app._exception is None
