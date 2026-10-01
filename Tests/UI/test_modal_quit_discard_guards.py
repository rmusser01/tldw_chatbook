"""Dirty-guarded modals ask before Ctrl+Q discards their edits (TASK-33622.10).

Ctrl+Q became a priority binding, so the quit flow now starts while a modal
is open and asks every screen from that modal down to the destination for
``confirm_quit``. Eight modals guard their OWN close with a discard prompt
when closing would lose something -- before the fix, Ctrl+Q was swallowed
under them, which protected that work by accident; after it, quitting went
straight past the guard. Each of them now answers ``confirm_quit`` through
the shared quit prompt, and only when its close guard would fire.

* Five ask the plain "Discard changes and quit?" for unsaved edits: the
  reminder and automation forms, Console Library access, prompts and the
  prompt queue.
* Console Settings (a review follow-up) asks that too for unapplied edits,
  and also stops at its two side-effect close guards -- an undoable memory
  reset and a running compaction. Quitting KEEPS the reset, so when only
  those apply the prompt is worded neutrally, never "Discard".
* The generated-video choice and a memory-only profile interview (a later
  review follow-up) ask in their own words: the video "cannot be
  recovered", and the interview "exists only in memory".

The real-app proofs (Ctrl+Q over a dirty ReminderForm, a generated-video
choice and a memory-only interview on a live Console) live in
``test_app_quit_under_modal.py``. These pin the shared prompt and each
modal's wiring: mounted, driven by a priority Ctrl+Q that runs the app's real
quit walk where the modal mounts standalone, stubbed where it needs the
Console's heavy fixtures.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.binding import Binding
from textual.widgets import Button, Input

# Imported at collection, under the session profile: the UI conftest's
# autouse fixture otherwise imports the app lazily under each test's own
# profile, and run alone this file then errored at setup on every test
# (RecoveryRequired: raw_source_selection_changed).
import tldw_chatbook.app  # noqa: F401
import tldw_chatbook.Widgets.confirmation_dialog as confirmation_dialog
from Tests.Chat.test_console_session_settings import (
    _settings_close_modal,
    _SettingsCloseHarness,
)
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.test_profile_interview_screen import (
    _Coordinator as _InterviewCoordinator,
)
from Tests.UI.test_profile_interview_screen import (
    _session_base as _interview_session,
)
from tldw_chatbook.UI.Screens.profile_interview_screen import ProfileInterviewScreen
from tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form import (
    AutomationDefinitionForm,
)
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console.console_library_access_modal import (
    ConsoleLibraryAccessModal,
)
from tldw_chatbook.Widgets.Console.console_prompt_queue_modal import (
    ConsolePromptQueueModal,
)
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsModal
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal
from tldw_chatbook.Widgets.Console.console_settings_unsaved import (
    QUIT_COMPACTION_COPY,
    QUIT_RESET_COPY,
)
from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
    ConsoleVideoCapacityModal,
)

_TITLE = "Discard changes and quit?"
_VIDEO_TITLE = "Discard generated video and quit?"
_INTERVIEW_TITLE = "Discard interview and quit?"
_MIB = 1024 * 1024


async def _until(pilot, predicate, what: str, timeout: float = 5.0) -> None:
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


class _QuitWalkHost(ConsolidatedCSSApp):
    """A host whose priority Ctrl+Q runs the app's real quit walk.

    ``TldwCli``'s quit flow asks ``confirm_quit_screens`` over
    ``quit_confirmation_screens`` before anything irreversible. This host runs
    exactly that from a priority Ctrl+Q, as the app does, and records the
    answer instead of shutting down -- so a mounted modal is driven by the
    real key, the real walk and the real prompt.
    """

    CSS_PATH = [str(BUNDLED_STYLESHEET)]
    BINDINGS = [Binding("ctrl+q", "quit_walk", "Quit", priority=True)]

    def __init__(self) -> None:
        super().__init__()
        self.walks: list[bool] = []
        self.results: list[object] = []

    def action_quit_walk(self) -> None:
        self.run_worker(self._walk(), group="quit-walk")

    async def _walk(self) -> None:
        screens = confirmation_dialog.quit_confirmation_screens(self)
        self.walks.append(await confirmation_dialog.confirm_quit_screens(screens))


async def _ctrl_q_asks(pilot, host: _QuitWalkHost, title: str) -> ConfirmationDialog:
    """Press Ctrl+Q and wait for the quit prompt titled ``title``.

    Fails fast, naming the answer, if the walk finishes without asking.
    """
    answered = len(host.walks)
    await pilot.press("ctrl+q")

    def _shown() -> bool:
        if len(host.walks) > answered:
            raise AssertionError(
                f"the quit walk answered {host.walks[-1]} without asking {title!r}"
            )
        screen = host.screen
        return isinstance(screen, ConfirmationDialog) and screen.title == title

    await _until(pilot, _shown, f"the {title!r} prompt")
    return host.screen


@pytest.mark.asyncio
async def test_discard_and_quit_prompt_keep_editing_stays_and_discard_quits():
    """The shared prompt: Keep editing (Escape) answers False, Discard True."""
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        base = app.screen
        message = "You have unsaved changes in this form."

        asking = app.run_worker(
            confirmation_dialog.confirm_quit_discarding_edits(base, message),
            exit_on_error=False,
        )
        await _until(
            pilot,
            lambda: isinstance(app.screen, ConfirmationDialog),
            "the discard-and-quit prompt",
        )
        prompt = app.screen
        assert prompt.title == _TITLE
        assert prompt.message == message
        assert prompt.cancel_label == "Keep editing"
        assert prompt.confirm_label == "Discard and quit"
        await pilot.press("escape")
        await asking.wait()
        assert asking.result is False
        assert app.screen is base

        asking = app.run_worker(
            confirmation_dialog.confirm_quit_discarding_edits(base, message),
            exit_on_error=False,
        )
        await _until(
            pilot,
            lambda: isinstance(app.screen, ConfirmationDialog),
            "the discard-and-quit prompt (second ask)",
        )
        await pilot.click("#confirm-button")
        await asking.wait()
        assert asking.result is True
        assert app.screen is base


def _reminder_form(dirty: bool):
    form = ReminderForm.__new__(ReminderForm)
    form._dirty = dirty
    return form


def _automation_form(dirty: bool):
    form = AutomationDefinitionForm.__new__(AutomationDefinitionForm)
    form._dirty = dirty
    return form


def _library_access_modal(dirty: bool):
    modal = ConsoleLibraryAccessModal.__new__(ConsoleLibraryAccessModal)
    modal._dirty = dirty
    return modal


def _prompts_modal(dirty: bool, mode: str = "edit", *, applying: bool = False):
    modal = ConsolePromptsModal.__new__(ConsolePromptsModal)
    modal.state = SimpleNamespace(mode=mode, dirty=dirty)
    modal._apply_in_progress = applying
    return modal


def _prompt_queue_modal(dirty: bool):
    modal = ConsolePromptQueueModal.__new__(ConsolePromptQueueModal)
    modal.has_unsaved_edit = lambda: dirty
    return modal


def _console_settings_modal(
    dirty: bool, *, reset: bool = False, compacting: bool = False
):
    """A Console Settings modal holding exactly the given close-guard states."""
    modal = ConsoleSettingsModal.__new__(ConsoleSettingsModal)
    modal._memory_reset_token = ("memory-revision", 1) if reset else None
    modal._compaction_is_active = lambda: compacting
    modal._unsaved_field_labels = lambda: ("Temperature",) if dirty else ()
    return modal


_GUARDED_MODALS = [
    pytest.param(_reminder_form, id="reminder-form"),
    pytest.param(_automation_form, id="automation-definition-form"),
    pytest.param(_library_access_modal, id="console-library-access"),
    pytest.param(_prompts_modal, id="console-prompts"),
    pytest.param(_prompt_queue_modal, id="console-prompt-queue"),
    pytest.param(_console_settings_modal, id="console-settings"),
]


@pytest.fixture
def asked(monkeypatch):
    """Record every quit prompt instead of pushing one.

    Each entry is ``(screen, message, copy)``; ``copy`` holds the title and
    button overrides, empty for the default "Discard changes and quit?".
    """
    calls: list[tuple[object, str, dict[str, str]]] = []

    async def _keep_editing(screen, message: str, **copy: str) -> bool:
        calls.append((screen, message, copy))
        return False

    monkeypatch.setattr(
        confirmation_dialog, "confirm_quit_discarding_edits", _keep_editing
    )
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("build", _GUARDED_MODALS)
async def test_dirty_guarded_modal_asks_before_quit_and_keep_editing_vetoes(
    build, asked
):
    modal = build(True)

    assert await modal.confirm_quit() is False
    assert len(asked) == 1
    screen, message, _copy = asked[0]
    assert screen is modal
    assert message.strip(), "the prompt must say what would be lost"


@pytest.mark.asyncio
@pytest.mark.parametrize("build", _GUARDED_MODALS)
async def test_clean_guarded_modal_lets_quit_through_without_asking(build, asked):
    modal = build(False)

    assert await modal.confirm_quit() is True
    assert asked == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["browse", "improve"])
async def test_console_prompts_modal_asks_only_where_its_close_guard_fires(mode, asked):
    """The prompts modal's close guard covers edit/recipe/draft_edit only; a
    dirty flag left over in another mode loses nothing on close, so quitting
    does not ask either."""
    modal = _prompts_modal(True, mode=mode)

    assert await modal.confirm_quit() is True
    assert asked == []


@pytest.mark.asyncio
async def test_console_prompts_modal_refuses_quit_while_an_apply_is_in_flight(
    asked,
):
    """Close is refused while an apply is mid-flight, so quitting is too.

    The apply writes the reviewed prompt into the live Console; quitting
    under it would drop that write with no prompt. Nothing is dirty here, so
    only the apply guard can stop the quit -- and it says why.
    """
    modal = _prompts_modal(False, mode="improve", applying=True)
    notices: list[str] = []
    modal.notify = lambda message, **_kwargs: notices.append(str(message))

    assert await modal.confirm_quit() is False
    assert asked == [], "an apply in flight is not a discard prompt"
    assert len(notices) == 1 and "applying" in notices[0].lower()


@pytest.mark.asyncio
async def test_console_settings_unsaved_quit_prompt_names_the_edited_fields(asked):
    modal = _console_settings_modal(True)

    assert await modal.confirm_quit() is False
    [(_screen, message, _copy)] = asked
    assert "Temperature" in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("guard", "named"),
    [
        (dict(reset=True), "memory was reset"),
        (dict(compacting=True), "compaction"),
    ],
    ids=["memory-reset", "compaction"],
)
async def test_console_settings_side_effect_guards_ask_before_quit(guard, named, asked):
    """Close stops at an undoable memory reset or a running compaction; so
    does quitting, even with no field edited."""
    modal = _console_settings_modal(False, **guard)

    assert await modal.confirm_quit() is False
    [(_screen, message, _copy)] = asked
    assert named in message.lower()


# --- Console Settings: the side-effect guards are worded neutrally -----------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("guard", "lines"),
    [
        (dict(reset=True), [QUIT_RESET_COPY]),
        (dict(compacting=True), [QUIT_COMPACTION_COPY]),
        (dict(reset=True, compacting=True), [QUIT_RESET_COPY, QUIT_COMPACTION_COPY]),
    ],
    ids=["memory-reset", "compaction", "both"],
)
async def test_console_settings_side_effect_quit_prompt_never_says_discard(
    guard, lines, asked
):
    """Review follow-up: quitting KEEPS a memory reset (the body says so), and
    a running compaction holds nothing the user typed. A "Discard changes and
    quit?" title or a "Discard and quit" button would read as undoing the
    reset, so with no edit to discard the prompt is worded neutrally."""
    modal = _console_settings_modal(False, **guard)

    assert await modal.confirm_quit() is False
    [(_screen, message, copy)] = asked
    assert message.splitlines() == lines
    assert copy == {
        "title": "Quit now?",
        "confirm_label": "Quit anyway",
        "cancel_label": "Stay",
    }
    assert "discard" not in " ".join(copy.values()).lower()


@pytest.mark.asyncio
async def test_console_settings_quit_prompt_with_an_edit_still_says_discard(asked):
    """An unapplied edit IS discarded by quitting, so once one applies the
    default discard wording returns, even alongside a reset."""
    modal = _console_settings_modal(True, reset=True)

    assert await modal.confirm_quit() is False
    [(_screen, message, copy)] = asked
    assert message.splitlines()[0] == QUIT_RESET_COPY
    assert "Temperature" in message.splitlines()[1]
    assert copy == {}


async def _settled_settings(pilot, modal) -> None:
    """Wait until the modal has recorded what an unedited draft looks like."""
    await _until(
        pilot,
        lambda: getattr(modal, "_unsaved_baseline", None) is not None,
        "Console Settings to record its unedited baseline",
    )


@pytest.mark.asyncio
async def test_mounted_console_settings_reset_quit_prompt_is_neutral_and_stay_keeps_it():
    """The real modal, a real reset, the real prompt: neutral words, and
    Stay (Escape) keeps Settings open with the reset, and its Undo, intact."""
    app = _SettingsCloseHarness()
    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: True,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        await _settled_settings(pilot, modal)
        modal.query_one("#console-context-reset-current", Button).press()
        await _until(
            pilot,
            lambda: modal._memory_reset_token is not None,
            "the memory reset to take effect",
        )

        asking = app.run_worker(modal.confirm_quit(), exit_on_error=False)
        await _until(
            pilot,
            lambda: isinstance(app.screen, ConfirmationDialog),
            "the Console Settings quit prompt",
        )
        prompt = app.screen
        assert prompt.title == "Quit now?"
        assert prompt.confirm_label == "Quit anyway"
        assert prompt.cancel_label == "Stay"
        assert prompt.message == QUIT_RESET_COPY

        await pilot.press("escape")
        await asking.wait()
        assert asking.result is False
        assert app.screen is modal
        assert modal._memory_reset_token is not None
        assert app.results == []


# --- The generated-video choice ----------------------------------------------


def _video_choice() -> ConsoleVideoCapacityModal:
    return ConsoleVideoCapacityModal(
        reason="over_capacity", size_bytes=300 * _MIB, max_bytes=256 * _MIB
    )


@pytest.mark.asyncio
async def test_ctrl_q_over_the_generated_video_choice_asks_and_stay_keeps_it():
    """Escape on the video choice asks before discarding a result that
    "cannot be recovered" (``_perform_safe_cancel``). The video only exists
    while this choice is open, so Ctrl+Q always asks the same: Stay keeps the
    choice open with nothing decided, Discard and quit lets the quit run."""
    host = _QuitWalkHost()
    async with host.run_test(size=(100, 30)) as pilot:
        modal = _video_choice()
        await host.push_screen(modal, callback=host.results.append)
        await _until(pilot, lambda: host.screen is modal, "the video choice")

        prompt = await _ctrl_q_asks(pilot, host, _VIDEO_TITLE)
        assert "cannot be recovered" in prompt.message
        assert prompt.confirm_label == "Discard and quit"
        assert prompt.cancel_label == "Stay"
        assert modal in host.screen_stack

        await pilot.press("escape")
        await _until(pilot, lambda: host.walks == [False], "Stay to veto the quit")
        assert host.screen is modal
        assert host.results == [], "Stay must not decide the video's fate"

        await _ctrl_q_asks(pilot, host, _VIDEO_TITLE)
        await pilot.click("#confirm-button")
        await _until(
            pilot,
            lambda: host.walks == [False, True],
            "Discard and quit to let the quit run",
        )


@pytest.mark.asyncio
async def test_ctrl_q_over_the_video_choices_own_discard_prompt_still_asks():
    """With the choice's own "Discard generated video?" prompt open on top,
    the walk passes that hook-less prompt and still reaches the choice."""
    host = _QuitWalkHost()
    async with host.run_test(size=(100, 30)) as pilot:
        modal = _video_choice()
        await host.push_screen(modal, callback=host.results.append)
        await _until(pilot, lambda: host.screen is modal, "the video choice")
        await pilot.press("escape")
        await _until(
            pilot,
            lambda: host.screen is not modal and modal._discard_confirmation_open,
            "the choice's own discard prompt",
        )
        own_prompt = host.screen

        await _ctrl_q_asks(pilot, host, _VIDEO_TITLE)
        await pilot.press("escape")
        await _until(pilot, lambda: host.walks == [False], "Stay to veto the quit")
        assert host.screen is own_prompt
        assert host.results == []


# --- The profile interview ---------------------------------------------------


def _interview(*, memory_only: bool, resume: bool = False, **session_changes):
    coordinator = _InterviewCoordinator(
        replace(
            _interview_session(), draft_is_memory_only=memory_only, **session_changes
        )
    )
    screen = ProfileInterviewScreen(
        coordinator,
        kind="personal",
        scope_id="scope-global",
        mode="fixed",
        session_id="session-1" if resume else None,
    )
    return coordinator, screen


async def _open_interview(pilot, host: _QuitWalkHost, screen) -> None:
    await host.push_screen(screen, callback=host.results.append)
    await _until(
        pilot,
        lambda: host.screen is screen and screen._session is not None,
        "the interview to load",
    )
    await host.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
async def test_ctrl_q_over_a_memory_only_interview_asks_and_continue_keeps_it():
    """A memory-only interview has no draft to keep: its own Leave prompt
    offers only Continue or Discard, and quitting discards it. So Ctrl+Q asks
    first; Continue interview keeps the screen, its session and a typed
    answer, and discards nothing; Discard and quit lets the quit run."""
    host = _QuitWalkHost()
    coordinator, screen = _interview(memory_only=True)
    async with host.run_test(size=(100, 30)) as pilot:
        await _open_interview(pilot, host, screen)
        answer = screen.query_one("#profile-interview-answer", Input)
        answer.focus()
        await pilot.press(*"Kim")
        assert answer.value == "Kim"
        session = screen._session

        prompt = await _ctrl_q_asks(pilot, host, _INTERVIEW_TITLE)
        assert "only in memory" in prompt.message
        assert prompt.confirm_label == "Discard and quit"
        assert prompt.cancel_label == "Continue interview"

        await pilot.press("escape")
        await _until(pilot, lambda: host.walks == [False], "Continue to veto the quit")
        assert host.screen is screen
        assert answer.value == "Kim"
        assert screen._session is session
        assert host.results == []
        assert not [call for call in coordinator.calls if call[0] == "discard"]

        await _ctrl_q_asks(pilot, host, _INTERVIEW_TITLE)
        await pilot.click("#confirm-button")
        await _until(
            pilot,
            lambda: host.walks == [False, True],
            "Discard and quit to let the quit run",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_only", "resume", "changes"),
    [
        pytest.param(False, False, {}, id="encrypted-draft-is-already-saved"),
        pytest.param(
            True,
            True,
            {
                "status": "committed",
                "question": None,
                "committed_record_ids": ("record-1",),
            },
            id="memory-only-but-already-committed",
        ),
    ],
)
async def test_ctrl_q_over_an_interview_with_nothing_to_lose_quits_without_asking(
    memory_only, resume, changes
):
    """An encrypted draft already holds every answer and stays resumable, so
    quitting is Keep draft and loses nothing; a committed interview's
    records are saved, and its own close does not ask either."""
    host = _QuitWalkHost()
    _coordinator, screen = _interview(memory_only=memory_only, resume=resume, **changes)
    async with host.run_test(size=(100, 30)) as pilot:
        await _open_interview(pilot, host, screen)

        await pilot.press("ctrl+q")
        await _until(pilot, lambda: host.walks == [True], "the quit to pass")
        assert not [
            item for item in host.screen_stack if isinstance(item, ConfirmationDialog)
        ]
        assert host.screen is screen
