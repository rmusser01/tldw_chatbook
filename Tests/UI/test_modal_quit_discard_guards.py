"""Dirty-guarded modals ask before Ctrl+Q discards their edits (TASK-33622.10).

Ctrl+Q became a priority binding, so the quit flow now starts while a modal
is open and asks every screen from that modal down to the destination for
``confirm_quit``. Six modals guard their OWN close with a discard prompt
when they hold unsaved edits -- before the fix, Ctrl+Q was swallowed under
them, which protected that work by accident; after it, quitting went
straight past the guard. Each of them now answers ``confirm_quit`` with the
shared "Discard changes and quit?" prompt, and only when its close guard
would fire. Console Settings (the sixth, a review follow-up) also asks for
its two side-effect close guards: an undoable memory reset and a running
compaction.

The real-app proof (a dirty ReminderForm over a live Console, real Ctrl+Q)
lives in ``test_app_quit_under_modal.py``; these pin the shared prompt and
the per-modal wiring without each modal's heavy fixtures.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.app import App

import tldw_chatbook.Widgets.confirmation_dialog as confirmation_dialog
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

_TITLE = "Discard changes and quit?"


async def _until(pilot, predicate, what: str, timeout: float = 5.0) -> None:
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


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
    """Record every discard-and-quit prompt instead of pushing one."""
    calls: list[tuple[object, str]] = []

    async def _keep_editing(screen, message: str) -> bool:
        calls.append((screen, message))
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
    screen, message = asked[0]
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
    [(_screen, message)] = asked
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
    [(_screen, message)] = asked
    assert named in message.lower()
