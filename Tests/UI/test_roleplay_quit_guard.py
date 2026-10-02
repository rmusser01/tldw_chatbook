"""Ctrl+Q asks before quitting past unsaved Roleplay drafts (TASK-33622.14).

Leaving the Roleplay screen with unsaved drafts asks one aggregate question
-- Save and continue / Discard and continue / Stay -- and a save that fails
offers Retry / Stay instead of dropping the drafts. Quitting is leaving too,
but ``PersonasScreen`` had no ``confirm_quit``, so the app's quit walk found
nothing to ask and Ctrl+Q quit straight past the drafts.

These drive the real ``TldwCli`` and press the real key: the quit walk, the
Roleplay screen, its editor's change tracking and its save worker are all
real. Only the character store is stubbed (so a save can be observed and
made to fail), and the irreversible shutdown is replaced by a recorder, as
the other quit-flow tests do.

The quit prompts go through the quit flow's ``await_quit_prompt`` choke point
(TASK-33622.10): a prompt that leaves the screen stack unanswered -- popped
by a covered modal closing itself -- ends the quit as Stay instead of hanging
the quit worker.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.screen import ModalScreen
from textual.widgets import Input, Static

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_personas_dictionaries import patch_character_paging
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    RoleplayDraftNavigationDialog,
    RoleplayDraftRecoveryDialog,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditCharacterRequested,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: Generous: the first test in a process pays the cold imports of the
#: Roleplay screen and its lazily mounted editor.
_SETTLE_SECONDS = 30.0
_SAVED_NAME = "Detective Sam"
_DRAFT_NAME = "Detective Sam (unsaved draft)"
_CHARACTERS = [{"id": 1, "name": _SAVED_NAME, "description": "", "version": 1}]


async def _until(pilot, predicate, what: str, timeout: float = _SETTLE_SECONDS):
    """Pump the app until ``predicate()`` holds, or fail naming ``what``."""
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


class _Store:
    """The stubbed character store: records saves and can be made to fail."""

    def __init__(self, monkeypatch, events: list[object]) -> None:
        self.events = events
        self.fail = False
        monkeypatch.setattr(
            character_handler,
            "fetch_all_characters",
            lambda: [dict(record) for record in _CHARACTERS],
        )
        monkeypatch.setattr(
            character_handler,
            "fetch_character_by_id",
            lambda character_id: next(
                dict(record)
                for record in _CHARACTERS
                if str(record["id"]) == str(character_id)
            ),
        )
        monkeypatch.setattr(character_handler, "update_character", self._update)
        patch_character_paging(monkeypatch)

    def _update(self, character_id, data) -> bool:
        if self.fail:
            self.events.append(("save failed", str(character_id)))
            return False
        self.events.append(("saved", str(character_id), data.get("name")))
        return True


def _quit_recorder(app, events: list[object]):
    """Stand in for the irreversible shutdown the approved quit runs."""

    async def _record() -> None:
        events.append("quit")

    return _record


async def _roleplay_editor(app, pilot):
    """Open Roleplay on the saved character's editor; return (screen, name)."""
    await _until(
        pilot,
        lambda: (
            type(app.screen).__name__ == "PersonasScreen"
            and app.screen.is_mounted
            and bool(app.screen.query("#personas-library-row-character-1"))
        ),
        "the Roleplay screen and its character library",
    )
    screen = app.screen
    screen.post_message(EditCharacterRequested("1"))

    def _editing() -> bool:
        editors = screen.query(PersonasCharacterEditorWidget)
        if not editors or screen._edit_mode != "edit":
            return False
        name = editors.first().query("#personas-char-editor-name")
        return bool(name) and name.first().value == _SAVED_NAME

    await _until(pilot, _editing, "the character editor to load the saved card")
    await pilot.pause(0.2)
    name = screen.query_one(PersonasCharacterEditorWidget).query_one(
        "#personas-char-editor-name", Input
    )
    assert screen.state.has_unsaved_changes is False
    return screen, name


async def _dirty(pilot, screen, name: Input) -> None:
    """Edit the name the way typing does; the editor's tracking marks it dirty."""
    name.value = _DRAFT_NAME
    await _until(
        pilot,
        lambda: screen.state.has_unsaved_changes,
        "the edit to mark the Roleplay draft unsaved",
    )


async def _ctrl_q_asks(pilot, app, events: list[object], prompt_type, what: str):
    """Press Ctrl+Q and wait for ``prompt_type``; fail if the app quit instead."""
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: isinstance(app.screen, prompt_type) or "quit" in events,
        what,
        timeout=5.0,
    )
    assert "quit" not in events, "Ctrl+Q quit straight past the unsaved Roleplay draft"
    return app.screen


def _domains(prompt) -> str:
    selector = (
        "#roleplay-draft-navigation-domains"
        if isinstance(prompt, RoleplayDraftNavigationDialog)
        else "#roleplay-draft-recovery-domains"
    )
    return str(prompt.query_one(selector, Static).renderable)


async def _assert_stayed(pilot, app, screen, name: Input, events) -> None:
    """The quit ended as Stay: the app, the screen and every draft are intact."""
    await _until(pilot, lambda: app.screen is screen, "Stay to return to Roleplay")
    await _until(
        pilot,
        lambda: app._quit_in_progress is False,
        "the quit guard to clear after Stay",
    )
    assert "quit" not in events
    assert app._shutting_down is False
    assert app.is_running
    assert name.value == _DRAFT_NAME
    assert screen.state.has_unsaved_changes is True


async def test_ctrl_q_over_a_dirty_draft_asks_and_stay_keeps_every_draft(
    monkeypatch,
):
    """AC#1/#4: the same Save / Discard / Stay question; Stay keeps it all."""
    events: list[object] = []
    _Store(monkeypatch, events)
    app = _build_test_app(configured_default="personas")
    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _quit_recorder(app, events))
    async with app.run_test(size=(160, 48)) as pilot:
        screen, name = await _roleplay_editor(app, pilot)
        await _dirty(pilot, screen, name)

        prompt = await _ctrl_q_asks(
            pilot,
            app,
            events,
            RoleplayDraftNavigationDialog,
            "Ctrl+Q to ask the Roleplay Save / Discard / Stay question",
        )
        assert "character form" in _domains(prompt)
        assert app._quit_in_progress is True

        # Ctrl+Q again while the question is up: still one question.
        await pilot.press("ctrl+q")
        await pilot.pause(0.3)
        assert [
            s for s in app.screen_stack if isinstance(s, RoleplayDraftNavigationDialog)
        ] == [prompt]

        await pilot.click("#roleplay-draft-stay")
        await _assert_stayed(pilot, app, screen, name, events)
        assert events == []

        # Escape is Stay too, and the next Ctrl+Q asks again.
        await _ctrl_q_asks(
            pilot, app, events, RoleplayDraftNavigationDialog, "the second Ctrl+Q"
        )
        await pilot.press("escape")
        await _assert_stayed(pilot, app, screen, name, events)
        assert events == []


async def test_ctrl_q_discard_drops_the_draft_and_quits(monkeypatch):
    events: list[object] = []
    _Store(monkeypatch, events)
    app = _build_test_app(configured_default="personas")
    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _quit_recorder(app, events))
    async with app.run_test(size=(160, 48)) as pilot:
        screen, name = await _roleplay_editor(app, pilot)
        await _dirty(pilot, screen, name)
        await _ctrl_q_asks(
            pilot, app, events, RoleplayDraftNavigationDialog, "the Roleplay question"
        )

        await pilot.click("#roleplay-draft-discard-continue")
        await _until(pilot, lambda: "quit" in events, "Discard to let the quit run")
        assert events == ["quit"], "Discard must not save the draft"
        assert screen.state.has_unsaved_changes is False
        assert name.value == _SAVED_NAME


async def test_ctrl_q_save_saves_the_draft_then_quits(monkeypatch):
    events: list[object] = []
    _Store(monkeypatch, events)
    app = _build_test_app(configured_default="personas")
    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _quit_recorder(app, events))
    async with app.run_test(size=(160, 48)) as pilot:
        screen, name = await _roleplay_editor(app, pilot)
        await _dirty(pilot, screen, name)
        await _ctrl_q_asks(
            pilot, app, events, RoleplayDraftNavigationDialog, "the Roleplay question"
        )

        await pilot.click("#roleplay-draft-save-continue")
        await _until(pilot, lambda: "quit" in events, "Save to let the quit run")
        assert events == [("saved", "1", _DRAFT_NAME), "quit"], (
            "the draft must be saved before the quit runs"
        )
        assert screen.state.has_unsaved_changes is False


async def test_ctrl_q_failed_save_offers_retry_and_never_quits_silently(
    monkeypatch,
):
    """AC#2: a failed save routes through the recovery dialog, never a quit."""
    events: list[object] = []
    store = _Store(monkeypatch, events)
    store.fail = True
    app = _build_test_app(configured_default="personas")
    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _quit_recorder(app, events))
    async with app.run_test(size=(160, 48)) as pilot:
        screen, name = await _roleplay_editor(app, pilot)
        await _dirty(pilot, screen, name)
        await _ctrl_q_asks(
            pilot, app, events, RoleplayDraftNavigationDialog, "the Roleplay question"
        )

        await pilot.click("#roleplay-draft-save-continue")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, RoleplayDraftRecoveryDialog) or "quit" in events
            ),
            "the failed save to offer Retry / Stay",
        )
        assert "quit" not in events, "a failed save quit silently"
        first = app.screen
        assert "character form" in _domains(first)

        # Retry runs the save again; it fails again, so it asks again.
        await pilot.click("#roleplay-draft-retry")
        await _until(
            pilot,
            lambda: (
                (
                    isinstance(app.screen, RoleplayDraftRecoveryDialog)
                    and app.screen is not first
                )
                or "quit" in events
            ),
            "the retried save to fail and ask again",
        )
        assert events == [("save failed", "1"), ("save failed", "1")]

        await pilot.click("#roleplay-draft-recovery-stay")
        await _assert_stayed(pilot, app, screen, name, events)

        # Once the store recovers, the same Ctrl+Q saves and then quits.
        store.fail = False
        await _ctrl_q_asks(
            pilot, app, events, RoleplayDraftNavigationDialog, "Ctrl+Q after Stay"
        )
        await pilot.click("#roleplay-draft-save-continue")
        await _until(pilot, lambda: "quit" in events, "the recovered save to quit")
        assert events[-2:] == [("saved", "1", _DRAFT_NAME), "quit"]


class _CoveringModal(ModalScreen[None]):
    """A modal with no quit hooks, like a picker open over Roleplay."""

    def compose(self):
        yield Static("covering modal")


def _close_from_its_own_timer(modal) -> None:
    """Make ``modal`` call a bare ``dismiss()`` from its own timer.

    ``Screen.dismiss()`` pops the TOP screen, so with a quit prompt above it
    this pops the prompt unanswered -- the hazard ``await_quit_prompt``
    exists for. The ``AwaitComplete`` is deliberately not returned: a timer
    awaits its callback's result, and awaiting a screen's own dismiss from
    its own handler raises.
    """

    def _fire() -> None:
        modal.dismiss(None)

    modal.set_timer(0.05, _fire)


@pytest.mark.parametrize("vanishing", ["question", "recovery"])
async def test_a_vanished_roleplay_quit_prompt_means_stay(monkeypatch, vanishing):
    """AC#3: either prompt leaving the stack unanswered ends the quit as Stay."""
    events: list[object] = []
    store = _Store(monkeypatch, events)
    store.fail = vanishing == "recovery"
    app = _build_test_app(configured_default="personas")
    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _quit_recorder(app, events))
    async with app.run_test(size=(160, 48)) as pilot:
        screen, name = await _roleplay_editor(app, pilot)
        await _dirty(pilot, screen, name)
        modal = _CoveringModal()
        await app.push_screen(modal)
        await _until(pilot, lambda: app.screen is modal, "the covering modal")

        prompt = await _ctrl_q_asks(
            pilot,
            app,
            events,
            RoleplayDraftNavigationDialog,
            "Ctrl+Q under the modal to ask the Roleplay question",
        )
        assert modal in app.screen_stack
        if vanishing == "recovery":
            await pilot.click("#roleplay-draft-save-continue")
            await _until(
                pilot,
                lambda: (
                    isinstance(app.screen, RoleplayDraftRecoveryDialog)
                    or "quit" in events
                ),
                "the failed save to offer Retry / Stay",
            )
            assert "quit" not in events
            prompt = app.screen

        _close_from_its_own_timer(modal)
        await _until(
            pilot,
            lambda: prompt not in app.screen_stack,
            "the covered modal's dismiss to pop the quit prompt",
            timeout=5.0,
        )
        await _assert_stayed(pilot, app, screen, name, events)
        await _until(
            pilot,
            lambda: any(n.message == "Quit cancelled." for n in app._notifications),
            'the "Quit cancelled." toast',
            timeout=5.0,
        )
        assert modal not in app.screen_stack
        assert app._exception is None


def _waiting_notice(app) -> str | None:
    """The quit flow's "waiting for Roleplay work" toast, if one was shown."""
    for notification in app._notifications:
        if notification.message.startswith("Waiting for Roleplay work"):
            return notification.message
    return None


async def test_ctrl_q_during_roleplay_work_says_it_is_waiting_then_asks(
    monkeypatch,
):
    """Ctrl+Q over a running Roleplay operation says what it waits for.

    The quit waits for in-flight Roleplay work (a save, or a reaction
    generation holding the visual-identity operation) before asking, so it
    never asks about a draft that is still changing. That wait used to be
    silent: Ctrl+Q appeared to do nothing, and pressing it again did nothing
    either because the quit was already in progress.
    """
    events: list[object] = []
    _Store(monkeypatch, events)
    app = _build_test_app(configured_default="personas")
    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _quit_recorder(app, events))
    async with app.run_test(size=(160, 48)) as pilot:
        screen, name = await _roleplay_editor(app, pilot)
        await _dirty(pilot, screen, name)
        # Stand in for a running reaction generation: the screen holds the
        # visual-identity operation task for as long as the work runs.
        finish = asyncio.Event()
        operation = asyncio.get_running_loop().create_task(finish.wait())
        screen._visual_identity_operation_task = operation
        try:
            await pilot.press("ctrl+q")
            await _until(
                pilot,
                lambda: _waiting_notice(app) is not None or "quit" in events,
                "Ctrl+Q to say what it is waiting for",
                timeout=5.0,
            )
            assert "quit" not in events, (
                "Ctrl+Q quit straight past the unsaved Roleplay draft"
            )
            assert "character visuals" in _waiting_notice(app)

            # Still waiting: no question and no quit until the work finishes.
            await pilot.pause(0.3)
            assert not isinstance(app.screen, RoleplayDraftNavigationDialog)
            assert "quit" not in events
            assert app._quit_in_progress is True

            finish.set()
            await _until(
                pilot,
                lambda: (
                    isinstance(app.screen, RoleplayDraftNavigationDialog)
                    or "quit" in events
                ),
                "the finished work to hand over to the Roleplay question",
            )
            assert "quit" not in events
            assert "character visuals" not in _domains(app.screen)
            assert "character form" in _domains(app.screen)
            await pilot.click("#roleplay-draft-stay")
            await _assert_stayed(pilot, app, screen, name, events)
            assert events == []
        finally:
            finish.set()
            await operation


async def test_ctrl_q_with_clean_drafts_quits_without_asking(monkeypatch):
    events: list[object] = []
    _Store(monkeypatch, events)
    app = _build_test_app(configured_default="personas")
    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _quit_recorder(app, events))
    async with app.run_test(size=(160, 48)) as pilot:
        screen, _name = await _roleplay_editor(app, pilot)
        assert screen._aggregate_roleplay_draft_snapshot().is_clean

        asked: list[object] = []

        def _quit_ran_without_asking() -> bool:
            if isinstance(app.screen, RoleplayDraftNavigationDialog):
                asked.append(app.screen)
            return "quit" in events

        await pilot.press("ctrl+q")
        await _until(pilot, _quit_ran_without_asking, "a clean Roleplay to quit")
        assert asked == []
        assert events == ["quit"]
