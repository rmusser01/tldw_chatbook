"""PersonasScreen state persistence across navigation (task-434, Task 2).

A Personas -> Console -> back round-trip pushes/pops ``PersonasScreen``.
``BaseAppScreen``'s default ``save_state``/``restore_state`` only round-trips
``self.state_data`` (unused here), so without an override the workbench
selection and the ephemeral preview conversation (which lives outside
``self.state`` entirely - see ``PersonasPreviewController``) were lost on
every round-trip.

This module covers ``PersonasScreen.save_state``/``restore_state`` and
``_apply_pending_restore``:

- AC#1: the previously selected item, mode, and center view are restored.
- AC#2: the preview conversation (greeting + turns) survives the round-trip.
- The ``:133`` seeded-for guard in
  ``PersonasPreviewController.handle_character_loaded`` must not erase a
  preview transcript rebuilt by ``restore_conversation``.

Harness: mirrors ``Tests/UI/test_personas_dictionaries.py``'s
``PersonasTestApp`` (a delegating ``App`` that ``push_screen(PersonasScreen(
self))``) and its ``_mounted(pilot)`` helper. The "fresh screen restores"
half of each test mirrors how ``app.py``'s navigation actually does it
(``tldw_chatbook/app.py`` around the ``_screen_states``/``switch_screen``
code): construct a new screen, call ``restore_state`` on it, THEN push it -
never mount a screen before its saved state has been seeded.
"""

from unittest.mock import AsyncMock

import pytest
from textual.app import App

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
    PersonasPreviewPane,
)

from Tests.UI.test_personas_dictionaries import (
    PersonasTestApp,
    _mounted,
    patch_character_paging,
)

pytestmark = pytest.mark.asyncio

CHARACTERS = [
    {
        "id": "char-1",
        "name": "Elara",
        "description": "A wandering healer",
        "first_message": "Greetings, traveller.",
        "version": 1,
    },
]


@pytest.fixture
def stub_characters(monkeypatch):
    """Stub the character library + loader for a single seeded character.

    ``CCPCharacterHandler.load_character`` is patched to a no-op AsyncMock:
    this module tests ``PersonasScreen.save_state``/``restore_state`` and the
    preview-restoration path, not the (separately covered elsewhere)
    character-load worker pipeline. Leaving the real thread worker running
    would race its own ``CharacterMessage.Loaded`` -> ``handle_character_loaded``
    delivery against this module's manual preview-seeding calls, since both
    touch the same ``seeded_for``/transcript state.
    """
    monkeypatch.setattr(
        character_handler_module,
        "fetch_all_characters",
        lambda: [dict(c) for c in CHARACTERS],
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            (dict(c) for c in CHARACTERS if str(c["id"]) == str(character_id)), None
        ),
    )
    patch_character_paging(monkeypatch)
    monkeypatch.setattr(
        character_handler_module.CCPCharacterHandler, "load_character", AsyncMock()
    )


class _RestoringPersonasTestApp(App):
    """Harness that seeds ``restore_state`` before the screen ever mounts.

    Mirrors ``tldw_chatbook/app.py``'s navigation handler, which constructs
    the destination screen, calls ``new_screen.restore_state(...)``, and only
    then pushes/switches to it - restore_state must run on a screen that does
    not exist in the DOM yet.

    Deliberately NOT a ``PersonasTestApp`` subclass: Textual's message
    dispatch (``MessagePump._get_dispatch_methods``) looks up ``on_mount`` in
    *every* class's own ``__dict__`` along the MRO and invokes each one it
    finds - not just the most-derived override. Subclassing ``PersonasTestApp``
    (which defines its own unconditional, non-restoring ``on_mount``) would
    therefore mount a SECOND, un-restored ``PersonasScreen`` alongside this
    one and leave it on top of the screen stack. This class duplicates
    ``PersonasTestApp``'s tiny delegating ``__getattr__``/``compose`` instead.
    """

    _NON_DELEGATED_PREFIXES = (
        "_",
        "watch_",
        "compute_",
        "validate_",
        "action_",
        "key_",
        "on_",
    )

    def __init__(self, mock_app_instance, saved_state):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = (
            mock_app_instance.character_persona_scope_service
        )
        self._saved_state = saved_state

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self):
        yield AppFooterStatus(id="app-footer-status")

    def on_mount(self) -> None:
        screen = PersonasScreen(self)
        screen.restore_state(self._saved_state)
        self.push_screen(screen)


def _seed_preview_turns(screen) -> "PersonasPreviewPane":
    """Manually seed a greeting + one user/assistant turn in the preview."""
    pane = screen.query_one(PersonasPreviewPane)
    return pane


class TestWorkbenchSelectionRestore:
    """AC#1: selection, mode, and center view survive the round-trip."""

    async def test_save_restore_preserves_character_selection_and_center(
        self, mock_app_instance, stub_characters
    ):
        mock_app_instance.chat_dictionary_scope_service = None
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._select_character("char-1", "Elara")
            await pilot.pause()
            assert screen.state.selected_entity_id == "char-1"

            saved = screen.save_state()
            assert saved["personas_workbench"]["selected_entity_id"] == "char-1"
            assert saved["personas_workbench"]["selected_entity_kind"] == "character"

        app2 = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app2.run_test() as pilot2:
            screen2 = await _mounted(pilot2)
            assert screen2.state.selected_entity_id == "char-1"
            assert screen2.state.selected_entity_kind == "character"
            assert screen2.state.selected_entity_name == "Elara"
            # center shows the character card view, not blank
            assert screen2.query_one("#ccp-character-card-view").display is True

    async def test_fresh_screen_without_saved_state_shows_blank_center(
        self, mock_app_instance, stub_characters
    ):
        """No prior selection: on_mount's default (blank) center is untouched."""
        mock_app_instance.chat_dictionary_scope_service = None
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            assert screen.state.selected_entity_id is None
            assert screen.query_one("#ccp-character-card-view").display is False


class TestPreviewRestore:
    """AC#2: the preview conversation (greeting + turns) survives the round-trip."""

    async def test_save_restore_preserves_preview_greeting_and_turns(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._select_character("char-1", "Elara")
            await pilot.pause()
            pane = _seed_preview_turns(screen)
            await pane.seed_greeting("Greetings, traveller.")
            pane.append_user("hi")
            screen.preview.history.append({"role": "user", "content": "hi"})
            pane.append_reply("well met")
            screen.preview.history.append(
                {"role": "assistant", "content": "well met"}
            )

            saved = screen.save_state()
            assert saved["personas_preview"]["greeting"] == "Greetings, traveller."
            assert saved["personas_preview"]["history"] == [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "well met"},
            ]

        app2 = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app2.run_test() as pilot2:
            screen2 = await _mounted(pilot2)
            pane2 = screen2.query_one(PersonasPreviewPane)
            assert pane2.greeting_text == "Greetings, traveller."
            text = pane2.transcript_text()
            assert "Greetings, traveller." in text
            assert "hi" in text
            assert "well met" in text
            assert screen2.preview.history == saved["personas_preview"]["history"]

    async def test_late_character_loaded_worker_does_not_erase_restored_turns(
        self, mock_app_instance, stub_characters
    ):
        """The seeded-for guard (personas_preview_controller.py ``:159``):

        a character-load worker's ``CharacterMessage.Loaded`` delivered after
        the screen already restored a preview transcript must refresh the
        reset-seed greeting only, never invalidate/erase the restored turns.
        ``load_character`` is mocked to a no-op by ``stub_characters`` (no
        real background thread races this), so this test drives
        ``handle_character_loaded`` directly to simulate the late delivery.
        """
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._select_character("char-1", "Elara")
            await pilot.pause()
            pane = _seed_preview_turns(screen)
            await pane.seed_greeting("Greetings, traveller.")
            pane.append_user("hi")
            screen.preview.history.append({"role": "user", "content": "hi"})
            pane.append_reply("well met")
            screen.preview.history.append(
                {"role": "assistant", "content": "well met"}
            )
            saved = screen.save_state()

        app2 = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app2.run_test() as pilot2:
            screen2 = await _mounted(pilot2)
            pane2 = screen2.query_one(PersonasPreviewPane)
            assert screen2.preview.seeded_for == "char-1"

            await screen2.preview.handle_character_loaded(
                character_id="char-1",
                card_data={"name": "Elara", "first_message": "Greetings, traveller."},
            )

            text = pane2.transcript_text()
            assert "Greetings, traveller." in text
            assert "hi" in text
            assert "well met" in text
            assert screen2.preview.history == saved["personas_preview"]["history"]


class TestPendingRestoreGuards:
    """_apply_pending_restore must degrade gracefully, never crash on_mount."""

    async def test_selection_failure_during_restore_degrades_to_blank_center(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """A ``_select_*`` raising for a stale/deleted entity must not crash
        ``on_mount``; the screen falls back to the blank center instead."""
        mock_app_instance.chat_dictionary_scope_service = None
        saved = {
            "personas_workbench": {
                "active_mode": "characters",
                "selected_entity_kind": "character",
                "selected_entity_id": "char-1",
                "selected_entity_name": "Elara",
            },
            "personas_preview": None,
        }

        def _boom(self, *args, **kwargs):
            raise RuntimeError("stale entity")

        monkeypatch.setattr(PersonasScreen, "_select_character", _boom)

        app = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            # Must not have crashed the app; center stays blank.
            assert screen.query_one("#ccp-character-card-view").display is False
