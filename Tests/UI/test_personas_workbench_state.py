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

import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Persona_Buddy import (
    PersonaBuddyController,
    PersonaBuddyGeometry,
    PersonaBuddyPreferences,
    PersonaBuddySelection,
    parse_persona_buddy_preferences,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
    PersonasPreviewPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaBuddyActionRequested,
)

from Tests.UI.test_personas_dictionaries import (
    PersonasTestApp,
    _mounted,
    patch_character_paging,
)

pytestmark = pytest.mark.asyncio


async def test_restart_restores_selection_open_collapsed_and_geometry(
    mock_app_instance,
    stub_characters,
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "restart" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    record = {
        "id": "persona-7",
        "name": "Archivist",
        "version": 3,
        "is_active": True,
        "deleted": False,
    }

    def local_record(persona_id: str):
        return dict(record) if persona_id == "persona-7" else None

    async def scoped_record(persona_id: str, *, mode: str):
        assert mode == "local"
        found = local_record(persona_id)
        if found is None:
            raise ValueError("persona missing")
        return found

    scope = SimpleNamespace(
        local_service=SimpleNamespace(get_persona_profile=local_record),
        list_persona_profiles=AsyncMock(
            return_value={"items": [dict(record)], "total": 1}
        ),
        get_persona_profile=AsyncMock(side_effect=scoped_record),
    )
    initial = PersonaBuddyPreferences(
        enabled=True,
        selection=PersonaBuddySelection("local", "persona-7"),
        open=True,
        collapsed=True,
        geometry=PersonaBuddyGeometry(x=17, y=9, width=42, height=14),
    )
    controller = PersonaBuddyController(
        preferences=initial,
        local_persona_service=scope.local_service,
    )
    mock_app_instance.runtime_backend = "local"
    mock_app_instance.character_persona_scope_service = scope
    mock_app_instance.persona_buddy_controller = controller
    mock_app_instance.reconcile_persona_buddy_view = AsyncMock(return_value=True)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("persona-7", "Archivist")
        screen.post_message(
            PersonaBuddyActionRequested(
                action="close",
                source="local",
                persona_id="persona-7",
                revision=3,
            )
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
    await controller.shutdown()

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    expected = PersonaBuddyPreferences(
        enabled=True,
        selection=PersonaBuddySelection("local", "persona-7"),
        open=False,
        collapsed=True,
        geometry=PersonaBuddyGeometry(x=17, y=9, width=42, height=14),
    )
    assert parse_persona_buddy_preferences(raw["persona_buddy"]) == expected

    restarted = PersonaBuddyController(
        preferences=parse_persona_buddy_preferences(raw["persona_buddy"]),
        local_persona_service=scope.local_service,
    )
    assert restarted.current_preferences() == expected

    durable_before = config_path.read_bytes()
    restarted._preference_writer = lambda _preferences: False
    mock_app_instance.persona_buddy_controller = restarted
    mock_app_instance.reconcile_persona_buddy_view.reset_mock()
    restarted_app = PersonasTestApp(mock_app_instance)
    async with restarted_app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("persona-7", "Archivist")
        screen.post_message(
            PersonaBuddyActionRequested(
                action="show",
                source="local",
                persona_id="persona-7",
                revision=3,
            )
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
    await restarted.shutdown()

    assert restarted.current_preferences() == expected
    assert config_path.read_bytes() == durable_before
    mock_app_instance.reconcile_persona_buddy_view.assert_not_awaited()


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


class _RestoringPersonasTestApp(ConsolidatedCSSApp):
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

    async def test_fresh_screen_without_saved_state_auto_selects_first_row(
        self, mock_app_instance, stub_characters
    ):
        """No prior selection + a non-empty library: F-031 first-paint
        auto-select picks the first row and shows its card, not a void."""
        mock_app_instance.chat_dictionary_scope_service = None
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "char-1"
            assert screen.query_one("#ccp-character-card-view").display is True


class TestPreviewRestore:
    """AC#2: the preview conversation (greeting + turns) survives the round-trip."""

    async def test_delayed_character_load_keeps_user_label_neutral(
        self, mock_app_instance, stub_characters
    ):
        """An id/name-only selection uses ``User`` before and after its card arrives."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._select_character("char-1", "Elara")
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)

            pane.append_user("waiting for the card")
            await pilot.pause()
            assert pane.transcript_text() == "User: waiting for the card"

            await screen.preview.handle_character_loaded(
                character_id="char-1",
                card_data={"name": "Elara", "first_message": "Greetings, traveller."},
            )
            pane.append_user("hi")
            await pilot.pause()
            assert pane.transcript_text() == (
                "Elara: Greetings, traveller.\nUser: hi"
            )

    async def test_restore_refreshes_provider_readout_without_character_load(
        self, mock_app_instance, stub_characters
    ):
        """Restoring preview state also restores its provider affordances.

        ``stub_characters`` deliberately makes ``load_character`` a no-op, so
        this covers the valid navigation-restore path where no later
        ``CharacterMessage.Loaded`` event repaints the readout.
        """
        from textual.widgets import Static

        mock_app_instance.app_config = {
            "character_defaults": {
                "provider": "anthropic",
                "model": "claude-3-haiku",
            },
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._select_character("char-1", "Elara")
            await pilot.pause()
            saved = screen.save_state()

        app2 = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app2.run_test() as pilot2:
            screen2 = await _mounted(pilot2)
            readout = str(
                screen2.query_one("#personas-preview-provider", Static).renderable
            )

            assert readout == (
                "Provider: Anthropic / claude-3-haiku"
                " - Console default if unavailable: llama.cpp"
            )
            assert screen2.preview._readout_nav_provider == "anthropic"

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
            expected_transcript = (
                "Elara: Greetings, traveller.\nUser: hi\nElara: well met"
            )
            assert pane.transcript_text() == expected_transcript
            # simulate a chosen alternate greeting (task-438)
            screen.preview._current_greeting_index = 1

            saved = screen.save_state()
            assert saved["personas_preview"]["greeting"] == "Greetings, traveller."
            assert saved["personas_preview"]["greeting_index"] == 1
            assert saved["personas_preview"]["history"] == [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "well met"},
            ]

        app2 = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app2.run_test() as pilot2:
            screen2 = await _mounted(pilot2)
            pane2 = screen2.query_one(PersonasPreviewPane)
            assert pane2.greeting_text == "Greetings, traveller."
            # the chosen greeting index survives the round-trip (task-438 review)
            assert screen2.preview._current_greeting_index == 1
            assert pane2.transcript_text() == expected_transcript
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
        ``on_mount``; the screen falls back to a fully cleared selection (not
        just a blank center) so ``_console_action_allowed()`` cannot still
        treat the stale entity as attachable."""
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
            # The workbench selection itself must degrade to NO selection -
            # a lingering stale selected_entity_id would keep
            # _console_action_allowed() (attach/Start-Chat) wrongly enabled
            # and the inspector showing a stale name/kind.
            assert screen.state.selected_entity_id is None
            assert screen.state.selected_entity_kind is None
            assert screen._console_action_allowed() is False


class TestNonCharacterModeRestore:
    """F-040: saved non-Characters modes restore mode AND selection.

    ``_apply_pending_restore`` runs the full ``_apply_mode`` for the saved
    mode before re-selecting, so the mode's library rows and mode-specific
    panes are live when the selection lands. (Before F-040 these round-trips
    fell back to the fresh Characters default - the documented task-434
    floor this task raised.)
    """

    async def test_saved_dictionaries_mode_restores_mode_and_selection(
        self, mock_app_instance, stub_characters
    ):
        mock_app_instance.chat_dictionary_scope_service = SimpleNamespace(
            list_dictionaries=AsyncMock(
                return_value={
                    "dictionaries": [
                        {
                            "id": 91,
                            "name": "Some Dictionary",
                            "entry_count": 2,
                            "enabled": True,
                        }
                    ]
                }
            ),
            get_dictionary=AsyncMock(
                return_value={
                    "id": 91,
                    "name": "Some Dictionary",
                    "entries": [],
                    "version": 1,
                }
            ),
            get_statistics=AsyncMock(return_value={}),
        )
        saved = {
            "personas_workbench": {
                "active_mode": "dictionaries",
                "selected_entity_kind": "dictionary",
                "selected_entity_id": "91",
                "selected_entity_name": "Some Dictionary",
            },
            "personas_preview": None,
        }

        app = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app.run_test() as pilot:
            screen2 = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen2.state.active_mode == "dictionaries"
            assert screen2.state.selected_entity_id == "91"
            assert screen2.state.selected_entity_kind == "dictionary"
            assert screen2.state.selected_entity_name == "Some Dictionary"
            assert screen2._pending_restore is None
            assert (
                screen2.query_one("#personas-dictionary-detail").display is True
            )

    async def test_saved_personas_mode_restores_mode_and_selection(
        self, mock_app_instance, stub_characters
    ):
        """Personas round-trips too, not just Characters (F-040)."""
        mock_app_instance.character_persona_scope_service = SimpleNamespace(
            list_persona_profiles=AsyncMock(
                return_value={
                    "items": [{"id": "persona-1", "name": "Some Persona"}],
                    "total": 1,
                }
            ),
            get_persona_profile=AsyncMock(
                return_value={
                    "id": "persona-1",
                    "name": "Some Persona",
                    "description": "Keeps notes.",
                    "system_prompt": "You are archival.",
                }
            ),
        )
        saved = {
            "personas_workbench": {
                "active_mode": "personas",
                "selected_entity_kind": "persona",
                "selected_entity_id": "persona-1",
                "selected_entity_name": "Some Persona",
            },
            "personas_preview": None,
        }

        app = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app.run_test() as pilot:
            screen2 = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen2.state.active_mode == "personas"
            assert screen2.state.selected_entity_id == "persona-1"
            assert screen2.state.selected_entity_kind == "persona"
            assert screen2._pending_restore is None
            assert screen2.query_one("#ccp-persona-card-view").display is True


class TestInvalidRestoreStillAutoSelects:
    """Qodo review: an unusable saved payload must not suppress F-031.

    ``restore_state`` used to set ``_restored_from_saved_state`` for ANY
    dict-shaped payload, so a stale/invalid saved state landed the user on
    the dead no-selection first paint instead of auto-selecting a row.
    """

    async def test_invalid_mode_saved_state_still_auto_selects_first_row(
        self, mock_app_instance, stub_characters
    ):
        """A saved mode that does not resolve to a chip mode is not a real
        restore - first-paint auto-select still fires."""
        saved = {
            "personas_workbench": {
                # Not a MODE_CHIP_ORDER mode: nothing here can be restored.
                "active_mode": "prompts",
                "selected_entity_kind": "prompt",
                "selected_entity_id": "p-1",
                "selected_entity_name": "Some Prompt",
            },
            "personas_preview": None,
        }

        app = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "char-1"
            assert screen.query_one("#ccp-character-card-view").display is True

    async def test_stale_saved_selection_falls_back_to_auto_select(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """A valid payload whose entity no longer resolves: the restore
        clears the stale selection AND the round-trip flag, so auto-select
        still rescues the paint."""
        saved = {
            "personas_workbench": {
                "active_mode": "characters",
                "selected_entity_kind": "character",
                "selected_entity_id": "ghost-9",
                "selected_entity_name": "Ghost",
            },
            "personas_preview": None,
        }

        real_select = PersonasScreen._select_character

        async def _flaky_select(self, entity_id, entity_name, **kwargs):
            if str(entity_id) == "ghost-9":
                raise RuntimeError("stale entity")
            await real_select(self, entity_id, entity_name, **kwargs)

        monkeypatch.setattr(PersonasScreen, "_select_character", _flaky_select)

        app = _RestoringPersonasTestApp(mock_app_instance, saved)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # The stale restore failed, so the mount fell back to the
            # first-paint behavior and selected the first real row.
            assert screen.state.selected_entity_id == "char-1"
            assert screen.query_one("#ccp-character-card-view").display is True
