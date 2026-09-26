"""TASK-32954 Task 5: ``CharacterCardChanged`` -> Personas refresh.

Unit-level: ``PersonasScreen._on_character_card_changed`` is exercised by
calling the unbound method against a lightweight stand-in object (rather
than ``PersonasScreen.__new__``, which pulls in Textual widget machinery
the handler never touches) -- see the task-5 controller ruling R3. The
app-level forwarder (``TldwCli.on_character_card_changed``) is exercised
the same way, against a fake app exposing only ``screen_stack``/
``run_worker`` -- it never touches ``self`` beyond those two attributes.

Importing ``TldwCli`` from ``tldw_chatbook.app`` below at module scope runs
that module's import at collection time, before ``Tests/UI/conftest.py``'s
autouse ``_disable_model_catalog_refresh`` fixture would otherwise import it
for the first time from *inside* a per-test ``isolate_test_environment``
sandbox -- the combination that trips this checkout's ``RecoveryRequired:
raw_source_selection_changed`` (ADR-126) at fixture setup, before any test
body runs. Pre-warming the import here (matching how ``Tests/LLM_Provider_
Catalog/test_app_model_catalog_wiring.py`` already imports ``tldw_chatbook.
app`` at collection time without issue) lets this file's tests execute
normally under plain pytest (fix round 1).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Character_Chat.character_events import CharacterCardChanged
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterEditorCancelled,
    CharacterSaveRequested,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# PersonasScreen._on_character_card_changed
# ---------------------------------------------------------------------------


def _screen(
    selected_id: int,
    *,
    has_unsaved_changes: bool = False,
    visual_identity_unsaved: bool = False,
    persona_shared_visual_identity_unsaved: bool = False,
    persona_visual_unsaved: bool = False,
    select_character: Any = None,
    save_state: Any = None,
):
    calls = {"reload": [], "notice": []}

    async def _default_reload(entity_id, entity_name, **kwargs):
        calls["reload"].append((entity_id, entity_name, kwargs))

    def _default_save_state():
        return {"personas_preview": {"greeting": "hi", "history": [], "seeded_for": None, "greeting_index": 0}}

    screen = SimpleNamespace(
        state=SimpleNamespace(
            selected_entity_kind="character",
            selected_entity_id=str(selected_id),
            selected_entity_name="Aria",
            runtime_source="local",
            has_unsaved_changes=has_unsaved_changes,
        ),
        _select_character=select_character or _default_reload,
        _visual_identity_has_unsaved_authoring=lambda: visual_identity_unsaved,
        _persona_shared_visual_identity_has_unsaved_authoring=lambda: (
            persona_shared_visual_identity_unsaved
        ),
        _persona_visual_has_unsaved_authoring=lambda: persona_visual_unsaved,
        save_state=save_state or _default_save_state,
        _notify=lambda msg, sev="information": calls["notice"].append(msg),
        _console_changed_character_ids=set(),
        character_handler=SimpleNamespace(
            current_character_id=str(selected_id),
            current_character_data={"id": selected_id, "name": "Aria", "version": 1},
        ),
    )
    return screen, calls


async def test_clean_editor_reloads_changed_character():
    screen, calls = _screen(7)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    assert len(calls["reload"]) == 1


async def test_form_dirty_keeps_edits_and_warns():
    screen, calls = _screen(7, has_unsaved_changes=True)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    assert not calls["reload"]
    assert "changed from the Console" in calls["notice"][0]
    # Final review I1: the copy says what Save and Cancel actually do.
    assert "Cancel" in calls["notice"][0] and "overwrite" in calls["notice"][0]
    assert screen._console_changed_character_ids == {"7"}


async def test_only_visual_identity_authoring_unsaved_keeps_edits_and_warns():
    """Fix round 1, review point 2: the form itself can be clean while a
    visual-identity authoring session (persona/character portrait, etc.) is
    still in progress -- that must block the reload too, matching the
    ``_run_guarded`` predicate exactly."""
    screen, calls = _screen(7, visual_identity_unsaved=True)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    assert not calls["reload"]
    assert "changed from the Console" in calls["notice"][0]


async def test_other_character_ignored():
    screen, calls = _screen(7)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(8))
    assert not calls["reload"] and not calls["notice"]


async def test_reload_restores_the_live_preview_snapshot():
    """Fix round 1, review point 3: a reload must not wipe the in-progress
    preview conversation -- ``_select_character`` gets the same
    ``personas_preview`` snapshot ``save_state`` would have handed a
    navigation round-trip (see ``_apply_pending_restore``)."""
    preview_snapshot = {
        "greeting": "Hello there.",
        "history": [{"role": "user", "content": "hi"}],
        "seeded_for": "7",
        "greeting_index": 1,
    }
    screen, calls = _screen(
        7, save_state=lambda: {"personas_preview": preview_snapshot}
    )
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    assert len(calls["reload"]) == 1
    _entity_id, _entity_name, kwargs = calls["reload"][0]
    assert kwargs.get("restore_preview") == preview_snapshot


async def test_server_selection_ignores_a_local_change_event():
    """Qodo #8: local and tldw_server character ids share one integer space,
    so a local Console save must not reload or stale-mark a server card."""
    screen, calls = _screen(7, has_unsaved_changes=True)
    screen.state.runtime_source = "server"
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    assert not calls["reload"] and not calls["notice"]
    assert screen._console_changed_character_ids == set()


async def test_reload_drops_the_stale_cached_card_before_it_yields():
    """Qodo #1: the fresh card lands later from a thread worker. Until then
    Edit must not open the pre-Console-save card (its Save would overwrite
    the Console's change with no warning, since the reload cleared the
    console-changed mark)."""
    seen = []

    async def _reload(entity_id, entity_name, **kwargs):
        seen.append(PersonasScreen._full_character_record(screen, entity_id))

    screen, _calls = _screen(7, select_character=_reload)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    assert seen == [None]


async def test_select_character_exception_is_logged_and_does_not_propagate():
    """Fix round 1, review point 1: a background notification handler must
    never crash the app. ``_select_character`` performs unguarded
    query_one/DB/server work; a failure there is caught, not raised."""

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("select_character exploded")

    screen, _calls = _screen(7, select_character=_boom)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    # No assertion beyond "did not raise" -- pytest fails the test itself if
    # the exception above propagates out of the awaited coroutine.


# ---------------------------------------------------------------------------
# Final review I1: a Personas save after a Console change is never silent
# ---------------------------------------------------------------------------


class _Pane:
    def show_validation(self, *_args):
        pass

    def show_validation_editing(self):
        pass


def _save_screen(changed_ids):
    screen, calls = _screen(7)
    calls["saves"] = []
    screen._console_changed_character_ids = set(changed_ids)
    screen._local_character_actions_allowed = lambda: True
    screen._actor_pack_session = None
    screen._character_save_inflight = False
    screen._edit_mode = "edit"
    screen._validate_character = lambda data: ()
    screen.query_one = lambda *_a, **_k: _Pane()
    screen._save_character_worker = lambda data, sid, mode: calls["saves"].append(sid)
    return screen, calls


def _save(screen):
    PersonasScreen._handle_save_requested(screen, CharacterSaveRequested({"name": "Aria"}))


async def test_first_save_after_a_console_change_is_refused_with_a_way_forward():
    screen, calls = _save_screen({"7"})
    _save(screen)
    assert calls["saves"] == []
    assert "changed from the Console" in calls["notice"][-1]
    assert "Save again" in calls["notice"][-1]


async def test_second_deliberate_save_after_a_console_change_proceeds():
    screen, calls = _save_screen({"7"})
    _save(screen)
    screen._character_save_inflight = False
    _save(screen)
    assert calls["saves"] == ["7"]


async def test_save_without_a_console_change_is_normal():
    screen, calls = _save_screen(set())
    _save(screen)
    assert calls["saves"] == ["7"] and calls["notice"] == []


async def test_cancel_after_a_console_change_loads_the_new_version():
    screen, calls = _screen(7)
    screen._console_changed_character_ids = {"7"}
    screen._finish_cancel_edit = lambda: None

    async def _run_guarded(fn):
        await fn()

    screen._run_guarded = _run_guarded
    await PersonasScreen._handle_editor_cancelled(screen, CharacterEditorCancelled())
    assert [r[0] for r in calls["reload"]] == ["7"]
    assert screen._console_changed_character_ids == set()


# ---------------------------------------------------------------------------
# TldwCli.on_character_card_changed (app-level forwarder)
# ---------------------------------------------------------------------------


class _RecordingFakeApp:
    """``screen_stack`` for ``TldwCli.on_character_card_changed`` to walk.

    ``run_worker`` is called on the *screen* it finds, not the app, so this
    fake only needs to hold the stack.
    """

    def __init__(self, screen_stack):
        self.screen_stack = screen_stack


def _stand_in_personas_screen():
    """A bare ``PersonasScreen`` identity for ``isinstance`` only.

    ``__new__`` (never ``__init__``) leaves this unmounted, so both
    ``_on_character_card_changed`` and the inherited (mount-dependent)
    ``run_worker`` are stubbed on the instance -- nothing else on it is
    ever read.
    """
    screen = PersonasScreen.__new__(PersonasScreen)
    calls: list[CharacterCardChanged] = []
    worker_calls: list[tuple[Any, dict]] = []

    async def _stub(message):
        calls.append(message)

    def _run_worker(coro, **kwargs):
        worker_calls.append((coro, kwargs))

    screen._on_character_card_changed = _stub
    screen.run_worker = _run_worker
    return screen, calls, worker_calls


async def test_forwarder_reaches_personas_screen_under_a_modal():
    """Fix round 1, review point 4: walk the full stack, not just
    ``self.screen``, so a modal on top of Personas does not drop this."""
    personas_screen, calls, worker_calls = _stand_in_personas_screen()
    modal_on_top = object()  # anything that is not a PersonasScreen
    app = _RecordingFakeApp(screen_stack=[personas_screen, modal_on_top])
    message = CharacterCardChanged(9)

    TldwCli.on_character_card_changed(app, message)

    assert len(worker_calls) == 1
    coro, kwargs = worker_calls[0]
    await coro
    assert calls == [message]
    assert kwargs.get("group") == "personas-character-changed"
    assert kwargs.get("exit_on_error") is False


async def test_forwarder_ignores_stack_without_a_personas_screen():
    app = _RecordingFakeApp(screen_stack=[object(), object()])
    # Must not raise (there is no PersonasScreen.run_worker to call, and
    # none should be attempted).
    TldwCli.on_character_card_changed(app, CharacterCardChanged(9))
