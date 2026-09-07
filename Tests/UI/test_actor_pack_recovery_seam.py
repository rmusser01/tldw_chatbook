# Tests/UI/test_actor_pack_recovery_seam.py
"""task-21106: Actor Pack recovery moved out of ``TldwCli.__init__``.

Startup recovery used to run synchronous SQLite during app construction —
which also crashed ``_build_test_app`` (its ChaChaNotes DB is None) and
silently disarmed the CSS parse-cache cliff guard. These pins cover the new
seam end to end:

- construction performs NO recovery (and no longer needs a live DB);
- ``ensure_actor_pack_recovery`` skips cleanly without a profile DB, and maps
  the coordinator outcome onto ``actor_pack_recovery_error`` exactly the way
  ``__init__`` used to;
- the deferred-startup worker actually kicks recovery on a real mounted app;
- the Personas surface gates recovery ahead of its first library read.

Imported pytest fixtures are intentionally rebound as test parameters.
"""

# ruff: noqa: F811

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_personas_workbench import (
    PersonasTestApp,
    stub_characters as _stub_characters,  # noqa: F401 - registers fixture
    stub_scope_service as _stub_scope_service,  # noqa: F401 - registers fixture
)
from tldw_chatbook.UI.CCP_Modules.ccp_character_handler import CCPCharacterHandler


def test_app_construction_defers_actor_pack_recovery():
    """__init__ wires the coordinator but never runs recovery itself.

    Red on the pre-fix tree in the strongest possible way: `_build_test_app`
    itself crashed with ``AttributeError: 'NoneType' object has no attribute
    'execute_query'`` (repository.py:276) before this test body could assert
    anything.
    """
    app = _build_test_app()
    coordinator = app.persona_actor_pack_coordinator
    assert coordinator is not None
    assert coordinator.recovery_attempted is False
    assert app.actor_pack_recovery_error is None


def test_ensure_recovery_skips_without_a_profile_database():
    """No ChaChaNotes DB (the factory harness) means recovery is a no-op."""
    app = _build_test_app()
    assert app.chachanotes_db is None

    app.ensure_actor_pack_recovery()

    assert app.persona_actor_pack_coordinator.recovery_attempted is False
    assert app.actor_pack_recovery_error is None


@pytest.mark.parametrize(
    ("recovery_error", "blocked_ids", "expected"),
    [
        (None, (), None),
        (None, ("i" * 32,), "actor_pack_recovery_blocked"),
        ("actor_pack_recovery_failed", (), "actor_pack_recovery_failed"),
    ],
    ids=["clean", "blocked", "failed"],
)
def test_ensure_recovery_maps_coordinator_outcomes(
    recovery_error, blocked_ids, expected
):
    """The __init__-era outcome mapping survives the move verbatim."""
    app = _build_test_app()
    app.chachanotes_db = object()  # the harness builds without one
    result = (
        None
        if recovery_error is not None
        else SimpleNamespace(blocked_intent_ids=blocked_ids)
    )
    app.persona_actor_pack_coordinator = SimpleNamespace(
        recovery_attempted=False,
        recovery_error=recovery_error,
        ensure_recovered=lambda: result,
    )

    app.ensure_actor_pack_recovery()

    assert app.actor_pack_recovery_error == expected


def test_ensure_recovery_runs_real_recovery_through_the_app_method(tmp_path):
    """The app seam drives genuine SQLite recovery, once, when a DB exists.

    The factory harness has no ChaChaNotes DB, so the mounted pins above can
    only prove the wiring; this closes the loop against a real store — the
    exact combination production boots with.
    """
    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinator,
    )
    from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
    from tldw_chatbook.Character_Chat.local_character_persona_service import (
        LocalCharacterPersonaService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    app = _build_test_app()
    database = CharactersRAGDB(tmp_path / "actors.db", client_id="seam-test")
    try:
        repository = ActorPackRepository(database)
        service = LocalCharacterPersonaService(
            database, persona_store_path=tmp_path / "personas.json"
        )
        app.chachanotes_db = database
        app.persona_actor_pack_coordinator = PersonaActorPackCoordinator(
            repository, service
        )

        app.ensure_actor_pack_recovery()
        app.ensure_actor_pack_recovery()  # idempotent second call

        assert app.persona_actor_pack_coordinator.recovery_attempted is True
        assert app.persona_actor_pack_coordinator.recovery_error is None
        assert app.actor_pack_recovery_error is None
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_deferred_startup_kicks_the_recovery_worker():
    """A real mounted app runs recovery via the deferred-startup thread worker.

    This is the only automatic runner left after the move, so it must be
    proven on the genuine app, not a harness double: swap the app-level
    ensure for a recorder and mount for real.
    """
    app = _build_test_app()
    ran = threading.Event()
    app.ensure_actor_pack_recovery = ran.set  # instance attr shadows the method
    async with app.run_test() as pilot:
        for _ in range(80):
            if ran.is_set():
                break
            await pilot.pause(0.05)
    assert ran.is_set(), (
        "deferred startup never invoked ensure_actor_pack_recovery — the "
        "task-21106 background kick is unwired"
    )


@pytest.mark.asyncio
async def test_personas_mount_gates_recovery_before_the_library_read(
    mock_app_instance, _stub_characters, _stub_scope_service, monkeypatch
):
    """The Personas surface awaits recovery before its first library read."""
    order: list[str] = []
    mock_app_instance.ensure_actor_pack_recovery = lambda: order.append("recovery")

    async def recording_refresh(self):
        order.append("library_read")

    monkeypatch.setattr(
        CCPCharacterHandler, "refresh_character_list", recording_refresh
    )

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        for _ in range(80):
            if "library_read" in order:
                break
            await pilot.pause(0.05)

    assert "library_read" in order, "the mount-time library read never ran"
    assert order[0] == "recovery", (
        f"recovery must land before the first library read; observed {order!r}"
    )
    assert order.count("recovery") == 1
