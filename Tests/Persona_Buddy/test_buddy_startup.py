"""Deferred legacy migration shares first-controller preference ownership."""

import threading
from dataclasses import replace
from types import SimpleNamespace

from loguru import logger

from tldw_chatbook.Persona_Buddy.preferences import (
    BuddySelection,
    PersonaBuddyPreferences,
    PersonaBuddySelection,
    parse_persona_buddy_preferences,
    serialize_persona_buddy_preferences,
)


def test_unclaimed_migration_commits_before_first_controller_can_read(monkeypatch):
    import tldw_chatbook.app as app_module
    from tldw_chatbook.Persona_Buddy import controller, library, preferences

    previous = PersonaBuddyPreferences(
        enabled=True, selection=PersonaBuddySelection("local", "legacy")
    )
    candidate = replace(previous, selection=BuddySelection("copied"))
    host = SimpleNamespace(
        app_config={"persona_buddy": serialize_persona_buddy_preferences(previous)},
        chachanotes_db=object(),
        local_character_persona_service=None,
        persona_actor_pack_coordinator=SimpleNamespace(
            recovery_attempted=True,
            recovery_error=None,
            ensure_recovered=lambda: None,
        ),
        actor_pack_recovery_error=None,
        loguru_logger=logger,
        _persona_buddy_controller=None,
        _persona_buddy_controller_lock=threading.Lock(),
        call_after_refresh=lambda *_args: None,
        _notify_persona_buddy_changed=lambda: None,
        call_from_thread=lambda callback: callback(),
        run_worker=lambda coroutine, **_kwargs: coroutine.close(),
    )
    entered, release = threading.Event(), threading.Event()
    lock_was_held = []

    def persist(_candidate):
        acquired = host._persona_buddy_controller_lock.acquire(blocking=False)
        lock_was_held.append(not acquired)
        if acquired:
            host._persona_buddy_controller_lock.release()
        entered.set()
        assert release.wait(5)
        return True

    class Library:
        def __init__(self, *_args, **_kwargs):
            pass

        def ensure_builtin(self, **_kwargs):
            pass

        def migrate_legacy_selection(self, old, *, writer):
            return candidate if writer(candidate) else old

    class Controller:
        def __init__(self, **kwargs):
            self.preferences = kwargs["preferences"]

        async def migrate_legacy_selection(self, _library):
            pass

    monkeypatch.setattr(library, "BuddyLibrary", Library)
    monkeypatch.setattr(controller, "PersonaBuddyController", Controller)
    monkeypatch.setattr(preferences, "persist_persona_buddy_preferences", persist)
    worker = threading.Thread(
        target=app_module.TldwCli.ensure_actor_pack_recovery, args=(host,)
    )
    worker.start()
    try:
        assert entered.wait(5)
        builder = threading.Thread(
            target=app_module.TldwCli._build_persona_buddy_controller, args=(host,)
        )
        builder.start()
    finally:
        release.set()
        worker.join(5)
    builder.join(5)
    assert not worker.is_alive() and not builder.is_alive()
    assert lock_was_held == [True]
    assert host._persona_buddy_controller.preferences == candidate
    assert (
        parse_persona_buddy_preferences(host.app_config["persona_buddy"]) == candidate
    )


def test_controller_claim_during_copy_hands_off_without_overwriting(monkeypatch):
    import tldw_chatbook.app as app_module
    from tldw_chatbook.Persona_Buddy import library, preferences

    previous = PersonaBuddyPreferences(
        selection=PersonaBuddySelection("local", "legacy")
    )
    selected = replace(previous, selection=BuddySelection("user-choice"))
    delegated = []

    class Controller:
        def migrate_legacy_selection(self, value):
            delegated.append(value)
            return "delegated"

    host = SimpleNamespace(
        app_config={"persona_buddy": serialize_persona_buddy_preferences(previous)},
        chachanotes_db=object(),
        local_character_persona_service=None,
        persona_actor_pack_coordinator=SimpleNamespace(
            recovery_attempted=True,
            recovery_error=None,
            ensure_recovered=lambda: None,
        ),
        actor_pack_recovery_error=None,
        loguru_logger=logger,
        _persona_buddy_controller=None,
        _persona_buddy_controller_lock=threading.Lock(),
        call_from_thread=lambda callback: callback(),
        run_worker=lambda value, **_kwargs: None,
    )

    class Library:
        def __init__(self, *_args, **_kwargs):
            pass

        def ensure_builtin(self, **_kwargs):
            pass

        def migrate_legacy_selection(self, old, *, writer):
            # First feature use wins while the heavy visual copy is in flight.
            with host._persona_buddy_controller_lock:
                host._persona_buddy_controller = Controller()
                host.app_config["persona_buddy"] = serialize_persona_buddy_preferences(
                    selected
                )
            assert not writer(replace(old, selection=BuddySelection("copied")))
            return old

    monkeypatch.setattr(library, "BuddyLibrary", Library)
    monkeypatch.setattr(
        preferences,
        "persist_persona_buddy_preferences",
        lambda _value: (_ for _ in ()).throw(AssertionError("stale migration wrote")),
    )
    app_module.TldwCli.ensure_actor_pack_recovery(host)
    assert parse_persona_buddy_preferences(host.app_config["persona_buddy"]) == selected
    assert len(delegated) == 1
