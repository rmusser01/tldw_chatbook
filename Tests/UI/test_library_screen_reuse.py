"""Library screen-instance reuse contracts (TASK-31521).

The library route is reusable (`ScreenRoute.reusable`): one LibraryScreen
per app run, suspended on switch-away, resumed on return. These pin the
four behaviors the 2026-09-04 audit gated enablement on:

1. same-instance resume (the reuse itself);
2. suspend stops every armed debounce timer -- Textual only auto-cancels
   timers on real removal, and three of Library's five relied entirely on
   that removal (which reuse removes);
3. resume is the per-visit refresh seam (revisits re-kick the active
   surface, so data changed elsewhere while hidden appears);
4. the ingest-registry listener's DOM/DB branches gate on the suspended
   flag with one resume-time reconciliation (the counting/toast half
   stays live -- it is a cross-tab signal).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest
import pytest_asyncio

from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route


def _notifier_owner(notifier):
    """Return the instance behind a bound notifier, if any."""
    return getattr(notifier, "__self__", None)


class _RawAppIngestionStateGuard:
    """Restore process-wide ingestion state borrowed by one real-app test."""

    def __init__(self) -> None:
        from tldw_chatbook.DB import Client_Media_DB_v2 as media_db_module
        from tldw_chatbook.RAG_Search import ingestion_indexing

        self._media_db_module = media_db_module
        self._ingestion_indexing = ingestion_indexing
        self._prior_indexer = ingestion_indexing._indexer
        self._prior_hook_installed = ingestion_indexing._hook_installed
        self._prior_failure_notifier = getattr(
            self._prior_indexer, "_failure_notifier", None
        )
        self._prior_guidance_notifier = getattr(
            self._prior_indexer, "_guidance_notifier", None
        )
        self._prior_ingest_callback = (
            ingestion_indexing._media_post_ingest_hook
            in media_db_module._MEDIA_POST_INGEST_CALLBACKS
        )
        self._prior_delete_callback = (
            ingestion_indexing._media_post_delete_hook
            in media_db_module._MEDIA_POST_DELETE_CALLBACKS
        )
        self._app = None
        self._closed = False

    def claim(self, app) -> None:
        """Identify the app whose notifier registrations this guard owns."""
        assert self._app is None
        self._app = app

    def _stop_owned_indexer(self, indexer, app) -> bool:
        """Stop *indexer* only while it remains this guard's global service."""
        ingestion_indexing = self._ingestion_indexing
        with ingestion_indexing._hook_lock:
            with ingestion_indexing._indexer_lock:
                if ingestion_indexing._indexer is not indexer:
                    if _notifier_owner(indexer._failure_notifier) is app:
                        indexer.set_failure_notifier(None)
                    if _notifier_owner(indexer._guidance_notifier) is app:
                        indexer.set_guidance_notifier(None)
                    return False

                current_failure = indexer._failure_notifier
                current_guidance = indexer._guidance_notifier
                notifier_transferred = any(
                    notifier is not None and _notifier_owner(notifier) is not app
                    for notifier in (current_failure, current_guidance)
                )
                if notifier_transferred:
                    if _notifier_owner(current_failure) is app:
                        indexer.set_failure_notifier(None)
                    if _notifier_owner(current_guidance) is app:
                        indexer.set_guidance_notifier(None)
                    return False
                if not any(
                    _notifier_owner(notifier) is app
                    for notifier in (current_failure, current_guidance)
                ):
                    return False

                ingestion_indexing._indexer = None
                owned_thread = indexer._thread
                indexer.stop()
                assert indexer._stopped is True
                assert owned_thread is None or not owned_thread.is_alive()
                if not self._prior_ingest_callback:
                    self._media_db_module.unregister_media_post_ingest_callback(
                        ingestion_indexing._media_post_ingest_hook
                    )
                if not self._prior_delete_callback:
                    self._media_db_module.unregister_media_post_delete_callback(
                        ingestion_indexing._media_post_delete_hook
                    )
                ingestion_indexing._hook_installed = self._prior_hook_installed
                assert (
                    ingestion_indexing._media_post_ingest_hook
                    in self._media_db_module._MEDIA_POST_INGEST_CALLBACKS
                ) is self._prior_ingest_callback
                assert (
                    ingestion_indexing._media_post_delete_hook
                    in self._media_db_module._MEDIA_POST_DELETE_CALLBACKS
                ) is self._prior_delete_callback
                return True

    async def close(self) -> None:
        """Release only state still owned by the claimed app."""
        if self._closed:
            return
        self._closed = True
        app = self._app
        assert app is not None
        ingestion_indexing = self._ingestion_indexing
        current = ingestion_indexing._indexer
        registration_owned = False

        if current is self._prior_indexer:
            current_failure = getattr(current, "_failure_notifier", None)
            current_guidance = getattr(current, "_guidance_notifier", None)
            notifier_transferred = any(
                notifier is not prior and _notifier_owner(notifier) is not app
                for notifier, prior in (
                    (current_failure, self._prior_failure_notifier),
                    (current_guidance, self._prior_guidance_notifier),
                )
            )
            if current is not None:
                if _notifier_owner(current_failure) is app:
                    current.set_failure_notifier(self._prior_failure_notifier)
                if _notifier_owner(current_guidance) is app:
                    current.set_guidance_notifier(self._prior_guidance_notifier)
            registration_owned = not notifier_transferred
        elif self._prior_indexer is None and current is not None:
            await asyncio.to_thread(self._stop_owned_indexer, current, app)
        elif current is not None:
            current_failure = getattr(current, "_failure_notifier", None)
            current_guidance = getattr(current, "_guidance_notifier", None)
            if _notifier_owner(current_failure) is app:
                current.set_failure_notifier(None)
            if _notifier_owner(current_guidance) is app:
                current.set_guidance_notifier(None)
        else:
            registration_owned = self._prior_indexer is None

        if registration_owned:
            if not self._prior_ingest_callback:
                self._media_db_module.unregister_media_post_ingest_callback(
                    ingestion_indexing._media_post_ingest_hook
                )
            if not self._prior_delete_callback:
                self._media_db_module.unregister_media_post_delete_callback(
                    ingestion_indexing._media_post_delete_hook
                )
            ingestion_indexing._hook_installed = self._prior_hook_installed
            assert (
                ingestion_indexing._media_post_ingest_hook
                in self._media_db_module._MEDIA_POST_INGEST_CALLBACKS
            ) is self._prior_ingest_callback
            assert (
                ingestion_indexing._media_post_delete_hook
                in self._media_db_module._MEDIA_POST_DELETE_CALLBACKS
            ) is self._prior_delete_callback

        active_indexer = ingestion_indexing._indexer
        if active_indexer is not None:
            assert _notifier_owner(active_indexer._failure_notifier) is not app
            assert _notifier_owner(active_indexer._guidance_notifier) is not app


@pytest_asyncio.fixture
async def _isolated_raw_app_ingestion_state():
    """Keep a real app's process-wide ingestion lease inside one test."""
    guard = _RawAppIngestionStateGuard()
    yield guard.claim
    await guard.close()


class _NotifierOwner:
    def notify_failure(self, _message):
        return None

    def notify_guidance(self, _message):
        return None


def _scratch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    data = tmp_path / "data"
    config = tmp_path / "config"
    for sub in (home, data, config):
        sub.mkdir(parents=True, exist_ok=True)
    config_file = config / "tldw_cli" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "library_screen_reuse")
    return home


async def _boot_settled(app, pilot) -> None:
    while not getattr(app, "_ui_ready", False):
        await asyncio.sleep(0.01)
    for _ in range(20):
        await asyncio.sleep(0.05)
        await pilot.pause()


async def _press_until_screen(pilot, key: str, expected: str) -> None:
    deadline = asyncio.get_running_loop().time() + 30.0
    await pilot.press(key)
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if type(pilot.app.screen).__name__ == expected:
            break
    assert type(pilot.app.screen).__name__ == expected
    for _ in range(6):
        await asyncio.sleep(0.05)
        await pilot.pause()


def test_library_route_is_flagged_reusable() -> None:
    route = resolve_screen_route("library")
    assert route is not None and route.reusable is True


@pytest.mark.ui
@pytest.mark.asyncio
async def test_library_reuse_and_suspend_timer_quiescence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_raw_app_ingestion_state,
) -> None:
    """One journey pins reuse, timer quiescence, and the resume seam.

    A single boot exercises all three because they are one lifecycle: visit
    Library, arm a debounce timer, leave (suspend must stop it), return
    (same instance, visit surfaces re-kicked).
    """
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    _isolated_raw_app_ingestion_state(app)
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)

        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        library = app.screen
        assert library._library_visit_entered is True, (
            "the first ScreenResume must run the visit-surface kicks"
        )

        # Arm a debounce timer the way a mid-keystroke filter would.
        library._media_state.filter_timer = library.set_timer(
            60.0, lambda: None
        )

        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        assert library._library_screen_suspended is True
        for attr in (
            "_library_source_snapshot_timeout_timer",
            "_library_list_entry_focus_timer",
        ):
            assert getattr(library, attr, None) is None, (
                f"{attr} still armed on the suspended screen -- Textual "
                "does not auto-cancel a suspended installed screen's "
                "timers, so suspend must"
            )
        # (wave-8 task 3) The notes autosave timer is a `LibraryNotesState`
        # field, not a flat screen attribute -- the screen's generated shim
        # block was deleted in the notes cleanup PR, so a `getattr` on the old
        # flat name passes VACUOUSLY. It leaves the string loop above for the
        # same explicit block the ingest, prompts and media timers already use.
        assert library._notes_state.autosave_timer is None, (
            "the notes autosave timer is still armed on the suspended "
            "screen -- Textual does not auto-cancel a suspended installed "
            "screen's timers, so suspend must"
        )
        # (wave-7 task 3) The two media debounce timers are
        # `LibraryMediaState` fields, not flat screen attributes -- the
        # screen's generated shim block was deleted in the media cleanup PR,
        # so a `getattr` on either old flat name passes VACUOUSLY. They leave
        # the string loop above for the same explicit block the ingest and
        # prompts timers already use.
        for media_timer_field in ("filter_timer", "selection_timer"):
            assert getattr(library._media_state, media_timer_field) is None, (
                f"the media {media_timer_field} is still armed on the "
                "suspended screen -- Textual does not auto-cancel a suspended "
                "installed screen's timers, so suspend must"
            )
        # (wave-5 merge) The ingest path-debounce timer is a
        # `LibraryIngestState` field, not a flat screen attribute -- the
        # screen's generated shim block was deleted in the ingest cleanup
        # PR, so a `getattr` on the old flat name passes VACUOUSLY.
        assert library._ingest_state.path_debounce_timer is None, (
            "the ingest path-debounce timer is still armed on the "
            "suspended screen -- Textual does not auto-cancel a suspended "
            "installed screen's timers, so suspend must"
        )
        # (wave-6 task 3) Same shape for the prompts search-debounce timer:
        # its flat screen shim was deleted in the prompts cleanup PR, so a
        # `getattr` on the old flat name would pass VACUOUSLY.
        assert library._prompts_state.debounce_timer is None, (
            "the prompts search-debounce timer is still armed on the "
            "suspended screen -- Textual does not auto-cancel a suspended "
            "installed screen's timers, so suspend must"
        )

        # The resume seam: revisits must re-kick the visit surfaces.
        kicks: list[str] = []
        real_refresh = library._refresh_library_visit_surfaces
        monkeypatch.setattr(
            library,
            "_refresh_library_visit_surfaces",
            lambda: (kicks.append("visit"), real_refresh())[1],
        )
        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        assert app.screen is library, (
            "library is a reusable route: returning must resume the "
            "installed instance, not construct a new one"
        )
        assert library._library_screen_suspended is False
        assert kicks == ["visit"], (
            "on_screen_resume must dispatch the per-visit surface refresh "
            "-- without it a revisit shows the previous visit's data"
        )


@pytest.mark.ui
@pytest.mark.asyncio
async def test_suspended_library_gates_ingest_dom_work_until_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_raw_app_ingestion_state,
) -> None:
    """Registry events against a hidden Library defer DOM work to resume."""
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    _isolated_raw_app_ingestion_state(app)
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        library = app.screen
        library._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA

        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")

        dynamic = Mock()
        snapshot = Mock()
        monkeypatch.setattr(
            library, "_update_library_ingest_dynamic_regions", dynamic
        )
        monkeypatch.setattr(
            library, "_refresh_local_source_snapshot", snapshot
        )
        # A registry mutation lands while the screen is hidden.
        library._handle_library_ingest_registry_changed()
        assert dynamic.call_count == 0, (
            "a suspended screen must not rebuild ingest widgets per event"
        )
        assert library._library_ingest_suspended_activity is True

        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        assert dynamic.call_count >= 1, (
            "resume must run exactly one ingest-UI reconciliation for the "
            "events gated while suspended"
        )
        assert snapshot.call_count >= 1, (
            "resume's visit refresh must re-read the source snapshot the "
            "suspended gate skipped"
        )
        assert library._library_ingest_suspended_activity is False


@pytest.mark.asyncio
async def test_raw_app_ingestion_guard_stops_owned_indexer_and_removes_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.DB import Client_Media_DB_v2 as media_db_module
    from tldw_chatbook.RAG_Search import ingestion_indexing

    monkeypatch.setattr(ingestion_indexing, "_indexer", None)
    monkeypatch.setattr(ingestion_indexing, "_hook_installed", False)
    monkeypatch.setattr(media_db_module, "_MEDIA_POST_INGEST_CALLBACKS", [])
    monkeypatch.setattr(media_db_module, "_MEDIA_POST_DELETE_CALLBACKS", [])

    owner = _NotifierOwner()
    guard = _RawAppIngestionStateGuard()
    guard.claim(owner)
    ingestion_indexing.install_media_ingest_hook(
        failure_notifier=owner.notify_failure,
        guidance_notifier=owner.notify_guidance,
    )
    owned_indexer = ingestion_indexing._indexer
    with owned_indexer._thread_lock:
        owned_indexer._ensure_thread_locked()
    owned_thread = owned_indexer._thread
    assert owned_thread is not None and owned_thread.is_alive()

    await guard.close()

    assert ingestion_indexing._indexer is None
    assert owned_indexer._stopped is True
    assert not owned_thread.is_alive()
    assert ingestion_indexing._hook_installed is False
    assert media_db_module._MEDIA_POST_INGEST_CALLBACKS == []
    assert media_db_module._MEDIA_POST_DELETE_CALLBACKS == []


@pytest.mark.asyncio
async def test_raw_app_ingestion_guard_restores_borrowed_indexer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.DB import Client_Media_DB_v2 as media_db_module
    from tldw_chatbook.RAG_Search import ingestion_indexing

    borrowed = ingestion_indexing.IngestionIndexer()
    prior_failure = Mock()
    prior_guidance = Mock()
    borrowed.set_failure_notifier(prior_failure)
    borrowed.set_guidance_notifier(prior_guidance)
    monkeypatch.setattr(ingestion_indexing, "_indexer", borrowed)
    monkeypatch.setattr(ingestion_indexing, "_hook_installed", True)
    monkeypatch.setattr(
        media_db_module,
        "_MEDIA_POST_INGEST_CALLBACKS",
        [ingestion_indexing._media_post_ingest_hook],
    )
    monkeypatch.setattr(
        media_db_module,
        "_MEDIA_POST_DELETE_CALLBACKS",
        [ingestion_indexing._media_post_delete_hook],
    )

    owner = _NotifierOwner()
    guard = _RawAppIngestionStateGuard()
    guard.claim(owner)
    ingestion_indexing.install_media_ingest_hook(
        failure_notifier=owner.notify_failure,
        guidance_notifier=owner.notify_guidance,
    )

    await guard.close()

    assert ingestion_indexing._indexer is borrowed
    assert borrowed._stopped is False
    assert borrowed._failure_notifier is prior_failure
    assert borrowed._guidance_notifier is prior_guidance
    assert ingestion_indexing._hook_installed is True
    assert media_db_module._MEDIA_POST_INGEST_CALLBACKS == [
        ingestion_indexing._media_post_ingest_hook
    ]
    assert media_db_module._MEDIA_POST_DELETE_CALLBACKS == [
        ingestion_indexing._media_post_delete_hook
    ]


@pytest.mark.asyncio
async def test_raw_app_ingestion_guard_preserves_transferred_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.DB import Client_Media_DB_v2 as media_db_module
    from tldw_chatbook.RAG_Search import ingestion_indexing

    monkeypatch.setattr(ingestion_indexing, "_indexer", None)
    monkeypatch.setattr(ingestion_indexing, "_hook_installed", False)
    monkeypatch.setattr(media_db_module, "_MEDIA_POST_INGEST_CALLBACKS", [])
    monkeypatch.setattr(media_db_module, "_MEDIA_POST_DELETE_CALLBACKS", [])

    owner = _NotifierOwner()
    guard = _RawAppIngestionStateGuard()
    guard.claim(owner)
    ingestion_indexing.install_media_ingest_hook(
        failure_notifier=owner.notify_failure,
        guidance_notifier=owner.notify_guidance,
    )
    transferred = _NotifierOwner()
    ingestion_indexing.install_media_ingest_hook(
        guidance_notifier=transferred.notify_guidance
    )
    active_indexer = ingestion_indexing._indexer

    await guard.close()

    assert ingestion_indexing._indexer is active_indexer
    assert active_indexer._stopped is False
    assert active_indexer._failure_notifier is None
    assert _notifier_owner(active_indexer._guidance_notifier) is transferred
    assert ingestion_indexing._hook_installed is True
    assert media_db_module._MEDIA_POST_INGEST_CALLBACKS == [
        ingestion_indexing._media_post_ingest_hook
    ]
    assert media_db_module._MEDIA_POST_DELETE_CALLBACKS == [
        ingestion_indexing._media_post_delete_hook
    ]


@pytest.mark.asyncio
async def test_raw_app_ingestion_guard_does_not_stop_replacement_at_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.DB import Client_Media_DB_v2 as media_db_module
    from tldw_chatbook.RAG_Search import ingestion_indexing

    monkeypatch.setattr(ingestion_indexing, "_indexer", None)
    monkeypatch.setattr(ingestion_indexing, "_hook_installed", False)
    monkeypatch.setattr(media_db_module, "_MEDIA_POST_INGEST_CALLBACKS", [])
    monkeypatch.setattr(media_db_module, "_MEDIA_POST_DELETE_CALLBACKS", [])

    owner = _NotifierOwner()
    guard = _RawAppIngestionStateGuard()
    guard.claim(owner)
    ingestion_indexing.install_media_ingest_hook(
        failure_notifier=owner.notify_failure,
        guidance_notifier=owner.notify_guidance,
    )
    retiring_indexer = ingestion_indexing._indexer
    replacement_owner = _NotifierOwner()
    replacement = ingestion_indexing.IngestionIndexer()
    replacement.set_failure_notifier(replacement_owner.notify_failure)
    replacement.set_guidance_notifier(replacement_owner.notify_guidance)

    async def replace_before_dispatch(function, *args):
        ingestion_indexing._indexer = replacement
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", replace_before_dispatch)
    await guard.close()

    assert ingestion_indexing._indexer is replacement
    assert replacement._stopped is False
    assert _notifier_owner(replacement._failure_notifier) is replacement_owner
    assert _notifier_owner(replacement._guidance_notifier) is replacement_owner
    assert retiring_indexer._stopped is False
    assert retiring_indexer._failure_notifier is None
    assert retiring_indexer._guidance_notifier is None
    assert ingestion_indexing._hook_installed is True
    assert media_db_module._MEDIA_POST_INGEST_CALLBACKS == [
        ingestion_indexing._media_post_ingest_hook
    ]
    assert media_db_module._MEDIA_POST_DELETE_CALLBACKS == [
        ingestion_indexing._media_post_delete_hook
    ]


class _RecordingTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_on_screen_suspend_stops_every_timer_in_isolation() -> None:
    """Unit contract for the suspend hook, no Textual app involved.

    (Qodo #2414 finding 1.) Every timer attribute the hook owns is armed
    with a recording stub; one call must stop and clear all seven and set
    the suspended flag. Enumerating them HERE too means a new timer added
    to the hook without updating this table fails loudly.
    """
    from types import SimpleNamespace

    from textual.app import active_app

    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        UnresolvedConversationKey,
    )
    from tldw_chatbook.UI.Library_Modules import library_unavailable_navigation
    from tldw_chatbook.UI.Library_Modules.library_navigation_controller import (
        LibraryNavigationController,
    )
    from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
        LibraryUnavailableConversationsBrowse,
        RoleplayReturnTarget,
    )
    from tldw_chatbook.UI.Screens.library_screen import (
        LibraryIngestState,
        LibraryMediaState,
        LibraryNotesState,
        LibraryPromptsState,
        LibraryScreen,
    )

    screen = LibraryScreen.__new__(LibraryScreen)
    # (wave-8 task 1, retargeted by task 3) The notes autosave timer is a
    # `LibraryNotesState` field, not a flat screen attribute -- the screen's
    # generated shim block was deleted in the notes cleanup PR, so
    # `setattr`/`getattr` on the old flat name would arm and assert a field
    # the hook never reads. An `object.__new__`/`__new__` screen also skips
    # `__init__`'s state construction, hence the explicit seed, exactly like
    # the `_media_state`/`_prompts_state`/`_ingest_state` ones here.
    screen._notes_state = LibraryNotesState()
    notes_timer = _RecordingTimer()
    screen._notes_state.autosave_timer = notes_timer
    # Seed the constructor-owned navigation seam too: suspend must execute
    # the real return cleanup before reaching the timer helpers below.
    screen._unavailable_navigation = library_unavailable_navigation
    screen._navigation_controller = LibraryNavigationController(
        screen,
        invalidate_media_browse=lambda: None,
        unmount_collections_capture=lambda: None,
    )
    admission = library_unavailable_navigation._LibraryCharacterNavigationAdmission(
        route=LibraryUnavailableConversationsBrowse(
            UnresolvedConversationKey("test-profile", "conversation-1"),
            RoleplayReturnTarget.console_context_character(),
        ),
        database=object(),
        generation=4,
    )
    screen._navigation_controller.character_route = admission
    screen._navigation_controller.character_candidate = admission
    screen._library_navigation_context_generation = 4
    return_control = SimpleNamespace(display=True)

    def query_return_control(selector):
        assert selector == "#library-character-return"
        return [return_control]

    screen.query = query_return_control
    # (wave-7 task 1, retargeted by task 3) Every media name this test seeds
    # -- the two debounce timers, and three of the five settlement fields the
    # focus-disarm helper resets -- lives on `_media_state`. An
    # `object.__new__`/`__new__` screen skips `__init__`'s state
    # construction, hence the explicit seed, exactly like the
    # `_ingest_state`/`_prompts_state` seeds below.
    screen._media_state = LibraryMediaState()
    # (wave-6 task 3) The prompts search-debounce timer is a
    # `LibraryPromptsState` field, not a flat screen attribute -- the
    # screen's generated shim block was deleted in the prompts cleanup PR,
    # so `setattr`/`getattr` on the old flat name would arm and assert a
    # field the hook never reads. An `object.__new__` screen also skips
    # `__init__`'s state construction, hence the explicit seed.
    screen._prompts_state = LibraryPromptsState()
    prompts_timer = _RecordingTimer()
    screen._prompts_state.debounce_timer = prompts_timer
    timer_attrs = (
        "_library_list_entry_focus_timer",
        "_library_source_snapshot_timeout_timer",
    )
    timers = {}
    for attr in timer_attrs:
        timers[attr] = _RecordingTimer()
        setattr(screen, attr, timers[attr])
    # (wave-7 task 3) The two media debounce timers are `LibraryMediaState`
    # fields, not flat screen attributes -- the screen's generated shim block
    # was deleted in the media cleanup PR, so `setattr`/`getattr` on the old
    # flat names would arm and assert fields the hook never reads. Same
    # explicit-block treatment as the ingest and prompts timers below.
    media_timers = {}
    for media_timer_field in ("selection_timer", "filter_timer"):
        media_timers[media_timer_field] = _RecordingTimer()
        setattr(screen._media_state, media_timer_field, media_timers[media_timer_field])
    # (wave-5 merge) The ingest path-debounce timer is a
    # `LibraryIngestState` field, not a flat screen attribute -- the
    # screen's generated shim block was
    # deleted in the ingest cleanup PR, so `setattr`/`getattr` on the old
    # flat name would arm and assert a field the hook never reads. An
    # `object.__new__` screen also skips `__init__`'s state construction,
    # hence the explicit seed.
    screen._ingest_state = LibraryIngestState()
    ingest_timer = _RecordingTimer()
    screen._ingest_state.path_debounce_timer = ingest_timer
    # State the focus-disarm helper resets alongside its timer.
    screen._library_screen_suspended = False
    screen._library_list_entry_focus_generation = 0
    screen._library_pending_list_entry_focus = False
    screen._library_pending_list_entry_media_return = None
    screen._library_pending_list_entry_focus_anchor = None
    screen._media_state.return_settlement = None
    screen._media_state.last_exact_settlement = None
    screen._media_state.last_successful_settlement = None

    active_app_token = active_app.set(SimpleNamespace(screen=object()))
    try:
        LibraryScreen.on_screen_suspend(screen)
    finally:
        active_app.reset(active_app_token)

    assert screen._library_screen_suspended is True
    assert screen._navigation_controller.character_route is None
    assert screen._navigation_controller.character_candidate is None
    assert screen._library_navigation_context_generation == 5
    assert return_control.display is False
    for attr in timer_attrs:
        assert timers[attr].stopped, f"{attr} was not stopped"
        assert getattr(screen, attr) is None, f"{attr} was not cleared"
    for media_timer_field, media_timer in media_timers.items():
        assert media_timer.stopped, f"the media {media_timer_field} was not stopped"
        assert getattr(screen._media_state, media_timer_field) is None, (
            f"the media {media_timer_field} was not cleared"
        )
    assert ingest_timer.stopped, "the ingest path-debounce timer was not stopped"
    assert screen._ingest_state.path_debounce_timer is None, (
        "the ingest path-debounce timer was not cleared"
    )
    assert prompts_timer.stopped, "the prompts search-debounce timer was not stopped"
    assert screen._prompts_state.debounce_timer is None, (
        "the prompts search-debounce timer was not cleared"
    )
    assert notes_timer.stopped, "the notes autosave timer was not stopped"
    assert screen._notes_state.autosave_timer is None, (
        "the notes autosave timer was not cleared"
    )
