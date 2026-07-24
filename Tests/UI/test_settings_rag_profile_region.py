"""Screen-level coverage for the Settings > Library/RAG profile-manager region
(Task 2 of SP3).

Task-2 review Finding 1: ~450 new lines (profile picker + clone/rename/delete
+ set-active dirty-draft prompt + worker completion paths + first-paint
read-only rendering) shipped with zero screen-level tests. This file plus the
regression tests added to ``test_settings_rag_profile_adapter.py`` (Finding 3)
close that gap.

Two test styles are used, matching existing repo conventions:

- Sync-constructed ``SettingsScreen(app)`` instances (never mounted/piloted),
  the same pattern as
  ``test_settings_console_background_workbench_raw_scope_unrelated_save_includes_fallback``
  in ``test_settings_configuration_hub.py``. Any codepath that touches
  Textual's ``self.app`` property (``.notify``/``.push_screen``) needs a
  monkeypatched ``SettingsScreen.app`` -- see the ``fake_app`` fixture below;
  an un-mounted widget's ``.app`` raises ``NoActiveAppError`` otherwise.
- One full pilot test (``_build_test_app`` + ``DestinationHarness``) for the
  first-paint read-only rendering, since composing widgets standalone
  requires faking Textual's internal compose-stack bookkeeping, which is far
  more fragile than just mounting the real screen.
"""

import inspect
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Select

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _wait_for_settings_text,
    _wire_rag_profile_adapter,
)
from tldw_chatbook.RAG_Search.config_profiles import reset_profile_manager_cache
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)
from tldw_chatbook.UI.Screens.settings_screen import (
    RagProfileSwitchConfirmModal,
    SettingsScreen,
)


@pytest.fixture(autouse=True)
def _reset_profile_manager_cache_after_test():
    yield
    reset_profile_manager_cache()


class _FakeApp:
    """Minimal stand-in for Textual's ``self.app`` -- records notify/push_screen
    calls instead of requiring a running application context."""

    def __init__(self):
        self.notifications: list[tuple[str, str]] = []
        self.pushed_screens: list[tuple[object, object]] = []

    def notify(self, message, *, severity="information", **kwargs):
        self.notifications.append((message, severity))

    def push_screen(self, screen, callback=None):
        self.pushed_screens.append((screen, callback))

    def call_from_thread(self, fn, *args, **kwargs):
        """Stand-in for Textual's cross-thread marshalling: invokes the
        callback immediately (same idiom as test_console_mcp_approval.py),
        since these sync-constructed tests never span a real thread."""
        return fn(*args, **kwargs)


@pytest.fixture
def fake_app(monkeypatch):
    """Monkeypatch ``SettingsScreen.app`` (a class-level property override,
    auto-reverted by pytest's monkeypatch) so un-mounted screens can exercise
    ``self.app.notify``/``self.app.push_screen`` call sites."""
    app = _FakeApp()
    monkeypatch.setattr(SettingsScreen, "app", property(lambda self: app), raising=False)
    return app


def _dirty_library_rag_screen(app_instance) -> SettingsScreen:
    screen = SettingsScreen(app_instance)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    draft = SettingsDraft(category=SettingsCategoryId.LIBRARY_RAG)
    draft.set_value("default_top_k", 10, 12)
    screen._settings_drafts[SettingsCategoryId.LIBRARY_RAG] = draft
    return screen


# --- Finding 1: dirty-prompt routing (Set-active while a draft is dirty) ---


def _dirty_screen_with_switch_pushed(monkeypatch, tmp_path, fake_app):
    """Wire an isolated adapter, build a dirty-draft screen, select a
    different (non-active) profile, and click Set active -- returns the
    screen, the modal's dismiss callback, and the target profile id."""
    mgr, _profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    other = mgr.clone_profile("hybrid_basic", "Other RAG")
    mgr.save_profile(other)

    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)
    monkeypatch.setattr(screen, "_library_rag_selected_profile_id", lambda: other.id)

    button = Button(id="settings-library-rag-profile-set-active")
    screen.handle_library_rag_profile_set_active(Button.Pressed(button))

    assert len(fake_app.pushed_screens) == 1
    modal, callback = fake_app.pushed_screens[0]
    assert isinstance(modal, RagProfileSwitchConfirmModal)
    return screen, callback, other.id


def test_set_active_with_dirty_draft_pushes_confirm_modal(monkeypatch, tmp_path, fake_app):
    screen, _callback, _other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    # The push itself must not have side-effected the draft or dispatched
    # anything -- the modal is the ONLY thing that happened.
    assert SettingsCategoryId.LIBRARY_RAG in screen._settings_drafts
    assert screen._rag_profile_pending_activate is None


def test_confirm_modal_cancel_makes_no_dispatch_and_leaves_pending_clear(
    monkeypatch, tmp_path, fake_app
):
    screen, callback, _other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    dispatched: list[str] = []
    screen._dispatch_rag_set_active = dispatched.append

    callback("cancel")

    assert dispatched == []
    assert screen._rag_profile_pending_activate is None
    # Draft is left untouched by Cancel.
    assert SettingsCategoryId.LIBRARY_RAG in screen._settings_drafts


def test_confirm_modal_discard_pops_draft_before_dispatching_set_active(
    monkeypatch, tmp_path, fake_app
):
    screen, callback, other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    calls: list[str] = []
    draft_present_at_dispatch: list[bool] = []

    def _spy_dispatch(profile_id):
        calls.append(profile_id)
        draft_present_at_dispatch.append(
            SettingsCategoryId.LIBRARY_RAG in screen._settings_drafts
        )

    screen._dispatch_rag_set_active = _spy_dispatch

    callback("discard")

    assert calls == [other_id]
    # Ordering: the draft must already be gone by the time dispatch runs.
    assert draft_present_at_dispatch == [False]
    assert SettingsCategoryId.LIBRARY_RAG not in screen._settings_drafts


def test_confirm_modal_save_arms_pending_activate_and_routes_through_save_action(
    monkeypatch, tmp_path, fake_app
):
    screen, callback, other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    save_calls: list[dict] = []
    screen.action_settings_save_category = lambda **kwargs: save_calls.append(kwargs)

    callback("save")

    assert screen._rag_profile_pending_activate == other_id
    assert save_calls == [{"allow_text_entry_focus": True}]


# --- Finding 2: `_rag_profile_pending_activate` must not leak past an
# early return in the Save action's LIBRARY_RAG branch. ---


def test_pending_activate_cleared_on_validation_failure(monkeypatch, tmp_path, fake_app):
    """Regression for Finding 2: Set-active(dirty) -> Save -> validation
    fails -> action_settings_save_category returns BEFORE the save worker
    (the only prior clearing site, _apply_library_rag_save_result) ever
    runs. Without the fix this pending id would silently fire a profile
    switch on a later, unrelated successful save.
    """
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    draft = SettingsDraft(category=SettingsCategoryId.LIBRARY_RAG)
    draft.set_value("default_top_k", 10, 0)  # 0 fails validation (min 1)
    screen._settings_drafts[SettingsCategoryId.LIBRARY_RAG] = draft
    screen._rag_profile_pending_activate = "some-other-profile-id"

    screen.action_settings_save_category(allow_text_entry_focus=True)

    assert screen._rag_profile_pending_activate is None


def test_pending_activate_cleared_when_no_unsaved_changes(monkeypatch, tmp_path, fake_app):
    """Same leak, via the OTHER early return in the LIBRARY_RAG save branch."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    # No draft staged at all -> _category_has_unsaved_changes is False.
    screen._rag_profile_pending_activate = "some-other-profile-id"

    screen.action_settings_save_category(allow_text_entry_focus=True)

    assert screen._rag_profile_pending_activate is None


def test_pending_activate_survives_into_worker_dispatch_on_valid_save(
    monkeypatch, tmp_path, fake_app
):
    """The capture-and-rearm fix must not break the legitimate path: a valid
    save still carries the pending id through to the worker dispatch so
    _apply_library_rag_save_result can fire the deferred switch."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)  # valid dirty value (12)
    screen._rag_profile_pending_activate = "target-profile-id"
    worker_calls: list[tuple] = []
    # Task 4 (SP3) added a second positional arg (`index_will_change`) to the
    # worker dispatch -- capture the full call, not just a single value.
    screen._settings_save_library_rag_worker = lambda *args: worker_calls.append(args)

    screen.action_settings_save_category(allow_text_entry_focus=True)

    assert screen._rag_profile_pending_activate == "target-profile-id"
    assert len(worker_calls) == 1


# --- Worker completion path: `_rag_after_set_active` ---


def test_after_set_active_success_clears_draft_and_notifies(monkeypatch, tmp_path, fake_app):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)

    screen._rag_after_set_active(True, "")

    assert SettingsCategoryId.LIBRARY_RAG not in screen._settings_drafts
    assert fake_app.notifications
    assert fake_app.notifications[-1][1] == "information"


def test_after_set_active_failure_syncs_profile_widgets_and_notifies_error(
    monkeypatch, tmp_path, fake_app
):
    """Finding 4: a failed set-active must still resync the profile Select
    (it may already show the user's failed target choice) back to the real
    active profile, not just report the error."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    sync_calls: list[bool] = []
    screen._sync_library_rag_profile_widgets = lambda: sync_calls.append(True)

    screen._rag_after_set_active(False, "disk full")

    assert sync_calls == [True]
    assert fake_app.notifications[-1] == (
        "Couldn't switch active profile: disk full",
        "error",
    )


# --- First-paint read-only rendering (pilot: real compose/mount) ---


@pytest.mark.asyncio
async def test_library_rag_detail_renders_fields_disabled_for_readonly_active_profile(
    monkeypatch, tmp_path
):
    """A built-in active profile (e.g. a brand-new install's default) must
    render every editable field disabled from the very FIRST paint, not just
    after a later set-active/clone/rename/delete resync."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path, active_id="hybrid_basic")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-library-rag-search-mode", Select).disabled
        assert screen.query_one("#settings-library-rag-default-top-k", Input).disabled
        assert screen.query_one("#settings-library-rag-fts-top-k", Input).disabled
        assert screen.query_one("#settings-library-rag-vector-top-k", Input).disabled
        assert screen.query_one("#settings-library-rag-hybrid-alpha", Input).disabled
        assert screen.query_one("#settings-library-rag-score-threshold", Input).disabled
        assert screen.query_one("#settings-library-rag-include-citations", Button).disabled
        assert screen.query_one("#settings-library-rag-citation-style", Select).disabled
        assert screen.query_one("#settings-library-rag-snippet-max-chars", Input).disabled
        assert screen.query_one("#settings-library-rag-max-context-size", Input).disabled


# --- Task 4 (SP3): index status readout + Backfill + honest re-index warnings ---


@pytest.mark.asyncio
async def test_library_rag_index_status_worker_updates_the_static(
    monkeypatch, tmp_path
):
    """The off-thread status fetch dispatched on category show (see
    _select_category -> _refresh_library_rag_index_status) populates the
    status row imperatively via _apply_library_rag_index_status, never
    during compose (which only ever renders the "checking…" placeholder)."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    fake_status = {
        "state": "built",
        "count": 42,
        "provenance": {
            "embedding_model": "mxbai-embed-large-v1",
            "chunk_size": 400,
            "chunk_overlap": 100,
        },
    }
    monkeypatch.setattr(
        settings_screen_module, "fetch_index_status", lambda: fake_status
    )

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        expected = (
            "Index: built · 42 vectors · built with mxbai-embed-large-v1 / "
            "chunk 400·100"
        )
        assert expected in _visible_text(screen)
        assert screen._library_rag_index_status_text == expected


@pytest.mark.asyncio
async def test_library_rag_save_with_index_change_includes_the_warning(
    monkeypatch, tmp_path
):
    """Save-path trigger (a): editing an index-determining field (chunk
    size) and saving must surface the shared honest re-index warning
    alongside the success notification (index_change_pending computed
    before the save mutates the profile)."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(
        settings_screen_module,
        "fetch_index_status",
        lambda: {"state": "absent", "count": 0, "provenance": {}},
    )

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        chunk_size = screen.query_one("#settings-library-rag-chunk-size", Input)
        chunk_size.value = str(profile.rag_config.chunking.chunk_size + 50)
        screen.handle_library_rag_chunk_size_changed(
            Input.Changed(chunk_size, chunk_size.value)
        )

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Library/RAG defaults saved.")

        assert (
            "This change re-points to a new (empty) index — run Backfill."
            in _visible_text(screen)
        )


@pytest.mark.asyncio
async def test_library_rag_save_without_index_change_omits_the_warning(
    monkeypatch, tmp_path
):
    """Save-path trigger (a), negative case: a query-time-only field
    (default_top_k, not in the fingerprint's index-determining set) must
    never surface the re-index warning."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(
        settings_screen_module,
        "fetch_index_status",
        lambda: {"state": "built", "count": 1, "provenance": {}},
    )

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        top_k = screen.query_one("#settings-library-rag-default-top-k", Input)
        top_k.value = str(profile.rag_config.search.default_top_k + 1)
        screen.handle_library_rag_default_top_k_changed(
            Input.Changed(top_k, top_k.value)
        )

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Library/RAG defaults saved.")

        assert (
            "This change re-points to a new (empty) index — run Backfill."
            not in _visible_text(screen)
        )


def test_backfill_button_click_starts_a_worker_and_notifies(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    worker_calls: list[bool] = []
    screen._rag_backfill_worker = lambda: worker_calls.append(True)

    button = Button(id="settings-library-rag-index-backfill")
    screen.handle_library_rag_index_backfill(Button.Pressed(button))

    assert screen._library_rag_backfill_in_flight is True
    assert worker_calls == [True]
    assert fake_app.notifications[-1][1] == "information"


def test_backfill_button_click_while_in_flight_does_not_start_a_second_worker(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    screen._library_rag_backfill_in_flight = True
    worker_calls: list[bool] = []
    screen._rag_backfill_worker = lambda: worker_calls.append(True)

    button = Button(id="settings-library-rag-index-backfill")
    screen.handle_library_rag_index_backfill(Button.Pressed(button))

    assert worker_calls == []
    assert fake_app.notifications[-1] == ("Backfill is already running.", "warning")


# --- Task 4 review Finding 1: backfill worker must be thread-isolated, not
# an async worker awaiting on the UI event loop (backfill_semantic_index has
# long synchronous stretches between awaits that would otherwise freeze the
# whole TUI). Mirrors SearchRAGWindow._run_index_backfill's thread + transient
# asyncio.run pattern. ---


def test_rag_backfill_worker_is_dispatched_as_a_thread_worker():
    """Source-based check, same idiom as
    test_settings_library_rag_save_uses_exclusive_thread_worker in
    test_settings_configuration_hub.py: confirms the worker is decorated
    ``thread=True`` (not the async-on-UI-loop shape it originally shipped
    with) and that its body is a plain (non-coroutine) function, since
    ``asyncio.run`` -- not ``await`` -- drives ``backfill_semantic_index``
    now."""
    worker = SettingsScreen.__dict__["_rag_backfill_worker"]
    wrapped = getattr(worker, "__wrapped__", None)
    source = inspect.getsource(SettingsScreen)

    assert wrapped is not None
    assert not inspect.iscoroutinefunction(wrapped)
    assert (
        '@work(exclusive=True, thread=True, group="settings-rag-backfill")\n'
        "    def _rag_backfill_worker"
    ) in source


def test_rag_backfill_worker_failure_notifies_and_clears_in_flight_without_raising(
    monkeypatch, tmp_path, fake_app
):
    """The thread-worker body must never let an exception escape -- it's
    marshalled back to the UI thread as a notify (via the fake app's
    call_from_thread, invoked inline for this sync-constructed test) and
    the in-flight flag is still cleared, exactly like the pre-fix async
    worker's try/except/finally contract."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(
        settings_screen_module, "semantic_indexing_available", lambda: True
    )

    def _boom(*, media_db, chachanotes_db):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(settings_screen_module, "backfill_semantic_index", _boom)

    app_instance = SimpleNamespace(
        app_config={}, media_db=object(), chachanotes_db=None
    )
    screen = SettingsScreen(app_instance)
    screen._library_rag_backfill_in_flight = True

    worker = SettingsScreen.__dict__["_rag_backfill_worker"]
    wrapped = getattr(worker, "__wrapped__", worker)
    wrapped(screen)  # invoke the thread-body directly, bypassing @work dispatch

    assert screen._library_rag_backfill_in_flight is False
    message, severity = fake_app.notifications[-1]
    assert severity == "error"
    assert "Backfill failed" in message
    assert "kaboom" in message


# --- Task 4 review Finding 2: _rag_after_set_active must not misreport a
# transient status-read failure ("unknown") as "re-points to a new (empty)
# index" -- that's a false claim the index changed. ---


def test_after_set_active_with_absent_index_status_includes_the_warning(
    monkeypatch, tmp_path, fake_app
):
    """Regression lock for the genuine case: a truly absent/empty index
    still gets the honest re-index warning."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    screen._rag_after_set_active(
        True, "", {"state": "absent", "count": 0, "provenance": {}}
    )

    message, severity = fake_app.notifications[-1]
    assert settings_screen_module.RAG_INDEX_CHANGE_WARNING in message
    assert severity == "warning"


def test_after_set_active_with_unknown_index_status_shows_honest_notice_without_the_warning(
    monkeypatch, tmp_path, fake_app
):
    """Finding 2: fetch_index_status returns state="unknown" when the read
    itself failed (see its own except-fallback) -- it says nothing about
    whether the index actually changed, so the change-warning constant must
    NOT appear. A distinct, honest "status unavailable" notice is shown
    instead."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    screen._rag_after_set_active(
        True, "", {"state": "unknown", "count": 0, "provenance": {}}
    )

    message, severity = fake_app.notifications[-1]
    assert settings_screen_module.RAG_INDEX_CHANGE_WARNING not in message
    assert "unavailable" in message.lower()
