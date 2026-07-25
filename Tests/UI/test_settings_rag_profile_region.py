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
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static

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
    RagProfileNameModal,
    RagProfileSwitchConfirmModal,
    SettingsScreen,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


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


def test_confirm_modal_discard_from_a_preview_clears_the_preview_first(
    monkeypatch, tmp_path, fake_app
):
    """Task 4 review (Important, coherence): reaching this modal at all
    requires the Select to be on a DIFFERENT profile than active -- which
    is exactly what a genuine browse would have already put into preview.
    A stale preview must not survive into the (async) gap before the
    set-active worker completes: `_rag_preview_profile_id` must be cleared
    before the discard's bare resync, not after."""
    screen, callback, other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    screen._rag_preview_profile_id = other_id
    screen._dispatch_rag_set_active = lambda profile_id: None

    callback("discard")

    assert screen._rag_preview_profile_id is None


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


def test_confirm_modal_save_from_a_preview_clears_preview_and_the_save_actually_dispatches(
    monkeypatch, tmp_path, fake_app
):
    """Task 4 review CRITICAL regression: picking a non-active profile in
    the Select now ALWAYS enters preview (handle_library_rag_profile_
    select_changed), so "Set active" while dirty -> "Save" reaches this
    callback with `_rag_preview_profile_id` still armed at the target
    profile. Before the fix, `action_settings_save_category`'s own preview
    guard (added earlier for the plain Save button) would silently no-op
    THIS save too -- and since that guard sits above the
    `_rag_profile_pending_activate` capture-and-clear, the pending id would
    leak forever, ready to fire an unrelated later save's profile switch.
    This exercises the REAL `action_settings_save_category` (not
    monkeypatched) to prove the save actually reaches dispatch."""
    screen, callback, other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    # Simulate the genuine browse that got the user here: selecting `other`
    # in the dropdown already entered preview before "Set active" was ever
    # clicked.
    screen._rag_preview_profile_id = other_id
    dispatched: list[tuple] = []
    screen._confirm_reindex_then_save = lambda values, pending: dispatched.append(
        (values, pending)
    )

    callback("save")

    assert screen._rag_preview_profile_id is None
    # The save must have actually reached the reindex-confirm gate with the
    # correct pending-activate target -- not been silently blocked.
    assert len(dispatched) == 1
    _values, pending = dispatched[0]
    assert pending == other_id
    # And the pending id must NOT be left armed for a later, unrelated save
    # to pick up (the leak the reviewer flagged).
    assert screen._rag_profile_pending_activate is None
    assert not any(
        "Return to the active profile to save" in message
        for message, _severity in fake_app.notifications
    )


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


# --- M3 (SP3 final review): `_library_rag_invalid_field_key` must match
# RAGConfig.validate()'s ACTUAL wording (routed through the adapter's
# hard_config_errors()), not a Title Case prefix that message never uses. ---


def test_invalid_field_key_matches_chunk_overlap_wording(monkeypatch, tmp_path):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    draft = SettingsDraft(category=SettingsCategoryId.LIBRARY_RAG)
    # Stage chunk_overlap >= chunk_size (RAGConfig.validate() message:
    # "chunk_overlap must be less than chunk_size") -- must resolve to
    # chunk_overlap, not chunk_size (both substrings appear in that message).
    overlap_over_size = profile.rag_config.chunking.chunk_size
    draft.set_value(
        "chunk_overlap", profile.rag_config.chunking.chunk_overlap, overlap_over_size
    )
    screen._settings_drafts[SettingsCategoryId.LIBRARY_RAG] = draft

    assert screen._library_rag_invalid_field_key() == "chunk_overlap"


def test_invalid_field_key_matches_embedding_batch_size_wording(monkeypatch, tmp_path):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    draft = SettingsDraft(category=SettingsCategoryId.LIBRARY_RAG)
    # RAGConfig.validate() message: "embedding batch_size must be positive".
    draft.set_value(
        "embedding_batch_size", profile.rag_config.embedding.batch_size, 0
    )
    screen._settings_drafts[SettingsCategoryId.LIBRARY_RAG] = draft

    assert screen._library_rag_invalid_field_key() == "embedding_batch_size"


def test_invalid_field_key_matches_distance_metric_wording(monkeypatch, tmp_path):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    draft = SettingsDraft(category=SettingsCategoryId.LIBRARY_RAG)
    # RAGConfig.validate() message: "Unknown distance metric: bogus".
    # normalise_library_rag_distance_metric only runs at load time, and the
    # draft layer stages the raw staged value straight through -- "bogus"
    # reaches RAGConfig.validate() unmodified via apply_defaults_to_profile.
    draft.set_value(
        "distance_metric", profile.rag_config.vector_store.distance_metric, "bogus"
    )
    screen._settings_drafts[SettingsCategoryId.LIBRARY_RAG] = draft

    assert screen._library_rag_invalid_field_key() == "distance_metric"


def test_invalid_field_key_matches_chunk_size_wording(monkeypatch, tmp_path):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    draft = SettingsDraft(category=SettingsCategoryId.LIBRARY_RAG)
    # RAGConfig.validate() message: "chunk_size must be positive". Also stage
    # chunk_overlap down to 0 so it doesn't independently gate on its own
    # "cannot be negative"/"less than chunk_size" rule ahead of this one.
    draft.set_value("chunk_size", profile.rag_config.chunking.chunk_size, 0)
    draft.set_value("chunk_overlap", profile.rag_config.chunking.chunk_overlap, 0)
    screen._settings_drafts[SettingsCategoryId.LIBRARY_RAG] = draft

    assert screen._library_rag_invalid_field_key() == "chunk_size"


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


# --- UX review item 1 (P0, clone flow): a successful clone must land the
# user ON the new clone (picker selection) with an actionable next step --
# not silently snap back to whatever was active (the pre-fix behaviour:
# clone_profile_as's returned new id was discarded entirely, and
# _sync_library_rag_profile_widgets always re-selected active_id). ---


def test_after_profile_action_clone_selects_the_clone_and_prompts_set_active(
    monkeypatch, tmp_path, fake_app
):
    mgr, _profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    clone = mgr.clone_profile("hybrid_basic", "My Clone")
    mgr.save_profile(clone)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    sync_widgets_calls: list[bool] = []
    sync_profile_calls: list[dict] = []
    screen._sync_library_rag_widgets = lambda: sync_widgets_calls.append(True)
    screen._sync_library_rag_profile_widgets = lambda **kwargs: sync_profile_calls.append(
        kwargs
    )

    # `result` is clone_profile_as's own return shape on success: the new
    # profile's id (see settings_rag_profile_adapter.clone_profile_as).
    screen._rag_after_profile_action("clone", True, clone.id)

    assert sync_widgets_calls == [True]
    assert sync_profile_calls == [{"select_override": clone.id}]
    message, severity = fake_app.notifications[-1]
    assert message == "Cloned to 'My Clone'. Select 'Set active' to edit it."
    assert severity == "information"


def test_after_profile_action_rename_and_delete_still_call_sync_with_no_override(
    monkeypatch, tmp_path, fake_app
):
    """Non-clone actions must keep calling
    ``_sync_library_rag_profile_widgets()`` with NO arguments -- the
    ``select_override`` kwarg is clone-only. Guards the exact call shape the
    pre-existing delete tests (see above) already monkeypatch a zero-arg
    lambda for."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    sync_profile_calls: list[dict] = []
    screen._sync_library_rag_profile_widgets = lambda **kwargs: sync_profile_calls.append(
        kwargs
    )

    screen._rag_after_profile_action("rename", True, "")
    screen._rag_after_profile_action("delete", True, "")

    assert sync_profile_calls == [{}, {}]


@pytest.mark.asyncio
async def test_clone_success_selects_the_clone_in_the_real_select_widget(
    monkeypatch, tmp_path
):
    """Full-mount regression lock for item 1: after a clone completes, the
    ACTUAL profile Select widget's value is the clone's id (not snapped
    back to the still-active source profile)."""
    mgr, _profile, _state = _wire_rag_profile_adapter(
        monkeypatch, tmp_path, active_id="hybrid_basic"
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        toasts = []
        host.notify = lambda message, **kwargs: toasts.append((message, kwargs))

        clone = mgr.clone_profile("hybrid_basic", "My Clone")
        mgr.save_profile(clone)
        screen._rag_after_profile_action("clone", True, clone.id)
        await pilot.pause()

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        assert select.value == clone.id
        assert toasts, "clone produced no toast"
        message, kwargs = toasts[-1]
        assert "Set active" in message
        # The active profile hasn't changed (only the picker's highlight
        # has) -- the decoupling caption (item 2) must still name the
        # original active profile, not the clone.
        assert "Editing: Hybrid Basic." in _visible_text(screen)


# --- UX review item 2 (P0 root cause, decoupling caption): the profile
# Select lets a user BROWSE profiles without editing them -- only "Set
# active" actually switches which profile the fields below edit. The
# caption must always name the ACTIVE profile, never whatever the Select
# happens to be showing. ---


@pytest.mark.asyncio
async def test_editing_caption_names_the_active_profile_not_the_select_value(
    monkeypatch, tmp_path
):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    other = mgr.clone_profile("hybrid_basic", "Other RAG")
    mgr.save_profile(other)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        assert (
            f"Editing: {profile.name}. Pick a profile and press 'Set active' "
            "to edit a different one."
        ) in _visible_text(screen)

        # Browse to a different profile in the dropdown WITHOUT pressing
        # "Set active" -- the caption must not follow the Select.
        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other.id
        await pilot.pause()

        assert f"Editing: {profile.name}." in _visible_text(screen)
        assert "Editing: Other RAG." not in _visible_text(screen)


@pytest.mark.asyncio
async def test_sync_profile_widgets_caption_ignores_select_override(
    monkeypatch, tmp_path
):
    """The caption must read the ACTIVE profile even when
    ``_sync_library_rag_profile_widgets`` is called with a
    ``select_override`` (the clone-flow case, item 1) -- the override only
    steers the Select, never the caption."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    other = mgr.clone_profile("hybrid_basic", "Other RAG")
    mgr.save_profile(other)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        screen._sync_library_rag_profile_widgets(select_override=other.id)
        await pilot.pause()

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        assert select.value == other.id
        assert f"Editing: {profile.name}." in _visible_text(screen)


# --- I1 (SP3 final review): `_rag_after_profile_action`'s delete branch
# must surface the adapter's hybrid_basic-fallback note and still resync the
# profile widgets (the Select may now be showing a deleted id / the picker
# needs the new active id highlighted). ---


def test_after_profile_action_delete_with_fallback_note_resyncs_and_notifies(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    sync_widgets_calls: list[bool] = []
    sync_profile_calls: list[bool] = []
    screen._sync_library_rag_widgets = lambda: sync_widgets_calls.append(True)
    screen._sync_library_rag_profile_widgets = lambda: sync_profile_calls.append(True)

    screen._rag_after_profile_action(
        "delete", True, "Active profile is now Hybrid Basic."
    )

    assert sync_widgets_calls == [True]
    assert sync_profile_calls == [True]
    message, severity = fake_app.notifications[-1]
    assert message == "Profile deleted. Active profile is now Hybrid Basic."
    assert severity == "information"


def test_after_profile_action_delete_without_fallback_note_omits_it(
    monkeypatch, tmp_path, fake_app
):
    """Non-active delete: the adapter's `result` is "" -- the notify text
    must stay the plain "Profile deleted." (no trailing space/empty note)."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    screen._rag_after_profile_action("delete", True, "")

    message, severity = fake_app.notifications[-1]
    assert message == "Profile deleted."
    assert severity == "information"


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
        assert screen.query_one(
            "#settings-library-rag-include-citations", Checkbox
        ).disabled
        assert screen.query_one("#settings-library-rag-citation-style", Select).disabled
        assert screen.query_one("#settings-library-rag-snippet-max-chars", Input).disabled
        assert screen.query_one("#settings-library-rag-max-context-size", Input).disabled


@pytest.mark.asyncio
async def test_rerank_fields_disabled_for_readonly_builtin_omit_reranking_suffix(
    monkeypatch, tmp_path
):
    """Review fix (AC #4): a builtin read-only active profile with reranking
    OFF (e.g. a fresh install's default `hybrid_basic`, and 9 of the 12
    builtins) must show the rerank Inputs disabled WITHOUT the
    "(enable reranking to edit)" suffix -- that instruction is unactionable
    here since the Enable-reranking checkbox is itself disabled by the
    builtin lock, not by the user's own choice. The suffix is reserved for
    the case where reranking-off is the ACTUAL, user-actionable reason."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path, active_id="hybrid_basic")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        model_input = screen.query_one("#settings-library-rag-reranker-model", Input)
        top_k_input = screen.query_one("#settings-library-rag-reranker-top-k", Input)
        assert model_input.disabled is True
        assert top_k_input.disabled is True
        visible_text = _visible_text(screen)
        assert "(enable reranking to edit)" not in visible_text
        assert "Reranker model" in visible_text
        assert "Rerank results" in visible_text


# --- Task 1 (RAG settings v2 UX, AC #4): the citations/reranking toggles are
# real Checkboxes (not Buttons whose label just says "Enabled"/"Disabled"),
# and the rerank model/results Inputs are dimmed -- never hidden -- whenever
# reranking itself is off, distinct from the builtin read-only lock. ---


@pytest.mark.asyncio
async def test_citation_and_reranking_toggles_are_checkboxes_mirroring_loaded_values(
    monkeypatch, tmp_path
):
    from tldw_chatbook.RAG_Search.reranker import RerankingConfig

    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.rag_config.search.include_citations = False
    profile.reranking_config = RerankingConfig()
    mgr.save_profile(profile)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        citations_checkbox = screen.query_one(
            "#settings-library-rag-include-citations", Checkbox
        )
        rerank_checkbox = screen.query_one(
            "#settings-library-rag-enable-reranking", Checkbox
        )
        assert citations_checkbox.value is False
        assert rerank_checkbox.value is True


@pytest.mark.asyncio
async def test_rerank_fields_dimmed_when_reranking_off_and_re_enable_on_toggle(
    monkeypatch, tmp_path
):
    """The default hybrid_basic clone has no reranking_config (reranking
    off) -- the rerank model/results Inputs must compose disabled with the
    "(enable reranking to edit)" suffix, and toggling the checkbox on must
    immediately re-enable them (live, before any Save)."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        model_input = screen.query_one("#settings-library-rag-reranker-model", Input)
        top_k_input = screen.query_one("#settings-library-rag-reranker-top-k", Input)
        assert model_input.disabled is True
        assert top_k_input.disabled is True
        assert (
            "Reranker model (enable reranking to edit)" in _visible_text(screen)
        )
        assert (
            "Rerank results (enable reranking to edit)" in _visible_text(screen)
        )

        rerank_checkbox = screen.query_one(
            "#settings-library-rag-enable-reranking", Checkbox
        )
        screen.handle_library_rag_enable_reranking_changed(
            Checkbox.Changed(rerank_checkbox, True)
        )
        await pilot.pause()

        assert model_input.disabled is False
        assert top_k_input.disabled is False
        assert "(enable reranking to edit)" not in _visible_text(screen)

        # And back off again -- the dimming (and suffix) must reapply.
        screen.handle_library_rag_enable_reranking_changed(
            Checkbox.Changed(rerank_checkbox, False)
        )
        await pilot.pause()

        assert model_input.disabled is True
        assert top_k_input.disabled is True
        assert "(enable reranking to edit)" in _visible_text(screen)


@pytest.mark.asyncio
async def test_rerank_fields_stay_dimmed_after_a_profile_switch_resync(
    monkeypatch, tmp_path
):
    """Regression: `_sync_library_rag_profile_widgets` runs AFTER
    `_sync_library_rag_widgets` on every set-active/clone/rename/delete
    resync and used to blanket-set `disabled = read_only` for every field
    including the rerank Inputs -- silently re-enabling them after
    switching to a non-builtin profile with reranking off. Exercises the
    same resync path a real Set-active click uses."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    mgr.save_profile(profile)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        model_input = screen.query_one("#settings-library-rag-reranker-model", Input)
        assert model_input.disabled is True

        screen._sync_library_rag_widgets()
        screen._sync_library_rag_profile_widgets()
        await pilot.pause()

        assert model_input.disabled is True
        assert (
            "Reranker model (enable reranking to edit)" in _visible_text(screen)
        )


@pytest.mark.asyncio
async def test_toggling_include_citations_checkbox_stages_draft_value_aware(
    monkeypatch, tmp_path
):
    """Retargets the pre-existing Button.Pressed dirty-marking assertions
    (see test_settings_library_rag_renders_guided_defaults_and_validates in
    test_settings_configuration_hub.py) at the new Checkbox.Changed handler:
    toggling away from the loaded value stages a draft, and staging the SAME
    value the profile already has must NOT mark the category dirty."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.rag_config.search.include_citations = True
    mgr.save_profile(profile)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-save-category", Button).disabled is True

        checkbox = screen.query_one(
            "#settings-library-rag-include-citations", Checkbox
        )
        screen.handle_library_rag_include_citations_changed(
            Checkbox.Changed(checkbox, False)
        )

        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert "Unsaved" in _visible_text(screen)

        # Staging the SAME value as loaded must clear the draft again
        # (value-aware -- exactly like the old Button.Pressed path).
        screen.handle_library_rag_include_citations_changed(
            Checkbox.Changed(checkbox, True)
        )

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert "No unsaved changes" in _visible_text(screen)


# --- UX review item 6 (P2, imported-settings provenance): the active
# profile's own description renders as a dim sub-line under "Active: <name>"
# when non-empty -- most useful for a first-run "Imported settings" snapshot
# ("Snapshot of your active RAG profile... edit freely"), which is
# otherwise indistinguishable from a hand-authored profile. ---


@pytest.mark.asyncio
async def test_active_profile_description_renders_as_a_subline_when_present(
    monkeypatch, tmp_path
):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.description = (
        "Snapshot of your active RAG profile (plus any RAG_* env "
        "overrides) at first run -- edit freely."
    )
    mgr.save_profile(profile)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        row = screen.query_one(
            "#settings-library-rag-active-profile-description", Static
        )
        assert row.display is True
        assert "Snapshot of your active RAG profile" in _visible_text(screen)


@pytest.mark.asyncio
async def test_active_profile_description_hidden_when_blank(monkeypatch, tmp_path):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.description = ""
    mgr.save_profile(profile)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        row = screen.query_one(
            "#settings-library-rag-active-profile-description", Static
        )
        assert row.display is False


@pytest.mark.asyncio
async def test_active_profile_description_resyncs_after_set_active(
    monkeypatch, tmp_path
):
    """Switching the active profile must refresh the sub-line to the NEW
    active profile's description, not leave the previous one showing."""
    mgr, profile, state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.description = "Original description."
    mgr.save_profile(profile)
    other = mgr.clone_profile("hybrid_basic", "Other RAG")
    other.description = "Other description."
    mgr.save_profile(other)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        assert "Original description." in _visible_text(screen)

        state["active"] = other.id
        screen._sync_library_rag_profile_widgets()
        await pilot.pause()

        assert "Other description." in _visible_text(screen)
        assert "Original description." not in _visible_text(screen)


# --- UX review item 5 (P2, ⚠ legend): the ⚠ markers scattered across
# individual field labels (Embedding model, Max length, Chunk size/overlap/
# method, Distance metric) are otherwise unexplained. ---


@pytest.mark.asyncio
async def test_warning_legend_is_rendered_under_the_profiles_block(
    monkeypatch, tmp_path
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        assert (
            "⚠ = changing this field rebuilds the index — run Backfill "
            "after saving." in _visible_text(screen)
        )


# --- UX review item 7 (P2, delete danger styling): the Delete button in the
# profile-manager row must read as destructive and be visually separated
# from Set active/Clone/Rename. ---


@pytest.mark.asyncio
async def test_delete_button_has_error_variant_and_spacer_class(monkeypatch, tmp_path):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        delete_button = screen.query_one(
            "#settings-library-rag-profile-delete", Button
        )
        assert delete_button.variant == "error"
        assert delete_button.has_class(
            "settings-library-rag-profile-delete-button"
        )


# --- Task 4 (SP3): index status readout + Backfill + honest re-index warnings ---

# --- UX review item 3 (P1, first-run Backfill nudge): a brand-new install's
# absent index must say WHY it matters (results are keyword-only) when the
# active profile's search mode actually needs the vector index, instead of
# the generic "will be created on next backfill" notice that reads as if
# nothing is missing yet. ---


def test_index_status_line_nudges_for_hybrid_mode_when_absent(
    monkeypatch, tmp_path, fake_app
):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.rag_config.search.default_search_mode = "hybrid"
    mgr.save_profile(profile)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    line = screen._library_rag_index_status_line({"state": "absent"})

    assert line == (
        "Semantic index not built — Hybrid search is keyword-only until "
        "you Backfill."
    )


def test_index_status_line_nudges_for_semantic_mode_when_absent(
    monkeypatch, tmp_path, fake_app
):
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.rag_config.search.default_search_mode = "semantic"
    mgr.save_profile(profile)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    line = screen._library_rag_index_status_line({"state": "absent"})

    assert line == (
        "Semantic index not built — Semantic search is keyword-only until "
        "you Backfill."
    )


def test_index_status_line_keeps_plain_notice_for_plain_mode_when_absent(
    monkeypatch, tmp_path, fake_app
):
    """A `plain`-mode profile never needs the vector index -- the generic
    notice stays, no nudge (the semantic/hybrid consequence doesn't apply)."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.rag_config.search.default_search_mode = "plain"
    mgr.save_profile(profile)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    line = screen._library_rag_index_status_line({"state": "absent"})

    assert line == settings_screen_module.RAG_INDEX_ABSENT_STATUS_TEXT


def test_index_status_line_ignores_search_mode_when_not_absent(
    monkeypatch, tmp_path, fake_app
):
    """The nudge is absent-state-only -- a built/empty index must keep its
    normal count/provenance rendering regardless of search mode."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.rag_config.search.default_search_mode = "hybrid"
    mgr.save_profile(profile)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    line = screen._library_rag_index_status_line({"state": "empty", "count": 0})

    assert line == "Index: empty · 0 vectors"


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


# --- UX review item 8 (P2, wire 't' for RAG): 't test category' used to
# fall all the way through to the generic "No test action is available..."
# toast for RAG. Now it refetches index status (same off-thread pattern as
# the other triggers) and reports it alongside the current preview
# defaults. ---


def test_test_category_action_dispatches_the_rag_check_worker_for_rag(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    worker_calls: list[bool] = []
    screen._rag_test_category_worker = lambda: worker_calls.append(True)

    screen.action_settings_test_category(allow_text_entry_focus=True)

    assert worker_calls == [True]
    all_messages = " ".join(m for m, _ in fake_app.notifications)
    assert "No test action is available" not in all_messages


def test_test_category_action_does_not_dispatch_rag_worker_for_other_categories(
    monkeypatch, tmp_path, fake_app
):
    """Regression guard: the new RAG branch must not swallow the existing
    generic fallback for a category with no test action."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.THEME.value
    worker_calls: list[bool] = []
    screen._rag_test_category_worker = lambda: worker_calls.append(True)

    screen.action_settings_test_category(allow_text_entry_focus=True)

    assert worker_calls == []
    assert fake_app.notifications[-1] == (
        "No test action is available for this Settings category yet.",
        "warning",
    )


def test_rag_test_category_worker_completion_notifies_state_and_preview(
    monkeypatch, tmp_path, fake_app
):
    """Invokes the thread-body directly (same idiom as the backfill-worker
    tests below), feeding a canned status through -- verifies the completion
    handler both refreshes the index-status Static AND notifies the honest
    one-line summary."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.rag_config.search.default_top_k = 15
    profile.rag_config.search.default_search_mode = "hybrid"
    profile.rag_config.search.include_citations = False
    mgr.save_profile(profile)
    monkeypatch.setattr(
        settings_screen_module,
        "fetch_index_status",
        lambda: {"state": "absent", "count": 0, "provenance": {}},
    )
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    worker = SettingsScreen.__dict__["_rag_test_category_worker"]
    wrapped = getattr(worker, "__wrapped__", worker)
    wrapped(screen)  # invoke the thread-body directly, bypassing @work dispatch

    message, severity = fake_app.notifications[-1]
    assert message.startswith("RAG check: absent index")
    assert "Hybrid search" in message  # the preview summary line
    assert severity == "information"
    assert (
        screen._library_rag_index_status_text
        == "Semantic index not built — Hybrid search is keyword-only "
        "until you Backfill."
    )


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
    # M5's pre-resolve call must never be the real (potentially heavy/
    # network-touching) service construction in a unit test.
    monkeypatch.setattr(
        settings_screen_module, "get_shared_rag_service", lambda: None
    )

    def _boom(*, media_db, chachanotes_db, rag_service=None):
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


# --- M5 (SP3 final review): the shared RAG service must be resolved OUTSIDE
# the transient asyncio.run loop and threaded through as the `rag_service`
# kwarg -- mirrors SearchRAGWindow._run_index_backfill's PR #700-hardened
# pattern (keeps first-time service construction from ever happening inside
# a loop that closes the instant this run finishes). ---


def test_rag_backfill_worker_pre_resolves_the_shared_service_outside_the_loop(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(
        settings_screen_module, "semantic_indexing_available", lambda: True
    )
    sentinel_service = object()
    resolve_calls: list[bool] = []

    def _fake_get_shared_rag_service():
        resolve_calls.append(True)
        return sentinel_service

    monkeypatch.setattr(
        settings_screen_module, "get_shared_rag_service", _fake_get_shared_rag_service
    )
    captured_kwargs: dict = {}

    async def _fake_backfill(**kwargs):
        captured_kwargs.update(kwargs)
        return {"status": "ok", "indexed": 0, "skipped": 0, "failed": 0, "errors": []}

    monkeypatch.setattr(
        settings_screen_module, "backfill_semantic_index", _fake_backfill
    )

    app_instance = SimpleNamespace(
        app_config={}, media_db=object(), chachanotes_db=None
    )
    screen = SettingsScreen(app_instance)

    worker = SettingsScreen.__dict__["_rag_backfill_worker"]
    wrapped = getattr(worker, "__wrapped__", worker)
    wrapped(screen)  # invoke the thread-body directly, bypassing @work dispatch

    assert resolve_calls == [True]
    assert captured_kwargs.get("rag_service") is sentinel_service


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


# --- Task 2 (541 v2 UX): pre-commit re-index confirmation. A save that
# would re-point the active profile at a fresh, EMPTY vector collection
# while the CURRENT collection is actually built (has vectors worth
# losing) must be confirmed BEFORE the save worker dispatches -- not just
# warned about after the fact (that's the existing RAG_INDEX_CHANGE_WARNING
# post-save notice, which still covers the absent/empty/unknown case). ---


def _dirty_screen_ready_for_reindex_gate(
    monkeypatch, tmp_path, fake_app, *, cached_status, pending_activate=None
):
    """Wire an isolated adapter, force `index_change_pending` True (so the
    gate always evaluates the STATUS branch regardless of which field the
    draft actually touches), seed the screen's cached index-status with
    `cached_status`, and stub the save worker so dispatch is observable
    without touching real profile files."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(
        settings_screen_module, "index_change_pending", lambda values: True
    )
    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)
    screen._library_rag_index_status_cache = cached_status
    if pending_activate is not None:
        screen._rag_profile_pending_activate = pending_activate
    worker_calls: list[tuple] = []
    screen._settings_save_library_rag_worker = lambda *args: worker_calls.append(args)
    return screen, worker_calls


def test_save_with_built_index_pushes_confirm_modal_with_count_and_backfill(
    monkeypatch, tmp_path, fake_app
):
    screen, worker_calls = _dirty_screen_ready_for_reindex_gate(
        monkeypatch,
        tmp_path,
        fake_app,
        cached_status={"state": "built", "count": 1234, "provenance": {}},
    )

    screen.action_settings_save_category(allow_text_entry_focus=True)

    # Nothing dispatched yet -- the modal gates the worker.
    assert worker_calls == []
    assert len(fake_app.pushed_screens) == 1
    modal, callback = fake_app.pushed_screens[0]
    assert isinstance(modal, ConfirmationDialog)
    assert "1234" in modal.message
    assert "Backfill" in modal.message


def test_reindex_confirm_cancel_does_not_save_and_clears_pending_activate(
    monkeypatch, tmp_path, fake_app
):
    screen, worker_calls = _dirty_screen_ready_for_reindex_gate(
        monkeypatch,
        tmp_path,
        fake_app,
        cached_status={"state": "built", "count": 1234, "provenance": {}},
        pending_activate="target-profile-id",
    )

    screen.action_settings_save_category(allow_text_entry_focus=True)
    _modal, callback = fake_app.pushed_screens[0]

    callback(False)

    assert worker_calls == []
    assert screen._rag_profile_pending_activate is None
    # The draft is left staged -- Cancel must not lose the user's edits.
    assert SettingsCategoryId.LIBRARY_RAG in screen._settings_drafts


def test_reindex_confirm_confirm_dispatches_save_and_rearms_pending_activate(
    monkeypatch, tmp_path, fake_app
):
    screen, worker_calls = _dirty_screen_ready_for_reindex_gate(
        monkeypatch,
        tmp_path,
        fake_app,
        cached_status={"state": "built", "count": 1234, "provenance": {}},
        pending_activate="target-profile-id",
    )

    screen.action_settings_save_category(allow_text_entry_focus=True)
    _modal, callback = fake_app.pushed_screens[0]

    callback(True)

    assert len(worker_calls) == 1
    values, index_will_change = worker_calls[0]
    assert index_will_change is True
    assert screen._rag_profile_pending_activate == "target-profile-id"


@pytest.mark.parametrize("state", ["absent", "empty"])
def test_save_with_index_change_but_nothing_built_skips_modal_and_saves_directly(
    monkeypatch, tmp_path, fake_app, state
):
    screen, worker_calls = _dirty_screen_ready_for_reindex_gate(
        monkeypatch,
        tmp_path,
        fake_app,
        cached_status={"state": state, "count": 0, "provenance": {}},
    )

    screen.action_settings_save_category(allow_text_entry_focus=True)

    assert fake_app.pushed_screens == []
    assert len(worker_calls) == 1
    values, index_will_change = worker_calls[0]
    assert index_will_change is True


def test_save_then_switch_reindex_confirm_survives_a_confirm(
    monkeypatch, tmp_path, fake_app
):
    """The save-then-switch flow (dirty prompt -> Save) must still complete
    end to end even when the save it defers to gets its own re-index
    confirm: pending-activate armed by "Save" on the switch prompt must
    survive a Confirm on the (second) reindex-confirm modal."""
    screen, callback, other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    monkeypatch.setattr(
        settings_screen_module, "index_change_pending", lambda values: True
    )
    screen._library_rag_index_status_cache = {
        "state": "built",
        "count": 7,
        "provenance": {},
    }
    worker_calls: list[tuple] = []
    screen._settings_save_library_rag_worker = lambda *args: worker_calls.append(args)

    callback("save")  # arms _rag_profile_pending_activate=other_id, calls Save

    assert worker_calls == []
    assert len(fake_app.pushed_screens) == 2
    reindex_modal, reindex_callback = fake_app.pushed_screens[-1]
    assert isinstance(reindex_modal, ConfirmationDialog)

    reindex_callback(True)

    assert len(worker_calls) == 1
    assert screen._rag_profile_pending_activate == other_id


def test_save_then_switch_reindex_confirm_survives_a_cancel(
    monkeypatch, tmp_path, fake_app
):
    """Same setup, Cancel branch: the deferred switch must NOT fire and
    _rag_profile_pending_activate must end up clear, not stuck armed."""
    screen, callback, _other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    monkeypatch.setattr(
        settings_screen_module, "index_change_pending", lambda values: True
    )
    screen._library_rag_index_status_cache = {
        "state": "built",
        "count": 7,
        "provenance": {},
    }
    worker_calls: list[tuple] = []
    screen._settings_save_library_rag_worker = lambda *args: worker_calls.append(args)

    callback("save")
    _reindex_modal, reindex_callback = fake_app.pushed_screens[-1]

    reindex_callback(False)

    assert worker_calls == []
    assert screen._rag_profile_pending_activate is None


def test_save_with_no_cached_status_fetches_then_confirms_when_built(
    monkeypatch, tmp_path, fake_app
):
    """Step 2: when nothing has been cached yet (category never shown /
    status never fetched), the gate dispatches its own off-thread fetch
    before deciding. `_FakeApp` can't run the real Textual worker
    machinery, so the dispatch call itself is stubbed (same idiom as
    ``test_backfill_button_click_starts_a_worker_and_notifies``) and the
    thread-body is invoked directly afterwards (same idiom as
    ``test_rag_test_category_worker_completion_notifies_state_and_preview``)
    to simulate the off-thread fetch completing."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(
        settings_screen_module, "index_change_pending", lambda values: True
    )
    monkeypatch.setattr(
        settings_screen_module,
        "fetch_index_status",
        lambda: {"state": "built", "count": 99, "provenance": {}},
    )
    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)
    assert screen._library_rag_index_status_cache is None
    worker_calls: list[tuple] = []
    screen._settings_save_library_rag_worker = lambda *args: worker_calls.append(args)
    fetch_dispatches: list[tuple] = []
    screen._rag_reindex_confirm_status_worker = lambda *args: fetch_dispatches.append(
        args
    )

    screen.action_settings_save_category(allow_text_entry_focus=True)

    # No modal from the (synchronous) action call itself -- only the fetch
    # was dispatched, off-thread.
    assert fake_app.pushed_screens == []
    assert len(fetch_dispatches) == 1
    values, pending_activate = fetch_dispatches[0]

    # Simulate the off-thread fetch completing by invoking the real
    # thread-body directly, bypassing @work's dispatch machinery.
    worker = SettingsScreen.__dict__["_rag_reindex_confirm_status_worker"]
    wrapped = getattr(worker, "__wrapped__", worker)
    wrapped(screen, values, pending_activate)

    assert worker_calls == []  # gated behind the modal, not dispatched yet
    assert screen._library_rag_index_status_cache == {
        "state": "built",
        "count": 99,
        "provenance": {},
    }
    assert len(fake_app.pushed_screens) == 1
    modal, callback = fake_app.pushed_screens[0]
    assert isinstance(modal, ConfirmationDialog)
    assert "99" in modal.message

    callback(True)

    assert len(worker_calls) == 1


def test_save_with_no_cached_status_fetches_then_saves_directly_when_absent(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(
        settings_screen_module, "index_change_pending", lambda values: True
    )
    monkeypatch.setattr(
        settings_screen_module,
        "fetch_index_status",
        lambda: {"state": "absent", "count": 0, "provenance": {}},
    )
    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)
    worker_calls: list[tuple] = []
    screen._settings_save_library_rag_worker = lambda *args: worker_calls.append(args)
    fetch_dispatches: list[tuple] = []
    screen._rag_reindex_confirm_status_worker = lambda *args: fetch_dispatches.append(
        args
    )

    screen.action_settings_save_category(allow_text_entry_focus=True)
    assert len(fetch_dispatches) == 1
    values, pending_activate = fetch_dispatches[0]

    worker = SettingsScreen.__dict__["_rag_reindex_confirm_status_worker"]
    wrapped = getattr(worker, "__wrapped__", worker)
    wrapped(screen, values, pending_activate)

    assert fake_app.pushed_screens == []
    assert len(worker_calls) == 1


# --- Task 2 review (Important): the cache-miss branch's off-thread status
# fetch must be debounced against a second Save click landing before the
# first fetch completes. Without a guard, a second click while
# `_library_rag_index_status_cache` is still None dispatches a SECOND
# `_rag_reindex_confirm_status_worker` in the same `exclusive=True` @work
# group as the first -- which CANCELS the first call, silently dropping ITS
# `pending_activate` (a function-local the cancelled call never hands back
# to anything: no notification, no error, the deferred profile switch just
# never happens). ---


def test_double_save_click_during_no_cache_fetch_does_not_drop_pending_activate(
    monkeypatch, tmp_path, fake_app
):
    screen, callback, other_id = _dirty_screen_with_switch_pushed(
        monkeypatch, tmp_path, fake_app
    )
    monkeypatch.setattr(
        settings_screen_module, "index_change_pending", lambda values: True
    )
    monkeypatch.setattr(
        settings_screen_module,
        "fetch_index_status",
        lambda: {"state": "built", "count": 3, "provenance": {}},
    )
    assert screen._library_rag_index_status_cache is None
    assert screen._rag_reindex_confirm_in_flight is False
    worker_calls: list[tuple] = []
    screen._settings_save_library_rag_worker = lambda *args: worker_calls.append(args)
    fetch_dispatches: list[tuple] = []
    screen._rag_reindex_confirm_status_worker = lambda *args: fetch_dispatches.append(
        args
    )

    callback("save")  # 1st Save click: arms pending_activate=other_id, cache miss

    assert len(fetch_dispatches) == 1
    assert screen._rag_reindex_confirm_in_flight is True

    # 2nd Save click while the 1st fetch is still "in flight" -- must be
    # debounced (no 2nd dispatch), or the real exclusive @work group would
    # cancel the 1st worker and drop its pending_activate.
    screen.action_settings_save_category(allow_text_entry_focus=True)
    assert len(fetch_dispatches) == 1

    # Complete the 1st (and only) fetch, simulating the off-thread worker
    # landing.
    values, pending_activate = fetch_dispatches[0]
    assert pending_activate == other_id
    worker = SettingsScreen.__dict__["_rag_reindex_confirm_status_worker"]
    wrapped = getattr(worker, "__wrapped__", worker)
    wrapped(screen, values, pending_activate)

    # The guard clears once the flow's decision is made. pushed_screens[0]
    # is the earlier RagProfileSwitchConfirmModal from `_dirty_screen_with_
    # switch_pushed`'s own Set-active click -- this is the SECOND push.
    assert screen._rag_reindex_confirm_in_flight is False
    assert len(fake_app.pushed_screens) == 2
    modal, modal_callback = fake_app.pushed_screens[-1]
    assert isinstance(modal, ConfirmationDialog)

    modal_callback(True)

    # The 1st click's pending_activate survived the double-click race.
    assert len(worker_calls) == 1
    assert screen._rag_profile_pending_activate == other_id

    # The guard is not stuck True: a LATER cache-miss window dispatches a
    # fresh fetch normally -- a subsequent Save is not bricked.
    screen._library_rag_index_status_cache = None
    screen._confirm_reindex_then_save(values, None)
    assert len(fetch_dispatches) == 2


def test_reindex_confirm_in_flight_cleared_on_cancel(monkeypatch, tmp_path, fake_app):
    """The in-flight guard must clear on the Cancel branch too (defensive
    -- by the time the modal resolves it's normally already cleared by the
    worker callback, but the handler clears it unconditionally as well)."""
    screen, worker_calls = _dirty_screen_ready_for_reindex_gate(
        monkeypatch,
        tmp_path,
        fake_app,
        cached_status={"state": "built", "count": 5, "provenance": {}},
    )
    screen._rag_reindex_confirm_in_flight = True

    screen.action_settings_save_category(allow_text_entry_focus=True)
    _modal, callback = fake_app.pushed_screens[0]

    callback(False)

    assert screen._rag_reindex_confirm_in_flight is False
    assert worker_calls == []


# --- Task 3 (541 v2 UX AC3): context-sensitive Scope Inspector guidance --
# the impact pane follows the focused RAG field / expanded Collapsible
# group instead of always showing the same static blurb. Mirrors the
# Providers-category machinery (_provider_field_guidance_rows /
# _refresh_provider_field_guidance / the DescendantFocus hook). ---


def _flattened_guidance_text(rows) -> str:
    return " ".join(f"{label} {value}" for label, value in rows).lower()


def test_rag_field_guidance_no_field_focused_matches_static_fallback(
    monkeypatch, tmp_path
):
    """RED case 1: with nothing focused and no group ever expanded (fresh
    first paint), the guidance must be byte-for-byte the same terse rows
    the UX review (item 9) shortened this to -- unchanged by this task."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    assert screen._active_settings_field_id is None
    assert screen._active_rag_scope_group is None
    assert screen._rag_field_guidance_rows() == (
        ("Search mode", "plain=keyword, semantic=embeddings, hybrid=blend"),
        ("Result limits", "bounds default/keyword/vector result counts"),
        ("Hybrid balance", "0.0=keyword, 1.0=semantic"),
        ("Citations", "adds source markers to answers when supported"),
        ("Snippet/context", "snippet length + context budget for retrieved text"),
    )
    # Calling the refresh on an unmounted screen must be a safe no-op
    # (query_one raises, caught the same way _refresh_provider_field_guidance
    # already handles it) rather than raising.
    screen._refresh_rag_field_guidance()


def test_rag_field_guidance_reranking_field_focused_mentions_reranking(
    monkeypatch, tmp_path
):
    """RED case 2: focusing a Reranking field yields guidance rows
    mentioning reranking (simulated the way the brief describes: set
    `_active_settings_field_id` directly, then call the refresh)."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    screen._active_settings_field_id = "settings-library-rag-reranker-model"
    screen._refresh_rag_field_guidance()

    text = _flattened_guidance_text(screen._rag_field_guidance_rows())
    assert "reranking" in text
    # Reranking is explicitly NOT index-determining -- must not tell the
    # user to Backfill (the guidance does say "no index rebuild", which is
    # the informative negative, not a rebuild instruction).
    assert "backfill" not in text


def test_rag_field_guidance_chunking_field_focused_mentions_reindex_backfill(
    monkeypatch, tmp_path
):
    """RED case 3: focusing a Chunking field yields re-index/backfill
    guidance (chunk_size/chunk_overlap/chunking_method are all
    index-determining -- see collection_fingerprint._index_fields())."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    for field_id in (
        "settings-library-rag-chunk-size",
        "settings-library-rag-chunk-overlap",
        "settings-library-rag-chunking-method",
    ):
        screen._active_settings_field_id = field_id
        text = _flattened_guidance_text(screen._rag_field_guidance_rows())
        assert "backfill" in text, field_id
        assert "rebuild" in text, field_id


def test_rag_field_guidance_embedding_and_vector_store_fields_flag_index_rebuild(
    monkeypatch, tmp_path
):
    """The other two index-determining groups (embedding model/max length,
    distance metric) also surface the Backfill/rebuild warning. Guidance is
    per-GROUP (one concise entry, not one per field -- see
    _RAG_GROUP_GUIDANCE), so device/batch-size (in the same Embedding
    group, but NOT themselves index-determining) show the identical
    group entry -- which is worded to name model/max length specifically
    as the ⚠ fields, not device/batch size."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    for field_id in (
        "settings-library-rag-embedding-model",
        "settings-library-rag-embedding-max-length",
        "settings-library-rag-embedding-device",
        "settings-library-rag-embedding-batch-size",
        "settings-library-rag-distance-metric",
    ):
        screen._active_settings_field_id = field_id
        text = _flattened_guidance_text(screen._rag_field_guidance_rows())
        assert "backfill" in text, field_id
        assert "rebuild" in text, field_id

    screen._active_settings_field_id = "settings-library-rag-embedding-device"
    embedding_text = _flattened_guidance_text(screen._rag_field_guidance_rows())
    # The group entry itself scopes the ⚠ down to model + max length, not
    # every field in the group.
    assert "model + max length" in embedding_text


def test_rag_field_guidance_profile_and_index_fields_have_dedicated_entries(
    monkeypatch, tmp_path
):
    """Profile controls (select/set-active/clone/rename/delete) and the
    index row (Backfill button) each get their own group entry -- not the
    generic search-mode fallback."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    screen._active_settings_field_id = "settings-library-rag-profile-set-active"
    profile_text = _flattened_guidance_text(screen._rag_field_guidance_rows())
    assert "profile" in profile_text

    screen._active_settings_field_id = "settings-library-rag-index-backfill"
    index_text = _flattened_guidance_text(screen._rag_field_guidance_rows())
    assert "index" in index_text
    assert "backfill" in index_text


def test_rag_field_guidance_focused_field_takes_priority_over_expanded_group(
    monkeypatch, tmp_path
):
    """A focused field's own group always wins over a merely-expanded
    group -- e.g. tabbing into the Reranking checkbox while Chunking is
    still expanded from an earlier click must show Reranking guidance."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    screen._active_rag_scope_group = "chunking"
    screen._active_settings_field_id = "settings-library-rag-enable-reranking"

    text = _flattened_guidance_text(screen._rag_field_guidance_rows())
    assert "reranking" in text
    assert "chunking" not in text


def test_every_rag_field_group_has_a_guidance_entry():
    """Coverage: no field-id -> group mapping can point at a group key
    with no `_RAG_GROUP_GUIDANCE` entry (a typo'd key would otherwise
    silently fall back to the generic guidance instead of failing loud)."""
    assert settings_screen_module._RAG_FIELD_GROUP_BY_ID
    for field_id, group in settings_screen_module._RAG_FIELD_GROUP_BY_ID.items():
        assert group in settings_screen_module._RAG_GROUP_GUIDANCE, (
            f"{field_id} -> {group!r} has no guidance entry"
        )


def test_every_library_rag_editable_field_id_has_a_guidance_group(
    monkeypatch, tmp_path
):
    """Coverage: every widget id `_library_rag_field_selector` resolves
    (the 19 validated/staged fields) plus the two Checkbox ids Task 1
    introduced must each map to a guidance group -- so no RAG control is
    ever focusable without the inspector explaining it."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)

    field_keys = [
        "default_search_mode",
        "default_top_k",
        "fts_top_k",
        "vector_top_k",
        "hybrid_alpha",
        "score_threshold",
        "citation_style",
        "snippet_max_chars",
        "max_context_size",
        "embedding_model",
        "embedding_device",
        "embedding_batch_size",
        "embedding_max_length",
        "chunk_size",
        "chunk_overlap",
        "chunking_method",
        "distance_metric",
        "reranker_model",
        "reranker_top_k",
    ]
    for key in field_keys:
        selector = screen._library_rag_field_selector(key)
        assert selector is not None, key
        widget_id = selector.removeprefix("#")
        assert widget_id in settings_screen_module._RAG_FIELD_GROUP_BY_ID, key

    for checkbox_id in (
        "settings-library-rag-include-citations",
        "settings-library-rag-enable-reranking",
    ):
        assert checkbox_id in settings_screen_module._RAG_FIELD_GROUP_BY_ID


@pytest.mark.asyncio
async def test_expanding_chunking_collapsible_switches_context_without_focus(
    monkeypatch, tmp_path
):
    """Expanding a group (without focusing any field inside it) already
    switches the inspector's context, via `@on(Collapsible.Toggled)`."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        # First paint: "Search" composes with collapsed=False, so Textual's
        # Collapsible posts its own Expanded message during construction
        # (see the comment on _RAG_GROUP_GUIDANCE_FALLBACK) -- the resolved
        # scope is "search" here, not None. That's harmless: the "search"
        # entry is the same fallback tuple, so the rendered text is
        # unaffected either way.
        assert screen._active_rag_scope_group == "search"
        chunking = screen.query_one(
            "#settings-library-rag-chunking-group", Collapsible
        )
        assert chunking.collapsed is True

        chunking.collapsed = False
        screen.handle_settings_library_rag_collapsible_toggled(
            Collapsible.Toggled(chunking)
        )
        await pilot.pause()

        assert screen._active_rag_scope_group == "chunking"
        await _wait_for_settings_text(screen, pilot, "Focused group: Chunking")
        text = _visible_text(screen)
        assert "Backfill" in text

        # Collapsing the same group again falls back to the static guidance.
        chunking.collapsed = True
        screen.handle_settings_library_rag_collapsible_toggled(
            Collapsible.Toggled(chunking)
        )
        await pilot.pause()

        assert screen._active_rag_scope_group is None
        await _wait_for_settings_text(screen, pilot, "Snippet/context:")


@pytest.mark.asyncio
async def test_focusing_reranker_model_input_updates_inspector_end_to_end(
    monkeypatch, tmp_path
):
    """End-to-end (not simulated): focusing a real mounted Reranking Input
    via the actual DescendantFocus path updates `_active_settings_field_id`
    and the rendered pane, exactly like the Providers category's own
    focus-follows-field behavior."""
    from tldw_chatbook.RAG_Search.reranker import RerankingConfig

    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    profile.reranking_config = RerankingConfig()
    mgr.save_profile(profile)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        reranker_input = screen.query_one(
            "#settings-library-rag-reranker-model", Input
        )
        assert reranker_input.disabled is False
        reranker_input.focus()
        await pilot.pause()

        assert (
            screen._active_settings_field_id
            == "settings-library-rag-reranker-model"
        )
        await _wait_for_settings_text(screen, pilot, "Focused group: Reranking")


# --- Task 4 (541 v2 UX AC1): manage-vs-edit split + read-only preview-on-
# select. Browsing the profile picker to a NON-active profile previews that
# profile's values read-only, WITHOUT staging a draft -- drafts belong to
# the active profile only. Selecting the active profile's own id again
# restores the ordinary, draft-aware editor (including any still-staged
# draft: the draft is never touched by preview). ---


def _wire_library_rag_with_other_profile(monkeypatch, tmp_path):
    """Wire an isolated adapter with a distinctive NON-active "Other RAG"
    profile and a fresh test app/harness, ready for a test to open the
    Library/RAG category and drive the Select with."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    other = mgr.clone_profile("hybrid_basic", "Other RAG")
    other.rag_config.search.default_top_k = 77
    mgr.save_profile(other)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    return mgr, profile, other, host


@pytest.mark.asyncio
async def test_browsing_to_a_non_active_profile_previews_it_read_only_without_staging(
    monkeypatch, tmp_path
):
    mgr, profile, other, host = _wire_library_rag_with_other_profile(
        monkeypatch, tmp_path
    )
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        top_k = screen.query_one("#settings-library-rag-default-top-k", Input)
        assert top_k.value == str(profile.rag_config.search.default_top_k)
        assert screen._rag_preview_profile_id is None

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other.id
        await pilot.pause()

        assert screen._rag_preview_profile_id == other.id
        assert top_k.value == "77"
        assert top_k.disabled is True
        # Every other editor field is ALSO forced disabled during preview,
        # not just the one field under test.
        assert (
            screen.query_one("#settings-library-rag-search-mode", Select).disabled
            is True
        )
        assert (
            screen.query_one("#settings-library-rag-enable-reranking", Checkbox).disabled
            is True
        )
        # No draft was created by merely browsing.
        assert SettingsCategoryId.LIBRARY_RAG not in screen._settings_drafts
        banner_text = _visible_text(screen)
        assert (
            "Previewing 'Other RAG' (read-only) — press Set active to edit it"
            in banner_text
        )


@pytest.mark.asyncio
async def test_field_changed_events_during_preview_never_stage_a_draft(
    monkeypatch, tmp_path
):
    mgr, profile, other, host = _wire_library_rag_with_other_profile(
        monkeypatch, tmp_path
    )
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other.id
        await pilot.pause()
        assert screen._rag_preview_profile_id == other.id

        top_k = screen.query_one("#settings-library-rag-default-top-k", Input)
        # Simulate a (disabled, but the handler must be robust regardless
        # of widget-level enforcement) edit attempt while previewing.
        screen.handle_library_rag_default_top_k_changed(
            Input.Changed(top_k, "999")
        )

        assert SettingsCategoryId.LIBRARY_RAG not in screen._settings_drafts
        assert screen._category_has_unsaved_changes(SettingsCategoryId.LIBRARY_RAG) is False


@pytest.mark.asyncio
async def test_returning_to_active_profile_restores_a_staged_draft_after_preview(
    monkeypatch, tmp_path
):
    """The hard round-trip: stage a draft on the ACTIVE profile, browse away
    (preview), browse back -- the staged value must still be showing AND
    still dirty. The draft is never touched by the preview round-trip."""
    mgr, profile, other, host = _wire_library_rag_with_other_profile(
        monkeypatch, tmp_path
    )
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        top_k = screen.query_one("#settings-library-rag-default-top-k", Input)
        top_k.value = "12"
        screen.handle_library_rag_default_top_k_changed(
            Input.Changed(top_k, top_k.value)
        )
        assert screen._category_has_unsaved_changes(SettingsCategoryId.LIBRARY_RAG) is True

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other.id
        await pilot.pause()
        assert screen._rag_preview_profile_id == other.id
        assert top_k.value == "77"

        select.value = profile.id
        await pilot.pause()

        assert screen._rag_preview_profile_id is None
        assert top_k.value == "12"
        assert top_k.disabled is False
        assert screen._category_has_unsaved_changes(SettingsCategoryId.LIBRARY_RAG) is True


@pytest.mark.asyncio
async def test_set_active_from_preview_exits_preview_on_success(monkeypatch, tmp_path):
    mgr, profile, other, host = _wire_library_rag_with_other_profile(
        monkeypatch, tmp_path
    )
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other.id
        await pilot.pause()
        assert screen._rag_preview_profile_id == other.id

        # Not dirty (no draft staged) -- handle_library_rag_profile_set_active
        # dispatches straight through, same as the pre-existing flow.
        screen.handle_library_rag_profile_set_active(
            Button.Pressed(Button(id="settings-library-rag-profile-set-active"))
        )
        await pilot.pause()
        # Simulate the worker's completion callback (existing test
        # convention -- see test_after_set_active_success_clears_draft_and_notifies).
        screen._rag_after_set_active(True, "")
        await pilot.pause()

        assert screen._rag_preview_profile_id is None
        top_k = screen.query_one("#settings-library-rag-default-top-k", Input)
        assert top_k.disabled is False


def test_set_active_from_preview_with_dirty_draft_still_pushes_confirm_modal(
    monkeypatch, tmp_path, fake_app
):
    """Set active from a preview must honor the EXISTING dirty-prompt flow
    exactly like a non-preview Set active does -- previewing a DIFFERENT
    profile never bypasses the "you have unsaved changes" gate for the
    ACTIVE profile's own draft."""
    mgr, _profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    other = mgr.clone_profile("hybrid_basic", "Other RAG")
    mgr.save_profile(other)
    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)
    screen._rag_preview_profile_id = other.id
    monkeypatch.setattr(screen, "_library_rag_selected_profile_id", lambda: other.id)

    button = Button(id="settings-library-rag-profile-set-active")
    screen.handle_library_rag_profile_set_active(Button.Pressed(button))

    assert len(fake_app.pushed_screens) == 1
    modal, _callback = fake_app.pushed_screens[0]
    assert isinstance(modal, RagProfileSwitchConfirmModal)
    # The preview flag itself is untouched by merely OPENING the prompt --
    # only a completed switch (_rag_after_set_active) clears it.
    assert screen._rag_preview_profile_id == other.id


def test_save_is_blocked_with_a_notification_while_previewing(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    screen._rag_preview_profile_id = "some-other-profile-id"
    dispatched: list[object] = []
    screen._confirm_reindex_then_save = lambda *a, **k: dispatched.append(True)

    screen.action_settings_save_category(allow_text_entry_focus=True)

    assert dispatched == []
    message, severity = fake_app.notifications[-1]
    assert message == "Return to the active profile to save."
    assert severity == "warning"


def test_revert_is_blocked_with_a_notification_while_previewing(
    monkeypatch, tmp_path, fake_app
):
    """Task 4 review IMPORTANT: `action_settings_revert_category` had no
    preview guard (unlike Save) -- pressing 'r' while previewing would pop
    the ACTIVE profile's own (unrelated) staged draft, a silent data-loss
    bug, and leave stale preview chrome on screen."""
    mgr, _profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)  # dirty draft on the ACTIVE profile
    screen._rag_preview_profile_id = "some-other-profile-id"

    screen.action_settings_revert_category(allow_text_entry_focus=True)

    # The draft must be intact -- revert must not have popped it.
    assert SettingsCategoryId.LIBRARY_RAG in screen._settings_drafts
    message, severity = fake_app.notifications[-1]
    assert message == "Return to the active profile to revert."
    assert severity == "warning"


def test_cloning_while_previewing_uses_the_previewed_profile_as_the_source(
    monkeypatch, tmp_path, fake_app
):
    """Task 4 review (cheap Clone-from-preview coverage): the Clone button
    resolves its source from `_library_rag_selected_profile_id()` -- the
    Select's current value, which is exactly the previewed profile while
    `_rag_preview_profile_id` is armed. Completing the clone must still
    clear the preview (the ok branch of `_rag_after_profile_action`)."""
    mgr, _profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    other = mgr.clone_profile("hybrid_basic", "Other RAG")
    mgr.save_profile(other)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    screen._rag_preview_profile_id = other.id
    monkeypatch.setattr(screen, "_library_rag_selected_profile_id", lambda: other.id)

    button = Button(id="settings-library-rag-profile-clone")
    screen.handle_library_rag_profile_clone(Button.Pressed(button))

    assert len(fake_app.pushed_screens) == 1
    _modal, callback = fake_app.pushed_screens[0]
    dispatched: list[tuple] = []
    screen._dispatch_rag_profile_action = lambda action, profile_id, arg: dispatched.append(
        (action, profile_id, arg)
    )

    callback("My Clone")

    assert dispatched == [("clone", other.id, "My Clone")]

    screen._rag_after_profile_action("clone", True, "new-clone-id")
    assert screen._rag_preview_profile_id is None


def test_library_rag_save_enabled_is_false_while_previewing_even_with_a_dirty_draft(
    monkeypatch, tmp_path
):
    mgr, _profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = _dirty_library_rag_screen(app)  # valid dirty value (12)
    assert screen._library_rag_save_enabled() is True

    screen._rag_preview_profile_id = "some-other-profile-id"

    assert screen._library_rag_save_enabled() is False


@pytest.mark.asyncio
async def test_profiles_and_editor_render_inside_their_own_titled_containers(
    monkeypatch, tmp_path
):
    mgr, profile, other, host = _wire_library_rag_with_other_profile(
        monkeypatch, tmp_path
    )
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        profiles_card = screen.query_one("#settings-library-rag-profiles-card")
        editor_card = screen.query_one("#settings-library-rag-editor-card")
        assert profiles_card.border_title == "Profiles"
        assert editor_card.border_title == f"Editing: {profile.name}"

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other.id
        await pilot.pause()

        assert editor_card.border_title == f"Previewing: {other.name}"

        select.value = profile.id
        await pilot.pause()

        assert editor_card.border_title == f"Editing: {profile.name}"


@pytest.mark.asyncio
async def test_preview_banner_and_title_escape_markup_significant_profile_names(
    monkeypatch, tmp_path
):
    """Repo lesson: profile names can contain markup-significant characters
    (e.g. `[bold]`) -- both the preview banner and the editor title must
    escape them rather than let Rich interpret them as markup tags."""
    mgr, profile, _state = _wire_rag_profile_adapter(monkeypatch, tmp_path)
    other = mgr.clone_profile("hybrid_basic", "[bold]Other[/bold]")
    mgr.save_profile(other)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other.id
        await pilot.pause()

        editor_card = screen.query_one("#settings-library-rag-editor-card")
        assert editor_card.border_title == r"Previewing: \[bold]Other\[/bold]"
        banner = screen.query_one("#settings-library-rag-preview-banner", Static)
        assert r"\[bold]Other\[/bold]" in str(banner.renderable)


def test_cloning_leaves_the_select_on_the_clone_without_auto_entering_preview(
    monkeypatch, tmp_path, fake_app
):
    """The profile-picker's own imperative resync
    (`_sync_library_rag_profile_widgets`, e.g. the clone flow's
    `select_override`) must never itself trigger a preview -- only a
    genuine user browse (a real Select.Changed from interacting with the
    dropdown) does. Sync-constructed (unmounted): the real Select never
    exists, so this exercises the state directly rather than the message
    cascade -- see the full-mount clone regression test above for the
    Select-widget-value side of this flow."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    assert screen._rag_preview_profile_id is None

    screen._rag_after_profile_action("clone", True, "some-clone-id")

    assert screen._rag_preview_profile_id is None


def test_leaving_the_library_rag_category_clears_a_stale_preview(monkeypatch, tmp_path):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    screen._rag_preview_profile_id = "some-other-profile-id"

    screen._select_category(SettingsCategoryId.STORAGE.value)

    assert screen._rag_preview_profile_id is None


# --- Task 5 (541 v2 UX AC5): first-run starter panel -- replaces the "wall
# of disabled fields" a brand-new install (builtin active, no user profiles,
# no vector index yet) used to show with a direct next step. The panel keys
# off the SAME cached index-status fetch the status row uses (never an extra
# fetch of its own), so its visibility is toggled from the SAME
# _apply_library_rag_index_status funnel every existing fetch trigger
# (category show / 't' test / backfill completion / set-active) already goes
# through -- see is_first_run_state in test_settings_rag_profile_adapter.py
# for the pure predicate itself. ---


def _stub_index_status(monkeypatch, state: str) -> None:
    monkeypatch.setattr(
        settings_screen_module,
        "fetch_index_status",
        lambda: {"state": state, "count": 0, "provenance": {}},
    )


def _wire_rag_profile_adapter_no_user_profiles(
    monkeypatch, tmp_path, *, active_id: str = "hybrid_basic"
):
    """Like `_wire_rag_profile_adapter`, but WITHOUT its always-present "My
    RAG" user-profile clone -- the first-run predicate specifically
    requires a genuinely EMPTY user-profile list, which the shared helper
    can never produce (it registers "My RAG" unconditionally, active or
    not)."""
    from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as rag_adapter_module

    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    state = {"active": active_id}
    monkeypatch.setattr(rag_adapter_module, "_manager", lambda: mgr, raising=False)
    monkeypatch.setattr(
        rag_adapter_module, "_active_profile_id", lambda: state["active"], raising=False
    )
    return mgr, state


@pytest.mark.asyncio
async def test_starter_panel_shown_when_builtin_active_no_users_and_index_absent(
    monkeypatch, tmp_path
):
    """The true first-run case: renders the exact copy naming the active
    profile, and the Search group -- the only one that composes expanded by
    default -- ends up collapsed alongside it (Embedding/Chunking/Vector
    store/Reranking are already collapsed by default)."""
    _wire_rag_profile_adapter_no_user_profiles(monkeypatch, tmp_path)
    _stub_index_status(monkeypatch, "absent")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = screen.query_one("#settings-library-rag-starter-panel")
        assert panel.display is True
        copy_text = str(
            screen.query_one("#settings-library-rag-starter-copy", Static).renderable
        )
        assert (
            "Search already works on Hybrid Basic. Clone it to tune retrieval, "
            "or run Backfill to enable semantic results." == copy_text
        )
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is True
        )
        clone_button = screen.query_one(
            "#settings-library-rag-starter-clone", Button
        )
        backfill_button = screen.query_one(
            "#settings-library-rag-starter-backfill", Button
        )
        assert str(clone_button.label) == "Clone to tune…"
        assert str(backfill_button.label) == "Backfill now"


@pytest.mark.asyncio
async def test_starter_panel_hidden_when_active_profile_is_not_a_builtin(
    monkeypatch, tmp_path
):
    """`_wire_rag_profile_adapter`'s default active profile is the writable
    "My RAG" clone -- never first-run even with an absent index and no
    OTHER user profiles."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    _stub_index_status(monkeypatch, "absent")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = screen.query_one("#settings-library-rag-starter-panel")
        assert panel.display is False
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is False
        )


@pytest.mark.asyncio
async def test_starter_panel_hidden_when_a_user_profile_already_exists(
    monkeypatch, tmp_path
):
    mgr, _state = _wire_rag_profile_adapter_no_user_profiles(monkeypatch, tmp_path)
    other = mgr.clone_profile("hybrid_basic", "Other RAG")
    mgr.save_profile(other)
    _stub_index_status(monkeypatch, "absent")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = screen.query_one("#settings-library-rag-starter-panel")
        assert panel.display is False


@pytest.mark.asyncio
async def test_starter_panel_hidden_when_the_index_is_already_built(
    monkeypatch, tmp_path
):
    _wire_rag_profile_adapter_no_user_profiles(monkeypatch, tmp_path)
    _stub_index_status(monkeypatch, "built")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = screen.query_one("#settings-library-rag-starter-panel")
        assert panel.display is False


@pytest.mark.asyncio
async def test_starter_panel_disappears_after_a_clone_completes(monkeypatch, tmp_path):
    """State-driven, no dismissal persistence: a successful clone gives the
    active builtin its first user profile, which falsifies the predicate on
    the very next sync -- no explicit "dismiss" affordance needed.

    Reviewer finding (541 v2 UX AC5, Important): the fix must be
    TRANSITION-gated, not unconditional -- the Search group is forced back
    OPEN here specifically because first-run just ENDED, not merely because
    a status happened to refresh. See
    ``test_normal_state_status_refresh_never_reopens_a_deliberately_collapsed_search_group``
    below for the companion guard."""
    mgr, _state = _wire_rag_profile_adapter_no_user_profiles(monkeypatch, tmp_path)
    _stub_index_status(monkeypatch, "absent")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#settings-library-rag-starter-panel").display is True
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is True
        )

        clone = mgr.clone_profile("hybrid_basic", "My Clone")
        mgr.save_profile(clone)
        screen._rag_after_profile_action("clone", True, clone.id)
        await pilot.pause()

        assert screen.query_one("#settings-library-rag-starter-panel").display is False
        # The user who just cloned to tune retrieval must land on an
        # editable, EXPANDED Search group -- not one still collapsed behind
        # a now-hidden starter panel.
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is False
        )


@pytest.mark.asyncio
async def test_starter_panel_disappears_after_backfill_completes(monkeypatch, tmp_path):
    """Backfill itself only fills the vector store; the fake status swap
    below stands in for "the fetch that runs right after Backfill now sees a
    non-absent index" -- exercised through the real
    _refresh_library_rag_index_status -> _rag_index_status_worker funnel the
    backfill worker's own completion already dispatches."""
    _wire_rag_profile_adapter_no_user_profiles(monkeypatch, tmp_path)
    _stub_index_status(monkeypatch, "absent")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#settings-library-rag-starter-panel").display is True
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is True
        )

        _stub_index_status(monkeypatch, "built")
        screen._refresh_library_rag_index_status()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert screen.query_one("#settings-library-rag-starter-panel").display is False
        # Same reviewer finding as the clone path above: Backfill completing
        # is exactly as much a first-run EXIT as a clone is -- the Search
        # group must not stay collapsed behind a now-hidden starter panel.
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is False
        )


@pytest.mark.asyncio
async def test_normal_state_status_refresh_never_reopens_a_deliberately_collapsed_search_group(
    monkeypatch, tmp_path
):
    """Guard for the transition-gating itself: a user already in NORMAL
    (non-first-run) state who deliberately collapses Search must not have it
    forcibly reopened by an ordinary status refresh (category re-show / Save
    / 't' test / set-active) that lands while first-run was never active this
    session. Only the actual first-run -> not-first-run TRANSITION (see the
    two tests above) may flip ``collapsed`` back to False."""
    # `_wire_rag_profile_adapter` (not the `_no_user_profiles` variant) seeds
    # a writable "My RAG" user profile as active -- never first-run, per
    # `is_first_run_state`.
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    _stub_index_status(monkeypatch, "built")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#settings-library-rag-starter-panel").display is False

        search_group = screen.query_one(
            "#settings-library-rag-search-group", Collapsible
        )
        search_group.collapsed = True

        # An unchanged, still-non-first-run status landing again (e.g. a
        # plain category re-show or Save-path refresh) must leave the
        # user's own collapse alone.
        screen._apply_library_rag_index_status(
            {"state": "built", "count": 3, "provenance": {}}
        )
        await pilot.pause()

        assert screen.query_one("#settings-library-rag-starter-panel").display is False
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is True
        )


@pytest.mark.asyncio
async def test_preview_started_while_starter_panel_visible_leaves_panel_state_coherent(
    monkeypatch, tmp_path
):
    """Reviewer-requested coverage: browsing the profile picker into a
    PREVIEW of a different builtin (via the Select's own Changed handler,
    exactly like a real user browse) while the first-run starter panel is
    showing must not crash or desync the panel -- previewing never touches
    the first-run predicate's own trigger funnel
    (`_apply_library_rag_index_status` / `_rag_after_profile_action`)."""
    _wire_rag_profile_adapter_no_user_profiles(monkeypatch, tmp_path)
    _stub_index_status(monkeypatch, "absent")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#settings-library-rag-starter-panel").display is True

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = "bm25_only"
        await pilot.pause()

        assert screen._rag_preview_profile_id == "bm25_only"
        assert screen.query_one("#settings-library-rag-starter-panel").display is True
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is True
        )


@pytest.mark.asyncio
async def test_set_active_to_another_first_run_eligible_builtin_keeps_panel_visible(
    monkeypatch, tmp_path
):
    """Reviewer-requested coverage: switching the active profile from one
    read-only builtin to ANOTHER read-only builtin, still with no user
    profiles and an absent index, stays first-run throughout
    (`is_first_run_state` doesn't care WHICH builtin is active) -- no
    spurious collapse/expand thrash across a non-transition."""
    _wire_rag_profile_adapter_no_user_profiles(
        monkeypatch, tmp_path, active_id="bm25_only"
    )
    _stub_index_status(monkeypatch, "absent")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#settings-library-rag-starter-panel").display is True

        screen._rag_after_set_active(
            True, "", {"state": "absent", "count": 0, "provenance": {}}
        )
        await pilot.pause()

        assert screen.query_one("#settings-library-rag-starter-panel").display is True
        assert (
            screen.query_one(
                "#settings-library-rag-search-group", Collapsible
            ).collapsed
            is True
        )


def test_starter_panel_clone_button_opens_the_same_clone_modal(
    monkeypatch, tmp_path, fake_app
):
    """The starter panel's "Clone to tune…" button reuses the EXACT same
    modal + dispatch path as the Profiles block's own "Clone…" button --
    never a bespoke implementation."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path, active_id="hybrid_basic")
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value

    button = Button(id="settings-library-rag-starter-clone")
    screen.handle_library_rag_starter_clone(Button.Pressed(button))

    assert len(fake_app.pushed_screens) == 1
    modal, _callback = fake_app.pushed_screens[0]
    assert isinstance(modal, RagProfileNameModal)
    assert modal._modal_title == "Clone profile"


def test_starter_panel_backfill_button_starts_the_same_backfill_worker(
    monkeypatch, tmp_path, fake_app
):
    """The starter panel's "Backfill now" button reuses the EXACT same
    in-flight guard + thread-worker dispatch as the Index row's own
    "Backfill" button."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path, active_id="hybrid_basic")
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    worker_calls: list[bool] = []
    screen._rag_backfill_worker = lambda: worker_calls.append(True)

    button = Button(id="settings-library-rag-starter-backfill")
    screen.handle_library_rag_starter_backfill(Button.Pressed(button))

    assert screen._library_rag_backfill_in_flight is True
    assert worker_calls == [True]
    assert fake_app.notifications[-1][1] == "information"


# --- Task 6 (541 AC6): keyboard accelerators for the profile workflow
# (Set active / Clone / Backfill), guarded to the LIBRARY_RAG category and
# to the SAME text-entry-focus check s/r/t already use. Each action
# delegates to the EXACT SAME trigger its button uses -- no bespoke
# reimplementation. ---


def test_settings_rag_set_active_action_dispatches_for_rag_category(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    calls: list[bool] = []
    screen._trigger_library_rag_profile_set_active = lambda: calls.append(True)

    screen.action_settings_rag_set_active(allow_text_entry_focus=True)

    assert calls == [True]


def test_settings_rag_clone_action_dispatches_for_rag_category(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    calls: list[bool] = []
    screen._trigger_library_rag_profile_clone = lambda: calls.append(True)

    screen.action_settings_rag_clone(allow_text_entry_focus=True)

    assert calls == [True]


def test_settings_rag_backfill_action_dispatches_for_rag_category(
    monkeypatch, tmp_path, fake_app
):
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    calls: list[bool] = []
    screen._trigger_library_rag_index_backfill = lambda: calls.append(True)

    screen.action_settings_rag_backfill(allow_text_entry_focus=True)

    assert calls == [True]


def test_settings_rag_accelerators_no_op_for_a_non_rag_category(
    monkeypatch, tmp_path, fake_app
):
    """Regression guard: unlike s/r/t (which dispatch per-category from a
    single shared action with a generic fallback), the RAG accelerators
    have no meaning outside LIBRARY_RAG -- they must produce zero side
    effects (no trigger call, no notification) for another category."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.THEME.value
    calls: list[str] = []
    screen._trigger_library_rag_profile_set_active = lambda: calls.append("set_active")
    screen._trigger_library_rag_profile_clone = lambda: calls.append("clone")
    screen._trigger_library_rag_index_backfill = lambda: calls.append("backfill")

    screen.action_settings_rag_set_active(allow_text_entry_focus=True)
    screen.action_settings_rag_clone(allow_text_entry_focus=True)
    screen.action_settings_rag_backfill(allow_text_entry_focus=True)

    assert calls == []
    assert fake_app.notifications == []


def test_settings_rag_accelerators_no_op_while_text_entry_has_focus(
    monkeypatch, tmp_path, fake_app
):
    """Same defense-in-depth guard s/r/t use (see
    `_settings_text_entry_has_focus`): a direct call while a text-entry
    widget is focused (not routed through `allow_text_entry_focus=True`)
    must be a no-op even with the RAG category active."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.LIBRARY_RAG.value
    monkeypatch.setattr(screen, "_settings_text_entry_has_focus", lambda: True)
    calls: list[str] = []
    screen._trigger_library_rag_profile_set_active = lambda: calls.append("set_active")
    screen._trigger_library_rag_profile_clone = lambda: calls.append("clone")
    screen._trigger_library_rag_index_backfill = lambda: calls.append("backfill")

    screen.action_settings_rag_set_active()
    screen.action_settings_rag_clone()
    screen.action_settings_rag_backfill()

    assert calls == []


@pytest.mark.asyncio
async def test_typing_accelerator_letters_into_the_top_k_input_does_not_fire_them(
    monkeypatch, tmp_path
):
    """Real end-to-end key dispatch (not a simulated action call), mirroring
    `test_settings_provider_text_inputs_do_not_trigger_footer_shortcuts`'s
    pattern for s/r/t: typing 'a'/'c'/'b' while the top-k Input is focused
    must not clone (or set-active/backfill) -- Input consumes printable
    keys (`Input._on_key` calls `event.stop()` unconditionally for any
    printable character, even one `restrict` then rejects) before they ever
    bubble to the screen's BINDINGS."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        calls: list[str] = []
        screen._trigger_library_rag_profile_set_active = lambda: calls.append(
            "set_active"
        )
        screen._trigger_library_rag_profile_clone = lambda: calls.append("clone")
        screen._trigger_library_rag_index_backfill = lambda: calls.append("backfill")

        top_k_input = screen.query_one(
            "#settings-library-rag-default-top-k", Input
        )
        original_value = top_k_input.value
        top_k_input.focus()
        await pilot.pause()
        assert screen.app.focused is top_k_input

        await pilot.press("a", "c", "b")
        await pilot.pause()

        # `restrict=r"^[0-9]*$"` on this field rejects the (non-digit)
        # insertion, but the KEY itself was still consumed by the Input --
        # the point under test.
        assert top_k_input.value == original_value
        assert calls == []

        # Defense-in-depth: calling the actions directly (bypassing normal
        # key dispatch) while the Input still has focus is ALSO a no-op.
        screen.action_settings_rag_set_active()
        screen.action_settings_rag_clone()
        screen.action_settings_rag_backfill()
        assert calls == []


@pytest.mark.asyncio
async def test_rag_category_footer_advertises_the_new_accelerators(
    monkeypatch, tmp_path
):
    """The footer/hint line for LIBRARY_RAG advertises 'a set active',
    'c clone', 'b backfill' alongside s/r/t; another category's footer
    does not carry them."""
    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        overview_footer = screen.query_one(AppFooterStatus)
        for token in ("a set active", "c clone", "b backfill"):
            assert token not in overview_footer.shortcut_text
        for token in ("s save category", "r revert category", "t test category"):
            assert token in overview_footer.shortcut_text

        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        rag_footer = screen.query_one(AppFooterStatus)
        for token in (
            "a set active",
            "c clone",
            "b backfill",
            "s save category",
            "r revert category",
            "t test category",
        ):
            assert token in rag_footer.shortcut_text

        # Leaving the category again drops the RAG-only hints.
        await _open_settings_category(pilot, "#settings-category-theme")
        screen = _active_destination_screen(host)
        theme_footer = screen.query_one(AppFooterStatus)
        for token in ("a set active", "c clone", "b backfill"):
            assert token not in theme_footer.shortcut_text
