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
from textual.widgets import Button, Checkbox, Input, Select, Static

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
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


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
