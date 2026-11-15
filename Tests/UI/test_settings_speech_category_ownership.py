"""Mounted Speech category boundaries and safe shell draft projection."""

import inspect
from unittest.mock import AsyncMock

import pytest
from textual import on
from textual.widgets import Button, Select

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import DestinationHarness, _build_test_app
from Tests.UI.test_settings_speech_tts_panel import (
    _PanelHarness,
    _open_speech_tts,
    _settle,
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.state.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Speech.settings_speech_tts import load_global_speech_tts_state
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSSettingsPanel,
)


class _SaveHarness(DestinationHarness):
    def __init__(self, *, persisted):
        super().__init__(_build_test_app(), "settings")
        self.persisted = persisted
        self.saves = []

    @on(STTSSettingsSaveEvent)
    def answer_save(self, event):
        event.stop()
        self.saves.append(event)
        self.screen.query_one(SpeechTTSSettingsPanel).receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=event.request_id,
                persisted=self.persisted,
                provider_statuses={},
                failure_phase=None if self.persisted else "before_replace",
            )
        )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    "choice,persisted", [("discard", True), ("save", True), ("save", False)]
)
async def test_speech_category_leave_outcome(request, choice, persisted):
    host = _SaveHarness(persisted=persisted)
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        await _settle(pilot)
        panel = screen.query_one(SpeechTTSSettingsPanel)
        model = panel.query_one("#settings-speech-model-value", Select)
        saved = model.value
        edited = "tts-1-hd" if saved != "tts-1-hd" else "tts-1"
        model.value = edited
        model.focus()
        panel._ask_leave_choice = AsyncMock(return_value=choice)
        await pilot.pause()
        screen.query_one("#settings-category-overview", Button).press()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        allowed = choice == "discard" or persisted
        assert screen.active_category == (
            SettingsCategoryId.OVERVIEW.value
            if allowed
            else SettingsCategoryId.SPEECH_TTS.value
        )
        assert len(host.saves) == (choice == "save")
        if allowed:
            assert screen._speech_tts_draft_snapshot is None
            assert not screen._category_has_unsaved_changes(
                SettingsCategoryId.SPEECH_TTS
            )
        else:
            assert screen.query_one(SpeechTTSSettingsPanel) is panel
            assert model.value == edited
            assert host.focused is model
            assert panel.has_unsaved_changes() is True


@pytest.mark.asyncio
@private_profile_test
async def test_speech_pending_edit_capture_rejects_older_queued_projection(
    request, monkeypatch
):
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        await _settle(pilot)
        panel = screen.query_one(SpeechTTSSettingsPanel)
        queued = []
        post = panel.post_message

        def hold_draft(message):
            if isinstance(message, panel.DraftModified):
                queued.append(message)
                return True
            return post(message)

        monkeypatch.setattr(panel, "post_message", hold_draft)
        panel._announce_draft_state()
        model = panel.query_one("#settings-speech-model-value", Select)
        edited = "tts-1-hd" if model.value != "tts-1-hd" else "tts-1"
        model.value = edited
        # No event-loop yield: Changed has not reached the panel yet.
        retained = screen.save_state()
        assert retained["speech_tts_panel_draft"].state.defaults.model_id == edited
        screen.handle_speech_tts_draft_modified(queued[0])
        assert screen._speech_tts_draft_snapshot.state.defaults.model_id == edited
        assert screen._category_has_unsaved_changes(SettingsCategoryId.SPEECH_TTS)
        panel._announce_draft_state()
        obsolete = queued[-1]
        panel._ask_leave_choice = AsyncMock(return_value="discard")
        screen._select_category(SettingsCategoryId.OVERVIEW.value)
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert screen._speech_tts_draft_snapshot is None
        screen.handle_speech_tts_draft_modified(obsolete)
        assert screen._speech_tts_draft_snapshot is None
        await _open_speech_tts(host, pilot)
        await _settle(pilot)
        replacement = screen.query_one(SpeechTTSSettingsPanel)
        current = screen._speech_tts_draft_snapshot
        assert replacement is not panel
        screen.handle_speech_tts_draft_modified(obsolete)
        assert screen._speech_tts_draft_snapshot is current


@pytest.mark.asyncio
@private_profile_test
async def test_speech_repaint_uses_snapshot_without_collecting_fields(
    request, monkeypatch
):
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        await _settle(pilot)
        panel = screen.query_one(SpeechTTSSettingsPanel)
        model = panel.query_one("#settings-speech-model-value", Select)
        model.value = "tts-1-hd" if model.value != "tts-1-hd" else "tts-1"
        await pilot.pause()

        def forbid_collect():
            raise AssertionError("ordinary repaint collected mounted Speech fields")

        monkeypatch.setattr(panel, "_collect_visible_state", forbid_collect)
        category = SettingsCategoryId.SPEECH_TTS
        assert screen._category_has_unsaved_changes(category)
        assert screen._category_unsaved_count(category) == 1
        assert "1 unsaved" in screen._category_state_banner_text(category)
        screen._update_draft_status_widgets(category)


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("invalid", [False, True])
async def test_speech_snapshot_cleanup_is_dirty_even_without_counted_fields(
    request, invalid
):
    state = load_global_speech_tts_state({})
    state.openai_plaintext_confirmation_cleanup_needed = True
    host = _PanelHarness(configure_provider="audio_cpp", state=state)
    async with host.run_test(size=(150, 80)) as pilot:
        await _settle(pilot)
        panel = host.query_one(SpeechTTSSettingsPanel)
        if invalid:
            from textual.widgets import Input

            panel.query_one(
                "#settings-speech-audio_cpp-synthesis-timeout-seconds", Input
            ).value = "invalid"
        snapshot = panel.draft_snapshot()
        assert panel.has_unsaved_changes()
        assert snapshot.unsaved_field_count() == (8 if invalid else 0)
        assert snapshot.has_unsaved_changes()


@pytest.mark.asyncio
@private_profile_test
async def test_speech_actions_capture_pending_edit_before_changed_delivery(request):
    host = _SaveHarness(persisted=False)
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        await _settle(pilot)
        panel = screen.query_one(SpeechTTSSettingsPanel)
        model = panel.query_one("#settings-speech-model-value", Select)
        edited = "tts-1-hd" if model.value != "tts-1-hd" else "tts-1"
        model.value = edited
        screen.action_settings_save_category(allow_text_entry_focus=True)
        assert screen._speech_tts_draft_snapshot.state.defaults.model_id == edited
        await _settle(pilot)
        assert len(host.saves) == 1
        model.value = "tts-1" if edited != "tts-1" else "tts-1-hd"
        latest = model.value
        panel.revert_to_saved = AsyncMock()
        screen.action_settings_revert_category(allow_text_entry_focus=True)
        assert screen._speech_tts_draft_snapshot.state.defaults.model_id == latest
        await _settle(pilot)
        # The live value may equal the saved default, in which case Revert is a no-op.
        model.value = edited
        panel._ask_leave_choice = AsyncMock(return_value="cancel")
        screen._select_category(SettingsCategoryId.OVERVIEW.value)
        assert screen._speech_tts_draft_snapshot.state.defaults.model_id == edited
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert screen.active_category == SettingsCategoryId.SPEECH_TTS.value
        panel._ask_leave_choice.assert_awaited_once()


@pytest.mark.asyncio
@private_profile_test
async def test_speech_widget_save_captures_pending_edit_in_shell(request):
    host = _SaveHarness(persisted=False)
    async with host.run_test(size=(190, 55)) as pilot:
        screen = await _open_speech_tts(host, pilot)
        await _settle(pilot)
        panel = screen.query_one(SpeechTTSSettingsPanel)
        model = panel.query_one("#settings-speech-model-value", Select)
        edited = "tts-1-hd" if model.value != "tts-1-hd" else "tts-1"
        model.value = edited
        panel.request_save()
        assert screen._speech_tts_draft_snapshot.state.defaults.model_id == edited


# These inherited scenarios predate private-profile qualification. Execute their
# actual bodies in the established profile fixture, preserving bootstrap guards.
@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    "scenario",
    [
        "bounded_speech_settings_deep_link_restores_provider_without_action",
        "provider_search_opens_speech_tts_with_named_provider_selected",
        "dirty_global_link_cancel_preserves_draft_focus_and_navigation",
        "dirty_global_link_failed_save_keeps_owner_draft_and_focus",
        "dirty_global_link_successful_save_continues_with_bounded_intent",
        "every_leave_guard_waits_for_an_existing_save_instead_of_discarding",
        "dirty_provider_switch_discard_removes_hidden_draft",
        "settings_dismissal_cancel_preserves_dirty_speech_owner",
        "settings_generic_save_and_revert_actions_route_to_speech_panel",
        "speech_shortcuts_defer_to_focused_text_entry_and_resume_after_blur",
        "speech_save_and_revert_clicks_work_while_a_field_is_focused",
    ],
)
async def test_speech_inherited_private_profile_contract(request, scenario):
    from Tests.UI import test_settings_speech_tts_panel as inherited

    body = inspect.unwrap(getattr(inherited, "test_" + scenario))
    kwargs = (
        {"request": request} if "request" in inspect.signature(body).parameters else {}
    )
    await body(**kwargs)


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("trigger", ["deep-link", "search"])
async def test_speech_same_category_navigation_contract(request, trigger):
    from Tests.UI.test_settings_speech_tts_panel import (
        test_dirty_same_category_provider_target_cancel_survives_recompose,
    )

    await test_dirty_same_category_provider_target_cancel_survives_recompose(trigger)


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("unrelated_first", [True, False])
async def test_speech_model_library_route_bypass_contract(
    request, unrelated_first, monkeypatch
):
    from tldw_chatbook.Widgets.Settings_Widgets import (
        speech_tts_settings_panel as panel_module,
    )

    monkeypatch.setattr(panel_module, "_AUDIO_CPP_MANAGED_UI_SUPPORTED", True)
    from Tests.UI.test_audio_cpp_model_library_handoff import (
        test_model_library_leave_bypass_is_fifo_route_exclusive,
    )

    await test_model_library_leave_bypass_is_fifo_route_exclusive(unrelated_first)


@pytest.mark.asyncio
@private_profile_test
async def test_speech_model_library_cleanup_fence_contract(
    request, tmp_path, monkeypatch
):
    """Qualify UI cleanup orchestration with a validated scanner-boundary fixture.

    Windows intentionally lacks this scanner's POSIX no-follow capability.
    This fixture supplies its accepted-package input; it does not qualify
    scanning, filesystem security or a managed runtime on Windows.
    """
    from hashlib import sha256

    from tldw_chatbook.TTS import audio_cpp_package_scanner as scanner_module
    from tldw_chatbook.TTS.audio_cpp_recipes import (
        AUDIO_CPP_RECIPE_REGISTRY,
        AudioCppMatchResult,
        AudioCppMatchState,
        AudioCppPackageCandidate,
        AudioCppPackageDescription,
        AudioCppPackageFileEvidence,
    )
    from tldw_chatbook.Widgets.Settings_Widgets import (
        speech_tts_settings_panel as panel_module,
    )

    monkeypatch.setattr(panel_module, "_AUDIO_CPP_MANAGED_UI_SUPPORTED", True)

    def accepted_package_boundary(root, **kwargs):
        root = root.resolve()
        weight = root / "supertonic-3-orig.gguf"
        digest = sha256(weight.read_bytes()).hexdigest()
        root_identity = scanner_module._filesystem_identity(root.stat())
        recipe = AUDIO_CPP_RECIPE_REGISTRY.for_package("supertonic_3_orig")
        candidate = AudioCppPackageCandidate(
            recipe=recipe,
            canonical_root=str(root),
            canonical_root_identity=root_identity,
            configuration_identity=sha256(b"UI fixture configuration").hexdigest(),
            weight_identity=digest,
            safe_name=root.name,
            evidence_relative_paths=(weight.name,),
        )
        # Exercise the real accepted-package validation at this boundary.
        candidate.accept(managed_artifact=kwargs["expected_managed_artifact"])
        description = AudioCppPackageDescription(
            canonical_root=str(root),
            canonical_root_identity=root_identity,
            safe_name=root.name,
            files=(
                AudioCppPackageFileEvidence(
                    weight.name, weight.stat().st_size, digest, True, True
                ),
            ),
        )
        return scanner_module.AudioCppPackageScanResult(
            outcome=scanner_module.AudioCppScanOutcome.COMPLETE,
            request_revision=kwargs.get("request_revision", 0),
            selected_root_name=root.name,
            canonical_root=str(root),
            canonical_root_identity=root_identity,
            root_was_symlink=False,
            discoveries=(
                scanner_module.AudioCppPackageDiscovery(
                    description,
                    AudioCppMatchResult(
                        AudioCppMatchState.EXACT, (recipe.recipe_id,), (candidate,)
                    ),
                ),
            ),
            limits_reached=(),
            issues=(),
            issues_truncated=False,
            unknown_names=(),
            unknown_names_truncated=False,
            visited_entries=1,
            metadata_bytes_read=8,
        )

    monkeypatch.setattr(
        scanner_module, "scan_audio_cpp_package_root", accepted_package_boundary
    )
    from Tests.UI.test_audio_cpp_model_library_handoff import (
        test_transaction_failure_rolls_back_and_requeues_exact_result,
    )

    await test_transaction_failure_rolls_back_and_requeues_exact_result(
        tmp_path, monkeypatch, "preack_overlap"
    )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    "extra",
    [
        {"api_key": "private"},
        {"text": "private synthesis text"},
        {"url": "https://user:secret@example.invalid"},
    ],
)
async def test_speech_navigation_rejects_extra_payload_contract(request, extra):
    from Tests.UI.test_settings_speech_tts_panel import (
        test_speech_settings_deep_link_rejects_extra_payload,
    )

    test_speech_settings_deep_link_rejects_extra_payload(extra)
