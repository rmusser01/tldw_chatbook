"""The dirty State line keeps its save-model badge and counts unsaved fields.

TASK-33002.4: with unsaved edits the State line used to drop ADR-303's
persistence badge for "State: Unsaved changes | ...", and nothing said how
much was unsaved. It now reads "State: {badge} · N unsaved | ..." where N is
the number of fields that differ from their saved values.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _static_text,
    _wait_for_selector,
)
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook import config as config_module
from tldw_chatbook.Backup_Recovery.unsaved_editors import (
    UnsavedEditor,
    probe_unsaved_editors,
)
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.state.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
    SettingsValidationResult,
)
from tldw_chatbook.UI.Navigation.screen_state_store import (
    RuntimeIdentity,
    ScreenStateStore,
)
from tldw_chatbook.UI.Screens import settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_network_defaults import (
    SettingsNetworkTLS,
    build_network_save_sections,
    load_network_tls,
)
from tldw_chatbook.UI.Screens.settings_screen import (
    GUIDED_SETTINGS_MUTATION_CATEGORIES,
    PROVIDER_NO_CHANGES_COPY,
    PROVIDER_SAVED_COPY,
    SettingsScreen,
)
from tldw_chatbook.UI.Settings_Modules.category_panels import (
    CATEGORY_PANELS,
    NETWORK_PANEL,
)
from tldw_chatbook.UI.Settings_Modules.settings_config_adapter import (
    SettingsConfigAdapter,
)
from tldw_chatbook.UI.Settings_Modules.settings_quit_guard import (
    unsaved_settings_titles,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.UI.Speech.settings_speech_tts import load_global_speech_tts_state
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_panel_types import (
    SpeechTTSPanelDraftSnapshot,
    _RealtimeSettingsDraft,
)

_REALTIME = _RealtimeSettingsDraft(
    False,
    "openai",
    "gpt-realtime",
    "",
    "30",
    "auto",
    "semantic_vad",
    "0.5",
    "500",
    "",
    False,
)
_STAGING_CATEGORIES = (
    *sorted(GUIDED_SETTINGS_MUTATION_CATEGORIES, key=lambda c: c.value),
    SettingsCategoryId.IMAGE_GENERATION,
    SettingsCategoryId.VIDEO_GENERATION,
    SettingsCategoryId.ADVANCED_CONFIG,
    SettingsCategoryId.NETWORK,  # registered panel; review N5
)


def _speech_snapshot(**changes) -> SpeechTTSPanelDraftSnapshot:
    original = load_global_speech_tts_state({}, environment={})
    state = load_global_speech_tts_state({}, environment={})
    for provider, values in changes.pop("providers", {}).items():
        state.providers[provider].update(values)
    if "speed" in changes:
        state.defaults.speed = changes.pop("speed")
    return SpeechTTSPanelDraftSnapshot(
        state=state,
        original_state=original,
        realtime_draft=replace(_REALTIME, **changes),
        realtime_original=replace(_REALTIME),
        configure_provider="kokoro",
        draft_revision=1,
    )


def test_speech_snapshot_counts_fields_that_differ_from_saved() -> None:
    """Validated values compare, so text that parses to the saved value is clean."""
    assert _speech_snapshot().unsaved_field_count() == 0
    snapshot = _speech_snapshot(
        speed=1.5,
        providers={
            "kokoro": {"max_tokens": 600},
            # Collected from an Input as text; validates to the saved 0.5.
            "elevenlabs": {"stability": "0.5"},
        },
        idle_timeout_minutes="45",
    )
    assert snapshot.unsaved_field_count() == 3


@pytest.mark.asyncio
@private_profile_test
async def test_console_behavior_line_keeps_badge_and_counts_fields(request) -> None:
    screen = SettingsScreen(_build_test_app())
    category = SettingsCategoryId.CONSOLE_BEHAVIOR
    draft = SettingsDraft(category=category)
    draft.set_value("paste_collapse_threshold", 50, 503)
    draft.set_value("max_parallel_runs", 2, 3)
    screen._settings_drafts[category] = draft
    scope = "Changes affect global Console fallbacks after save."

    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s · 2 unsaved · revert with r | {scope}"
    )
    draft.set_value("max_parallel_runs", 2, 2)  # reverted back by hand
    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s · 1 unsaved · revert with r | {scope}"
    )


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_counts_fields_not_draft_keys(request) -> None:
    """The context window and its reset flag are one field; snapshot
    preferences are two fields that live outside the draft."""
    screen = SettingsScreen(_build_test_app())
    category = SettingsCategoryId.PROVIDERS_MODELS
    draft = SettingsDraft(category=category)
    draft.set_value("provider", "openai", "openai")
    draft.set_value("model_context_window", "", "32768")
    draft.set_value("model_context_window_reset", False, True)
    screen._settings_drafts[category] = draft
    scope = screen._category_state_scope_text(category)

    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s · 1 unsaved | {scope}"
    )
    draft.set_value("model_profile_temperature", "", "0.4")
    loaded = type("Loaded", (), {"enabled": True, "keep_count": 5})()
    screen._snapshot_preferences_loaded = loaded
    screen._snapshot_preferences_raw = (False, "7")
    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s · 4 unsaved | {scope}"
    )


@pytest.mark.asyncio
@private_profile_test
async def test_other_staging_lines_keep_their_own_badge(request) -> None:
    screen = SettingsScreen(_build_test_app())
    image = SettingsCategoryId.IMAGE_GENERATION
    draft = SettingsDraft(category=image)
    draft.set_value("default_backend", "a", "b")
    screen._settings_drafts[image] = draft
    assert screen._category_state_banner_text(image) == (
        "State: Draft — save/revert below · 1 unsaved | "
        "Defaults affect future Console image generations."
    )

    advanced = SettingsCategoryId.ADVANCED_CONFIG
    screen._category_has_unsaved_changes = lambda category: category is advanced
    assert screen._category_state_banner_text(advanced) == (
        "State: Validate, then Save · 1 unsaved | Draft kept when you leave; "
        "use raw editor controls."
    )


@pytest.mark.asyncio
@private_profile_test
async def test_invalid_draft_keeps_badge_and_count(request) -> None:
    """Final review (Task 4 ruling): "State: Needs correction | ..." dropped
    both the badge and the count on Appearance, Library & RAG and Storage."""
    screen = SettingsScreen(_build_test_app())
    problem = "Media database path must not be blank."
    invalid = SettingsValidationResult(False, problem)
    screen._appearance_validation_result = lambda: invalid
    screen._library_rag_validation_result = lambda: invalid
    screen._storage_validation_result = lambda: invalid
    for category in (
        SettingsCategoryId.APPEARANCE,
        SettingsCategoryId.LIBRARY_RAG,
        SettingsCategoryId.STORAGE,
    ):
        screen._category_has_unsaved_changes = lambda c, category=category: (
            c is category
        )
        screen._category_unsaved_count = lambda c: 2
        assert screen._category_state_banner_text(category) == (
            f"State: Draft — save with s · 2 unsaved | Needs correction: {problem}"
        ), category


@pytest.mark.asyncio
@private_profile_test
async def test_categories_without_a_draft_keep_todays_badge(request) -> None:
    """AC#4: the clean line is "State: {badge} | {scope}", as before."""
    screen = SettingsScreen(_build_test_app())
    expected = {
        SettingsCategoryId.NETWORK: "Pending — save with s",
        # TASK-36387, rewritten on purpose: the category list's own words.
        SettingsCategoryId.WORKSPACES: "Immediate actions",
        SettingsCategoryId.SPLASH_SCREEN: "Auto-saved",
        SettingsCategoryId.INTERNAL_PROMPTS: "Per-item Save/Reset",
        # Capture finding 3 (TASK-33007), rewritten on purpose: at rest a
        # draft category says it holds no changes, not "Draft".
        SettingsCategoryId.CONSOLE_BEHAVIOR: "No changes · edits save with s",
    }
    for category, badge in expected.items():
        text = screen._category_state_banner_text(category)
        assert text == f"State: {badge} | {screen._category_state_scope_text(category)}"
        assert "unsaved" not in text


@pytest.mark.asyncio
@private_profile_test
async def test_dirty_state_line_renders_without_a_new_row_at_211x44(request) -> None:
    """AC#1/AC#2 on the real banner (production stylesheet, 211x44).

    Every staging category's worst-case dirty line (99 unsaved) renders in no
    more rows than its clean line, and a real Console Behavior edit counts up
    and back down as fields are edited and reverted by hand.
    """
    host = _SettingsCssHarness(_build_test_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        screen._select_category(category.value)
        await _wait_for_selector(
            screen, pilot, "#settings-console-paste-collapse-threshold"
        )
        banner = screen.query_one("#settings-category-state-banner", Static)
        assert banner.content_region.width == 120

        async def rows(text: str) -> int:
            banner.update(text)
            await pilot.pause()
            return banner.content_region.height

        real_dirty = screen._category_has_unsaved_changes
        real_count = screen._category_unsaved_count
        for staged in _STAGING_CATEGORIES:
            clean = screen._category_state_banner_text(staged)
            screen._category_has_unsaved_changes = lambda c, staged=staged: c is staged
            screen._category_unsaved_count = lambda c: 99
            dirty = screen._category_state_banner_text(staged)
            screen._category_has_unsaved_changes = real_dirty
            screen._category_unsaved_count = real_count
            assert "· 99 unsaved" in dirty, dirty
            assert await rows(dirty) <= await rows(clean), (staged, dirty)
        banner.update(screen._category_state_banner_text(category))

        threshold = screen.query_one(
            "#settings-console-paste-collapse-threshold", Input
        )
        runs = screen.query_one("#settings-console-max-parallel-runs", Input)
        saved_threshold, saved_runs = threshold.value, runs.value
        threshold.value = str(int(saved_threshold or 0) + 7)
        await pilot.pause()
        assert "· 1 unsaved · revert with r | " in _static_text(banner)
        runs.value = str(int(saved_runs or 1) + 1)
        await pilot.pause()
        assert "· 2 unsaved · revert with r | " in _static_text(banner)
        threshold.value = saved_threshold
        await pilot.pause()
        assert "· 1 unsaved · revert with r | " in _static_text(banner)
        assert banner.content_region.height == 1
        runs.value = saved_runs
        await pilot.pause()
        assert _static_text(banner) == (
            "State: No changes · edits save with s | "
            "Changes affect global Console fallbacks after save."
        )


@pytest.mark.asyncio
@private_profile_test
async def test_image_gen_counts_a_two_key_clear_as_one_field(request) -> None:
    """TASK-33002.11: Clear on SwarmUI's token stages the token and its legacy
    api_key spelling, two draft keys for one field, and the count said 2."""
    host = _SettingsCssHarness(_build_test_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        category = SettingsCategoryId.IMAGE_GENERATION
        screen._select_category(category.value)
        base_url = "#settings-imagegen-field-swarmui-base_url"
        await _wait_for_selector(screen, pilot, base_url, timeout=8.0)
        banner = screen.query_one("#settings-category-state-banner", Static)

        screen._handle_image_gen_clear("swarmui", "swarm_token")
        await pilot.pause()
        assert len(screen._settings_drafts[category].dirty_keys) == 2
        assert "· 1 unsaved | " in _static_text(banner)

        field = screen.query_one(base_url, Input)
        saved = field.value
        field.value = saved + "/edited"
        await pilot.pause()
        assert "· 2 unsaved | " in _static_text(banner)
        field.value = saved  # set back by hand
        await pilot.pause()
        assert "· 1 unsaved | " in _static_text(banner)


@pytest.mark.asyncio
@private_profile_test
async def test_video_gen_counts_an_edit_and_a_clear_of_one_field_once(request) -> None:
    """TASK-33002.11: a backend field staged as an edit and as a Clear is one
    field; a global key beside it is another."""
    screen = SettingsScreen(_build_test_app())
    video = SettingsCategoryId.VIDEO_GENERATION
    draft = SettingsDraft(category=video)
    draft.set_value("field::fal::api_key", "", "typed")
    draft.set_value("cleared::fal::api_key", False, True)
    screen._settings_drafts[video] = draft
    assert screen._category_unsaved_count(video) == 1
    draft.set_value("max_store_mb", "500", "600")
    assert screen._category_unsaved_count(video) == 2
    draft.set_value("max_store_mb", "500", "500")  # set back by hand
    assert screen._category_unsaved_count(video) == 1


@pytest.mark.asyncio
@private_profile_test
async def test_f1_help_says_switching_categories_keeps_the_draft(request) -> None:
    """TASK-33002.10: the State line dropped "switching categories keeps this
    draft" to fit the count (TASK-33002.4), and no in-app surface said it.
    F1 help says it on every category whose draft a switch keeps; Speech &
    TTS asks on leave instead, and Advanced Config's State line says it."""
    screen = SettingsScreen(_build_test_app())
    keeps = {
        *GUIDED_SETTINGS_MUTATION_CATEGORIES,
        SettingsCategoryId.IMAGE_GENERATION,
        SettingsCategoryId.VIDEO_GENERATION,
        SettingsCategoryId.NETWORK,  # TASK-1378.1: kept on switch, marked *
    } - {SettingsCategoryId.SPEECH_TTS}
    note = "Switching categories keeps the draft; the rail marks it with *."
    for category in SettingsCategoryId:
        body = screen._workbench_help_state(category).render_text()
        assert (note in body) is (category in keeps), category


# --- Network, served by its registered panel (TASK-1378.1) ---------------
# ADR-303 staged model with no SettingsDraft: an edit is unsaved (rail *,
# banner count, quit prompt), s saves it, r discards it after asking.

_NETWORK = SettingsCategoryId.NETWORK
_NETWORK_SCOPE = "Trust applies to new connections; open ones keep theirs."
_NETWORK_GUIDED = (
    "Save with s: the TLS trust setting applies to newly created clients; "
    "already-open connections keep their trust policy until restart."
)


async def _open_network(host, pilot) -> SettingsScreen:
    await _settle_settings(pilot)
    await _click_settings_category(pilot, "network")
    return _active_destination_screen(host)


def _capture_toasts(screen, monkeypatch) -> list[tuple[str, str]]:
    toasts: list[tuple[str, str]] = []

    def _notify(message, *args, severity="information", **kwargs):
        toasts.append((str(message), severity))

    monkeypatch.setattr(screen.app, "notify", _notify)
    return toasts


@pytest.mark.asyncio
@private_profile_test
async def test_registered_panels_answer_the_screens_category_questions(request) -> None:
    """For a registered category the screen asks its panel, so the two give one
    answer: unsaved state (and the quit guard), badge, both scope lines, and
    what s and r may do (from the panel's ADR-303 commit model). Leaving
    Settings keeps a panel's staged edit, so the quit prompt stays true."""
    screen = SettingsScreen(_build_test_app())
    for category, panel in CATEGORY_PANELS.items():
        scope = panel.scope_text(screen)
        assert screen._persistence_badge(category) == panel.commit_model.badge
        assert screen._guided_action_message(category) == scope.guided
        assert screen._category_state_scope_text(category) == scope.banner
        for action, allowed in (
            ("save", panel.commit_model.saves),
            ("revert", panel.commit_model.reverts),
        ):
            assert SettingsScreen._category_action_supported(category, action) is allowed
    assert screen._category_has_unsaved_changes(_NETWORK) is False
    screen._category_panel_states[_NETWORK] = {"mode": "off"}
    assert NETWORK_PANEL.has_unsaved_changes(screen) is True
    assert screen._category_has_unsaved_changes(_NETWORK) is True
    assert "Network" in unsaved_settings_titles(screen)

    revisit = SettingsScreen(_build_test_app())
    revisit.restore_state(screen.save_state())
    assert NETWORK_PANEL.pending(revisit) == {"mode": "off"}
    assert revisit._category_has_unsaved_changes(_NETWORK) is True

    # Recovery mode's unsaved-work check gives the same answer, live and kept.
    live = probe_unsaved_editors(editors=(screen,))
    assert live == (UnsavedEditor("Settings", "needs-user-save-discard"),)
    store = ScreenStateStore()
    store.save("settings", screen.save_state(), RuntimeIdentity("server", "other"))
    kept = probe_unsaved_editors(screen_state_store=store)
    assert kept == (UnsavedEditor("Retained Settings", "needs-user-save-discard"),)


@pytest.mark.asyncio
@private_profile_test
async def test_network_select_edit_marks_the_rail_banner_and_quit_prompt(
    request,
) -> None:
    """Review I2 + M4: an edit through the real Select reaches the card's
    handler, repaints the warning, marks the rail and the banner, and makes
    the quit prompt name Network; a category hop keeps it without asking."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_network(host, pilot)
        # The Select and Input echo their seeded values on mount: no edit.
        assert screen._category_has_unsaved_changes(_NETWORK) is False
        assert screen._persistence_badge(_NETWORK) == "Pending — save with s"
        assert screen._guided_action_message(_NETWORK) == _NETWORK_GUIDED
        assert screen._category_state_banner_text(_NETWORK) == (
            f"State: Pending — save with s | {_NETWORK_SCOPE}"
        )

        screen.query_one("#settings-network-ssl-mode", Select).value = "off"
        await pilot.pause()
        warning = screen.query_one("#settings-network-warning", Static)
        assert _static_text(warning).startswith("Verification is DISABLED")
        rail = screen.query_one("#settings-category-network", Button)
        assert str(rail.label).endswith(" *")
        banner = screen.query_one("#settings-category-state-banner", Static)
        assert _static_text(banner) == (
            "State: Pending — save with s · 1 unsaved · revert with r | "
            f"{_NETWORK_SCOPE}"
        )

        quit_flow = host.run_worker(screen.confirm_quit(), exit_on_error=False)
        await pilot.pause(0.3)
        prompt = host.screen
        assert isinstance(prompt, ConfirmationDialog)
        assert "Network" in prompt.message
        await pilot.click("#cancel-button")  # Keep editing
        await pilot.pause(0.2)
        assert await quit_flow.wait() is False

        await _click_settings_category(pilot, "appearance")
        assert screen.active_category == "appearance"  # no leave prompt
        await _click_settings_category(pilot, "network")
        assert screen.query_one("#settings-network-ssl-mode", Select).value == "off"
        assert NETWORK_PANEL.pending(screen)["mode"] == "off"


@pytest.mark.asyncio
@private_profile_test
async def test_network_s_saves_through_the_screens_config_adapter(
    request, monkeypatch
) -> None:
    """Review M5: a panel saves through the screen's adapter seam, so the
    dominant patch (``settings_screen.SettingsConfigAdapter``) reaches it."""
    saved: list[dict] = []

    class _FakeAdapter:
        def save_sections(self, sections):
            saved.append({k: dict(v) for k, v in dict(sections).items()})
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", _FakeAdapter)
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_network(host, pilot)
        screen.query_one("#settings-network-ssl-mode", Select).value = "off"
        await pilot.pause()
        rail = screen.query_one("#settings-category-network", Button)
        rail.focus()
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        assert saved == [build_network_save_sections(SettingsNetworkTLS("off"))]
        assert screen._category_has_unsaved_changes(_NETWORK) is False
        assert not str(rail.label).endswith(" *")


@pytest.mark.asyncio
@private_profile_test
async def test_network_rejects_missing_ca_then_saves_a_valid_one(
    request, tmp_path, monkeypatch
) -> None:
    saved: list[dict] = []

    def _capture(sections):
        saved.append({k: dict(v) for k, v in dict(sections).items()})
        return True

    monkeypatch.setattr(SettingsConfigAdapter, "save_sections", staticmethod(_capture))
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_network(host, pilot)
        toasts = _capture_toasts(screen, monkeypatch)
        NETWORK_PANEL.stage(
            screen, mode="custom-ca", ca_bundle_path=str(tmp_path / "missing.pem")
        )
        screen.action_settings_save_category()  # verified sync: -> None
        assert saved == []  # invalid path rejected, nothing written
        assert len(toasts) == 1 and toasts[0][1] == "error"
        assert NETWORK_PANEL.pending(screen)["mode"] == "custom-ca"

        toasts.clear()
        ca = tmp_path / "corp.pem"
        ca.write_text("# corp ca")
        NETWORK_PANEL.stage(screen, ca_bundle_path=str(ca))
        screen.action_settings_save_category()
        assert saved == [{"network": {"ssl_verify": str(ca)}}]
        assert toasts == [("Network TLS setting saved.", "information")]
        assert NETWORK_PANEL.pending(screen) == {}  # cleared after the save
        # qodo PR #2223 bug 5: the app's in-memory config mapping must
        # reflect the save, or the next detail render shows the stale
        # pre-save mode/path.
        assert load_network_tls(screen._app_config_mapping()).ca_bundle_path == str(ca)
        guided_text = screen.query_one("#settings-guided-action-state", Static)
        assert "Read-only" not in _static_text(guided_text)


@pytest.mark.asyncio
@private_profile_test
async def test_network_save_failure_keeps_the_edit_and_r_discards_it_after_asking(
    request, monkeypatch
) -> None:
    monkeypatch.setattr(
        SettingsConfigAdapter, "save_sections", staticmethod(lambda sections: False)
    )
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_network(host, pilot)
        toasts = _capture_toasts(screen, monkeypatch)
        screen.query_one("#settings-network-ssl-mode", Select).value = "off"
        await pilot.pause()
        screen.action_settings_save_category()
        assert toasts == [("Failed to save Network TLS setting.", "error")]
        assert screen._category_has_unsaved_changes(_NETWORK) is True

        screen.action_settings_revert_category()
        await pilot.pause(0.2)
        assert isinstance(host.screen, ConfirmationDialog)
        await pilot.click("#cancel-button")  # Keep editing
        await pilot.pause(0.2)
        assert screen._category_has_unsaved_changes(_NETWORK) is True

        toasts.clear()
        screen.action_settings_revert_category()
        await pilot.pause(0.2)
        await pilot.click("#confirm-button")  # Discard changes
        await _settle_settings(pilot)
        assert screen._category_has_unsaved_changes(_NETWORK) is False
        assert toasts == [("Settings category changes reverted.", "information")]
        select = screen.query_one("#settings-network-ssl-mode", Select)
        assert select.value == "verify"  # repainted from the loaded config


@pytest.mark.asyncio
@pytest.mark.parametrize("ssl_verify", ["moved-ca-file", 7], ids=["moved-ca-file", "not-a-mode"])
@private_profile_test
async def test_network_with_an_unreadable_ssl_verify_opens_clean_and_r_keeps_it_clean(
    request, tmp_path, ssl_verify
) -> None:
    """Review N1: an ``ssl_verify`` the loader reads as "invalid" (a CA file
    since moved away, or a value of no known form) seeds the Select with
    "verify", and the Select's mount echo was staged as an edit: opening
    Network raised a ``*`` no ``r`` could clear and blocked Recovery. Opening
    is no edit; a real choice on that Select still is, and ``r`` clears it."""
    if ssl_verify == "moved-ca-file":
        ssl_verify = str(tmp_path / "corp-ca.pem")  # configured, then moved
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        screen = _active_destination_screen(host)
        screen._app_config_update_target().update({"network": {"ssl_verify": ssl_verify}})
        await _click_settings_category(pilot, "network")
        assert screen.query_one("#settings-network-invalid-row", Static)

        def marked() -> bool:
            rail = screen.query_one("#settings-category-network", Button)
            return str(rail.label).endswith(" *")

        assert screen._category_has_unsaved_changes(_NETWORK) is False
        assert not marked()
        assert "Network" not in unsaved_settings_titles(screen)
        store = ScreenStateStore()
        store.save("settings", screen.save_state(), RuntimeIdentity("server", "other"))
        assert probe_unsaved_editors(screen_state_store=store) == ()

        screen.query_one("#settings-network-ssl-mode", Select).value = "off"
        await pilot.pause()
        assert NETWORK_PANEL.pending(screen)["mode"] == "off"
        assert screen._category_has_unsaved_changes(_NETWORK) is True and marked()

        screen.action_settings_revert_category()
        await pilot.pause(0.2)
        await pilot.click("#confirm-button")  # Discard changes
        await _settle_settings(pilot)
        assert screen._category_has_unsaved_changes(_NETWORK) is False
        assert not marked()
        screen.action_settings_revert_category()  # nothing left: no dialog
        await pilot.pause(0.2)
        assert not isinstance(host.screen, ConfirmationDialog)


def _saved_ssl_verify():
    """``[network] ssl_verify`` as written to this private profile's config."""
    return SettingsConfigAdapter().load(force_reload=True)["network"]["ssl_verify"]


@pytest.mark.asyncio
@private_profile_test
async def test_network_pick_back_to_verify_after_a_save_is_staged_and_saved(
    request, monkeypatch
) -> None:
    """Review N6: after "off" was saved, picking "verify" again (the value the
    card first showed) was dropped as a mount echo, and the next ``s`` wrote
    false again while toasting success. Every user pick is an edit."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_network(host, pilot)
        toasts = _capture_toasts(screen, monkeypatch)
        select = screen.query_one("#settings-network-ssl-mode", Select)
        rail = screen.query_one("#settings-category-network", Button)
        warning = screen.query_one("#settings-network-warning", Static)

        select.value = "off"
        await pilot.pause()
        screen.action_settings_save_category()
        assert _saved_ssl_verify() is False

        select.value = "verify"
        await pilot.pause()
        assert NETWORK_PANEL.pending(screen) == {"mode": "verify"}
        assert str(rail.label).endswith(" *")
        assert "Network" in unsaved_settings_titles(screen)
        assert _static_text(warning) == ""  # no "Verification is DISABLED"

        screen.action_settings_save_category()
        assert _saved_ssl_verify() is True
        assert toasts[-1] == ("Network TLS setting saved.", "information")
        assert screen._category_has_unsaved_changes(_NETWORK) is False
        assert not str(rail.label).endswith(" *")


@pytest.mark.asyncio
@private_profile_test
async def test_network_ca_path_typed_back_after_a_save_is_staged_and_saved(
    request, tmp_path
) -> None:
    """Review N6 on the CA path Input: after another bundle was saved, typing
    the path the card first showed is an edit, and ``s`` writes it."""
    first, second = tmp_path / "first-ca.pem", tmp_path / "second-ca.pem"
    first.write_text("# first ca")
    second.write_text("# second ca")
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        screen = _active_destination_screen(host)
        screen._app_config_update_target().update({"network": {"ssl_verify": str(first)}})
        await _click_settings_category(pilot, "network")
        path = screen.query_one("#settings-network-ca-path", Input)
        assert path.value == str(first)

        path.value = str(second)
        await pilot.pause()
        screen.action_settings_save_category()
        assert _saved_ssl_verify() == str(second)

        path.value = str(first)
        await pilot.pause()
        assert NETWORK_PANEL.pending(screen) == {"ca_bundle_path": str(first)}
        assert "Network" in unsaved_settings_titles(screen)
        screen.action_settings_save_category()
        assert _saved_ssl_verify() == str(first)
        assert screen._category_has_unsaved_changes(_NETWORK) is False


# --- Providers & Models' ADR-303 save behaviour (TASK-1378.2) ------------
# Pinned through the real screen before the category moved onto its
# registered panel, and unchanged by the move: a staged draft ("Draft — save
# with s"); an edit marks the rail, counts on the banner (no "revert with r":
# the scope fills the row) and makes the quit prompt name the category; a
# category hop keeps it without asking; s saves, r discards after asking.

_PM = SettingsCategoryId.PROVIDERS_MODELS
_PM_SCOPE = (
    "Applies to new and unused open chats · used chats keep theirs (Console: Alt+M)"
)
_PM_SAVED_TOAST = (
    "Provider and model settings saved: new chats and open chats nobody has "
    "used yet take them; chats with work keep their own settings (change them "
    "in Console with Alt+M)."
)


def _providers_app():
    """A keyless local provider whose saved model is ``model-a``."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        **app.app_config.get("chat_defaults", {}),
        "provider": "llama_cpp",
        "model": "model-a",
    }
    api_settings = app.app_config.setdefault("api_settings", {})
    api_settings["llama_cpp"] = {**api_settings.get("llama_cpp", {}), "model": "model-a"}
    return app


async def _open_providers(host, pilot) -> SettingsScreen:
    await _settle_settings(pilot)
    await _click_settings_category(pilot, _PM.value)
    screen = _active_destination_screen(host)
    await _wait_for_selector(screen, pilot, "#settings-model-value", timeout=30.0)
    await _settle_settings(pilot)
    return screen


def _rail_marked(screen) -> bool:
    rail = screen.query_one(f"#settings-category-{_PM.value}", Button)
    return str(rail.label).endswith(" *")


def _capture_persists(monkeypatch, result) -> list[dict]:
    """Patch the writer on the settings_screen module, as existing tests do."""
    calls: list[dict] = []

    def _persist(setup_mutation, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(
        settings_screen_module, "persist_provider_settings_atomic", _persist
    )
    return calls


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_commit_model_and_copy_are_pinned(request) -> None:
    """The badge, both scope lines, the guided copy and what s, r and t do."""
    screen = SettingsScreen(_providers_app())
    assert screen._persistence_badge(_PM) == "Draft — save with s"
    assert screen._category_state_scope_text(_PM) == _PM_SCOPE
    assert screen._guided_action_message(_PM) == "Guided edits: change a field first."
    assert screen._category_state_banner_text(_PM) == (
        f"State: No changes · edits save with s | {_PM_SCOPE}"
    )
    for action in ("save", "revert", "test"):
        assert SettingsScreen._category_action_supported(_PM, action) is True
    assert screen._category_footer_shortcuts(_PM) == (
        ("s", "save category"),
        ("r", "revert category"),
        ("t", "test provider"),
    )

    draft = SettingsDraft(category=_PM)
    draft.set_value("model", "model-a", "model-b")
    screen._settings_drafts[_PM] = draft
    assert screen._category_has_unsaved_changes(_PM) is True
    assert screen._guided_action_message(_PM) == "Guided edits: Save or Revert changes."
    assert screen._category_state_banner_text(_PM) == (
        f"State: Draft — save with s · 1 unsaved | {_PM_SCOPE}"
    )
    assert "Providers & Models" in unsaved_settings_titles(screen)

    revisit = SettingsScreen(_providers_app())
    revisit.restore_state(screen.save_state())
    assert revisit._category_has_unsaved_changes(_PM) is True

    # Snapshot preferences live outside the draft and are unsaved edits too.
    snapshot = SettingsScreen(_providers_app())
    snapshot._snapshot_preferences_loaded = type(
        "Loaded", (), {"enabled": True, "keep_count": 5}
    )()
    snapshot._snapshot_preferences_raw = (True, "7")
    assert snapshot._category_has_unsaved_changes(_PM) is True
    assert "Providers & Models" in unsaved_settings_titles(snapshot)


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_edit_marks_the_rail_banner_and_quit_prompt(
    request,
) -> None:
    """Opening is clean (no mount echo is an edit); a typed model marks the
    rail and banner, the quit prompt names the category, and a hop away and
    back keeps the edit without asking."""
    host = DestinationHarness(_providers_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = await _open_providers(host, pilot)
        model = screen.query_one("#settings-model-value", Input)
        assert model.value == "model-a"
        assert screen._category_has_unsaved_changes(_PM) is False
        assert not _rail_marked(screen)
        assert "Providers & Models" not in unsaved_settings_titles(screen)

        model.value = "model-b"
        await pilot.pause()
        assert screen._category_has_unsaved_changes(_PM) is True
        assert _rail_marked(screen)
        banner = screen.query_one("#settings-category-state-banner", Static)
        assert _static_text(banner) == (
            f"State: Draft — save with s · 1 unsaved | {_PM_SCOPE}"
        )
        assert _static_text(
            screen.query_one("#settings-selected-category-draft-status", Static)
        ) == "Unsaved changes"

        quit_flow = host.run_worker(screen.confirm_quit(), exit_on_error=False)
        await pilot.pause(0.3)
        prompt = host.screen
        assert isinstance(prompt, ConfirmationDialog)
        assert "Providers & Models" in prompt.message
        await pilot.click("#cancel-button")  # Keep editing
        await pilot.pause(0.2)
        assert await quit_flow.wait() is False

        await _click_settings_category(pilot, "appearance")
        assert screen.active_category == "appearance"  # no leave prompt
        assert _rail_marked(screen)
        screen = await _open_providers(host, pilot)
        assert screen.query_one("#settings-model-value", Input).value == "model-b"
        assert screen._category_has_unsaved_changes(_PM) is True


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_s_saves_through_the_screens_writer_seam(
    request, monkeypatch
) -> None:
    """``s`` pressed through the pilot reaches a writer patched on the
    settings_screen module (the idiom four existing files use)."""
    calls = _capture_persists(monkeypatch, ConfigMutationResult(True, True, None))
    host = DestinationHarness(_providers_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = await _open_providers(host, pilot)
        toasts = _capture_toasts(screen, monkeypatch)
        screen.query_one("#settings-model-value", Input).value = "model-b"
        await pilot.pause()
        rail = screen.query_one(f"#settings-category-{_PM.value}", Button)
        rail.focus()
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        assert [(c["provider"], c["model"]) for c in calls] == [("llama_cpp", "model-b")]
        assert (_PM_SAVED_TOAST, "information") in toasts
        assert screen._provider_save_result == PROVIDER_SAVED_COPY
        assert screen._category_has_unsaved_changes(_PM) is False
        assert not _rail_marked(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_save_failure_keeps_the_draft_and_r_discards_after_asking(
    request, monkeypatch
) -> None:
    calls = _capture_persists(
        monkeypatch, ConfigMutationResult(False, False, "before_replace")
    )
    host = DestinationHarness(_providers_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = await _open_providers(host, pilot)
        toasts = _capture_toasts(screen, monkeypatch)
        model = screen.query_one("#settings-model-value", Input)
        model.value = "model-b"
        await pilot.pause()
        screen.action_settings_save_category()
        assert len(calls) == 1
        failure = (
            "Provider settings not fully applied: the file was not written. "
            "Draft retained; check config access and retry."
        )
        assert toasts == [(failure, "error")]
        assert screen._category_has_unsaved_changes(_PM) is True
        assert _rail_marked(screen)

        screen.action_settings_revert_category()
        await pilot.pause(0.2)
        prompt = host.screen
        assert isinstance(prompt, ConfirmationDialog)
        assert prompt.message == "Discard all unsaved changes to Providers & Models?"
        await pilot.click("#cancel-button")  # Keep editing
        await pilot.pause(0.2)
        assert screen._category_has_unsaved_changes(_PM) is True
        assert model.value == "model-b"

        toasts.clear()
        screen.action_settings_revert_category()
        await pilot.pause(0.2)
        await pilot.click("#confirm-button")  # Discard changes
        await _settle_settings(pilot)
        assert screen._category_has_unsaved_changes(_PM) is False
        assert not _rail_marked(screen)
        assert ("Settings category changes reverted.", "information") in toasts
        assert screen.query_one("#settings-model-value", Input).value == "model-a"


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_s_and_r_with_nothing_unsaved_say_so(
    request, monkeypatch
) -> None:
    calls = _capture_persists(monkeypatch, ConfigMutationResult(True, True, None))
    host = DestinationHarness(_providers_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = await _open_providers(host, pilot)
        toasts = _capture_toasts(screen, monkeypatch)
        screen.action_settings_save_category()
        assert calls == []
        assert toasts == [("No Settings changes to save.", "information")]
        assert screen._provider_save_result == PROVIDER_NO_CHANGES_COPY

        toasts.clear()
        screen.action_settings_revert_category()
        await pilot.pause(0.2)
        assert not isinstance(host.screen, ConfirmationDialog)
        assert toasts == [("No Settings changes to revert.", "information")]


def _saved_chat_model():
    """``[chat_defaults] model`` as written to this private profile's config."""
    return SettingsConfigAdapter().load(force_reload=True)["chat_defaults"]["model"]


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_model_typed_back_after_a_save_is_staged_and_saved(
    request,
) -> None:
    """1378.1's N6 shape on this category's controls, with the real writer:
    after another model was saved, typing back the model the card first
    showed is an edit (rail ``*``, quit guard), and ``s`` writes it."""
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    host = DestinationHarness(_providers_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = await _open_providers(host, pilot)
        model = screen.query_one("#settings-model-value", Input)
        model.value = "model-b"
        await pilot.pause()
        screen.action_settings_save_category()
        await pilot.pause()
        assert _saved_chat_model() == "model-b"
        assert screen._category_has_unsaved_changes(_PM) is False

        model.value = "model-a"
        await pilot.pause()
        assert screen._category_has_unsaved_changes(_PM) is True
        assert _rail_marked(screen)
        assert "Providers & Models" in unsaved_settings_titles(screen)
        screen.action_settings_save_category()
        await pilot.pause()
        assert _saved_chat_model() == "model-a"
        assert screen._category_has_unsaved_changes(_PM) is False
        assert not _rail_marked(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_unreadable_snapshot_preferences_keep_the_reload_revert(
    request, monkeypatch
) -> None:
    """TASK-1378.2 review I1: while snapshot preferences cannot be read, ``r``
    reloads them, and the Inspector's Revert says so and stays enabled after
    every refresh of the Save/Revert pair, the card's own deferred one too."""
    from tldw_chatbook.LLM_Management import snapshot_settings

    def _unreadable(*args, **kwargs):
        raise ValueError("unreadable snapshot preferences")

    monkeypatch.setattr(snapshot_settings, "load_snapshot_preferences", _unreadable)
    host = DestinationHarness(_providers_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = await _open_providers(host, pilot)
        assert screen._snapshot_preferences_unavailable is True

        def revert_state() -> tuple[str, bool]:
            revert = screen.query_one("#settings-revert-category", Button)
            return str(revert.label), revert.disabled

        reload = ("Revert (r) — reload preferences", False)
        assert revert_state() == reload
        screen._update_guided_action_widgets()  # the card schedules it alone
        assert revert_state() == reload


@pytest.mark.asyncio
@private_profile_test
async def test_revert_category_refuses_a_category_its_panel_serves(request) -> None:
    """TASK-1378.2 review M3: ``_revert_category`` would pop a registered
    category's draft and toast "reverted" without repainting it; the panel's
    ``revert`` owns that discard, so the screen's path refuses."""
    screen = SettingsScreen(_providers_app())
    for category in CATEGORY_PANELS:
        with pytest.raises(ValueError, match="registered panel"):
            screen._revert_category(category)
