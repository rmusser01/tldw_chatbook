import pytest
from textual.widgets import Select, Static

from tldw_chatbook.UI.Settings_Modules.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_network_defaults import load_network_tls

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)

pytestmark = pytest.mark.ui


async def test_network_category_rejects_missing_ca_and_saves_valid_one(
    tmp_path, monkeypatch
):
    saved: list[dict] = []

    def _capture(sections):
        saved.append({k: dict(v) for k, v in dict(sections).items()})
        return True

    monkeypatch.setattr(SettingsConfigAdapter, "save_sections", staticmethod(_capture))
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "network")
        screen = _active_destination_screen(host)
        assert screen.query_one("#settings-network-ssl-mode") is not None

        screen._network_pending["mode"] = "custom-ca"
        screen._network_pending["ca_bundle_path"] = "/definitely/not/here.pem"
        screen.action_settings_save_category()  # verified sync: -> None
        assert saved == []  # invalid path rejected, nothing written

        ca = tmp_path / "corp.pem"
        ca.write_text("# corp ca")
        screen._network_pending["ca_bundle_path"] = str(ca)
        screen.action_settings_save_category()
        assert saved == [{"network": {"ssl_verify": str(ca)}}]
        assert screen._network_pending == {}  # draft cleared after successful save
        # qodo PR #2223 bug 5: the app's in-memory config mapping must
        # reflect the save, or the next detail render shows the stale
        # pre-save mode/path.
        assert load_network_tls(screen._app_config_mapping()).ca_bundle_path == str(ca)

        # Fix round 1: the Network banner/guided rows must not claim
        # read-only -- `s` saves, and the badge names that save model.
        banner_text = str(
            screen.query_one("#settings-category-state-banner", Static).renderable
        )
        assert "Read-only" not in banner_text
        assert "save with s" in banner_text
        guided_text = str(
            screen.query_one("#settings-guided-action-state", Static).renderable
        )
        assert "Read-only" not in guided_text


async def test_network_category_hop_reseeds_select_from_pending_mode():
    """Final-review fix: a category hop must not desync the Select from pending.

    ``_network_pending`` survives a category switch, but
    ``_render_network_detail`` used to seed the Select from the LOADED
    config -- so a hop away and back repainted the Select with the loaded
    mode while ``s`` still saved the pending one. The hop below is the real
    mechanism (sidebar click -> ``watch_active_category`` -> detail-pane
    region rebuild), not a hand-invoked compose.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "network")
        screen = _active_destination_screen(host)
        screen._network_pending["mode"] = "off"

        await _click_settings_category(pilot, "appearance")
        await _click_settings_category(pilot, "network")

        select = screen.query_one("#settings-network-ssl-mode", Select)
        assert select.value == "off"


# --- ADR-303 pins (TASK-1378.1): Network's commit behaviour as it is today. ---
# These run unmodified before and after Network moves onto its registered panel.


def _capture_toasts(screen, monkeypatch) -> list[tuple[str, str]]:
    toasts: list[tuple[str, str]] = []

    def _notify(message, *args, severity="information", **kwargs):
        toasts.append((str(message), severity))

    monkeypatch.setattr(screen.app, "notify", _notify)
    return toasts


async def test_network_commit_model_copy_is_pinned():
    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "network")
        screen = _active_destination_screen(host)
        category = SettingsCategoryId.NETWORK
        assert screen._persistence_badge(category) == "Pending — save with s"
        assert screen._guided_action_message(category) == (
            "Save with s: the TLS trust setting applies to newly created "
            "clients; already-open connections keep their trust policy "
            "until restart."
        )
        assert screen._category_state_banner_text(category) == (
            "State: Pending — save with s | Review readiness across Settings categories."
        )
        assert screen._category_action_supported(category, "save") is True
        assert screen._category_action_supported(category, "revert") is False


async def test_network_pending_edits_leave_without_prompt_and_outside_quit_guard():
    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId
    from tldw_chatbook.UI.Settings_Modules.settings_quit_guard import (
        unsaved_settings_titles,
    )

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "network")
        screen = _active_destination_screen(host)
        screen._network_pending["mode"] = "off"
        # Staged, but no draft machinery: not counted as unsaved, not in the
        # quit guard, and navigating away raises no confirmation.
        assert screen._category_has_unsaved_changes(SettingsCategoryId.NETWORK) is False
        assert "Network" not in unsaved_settings_titles(screen)
        await _click_settings_category(pilot, "appearance")
        assert screen.active_category == "appearance"
        assert screen._network_pending == {"mode": "off"}


async def test_network_save_failure_keeps_pending_and_revert_is_guidance(monkeypatch):
    monkeypatch.setattr(
        SettingsConfigAdapter, "save_sections", staticmethod(lambda sections: False)
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "network")
        screen = _active_destination_screen(host)
        toasts = _capture_toasts(screen, monkeypatch)
        screen._network_pending["mode"] = "off"
        screen.action_settings_save_category()
        assert toasts == [("Failed to save Network TLS setting.", "error")]
        assert screen._network_pending == {"mode": "off"}

        toasts.clear()
        screen.action_settings_revert_category()
        assert screen._network_pending == {"mode": "off"}  # r never discards here
        assert toasts == [("No Settings changes to revert.", "information")]


async def test_network_invalid_ca_and_success_toasts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        SettingsConfigAdapter, "save_sections", staticmethod(lambda sections: True)
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "network")
        screen = _active_destination_screen(host)
        toasts = _capture_toasts(screen, monkeypatch)
        screen._network_pending["mode"] = "custom-ca"
        screen._network_pending["ca_bundle_path"] = str(tmp_path / "missing.pem")
        screen.action_settings_save_category()
        assert len(toasts) == 1 and toasts[0][1] == "error"
        assert screen._network_pending["mode"] == "custom-ca"

        toasts.clear()
        screen._network_pending.clear()
        screen._network_pending["mode"] = "off"
        screen.action_settings_save_category()
        assert toasts == [("Network TLS setting saved.", "information")]
        assert screen._network_pending == {}
