import pytest
from textual.widgets import Select, Static

from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
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


async def test_network_category_rejects_missing_ca_and_saves_valid_one(tmp_path, monkeypatch):
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
        assert (
            load_network_tls(screen._app_config_mapping()).ca_bundle_path
            == str(ca)
        )

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
