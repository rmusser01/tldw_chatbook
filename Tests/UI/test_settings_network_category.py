import pytest
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

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
