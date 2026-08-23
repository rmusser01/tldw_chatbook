"""The test app factory must hand Console a disk-shaped configuration."""

from __future__ import annotations

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def test_factory_config_carries_the_disk_load_markers():
    """The snapshot must look disk-loaded, or Console never refreshes it."""
    app = _build_test_app()

    assert ChatScreen._console_config_snapshot_is_disk_loaded(app.app_config) is True
