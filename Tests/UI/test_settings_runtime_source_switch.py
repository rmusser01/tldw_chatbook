"""Server-switch save flow writes the token to the credential store eagerly."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def _fake_self(app, server_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        app_instance=app,
        app=app,
        _runtime_source_state=lambda: SimpleNamespace(active_server_id=server_id),
        _refresh_manual_sync_rows=lambda: None,
    )


def _fake_app(*, switched: bool = True) -> tuple:
    app = SimpleNamespace()

    async def _switch(*args, **kwargs):
        return switched

    app.handle_runtime_backend_changed = MagicMock(side_effect=_switch)
    notified: list[tuple[str, str]] = []
    app.notify = lambda message, severity="information": notified.append(
        (severity, message)
    )
    provider = MagicMock()
    provider.store_static_server_credential = MagicMock(return_value="bearer_token")
    app.server_context_provider = provider
    app.sync_scope_service = None
    return app, notified, provider


def test_switch_persists_token_to_credential_store(monkeypatch, tmp_path):
    app, _notified, provider = _fake_app()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    fake = _fake_self(app, "https://server.example.com/api")

    asyncio.run(
        SettingsScreen._perform_runtime_source_switch(
            fake,
            {
                "action": "activate",
                "base_url": "https://server.example.com/api",
                "auth_token": "tok-123",
            },
        )
    )

    provider.store_static_server_credential.assert_called_once_with(
        "https://server.example.com/api", "tok-123"
    )


def test_switch_notifies_when_keyring_write_fails(monkeypatch, tmp_path):
    app, notified, provider = _fake_app()
    provider.store_static_server_credential.side_effect = RuntimeError("boom")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    fake = _fake_self(app, "https://server.example.com/api")

    asyncio.run(
        SettingsScreen._perform_runtime_source_switch(
            fake,
            {
                "action": "activate",
                "base_url": "https://server.example.com/api",
                "auth_token": "tok-123",
            },
        )
    )

    assert any(
        severity == "warning" and "keyring" in message for severity, message in notified
    )


def test_switch_without_token_skips_credential_store(monkeypatch, tmp_path):
    app, _notified, provider = _fake_app()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    fake = _fake_self(app, "https://server.example.com/api")

    asyncio.run(
        SettingsScreen._perform_runtime_source_switch(
            fake,
            {
                "action": "activate",
                "base_url": "https://server.example.com/api",
                "auth_token": "",
            },
        )
    )

    provider.store_static_server_credential.assert_not_called()
