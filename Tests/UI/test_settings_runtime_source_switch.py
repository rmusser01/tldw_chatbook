"""Server-switch save flow writes the token to the credential store eagerly."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.Sync_Interop.sync_state import SyncV2EnrollmentError
from tldw_chatbook.UI.Screens import settings_screen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.Settings_Widgets.server_switch_modal import ServerSwitchModal
from tldw_chatbook.Widgets.state_checkbox import StateCheckbox


@pytest.fixture(autouse=True)
def _saved_server_config(monkeypatch):
    monkeypatch.setattr(settings_screen, "save_settings_to_cli_config", lambda _: True)
    monkeypatch.setattr(settings_screen, "load_settings", lambda **_: {})


def _fake_self(app, server_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        app_instance=app,
        app=app,
        _runtime_source_state=lambda: SimpleNamespace(active_server_id=server_id),
        _refresh_manual_sync_rows=lambda: None,
        _active_workspace_record=lambda: None,
        _active_sync_scope=lambda *args: {
            "server_profile_id": server_id,
            "authenticated_principal_id": "user-a",
            "workspace_scope": "workspace-default",
        },
        _manual_sync_setup_issue=None,
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


def test_switch_persists_token_to_credential_store():
    app, _notified, provider = _fake_app()
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


def test_switch_notifies_when_keyring_write_fails():
    app, notified, provider = _fake_app()
    provider.store_static_server_credential.side_effect = RuntimeError("boom")
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


def test_switch_without_token_skips_credential_store():
    app, _notified, provider = _fake_app()
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


def test_activate_server_requires_separate_sync_consent():
    app, _notified, _provider = _fake_app()
    prepare = AsyncMock()
    app.sync_scope_service = SimpleNamespace(prepare_sync_v2_profile_mode=prepare)
    fake = _fake_self(app, "server-a")

    asyncio.run(
        SettingsScreen._perform_runtime_source_switch(
            fake,
            {"action": "activate", "base_url": "https://server.example.com"},
        )
    )

    app.handle_runtime_backend_changed.assert_called_once()
    prepare.assert_not_called()


@pytest.mark.asyncio
async def test_approved_activation_enrolls_the_same_scope_used_by_manual_sync():
    app, notified, _provider = _fake_app()
    prepare = AsyncMock()
    app.sync_scope_service = SimpleNamespace(prepare_sync_v2_profile_mode=prepare)
    fake = _fake_self(app, "server-a")

    await SettingsScreen._perform_runtime_source_switch(
        fake,
        {
            "action": "activate",
            "base_url": "https://server.example.com",
            "approve_server_trusted_sync": True,
        },
    )

    prepare.assert_awaited_once()
    for key, value in fake._active_sync_scope().items():
        assert prepare.await_args.kwargs[key] == value
    assert prepare.await_args.kwargs["encryption_policy"] == "server_trusted_v1"
    assert prepare.await_args.kwargs["enrollment_only"] is True
    assert any("preview is available" in message for _, message in notified)


@pytest.mark.asyncio
async def test_manual_sync_keeps_policy_refusal_visible_after_preview_refresh():
    app, _notified, _provider = _fake_app()
    refusal = "Manual Sync is unavailable. The server does not advertise ready server-trusted encryption."
    app.sync_scope_service = SimpleNamespace(
        prepare_sync_v2_profile_mode=AsyncMock(
            side_effect=SyncV2EnrollmentError(refusal)
        )
    )
    app.manual_sync_control_service = SimpleNamespace(
        preview=MagicMock(), run_once=AsyncMock()
    )
    fake = _fake_self(app, "server-a")
    applied_rows = []
    fake._apply_manual_sync_rows = applied_rows.append

    await SettingsScreen._perform_runtime_source_switch(
        fake,
        {
            "action": "activate",
            "base_url": "https://server.example.com",
            "approve_server_trusted_sync": True,
        },
    )

    rows = dict(SettingsScreen._manual_sync_rows(fake))
    assert rows["Manual sync status"] == "blocked"
    assert rows["Manual sync preview"] == refusal
    app.manual_sync_control_service.preview.assert_not_called()

    await SettingsScreen._run_manual_sync_once(fake)

    assert dict(applied_rows[-1])["Manual sync result"] == refusal
    app.manual_sync_control_service.run_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_server_switch_modal_allows_explicit_consent_at_80_by_24():
    app = App()
    modal = ServerSwitchModal(current_base_url="https://server.example.com")
    decisions = []
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(modal, decisions.append)
        consent = modal.query_one("#server-switch-sync-consent", StateCheckbox)
        assert consent.value is False
        consent.focus()
        await pilot.press("space")
        modal.query_one("#server-switch-modal").scroll_end(animate=False)
        await pilot.pause()
        modal.query_one("#server-switch-activate", Button).focus()
        await pilot.pause()
        assert await pilot.click("#server-switch-activate")
        await pilot.pause()

    assert decisions[0]["approve_server_trusted_sync"] is True


@pytest.mark.parametrize(
    "status, expected_title",
    [
        ("preview-only", None),
        ("blocked", None),
        ("setup", "Prepare Notes organization"),
    ],
)
def test_manual_sync_action_only_offers_supported_operations(status, expected_title):
    app, _notified, _provider = _fake_app()
    app.push_screen = MagicMock()
    fake = _fake_self(app, "server-a")
    fake.manual_sync_rows = (("Manual sync status", status),)
    fake._manual_sync_action = lambda: SettingsScreen._manual_sync_action(fake)

    SettingsScreen.handle_manual_sync_run(fake, MagicMock())

    if expected_title is None:
        app.push_screen.assert_not_called()
    else:
        dialog = app.push_screen.call_args.args[0]
        assert dialog.title == expected_title
        assert "adoption conflicts" in dialog.message
