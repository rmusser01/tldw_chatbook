from __future__ import annotations

import copy
from contextlib import asynccontextmanager

import pytest

import tldw_chatbook.app as app_module
import tldw_chatbook.runtime_policy.bootstrap as bootstrap_module
import tldw_chatbook.UI.Screens.settings_screen as settings_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import resolve_tldw_api_config
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


class RuntimeStoreFailureInjector:
    def __init__(self) -> None:
        self.state = RuntimeSourceState()
        self.fail_with: Exception | None = None

    def load(self) -> RuntimeSourceState:
        return self.state

    def save(self, state: RuntimeSourceState) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.state = state


def _configure_settings_startup(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app.app_config["_first_run"] = False
    app._initial_tab_value = "settings"
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(
        app_module,
        "get_cli_setting",
        get_cli_setting_without_splash,
    )


@asynccontextmanager
async def _mounted_settings_screen(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_settings_startup(app, monkeypatch)
    async with app.run_test() as pilot:
        for _ in range(100):
            if getattr(app, "_initial_screen_pushed", False) and isinstance(
                app.screen,
                SettingsScreen,
            ):
                yield app.screen
                return
            await pilot.pause(0.01)
        raise AssertionError("full app did not mount its configured SettingsScreen")


def _record_notifications(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str | None]]:
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, *, severity=None, **_kwargs: notifications.append(
            (message, severity)
        ),
    )
    return notifications


def _server_result(
    base_url: str = "https://settings-switch.example.test/api/",
    auth_token: str = "settings-token",
) -> dict[str, str]:
    return {
        "action": "server",
        "base_url": base_url,
        "auth_token": auth_token,
    }


@pytest.mark.asyncio
async def test_mounted_settings_local_failure_has_no_success_notice_or_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    coordinator_calls: list[tuple[str, dict[str, object]]] = []

    async def reject_switch(runtime_backend: str, **kwargs) -> bool:
        coordinator_calls.append((runtime_backend, kwargs))
        return False

    monkeypatch.setattr(app, "handle_runtime_backend_changed", reject_switch)

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch({"action": "local"})

        assert app.screen is screen
        assert coordinator_calls == [("local", {})]
        assert notifications == []
        assert refreshes == []


@pytest.mark.asyncio
async def test_mounted_settings_local_success_notifies_and_refreshes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)

    async def accept_switch(runtime_backend: str, **kwargs) -> bool:
        assert runtime_backend == "local"
        assert kwargs == {}
        return True

    monkeypatch.setattr(app, "handle_runtime_backend_changed", accept_switch)

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch({"action": "local"})

        assert notifications == [("Runtime source set to local.", "information")]
        assert refreshes == ["refresh"]


@pytest.mark.asyncio
async def test_mounted_settings_batches_url_and_empty_token_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    saved_batches: list[dict] = []

    def reject_after_record(section_values) -> bool:
        saved_batches.append(copy.deepcopy(section_values))
        return False

    monkeypatch.setattr(
        settings_module,
        "save_settings_to_cli_config",
        reject_after_record,
    )

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        await screen._perform_runtime_source_switch(_server_result(auth_token=""))

    assert saved_batches == [
        {
            "tldw_api": {
                "base_url": "https://settings-switch.example.test/api/",
                "auth_token": "",
            }
        }
    ]


@pytest.mark.asyncio
async def test_mounted_settings_failed_batch_save_stops_before_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    old_context = app.runtime_policy
    old_snapshot = old_context.snapshot()
    old_projection = app._runtime_policy_projection_snapshot
    old_app_config = app.app_config
    old_provider_config = app.server_context_provider.app_config

    monkeypatch.setattr(
        settings_module,
        "save_settings_to_cli_config",
        lambda _values: False,
    )
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed save must not reload")
        ),
    )

    async def reject_coordinator(*_args, **_kwargs) -> bool:
        raise AssertionError("failed save must not activate")

    monkeypatch.setattr(
        app,
        "handle_runtime_backend_changed",
        reject_coordinator,
    )

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch(_server_result())

        assert app.screen is screen
        assert app.runtime_policy is old_context
        assert app.runtime_policy.snapshot() == old_snapshot
        assert app._runtime_policy_projection_snapshot == old_projection
        assert app.app_config is old_app_config
        assert app.server_context_provider.app_config is old_provider_config
        assert notifications == [
            (
                "Server settings could not be saved; "
                "the previous source remains active.",
                "error",
            )
        ]
        assert refreshes == []


@pytest.mark.asyncio
async def test_mounted_settings_reload_failure_is_bounded_and_retains_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    old_context = app.runtime_policy
    old_snapshot = old_context.snapshot()
    old_projection = app._runtime_policy_projection_snapshot
    old_app_config = app.app_config
    old_provider_config = app.server_context_provider.app_config
    url_sentinel = "https://reload-url-sentinel.example.test/api/"
    token_sentinel = "reload-token-sentinel"
    path_sentinel = "/private/reload-path-sentinel/config.toml"
    exception_sentinel = "reload-exception-sentinel"

    monkeypatch.setattr(
        settings_module,
        "save_settings_to_cli_config",
        lambda _values: True,
    )

    def fail_reload(**_kwargs):
        raise RuntimeError(
            f"{exception_sentinel} {path_sentinel} {url_sentinel} {token_sentinel}"
        )

    monkeypatch.setattr(settings_module, "load_settings", fail_reload)

    async def reject_coordinator(*_args, **_kwargs) -> bool:
        raise AssertionError("reload failure must not activate")

    monkeypatch.setattr(
        app,
        "handle_runtime_backend_changed",
        reject_coordinator,
    )
    async with _mounted_settings_screen(app, monkeypatch) as screen:
        diagnostics: list[str] = []

        def record_warning(message: str, *args, **_kwargs) -> None:
            diagnostics.append(message % args)

        monkeypatch.setattr(settings_module.logger, "warning", record_warning)
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch(
            _server_result(url_sentinel, token_sentinel)
        )

        assert app.screen is screen
        assert app.runtime_policy is old_context
        assert app.runtime_policy.snapshot() == old_snapshot
        assert app._runtime_policy_projection_snapshot == old_projection
        assert app.app_config is old_app_config
        assert app.server_context_provider.app_config is old_provider_config
        assert notifications == [
            (
                "Server settings were saved but could not be activated; "
                "the previous source remains active.",
                "error",
            )
        ]
        assert refreshes == []

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert "exception_category=RuntimeError" in diagnostic
    assert exception_sentinel not in diagnostic
    assert path_sentinel not in diagnostic
    assert url_sentinel not in diagnostic
    assert token_sentinel not in diagnostic


@pytest.mark.asyncio
async def test_mounted_settings_passes_reloaded_config_only_to_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    old_app_config = app.app_config
    old_provider_config = app.server_context_provider.app_config
    refreshed_config = copy.deepcopy(old_app_config)
    refreshed_config["tldw_api"] = {
        "base_url": "https://override-only.example.test/api",
        "auth_token": "override-only-token",
    }
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        settings_module,
        "save_settings_to_cli_config",
        lambda _values: True,
    )
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda **_kwargs: refreshed_config,
    )

    async def reject_after_observation(runtime_backend: str, **kwargs) -> bool:
        assert app.app_config is old_app_config
        assert app.server_context_provider.app_config is old_provider_config
        calls.append((runtime_backend, kwargs))
        return False

    monkeypatch.setattr(
        app,
        "handle_runtime_backend_changed",
        reject_after_observation,
    )

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch(_server_result())

        assert calls == [
            (
                "server",
                {"app_config_override": refreshed_config},
            )
        ]
        assert app.app_config is old_app_config
        assert app.server_context_provider.app_config is old_provider_config
        assert notifications == []
        assert refreshes == []


@pytest.mark.asyncio
async def test_mounted_settings_coordinator_rejection_leaves_saved_file_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    old_app_config = app.app_config
    old_provider_config = app.server_context_provider.app_config
    old_context = app.runtime_policy
    sync_calls: list[dict[str, object]] = []

    async def reject_after_file_save(runtime_backend: str, **kwargs) -> bool:
        assert runtime_backend == "server"
        api_config = resolve_tldw_api_config(kwargs["app_config_override"])
        assert api_config["base_url"] == ("https://retry-file.example.test/api/")
        return False

    monkeypatch.setattr(
        app,
        "handle_runtime_backend_changed",
        reject_after_file_save,
    )

    async def unexpected_sync(**kwargs):
        sync_calls.append(kwargs)
        raise AssertionError("coordinator rejection must not prepare Sync v2")

    monkeypatch.setattr(
        app.sync_scope_service,
        "prepare_sync_v2_profile_mode",
        unexpected_sync,
    )

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch(
            _server_result(
                "https://retry-file.example.test/api/",
                "retry-file-token",
            )
        )

        saved_config = settings_module.load_settings(force_reload=True)
        saved_api_config = resolve_tldw_api_config(saved_config)
        assert saved_api_config["base_url"] == ("https://retry-file.example.test/api/")
        assert saved_api_config["auth_token"] == "retry-file-token"
        assert app.screen is screen
        assert app.runtime_policy is old_context
        assert app.app_config is old_app_config
        assert app.server_context_provider.app_config is old_provider_config
        assert sync_calls == []
        assert notifications == []
        assert refreshes == []


@pytest.mark.asyncio
async def test_mounted_settings_validates_committed_server_id_before_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    context = app.runtime_policy
    _, revision = context.snapshot()
    assert context.commit_state(
        RuntimeSourceState(
            active_source="server",
            active_server_id=None,
            server_configured=False,
        ),
        expected_revision=revision,
    )
    refreshed_config = copy.deepcopy(app.app_config)
    refreshed_config["tldw_api"] = {
        "base_url": "https://missing-id.example.test/api",
        "auth_token": "",
    }
    monkeypatch.setattr(
        settings_module,
        "save_settings_to_cli_config",
        lambda _values: True,
    )
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda **_kwargs: refreshed_config,
    )

    async def report_success_without_commit(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(
        app,
        "handle_runtime_backend_changed",
        report_success_without_commit,
    )
    sync_calls: list[dict[str, object]] = []

    async def unexpected_sync(**kwargs):
        sync_calls.append(kwargs)

    monkeypatch.setattr(
        app.sync_scope_service,
        "prepare_sync_v2_profile_mode",
        unexpected_sync,
    )

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch(_server_result())

        assert notifications == [
            (
                "Server could not be bound from the entered URL.",
                "error",
            )
        ]
        assert sync_calls == []
        assert refreshes == []


@pytest.mark.asyncio
async def test_mounted_settings_success_prepares_sync_for_committed_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    refreshed_config = copy.deepcopy(app.app_config)
    refreshed_config["tldw_api"] = {
        "base_url": "https://sync-ready.example.test/api/",
        "auth_token": "sync-ready-token",
    }
    monkeypatch.setattr(
        settings_module,
        "save_settings_to_cli_config",
        lambda _values: True,
    )
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda **_kwargs: refreshed_config,
    )
    sync_calls: list[dict[str, object]] = []

    async def prepare_sync(**kwargs):
        sync_calls.append(kwargs)
        return {"dataset_id": "dataset-ready"}

    monkeypatch.setattr(
        app.sync_scope_service,
        "prepare_sync_v2_profile_mode",
        prepare_sync,
    )

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch(_server_result())

        assert app.runtime_policy.state.active_server_id == (
            "https://sync-ready.example.test/api"
        )
        assert sync_calls == [
            {
                "profile_mode": "local_first_sync",
                "server_profile_id": "https://sync-ready.example.test/api",
                "display_name": sync_calls[0]["display_name"],
            }
        ]
        assert sync_calls[0]["display_name"]
        assert notifications == [
            (
                "Server activated; Sync v2 prepared (dataset dataset-ready).",
                "information",
            )
        ]
        assert refreshes == ["refresh"]


@pytest.mark.asyncio
async def test_mounted_settings_sync_failure_diagnostics_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    url_sentinel = "https://sync-failure-url.example.test/api/"
    token_sentinel = "sync-failure-token"
    path_sentinel = "/private/sync-failure-path/state.json"
    exception_sentinel = "sync-failure-exception"
    server_id_sentinel = "https://sync-failure-url.example.test/api"
    refreshed_config = copy.deepcopy(app.app_config)
    refreshed_config["tldw_api"] = {
        "base_url": url_sentinel,
        "auth_token": token_sentinel,
    }
    monkeypatch.setattr(
        settings_module,
        "save_settings_to_cli_config",
        lambda _values: True,
    )
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda **_kwargs: refreshed_config,
    )

    async def fail_sync(**_kwargs):
        raise RuntimeError(
            f"{exception_sentinel} {path_sentinel} {url_sentinel} "
            f"{token_sentinel} {server_id_sentinel}"
        )

    monkeypatch.setattr(
        app.sync_scope_service,
        "prepare_sync_v2_profile_mode",
        fail_sync,
    )

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        diagnostics: list[str] = []

        def record_warning(message: str, *args, **_kwargs) -> None:
            diagnostics.append(message % args)

        monkeypatch.setattr(settings_module.logger, "warning", record_warning)
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch(
            _server_result(url_sentinel, token_sentinel)
        )

        assert app.screen is screen
        assert app.runtime_policy.state.active_source == "server"
        assert app.runtime_policy.state.active_server_id == server_id_sentinel
        assert notifications == [
            (
                "Server activated, but Sync v2 setup could not be completed.",
                "warning",
            )
        ]
        assert refreshes == ["refresh"]

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert "exception_category=RuntimeError" in diagnostic
    for sentinel in (
        exception_sentinel,
        path_sentinel,
        url_sentinel,
        token_sentinel,
        server_id_sentinel,
    ):
        assert sentinel not in diagnostic
        assert sentinel not in notifications[0][0]


@pytest.mark.asyncio
async def test_mounted_settings_end_to_end_rebinds_existing_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    original_context = app.runtime_policy
    sync_calls: list[dict[str, object]] = []

    async def prepare_sync(**kwargs):
        sync_calls.append(kwargs)
        return {"dataset_id": "dataset-e2e"}

    monkeypatch.setattr(
        app.sync_scope_service,
        "prepare_sync_v2_profile_mode",
        prepare_sync,
    )
    base_url = "https://settings-e2e.example.test/api/"

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )

        await screen._perform_runtime_source_switch(
            _server_result(base_url, "settings-e2e-token")
        )

        saved_config = settings_module.load_settings(force_reload=True)
        active_api_config = resolve_tldw_api_config(app.app_config)
        saved_api_config = resolve_tldw_api_config(saved_config)
        assert app.screen is screen
        assert app.runtime_policy is original_context
        assert app.service_policy_enforcer.current_state() is original_context.state
        assert app.server_context_provider.runtime_context is original_context
        assert app.app_config is app.server_context_provider.app_config
        assert active_api_config["base_url"] == base_url
        assert saved_api_config["base_url"] == base_url
        assert saved_api_config["auth_token"] == "settings-e2e-token"
        assert app.current_runtime_backend == "server"
        assert app.runtime_backend == "server"
        assert app.active_server_id == "https://settings-e2e.example.test/api"
        assert original_context.state.active_server_id == app.active_server_id
        target = app.server_context_provider.target_store.get_target(
            app.active_server_id
        )
        assert target is not None
        assert target.is_default is True
        assert sync_calls[0]["server_profile_id"] == app.active_server_id
        assert notifications == [
            (
                "Server activated; Sync v2 prepared (dataset dataset-e2e).",
                "information",
            )
        ]
        assert refreshes == ["refresh"]


@pytest.mark.asyncio
async def test_mounted_settings_store_failure_keeps_memory_and_saved_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RuntimeStoreFailureInjector()
    monkeypatch.setattr(
        bootstrap_module,
        "RuntimeSourceStateStore",
        lambda *_args, **_kwargs: store,
    )
    app = TldwCli()
    notifications = _record_notifications(app, monkeypatch)
    provider = app.server_context_provider
    original_context = app.runtime_policy
    old_snapshot = original_context.snapshot()
    old_projection = app._runtime_policy_projection_snapshot
    old_app_config = app.app_config
    old_provider_config = provider.app_config
    old_targets = provider.target_store.list_targets()

    class CachedClientSentinel:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    cached_client = CachedClientSentinel()
    cached_key = object()
    provider._cached_client = cached_client
    provider._cached_client_key = cached_key
    url_sentinel = "https://store-failure-url.example.test/api/"
    token_sentinel = "store-failure-token"
    path_sentinel = "/private/store-failure-path/config.toml"
    server_id_sentinel = "https://store-failure-url.example.test/api"
    exception_sentinel = "store-failure-exception"
    store.fail_with = OSError(
        f"{exception_sentinel} {path_sentinel} {url_sentinel} "
        f"{token_sentinel} {server_id_sentinel}"
    )

    async with _mounted_settings_screen(app, monkeypatch) as screen:
        notifications.clear()
        refreshes: list[str] = []
        monkeypatch.setattr(
            screen,
            "_refresh_manual_sync_rows",
            lambda: refreshes.append("refresh"),
        )
        warnings: list[str] = []
        sink = app_module.logger.add(
            warnings.append,
            level="WARNING",
            format="{message}",
        )
        try:
            await screen._perform_runtime_source_switch(
                _server_result(url_sentinel, token_sentinel)
            )
        finally:
            app_module.logger.remove(sink)

        saved_config = settings_module.load_settings(force_reload=True)
        saved_api_config = resolve_tldw_api_config(saved_config)
        assert app.screen is screen
        assert app.runtime_policy is original_context
        assert original_context.snapshot() == old_snapshot
        assert app._runtime_policy_projection_snapshot == old_projection
        assert app.app_config is old_app_config
        assert provider.app_config is old_provider_config
        assert provider._cached_client is cached_client
        assert provider._cached_client_key is cached_key
        assert cached_client.close_calls == 0
        assert provider.target_store.list_targets() == old_targets
        assert saved_api_config["base_url"] == url_sentinel
        assert saved_api_config["auth_token"] == token_sentinel
        assert not any(
            message.startswith("Server activated")
            for message, _severity in notifications
        )
        assert refreshes == []

    commit_warnings = [
        warning
        for warning in warnings
        if "Runtime source change was not committed" in warning
    ]
    assert len(commit_warnings) == 1
    assert "exception_category=OSError" in commit_warnings[0]
    assert exception_sentinel not in commit_warnings[0]
    assert path_sentinel not in commit_warnings[0]
    assert url_sentinel not in commit_warnings[0]
    assert token_sentinel not in commit_warnings[0]
    assert server_id_sentinel not in commit_warnings[0]
