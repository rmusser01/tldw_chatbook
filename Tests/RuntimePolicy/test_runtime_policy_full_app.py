from __future__ import annotations

from collections.abc import Callable
import logging
from datetime import datetime, timezone

import pytest
import pytest_asyncio

import tldw_chatbook.app as app_module
import tldw_chatbook.runtime_policy.bootstrap as bootstrap_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.runtime_policy.bootstrap import (
    _apply_runtime_policy_to_app,
    load_runtime_policy_for_app,
    set_authoritative_runtime_source,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.MediaWindow_v2 import MediaWindow
from tldw_chatbook.UI.Screens.media_screen import MediaScreen


class ControllableRuntimeStore:
    def __init__(self) -> None:
        self.loaded_state = RuntimeSourceState()
        self.saved_states: list[RuntimeSourceState] = []
        self.events: list[str] = []
        self.fail_with: Exception | None = None
        self.before_save: Callable[[RuntimeSourceState], None] | None = None

    def load(self) -> RuntimeSourceState:
        return self.loaded_state

    def save(self, state: RuntimeSourceState) -> None:
        self.events.append("store_save")
        if self.before_save is not None:
            self.before_save(state)
        if self.fail_with is not None:
            raise self.fail_with
        self.loaded_state = state
        self.saved_states.append(state)


@pytest_asyncio.fixture
async def full_app_with_controllable_runtime_store(
    monkeypatch: pytest.MonkeyPatch,
):
    store = ControllableRuntimeStore()
    monkeypatch.setattr(
        bootstrap_module,
        "RuntimeSourceStateStore",
        lambda *_args, **_kwargs: store,
    )
    app = TldwCli()
    store.events.clear()
    store.saved_states.clear()

    yield app, store

    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


def _configure_full_app_media_startup(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app.app_config["_first_run"] = False
    app._initial_tab_value = "media"
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


async def _wait_for_mounted_media_screen(app: TldwCli, pilot) -> MediaScreen:
    for _ in range(100):
        if getattr(app, "_initial_screen_pushed", False) and isinstance(
            app.screen,
            MediaScreen,
        ):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("full app did not mount its configured MediaScreen")


@pytest.mark.asyncio
async def test_full_app_wires_one_runtime_context_to_long_lived_consumers(
    app_with_cleanup: TldwCli,
) -> None:
    app = app_with_cleanup

    assert app.server_context_provider.runtime_context is app.runtime_policy
    assert app.active_server_capability_service.runtime_context is app.runtime_policy
    assert app.home_active_work_adapter.runtime_policy is app.runtime_policy
    assert app.service_policy_enforcer.current_state() is app.runtime_policy.state
    assert app.server_context_provider.target_store is app.unified_mcp_target_store
    assert app.server_context_provider.credential_store is app.server_credential_store
    assert app.server_context_provider.app_config is app.app_config


@pytest.mark.asyncio
async def test_full_app_rejects_second_runtime_context_install_before_store_io(
    app_with_cleanup: TldwCli,
) -> None:
    app = app_with_cleanup
    installed_context = app.runtime_policy

    class NeverLoadStore:
        def __init__(self) -> None:
            self.load_calls = 0
            self.save_calls = 0

        def load(self) -> RuntimeSourceState:
            self.load_calls += 1
            raise AssertionError("installed context must reject before load")

        def save(self, state: RuntimeSourceState) -> None:
            self.save_calls += 1
            raise AssertionError("installed context must reject before save")

    store = NeverLoadStore()

    with pytest.raises(RuntimeError, match="already installed"):
        load_runtime_policy_for_app(app, store=store)

    assert app.runtime_policy is installed_context
    assert store.load_calls == 0
    assert store.save_calls == 0


@pytest.mark.asyncio
async def test_full_app_wiring_uses_unavailable_store_when_secure_store_is_missing(
    app_with_cleanup: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.app as app_module
    from tldw_chatbook.runtime_policy.server_credentials import (
        CredentialStoreUnavailable,
        UnavailableServerCredentialStore,
    )

    app = app_with_cleanup

    def raise_unavailable() -> None:
        raise CredentialStoreUnavailable(
            "No secure OS-backed credential store is available."
        )

    monkeypatch.setattr(
        app_module,
        "build_default_server_credential_store",
        raise_unavailable,
    )

    app._wire_server_context_provider()

    assert isinstance(
        app.server_credential_store,
        UnavailableServerCredentialStore,
    )
    assert app.server_context_provider.credential_store is app.server_credential_store
    assert app.server_context_provider.runtime_context is app.runtime_policy


@pytest.mark.asyncio
async def test_full_app_projection_attributes_reject_direct_and_alias_assignment(
    app_with_cleanup: TldwCli,
) -> None:
    app = app_with_cleanup
    alias = app

    with pytest.raises(AttributeError):
        app.current_runtime_backend = "server"
    with pytest.raises(AttributeError):
        alias.runtime_backend = "server"
    with pytest.raises(AttributeError):
        alias.active_server_id = "server-alias"


@pytest.mark.asyncio
async def test_full_app_projection_boundary_publishes_one_coherent_state(
    app_with_cleanup: TldwCli,
) -> None:
    app = app_with_cleanup
    state = RuntimeSourceState(
        active_source="server",
        active_server_id="server-projection",
        server_configured=True,
    )

    _apply_runtime_policy_to_app(app, state)

    assert (
        app.current_runtime_backend,
        app.runtime_backend,
        app.active_server_id,
    ) == ("server", "server", "server-projection")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("publisher_mode", "expected_exception"),
    (
        ("missing", AttributeError),
        ("descriptor_failure", AttributeError),
        ("non_callable", TypeError),
        ("throws", RuntimeError),
    ),
)
async def test_full_app_projection_boundary_never_falls_back_to_public_writes(
    app_with_cleanup: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
    publisher_mode: str,
    expected_exception: type[Exception],
) -> None:
    app = app_with_cleanup
    publisher_name = "_publish_runtime_policy_projection"
    before = (
        app.current_runtime_backend,
        app.runtime_backend,
        app.active_server_id,
    )

    if publisher_mode == "missing":
        monkeypatch.delattr(TldwCli, publisher_name, raising=False)
    elif publisher_mode == "descriptor_failure":

        class RaisingPublisherDescriptor:
            def __get__(self, instance, owner):
                raise AttributeError("publisher descriptor failed")

        monkeypatch.setattr(
            TldwCli,
            publisher_name,
            RaisingPublisherDescriptor(),
            raising=False,
        )
    elif publisher_mode == "non_callable":
        monkeypatch.setattr(TldwCli, publisher_name, None, raising=False)
    else:

        def raise_from_publisher(
            _app: TldwCli,
            _state: RuntimeSourceState,
        ) -> None:
            raise RuntimeError("publisher failed")

        monkeypatch.setattr(
            TldwCli,
            publisher_name,
            raise_from_publisher,
            raising=False,
        )

    with pytest.raises(expected_exception):
        _apply_runtime_policy_to_app(
            app,
            RuntimeSourceState(
                active_source="server",
                active_server_id="server-should-not-publish",
                server_configured=True,
            ),
        )

    assert (
        app.current_runtime_backend,
        app.runtime_backend,
        app.active_server_id,
    ) == before


@pytest.mark.asyncio
async def test_full_app_authoritative_source_change_clears_stale_probe_state(
    app_with_cleanup: TldwCli,
) -> None:
    app = app_with_cleanup
    context = app.runtime_policy
    _, revision = context.snapshot()
    old_state = RuntimeSourceState(
        active_source="server",
        active_server_id="https://old.example.com/api",
        server_configured=True,
        server_reachability="reachable",
        server_reachability_checked_at=datetime(
            2026, 4, 21, 12, 0, tzinfo=timezone.utc
        ),
        server_auth_state="authenticated",
        server_auth_checked_at=datetime(2026, 4, 21, 12, 5, tzinfo=timezone.utc),
        last_known_server_label="old.example.com",
    )
    assert context.commit_state(old_state, expected_revision=revision)
    app.app_config = {
        "tldw_api": {
            "base_url": "https://new.example.com/v1/",
        }
    }

    updated_state = set_authoritative_runtime_source(
        context,
        "server",
        app_config=app.app_config,
    )

    assert updated_state.active_source == "server"
    assert updated_state.active_server_id == "https://new.example.com/v1"
    assert updated_state.server_configured is True
    assert updated_state.server_reachability == "unknown"
    assert updated_state.server_reachability_checked_at is None
    assert updated_state.server_auth_state == "unknown"
    assert updated_state.server_auth_checked_at is None
    assert context.state == updated_state


@pytest.mark.asyncio
async def test_full_app_coordinator_store_failure_retains_every_precommit_surface(
    full_app_with_controllable_runtime_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store = full_app_with_controllable_runtime_store
    provider = app.server_context_provider
    old_app_config = app.app_config
    old_provider_config = provider.app_config
    old_context_snapshot = app.runtime_policy.snapshot()
    old_projection = app._runtime_policy_projection_snapshot
    old_targets = provider.target_store.list_targets()
    old_media_backend = app.media_runtime_state.runtime_backend

    class CachedClientSentinel:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    cached_client = CachedClientSentinel()
    cached_key = object()
    provider._cached_client = cached_client
    provider._cached_client_key = cached_key
    candidate_config = {
        "tldw_api": {
            "base_url": "https://candidate.example.test/api",
            "bearer_token": "candidate-secret",
            "auth_mode": "bearer",
        }
    }
    save_sentinel = "FULL-APP-SAVE-SENTINEL"
    store.fail_with = OSError(save_sentinel)

    def verify_precommit_surfaces(_candidate: RuntimeSourceState) -> None:
        assert app.app_config is old_app_config
        assert provider.app_config is old_provider_config
        assert app.runtime_policy.snapshot() == old_context_snapshot
        assert app._runtime_policy_projection_snapshot == old_projection
        assert provider._cached_client is cached_client
        assert provider._cached_client_key is cached_key
        assert provider.target_store.list_targets() == old_targets

    store.before_save = verify_precommit_surfaces
    screen_calls: list[str] = []

    async def record_screen_callback(
        _screen: MediaScreen,
        runtime_backend: str,
    ) -> None:
        screen_calls.append(runtime_backend)

    monkeypatch.setattr(
        MediaScreen,
        "handle_runtime_backend_changed",
        record_screen_callback,
    )
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, *, severity=None, **_kwargs: notifications.append(
            (message, severity)
        ),
    )
    _configure_full_app_media_startup(app, monkeypatch)

    async with app.run_test() as pilot:
        mounted_screen = await _wait_for_mounted_media_screen(app, pilot)
        assert app.screen is mounted_screen
        notifications.clear()
        warnings: list[str] = []
        sink = app_module.logger.add(
            warnings.append,
            level="WARNING",
            format="{message}",
        )
        try:
            result = await app.handle_runtime_backend_changed(
                "server",
                app_config_override=candidate_config,
            )
        finally:
            app_module.logger.remove(sink)

        assert app.screen is mounted_screen
        assert provider._cached_client is cached_client
        assert provider._cached_client_key is cached_key
        assert cached_client.close_calls == 0

    assert result is False
    assert store.events == ["store_save"]
    assert app.app_config is old_app_config
    assert provider.app_config is old_provider_config
    assert app.runtime_policy.snapshot() == old_context_snapshot
    assert app._runtime_policy_projection_snapshot == old_projection
    assert provider.target_store.list_targets() == old_targets
    assert app.media_runtime_state.runtime_backend == old_media_backend
    assert screen_calls == []
    assert notifications == [
        (
            "Runtime source could not be changed; the previous source remains active.",
            "warning",
        )
    ]
    commit_warnings = [
        warning
        for warning in warnings
        if "Runtime source change was not committed" in warning
    ]
    assert len(commit_warnings) == 1
    assert "exception_category=OSError" in commit_warnings[0]
    assert save_sentinel not in commit_warnings[0]
    assert "candidate.example.test" not in commit_warnings[0]
    assert "candidate-secret" not in commit_warnings[0]


@pytest.mark.asyncio
async def test_full_app_coordinator_cas_rejection_retains_precommit_surfaces(
    full_app_with_controllable_runtime_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store = full_app_with_controllable_runtime_store
    provider = app.server_context_provider
    context = app.runtime_policy
    old_app_config = app.app_config
    old_provider_config = provider.app_config
    old_snapshot = context.snapshot()
    old_projection = app._runtime_policy_projection_snapshot
    old_targets = provider.target_store.list_targets()

    class CachedClientSentinel:
        async def close(self) -> None:
            return None

    cached_client = CachedClientSentinel()
    cached_key = object()
    provider._cached_client = cached_client
    provider._cached_client_key = cached_key
    candidate_config = {
        "tldw_api": {
            "base_url": "https://cas-rejected.example.test/api",
        }
    }

    def reject_commit(
        self,
        candidate: RuntimeSourceState,
        *,
        expected_revision: int,
    ) -> bool:
        assert self is context
        assert candidate.active_server_id == "https://cas-rejected.example.test/api"
        assert expected_revision == old_snapshot[1]
        return False

    monkeypatch.setattr(
        bootstrap_module.RuntimePolicyContext,
        "commit_state",
        reject_commit,
    )
    monkeypatch.setattr(
        provider,
        "rebind_app_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejected CAS must not rebind provider")
        ),
    )
    notifications: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_kwargs: notifications.append(message),
    )

    warnings: list[str] = []
    sink = app_module.logger.add(
        warnings.append,
        level="WARNING",
        format="{message}",
    )
    try:
        result = await app.handle_runtime_backend_changed(
            "server",
            app_config_override=candidate_config,
        )
    finally:
        app_module.logger.remove(sink)

    assert result is False
    assert store.events == []
    assert app.app_config is old_app_config
    assert provider.app_config is old_provider_config
    assert context.snapshot() == old_snapshot
    assert app._runtime_policy_projection_snapshot == old_projection
    assert provider._cached_client is cached_client
    assert provider._cached_client_key is cached_key
    assert provider.target_store.list_targets() == old_targets
    assert notifications == [
        "Runtime source could not be changed; the previous source remains active."
    ]
    commit_warnings = [
        warning
        for warning in warnings
        if "Runtime source change was not committed" in warning
    ]
    assert len(commit_warnings) == 1
    assert "exception_category=RuntimeError" in commit_warnings[0]
    assert "cas-rejected.example.test" not in commit_warnings[0]


@pytest.mark.asyncio
async def test_full_app_coordinator_success_orders_commit_rebind_and_actual_screen(
    full_app_with_controllable_runtime_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store = full_app_with_controllable_runtime_store
    provider = app.server_context_provider
    events = store.events
    candidate_config = {
        "tldw_api": {
            "base_url": "https://ordered.example.test/api/",
            "bearer_token": "ordered-secret",
            "auth_mode": "bearer",
        }
    }
    original_publisher = TldwCli._publish_runtime_policy_projection

    def record_projection(
        app_instance: TldwCli,
        state: RuntimeSourceState,
    ) -> None:
        events.append("projection")
        original_publisher(app_instance, state)

    monkeypatch.setattr(
        TldwCli,
        "_publish_runtime_policy_projection",
        record_projection,
    )
    original_rebind = provider.rebind_app_config

    def record_rebind(
        app_config,
        *,
        previous_server_id,
        next_server_id,
    ) -> None:
        assert app.app_config is candidate_config
        events.append("app_config")
        events.append("provider_rebind")
        original_rebind(
            app_config,
            previous_server_id=previous_server_id,
            next_server_id=next_server_id,
        )

    monkeypatch.setattr(provider, "rebind_app_config", record_rebind)

    async def record_screen_callback(
        screen: MediaScreen,
        runtime_backend: str,
    ) -> None:
        assert app.screen is screen
        assert app.app_config is candidate_config
        assert provider.app_config is candidate_config
        events.append("screen_callback")
        screen.media_runtime_state.reset_for_backend(runtime_backend)

    monkeypatch.setattr(
        MediaScreen,
        "handle_runtime_backend_changed",
        record_screen_callback,
    )
    _configure_full_app_media_startup(app, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for_mounted_media_screen(app, pilot)

        result = await app.handle_runtime_backend_changed(
            "server",
            app_config_override=candidate_config,
        )

    assert result is True
    assert events == [
        "store_save",
        "projection",
        "app_config",
        "provider_rebind",
        "screen_callback",
    ]
    assert app.runtime_policy.state.active_source == "server"
    assert (
        app.runtime_policy.state.active_server_id == "https://ordered.example.test/api"
    )
    assert app.current_runtime_backend == "server"
    assert app.active_server_id == "https://ordered.example.test/api"
    assert app.app_config is candidate_config
    assert provider.app_config is candidate_config
    assert app.media_runtime_state.runtime_backend == "server"


@pytest.mark.asyncio
async def test_full_app_coordinator_contains_actual_screen_callback_failure_after_commit(
    full_app_with_controllable_runtime_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _store = full_app_with_controllable_runtime_store
    candidate_config = {
        "tldw_api": {
            "base_url": "https://screen-failure.example.test/api",
        }
    }
    callback_sentinel = "SCREEN-CALLBACK-PRIVATE-SENTINEL"

    async def raise_from_screen(
        _screen: MediaScreen,
        _runtime_backend: str,
    ) -> None:
        raise RuntimeError(callback_sentinel)

    monkeypatch.setattr(
        MediaScreen,
        "handle_runtime_backend_changed",
        raise_from_screen,
    )
    _configure_full_app_media_startup(app, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for_mounted_media_screen(app, pilot)
        warnings: list[str] = []
        sink = app_module.logger.add(
            warnings.append,
            level="WARNING",
            format="{message}",
        )
        try:
            result = await app.handle_runtime_backend_changed(
                "server",
                app_config_override=candidate_config,
            )
        finally:
            app_module.logger.remove(sink)

    assert result is True
    assert app.runtime_policy.state.active_source == "server"
    assert (
        app.runtime_policy.state.active_server_id
        == "https://screen-failure.example.test/api"
    )
    assert app.app_config is candidate_config
    assert app.server_context_provider.app_config is candidate_config
    callback_warnings = [
        warning for warning in warnings if "screen callback failed" in warning.lower()
    ]
    assert len(callback_warnings) == 1
    assert "exception_category=RuntimeError" in callback_warnings[0]
    assert callback_sentinel not in callback_warnings[0]


@pytest.mark.asyncio
async def test_full_app_coordinator_without_candidate_updates_mounted_media_screen(
    full_app_with_controllable_runtime_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _store = full_app_with_controllable_runtime_store
    provider = app.server_context_provider
    app.media_runtime_state.reset_for_backend("server")
    invalidations: list[tuple[str | None, str | None]] = []
    original_invalidate = provider.invalidate_for_server_switch

    def record_invalidation(
        previous_server_id: str | None,
        next_server_id: str | None,
    ) -> None:
        invalidations.append((previous_server_id, next_server_id))
        original_invalidate(previous_server_id, next_server_id)

    monkeypatch.setattr(
        provider,
        "invalidate_for_server_switch",
        record_invalidation,
    )
    refreshes: list[str] = []

    async def record_refresh(window: MediaWindow, runtime_backend: str) -> None:
        window.runtime_state.reset_for_backend(runtime_backend)
        refreshes.append(window.runtime_state.runtime_backend)

    monkeypatch.setattr(
        MediaWindow,
        "handle_runtime_backend_changed",
        record_refresh,
    )
    _configure_full_app_media_startup(app, monkeypatch)

    async with app.run_test() as pilot:
        mounted_screen = await _wait_for_mounted_media_screen(app, pilot)
        refreshes.clear()

        result = await app.handle_runtime_backend_changed("local")

        assert app.screen is mounted_screen
        assert mounted_screen.media_runtime_state is app.media_runtime_state
        assert (
            mounted_screen.media_window.runtime_state is app.media_runtime_state
        )

    assert result is True
    assert app.current_runtime_backend == "local"
    assert app.runtime_backend == "local"
    assert app.media_runtime_state.runtime_backend == "local"
    assert refreshes == ["local"]
    assert invalidations == [
        (
            app.runtime_policy.state.active_server_id,
            app.runtime_policy.state.active_server_id,
        )
    ]


@pytest.mark.asyncio
async def test_full_app_coordinator_invalid_input_has_no_side_effects(
    full_app_with_controllable_runtime_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store = full_app_with_controllable_runtime_store
    provider = app.server_context_provider
    old_config = app.app_config
    old_provider_config = provider.app_config
    old_snapshot = app.runtime_policy.snapshot()
    old_projection = app._runtime_policy_projection_snapshot

    monkeypatch.setattr(
        provider,
        "rebind_app_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid input must not rebind provider")
        ),
    )
    monkeypatch.setattr(
        provider,
        "invalidate_for_server_switch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid input must not invalidate provider")
        ),
    )

    result = await app.handle_runtime_backend_changed(
        "invalid-source",
        app_config_override={
            "tldw_api": {
                "base_url": "https://unused.example.test/api",
            }
        },
    )

    assert result is False
    assert store.events == []
    assert app.app_config is old_config
    assert provider.app_config is old_provider_config
    assert app.runtime_policy.snapshot() == old_snapshot
    assert app._runtime_policy_projection_snapshot == old_projection
