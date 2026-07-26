from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.runtime_policy.bootstrap import (
    _apply_runtime_policy_to_app,
    load_runtime_policy_for_app,
    set_authoritative_runtime_source,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


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

    updated_state = set_authoritative_runtime_source(app, "server")

    assert updated_state.active_source == "server"
    assert updated_state.active_server_id == "https://new.example.com/v1"
    assert updated_state.server_configured is True
    assert updated_state.server_reachability == "unknown"
    assert updated_state.server_reachability_checked_at is None
    assert updated_state.server_auth_state == "unknown"
    assert updated_state.server_auth_checked_at is None
    assert context.state == updated_state
