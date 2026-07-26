from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.runtime_policy.bootstrap import set_authoritative_runtime_source
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
