from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.state.app_state import AppState


def _make_app_like(*, base_url: str = "https://Example.COM:8443/api/") -> SimpleNamespace:
    return SimpleNamespace(
        app_config={
            "tldw_api": {
                "base_url": base_url,
            }
        },
        app_state=AppState(),
    )


def test_app_state_round_trips_runtime_source_state():
    original = AppState(
        runtime_source=RuntimeSourceState(
            active_source="server",
            active_server_id="server-alpha",
            server_configured=True,
            server_reachability="reachable",
            server_reachability_checked_at=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
            server_auth_state="authenticated",
            server_auth_checked_at=datetime(2026, 4, 21, 12, 5, tzinfo=timezone.utc),
            last_known_server_label="Primary Server",
        )
    )

    restored = AppState.from_dict(original.to_dict())

    assert restored.runtime_source == original.runtime_source


def test_app_state_from_dict_ignores_malformed_runtime_source_payload():
    restored = AppState.from_dict(
        {
            "runtime_source": ["not", "a", "mapping"],
        }
    )

    assert restored.runtime_source == RuntimeSourceState()


def test_runtime_source_state_store_round_trips_json(tmp_path):
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    state = RuntimeSourceState(
        active_source="server",
        active_server_id="server-alpha",
        server_configured=True,
        server_reachability="reachable",
        server_auth_state="authenticated",
    )

    store.save(state)
    restored = store.load()

    assert restored == state


def test_runtime_source_state_store_loads_safe_default_on_malformed_json(tmp_path):
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    path = tmp_path / "runtime_policy.json"
    path.write_text("{not-json", encoding="utf-8")

    restored = RuntimeSourceStateStore(path).load()

    assert restored == RuntimeSourceState()


def test_build_runtime_api_client_uses_api_key_auth_from_config():
    from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client

    client = build_runtime_api_client(
        app_config={
            "tldw_api": {
                "base_url": "https://example.com/api/",
                "api_key": "secret-key",
            }
        }
    )

    assert client.base_url == "https://example.com/api"
    assert client.token == "secret-key"
    assert client.bearer_token is None


def test_build_runtime_api_client_supports_explicit_custom_token_overrides():
    from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client

    client = build_runtime_api_client(
        app_config={"tldw_api": {"base_url": "https://example.com/api/"}},
        endpoint_url="https://override.example.com/v1/",
        auth_method="custom_token",
        auth_token="bearer-secret",
    )

    assert client.base_url == "https://override.example.com/v1"
    assert client.token is None
    assert client.bearer_token == "bearer-secret"


def test_build_server_chatbook_service_wraps_authoritative_client_builder():
    from tldw_chatbook.runtime_policy.bootstrap import build_server_chatbook_service

    service = build_server_chatbook_service(
        app_config={
            "tldw_api": {
                "base_url": "https://example.com/api/",
                "api_key": "secret-key",
            }
        }
    )

    assert service.client is not None
    assert service.client.base_url == "https://example.com/api"
    assert service.client.token == "secret-key"


@pytest.mark.parametrize(
    "import_error",
    [
        ImportError("no backend client"),
        ModuleNotFoundError("no backend client"),
        AttributeError("lazy import failed"),
        TypeError("python compatibility failure"),
    ],
)
def test_runtime_api_client_class_falls_back_for_compatibility_import_failures(import_error):
    from tldw_chatbook.runtime_policy import bootstrap

    def failing_import(_module_name: str):
        raise import_error

    resolved = bootstrap._runtime_api_client_class(module_importer=failing_import)

    assert resolved is bootstrap._CompatibilityRuntimeAPIClient


def test_runtime_api_client_class_falls_back_when_symbol_is_missing():
    from tldw_chatbook.runtime_policy import bootstrap

    resolved = bootstrap._runtime_api_client_class(module_importer=lambda _module_name: SimpleNamespace())

    assert resolved is bootstrap._CompatibilityRuntimeAPIClient


@pytest.mark.parametrize(
    "import_error",
    [
        ImportError("no service"),
        ModuleNotFoundError("no service"),
        AttributeError("lazy import failed"),
        TypeError("python compatibility failure"),
    ],
)
def test_server_chatbook_service_class_falls_back_for_compatibility_import_failures(import_error):
    from tldw_chatbook.runtime_policy import bootstrap

    def failing_import(_module_name: str):
        raise import_error

    resolved = bootstrap._server_chatbook_service_class(module_importer=failing_import)

    assert resolved is bootstrap._CompatibilityServerChatbookService


def test_server_chatbook_service_class_falls_back_when_symbol_is_missing():
    from tldw_chatbook.runtime_policy import bootstrap

    resolved = bootstrap._server_chatbook_service_class(module_importer=lambda _module_name: SimpleNamespace())

    assert resolved is bootstrap._CompatibilityServerChatbookService


def test_load_runtime_policy_for_app_derives_and_persists_authoritative_server_binding_from_app_config(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = _make_app_like()

    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "local"
    assert context.state.active_server_id == "https://example.com:8443/api"
    assert context.state.server_configured is True
    assert context.state.last_known_server_label == "example.com:8443"
    assert store.load() == context.state
    assert app_like.current_runtime_backend == "local"
    assert app_like.runtime_backend == "local"


def test_load_runtime_policy_for_app_rebinds_persisted_runtime_state_to_configured_server_identity(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    store.save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://old.example.com/api",
            server_configured=True,
            last_known_server_label="old.example.com",
        )
    )
    app_like = _make_app_like(base_url="https://new.example.com/v1/")

    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "server"
    assert context.state.active_server_id == "https://new.example.com/v1"
    assert context.state.server_configured is True
    assert context.state.last_known_server_label == "new.example.com"
    assert store.load() == context.state


def test_load_runtime_policy_for_app_downgrades_stale_server_mode_when_server_config_is_missing(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    store.save(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
            last_known_server_label="server.example.com",
        )
    )
    app_like = SimpleNamespace(
        app_config={},
        app_state=AppState(),
    )

    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "local"
    assert context.state.active_server_id is None
    assert context.state.server_configured is False
    assert store.load() == context.state


def test_reconcile_saved_screen_state_drops_wrong_server_snapshot_against_bootstrapped_authority(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import (
        load_runtime_policy_for_app,
        reconcile_saved_screen_state,
    )
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = _make_app_like(base_url="https://server-b.example.com/api/")
    context = load_runtime_policy_for_app(app_like, store=store)
    context.state = RuntimeSourceState(
        active_source="server",
        active_server_id=context.state.active_server_id,
        server_configured=context.state.server_configured,
        last_known_server_label=context.state.last_known_server_label,
    )
    saved_state = {
        "chat_state": {
            "tabs": [
                {
                    "tab_id": "tab-1",
                    "title": "Server Chat",
                    "runtime_backend": "server",
                }
            ]
        },
        "runtime_policy_snapshot": {
            "active_source": "server",
            "active_server_id": "https://server-a.example.com/api",
        },
    }

    assert reconcile_saved_screen_state(saved_state, context.state) is None


def test_runtime_policy_snapshot_remains_authoritative_when_destination_local_mcp_state_differs():
    from tldw_chatbook.runtime_policy.bootstrap import add_runtime_policy_snapshot
    from tldw_chatbook.runtime_policy.source_state import runtime_source_state_with_override

    authoritative_state = RuntimeSourceState(
        active_source="local",
        active_server_id="https://server-a.example.com/api",
        server_configured=True,
        last_known_server_label="server-a.example.com",
    )
    destination_state = runtime_source_state_with_override(
        authoritative_state,
        active_source="server",
        active_server_id="https://server-b.example.com/api",
    )

    restored = add_runtime_policy_snapshot({"panel": "unified-mcp"}, authoritative_state)

    assert restored["runtime_policy_snapshot"] == {
        "active_source": "local",
        "active_server_id": "https://server-a.example.com/api",
    }
    assert destination_state.active_source == "server"
    assert destination_state.active_server_id == "https://server-b.example.com/api"
    assert authoritative_state.active_source == "local"
    assert authoritative_state.active_server_id == "https://server-a.example.com/api"


@pytest.mark.asyncio
async def test_handle_runtime_backend_changed_updates_authoritative_runtime_policy_and_persists(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import (
        load_runtime_policy_for_app,
        set_authoritative_runtime_source,
    )
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    forwarded = []

    async def screen_callback(runtime_backend: str) -> None:
        forwarded.append(runtime_backend)

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = _make_app_like(base_url="https://Server.EXAMPLE.com/api/")
    app_like.screen = SimpleNamespace(handle_runtime_backend_changed=screen_callback)
    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "local"
    assert context.state.active_server_id == "https://server.example.com/api"

    await screen_callback(set_authoritative_runtime_source(app_like, "server").active_source)
    await screen_callback(set_authoritative_runtime_source(app_like, "local").active_source)

    persisted = store.load()

    assert app_like.current_runtime_backend == "local"
    assert app_like.runtime_backend == "local"
    assert app_like.runtime_policy.state.active_source == "local"
    assert persisted.active_source == "local"
    assert persisted.active_server_id == "https://server.example.com/api"
    assert persisted.server_configured is True
    assert persisted.last_known_server_label == "server.example.com"
    assert forwarded == ["server", "local"]


@pytest.mark.asyncio
async def test_handle_runtime_backend_changed_forwards_resolved_authoritative_backend_to_screen(
    tmp_path,
):
    from tldw_chatbook.runtime_policy.bootstrap import (
        load_runtime_policy_for_app,
        set_authoritative_runtime_source,
    )
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    forwarded = []

    async def screen_callback(runtime_backend: str) -> None:
        forwarded.append(runtime_backend)

    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    app_like = SimpleNamespace(
        app_config={},
        app_state=AppState(),
        screen=SimpleNamespace(handle_runtime_backend_changed=screen_callback),
    )
    context = load_runtime_policy_for_app(app_like, store=store)

    assert context.state.active_source == "local"

    await screen_callback(set_authoritative_runtime_source(app_like, "server").active_source)

    assert app_like.runtime_policy.state.active_source == "local"
    assert forwarded == ["local"]
