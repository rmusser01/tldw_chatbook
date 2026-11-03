"""Optional service reads preserve received-source fences and evaluation order."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("source", ["available", "absent", "error"])
def test_received_prompt_scope_keeps_optional_fallback_and_stale_app_fence(source):
    from tldw_chatbook.Chat.console_received_dispatch import (
        RecoveryRequired,
        received_preparation_source,
        require_received_source,
    )

    scope = object()
    reads = []

    class App:
        @property
        def prompt_scope_service(self):
            reads.append(source)
            if source == "absent":
                raise AttributeError("optional service absent")
            if source == "error":
                raise RuntimeError("source unavailable")
            return scope

    app, store = App(), object()
    controller = SimpleNamespace(app=app, store=store)
    runtime = SimpleNamespace(
        _app=app,
        _chat_store=store,
        _chat_controller=controller,
        _hook_permissions=None,
        _raise_if_disposed_or_session_fenced=lambda _session_id: None,
    )
    if source == "error":
        with pytest.raises(RuntimeError, match="source unavailable"):
            received_preparation_source(runtime)
        assert reads == [source]
        return
    captured = received_preparation_source(runtime)
    assert captured.prompt_scope is (scope if source == "available" else None)
    record = SimpleNamespace(session_id="session", received_claim=None)
    require_received_source(runtime, record, captured, inputs=False)
    assert reads == [source, source]
    runtime._app = object()
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        require_received_source(runtime, record, captured, inputs=False)
    assert reads == [source, source]  # stale app refuses before touching its service


@pytest.mark.parametrize("factory_current", [False, True])
def test_stock_catalog_checks_factory_before_evaluating_app_receiver(
    monkeypatch, factory_current
):
    from tldw_chatbook.Agents import mcp_tool_provider
    from tldw_chatbook.Chat.console_chat_controller import (
        _stock_console_tool_composition_current,
    )

    reads = []

    class Controller:
        @property
        def app(self):
            reads.append("app")
            raise AttributeError("receiver unavailable")

    monkeypatch.setattr(
        mcp_tool_provider, "_controller_factory_current", lambda _: factory_current
    )
    if factory_current:
        with pytest.raises(AttributeError, match="receiver unavailable"):
            _stock_console_tool_composition_current(Controller())
        assert reads == ["app"]
    else:
        assert _stock_console_tool_composition_current(Controller()) is False
        assert reads == []
