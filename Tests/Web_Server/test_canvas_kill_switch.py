"""Served-mode Canvas disabled-startup and live-revocation tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from types import MethodType

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.Web_Server import serve


class _BaseAppService:
    def __init__(self, command: str, **_kwargs) -> None:
        self.command = command
        self.app_service_id = "terminal-only-child"

    def _build_environment(self, width: int = 80, height: int = 24):
        return {"COLUMNS": str(width), "ROWS": str(height)}

    async def start(self, width: int, height: int) -> None:
        self.environment = self._build_environment(width, height)

    async def stop(self) -> None:
        return None


@pytest.mark.asyncio
async def test_disabled_served_child_starts_without_canvas_control_secret() -> None:
    cls = serve.build_chatbook_app_service_class(_BaseAppService)
    service = cls("python -m tldw_chatbook.app", canvas_control_broker=None)

    await service.start(100, 40)

    assert service.environment == {"COLUMNS": "100", "ROWS": "40"}
    await service.stop()


@pytest.mark.asyncio
async def test_live_disable_revokes_served_broker_gateway_and_browser_bindings() -> (
    None
):
    order: list[str] = []

    class Broker:
        async def aclose(self) -> None:
            order.append("broker")

    class Gateway:
        async def aclose(self) -> None:
            order.append("gateway")

    owner = SimpleNamespace(
        _canvas_control_broker=Broker(),
        _served_canvas_gateway=Gateway(),
        _served_canvas_launches={"browser": object()},
        _served_browser_children={"browser": "child"},
        _canvas_disabled_latched=False,
    )

    await serve.ChatbookWebServerMixin._disable_canvas_runtime(owner)

    assert order == ["broker", "gateway"]
    assert owner._canvas_control_broker is None
    assert owner._served_canvas_launches == {}
    assert owner._served_browser_children == {}
    assert owner._canvas_disabled_latched is True


@pytest.mark.asyncio
async def test_policy_watcher_revokes_all_sibling_served_children() -> None:
    enabled = [True]
    order: list[str] = []

    class Broker:
        async def aclose(self) -> None:
            order.append("all-child-controls")

    class Gateway:
        async def aclose(self) -> None:
            order.append("all-browser-previews")

    owner = SimpleNamespace(
        _canvas_control_broker=Broker(),
        _served_canvas_gateway=Gateway(),
        _served_canvas_launches={"browser-a": object(), "browser-b": object()},
        _served_browser_children={"browser-a": "child-a", "browser-b": "child-b"},
        _canvas_disabled_latched=False,
    )
    owner._canvas_enabled = lambda: enabled[0]
    owner._disable_canvas_runtime = MethodType(
        serve.ChatbookWebServerMixin._disable_canvas_runtime, owner
    )

    task = asyncio.create_task(
        serve.ChatbookWebServerMixin._watch_canvas_policy(owner, interval_seconds=0.001)
    )
    enabled[0] = False
    await task

    assert order == ["all-child-controls", "all-browser-previews"]
    assert owner._served_browser_children == {}
    assert owner._served_canvas_launches == {}


def test_create_server_passes_disabled_policy_to_served_owner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Server:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(serve, "_load_textual_serve_server_class", lambda: object)
    monkeypatch.setattr(serve, "_load_textual_serve_app_service_class", lambda: object)
    monkeypatch.setattr(serve, "build_chatbook_web_server_class", lambda *_: Server)
    monkeypatch.setattr(
        serve,
        "get_canvas_config_policy",
        lambda **_kwargs: config_module.build_canvas_config_policy(
            {"canvas": {"enabled": False}}
        ),
        raising=False,
    )

    serve.create_server(host="127.0.0.1", port=8000)

    assert captured["canvas_policy"].enabled is False
