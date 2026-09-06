import inspect
from importlib.metadata import version
from types import SimpleNamespace

import pytest

from tldw_chatbook.Web_Server import serve

pytestmark = pytest.mark.skipif(
    not serve.check_web_server_available(),
    reason="web server optional dependencies are unavailable",
)


def test_pinned_textual_serve_exposes_the_supported_child_environment_seam() -> None:
    from textual_serve.app_service import AppService
    from textual_serve.server import Server

    assert version("textual-serve") == "1.1.3"
    assert str(inspect.signature(AppService._build_environment)) == (
        "(self, width: 'int' = 80, height: 'int' = 24) -> 'dict[str, str]'"
    )
    assert str(inspect.signature(AppService._open_app_process)) == (
        "(self, width: 'int' = 80, height: 'int' = 24) -> 'Process'"
    )
    assert str(inspect.signature(Server.handle_websocket)) == (
        "(self, request: 'web.Request') -> 'web.WebSocketResponse'"
    )


class _BaseAppService:
    def __init__(self, command: str, **_kwargs) -> None:
        self.command = command
        self.app_service_id = "app-service-a"
        self.starts = 0
        self.stops = 0

    def _build_environment(self, width: int = 80, height: int = 24):
        return {"COLUMNS": str(width), "ROWS": str(height)}

    async def start(self, width: int, height: int) -> None:
        self.starts += 1
        self.environment = self._build_environment(width, height)

    async def stop(self) -> None:
        self.stops += 1


class _Broker:
    def __init__(self) -> None:
        self.issued = []
        self.revoked = []

    def issue_child(self, child_id: str):
        self.issued.append(child_id)

        return SimpleNamespace(
            environment={
                "CHATBOOK_CANVAS_CONTROL_HOST": "127.0.0.1",
                "CHATBOOK_CANVAS_CONTROL_PORT": "43210",
                "CHATBOOK_CANVAS_CONTROL_CHILD_ID": child_id,
                "CHATBOOK_CANVAS_CONTROL_SECRET": "private-secret",
                "CHATBOOK_CANVAS_CONTROL_VERSION": "1",
            }
        )

    async def revoke_child(self, child_id: str) -> None:
        self.revoked.append(child_id)


def test_chatbook_app_service_injects_control_data_via_environment_only() -> None:
    async def scenario() -> None:
        broker = _Broker()
        cls = serve.build_chatbook_app_service_class(_BaseAppService)
        service = cls("python -m tldw_chatbook.app", canvas_control_broker=broker)

        await service.start(100, 40)

        assert service.command == "python -m tldw_chatbook.app"
        assert service.environment["COLUMNS"] == "100"
        assert service.environment["CHATBOOK_CANVAS_CONTROL_CHILD_ID"] == (
            "app-service-a"
        )
        assert broker.issued == ["app-service-a"]
        await service.stop()
        assert broker.revoked == ["app-service-a"]

    import asyncio

    asyncio.run(scenario())


def test_textual_child_app_starts_and_closes_its_control_client() -> None:
    async def scenario() -> None:
        from tldw_chatbook import app as app_module

        class _Client:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.closed = False

            async def start(self) -> None:
                self.started.set()

            async def aclose(self) -> None:
                self.closed = True

        client = _Client()

        class _App:
            served_canvas_control = client
            _served_canvas_control_start_task = None
            loguru_logger = app_module.loguru_logger

        app = _App()
        app_module.TldwCli._start_served_canvas_control(app)
        await asyncio.wait_for(client.started.wait(), timeout=1)
        await app_module.TldwCli._stop_served_canvas_control(app)
        assert client.closed is True
        assert app._served_canvas_control_start_task is None

    import asyncio

    asyncio.run(scenario())
