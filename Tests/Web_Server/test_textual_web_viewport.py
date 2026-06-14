import asyncio

import pytest

from tldw_chatbook.Web_Server import serve


pytestmark = pytest.mark.skipif(
    not serve.check_web_server_available(),
    reason="web server optional dependencies are unavailable",
)


class FakeTextualServeServer:
    """Server base that exercises Chatbook overrides without spawning a server."""

    def __init__(
        self,
        command: str,
        host: str,
        port: int,
        title: str,
        *,
        public_url: str | None = None,
        statics_path: str = "/tmp/static",
        templates_path: str = "/tmp/templates",
    ):
        self.command = command
        self.host = host
        self.port = port
        self.title = title
        self.public_url = public_url or f"http://{host}:{port}"
        self.statics_path = statics_path
        self.templates_path = templates_path

    async def handle_websocket(self, request):
        raise NotImplementedError

    async def handle_download(self, request):
        raise NotImplementedError

    async def on_startup(self, app):
        return None

    async def on_shutdown(self, app):
        return None


def _make_test_server(**kwargs):
    server_class = serve.build_chatbook_web_server_class(FakeTextualServeServer)
    return server_class(
        command="python -m tldw_chatbook.app",
        host="127.0.0.1",
        port=0,
        title="test",
        **kwargs,
    )


def test_web_font_size_defaults_to_dense_terminal_cells(monkeypatch):
    monkeypatch.setattr(serve, "get_cli_setting", lambda *_, default=None: default)

    assert serve.resolve_web_font_size(None) == 12


def test_web_font_size_query_overrides_default(monkeypatch):
    monkeypatch.setattr(serve, "get_cli_setting", lambda *_, default=None: default)

    assert serve.resolve_web_font_size("16") == 16


def test_web_font_size_config_fills_missing_or_invalid_query(monkeypatch):
    monkeypatch.setattr(serve, "get_cli_setting", lambda *_, default=None: "14")

    assert serve.resolve_web_font_size(None) == 14
    assert serve.resolve_web_font_size("not-a-size") == 14
    assert serve.resolve_web_font_size("64") == 14
    assert serve.resolve_web_font_size("12.5") == 14


def test_textual_serve_resize_patch_forces_full_terminal_repaint():
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} after"
    )

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert "window.addEventListener(\"resize\",this._chatbookViewportResize)" in patched
    assert "window.onresize=this._chatbookViewportResize" not in patched
    assert "ResizeObserver" in patched
    assert "this.terminal.refresh(0,this.terminal.rows-1)" in patched
    assert "this.terminal.clearTextureAtlas" in patched
    assert "this.sendSize&&this.sendSize()" in patched
    assert "new p.WebglAddon" not in patched
    assert "new m.CanvasAddon" not in patched
    assert "this.webglAddon=null,this.canvasAddon=null," in patched
    assert patched != upstream


def test_textual_serve_resize_patch_repaints_after_connection_and_first_byte():
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} "
        'document.querySelector("body").classList.add("-loaded") '
        't.length>10&&document.querySelector("body").classList.add("-first-byte") after'
    )

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert 'document.querySelector("body").classList.add("-loaded"),this._chatbookViewportResize()' in patched
    assert (
        't.length>10&&(document.querySelector("body").classList.add("-first-byte"),'
        'this._chatbookViewportResize())'
    ) in patched


def test_textual_serve_resize_patch_repaints_after_terminal_writes():
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} "
        "this.terminal.write(t,(()=>{this.bufferedBytes-=t.length})) "
        't.length>10&&document.querySelector("body").classList.add("-first-byte") after'
    )

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert "this._chatbookTerminalRepaint" in patched
    assert "this._chatbookViewportAfterWrite" in patched
    assert (
        "this.terminal.write(t,(()=>{this.bufferedBytes-=t.length,"
        "this._chatbookViewportAfterWrite&&this._chatbookViewportAfterWrite()}))"
    ) in patched


def test_textual_serve_resize_patch_fails_closed_when_upstream_changes():
    upstream = "before window.onresize=()=>{this.fit()} after"

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert patched == upstream


def test_chatbook_web_server_overrides_textual_js_before_static_assets(tmp_path):
    server = _make_test_server(statics_path=str(tmp_path))

    app = asyncio.run(server._make_app())
    resources = list(app.router.resources())
    route_keys = [
        resource.get_info().get("path") or resource.get_info().get("prefix")
        for resource in resources
    ]
    static_resource = resources[route_keys.index("/static")]

    assert "/static/js/textual.js" in route_keys
    assert route_keys.index("/static/js/textual.js") < route_keys.index("/static")
    assert getattr(static_resource, "_show_index", None) is False


def test_chatbook_web_server_uses_urlparse_for_ipv6_websocket_url():
    server = _make_test_server(public_url="https://[::1]:8443")

    assert server._app_websocket_url == "wss://[::1]:8443/ws"


def test_patched_textual_js_is_cached_until_source_changes(tmp_path, monkeypatch):
    js_dir = tmp_path / "js"
    js_dir.mkdir()
    source = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} after"
    )
    (js_dir / "textual.js").write_text(source, encoding="utf-8")
    server = _make_test_server(statics_path=str(tmp_path))
    patch_calls = 0
    original_patch = serve.patch_textual_serve_viewport_js

    def counted_patch(js_source: str) -> str:
        nonlocal patch_calls
        patch_calls += 1
        return original_patch(js_source)

    monkeypatch.setattr(serve, "patch_textual_serve_viewport_js", counted_patch)

    first = server._patched_textual_js()
    second = server._patched_textual_js()

    assert first == second
    assert patch_calls == 1
