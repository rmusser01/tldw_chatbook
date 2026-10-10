import asyncio
from importlib.resources import files as _resource_files

import pytest

from tldw_chatbook import config
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
    kwargs.setdefault(
        "canvas_policy",
        config.build_canvas_config_policy({"canvas": {"enabled": False}}),
    )
    server_class = serve.build_chatbook_web_server_class(FakeTextualServeServer)
    return server_class(
        command="python -m tldw_chatbook.app",
        host="127.0.0.1",
        port=0,
        title="test",
        **kwargs,
    )


@pytest.mark.parametrize("settings", [{}, config.DEFAULT_CONFIG_FROM_TOML])
def test_web_font_size_defaults_to_16px_for_missing_and_generated_config(
    monkeypatch, settings
):
    monkeypatch.setattr(
        serve,
        "get_cli_setting",
        lambda section, key, default=None: settings.get(section, {}).get(key, default),
    )

    assert serve.resolve_web_font_size(None) == 16


def test_web_font_size_query_overrides_default(monkeypatch):
    monkeypatch.setattr(serve, "get_cli_setting", lambda *_, default=None: default)

    assert serve.resolve_web_font_size("12") == 12


@pytest.mark.parametrize("configured_size", [12, "14"])
def test_web_font_size_config_fills_missing_or_invalid_query(
    monkeypatch, configured_size
):
    monkeypatch.setattr(
        serve, "get_cli_setting", lambda *_, default=None: configured_size
    )

    assert serve.resolve_web_font_size(None) == int(configured_size)
    assert serve.resolve_web_font_size("not-a-size") == int(configured_size)
    assert serve.resolve_web_font_size("64") == int(configured_size)
    assert serve.resolve_web_font_size("12.5") == int(configured_size)


@pytest.mark.parametrize("configured_size", [None, "huge", 5, 33, 12.5])
def test_web_font_size_invalid_config_uses_16px_fallback(monkeypatch, configured_size):
    monkeypatch.setattr(
        serve, "get_cli_setting", lambda *_, default=None: configured_size
    )

    assert serve.resolve_web_font_size(None) == 16
    assert serve.resolve_web_font_size("not-a-size") == 16
    assert serve.resolve_web_font_size("12") == 12


def test_textual_serve_resize_patch_forces_full_terminal_repaint():
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} "
        'document.querySelector("body").classList.add("-loaded") '
        't.length>10&&document.querySelector("body").classList.add("-first-byte") '
        "this.terminal.write(t,(()=>{this.bufferedBytes-=t.length})) after"
    )

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert 'window.addEventListener("resize",this._chatbookViewportResize)' in patched
    assert "window.onresize=this._chatbookViewportResize" not in patched
    assert "ResizeObserver" in patched
    assert "this.terminal.refresh(0,this.terminal.rows-1)" in patched
    assert "this.terminal.clearTextureAtlas" in patched
    assert patched != upstream


def test_textual_serve_resize_patch_keeps_upstream_gpu_renderers():
    """task-33130: nulling the addons forced xterm's DOM renderer, which
    turned every forced repaint into a full-screen DOM rebuild (~44k added
    nodes measured for one click). Upstream's WebGL/Canvas renderers stay."""
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} "
        'document.querySelector("body").classList.add("-loaded") '
        't.length>10&&document.querySelector("body").classList.add("-first-byte") '
        "this.terminal.write(t,(()=>{this.bufferedBytes-=t.length})) after"
    )

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert "new p.WebglAddon" in patched
    assert "new m.CanvasAddon" in patched
    assert "this.webglAddon=null" not in patched
    assert "this.canvasAddon=null" not in patched


def test_textual_serve_resize_patch_gates_send_size_on_grid_changes():
    """task-33130: an unconditional sendSize() inside every repaint fed a
    self-sustaining loop -- every app output frame triggered a repaint,
    every repaint sent a resize, and the app answered every resize with
    fresh output (~54 resizes/second at idle, both processes pegged). The
    resize is now sent only when fit() actually changed the grid."""
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} "
        'document.querySelector("body").classList.add("-loaded") '
        't.length>10&&document.querySelector("body").classList.add("-first-byte") '
        "this.terminal.write(t,(()=>{this.bufferedBytes-=t.length})) after"
    )

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert "this.sendSize&&this.sendSize()" in patched
    assert (
        "const t=this.terminal.cols,r=this.terminal.rows;"
        "this.fit();"
        "if(this.terminal.cols!==t||this.terminal.rows!==r){"
        "try{this.sendSize&&this.sendSize()}catch(e){}}"
    ) in patched


def test_textual_serve_resize_patch_repaints_after_connection_and_first_byte():
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} "
        "this.terminal.write(t,(()=>{this.bufferedBytes-=t.length})) "
        'document.querySelector("body").classList.add("-loaded") '
        't.length>10&&document.querySelector("body").classList.add("-first-byte") after'
    )

    patched = serve.patch_textual_serve_viewport_js(upstream)

    # The loaded hook fires once per connection (websocket "open"), so the
    # resize handler may run there directly.
    assert (
        'document.querySelector("body").classList.add("-loaded"),this._chatbookViewportResize()'
        in patched
    )
    # The first-byte hook fires for EVERY output frame, so it must go
    # through a once-per-connection guard instead of the resize handler.
    assert (
        't.length>10&&(document.querySelector("body").classList.add("-first-byte"),'
        "this._chatbookViewportFirstByte())"
    ) in patched
    assert "this._chatbookViewportFirstByteDone=!0" in patched
    assert 'add("-first-byte"),this._chatbookViewportResize()' not in patched


def test_textual_serve_resize_patch_repaints_after_terminal_writes():
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} "
        'document.querySelector("body").classList.add("-loaded") '
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
    # task-33130: the after-write repaint is a trailing debounce -- a
    # per-write requestAnimationFrame full refresh rebuilt every row once
    # per output frame during streams.
    assert (
        "this._chatbookViewportAfterWriteTimer=setTimeout("
        "this._chatbookTerminalRepaint,250)"
    ) in patched
    assert "cancelAnimationFrame(this._chatbookViewportAfterWriteRaf)" not in patched
    assert "requestAnimationFrame(this._chatbookTerminalRepaint)" not in patched


def test_textual_serve_resize_patch_fails_closed_when_upstream_changes():
    upstream = "before window.onresize=()=>{this.fit()} after"

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert patched == upstream


@pytest.mark.parametrize(
    "missing_hook",
    [
        serve._TEXTUAL_SERVE_RESIZE_HOOK,
        serve._TEXTUAL_SERVE_CANVAS_RENDERERS,
        serve._TEXTUAL_SERVE_WRITE_CALLBACK_HOOK,
        serve._TEXTUAL_SERVE_LOADED_HOOK,
        serve._TEXTUAL_SERVE_FIRST_BYTE_HOOK,
    ],
)
def test_textual_serve_resize_patch_fails_closed_when_any_required_hook_is_missing(
    missing_hook,
):
    upstream = (
        "before "
        f"{serve._TEXTUAL_SERVE_CANVAS_RENDERERS}"
        f"{serve._TEXTUAL_SERVE_RESIZE_HOOK} "
        f"{serve._TEXTUAL_SERVE_WRITE_CALLBACK_HOOK} "
        f"{serve._TEXTUAL_SERVE_LOADED_HOOK} "
        f"{serve._TEXTUAL_SERVE_FIRST_BYTE_HOOK} "
        "after"
    )
    changed_upstream = upstream.replace(missing_hook, "upstream changed", 1)

    patched = serve.patch_textual_serve_viewport_js(changed_upstream)

    assert patched == changed_upstream


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


def _served_shell_js() -> str:
    return (
        _resource_files("tldw_chatbook.Web_Server")
        .joinpath("static")
        .joinpath("served_shell.js")
        .read_text(encoding="utf-8")
    )


def _served_shell_html() -> str:
    return (
        _resource_files("tldw_chatbook.Web_Server")
        .joinpath("static")
        .joinpath("served_shell.html")
        .read_text(encoding="utf-8")
    )


def test_served_shell_derives_same_origin_terminal_websocket_url():
    """task-33130: the server-substituted absolute websocket URL (from
    public_url, default localhost) crossed the session-cookie boundary
    whenever the page was opened via 127.0.0.1, a LAN IP, or another name,
    and the terminal died at the handshake. The shell now derives the URL
    from the page's own location; the substituted attribute remains as a
    fallback when this script fails to load."""
    shell = _served_shell_js()

    assert (
        '(location.protocol === "https:" ? "wss://" : "ws://")'
        ' + location.host + "/ws"' in shell
    )
    assert 'data-session-websocket-url="__APP_WEBSOCKET_URL__"' in _served_shell_html()


def test_served_shell_stops_canvas_session_poll_after_disable():
    """task-33130: a 404 from /canvas/api/session is a server-side kill
    switch that cannot recover without a restart, so the 1 Hz poll must
    stop instead of firing forever."""
    shell = _served_shell_js()
    disable_body = shell[
        shell.index("function disableCanvas") : shell.index("function applyState")
    ]

    assert "clearInterval" in disable_body


def test_textual_js_response_uses_public_cache_policy(tmp_path):
    """The handler only needs statics_path; building it without the full
    server constructor keeps this test independent of the ADR-126 config
    admission fence that gates server construction in some environments."""
    js_dir = tmp_path / "js"
    js_dir.mkdir()
    source = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} after"
    )
    (js_dir / "textual.js").write_text(source, encoding="utf-8")
    server = serve.ChatbookWebServerMixin.__new__(serve.ChatbookWebServerMixin)
    server.statics_path = str(tmp_path)

    response = asyncio.run(server.handle_textual_js(request=None))

    assert response.headers["Cache-Control"] == "public, max-age=3600"


def test_served_shell_versions_the_cached_textual_bundle():
    """task-33130 (Qodo #6): the bundle URL is fingerprinted so the one-hour
    cache cannot pin clients to a superseded patch after an upgrade."""
    shell = _served_shell_html()

    assert 'src="/static/js/textual.js?v=__TEXTUAL_JS_VERSION__"' in shell


def test_handle_index_substitutes_bundle_fingerprint(tmp_path):
    js_dir = tmp_path / "js"
    js_dir.mkdir()
    source = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} after"
    )
    (js_dir / "textual.js").write_text(source, encoding="utf-8")
    server = serve.ChatbookWebServerMixin.__new__(serve.ChatbookWebServerMixin)
    server.statics_path = str(tmp_path)
    server.title = "test"
    server.public_url = "http://127.0.0.1"
    # handle_index reads the font-size setting from the shared config cache;
    # under pytest the ADR-126 admission fence makes that read raise, so
    # serve the default (same mitigation as the browser integration suite).
    original_get_cli_setting = serve.get_cli_setting
    serve.get_cli_setting = lambda *_, default=None: default

    class _Request:
        def __init__(self):
            self.query = {}

        def __getitem__(self, key):
            return {"chatbook_csrf": "token"}[key]

    request = _Request()

    import asyncio as _asyncio

    response = _asyncio.run(server.handle_index(request))
    import re as _re

    match = _re.search(r'textual\.js\?v=([0-9a-f]{16})"', response.text)
    assert match, response.text[:200]
    # The fingerprint follows content: same bundle, same version.
    try:
        again = _asyncio.run(server.handle_index(request))
    finally:
        serve.get_cli_setting = original_get_cli_setting
    assert _re.search(r"textual\.js\?v=" + match.group(1) + '"', again.text)
