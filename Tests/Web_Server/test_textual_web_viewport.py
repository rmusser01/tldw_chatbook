import asyncio

from tldw_chatbook.Web_Server import serve


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


def test_textual_serve_resize_patch_forces_full_terminal_repaint():
    upstream = (
        "before this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
        "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
        "window.onresize=()=>{this.fit()} after"
    )

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert "window.addEventListener(\"resize\",this._chatbookViewportResize)" in patched
    assert "ResizeObserver" in patched
    assert "this.terminal.refresh(0,this.terminal.rows-1)" in patched
    assert "this.terminal.clearTextureAtlas" in patched
    assert "new p.WebglAddon" not in patched
    assert "new m.CanvasAddon" not in patched
    assert patched != upstream


def test_textual_serve_resize_patch_fails_closed_when_upstream_changes():
    upstream = "before window.onresize=()=>{this.fit()} after"

    patched = serve.patch_textual_serve_viewport_js(upstream)

    assert patched == upstream


def test_chatbook_web_server_overrides_textual_js_before_static_assets():
    server = serve.ChatbookWebServer(
        command="python -m tldw_chatbook.app",
        host="127.0.0.1",
        port=0,
        title="test",
    )

    app = asyncio.run(server._make_app())
    route_keys = [
        resource.get_info().get("path") or resource.get_info().get("prefix")
        for resource in app.router.resources()
    ]

    assert "/static/js/textual.js" in route_keys
    assert route_keys.index("/static/js/textual.js") < route_keys.index("/static")
