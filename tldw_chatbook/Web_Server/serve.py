# serve.py
"""
Web server module for running tldw_chatbook in a browser using textual-serve.

This module provides functions to launch the Textual application as a web server,
allowing users to access the TUI through their web browser.
"""

import sys
from typing import Optional
from pathlib import Path
from loguru import logger

from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE, require_dependency
from ..config import get_cli_setting

try:
    from textual_serve.server import Server as TextualServeServer
except ImportError:  # pragma: no cover - create_server reports the missing optional dep.
    TextualServeServer = object


_TEXTUAL_SERVE_RESIZE_HOOK = "window.onresize=()=>{this.fit()}"
_TEXTUAL_SERVE_CANVAS_RENDERERS = (
    "this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
    "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
)
_CHATBOOK_VIEWPORT_PATCH_MARKER = "this._chatbookViewportResize"
_CHATBOOK_DEFAULT_WEB_FONT_SIZE = 12


def _to_int(value: object, default: int) -> int:
    """Convert a value to int or return the supplied default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_web_font_size(query_value: str | None) -> int:
    """Resolve the Textual Web font size using query, config, then app default."""
    configured_value = get_cli_setting(
        "web_server",
        "font_size",
        default=_CHATBOOK_DEFAULT_WEB_FONT_SIZE,
    )
    configured_font_size = _to_int(configured_value, _CHATBOOK_DEFAULT_WEB_FONT_SIZE)
    if query_value is None:
        return configured_font_size
    return _to_int(query_value, configured_font_size)


def patch_textual_serve_viewport_js(source: str) -> str:
    """Patch textual-serve's browser resize hook to repaint after viewport changes."""
    if _CHATBOOK_VIEWPORT_PATCH_MARKER in source:
        return source
    if (
        _TEXTUAL_SERVE_RESIZE_HOOK not in source
        or _TEXTUAL_SERVE_CANVAS_RENDERERS not in source
    ):
        return source

    patched = source.replace(
        _TEXTUAL_SERVE_CANVAS_RENDERERS,
        "this.webglAddon=null,this.canvasAddon=null,",
        1,
    )

    resize_replacement = (
        "this._chatbookViewportRepaint=()=>{"
        "this.fit();"
        "try{this.terminal.clearTextureAtlas&&this.terminal.clearTextureAtlas()}catch(e){}"
        "try{this.terminal.refresh(0,this.terminal.rows-1)}catch(e){}"
        "};"
        "this._chatbookViewportResize=()=>{"
        "this._chatbookViewportRepaint();"
        "clearTimeout(this._chatbookViewportResizeTimer);"
        "this._chatbookViewportResizeTimer=setTimeout(this._chatbookViewportRepaint,75);"
        "requestAnimationFrame(this._chatbookViewportRepaint)"
        "};"
        "window.addEventListener(\"resize\",this._chatbookViewportResize);"
        "window.onresize=this._chatbookViewportResize;"
        "try{new ResizeObserver(this._chatbookViewportResize).observe(this.element)}catch(e){}"
    )
    return patched.replace(_TEXTUAL_SERVE_RESIZE_HOOK, resize_replacement, 1)


class ChatbookWebServer(TextualServeServer):
    """Textual web server with Chatbook-specific viewport resize hardening."""

    async def _make_app(self):
        """Make the aiohttp app and override only the resize-sensitive JS asset."""
        import aiohttp_jinja2
        import jinja2
        from aiohttp import web

        app = web.Application()
        aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(self.templates_path))

        routes = [
            web.get("/", self.handle_index, name="index"),
            web.get("/ws", self.handle_websocket, name="websocket"),
            web.get("/download/{key}", self.handle_download, name="download"),
            web.get("/static/js/textual.js", self.handle_textual_js, name="textual_js"),
            web.static("/static", self.statics_path, show_index=True, name="static"),
        ]
        app.add_routes(routes)

        app.on_startup.append(self.on_startup)
        app.on_shutdown.append(self.on_shutdown)
        return app

    @property
    def _static_url(self) -> str:
        """Return the public static asset URL with a trailing slash."""
        return f"{self.public_url}/static/"

    @property
    def _app_websocket_url(self) -> str:
        """Return the public websocket URL used by textual-serve's browser client."""
        websocket_url = f"{self.public_url}/ws"
        if websocket_url.startswith("https"):
            return "wss:" + websocket_url.split(":", 1)[1]
        return "ws:" + websocket_url.split(":", 1)[1]

    async def handle_index(self, request):
        """Serve the HTML shell with Chatbook's denser terminal default."""
        import aiohttp_jinja2

        font_size = resolve_web_font_size(request.query.get("fontsize"))
        context = {
            "font_size": font_size,
            "app_websocket_url": self._app_websocket_url,
            "config": {"static": {"url": self._static_url}},
            "application": {"name": self.title},
        }
        return aiohttp_jinja2.render_template("app_index.html", request, context)

    async def handle_textual_js(self, request):
        """Serve textual-serve JS with a full repaint after browser viewport resize."""
        from aiohttp import web

        source_path = Path(self.statics_path) / "js" / "textual.js"
        source = source_path.read_text(encoding="utf-8")
        return web.Response(
            text=patch_textual_serve_viewport_js(source),
            content_type="application/javascript",
        )


def check_web_server_available() -> bool:
    """Check if web server dependencies are available."""
    return DEPENDENCIES_AVAILABLE.get('web', False)


def create_server(
    host: str = "localhost",
    port: int = 8000,
    title: Optional[str] = None,
    debug: bool = False
):
    """
    Create and configure a textual-serve Server instance.
    
    Args:
        host: The host address to bind to (default: localhost)
        port: The port to bind to (default: 8000) 
        title: Title for the web page (default: "tldw chatbook")
        debug: Enable debug mode (default: False)
        
    Returns:
        Configured Server instance
        
    Raises:
        ImportError: If textual-serve is not installed
    """
    # Require the dependency
    textual_serve = require_dependency('textual_serve', 'web')
    
    # Create the command to run the app
    # textual-serve expects a command string, not a list
    command = f"{sys.executable} -m tldw_chatbook.app"
    
    # Configure title
    if title is None:
        title = get_cli_setting("web_server", "title", default="tldw chatbook")
    
    # Create the server
    logger.info(f"Creating web server on {host}:{port}")
    server = ChatbookWebServer(
        command=command,
        host=host,
        port=port,
        title=title
    )
    
    return server


def run_web_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    title: Optional[str] = None,
    debug: Optional[bool] = None
):
    """
    Run the tldw_chatbook application as a web server.
    
    This function starts a web server that serves the Textual application,
    allowing users to access it through their web browser.
    
    Args:
        host: Host address (default: from config or "localhost")
        port: Port number (default: from config or 8000)
        title: Page title (default: from config or "tldw chatbook")
        debug: Enable debug mode (default: from config or False)
        
    Raises:
        ImportError: If textual-serve is not installed
    """
    if not check_web_server_available():
        logger.error("Web server dependencies not available.")
        logger.error("Install with: pip install tldw_chatbook[web]")
        sys.exit(1)
    
    # Load settings from config with defaults
    web_config = get_cli_setting("web_server", default={})
    
    # Use provided values or fall back to config/defaults
    host = host if host is not None else web_config.get("host", "localhost")
    port = port if port is not None else web_config.get("port", 8000)
    title = title if title is not None else web_config.get("title", "tldw chatbook")
    debug = debug if debug is not None else web_config.get("debug", False)
    
    # Create and run the server
    server = create_server(host=host, port=port, title=title, debug=debug)
    
    logger.info(f"Starting web server at http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("Web server stopped by user")
    except Exception as e:
        logger.error(f"Error running web server: {e}")
        raise


def main():
    """Entry point for the tldw-serve command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run tldw_chatbook in a web browser",
        prog="tldw-serve"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title for the web page"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Run the web server with provided arguments
    run_web_server(
        host=args.host,
        port=args.port,
        title=args.title,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
