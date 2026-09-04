# serve.py
"""
Web server module for running tldw_chatbook in a browser using textual-serve.

This module provides functions to launch the Textual application as a web server,
allowing users to access the TUI through their web browser.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

from loguru import logger

from ..Canvas.control_protocol import CanvasControlBroker
from ..config import get_cli_setting
from ..Utils.input_validation import validate_number_range
from ..Utils.optional_deps import (
    DEPENDENCIES_AVAILABLE,
    check_web_server_deps,
    require_dependency,
)

_TEXTUAL_SERVE_RESIZE_HOOK = "window.onresize=()=>{this.fit()}"
_TEXTUAL_SERVE_CANVAS_RENDERERS = (
    "this.webglAddon=new p.WebglAddon,this.terminal.loadAddon(this.webglAddon),"
    "this.canvasAddon=new m.CanvasAddon,this.terminal.loadAddon(this.canvasAddon),"
)
_TEXTUAL_SERVE_LOADED_HOOK = 'document.querySelector("body").classList.add("-loaded")'
_TEXTUAL_SERVE_FIRST_BYTE_HOOK = (
    't.length>10&&document.querySelector("body").classList.add("-first-byte")'
)
_TEXTUAL_SERVE_WRITE_CALLBACK_HOOK = (
    "this.terminal.write(t,(()=>{this.bufferedBytes-=t.length}))"
)
_TEXTUAL_SERVE_REQUIRED_VIEWPORT_HOOKS = (
    _TEXTUAL_SERVE_RESIZE_HOOK,
    _TEXTUAL_SERVE_CANVAS_RENDERERS,
    _TEXTUAL_SERVE_WRITE_CALLBACK_HOOK,
    _TEXTUAL_SERVE_LOADED_HOOK,
    _TEXTUAL_SERVE_FIRST_BYTE_HOOK,
)
_CHATBOOK_VIEWPORT_PATCH_MARKER = "this._chatbookViewportResize"
_CHATBOOK_DEFAULT_WEB_FONT_SIZE = 12
_CHATBOOK_MIN_WEB_FONT_SIZE = 6
_CHATBOOK_MAX_WEB_FONT_SIZE = 32


def _coerce_web_font_size(value: object, default: int) -> int:
    """Validate and coerce a configured or requested web terminal font size."""
    if not validate_number_range(
        value,
        min_val=_CHATBOOK_MIN_WEB_FONT_SIZE,
        max_val=_CHATBOOK_MAX_WEB_FONT_SIZE,
    ):
        return default

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default

    if not numeric_value.is_integer():
        return default
    return int(numeric_value)


def resolve_web_font_size(query_value: str | None) -> int:
    """Resolve the Textual Web font size using query, config, then app default."""
    configured_value = get_cli_setting(
        "web_server",
        "font_size",
        default=_CHATBOOK_DEFAULT_WEB_FONT_SIZE,
    )
    configured_font_size = _coerce_web_font_size(
        configured_value,
        _CHATBOOK_DEFAULT_WEB_FONT_SIZE,
    )
    if query_value is None:
        return configured_font_size
    return _coerce_web_font_size(query_value, configured_font_size)


def patch_textual_serve_viewport_js(source: str) -> str:
    """Patch textual-serve's browser resize hook to repaint after viewport changes."""
    if _CHATBOOK_VIEWPORT_PATCH_MARKER in source:
        return source
    if any(hook not in source for hook in _TEXTUAL_SERVE_REQUIRED_VIEWPORT_HOOKS):
        return source

    patched = source.replace(
        _TEXTUAL_SERVE_CANVAS_RENDERERS,
        "this.webglAddon=null,this.canvasAddon=null,",
        1,
    )

    resize_replacement = (
        "this._chatbookTerminalRepaint=()=>{"
        "try{this.terminal.clearTextureAtlas&&this.terminal.clearTextureAtlas()}catch(e){}"
        "try{this.terminal.refresh(0,this.terminal.rows-1)}catch(e){}"
        "};"
        "this._chatbookViewportRepaint=()=>{"
        "this.fit();"
        "try{this.sendSize&&this.sendSize()}catch(e){}"
        "this._chatbookTerminalRepaint();"
        "};"
        "this._chatbookViewportAfterWrite=()=>{"
        "clearTimeout(this._chatbookViewportAfterWriteTimer);"
        "this._chatbookViewportAfterWriteTimer=setTimeout(this._chatbookTerminalRepaint,50);"
        "cancelAnimationFrame(this._chatbookViewportAfterWriteRaf);"
        "this._chatbookViewportAfterWriteRaf=requestAnimationFrame("
        "this._chatbookTerminalRepaint);"
        "};"
        "this._chatbookViewportResize=()=>{"
        "this._chatbookViewportRepaint();"
        "clearTimeout(this._chatbookViewportResizeTimer);"
        "this._chatbookViewportResizeTimer=setTimeout(this._chatbookViewportRepaint,75);"
        "requestAnimationFrame(this._chatbookViewportRepaint)"
        "};"
        'window.addEventListener("resize",this._chatbookViewportResize);'
        "try{new ResizeObserver(this._chatbookViewportResize).observe(this.element)}catch(e){}"
    )
    patched = patched.replace(_TEXTUAL_SERVE_RESIZE_HOOK, resize_replacement, 1)
    patched = patched.replace(
        _TEXTUAL_SERVE_WRITE_CALLBACK_HOOK,
        (
            "this.terminal.write(t,(()=>{this.bufferedBytes-=t.length,"
            "this._chatbookViewportAfterWrite&&this._chatbookViewportAfterWrite()}))"
        ),
        1,
    )
    patched = patched.replace(
        _TEXTUAL_SERVE_LOADED_HOOK,
        f"{_TEXTUAL_SERVE_LOADED_HOOK},this._chatbookViewportResize()",
        1,
    )
    return patched.replace(
        _TEXTUAL_SERVE_FIRST_BYTE_HOOK,
        (
            f"t.length>10&&({_TEXTUAL_SERVE_LOADED_HOOK.replace('-loaded', '-first-byte')},"
            "this._chatbookViewportResize())"
        ),
        1,
    )


class ChatbookWebServerMixin:
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
            web.static("/static", self.statics_path, show_index=False, name="static"),
        ]
        app.add_routes(routes)

        app.on_startup.append(self.on_startup)
        app.on_shutdown.append(self.on_shutdown)
        return app

    async def on_startup(self, app) -> None:
        """Start the private loopback control broker before children spawn."""

        self._canvas_control_broker = CanvasControlBroker()
        await self._canvas_control_broker.start()
        await super().on_startup(app)

    async def on_shutdown(self, app) -> None:
        """Revoke all child control capabilities with the served process."""

        broker = getattr(self, "_canvas_control_broker", None)
        if broker is not None:
            await broker.aclose()
            self._canvas_control_broker = None
        await super().on_shutdown(app)

    async def handle_websocket(self, request):
        """Bind one browser websocket to exactly one authenticated child."""

        from aiohttp import web

        websocket = web.WebSocketResponse(heartbeat=15)
        width = _web_dimension(request.query.get("width"), 80)
        height = _web_dimension(request.query.get("height"), 24)
        app_service = None
        try:
            await websocket.prepare(request)
            app_service_class = getattr(self, "_chatbook_app_service_class", None)
            broker = getattr(self, "_canvas_control_broker", None)
            if app_service_class is None or broker is None:
                raise RuntimeError("served Canvas control broker is unavailable")
            app_service = app_service_class(
                self.command,
                write_bytes=websocket.send_bytes,
                write_str=websocket.send_str,
                close=websocket.close,
                download_manager=self.download_manager,
                debug=self.debug,
                canvas_control_broker=broker,
            )
            await app_service.start(width, height)
            try:
                await self._process_messages(websocket, app_service)
            finally:
                await app_service.stop()
        except asyncio.CancelledError:
            await websocket.close()
        except Exception as error:  # noqa: BLE001 - websocket boundary stays alive
            # No traceback here: the spawn frame holds the per-child secret.
            logger.error(
                "Served terminal session failed type={}", type(error).__name__
            )
        finally:
            if app_service is not None:
                await app_service.stop()
        return websocket

    @property
    def _static_url(self) -> str:
        """Return the public static asset URL with a trailing slash."""
        return f"{self.public_url.rstrip('/')}/static/"

    @property
    def _app_websocket_url(self) -> str:
        """Return the public websocket URL used by textual-serve's browser client."""
        parsed_url = urlparse(f"{self.public_url.rstrip('/')}/ws")
        websocket_scheme = "wss" if parsed_url.scheme == "https" else "ws"
        return urlunparse(parsed_url._replace(scheme=websocket_scheme))

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

    def _patched_textual_js(self) -> str:
        """Return cached patched textual-serve JS, refreshing when the file changes."""
        source_path = Path(self.statics_path) / "js" / "textual.js"
        source_stat = source_path.stat()
        cached_mtime = getattr(self, "_cached_textual_js_mtime_ns", None)
        cached_text = getattr(self, "_cached_textual_js", None)

        if cached_text is not None and cached_mtime == source_stat.st_mtime_ns:
            return cached_text

        source = source_path.read_text(encoding="utf-8")
        patched = patch_textual_serve_viewport_js(source)
        self._cached_textual_js = patched
        self._cached_textual_js_mtime_ns = source_stat.st_mtime_ns
        return patched

    async def handle_textual_js(self, request):
        """Serve textual-serve JS with a full repaint after browser viewport resize."""
        from aiohttp import web

        return web.Response(
            text=self._patched_textual_js(),
            content_type="application/javascript",
        )


def _load_textual_serve_server_class() -> type:
    """Load textual-serve's Server class after the optional dependency gate."""
    require_dependency("textual_serve", "web")
    from textual_serve.server import Server as TextualServeServer

    return TextualServeServer


def _load_textual_serve_app_service_class() -> type:
    """Load textual-serve's supported child process service."""

    require_dependency("textual_serve", "web")
    from textual_serve.app_service import AppService as TextualServeAppService

    return TextualServeAppService


def _web_dimension(value: object, default: int) -> int:
    """Match textual-serve's forgiving positive terminal-size parsing."""

    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def build_chatbook_app_service_class(textual_app_service_class: type) -> type:
    """Extend only textual-serve's documented child-environment build seam."""

    class ChatbookAppService(textual_app_service_class):
        def __init__(self, *args, canvas_control_broker, **kwargs):
            super().__init__(*args, **kwargs)
            self._canvas_control_broker = canvas_control_broker
            self._canvas_control_environment: dict[str, str] = {}
            self._canvas_control_started = False

        def _build_environment(self, width: int = 80, height: int = 24):
            environment = super()._build_environment(width, height)
            environment.update(self._canvas_control_environment)
            return environment

        async def start(self, width: int, height: int) -> None:
            launch = self._canvas_control_broker.issue_child(self.app_service_id)
            self._canvas_control_environment = dict(launch.environment)
            try:
                await super().start(width, height)
            except BaseException:
                await self._canvas_control_broker.revoke_child(self.app_service_id)
                self._canvas_control_environment.clear()
                raise
            self._canvas_control_started = True

        async def stop(self) -> None:
            try:
                await super().stop()
            finally:
                if self._canvas_control_started:
                    self._canvas_control_started = False
                    await self._canvas_control_broker.revoke_child(
                        self.app_service_id
                    )
                self._canvas_control_environment.clear()

    ChatbookAppService.__name__ = "ChatbookAppService"
    return ChatbookAppService


def build_chatbook_web_server_class(
    textual_serve_server_class: type,
    textual_app_service_class: type | None = None,
) -> type:
    """Build the Chatbook server subclass from a provided textual-serve base."""

    class ChatbookWebServer(ChatbookWebServerMixin, textual_serve_server_class):
        _chatbook_app_service_class = (
            build_chatbook_app_service_class(textual_app_service_class)
            if textual_app_service_class is not None
            else None
        )

    ChatbookWebServer.__name__ = "ChatbookWebServer"
    return ChatbookWebServer


def check_web_server_available() -> bool:
    """Check if web server dependencies are available.

    Returns:
        True when textual-web dependencies are available, otherwise False.
    """
    if DEPENDENCIES_AVAILABLE.get("web", False):
        return True
    try:
        return check_web_server_deps()
    except Exception as exc:
        logger.warning(
            f"Web server dependency probe failed. Web mode is unavailable. Reason: {exc}"
        )
        DEPENDENCIES_AVAILABLE["web"] = False
        DEPENDENCIES_AVAILABLE["textual_serve"] = False
        return False


def create_server(
    host: str = "localhost",
    port: int = 8000,
    title: Optional[str] = None,
    debug: bool = False,
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
    textual_serve_server_class = _load_textual_serve_server_class()
    textual_app_service_class = _load_textual_serve_app_service_class()
    chatbook_web_server_class = build_chatbook_web_server_class(
        textual_serve_server_class,
        textual_app_service_class,
    )

    # Create the command to run the app
    # textual-serve expects a command string, not a list
    command = f"{sys.executable} -m tldw_chatbook.app"

    # Configure title
    if title is None:
        title = get_cli_setting("web_server", "title", default="tldw chatbook")

    # Create the server
    logger.info(f"Creating web server on {host}:{port}")
    server = chatbook_web_server_class(
        command=command, host=host, port=port, title=title
    )

    return server


def run_web_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    title: Optional[str] = None,
    debug: Optional[bool] = None,
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
        description="Run tldw_chatbook in a web browser", prog="tldw-serve"
    )
    parser.add_argument(
        "--host", type=str, help="Host address to bind to (default: localhost)"
    )
    parser.add_argument("--port", type=int, help="Port to bind to (default: 8000)")
    parser.add_argument("--title", type=str, help="Title for the web page")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Run the web server with provided arguments
    run_web_server(host=args.host, port=args.port, title=args.title, debug=args.debug)


if __name__ == "__main__":
    main()
