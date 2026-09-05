# serve.py
"""
Web server module for running tldw_chatbook in a browser using textual-serve.

This module provides functions to launch the Textual application as a web server,
allowing users to access the TUI through their web browser.
"""

from __future__ import annotations

import asyncio
import html
import signal
import ssl
import sys
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

from loguru import logger

from ..Canvas.compiler import compile_canvas_document
from ..Canvas.control_protocol import CanvasControlBroker, ControlProtocolError
from ..Canvas.gateway import (
    BridgeConfirmationResponse,
    BridgePreparationResponse,
    CanvasGateway,
    CanvasGatewayEvent,
    CanvasGatewayNavigation,
    CanvasGatewayOption,
    CanvasGatewayProjection,
    CanvasGatewayScope,
    CanvasSourceResponse,
)
from ..Canvas.web_auth import (
    CSRF_HEADER_NAME,
    SESSION_COOKIE_NAME,
    WEBSOCKET_PROTOCOL,
    AuthenticationError,
    RequestFacts,
    WebAuthManager,
    WebAuthPolicy,
    build_web_auth_policy,
    resolve_web_access_token,
)
from ..config import CanvasConfigPolicy, get_canvas_config_policy, get_cli_setting
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
_CHATBOOK_AUTH_BOOTSTRAP_JS = """(() => {
  "use strict";
  const NativeWebSocket = window.WebSocket;
  if (NativeWebSocket.__chatbookAuthBridge) return;

  function ChatbookWebSocket(url, protocols) {
    const target = new URL(url, document.baseURI);
    const csrf = document.querySelector('meta[name="chatbook-csrf"]')?.content;
    const isChatbookSocket =
      (target.protocol === "ws:" || target.protocol === "wss:") &&
      target.host === window.location.host &&
      target.pathname === "/ws";
    if (isChatbookSocket && protocols === undefined && csrf) {
      return new NativeWebSocket(url, ["chatbook-v1", `csrf.${csrf}`]);
    }
    return protocols === undefined
      ? new NativeWebSocket(url)
      : new NativeWebSocket(url, protocols);
  }

  Object.setPrototypeOf(ChatbookWebSocket, NativeWebSocket);
  ChatbookWebSocket.prototype = NativeWebSocket.prototype;
  Object.defineProperty(ChatbookWebSocket, "__chatbookAuthBridge", {
    value: true,
  });
  window.WebSocket = ChatbookWebSocket;
})();
"""
_SERVED_STATIC_ROOT = files("tldw_chatbook.Web_Server").joinpath("static")
_SERVED_SHELL_HTML = _SERVED_STATIC_ROOT.joinpath("served_shell.html").read_text(
    encoding="utf-8"
)
_SERVED_SHELL_ASSETS = {
    "served-shell.css": ("text/css", "served_shell.css"),
    "served-shell.js": ("application/javascript", "served_shell.js"),
}


class ServedCanvasUnavailable(RuntimeError):
    """The authenticated browser has no usable exact child Canvas scope."""


def _json_wire(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_wire(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_wire(child) for child in value]
    return value


class _ServedCanvasAuthorityProxy:
    """Bounded parent transport for one browser-scoped AppService child."""

    def __init__(self, owner: ChatbookWebServerMixin) -> None:
        self._owner = owner

    async def _request(self, scope, message_type, payload):
        child_id = self._owner._served_browser_children.get(scope.browser_session_id)
        broker = getattr(self._owner, "_canvas_control_broker", None)
        if child_id is None or broker is None:
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        try:
            return await broker.request(child_id, message_type, payload, timeout=2.0)
        except Exception:  # noqa: BLE001 - private transport failures stay bounded
            raise ServedCanvasUnavailable("canvas_session_unavailable") from None

    async def _read(self, scope: CanvasGatewayScope):
        response = await self._request(
            scope,
            "canvas.read.request",
            {"canvas_id": scope.canvas_id, "revision_id": scope.revision_id},
        )
        payload = response.payload
        if (
            payload.get("canvas_id") != scope.canvas_id
            or payload.get("revision_id") != scope.revision_id
        ):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        metadata = payload.get("render_metadata")
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("source"), str
        ):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        return payload, metadata

    async def resolve_render_plan(self, scope: CanvasGatewayScope):
        _payload, metadata = await self._read(scope)
        return compile_canvas_document(metadata["source"])

    async def read_source(self, scope: CanvasGatewayScope):
        payload, metadata = await self._read(scope)
        return CanvasSourceResponse(metadata["source"], str(payload["content_sha256"]))

    async def describe_selection(self, scope: CanvasGatewayScope):
        read_payload, metadata = await self._read(scope)
        list_response = await self._request(scope, "canvas.list.request", {})
        canvases = list_response.payload.get("canvases")
        projection = metadata.get("projection")
        if not isinstance(canvases, list) or not isinstance(projection, dict):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        projection_metadata = projection.get("metadata")
        if not isinstance(projection_metadata, dict):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        options: list[CanvasGatewayOption] = []
        for item in canvases:
            if not isinstance(item, dict):
                raise ServedCanvasUnavailable("canvas_session_unavailable")
            try:
                options.append(
                    CanvasGatewayOption(
                        str(item["canvas_id"]),
                        str(item["revision_id"]),
                        str(item["title"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                raise ServedCanvasUnavailable("canvas_session_unavailable") from None
        try:
            return CanvasGatewayProjection(
                scope=scope,
                options=tuple(options),
                title=str(read_payload["title"]),
                sequence=int(projection_metadata["sequence"]),
                parent_revision_id=projection_metadata.get("parent_revision_id"),
                source_bytes=int(read_payload["source_bytes"]),
                content_sha256=str(read_payload["content_sha256"]),
                origin_message_id=str(projection_metadata["origin_message_id"]),
                origin_turn_id=str(projection_metadata["origin_turn_id"]),
                temporary=projection_metadata.get("temporary") is True,
                following=projection.get("following") is True,
            )
        except (KeyError, TypeError, ValueError):
            raise ServedCanvasUnavailable("canvas_session_unavailable") from None

    async def navigate(self, scope, *, action, canvas_id=None, title=None):
        payload: dict[str, object] = {"action": action}
        if canvas_id is not None:
            payload["canvas_id"] = canvas_id
        if title is not None:
            payload["title"] = title
        response = await self._request(scope, "selection.request", payload)
        next_scope = CanvasGatewayScope(
            browser_session_id=scope.browser_session_id,
            conversation_session_id=scope.conversation_session_id,
            canvas_id=str(response.payload["canvas_id"]),
            revision_id=str(response.payload["revision_id"]),
        )
        return CanvasGatewayNavigation(
            next_scope, await self.describe_selection(next_scope)
        )

    async def read_events(self, scope, *, after_event_id):
        response = await self._request(
            scope,
            "canvas.events.request",
            {"after_event_id": after_event_id},
        )
        wire_events = response.payload.get("events")
        if not isinstance(wire_events, list):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        events: list[CanvasGatewayEvent] = []
        try:
            for value in wire_events:
                if not isinstance(value, dict):
                    raise TypeError("invalid event")
                event = CanvasGatewayEvent(
                    event_id=str(value["event_id"]),
                    kind=value["kind"],
                    canvas_id=str(value["canvas_id"]),
                    revision_id=str(value["revision_id"]),
                    metadata=value["metadata"],
                )
                if event.canvas_id != scope.canvas_id:
                    raise ValueError("event scope mismatch")
                events.append(event)
        except (KeyError, TypeError, ValueError):
            raise ServedCanvasUnavailable("canvas_session_unavailable") from None
        return tuple(events)

    async def prepare_bridge(self, scope, request):
        response = await self._request(
            scope,
            "bridge.request",
            {
                "request": {
                    "version": request.version,
                    "request_id": request.request_id,
                    "kind": request.kind,
                    "value": _json_wire(request.value),
                }
            },
        )
        presentation = response.payload.get("presentation")
        if not isinstance(presentation, dict):
            raise ServedCanvasUnavailable("bridge_refused")
        try:
            result = BridgePreparationResponse(
                request_id=str(presentation["request_id"]),
                kind=str(presentation["kind"]),
                conversation_id=str(presentation["conversation_id"]),
                canvas_id=str(presentation["canvas_id"]),
                revision_id=str(presentation["revision_id"]),
                canvas_title=str(presentation["canvas_title"]),
                revision_number=int(presentation["revision_number"]),
                complete_text=presentation.get("complete_text"),
                filename=presentation.get("filename"),
                mime_type=presentation.get("mime_type"),
                byte_size=presentation.get("byte_size"),
            )
        except (KeyError, TypeError, ValueError):
            raise ServedCanvasUnavailable("bridge_refused") from None
        preparation_nonce = response.payload.get("preparation_nonce")
        if not isinstance(preparation_nonce, str):
            raise ServedCanvasUnavailable("bridge_refused")
        return result, preparation_nonce

    async def confirm_bridge(self, scope, request, *, settlement, preparation=None):
        if not isinstance(preparation, str):
            return BridgeConfirmationResponse(request.request.request_id, "refused")
        reserve_external = getattr(settlement, "reserve_external", None)
        commit_external = getattr(settlement, "commit_external", None)
        if request.approved and (
            not callable(reserve_external) or not reserve_external()
        ):
            try:
                await self._request(
                    scope,
                    "bridge.decision.request",
                    {
                        "request_id": request.request.request_id,
                        "preparation_nonce": preparation,
                        "approved": False,
                    },
                )
            except ServedCanvasUnavailable:
                pass
            return BridgeConfirmationResponse(request.request.request_id, "refused")
        response = None
        for attempt in range(2):
            try:
                response = await self._request(
                    scope,
                    "bridge.decision.request",
                    {
                        "request_id": request.request.request_id,
                        "preparation_nonce": preparation,
                        "approved": request.approved,
                    },
                )
                break
            except ServedCanvasUnavailable:
                if attempt:
                    raise
        assert response is not None
        status = response.payload.get("status")
        if status == "confirmed" and (
            not callable(commit_external) or not commit_external()
        ):
            status = "refused"
        if status not in {"confirmed", "cancelled", "refused"}:
            status = "refused"
        return BridgeConfirmationResponse(request.request.request_id, status)


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

    def __init__(
        self,
        *args,
        web_auth_policy: WebAuthPolicy | None = None,
        web_ssl_context: ssl.SSLContext | None = None,
        canvas_policy: CanvasConfigPolicy | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if web_auth_policy is None:
            web_auth_policy = build_web_auth_policy(
                host=self.host,
                port=self.port,
                access_token=None,
            )
        self._web_auth = WebAuthManager(web_auth_policy)
        self._web_ssl_context = web_ssl_context
        self._canvas_policy = canvas_policy or get_canvas_config_policy()
        self._canvas_disabled_latched = not self._canvas_policy.enabled
        self._canvas_policy_watch_task: asyncio.Task[None] | None = None
        self._served_browser_children: dict[str, str] = {}
        self._served_canvas_gateway = CanvasGateway(
            authority=_ServedCanvasAuthorityProxy(self)
        )
        self._served_canvas_launches: dict[str, tuple[CanvasGatewayScope, object]] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(host={self.host!r}, port={self.port!r}, "
            f"title={self.title!r})"
        )

    def serve(self, debug: bool = False) -> None:
        """Run the owned aiohttp origin, including optional direct TLS."""

        from aiohttp import web

        self.debug = debug
        self.initialize_logging()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self.request_exit)
            loop.add_signal_handler(signal.SIGTERM, self.request_exit)
        except NotImplementedError:
            pass
        web.run_app(
            self._make_app(),
            host=self.host,
            port=self.port,
            handle_signals=False,
            loop=loop,
            print=lambda *args: None,
            ssl_context=self._web_ssl_context,
            access_log=None,
        )

    async def _make_app(self):
        """Make the aiohttp app and override only the resize-sensitive JS asset."""
        import aiohttp_jinja2
        import jinja2
        from aiohttp import web

        @web.middleware
        async def web_auth_middleware(request, handler):
            return await self._web_auth_middleware(request, handler)

        @web.middleware
        async def served_canvas_middleware(request, handler):
            return await self._served_canvas_middleware(request, handler)

        app = web.Application(
            middlewares=[web_auth_middleware, served_canvas_middleware],
            client_max_size=16 * 1024,
        )
        aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(self.templates_path))

        routes = [
            web.get("/auth/login", self.handle_login, name="login"),
            web.post("/auth/login", self.handle_login, name="login_submit"),
            web.get(
                "/auth/bootstrap", self.handle_login_bootstrap, name="login_bootstrap"
            ),
            web.get("/", self.handle_index, name="index"),
            web.get("/ws", self.handle_websocket, name="websocket"),
            web.get(
                "/canvas/api/session",
                self.handle_served_canvas_session,
                name="served_canvas_session",
            ),
            web.get("/download/{key}", self.handle_download, name="download"),
            web.get(
                "/static/js/chatbook-auth.js",
                self.handle_chatbook_auth_js,
                name="chatbook_auth_js",
            ),
            web.get(
                "/static/chatbook/{name}",
                self.handle_served_shell_asset,
                name="served_shell_asset",
            ),
            web.get("/static/js/textual.js", self.handle_textual_js, name="textual_js"),
            web.static("/static", self.statics_path, show_index=False, name="static"),
        ]
        if self._canvas_enabled():
            routes.insert(
                -2,
                web.get(
                    "/static/chatbook-canvas/{name}",
                    self._served_canvas_gateway.public_runtime_asset,
                    name="served_canvas_runtime_asset",
                ),
            )
        app.add_routes(routes)
        if self._canvas_enabled():
            self._served_canvas_gateway.mount_on_app(app, origin=self.public_url)

        app.on_startup.append(self.on_startup)
        app.on_shutdown.append(self.on_shutdown)
        return app

    def _request_facts(self, request) -> RequestFacts:
        peer_ip = request.remote or ""
        websocket_protocols = tuple(
            protocol.strip()
            for protocol in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
            if protocol.strip()
        )
        return RequestFacts(
            method=request.method,
            path=request.path,
            peer_ip=peer_ip,
            scheme=request.scheme,
            host=request.headers.get("Host", ""),
            origin=request.headers.get("Origin"),
            cookie_value=request.cookies.get(SESSION_COOKIE_NAME),
            csrf_token=request.headers.get(CSRF_HEADER_NAME),
            upgrade=request.headers.get("Upgrade", ""),
            connection=request.headers.get("Connection", ""),
            websocket_protocols=websocket_protocols,
            forwarded_for=request.headers.get("X-Forwarded-For"),
            forwarded_proto=request.headers.get("X-Forwarded-Proto"),
            forwarded_host=request.headers.get("X-Forwarded-Host"),
            fetch_site=request.headers.get("Sec-Fetch-Site"),
        )

    @staticmethod
    def _set_session_cookie(response, grant, *, secure: bool) -> None:
        max_age = max(1, int(grant.expires_at - asyncio.get_running_loop().time()))
        response.set_cookie(
            SESSION_COOKIE_NAME,
            grant.cookie_value,
            max_age=max_age,
            httponly=True,
            secure=secure,
            samesite="Strict",
            path="/",
        )

    @property
    def _public_auth_paths(self) -> frozenset[str]:
        return frozenset({"/auth/login", "/auth/bootstrap"})

    @property
    def _public_asset_paths(self) -> tuple[str, ...]:
        return ("/static/",)

    async def _web_auth_middleware(self, request, handler):
        from aiohttp import web

        facts = self._request_facts(request)
        if request.path in self._public_auth_paths:
            try:
                self._web_auth.validate_public_request(
                    facts, require_origin=request.method != "GET"
                )
            except AuthenticationError:
                raise web.HTTPUnauthorized(text="Authentication denied") from None
            return await handler(request)
        if request.path.startswith(self._public_asset_paths):
            try:
                self._web_auth.validate_public_request(facts, require_origin=False)
            except AuthenticationError:
                raise web.HTTPUnauthorized(text="Authentication denied") from None
            return await handler(request)

        grant = None
        try:
            websocket = request.path == "/ws"
            mounted_canvas = request.path.startswith("/canvas/gateway-")
            session = self._web_auth.authenticate_request(
                facts,
                # Mounted Canvas mutations carry their own session-scoped
                # X-Canvas-CSRF proof. Requiring the outer token as well would
                # force that separate credential into the nested shell.
                require_csrf=(
                    request.method in {"POST", "PUT", "PATCH", "DELETE"}
                    and not mounted_canvas
                ),
                websocket=websocket,
            )
            request["chatbook_browser_session"] = session
            request["chatbook_csrf"] = session.csrf_token
        except AuthenticationError:
            if (
                self._web_auth.policy.automatic_local_login
                and request.path == "/"
                and request.method == "GET"
                and facts.fetch_site in {None, "none", "same-origin"}
            ):
                try:
                    self._web_auth.validate_public_request(
                        facts, require_origin=facts.origin is not None
                    )
                    grant = self._web_auth.authenticate_local(client_ip=facts.peer_ip)
                except AuthenticationError:
                    raise web.HTTPUnauthorized(text="Authentication denied") from None
                request["chatbook_csrf"] = grant.csrf_token
            elif request.path == "/" and request.method == "GET":
                raise web.HTTPSeeOther(location="/auth/login") from None
            else:
                raise web.HTTPUnauthorized(text="Authentication denied") from None

        response = await handler(request)
        if grant is not None:
            self._set_session_cookie(
                response,
                grant,
                secure=self._web_auth.policy.secure_cookies,
            )
        return response

    async def _served_canvas_middleware(self, request, handler):
        """Fence every mounted Canvas URL to its exact outer browser session."""

        if not request.path.startswith("/canvas/gateway-"):
            return await handler(request)
        if not self._canvas_enabled():
            await self._disable_canvas_runtime()
            from aiohttp import web

            raise web.HTTPNotFound(text="Canvas unavailable")
        session = request.get("chatbook_browser_session")
        browser_session_id = "" if session is None else session.session_id
        return await self._served_canvas_gateway.handle_mounted_request(
            request,
            browser_session_id=browser_session_id,
            handler=handler,
        )

    async def handle_login(self, request):
        from aiohttp import web

        if request.method == "GET":
            return web.Response(
                text=(
                    '<!doctype html><html><head><meta charset="utf-8">'
                    "<title>Chatbook sign in</title></head><body>"
                    '<main><h1>Chatbook</h1><form method="post" action="/auth/login">'
                    '<label>Access token <input type="password" name="access_token" '
                    'autocomplete="current-password" required></label>'
                    '<button type="submit">Sign in</button></form></main></body></html>'
                ),
                content_type="text/html",
                headers={"Cache-Control": "no-store"},
            )
        if request.content_length is not None and request.content_length > 4096:
            raise web.HTTPRequestEntityTooLarge(
                max_size=4096, actual_size=request.content_length
            )
        data = await request.post()
        facts = self._request_facts(request)
        try:
            client_ip, _scheme, _host = self._web_auth.validate_public_request(
                facts, require_origin=True
            )
            grant = self._web_auth.login_with_access_token(
                str(data.get("access_token", "")), client_ip=client_ip
            )
        except AuthenticationError:
            raise web.HTTPUnauthorized(text="Authentication denied") from None
        response = web.Response(status=303, headers={"Location": "/"})
        self._set_session_cookie(
            response, grant, secure=self._web_auth.policy.secure_cookies
        )
        return response

    async def handle_login_bootstrap(self, request):
        from aiohttp import web

        facts = self._request_facts(request)
        try:
            client_ip, _scheme, _host = self._web_auth.validate_public_request(
                facts, require_origin=False
            )
            grant = self._web_auth.exchange_bootstrap(
                request.query.get("nonce", ""), client_ip=client_ip
            )
        except AuthenticationError:
            raise web.HTTPUnauthorized(text="Authentication denied") from None
        response = web.Response(status=303, headers={"Location": "/"})
        self._set_session_cookie(
            response, grant, secure=self._web_auth.policy.secure_cookies
        )
        return response

    def issue_browser_bootstrap_url(self) -> str:
        """Return a one-time bootstrap URL without exposing the configured token."""

        nonce = self._web_auth.issue_bootstrap()
        return f"{self.public_url.rstrip('/')}/auth/bootstrap?nonce={quote(nonce)}"

    async def on_startup(self, app) -> None:
        """Start the private loopback control broker before children spawn."""

        if self._canvas_enabled():
            self._canvas_control_broker = CanvasControlBroker()
            await self._canvas_control_broker.start()
            self._canvas_policy_watch_task = asyncio.create_task(
                self._watch_canvas_policy(),
                name="chatbook-served-canvas-policy",
            )
        else:
            self._canvas_control_broker = None
        await super().on_startup(app)

    async def on_shutdown(self, app) -> None:
        """Revoke all child control capabilities with the served process."""

        policy_task = getattr(self, "_canvas_policy_watch_task", None)
        self._canvas_policy_watch_task = None
        if policy_task is not None:
            policy_task.cancel()
            try:
                await policy_task
            except asyncio.CancelledError:
                pass
        broker = getattr(self, "_canvas_control_broker", None)
        if broker is not None:
            await broker.aclose()
            self._canvas_control_broker = None
        await self._served_canvas_gateway.aclose()
        self._served_canvas_launches.clear()
        self._served_browser_children.clear()
        self._web_auth.revoke_all()
        await super().on_shutdown(app)

    def _canvas_enabled(self) -> bool:
        """Read the global kill switch while honoring the restart latch."""

        if self._canvas_disabled_latched:
            return False
        try:
            return get_canvas_config_policy().enabled is True
        except Exception:  # noqa: BLE001 - server feature gate fails closed
            return False

    async def _watch_canvas_policy(self, *, interval_seconds: float = 0.25) -> None:
        """Revoke every served child promptly after the shared switch turns off."""

        while self._canvas_enabled():
            await asyncio.sleep(interval_seconds)
        await self._disable_canvas_runtime()

    async def _disable_canvas_runtime(self) -> None:
        """Idempotently revoke served Canvas control and browser delivery."""

        if self._canvas_disabled_latched and getattr(
            self, "_canvas_control_broker", None
        ) is None:
            self._served_canvas_launches.clear()
            self._served_browser_children.clear()
            return
        self._canvas_disabled_latched = True
        broker = getattr(self, "_canvas_control_broker", None)
        self._canvas_control_broker = None
        if broker is not None:
            await broker.aclose()
        await self._served_canvas_gateway.aclose()
        self._served_canvas_launches.clear()
        self._served_browser_children.clear()

    async def _expire_websocket_session(self, session) -> None:
        """Revoke a connected channel at its moving idle/absolute deadline."""

        while not session.revoked:
            await asyncio.sleep(self._web_auth.seconds_until_expiry(session))
            if self._web_auth.expire_session_if_due(session):
                return

    async def _process_authenticated_messages(
        self, websocket, app_service, session
    ) -> None:
        """Process terminal frames while keeping browser-session idle time honest."""

        from aiohttp import WSMsgType

        async for message in websocket:
            if message.type != WSMsgType.TEXT:
                continue
            if session is not None:
                self._web_auth.touch_session(session)
            envelope = message.json()
            if not isinstance(envelope, list) or not envelope:
                continue
            type_ = envelope[0]
            if type_ == "stdin":
                await app_service.send_bytes(envelope[1].encode("utf-8"))
            elif type_ == "resize":
                data = envelope[1]
                await app_service.set_terminal_size(data["width"], data["height"])
            elif type_ == "ping":
                await websocket.send_json(["pong", envelope[1]])
            elif type_ == "blur":
                await app_service.blur()
            elif type_ == "focus":
                await app_service.focus()

    def bind_served_browser(self, browser_session_id: str, child_id: str) -> None:
        """Bind one authenticated browser to exactly one AppService child."""

        if not isinstance(browser_session_id, str) or not browser_session_id:
            raise ValueError("browser session ID is required")
        if not isinstance(child_id, str) or not child_id:
            raise ValueError("child ID is required")
        current = self._served_browser_children.get(browser_session_id)
        if current is not None and current != child_id:
            raise ServedCanvasUnavailable("browser_session_already_bound")
        if any(
            owner != browser_session_id and owned_child == child_id
            for owner, owned_child in self._served_browser_children.items()
        ):
            raise ServedCanvasUnavailable("child_session_already_bound")
        self._served_browser_children[browser_session_id] = child_id

    def unbind_served_browser(self, browser_session_id: str, child_id: str) -> None:
        """Remove a binding only when its exact child incarnation still owns it."""

        if self._served_browser_children.get(browser_session_id) == child_id:
            self._served_browser_children.pop(browser_session_id, None)
            self._served_canvas_launches.pop(browser_session_id, None)
            self._served_canvas_gateway.mark_browser_session_unavailable(
                browser_session_id
            )

    async def served_canvas_state(self, browser_session_id: str) -> dict[str, object]:
        """Return only the Canvas state owned by one exact authenticated child."""

        if not self._canvas_enabled():
            await self._disable_canvas_runtime()
            raise ServedCanvasUnavailable("canvas_disabled")
        child_id = self._served_browser_children.get(browser_session_id)
        broker = getattr(self, "_canvas_control_broker", None)
        if child_id is None or broker is None:
            raise ServedCanvasUnavailable("canvas_session_unavailable")

        fixture_reader = getattr(broker, "browser_state", None)
        try:
            if callable(fixture_reader):
                state = await fixture_reader(child_id)
            else:
                response = await broker.request(
                    child_id,
                    "scope.snapshot.request",
                    {},
                    timeout=1.0,
                )
                payload = response.payload
                canvas_id = payload.get("selected_canvas_id")
                revision_id = payload.get("selected_revision_id")
                if not isinstance(canvas_id, str) or not isinstance(revision_id, str):
                    return {"status": "terminal_only"}
                state = {
                    "status": "ready",
                    "canvas_id": canvas_id,
                    "revision_id": revision_id,
                    "conversation_session_id": payload["session_id"],
                }
        except ControlProtocolError as error:
            if error.code == "scope_unavailable":
                return {"status": "terminal_only"}
            logger.debug(
                "Served Canvas unavailable child_type={} code=control_unavailable",
                type(error).__name__,
            )
            raise ServedCanvasUnavailable("canvas_session_unavailable") from None
        except Exception as error:  # noqa: BLE001 - private transport errors stay bounded
            logger.debug(
                "Served Canvas unavailable child_type={} code=control_unavailable",
                type(error).__name__,
            )
            raise ServedCanvasUnavailable("canvas_session_unavailable") from None

        if not isinstance(state, dict):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        status = state.get("status")
        if status not in {"ready", "terminal_only", "disconnected", "reconnecting"}:
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        result = dict(state)
        if status != "ready":
            return result
        canvas_id = result.get("canvas_id")
        revision_id = result.get("revision_id")
        if not isinstance(canvas_id, str) or not isinstance(revision_id, str):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        scope = CanvasGatewayScope(
            browser_session_id=browser_session_id,
            conversation_session_id=str(
                result.get("conversation_session_id", browser_session_id)
            ),
            canvas_id=canvas_id,
            revision_id=revision_id,
        )
        existing = self._served_canvas_launches.get(browser_session_id)
        if existing is None or not self._served_canvas_gateway.has_shell_binding(
            browser_session_id
        ):
            launch = await self._served_canvas_gateway.open_shell(scope)
            self._served_canvas_launches[browser_session_id] = (scope, launch)
        else:
            prior_scope, launch = existing
            if prior_scope != scope:
                self._served_canvas_gateway.change_selection(
                    browser_session_id=browser_session_id, scope=scope
                )
                self._served_canvas_launches[browser_session_id] = (scope, launch)
        result["url"] = launch.browser_url
        return result

    async def handle_served_canvas_session(self, request):
        """Expose bounded child-owned state to its matching authenticated browser."""

        from aiohttp import web

        session = request.get("chatbook_browser_session")
        if session is None:
            raise web.HTTPNotFound(text="Canvas unavailable")
        if not self._canvas_enabled():
            await self._disable_canvas_runtime()
            raise web.HTTPNotFound(text="Canvas unavailable")
        try:
            state = await self.served_canvas_state(session.session_id)
        except ServedCanvasUnavailable:
            state = {
                "status": (
                    "reconnecting"
                    if session.session_id in self._served_browser_children
                    else "terminal_only"
                )
            }
        return web.json_response(state, headers={"Cache-Control": "no-store"})

    async def handle_websocket(self, request):
        """Bind one browser websocket to exactly one authenticated child."""

        from aiohttp import web

        websocket = web.WebSocketResponse(
            heartbeat=15,
            protocols=(WEBSOCKET_PROTOCOL,),
        )
        width = _web_dimension(request.query.get("width"), 80)
        height = _web_dimension(request.query.get("height"), 24)
        app_service = None
        try:
            await websocket.prepare(request)
            session = request.get("chatbook_browser_session")
            unregister_channel = (
                self._web_auth.register_channel(
                    session,
                    lambda: asyncio.create_task(websocket.close(code=4001)),
                )
                if session is not None
                else (lambda: None)
            )
            expiry_task = (
                asyncio.create_task(self._expire_websocket_session(session))
                if session is not None
                else None
            )
            app_service_class = getattr(self, "_chatbook_app_service_class", None)
            broker = getattr(self, "_canvas_control_broker", None)
            if app_service_class is None:
                raise RuntimeError("served terminal app service is unavailable")
            app_service = app_service_class(
                self.command,
                write_bytes=websocket.send_bytes,
                write_str=websocket.send_str,
                close=websocket.close,
                download_manager=self.download_manager,
                debug=self.debug,
                canvas_control_broker=broker,
            )
            child_id = app_service.app_service_id
            if session is not None:
                self.bind_served_browser(session.session_id, child_id)
            await app_service.start(width, height)
            try:
                await self._process_authenticated_messages(
                    websocket, app_service, session
                )
            finally:
                await app_service.stop()
        except asyncio.CancelledError:
            await websocket.close()
        except Exception as error:  # noqa: BLE001 - websocket boundary stays alive
            # No traceback here: the spawn frame holds the per-child secret.
            logger.error("Served terminal session failed type={}", type(error).__name__)
        finally:
            expiry = locals().get("expiry_task")
            if expiry is not None:
                expiry.cancel()
            unregister = locals().get("unregister_channel")
            if unregister is not None:
                unregister()
            if app_service is not None:
                session = locals().get("session")
                child_id = getattr(app_service, "app_service_id", None)
                if session is not None and isinstance(child_id, str):
                    self.unbind_served_browser(session.session_id, child_id)
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
        """Serve Chatbook's owned terminal-first, Canvas-ready browser shell."""
        from aiohttp import web

        font_size = resolve_web_font_size(request.query.get("fontsize"))
        csrf = html.escape(str(request["chatbook_csrf"]), quote=True)
        body = (
            _SERVED_SHELL_HTML.replace("__CHATBOOK_CSRF__", csrf)
            .replace("__CHATBOOK_TITLE__", html.escape(self.title, quote=True))
            .replace(
                "__APP_WEBSOCKET_URL__",
                html.escape(self._app_websocket_url, quote=True),
            )
            .replace("__FONT_SIZE__", str(font_size))
        )
        return web.Response(
            text=body,
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-store"},
        )

    async def handle_served_shell_asset(self, request):
        """Serve only the two immutable, state-free owned shell assets."""

        from aiohttp import web

        asset = _SERVED_SHELL_ASSETS.get(request.match_info.get("name", ""))
        if asset is None:
            raise web.HTTPNotFound(text="Asset not found")
        content_type, filename = asset
        return web.Response(
            body=_SERVED_STATIC_ROOT.joinpath(filename).read_bytes(),
            content_type=content_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    async def handle_chatbook_auth_js(self, request):
        """Serve the owned, state-free browser authentication bootstrap."""
        from aiohttp import web

        return web.Response(
            text=_CHATBOOK_AUTH_BOOTSTRAP_JS,
            content_type="application/javascript",
            headers={"Cache-Control": "public, max-age=3600"},
        )

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
            if self._canvas_control_broker is not None:
                launch = self._canvas_control_broker.issue_child(self.app_service_id)
                self._canvas_control_environment = dict(launch.environment)
            try:
                await super().start(width, height)
            except BaseException:
                if self._canvas_control_broker is not None:
                    await self._canvas_control_broker.revoke_child(self.app_service_id)
                self._canvas_control_environment.clear()
                raise
            self._canvas_control_started = self._canvas_control_broker is not None

        async def stop(self) -> None:
            try:
                await super().stop()
            finally:
                if self._canvas_control_started:
                    self._canvas_control_started = False
                    if self._canvas_control_broker is not None:
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
    except Exception as exc:  # noqa: BLE001 - optional dependency boundary
        logger.warning(
            f"Web server dependency probe failed. Web mode is unavailable. Reason: {exc}"
        )
        DEPENDENCIES_AVAILABLE["web"] = False
        DEPENDENCIES_AVAILABLE["textual_serve"] = False
        return False


def create_server(
    host: str = "localhost",
    port: int = 8000,
    title: str | None = None,
    debug: bool = False,
    public_url: str | None = None,
    allow_insecure_remote_http: bool | None = None,
    trusted_proxy_addresses: list[str] | None = None,
    access_token: str | None = None,
    tls_certificate: str | None = None,
    tls_private_key: str | None = None,
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

    if public_url is None:
        public_url = get_cli_setting("web_server", "public_url", default="") or None
    if allow_insecure_remote_http is None:
        allow_insecure_remote_http = bool(
            get_cli_setting("web_server", "allow_insecure_remote_http", default=False)
        )
    if trusted_proxy_addresses is None:
        configured_proxies = get_cli_setting(
            "web_server", "trusted_proxy_addresses", default=[]
        )
        trusted_proxy_addresses = (
            [str(value) for value in configured_proxies]
            if isinstance(configured_proxies, list)
            else []
        )
    configured_access_token = (
        access_token
        if access_token is not None
        else get_cli_setting("web_server", "access_token", default="")
    )
    resolved_access_token = resolve_web_access_token(configured_access_token)
    if tls_certificate is None:
        tls_certificate = (
            get_cli_setting("web_server", "tls_certificate", default="") or None
        )
    if tls_private_key is None:
        tls_private_key = (
            get_cli_setting("web_server", "tls_private_key", default="") or None
        )
    if bool(tls_certificate) != bool(tls_private_key):
        raise ValueError("direct TLS requires both certificate and private key files")
    ssl_context = None
    if tls_certificate and tls_private_key:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(tls_certificate, tls_private_key)

    web_auth_policy = build_web_auth_policy(
        host=host,
        port=port,
        access_token=resolved_access_token,
        public_url=public_url,
        allow_insecure_remote_http=allow_insecure_remote_http,
        trusted_proxy_addresses=trusted_proxy_addresses,
        direct_tls=ssl_context is not None,
    )
    if public_url is None and ssl_context is not None:
        formatted_host = (
            f"[{host}]" if ":" in host and not host.startswith("[") else host
        )
        public_url = (
            f"https://{formatted_host}"
            if port == 443
            else f"https://{formatted_host}:{port}"
        )
    if web_auth_policy.insecure_remote_http:
        logger.warning(
            "INSECURE DEVELOPMENT MODE: remote Chatbook content is using plaintext HTTP"
        )

    logger.info("Creating web server on {}:{}", host, port)
    server = chatbook_web_server_class(
        command=command,
        host=host,
        port=port,
        title=title,
        public_url=public_url,
        web_auth_policy=web_auth_policy,
        web_ssl_context=ssl_context,
        canvas_policy=get_canvas_config_policy(),
    )

    return server


def run_web_server(
    host: str | None = None,
    port: int | None = None,
    title: str | None = None,
    debug: bool | None = None,
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

    logger.info("Starting web server on {}:{}", host, port)
    logger.info("Press Ctrl+C to stop the server")

    try:
        server.serve(debug=bool(debug))
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
