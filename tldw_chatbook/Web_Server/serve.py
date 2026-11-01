# serve.py
"""
Web server module for running tldw_chatbook in a browser using textual-serve.

This module provides functions to launch the Textual application as a web server,
allowing users to access the TUI through their web browser.
"""

from __future__ import annotations

# ADR-126: fence recovery and enroll before any runtime/config imports.
from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup

admit_startup()


import asyncio
import html
import json
import os
import signal
import ssl
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlsplit, urlunparse

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
    CanvasSelectionChanged,
    CanvasSourceResponse,
)
from ..Canvas.limits import UnsupportedCanvasRuntimeProfile
from ..Canvas.profiles import (
    ProfileSnapshot,
    load_application_profile_snapshot,
    load_profile_snapshot,
    resolve_profile,
    runtime_snapshot_id,
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
from ..config import (
    CanvasConfigPolicy,
    get_canvas_config_policy,
    get_canvas_execution_enabled,
    get_cli_setting,
)
from ..Utils.input_validation import validate_number_range
from ..Utils.optional_deps import (
    DEPENDENCIES_AVAILABLE,
    check_web_server_deps,
    require_dependency,
)
from ..Utils.paths import get_user_data_dir
from .mobile_api import (
    MOBILE_MAX_BODY_BYTES,
    MOBILE_STREAM_PATH,
    build_mobile_routes,
    mobile_surface_enabled,
)
from .session_store import WebSessionStore

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
_CHATBOOK_DEFAULT_WEB_FONT_SIZE = 16
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

#: One-hour public cache policy shared by every state-free static asset the
#: server owns (the served-shell assets, the auth bootstrap, and the patched
#: textual-serve bundle): each is deterministic for a given installed build,
#: so a single named policy is the one place to change if that ever stops
#: being true.
_STATIC_ASSET_CACHE_CONTROL = "public, max-age=3600"
_SERVED_SHELL_HTML = _SERVED_STATIC_ROOT.joinpath("served_shell.html").read_text(
    encoding="utf-8"
)
_SERVED_SHELL_ASSETS = {
    "served-shell.css": ("text/css", "served_shell.css"),
    "served-shell.js": ("application/javascript", "served_shell.js"),
    "devices.js": ("application/javascript", "devices.js"),
    "qrcodegen.js": ("application/javascript", "qrcodegen.js"),
    "passkeys.js": ("application/javascript", "passkeys.js"),
}

#: Mobile companion assets (ADR-234 Phase 1b): name -> (content type,
#: filename, cache policy). The page shell is never cached (it carries the
#: per-session CSRF substitution through ``handle_mobile_page`` — the raw
#: ``/static/mobile/index.html`` copy serves only the inert placeholder);
#: the state-free app/manifest/icon assets share the one-hour public
#: policy. The committed PNGs are regenerated by
#: ``Helper_Scripts/generate_mobile_icons.py`` (run once by hand, not CI).
_MOBILE_STATIC_ROOT = _SERVED_STATIC_ROOT.joinpath("mobile")
_MOBILE_PAGE_HTML = _MOBILE_STATIC_ROOT.joinpath("index.html").read_text(
    encoding="utf-8"
)
_MOBILE_STATIC_ASSETS = {
    "index.html": ("text/html", "index.html", "no-store"),
    "app.css": ("text/css", "app.css", _STATIC_ASSET_CACHE_CONTROL),
    "app.js": ("application/javascript", "app.js", _STATIC_ASSET_CACHE_CONTROL),
    "manifest.webmanifest": (
        "application/manifest+json",
        "manifest.webmanifest",
        _STATIC_ASSET_CACHE_CONTROL,
    ),
    "icon-192.png": ("image/png", "icon-192.png", _STATIC_ASSET_CACHE_CONTROL),
    "icon-512.png": ("image/png", "icon-512.png", _STATIC_ASSET_CACHE_CONTROL),
    "apple-touch-icon.png": (
        "image/png",
        "apple-touch-icon.png",
        _STATIC_ASSET_CACHE_CONTROL,
    ),
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
        from tldw_chatbook.Canvas.compilation import CanvasCompilation

        self._owner = owner
        self._compilation = CanvasCompilation()

    async def _request(self, scope, message_type, payload):
        child_id = self._owner._served_browser_children.get(scope.browser_session_id)
        broker = getattr(self._owner, "_canvas_control_broker", None)
        if child_id is None or broker is None:
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        try:
            return await broker.request(child_id, message_type, payload, timeout=2.0)
        except ControlProtocolError as error:
            if (
                message_type == "selection.request"
                and error.code == "selection_refused"
            ):
                raise CanvasSelectionChanged("selection_changed") from None
            raise ServedCanvasUnavailable("canvas_session_unavailable") from None
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
        if (
            scope.selection_generation is None
            or metadata.get("selection_generation") != scope.selection_generation
        ):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        return payload, metadata

    @property
    def profile_snapshot(self) -> ProfileSnapshot:
        return self._owner._canvas_profile_snapshot

    async def resolve_render_plan(self, scope: CanvasGatewayScope):
        captured = await self._plan_scope(scope)
        _payload, metadata = await self._read(scope)
        snapshot = self._owner._canvas_profile_snapshot
        resolved = resolve_profile(
            snapshot,
            operation="load",
            parent_profile=metadata.get("runtime_profile"),
            has_diagrams=False,
        )
        if not resolved.executable:
            raise UnsupportedCanvasRuntimeProfile("unsupported Canvas runtime profile")
        plan = await self._compilation.run_async(
            lambda: compile_canvas_document(
                metadata["source"],
                runtime_profile=resolved.profile_id,
                snapshot=snapshot,
            )
        )
        if await self._plan_scope(scope) != captured:
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        return plan

    async def _plan_scope(self, scope: CanvasGatewayScope) -> tuple:
        """Recheck the child-owned branch without transporting another source copy."""
        response = await self._request(scope, "scope.snapshot.request", {})
        payload = response.payload
        if (
            payload["session_id"] != scope.conversation_session_id
            or payload["selected_canvas_id"] != scope.canvas_id
            or payload["selected_revision_id"] != scope.revision_id
        ):
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        return (
            payload["session_id"],
            payload["conversation_id"],
            tuple(payload["active_message_ids"]),
            payload["selected_canvas_id"],
            payload["selected_revision_id"],
            payload["selection_generation"],
        )

    async def read_source(self, scope: CanvasGatewayScope):
        payload, metadata = await self._read(scope)
        return CanvasSourceResponse(
            metadata["source"],
            str(payload["content_sha256"]),
            metadata.get("runtime_profile"),
        )

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
        if scope.selection_generation is None:
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        payload: dict[str, object] = {
            "action": action,
            "expected_session_id": scope.conversation_session_id,
            "expected_canvas_id": scope.canvas_id,
            "expected_revision_id": scope.revision_id,
            "expected_selection_generation": scope.selection_generation,
        }
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
            selection_generation=response.payload["selection_generation"],
        )
        return CanvasGatewayNavigation(
            next_scope, await self.describe_selection(next_scope)
        )

    async def read_events(self, scope, *, after_event_id):
        child_id = self._owner._served_browser_children.get(scope.browser_session_id)
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
                events.append(event)
        except (KeyError, TypeError, ValueError):
            raise ServedCanvasUnavailable("canvas_session_unavailable") from None
        if any(event.canvas_id != scope.canvas_id for event in events):
            # Publication may bind the child to a newly created Canvas before
            # the parent's periodic snapshot observes that selection. Never
            # deliver mismatched events: reconcile only the same live owner,
            # then let the gateway classify its independently advanced epoch.
            await self._owner.served_canvas_state(
                scope.browser_session_id,
                expected_scope=scope,
                expected_child_id=child_id,
            )
            raise ServedCanvasUnavailable("canvas_session_unavailable")
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
    """Patch textual-serve's browser resize hook to repaint after viewport changes.

    task-33130: the repaint machinery must not feed back into the app. An
    earlier revision ran the full resize handler on EVERY output frame (the
    first-byte hook replacement) with an unconditional ``sendSize()`` inside
    every repaint; because the app answers each resize with fresh output,
    that became a self-sustaining loop measured at ~54 resize
    round-trips/second at idle, with the browser tab and the app child both
    pegged (60-100% CPU) and an all-rows xterm refresh per frame. The
    invariants this patch now preserves:

    - ``sendSize()`` fires only when ``fit()`` actually changed the grid
      (textual-serve's own ``terminal.onResize -> sendSize`` wiring remains
      the primary size-sync path);
    - the first-byte trigger repaints once per connection, not per message;
    - the after-write repaint is a 250ms trailing debounce -- a self-healing
      repaint after traffic pauses, never a per-frame full refresh;
    - upstream's WebGL/Canvas renderers are left in place (nulling them
      forced the slow DOM renderer, making every forced refresh a full DOM
      rebuild); the resize repaint still calls ``clearTextureAtlas`` plus a
      full ``refresh``, which is the remedy for the GPU-renderer staleness
      this patch family was built for.

    Args:
        source: The full minified textual-serve browser bundle, as served by
            the installed ``textual_serve`` package.

    Returns:
        The patched bundle, or ``source`` unchanged (fail closed) when the
        bundle does not contain every required upstream hook -- i.e. when
        the installed textual-serve version no longer matches the shape
        this patch was built against.
    """
    if _CHATBOOK_VIEWPORT_PATCH_MARKER in source:
        return source
    if any(hook not in source for hook in _TEXTUAL_SERVE_REQUIRED_VIEWPORT_HOOKS):
        return source

    resize_replacement = (
        "this._chatbookTerminalRepaint=()=>{"
        "try{this.terminal.clearTextureAtlas&&this.terminal.clearTextureAtlas()}catch(e){}"
        "try{this.terminal.refresh(0,this.terminal.rows-1)}catch(e){}"
        "};"
        "this._chatbookViewportRepaint=()=>{"
        "const t=this.terminal.cols,r=this.terminal.rows;"
        "this.fit();"
        "if(this.terminal.cols!==t||this.terminal.rows!==r){"
        "try{this.sendSize&&this.sendSize()}catch(e){}"
        "}"
        "this._chatbookTerminalRepaint();"
        "};"
        "this._chatbookViewportAfterWrite=()=>{"
        "clearTimeout(this._chatbookViewportAfterWriteTimer);"
        "this._chatbookViewportAfterWriteTimer=setTimeout("
        "this._chatbookTerminalRepaint,250)"
        "};"
        "this._chatbookViewportFirstByte=()=>{"
        "if(this._chatbookViewportFirstByteDone)return;"
        "this._chatbookViewportFirstByteDone=!0;"
        "this._chatbookViewportResize();"
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
    patched = source.replace(_TEXTUAL_SERVE_RESIZE_HOOK, resize_replacement, 1)
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
            f't.length>10&&({_TEXTUAL_SERVE_LOADED_HOOK.replace("-loaded", "-first-byte")},'
            "this._chatbookViewportFirstByte())"
        ),
        1,
    )


#: First-match platform markers for the served device-manager labels; the
#: given order is a precedence chain (Android before Linux, etc.).
_DEVICE_PLATFORM_MARKERS = (
    "iPhone",
    "iPad",
    "Android",
    "Macintosh",
    "Windows",
    "Linux",
)
#: First-match browser markers; "Edg/" must win over the Chrome/Safari tokens
#: every Edge user agent also carries.
_DEVICE_BROWSER_MARKERS = (
    ("Edg/", "Edge"),
    ("Firefox", "Firefox"),
    ("Chrome", "Chrome"),
    ("Safari", "Safari"),
)
_DEVICE_LABEL_MAX_LENGTH = 64

#: Attestation bodies can pack x5c certificate chains well beyond the 4 KiB
#: device-JSON default, so the passkey register/complete route alone admits
#: bodies up to 32 KiB (#3130); every other device-JSON route stays at 4 KiB.
_ATTESTATION_BODY_MAX_BYTES = 32 * 1024


def _device_label_from_user_agent(ua: str) -> str:
    """Derive a short, non-identifying device label from a User-Agent string.

    Only the first platform and first browser marker win, joined with a
    middle dot; anything unmatched falls back to ``"Unknown device"``.  The
    result is bounded to 64 characters (the same cap ``rename_device``
    applies) and carries no version or detail tokens from the source string.
    """

    platform_label = next(
        (name for name in _DEVICE_PLATFORM_MARKERS if name in ua), ""
    )
    browser_label = next(
        (label for marker, label in _DEVICE_BROWSER_MARKERS if marker in ua), ""
    )
    label = " · ".join(part for part in (platform_label, browser_label) if part)
    return (label or "Unknown device")[:_DEVICE_LABEL_MAX_LENGTH]


async def _read_device_json(request, max_bytes: int = 4096) -> dict:
    """Read one bounded JSON object body for the device-manager endpoints.

    ``max_bytes`` lets a single route widen the bound (the passkey
    register/complete route admits attestation bodies up to
    ``_ATTESTATION_BODY_MAX_BYTES``); the 4 KiB default covers every other
    device-JSON route.
    """

    from aiohttp import web

    if request.content_length is not None and request.content_length > max_bytes:
        raise web.HTTPRequestEntityTooLarge(
            max_size=max_bytes, actual_size=request.content_length
        )
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _required_str(payload: dict, key: str) -> str:
    """Return ``payload[key]`` when it is a str; otherwise fail closed."""

    value = payload.get(key)
    if not isinstance(value, str):
        raise AuthenticationError("authentication denied")
    return value


class ChatbookWebServerMixin:
    """Textual web server with Chatbook-specific viewport resize hardening."""

    def __init__(
        self,
        *args,
        web_auth_policy: WebAuthPolicy | None = None,
        web_ssl_context: ssl.SSLContext | None = None,
        canvas_policy: CanvasConfigPolicy | None = None,
        canvas_profile_snapshot: ProfileSnapshot | None = None,
        session_store: WebSessionStore | None = None,
        session_idle_timeout_seconds: float | None = None,
        session_absolute_timeout_seconds: float | None = None,
        mobile_service: Any | None = None,
        webauthn_service: Any | None = None,
        **kwargs,
    ) -> None:
        """Initialize the served shell and its feature surfaces.

        Args:
            mobile_service: Injected mobile companion service (ADR-234).
                ``None`` (the default) constructs ``MobileChatService``
                lazily on the first ``/api/v1/mobile`` request.
            webauthn_service: Injected WebAuthn ceremony service
                (``Web_Server.webauthn_service.WebAuthnService``, ADR-235).
                ``None`` (the default) means passkeys are unavailable:
                every passkey route answers ``503`` and the login/devices
                pages omit their passkey controls. Tests inject a service
                over a tmp store; ``create_server`` builds the real one
                and passes ``None`` when RP derivation fails.
        """
        super().__init__(*args, **kwargs)
        if web_auth_policy is None:
            web_auth_policy = build_web_auth_policy(
                host=self.host,
                port=self.port,
                access_token=None,
            )
        manager_kwargs: dict[str, object] = {"session_store": session_store}
        if session_idle_timeout_seconds is not None:
            manager_kwargs["idle_timeout_seconds"] = session_idle_timeout_seconds
        if session_absolute_timeout_seconds is not None:
            manager_kwargs[
                "absolute_timeout_seconds"
            ] = session_absolute_timeout_seconds
        self._web_auth = WebAuthManager(web_auth_policy, **manager_kwargs)
        self._web_ssl_context = web_ssl_context
        self._canvas_policy = canvas_policy or get_canvas_config_policy()
        self._canvas_disabled_latched = not self._canvas_policy.enabled
        self._canvas_policy_watch_task: asyncio.Task[None] | None = None
        self._served_browser_children: dict[str, str] = {}
        from .served_console_hosts import ServedConsoleHosts

        self._served_console_hosts = ServedConsoleHosts(self._create_console_host)
        self._canvas_profile_snapshot = (
            canvas_profile_snapshot
            or load_application_profile_snapshot(loader=load_profile_snapshot)
        )
        self._served_canvas_gateway = CanvasGateway(
            authority=_ServedCanvasAuthorityProxy(self),
            profile_snapshot=self._canvas_profile_snapshot,
        )
        self._served_canvas_launches: dict[str, tuple[CanvasGatewayScope, object]] = {}
        self._mobile_service = mobile_service
        self._webauthn_service = webauthn_service

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
            # Mobile sends carry up to 32 KiB of text (ADR-234); every
            # other surface keeps its own tighter per-route body guard.
            client_max_size=MOBILE_MAX_BODY_BYTES,
        )
        aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(self.templates_path))

        routes = [
            web.get("/auth/login", self.handle_login, name="login"),
            web.post("/auth/login", self.handle_login, name="login_submit"),
            web.get(
                "/auth/bootstrap", self.handle_login_bootstrap, name="login_bootstrap"
            ),
            web.get("/auth/devices", self.handle_devices_page, name="devices"),
            web.post("/auth/logout", self.handle_logout, name="logout"),
            web.post(
                "/auth/devices/revoke",
                self.handle_devices_revoke,
                name="devices_revoke",
            ),
            web.post(
                "/auth/devices/revoke-all",
                self.handle_devices_revoke_all,
                name="devices_revoke_all",
            ),
            web.post(
                "/auth/devices/rename",
                self.handle_devices_rename,
                name="devices_rename",
            ),
            web.post(
                "/auth/devices/pair",
                self.handle_devices_pair,
                name="devices_pair",
            ),
            web.post(
                "/auth/passkeys/register/begin",
                self.handle_passkeys_register_begin,
                name="passkeys_register_begin",
            ),
            web.post(
                "/auth/passkeys/register/complete",
                self.handle_passkeys_register_complete,
                name="passkeys_register_complete",
            ),
            web.post(
                "/auth/passkeys/login/begin",
                self.handle_passkeys_login_begin,
                name="passkeys_login_begin",
            ),
            web.post(
                "/auth/passkeys/login/complete",
                self.handle_passkeys_login_complete,
                name="passkeys_login_complete",
            ),
            web.post(
                "/auth/passkeys/remove",
                self.handle_passkeys_remove,
                name="passkeys_remove",
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
        if mobile_surface_enabled():
            # Authed mobile companion routes (ADR-234); registered — like
            # the Canvas routes above — only while the feature gate is on,
            # so a disabled surface 404s instead of authing a dead API.
            # The /m asset route must be spliced in ahead of the catch-all
            # ``/static`` prefix route below (first-registered wins in
            # aiohttp), mirroring how the Canvas runtime asset route
            # inserts itself above — hence insert BEFORE the REST extend
            # moves ``web.static`` off the list tail.
            routes.insert(
                -2,
                web.get(
                    "/static/mobile/{name}",
                    self.handle_mobile_asset,
                    name="mobile_asset",
                ),
            )
            routes.extend(build_mobile_routes(self))
            routes.append(
                web.get("/m", self.handle_mobile_page, name="mobile_page")
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
        max_age = max(1, int(grant.expires_at - time.time()))
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
        return frozenset(
            {
                "/auth/login",
                "/auth/bootstrap",
                # Public origin-checked passkey login ceremonies (ADR-235);
                # registration stays behind the session + CSRF middleware.
                "/auth/passkeys/login/begin",
                "/auth/passkeys/login/complete",
            }
        )

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
            # The mobile event stream upgrades with the same subprotocol
            # challenge as the terminal WS (csrf.<token> + chatbook-v1),
            # but only while its surface is registered — a gate-off
            # stream path must 404 as a dead route, not auth-challenge.
            websocket = request.path == "/ws" or (
                request.path == MOBILE_STREAM_PATH and mobile_surface_enabled()
            )
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
                    grant = self._web_auth.authenticate_local(
                        client_ip=facts.peer_ip,
                        device_label=_device_label_from_user_agent(
                            request.headers.get("User-Agent", "")
                        ),
                        # Spec §3.1: loopback auto-login stays the in-memory
                        # session it always was; it must never mint a disk
                        # record (ADR-233).
                        persist=False,
                    )
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
            # Passkey login is offered only when the ceremony service exists
            # (domain-origin public_url); the button ships hidden and
            # passkeys.js reveals it only in WebAuthn-capable browsers.
            passkey_login = (
                (
                    '<button type="button" class="passkey-login" hidden>'
                    "Use passkey</button>"
                    '<p class="passkey-message" hidden></p>'
                    '<script src="/static/chatbook/passkeys.js" defer></script>'
                )
                if self._webauthn_service is not None
                else ""
            )
            return web.Response(
                text=(
                    '<!doctype html><html><head><meta charset="utf-8">'
                    "<title>Chatbook sign in</title>"
                    "<style>.phone-view-link{display:none}"
                    "@media (max-width:720px){.phone-view-link{display:inline}}"
                    "</style></head><body>"
                    "<main><h1>Chatbook</h1>"
                    '<p class="phone-view-link">On a phone? '
                    '<a href="/m">Phone view</a> after signing in.</p>'
                    '<form method="post" action="/auth/login">'
                    '<label>Access token <input type="password" name="access_token" '
                    'autocomplete="current-password" required></label> '
                    '<label><input type="checkbox" name="remember" value="1" checked> '
                    "Remember this device — checked signs in a persistent "
                    "device session; unchecked uses a short-lived in-memory "
                    "session that ends when the server stops (safest on shared "
                    "machines)</label>"
                    '<button type="submit">Sign in</button></form>'
                    + passkey_login
                    + "</main></body></html>"
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
            # The "remember me" checkbox opts the session out of disk
            # persistence (ADR-233); the device label comes from the UA only.
            remember = data.get("remember") == "1"
            grant = self._web_auth.login_with_access_token(
                str(data.get("access_token", "")),
                client_ip=client_ip,
                device_label=_device_label_from_user_agent(
                    request.headers.get("User-Agent", "")
                ),
                persist=remember,
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
                request.query.get("nonce", ""),
                client_ip=client_ip,
                device_label=_device_label_from_user_agent(
                    request.headers.get("User-Agent", "")
                ),
            )
        except AuthenticationError:
            raise web.HTTPUnauthorized(text="Authentication denied") from None
        response = web.Response(status=303, headers={"Location": "/"})
        self._set_session_cookie(
            response, grant, secure=self._web_auth.policy.secure_cookies
        )
        return response

    async def handle_devices_page(self, request):
        """List every live device session and its manager controls.

        The section contract consumed by ``devices.js`` and the integration
        tests: one ``<section class="device">`` per session carrying
        ``data-session-id`` and a ``data-current-device`` flag for the
        requesting device, with all dynamic values HTML-escaped and the
        per-session CSRF token in the ``chatbook-csrf`` meta tag. A
        header-level "Sign out all devices" incident-response control sits
        outside those sections (ADR-233 decision 4).
        """

        from aiohttp import web

        current = request["chatbook_browser_session"]
        csrf = html.escape(str(request["chatbook_csrf"]), quote=True)
        sections = []
        for session in self._web_auth.sessions_snapshot():
            label = html.escape(
                session.device_label or "Unknown device", quote=True
            )
            session_id = html.escape(session.session_id, quote=True)
            created = datetime.fromtimestamp(session.created_at, UTC).isoformat()
            last_seen = datetime.fromtimestamp(session.last_seen_at, UTC).isoformat()
            is_current = session.session_id == current.session_id
            marker = (
                '<p class="device-current">This device</p>' if is_current else ""
            )
            action = (
                (
                    '<button type="button" class="device-signout">'
                    "Sign out this device</button>"
                )
                if is_current
                else (
                    '<button type="button" class="device-revoke">'
                    "Sign out device</button>"
                )
            )
            sections.append(
                f'<section class="device" data-session-id="{session_id}" '
                f'data-current-device="{str(is_current).lower()}">'
                f"<h2>{label}</h2>{marker}"
                f'<p class="device-times">Added {created} · Last active '
                f"{last_seen}</p>"
                f'<input type="text" maxlength="64" name="label" '
                f'aria-label="Device name" value="{label}"> '
                '<button type="button" class="device-rename">Rename</button> '
                f"{action}</section>"
            )
        # Passkeys (ADR-235): with a ceremony service, the page offers
        # registration ("Add a passkey", revealed by passkeys.js only in
        # WebAuthn-capable browsers) and lists every stored credential with a
        # Remove control keyed by its full public credential id; without one
        # (IP/localhost/missing public_url) it renders the domain-origin
        # notice instead of a dead button.
        if self._webauthn_service is None:
            passkey_section = (
                '<section class="device-passkeys"><h2>Passkeys</h2>'
                '<p class="passkey-unavailable">Passkeys need a domain origin '
                "(`public_url`)</p></section>"
            )
        else:
            items = []
            for entry in self._webauthn_service.list_credentials():
                credential_id = html.escape(
                    str(entry["credential_id_hex"]), quote=True
                )
                item_label = html.escape(
                    str(entry["label"] or "Passkey"), quote=True
                )
                prefix = html.escape(
                    str(entry["credential_id_prefix"]), quote=True
                )
                created = datetime.fromtimestamp(entry["created_at"], UTC).isoformat()
                last_used = datetime.fromtimestamp(
                    entry["last_used_at"], UTC
                ).isoformat()
                items.append(
                    f'<li class="passkey" data-credential-id="{credential_id}">'
                    f"<strong>{item_label}</strong> "
                    f'<span class="passkey-id">{prefix}</span>'
                    f'<p class="passkey-times">Added {created} · Last used '
                    f"{last_used}</p> "
                    '<button type="button" class="passkey-remove">Remove'
                    "</button></li>"
                )
            passkey_section = (
                '<section class="device-passkeys"><h2>Passkeys</h2>'
                '<label>Passkey name <input type="text" maxlength="64" '
                'class="passkey-label" placeholder="e.g. YubiKey 5"></label> '
                '<button type="button" class="passkey-add" hidden>Add a passkey'
                "</button>"
                '<p class="passkey-message" hidden></p>'
                '<ul class="passkey-list">'
                + "".join(items)
                + "</ul>"
                '<script src="/static/chatbook/passkeys.js" defer></script>'
                "</section>"
            )
        body = (
            '<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="chatbook-csrf" content="{csrf}">'
            "<title>Chatbook devices</title></head><body>"
            "<main><h1>Your devices</h1>"
            '<button type="button" class="device-revoke-all">'
            "Sign out all devices</button>"
            '<section class="device-pairing"><h2>Add a device</h2>'
            '<button type="button" class="device-pair">Add a device</button>'
            '<canvas class="device-pair-qr" hidden></canvas>'
            '<p class="device-pair-hint" hidden>Scan with the other device, or '
            "open the link below. A fresh link is fetched automatically every "
            "100&nbsp;seconds (each link works once and expires 2 minutes "
            "after issue).</p>"
            '<p class="device-pair-url"></p></section>'
            + passkey_section
            + "".join(sections)
            + "</main>"
            '<script src="/static/chatbook/qrcodegen.js" defer></script>'
            '<script src="/static/chatbook/devices.js" defer></script>'
            "</body></html>"
        )
        return web.Response(
            text=body,
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def handle_logout(self, request):
        """Revoke the requesting browser's session and delete its cookie."""

        from aiohttp import web

        self._web_auth.revoke(request.cookies.get(SESSION_COOKIE_NAME, ""))
        response = web.HTTPSeeOther(location="/auth/login")
        response.del_cookie(SESSION_COOKIE_NAME, path="/")
        return response

    async def handle_devices_revoke(self, request):
        """Revoke one listed device session by its public identifier."""

        from aiohttp import web

        payload = await _read_device_json(request)
        session_id = payload.get("session_id")
        revoked = (
            self._web_auth.revoke_by_session_id(session_id)
            if isinstance(session_id, str)
            else False
        )
        return web.json_response(
            {"ok": revoked}, headers={"Cache-Control": "no-store"}
        )

    async def handle_devices_revoke_all(self, request):
        """Revoke every device session, including the requesting browser's.

        The incident-response control (ADR-233 decision 4): one authed POST
        empties the session table and persists that emptiness, so even a
        restart cannot resurrect a revoked device. The requesting session
        dies with the rest, so the browser is expected to navigate away
        rather than reload.
        """

        from aiohttp import web

        await _read_device_json(request)
        self._web_auth.revoke_all()
        return web.json_response(
            {"ok": True}, headers={"Cache-Control": "no-store"}
        )

    async def handle_devices_rename(self, request):
        """Rename one listed device session (bounded, trimmed, persisted)."""

        from aiohttp import web

        payload = await _read_device_json(request)
        session_id = payload.get("session_id")
        label = payload.get("label")
        renamed = (
            self._web_auth.rename_device(session_id, label)
            if isinstance(session_id, str) and isinstance(label, str)
            else False
        )
        return web.json_response(
            {"ok": renamed}, headers={"Cache-Control": "no-store"}
        )

    async def handle_devices_pair(self, request):
        """Issue a one-use pairing URL a new device can scan (ADR-233).

        The route is an authenticated POST, so the shared middleware has
        already enforced the CSRF header; the body is read only to keep the
        bounded-shape guard consistent with the sibling device endpoints.
        The returned URL embeds a single-use nonce that expires with the
        manager's bootstrap TTL (120 s), so the response is never cacheable.
        """

        from aiohttp import web

        await _read_device_json(request)
        return web.json_response(
            {"url": self.issue_browser_bootstrap_url()},
            headers={"Cache-Control": "no-store"},
        )

    def issue_browser_bootstrap_url(self) -> str:
        """Return a one-time bootstrap URL without exposing the configured token."""

        nonce = self._web_auth.issue_bootstrap()
        return f"{self.public_url.rstrip('/')}/auth/bootstrap?nonce={quote(nonce)}"

    # --- Passkey routes (ADR-235) ------------------------------------------
    #
    # Availability is SERVICE PRESENCE, not a property on it: a service
    # exists only when RP derivation succeeded (domain public_url), so
    # ``self._webauthn_service is None`` is the single gate — every handler
    # answers ``503 {"error": "passkeys_unavailable"}`` and both pages omit
    # their passkey controls. Registration routes sit behind the session +
    # CSRF middleware; the two login routes are public origin-checked POSTs
    # (``_public_auth_paths``). All responses are uncacheable: every one
    # embeds a one-use challenge or a session outcome.

    def _passkeys_unavailable(self):
        from aiohttp import web

        return web.json_response(
            {"error": "passkeys_unavailable"},
            status=503,
            headers={"Cache-Control": "no-store"},
        )

    async def handle_passkeys_register_begin(self, request):
        """Start a passkey registration ceremony for the signed-in user."""

        from aiohttp import web

        if self._webauthn_service is None:
            return self._passkeys_unavailable()
        payload = await _read_device_json(request)
        label = payload.get("label")
        label = (
            label.strip()[:_DEVICE_LABEL_MAX_LENGTH]
            if isinstance(label, str)
            else ""
        )
        options = self._webauthn_service.register_begin(label=label or "Passkey")
        return web.json_response(options, headers={"Cache-Control": "no-store"})

    async def handle_passkeys_register_complete(self, request):
        """Verify a registration response and persist the new credential."""

        from aiohttp import web

        if self._webauthn_service is None:
            return self._passkeys_unavailable()
        # Attestation objects may carry packed x5c chains, so this route alone
        # widens the device-JSON body bound to 32 KiB (#3130).
        payload = await _read_device_json(
            request, max_bytes=_ATTESTATION_BODY_MAX_BYTES
        )
        try:
            result = self._webauthn_service.register_complete(
                _required_str(payload, "challenge_id"),
                _required_str(payload, "client_data_json"),
                _required_str(payload, "attestation_object_b64"),
            )
        except AuthenticationError:
            return web.json_response(
                {"error": "authentication_denied"},
                status=401,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(
            {"ok": True, **result}, headers={"Cache-Control": "no-store"}
        )

    async def handle_passkeys_login_begin(self, request):
        """Start a discoverable-credential login ceremony (public route)."""

        from aiohttp import web

        if self._webauthn_service is None:
            return self._passkeys_unavailable()
        return web.json_response(
            self._webauthn_service.login_begin(),
            headers={"Cache-Control": "no-store"},
        )

    async def handle_passkeys_login_complete(self, request):
        """Verify an assertion and mint a persistent device session (ADR-235).

        The middleware has already origin-checked this public POST; the
        handler re-validates for the effective client IP exactly like the
        bootstrap handler, then mints the session through
        ``login_device`` — the ``login_with_access_token``-equivalent with
        no token check — and sets the same cookie the bootstrap flow uses.
        Any ``AuthenticationError`` (ceremony failure or capacity
        exhaustion) is a plain 401 with no ceremony detail.
        """

        from aiohttp import web

        if self._webauthn_service is None:
            return self._passkeys_unavailable()
        facts = self._request_facts(request)
        payload = await _read_device_json(request)
        try:
            client_ip, _scheme, _host = self._web_auth.validate_public_request(
                facts, require_origin=True
            )
            self._webauthn_service.login_complete(
                _required_str(payload, "challenge_id"),
                _required_str(payload, "client_data_json"),
                _required_str(payload, "authenticator_data_b64"),
                _required_str(payload, "signature_b64"),
            )
            grant = self._web_auth.login_device(
                client_ip=client_ip,
                device_label=_device_label_from_user_agent(
                    request.headers.get("User-Agent", "")
                )
                + " (passkey)",
            )
        except AuthenticationError:
            raise web.HTTPUnauthorized(text="Authentication denied") from None
        response = web.Response(status=303, headers={"Location": "/"})
        self._set_session_cookie(
            response, grant, secure=self._web_auth.policy.secure_cookies
        )
        return response

    async def handle_passkeys_remove(self, request):
        """Remove one stored passkey by its full public credential id."""

        from aiohttp import web

        if self._webauthn_service is None:
            return self._passkeys_unavailable()
        payload = await _read_device_json(request)
        credential_id = payload.get("credential_id")
        removed = (
            self._webauthn_service.remove_credential(credential_id)
            if isinstance(credential_id, str)
            else False
        )
        return web.json_response(
            {"ok": removed}, headers={"Cache-Control": "no-store"}
        )

    def _canonical_console_selection(self):
        """Resolve only admitted server configuration; browsers select no storage."""
        from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
            CanonicalConsoleStores,
        )
        from tldw_chatbook.Backup_Recovery.profile_paths import (
            canonical_console_paths,
            database_path,
            legacy_mobile_dir,
        )
        from tldw_chatbook.config import get_atomic_config_snapshot

        settings = get_atomic_config_snapshot().values
        return (CanonicalConsoleStores(*canonical_console_paths(settings)),
                database_path(settings, "media_db_path"), legacy_mobile_dir(settings))

    def _get_mobile_service(self) -> Any:
        """Return the canonical owner adapter without starting an execution host."""
        if self._mobile_service is None:
            from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
                require_mobile_cutover,
            )

            from .mobile_service import CanonicalMobileReads, MobileChatService

            stores, media_path, source_dir = self._canonical_console_selection()
            def require_cutover():
                from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
                from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
                    MobileConsolidationRefused,
                )
                from tldw_chatbook.Backup_Recovery.service_storage import (
                    default_control_root,
                )
                try:
                    require_mobile_cutover(source_dir, stores)
                except (ValueError, RecoveryRequired) as error:
                    raise MobileConsolidationRefused(str(error), control_root=default_control_root()) from error

            self._mobile_service = MobileChatService(
                hosts=self._served_console_hosts, storage_identity=stores.identity,
                broker=self._canvas_control_broker,
                readers=CanonicalMobileReads(stores.chat_path, media_path, stores.run_path),
                require_cutover=require_cutover,
            )
        return self._mobile_service

    async def on_startup(self, app) -> None:
        """Start the private loopback control broker before children spawn."""

        self._canvas_control_broker = CanvasControlBroker(
            runtime_snapshot_id=runtime_snapshot_id(self._canvas_profile_snapshot)
        )
        await self._canvas_control_broker.start()
        self._canvas_policy_watch_task = asyncio.create_task(
            self._watch_canvas_policy(), name="chatbook-served-canvas-policy"
        )
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
        hosts = getattr(self, "_served_console_hosts", None)
        retirement_error = None
        if hosts is not None:
            try:
                await hosts.aclose()
            except Exception as exc:  # noqa: BLE001 - finish teardown, then re-raise
                retirement_error = exc
        broker = getattr(self, "_canvas_control_broker", None)
        if broker is not None:
            await broker.aclose()
            self._canvas_control_broker = None
        await self._served_canvas_gateway.aclose()
        self._served_canvas_launches.clear()
        self._served_browser_children.clear()
        mobile_service = getattr(self, "_mobile_service", None)
        if mobile_service is not None:
            try:
                await mobile_service.shutdown()
            except Exception:  # noqa: BLE001 - teardown must finish regardless
                logger.exception("mobile service shutdown failed")
            self._mobile_service = None
        self._web_auth.revoke_all()
        await super().on_shutdown(app)
        if retirement_error is not None:
            raise retirement_error

    def _canvas_enabled(self) -> bool:
        """Read the global kill switch while honoring the restart latch."""

        if self._canvas_disabled_latched:
            return False
        try:
            return get_canvas_execution_enabled() is True
        except Exception:  # noqa: BLE001 - server feature gate fails closed
            return False

    async def _watch_canvas_policy(self, *, interval_seconds: float = 0.25) -> None:
        """Revoke every served child promptly after the shared switch turns off."""

        while self._canvas_enabled():
            await asyncio.sleep(interval_seconds)
        await self._disable_canvas_runtime()

    async def _disable_canvas_runtime(self) -> None:
        """Revoke Canvas capabilities while retaining Console runtime control."""

        if self._canvas_disabled_latched:
            return
        self._canvas_disabled_latched = True
        await self._served_canvas_gateway.aclose()
        self._served_canvas_launches.clear()

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

    async def served_canvas_state(
        self,
        browser_session_id: str,
        *,
        expected_scope: CanvasGatewayScope | None = None,
        expected_child_id: str | None = None,
    ) -> dict[str, object]:
        """Return only the Canvas state owned by one exact authenticated child."""

        if not self._canvas_enabled():
            await self._disable_canvas_runtime()
            raise ServedCanvasUnavailable("canvas_disabled")
        child_id = self._served_browser_children.get(browser_session_id)
        broker = getattr(self, "_canvas_control_broker", None)
        if child_id is None or broker is None:
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        captured_launch = self._served_canvas_launches.get(browser_session_id)
        if expected_scope is not None and (
            child_id != expected_child_id
            or captured_launch is None
            or captured_launch[0] != expected_scope
            or not self._served_canvas_gateway.has_shell_binding(browser_session_id)
        ):
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
                    "selection_generation": payload["selection_generation"],
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
        if expected_scope is not None and (
            self._served_browser_children.get(browser_session_id) != expected_child_id
            or self._canvas_control_broker is not broker
            or self._served_canvas_launches.get(browser_session_id)
            is not captured_launch
            or not self._served_canvas_gateway.has_shell_binding(browser_session_id)
            or state.get("conversation_session_id")
            != expected_scope.conversation_session_id
        ):
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
        generation = result.get("selection_generation")
        if not isinstance(generation, str) or not generation:
            raise ServedCanvasUnavailable("canvas_session_unavailable")
        scope = CanvasGatewayScope(
            browser_session_id=browser_session_id,
            conversation_session_id=str(
                result.get("conversation_session_id", browser_session_id)
            ),
            canvas_id=canvas_id,
            revision_id=revision_id,
            selection_generation=result["selection_generation"],
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
                    browser_session_id=browser_session_id,
                    scope=scope,
                    synchronize_only=True,
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

    def _create_console_host(self, _storage_identity: str):
        from .served_console_hosts import ServedConsoleHost

        service_class = self._chatbook_app_service_class
        if service_class is None:
            raise RuntimeError("served terminal app service is unavailable")
        return ServedConsoleHost(
            lambda **callbacks: service_class(
                self.command,
                **callbacks,
                download_manager=self.download_manager,
                debug=self.debug,
                canvas_control_broker=self._canvas_control_broker,
            )
        )

    async def _process_attached_terminal(self, request, websocket, attachment) -> None:
        """Keep socket writes outside the child's continuously draining callback."""
        from .served_console_hosts import StaleTerminalAttachment

        session = request.get("chatbook_browser_session")

        async def output() -> None:
            while True:
                kind, data = await attachment.receive()
                if kind == "bytes":
                    await websocket.send_bytes(data)
                else:
                    await websocket.send_str(data)

        async def input_messages() -> None:
            from aiohttp import WSMsgType

            async for message in websocket:
                if message.type != WSMsgType.TEXT:
                    continue
                if session.revoked:
                    return
                self._web_auth.touch_session(session)
                envelope = message.json()
                if not isinstance(envelope, list) or not 1 <= len(envelope) <= 2:
                    continue
                kind = envelope[0]
                data = envelope[1] if len(envelope) == 2 else None
                try:
                    if kind == "stdin" and isinstance(data, str):
                        await attachment.send_bytes(data.encode("utf-8"))
                    elif kind == "resize" and isinstance(data, dict):
                        await attachment.resize(
                            _web_dimension(data.get("width"), 80),
                            _web_dimension(data.get("height"), 24),
                        )
                    elif kind == "ping":
                        await websocket.send_json(["pong", data])
                    elif kind in {"blur", "focus"}:
                        await attachment.send_meta({"type": kind})
                except StaleTerminalAttachment:
                    if attachment.closed.is_set():
                        return

        tasks = [
            asyncio.create_task(output()),
            asyncio.create_task(input_messages()),
            asyncio.create_task(attachment.closed.wait()),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await websocket.close()

    async def handle_websocket(self, request):
        """Attach one authenticated terminal to the retained profile child."""
        from aiohttp import web

        websocket = web.WebSocketResponse(heartbeat=15, protocols=(WEBSOCKET_PROTOCOL,))
        session = request.get("chatbook_browser_session")
        if session is None:
            raise web.HTTPUnauthorized()
        attachment = None
        host = None
        expiry_task = None
        unregister_channel = lambda: None
        try:
            await websocket.prepare(request)
            unregister_channel = self._web_auth.register_channel(
                session, lambda: asyncio.create_task(websocket.close(code=4001))
            )
            expiry_task = asyncio.create_task(self._expire_websocket_session(session))
            # Server-admitted configuration only; browser parameters never select storage.
            stores, _media_path, _source_dir = self._canonical_console_selection()
            storage_identity = stores.identity
            host = await self._served_console_hosts.get_or_start(storage_identity)
            if session.revoked or websocket.closed:
                return websocket
            attachment = await host.attach_terminal(
                session.session_id,
                _web_dimension(request.query.get("width"), 80),
                _web_dimension(request.query.get("height"), 24),
            )
            if session.revoked or websocket.closed:
                await attachment.detach()
                return websocket
            for browser_id, child_id in tuple(self._served_browser_children.items()):
                if child_id == host.service.app_service_id:
                    self.unbind_served_browser(browser_id, child_id)
            self.bind_served_browser(session.session_id, host.service.app_service_id)
            await self._process_attached_terminal(request, websocket, attachment)
        except asyncio.CancelledError:
            await websocket.close()
            raise
        except Exception as error:  # noqa: BLE001 - bounded websocket failure, no spawn secrets
            logger.error("Served terminal session failed type={}", type(error).__name__)
        finally:
            if expiry_task is not None:
                expiry_task.cancel()
                await asyncio.gather(expiry_task, return_exceptions=True)
            unregister_channel()
            if attachment is not None:
                if (
                    host.current_attachment is attachment
                    or host.current_attachment is None
                ):
                    self.unbind_served_browser(
                        session.session_id, host.service.app_service_id
                    )
                await attachment.detach()
            await websocket.close()
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
            .replace("__TEXTUAL_JS_VERSION__", self._textual_js_version())
        )
        return web.Response(
            text=body,
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-store"},
        )

    async def handle_served_shell_asset(self, request):
        """Serve only the immutable, state-free owned shell assets."""

        from aiohttp import web

        asset = _SERVED_SHELL_ASSETS.get(request.match_info.get("name", ""))
        if asset is None:
            raise web.HTTPNotFound(text="Asset not found")
        content_type, filename = asset
        return web.Response(
            body=_SERVED_STATIC_ROOT.joinpath(filename).read_bytes(),
            content_type=content_type,
            headers={"Cache-Control": _STATIC_ASSET_CACHE_CONTROL},
        )

    async def handle_mobile_page(self, request):
        """Serve the phone-first companion page (ADR-234 Phase 1b).

        Authed HTML in the devices-page idiom: the per-session CSRF token
        is embedded in the ``chatbook-csrf`` meta tag (HTML-escaped) for
        ``app.js`` fetches and the WebSocket ``csrf.`` subprotocol. The
        shell is read from ``static/mobile/index.html`` and never cached,
        because the substitution makes it session-specific.
        """

        from aiohttp import web

        csrf = html.escape(str(request["chatbook_csrf"]), quote=True)
        body = _MOBILE_PAGE_HTML.replace("__CHATBOOK_CSRF__", csrf)
        return web.Response(
            text=body,
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-store"},
        )

    async def handle_mobile_asset(self, request):
        """Serve only the mobile page's state-free owned assets."""

        from aiohttp import web

        asset = _MOBILE_STATIC_ASSETS.get(request.match_info.get("name", ""))
        if asset is None:
            raise web.HTTPNotFound(text="Asset not found")
        content_type, filename, cache_control = asset
        return web.Response(
            body=_MOBILE_STATIC_ROOT.joinpath(filename).read_bytes(),
            content_type=content_type,
            headers={"Cache-Control": cache_control},
        )

    async def handle_chatbook_auth_js(self, request):
        """Serve the owned, state-free browser authentication bootstrap."""
        from aiohttp import web

        return web.Response(
            text=_CHATBOOK_AUTH_BOOTSTRAP_JS,
            content_type="application/javascript",
            headers={"Cache-Control": _STATIC_ASSET_CACHE_CONTROL},
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

    def _textual_js_version(self) -> str:
        """Fingerprint the patched bundle so its cache key follows content.

        The bundle URL is unversioned and served with a one-hour public
        cache policy (task-33130, Qodo #6 on PR #2856): without a
        fingerprint, a browser holding a fresh cached copy would keep
        running the PREVIOUS patch -- including a bundle whose upstream
        hook shape no longer matches, which fails closed unpatched and
        resurrects the resize feedback loop -- until the hour expires.
        Content-addressing the URL makes the long cache safe: a changed
        bundle is a different URL.
        """
        import hashlib

        try:
            source = self._patched_textual_js()
        except OSError:
            return "unavailable"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    async def handle_textual_js(self, request):
        """Serve textual-serve JS with a full repaint after browser viewport resize.

        The ~514KB bundle is deterministic for a given installed
        textual-serve version, so it gets the same one-hour public cache
        policy as the immutable shell assets instead of re-downloading on
        every page load.

        Args:
            request: The aiohttp request (unused; the response depends only
                on the installed bundle).

        Returns:
            A successful ``application/javascript`` response carrying the
            patched bundle with the shared static-asset cache policy.
        """
        from aiohttp import web

        return web.Response(
            text=self._patched_textual_js(),
            content_type="application/javascript",
            headers={"Cache-Control": _STATIC_ASSET_CACHE_CONTROL},
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
    """Adapt textual-serve child metadata, environment and physical retirement."""

    class ChatbookAppService(textual_app_service_class):
        def __init__(self, *args, canvas_control_broker, **kwargs):
            super().__init__(*args, **kwargs)
            self._canvas_control_broker = canvas_control_broker
            self._canvas_control_environment: dict[str, str] = {}
            self._canvas_control_started = False

        def _build_environment(self, width: int = 80, height: int = 24):
            environment = super()._build_environment(width, height)
            environment.update(self._canvas_control_environment)
            # Launch identity survives disabled or unavailable Canvas transport.
            environment["CHATBOOK_SERVED_CHILD"] = "1"
            environment["TEXTUAL_DRIVER"] = (
                "tldw_chatbook.Web_Server.console_web_driver:ConsoleWebDriver"
            )
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

        async def on_meta(self, data: bytes) -> None:
            metadata = json.loads(data)
            if metadata.get("type") in {
                "console_attachment_begin",
                "console_attachment_ready",
            }:
                await self.remote_write_str(json.dumps([metadata["type"], metadata]))
                return
            await super().on_meta(data)

        async def run(self) -> None:
            try:
                await super().run()
            finally:
                await self.remote_close()

        async def _open_app_process(self, width: int = 80, height: int = 24):
            self._process = process = await asyncio.create_subprocess_shell(
                self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_environment(width, height),
                start_new_session=True,
            )
            self._stdin = process.stdin
            return process

        async def stop(self) -> None:
            retirement = getattr(self, "_retirement_task", None)
            if retirement is None:
                retirement = self._retirement_task = asyncio.create_task(self._retire())
            await asyncio.shield(retirement)

        async def _retire(self) -> None:
            stopping = asyncio.create_task(super().stop())
            process = getattr(self, "_process", None)
            try:
                await asyncio.wait_for(asyncio.shield(stopping), 5)
                if process is not None:
                    await asyncio.wait_for(process.wait(), 2)
            except TimeoutError:
                if process is None:
                    raise
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    if process.returncode is not None:
                        break
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(process.wait(), 2)
                        break
                    except TimeoutError:
                        if sig == signal.SIGKILL:
                            raise
            finally:
                # Only physical exit permits cancelling a lagging pipe observer.
                # Otherwise keep draining; the live child still owns its lease.
                self._stop_observer = stopping
                if not stopping.done() and (
                    process is None or process.returncode is not None
                ):
                    stopping.cancel()
                    await asyncio.gather(stopping, return_exceptions=True)
            if self._canvas_control_started:
                self._canvas_control_started = False
                if self._canvas_control_broker is not None:
                    await self._canvas_control_broker.revoke_child(self.app_service_id)
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

    # Passkeys (ADR-235): the ceremony service exists only when the finalized
    # public_url anchors a registrable-domain RP. An IP literal, localhost, or
    # a missing public_url cannot, so the server degrades to token-only login
    # (service None -> 503 routes, no UI controls) instead of failing to boot.
    # python-fido2 rides the optional [web] extra and webauthn_service.py
    # imports it at module top, so a stale install that upgraded code without
    # reinstalling must degrade the same way — never kill served boot.
    webauthn_service = None
    try:
        from .webauthn_service import PasskeysUnavailable, WebAuthnService
        from .webauthn_store import WebAuthnCredentialStore

        webauthn_service = WebAuthnService(
            public_url=public_url,
            store=WebAuthnCredentialStore(
                get_user_data_dir() / "webauthn_credentials.json"
            ),
        )
    except ImportError:
        logger.warning(
            "Passkeys disabled: python-fido2 is missing; reinstall the app with "
            'the [web] extra (e.g. pip install -e ".[web]") to enable passkeys'
        )
    except PasskeysUnavailable:
        webauthn_service = None

    # A bad day value must name its key (mirroring the absolute<idle message
    # below), not surface as a bare int() parse error (#3127 review minor).
    try:
        session_idle_timeout_days = max(
            1,
            int(get_cli_setting("web_server", "session_idle_timeout_days", default=30)),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "web_server.session_idle_timeout_days must be an integer number of days"
        ) from error
    try:
        session_absolute_timeout_days = max(
            1,
            int(
                get_cli_setting(
                    "web_server", "session_absolute_timeout_days", default=180
                )
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "web_server.session_absolute_timeout_days must be an integer number of days"
        ) from error
    if session_absolute_timeout_days < session_idle_timeout_days:
        raise ValueError(
            "web_server.session_absolute_timeout_days must be greater than or "
            "equal to web_server.session_idle_timeout_days"
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
        canvas_policy=get_canvas_config_policy(web_auth_policy=web_auth_policy),
        session_store=WebSessionStore(get_user_data_dir() / "web_sessions.json"),
        session_idle_timeout_seconds=session_idle_timeout_days * 86400,
        session_absolute_timeout_seconds=session_absolute_timeout_days * 86400,
        webauthn_service=webauthn_service,
    )

    return server


#: Bind hosts a QR can never reach: a URL carrying one of these as its host
#: (e.g. the auto-derived ``https://<bind-host>:<port>`` when TLS is set
#: without an explicit public_url) is unusable from any other device.
_PAIR_QR_UNREACHABLE_HOSTS = frozenset({"0.0.0.0", "::", ""})


def _print_pair_qr(server, *, echo=print) -> bool:
    """Print a one-use pairing QR for a server about to start (ADR-233 §3.3).

    A one-line "server starting" hint precedes the QR: it prints before the
    server binds, and a phone scanning during startup gets a dead link until
    the listener is up (#3127 minor 8).
    """
    public_url = getattr(server, "public_url", None)
    if not public_url:
        echo("--pair skipped: [web_server] public_url is not configured.")
        return False
    url_host = urlsplit(public_url).hostname
    if (url_host or "").lower() in _PAIR_QR_UNREACHABLE_HOSTS:
        # create_server auto-derives https://<bind-host>:<port> under TLS, so
        # a wildcard bind (0.0.0.0, ::) yields a URL no phone could open;
        # printing it as a QR would advertise a dead link.
        echo(
            "--pair skipped: [web_server] public_url "
            f"{public_url!r} has a wildcard/unspecified host; "
            "configure an explicit public_url to pair."
        )
        return False
    url = server.issue_browser_bootstrap_url()
    # --pair prints BEFORE the server binds (run_web_server calls this
    # ahead of serve()), so a phone scanning immediately hits a dead link
    # until startup finishes — say so before the QR (#3127 minor 8).
    echo("Server starting — scan once it is up.")
    try:
        import qrcode
    except ImportError:
        echo(f"Pair with this one-use link (install 'qrcode' for a QR): {url}")
        return True
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    echo("Scan with your phone to sign in (valid once, for 120 seconds):")
    qr.print_ascii(invert=True)
    echo(url)
    return True


def run_web_server(
    host: str | None = None,
    port: int | None = None,
    title: str | None = None,
    debug: bool | None = None,
    pair: bool = False,
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
        pair: Print a one-use phone pairing QR before serving
            (default: False; skipped with a notice when public_url is unset
            or its host is a wildcard/unspecified bind address)

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

    if pair:
        _print_pair_qr(server)

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
    parser.add_argument(
        "--pair",
        action="store_true",
        help="Print a QR to pair a phone before serving",
    )

    args = parser.parse_args()

    # Run the web server with provided arguments
    run_web_server(
        host=args.host,
        port=args.port,
        title=args.title,
        debug=args.debug,
        pair=args.pair,
    )


if __name__ == "__main__":
    main()
