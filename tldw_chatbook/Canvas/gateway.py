"""Trusted native loopback gateway for Canvas browser delivery.

The Chatbook process remains the only conversation and branch authority.  This
module brokers one already-selected Canvas revision to one browser session; no
HTTP request may supply or enumerate conversation authority.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import inspect
import ipaddress
import json
import math
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias
from urllib.parse import quote
from uuid import uuid4

from aiohttp import web

from .capabilities import (
    CanvasCapabilityAction,
    CanvasCapabilityError,
    CanvasCapabilityScope,
    CanvasCapabilityStore,
)
from .limits import (
    CanvasLimitError,
    CanvasLimits,
    sha256_utf8,
    validate_opaque_identifier,
    validate_utf8_text,
)
from .models import CanvasBridgeRequest, CanvasRenderPlan, RenderNode
from .runtime_assets import CanvasRuntimeAssets, load_canvas_runtime_assets

BridgeConfirmationStatus: TypeAlias = Literal["confirmed", "cancelled", "refused"]
CanvasGatewayEventKind: TypeAlias = Literal[
    "updated", "selection_changed", "disconnected", "discarded"
]

_SESSION_COOKIE = "canvas_session"
_FRAME_COOKIE = "canvas_frame"
_PLAN_COOKIE = "canvas_plan"
_BOOT_TTL_SECONDS = 30.0
_FRAME_TTL_SECONDS = 20.0
_ACTION_TTL_SECONDS = 30.0
_BROWSER_SESSION_TTL_SECONDS = 30 * 60.0
_BRIDGE_SETTLEMENT_TTL_SECONDS = 300.0
_MAX_BRIDGE_SETTLEMENTS = 64
_MAX_BRIDGE_WAITERS = 16
_MAX_BROWSER_SESSIONS = 64
_MAX_SHELL_BINDINGS = 64
_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "access_token",
        "boot",
        "bootstrap",
        "canvas_frame",
        "canvas_plan",
        "canvas_session",
        "capability",
        "csrf",
        "secret",
        "token",
    }
)
_OTHER_PROXY_HEADERS = frozenset({"x-real-ip", "x-original-host"})
_EVENT_METADATA_FIELDS = frozenset(
    {
        "title",
        "sequence",
        "source_bytes",
        "content_sha256",
        "temporary",
        "origin_message_id",
        "origin_turn_id",
        "notice",
    }
)
_SHELL_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "frame-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)
_RENDERER_CSP = (
    "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; worker-src data:; "
    "style-src 'unsafe-inline'; img-src blob:; connect-src 'none'; font-src 'none'; "
    "media-src 'none'; object-src 'none'; frame-src 'none'; child-src 'none'; "
    "form-action 'none'; base-uri 'none'; manifest-src 'none'; "
    "frame-ancestors 'self'; sandbox allow-scripts"
)
_STATIC_ROOT = files("tldw_chatbook.Canvas").joinpath("static")
_SHELL_HTML = _STATIC_ROOT.joinpath("canvas_shell.html").read_bytes()
_SHELL_ASSETS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "canvas_shell.css": ("text/css", "utf-8"),
        "canvas_shell.js": ("text/javascript", "utf-8"),
    }
)


@dataclass(frozen=True, slots=True)
class CanvasGatewayScope:
    """Server-owned browser projection of one exact selected revision."""

    browser_session_id: str
    conversation_session_id: str
    canvas_id: str
    revision_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "browser_session_id",
            "conversation_session_id",
            "canvas_id",
            "revision_id",
        ):
            validate_opaque_identifier(
                getattr(self, field_name), field_name=field_name.replace("_", " ")
            )


@dataclass(frozen=True, slots=True)
class CanvasGatewayEvent:
    """Bounded source-free event delivered to one exact shell session."""

    event_id: str
    kind: CanvasGatewayEventKind
    canvas_id: str
    revision_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_opaque_identifier(self.event_id, field_name="Canvas event ID")
        validate_opaque_identifier(self.canvas_id, field_name="Canvas ID")
        validate_opaque_identifier(self.revision_id, field_name="revision ID")
        if self.kind not in {
            "updated",
            "selection_changed",
            "disconnected",
            "discarded",
        }:
            raise ValueError("unsupported Canvas gateway event kind")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Canvas event metadata must be an object")
        unknown = set(self.metadata) - _EVENT_METADATA_FIELDS
        if unknown:
            raise ValueError("unsupported Canvas event metadata field")
        copied = dict(self.metadata)
        for key, value in copied.items():
            if not isinstance(value, (str, int, bool)) or isinstance(value, float):
                raise TypeError("Canvas event metadata values must be scalar")
            if isinstance(value, str):
                validate_utf8_text(
                    value, limit=4 * 1024, field_name=f"Canvas event {key}"
                )
            if key in {"sequence", "source_bytes"} and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"Canvas event {key} must be a non-negative integer")
            if key == "temporary" and not isinstance(value, bool):
                raise TypeError("Canvas event temporary must be a boolean")
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        validate_utf8_text(encoded, limit=16 * 1024, field_name="Canvas event metadata")
        object.__setattr__(self, "metadata", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class CanvasSourceResponse:
    """Exact inert source returned by the application authority."""

    source: str = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        validate_utf8_text(
            self.source, limit=CanvasLimits().html_bytes, field_name="Canvas source"
        )
        if (
            not isinstance(self.content_sha256, str)
            or len(self.content_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.content_sha256
            )
        ):
            raise ValueError("invalid Canvas source digest")


@dataclass(frozen=True, slots=True)
class CanvasGatewayOption:
    """One reachable source-free Canvas selector option."""

    canvas_id: str
    revision_id: str
    title: str


@dataclass(frozen=True, slots=True)
class CanvasGatewayProjection:
    """Server-owned displayed revision and reachable selector state."""

    scope: CanvasGatewayScope
    options: tuple[CanvasGatewayOption, ...]
    title: str
    sequence: int
    parent_revision_id: str | None
    source_bytes: int
    content_sha256: str
    origin_message_id: str
    origin_turn_id: str
    temporary: bool
    following: bool


@dataclass(frozen=True, slots=True)
class CanvasGatewayNavigation:
    """Atomic authority response for one browser navigation mutation."""

    scope: CanvasGatewayScope
    projection: CanvasGatewayProjection


@dataclass(frozen=True, slots=True)
class BridgeConfirmationRequest:
    """Closed trusted-shell decision for one validated bridge request."""

    approved: bool
    request: CanvasBridgeRequest

    @classmethod
    def from_wire(cls, value: object) -> BridgeConfirmationRequest:
        if not isinstance(value, Mapping) or set(value) != {"approved", "request"}:
            raise ValueError("invalid bridge confirmation request")
        approved = value["approved"]
        if not isinstance(approved, bool):
            raise TypeError("bridge confirmation approval must be a boolean")
        return cls(
            approved=approved,
            request=CanvasBridgeRequest.from_wire(value["request"]),
        )


@dataclass(frozen=True, slots=True)
class BridgeConfirmationResponse:
    """Bounded result of revalidation by the Chatbook process."""

    request_id: str
    status: BridgeConfirmationStatus

    def __post_init__(self) -> None:
        validate_opaque_identifier(self.request_id, field_name="bridge request ID")
        if self.status not in {"confirmed", "cancelled", "refused"}:
            raise ValueError("unsupported bridge confirmation status")


class CanvasGatewayAuthority(Protocol):
    """Narrow callback seam into the application-owned Canvas authority."""

    def resolve_render_plan(
        self, scope: CanvasGatewayScope
    ) -> CanvasRenderPlan | Awaitable[CanvasRenderPlan]: ...

    def read_source(
        self, scope: CanvasGatewayScope
    ) -> CanvasSourceResponse | Awaitable[CanvasSourceResponse]: ...

    def describe_selection(
        self, scope: CanvasGatewayScope
    ) -> CanvasGatewayProjection | Awaitable[CanvasGatewayProjection]: ...

    def navigate(
        self,
        scope: CanvasGatewayScope,
        *,
        action: str,
        canvas_id: str | None = None,
        title: str | None = None,
    ) -> CanvasGatewayNavigation | Awaitable[CanvasGatewayNavigation]: ...

    def read_events(
        self, scope: CanvasGatewayScope, *, after_event_id: str | None
    ) -> tuple[CanvasGatewayEvent, ...] | Awaitable[tuple[CanvasGatewayEvent, ...]]: ...

    def confirm_bridge(
        self,
        scope: CanvasGatewayScope,
        request: BridgeConfirmationRequest,
        *,
        settlement: CanvasBridgeSettlementLease,
    ) -> BridgeConfirmationResponse | Awaitable[BridgeConfirmationResponse]: ...


@dataclass(frozen=True, slots=True, repr=False)
class CanvasGatewayLaunch:
    """Browser-open result; its one-time fragment is deliberately redacted."""

    clean_url: str
    browser_url: str
    opened: bool | None
    error_code: str | None

    def __repr__(self) -> str:
        return (
            "CanvasGatewayLaunch(browser_url=<redacted>, "
            f"clean_url={self.clean_url!r}, opened={self.opened!r}, "
            f"error_code={self.error_code!r})"
        )


@dataclass(slots=True, repr=False)
class _BridgeSettlementRecord:
    browser_session_id: str
    load_id: str
    selection_epoch: int
    request_id: str
    request_kind: str
    payload_digest: bytes
    expires_at: float
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    response: BridgeConfirmationResponse | None = None
    waiter_count: int = 0
    expiry_handle: asyncio.TimerHandle | None = None
    terminal_reason: str | None = None


@dataclass(slots=True, repr=False)
class _BrowserSession:
    digest: bytes
    csrf_digest: bytes
    scope: CanvasGatewayScope
    expires_at: float
    shell_incarnation_id: str
    selection_epoch: int = 0
    current_load_id: str | None = None
    bridge_settlements: dict[str, _BridgeSettlementRecord] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ShellBinding:
    browser_session_id: str
    expires_at: float


class CanvasBridgeSettlementLease:
    """Single-use, exact-load lease for one synchronous host bridge effect."""

    __slots__ = (
        "_effect_allowed",
        "_finalized",
        "_gateway",
        "_load_id",
        "_record",
        "_selection_epoch",
        "_session",
        "_settled",
        "_stale",
        "_used",
    )

    def __init__(
        self,
        *,
        gateway: CanvasGateway,
        session: _BrowserSession,
        load_id: str,
        record: _BridgeSettlementRecord,
        selection_epoch: int,
        effect_allowed: bool,
    ) -> None:
        self._gateway = gateway
        self._session = session
        self._load_id = load_id
        self._record = record
        self._selection_epoch = selection_epoch
        self._effect_allowed = effect_allowed
        self._finalized = False
        self._used = False
        self._settled = False
        self._stale = False

    @property
    def settled(self) -> bool:
        return self._settled

    @property
    def stale(self) -> bool:
        return self._stale

    @property
    def committed_response(self) -> BridgeConfirmationResponse | None:
        """Return the exact linearized response, if a host effect committed."""

        with self._gateway._state_lock:
            if not self._settled:
                return None
            return self._record.response

    def try_settle(self, effect: Callable[[], object]) -> bool:
        """Run one synchronous effect only while the captured load is current."""

        if not callable(effect):
            raise TypeError("Canvas bridge settlement effect must be callable")
        with self._gateway._state_lock:
            if self._finalized or self._used:
                return False
            self._used = True
            if (
                not self._effect_allowed
                or not self._gateway._bridge_lease_is_current(
                    self._session,
                    load_id=self._load_id,
                    selection_epoch=self._selection_epoch,
                )
                or not self._gateway._bridge_record_is_current(
                    self._session, self._record
                )
            ):
                self._stale = True
                return False
            committed = BridgeConfirmationResponse(
                request_id=self._record.request_id,
                status="confirmed",
            )
            committed_expires_at = min(
                self._session.expires_at,
                self._gateway._clock() + self._gateway._bridge_settlement_ttl_seconds,
            )
            self._gateway._prepare_bridge_commit(
                self._session,
                self._record,
                expires_at=committed_expires_at,
            )
            result = effect()
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                raise TypeError("Canvas bridge settlement effect must be synchronous")
            self._gateway._commit_bridge_record(
                self._record,
                committed,
                expires_at=committed_expires_at,
            )
            self._settled = True
            return True

    def _finalize(self) -> None:
        """Permanently close authority access when its callback terminates."""

        with self._gateway._state_lock:
            self._finalized = True
            if not self._settled:
                self._gateway._abandon_bridge_record(self._session, self._record)

    def __repr__(self) -> str:
        return (
            "CanvasBridgeSettlementLease(used="
            f"{self._used}, finalized={self._finalized}, "
            f"settled={self._settled}, stale={self._stale})"
        )


class CanvasGateway:
    """One lazy aiohttp gateway owned by one Chatbook application runtime."""

    def __init__(
        self,
        *,
        authority: CanvasGatewayAuthority,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int | None = None,
        max_browser_sessions: int = _MAX_BROWSER_SESSIONS,
        max_shell_bindings: int = _MAX_SHELL_BINDINGS,
        max_bridge_settlements: int = _MAX_BRIDGE_SETTLEMENTS,
        max_bridge_waiters: int = _MAX_BRIDGE_WAITERS,
        bridge_settlement_ttl_seconds: float = _BRIDGE_SETTLEMENT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError(
                "Canvas gateway requires a numeric loopback address"
            ) from exc
        if not address.is_loopback:
            raise ValueError("Canvas gateway requires a numeric loopback address")
        if port != 0:
            raise ValueError("Canvas gateway requires an OS-assigned port")
        self._authority = authority
        self._gateway_namespace = f"gateway-{uuid4().hex}"
        self._host = address.compressed
        self._port = port
        self._clock = clock
        self._max_request_bytes = (
            max_request_bytes
            if max_request_bytes is not None
            else CanvasLimits().download_payload_bytes + 64 * 1024
        )
        if self._max_request_bytes < 64:
            raise ValueError("max_request_bytes is too small")
        if (
            not isinstance(max_browser_sessions, int)
            or isinstance(max_browser_sessions, bool)
            or max_browser_sessions < 1
            or max_browser_sessions > _MAX_BROWSER_SESSIONS
        ):
            raise ValueError("max_browser_sessions is outside the safe range")
        self._max_browser_sessions = max_browser_sessions
        if (
            not isinstance(max_shell_bindings, int)
            or isinstance(max_shell_bindings, bool)
            or max_shell_bindings < 1
            or max_shell_bindings > _MAX_SHELL_BINDINGS
        ):
            raise ValueError("max_shell_bindings is outside the safe range")
        self._max_shell_bindings = max_shell_bindings
        if (
            not isinstance(max_bridge_settlements, int)
            or isinstance(max_bridge_settlements, bool)
            or max_bridge_settlements < 1
            or max_bridge_settlements > _MAX_BRIDGE_SETTLEMENTS
        ):
            raise ValueError("max_bridge_settlements is outside the safe range")
        self._max_bridge_settlements = max_bridge_settlements
        if (
            not isinstance(max_bridge_waiters, int)
            or isinstance(max_bridge_waiters, bool)
            or max_bridge_waiters < 1
            or max_bridge_waiters > _MAX_BRIDGE_WAITERS
        ):
            raise ValueError("max_bridge_waiters is outside the safe range")
        self._max_bridge_waiters = max_bridge_waiters
        if (
            isinstance(bridge_settlement_ttl_seconds, bool)
            or not isinstance(bridge_settlement_ttl_seconds, (int, float))
            or not math.isfinite(bridge_settlement_ttl_seconds)
            or bridge_settlement_ttl_seconds <= 0
            or bridge_settlement_ttl_seconds > _BRIDGE_SETTLEMENT_TTL_SECONDS
        ):
            raise ValueError("bridge_settlement_ttl_seconds is outside the safe range")
        self._bridge_settlement_ttl_seconds = float(bridge_settlement_ttl_seconds)
        self.capabilities = CanvasCapabilityStore(clock=clock)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._origin: str | None = None
        self._start_lock = asyncio.Lock()
        self._state_lock = RLock()
        self._closed = False
        self._start_count = 0
        self._sessions: dict[bytes, _BrowserSession] = {}
        self._session_ids: dict[str, bytes] = {}
        self._shell_bindings: dict[str, _ShellBinding] = {}
        self._assets: CanvasRuntimeAssets | None = None
        self._app = web.Application(
            middlewares=[
                self._security_headers_middleware,
                self._request_policy_middleware,
            ]
        )
        self._install_routes()

    def __repr__(self) -> str:
        return (
            f"CanvasGateway(host={self._host!r}, started={self.started}, "
            f"browser_sessions={len(self._sessions)}, closed={self._closed})"
        )

    @property
    def started(self) -> bool:
        return (
            self._runner is not None and self._origin is not None and not self._closed
        )

    @property
    def origin(self) -> str | None:
        return self._origin

    @property
    def start_count(self) -> int:
        return self._start_count

    @property
    def browser_session_count(self) -> int:
        self._discard_expired_sessions()
        return len(self._sessions)

    @property
    def bridge_settlement_count(self) -> int:
        """Return bounded current-load idempotency records across live sessions."""

        with self._state_lock:
            self._discard_expired_sessions()
            return sum(
                len(session.bridge_settlements) for session in self._sessions.values()
            )

    @property
    def routes(self) -> tuple[web.AbstractRoute, ...]:
        return tuple(self._app.router.routes())

    async def start(self) -> str:
        """Start once on an OS-assigned loopback port and return its origin."""

        async with self._start_lock:
            if self.started:
                assert self._origin is not None
                return self._origin
            if self._closed:
                raise RuntimeError("Canvas gateway is closed")
            if self._runner is not None:
                try:
                    await self._cleanup_runner()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - platform cleanup failures vary
                    raise RuntimeError("Canvas gateway could not start") from None
            try:
                await self._bind()
            except Exception:  # noqa: BLE001 - aiohttp bind failures are platform-specific
                try:
                    await self._cleanup_runner()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001, S110 - retain runner for retry
                    pass
                raise RuntimeError("Canvas gateway could not start") from None
            self._start_count += 1
            assert self._origin is not None
            return self._origin

    async def _bind(self) -> None:
        runner = web.AppRunner(self._app, access_log=None)
        await runner.setup()
        self._runner = runner
        site = web.TCPSite(runner, self._host, self._port)
        self._site = site
        await site.start()
        addresses = runner.addresses
        if len(addresses) != 1:
            raise RuntimeError("Canvas gateway bind was ambiguous")
        bound_host, bound_port = addresses[0][:2]
        if ipaddress.ip_address(bound_host) != ipaddress.ip_address(self._host):
            raise RuntimeError("Canvas gateway bound an unexpected address")
        display_host = f"[{bound_host}]" if ":" in bound_host else bound_host
        self._origin = f"http://{display_host}:{bound_port}"

    async def open_shell(
        self,
        scope: CanvasGatewayScope,
        *,
        opener: Callable[[str], object] | None = None,
    ) -> CanvasGatewayLaunch:
        """Mint one shell bootstrap and optionally ask Textual to open it."""

        origin = await self.start()
        with self._state_lock:
            self._discard_expired_sessions()
            # A browser-session ID names exactly one shell incarnation. Reusing
            # it for a new launch invalidates any pending bootstrap or live shell
            # before the replacement token is minted.
            self._revoke_session_id(scope.browser_session_id)
            if len(self._shell_bindings) >= self._max_shell_bindings:
                raise CanvasCapabilityError("Canvas shell capacity reached")
            shell_incarnation_id = f"shell-{uuid4().hex}"
            load_id = f"boot-{uuid4()}"
            # Mint before publishing the route binding: a full/closed capability
            # store must never leave an unreachable pending shell behind.
            grant = self.capabilities.issue(
                _capability_scope(
                    scope,
                    load_id=load_id,
                    action="shell_boot",
                    gateway_namespace=self._gateway_namespace,
                    shell_incarnation_id=shell_incarnation_id,
                ),
                ttl_seconds=_BOOT_TTL_SECONDS,
            )
            self._shell_bindings[shell_incarnation_id] = _ShellBinding(
                browser_session_id=scope.browser_session_id,
                expires_at=self._clock() + _BOOT_TTL_SECONDS,
            )
        clean_url = f"{origin}{self._route_prefix(shell_incarnation_id)}/"
        browser_url = f"{clean_url}#boot={quote(grant.token, safe='')}"
        opened: bool | None = None
        error_code: str | None = None
        if opener is not None:
            try:
                result = opener(browser_url)
                if inspect.isawaitable(result):
                    await result
                opened = True
            except Exception:  # noqa: BLE001 - platform browser openers are untyped
                opened = False
                error_code = "browser_unavailable"
        return CanvasGatewayLaunch(
            clean_url=clean_url,
            browser_url=browser_url,
            opened=opened,
            error_code=error_code,
        )

    def change_selection(
        self, *, browser_session_id: str, scope: CanvasGatewayScope
    ) -> None:
        """Replace one shell's exact selection and revoke its previous load."""

        if scope.browser_session_id != browser_session_id:
            raise ValueError("Canvas browser session mismatch")
        with self._state_lock:
            digest = self._session_ids.get(browser_session_id)
            session = self._sessions.get(digest) if digest is not None else None
            if session is None:
                raise ValueError("Canvas browser session is unavailable")
            previous = session.scope
            self.capabilities.revoke_selection(
                browser_session_id=previous.browser_session_id,
                conversation_session_id=previous.conversation_session_id,
                canvas_id=previous.canvas_id,
                revision_id=previous.revision_id,
            )
            self._clear_bridge_records(session)
            session.scope = scope
            session.selection_epoch += 1
            session.current_load_id = None

    async def aclose(self) -> None:
        """Revoke all browser authority and cleanly stop the listener."""

        async with self._start_lock:
            with self._state_lock:
                self._closed = True
                self.capabilities.close()
                for session in self._sessions.values():
                    self._clear_bridge_records(session)
                self._sessions.clear()
                self._session_ids.clear()
                self._shell_bindings.clear()
            await self._cleanup_runner()

    async def _cleanup_runner(self) -> None:
        runner = self._runner
        if runner is None:
            self._site = None
            self._origin = None
            return
        await runner.cleanup()
        if self._runner is runner:
            self._site = None
            self._runner = None
            self._origin = None

    def _install_routes(self) -> None:
        root = f"/canvas/{self._gateway_namespace}/{{shell_incarnation_id}}"
        self._app.router.add_get(f"{root}/", self._shell, allow_head=False)
        self._app.router.add_post(f"{root}/api/boot", self._boot)
        self._app.router.add_post(f"{root}/api/frame", self._frame)
        self._app.router.add_get(f"{root}/api/state", self._state, allow_head=False)
        self._app.router.add_post(f"{root}/api/navigate", self._navigate)
        self._app.router.add_get(f"{root}/render", self._renderer, allow_head=False)
        self._app.router.add_get(f"{root}/api/plan", self._plan, allow_head=False)
        self._app.router.add_get(f"{root}/api/events", self._events, allow_head=False)
        self._app.router.add_post(f"{root}/api/actions", self._action_capability)
        self._app.router.add_get(f"{root}/api/source", self._source, allow_head=False)
        self._app.router.add_get(
            f"{root}/api/source-download", self._source_download, allow_head=False
        )
        self._app.router.add_post(f"{root}/api/bridge", self._bridge)
        self._app.router.add_post(f"{root}/api/close", self._close_session)
        self._app.router.add_get(
            f"{root}/static/{{name}}", self._static_asset, allow_head=False
        )

    @web.middleware
    async def _security_headers_middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = _error_response("request_refused", exc.status)
        except Exception:  # noqa: BLE001 - HTTP boundary returns a content-free refusal
            response = _error_response("gateway_unavailable", 503)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = (
            "SAMEORIGIN" if request.path.endswith("/render") else "DENY"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=(), "
            "serial=(), bluetooth=(), clipboard-read=(), clipboard-write=()"
        )
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if "/static/" in request.path:
            # The renderer has an intentionally opaque sandbox origin. These
            # verified static bytes contain no session or Canvas data.
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        response.headers["Content-Security-Policy"] = (
            _RENDERER_CSP if request.path.endswith("/render") else _SHELL_CSP
        )
        return response

    @web.middleware
    async def _request_policy_middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        if any(
            name.lower() == "forwarded"
            or name.lower().startswith("x-forwarded-")
            or name.lower() in _OTHER_PROXY_HEADERS
            for name in request.headers
        ):
            return _error_response("proxy_headers_refused", 400)
        host_headers = request.headers.getall("Host", [])
        if (
            self._origin is None
            or len(host_headers) != 1
            or request.host != self._origin.split("//", 1)[1]
        ):
            return _error_response("host_refused", 400)
        if any(key.lower() in _SENSITIVE_QUERY_KEYS for key in request.query):
            return _error_response("query_credentials_refused", 400)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin_headers = request.headers.getall("Origin", [])
            if len(origin_headers) != 1 or origin_headers[0] != self._origin:
                return _error_response("origin_refused", 403)
            content_types = request.headers.getall("Content-Type", [])
            if len(content_types) != 1 or request.content_type != "application/json":
                return _error_response("content_type_refused", 415)
            if (
                request.content_length is not None
                and request.content_length > self._max_request_bytes
            ):
                return _error_response("request_too_large", 413)
        return await handler(request)

    async def _shell(self, _request: web.Request) -> web.Response:
        if self._request_shell_incarnation(_request) is None:
            return _error_response("shell_unavailable", 404)
        return web.Response(body=_SHELL_HTML, content_type="text/html", charset="utf-8")

    async def _boot(self, request: web.Request) -> web.Response:
        self._discard_expired_sessions()
        shell_incarnation_id = self._request_shell_incarnation(request)
        if shell_incarnation_id is None:
            return _error_response("boot_unavailable", 401)
        binding = self._shell_bindings[shell_incarnation_id]
        value = await self._read_json(request)
        if not isinstance(value, Mapping) or set(value) != {"bootstrap"}:
            return _error_response("invalid_boot_exchange", 400)
        token = value["bootstrap"]
        try:
            scope = self.capabilities.consume(
                token,
                expected_action="shell_boot",
                expected_gateway_namespace=self._gateway_namespace,
                expected_shell_incarnation_id=shell_incarnation_id,
            )
        except CanvasCapabilityError:
            return _error_response("boot_unavailable", 401)
        gateway_scope = CanvasGatewayScope(
            browser_session_id=scope.browser_session_id,
            conversation_session_id=scope.conversation_session_id,
            canvas_id=scope.canvas_id,
            revision_id=scope.revision_id,
        )
        if binding.browser_session_id != gateway_scope.browser_session_id:
            return _error_response("boot_unavailable", 401)
        replacing = gateway_scope.browser_session_id in self._session_ids
        if not replacing and len(self._sessions) >= self._max_browser_sessions:
            return _error_response("browser_session_capacity", 503)
        self._revoke_session_id(
            gateway_scope.browser_session_id,
            except_shell_incarnation_id=shell_incarnation_id,
        )
        session_token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        digest = _secret_digest(session_token)
        session = _BrowserSession(
            digest=digest,
            csrf_digest=_secret_digest(csrf),
            scope=gateway_scope,
            expires_at=self._clock() + _BROWSER_SESSION_TTL_SECONDS,
            shell_incarnation_id=shell_incarnation_id,
        )
        self._sessions[digest] = session
        self._session_ids[gateway_scope.browser_session_id] = digest
        self._shell_bindings[shell_incarnation_id] = _ShellBinding(
            browser_session_id=gateway_scope.browser_session_id,
            expires_at=session.expires_at,
        )
        response = web.json_response(
            {
                "browser_session_id": gateway_scope.browser_session_id,
                "csrf": csrf,
                "selection": {
                    "canvas_id": gateway_scope.canvas_id,
                    "revision_id": gateway_scope.revision_id,
                },
            }
        )
        response.set_cookie(
            _SESSION_COOKIE,
            session_token,
            httponly=True,
            samesite="Strict",
            path=f"{self._route_prefix(shell_incarnation_id)}/",
            max_age=int(_BROWSER_SESSION_TTL_SECONDS),
        )
        return response

    async def _frame(self, request: web.Request) -> web.Response:
        session = self._require_session(request, csrf=True)
        if session is None:
            return _error_response("session_refused", 403)
        value = await self._read_json(request)
        if value != {}:
            return _error_response("invalid_frame_request", 400)
        with self._state_lock:
            if not self._session_is_current(session, session.scope):
                return _error_response("session_refused", 403)
            if session.current_load_id is not None:
                self._clear_bridge_records(session)
                self.capabilities.revoke_load(
                    session.scope.browser_session_id, session.current_load_id
                )
            load_id = f"load-{uuid4()}"
            session.current_load_id = load_id
            renderer = self.capabilities.issue(
                self._session_capability_scope(
                    session, load_id=load_id, action="renderer_load"
                ),
                ttl_seconds=_FRAME_TTL_SECONDS,
            )
            plan = self.capabilities.issue(
                self._session_capability_scope(
                    session, load_id=load_id, action="render_plan"
                ),
                ttl_seconds=_FRAME_TTL_SECONDS,
            )
        response = web.json_response(
            {
                "load_id": load_id,
                "renderer_url": f"{self._route_prefix(session.shell_incarnation_id)}/render",
            }
        )
        route_prefix = self._route_prefix(session.shell_incarnation_id)
        response.set_cookie(
            _FRAME_COOKIE,
            renderer.token,
            httponly=True,
            samesite="Strict",
            path=f"{route_prefix}/render",
            max_age=int(_FRAME_TTL_SECONDS),
        )
        response.set_cookie(
            _PLAN_COOKIE,
            plan.token,
            httponly=True,
            samesite="Strict",
            path=f"{route_prefix}/api/plan",
            max_age=int(_FRAME_TTL_SECONDS),
        )
        return response

    async def _state(self, request: web.Request) -> web.Response:
        session = self._require_session(request)
        if session is None:
            return _error_response("session_refused", 401)
        scope = session.scope
        projection = await _maybe_await(self._authority.describe_selection(scope))
        if (
            not self._session_is_current(session, scope)
            or not isinstance(projection, CanvasGatewayProjection)
            or projection.scope != scope
        ):
            return _error_response("state_unavailable", 503)
        return web.json_response(_projection_wire(projection))

    async def _navigate(self, request: web.Request) -> web.Response:
        session = self._require_session(request, csrf=True)
        if session is None:
            return _error_response("session_refused", 403)
        value = await self._read_json(request)
        if not isinstance(value, Mapping) or set(value) - {"action", "canvas_id", "title"}:
            return _error_response("invalid_navigation", 400)
        action = value.get("action")
        canvas_id = value.get("canvas_id")
        title = value.get("title")
        if (
            action not in {"select", "pin", "follow", "previous", "rename"}
            or (canvas_id is not None and not isinstance(canvas_id, str))
            or (title is not None and not isinstance(title, str))
        ):
            return _error_response("invalid_navigation", 400)
        captured = session.scope
        try:
            navigation = await _maybe_await(
                self._authority.navigate(
                    captured, action=action, canvas_id=canvas_id, title=title
                )
            )
        except (TypeError, ValueError):
            return _error_response("navigation_refused", 409)
        if (
            not self._session_is_current(session, captured)
            or not isinstance(navigation, CanvasGatewayNavigation)
            or navigation.scope.browser_session_id != captured.browser_session_id
            or navigation.scope.conversation_session_id
            != captured.conversation_session_id
            or navigation.projection.scope != navigation.scope
        ):
            return _error_response("navigation_unavailable", 503)
        self.change_selection(
            browser_session_id=captured.browser_session_id,
            scope=navigation.scope,
        )
        return web.json_response(_projection_wire(navigation.projection))

    async def _renderer(self, request: web.Request) -> web.Response:
        if (
            request.headers.get("Sec-Fetch-Dest") != "iframe"
            or request.headers.get("Sec-Fetch-Site") != "same-origin"
        ):
            return _error_response("renderer_context_refused", 403)
        session = self._require_session(request)
        token = request.cookies.get(_FRAME_COOKIE)
        if session is None or token is None or session.current_load_id is None:
            return _error_response("renderer_unavailable", 401)
        expected = self._session_capability_scope(
            session,
            load_id=session.current_load_id,
            action="renderer_load",
        )
        try:
            self.capabilities.consume(token, expected_scope=expected)
        except CanvasCapabilityError:
            return _error_response("renderer_unavailable", 401)
        assets = self._runtime_assets()
        if not assets.enabled or assets.renderer_javascript is None:
            return _error_response("runtime_unavailable", 503)
        integrity = base64.b64encode(
            hashlib.sha384(assets.renderer_javascript).digest()
        ).decode("ascii")
        body = (
            '<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="referrer" content="no-referrer">'
            f'<script type="module" src="{self._route_prefix(session.shell_incarnation_id)}/static/canvas_renderer.js" '
            f'integrity="sha384-{integrity}" crossorigin="anonymous"></script>'
            '</head><body><div id="canvas-root"></div></body></html>'
        ).encode()
        return web.Response(body=body, content_type="text/html", charset="utf-8")

    async def _plan(self, request: web.Request) -> web.Response:
        if (
            request.headers.get("Sec-Fetch-Dest") != "empty"
            or request.headers.get("Sec-Fetch-Site") != "same-origin"
        ):
            return _error_response("plan_context_refused", 403)
        session = self._require_session(request)
        token = request.cookies.get(_PLAN_COOKIE)
        if session is None or token is None or session.current_load_id is None:
            return _error_response("plan_unavailable", 401)
        scope = session.scope
        load_id = session.current_load_id
        expected = self._session_capability_scope(
            session,
            load_id=load_id,
            action="render_plan",
        )
        try:
            self.capabilities.consume(token, expected_scope=expected)
        except CanvasCapabilityError:
            return _error_response("plan_unavailable", 401)
        plan = await _maybe_await(self._authority.resolve_render_plan(scope))
        if not self._session_is_current(session, scope) or not isinstance(
            plan, CanvasRenderPlan
        ):
            return _error_response("plan_unavailable", 503)
        return web.json_response(_render_plan_wire(plan))

    async def _events(self, request: web.Request) -> web.Response:
        session = self._require_session(request)
        if session is None:
            return _error_response("session_refused", 401)
        scope = session.scope
        after = request.headers.get("Last-Event-ID")
        if after is not None:
            try:
                validate_opaque_identifier(after, field_name="last event ID")
            except CanvasLimitError:
                return _error_response("invalid_event_cursor", 400)
        events = await _maybe_await(
            self._authority.read_events(scope, after_event_id=after)
        )
        if (
            not self._session_is_current(session, scope)
            or not isinstance(events, tuple)
            or not all(isinstance(event, CanvasGatewayEvent) for event in events)
            or not all(event.canvas_id == scope.canvas_id for event in events)
        ):
            return _error_response("events_unavailable", 503)
        return web.json_response(
            {
                "events": [
                    {
                        "event_id": event.event_id,
                        "kind": event.kind,
                        "canvas_id": event.canvas_id,
                        "revision_id": event.revision_id,
                        "metadata": dict(event.metadata),
                    }
                    for event in events
                ]
            }
        )

    async def _action_capability(self, request: web.Request) -> web.Response:
        session = self._require_session(request, csrf=True)
        if session is None:
            return _error_response("session_refused", 403)
        value = await self._read_json(request)
        if not isinstance(value, Mapping) or set(value) != {"action"}:
            return _error_response("invalid_action_request", 400)
        action = value["action"]
        if action not in {"source_read", "source_download", "bridge_confirm"}:
            return _error_response("action_refused", 400)
        load_id = session.current_load_id or f"action-{uuid4()}"
        grant = self.capabilities.issue(
            self._session_capability_scope(session, load_id=load_id, action=action),
            ttl_seconds=_ACTION_TTL_SECONDS,
        )
        return web.json_response(
            {"capability": grant.token, "expires_in_seconds": grant.expires_in_seconds}
        )

    async def _source(self, request: web.Request) -> web.Response:
        return await self._source_response(
            request, action="source_read", download=False
        )

    async def _source_download(self, request: web.Request) -> web.Response:
        return await self._source_response(
            request, action="source_download", download=True
        )

    async def _source_response(
        self,
        request: web.Request,
        *,
        action: Literal["source_read", "source_download"],
        download: bool,
    ) -> web.Response:
        session = self._require_session(request)
        token = _authorization_capability(request)
        if session is None or token is None:
            return _error_response("source_unavailable", 401)
        scope = session.scope
        load_id = session.current_load_id
        if load_id is None:
            # Action grants issued before the first renderer use a generated ID;
            # inspect the action directly while still binding every other scope.
            try:
                granted = self.capabilities.consume(token, expected_action=action)
            except CanvasCapabilityError:
                return _error_response("source_unavailable", 401)
            if _gateway_scope(granted) != scope:
                return _error_response("source_unavailable", 401)
        else:
            expected = self._session_capability_scope(
                session, load_id=load_id, action=action
            )
            try:
                self.capabilities.consume(token, expected_scope=expected)
            except CanvasCapabilityError:
                return _error_response("source_unavailable", 401)
        source = await _maybe_await(self._authority.read_source(scope))
        if not self._session_is_current(session, scope) or not isinstance(
            source, CanvasSourceResponse
        ):
            return _error_response("source_unavailable", 503)
        if sha256_utf8(source.source) != source.content_sha256:
            return _error_response("source_unavailable", 503)
        response = web.Response(
            text=source.source, content_type="text/plain", charset="utf-8"
        )
        if download:
            response.headers["Content-Disposition"] = (
                'attachment; filename="canvas-source.canvas.html.txt"'
            )
        return response

    async def _bridge(self, request: web.Request) -> web.Response:
        session = self._require_session(request, csrf=True)
        token = _authorization_capability(request)
        if session is None or token is None:
            return _error_response("bridge_refused", 403)
        scope = session.scope
        value = await self._read_json(request)
        try:
            confirmation = BridgeConfirmationRequest.from_wire(value)
        except (TypeError, ValueError):
            return _error_response("invalid_bridge_request", 400)
        load_id = session.current_load_id
        if load_id is None:
            return _error_response("bridge_refused", 409)
        expected = self._session_capability_scope(
            session, load_id=load_id, action="bridge_confirm"
        )
        try:
            self.capabilities.consume(token, expected_scope=expected)
        except CanvasCapabilityError:
            return _error_response("bridge_refused", 401)
        reservation, record = self._reserve_bridge_record(
            session,
            load_id=load_id,
            confirmation=confirmation,
        )
        if reservation == "collision":
            return _error_response("bridge_request_collision", 409)
        if reservation == "capacity":
            return _error_response("bridge_settlement_capacity", 503)
        if reservation == "waiter_capacity":
            return _error_response("bridge_waiter_capacity", 503)
        assert record is not None
        if reservation == "replay":
            assert record.response is not None
            return _bridge_confirmation_response(record.response)
        if reservation == "wait":
            replay = await self._wait_for_bridge_record(session, record)
            if replay is None:
                return _error_response("bridge_refused", 503)
            return _bridge_confirmation_response(replay)
        settlement = CanvasBridgeSettlementLease(
            gateway=self,
            session=session,
            load_id=load_id,
            record=record,
            selection_epoch=session.selection_epoch,
            effect_allowed=confirmation.approved,
        )
        callback_failed = False
        result: object = None
        try:
            result = await _maybe_await(
                self._authority.confirm_bridge(
                    scope,
                    confirmation,
                    settlement=settlement,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - committed effects remain authoritative
            callback_failed = True
        finally:
            settlement._finalize()
        committed = settlement.committed_response
        if committed is not None:
            return _bridge_confirmation_response(committed)
        if record.terminal_reason in {"expired", "revoked"}:
            return _error_response("bridge_settlement_stale", 409)
        if callback_failed:
            return _error_response("bridge_refused", 503)
        if not isinstance(result, BridgeConfirmationResponse):
            return _error_response("bridge_refused", 503)
        if settlement.stale or (
            result.status == "confirmed" and not settlement.settled
        ):
            return _error_response("bridge_settlement_stale", 409)
        if (
            not self._session_is_current(session, scope)
            or result.request_id != confirmation.request.request_id
            or (settlement.settled and result.status != "confirmed")
        ):
            return _error_response("bridge_refused", 503)
        return _bridge_confirmation_response(result)

    async def _close_session(self, request: web.Request) -> web.Response:
        session = self._require_session(request, csrf=True)
        if session is None:
            return _error_response("session_refused", 403)
        value = await self._read_json(request)
        if value != {}:
            return _error_response("invalid_close_request", 400)
        self._revoke_session_id(session.scope.browser_session_id)
        response = web.json_response({"status": "closed"})
        response.del_cookie(
            _SESSION_COOKIE,
            path=f"{self._route_prefix(session.shell_incarnation_id)}/",
        )
        return response

    async def _static_asset(self, request: web.Request) -> web.Response:
        if self._request_shell_incarnation(request) is None:
            return _error_response("asset_not_found", 404)
        name = request.match_info["name"]
        shell_asset = _SHELL_ASSETS.get(name)
        if shell_asset is not None:
            content_type, charset = shell_asset
            return web.Response(
                body=_STATIC_ROOT.joinpath(name).read_bytes(),
                content_type=content_type,
                charset=charset,
            )
        assets = self._runtime_assets()
        inventory = {
            "canvas_renderer.js": assets.renderer_javascript,
            "canvas_runtime_worker.js": assets.worker_javascript,
            "quickjs-runtime.js": assets.javascript,
        }
        if name not in inventory or inventory[name] is None or not assets.enabled:
            return _error_response("asset_not_found", 404)
        return web.Response(
            body=inventory[name], content_type="text/javascript", charset="utf-8"
        )

    async def _read_json(self, request: web.Request) -> object:
        body = await request.content.read(self._max_request_bytes + 1)
        if len(body) > self._max_request_bytes:
            raise web.HTTPRequestEntityTooLarge(
                max_size=self._max_request_bytes, actual_size=len(body)
            )
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise web.HTTPBadRequest() from exc

    def _require_session(
        self, request: web.Request, *, csrf: bool = False
    ) -> _BrowserSession | None:
        self._discard_expired_sessions()
        shell_incarnation_id = self._request_shell_incarnation(request)
        if shell_incarnation_id is None:
            return None
        token = request.cookies.get(_SESSION_COOKIE)
        if token is None:
            return None
        digest = _secret_digest(token)
        session = self._sessions.get(digest)
        if (
            session is None
            or session.shell_incarnation_id != shell_incarnation_id
            or self._shell_bindings[shell_incarnation_id].browser_session_id
            != session.scope.browser_session_id
            or not hmac.compare_digest(session.digest, digest)
        ):
            return None
        if csrf:
            csrf_values = request.headers.getall("X-Canvas-CSRF", [])
            if len(csrf_values) != 1 or not hmac.compare_digest(
                session.csrf_digest, _secret_digest(csrf_values[0])
            ):
                return None
        return session

    def _discard_expired_sessions(self) -> None:
        with self._state_lock:
            for session in tuple(self._sessions.values()):
                self._discard_expired_bridge_records(session)
            expired = [
                session.scope.browser_session_id
                for session in self._sessions.values()
                if self._clock() >= session.expires_at
            ]
            for browser_session_id in expired:
                self._revoke_session_id(browser_session_id)
            now = self._clock()
            expired_shells = [
                shell_incarnation_id
                for shell_incarnation_id, binding in self._shell_bindings.items()
                if now >= binding.expires_at
                and binding.browser_session_id not in self._session_ids
            ]
            for shell_incarnation_id in expired_shells:
                self._shell_bindings.pop(shell_incarnation_id, None)

    def _session_is_current(
        self, session: _BrowserSession, scope: CanvasGatewayScope
    ) -> bool:
        """Check that an awaited authority response still targets a live scope."""

        with self._state_lock:
            return (
                not self._closed
                and session.scope == scope
                and self._sessions.get(session.digest) is session
                and self._session_ids.get(scope.browser_session_id) == session.digest
                and self._clock() < session.expires_at
            )

    def _bridge_lease_is_current(
        self,
        session: _BrowserSession,
        *,
        load_id: str,
        selection_epoch: int,
    ) -> bool:
        return (
            self._session_is_current(session, session.scope)
            and session.current_load_id == load_id
            and session.selection_epoch == selection_epoch
        )

    def _reserve_bridge_record(
        self,
        session: _BrowserSession,
        *,
        load_id: str,
        confirmation: BridgeConfirmationRequest,
    ) -> tuple[str, _BridgeSettlementRecord | None]:
        """Reserve or replay one exact, source-free bridge idempotency record."""

        with self._state_lock:
            for live_session in self._sessions.values():
                self._discard_expired_bridge_records(live_session)
            request = confirmation.request
            payload_digest = _bridge_payload_digest(request.value)
            existing = session.bridge_settlements.get(request.request_id)
            if existing is not None:
                matches = (
                    existing.browser_session_id == session.scope.browser_session_id
                    and existing.load_id == load_id
                    and existing.selection_epoch == session.selection_epoch
                    and existing.request_kind == request.kind
                    and hmac.compare_digest(existing.payload_digest, payload_digest)
                )
                if not matches:
                    return "collision", None
                if existing.response is not None:
                    return "replay", existing
                if existing.waiter_count >= self._max_bridge_waiters:
                    return "waiter_capacity", None
                existing.waiter_count += 1
                return "wait", existing
            active_records = sum(
                len(live_session.bridge_settlements)
                for live_session in self._sessions.values()
            )
            if active_records >= self._max_bridge_settlements:
                return "capacity", None
            record = _BridgeSettlementRecord(
                browser_session_id=session.scope.browser_session_id,
                load_id=load_id,
                selection_epoch=session.selection_epoch,
                request_id=request.request_id,
                request_kind=request.kind,
                payload_digest=payload_digest,
                expires_at=min(
                    session.expires_at,
                    self._clock() + self._bridge_settlement_ttl_seconds,
                ),
            )
            session.bridge_settlements[request.request_id] = record
            try:
                self._schedule_bridge_expiry(session, record)
            except Exception:  # noqa: BLE001 - scheduling failures stay content-free
                self._terminally_remove_bridge_record(
                    session, record, reason="schedule_failed"
                )
                raise RuntimeError("Canvas bridge expiry scheduling failed") from None
            return "owner", record

    def _bridge_record_is_current(
        self,
        session: _BrowserSession,
        record: _BridgeSettlementRecord,
    ) -> bool:
        return (
            self._clock() < record.expires_at
            and session.bridge_settlements.get(record.request_id) is record
            and record.response is None
        )

    def _schedule_bridge_expiry(
        self,
        session: _BrowserSession,
        record: _BridgeSettlementRecord,
    ) -> None:
        """Arm the single self-enforcing deadline for one settlement record."""

        self._cancel_bridge_expiry(record)
        delay = max(0.0, record.expires_at - self._clock())
        record.expiry_handle = asyncio.get_running_loop().call_later(
            delay,
            self._expire_bridge_record,
            session,
            record,
        )

    @staticmethod
    def _cancel_bridge_expiry(record: _BridgeSettlementRecord) -> None:
        handle = record.expiry_handle
        record.expiry_handle = None
        if handle is not None and not handle.cancelled():
            handle.cancel()

    def _expire_bridge_record(
        self,
        session: _BrowserSession,
        record: _BridgeSettlementRecord,
    ) -> None:
        """Expire only the record still occupying its exact session slot."""

        with self._state_lock:
            if session.bridge_settlements.get(record.request_id) is not record:
                self._cancel_bridge_expiry(record)
                return
            record.expiry_handle = None
            if self._clock() < record.expires_at:
                self._schedule_bridge_expiry(session, record)
                return
            self._terminally_remove_bridge_record(
                session,
                record,
                reason="expired",
            )

    def _terminally_remove_bridge_record(
        self,
        session: _BrowserSession,
        record: _BridgeSettlementRecord,
        *,
        reason: str,
    ) -> None:
        """Remove an exact record, cancel its timer, and wake every joiner."""

        if session.bridge_settlements.get(record.request_id) is not record:
            self._cancel_bridge_expiry(record)
            return
        session.bridge_settlements.pop(record.request_id, None)
        record.terminal_reason = reason
        self._cancel_bridge_expiry(record)
        record.completed.set()

    def _prepare_bridge_commit(
        self,
        session: _BrowserSession,
        record: _BridgeSettlementRecord,
        *,
        expires_at: float,
    ) -> None:
        """Install the committed receipt deadline before its effect can run."""

        record.expires_at = expires_at
        self._schedule_bridge_expiry(session, record)

    def _commit_bridge_record(
        self,
        record: _BridgeSettlementRecord,
        response: BridgeConfirmationResponse,
        *,
        expires_at: float,
    ) -> None:
        # `try_settle` already checked this record while holding `_state_lock`.
        # Do not re-read the clock after the synchronous effect: once that
        # effect returns, publishing its receipt must be exception-free.
        record.response = response
        record.expires_at = expires_at
        record.completed.set()

    async def _wait_for_bridge_record(
        self,
        session: _BrowserSession,
        record: _BridgeSettlementRecord,
    ) -> BridgeConfirmationResponse | None:
        """Wait no longer than the record's monotonic deadline for its receipt."""

        try:
            while not record.completed.is_set():
                remaining = record.expires_at - self._clock()
                if remaining <= 0:
                    self._expire_bridge_record(session, record)
                    break
                try:
                    await asyncio.wait_for(record.completed.wait(), timeout=remaining)
                except TimeoutError:
                    self._expire_bridge_record(session, record)
            return record.response
        finally:
            with self._state_lock:
                if record.waiter_count > 0:
                    record.waiter_count -= 1

    def _abandon_bridge_record(
        self,
        session: _BrowserSession,
        record: _BridgeSettlementRecord,
    ) -> None:
        if (
            record.response is None
            and session.bridge_settlements.get(record.request_id) is record
        ):
            self._terminally_remove_bridge_record(
                session,
                record,
                reason="callback_complete",
            )

    def _discard_expired_bridge_records(self, session: _BrowserSession) -> None:
        now = self._clock()
        expired = [
            record
            for record in session.bridge_settlements.values()
            if now >= record.expires_at
        ]
        for record in expired:
            self._terminally_remove_bridge_record(
                session,
                record,
                reason="expired",
            )

    def _clear_bridge_records(self, session: _BrowserSession) -> None:
        records = tuple(session.bridge_settlements.values())
        for record in records:
            self._terminally_remove_bridge_record(
                session,
                record,
                reason="revoked",
            )

    def _revoke_session_id(
        self,
        browser_session_id: str,
        *,
        except_shell_incarnation_id: str | None = None,
    ) -> None:
        with self._state_lock:
            digest = self._session_ids.pop(browser_session_id, None)
            if digest is not None:
                session = self._sessions.pop(digest, None)
                if session is not None:
                    self._clear_bridge_records(session)
            shells = [
                shell_incarnation_id
                for shell_incarnation_id, binding in self._shell_bindings.items()
                if binding.browser_session_id == browser_session_id
                and shell_incarnation_id != except_shell_incarnation_id
            ]
            for shell_incarnation_id in shells:
                self._shell_bindings.pop(shell_incarnation_id, None)
            self.capabilities.revoke_browser_session(browser_session_id)

    def _request_shell_incarnation(self, request: web.Request) -> str | None:
        shell_incarnation_id = request.match_info.get("shell_incarnation_id")
        try:
            validate_opaque_identifier(
                shell_incarnation_id, field_name="shell incarnation ID"
            )
        except (CanvasLimitError, TypeError):
            return None
        return (
            shell_incarnation_id
            if shell_incarnation_id in self._shell_bindings
            else None
        )

    def _route_prefix(self, shell_incarnation_id: str) -> str:
        return f"/canvas/{self._gateway_namespace}/{shell_incarnation_id}"

    def _session_capability_scope(
        self,
        session: _BrowserSession,
        *,
        load_id: str,
        action: CanvasCapabilityAction,
    ) -> CanvasCapabilityScope:
        return _capability_scope(
            session.scope,
            load_id=load_id,
            action=action,
            gateway_namespace=self._gateway_namespace,
            shell_incarnation_id=session.shell_incarnation_id,
        )

    def _runtime_assets(self) -> CanvasRuntimeAssets:
        if self._assets is None:
            self._assets = load_canvas_runtime_assets()
        return self._assets


def _capability_scope(
    scope: CanvasGatewayScope,
    *,
    load_id: str,
    action: CanvasCapabilityAction,
    gateway_namespace: str,
    shell_incarnation_id: str,
) -> CanvasCapabilityScope:
    return CanvasCapabilityScope(
        browser_session_id=scope.browser_session_id,
        load_id=load_id,
        conversation_session_id=scope.conversation_session_id,
        canvas_id=scope.canvas_id,
        revision_id=scope.revision_id,
        action=action,
        gateway_namespace=gateway_namespace,
        shell_incarnation_id=shell_incarnation_id,
    )


def _gateway_scope(scope: CanvasCapabilityScope) -> CanvasGatewayScope:
    return CanvasGatewayScope(
        browser_session_id=scope.browser_session_id,
        conversation_session_id=scope.conversation_session_id,
        canvas_id=scope.canvas_id,
        revision_id=scope.revision_id,
    )


def _secret_digest(value: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 256:
        return b"\x00" * 32
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return b"\x00" * 32
    return hashlib.sha256(encoded).digest()


def _authorization_capability(request: web.Request) -> str | None:
    values = request.headers.getall("Authorization", [])
    if len(values) != 1:
        return None
    value = values[0]
    prefix = "CanvasCapability "
    if value is None or not value.startswith(prefix):
        return None
    token = value[len(prefix) :]
    return token if token and " " not in token else None


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _bridge_payload_digest(value: object) -> bytes:
    encoded = json.dumps(
        _canonical_json_wire(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _canonical_json_wire(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _canonical_json_wire(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_wire(child) for child in value]
    return value


def _bridge_confirmation_response(result: BridgeConfirmationResponse) -> web.Response:
    return web.json_response({"request_id": result.request_id, "status": result.status})


def _render_plan_wire(plan: CanvasRenderPlan) -> dict[str, Any]:
    return {
        "runtime_profile": plan.runtime_profile,
        "source_identity": {
            "source_bytes": plan.source_identity.source_bytes,
            "sha256": plan.source_identity.sha256,
        },
        "root": _render_node_wire(plan.root),
        "assets": [
            {
                "asset_id": asset.asset_id,
                "mime_type": asset.mime_type,
                "data_base64": base64.b64encode(asset.data).decode("ascii"),
            }
            for asset in plan.assets
        ],
        "css_rules": list(plan.css_rules),
        "scripts": list(plan.scripts),
        "compatibility_issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "location": issue.location,
            }
            for issue in plan.compatibility_issues
        ],
    }


def _projection_wire(projection: CanvasGatewayProjection) -> dict[str, Any]:
    """Return the bounded source-free shell state payload."""

    return {
        "selection": {
            "canvas_id": projection.scope.canvas_id,
            "revision_id": projection.scope.revision_id,
        },
        "options": [
            {
                "canvas_id": option.canvas_id,
                "revision_id": option.revision_id,
                "title": option.title,
            }
            for option in projection.options
        ],
        "metadata": {
            "title": projection.title,
            "sequence": projection.sequence,
            "parent_revision_id": projection.parent_revision_id,
            "source_bytes": projection.source_bytes,
            "content_sha256": projection.content_sha256,
            "origin_message_id": projection.origin_message_id,
            "origin_turn_id": projection.origin_turn_id,
            "temporary": projection.temporary,
        },
        "following": projection.following,
    }


def _render_node_wire(node: RenderNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "tag": node.tag,
        "attributes": [list(attribute) for attribute in node.attributes],
        "text": node.text,
        "children": [_render_node_wire(child) for child in node.children],
    }


def _error_response(code: str, status: int) -> web.Response:
    return web.json_response({"error": code}, status=status)


__all__ = [
    "BridgeConfirmationRequest",
    "BridgeConfirmationResponse",
    "CanvasBridgeSettlementLease",
    "CanvasGateway",
    "CanvasGatewayAuthority",
    "CanvasGatewayEvent",
    "CanvasGatewayLaunch",
    "CanvasGatewayScope",
    "CanvasSourceResponse",
]
