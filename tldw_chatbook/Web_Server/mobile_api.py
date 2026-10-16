"""Mobile companion REST API + event stream under ``/api/v1/mobile``.

ADR-234, Phase 1b: task 4's thin authed HTTP wrappers and task 5's
``GET /api/v1/mobile/stream`` WebSocket over the Task-3
``MobileChatService``
(:mod:`tldw_chatbook.Web_Server.mobile_service`): every REST handler
parses one bounded JSON body (or query parameter), makes exactly one
service call, and answers with ``web.json_response`` plus
``Cache-Control: no-store``. Authentication and CSRF are NOT handled
here — the routes are registered among the authed serve routes (never in
``_public_auth_paths``), so the Phase-1a middleware enforces the session
cookie on every method and the CSRF header on every POST before any
handler runs; the stream route receives the middleware's websocket
treatment (GET upgrade + BOTH subprotocols ``csrf.<token>`` and
``chatbook-v1``) before ``handle_mobile_stream`` runs.

Error contract: an unknown session answers ``404 {"error":
"unknown_session"}`` and an unknown approval round answers ``200
{"ok": false}``; nothing on this surface may 5xx for a caller mistake.

Logging discipline (ADR-234): these handlers log nothing except one
transport-failure diagnostic on the stream route (session id + error
type only) — the service owns the rest of logging, and it logs session
ids and statuses only.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from functools import partial
from typing import Any

from loguru import logger

from ..Canvas.web_auth import WEBSOCKET_PROTOCOL
from ..config import get_cli_setting

__all__ = [
    "CLOSE_AUTH_EXPIRED",
    "CLOSE_ENGINE_ERROR",
    "IDLE_PING_SECONDS",
    "MAX_SEND_TEXT_CHARS",
    "MOBILE_MAX_BODY_BYTES",
    "MOBILE_STREAM_PATH",
    "build_mobile_routes",
    "mobile_surface_enabled",
]

#: Send-route text bound (characters): non-empty, at most 32 KiB of text.
MAX_SEND_TEXT_CHARS = 32768

#: Authed WebSocket path for the mobile event stream — registered, like
#: the REST table, only while the ``mobile_surface`` gate is on.
MOBILE_STREAM_PATH = "/api/v1/mobile/stream"

#: Seconds without an outbound frame before the stream sends one JSON
#: ``{"type": "ping"}`` idle frame. v1 ignores missing ``pong`` replies
#: — no liveness kill (ADR-234).
IDLE_PING_SECONDS = 30.0

#: Stream close code for an expired/revoked session: the client redirects
#: to ``/auth/login``, preserving the composer draft in sessionStorage.
CLOSE_AUTH_EXPIRED = 4401

#: Stream close code for an engine-side failure (the event subscription
#: itself broke — never a caller mistake).
CLOSE_ENGINE_ERROR = 1011

#: Largest accepted mobile JSON body (bytes). The send route allows up to
#: 32,768 CHARS of text, and a JSON encoder may spend six bytes per
#: character (every char a ``\\uXXXX`` escape), so the byte ceiling covers
#: that worst case plus envelope overhead. Served as the app-wide
#: ``client_max_size`` — Phase 1a's 16 KiB could not carry a full
#: 32 KiB text — while every non-mobile route keeps its own tighter body
#: guard.
MOBILE_MAX_BODY_BYTES = 200 * 1024

#: Small-body ceiling shared with the Phase-1a device endpoints.
_SMALL_BODY_BYTES = 4096

#: Route-layer page ceiling for the paginated reading routes (Phase 2b
#: rider, task-3 review): a page beyond 10,000 can only compute row
#: offsets far past any real table and, for pathological inputs, an
#: offset whose product exceeds SQLite's int64 binding range
#: (``OverflowError``). Rejected as the uniform ``400 invalid_page``
#: before the service is touched.
_MAX_PAGE = 10_000

_NO_STORE = {"Cache-Control": "no-store"}

#: Title used when a new-conversation POST carries no usable title.
_DEFAULT_SESSION_TITLE = "Mobile chat"

#: Allowed approval verdicts (the service contract's closed set).
_APPROVAL_DECISIONS = frozenset(
    {"approve_once", "approve_session", "always_allow", "deny"}
)


def mobile_surface_enabled() -> bool:
    """Read the ``[web_server] mobile_surface`` feature gate (default on).

    Fails closed like the Canvas gate: a config read that raises leaves
    the surface unregistered rather than serving it unconfigured.
    """
    try:
        return bool(get_cli_setting("web_server", "mobile_surface", default=True))
    except Exception:  # noqa: BLE001 - server feature gate fails closed
        return False


def build_mobile_routes(server: Any) -> list:
    """Build the authed ``/api/v1/mobile`` route table for one server.

    Handlers are partials over the server so they reach its lazily
    constructed mobile service (``server._get_mobile_service()``).
    """
    from aiohttp import web

    return [
        web.get(
            "/api/v1/mobile/conversations",
            partial(handle_conversations, server),
            name="mobile_conversations",
        ),
        web.post(
            "/api/v1/mobile/conversations",
            partial(handle_new_conversation, server),
            name="mobile_new_conversation",
        ),
        web.get(
            "/api/v1/mobile/conversations/{sid}/messages",
            partial(handle_messages, server),
            name="mobile_messages",
        ),
        web.post(
            "/api/v1/mobile/conversations/{sid}/messages",
            partial(handle_send, server),
            name="mobile_send",
        ),
        web.post(
            "/api/v1/mobile/conversations/{sid}/cancel",
            partial(handle_cancel, server),
            name="mobile_cancel",
        ),
        web.get(
            "/api/v1/mobile/approvals",
            partial(handle_approvals, server),
            name="mobile_approvals",
        ),
        web.post(
            "/api/v1/mobile/approvals/{round_id}",
            partial(handle_decide, server),
            name="mobile_decide",
        ),
        web.get(
            "/api/v1/mobile/notes",
            partial(handle_notes_page, server),
            name="mobile_notes",
        ),
        web.get(
            "/api/v1/mobile/notes/{note_id}",
            partial(handle_note_detail, server),
            name="mobile_note_detail",
        ),
        web.get(
            "/api/v1/mobile/library",
            partial(handle_library_page, server),
            name="mobile_library",
        ),
        web.get(
            "/api/v1/mobile/library/{media_id}",
            partial(handle_library_detail, server),
            name="mobile_library_detail",
        ),
        web.get(
            MOBILE_STREAM_PATH,
            partial(handle_mobile_stream, server),
            name="mobile_stream",
        ),
    ]


class _BodyTooLarge(Exception):
    """Internal marker: the request body exceeded the route's byte cap.

    Raised by :func:`_read_mobile_json` (handlers map it to the surface's
    uniform ``413 body_too_large`` JSON shape instead of aiohttp's
    plain-text ``HTTPRequestEntityTooLarge``).
    """


async def _read_mobile_json(request, *, max_bytes: int) -> dict:
    """Read one bounded JSON object body for the mobile endpoints.

    The Phase-1a ``_read_device_json`` guard (``Web_Server/serve.py``)
    generalized over the byte cap: the send route's text allowance cannot
    fit the device endpoints' fixed 4 KiB ceiling. Malformed or
    non-object bodies read as ``{}`` so per-field validation answers.

    Raises:
        _BodyTooLarge: The declared Content-Length or the received body
            exceeds ``max_bytes`` (the latter via aiohttp's
            ``client_max_size`` enforcement during the read).
    """
    from aiohttp import web

    if request.content_length is not None and request.content_length > max_bytes:
        raise _BodyTooLarge()
    try:
        payload = await request.json()
    except web.HTTPRequestEntityTooLarge as exc:
        raise _BodyTooLarge() from exc
    except (ValueError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _error(status: int, code: str):
    """Build the surface's uniform ``{"error": code}`` JSON response."""
    from aiohttp import web

    return web.json_response({"error": code}, status=status, headers=_NO_STORE)


async def handle_conversations(server: Any, request) -> Any:
    """List the mobile sessions, newest first."""
    from aiohttp import web

    service = server._get_mobile_service()
    return web.json_response({"sessions": service.sessions}, headers=_NO_STORE)


async def handle_new_conversation(server: Any, request) -> Any:
    """Create one mobile session; answers ``201 {"session": {...}}``."""
    from aiohttp import web

    service = server._get_mobile_service()
    try:
        payload = await _read_mobile_json(request, max_bytes=_SMALL_BODY_BYTES)
    except _BodyTooLarge:
        return _error(413, "body_too_large")
    title = payload.get("title")
    workspace_id = payload.get("workspace_id")
    if title is not None and not isinstance(title, str):
        return _error(400, "invalid_title")
    if workspace_id is not None and not isinstance(workspace_id, str):
        return _error(400, "invalid_workspace")
    usable_title = title if isinstance(title, str) and title.strip() else None
    session = await service.create_session(
        title=usable_title or _DEFAULT_SESSION_TITLE,
        workspace_id=workspace_id if isinstance(workspace_id, str) else None,
    )
    return web.json_response({"session": session}, status=201, headers=_NO_STORE)


async def handle_messages(server: Any, request) -> Any:
    """Return one session's transcript, optionally past a cursor."""
    from aiohttp import web

    service = server._get_mobile_service()
    after = request.query.get("after") or None
    try:
        result = service.session_messages(
            request.match_info["sid"], after_message_id=after
        )
    except KeyError:
        return _error(404, "unknown_session")
    except ValueError:
        return _error(400, "invalid_cursor")
    return web.json_response(result, headers=_NO_STORE)


async def handle_send(server: Any, request) -> Any:
    """Submit one draft; answers ``202 {"ok": true}`` once it is scheduled.

    Non-blocking (spec §4.2 / live-check round 1): ``service.send``
    pre-validates synchronously — unknown session ``404``, a turn already
    in flight ``409 send_refused`` — then schedules the turn on a
    background task and returns, because the engine's ``submit_draft``
    spans the whole run including any tool-approval wait. The phone
    observes streaming and the outcome via the WS stream (or the messages
    resync), including a submit-time refusal, which arrives as a
    ``run_failed {"status": "send_refused"}`` event.
    """
    from aiohttp import web
    from .mobile_service import MobileSendRefused

    service = server._get_mobile_service()
    try:
        payload = await _read_mobile_json(request, max_bytes=MOBILE_MAX_BODY_BYTES)
    except _BodyTooLarge:
        return _error(413, "body_too_large")
    text = payload.get("text")
    if not isinstance(text, str) or not text:
        return _error(400, "invalid_text")
    if len(text) > MAX_SEND_TEXT_CHARS:
        return _error(413, "text_too_large")
    try:
        await service.send(request.match_info["sid"], text)
    except KeyError:
        return _error(404, "unknown_session")
    except MobileSendRefused:
        return _error(409, "send_refused")
    return web.json_response({"ok": True}, status=202, headers=_NO_STORE)


async def handle_cancel(server: Any, request) -> Any:
    """Stop a session's active run; answers ``{"ok": bool}``."""
    from aiohttp import web

    service = server._get_mobile_service()
    try:
        stopped = service.cancel(request.match_info["sid"])
    except KeyError:
        return _error(404, "unknown_session")
    return web.json_response({"ok": bool(stopped)}, headers=_NO_STORE)


async def handle_approvals(server: Any, request) -> Any:
    """List the pending approval rounds across mobile sessions."""
    from aiohttp import web

    service = server._get_mobile_service()
    return web.json_response(
        {"approvals": service.list_approvals()}, headers=_NO_STORE
    )


async def handle_decide(server: Any, request) -> Any:
    """Resolve one approval round; answers ``{"ok": bool}``, never 5xx.

    An unknown or already-dead round is a reported ``{"ok": false}``, not
    an error: the phone may legitimately race a round that expired while
    the decision was in flight. Desktop-relayed rounds (ADR-236) route
    through the canvas control broker inside this await — the 2 s routing
    budget bounds the response.
    """
    from aiohttp import web

    service = server._get_mobile_service()
    try:
        payload = await _read_mobile_json(request, max_bytes=_SMALL_BODY_BYTES)
    except _BodyTooLarge:
        return _error(413, "body_too_large")
    decisions = payload.get("decisions")
    if not isinstance(decisions, Mapping) or not all(
        isinstance(key, str) and value in _APPROVAL_DECISIONS
        for key, value in decisions.items()
    ):
        return _error(400, "invalid_decisions")
    try:
        delivered = await service.decide_approval(
            request.match_info["round_id"], dict(decisions)
        )
    except KeyError:
        delivered = False
    return web.json_response({"ok": bool(delivered)}, headers=_NO_STORE)


def _parse_page(raw: str | None) -> int | None:
    """Read one 1-based ``?page=`` value; ``None`` when unusable.

    Missing or empty reads as the default page 1; anything that is not a
    decimal integer, is below 1, or exceeds ``_MAX_PAGE`` is a caller
    mistake (the uniform ``400 invalid_page``).
    """
    if raw is None or raw == "":
        return 1
    if not (raw.isascii() and raw.isdigit()):
        return None
    page = int(raw)
    return page if 1 <= page <= _MAX_PAGE else None


async def handle_notes_page(server: Any, request) -> Any:
    """List one page of notes; answers the service's page envelope."""
    from aiohttp import web

    page = _parse_page(request.query.get("page"))
    if page is None:
        return _error(400, "invalid_page")
    service = server._get_mobile_service()
    return web.json_response(service.notes_page(page=page), headers=_NO_STORE)


async def handle_note_detail(server: Any, request) -> Any:
    """Return one note's full row, or ``404 unknown_note``."""
    from aiohttp import web

    service = server._get_mobile_service()
    note = service.note_detail(request.match_info["note_id"])
    if note is None:
        return _error(404, "unknown_note")
    return web.json_response({"note": note}, headers=_NO_STORE)


async def handle_library_page(server: Any, request) -> Any:
    """List one page of media items; answers the service's page envelope."""
    from aiohttp import web

    page = _parse_page(request.query.get("page"))
    if page is None:
        return _error(400, "invalid_page")
    service = server._get_mobile_service()
    return web.json_response(service.library_page(page=page), headers=_NO_STORE)


async def handle_library_detail(server: Any, request) -> Any:
    """Return one media item with transcripts, or ``404 unknown_item``.

    The path segment must be a plain decimal integer — anything else is
    the uniform ``400 invalid_item`` before the service is touched.
    """
    from aiohttp import web

    raw_media_id = request.match_info["media_id"]
    if not (raw_media_id.isascii() and raw_media_id.isdigit()):
        return _error(400, "invalid_item")
    service = server._get_mobile_service()
    item = service.library_detail(int(raw_media_id))
    if item is None:
        return _error(404, "unknown_item")
    return web.json_response({"item": item}, headers=_NO_STORE)


class _StreamEngineError(Exception):
    """Internal marker: the service's event subscription failed (1011)."""


async def handle_mobile_stream(server: Any, request) -> Any:
    """Stream the service's events to one authenticated browser socket.

    The auth middleware has already authenticated the upgrade against
    BOTH subprotocols (``csrf.<session-csrf>`` proof + ``chatbook-v1``);
    this handler drains one ``service.events()`` subscription per socket,
    interleaved with the client's ``{"type": "ping"}`` frames (answered
    with a JSON ``pong``) and one ``{"type": "ping"}`` idle frame per
    ``IDLE_PING_SECONDS`` without outbound traffic. Inbound frames other
    than a ping are ignored in v1. Close codes: ``4401`` when the session
    is revoked mid-stream (the client redirects to ``/auth/login``),
    ``1011`` when the event subscription itself fails.
    """
    from aiohttp import WSMsgType, web

    # Receive-loop message types that end the stream (client close or a
    # broken transport); everything else is either a TEXT frame handled
    # below or a protocol frame aiohttp's autoping already consumed.
    close_types = {
        WSMsgType.CLOSE,
        WSMsgType.CLOSING,
        WSMsgType.CLOSED,
        WSMsgType.ERROR,
    }
    websocket = web.WebSocketResponse(
        heartbeat=15,
        protocols=(WEBSOCKET_PROTOCOL,),
    )
    await websocket.prepare(request)
    session = request.get("chatbook_browser_session")
    unregister_channel = (
        server._web_auth.register_channel(
            session,
            lambda: asyncio.create_task(
                websocket.close(code=CLOSE_AUTH_EXPIRED)
            ),
        )
        if session is not None
        else (lambda: None)
    )
    expiry_task = (
        asyncio.create_task(server._expire_websocket_session(session))
        if session is not None
        else None
    )
    events = None
    receive_task: asyncio.Task | None = None
    event_task: asyncio.Task | None = None
    close_code: int | None = None
    try:
        events = server._get_mobile_service().events()
        loop = asyncio.get_running_loop()
        last_send = loop.time()
        receive_task = asyncio.create_task(websocket.receive())
        while True:
            if event_task is None:
                event_task = asyncio.create_task(events.__anext__())
            idle_for = loop.time() - last_send
            done, _pending = await asyncio.wait(
                (receive_task, event_task),
                timeout=max(0.0, IDLE_PING_SECONDS - idle_for),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await websocket.send_json({"type": "ping"})
                last_send = loop.time()
                continue
            if receive_task in done:
                message = receive_task.result()
                if message.type in close_types:
                    break
                if message.type == WSMsgType.TEXT:
                    if session is not None:
                        server._web_auth.touch_session(session)
                    try:
                        frame = message.json()
                    except ValueError:
                        frame = None
                    if isinstance(frame, dict) and frame.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        last_send = loop.time()
                receive_task = asyncio.create_task(websocket.receive())
            if event_task in done:
                try:
                    event = event_task.result()
                except StopAsyncIteration:
                    event_task = None
                    break  # service closed the stream; end politely
                except Exception as error:
                    raise _StreamEngineError() from error
                event_task = None
                await websocket.send_json(event.to_wire())
                last_send = loop.time()
    except asyncio.CancelledError:
        pass  # server shutdown: drain the subscriber below
    except _StreamEngineError:
        close_code = CLOSE_ENGINE_ERROR
    except Exception as error:  # noqa: BLE001 - transport failure; the socket is gone
        # One diagnostic line, ids/types only (no content, no credentials),
        # so a dead socket is distinguishable from a silent handler.
        logger.warning(
            "mobile stream transport failure session_id={} error_type={}",
            getattr(session, "session_id", None),
            type(error).__name__,
        )
    finally:
        for task in (receive_task, event_task, expiry_task):
            if task is not None:
                task.cancel()
        pending_tasks = [
            task
            for task in (receive_task, event_task, expiry_task)
            if task is not None
        ]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        if events is not None:
            # Runs Task 3's generator finally: ``_subscribers.discard``.
            with contextlib.suppress(Exception):
                await events.aclose()
        unregister_channel()
        with contextlib.suppress(Exception):
            if close_code is None:
                await websocket.close()
            else:
                await websocket.close(code=close_code)
    return websocket
