"""Mobile companion REST API under ``/api/v1/mobile`` (ADR-234, task 4).

Thin authed HTTP wrappers over the Task-3 ``MobileChatService``
(:mod:`tldw_chatbook.Web_Server.mobile_service`): every handler parses one
bounded JSON body (or query parameter), makes exactly one service call,
and answers with ``web.json_response`` plus ``Cache-Control: no-store``.
Authentication and CSRF are NOT handled here — the routes are registered
among the authed serve routes (never in ``_public_auth_paths``), so the
Phase-1a middleware enforces the session cookie on every method and the
CSRF header on every POST before any handler runs.

Error contract: an unknown session answers ``404 {"error":
"unknown_session"}`` and an unknown approval round answers ``200
{"ok": false}``; nothing on this surface may 5xx for a caller mistake.

Logging discipline (ADR-234): these handlers log nothing — the service
owns logging, and it logs session ids and statuses only.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from ..config import get_cli_setting

__all__ = [
    "MAX_SEND_TEXT_CHARS",
    "MOBILE_MAX_BODY_BYTES",
    "build_mobile_routes",
    "mobile_surface_enabled",
]

#: Send-route text bound (characters): non-empty, at most 32 KiB of text.
MAX_SEND_TEXT_CHARS = 32768

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
    ]


async def _read_mobile_json(request, *, max_bytes: int) -> dict:
    """Read one bounded JSON object body for the mobile endpoints.

    The Phase-1a ``_read_device_json`` guard (``Web_Server/serve.py``)
    generalized over the byte cap: the send route's text allowance cannot
    fit the device endpoints' fixed 4 KiB ceiling. Malformed or
    non-object bodies read as ``{}`` so per-field validation answers.
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
    payload = await _read_mobile_json(request, max_bytes=_SMALL_BODY_BYTES)
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
    """Submit one draft; answers ``202 {"ok": true}`` once the turn is in.

    The phone learns the turn's streaming and outcome by polling the
    messages route (the service's event stream feeds the Phase-1b WS
    surface, not this REST table).
    """
    from aiohttp import web
    from .mobile_service import MobileSendRefused

    service = server._get_mobile_service()
    payload = await _read_mobile_json(request, max_bytes=MOBILE_MAX_BODY_BYTES)
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
    the decision was in flight.
    """
    from aiohttp import web

    service = server._get_mobile_service()
    payload = await _read_mobile_json(request, max_bytes=_SMALL_BODY_BYTES)
    decisions = payload.get("decisions")
    if not isinstance(decisions, Mapping) or not all(
        isinstance(key, str) and value in _APPROVAL_DECISIONS
        for key, value in decisions.items()
    ):
        return _error(400, "invalid_decisions")
    try:
        delivered = service.decide_approval(
            request.match_info["round_id"], dict(decisions)
        )
    except KeyError:
        delivered = False
    return web.json_response({"ok": bool(delivered)}, headers=_NO_STORE)
