"""Mobile companion REST API + event stream under ``/api/v1/mobile``.

ADR-244: canonical Console protocol 1 routes exact native tickets, commands,
receipts and bounded snapshots through the retained host. Saved history and
Library reads use noncreating native read-only owners. Legacy mutations return
reload_required. Authentication, same-origin and CSRF are enforced by serve
middleware before handlers; websocket upgrades require both csrf.<token> and
chatbook-v1 subprotocols. Private child control protocol 3 is independent.

Logging discipline (ADR-234): these handlers log nothing except one
transport-failure diagnostic on the stream route (session id + error
type only) — the service owns the rest of logging, and it logs session
ids and statuses only.
"""

from __future__ import annotations

import asyncio
import contextlib
from functools import partial
from typing import Any, Literal
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError

from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget

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

#: Conversations-route page size (Task 2, #3128 minor 7): default
#: ``per_page`` and its ceiling, mirroring the reading routes' bounded
#: page parameters. A ``per_page`` beyond the ceiling is the uniform
#: ``400 invalid_per_page`` before the service is touched.
_DEFAULT_PER_PAGE = 50
_MAX_PER_PAGE = 100

_NO_STORE = {"Cache-Control": "no-store"}


def mobile_surface_enabled() -> bool:
    """Read the ``[web_server] mobile_surface`` feature gate (default on).

    Fails closed like the Canvas gate: a config read that raises leaves
    the surface unregistered rather than serving it unconfigured.
    """
    try:
        return bool(get_cli_setting("web_server", "mobile_surface", default=True))
    except Exception:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return False


def build_mobile_routes(server: Any) -> list:
    """Build the authed ``/api/v1/mobile`` route table for one server.

    Handlers are partials over the server so they reach its lazily
    constructed mobile service (``server._get_mobile_service()``).
    """
    from aiohttp import web

    return [
        web.post("/api/v1/mobile/console/host", partial(handle_console_host, server)),
        web.get(
            "/api/v1/mobile/console/snapshot", partial(handle_console_snapshot, server)
        ),
        web.post(
            "/api/v1/mobile/commands/tickets", partial(handle_command_ticket, server)
        ),
        web.post("/api/v1/mobile/commands", partial(handle_command, server)),
        web.get(
            "/api/v1/mobile/commands/{command_id}",
            partial(handle_command_query, server),
        ),
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


class _HostBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol_version: Literal[1]


class _CommandTicketBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol_version: Literal[1]
    action: Literal[
        "create",
        "open",
        "send",
        "steer",
        "redirect",
        "stop",
        "subagent_guidance",
        "subagent_stop",
        "decision",
    ]
    target: ConsoleTarget | ConsoleHostIdentity
    payload: dict


class _CommandBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol_version: Literal[1]
    ticket: StrictStr = Field(min_length=1, max_length=16384)
    payload: dict


class _SnapshotQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol_version: Literal[1]
    target: ConsoleTarget | None = None
    cursor: StrictStr | None = Field(default=None, max_length=2048)
    receipt_cursor: StrictStr | None = Field(default=None, max_length=2048)
    limit: StrictInt = Field(default=50, ge=1, le=100)


async def _command_error(error):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        MobileConsolidationRefused,
    )
    from tldw_chatbook.Canvas.control_protocol import ControlProtocolError

    from .mobile_service import MobileOwnerUnavailable

    if isinstance(error, MobileConsolidationRefused):
        from aiohttp import web

        return web.json_response(
            {
                "error": "cutover_required",
                "reason": str(error),
                "operation_id": error.operation_id,
                "review_path": str(error.control_root) if error.control_root else None,
            },
            status=409,
            headers=_NO_STORE,
        )
    if isinstance(error, ControlProtocolError):
        if error.code == "command_expired":
            return _error(410, "command_expired")
        if error.code == "invalid_console_request":
            return _error(409, "command_refused")
        return _error(503, "console_unavailable")
    if isinstance(error, (MobileOwnerUnavailable, TimeoutError)):
        return _error(503, "console_unavailable")
    if isinstance(error, (ValueError, TypeError)):
        return _error(409, "command_refused")
    return _error(503, "console_unavailable")


async def _command_body(request, model):
    value = await _read_mobile_json(request, max_bytes=MOBILE_MAX_BODY_BYTES)
    if type(value.get("protocol_version")) is not int or value["protocol_version"] != 1:
        raise RuntimeError("reload_required")
    return model.model_validate(value)


async def handle_console_host(server, request):
    from aiohttp import web

    try:
        await _command_body(request, _HostBody)
        host = await server._get_mobile_service().bootstrap_host()
        return web.json_response({"host": host.model_dump()}, headers=_NO_STORE)
    except _BodyTooLarge:
        return _error(413, "body_too_large")
    except ValidationError:
        return _error(400, "invalid_command")
    except RuntimeError as error:
        if str(error) == "reload_required":
            return _error(409, "reload_required")
        return await _command_error(error)
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


async def handle_command_ticket(server, request):
    from aiohttp import web

    try:
        body = await _command_body(request, _CommandTicketBody)
        result = await server._get_mobile_service().issue_command_ticket(
            body.action, body.target, body.payload
        )
        return web.json_response(result, headers=_NO_STORE)
    except _BodyTooLarge:
        return _error(413, "body_too_large")
    except ValidationError:
        return _error(400, "invalid_command")
    except RuntimeError as error:
        if str(error) == "reload_required":
            return _error(409, "reload_required")
        return await _command_error(error)
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


async def handle_command(server, request):
    from aiohttp import web

    try:
        body = await _command_body(request, _CommandBody)
        receipt = await server._get_mobile_service().submit_command(
            body.ticket, body.payload
        )
        return web.json_response(
            receipt.public(),
            status=202 if receipt.state in {"pending", "accepted", "queued"} else 200,
            headers=_NO_STORE,
        )
    except _BodyTooLarge:
        return _error(413, "body_too_large")
    except ValidationError:
        return _error(400, "invalid_command")
    except RuntimeError as error:
        if str(error) == "reload_required":
            return _error(409, "reload_required")
        return await _command_error(error)
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


async def handle_command_query(server, request):
    from aiohttp import web

    from tldw_chatbook.Canvas.limits import validate_opaque_identifier

    try:
        validate_opaque_identifier(request.match_info["command_id"])
        receipt = await server._get_mobile_service().query_command(
            request.match_info["command_id"]
        )
        if receipt is None:
            return _error(404, "unknown_command")
        return web.json_response(receipt.public(), headers=_NO_STORE)
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


async def handle_console_snapshot(server, request):
    import json

    from aiohttp import web

    try:
        data = dict(request.query)
        if data.pop("protocol_version", None) != "1":
            return _error(409, "reload_required")
        if "target" in data:
            data["target"] = json.loads(data["target"])
        if "limit" in data:
            data["limit"] = int(data["limit"])
        body = _SnapshotQuery.model_validate(dict(protocol_version=1, **data))
        result = await server._get_mobile_service().snapshot(
            **body.model_dump(exclude={"protocol_version", "target"}),
            target=body.target,
        )
        return web.json_response(result, headers=_NO_STORE)
    except (ValueError, TypeError):
        return _error(400, "invalid_snapshot")
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


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


async def _read_service(server, method, *args, **kwargs):
    from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

    service = server._get_mobile_service()
    return await asyncio.to_thread(
        run_finite_local_worker, getattr(service, method), *args, **kwargs
    )


async def handle_conversations(server: Any, request) -> Any:
    """List one page of the mobile sessions, newest first.

    Task 2 (#3128 minor 7): the route joins the reading routes' paginated
    contract — ``?page=``/``?per_page=`` (defaults 1/50, per_page capped
    at 100) with the same ``{"items", "page", "has_more"}`` envelope and
    the same ``400 invalid_page`` validation via the shared parser.
    """
    from aiohttp import web

    page = _parse_page(request.query.get("page"))
    if page is None:
        return _error(400, "invalid_page")
    per_page = _parse_per_page(request.query.get("per_page"))
    if per_page is None:
        return _error(400, "invalid_per_page")
    try:
        result = await _read_service(
            server, "sessions_page", page=page, per_page=per_page
        )
        return web.json_response(result, headers=_NO_STORE)
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


async def handle_new_conversation(server: Any, request) -> Any:
    """Refuse retired mutation envelopes before native effects."""
    return _error(409, "reload_required")


async def handle_messages(server: Any, request) -> Any:
    """Read saved canonical history without hydration or host startup."""
    from aiohttp import web

    try:
        allowed = {
            "message_offset",
            "message_limit",
            "max_chars",
            "message_id",
            "char_start",
        }
        if set(request.query) - allowed:
            return _error(400, "invalid_cursor")
        data = dict(request.query)
        for key, maximum in (
            ("message_offset", 1000000),
            ("message_limit", 100),
            ("max_chars", 8000),
            ("char_start", 100000000),
        ):
            if key in data:
                if not data[key].isascii() or not data[key].isdigit():
                    return _error(400, "invalid_cursor")
                data[key] = int(data[key])
                if (
                    not 0 <= data[key] <= maximum
                    or key in {"message_limit", "max_chars"}
                    and data[key] == 0
                ):
                    return _error(400, "invalid_cursor")
        # Compatibility fakes may expose an async method; production reads are synchronous.
        service = server._get_mobile_service()
        result = await service.session_messages(request.match_info["sid"], **data)
        return web.json_response(result, headers=_NO_STORE)
    except KeyError:
        return _error(404, "unknown_session")
    except ValueError:
        return _error(400, "invalid_cursor")
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


async def handle_send(server: Any, request) -> Any:
    """Refuse retired mutation envelopes before native effects."""
    return _error(409, "reload_required")


async def handle_cancel(server: Any, request) -> Any:
    """Refuse retired mutation envelopes before native effects."""
    return _error(409, "reload_required")


async def handle_approvals(server: Any, request) -> Any:
    """List the pending approval rounds across mobile sessions."""

    return _error(409, "reload_required")


async def handle_decide(server: Any, request) -> Any:
    """Refuse retired mutation envelopes before native effects."""
    return _error(409, "reload_required")


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


def _parse_per_page(raw: str | None) -> int | None:
    """Read one ``?per_page=`` value; ``None`` when unusable.

    Same shape as :func:`_parse_page`: missing or empty reads as the
    default ``_DEFAULT_PER_PAGE``; anything that is not a decimal integer,
    is below 1, or exceeds ``_MAX_PER_PAGE`` is a caller mistake (the
    uniform ``400 invalid_per_page``).
    """
    if raw is None or raw == "":
        return _DEFAULT_PER_PAGE
    if not (raw.isascii() and raw.isdigit()):
        return None
    per_page = int(raw)
    return per_page if 1 <= per_page <= _MAX_PER_PAGE else None


async def handle_notes_page(server: Any, request) -> Any:
    """List one page of notes; answers the service's page envelope."""
    from aiohttp import web

    page = _parse_page(request.query.get("page"))
    if page is None:
        return _error(400, "invalid_page")
    try:
        result = await _read_service(server, "notes_page", page=page)
        return web.json_response(result, headers=_NO_STORE)
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


async def handle_note_detail(server: Any, request) -> Any:
    """Return one note's full row, or ``404 unknown_note``."""
    from aiohttp import web

    try:
        note = await _read_service(server, "note_detail", request.match_info["note_id"])
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)
    if note is None:
        return _error(404, "unknown_note")
    return web.json_response({"note": note}, headers=_NO_STORE)


async def handle_library_page(server: Any, request) -> Any:
    """List one page of media items; answers the service's page envelope."""
    from aiohttp import web

    page = _parse_page(request.query.get("page"))
    if page is None:
        return _error(400, "invalid_page")
    try:
        result = await _read_service(server, "library_page", page=page)
        return web.json_response(result, headers=_NO_STORE)
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)


async def handle_library_detail(server: Any, request) -> Any:
    """Return one media item with transcripts, or ``404 unknown_item``.

    The path segment must be a plain decimal integer — anything else is
    the uniform ``400 invalid_item`` before the service is touched.
    """
    from aiohttp import web

    raw_media_id = request.match_info["media_id"]
    if not (raw_media_id.isascii() and raw_media_id.isdigit()):
        return _error(400, "invalid_item")
    try:
        item = await _read_service(server, "library_detail", int(raw_media_id))
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
        return await _command_error(error)
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
    is revoked mid-stream (the client redirects to ``/auth/login``);
    ``1011`` for every engine-side end — the event subscription failing,
    the service/engine failing to start, and the handler's shutdown
    cancellation — so the phone reads it as a transport failure and
    reconnects with backoff rather than treating the stream as ended.
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
        max_msg_size=_SMALL_BODY_BYTES,
    )
    await websocket.prepare(request)
    session = request.get("chatbook_browser_session")
    unregister_channel = (
        server._web_auth.register_channel(
            session,
            lambda: asyncio.create_task(websocket.close(code=CLOSE_AUTH_EXPIRED)),
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
    subscription_id = uuid4().hex
    receive_task: asyncio.Task | None = None
    event_task: asyncio.Task | None = None
    close_code: int | None = None
    try:
        try:
            events = server._get_mobile_service().events(client_id=subscription_id)
        except Exception as error:
            # Engine-start failure (service construction / relay attach):
            # engine-side, never a caller mistake — 1011, not the generic
            # transport-failure normal close (#3128 minor 7).
            raise _StreamEngineError() from error
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
                    elif (
                        isinstance(frame, dict)
                        and set(frame)
                        == {"type", "protocol_version", "target", "answerable"}
                        and frame["type"] == "answerable"
                    ):
                        if (
                            type(frame["protocol_version"]) is int
                            and frame["protocol_version"] == 1
                            and type(frame["answerable"]) is bool
                        ):
                            try:
                                await server._get_mobile_service().set_answerable(
                                    subscription_id,
                                    ConsoleTarget.model_validate(frame["target"]),
                                    frame["answerable"],
                                )
                            except (ValueError, RuntimeError):
                                await websocket.send_json(
                                    {
                                        "type": "reset",
                                        "payload": {"reset_required": True},
                                    }
                                )
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
        # Server shutdown cancels the handler mid-drain. The stream ends
        # for an engine-side reason, not a client one, so close 1011 —
        # the phone reconnects with backoff instead of reading a normal
        # close as a politely-ended stream (#3128 minor 7). The subscriber
        # still drains below.
        close_code = CLOSE_ENGINE_ERROR
    except _StreamEngineError:
        close_code = CLOSE_ENGINE_ERROR
    except Exception as error:  # noqa: BLE001 - bounded public failure or independent stream cleanup.
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
            task for task in (receive_task, event_task, expiry_task) if task is not None
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
