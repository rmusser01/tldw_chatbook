"""Generic async WebSocket transport for realtime voice sessions.

Deliberately provider-agnostic: carries no OpenAI (or any other provider)
wire-protocol knowledge -- just connect/send/receive/close over a raw
WebSocket. Provider-specific sessions (e.g. `openai_session.OpenAIRealtimeSession`)
build their wire protocol on top of `WsTransport`.

`websockets` is an OPTIONAL dependency (the `realtime` extra) and is
resolved through `Utils/optional_deps.py` at USE time, never imported at
module scope. Two reasons, and the second is why the module-scope import
that used to live here was wrong even though nothing crashed:

  * Repo policy: optional dependencies go through that one accessor, so
    availability is cached and reported in one place rather than being
    rediscovered by every importer's own try/except (PR #1350 review, Q2).
  * A baseline install that reaches this module -- e.g. a user who enabled
    the realtime engine in config without installing the extra -- deserves
    "install the realtime extra", not an `ImportError` traceback naming a
    package they have never heard of. Failing at use time is what lets the
    caller turn that into the loop's ordinary loud fallback.

This module is still only imported by a provider session module, which is
itself the lazy-import boundary documented in
`LLM_Calls/realtime/__init__.py` -- `import tldw_chatbook.LLM_Calls.realtime`
alone never reaches this file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...Utils.optional_deps import require_dependency
from ...Utils.tls_trust import ssl_context_for_transport

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    # Annotated from its DEFINING module rather than the top-level package:
    # the `realtime` extra's floor is `websockets>=14.0` (the release where
    # the asyncio client -- and with it `connect(additional_headers=...)`,
    # used below -- became what the top-level `connect` resolves to; 12/13's
    # legacy client takes `extra_headers` and raises TypeError on ours).
    # This module path is where `ClientConnection` lives across that whole
    # supported range.
    from websockets.asyncio.client import ClientConnection

#: Message shown when the extra is missing, naming the extra rather than
#: the package: `pip install websockets` alone would leave the audio
#: backends missing and fail one layer further in.
_MISSING_WEBSOCKETS = (
    "The realtime voice engine needs the 'realtime' extra. Install it with: "
    'pip install "tldw_chatbook[realtime]"'
)

#: Shown when websockets is installed but predates the asyncio client's
#: `additional_headers` (i.e. below the extra's declared floor). Named
#: explicitly because the raw failure is a bare TypeError about a keyword
#: argument, which reads as an app bug rather than a stale install.
_WEBSOCKETS_TOO_OLD = (
    "The installed 'websockets' is older than the realtime engine's floor "
    "(>=14.0): its connect() does not accept additional_headers. Upgrade "
    'with: pip install --upgrade "tldw_chatbook[realtime]"'
)


def _websockets():
    """Return the `websockets` module, or raise a message worth reading.

    Returns:
        The imported `websockets` module.

    Raises:
        ImportError: When the `realtime` extra is not installed.
    """
    try:
        return require_dependency("websockets", "realtime")
    except ImportError as exc:
        raise ImportError(_MISSING_WEBSOCKETS) from exc


class WsTransport:
    """Thin async WebSocket wrapper: connect, send JSON, receive-loop, close.

    Owns exactly one `websockets` client connection at a time. Not
    reusable across `connect()` calls after `close()` -- construct a new
    instance for a new connection.
    """

    def __init__(self) -> None:
        """Initialize a transport with no active connection."""
        self._ws: ClientConnection | None = None
        self._closed = False

    async def connect(self, url: str, headers: dict[str, str]) -> None:
        """Open the WebSocket connection.

        Args:
            url: The WebSocket URL to connect to (`ws://` or `wss://`).
            headers: Additional HTTP headers (e.g. `Authorization`) to send
                during the opening handshake.

        Returns:
            None.

        Raises:
            Exception: Whatever `websockets.connect` raises on failure to
                connect or complete the opening handshake (connection
                refused, DNS failure, handshake rejection, etc.).
        """
        websockets = _websockets()
        try:
            ssl_arg = ssl_context_for_transport()
            if ssl_arg is not None and not url.startswith("wss://"):
                # websockets rejects a non-None ssl argument for ws:// URIs.
                ssl_arg = None
            connect_kwargs: dict = {"additional_headers": headers}
            if ssl_arg is not None:
                connect_kwargs["ssl"] = ssl_arg
            self._ws = await websockets.connect(url, **connect_kwargs)
        except TypeError as exc:
            # Precisely the below-floor signature mismatch: the legacy
            # client takes `extra_headers`, so it rejects this call with a
            # bare "unexpected keyword argument" that names nothing a user
            # could act on.
            if "additional_headers" in str(exc):
                raise ImportError(_WEBSOCKETS_TOO_OLD) from exc
            raise
        self._closed = False

    async def send_json(self, obj: dict[str, Any]) -> None:
        """Serialize `obj` as JSON and send it over the connection.

        Args:
            obj: JSON-serializable payload to send as a single WebSocket
                text frame.

        Returns:
            None.

        Raises:
            RuntimeError: If called before `connect()` has succeeded.
            Exception: Whatever the underlying `websockets` send raises
                (e.g. `ConnectionClosed` if the peer has disconnected).
        """
        if self._ws is None:
            raise RuntimeError("WsTransport.send_json called before connect()")
        await self._ws.send(json.dumps(obj))

    async def recv_loop(self, on_event: Callable[[dict], None]) -> str:
        """Receive frames until the connection closes, decoding each as JSON
        and invoking `on_event` with the resulting dict.

        Args:
            on_event: Callback invoked once per decoded JSON message. Any
                exception it raises propagates out of this method -- callers
                that need per-callback isolation must isolate inside
                `on_event` itself (this transport has no knowledge of what
                `on_event` does).

        Returns:
            A short string describing why the loop ended: the WebSocket
            close reason if the peer supplied one, `"code=<n>"` if it
            supplied only a close code, or `"closed"` as a last resort.

        Raises:
            RuntimeError: If called before `connect()` has succeeded.
        """
        if self._ws is None:
            raise RuntimeError("WsTransport.recv_loop called before connect()")
        # Resolved once, into a local: this loop can only run after
        # `connect()` succeeded, so the module is necessarily importable
        # here, and looking it up inside the `except` clause would re-enter
        # the dependency accessor on every raised exception.
        connection_closed = _websockets().exceptions.ConnectionClosed
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        f"WsTransport.recv_loop: dropping non-JSON frame: "
                        f"op=recv_loop error={exc!r}"
                    )
                    continue
                on_event(event)
        except connection_closed:
            # Both clean (1000/1001) and abnormal closes resolve to a
            # returned reason string, per this method's contract -- never
            # an exception out of a plain "the peer hung up" condition.
            pass
        return self._close_reason()

    def _close_reason(self) -> str:
        """Build a short human-readable reason string from the connection's
        close code/reason, once it has closed.

        Returns:
            The peer-supplied close reason, `"code=<n>"` if only a code is
            available, or `"closed"` if neither is available.
        """
        if self._ws is None:
            return "closed"
        reason = getattr(self._ws, "close_reason", None)
        if reason:
            return reason
        code = getattr(self._ws, "close_code", None)
        if code is not None:
            return f"code={code}"
        return "closed"

    async def close(self) -> None:
        """Close the connection if open. Safe to call multiple times.

        Returns:
            None.
        """
        if self._ws is None or self._closed:
            return
        self._closed = True
        try:
            await self._ws.close()
        except Exception as exc:
            logger.warning(f"WsTransport.close: error closing socket: op=close error={exc!r}")
