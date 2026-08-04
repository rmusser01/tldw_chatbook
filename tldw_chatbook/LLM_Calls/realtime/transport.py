"""Generic async WebSocket transport for realtime voice sessions.

Deliberately provider-agnostic: carries no OpenAI (or any other provider)
wire-protocol knowledge -- just connect/send/receive/close over a raw
WebSocket. Provider-specific sessions (e.g. `openai_session.OpenAIRealtimeSession`)
build their wire protocol on top of `WsTransport`.

`websockets` is imported at module level here (not lazily) because this
module itself is only ever imported by a provider session module, which is
already the lazy-import boundary documented in
`LLM_Calls/realtime/__init__.py` -- `import tldw_chatbook.LLM_Calls.realtime`
alone never reaches this file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import websockets
from loguru import logger


class WsTransport:
    """Thin async WebSocket wrapper: connect, send JSON, receive-loop, close.

    Owns exactly one `websockets` client connection at a time. Not
    reusable across `connect()` calls after `close()` -- construct a new
    instance for a new connection.
    """

    def __init__(self) -> None:
        """Initialize a transport with no active connection."""
        self._ws: websockets.ClientConnection | None = None
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
        self._ws = await websockets.connect(url, additional_headers=headers)
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
        except websockets.exceptions.ConnectionClosed:
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
