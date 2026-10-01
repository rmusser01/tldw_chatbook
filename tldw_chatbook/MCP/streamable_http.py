"""Bounded direct Streamable HTTP below the existing MCP client authority.

No redirects, proxies, TLS weakening, automatic invocation retries or legacy
HTTP+SSE endpoint fallback. Legacy GET only resumes an identified response.
"""

from __future__ import annotations

import asyncio
import json
from itertools import count
from types import SimpleNamespace
from typing import Any

import httpx
from loguru import logger

from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
    ProducerLifetime,
    producer_call,
)

from .activation import guarded
from .client import _JSONRPCError, _StdioJSONRPCConnection
from .local_store import TransportProfile
from .protocol_profiles import header_value, protocol_profile, tool_header_paths
from .tool_results import (
    MAX_OUTPUT_LINE_BYTES,
    MCPDispatchObservation,
    decode_protocol_frame,
)

MAX_HTTP_BYTES = MAX_OUTPUT_LINE_BYTES
MAX_SSE_EVENTS = 1024
MAX_RESUMES = 3


class StreamableHTTPConnection(_StdioJSONRPCConnection):
    """HTTP I/O with the same catalog and complete-result normalization as stdio."""

    def __init__(
        self,
        profile: TransportProfile,
        *,
        client_name: str = "tldw_chatbook_client",
        credential_service=None,
    ) -> None:
        if profile.transport != "streamable_http":
            raise ValueError("mcp_profile_invalid")
        self._producer_lifetime = ProducerLifetime()
        self.profile = protocol_profile(profile.protocol_version)
        self._allow_negotiation = True
        self.url = profile.url
        self._credential_reference = profile.credential_reference
        self._credential_generation = profile.credential_generation
        self._credential_service = credential_service
        self.client_name = client_name
        self.request_timeout_seconds = 10.0
        self.protocol_version = ""
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self._session_id: str | None = None
        self._request_ids = count(1)
        self._closed = False
        self._requests: dict[asyncio.Task, asyncio.Future] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_complete = False
        self._tool_headers: dict[str, tuple] = {}
        self._on_transport_failure = None
        self._http = httpx.AsyncClient(
            follow_redirects=False, trust_env=False, timeout=10.0
        )
        self.diagnostics: set[str] = set()

    async def list_tools(self) -> SimpleNamespace:
        response = await super().list_tools()
        if not self.profile.modern:
            return response
        accepted, bindings = [], {}
        for tool in response.tools:
            try:
                bindings[tool.name] = tool_header_paths(tool.inputSchema)
            except ValueError:
                self.diagnostics.add("mcp_header_schema_invalid")
                continue
            accepted.append(tool)
        self._tool_headers = bindings
        return SimpleNamespace(tools=accepted)

    async def _headers(
        self, method: str | None = None, params: dict | None = None
    ) -> dict[str, str | bytes]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        if self.profile.modern:
            headers["MCP-Protocol-Version"] = self.profile.version
            if method is not None:
                headers["Mcp-Method"] = method
                if method in {"tools/call", "resources/read", "prompts/get"}:
                    headers["Mcp-Name"] = header_value(
                        (params or {}).get(
                            "uri" if method == "resources/read" else "name"
                        )
                    )
                if method == "tools/call":
                    name = (params or {}).get("name")
                    if name not in self._tool_headers:
                        raise ValueError("mcp_tool_unavailable")
                    for path, header, kind in self._tool_headers[name]:
                        value = (params or {}).get("arguments", {})
                        for part in path:
                            value = value.get(part) if isinstance(value, dict) else None
                        if value is None:
                            continue
                        expected = {"string": str, "integer": int, "boolean": bool}[
                            kind
                        ]
                        if type(value) is not expected:
                            raise ValueError("mcp_header_value_invalid")
                        headers["Mcp-Param-" + header] = header_value(value)
        else:
            if self.protocol_version and self.profile.version != "2025-03-26":
                headers["MCP-Protocol-Version"] = self.profile.version
            if self._session_id is not None:
                headers["Mcp-Session-Id"] = self._session_id
        if self._credential_reference is not None:
            from .credential_bindings import CredentialError, endpoint_origin

            if self._credential_service is None:
                raise CredentialError("credential_missing")
            # Only this boundary sees secrets; never cache them across requests.
            async with asyncio.timeout(self.request_timeout_seconds):
                resolved = await self._credential_service.resolve_async(
                    self._credential_reference,
                    self._credential_generation,
                    endpoint_origin(self.url),
                )
            # RFC 9110 obs-text is octets. Host mapping supports Latin-1 exactly,
            # refuses other Unicode, and never rewrites/re-encodes values silently.
            headers.update(
                {name: value.encode("latin-1") for name, value in resolved.items()}
            )
        return headers

    @guarded
    @producer_call
    async def request(
        self,
        method: str,
        params: dict | None = None,
        *,
        timeout_seconds: float | None = None,
        _dispatch: MCPDispatchObservation | None = None,
    ) -> dict:
        if self._closed:
            raise RuntimeError("mcp_connection_closed")
        request_id = next(self._request_ids)
        params = self.profile.params(params or {}, self.client_name)
        body = self._encode(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        task = asyncio.current_task()
        drained = asyncio.get_running_loop().create_future()
        self._requests[task] = drained
        failed = False
        exchange_started = False
        try:
            async with asyncio.timeout(
                timeout_seconds
                if timeout_seconds is not None
                else self.request_timeout_seconds
            ):
                headers = await self._headers(method, params)

                # HTTPX trace fires immediately before request headers are written,
                # after DNS/connect/TLS. Outer cancellation never creates certainty.
                async def trace(event, info):
                    if (
                        event.endswith("send_request_headers.started")
                        and _dispatch is not None
                    ):
                        _dispatch.state = "uncertain"

                exchange_started = True
                result = await self._exchange(body, headers, request_id, method, trace)
                return result
        except (TimeoutError, asyncio.CancelledError):
            if (
                exchange_started
                and not self.profile.modern
                and method != "initialize"
                and not self._closed
            ):
                try:
                    await asyncio.wait_for(
                        self.notify(
                            "notifications/cancelled", {"requestId": request_id}
                        ),
                        timeout=1.0,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("MCP cancellation notification could not be sent")
            raise
        except _JSONRPCError:
            raise
        except Exception:
            # Header/credential refusal precedes transport I/O and must not kill
            # a healthy connection or imply an uncertain remote invocation.
            failed = exchange_started
            raise
        finally:
            self._requests.pop(task, None)
            drained.set_result(None)
            if (
                (failed or "mcp_catalog_changed" in self.diagnostics)
                and self.protocol_version
                and self._on_transport_failure is not None
            ):
                await self._on_transport_failure()

    @staticmethod
    def _encode(payload: dict) -> bytes:
        body = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        if len(body) > MAX_HTTP_BYTES:
            raise ValueError("mcp_request_too_large")
        return body

    async def _exchange(self, body, headers, request_id, method, trace) -> dict:
        last_id = None
        retry = 0.0
        total = 0
        for attempt in range(MAX_RESUMES + 1):
            verb = "GET" if attempt else "POST"
            if attempt:
                await asyncio.sleep(retry)
                headers = {**(await self._headers()), "Last-Event-ID": last_id}
            async with self._http.stream(
                verb,
                self.url,
                content=None if attempt else body,
                headers=headers,
                extensions={"trace": trace},
            ) as response:
                self._status(response)
                if (
                    response.headers.get("content-encoding", "identity").lower()
                    != "identity"
                ):
                    raise ValueError("mcp_content_encoding_unsupported")
                if method == "initialize" and not attempt:
                    session = response.headers.get("mcp-session-id")
                    if session is not None:
                        if (
                            not session
                            or len(session) > 1024
                            or any(not 33 <= ord(c) <= 126 for c in session)
                        ):
                            raise ValueError("mcp_session_invalid")
                        self._session_id = session
                media = (
                    response.headers.get("content-type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                if media == "application/json":
                    raw = bytearray()
                    async for chunk in response.aiter_raw():
                        total += len(chunk)
                        if total > MAX_HTTP_BYTES:
                            raise ValueError("mcp_response_too_large")
                        raw.extend(chunk)
                    payload = decode_protocol_frame(bytes(raw))
                    if not isinstance(payload, dict):
                        raise ValueError("mcp_protocol_invalid")
                    return await self._message(
                        payload, request_id, allow_notification=False
                    )
                if media != "text/event-stream":
                    raise ValueError("mcp_media_type_invalid")
                buffer, data = bytearray(), []
                terminal = None
                event_name = ""
                events = 0
                skip_lf = False
                async for chunk in response.aiter_raw():
                    total += len(chunk)
                    if total > MAX_HTTP_BYTES:
                        raise ValueError("mcp_response_too_large")
                    buffer.extend(chunk)
                    if skip_lf:
                        if buffer.startswith(b"\n"):
                            del buffer[:1]
                        skip_lf = False
                    while b"\n" in buffer or b"\r" in buffer:
                        end = min(
                            index
                            for marker in (b"\n", b"\r")
                            if (index := buffer.find(marker)) >= 0
                        )
                        line = buffer[:end]
                        was_cr = buffer[end] == 13
                        del buffer[: end + 1]
                        if was_cr:
                            if buffer.startswith(b"\n"):
                                del buffer[:1]
                            elif not buffer:
                                skip_lf = True
                        if not line:
                            events += 1
                            if events > MAX_SSE_EVENTS:
                                raise ValueError("mcp_event_limit")
                            if event_name not in ("", "message"):
                                raise ValueError("mcp_event_invalid")
                            if data and b"\n".join(data):
                                messages = decode_protocol_frame(b"\n".join(data))
                                if isinstance(messages, list):
                                    if not self.profile.batches or not messages:
                                        raise ValueError("mcp_protocol_invalid")
                                else:
                                    messages = [messages]
                                for message in messages:
                                    value = await self._message(
                                        message, request_id, allow_notification=True
                                    )
                                    if value is not None:
                                        if terminal is not None:
                                            raise ValueError("mcp_duplicate_response")
                                        terminal = value
                            data, event_name = [], ""
                        elif not line.startswith(b":"):
                            field, _, value = line.partition(b":")
                            value = value.removeprefix(b" ")
                            if field == b"data":
                                data.append(bytes(value))
                            elif field == b"event":
                                event_name = value.decode("utf-8")
                            elif field == b"id":
                                last_id = value.decode("utf-8")
                                if "\0" in last_id or len(last_id) > 1024:
                                    raise ValueError("mcp_event_invalid")
                            elif field == b"retry" and value.isdigit():
                                retry = int(value) / 1000
                    if terminal is not None:
                        if buffer or data:
                            raise ValueError("mcp_event_invalid")
                        return terminal
                if buffer or data:
                    raise ValueError("mcp_event_invalid")
                if terminal is not None:
                    return terminal
                if self.profile.modern or not last_id:
                    raise ValueError("mcp_response_incomplete")
        raise ValueError("mcp_resume_limit")

    async def _message(
        self, payload: dict, request_id: int, *, allow_notification: bool
    ) -> dict | None:
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
            raise ValueError("mcp_protocol_invalid")
        if "method" in payload:
            if not allow_notification or not isinstance(payload["method"], str):
                raise ValueError("mcp_protocol_invalid")
            if "id" in payload:
                if self.profile.modern:
                    raise ValueError("mcp_server_request_unsupported")
                reply = {"jsonrpc": "2.0", "id": payload["id"]}
                if payload["method"] == "ping":
                    reply["result"] = {}
                else:
                    reply["error"] = {"code": -32601, "message": "Method not supported"}
                await self._post_oneway(reply)
            elif payload["method"].endswith("/list_changed"):
                self.diagnostics.add("mcp_catalog_changed")
                # New discovery is required before another tool can be sent.
                self._tool_headers.clear()
            return None
        if type(payload.get("id")) is not int or payload["id"] != request_id:
            raise ValueError("mcp_response_id_invalid")
        if ("error" in payload) == ("result" in payload):
            raise ValueError("mcp_protocol_invalid")
        if "error" in payload:
            error = payload["error"]
            if (
                not isinstance(error, dict)
                or type(error.get("code")) is not int
                or not isinstance(error.get("message"), str)
            ):
                raise ValueError("mcp_protocol_invalid")
            raise _JSONRPCError(error)
        if not isinstance(payload["result"], dict):
            raise ValueError("mcp_protocol_invalid")  # noqa: TRY004
        if (
            self.profile.modern
            and payload["result"].get("resultType", "complete") != "complete"
        ):
            raise ValueError("mcp_capability_unsupported")
        return payload["result"]

    def _status(self, response: httpx.Response) -> None:
        if response.status_code in (401, 403):
            raise ValueError("mcp_authentication_unsupported")
        if 300 <= response.status_code < 400:
            raise ValueError("mcp_redirect_refused")
        if response.status_code == 404 and self._session_id is not None:
            self._session_id = None
            raise ValueError("mcp_session_expired")
        # Modern protocol errors use HTTP 400/404 with a JSON-RPC envelope.
        if response.status_code not in (200, 400, 404):
            raise ValueError("mcp_http_error")

    async def _post_oneway(self, payload: dict) -> None:
        async with self._http.stream(
            "POST",
            self.url,
            content=self._encode(payload),
            headers=await self._headers(),
        ) as response:
            if response.status_code != 202:
                self._status(response)
                raise ValueError("mcp_notification_rejected")
            async for chunk in response.aiter_raw():
                if chunk:
                    raise ValueError("mcp_notification_invalid")

    @guarded
    @producer_call
    async def notify(self, method: str, params: dict | None = None) -> None:
        if self._closed:
            raise RuntimeError("mcp_connection_closed")
        if self.profile.modern:
            raise ValueError("mcp_notification_unsupported")
        await self._post_oneway(
            {"jsonrpc": "2.0", "method": method, "params": params or {}}
        )

    async def close(self) -> None:
        # Admission closure does not mean the HTTP client/pool finished closing.
        # A cancelled wait can rejoin this owned task. Never restart a failed
        # aclose: HTTPX sets its CLOSED bit before lower transport cleanup.
        self._closed = True
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._close_resources())
        await asyncio.shield(self._cleanup_task)

    async def _close_resources(self) -> None:
        try:
            requests = list(self._requests.items())
            for task, _ in requests:
                task.cancel()
            # Only the request scope belongs to us; callers may perform arbitrary
            # asynchronous finalization after request() has already unwound.
            await asyncio.gather(*(drained for _, drained in requests))
            if self._session_id is not None:
                try:
                    async with asyncio.timeout(1.0):
                        async with self._http.stream(
                            "DELETE", self.url, headers=await self._headers()
                        ):
                            pass
                except Exception:  # noqa: BLE001 -- best-effort session close
                    logger.debug("MCP session termination could not be confirmed")
        finally:
            await self._http.aclose()
            self._cleanup_complete = True
