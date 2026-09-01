"""
MCP Client implementation for tldw_chatbook

This module provides client functionality to connect to external MCP servers
and use their tools, resources, and prompts within tldw_chatbook.
"""

from __future__ import annotations

# subprocess supplies only PIPE constants; launches use an argument-vector API.
import asyncio
from collections.abc import Awaitable, Callable, Mapping
import json
import math
import re
import subprocess  # nosec B404
from datetime import datetime
from itertools import count
from time import monotonic as _monotonic
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from loguru import logger

from tldw_chatbook.MCP.spawn_guard import screen_spawn_command

_MCP_PROTOCOL_VERSION = "2025-03-26"
_REQUEST_TIMEOUT_SECONDS = 10.0
_TERMINATE_TIMEOUT_SECONDS = 2.0
CONNECT_TIMEOUT_SECONDS = 30.0
CATALOG_TIMEOUT_SECONDS = 10.0
CLEANUP_TIMEOUT_SECONDS = 5.0
MAX_OUTPUT_LINE_BYTES = 1_048_576
MAX_RESULT_BYTES = 786_432
MAX_JSON_DEPTH = 64
MAX_SCHEMA_BYTES = 262_144
MAX_SCHEMA_DEPTH = 32
MAX_DESCRIPTOR_STRING_LENGTH = 4096
MAX_DESCRIPTOR_NAME_LENGTH = 128
MAX_RESOURCE_NAME_LENGTH = 512
MAX_RESOURCE_URI_LENGTH = 2048
MAX_MIME_TYPE_LENGTH = 255
MAX_CATALOG_PAGES = 100
MAX_CATALOG_ITEMS = 10_000
_URI_SCHEME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*\Z")


class MCPClientError(RuntimeError):
    """Bounded client-side validation failure."""


def _remaining(deadline: float, message: str) -> float:
    remaining = deadline - _monotonic()
    if remaining <= 0:
        raise MCPClientError(message)
    return remaining


def _bounded_json_copy(
    value: Any,
    *,
    message: str,
    max_bytes: int = MAX_RESULT_BYTES,
    max_depth: int = MAX_JSON_DEPTH,
    mapping: bool = False,
) -> Any:
    def validate(item: Any, depth: int, ancestors: set[int]) -> None:
        if depth > max_depth:
            raise ValueError
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ValueError
            identity = id(item)
            if identity in ancestors:
                raise ValueError
            ancestors.add(identity)
            try:
                for child in item.values():
                    validate(child, depth + 1, ancestors)
            finally:
                ancestors.remove(identity)
            return
        if isinstance(item, list):
            identity = id(item)
            if identity in ancestors:
                raise ValueError
            ancestors.add(identity)
            try:
                for child in item:
                    validate(child, depth + 1, ancestors)
            finally:
                ancestors.remove(identity)
            return
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        raise ValueError

    try:
        if mapping and not isinstance(value, Mapping):
            raise ValueError
        validate(value, 1, set())
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > max_bytes:
            raise ValueError
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise MCPClientError(message) from None


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    maximum: int = MAX_DESCRIPTOR_STRING_LENGTH,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError
    return value


def _optional_string(
    payload: Mapping[str, Any],
    key: str,
    default: str = "",
    *,
    maximum: int = MAX_DESCRIPTOR_STRING_LENGTH,
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError
    return value


def _descriptor_name(payload: Mapping[str, Any]) -> str:
    name = payload.get("name")
    if not isinstance(name, str):
        raise ValueError
    return name


def _resource_uri(payload: Mapping[str, Any]) -> str:
    uri = _required_string(payload, "uri", maximum=MAX_RESOURCE_URI_LENGTH)
    if any(character.isspace() or ord(character) < 32 for character in uri):
        raise ValueError
    try:
        parsed = urlsplit(uri)
        parsed.port
    except ValueError:
        raise ValueError from None
    if _URI_SCHEME_PATTERN.fullmatch(parsed.scheme) is None:
        raise ValueError
    if parsed.scheme.lower() in {"http", "https"} and not parsed.hostname:
        raise ValueError
    if parsed.username is not None or parsed.password is not None:
        raise ValueError
    return uri


def _annotations(payload: Mapping[str, Any], *, tool: bool = False) -> Dict[str, Any]:
    annotations = _bounded_json_copy(
        payload.get("annotations", {}),
        message="Invalid MCP catalog items",
        mapping=True,
    )
    audience = annotations.get("audience")
    if audience is not None and (
        not isinstance(audience, list)
        or not all(role in {"user", "assistant"} for role in audience)
    ):
        raise ValueError
    priority = annotations.get("priority")
    if priority is not None and (
        isinstance(priority, bool)
        or not isinstance(priority, (int, float))
        or not 0 <= priority <= 1
    ):
        raise ValueError
    if "lastModified" in annotations and not isinstance(
        annotations["lastModified"], str
    ):
        raise ValueError
    if tool:
        if "title" in annotations:
            _optional_string(annotations, "title")
        for field in (
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        ):
            if field in annotations and not isinstance(annotations[field], bool):
                raise ValueError
    return annotations


def _copy_resource_metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    return _bounded_json_copy(
        value,
        message="Invalid MCP resource metadata",
        mapping=True,
    )


def _tool_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    if not isinstance(payload, Mapping):
        raise ValueError
    input_schema = _bounded_json_copy(
        payload.get("inputSchema"),
        message="Invalid MCP catalog items",
        max_bytes=MAX_SCHEMA_BYTES,
        max_depth=MAX_SCHEMA_DEPTH,
        mapping=True,
    )
    if input_schema.get("type") != "object":
        raise ValueError
    return SimpleNamespace(
        name=_descriptor_name(payload),
        description=_optional_string(payload, "description"),
        inputSchema=input_schema,
        annotations=_annotations(payload, tool=True),
    )


def _resource_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    if not isinstance(payload, Mapping):
        raise ValueError
    size = payload.get("size")
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
        raise ValueError
    return SimpleNamespace(
        uri=_resource_uri(payload),
        name=_required_string(payload, "name", maximum=MAX_RESOURCE_NAME_LENGTH),
        description=_optional_string(payload, "description"),
        mimeType=(
            _required_string(payload, "mimeType", maximum=MAX_MIME_TYPE_LENGTH)
            if "mimeType" in payload
            else "text/plain"
        ),
        annotations=_annotations(payload),
        size=size,
    )


def _prompt_argument_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    if not isinstance(payload, Mapping):
        raise ValueError
    required = payload.get("required", False)
    if not isinstance(required, bool):
        raise ValueError
    return SimpleNamespace(
        name=_required_string(payload, "name", maximum=MAX_DESCRIPTOR_NAME_LENGTH),
        description=_optional_string(payload, "description"),
        required=required,
    )


def _prompt_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    if not isinstance(payload, Mapping):
        raise ValueError
    arguments = payload.get("arguments", [])
    if not isinstance(arguments, list):
        raise ValueError
    converted_arguments = [_prompt_argument_from_payload(arg) for arg in arguments]
    argument_names = [argument.name for argument in converted_arguments]
    if len(argument_names) != len(set(argument_names)):
        raise ValueError
    return SimpleNamespace(
        name=_descriptor_name(payload),
        description=_optional_string(payload, "description"),
        arguments=converted_arguments,
        annotations=_annotations(payload),
    )


def _resource_content_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        uri=payload.get("uri", ""),
        mimeType=payload.get("mimeType", "text/plain"),
        text=payload.get("text", ""),
        blob=payload.get("blob"),
    )


def _prompt_message_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    content = payload.get("content", {}) or {}
    if isinstance(content, dict):
        content_value = SimpleNamespace(
            type=content.get("type", "text"),
            text=content.get("text", ""),
        )
    else:
        content_value = content

    return SimpleNamespace(
        role=payload.get("role", "user"),
        content=content_value,
    )


class _JSONRPCError(RuntimeError):
    def __init__(self, error: Dict[str, Any]):
        self.error = error
        message = error.get("message") or "JSON-RPC error"
        code = error.get("code")
        if code is not None:
            super().__init__(f"[{code}] {message}")
        else:
            super().__init__(message)


class _StdioJSONRPCConnection:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        client_name: str,
        request_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
        on_transport_failure: Optional[Callable[[], Awaitable[None]]] = None,
        server_request_dispatcher: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[Any]]
        ] = None,
    ) -> None:
        self.process = process
        self.client_name = client_name
        self.request_timeout_seconds = request_timeout_seconds
        self.server_info: Dict[str, Any] = {}
        self.server_capabilities: Dict[str, Any] = {}
        self.protocol_version = ""

        self._request_ids = count(1)
        self._pending_requests: Dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._reader_unavailable = False
        self._cleanup_complete = False
        self._on_transport_failure = on_transport_failure
        # TASK-26029: optional handler for server-initiated sampling/elicitation
        # requests. When None, such requests get method-not-found as before.
        self._server_request_dispatcher = server_request_dispatcher
        self._transport_cleanup_task: Optional[asyncio.Task[None]] = None
        self._read_task = asyncio.create_task(self._read_loop())
        self._stderr_task = (
            asyncio.create_task(self._stderr_loop())
            if getattr(process, "stderr", None) is not None
            else None
        )

    async def initialize(self) -> Dict[str, Any]:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": "1.0.0",
                },
            },
        )
        protocol_version = result.get("protocolVersion")
        if protocol_version != _MCP_PROTOCOL_VERSION:
            raise MCPClientError("Unexpected MCP protocol version")
        self.protocol_version = protocol_version
        self.server_capabilities = _bounded_json_copy(
            result.get("capabilities"),
            message="Invalid MCP initialization metadata",
            mapping=True,
        )
        self.server_info = _bounded_json_copy(
            result.get("serverInfo"),
            message="Invalid MCP initialization metadata",
            mapping=True,
        )
        await self.notify("notifications/initialized")
        return result

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(
            tools=await self._collect_catalog("tools/list", "tools", _tool_from_payload)
        )

    async def list_resources(self) -> SimpleNamespace:
        return SimpleNamespace(
            resources=await self._collect_catalog(
                "resources/list", "resources", _resource_from_payload
            )
        )

    async def list_prompts(self) -> SimpleNamespace:
        return SimpleNamespace(
            prompts=await self._collect_catalog(
                "prompts/list", "prompts", _prompt_from_payload
            )
        )

    async def _collect_catalog(
        self,
        method: str,
        item_key: str,
        converter: Callable[[Dict[str, Any]], SimpleNamespace],
    ) -> List[SimpleNamespace]:
        collected: List[SimpleNamespace] = []
        seen_cursors: set[str] = set()
        params: Dict[str, Any] = {}
        deadline = _monotonic() + CATALOG_TIMEOUT_SECONDS

        for page_number in range(1, MAX_CATALOG_PAGES + 1):
            result = await self.request(
                method,
                params,
                timeout_seconds=_remaining(deadline, "MCP catalog deadline exceeded"),
            )
            page_items = result.get(item_key)
            if not isinstance(page_items, list):
                raise MCPClientError("Invalid MCP catalog items")

            cursor = result.get("nextCursor")
            if cursor is not None:
                if not isinstance(cursor, str) or not cursor:
                    raise MCPClientError("Invalid MCP catalog cursor")
                if cursor in seen_cursors:
                    raise MCPClientError("Repeated MCP catalog cursor")

            if len(collected) + len(page_items) > MAX_CATALOG_ITEMS:
                raise MCPClientError("MCP catalog item limit exceeded")
            if cursor is not None and page_number == MAX_CATALOG_PAGES:
                raise MCPClientError("MCP catalog page limit exceeded")

            try:
                converted = [converter(item) for item in page_items]
            except Exception:
                raise MCPClientError("Invalid MCP catalog items") from None
            collected.extend(converted)

            if cursor is None:
                return collected
            seen_cursors.add(cursor)
            params = {"cursor": cursor}

        raise MCPClientError("MCP catalog page limit exceeded")

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> SimpleNamespace:
        result = await self.request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )
        return SimpleNamespace(content=result.get("content", []))

    async def read_resource(self, resource_uri: str) -> SimpleNamespace:
        result = await self.request(
            "resources/read",
            {
                "uri": resource_uri,
            },
        )
        return SimpleNamespace(
            contents=[
                _resource_content_from_payload(item)
                for item in result.get("contents", [])
            ],
            _meta=_copy_resource_metadata(result.get("_meta")),
        )

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> SimpleNamespace:
        params: Dict[str, Any] = {"name": prompt_name}
        if arguments:
            params["arguments"] = arguments
        result = await self.request("prompts/get", params)
        return SimpleNamespace(
            messages=[
                _prompt_message_from_payload(message)
                for message in result.get("messages", [])
            ]
        )

    async def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._reader_unavailable or self._cleanup_complete:
            raise RuntimeError("Connection is closed")

        request_id = next(self._request_ids)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self._pending_requests[request_id] = future

        try:
            await self._send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            return await asyncio.wait_for(
                future,
                timeout=(
                    self.request_timeout_seconds
                    if timeout_seconds is None
                    else timeout_seconds
                ),
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Timed out waiting for MCP response to '{method}'"
            ) from exc
        finally:
            self._pending_requests.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def notify(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        await self._send_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                **({"params": params} if params else {}),
            }
        )

    async def close(self) -> None:
        close_lock = getattr(self, "_close_lock", None)
        if close_lock is None:
            close_lock = self._close_lock = asyncio.Lock()
        async with close_lock:
            if self._cleanup_complete:
                return

            self._reader_unavailable = True
            self._fail_pending_requests(RuntimeError("MCP connection closed"))

            stdin = getattr(self.process, "stdin", None)
            if stdin is not None:
                try:
                    stdin.close()
                except Exception:
                    logger.warning("Failed to close MCP subprocess stdin")
                wait_closed = getattr(stdin, "wait_closed", None)
                if callable(wait_closed):
                    try:
                        await wait_closed()
                    except Exception:
                        logger.warning(
                            "Failed to wait for MCP subprocess stdin closure"
                        )

            if self.process.returncode is None:
                should_kill = False
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    logger.debug("MCP subprocess exited before termination cleanup")
                except Exception:
                    logger.warning("Failed to terminate MCP subprocess cleanly")
                    should_kill = True

                if not should_kill:
                    try:
                        await asyncio.wait_for(
                            self.process.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        should_kill = True
                    except Exception:
                        logger.warning("Failed to wait for MCP subprocess termination")
                        should_kill = True

                if should_kill:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        logger.debug("MCP subprocess exited before kill cleanup")
                    except Exception:
                        logger.warning("Failed to kill MCP subprocess cleanly")
                    else:
                        try:
                            await asyncio.wait_for(
                                self.process.wait(),
                                timeout=_TERMINATE_TIMEOUT_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Timed out reaping MCP subprocess after kill"
                            )
                        except Exception:
                            logger.warning("Failed to reap MCP subprocess after kill")

            current_task = asyncio.current_task()
            for task in (self._read_task, self._stderr_task):
                if task is None or task is current_task:
                    continue
                if task.done():
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.debug("MCP transport task was already cancelled")
                    except Exception:
                        logger.warning("MCP transport task failed before cleanup")
                    continue
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug("MCP transport task cancelled during cleanup")
                except Exception:
                    logger.warning("MCP transport task failed during cleanup")

            self._cleanup_complete = True

    async def _send_message(self, payload: Dict[str, Any]) -> None:
        stdin = getattr(self.process, "stdin", None)
        if stdin is None:
            raise RuntimeError("MCP subprocess stdin is unavailable")

        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if "\n" in serialized:
            raise ValueError("MCP JSON-RPC messages must not contain embedded newlines")

        async with self._write_lock:
            stdin.write(serialized.encode("utf-8") + b"\n")
            drain = getattr(stdin, "drain", None)
            if callable(drain):
                await drain()

    async def _read_loop(self) -> None:
        stdout = getattr(self.process, "stdout", None)
        if stdout is None:
            self._mark_reader_unavailable()
            return

        try:
            while True:
                line = await stdout.readline()
                if not line:
                    self._mark_reader_unavailable()
                    return
                if len(line) > MAX_OUTPUT_LINE_BYTES:
                    raise ValueError

                decoded_line = line.decode("utf-8").rstrip("\r\n")
                if not decoded_line:
                    continue

                payload = json.loads(decoded_line)
                await self._handle_incoming_payload(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._mark_reader_unavailable()

    def _mark_reader_unavailable(self) -> None:
        self._reader_unavailable = True
        self._fail_pending_requests(RuntimeError("MCP transport unavailable"))
        logger.warning("MCP transport unavailable")
        cleanup_handler = getattr(self, "_on_transport_failure", None)
        cleanup_task = getattr(self, "_transport_cleanup_task", None)
        if cleanup_handler is not None and cleanup_task is None:
            self._transport_cleanup_task = asyncio.create_task(
                self._run_transport_failure_cleanup()
            )

    async def _run_transport_failure_cleanup(self) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(0)
            cleanup_handler = self._on_transport_failure
            if cleanup_handler is not None:
                await cleanup_handler()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("MCP transport cleanup failed")
        finally:
            if self._transport_cleanup_task is current_task:
                self._transport_cleanup_task = None

    async def _stderr_loop(self) -> None:
        stderr = getattr(self.process, "stderr", None)
        if stderr is None:
            return

        try:
            while True:
                line = await stderr.readline()
                if not line:
                    break
                message = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if message:
                    logger.debug("MCP server stderr: {}", message)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).debug("MCP stderr reader exited with error")

    async def _handle_incoming_payload(self, payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                await self._handle_incoming_payload(item)
            return

        if not isinstance(payload, dict):
            logger.debug("Ignoring unexpected MCP payload")
            return

        if "method" in payload and "id" in payload:
            await self._handle_server_request(payload)
            return

        if "method" in payload:
            logger.debug("Ignoring MCP server notification")
            return

        if "id" in payload:
            self._handle_response(payload)
            return

        logger.debug("Ignoring unrecognized MCP payload")

    async def _handle_server_request(self, payload: Dict[str, Any]) -> None:
        request_id = payload.get("id")
        method = payload.get("method")

        if method == "ping":
            await self._send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {},
                }
            )
            return

        # TASK-26029: sampling/elicitation via the injected dispatcher. A
        # dispatcher returns either a result mapping or a JsonRpcError; any
        # exception becomes an internal-error response so the server never
        # hangs waiting on a reply.
        dispatcher = self._server_request_dispatcher
        if dispatcher is not None:
            params = payload.get("params")
            if not isinstance(params, dict):
                params = {}
            try:
                outcome = await dispatcher(method, params)
            except Exception as exc:  # noqa: BLE001 - must always reply
                logger.opt(exception=True).warning(
                    "MCP server request handler failed for {}", method
                )
                await self._send_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": f"Internal error: {exc}"},
                    }
                )
                return
            error_payload = getattr(outcome, "to_payload", None)
            if error_payload is not None and hasattr(outcome, "code"):
                await self._send_message(
                    {"jsonrpc": "2.0", "id": request_id, "error": outcome.to_payload()}
                )
            else:
                await self._send_message(
                    {"jsonrpc": "2.0", "id": request_id, "result": outcome}
                )
            return

        await self._send_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        )

    def _handle_response(self, payload: Dict[str, Any]) -> None:
        request_id = payload.get("id")
        if isinstance(request_id, bool) or not isinstance(request_id, int):
            logger.debug("Ignoring MCP response with invalid id")
            return

        future = self._pending_requests.pop(request_id, None)
        if future is None:
            logger.debug("Ignoring MCP response for unknown request id: {}", request_id)
            return

        if future.done():
            return

        if "error" in payload:
            future.set_exception(_JSONRPCError(dict(payload.get("error") or {})))
            return

        future.set_result(dict(payload.get("result") or {}))

    def _fail_pending_requests(self, exc: Exception) -> None:
        for request_id, future in list(self._pending_requests.items()):
            self._pending_requests.pop(request_id, None)
            if future.done():
                continue
            future.set_exception(exc)


class _PendingConnection:
    """Private ownership of a spawned child until publication or reap."""

    def __init__(self, process: Any) -> None:
        self.process = process
        self.session: Optional[_StdioJSONRPCConnection] = None
        self.server: Dict[str, Any] = {}


class MCPClient:
    """MCP Client for connecting to external MCP servers."""

    def __init__(self, name: str = "tldw_chatbook_client"):
        """Initialize the MCP client."""
        self.name = name
        self.sessions: Dict[str, _StdioJSONRPCConnection] = {}
        self.servers: Dict[str, Dict[str, Any]] = {}
        self._pending_connections: Dict[str, _PendingConnection] = {}
        self._connect_reservations: Dict[str, object] = {}
        # TASK-26029: set by the app to enable server-initiated sampling/
        # elicitation; None keeps the method-not-found behavior.
        self._server_request_dispatcher: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[Any]]
        ] = None

        logger.info("MCP Client '{}' initialized", name)

    async def connect_to_server(
        self,
        server_id: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Connect to an MCP server via stdio.

        Args:
            server_id: Unique identifier for this server connection
            command: Command to run the server
            args: Optional command arguments
            env: Optional environment variables

        Returns:
            True if connection successful
        """
        # TASK-26013: the guard runs at SPAWN time too, so a config edited on
        # disk to a dangerous shape cannot bypass the save-time check.
        spawn_verdict = screen_spawn_command(command, args)
        if spawn_verdict is not None:
            logger.error(
                "MCP spawn refused for '{}': {} (rule: {})",
                server_id,
                spawn_verdict.reason,
                spawn_verdict.rule,
            )
            return False
        if server_id in self._connect_reservations:
            logger.warning("MCP connection attempt already in progress")
            return False
        reservation = object()
        self._connect_reservations[server_id] = reservation
        session = None
        pending = None
        deadline = _monotonic() + CONNECT_TIMEOUT_SECONDS
        try:
            if server_id in self.sessions or server_id in self._pending_connections:
                try:
                    await self._bounded_teardown_connection(server_id)
                except MCPClientError:
                    logger.warning("MCP connection cleanup incomplete before reconnect")
                    return False

            spawn_timeout = _remaining(deadline, "MCP connection deadline exceeded")
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    command,
                    *(args or []),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    limit=MAX_OUTPUT_LINE_BYTES,
                ),
                timeout=spawn_timeout,
            )
            pending = _PendingConnection(process)
            if (
                self._connect_reservations.get(server_id) is not reservation
                or server_id in self.sessions
                or server_id in self._pending_connections
            ):
                try:
                    await self._bounded_teardown_connection(server_id, pending=pending)
                except MCPClientError:
                    logger.warning(
                        "MCP connection cleanup incomplete after ownership change"
                    )
                return False
            self._pending_connections[server_id] = pending

            async def cleanup_failed_transport() -> None:
                active_owner = (
                    session is not None and self.sessions.get(server_id) is session
                )
                pending_owner = self._pending_connections.get(server_id) is pending
                if active_owner or pending_owner:
                    await self._bounded_teardown_connection(
                        server_id, session=session, pending=pending
                    )

            session = _StdioJSONRPCConnection(
                process,
                client_name=self.name,
                server_request_dispatcher=self._server_request_dispatcher,
            )
            pending.session = session
            session._on_transport_failure = cleanup_failed_transport
            initialize_timeout = _remaining(
                deadline, "MCP connection deadline exceeded"
            )
            await asyncio.wait_for(session.initialize(), timeout=initialize_timeout)

            pending.server = {
                "command": command,
                "args": list(args or []),
                "connected_at": datetime.now().isoformat(),
                "tools": [],
                "resources": [],
                "prompts": [],
                "protocol_version": session.protocol_version,
                "server_info": _bounded_json_copy(
                    session.server_info,
                    message="Invalid MCP initialization metadata",
                    mapping=True,
                ),
                "server_capabilities": _bounded_json_copy(
                    session.server_capabilities,
                    message="Invalid MCP initialization metadata",
                    mapping=True,
                ),
            }

            discovery_timeout = _remaining(deadline, "MCP connection deadline exceeded")
            await asyncio.wait_for(
                self._discover_server_capabilities(server_id),
                timeout=discovery_timeout,
            )
            if (
                self._connect_reservations.get(server_id) is not reservation
                or self._pending_connections.get(server_id) is not pending
                or server_id in self.sessions
            ):
                raise MCPClientError("MCP connection ownership changed")
            self.sessions[server_id] = session
            self.servers[server_id] = pending.server
            self._pending_connections.pop(server_id, None)

            logger.info("Successfully connected to MCP server: {}", server_id)
            return True

        except asyncio.CancelledError:
            try:
                await self._bounded_teardown_connection(
                    server_id, session=session, pending=pending
                )
            except MCPClientError:
                logger.warning("MCP connection cleanup incomplete after cancellation")
            raise
        except Exception:
            try:
                await self._bounded_teardown_connection(
                    server_id, session=session, pending=pending
                )
            except MCPClientError:
                logger.warning(
                    "MCP connection cleanup incomplete after connection failure"
                )
            logger.error("Failed to connect to MCP server")
            return False
        finally:
            if self._connect_reservations.get(server_id) is reservation:
                self._connect_reservations.pop(server_id, None)

    async def disconnect_from_server(self, server_id: str) -> bool:
        """Disconnect from an MCP server.

        Args:
            server_id: Server identifier

        Returns:
            True if disconnection successful
        """
        try:
            if server_id in self.sessions or server_id in self._pending_connections:
                await self._bounded_teardown_connection(server_id)
                logger.info("Disconnected from MCP server: {}", server_id)
                return True
            else:
                logger.warning("Server {} not found", server_id)
                return False

        except Exception as e:
            logger.error("Error disconnecting from server {}: {}", server_id, e)
            return False

    async def _discover_server_capabilities(self, server_id: str) -> None:
        """Discover tools, resources, and prompts from a server.

        Args:
            server_id: Server identifier
        """
        session = self.sessions.get(server_id)
        server = self.servers.get(server_id)
        if session is None:
            pending = self._pending_connections.get(server_id)
            if pending is not None:
                session = pending.session
                server = pending.server
        if session is None or server is None:
            raise RuntimeError(f"Server session not found for {server_id}")

        tools_response = await session.list_tools()
        server["tools"] = tools_response.tools

        resources_response = await session.list_resources()
        server["resources"] = resources_response.resources

        prompts_response = await session.list_prompts()
        server["prompts"] = prompts_response.prompts

        logger.info(
            "Discovered MCP capabilities: {} tools, {} resources, {} prompts",
            len(server["tools"]),
            len(server["resources"]),
            len(server["prompts"]),
        )

        if not (server["tools"] or server["resources"] or server["prompts"]):
            raise RuntimeError(
                f"Server {server_id} returned no discoverable capabilities"
            )

    async def call_tool(
        self, server_id: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on a connected server.

        Args:
            server_id: Server identifier
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        try:
            session = self.sessions.get(server_id)
            if not session:
                return {"error": f"Server {server_id} not connected"}

            result = await session.call_tool(tool_name, arguments)

            if hasattr(result, "content"):
                return {"result": result.content}
            else:
                return {"result": str(result)}

        except Exception as e:
            logger.error("Error calling tool {} on {}: {}", tool_name, server_id, e)
            return {"error": str(e)}

    async def read_resource(self, server_id: str, resource_uri: str) -> Dict[str, Any]:
        """Read a resource from a connected server.

        Args:
            server_id: Server identifier
            resource_uri: Resource URI

        Returns:
            Resource content
        """
        try:
            session = self.sessions.get(server_id)
            if not session:
                return {"error": f"Server {server_id} not connected"}

            result = await session.read_resource(resource_uri)

            return {
                "uri": resource_uri,
                "content": result.contents[0].text if result.contents else "",
                "mimeType": result.contents[0].mimeType
                if result.contents
                else "text/plain",
                "_meta": _copy_resource_metadata(getattr(result, "_meta", None)),
            }

        except Exception as e:
            logger.error(
                "Error reading resource {} from {}: {}", resource_uri, server_id, e
            )
            return {"error": str(e)}

    async def get_prompt(
        self,
        server_id: str,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Get a prompt from a connected server.

        Args:
            server_id: Server identifier
            prompt_name: Name of the prompt
            arguments: Optional prompt arguments

        Returns:
            List of prompt messages
        """
        try:
            session = self.sessions.get(server_id)
            if not session:
                return [
                    {
                        "role": "user",
                        "content": f"Error: Server {server_id} not connected",
                    }
                ]

            result = await session.get_prompt(prompt_name, arguments or {})

            messages = []
            for msg in result.messages:
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content.text
                        if hasattr(msg.content, "text")
                        else str(msg.content),
                    }
                )

            return messages

        except Exception as e:
            logger.error(
                "Error getting prompt {} from {}: {}", prompt_name, server_id, e
            )
            return [{"role": "user", "content": f"Error: {str(e)}"}]

    def list_connected_servers(self) -> List[Dict[str, Any]]:
        """List all connected servers and their capabilities.

        Returns:
            List of server information
        """
        servers = []
        for server_id, info in self.servers.items():
            servers.append(
                {
                    "id": server_id,
                    "command": info["command"],
                    "connected_at": info["connected_at"],
                    "tools_count": len(info["tools"]),
                    "resources_count": len(info["resources"]),
                    "prompts_count": len(info["prompts"]),
                }
            )
        return servers

    def get_server_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """Get list of tools from a server.

        Args:
            server_id: Server identifier

        Returns:
            List of tool definitions
        """
        if server_id not in self.servers:
            return []

        tools = []
        for tool in self.servers[server_id]["tools"]:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": _bounded_json_copy(
                        tool.inputSchema,
                        message="Invalid MCP catalog items",
                        max_bytes=MAX_SCHEMA_BYTES,
                        max_depth=MAX_SCHEMA_DEPTH,
                        mapping=True,
                    ),
                    "annotations": _bounded_json_copy(
                        getattr(tool, "annotations", {}),
                        message="Invalid MCP catalog items",
                        mapping=True,
                    ),
                }
            )
        return tools

    def get_server_resources(self, server_id: str) -> List[Dict[str, Any]]:
        """Get list of resources from a server.

        Args:
            server_id: Server identifier

        Returns:
            List of resource definitions
        """
        if server_id not in self.servers:
            return []

        resources = []
        for resource in self.servers[server_id]["resources"]:
            resources.append(
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mimeType,
                    "annotations": _bounded_json_copy(
                        getattr(resource, "annotations", {}),
                        message="Invalid MCP catalog items",
                        mapping=True,
                    ),
                    "size": getattr(resource, "size", None),
                }
            )
        return resources

    def get_server_prompts(self, server_id: str) -> List[Dict[str, Any]]:
        """Get list of prompts from a server.

        Args:
            server_id: Server identifier

        Returns:
            List of prompt definitions
        """
        if server_id not in self.servers:
            return []

        prompts = []
        for prompt in self.servers[server_id]["prompts"]:
            prompts.append(
                {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": [
                        {
                            "name": arg.name,
                            "description": arg.description,
                            "required": arg.required,
                        }
                        for arg in (prompt.arguments or [])
                    ],
                    "annotations": _bounded_json_copy(
                        getattr(prompt, "annotations", {}),
                        message="Invalid MCP catalog items",
                        mapping=True,
                    ),
                }
            )
        return prompts

    async def describe_server(self, server_id: str) -> Dict[str, Any]:
        """Describe a connected server using the cached discovery state."""
        info = self.servers.get(server_id)
        if info is None:
            raise KeyError(f"Unknown server_id: {server_id}")

        return {
            "server_id": server_id,
            "command": info.get("command"),
            "args": list(info.get("args") or []),
            "connected_at": info.get("connected_at"),
            "tools": self.get_server_tools(server_id),
            "resources": self.get_server_resources(server_id),
            "prompts": self.get_server_prompts(server_id),
        }

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        server_ids = list(
            dict.fromkeys((*self.sessions.keys(), *self._pending_connections.keys()))
        )
        cancelled = False
        incomplete = False
        for server_id in server_ids:
            try:
                if not await self.disconnect_from_server(server_id):
                    incomplete = True
            except asyncio.CancelledError:
                cancelled = True
                if server_id in self.sessions or server_id in self._pending_connections:
                    incomplete = True
        if incomplete:
            logger.warning("MCP disconnect_all cleanup incomplete")
        elif not cancelled:
            logger.info("Disconnected from all MCP servers")
        if cancelled:
            raise asyncio.CancelledError

    async def _teardown_connection(
        self,
        server_id: str,
        *,
        session: Optional[_StdioJSONRPCConnection] = None,
    ) -> None:
        active_session = (
            session if session is not None else self.sessions.get(server_id)
        )

        if active_session is not None:
            try:
                await active_session.close()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed to close MCP connection during teardown")

    async def _force_stop_process(self, process: Any) -> bool:
        stdin = getattr(process, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except Exception:
                logger.warning(
                    "Failed to close MCP subprocess stdin during forced cleanup"
                )
        if process is None or getattr(process, "returncode", None) is not None:
            return True
        try:
            process.terminate()
        except Exception:
            logger.warning("Failed to terminate MCP subprocess during forced cleanup")
        try:
            await asyncio.wait_for(process.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)
            return True
        except asyncio.TimeoutError:
            logger.debug("MCP subprocess did not terminate before forced cleanup")
        except Exception:
            logger.warning(
                "Failed to wait for MCP subprocess termination during forced cleanup"
            )
        try:
            process.kill()
        except Exception:
            logger.warning("Failed to kill MCP subprocess during forced cleanup")
            return False
        try:
            await asyncio.wait_for(process.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)
            return True
        except asyncio.TimeoutError:
            logger.warning("Timed out reaping MCP subprocess after forced cleanup")
        except Exception:
            logger.warning("Failed to reap MCP subprocess after forced cleanup")
        return False

    async def _finish_connection_cleanup(
        self,
        server_id: str,
        cleanup_session: Optional[_StdioJSONRPCConnection],
        active_owner: Optional[_StdioJSONRPCConnection],
        pending_owner: Optional[_PendingConnection],
    ) -> None:
        if cleanup_session is not None:
            cleanup = asyncio.create_task(
                self._teardown_connection(server_id, session=cleanup_session)
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(cleanup), timeout=CLEANUP_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                cleanup.cancel()
                await asyncio.gather(cleanup, return_exceptions=True)
        process = (
            pending_owner.process
            if pending_owner is not None
            else getattr(cleanup_session, "process", None)
        )
        if process is not None and getattr(process, "returncode", None) is None:
            if not await self._force_stop_process(process):
                raise MCPClientError("MCP subprocess cleanup incomplete")
        if active_owner is not None and self.sessions.get(server_id) is active_owner:
            self.sessions.pop(server_id, None)
            self.servers.pop(server_id, None)
        if (
            pending_owner is not None
            and self._pending_connections.get(server_id) is pending_owner
        ):
            self._pending_connections.pop(server_id, None)
        if (
            cleanup_session is None
            and active_owner is None
            and pending_owner is None
            and server_id not in self.sessions
        ):
            self.servers.pop(server_id, None)

    async def _bounded_teardown_connection(
        self,
        server_id: str,
        *,
        session: Optional[_StdioJSONRPCConnection] = None,
        pending: Optional[_PendingConnection] = None,
    ) -> None:
        registered_active = self.sessions.get(server_id)
        registered_pending = self._pending_connections.get(server_id)
        active_owner = (
            registered_active
            if (session is None and pending is None) or registered_active is session
            else None
        )
        pending_owner = pending
        if pending_owner is None and active_owner is None:
            if registered_pending is not None and (
                session is None or registered_pending.session is session
            ):
                pending_owner = registered_pending
        cleanup_session = session
        if cleanup_session is None:
            cleanup_session = active_owner
        if cleanup_session is None and pending_owner is not None:
            cleanup_session = pending_owner.session
        cleanup = asyncio.create_task(
            self._finish_connection_cleanup(
                server_id, cleanup_session, active_owner, pending_owner
            )
        )
        cancelled = False
        cleanup_failure: Optional[Exception] = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
            except Exception as exc:
                cleanup_failure = exc
        if cleanup_failure is None:
            try:
                cleanup.result()
            except Exception as exc:
                cleanup_failure = exc
        if cancelled:
            if cleanup_failure is not None:
                logger.warning("MCP connection cleanup incomplete after cancellation")
            raise asyncio.CancelledError
        if cleanup_failure is not None:
            raise cleanup_failure
