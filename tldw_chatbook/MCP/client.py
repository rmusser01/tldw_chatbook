"""
MCP Client implementation for tldw_chatbook

This module provides client functionality to connect to external MCP servers
and use their tools, resources, and prompts within tldw_chatbook.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime
from itertools import count
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from loguru import logger

_MCP_PROTOCOL_VERSION = "2025-03-26"
_REQUEST_TIMEOUT_SECONDS = 10.0
_TERMINATE_TIMEOUT_SECONDS = 2.0


def _tool_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        name=payload.get("name", ""),
        description=payload.get("description", ""),
        inputSchema=payload.get("inputSchema", {}),
    )


def _resource_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        uri=payload.get("uri", ""),
        name=payload.get("name", ""),
        description=payload.get("description", ""),
        mimeType=payload.get("mimeType", "text/plain"),
    )


def _prompt_argument_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        name=payload.get("name", ""),
        description=payload.get("description", ""),
        required=bool(payload.get("required", False)),
    )


def _prompt_from_payload(payload: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        name=payload.get("name", ""),
        description=payload.get("description", ""),
        arguments=[_prompt_argument_from_payload(arg) for arg in payload.get("arguments", [])],
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
    ) -> None:
        self.process = process
        self.client_name = client_name
        self.request_timeout_seconds = request_timeout_seconds
        self.server_info: Dict[str, Any] = {}
        self.server_capabilities: Dict[str, Any] = {}

        self._request_ids = count(1)
        self._pending_requests: Dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._closed = False
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
        self.server_capabilities = dict(result.get("capabilities") or {})
        self.server_info = dict(result.get("serverInfo") or {})
        await self.notify("notifications/initialized")
        return result

    async def list_tools(self) -> SimpleNamespace:
        result = await self.request("tools/list", {})
        return SimpleNamespace(
            tools=[_tool_from_payload(tool) for tool in result.get("tools", [])]
        )

    async def list_resources(self) -> SimpleNamespace:
        result = await self.request("resources/list", {})
        return SimpleNamespace(
            resources=[_resource_from_payload(resource) for resource in result.get("resources", [])]
        )

    async def list_prompts(self) -> SimpleNamespace:
        result = await self.request("prompts/list", {})
        return SimpleNamespace(
            prompts=[_prompt_from_payload(prompt) for prompt in result.get("prompts", [])]
        )

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> SimpleNamespace:
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
            contents=[_resource_content_from_payload(item) for item in result.get("contents", [])]
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
            messages=[_prompt_message_from_payload(message) for message in result.get("messages", [])]
        )

    async def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._closed:
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
                timeout=timeout_seconds or self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            self._pending_requests.pop(request_id, None)
            if not future.done():
                future.cancel()
            raise TimeoutError(f"Timed out waiting for MCP response to '{method}'") from exc
        except Exception:
            self._pending_requests.pop(request_id, None)
            if not future.done():
                future.cancel()
            raise

    async def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        await self._send_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                **({"params": params} if params else {}),
            }
        )

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._fail_pending_requests(RuntimeError("MCP connection closed"))

        stdin = getattr(self.process, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except Exception:
                pass
            wait_closed = getattr(stdin, "wait_closed", None)
            if callable(wait_closed):
                try:
                    await wait_closed()
                except Exception:
                    pass

        if self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            except Exception:
                logger.debug("Failed to terminate MCP subprocess cleanly", exc_info=True)

            try:
                await asyncio.wait_for(self.process.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                except Exception:
                    logger.debug("Failed to kill MCP subprocess cleanly", exc_info=True)
                try:
                    await self.process.wait()
                except Exception:
                    pass
            except Exception:
                pass

        for task in (self._read_task, self._stderr_task):
            if task is None:
                continue
            if task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

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
            self._fail_pending_requests(RuntimeError("MCP subprocess stdout is unavailable"))
            return

        try:
            while True:
                line = await stdout.readline()
                if not line:
                    break

                decoded_line = line.decode("utf-8").rstrip("\r\n")
                if not decoded_line:
                    continue

                try:
                    payload = json.loads(decoded_line)
                except json.JSONDecodeError:
                    logger.warning("Ignoring invalid MCP JSON-RPC payload from server: {}", decoded_line)
                    continue

                await self._handle_incoming_payload(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending_requests(RuntimeError(f"MCP read loop failed: {exc}"))
            return

        if not self._closed:
            self._closed = True
            self._fail_pending_requests(RuntimeError("MCP server closed the stdio transport"))

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
            logger.debug("MCP stderr reader exited with error", exc_info=True)

    async def _handle_incoming_payload(self, payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                await self._handle_incoming_payload(item)
            return

        if not isinstance(payload, dict):
            logger.debug("Ignoring unexpected MCP payload type: {}", type(payload).__name__)
            return

        if "method" in payload and "id" in payload:
            await self._handle_server_request(payload)
            return

        if "method" in payload:
            logger.debug("Ignoring MCP server notification: {}", payload.get("method"))
            return

        if "id" in payload:
            self._handle_response(payload)
            return

        logger.debug("Ignoring unrecognized MCP payload: {}", payload)

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
        if not isinstance(request_id, int):
            logger.debug("Ignoring MCP response with non-integer id: {}", request_id)
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
            if future.done():
                continue
            future.set_exception(exc)
            self._pending_requests.pop(request_id, None)


class MCPClient:
    """MCP Client for connecting to external MCP servers."""

    def __init__(self, name: str = "tldw_chatbook_client"):
        """Initialize the MCP client."""
        self.name = name
        self.sessions: Dict[str, _StdioJSONRPCConnection] = {}
        self.servers: Dict[str, Dict[str, Any]] = {}

        logger.info("MCP Client '{}' initialized", name)

    async def connect_to_server(
        self,
        server_id: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
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
        if server_id in self.sessions:
            await self._teardown_connection(server_id)

        session = None
        try:
            process = await asyncio.create_subprocess_exec(
                command,
                *(args or []),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            session = _StdioJSONRPCConnection(process, client_name=self.name)
            await session.initialize()

            self.sessions[server_id] = session
            self.servers[server_id] = {
                "command": command,
                "args": list(args or []),
                "connected_at": datetime.now().isoformat(),
                "tools": [],
                "resources": [],
                "prompts": [],
                "server_info": dict(session.server_info),
                "server_capabilities": dict(session.server_capabilities),
            }

            await self._discover_server_capabilities(server_id)

            logger.info("Successfully connected to MCP server: {}", server_id)
            return True

        except Exception as e:
            await self._teardown_connection(server_id, session=session)
            logger.error("Failed to connect to MCP server {}: {}", server_id, e)
            return False

    async def disconnect_from_server(self, server_id: str) -> bool:
        """Disconnect from an MCP server.

        Args:
            server_id: Server identifier

        Returns:
            True if disconnection successful
        """
        try:
            if server_id in self.sessions:
                await self._teardown_connection(server_id)
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
        if not session:
            raise RuntimeError(f"Server session not found for {server_id}")

        tools_response = await session.list_tools()
        self.servers[server_id]["tools"] = tools_response.tools

        resources_response = await session.list_resources()
        self.servers[server_id]["resources"] = resources_response.resources

        prompts_response = await session.list_prompts()
        self.servers[server_id]["prompts"] = prompts_response.prompts

        logger.info(
            "Discovered capabilities for {}: {} tools, {} resources, {} prompts",
            server_id,
            len(self.servers[server_id]["tools"]),
            len(self.servers[server_id]["resources"]),
            len(self.servers[server_id]["prompts"]),
        )

        if not (
            self.servers[server_id]["tools"]
            or self.servers[server_id]["resources"]
            or self.servers[server_id]["prompts"]
        ):
            raise RuntimeError(f"Server {server_id} returned no discoverable capabilities")

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
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

    async def read_resource(
        self,
        server_id: str,
        resource_uri: str
    ) -> Dict[str, Any]:
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
                "mimeType": result.contents[0].mimeType if result.contents else "text/plain"
            }

        except Exception as e:
            logger.error("Error reading resource {} from {}: {}", resource_uri, server_id, e)
            return {"error": str(e)}

    async def get_prompt(
        self,
        server_id: str,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None
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
                return [{"role": "user", "content": f"Error: Server {server_id} not connected"}]

            result = await session.get_prompt(prompt_name, arguments or {})

            messages = []
            for msg in result.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content.text if hasattr(msg.content, "text") else str(msg.content)
                })

            return messages

        except Exception as e:
            logger.error("Error getting prompt {} from {}: {}", prompt_name, server_id, e)
            return [{"role": "user", "content": f"Error: {str(e)}"}]

    def list_connected_servers(self) -> List[Dict[str, Any]]:
        """List all connected servers and their capabilities.

        Returns:
            List of server information
        """
        servers = []
        for server_id, info in self.servers.items():
            servers.append({
                "id": server_id,
                "command": info["command"],
                "connected_at": info["connected_at"],
                "tools_count": len(info["tools"]),
                "resources_count": len(info["resources"]),
                "prompts_count": len(info["prompts"])
            })
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
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            })
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
            resources.append({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mimeType
            })
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
            prompts.append({
                "name": prompt.name,
                "description": prompt.description,
                "arguments": [
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required
                    }
                    for arg in (prompt.arguments or [])
                ]
            })
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
        server_ids = list(self.sessions.keys())
        for server_id in server_ids:
            await self.disconnect_from_server(server_id)
        logger.info("Disconnected from all MCP servers")

    async def _teardown_connection(
        self,
        server_id: str,
        *,
        session: Optional[_StdioJSONRPCConnection] = None,
    ) -> None:
        active_session = session if session is not None else self.sessions.get(server_id)

        if active_session is not None:
            try:
                await active_session.close()
            except Exception:
                pass

        self.sessions.pop(server_id, None)
        self.servers.pop(server_id, None)
