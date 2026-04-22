"""
MCP Client implementation for tldw_chatbook

This module provides client functionality to connect to external MCP servers
and use their tools, resources, and prompts within tldw_chatbook.
"""

from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import MCP client components conditionally
try:
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.session import ClientSession
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    StdioServerParameters = None
    stdio_client = None
    ClientSession = None

from loguru import logger


class MCPClient:
    """MCP Client for connecting to external MCP servers."""
    
    def __init__(self, name: str = "tldw_chatbook_client"):
        """Initialize the MCP client."""
        if not MCP_CLIENT_AVAILABLE:
            raise ImportError("MCP client dependencies not available. Install with: pip install tldw-chatbook[mcp]")
        
        self.name = name
        self.sessions: Dict[str, ClientSession] = {}
        self.servers: Dict[str, Dict[str, Any]] = {}
        self._connection_stacks: Dict[str, AsyncExitStack] = {}
        self._session_context_managed: set[str] = set()
        
        logger.info(f"MCP Client '{name}' initialized")
    
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
        if not all((MCP_CLIENT_AVAILABLE, StdioServerParameters, stdio_client, ClientSession)):
            logger.error("MCP client dependencies not available for stdio server connection")
            return False

        if server_id in self.sessions or server_id in self._connection_stacks:
            await self._teardown_connection(server_id)

        stack = AsyncExitStack()
        session = None
        session_context_managed = False
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args or [],
                env=env
            )

            read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
            session_candidate = ClientSession(read_stream, write_stream)
            if hasattr(session_candidate, "__aenter__") and hasattr(session_candidate, "__aexit__"):
                session = await stack.enter_async_context(session_candidate)
                session_context_managed = True
            else:
                session = session_candidate

            await session.initialize()

            # Store session and server info
            self.sessions[server_id] = session
            self._connection_stacks[server_id] = stack
            if session_context_managed:
                self._session_context_managed.add(server_id)
            self.servers[server_id] = {
                "command": command,
                "args": list(args or []),
                "connected_at": datetime.now().isoformat(),
                "tools": [],
                "resources": [],
                "prompts": []
            }
            
            # Discover server capabilities
            await self._discover_server_capabilities(server_id)
            
            logger.info(f"Successfully connected to MCP server: {server_id}")
            return True

        except Exception as e:
            await self._teardown_connection(
                server_id,
                session=session,
                stack=stack,
                session_context_managed=session_context_managed,
            )
            logger.error(f"Failed to connect to MCP server {server_id}: {e}")
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
                logger.info(f"Disconnected from MCP server: {server_id}")
                return True
            else:
                logger.warning(f"Server {server_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error disconnecting from server {server_id}: {e}")
            return False
    
    async def _discover_server_capabilities(self, server_id: str) -> None:
        """Discover tools, resources, and prompts from a server.
        
        Args:
            server_id: Server identifier
        """
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"Server session not found for {server_id}")

        # List tools
        tools_response = await session.list_tools()
        self.servers[server_id]["tools"] = tools_response.tools

        # List resources
        resources_response = await session.list_resources()
        self.servers[server_id]["resources"] = resources_response.resources

        # List prompts
        prompts_response = await session.list_prompts()
        self.servers[server_id]["prompts"] = prompts_response.prompts

        logger.info(f"Discovered capabilities for {server_id}: "
                   f"{len(self.servers[server_id]['tools'])} tools, "
                   f"{len(self.servers[server_id]['resources'])} resources, "
                   f"{len(self.servers[server_id]['prompts'])} prompts")

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
            
            # Call the tool
            result = await session.call_tool(tool_name, arguments)
            
            # Convert result to dict
            if hasattr(result, 'content'):
                return {"result": result.content}
            else:
                return {"result": str(result)}
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_id}: {e}")
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
            
            # Read the resource
            result = await session.read_resource(resource_uri)
            
            # Convert result to dict
            return {
                "uri": resource_uri,
                "content": result.contents[0].text if result.contents else "",
                "mimeType": result.contents[0].mimeType if result.contents else "text/plain"
            }
                
        except Exception as e:
            logger.error(f"Error reading resource {resource_uri} from {server_id}: {e}")
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
            
            # Get the prompt
            result = await session.get_prompt(prompt_name, arguments or {})
            
            # Convert messages
            messages = []
            for msg in result.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
                })
            
            return messages
                
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_name} from {server_id}: {e}")
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
        session: Optional[Any] = None,
        stack: Optional[AsyncExitStack] = None,
        session_context_managed: Optional[bool] = None,
    ) -> None:
        active_session = session if session is not None else self.sessions.get(server_id)
        active_stack = stack if stack is not None else self._connection_stacks.pop(server_id, None)
        managed_by_context = (
            session_context_managed
            if session_context_managed is not None
            else server_id in self._session_context_managed
        )

        if active_session is not None and not managed_by_context and hasattr(active_session, "close"):
            try:
                await active_session.close()
            except Exception:
                pass

        if active_stack is not None:
            try:
                await active_stack.aclose()
            except Exception:
                pass

        self._session_context_managed.discard(server_id)
        self.sessions.pop(server_id, None)
        self.servers.pop(server_id, None)
