"""
MCP Client implementation for tldw_chatbook

This module provides client functionality to connect to external MCP servers
and use their tools, resources, and prompts within tldw_chatbook.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import MCP client components conditionally
try:
    from mcp.client import Client, StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.types import Tool, Resource, Prompt
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    Client = None
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
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args or [],
                env=env
            )
            
            # Create and store session
            session = ClientSession(
                server=server_params,
                name=f"{self.name}:{server_id}"
            )
            
            # Initialize the session
            await session.initialize()
            
            # Store session and server info
            self.sessions[server_id] = session
            self.servers[server_id] = {
                "command": command,
                "args": args,
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
                session = self.sessions[server_id]
                await session.close()
                del self.sessions[server_id]
                del self.servers[server_id]
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
        try:
            session = self.sessions.get(server_id)
            if not session:
                return
            
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
                       
        except Exception as e:
            logger.error(f"Error discovering capabilities for {server_id}: {e}")
    
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
    
    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        server_ids = list(self.sessions.keys())
        for server_id in server_ids:
            await self.disconnect_from_server(server_id)
        logger.info("Disconnected from all MCP servers")