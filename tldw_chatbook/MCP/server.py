"""
MCP Server implementation for tldw_chatbook

This module provides the main MCP server that exposes tldw_chatbook's functionality
through the Model Context Protocol.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import MCP server components conditionally
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.models import InitializationOptions
    from mcp.types import Tool, Resource, Prompt, TextContent, ImageContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None

from loguru import logger

# Import tldw_chatbook components
from ..config import get_cli_setting, CLI_APP_CLIENT_ID
from ..DB.ChaChaNotes_DB import ChaChaNotes_DB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Chat.Chat_Functions import create_chat_message, save_conversation
from ..Notes.Notes_Library import NotesInteropService
from ..Character_Chat.Character_Chat_Lib import CharacterInteropService
from ..LLM_Calls.LLM_API_Calls import chat_with_provider

# Import MCP components
from .tools import MCPTools
from .resources import MCPResources
from .prompts import MCPPrompts


class TldwMCPServer:
    """MCP Server for tldw_chatbook"""
    
    def __init__(self, name: str = "tldw_chatbook", version: str = "0.1.0"):
        """Initialize the MCP server."""
        if not MCP_AVAILABLE:
            raise ImportError("MCP dependencies not available. Install with: pip install tldw-chatbook[mcp]")
        
        self.name = name
        self.version = version
        self.mcp = FastMCP(name)
        
        # Initialize databases
        self._init_databases()
        
        # Initialize MCP components
        self.tools = MCPTools(self.chachanotes_db, self.media_db)
        self.resources = MCPResources(self.chachanotes_db, self.media_db)
        self.prompts = MCPPrompts(self.chachanotes_db, self.media_db)
        
        # Register tools, resources, and prompts
        self._register_tools()
        self._register_resources()
        self._register_prompts()
        
        logger.info(f"MCP Server '{name}' initialized")
    
    def _init_databases(self):
        """Initialize database connections."""
        try:
            # Initialize character/chat/notes database
            self.chachanotes_db = ChaChaNotes_DB(
                db_name="chachanotes_db.sqlite",
                client_id=CLI_APP_CLIENT_ID
            )
            
            # Initialize media database
            media_db_path = get_cli_setting("database", "media_db", "media_library.db")
            self.media_db = MediaDatabase(media_db_path)
            
            # Initialize services
            self.notes_service = NotesInteropService(self.chachanotes_db)
            self.character_service = CharacterInteropService(self.chachanotes_db)
            
            logger.info("Databases initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            raise
    
    def _register_tools(self):
        """Register MCP tools."""
        
        # Basic chat tool
        @self.mcp.tool()
        async def chat_with_llm(
            message: str,
            provider: str = "openai",
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 4096,
            conversation_id: Optional[int] = None
        ) -> Dict[str, Any]:
            """Send a message to an LLM and get a response."""
            # For basic chat, we'll implement directly here
            try:
                api_key = get_cli_setting("API", f"{provider.lower()}_api_key", "")
                if not api_key:
                    return {"error": f"No API key configured for {provider}"}
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": message})
                
                response = await asyncio.to_thread(
                    chat_with_provider,
                    api_key=api_key,
                    provider=provider,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "response": response,
                    "conversation_id": conversation_id or "new_conversation"
                }
            except Exception as e:
                logger.error(f"Error in chat_with_llm: {e}")
                return {"error": str(e)}
        
        # Character chat tool
        @self.mcp.tool()
        async def chat_with_character(
            message: str,
            character_id: int,
            provider: str = "openai",
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 4096,
            conversation_id: Optional[int] = None
        ) -> Dict[str, Any]:
            """Chat with a specific character."""
            return await self.tools.chat_with_character(
                message=message,
                character_id=character_id,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                conversation_id=conversation_id
            )
        
        # RAG search tool
        @self.mcp.tool()
        async def search_rag(
            query: str,
            limit: int = 10,
            media_types: Optional[List[str]] = None,
            use_semantic: bool = True
        ) -> List[Dict[str, Any]]:
            """Search the RAG database for relevant content."""
            return await self.tools.perform_rag_search(
                query=query,
                limit=limit,
                media_types=media_types,
                use_semantic=use_semantic
            )
        
        # Search conversations tool
        @self.mcp.tool()
        async def search_conversations(
            query: str,
            limit: int = 10,
            character_id: Optional[int] = None
        ) -> List[Dict[str, Any]]:
            """Search conversations by content."""
            return await self.tools.search_conversations(
                query=query,
                limit=limit,
                character_id=character_id
            )
        
        # Note creation tool
        @self.mcp.tool()
        async def create_note(
            title: str,
            content: str,
            tags: Optional[List[str]] = None,
            template: Optional[str] = None
        ) -> Dict[str, Any]:
            """Create a new note."""
            try:
                note_id = await asyncio.to_thread(
                    self.notes_service.create_note,
                    title=title,
                    content=content,
                    tags=tags or [],
                    template=template
                )
                return {
                    "id": note_id,
                    "title": title,
                    "created": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error in create_note: {e}")
                return {"error": str(e)}
        
        # Note search tool
        @self.mcp.tool()
        async def search_notes(
            query: str,
            limit: int = 10
        ) -> List[Dict[str, Any]]:
            """Search notes by content or title."""
            try:
                results = await asyncio.to_thread(
                    self.notes_service.search_notes,
                    query=query,
                    limit=limit
                )
                return [
                    {
                        "id": note.id,
                        "title": note.title,
                        "preview": note.content[:200] + "..." if len(note.content) > 200 else note.content,
                        "created": note.created_at,
                        "modified": note.updated_at
                    }
                    for note in results
                ]
            except Exception as e:
                logger.error(f"Error in search_notes: {e}")
                return [{"error": str(e)}]
        
        # List characters tool
        @self.mcp.tool()
        async def list_characters() -> List[Dict[str, Any]]:
            """List all available characters."""
            return await self.tools.list_available_characters()
        
        # Get conversation history tool
        @self.mcp.tool()
        async def get_conversation_history(
            conversation_id: int,
            limit: Optional[int] = None
        ) -> Dict[str, Any]:
            """Get conversation history."""
            return await self.tools.get_conversation_history(
                conversation_id=conversation_id,
                limit=limit
            )
        
        # Export conversation tool
        @self.mcp.tool()
        async def export_conversation(
            conversation_id: int,
            format: str = "markdown"
        ) -> Dict[str, Any]:
            """Export a conversation in various formats."""
            return await self.tools.export_conversation(
                conversation_id=conversation_id,
                format=format
            )
        
        # Media ingestion tool (placeholder)
        @self.mcp.tool()
        async def ingest_media(
            url: Optional[str] = None,
            file_path: Optional[str] = None,
            media_type: str = "auto",
            title: Optional[str] = None,
            tags: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """Ingest media from URL or file path."""
            try:
                if not url and not file_path:
                    return {"error": "Either url or file_path must be provided"}
                
                # TODO: Implement actual media ingestion
                return {
                    "status": "queued",
                    "media_id": "placeholder_id",
                    "message": "Media ingestion queued"
                }
            except Exception as e:
                logger.error(f"Error in ingest_media: {e}")
                return {"error": str(e)}
    
    def _register_resources(self):
        """Register MCP resources."""
        
        @self.mcp.resource("conversation://{conversation_id}")
        async def get_conversation(conversation_id: str) -> Dict[str, Any]:
            """Get a conversation by ID."""
            return await self.resources.get_conversation_resource(conversation_id)
        
        @self.mcp.resource("note://{note_id}")
        async def get_note(note_id: str) -> Dict[str, Any]:
            """Get a note by ID."""
            return await self.resources.get_note_resource(note_id)
        
        @self.mcp.resource("character://{character_id}")
        async def get_character(character_id: str) -> Dict[str, Any]:
            """Get a character profile by ID."""
            return await self.resources.get_character_resource(character_id)
        
        @self.mcp.resource("media://{media_id}")
        async def get_media(media_id: str) -> Dict[str, Any]:
            """Get media content by ID."""
            return await self.resources.get_media_resource(media_id)
        
        @self.mcp.resource("rag-chunk://{chunk_id}")
        async def get_rag_chunk(chunk_id: str) -> Dict[str, Any]:
            """Get a RAG chunk by ID."""
            return await self.resources.get_rag_chunk_resource(chunk_id)
        
        # List resources
        @self.mcp.list_resources()
        async def list_resources() -> List[Dict[str, Any]]:
            """List available resources."""
            resources = []
            
            # Add recent conversations
            recent_convs = await self.resources.list_recent_conversations(limit=5)
            resources.extend(recent_convs)
            
            # Add recent notes
            recent_notes = await self.resources.list_recent_notes(limit=5)
            resources.extend(recent_notes)
            
            return resources
    
    def _register_prompts(self):
        """Register MCP prompts."""
        
        @self.mcp.prompt()
        async def summarize_conversation(
            conversation_id: int,
            style: str = "concise",
            focus: Optional[str] = None
        ) -> List[Dict[str, str]]:
            """Generate a prompt to summarize a conversation."""
            return await self.prompts.summarize_conversation_prompt(
                conversation_id=conversation_id,
                style=style,
                focus=focus
            )
        
        @self.mcp.prompt()
        async def generate_document(
            conversation_id: int,
            doc_type: str = "summary",
            format: str = "markdown"
        ) -> List[Dict[str, str]]:
            """Generate a prompt to create a document from a conversation."""
            return await self.prompts.generate_document_prompt(
                conversation_id=conversation_id,
                doc_type=doc_type,
                format=format
            )
        
        @self.mcp.prompt()
        async def analyze_media(
            media_id: int,
            analysis_type: str = "summary",
            detail_level: str = "medium"
        ) -> List[Dict[str, str]]:
            """Generate a prompt to analyze ingested media."""
            return await self.prompts.analyze_media_prompt(
                media_id=media_id,
                analysis_type=analysis_type,
                detail_level=detail_level
            )
        
        @self.mcp.prompt()
        async def search_and_synthesize(
            query: str,
            num_sources: int = 5,
            synthesis_type: str = "overview"
        ) -> List[Dict[str, str]]:
            """Generate a prompt to search RAG and synthesize results."""
            return await self.prompts.search_and_synthesize_prompt(
                query=query,
                num_sources=num_sources,
                synthesis_type=synthesis_type
            )
        
        @self.mcp.prompt()
        async def character_writing(
            character_id: int,
            writing_type: str = "response",
            context: Optional[str] = None,
            style_notes: Optional[str] = None
        ) -> List[Dict[str, str]]:
            """Generate a prompt for character-based writing."""
            return await self.prompts.character_writing_prompt(
                character_id=character_id,
                writing_type=writing_type,
                context=context,
                style_notes=style_notes
            )
    
    async def run(self, transport: str = "stdio"):
        """Run the MCP server.
        
        Args:
            transport: Transport type (stdio, http)
        """
        if transport == "stdio":
            # Run with stdio transport (for Claude Desktop)
            import sys
            from mcp.server.stdio import stdio_server
            
            async with stdio_server() as (read_stream, write_stream):
                await self.mcp.run(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    InitializationOptions(
                        server_name=self.name,
                        server_version=self.version
                    )
                )
        else:
            # TODO: Implement HTTP transport
            raise NotImplementedError(f"Transport {transport} not implemented yet")


async def main():
    """Main entry point for running the MCP server."""
    server = TldwMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())