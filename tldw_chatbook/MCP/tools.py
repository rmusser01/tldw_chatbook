"""
MCP Tools implementation for tldw_chatbook

This module provides the tool implementations that expose tldw_chatbook's
functionality through MCP.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import json

from loguru import logger

# Import tldw_chatbook components
from ..config import get_cli_setting, CLI_APP_CLIENT_ID
from ..DB.ChaChaNotes_DB import ChaChaNotes_DB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Chat.Chat_Functions import save_conversation_from_messages
from ..LLM_Calls.LLM_API_Calls import chat_with_provider
from ..RAG_Search.simplified.search_service import SimplifiedRAGSearchService
from ..Character_Chat.Character_Chat_Lib import get_character_by_id, list_characters


class MCPTools:
    """Container for MCP tool implementations."""
    
    def __init__(self, chachanotes_db: ChaChaNotes_DB, media_db: MediaDatabase):
        """Initialize tools with database connections."""
        self.chachanotes_db = chachanotes_db
        self.media_db = media_db
        self.rag_service = SimplifiedRAGSearchService(media_db)
    
    async def chat_with_character(
        self,
        message: str,
        character_id: int,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        conversation_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Chat with a specific character.
        
        Args:
            message: The message to send
            character_id: ID of the character to chat with
            provider: LLM provider
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            conversation_id: Optional conversation ID to continue
        
        Returns:
            Dict with response and conversation details
        """
        try:
            # Get character
            character = get_character_by_id(self.chachanotes_db, character_id)
            if not character:
                return {"error": f"Character {character_id} not found"}
            
            # Get API key
            api_key = get_cli_setting("API", f"{provider.lower()}_api_key", "")
            if not api_key:
                return {"error": f"No API key configured for {provider}"}
            
            # Build messages with character context
            messages = []
            
            # Add character system prompt
            system_prompt = f"You are {character['name']}. {character.get('description', '')}"
            if character.get('personality'):
                system_prompt += f"\n\nPersonality: {character['personality']}"
            messages.append({"role": "system", "content": system_prompt})
            
            # Load conversation history if continuing
            if conversation_id:
                conv_messages = self.chachanotes_db.get_conversation_messages(conversation_id)
                for msg in conv_messages:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # Add new message
            messages.append({"role": "user", "content": message})
            
            # Call LLM
            response = await asyncio.to_thread(
                chat_with_provider,
                api_key=api_key,
                provider=provider,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Save conversation
            if not conversation_id:
                conversation_id = save_conversation_from_messages(
                    db=self.chachanotes_db,
                    messages=messages + [{"role": "assistant", "content": response}],
                    title=f"Chat with {character['name']}",
                    character_id=character_id
                )
            else:
                # Add new messages to existing conversation
                self.chachanotes_db.add_message_to_conversation(
                    conversation_id=conversation_id,
                    role="user",
                    content=message
                )
                self.chachanotes_db.add_message_to_conversation(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response
                )
            
            return {
                "response": response,
                "conversation_id": conversation_id,
                "character_name": character['name']
            }
            
        except Exception as e:
            logger.error(f"Error in chat_with_character: {e}")
            return {"error": str(e)}
    
    async def search_conversations(
        self,
        query: str,
        limit: int = 10,
        character_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search conversations by content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            character_id: Optional character ID to filter by
        
        Returns:
            List of matching conversations
        """
        try:
            results = await asyncio.to_thread(
                self.chachanotes_db.search_all_content,
                search_query=query,
                content_type="conversation",
                limit=limit
            )
            
            conversations = []
            for result in results:
                if character_id and result.get('character_id') != character_id:
                    continue
                
                conversations.append({
                    "id": result['id'],
                    "title": result['title'],
                    "preview": result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                    "created": result['created_at'],
                    "character_id": result.get('character_id'),
                    "message_count": result.get('message_count', 0)
                })
            
            return conversations
        except Exception as e:
            logger.error(f"Error in search_conversations: {e}")
            return [{"error": str(e)}]
    
    async def perform_rag_search(
        self,
        query: str,
        limit: int = 10,
        media_types: Optional[List[str]] = None,
        use_semantic: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform RAG search across ingested media.
        
        Args:
            query: Search query
            limit: Maximum number of results
            media_types: Optional list of media types to filter
            use_semantic: Whether to use semantic search (if available)
        
        Returns:
            List of search results with content and metadata
        """
        try:
            # Perform search
            if use_semantic and hasattr(self.rag_service, 'semantic_search'):
                results = await asyncio.to_thread(
                    self.rag_service.semantic_search,
                    query=query,
                    limit=limit,
                    media_types=media_types
                )
            else:
                results = await asyncio.to_thread(
                    self.rag_service.keyword_search,
                    query=query,
                    limit=limit,
                    media_types=media_types
                )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get('id'),
                    "title": result.get('title', 'Untitled'),
                    "content": result.get('content', ''),
                    "media_type": result.get('media_type', 'unknown'),
                    "source": result.get('url') or result.get('file_path', ''),
                    "score": result.get('score', 0.0),
                    "metadata": result.get('metadata', {})
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error in perform_rag_search: {e}")
            return [{"error": str(e)}]
    
    async def list_available_characters(self) -> List[Dict[str, Any]]:
        """List all available characters.
        
        Returns:
            List of character profiles
        """
        try:
            characters = list_characters(self.chachanotes_db)
            return [
                {
                    "id": char['id'],
                    "name": char['name'],
                    "description": char.get('description', ''),
                    "message_count": char.get('message_count', 0)
                }
                for char in characters
            ]
        except Exception as e:
            logger.error(f"Error listing characters: {e}")
            return [{"error": str(e)}]
    
    async def get_conversation_history(
        self,
        conversation_id: int,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get conversation history.
        
        Args:
            conversation_id: ID of the conversation
            limit: Optional limit on number of messages
        
        Returns:
            Dict with conversation details and messages
        """
        try:
            # Get conversation details
            conv_details = self.chachanotes_db.get_conversation_by_id(conversation_id)
            if not conv_details:
                return {"error": f"Conversation {conversation_id} not found"}
            
            # Get messages
            messages = self.chachanotes_db.get_conversation_messages(conversation_id)
            if limit:
                messages = messages[-limit:]
            
            return {
                "id": conversation_id,
                "title": conv_details['title'],
                "created": conv_details['created_at'],
                "character_id": conv_details.get('character_id'),
                "messages": [
                    {
                        "role": msg['role'],
                        "content": msg['content'],
                        "timestamp": msg.get('created_at')
                    }
                    for msg in messages
                ]
            }
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return {"error": str(e)}
    
    async def export_conversation(
        self,
        conversation_id: int,
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """Export a conversation in various formats.
        
        Args:
            conversation_id: ID of the conversation
            format: Export format (markdown, json, text)
        
        Returns:
            Dict with exported content
        """
        try:
            # Get conversation data
            conv_data = await self.get_conversation_history(conversation_id)
            if "error" in conv_data:
                return conv_data
            
            if format == "markdown":
                # Export as markdown
                content = f"# {conv_data['title']}\n\n"
                content += f"*Created: {conv_data['created']}*\n\n"
                
                for msg in conv_data['messages']:
                    role = msg['role'].capitalize()
                    content += f"## {role}\n\n{msg['content']}\n\n"
                
                return {"format": "markdown", "content": content}
            
            elif format == "json":
                # Export as JSON
                return {"format": "json", "content": json.dumps(conv_data, indent=2)}
            
            elif format == "text":
                # Export as plain text
                content = f"{conv_data['title']}\n"
                content += f"Created: {conv_data['created']}\n"
                content += "-" * 50 + "\n\n"
                
                for msg in conv_data['messages']:
                    content += f"{msg['role'].upper()}: {msg['content']}\n\n"
                
                return {"format": "text", "content": content}
            
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            return {"error": str(e)}