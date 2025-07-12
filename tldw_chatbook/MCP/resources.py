"""
MCP Resources implementation for tldw_chatbook

This module provides resource access to tldw_chatbook's data through MCP.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import base64
from pathlib import Path

from loguru import logger

# Import tldw_chatbook components
from ..DB.ChaChaNotes_DB import ChaChaNotes_DB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Notes.Notes_Library import get_note_by_id
from ..Character_Chat.Character_Chat_Lib import get_character_by_id


class MCPResources:
    """Container for MCP resource implementations."""
    
    def __init__(self, chachanotes_db: ChaChaNotes_DB, media_db: MediaDatabase):
        """Initialize resources with database connections."""
        self.chachanotes_db = chachanotes_db
        self.media_db = media_db
    
    async def get_conversation_resource(self, conversation_id: str) -> Dict[str, Any]:
        """Get a conversation as a resource.
        
        Args:
            conversation_id: ID of the conversation
        
        Returns:
            Resource dict with content and metadata
        """
        try:
            conv_id = int(conversation_id)
            
            # Get conversation details
            conv = self.chachanotes_db.get_conversation_by_id(conv_id)
            if not conv:
                return {
                    "uri": f"conversation://{conversation_id}",
                    "name": "Not Found",
                    "mimeType": "text/plain",
                    "content": "Conversation not found"
                }
            
            # Get messages
            messages = self.chachanotes_db.get_conversation_messages(conv_id)
            
            # Format as markdown
            content = f"# {conv['title']}\n\n"
            content += f"*Created: {conv['created_at']}*\n\n"
            
            if conv.get('character_id'):
                char = get_character_by_id(self.chachanotes_db, conv['character_id'])
                if char:
                    content += f"**Character**: {char['name']}\n\n"
            
            content += "---\n\n"
            
            for msg in messages:
                role = msg['role'].capitalize()
                content += f"### {role}\n\n{msg['content']}\n\n"
            
            return {
                "uri": f"conversation://{conversation_id}",
                "name": conv['title'],
                "mimeType": "text/markdown",
                "content": content,
                "metadata": {
                    "created": conv['created_at'],
                    "updated": conv.get('updated_at'),
                    "character_id": conv.get('character_id'),
                    "message_count": len(messages)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation resource: {e}")
            return {
                "uri": f"conversation://{conversation_id}",
                "name": "Error",
                "mimeType": "text/plain",
                "content": f"Error loading conversation: {str(e)}"
            }
    
    async def get_note_resource(self, note_id: str) -> Dict[str, Any]:
        """Get a note as a resource.
        
        Args:
            note_id: ID of the note
        
        Returns:
            Resource dict with content and metadata
        """
        try:
            note = get_note_by_id(self.chachanotes_db, int(note_id))
            if not note:
                return {
                    "uri": f"note://{note_id}",
                    "name": "Not Found",
                    "mimeType": "text/plain",
                    "content": "Note not found"
                }
            
            # Format note content
            content = f"# {note['title']}\n\n"
            
            if note.get('tags'):
                tags = ", ".join(note['tags'])
                content += f"*Tags: {tags}*\n\n"
            
            content += f"*Created: {note['created_at']}*\n"
            if note.get('updated_at'):
                content += f"*Updated: {note['updated_at']}*\n"
            
            content += "\n---\n\n"
            content += note['content']
            
            return {
                "uri": f"note://{note_id}",
                "name": note['title'],
                "mimeType": "text/markdown",
                "content": content,
                "metadata": {
                    "created": note['created_at'],
                    "updated": note.get('updated_at'),
                    "tags": note.get('tags', []),
                    "template": note.get('template')
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting note resource: {e}")
            return {
                "uri": f"note://{note_id}",
                "name": "Error",
                "mimeType": "text/plain",
                "content": f"Error loading note: {str(e)}"
            }
    
    async def get_character_resource(self, character_id: str) -> Dict[str, Any]:
        """Get a character profile as a resource.
        
        Args:
            character_id: ID of the character
        
        Returns:
            Resource dict with content and metadata
        """
        try:
            char = get_character_by_id(self.chachanotes_db, int(character_id))
            if not char:
                return {
                    "uri": f"character://{character_id}",
                    "name": "Not Found",
                    "mimeType": "text/plain",
                    "content": "Character not found"
                }
            
            # Format character profile
            content = f"# {char['name']}\n\n"
            
            if char.get('description'):
                content += f"## Description\n\n{char['description']}\n\n"
            
            if char.get('personality'):
                content += f"## Personality\n\n{char['personality']}\n\n"
            
            if char.get('scenario'):
                content += f"## Scenario\n\n{char['scenario']}\n\n"
            
            if char.get('greeting'):
                content += f"## Greeting\n\n{char['greeting']}\n\n"
            
            if char.get('example_dialogue'):
                content += f"## Example Dialogue\n\n{char['example_dialogue']}\n\n"
            
            # Add metadata
            content += f"\n---\n\n"
            content += f"*Created: {char['created_at']}*\n"
            
            return {
                "uri": f"character://{character_id}",
                "name": char['name'],
                "mimeType": "text/markdown",
                "content": content,
                "metadata": {
                    "created": char['created_at'],
                    "updated": char.get('updated_at'),
                    "message_count": char.get('message_count', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting character resource: {e}")
            return {
                "uri": f"character://{character_id}",
                "name": "Error",
                "mimeType": "text/plain",
                "content": f"Error loading character: {str(e)}"
            }
    
    async def get_media_resource(self, media_id: str) -> Dict[str, Any]:
        """Get media content as a resource.
        
        Args:
            media_id: ID of the media
        
        Returns:
            Resource dict with content and metadata
        """
        try:
            # Get media entry
            media = self.media_db.get_media_by_id(int(media_id))
            if not media:
                return {
                    "uri": f"media://{media_id}",
                    "name": "Not Found",
                    "mimeType": "text/plain",
                    "content": "Media not found"
                }
            
            # Get transcript/content
            transcript = self.media_db.get_media_transcript(int(media_id))
            
            # Format content
            content = f"# {media['title']}\n\n"
            content += f"*Type: {media['media_type']}*\n"
            content += f"*Created: {media['created_at']}*\n\n"
            
            if media.get('url'):
                content += f"**Source**: {media['url']}\n\n"
            
            content += "---\n\n"
            
            if transcript:
                content += "## Transcript/Content\n\n"
                content += transcript
            else:
                content += "## Summary\n\n"
                content += media.get('content', 'No content available')
            
            return {
                "uri": f"media://{media_id}",
                "name": media['title'],
                "mimeType": "text/markdown",
                "content": content,
                "metadata": {
                    "media_type": media['media_type'],
                    "url": media.get('url'),
                    "created": media['created_at'],
                    "duration": media.get('duration'),
                    "author": media.get('author')
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting media resource: {e}")
            return {
                "uri": f"media://{media_id}",
                "name": "Error",
                "mimeType": "text/plain",
                "content": f"Error loading media: {str(e)}"
            }
    
    async def get_rag_chunk_resource(self, chunk_id: str) -> Dict[str, Any]:
        """Get a RAG chunk as a resource.
        
        Args:
            chunk_id: ID of the chunk
        
        Returns:
            Resource dict with content and metadata
        """
        try:
            # Get chunk from media database
            chunk = self.media_db.get_chunk_by_id(int(chunk_id))
            if not chunk:
                return {
                    "uri": f"rag-chunk://{chunk_id}",
                    "name": "Not Found",
                    "mimeType": "text/plain",
                    "content": "Chunk not found"
                }
            
            # Get parent media info
            media = self.media_db.get_media_by_id(chunk['media_id'])
            
            # Format content
            content = f"# RAG Chunk {chunk_id}\n\n"
            if media:
                content += f"**From**: {media['title']}\n"
            content += f"**Position**: {chunk.get('start_char', 0)} - {chunk.get('end_char', 0)}\n\n"
            content += "---\n\n"
            content += chunk['text']
            
            return {
                "uri": f"rag-chunk://{chunk_id}",
                "name": f"Chunk from {media['title'] if media else 'Unknown'}",
                "mimeType": "text/plain",
                "content": content,
                "metadata": {
                    "media_id": chunk['media_id'],
                    "start_char": chunk.get('start_char'),
                    "end_char": chunk.get('end_char'),
                    "embedding_id": chunk.get('embedding_id')
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting RAG chunk resource: {e}")
            return {
                "uri": f"rag-chunk://{chunk_id}",
                "name": "Error",
                "mimeType": "text/plain",
                "content": f"Error loading chunk: {str(e)}"
            }
    
    async def list_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent conversations as resources.
        
        Args:
            limit: Maximum number of conversations
        
        Returns:
            List of resource references
        """
        try:
            conversations = self.chachanotes_db.get_recent_conversations(limit=limit)
            
            resources = []
            for conv in conversations:
                resources.append({
                    "uri": f"conversation://{conv['id']}",
                    "name": conv['title'],
                    "mimeType": "text/markdown",
                    "description": f"Conversation from {conv['created_at']}"
                })
            
            return resources
            
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    async def list_recent_notes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent notes as resources.
        
        Args:
            limit: Maximum number of notes
        
        Returns:
            List of resource references
        """
        try:
            notes = self.chachanotes_db.get_recent_notes(limit=limit)
            
            resources = []
            for note in notes:
                resources.append({
                    "uri": f"note://{note['id']}",
                    "name": note['title'],
                    "mimeType": "text/markdown",
                    "description": f"Note from {note['created_at']}"
                })
            
            return resources
            
        except Exception as e:
            logger.error(f"Error listing notes: {e}")
            return []