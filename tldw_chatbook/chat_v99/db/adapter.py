"""
Database adapter for Chat v99 to interface with existing ChaChaNotes_DB.

This adapter provides a reactive-friendly interface to the existing database
while maintaining full compatibility with the current schema.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from uuid import uuid4
import json

from loguru import logger

# Import existing database
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB, 
    ConflictError, 
    CharactersRAGDBError, 
    InputError
)
from tldw_chatbook.config import get_chachanotes_db_lazy

# Import Chat v99 models
from ..models import ChatMessage, ChatSession


class DatabaseChangeEvent:
    """Event emitted when database changes occur."""
    def __init__(self, event_type: str, data: Any):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()


@dataclass
class ConversationRecord:
    """Database record for a conversation."""
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    character_id: Optional[int] = None
    character_name: Optional[str] = None
    keywords: Optional[str] = None
    message_count: int = 0
    is_ephemeral: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ChatV99DatabaseAdapter:
    """
    Adapter to connect Chat v99's reactive architecture to the existing database.
    
    This adapter:
    - Provides async methods for all database operations
    - Converts between Chat v99 models and database formats
    - Handles transactions and error recovery
    - Emits events for state synchronization
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database adapter.
        
        Args:
            db_path: Optional path to database. If None, uses default from config.
        """
        self._db_path = db_path
        self._db: Optional[CharactersRAGDB] = None
        self._listeners: List[Callable[[DatabaseChangeEvent], None]] = []
        self._transaction_lock = asyncio.Lock()
        
        logger.info("ChatV99DatabaseAdapter initialized")
    
    @property
    def db(self) -> CharactersRAGDB:
        """Get or create database instance."""
        if self._db is None:
            if self._db_path:
                self._db = CharactersRAGDB(self._db_path)
            else:
                self._db = get_chachanotes_db_lazy()
        return self._db
    
    def add_listener(self, callback: Callable[[DatabaseChangeEvent], None]):
        """Add a listener for database change events."""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[DatabaseChangeEvent], None]):
        """Remove a listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _emit_event(self, event_type: str, data: Any):
        """Emit an event to all listeners."""
        event = DatabaseChangeEvent(event_type, data)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error in database event listener: {e}")
    
    # ==================== Session Management ====================
    
    async def create_session(self, session: ChatSession) -> ChatSession:
        """
        Create a new chat session in the database.
        
        Args:
            session: The ChatSession to create
            
        Returns:
            The created session with database ID
        """
        async with self._transaction_lock:
            try:
                # Generate conversation ID if not provided
                if not session.id:
                    session.id = str(uuid4())
                
                # Convert to database format
                keywords = ", ".join(session.keywords) if session.keywords else ""
                
                # Create conversation in database
                conv_id = await asyncio.to_thread(
                    self.db.add_new_conversation,
                    title=session.title,
                    conversation_id=session.id,
                    keywords=keywords,
                    character_id=session.character_id
                )
                
                # Update session with database ID
                session.id = conv_id
                session.created_at = datetime.now()
                session.updated_at = datetime.now()
                
                # Emit event
                self._emit_event("session_created", session)
                
                logger.info(f"Created session: {session.id}")
                return session
                
            except (CharactersRAGDBError, InputError) as e:
                logger.error(f"Failed to create session: {e}")
                raise
    
    async def load_session(self, conversation_id: str) -> Optional[ChatSession]:
        """
        Load a session from the database.
        
        Args:
            conversation_id: The ID of the conversation to load
            
        Returns:
            The loaded ChatSession or None if not found
        """
        try:
            # Load conversation details
            conv_data = await asyncio.to_thread(
                self.db.get_conversation_details,
                conversation_id
            )
            
            if not conv_data:
                return None
            
            # Load messages
            messages = await self.load_messages(conversation_id)
            
            # Create ChatSession
            session = ChatSession(
                id=conversation_id,
                title=conv_data.get('title', 'Untitled'),
                messages=messages,
                created_at=datetime.fromisoformat(conv_data.get('created_at', '')),
                updated_at=datetime.fromisoformat(conv_data.get('updated_at', '')),
                metadata={
                    'keywords': conv_data.get('keywords', ''),
                    'character_id': conv_data.get('character_id'),
                    'character_name': conv_data.get('character_name')
                }
            )
            
            logger.info(f"Loaded session: {conversation_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {conversation_id}: {e}")
            return None
    
    async def save_session(self, session: ChatSession) -> bool:
        """
        Save or update a session in the database.
        
        Args:
            session: The ChatSession to save
            
        Returns:
            True if successful, False otherwise
        """
        async with self._transaction_lock:
            try:
                # Update conversation details
                keywords = ", ".join(session.keywords) if session.keywords else ""
                
                await asyncio.to_thread(
                    self.db.update_conversation_details,
                    conversation_id=session.id,
                    title=session.title,
                    keywords=keywords
                )
                
                # Save all messages
                for message in session.messages:
                    await self.save_message(session.id, message)
                
                # Update timestamp
                session.updated_at = datetime.now()
                
                # Emit event
                self._emit_event("session_saved", session)
                
                logger.info(f"Saved session: {session.id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save session {session.id}: {e}")
                return False
    
    async def delete_session(self, conversation_id: str) -> bool:
        """
        Delete a session from the database.
        
        Args:
            conversation_id: The ID of the conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = await asyncio.to_thread(
                self.db.delete_conversation,
                conversation_id
            )
            
            if success:
                self._emit_event("session_deleted", conversation_id)
                logger.info(f"Deleted session: {conversation_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete session {conversation_id}: {e}")
            return False
    
    async def list_sessions(self, 
                           search_query: Optional[str] = None,
                           character_id: Optional[int] = None,
                           limit: int = 50) -> List[ConversationRecord]:
        """
        List conversations from the database.
        
        Args:
            search_query: Optional search query
            character_id: Optional character filter
            limit: Maximum number of results
            
        Returns:
            List of conversation records
        """
        try:
            # Use existing search functionality
            results = await asyncio.to_thread(
                self.db.search_conversations,
                query=search_query,
                character_id=character_id,
                limit=limit
            )
            
            # Convert to ConversationRecord objects
            conversations = []
            for row in results:
                conv = ConversationRecord(
                    conversation_id=row[0],
                    title=row[1],
                    created_at=row[2],
                    updated_at=row[3],
                    character_id=row[4] if len(row) > 4 else None,
                    character_name=row[5] if len(row) > 5 else None,
                    keywords=row[6] if len(row) > 6 else None,
                    message_count=row[7] if len(row) > 7 else 0
                )
                conversations.append(conv)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    # ==================== Message Management ====================
    
    async def save_message(self, conversation_id: str, message: ChatMessage) -> bool:
        """
        Save a message to the database.
        
        Args:
            conversation_id: The conversation this message belongs to
            message: The ChatMessage to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert message to database format
            timestamp = message.timestamp or datetime.now().isoformat()
            
            # Handle attachments
            attachments = None
            if message.attachments:
                # Store attachments as JSON
                attachments = json.dumps(message.attachments)
            
            # Handle tool messages
            if message.role == "tool":
                # Store tool messages with special formatting
                await asyncio.to_thread(
                    self.db.add_tool_message,
                    conversation_id=conversation_id,
                    tool_name=message.tool_name or "unknown",
                    tool_input=message.tool_input or {},
                    tool_output=message.content,
                    timestamp=timestamp
                )
            else:
                # Regular message
                await asyncio.to_thread(
                    self.db.add_message,
                    conversation_id=conversation_id,
                    role=message.role,
                    message=message.content,
                    timestamp=timestamp,
                    attachments=attachments
                )
            
            # Emit event
            self._emit_event("message_saved", {
                "conversation_id": conversation_id,
                "message": message
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    async def load_messages(self, conversation_id: str) -> List[ChatMessage]:
        """
        Load all messages for a conversation.
        
        Args:
            conversation_id: The conversation to load messages for
            
        Returns:
            List of ChatMessage objects
        """
        try:
            # Load messages from database
            raw_messages = await asyncio.to_thread(
                self.db.get_messages_for_conversation,
                conversation_id
            )
            
            # Convert to ChatMessage objects
            messages = []
            for msg in raw_messages:
                # Parse message data
                role = msg.get('role', 'user')
                content = msg.get('message', '')
                timestamp = msg.get('timestamp')
                
                # Parse attachments if present
                attachments = None
                if msg.get('attachments'):
                    try:
                        attachments = json.loads(msg['attachments'])
                    except json.JSONDecodeError:
                        attachments = None
                
                # Create ChatMessage
                message = ChatMessage(
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    attachments=attachments
                )
                
                # Handle tool messages
                if role == "tool":
                    message.tool_name = msg.get('tool_name')
                    message.tool_input = msg.get('tool_input')
                
                messages.append(message)
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to load messages for {conversation_id}: {e}")
            return []
    
    async def delete_message(self, conversation_id: str, message_index: int) -> bool:
        """
        Delete a message from a conversation.
        
        Args:
            conversation_id: The conversation ID
            message_index: The index of the message to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load current messages
            messages = await self.load_messages(conversation_id)
            
            if 0 <= message_index < len(messages):
                # Remove the message
                deleted_message = messages.pop(message_index)
                
                # Clear and re-save all messages (maintaining order)
                # This is not ideal but works with current DB structure
                await asyncio.to_thread(
                    self.db.clear_messages_for_conversation,
                    conversation_id
                )
                
                for msg in messages:
                    await self.save_message(conversation_id, msg)
                
                # Emit event
                self._emit_event("message_deleted", {
                    "conversation_id": conversation_id,
                    "message": deleted_message,
                    "index": message_index
                })
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
            return False
    
    # ==================== Character Management ====================
    
    async def load_character(self, character_id: int) -> Optional[Dict[str, Any]]:
        """
        Load character data from the database.
        
        Args:
            character_id: The ID of the character to load
            
        Returns:
            Character data dictionary or None
        """
        try:
            character = await asyncio.to_thread(
                self.db.get_character_by_id,
                character_id
            )
            
            if character:
                logger.info(f"Loaded character: {character_id}")
            
            return character
            
        except Exception as e:
            logger.error(f"Failed to load character {character_id}: {e}")
            return None
    
    async def list_characters(self) -> List[Dict[str, Any]]:
        """
        List all available characters.
        
        Returns:
            List of character dictionaries
        """
        try:
            characters = await asyncio.to_thread(
                self.db.get_all_characters
            )
            return characters or []
            
        except Exception as e:
            logger.error(f"Failed to list characters: {e}")
            return []
    
    # ==================== Transaction Support ====================
    
    async def begin_transaction(self):
        """Begin a database transaction."""
        await self._transaction_lock.acquire()
        # Note: Actual transaction handled by context manager in DB
    
    async def commit_transaction(self):
        """Commit the current transaction."""
        self._transaction_lock.release()
    
    async def rollback_transaction(self):
        """Rollback the current transaction."""
        self._transaction_lock.release()
    
    # ==================== Utility Methods ====================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = await asyncio.to_thread(
                self.db.get_database_statistics
            )
            return stats or {}
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def optimize_database(self) -> bool:
        """
        Optimize the database (VACUUM, etc.).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            await asyncio.to_thread(
                self.db.vacuum_database
            )
            logger.info("Database optimized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self._db:
            try:
                self._db.close()
                self._db = None
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")