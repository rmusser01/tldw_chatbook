# indexing_service.py
# Description: Service for indexing documents into vector storage
#
# Imports
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
import asyncio
from pathlib import Path
from loguru import logger
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
#
# Local Imports
from ...DB.Client_Media_DB_v2 import MediaDatabase
from ...DB.ChaChaNotes_DB import CharactersRAGDB
from ...DB.RAG_Indexing_DB import RAGIndexingDB
from .embeddings_service import EmbeddingsService
from .chunking_service import ChunkingService
from ...Utils.paths import get_user_data_dir

logger = logger.bind(module="indexing_service")

class IndexingService:
    """Service for indexing documents into vector storage for RAG"""
    
    def __init__(self, embeddings_service: EmbeddingsService, chunking_service: ChunkingService):
        """
        Initialize the indexing service
        
        Args:
            embeddings_service: Service for creating embeddings
            chunking_service: Service for chunking documents
        """
        self.embeddings_service = embeddings_service
        self.chunking_service = chunking_service
        
        # Initialize indexing tracking database
        db_path = get_user_data_dir() / "databases" / "rag_indexing.db"
        self.indexing_db = RAGIndexingDB(db_path)
        
        # Performance settings
        self.max_workers = 4
        self.enable_parallel_indexing = True
        self._executor = None
        self._executor_lock = threading.Lock()
        
    async def index_media_items(
        self,
        media_db: MediaDatabase,
        batch_size: int = 10,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = True
    ) -> int:
        """
        Index media items into ChromaDB
        
        Args:
            media_db: Media database instance
            batch_size: Number of items to process at once
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            progress_callback: Optional callback for progress updates (type, current, total)
            incremental: If True, only index new/modified items
            
        Returns:
            Number of items indexed
        """
        logger.info(f"Starting media items indexing (incremental={incremental})...")
        indexed_count = 0
        
        try:
            # Get already indexed items if incremental
            indexed_items = {}
            if incremental:
                indexed_items = self.indexing_db.get_indexed_items_by_type("media")
                logger.info(f"Found {len(indexed_items)} already indexed media items")
            
            # First, get total count
            total_count = 0
            items_to_process = []
            
            # Get all media items to check which need indexing
            page = 1
            while True:
                results = await asyncio.to_thread(
                    media_db.fetch_paginated_media,
                    page=page,
                    results_per_page=100  # Larger batch for checking
                )
                
                if not results:
                    break
                    
                media_items = results
                if isinstance(results, tuple):
                    media_items = results[0]
                
                if not media_items:
                    break
                
                # Filter items that need indexing
                for item in media_items:
                    item_id = str(item['id'])
                    last_modified = datetime.fromisoformat(item.get('last_modified', item.get('ingestion_date', '')))
                    
                    # Check if item needs indexing
                    if incremental:
                        if item_id in indexed_items:
                            # Check if modified since last index
                            if last_modified <= indexed_items[item_id]:
                                continue  # Skip, already indexed and not modified
                            else:
                                # Remove old chunks before re-indexing
                                await self._remove_item_chunks("media", item_id)
                    
                    items_to_process.append(item)
                
                page += 1
            
            total_count = len(items_to_process)
            logger.info(f"Found {total_count} media items to index")
            
            if progress_callback:
                progress_callback("media", 0, total_count)
            
            # Process items in batches
            for i in range(0, len(items_to_process), batch_size):
                batch = items_to_process[i:i + batch_size]
                
                # Process batch
                chunk_counts = await self._index_media_batch(
                    batch, chunk_size, chunk_overlap
                )
                
                # Track indexed items
                for item, chunk_count in zip(batch, chunk_counts):
                    self.indexing_db.mark_item_indexed(
                        item_id=str(item['id']),
                        item_type="media",
                        last_modified=datetime.fromisoformat(item.get('last_modified', item.get('ingestion_date', ''))),
                        chunk_count=chunk_count
                    )
                
                indexed_count += len(batch)
                
                # Update progress
                if progress_callback:
                    progress_callback("media", indexed_count, total_count)
                
        except Exception as e:
            logger.error(f"Error indexing media items: {e}")
            
        logger.info(f"Indexed {indexed_count} media items")
        return indexed_count
    
    async def _remove_item_chunks(self, item_type: str, item_id: str):
        """Remove all chunks for a specific item from the vector store."""
        try:
            collection_name = f"{item_type}_chunks"
            collection = self.embeddings_service.get_or_create_collection(collection_name)
            
            # Query for all chunks belonging to this item
            results = collection.get(
                where={f"{item_type}_id": int(item_id) if item_type == "media" else item_id}
            )
            
            if results and results['ids']:
                # Delete all chunks for this item
                collection.delete(ids=results['ids'])
                logger.debug(f"Removed {len(results['ids'])} chunks for {item_type} {item_id}")
        except Exception as e:
            logger.error(f"Error removing chunks for {item_type} {item_id}: {e}")
    
    async def _index_media_batch(
        self,
        media_items: List[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[int]:
        """Index a batch of media items and return chunk counts."""
        chunk_counts = []
        
        for item in media_items:
            documents = []
            metadatas = []
            ids = []
            
            if not item.get('content'):
                chunk_counts.append(0)
                continue
                
            # Chunk the content
            chunks = self.chunking_service.chunk_document(
                item, chunk_size, chunk_overlap
            )
            
            for chunk in chunks:
                # Prepare document
                doc_text = chunk['text']
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    'media_id': item['id'],
                    'title': item.get('title', 'Untitled'),
                    'type': item.get('type', 'unknown'),
                    'author': item.get('author', 'Unknown'),
                    'chunk_index': chunk['chunk_index'],
                    'chunk_id': chunk['chunk_id']
                }
                metadatas.append(metadata)
                
                # Create unique ID
                chunk_uuid = f"media_{item['id']}_{chunk['chunk_id']}"
                ids.append(chunk_uuid)
            
            if documents:
                # Create embeddings
                chunk_embeddings = self.embeddings_service.create_embeddings(documents)
                if chunk_embeddings:
                    # Add to collection
                    self.embeddings_service.add_documents_to_collection(
                        collection_name="media_chunks",
                        documents=documents,
                        embeddings=chunk_embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
            
            chunk_counts.append(len(chunks))
        
        return chunk_counts
    
    async def index_conversations(
        self,
        chachanotes_db: CharactersRAGDB,
        batch_size: int = 10,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = True
    ) -> int:
        """
        Index conversations into ChromaDB
        
        Args:
            chachanotes_db: ChaChaNotes database instance
            batch_size: Number of conversations to process at once
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            progress_callback: Optional callback for progress updates (type, current, total)
            incremental: If True, only index new/modified conversations
            
        Returns:
            Number of conversations indexed
        """
        logger.info(f"Starting conversations indexing (incremental={incremental})...")
        indexed_count = 0
        
        try:
            # Get already indexed items if incremental
            indexed_items = {}
            if incremental:
                indexed_items = self.indexing_db.get_indexed_items_by_type("conversation")
                logger.info(f"Found {len(indexed_items)} already indexed conversations")
            
            # Get all conversations
            conversations = await asyncio.to_thread(
                chachanotes_db.list_conversations
            )
            
            # Filter conversations that need indexing
            conversations_to_process = []
            for conv in conversations:
                conv_id = conv['id']
                last_modified = datetime.fromisoformat(conv.get('last_modified', conv.get('created_at', '')))
                
                # Check if conversation needs indexing
                if incremental:
                    if conv_id in indexed_items:
                        # Check if modified since last index
                        if last_modified <= indexed_items[conv_id]:
                            continue  # Skip, already indexed and not modified
                        else:
                            # Remove old chunks before re-indexing
                            await self._remove_item_chunks("conversation", conv_id)
                
                conversations_to_process.append(conv)
            
            total_count = len(conversations_to_process)
            logger.info(f"Found {total_count} conversations to index")
            
            if progress_callback:
                progress_callback("conversations", 0, total_count)
            
            # Process in batches
            for i in range(0, len(conversations_to_process), batch_size):
                batch = conversations_to_process[i:i + batch_size]
                chunk_counts = await self._index_conversation_batch(
                    chachanotes_db, batch, chunk_size, chunk_overlap
                )
                
                # Track indexed conversations
                for conv, chunk_count in zip(batch, chunk_counts):
                    self.indexing_db.mark_item_indexed(
                        item_id=conv['id'],
                        item_type="conversation",
                        last_modified=datetime.fromisoformat(conv.get('last_modified', conv.get('created_at', ''))),
                        chunk_count=chunk_count
                    )
                
                indexed_count += len(batch)
                
                # Update progress
                if progress_callback:
                    progress_callback("conversations", indexed_count, total_count)
                
        except Exception as e:
            logger.error(f"Error indexing conversations: {e}")
            
        logger.info(f"Indexed {indexed_count} conversations")
        return indexed_count
    
    async def _index_conversation_batch(
        self,
        chachanotes_db: CharactersRAGDB,
        conversations: List[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[int]:
        """Index a batch of conversations and return chunk counts."""
        chunk_counts = []
        
        for conv in conversations:
            documents = []
            metadatas = []
            ids = []
            
            # Get messages for conversation
            messages = await asyncio.to_thread(
                chachanotes_db.get_messages_for_conversation,
                conversation_id=conv['id']
            )
            
            if not messages:
                chunk_counts.append(0)
                continue
            
            # Combine messages into conversation text
            conv_text = ""
            for msg in messages:
                conv_text += f"{msg['sender']}: {msg['content']}\n\n"
            
            # Create document for chunking
            doc = {
                'id': conv['id'],
                'title': conv.get('title', 'Untitled Conversation'),
                'content': conv_text,
                'type': 'conversation'
            }
            
            # Chunk the conversation
            chunks = self.chunking_service.chunk_document(
                doc, chunk_size, chunk_overlap
            )
            
            for chunk in chunks:
                # Prepare document
                documents.append(chunk['text'])
                
                # Create metadata
                metadata = {
                    'conversation_id': conv['id'],
                    'title': conv.get('title', 'Untitled Conversation'),
                    'character_id': conv.get('character_id'),
                    'chunk_index': chunk['chunk_index'],
                    'chunk_id': chunk['chunk_id']
                }
                metadatas.append(metadata)
                
                # Create unique ID
                chunk_uuid = f"conv_{conv['id']}_{chunk['chunk_id']}"
                ids.append(chunk_uuid)
            
            if documents:
                # Create embeddings
                chunk_embeddings = self.embeddings_service.create_embeddings(documents)
                if chunk_embeddings:
                    # Add to collection
                    self.embeddings_service.add_documents_to_collection(
                        collection_name="conversation_chunks",
                        documents=documents,
                        embeddings=chunk_embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
            
            chunk_counts.append(len(chunks))
        
        return chunk_counts
    
    async def index_notes(
        self,
        chachanotes_db: CharactersRAGDB,
        batch_size: int = 20,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = True
    ) -> int:
        """
        Index notes into ChromaDB
        
        Args:
            chachanotes_db: ChaChaNotes database instance
            batch_size: Number of notes to process at once
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            progress_callback: Optional callback for progress updates (type, current, total)
            incremental: If True, only index new/modified notes
            
        Returns:
            Number of notes indexed
        """
        logger.info(f"Starting notes indexing (incremental={incremental})...")
        indexed_count = 0
        
        try:
            # Get already indexed items if incremental
            indexed_items = {}
            if incremental:
                indexed_items = self.indexing_db.get_indexed_items_by_type("note")
                logger.info(f"Found {len(indexed_items)} already indexed notes")
            
            # Get all notes
            notes = await asyncio.to_thread(
                chachanotes_db.list_notes
            )
            
            # Filter notes that need indexing
            notes_to_process = []
            for note in notes:
                note_id = note['id']
                last_modified = datetime.fromisoformat(note.get('last_modified', note.get('created_at', '')))
                
                # Check if note needs indexing
                if incremental:
                    if note_id in indexed_items:
                        # Check if modified since last index
                        if last_modified <= indexed_items[note_id]:
                            continue  # Skip, already indexed and not modified
                        else:
                            # Remove old chunks before re-indexing
                            await self._remove_item_chunks("note", note_id)
                
                notes_to_process.append(note)
            
            total_count = len(notes_to_process)
            logger.info(f"Found {total_count} notes to index")
            
            if progress_callback:
                progress_callback("notes", 0, total_count)
            
            # Process in batches
            for i in range(0, len(notes_to_process), batch_size):
                batch = notes_to_process[i:i + batch_size]
                chunk_counts = await self._index_notes_batch(
                    batch, chunk_size, chunk_overlap
                )
                
                # Track indexed notes
                for note, chunk_count in zip(batch, chunk_counts):
                    self.indexing_db.mark_item_indexed(
                        item_id=note['id'],
                        item_type="note",
                        last_modified=datetime.fromisoformat(note.get('last_modified', note.get('created_at', ''))),
                        chunk_count=chunk_count
                    )
                
                indexed_count += len(batch)
                
                # Update progress
                if progress_callback:
                    progress_callback("notes", indexed_count, total_count)
                
        except Exception as e:
            logger.error(f"Error indexing notes: {e}")
            
        logger.info(f"Indexed {indexed_count} notes")
        return indexed_count
    
    async def _index_notes_batch(
        self,
        notes: List[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[int]:
        """Index a batch of notes and return chunk counts."""
        chunk_counts = []
        
        for note in notes:
            documents = []
            metadatas = []
            ids = []
            
            if not note.get('content'):
                chunk_counts.append(0)
                continue
            
            # Create document for chunking
            doc = {
                'id': note['id'],
                'title': note.get('title', 'Untitled Note'),
                'content': note['content'],
                'type': 'note'
            }
            
            # Chunk the note
            chunks = self.chunking_service.chunk_document(
                doc, chunk_size, chunk_overlap
            )
            
            for chunk in chunks:
                # Prepare document
                documents.append(chunk['text'])
                
                # Create metadata
                metadata = {
                    'note_id': note['id'],
                    'title': note.get('title', 'Untitled Note'),
                    'tags': note.get('tags', []),
                    'chunk_index': chunk['chunk_index'],
                    'chunk_id': chunk['chunk_id']
                }
                metadatas.append(metadata)
                
                # Create unique ID
                chunk_uuid = f"note_{note['id']}_{chunk['chunk_id']}"
                ids.append(chunk_uuid)
            
            if documents:
                # Create embeddings
                chunk_embeddings = self.embeddings_service.create_embeddings(documents)
                if chunk_embeddings:
                    # Add to collection
                    self.embeddings_service.add_documents_to_collection(
                        collection_name="notes_chunks",
                        documents=documents,
                        embeddings=chunk_embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
            
            chunk_counts.append(len(chunks))
        
        return chunk_counts
    
    async def index_all(
        self,
        media_db: Optional[MediaDatabase] = None,
        chachanotes_db: Optional[CharactersRAGDB] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = True
    ) -> Dict[str, int]:
        """
        Index all available content
        
        Args:
            media_db: Media database instance
            chachanotes_db: ChaChaNotes database instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            progress_callback: Optional callback for progress updates (type, current, total)
            incremental: If True, only index new/modified items
            
        Returns:
            Dict with counts of indexed items by type
        """
        results = {
            'media': 0,
            'conversations': 0,
            'notes': 0
        }
        
        if media_db:
            results['media'] = await self.index_media_items(
                media_db, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                progress_callback=progress_callback,
                incremental=incremental
            )
        
        if chachanotes_db:
            results['conversations'] = await self.index_conversations(
                chachanotes_db, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                progress_callback=progress_callback,
                incremental=incremental
            )
            results['notes'] = await self.index_notes(
                chachanotes_db, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                progress_callback=progress_callback,
                incremental=incremental
            )
        
        # Update collection state
        if media_db and results['media'] > 0:
            self.indexing_db.update_collection_state(
                "media_chunks", 
                total_items=results['media'],
                indexed_items=results['media']
            )
        
        if chachanotes_db:
            if results['conversations'] > 0:
                self.indexing_db.update_collection_state(
                    "conversation_chunks",
                    total_items=results['conversations'],
                    indexed_items=results['conversations']
                )
            if results['notes'] > 0:
                self.indexing_db.update_collection_state(
                    "notes_chunks",
                    total_items=results['notes'],
                    indexed_items=results['notes']
                )
        
        logger.info(f"Indexing complete: {results}")
        return results
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get statistics about the current indexing state."""
        return self.indexing_db.get_indexing_stats()
    
    async def clear_all_indexes(self):
        """Clear all indexed data and tracking information."""
        logger.warning("Clearing all RAG indexes...")
        
        # Clear vector collections
        for collection_name in ["media_chunks", "conversation_chunks", "notes_chunks"]:
            try:
                collection = self.embeddings_service.get_or_create_collection(collection_name)
                # Delete all documents from collection
                all_docs = collection.get()
                if all_docs and all_docs['ids']:
                    collection.delete(ids=all_docs['ids'])
                    logger.info(f"Cleared {len(all_docs['ids'])} documents from {collection_name}")
            except Exception as e:
                logger.error(f"Error clearing collection {collection_name}: {e}")
        
        # Clear indexing tracking
        self.indexing_db.clear_all()
        logger.info("All RAG indexes cleared")
    
    def configure_performance(
        self,
        max_workers: Optional[int] = None,
        enable_parallel: Optional[bool] = None
    ):
        """
        Configure performance settings.
        
        Args:
            max_workers: Number of worker threads for parallel processing
            enable_parallel: Enable/disable parallel processing
        """
        if max_workers is not None:
            self.max_workers = max_workers
            # Close existing executor to recreate with new worker count
            self._close_executor()
            
        if enable_parallel is not None:
            self.enable_parallel_indexing = enable_parallel
            
        # Also configure the embeddings service
        if hasattr(self.embeddings_service, 'configure_performance'):
            self.embeddings_service.configure_performance(
                max_workers=max_workers,
                enable_parallel=enable_parallel
            )
            
        logger.info(f"Indexing performance configured: workers={self.max_workers}, parallel={self.enable_parallel_indexing}")
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="indexing"
                )
            return self._executor
    
    def _close_executor(self):
        """Close the thread pool executor with timeout."""
        with self._executor_lock:
            if self._executor:
                try:
                    # Try to shutdown gracefully with timeout
                    self._executor.shutdown(wait=True, timeout=5.0)
                except Exception as e:
                    logger.warning(f"Error during executor shutdown: {e}")
                    # Force shutdown if graceful shutdown fails
                    try:
                        self._executor.shutdown(wait=False)
                    except:
                        pass
                finally:
                    self._executor = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self._close_executor()
        return False
    
    def __del__(self):
        """Cleanup on destruction."""
        self._close_executor()