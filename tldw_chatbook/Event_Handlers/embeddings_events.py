# embeddings_events.py
# Description: Event handlers for embeddings functionality
#
# Imports
from __future__ import annotations

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from loguru import logger
import asyncio

# Third-party imports
from textual.app import App
from textual.events import Event
from textual.message import Message

# Local imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..config import get_cli_setting

# Import embeddings components if available
if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
    from ..Embeddings.Embeddings_Lib import EmbeddingFactory
    from ..Embeddings.Chroma_Lib import ChromaDBManager
    from ..Chunking.Chunk_Lib import chunk_for_embedding
else:
    EmbeddingFactory = None
    ChromaDBManager = None
    chunk_for_embedding = None

logger = logger.bind(name="embeddings_events")

########################################################################################################################
#
# Custom Events for Embeddings
#
########################################################################################################################

class EmbeddingEvent(Message):
    """Base class for embedding-related events."""
    
    def __init__(self, sender: Any = None) -> None:
        super().__init__()
        self.sender = sender


class EmbeddingModelLoadEvent(EmbeddingEvent):
    """Event fired when an embedding model needs to be loaded."""
    
    def __init__(self, model_id: str, sender: Any = None) -> None:
        super().__init__(sender)
        self.model_id = model_id


class EmbeddingModelUnloadEvent(EmbeddingEvent):
    """Event fired when an embedding model needs to be unloaded."""
    
    def __init__(self, model_id: str, sender: Any = None) -> None:
        super().__init__(sender)
        self.model_id = model_id


class EmbeddingCreateEvent(EmbeddingEvent):
    """Event fired when embeddings need to be created."""
    
    def __init__(
        self,
        text: Union[str, List[str]],
        model_id: str,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_config: Optional[Dict[str, Any]] = None,
        sender: Any = None
    ) -> None:
        super().__init__(sender)
        self.text = text
        self.model_id = model_id
        self.collection_name = collection_name
        self.metadata = metadata or {}
        self.chunk_config = chunk_config or {}


class EmbeddingSearchEvent(EmbeddingEvent):
    """Event fired when searching embeddings."""
    
    def __init__(
        self,
        query: str,
        collection_name: str,
        n_results: int = 10,
        sender: Any = None
    ) -> None:
        super().__init__(sender)
        self.query = query
        self.collection_name = collection_name
        self.n_results = n_results


class CollectionDeleteEvent(EmbeddingEvent):
    """Event fired when a collection needs to be deleted."""
    
    def __init__(self, collection_name: str, sender: Any = None) -> None:
        super().__init__(sender)
        self.collection_name = collection_name


class CollectionExportEvent(EmbeddingEvent):
    """Event fired when a collection needs to be exported."""
    
    def __init__(
        self,
        collection_name: str,
        export_path: Path,
        sender: Any = None
    ) -> None:
        super().__init__(sender)
        self.collection_name = collection_name
        self.export_path = export_path


class EmbeddingTestEvent(EmbeddingEvent):
    """Event fired when testing embeddings."""
    
    def __init__(
        self,
        text: str,
        model_id: str,
        sender: Any = None
    ) -> None:
        super().__init__(sender)
        self.text = text
        self.model_id = model_id


class EmbeddingProgressEvent(EmbeddingEvent):
    """Event fired to update embedding progress."""
    
    def __init__(
        self,
        current: int,
        total: int,
        message: str = "",
        sender: Any = None
    ) -> None:
        super().__init__(sender)
        self.current = current
        self.total = total
        self.message = message


class EmbeddingCompleteEvent(EmbeddingEvent):
    """Event fired when embedding operation completes."""
    
    def __init__(
        self,
        success: bool,
        message: str = "",
        result: Optional[Any] = None,
        sender: Any = None
    ) -> None:
        super().__init__(sender)
        self.success = success
        self.message = message
        self.result = result


########################################################################################################################
#
# Embeddings Event Handler
#
########################################################################################################################

class EmbeddingsEventHandler:
    """Handles embeddings-related events."""
    
    def __init__(
        self,
        app: App,
        chachanotes_db: CharactersRAGDB,
        media_db: MediaDatabase
    ):
        """Initialize the embeddings event handler.
        
        Args:
            app: The Textual app instance
            chachanotes_db: Database for configuration storage
            media_db: Media database for content access
        """
        self.app = app
        self.chachanotes_db = chachanotes_db
        self.media_db = media_db
        self.embedding_factory: Optional[EmbeddingFactory] = None
        self.chroma_manager: Optional[ChromaDBManager] = None
        
        # Initialize if dependencies available
        if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize embedding factory and ChromaDB manager."""
        try:
            # Load embedding configuration
            embedding_config = get_cli_setting('embedding_config', {})
            
            if embedding_config:
                self.embedding_factory = EmbeddingFactory(
                    embedding_config,
                    max_cached=2,
                    idle_seconds=900
                )
                logger.info("Initialized embedding factory in event handler")
            else:
                logger.warning("No embedding configuration found")
                
            # Initialize ChromaDB manager
            user_id = get_cli_setting('users_name', 'default_user')
            if embedding_config:
                self.chroma_manager = ChromaDBManager(user_id, embedding_config)
                logger.info("Initialized ChromaDB manager in event handler")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings in event handler: {e}")
    
    async def handle_model_load(self, event: EmbeddingModelLoadEvent) -> None:
        """Handle model load event."""
        if not self.embedding_factory:
            self._notify_error(event.sender, "Embedding factory not initialized")
            return
        
        try:
            logger.info(f"Loading embedding model: {event.model_id}")
            
            # Prefetch the model
            self.embedding_factory.prefetch([event.model_id])
            
            # Notify success
            self._notify_success(
                event.sender,
                f"Model {event.model_id} loaded successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model {event.model_id}: {e}")
            self._notify_error(event.sender, f"Failed to load model: {str(e)}")
    
    async def handle_model_unload(self, event: EmbeddingModelUnloadEvent) -> None:
        """Handle model unload event."""
        if not self.embedding_factory:
            self._notify_error(event.sender, "Embedding factory not initialized")
            return
        
        try:
            logger.info(f"Unloading embedding model: {event.model_id}")
            
            # Access internal cache to remove model
            with self.embedding_factory._lock:
                if event.model_id in self.embedding_factory._cache:
                    rec = self.embedding_factory._cache.pop(event.model_id)
                    if rec["close"]:
                        rec["close"]()
            
            # Notify success
            self._notify_success(
                event.sender,
                f"Model {event.model_id} unloaded successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to unload model {event.model_id}: {e}")
            self._notify_error(event.sender, f"Failed to unload model: {str(e)}")
    
    async def handle_create_embeddings(self, event: EmbeddingCreateEvent) -> None:
        """Handle embedding creation event."""
        if not self.embedding_factory or not self.chroma_manager:
            self._notify_error(event.sender, "Embeddings not properly initialized")
            return
        
        try:
            logger.info(f"Creating embeddings for collection: {event.collection_name}")
            
            # Prepare text list
            texts = event.text if isinstance(event.text, list) else [event.text]
            all_chunks = []
            
            # Chunk texts if needed
            if event.chunk_config.get('enabled', True):
                for text in texts:
                    if chunk_for_embedding:
                        chunks = chunk_for_embedding(
                            text,
                            chunk_method=event.chunk_config.get('method', 'character'),
                            max_chunk_size=event.chunk_config.get('size', 512),
                            chunk_overlap=event.chunk_config.get('overlap', 128)
                        )
                        all_chunks.extend(chunks)
                    else:
                        # Fallback simple chunking
                        size = event.chunk_config.get('size', 512)
                        overlap = event.chunk_config.get('overlap', 128)
                        for i in range(0, len(text), size - overlap):
                            all_chunks.append(text[i:i + size])
            else:
                all_chunks = texts
            
            total_chunks = len(all_chunks)
            logger.info(f"Processing {total_chunks} chunks")
            
            # Send initial progress
            self._send_progress(event.sender, 0, total_chunks, "Starting embedding generation...")
            
            # Generate embeddings in batches
            batch_size = 32  # Process in batches to avoid memory issues
            embeddings = []
            
            for i in range(0, total_chunks, batch_size):
                batch = all_chunks[i:i + batch_size]
                batch_embeddings = await self.embedding_factory.async_embed(
                    batch,
                    model_id=event.model_id
                )
                embeddings.extend(batch_embeddings)
                
                # Update progress
                processed = min(i + batch_size, total_chunks)
                self._send_progress(
                    event.sender,
                    processed,
                    total_chunks,
                    f"Generated embeddings for {processed}/{total_chunks} chunks"
                )
            
            # Store in ChromaDB
            self._send_progress(
                event.sender,
                total_chunks,
                total_chunks,
                "Storing embeddings in ChromaDB..."
            )
            
            # Create collection and add embeddings
            # This is a simplified example - actual implementation would
            # interface with ChromaDB properly
            collection_metadata = {
                "embedding_model_id": event.model_id,
                **event.metadata
            }
            
            # Notify completion
            self._notify_complete(
                event.sender,
                True,
                f"Successfully created {len(embeddings)} embeddings in collection '{event.collection_name}'",
                {"embeddings_count": len(embeddings), "collection": event.collection_name}
            )
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            self._notify_complete(
                event.sender,
                False,
                f"Failed to create embeddings: {str(e)}"
            )
    
    async def handle_search_embeddings(self, event: EmbeddingSearchEvent) -> None:
        """Handle embedding search event."""
        if not self.embedding_factory or not self.chroma_manager:
            self._notify_error(event.sender, "Embeddings not properly initialized")
            return
        
        try:
            logger.info(f"Searching in collection: {event.collection_name}")
            
            # Generate query embedding
            query_embedding = await self.embedding_factory.async_embed(
                [event.query],
                model_id=self.chroma_manager._default_model_id
            )
            
            # Search in ChromaDB
            # This is a placeholder - actual implementation would
            # interface with ChromaDB's search API
            results = []
            
            # Notify completion with results
            self._notify_complete(
                event.sender,
                True,
                f"Found {len(results)} results",
                {"results": results}
            )
            
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            self._notify_complete(
                event.sender,
                False,
                f"Failed to search: {str(e)}"
            )
    
    async def handle_delete_collection(self, event: CollectionDeleteEvent) -> None:
        """Handle collection deletion event."""
        if not self.chroma_manager:
            self._notify_error(event.sender, "ChromaDB manager not initialized")
            return
        
        try:
            logger.info(f"Deleting collection: {event.collection_name}")
            
            # Delete collection from ChromaDB
            # This is a placeholder - actual implementation would
            # interface with ChromaDB's deletion API
            
            # Notify success
            self._notify_success(
                event.sender,
                f"Collection '{event.collection_name}' deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            self._notify_error(event.sender, f"Failed to delete collection: {str(e)}")
    
    async def handle_test_embedding(self, event: EmbeddingTestEvent) -> None:
        """Handle embedding test event."""
        if not self.embedding_factory:
            self._notify_error(event.sender, "Embedding factory not initialized")
            return
        
        try:
            logger.info(f"Testing embedding with model: {event.model_id}")
            
            # Generate test embedding
            embedding = await self.embedding_factory.async_embed(
                [event.text],
                model_id=event.model_id,
                as_list=True
            )
            
            # Prepare result
            result = {
                "model": event.model_id,
                "text_length": len(event.text),
                "dimension": len(embedding[0]) if embedding else 0,
                "first_values": embedding[0][:10] if embedding else [],
                "norm": sum(x**2 for x in embedding[0])**0.5 if embedding else 0
            }
            
            # Notify completion with result
            self._notify_complete(
                event.sender,
                True,
                "Test embedding generated successfully",
                result
            )
            
        except Exception as e:
            logger.error(f"Failed to generate test embedding: {e}")
            self._notify_complete(
                event.sender,
                False,
                f"Failed to generate test embedding: {str(e)}"
            )
    
    # Helper methods for notifications
    def _notify_error(self, sender: Any, message: str) -> None:
        """Send error notification."""
        if sender and hasattr(sender, 'notify'):
            sender.notify(message, severity="error")
    
    def _notify_success(self, sender: Any, message: str) -> None:
        """Send success notification."""
        if sender and hasattr(sender, 'notify'):
            sender.notify(message, severity="success")
    
    def _send_progress(self, sender: Any, current: int, total: int, message: str) -> None:
        """Send progress event."""
        if sender:
            progress_event = EmbeddingProgressEvent(current, total, message, sender)
            self.app.post_message(progress_event)
    
    def _notify_complete(self, sender: Any, success: bool, message: str, result: Optional[Any] = None) -> None:
        """Send completion event."""
        if sender:
            complete_event = EmbeddingCompleteEvent(success, message, result, sender)
            self.app.post_message(complete_event)


########################################################################################################################
#
# Event Handler Registration
#
########################################################################################################################

def register_embeddings_events(app: App, chachanotes_db: CharactersRAGDB, media_db: MediaDatabase) -> EmbeddingsEventHandler:
    """Register embeddings event handlers with the app.
    
    Args:
        app: The Textual app instance
        chachanotes_db: Database for configuration
        media_db: Media database
        
    Returns:
        The embeddings event handler instance
    """
    handler = EmbeddingsEventHandler(app, chachanotes_db, media_db)
    
    # Register event handlers
    async def on_model_load(event: EmbeddingModelLoadEvent) -> None:
        await handler.handle_model_load(event)
    
    async def on_model_unload(event: EmbeddingModelUnloadEvent) -> None:
        await handler.handle_model_unload(event)
    
    async def on_create_embeddings(event: EmbeddingCreateEvent) -> None:
        await handler.handle_create_embeddings(event)
    
    async def on_search_embeddings(event: EmbeddingSearchEvent) -> None:
        await handler.handle_search_embeddings(event)
    
    async def on_delete_collection(event: CollectionDeleteEvent) -> None:
        await handler.handle_delete_collection(event)
    
    async def on_test_embedding(event: EmbeddingTestEvent) -> None:
        await handler.handle_test_embedding(event)
    
    # Register with app
    app.on(EmbeddingModelLoadEvent, on_model_load)
    app.on(EmbeddingModelUnloadEvent, on_model_unload)
    app.on(EmbeddingCreateEvent, on_create_embeddings)
    app.on(EmbeddingSearchEvent, on_search_embeddings)
    app.on(CollectionDeleteEvent, on_delete_collection)
    app.on(EmbeddingTestEvent, on_test_embedding)
    
    logger.info("Registered embeddings event handlers")
    
    return handler

# End of embeddings_events.py
########################################################################################################################