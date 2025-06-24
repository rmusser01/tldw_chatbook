"""
Retrieval strategies for the RAG service.

This module provides various retrieval strategies optimized for a single-user
TUI application with local databases.
"""

import asyncio
from abc import abstractmethod
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import sqlite3
from contextlib import contextmanager

from loguru import logger
import chromadb
from chromadb.config import Settings
import numpy as np

from .types import (
    RetrieverStrategy, DataSource, Document, SearchResult,
    RetrievalError
)
from .utils import chunk_text, create_document_id, TokenCounter, extract_keywords


class BaseRetriever(RetrieverStrategy):
    """
    Base retriever with common functionality for single-user app.
    
    Since this is a single-user app, we can:
    - Keep database connections open longer
    - Use simpler caching strategies
    - Optimize for local file access
    """
    
    def __init__(self, source: DataSource, config: Dict[str, Any] = None):
        self.source = source
        self.config = config or {}
        self._token_counter = TokenCounter()
        
    @property
    def source_type(self) -> DataSource:
        return self.source
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """Base retrieve method - must be implemented by subclasses."""
        raise NotImplementedError
    
    def _create_document(
        self,
        content: str,
        doc_id: str,
        metadata: Dict[str, Any],
        score: float = 0.0
    ) -> Document:
        """Helper to create a Document instance."""
        return Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            source=self.source,
            score=score
        )


class MediaDBRetriever(BaseRetriever):
    """
    Retriever for Media Database content.
    
    Optimized for single-user access to local SQLite database.
    """
    
    def __init__(self, db_path: Path, config: Dict[str, Any] = None):
        super().__init__(DataSource.MEDIA_DB, config)
        self.db_path = db_path
        
        # For single-user app, we can keep a connection pool
        self._db_connection = None
        self._ensure_fts_table()
    
    @contextmanager
    def _get_db(self):
        """Get database connection (reuses for single user)."""
        if self._db_connection is None:
            self._db_connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Safe for single user
            )
            self._db_connection.row_factory = sqlite3.Row
        
        yield self._db_connection
    
    def _ensure_fts_table(self):
        """Ensure FTS5 table exists for full-text search."""
        with self._get_db() as conn:
            # Check if FTS table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='media_fts'"
            )
            if not cursor.fetchone():
                logger.info("Creating FTS5 table for media search")
                conn.execute("""
                    CREATE VIRTUAL TABLE media_fts USING fts5(
                        media_id,
                        title,
                        content,
                        content=media,
                        content_rowid=id,
                        tokenize='porter unicode61'
                    )
                """)
                conn.commit()
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """Retrieve documents from media database using FTS5."""
        try:
            documents = []
            
            with self._get_db() as conn:
                # Build the query with filters
                where_clauses = ["media_fts MATCH ?"]
                params = [query]
                
                if filters:
                    if "media_type" in filters:
                        where_clauses.append("media_type = ?")
                        params.append(filters["media_type"])
                    
                    if "date_range" in filters:
                        # Handle date range filter
                        pass
                
                sql = f"""
                    SELECT 
                        m.id,
                        m.title,
                        m.content,
                        m.media_type,
                        m.url,
                        m.created_at,
                        rank
                    FROM media_fts
                    JOIN media_files m ON media_fts.media_id = m.id
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY rank
                    LIMIT ?
                """
                params.append(top_k)
                
                cursor = conn.execute(sql, params)
                
                for row in cursor:
                    doc = self._create_document(
                        content=row["content"] or "",
                        doc_id=f"media_{row['id']}",
                        metadata={
                            "media_id": row["id"],
                            "title": row["title"],
                            "media_type": row["media_type"],
                            "url": row["url"],
                            "created_at": row["created_at"]
                        },
                        score=-row["rank"]  # FTS5 rank is negative
                    )
                    documents.append(doc)
            
            logger.debug(f"MediaDB FTS retrieved {len(documents)} documents")
            
            return SearchResult(
                documents=documents,
                query=query,
                search_type="fts",
                metadata={"database": "media_db"}
            )
            
        except Exception as e:
            logger.error(f"Error in MediaDB retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve from media database: {e}")
    
    def close(self):
        """Close database connection."""
        if self._db_connection:
            self._db_connection.close()
            self._db_connection = None


class ChatHistoryRetriever(BaseRetriever):
    """
    Retriever for chat conversation history.
    
    Optimized for single-user chat history stored in ChaChaNotes database.
    """
    
    def __init__(self, db_path: Path, config: Dict[str, Any] = None):
        super().__init__(DataSource.CHAT_HISTORY, config)
        self.db_path = db_path
        self._db_connection = None
    
    @contextmanager
    def _get_db(self):
        """Get database connection."""
        if self._db_connection is None:
            self._db_connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self._db_connection.row_factory = sqlite3.Row
        
        yield self._db_connection
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """Retrieve relevant chat messages."""
        try:
            documents = []
            
            with self._get_db() as conn:
                # For chat history, we want to search in messages
                where_clauses = []
                params = []
                
                # Add filters
                if filters:
                    if "character_id" in filters:
                        where_clauses.append("c.character_id = ?")
                        params.append(filters["character_id"])
                    
                    if "conversation_ids" in filters:
                        placeholders = ",".join("?" * len(filters["conversation_ids"]))
                        where_clauses.append(f"c.id IN ({placeholders})")
                        params.extend(filters["conversation_ids"])
                
                where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                
                # Simple keyword search in messages
                sql = f"""
                    SELECT 
                        m.id,
                        m.conversation_id,
                        m.sender,
                        m.content,
                        m.timestamp,
                        c.title as conversation_title,
                        c.character_id
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    {where_clause}
                    {"AND" if where_clause else "WHERE"} m.content LIKE ?
                    ORDER BY m.timestamp DESC
                    LIMIT ?
                """
                params.append(f"%{query}%")
                params.append(top_k)
                
                cursor = conn.execute(sql, params)
                
                for row in cursor:
                    # Create a document for each message
                    doc = self._create_document(
                        content=row["content"],
                        doc_id=f"chat_msg_{row['id']}",
                        metadata={
                            "message_id": row["id"],
                            "conversation_id": row["conversation_id"],
                            "conversation_title": row["conversation_title"],
                            "sender": row["sender"],
                            "timestamp": row["timestamp"],
                            "character_id": row["character_id"]
                        },
                        score=1.0  # Simple scoring for now
                    )
                    documents.append(doc)
            
            logger.debug(f"ChatHistory retrieved {len(documents)} messages")
            
            return SearchResult(
                documents=documents,
                query=query,
                search_type="keyword",
                metadata={"database": "chat_history"}
            )
            
        except Exception as e:
            logger.error(f"Error in ChatHistory retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve from chat history: {e}")


class NotesRetriever(BaseRetriever):
    """
    Retriever for user notes.
    
    Optimized for single-user notes in ChaChaNotes database.
    """
    
    def __init__(self, db_path: Path, config: Dict[str, Any] = None):
        super().__init__(DataSource.NOTES, config)
        self.db_path = db_path
        self._db_connection = None
    
    @contextmanager
    def _get_db(self):
        """Get database connection."""
        if self._db_connection is None:
            self._db_connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self._db_connection.row_factory = sqlite3.Row
        
        yield self._db_connection
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """Retrieve relevant notes."""
        try:
            documents = []
            
            with self._get_db() as conn:
                # Build query with filters
                where_clauses = ["(n.title LIKE ? OR n.content LIKE ?)"]
                params = [f"%{query}%", f"%{query}%"]
                
                if filters:
                    if "note_ids" in filters:
                        placeholders = ",".join("?" * len(filters["note_ids"]))
                        where_clauses.append(f"n.id IN ({placeholders})")
                        params.extend(filters["note_ids"])
                    
                    if "keywords" in filters:
                        # Add keyword filtering
                        keyword_conditions = []
                        for keyword in filters["keywords"]:
                            keyword_conditions.append("nk.keyword LIKE ?")
                            params.append(f"%{keyword}%")
                        if keyword_conditions:
                            where_clauses.append(f"({' OR '.join(keyword_conditions)})")
                
                sql = f"""
                    SELECT DISTINCT
                        n.id,
                        n.title,
                        n.content,
                        n.created_at,
                        n.updated_at,
                        GROUP_CONCAT(nk.keyword) as keywords
                    FROM notes n
                    LEFT JOIN note_keywords nk ON n.id = nk.note_id
                    WHERE {' AND '.join(where_clauses)}
                    GROUP BY n.id
                    ORDER BY n.updated_at DESC
                    LIMIT ?
                """
                params.append(top_k)
                
                cursor = conn.execute(sql, params)
                
                for row in cursor:
                    # Score based on query occurrence
                    content = row["content"] or ""
                    title = row["title"] or ""
                    score = (
                        content.lower().count(query.lower()) * 0.5 +
                        title.lower().count(query.lower()) * 2.0
                    )
                    
                    doc = self._create_document(
                        content=content,
                        doc_id=f"note_{row['id']}",
                        metadata={
                            "note_id": row["id"],
                            "title": title,
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                            "keywords": row["keywords"].split(",") if row["keywords"] else []
                        },
                        score=score
                    )
                    documents.append(doc)
            
            # Sort by score
            documents.sort(key=lambda d: d.score, reverse=True)
            
            logger.debug(f"Notes retriever found {len(documents)} notes")
            
            return SearchResult(
                documents=documents,
                query=query,
                search_type="keyword",
                metadata={"database": "notes"}
            )
            
        except Exception as e:
            logger.error(f"Error in Notes retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve from notes: {e}")


class VectorRetriever(BaseRetriever):
    """
    Vector-based retriever using ChromaDB.
    
    For single-user app, we use a persistent local ChromaDB instance.
    """
    
    def __init__(
        self,
        source: DataSource,
        chroma_path: Path,
        collection_name: str,
        config: Dict[str, Any] = None
    ):
        super().__init__(source, config)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client (persistent for single user)
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"source": source.name}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """Retrieve documents using vector similarity."""
        try:
            # Build ChromaDB where clause from filters
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where[key] = {"$in": value}
                    else:
                        where[key] = value
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = self._create_document(
                        content=results["documents"][0][i],
                        doc_id=doc_id,
                        metadata=results["metadatas"][0][i],
                        score=1.0 - results["distances"][0][i]  # Convert distance to similarity
                    )
                    documents.append(doc)
            
            logger.debug(f"Vector search retrieved {len(documents)} documents from {self.collection_name}")
            
            return SearchResult(
                documents=documents,
                query=query,
                search_type="vector",
                metadata={"collection": self.collection_name}
            )
            
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve from vector store: {e}")
    
    async def embed_and_store(self, documents: List[Document]) -> None:
        """Embed and store documents in the vector database."""
        if not documents:
            return
        
        try:
            # Prepare data for ChromaDB
            ids = []
            contents = []
            metadatas = []
            
            for doc in documents:
                ids.append(doc.id)
                contents.append(doc.content)
                metadatas.append({
                    **doc.metadata,
                    "source": self.source.name
                })
            
            # Add to collection (ChromaDB handles embedding)
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(documents)} documents in {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise RetrievalError(f"Failed to store documents: {e}")


class HybridRetriever(BaseRetriever):
    """
    Combines vector and keyword search for better results.
    
    This is especially useful for a single-user app where we can
    afford slightly more computation for better quality.
    """
    
    def __init__(
        self,
        keyword_retriever: BaseRetriever,
        vector_retriever: VectorRetriever,
        alpha: float = 0.5,
        config: Dict[str, Any] = None
    ):
        super().__init__(keyword_retriever.source, config)
        self.keyword_retriever = keyword_retriever
        self.vector_retriever = vector_retriever
        self.alpha = alpha  # Weight for vector search (1-alpha for keyword)
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """Perform hybrid search combining keyword and vector results."""
        try:
            # Run both searches in parallel
            keyword_task = self.keyword_retriever.retrieve(query, filters, top_k * 2)
            vector_task = self.vector_retriever.retrieve(query, filters, top_k * 2)
            
            keyword_result, vector_result = await asyncio.gather(
                keyword_task, vector_task
            )
            
            # Combine results with scoring
            doc_scores = {}
            
            # Add keyword results
            for i, doc in enumerate(keyword_result.documents):
                # Normalize rank to score
                score = 1.0 - (i / len(keyword_result.documents))
                doc_scores[doc.id] = {
                    "doc": doc,
                    "keyword_score": score * (1 - self.alpha),
                    "vector_score": 0.0
                }
            
            # Add vector results
            for i, doc in enumerate(vector_result.documents):
                if doc.id in doc_scores:
                    doc_scores[doc.id]["vector_score"] = doc.score * self.alpha
                else:
                    doc_scores[doc.id] = {
                        "doc": doc,
                        "keyword_score": 0.0,
                        "vector_score": doc.score * self.alpha
                    }
            
            # Calculate combined scores and sort
            combined_docs = []
            for doc_id, scores in doc_scores.items():
                doc = scores["doc"]
                doc.score = scores["keyword_score"] + scores["vector_score"]
                combined_docs.append(doc)
            
            combined_docs.sort(key=lambda d: d.score, reverse=True)
            
            # Return top k
            final_docs = combined_docs[:top_k]
            
            return SearchResult(
                documents=final_docs,
                query=query,
                search_type="hybrid",
                metadata={
                    "alpha": self.alpha,
                    "keyword_count": len(keyword_result.documents),
                    "vector_count": len(vector_result.documents)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            raise RetrievalError(f"Failed to perform hybrid search: {e}")