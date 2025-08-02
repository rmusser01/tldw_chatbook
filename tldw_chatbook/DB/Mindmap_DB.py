# Mindmap_DB.py
# Description: Database operations for mindmaps
#
"""
Mindmap Database
---------------

Database module for storing and retrieving mindmaps with:
- Mindmap metadata storage
- Node hierarchy persistence
- Version history tracking
- Collaborative features
"""

import sqlite3
import json
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Any
from datetime import datetime
from contextlib import contextmanager
from anytree import Node
from loguru import logger

from .base_db import BaseDB
from ..Tools.Mind_Map.mermaid_parser import NodeShape


class MindmapDatabase(BaseDB):
    """Database operations for mindmaps"""
    
    def __init__(self, db_path: Union[str, Path], client_id: str = "default", 
                 check_integrity_on_startup: bool = False):
        """Initialize the mindmap database
        
        Args:
            db_path: Path to the SQLite database file or ':memory:'
            client_id: Client identifier for multi-client support
            check_integrity_on_startup: Whether to run integrity check on startup
        """
        super().__init__(db_path, client_id, check_integrity_on_startup)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema"""
        self._create_schema()
        
        if self.check_integrity_on_startup and not self.is_memory_db:
            self.run_integrity_check()
    
    def _create_schema(self) -> None:
        """Create the mindmap database schema"""
        schema_sql = """
        -- Mindmap storage schema
        CREATE TABLE IF NOT EXISTS mindmaps (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            mermaid_source TEXT,  -- Original Mermaid code
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            creator_id TEXT,
            client_id TEXT DEFAULT 'default',
            is_public BOOLEAN DEFAULT FALSE,
            metadata JSON
        );
        
        CREATE TABLE IF NOT EXISTS mindmap_nodes (
            id TEXT PRIMARY KEY,
            mindmap_id TEXT NOT NULL,
            node_id TEXT NOT NULL,  -- ID from Mermaid syntax
            parent_id TEXT,
            text TEXT NOT NULL,
            shape TEXT DEFAULT 'DEFAULT',
            position_index INTEGER DEFAULT 0,  -- Order among siblings
            icon TEXT,
            css_class TEXT,
            metadata JSON,
            FOREIGN KEY (mindmap_id) REFERENCES mindmaps(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_id) REFERENCES mindmap_nodes(id) ON DELETE CASCADE,
            UNIQUE(mindmap_id, node_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_mindmap_nodes_parent ON mindmap_nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_mindmap_nodes_mindmap ON mindmap_nodes(mindmap_id);
        CREATE INDEX IF NOT EXISTS idx_mindmaps_client ON mindmaps(client_id);
        CREATE INDEX IF NOT EXISTS idx_mindmaps_updated ON mindmaps(updated_at);
        
        -- Collaborative features
        CREATE TABLE IF NOT EXISTS mindmap_collaborators (
            mindmap_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            permission TEXT CHECK(permission IN ('view', 'edit', 'admin')),
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (mindmap_id, user_id),
            FOREIGN KEY (mindmap_id) REFERENCES mindmaps(id) ON DELETE CASCADE
        );
        
        -- Version history
        CREATE TABLE IF NOT EXISTS mindmap_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mindmap_id TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            mermaid_source TEXT NOT NULL,
            changed_by TEXT,
            change_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mindmap_id) REFERENCES mindmaps(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_mindmap_versions_mindmap ON mindmap_versions(mindmap_id);
        
        -- Update trigger for mindmaps
        CREATE TRIGGER IF NOT EXISTS update_mindmap_timestamp 
        AFTER UPDATE ON mindmaps
        BEGIN
            UPDATE mindmaps SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        """
        
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def save_mindmap(self, mindmap_id: str, title: str, 
                     mermaid_source: str, root_node: Node,
                     description: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save mindmap to database
        
        Args:
            mindmap_id: Unique identifier for the mindmap
            title: Mindmap title
            mermaid_source: Original Mermaid source code
            root_node: Root node of the mindmap tree
            description: Optional description
            metadata: Optional metadata dictionary
        """
        with self.transaction() as cursor:
            # Save mindmap metadata
            cursor.execute("""
                INSERT OR REPLACE INTO mindmaps 
                (id, title, description, mermaid_source, client_id, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                mindmap_id,
                title,
                description,
                mermaid_source,
                self.client_id,
                json.dumps(metadata) if metadata else None
            ))
            
            # Clear existing nodes
            cursor.execute("DELETE FROM mindmap_nodes WHERE mindmap_id = ?", 
                          (mindmap_id,))
            
            # Save nodes recursively
            self._save_node_recursive(cursor, mindmap_id, root_node, None, 0)
            
            # Save version history
            version_number = self._get_next_version_number(cursor, mindmap_id)
            cursor.execute("""
                INSERT INTO mindmap_versions 
                (mindmap_id, version_number, mermaid_source, changed_by, change_description)
                VALUES (?, ?, ?, ?, ?)
            """, (
                mindmap_id,
                version_number,
                mermaid_source,
                self.client_id,
                "Saved via MindmapDatabase"
            ))
            
            logger.info(f"Saved mindmap {mindmap_id} with {self._count_nodes(root_node)} nodes")
    
    def _save_node_recursive(self, cursor, mindmap_id: str, 
                             node: Node, parent_id: Optional[str], index: int):
        """Recursively save nodes
        
        Args:
            cursor: Database cursor
            mindmap_id: Mindmap ID
            node: Current node to save
            parent_id: Parent node's database ID
            index: Position among siblings
        """
        # Generate unique database ID for this node
        node_db_id = f"{mindmap_id}_{node.name}"
        
        node_data = {
            'id': node_db_id,
            'mindmap_id': mindmap_id,
            'node_id': node.name,
            'parent_id': parent_id,
            'text': getattr(node, 'text', node.name),
            'shape': getattr(node, 'shape', NodeShape.DEFAULT).name,
            'position_index': index,
            'icon': getattr(node, 'icon', None),
            'css_class': getattr(node, 'css_class', None),
            'metadata': json.dumps(getattr(node, 'metadata', {})) if hasattr(node, 'metadata') else None
        }
        
        cursor.execute("""
            INSERT INTO mindmap_nodes 
            (id, mindmap_id, node_id, parent_id, text, shape, 
             position_index, icon, css_class, metadata)
            VALUES (:id, :mindmap_id, :node_id, :parent_id, :text, 
                    :shape, :position_index, :icon, :css_class, :metadata)
        """, node_data)
        
        # Save children
        for i, child in enumerate(node.children):
            self._save_node_recursive(cursor, mindmap_id, child, node_db_id, i)
    
    def load_mindmap(self, mindmap_id: str) -> Tuple[Dict[str, Any], Node]:
        """Load mindmap from database
        
        Args:
            mindmap_id: ID of the mindmap to load
            
        Returns:
            Tuple of (mindmap metadata dict, root node)
            
        Raises:
            ValueError: If mindmap not found
        """
        with self.transaction() as cursor:
            # Get mindmap metadata
            cursor.execute("""
                SELECT title, description, mermaid_source, created_at, 
                       updated_at, creator_id, metadata
                FROM mindmaps 
                WHERE id = ? AND client_id = ?
            """, (mindmap_id, self.client_id))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Mindmap {mindmap_id} not found")
            
            title, description, mermaid_source, created_at, updated_at, creator_id, metadata_json = result
            
            mindmap_data = {
                'id': mindmap_id,
                'title': title,
                'description': description,
                'mermaid_source': mermaid_source,
                'created_at': created_at,
                'updated_at': updated_at,
                'creator_id': creator_id,
                'metadata': json.loads(metadata_json) if metadata_json else {}
            }
            
            # Load nodes
            cursor.execute("""
                SELECT id, node_id, parent_id, text, shape, icon, 
                       css_class, metadata, position_index
                FROM mindmap_nodes 
                WHERE mindmap_id = ?
                ORDER BY parent_id, position_index
            """, (mindmap_id,))
            
            nodes = cursor.fetchall()
            
            # Build tree
            node_map = {}
            root = None
            
            for node_data in nodes:
                db_id, node_id, parent_id, text, shape, icon, css_class, metadata_json, pos_idx = node_data
                
                node = Node(
                    node_id,
                    text=text,
                    shape=NodeShape[shape] if shape else NodeShape.DEFAULT,
                    icon=icon,
                    css_class=css_class,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                
                node_map[db_id] = node
                
                if not parent_id:
                    root = node
                else:
                    parent = node_map.get(parent_id)
                    if parent:
                        node.parent = parent
            
            if not root:
                raise ValueError(f"No root node found for mindmap {mindmap_id}")
            
            logger.info(f"Loaded mindmap {mindmap_id} with {len(nodes)} nodes")
            return mindmap_data, root
    
    def list_mindmaps(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List available mindmaps
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of mindmap metadata dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, description, created_at, updated_at, 
                       (SELECT COUNT(*) FROM mindmap_nodes WHERE mindmap_id = m.id) as node_count
                FROM mindmaps m
                WHERE client_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (self.client_id, limit, offset))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def delete_mindmap(self, mindmap_id: str) -> bool:
        """Delete a mindmap
        
        Args:
            mindmap_id: ID of the mindmap to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self.transaction() as cursor:
            cursor.execute("""
                DELETE FROM mindmaps 
                WHERE id = ? AND client_id = ?
            """, (mindmap_id, self.client_id))
            
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted mindmap {mindmap_id}")
            
            return deleted
    
    def search_mindmaps(self, query: str) -> List[Dict[str, Any]]:
        """Search mindmaps by title or content
        
        Args:
            query: Search query
            
        Returns:
            List of matching mindmap metadata
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            search_pattern = f"%{query}%"
            
            cursor.execute("""
                SELECT DISTINCT m.id, m.title, m.description, m.created_at, m.updated_at
                FROM mindmaps m
                LEFT JOIN mindmap_nodes n ON m.id = n.mindmap_id
                WHERE m.client_id = ?
                  AND (m.title LIKE ? OR m.description LIKE ? OR n.text LIKE ?)
                ORDER BY m.updated_at DESC
                LIMIT 50
            """, (self.client_id, search_pattern, search_pattern, search_pattern))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_mindmap_versions(self, mindmap_id: str) -> List[Dict[str, Any]]:
        """Get version history for a mindmap
        
        Args:
            mindmap_id: ID of the mindmap
            
        Returns:
            List of version records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version_number, changed_by, change_description, created_at
                FROM mindmap_versions
                WHERE mindmap_id = ?
                ORDER BY version_number DESC
                LIMIT 50
            """, (mindmap_id,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def restore_version(self, mindmap_id: str, version_number: int) -> None:
        """Restore a specific version of a mindmap
        
        Args:
            mindmap_id: ID of the mindmap
            version_number: Version to restore
        """
        with self.transaction() as cursor:
            # Get the version
            cursor.execute("""
                SELECT mermaid_source
                FROM mindmap_versions
                WHERE mindmap_id = ? AND version_number = ?
            """, (mindmap_id, version_number))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Version {version_number} not found for mindmap {mindmap_id}")
            
            mermaid_source = result[0]
            
            # Update the mindmap
            cursor.execute("""
                UPDATE mindmaps
                SET mermaid_source = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND client_id = ?
            """, (mermaid_source, mindmap_id, self.client_id))
            
            # Create a new version entry
            new_version = self._get_next_version_number(cursor, mindmap_id)
            cursor.execute("""
                INSERT INTO mindmap_versions
                (mindmap_id, version_number, mermaid_source, changed_by, change_description)
                VALUES (?, ?, ?, ?, ?)
            """, (
                mindmap_id,
                new_version,
                mermaid_source,
                self.client_id,
                f"Restored from version {version_number}"
            ))
            
            logger.info(f"Restored mindmap {mindmap_id} to version {version_number}")
    
    def add_collaborator(self, mindmap_id: str, user_id: str, 
                        permission: str = "view") -> None:
        """Add a collaborator to a mindmap
        
        Args:
            mindmap_id: ID of the mindmap
            user_id: ID of the user to add
            permission: Permission level (view, edit, admin)
        """
        if permission not in ('view', 'edit', 'admin'):
            raise ValueError(f"Invalid permission: {permission}")
        
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO mindmap_collaborators
                (mindmap_id, user_id, permission)
                VALUES (?, ?, ?)
            """, (mindmap_id, user_id, permission))
            
            logger.info(f"Added collaborator {user_id} to mindmap {mindmap_id} with {permission} permission")
    
    def _get_next_version_number(self, cursor, mindmap_id: str) -> int:
        """Get the next version number for a mindmap
        
        Args:
            cursor: Database cursor
            mindmap_id: ID of the mindmap
            
        Returns:
            Next version number
        """
        cursor.execute("""
            SELECT MAX(version_number) 
            FROM mindmap_versions 
            WHERE mindmap_id = ?
        """, (mindmap_id,))
        
        result = cursor.fetchone()
        return (result[0] or 0) + 1
    
    def _count_nodes(self, node: Node) -> int:
        """Count total nodes in a tree
        
        Args:
            node: Root node
            
        Returns:
            Total node count
        """
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def export_mindmap_json(self, mindmap_id: str) -> Dict[str, Any]:
        """Export mindmap as JSON
        
        Args:
            mindmap_id: ID of the mindmap
            
        Returns:
            JSON-serializable dictionary
        """
        metadata, root = self.load_mindmap(mindmap_id)
        
        def node_to_dict(node: Node) -> Dict[str, Any]:
            return {
                'id': node.name,
                'text': getattr(node, 'text', node.name),
                'shape': getattr(node, 'shape', NodeShape.DEFAULT).name,
                'icon': getattr(node, 'icon', None),
                'css_class': getattr(node, 'css_class', None),
                'metadata': getattr(node, 'metadata', {}),
                'children': [node_to_dict(child) for child in node.children]
            }
        
        return {
            'metadata': metadata,
            'root': node_to_dict(root)
        }