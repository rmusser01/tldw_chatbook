# mindmap_model.py
# Description: Model for mindmap data management
#
"""
Mindmap Model
------------

Manages mindmap data using anytree, including:
- Tree structure management
- Node selection and expansion state
- Search and filtering
- Lazy loading support
"""

from anytree import Node, PreOrderIter, find_by_attr
from typing import List, Optional, Set, Dict, Any
from loguru import logger

from .mermaid_parser import MermaidMindmapParser, ExtendedMermaidParser, NodeShape
from .jsoncanvas_handler import JSONCanvasHandler


class MindmapModel:
    """Model for mindmap data using anytree"""
    
    def __init__(self):
        self.root: Optional[Node] = None
        self.selected_node: Optional[Node] = None
        self.expanded_nodes: Set[Node] = set()
        self.search_results: List[Node] = []
        self.search_query: str = ""
        self.parser = ExtendedMermaidParser()  # Use extended parser by default
        
    def load_from_mermaid(self, mermaid_code: str) -> None:
        """Load mindmap from Mermaid syntax
        
        Args:
            mermaid_code: Mermaid mindmap syntax string
            
        Raises:
            ValueError: If parsing fails
        """
        try:
            self.root = self.parser.parse(mermaid_code)
            self.selected_node = self.root
            self.expanded_nodes = {self.root}
            logger.info(f"Loaded mindmap with root: {self.root.text}")
        except Exception as e:
            logger.error(f"Failed to load mindmap: {e}")
            raise
        
    def load_from_json_canvas(self, canvas_data: str) -> None:
        """Load mindmap from JSON Canvas format
        
        Args:
            canvas_data: JSON Canvas format string
            
        Raises:
            ValueError: If parsing fails
        """
        try:
            self.root = JSONCanvasHandler.from_json_canvas(canvas_data)
            self.selected_node = self.root
            self.expanded_nodes = {self.root}
            logger.info(f"Loaded JSON Canvas mindmap with root: {self.root.text}")
        except Exception as e:
            logger.error(f"Failed to load JSON Canvas: {e}")
            raise
    
    def load_from_database(self, mindmap_id: str, db) -> None:
        """Load mindmap from database
        
        Args:
            mindmap_id: ID of the mindmap to load
            db: Database instance
        """
        # Implementation will be added when DB module is created
        pass
    
    def get_visible_nodes(self) -> List[Node]:
        """Get all nodes that should be visible (respecting collapse state)
        
        Returns:
            List of visible nodes in pre-order traversal
        """
        if not self.root:
            return []
        
        visible = []
        
        def _traverse(node: Node, parent_expanded: bool = True):
            """Recursively traverse and collect visible nodes"""
            if parent_expanded:
                visible.append(node)
                
                # Check if this node is expanded
                is_expanded = node in self.expanded_nodes
                
                # Traverse children only if node is expanded
                if is_expanded and node.children:
                    for child in node.children:
                        _traverse(child, True)
        
        _traverse(self.root)
        return visible
    
    def toggle_node(self, node: Node) -> None:
        """Toggle expand/collapse state of a node
        
        Args:
            node: Node to toggle
        """
        if node.children:  # Only toggle if node has children
            if node in self.expanded_nodes:
                self.expanded_nodes.remove(node)
                logger.debug(f"Collapsed node: {node.text}")
            else:
                self.expanded_nodes.add(node)
                logger.debug(f"Expanded node: {node.text}")
    
    def expand_node(self, node: Node) -> None:
        """Expand a specific node
        
        Args:
            node: Node to expand
        """
        if node.children:
            self.expanded_nodes.add(node)
    
    def collapse_node(self, node: Node) -> None:
        """Collapse a specific node
        
        Args:
            node: Node to collapse
        """
        self.expanded_nodes.discard(node)
    
    def expand_all(self) -> None:
        """Expand all nodes in the tree"""
        if not self.root:
            return
            
        for node in PreOrderIter(self.root):
            if node.children:
                self.expanded_nodes.add(node)
    
    def collapse_all(self) -> None:
        """Collapse all nodes except root"""
        if not self.root:
            return
            
        self.expanded_nodes = {self.root}
    
    def search(self, query: str) -> List[Node]:
        """Search for nodes containing query text
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching nodes
        """
        if not self.root or not query:
            self.search_results = []
            self.search_query = ""
            return []
        
        self.search_query = query
        self.search_results = []
        query_lower = query.lower()
        
        for node in PreOrderIter(self.root):
            # Search in text
            if hasattr(node, 'text') and query_lower in node.text.lower():
                self.search_results.append(node)
                # Ensure path to result is expanded
                self._expand_path_to_node(node)
            # Also search in node ID/name
            elif query_lower in node.name.lower():
                self.search_results.append(node)
                self._expand_path_to_node(node)
        
        logger.info(f"Search for '{query}' found {len(self.search_results)} results")
        return self.search_results
    
    def _expand_path_to_node(self, node: Node) -> None:
        """Expand all ancestors of a node
        
        Args:
            node: Target node
        """
        current = node.parent
        while current:
            self.expanded_nodes.add(current)
            current = current.parent
    
    def get_node_path(self, node: Node) -> List[Node]:
        """Get path from root to node
        
        Args:
            node: Target node
            
        Returns:
            List of nodes from root to target
        """
        path = []
        current = node
        
        while current:
            path.append(current)
            current = current.parent
        
        path.reverse()
        return path
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Find node by ID
        
        Args:
            node_id: Node ID to search for
            
        Returns:
            Node if found, None otherwise
        """
        if not self.root:
            return None
            
        return find_by_attr(self.root, name='name', value=node_id)
    
    def move_selection_up(self) -> bool:
        """Move selection to previous visible node
        
        Returns:
            True if selection moved, False otherwise
        """
        if not self.selected_node:
            return False
            
        visible_nodes = self.get_visible_nodes()
        try:
            current_idx = visible_nodes.index(self.selected_node)
            if current_idx > 0:
                self.selected_node = visible_nodes[current_idx - 1]
                return True
        except ValueError:
            logger.warning("Selected node not in visible nodes")
            
        return False
    
    def move_selection_down(self) -> bool:
        """Move selection to next visible node
        
        Returns:
            True if selection moved, False otherwise
        """
        if not self.selected_node:
            return False
            
        visible_nodes = self.get_visible_nodes()
        try:
            current_idx = visible_nodes.index(self.selected_node)
            if current_idx < len(visible_nodes) - 1:
                self.selected_node = visible_nodes[current_idx + 1]
                return True
        except ValueError:
            logger.warning("Selected node not in visible nodes")
            
        return False
    
    def jump_to_parent(self) -> bool:
        """Jump selection to parent node
        
        Returns:
            True if selection moved, False otherwise
        """
        if self.selected_node and self.selected_node.parent:
            self.selected_node = self.selected_node.parent
            return True
        return False
    
    def jump_to_first_child(self) -> bool:
        """Jump selection to first child
        
        Returns:
            True if selection moved, False otherwise
        """
        if self.selected_node and self.selected_node.children:
            # Ensure node is expanded first
            self.expand_node(self.selected_node)
            self.selected_node = self.selected_node.children[0]
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the mindmap
        
        Returns:
            Dictionary with statistics
        """
        if not self.root:
            return {
                'total_nodes': 0,
                'visible_nodes': 0,
                'expanded_nodes': 0,
                'max_depth': 0
            }
        
        total_nodes = len(list(PreOrderIter(self.root)))
        visible_nodes = len(self.get_visible_nodes())
        expanded_nodes = len(self.expanded_nodes)
        
        # Calculate max depth
        max_depth = 0
        for node in PreOrderIter(self.root):
            depth = node.depth
            if depth > max_depth:
                max_depth = depth
        
        return {
            'total_nodes': total_nodes,
            'visible_nodes': visible_nodes,
            'expanded_nodes': expanded_nodes,
            'max_depth': max_depth
        }