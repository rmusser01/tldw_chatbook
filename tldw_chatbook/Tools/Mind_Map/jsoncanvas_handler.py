# jsoncanvas_handler.py
# Description: JSON Canvas format handler for import/export of mindmaps
#
"""
JSON Canvas Handler
------------------

Handles import and export of mindmaps to/from Obsidian's JSON Canvas format.
Supports:
- Text, file, link, and group nodes
- Edges with labels and styles
- Color coding
- Hierarchical layout algorithms
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from anytree import Node, PreOrderIter
from datetime import datetime
from loguru import logger

from .mermaid_parser import NodeShape


class JSONCanvasHandler:
    """Handler for JSON Canvas format import/export"""
    
    # Color mapping from our system to JSON Canvas preset colors
    COLOR_MAP = {
        'red': '1',
        'orange': '2', 
        'yellow': '3',
        'green': '4',
        'cyan': '5',
        'purple': '6'
    }
    
    # Reverse color mapping
    REVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}
    
    # Default canvas dimensions
    DEFAULT_NODE_WIDTH = 250
    DEFAULT_NODE_HEIGHT = 60
    NODE_PADDING_X = 50
    NODE_PADDING_Y = 30
    
    @classmethod
    def from_json_canvas(cls, canvas_data: str) -> Node:
        """Import mindmap from JSON Canvas format
        
        Args:
            canvas_data: JSON Canvas format string
            
        Returns:
            Root node of the imported mindmap
            
        Raises:
            ValueError: If JSON is invalid or required fields are missing
        """
        try:
            data = json.loads(canvas_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        nodes_data = data.get('nodes', [])
        edges_data = data.get('edges', [])
        
        if not nodes_data:
            raise ValueError("No nodes found in JSON Canvas")
        
        # Create node objects dictionary
        node_objects = {}
        
        # First pass: Create all nodes
        for node_data in nodes_data:
            node_id = node_data.get('id')
            if not node_id:
                logger.warning("Node without ID found, skipping")
                continue
            
            node_type = node_data.get('type', 'text')
            
            # Extract text content based on node type
            if node_type == 'text':
                text = node_data.get('text', '')
            elif node_type == 'file':
                text = f"📄 {Path(node_data.get('file', '')).name}"
            elif node_type == 'link':
                text = f"🔗 {node_data.get('url', '')}"
            elif node_type == 'group':
                text = node_data.get('label', 'Group')
            else:
                text = f"Unknown type: {node_type}"
            
            # Create node with metadata
            node = Node(
                node_id,
                text=text,
                x=node_data.get('x', 0),
                y=node_data.get('y', 0),
                width=node_data.get('width', cls.DEFAULT_NODE_WIDTH),
                height=node_data.get('height', cls.DEFAULT_NODE_HEIGHT),
                canvas_type=node_type,
                color=node_data.get('color'),
                metadata={
                    'type': node_type,
                    'original_data': node_data
                }
            )
            
            # Store additional type-specific data
            if node_type == 'file':
                node.file_path = node_data.get('file')
                node.subpath = node_data.get('subpath')
            elif node_type == 'link':
                node.url = node_data.get('url')
            elif node_type == 'group':
                node.background = node_data.get('background')
                node.backgroundStyle = node_data.get('backgroundStyle')
            
            node_objects[node_id] = node
        
        # Build parent-child relationships from edges
        parent_map = {}  # child_id -> parent_id
        
        for edge_data in edges_data:
            from_node = edge_data.get('fromNode')
            to_node = edge_data.get('toNode')
            
            if from_node and to_node:
                if to_node not in parent_map:
                    parent_map[to_node] = from_node
                else:
                    logger.warning(f"Node {to_node} already has a parent, skipping edge from {from_node}")
        
        # Find root nodes (nodes without parents)
        root_candidates = []
        for node_id, node in node_objects.items():
            if node_id not in parent_map:
                root_candidates.append(node)
        
        # Determine the root node
        if len(root_candidates) == 0:
            # If no clear root (circular), pick the first node
            root = list(node_objects.values())[0]
            logger.warning("No clear root found, using first node")
        elif len(root_candidates) == 1:
            root = root_candidates[0]
        else:
            # Multiple roots - create a synthetic root
            root = Node("root", text="Root", x=0, y=0)
            for candidate in root_candidates:
                candidate.parent = root
        
        # Build the tree structure
        for child_id, parent_id in parent_map.items():
            if child_id in node_objects and parent_id in node_objects:
                child_node = node_objects[child_id]
                parent_node = node_objects[parent_id]
                child_node.parent = parent_node
        
        logger.info(f"Imported JSON Canvas with {len(node_objects)} nodes")
        return root
    
    @classmethod
    def to_json_canvas(cls, root: Node, 
                       layout: str = 'hierarchical',
                       include_metadata: bool = True) -> str:
        """Export mindmap to JSON Canvas format
        
        Args:
            root: Root node of the mindmap
            layout: Layout algorithm ('hierarchical', 'radial', 'grid')
            include_metadata: Whether to include node metadata
            
        Returns:
            JSON Canvas format string
        """
        nodes = []
        edges = []
        
        # Calculate positions based on layout
        positions = cls._calculate_layout(root, layout)
        
        # Create nodes
        for node in PreOrderIter(root):
            node_id = cls._generate_node_id(node)
            pos = positions.get(node, (0, 0))
            
            # Determine node type and content
            node_type = 'text'  # Default type
            node_data = {
                'id': node_id,
                'type': node_type,
                'x': pos[0],
                'y': pos[1],
                'width': cls.DEFAULT_NODE_WIDTH,
                'height': cls.DEFAULT_NODE_HEIGHT
            }
            
            # Add text content
            text = node.text if hasattr(node, 'text') else node.name
            node_data['text'] = text
            
            # Add color if available
            if hasattr(node, 'color') and node.color:
                if node.color in cls.COLOR_MAP:
                    node_data['color'] = cls.COLOR_MAP[node.color]
                elif node.color.startswith('#'):
                    node_data['color'] = node.color
            
            # Check for special node types based on metadata
            if hasattr(node, 'metadata') and node.metadata:
                meta = node.metadata
                if meta.get('type') == 'file':
                    node_data['type'] = 'file'
                    node_data['file'] = meta.get('file_path', '')
                elif meta.get('type') == 'link':
                    node_data['type'] = 'link'
                    node_data['url'] = meta.get('url', '')
                elif meta.get('type') == 'group':
                    node_data['type'] = 'group'
                    node_data['label'] = text
            
            nodes.append(node_data)
            
            # Create edges to children
            if node.children:
                for child in node.children:
                    child_id = cls._generate_node_id(child)
                    edge = {
                        'id': str(uuid.uuid4()),
                        'fromNode': node_id,
                        'toNode': child_id
                    }
                    
                    # Add optional edge styling
                    if layout == 'hierarchical':
                        edge['fromSide'] = 'bottom'
                        edge['toSide'] = 'top'
                    elif layout == 'radial':
                        edge['fromSide'] = 'right'
                        edge['toSide'] = 'left'
                    
                    edges.append(edge)
        
        # Create the JSON Canvas structure
        canvas = {
            'nodes': nodes,
            'edges': edges
        }
        
        return json.dumps(canvas, indent=2, ensure_ascii=False)
    
    @classmethod
    def _generate_node_id(cls, node: Node) -> str:
        """Generate a unique ID for a node
        
        Args:
            node: Node to generate ID for
            
        Returns:
            Unique string ID
        """
        if hasattr(node, 'canvas_id'):
            return node.canvas_id
        
        # Use node name if it looks like an ID
        if node.name and not ' ' in node.name:
            return node.name
        
        # Generate a new UUID
        return str(uuid.uuid4())
    
    @classmethod
    def _calculate_layout(cls, root: Node, layout: str) -> Dict[Node, Tuple[int, int]]:
        """Calculate node positions based on layout algorithm
        
        Args:
            root: Root node of the mindmap
            layout: Layout algorithm to use
            
        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        positions = {}
        
        if layout == 'hierarchical':
            positions = cls._hierarchical_layout(root)
        elif layout == 'radial':
            positions = cls._radial_layout(root)
        elif layout == 'grid':
            positions = cls._grid_layout(root)
        else:
            # Default to hierarchical
            positions = cls._hierarchical_layout(root)
        
        return positions
    
    @classmethod
    def _hierarchical_layout(cls, root: Node) -> Dict[Node, Tuple[int, int]]:
        """Calculate hierarchical (top-down tree) layout
        
        Args:
            root: Root node of the mindmap
            
        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        positions = {}
        levels = {}  # depth -> list of nodes
        
        # Group nodes by depth
        for node in PreOrderIter(root):
            depth = node.depth
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node)
        
        # Calculate positions level by level
        y = 0
        for depth in sorted(levels.keys()):
            nodes_at_level = levels[depth]
            total_width = len(nodes_at_level) * (cls.DEFAULT_NODE_WIDTH + cls.NODE_PADDING_X)
            
            # Center the level horizontally
            x = -(total_width // 2)
            
            for node in nodes_at_level:
                positions[node] = (x, y)
                x += cls.DEFAULT_NODE_WIDTH + cls.NODE_PADDING_X
            
            y += cls.DEFAULT_NODE_HEIGHT + cls.NODE_PADDING_Y
        
        return positions
    
    @classmethod
    def _radial_layout(cls, root: Node) -> Dict[Node, Tuple[int, int]]:
        """Calculate radial (circular) layout
        
        Args:
            root: Root node of the mindmap
            
        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        import math
        
        positions = {}
        positions[root] = (0, 0)  # Root at center
        
        # Process each level as a ring
        levels = {}
        for node in PreOrderIter(root):
            if node == root:
                continue
            depth = node.depth
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node)
        
        for depth, nodes in levels.items():
            radius = depth * 200  # Increase radius for each level
            angle_step = 2 * math.pi / len(nodes)
            
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = int(radius * math.cos(angle))
                y = int(radius * math.sin(angle))
                positions[node] = (x, y)
        
        return positions
    
    @classmethod
    def _grid_layout(cls, root: Node) -> Dict[Node, Tuple[int, int]]:
        """Calculate grid layout
        
        Args:
            root: Root node of the mindmap
            
        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        import math
        
        positions = {}
        all_nodes = list(PreOrderIter(root))
        
        # Calculate grid dimensions
        total_nodes = len(all_nodes)
        cols = math.ceil(math.sqrt(total_nodes))
        
        for i, node in enumerate(all_nodes):
            row = i // cols
            col = i % cols
            x = col * (cls.DEFAULT_NODE_WIDTH + cls.NODE_PADDING_X)
            y = row * (cls.DEFAULT_NODE_HEIGHT + cls.NODE_PADDING_Y)
            positions[node] = (x, y)
        
        return positions
    
    @classmethod
    def validate_canvas(cls, canvas_data: str) -> Tuple[bool, List[str]]:
        """Validate JSON Canvas format
        
        Args:
            canvas_data: JSON Canvas format string to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            data = json.loads(canvas_data)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]
        
        # Check top-level structure
        if not isinstance(data, dict):
            errors.append("Root must be an object")
            return False, errors
        
        # Validate nodes
        if 'nodes' in data:
            if not isinstance(data['nodes'], list):
                errors.append("'nodes' must be an array")
            else:
                for i, node in enumerate(data['nodes']):
                    node_errors = cls._validate_node(node, i)
                    errors.extend(node_errors)
        
        # Validate edges
        if 'edges' in data:
            if not isinstance(data['edges'], list):
                errors.append("'edges' must be an array")
            else:
                node_ids = {n.get('id') for n in data.get('nodes', [])}
                for i, edge in enumerate(data['edges']):
                    edge_errors = cls._validate_edge(edge, i, node_ids)
                    errors.extend(edge_errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_node(cls, node: Dict, index: int) -> List[str]:
        """Validate a single node
        
        Args:
            node: Node dictionary to validate
            index: Node index in array
            
        Returns:
            List of validation errors
        """
        errors = []
        prefix = f"Node {index}"
        
        # Required fields
        if 'id' not in node:
            errors.append(f"{prefix}: missing required field 'id'")
        if 'type' not in node:
            errors.append(f"{prefix}: missing required field 'type'")
        elif node['type'] not in ['text', 'file', 'link', 'group']:
            errors.append(f"{prefix}: invalid type '{node['type']}'")
        
        # Position and dimensions
        for field in ['x', 'y', 'width', 'height']:
            if field not in node:
                errors.append(f"{prefix}: missing required field '{field}'")
            elif not isinstance(node[field], (int, float)):
                errors.append(f"{prefix}: field '{field}' must be a number")
        
        # Type-specific validation
        node_type = node.get('type')
        if node_type == 'text' and 'text' not in node:
            errors.append(f"{prefix}: text node missing 'text' field")
        elif node_type == 'file' and 'file' not in node:
            errors.append(f"{prefix}: file node missing 'file' field")
        elif node_type == 'link' and 'url' not in node:
            errors.append(f"{prefix}: link node missing 'url' field")
        
        # Optional color validation
        if 'color' in node:
            color = node['color']
            if not (color in ['1', '2', '3', '4', '5', '6'] or 
                   (isinstance(color, str) and color.startswith('#'))):
                errors.append(f"{prefix}: invalid color '{color}'")
        
        return errors
    
    @classmethod
    def _validate_edge(cls, edge: Dict, index: int, node_ids: set) -> List[str]:
        """Validate a single edge
        
        Args:
            edge: Edge dictionary to validate
            index: Edge index in array
            node_ids: Set of valid node IDs
            
        Returns:
            List of validation errors
        """
        errors = []
        prefix = f"Edge {index}"
        
        # Required fields
        if 'id' not in edge:
            errors.append(f"{prefix}: missing required field 'id'")
        if 'fromNode' not in edge:
            errors.append(f"{prefix}: missing required field 'fromNode'")
        elif edge['fromNode'] not in node_ids:
            errors.append(f"{prefix}: fromNode '{edge['fromNode']}' does not exist")
        if 'toNode' not in edge:
            errors.append(f"{prefix}: missing required field 'toNode'")
        elif edge['toNode'] not in node_ids:
            errors.append(f"{prefix}: toNode '{edge['toNode']}' does not exist")
        
        # Optional side validation
        valid_sides = ['top', 'right', 'bottom', 'left']
        if 'fromSide' in edge and edge['fromSide'] not in valid_sides:
            errors.append(f"{prefix}: invalid fromSide '{edge['fromSide']}'")
        if 'toSide' in edge and edge['toSide'] not in valid_sides:
            errors.append(f"{prefix}: invalid toSide '{edge['toSide']}'")
        
        return errors