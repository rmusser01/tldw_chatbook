# mindmap_exporter.py
# Description: Export functionality for mindmaps
#
"""
Mindmap Exporter
---------------

Export mindmaps to various formats:
- Markdown outline
- GraphViz DOT
- ASCII art
- Anki flashcards
- JSON
- HTML
- JSON Canvas (Obsidian)
- OPML
"""

from typing import List, Dict, Any, Optional, TextIO
from pathlib import Path
from anytree import Node, PreOrderIter, RenderTree
from datetime import datetime
import json
import html
from loguru import logger

from .mermaid_parser import NodeShape
from .jsoncanvas_handler import JSONCanvasHandler


class MindmapExporter:
    """Export mindmap to various formats"""
    
    @staticmethod
    def to_markdown(root: Node, include_metadata: bool = False) -> str:
        """Export to Markdown outline
        
        Args:
            root: Root node of the mindmap
            include_metadata: Whether to include node metadata
            
        Returns:
            Markdown formatted string
        """
        lines = []
        lines.append(f"# {root.text if hasattr(root, 'text') else root.name}\n")
        
        if include_metadata:
            lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        def render_node(node: Node, level: int):
            if node == root:
                return  # Skip root as it's the title
            
            indent = "  " * (level - 1)
            
            # Choose markdown header or bullet based on level
            if level <= 6:
                prefix = "#" * level
                lines.append(f"\n{prefix} {node.text if hasattr(node, 'text') else node.name}")
            else:
                bullet = "-" if level % 2 == 0 else "*"
                lines.append(f"{indent}{bullet} {node.text if hasattr(node, 'text') else node.name}")
            
            # Add metadata if requested
            if include_metadata and hasattr(node, 'metadata') and node.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in node.metadata.items())
                lines.append(f"{indent}  _{meta_str}_")
            
            # Render children
            for child in node.children:
                render_node(child, level + 1)
        
        for child in root.children:
            render_node(child, 1)
        
        return "\n".join(lines)
    
    @staticmethod
    def to_graphviz(root: Node, 
                    rankdir: str = "TB",
                    node_shape: str = "box",
                    include_style: bool = True) -> str:
        """Export to GraphViz DOT format
        
        Args:
            root: Root node of the mindmap
            rankdir: Graph direction (TB, LR, BT, RL)
            node_shape: Default node shape
            include_style: Whether to include styling
            
        Returns:
            DOT format string
        """
        lines = ["digraph mindmap {"]
        
        # Graph attributes
        lines.append(f'  rankdir="{rankdir}";')
        lines.append(f'  node [shape={node_shape}];')
        
        if include_style:
            lines.append('  node [style="rounded,filled", fillcolor="lightblue"];')
            lines.append('  edge [color="gray"];')
        
        # Track node IDs to handle duplicates
        node_ids = {}
        node_counter = 0
        
        def get_node_id(node: Node) -> str:
            if node not in node_ids:
                nonlocal node_counter
                node_ids[node] = f"node_{node_counter}"
                node_counter += 1
            return node_ids[node]
        
        def escape_label(text: str) -> str:
            """Escape special characters for GraphViz"""
            return text.replace('"', '\\"').replace('\n', '\\n')
        
        # Add nodes
        for node in PreOrderIter(root):
            node_id = get_node_id(node)
            label = escape_label(node.text if hasattr(node, 'text') else node.name)
            
            # Style based on node type
            attrs = [f'label="{label}"']
            
            if hasattr(node, 'shape'):
                shape_map = {
                    NodeShape.DOUBLE_CIRCLE: 'doublecircle',
                    NodeShape.CIRCLE: 'circle',
                    NodeShape.SQUARE: 'box',
                    NodeShape.ROUNDED: 'box',
                    NodeShape.HEXAGON: 'hexagon',
                    NodeShape.CLOUD: 'ellipse'
                }
                dot_shape = shape_map.get(node.shape, node_shape)
                attrs.append(f'shape={dot_shape}')
            
            if node == root:
                attrs.append('fillcolor="lightgreen"')
                attrs.append('penwidth=2')
            
            lines.append(f'  {node_id} [{", ".join(attrs)}];')
        
        # Add edges
        for node in PreOrderIter(root):
            if node.parent:
                parent_id = get_node_id(node.parent)
                child_id = get_node_id(node)
                lines.append(f'  {parent_id} -> {child_id};')
        
        lines.append("}")
        return "\n".join(lines)
    
    @staticmethod
    def to_ascii_art(root: Node, max_width: int = 80) -> str:
        """Export to ASCII art representation
        
        Args:
            root: Root node of the mindmap
            max_width: Maximum width for the output
            
        Returns:
            ASCII art string
        """
        lines = []
        lines.append("=" * max_width)
        lines.append(f" MINDMAP: {root.text if hasattr(root, 'text') else root.name} ".center(max_width))
        lines.append("=" * max_width)
        lines.append("")
        
        # Use anytree's RenderTree for nice ASCII tree
        for pre, _, node in RenderTree(root):
            text = node.text if hasattr(node, 'text') else node.name
            
            # Truncate if too long
            available_width = max_width - len(pre) - 3
            if len(text) > available_width:
                text = text[:available_width-3] + "..."
            
            lines.append(f"{pre}{text}")
        
        lines.append("")
        lines.append("-" * max_width)
        
        # Add statistics
        total_nodes = len(list(PreOrderIter(root)))
        max_depth = max(node.depth for node in PreOrderIter(root))
        
        lines.append(f"Total nodes: {total_nodes}")
        lines.append(f"Max depth: {max_depth}")
        lines.append("=" * max_width)
        
        return "\n".join(lines)
    
    @staticmethod
    def to_flashcards(root: Node, 
                      card_type: str = "basic",
                      include_context: bool = True) -> List[Dict[str, Any]]:
        """Generate Anki-compatible flashcards from mindmap
        
        Args:
            root: Root node of the mindmap
            card_type: Type of flashcard (basic, cloze)
            include_context: Whether to include parent context
            
        Returns:
            List of flashcard dictionaries
        """
        flashcards = []
        
        def create_basic_cards(node: Node):
            """Create basic Q&A flashcards"""
            if node.parent and node != root:
                # Create question from parent, answer from node
                question = node.parent.text if hasattr(node.parent, 'text') else node.parent.name
                answer = node.text if hasattr(node, 'text') else node.name
                
                # Add context if requested
                if include_context and node.parent.parent:
                    context = node.parent.parent.text if hasattr(node.parent.parent, 'text') else node.parent.parent.name
                    question = f"{context} - {question}"
                
                flashcard = {
                    "type": "basic",
                    "question": question,
                    "answer": answer,
                    "tags": ["mindmap", "auto-generated"],
                    "deck": "Mindmap Cards"
                }
                
                # Add metadata tags
                if hasattr(node, 'metadata') and node.metadata:
                    if 'type' in node.metadata:
                        flashcard['tags'].append(f"type:{node.metadata['type']}")
                
                flashcards.append(flashcard)
            
            # Recurse to children
            for child in node.children:
                create_basic_cards(child)
        
        def create_cloze_cards(node: Node, path: List[str] = None):
            """Create cloze deletion cards"""
            if path is None:
                path = []
            
            current_text = node.text if hasattr(node, 'text') else node.name
            current_path = path + [current_text]
            
            # Create cloze card if we have enough context
            if len(current_path) >= 2:
                # Build cloze text
                cloze_parts = []
                for i, part in enumerate(current_path):
                    if i == len(current_path) - 1:
                        cloze_parts.append(f"{{{{c1::{part}}}}}")
                    else:
                        cloze_parts.append(part)
                
                cloze_text = " → ".join(cloze_parts)
                
                flashcard = {
                    "type": "cloze",
                    "text": cloze_text,
                    "tags": ["mindmap", "cloze", "auto-generated"],
                    "deck": "Mindmap Cloze Cards"
                }
                
                flashcards.append(flashcard)
            
            # Recurse to children
            for child in node.children:
                create_cloze_cards(child, current_path)
        
        # Generate cards based on type
        if card_type == "basic":
            create_basic_cards(root)
        elif card_type == "cloze":
            create_cloze_cards(root)
        else:
            # Generate both types
            create_basic_cards(root)
            create_cloze_cards(root)
        
        return flashcards
    
    @staticmethod
    def to_json(root: Node, pretty: bool = True) -> str:
        """Export to JSON format
        
        Args:
            root: Root node of the mindmap
            pretty: Whether to pretty-print the JSON
            
        Returns:
            JSON string
        """
        def node_to_dict(node: Node) -> Dict[str, Any]:
            """Convert node to dictionary"""
            node_dict = {
                'id': node.name,
                'text': node.text if hasattr(node, 'text') else node.name,
                'children': []
            }
            
            # Add optional attributes
            if hasattr(node, 'shape'):
                node_dict['shape'] = node.shape.name
            
            if hasattr(node, 'icon') and node.icon:
                node_dict['icon'] = node.icon
            
            if hasattr(node, 'css_class') and node.css_class:
                node_dict['css_class'] = node.css_class
            
            if hasattr(node, 'metadata') and node.metadata:
                node_dict['metadata'] = node.metadata
            
            # Add children
            for child in node.children:
                node_dict['children'].append(node_to_dict(child))
            
            return node_dict
        
        data = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'root': node_to_dict(root)
        }
        
        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(data, ensure_ascii=False)
    
    @staticmethod
    def to_html(root: Node, 
                include_style: bool = True,
                collapsible: bool = True) -> str:
        """Export to HTML format
        
        Args:
            root: Root node of the mindmap
            include_style: Whether to include CSS styling
            collapsible: Whether to make the tree collapsible
            
        Returns:
            HTML string
        """
        lines = ['<!DOCTYPE html>', '<html>', '<head>']
        lines.append('<meta charset="UTF-8">')
        lines.append(f'<title>{html.escape(root.text if hasattr(root, "text") else root.name)}</title>')
        
        if include_style:
            lines.append('<style>')
            lines.append("""
            body { font-family: Arial, sans-serif; margin: 20px; }
            .mindmap { background: #f5f5f5; padding: 20px; border-radius: 8px; }
            .node { margin-left: 20px; }
            .node-content { 
                padding: 5px 10px; 
                margin: 5px 0; 
                background: white; 
                border-radius: 4px;
                border-left: 3px solid #4CAF50;
                cursor: pointer;
            }
            .node-content:hover { background: #f0f0f0; }
            .children { margin-left: 20px; }
            .collapsed { display: none; }
            .expand-icon { 
                display: inline-block; 
                width: 20px; 
                color: #666;
                user-select: none;
            }
            .shape-circle { border-left-color: #2196F3; }
            .shape-square { border-left-color: #FF9800; }
            .shape-hexagon { border-left-color: #9C27B0; }
            """)
            lines.append('</style>')
        
        if collapsible:
            lines.append('<script>')
            lines.append("""
            function toggleNode(element) {
                const children = element.nextElementSibling;
                if (children) {
                    children.classList.toggle('collapsed');
                    const icon = element.querySelector('.expand-icon');
                    if (icon) {
                        icon.textContent = children.classList.contains('collapsed') ? '▶' : '▼';
                    }
                }
            }
            """)
            lines.append('</script>')
        
        lines.append('</head>')
        lines.append('<body>')
        lines.append('<div class="mindmap">')
        lines.append(f'<h1>{html.escape(root.text if hasattr(root, "text") else root.name)}</h1>')
        
        def render_node_html(node: Node, is_root: bool = False):
            """Render node as HTML"""
            if is_root:
                # Skip root node content, just render children
                if node.children:
                    lines.append('<div class="children">')
                    for child in node.children:
                        render_node_html(child)
                    lines.append('</div>')
            else:
                lines.append('<div class="node">')
                
                # Node content
                shape_class = ""
                if hasattr(node, 'shape'):
                    shape_map = {
                        NodeShape.CIRCLE: 'shape-circle',
                        NodeShape.SQUARE: 'shape-square',
                        NodeShape.HEXAGON: 'shape-hexagon'
                    }
                    shape_class = shape_map.get(node.shape, '')
                
                onclick = 'onclick="toggleNode(this)"' if collapsible and node.children else ''
                lines.append(f'<div class="node-content {shape_class}" {onclick}>')
                
                if collapsible and node.children:
                    lines.append('<span class="expand-icon">▼</span>')
                
                text = html.escape(node.text if hasattr(node, 'text') else node.name)
                lines.append(f'<span>{text}</span>')
                lines.append('</div>')
                
                # Children
                if node.children:
                    lines.append('<div class="children">')
                    for child in node.children:
                        render_node_html(child)
                    lines.append('</div>')
                
                lines.append('</div>')
        
        render_node_html(root, is_root=True)
        
        lines.append('</div>')
        lines.append('</body>')
        lines.append('</html>')
        
        return '\n'.join(lines)
    
    @staticmethod
    def to_opml(root: Node) -> str:
        """Export to OPML (Outline Processor Markup Language) format
        
        Args:
            root: Root node of the mindmap
            
        Returns:
            OPML XML string
        """
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<opml version="2.0">')
        lines.append('<head>')
        lines.append(f'<title>{html.escape(root.text if hasattr(root, "text") else root.name)}</title>')
        lines.append(f'<dateCreated>{datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")}</dateCreated>')
        lines.append('</head>')
        lines.append('<body>')
        
        def render_outline(node: Node, level: int = 0):
            """Render node as OPML outline"""
            indent = "  " * level
            text = html.escape(node.text if hasattr(node, 'text') else node.name)
            
            if node.children:
                lines.append(f'{indent}<outline text="{text}">')
                for child in node.children:
                    render_outline(child, level + 1)
                lines.append(f'{indent}</outline>')
            else:
                lines.append(f'{indent}<outline text="{text}" />')
        
        render_outline(root)
        
        lines.append('</body>')
        lines.append('</opml>')
        
        return '\n'.join(lines)
    
    @staticmethod
    def to_json_canvas(root: Node, 
                       layout: str = 'hierarchical',
                       include_metadata: bool = True) -> str:
        """Export to JSON Canvas format (Obsidian)
        
        Args:
            root: Root node of the mindmap
            layout: Layout algorithm ('hierarchical', 'radial', 'grid')
            include_metadata: Whether to include node metadata
            
        Returns:
            JSON Canvas format string
        """
        return JSONCanvasHandler.to_json_canvas(root, layout, include_metadata)
    
    @staticmethod
    def from_json_canvas(canvas_data: str) -> Node:
        """Import from JSON Canvas format
        
        Args:
            canvas_data: JSON Canvas format string
            
        Returns:
            Root node of the imported mindmap
        """
        return JSONCanvasHandler.from_json_canvas(canvas_data)
    
    @staticmethod
    def save_to_file(content: str, filepath: Path, encoding: str = 'utf-8') -> None:
        """Save exported content to file
        
        Args:
            content: Content to save
            filepath: Path to save to
            encoding: File encoding
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        
        logger.info(f"Exported mindmap to {filepath}")