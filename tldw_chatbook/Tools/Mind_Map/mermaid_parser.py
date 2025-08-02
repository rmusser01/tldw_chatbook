# mermaid_parser.py
# Description: Parser for Mermaid mindmap syntax
#
"""
Mermaid Mindmap Parser
---------------------

Parses Mermaid mindmap syntax into tree structures using anytree.

Supports:
- All Mermaid node shapes: ((circle)), [square], (rounded), {hexagon}, {{cloud}}
- Node IDs and hierarchical relationships
- Icons and CSS classes
- Markdown formatting in node text
- Error handling with line numbers
"""

from typing import List, Dict, Optional, Any
import re
from anytree import Node
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from rich.text import Text


class NodeShape(Enum):
    """Mermaid node shapes mapped to Unicode symbols"""
    CIRCLE = "○"
    DOUBLE_CIRCLE = "◉"
    SQUARE = "□"
    ROUNDED = "▢"
    HEXAGON = "⬡"
    CLOUD = "☁"
    DEFAULT = "•"


@dataclass
class MindmapNode:
    """Represents a parsed mindmap node"""
    id: str
    text: str
    shape: NodeShape
    level: int
    children: List['MindmapNode'] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}


class MermaidMindmapParser:
    """Parse Mermaid mindmap syntax into tree structure"""
    
    # Regex patterns for different node types with improved handling
    PATTERNS = {
        'double_circle': r'(\w+)\(\((.+?)\)\)(?!\))',  # id((text))
        'square': r'(\w+)\[([^\[\]]+)\]',              # id[text]
        'rounded': r'(\w+)\(([^()]+|(?:\([^()]*\))*)\)(?!\))',  # id(text) with nested parens
        'hexagon': r'(\w+)\{([^{}]+)\}(?!\})',         # id{text}
        'cloud': r'(\w+)\{\{(.+?)\}\}',                # id{{text}}
    }
    
    SHAPE_MAP = {
        'double_circle': NodeShape.DOUBLE_CIRCLE,
        'square': NodeShape.SQUARE,
        'rounded': NodeShape.ROUNDED,
        'hexagon': NodeShape.HEXAGON,
        'cloud': NodeShape.CLOUD,
    }
    
    def parse(self, mermaid_code: str) -> Node:
        """Parse Mermaid mindmap code into anytree structure
        
        Args:
            mermaid_code: Mermaid mindmap syntax string
            
        Returns:
            Root node of the parsed tree
            
        Raises:
            ValueError: If the Mermaid syntax is invalid
        """
        lines = mermaid_code.strip().split('\n')
        root_node = None
        node_stack = []  # Stack of (node, indent_level)
        line_number = 0
        
        try:
            for line_number, line in enumerate(lines, 1):
                # Skip empty lines and the 'mindmap' declaration
                if not line.strip() or line.strip() == 'mindmap':
                    continue
                
                # Calculate indentation level
                indent_level = self._get_indent_level(line)
                
                # Parse node
                node_info = self._parse_node_line(line.strip())
                if not node_info:
                    logger.warning(f"Line {line_number}: Could not parse '{line.strip()}'")
                    continue
                
                # Create anytree node
                if root_node is None:
                    root_node = Node(
                        node_info['id'],
                        text=node_info['text'],
                        shape=node_info['shape'],
                        level=0,
                        icon=node_info.get('icon'),
                        css_class=node_info.get('css_class'),
                        formatted_text=node_info.get('formatted_text')
                    )
                    node_stack = [(root_node, indent_level)]
                else:
                    # Find parent based on indentation
                    parent_node = self._find_parent_node(node_stack, indent_level)
                    
                    new_node = Node(
                        node_info['id'],
                        parent=parent_node,
                        text=node_info['text'],
                        shape=node_info['shape'],
                        level=indent_level,
                        icon=node_info.get('icon'),
                        css_class=node_info.get('css_class'),
                        formatted_text=node_info.get('formatted_text')
                    )
                    
                    # Update stack
                    node_stack = [(n, l) for n, l in node_stack if l < indent_level]
                    node_stack.append((new_node, indent_level))
        
        except Exception as e:
            raise ValueError(f"Error parsing Mermaid mindmap at line {line_number}: {str(e)}")
        
        if not root_node:
            raise ValueError("No valid mindmap structure found")
            
        return root_node
    
    def _get_indent_level(self, line: str) -> int:
        """Calculate indentation level (2 spaces = 1 level)"""
        return (len(line) - len(line.lstrip())) // 2
    
    def _parse_node_line(self, line: str) -> Optional[Dict]:
        """Parse a single node line"""
        for shape_type, pattern in self.PATTERNS.items():
            match = re.match(pattern, line)
            if match:
                return {
                    'id': match.group(1),
                    'text': match.group(2),
                    'shape': self.SHAPE_MAP[shape_type]
                }
        
        # Default format: just text
        if line:
            return {
                'id': line.replace(' ', '_'),
                'text': line,
                'shape': NodeShape.DEFAULT
            }
        
        return None
    
    def _find_parent_node(self, node_stack: List[tuple], 
                          indent_level: int) -> Node:
        """Find the appropriate parent node based on indentation"""
        for node, level in reversed(node_stack):
            if level < indent_level:
                return node
        return node_stack[0][0]  # Default to root


class ExtendedMermaidParser(MermaidMindmapParser):
    """Extended parser with support for icons, classes, and markdown"""
    
    # Additional patterns for advanced features
    EXTENDED_PATTERNS = {
        'icon': r'::icon\(([^)]+)\)',           # ::icon(fa fa-star)
        'class': r':::(\w+)',                   # :::className
        'markdown_bold': r'\*\*(.+?)\*\*',      # **bold**
        'markdown_italic': r'\*(.+?)\*',        # *italic*
        'markdown_code': r'`(.+?)`',            # `code`
    }
    
    def _parse_node_line(self, line: str) -> Optional[Dict]:
        """Parse node with extended features"""
        # First try standard parsing
        node_info = super()._parse_node_line(line)
        if not node_info:
            return None
        
        # Extract icon if present
        icon_match = re.search(self.EXTENDED_PATTERNS['icon'], line)
        if icon_match:
            node_info['icon'] = icon_match.group(1)
            # Remove icon syntax from text
            node_info['text'] = re.sub(self.EXTENDED_PATTERNS['icon'], '', node_info['text']).strip()
        
        # Extract class if present
        class_match = re.search(self.EXTENDED_PATTERNS['class'], line)
        if class_match:
            node_info['css_class'] = class_match.group(1)
            node_info['text'] = re.sub(self.EXTENDED_PATTERNS['class'], '', node_info['text']).strip()
        
        # Process markdown in text
        node_info['formatted_text'] = self._process_markdown(node_info['text'])
        
        return node_info
    
    def _process_markdown(self, text: str) -> Text:
        """Convert markdown to Rich Text object"""
        result = Text()
        remaining = text
        
        while remaining:
            # Find the next markdown element
            bold_match = re.search(self.EXTENDED_PATTERNS['markdown_bold'], remaining)
            italic_match = re.search(self.EXTENDED_PATTERNS['markdown_italic'], remaining)
            code_match = re.search(self.EXTENDED_PATTERNS['markdown_code'], remaining)
            
            # Find which comes first
            matches = []
            if bold_match:
                matches.append(('bold', bold_match))
            if italic_match:
                matches.append(('italic', italic_match))
            if code_match:
                matches.append(('code', code_match))
            
            if not matches:
                # No more markdown, append the rest
                result.append(remaining)
                break
            
            # Sort by position
            matches.sort(key=lambda x: x[1].start())
            style_type, match = matches[0]
            
            # Append text before the match
            if match.start() > 0:
                result.append(remaining[:match.start()])
            
            # Append styled text
            if style_type == 'bold':
                result.append(match.group(1), style="bold")
            elif style_type == 'italic':
                result.append(match.group(1), style="italic")
            elif style_type == 'code':
                result.append(match.group(1), style="bold cyan on grey23")
            
            # Continue with remaining text
            remaining = remaining[match.end():]
        
        return result