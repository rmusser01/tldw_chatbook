#!/usr/bin/env python3
"""
Tests for JSON Canvas handler
"""

import json
import pytest
from anytree import Node, PreOrderIter

# Skip if mindmap dependencies not available
pytest.importorskip("anytree")

from tldw_chatbook.Tools.Mind_Map.jsoncanvas_handler import JSONCanvasHandler


class TestJSONCanvasExport:
    """Test JSON Canvas export functionality"""
    
    def test_simple_tree_export(self):
        """Test exporting a simple tree to JSON Canvas"""
        # Create a simple tree
        root = Node("root", text="Root Node")
        child1 = Node("child1", text="Child 1", parent=root)
        child2 = Node("child2", text="Child 2", parent=root)
        grandchild = Node("gc1", text="Grandchild", parent=child1)
        
        # Export to JSON Canvas
        canvas_json = JSONCanvasHandler.to_json_canvas(root, layout='hierarchical')
        canvas_data = json.loads(canvas_json)
        
        # Verify structure
        assert 'nodes' in canvas_data
        assert 'edges' in canvas_data
        assert len(canvas_data['nodes']) == 4
        assert len(canvas_data['edges']) == 3
        
        # Verify node properties
        for node in canvas_data['nodes']:
            assert 'id' in node
            assert 'type' in node
            assert 'x' in node
            assert 'y' in node
            assert 'width' in node
            assert 'height' in node
            assert 'text' in node
    
    def test_node_types(self):
        """Test different node types in export"""
        root = Node("root", text="Root", metadata={'type': 'group'})
        file_node = Node("file", text="Document", parent=root, 
                        metadata={'type': 'file', 'file_path': '/path/to/file.md'})
        link_node = Node("link", text="Website", parent=root,
                        metadata={'type': 'link', 'url': 'https://example.com'})
        
        canvas_json = JSONCanvasHandler.to_json_canvas(root)
        canvas_data = json.loads(canvas_json)
        
        # Find nodes by type
        nodes_by_id = {n['id']: n for n in canvas_data['nodes']}
        
        # Root should be group type
        assert nodes_by_id['root']['type'] == 'group'
        
        # File node
        assert nodes_by_id['file']['type'] == 'file'
        assert 'file' in nodes_by_id['file']
        
        # Link node  
        assert nodes_by_id['link']['type'] == 'link'
        assert 'url' in nodes_by_id['link']
    
    def test_color_mapping(self):
        """Test color conversion to JSON Canvas format"""
        root = Node("root", text="Root", color="red")
        child = Node("child", text="Child", parent=root, color="green")
        
        canvas_json = JSONCanvasHandler.to_json_canvas(root)
        canvas_data = json.loads(canvas_json)
        
        nodes_by_id = {n['id']: n for n in canvas_data['nodes']}
        
        # Check color mapping
        assert nodes_by_id['root']['color'] == '1'  # red -> 1
        assert nodes_by_id['child']['color'] == '4'  # green -> 4
    
    def test_layout_algorithms(self):
        """Test different layout algorithms"""
        root = Node("root", text="Root")
        Node("c1", text="Child 1", parent=root)
        Node("c2", text="Child 2", parent=root)
        
        # Test hierarchical layout
        hier_json = JSONCanvasHandler.to_json_canvas(root, layout='hierarchical')
        hier_data = json.loads(hier_json)
        
        # Test radial layout
        radial_json = JSONCanvasHandler.to_json_canvas(root, layout='radial')
        radial_data = json.loads(radial_json)
        
        # Test grid layout
        grid_json = JSONCanvasHandler.to_json_canvas(root, layout='grid')
        grid_data = json.loads(grid_json)
        
        # All should have same number of nodes/edges
        assert len(hier_data['nodes']) == len(radial_data['nodes']) == len(grid_data['nodes'])
        assert len(hier_data['edges']) == len(radial_data['edges']) == len(grid_data['edges'])
        
        # But positions should differ
        hier_positions = {n['id']: (n['x'], n['y']) for n in hier_data['nodes']}
        radial_positions = {n['id']: (n['x'], n['y']) for n in radial_data['nodes']}
        grid_positions = {n['id']: (n['x'], n['y']) for n in grid_data['nodes']}
        
        # At least some positions should differ between layouts
        assert hier_positions != radial_positions
        assert hier_positions != grid_positions


class TestJSONCanvasImport:
    """Test JSON Canvas import functionality"""
    
    def test_simple_import(self):
        """Test importing a simple JSON Canvas"""
        canvas_data = {
            "nodes": [
                {"id": "1", "type": "text", "text": "Root", "x": 0, "y": 0, "width": 250, "height": 60},
                {"id": "2", "type": "text", "text": "Child 1", "x": -150, "y": 100, "width": 250, "height": 60},
                {"id": "3", "type": "text", "text": "Child 2", "x": 150, "y": 100, "width": 250, "height": 60}
            ],
            "edges": [
                {"id": "e1", "fromNode": "1", "toNode": "2"},
                {"id": "e2", "fromNode": "1", "toNode": "3"}
            ]
        }
        
        root = JSONCanvasHandler.from_json_canvas(json.dumps(canvas_data))
        
        # Verify tree structure
        assert root.text == "Root"
        assert len(root.children) == 2
        
        children_texts = [c.text for c in root.children]
        assert "Child 1" in children_texts
        assert "Child 2" in children_texts
    
    def test_import_node_types(self):
        """Test importing different node types"""
        canvas_data = {
            "nodes": [
                {"id": "1", "type": "group", "label": "Group", "x": 0, "y": 0, "width": 300, "height": 200},
                {"id": "2", "type": "file", "file": "/path/to/file.md", "x": 50, "y": 50, "width": 200, "height": 60},
                {"id": "3", "type": "link", "url": "https://example.com", "x": 50, "y": 120, "width": 200, "height": 60}
            ],
            "edges": [
                {"id": "e1", "fromNode": "1", "toNode": "2"},
                {"id": "e2", "fromNode": "1", "toNode": "3"}
            ]
        }
        
        root = JSONCanvasHandler.from_json_canvas(json.dumps(canvas_data))
        
        # Check root
        assert root.text == "Group"
        assert root.canvas_type == "group"
        
        # Check children
        for child in root.children:
            if child.canvas_type == "file":
                assert "📄" in child.text
                assert hasattr(child, 'file_path')
            elif child.canvas_type == "link":
                assert "🔗" in child.text
                assert hasattr(child, 'url')
    
    def test_import_with_colors(self):
        """Test importing nodes with colors"""
        canvas_data = {
            "nodes": [
                {"id": "1", "type": "text", "text": "Red", "x": 0, "y": 0, "width": 250, "height": 60, "color": "1"},
                {"id": "2", "type": "text", "text": "Green", "x": 0, "y": 100, "width": 250, "height": 60, "color": "4"}
            ],
            "edges": [
                {"id": "e1", "fromNode": "1", "toNode": "2"}
            ]
        }
        
        root = JSONCanvasHandler.from_json_canvas(json.dumps(canvas_data))
        
        assert root.color == "1"
        assert root.children[0].color == "4"
    
    def test_import_orphan_nodes(self):
        """Test importing with orphan nodes (multiple roots)"""
        canvas_data = {
            "nodes": [
                {"id": "1", "type": "text", "text": "Node 1", "x": 0, "y": 0, "width": 250, "height": 60},
                {"id": "2", "type": "text", "text": "Node 2", "x": 300, "y": 0, "width": 250, "height": 60},
                {"id": "3", "type": "text", "text": "Child", "x": 0, "y": 100, "width": 250, "height": 60}
            ],
            "edges": [
                {"id": "e1", "fromNode": "1", "toNode": "3"}
            ]
        }
        
        root = JSONCanvasHandler.from_json_canvas(json.dumps(canvas_data))
        
        # Should create synthetic root for multiple root nodes
        if root.name == "root":
            assert len(root.children) == 2
        else:
            # Or use first node as root
            assert root.text in ["Node 1", "Node 2"]


class TestJSONCanvasValidation:
    """Test JSON Canvas validation"""
    
    def test_valid_canvas(self):
        """Test validation of valid canvas"""
        canvas_data = {
            "nodes": [
                {"id": "1", "type": "text", "text": "Node", "x": 0, "y": 0, "width": 250, "height": 60}
            ],
            "edges": []
        }
        
        is_valid, errors = JSONCanvasHandler.validate_canvas(json.dumps(canvas_data))
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_json(self):
        """Test validation with invalid JSON"""
        is_valid, errors = JSONCanvasHandler.validate_canvas("not valid json")
        assert not is_valid
        assert len(errors) > 0
        assert "Invalid JSON" in errors[0]
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        canvas_data = {
            "nodes": [
                {"type": "text", "text": "Node"}  # Missing id, x, y, width, height
            ]
        }
        
        is_valid, errors = JSONCanvasHandler.validate_canvas(json.dumps(canvas_data))
        assert not is_valid
        assert any("missing required field 'id'" in e for e in errors)
        assert any("missing required field 'x'" in e for e in errors)
    
    def test_invalid_node_type(self):
        """Test validation with invalid node type"""
        canvas_data = {
            "nodes": [
                {"id": "1", "type": "invalid", "x": 0, "y": 0, "width": 250, "height": 60}
            ]
        }
        
        is_valid, errors = JSONCanvasHandler.validate_canvas(json.dumps(canvas_data))
        assert not is_valid
        assert any("invalid type" in e for e in errors)
    
    def test_invalid_edge_references(self):
        """Test validation with edges referencing non-existent nodes"""
        canvas_data = {
            "nodes": [
                {"id": "1", "type": "text", "text": "Node", "x": 0, "y": 0, "width": 250, "height": 60}
            ],
            "edges": [
                {"id": "e1", "fromNode": "1", "toNode": "999"}  # Node 999 doesn't exist
            ]
        }
        
        is_valid, errors = JSONCanvasHandler.validate_canvas(json.dumps(canvas_data))
        assert not is_valid
        assert any("does not exist" in e for e in errors)
    
    def test_invalid_color(self):
        """Test validation with invalid color"""
        canvas_data = {
            "nodes": [
                {"id": "1", "type": "text", "text": "Node", "x": 0, "y": 0, 
                 "width": 250, "height": 60, "color": "invalid"}
            ]
        }
        
        is_valid, errors = JSONCanvasHandler.validate_canvas(json.dumps(canvas_data))
        assert not is_valid
        assert any("invalid color" in e for e in errors)


class TestRoundTrip:
    """Test round-trip conversion (export then import)"""
    
    def test_round_trip_preservation(self):
        """Test that export->import preserves structure"""
        # Create original tree
        root = Node("root", text="Root Node")
        c1 = Node("c1", text="Child 1", parent=root)
        c2 = Node("c2", text="Child 2", parent=root)
        gc1 = Node("gc1", text="Grandchild 1", parent=c1)
        gc2 = Node("gc2", text="Grandchild 2", parent=c1)
        
        # Export to JSON Canvas
        canvas_json = JSONCanvasHandler.to_json_canvas(root)
        
        # Import back
        imported_root = JSONCanvasHandler.from_json_canvas(canvas_json)
        
        # Compare structures
        original_texts = [(n.text, n.depth) for n in PreOrderIter(root)]
        imported_texts = [(n.text, n.depth) for n in PreOrderIter(imported_root)]
        
        assert len(original_texts) == len(imported_texts)
        assert original_texts == imported_texts
    
    def test_round_trip_with_metadata(self):
        """Test that metadata is preserved in round trip"""
        root = Node("root", text="Root", 
                   metadata={'type': 'group', 'custom': 'value'})
        child = Node("child", text="Child", parent=root,
                    metadata={'type': 'file', 'file_path': '/test.md'})
        
        # Export and re-import
        canvas_json = JSONCanvasHandler.to_json_canvas(root, include_metadata=True)
        imported_root = JSONCanvasHandler.from_json_canvas(canvas_json)
        
        # Check metadata preservation
        assert imported_root.metadata['type'] == 'group'
        assert imported_root.children[0].metadata['type'] == 'file'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])