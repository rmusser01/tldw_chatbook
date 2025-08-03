"""
Unit tests for repository tree widgets.

Tests the TreeNode and TreeView widgets in isolation with mocked dependencies.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import os

from tldw_chatbook.Widgets.repo_tree_widgets import (
    TreeNode, TreeView, TreeNodeSelected, TreeNodeExpanded
)
from textual.widgets import Button, Checkbox, Static
from textual.containers import Container
from Tests.textual_test_utils import widget_pilot, wait_for_widget_mount


class TestTreeNode:
    """Test suite for TreeNode widget."""
    
    @pytest.fixture
    def sample_file_data(self):
        """Sample file data for testing."""
        return {
            'path': 'src/main.py',
            'name': 'main.py',
            'is_directory': False,
            'level': 1,
            'size': 1024
        }
    
    @pytest.fixture
    def sample_dir_data(self):
        """Sample directory data for testing."""
        return {
            'path': 'src/components',
            'name': 'components',
            'is_directory': True,
            'level': 1,
            'size': None
        }
    
    def test_tree_node_creation_file(self, sample_file_data):
        """Test creating a tree node for a file."""
        node = TreeNode(**sample_file_data)
        
        assert node.path == 'src/main.py'
        assert node.node_name == 'main.py'
        assert not node.is_directory
        assert node.level == 1
        assert node.file_size == 1024
        # reactive values need to be accessed differently in tests
        assert node.expanded == False
        assert node.selected == False
        assert not node.children_loaded
    
    def test_tree_node_creation_directory(self, sample_dir_data):
        """Test creating a tree node for a directory."""
        node = TreeNode(**sample_dir_data)
        
        assert node.path == 'src/components'
        assert node.node_name == 'components'
        assert node.is_directory
        assert node.level == 1
        assert node.file_size is None
        # reactive values need to be accessed differently in tests
        assert node.expanded == False
        assert node.selected == False
        assert not node.children_loaded
    
    @pytest.mark.parametrize("filename,expected_icon", [
        ("script.py", "🐍"),
        ("app.js", "📜"),
        ("component.jsx", "⚛️"),
        ("types.ts", "📘"),
        ("App.tsx", "⚛️"),
        ("README.md", "📝"),
        ("config.json", "📊"),
        ("docker-compose.yaml", "⚙️"),
        ("settings.yml", "⚙️"),
        ("notes.txt", "📄"),
        ("index.html", "🌐"),
        ("styles.css", "🎨"),
        ("image.png", "🖼️"),
        ("photo.jpg", "🖼️"),
        ("document.pdf", "📑"),
        ("archive.zip", "📦"),
        (".gitignore", "🚫"),
        (".env", "🔐"),
        ("unknown.xyz", "📄"),  # Default icon
    ])
    def test_file_icon_mapping(self, filename, expected_icon):
        """Test that files get the correct icons based on extension."""
        node = TreeNode(
            path=f"test/{filename}",
            name=filename,
            is_directory=False,
            level=0
        )
        
        assert node._get_icon() == expected_icon
    
    def test_directory_icon_states(self):
        """Test directory icons change based on expanded state."""
        node = TreeNode(
            path="src",
            name="src",
            is_directory=True,
            level=0
        )
        
        # Collapsed state
        assert node._get_icon() == "📁"
        
        # Expanded state
        node.expanded = True
        assert node._get_icon() == "📂"
    
    @pytest.mark.parametrize("size,expected", [
        (0, "0.0 B"),
        (512, "512.0 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1048576, "1.0 MB"),
        (1073741824, "1.0 GB"),
        (1099511627776, "1.0 TB"),
        (None, ""),
    ])
    def test_file_size_formatting(self, size, expected):
        """Test human-readable file size formatting."""
        node = TreeNode(
            path="test.txt",
            name="test.txt",
            is_directory=False,
            level=0,
            size=size
        )
        
        assert node._format_size(size) == expected
    
    @pytest.mark.asyncio
    async def test_tree_node_compose(self, widget_pilot):
        """Test tree node composition and structure."""
        async with widget_pilot(
            TreeNode,
            path="src/main.py",
            name="main.py",
            is_directory=False,
            level=1,
            size=2048
        ) as pilot:
            node = pilot.app.test_widget
            
            # Check that all components are present
            containers = node.query(Container)
            assert len(containers) > 0
            
            # Check for indent static
            indent_widgets = node.query(".tree-indent")
            assert len(indent_widgets) == 1
            assert indent_widgets[0].renderable == "  "  # 2 spaces for level 1
            
            # Check for checkbox
            checkboxes = node.query(Checkbox)
            assert len(checkboxes) == 1
            assert checkboxes[0].id == "select-src/main.py"
            
            # Check content
            content_widgets = node.query(".tree-content")
            assert len(content_widgets) == 1
            assert "🐍 main.py (2.0 KB)" in str(content_widgets[0].renderable)
    
    @pytest.mark.asyncio
    async def test_directory_expand_button(self, widget_pilot):
        """Test that directories have expand/collapse button."""
        async with widget_pilot(
            TreeNode,
            path="src",
            name="src",
            is_directory=True,
            level=0
        ) as pilot:
            node = pilot.app.test_widget
            
            # Check for expand button
            buttons = node.query(Button)
            assert len(buttons) == 1
            assert buttons[0].label == "▶"
            assert buttons[0].id == "expand-src"
    
    @pytest.mark.asyncio
    async def test_file_no_expand_button(self, widget_pilot):
        """Test that files don't have expand button."""
        async with widget_pilot(
            TreeNode,
            path="test.py",
            name="test.py",
            is_directory=False,
            level=0
        ) as pilot:
            node = pilot.app.test_widget
            
            # Check for spacer instead of button
            buttons = node.query(Button)
            assert len(buttons) == 0
            
            spacers = node.query(".tree-expand-spacer")
            assert len(spacers) == 1
    
    @pytest.mark.asyncio
    async def test_expand_button_click(self, widget_pilot):
        """Test clicking expand button toggles state."""
        events = []
        
        async with widget_pilot(
            TreeNode,
            path="src",
            name="src",
            is_directory=True,
            level=0
        ) as pilot:
            node = pilot.app.test_widget
            
            # Set up event capture
            def capture_event(event):
                events.append(event)
            
            node.post_message = MagicMock(side_effect=capture_event)
            
            # Click expand button
            button = node.query_one(".tree-expand-btn", Button)
            await pilot.click(button)
            await pilot.pause()
            
            # Check state changed
            assert node.expanded == True
            assert button.label == "▼"
            
            # Check event was posted
            assert len(events) == 1
            assert isinstance(events[0], TreeNodeExpanded)
            assert events[0].path == "src"
            assert events[0].expanded is True
    
    @pytest.mark.asyncio
    async def test_checkbox_selection(self, widget_pilot):
        """Test checkbox selection posts event."""
        events = []
        
        async with widget_pilot(
            TreeNode,
            path="test.py",
            name="test.py",
            is_directory=False,
            level=0
        ) as pilot:
            node = pilot.app.test_widget
            
            # Set up event capture
            def capture_event(event):
                events.append(event)
            
            node.post_message = MagicMock(side_effect=capture_event)
            
            # Click checkbox
            checkbox = node.query_one(".tree-checkbox", Checkbox)
            checkbox.value = True
            await checkbox._on_toggle()  # Simulate change event
            await pilot.pause()
            
            # Check state
            assert node.selected == True
            
            # Check event was posted
            assert len(events) == 1
            assert isinstance(events[0], TreeNodeSelected)
            assert events[0].path == "test.py"
            assert events[0].selected is True
    
    @pytest.mark.asyncio
    async def test_visual_selection_state(self, widget_pilot):
        """Test visual feedback for selected nodes."""
        async with widget_pilot(
            TreeNode,
            path="test.py",
            name="test.py",
            is_directory=False,
            level=0
        ) as pilot:
            node = pilot.app.test_widget
            
            # Initially not selected
            container = node.query_one(".tree-node-row", Container)
            assert not container.has_class("tree-node-selected")
            
            # Select the node
            checkbox = node.query_one(".tree-checkbox", Checkbox) 
            checkbox.value = True
            await checkbox._on_toggle()
            await pilot.pause()
            
            # Should have selection class
            assert container.has_class("tree-node-selected")


class TestTreeView:
    """Test suite for TreeView widget."""
    
    @pytest.fixture
    def sample_tree_data(self):
        """Sample tree data structure."""
        return [
            {
                'path': 'README.md',
                'name': 'README.md',
                'type': 'blob',
                'size': 1234
            },
            {
                'path': 'src',
                'name': 'src',
                'type': 'tree',
                'children': [
                    {
                        'path': 'src/main.py',
                        'name': 'main.py',
                        'type': 'blob',
                        'size': 2048
                    },
                    {
                        'path': 'src/utils.py',
                        'name': 'utils.py',
                        'type': 'blob',
                        'size': 1024
                    }
                ]
            }
        ]
    
    def test_tree_view_creation(self):
        """Test TreeView initialization."""
        selection_changes = []
        expansion_changes = []
        
        def on_selection(path, selected):
            selection_changes.append((path, selected))
        
        def on_expansion(path, expanded):
            expansion_changes.append((path, expanded))
        
        tree = TreeView(
            on_selection_change=on_selection,
            on_node_expanded=on_expansion
        )
        
        assert tree.nodes == {}
        assert tree.selection == set()
        assert tree.on_selection_change is on_selection
        assert tree.on_node_expanded is on_expansion
        assert tree._tree_data is None
    
    @pytest.mark.asyncio
    async def test_load_empty_tree(self, widget_pilot):
        """Test loading empty tree shows appropriate message."""
        async with widget_pilot(TreeView) as pilot:
            tree = pilot.app.test_widget
            
            # Load empty tree
            await tree.load_tree([])
            await pilot.pause()
            
            # Should show empty message
            empty_msgs = tree.query(".tree-empty")
            assert len(empty_msgs) == 1
            assert "No files found" in str(empty_msgs[0].renderable)
    
    @pytest.mark.asyncio
    async def test_load_tree_with_data(self, widget_pilot, sample_tree_data):
        """Test loading tree with data creates nodes."""
        async with widget_pilot(TreeView) as pilot:
            tree = pilot.app.test_widget
            
            # Load tree data
            await tree.load_tree(sample_tree_data)
            await pilot.pause()
            
            # Check nodes were created
            assert len(tree.nodes) == 2
            assert 'README.md' in tree.nodes
            assert 'src' in tree.nodes
            
            # Check node properties
            readme_node = tree.nodes['README.md']
            assert readme_node.node_name == 'README.md'
            assert not readme_node.is_directory
            assert readme_node.file_size == 1234
            assert readme_node.level == 0
            
            src_node = tree.nodes['src']
            assert src_node.node_name == 'src'
            assert src_node.is_directory
            assert src_node.level == 0
    
    def test_select_node_file(self):
        """Test selecting a file node."""
        tree = TreeView()
        
        # Create mock nodes
        file_node = MagicMock()
        file_node.is_directory = False
        file_node.selected = False
        
        tree.nodes = {'test.py': file_node}
        tree.selection = set()
        
        # Select the file
        tree.select_node('test.py', True)
        
        # Check selection
        assert 'test.py' in tree.selection
        assert file_node.selected == True
        
        # Deselect
        tree.select_node('test.py', False)
        assert 'test.py' not in tree.selection
        assert file_node.selected == False
    
    def test_select_node_directory_cascades(self):
        """Test selecting directory cascades to children."""
        tree = TreeView()
        
        # Create mock nodes
        dir_node = MagicMock()
        dir_node.is_directory = True
        dir_node.selected = False
        
        child1 = MagicMock()
        child1.is_directory = False
        child1.selected = False
        
        child2 = MagicMock()
        child2.is_directory = False
        child2.selected = False
        
        tree.nodes = {
            'src': dir_node,
            'src/file1.py': child1,
            'src/file2.py': child2
        }
        tree.selection = set()
        
        # Select directory
        tree.select_node('src', True)
        
        # Check all are selected
        assert tree.selection == {'src', 'src/file1.py', 'src/file2.py'}
        assert dir_node.selected == True
        assert child1.selected == True
        assert child2.selected == True
        
        # Deselect directory
        tree.select_node('src', False)
        
        # Check all are deselected
        assert tree.selection == set()
        assert dir_node.selected == False
        assert child1.selected == False
        assert child2.selected == False
    
    def test_get_selected_files(self):
        """Test getting only selected files (not directories)."""
        tree = TreeView()
        
        # Create mock nodes
        dir_node = MagicMock()
        dir_node.is_directory = True
        
        file1 = MagicMock()
        file1.is_directory = False
        
        file2 = MagicMock()
        file2.is_directory = False
        
        tree.nodes = {
            'src': dir_node,
            'src/file1.py': file1,
            'src/file2.py': file2
        }
        tree.selection = {'src', 'src/file1.py', 'src/file2.py'}
        
        # Get selected files
        files = tree.get_selected_files()
        
        # Should only return files, not directories
        assert set(files) == {'src/file1.py', 'src/file2.py'}
        assert 'src' not in files
    
    def test_get_selection_stats(self):
        """Test getting selection statistics."""
        tree = TreeView()
        
        # Create mock nodes
        dir_node = MagicMock()
        dir_node.is_directory = True
        
        file1 = MagicMock()
        file1.is_directory = False
        file1.file_size = 1024
        
        file2 = MagicMock()
        file2.is_directory = False
        file2.file_size = 2048
        
        file3 = MagicMock()
        file3.is_directory = False
        file3.file_size = None  # No size info
        
        tree.nodes = {
            'src': dir_node,
            'src/file1.py': file1,
            'src/file2.py': file2,
            'src/file3.py': file3
        }
        tree.selection = {'src', 'src/file1.py', 'src/file2.py', 'src/file3.py'}
        
        # Get stats
        stats = tree.get_selection_stats()
        
        assert stats['files'] == 3  # Only files, not directories
        assert stats['size'] == 3072  # 1024 + 2048
    
    @pytest.mark.asyncio
    async def test_expand_node(self, widget_pilot):
        """Test expanding a node adds children."""
        async with widget_pilot(TreeView) as pilot:
            tree = pilot.app.test_widget
            
            # Create parent node
            parent = TreeNode(
                path='src',
                name='src',
                is_directory=True,
                level=0
            )
            parent.children_loaded = False
            
            container = tree.query_one("#tree-container", Container)
            await container.mount(parent)
            tree.nodes['src'] = parent
            
            # Expand with children
            children = [
                {'path': 'src/file1.py', 'name': 'file1.py', 'type': 'blob', 'size': 100},
                {'path': 'src/file2.py', 'name': 'file2.py', 'type': 'blob', 'size': 200}
            ]
            
            await tree.expand_node('src', children)
            await pilot.pause()
            
            # Check children were added
            assert 'src/file1.py' in tree.nodes
            assert 'src/file2.py' in tree.nodes
            assert tree.nodes['src/file1.py'].level == 1
            assert tree.nodes['src/file2.py'].level == 1
            assert parent.children_loaded
    
    @pytest.mark.asyncio
    async def test_collapse_node(self, widget_pilot):
        """Test collapsing a node removes children."""
        async with widget_pilot(TreeView) as pilot:
            tree = pilot.app.test_widget
            
            # Set up nodes
            parent = MagicMock()
            parent.is_directory = True
            parent.path = 'src'
            
            child1 = MagicMock()
            child1.path = 'src/file1.py'
            child1.remove = AsyncMock()
            
            child2 = MagicMock()
            child2.path = 'src/file2.py'
            child2.remove = AsyncMock()
            
            tree.nodes = {
                'src': parent,
                'src/file1.py': child1,
                'src/file2.py': child2
            }
            tree.selection = {'src/file1.py', 'src/file2.py'}
            
            # Collapse
            await tree.collapse_node('src')
            
            # Check children removed
            assert 'src/file1.py' not in tree.nodes
            assert 'src/file2.py' not in tree.nodes
            assert 'src/file1.py' not in tree.selection
            assert 'src/file2.py' not in tree.selection
            
            # Parent should remain
            assert 'src' in tree.nodes
    
    @pytest.mark.asyncio
    async def test_event_handling(self, widget_pilot):
        """Test TreeView handles events from nodes."""
        selection_changes = []
        expansion_changes = []
        
        def on_selection(path, selected):
            selection_changes.append((path, selected))
        
        def on_expansion(path, expanded):
            expansion_changes.append((path, expanded))
        
        async with widget_pilot(
            TreeView,
            on_selection_change=on_selection,
            on_node_expanded=on_expansion
        ) as pilot:
            tree = pilot.app.test_widget
            
            # Post selection event
            tree.post_message(TreeNodeSelected('test.py', True))
            await pilot.pause()
            
            # Check callback was called
            assert selection_changes == [('test.py', True)]
            
            # Post expansion event
            tree.post_message(TreeNodeExpanded('src', True))
            await pilot.pause()
            
            # Check callback was called
            assert expansion_changes == [('src', True)]