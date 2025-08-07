"""
Simplified unit tests for repository tree widgets.

Tests basic functionality without full Textual app context.
"""

from unittest.mock import MagicMock

from tldw_chatbook.Widgets.Coding_Widgets.repo_tree_widgets import (
    TreeNode, TreeView, TreeNodeSelected, TreeNodeExpanded
)


class TestTreeNodeSimple:
    """Simple tests for TreeNode functionality."""
    
    def test_format_size_method(self):
        """Test the _format_size method directly."""
        node = MagicMock()
        node._format_size = TreeNode._format_size.__get__(node)
        
        assert node._format_size(None) == ""
        assert node._format_size(0) == "0.0 B"
        assert node._format_size(512) == "512.0 B"
        assert node._format_size(1024) == "1.0 KB"
        assert node._format_size(1536) == "1.5 KB"
        assert node._format_size(1048576) == "1.0 MB"
        assert node._format_size(1073741824) == "1.0 GB"
        assert node._format_size(1099511627776) == "1.0 TB"
    
    def test_get_icon_method(self):
        """Test the _get_icon method for file types."""
        # Test file icons
        node = MagicMock()
        node._get_icon = TreeNode._get_icon.__get__(node)
        node.is_directory = False
        
        node.node_name = "script.py"
        assert node._get_icon() == "🐍"
        
        node.node_name = "app.js"
        assert node._get_icon() == "📜"
        
        node.node_name = "README.md"
        assert node._get_icon() == "📝"
        
        node.node_name = ".gitignore"
        assert node._get_icon() == "🚫"
        
        node.node_name = "unknown.xyz"
        assert node._get_icon() == "📄"
        
        # Test directory icons
        node.is_directory = True
        node.expanded = False
        assert node._get_icon() == "📁"
        
        node.expanded = True
        assert node._get_icon() == "📂"


class TestTreeViewSimple:
    """Simple tests for TreeView functionality."""
    
    def test_get_selected_files(self):
        """Test filtering selected files."""
        tree = TreeView()
        
        # Create mock nodes
        file1 = MagicMock()
        file1.is_directory = False
        
        file2 = MagicMock()
        file2.is_directory = False
        
        dir1 = MagicMock()
        dir1.is_directory = True
        
        tree.nodes = {
            'file1.py': file1,
            'file2.py': file2,
            'src': dir1
        }
        tree.selection = {'file1.py', 'file2.py', 'src'}
        
        # Should only return files
        files = tree.get_selected_files()
        assert set(files) == {'file1.py', 'file2.py'}
        assert 'src' not in files
    
    def test_get_selection_stats(self):
        """Test selection statistics calculation."""
        tree = TreeView()
        
        # Create mock nodes
        file1 = MagicMock()
        file1.is_directory = False
        file1.file_size = 1024
        
        file2 = MagicMock()
        file2.is_directory = False
        file2.file_size = 2048
        
        file3 = MagicMock()
        file3.is_directory = False
        file3.file_size = None
        
        dir1 = MagicMock()
        dir1.is_directory = True
        
        tree.nodes = {
            'file1.py': file1,
            'file2.py': file2,
            'file3.py': file3,
            'src': dir1
        }
        tree.selection = {'file1.py', 'file2.py', 'file3.py', 'src'}
        
        stats = tree.get_selection_stats()
        assert stats['files'] == 3
        assert stats['size'] == 3072
    
    def test_select_node_cascading(self):
        """Test that selecting a directory cascades to children."""
        tree = TreeView()
        
        # Create simple mock nodes that track their selected state
        class MockNode:
            def __init__(self, is_directory):
                self.is_directory = is_directory
                self.selected = False
        
        dir_node = MockNode(True)
        child1 = MockNode(False)
        child2 = MockNode(False)
        
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


class TestMessages:
    """Test custom message classes."""
    
    def test_tree_node_selected_message(self):
        """Test TreeNodeSelected message."""
        msg = TreeNodeSelected('test/path', True)
        assert msg.path == 'test/path'
        assert msg.selected is True
        
        msg2 = TreeNodeSelected('other/path', False)
        assert msg2.path == 'other/path'
        assert msg2.selected is False
    
    def test_tree_node_expanded_message(self):
        """Test TreeNodeExpanded message."""
        msg = TreeNodeExpanded('src/components', True)
        assert msg.path == 'src/components'
        assert msg.expanded is True
        
        msg2 = TreeNodeExpanded('src/utils', False)
        assert msg2.path == 'src/utils'
        assert msg2.expanded is False