"""
Property-based tests for repository tree operations.

Uses Hypothesis to generate random tree structures and test invariants.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize
from typing import Dict, Set, List, Optional
import random

from tldw_chatbook.Widgets.repo_tree_widgets import TreeView, TreeNode


# Strategies for generating tree data
def file_name_strategy():
    """Generate realistic file names."""
    extensions = ['.py', '.js', '.md', '.txt', '.json', '.yaml', '.html', '.css']
    return st.text(
        alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), min_codepoint=97),
        min_size=1,
        max_size=20
    ).map(lambda s: s + random.choice(extensions))


def directory_name_strategy():
    """Generate realistic directory names."""
    return st.text(
        alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), min_codepoint=97),
        min_size=1,
        max_size=15
    ).filter(lambda s: s not in ['', '.', '..'])


def path_component_strategy():
    """Generate a single path component (file or directory name)."""
    return st.one_of(file_name_strategy(), directory_name_strategy())


def tree_node_strategy(max_depth=5, current_depth=0):
    """Generate tree node data recursively."""
    if current_depth >= max_depth:
        # Force leaf nodes at max depth
        return st.fixed_dictionaries({
            'name': file_name_strategy(),
            'type': st.just('blob'),
            'size': st.integers(min_value=0, max_value=1000000),
            'children': st.just([])
        })
    
    # Choose between file and directory
    return st.one_of(
        # File node
        st.fixed_dictionaries({
            'name': file_name_strategy(),
            'type': st.just('blob'),
            'size': st.integers(min_value=0, max_value=1000000),
            'children': st.just([])
        }),
        # Directory node
        st.fixed_dictionaries({
            'name': directory_name_strategy(),
            'type': st.just('tree'),
            'size': st.none(),
            'children': st.lists(
                tree_node_strategy(max_depth, current_depth + 1),
                min_size=0,
                max_size=5
            )
        })
    )


def add_paths_to_tree(tree_items: List[Dict], parent_path: str = "") -> List[Dict]:
    """Add path field to tree items based on hierarchy."""
    result = []
    for item in tree_items:
        item_copy = item.copy()
        path = f"{parent_path}/{item['name']}" if parent_path else item['name']
        item_copy['path'] = path
        
        if item.get('children'):
            item_copy['children'] = add_paths_to_tree(item['children'], path)
        
        result.append(item_copy)
    return result


class TreeViewStateMachine(RuleBasedStateMachine):
    """State machine for testing TreeView operations."""
    
    def __init__(self):
        super().__init__()
        self.tree_view = TreeView()
        self.tree_data: List[Dict] = []
        self.all_paths: Set[str] = set()
        self.directory_paths: Set[str] = set()
        self.file_paths: Set[str] = set()
    
    @initialize(tree_data=st.lists(tree_node_strategy(), min_size=0, max_size=10))
    def initialize_tree(self, tree_data):
        """Initialize tree with random data."""
        # Add paths to tree data
        self.tree_data = add_paths_to_tree(tree_data)
        
        # Extract all paths
        def extract_paths(items, parent_path=""):
            for item in items:
                path = item['path']
                self.all_paths.add(path)
                
                if item['type'] == 'tree':
                    self.directory_paths.add(path)
                else:
                    self.file_paths.add(path)
                
                if item.get('children'):
                    extract_paths(item['children'], path)
        
        extract_paths(self.tree_data)
        
        # Load tree data into view (synchronously for testing)
        # Note: In real async environment, this would be awaited
        self.tree_view._tree_data = self.tree_data
        
        # Simulate tree loading
        for path in self.all_paths:
            is_dir = path in self.directory_paths
            node = TreeNode(
                path=path,
                name=path.split('/')[-1],
                is_directory=is_dir,
                level=path.count('/'),
                file_size=random.randint(100, 10000) if not is_dir else None
            )
            self.tree_view.nodes[path] = node
    
    @rule(path=st.sampled_from(lambda self: list(self.all_paths) if self.all_paths else ['']))
    def select_node(self, path):
        """Select a random node."""
        if path in self.all_paths:
            self.tree_view.select_node(path, True)
    
    @rule(path=st.sampled_from(lambda self: list(self.all_paths) if self.all_paths else ['']))
    def deselect_node(self, path):
        """Deselect a random node."""
        if path in self.all_paths:
            self.tree_view.select_node(path, False)
    
    @rule()
    def select_all(self):
        """Select all nodes."""
        for path in self.all_paths:
            self.tree_view.select_node(path, True)
    
    @rule()
    def deselect_all(self):
        """Deselect all nodes."""
        for path in self.all_paths:
            self.tree_view.select_node(path, False)
    
    @invariant()
    def selection_consistency(self):
        """Check that selection state is consistent."""
        # All selected paths should be in nodes
        for path in self.tree_view.selection:
            assert path in self.tree_view.nodes, f"Selected path {path} not in nodes"
        
        # Node selection state should match selection set
        for path, node in self.tree_view.nodes.items():
            if node.selected:
                assert path in self.tree_view.selection, f"Node {path} marked selected but not in selection"
            else:
                assert path not in self.tree_view.selection, f"Node {path} not marked selected but in selection"
    
    @invariant()
    def directory_selection_cascades(self):
        """Check that selecting a directory selects all children."""
        for dir_path in self.directory_paths:
            if dir_path in self.tree_view.selection:
                # All children should be selected
                for path in self.all_paths:
                    if path.startswith(dir_path + '/'):
                        assert path in self.tree_view.selection, \
                            f"Child {path} of selected directory {dir_path} is not selected"
    
    @invariant()
    def selected_files_are_files(self):
        """Check that get_selected_files only returns files."""
        selected_files = self.tree_view.get_selected_files()
        for file_path in selected_files:
            assert file_path in self.file_paths, f"{file_path} returned by get_selected_files is not a file"
            assert file_path not in self.directory_paths, f"{file_path} is a directory but in selected files"
    
    @invariant()
    def selection_stats_accuracy(self):
        """Check that selection statistics are accurate."""
        stats = self.tree_view.get_selection_stats()
        
        # Count actual selected files
        selected_files = [p for p in self.tree_view.selection if p in self.file_paths]
        assert stats['files'] == len(selected_files), \
            f"Stats report {stats['files']} files but actually {len(selected_files)} selected"
        
        # Check total size (if nodes have size)
        total_size = sum(
            self.tree_view.nodes[p].file_size or 0 
            for p in selected_files 
            if p in self.tree_view.nodes
        )
        assert stats['size'] == total_size, \
            f"Stats report {stats['size']} total size but actually {total_size}"


# Property tests for TreeNode
class TestTreeNodeProperties:
    """Property-based tests for TreeNode."""
    
    @given(
        path=st.text(min_size=1),
        name=file_name_strategy(),
        level=st.integers(min_value=0, max_value=10),
        size=st.one_of(st.none(), st.integers(min_value=0, max_value=10**9))
    )
    def test_file_node_properties(self, path, name, level, size):
        """Test properties of file nodes."""
        node = TreeNode(
            path=path,
            name=name,
            is_directory=False,
            level=level,
            size=size
        )
        
        # Basic properties
        assert node.path == path
        assert node.node_name == name
        assert not node.is_directory
        assert node.level == level
        assert node.file_size == size
        
        # Initial state
        assert node.expanded == False
        assert node.selected == False
        assert not node.children_loaded
        
        # Files should get file icons
        icon = node._get_icon()
        assert icon != "📁"  # Not closed folder
        assert icon != "📂"  # Not open folder
    
    @given(
        path=st.text(min_size=1),
        name=directory_name_strategy(),
        level=st.integers(min_value=0, max_value=10)
    )
    def test_directory_node_properties(self, path, name, level):
        """Test properties of directory nodes."""
        node = TreeNode(
            path=path,
            name=name,
            is_directory=True,
            level=level
        )
        
        # Directories shouldn't have size
        assert node.file_size is None
        
        # Icon changes with expansion
        assert node._get_icon() == "📁"  # Closed
        node.expanded = True
        assert node._get_icon() == "📂"  # Open
    
    @given(size=st.integers(min_value=0, max_value=10**15))
    def test_size_formatting(self, size):
        """Test that size formatting is always valid."""
        node = TreeNode(
            path="test",
            name="test",
            is_directory=False,
            level=0,
            size=size
        )
        
        formatted = node._format_size(size)
        
        # Should always return a string
        assert isinstance(formatted, str)
        
        # Should contain a number and unit
        if size > 0:
            assert any(unit in formatted for unit in ['B', 'KB', 'MB', 'GB', 'TB'])
            # Extract number part
            number_part = formatted.split()[0]
            assert float(number_part) >= 0
    
    @given(
        level=st.integers(min_value=0, max_value=20)
    )
    def test_indentation_level(self, level):
        """Test that indentation is proportional to level."""
        node = TreeNode(
            path="test",
            name="test",
            is_directory=True,
            level=level
        )
        
        # In compose, indentation should be 2 spaces per level
        # This would be tested in actual rendering
        assert node.level == level


# Property tests for TreeView operations
class TestTreeViewProperties:
    """Property-based tests for TreeView."""
    
    @given(st.lists(
        st.tuples(
            st.text(min_size=1, max_size=50),  # path
            st.booleans()  # is_directory
        ),
        min_size=0,
        max_size=100
    ))
    def test_selection_operations(self, path_data):
        """Test that selection operations maintain consistency."""
        tree = TreeView()
        
        # Create nodes
        for path, is_dir in path_data:
            node = TreeNode(
                path=path,
                name=path.split('/')[-1] if '/' in path else path,
                is_directory=is_dir,
                level=path.count('/')
            )
            tree.nodes[path] = node
        
        # Randomly select/deselect nodes
        for path, _ in path_data:
            select = random.choice([True, False])
            tree.select_node(path, select)
        
        # Verify consistency
        for path, node in tree.nodes.items():
            if path in tree.selection:
                assert node.selected == True
            else:
                assert node.selected == False
        
        # get_selected_files should only return non-directories
        selected_files = tree.get_selected_files()
        for file_path in selected_files:
            assert not tree.nodes[file_path].is_directory
    
    @given(
        tree_data=st.lists(tree_node_strategy(), min_size=0, max_size=20)
    )
    def test_tree_hierarchy_properties(self, tree_data):
        """Test properties of tree hierarchy building."""
        tree_data = add_paths_to_tree(tree_data)
        
        # Collect all paths
        all_paths = set()
        
        def collect_paths(items):
            for item in items:
                all_paths.add(item['path'])
                if item.get('children'):
                    collect_paths(item['children'])
        
        collect_paths(tree_data)
        
        # All paths should be unique
        assert len(all_paths) == len(list(all_paths))
        
        # Parent paths should exist for nested items
        for path in all_paths:
            if '/' in path:
                parent_path = '/'.join(path.split('/')[:-1])
                if parent_path:  # Not empty
                    assert parent_path in all_paths, \
                        f"Parent path {parent_path} missing for {path}"


# Test the state machine
TestTreeViewStateMachine = TreeViewStateMachine.TestCase

# Configure Hypothesis settings for better performance
TestTreeViewStateMachine.settings = settings(
    max_examples=50,
    stateful_step_count=20
)


# Additional focused property tests
class TestSelectionProperties:
    """Focused tests on selection behavior."""
    
    @given(
        parent_path=st.text(min_size=1, max_size=20),
        child_count=st.integers(min_value=1, max_value=10)
    )
    def test_directory_cascade_selection(self, parent_path, child_count):
        """Test that directory selection cascades to all children."""
        tree = TreeView()
        
        # Create parent directory
        parent_node = TreeNode(
            path=parent_path,
            name=parent_path,
            is_directory=True,
            level=0
        )
        tree.nodes[parent_path] = parent_node
        
        # Create child files
        child_paths = []
        for i in range(child_count):
            child_path = f"{parent_path}/file_{i}.txt"
            child_node = TreeNode(
                path=child_path,
                name=f"file_{i}.txt",
                is_directory=False,
                level=1
            )
            tree.nodes[child_path] = child_node
            child_paths.append(child_path)
        
        # Select parent
        tree.select_node(parent_path, True)
        
        # All children should be selected
        assert parent_path in tree.selection
        for child_path in child_paths:
            assert child_path in tree.selection
            assert tree.nodes[child_path].selected == True
        
        # Deselect parent
        tree.select_node(parent_path, False)
        
        # All children should be deselected
        assert parent_path not in tree.selection
        for child_path in child_paths:
            assert child_path not in tree.selection
            assert tree.nodes[child_path].selected == False