from anytree import Node
from textual.app import App, ComposeResult
from textual.widgets import Tree, Input, Button


class MindMapApp(App):
    def __init__(self):
        super().__init__()
        self.root_node = Node("LLM-Generated Topic")  # Start with a root

    def compose(self) -> ComposeResult:
        yield Tree("MindMap")  # Tree widget for display
        yield Input(placeholder="Add a child node...")
        yield Button("Add Node", variant="primary")

    def on_mount(self) -> None:
        self.update_tree()

    def update_tree(self) -> None:
        """Refresh the tree widget from anytree data."""
        tree = self.query_one(Tree)
        tree.clear()
        root = tree.root.add(self.root_node.name)
        self._add_children_to_tree(root, self.root_node)

    def _add_children_to_tree(self, tree_node, anytree_node):
        """Recursively add children from anytree to the Textual tree."""
        for child in anytree_node.children:
            new_node = tree_node.add(child.name)
            self._add_children_to_tree(new_node, child)

    def on_button_pressed(self) -> None:
        """Add a new node to the selected item."""
        input = self.query_one(Input)
        if input.value:
            selected_node = self.query_one(Tree).cursor_node
            if selected_node == self.query_one(Tree).root:
                parent = self.root_node
            else:
                parent = self._find_anytree_node(selected_node)
            Node(input.value, parent=parent)
            self.update_tree()
            input.value = ""

    def _find_anytree_node(self, tree_node) -> Node:
        """Helper to map Textual Tree nodes back to anytree Nodes."""
        path = [tree_node.label]
        while tree_node.parent:
            tree_node = tree_node.parent
            path.append(tree_node.label)
        path = path[::-1][1:]  # Remove root, reverse to top-down
        current = self.root_node
        for name in path:
            current = next(child for child in current.children if child.name == name)
        return current


if __name__ == "__main__":
    MindMapApp().run()
