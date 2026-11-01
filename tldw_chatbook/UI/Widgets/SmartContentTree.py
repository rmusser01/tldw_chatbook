# SmartContentTree.py
# Description: Enhanced tree widget with search, filtering, and bulk operations
#
"""
Smart Content Tree
------------------

Enhanced tree widget for content selection with:
- Real-time search filtering
- Category toggles
- Bulk selection operations
- Visual selection indicators
- Lazy loading support
"""

from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Tree, Input, Checkbox, Button, Static
from textual.widgets.tree import TreeNode
from textual.reactive import reactive
from textual.message import Message
from loguru import logger

from ...Chatbooks.chatbook_models import ContentType


@dataclass
class ContentNodeData:
    """Data associated with a content tree node."""

    type: ContentType
    id: str
    title: str
    subtitle: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    selectable: bool = True
    loaded: bool = True
    # Precomputed lowercase-searchable text (B18): built once when the node
    # is added to the tree so filtering never re-joins title/subtitle/metadata
    # per keystroke. Left None by callers; the tree fills it in at load time.
    search_text: Optional[str] = None


class SelectionMode(Enum):
    """Selection modes for bulk operations."""

    ALL = "all"
    NONE = "none"
    INVERT = "invert"


class ContentSelectionChanged(Message):
    """Message sent when content selection changes."""

    def __init__(self, selections: Dict[ContentType, Set[str]]) -> None:
        super().__init__()
        self.selections = selections


class SmartContentTree(Container):
    """Enhanced tree widget with search and filtering capabilities."""

    # Debounce window for search-as-you-type filter passes (seconds).
    SEARCH_DEBOUNCE_SECONDS = 0.3

    # Reactive properties
    search_query = reactive("")
    selected_count = reactive(0)
    filtered_count = reactive(0)

    def __init__(
        self,
        load_content: Optional[
            Callable[[], Dict[ContentType, List[ContentNodeData]]]
        ] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        """
        Initialize the smart content tree.

        Args:
            load_content: Callback to load content data
            lazy_load: Whether to use lazy loading for large datasets
        """
        super().__init__(**kwargs)
        self.load_content_callback = load_content
        self.lazy_load = lazy_load

        # State
        self.selected_content: Dict[ContentType, Set[str]] = {
            content_type: set() for content_type in ContentType
        }
        self.category_filters: Dict[ContentType, bool] = {
            content_type: True for content_type in ContentType
        }
        self.all_nodes: List[TreeNode] = []
        self.content_nodes: Dict[str, TreeNode] = {}  # id -> node mapping
        self.original_labels: Dict[TreeNode, str] = {}  # Store original labels

        # Debounce timer handle for search-as-you-type filter passes (B18).
        self._search_debounce_timer = None

    def compose(self) -> ComposeResult:
        """Compose the tree UI."""
        with Container(classes="tree-controls"):
            # Search row
            with Horizontal(classes="search-row"):
                yield Input(
                    placeholder="Search content...",
                    id="content-search",
                    classes="search-input",
                )
                yield Button(
                    "🔍 Filter",
                    id="apply-filter",
                    classes="filter-button",
                    tooltip="Filter the content tree using the search text and enabled categories.",
                )

            # Category filters
            with Horizontal(classes="category-filters"):
                yield Checkbox(
                    "Conversations",
                    True,
                    id="filter-conversations",
                    classes="category-checkbox",
                )
                yield Checkbox(
                    "Notes", True, id="filter-notes", classes="category-checkbox"
                )
                yield Checkbox(
                    "Characters",
                    True,
                    id="filter-characters",
                    classes="category-checkbox",
                )
                yield Checkbox(
                    "Media", True, id="filter-media", classes="category-checkbox"
                )
                yield Checkbox(
                    "Prompts", True, id="filter-prompts", classes="category-checkbox"
                )
                yield Checkbox(
                    "Kept Briefings",
                    True,
                    id="filter-kept-briefings",
                    classes="category-checkbox",
                )

            # Selection controls
            with Horizontal(classes="selection-controls"):
                yield Button(
                    "Select All",
                    id="select-all",
                    classes="selection-button",
                    variant="default",
                    tooltip="Select every currently visible content item.",
                )
                yield Button(
                    "Select None",
                    id="select-none",
                    classes="selection-button",
                    variant="default",
                    tooltip="Clear every selected content item.",
                )
                yield Button(
                    "Invert",
                    id="select-invert",
                    classes="selection-button",
                    variant="default",
                    tooltip="Reverse selection for the currently visible content items.",
                )

        # Tree container
        with Container(classes="tree-container"):
            yield Tree("Content", id="content-tree")

        # Stats
        yield Static("0 items selected", id="tree-stats", classes="tree-stats")

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        # Load initial content off the UI thread (B18): the callback may hit
        # databases or the filesystem, so it runs in a thread worker and the
        # population hops back via call_from_thread.
        if self.load_content_callback:
            self.run_worker(
                self._load_content_worker, exclusive=True, thread=True
            )

    def _load_content_worker(self) -> None:
        """Worker: run the load callback off the UI thread (B18)."""
        try:
            content_data = self.load_content_callback()
        except Exception as e:
            logger.error(f"Error loading content: {e}")
            self.app.call_from_thread(self._report_load_failure, e)
            return
        self.app.call_from_thread(self._populate_tree, content_data)

    def _report_load_failure(self, error: Exception) -> None:
        """Surface a load failure on the UI thread.

        The exception text renders literally (``markup=False``, TASK-1513):
        a ``[/b]`` in it must not raise ``MarkupError`` in the toast.
        """
        self.notify(
            f"Error loading content: {error}", severity="error", markup=False
        )

    async def load_all_content(self) -> None:
        """Load all content into the tree (legacy async entry point)."""
        if not self.load_content_callback:
            return

        try:
            self._populate_tree(self.load_content_callback())
        except Exception as e:
            logger.error(f"Error loading content: {e}")
            self._report_load_failure(e)

    @staticmethod
    def _build_search_text(item: ContentNodeData) -> str:
        """Build the searchable haystack for a node payload.

        Byte-parity with the historical per-keystroke join: the title and
        subtitle are lowercased, metadata values are joined verbatim (the
        query is lowercased, so verbatim uppercase metadata never matched
        before either).
        """
        searchable_text = f"{item.title} {item.subtitle or ''}".lower()
        if item.metadata:
            searchable_text += " " + " ".join(
                str(v) for v in item.metadata.values()
            )
        return searchable_text

    def _populate_tree(self, content_data: Dict[ContentType, List[ContentNodeData]]) -> None:
        """Populate the tree from loaded content data (runs on the UI thread)."""
        tree = self.query_one("#content-tree", Tree)
        tree.clear()
        self.all_nodes.clear()
        self.content_nodes.clear()
        self.original_labels.clear()

        root = tree.root

        # Create category nodes
        category_nodes = {
            ContentType.CONVERSATION: root.add("💬 Conversations", expand=True),
            ContentType.NOTE: root.add("📝 Notes", expand=True),
            ContentType.CHARACTER: root.add("👤 Characters", expand=True),
            ContentType.MEDIA: root.add("🎬 Media", expand=False),
            ContentType.PROMPT: root.add("💡 Prompts", expand=False),
            ContentType.KEPT_BRIEFING: root.add("📰 Kept Briefings", expand=False),
        }

        # Add content nodes
        for content_type, items in content_data.items():
            parent_node = category_nodes.get(content_type)
            if not parent_node:
                continue

            for item in items:
                label = item.title
                if item.subtitle:
                    label += f" ({item.subtitle})"

                # Precompute the searchable text once per node (B18).
                if item.search_text is None:
                    item.search_text = self._build_search_text(item)

                node = parent_node.add(label)
                node.data = item

                # Store references
                self.all_nodes.append(node)
                self.content_nodes[item.id] = node
                self.original_labels[node] = label

                # Restore selection state
                if item.id in self.selected_content.get(item.type, set()):
                    self._mark_node_selected(node, True)

        # Filters may have changed while the content worker was loading.
        self._apply_filters()

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        if hasattr(node, "data") and node.data and node.data.selectable:
            item = node.data

            # Toggle selection
            if item.id in self.selected_content[item.type]:
                self.selected_content[item.type].remove(item.id)
                self._mark_node_selected(node, False)
            else:
                self.selected_content[item.type].add(item.id)
                self._mark_node_selected(node, True)

            # Update count and notify
            self._update_stats()
            self.post_message(ContentSelectionChanged(self.selected_content))

    def _mark_node_selected(self, node: TreeNode, selected: bool) -> None:
        """Mark a node as selected or unselected."""
        original_label = self.original_labels.get(node, node.label.plain)

        if selected:
            node.set_label(f"✓ {original_label}")
            # TreeNode doesn't have add_class, just update the label
        else:
            node.set_label(original_label)
            # TreeNode doesn't have remove_class, just update the label

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "apply-filter":
            self._apply_filters()
        elif button_id == "select-all":
            self._bulk_select(SelectionMode.ALL)
        elif button_id == "select-none":
            self._bulk_select(SelectionMode.NONE)
        elif button_id == "select-invert":
            self._bulk_select(SelectionMode.INVERT)

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle category filter changes."""
        checkbox_id = event.checkbox.id

        # Map checkbox IDs to content types
        type_map = {
            "filter-conversations": ContentType.CONVERSATION,
            "filter-notes": ContentType.NOTE,
            "filter-characters": ContentType.CHARACTER,
            "filter-media": ContentType.MEDIA,
            "filter-prompts": ContentType.PROMPT,
            "filter-kept-briefings": ContentType.KEPT_BRIEFING,
        }

        content_type = type_map.get(checkbox_id)
        if content_type:
            self.category_filters[content_type] = event.value
            self._apply_filters()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes (debounced -- one sweep per settled query)."""
        if event.input.id == "content-search":
            query = event.value.lower()
            if self._search_debounce_timer is not None:
                self._search_debounce_timer.stop()
            self._search_debounce_timer = self.set_timer(
                self.SEARCH_DEBOUNCE_SECONDS,
                lambda: self._apply_search_query(query),
            )

    def _apply_search_query(self, query: str) -> None:
        """Apply the settled search query with a single filter pass (B18)."""
        self._search_debounce_timer = None
        self.search_query = query
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Apply search and category filters to the tree.

        Reads the per-node precomputed ``search_text`` (B18) and only writes
        ``node.display`` when visibility actually changes.
        """
        visible_count = 0
        query = self.search_query

        for node in self.all_nodes:
            if not hasattr(node, "data") or not node.data:
                continue

            item = node.data
            visible = self.category_filters.get(item.type, True)

            # Check search query against the precomputed haystack.
            if visible and query:
                search_text = item.search_text
                if search_text is None:
                    # Node added outside _populate_tree: build once lazily.
                    search_text = item.search_text = self._build_search_text(item)
                if query not in search_text:
                    visible = False

            # Only touch display when visibility actually changes.
            if getattr(node, "display", None) is not visible:
                node.display = visible

            if visible:
                visible_count += 1

                # Ensure parent is expanded if node is visible
                if node.parent and hasattr(node.parent, "expand"):
                    node.parent.expand()

        self.filtered_count = visible_count
        self._update_stats()

    def _bulk_select(self, mode: SelectionMode) -> None:
        """Perform bulk selection operation."""
        for node in self.all_nodes:
            if not hasattr(node, "data") or not node.data or not node.data.selectable:
                continue

            # Only operate on visible nodes
            if not node.display:
                continue

            item = node.data

            if mode == SelectionMode.ALL:
                if item.id not in self.selected_content[item.type]:
                    self.selected_content[item.type].add(item.id)
                    self._mark_node_selected(node, True)

            elif mode == SelectionMode.NONE:
                if item.id in self.selected_content[item.type]:
                    self.selected_content[item.type].remove(item.id)
                    self._mark_node_selected(node, False)

            elif mode == SelectionMode.INVERT:
                if item.id in self.selected_content[item.type]:
                    self.selected_content[item.type].remove(item.id)
                    self._mark_node_selected(node, False)
                else:
                    self.selected_content[item.type].add(item.id)
                    self._mark_node_selected(node, True)

        self._update_stats()
        self.post_message(ContentSelectionChanged(self.selected_content))

    def _update_stats(self) -> None:
        """Update selection statistics."""
        total_selected = sum(len(items) for items in self.selected_content.values())
        self.selected_count = total_selected

        stats_text = f"{total_selected} items selected"
        if self.search_query or not all(self.category_filters.values()):
            stats_text += f" ({self.filtered_count} visible)"

        stats_widget = self.query_one("#tree-stats", Static)
        stats_widget.update(stats_text)

    def get_selections(self) -> Dict[ContentType, List[str]]:
        """Get current selections as lists."""
        return {
            content_type: list(items)
            for content_type, items in self.selected_content.items()
            if items
        }

    def set_selections(self, selections: Dict[ContentType, List[str]]) -> None:
        """Set selections programmatically."""
        # Clear current selections
        for node in self.all_nodes:
            if hasattr(node, "data") and node.data:
                self._mark_node_selected(node, False)

        # Apply new selections
        self.selected_content = {
            content_type: set(items) for content_type, items in selections.items()
        }

        # Update visual state
        for content_type, item_ids in selections.items():
            for item_id in item_ids:
                node = self.content_nodes.get(item_id)
                if node:
                    self._mark_node_selected(node, True)

        self._update_stats()
        self.post_message(ContentSelectionChanged(self.selected_content))

    def clear_selections(self) -> None:
        """Clear all selections."""
        self._bulk_select(SelectionMode.NONE)
