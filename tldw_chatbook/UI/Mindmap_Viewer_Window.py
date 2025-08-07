# Mindmap_Viewer_Window.py
# Description: Mindmap viewer window for the Study tab
#
"""
Mindmap Viewer Window
--------------------

Visual learning with mindmaps:
- Interactive mindmap visualization
- Content selection and integration
- Export options for study materials
"""

from typing import Optional, Dict, List, Any
from textual import on, work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, Label, TabbedContent, TabPane, LoadingIndicator
from textual.reactive import reactive
from loguru import logger
from datetime import datetime

from ..Chatbooks.chatbook_models import ContentType
from ..Utils.optional_deps import check_mindmap_available

# Lazy imports
SmartContentTree = None
ContentNodeData = None
ContentSelectionChanged = None
MindmapViewer = None
MindmapNodeSelected = None
MindmapIntegration = None
MindmapExporter = None

MINDMAP_AVAILABLE = False


def _lazy_import_mindmap():
    """Lazy import mindmap components"""
    global SmartContentTree, ContentNodeData, ContentSelectionChanged
    global MindmapViewer, MindmapNodeSelected
    global MindmapIntegration, MindmapExporter
    global MINDMAP_AVAILABLE
    
    if MINDMAP_AVAILABLE:
        return True
    
    try:
        from ..UI.Widgets import SmartContentTree, ContentNodeData, ContentSelectionChanged
        from ..UI.Widgets import MindmapViewer, MindmapNodeSelected
        from ..Tools.Mind_Map.mindmap_integration import MindmapIntegration
        from ..Tools.Mind_Map.mindmap_exporter import MindmapExporter
        
        MINDMAP_AVAILABLE = True
        return True
    except ImportError as e:
        logger.warning(f"Mindmap components not available: {e}")
        return False


class MindmapViewerWindow(Screen):
    """Mindmap visualization window"""
    
    DEFAULT_CSS = """
    MindmapViewerWindow {
        layout: vertical;
    }
    
    #mindmap-header {
        height: 3;
        background: $boost;
        padding: 1;
        border: round $background-darken-1;
    }
    
    #mindmap-content {
        height: 1fr;
    }
    
    .mindmap-main-container {
        layout: horizontal;
        height: 100%;
    }
    
    #content-selector {
        width: 30%;
        padding: 1;
        border-right: solid $primary;
    }
    
    #mindmap-display-container {
        width: 70%;
        padding: 1;
    }
    
    #export-controls {
        height: auto;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
        margin-top: 1;
    }
    
    .export-button {
        margin-right: 1;
        min-width: 15;
    }
    
    #no-mindmap-message {
        height: 100%;
        align: center middle;
        text-align: center;
    }
    
    .loading-container {
        height: 100%;
        align: center middle;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """
    
    # Reactive properties
    has_content = reactive(False)
    mindmap_loaded = reactive(False)
    
    def __init__(self, app_instance, **kwargs):
        """Initialize the mindmap viewer window
        
        Args:
            app_instance: Main application instance
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.db = app_instance.chachanotes_db
        self.current_mindmap_root = None
        
        # Check if mindmap is available
        self.mindmap_available = check_mindmap_available()
        if self.mindmap_available:
            _lazy_import_mindmap()
    
    def compose(self) -> ComposeResult:
        """Compose the mindmap viewer UI"""
        # Header
        with Container(id="mindmap-header"):
            yield Label("🧠 Mindmap Viewer - Visual Knowledge Mapping")
        
        # Main content
        with Container(id="mindmap-content"):
            if not self.mindmap_available:
                yield from self._compose_unavailable_view()
            else:
                yield from self._compose_mindmap_view()
    
    def _compose_unavailable_view(self) -> ComposeResult:
        """Compose view when mindmap is not available"""
        with Container(id="no-mindmap-message"):
            yield Label("🧠 Mindmap Viewer", classes="title")
            yield Label("")
            yield Label("The mindmap viewer requires additional dependencies.")
            yield Label("Install with: pip install tldw_chatbook[mindmap]")
            yield Label("")
            yield Button("Check Again", id="check-deps", variant="primary")
    
    def on_mount(self) -> None:
        """Called when the window is mounted."""
        # If mindmap is not available, show the alert dialog
        if not self.mindmap_available:
            from ..Utils.widget_helpers import alert_mindmap_not_available
            # Show alert after a short delay to ensure UI is ready
            self.set_timer(0.1, lambda: alert_mindmap_not_available(self))
    
    def _compose_mindmap_view(self) -> ComposeResult:
        """Compose the main mindmap view"""
        with Container(classes="mindmap-main-container"):
            # Content selector
            with Vertical(id="content-selector"):
                yield Label("Select Content", classes="section-title")
                
                if SmartContentTree:
                    yield SmartContentTree(
                        load_content=self._load_content,
                        id="content-tree"
                    )
                else:
                    yield Label("Content tree not available")
                
                yield Button("Create Mindmap", id="create-mindmap", variant="primary")
                yield Button("Clear Selection", id="clear-selection", variant="default")
                yield Button("Import Canvas", id="import-canvas", variant="default", tooltip="Import from Obsidian JSON Canvas")
            
            # Mindmap display
            with Vertical(id="mindmap-display-container"):
                if MindmapViewer:
                    # Sample mindmap for initial display
                    sample_mermaid = """
                    mindmap
                      root((Mindmap Viewer))
                        Features[Key Features]
                          Visual[Visual Mapping]
                            Tree[Tree View]
                            Outline[Outline View]
                            ASCII[ASCII Art]
                          Navigation[Navigation]
                            Keys[Keyboard]
                            Search[Search]
                            Jump[Quick Jump]
                          Export[Export Options]
                            MD[Markdown]
                            HTML[Interactive HTML]
                            Cards[Flashcards]
                        Usage[How to Use]
                          Select[1. Select Content]
                          Create[2. Create Mindmap]
                          Navigate[3. Navigate & Explore]
                          Export[4. Export Results]
                        Content[Content Types]
                          Notes[Study Notes]
                          Convs[Conversations]
                          Media[Media Items]
                    """
                    
                    yield MindmapViewer(
                        mermaid_code=sample_mermaid,
                        id="mindmap-viewer",
                        show_controls=True
                    )
                else:
                    yield Label("Mindmap viewer not available", id="mindmap-placeholder")
                
                # Export controls
                with Horizontal(id="export-controls"):
                    yield Button("📝 Markdown", id="export-markdown", classes="export-button", variant="default")
                    yield Button("🃏 Flashcards", id="export-flashcards", classes="export-button", variant="default")
                    yield Button("🌐 HTML", id="export-html", classes="export-button", variant="default")
                    yield Button("📊 GraphViz", id="export-graphviz", classes="export-button", variant="default")
                    yield Button("📋 JSON", id="export-json", classes="export-button", variant="default")
                    yield Button("🗺️ Canvas", id="export-canvas", classes="export-button", variant="default", tooltip="Export to Obsidian JSON Canvas")
    
    def _load_content(self) -> Dict[ContentType, List[Any]]:
        """Load available content
        
        Returns:
            Dictionary of content by type
        """
        content = {}
        
        try:
            # Load notes
            notes = self.db.list_notes(limit=100)
            if notes:
                content[ContentType.NOTE] = [
                    ContentNodeData(
                        type=ContentType.NOTE,
                        id=note['id'],
                        title=note['title'],
                        subtitle=f"Updated: {note.get('updated_at', 'Unknown')[:10]}",
                        metadata={'tags': note.get('tags', [])}
                    )
                    for note in notes
                ]
            
            # Load conversations
            conversations = self.db.get_conversations(limit=50)
            if conversations:
                content[ContentType.CONVERSATION] = [
                    ContentNodeData(
                        type=ContentType.CONVERSATION,
                        id=conv['id'],
                        title=conv['title'],
                        subtitle=f"Messages: {conv.get('message_count', 0)}",
                        metadata={'created': conv.get('created_at')}
                    )
                    for conv in conversations
                ]
            
            # Load characters
            characters = self.db.get_all_characters()
            if characters:
                content[ContentType.CHARACTER] = [
                    ContentNodeData(
                        type=ContentType.CHARACTER,
                        id=char['id'],
                        title=char['name'],
                        subtitle=char.get('short_description', '')[:50],
                        metadata={'version': char.get('version')}
                    )
                    for char in characters
                ]
            
        except Exception as e:
            logger.error(f"Error loading content: {e}")
        
        return content
    
    @on(Button.Pressed, "#check-deps")
    def check_dependencies_again(self) -> None:
        """Re-check for mindmap dependencies"""
        self.mindmap_available = check_mindmap_available()
        if self.mindmap_available and _lazy_import_mindmap():
            # Refresh the UI
            self.refresh(recompose=True)
            self.notify("Mindmap dependencies found! Reloading...")
        else:
            self.notify("Mindmap dependencies still not available", severity="warning")
    
    @on(Button.Pressed, "#create-mindmap")
    @work(exclusive=True)
    async def create_mindmap(self) -> None:
        """Create mindmap from selected content"""
        if not self.mindmap_available or not SmartContentTree:
            self.notify("Mindmap features not available", severity="error")
            return
        
        # Show loading
        # TODO: Add loading indicator
        
        try:
            # Get selections from content tree
            content_tree = self.query_one("#content-tree", SmartContentTree)
            selections = content_tree.get_selections()
            
            # Check if anything is selected
            total_selections = sum(len(items) for items in selections.values())
            if total_selections == 0:
                self.notify("Please select content to create a mindmap", severity="warning")
                return
            
            # Create mindmap from selections
            integration = MindmapIntegration(self.app_instance)
            root = await self.run_worker(
                integration.create_from_smart_tree_selection,
                content_tree,
                exclusive=True
            )
            
            self.current_mindmap_root = root
            
            # Update mindmap viewer
            viewer = self.query_one("#mindmap-viewer", MindmapViewer)
            viewer.model.root = root
            viewer.model.selected_node = root
            viewer.model.expanded_nodes = {root}
            viewer.refresh_display()
            
            self.mindmap_loaded = True
            self.notify(f"Created mindmap with {total_selections} items")
            
        except Exception as e:
            logger.error(f"Error creating mindmap: {e}")
            self.notify(f"Error creating mindmap: {str(e)}", severity="error")
        finally:
            # Hide loading
            # TODO: Remove loading indicator
            pass
    
    @on(Button.Pressed, "#clear-selection")
    def clear_selection(self) -> None:
        """Clear all content selections"""
        if SmartContentTree:
            content_tree = self.query_one("#content-tree", SmartContentTree)
            content_tree.clear_selections()
            self.notify("Cleared all selections")
    
    @on(Button.Pressed, "#import-canvas")
    @work(exclusive=True)
    async def import_canvas_file(self) -> None:
        """Import mindmap from JSON Canvas file"""
        if not self.mindmap_available or not MindmapViewer:
            self.notify("Mindmap features not available", severity="error")
            return
        
        try:
            # Use file dialog if available
            from pathlib import Path
            import json
            
            # For now, use a hardcoded path - in production would use file picker
            # TODO: Integrate with proper file picker when available
            import_path = Path.home() / "Documents" / "tldw_exports"
            canvas_files = list(import_path.glob("*.canvas"))
            
            if not canvas_files:
                self.notify("No .canvas files found in Documents/tldw_exports", severity="warning")
                return
            
            # Use the most recent canvas file
            latest_file = max(canvas_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                canvas_data = f.read()
            
            # Import the canvas
            root = MindmapExporter.from_json_canvas(canvas_data)
            self.current_mindmap_root = root
            
            # Update mindmap viewer
            viewer = self.query_one("#mindmap-viewer", MindmapViewer)
            viewer.model.root = root
            viewer.model.selected_node = root
            viewer.model.expanded_nodes = {root}
            viewer.refresh_display()
            
            self.mindmap_loaded = True
            self.notify(f"Imported mindmap from {latest_file.name}")
            
        except Exception as e:
            logger.error(f"Error importing canvas: {e}")
            self.notify(f"Error importing canvas: {str(e)}", severity="error")
    
    @on(ContentSelectionChanged)
    def handle_selection_change(self, event: ContentSelectionChanged) -> None:
        """Update UI when content selection changes"""
        total_selections = sum(len(items) for items in event.selections.values())
        self.has_content = total_selections > 0
        
        # Update create button text
        create_btn = self.query_one("#create-mindmap", Button)
        if total_selections > 0:
            create_btn.label = f"Create Mindmap ({total_selections})"
        else:
            create_btn.label = "Create Mindmap"
    
    @on(Button.Pressed, "#export-markdown")
    async def export_to_markdown(self) -> None:
        """Export mindmap to Markdown"""
        await self._export_mindmap("markdown")
    
    @on(Button.Pressed, "#export-flashcards")
    async def export_flashcards(self) -> None:
        """Export mindmap as flashcards"""
        await self._export_mindmap("flashcards")
    
    @on(Button.Pressed, "#export-html")
    async def export_to_html(self) -> None:
        """Export mindmap to HTML"""
        await self._export_mindmap("html")
    
    @on(Button.Pressed, "#export-graphviz")
    async def export_to_graphviz(self) -> None:
        """Export mindmap to GraphViz"""
        await self._export_mindmap("graphviz")
    
    @on(Button.Pressed, "#export-json")
    async def export_to_json(self) -> None:
        """Export mindmap to JSON"""
        await self._export_mindmap("json")
    
    @on(Button.Pressed, "#export-canvas")
    async def export_to_canvas(self) -> None:
        """Export mindmap to Obsidian JSON Canvas format"""
        await self._export_mindmap("canvas")
    
    async def _export_mindmap(self, format_type: str) -> None:
        """Export mindmap in specified format
        
        Args:
            format_type: Export format (markdown, flashcards, html, graphviz, json)
        """
        if not self.current_mindmap_root or not MindmapExporter:
            self.notify("No mindmap to export", severity="warning")
            return
        
        try:
            from pathlib import Path
            export_dir = Path.home() / "Documents" / "tldw_exports"
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format_type == "markdown":
                content = MindmapExporter.to_markdown(self.current_mindmap_root, include_metadata=True)
                filename = f"mindmap_{timestamp}.md"
                
            elif format_type == "flashcards":
                flashcards = MindmapExporter.to_flashcards(
                    self.current_mindmap_root,
                    card_type="basic",
                    include_context=True
                )
                if not flashcards:
                    self.notify("No flashcards could be generated", severity="warning")
                    return
                
                import json
                content = json.dumps(flashcards, indent=2, ensure_ascii=False)
                filename = f"flashcards_{timestamp}.json"
                self.notify(f"Generated {len(flashcards)} flashcards")
                
            elif format_type == "html":
                content = MindmapExporter.to_html(
                    self.current_mindmap_root,
                    include_style=True,
                    collapsible=True
                )
                filename = f"mindmap_{timestamp}.html"
                
            elif format_type == "graphviz":
                content = MindmapExporter.to_graphviz(
                    self.current_mindmap_root,
                    rankdir="TB",
                    include_style=True
                )
                filename = f"mindmap_{timestamp}.dot"
                
            elif format_type == "json":
                content = MindmapExporter.to_json(self.current_mindmap_root, pretty=True)
                filename = f"mindmap_{timestamp}.json"
            
            elif format_type == "canvas":
                content = MindmapExporter.to_json_canvas(
                    self.current_mindmap_root,
                    layout="hierarchical",
                    include_metadata=True
                )
                filename = f"mindmap_{timestamp}.canvas"
                self.notify("Exported to Obsidian JSON Canvas format")
            
            else:
                self.notify(f"Unknown export format: {format_type}", severity="error")
                return
            
            filepath = export_dir / filename
            MindmapExporter.save_to_file(content, filepath)
            self.notify(f"Exported to {filepath}")
            
            # Open HTML files in browser
            if format_type == "html":
                import webbrowser
                webbrowser.open(filepath.as_uri())
                
        except Exception as e:
            logger.error(f"Export error: {e}")
            self.notify(f"Export failed: {str(e)}", severity="error")
    
    @on(MindmapNodeSelected)
    def handle_node_selection(self, event: MindmapNodeSelected) -> None:
        """Handle mindmap node selection"""
        node = event.node
        
        # Show info about selected node
        if hasattr(node, 'metadata') and node.metadata:
            node_type = node.metadata.get('type', 'unknown')
            node_id = node.metadata.get('id', '')
            
            if node_type == 'note' and node_id:
                self.notify(f"Selected note: {node.text}")
                # Could open the note in editor
                
            elif node_type == 'conversation' and node_id:
                self.notify(f"Selected conversation: {node.text}")
                # Could switch to conversation view
                
            elif node_type == 'media' and node_id:
                self.notify(f"Selected media: {node.text}")
                # Could show media details