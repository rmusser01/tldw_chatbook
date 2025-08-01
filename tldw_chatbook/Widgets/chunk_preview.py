# tldw_chatbook/Widgets/chunk_preview.py
# Chunk preview widget for visualizing text chunking
#
# Imports
from typing import List, Optional, Tuple
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Container
from textual.widgets import Static, Label
from textual.widget import Widget
from textual.reactive import reactive

class ChunkPreview(Widget):
    """Widget to preview text chunks with boundaries and metadata.
    
    Features:
    - Shows first N chunks
    - Highlights chunk boundaries
    - Displays token/word counts
    - Shows overlap regions
    """
    
    DEFAULT_CLASSES = "chunk-preview-widget"
    
    # Reactive data
    chunks: reactive[List[Tuple[str, int, int]]] = reactive([])  # (text, start, end)
    overlap_size: reactive[int] = reactive(0)
    
    def __init__(
        self,
        max_chunks: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_chunks = max_chunks
        
    def compose(self) -> ComposeResult:
        """Compose the chunk preview display."""
        with VerticalScroll(classes="chunk-preview-container"):
            yield Label("Chunk Preview", classes="chunk-preview-title")
            
            if not self.chunks:
                yield Static(
                    "No preview available. Select content and configure chunking to see preview.",
                    classes="chunk-preview-empty"
                )
            else:
                for i, (chunk_text, start, end) in enumerate(self.chunks[:self.max_chunks]):
                    if i >= self.max_chunks:
                        break
                        
                    with Container(classes="chunk-item"):
                        # Chunk header
                        with Horizontal(classes="chunk-header"):
                            yield Static(f"Chunk {i + 1}", classes="chunk-number")
                            yield Static(f"[{start}-{end}]", classes="chunk-range")
                            yield Static(f"{len(chunk_text.split())} words", classes="chunk-size")
                        
                        # Chunk content with overlap highlighting
                        if i > 0 and self.overlap_size > 0:
                            # Show overlap from previous chunk
                            overlap_text = self._get_overlap_text(i, chunk_text)
                            if overlap_text:
                                yield Static(
                                    overlap_text,
                                    classes="chunk-overlap"
                                )
                        
                        # Main chunk content
                        yield Static(
                            self._truncate_text(chunk_text, 200),
                            classes="chunk-content"
                        )
                        
                        # Show if there's more content
                        if len(chunk_text) > 200:
                            yield Static("... (truncated)", classes="chunk-truncated")
                
                # Show if there are more chunks
                if len(self.chunks) > self.max_chunks:
                    yield Static(
                        f"... and {len(self.chunks) - self.max_chunks} more chunks",
                        classes="chunk-more"
                    )
    
    def update_chunks(self, chunks: List[Tuple[str, int, int]], overlap_size: int = 0) -> None:
        """Update the preview with new chunks."""
        self.chunks = chunks
        self.overlap_size = overlap_size
        self.refresh()
    
    def _get_overlap_text(self, chunk_index: int, chunk_text: str) -> Optional[str]:
        """Extract overlap text if applicable."""
        if chunk_index == 0 or self.overlap_size == 0:
            return None
            
        # Simple approximation - show first N characters as overlap
        overlap_chars = min(self.overlap_size * 5, len(chunk_text) // 4)  # Rough estimate
        if overlap_chars > 0:
            return chunk_text[:overlap_chars]
        return None
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length]
    
    def clear(self) -> None:
        """Clear the preview."""
        self.chunks = []
        self.refresh()


class ChunkBoundaryIndicator(Static):
    """Visual indicator for chunk boundaries."""
    
    def __init__(self, label: str = "───", **kwargs):
        super().__init__(label, classes="chunk-boundary", **kwargs)