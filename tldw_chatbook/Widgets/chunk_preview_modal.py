"""
Modal screen for previewing document chunks with different configurations.
"""

from typing import Dict, Any, List
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Button, Label, DataTable
from textual.screen import ModalScreen
from loguru import logger

from ..Chunking.Chunk_Lib import Chunker
from ..RAG_Search.enhanced_chunking_service import EnhancedChunkingService


class ChunkPreviewModal(ModalScreen):
    """Modal to preview chunking results."""
    
    DEFAULT_CSS = """
    ChunkPreviewModal {
        align: center middle;
    }
    
    #chunk-preview-container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    
    #preview-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    #chunks-preview-table {
        height: 80%;
        margin: 1 0;
    }
    
    #chunk-stats {
        height: 3;
        margin: 1 0;
        padding: 0 1;
        border: solid $secondary;
    }
    
    .preview-actions {
        align: center middle;
        height: 3;
        margin-top: 1;
    }
    """
    
    def __init__(self, content: str, config: Dict[str, Any], media_title: str = "Document", **kwargs):
        """
        Initialize the chunk preview modal.
        
        Args:
            content: The document content to chunk
            config: Chunking configuration
            media_title: Title of the media document
        """
        super().__init__(**kwargs)
        self.content = content
        self.config = config
        self.media_title = media_title
        self.chunks = []
    
    def compose(self) -> ComposeResult:
        """Compose the modal UI."""
        with Container(id="chunk-preview-container"):
            yield Label(f"Chunk Preview: {self.media_title}", id="preview-title")
            
            # Chunks table
            with VerticalScroll():
                table = DataTable(id="chunks-preview-table")
                table.add_columns("Index", "Text Preview", "Words", "Chars", "Type")
                yield table
            
            # Statistics
            yield Static("", id="chunk-stats")
            
            # Actions
            with Horizontal(classes="preview-actions"):
                yield Button("Close", id="close-preview", variant="primary")
                yield Button("Export Preview", id="export-preview", variant="default")
    
    def on_mount(self) -> None:
        """Generate and display chunks when modal is mounted."""
        self._generate_chunks()
        self._populate_table()
        self._update_stats()
    
    def _generate_chunks(self) -> None:
        """Generate chunks based on configuration."""
        try:
            method = self.config.get('method', 'words')
            
            if method in ['hierarchical', 'structural', 'contextual']:
                # Use enhanced chunking service
                service = EnhancedChunkingService()
                structured_chunks = service.chunk_text_with_structure(
                    content=self.content,
                    chunk_size=self.config.get('chunk_size', 400),
                    chunk_overlap=self.config.get('chunk_overlap', 100),
                    method=method
                )
                
                # Convert to simpler format for display
                self.chunks = []
                for chunk in structured_chunks:
                    self.chunks.append({
                        'text': chunk.text,
                        'index': chunk.chunk_index,
                        'word_count': chunk.word_count,
                        'char_count': chunk.char_count,
                        'type': chunk.chunk_type.value,
                        'metadata': chunk.metadata
                    })
            else:
                # Use basic chunker
                chunker = Chunker()
                chunker.options['max_size'] = self.config.get('chunk_size', 400)
                chunker.options['overlap'] = self.config.get('chunk_overlap', 100)
                
                chunk_results = chunker.chunk_text(self.content, method=method)
                
                # Convert to consistent format
                self.chunks = []
                for i, chunk in enumerate(chunk_results):
                    if isinstance(chunk, dict):
                        self.chunks.append(chunk)
                    else:
                        # Convert string to dict format
                        self.chunks.append({
                            'text': chunk,
                            'index': i,
                            'word_count': len(chunk.split()),
                            'char_count': len(chunk),
                            'type': 'text',
                            'metadata': {}
                        })
                        
        except Exception as e:
            logger.error(f"Error generating chunks: {e}")
            self.chunks = [{
                'text': f"Error: {str(e)}",
                'index': 0,
                'word_count': 0,
                'char_count': 0,
                'type': 'error',
                'metadata': {}
            }]
    
    def _populate_table(self) -> None:
        """Populate the data table with chunks."""
        table = self.query_one("#chunks-preview-table", DataTable)
        
        for chunk in self.chunks:
            # Truncate text for preview
            text_preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
            text_preview = text_preview.replace('\n', ' ')  # Remove newlines for table display
            
            table.add_row(
                str(chunk.get('index', 0)),
                text_preview,
                str(chunk.get('word_count', 0)),
                str(chunk.get('char_count', 0)),
                chunk.get('type', 'text')
            )
    
    def _update_stats(self) -> None:
        """Update the statistics display."""
        stats = self.query_one("#chunk-stats", Static)
        
        total_chunks = len(self.chunks)
        total_words = sum(chunk.get('word_count', 0) for chunk in self.chunks)
        total_chars = sum(chunk.get('char_count', 0) for chunk in self.chunks)
        avg_chunk_size = total_words // total_chunks if total_chunks > 0 else 0
        
        # Check for different chunk types
        chunk_types = set(chunk.get('type', 'text') for chunk in self.chunks)
        
        stats_text = (
            f"Total Chunks: {total_chunks} | "
            f"Total Words: {total_words} | "
            f"Total Characters: {total_chars} | "
            f"Average Chunk Size: {avg_chunk_size} words | "
            f"Chunk Types: {', '.join(chunk_types)}"
        )
        
        stats.update(stats_text)
    
    @on(Button.Pressed, "#close-preview")
    def close_modal(self) -> None:
        """Close the modal."""
        self.dismiss()
    
    @on(Button.Pressed, "#export-preview")
    def export_preview(self) -> None:
        """Export the chunk preview to a file."""
        try:
            # Generate export content
            export_lines = [
                f"Chunk Preview for: {self.media_title}",
                f"Configuration: {self.config}",
                f"Total Chunks: {len(self.chunks)}",
                "=" * 80,
                ""
            ]
            
            for chunk in self.chunks:
                export_lines.extend([
                    f"Chunk {chunk.get('index', 0)}:",
                    f"Type: {chunk.get('type', 'text')}",
                    f"Words: {chunk.get('word_count', 0)}, Characters: {chunk.get('char_count', 0)}",
                    "-" * 40,
                    chunk['text'],
                    "",
                    "=" * 80,
                    ""
                ])
            
            # Save to file
            from pathlib import Path
            export_path = Path.home() / "chunk_preview_export.txt"
            
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(export_lines))
            
            # Notify user
            self.app.notify(f"Preview exported to: {export_path}", severity="information")
            
        except Exception as e:
            logger.error(f"Error exporting preview: {e}")
            self.app.notify(f"Error exporting preview: {str(e)}", severity="error")
    
    @on(DataTable.RowSelected)
    def show_chunk_detail(self, event: DataTable.RowSelected) -> None:
        """Show detailed view of selected chunk."""
        if event.row_index is not None and 0 <= event.row_index < len(self.chunks):
            chunk = self.chunks[event.row_index]
            
            # Could show a detail view or copy to clipboard
            # For now, just log it
            logger.info(f"Selected chunk {chunk.get('index', 0)}: {chunk['text'][:50]}...")