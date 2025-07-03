"""
File extraction dialog for previewing and saving extracted files from LLM responses.
"""
from pathlib import Path
from typing import List, Dict, Optional, Callable
import logging

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Static, Input
from textual.reactive import reactive

from tldw_chatbook.Utils.file_extraction import ExtractedFile
from tldw_chatbook.Utils.path_validation import validate_filename
from tldw_chatbook.Utils.secure_temp_files import secure_temp_file

logger = logging.getLogger(__name__)


class FileExtractionDialog(ModalScreen):
    """Dialog to preview and save extracted files."""
    
    CSS = """
    FileExtractionDialog {
        align: center middle;
    }
    
    FileExtractionDialog > Vertical {
        width: 80%;
        height: 80%;
        max-width: 100;
        max-height: 40;
        border: thick $background;
        background: $surface;
        padding: 1;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        background: $boost;
        width: 100%;
    }
    
    #file-list {
        height: 15;
        margin: 1 0;
    }
    
    #file-preview {
        height: 15;
        border: solid $primary;
        padding: 1;
        margin: 1 0;
    }
    
    .dialog-buttons {
        dock: bottom;
        height: 3;
        align: center middle;
        padding: 1;
    }
    
    .dialog-buttons Button {
        margin: 0 1;
    }
    
    #filename-input {
        width: 100%;
        margin: 1 0;
    }
    """
    
    # Store the extracted files
    extracted_files: reactive[List[ExtractedFile]] = reactive([])
    selected_index: reactive[Optional[int]] = reactive(None)
    
    def __init__(
        self, 
        files: List[ExtractedFile],
        callback: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the dialog.
        
        Args:
            files: List of extracted files
            callback: Optional callback function to call with results
        """
        super().__init__(**kwargs)
        self.extracted_files = files
        self.callback = callback
        self._selected_files = set(range(len(files)))  # All selected by default
        
    def compose(self) -> ComposeResult:
        """Build the dialog UI."""
        with Vertical():
            yield Label("📎 Extracted Files", classes="dialog-title")
            
            # File list table
            file_table = DataTable(id="file-list")
            file_table.add_columns("✓", "Filename", "Type", "Size")
            yield file_table
            
            # Filename editor
            yield Label("Filename:")
            yield Input(id="filename-input", placeholder="Edit filename here...")
            
            # File preview
            yield Label("Preview:")
            with VerticalScroll(id="file-preview"):
                yield Static("", id="preview-content")
            
            # Buttons
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save Selected", id="save-selected", variant="primary")
                yield Button("Save All", id="save-all", variant="success")
                yield Button("Cancel", id="cancel", variant="error")
    
    def on_mount(self) -> None:
        """Initialize the table when mounted."""
        table = self.query_one("#file-list", DataTable)
        
        for i, file in enumerate(self.extracted_files):
            size_str = f"{len(file.content)} bytes"
            table.add_row(
                "✓" if i in self._selected_files else " ",
                file.filename,
                file.language,
                size_str,
                key=str(i)
            )
        
        # Select first file
        if self.extracted_files:
            table.cursor_type = "row"
            self.selected_index = 0
            self._update_preview(0)
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the file list."""
        if event.row_key is not None:
            index = int(event.row_key.value)
            self.selected_index = index
            self._update_preview(index)
            
            # Update filename input
            filename_input = self.query_one("#filename-input", Input)
            filename_input.value = self.extracted_files[index].filename
    
    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Handle cell clicks for checkbox toggling."""
        if event.coordinate.column == 0 and event.cell_key.row_key is not None:
            index = int(event.cell_key.row_key.value)
            table = self.query_one("#file-list", DataTable)
            
            # Toggle selection
            if index in self._selected_files:
                self._selected_files.remove(index)
                new_value = " "
            else:
                self._selected_files.add(index)
                new_value = "✓"
            
            # Update the cell
            table.update_cell(event.cell_key.row_key.value, "✓", new_value)
    
    def _update_preview(self, index: int) -> None:
        """Update the preview pane with the selected file."""
        if 0 <= index < len(self.extracted_files):
            file = self.extracted_files[index]
            preview = self.query_one("#preview-content", Static)
            
            # Truncate preview if too long
            content = file.content
            if len(content) > 5000:
                content = content[:5000] + "\n\n... (truncated)"
            
            # Add syntax highlighting hint
            preview.update(f"```{file.language}\n{content}\n```")
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filename changes."""
        if event.input.id == "filename-input" and self.selected_index is not None:
            # Update the filename in our data
            new_filename = event.value.strip()
            if new_filename and self.selected_index < len(self.extracted_files):
                self.extracted_files[self.selected_index].filename = new_filename
                
                # Update the table
                table = self.query_one("#file-list", DataTable)
                table.update_cell(str(self.selected_index), "Filename", new_filename)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.dismiss(None)
            
        elif event.button.id in ["save-selected", "save-all"]:
            # Determine which files to save
            if event.button.id == "save-all":
                files_to_save = self.extracted_files
            else:
                files_to_save = [
                    file for i, file in enumerate(self.extracted_files)
                    if i in self._selected_files
                ]
            
            if not files_to_save:
                self.app.notify("No files selected", severity="warning")
                return
            
            # Validate filenames
            for file in files_to_save:
                try:
                    file.filename = validate_filename(file.filename)
                except Exception as e:
                    self.app.notify(f"Invalid filename '{file.filename}': {e}", severity="error")
                    return
            
            # Save files
            saved_files = await self._save_files(files_to_save)
            
            # Call callback if provided
            if self.callback:
                self.callback({
                    'action': event.button.id,
                    'files': saved_files,
                    'selected_indices': list(self._selected_files)
                })
            
            # Dismiss with result
            self.dismiss({
                'action': event.button.id,
                'files': saved_files
            })
    
    async def _save_files(self, files: List[ExtractedFile]) -> List[Dict]:
        """
        Save files to the Downloads folder.
        
        Returns:
            List of dictionaries with file info including saved paths
        """
        saved_files = []
        downloads_path = Path.home() / "Downloads"
        
        for file in files:
            try:
                # Ensure unique filename
                save_path = downloads_path / file.filename
                counter = 1
                while save_path.exists():
                    stem = save_path.stem
                    suffix = save_path.suffix
                    save_path = downloads_path / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Save the file
                save_path.write_text(file.content, encoding='utf-8')
                
                saved_files.append({
                    'filename': file.filename,
                    'path': str(save_path),
                    'size': len(file.content),
                    'language': file.language
                })
                
                logger.info(f"Saved file: {save_path}")
                
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")
                self.app.notify(f"Failed to save {file.filename}: {e}", severity="error")
        
        if saved_files:
            self.app.notify(
                f"Saved {len(saved_files)} file{'s' if len(saved_files) > 1 else ''} to Downloads", 
                severity="success"
            )
        
        return saved_files