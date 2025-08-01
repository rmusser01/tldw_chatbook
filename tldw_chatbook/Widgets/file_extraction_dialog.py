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
            # Get file icon based on type
            icon = self._get_file_icon(file.filename, file.language)
            
            # Format file size
            size_bytes = len(file.content.encode('utf-8'))
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            table.add_row(
                "✓" if i in self._selected_files else " ",
                f"{icon} {file.filename}",
                file.language,
                size_str,
                key=str(i)
            )
        
        # Select first file
        if self.extracted_files:
            table.cursor_type = "row"
            self.selected_index = 0
            self._update_preview(0)
            
        # Update dialog title with file count
        title = self.query_one(".dialog-title", Label)
        file_count = len(self.extracted_files)
        title.update(f"📎 Extracted Files ({file_count} file{'s' if file_count != 1 else ''})")
    
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
    
    def _get_file_icon(self, filename: str, language: str) -> str:
        """Get an appropriate icon for the file type."""
        ext = Path(filename).suffix.lower()
        
        # Map extensions to icons
        icon_map = {
            # Data files
            '.csv': '📊',
            '.tsv': '📊',
            '.json': '📋',
            '.yaml': '📋',
            '.yml': '📋',
            '.xml': '📄',
            '.sql': '🗃️',
            
            # Code files
            '.py': '🐍',
            '.js': '🟨',
            '.ts': '🔷',
            '.html': '🌐',
            '.css': '🎨',
            '.java': '☕',
            '.cpp': '⚙️',
            '.c': '⚙️',
            '.go': '🐹',
            '.rs': '🦀',
            '.rb': '💎',
            '.php': '🐘',
            '.swift': '🦉',
            '.kt': '🟪',
            '.r': '📊',
            '.R': '📊',
            
            # Script files
            '.sh': '🖥️',
            '.bash': '🖥️',
            '.ps1': '🖥️',
            
            # Document files
            '.md': '📝',
            '.txt': '📄',
            '.ini': '⚙️',
            '.toml': '⚙️',
            '.env': '🔐',
            
            # New types
            '.vcf': '👤',
            '.vcard': '👤',
            '.ics': '📅',
            '.ical': '📅',
            '.gpx': '🗺️',
            '.kml': '🌍',
            '.dot': '🔀',
            '.puml': '📊',
            '.plantuml': '📊',
            '.mmd': '📊',
            '.mermaid': '📊',
            '.svg': '🎨',
            
            # Jupyter notebooks
            '.ipynb': '📓',
            
            # Infrastructure as Code
            '.tf': '🔧',
            '.tfvars': '🔧',
            
            # CI/CD
            '.yml': '⚙️',
            '.yaml': '⚙️',
            
            # API definitions
            '.proto': '🔌',
            '.graphql': '🔗',
            '.gql': '🔗',
            
            # Data formats
            '.ndjson': '📋',
            '.jsonl': '📋',
            '.parquet': '🗄️',
            '.avro': '🗄️',
            
            # System files
            '.service': '⚡',
            
            # Dependencies
            '.lock': '🔒',
            
            # Configuration files
            '.conf': '⚙️',
            '.cfg': '⚙️',
            '.properties': '⚙️',
            '.gradle': '🐘',
            '.sbt': '🔧',
            '.cmake': '🔧',
            '.pri': '🔧',
            '.pro': '🔧',
            
            # Template files
            '.hbs': '📝',
            '.handlebars': '📝',
            '.ejs': '📝',
            '.pug': '🐶',
            '.jade': '🐶',
            '.liquid': '💧',
            '.mustache': '👨',
            '.njk': '📝',
            '.j2': '📝',
            
            # Script files
            '.psm1': '💠',
            '.psd1': '💠',
            '.ps1': '💠',
            '.bat': '🖥️',
            '.cmd': '🖥️',
            '.awk': '🔧',
            '.sed': '🔧',
            '.vim': '📝',
            '.vimrc': '📝',
            '.el': '🧬',
            '.lisp': '🧬',
            '.scm': '🧬',
            '.rkt': '🧬',
            
            # Programming languages
            '.dart': '🎯',
            '.scala': '🏛️',
            '.clj': '☯️',
            '.cljs': '☯️',
            '.cljc': '☯️',
            '.ex': '💧',
            '.exs': '💧',
            '.erl': '📡',
            '.hrl': '📡',
            '.nim': '👑',
            '.nims': '👑',
            '.zig': '⚡',
            '.v': '✌️',
            '.vsh': '✌️',
            '.jl': '🔬',
            '.pas': '📐',
            '.pp': '📐',
            '.inc': '📐',
            '.hs': '🎓',
            '.lhs': '🎓',
            '.elm': '🌳',
            '.purs': '🎨',
            '.idr': '🎓',
            '.agda': '🎓',
            '.lean': '🎓',
            '.coq': '🎓',
            '.ml': '🐫',
            '.mli': '🐫',
            '.fs': '🔷',
            '.fsx': '🔷',
            '.fsi': '🔷',
            
            # Web Assembly & Low Level
            '.wat': '🔤',
            '.wasm': '🔤',
            '.ll': '🔧',
            '.s': '⚙️',
            '.asm': '⚙️',
            '.nasm': '⚙️',
            '.masm': '⚙️',
            
            # Documentation
            '.texi': '📖',
            '.texinfo': '📖',
            '.man': '📖',
            '.rdoc': '📖',
            '.pod': '📖',
            '.adoc': '📖',
            '.asciidoc': '📖',
            '.org': '📖',
            
            # Build & Project files
            '.proj': '🏗️',
            '.csproj': '🏗️',
            '.vbproj': '🏗️',
            '.fsproj': '🏗️',
            '.vcxproj': '🏗️',
            '.vcproj': '🏗️',
            '.sln': '🏗️',
            '.cabal': '🏗️',
            '.mix': '🏗️',
            '.bazel': '🏗️',
            '.bzl': '🏗️',
            '.buck': '🏗️',
            '.pants': '🏗️',
            
            # API & Testing
            '.http': '🌐',
            '.rest': '🌐',
            '.feature': '🥒',
            '.spec': '🧪',
            
            # Data formats
            '.jsonld': '🔗',
            '.geojson': '🗺️',
            '.rdf': '🔗',
            '.ttl': '🔗',
            '.xsd': '📋',
            
            # Other files
            '.env': '🔐',
            '.example': '📋',
            '.sample': '📋',
            '.tmpl': '📝',
            '.tpl': '📝',
            '.in': '📥',
            '.ac': '🔧',
            '.am': '🔧',
            '.m4': '🔧',
            '.mk': '🔧',
            '.mak': '🔧',
        }
        
        # Special cases for specific filenames
        filename_lower = filename.lower()
        if filename_lower == 'dockerfile':
            return '🐳'
        elif filename_lower == 'makefile':
            return '🔨'
        elif filename_lower == 'jenkinsfile':
            return '🔧'
        elif filename_lower == 'pipfile' or filename_lower == 'pipfile.lock':
            return '🐍'
        elif filename_lower == 'gemfile' or filename_lower == 'gemfile.lock':
            return '💎'
        elif filename_lower == '.gitignore':
            return '🚫'
        elif filename_lower == '.htaccess':
            return '🔐'
        elif filename_lower == 'requirements.txt' or filename_lower == 'requirements.in':
            return '🐍'
        elif filename_lower == 'package.json' or filename_lower == 'package-lock.json':
            return '📦'
        elif filename_lower == 'composer.json' or filename_lower == 'composer.lock':
            return '🎼'
        elif filename_lower == 'cargo.toml' or filename_lower == 'cargo.lock':
            return '🦀'
        elif filename_lower == 'docker-compose.yml' or filename_lower == 'docker-compose.yaml':
            return '🐳'
        elif filename_lower.endswith(('.github/workflows/', 'workflow.yml', 'workflow.yaml')):
            return '🎯'
        elif filename_lower == '.gitlab-ci.yml':
            return '🦊'
        elif filename_lower == '.travis.yml':
            return '🏗️'
        elif filename_lower == 'rakefile':
            return '💎'
        elif filename_lower == 'guardfile':
            return '💂'
        elif filename_lower == 'capfile':
            return '🚀'
        elif filename_lower == 'vagrantfile':
            return '📦'
        elif filename_lower == 'berksfile':
            return '👨‍🍳'
        elif filename_lower == 'appfile':
            return '📱'
        elif filename_lower == 'deliverfile':
            return '🚚'
        elif filename_lower == 'fastfile':
            return '🏃'
        elif filename_lower == 'scanfile':
            return '🔍'
        elif filename_lower == 'snapfile':
            return '📸'
        elif filename_lower == 'gymfile':
            return '🏋️'
        elif filename_lower == 'matchfile':
            return '🎯'
        elif filename_lower == 'podfile' or filename_lower == 'podfile.lock':
            return '🌱'
        elif filename_lower == 'cartfile' or filename_lower == 'cartfile.resolved':
            return '🛒'
        elif filename_lower == 'mintfile':
            return '🌿'
        elif filename_lower == 'brewfile':
            return '🍺'
        elif filename_lower == '.eslintrc.json' or filename_lower == '.eslintrc.js' or filename_lower == '.eslintrc':
            return '✔️'
        elif filename_lower == '.prettierrc' or filename_lower == '.prettierrc.json' or filename_lower == '.prettierrc.js':
            return '💅'
        elif filename_lower == '.babelrc' or filename_lower == 'babel.config.js':
            return '🐦'
        elif filename_lower == '.editorconfig':
            return '📝'
        elif filename_lower == 'jest.config.js' or filename_lower == 'jest.config.ts':
            return '🃏'
        elif filename_lower == 'webpack.config.js' or filename_lower == 'webpack.config.ts':
            return '📦'
        elif filename_lower == 'rollup.config.js' or filename_lower == 'rollup.config.ts':
            return '🎯'
        elif filename_lower == 'vite.config.js' or filename_lower == 'vite.config.ts':
            return '⚡'
        elif filename_lower == 'nginx.conf':
            return '🌐'
        elif filename_lower == 'httpd.conf' or filename_lower == 'apache2.conf':
            return '🪶'
        elif filename_lower == 'redis.conf':
            return '🔴'
        elif filename_lower == 'my.cnf' or filename_lower == 'mysql.conf':
            return '🐬'
        elif filename_lower == 'postgresql.conf' or filename_lower == 'pg_hba.conf':
            return '🐘'
        elif filename_lower == 'ssh_config' or filename_lower == 'sshd_config':
            return '🔐'
        
        return icon_map.get(ext, '📄')  # Default icon
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filename changes."""
        if event.input.id == "filename-input" and self.selected_index is not None:
            # Update the filename in our data
            new_filename = event.value.strip()
            if new_filename and self.selected_index < len(self.extracted_files):
                self.extracted_files[self.selected_index].filename = new_filename
                
                # Update the table with icon
                table = self.query_one("#file-list", DataTable)
                icon = self._get_file_icon(new_filename, self.extracted_files[self.selected_index].language)
                table.update_cell(str(self.selected_index), "Filename", f"{icon} {new_filename}")
    
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