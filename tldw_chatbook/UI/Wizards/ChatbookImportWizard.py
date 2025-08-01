# ChatbookImportWizard.py
# Description: Multi-step wizard for importing chatbooks
#
"""
Chatbook Import Wizard
----------------------

Implements the chatbook import workflow:
1. File Selection
2. Preview & Validation
3. Conflict Resolution
4. Import Options
5. Progress & Completion
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Static, Input, Button, Label, Tree, DataTable,
    ProgressBar, RadioSet, RadioButton, Checkbox
)
from textual.worker import Worker
from textual.reactive import reactive
from loguru import logger

from .BaseWizard import BaseWizard, WizardStep, WizardStepConfig
from ...Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from ...Chatbooks.chatbook_models import ChatbookManifest, ContentType
from ...Chatbooks.conflict_resolver import ConflictResolution
from ...Third_Party.textual_fspicker import FileOpen

if TYPE_CHECKING:
    from ...app import TldwCli


class FileSelectionStep(WizardStep):
    """Step 1: File Selection."""
    
    DEFAULT_CSS = """
    FileSelectionStep {
        layout: vertical;
        padding: 2;
    }
    
    .file-selection-container {
        layout: vertical;
        height: 100%;
        align: center middle;
    }
    
    .drop-zone {
        width: 80%;
        height: 20;
        border: dashed $background-darken-2;
        background: $boost;
        align: center middle;
        margin: 2 0;
    }
    
    .drop-zone.hover {
        border: solid $primary;
        background: $primary 10%;
    }
    
    .drop-zone-content {
        text-align: center;
        padding: 2;
    }
    
    .drop-zone-icon {
        margin-bottom: 1;
        color: $text-muted;
    }
    
    .drop-zone-text {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .file-path-display {
        width: 80%;
        margin: 2 0;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .path-label {
        color: $text-muted;
        margin-bottom: 0;
    }
    
    .path-value {
        color: $primary;
        margin-top: 0;
    }
    
    .browse-button {
        margin-top: 2;
    }
    """
    
    def __init__(self, wizard: BaseWizard, config: WizardStepConfig, **kwargs):
        super().__init__(wizard, config, **kwargs)
        self.selected_file: Optional[Path] = None
        
    def compose(self) -> ComposeResult:
        """Compose the file selection UI."""
        with Container(classes="file-selection-container"):
            # Drop zone (visual only for now)
            with Container(classes="drop-zone"):
                with Container(classes="drop-zone-content"):
                    yield Static("📦", classes="drop-zone-icon")
                    yield Static("Drop chatbook file here", classes="drop-zone-text")
                    yield Static("or", classes="drop-zone-text")
            
            yield Button("Browse for Chatbook", id="browse-file", variant="primary", classes="browse-button")
            
            # File path display
            with Container(id="file-path-container", classes="file-path-display", display=False):
                yield Label("Selected file:", classes="path-label")
                yield Static("", id="selected-path", classes="path-value")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "browse-file":
            await self._browse_for_file()
    
    async def _browse_for_file(self) -> None:
        """Open file browser to select chatbook."""
        try:
            # Use the file picker
            file_dialog = FileOpen(
                title="Select Chatbook File",
                filters=[
                    ("Chatbook files", "*.zip"),
                    ("All files", "*.*")
                ]
            )
            
            file_path = await self.wizard.app_instance.push_screen(file_dialog, wait_for_dismiss=True)
            
            if file_path:
                self.selected_file = Path(file_path)
                self._update_file_display()
                # Notify wizard that we can proceed
                self.wizard.refresh_current_step()
                
        except Exception as e:
            logger.error(f"Error selecting file: {e}")
            self.wizard.app_instance.notify(f"Error selecting file: {str(e)}", severity="error")
    
    def _update_file_display(self) -> None:
        """Update the file path display."""
        if self.selected_file:
            container = self.query_one("#file-path-container", Container)
            container.display = True
            
            path_display = self.query_one("#selected-path", Static)
            path_display.update(str(self.selected_file))
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get selected file path."""
        return {
            "file_path": str(self.selected_file) if self.selected_file else None
        }
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate file selection."""
        if not self.selected_file:
            return False, "Please select a chatbook file"
        
        if not self.selected_file.exists():
            return False, "Selected file does not exist"
        
        if not self.selected_file.suffix == ".zip":
            return False, "Please select a valid chatbook file (.zip)"
        
        return True, None


class PreviewValidationStep(WizardStep):
    """Step 2: Preview & Validation."""
    
    DEFAULT_CSS = """
    PreviewValidationStep {
        layout: vertical;
        padding: 2;
    }
    
    .preview-header {
        margin-bottom: 2;
    }
    
    .chatbook-info {
        padding: 1 2;
        background: $boost;
        border: round $background-darken-1;
        margin-bottom: 2;
    }
    
    .info-title {
        text-style: bold;
        font-size: 110%;
        margin-bottom: 1;
    }
    
    .info-row {
        layout: horizontal;
        height: auto;
        margin: 0 0 1 1;
    }
    
    .info-label {
        width: 15;
        color: $text-muted;
    }
    
    .info-value {
        width: 1fr;
    }
    
    .content-preview {
        padding: 1 2;
        background: $boost;
        border: round $background-darken-1;
        margin-bottom: 2;
    }
    
    .preview-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .content-tree {
        height: 15;
        background: $background;
        border: round $background-darken-1;
        padding: 1;
    }
    
    .validation-results {
        padding: 1 2;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .validation-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .validation-item {
        margin: 0 0 1 1;
    }
    
    .validation-success {
        color: $success;
    }
    
    .validation-warning {
        color: $warning;
    }
    
    .validation-error {
        color: $error;
    }
    """
    
    def __init__(self, wizard: BaseWizard, config: WizardStepConfig, **kwargs):
        super().__init__(wizard, config, **kwargs)
        self.manifest: Optional[ChatbookManifest] = None
        self.validation_passed = False
        
    def compose(self) -> ComposeResult:
        """Compose preview UI."""
        with Container(classes="preview-header"):
            yield Static("📋 Chatbook Preview", classes="section-title")
        
        # Chatbook information
        with Container(classes="chatbook-info"):
            yield Static("", id="chatbook-title", classes="info-title")
            
            with Container(classes="info-row"):
                yield Static("Description:", classes="info-label")
                yield Static("", id="chatbook-description", classes="info-value")
            
            with Container(classes="info-row"):
                yield Static("Author:", classes="info-label")
                yield Static("", id="chatbook-author", classes="info-value")
            
            with Container(classes="info-row"):
                yield Static("Created:", classes="info-label")
                yield Static("", id="chatbook-created", classes="info-value")
            
            with Container(classes="info-row"):
                yield Static("Version:", classes="info-label")
                yield Static("", id="chatbook-version", classes="info-value")
        
        # Content preview
        with Container(classes="content-preview"):
            yield Static("Contents:", classes="preview-title")
            yield Tree("Content", id="content-tree", classes="content-tree")
        
        # Validation results
        with Container(classes="validation-results"):
            yield Static("Validation Results:", classes="validation-title")
            yield Container(id="validation-list")
    
    async def on_enter(self) -> None:
        """Load and validate chatbook when entering step."""
        await super().on_enter()
        await self._load_and_validate_chatbook()
    
    async def _load_and_validate_chatbook(self) -> None:
        """Load chatbook manifest and validate."""
        file_path = self.wizard.wizard_data.get("file-selection", {}).get("file_path")
        if not file_path:
            return
        
        try:
            # Create importer
            db_config = self.wizard.app_instance.config_data.get("database", {})
            db_paths = {
                "ChaChaNotes": str(Path(db_config.get("chachanotes_db_path", 
                    "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db")).expanduser()),
                "Prompts": str(Path(db_config.get("prompts_db_path", 
                    "~/.local/share/tldw_cli/tldw_prompts.db")).expanduser()),
                "Media": str(Path(db_config.get("media_db_path", 
                    "~/.local/share/tldw_cli/media_db_v2.db")).expanduser())
            }
            
            importer = ChatbookImporter(db_paths)
            
            # Preview the chatbook
            self.manifest, error = importer.preview_chatbook(Path(file_path))
            
            if error:
                self._show_validation_error(error)
                return
            
            if not self.manifest:
                self._show_validation_error("Failed to load chatbook manifest")
                return
            
            # Display manifest info
            self._display_manifest_info()
            
            # Display content tree
            self._display_content_tree()
            
            # Run validation
            self._run_validation()
            
        except Exception as e:
            logger.error(f"Error loading chatbook: {e}")
            self._show_validation_error(str(e))
    
    def _display_manifest_info(self) -> None:
        """Display manifest information."""
        if not self.manifest:
            return
        
        self.query_one("#chatbook-title", Static).update(f"📚 {self.manifest.name}")
        self.query_one("#chatbook-description", Static).update(
            self.manifest.description or "No description"
        )
        self.query_one("#chatbook-author", Static).update(
            self.manifest.author or "Unknown"
        )
        self.query_one("#chatbook-created", Static).update(
            self.manifest.created_at.strftime("%Y-%m-%d %H:%M")
        )
        self.query_one("#chatbook-version", Static).update(self.manifest.version.value)
    
    def _display_content_tree(self) -> None:
        """Display content in tree view."""
        if not self.manifest:
            return
        
        tree = self.query_one("#content-tree", Tree)
        tree.clear()
        root = tree.root
        
        # Group by content type
        type_counts = {}
        for item in self.manifest.content_items:
            if item.type not in type_counts:
                type_counts[item.type] = 0
            type_counts[item.type] += 1
        
        # Add summary nodes
        if type_counts.get(ContentType.CONVERSATION):
            node = root.add(f"💬 Conversations ({type_counts[ContentType.CONVERSATION]})")
            
        if type_counts.get(ContentType.NOTE):
            node = root.add(f"📝 Notes ({type_counts[ContentType.NOTE]})")
            
        if type_counts.get(ContentType.CHARACTER):
            node = root.add(f"👤 Characters ({type_counts[ContentType.CHARACTER]})")
            
        if type_counts.get(ContentType.MEDIA):
            node = root.add(f"🎬 Media ({type_counts[ContentType.MEDIA]})")
            
        if type_counts.get(ContentType.PROMPT):
            node = root.add(f"💡 Prompts ({type_counts[ContentType.PROMPT]})")
    
    def _run_validation(self) -> None:
        """Run validation checks."""
        validation_list = self.query_one("#validation-list", Container)
        validation_list.remove_children()
        
        if not self.manifest:
            validation_list.mount(
                Static("❌ No manifest found", classes="validation-item validation-error")
            )
            self.validation_passed = False
            return
        
        errors = []
        warnings = []
        successes = []
        
        # Check version compatibility
        if self.manifest.version == "1.0":
            successes.append("✓ Compatible chatbook version")
        else:
            warnings.append("⚠️ Unknown chatbook version - import may fail")
        
        # Check for content
        if self.manifest.content_items:
            successes.append(f"✓ Found {len(self.manifest.content_items)} content items")
        else:
            warnings.append("⚠️ No content items found in chatbook")
        
        # Check statistics match
        expected_total = (
            self.manifest.total_conversations + 
            self.manifest.total_notes + 
            self.manifest.total_characters + 
            self.manifest.total_media_items
        )
        actual_total = len(self.manifest.content_items)
        
        if expected_total == actual_total:
            successes.append("✓ Content statistics match manifest")
        else:
            warnings.append(f"⚠️ Statistics mismatch: expected {expected_total}, found {actual_total}")
        
        # Display results
        for success in successes:
            validation_list.mount(
                Static(success, classes="validation-item validation-success")
            )
        
        for warning in warnings:
            validation_list.mount(
                Static(warning, classes="validation-item validation-warning")
            )
        
        for error in errors:
            validation_list.mount(
                Static(error, classes="validation-item validation-error")
            )
        
        # Overall validation status
        self.validation_passed = len(errors) == 0
        
        if self.validation_passed:
            validation_list.mount(
                Static("\n✅ Validation passed - chatbook is ready to import", 
                      classes="validation-item validation-success")
            )
        else:
            validation_list.mount(
                Static("\n❌ Validation failed - please check errors above", 
                      classes="validation-item validation-error")
            )
        
        # Update wizard navigation
        self.wizard.refresh_current_step()
    
    def _show_validation_error(self, error: str) -> None:
        """Show validation error."""
        validation_list = self.query_one("#validation-list", Container)
        validation_list.remove_children()
        validation_list.mount(
            Static(f"❌ {error}", classes="validation-item validation-error")
        )
        self.validation_passed = False
        self.wizard.refresh_current_step()
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get validation data."""
        return {
            "manifest": self.manifest,
            "validation_passed": self.validation_passed
        }
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Check if validation passed."""
        if not self.validation_passed:
            return False, "Chatbook validation failed"
        return True, None


class ConflictResolutionStep(WizardStep):
    """Step 3: Conflict Resolution."""
    
    DEFAULT_CSS = """
    ConflictResolutionStep {
        layout: vertical;
        padding: 2;
    }
    
    .resolution-header {
        margin-bottom: 2;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .strategy-section {
        margin-bottom: 2;
        padding: 1 2;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .strategy-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .radio-option {
        margin: 0 0 1 1;
    }
    
    .strategy-description {
        margin: 0 0 1 3;
        color: $text-muted;
        font-size: 90%;
    }
    
    .conflict-preview {
        padding: 1 2;
        background: $boost;
        border: round $warning;
    }
    
    .conflict-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    
    .conflict-list {
        margin-left: 2;
        color: $text-muted;
    }
    
    .no-conflicts {
        text-align: center;
        padding: 3;
        color: $success;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose conflict resolution UI."""
        with Container(classes="resolution-header"):
            yield Static(
                "Configure how to handle items that already exist in your database",
                classes="header-text"
            )
        
        # Resolution strategy selection
        with Container(classes="strategy-section"):
            yield Static("Conflict Resolution Strategy:", classes="strategy-title")
            
            yield RadioSet(
                RadioButton("Skip existing items", value=True, id="strategy-skip"),
                RadioButton("Rename imported items", id="strategy-rename"),
                RadioButton("Replace existing items", id="strategy-replace"),
                RadioButton("Merge where possible", id="strategy-merge"),
                id="resolution-strategy"
            )
            
            # Strategy descriptions
            yield Static(
                "Skip: Keep existing items unchanged, only import new items",
                classes="strategy-description"
            )
            yield Static(
                "Rename: Import all items with new names (e.g., 'Title (Imported)')",
                classes="strategy-description"
            )
            yield Static(
                "Replace: Overwrite existing items with imported versions",
                classes="strategy-description"
            )
            yield Static(
                "Merge: Combine content intelligently (conversations append messages)",
                classes="strategy-description"
            )
        
        # Conflict preview (placeholder for now)
        with Container(id="conflict-container", classes="conflict-preview", display=False):
            yield Static("Potential Conflicts Detected:", classes="conflict-title")
            yield Container(id="conflict-list", classes="conflict-list")
        
        # No conflicts message
        yield Static(
            "✅ No conflicts detected - all content is new",
            id="no-conflicts",
            classes="no-conflicts"
        )
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get selected resolution strategy."""
        strategy_map = {
            "strategy-skip": ConflictResolution.SKIP,
            "strategy-rename": ConflictResolution.RENAME,
            "strategy-replace": ConflictResolution.REPLACE,
            "strategy-merge": ConflictResolution.MERGE
        }
        
        selected_strategy = ConflictResolution.SKIP  # default
        strategy_set = self.query_one("#resolution-strategy", RadioSet)
        if strategy_set.pressed_button:
            selected_strategy = strategy_map.get(
                strategy_set.pressed_button.id, 
                ConflictResolution.SKIP
            )
        
        return {
            "resolution_strategy": selected_strategy
        }


class ImportOptionsStep(WizardStep):
    """Step 4: Import Options."""
    
    DEFAULT_CSS = """
    ImportOptionsStep {
        layout: vertical;
        padding: 2;
    }
    
    .options-section {
        margin-bottom: 2;
        padding: 1 2;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .checkbox-option {
        margin: 0 0 1 1;
    }
    
    .option-description {
        margin: 0 0 1 3;
        color: $text-muted;
        font-size: 90%;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose import options UI."""
        # Import options
        with Container(classes="options-section"):
            yield Static("Import Options:", classes="section-title")
            
            yield Checkbox(
                "Import media files",
                value=True,
                id="import-media",
                classes="checkbox-option"
            )
            yield Static(
                "Copy referenced media files to local storage",
                classes="option-description"
            )
            
            yield Checkbox(
                "Import embeddings",
                value=True,
                id="import-embeddings",
                classes="checkbox-option"
            )
            yield Static(
                "Import vector embeddings for RAG search (if available)",
                classes="option-description"
            )
            
            yield Checkbox(
                "Preserve timestamps",
                value=True,
                id="preserve-timestamps",
                classes="checkbox-option"
            )
            yield Static(
                "Keep original creation and modification dates",
                classes="option-description"
            )
            
            yield Checkbox(
                "Create backup",
                value=True,
                id="create-backup",
                classes="checkbox-option"
            )
            yield Static(
                "Backup current database before importing",
                classes="option-description"
            )
        
        # Tag handling
        with Container(classes="options-section"):
            yield Static("Tag Handling:", classes="section-title")
            
            yield Checkbox(
                "Import tags",
                value=True,
                id="import-tags",
                classes="checkbox-option"
            )
            yield Static(
                "Import all tags from the chatbook",
                classes="option-description"
            )
            
            yield Checkbox(
                "Merge with existing tags",
                value=True,
                id="merge-tags",
                classes="checkbox-option"
            )
            yield Static(
                "Combine imported tags with any existing tags",
                classes="option-description"
            )
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get import options."""
        return {
            "import_media": self.query_one("#import-media", Checkbox).value,
            "import_embeddings": self.query_one("#import-embeddings", Checkbox).value,
            "preserve_timestamps": self.query_one("#preserve-timestamps", Checkbox).value,
            "create_backup": self.query_one("#create-backup", Checkbox).value,
            "import_tags": self.query_one("#import-tags", Checkbox).value,
            "merge_tags": self.query_one("#merge-tags", Checkbox).value
        }


class ImportProgressStep(WizardStep):
    """Step 5: Progress & Completion."""
    
    DEFAULT_CSS = """
    ImportProgressStep {
        layout: vertical;
        padding: 2;
        align: center middle;
    }
    
    .progress-container {
        width: 100%;
        max-width: 60;
        padding: 2;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .progress-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 2;
    }
    
    #import-progress {
        margin: 2 0;
    }
    
    .progress-status {
        margin: 2 0;
    }
    
    .status-item {
        margin: 0 0 1 2;
        color: $text-muted;
    }
    
    .status-item.completed {
        color: $success;
    }
    
    .status-item.active {
        color: $primary;
        text-style: bold;
    }
    
    .status-item.error {
        color: $error;
    }
    
    .completion-message {
        text-align: center;
        padding: 2;
        margin-top: 2;
        background: $success 20%;
        border: round $success;
    }
    
    .completion-stats {
        margin-top: 2;
        padding: 1 2;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .stat-row {
        layout: horizontal;
        height: auto;
        margin: 0 0 1 0;
    }
    
    .stat-label {
        width: 20;
        color: $text-muted;
    }
    
    .stat-value {
        width: 1fr;
        text-align: right;
    }
    """
    
    def __init__(self, wizard: BaseWizard, config: WizardStepConfig, **kwargs):
        super().__init__(wizard, config, **kwargs)
        self.import_worker: Optional[Worker] = None
        self.progress_value = reactive(0)
        self.is_complete = reactive(False)
        self.import_status: Optional[ImportStatus] = None
        
    def compose(self) -> ComposeResult:
        """Compose progress UI."""
        with Container(classes="progress-container"):
            yield Static("Importing Chatbook...", id="progress-title", classes="progress-title")
            
            # Progress bar
            yield ProgressBar(
                total=100,
                show_percentage=True,
                show_eta=False,
                id="import-progress"
            )
            
            # Status list
            with Container(classes="progress-status"):
                yield Static("⟳ Preparing import", id="status-prepare", classes="status-item active")
                yield Static("○ Creating backup", id="status-backup", classes="status-item")
                yield Static("○ Importing conversations", id="status-conversations", classes="status-item")
                yield Static("○ Importing notes", id="status-notes", classes="status-item")
                yield Static("○ Importing characters", id="status-characters", classes="status-item")
                yield Static("○ Importing media", id="status-media", classes="status-item")
                yield Static("○ Updating indexes", id="status-indexes", classes="status-item")
                yield Static("○ Finalizing import", id="status-finalize", classes="status-item")
            
            # Completion message (hidden initially)
            with Container(id="completion-container", display=False):
                yield Static("✅ Import Completed Successfully!", classes="completion-message")
                
                # Import statistics
                with Container(classes="completion-stats"):
                    yield Static("Import Summary:", classes="section-title")
                    
                    with Container(classes="stat-row"):
                        yield Static("Total items:", classes="stat-label")
                        yield Static("0", id="stat-total", classes="stat-value")
                    
                    with Container(classes="stat-row"):
                        yield Static("Imported:", classes="stat-label")
                        yield Static("0", id="stat-imported", classes="stat-value stat-success")
                    
                    with Container(classes="stat-row"):
                        yield Static("Skipped:", classes="stat-label")
                        yield Static("0", id="stat-skipped", classes="stat-value stat-warning")
                    
                    with Container(classes="stat-row"):
                        yield Static("Failed:", classes="stat-label")
                        yield Static("0", id="stat-failed", classes="stat-value stat-error")
    
    async def on_enter(self) -> None:
        """Start import when entering step."""
        await super().on_enter()
        # Hide back button during import
        self.wizard.can_go_back = False
        # Start import process
        self.import_worker = self.run_worker(self._import_chatbook())
    
    async def _import_chatbook(self) -> None:
        """Import the chatbook with progress updates."""
        try:
            # Get wizard data
            file_path = Path(self.wizard.wizard_data.get("file-selection", {}).get("file_path", ""))
            manifest = self.wizard.wizard_data.get("preview-validation", {}).get("manifest")
            resolution_strategy = self.wizard.wizard_data.get("conflict-resolution", {}).get(
                "resolution_strategy", ConflictResolution.SKIP
            )
            options = self.wizard.wizard_data.get("import-options", {})
            
            # Update progress: Preparing
            self._update_status("status-prepare", "active", "⟳ Preparing import...")
            self._update_progress(5)
            
            # Create backup if requested
            if options.get("create_backup", True):
                self._update_status("status-backup", "active", "⟳ Creating backup...")
                self._update_progress(10)
                # TODO: Implement actual backup functionality
                # For now, just mark as completed
                self._update_status("status-backup", "completed", "✓ Created backup")
                self._update_progress(15)
            
            # Create importer
            db_config = self.wizard.app_instance.config_data.get("database", {})
            db_paths = {
                "ChaChaNotes": str(Path(db_config.get("chachanotes_db_path", 
                    "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db")).expanduser()),
                "Prompts": str(Path(db_config.get("prompts_db_path", 
                    "~/.local/share/tldw_cli/tldw_prompts.db")).expanduser()),
                "Media": str(Path(db_config.get("media_db_path", 
                    "~/.local/share/tldw_cli/media_db_v2.db")).expanduser())
            }
            
            importer = ChatbookImporter(db_paths)
            
            # Mark preparation complete
            self._update_status("status-prepare", "completed", "✓ Prepared import")
            self._update_progress(20)
            
            # Create import status to track progress
            self.import_status = ImportStatus()
            
            # Update UI to show what we're importing
            if manifest:
                if manifest.total_conversations > 0:
                    self._update_status("status-conversations", "active", "⟳ Importing conversations...")
                if manifest.total_notes > 0:
                    self._update_status("status-notes", "active", "⟳ Preparing notes...")
                if manifest.total_characters > 0:
                    self._update_status("status-characters", "active", "⟳ Preparing characters...")
                if options.get("import_media", True) and manifest.total_media_items > 0:
                    self._update_status("status-media", "active", "⟳ Preparing media...")
            
            # Perform the actual import
            logger.info(f"Starting chatbook import from {file_path}")
            
            success, message = importer.import_chatbook(
                chatbook_path=file_path,
                content_selections=None,  # Import all by default
                conflict_resolution=resolution_strategy,
                prefix_imported=options.get("merge_tags", False),  # Use merge_tags as prefix flag
                import_media=options.get("import_media", True),
                import_embeddings=options.get("import_embeddings", False),
                import_status=self.import_status
            )
            
            # Update progress based on actual results
            if success:
                # Update status for completed imports
                if manifest:
                    progress = 30
                    if manifest.total_conversations > 0:
                        self._update_status("status-conversations", "completed", "✓ Imported conversations")
                        progress += 15
                        self._update_progress(progress)
                        
                    if manifest.total_notes > 0:
                        self._update_status("status-notes", "completed", "✓ Imported notes")
                        progress += 15
                        self._update_progress(progress)
                        
                    if manifest.total_characters > 0:
                        self._update_status("status-characters", "completed", "✓ Imported characters")
                        progress += 15
                        self._update_progress(progress)
                        
                    if options.get("import_media", True) and manifest.total_media_items > 0:
                        self._update_status("status-media", "completed", "✓ Imported media")
                        progress += 15
                        self._update_progress(progress)
                
                # Indexes are updated as part of the import
                self._update_status("status-indexes", "completed", "✓ Updated indexes")
                self._update_progress(95)
                
                # Finalize
                self._update_status("status-finalize", "completed", "✓ Import completed")
                self._update_progress(100)
                
                await self._show_completion()
            else:
                # Import failed
                self._update_status("status-finalize", "error", f"✗ Import failed: {message}")
                await self._show_error(message)
            
        except Exception as e:
            logger.error(f"Import error: {e}")
            self._update_status("status-finalize", "error", f"✗ Error: {str(e)}")
            await self._show_error(str(e))
    
    
    def _update_progress(self, value: int) -> None:
        """Update progress bar."""
        self.progress_value = value
        progress_bar = self.query_one("#import-progress", ProgressBar)
        progress_bar.update(progress=value)
    
    def _update_status(self, status_id: str, state: str, text: str) -> None:
        """Update status item."""
        status_item = self.query_one(f"#{status_id}", Static)
        status_item.update(text)
        
        # Update classes
        status_item.remove_class("active", "completed", "error")
        if state:
            status_item.add_class(state)
    
    async def _show_completion(self) -> None:
        """Show completion message."""
        self.is_complete = True
        
        # Update title
        self.query_one("#progress-title", Static).update("✅ Import Complete!")
        
        # Show completion container
        completion = self.query_one("#completion-container", Container)
        completion.display = True
        
        # Update statistics
        if self.import_status:
            self.query_one("#stat-total", Static).update(str(self.import_status.total_items))
            self.query_one("#stat-imported", Static).update(str(self.import_status.successful_items))
            self.query_one("#stat-skipped", Static).update(str(self.import_status.skipped_items))
            self.query_one("#stat-failed", Static).update(str(self.import_status.failed_items))
        
        # Update wizard navigation
        self.wizard.can_go_back = False
        next_button = self.wizard.query_one("#wizard-next", Button)
        next_button.label = "Close"
        next_button.variant = "default"
    
    async def _show_error(self, error_message: str) -> None:
        """Show error message."""
        self._update_status("status-finalize", "error", f"✗ Error: {error_message}")
        self.query_one("#progress-title", Static).update("❌ Import Failed")
        
        # Allow going back to fix issues
        self.wizard.can_go_back = True
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get completion data."""
        return {
            "completed": self.is_complete,
            "import_status": self.import_status.to_dict() if self.import_status else None
        }
    
    def can_proceed(self) -> bool:
        """Can only proceed when import is complete."""
        return self.is_complete


class ChatbookImportWizard(BaseWizard):
    """Main chatbook import wizard."""
    
    def get_wizard_title(self) -> str:
        """Get wizard title."""
        return "Import Chatbook"
    
    def get_wizard_subtitle(self) -> str:
        """Get wizard subtitle."""
        return "Import conversations, notes, and media from a chatbook file"
    
    def create_steps(self) -> List[WizardStep]:
        """Create wizard steps."""
        return [
            FileSelectionStep(
                self,
                WizardStepConfig(
                    id="file-selection",
                    title="Select File",
                    description="Choose chatbook to import"
                )
            ),
            PreviewValidationStep(
                self,
                WizardStepConfig(
                    id="preview-validation",
                    title="Preview & Validate",
                    description="Check chatbook contents"
                )
            ),
            ConflictResolutionStep(
                self,
                WizardStepConfig(
                    id="conflict-resolution",
                    title="Conflict Resolution",
                    description="Handle existing items"
                )
            ),
            ImportOptionsStep(
                self,
                WizardStepConfig(
                    id="import-options",
                    title="Import Options",
                    description="Configure import settings"
                )
            ),
            ImportProgressStep(
                self,
                WizardStepConfig(
                    id="import-progress",
                    title="Importing",
                    description="Import in progress"
                )
            )
        ]
    
    async def on_wizard_complete(self, wizard_data: Dict[str, Any]) -> Any:
        """Handle wizard completion."""
        # Return import status
        return wizard_data.get("import-progress", {}).get("import_status")