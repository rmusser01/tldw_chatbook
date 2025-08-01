# ChatbookCreationWizard.py
# Description: Multi-step wizard for creating chatbooks
#
"""
Chatbook Creation Wizard
------------------------

Implements the 5-step chatbook creation workflow:
1. Basic Information
2. Content Selection
3. Export Options
4. Preview & Confirm
5. Progress & Completion
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, Grid
from textual.widgets import (
    Static, Input, TextArea, Button, Label, 
    Checkbox, RadioSet, RadioButton, Tree, DataTable, ProgressBar
)
from textual.worker import Worker, WorkerState
from textual.reactive import reactive
from loguru import logger

from .BaseWizard import BaseWizard, WizardStep, WizardStepConfig
from ..Widgets.SmartContentTree import SmartContentTree, ContentNodeData
from ...Chatbooks.chatbook_creator import ChatbookCreator
from ...Chatbooks.chatbook_models import ContentType

if TYPE_CHECKING:
    from ...app import TldwCli


class BasicInfoStep(WizardStep):
    """Step 1: Basic Information."""
    
    DEFAULT_CSS = """
    BasicInfoStep {
        layout: vertical;
        padding: 2;
    }
    
    .form-group {
        margin-bottom: 2;
    }
    
    .form-label {
        color: $text-muted;
        margin-bottom: 0;
    }
    
    .form-input {
        width: 100%;
        margin-top: 0;
    }
    
    #chatbook-description {
        height: 6;
    }
    
    .info-box {
        padding: 1;
        background: $boost;
        border: round $primary;
        margin-top: 2;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the basic info form."""
        with Container(classes="form-group"):
            yield Label("Chatbook Name *", classes="form-label")
            yield Input(
                placeholder="My Research Collection",
                id="chatbook-name",
                classes="form-input"
            )
        
        with Container(classes="form-group"):
            yield Label("Description", classes="form-label")
            yield TextArea(
                "A collection of AI research conversations and notes...",
                id="chatbook-description",
                classes="form-input"
            )
        
        with Container(classes="form-group"):
            yield Label("Tags (comma-separated)", classes="form-label")
            yield Input(
                placeholder="research, AI, machine-learning",
                id="chatbook-tags",
                classes="form-input"
            )
        
        with Container(classes="form-group"):
            yield Label("Author", classes="form-label")
            yield Input(
                placeholder="John Doe",
                id="chatbook-author",
                classes="form-input"
            )
        
        with Container(classes="info-box"):
            yield Static(
                "📚 A chatbook packages your conversations, notes, and media "
                "into a shareable knowledge pack that others can import."
            )
    
    async def on_mount(self) -> None:
        """Set default values on mount."""
        # Try to get username from config
        if hasattr(self, 'wizard') and self.wizard and hasattr(self.wizard, 'app_instance'):
            if hasattr(self.wizard.app_instance, 'config_data'):
                username = self.wizard.app_instance.config_data.get("general", {}).get("users_name")
                if username:
                    author_input = self.query_one("#chatbook-author", Input)
                    author_input.value = username
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get form data."""
        tags_value = self.query_one("#chatbook-tags", Input).value
        tags = [tag.strip() for tag in tags_value.split(",") if tag.strip()] if tags_value else []
        
        return {
            "name": self.query_one("#chatbook-name", Input).value,
            "description": self.query_one("#chatbook-description", TextArea).text,
            "tags": tags,
            "author": self.query_one("#chatbook-author", Input).value
        }
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate the form."""
        name = self.query_one("#chatbook-name", Input).value.strip()
        if not name:
            return False, "Please enter a chatbook name"
        return True, None


class ContentSelectionStep(WizardStep):
    """Step 2: Content Selection."""
    
    DEFAULT_CSS = """
    ContentSelectionStep {
        layout: vertical;
        height: 100%;
    }
    
    .selection-header {
        height: 3;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
        margin-bottom: 1;
    }
    
    .content-tree-container {
        height: 1fr;
    }
    """
    
    def __init__(self, wizard: BaseWizard, config: WizardStepConfig, **kwargs):
        super().__init__(wizard, config, **kwargs)
        self.content_tree: Optional[SmartContentTree] = None
        
    def compose(self) -> ComposeResult:
        """Compose the content selection UI."""
        with Container(classes="selection-header"):
            yield Static(
                "Select the conversations, notes, characters, media, and prompts to include in your chatbook."
            )
        
        # Smart content tree
        self.content_tree = SmartContentTree(
            load_content=self._load_content,
            classes="content-tree-container"
        )
        yield self.content_tree
    
    def _load_content(self) -> Dict[ContentType, List[ContentNodeData]]:
        """Load available content from databases."""
        content_data = {
            ContentType.CONVERSATION: [],
            ContentType.NOTE: [],
            ContentType.CHARACTER: [],
            ContentType.MEDIA: [],
            ContentType.PROMPT: []
        }
        
        try:
            # Load from app's database connections
            if hasattr(self, 'wizard') and self.wizard and hasattr(self.wizard, 'app_instance'):
                if hasattr(self.wizard.app_instance, 'db') and self.wizard.app_instance.db:
                    # Conversations
                    conversations = self.wizard.app_instance.db.get_all_conversations()
                for conv_id, title in conversations:
                    # Get message count
                    messages = self.wizard.app_instance.db.get_messages_for_conversation(conv_id)
                    msg_count = len(messages) if messages else 0
                    
                    content_data[ContentType.CONVERSATION].append(
                        ContentNodeData(
                            type=ContentType.CONVERSATION,
                            id=str(conv_id),
                            title=title or "Untitled",
                            subtitle=f"{msg_count} messages"
                        )
                    )
                    
                    # Notes
                    notes = self.wizard.app_instance.db.get_all_notes()
                for note_id, title, content, _ in notes:
                    word_count = len(content.split()) if content else 0
                    
                    content_data[ContentType.NOTE].append(
                        ContentNodeData(
                            type=ContentType.NOTE,
                            id=str(note_id),
                            title=title or "Untitled",
                            subtitle=f"{word_count} words"
                        )
                    )
                    
                    # Characters
                    characters = self.wizard.app_instance.db.get_all_characters()
                    for char_id, name, description in characters:
                        content_data[ContentType.CHARACTER].append(
                            ContentNodeData(
                                type=ContentType.CHARACTER,
                                id=str(char_id),
                                title=name,
                                subtitle=description[:50] + "..." if description and len(description) > 50 else description
                            )
                        )
                
                # Prompts
                if hasattr(self.wizard.app_instance, 'prompts_db') and self.wizard.app_instance.prompts_db:
                    prompts = self.wizard.app_instance.prompts_db.get_all_prompts()
                    for prompt in prompts:
                        content_data[ContentType.PROMPT].append(
                            ContentNodeData(
                                type=ContentType.PROMPT,
                                id=str(prompt['id']),
                                title=prompt['name'],
                                subtitle=prompt.get('details', '')[:50] + "..." if prompt.get('details') else None
                            )
                        )
                
                # Media
                if hasattr(self.wizard.app_instance, 'media_db') and self.wizard.app_instance.media_db:
                    media_items = self.wizard.app_instance.media_db.get_all_media_items(limit=100)
                    for item in media_items:
                        if isinstance(item, dict):
                            media_id = item.get('media_id', item.get('id', 'unknown'))
                            title = item.get('title', 'Untitled')
                            media_type = item.get('media_type', 'unknown')
                        else:
                            media_id = getattr(item, 'media_id', getattr(item, 'id', 'unknown'))
                            title = getattr(item, 'title', 'Untitled')
                            media_type = getattr(item, 'media_type', 'unknown')
                        
                        content_data[ContentType.MEDIA].append(
                            ContentNodeData(
                                type=ContentType.MEDIA,
                                id=str(media_id),
                                title=title,
                                subtitle=media_type
                            )
                        )
                    
        except Exception as e:
            logger.error(f"Error loading content: {e}")
            if hasattr(self, 'wizard') and self.wizard and hasattr(self.wizard, 'app_instance'):
                self.wizard.app_instance.notify(f"Error loading content: {str(e)}", severity="error")
        
        return content_data
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get selected content."""
        if self.content_tree:
            return {"selections": self.content_tree.get_selections()}
        return {"selections": {}}
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate content selection."""
        if not self.content_tree:
            return False, "Content tree not initialized"
            
        selections = self.content_tree.get_selections()
        total_selected = sum(len(items) for items in selections.values())
        
        if total_selected == 0:
            return False, "Please select at least one item to include in the chatbook"
            
        return True, None


class ExportOptionsStep(WizardStep):
    """Step 3: Export Options."""
    
    DEFAULT_CSS = """
    ExportOptionsStep {
        layout: vertical;
        padding: 2;
    }
    
    .option-group {
        margin-bottom: 2;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .option-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .radio-option {
        margin: 0 0 1 1;
    }
    
    .checkbox-option {
        margin: 0 0 1 1;
    }
    
    .compression-level {
        margin-left: 3;
        layout: horizontal;
        height: 3;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose export options UI."""
        # Format selection
        with Container(classes="option-group"):
            yield Static("Format", classes="option-title")
            yield RadioSet(
                RadioButton("ZIP Archive (Recommended)", value=True, id="format-zip"),
                RadioButton("JSON Bundle", id="format-json"),
                RadioButton("SQLite Database", id="format-sqlite"),
                RadioButton("Markdown Collection", id="format-markdown"),
                id="export-format"
            )
        
        # Compression options
        with Container(classes="option-group"):
            yield Static("Compression", classes="option-title")
            yield Checkbox(
                "Enable compression (reduce size)",
                value=True,
                id="enable-compression",
                classes="checkbox-option"
            )
        
        # Include options
        with Container(classes="option-group"):
            yield Static("Include", classes="option-title")
            yield Checkbox(
                "Embeddings (if available)",
                value=True,
                id="include-embeddings",
                classes="checkbox-option"
            )
            yield Checkbox(
                "Media files",
                value=True,
                id="include-media",
                classes="checkbox-option"
            )
            yield Checkbox(
                "Metadata and timestamps",
                value=True,
                id="include-metadata",
                classes="checkbox-option"
            )
            yield Checkbox(
                "User preferences",
                value=False,
                id="include-preferences",
                classes="checkbox-option"
            )
        
        # Privacy options
        with Container(classes="option-group"):
            yield Static("Privacy", classes="option-title")
            yield Checkbox(
                "Anonymize user names",
                value=False,
                id="anonymize-users",
                classes="checkbox-option"
            )
            yield Checkbox(
                "Remove sensitive data",
                value=False,
                id="remove-sensitive",
                classes="checkbox-option"
            )
            yield Checkbox(
                "Include license file",
                value=True,
                id="include-license",
                classes="checkbox-option"
            )
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get export options."""
        # Get selected format
        format_map = {
            "format-zip": "zip",
            "format-json": "json",
            "format-sqlite": "sqlite",
            "format-markdown": "markdown"
        }
        
        selected_format = "zip"  # default
        format_set = self.query_one("#export-format", RadioSet)
        if format_set.pressed_button:
            selected_format = format_map.get(format_set.pressed_button.id, "zip")
        
        return {
            "format": selected_format,
            "compression": self.query_one("#enable-compression", Checkbox).value,
            "include_embeddings": self.query_one("#include-embeddings", Checkbox).value,
            "include_media": self.query_one("#include-media", Checkbox).value,
            "include_metadata": self.query_one("#include-metadata", Checkbox).value,
            "include_preferences": self.query_one("#include-preferences", Checkbox).value,
            "anonymize_users": self.query_one("#anonymize-users", Checkbox).value,
            "remove_sensitive": self.query_one("#remove-sensitive", Checkbox).value,
            "include_license": self.query_one("#include-license", Checkbox).value
        }


class PreviewConfirmStep(WizardStep):
    """Step 4: Preview & Confirm."""
    
    DEFAULT_CSS = """
    PreviewConfirmStep {
        layout: vertical;
        padding: 2;
    }
    
    .preview-header {
        margin-bottom: 2;
    }
    
    .chatbook-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .preview-section {
        margin-bottom: 2;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .content-list {
        margin-left: 2;
        color: $text-muted;
    }
    
    .file-tree {
        height: 12;
        border: round $background-darken-1;
        background: $background;
        padding: 1;
        margin: 1 0;
    }
    
    .location-section {
        margin-top: 2;
        padding: 1;
        background: $panel;
        border: round $primary;
    }
    
    .location-path {
        color: $primary;
        margin: 1 0;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose preview UI."""
        # Header
        with Container(classes="preview-header"):
            yield Static("", id="preview-title", classes="chatbook-title")
            yield Static("", id="preview-description")
        
        # Contents summary
        with Container(classes="preview-section"):
            yield Static("Contents:", classes="section-title")
            yield Static("", id="content-summary", classes="content-list")
        
        # File structure preview
        with Container(classes="preview-section"):
            yield Static("Preview:", classes="section-title")
            yield Tree("chatbook/", id="file-preview", classes="file-tree")
        
        # Export location
        with Container(classes="location-section"):
            yield Static("Export Location:", classes="section-title")
            yield Static("", id="export-path", classes="location-path")
            yield Button("Change Location", id="change-location", variant="default")
    
    async def on_enter(self) -> None:
        """Update preview when entering step."""
        await super().on_enter()
        self._update_preview()
    
    def _update_preview(self) -> None:
        """Update the preview based on wizard data."""
        # Get data from previous steps
        basic_info = self.wizard.wizard_data.get("basic-info", {})
        selections = self.wizard.wizard_data.get("content-selection", {}).get("selections", {})
        export_options = self.wizard.wizard_data.get("export-options", {})
        
        # Update header
        title = basic_info.get("name", "Untitled Chatbook")
        self.query_one("#preview-title", Static).update(f"📚 {title}")
        self.query_one("#preview-description", Static).update(
            basic_info.get("description", "No description provided")
        )
        
        # Update content summary
        summary_parts = []
        total_size = 0
        
        for content_type, items in selections.items():
            if items:
                count = len(items)
                type_name = content_type.value.title()
                summary_parts.append(f"• {count} {type_name}")
                # Estimate size (placeholder calculation)
                total_size += count * 10  # 10KB per item estimate
        
        summary_text = "\n".join(summary_parts)
        summary_text += f"\n• Total size: ~{total_size / 1024:.1f} MB (compressed)"
        self.query_one("#content-summary", Static).update(summary_text)
        
        # Update file tree preview
        tree = self.query_one("#file-preview", Tree)
        tree.clear()
        root = tree.root
        
        # Add expected file structure
        root.add("📄 manifest.json")
        root.add("📄 README.md")
        
        content_node = root.add("📁 content/")
        if selections.get(ContentType.CONVERSATION):
            conv_node = content_node.add("📁 conversations/")
            for i in range(min(3, len(selections[ContentType.CONVERSATION]))):
                conv_node.add(f"📄 conversation_{i+1}.json")
            if len(selections[ContentType.CONVERSATION]) > 3:
                conv_node.add("...")
                
        if selections.get(ContentType.NOTE):
            notes_node = content_node.add("📁 notes/")
            for i in range(min(3, len(selections[ContentType.NOTE]))):
                notes_node.add(f"📄 note_{i+1}.md")
            if len(selections[ContentType.NOTE]) > 3:
                notes_node.add("...")
        
        # Update export path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in title if c.isalnum() or c in " -_").strip()
        export_format = export_options.get("format", "zip")
        filename = f"{safe_name}_{timestamp}.{export_format}"
        export_path = Path.home() / "Documents" / "Chatbooks" / filename
        
        self.query_one("#export-path", Static).update(str(export_path))
        
        # Store export path in wizard data
        self.wizard.wizard_data["export_path"] = str(export_path)
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get confirmation data."""
        return {
            "confirmed": True,
            "export_path": self.wizard.wizard_data.get("export_path", "")
        }


class ProgressStep(WizardStep):
    """Step 5: Progress & Completion."""
    
    DEFAULT_CSS = """
    ProgressStep {
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
    
    #export-progress {
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
    
    .completion-actions {
        layout: horizontal;
        height: auto;
        align: center middle;
        margin-top: 2;
    }
    
    .completion-actions Button {
        margin: 0 1;
    }
    
    .hidden {
        display: none;
    }
    """
    
    def __init__(self, wizard: BaseWizard, config: WizardStepConfig, **kwargs):
        super().__init__(wizard, config, **kwargs)
        self.export_worker: Optional[Worker] = None
        self.progress_value = reactive(0)
        self.is_complete = reactive(False)
        self.export_path: Optional[Path] = None
        
    def compose(self) -> ComposeResult:
        """Compose progress UI."""
        with Container(classes="progress-container"):
            yield Static("Creating Chatbook...", id="progress-title", classes="progress-title")
            
            # Progress bar
            yield ProgressBar(
                total=100,
                show_percentage=True,
                show_eta=False,
                id="export-progress"
            )
            
            # Status list
            with Container(classes="progress-status"):
                yield Static("⟳ Validating content", id="status-validate", classes="status-item active")
                yield Static("○ Exporting conversations", id="status-conversations", classes="status-item")
                yield Static("○ Exporting notes", id="status-notes", classes="status-item")
                yield Static("○ Exporting characters", id="status-characters", classes="status-item")
                yield Static("○ Compressing archive", id="status-compress", classes="status-item")
                yield Static("○ Finalizing metadata", id="status-finalize", classes="status-item")
            
            # Completion message (hidden initially)
            with Container(id="completion-container", classes="completion-message hidden"):
                yield Static("✅ Chatbook Created Successfully!", classes="completion-title")
                yield Static("", id="completion-details")
                
                with Horizontal(classes="completion-actions"):
                    yield Button("Open Folder", id="open-folder", variant="primary")
                    yield Button("Share", id="share-chatbook", variant="default")
                    yield Button("Create Another", id="create-another", variant="default")
    
    async def on_enter(self) -> None:
        """Start export when entering step."""
        await super().on_enter()
        # Hide back button during export
        self.wizard.can_go_back = False
        # Start export process
        self.export_worker = self.run_worker(self._export_chatbook())
    
    async def _export_chatbook(self) -> None:
        """Export the chatbook with progress updates."""
        try:
            # Get all wizard data
            basic_info = self.wizard.wizard_data.get("basic-info", {})
            selections = self.wizard.wizard_data.get("content-selection", {}).get("selections", {})
            export_options = self.wizard.wizard_data.get("export-options", {})
            export_path = Path(self.wizard.wizard_data.get("export_path", ""))
            
            # Update progress: Validating
            self._update_status("status-validate", "active", "⟳ Validating content...")
            self._update_progress(5)
            
            # Create chatbook using the creator service
            if not hasattr(self, 'wizard') or not self.wizard or not hasattr(self.wizard, 'app_instance'):
                raise ValueError("Wizard not properly initialized")
                
            db_config = self.wizard.app_instance.config_data.get("database", {})
            db_paths = {
                "ChaChaNotes": str(Path(db_config.get("chachanotes_db_path", 
                    "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db")).expanduser()),
                "Prompts": str(Path(db_config.get("prompts_db_path", 
                    "~/.local/share/tldw_cli/tldw_prompts.db")).expanduser()),
                "Media": str(Path(db_config.get("media_db_path", 
                    "~/.local/share/tldw_cli/media_db_v2.db")).expanduser())
            }
            
            creator = ChatbookCreator(db_paths)
            
            # Ensure export directory exists
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update status before starting
            self._update_status("status-validate", "completed", "✓ Validated content")
            self._update_progress(10)
            
            # Update progress based on what's being exported
            if selections.get(ContentType.CONVERSATION):
                self._update_status("status-conversations", "active", "⟳ Exporting conversations...")
                self._update_progress(20)
            
            if selections.get(ContentType.NOTE):
                self._update_status("status-notes", "active", "⟳ Preparing notes...")
                
            if selections.get(ContentType.CHARACTER):
                self._update_status("status-characters", "active", "⟳ Preparing characters...")
                
            # Create the chatbook - this does all the actual work
            logger.info(f"Starting chatbook creation: {basic_info.get('name', 'Untitled')}")
            
            success, message, dependency_info = creator.create_chatbook(
                name=basic_info.get("name", "Untitled"),
                description=basic_info.get("description", ""),
                content_selections=selections,
                output_path=export_path,
                author=basic_info.get("author"),
                include_media=export_options.get("include_media", True),
                include_embeddings=export_options.get("include_embeddings", False),
                media_quality=export_options.get("media_quality", "thumbnail"),
                tags=basic_info.get("tags", []),
                auto_include_dependencies=not export_options.get("anonymize_users", False)
            )
            
            # Update progress after export completes
            if success:
                # Mark all selected content types as completed
                if selections.get(ContentType.CONVERSATION):
                    self._update_status("status-conversations", "completed", "✓ Exported conversations")
                    self._update_progress(40)
                    
                if selections.get(ContentType.NOTE):
                    self._update_status("status-notes", "completed", "✓ Exported notes")
                    self._update_progress(55)
                    
                if selections.get(ContentType.CHARACTER):
                    self._update_status("status-characters", "completed", "✓ Exported characters")
                    self._update_progress(70)
                
                # Archive was created by creator
                self._update_status("status-compress", "completed", "✓ Created archive")
                self._update_progress(85)
                
                # Metadata was finalized by creator
                self._update_status("status-finalize", "completed", "✓ Finalized metadata")
                self._update_progress(100)
                
                self.export_path = export_path
                await self._show_completion(export_path, dependency_info)
            else:
                # Mark failed steps
                self._update_status("status-compress", "error", "✗ Export failed")
                await self._show_error(message)
                
        except Exception as e:
            logger.error(f"Export error: {e}")
            self._update_status("status-finalize", "error", f"✗ Error: {str(e)}")
            await self._show_error(str(e))
    
    
    def _update_progress(self, value: int) -> None:
        """Update progress bar."""
        self.progress_value = value
        progress_bar = self.query_one("#export-progress", ProgressBar)
        progress_bar.update(progress=value)
    
    def _update_status(self, status_id: str, state: str, text: str) -> None:
        """Update status item."""
        status_item = self.query_one(f"#{status_id}", Static)
        status_item.update(text)
        
        # Update classes
        status_item.remove_class("active", "completed", "error")
        if state:
            status_item.add_class(state)
    
    async def _show_completion(self, export_path: Path, dependency_info: Dict[str, Any]) -> None:
        """Show completion message."""
        self.is_complete = True
        
        # Update title
        self.query_one("#progress-title", Static).update("✅ Chatbook Created Successfully!")
        
        # Show completion container
        completion = self.query_one("#completion-container", Container)
        completion.remove_class("hidden")
        
        # Update details
        size_mb = export_path.stat().st_size / (1024 * 1024)
        details = f"📦 {export_path.name}\nSize: {size_mb:.1f} MB\nLocation: {export_path.parent}"
        
        if dependency_info.get("auto_included"):
            count = len(dependency_info["auto_included"])
            details += f"\n\nAuto-included {count} character dependencies"
            
        self.query_one("#completion-details", Static).update(details)
        
        # Update wizard navigation
        self.wizard.can_go_back = False
        next_button = self.wizard.query_one("#wizard-next", Button)
        next_button.label = "Close"
        next_button.variant = "default"
    
    async def _show_error(self, error_message: str) -> None:
        """Show error message."""
        self._update_status("status-finalize", "error", f"✗ Error: {error_message}")
        self.query_one("#progress-title", Static).update("❌ Export Failed")
        
        # Allow going back to fix issues
        self.wizard.can_go_back = True
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle completion buttons."""
        button_id = event.button.id
        
        if button_id == "open-folder" and self.export_path:
            # Open the folder containing the export
            import subprocess
            import platform
            
            folder = str(self.export_path.parent)
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", folder])
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", folder])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", folder])
                
        elif button_id == "create-another":
            # Dismiss wizard to create another
            self.wizard.dismiss("create_another")
            
        elif button_id == "share-chatbook":
            if hasattr(self, 'wizard') and self.wizard and hasattr(self.wizard, 'app_instance'):
                self.wizard.app_instance.notify("Sharing functionality coming soon!", severity="info")
    
    def get_step_data(self) -> Dict[str, Any]:
        """Get completion data."""
        return {
            "completed": self.is_complete,
            "export_path": str(self.export_path) if self.export_path else None
        }
    
    def can_proceed(self) -> bool:
        """Can only proceed when export is complete."""
        return self.is_complete


class ChatbookCreationWizard(BaseWizard):
    """Main chatbook creation wizard."""
    
    def get_wizard_title(self) -> str:
        """Get wizard title."""
        return "Create New Chatbook"
    
    def get_wizard_subtitle(self) -> str:
        """Get wizard subtitle."""
        return "Package your conversations, notes, and media into a shareable chatbook"
    
    def create_steps(self) -> List[WizardStep]:
        """Create wizard steps."""
        return [
            BasicInfoStep(
                self,
                WizardStepConfig(
                    id="basic-info",
                    title="Basic Information",
                    description="Enter chatbook details"
                )
            ),
            ContentSelectionStep(
                self,
                WizardStepConfig(
                    id="content-selection",
                    title="Select Content",
                    description="Choose items to include"
                )
            ),
            ExportOptionsStep(
                self,
                WizardStepConfig(
                    id="export-options",
                    title="Export Options",
                    description="Configure export settings"
                )
            ),
            PreviewConfirmStep(
                self,
                WizardStepConfig(
                    id="preview-confirm",
                    title="Preview & Confirm",
                    description="Review your chatbook"
                )
            ),
            ProgressStep(
                self,
                WizardStepConfig(
                    id="progress",
                    title="Creating Chatbook",
                    description="Export in progress"
                )
            )
        ]
    
    async def on_wizard_complete(self, wizard_data: Dict[str, Any]) -> Any:
        """Handle wizard completion."""
        # The actual export happens in the ProgressStep
        # This is called when the user clicks "Close" after completion
        return wizard_data.get("progress", {}).get("export_path")