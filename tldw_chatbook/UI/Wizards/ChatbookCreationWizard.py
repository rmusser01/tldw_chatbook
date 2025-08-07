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
from textual import on
from loguru import logger

from .BaseWizard import WizardContainer, WizardStep, WizardStepConfig, WizardScreen
from ..Widgets.SmartContentTree import SmartContentTree, ContentNodeData, ContentSelectionChanged
from ...Chatbooks.chatbook_creator import ChatbookCreator
from ...Chatbooks.chatbook_models import ContentType

if TYPE_CHECKING:
    from ...app import TldwCli


class BasicInfoStep(WizardStep):
    """Step 1: Basic Information."""
    
    def __init__(self, wizard: WizardContainer, config: WizardStepConfig, **kwargs):
        super().__init__(wizard, config, **kwargs)
        self.wizard = wizard  # Ensure wizard reference is set
        logger.debug(f"BasicInfoStep initialized with wizard: {wizard}")
    
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
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate the form."""
        errors = []
        name = self.query_one("#chatbook-name", Input).value.strip()
        if not name:
            errors.append("Please enter a chatbook name")
        
        # Debug logging
        logger.debug(f"BasicInfoStep.validate: name='{name}', valid={len(errors) == 0}")
        
        return len(errors) == 0, errors
    
    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """Trigger validation when inputs change."""
        logger.debug(f"BasicInfoStep: Input changed - {event.input.id} = '{event.value}'")
        # Trigger validation in the parent wizard
        if hasattr(self, 'wizard') and self.wizard:
            logger.debug("BasicInfoStep: Triggering wizard validation")
            self.wizard.validate_step()
        else:
            logger.warning("BasicInfoStep: No wizard reference to trigger validation")
    
    @on(TextArea.Changed)
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Trigger validation when text area changes."""
        # Trigger validation in the parent wizard
        if hasattr(self, 'wizard') and self.wizard:
            self.wizard.validate_step()


class ContentSelectionStep(WizardStep):
    """Step 2: Content Selection."""
    
    
    def __init__(self, wizard: WizardContainer, config: WizardStepConfig, **kwargs):
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
                logger.debug(f"App instance found: {self.wizard.app_instance}")
                
                # Try different attributes for the main database
                main_db = None
                if hasattr(self.wizard.app_instance, 'chachanotes_db') and self.wizard.app_instance.chachanotes_db:
                    main_db = self.wizard.app_instance.chachanotes_db
                    logger.debug(f"Main DB found via chachanotes_db: {main_db}")
                elif hasattr(self.wizard.app_instance, 'db') and self.wizard.app_instance.db:
                    main_db = self.wizard.app_instance.db
                    logger.debug(f"Main DB found via db: {main_db}")
                
                if main_db:
                    
                    # Conversations
                    conversations = main_db.list_all_active_conversations(limit=100)
                    logger.debug(f"Found {len(conversations) if conversations else 0} conversations")
                    
                    for conv in conversations:
                        conv_id = conv.get('id')
                        title = conv.get('title', 'Untitled')
                        msg_count = conv.get('message_count', 0)  # If available in the result
                        
                        content_data[ContentType.CONVERSATION].append(
                            ContentNodeData(
                                type=ContentType.CONVERSATION,
                                id=str(conv_id),
                                title=title or "Untitled",
                                subtitle=f"{msg_count} messages" if msg_count else "No messages"
                            )
                        )
                    
                    # Notes
                    notes = main_db.list_notes(limit=100)
                    logger.debug(f"Found {len(notes) if notes else 0} notes")
                    
                    for note in notes:
                        note_id = note.get('id')
                        title = note.get('title', 'Untitled')
                        content = note.get('content', '')
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
                    characters = main_db.list_character_cards(limit=100)
                    logger.debug(f"Found {len(characters) if characters else 0} characters")
                    
                    for char in characters:
                        char_id = char.get('id')
                        name = char.get('name', 'Unnamed')
                        description = char.get('description', '')
                        
                        content_data[ContentType.CHARACTER].append(
                            ContentNodeData(
                                type=ContentType.CHARACTER,
                                id=str(char_id),
                                title=name,
                                subtitle=description[:50] + "..." if description and len(description) > 50 else description
                            )
                        )
                
                else:
                    logger.warning("Main DB not found or not initialized")
                
                # Prompts
                if hasattr(self.wizard.app_instance, 'prompts_db') and self.wizard.app_instance.prompts_db:
                    logger.debug(f"Prompts DB found: {self.wizard.app_instance.prompts_db}")
                    prompts = self.wizard.app_instance.prompts_db.get_all_prompts()
                    logger.debug(f"Found {len(prompts) if prompts else 0} prompts")
                    
                    for prompt in prompts:
                        content_data[ContentType.PROMPT].append(
                            ContentNodeData(
                                type=ContentType.PROMPT,
                                id=str(prompt['id']),
                                title=prompt['name'],
                                subtitle=prompt.get('details', '')[:50] + "..." if prompt.get('details') else None
                            )
                        )
                
                else:
                    logger.warning("Prompts DB not found or not initialized")
                
                # Media
                if hasattr(self.wizard.app_instance, 'media_db') and self.wizard.app_instance.media_db:
                    logger.debug(f"Media DB found: {self.wizard.app_instance.media_db}")
                    # Use get_paginated_media_list to get media items
                    media_items, total_pages, current_page, total_items = self.wizard.app_instance.media_db.get_paginated_media_list(
                        page=1, results_per_page=100
                    )
                    logger.debug(f"Found {len(media_items) if media_items else 0} media items (total: {total_items})")
                    for item in media_items:
                        media_id = item.get('id', 'unknown')
                        title = item.get('title', 'Untitled')
                        media_type = item.get('type', 'unknown')
                        
                        content_data[ContentType.MEDIA].append(
                            ContentNodeData(
                                type=ContentType.MEDIA,
                                id=str(media_id),
                                title=title,
                                subtitle=media_type
                            )
                        )
                else:
                    logger.warning("Media DB not found or not initialized")
            else:
                logger.error("Wizard or app_instance not properly initialized")
                    
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
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate content selection."""
        errors = []
        if not self.content_tree:
            errors.append("Content tree not initialized")
            return False, errors
            
        selections = self.content_tree.get_selections()
        total_selected = sum(len(items) for items in selections.values())
        
        if total_selected == 0:
            errors.append("Please select at least one item to include in the chatbook")
            
        return len(errors) == 0, errors
    
    @on(ContentSelectionChanged)
    def on_content_selection_changed(self, event: ContentSelectionChanged) -> None:
        """Handle selection changes from the content tree."""
        logger.debug("ContentSelectionStep: Content selection changed")
        # Trigger validation in the parent wizard
        if hasattr(self, 'wizard') and self.wizard:
            logger.debug("ContentSelectionStep: Triggering wizard validation")
            self.wizard.validate_step()
        else:
            logger.warning("ContentSelectionStep: No wizard reference to trigger validation")


class ExportOptionsStep(WizardStep):
    """Step 3: Export Options."""
    
    
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
    
    def on_show(self) -> None:
        """Update preview when entering step."""
        super().on_show()
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
    
    
    def __init__(self, wizard: WizardContainer, config: WizardStepConfig, **kwargs):
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
    
    def on_show(self) -> None:
        """Start export when entering step."""
        super().on_show()
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
                
            # Get config - try different attributes
            if hasattr(self.wizard.app_instance, 'app_config'):
                config = self.wizard.app_instance.app_config
            elif hasattr(self.wizard.app_instance, 'config_data'):
                config = self.wizard.app_instance.config_data
            else:
                # Fallback to loading config directly
                from ...config import load_settings
                config = load_settings()
            
            db_config = config.get("database", {})
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
            # Find and dismiss the parent screen (following the pattern from BaseWizard)
            parent_screen = self.wizard.ancestors_with_self[1] if len(self.wizard.ancestors_with_self) > 1 else None
            if parent_screen and hasattr(parent_screen, 'dismiss'):
                parent_screen.dismiss("create_another")
            
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


class ChatbookCreationWizard(WizardScreen):
    """Chatbook creation wizard screen."""
    
    def compose(self) -> ComposeResult:
        """Compose the wizard screen."""
        wizard = ChatbookCreationWizardContainer(
            self.app_instance,
            **self.wizard_kwargs
        )
        yield wizard


class ChatbookCreationWizardContainer(WizardContainer):
    """The actual chatbook creation wizard implementation."""
    
    def __init__(self, app_instance, template_data=None, **kwargs):
        # Create steps with proper config
        steps = self._create_steps()
        
        super().__init__(
            app_instance=app_instance,
            steps=steps,
            title="Create New Chatbook",
            template_data=template_data,
            on_complete=self.handle_wizard_complete,
            **kwargs
        )
        
    def _create_steps(self) -> List[WizardStep]:
        """Create wizard steps with proper configuration."""
        return [
            BasicInfoStep(
                wizard=self,
                config=WizardStepConfig(
                    id="basic-info",
                    title="Basic Information",
                    description="Enter chatbook details",
                    step_number=1
                )
            ),
            ContentSelectionStep(
                wizard=self,
                config=WizardStepConfig(
                    id="content-selection",
                    title="Select Content",
                    description="Choose items to include",
                    step_number=2
                )
            ),
            ExportOptionsStep(
                wizard=self,
                config=WizardStepConfig(
                    id="export-options",
                    title="Export Options",
                    description="Configure export settings",
                    step_number=3
                )
            ),
            PreviewConfirmStep(
                wizard=self,
                config=WizardStepConfig(
                    id="preview-confirm",
                    title="Preview & Confirm",
                    description="Review your chatbook",
                    step_number=4
                )
            ),
            ProgressStep(
                wizard=self,
                config=WizardStepConfig(
                    id="progress",
                    title="Creating Chatbook",
                    description="Export in progress",
                    step_number=5
                )
            )
        ]
    
    def handle_wizard_complete(self, wizard_data: Dict[str, Any]) -> None:
        """Handle wizard completion."""
        # The actual export happens in the ProgressStep
        # This is called when the user clicks "Close" after completion
        result = wizard_data.get("progress", {}).get("export_path")
        if self.app_instance and hasattr(self.app_instance, 'pop_screen'):
            self.app_instance.pop_screen(result)