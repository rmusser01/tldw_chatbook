# tldw_chatbook/UI/Wizards/EmbeddingSteps.py
# Description: Wizard steps for the embeddings creation flow
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from pathlib import Path

# 3rd-Party Imports
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.reactive import reactive
from textual.widgets import Static, Button, Label, Input, Select, TextArea, Checkbox, RadioButton, RadioSet
from textual.css.query import NoMatches

# Local Imports
from .BaseWizard import WizardStep
from ...DB.ChaChaNotes_DB import CharactersRAGDB
from ...DB.Client_Media_DB_v2 import MediaDatabase

# Configure logger
logger = logger.bind(module="EmbeddingSteps")

if TYPE_CHECKING:
    from textual.app import App

########################################################################################################################
#
# Content Type Selection Components
#
########################################################################################################################

class ContentTypeCard(Container):
    """Visual card for content type selection."""
    
    is_selected = reactive(False)
    
    def __init__(
        self,
        icon: str,
        title: str,
        description: str,
        content_type: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.icon = icon
        self.title = title
        self.description = description
        self.content_type = content_type
        self.add_class("content-type-card")
        
    def compose(self) -> ComposeResult:
        """Compose the card UI."""
        yield Static(self.icon, classes="content-type-icon")
        yield Label(self.title, classes="content-type-title")
        yield Label(self.description, classes="content-type-description")
        
    def on_click(self) -> None:
        """Handle card click."""
        logger.debug(f"ContentTypeCard.on_click: {self.content_type} card clicked")
        
        # Find the parent step by walking up ancestors
        parent_step = None
        for i, ancestor in enumerate(self.ancestors):
            logger.debug(f"ContentTypeCard.on_click: ancestor[{i}] = {ancestor.__class__.__name__}")
            if hasattr(ancestor, 'select_content_type'):
                parent_step = ancestor
                logger.debug(f"ContentTypeCard.on_click: Found parent step: {parent_step.__class__.__name__}")
                break
                
        if parent_step:
            parent_step.select_content_type(self.content_type)
        else:
            logger.warning("ContentTypeCard.on_click: Could not find parent step with select_content_type method")
            
    def watch_is_selected(self, is_selected: bool) -> None:
        """React to selection changes."""
        if is_selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")


class ContentSelectionStep(WizardStep):
    """Step 1: Select what type of content to make searchable."""
    
    selected_content_type: reactive[Optional[str]] = reactive(None)
    
    def __init__(self):
        super().__init__(
            step_number=1,
            step_title="Select Content Type",
            step_description="Choose what you want to make searchable"
        )
        self.content_cards: Dict[str, ContentTypeCard] = {}
        
    def compose(self) -> ComposeResult:
        """Compose the step UI."""
        yield Label("What would you like to search?", classes="wizard-step-title")
        yield Label(
            "Choose the type of content to make searchable with AI-powered semantic search",
            classes="wizard-step-subtitle"
        )
        
        with Grid(classes="content-type-grid"):
            # Create content type cards
            cards_config = [
                ("📁", "Files", "Documents, PDFs, text files", "files"),
                ("📝", "Notes", "Your personal notes and documents", "notes"),
                ("💬", "Chats", "Conversation history", "chats"),
                ("🎥", "Media", "Videos and audio transcripts", "media"),
                ("📚", "eBooks", "EPUB and other book formats", "ebooks"),
                ("🌐", "Websites", "Web pages and online content", "websites"),
            ]
            
            for icon, title, desc, content_type in cards_config:
                card = ContentTypeCard(
                    icon=icon,
                    title=title,
                    description=desc,
                    content_type=content_type
                )
                self.content_cards[content_type] = card
                yield card
                
    def select_content_type(self, content_type: str) -> None:
        """Select a content type."""
        logger.debug(f"ContentSelectionStep.select_content_type called with: {content_type}")
        
        # Deselect all cards
        for card in self.content_cards.values():
            card.is_selected = False
            
        # Select the chosen card
        if content_type in self.content_cards:
            self.content_cards[content_type].is_selected = True
            self.selected_content_type = content_type
            logger.debug(f"Selected content type set to: {self.selected_content_type}")
            
        # Trigger validation and notify parent wizard
        self.validate()
        
        # Find parent wizard and trigger its validation
        try:
            from .BaseWizard import WizardContainer
            # Walk up the ancestor tree to find WizardContainer
            found_wizard = False
            for ancestor in self.ancestors:
                logger.debug(f"Checking ancestor: {ancestor.__class__.__name__}")
                if isinstance(ancestor, WizardContainer):
                    logger.debug("Found WizardContainer, calling validate_step()")
                    ancestor.validate_step()
                    found_wizard = True
                    break
                    
            if not found_wizard:
                logger.warning("Could not find WizardContainer ancestor")
                # Try to find any widget with validate_step method as fallback
                for ancestor in self.ancestors:
                    if hasattr(ancestor, 'validate_step'):
                        logger.debug(f"Found ancestor with validate_step: {ancestor.__class__.__name__}")
                        ancestor.validate_step()
                        found_wizard = True
                        break
                        
            if found_wizard:
                logger.info("Successfully triggered wizard validation")
            else:
                logger.warning("Could not find any ancestor with validate_step method")
                
        except Exception as e:
            logger.error(f"Exception while finding wizard to trigger validation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
    def validate(self) -> tuple[bool, List[str]]:
        """Validate that a content type is selected."""
        errors = []
        if not self.selected_content_type:
            errors.append("Please select a content type")
            
        is_valid = len(errors) == 0
        return is_valid, errors
        
    def get_data(self) -> Dict[str, Any]:
        """Get the selected content type."""
        return {
            "content_type": self.selected_content_type
        }


########################################################################################################################
#
# Content Selection Components
#
########################################################################################################################

class ContentItem(Container):
    """Individual content item in the selection list."""
    
    is_selected = reactive(False)
    
    def __init__(
        self,
        item_id: str,
        title: str,
        subtitle: str = "",
        icon: str = "📄",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.item_id = item_id
        self.title = title
        self.subtitle = subtitle
        self.icon = icon
        self.add_class("content-item")
        
    def compose(self) -> ComposeResult:
        """Compose the item UI."""
        with Horizontal(classes="content-item-row"):
            yield Checkbox(value=self.is_selected, classes="content-item-checkbox")
            yield Static(self.icon, classes="content-item-icon")
            with Vertical(classes="content-item-info"):
                yield Label(self.title, classes="content-item-title")
                if self.subtitle:
                    yield Label(self.subtitle, classes="content-item-subtitle")
                    
    @on(Checkbox.Changed)
    def handle_checkbox_change(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state change."""
        self.is_selected = event.value
        
    def watch_is_selected(self, is_selected: bool) -> None:
        """React to selection changes."""
        try:
            checkbox = self.query_one(".content-item-checkbox", Checkbox)
            checkbox.value = is_selected
        except NoMatches:
            pass


class SmartContentSelector(Container):
    """Smart content selector with search and filtering."""
    
    selected_items: reactive[List[str]] = reactive([])
    search_query = reactive("")
    
    def __init__(
        self,
        content_type: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.content_type = content_type
        self.items: Dict[str, ContentItem] = {}
        self.add_class("smart-content-selector")
        
    def compose(self) -> ComposeResult:
        """Compose the selector UI."""
        # Selection mode options
        with RadioSet(id="selection-mode", classes="selection-mode-set"):
            yield RadioButton("All items", value=True)
            yield RadioButton("Select by tags/categories")
            yield RadioButton("Search by keywords")
            yield RadioButton("Select specific items")
            
        # Search/filter input
        yield Input(
            placeholder="Search or filter...",
            id="content-search",
            classes="content-search-input hidden"
        )
        
        # Quick selection buttons
        with Horizontal(classes="quick-selection-buttons"):
            yield Button("Select All", id="select-all", variant="primary")
            yield Button("Clear All", id="clear-all", variant="default")
            yield Button("Invert", id="invert-selection", variant="default")
            
        # Preview section
        yield Label("Preview (0 items selected)", id="preview-label", classes="preview-label")
        
        # Content list
        with Container(classes="content-list-container"):
            yield Container(id="content-list", classes="content-list")
            
    def on_mount(self) -> None:
        """Load content based on type."""
        self.load_content()
        
    def load_content(self) -> None:
        """Load content items based on the content type."""
        content_list = self.query_one("#content-list", Container)
        
        if self.content_type == "notes":
            self.load_notes(content_list)
        elif self.content_type == "files":
            self.load_files(content_list)
        elif self.content_type == "media":
            self.load_media(content_list)
        # Add more content types as needed
        
    def load_notes(self, container: Container) -> None:
        """Load notes from database."""
        try:
            db = CharactersRAGDB()
            notes = db.get_all_notes()
            
            for note in notes[:50]:  # Limit to first 50 for performance
                item = ContentItem(
                    item_id=str(note['id']),
                    title=note['title'],
                    subtitle=f"Modified: {note['updated_at']}",
                    icon="📝"
                )
                self.items[str(note['id'])] = item
                container.mount(item)
                
        except Exception as e:
            logger.error(f"Error loading notes: {e}")
            container.mount(Label("Error loading notes", classes="error-message"))
            
    def load_files(self, container: Container) -> None:
        """Load files - placeholder for file picker."""
        container.mount(
            Label(
                "Click 'Browse Files' to select files to index",
                classes="placeholder-text"
            )
        )
        container.mount(
            Button("Browse Files", id="browse-files", variant="primary")
        )
        
    def load_media(self, container: Container) -> None:
        """Load media from database."""
        try:
            db = MediaDatabase()
            media_items = db.get_all_media_items()
            
            for item in media_items[:50]:  # Limit to first 50
                content_item = ContentItem(
                    item_id=str(item['id']),
                    title=item['title'],
                    subtitle=f"Type: {item['type']} | Added: {item['created_at']}",
                    icon="🎥" if item['type'] == 'video' else "🎵"
                )
                self.items[str(item['id'])] = content_item
                container.mount(content_item)
                
        except Exception as e:
            logger.error(f"Error loading media: {e}")
            container.mount(Label("Error loading media", classes="error-message"))
            
    @on(Button.Pressed, "#select-all")
    def handle_select_all(self) -> None:
        """Select all items."""
        for item in self.items.values():
            item.is_selected = True
        self.update_selection()
        
    @on(Button.Pressed, "#clear-all")
    def handle_clear_all(self) -> None:
        """Clear all selections."""
        for item in self.items.values():
            item.is_selected = False
        self.update_selection()
        
    @on(Button.Pressed, "#invert-selection")
    def handle_invert_selection(self) -> None:
        """Invert current selection."""
        for item in self.items.values():
            item.is_selected = not item.is_selected
        self.update_selection()
        
    def update_selection(self) -> None:
        """Update selected items list and preview."""
        self.selected_items = [
            item_id for item_id, item in self.items.items()
            if item.is_selected
        ]
        
        # Update preview label
        try:
            preview = self.query_one("#preview-label", Label)
            count = len(self.selected_items)
            preview.update(f"Preview ({count} items selected)")
        except NoMatches:
            pass
            
    @on(RadioSet.Changed)
    def handle_selection_mode_change(self, event: RadioSet.Changed) -> None:
        """Handle selection mode change."""
        search_input = self.query_one("#content-search", Input)
        
        if event.pressed.label.plain == "All items":
            search_input.add_class("hidden")
            self.handle_select_all()
        elif event.pressed.label.plain == "Search by keywords":
            search_input.remove_class("hidden")
            search_input.placeholder = "Enter keywords to search..."
        elif event.pressed.label.plain == "Select by tags/categories":
            search_input.remove_class("hidden")
            search_input.placeholder = "Enter tags or categories..."
        else:  # Select specific items
            search_input.add_class("hidden")


class SpecificContentStep(WizardStep):
    """Step 2: Select specific content to index."""
    
    selected_items: reactive[List[str]] = reactive([])
    
    def __init__(self, content_type: str):
        super().__init__(
            step_number=2,
            step_title="Select Content",
            step_description=f"Choose which {content_type} to make searchable"
        )
        self.content_type = content_type
        self.content_selector: Optional[SmartContentSelector] = None
        
    def compose(self) -> ComposeResult:
        """Compose the step UI."""
        yield Label(f"Select {self.content_type.title()}", classes="wizard-step-title")
        yield Label(
            f"Choose which {self.content_type} you want to include in your search collection",
            classes="wizard-step-subtitle"
        )
        
        # Create smart content selector
        self.content_selector = SmartContentSelector(self.content_type)
        yield self.content_selector
        
    def validate(self) -> tuple[bool, List[str]]:
        """Validate that at least one item is selected."""
        errors = []
        
        if self.content_selector:
            selected_count = len(self.content_selector.selected_items)
            if selected_count == 0:
                errors.append(f"Please select at least one {self.content_type[:-1]} to index")
                
        is_valid = len(errors) == 0
        return is_valid, errors
        
    def get_data(self) -> Dict[str, Any]:
        """Get the selected items."""
        if self.content_selector:
            return {
                "selected_items": self.content_selector.selected_items,
                "item_count": len(self.content_selector.selected_items)
            }
        return {"selected_items": [], "item_count": 0}


########################################################################################################################
#
# Quick Settings Components  
#
########################################################################################################################

class PresetCard(Container):
    """Visual card for preset selection."""
    
    is_selected = reactive(False)
    
    def __init__(
        self,
        name: str,
        description: str,
        icon: str,
        settings: Dict[str, Any],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.description = description
        self.icon = icon
        self.settings = settings
        self.add_class("preset-card")
        
    def compose(self) -> ComposeResult:
        """Compose the preset card."""
        yield Static(self.icon, classes="preset-icon")
        yield Label(self.name, classes="preset-name")
        yield Label(self.description, classes="preset-description")
        
        # Show key settings
        with Container(classes="preset-details"):
            if "chunk_size" in self.settings:
                yield Label(f"Chunk size: {self.settings['chunk_size']}", classes="preset-detail")
            if "model" in self.settings:
                model_name = self.settings['model'].split('/')[-1]  # Simplify model name
                yield Label(f"Model: {model_name}", classes="preset-detail")
                
    def on_click(self) -> None:
        """Handle card click."""
        parent_step = self.ancestors[1]
        if hasattr(parent_step, 'select_preset'):
            parent_step.select_preset(self.name)
            
    def watch_is_selected(self, is_selected: bool) -> None:
        """React to selection changes."""
        if is_selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")


class QuickSettingsStep(WizardStep):
    """Step 3: Quick settings with smart presets."""
    
    PRESETS = {
        "balanced": {
            "name": "Balanced",
            "description": "Good balance of speed and accuracy",
            "icon": "⚖️",
            "settings": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "model": "text-embedding-ada-002",
                "chunk_method": "sentences"
            }
        },
        "precise": {
            "name": "Precise",
            "description": "Higher accuracy, slower processing",
            "icon": "🎯",
            "settings": {
                "chunk_size": 256,
                "chunk_overlap": 100,
                "model": "text-embedding-3-small",
                "chunk_method": "semantic"
            }
        },
        "fast": {
            "name": "Fast",
            "description": "Quick processing, good for large datasets",
            "icon": "⚡",
            "settings": {
                "chunk_size": 1024,
                "chunk_overlap": 0,
                "model": "text-embedding-ada-002",
                "chunk_method": "paragraphs"
            }
        }
    }
    
    selected_preset: reactive[Optional[str]] = reactive("balanced")
    collection_name = reactive("")
    show_advanced = reactive(False)
    
    def __init__(self):
        super().__init__(
            step_number=3,
            step_title="Quick Settings",
            step_description="Configure your search collection"
        )
        self.preset_cards: Dict[str, PresetCard] = {}
        
    def compose(self) -> ComposeResult:
        """Compose the settings UI."""
        yield Label("Quick Settings", classes="wizard-step-title")
        yield Label(
            "Choose a preset configuration or customize advanced settings",
            classes="wizard-step-subtitle"
        )
        
        # Collection name input
        with Container(classes="settings-section"):
            yield Label("Collection Name", classes="settings-label")
            yield Input(
                placeholder="my_search_collection",
                id="collection-name",
                classes="settings-input"
            )
            yield Label(
                "Give your collection a memorable name",
                classes="settings-help"
            )
            
        # Preset selection
        yield Label("Search Precision", classes="settings-label")
        with Horizontal(classes="preset-cards-container"):
            for preset_id, preset_config in self.PRESETS.items():
                card = PresetCard(
                    name=preset_config["name"],
                    description=preset_config["description"],
                    icon=preset_config["icon"],
                    settings=preset_config["settings"]
                )
                if preset_id == "balanced":
                    card.is_selected = True
                self.preset_cards[preset_id] = card
                yield card
                
        # Advanced settings (collapsible)
        with Container(classes="advanced-settings-container"):
            yield Button(
                "▶ Advanced Settings (optional)",
                id="toggle-advanced",
                classes="advanced-toggle"
            )
            
            with Container(id="advanced-settings", classes="advanced-settings hidden"):
                # Model selection
                yield Label("Embedding Model", classes="settings-label")
                yield Select(
                    [
                        ("text-embedding-ada-002", "OpenAI Ada v2 (Recommended)"),
                        ("text-embedding-3-small", "OpenAI v3 Small"),
                        ("text-embedding-3-large", "OpenAI v3 Large"),
                        ("all-MiniLM-L6-v2", "Local Model (Free)")
                    ],
                    id="model-select",
                    value="text-embedding-ada-002"
                )
                
                # Chunk method
                yield Label("Chunking Method", classes="settings-label")
                yield Select(
                    [
                        ("sentences", "By Sentences"),
                        ("paragraphs", "By Paragraphs"),
                        ("words", "By Word Count"),
                        ("semantic", "Smart Semantic (Beta)")
                    ],
                    id="chunk-method-select",
                    value="sentences"
                )
                
                # Chunk size
                yield Label("Chunk Size", classes="settings-label")
                yield Input(
                    placeholder="512",
                    id="chunk-size",
                    value="512"
                )
                
                # Chunk overlap
                yield Label("Chunk Overlap", classes="settings-label")
                yield Input(
                    placeholder="50",
                    id="chunk-overlap",
                    value="50"
                )
                
    def select_preset(self, preset_name: str) -> None:
        """Select a preset configuration."""
        # Deselect all cards
        for card in self.preset_cards.values():
            card.is_selected = False
            
        # Select the chosen preset
        preset_id = preset_name.lower()
        if preset_id in self.preset_cards:
            self.preset_cards[preset_id].is_selected = True
            self.selected_preset = preset_id
            
            # Apply preset settings to advanced fields if shown
            if self.show_advanced and preset_id in self.PRESETS:
                self.apply_preset_settings(self.PRESETS[preset_id]["settings"])
                
    def apply_preset_settings(self, settings: Dict[str, Any]) -> None:
        """Apply preset settings to advanced fields."""
        try:
            if "model" in settings:
                model_select = self.query_one("#model-select", Select)
                model_select.value = settings["model"]
                
            if "chunk_method" in settings:
                method_select = self.query_one("#chunk-method-select", Select)
                method_select.value = settings["chunk_method"]
                
            if "chunk_size" in settings:
                size_input = self.query_one("#chunk-size", Input)
                size_input.value = str(settings["chunk_size"])
                
            if "chunk_overlap" in settings:
                overlap_input = self.query_one("#chunk-overlap", Input)
                overlap_input.value = str(settings["chunk_overlap"])
        except NoMatches:
            pass
            
    @on(Button.Pressed, "#toggle-advanced")
    def toggle_advanced_settings(self) -> None:
        """Toggle advanced settings visibility."""
        self.show_advanced = not self.show_advanced
        
        advanced_container = self.query_one("#advanced-settings", Container)
        toggle_btn = self.query_one("#toggle-advanced", Button)
        
        if self.show_advanced:
            advanced_container.remove_class("hidden")
            toggle_btn.label = "▼ Advanced Settings (optional)"
            
            # Apply current preset settings
            if self.selected_preset and self.selected_preset in self.PRESETS:
                self.apply_preset_settings(self.PRESETS[self.selected_preset]["settings"])
        else:
            advanced_container.add_class("hidden")
            toggle_btn.label = "▶ Advanced Settings (optional)"
            
    @on(Input.Changed, "#collection-name")
    def handle_name_change(self, event: Input.Changed) -> None:
        """Handle collection name changes."""
        self.collection_name = event.value
        
    def validate(self) -> tuple[bool, List[str]]:
        """Validate settings."""
        errors = []
        
        # Check collection name
        name_input = self.query_one("#collection-name", Input)
        if not name_input.value.strip():
            errors.append("Please enter a collection name")
        elif not name_input.value.replace("_", "").replace("-", "").isalnum():
            errors.append("Collection name must contain only letters, numbers, underscores, and hyphens")
            
        # If advanced settings are shown, validate them
        if self.show_advanced:
            try:
                chunk_size = int(self.query_one("#chunk-size", Input).value)
                if chunk_size < 100 or chunk_size > 4000:
                    errors.append("Chunk size must be between 100 and 4000")
            except (ValueError, NoMatches):
                errors.append("Invalid chunk size")
                
            try:
                overlap = int(self.query_one("#chunk-overlap", Input).value)
                if overlap < 0 or overlap > 500:
                    errors.append("Chunk overlap must be between 0 and 500")
            except (ValueError, NoMatches):
                errors.append("Invalid chunk overlap")
                
        is_valid = len(errors) == 0
        return is_valid, errors
        
    def get_data(self) -> Dict[str, Any]:
        """Get the configured settings."""
        data = {
            "collection_name": self.query_one("#collection-name", Input).value,
            "preset": self.selected_preset
        }
        
        # Get settings from preset or advanced
        if self.show_advanced:
            # Use advanced settings
            data.update({
                "model": self.query_one("#model-select", Select).value,
                "chunk_method": self.query_one("#chunk-method-select", Select).value,
                "chunk_size": int(self.query_one("#chunk-size", Input).value),
                "chunk_overlap": int(self.query_one("#chunk-overlap", Input).value)
            })
        else:
            # Use preset settings
            if self.selected_preset and self.selected_preset in self.PRESETS:
                data.update(self.PRESETS[self.selected_preset]["settings"])
                
        return data


########################################################################################################################
#
# Progress and Completion Step
#
########################################################################################################################

class ProcessingStep(WizardStep):
    """Step 4: Processing with progress visualization."""
    
    is_processing = reactive(False)
    progress_percent = reactive(0)
    current_item = reactive("")
    items_processed = reactive(0)
    total_items = reactive(0)
    
    def __init__(self):
        super().__init__(
            step_number=4,
            step_title="Creating Collection",
            step_description="Processing your content"
        )
        
    def compose(self) -> ComposeResult:
        """Compose the processing UI."""
        yield Label("Creating Your Search Collection", classes="wizard-step-title")
        
        # Progress visualization
        with Container(classes="processing-container"):
            # Main progress bar
            yield ProgressBar(
                id="main-progress",
                classes="main-progress-bar"
            )
            
            # Progress text
            yield Label(
                "Preparing to process...",
                id="progress-text",
                classes="progress-text"
            )
            
            # Current item
            yield Label(
                "",
                id="current-item",
                classes="current-item"
            )
            
            # Stats
            with Horizontal(classes="progress-stats"):
                yield Label("Items: 0/0", id="items-stat")
                yield Label("Time: 0:00", id="time-stat")
                yield Label("Speed: 0 items/sec", id="speed-stat")
                
        # What's happening explanation
        with Container(classes="explanation-box"):
            yield Label("What's happening?", classes="explanation-title")
            yield Label(
                "We're creating a searchable index of your content that understands "
                "meaning, not just keywords. This allows you to find information "
                "using natural language queries.",
                classes="explanation-text"
            )
            
    def start_processing(self, config: Dict[str, Any]) -> None:
        """Start the processing with the given configuration."""
        self.is_processing = True
        self.total_items = config.get("item_count", 0)
        
        # Update UI
        progress_text = self.query_one("#progress-text", Label)
        progress_text.update("Initializing embeddings engine...")
        
        # Start processing in background
        # (This would actually call the embedding creation logic)
        
    def update_progress(self, percent: int, current_item: str, items_done: int) -> None:
        """Update progress visualization."""
        self.progress_percent = percent
        self.current_item = current_item
        self.items_processed = items_done
        
        # Update progress bar
        progress_bar = self.query_one("#main-progress", ProgressBar)
        progress_bar.update(progress=percent)
        
        # Update text
        progress_text = self.query_one("#progress-text", Label)
        progress_text.update(f"Processing: {percent}% complete")
        
        current_label = self.query_one("#current-item", Label)
        current_label.update(f"Currently processing: {current_item}")
        
        # Update stats
        items_stat = self.query_one("#items-stat", Label)
        items_stat.update(f"Items: {items_done}/{self.total_items}")
        
    def validate(self) -> tuple[bool, List[str]]:
        """Processing step is always valid once complete."""
        if not self.is_processing:
            return True, []
        return self.progress_percent >= 100, []
        
    def get_data(self) -> Dict[str, Any]:
        """Return processing results."""
        return {
            "processing_complete": self.progress_percent >= 100,
            "items_processed": self.items_processed
        }