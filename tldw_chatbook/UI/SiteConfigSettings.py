# SiteConfigSettings.py
# Description: UI component for managing per-site configurations in subscription settings
#
# Imports
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
#
# Third-Party Imports
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Button, Input, Select, Switch, Label, DataTable, TabbedContent, TabPane
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.binding import Binding
from textual.events import Mount
from rich.table import Table
from rich import box
from loguru import logger
#
# Local Imports
from ..Widgets.form_components import FormBuilder
from ..Subscriptions.site_config_manager import get_site_config_manager, SiteConfig
#
########################################################################################################################
#
# Site Configuration Settings Widget
#
########################################################################################################################

class SiteConfigSettings(Container):
    """Settings widget for managing per-site configurations."""
    
    CSS = """
    SiteConfigSettings {
        height: 100%;
        overflow: hidden;
    }
    
    .site-list-container {
        height: 100%;
        width: 40%;
        border: solid $primary;
        padding: 1;
    }
    
    .site-config-container {
        height: 100%;
        width: 60%;
        border: solid $secondary;
        padding: 1;
        margin-left: 1;
    }
    
    .config-header {
        height: 3;
        background: $surface;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    .config-section {
        margin-bottom: 2;
        padding: 1;
        border: solid $accent;
    }
    
    .section-title {
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .config-table {
        margin: 1 0;
        height: auto;
        max-height: 15;
    }
    
    .action-buttons {
        dock: bottom;
        height: 3;
        padding: 1 0;
    }
    
    .preset-selector {
        width: 100%;
        margin: 1 0;
    }
    
    .stats-display {
        padding: 1;
        background: $surface;
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+n", "new_config", "New Config"),
        Binding("ctrl+d", "delete_config", "Delete Config"),
        Binding("ctrl+s", "save_config", "Save Config"),
        Binding("ctrl+e", "export_configs", "Export"),
        Binding("ctrl+i", "import_configs", "Import"),
    ]
    
    selected_domain = reactive(None)
    current_config = reactive(None)
    
    def __init__(self):
        """Initialize site config settings."""
        super().__init__()
        self.config_manager = get_site_config_manager()
        self.form_builder = FormBuilder()
        self.unsaved_changes = False
    
    def compose(self) -> ComposeResult:
        """Compose the site config UI."""
        with Horizontal():
            # Left panel - Site list
            with Container(classes="site-list-container"):
                yield Static("Configured Sites", classes="config-header")
                
                # Site list table
                site_table = DataTable(
                    id="site-list-table",
                    classes="config-table",
                    show_header=True,
                    show_row_labels=False,
                    zebra_stripes=True
                )
                site_table.add_columns("Domain", "Rate Limit", "JS", "Auth")
                yield site_table
                
                # Add new site
                yield Input(
                    placeholder="Enter domain (e.g., example.com)",
                    id="new-domain-input"
                )
                yield Button("Add Site", variant="primary", id="add-site-btn")
            
            # Right panel - Site configuration
            with ScrollableContainer(classes="site-config-container"):
                yield Static("Site Configuration", classes="config-header", id="config-title")
                
                with TabbedContent():
                    # General settings tab
                    with TabPane("General", id="general-tab"):
                        yield from self._compose_general_settings()
                    
                    # Authentication tab
                    with TabPane("Authentication", id="auth-tab"):
                        yield from self._compose_auth_settings()
                    
                    # Content extraction tab
                    with TabPane("Extraction", id="extraction-tab"):
                        yield from self._compose_extraction_settings()
                    
                    # Advanced tab
                    with TabPane("Advanced", id="advanced-tab"):
                        yield from self._compose_advanced_settings()
                
                # Action buttons
                with Horizontal(classes="action-buttons"):
                    yield Button("Save", variant="success", id="save-config-btn")
                    yield Button("Delete", variant="error", id="delete-config-btn")
                    yield Button("Apply Preset", variant="primary", id="apply-preset-btn")
                    yield Button("Export All", variant="default", id="export-btn")
    
    def _compose_general_settings(self) -> ComposeResult:
        """Compose general settings section."""
        with Container(classes="config-section"):
            yield Static("Rate Limiting", classes="section-title")
            
            # Rate limit settings
            yield self.form_builder.create_form_field(
                "Requests per minute",
                Input(
                    value="60",
                    id="rate-limit-requests",
                    type="number"
                )
            )
            
            yield self.form_builder.create_form_field(
                "Concurrent requests",
                Input(
                    value="1",
                    id="concurrent-requests",
                    type="number"
                )
            )
            
            yield Static("Request Settings", classes="section-title")
            
            yield self.form_builder.create_form_field(
                "Timeout (seconds)",
                Input(
                    value="30",
                    id="timeout",
                    type="number"
                )
            )
            
            yield self.form_builder.create_form_field(
                "Retry count",
                Input(
                    value="3",
                    id="retry-count",
                    type="number"
                )
            )
            
            # JavaScript rendering
            yield Static("JavaScript Rendering", classes="section-title")
            
            yield self.form_builder.create_form_field(
                "Requires JavaScript",
                Switch(
                    value=False,
                    id="requires-javascript"
                )
            )
            
            yield Input(
                placeholder="CSS selector to wait for",
                id="wait-for-selector"
            )
            
            # Statistics
            yield Static("Statistics", classes="section-title")
            yield Container(id="stats-container", classes="stats-display")
    
    def _compose_auth_settings(self) -> ComposeResult:
        """Compose authentication settings section."""
        with Container(classes="config-section"):
            yield Static("Authentication Type", classes="section-title")
            
            auth_select = Select(
                options=[
                    ("none", "None"),
                    ("basic", "Basic Auth"),
                    ("bearer", "Bearer Token"),
                    ("api_key", "API Key"),
                ],
                id="auth-type-select"
            )
            yield auth_select
            
            # Basic auth fields
            with Container(id="basic-auth-container", classes="hidden"):
                yield Input(placeholder="Username", id="auth-username")
                yield Input(placeholder="Password", password=True, id="auth-password")
            
            # Bearer token field
            with Container(id="bearer-auth-container", classes="hidden"):
                yield Input(placeholder="Bearer Token", password=True, id="bearer-token")
            
            # API key fields
            with Container(id="api-key-container", classes="hidden"):
                yield Input(placeholder="Header Name (e.g., X-API-Key)", id="api-key-name")
                yield Input(placeholder="API Key Value", password=True, id="api-key-value")
            
            # Custom headers
            yield Static("Custom Headers", classes="section-title")
            yield Container(id="headers-container")
            yield Button("Add Header", id="add-header-btn")
    
    def _compose_extraction_settings(self) -> ComposeResult:
        """Compose content extraction settings section."""
        with Container(classes="config-section"):
            yield Static("Content Selectors", classes="section-title")
            
            yield Input(
                placeholder="Content CSS selector",
                id="content-selector"
            )
            
            yield Input(
                placeholder="Title CSS selector",
                id="title-selector"
            )
            
            yield Input(
                placeholder="Date CSS selector",
                id="date-selector"
            )
            
            yield Input(
                placeholder="Author CSS selector",
                id="author-selector"
            )
            
            yield Static("Exclude Selectors", classes="section-title")
            yield Container(id="exclude-selectors-container")
            yield Button("Add Exclude Selector", id="add-exclude-btn")
            
            yield Static("Change Detection", classes="section-title")
            
            yield Input(
                placeholder="Ignore selectors (comma-separated)",
                id="ignore-selectors"
            )
            
            yield self.form_builder.create_form_field(
                "Change threshold (0-1)",
                Input(
                    value="0.1",
                    id="change-threshold",
                    type="number"
                )
            )
    
    def _compose_advanced_settings(self) -> ComposeResult:
        """Compose advanced settings section."""
        with Container(classes="config-section"):
            yield Static("Content Processing", classes="section-title")
            
            yield self.form_builder.create_form_field(
                "Remove scripts",
                Switch(
                    value=True,
                    id="remove-scripts"
                )
            )
            
            yield self.form_builder.create_form_field(
                "Remove styles",
                Switch(
                    value=True,
                    id="remove-styles"
                )
            )
            
            yield self.form_builder.create_form_field(
                "Preserve links",
                Switch(
                    value=True,
                    id="preserve-links"
                )
            )
            
            yield self.form_builder.create_form_field(
                "Extract images",
                Switch(
                    value=False,
                    id="extract-images"
                )
            )
            
            yield Static("Viewport Settings", classes="section-title")
            
            yield self.form_builder.create_form_field(
                "Viewport width",
                Input(
                    value="1920",
                    id="viewport-width",
                    type="number"
                )
            )
            
            yield self.form_builder.create_form_field(
                "Viewport height",
                Input(
                    value="1080",
                    id="viewport-height",
                    type="number"
                )
            )
            
            yield Static("Notes", classes="section-title")
            yield Input(
                placeholder="Notes about this configuration",
                id="config-notes"
            )
    
    async def on_mount(self):
        """Load site configurations on mount."""
        await self.load_site_list()
    
    @work(thread=True)
    def load_site_list(self):
        """Load list of configured sites."""
        try:
            configs = self.config_manager.list_configs()
            self.call_from_thread(self.update_site_table, configs)
        except Exception as e:
            logger.error(f"Error loading site configs: {str(e)}")
    
    def update_site_table(self, configs: List[Dict[str, Any]]):
        """Update the site list table."""
        table = self.query_one("#site-list-table", DataTable)
        table.clear()
        
        for config in configs:
            table.add_row(
                config['domain'],
                config['rate_limit'],
                "✓" if config['requires_js'] else "✗",
                "✓" if config['has_auth'] else "✗"
            )
    
    @on(DataTable.RowSelected, "#site-list-table")
    async def on_site_selected(self, event: DataTable.RowSelected):
        """Handle site selection."""
        if self.unsaved_changes:
            # TODO: Show confirmation dialog
            pass
        
        table = self.query_one("#site-list-table", DataTable)
        row_data = table.get_row(event.row_key)
        domain = row_data[0]
        
        self.selected_domain = domain
        await self.load_site_config(domain)
    
    @work(thread=True)
    def load_site_config(self, domain: str):
        """Load configuration for selected site."""
        try:
            config = self.config_manager.get_config(f"https://{domain}")
            self.call_from_thread(self.display_config, config)
        except Exception as e:
            logger.error(f"Error loading config for {domain}: {str(e)}")
    
    def display_config(self, config: SiteConfig):
        """Display configuration in the form."""
        self.current_config = config
        
        # Update title
        title = self.query_one("#config-title", Static)
        title.update(f"Configuration: {config.domain}")
        
        # General settings
        self.query_one("#rate-limit-requests", Input).value = str(config.rate_limit_requests)
        self.query_one("#concurrent-requests", Input).value = str(config.concurrent_requests)
        self.query_one("#timeout", Input).value = str(config.timeout)
        self.query_one("#retry-count", Input).value = str(config.retry_count)
        
        # JavaScript settings
        self.query_one("#requires-javascript", Switch).value = config.requires_javascript
        self.query_one("#wait-for-selector", Input).value = config.wait_for_selector or ""
        
        # Authentication
        auth_select = self.query_one("#auth-type-select", Select)
        auth_select.value = config.auth_type or "none"
        self.update_auth_fields(config.auth_type)
        
        # Content extraction
        self.query_one("#content-selector", Input).value = config.content_selector or ""
        self.query_one("#title-selector", Input).value = config.title_selector or ""
        self.query_one("#date-selector", Input).value = config.date_selector or ""
        self.query_one("#author-selector", Input).value = config.author_selector or ""
        
        # Advanced settings
        self.query_one("#remove-scripts", Switch).value = config.remove_scripts
        self.query_one("#remove-styles", Switch).value = config.remove_styles
        self.query_one("#preserve-links", Switch).value = config.preserve_links
        self.query_one("#extract-images", Switch).value = config.extract_images
        
        # Statistics
        self.update_statistics(config)
        
        self.unsaved_changes = False
    
    def update_statistics(self, config: SiteConfig):
        """Update statistics display."""
        container = self.query_one("#stats-container", Container)
        container.remove_children()
        
        stats_text = f"""Success: {config.success_count} | Errors: {config.error_count}
Last Updated: {config.updated_at}
Created: {config.created_at}"""
        
        if config.last_error:
            stats_text += f"\nLast Error: {config.last_error}"
        
        container.mount(Static(stats_text))
    
    def update_auth_fields(self, auth_type: Optional[str]):
        """Show/hide auth fields based on type."""
        # Hide all auth containers
        self.query_one("#basic-auth-container").add_class("hidden")
        self.query_one("#bearer-auth-container").add_class("hidden")
        self.query_one("#api-key-container").add_class("hidden")
        
        # Show relevant container
        if auth_type == "basic":
            self.query_one("#basic-auth-container").remove_class("hidden")
        elif auth_type == "bearer":
            self.query_one("#bearer-auth-container").remove_class("hidden")
        elif auth_type == "api_key":
            self.query_one("#api-key-container").remove_class("hidden")
    
    @on(Button.Pressed, "#add-site-btn")
    async def on_add_site(self):
        """Handle adding a new site."""
        input_field = self.query_one("#new-domain-input", Input)
        domain = input_field.value.strip()
        
        if not domain:
            return
        
        # Create new config
        config = SiteConfig(domain)
        if self.config_manager.save_config(config):
            input_field.value = ""
            await self.load_site_list()
            
            # Select the new site
            self.selected_domain = domain
            await self.load_site_config(domain)
    
    @on(Button.Pressed, "#save-config-btn")
    async def on_save_config(self):
        """Save current configuration."""
        if not self.current_config:
            return
        
        # Gather all values
        config = self.current_config
        
        # General settings
        config.rate_limit_requests = int(self.query_one("#rate-limit-requests", Input).value or 60)
        config.concurrent_requests = int(self.query_one("#concurrent-requests", Input).value or 1)
        config.timeout = int(self.query_one("#timeout", Input).value or 30)
        config.retry_count = int(self.query_one("#retry-count", Input).value or 3)
        
        # JavaScript settings
        config.requires_javascript = self.query_one("#requires-javascript", Switch).value
        config.wait_for_selector = self.query_one("#wait-for-selector", Input).value or None
        
        # Save configuration
        if self.config_manager.save_config(config):
            self.unsaved_changes = False
            await self.load_site_list()
            self.notify("Configuration saved successfully", severity="information")
    
    @on(Button.Pressed, "#delete-config-btn")
    async def on_delete_config(self):
        """Delete current configuration."""
        if not self.current_config:
            return
        
        # TODO: Show confirmation dialog
        
        if self.config_manager.delete_config(self.current_config.domain):
            self.selected_domain = None
            self.current_config = None
            await self.load_site_list()
            
            # Clear form
            title = self.query_one("#config-title", Static)
            title.update("Site Configuration")
    
    @on(Button.Pressed, "#apply-preset-btn")
    async def on_apply_preset(self):
        """Show preset selection dialog."""
        if not self.current_config:
            return
        
        # TODO: Show preset selection dialog
        presets = self.config_manager.get_presets()
        logger.info(f"Available presets: {list(presets.keys())}")
    
    @on(Select.Changed, "#auth-type-select")
    def on_auth_type_changed(self, event: Select.Changed):
        """Handle auth type change."""
        self.update_auth_fields(event.value)
        self.unsaved_changes = True
    
    @on(Input.Changed)
    @on(Switch.Changed)
    def on_field_changed(self):
        """Mark configuration as having unsaved changes."""
        if self.current_config:
            self.unsaved_changes = True


# End of SiteConfigSettings.py