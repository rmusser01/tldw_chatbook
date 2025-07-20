# Subscription_Window.py
# Description: UI components for the Subscriptions Window
#
# This window provides a comprehensive interface for managing content subscriptions,
# including RSS/Atom feeds, URL monitoring, and automated briefing generation.
#
# Imports
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Button, TextArea, Static, Label, Input, Select, ListView, ListItem,
    Checkbox, TabbedContent, TabPane, DataTable, LoadingIndicator, 
    Collapsible, RadioSet, RadioButton
)
from textual.screen import Screen
#
# Local Imports
# TODO: Uncomment when SubscriptionsDB is implemented
# from ..DB.Subscriptions_DB import SubscriptionsDB
# from ..Event_Handlers.subscription_events import (
#     NewSubscriptionItems, SubscriptionCheckComplete, SubscriptionError,
#     SubscriptionHealthUpdate, handle_add_subscription, handle_check_all_subscriptions
# )
from ..Metrics.metrics_logger import log_counter

# Stub classes/functions until proper implementation
class SubscriptionsDB:
    """Stub class for subscriptions database"""
    pass

async def handle_add_subscription(app_instance, event):
    """Stub handler"""
    app_instance.notify("Add subscription feature coming soon", severity="info")

async def handle_check_all_subscriptions(app_instance, event):
    """Stub handler"""
    app_instance.notify("Check subscriptions feature coming soon", severity="info")
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Main Window Class
#
########################################################################################################################

class SubscriptionWindow(Screen):
    """
    Main container for the Subscriptions Tab UI.
    
    Provides:
    - Subscription management (add, edit, delete)
    - New items review interface
    - Health monitoring dashboard
    - Briefing configuration
    - Import/export functionality
    """
    
    DEFAULT_CSS = """
    SubscriptionWindow {
        height: 100%;
    }
    
    /* Main layout containers */
    #subscription-main-layout {
        layout: horizontal;
        height: 100%;
    }
    
    #subscription-sidebar {
        width: 25%;
        min-width: 30;
        border-right: solid $primary;
        padding: 1;
    }
    
    #subscription-content {
        width: 75%;
        padding: 1;
    }
    
    /* Subscription list styling */
    #subscription-list {
        height: 1fr;
        border: round $primary;
        margin: 1 0;
    }
    
    .subscription-item {
        padding: 0 1;
    }
    
    .subscription-item.active {
        background: $boost;
    }
    
    .subscription-item.paused {
        color: $warning;
    }
    
    .subscription-item.error {
        color: $error;
    }
    
    /* Status indicators */
    .health-indicator {
        width: 20;
        border: round $primary;
        align: center middle;
        margin: 1;
    }
    
    .health-indicator.good {
        background: $success 20%;
        color: $success;
    }
    
    .health-indicator.warning {
        background: $warning 20%;
        color: $warning;
    }
    
    .health-indicator.error {
        background: $error 20%;
        color: $error;
    }
    
    /* Form styling */
    .form-group {
        margin: 1 0;
    }
    
    .form-label {
        margin: 0 0 1 0;
        color: $text-muted;
    }
    
    /* New items review */
    #items-review-list {
        height: 1fr;
        border: round $primary;
        margin: 1 0;
    }
    
    .item-preview {
        padding: 1;
        margin: 0 0 1 0;
        border: round $primary-background;
    }
    
    .item-preview:hover {
        background: $boost;
    }
    
    /* Action buttons */
    .action-buttons {
        layout: horizontal;
        height: 3;
        align: center middle;
        margin: 1 0;
    }
    
    .action-buttons Button {
        margin: 0 1;
    }
    
    /* Briefing configuration */
    .briefing-schedule {
        border: round $primary;
        padding: 1;
        margin: 1 0;
    }
    
    /* Loading states */
    .loading-container {
        align: center middle;
        height: 100%;
    }
    
    /* Stats display */
    .stats-grid {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 1;
        margin: 1 0;
    }
    
    .stat-card {
        border: round $primary;
        padding: 1;
        align: center middle;
    }
    
    .stat-value {
        text-style: bold;
        color: $primary;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the Subscription Window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.subscriptions_db: Optional[SubscriptionsDB] = None
        self.selected_subscription_id: Optional[int] = None
        self.selected_items: List[int] = []
        
    def compose(self) -> ComposeResult:
        """Build the subscription window UI."""
        with Container(id="subscription-main-layout"):
            # Left sidebar with subscription list
            with Container(id="subscription-sidebar"):
                yield Label("📡 Subscriptions", classes="section-header")
                
                # Quick stats
                with Horizontal(classes="stats-grid"):
                    yield Static("0", id="stat-active-count", classes="health-indicator good")
                    yield Static("0", id="stat-paused-count", classes="health-indicator warning")
                    yield Static("0", id="stat-error-count", classes="health-indicator error")
                
                # Control buttons
                with Horizontal(classes="action-buttons"):
                    yield Button("➕ Add", id="subscription-add-new-button", variant="primary")
                    yield Button("🔄 Check All", id="subscription-check-all-button", variant="default")
                    yield Button("⚙️", id="subscription-settings-button", variant="default")
                
                # Subscription list
                yield ListView(id="subscription-list")
                
                # Import/Export buttons
                with Horizontal(classes="action-buttons"):
                    yield Button("📥 Import", id="subscription-import-button", variant="default")
                    yield Button("📤 Export", id="subscription-export-button", variant="default")
            
            # Main content area with tabs
            with Container(id="subscription-content"):
                with TabbedContent():
                    # Add/Edit subscription tab
                    with TabPane("Add/Edit", id="tab-add-edit"):
                        yield from self._compose_add_edit_form()
                    
                    # Review new items tab
                    with TabPane("New Items", id="tab-review-items"):
                        yield from self._compose_review_items()
                    
                    # Health monitoring tab
                    with TabPane("Health", id="tab-health"):
                        yield from self._compose_health_dashboard()
                    
                    # Briefing configuration tab
                    with TabPane("Briefings", id="tab-briefings"):
                        yield from self._compose_briefing_config()
                    
                    # Settings tab
                    with TabPane("Settings", id="tab-settings"):
                        yield from self._compose_settings()
    
    def _compose_add_edit_form(self) -> ComposeResult:
        """Build the add/edit subscription form."""
        with ScrollableContainer():
            yield Label("Add/Edit Subscription", classes="section-header")
            
            # Basic Information
            with Collapsible(title="Basic Information", collapsed=False):
                with Container(classes="form-group"):
                    yield Label("Name *", classes="form-label")
                    yield Input(placeholder="My Tech Blog", id="subscription-name-input")
                
                with Container(classes="form-group"):
                    yield Label("Type *", classes="form-label")
                    yield Select(
                        [(x, x) for x in ["rss", "atom", "url", "podcast", "api"]],
                        id="subscription-type-select",
                        value="rss"
                    )
                
                with Container(classes="form-group"):
                    yield Label("URL/Feed *", classes="form-label")
                    yield Input(placeholder="https://example.com/feed.xml", id="subscription-url-input")
                
                with Container(classes="form-group"):
                    yield Label("Description", classes="form-label")
                    yield Input(placeholder="Optional description", id="subscription-description-input")
                
                with Container(classes="form-group"):
                    yield Label("Tags (comma-separated)", classes="form-label")
                    yield Input(placeholder="tech, ai, news", id="subscription-tags-input")
                
                with Container(classes="form-group"):
                    yield Label("Folder", classes="form-label")
                    yield Input(placeholder="Tech News", id="subscription-folder-input")
            
            # Monitoring Configuration
            with Collapsible(title="Monitoring Configuration", collapsed=True):
                with Container(classes="form-group"):
                    yield Label("Priority", classes="form-label")
                    yield Select(
                        [("1 - Lowest", "1"), ("2", "2"), ("3 - Normal", "3"), ("4", "4"), ("5 - Highest", "5")],
                        id="subscription-priority-select",
                        value="3"
                    )
                
                with Container(classes="form-group"):
                    yield Label("Check Frequency", classes="form-label")
                    yield Select(
                        [
                            ("5 minutes", "300"),
                            ("15 minutes", "900"),
                            ("30 minutes", "1800"),
                            ("1 hour", "3600"),
                            ("2 hours", "7200"),
                            ("6 hours", "21600"),
                            ("12 hours", "43200"),
                            ("24 hours", "86400")
                        ],
                        id="subscription-frequency-select",
                        value="3600"
                    )
                
                yield Checkbox("Auto-ingest new items", id="subscription-auto-ingest", value=False)
                yield Checkbox("Extract full content", id="subscription-extract-full", value=False)
                
                with Container(classes="form-group"):
                    yield Label("Change Threshold (% for URLs)", classes="form-label")
                    yield Input(value="10", id="subscription-change-threshold", type="number")
                
                with Container(classes="form-group"):
                    yield Label("Auto-pause after failures", classes="form-label")
                    yield Input(value="10", id="subscription-auto-pause-threshold", type="number")
            
            # Authentication
            with Collapsible(title="Authentication", collapsed=True):
                with Container(classes="form-group"):
                    yield Label("Auth Type", classes="form-label")
                    yield Select(
                        [("None", "none"), ("Basic", "basic"), ("Bearer Token", "bearer"), ("API Key", "api_key")],
                        id="subscription-auth-type",
                        value="none"
                    )
                
                with Container(classes="form-group"):
                    yield Label("Username/Key", classes="form-label")
                    yield Input(id="subscription-auth-username", password=False)
                
                with Container(classes="form-group"):
                    yield Label("Password/Token", classes="form-label")
                    yield Input(id="subscription-auth-password", password=True)
            
            # Advanced Options
            with Collapsible(title="Advanced Options", collapsed=True):
                with Container(classes="form-group"):
                    yield Label("Custom Headers (JSON)", classes="form-label")
                    yield TextArea(id="subscription-custom-headers", classes="code-input")
                
                with Container(classes="form-group"):
                    yield Label("Ignore Selectors (CSS, one per line)", classes="form-label")
                    yield TextArea(id="subscription-ignore-selectors")
                
                with Container(classes="form-group"):
                    yield Label("Rate Limit (requests/minute)", classes="form-label")
                    yield Input(value="60", id="subscription-rate-limit", type="number")
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("Save", id="subscription-add-button", variant="primary")
                yield Button("Cancel", id="subscription-cancel-button", variant="default")
                yield Button("Delete", id="subscription-delete-button", variant="error")
    
    def _compose_review_items(self) -> ComposeResult:
        """Build the new items review interface."""
        with Container():
            yield Label("Review New Items", classes="section-header")
            
            # Filter controls
            with Horizontal(classes="action-buttons"):
                yield Select(
                    [("All Subscriptions", "all"), ("Selected Only", "selected")],
                    id="items-filter-subscription",
                    value="all"
                )
                yield Select(
                    [("New", "new"), ("Reviewed", "reviewed"), ("Ingested", "ingested"), ("Ignored", "ignored")],
                    id="items-filter-status",
                    value="new"
                )
                yield Button("🔄 Refresh", id="items-refresh-button", variant="default")
            
            # Items list
            yield ListView(id="items-review-list")
            
            # Bulk actions
            with Horizontal(classes="action-buttons"):
                yield Button("✅ Accept Selected", id="subscription-accept-button", variant="success")
                yield Button("👁️ Mark Reviewed", id="subscription-mark-reviewed-button", variant="default")
                yield Button("❌ Ignore Selected", id="subscription-ignore-button", variant="warning")
                yield Button("🤖 Analyze", id="subscription-analyze-button", variant="primary")
    
    def _compose_health_dashboard(self) -> ComposeResult:
        """Build the health monitoring dashboard."""
        with ScrollableContainer():
            yield Label("Subscription Health", classes="section-header")
            
            # Overall stats grid
            with Container(classes="stats-grid"):
                with Container(classes="stat-card"):
                    yield Label("Active", classes="form-label")
                    yield Label("0", id="health-active-count", classes="stat-value")
                
                with Container(classes="stat-card"):
                    yield Label("Paused", classes="form-label")
                    yield Label("0", id="health-paused-count", classes="stat-value")
                
                with Container(classes="stat-card"):
                    yield Label("Errors", classes="form-label")
                    yield Label("0", id="health-error-count", classes="stat-value")
                
                with Container(classes="stat-card"):
                    yield Label("Items/Day", classes="form-label")
                    yield Label("0", id="health-items-per-day", classes="stat-value")
                
                with Container(classes="stat-card"):
                    yield Label("Success Rate", classes="form-label")
                    yield Label("0%", id="health-success-rate", classes="stat-value")
                
                with Container(classes="stat-card"):
                    yield Label("Avg Response", classes="form-label")
                    yield Label("0ms", id="health-avg-response", classes="stat-value")
            
            # Failing subscriptions alert
            with Container(id="failing-subscriptions-alert", classes="hidden"):
                yield Label("⚠️ Failing Subscriptions", classes="section-header error")
                yield ListView(id="failing-subscriptions-list")
            
            # Activity log
            yield Label("Recent Activity", classes="section-header")
            yield TextArea(id="subscription-activity-log", read_only=True)
    
    def _compose_briefing_config(self) -> ComposeResult:
        """Build the briefing configuration interface."""
        with ScrollableContainer():
            yield Label("Briefing Configuration", classes="section-header")
            
            # Briefing list
            yield ListView(id="briefing-list")
            
            # Add new briefing
            with Collapsible(title="Create New Briefing", collapsed=False):
                with Container(classes="form-group"):
                    yield Label("Briefing Name", classes="form-label")
                    yield Input(placeholder="Morning Tech Digest", id="briefing-name-input")
                
                with Container(classes="form-group"):
                    yield Label("Schedule", classes="form-label")
                    yield RadioSet(
                        "daily",
                        "weekly", 
                        "custom",
                        id="briefing-schedule-type"
                    )
                
                with Container(classes="form-group"):
                    yield Label("Time", classes="form-label")
                    yield Input(value="06:00", id="briefing-time-input")
                
                with Container(classes="form-group"):
                    yield Label("Source Selection", classes="form-label")
                    yield Select(
                        [("All Active", "all"), ("By Tags", "tags"), ("By Folders", "folders"), ("Manual Selection", "manual")],
                        id="briefing-source-type",
                        value="all"
                    )
                
                with Container(classes="form-group"):
                    yield Label("Tags/Folders (comma-separated)", classes="form-label")
                    yield Input(id="briefing-source-filter")
                
                with Container(classes="form-group"):
                    yield Label("Analysis Prompt", classes="form-label")
                    yield TextArea(
                        "Summarize the key developments and their implications. Focus on actionable insights.",
                        id="briefing-analysis-prompt"
                    )
                
                with Container(classes="form-group"):
                    yield Label("Output Format", classes="form-label")
                    yield Select(
                        [("Markdown", "markdown"), ("HTML", "html"), ("PDF", "pdf")],
                        id="briefing-output-format",
                        value="markdown"
                    )
                
                yield Checkbox("Save to Notes", id="briefing-save-to-notes", value=True)
                yield Checkbox("Email notification", id="briefing-email-notify", value=False)
                
                with Horizontal(classes="action-buttons"):
                    yield Button("Create Briefing", id="briefing-create-button", variant="primary")
    
    def _compose_settings(self) -> ComposeResult:
        """Build the settings interface."""
        with ScrollableContainer():
            yield Label("Subscription Settings", classes="section-header")
            
            with Collapsible(title="General Settings", collapsed=False):
                yield Checkbox("Enable background checking", id="settings-enable-background", value=True)
                yield Checkbox("Show notifications for new items", id="settings-show-notifications", value=True)
                yield Checkbox("Auto-analyze items before ingestion", id="settings-auto-analyze", value=False)
                
                with Container(classes="form-group"):
                    yield Label("Default check interval (seconds)", classes="form-label")
                    yield Input(value="3600", id="settings-default-interval", type="number")
                
                with Container(classes="form-group"):
                    yield Label("Max concurrent checks", classes="form-label")
                    yield Input(value="10", id="settings-max-concurrent", type="number")
            
            with Collapsible(title="Security Settings", collapsed=True):
                yield Checkbox("Verify SSL certificates", id="settings-verify-ssl", value=True)
                yield Checkbox("Enable SSRF protection", id="settings-ssrf-protection", value=True)
                yield Checkbox("Enable XXE protection", id="settings-xxe-protection", value=True)
                
                with Container(classes="form-group"):
                    yield Label("Request timeout (seconds)", classes="form-label")
                    yield Input(value="30", id="settings-request-timeout", type="number")
            
            with Collapsible(title="Performance Settings", collapsed=True):
                yield Checkbox("Use connection pooling", id="settings-connection-pooling", value=True)
                yield Checkbox("Enable response caching", id="settings-enable-caching", value=True)
                yield Checkbox("Use HTTP/2", id="settings-use-http2", value=True)
                
                with Container(classes="form-group"):
                    yield Label("Cache TTL (seconds)", classes="form-label")
                    yield Input(value="300", id="settings-cache-ttl", type="number")
            
            with Horizontal(classes="action-buttons"):
                yield Button("Save Settings", id="settings-save-button", variant="primary")
                yield Button("Reset to Defaults", id="settings-reset-button", variant="warning")
    
    # Event handlers
    async def on_mount(self) -> None:
        """Initialize the subscription window when mounted."""
        try:
            # TODO: Initialize database connection when SubscriptionsDB is implemented
            # from ..config import get_subscriptions_db_path
            # db_path = get_subscriptions_db_path()
            # self.subscriptions_db = SubscriptionsDB(db_path, self.app_instance.client_id)
            
            # TODO: Load initial data when database is available
            # await self._refresh_subscription_list()
            # await self._update_health_stats()
            
            # Log mounting
            log_counter("subscription_window_mounted", 1)
            
            # Notify user that subscriptions feature is under development
            self.app_instance.notify("Subscriptions feature is under development", severity="warning")
            
        except Exception as e:
            self.app_instance.notify(f"Error initializing subscriptions: {e}", severity="error")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "subscription-add-new-button":
            # Switch to add/edit tab
            tabs = self.query_one(TabbedContent)
            tabs.active = "tab-add-edit"
            
        elif event.button.id == "subscription-check-all-button":
            await handle_check_all_subscriptions(self.app_instance, event)
            
        elif event.button.id == "subscription-add-button":
            await handle_add_subscription(self.app_instance, event)
            
        # Add more button handlers as needed
    
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle subscription selection from list."""
        # Get subscription ID from selected item
        # Load subscription details into form
        pass
    
    # Helper methods
    async def _refresh_subscription_list(self) -> None:
        """Refresh the subscription list in the sidebar."""
        if not self.subscriptions_db:
            return
            
        try:
            # TODO: Implement when database is available
            subscriptions = []  # self.subscriptions_db.get_all_subscriptions()
            list_view = self.query_one("#subscription-list", ListView)
            
            await list_view.clear()
            
            for sub in subscriptions:
                # Determine status icon
                if sub['is_paused']:
                    status = "⏸️"
                    css_class = "paused"
                elif sub['consecutive_failures'] > 3:
                    status = "❌"
                    css_class = "error"
                elif sub['is_active']:
                    status = "✅"
                    css_class = "active"
                else:
                    status = "⭕"
                    css_class = ""
                
                # Create list item
                item_text = f"{status} {sub['name']} [{sub['type']}]"
                if sub['last_checked']:
                    # Format last checked time
                    last_checked = datetime.fromisoformat(sub['last_checked'])
                    item_text += f"\n    Last: {last_checked.strftime('%Y-%m-%d %H:%M')}"
                
                item = ListItem(
                    Static(item_text, classes=f"subscription-item {css_class}"),
                    name=str(sub['id'])
                )
                await list_view.append(item)
                
        except Exception as e:
            self.app_instance.notify(f"Error refreshing subscriptions: {e}", severity="error")
    
    async def _update_health_stats(self) -> None:
        """Update the health statistics displays."""
        if not self.subscriptions_db:
            return
            
        try:
            # TODO: Implement when database is available
            # Get subscription counts
            counts = {"active": 0, "paused": 0, "total": 0}  # self.subscriptions_db.get_subscription_count(active_only=False)
            total = sum(counts.values())
            
            # Update sidebar stats
            self.query_one("#stat-active-count").update(str(total))
            
            # Get failing subscriptions
            failing = []  # self.subscriptions_db.get_failing_subscriptions()
            self.query_one("#stat-error-count").update(str(len(failing)))
            
            # Update health dashboard
            self.query_one("#health-active-count").update(str(total))
            self.query_one("#health-error-count").update(str(len(failing)))
            
            # Show/hide failing alert
            if failing:
                alert = self.query_one("#failing-subscriptions-alert")
                alert.remove_class("hidden")
                
                failing_list = self.query_one("#failing-subscriptions-list", ListView)
                await failing_list.clear()
                
                for sub in failing[:5]:  # Show top 5
                    info = f"⚠️ {sub['name']} - {sub['consecutive_failures']} failures\n   Last error: {sub['last_error']}"
                    await failing_list.append(ListItem(Static(info)))
                    
        except Exception as e:
            self.app_instance.notify(f"Error updating health stats: {e}", severity="error")


# End of Subscription_Window.py