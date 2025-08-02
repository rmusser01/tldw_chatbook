# SubscriptionWindow.py
# Description: Main subscription management UI for tldw_chatbook
#
# This window provides:
# - Subscription CRUD operations
# - Real-time monitoring dashboard
# - New items review interface
# - Briefing generation controls
# - Import/export functionality
#
# Imports
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING
import json
#
# Third-Party Imports
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container, VerticalScroll
from textual.widgets import (
    Button, Input, Select, Label, Static, ListView, ListItem, 
    TextArea, Checkbox, DataTable, TabbedContent, TabPane,
    Placeholder, ProgressBar, Sparkline
)
from textual.css.query import QueryError
from textual.reactive import reactive
from loguru import logger
#
# Local Imports
from ..Widgets.form_components import (
    FormField, FormFieldSet, FormSubmitButton, 
    ValidationStatus, form_validator
)
from ..Event_Handlers.subscription_events import (
    SubscriptionCheckStarted, SubscriptionCheckComplete,
    NewSubscriptionItems, SubscriptionError, BriefingGenerated,
    SubscriptionHealthUpdate, BulkOperationComplete
)
from ..DB.Subscriptions_DB import SubscriptionsDB, SubscriptionError as DBSubscriptionError
from ..Subscriptions.textual_scheduler_worker import SubscriptionSchedulerWorker

# Try importing briefing-related modules (require optional dependencies)
try:
    from ..Subscriptions.briefing_generator import BriefingGenerator
    from ..Subscriptions.briefing_templates import BriefingTemplateManager
    BRIEFING_AVAILABLE = True
except ImportError:
    BriefingGenerator = None
    BriefingTemplateManager = None
    BRIEFING_AVAILABLE = False
    logger.warning("Briefing functionality unavailable - install with: pip install tldw_chatbook[subscriptions]")
from ..config import get_subscriptions_db_path
from ..Constants import SUBSCRIPTION_TYPES, SUBSCRIPTION_UPDATE_FREQUENCIES
from .SiteConfigSettings import SiteConfigSettings
from .ScraperBuilderWindow import ScraperBuilderWindow
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Subscription Window
#
########################################################################################################################

class SubscriptionWindow(Container):
    """Main subscription management window."""
    
    CSS = """
    SubscriptionWindow {
        background: $background;
    }
    
    /* Tab styling */
    TabbedContent {
        height: 100%;
        background: $panel;
    }
    
    TabPane {
        padding: 1;
    }
    
    /* Form styling */
    .subscription-form {
        width: 100%;
        padding: 1;
        background: $surface;
        border: tall $primary;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .form-label {
        width: 20;
        content-align: right middle;
        padding-right: 1;
    }
    
    /* List styling */
    .subscription-list {
        height: 20;
        border: solid $primary;
        background: $surface;
    }
    
    .items-review-list {
        height: 30;
        border: solid $primary;
        background: $surface;
    }
    
    /* Dashboard styling */
    .stat-card {
        width: 25;
        height: 7;
        border: tall $primary;
        padding: 1;
        margin: 1;
        background: $surface;
    }
    
    .stat-value {
        text-style: bold;
        text-align: center;
        color: $success;
        margin-top: 1;
    }
    
    .health-alert {
        width: 100%;
        padding: 1;
        background: $error-darken-2;
        border: tall $error;
        margin: 1;
    }
    
    /* Button styling */
    .action-button {
        width: 20;
        margin: 0 1;
    }
    
    .primary-button {
        background: $primary;
        color: $text;
    }
    
    .danger-button {
        background: $error;
        color: $text;
    }
    
    /* Progress indicators */
    .progress-container {
        width: 100%;
        height: 3;
        padding: 1;
    }
    
    /* Item preview */
    .item-preview {
        width: 100%;
        height: 15;
        border: tall $primary;
        padding: 1;
        background: $surface;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', *args, **kwargs):
        """
        Initialize subscription window.
        
        Args:
            app_instance: Reference to the main app
        """
        super().__init__(*args, **kwargs)
        self.app_instance = app_instance  # Changed from self.app to self.app_instance
        self.client_id = "cli"
        self.db: Optional[SubscriptionsDB] = None
        self.scheduler_worker: Optional[SubscriptionSchedulerWorker] = None
        self.briefing_generator: Optional[BriefingGenerator] = None
        self.template_manager: Optional[BriefingTemplateManager] = None
        
        # State
        self.selected_subscription: Optional[int] = None
        self.selected_items: List[int] = []
        self.is_checking = reactive(False)
        self.check_progress = reactive(0.0)
    
    def compose(self) -> ComposeResult:
        """Compose the subscription UI."""
        with TabbedContent(initial="subscriptions"):
            # Subscriptions tab
            with TabPane("Subscriptions", id="subscriptions"):
                yield from self._compose_subscriptions_tab()
            
            # Review tab
            with TabPane("Review Items", id="review"):
                yield from self._compose_review_tab()
            
            # Dashboard tab
            with TabPane("Dashboard", id="dashboard"):
                yield from self._compose_dashboard_tab()
            
            # Briefings tab
            with TabPane("Briefings", id="briefings"):
                yield from self._compose_briefings_tab()
            
            # Settings tab
            with TabPane("Settings", id="settings"):
                yield from self._compose_settings_tab()
    
    def _compose_subscriptions_tab(self) -> ComposeResult:
        """Compose subscriptions management tab."""
        with Horizontal():
            # Left side - subscription list
            with Vertical(classes="subscription-list-container"):
                yield Label("Active Subscriptions")
                yield ListView(id="subscription-list", classes="subscription-list")
                
                with Horizontal(classes="list-actions"):
                    yield Button("Add New", id="add-subscription-btn", classes="action-button primary-button")
                    yield Button("Edit", id="edit-subscription-btn", classes="action-button")
                    yield Button("Delete", id="delete-subscription-btn", classes="action-button danger-button")
                    yield Button("Scraper Builder", id="scraper-builder-btn", classes="action-button")
            
            # Right side - subscription form
            with ScrollableContainer(classes="subscription-form-container"):
                yield from self._compose_subscription_form()
    
    def _compose_subscription_form(self) -> ComposeResult:
        """Compose subscription add/edit form."""
        with Vertical(classes="subscription-form", id="subscription-form"):
            yield Label("Subscription Details", classes="form-header")
            
            # Basic fields
            with FormFieldSet("Basic Information"):
                yield FormField(
                    "Name",
                    Input(placeholder="My News Feed", id="sub-name"),
                    required=True
                )
                
                yield FormField(
                    "Type",
                    Select(
                        [(t, t) for t in SUBSCRIPTION_TYPES],
                        id="sub-type",
                        value="rss"
                    ),
                    required=True
                )
                
                yield FormField(
                    "URL/Source",
                    Input(placeholder="https://example.com/feed.xml", id="sub-url"),
                    required=True
                )
                
                yield FormField(
                    "Description",
                    TextArea(id="sub-description")
                )
            
            # Organization
            with FormFieldSet("Organization"):
                yield FormField(
                    "Tags",
                    Input(placeholder="news, tech, updates (comma separated)", id="sub-tags")
                )
                
                yield FormField(
                    "Folder",
                    Input(placeholder="News Feeds", id="sub-folder")
                )
                
                yield FormField(
                    "Priority",
                    Select(
                        [("1", "High"), ("3", "Medium"), ("5", "Low")],
                        id="sub-priority",
                        value="3"
                    )
                )
            
            # Schedule
            with FormFieldSet("Update Schedule"):
                yield FormField(
                    "Check Frequency",
                    Select(
                        [(str(v), k) for k, v in SUBSCRIPTION_UPDATE_FREQUENCIES.items()],
                        id="sub-frequency",
                        value="3600"
                    )
                )
                
                yield FormField(
                    "Auto-ingest",
                    Checkbox(id="sub-auto-ingest", value=False)
                )
            
            # Advanced options
            with FormFieldSet("Advanced Options", collapsed=True):
                yield FormField(
                    "Authentication",
                    Select(
                        [("none", "None"), ("basic", "Basic Auth"), ("bearer", "Bearer Token")],
                        id="sub-auth-type",
                        value="none"
                    )
                )
                
                yield FormField(
                    "Custom Headers",
                    TextArea(
                        id="sub-headers"
                    )
                )
                
                yield FormField(
                    "Change Threshold (%)",
                    Input(value="15", id="sub-change-threshold", type="number")
                )
                
                yield FormField(
                    "Rate Limit (req/min)",
                    Input(value="60", id="sub-rate-limit", type="number")
                )
            
            # Form actions
            with Horizontal(classes="form-actions"):
                yield Button("Save", id="save-subscription-btn", classes="primary-button")
                yield Button("Cancel", id="cancel-subscription-btn")
                yield Button("Test", id="test-subscription-btn")
    
    def _compose_review_tab(self) -> ComposeResult:
        """Compose items review tab."""
        with Vertical():
            # Action bar
            with Horizontal(classes="review-actions"):
                yield Button("Check All", id="check-all-btn", classes="primary-button")
                yield Button("Accept Selected", id="accept-items-btn", classes="action-button")
                yield Button("Ignore Selected", id="ignore-items-btn", classes="action-button")
                yield Button("Mark Reviewed", id="mark-reviewed-btn", classes="action-button")
                yield Static("", id="items-count")
            
            # Progress bar
            yield ProgressBar(id="check-progress", classes="progress-container")
            
            # Items list
            with Horizontal():
                # Items list
                with Vertical(classes="items-list-container"):
                    yield Label("New Items")
                    yield ListView(id="items-review-list", classes="items-review-list")
                
                # Item preview
                with Vertical(classes="item-preview-container"):
                    yield Label("Preview")
                    yield TextArea(id="item-preview", classes="item-preview", read_only=True)
    
    def _compose_dashboard_tab(self) -> ComposeResult:
        """Compose monitoring dashboard tab."""
        with ScrollableContainer():
            # Summary stats
            with Horizontal(classes="stats-row"):
                with Container(classes="stat-card"):
                    yield Label("Active Subscriptions")
                    yield Static("0", id="stat-active", classes="stat-value")
                
                with Container(classes="stat-card"):
                    yield Label("Items Today")
                    yield Static("0", id="stat-items-today", classes="stat-value")
                
                with Container(classes="stat-card"):
                    yield Label("Last Check")
                    yield Static("Never", id="stat-last-check", classes="stat-value")
                
                with Container(classes="stat-card"):
                    yield Label("Error Rate")
                    yield Static("0%", id="stat-error-rate", classes="stat-value")
            
            # Health alerts
            yield Container(id="health-alerts-container")
            
            # Activity chart
            yield Label("Activity (Last 7 Days)")
            yield Sparkline(
                data=[0] * 7,
                id="activity-chart"
            )
            
            # Recent activity log
            yield Label("Recent Activity")
            yield TextArea(
                id="activity-log",
                read_only=True,
                max_lines=20
            )
    
    def _compose_briefings_tab(self) -> ComposeResult:
        """Compose briefings tab."""
        with Horizontal():
            # Left side - briefing controls
            with Vertical(classes="briefing-controls"):
                yield Label("Generate Briefing")
                
                with FormFieldSet("Briefing Options"):
                    yield FormField(
                        "Name",
                        Input(
                            placeholder="Daily Digest",
                            id="briefing-name",
                            value=f"Briefing {datetime.now().strftime('%Y-%m-%d')}"
                        )
                    )
                    
                    yield FormField(
                        "Template",
                        Select(
                            id="briefing-template",
                            options=[
                                ("executive_summary", "Executive Summary"),
                                ("technical_digest", "Technical Digest"),
                                ("news_briefing", "News Briefing"),
                                ("custom", "Custom Template")
                            ],
                            value="news_briefing"
                        )
                    )
                    
                    yield FormField(
                        "Time Range",
                        Select(
                            id="briefing-time-range",
                            options=[
                                ("today", "Today"),
                                ("yesterday", "Yesterday"),
                                ("week", "Last 7 Days"),
                                ("custom", "Custom Range")
                            ],
                            value="today"
                        )
                    )
                    
                    yield FormField(
                        "Sources",
                        Select(
                            id="briefing-sources",
                            options=[
                                ("all", "All Sources"),
                                ("high_priority", "High Priority Only"),
                                ("selected", "Selected Sources"),
                                ("tags", "By Tags")
                            ],
                            value="all"
                        )
                    )
                    
                    yield FormField(
                        "Format",
                        Select(
                            id="briefing-format",
                            options=[
                                ("markdown", "Markdown"),
                                ("html", "HTML"),
                                ("text", "Plain Text"),
                                ("json", "JSON")
                            ],
                            value="markdown"
                        )
                    )
                    
                    yield FormField(
                        "Enhance with LLM",
                        Checkbox(id="briefing-enhance", value=True)
                    )
                
                with Horizontal(classes="briefing-actions"):
                    yield Button("Generate", id="generate-briefing-btn", classes="primary-button")
                    yield Button("Save Template", id="save-template-btn")
            
            # Right side - briefing preview
            with Vertical(classes="briefing-preview-container"):
                yield Label("Preview")
                yield TextArea(
                    id="briefing-preview",
                    read_only=True,
                    language="markdown"
                )
                
                with Horizontal(classes="preview-actions"):
                    yield Button("Save to Notes", id="save-briefing-notes-btn")
                    yield Button("Export", id="export-briefing-btn")
                    yield Button("Copy", id="copy-briefing-btn")
    
    def _compose_settings_tab(self) -> ComposeResult:
        """Compose settings tab."""
        with ScrollableContainer():
            # Import/Export
            with FormFieldSet("Import/Export"):
                yield Label("Import subscriptions from OPML or JSON")
                with Horizontal():
                    yield Input(placeholder="Select file...", id="import-file-path")
                    yield Button("Browse", id="browse-import-btn")
                    yield Button("Import", id="import-subscriptions-btn")
                
                yield Label("Export subscriptions")
                with Horizontal():
                    yield Select(
                        [("opml", "OPML"), ("json", "JSON")],
                        id="export-format",
                        value="opml"
                    )
                    yield Button("Export", id="export-subscriptions-btn")
            
            # Scheduler settings
            with FormFieldSet("Scheduler Settings"):
                yield FormField(
                    "Enable Background Checks",
                    Checkbox(id="enable-scheduler", value=True)
                )
                
                yield FormField(
                    "Check Interval (seconds)",
                    Input(value="60", id="scheduler-interval", type="number")
                )
                
                yield FormField(
                    "Max Concurrent Checks",
                    Input(value="10", id="max-concurrent", type="number")
                )
            
            # Database maintenance
            with FormFieldSet("Database Maintenance"):
                yield Label("Clean up old items and optimize database")
                
                yield FormField(
                    "Delete items older than (days)",
                    Input(value="30", id="cleanup-days", type="number")
                )
                
                with Horizontal():
                    yield Button("Clean Database", id="clean-db-btn")
                    yield Button("Optimize Database", id="optimize-db-btn")
            
            # Site Configuration
            with FormFieldSet("Site Configuration"):
                yield Label("Configure per-site settings like rate limits, authentication, and content extraction")
                yield SiteConfigSettings()
    
    async def on_mount(self) -> None:
        """Initialize when window is mounted."""
        try:
            # Initialize database
            db_path = get_subscriptions_db_path()
            self.db = SubscriptionsDB(db_path, self.client_id)
            
            # Initialize components
            self.briefing_generator = BriefingGenerator(self.db)
            self.template_manager = BriefingTemplateManager()
            
            # Initialize scheduler worker
            self.scheduler_worker = SubscriptionSchedulerWorker(
                self.app_instance,
                self.db,
                max_concurrent=10,
                check_interval=60
            )
            
            # Load initial data
            await self.refresh_subscription_list()
            await self.refresh_dashboard()
            await self.load_new_items()
            await self.load_briefing_templates()
            
            # Start scheduler if enabled
            if self.query_one("#enable-scheduler", Checkbox).value:
                self.run_worker(self.scheduler_worker.start_scheduler())
            
            logger.info("Subscription window initialized")
            
        except Exception as e:
            logger.error(f"Error initializing subscription window: {e}")
            self.notify(f"Initialization error: {str(e)}", severity="error")
    
    async def refresh_subscription_list(self) -> None:
        """Refresh the subscription list."""
        try:
            if not self.db:
                return
            
            subscriptions = self.db.get_all_subscriptions(include_inactive=False)
            list_view = self.query_one("#subscription-list", ListView)
            
            await list_view.clear()
            
            for sub in subscriptions:
                # Format subscription info
                status_icon = "✓" if sub['is_active'] and not sub['is_paused'] else "⏸"
                if sub['consecutive_failures'] > 3:
                    status_icon = "⚠️"
                
                label = f"{status_icon} {sub['name']} [{sub['type']}]"
                if sub.get('folder'):
                    label += f" 📁{sub['folder']}"
                
                item = ListItem(Static(label, classes="subscription-item"))
                item.data = sub  # Store subscription data
                await list_view.append(item)
            
        except Exception as e:
            logger.error(f"Error refreshing subscription list: {e}")
    
    async def refresh_dashboard(self) -> None:
        """Refresh dashboard statistics."""
        try:
            if not self.db:
                return
            
            # Get statistics
            stats = self.db.get_subscription_stats()
            
            # Update stat cards
            self.query_one("#stat-active", Static).update(str(stats.get('active_count', 0)))
            self.query_one("#stat-items-today", Static).update(str(stats.get('items_today', 0)))
            
            # Last check time
            last_check = stats.get('last_check_time')
            if last_check:
                last_check_str = datetime.fromisoformat(last_check).strftime("%H:%M")
            else:
                last_check_str = "Never"
            self.query_one("#stat-last-check", Static).update(last_check_str)
            
            # Error rate
            error_rate = stats.get('error_rate', 0.0)
            self.query_one("#stat-error-rate", Static).update(f"{error_rate:.1f}%")
            
            # Update activity chart
            activity_data = stats.get('activity_last_7_days', [0] * 7)
            chart = self.query_one("#activity-chart", Sparkline)
            chart.data = activity_data
            
            # Check for health alerts
            await self.check_health_alerts()
            
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {e}")
    
    async def check_health_alerts(self) -> None:
        """Check for subscription health issues."""
        try:
            if not self.db:
                return
            
            failing = self.db.get_failing_subscriptions(threshold=3)
            alerts_container = self.query_one("#health-alerts-container")
            
            # Clear existing alerts
            await alerts_container.remove_children()
            
            if failing:
                alert = Container(classes="health-alert")
                alert_text = f"⚠️ {len(failing)} subscription(s) are failing:\n"
                for sub in failing[:5]:
                    alert_text += f"  • {sub['name']} - {sub['consecutive_failures']} failures\n"
                if len(failing) > 5:
                    alert_text += f"  ... and {len(failing) - 5} more"
                
                await alert.mount(Static(alert_text))
                await alerts_container.mount(alert)
                
        except Exception as e:
            logger.error(f"Error checking health alerts: {e}")
    
    async def load_new_items(self) -> None:
        """Load new items for review."""
        try:
            if not self.db:
                return
            
            items = self.db.get_new_items(limit=100)
            list_view = self.query_one("#items-review-list", ListView)
            
            await list_view.clear()
            self.selected_items.clear()
            
            for item in items:
                # Format item display
                title = item.get('title', 'Untitled')
                source = item.get('subscription_name', 'Unknown')
                date = item.get('created_at', '')
                
                if date:
                    try:
                        date_obj = datetime.fromisoformat(date)
                        date_str = date_obj.strftime("%m/%d %H:%M")
                    except:
                        date_str = date
                else:
                    date_str = ""
                
                label = f"□ {title}\n   {source} • {date_str}"
                
                list_item = ListItem(Static(label, classes="review-item"))
                list_item.data = item
                await list_view.append(list_item)
            
            # Update count
            self.query_one("#items-count", Static).update(f"{len(items)} new items")
            
        except Exception as e:
            logger.error(f"Error loading new items: {e}")
    
    async def load_briefing_templates(self) -> None:
        """Load available briefing templates."""
        try:
            templates = self.template_manager.list_templates()
            template_select = self.query_one("#briefing-template", Select)
            
            options = []
            for template in templates:
                options.append((template.id, template.name))
            
            # template_select.options = options  # Update options
            
        except Exception as e:
            logger.error(f"Error loading briefing templates: {e}")
    
    @on(Button.Pressed, "#add-subscription-btn")
    async def handle_add_subscription(self, event: Button.Pressed) -> None:
        """Handle add subscription button."""
        # Clear form
        self.query_one("#sub-name", Input).value = ""
        self.query_one("#sub-url", Input).value = ""
        self.query_one("#sub-description", TextArea).value = ""
        self.selected_subscription = None
        
        # Focus name field
        self.query_one("#sub-name", Input).focus()
    
    @on(Button.Pressed, "#scraper-builder-btn")
    async def handle_scraper_builder(self, event: Button.Pressed) -> None:
        """Launch the visual scraper builder."""
        # Get current URL from form if available
        url = self.query_one("#sub-url", Input).value.strip()
        
        # Launch scraper builder
        await self.app_instance.push_screen(ScraperBuilderWindow(url=url))
    
    @on(Button.Pressed, "#save-subscription-btn")
    async def handle_save_subscription(self, event: Button.Pressed) -> None:
        """Handle save subscription."""
        try:
            # Validate form
            name = self.query_one("#sub-name", Input).value.strip()
            url = self.query_one("#sub-url", Input).value.strip()
            
            if not name or not url:
                self.notify("Name and URL are required", severity="warning")
                return
            
            # Gather form data
            sub_data = {
                'name': name,
                'type': self.query_one("#sub-type", Select).value,
                'source': url,
                'description': self.query_one("#sub-description", TextArea).text.strip(),
                'tags': [t.strip() for t in self.query_one("#sub-tags", Input).value.split(',') if t.strip()],
                'folder': self.query_one("#sub-folder", Input).value.strip() or None,
                'priority': int(self.query_one("#sub-priority", Select).value),
                'check_frequency': int(self.query_one("#sub-frequency", Select).value),
                'auto_ingest': self.query_one("#sub-auto-ingest", Checkbox).value,
            }
            
            # Add advanced options if needed
            auth_type = self.query_one("#sub-auth-type", Select).value
            if auth_type != "none":
                sub_data['auth_config'] = {'type': auth_type}
            
            headers_text = self.query_one("#sub-headers", TextArea).text.strip()
            if headers_text:
                try:
                    sub_data['custom_headers'] = json.loads(headers_text)
                except json.JSONDecodeError:
                    self.notify("Invalid JSON in custom headers", severity="error")
                    return
            
            # Save to database
            if self.selected_subscription:
                # Update existing
                self.db.update_subscription(self.selected_subscription, **sub_data)
                self.notify("Subscription updated", severity="information")
            else:
                # Create new
                sub_id = self.db.add_subscription(**sub_data)
                self.notify("Subscription created", severity="information")
            
            # Refresh list
            await self.refresh_subscription_list()
            
            # Clear form
            await self.handle_add_subscription(None)
            
        except Exception as e:
            logger.error(f"Error saving subscription: {e}")
            self.notify(f"Error: {str(e)}", severity="error")
    
    @on(ListView.Selected, "#subscription-list")
    async def handle_subscription_selected(self, event: ListView.Selected) -> None:
        """Handle subscription selection."""
        if event.item and event.item.data:
            sub = event.item.data
            self.selected_subscription = sub['id']
            
            # Load into form
            self.query_one("#sub-name", Input).value = sub['name']
            self.query_one("#sub-type", Select).value = sub['type']
            self.query_one("#sub-url", Input).value = sub['source']
            self.query_one("#sub-description", TextArea).load_text(sub.get('description', ''))
            
            # Tags
            tags = sub.get('tags', [])
            if isinstance(tags, str):
                tags = json.loads(tags) if tags else []
            self.query_one("#sub-tags", Input).value = ', '.join(tags)
            
            self.query_one("#sub-folder", Input).value = sub.get('folder', '')
            self.query_one("#sub-priority", Select).value = str(sub.get('priority', 3))
            self.query_one("#sub-frequency", Select).value = str(sub.get('check_frequency', 3600))
            self.query_one("#sub-auto-ingest", Checkbox).value = sub.get('auto_ingest', False)
    
    @on(Button.Pressed, "#check-all-btn")
    async def handle_check_all(self, event: Button.Pressed) -> None:
        """Handle check all subscriptions."""
        if self.is_checking:
            self.notify("Check already in progress", severity="warning")
            return
        
        self.is_checking = True
        event.button.disabled = True
        event.button.label = "Checking..."
        
        try:
            # Run check via worker
            await self.scheduler_worker.check_all_subscriptions()
            
        except Exception as e:
            logger.error(f"Error checking subscriptions: {e}")
            self.notify(f"Check failed: {str(e)}", severity="error")
        finally:
            self.is_checking = False
            event.button.disabled = False
            event.button.label = "Check All"
    
    @on(Button.Pressed, "#generate-briefing-btn")
    async def handle_generate_briefing(self, event: Button.Pressed) -> None:
        """Handle briefing generation."""
        try:
            # Get form values
            name = self.query_one("#briefing-name", Input).value
            template = self.query_one("#briefing-template", Select).value
            time_range = self.query_one("#briefing-time-range", Select).value
            sources = self.query_one("#briefing-sources", Select).value
            format = self.query_one("#briefing-format", Select).value
            enhance = self.query_one("#briefing-enhance", Checkbox).value
            
            # Calculate time range
            now = datetime.now(timezone.utc)
            if time_range == "today":
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = now
            elif time_range == "yesterday":
                yesterday = now - timedelta(days=1)
                start_time = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = yesterday.replace(hour=23, minute=59, second=59)
            elif time_range == "week":
                start_time = now - timedelta(days=7)
                end_time = now
            else:
                start_time = now - timedelta(days=1)
                end_time = now
            
            # Build source filter
            source_filter = None
            if sources == "high_priority":
                source_filter = {'priority': [1]}
            elif sources == "selected" and self.selected_subscription:
                source_filter = {'subscription_ids': [self.selected_subscription]}
            
            # Disable button
            event.button.disabled = True
            event.button.label = "Generating..."
            
            # Generate briefing (would be async with aggregation engine)
            # For now, create a sample
            briefing_content = f"""# {name}
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Executive Summary
This briefing covers updates from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}.

## Key Updates
- Sample update 1
- Sample update 2
- Sample update 3

## Recommendations
Based on the analyzed content, consider the following actions:
1. Review high-priority items
2. Follow up on trending topics
3. Archive processed content

---
*This is a sample briefing. Full implementation pending.*"""
            
            # Show preview
            self.query_one("#briefing-preview", TextArea).load_text(briefing_content)
            
            # Store for saving
            self.current_briefing = {
                'name': name,
                'content': briefing_content,
                'format': format,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.notify("Briefing generated", severity="information")
            
        except Exception as e:
            logger.error(f"Error generating briefing: {e}")
            self.notify(f"Generation failed: {str(e)}", severity="error")
        finally:
            event.button.disabled = False
            event.button.label = "Generate"
    
    # Event handlers from scheduler
    async def on_subscription_check_started(self, event: SubscriptionCheckStarted) -> None:
        """Handle subscription check started."""
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Checking {event.subscription_name or 'subscriptions'}..."
        await self.append_to_activity_log(log_entry)
    
    async def on_subscription_check_complete(self, event: SubscriptionCheckComplete) -> None:
        """Handle subscription check complete."""
        if event.success:
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {event.subscription_name}: {event.items_found} new items"
        else:
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✗ {event.subscription_name}: {event.error}"
        
        await self.append_to_activity_log(log_entry)
        
        # Refresh if items found
        if event.items_found > 0:
            await self.load_new_items()
    
    async def on_new_subscription_items(self, event: NewSubscriptionItems) -> None:
        """Handle new subscription items."""
        self.notify(f"{event.count} new items from {event.subscription_name}", severity="information")
        await self.load_new_items()
        await self.refresh_dashboard()
    
    async def append_to_activity_log(self, entry: str) -> None:
        """Append entry to activity log."""
        try:
            log = self.query_one("#activity-log", TextArea)
            current = log.text
            new_text = f"{entry}\n{current}"
            # Keep last 100 lines
            lines = new_text.split('\n')[:100]
            log.load_text('\n'.join(lines))
        except:
            pass


# End of SubscriptionWindow.py