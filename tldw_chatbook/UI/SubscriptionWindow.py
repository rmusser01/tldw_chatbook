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
from .Subscription_Modules import NotificationsInboxController, SubscriptionBackendController
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
    
    /* Activity log */
    #activity-log {
        height: 20;
        border: solid $primary;
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
        self.notifications_store = getattr(app_instance, "client_notifications_db", None)
        self.watchlist_scope_service = getattr(app_instance, "watchlist_scope_service", None)
        self.notification_dispatch_service = getattr(app_instance, "notification_dispatch_service", None)
        self.backend_controller = SubscriptionBackendController(
            window=self,
            app_instance=app_instance,
            scope_service=self.watchlist_scope_service,
            notification_dispatch_service=self.notification_dispatch_service,
        )
        self.notifications_controller = NotificationsInboxController(
            app_instance=app_instance,
            store=self.notifications_store,
        )
        
        # State
        self.selected_subscription: Optional[Union[int, str]] = None
        self.selected_items: List[int] = []
        self._selected_watch_item: Optional[Dict[str, Any]] = None
        self._selected_local_subscription_row: Optional[Dict[str, Any]] = None
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

            # Notifications tab
            with TabPane("Notifications", id="notifications"):
                yield from self._compose_notifications_tab()
            
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
                        [("High", "1"), ("Medium", "3"), ("Low", "5")],
                        id="sub-priority",
                        value="3"
                    )
                )
            
            # Schedule
            with FormFieldSet("Update Schedule"):
                yield FormField(
                    "Check Frequency",
                    Select(
                        [(k, str(v)) for k, v in SUBSCRIPTION_UPDATE_FREQUENCIES.items()],
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
                        [("None", "none"), ("Basic Auth", "basic"), ("Bearer Token", "bearer")],
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
        review_local_only = Static("", id="review-local-only-state")
        review_local_only.display = False
        yield review_local_only

        with Vertical(id="review-main"):
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

    def _compose_notifications_tab(self) -> ComposeResult:
        """Compose the client notifications inbox tab."""
        with Vertical():
            yield Label("Notifications Inbox")
            yield ListView(id="notifications-list")
            with Horizontal():
                yield Button("Mark Read", id="notification-mark-read-btn")
                yield Button("Dismiss", id="notification-dismiss-btn")
    
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
                read_only=True
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
                                ("Executive Summary", "executive_summary"),
                                ("Technical Digest", "technical_digest"),
                                ("News Briefing", "news_briefing"),
                                ("Custom Template", "custom")
                            ],
                            value="news_briefing"
                        )
                    )
                    
                    yield FormField(
                        "Time Range",
                        Select(
                            id="briefing-time-range",
                            options=[
                                ("Today", "today"),
                                ("Yesterday", "yesterday"),
                                ("Last 7 Days", "week"),
                                ("Custom Range", "custom")
                            ],
                            value="today"
                        )
                    )
                    
                    yield FormField(
                        "Sources",
                        Select(
                            id="briefing-sources",
                            options=[
                                ("All Sources", "all"),
                                ("High Priority Only", "high_priority"),
                                ("Selected Sources", "selected"),
                                ("By Tags", "tags")
                            ],
                            value="all"
                        )
                    )
                    
                    yield FormField(
                        "Format",
                        Select(
                            id="briefing-format",
                            options=[
                                ("Markdown", "markdown"),
                                ("HTML", "html"),
                                ("Plain Text", "text"),
                                ("JSON", "json")
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
                        [("OPML", "opml"), ("JSON", "json")],
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
        # Check if subscriptions dependencies are available
        from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
        if not DEPENDENCIES_AVAILABLE.get('subscriptions', False):
            from ..Utils.widget_helpers import alert_subscriptions_not_available
            # Show alert after a short delay to ensure UI is ready
            self.set_timer(0.1, lambda: alert_subscriptions_not_available(self))
        
        try:
            # Initialize database
            db_path = get_subscriptions_db_path()
            self.db = SubscriptionsDB(db_path, self.client_id)
            
            # Initialize components (if available)
            if BRIEFING_AVAILABLE:
                self.briefing_generator = BriefingGenerator(self.db)
                self.template_manager = BriefingTemplateManager()
            else:
                self.briefing_generator = None
                self.template_manager = None

            await self.refresh_backend_view()
            await self.refresh_dashboard()
            await self.load_briefing_templates()
            await self.refresh_notifications_inbox()
            
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

    def _runtime_backend(self) -> str:
        """Resolve the active runtime backend from app-level state."""
        for candidate in (
            getattr(self, "runtime_backend", None),
            getattr(self.app_instance, "current_runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
        ):
            normalized = str(candidate or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return "local"

    def _local_only_selectors(self, tab_id: str) -> Tuple[str, str]:
        return {
            "review": ("#review-local-only-state", "#review-main"),
        }.get(tab_id, ("", ""))

    def _render_local_only_state(self, *, tab_id: str, message: str) -> None:
        """Show a local-only degradation message for a backend-incompatible tab."""
        state_selector, main_selector = self._local_only_selectors(tab_id)
        if not state_selector or not main_selector:
            return
        try:
            state_widget = self.query_one(state_selector, Static)
            main_widget = self.query_one(main_selector)
        except (AssertionError, QueryError):
            return
        state_widget.update(message)
        state_widget.display = True
        main_widget.display = False

    def _clear_local_only_state(self, *, tab_id: str) -> None:
        """Restore a tab after leaving a degraded backend mode."""
        state_selector, main_selector = self._local_only_selectors(tab_id)
        if not state_selector or not main_selector:
            return
        try:
            state_widget = self.query_one(state_selector, Static)
            main_widget = self.query_one(main_selector)
        except (AssertionError, QueryError):
            return
        state_widget.display = False
        main_widget.display = True

    async def _render_watch_item_list(self, items: List[Dict[str, Any]]) -> None:
        """Populate the subscriptions list with normalized watchlist rows."""
        try:
            list_view = self.query_one("#subscription-list", ListView)
        except (AssertionError, QueryError):
            return

        await list_view.clear()
        for item in items:
            label = f"{item.get('title', item.get('name', 'Untitled'))} [{item.get('source_type', item.get('type', 'source'))}]"
            list_item = ListItem(Static(label, classes="subscription-item"))
            list_item.data = item
            await list_view.append(list_item)

    async def refresh_notifications_inbox(self) -> None:
        """Reload the notifications list from the local client store."""
        rows = await self.notifications_controller.load_rows()
        try:
            list_view = self.query_one("#notifications-list", ListView)
        except (AssertionError, QueryError):
            return

        await list_view.clear()
        for row in rows:
            status = "Read" if row.get("is_read") else "Unread"
            label = f"{status}: {row.get('title', 'Notification')}"
            item = ListItem(Static(label))
            item.data = row
            await list_view.append(item)

    def _selected_list_item_data(self, selector: str) -> Optional[Dict[str, Any]]:
        """Best-effort access to the selected ListView row payload."""
        try:
            list_view = self.query_one(selector, ListView)
        except (AssertionError, QueryError):
            return None

        highlighted_child = getattr(list_view, "highlighted_child", None)
        data = getattr(highlighted_child, "data", None)
        if isinstance(data, dict):
            return data

        index = getattr(list_view, "index", None)
        items = getattr(list_view, "children", None) or getattr(list_view, "items", None)
        if isinstance(index, int) and items is not None:
            try:
                selected = list(items)[index]
            except Exception:
                return None
            data = getattr(selected, "data", None)
            if isinstance(data, dict):
                return data
        return None

    def _active_watch_item_id(self) -> Optional[str]:
        """Return the normalized id for the currently selected watch item."""
        selected = self._selected_list_item_data("#subscription-list")
        if not selected and isinstance(self._selected_watch_item, dict):
            selected = self._selected_watch_item
        if not selected:
            if self.selected_subscription in (None, ""):
                return None
            return str(self.selected_subscription)
        item_id = selected.get("id")
        if item_id not in (None, ""):
            return str(item_id)
        source_id = selected.get("source_id")
        if source_id in (None, ""):
            return None
        entity_kind = str(selected.get("entity_kind") or ("subscription" if self._runtime_backend() == "local" else "watchlist_source"))
        return f"{self._runtime_backend()}:{entity_kind}:{source_id}"

    def _parse_tags_field(self) -> List[str]:
        return [tag.strip() for tag in self.query_one("#sub-tags", Input).value.split(',') if tag.strip()]

    def _current_watch_item_context(self) -> Dict[str, Any]:
        current: Dict[str, Any] = {}
        if isinstance(self._selected_watch_item, dict):
            current.update(self._selected_watch_item)
        if current or self.selected_subscription not in (None, ""):
            selected = self._selected_list_item_data("#subscription-list")
            if isinstance(selected, dict):
                current.update(selected)
        if self.selected_subscription not in (None, "") and "id" not in current:
            current["id"] = self.selected_subscription
        return current

    def _watch_item_payload_from_form(self) -> Dict[str, Any]:
        current = self._current_watch_item_context()
        payload: Dict[str, Any] = {
            "name": self.query_one("#sub-name", Input).value.strip(),
            "url": self.query_one("#sub-url", Input).value.strip(),
            "source_type": self.query_one("#sub-type", Select).value,
            "active": bool(current.get("active", True)),
            "tags": self._parse_tags_field(),
        }
        if current.get("id") not in (None, ""):
            payload["id"] = current["id"]
        if current.get("settings") is not None:
            payload["settings"] = current["settings"]
        return payload

    def _local_form_extras(self) -> Dict[str, Any]:
        extras: Dict[str, Any] = {
            "description": self.query_one("#sub-description", TextArea).text.strip() or None,
            "folder": self.query_one("#sub-folder", Input).value.strip() or None,
            "priority": int(self.query_one("#sub-priority", Select).value),
            "check_frequency": int(self.query_one("#sub-frequency", Select).value),
            "auto_ingest": self.query_one("#sub-auto-ingest", Checkbox).value,
        }

        auth_type = self.query_one("#sub-auth-type", Select).value
        extras["auth_config"] = {"type": auth_type} if auth_type != "none" else None

        headers_text = self.query_one("#sub-headers", TextArea).text.strip()
        extras["custom_headers"] = json.loads(headers_text) if headers_text else None
        return extras

    def _decode_json_mapping(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return {}
            if isinstance(decoded, dict):
                return decoded
        return {}

    def _clear_subscription_list_selection(self) -> None:
        try:
            list_view = self.query_one("#subscription-list", ListView)
        except (AssertionError, QueryError):
            return
        for attr, value in (("highlighted_child", None), ("index", None)):
            try:
                setattr(list_view, attr, value)
            except Exception:
                continue

    def _active_notification_id(self) -> Optional[int]:
        """Return the selected notification inbox row id."""
        selected = self._selected_list_item_data("#notifications-list")
        if not selected:
            return None
        notification_id = selected.get("id")
        if notification_id in (None, ""):
            return None
        return int(notification_id)

    async def refresh_backend_view(self) -> None:
        """Refresh the backend-aware subscription/watchlist shell."""
        runtime_backend = self._runtime_backend()
        await self.backend_controller.refresh_backend_view(runtime_backend=runtime_backend)

    async def stop_active_backend_workers(self) -> None:
        """Stop any backend-specific workers owned by the shell."""
        await self.backend_controller.stop_active_backend_workers()

    async def delete_selected_watch_item(self) -> Dict[str, Any] | None:
        """Delete the current watch item through the backend controller."""
        item_id = self._active_watch_item_id()
        if item_id is None:
            return None
        return await self.backend_controller.delete_watch_item(item_id)

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        """Refresh window state after a runtime backend switch."""
        self.runtime_backend = str(runtime_backend or "local").strip().lower() or "local"
        await self.handle_add_subscription(None)
        await self.refresh_backend_view()
        await self.refresh_notifications_inbox()
    
    @on(Button.Pressed, "#add-subscription-btn")
    async def handle_add_subscription(self, event: Button.Pressed) -> None:
        """Handle add subscription button."""
        del event
        # Clear form
        self.query_one("#sub-name", Input).value = ""
        self.query_one("#sub-url", Input).value = ""
        self.query_one("#sub-description", TextArea).load_text("")
        self.query_one("#sub-tags", Input).value = ""
        self.query_one("#sub-folder", Input).value = ""
        self.query_one("#sub-priority", Select).value = "3"
        self.query_one("#sub-frequency", Select).value = "3600"
        self.query_one("#sub-auto-ingest", Checkbox).value = False
        self.query_one("#sub-auth-type", Select).value = "none"
        self.query_one("#sub-headers", TextArea).load_text("")
        self.selected_subscription = None
        self._selected_watch_item = None
        self._selected_local_subscription_row = None
        self._clear_subscription_list_selection()
        
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
        del event
        try:
            # Validate form
            name = self.query_one("#sub-name", Input).value.strip()
            url = self.query_one("#sub-url", Input).value.strip()
            runtime_backend = self._runtime_backend()
            
            if not name or not url:
                self.notify("Name and URL are required", severity="warning")
                return
            
            if self.watchlist_scope_service is not None:
                payload = self._watch_item_payload_from_form()
                if runtime_backend == "local":
                    try:
                        payload.update(self._local_form_extras())
                    except json.JSONDecodeError:
                        self.notify("Invalid JSON in custom headers", severity="error")
                        return

                saved_item = await self.backend_controller.save_watch_item(payload)
                self._selected_watch_item = dict(saved_item)
                self.selected_subscription = saved_item.get("id")
                self.notify(
                    "Subscription updated" if payload.get("id") else "Subscription created",
                    severity="information",
                )
                await self.refresh_backend_view()
                return

            # Legacy fallback when the watchlist scope service is unavailable.
            sub_data = {
                'name': name,
                'type': self.query_one("#sub-type", Select).value,
                'source': url,
                'description': self.query_one("#sub-description", TextArea).text.strip(),
                'tags': self._parse_tags_field(),
                'folder': self.query_one("#sub-folder", Input).value.strip() or None,
                'priority': int(self.query_one("#sub-priority", Select).value),
                'check_frequency': int(self.query_one("#sub-frequency", Select).value),
                'auto_ingest': self.query_one("#sub-auto-ingest", Checkbox).value,
            }

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

            if self.selected_subscription:
                self.db.update_subscription(self.selected_subscription, **sub_data)
                self.notify("Subscription updated", severity="information")
            else:
                self.db.add_subscription(**sub_data)
                self.notify("Subscription created", severity="information")

            await self.refresh_subscription_list()
            
        except Exception as e:
            logger.error(f"Error saving subscription: {e}")
            self.notify(f"Error: {str(e)}", severity="error")
    
    @on(ListView.Selected, "#subscription-list")
    async def handle_subscription_selected(self, event: ListView.Selected) -> None:
        """Handle subscription selection."""
        if event.item and event.item.data:
            sub = dict(event.item.data)
            runtime_backend = self._runtime_backend()
            self._selected_watch_item = sub
            self.selected_subscription = sub.get('id')
            self._selected_local_subscription_row = None

            raw_local_row: Dict[str, Any] = {}
            if runtime_backend == "local" and self.db is not None and sub.get("source_id") not in (None, ""):
                local_row = self.db.get_subscription(int(sub["source_id"]))
                if isinstance(local_row, dict):
                    raw_local_row = dict(local_row)
                    self._selected_local_subscription_row = raw_local_row
            
            # Load into form
            self.query_one("#sub-name", Input).value = sub.get('title', sub.get('name', ''))
            self.query_one("#sub-type", Select).value = sub.get('source_type', sub.get('type', 'rss'))
            self.query_one("#sub-url", Input).value = sub.get('url', sub.get('source', ''))
            self.query_one("#sub-description", TextArea).load_text(raw_local_row.get('description', ''))
            
            # Tags
            tags = raw_local_row.get('tags', sub.get('tags', []))
            if isinstance(tags, str):
                stripped_tags = tags.strip()
                if stripped_tags.startswith("["):
                    tags = json.loads(stripped_tags) if stripped_tags else []
                else:
                    tags = [tag.strip() for tag in stripped_tags.split(",") if tag.strip()]
            self.query_one("#sub-tags", Input).value = ', '.join(tags)
            
            self.query_one("#sub-folder", Input).value = raw_local_row.get('folder', '')
            self.query_one("#sub-priority", Select).value = str(raw_local_row.get('priority', 3))
            self.query_one("#sub-frequency", Select).value = str(raw_local_row.get('check_frequency', 3600))
            self.query_one("#sub-auto-ingest", Checkbox).value = bool(raw_local_row.get('auto_ingest', False))

            auth_config = self._decode_json_mapping(raw_local_row.get('auth_config'))
            self.query_one("#sub-auth-type", Select).value = auth_config.get("type", "none")
            custom_headers = self._decode_json_mapping(raw_local_row.get('custom_headers'))
            self.query_one("#sub-headers", TextArea).load_text(
                json.dumps(custom_headers, indent=2) if custom_headers else ""
            )

    @on(Button.Pressed, "#delete-subscription-btn")
    async def handle_delete_subscription(self, event: Button.Pressed) -> None:
        """Delete the selected watch item through the backend-aware controller."""
        del event
        deleted = await self.delete_selected_watch_item()
        if deleted:
            await self.handle_add_subscription(None)
            await self.refresh_backend_view()
    
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

    @on(Button.Pressed, "#notification-mark-read-btn")
    async def handle_mark_notification_read(self, event: Button.Pressed) -> None:
        """Mark the selected notification as read."""
        del event
        notification_id = self._active_notification_id()
        if notification_id is None:
            return
        await self.notifications_controller.mark_read(notification_id, is_read=True)
        await self.refresh_notifications_inbox()

    @on(Button.Pressed, "#notification-dismiss-btn")
    async def handle_dismiss_notification(self, event: Button.Pressed) -> None:
        """Dismiss the selected notification from the inbox."""
        del event
        notification_id = self._active_notification_id()
        if notification_id is None:
            return
        await self.notifications_controller.dismiss(notification_id, is_dismissed=True)
        await self.refresh_notifications_inbox()
    
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
