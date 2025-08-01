# subscription_events.py
# Description: Event handlers and message types for subscription monitoring system
#
# Imports
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Union
#
# Third-Party Imports
from loguru import logger
from textual.message import Message
from textual.worker import Worker, WorkerState
from textual.widgets import Button, Input, Select, ListView, ListItem, Static, TextArea, Checkbox
from textual.css.query import QueryError
#
# Local Imports
from ..DB.Subscriptions_DB import SubscriptionsDB, SubscriptionError, AuthenticationError, RateLimitError
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Custom Message Types for Subscription System
#
########################################################################################################################

class SubscriptionEvent(Message):
    """Base class for subscription-related events."""
    pass


class SubscriptionCheckStarted(SubscriptionEvent):
    """A subscription check has started."""
    def __init__(self, worker: Optional[Worker] = None, 
                 subscription_id: Optional[int] = None,
                 subscription_name: Optional[str] = None) -> None:
        super().__init__()
        self.worker = worker
        self.subscription_id = subscription_id
        self.subscription_name = subscription_name
        self.timestamp = datetime.now()


class NewSubscriptionItems(SubscriptionEvent):
    """New items available for review from subscriptions."""
    def __init__(self, worker: Optional[Worker] = None, 
                 items: List[Dict[str, Any]] = None, 
                 subscription_id: Optional[int] = None,
                 subscription_name: Optional[str] = None,
                 count: Optional[int] = None) -> None:
        super().__init__()
        self.worker = worker
        self.items = items or []
        self.subscription_id = subscription_id
        self.subscription_name = subscription_name
        self.count = count if count is not None else len(self.items)


class SubscriptionCheckComplete(SubscriptionEvent):
    """A subscription check has completed."""
    def __init__(self, worker: Optional[Worker] = None,
                 subscription_id: Optional[int] = None, 
                 subscription_name: Optional[str] = None,
                 success: bool = True,
                 items_found: int = 0,
                 error: Optional[str] = None,
                 stats: Optional[Dict] = None) -> None:
        super().__init__()
        self.worker = worker
        self.subscription_id = subscription_id
        self.subscription_name = subscription_name
        self.success = success
        self.items_found = items_found
        self.error = error
        self.stats = stats or {}
        self.timestamp = datetime.now()


class SubscriptionError(SubscriptionEvent):
    """Subscription check failed with error."""
    def __init__(self, worker: Optional[Worker] = None,
                 subscription_id: Optional[int] = None, 
                 subscription_name: Optional[str] = None,
                 error: str = "",
                 error_type: str = "general") -> None:
        super().__init__()
        self.worker = worker
        self.subscription_id = subscription_id
        self.subscription_name = subscription_name
        self.error = error
        self.error_type = error_type  # 'auth', 'rate_limit', 'network', 'parse', etc.
        self.timestamp = datetime.now()


class BriefingGenerated(SubscriptionEvent):
    """A briefing has been generated."""
    def __init__(self, worker: Optional[Worker] = None,
                 briefing: Optional[str] = None,
                 item_count: int = 0,
                 source_count: int = 0) -> None:
        super().__init__()
        self.worker = worker
        self.briefing = briefing
        self.item_count = item_count
        self.source_count = source_count
        self.timestamp = datetime.now()


class SubscriptionHealthUpdate(SubscriptionEvent):
    """Health status update for subscriptions."""
    def __init__(self, active_count: int, paused_count: int, 
                 error_count: int, failing_subscriptions: List[Dict]) -> None:
        super().__init__()
        self.active_count = active_count
        self.paused_count = paused_count
        self.error_count = error_count
        self.failing_subscriptions = failing_subscriptions


class BulkOperationComplete(SubscriptionEvent):
    """Bulk operation on subscription items completed."""
    def __init__(self, operation: str, item_count: int, 
                 success_count: int, error_count: int = 0) -> None:
        super().__init__()
        self.operation = operation  # 'accept', 'ignore', 'mark_reviewed'
        self.item_count = item_count
        self.success_count = success_count
        self.error_count = error_count


class SubscriptionImportComplete(SubscriptionEvent):
    """OPML or other format import completed."""
    def __init__(self, format: str, total: int, imported: int, errors: List[str] = None) -> None:
        super().__init__()
        self.format = format  # 'opml', 'json', 'csv'
        self.total = total
        self.imported = imported
        self.errors = errors or []


########################################################################################################################
#
# Event Handler Functions
#
########################################################################################################################

async def handle_add_subscription(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle adding a new subscription."""
    if event.button.id != "subscription-add-button":
        return
        
    logger.info("Add subscription button pressed")
    
    try:
        # Get form values
        type_select = app.query_one("#subscription-type-select", Select)
        url_input = app.query_one("#subscription-url-input", Input)
        name_input = app.query_one("#subscription-name-input", Input)
        desc_input = app.query_one("#subscription-description-input", Input)
        tags_input = app.query_one("#subscription-tags-input", Input)
        folder_input = app.query_one("#subscription-folder-input", Input)
        priority_select = app.query_one("#subscription-priority-select", Select)
        frequency_select = app.query_one("#subscription-frequency-select", Select)
        
        # Validate required fields
        if not url_input.value.strip():
            app.notify("Please enter a URL/Feed address", severity="warning")
            return
            
        if not name_input.value.strip():
            app.notify("Please enter a subscription name", severity="warning")
            return
        
        # Get authentication config if provided
        auth_type = app.query_one("#subscription-auth-type", Select).value
        auth_config = None
        if auth_type != "none":
            username = app.query_one("#subscription-auth-username", Input).value
            password = app.query_one("#subscription-auth-password", Input).value
            if username or password:
                auth_config = {
                    "type": auth_type,
                    "username": username,
                    "password": password  # In real implementation, this should be encrypted
                }
        
        # Get custom headers
        custom_headers = None
        headers_text = app.query_one("#subscription-custom-headers", TextArea).text.strip()
        if headers_text:
            try:
                custom_headers = json.loads(headers_text)
            except json.JSONDecodeError:
                app.notify("Invalid JSON in custom headers", severity="error")
                return
        
        # Get advanced options
        auto_ingest = app.query_one("#subscription-auto-ingest", Checkbox).value
        extract_full = app.query_one("#subscription-extract-full", Checkbox).value
        change_threshold = app.query_one("#subscription-change-threshold", Input).value
        ignore_selectors = app.query_one("#subscription-ignore-selectors", TextArea).text.strip()
        rate_limit = app.query_one("#subscription-rate-limit", Input).value
        auto_pause_threshold = app.query_one("#subscription-auto-pause-threshold", Input).value
        
        # Parse tags
        tags = [tag.strip() for tag in tags_input.value.split(',') if tag.strip()]
        
        # Initialize database if needed
        if not hasattr(app, 'subscriptions_db'):
            from ..config import get_subscriptions_db_path
            db_path = get_subscriptions_db_path()
            app.subscriptions_db = SubscriptionsDB(db_path, app.client_id)
        
        # Add subscription to database
        try:
            subscription_id = app.subscriptions_db.add_subscription(
                name=name_input.value.strip(),
                type=type_select.value,
                source=url_input.value.strip(),
                description=desc_input.value.strip() or None,
                tags=tags,
                priority=int(priority_select.value),
                folder=folder_input.value.strip() or None,
                auth_config=auth_config,
                check_frequency=int(frequency_select.value),
                custom_headers=custom_headers,
                auto_ingest=auto_ingest,
                extraction_method='full' if extract_full else 'auto',
                change_threshold=float(change_threshold) / 100,
                ignore_selectors=ignore_selectors or None,
                rate_limit_config={"requests_per_minute": int(rate_limit)} if rate_limit else None,
                auto_pause_threshold=int(auto_pause_threshold)
            )
            
            app.notify(f"Successfully added subscription: {name_input.value}", severity="information")
            
            # Clear form
            url_input.value = ""
            name_input.value = ""
            desc_input.value = ""
            tags_input.value = ""
            folder_input.value = ""
            
            # Refresh subscription list
            await refresh_subscription_list(app)
            
        except Exception as e:
            logger.error(f"Error adding subscription: {e}")
            app.notify(f"Error adding subscription: {str(e)}", severity="error")
            
    except QueryError as e:
        logger.error(f"Error finding form elements: {e}")
        app.notify("Error: Could not find form elements", severity="error")


async def handle_check_all_subscriptions(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle checking all active subscriptions."""
    if event.button.id != "subscription-check-all-button":
        return
        
    logger.info("Check all subscriptions button pressed")
    
    # Disable button during check
    event.button.disabled = True
    event.button.label = "Checking..."
    
    try:
        # Start check worker
        app.run_worker(
            check_all_subscriptions_worker,
            app,
            name="check_all_subscriptions"
        )
    except Exception as e:
        logger.error(f"Error starting subscription check: {e}")
        app.notify(f"Error starting check: {str(e)}", severity="error")
        event.button.disabled = False
        event.button.label = "Check All Now"


async def handle_subscription_item_action(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle actions on subscription items (accept, ignore, review)."""
    if event.button.id not in ["subscription-accept-button", "subscription-ignore-button", 
                               "subscription-mark-reviewed-button"]:
        return
    
    action_map = {
        "subscription-accept-button": ("accept", "ingested"),
        "subscription-ignore-button": ("ignore", "ignored"),
        "subscription-mark-reviewed-button": ("review", "reviewed")
    }
    
    action_name, new_status = action_map[event.button.id]
    logger.info(f"Subscription item action: {action_name}")
    
    try:
        # Get selected items from list
        items_list = app.query_one("#subscription-new-items-list", ListView)
        selected_items = []
        
        # This is placeholder - in real implementation, track selected items properly
        # For now, we'll just show the notification
        if not selected_items:
            app.notify("Please select items first", severity="warning")
            return
        
        # Process items
        success_count = 0
        error_count = 0
        
        if action_name == "accept":
            # Import ingestion worker
            from .subscription_ingest_worker import bulk_ingest_subscription_items
            
            # Ingest items
            try:
                success_count, error_count = await bulk_ingest_subscription_items(app, selected_items)
            except Exception as e:
                logger.error(f"Error ingesting items: {e}")
                app.notify(f"Error ingesting items: {str(e)}", severity="error")
                return
        else:
            # For other actions, just update status
            for item_id in selected_items:
                try:
                    if hasattr(app, 'subscriptions_db'):
                        app.subscriptions_db.mark_item_status(item_id, new_status)
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing item {item_id}: {e}")
                    error_count += 1
        
        # Post completion event
        app.post_message(BulkOperationComplete(
            operation=action_name,
            item_count=len(selected_items),
            success_count=success_count,
            error_count=error_count
        ))
        
        # Refresh items list
        await refresh_subscription_items(app)
        
    except Exception as e:
        logger.error(f"Error in subscription item action: {e}")
        app.notify(f"Error processing items: {str(e)}", severity="error")


########################################################################################################################
#
# Worker Functions
#
########################################################################################################################

async def check_all_subscriptions_worker(app: 'TldwCli') -> None:
    """Worker function to check all active subscriptions."""
    logger.info("Starting subscription check worker")
    
    try:
        if not hasattr(app, 'subscriptions_db'):
            raise SubscriptionError("Subscription database not initialized")
        
        # Get pending subscriptions
        pending = app.subscriptions_db.get_pending_checks(limit=50, priority_order=True)
        
        if not pending:
            app.call_from_thread(app.notify, "No subscriptions due for checking", severity="information")
            return
        
        total_new_items = 0
        errors = []
        
        # Update status
        app.call_from_thread(
            update_subscription_status,
            app,
            f"Checking {len(pending)} subscriptions..."
        )
        
        for i, subscription in enumerate(pending):
            try:
                # Update progress
                app.call_from_thread(
                    update_subscription_status,
                    app,
                    f"Checking {i+1}/{len(pending)}: {subscription['name']}"
                )
                
                # Import monitoring engine components
                from ..Subscriptions import FeedMonitor, URLMonitor, RateLimiter, SecurityValidator
                
                # Create monitors if not already created
                if not hasattr(app, '_feed_monitor'):
                    rate_limiter = RateLimiter()
                    security_validator = SecurityValidator()
                    app._feed_monitor = FeedMonitor(rate_limiter, security_validator)
                    app._url_monitor = URLMonitor(app.subscriptions_db, rate_limiter)
                
                # Check subscription based on type
                items = []
                if subscription['type'] in ['rss', 'atom', 'json_feed', 'podcast']:
                    items = await app._feed_monitor.check_feed(subscription)
                elif subscription['type'] in ['url', 'url_list']:
                    result = await app._url_monitor.check_url(subscription)
                    if result:
                        items = [result]
                
                # Record result
                app.subscriptions_db.record_check_result(
                    subscription['id'],
                    items=items,
                    stats={'new_items_found': len(items)}
                )
                
                total_new_items += len(items)
                
                # Post event for UI update
                if len(items) > 0:
                    app.call_from_thread(
                        app.post_message,
                        NewSubscriptionItems(items, subscription['name'])
                    )
                
            except Exception as e:
                logger.error(f"Error checking subscription {subscription['name']}: {e}")
                errors.append(f"{subscription['name']}: {str(e)}")
                
                # Record error
                app.subscriptions_db.record_check_error(
                    subscription['id'],
                    str(e),
                    should_pause=isinstance(e, (AuthenticationError, RateLimitError))
                )
                
                # Post error event
                app.call_from_thread(
                    app.post_message,
                    SubscriptionError(
                        subscription['id'],
                        subscription['name'],
                        str(e),
                        error_type='auth' if isinstance(e, AuthenticationError) else 'general'
                    )
                )
        
        # Update final status
        status_msg = f"Check complete: {total_new_items} new items found"
        if errors:
            status_msg += f", {len(errors)} errors"
        
        app.call_from_thread(update_subscription_status, app, status_msg)
        app.call_from_thread(
            app.notify,
            status_msg,
            severity="warning" if errors else "information"
        )
        
        # Update health dashboard
        await update_health_dashboard(app)
        
    except Exception as e:
        logger.error(f"Critical error in subscription check worker: {e}")
        app.call_from_thread(
            app.notify,
            f"Error checking subscriptions: {str(e)}",
            severity="error"
        )
    finally:
        # Re-enable check button
        try:
            button = app.query_one("#subscription-check-all-button", Button)
            app.call_from_thread(setattr, button, 'disabled', False)
            app.call_from_thread(setattr, button, 'label', "Check All Now")
        except:
            pass


########################################################################################################################
#
# Helper Functions
#
########################################################################################################################

async def refresh_subscription_list(app: 'TldwCli') -> None:
    """Refresh the active subscriptions list."""
    try:
        if not hasattr(app, 'subscriptions_db'):
            return
            
        subscriptions = app.subscriptions_db.get_all_subscriptions(include_inactive=False)
        list_view = app.query_one("#subscription-active-list", ListView)
        
        await list_view.clear()
        
        for sub in subscriptions:
            # Format subscription info
            status = "✓" if sub['is_active'] and not sub['is_paused'] else "⏸"
            if sub['consecutive_failures'] > 0:
                status = "⚠️"
            
            info = f"{status} {sub['name']} [{sub['type']}]"
            if sub['last_checked']:
                info += f" - Last: {sub['last_checked']}"
            
            item = ListItem(Static(info))
            await list_view.append(item)
            
    except Exception as e:
        logger.error(f"Error refreshing subscription list: {e}")


async def refresh_subscription_items(app: 'TldwCli') -> None:
    """Refresh the subscription items list."""
    try:
        if not hasattr(app, 'subscriptions_db'):
            return
            
        items = app.subscriptions_db.get_new_items(limit=100)
        list_view = app.query_one("#subscription-new-items-list", ListView)
        
        await list_view.clear()
        
        for item in items:
            # Format item info
            info = f"□ {item['title'] or 'Untitled'}"
            info += f"\n   From: {item['subscription_name']} | {item['created_at']}"
            
            list_item = ListItem(Static(info))
            await list_view.append(list_item)
            
    except Exception as e:
        logger.error(f"Error refreshing subscription items: {e}")


def update_subscription_status(app: 'TldwCli', message: str) -> None:
    """Update the subscription status area."""
    try:
        status_area = app.query_one("#subscription-status-area", TextArea)
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_area.load_text(f"[{timestamp}] {message}\n" + status_area.text)
    except Exception as e:
        logger.error(f"Error updating status: {e}")


async def update_health_dashboard(app: 'TldwCli') -> None:
    """Update the subscription health dashboard."""
    try:
        if not hasattr(app, 'subscriptions_db'):
            return
        
        # Get counts
        counts = app.subscriptions_db.get_subscription_count(active_only=False)
        active_count = sum(counts.values())
        
        # Get failing subscriptions
        failing = app.subscriptions_db.get_failing_subscriptions(threshold=5)
        
        # Update UI elements
        app.query_one("#stat-active-count", Static).update(str(active_count))
        app.query_one("#stat-error-count", Static).update(str(len(failing)))
        
        # Show/hide failing alert
        if failing:
            alert = app.query_one("#failing-subscriptions-alert")
            alert.remove_class("hidden")
            
            # Update failing list
            failing_list = app.query_one("#failing-subscriptions-list", ListView)
            await failing_list.clear()
            
            for sub in failing[:5]:  # Show top 5
                info = f"⚠️ {sub['name']} - {sub['consecutive_failures']} failures"
                await failing_list.append(ListItem(Static(info)))
                
    except Exception as e:
        logger.error(f"Error updating health dashboard: {e}")


# End of subscription_events.py