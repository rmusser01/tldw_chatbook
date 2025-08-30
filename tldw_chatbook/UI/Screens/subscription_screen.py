"""
Subscription Screen
Screen wrapper for Subscription functionality in screen-based navigation.
"""

from textual.screen import Screen
from textual.app import ComposeResult
from textual.reactive import reactive
from typing import Optional, List, Dict, Any
from loguru import logger


class SubscriptionScreen(Screen):
    """Screen wrapper for Subscription management functionality."""
    
    # Screen-specific state
    subscriptions: reactive[List[Dict[str, Any]]] = reactive([])
    active_subscription: reactive[Optional[Dict[str, Any]]] = reactive(None)
    is_checking_updates: reactive[bool] = reactive(False)
    last_check_time: reactive[Optional[str]] = reactive(None)
    
    def compose(self) -> ComposeResult:
        """Compose the Subscription screen."""
        logger.info("Composing Subscription screen")
        
        # Check if SubscriptionWindow is available
        try:
            from ..SubscriptionWindow import SubscriptionWindow
            yield SubscriptionWindow()
        except ImportError:
            # Fallback if dependencies not installed
            from textual.widgets import Static
            yield Static(
                "[yellow]Subscription features require additional dependencies.[/yellow]\n"
                "Install with: pip install -e '.[subscriptions]'",
                classes="subscription-unavailable"
            )
    
    async def on_mount(self) -> None:
        """Initialize Subscription features when screen is mounted."""
        logger.info("Subscription screen mounted")
        
        # Try to get the Subscription window if available
        try:
            from ..SubscriptionWindow import SubscriptionWindow
            subscription_window = self.query_one(SubscriptionWindow)
            
            # Load subscriptions
            if hasattr(subscription_window, 'load_subscriptions'):
                subs = await subscription_window.load_subscriptions()
                self.subscriptions = subs
            
            # Initialize subscription features
            if hasattr(subscription_window, 'initialize'):
                await subscription_window.initialize()
        except (ImportError, Exception) as e:
            logger.warning(f"Subscription features unavailable: {e}")
    
    async def on_screen_suspend(self) -> None:
        """Clean up when screen is suspended (navigated away)."""
        logger.debug("Subscription screen suspended")
        
        # Stop any update checks
        if self.is_checking_updates:
            try:
                from ..SubscriptionWindow import SubscriptionWindow
                subscription_window = self.query_one(SubscriptionWindow)
                if hasattr(subscription_window, 'stop_update_check'):
                    await subscription_window.stop_update_check()
            except (ImportError, Exception):
                pass
            
            self.is_checking_updates = False
    
    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("Subscription screen resumed")
        
        # Refresh subscriptions list
        try:
            from ..SubscriptionWindow import SubscriptionWindow
            subscription_window = self.query_one(SubscriptionWindow)
            if hasattr(subscription_window, 'refresh_subscriptions'):
                subs = await subscription_window.refresh_subscriptions()
                self.subscriptions = subs
        except (ImportError, Exception):
            pass
    
    def add_subscription(self, url: str, name: str, check_interval: int = 3600) -> None:
        """Add a new subscription."""
        new_subscription = {
            "url": url,
            "name": name,
            "check_interval": check_interval,
            "last_checked": None,
            "is_active": True
        }
        
        # Add to local list
        current_subs = list(self.subscriptions)
        current_subs.append(new_subscription)
        self.subscriptions = current_subs
        
        logger.info(f"Added subscription: {name} ({url})")
    
    def remove_subscription(self, subscription_id: int) -> None:
        """Remove a subscription."""
        # Remove from local list
        if 0 <= subscription_id < len(self.subscriptions):
            subs = list(self.subscriptions)
            removed = subs.pop(subscription_id)
            self.subscriptions = subs
            logger.info(f"Removed subscription: {removed.get('name', 'Unknown')}")
    
    def toggle_subscription(self, subscription_id: int) -> None:
        """Toggle a subscription's active state."""
        if 0 <= subscription_id < len(self.subscriptions):
            subs = list(self.subscriptions)
            subs[subscription_id]["is_active"] = not subs[subscription_id].get("is_active", True)
            self.subscriptions = subs
            
            state = "activated" if subs[subscription_id]["is_active"] else "deactivated"
            logger.info(f"Subscription {subs[subscription_id].get('name', 'Unknown')} {state}")
    
    async def check_for_updates(self) -> None:
        """Check all active subscriptions for updates."""
        self.is_checking_updates = True
        logger.info("Checking subscriptions for updates...")
        
        # This would be implemented by SubscriptionWindow
        # Just updating state here for UI purposes
        from datetime import datetime
        self.last_check_time = datetime.now().isoformat()