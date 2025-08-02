# textual_scheduler_worker.py
# Description: Textual worker integration for subscription scheduling
#
# This module provides the bridge between the subscription scheduler
# and Textual's worker system for background processing.
#
# Imports
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Callable
import json
#
# Third-Party Imports
from textual.worker import Worker
from textual import work
from textual.app import App
from textual import events
from loguru import logger
#
# Local Imports
from .scheduler import SubscriptionScheduler, ScheduledTask, UpdatePattern
from .monitoring_engine import FeedMonitor, URLMonitor
from .website_monitor import WebsiteMonitor
from .aggregation_engine import AggregationEngine, AggregationConfig
from .briefing_generator import BriefingGenerator
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..Event_Handlers.subscription_events import (
    SubscriptionCheckStarted,
    SubscriptionCheckComplete,
    SubscriptionError,
    NewSubscriptionItems,
    BriefingGenerated
)
from ..Metrics.metrics_logger import log_histogram, log_counter, log_gauge
#
########################################################################################################################
#
# Textual Worker for Subscription Scheduling
#
########################################################################################################################

class SubscriptionSchedulerWorker(Worker):
    """
    Textual worker for running subscription checks in the background.
    
    This worker:
    - Manages the subscription scheduler lifecycle
    - Handles scheduled checks asynchronously
    - Posts events to update the UI
    - Manages concurrent check limits
    """
    
    def __init__(self, 
                 app: App,
                 db: SubscriptionsDB,
                 max_concurrent: int = 10,
                 check_interval: int = 60):
        """
        Initialize scheduler worker.
        
        Args:
            app: Textual app instance
            db: Subscriptions database
            max_concurrent: Maximum concurrent checks
            check_interval: Interval between scheduler runs (seconds)
        """
        super().__init__()
        self.app = app
        self.db = db
        self.max_concurrent = max_concurrent
        self.check_interval = check_interval
        
        # Initialize components
        self.scheduler = SubscriptionScheduler(db, max_concurrent)
        self.website_monitor = WebsiteMonitor(db)
        self.aggregation_engine = AggregationEngine(db)
        self.briefing_generator = BriefingGenerator(db)
        
        # State tracking
        self.is_running = False
        self.check_count = 0
        self.error_count = 0
        self.last_run = None
        
        # Register callbacks
        self.scheduler.add_check_callback(self._on_check_complete)
    
    @work(exclusive=True, thread=True)
    async def start_scheduler(self):
        """Start the subscription scheduler."""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        try:
            self.is_running = True
            logger.info("Starting subscription scheduler worker")
            
            # Post start event
            self.post_message(SubscriptionCheckStarted(worker=self))
            
            # Run scheduler
            await self._run_scheduler()
            
        except Exception as e:
            logger.error(f"Scheduler worker error: {str(e)}")
            self.error_count += 1
            self.post_message(SubscriptionError(
                worker=self,
                error=str(e),
                subscription_id=None
            ))
        finally:
            self.is_running = False
    
    async def _run_scheduler(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Get pending subscriptions
                pending = await self._get_pending_tasks()
                
                if pending:
                    logger.info(f"Processing {len(pending)} scheduled tasks")
                    
                    # Process tasks concurrently with limit
                    semaphore = asyncio.Semaphore(self.max_concurrent)
                    tasks = []
                    
                    for task in pending:
                        task_coro = self._process_task_with_limit(task, semaphore)
                        tasks.append(asyncio.create_task(task_coro))
                    
                    # Wait for all tasks
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for task, result in zip(pending, results):
                        if isinstance(result, Exception):
                            logger.error(f"Task {task.subscription_id} failed: {result}")
                            self.error_count += 1
                        else:
                            self.check_count += 1
                
                # Update metrics
                self._update_metrics()
                
                # Sleep until next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("Scheduler worker cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _get_pending_tasks(self) -> List[ScheduledTask]:
        """Get tasks that are due for execution."""
        now = datetime.now(timezone.utc)
        tasks = []
        
        # Get subscription checks
        pending_subs = self.db.get_pending_checks(limit=50)
        for sub in pending_subs:
            task = ScheduledTask(
                next_run=now.timestamp(),
                subscription_id=sub['id'],
                priority=sub.get('priority', 3),
                task_type='check'
            )
            tasks.append(task)
        
        # Get scheduled briefings
        # TODO: Implement briefing schedule check
        
        # Sort by priority
        tasks.sort()
        
        return tasks
    
    async def _process_task_with_limit(self, task: ScheduledTask, semaphore: asyncio.Semaphore):
        """Process a task with concurrency limit."""
        async with semaphore:
            return await self._process_task(task)
    
    async def _process_task(self, task: ScheduledTask):
        """Process a single scheduled task."""
        start_time = datetime.now(timezone.utc)
        
        try:
            if task.task_type == 'check':
                await self._process_subscription_check(task)
            elif task.task_type == 'briefing':
                await self._process_briefing_generation(task)
            elif task.task_type == 'cleanup':
                await self._process_cleanup(task)
            
            # Record success
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            log_histogram("scheduler_task_duration", duration, labels={
                "task_type": task.task_type,
                "success": "true"
            })
            
        except Exception as e:
            logger.error(f"Task processing error: {str(e)}")
            
            # Record failure
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            log_histogram("scheduler_task_duration", duration, labels={
                "task_type": task.task_type,
                "success": "false"
            })
            
            raise
    
    async def _process_subscription_check(self, task: ScheduledTask):
        """Process a subscription check."""
        subscription = self.db.get_subscription(task.subscription_id)
        if not subscription:
            logger.error(f"Subscription {task.subscription_id} not found")
            return
        
        logger.info(f"Checking subscription: {subscription['name']}")
        
        # Post start event
        self.post_message(SubscriptionCheckStarted(
            worker=self,
            subscription_id=task.subscription_id,
            subscription_name=subscription['name']
        ))
        
        try:
            # Determine check type
            sub_type = subscription['type']
            
            if sub_type in ['rss', 'atom', 'json_feed', 'podcast']:
                # Use feed monitor
                result = await self._check_feed(subscription)
            else:
                # Use website monitor for all other types
                result = await self.website_monitor.monitor_website(subscription)
            
            # Process results
            if result.get('items'):
                # Record new items
                self.db.record_check_result(
                    subscription['id'],
                    items=result['items'],
                    stats={
                        'new_items_found': len(result['items']),
                        'response_time_ms': result.get('response_time', 0)
                    }
                )
                
                # Post new items event
                self.post_message(NewSubscriptionItems(
                    worker=self,
                    subscription_id=subscription['id'],
                    subscription_name=subscription['name'],
                    items=result['items'],
                    count=len(result['items'])
                ))
            else:
                # No new items, just update last check
                self.db.record_check_result(subscription['id'])
            
            # Post completion event
            self.post_message(SubscriptionCheckComplete(
                worker=self,
                subscription_id=subscription['id'],
                subscription_name=subscription['name'],
                success=True,
                items_found=len(result.get('items', [])),
                error=None
            ))
            
        except Exception as e:
            logger.error(f"Check failed for {subscription['name']}: {str(e)}")
            
            # Record error
            self.db.record_check_error(subscription['id'], str(e))
            
            # Post error event
            self.post_message(SubscriptionError(
                worker=self,
                subscription_id=subscription['id'],
                subscription_name=subscription['name'],
                error=str(e)
            ))
            
            # Post completion with error
            self.post_message(SubscriptionCheckComplete(
                worker=self,
                subscription_id=subscription['id'],
                subscription_name=subscription['name'],
                success=False,
                items_found=0,
                error=str(e)
            ))
    
    async def _check_feed(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """Check RSS/Atom feed."""
        # Initialize feed monitor if needed
        if not hasattr(self, 'feed_monitor'):
            from .monitoring_engine import FeedMonitor, RateLimiter
            from .security import SecurityValidator
            self.feed_monitor = FeedMonitor(RateLimiter(), SecurityValidator())
        
        # Check feed
        items = await self.feed_monitor.check_feed(subscription)
        
        return {
            'subscription_id': subscription['id'],
            'items': items,
            'response_time': 0  # TODO: Track response time
        }
    
    async def _process_briefing_generation(self, task: ScheduledTask):
        """Process briefing generation task."""
        # TODO: Implement briefing generation
        logger.info(f"Processing briefing generation task: {task}")
    
    async def _process_cleanup(self, task: ScheduledTask):
        """Process cleanup task."""
        # TODO: Implement cleanup (old baselines, etc.)
        logger.info(f"Processing cleanup task: {task}")
    
    def _on_check_complete(self, subscription_id: int, result: Dict[str, Any]):
        """Callback when a check completes."""
        # Update patterns for adaptive scheduling
        if hasattr(self, 'update_patterns'):
            self.update_patterns.record_update(
                subscription_id,
                datetime.now(timezone.utc),
                bool(result.get('items'))
            )
    
    def _update_metrics(self):
        """Update scheduler metrics."""
        log_gauge("scheduler_check_count", self.check_count)
        log_gauge("scheduler_error_count", self.error_count)
        log_gauge("scheduler_is_running", 1 if self.is_running else 0)
        
        # Calculate check rate
        if self.last_run:
            elapsed = (datetime.now(timezone.utc) - self.last_run).total_seconds()
            if elapsed > 0:
                check_rate = self.check_count / elapsed * 3600  # Per hour
                log_gauge("scheduler_check_rate_per_hour", check_rate)
        
        self.last_run = datetime.now(timezone.utc)
    
    @work(exclusive=True)
    async def stop_scheduler(self):
        """Stop the scheduler."""
        logger.info("Stopping subscription scheduler worker")
        self.is_running = False
        
        # Clean up
        if hasattr(self, 'scheduler'):
            await self.scheduler.stop()
    
    @work
    async def check_subscription_now(self, subscription_id: int):
        """Check a specific subscription immediately."""
        task = ScheduledTask(
            next_run=datetime.now(timezone.utc).timestamp(),
            subscription_id=subscription_id,
            priority=5,  # High priority
            task_type='check'
        )
        
        await self._process_task(task)
    
    @work
    async def check_all_subscriptions(self):
        """Check all active subscriptions."""
        subscriptions = self.db.get_all_subscriptions(include_inactive=False)
        
        for sub in subscriptions:
            if sub['is_active'] and not sub['is_paused']:
                await self.check_subscription_now(sub['id'])
                
                # Brief pause between checks
                await asyncio.sleep(0.5)
    
    @work
    async def generate_briefing_now(self, config: Dict[str, Any]):
        """Generate a briefing immediately."""
        try:
            # Get items for briefing
            items = self.db.get_new_items(status='new', limit=1000)
            
            if not items:
                logger.info("No new items for briefing")
                return
            
            # Get subscriptions for metadata
            sub_ids = list(set(item['subscription_id'] for item in items))
            subscriptions = {
                sub['id']: sub 
                for sub in [self.db.get_subscription(sid) for sid in sub_ids]
                if sub
            }
            
            # Aggregate content
            agg_config = AggregationConfig.from_dict(config.get('aggregation', {}))
            aggregated = await self.aggregation_engine.aggregate_items(items, subscriptions)
            
            # Generate briefing
            briefing = await self.briefing_generator.generate_briefing(
                aggregated,
                template=config.get('template', 'default'),
                format=config.get('format', 'markdown')
            )
            
            # Post event
            self.post_message(BriefingGenerated(
                worker=self,
                briefing=briefing,
                item_count=aggregated.total_items,
                source_count=aggregated.total_sources
            ))
            
            logger.info(f"Generated briefing with {aggregated.total_items} items")
            
        except Exception as e:
            logger.error(f"Briefing generation failed: {str(e)}")
            self.post_message(SubscriptionError(
                worker=self,
                error=f"Briefing generation failed: {str(e)}"
            ))


# End of textual_scheduler_worker.py