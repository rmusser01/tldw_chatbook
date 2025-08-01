# scheduler.py
# Description: Background scheduler for subscription checking
#
# This module provides:
# - Async task scheduling for subscription checks
# - Priority-based queue management
# - Adaptive frequency based on update patterns
# - Integration with Textual workers
# - Concurrent check limiting
#
# Imports
import asyncio
import heapq
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
#
# Third-Party Imports
from loguru import logger
from textual.worker import Worker
#
# Local Imports
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..Metrics.metrics_logger import log_histogram, log_counter, log_gauge
from .monitoring_engine import FeedMonitor, URLMonitor, RateLimiter
from .security import SecurityValidator
#
########################################################################################################################
#
# Data Classes
#
########################################################################################################################

@dataclass(order=True)
class ScheduledTask:
    """Represents a scheduled subscription check."""
    next_run: float  # Unix timestamp
    subscription_id: int = field(compare=False)
    priority: int = field(compare=False)
    attempt: int = field(compare=False, default=0)
    task_type: str = field(compare=False, default='check')  # 'check', 'briefing', 'cleanup'
    
    def __lt__(self, other):
        """Compare tasks by next_run time, then priority."""
        if self.next_run == other.next_run:
            return self.priority > other.priority  # Higher priority first
        return self.next_run < other.next_run


class UpdatePattern:
    """Analyze and predict subscription update patterns."""
    
    def __init__(self):
        """Initialize pattern analyzer."""
        self.update_history = defaultdict(list)  # subscription_id -> list of update timestamps
        self.pattern_cache = {}  # subscription_id -> pattern info
        
    def record_update(self, subscription_id: int, timestamp: datetime, had_updates: bool):
        """
        Record an update check result.
        
        Args:
            subscription_id: Subscription ID
            timestamp: When the check occurred
            had_updates: Whether new items were found
        """
        history = self.update_history[subscription_id]
        
        if had_updates:
            history.append(timestamp.timestamp())
            
        # Keep only last 100 updates
        if len(history) > 100:
            history = history[-100:]
            self.update_history[subscription_id] = history
            
        # Invalidate pattern cache
        if subscription_id in self.pattern_cache:
            del self.pattern_cache[subscription_id]
    
    def analyze_pattern(self, subscription_id: int) -> Dict[str, Any]:
        """
        Analyze update pattern for a subscription.
        
        Args:
            subscription_id: Subscription ID
            
        Returns:
            Pattern analysis dictionary
        """
        # Check cache
        if subscription_id in self.pattern_cache:
            return self.pattern_cache[subscription_id]
        
        history = self.update_history.get(subscription_id, [])
        
        if len(history) < 5:
            # Not enough data
            pattern = {
                'type': 'unknown',
                'confidence': 0.0,
                'suggested_interval': None
            }
        else:
            # Calculate intervals between updates
            intervals = []
            for i in range(1, len(history)):
                interval = history[i] - history[i-1]
                intervals.append(interval)
            
            # Analyze intervals
            avg_interval = sum(intervals) / len(intervals)
            
            # Check for regular pattern
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            std_dev = variance ** 0.5
            
            if std_dev < avg_interval * 0.2:  # Low variance
                pattern = {
                    'type': 'regular',
                    'confidence': 1.0 - (std_dev / avg_interval),
                    'suggested_interval': int(avg_interval),
                    'average_interval': avg_interval
                }
            else:
                # Check for business hours pattern
                business_hours = self._check_business_hours_pattern(history)
                if business_hours['is_business_hours']:
                    pattern = business_hours
                else:
                    pattern = {
                        'type': 'sporadic',
                        'confidence': 0.5,
                        'suggested_interval': int(avg_interval * 1.5),
                        'average_interval': avg_interval
                    }
        
        # Cache the pattern
        self.pattern_cache[subscription_id] = pattern
        return pattern
    
    def _check_business_hours_pattern(self, timestamps: List[float]) -> Dict[str, Any]:
        """Check if updates follow business hours pattern."""
        # Convert to datetime objects
        datetimes = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]
        
        # Count updates by hour of day and day of week
        hour_counts = defaultdict(int)
        weekday_counts = defaultdict(int)
        
        for dt in datetimes:
            hour_counts[dt.hour] += 1
            weekday_counts[dt.weekday()] += 1
        
        # Check if mostly during business hours (9-17 UTC)
        business_hour_updates = sum(hour_counts[h] for h in range(9, 18))
        total_updates = sum(hour_counts.values())
        
        if total_updates > 0 and business_hour_updates / total_updates > 0.8:
            # Check if mostly on weekdays
            weekday_updates = sum(weekday_counts[d] for d in range(5))
            if weekday_updates / total_updates > 0.8:
                return {
                    'type': 'business_hours',
                    'confidence': 0.8,
                    'suggested_interval': 3600,  # Check hourly during business hours
                    'is_business_hours': True,
                    'peak_hours': list(range(9, 18))
                }
        
        return {'is_business_hours': False}
    
    def get_next_check_time(self, subscription_id: int, default_interval: int) -> datetime:
        """
        Calculate optimal next check time based on patterns.
        
        Args:
            subscription_id: Subscription ID
            default_interval: Default interval in seconds
            
        Returns:
            Next check datetime
        """
        pattern = self.analyze_pattern(subscription_id)
        now = datetime.now(timezone.utc)
        
        if pattern['type'] == 'regular' and pattern['confidence'] > 0.7:
            # Use suggested interval
            interval = pattern['suggested_interval']
        elif pattern['type'] == 'business_hours':
            # Check more frequently during business hours
            hour = now.hour
            if 9 <= hour < 18 and now.weekday() < 5:
                interval = pattern['suggested_interval']
            else:
                # Check less frequently outside business hours
                interval = default_interval * 2
        else:
            # Use default interval
            interval = default_interval
        
        return now + timedelta(seconds=interval)


class SubscriptionScheduler:
    """Main scheduler for subscription checking."""
    
    def __init__(self, db: SubscriptionsDB, max_concurrent: int = 10):
        """
        Initialize scheduler.
        
        Args:
            db: Subscriptions database
            max_concurrent: Maximum concurrent checks
        """
        self.db = db
        self.max_concurrent = max_concurrent
        self.task_queue: List[ScheduledTask] = []
        self.active_tasks: Set[int] = set()
        self.running = False
        self.rate_limiter = RateLimiter()
        self.security_validator = SecurityValidator()
        self.feed_monitor = FeedMonitor(self.rate_limiter, self.security_validator)
        self.url_monitor = URLMonitor(self.db, self.rate_limiter)
        self.update_patterns = UpdatePattern()
        self._workers: List[asyncio.Task] = []
        self._check_callbacks: List[Callable] = []
        
    def add_check_callback(self, callback: Callable):
        """Add a callback to be called when checks complete."""
        self._check_callbacks.append(callback)
        
    async def start(self):
        """Start the scheduler."""
        if self.running:
            return
            
        self.running = True
        logger.info("Starting subscription scheduler")
        
        # Load initial subscriptions
        await self._load_pending_subscriptions()
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        # Start scheduler loop
        asyncio.create_task(self._scheduler_loop())
        
        # Log startup metrics
        log_counter("subscription_scheduler_started")
        log_gauge("subscription_scheduler_workers", self.max_concurrent)
    
    async def stop(self):
        """Stop the scheduler."""
        if not self.running:
            return
            
        logger.info("Stopping subscription scheduler")
        self.running = False
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        log_counter("subscription_scheduler_stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Reload subscriptions periodically
                await self._load_pending_subscriptions()
                
                # Update metrics
                log_gauge("subscription_queue_size", len(self.task_queue))
                log_gauge("subscription_active_checks", len(self.active_tasks))
                
                # Sleep for a bit
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)
    
    async def _load_pending_subscriptions(self):
        """Load subscriptions that need checking."""
        try:
            # Get subscriptions due for checking
            pending = self.db.get_pending_checks(limit=100)
            
            now = time.time()
            added = 0
            
            for sub in pending:
                # Skip if already scheduled or active
                if sub['id'] in self.active_tasks:
                    continue
                    
                # Check if already in queue
                if any(task.subscription_id == sub['id'] for task in self.task_queue):
                    continue
                
                # Calculate next run time
                if sub['last_checked']:
                    last_checked = datetime.fromisoformat(sub['last_checked'])
                    # Use pattern analysis for next check
                    next_check = self.update_patterns.get_next_check_time(
                        sub['id'], 
                        sub['check_frequency']
                    )
                    next_run = next_check.timestamp()
                else:
                    # Never checked - run immediately
                    next_run = now
                
                # Create task
                task = ScheduledTask(
                    next_run=next_run,
                    subscription_id=sub['id'],
                    priority=sub['priority']
                )
                
                # Add to queue
                heapq.heappush(self.task_queue, task)
                added += 1
            
            if added > 0:
                logger.info(f"Added {added} subscriptions to check queue")
                
        except Exception as e:
            logger.error(f"Error loading pending subscriptions: {e}")
    
    async def _worker(self, worker_id: int):
        """Worker coroutine for processing subscription checks."""
        logger.info(f"Subscription worker {worker_id} started")
        
        while self.running:
            try:
                # Get next task
                task = await self._get_next_task()
                if not task:
                    await asyncio.sleep(1)
                    continue
                
                # Process task
                await self._process_subscription(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Subscription worker {worker_id} stopped")
    
    async def _get_next_task(self) -> Optional[ScheduledTask]:
        """Get the next task to process."""
        now = time.time()
        
        while self.task_queue:
            # Peek at next task
            if self.task_queue[0].next_run <= now:
                # Task is due
                task = heapq.heappop(self.task_queue)
                
                # Add to active set
                self.active_tasks.add(task.subscription_id)
                
                return task
            else:
                # No tasks due yet
                break
        
        return None
    
    async def _process_subscription(self, task: ScheduledTask):
        """Process a subscription check."""
        start_time = time.time()
        subscription_id = task.subscription_id
        
        try:
            # Get subscription details
            subscription = self.db.get_subscription(subscription_id)
            if not subscription:
                logger.warning(f"Subscription {subscription_id} not found")
                return
            
            # Skip if paused or inactive
            if subscription['is_paused'] or not subscription['is_active']:
                logger.info(f"Skipping paused/inactive subscription {subscription_id}")
                return
            
            logger.info(f"Checking subscription '{subscription['name']}' (ID: {subscription_id})")
            
            # Perform check based on type
            items = []
            if subscription['type'] in ['rss', 'atom', 'json_feed', 'podcast']:
                items = await self.feed_monitor.check_feed(subscription)
            elif subscription['type'] in ['url', 'url_list']:
                result = await self.url_monitor.check_url(subscription)
                if result:
                    items = [result]
            else:
                logger.warning(f"Unknown subscription type: {subscription['type']}")
            
            # Record successful check
            self.db.record_check_result(
                subscription_id,
                items=items,
                stats={
                    'new_items_found': len(items),
                    'response_time_ms': int((time.time() - start_time) * 1000)
                }
            )
            
            # Update pattern analysis
            self.update_patterns.record_update(
                subscription_id,
                datetime.now(timezone.utc),
                len(items) > 0
            )
            
            # Call callbacks
            for callback in self._check_callbacks:
                try:
                    await callback(subscription, items)
                except Exception as e:
                    logger.error(f"Error in check callback: {e}")
            
            # Log success
            logger.info(f"Subscription check complete: '{subscription['name']}' - {len(items)} new items")
            
            # Schedule next check
            next_check = self.update_patterns.get_next_check_time(
                subscription_id,
                subscription['check_frequency']
            )
            
            next_task = ScheduledTask(
                next_run=next_check.timestamp(),
                subscription_id=subscription_id,
                priority=subscription['priority'],
                attempt=0
            )
            heapq.heappush(self.task_queue, next_task)
            
        except Exception as e:
            logger.error(f"Error checking subscription {subscription_id}: {e}")
            
            # Record error
            self.db.record_check_error(subscription_id, str(e))
            
            # Retry with exponential backoff
            if task.attempt < 3:
                retry_delay = (2 ** task.attempt) * 60  # 1, 2, 4 minutes
                next_task = ScheduledTask(
                    next_run=time.time() + retry_delay,
                    subscription_id=subscription_id,
                    priority=task.priority,
                    attempt=task.attempt + 1
                )
                heapq.heappush(self.task_queue, next_task)
                logger.info(f"Scheduled retry for subscription {subscription_id} in {retry_delay} seconds")
            
        finally:
            # Remove from active set
            self.active_tasks.discard(subscription_id)
            
            # Log metrics
            duration = time.time() - start_time
            log_histogram("subscription_check_duration", duration, labels={
                "status": "success" if 'items' in locals() else "error"
            })


class TextualSchedulerWorker(Worker):
    """Textual worker for running the scheduler in the background."""
    
    def __init__(self, scheduler: SubscriptionScheduler):
        """
        Initialize worker.
        
        Args:
            scheduler: The scheduler instance to run
        """
        super().__init__()
        self.scheduler = scheduler
        
    async def run(self):
        """Run the scheduler."""
        try:
            await self.scheduler.start()
            
            # Keep running until cancelled
            while not self.is_cancelled:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            pass
        finally:
            await self.scheduler.stop()


# Helper function for quick scheduler creation
def create_scheduler(db: SubscriptionsDB, max_concurrent: int = 10) -> SubscriptionScheduler:
    """
    Create and configure a subscription scheduler.
    
    Args:
        db: Subscriptions database
        max_concurrent: Maximum concurrent checks
        
    Returns:
        Configured scheduler instance
    """
    scheduler = SubscriptionScheduler(db, max_concurrent)
    return scheduler


# End of scheduler.py