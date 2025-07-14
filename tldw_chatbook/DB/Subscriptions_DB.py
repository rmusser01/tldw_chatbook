# Subscriptions_DB.py
#########################################
# Subscriptions Database Library
# Manages RSS/Atom feeds and URL monitoring subscriptions
#
# This library provides a comprehensive subscription management system for:
# - RSS/Atom feed monitoring
# - URL change detection
# - API endpoint monitoring
# - Automated content ingestion
#
# Key Features:
# - Unified subscription model for multiple content types
# - Priority-based checking with adaptive scheduling
# - Smart error handling with auto-pause
# - Content deduplication across feeds
# - Performance optimization with conditional requests
# - Subscription health monitoring and statistics
# - Template-based quick setup
# - Smart filtering rules for automation
#
#########################################

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from urllib.parse import urlparse, urlunparse

# Third-Party Libraries
from loguru import logger

# Local Imports
from .base_db import BaseDB
from .sql_validation import validate_table_name, validate_column_name
from ..Metrics.metrics_logger import log_counter, log_histogram


# --- Custom Exceptions ---
class SubscriptionError(Exception):
    """Base exception for subscription-related errors."""
    pass


class AuthenticationError(SubscriptionError):
    """Exception for authentication failures."""
    pass


class RateLimitError(SubscriptionError):
    """Exception for rate limit violations."""
    pass


# --- Database Class ---
class SubscriptionsDB(BaseDB):
    """Database operations for subscription management."""
    
    _CURRENT_SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Union[str, Path], client_id: str = "default"):
        """
        Initialize the Subscriptions database.
        
        Args:
            db_path: Path to the SQLite database file or ':memory:'
            client_id: Client identifier for multi-client support
        """
        self._local = threading.local()
        super().__init__(db_path, client_id)
    
    def _initialize_schema(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
            PRAGMA foreign_keys = ON;
            
            -- Schema version tracking
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY NOT NULL
            );
            INSERT OR IGNORE INTO schema_version (version) VALUES (1);
            
            -- Unified subscription table with enhanced features
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('rss', 'atom', 'json_feed', 'url', 'url_list', 'podcast', 'sitemap', 'api')),
                source TEXT NOT NULL,
                description TEXT,
                
                -- Organization
                tags TEXT,
                priority INTEGER DEFAULT 3 CHECK(priority BETWEEN 1 AND 5),
                folder TEXT,
                
                -- Monitoring configuration
                check_frequency INTEGER DEFAULT 3600,
                last_checked DATETIME,
                last_successful_check DATETIME,
                last_error TEXT,
                error_count INTEGER DEFAULT 0,
                consecutive_failures INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                is_paused BOOLEAN DEFAULT 0,
                auto_pause_threshold INTEGER DEFAULT 10,
                
                -- Authentication & Headers
                auth_config TEXT,
                custom_headers TEXT,
                rate_limit_config TEXT,
                
                -- Processing options
                extraction_method TEXT DEFAULT 'auto',
                extraction_rules TEXT,
                processing_options TEXT,
                auto_ingest BOOLEAN DEFAULT 0,
                notification_config TEXT,
                
                -- Change detection for URLs
                change_threshold FLOAT DEFAULT 0.1,
                ignore_selectors TEXT,
                
                -- Performance optimization
                etag TEXT,
                last_modified TEXT,
                
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Items from subscriptions with enhanced metadata
            CREATE TABLE IF NOT EXISTS subscription_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subscription_id INTEGER NOT NULL,
                
                -- Common fields
                url TEXT NOT NULL,
                title TEXT,
                content_hash TEXT,
                published_date DATETIME,
                
                -- Enhanced metadata
                author TEXT,
                categories TEXT,
                enclosures TEXT,
                extracted_data TEXT,
                
                -- Status tracking
                status TEXT DEFAULT 'new' CHECK(status IN ('new', 'reviewed', 'ingested', 'ignored', 'error')),
                media_id INTEGER,
                processing_error TEXT,
                
                -- Change tracking for URLs
                previous_hash TEXT,
                change_percentage FLOAT,
                diff_summary TEXT,
                change_type TEXT CHECK(change_type IN (NULL, 'content', 'metadata', 'structural', 'new', 'removed')),
                
                -- Deduplication
                canonical_url TEXT,
                duplicate_of INTEGER,
                
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE,
                FOREIGN KEY (duplicate_of) REFERENCES subscription_items(id),
                UNIQUE(subscription_id, url, content_hash)
            );
            
            -- URL monitoring snapshots
            CREATE TABLE IF NOT EXISTS url_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subscription_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                extracted_content TEXT,
                raw_html TEXT,
                headers TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
            );
            
            -- Subscription statistics for health monitoring
            CREATE TABLE IF NOT EXISTS subscription_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subscription_id INTEGER NOT NULL,
                date DATE NOT NULL,
                
                -- Daily statistics
                checks_performed INTEGER DEFAULT 0,
                successful_checks INTEGER DEFAULT 0,
                new_items_found INTEGER DEFAULT 0,
                items_ingested INTEGER DEFAULT 0,
                errors_encountered INTEGER DEFAULT 0,
                
                -- Performance metrics
                avg_response_time_ms INTEGER,
                total_bytes_transferred INTEGER,
                
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE,
                UNIQUE(subscription_id, date)
            );
            
            -- Smart filters for automatic processing
            CREATE TABLE IF NOT EXISTS subscription_filters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subscription_id INTEGER,
                name TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                
                -- Filter conditions (JSON)
                conditions TEXT NOT NULL,
                
                -- Actions
                action TEXT NOT NULL CHECK(action IN ('auto_ingest', 'auto_ignore', 'tag', 'priority', 'notify')),
                action_params TEXT,
                
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
            );
            
            -- Subscription templates for quick setup
            CREATE TABLE IF NOT EXISTS subscription_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                
                -- Template configuration
                type TEXT NOT NULL,
                check_frequency INTEGER,
                extraction_method TEXT,
                extraction_rules TEXT,
                processing_options TEXT,
                auth_config_template TEXT,
                
                -- Popularity tracking
                usage_count INTEGER DEFAULT 0,
                
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Create indices
            CREATE INDEX IF NOT EXISTS idx_subscriptions_priority_active ON subscriptions(priority DESC, is_active, is_paused);
            CREATE INDEX IF NOT EXISTS idx_subscriptions_tags ON subscriptions(tags);
            CREATE INDEX IF NOT EXISTS idx_subscriptions_folder ON subscriptions(folder);
            CREATE INDEX IF NOT EXISTS idx_subscriptions_last_checked ON subscriptions(last_checked);
            CREATE INDEX IF NOT EXISTS idx_subscription_items_status_created ON subscription_items(subscription_id, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_subscription_items_canonical_url ON subscription_items(canonical_url);
            CREATE INDEX IF NOT EXISTS idx_url_snapshots_lookup ON url_snapshots(subscription_id, url, created_at);
            CREATE INDEX IF NOT EXISTS idx_subscription_stats_date ON subscription_stats(date);
            
            -- Create triggers for updated_at
            CREATE TRIGGER IF NOT EXISTS update_subscriptions_timestamp 
            AFTER UPDATE ON subscriptions
            BEGIN
                UPDATE subscriptions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
            
            CREATE TRIGGER IF NOT EXISTS update_subscription_items_timestamp 
            AFTER UPDATE ON subscription_items
            BEGIN
                UPDATE subscription_items SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
            
            CREATE TRIGGER IF NOT EXISTS update_subscription_filters_timestamp 
            AFTER UPDATE ON subscription_filters
            BEGIN
                UPDATE subscription_filters SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
            
            CREATE TRIGGER IF NOT EXISTS update_subscription_templates_timestamp 
            AFTER UPDATE ON subscription_templates
            BEGIN
                UPDATE subscription_templates SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
            """)
            conn.commit()
    
    @property
    def conn(self):
        """Thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = self._get_connection()
        return self._local.conn
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    # --- Core Subscription Management ---
    
    def add_subscription(self, name: str, type: str, source: str, 
                        tags: Optional[List[str]] = None, priority: int = 3,
                        folder: Optional[str] = None, auth_config: Optional[Dict] = None,
                        **kwargs) -> int:
        """
        Add a new subscription with enhanced metadata.
        
        Args:
            name: Display name for the subscription
            type: Type of subscription (rss, atom, url, etc.)
            source: URL or source identifier
            tags: List of tags for categorization
            priority: Priority level (1-5, default 3)
            folder: Folder/group for organization
            auth_config: Authentication configuration dict
            **kwargs: Additional fields (description, check_frequency, etc.)
            
        Returns:
            ID of the created subscription
        """
        start_time = time.time()
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Prepare fields
            fields = {
                'name': name,
                'type': type,
                'source': source,
                'tags': ','.join(tags) if tags else None,
                'priority': priority,
                'folder': folder,
                'auth_config': json.dumps(auth_config) if auth_config else None,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Add optional fields from kwargs
            allowed_fields = [
                'description', 'check_frequency', 'extraction_method',
                'extraction_rules', 'processing_options', 'auto_ingest',
                'notification_config', 'change_threshold', 'ignore_selectors',
                'custom_headers', 'rate_limit_config', 'auto_pause_threshold'
            ]
            
            for field in allowed_fields:
                if field in kwargs:
                    value = kwargs[field]
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    fields[field] = value
            
            # Build insert query
            columns = ', '.join(fields.keys())
            placeholders = ', '.join(['?' for _ in fields])
            
            cursor.execute(f"""
                INSERT INTO subscriptions ({columns})
                VALUES ({placeholders})
            """, list(fields.values()))
            
            subscription_id = cursor.lastrowid
            logger.info(f"Added subscription '{name}' (ID: {subscription_id}, Type: {type})")
            
            # Log success metrics
            duration = time.time() - start_time
            log_histogram("subscriptions_db_operation_duration", duration, labels={
                "operation": "add_subscription",
                "type": type,
                "priority": str(priority)
            })
            log_counter("subscriptions_db_operation_count", labels={
                "operation": "add_subscription",
                "type": type,
                "status": "success",
                "has_auth": "true" if auth_config else "false",
                "has_tags": "true" if tags else "false"
            })
            
            return subscription_id
    
    def get_subscription(self, subscription_id: int) -> Optional[Dict[str, Any]]:
        """Get a subscription by ID."""
        start_time = time.time()
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM subscriptions WHERE id = ?", (subscription_id,))
        row = cursor.fetchone()
        result = dict(row) if row else None
        
        # Log metrics
        duration = time.time() - start_time
        log_histogram("subscriptions_db_operation_duration", duration, labels={
            "operation": "get_subscription",
            "found": "true" if result else "false"
        })
        log_counter("subscriptions_db_operation_count", labels={
            "operation": "get_subscription",
            "status": "success",
            "found": "true" if result else "false"
        })
        
        return result
    
    def update_subscription(self, subscription_id: int, **kwargs) -> bool:
        """Update subscription fields."""
        start_time = time.time()
        
        if not kwargs:
            return False
            
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Build update query
            allowed_fields = [
                'name', 'description', 'tags', 'priority', 'folder',
                'check_frequency', 'is_active', 'is_paused', 'auth_config',
                'custom_headers', 'rate_limit_config', 'extraction_method',
                'extraction_rules', 'processing_options', 'auto_ingest',
                'notification_config', 'change_threshold', 'ignore_selectors',
                'etag', 'last_modified', 'auto_pause_threshold'
            ]
            
            updates = []
            values = []
            
            for field, value in kwargs.items():
                if field in allowed_fields:
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    elif field == 'tags' and isinstance(value, list):
                        value = ','.join(value)
                    updates.append(f"{field} = ?")
                    values.append(value)
            
            if not updates:
                return False
            
            values.append(subscription_id)
            cursor.execute(f"""
                UPDATE subscriptions 
                SET {', '.join(updates)}
                WHERE id = ?
            """, values)
            
            success = cursor.rowcount > 0
            
            # Log metrics
            duration = time.time() - start_time
            log_histogram("subscriptions_db_operation_duration", duration, labels={
                "operation": "update_subscription",
                "fields_updated": str(len(updates)),
                "success": str(success)
            })
            log_counter("subscriptions_db_operation_count", labels={
                "operation": "update_subscription",
                "status": "success" if success else "not_found",
                "fields_updated": str(len(updates))
            })
            
            return success
    
    def delete_subscription(self, subscription_id: int) -> bool:
        """Delete a subscription and all related data."""
        start_time = time.time()
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM subscriptions WHERE id = ?", (subscription_id,))
            success = cursor.rowcount > 0
            
            # Log metrics
            duration = time.time() - start_time
            log_histogram("subscriptions_db_operation_duration", duration, labels={
                "operation": "delete_subscription",
                "success": str(success)
            })
            log_counter("subscriptions_db_operation_count", labels={
                "operation": "delete_subscription",
                "status": "success" if success else "not_found"
            })
            
            return success
    
    def get_pending_checks(self, limit: int = 10, priority_order: bool = True) -> List[Dict[str, Any]]:
        """
        Get subscriptions due for checking, ordered by priority.
        
        Args:
            limit: Maximum number of subscriptions to return
            priority_order: Whether to order by priority (highest first)
            
        Returns:
            List of subscriptions due for checking
        """
        start_time = time.time()
        
        cursor = self.conn.cursor()
        
        order_clause = "ORDER BY priority DESC, last_checked ASC" if priority_order else "ORDER BY last_checked ASC"
        
        cursor.execute(f"""
            SELECT * FROM subscriptions
            WHERE is_active = 1 
            AND is_paused = 0
            AND (
                last_checked IS NULL 
                OR datetime(last_checked, '+' || check_frequency || ' seconds') <= datetime('now')
            )
            {order_clause}
            LIMIT ?
        """, (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        
        # Log metrics
        duration = time.time() - start_time
        log_histogram("subscriptions_db_operation_duration", duration, labels={
            "operation": "get_pending_checks",
            "limit": str(limit),
            "result_count": str(len(results))
        })
        log_counter("subscriptions_db_operation_count", labels={
            "operation": "get_pending_checks",
            "status": "success",
            "result_count": str(len(results)),
            "priority_order": str(priority_order)
        })
        
        return results
    
    def get_subscriptions_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Filter subscriptions by tag."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM subscriptions
            WHERE is_active = 1 AND tags LIKE ?
            ORDER BY name
        """, (f'%{tag}%',))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_subscriptions_by_folder(self, folder: str) -> List[Dict[str, Any]]:
        """Get all subscriptions in a folder."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM subscriptions
            WHERE is_active = 1 AND folder = ?
            ORDER BY priority DESC, name
        """, (folder,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # --- Check Results and Error Handling ---
    
    def record_check_result(self, subscription_id: int, items: List[Dict] = None, 
                          error: Optional[str] = None, stats: Optional[Dict] = None) -> None:
        """
        Record the result of a subscription check.
        
        Args:
            subscription_id: ID of the subscription
            items: List of new/changed items found
            error: Error message if check failed
            stats: Performance statistics (response_time_ms, bytes_transferred)
        """
        start_time = time.time()
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            now = datetime.now(timezone.utc).isoformat()
            
            if error:
                # Update error tracking
                cursor.execute("""
                    UPDATE subscriptions
                    SET last_checked = ?,
                        last_error = ?,
                        error_count = error_count + 1,
                        consecutive_failures = consecutive_failures + 1
                    WHERE id = ?
                """, (now, error, subscription_id))
                
                # Check if we should auto-pause
                cursor.execute("""
                    SELECT consecutive_failures, auto_pause_threshold
                    FROM subscriptions WHERE id = ?
                """, (subscription_id,))
                
                row = cursor.fetchone()
                if row and row['consecutive_failures'] >= row['auto_pause_threshold']:
                    cursor.execute("""
                        UPDATE subscriptions
                        SET is_paused = 1
                        WHERE id = ?
                    """, (subscription_id,))
                    logger.warning(f"Auto-paused subscription {subscription_id} after {row['consecutive_failures']} failures")
            
            else:
                # Successful check
                cursor.execute("""
                    UPDATE subscriptions
                    SET last_checked = ?,
                        last_successful_check = ?,
                        last_error = NULL,
                        error_count = 0,
                        consecutive_failures = 0
                    WHERE id = ?
                """, (now, now, subscription_id))
                
                # Add new items if provided
                if items:
                    for item in items:
                        self._add_subscription_item(cursor, subscription_id, item)
            
            # Update statistics if provided
            if stats:
                self._update_subscription_stats(subscription_id, stats, error is not None)
                
            # Log metrics
            duration = time.time() - start_time
            log_histogram("subscriptions_db_operation_duration", duration, labels={
                "operation": "record_check_result",
                "has_error": "true" if error else "false",
                "has_items": "true" if items else "false"
            })
            log_counter("subscriptions_db_operation_count", labels={
                "operation": "record_check_result",
                "status": "error" if error else "success",
                "item_count": str(len(items)) if items else "0",
                "auto_paused": "true" if error and 'Auto-paused' in str(error) else "false"
            })
    
    def record_check_error(self, subscription_id: int, error: str, should_pause: bool = False) -> None:
        """Record an error with optional auto-pause."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            now = datetime.now(timezone.utc).isoformat()
            
            cursor.execute("""
                UPDATE subscriptions
                SET last_checked = ?,
                    last_error = ?,
                    error_count = error_count + 1,
                    consecutive_failures = consecutive_failures + 1,
                    is_paused = ?
                WHERE id = ?
            """, (now, error, 1 if should_pause else 0, subscription_id))
    
    def reset_subscription_errors(self, subscription_id: int) -> None:
        """Reset error count after successful check."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE subscriptions
                SET error_count = 0,
                    consecutive_failures = 0,
                    last_error = NULL,
                    is_paused = 0
                WHERE id = ?
            """, (subscription_id,))
    
    # --- Item Management ---
    
    def get_new_items(self, subscription_id: Optional[int] = None, 
                     status: str = 'new', limit: int = 100) -> List[Dict[str, Any]]:
        """Get items with filtering and pagination."""
        cursor = self.conn.cursor()
        
        if subscription_id:
            cursor.execute("""
                SELECT i.*, s.name as subscription_name, s.type as subscription_type
                FROM subscription_items i
                JOIN subscriptions s ON i.subscription_id = s.id
                WHERE i.subscription_id = ? AND i.status = ?
                ORDER BY i.created_at DESC
                LIMIT ?
            """, (subscription_id, status, limit))
        else:
            cursor.execute("""
                SELECT i.*, s.name as subscription_name, s.type as subscription_type
                FROM subscription_items i
                JOIN subscriptions s ON i.subscription_id = s.id
                WHERE i.status = ?
                ORDER BY i.created_at DESC
                LIMIT ?
            """, (status, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def mark_item_status(self, item_id: int, status: str, 
                        media_id: Optional[int] = None, error: Optional[str] = None) -> bool:
        """Update item status with error tracking."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            updates = ['status = ?']
            values = [status]
            
            if media_id is not None:
                updates.append('media_id = ?')
                values.append(media_id)
            
            if error is not None:
                updates.append('processing_error = ?')
                values.append(error)
            
            values.append(item_id)
            
            cursor.execute(f"""
                UPDATE subscription_items
                SET {', '.join(updates)}
                WHERE id = ?
            """, values)
            
            return cursor.rowcount > 0
    
    def find_duplicate_items(self, item_url: str, item_hash: str) -> List[Dict[str, Any]]:
        """Check for existing duplicates."""
        cursor = self.conn.cursor()
        
        # Canonicalize URL for comparison
        canonical_url = self._canonicalize_url(item_url)
        
        cursor.execute("""
            SELECT * FROM subscription_items
            WHERE (canonical_url = ? OR content_hash = ?)
            AND status != 'ignored'
            ORDER BY created_at DESC
        """, (canonical_url, item_hash))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def bulk_update_items(self, item_ids: List[int], status: str) -> int:
        """Efficient bulk status updates."""
        if not item_ids:
            return 0
            
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in item_ids])
            cursor.execute(f"""
                UPDATE subscription_items
                SET status = ?
                WHERE id IN ({placeholders})
            """, [status] + item_ids)
            
            return cursor.rowcount
    
    # --- Statistics and Health Monitoring ---
    
    def update_subscription_stats(self, subscription_id: int, date: str, stats: Dict[str, Any]) -> None:
        """Record daily statistics."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Insert or update stats for the day
            cursor.execute("""
                INSERT INTO subscription_stats (subscription_id, date, checks_performed,
                    successful_checks, new_items_found, items_ingested, errors_encountered,
                    avg_response_time_ms, total_bytes_transferred)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subscription_id, date) DO UPDATE SET
                    checks_performed = checks_performed + excluded.checks_performed,
                    successful_checks = successful_checks + excluded.successful_checks,
                    new_items_found = new_items_found + excluded.new_items_found,
                    items_ingested = items_ingested + excluded.items_ingested,
                    errors_encountered = errors_encountered + excluded.errors_encountered,
                    avg_response_time_ms = (avg_response_time_ms + excluded.avg_response_time_ms) / 2,
                    total_bytes_transferred = total_bytes_transferred + excluded.total_bytes_transferred
            """, (
                subscription_id, date,
                stats.get('checks_performed', 1),
                stats.get('successful_checks', 0),
                stats.get('new_items_found', 0),
                stats.get('items_ingested', 0),
                stats.get('errors_encountered', 0),
                stats.get('avg_response_time_ms', 0),
                stats.get('total_bytes_transferred', 0)
            ))
    
    def get_subscription_health(self, subscription_id: int, days: int = 30) -> Dict[str, Any]:
        """Get health metrics for dashboard."""
        start_time = time.time()
        cursor = self.conn.cursor()
        
        # Get recent stats
        cursor.execute("""
            SELECT 
                SUM(checks_performed) as total_checks,
                SUM(successful_checks) as successful_checks,
                SUM(new_items_found) as total_items_found,
                SUM(items_ingested) as total_items_ingested,
                SUM(errors_encountered) as total_errors,
                AVG(avg_response_time_ms) as avg_response_time,
                SUM(total_bytes_transferred) as total_bytes
            FROM subscription_stats
            WHERE subscription_id = ?
            AND date >= date('now', '-' || ? || ' days')
        """, (subscription_id, days))
        
        stats = dict(cursor.fetchone() or {})
        
        # Calculate health score (0-100)
        if stats.get('total_checks', 0) > 0:
            success_rate = stats.get('successful_checks', 0) / stats['total_checks']
            stats['health_score'] = int(success_rate * 100)
        else:
            stats['health_score'] = 0
        
        # Get current subscription status
        cursor.execute("""
            SELECT consecutive_failures, last_error, is_paused
            FROM subscriptions WHERE id = ?
        """, (subscription_id,))
        
        current = cursor.fetchone()
        if current:
            stats.update(dict(current))
        
        # Log metrics
        duration = time.time() - start_time
        log_histogram("subscriptions_db_operation_duration", duration, labels={
            "operation": "get_subscription_health",
            "days": str(days)
        })
        log_counter("subscriptions_db_operation_count", labels={
            "operation": "get_subscription_health",
            "status": "success",
            "health_score": str(stats.get('health_score', 0))
        })
        
        return stats
    
    def get_failing_subscriptions(self, threshold: int = 5) -> List[Dict[str, Any]]:
        """Find subscriptions needing attention."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM subscriptions
            WHERE consecutive_failures >= ?
            OR (error_count > 0 AND last_successful_check < datetime('now', '-7 days'))
            ORDER BY consecutive_failures DESC, error_count DESC
        """, (threshold,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # --- Filters and Templates ---
    
    def add_filter(self, name: str, conditions: Dict[str, Any], action: str,
                  subscription_id: Optional[int] = None, action_params: Optional[Dict] = None) -> int:
        """Add smart filter rule."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO subscription_filters 
                (subscription_id, name, conditions, action, action_params)
                VALUES (?, ?, ?, ?, ?)
            """, (
                subscription_id,
                name,
                json.dumps(conditions),
                action,
                json.dumps(action_params) if action_params else None
            ))
            
            return cursor.lastrowid
    
    def get_active_filters(self, subscription_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get filters for processing."""
        cursor = self.conn.cursor()
        
        if subscription_id is not None:
            cursor.execute("""
                SELECT * FROM subscription_filters
                WHERE is_active = 1 AND (subscription_id = ? OR subscription_id IS NULL)
                ORDER BY subscription_id DESC
            """, (subscription_id,))
        else:
            cursor.execute("""
                SELECT * FROM subscription_filters
                WHERE is_active = 1 AND subscription_id IS NULL
            """)
        
        filters = []
        for row in cursor.fetchall():
            filter_dict = dict(row)
            filter_dict['conditions'] = json.loads(filter_dict['conditions'])
            if filter_dict['action_params']:
                filter_dict['action_params'] = json.loads(filter_dict['action_params'])
            filters.append(filter_dict)
        
        return filters
    
    def save_template(self, name: str, config: Dict[str, Any], category: Optional[str] = None) -> int:
        """Save subscription template."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO subscription_templates
                (name, description, category, type, check_frequency,
                 extraction_method, extraction_rules, processing_options, auth_config_template)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                config.get('description'),
                category,
                config['type'],
                config.get('check_frequency'),
                config.get('extraction_method'),
                json.dumps(config.get('extraction_rules')) if config.get('extraction_rules') else None,
                json.dumps(config.get('processing_options')) if config.get('processing_options') else None,
                json.dumps(config.get('auth_config_template')) if config.get('auth_config_template') else None
            ))
            
            return cursor.lastrowid
    
    def get_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve available templates."""
        cursor = self.conn.cursor()
        
        if category:
            cursor.execute("""
                SELECT * FROM subscription_templates
                WHERE category = ?
                ORDER BY usage_count DESC, name
            """, (category,))
        else:
            cursor.execute("""
                SELECT * FROM subscription_templates
                ORDER BY usage_count DESC, name
            """)
        
        templates = []
        for row in cursor.fetchall():
            template = dict(row)
            # Parse JSON fields
            for field in ['extraction_rules', 'processing_options', 'auth_config_template']:
                if template.get(field):
                    template[field] = json.loads(template[field])
            templates.append(template)
        
        return templates
    
    # --- Helper Methods ---
    
    def _add_subscription_item(self, cursor, subscription_id: int, item: Dict[str, Any]) -> int:
        """Add a new subscription item."""
        # Canonicalize URL for deduplication
        canonical_url = self._canonicalize_url(item['url'])
        
        # Check for duplicates
        cursor.execute("""
            SELECT id FROM subscription_items
            WHERE subscription_id = ? AND canonical_url = ? AND content_hash = ?
        """, (subscription_id, canonical_url, item.get('content_hash')))
        
        existing = cursor.fetchone()
        if existing:
            return existing['id']
        
        # Insert new item
        cursor.execute("""
            INSERT INTO subscription_items
            (subscription_id, url, title, content_hash, published_date,
             author, categories, enclosures, extracted_data, canonical_url,
             previous_hash, change_percentage, diff_summary, change_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subscription_id,
            item['url'],
            item.get('title'),
            item.get('content_hash'),
            item.get('published_date'),
            item.get('author'),
            json.dumps(item.get('categories')) if item.get('categories') else None,
            json.dumps(item.get('enclosures')) if item.get('enclosures') else None,
            json.dumps(item.get('extracted_data')) if item.get('extracted_data') else None,
            canonical_url,
            item.get('previous_hash'),
            item.get('change_percentage'),
            item.get('diff_summary'),
            item.get('change_type')
        ))
        
        return cursor.lastrowid
    
    def _update_subscription_stats(self, subscription_id: int, stats: Dict[str, Any], 
                                 had_error: bool) -> None:
        """Update subscription statistics."""
        today = datetime.now(timezone.utc).date().isoformat()
        
        self.update_subscription_stats(subscription_id, today, {
            'checks_performed': 1,
            'successful_checks': 0 if had_error else 1,
            'errors_encountered': 1 if had_error else 0,
            'avg_response_time_ms': stats.get('response_time_ms', 0),
            'total_bytes_transferred': stats.get('bytes_transferred', 0),
            'new_items_found': stats.get('new_items_found', 0),
            'items_ingested': stats.get('items_ingested', 0)
        })
    
    def _canonicalize_url(self, url: str) -> str:
        """Canonicalize URL for deduplication."""
        try:
            parsed = urlparse(url.lower())
            # Remove common tracking parameters
            # In a real implementation, this would be more sophisticated
            canonical = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path.rstrip('/'),
                '',  # params
                '',  # query (removed for now, could clean selectively)
                ''   # fragment
            ))
            return canonical
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse URL '{url}' for canonicalization: {e}")
            return url.lower()
    
    def get_all_subscriptions(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Get all subscriptions with optional filtering."""
        start_time = time.time()
        
        cursor = self.conn.cursor()
        
        if include_inactive:
            cursor.execute("SELECT * FROM subscriptions ORDER BY name")
        else:
            cursor.execute("SELECT * FROM subscriptions WHERE is_active = 1 ORDER BY name")
        
        results = [dict(row) for row in cursor.fetchall()]
        
        # Log metrics
        duration = time.time() - start_time
        log_histogram("subscriptions_db_operation_duration", duration, labels={
            "operation": "get_all_subscriptions",
            "include_inactive": str(include_inactive),
            "result_count": str(len(results))
        })
        log_counter("subscriptions_db_operation_count", labels={
            "operation": "get_all_subscriptions",
            "status": "success",
            "result_count": str(len(results))
        })
        
        return results
    
    def get_subscription_count(self, active_only: bool = True) -> Dict[str, int]:
        """Get count of subscriptions by type."""
        start_time = time.time()
        
        cursor = self.conn.cursor()
        
        where_clause = "WHERE is_active = 1" if active_only else ""
        
        cursor.execute(f"""
            SELECT type, COUNT(*) as count
            FROM subscriptions
            {where_clause}
            GROUP BY type
        """)
        
        results = {row['type']: row['count'] for row in cursor.fetchall()}
        
        # Log metrics
        duration = time.time() - start_time
        total_count = sum(results.values())
        log_histogram("subscriptions_db_operation_duration", duration, labels={
            "operation": "get_subscription_count",
            "active_only": str(active_only)
        })
        log_counter("subscriptions_db_operation_count", labels={
            "operation": "get_subscription_count",
            "status": "success",
            "total_count": str(total_count),
            "type_count": str(len(results))
        })
        
        return results
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# End of Subscriptions_DB.py