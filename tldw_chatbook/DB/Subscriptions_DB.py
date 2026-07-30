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
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse, urlunparse

# Third-Party Libraries
from loguru import logger

# Local Imports
from .private_sqlite import connect_private_sqlite
from .base_db import BaseDB
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
#: The one definition of `site_configs`. Applied by
#: `SubscriptionsDB._initialize_schema`, which owns the table, and by
#: `ensure_site_configs_schema` for `SiteConfigManager`, which needs the table
#: to exist but must not impose the whole subscriptions schema on a path a
#: caller supplied. Two call sites, one DDL: duplicating it would recreate the
#: "no class's schema describes this file" problem TASK-896 set out to fix.
SITE_CONFIGS_DDL = """
    CREATE TABLE IF NOT EXISTS site_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT UNIQUE NOT NULL,
        config_data TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_site_configs_domain
    ON site_configs(domain);
"""


def ensure_site_configs_schema(db_path) -> None:
    """Create `site_configs` on `db_path`, and nothing else.

    `SiteConfigManager` accepts a caller-supplied path. Opening a full
    `SubscriptionsDB` there just to guarantee this one table would run the
    entire subscriptions schema against that file -- around fifteen unrelated
    tables plus their indices and triggers -- a side effect no caller asked
    for. This applies only the table the manager actually needs.
    """
    # Goes through `connect_private_sqlite`, not `sqlite3.connect`. Every
    # database this app opens is meant to pass the private-path guards
    # (ownership, no shared-writable ancestor, no untrusted symlink) and land
    # with private file modes. A raw connect here would open a database
    # outside all of that, and would be an undocumented raw call site --
    # `Tests/DB/test_private_sqlite_inventory.py` audits exactly that and
    # caught this when the helper was first written.
    with closing(connect_private_sqlite("db.subscriptions.site_configs", db_path)) as conn:
        conn.executescript(SITE_CONFIGS_DDL)
        conn.commit()


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

    def _get_connection(self) -> sqlite3.Connection:
        """Return a connection with foreign-key enforcement enabled.

        ``PRAGMA foreign_keys`` is per-connection and defaults to OFF, and
        ``BaseDB._get_connection`` sets only ``row_factory``. Without this
        override every ``ON DELETE CASCADE`` in this schema is inert, which
        silently orphaned ``subscription_items`` whenever a subscription was
        deleted. Matches ``ChaChaNotes_DB`` and ``Client_Media_DB_v2``, which
        each enable it per connection.
        """
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _initialize_schema(self):
        """Initialize the database schema.

        Runs on ``self.conn`` (the thread-local connection everything else on
        this thread reuses) rather than a throwaway connection that used to
        get closed immediately afterwards. For a file-backed database both
        approaches land on the same file, so it made no observable
        difference. For ``:memory:``, every ``sqlite3.connect(':memory:')``
        call opens a brand-new, private, empty database -- so the old
        close-then-reopen sequence built the schema somewhere the rest of
        the class could never see, leaving ``.conn`` pointed at zero tables.
        Matches the pattern ``ChaChaNotes_DB._initialize_schema`` already
        uses (``self.get_connection()``) for the same reason.

        Trade-off carried over from that same precedent: this only makes the
        constructing thread's connection schema-bearing. If a *second*
        thread later touches ``.conn`` on this same in-memory instance, its
        own thread-local slot is empty, so the ``.conn`` property lazily
        opens yet another private ``:memory:`` connection with no schema --
        identical to the limitation already accepted in ``ChaChaNotes_DB``.
        The only current ``:memory:`` caller, ``WatchlistPreviewService.
        preview()``, constructs, uses, and discards its instance within a
        single coroutine on one thread (it is scheduled via
        ``run_worker(coroutine)``, not ``thread=True``), so this does not
        apply there today; a future caller that hands an in-memory
        ``SubscriptionsDB`` across a thread boundary would need a different
        fix (e.g. a shared-cache ``file::memory:?cache=shared`` URI plus a
        dedicated keepalive connection).
        """
        with self.transaction() as conn:
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
                change_threshold FLOAT DEFAULT 0.0,
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
                -- Stable hash of the ignore_selectors/extraction_method in
                -- force when this snapshot's extracted_content was captured
                -- (TASK-1362). Nullable: pre-migration rows have none, and
                -- comparing across a settings change must re-baseline
                -- rather than diff (see Subscriptions/noise_defaults.py).
                extraction_fingerprint TEXT,
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

            -- Local watchlist run history. Owned here rather than created
            -- lazily by LocalWatchlistsService, so one place owns schema and
            -- additive migrations can ALTER it unconditionally.
            CREATE TABLE IF NOT EXISTS local_watchlist_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                job_id INTEGER,
                batch_id TEXT,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                stats_json TEXT,
                error_msg TEXT,
                log_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES subscriptions(id) ON DELETE CASCADE
            );

            -- Local watchlist alert rules. Same reasoning as
            -- local_watchlist_runs above: owned here rather than created
            -- lazily by LocalWatchlistsService, so a lazily-created table can
            -- never race an additive migration that needs it to already
            -- exist.
            CREATE TABLE IF NOT EXISTS local_watchlist_alert_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                name TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                condition_type TEXT NOT NULL,
                condition_value_json TEXT NOT NULL DEFAULT '{}',
                severity TEXT NOT NULL DEFAULT 'warning',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES subscriptions(id) ON DELETE CASCADE
            );

            -- Per-site scraping configuration (rate limits, headers,
            -- extraction/change-detection rules) used by SiteConfigManager.
            -- Owned here, not by CharactersRAGDB: SiteConfigManager always
            -- points its CharactersRAGDB connection at *this* database's
            -- file (get_subscriptions_db_path()), so this table has always
            -- physically lived in the subscriptions database file even
            -- though it used to be declared -- lazily, at runtime -- by the
            -- wrong class. Same reasoning as local_watchlist_runs and
            -- local_watchlist_alert_rules above: a lazily-created table can
            -- never race an additive migration that needs it to already
            -- exist, and this is the last such table in this package.
            -- SiteConfigManager still reads/writes it through its own
            -- CharactersRAGDB connection (unchanged) -- it now also opens a
            -- SubscriptionsDB against the same path first, purely so this
            -- schema (below) is guaranteed to have run.
            -- site_configs: see SITE_CONFIGS_DDL, applied just below.

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
        conn.executescript(SITE_CONFIGS_DDL)
        self._ensure_watchlists_schema(conn)

    def _ensure_watchlists_schema(self, conn=None):
        """Idempotent migration for watchlists screen schema additions."""
        if conn is None:
            # Same in-memory-safe reasoning as _initialize_schema above: reuse
            # the thread-local connection rather than opening-and-closing a
            # throwaway one, so this is also correct when called standalone
            # (e.g. from a test) against an in-memory instance.
            conn = self.conn

        cursor = conn.cursor()

        # Add columns to subscription_items
        items_cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_items)")}
        if "queued_for_briefing" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN queued_for_briefing BOOLEAN DEFAULT 0")
        if "run_id" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN run_id INTEGER")
        if "alert_matches" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN alert_matches TEXT")

        # Add columns to subscription_filters
        filters_cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_filters)")}
        if "priority" not in filters_cols:
            cursor.execute("ALTER TABLE subscription_filters ADD COLUMN priority INTEGER DEFAULT 0")
        if "is_include_required" not in filters_cols:
            cursor.execute("ALTER TABLE subscription_filters ADD COLUMN is_include_required BOOLEAN DEFAULT 0")

        # Widen CHECK constraint on subscription_filters.action.
        # Must check for the literal action value 'include' rather than the
        # bare substring, because the new column `is_include_required` would
        # otherwise make the substring match and skip the migration.
        existing_check = None
        for row in cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='subscription_filters'"):
            existing_check = row[0]
        if existing_check and "'include'" not in existing_check:
            # This rebuild predates FK enforcement (Task 1a) and may run
            # against a real database that already has a subscription_filters
            # row whose subscription_id no longer exists -- the orphan
            # condition Task 1a exists to stop *creating*, not to clean up
            # retroactively (cleanup is out of scope; already-orphaned rows
            # must survive). Copying such a row into subscription_filters_new,
            # which declares the FK, would raise IntegrityError now that
            # enforcement is on. Disable enforcement for this rebuild only,
            # then restore it -- this is the documented SQLite procedure for
            # a table rebuild that must tolerate pre-existing violations.
            #
            # PRAGMA foreign_keys is a no-op while a transaction is pending,
            # so commit immediately before toggling it off, and again after
            # the rebuild before toggling it back on. Read the pragma back
            # rather than assuming the toggle took effect.
            conn.commit()
            cursor.execute("PRAGMA foreign_keys = OFF;")
            if cursor.execute("PRAGMA foreign_keys").fetchone()[0] != 0:
                raise RuntimeError(
                    "Could not disable foreign_keys enforcement for the "
                    "subscription_filters rebuild; refusing to risk a "
                    "silent partial migration."
                )
            try:
                # CREATE TABLE runs in autocommit in Python's sqlite3 module
                # (only DML gets an implicit transaction), so the
                # conn.rollback() in the except below cannot undo it. If a
                # previous run of this rebuild died after creating this table
                # but before the rename below, `_new` survives indefinitely
                # and every later open hits "table subscription_filters_new
                # already exists" here -- permanently. Drop it first so a
                # stray table from an earlier failed attempt never blocks a
                # fresh one.
                cursor.execute("DROP TABLE IF EXISTS subscription_filters_new")
                cursor.execute("""
                    CREATE TABLE subscription_filters_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subscription_id INTEGER,
                        name TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        conditions TEXT NOT NULL,
                        action TEXT NOT NULL CHECK(action IN ('auto_ingest','auto_ignore','tag','priority','notify','include','exclude','flag')),
                        action_params TEXT,
                        priority INTEGER DEFAULT 0,
                        is_include_required BOOLEAN DEFAULT 0,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
                    )
                """)
                cursor.execute("""
                    INSERT INTO subscription_filters_new
                        (id, subscription_id, name, is_active, conditions, action, action_params, priority, is_include_required, created_at, updated_at)
                    SELECT id, subscription_id, name, is_active, conditions, action, action_params, priority, is_include_required, created_at, updated_at
                    FROM subscription_filters
                """)
                cursor.execute("DROP TABLE subscription_filters")
                cursor.execute("ALTER TABLE subscription_filters_new RENAME TO subscription_filters")
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_subscription_filters_timestamp
                    AFTER UPDATE ON subscription_filters
                    BEGIN
                        UPDATE subscription_filters SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                    END
                """)
                conn.commit()
            except Exception:
                # Roll back any partial rebuild so no pending transaction
                # remains -- otherwise the PRAGMA restore below would be a
                # silent no-op and leave enforcement off for the rest of
                # this connection's life, reintroducing the bug Task 1a
                # fixed.
                conn.rollback()
                # conn.rollback() does not remove subscription_filters_new
                # itself (CREATE TABLE is autocommit -- see the comment
                # above), so this attempt's own partially-built table would
                # otherwise become the exact stray table the DROP above
                # exists to guard against, for the next open.
                cursor.execute("DROP TABLE IF EXISTS subscription_filters_new")
                raise
            finally:
                cursor.execute("PRAGMA foreign_keys = ON;")
                if cursor.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                    logger.error(
                        "Failed to re-enable foreign_keys enforcement after "
                        "the subscription_filters rebuild."
                    )

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscription_items_run_id ON subscription_items(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscription_items_queued ON subscription_items(queued_for_briefing, status)")

        # Reader/body columns. `content` holds the renderable body: article text
        # for feed items, diff text for site changes. `url_snapshots` remains the
        # authority for full-page and previous-snapshot views.
        if "content" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN content TEXT")
        if "content_format" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN content_format TEXT")
        if "content_kind" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN content_kind TEXT")
        # Flag is a separate boolean, not a status: the status CHECK has no
        # 'flagged' value, and an item can be flagged *and* reviewed at once.
        if "is_flagged" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN is_flagged BOOLEAN DEFAULT 0")

        # Snapshot extraction fingerprint + one-time TASK-1362 "noise, not
        # volume" data migration (spec:
        # Docs/superpowers/specs/2026-07-29-watchlists-noise-not-volume-design.md
        # -- see .superpowers/sdd/2026-07-29-watchlists-noise-not-volume).
        # The column's absence IS the one-time gate: a database that already
        # has `extraction_fingerprint` has already had its url-family
        # thresholds/selectors migrated (or was created fresh, after the
        # CREATE TABLE above already declared the column and the 0.0
        # default), so the data migration below can never re-run and clobber
        # a user's subsequent edits.
        #
        # The ALTER and the two UPDATEs are wrapped in one EXPLICIT
        # transaction below -- they do NOT get this for free. Python's
        # sqlite3 module (default isolation_level, no override anywhere in
        # BaseDB/connect_private_sqlite) opens an implicit transaction only
        # before DML (INSERT/UPDATE/DELETE/REPLACE), never before DDL, so a
        # bare `cursor.execute("ALTER TABLE ...")` autocommits immediately --
        # it is not protected by whatever transaction the caller thinks it
        # is in. Verified empirically: without an explicit BEGIN here, an
        # exception between the ALTER and the second UPDATE (e.g. from
        # default_ignore_selectors_text() below) leaves the column present
        # -- the one-time gate durably spent -- with change_threshold moved
        # but ignore_selectors permanently NULL, and *unrepairable*: a clean
        # re-run sees the column already there and skips entirely. SQLite's
        # DDL is itself fully transactional; only the sqlite3 module's
        # implicit-BEGIN policy is not. Wrapping the ALTER and both UPDATEs
        # in one explicit BEGIN IMMEDIATE / COMMIT (rolled back together on
        # any exception) restores atomicity, so the write structurally
        # gates the marker instead of merely being re-runnable until it
        # eventually completes.
        #
        # Which is why this block deliberately does NOT use the shared
        # `transaction()` helper, in knowing exemption from the repo-wide
        # compliance rule that every write goes through
        # `with db.transaction() as cursor:` (CLAUDE.md/AGENTS.md, "Key
        # Patterns -> Database Operations", restated as gotcha 5 "Thread
        # safety"): the helper can only ask the sqlite3 driver for a
        # transaction, and the driver autocommits DDL under its implicit-BEGIN
        # policy regardless, so `transaction()` cannot make ALTER + UPDATE
        # atomic here -- proven by probe during the whole-branch review, and
        # adopting it would reintroduce exactly the unrepairable half-migration
        # described above (a crash between the ALTER and the UPDATEs spends the
        # one-time gate with the data unmigrated and no way back). The explicit
        # BEGIN IMMEDIATE exists precisely for that. The exemption is
        # deliberate, not ignorance of the rule; pinned by
        # `test_migration_rolls_back_atomically_on_mid_migration_failure`.
        snapshot_cols = {row[1] for row in cursor.execute("PRAGMA table_info(url_snapshots)")}
        if "extraction_fingerprint" not in snapshot_cols:
            from ..Subscriptions.noise_defaults import default_ignore_selectors_text

            if conn.in_transaction:
                conn.commit()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                cursor.execute(
                    "ALTER TABLE url_snapshots ADD COLUMN extraction_fingerprint TEXT"
                )
                cursor.execute(
                    "UPDATE subscriptions SET change_threshold = 0.0"
                    " WHERE type IN ('url','url_list','sitemap')"
                )
                cursor.execute(
                    "UPDATE subscriptions SET ignore_selectors = ?"
                    " WHERE type IN ('url','url_list','sitemap')"
                    "   AND (ignore_selectors IS NULL OR TRIM(ignore_selectors) = '')",
                    (default_ignore_selectors_text(),),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Watchlist bundle entity. `name` is intentionally not UNIQUE — uniqueness
        # is enforced case-insensitively in WatchlistBundleService with
        # auto-suffixing, because a SQL constraint would raise mid-migration.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                is_active BOOLEAN DEFAULT 1,
                sort_order INTEGER DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Briefings (spec #2 phase 1): per-watchlist selection mode + optional
        # default preset. Column-presence idiom, same pattern as content_kind
        # above -- additive only, no data migration (nothing to migrate; see
        # Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md).
        wcols = {row[1] for row in cursor.execute("PRAGMA table_info(watchlists)")}
        if "briefing_selection_mode" not in wcols:
            cursor.execute(
                "ALTER TABLE watchlists ADD COLUMN briefing_selection_mode "
                "TEXT DEFAULT 'auto_featured'"
            )
        if "default_briefing_preset_id" not in wcols:
            cursor.execute(
                "ALTER TABLE watchlists ADD COLUMN default_briefing_preset_id INTEGER"
            )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_sources (
                watchlist_id    INTEGER NOT NULL REFERENCES watchlists(id)     ON DELETE CASCADE,
                subscription_id INTEGER NOT NULL REFERENCES subscriptions(id)  ON DELETE CASCADE,
                added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (watchlist_id, subscription_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_watchlist_sources_subscription "
            "ON watchlist_sources(subscription_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_subscription_items_flagged "
            "ON subscription_items(is_flagged, status)"
        )
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_watchlists_timestamp
            AFTER UPDATE ON watchlists
            BEGIN
                UPDATE watchlists SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """)

        # Briefings (spec #2 phase 1): the on-demand text digest for a
        # watchlist, and the items it covered. Additive CREATE TABLE IF NOT
        # EXISTS -- no data migration exists in this design, so the
        # TASK-1362 BEGIN IMMEDIATE machinery above is deliberately not
        # cargo-culted in here (see
        # Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS briefings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watchlist_id INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'generating',
                error TEXT,
                covers_through_item_id INTEGER,
                covers_from_ts DATETIME,
                selection_mode TEXT,
                preset_id INTEGER,
                model_used TEXT,
                body_markdown TEXT,
                item_count INTEGER DEFAULT 0,
                featured_count INTEGER DEFAULT 0,
                overflow_count INTEGER DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS briefing_items (
                briefing_id INTEGER NOT NULL REFERENCES briefings(id) ON DELETE CASCADE,
                item_id     INTEGER NOT NULL REFERENCES subscription_items(id) ON DELETE CASCADE,
                featured BOOLEAN DEFAULT 0,
                PRIMARY KEY (briefing_id, item_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_briefings_watchlist_status "
            "ON briefings(watchlist_id, status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_briefing_items_item "
            "ON briefing_items(item_id)"
        )

        # local_watchlist_runs is guaranteed to exist: BaseDB.__init__ runs
        # _initialize_schema (base_db.py:76), which creates it and then calls
        # this method. Only the column needs checking, for databases created
        # before batch_id existed.
        run_cols = {row[1] for row in cursor.execute("PRAGMA table_info(local_watchlist_runs)")}
        if "batch_id" not in run_cols:
            cursor.execute("ALTER TABLE local_watchlist_runs ADD COLUMN batch_id TEXT")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_local_watchlist_runs_batch "
            "ON local_watchlist_runs(batch_id)"
        )

        # External-content FTS over items, matching the pattern used by
        # character_cards_fts / conversations_fts / media_fts. Triggers rather
        # than explicit index writes, so every INSERT path stays indexed.
        #
        # Do NOT add `columnsize=0` to the fts5 options below: that option
        # removes the `_docsize` shadow table, which both the guarded delete
        # legs below and `backfill_items_fts()` depend on to answer "has this
        # rowid actually been written into the FTS index yet".
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS subscription_items_fts USING fts5(
                title,
                content,
                author,
                content='subscription_items',
                content_rowid='id'
            )
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS subscription_items_fts_ai
            AFTER INSERT ON subscription_items BEGIN
                INSERT INTO subscription_items_fts(rowid, title, content, author)
                VALUES (new.id, new.title, new.content, new.author);
            END
        """)
        # subscription_items_fts is an external-content FTS5 table: its
        # 'delete' command is only legal for a rowid that is actually present
        # in the index (has a row in the `_docsize` shadow table). This
        # migration creates the index over a `subscription_items` table that
        # may already hold rows from before this branch existed, so without a
        # guard the very first UPDATE/DELETE of a pre-existing, never-indexed
        # item would fire 'delete' against a rowid FTS5 has never seen --
        # FTS5 rejects the whole statement, and it surfaces to the caller as
        # "database disk image is malformed" even though nothing is actually
        # corrupt. Guard the delete legs on index membership so an unindexed
        # row is skipped instead of fatal. `_au`'s insert leg stays
        # unconditional, so a skipped/unindexed row simply becomes indexed on
        # its next update.
        #
        # These two triggers are dropped and recreated unconditionally
        # (rather than `CREATE ... IF NOT EXISTS`) so that a database that
        # already ran the previous, unguarded versions of these triggers on
        # this branch picks up the fix too -- `IF NOT EXISTS` would otherwise
        # silently keep the old, unguarded trigger bodies in place. This is
        # idempotent: re-running it just drops-and-recreates the same guarded
        # triggers again.
        #
        # Do not split `_au` into two separate triggers (one delete, one
        # insert): SQLite does not guarantee firing order between multiple
        # triggers on the same event, and an insert-then-guarded-delete
        # ordering would corrupt the index.
        cursor.execute("DROP TRIGGER IF EXISTS subscription_items_fts_ad")
        cursor.execute("""
            CREATE TRIGGER subscription_items_fts_ad
            AFTER DELETE ON subscription_items BEGIN
                INSERT INTO subscription_items_fts(subscription_items_fts, rowid, title, content, author)
                SELECT 'delete', old.id, old.title, old.content, old.author
                WHERE EXISTS (SELECT 1 FROM subscription_items_fts_docsize WHERE id = old.id);
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS subscription_items_fts_au")
        cursor.execute("""
            CREATE TRIGGER subscription_items_fts_au
            AFTER UPDATE ON subscription_items BEGIN
                INSERT INTO subscription_items_fts(subscription_items_fts, rowid, title, content, author)
                SELECT 'delete', old.id, old.title, old.content, old.author
                WHERE EXISTS (SELECT 1 FROM subscription_items_fts_docsize WHERE id = old.id);
                INSERT INTO subscription_items_fts(rowid, title, content, author)
                VALUES (new.id, new.title, new.content, new.author);
            END
        """)

        conn.commit()

    def backfill_items_fts(self, chunk_size: int = 500) -> int:
        """Index one chunk of items that are missing from the FTS table.

        Runs in a background worker, never inline in migration — a synchronous
        backfill of a large subscription_items table would block app boot.
        Resumes by rowid, so an interrupted backfill continues rather than
        restarting.

        The "not yet indexed" check reads ``subscription_items_fts_docsize``
        rather than ``subscription_items_fts`` itself. For an external-content
        FTS5 table (``content='subscription_items'``), an unfiltered/no-MATCH
        query against the fts5 table is satisfied straight from the external
        content table's rowids -- it does not reflect the actual state of the
        FTS index. ``%_docsize`` is the shadow table SQLite documents for this
        purpose: it is populated only by real writes into the fts5 table (the
        insert/update triggers, or this method), so it is the only place that
        truthfully answers "has this rowid been indexed yet".

        Args:
            chunk_size: Maximum rows to index in this call. Must be >= 1.

        Returns:
            Number of rows indexed. ``0`` means the backfill is complete.

        Raises:
            ValueError: If ``chunk_size`` is less than 1. A non-positive
                value would make ``LIMIT ?`` return zero rows regardless of
                how large the real backlog is, and this method would then
                report ``0`` ("complete") while unindexed rows remain --
                silently stranding them once this becomes the repair path
                for legacy rows the guarded FTS triggers skip.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size!r}")
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, title, content, author
                FROM subscription_items
                WHERE id NOT IN (SELECT rowid FROM subscription_items_fts_docsize)
                ORDER BY id
                LIMIT ?
                """,
                (chunk_size,),
            ).fetchall()
            if not rows:
                return 0
            conn.executemany(
                "INSERT INTO subscription_items_fts(rowid, title, content, author) "
                "VALUES (?, ?, ?, ?)",
                [(row[0], row[1], row[2], row[3]) for row in rows],
            )
            return len(rows)

    # Sentinel bucket ids for the watchlists tree roots.
    UNASSIGNED_BUCKET = -1
    ALL_SOURCES_BUCKET = -2

    def get_watchlist_item_counts(self) -> Dict[int, Dict[str, int]]:
        """Item totals and unread counts for every watchlists tree node.

        Returned in a single query so that adding watchlists never adds
        round-trips. ``SUM(CASE …)`` is used rather than ``COUNT(*) FILTER``
        to avoid depending on a newer SQLite than the bundled one.

        The per-watchlist leg is anchored on ``watchlists`` with LEFT JOINs
        (not an INNER JOIN from ``watchlist_sources``), so a watchlist with
        no sources yet -- or sources with no items yet -- still appears with
        ``{"total": 0, "unread": 0}`` instead of being missing from the
        result entirely. With a LEFT JOIN, ``COUNT(si.id)`` would still be
        correct (``COUNT`` ignores NULLs), but ``COUNT(*)`` would wrongly
        count the null-padded row for a sourceless watchlist -- the
        ``SUM(CASE WHEN si.id IS NOT NULL ...)`` form is used to make that
        unambiguous rather than relying on a NULL-counting subtlety.

        Returns:
            Mapping of bucket id to ``{"total": int, "unread": int}``. Bucket
            ``-1`` is Unassigned (sources in no watchlist) and ``-2`` is All
            sources. Real watchlist ids are positive.
        """
        rows = self.conn.execute(
            """
            SELECT w.id AS bucket,
                   SUM(CASE WHEN si.id IS NOT NULL THEN 1 ELSE 0 END) AS total,
                   SUM(CASE WHEN si.status = 'new' THEN 1 ELSE 0 END) AS unread
            FROM watchlists w
            LEFT JOIN watchlist_sources ws  ON ws.watchlist_id = w.id
            LEFT JOIN subscription_items si ON si.subscription_id = ws.subscription_id
            GROUP BY w.id

            UNION ALL

            SELECT ?, COUNT(si.id),
                   SUM(CASE WHEN si.status = 'new' THEN 1 ELSE 0 END)
            FROM subscription_items si
            WHERE NOT EXISTS (
                SELECT 1 FROM watchlist_sources ws
                WHERE ws.subscription_id = si.subscription_id
            )

            UNION ALL

            SELECT ?, COUNT(si.id),
                   SUM(CASE WHEN si.status = 'new' THEN 1 ELSE 0 END)
            FROM subscription_items si
            """,
            (self.UNASSIGNED_BUCKET, self.ALL_SOURCES_BUCKET),
        ).fetchall()

        # No `if row[0] is not None` filter here: watchlists.id and
        # watchlist_sources.watchlist_id are NOT NULL, and both sentinels
        # bind non-null literals, so every row's bucket id is always non-null.
        return {
            row[0]: {"total": row[1] or 0, "unread": row[2] or 0}
            for row in rows
        }

    @property
    def conn(self):
        """Thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
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

    def add_subscription(
        self,
        name: str,
        type: str,
        source: str,
        tags: Optional[List[str]] = None,
        priority: int = 3,
        folder: Optional[str] = None,
        auth_config: Optional[Dict] = None,
        **kwargs,
    ) -> int:
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
                "name": name,
                "type": type,
                "source": source,
                "tags": ",".join(tags) if tags else None,
                "priority": priority,
                "folder": folder,
                "auth_config": json.dumps(auth_config) if auth_config else None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Add optional fields from kwargs
            allowed_fields = [
                "description",
                "check_frequency",
                "extraction_method",
                "extraction_rules",
                "processing_options",
                "auto_ingest",
                "notification_config",
                "change_threshold",
                "ignore_selectors",
                "custom_headers",
                "rate_limit_config",
                "auto_pause_threshold",
                "is_active",
            ]

            for field in allowed_fields:
                if field in kwargs:
                    value = kwargs[field]
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    fields[field] = value

            # Build insert query
            columns = ", ".join(fields.keys())
            placeholders = ", ".join(["?" for _ in fields])

            cursor.execute(
                f"""
                INSERT INTO subscriptions ({columns})
                VALUES ({placeholders})
            """,
                list(fields.values()),
            )

            subscription_id = cursor.lastrowid
            logger.info(
                f"Added subscription '{name}' (ID: {subscription_id}, Type: {type})"
            )

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "subscriptions_db_operation_duration",
                duration,
                labels={
                    "operation": "add_subscription",
                    "type": type,
                    "priority": str(priority),
                },
            )
            log_counter(
                "subscriptions_db_operation_count",
                labels={
                    "operation": "add_subscription",
                    "type": type,
                    "status": "success",
                    "has_auth": "true" if auth_config else "false",
                    "has_tags": "true" if tags else "false",
                },
            )

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
        log_histogram(
            "subscriptions_db_operation_duration",
            duration,
            labels={
                "operation": "get_subscription",
                "found": "true" if result else "false",
            },
        )
        log_counter(
            "subscriptions_db_operation_count",
            labels={
                "operation": "get_subscription",
                "status": "success",
                "found": "true" if result else "false",
            },
        )

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
                "name",
                "type",
                "source",
                "description",
                "tags",
                "priority",
                "folder",
                "check_frequency",
                "is_active",
                "is_paused",
                "auth_config",
                "custom_headers",
                "rate_limit_config",
                "extraction_method",
                "extraction_rules",
                "processing_options",
                "auto_ingest",
                "notification_config",
                "change_threshold",
                "ignore_selectors",
                "etag",
                "last_modified",
                "auto_pause_threshold",
            ]

            updates = []
            values = []

            for field, value in kwargs.items():
                if field in allowed_fields:
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    elif field == "tags" and isinstance(value, list):
                        value = ",".join(value)
                    updates.append(f"{field} = ?")
                    values.append(value)

            if not updates:
                return False

            values.append(subscription_id)
            cursor.execute(
                f"""
                UPDATE subscriptions 
                SET {", ".join(updates)}
                WHERE id = ?
            """,
                values,
            )

            success = cursor.rowcount > 0

            # Log metrics
            duration = time.time() - start_time
            log_histogram(
                "subscriptions_db_operation_duration",
                duration,
                labels={
                    "operation": "update_subscription",
                    "fields_updated": str(len(updates)),
                    "success": str(success),
                },
            )
            log_counter(
                "subscriptions_db_operation_count",
                labels={
                    "operation": "update_subscription",
                    "status": "success" if success else "not_found",
                    "fields_updated": str(len(updates)),
                },
            )

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
            log_histogram(
                "subscriptions_db_operation_duration",
                duration,
                labels={"operation": "delete_subscription", "success": str(success)},
            )
            log_counter(
                "subscriptions_db_operation_count",
                labels={
                    "operation": "delete_subscription",
                    "status": "success" if success else "not_found",
                },
            )

            return success

    def get_pending_checks(
        self, limit: int = 10, priority_order: bool = True
    ) -> List[Dict[str, Any]]:
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

        order_clause = (
            "ORDER BY priority DESC, last_checked ASC"
            if priority_order
            else "ORDER BY last_checked ASC"
        )

        cursor.execute(
            f"""
            SELECT * FROM subscriptions
            WHERE is_active = 1 
            AND is_paused = 0
            AND (
                last_checked IS NULL 
                OR datetime(last_checked, '+' || check_frequency || ' seconds') <= datetime('now')
            )
            {order_clause}
            LIMIT ?
        """,
            (limit,),
        )

        results = [dict(row) for row in cursor.fetchall()]

        # Log metrics
        duration = time.time() - start_time
        log_histogram(
            "subscriptions_db_operation_duration",
            duration,
            labels={
                "operation": "get_pending_checks",
                "limit": str(limit),
                "result_count": str(len(results)),
            },
        )
        log_counter(
            "subscriptions_db_operation_count",
            labels={
                "operation": "get_pending_checks",
                "status": "success",
                "result_count": str(len(results)),
                "priority_order": str(priority_order),
            },
        )

        return results

    def get_subscriptions_by_tag(
        self, tag: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Filter subscriptions by tag.

        Args:
            tag: The tag to filter by
            limit: Maximum number of subscriptions to return
            offset: Number of subscriptions to skip

        Returns:
            List of subscription dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM subscriptions
            WHERE is_active = 1 AND tags LIKE ?
            ORDER BY name
            LIMIT ? OFFSET ?
        """,
            (f"%{tag}%", limit, offset),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_subscriptions_by_folder(
        self, folder: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all subscriptions in a folder.

        Args:
            folder: The folder name to filter by
            limit: Maximum number of subscriptions to return
            offset: Number of subscriptions to skip

        Returns:
            List of subscription dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM subscriptions
            WHERE is_active = 1 AND folder = ?
            ORDER BY priority DESC, name
            LIMIT ? OFFSET ?
        """,
            (folder, limit, offset),
        )

        return [dict(row) for row in cursor.fetchall()]

    # --- Check Results and Error Handling ---

    def record_check_result(
        self,
        subscription_id: int,
        items: List[Dict] = None,
        error: Optional[str] = None,
        stats: Optional[Dict] = None,
    ) -> None:
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
                cursor.execute(
                    """
                    UPDATE subscriptions
                    SET last_checked = ?,
                        last_error = ?,
                        error_count = error_count + 1,
                        consecutive_failures = consecutive_failures + 1
                    WHERE id = ?
                """,
                    (now, error, subscription_id),
                )

                # Check if we should auto-pause
                cursor.execute(
                    """
                    SELECT consecutive_failures, auto_pause_threshold
                    FROM subscriptions WHERE id = ?
                """,
                    (subscription_id,),
                )

                row = cursor.fetchone()
                if row and row["consecutive_failures"] >= row["auto_pause_threshold"]:
                    cursor.execute(
                        """
                        UPDATE subscriptions
                        SET is_paused = 1
                        WHERE id = ?
                    """,
                        (subscription_id,),
                    )
                    logger.warning(
                        f"Auto-paused subscription {subscription_id} after {row['consecutive_failures']} failures"
                    )

            else:
                # Successful check
                cursor.execute(
                    """
                    UPDATE subscriptions
                    SET last_checked = ?,
                        last_successful_check = ?,
                        last_error = NULL,
                        error_count = 0,
                        consecutive_failures = 0
                    WHERE id = ?
                """,
                    (now, now, subscription_id),
                )

                # Add new items if provided
                if items:
                    for item in items:
                        self._add_subscription_item(cursor, subscription_id, item)

            # Update statistics if provided
            if stats:
                self._update_subscription_stats(
                    subscription_id, stats, error is not None
                )

            # Log metrics
            duration = time.time() - start_time
            log_histogram(
                "subscriptions_db_operation_duration",
                duration,
                labels={
                    "operation": "record_check_result",
                    "has_error": "true" if error else "false",
                    "has_items": "true" if items else "false",
                },
            )
            log_counter(
                "subscriptions_db_operation_count",
                labels={
                    "operation": "record_check_result",
                    "status": "error" if error else "success",
                    "item_count": str(len(items)) if items else "0",
                    "auto_paused": "true"
                    if error and "Auto-paused" in str(error)
                    else "false",
                },
            )

    def record_check_error(
        self, subscription_id: int, error: str, should_pause: bool = False
    ) -> None:
        """Record an error with optional auto-pause."""
        with self.transaction() as conn:
            cursor = conn.cursor()

            now = datetime.now(timezone.utc).isoformat()

            cursor.execute(
                """
                UPDATE subscriptions
                SET last_checked = ?,
                    last_error = ?,
                    error_count = error_count + 1,
                    consecutive_failures = consecutive_failures + 1,
                    is_paused = ?
                WHERE id = ?
            """,
                (now, error, 1 if should_pause else 0, subscription_id),
            )

    def reset_subscription_errors(self, subscription_id: int) -> None:
        """Reset error count after successful check."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE subscriptions
                SET error_count = 0,
                    consecutive_failures = 0,
                    last_error = NULL,
                    is_paused = 0
                WHERE id = ?
            """,
                (subscription_id,),
            )

    # --- Item Management ---

    def get_new_items(
        self,
        subscription_id: Optional[int] = None,
        status: str = "new",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get items with filtering and pagination."""
        cursor = self.conn.cursor()

        if subscription_id:
            cursor.execute(
                """
                SELECT i.*, s.name as subscription_name, s.type as subscription_type
                FROM subscription_items i
                JOIN subscriptions s ON i.subscription_id = s.id
                WHERE i.subscription_id = ? AND i.status = ?
                ORDER BY i.created_at DESC
                LIMIT ?
            """,
                (subscription_id, status, limit),
            )
        else:
            cursor.execute(
                """
                SELECT i.*, s.name as subscription_name, s.type as subscription_type
                FROM subscription_items i
                JOIN subscriptions s ON i.subscription_id = s.id
                WHERE i.status = ?
                ORDER BY i.created_at DESC
                LIMIT ?
            """,
                (status, limit),
            )

        return [dict(row) for row in cursor.fetchall()]

    def get_item_status(self, item_id: int) -> str:
        """Read one item's current status by its own row id.

        The counterpart to `mark_item_status`: a single-row read, so it is
        authoritative regardless of how many items share a status. A caller
        that instead scans `get_new_items` per candidate status is answering
        "is this item `ingested`?" from a page of at most `limit` rows, and
        an item beyond that page is indistinguishable from an item that is
        not there at all -- see
        `WatchlistsCollectionsScreen._blocking_status_for`, which used to do
        exactly that.

        Args:
            item_id: The `subscription_items` row id.

        Returns:
            The row's status, defaulting to ``"new"`` when the column is
            NULL -- matching `normalize_watchlist_item`.

        Raises:
            KeyError: If no item has that id. Callers deciding whether a
                destructive write is safe must treat a missing row as an
                unanswered question rather than as permission.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM subscription_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Subscription item not found: {item_id}")
        return str(row["status"] or "new")

    def mark_item_status(
        self,
        item_id: int,
        status: str,
        media_id: Optional[int] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update item status with error tracking."""
        with self.transaction() as conn:
            cursor = conn.cursor()

            updates = ["status = ?"]
            values = [status]

            if media_id is not None:
                updates.append("media_id = ?")
                values.append(media_id)

            if error is not None:
                updates.append("processing_error = ?")
                values.append(error)

            values.append(item_id)

            cursor.execute(
                f"""
                UPDATE subscription_items
                SET {", ".join(updates)}
                WHERE id = ?
            """,
                values,
            )

            return cursor.rowcount > 0

    # --- Briefings (spec #2 phase 1) ---

    def set_item_briefing_queued(self, item_id: int, queued: bool) -> None:
        """Set or clear the global "queued for briefing" flag on one item.

        The flag is global (ADR-018): the same source item can sit in
        several watchlists, and generation never auto-clears it -- see
        Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md,
        "The queue flag is global, and never auto-cleared". Only the user
        (or this explicit call) empties it.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE subscription_items SET queued_for_briefing = ? WHERE id = ?",
                (1 if queued else 0, item_id),
            )

    def insert_briefing(self, watchlist_id: int, status: str = "generating") -> int:
        """Create a new `briefings` row for a watchlist and return its id."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO briefings (watchlist_id, status) VALUES (?, ?)",
                (watchlist_id, status),
            )
            return cursor.lastrowid

    def update_briefing(self, briefing_id: int, **fields: Any) -> None:
        """Update arbitrary columns on a `briefings` row by id.

        `fields` keys are always passed by trusted callers as keyword
        arguments naming real `briefings` columns (e.g.
        `status="complete", covers_through_item_id=40`) -- never sourced
        from unvalidated external input -- so building the SET clause from
        the keys is not a SQL-injection surface here.
        """
        if not fields:
            return
        set_clause = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values())
        values.append(briefing_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE briefings SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                values,
            )

    def get_briefing(self, briefing_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one `briefings` row by id, or None if it doesn't exist."""
        row = self.conn.execute(
            "SELECT * FROM briefings WHERE id = ?", (briefing_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def list_briefings(self, watchlist_id: int) -> List[Dict[str, Any]]:
        """List a watchlist's briefings, newest first."""
        cursor = self.conn.execute(
            "SELECT * FROM briefings WHERE watchlist_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (watchlist_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def latest_completed_watermark(self, watchlist_id: int) -> Optional[int]:
        """Max `covers_through_item_id` over this watchlist's complete/empty briefings.

        THE coverage invariant's DB half: a `failed` briefing must never
        advance the window (failure never loses items -- the next attempt
        re-covers the same window), so 'failed' is deliberately excluded
        from this status set. Do not add it here.
        """
        row = self.conn.execute(
            "SELECT MAX(covers_through_item_id) FROM briefings "
            "WHERE watchlist_id = ? AND status IN ('complete', 'empty')",
            (watchlist_id,),
        ).fetchone()
        return row[0] if row is not None else None

    def find_duplicate_items(
        self, item_url: str, item_hash: str
    ) -> List[Dict[str, Any]]:
        """Check for existing duplicates."""
        cursor = self.conn.cursor()

        # Canonicalize URL for comparison
        canonical_url = self._canonicalize_url(item_url)

        cursor.execute(
            """
            SELECT * FROM subscription_items
            WHERE (canonical_url = ? OR content_hash = ?)
            AND status != 'ignored'
            ORDER BY created_at DESC
        """,
            (canonical_url, item_hash),
        )

        return [dict(row) for row in cursor.fetchall()]

    def bulk_update_items(self, item_ids: List[int], status: str) -> int:
        """Efficient bulk status updates."""
        if not item_ids:
            return 0

        with self.transaction() as conn:
            cursor = conn.cursor()

            placeholders = ",".join(["?" for _ in item_ids])
            cursor.execute(
                f"""
                UPDATE subscription_items
                SET status = ?
                WHERE id IN ({placeholders})
            """,
                [status] + item_ids,
            )

            return cursor.rowcount

    # --- Statistics and Health Monitoring ---

    def update_subscription_stats(
        self, subscription_id: int, date: str, stats: Dict[str, Any]
    ) -> None:
        """Record daily statistics."""
        with self.transaction() as conn:
            cursor = conn.cursor()

            # Insert or update stats for the day
            cursor.execute(
                """
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
            """,
                (
                    subscription_id,
                    date,
                    stats.get("checks_performed", 1),
                    stats.get("successful_checks", 0),
                    stats.get("new_items_found", 0),
                    stats.get("items_ingested", 0),
                    stats.get("errors_encountered", 0),
                    stats.get("avg_response_time_ms", 0),
                    stats.get("total_bytes_transferred", 0),
                ),
            )

    def get_subscription_health(
        self, subscription_id: int, days: int = 30
    ) -> Dict[str, Any]:
        """Get health metrics for dashboard."""
        start_time = time.time()
        cursor = self.conn.cursor()

        # Get recent stats
        cursor.execute(
            """
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
        """,
            (subscription_id, days),
        )

        stats = dict(cursor.fetchone() or {})

        # Calculate health score (0-100)
        if stats.get("total_checks", 0) > 0:
            success_rate = stats.get("successful_checks", 0) / stats["total_checks"]
            stats["health_score"] = int(success_rate * 100)
        else:
            stats["health_score"] = 0

        # Get current subscription status
        cursor.execute(
            """
            SELECT consecutive_failures, last_error, is_paused
            FROM subscriptions WHERE id = ?
        """,
            (subscription_id,),
        )

        current = cursor.fetchone()
        if current:
            stats.update(dict(current))

        # Log metrics
        duration = time.time() - start_time
        log_histogram(
            "subscriptions_db_operation_duration",
            duration,
            labels={"operation": "get_subscription_health", "days": str(days)},
        )
        log_counter(
            "subscriptions_db_operation_count",
            labels={
                "operation": "get_subscription_health",
                "status": "success",
                "health_score": str(stats.get("health_score", 0)),
            },
        )

        return stats

    def get_failing_subscriptions(self, threshold: int = 5) -> List[Dict[str, Any]]:
        """Find subscriptions needing attention."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT * FROM subscriptions
            WHERE consecutive_failures >= ?
            OR (error_count > 0 AND last_successful_check < datetime('now', '-7 days'))
            ORDER BY consecutive_failures DESC, error_count DESC
        """,
            (threshold,),
        )

        return [dict(row) for row in cursor.fetchall()]

    # --- Filters and Templates ---

    def add_filter(
        self,
        name: str,
        conditions: Dict[str, Any],
        action: str,
        subscription_id: Optional[int] = None,
        action_params: Optional[Dict] = None,
    ) -> int:
        """Add smart filter rule."""
        with self.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO subscription_filters 
                (subscription_id, name, conditions, action, action_params)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    subscription_id,
                    name,
                    json.dumps(conditions),
                    action,
                    json.dumps(action_params) if action_params else None,
                ),
            )

            return cursor.lastrowid

    def get_active_filters(
        self, subscription_id: Optional[int] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get filters for processing.

        Args:
            subscription_id: Optional subscription ID to filter by
            limit: Maximum number of filters to return
            offset: Number of filters to skip

        Returns:
            List of filter dictionaries
        """
        cursor = self.conn.cursor()

        if subscription_id is not None:
            cursor.execute(
                """
                SELECT * FROM subscription_filters
                WHERE is_active = 1 AND (subscription_id = ? OR subscription_id IS NULL)
                ORDER BY subscription_id DESC
                LIMIT ? OFFSET ?
            """,
                (subscription_id, limit, offset),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM subscription_filters
                WHERE is_active = 1 AND subscription_id IS NULL
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            )

        filters = []
        for row in cursor.fetchall():
            filter_dict = dict(row)
            filter_dict["conditions"] = json.loads(filter_dict["conditions"])
            if filter_dict["action_params"]:
                filter_dict["action_params"] = json.loads(filter_dict["action_params"])
            filters.append(filter_dict)

        return filters

    def save_template(
        self, name: str, config: Dict[str, Any], category: Optional[str] = None
    ) -> int:
        """Save subscription template."""
        with self.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO subscription_templates
                (name, description, category, type, check_frequency,
                 extraction_method, extraction_rules, processing_options, auth_config_template)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    config.get("description"),
                    category,
                    config["type"],
                    config.get("check_frequency"),
                    config.get("extraction_method"),
                    json.dumps(config.get("extraction_rules"))
                    if config.get("extraction_rules")
                    else None,
                    json.dumps(config.get("processing_options"))
                    if config.get("processing_options")
                    else None,
                    json.dumps(config.get("auth_config_template"))
                    if config.get("auth_config_template")
                    else None,
                ),
            )

            return cursor.lastrowid

    def get_templates(
        self, category: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve available templates.

        Args:
            category: Optional category to filter by
            limit: Maximum number of templates to return
            offset: Number of templates to skip

        Returns:
            List of template dictionaries
        """
        cursor = self.conn.cursor()

        if category:
            cursor.execute(
                """
                SELECT * FROM subscription_templates
                WHERE category = ?
                ORDER BY usage_count DESC, name
                LIMIT ? OFFSET ?
            """,
                (category, limit, offset),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM subscription_templates
                ORDER BY usage_count DESC, name
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            )

        templates = []
        for row in cursor.fetchall():
            template = dict(row)
            # Parse JSON fields
            for field in [
                "extraction_rules",
                "processing_options",
                "auth_config_template",
            ]:
                if template.get(field):
                    template[field] = json.loads(template[field])
            templates.append(template)

        return templates

    # --- Helper Methods ---

    def _add_subscription_item(
        self, cursor, subscription_id: int, item: Dict[str, Any]
    ) -> int:
        """Add a new subscription item."""
        # Canonicalize URL for deduplication
        canonical_url = self._canonicalize_url(item["url"])

        # Check for duplicates
        cursor.execute(
            """
            SELECT id FROM subscription_items
            WHERE subscription_id = ? AND canonical_url = ? AND content_hash = ?
        """,
            (subscription_id, canonical_url, item.get("content_hash")),
        )

        existing = cursor.fetchone()
        if existing:
            return existing["id"]

        # Insert new item via the shared persistence path so the full
        # column set (content, content_kind, content_format, run_id,
        # alert_matches, ...) is written, not just the change/dedup fields
        # this path used to carry alone. The canonical-URL dedupe guard
        # above is kept unchanged; persist_subscription_item's own
        # ON CONFLICT target (subscription_id, url, content_hash) is a
        # narrower, independent dedupe rule that still applies underneath
        # it — the two dedupe rules are deliberately not unified.
        #
        # Imported locally: Subscriptions/__init__.py imports
        # LocalWatchlistsService, which imports this module, so a
        # module-level import here would be circular.
        from ..Subscriptions.item_persist import persist_subscription_item

        now = datetime.now(timezone.utc).isoformat()
        return persist_subscription_item(
            cursor.connection,
            subscription_id,
            {**item, "canonical_url": canonical_url},
            run_id=None,
            now=now,
        )

    def _update_subscription_stats(
        self, subscription_id: int, stats: Dict[str, Any], had_error: bool
    ) -> None:
        """Update subscription statistics."""
        today = datetime.now(timezone.utc).date().isoformat()

        self.update_subscription_stats(
            subscription_id,
            today,
            {
                "checks_performed": 1,
                "successful_checks": 0 if had_error else 1,
                "errors_encountered": 1 if had_error else 0,
                "avg_response_time_ms": stats.get("response_time_ms", 0),
                "total_bytes_transferred": stats.get("bytes_transferred", 0),
                "new_items_found": stats.get("new_items_found", 0),
                "items_ingested": stats.get("items_ingested", 0),
            },
        )

    def _canonicalize_url(self, url: str) -> str:
        """Canonicalize URL for deduplication."""
        try:
            parsed = urlparse(url.lower())
            # Remove common tracking parameters
            # In a real implementation, this would be more sophisticated
            canonical = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path.rstrip("/"),
                    "",  # params
                    "",  # query (removed for now, could clean selectively)
                    "",  # fragment
                )
            )
            return canonical
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse URL '{url}' for canonicalization: {e}")
            return url.lower()

    def get_all_subscriptions(
        self, include_inactive: bool = False, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all subscriptions with optional filtering.

        Args:
            include_inactive: Whether to include inactive subscriptions
            limit: Maximum number of subscriptions to return
            offset: Number of subscriptions to skip

        Returns:
            List of subscription dictionaries
        """
        start_time = time.time()

        cursor = self.conn.cursor()

        if include_inactive:
            cursor.execute(
                "SELECT * FROM subscriptions ORDER BY name LIMIT ? OFFSET ?",
                (limit, offset),
            )
        else:
            cursor.execute(
                "SELECT * FROM subscriptions WHERE is_active = 1 ORDER BY name LIMIT ? OFFSET ?",
                (limit, offset),
            )

        results = [dict(row) for row in cursor.fetchall()]

        # Log metrics
        duration = time.time() - start_time
        log_histogram(
            "subscriptions_db_operation_duration",
            duration,
            labels={
                "operation": "get_all_subscriptions",
                "include_inactive": str(include_inactive),
                "result_count": str(len(results)),
            },
        )
        log_counter(
            "subscriptions_db_operation_count",
            labels={
                "operation": "get_all_subscriptions",
                "status": "success",
                "result_count": str(len(results)),
            },
        )

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

        results = {row["type"]: row["count"] for row in cursor.fetchall()}

        # Log metrics
        duration = time.time() - start_time
        total_count = sum(results.values())
        log_histogram(
            "subscriptions_db_operation_duration",
            duration,
            labels={
                "operation": "get_subscription_count",
                "active_only": str(active_only),
            },
        )
        log_counter(
            "subscriptions_db_operation_count",
            labels={
                "operation": "get_subscription_count",
                "status": "success",
                "total_count": str(total_count),
                "type_count": str(len(results)),
            },
        )

        return results

    def close(self):
        """Close database connections."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# End of Subscriptions_DB.py
