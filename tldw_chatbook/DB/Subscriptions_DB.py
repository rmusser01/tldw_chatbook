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
from typing import List, Dict, Any, Optional, Sequence, Union
from urllib.parse import urlparse, urlunparse

# Third-Party Libraries
from loguru import logger

# Local Imports
from .private_sqlite import connect_private_sqlite
from .base_db import BaseDB
from .sql_validation import validate_identifier
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


#: Sentinel default for `SubscriptionsDB.set_watchlist_briefing_settings`'s
#: `default_preset_id` parameter, distinguishing "leave this column alone"
#: (the default, when the caller doesn't pass the argument at all) from
#: `None` (a real value meaning "clear the default preset"). A plain `None`
#: default could not carry both meanings.
_UNSET = object()

#: Max ids bound in one `get_subscription_items_by_ids` statement's
#: `IN (...)` clause. Chunked rather than bound in a single unbounded
#: statement for the same reason `briefing_selection._window_rows` avoids a
#: per-id `NOT IN` (see its docstring): SQLite has a host-parameter limit,
#: and a heavy user's briefing can reference far more items than is safe to
#: bind in one query.
_ITEM_ID_LOOKUP_CHUNK_SIZE = 500


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
        # Briefings phase 4: per-watchlist scheduled-generation cadence.
        # NULL means never -- scheduled briefings are opt-in per watchlist
        # (Locked Decision 4, Docs/superpowers/plans/2026-08-01-watchlists-
        # briefings-phase-4.md), so a watchlist with no explicit cadence
        # must never surface from `list_briefing_schedules`. Same additive
        # column-presence idiom as the two columns above.
        if "briefing_cadence_seconds" not in wcols:
            cursor.execute(
                "ALTER TABLE watchlists ADD COLUMN briefing_cadence_seconds INTEGER"
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

        # Briefing presets + scripts (spec #2 phase 2a): a preset is a named,
        # reusable N-speaker roster (plus optional style notes and a
        # provider/model override); a script is one cast run of a specific
        # `briefings` row. Scripts snapshot the roster and preset name at
        # cast time (`roster_snapshot_json`, `preset_name`) rather than
        # joining back to `briefing_presets` live -- editing or deleting a
        # preset later must never change the meaning of a script someone
        # already cast, so `preset_id` here is deliberately NOT a foreign
        # key: it is a best-effort back-reference only, and outliving its
        # target is expected, not an error. `briefing_id`, in contrast, IS a
        # real FK with `ON DELETE CASCADE` -- a script has no meaning once
        # the briefing it narrates is gone. Additive `CREATE TABLE IF NOT
        # EXISTS`, no data migration, so the `BEGIN IMMEDIATE` machinery used
        # for other rebuilds in this file is deliberately not cargo-culted in
        # here (see Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS briefing_presets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                style_notes TEXT,
                provider TEXT,
                model TEXT,
                roster_json TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS briefing_scripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                briefing_id INTEGER NOT NULL REFERENCES briefings(id) ON DELETE CASCADE,
                preset_id INTEGER,
                preset_name TEXT NOT NULL,
                roster_snapshot_json TEXT NOT NULL,
                turns_json TEXT,
                status TEXT NOT NULL DEFAULT 'generating',
                error TEXT,
                model_used TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_briefing_scripts_briefing "
            "ON briefing_scripts(briefing_id, status)"
        )

        # Briefing audio (spec #2 phase 2b): one row per audio-synthesis run
        # of a specific `briefing_scripts` row, turning that cast script into
        # a playable recording. `voice_snapshot_json` freezes the voice
        # assignment used for this render -- exactly like `roster_snapshot_
        # json` above, it is deliberately NOT in `update_briefing_audio`'s
        # allowlist (see that method's docstring): a synthesized artifact's
        # provenance must never be revisable after the fact. `script_id` IS a
        # real FK with `ON DELETE CASCADE` -- audio has no meaning once the
        # script it narrates is gone. Additive `CREATE TABLE IF NOT EXISTS`,
        # no data migration, so (matching `briefing_scripts` above) the
        # `BEGIN IMMEDIATE` rebuild machinery used elsewhere in this file is
        # deliberately not cargo-culted in here.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS briefing_audio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                script_id INTEGER NOT NULL REFERENCES briefing_scripts(id) ON DELETE CASCADE,
                voice_snapshot_json TEXT NOT NULL,
                file_path TEXT,
                duration_seconds REAL,
                turn_count INTEGER,
                status TEXT NOT NULL DEFAULT 'generating',
                error TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_briefing_audio_script "
            "ON briefing_audio(script_id, status)"
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

        Args:
            item_id: `subscription_items.id` to update.
            queued: `True` to mark the item queued for the next briefing,
                `False` to clear the flag.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE subscription_items SET queued_for_briefing = ? WHERE id = ?",
                (1 if queued else 0, item_id),
            )

    def insert_briefing(self, watchlist_id: int, status: str = "generating") -> int:
        """Create a new `briefings` row for a watchlist.

        Args:
            watchlist_id: The watchlist this briefing belongs to.
            status: Initial status. Defaults to `"generating"`, matching
                every real caller's use -- the row exists before the LLM
                call is even made, so a crash before the first write still
                leaves a `generating` row for zombie recovery to find.

        Returns:
            The new row's `id`.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO briefings (watchlist_id, status) VALUES (?, ?)",
                (watchlist_id, status),
            )
            return cursor.lastrowid

    def update_briefing(self, briefing_id: int, **fields: Any) -> None:
        """Update named columns on a `briefings` row by id.

        Matches the sibling `update_subscription`'s pattern: keys are
        validated against an explicit allowlist of real `briefings` columns
        rather than trusted blindly, so a typo'd or renamed field raises
        immediately instead of silently building a SET clause for a column
        that was never meant to be settable this way. Each key surviving
        the allowlist is additionally run through
        `sql_validation.validate_identifier` before it reaches the SQL
        text -- belt-and-suspenders against the allowlist itself being
        edited to something unsafe later, since `validate_column_name`'s
        table-scoped form fails closed for any table (like `briefings`)
        outside its own `VALID_COLUMNS` map and would reject every field
        unconditionally.

        Args:
            briefing_id: `briefings.id` of the row to update.
            **fields: Column/value pairs to set. A no-op (returns
                immediately) when empty. `updated_at` is bumped to
                `CURRENT_TIMESTAMP` automatically unless the caller passes
                it explicitly.

        Raises:
            ValueError: If any key in `fields` is not an allowed column, or
                fails `validate_identifier`.
        """
        allowed_fields = (
            "status",
            "error",
            "covers_through_item_id",
            "covers_from_ts",
            "selection_mode",
            "preset_id",
            "model_used",
            "body_markdown",
            "item_count",
            "featured_count",
            "overflow_count",
            "updated_at",
        )
        if not fields:
            return
        for key in fields:
            if key not in allowed_fields:
                raise ValueError(f"update_briefing: unknown field {key!r}")
            if not validate_identifier(key, "column name"):
                raise ValueError(f"update_briefing: invalid field {key!r}")

        set_clause = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values())
        # Only append the automatic timestamp bump when the caller didn't
        # already supply `updated_at` explicitly -- otherwise the column
        # would appear twice in the same SET clause.
        extra = "" if "updated_at" in fields else ", updated_at = CURRENT_TIMESTAMP"
        values.append(briefing_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE briefings SET {set_clause}{extra} WHERE id = ?",
                values,
            )

    def get_briefing(self, briefing_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one `briefings` row by id.

        Args:
            briefing_id: `briefings.id` to look up.

        Returns:
            The row as a dict, or `None` if no row has that id.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM briefings WHERE id = ?", (briefing_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_briefings(self, watchlist_id: int) -> List[Dict[str, Any]]:
        """List a watchlist's briefings, newest first.

        Args:
            watchlist_id: The watchlist to list briefings for.

        Returns:
            Every `briefings` row for `watchlist_id`, newest first by
            `created_at` then `id` (the tiebreaker for rows created within
            the same timestamp resolution).
        """
        with self.transaction() as conn:
            cursor = conn.execute(
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
        from this status set. Do not add it here. Pact partner:
        `list_briefing_schedules` below uses the SAME
        `('complete','empty')` allowlist for `last_completed_at`, because a
        failed run must never advance the SCHEDULE either -- grep
        "list_briefing_schedules" before changing this set, and change both
        or neither.

        Args:
            watchlist_id: The watchlist to compute the watermark for.

        Returns:
            The max `covers_through_item_id` among this watchlist's
            `complete`/`empty` briefings, or `None` if it has none -- read
            by callers as "there is no watermark yet" (the first-briefing
            case).
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT MAX(covers_through_item_id) FROM briefings "
                "WHERE watchlist_id = ? AND status IN ('complete', 'empty')",
                (watchlist_id,),
            ).fetchone()
        return row[0] if row is not None else None

    def list_briefing_schedules(self) -> List[Dict[str, Any]]:
        """List every watchlist with a scheduled briefing cadence.

        One row per watchlist with a non-NULL `briefing_cadence_seconds`
        (Locked Decision 4 of the briefings phase 4 plan: scheduled
        briefings are opt-in per watchlist, so an un-cadenced watchlist
        must not appear here at all). Meant for the phase 4 scheduler
        projection to turn into due jobs.

        `last_completed_at` is computed with the exact same
        `status IN ('complete', 'empty')` allowlist as
        `latest_completed_watermark` above -- a `failed` briefing must not
        advance the schedule any more than it advances the coverage
        watermark (the schedule-side mirror of "failure never advances
        coverage": a run that failed did not complete, so the next
        scheduled attempt stays due from the same last-success point
        rather than being pushed out by the failure). Pact partner: grep
        `status IN ('complete', 'empty')` if you are touching either
        allowlist -- they must move together.

        Returns:
            A list of dicts, one per watchlist with
            `briefing_cadence_seconds IS NOT NULL`, each with keys
            `watchlist_id`, `name`, `briefing_cadence_seconds`, and
            `last_completed_at` (the max `created_at` among that
            watchlist's `complete`/`empty` briefings, or `None` if it has
            never completed one).
        """
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT
                    w.id AS watchlist_id,
                    w.name AS name,
                    w.briefing_cadence_seconds AS briefing_cadence_seconds,
                    (
                        SELECT MAX(b.created_at)
                        FROM briefings AS b
                        WHERE b.watchlist_id = w.id
                          AND b.status IN ('complete', 'empty')
                    ) AS last_completed_at
                FROM watchlists AS w
                WHERE w.briefing_cadence_seconds IS NOT NULL
                ORDER BY w.id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    # --- Briefing presets & scripts (spec #2 phase 2a) ---

    def insert_briefing_preset(
        self,
        name: str,
        *,
        roster_json: str,
        style_notes: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        """Create a new `briefing_presets` row.

        Args:
            name: Display name for the preset.
            roster_json: Canonical JSON encoding of the speaker roster (see
                `briefing_cast.dump_roster`). Stored verbatim; this layer
                does not parse or validate it.
            style_notes: Optional free-text style guidance appended to the
                cast prompt.
            provider: Optional LLM provider override for casting. `None`
                defers to `generate_script`'s own default resolution.
            model: Optional LLM model override for casting.

        Returns:
            The new row's `id`.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO briefing_presets (name, roster_json, style_notes, "
                "provider, model) VALUES (?, ?, ?, ?, ?)",
                (name, roster_json, style_notes, provider, model),
            )
            return cursor.lastrowid

    def update_briefing_preset(self, preset_id: int, **fields: Any) -> None:
        """Update named columns on a `briefing_presets` row by id.

        Matches `update_briefing`'s pattern: keys are validated against an
        explicit allowlist of real `briefing_presets` columns rather than
        trusted blindly, so a typo'd or renamed field raises immediately
        instead of silently building a SET clause for a column that was
        never meant to be settable this way. Each key surviving the
        allowlist is additionally run through `sql_validation.
        validate_identifier` before it reaches the SQL text --
        belt-and-suspenders against the allowlist itself being edited to
        something unsafe later.

        Args:
            preset_id: `briefing_presets.id` of the row to update.
            **fields: Column/value pairs to set. A no-op (returns
                immediately) when empty. `updated_at` is bumped to
                `CURRENT_TIMESTAMP` automatically unless the caller passes
                it explicitly.

        Raises:
            ValueError: If any key in `fields` is not an allowed column, or
                fails `validate_identifier`.
        """
        allowed_fields = (
            "name",
            "roster_json",
            "style_notes",
            "provider",
            "model",
            "updated_at",
        )
        if not fields:
            return
        for key in fields:
            if key not in allowed_fields:
                raise ValueError(f"update_briefing_preset: unknown field {key!r}")
            if not validate_identifier(key, "column name"):
                raise ValueError(f"update_briefing_preset: invalid field {key!r}")

        set_clause = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values())
        # Only append the automatic timestamp bump when the caller didn't
        # already supply `updated_at` explicitly -- otherwise the column
        # would appear twice in the same SET clause.
        extra = "" if "updated_at" in fields else ", updated_at = CURRENT_TIMESTAMP"
        values.append(preset_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE briefing_presets SET {set_clause}{extra} WHERE id = ?",
                values,
            )

    def get_briefing_preset(self, preset_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one `briefing_presets` row by id.

        Args:
            preset_id: `briefing_presets.id` to look up.

        Returns:
            The row as a dict, or `None` if no row has that id.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM briefing_presets WHERE id = ?", (preset_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_briefing_presets(
        self, *, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List `briefing_presets` rows, alphabetically by name, paginated.

        Args:
            limit: Maximum number of rows to return (CLAUDE.md Performance
                Rules: paginate DB results). Defaults to 200, well above
                any real preset count today, so existing callers keep
                working unchanged.
            offset: Number of rows to skip before the page starts.

        Returns:
            Up to `limit` `briefing_presets` rows, starting at `offset`,
            ordered by `name` ascending (case-sensitive SQLite default
            collation), then `id` as the tiebreaker for two presets
            sharing a name.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM briefing_presets ORDER BY name ASC, id ASC "
                "LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_briefing_preset(self, preset_id: int) -> bool:
        """Hard-delete a `briefing_presets` row.

        `briefing_scripts.preset_id` is not a foreign key (by design -- see
        the DDL comment in `_ensure_watchlists_schema`), so any script
        already cast from this preset is untouched: its `preset_name` and
        `roster_snapshot_json` already hold everything it needs, and its
        `preset_id` simply becomes a dangling back-reference, exactly as
        expected once the preset it named is gone.

        Args:
            preset_id: `briefing_presets.id` of the row to delete.

        Returns:
            `True` if a row was deleted, `False` if no row had that id.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM briefing_presets WHERE id = ?", (preset_id,)
            )
            return cursor.rowcount > 0

    def insert_briefing_script(
        self,
        briefing_id: int,
        *,
        preset_id: Optional[int],
        preset_name: str,
        roster_snapshot_json: str,
        status: str = "generating",
    ) -> int:
        """Create a new `briefing_scripts` row for a briefing.

        Args:
            briefing_id: The `briefings.id` this script narrates. The row
                is deleted automatically if that briefing is ever deleted
                (`ON DELETE CASCADE`).
            preset_id: The preset this cast started from, or `None` if the
                preset was deleted before this call (or never existed).
                Not a foreign key -- see `delete_briefing_preset`.
            preset_name: The preset's name at cast time, snapshotted so a
                later preset rename/delete never changes what this script
                says it was cast from.
            roster_snapshot_json: Canonical JSON encoding of the resolved
                roster at cast time (see `briefing_cast.dump_roster`).
            status: Initial status. Defaults to `"generating"`, matching
                every real caller's use -- the row exists before the LLM
                call is even made, so a crash before the first write still
                leaves a `generating` row for `fail_interrupted_scripts` to
                find.

        Returns:
            The new row's `id`.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO briefing_scripts (briefing_id, preset_id, preset_name, "
                "roster_snapshot_json, status) VALUES (?, ?, ?, ?, ?)",
                (briefing_id, preset_id, preset_name, roster_snapshot_json, status),
            )
            return cursor.lastrowid

    def update_briefing_script(self, script_id: int, **fields: Any) -> None:
        """Update named columns on a `briefing_scripts` row by id.

        Matches `update_briefing`'s pattern: keys are validated against an
        explicit allowlist of real `briefing_scripts` columns rather than
        trusted blindly, so a typo'd or renamed field raises immediately
        instead of silently building a SET clause for a column that was
        never meant to be settable this way. Each key surviving the
        allowlist is additionally run through `sql_validation.
        validate_identifier` before it reaches the SQL text --
        belt-and-suspenders against the allowlist itself being edited to
        something unsafe later.

        Args:
            script_id: `briefing_scripts.id` of the row to update.
            **fields: Column/value pairs to set. A no-op (returns
                immediately) when empty. `updated_at` is bumped to
                `CURRENT_TIMESTAMP` automatically unless the caller passes
                it explicitly.

        Raises:
            ValueError: If any key in `fields` is not an allowed column, or
                fails `validate_identifier`.
        """
        allowed_fields = (
            "status",
            "error",
            "turns_json",
            "model_used",
            "updated_at",
        )
        if not fields:
            return
        for key in fields:
            if key not in allowed_fields:
                raise ValueError(f"update_briefing_script: unknown field {key!r}")
            if not validate_identifier(key, "column name"):
                raise ValueError(f"update_briefing_script: invalid field {key!r}")

        set_clause = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values())
        # Only append the automatic timestamp bump when the caller didn't
        # already supply `updated_at` explicitly -- otherwise the column
        # would appear twice in the same SET clause.
        extra = "" if "updated_at" in fields else ", updated_at = CURRENT_TIMESTAMP"
        values.append(script_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE briefing_scripts SET {set_clause}{extra} WHERE id = ?",
                values,
            )

    def get_briefing_script(self, script_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one `briefing_scripts` row by id.

        Args:
            script_id: `briefing_scripts.id` to look up.

        Returns:
            The row as a dict, or `None` if no row has that id.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM briefing_scripts WHERE id = ?", (script_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_briefing_scripts(
        self, briefing_id: int, *, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List a briefing's cast scripts, newest first, paginated.

        Args:
            briefing_id: The briefing to list scripts for.
            limit: Maximum number of rows to return (CLAUDE.md Performance
                Rules: paginate DB results). Defaults to 200, well above
                any real per-briefing script count today, so existing
                callers keep working unchanged.
            offset: Number of rows to skip before the page starts.

        Returns:
            Up to `limit` `briefing_scripts` rows for `briefing_id`,
            starting at `offset`, newest first by `created_at` then `id`
            (the tiebreaker for rows created within the same timestamp
            resolution).
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM briefing_scripts WHERE briefing_id = ? "
                "ORDER BY created_at DESC, id DESC "
                "LIMIT ? OFFSET ?",
                (briefing_id, limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def create_briefing_audio(
        self,
        script_id: int,
        *,
        voice_snapshot_json: str,
        status: str = "generating",
    ) -> int:
        """Create a new `briefing_audio` row for a cast script.

        Args:
            script_id: The `briefing_scripts.id` this audio narrates. The
                row is deleted automatically if that script is ever deleted
                (`ON DELETE CASCADE`).
            voice_snapshot_json: Canonical JSON encoding of the voice
                assignment used for this render, frozen at synthesis-start
                time. Write-once: not settable via `update_briefing_audio`
                (see that method's docstring) -- a synthesized artifact's
                provenance must never be revisable after the fact.
            status: Initial status. Defaults to `"generating"`, matching
                `insert_briefing_script`'s reasoning -- the row exists
                before the synthesis call is even made, so a crash before
                the first write still leaves a `generating` row behind.

        Returns:
            The new row's `id`.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO briefing_audio (script_id, voice_snapshot_json, status) "
                "VALUES (?, ?, ?)",
                (script_id, voice_snapshot_json, status),
            )
            return cursor.lastrowid

    def update_briefing_audio(self, audio_id: int, **fields: Any) -> None:
        """Update named columns on a `briefing_audio` row by id.

        Matches `update_briefing_script`'s pattern: keys are validated
        against an explicit allowlist of real `briefing_audio` columns
        rather than trusted blindly, so a typo'd or renamed field raises
        immediately instead of silently building a SET clause for a column
        that was never meant to be settable this way. Each key surviving the
        allowlist is additionally run through `sql_validation.
        validate_identifier` before it reaches the SQL text --
        belt-and-suspenders against the allowlist itself being edited to
        something unsafe later. `voice_snapshot_json` is deliberately absent
        from the allowlist: it is write-once, exactly as `roster_snapshot_
        json` is on `briefing_scripts` -- a synthesized artifact's snapshot
        must not be revisable after the fact.

        Args:
            audio_id: `briefing_audio.id` of the row to update.
            **fields: Column/value pairs to set. A no-op (returns
                immediately) when empty. `updated_at` is bumped to
                `CURRENT_TIMESTAMP` automatically unless the caller passes
                it explicitly.

        Raises:
            ValueError: If any key in `fields` is not an allowed column, or
                fails `validate_identifier`.
        """
        allowed_fields = (
            "status",
            "error",
            "file_path",
            "duration_seconds",
            "turn_count",
            "updated_at",
        )
        if not fields:
            return
        for key in fields:
            if key not in allowed_fields:
                raise ValueError(f"update_briefing_audio: unknown field {key!r}")
            if not validate_identifier(key, "column name"):
                raise ValueError(f"update_briefing_audio: invalid field {key!r}")

        set_clause = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values())
        # Only append the automatic timestamp bump when the caller didn't
        # already supply `updated_at` explicitly -- otherwise the column
        # would appear twice in the same SET clause.
        extra = "" if "updated_at" in fields else ", updated_at = CURRENT_TIMESTAMP"
        values.append(audio_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE briefing_audio SET {set_clause}{extra} WHERE id = ?",
                values,
            )

    def get_briefing_audio(self, audio_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one `briefing_audio` row by id.

        Args:
            audio_id: `briefing_audio.id` to look up.

        Returns:
            The row as a dict, or `None` if no row has that id.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM briefing_audio WHERE id = ?", (audio_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_briefing_audio(
        self, script_id: int, *, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List a script's audio renders, newest first, paginated.

        Args:
            script_id: The `briefing_scripts.id` to list audio for.
            limit: Maximum number of rows to return (CLAUDE.md Performance
                Rules: paginate DB results). Defaults to 200, well above any
                real per-script audio render count today, so existing
                callers keep working unchanged.
            offset: Number of rows to skip before the page starts.

        Returns:
            Up to `limit` `briefing_audio` rows for `script_id`, starting at
            `offset`, newest first by `created_at` then `id` (the
            tiebreaker for rows created within the same timestamp
            resolution).
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM briefing_audio WHERE script_id = ? "
                "ORDER BY created_at DESC, id DESC "
                "LIMIT ? OFFSET ?",
                (script_id, limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def list_watchlist_audio_episodes(
        self, watchlist_id: int, *, limit: int = 500, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List a watchlist's finished audio episodes for feed export.

        One joined `SELECT` across `briefing_audio -> briefing_scripts ->
        briefings`, scoped to `watchlist_id` through the `briefings` row
        each script ultimately belongs to (`briefing_audio` has no
        `watchlist_id` column of its own). Only rows a listener -- or an
        RSS reader -- can actually play are returned: `audio.status =
        'complete'` AND `audio.file_path IS NOT NULL` are two independent
        predicates, since a `complete` render whose file write never
        landed is just as unplayable as one still `generating` or
        `failed`.

        Ordered by `briefings.created_at DESC, audio.id DESC` --
        deliberately *not* `list_briefing_audio`'s `audio.created_at`: a
        podcast feed reads newest-briefing-first (episode recency), not
        newest-render-first (when the audio happened to be synthesized).

        Args:
            watchlist_id: The `watchlists.id` to list episodes for. Scopes
                the join by identity -- audio belonging to any other
                watchlist's briefings is never returned, regardless of how
                many rows exist elsewhere.
            limit: Maximum number of rows to return (CLAUDE.md Performance
                Rules: paginate DB results). Defaults to 500.
            offset: Number of rows to skip before the page starts.

        Returns:
            Up to `limit` rows, starting at `offset`, newest-briefing-first.
            Each row is a dict with keys `audio_id`, `script_id`,
            `briefing_id`, `file_path`, `duration_seconds`, `turn_count`,
            `preset_name`, `briefing_created_at`, `briefing_status`,
            `covers_from_ts`, `model_used` -- the exact aliases Tasks 3
            (RSS feed generation) and 5 (the export button) quote by name.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT
                    audio.id AS audio_id,
                    audio.script_id AS script_id,
                    scripts.briefing_id AS briefing_id,
                    audio.file_path AS file_path,
                    audio.duration_seconds AS duration_seconds,
                    audio.turn_count AS turn_count,
                    scripts.preset_name AS preset_name,
                    briefings.created_at AS briefing_created_at,
                    briefings.status AS briefing_status,
                    briefings.covers_from_ts AS covers_from_ts,
                    briefings.model_used AS model_used
                FROM briefing_audio AS audio
                JOIN briefing_scripts AS scripts ON scripts.id = audio.script_id
                JOIN briefings ON briefings.id = scripts.briefing_id
                WHERE briefings.watchlist_id = ?
                  AND audio.status = 'complete'
                  AND audio.file_path IS NOT NULL
                ORDER BY briefings.created_at DESC, audio.id DESC
                LIMIT ? OFFSET ?
                """,
                (watchlist_id, limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def set_watchlist_briefing_settings(
        self,
        watchlist_id: int,
        *,
        selection_mode: Optional[str] = None,
        default_preset_id: object = _UNSET,
        briefing_cadence_seconds: object = _UNSET,
    ) -> None:
        """Write a watchlist's briefing selection mode, preset, and/or cadence.

        Three independent, optional writes in one call:

        - `selection_mode`: when given, must be one of the valid modes
          below and replaces `watchlists.briefing_selection_mode`. `None`
          (the default) leaves the column untouched -- there is no way to
          ask this column to be cleared to NULL, since a watchlist's
          selection mode is never meant to be absent.
        - `default_preset_id`: uses the module-level `_UNSET` sentinel
          (rather than `None`) as its "leave alone" default, because `None`
          is itself a legitimate value here -- it clears
          `watchlists.default_briefing_preset_id` back to "no default
          preset". Passing nothing leaves the column untouched; passing
          `None` explicitly clears it; passing an id sets it.
        - `briefing_cadence_seconds`: same `_UNSET`-sentinel shape as
          `default_preset_id`. Passing nothing leaves the column untouched;
          passing `None` explicitly clears `watchlists.briefing_cadence_seconds`
          back to "never scheduled" (Locked Decision 4 of the briefings
          phase 4 plan: scheduled briefings are opt-in per watchlist,
          `NULL` means never); passing a positive int sets the cadence.

        Args:
            watchlist_id: `watchlists.id` of the row to update.
            selection_mode: One of `("auto", "curated", "auto_featured")`,
                or `None` to leave the current value alone.
            default_preset_id: A `briefing_presets.id`, `None` to clear, or
                the `_UNSET` sentinel (default) to leave the current value
                alone.
            briefing_cadence_seconds: A positive number of seconds between
                scheduled briefings, `None` to clear (never scheduled), or
                the `_UNSET` sentinel (default) to leave the current value
                alone.

        Returns:
            None.

        Raises:
            ValueError: If `selection_mode` is given and is not one of the
                valid modes, or if `briefing_cadence_seconds` is given and
                is not a positive number.
        """
        # Pact: this tuple must name the exact same three strings, in the
        # same meaning, as `briefing_selection.VALID_MODES`
        # (`tldw_chatbook/Subscriptions/briefing_selection.py`) -- this DB
        # module cannot import from `Subscriptions/` (the dependency runs
        # the other way: `Subscriptions/` imports `DB/`), so the two cannot
        # share a single source of truth in code. TASK-1393 ordering-pact
        # convention: grep "briefing_selection.VALID_MODES" if you are
        # changing either side, and change both together.
        valid_modes = ("auto", "curated", "auto_featured")

        updates: List[str] = []
        values: List[Any] = []
        if selection_mode is not None:
            if selection_mode not in valid_modes:
                raise ValueError(
                    f"set_watchlist_briefing_settings: unknown selection_mode "
                    f"{selection_mode!r}; valid modes: {list(valid_modes)}"
                )
            updates.append("briefing_selection_mode = ?")
            values.append(selection_mode)
        if default_preset_id is not _UNSET:
            updates.append("default_briefing_preset_id = ?")
            values.append(default_preset_id)
        if briefing_cadence_seconds is not _UNSET:
            if briefing_cadence_seconds is not None and briefing_cadence_seconds <= 0:
                raise ValueError(
                    f"set_watchlist_briefing_settings: briefing_cadence_seconds "
                    f"must be a positive number of seconds or None (never); got "
                    f"{briefing_cadence_seconds!r}"
                )
            updates.append("briefing_cadence_seconds = ?")
            values.append(briefing_cadence_seconds)

        if not updates:
            return

        values.append(watchlist_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE watchlists SET {', '.join(updates)} WHERE id = ?",
                values,
            )

    def get_subscription_items_by_ids(
        self, item_ids: Sequence[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Fetch `subscription_items` rows by id, keyed by id.

        Chunks the `IN (...)` lookup at `_ITEM_ID_LOOKUP_CHUNK_SIZE` ids per
        statement rather than binding the entire `item_ids` sequence as one
        query (the Qodo unbounded-`NOT IN` lesson from phase 1's
        `briefing_selection._window_rows` -- see its docstring): a single
        statement bound with hundreds or thousands of placeholders risks
        SQLite's host-parameter limit for a heavy user's briefing.

        Args:
            item_ids: `subscription_items.id` values to fetch. Duplicates
                and any order are fine; the return value is keyed by id.

        Returns:
            A dict mapping each id in `item_ids` that actually has a row to
            that row as a dict. Ids with no matching row are simply absent
            -- not mapped to `None`. Returns `{}` for empty input.
        """
        if not item_ids:
            return {}

        result: Dict[int, Dict[str, Any]] = {}
        ids = list(item_ids)
        with self.transaction() as conn:
            for start in range(0, len(ids), _ITEM_ID_LOOKUP_CHUNK_SIZE):
                chunk = ids[start : start + _ITEM_ID_LOOKUP_CHUNK_SIZE]
                placeholders = ", ".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT * FROM subscription_items WHERE id IN ({placeholders})",
                    chunk,
                )
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    result[row_dict["id"]] = row_dict
        return result

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
