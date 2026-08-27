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

import atexit
import json
import sqlite3
import threading
import time
import weakref
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, TYPE_CHECKING, Union
from urllib.parse import urlparse, urlunparse
from urllib.parse import urlsplit, urlunsplit

# Third-Party Libraries
from loguru import logger

# Local Imports
from .private_sqlite import connect_private_sqlite
from .base_db import BaseDB
from .sql_validation import get_safe_order_by_clause, validate_identifier
from ..config import get_cli_setting
from ..Metrics.metrics_logger import log_counter, log_histogram
from ..Utils.fts5_match_forms import quote_fts5_token

if TYPE_CHECKING:
    from ..Subscriptions.watchlist_item_page import WatchlistItemCursor, WatchlistItemPage


_CURRENT_SCHEMA_VERSION = 2

INTERRUPTED_RUN_ERROR = (
    "Interrupted: the application stopped before this run finished."
)
INTERRUPTED_BRIEFING_ERROR = "interrupted"

SUBSCRIPTIONS_V1_TO_V2_SQL = """
CREATE TABLE briefing_items (
    briefing_id INTEGER NOT NULL REFERENCES briefings(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL,
    live_item_id INTEGER REFERENCES subscription_items(id) ON DELETE SET NULL,
    selection_position INTEGER,
    citation_position INTEGER,
    featured INTEGER NOT NULL DEFAULT 0,
    cited INTEGER NOT NULL DEFAULT 0,
    item_title TEXT,
    item_url TEXT,
    item_published_date TEXT,
    item_created_at TEXT,
    item_effective_date TEXT,
    source_id INTEGER,
    source_name TEXT,
    source_type TEXT,
    source_url TEXT,
    provenance_version INTEGER NOT NULL,
    PRIMARY KEY (briefing_id, item_id)
)
"""


def _sanitize_provenance_url(value: object) -> str | None:
    """Strip credentials, query, and fragment from one HTTP(S) snapshot URL."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if parsed.scheme.casefold() not in {"http", "https"} or not hostname:
            return None
        host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme.casefold(), netloc, parsed.path, "", ""))
    except ValueError:
        return None


@dataclass(frozen=True)
class BriefingProvenanceRow:
    """One selected item's immutable snapshot at briefing publication."""

    item_id: int
    selection_position: int
    citation_position: int | None
    featured: bool
    cited: bool
    item_title: str | None
    item_url: str | None
    item_published_date: str | None
    item_created_at: str | None
    item_effective_date: str | None
    source_id: int | None
    source_name: str | None
    source_type: str | None
    source_url: str | None


#: Fallback `auto_pause_threshold` when the config value is missing or
#: unusable (task-1410 AC#3) -- matches the schema column's own `CREATE
#: TABLE ... DEFAULT 10` and the `[subscriptions]` config template's own
#: `auto_pause_after_failures = 10`, so a broken/hand-edited config still
#: produces the same default a fresh install would get.
_DEFAULT_AUTO_PAUSE_THRESHOLD = 10

#: Lock-wait ceiling for every connection this database opens, in
#: milliseconds (task-19562 AC4).
#:
#: **Measured, not assumed** -- the acceptance criterion demanded exactly
#: that. Against a real `SubscriptionsDB`, with one connection holding
#: `BEGIN IMMEDIATE` for 1.0 s while a second timed its own:
#:
#:     busy_timeout (ms): 5000        <- inherited, nothing set it
#:     journal_mode     : wal
#:     second writer blocked for 1.07s -> acquired
#:
#: So the lane's PLAUSIBLE rating is CONFIRMED: nothing in this file,
#: `base_db.py` or the private-path connector ever set `busy_timeout`, and
#: Python's `sqlite3.connect(timeout=5.0)` default made it 5 s -- a writer
#: collision really does block its caller for as long as the lock is held,
#: up to that ceiling, and then raises `OperationalError`.
#:
#: Two narrowings the number does NOT support, both worth stating because
#: the obvious readings are wrong:
#:
#: * `journal_mode = wal`, so readers never block writers and writers never
#:   block readers. The exposure is **writer-vs-writer only**, not "any of
#:   the 22 async service methods".
#: * Setting this pragma is **not** the fix for the stall. 5000 is what the
#:   connection already had; the value is written down here so it is pinned
#:   and cannot drift silently if the connector ever passes its own
#:   `timeout=`. Lowering it would only convert a stall into an earlier
#:   `database is locked` exception on a path with no retry. The stall stops
#:   mattering because the sqlite work no longer runs on the event loop
#:   (part B, `Subscriptions/db_offload.py`), not because of this line.
BUSY_TIMEOUT_MS = 5000

#: Every live, writable `SubscriptionsDB`, weakly held, so the interpreter
#: can checkpoint their WALs on the way out (see
#: `_checkpoint_open_databases_at_exit`).
_OPEN_SUBSCRIPTIONS_DBS: "weakref.WeakSet[SubscriptionsDB]" = weakref.WeakSet()
_OPEN_DBS_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False

#: True once the exit hook is running. Checked before every `logger` call on
#: the settle path, and it is not defensive decoration: the first version of
#: this hook logged a genuine warning from a test process whose temporary
#: database directory had already been removed, and loguru's sink was gone
#: too -- so the *diagnostic* raised `ValueError: I/O operation on closed
#: file` and printed a logging traceback on every exit. A settle running
#: during teardown reports nothing; there is nobody left to report to.
_INTERPRETER_EXITING = False


class _ThreadExitCleanup:
    """Close and de-register one thread's connection when that thread ends.

    Review of PR #1964. `SubscriptionsDB._connections` holds a **strong**
    reference to every thread's connection so shutdown can count them, but
    `close()` only removes the *calling* thread's entry. A worker thread that
    ended without calling `close()` therefore left its connection pinned by
    that dict for the life of the process -- descriptor, `-wal` and `-shm`
    handles included. Measured over 20 concurrent short-lived threads: the
    registry stayed at 21 entries and 43 open descriptors, permanently, and a
    `gc.collect()` could not reclaim any of it.

    Nothing outside the owning thread may close a sqlite3 connection --

        ProgrammingError: SQLite objects created in a thread can only be
        used in that same thread.

    -- which is why `close_all_connections` reports other threads' connections
    instead of closing them. The one place the rule *is* satisfied is the
    dying thread itself: CPython clears a thread's `threading.local` storage
    on that thread as it exits, so an object living only in that storage gets
    finalized there. That is this class. Verified rather than assumed: over 10
    threads, `__del__` ran 10 times, `threading.get_ident()` inside it matched
    the ident recorded at construction every time, and the descriptor count on
    the database returned to its pre-thread baseline (20 -> 0).

    It deliberately does not checkpoint. The `-wal` is settled by SQLite when
    the last connection to the database closes, and by `checkpoint_wal` /
    `close_all_connections` on the shutdown path; a thread ending is not the
    place to add I/O that could raise.

    The instance must be reachable ONLY from the owning thread's local
    storage. Handing a reference to anything longer-lived (the registry
    included) would postpone the finalization this exists to trigger.
    """

    __slots__ = ("_connection", "_registry", "_lock", "_ident")

    def __init__(self, connection, registry, lock, ident: int) -> None:
        self._connection = connection
        self._registry = registry
        self._lock = lock
        self._ident = ident

    def detach(self) -> None:
        """Give up ownership -- the connection was closed explicitly instead."""
        self._connection = None

    def __del__(self) -> None:
        connection = self._connection
        if connection is None:
            return
        self._connection = None
        # Best-effort throughout: this runs during thread teardown, where a
        # raised exception becomes an "Exception ignored in" traceback on
        # stderr and helps nobody.
        try:
            with self._lock:
                if self._registry.get(self._ident) is connection:
                    del self._registry[self._ident]
        except Exception:  # noqa: BLE001 -- thread teardown, best effort
            pass
        try:
            connection.close()
        except Exception:  # noqa: BLE001 -- thread teardown, best effort
            pass


def _checkpoint_open_databases_at_exit() -> None:
    """Settle every open subscriptions database at interpreter exit.

    task-19562, and the measurement matters more than the intent here.
    `SubscriptionsDB` keeps **thread-local** connections and nothing ever
    closed them, so an app that ran watchlist checks exited with a
    connection still open per worker thread. The obvious conclusion --
    that the `-wal` is therefore left behind -- was **tested and is false
    for a clean exit**: a child process that wrote a 4.1 MB `-wal` and
    exited normally left only `subs.db` on disk, with this hook suppressed
    exactly as with it enabled. CPython finalizes the connection objects,
    and SQLite checkpoints and removes the `-wal` when the last connection
    to a database closes.

    So this hook is not what saves the `-wal` on the ordinary path. What it
    does buy is a *defined* moment and a defined error path: `atexit` runs
    while imports and sqlite are still usable, rather than depending on
    garbage-collection order during interpreter teardown (the regime that
    produces "Exception ignored in:" noise). The behaviour it performs is
    covered directly by
    `Tests/Subscriptions/test_subscriptions_db_connection_lifecycle.py`.

    The path where the `-wal` genuinely does survive is `app.py`'s
    SIGINT/SIGTERM handler, which calls `os._exit(0)` -- that skips
    `atexit` too, so no hook here can reach it. Recorded rather than
    papered over; the hard-exit itself is task-19561's subject.

    Deliberately best-effort and silent on failure: a diagnostic must never
    be the thing that breaks the exit.
    """
    global _INTERPRETER_EXITING
    _INTERPRETER_EXITING = True
    with _OPEN_DBS_LOCK:
        databases = list(_OPEN_SUBSCRIPTIONS_DBS)
    for database in databases:
        try:
            if not Path(database.db_path_str).exists():
                # The file went away under a still-live instance (routine
                # for a temporary-directory test). There is nothing to
                # settle, and touching it would only re-create it.
                continue
            database.close_all_connections()
        except Exception:  # noqa: BLE001 -- interpreter shutdown, best effort
            pass


def _sqlite_unicode_casefold(value: Any) -> str:
    """Casefold SQLite text without raising on NULL or malformed values."""
    return value.casefold() if isinstance(value, str) else ""


def _default_auto_pause_threshold() -> int:
    """The `auto_pause_threshold` column default for a NEW subscription.

    task-1410 AC#3. `[subscriptions].auto_pause_after_failures` (the
    user-facing knob documented in
    `Docs/Features/SUBSCRIPTION_IMPLEMENTATION_PLAN.md`) was, until this
    task, read by nothing: `auto_pause_threshold` is a separate
    per-subscription column that only ever got the schema's hardcoded
    `DEFAULT 10` for any row inserted without it. `add_subscription` calls
    this to seed that column when the caller does not pass
    `auto_pause_threshold` explicitly, reconciling the two:

    Precedence: an explicit `auto_pause_threshold` kwarg to
    `add_subscription` always wins -- this is consulted only when the
    caller omits it. Existing subscriptions are never touched (this only
    affects the INSERT default for a brand-new row); an update to the
    config afterwards does not retroactively change any subscription's
    stored column value.

    Uses the traditional three-argument `get_cli_setting(section, key,
    default)` form, never the two-argument dotted form -- TASK-1771: a
    caller-supplied default in the dotted form's second positional slot is
    walked as a path segment instead of honoured, and silently returns
    `None` rather than either the configured value or this default.

    Returns:
        The configured `auto_pause_after_failures` as a positive `int`, or
        `_DEFAULT_AUTO_PAUSE_THRESHOLD` if the config value is absent, not
        parseable as an int, or not positive.
    """
    configured = get_cli_setting(
        "subscriptions", "auto_pause_after_failures", _DEFAULT_AUTO_PAUSE_THRESHOLD
    )
    try:
        threshold = int(configured)
    except (TypeError, ValueError):
        return _DEFAULT_AUTO_PAUSE_THRESHOLD
    return threshold if threshold > 0 else _DEFAULT_AUTO_PAUSE_THRESHOLD


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


class SubscriptionsDBUnavailableError(SubscriptionError):
    """Fixed failure for an unavailable agent-readable Watchlists database."""

    def __init__(self) -> None:
        super().__init__("Watchlists database is unavailable")


class SubscriptionsDBReadError(SubscriptionError):
    """Fixed failure for a transient agent-read readiness operation."""

    def __init__(self) -> None:
        super().__init__("Watchlists database read failed")


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
    with closing(
        connect_private_sqlite("db.subscriptions.site_configs", db_path)
    ) as conn:
        if str(db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local watchlist/feed
        # cache) and avoids an fsync per commit -- this connection is
        # short-lived (one DDL script, one commit, close), but it can be the
        # first connection this file ever sees, so it must not leave the
        # file on DELETE+FULL for whichever connection opens it next
        # (task-15465).
        conn.execute("PRAGMA synchronous = NORMAL")
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

    _CURRENT_SCHEMA_VERSION = _CURRENT_SCHEMA_VERSION

    _AGENT_READ_REQUIRED_COLUMNS = {
        "subscriptions": frozenset(
            {
                "id",
                "name",
                "type",
                "source",
                "is_active",
                "is_paused",
                "check_frequency",
                "last_checked",
                "last_successful_check",
                "consecutive_failures",
                "created_at",
                "updated_at",
            }
        ),
        "subscription_items": frozenset(
            {
                "id",
                "subscription_id",
                "url",
                "title",
                "content",
                "published_date",
                "author",
                "status",
                "diff_summary",
                "change_percentage",
                "change_type",
                "canonical_url",
                "created_at",
                "updated_at",
                "content_format",
                "content_kind",
                "effective_date",
            }
        ),
        "watchlists": frozenset(
            {
                "id",
                "name",
                "is_active",
                "briefing_selection_mode",
                "default_briefing_preset_id",
                "briefing_cadence_seconds",
                "created_at",
                "updated_at",
            }
        ),
        "watchlist_sources": frozenset({"watchlist_id", "subscription_id"}),
        "local_watchlist_runs": frozenset(
            {
                "id",
                "source_id",
                "status",
                "started_at",
                "finished_at",
                "stats_json",
                "error_msg",
                "created_at",
                "updated_at",
            }
        ),
        "briefings": frozenset(
            {
                "id",
                "watchlist_id",
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
                "created_at",
                "updated_at",
            }
        ),
        "briefing_items": frozenset(
            {
                "briefing_id",
                "item_id",
                "live_item_id",
                "selection_position",
                "citation_position",
                "featured",
                "cited",
                "item_title",
                "item_url",
                "item_published_date",
                "item_created_at",
                "item_effective_date",
                "source_id",
                "source_name",
                "source_type",
                "source_url",
                "provenance_version",
            }
        ),
        "briefing_presets": frozenset({"id", "name"}),
    }
    _AGENT_READ_REQUIRED_INDEXES = frozenset(
        {
            "idx_watchlist_sources_subscription",
            "idx_briefings_watchlist_status",
            "idx_briefing_items_item",
            "idx_local_watchlist_runs_batch",
            "uq_local_watchlist_runs_active_source",
            "uq_briefings_generating_watchlist",
        }
    )

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str = "default",
        *,
        read_only: bool = False,
    ):
        """
        Initialize the Subscriptions database.

        Args:
            db_path: Path to the SQLite database file or ':memory:'
            client_id: Client identifier for multi-client support
            read_only: Open an existing database without initializing schema
        """
        self._local = threading.local()
        # Every thread-local connection this instance has open, keyed by the
        # ident of the thread that owns it (task-19562). `threading.local` is
        # invisible from any other thread, so without this registry nothing --
        # not shutdown, not a test -- could even *count* the connections, let
        # alone checkpoint behind them. Assigned before `super().__init__`
        # because schema initialization touches `self.conn`.
        #
        # The reference held here is a STRONG one, which is why every entry is
        # paired with a `_ThreadExitCleanup` in the owning thread's local
        # storage (review of PR #1964): without that, a thread that ended
        # without calling `close()` left its connection pinned by this dict for
        # the life of the process -- descriptor and WAL lock included --
        # inverting the very leak the registry was added to expose. A
        # `weakref.WeakValueDictionary` would be tidier and is not available:
        # CPython raises `TypeError: cannot create weak reference to
        # 'sqlite3.Connection' object` (measured on 3.12.11).
        self._connections: Dict[int, sqlite3.Connection] = {}
        self._connections_lock = threading.Lock()
        self._read_only = read_only
        # Only the monotonic complete state is retained. A false/incomplete
        # probe is deliberately not cached: a background FTS backfill may
        # make the same long-lived owner complete before its next search.
        self._fts_items_complete: Optional[bool] = None
        super().__init__(db_path, client_id, initialize_schema=not read_only)
        if read_only:
            try:
                self.conn
            except Exception:
                self.close()
                raise SubscriptionsDBUnavailableError() from None
        elif not self.is_memory_db:
            # Read-only instances have no `-wal` of their own to settle, and
            # an in-memory database ceases to exist with its connection.
            self._register_for_exit_checkpoint()

    def _get_connection(self) -> sqlite3.Connection:
        """Return a connection with foreign-key enforcement enabled.

        ``PRAGMA foreign_keys`` is per-connection and defaults to OFF, and
        ``BaseDB._get_connection`` sets only ``row_factory``. Without this
        override every ``ON DELETE CASCADE`` in this schema is inert, which
        silently orphaned ``subscription_items`` whenever a subscription was
        deleted. Matches ``ChaChaNotes_DB`` and ``Client_Media_DB_v2``, which
        each enable it per connection.

        task-22224 EXCEPTION -- connections here keep the legacy default
        isolation level for now instead of the store template's
        ``isolation_level = None`` (rule: ``Library_Ingest_Jobs_DB.py``
        module docstring). This file's write paths knowingly rely on the
        legacy implicit-BEGIN policy (see the long TASK-1362 comment above
        the extraction-fingerprint migration, which documents the reliance
        and works around its DDL gap with an explicit BEGIN IMMEDIATE), so
        flipping requires this file's own commit/write-site census first --
        its own task. Do NOT copy this pattern into new stores.
        """
        if self._read_only:
            conn = connect_private_sqlite(
                "db.subscriptions.agent_read",
                self.db_path_str,
                read_only=True,
                must_exist=True,
            )
            conn.row_factory = sqlite3.Row
            try:
                conn.create_function(
                    "unicode_casefold",
                    1,
                    _sqlite_unicode_casefold,
                    deterministic=True,
                )
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS};")
                conn.execute("PRAGMA query_only = ON;")
            except Exception:
                conn.close()
                raise
            return conn

        conn = super()._get_connection()
        conn.create_function(
            "unicode_casefold",
            1,
            _sqlite_unicode_casefold,
            deterministic=True,
        )
        conn.execute("PRAGMA foreign_keys = ON;")
        # Written down rather than inherited (task-19562 AC4, see
        # `BUSY_TIMEOUT_MS` for the measurement). Set BEFORE the WAL
        # conversion below for the reason `AgentRuns_DB` documents: turning a
        # rollback-journal file into a WAL one briefly needs an exclusive
        # lock, and that conversion must not run with no lock-wait budget.
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS};")
        if not self.is_memory_db:
            conn.execute("PRAGMA journal_mode = WAL;")
        # NORMAL is safe under WAL (SQLite-documented pairing: app-crash-safe,
        # only an OS/power crash can lose the last commit or two -- acceptable
        # for this local watchlist/feed cache) and avoids an fsync on every
        # commit; DELETE mode's default FULL previously made every writer
        # exclusive-lock readers too, a multi-second-stall candidate on slow
        # disks (task-15465). Unconditional: synchronous is per-connection,
        # so every connection this DB opens needs it, not just the first.
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def assert_agent_read_ready(self) -> None:
        """Require the exact core schema used by Watchlists agent reads.

        FTS tables are deliberately not part of readiness. Search checks FTS
        coverage separately and falls back to literal ``LIKE`` when the index
        is absent or incomplete.

        Raises:
            SubscriptionsDBUnavailableError: If a required table or column is
                unavailable. The fixed message contains no SQL, path, or
                stored value.
            SubscriptionsDBReadError: If the readiness read fails operationally.
                The fixed message contains no underlying exception payload.
        """
        try:
            conn = self.conn
            versions = [
                int(row[0])
                for row in conn.execute("SELECT version FROM schema_version")
            ]
            if versions != [_CURRENT_SCHEMA_VERSION]:
                raise SubscriptionsDBUnavailableError()
            for table, required_columns in self._AGENT_READ_REQUIRED_COLUMNS.items():
                columns = {
                    row[1] for row in conn.execute(f"PRAGMA table_xinfo({table})")
                }
                if not required_columns <= columns:
                    raise SubscriptionsDBUnavailableError()
            indexes = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_schema WHERE type = 'index'"
                )
            }
            if not self._AGENT_READ_REQUIRED_INDEXES <= indexes:
                raise SubscriptionsDBUnavailableError()
        except SubscriptionsDBUnavailableError:
            raise
        except (sqlite3.Error, OSError):
            raise SubscriptionsDBReadError() from None

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
        conn = self.conn
        has_version_table = conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
        if has_version_table:
            versions = [int(row[0]) for row in conn.execute("SELECT version FROM schema_version")]
            if versions == [1]:
                self._migrate_from_v1_to_v2(conn)
            elif versions != [_CURRENT_SCHEMA_VERSION]:
                raise SubscriptionError("Unsupported subscriptions schema version")

        with self.transaction() as conn:
            conn.executescript("""
            PRAGMA foreign_keys = ON;
            
            -- Schema version tracking
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY NOT NULL
            );
            INSERT OR IGNORE INTO schema_version (version) VALUES (2);
            
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

    def _migrate_from_v1_to_v2(self, conn: sqlite3.Connection | None = None) -> None:
        """Atomically rebuild durable briefing provenance and active claims."""
        conn = conn or self.conn
        if conn.in_transaction:
            conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("ALTER TABLE briefing_items RENAME TO briefing_items_v1")
            conn.execute(SUBSCRIPTIONS_V1_TO_V2_SQL)
            legacy_rows = conn.execute(
                "SELECT bi.briefing_id, bi.item_id, bi.featured, "
                "i.id AS live_item_id, i.title AS item_title, i.url AS item_url, "
                "i.published_date AS item_published_date, i.created_at AS item_created_at, "
                "i.effective_date AS item_effective_date, "
                "s.id AS source_id, s.name AS source_name, s.type AS source_type, "
                "s.source AS source_url "
                "FROM briefing_items_v1 bi "
                "LEFT JOIN subscription_items i ON i.id = bi.item_id "
                "LEFT JOIN subscriptions s ON s.id = i.subscription_id "
                "ORDER BY bi.briefing_id, bi.item_id"
            ).fetchall()
            for row in legacy_rows:
                conn.execute(
                    "INSERT INTO briefing_items "
                    "(briefing_id, item_id, live_item_id, featured, cited, "
                    "item_title, item_url, item_published_date, item_created_at, "
                    "item_effective_date, "
                    "source_id, source_name, source_type, source_url, provenance_version) "
                    "VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (
                        row["briefing_id"],
                        row["item_id"],
                        row["live_item_id"],
                        int(bool(row["featured"])),
                        row["item_title"],
                        _sanitize_provenance_url(row["item_url"]),
                        row["item_published_date"],
                        row["item_created_at"],
                        row["item_effective_date"],
                        row["source_id"],
                        row["source_name"],
                        row["source_type"],
                        _sanitize_provenance_url(row["source_url"]),
                    ),
                )
            conn.execute("DROP TABLE briefing_items_v1")
            conn.execute(
                "CREATE INDEX idx_briefing_items_item ON briefing_items(item_id)"
            )
            conn.execute(
                "UPDATE local_watchlist_runs AS older "
                "SET status = 'failed', error_msg = ?, "
                "finished_at = COALESCE(finished_at, updated_at) "
                "WHERE status IN ('queued', 'running') AND EXISTS ("
                "SELECT 1 FROM local_watchlist_runs AS newer "
                "WHERE newer.source_id = older.source_id "
                "AND newer.status IN ('queued', 'running') "
                "AND (newer.created_at > older.created_at "
                "OR (newer.created_at = older.created_at AND newer.id > older.id)))",
                (INTERRUPTED_RUN_ERROR,),
            )
            conn.execute(
                "UPDATE briefings AS older SET status = 'failed', error = ? "
                "WHERE status = 'generating' AND EXISTS ("
                "SELECT 1 FROM briefings AS newer "
                "WHERE newer.watchlist_id = older.watchlist_id "
                "AND newer.status = 'generating' "
                "AND (newer.created_at > older.created_at "
                "OR (newer.created_at = older.created_at AND newer.id > older.id)))",
                (INTERRUPTED_BRIEFING_ERROR,),
            )
            conn.execute(
                "CREATE UNIQUE INDEX uq_local_watchlist_runs_active_source "
                "ON local_watchlist_runs(source_id) "
                "WHERE status IN ('queued', 'running')"
            )
            conn.execute(
                "CREATE UNIQUE INDEX uq_briefings_generating_watchlist "
                "ON briefings(watchlist_id) WHERE status = 'generating'"
            )
            conn.execute("DELETE FROM schema_version")
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (_CURRENT_SCHEMA_VERSION,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

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
        items_cols = {
            row[1] for row in cursor.execute("PRAGMA table_info(subscription_items)")
        }
        if "queued_for_briefing" not in items_cols:
            cursor.execute(
                "ALTER TABLE subscription_items ADD COLUMN queued_for_briefing BOOLEAN DEFAULT 0"
            )
        if "run_id" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN run_id INTEGER")
        if "alert_matches" not in items_cols:
            cursor.execute(
                "ALTER TABLE subscription_items ADD COLUMN alert_matches TEXT"
            )

        # Add columns to subscription_filters
        filters_cols = {
            row[1] for row in cursor.execute("PRAGMA table_info(subscription_filters)")
        }
        if "priority" not in filters_cols:
            cursor.execute(
                "ALTER TABLE subscription_filters ADD COLUMN priority INTEGER DEFAULT 0"
            )
        if "is_include_required" not in filters_cols:
            cursor.execute(
                "ALTER TABLE subscription_filters ADD COLUMN is_include_required BOOLEAN DEFAULT 0"
            )

        # Widen CHECK constraint on subscription_filters.action.
        # Must check for the literal action value 'include' rather than the
        # bare substring, because the new column `is_include_required` would
        # otherwise make the substring match and skip the migration.
        existing_check = None
        for row in cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='subscription_filters'"
        ):
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
                cursor.execute(
                    "ALTER TABLE subscription_filters_new RENAME TO subscription_filters"
                )
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
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_subscription_items_run_id ON subscription_items(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_subscription_items_queued ON subscription_items(queued_for_briefing, status)"
        )

        # Reader/body columns. `content` holds the renderable body: article text
        # for feed items, diff text for site changes. `url_snapshots` remains the
        # authority for full-page and previous-snapshot views.
        if "content" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN content TEXT")
        if "content_format" not in items_cols:
            cursor.execute(
                "ALTER TABLE subscription_items ADD COLUMN content_format TEXT"
            )
        if "content_kind" not in items_cols:
            cursor.execute(
                "ALTER TABLE subscription_items ADD COLUMN content_kind TEXT"
            )
        # Flag is a separate boolean, not a status: the status CHECK has no
        # 'flagged' value, and an item can be flagged *and* reviewed at once.
        if "is_flagged" not in items_cols:
            cursor.execute(
                "ALTER TABLE subscription_items ADD COLUMN is_flagged BOOLEAN DEFAULT 0"
            )

        # TASK-15464: a stored, indexed effective-date column, replacing the
        # per-row `COALESCE(datetime(published_date), datetime(created_at))`
        # expression `get_new_items` used to ORDER BY -- unindexable, so
        # every Items-pane refresh sorted the WHOLE table before its LIMIT
        # ever applied.
        #
        # `GENERATED ALWAYS AS (...) VIRTUAL`, not a plain column maintained
        # by a trigger or by application code: it is auto-computed for every
        # existing row (no separate backfill UPDATE -- SQLite evaluates the
        # expression on demand, and materializes it into the index below at
        # index-build time, over however many legacy rows already exist) and
        # auto-maintained on every INSERT/UPDATE of `published_date` or
        # `created_at`, through every write path there is or ever will be --
        # unlike a trigger, which only covers paths someone remembered to
        # keep it in sync with, and unlike the current single write path
        # (`persist_subscription_item`), which would otherwise be the one
        # place a future column-writer would have to remember to update too.
        # `STORED` was tried first and rejected: SQLite's `ALTER TABLE ADD
        # COLUMN` refuses it outright ("cannot add a STORED column") --
        # only `VIRTUAL` can be added to an existing table after the fact;
        # probe-verified (task-15464) before deciding.
        #
        # The expression is deliberately byte-for-byte the same one
        # `get_new_items`'s ORDER BY and `since` predicate used to spell out
        # inline, and probe-verified (task-15464) against real SQLite
        # (3.49.1) before this was written:
        #   - NULL, `''`, or an unparseable `published_date` (`datetime()`
        #     returns NULL for all three) falls back to `created_at`,
        #     identically to the old expression -- because it IS the old
        #     expression, just stored instead of recomputed per row per
        #     query.
        #   - A mixed-format `created_at` (space-separated
        #     `CURRENT_TIMESTAMP` vs. ingest's ISO `T`+offset) normalizes
        #     through the same `datetime()` call either way.
        #   - Ties (equal effective date) sort by ascending `id` today, with
        #     or without this index -- SQLite appends the rowid as a
        #     non-unique index's own implicit final key, so an index scan
        #     produces the identical tie order a full-table sort already
        #     does. The ORDER BY clauses below make this explicit
        #     (`, i.id ASC`) rather than leaning on that implicit behaviour.
        #
        # TRAP, found by the same probe: `PRAGMA table_info` does NOT list a
        # virtual generated column at all -- SQLite reports it only through
        # `PRAGMA table_xinfo` (in that pragma's `hidden` field). The
        # idempotency guard below therefore reads `table_xinfo`, not the
        # `table_info`-sourced `items_cols` every other guard in this method
        # uses: guarding on `items_cols` here would find "effective_date"
        # absent FOREVER (it can never appear in `table_info`'s output) and
        # re-run the ALTER on every single schema init after the first,
        # crashing with "duplicate column name: effective_date" the very
        # next time this method runs.
        #
        # No `BEGIN IMMEDIATE` wrapper (contrast the `extraction_fingerprint`
        # migration above): that one needs atomicity because a DDL statement
        # autocommits immediately under Python's sqlite3 implicit-BEGIN
        # policy, and a crash between ITS ALTER and its follow-up UPDATEs
        # would leave the one-time gate spent with the data half-migrated.
        # There are no follow-up DML statements here to strand -- the ALTER
        # and the CREATE INDEX are each independently atomic DDL, and a
        # generated column cannot be written to directly (confirmed by the
        # same probe: an explicit INSERT/UPDATE naming it raises), so there
        # is no data-migration step for a crash to catch mid-way at all.
        items_xcols = {
            row[1] for row in cursor.execute("PRAGMA table_xinfo(subscription_items)")
        }
        if "effective_date" not in items_xcols:
            cursor.execute(
                "ALTER TABLE subscription_items ADD COLUMN effective_date TEXT "
                "GENERATED ALWAYS AS "
                "(COALESCE(datetime(published_date), datetime(created_at))) VIRTUAL"
            )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_subscription_items_effective_date "
            "ON subscription_items(effective_date DESC)"
        )

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
        snapshot_cols = {
            row[1] for row in cursor.execute("PRAGMA table_info(url_snapshots)")
        }
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
        if not cursor.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'briefing_items'"
        ).fetchone():
            cursor.execute(SUBSCRIPTIONS_V1_TO_V2_SQL)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_briefings_watchlist_status "
            "ON briefings(watchlist_id, status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_briefing_items_item "
            "ON briefing_items(item_id)"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_local_watchlist_runs_active_source "
            "ON local_watchlist_runs(source_id) "
            "WHERE status IN ('queued', 'running')"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_briefings_generating_watchlist "
            "ON briefings(watchlist_id) WHERE status = 'generating'"
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
        run_cols = {
            row[1] for row in cursor.execute("PRAGMA table_info(local_watchlist_runs)")
        }
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
    # One bound parameter per id in the undo restore's IN-list, and
    # `mark_all_read` batches are unbounded -- keep each chunk comfortably
    # under SQLITE_MAX_VARIABLE_NUMBER (999 on older builds).
    _RESTORE_ITEMS_CHUNK_SIZE = 500

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
        with self.transaction() as conn:
            rows = conn.execute(
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
        return {row[0]: {"total": row[1] or 0, "unread": row[2] or 0} for row in rows}

    def get_source_item_counts(self) -> Dict[int, Dict[str, int]]:
        """Per-source item totals and unread counts, for rail badges.

        One grouped query, mirroring `get_watchlist_item_counts`: adding
        sources never adds round-trips. Sources with no items are absent
        (a missing key renders as no badge, which is the honest state).

        Returns:
            Mapping of source id to ``{"total": int, "unread": int}``.
        """
        with self.transaction() as conn:
            rows = conn.execute(
                """
            SELECT subscription_id,
                   COUNT(id) AS total,
                   SUM(CASE WHEN status = 'new' THEN 1 ELSE 0 END) AS unread
            FROM subscription_items
            GROUP BY subscription_id
            """
            ).fetchall()
        return {row[0]: {"total": row[1] or 0, "unread": row[2] or 0} for row in rows}

    @property
    def conn(self):
        """Thread-local database connection, registered for shutdown."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            connection = self._get_connection()
            self._local.conn = connection
            with self._connections_lock:
                self._connections[threading.get_ident()] = connection
            # Assigned LAST, and only into this thread's local storage: from
            # here the cleanup owns closing and de-registering this connection
            # when the thread ends (review of PR #1964). Nothing else may hold
            # a reference to it, or it would never be finalized.
            self._local.connection_cleanup = _ThreadExitCleanup(
                connection,
                self._connections,
                self._connections_lock,
                threading.get_ident(),
            )
        return self._local.conn

    def _register_for_exit_checkpoint(self) -> None:
        """Join the set of databases the interpreter settles on the way out."""
        global _ATEXIT_REGISTERED
        with _OPEN_DBS_LOCK:
            _OPEN_SUBSCRIPTIONS_DBS.add(self)
            if not _ATEXIT_REGISTERED:
                atexit.register(_checkpoint_open_databases_at_exit)
                _ATEXIT_REGISTERED = True

    def checkpoint_wal(self) -> bool:
        """Fold the `-wal` back into the database file and truncate it.

        task-19562. Nothing in this app ever checkpointed this database
        explicitly. SQLite's automatic checkpoint keeps the `-wal` bounded
        but never truncates it, so a long-running app carries whatever the
        last burst of writes left there -- measured at 4.1 MB after 300
        inserts, and 0 bytes after one `wal_checkpoint(TRUNCATE)`. That is
        the standing cost this addresses; the file is separately (and
        adequately) settled by SQLite itself when the last connection to it
        closes, which is why the exit hook's own docstring is careful about
        what it does and does not buy.

        `TRUNCATE` needs every other connection to be idle; when one is not,
        SQLite reports busy rather than raising, and this falls back to
        `PASSIVE`, which folds in what it can without waiting. Either way the
        database file is complete afterwards.

        Returns:
            True when the `-wal` was truncated, False when only a partial
            (or no) checkpoint was possible -- including for an in-memory or
            read-only database, which have nothing to checkpoint.
        """
        if self.is_memory_db or self._read_only:
            return False
        try:
            connection = self.conn
            if connection.in_transaction:
                # A checkpoint cannot see past this connection's own open
                # transaction; committing here would durably persist work the
                # caller has not finished, so the honest answer is to decline.
                return False
            row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
        except Exception:  # noqa: BLE001
            # Broader than sqlite3.Error on purpose: `self.conn` above can
            # also raise from the private-path connector (the database file
            # deleted under a still-live instance -- routine in tests, and
            # possible at shutdown). A settle that cannot happen is a
            # warning, never a raise out of a close path.
            if not _INTERPRETER_EXITING:
                logger.warning("SubscriptionsDB WAL checkpoint failed during shutdown")
            return False
        # (busy, log_pages, checkpointed_pages); busy=0 means TRUNCATE ran.
        if row is not None and row[0] == 0:
            return True
        try:
            self.conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
        except sqlite3.Error:
            pass
        return False

    def close_all_connections(self) -> int:
        """Settle this database for shutdown: checkpoint, then close.

        task-19562. Closes the CALLING thread's connection after
        checkpointing the `-wal` (the checkpoint is database-wide, so one
        connection settles the file for all of them).

        Connections owned by *other, still-live* threads are counted and
        reported, not closed. That is a measured limitation, not an oversight:
        sqlite3 refuses a cross-thread close --

            ProgrammingError: SQLite objects created in a thread can only be
            used in that same thread.

        -- and an exception raised out of a shutdown path is worse than a
        connection the operating system is about to reclaim anyway. The part
        that actually matters for the file on disk (the checkpoint) is done
        regardless of which thread calls this.

        Connections whose thread has already EXITED are neither counted nor
        retained: `_ThreadExitCleanup` closed and de-registered each of them on
        its own thread as that thread ended (review of PR #1964). So the number
        returned is the number of connections a still-live thread could
        actually be using, and the registry itself holds no descriptor open.

        Returns:
            The number of connections still open on other live threads.
        """
        self.checkpoint_wal()
        self.close()
        with self._connections_lock:
            remaining = len(self._connections)
        if remaining and not _INTERPRETER_EXITING:
            logger.debug(
                "SubscriptionsDB connections remain open on other threads after "
                "WAL checkpoint"
            )
        return remaining

    @contextmanager
    def transaction(self, *, immediate: bool = False):
        """Context manager for database transactions, safe to nest.

        task-19562 part C. This used to commit unconditionally on exit. A
        nested `with self.transaction()` therefore had its INNER exit
        durably commit the OUTER transaction's work as well, so a later
        failure in the outer scope could no longer roll back what the inner
        block had already written -- silent partial persistence, with no
        error anywhere.

        Nesting is now tracked per thread (the connection is thread-local,
        so the depth must be too), mirroring `ChaChaNotes_DB`'s
        `TransactionContextManager`: only the OUTERMOST block commits or
        rolls back, and an inner block simply yields the same connection. An
        exception still propagates outward, so the outermost block rolls the
        whole unit back as a caller would expect.

        Measured, and the earlier note here was wrong. It claimed
        `record_check_result` -- the call site the task named -- did not nest
        ("instrumented depth 1"). Re-instrumented per argument shape:

            record_check_result WITH stats    -> 2 entries, depths [1, 2]
            record_check_result WITHOUT stats -> 1 entry,  depths [1]

        The nesting is `record_check_result` -> `_update_subscription_stats`
        -> `update_subscription_stats`, which opens its own `transaction()`
        for the `subscription_stats` upsert. It is reached whenever `stats`
        is truthy -- which is every real check, since `execute_run` always
        passes stats. The earlier measurement can only have exercised the
        `stats=None` path. So this was **live**, not latent: before this
        change, the daily-statistics write durably committed the enclosing
        subscription-health UPDATE. Nothing after that point in
        `record_check_result` can fail today (only metric logging follows),
        which is why no incident was ever observed -- but the ordering was
        one added statement away from silent partial persistence.

        Args:
            immediate: Acquire SQLite's write lock before yielding.

        Yields:
            The thread-local `sqlite3.Connection`. The same object is yielded
            to a nested block, so an inner `with` shares the outer's
            transaction rather than starting its own.

        Raises:
            Exception: Whatever the body raises, re-raised unchanged after the
                OUTERMOST block rolls back. An inner block does not roll back;
                the exception propagates so the outermost can.
        """
        conn = self.conn
        if not hasattr(self._local, "transaction_depth"):
            self._local.transaction_depth = 0

        if self._local.transaction_depth > 0:
            # Inner block: join the outer transaction. No commit, no
            # rollback -- the outermost owns both.
            self._local.transaction_depth += 1
            try:
                yield conn
            finally:
                self._local.transaction_depth -= 1
            return

        self._local.transaction_depth = 1
        try:
            if immediate:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._local.transaction_depth = 0

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

            # task-1410 AC#3: a caller that does not pass an explicit
            # `auto_pause_threshold` gets the configured
            # `[subscriptions].auto_pause_after_failures` default instead of
            # silently falling through to the schema's hardcoded `DEFAULT
            # 10` -- see `_default_auto_pause_threshold`'s docstring for the
            # full precedence. An explicit kwarg (including an explicit
            # `None`, which the loop above would already have captured)
            # always wins; this only fires when the field never made it
            # into `fields` at all.
            if "auto_pause_threshold" not in fields:
                fields["auto_pause_threshold"] = _default_auto_pause_threshold()

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
                # task-1410 AC#2: shared with `record_check_error` so the
                # two failure-recording paths cannot diverge on the
                # threshold check -- see `_advance_failure_and_maybe_pause`.
                just_paused = self._advance_failure_and_maybe_pause(
                    cursor, subscription_id, error, now
                )

            else:
                just_paused = False
                # Successful check
                cursor.execute(
                    """
                    UPDATE subscriptions
                    SET last_checked = ?,
                        last_successful_check = ?,
                        last_error = NULL,
                        error_count = 0,
                        consecutive_failures = 0,
                        is_paused = 0
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
                    # Fix wave for the task-1410 review (Finding #4): this used
                    # to inspect `error` for the substring "Auto-paused",
                    # which `_advance_failure_and_maybe_pause` never writes
                    # there (it goes to `logger.warning`, not `last_error`),
                    # so the label was always "false" even when this very
                    # call paused the subscription. Wired to the helper's own
                    # return value instead of a text-sniffing guess.
                    "auto_paused": "true" if just_paused else "false",
                },
            )

    def _advance_failure_and_maybe_pause(
        self,
        cursor: sqlite3.Cursor,
        subscription_id: int,
        error: str,
        now: str,
        *,
        force_pause: bool = False,
    ) -> bool:
        """Record one failed check and auto-pause once the streak meets threshold.

        task-1410 AC#2. The single write path both `record_check_result`'s
        error branch and `record_check_error` use for "a check just
        failed": it bumps `error_count`/`consecutive_failures`, stamps
        `last_checked`/`last_error`, then compares the POST-increment
        `consecutive_failures` against the subscription's own
        `auto_pause_threshold` column and pauses -- logging the same
        warning -- if it has been reached.

        Before this helper existed, `record_check_error` (the main
        failure path: `LocalWatchlistsService.record_run_failure` ->
        `record_check_error`) never consulted the threshold at all, so it
        could climb `consecutive_failures` forever without ever pausing,
        while `record_check_result`'s error branch (reachable only for
        all-error `url_list`/`sitemap` runs, task-1394) did. Routing both
        through this one method is what keeps them from diverging again.

        AC#1: this never clears `is_paused` -- the only `UPDATE ... SET
        is_paused = ...` below can set it to `1`, never `0`. A FAILURE never
        un-pauses a source; only a SUCCESS does. That un-pausing happens in
        `record_check_result`'s success branch (fix wave for the task-1410
        review: an auto-paused source had no writer that ever cleared
        `is_paused`, since this helper only runs on failures and
        `reset_subscription_errors` has no callers). This gives a paused
        source its only recourse: nothing re-checks a paused source on a
        schedule, but a manual re-check still runs (`launch_run`/
        `execute_run` have no paused guard), and a successful one resumes it.

        Args:
            cursor: An open cursor inside the caller's transaction.
            subscription_id: The subscription that just failed a check.
            error: The error message to record as `last_error`.
            now: UTC ISO timestamp for `last_checked`.
            force_pause: Pause on this failure regardless of the threshold
                comparison. Folds `record_check_error`'s legacy
                `should_pause` parameter into this single decision point
                instead of a second, independent write path -- the one
                that used to write `is_paused = 0` on every ordinary
                failure and clear an existing pause (AC#1).

        Returns:
            True only when this call TRANSITIONED the subscription from
            not-paused to paused, False otherwise (including a failure on an
            already-paused source) -- so `record_check_result`'s `auto_paused`
            metric and the warning count one pause per pause, not per failure
            (Finding #4 of the task-1410 review, Qodo transition-semantics
            follow-up) instead of text-sniffing `error`.
        """
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

        cursor.execute(
            """
            SELECT consecutive_failures, auto_pause_threshold, is_paused
            FROM subscriptions WHERE id = ?
            """,
            (subscription_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        already_paused = bool(row["is_paused"])
        threshold = row["auto_pause_threshold"]
        # A NULL/non-positive threshold means "auto-pause disabled for this
        # source", not "pause on the very first failure" (0/negative) or a
        # crash (`int >= None` raises TypeError). Production never seeds a
        # value like this (the config default is 10 and the service strips
        # `None`), but `update_subscription(auto_pause_threshold=...)`
        # accepts one directly, so the helper itself must not trust it.
        threshold_active = isinstance(threshold, (int, float)) and threshold > 0
        should_pause = force_pause or (
            threshold_active and row["consecutive_failures"] >= threshold
        )
        # Auto-pause is a state TRANSITION, not a per-failure event. A source
        # that is already paused and gets a failing MANUAL re-check (the
        # scheduler skips paused sources, so only a manual re-check reaches
        # here) still meets the threshold, but re-logging "Auto-paused" and
        # re-counting the metric on every such failure would over-count a
        # single pause (task-1410 review, Qodo). Only the 0->1 transition
        # logs and is reported.
        newly_paused = should_pause and not already_paused
        if should_pause and not already_paused:
            cursor.execute(
                "UPDATE subscriptions SET is_paused = 1 WHERE id = ?",
                (subscription_id,),
            )
            logger.warning(
                f"Auto-paused subscription {subscription_id} after "
                f"{row['consecutive_failures']} failures"
            )
        return newly_paused

    def record_check_error(
        self, subscription_id: int, error: str, should_pause: bool = False
    ) -> None:
        """Record a failed check, auto-pausing at `auto_pause_threshold`.

        task-1410: this used to write `is_paused = 1 if should_pause else
        0` unconditionally on every call. No production caller ever passed
        `should_pause=True` (grep confirms), so every recorded failure
        silently wrote `is_paused = 0` -- clearing any pause a *different*
        call had set, and never itself consulting `auto_pause_threshold`.
        That made this, the MAIN failure path
        (`LocalWatchlistsService.record_run_failure` calls this on every
        run failure), a live way to erase a pause and a dead way to create
        one.

        Both defects are fixed by routing through the shared
        `_advance_failure_and_maybe_pause` (AC#2): `is_paused` is now only
        ever set to `1`, never `0` (AC#1), and it pauses at the same
        threshold `record_check_result`'s error branch uses so the two
        cannot diverge (task-1394's all-error path already exercises that
        branch).

        `should_pause` is folded into that same decision rather than kept
        as a second, independent switch: passing `True` forces a pause on
        THIS failure regardless of the count -- for a caller that already
        knows the failure is terminal -- but, like the threshold path, it
        can still only ever set `is_paused = 1`, never clear it.

        Args:
            subscription_id: ID of the subscription.
            error: Error message to record.
            should_pause: Force a pause on this failure even if
                `consecutive_failures` has not yet reached
                `auto_pause_threshold`. No production caller sets this
                today; kept for callers that already know the failure is
                terminal.
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            self._advance_failure_and_maybe_pause(
                cursor, subscription_id, error, now, force_pause=should_pause
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

    @staticmethod
    def _quote_fts5_term(term: str) -> str:
        """Return `term` as a literal FTS5 string (embedded quotes doubled).

        The same rule `Library/library_fts_query.py`'s `_quote_fts_term`
        pins for the Library's RAG search: once every user term is a
        double-quoted literal, no input can introduce FTS5 operators
        (``OR``/``NEAR``/``NOT``, column filters, parentheses) -- the only
        bare syntax in the emitted query is the AND-join below. The
        Library's plural/singular widening is deliberately NOT copied: a
        reader scanning for a feed's own words wants exactly those words.
        """
        return quote_fts5_token(term)

    #: The list-page projection shared by `get_new_items` and its
    #: `_search_items_rows` search half (TASK-15464). Deliberately NOT
    #: `i.*`: the old `SELECT i.*` dragged two unbounded-size columns into
    #: every one of up to 100 list rows on every Items-pane refresh, and
    #: neither has a single reader on the list path --
    #:
    #: - `content`: the audit's named cost, full scraped article/diff text.
    #:   Read only by the DETAIL path now (`get_item_content`, a one-row
    #:   fetch on selection -- see that method's docstring), not by anything
    #:   that renders the list itself.
    #: - `extracted_data`: for an API-type subscription,
    #:   `LocalWatchlistsService._normalize_api_item` stores the entire raw
    #:   upstream item payload here (`"extracted_data": item`) -- just as
    #:   unbounded as `content`, and `normalize_watchlist_item` never maps
    #:   it into its output dict at all, on this query or any other. Nothing
    #:   downstream of `get_new_items` has ever read it.
    #:
    #: Most of the columns below ARE read somewhere downstream of this query
    #: today (`watchlist_normalizers.normalize_watchlist_item`, the reader's
    #: `item_dates` sort, or `WatchlistsCollectionsScreen` directly) --
    #: traced column-by-column in task-15464's Implementation Notes. Two
    #: kept anyway despite tracing to no reader at all: `categories` and
    #: `enclosures`. Unlike `content`/`extracted_data` above, neither is a
    #: large-payload column (small scalar/short-JSON fields), so cutting
    #: them would not have served the projection's actual goal -- narrowing
    #: away unbounded-size columns -- and would have been scope creep past
    #: what this task's ACs asked for. They stay, unread, until a future
    #: reader needs them or a separate cleanup task removes them on its own
    #: evidence.
    #:
    #: One exception traced the OTHER way: `article_list._render_row` (the
    #: Read tab's row renderer) DOES read `content` on the list path, for a
    #: 160-char preview snippet under the title (`html_text.body_snippet`).
    #: Rather than let that one reader drag the full column back in,
    #: `content_preview` is a CHEAP projected expression -- a 2000-character
    #: prefix (`substr` counts characters, not bytes), not the whole
    #: (possibly many-KB) body -- comfortably enough text for `body_snippet`
    #: to find its 160 plain-text characters after HTML-tag-stripping in
    #: every realistic case; a snippet cut a few characters short on some
    #: pathological markup-to-text ratio is an acceptable soft failure for a
    #: preview, unlike truncating the reader's actual body. `article_list.py`
    #: prefers this column and falls back to `content` only for a hand-built
    #: dict that never went through this query at all (tests; a future
    #: non-DB source).
    _LIST_ITEM_COLUMNS = (
        "i.id, i.subscription_id, i.url, i.title, i.content_hash, "
        "i.published_date, i.author, i.categories, i.enclosures, i.status, "
        "i.media_id, i.processing_error, i.previous_hash, "
        "i.change_percentage, i.diff_summary, i.change_type, "
        "i.canonical_url, i.duplicate_of, i.created_at, i.updated_at, "
        "i.queued_for_briefing, i.run_id, i.alert_matches, "
        "i.content_format, i.content_kind, i.is_flagged, "
        "i.effective_date, "
        "substr(i.content, 1, 2000) AS content_preview, "
        "s.name as subscription_name, s.type as subscription_type"
    )

    #: Narrow metadata projection for agent search. This is intentionally not
    #: based on the broader UI list projection: raw processing errors, alert
    #: matches, hashes, categories/enclosures, flags, and redundant previews
    #: are not part of the agent evidence contract.
    _AGENT_LIST_ITEM_COLUMNS = (
        "i.id, i.subscription_id, i.url, i.title, i.published_date, "
        "i.author, i.status, i.canonical_url, i.created_at, i.updated_at, "
        "i.content_format, i.content_kind, i.effective_date, "
        "s.name AS subscription_name, s.type AS subscription_type, "
        "s.source AS subscription_source, "
        "s.is_active AS subscription_is_active, "
        "s.is_paused AS subscription_is_paused, "
        "s.created_at AS subscription_created_at, "
        "s.updated_at AS subscription_updated_at, "
        "s.last_checked AS subscription_last_checked, "
        "s.last_successful_check AS subscription_last_successful_check"
    )
    _AGENT_SEARCH_PAGE_LIMIT = 50
    _AGENT_ITEM_ORDER_PROFILE = "subscription_items_agent"
    _READER_ITEM_ORDER_PROFILE = "subscription_items_reader"
    _AGENT_ITEM_ORDER_BY = get_safe_order_by_clause(_AGENT_ITEM_ORDER_PROFILE)
    _READER_ITEM_ORDER_BY = get_safe_order_by_clause(_READER_ITEM_ORDER_PROFILE)
    _AGENT_MEMBERSHIP_SOURCE_LIMIT = 50
    _AGENT_MEMBERSHIP_COLLECTION_LIMIT = 20
    _AGENT_RESOLUTION_CANDIDATE_LIMIT = 20

    @classmethod
    def _agent_search_projection(
        cls, search_terms: Sequence[str]
    ) -> tuple[str, List[Any]]:
        """Build the LIKE fallback's bounded literal-context projection.

        The body window is centered on the first query term found beyond its
        leading 1,000 characters. This includes a deep body match even when
        an earlier AND term matched only the title or author. With no deep
        body term, the ordinary leading preview is the useful bounded input.
        """
        deep_match_legs: List[str] = []
        projection_params: List[Any] = []
        for term in search_terms:
            deep_match_legs.append(
                "WHEN instr(lower(COALESCE(i.content, '')), lower(?)) > 1000 "
                "THEN instr(lower(COALESCE(i.content, '')), lower(?)) - 1000"
            )
            projection_params.extend((term, term))
        start_expression = (
            f"CASE {' '.join(deep_match_legs)} ELSE 1 END" if deep_match_legs else "1"
        )
        return (
            cls._AGENT_LIST_ITEM_COLUMNS
            + f", substr(i.content, {start_expression}, 2000) "
            "AS content_match_context",
            projection_params,
        )

    @classmethod
    def _agent_fts_search_projection(cls) -> str:
        """Build an FTS-tokenizer-aware, character-bounded body projection.

        FTS5 ``snippet`` uses the same Unicode/diacritic and punctuation
        tokenization as the MATCH that admitted the row. Restrict it to the
        content column (index 1), cap its token window, and apply a final
        character bound so unusually long tokens cannot make a list row
        unbounded. Empty markers keep public snippet formatting in the later
        service layer.
        """
        return (
            cls._AGENT_LIST_ITEM_COLUMNS
            + ", substr(snippet(subscription_items_fts, 1, '', '', '', 32), "
            "1, 2000) AS content_match_context"
        )

    @staticmethod
    def _item_scope_predicates(
        *,
        subscription_id: Optional[int],
        status: Optional[str],
        watchlist_id: Optional[int],
        statuses: Optional[Sequence[str]],
        since: Optional[str],
    ) -> tuple[List[str], List[Any]]:
        """Build the shared source, collection, status, and date predicates.

        The date floor targets the generated normalized ``effective_date``
        column and normalizes the bound value through SQLite ``datetime()``;
        this preserves the legacy mixed stored-date behavior and index path.
        """
        predicates: List[str] = []
        params: List[Any] = []
        if subscription_id is not None:
            predicates.append("i.subscription_id = ?")
            params.append(subscription_id)
        if status is not None:
            predicates.append("i.status = ?")
            params.append(status)
        if watchlist_id is not None:
            predicates.append(
                "i.subscription_id IN ("
                "SELECT subscription_id FROM watchlist_sources WHERE watchlist_id = ?"
                ")"
            )
            params.append(watchlist_id)
        if statuses is not None:
            placeholders = ", ".join("?" for _ in statuses)
            predicates.append(f"i.status IN ({placeholders})")
            params.extend(statuses)
        if since is not None:
            predicates.append("i.effective_date >= datetime(?)")
            params.append(since)
        return predicates, params

    def _subscription_items_fts_is_complete(self, conn: Any) -> bool:
        """Whether every current item id has a real FTS docsize row.

        An external-content FTS table can appear to contain all content rows
        even when its index is only partially backfilled. The shadow docsize
        table is the actual membership authority. An anti-join proves exact
        coverage of item ids; table presence or equal counts cannot.

        Only ``True`` is cached. Existing insert/update/delete triggers keep a
        proven-complete index complete, while an incomplete legacy database is
        rechecked on every later search so a background backfill can take
        effect without reopening this owner.
        """
        if self._fts_items_complete is True:
            return True
        try:
            row = conn.execute(
                """
                SELECT NOT EXISTS (
                    SELECT 1
                    FROM subscription_items i
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM subscription_items_fts_docsize d
                        WHERE d.id = i.id
                    )
                    LIMIT 1
                )
                """
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        complete = bool(row and row[0])
        if complete:
            self._fts_items_complete = True
        return complete

    def _search_items_rows(
        self,
        conn: Any,
        where_clause: str,
        params: List[Any],
        search_terms: List[str],
        limit: int,
        *,
        select_columns: Optional[str] = None,
        select_params: Sequence[Any] = (),
        fts_select_columns: Optional[str] = None,
        order_profile: str = _AGENT_ITEM_ORDER_PROFILE,
    ) -> List[Any]:
        """The `search` half of `get_new_items`: FTS5 MATCH, LIKE fallback.

        The happy path JOINs `subscription_items_fts` (external-content over
        title/content/author) and MATCHes the AND-of-quoted-terms query. The
        fallback fires on `sqlite3.OperationalError` -- the table missing on
        a pre-migration database, or fts5 compiled out of the bundled SQLite
        -- and answers the same question with AND-of-terms,
        OR-across-columns LIKEs whose wildcards are escaped (``%``/``_``/
        ``\\`` stay literal). Either way the caller gets rows; the search
        box must never raise into the reader.
        """
        selected_order_by = get_safe_order_by_clause(order_profile)
        columns = select_columns or self._LIST_ITEM_COLUMNS
        fts_columns = fts_select_columns or columns
        effective_fts_select_params = () if fts_select_columns else select_params
        match = " AND ".join(self._quote_fts5_term(term) for term in search_terms)
        fts_where = (
            f"{where_clause} AND subscription_items_fts MATCH ?"
            if where_clause
            else "WHERE subscription_items_fts MATCH ?"
        )
        if self._subscription_items_fts_is_complete(conn):
            try:
                return conn.execute(
                    f"""
                    SELECT {fts_columns}
                    FROM subscription_items i
                    JOIN subscription_items_fts ON subscription_items_fts.rowid = i.id
                    JOIN subscriptions s ON i.subscription_id = s.id
                    {fts_where}
                    ORDER BY {selected_order_by}
                    LIMIT ?
                    """,
                    tuple([*effective_fts_select_params, *params, match, limit]),
                ).fetchall()
            except sqlite3.OperationalError:
                logger.debug(
                    "subscription_items_fts unavailable; falling back to LIKE search."
                )
        like_clauses: List[str] = []
        like_params: List[Any] = []
        for term in search_terms:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like_clauses.append(
                "(i.title LIKE ? ESCAPE '\\' OR i.content LIKE ? ESCAPE '\\' "
                "OR i.author LIKE ? ESCAPE '\\')"
            )
            like_params.extend([f"%{escaped}%"] * 3)
        like_predicates = " AND ".join(like_clauses)
        like_where = (
            f"{where_clause} AND {like_predicates}"
            if where_clause
            else f"WHERE {like_predicates}"
        )
        return conn.execute(
            f"""
            SELECT {columns}
            FROM subscription_items i
            JOIN subscriptions s ON i.subscription_id = s.id
            {like_where}
            ORDER BY {selected_order_by}
            LIMIT ?
            """,
            tuple([*select_params, *params, *like_params, limit]),
        ).fetchall()

    def _reader_item_predicates(
        self,
        *,
        subscription_id: Optional[int],
        status: Optional[str],
        run_id: Optional[int],
        watchlist_id: Optional[int],
        unassigned_only: bool,
        statuses: Optional[Sequence[str]],
        is_flagged: Optional[bool],
        since: Optional[str],
    ) -> tuple[List[str], List[Any]]:
        """Build every non-text Reader item predicate in one place."""
        predicates, params = self._item_scope_predicates(
            subscription_id=subscription_id,
            status=status,
            watchlist_id=watchlist_id,
            statuses=statuses,
            since=since,
        )
        if run_id is not None:
            predicates.append("i.run_id = ?")
            params.append(run_id)
        if unassigned_only:
            predicates.append(
                "NOT EXISTS (SELECT 1 FROM watchlist_sources ws "
                "WHERE ws.subscription_id = i.subscription_id)"
            )
        if is_flagged is not None:
            predicates.append("i.is_flagged = ?")
            params.append(1 if is_flagged else 0)
        return predicates, params

    def _reader_search_parts(
        self, conn: Any, search_terms: Sequence[str]
    ) -> tuple[str, List[str], List[Any]]:
        """Return one stable FTS-or-LIKE search mode for a Reader query.

        A page's matching high-water, count, rows, and subsequent arrival
        count must answer the same search question.  Probe the FTS table once
        before building those statements; an absent/incomplete/broken index
        chooses the literal LIKE form for all of them.
        """
        if not search_terms:
            return "", [], []
        match = " AND ".join(self._quote_fts5_term(term) for term in search_terms)
        if self._subscription_items_fts_is_complete(conn):
            try:
                conn.execute(
                    "SELECT 1 FROM subscription_items_fts "
                    "WHERE subscription_items_fts MATCH ? LIMIT 1",
                    (match,),
                ).fetchone()
                return (
                    "JOIN subscription_items_fts "
                    "ON subscription_items_fts.rowid = i.id",
                    ["subscription_items_fts MATCH ?"],
                    [match],
                )
            except sqlite3.OperationalError:
                logger.debug(
                    "subscription_items_fts unavailable; Reader falling back to LIKE."
                )
        like_clauses: List[str] = []
        like_params: List[Any] = []
        for term in search_terms:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like_clauses.append(
                "(i.title LIKE ? ESCAPE '\\' OR i.content LIKE ? ESCAPE '\\' "
                "OR i.author LIKE ? ESCAPE '\\')"
            )
            like_params.extend([f"%{escaped}%"] * 3)
        return "", like_clauses, like_params

    @staticmethod
    def _reader_where_clause(predicates: Sequence[str]) -> str:
        """Return a WHERE clause for fixed, internally-built predicates."""
        return f"WHERE {' AND '.join(predicates)}" if predicates else ""

    def _reader_matching_parts(
        self,
        conn: Any,
        *,
        subscription_id: Optional[int],
        status: Optional[str],
        run_id: Optional[int],
        watchlist_id: Optional[int],
        unassigned_only: bool,
        statuses: Optional[Sequence[str]],
        is_flagged: Optional[bool],
        search: Optional[str],
        since: Optional[str],
    ) -> tuple[str, List[str], List[Any]]:
        """Return the shared Reader FROM join, predicates, and parameters."""
        predicates, params = self._reader_item_predicates(
            subscription_id=subscription_id,
            status=status,
            run_id=run_id,
            watchlist_id=watchlist_id,
            unassigned_only=unassigned_only,
            statuses=statuses,
            is_flagged=is_flagged,
            since=since,
        )
        search_terms = search.split() if search and search.strip() else []
        search_join, search_predicates, search_params = self._reader_search_parts(
            conn, search_terms
        )
        return search_join, [*predicates, *search_predicates], [*params, *search_params]

    @staticmethod
    def _validate_reader_query_inputs(
        *,
        status: Optional[str],
        statuses: Optional[Sequence[str]],
        limit: Optional[int] = None,
        snapshot_max_item_id: Optional[int] = None,
        after: Optional["WatchlistItemCursor"] = None,
    ) -> None:
        """Validate Reader API inputs shared by page and arrival queries."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        if status is not None and statuses is not None:
            raise ValueError("Pass either status or statuses, not both.")
        if after is None:
            return
        if snapshot_max_item_id is None:
            raise ValueError("snapshot watermark is required for continuation")
        if after.item_id < 1:
            raise ValueError("cursor item id must be positive")
        if snapshot_max_item_id < after.item_id:
            raise ValueError("snapshot watermark must not be below cursor item id")

    def _after_reader_page_high_water(self) -> None:
        """Test synchronization seam immediately after the first-page high-water."""

    def get_reader_items_page(
        self,
        *,
        subscription_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
        run_id: int | None = None,
        watchlist_id: int | None = None,
        unassigned_only: bool = False,
        statuses: Sequence[str] | None = None,
        is_flagged: bool | None = None,
        search: str | None = None,
        since: str | None = None,
        snapshot_max_item_id: int | None = None,
        after: "WatchlistItemCursor | None" = None,
    ) -> "WatchlistItemPage":
        """Return one Reader page in a stable DESC/DESC item snapshot.

        Args:
            subscription_id: Optional source scope.
            status: Optional single item status.
            limit: Number of returned rows; must be at least one.
            run_id: Optional producing-run scope.
            watchlist_id: Optional collection-membership scope.
            unassigned_only: Whether to include only unassigned sources.
            statuses: Optional multiple-status scope, exclusive with ``status``.
            is_flagged: Optional starred-state scope.
            search: Optional literal title/content/author search.
            since: Optional inclusive effective-date floor.
            snapshot_max_item_id: Existing snapshot high-water for continuation.
            after: Last returned Reader cursor for continuation.

        Returns:
            Immutable page data with raw SQLite row dictionaries.

        Raises:
            ValueError: If the page request or continuation cursor is invalid.
        """
        self._validate_reader_query_inputs(
            status=status,
            statuses=statuses,
            limit=limit,
            snapshot_max_item_id=snapshot_max_item_id,
            after=after,
        )
        from ..Subscriptions.watchlist_item_page import WatchlistItemCursor, WatchlistItemPage

        with self.transaction() as conn:
            # `transaction()` preserves nested write ownership but does not
            # itself issue a BEGIN for read-only statements.  A first Reader
            # page has three related SELECTs (high-water, count, rows), so
            # start a deferred SQLite read transaction when no caller-owned
            # transaction is active. It takes no write lock and pins one WAL
            # snapshot on the first read. An outer transaction owns its own
            # boundary, so never begin a nested transaction inside it.
            if not conn.in_transaction:
                conn.execute("BEGIN DEFERRED")
            search_join, predicates, params = self._reader_matching_parts(
                conn,
                subscription_id=subscription_id,
                status=status,
                run_id=run_id,
                watchlist_id=watchlist_id,
                unassigned_only=unassigned_only,
                statuses=statuses,
                is_flagged=is_flagged,
                search=search,
                since=since,
            )
            first_page = after is None
            if first_page and snapshot_max_item_id is None:
                high_water_row = conn.execute(
                    f"""
                    SELECT COALESCE(MAX(i.id), 0)
                    FROM subscription_items i
                    {search_join}
                    JOIN subscriptions s ON i.subscription_id = s.id
                    {self._reader_where_clause(predicates)}
                    """,
                    tuple(params),
                ).fetchone()
                snapshot_max_item_id = int(high_water_row[0])
                self._after_reader_page_high_water()
            assert snapshot_max_item_id is not None
            bounded_predicates = [*predicates, "i.id <= ?"]
            bounded_params = [*params, snapshot_max_item_id]
            snapshot_count: Optional[int] = None
            if first_page:
                count_row = conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM subscription_items i
                    {search_join}
                    JOIN subscriptions s ON i.subscription_id = s.id
                    {self._reader_where_clause(bounded_predicates)}
                    """,
                    tuple(bounded_params),
                ).fetchone()
                snapshot_count = int(count_row[0])
            if after is not None:
                if after.effective_date is None:
                    bounded_predicates.append("i.effective_date IS NULL AND i.id < ?")
                    bounded_params.append(after.item_id)
                else:
                    bounded_predicates.append(
                        "(i.effective_date IS NULL OR i.effective_date < datetime(?) "
                        "OR (i.effective_date = datetime(?) AND i.id < ?))"
                    )
                    bounded_params.extend(
                        [after.effective_date, after.effective_date, after.item_id]
                    )
            rows = conn.execute(
                f"""
                SELECT {self._LIST_ITEM_COLUMNS}
                FROM subscription_items i
                {search_join}
                JOIN subscriptions s ON i.subscription_id = s.id
                {self._reader_where_clause(bounded_predicates)}
                ORDER BY {self._READER_ITEM_ORDER_BY}
                LIMIT ?
                """,
                tuple([*bounded_params, limit + 1]),
            ).fetchall()
        visible_rows = [dict(row) for row in rows[:limit]]
        has_more = len(rows) > limit
        next_cursor = None
        if has_more and visible_rows:
            last_row = visible_rows[-1]
            next_cursor = WatchlistItemCursor(last_row["effective_date"], last_row["id"])
        return WatchlistItemPage(
            items=tuple(visible_rows),
            has_more=has_more,
            snapshot_max_item_id=snapshot_max_item_id,
            snapshot_count=snapshot_count,
            next_cursor=next_cursor,
        )

    def count_reader_item_arrivals(
        self,
        *,
        snapshot_max_item_id: int,
        subscription_id: int | None = None,
        status: str | None = None,
        run_id: int | None = None,
        watchlist_id: int | None = None,
        unassigned_only: bool = False,
        statuses: Sequence[str] | None = None,
        is_flagged: bool | None = None,
        search: str | None = None,
        since: str | None = None,
    ) -> int:
        """Count post-snapshot rows matching exactly one Reader query scope.

        Args:
            snapshot_max_item_id: Snapshot high-water that later rows exceed.
            subscription_id: Optional source scope.
            status: Optional single item status.
            run_id: Optional producing-run scope.
            watchlist_id: Optional collection-membership scope.
            unassigned_only: Whether to include only unassigned sources.
            statuses: Optional multiple-status scope, exclusive with ``status``.
            is_flagged: Optional starred-state scope.
            search: Optional literal title/content/author search.
            since: Optional inclusive effective-date floor.

        Returns:
            Count of matching rows created after the supplied high-water.

        Raises:
            ValueError: If mutually exclusive status inputs are supplied.
        """
        self._validate_reader_query_inputs(status=status, statuses=statuses)
        with self.transaction() as conn:
            search_join, predicates, params = self._reader_matching_parts(
                conn,
                subscription_id=subscription_id,
                status=status,
                run_id=run_id,
                watchlist_id=watchlist_id,
                unassigned_only=unassigned_only,
                statuses=statuses,
                is_flagged=is_flagged,
                search=search,
                since=since,
            )
            row = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM subscription_items i
                {search_join}
                JOIN subscriptions s ON i.subscription_id = s.id
                {self._reader_where_clause([*predicates, 'i.id > ?'])}
                """,
                tuple([*params, snapshot_max_item_id]),
            ).fetchone()
        return int(row[0])

    def search_items_for_agent(
        self,
        *,
        query: Optional[str] = None,
        subscription_id: Optional[int] = None,
        watchlist_id: Optional[int] = None,
        statuses: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        limit: int = 10,
        snapshot_max_item_id: Optional[int] = None,
        after_effective_date: Optional[str] = None,
        after_item_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return one bounded, stable-keyset page for agent evidence.

        This is a storage seam, not the public tool contract: validation,
        canonical IDs, cursors, JSON packing, URL handling, and excerpt text
        normalization belong to the tool service. Values are bound SQL
        parameters and all multi-row body material stays capped at 2,000
        characters.

        The first page captures the current maximum item id. Later pages pass
        that value back, admitting every pre-existing row (including one with
        a future effective date) while excluding all later inserts. The
        continuation key follows ``effective_date DESC, id ASC`` explicitly,
        including the NULL-date sink.

        Args:
            query: Literal whitespace-delimited search terms, or blank to
                browse without a text predicate.
            subscription_id: Optional source-row scope.
            watchlist_id: Optional collection-row scope.
            statuses: Optional status values to include; ``None`` means all.
            since: Optional inclusive effective-date floor.
            limit: Page size from 1 through 50.
            snapshot_max_item_id: First-page item-ID high-water, or ``None``
                to capture the current maximum.
            after_effective_date: Last row's effective date, or ``None`` for
                the NULL-date sink.
            after_item_id: Last row ID for keyset continuation.

        Returns:
            A mapping containing bounded ``items``, ``has_more``, and the
            traversal ``snapshot_max_item_id``.

        Raises:
            ValueError: If ``limit`` is outside 1..50 or an effective-date
                continuation omits ``after_item_id``.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if limit > self._AGENT_SEARCH_PAGE_LIMIT:
            raise ValueError(f"limit must be at most {self._AGENT_SEARCH_PAGE_LIMIT}")
        if after_effective_date is not None and after_item_id is None:
            raise ValueError("after_item_id is required with after_effective_date")

        search_terms = query.split() if query and query.strip() else []
        select_columns, select_params = self._agent_search_projection(search_terms)
        fts_select_columns = self._agent_fts_search_projection()
        predicates, params = self._item_scope_predicates(
            subscription_id=subscription_id,
            status=None,
            watchlist_id=watchlist_id,
            statuses=statuses,
            since=since,
        )

        with self.transaction() as conn:
            if snapshot_max_item_id is None:
                high_water_row = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM subscription_items"
                ).fetchone()
                snapshot_max_item_id = int(high_water_row[0])
            predicates.append("i.id <= ?")
            params.append(snapshot_max_item_id)

            if after_item_id is not None:
                if after_effective_date is None:
                    predicates.append("i.effective_date IS NULL AND i.id > ?")
                    params.append(after_item_id)
                else:
                    predicates.append(
                        "(i.effective_date IS NULL "
                        "OR i.effective_date < datetime(?) "
                        "OR (i.effective_date = datetime(?) AND i.id > ?))"
                    )
                    params.extend(
                        [after_effective_date, after_effective_date, after_item_id]
                    )

            where_clause = f"WHERE {' AND '.join(predicates)}"
            fetch_limit = limit + 1
            if search_terms:
                rows = self._search_items_rows(
                    conn,
                    where_clause,
                    params,
                    search_terms,
                    fetch_limit,
                    select_columns=select_columns,
                    select_params=select_params,
                    fts_select_columns=fts_select_columns,
                )
            else:
                rows = conn.execute(
                    f"""
                    SELECT {select_columns}
                    FROM subscription_items i
                    JOIN subscriptions s ON i.subscription_id = s.id
                    {where_clause}
                    ORDER BY i.effective_date DESC, i.id ASC
                    LIMIT ?
                    """,
                    tuple([*select_params, *params, fetch_limit]),
                ).fetchall()

        return {
            "items": [dict(row) for row in rows[:limit]],
            "has_more": len(rows) > limit,
            "snapshot_max_item_id": snapshot_max_item_id,
        }

    def get_item_detail_for_agent(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Return one authoritative item joined to its source, or ``None``.

        Unlike ``get_item_content``, a missing row is distinguishable from a
        present row whose article body is NULL. The single-row detail path may
        read full ``content``; raw ``extracted_data`` and source secrets are
        intentionally not projected.

        Args:
            item_id: ``subscription_items.id`` to retrieve.

        Returns:
            The allowlisted item/source row, including a possibly-null
            ``content`` value, or ``None`` when the item does not exist.
        """
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT
                    i.id, i.subscription_id, i.url, i.title, i.content,
                    i.published_date, i.author, i.status, i.diff_summary,
                    i.change_percentage, i.change_type, i.canonical_url,
                    i.created_at, i.updated_at, i.content_format,
                    i.content_kind, i.effective_date,
                    s.name AS subscription_name,
                    s.type AS subscription_type,
                    s.source AS subscription_source,
                    s.is_active AS subscription_is_active,
                    s.is_paused AS subscription_is_paused,
                    s.created_at AS subscription_created_at,
                    s.updated_at AS subscription_updated_at,
                    s.last_checked AS subscription_last_checked,
                    s.last_successful_check AS subscription_last_successful_check
                FROM subscription_items i
                JOIN subscriptions s ON s.id = i.subscription_id
                WHERE i.id = ?
                """,
                (item_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    @classmethod
    def _bounded_agent_candidate_limit(cls, limit: int) -> int:
        """Clamp a caller's candidate request to the storage boundary."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return min(limit, cls._AGENT_RESOLUTION_CANDIDATE_LIMIT)

    def resolve_source_candidates(
        self, query: Union[str, int], *, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Resolve an id, or exact name/URL first, else partial names.

        Both legs query ``subscriptions`` directly, so candidates beyond the
        legacy ``get_all_subscriptions``/UI scan ceiling remain reachable.
        Ambiguity is retained for the tool service to report rather than being
        silently resolved here.

        Args:
            query: Numeric source ID, or an exact/partial source name or exact
                configured URL. Exact names take precedence over exact URLs.
            limit: Requested candidate count, capped at 20.

        Returns:
            Deterministically ordered, allowlisted source candidates.

        Raises:
            ValueError: If ``limit`` is less than one.
        """
        bounded_limit = self._bounded_agent_candidate_limit(limit)
        source_columns = (
            "id, name, type, source, is_active, is_paused, created_at, "
            "updated_at, last_checked, last_successful_check"
        )
        with self.transaction() as conn:
            if isinstance(query, int):
                row = conn.execute(
                    f"SELECT {source_columns} FROM subscriptions WHERE id = ?",
                    (query,),
                ).fetchone()
                return [dict(row)] if row is not None else []
            rows = conn.execute(
                f"""
                SELECT {source_columns}
                FROM subscriptions
                WHERE unicode_casefold(name) = unicode_casefold(?)
                ORDER BY unicode_casefold(name), name, id
                LIMIT ?
                """,
                (query, bounded_limit),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    f"""
                    SELECT {source_columns}
                    FROM subscriptions
                    WHERE source = ?
                    ORDER BY unicode_casefold(name), name, id
                    LIMIT ?
                    """,
                    (query, bounded_limit),
                ).fetchall()
            if not rows:
                rows = conn.execute(
                    f"""
                    SELECT {source_columns}
                    FROM subscriptions
                    WHERE instr(unicode_casefold(name), unicode_casefold(?)) > 0
                    ORDER BY unicode_casefold(name), name, id
                    LIMIT ?
                    """,
                    (query, bounded_limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def resolve_collection_candidates(
        self, query: Union[str, int], *, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Resolve an id, or exact names first, else bounded partial names.

        Args:
            query: Numeric collection ID or an exact/partial collection name.
            limit: Requested candidate count, capped at 20.

        Returns:
            Deterministically ordered collection ID/name candidates.

        Raises:
            ValueError: If ``limit`` is less than one.
        """
        bounded_limit = self._bounded_agent_candidate_limit(limit)
        with self.transaction() as conn:
            if isinstance(query, int):
                row = conn.execute(
                    "SELECT id, name FROM watchlists WHERE id = ?", (query,)
                ).fetchone()
                return [dict(row)] if row is not None else []
            rows = conn.execute(
                """
                SELECT id, name
                FROM watchlists
                WHERE unicode_casefold(name) = unicode_casefold(?)
                ORDER BY unicode_casefold(name), name, id
                LIMIT ?
                """,
                (query, bounded_limit),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    """
                    SELECT id, name
                    FROM watchlists
                    WHERE instr(unicode_casefold(name), unicode_casefold(?)) > 0
                    ORDER BY unicode_casefold(name), name, id
                    LIMIT ?
                    """,
                    (query, bounded_limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_source_collection_memberships(
        self, subscription_ids: Sequence[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Load bounded collection memberships for at most one result page.

        One window-ranked ``IN`` query fetches at most one lookahead beyond
        the per-source collection cap. Each source therefore reports whether
        additional memberships were omitted, without an N+1 count/query.

        Args:
            subscription_ids: Source row IDs from one agent search page.

        Returns:
            Mapping of each requested source ID to ``collections`` (at most
            20 deterministic ID/name rows) and ``has_more`` truncation state.

        Raises:
            ValueError: If more than 50 distinct source IDs are supplied.
        """
        unique_ids = list(dict.fromkeys(subscription_ids))
        if len(unique_ids) > self._AGENT_MEMBERSHIP_SOURCE_LIMIT:
            raise ValueError(
                "source collection memberships accepts at most "
                f"{self._AGENT_MEMBERSHIP_SOURCE_LIMIT} source ids"
            )
        memberships: Dict[int, Dict[str, Any]] = {
            source_id: {"collections": [], "has_more": False}
            for source_id in unique_ids
        }
        if not unique_ids:
            return memberships
        placeholders = ", ".join("?" for _ in unique_ids)
        with self.transaction() as conn:
            rows = conn.execute(
                f"""
                WITH ranked_memberships AS (
                    SELECT ws.subscription_id, w.id, w.name,
                           ROW_NUMBER() OVER (
                               PARTITION BY ws.subscription_id
                               ORDER BY lower(w.name), w.name, w.id
                           ) AS membership_rank
                    FROM watchlist_sources ws
                    JOIN watchlists w ON w.id = ws.watchlist_id
                    WHERE ws.subscription_id IN ({placeholders})
                )
                SELECT subscription_id, id, name, membership_rank
                FROM ranked_memberships
                WHERE membership_rank <= ?
                ORDER BY subscription_id, membership_rank
                """,
                (*unique_ids, self._AGENT_MEMBERSHIP_COLLECTION_LIMIT + 1),
            ).fetchall()
        for row in rows:
            result = memberships[int(row["subscription_id"])]
            if row["membership_rank"] > self._AGENT_MEMBERSHIP_COLLECTION_LIMIT:
                result["has_more"] = True
                continue
            result["collections"].append({"id": int(row["id"]), "name": row["name"]})
        return memberships

    @staticmethod
    def _validate_agent_page(limit: int, *, maximum: int = 50) -> None:
        """Validate a bounded agent metadata page size."""
        if not 1 <= limit <= maximum:
            raise ValueError(f"limit must be between 1 and {maximum}")

    def list_sources_for_agent(
        self,
        *,
        name_query: Optional[str] = None,
        source_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_paused: Optional[bool] = None,
        watchlist_id: Optional[int] = None,
        limit: int = 10,
        after_name_casefold: Optional[str] = None,
        after_name: Optional[str] = None,
        after_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return one stable, allowlisted source-metadata page."""
        self._validate_agent_page(limit)
        cursor_values = (after_name_casefold, after_name, after_id)
        if any(value is not None for value in cursor_values) and any(
            value is None for value in cursor_values
        ):
            raise ValueError("source cursor fields must be supplied together")
        predicates: List[str] = ["typeof(s.name) = 'text'"]
        params: List[Any] = []
        if name_query is not None:
            predicates.append(
                "instr(unicode_casefold(s.name), unicode_casefold(?)) > 0"
            )
            params.append(name_query)
        if source_type is not None:
            predicates.append("s.type = ?")
            params.append(source_type)
        if is_active is not None:
            predicates.append("s.is_active = ?")
            params.append(int(is_active))
        if is_paused is not None:
            predicates.append("s.is_paused = ?")
            params.append(int(is_paused))
        if watchlist_id is not None:
            predicates.append(
                "EXISTS (SELECT 1 FROM watchlist_sources ws "
                "WHERE ws.subscription_id = s.id AND ws.watchlist_id = ?)"
            )
            params.append(watchlist_id)
        if after_id is not None:
            predicates.append(
                "s.id != ? AND (unicode_casefold(s.name) > ? "
                "OR (unicode_casefold(s.name) = ? AND s.name > ?) "
                "OR (unicode_casefold(s.name) = ? AND s.name = ? AND s.id > ?))"
            )
            params.extend(
                (
                    after_id,
                    after_name_casefold,
                    after_name_casefold,
                    after_name,
                    after_name_casefold,
                    after_name,
                    after_id,
                )
            )
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        with self.transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT s.id, s.name, s.type, s.source, s.is_active, s.is_paused,
                       s.check_frequency, s.last_checked,
                       s.last_successful_check, s.consecutive_failures,
                       s.created_at, s.updated_at
                FROM subscriptions s
                {where}
                ORDER BY unicode_casefold(s.name), s.name, s.id
                LIMIT ?
                """,
                (*params, limit + 1),
            ).fetchall()
        items = [dict(row) for row in rows[:limit]]
        for item in items:
            item["name_casefold"] = str(item["name"]).casefold()
        return {"items": items, "has_more": len(rows) > limit}

    def list_collections_for_agent(
        self,
        *,
        name_query: Optional[str] = None,
        limit: int = 10,
        after_name_casefold: Optional[str] = None,
        after_name: Optional[str] = None,
        after_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return one stable, allowlisted collection-metadata page."""
        self._validate_agent_page(limit)
        cursor_values = (after_name_casefold, after_name, after_id)
        if any(value is not None for value in cursor_values) and any(
            value is None for value in cursor_values
        ):
            raise ValueError("collection cursor fields must be supplied together")
        predicates: List[str] = ["typeof(w.name) = 'text'"]
        params: List[Any] = []
        if name_query is not None:
            predicates.append(
                "instr(unicode_casefold(w.name), unicode_casefold(?)) > 0"
            )
            params.append(name_query)
        if after_id is not None:
            predicates.append(
                "w.id != ? AND (unicode_casefold(w.name) > ? "
                "OR (unicode_casefold(w.name) = ? AND w.name > ?) "
                "OR (unicode_casefold(w.name) = ? AND w.name = ? AND w.id > ?))"
            )
            params.extend(
                (
                    after_id,
                    after_name_casefold,
                    after_name_casefold,
                    after_name,
                    after_name_casefold,
                    after_name,
                    after_id,
                )
            )
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        with self.transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT w.id, w.name, w.is_active, w.briefing_selection_mode,
                       w.default_briefing_preset_id, p.name AS default_preset_name,
                       w.briefing_cadence_seconds, w.created_at, w.updated_at,
                       COUNT(ws.subscription_id) AS source_count,
                       (SELECT b.created_at FROM briefings b
                        WHERE b.watchlist_id = w.id
                        ORDER BY datetime(b.created_at) DESC, b.id DESC LIMIT 1)
                           AS last_briefing_attempt_at,
                       (SELECT b.created_at FROM briefings b
                        WHERE b.watchlist_id = w.id AND b.status = 'complete'
                        ORDER BY datetime(b.created_at) DESC, b.id DESC LIMIT 1)
                           AS last_briefing_success_at,
                       (SELECT b.status FROM briefings b
                        WHERE b.watchlist_id = w.id
                        ORDER BY datetime(b.created_at) DESC, b.id DESC LIMIT 1)
                           AS last_briefing_status,
                       (SELECT b.id FROM briefings b
                        WHERE b.watchlist_id = w.id
                        ORDER BY datetime(b.created_at) DESC, b.id DESC LIMIT 1)
                           AS last_briefing_id
                FROM watchlists w
                LEFT JOIN watchlist_sources ws ON ws.watchlist_id = w.id
                LEFT JOIN briefing_presets p
                       ON p.id = w.default_briefing_preset_id
                {where}
                GROUP BY w.id
                ORDER BY unicode_casefold(w.name), w.name, w.id
                LIMIT ?
                """,
                (*params, limit + 1),
            ).fetchall()
        items = [dict(row) for row in rows[:limit]]
        for item in items:
            item["name_casefold"] = str(item["name"]).casefold()
        return {"items": items, "has_more": len(rows) > limit}

    @staticmethod
    def _briefing_agent_columns() -> str:
        """Return the fixed briefing-receipt projection."""
        return (
            "b.id, b.watchlist_id, w.name AS watchlist_name, b.status, "
            "b.covers_through_item_id, b.covers_from_ts, b.selection_mode, "
            "b.preset_id, p.name AS preset_name, b.model_used, "
            "b.item_count, b.featured_count, b.overflow_count, "
            "CASE WHEN b.body_markdown IS NOT NULL THEN 1 ELSE 0 END "
            "AS body_available, length(CAST(COALESCE(b.body_markdown, '') AS BLOB)) "
            "AS body_byte_count, b.created_at, b.updated_at, "
            "datetime(b.created_at) AS sort_created_at"
        )

    def list_briefings_for_agent(
        self,
        *,
        watchlist_id: Optional[int] = None,
        statuses: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        limit: int = 10,
        after_created_at: Optional[str] = None,
        after_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return one newest-first briefing-receipt page."""
        self._validate_agent_page(limit)
        predicates: List[str] = [
            "typeof(b.created_at) = 'text'",
            "length(b.created_at) <= 128",
        ]
        params: List[Any] = []
        if watchlist_id is not None:
            predicates.append("b.watchlist_id = ?")
            params.append(watchlist_id)
        if statuses is not None:
            placeholders = ", ".join("?" for _ in statuses)
            predicates.append(f"b.status IN ({placeholders})")
            params.extend(statuses)
        if since is not None:
            predicates.append("datetime(b.created_at) >= datetime(?)")
            params.append(since)
        if after_created_at is not None or after_id is not None:
            if after_created_at is None or after_id is None:
                raise ValueError("briefing cursor fields must be supplied together")
            predicates.append(
                "(datetime(b.created_at) < datetime(?) "
                "OR (datetime(b.created_at) = datetime(?) AND b.id < ?))"
            )
            params.extend((after_created_at, after_created_at, after_id))
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        with self.transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT {self._briefing_agent_columns()}
                FROM briefings b
                JOIN watchlists w ON w.id = b.watchlist_id
                LEFT JOIN briefing_presets p ON p.id = b.preset_id
                {where}
                ORDER BY datetime(b.created_at) DESC, b.id DESC
                LIMIT ?
                """,
                (*params, limit + 1),
            ).fetchall()
        return {
            "items": [dict(row) for row in rows[:limit]],
            "has_more": len(rows) > limit,
        }

    def get_briefing_for_agent(self, briefing_id: int) -> Optional[Dict[str, Any]]:
        """Return one allowlisted briefing receipt plus its Markdown body."""
        with self.transaction() as conn:
            row = conn.execute(
                f"""
                SELECT {self._briefing_agent_columns()}, b.body_markdown
                FROM briefings b
                JOIN watchlists w ON w.id = b.watchlist_id
                LEFT JOIN briefing_presets p ON p.id = b.preset_id
                WHERE b.id = ?
                """,
                (briefing_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_latest_completed_briefing_for_agent(
        self, watchlist_id: int, *, context_limit: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Return the newest readable completion plus newer attempt receipts."""
        self._validate_agent_page(context_limit, maximum=10)
        with self.transaction() as conn:
            completed = conn.execute(
                f"""
                SELECT {self._briefing_agent_columns()}
                FROM briefings b
                JOIN watchlists w ON w.id = b.watchlist_id
                LEFT JOIN briefing_presets p ON p.id = b.preset_id
                WHERE b.watchlist_id = ? AND b.status = 'complete'
                ORDER BY datetime(b.created_at) DESC, b.id DESC
                LIMIT 1
                """,
                (watchlist_id,),
            ).fetchone()
            if completed is None:
                return None
            newer = conn.execute(
                f"""
                SELECT {self._briefing_agent_columns()}
                FROM briefings b
                JOIN watchlists w ON w.id = b.watchlist_id
                LEFT JOIN briefing_presets p ON p.id = b.preset_id
                WHERE b.watchlist_id = ? AND b.status != 'complete'
                  AND (datetime(b.created_at) > datetime(?)
                       OR (datetime(b.created_at) = datetime(?) AND b.id > ?))
                ORDER BY datetime(b.created_at) DESC, b.id DESC
                LIMIT ?
                """,
                (
                    watchlist_id,
                    completed["created_at"],
                    completed["created_at"],
                    completed["id"],
                    context_limit,
                ),
            ).fetchall()
        return {
            "briefing": dict(completed),
            "newer_attempts": [dict(row) for row in newer],
        }

    def get_briefing_provenance_for_agent(
        self,
        briefing_id: int,
        *,
        limit: int = 50,
        selected_after: Optional[tuple[int, int, int]] = None,
        cited_after: Optional[tuple[int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Return bounded immutable selected and cited provenance snapshots."""
        self._validate_agent_page(limit)
        columns = (
            "item_id, live_item_id, selection_position, citation_position, "
            "featured, cited, item_title, item_url, item_published_date, "
            "item_created_at, item_effective_date, source_id, source_name, "
            "source_type, source_url, provenance_version"
        )
        def page(
            conn: sqlite3.Connection,
            *,
            position_column: str,
            cited_only: bool,
            after: Optional[tuple[int, int, int]],
        ) -> List[sqlite3.Row]:
            predicates = ["briefing_id = ?"]
            params: List[Any] = [briefing_id]
            if cited_only:
                predicates.append("cited = 1")
            if after is not None:
                predicates.append(
                    f"(({position_column} IS NULL), "
                    f"COALESCE({position_column}, 0), item_id) > (?, ?, ?)"
                )
                params.extend(after)
            return conn.execute(
                f"""
                SELECT {columns} FROM briefing_items
                WHERE {' AND '.join(predicates)}
                ORDER BY {position_column} IS NULL, {position_column}, item_id
                LIMIT ?
                """,
                (*params, limit + 1),
            ).fetchall()

        with self.transaction() as conn:
            selected = page(
                conn,
                position_column="selection_position",
                cited_only=False,
                after=selected_after,
            )
            cited = page(
                conn,
                position_column="citation_position",
                cited_only=True,
                after=cited_after,
            )
        return {
            "selected": [dict(row) for row in selected[:limit]],
            "selected_has_more": len(selected) > limit,
            "cited": [dict(row) for row in cited[:limit]],
            "cited_has_more": len(cited) > limit,
        }

    def list_operations_for_agent(
        self,
        *,
        source_id: Optional[int] = None,
        watchlist_id: Optional[int] = None,
        limit: int = 10,
        after_created_at: Optional[str] = None,
        after_kind: Optional[str] = None,
        after_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return bounded newest-first source-run and briefing receipts."""
        self._validate_agent_page(limit)
        cursor_values = (after_created_at, after_kind, after_id)
        if any(value is not None for value in cursor_values) and any(
            value is None for value in cursor_values
        ):
            raise ValueError("operation cursor fields must be supplied together")
        run_predicates: List[str] = [
            "typeof(r.created_at) = 'text'",
            "length(r.created_at) <= 128",
        ]
        run_params: List[Any] = []
        if source_id is not None:
            run_predicates.append("r.source_id = ?")
            run_params.append(source_id)
        if watchlist_id is not None:
            run_predicates.append(
                "EXISTS (SELECT 1 FROM watchlist_sources ws "
                "WHERE ws.subscription_id = r.source_id AND ws.watchlist_id = ?)"
            )
            run_params.append(watchlist_id)
        if after_created_at is not None:
            run_predicates.append(
                "(datetime(r.created_at) < datetime(?) OR "
                "(datetime(r.created_at) = datetime(?) AND "
                "('source_check' > ? OR "
                "('source_check' = ? AND r.id < ?))))"
            )
            run_params.extend(
                (after_created_at, after_created_at, after_kind, after_kind, after_id)
            )
        run_where = (
            f"WHERE {' AND '.join(run_predicates)}" if run_predicates else ""
        )
        briefing_predicates: List[str] = [
            "typeof(b.created_at) = 'text'",
            "length(b.created_at) <= 128",
        ]
        briefing_params: List[Any] = []
        if watchlist_id is not None:
            briefing_predicates.append("b.watchlist_id = ?")
            briefing_params.append(watchlist_id)
        if after_created_at is not None:
            briefing_predicates.append(
                "(datetime(b.created_at) < datetime(?) OR "
                "(datetime(b.created_at) = datetime(?) AND "
                "('briefing_generation' > ? OR "
                "('briefing_generation' = ? AND b.id < ?))))"
            )
            briefing_params.extend(
                (after_created_at, after_created_at, after_kind, after_kind, after_id)
            )
        briefing_where = (
            f"WHERE {' AND '.join(briefing_predicates)}"
            if briefing_predicates
            else ""
        )
        with self.transaction() as conn:
            runs = conn.execute(
                f"""
                SELECT r.id, r.source_id, s.name AS source_name, r.status,
                       r.started_at, r.finished_at, r.stats_json,
                       CASE WHEN r.error_msg IS NOT NULL THEN 1 ELSE 0 END AS has_error,
                       r.created_at, r.updated_at,
                       datetime(r.created_at) AS sort_created_at
                FROM local_watchlist_runs r
                JOIN subscriptions s ON s.id = r.source_id
                {run_where}
                ORDER BY datetime(r.created_at) DESC, r.id DESC
                LIMIT ?
                """,
                (*run_params, limit + 1),
            ).fetchall()
            briefings = conn.execute(
                f"""
                SELECT {self._briefing_agent_columns()}
                FROM briefings b
                JOIN watchlists w ON w.id = b.watchlist_id
                LEFT JOIN briefing_presets p ON p.id = b.preset_id
                {briefing_where}
                ORDER BY datetime(b.created_at) DESC, b.id DESC
                LIMIT ?
                """,
                (*briefing_params, limit + 1),
            ).fetchall()
        combined = [
            {"kind": "source_check", "row": dict(row)} for row in runs
        ] + [
            {"kind": "briefing_generation", "row": dict(row)}
            for row in briefings
        ]
        combined.sort(key=lambda item: int(item["row"]["id"]), reverse=True)
        combined.sort(key=lambda item: item["kind"])
        combined.sort(
            key=lambda item: str(item["row"]["sort_created_at"] or ""),
            reverse=True,
        )
        return {
            "operations": combined[:limit],
            "has_more": len(combined) > limit,
            "source_runs": [dict(row) for row in runs[:limit]],
            "briefings": [dict(row) for row in briefings[:limit]],
        }

    def get_watchlist_run_for_agent(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Return one exact allowlisted source-check receipt."""
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT r.id, r.source_id, s.name AS source_name, r.status,
                       r.started_at, r.finished_at, r.stats_json,
                       CASE WHEN r.error_msg IS NOT NULL THEN 1 ELSE 0 END AS has_error,
                       r.created_at, r.updated_at
                FROM local_watchlist_runs r
                JOIN subscriptions s ON s.id = r.source_id
                WHERE r.id = ?
                """,
                (run_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_new_items(
        self,
        subscription_id: Optional[int] = None,
        status: Optional[str] = "new",
        limit: int = 100,
        run_id: Optional[int] = None,
        watchlist_id: Optional[int] = None,
        unassigned_only: bool = False,
        statuses: Optional[List[str]] = None,
        is_flagged: Optional[bool] = None,
        search: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Items for a subscription (or all of them), newest first.

        TASK-2301. `status=None` means EVERY status, and until this task there
        was no way to ask for that: the status predicate was unconditional, so
        the only listing this class offered was a single-status one. The
        Watchlists Items tab reads through here, its filter reads "All
        statuses", and it was structurally incapable of showing anything but
        `new` -- so triaging an item (Ingest, Ignore, or merely opening it,
        which marks it read) made the row vanish from a list whose own filter
        said it should still be there. Acting on an item read as data loss.

        The default stays `"new"`, so every existing caller
        (`briefing_selection`, the smoke suite, the read-status tests) is
        untouched; only a caller that deliberately passes `None` sees the new
        behaviour.

        TASK-2513 adds the rail's scope dimensions: `watchlist_id` and
        `unassigned_only` filter by watchlist membership (independently of
        every other predicate, and of each other), and `statuses` is the
        multi-status form of `status` for a caller that wants more than one
        bucket without wanting all of them.

        TASK-15464: rows carry every `subscription_items` column EXCEPT
        `content` (full body text) and `extracted_data` (an API source's raw
        upstream payload) -- both unbounded-size, and neither has a reader
        on this list-page path (see `_LIST_ITEM_COLUMNS`'s docstring for the
        trace). A caller that needs the body of ONE already-listed item
        (opening it in the reader) fetches it separately through
        `get_item_content`, a single indexed-by-id read.

        Args:
            subscription_id: Restrict to one subscription, or `None` for all.
            status: The single status to return, or `None` for every status.
            limit: Maximum rows.
            run_id: Restrict to the items one run produced, or `None` for all
                runs. TASK-2306 -- the Runs tab's "Items" sub-region asks
                exactly this question, and `subscription_items.run_id` has
                carried the answer (with its own index) since the column was
                added; nothing had ever queried it.
            watchlist_id: Restrict to items whose subscription is a member of
                this watchlist, or `None` to not scope by membership.
            unassigned_only: Restrict to items whose subscription belongs to
                no watchlist at all -- the rail's Unassigned bucket.
            statuses: Several statuses to include, or `None` to defer to
                `status`. Requires `status=None` (a caller wanting the unread
                bucket by name passes `status="new"`, or names `"new"` in
                `statuses`); passing both is rejected rather than silently
                intersected.
            is_flagged: Restrict to starred rows (`True`) or unstarred rows
                (`False`), or `None` to not filter by the flag at all
                (TASK-3072 -- the Starred feed's page). Composes with every
                other predicate, the same as the membership scopes.
            search: Full-text terms over title/content/author (TASK-3791 --
                the reader's `/`). Whitespace-separated terms are ANDed, each
                matched literally (FTS5 operator syntax in the input is
                neutralized by quoting); the FTS table is used when it reads,
                with a LIKE fallback when it does not. `None` or blank passes
                no predicate at all.
            since: Effective-date floor (TASK-3791 -- the Today feed's page):
                only rows at/after `since` (inclusive). Both sides go through
                SQLite `datetime()` -- the stored columns are mixed-format
                (CURRENT_TIMESTAMP's space shape and ingest's ISO `T`+offset)
                and a bare string compare orders ' ' before 'T' (PR #1443
                review); an unparseable feed-supplied `published_date`
                normalizes to NULL and the COALESCE falls back to
                `created_at`.

        Returns:
            One dict per item row, joined to its subscription's name and type,
            ordered by EFFECTIVE date descending (`published_date`, falling
            back to `created_at` -- TASK-3072), both sides normalized through
            `datetime()` so mixed stored formats order correctly (PR #1443
            review); rows whose dates are both unparseable sink to the end of
            the page. The reader re-sorts its displayed page precisely in
            Python (`Subscriptions/item_dates.py`), so this clause's job is
            picking the right PAGE, not the final row order.

        Raises:
            ValueError: If both `status` and `statuses` are passed.
        """
        if status is not None and statuses is not None:
            raise ValueError("Pass either status or statuses, not both.")
        # Built as predicate fragments rather than hand-written SELECTs per
        # combination: the dimensions (subscription filter, status filter,
        # run filter, membership scope) are independent, and enumerating
        # their product is how the "all statuses" case came to be missing in
        # the first place. Values stay bound parameters -- only the fixed
        # predicate TEXT is assembled here.
        predicates, params = self._item_scope_predicates(
            subscription_id=subscription_id,
            status=status,
            watchlist_id=watchlist_id,
            statuses=statuses,
            since=since,
        )
        if run_id is not None:
            predicates.append("i.run_id = ?")
            params.append(run_id)
        if unassigned_only:
            predicates.append(
                "NOT EXISTS (SELECT 1 FROM watchlist_sources ws WHERE ws.subscription_id = i.subscription_id)"
            )
        if is_flagged is not None:
            predicates.append("i.is_flagged = ?")
            params.append(1 if is_flagged else 0)
        where_clause = f"WHERE {' AND '.join(predicates)}" if predicates else ""

        search_terms = search.split() if search and search.strip() else []
        with self.transaction() as conn:
            if search_terms:
                rows = self._search_items_rows(
                    conn, where_clause, params, search_terms, limit
                )
            else:
                params.append(limit)
                rows = conn.execute(
                    f"""
                    SELECT {self._LIST_ITEM_COLUMNS}
                    FROM subscription_items i
                    JOIN subscriptions s ON i.subscription_id = s.id
                    {where_clause}
                    ORDER BY i.effective_date DESC, i.id ASC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]

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

    def get_item_content(self, item_id: int) -> Optional[str]:
        """The full body text of one item -- the reader's DETAIL fetch.

        TASK-15464. `get_new_items`'s list-page projection (`_LIST_ITEM_
        COLUMNS`) deliberately excludes `content`: up to 100 rows' worth of
        full scraped article/diff text, dragged along on every Items-pane
        refresh for a column no list row ever rendered. This is the other
        half -- a single indexed-by-primary-key read, fetched once, only
        for the item actually opened in the reader.

        Deliberately `Optional[str]` rather than raising, UNLIKE the
        sibling `get_item_status` immediately above: `status` always has a
        value once a row exists (defaulting to `"new"`, so `None` can only
        mean "no such row" and a raise is the honest signal). `content` has
        no such guarantee -- a row that exists but was never scraped a body
        (mid-ingest, or a `change`-kind item whose renderable is
        `diff_summary` instead) legitimately has `content IS NULL`, which is
        not an error. Both "no such row" and "row exists, content is NULL"
        return `None` here, indistinguishably -- the caller
        (`WatchlistsCollectionsScreen._load_item_content`) treats a `None`
        the same way either way: leave whatever `content` the caller's own
        item dict already carries untouched, rather than raise or overwrite
        it with an empty body.

        Args:
            item_id: The `subscription_items` row id.

        Returns:
            The stored `content`, or `None` if no row has this id, or the
            row has one but its `content` column is itself NULL.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT content FROM subscription_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        return row["content"] if row is not None else None

    def get_url_snapshots(
        self, subscription_id: int, url: str, *, limit: int = 2
    ) -> List[Dict[str, Any]]:
        """Read the newest `limit` `url_snapshots` rows for one (subscription, url).

        TASK-1494: the reader's `[full page]` (index 0) and `[previous
        snapshot]` (index 1) affordances read this. `_SNAPSHOTS_KEPT_PER_URL`
        (`monitoring_engine.py`) prunes each `(subscription_id, url)` down to
        its newest three rows, and slot 2 exists specifically so this call
        always has something to answer "previous" with once a URL has
        changed at least twice.

        `ORDER BY created_at DESC, id DESC` is the SAME ordering the prune's
        own survivor-selection DELETE and `URLMonitor.check_url`'s baseline
        SELECT both use (the "TASK-1393 ordering pact", named in both of
        their comments) -- diverging it here would answer "newest"/
        "second-newest" against a different ordering than the one that
        decided which rows survived pruning, so this could return a row the
        prune already deleted, or skip one it kept.

        Args:
            subscription_id: Owning subscription (`url_snapshots.subscription_id`).
            url: The exact URL the snapshot was captured for. Load-bearing,
                not optional: a `url_list`/`sitemap` source's URLs all share
                one `subscription_id`, so without this predicate "the
                previous snapshot" would be whichever URL of the source was
                checked last, not this item's own page.
            limit: How many rows to return, newest first. `2` answers both
                affordances (full page, previous snapshot) in one call.

        Returns:
            Up to `limit` dicts with `id`, `extracted_content`, `created_at`
            -- newest first. Empty when this (subscription, url) pair has no
            stored snapshot at all.
        """
        # `transaction()` like the sibling reads (`get_item_status` above):
        # this file's convention, unlike some other DB modules (task-1494
        # Qodo round).
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, extracted_content, created_at
                FROM url_snapshots
                WHERE subscription_id = ? AND url = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (subscription_id, url, limit),
            ).fetchall()
        return [dict(row) for row in rows]

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

    def mark_all_read(
        self,
        subscription_id: Optional[int] = None,
        watchlist_id: Optional[int] = None,
        unassigned_only: bool = False,
    ) -> List[int]:
        """Mark every ``new`` item in scope ``reviewed``; return the affected ids.

        One transactional UPDATE. Only ``new`` rows are touched —
        ``reviewed``/``ingested``/``ignored``/``error`` all record deliberate
        user actions and are never rewritten here (same rule
        `persist_subscription_item`'s upsert follows, `item_persist.py:132-136`).
        The returned ids are the undo batch for
        `WatchlistsCollectionsScreen.action_undo_mark_all_read`.

        Args:
            subscription_id: Restrict to one source's items, or `None` for all.
            watchlist_id: Restrict to items of the sources in one watchlist
                (same sub-select `get_new_items` uses).
            unassigned_only: Restrict to items of sources belonging to no
                watchlist (same NOT EXISTS shape `get_new_items` uses).

        Returns:
            The ids of the rows moved to ``reviewed``.
        """
        predicates = ["status = 'new'"]
        params: List[Any] = []
        if subscription_id is not None:
            predicates.append("subscription_id = ?")
            params.append(subscription_id)
        if watchlist_id is not None:
            predicates.append(
                "subscription_id IN (SELECT subscription_id FROM watchlist_sources WHERE watchlist_id = ?)"
            )
            params.append(watchlist_id)
        if unassigned_only:
            predicates.append(
                "NOT EXISTS (SELECT 1 FROM watchlist_sources ws WHERE ws.subscription_id = subscription_items.subscription_id)"
            )
        with self.transaction() as conn:
            rows = conn.execute(
                f"UPDATE subscription_items SET status = 'reviewed' WHERE {' AND '.join(predicates)} RETURNING id",
                tuple(params),
            ).fetchall()
        return [row[0] for row in rows]

    def restore_items_new(self, item_ids: List[int]) -> int:
        """Move the given ids back to ``new`` — but only ones still ``reviewed``.

        The undo half of `mark_all_read`. The ``status = 'reviewed'`` guard
        means an item the user has since ingested or ignored is not yanked
        back to unread.

        Args:
            item_ids: The undo batch `mark_all_read` returned.

        Returns:
            How many rows were actually restored.
        """
        if not item_ids:
            return 0
        # Chunked (Qodo review, PR #1383): the IN-list binds one parameter
        # per id and `mark_all_read` batches are unbounded, so a single
        # statement could exceed SQLite's host-parameter limit. One
        # transaction still wraps every chunk, so a mid-batch failure
        # rolls the whole restore back.
        restored = 0
        with self.transaction() as conn:
            for offset in range(0, len(item_ids), self._RESTORE_ITEMS_CHUNK_SIZE):
                chunk = item_ids[offset : offset + self._RESTORE_ITEMS_CHUNK_SIZE]
                placeholders = ", ".join("?" for _ in chunk)
                cursor = conn.execute(
                    f"UPDATE subscription_items SET status = 'new' WHERE id IN ({placeholders}) AND status = 'reviewed'",
                    tuple(chunk),
                )
                restored += cursor.rowcount
        return restored

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

    def set_item_flagged(self, item_id: int, flagged: bool) -> None:
        """Set or clear the global "starred" flag on one item (TASK-3072).

        Same shape and same global semantics as `set_item_briefing_queued`:
        one row, one flag -- an item starred through any scope reads starred
        in all of them, and nothing but this explicit call changes it.
        `persist_subscription_item` never writes the column, so the flag
        survives re-fetches (pinned in
        `Tests/DB/test_subscriptions_db_watchlists.py`).

        Args:
            item_id: `subscription_items.id` to update.
            flagged: `True` to star the item, `False` to unstar it.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE subscription_items SET is_flagged = ? WHERE id = ?",
                (1 if flagged else 0, item_id),
            )

    def get_flagged_items_count(self) -> int:
        """How many items are starred, across every source and status.

        The Starred rail node's badge (TASK-3072). Status-agnostic on
        purpose: starring is orthogonal to triage, and a badge that shrank
        as the user read their starred items would read as data loss.

        Returns:
            The count of rows with ``is_flagged = 1``.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM subscription_items WHERE is_flagged = 1"
            ).fetchone()
        return int(row[0]) if row else 0

    def get_unread_items_count_since(self, since: str) -> int:
        """How many unread items fall at/after `since` -- the Today badge.

        TASK-3791. The floor compares the EFFECTIVE date (``published_date``,
        falling back to ``created_at``), the same expression `get_new_items`
        orders by and its `since` predicate filters on, so the badge and the
        node's page answer the same question.

        The predicate reads the ``effective_date`` generated column
        (TASK-15770), not the inline
        ``COALESCE(datetime(published_date), datetime(created_at))`` it used
        to spell out: the column IS that expression (task-15464's
        ``_ensure_watchlists_schema`` block), so the count is unchanged, but
        the column name lets ``idx_subscription_items_effective_date`` serve
        the floor as an index range instead of a full-table scan -- SQLite
        (probe-verified on 3.49.1) does NOT rewrite the byte-identical
        inline expression to the generated column on its own.

        Args:
            since: Inclusive ISO floor (the screen passes local midnight).

        Returns:
            The count of ``status = 'new'`` rows at or after the floor.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM subscription_items "
                "WHERE status = 'new' AND effective_date >= datetime(?)",
                (since,),
            ).fetchone()
        return int(row[0]) if row else 0

    def get_subscription_id_by_source(self, source: str) -> Optional[int]:
        """The id of the subscription with exactly this `source` URL, or None.

        TASK-3604. OPML import resolves each feed against the existing
        roster before creating anything -- `add_subscription` is a plain
        INSERT with no uniqueness constraint on `source`, so without this
        lookup a re-import duplicates every feed and the additive-only
        round-trip (ADR-043 rule 6) is impossible.

        Args:
            source: The exact source URL/identifier to match.

        Returns:
            The subscription's id, or `None` when no row carries it.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT id FROM subscriptions WHERE source = ?", (source,)
            ).fetchone()
        return int(row[0]) if row else None

    def accept_watchlist_run(
        self, source_id: int, *, created_at: str
    ) -> Dict[str, Any]:
        """Insert one queued source receipt or return its active winner."""
        with self.transaction(immediate=True) as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO local_watchlist_runs "
                    "(source_id, job_id, status, stats_json, created_at, updated_at) "
                    "VALUES (?, ?, 'queued', ?, ?, ?)",
                    (
                        source_id,
                        source_id,
                        json.dumps({"source_id": source_id}),
                        created_at,
                        created_at,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM local_watchlist_runs WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
                receipt = dict(row)
                receipt["_claim_acquired"] = True
                return receipt
            except sqlite3.IntegrityError as exc:
                if (
                    getattr(exc, "sqlite_errorcode", None)
                    != sqlite3.SQLITE_CONSTRAINT_UNIQUE
                ):
                    raise
                winner = conn.execute(
                    "SELECT * FROM local_watchlist_runs "
                    "WHERE source_id = ? AND status IN ('queued', 'running') "
                    "ORDER BY created_at DESC, id DESC LIMIT 1",
                    (source_id,),
                ).fetchone()
                if winner is not None:
                    receipt = dict(winner)
                    receipt["_claim_acquired"] = False
                    return receipt
                raise

    def transition_watchlist_run(
        self,
        run_id: int,
        *,
        status: str,
        finished_at: str,
        stats_json: str | None = None,
        error_msg: str | None = None,
        log_text: str | None = None,
    ) -> Dict[str, Any] | None:
        """Guardedly terminalize an active source receipt, releasing its claim."""
        if status in {"queued", "running"}:
            raise ValueError("Terminal run status required")
        with self.transaction() as conn:
            updated = conn.execute(
                "UPDATE local_watchlist_runs SET status = ?, finished_at = ?, "
                "stats_json = COALESCE(?, stats_json), error_msg = ?, log_text = ?, "
                "updated_at = ? WHERE id = ? AND status IN ('queued', 'running')",
                (
                    status,
                    finished_at,
                    stats_json,
                    error_msg,
                    log_text,
                    finished_at,
                    run_id,
                ),
            )
            if updated.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM local_watchlist_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def mark_watchlist_run_started(
        self, run_id: int, *, started_at: str
    ) -> Dict[str, Any] | None:
        """Guardedly move one queued receipt to running without releasing it."""
        with self.transaction() as conn:
            updated = conn.execute(
                "UPDATE local_watchlist_runs SET status = 'running', "
                "started_at = COALESCE(started_at, ?), updated_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (started_at, started_at, run_id),
            )
            if updated.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM local_watchlist_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def accept_briefing(
        self, watchlist_id: int, *, created_at: str
    ) -> Dict[str, Any]:
        """Insert one generating briefing or return its durable active winner."""
        with self.transaction(immediate=True) as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO briefings "
                    "(watchlist_id, status, created_at, updated_at) "
                    "VALUES (?, 'generating', ?, ?)",
                    (watchlist_id, created_at, created_at),
                )
                row = conn.execute(
                    "SELECT * FROM briefings WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                receipt = dict(row)
                receipt["_claim_acquired"] = True
                return receipt
            except sqlite3.IntegrityError as exc:
                if (
                    getattr(exc, "sqlite_errorcode", None)
                    != sqlite3.SQLITE_CONSTRAINT_UNIQUE
                ):
                    raise
                winner = conn.execute(
                    "SELECT * FROM briefings "
                    "WHERE watchlist_id = ? AND status = 'generating' "
                    "ORDER BY created_at DESC, id DESC LIMIT 1",
                    (watchlist_id,),
                ).fetchone()
                if winner is not None:
                    receipt = dict(winner)
                    receipt["_claim_acquired"] = False
                    return receipt
                raise

    def transition_briefing(
        self, briefing_id: int, *, status: str, error: str | None = None, **fields: Any
    ) -> Dict[str, Any] | None:
        """Guardedly terminalize one generating briefing, releasing its claim."""
        if status == "generating":
            raise ValueError("Terminal briefing status required")
        allowed_fields = {
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
        }
        for key in fields:
            if key not in allowed_fields or not validate_identifier(key, "column name"):
                raise ValueError(f"transition_briefing: invalid field {key!r}")
        assignments = ["status = ?", "error = ?"]
        values: list[Any] = [status, error]
        assignments.extend(f"{key} = ?" for key in fields)
        values.extend(fields.values())
        if "updated_at" not in fields:
            assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(briefing_id)
        with self.transaction() as conn:
            updated = conn.execute(
                f"UPDATE briefings SET {', '.join(assignments)} "
                "WHERE id = ? AND status = 'generating'",
                values,
            )
            if updated.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM briefings WHERE id = ?", (briefing_id,)
            ).fetchone()
            return dict(row) if row is not None else None

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

    def complete_briefing(
        self,
        briefing_id: int,
        *,
        body_markdown: str,
        model_used: str,
        covers_through_item_id: int | None,
        covers_from_ts: str | None,
        selection_mode: str,
        preset_id: int | None,
        overflow_count: int,
        provenance: Sequence[BriefingProvenanceRow],
    ) -> Dict[str, Any]:
        """Atomically snapshot provenance and publish one completed briefing."""
        with self.transaction() as conn:
            for row in provenance:
                live_item = conn.execute(
                    "SELECT id FROM subscription_items WHERE id = ?", (row.item_id,)
                ).fetchone()
                conn.execute(
                    "INSERT INTO briefing_items "
                    "(briefing_id, item_id, live_item_id, selection_position, "
                    "citation_position, featured, cited, item_title, item_url, "
                    "item_published_date, item_created_at, item_effective_date, "
                    "source_id, source_name, "
                    "source_type, source_url, provenance_version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 2)",
                    (
                        briefing_id,
                        row.item_id,
                        row.item_id if live_item is not None else None,
                        row.selection_position,
                        row.citation_position,
                        int(row.featured),
                        int(row.cited),
                        row.item_title,
                        _sanitize_provenance_url(row.item_url),
                        row.item_published_date,
                        row.item_created_at,
                        row.item_effective_date,
                        row.source_id,
                        row.source_name,
                        row.source_type,
                        _sanitize_provenance_url(row.source_url),
                    ),
                )
            updated = conn.execute(
                "UPDATE briefings SET status = 'complete', error = NULL, "
                "body_markdown = ?, model_used = ?, covers_through_item_id = ?, "
                "covers_from_ts = ?, selection_mode = ?, preset_id = ?, "
                "item_count = ?, featured_count = ?, overflow_count = ?, "
                "updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status = 'generating'",
                (
                    body_markdown,
                    model_used,
                    covers_through_item_id,
                    covers_from_ts,
                    selection_mode,
                    preset_id,
                    len(provenance),
                    sum(row.featured for row in provenance),
                    overflow_count,
                    briefing_id,
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("Briefing is not generating")
            published = conn.execute(
                "SELECT * FROM briefings WHERE id = ?", (briefing_id,)
            ).fetchone()
            return dict(published)

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
        or neither. (`list_briefing_schedules` also returns a second,
        deliberately status-blind `last_attempt_at` column for retry-cadence
        purposes -- that one is NOT part of this pact; see its own
        docstring.)

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

        `last_attempt_at`, in contrast, is deliberately status-blind: MAX
        `created_at` over that watchlist's briefings of ANY status,
        `failed`/`generating` included. It is NOT part of the
        `last_completed_at` pact above and must never gain a status
        filter -- the whole-branch review that added it found that a
        projection using `last_completed_at` alone made a FAILED schedule
        perpetually due (`next_run_at` frozen at the last success, so
        every ~30-minute queue reload re-emitted it, uncapped). The
        scheduler projection combines both columns (latest of the two)
        so a failed attempt defers the next retry by one cadence period
        instead of leaving it stuck in the past.

        Returns:
            A list of dicts, one per watchlist with
            `briefing_cadence_seconds IS NOT NULL`, each with keys
            `watchlist_id`, `name`, `briefing_cadence_seconds`,
            `last_completed_at` (the max `created_at` among that
            watchlist's `complete`/`empty` briefings, or `None` if it has
            never completed one), and `last_attempt_at` (the max
            `created_at` among ALL of that watchlist's briefings
            regardless of status, or `None` if it has never had one).
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
                    ) AS last_completed_at,
                    (
                        SELECT MAX(b.created_at)
                        FROM briefings AS b
                        WHERE b.watchlist_id = w.id
                    ) AS last_attempt_at
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
        error: str | None = None,
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
            error: Optional terminal error text, written in the SAME insert.
                For a create-and-immediately-fail path (a preflight that
                refuses before any synthesis happens, e.g. an unresolvable
                voice or a missing pydub), passing `status="failed"` +
                `error=...` here writes the finished row atomically, so
                there is no create-then-separate-update window a crash could
                land in (TASK-1718). The long-running synthesis path leaves
                this `None` and finalizes later via `update_briefing_audio` --
                that separation is inherent (see the module docstring).

        Returns:
            The new row's `id`.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO briefing_audio "
                "(script_id, voice_snapshot_json, status, error) "
                "VALUES (?, ?, ?, ?)",
                (script_id, voice_snapshot_json, status, error),
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
          `NULL` means never); passing a positive `int` sets the cadence.
          Must be a genuine `int`, not merely `int`-like: `bool` is a
          subclass of `int` in Python, so without an explicit
          `isinstance(..., bool)` exclusion `True` would silently pass as
          a cadence of one second -- a runaway schedule -- rather than
          being rejected (Qodo review).

        Args:
            watchlist_id: `watchlists.id` of the row to update.
            selection_mode: One of `("auto", "curated", "auto_featured")`,
                or `None` to leave the current value alone.
            default_preset_id: A `briefing_presets.id`, `None` to clear, or
                the `_UNSET` sentinel (default) to leave the current value
                alone.
            briefing_cadence_seconds: A positive `int` number of seconds
                between scheduled briefings, `None` to clear (never
                scheduled), or the `_UNSET` sentinel (default) to leave
                the current value alone.

        Returns:
            None.

        Raises:
            ValueError: If `selection_mode` is given and is not one of the
                valid modes, or if `briefing_cadence_seconds` is given and
                is not a positive `int` (a `bool`, a numeric string, a
                `float`, or a non-positive value all raise).
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
            if briefing_cadence_seconds is not None and (
                not isinstance(briefing_cadence_seconds, int)
                or isinstance(briefing_cadence_seconds, bool)
                or briefing_cadence_seconds <= 0
            ):
                raise ValueError(
                    f"set_watchlist_briefing_settings: briefing_cadence_seconds "
                    f"must be a positive int number of seconds or None (never); "
                    f"got {briefing_cadence_seconds!r}"
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
        """Close THIS thread's connection, checkpointing the `-wal` first.

        Scope is unchanged from before task-19562 and is load-bearing: this
        closes only the calling thread's connection and clears that thread's
        slot, and the `conn` property reopens lazily, which is what makes it
        safe for `app.py`'s FTS backfill to call it from a *pooled* thread
        that will later serve other watchlists work. Do not "improve" it into
        a close of the instance.

        What is new is that the connection is checkpointed on the way out
        (task-19562) and dropped from `_connections`, so the registry never
        reports a connection that is already gone. Use
        `close_all_connections` for the shutdown path, which adds the
        database-wide settle.

        The thread-exit cleanup is detached first (review of PR #1964): this
        thread has closed and de-registered its own connection, so the
        finalizer has nothing left to do and must not act on a connection this
        thread may since have reopened.
        """
        cleanup = getattr(self._local, "connection_cleanup", None)
        if cleanup is not None:
            cleanup.detach()
            self._local.connection_cleanup = None
        connection = getattr(self._local, "conn", None)
        if connection:
            try:
                if not self.is_memory_db and not connection.in_transaction:
                    mode = connection.execute("PRAGMA journal_mode;").fetchone()
                    if mode and str(mode[0]).lower() == "wal":
                        connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.Error as exc:
                if not _INTERPRETER_EXITING:
                    logger.warning(
                        f"WAL checkpoint before close failed for "
                        f"{self.db_path_str}: {exc}"
                    )
            connection.close()
            self._local.conn = None
        with self._connections_lock:
            self._connections.pop(threading.get_ident(), None)


# End of Subscriptions_DB.py
