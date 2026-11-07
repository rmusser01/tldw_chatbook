
====================================================================================================
# _get_connection: 20 defs, 13 distinct bodies, 13 distinct shapes

--- body 18266155 (shape 932453fe): 6 copies (core 6, interop 0)
   files: Scheduling/db/migrations/v1_to_v2.py:26, Scheduling/db/migrations/v2_to_v3.py:25, Scheduling/db/migrations/v3_to_v4.py:20, Scheduling/db/migrations/v4_to_v5.py:21, Scheduling/db/migrations/v5_to_v6.py:19, Scheduling/db/migrations/v6_to_v7.py:26
   | def _get_connection(self) -> Any: ...

--- body 483e53da (shape a925f34f): 3 copies (core 2, interop 1)
   files: Notifications/client_notifications_db.py:106, Notifications/event_state_repository.py:176, Sync_Interop/sync_state_repository.py:134
   | def _get_connection(self) -> sqlite3.Connection:
   |         self._ensure_schema()
   |         return self._open_connection()

--- body c0f768e7 (shape fe15f4a8): 1 copies (core 1, interop 0)
   files: DB/AgentRuns_DB.py:297
   | def _get_connection(self) -> sqlite3.Connection:
   |         _core_access(self)
   |         conn = super()._get_connection()
   |         _register_core_connection(self, conn)
   |         try:
   |             _core_access(self)
   |             conn.execute("PRAGMA foreign_keys = ON")
   |             # busy_timeout FIRST: the journal_mode=WAL conversion below is the
   |             # one PRAGMA here that can itself contend (switching a rollback-
   |             # journal file to WAL briefly needs an exclusive lock), so it must
   |             # not run while busy_timeout is still 0 -- a contended cross-process
   |             # first conversion would otherwise raise 'database is locked'
   |             # immediately instead of waiting. busy_timeout is harmless to set
   |             # for in-memory DBs too, so it's unconditional (kept for
   |             # uniformity); WAL itself is unavailable for in-memory DBs, so that
   |             # one stays guarded on is_memory_db.
   |             conn.execute("PRAGMA busy_timeout = 5000")
   |             if not self.is_memory_db:
   |                 conn.execute("PRAGMA journal_mode = WAL")
   |             # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
   |             # lose the last commit or two, acceptable for this local agent-run
   |             # ledger) and avoids an fsync on every commit -- the default FULL was
   |             # fsyncing the WAL on every commit despite WAL already being enabled,
   |             # on a per-agent-step persistence path. See Library_Ingest_Jobs_DB.py:
   |             # 57-61 for the original template (task-15465).
   |             conn.execute("PRAGMA synchronous = NORMAL")
   |             conn.row_factory = sqlite3.Row
   |             # task-3012: the held (long-lived) connection needs true autocommit.
   |    ...

--- body ba6db78e (shape 881ef21f): 1 copies (core 1, interop 0)
   files: DB/Evals_DB.py:196
   | def _get_connection(self) -> sqlite3.Connection:
   |         """Get thread-local database connection.

   |         task-22224 EXCEPTION -- this held connection deliberately keeps the
   |         legacy default isolation level instead of ``isolation_level = None``
   |         (the held-connection rule in ``Library_Ingest_Jobs_DB.py``'s module
   |         docstring, the store template). Every write path in this file relies
   |         on Python's implicit transactions via ``with conn:`` bodies, several
   |         of them multi-statement (e.g. ``store_result``'s result INSERT plus
   |         its completed-samples UPDATE, and ``delete_task``, whose cascade into
   |         ``delete_probe_annotations_for_run_groups`` deliberately NESTS
   |         ``with conn:`` blocks to share one implicit transaction -- explicit
   |         BEGIN cannot nest); there is no explicit-BEGIN transaction
   |         manager here, so flipping to autocommit would silently strip their
   |         atomicity. The degradation this store risks instead is bounded: no
   |         code path issues an explicit BEGIN on this connection, so the
   |         borrow/"cannot start a transaction" failure modes cannot fire.
   |         Converting this store to the template idiom means giving it an
   |         explicit-BEGIN manager and auditing all ~20 ``with conn:`` writes
   |         (including un-nesting the nested pair) -- do that as its own task,
   |         and do NOT copy this store's pattern into new code.
   |         """
   |         _core_access(self)
   |         conn = _core_cached_connection(self, getattr(self._local, "connection", None))
   |         if conn is None:
   |             conn = connect_private_sqlite("db.evals", self.db_path, check_same_thread=False)
   |             try:
   |                 _register_core_connection(self, conn)
   |    ...

--- body a296a4c7 (shape f99fc445): 1 copies (core 1, interop 0)
   files: DB/Library_Collections_DB.py:498
   | def _get_connection(self) -> sqlite3.Connection:
   |         from tldw_chatbook.Backup_Recovery.participants import _core_access

   |         _core_access(self)
   |         conn = super()._get_connection()
   |         _register_core_connection(self, conn)
   |         try:
   |             _core_access(self)
   |         except BaseException:
   |             conn.close()
   |             raise
   |         conn.execute("PRAGMA foreign_keys = ON")
   |         if not self.is_memory_db:
   |             self._enable_wal(conn)
   |         # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
   |         # lose the last commit, acceptable for this local collections cache)
   |         # and avoids an fsync per commit. Unlike journal_mode, which is
   |         # persisted in the file, synchronous is per-connection -- so it must
   |         # be re-applied here on every NEW connection, which is exactly why
   |         # this pairing lives in the one place connections are created
   |         # (task-15465).
   |         conn.execute("PRAGMA synchronous = NORMAL")
   |         # task-3012: a held (long-lived) connection needs true autocommit.
   |         # Python's default isolation mode auto-BEGINs on any DML, and an
   |         # implicit transaction accumulated outside `transaction()` makes the
   |         # explicit BEGIN there fail with "cannot start a transaction within a
   |         # transaction" -- and silently ROLLS BACK bare DML on close.
   |         # Audited (task-15466): every write in this file's own module and in
   |    ...

--- body 9bde84a7 (shape 7c33b192): 1 copies (core 1, interop 0)
   files: DB/Library_Ingest_Jobs_DB.py:84
   | def _get_connection(self) -> sqlite3.Connection:
   |         from tldw_chatbook.Backup_Recovery.participants import _core_access

   |         _core_access(self)
   |         self._conn = _core_cached_connection(self, self._conn)
   |         if self._conn is None:
   |             conn = connect_private_sqlite(
   |                 "db.library_ingest_jobs",
   |                 self.db_path_str,
   |                 check_same_thread=False,
   |             )
   |             _register_core_connection(self, conn)
   |             try:
   |                 _core_access(self)
   |                 conn.row_factory = sqlite3.Row
   |                 conn.execute("PRAGMA journal_mode=WAL")
   |                 # NORMAL is safe under WAL and avoids an fsync per commit.
   |                 conn.execute("PRAGMA synchronous=NORMAL")
   |                 conn.isolation_level = None
   |                 _core_access(self)
   |             except BaseException:
   |                 # Close this allocation, never a concurrently replaced cache.
   |                 conn.close()
   |                 raise
   |             self._conn = conn
   |         return self._conn

--- body 833921fd (shape ad672ee8): 1 copies (core 1, interop 0)
   files: DB/RAG_Indexing_DB.py:104
   | def _get_connection(self) -> sqlite3.Connection:
   |         """Open and configure a NEW database connection with row factory.

   |         Callers wanting the thread's long-lived connection should use
   |         ``connection``/``transaction``; this is the single place a
   |         connection is created, so every per-connection property lives here.
   |         """
   |         _core_access(self)
   |         conn = connect_private_sqlite("db.rag_indexing", self.db_path_str)
   |         _register_core_connection(self, conn)
   |         try:
   |             return self._configure_connection(conn)
   |         except BaseException:
   |             conn.close()
   |             raise

--- body c63b640e (shape c1c95897): 1 copies (core 1, interop 0)
   files: DB/Subscriptions_DB.py:614
   | def _get_connection(self) -> sqlite3.Connection:
   |         """Return a connection with foreign-key enforcement enabled.

   |         ``PRAGMA foreign_keys`` is per-connection and defaults to OFF, and
   |         ``BaseDB._get_connection`` sets only ``row_factory``. Without this
   |         override every ``ON DELETE CASCADE`` in this schema is inert, which
   |         silently orphaned ``subscription_items`` whenever a subscription was
   |         deleted. Matches ``ChaChaNotes_DB`` and ``Client_Media_DB_v2``, which
   |         each enable it per connection.

   |         task-22224 EXCEPTION -- connections here keep the legacy default
   |         isolation level for now instead of the store template's
   |         ``isolation_level = None`` (rule: ``Library_Ingest_Jobs_DB.py``
   |         module docstring). This file's write paths knowingly rely on the
   |         legacy implicit-BEGIN policy (see the long TASK-1362 comment above
   |         the extraction-fingerprint migration, which documents the reliance
   |         and works around its DDL gap with an explicit BEGIN IMMEDIATE), so
   |         flipping requires this file's own commit/write-site census first --
   |         its own task. Do NOT copy this pattern into new stores.
   |         """
   |         _core_access(self)
   |         if self._read_only:
   |             conn = connect_private_sqlite(
   |                 "db.subscriptions.agent_read", self.db_path_str,
   |                 read_only=True, must_exist=True,
   |             )
   |         else:
   |             conn = super()._get_connection()
   |    ...

--- body 5c641106 (shape 5737702d): 1 copies (core 1, interop 0)
   files: DB/Workspace_DB.py:353
   | def _get_connection(self) -> sqlite3.Connection:
   |         _core_access(self)
   |         conn = super()._get_connection()
   |         _register_core_connection(self, conn)
   |         try:
   |             _core_access(self)
   |             conn.execute("PRAGMA foreign_keys = ON")
   |             if not self.is_memory_db:
   |                 conn.execute("PRAGMA journal_mode = WAL")
   |             # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
   |             # lose the last commit, acceptable for this local registry cache) and
   |             # avoids an fsync per commit -- DELETE+FULL's writer-exclusive-locks-
   |             # readers behavior was a stall candidate on this held-connection,
   |             # query-heavy path (task-15465). Unconditional: synchronous is
   |             # per-connection, so every held connection needs it re-applied.
   |             conn.execute("PRAGMA synchronous = NORMAL")
   |             # task-3012 (missed at task-3011 port time; fixed at task-15480): a
   |             # held (long-lived) connection needs true autocommit. Python's
   |             # default isolation mode auto-BEGINs on any DML, and an implicit
   |             # transaction accumulated outside `transaction()` makes the explicit
   |             # `BEGIN` there fail with "cannot start a transaction within a
   |             # transaction" -- and silently ROLLS BACK bare DML on close (masked
   |             # pre-task-3011 by per-call connections, which committed
   |             # explicitly). Audited (task-15480): every `connection()` call site
   |             # in `Workspaces/registry_service.py` is read-only -- every write
   |             # there already goes through `transaction()` -- and this class's own
   |             # `connection()` sites (`_initialize_schema`'s `executescript`,
   |             # `get_schema_version`'s read) self-commit or don't write at all.
   |    ...

--- body e1490281 (shape 316c66c1): 1 copies (core 1, interop 0)
   files: DB/base_db.py:818
   | def _get_connection(self) -> sqlite3.Connection:
   |         """
   |         Get a database connection with row factory.
   |         Can be overridden by subclasses for custom connection handling.
   |         """
   |         conn = connect_private_sqlite("db.base", self.db_path_str)
   |         conn.row_factory = sqlite3.Row
   |         return conn

--- body 51446310 (shape 596c7138): 1 copies (core 1, interop 0)
   files: Notes/file_notes_replica.py:65
   | def _get_connection(self) -> sqlite3.Connection:
   |         _core_access(self)
   |         conn = _core_cached_connection(self, self._connection)
   |         if conn is None:
   |             conn = connect_private_sqlite(
   |                 "notes.file_notes_replica", self.db_path,
   |                 isolation_level=None, check_same_thread=False,
   |             )
   |             try:
   |                 _register_core_connection(self, conn)
   |                 _core_access(self)
   |                 conn.row_factory = sqlite3.Row
   |                 if not self.is_memory_db:
   |                     conn.execute("PRAGMA journal_mode = WAL")
   |                 conn.execute("PRAGMA synchronous = NORMAL")
   |                 _core_access(self)
   |             except BaseException:
   |                 conn.close()
   |                 raise
   |             self._connection = conn
   |         return conn

--- body 1d81e97a (shape 5fc0185c): 1 copies (core 1, interop 0)
   files: Notes/notes_device_state_store.py:684
   | def _get_connection(self) -> sqlite3.Connection:
   |         _core_access(self)
   |         connection = getattr(self._thread_local, "connection", None)
   |         if connection is not None and _core_cached_connection(self, connection) is None:
   |             self._thread_local.connection = None
   |             with self._connections_guard:
   |                 if connection in self._connections:
   |                     self._connections.remove(connection)
   |             connection = None
   |         if connection is not None:
   |             return connection
   |         connection = self._open_schema_ready_connection()
   |         self._thread_local.connection = connection
   |         with self._connections_guard:
   |             self._connections.append(connection)
   |         return connection

--- body 0e5ef3ce (shape 17818cb5): 1 copies (core 1, interop 0)
   files: Scheduling/db/scheduled_tasks_db.py:216
   | def _get_connection(self) -> sqlite3.Connection:
   |         """Open one fresh, caller-closed connection (see class usage).

   |         task-22224 EXCEPTION -- deliberately keeps the legacy default
   |         isolation level instead of ``isolation_level = None`` (the
   |         held-connection rule in ``Library_Ingest_Jobs_DB.py``'s module
   |         docstring, the store template). This store does not HOLD
   |         connections: every caller opens one here and closes it per
   |         operation (``closing(...)`` / ``transaction()``'s ``finally``), so
   |         an implicit transaction cannot leak across operations and nothing
   |         issues an explicit BEGIN outside migration scripts -- the
   |         degradation mechanism cannot fire. Write bodies rely on implicit
   |         transactions (``transaction()`` has no explicit BEGIN; migrations
   |         pair multi-statement spans with ``conn.commit()``), so flipping to
   |         autocommit here would strip their atomicity; converting means
   |         adding explicit BEGIN and auditing the ~22 ``transaction()`` bodies
   |         plus the ``Scheduling/db/migrations`` version stamps -- its own
   |         task. Do NOT copy this pattern into a store that holds connections.
   |         """
   |         _core_access(self)
   |         conn = super()._get_connection()
   |         _register_core_connection(self, conn)
   |         try:
   |             _core_access(self)
   |             if not self.is_memory_db:
   |                 conn.execute("PRAGMA journal_mode = WAL")
   |             # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
   |             # lose the last commit, acceptable for this local reminder/automation
   |    ...

====================================================================================================
# _initialize_schema: 16 defs, 16 distinct bodies, 15 distinct shapes

--- body 4113ed96 (shape 15318eb6): 1 copies (core 1, interop 0)
   files: DB/AgentRuns_DB.py:431
   | def _initialize_schema(self) -> None:
   |         with self.connection() as conn:
   |             conn.executescript(
   |                 """
   |                 PRAGMA foreign_keys = ON;

   |                 CREATE TABLE IF NOT EXISTS schema_version (
   |                     version INTEGER PRIMARY KEY NOT NULL
   |                 );
   |                 INSERT OR IGNORE INTO schema_version (version) VALUES (4);

   |                 CREATE TABLE IF NOT EXISTS agent_runs (
   |                     id TEXT PRIMARY KEY,
   |                     conversation_id TEXT NOT NULL,
   |                     parent_run_id TEXT,
   |                     agent_kind TEXT NOT NULL,
   |                     task TEXT,
   |                     status TEXT NOT NULL,
   |                     steps TEXT NOT NULL DEFAULT '[]',
   |                     result TEXT,
   |                     budget TEXT,
   |                     created_at TEXT NOT NULL,
   |                     updated_at TEXT NOT NULL,
   |                     assistant_message_id TEXT,
   |                     agent_definition TEXT,
   |                     definition_fingerprint TEXT,
   |                     wake_delivered_at TEXT,
   |                     -- v11 (fleet PR3b Task 4, spec SS6): when this run is
   |    ...

--- body 616b4aa9 (shape db38730b): 1 copies (core 1, interop 0)
   files: DB/ChaChaNotes_DB.py:8421
   | def _initialize_schema(self):
   |         """
   |         Initializes or migrates the database schema to `_CURRENT_SCHEMA_VERSION`.

   |         Checks the existing schema version.
   |         - If 0 (new DB): Applies the full current schema (`_apply_schema_v4`).
   |         - If current: Logs that schema is up to date.
   |         - If older: Raises SchemaError (migration paths not yet implemented beyond initial creation).
   |         - If newer: Raises SchemaError (database is newer than code supports).

   |         This method is called during `CharactersRAGDB` instantiation.
   |         Operations are performed within a transaction.

   |         Raises:
   |             SchemaError: If the database schema version is newer than supported by the code,
   |                          if a migration path is undefined for an older schema version,
   |                          or if any step in schema application/migration fails.
   |             CharactersRAGDBError: For unexpected errors during schema initialization.
   |         """
   |         conn = self.get_connection()
   |         current_initial_version = 0
   |         try:
   |             with TransactionContextManager(
   |                 self, immediate=True
   |             ):  # Ensures atomicity for schema changes
   |                 current_db_version = self._get_db_version(conn)
   |                 current_initial_version = (
   |                     current_db_version  # Store initial for messages
   |    ...

--- body 899e7b40 (shape 2dd9eff4): 1 copies (core 1, interop 0)
   files: DB/Client_Media_DB_v2.py:2132
   | def _initialize_schema(self):
   |         """Checks schema version and applies initial schema or migrations."""
   |         conn = self.get_connection()
   |         try:
   |             current_db_version = self._get_db_version(conn)
   |             target_version = self._CURRENT_SCHEMA_VERSION

   |             logging.info(
   |                 f"Checking DB schema. Current version: {current_db_version}. Code supports: {target_version}"
   |             )

   |             if current_db_version == target_version:
   |                 logging.debug("Database schema is up to date.")
   |                 # Ensure FTS and the read-it-later table exist even if schema version matches
   |                 try:
   |                     conn.executescript(
   |                         f"{self._FTS_TABLES_SQL}\n{self._LOCAL_ONLY_TABLES_SQL}"
   |                     )
   |                     conn.commit()
   |                     logging.debug("Verified FTS and read-it-later tables exist.")
   |                 except sqlite3.Error as fts_err:
   |                     logging.warning(
   |                         f"Could not verify/create FTS or read-it-later tables on already correct schema version: {fts_err}"
   |                     )
   |                 return

   |             if current_db_version > target_version:
   |                 raise SchemaError(
   |    ...

--- body 2dba23e5 (shape eafe5fa3): 1 copies (core 1, interop 0)
   files: DB/Library_Collections_DB.py:696
   | def _initialize_schema(self) -> None:
   |         """Atomically initialize or migrate the local Collections schema."""
   |         with self.transaction() as conn:
   |             has_version_table = (
   |                 conn.execute(
   |                     "SELECT 1 FROM sqlite_schema "
   |                     "WHERE type = 'table' AND name = 'schema_version'"
   |                 ).fetchone()
   |                 is not None
   |             )
   |             current_version = 0
   |             if has_version_table:
   |                 row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
   |                 current_version = int(row[0] or 0) if row is not None else 0
   |             if current_version > self._CURRENT_SCHEMA_VERSION:
   |                 raise LibraryCollectionsSchemaError("schema_too_new")

   |             if current_version == 0:
   |                 for statement in self._LEGACY_SCHEMA_DDL:
   |                     conn.execute(statement)
   |                 conn.execute(
   |                     "INSERT OR IGNORE INTO schema_version (version) VALUES (1)"
   |                 )

   |             for statement in self._CAPTURE_SCHEMA_DDL:
   |                 conn.execute(statement)
   |             if current_version == 2:
   |                 columns = {
   |    ...

--- body cd890e7e (shape 1ab1a6cb): 1 copies (core 1, interop 0)
   files: DB/Library_Ingest_Jobs_DB.py:216
   | def _initialize_schema(self) -> None:
   |         conn = self._get_connection()
   |         conn.executescript(
   |             """
   |             CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY NOT NULL);
   |             INSERT OR IGNORE INTO schema_version (version) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM schema_version);

   |             CREATE TABLE IF NOT EXISTS ingest_jobs (
   |                 seq INTEGER PRIMARY KEY,
   |                 job_id TEXT UNIQUE NOT NULL,
   |                 source_path TEXT NOT NULL,
   |                 title TEXT NOT NULL DEFAULT '',
   |                 author TEXT NOT NULL DEFAULT '',
   |                 keywords TEXT NOT NULL DEFAULT '[]',
   |                 perform_analysis INTEGER NOT NULL DEFAULT 0,
   |                 chunk_enabled INTEGER NOT NULL DEFAULT 0,
   |                 chunk_size INTEGER NOT NULL DEFAULT 0,
   |                 state TEXT NOT NULL CHECK (state IN ('queued','parsing','writing','done','failed','cancelled')),
   |                 retry_count INTEGER NOT NULL DEFAULT 0,
   |                 detected_type TEXT NOT NULL DEFAULT '',
   |                 error TEXT NOT NULL DEFAULT '',
   |                 finished_at_wall TEXT NOT NULL DEFAULT '',
   |                 media_id INTEGER,
   |                 superseded INTEGER NOT NULL DEFAULT 0,
   |                 dismissed INTEGER NOT NULL DEFAULT 0,
   |                 permanent INTEGER NOT NULL DEFAULT 0,
   |                 ingest_options TEXT DEFAULT '{}',
   |                 error_detail TEXT DEFAULT NULL,
   |    ...

--- body 8cb8c0bd (shape 6101bf8b): 1 copies (core 1, interop 0)
   files: DB/Prompts_DB.py:945
   | def _initialize_schema(self):
   |         conn = self.get_connection()
   |         try:
   |             current_db_version = self._get_db_version(conn)
   |             target_version = self._CURRENT_SCHEMA_VERSION
   |             logging.info(
   |                 f"Checking DB schema. Current: {current_db_version}, Code supports: {target_version}"
   |             )

   |             if current_db_version == target_version:
   |                 logging.debug("Database schema is up to date.")
   |                 try:  # Ensure FTS tables exist
   |                     conn.executescript(self._FTS_TABLES_SQL)
   |                     conn.commit()
   |                     logging.debug("Verified FTS tables exist.")
   |                 except sqlite3.Error as fts_err:
   |                     logging.warning(
   |                         f"Could not verify/create FTS tables on correct schema: {fts_err}"
   |                     )
   |                 return

   |             if current_db_version > target_version:
   |                 raise SchemaError(
   |                     f"DB schema version ({current_db_version}) is newer than supported ({target_version})."
   |                 )

   |             while current_db_version < target_version:
   |                 migration = self._MIGRATIONS.get(current_db_version)
   |    ...

--- body 74b1da10 (shape e4044d34): 1 copies (core 1, interop 0)
   files: DB/RAG_Indexing_DB.py:255
   | def _initialize_schema(self):
   |         """Initialize the database schema."""
   |         schema = """
   |         CREATE TABLE IF NOT EXISTS indexed_items (
   |             item_id TEXT NOT NULL,
   |             item_type TEXT NOT NULL,
   |             last_indexed DATETIME NOT NULL,
   |             last_modified DATETIME NOT NULL,
   |             chunk_count INTEGER DEFAULT 0,
   |             metadata TEXT,
   |             PRIMARY KEY (item_id, item_type)
   |         );
        
   |         CREATE INDEX IF NOT EXISTS idx_indexed_items_type 
   |         ON indexed_items(item_type);
        
   |         CREATE INDEX IF NOT EXISTS idx_indexed_items_modified 
   |         ON indexed_items(last_modified);
        
   |         CREATE INDEX IF NOT EXISTS idx_indexed_items_indexed 
   |         ON indexed_items(last_indexed);
        
   |         -- Table for tracking collection states
   |         CREATE TABLE IF NOT EXISTS collection_state (
   |             collection_name TEXT PRIMARY KEY,
   |             last_full_index DATETIME,
   |             total_items INTEGER DEFAULT 0,
   |             indexed_items INTEGER DEFAULT 0,
   |    ...

--- body 3265c168 (shape 8b60ce6d): 1 copies (core 1, interop 0)
   files: DB/Subscriptions_DB.py:703
   | def _initialize_schema(self):
   |         """Initialize the database schema.

   |         Runs on ``self.conn`` (the thread-local connection everything else on
   |         this thread reuses) rather than a throwaway connection that used to
   |         get closed immediately afterwards. For a file-backed database both
   |         approaches land on the same file, so it made no observable
   |         difference. For ``:memory:``, every ``sqlite3.connect(':memory:')``
   |         call opens a brand-new, private, empty database -- so the old
   |         close-then-reopen sequence built the schema somewhere the rest of
   |         the class could never see, leaving ``.conn`` pointed at zero tables.
   |         Matches the pattern ``ChaChaNotes_DB._initialize_schema`` already
   |         uses (``self.get_connection()``) for the same reason.

   |         Trade-off carried over from that same precedent: this only makes the
   |         constructing thread's connection schema-bearing. If a *second*
   |         thread later touches ``.conn`` on this same in-memory instance, its
   |         own thread-local slot is empty, so the ``.conn`` property lazily
   |         opens yet another private ``:memory:`` connection with no schema --
   |         identical to the limitation already accepted in ``ChaChaNotes_DB``.
   |         The only current ``:memory:`` caller, ``WatchlistPreviewService.
   |         preview()``, constructs, uses, and discards its instance within a
   |         single coroutine on one thread (it is scheduled via
   |         ``run_worker(coroutine)``, not ``thread=True``), so this does not
   |         apply there today; a future caller that hands an in-memory
   |         ``SubscriptionsDB`` across a thread boundary would need a different
   |         fix (e.g. a shared-cache ``file::memory:?cache=shared`` URI plus a
   |         dedicated keepalive connection).
   |    ...

--- body c8f2dac8 (shape d5dda264): 1 copies (core 1, interop 0)
   files: DB/Workspace_DB.py:478
   | def _initialize_schema(self) -> None:
   |         """Initialize the local workspace registry schema."""

   |         with self.connection() as conn:
   |             conn.executescript(
   |                 """
   |                 PRAGMA foreign_keys = ON;

   |                 CREATE TABLE IF NOT EXISTS schema_version (
   |                     version INTEGER PRIMARY KEY NOT NULL
   |                 );
   |                 INSERT OR IGNORE INTO schema_version (version) VALUES (1);

   |                 CREATE TABLE IF NOT EXISTS workspace_records (
   |                     workspace_id TEXT PRIMARY KEY,
   |                     name TEXT NOT NULL,
   |                     description TEXT NOT NULL DEFAULT '',
   |                     authority TEXT NOT NULL,
   |                     sync_status TEXT NOT NULL,
   |                     active INTEGER NOT NULL DEFAULT 0,
   |                     archived INTEGER NOT NULL DEFAULT 0,
   |                     created_at TEXT NOT NULL,
   |                     updated_at TEXT NOT NULL
   |                 );

   |                 CREATE TABLE IF NOT EXISTS workspace_memberships (
   |                     membership_id TEXT PRIMARY KEY,
   |                     workspace_id TEXT NOT NULL,
   |    ...

--- body 47e6fa4a (shape 47e6fa4a): 1 copies (core 1, interop 0)
   files: DB/base_db.py:811
   | def _initialize_schema(self):
   |         """
   |         Initialize the database schema.
   |         Must be implemented by subclasses.
   |         """
   |         pass

--- body dbd7c990 (shape bc722bb9): 1 copies (core 1, interop 0)
   files: Notes/file_notes_replica.py:630
   | def _initialize_schema(self) -> None:
   |         with self._locked_connection():
   |             self._connection.executescript(
   |                 """
   |                 CREATE TABLE IF NOT EXISTS files (
   |                     root TEXT NOT NULL,
   |                     relative_path TEXT NOT NULL,
   |                     raw_bytes BLOB NOT NULL,
   |                     content_hash TEXT NOT NULL,
   |                     decoded_text TEXT,
   |                     size INTEGER NOT NULL,
   |                     mtime_ns INTEGER NOT NULL,
   |                     deleted_at TEXT,
   |                     UNIQUE(root, relative_path)
   |                 );

   |                 CREATE TABLE IF NOT EXISTS revisions (
   |                     root TEXT NOT NULL,
   |                     relative_path TEXT NOT NULL,
   |                     raw_bytes BLOB NOT NULL,
   |                     content_hash TEXT NOT NULL,
   |                     kind TEXT NOT NULL,
   |                     session_key TEXT,
   |                     created_at TEXT NOT NULL,
   |                     UNIQUE(root, relative_path, kind, session_key)
   |                 );

   |                 CREATE TABLE IF NOT EXISTS protected_paths (
   |    ...

--- body 4fd44289 (shape 077d7697): 1 copies (core 1, interop 0)
   files: Notifications/client_notifications_db.py:257
   | def _initialize_schema(self) -> None:
   |         # Raw connection: runs under _ensure_schema's lock (TASK-21105), so
   |         # it cannot use connection()/_held_connection (both re-enter
   |         # _get_connection). File-backed: one short-lived connection, closed
   |         # below; the held per-thread connection opens on the first real
   |         # operation. :memory:: the shared cached connection, never closed.
   |         conn = self._open_connection()
   |         try:
   |             conn.executescript(
   |                 """
   |                 PRAGMA foreign_keys = ON;

   |                 CREATE TABLE IF NOT EXISTS schema_version (
   |                     version INTEGER PRIMARY KEY NOT NULL
   |                 );
   |                 INSERT OR IGNORE INTO schema_version (version) VALUES (1);

   |                 CREATE TABLE IF NOT EXISTS client_notifications (
   |                     id INTEGER PRIMARY KEY AUTOINCREMENT,
   |                     category TEXT NOT NULL,
   |                     title TEXT NOT NULL,
   |                     message TEXT NOT NULL,
   |                     severity TEXT NOT NULL DEFAULT 'information',
   |                     source_backend TEXT,
   |                     source_entity_kind TEXT,
   |                     source_entity_id TEXT,
   |                     payload TEXT NOT NULL DEFAULT '{}',
   |                     is_read INTEGER NOT NULL DEFAULT 0,
   |    ...

--- body d1d33e5e (shape 077d7697): 1 copies (core 1, interop 0)
   files: Notifications/event_state_repository.py:423
   | def _initialize_schema(self) -> None:
   |         # Raw connection: runs under _ensure_schema's lock (TASK-21105), so
   |         # it cannot use connection()/_held_connection (both re-enter
   |         # _get_connection). File-backed: one short-lived connection, closed
   |         # below; the held per-thread connection opens on the first real
   |         # operation. :memory:: the shared cached connection, never closed.
   |         conn = self._open_connection()
   |         try:
   |             conn.executescript(
   |                 """
   |                 PRAGMA foreign_keys = ON;

   |                 CREATE TABLE IF NOT EXISTS schema_version (
   |                     version INTEGER PRIMARY KEY NOT NULL
   |                 );
   |                 INSERT OR IGNORE INTO schema_version (version) VALUES (1);

   |                 CREATE TABLE IF NOT EXISTS event_records (
   |                     id INTEGER PRIMARY KEY AUTOINCREMENT,
   |                     event_key TEXT NOT NULL UNIQUE,
   |                     dedupe_key TEXT NOT NULL UNIQUE,
   |                     source_authority TEXT NOT NULL,
   |                     server_profile_id TEXT,
   |                     authenticated_principal_id TEXT,
   |                     stream_name TEXT NOT NULL,
   |                     stream_instance_id TEXT NOT NULL,
   |                     event_kind TEXT NOT NULL,
   |                     entity_ref TEXT NOT NULL,
   |    ...

--- body 96a195dc (shape 6be4bc96): 1 copies (core 1, interop 0)
   files: Personal_Context/repository.py:631
   | def _initialize_schema(self, connection: sqlite3.Connection) -> None:
   |         for statement in _SCHEMA_STATEMENTS:
   |             connection.execute(statement)
   |         connection.execute(
   |             "INSERT INTO personal_context_schema(singleton, version) VALUES (1, ?)",
   |             (SCHEMA_VERSION,),
   |         )

--- body d699c617 (shape e5ed1450): 1 copies (core 1, interop 0)
   files: Scheduling/db/scheduled_tasks_db.py:263
   | def _initialize_schema(self) -> None:
   |         """Create tables, indexes, and schema version row, migrating forward.

   |         Each migration checks its own applicability on the connection it
   |         will migrate. For a ``:memory:`` database every
   |         ``_get_connection()`` is a fresh empty database -- a version check
   |         done on one connection tells nothing about the next -- so the chain
   |         must not consult ``get_schema_version()`` between migrations the
   |         way a file-backed database could. The migrations themselves detect
   |         their condition structurally (the presence of their column), which
   |         is memory-correct: an empty memory database runs v0_to_v1, finds
   |         no ``missed_count``, adds it, finds no ``timeout_seconds``, adds
   |         it, finds no ``automation_runs``/``automation_results`` tables,
   |         adds those and the v4 columns, and every step lands on a
   |         consistent v4 schema even though each step sees its own
   |         connection.
   |         """
   |         if self._schema_is_current():
   |             # Warm-boot fast path (ADR-097 boot ratchet): a fully-migrated
   |             # file DB skips importing any of the migration modules entirely,
   |             # keeping them out of the `_ui_ready` module census on every
   |             # boot after the first. Only a PROOF of completeness skips --
   |             # any probe failure (missing table on a fresh or `:memory:`
   |             # per-connection DB, an older recorded version) falls through
   |             # to the full chain below, whose idempotence remains the
   |             # correctness backstop.
   |             return

   |    ...

--- body 7b7623b2 (shape 3d960595): 1 copies (core 0, interop 1)
   files: Sync_Interop/sync_state_repository.py:221
   | def _initialize_schema(self) -> None:
   |         # Raw connection: runs under _ensure_schema's lock (TASK-21105).
   |         # File-backed: one short-lived connection, closed below (:memory:
   |         # keeps its shared cached connection open). The inner ``with conn``
   |         # transaction block is load-bearing: _record_schema_version runs
   |         # bare DML whose implicit transaction it commits -- closing without
   |         # it would roll the version stamp back.
   |         conn = self._open_connection()
   |         try:
   |             with conn:
   |                 conn.executescript(
   |                     """
   |                 PRAGMA foreign_keys = ON;

   |                 CREATE TABLE IF NOT EXISTS schema_version (
   |                     version INTEGER PRIMARY KEY NOT NULL
   |                 );
   |                 INSERT OR IGNORE INTO schema_version (version) VALUES (5);

   |                 CREATE TABLE IF NOT EXISTS sync_identity_mappings (
   |                     mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
   |                     source_scope_key TEXT NOT NULL,
   |                     local_side_key TEXT,
   |                     remote_side_key TEXT,
   |                     source_authority TEXT NOT NULL,
   |                     server_profile_id TEXT,
   |                     authenticated_principal_id TEXT,
   |                     workspace_scope TEXT,
   |    ...

====================================================================================================
# _held_connection: 6 defs, 4 distinct bodies, 4 distinct shapes

--- body ef683408 (shape 2eb5feca): 3 copies (core 3, interop 0)
   files: DB/AgentRuns_DB.py:341, DB/RAG_Indexing_DB.py:155, DB/Workspace_DB.py:392
   | def _held_connection(self) -> sqlite3.Connection:
   |         """Return this thread's held connection, opening or reviving it.

   |         task-3012: mirrors ``WorkspaceDB._held_connection`` (itself the
   |         ChaChaNotes idiom). Every per-connection property this DB relies on
   |         — WAL, busy_timeout, foreign keys, row factory — is applied by
   |         ``_get_connection`` when the held connection is (re)opened.
   |         """
   |         _core_access(self)
   |         conn = getattr(self._thread_local, "conn", None)
   |         conn = _core_cached_connection(self, conn)
   |         if conn is not None:
   |             last_used = getattr(self._thread_local, "conn_last_used", None)
   |             if (
   |                 last_used is None
   |                 or (time.monotonic() - last_used) >= self._LIVENESS_PING_IDLE_SECONDS
   |             ):
   |                 try:
   |                     conn.execute("SELECT 1")
   |                 except (sqlite3.ProgrammingError, sqlite3.OperationalError):
   |                     # A failed probe does not retire a live native borrower.
   |                     try:
   |                         sqlite3.Connection.in_transaction.__get__(conn)
   |                     except sqlite3.ProgrammingError:
   |                         conn.close()  # Positively closed; failure retains cache.
   |                     else:
   |                         raise
   |                     conn = None
   |    ...

--- body 75f58473 (shape dbec12c6): 1 copies (core 1, interop 0)
   files: DB/Library_Collections_DB.py:550
   | def _held_connection(self) -> sqlite3.Connection:
   |         """Return this thread's held connection, opening or reviving it.

   |         The liveness probe is a plain no-op statement; a connection another
   |         component closed (or that SQLite invalidated) is transparently
   |         replaced, mirroring `Workspace_DB._held_connection`.
   |         """
   |         from tldw_chatbook.Backup_Recovery.participants import _core_access

   |         _core_access(self)
   |         conn = getattr(self._thread_local, "conn", None)
   |         conn = _core_cached_connection(self, conn)
   |         if conn is not None:
   |             last_used = getattr(self._thread_local, "conn_last_used", None)
   |             if (
   |                 last_used is None
   |                 or (time.monotonic() - last_used)
   |                 >= self._LIVENESS_PING_IDLE_SECONDS
   |             ):
   |                 try:
   |                     conn.execute("SELECT 1")
   |                 except (sqlite3.ProgrammingError, sqlite3.OperationalError):
   |                     # A failed ping does not prove a live native borrower can be
   |                     # revoked. Only SQLite's closed-handle state permits revival.
   |                     try:
   |                         sqlite3.Connection.in_transaction.__get__(conn)
   |                     except sqlite3.ProgrammingError:
   |                         conn.close()  # Native already closed; failures retain refs.
   |    ...

--- body 7aa82c7f (shape 2a060c73): 1 copies (core 1, interop 0)
   files: Notifications/client_notifications_db.py:163
   | def _held_connection(self) -> sqlite3.Connection:
   |         """Return this thread's held connection, opening or reviving it.

   |         In-memory stores share the single cached connection instead (see
   |         the class docstring). The liveness probe is a plain no-op
   |         statement. Only a positively closed native handle is replaced;
   |         a failed probe on a live handle preserves it and propagates the error.
   |         """
   |         if getattr(self, "is_memory_db", False):
   |             return self._get_connection()
   |         _core_access(self)
   |         conn = getattr(self._thread_local, "conn", None)
   |         conn = _core_cached_connection(self, conn)
   |         if conn is not None:
   |             last_used = getattr(self._thread_local, "conn_last_used", None)
   |             if (
   |                 last_used is None
   |                 or (time.monotonic() - last_used)
   |                 >= self._LIVENESS_PING_IDLE_SECONDS
   |             ):
   |                 try:
   |                     conn.execute("SELECT 1")
   |                 except (sqlite3.ProgrammingError, sqlite3.OperationalError):
   |                     # A failed probe does not retire a live native borrower.
   |                     try:
   |                         sqlite3.Connection.in_transaction.__get__(conn)
   |                     except sqlite3.ProgrammingError:
   |                         conn.close()  # Positively closed; failure retains cache.
   |    ...

--- body 99b396a3 (shape a4af56e8): 1 copies (core 1, interop 0)
   files: Notifications/event_state_repository.py:245
   | def _held_connection(self) -> sqlite3.Connection:
   |         """Return this thread's held sqlite3 connection (see `_held_entry`)."""
   |         return self._held_entry().conn

====================================================================================================
# _identity: 33 defs, 26 distinct bodies, 23 distinct shapes

--- body d45500cc (shape 3161b1df): 4 copies (core 4, interop 0)
   files: LLM_Management/snapshot_store.py:85, TTS/audio_cpp_guided_launch.py:155, TTS/profile_reference_materialization.py:676, TTS/voice_bundle_service.py:304
   | def _identity(info: os.stat_result) -> tuple[int, int]:
   |     return info.st_dev, info.st_ino

--- body 342b6912 (shape c09038d8): 3 copies (core 3, interop 0)
   files: Agents/activation.py:24, MCP/activation.py:25, RAG_Search/activation.py:141
   | def _identity():
   |     try:
   |         task = asyncio.current_task()
   |     except RuntimeError:
   |         task = None
   |     return os.getpid(), threading.get_ident(), task

--- body a20aab43 (shape 2dd7ba5a): 2 copies (core 2, interop 0)
   files: Backup_Recovery/archive_reader.py:70, Petdex/sources.py:283
   | def _identity(info):
   |     return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

--- body 05525345 (shape a1c783bf): 2 copies (core 2, interop 0)
   files: Backup_Recovery/persona_visual_participants.py:271, Backup_Recovery/visual_identity_participants.py:326
   | def _identity(path):
   |     try:
   |         info = os.stat(path, follow_symlinks=False)
   |     except FileNotFoundError:
   |         return None
   |     return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)

--- body d05a2ea3 (shape 52907946): 1 copies (core 1, interop 0)
   files: Backup_Recovery/inventory.py:103
   | def _identity(path: Path) -> tuple[int, int]:
   |     value = os.stat(path)
   |     return value.st_dev, value.st_ino

--- body 6990ff38 (shape 126291c5): 1 copies (core 1, interop 0)
   files: Backup_Recovery/local_content_lifetime.py:39
   | def _identity():
   |     return os.getpid(), threading.get_ident(), storage._task_identity()

--- body 354ae7a0 (shape 25cd752f): 1 copies (core 1, interop 0)
   files: Backup_Recovery/mcp_source_participants.py:160
   | def _identity(state, path):
   |     try:
   |         info = (
   |             os.stat(path.name, dir_fd=state.pins[path.parent], follow_symlinks=False)
   |             if state.pinned and path.parent in state.pins
   |             else os.stat(path, follow_symlinks=False)
   |         )
   |     except FileNotFoundError:
   |         return None
   |     if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
   |         raise bootstrap.RecoveryRequired("raw_not_regular")
   |     return info.st_dev, info.st_ino

--- body 0b4ae607 (shape d3172b27): 1 copies (core 1, interop 0)
   files: Backup_Recovery/recovered_media.py:58
   | def _identity(profile, message, slug, media_type):
   |     values = (profile, message, slug, media_type)
   |     if any(
   |         type(value) is not str or not value or len(value) > 4096 for value in values
   |     ):
   |         raise ValueError("invalid_recovered_reference")
   |     if not media_type.startswith(("image/", "video/")):
   |         raise ValueError("invalid_recovered_media_type")
   |     return values

--- body dd20c9c4 (shape 3102a35d): 1 copies (core 1, interop 0)
   files: DB/agent_worktrees.py:68
   | def _identity(name: str, value: object) -> tuple[tuple[str, int, int, int], ...]:
   |     if (
   |         not isinstance(value, tuple)
   |         or not value
   |         or any(
   |             not isinstance(component, tuple)
   |             or len(component) != 4
   |             or not isinstance(component[0], str)
   |             or not Path(component[0]).is_absolute()
   |             or "\x00" in component[0]
   |             or any(type(part) is not int or part < 0 for part in component[1:])
   |             for component in value
   |         )
   |     ):
   |         raise ValueError(
   |             f"{name} must be a non-empty (path, device, inode, mode) chain"
   |         )
   |     return value

--- body 031c5ed4 (shape 728a38b1): 1 copies (core 1, interop 0)
   files: DB/automatic_work.py:39
   | def _identity(value: str) -> str:
   |     if not isinstance(value, str) or not value or len(value) > 256:
   |         raise ValueError("identity must be a nonempty bounded string")
   |     return value

--- body b2f891e8 (shape 78400f6d): 1 copies (core 1, interop 0)
   files: LLM_Calls/recovery_review.py:462
   | def _identity():
   |     import asyncio

   |     try:
   |         task = asyncio.current_task()
   |     except RuntimeError:
   |         task = None
   |     return os.getpid(), threading.get_ident(), task

--- body 6b536729 (shape 523b8c64): 1 copies (core 1, interop 0)
   files: Library/server_collections_capture_service.py:257
   | def _identity(self, capture_id: Any) -> CaptureIdentity:
   |         parsed = _positive_id(capture_id, "invalid_server_response")
   |         return CaptureIdentity(self.authority.key, str(parsed))

--- body cebf9230 (shape c14e91b6): 1 copies (core 1, interop 0)
   files: MCP/execution_log.py:387
   | def _identity(path: Path) -> tuple[int, ...] | None:
   |         """Fingerprint a generation file, or ``None`` if it cannot be read.

   |         ``lstat`` deliberately: a symlink in the leaf's place reports the
   |         link's own inode, which can never match the fingerprint of the
   |         regular file this instance last wrote, so a swapped path is a cache
   |         MISS and falls through to the guarded read. ``None`` (missing, or any
   |         other stat failure) is also a miss, never a decision -- every failure
   |         mode is handed to ``_read_bytes`` and its private-path guards to
   |         classify, exactly as before this cache existed. The fingerprint
   |         decides staleness only.
   |         """
   |         try:
   |             stat = os.lstat(path)
   |         except OSError:
   |             return None
   |         return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

--- body 48aa5171 (shape 43d5574c): 1 copies (core 1, interop 0)
   files: Notes/git_process_containment.py:348
   | def _identity(tree: OwnedProcessTree) -> _WindowsJobIdentity:
   |         identity = tree.native_identity
   |         if not isinstance(identity, _WindowsJobIdentity):
   |             raise ValueError("Invalid retained Windows Job Object identity")
   |         return identity

--- body 0ed2dde6 (shape 3161b1df): 1 copies (core 1, interop 0)
   files: Notes/notes_sync_coordinator.py:33
   | def _identity(metadata: os.stat_result) -> tuple[int, int]:
   |     return metadata.st_dev, metadata.st_ino

--- body 2eb314bc (shape 4a3fefab): 1 copies (core 1, interop 0)
   files: Notes/recovery_review.py:55
   | def _identity(path):
   |     if path is None:
   |         raise ValueError("notes_pairing_source_unavailable")
   |     selected = lexical_path(path)
   |     info = selected.lstat()
   |     if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)) or (
   |         stat.S_ISREG(info.st_mode) and info.st_nlink != 1
   |     ):
   |         raise ValueError("notes_pairing_source_unsafe")
   |     if selected.resolve() != selected:
   |         raise ValueError("notes_pairing_source_unsafe")
   |     return (str(selected), info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))

--- body 9330c129 (shape 244569aa): 1 copies (core 1, interop 0)
   files: Personal_Context/interview_diff.py:50
   | def _identity(change: InterviewProposedChange) -> tuple[str, str, str] | None:
   |     if change.proposed_payload is None or change.semantic_key is None:
   |         return None
   |     return (
   |         change.proposed_payload.kind,
   |         change.semantic_key.namespace,
   |         change.semantic_key.subject,
   |     )

--- body dc5d1e4e (shape b8de2f91): 1 copies (core 0, interop 1)
   files: Skills_Interop/recovery_activation.py:76
   | def _identity(path):
   |     from tldw_chatbook.Backup_Recovery.native_files import pinned_directory

   |     path = lexical_path(path)
   |     try:
   |         with pinned_directory(path.parent) as parent:
   |             try:
   |                 info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
   |             except FileNotFoundError:
   |                 return str(path), None
   |             if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
   |                 raise ValueError("skills_recovery_source_unsafe")
   |             if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
   |                 raise ValueError("skills_recovery_source_unsafe")
   |             return str(path), info.st_dev, info.st_ino
   |     except FileNotFoundError:
   |         return str(path), None

--- body 4d5311e2 (shape 992faea4): 1 copies (core 0, interop 1)
   files: Sync_Interop/personal_context_adapter.py:329
   | def _identity(object_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
   |     _parsed_type, value = _parse_payload(object_type, payload)
   |     if object_type == "manifest":
   |         return {
   |             "object_id": value.profile_id,
   |             "parent_id": None,
   |             "profile_id": value.profile_id,
   |             "base_version": None,
   |             "entity_version": value.current_version_id,
   |             "object_revision": value.revision,
   |             "operation": "upsert",
   |             "purge_generation": value.purge_generation,
   |         }
   |     if object_type == "scope":
   |         return {
   |             "object_id": value.scope_id,
   |             "parent_id": value.profile_id,
   |             "profile_id": value.profile_id,
   |             "base_version": None,
   |             "entity_version": value.version_id,
   |             "object_revision": None,
   |             "operation": "upsert",
   |             "purge_generation": None,
   |         }
   |     if object_type == "record":
   |         return {
   |             "object_id": value.record_id,
   |             "parent_id": value.scope_id,
   |    ...

--- body 3951aa05 (shape ac42ee3b): 1 copies (core 1, interop 0)
   files: TTS/audio_cpp_recipes.py:509
   | def _identity(parts: tuple[str, ...]) -> str:
   |     return sha256("\x00".join(parts).encode("utf-8")).hexdigest()

--- body e44ba49f (shape a331c3ce): 1 copies (core 1, interop 0)
   files: TTS/profile_migration_journal.py:205
   | def _identity(self) -> tuple[int, int]:
   |         return self.__authority.identity()

--- body 19eeca5e (shape 01b63571): 1 copies (core 1, interop 0)
   files: Tool_Packs/activation.py:209
   | def _identity(rule: object) -> tuple[str, str, str]:
   |     return (
   |         rule.authority,  # type: ignore[attr-defined,no-any-return]
   |         rule.server_key,  # type: ignore[attr-defined,no-any-return]
   |         rule.tool_name,  # type: ignore[attr-defined,no-any-return]
   |     )

--- body 127d293c (shape 8a1139eb): 1 copies (core 1, interop 0)
   files: Tool_Packs/contracts.py:353
   | def _identity(value: object, *, operation: str, category: str) -> str:
   |     return _nfc_text(
   |         value,
   |         operation=operation,
   |         category=category,
   |         max_bytes=MAX_IDENTITY_BYTES,
   |     )

--- body a38334a9 (shape 3161b1df): 1 copies (core 1, interop 0)
   files: Tool_Packs/publication.py:434
   | def _identity(value: os.stat_result) -> tuple[int, int]:
   |     return value.st_dev, value.st_ino

--- body d5962926 (shape a764d30c): 1 copies (core 1, interop 0)
   files: Tool_Packs/receipt_store.py:134
   | def _identity(raw: object) -> tuple[str, str, str]:
   |     value = _exact_dict(raw, _IDENTITY_KEYS)
   |     authority = value["authority"]
   |     if type(authority) is not str or authority not in {"mcp", "builtin"}:
   |         raise _fail("payload_invalid")
   |     server_key = _text(value["server_key"])
   |     tool_name = _text(value["tool_name"])
   |     if (authority == "builtin") != (server_key == "agent:builtin"):
   |         raise _fail("payload_invalid")
   |     return authority, server_key, tool_name

--- body 27b68340 (shape 01b63571): 1 copies (core 1, interop 0)
   files: UI/stts_profile_library.py:1564
   | def _identity(info):
   |         return info.st_dev, info.st_ino, info.st_mode

====================================================================================================
# _dump: 39 defs, 23 distinct bodies, 11 distinct shapes

--- body 15f50893 (shape 2f7ed901): 11 copies (core 2, interop 9)
   files: Feedback_Interop/server_feedback_service.py:87, LLM_Provider_Catalog/server_llm_provider_catalog_service.py:89, Notifications/server_notifications_service.py:89, Outputs_Interop/server_outputs_service.py:87, Research_Interop/local_research_search_service.py:172, Research_Interop/server_research_search_service.py:89, Server_Runtime_Interop/server_runtime_service.py:93, Sharing_Interop/server_sharing_service.py:87 ... +3
   | def _dump(response: Any) -> dict[str, Any]:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         return dict(response or {})

--- body b82b6d12 (shape ef0a1cb0): 4 copies (core 0, interop 4)
   files: Companion_Interop/server_companion_service.py:95, Personalization_Interop/server_personalization_service.py:97, Translation_Interop/server_translation_service.py:89, Voice_Assistant_Interop/server_voice_assistant_service.py:97
   | def _dump(cls, response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="python", by_alias=True)
   |         if isinstance(response, dict):
   |             return {key: cls._dump(value) for key, value in response.items()}
   |         if isinstance(response, list):
   |             return [cls._dump(item) for item in response]
   |         return response

--- body ba32ae5c (shape 370cbe66): 3 copies (core 0, interop 3)
   files: Kanban_Interop/server_kanban_service.py:687, Meetings_Interop/server_meetings_service.py:97, Prompt_Studio_Interop/server_prompt_studio_service.py:114
   | def _dump(cls, response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="python", by_alias=True)
   |         if isinstance(response, list):
   |             return [cls._dump(item) for item in response]
   |         if isinstance(response, dict):
   |             return {key: cls._dump(value) for key, value in response.items()}
   |         return response

--- body cdc7d4e0 (shape d0b50f0a): 2 copies (core 0, interop 2)
   files: Chat_Grammars_Interop/server_chat_grammars_service.py:89, Collections_Interop/server_collections_feeds_service.py:89
   | def _dump(response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, (dict, list, bool)):
   |             return response
   |         return dict(response or {})

--- body cedd8d60 (shape 785d4371): 1 copies (core 0, interop 1)
   files: Audio_Services_Interop/audio_services_scope_service.py:139
   | def _dump(payload: Any) -> Any:
   |         if hasattr(payload, "model_dump"):
   |             return payload.model_dump(mode="python")
   |         if isinstance(payload, dict):
   |             return dict(payload)
   |         if isinstance(payload, list):
   |             return [AudioServicesScopeService._dump(item) for item in payload]
   |         return payload

--- body 09f869c4 (shape 131114da): 1 copies (core 0, interop 1)
   files: Audio_Services_Interop/server_audio_services_service.py:106
   | def _dump(cls, response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="python")
   |         if isinstance(response, list):
   |             return [cls._dump(item) for item in response]
   |         if isinstance(response, dict):
   |             return dict(response)
   |         return response

--- body 64e0a6a6 (shape 131114da): 1 copies (core 0, interop 1)
   files: Auth_Account_Interop/server_auth_account_service.py:111
   | def _dump(cls, response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, list):
   |             return [cls._dump(item) for item in response]
   |         if isinstance(response, dict):
   |             return dict(response)
   |         return response

--- body e690dcbb (shape 8e5a3f8a): 1 copies (core 0, interop 1)
   files: Claims_Interop/claims_scope_service.py:76
   | def _dump(payload: Any) -> Any:
   |         if hasattr(payload, "model_dump"):
   |             return payload.model_dump(mode="python")
   |         if isinstance(payload, dict):
   |             return {
   |                 key: ClaimsScopeService._dump(value) for key, value in payload.items()
   |             }
   |         if isinstance(payload, list):
   |             return [ClaimsScopeService._dump(item) for item in payload]
   |         return payload

--- body 75c830f5 (shape c634764f): 1 copies (core 0, interop 1)
   files: Claims_Interop/server_claims_service.py:102
   | def _dump(cls, response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="python")
   |         if isinstance(response, list):
   |             return [cls._dump(item) for item in response]
   |         if isinstance(response, dict):
   |             return {key: cls._dump(value) for key, value in response.items()}
   |         return response

--- body 2de718ac (shape 0e836e3d): 1 copies (core 0, interop 1)
   files: External_Connectors_Interop/server_connectors_service.py:87
   | def _dump(response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, list):
   |             return [ServerConnectorsService._dump(item) for item in response]
   |         if isinstance(response, (dict, bool)):
   |             return response
   |         return dict(response or {})

--- body 948ab1b1 (shape ef0a1cb0): 1 copies (core 0, interop 1)
   files: Kanban_Interop/kanban_scope_service.py:104
   | def _dump(payload: Any) -> Any:
   |         if hasattr(payload, "model_dump"):
   |             return payload.model_dump(mode="python", by_alias=True)
   |         if isinstance(payload, dict):
   |             return {
   |                 key: KanbanScopeService._dump(value) for key, value in payload.items()
   |             }
   |         if isinstance(payload, list):
   |             return [KanbanScopeService._dump(item) for item in payload]
   |         return payload

--- body af6fad65 (shape b165d056): 1 copies (core 0, interop 1)
   files: MCP_Governance_Interop/server_mcp_governance_service.py:109
   | def _dump(response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, list):
   |             return [ServerMCPGovernanceService._dump(item) for item in response]
   |         if isinstance(response, dict):
   |             return response
   |         return dict(response or {})

--- body 191d6333 (shape ef0a1cb0): 1 copies (core 0, interop 1)
   files: Meetings_Interop/meetings_scope_service.py:76
   | def _dump(payload: Any) -> Any:
   |         if hasattr(payload, "model_dump"):
   |             return payload.model_dump(mode="python", by_alias=True)
   |         if isinstance(payload, dict):
   |             return {
   |                 key: MeetingsScopeService._dump(value) for key, value in payload.items()
   |             }
   |         if isinstance(payload, list):
   |             return [MeetingsScopeService._dump(item) for item in payload]
   |         return payload

--- body 56440b0b (shape ef0a1cb0): 1 copies (core 0, interop 1)
   files: Prompt_Studio_Interop/prompt_studio_scope_service.py:80
   | def _dump(payload: Any) -> Any:
   |         if hasattr(payload, "model_dump"):
   |             return payload.model_dump(mode="python", by_alias=True)
   |         if isinstance(payload, dict):
   |             return {
   |                 key: PromptStudioScopeService._dump(value)
   |                 for key, value in payload.items()
   |             }
   |         if isinstance(payload, list):
   |             return [PromptStudioScopeService._dump(item) for item in payload]
   |         return payload

--- body 694eb80d (shape bee3abf3): 1 copies (core 0, interop 1)
   files: Research_Interop/server_research_service.py:90
   | def _dump(response: Any) -> ResearchRecord:
   |         if hasattr(response, "model_dump"):
   |             payload = response.model_dump(mode="json")
   |         else:
   |             payload = dict(response or {})
   |         payload.setdefault("source", "server")
   |         return ResearchRecord(payload)

--- body 7e091973 (shape 0e836e3d): 1 copies (core 0, interop 1)
   files: Skills_Interop/local_skills_service.py:338
   | def _dump(response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, list):
   |             return [LocalSkillsService._dump(item) for item in response]
   |         if isinstance(response, (dict, bool)):
   |             return response
   |         return dict(response or {})

--- body a992dec0 (shape 0e836e3d): 1 copies (core 0, interop 1)
   files: Skills_Interop/server_skills_service.py:87
   | def _dump(response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, list):
   |             return [ServerSkillsService._dump(item) for item in response]
   |         if isinstance(response, (dict, bool)):
   |             return response
   |         return dict(response or {})

--- body a4885902 (shape c634764f): 1 copies (core 0, interop 1)
   files: Sync_Interop/key_recovery_service.py:123
   | def _dump(value: Any) -> Any:
   |         if hasattr(value, "model_dump"):
   |             return value.model_dump(mode="json")
   |         if isinstance(value, list):
   |             return [SyncKeyRecoveryService._dump(item) for item in value]
   |         if isinstance(value, dict):
   |             return {
   |                 key: SyncKeyRecoveryService._dump(item) for key, item in value.items()
   |             }
   |         return value

--- body 7fdf2674 (shape c634764f): 1 copies (core 0, interop 1)
   files: Sync_Interop/local_first_sync_service.py:709
   | def _dump(value: Any) -> Any:
   |         if hasattr(value, "model_dump"):
   |             return value.model_dump(mode="json")
   |         if isinstance(value, list):
   |             return [LocalFirstSyncService._dump(item) for item in value]
   |         if isinstance(value, dict):
   |             return {
   |                 key: LocalFirstSyncService._dump(item) for key, item in value.items()
   |             }
   |         return value

--- body 8a245d19 (shape c634764f): 1 copies (core 0, interop 1)
   files: Sync_Interop/restore_service.py:253
   | def _dump(value: Any) -> Any:
   |         if hasattr(value, "model_dump"):
   |             return value.model_dump(mode="json")
   |         if isinstance(value, list):
   |             return [SyncRestoreService._dump(item) for item in value]
   |         if isinstance(value, dict):
   |             return {key: SyncRestoreService._dump(item) for key, item in value.items()}
   |         return value

--- body 0c503346 (shape c634764f): 1 copies (core 0, interop 1)
   files: Sync_Interop/server_sync_service.py:213
   | def _dump(response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, list):
   |             return [ServerSyncService._dump(item) for item in response]
   |         if isinstance(response, dict):
   |             return {
   |                 key: ServerSyncService._dump(value) for key, value in response.items()
   |             }
   |         return response

--- body 92a7ac00 (shape b165d056): 1 copies (core 0, interop 1)
   files: Text2SQL_Interop/server_text2sql_service.py:87
   | def _dump(response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, list):
   |             return [ServerText2SQLService._dump(item) for item in response]
   |         if isinstance(response, dict):
   |             return response
   |         return dict(response or {})

--- body 42b55eba (shape b165d056): 1 copies (core 0, interop 1)
   files: Tools_Interop/server_tools_service.py:87
   | def _dump(response: Any) -> Any:
   |         if hasattr(response, "model_dump"):
   |             return response.model_dump(mode="json")
   |         if isinstance(response, list):
   |             return [ServerToolsService._dump(item) for item in response]
   |         if isinstance(response, dict):
   |             return response
   |         return dict(response or {})
