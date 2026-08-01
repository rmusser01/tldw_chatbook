# SQLite Private Owner Inventory

Status: migrated and behaviorally verified for TASK-489

This inventory is the migration ledger for the SQLite private-storage
boundary established by ADR-029. Module paths are relative to the repository
root and omit the `.py` suffix. A row's owner ID is a literal key in
`SQLITE_OWNER_REGISTRY`.

Classifications have these meanings:

- `private_file`: a writable Chatbook-owned SQLite file whose main file,
  sidecars, and backups require the private path boundary.
- `memory`: SQLite's exact `:memory:` token; no filesystem artifact exists.
- `read_only_uri`: a path-based `mode=ro` URI built by the private SQLite
  boundary after validating the source file.

## Direct connection inventory

| ID | Module | Symbol | Owner ID | Classification | Intent | Migration disposition |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | tldw_chatbook/Writing_Interop/local_writing_service | LocalWritingService._connect | writing.local | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve the accepted `Path(":memory:")` form and route files through the checked seam. |
| C02 | tldw_chatbook/Research_Interop/local_research_service | LocalResearchService._connect | research.local | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve the accepted `Path(":memory:")` form and route files through the checked seam. |
| C03 | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | get_chrome_cookies | cookies.chrome | read_only_uri | read-only clone | Migrated via `connect_private_sqlite`. Open the owner-only temporary clone with a validated read-only URI. |
| C04 | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | get_firefox_cookies | cookies.firefox | read_only_uri | read-only clone | Migrated via `connect_private_sqlite`. Open the owner-only temporary clone with a validated read-only URI. |
| C05 | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | get_edge_cookies | cookies.edge | read_only_uri | read-only clone | Migrated via `connect_private_sqlite`. Open the owner-only temporary clone with a validated read-only URI. |
| C06 | tldw_chatbook/Sync_Interop/notes_mirror | NotesMirror.__init__ | sync.notes_mirror | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve `:memory:` and route an optional file target through the private seam. |
| C07 | tldw_chatbook/Sync_Interop/sync_state_repository | SyncStateRepository._get_connection | sync.state | memory | read/write | Migrated via `connect_private_sqlite`. Preserve the exact in-memory contract through the checked seam. |
| C08 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._vacuum_single_worker | settings.vacuum | private_file | maintenance write | Migrated via `connect_private_sqlite`. Route VACUUM through the checked writable private-file seam. |
| C09 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._check_single_worker | settings.integrity | read_only_uri | integrity read | Migrated via `connect_private_sqlite`. Route the integrity check through a validated read-only URI. |
| C10 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._get_schema_version | settings.schema | read_only_uri | schema read | Migrated via `connect_private_sqlite`. Route the schema lookup through a validated read-only URI. |
| C11 | tldw_chatbook/DB/RAG_Indexing_DB | RAGIndexingDB._get_connection | db.rag_indexing | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve memory support and route files through the checked seam. |
| C12 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB._get_thread_connection | db.chachanotes.primary | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve connection options while routing the target through the checked seam. |
| C13 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB.backup_database | db.chachanotes.backup | private_file | backup target | Migrated via `backup_connection_to_private`. Uses the centralized caller-connection backup operation. |
| C14 | tldw_chatbook/DB/base_db | BaseDB._get_connection | db.base | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Make the shared base connection the checked seam for its subclasses. |
| C15 | tldw_chatbook/DB/Evals_DB | EvalsDB._get_connection | db.evals | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve memory and thread options while routing files through the checked seam. |
| C16 | tldw_chatbook/DB/search_history_db | SearchHistoryDB._get_connection | db.search_history | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve memory support and route files through the checked seam. |
| C17 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase._get_thread_connection | db.media.primary | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve connection options while routing the target through the checked seam. |
| C18 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase.backup_database | db.media.backup | private_file | backup target | Migrated via `backup_connection_to_private`. Uses the centralized caller-connection backup operation. |
| C19 | tldw_chatbook/DB/Client_Media_DB_v2 | check_database_integrity | db.media.integrity | read_only_uri | integrity read | Migrated via `connect_private_sqlite`. Replace the interpolated URI with the validated URI builder. |
| C20 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase._get_thread_connection | db.prompts.primary | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve connection options while routing the target through the checked seam. |
| C21 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase.backup_database | db.prompts.backup | private_file | backup target | Migrated via `backup_connection_to_private`. Uses the centralized caller-connection backup operation. |
| C22 | tldw_chatbook/DB/Library_Ingest_Jobs_DB | LibraryIngestJobsDB._get_connection | db.library_ingest_jobs | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Route the override through the checked seam without changing its WAL contract. |
| C23 | tldw_chatbook/Kanban_Interop/local_kanban_db | open_connection | kanban.local | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve `:memory:` and route files through the checked seam. |
| C24 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage._init_db | tamagotchi.sqlite | private_file, memory | read/write | Migrated via `connect_private_sqlite`. Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C25 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.load | tamagotchi.sqlite | private_file, memory | read | Migrated via `connect_private_sqlite`. Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C26 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.save | tamagotchi.sqlite | private_file, memory | write | Migrated via `connect_private_sqlite`. Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C27 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.delete | tamagotchi.sqlite | private_file, memory | write | Migrated via `connect_private_sqlite`. Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C28 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.list_pets | tamagotchi.sqlite | private_file, memory | read | Migrated via `connect_private_sqlite`. Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C29 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.get_statistics | tamagotchi.sqlite | private_file, memory | read | Migrated via `connect_private_sqlite`. Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C30 | tldw_chatbook/Notifications/client_notifications_db | ClientNotificationsDB._get_connection | notifications.client | memory | read/write | Migrated via `connect_private_sqlite`. Preserve the exact in-memory contract through the checked seam. |
| C31 | tldw_chatbook/Notifications/event_state_repository | EventStateRepository._get_connection | notifications.event_state | memory | read/write | Migrated via `connect_private_sqlite`. Preserve the exact in-memory contract through the checked seam. |
| C32 | tldw_chatbook/TTS/profile_schema | open_profile_store | tts.profile_store | private_file | read/write | Migrated via `connect_private_sqlite`. Preserve the repository's exclusive lease and explicit create-versus-existing contract while enforcing the private path boundary. |
| C33 | tldw_chatbook/TTS/profile_schema | validate_profile_candidate | tts.profile_candidate | read_only_uri | immutable validation read | Migrated via `connect_private_sqlite`. Validate an owner-only immutable snapshot rather than opening the caller-selected candidate directly. |
| C34 | tldw_chatbook/TTS/profile_repository | TTSProfileRepository._worker_backup_to | tts.profile_backup | private_file | backup destination | Migrated via `connect_private_sqlite`. Open the already-created private temporary destination through the checked writable seam before online backup. |
| C35 | tldw_chatbook/TTS/profile_repository | TTSProfileRepository._worker_validate_standalone_snapshot | tts.profile_snapshot | read_only_uri | immutable integrity read | Migrated via `connect_private_sqlite`. Run full integrity checks through a validated immutable read-only handle. |
| C36 | tldw_chatbook/DB/Subscriptions_DB | ensure_site_configs_schema | db.subscriptions.site_configs | private_file | declare one table | Migrated via `connect_private_sqlite`. Declares `site_configs` on a caller-supplied path without opening the whole `SubscriptionsDB`, so the one table `SiteConfigManager` needs exists without imposing ~15 unrelated tables on that file. |
| C37 | tldw_chatbook/Notes/file_notes_replica | FileNotesReplica.__init__ | notes.file_notes_replica | private_file, memory | read/write recovery replica | Migrated via `connect_private_sqlite`. The independent File Notes replica stores exact private note bytes, so file targets use the checked private boundary while preserving the exact in-memory test target. |

## SQLite backup and restore inventory

| ID | Module | Symbol | Owner ID | Classification | Operation | Migration disposition |
| --- | --- | --- | --- | --- | --- | --- |
| B01 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB.backup_database | db.chachanotes.backup | private_file | backup_connection_to_private | Migrated via `backup_connection_to_private`. Verifies the explicit caller-owned source and creates the private target centrally. |
| B02 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase.backup_database | db.media.backup | private_file | backup_connection_to_private | Migrated via `backup_connection_to_private`. Verifies the explicit caller-owned source and creates the private target centrally. |
| B03 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase.backup_database | db.prompts.backup | private_file | backup_connection_to_private | Migrated via `backup_connection_to_private`. Verifies the explicit caller-owned source and creates the private target centrally. |
| B04 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker ChaChaNotes target | settings.bulk_backup | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. The shared six-owner loop opens the verified source read-only and transactionally backs it up to a private target. |
| B05 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker Prompts target | settings.bulk_backup | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. The shared six-owner loop opens the verified source read-only and transactionally backs it up to a private target. |
| B06 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker Media target | settings.bulk_backup | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. The shared six-owner loop opens the verified source read-only and transactionally backs it up to a private target. |
| B07 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker Evals target | settings.bulk_backup | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. The shared six-owner loop opens the verified source read-only and transactionally backs it up to a private target. |
| B08 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker RAG target | settings.bulk_backup | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. The shared six-owner loop opens the verified source read-only and transactionally backs it up to a private target. |
| B09 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker Subscriptions target | settings.bulk_backup | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. The shared six-owner loop opens the verified source read-only and transactionally backs it up to a private target. |
| B10 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_single_worker | settings.single_backup | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. Opens the verified source read-only and transactionally backs it up to a private target. |
| B11 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._restore_single_worker pre-restore branch | settings.pre_restore_backup | private_file, read_only_uri | restore_private_sqlite | Migrated via `restore_private_sqlite`. Creates the private safety snapshot inside the guarded destination lifecycle. |
| B12 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._restore_single_worker restore branch | settings.restore | private_file, read_only_uri | restore_private_sqlite | Migrated via `restore_private_sqlite`. Restores through verified source identity and prompt-fail destination quiescence. |
| B13 | tldw_chatbook/TTS/profile_repository | TTSProfileRepository._worker_online_backup | tts.profile_backup | private_file | backup_open_connections_to_private | Migrated via `backup_open_connections_to_private`. Preserve the repository's caller-owned connection and deadline callback contract while centralizing source pinning and page backup. |
| B14 | tldw_chatbook/TTS/profile_repository | TTSProfileRepository._worker_stage_candidate | tts.profile_restore_stage | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. Copy the pinned candidate to a private restore stage while retaining per-page deadline checks. |
| B15 | tldw_chatbook/TTS/profile_repository | TTSProfileRepository._worker_create_recovery_backup | tts.profile_recovery | private_file | backup_connection_to_private | Migrated via `backup_connection_to_private`. Back up the leased live connection to a private recovery target while retaining per-page deadline checks. |
| B16 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._restore_single_worker fresh-target branch | settings.restore | private_file, read_only_uri | copy_private_sqlite | Migrated via `copy_private_sqlite`. A never-before-created custom restore target has no live database to quiesce or snapshot, so it is copied directly into a fresh private target instead of going through the guarded live-restore path (TASK-927 follow-up). |

Restore retains one exclusive destination connection across the quiescence
probe, private safety snapshot, final transactional page backup, and
reassertion of the destination's original DELETE/WAL journal mode. A
post-commit validation or journal-mode failure triggers rollback from the
private snapshot. If rollback fails, the UI reports an indeterminate live
state, identifies the snapshot, and warns against an automatic retry. Active
readers/writers fail promptly without a success notification. A previously
queried idle WAL connection can also prevent SQLite from proving exclusivity;
that case fails closed and reports that live restore is unavailable rather
than replacing the database file.

## Database parent creator inventory

The disposition vocabulary is checked by the inventory test:
`secure_default`, `remove_custom_creation`, `centralize_backup`,
`remove_obsolete_creation`, or `justified_exclusion`.
`current` rows must retain the exact qualified creator-call anchor shown
below. A migration changes the row to `migrated`, after which the test requires
that legacy anchor to be absent; rows are not deleted when their call is
removed.

Parent discovery is intentionally curated because arbitrary `mkdir` calls do
not reveal whether a directory will own SQLite data. The guard derives and
checks every listed current anchor from production AST, but does not freeze all
production directory creation. A new non-direct database-parent owner must add
a checked `P` row when it is introduced.

| ID | Module | Qualified containing symbol | Creator call | State | Owner ID | Disposition | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | tldw_chatbook/config | get_user_data_dir | user_dir.mkdir(parents=True, exist_ok=True) | migrated | config.user_data_directory | secure_default | This application-owned default data directory becomes the explicit `0700` creation boundary. |
| P02 | tldw_chatbook/config | load_settings | main_db_file_path_server.parent.mkdir(parents=True, exist_ok=True) | migrated | config.server_sqlite_parent | remove_obsolete_creation | The `DATABASE_URL` branch has no Chatbook connection consumer, so its stale mkdir side effect is removed. |
| P03 | tldw_chatbook/config | load_settings | user_data_base_dir_server.mkdir(parents=True, exist_ok=True) | migrated | config.server_user_db_base | remove_obsolete_creation | The `USER_DB_BASE_DIR` branch has no Chatbook connection consumer, so its stale mkdir side effect is removed. |
| P04 | tldw_chatbook/Utils/paths | get_project_databases_dir | PROJECT_DATABASES_DIR.mkdir(parents=True, exist_ok=True) | current | utils.project_databases_directory | justified_exclusion | Project template and executable demonstration storage is not a Chatbook-owned production SQLite target. |
| P05 | tldw_chatbook/Utils/paths | get_user_database_path | USER_DB_DIR.mkdir(parents=True, exist_ok=True) | migrated | utils.legacy_user_database_path | remove_obsolete_creation | TASK-865 removed the unreachable creator without touching any database. |
| P06 | tldw_chatbook/DB/base_db | BaseDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | db.base | remove_custom_creation | Constructors stop resolving selected paths or creating arbitrary parents. |
| P07 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | db.chachanotes.primary | remove_custom_creation | The caller must supply an existing trusted custom parent or the secured default data directory. |
| P08 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB.backup_database | backup_db_path_obj.parent.mkdir(parents=True, exist_ok=True) | migrated | db.chachanotes.backup | centralize_backup | The centralized backup seam owns target-parent validation. |
| P09 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | db.media.primary | remove_custom_creation | The caller must supply an existing trusted custom parent or the secured default data directory. |
| P10 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase.backup_database | backup_db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | db.media.backup | centralize_backup | The centralized backup seam owns target-parent validation. |
| P11 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | db.prompts.primary | remove_custom_creation | The caller must supply an existing trusted custom parent or the secured default data directory. |
| P12 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase.backup_database | backup_db_path_obj.parent.mkdir(parents=True, exist_ok=True) | migrated | db.prompts.backup | centralize_backup | The centralized backup seam owns target-parent validation. |
| P13 | tldw_chatbook/DB/RAG_Indexing_DB | RAGIndexingDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | db.rag_indexing | remove_custom_creation | Constructors stop creating arbitrary selected parents. |
| P14 | tldw_chatbook/DB/Evals_DB | EvalsDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | db.evals | remove_custom_creation | Constructors stop creating arbitrary selected parents. |
| P15 | tldw_chatbook/DB/search_history_db | SearchHistoryDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | db.search_history | remove_custom_creation | Constructors stop creating arbitrary selected parents. |
| P16 | tldw_chatbook/Kanban_Interop/local_kanban_db | open_connection | Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True) | migrated | kanban.local | remove_custom_creation | The connection helper stops creating arbitrary selected parents. |
| P17 | tldw_chatbook/Research_Interop/local_research_service | LocalResearchService.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | research.local | remove_custom_creation | The path-backed constructor stops creating arbitrary selected parents. |
| P18 | tldw_chatbook/Writing_Interop/local_writing_service | LocalWritingService.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | writing.local | remove_custom_creation | The constructor stops creating arbitrary selected parents. |
| P19 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | tamagotchi.sqlite | remove_custom_creation | The constructor stops creating arbitrary selected parents. |
| P20 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker | backup_dir.mkdir(parents=True, exist_ok=True) | migrated | settings.bulk_backup | centralize_backup | Settings secures the application-owned timestamp backup directory before centralized backup. |
| P21 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_single_worker | backup_dir.mkdir(parents=True, exist_ok=True) | migrated | settings.single_backup | centralize_backup | Settings secures the application-owned per-database backup directory before centralized backup. |
| P22 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._restore_single_database | backup_dir.mkdir(parents=True, exist_ok=True) | migrated | settings.restore | centralize_backup | Settings secures the application-owned restore picker directory before restore. |
| P23 | tldw_chatbook/Evals/eval_orchestrator | EvaluationOrchestrator._initialize_database | Path(db_path).parent.mkdir(parents=True, exist_ok=True) | migrated | eval.orchestrator_parent | secure_default | Secure only the application-owned default; custom parents must already be trusted. |
| P25 | tldw_chatbook/app | TldwCli._init_prompts_service | prompts_db_path.parent.mkdir(parents=True, exist_ok=True) | migrated | app.prompts_parent | remove_custom_creation | Startup delegates parent policy to the configured default/custom path boundary. |
| P26 | tldw_chatbook/Notes/Notes_Library | NotesInteropService.__init__ | self.base_db_directory.mkdir(parents=True, exist_ok=True) | migrated | notes.library_parent | secure_default | Secure the application-owned per-user database root without changing a custom namespace. |
| P27 | tldw_chatbook/DB/Sync_Client | <module> | os.makedirs(os.path.dirname(DATABASE_PATH) or '.', exist_ok=True) | migrated | db.sync_client_example | secure_default | The executable-adjacent default is secured and fails closed instead of teaching unsafe parent creation. |
| P28 | tldw_chatbook/runtime_policy/server_parity_state | build_server_parity_state_repositories | resolved_data_dir.mkdir(parents=True, exist_ok=True) | migrated | runtime.server_parity_parent | secure_default | Preserve the lexical file-backed repository directory; secure the default and require a trusted custom namespace. |

## Explicit exclusions

| ID | Module | Symbol | Exclusion |
| --- | --- | --- | --- |
| X01 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | JSONStorage._create_backup | Copies a JSON adapter file, not a SQLite database or sidecar. |
| X02 | tldw_chatbook/DB/Client_Media_DB_v2 | create_incremental_backup | No-op placeholder; it creates no backup artifact. |
| X03 | tldw_chatbook/DB/Client_Media_DB_v2 | create_automated_backup | No-op placeholder; it creates no backup artifact. |
| X04 | production tree | aiosqlite.connect | No production `aiosqlite.connect` owner exists. |

The migrated boundary retains 37 classified connection sites and sixteen
classified backup/restore operations. Production has one raw
`sqlite3.connect` site and one direct `Connection.backup()` site, both inside
`DB/private_sqlite.py`; Settings has no SQLite database `shutil.copy2()` site.
