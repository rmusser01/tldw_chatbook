# SQLite Private Owner Inventory

Status: checked baseline for TASK-489

This inventory is the migration ledger for the SQLite private-storage
boundary established by ADR-022. Module paths are relative to the repository
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
| C01 | tldw_chatbook/Writing_Interop/local_writing_service | LocalWritingService._connect | writing.local | private_file, memory | read/write | Preserve the accepted `Path(":memory:")` form and route files through the checked seam. |
| C02 | tldw_chatbook/Research_Interop/local_research_service | LocalResearchService._connect | research.local | private_file, memory | read/write | Preserve the accepted `Path(":memory:")` form and route files through the checked seam. |
| C03 | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | get_chrome_cookies | cookies.chrome | read_only_uri | read-only clone | Open the owner-only temporary clone with a validated read-only URI. |
| C04 | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | get_firefox_cookies | cookies.firefox | read_only_uri | read-only clone | Open the owner-only temporary clone with a validated read-only URI. |
| C05 | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | get_edge_cookies | cookies.edge | read_only_uri | read-only clone | Open the owner-only temporary clone with a validated read-only URI. |
| C06 | tldw_chatbook/Sync_Interop/notes_mirror | NotesMirror.__init__ | sync.notes_mirror | private_file, memory | read/write | Preserve `:memory:` and route an optional file target through the private seam. |
| C07 | tldw_chatbook/Sync_Interop/sync_state_repository | SyncStateRepository._get_connection | sync.state | memory | read/write | Preserve the exact in-memory contract through the checked seam. |
| C08 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._vacuum_single_worker | settings.vacuum | private_file | maintenance write | Route VACUUM through the checked writable private-file seam. |
| C09 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._check_single_worker | settings.integrity | read_only_uri | integrity read | Route the integrity check through a validated read-only URI. |
| C10 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._get_schema_version | settings.schema | read_only_uri | schema read | Route the schema lookup through a validated read-only URI. |
| C11 | tldw_chatbook/DB/RAG_Indexing_DB | RAGIndexingDB._get_connection | db.rag_indexing | private_file, memory | read/write | Preserve memory support and route files through the checked seam. |
| C12 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB._get_thread_connection | db.chachanotes.primary | private_file, memory | read/write | Preserve connection options while routing the target through the checked seam. |
| C13 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB.backup_database | db.chachanotes.backup | private_file | backup target | Replace the raw target connection with the centralized SQLite backup operation. |
| C14 | tldw_chatbook/DB/base_db | BaseDB._get_connection | db.base | private_file, memory | read/write | Make the shared base connection the checked seam for its subclasses. |
| C15 | tldw_chatbook/DB/Evals_DB | EvalsDB._get_connection | db.evals | private_file, memory | read/write | Preserve memory and thread options while routing files through the checked seam. |
| C16 | tldw_chatbook/DB/search_history_db | SearchHistoryDB._get_connection | db.search_history | private_file, memory | read/write | Preserve memory support and route files through the checked seam. |
| C17 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase._get_thread_connection | db.media.primary | private_file, memory | read/write | Preserve connection options while routing the target through the checked seam. |
| C18 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase.backup_database | db.media.backup | private_file | backup target | Replace the raw target connection with the centralized SQLite backup operation. |
| C19 | tldw_chatbook/DB/Client_Media_DB_v2 | check_database_integrity | db.media.integrity | read_only_uri | integrity read | Replace the interpolated URI with the validated URI builder. |
| C20 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase._get_thread_connection | db.prompts.primary | private_file, memory | read/write | Preserve connection options while routing the target through the checked seam. |
| C21 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase.backup_database | db.prompts.backup | private_file | backup target | Replace the raw target connection with the centralized SQLite backup operation. |
| C22 | tldw_chatbook/DB/Library_Ingest_Jobs_DB | LibraryIngestJobsDB._get_connection | db.library_ingest_jobs | private_file, memory | read/write | Route the override through the checked seam without changing its WAL contract. |
| C23 | tldw_chatbook/Kanban_Interop/local_kanban_db | open_connection | kanban.local | private_file, memory | read/write | Preserve `:memory:` and route files through the checked seam. |
| C24 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage._init_db | tamagotchi.sqlite | private_file, memory | read/write | Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C25 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.load | tamagotchi.sqlite | private_file, memory | read | Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C26 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.save | tamagotchi.sqlite | private_file, memory | write | Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C27 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.delete | tamagotchi.sqlite | private_file, memory | write | Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C28 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.list_pets | tamagotchi.sqlite | private_file, memory | read | Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C29 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.get_statistics | tamagotchi.sqlite | private_file, memory | read | Preserve the accepted `Path(":memory:")` form and use the common checked owner. |
| C30 | tldw_chatbook/Notifications/client_notifications_db | ClientNotificationsDB._get_connection | notifications.client | memory | read/write | Preserve the exact in-memory contract through the checked seam. |
| C31 | tldw_chatbook/Notifications/event_state_repository | EventStateRepository._get_connection | notifications.event_state | memory | read/write | Preserve the exact in-memory contract through the checked seam. |

## SQLite backup and restore inventory

| ID | Module | Symbol | Owner ID | Classification | Operation | Migration disposition |
| --- | --- | --- | --- | --- | --- | --- |
| B01 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB.backup_database | db.chachanotes.backup | private_file | Connection.backup | Centralize source verification and private target creation. |
| B02 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase.backup_database | db.media.backup | private_file | Connection.backup | Centralize source verification and private target creation. |
| B03 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase.backup_database | db.prompts.backup | private_file | Connection.backup | Centralize source verification and private target creation. |
| B04 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker ChaChaNotes branch | settings.bulk_backup | private_file | shutil.copy2 | Replace live-file copying with centralized SQLite backup. |
| B05 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker Prompts branch | settings.bulk_backup | private_file | shutil.copy2 | Replace live-file copying with centralized SQLite backup. |
| B06 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker Media branch | settings.bulk_backup | private_file | shutil.copy2 | Replace live-file copying with centralized SQLite backup. |
| B07 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_single_worker | settings.single_backup | private_file | shutil.copy2 | Replace live-file copying with centralized SQLite backup. |
| B08 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._restore_single_worker pre-restore branch | settings.pre_restore_backup | private_file | shutil.copy2 | Create the pre-restore safety copy through centralized SQLite backup. |
| B09 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._restore_single_worker restore branch | settings.restore | private_file | shutil.copy2 | Restore through a verified SQLite source and private destination lifecycle. |

## Database parent creator inventory

The disposition vocabulary is checked by the inventory test:
`secure_default`, `remove_custom_creation`, `centralize_backup`,
`remove_obsolete_creation`, or `justified_exclusion`.
`current` rows must retain the exact qualified creator-call anchor shown
below. A migration changes the row to `migrated`, after which the test requires
that legacy anchor to be absent; rows are not deleted when their call is
removed.

| ID | Module | Qualified containing symbol | Creator call | State | Owner ID | Disposition | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | tldw_chatbook/config | get_user_data_dir | user_dir.mkdir(parents=True, exist_ok=True) | current | config.user_data_directory | secure_default | This application-owned default data directory becomes the explicit `0700` creation boundary. |
| P02 | tldw_chatbook/config | load_settings | main_db_file_path_server.parent.mkdir(parents=True, exist_ok=True) | current | config.server_sqlite_parent | remove_obsolete_creation | The `DATABASE_URL` branch has no Chatbook connection consumer, so its stale mkdir side effect is removed. |
| P03 | tldw_chatbook/config | load_settings | user_data_base_dir_server.mkdir(parents=True, exist_ok=True) | current | config.server_user_db_base | remove_obsolete_creation | The `USER_DB_BASE_DIR` branch has no Chatbook connection consumer, so its stale mkdir side effect is removed. |
| P04 | tldw_chatbook/Utils/paths | get_project_databases_dir | PROJECT_DATABASES_DIR.mkdir(parents=True, exist_ok=True) | current | utils.project_databases_directory | justified_exclusion | Project template and executable demonstration storage is not a Chatbook-owned production SQLite target. |
| P05 | tldw_chatbook/Utils/paths | get_user_database_path | USER_DB_DIR.mkdir(parents=True, exist_ok=True) | current | utils.legacy_user_database_path | justified_exclusion | The unused legacy helper has no production connection owner. |
| P06 | tldw_chatbook/DB/base_db | BaseDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | db.base | remove_custom_creation | Constructors stop resolving selected paths or creating arbitrary parents. |
| P07 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | db.chachanotes.primary | remove_custom_creation | The caller must supply an existing trusted custom parent or the secured default data directory. |
| P08 | tldw_chatbook/DB/ChaChaNotes_DB | CharactersRAGDB.backup_database | backup_db_path_obj.parent.mkdir(parents=True, exist_ok=True) | current | db.chachanotes.backup | centralize_backup | The centralized backup seam owns target-parent validation. |
| P09 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | db.media.primary | remove_custom_creation | The caller must supply an existing trusted custom parent or the secured default data directory. |
| P10 | tldw_chatbook/DB/Client_Media_DB_v2 | MediaDatabase.backup_database | backup_db_path.parent.mkdir(parents=True, exist_ok=True) | current | db.media.backup | centralize_backup | The centralized backup seam owns target-parent validation. |
| P11 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | db.prompts.primary | remove_custom_creation | The caller must supply an existing trusted custom parent or the secured default data directory. |
| P12 | tldw_chatbook/DB/Prompts_DB | PromptsDatabase.backup_database | backup_db_path_obj.parent.mkdir(parents=True, exist_ok=True) | current | db.prompts.backup | centralize_backup | The centralized backup seam owns target-parent validation. |
| P13 | tldw_chatbook/DB/RAG_Indexing_DB | RAGIndexingDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | db.rag_indexing | remove_custom_creation | Constructors stop creating arbitrary selected parents. |
| P14 | tldw_chatbook/DB/Evals_DB | EvalsDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | db.evals | remove_custom_creation | Constructors stop creating arbitrary selected parents. |
| P15 | tldw_chatbook/DB/search_history_db | SearchHistoryDB.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | db.search_history | remove_custom_creation | Constructors stop creating arbitrary selected parents. |
| P16 | tldw_chatbook/Kanban_Interop/local_kanban_db | open_connection | Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True) | current | kanban.local | remove_custom_creation | The connection helper stops creating arbitrary selected parents. |
| P17 | tldw_chatbook/Research_Interop/local_research_service | LocalResearchService.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | research.local | remove_custom_creation | The path-backed constructor stops creating arbitrary selected parents. |
| P18 | tldw_chatbook/Writing_Interop/local_writing_service | LocalWritingService.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | writing.local | remove_custom_creation | The constructor stops creating arbitrary selected parents. |
| P19 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | SQLiteStorage.__init__ | self.db_path.parent.mkdir(parents=True, exist_ok=True) | current | tamagotchi.sqlite | remove_custom_creation | The constructor stops creating arbitrary selected parents. |
| P20 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_worker | backup_dir.mkdir(parents=True, exist_ok=True) | current | settings.bulk_backup | centralize_backup | Settings secures the application-owned timestamp backup directory before centralized backup. |
| P21 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._backup_single_worker | backup_dir.mkdir(parents=True, exist_ok=True) | current | settings.single_backup | centralize_backup | Settings secures the application-owned per-database backup directory before centralized backup. |
| P22 | tldw_chatbook/UI/Tools_Settings_Window | ToolsSettingsWindow._restore_single_database | backup_dir.mkdir(parents=True, exist_ok=True) | current | settings.restore | centralize_backup | Settings secures the application-owned restore picker directory before restore. |
| P23 | tldw_chatbook/Evals/eval_orchestrator | EvaluationOrchestrator._initialize_database | Path(db_path).parent.mkdir(parents=True, exist_ok=True) | current | eval.orchestrator_parent | secure_default | Secure only the application-owned default; custom parents must already be trusted. |
| P24 | tldw_chatbook/Event_Handlers/eval_events | get_orchestrator | db_path.parent.mkdir(parents=True, exist_ok=True) | current | eval.events_parent | secure_default | Secure only the application-owned default; custom parents must already be trusted. |
| P25 | tldw_chatbook/app | TldwCli._init_prompts_service | prompts_db_path.parent.mkdir(parents=True, exist_ok=True) | current | app.prompts_parent | remove_custom_creation | Startup delegates parent policy to the configured default/custom path boundary. |
| P26 | tldw_chatbook/Notes/Notes_Library | NotesInteropService.__init__ | self.base_db_directory.mkdir(parents=True, exist_ok=True) | current | notes.library_parent | secure_default | Secure the application-owned per-user database root without changing a custom namespace. |
| P27 | tldw_chatbook/DB/Sync_Client | <module> | os.makedirs(os.path.dirname(DATABASE_PATH) or '.', exist_ok=True) | current | db.sync_client_example | secure_default | The executable-adjacent default is secured and fails closed instead of teaching unsafe parent creation. |
| P28 | tldw_chatbook/runtime_policy/server_parity_state | build_server_parity_state_repositories | resolved_data_dir.mkdir(parents=True, exist_ok=True) | current | runtime.server_parity_parent | secure_default | Preserve the lexical file-backed repository directory; secure the default and require a trusted custom namespace. |

## Explicit exclusions

| ID | Module | Symbol | Exclusion |
| --- | --- | --- | --- |
| X01 | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | JSONStorage._create_backup | Copies a JSON adapter file, not a SQLite database or sidecar. |
| X02 | tldw_chatbook/DB/Client_Media_DB_v2 | create_incremental_backup | No-op placeholder; it creates no backup artifact. |
| X03 | tldw_chatbook/DB/Client_Media_DB_v2 | create_automated_backup | No-op placeholder; it creates no backup artifact. |
| X04 | production tree | aiosqlite.connect | No production `aiosqlite.connect` owner exists. |

The checked baseline therefore contains exactly 31 direct
`sqlite3.connect` sites across 18 production modules, three direct
`Connection.backup()` sites, and six SQLite database `shutil.copy2()` sites in
`UI/Tools_Settings_Window.py`.
