# Backup recovery owner inventory

Task: TASK-31986. Contract: [ADR-126](../decisions/126-complete-local-backup-and-recovery.md), extending ADR-029/030/036/059/060.

This is a conservative discovery census, **not qualified complete backup coverage**.
Task 3 alone qualified no payload adapter. Task 6 now qualifies the five current core SQLite layouts through the native-held adapter boundary; other owners and Complete product coverage remain pending. Known database
selectors are resolved without opening SQLite. Other owner locators remain
unsupported until their owner cohort extracts a pure canonical resolver. Existing
unclassified children of selected data roots and the two canonical app namespaces
are reported as unknown; they are never copied or recursively followed. Configured
custom data roots do not authorize scanning their parents. A disabled feature and
an absent file do not prove that a store is unused.

All rows below are installed source declarations/review candidates. An unsupported
row blocks a complete claim; it is not an approved exclusion. The source census
includes generic helpers, dynamic `open` modes and non-filesystem `write/connect`
candidates on purpose. These stay conservative until reviewed by their specific
owner. Raw connection counts are supplemented by private writers, writable opens,
file exports, directory creators, model persistence APIs, keyring writers, and
BaseDB subclasses. Calls are inventoried by exact module, qualified symbol,
normalized call name, and count; no module-wide allowlist permits new producers.
Every syntactic bare or attribute `open` call remains a candidate, including
literal read-only forms: ordinary Python bindings can change the callable's meaning.
From-import function aliases are accumulated across the source before visiting calls;
shadowing or a different scope can add candidates but cannot remove them. This census
does not infer runtime signatures or prove that a candidate writes durable data.
Newly retained candidates stay unsupported pending specific owner review; existing
per-symbol exclusions remain explicit. Dynamic reflection/custom
third-party persistence cannot be proven absent by static AST scanning; a new persistence API must extend
the scanner vocabulary as part of owner review. Runtime unknown-root detection is
an additional fail-closed boundary, not a substitute for that review.

## Discovery and shared interfaces

`profile_paths` now owns the canonical database leaf table, default config/data
root selectors, lexical custom selection and username normalization. `config.py`
delegates these selections and retains its normal private-directory verification,
creation and validation. `paths.data_dir` wins over `Paths.data_dir` even when
empty; the uppercase fallback applies only to None/missing. TLDW_CONFIG_PATH wins
over the canonical config default. HOME is read at call time for default data;
XDG_DATA_HOME is intentionally ignored. Shipped legacy database sentinel strings
mean profile-local defaults. Ordinary custom DB spellings retain symlink evidence;
TTS retains its pre-existing resolved-path contract, and scheduled-task tilde
validation retains its pre-existing rejection. Relative paths remain cwd-relative.

`DiscoveryContext(config_path, profile_id)` is frozen local metadata. The reserved
`DISCOVERY_CONTEXT_KEY` is inserted only after parsing a fresh selected TOML;
source collisions are refused. Installed adapters call `discovery_context(config)`
and `storage_logical_id(context, owner, local_id)` to produce explicit final IDs.
The convention is `profile:<profile_id>:<owner>[:<local_id>]`; dependencies use the
same formatter, including references to a different owner. No implicit dependency
rewriting occurs. Adapter output supersedes that owner's unqualified declarations.
Registration grants only installed code authority, not tool/execution permission.

`StorageItem` keeps its five required fields and adds frozen `shared_group=None`
and `deletion_validated=False` metadata. Shared-group declarations require observed
matching device/inode identity; mismatched declarations, undeclared physical aliases,
duplicate IDs, missing dependencies and nested capture roots block completeness.
Same-owner paths shared by selected profiles get an explicit observed shared group.
This inventory is a preview; capture must revalidate identity under maintenance.
The scope digest covers selectors, lexical and observed resolved owner paths,
statuses, dependencies and shared relationships, and intentionally excludes
changing content, inode generations and row counts. Retargeting an alias changes
scope; atomic replacement at the same logical path does not. Capture
budgets remain a later service contract. Task9 adds locally validated selection
context and incorporates external/model/temporary/diagnostic selections plus the
planned output declaration into this digest. Never substitute the inventory digest
for a payload digest or target fingerprint.

`deletion_validated` is trusted installed-owner validation output, never a portable
archive or TOML authority bit. Task 3 has no tombstone validator; source discovery
rejects adapter attempts to emit this evidence until the recovered-media owner is
qualified. Readers must perform fresh owner validation; a serialized boolean can
never authorize payload absence. Unvalidated intentional deletion stays incomplete.

The registration tuple is initially empty: owner adapters qualify in their own
cohorts rather than installing pretend capture implementations. Generic classification
can describe included/unused/excluded/deleted states supplied by qualified installed
owners; source discovery itself currently reports unsupported/missing/unavailable
and reviewed default exclusions. No Complete backup or replacement UI is exposed.

## Owner cohorts and resolver evidence

Each source-symbol row below names its producer. This table records the canonical
resolver location or the explicit unresolved boundary for that cohort. Capture,
validation and relocation are **unsupported for every durable cohort** until its
own adapter evidence exists. Activation is required for all restored operational
owners; passive content never authorizes execution, downloads, sync or rebuilds.

| Cohort | Producers / resolver evidence | Class / dependencies | Capture / validation / relocation / activation |
| --- | --- | --- | --- |
| sqlite | DB constructors and BaseDB subclasses; canonical `config.get_*_db_path` and extracted `profile_paths.database_path`; DB-internal references remain unresolved | Authoritative stores; config and owned asset/reference groups | unsupported / unsupported / unsupported / per-owner review |
| config | config.py `_get_effective_config_path`, `_get_custom_database_path`, `get_user_data_dir`; config history producer symbols below | Settings and credential-bearing history; profile selector and credential cohort | unsupported / unsupported / unsupported / inactive definitions |
| notes | Notes/file_notes_service.py, file_notes_replica.py, note_import_receipts.py, sync_engine.py, sync_paths.py; selected root and DB-bound locators unresolved | File Notes disk authority, independent replica, lasting-sync receipts/recovery; source plus recovery state | unsupported / unsupported / unsupported / paused owner review |
| files | Character_Chat/local_character_persona_service.py `__init__`; local_chat_dictionary_service.py `__init__`; Chat_Grammars and Feedback constructors; app.py `_wire_*` selects stores | Persona/dictionary/grammar/feedback/audio history; config and source DBs | unsupported / unsupported / unsupported / owner review |
| assets | Persona_Visual repository/authoring workspace; Chat/attachment_core.py; Character_Chat visual identity/expression-set producers | References plus payloads; DB-contained locators unresolved | unsupported / unsupported / unsupported / inert payloads |
| tts | TTS/profile_schema.py and profile_repository.py; voice manager constructors; profile migration namespace/publication/recovery; canonical TTS path extracted | Profiles, voice/reference assets and retained recovery journals; one owner dependency group | unsupported / unsupported / unsupported / explicit owner review |
| mcp | MCP local_store.py `_default_local_mcp_store_path`, server_target_store.py `_default_server_targets_path`, unified_context_store.py `_default_unified_mcp_context_path`; permission_store.py constructor | Server definitions, targets, permission/history state; config/credentials | unsupported / unsupported / unsupported / no imported permission authority |
| skills | Skills_Interop/local_skills_service.py `default_local_skills_store_dir`; skill_trust_store.py `default_trust_store_dir` | Local definitions/trust snapshots/markers; script bundles and credential scopes | unsupported / unsupported / unsupported / trust and execution review |
| workspaces | Workspaces/change_tracking.py constructor, change_retention.py; DB/Workspace_DB and AgentRuns_DB constructors, caller-derived sibling agent_runs.db | Workspace metadata, retained Git change snapshots, agent runs; source roots and history | unsupported / unsupported / unsupported / no Git hooks or replay |
| agents | Agents/run_log.py root selection; project_instruction_resolver.py and Chat console context constructors | Run histories and project-instruction bindings; caller-selected roots unresolved | unsupported / unsupported / unsupported / no tool authority |
| rag | RAG_Search/simplified/config.py `_default_persist_directory`, vector_store.py and collection_indexes.py; profile/pipeline loaders | Durable Chroma/vector/index projections and definitions; source DBs plus embedding provenance | unsupported / unsupported / unsupported / reconcile first; never auto-rebuild |
| models | Model_Artifacts/service.py owner-selected catalog roots; fetch/acquisition/leases; TTS/STT/local-engine install call sites | Models optional; recipes/catalog and retained acquisition state may be durable baseline; unresolved separation | unsupported / unsupported / unsupported / inert, no downloads |
| evals | Evals config/template/task loaders and exporters; eval_orchestrator.py canonical DB resolver | Local benchmark definitions/history and caller-selected results | unsupported / unsupported / unsupported / no reruns |
| subscriptions | Subscriptions/site_config_manager.py constructor; briefing_audio.py/briefing_export.py | Site definitions and audio/export artifacts; subscriptions DB, credentials and references | unsupported / unsupported / unsupported / no reconnect |
| runtime | runtime_policy/source_state.py, server_credentials.py, server_parity_state.py constructors | Local event/sync DBs, server selections and credential scopes; local despite server association | unsupported / unsupported / unsupported / fresh identity review |
| miscellaneous (historical census bucket) | Exact source candidates below, not an emitted runtime owner ID; see task9 reconciliation | Installed durable defaults have named adapters; generic caller-selected paths grant no app ownership or traversal | Per-symbol disposition below; unsupported writer admission remains separate from capture qualification |
| generic | Private path/SQLite/file utility writers implement boundaries, not independent storage-root ownership; consuming source symbol selects locator | Inherits caller cohort, never independently captured | unsupported at caller / unsupported / unsupported / inherited |
| external | Reviewed external file tools and browser-cookie clone owners; source-selected external Notes/workspace/model roots | User-owned input or explicit tool output, not baseline app ownership | excluded by default; optional owner qualification required |
| process | Reviewed temporary media/playback/recording/process transport helpers | Session/temporary state; retained recovered-media references require a future durable owner | excluded by default; no execution state restored |
| diagnostics | Reviewed model catalog cache/logging/responsiveness artifacts | Disposable model-list cache or optional logs; not arbitrary directories called cache | excluded by default; optional history remains unsanitized arbitrary content |

Non-DB canonical resolvers currently live in service-heavy modules. This task does
not import those modules or duplicate their filenames as backup-local authorities.
Their observed files are unknown durable entries, alongside explicit unsupported
owner declarations, until the owner cohorts extract pure resolver seams. The
source table records even non-filesystem candidates as unsupported instead of
silently excluding an entire module. That conservative over-count is deliberate.

## Explicit exclusion review

Only exact reviewed source owners below receive non-unsupported classifications.
External browser cookie clones are input, not Chatbook profile data. File tools
operate on user-selected external roots. Video temporary store/recording/playback
and realtime transport do not establish durable profile ownership. Private utility
helpers inherit their caller's ownership, rather than becoming separate stores.
Model-list disk cache is rebuildable metadata; logs and responsiveness artifacts
are optional diagnostics. SQLite `:memory:` is excluded only for that exact target;
an owner that can also receive a filename is not globally excluded. In particular,
the old SQLite inventory's memory policy labels cannot erase the file-backed
server-parity constructor locators visible in current runtime composition.
Server-owned remote content is explicitly excluded in each source inventory;
local server selection/credential/event/sync metadata remains unsupported baseline.
Existing backup archives, active staging and rollback stores must be excluded
through qualified owner declarations; an unfamiliar suffix/root is currently unknown,
not a heuristic exclusion. No directory called cache is automatically disposable.

## SQLite owner registry cross-check

All installed policy IDs, including readers, backup destinations, memory forms and
parent boundaries, are retained here. `policy` is the existing SQLite boundary's
classification, not proof of recovery coverage. Resolver and dependencies belong
to the linked source module and the cohorts above; unresolved mappings remain blocked.

| Registry owner | Source module | Existing policy | Recovery cohort/status |
| --- | --- | --- | --- |
| sqlite:app.prompts_parent | tldw_chatbook/app | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:config.server_sqlite_parent | tldw_chatbook/config | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:config.server_user_db_base | tldw_chatbook/config | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:config.user_data_directory | tldw_chatbook/config | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:cookies.chrome | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | _READ_ONLY_URI | external/excluded |
| sqlite:cookies.edge | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | _READ_ONLY_URI | external/excluded |
| sqlite:cookies.firefox | tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner | _READ_ONLY_URI | external/excluded |
| sqlite:db.base | tldw_chatbook/DB/base_db | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:db.subscriptions.agent_read | tldw_chatbook/DB/Subscriptions_DB | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:db.subscriptions.site_configs | tldw_chatbook/DB/Subscriptions_DB | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:db.chachanotes.backup | tldw_chatbook/DB/ChaChaNotes_DB | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:db.chachanotes.primary | tldw_chatbook/DB/ChaChaNotes_DB | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:db.evals | tldw_chatbook/DB/Evals_DB | _PRIVATE_OR_MEMORY | sqlite/domain-recovery-qualified |
| sqlite:db.library_ingest_jobs | tldw_chatbook/DB/Library_Ingest_Jobs_DB | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:db.media.backup | tldw_chatbook/DB/Client_Media_DB_v2 | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:db.media.integrity | tldw_chatbook/DB/Client_Media_DB_v2 | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:db.media.primary | tldw_chatbook/DB/Client_Media_DB_v2 | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:db.prompts.backup | tldw_chatbook/DB/Prompts_DB | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:db.prompts.primary | tldw_chatbook/DB/Prompts_DB | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:db.rag_indexing | tldw_chatbook/DB/RAG_Indexing_DB | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:eval.orchestrator_parent | tldw_chatbook/Evals/eval_orchestrator | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:kanban.local | tldw_chatbook/Kanban_Interop/local_kanban_db | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:notes.file_notes_replica | tldw_chatbook/Notes/file_notes_replica | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:notes.library_parent | tldw_chatbook/Notes/Notes_Library | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:notes.sync_state | tldw_chatbook/Notes/note_import_receipts | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:notifications.client | tldw_chatbook/Notifications/client_notifications_db | _MEMORY | sqlite/unsupported |
| sqlite:notifications.event_state | tldw_chatbook/Notifications/event_state_repository | _MEMORY | sqlite/unsupported |
| sqlite:rag.chachanotes_keyword_leg | tldw_chatbook/RAG_Search/simplified/rag_service | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:rag.prompts_keyword_leg | tldw_chatbook/RAG_Search/simplified/rag_service | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:research.local | tldw_chatbook/Research_Interop/local_research_service | _PRIVATE_OR_MEMORY | sqlite/domain-recovery-qualified |
| sqlite:runtime.server_parity_parent | tldw_chatbook/runtime_policy/server_parity_state | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:settings.bulk_backup | tldw_chatbook/UI/Tools_Settings_Window | _PRIVATE_AND_READ_ONLY | sqlite/unsupported |
| sqlite:settings.integrity | tldw_chatbook/UI/Tools_Settings_Window | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:settings.pre_restore_backup | tldw_chatbook/UI/Tools_Settings_Window | _PRIVATE_AND_READ_ONLY | sqlite/unsupported |
| sqlite:settings.restore | tldw_chatbook/UI/Tools_Settings_Window | _PRIVATE_AND_READ_ONLY | sqlite/unsupported |
| sqlite:settings.schema | tldw_chatbook/UI/Tools_Settings_Window | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:settings.single_backup | tldw_chatbook/UI/Tools_Settings_Window | _PRIVATE_AND_READ_ONLY | sqlite/unsupported |
| sqlite:settings.vacuum | tldw_chatbook/UI/Tools_Settings_Window | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:sync.notes_mirror | tldw_chatbook/Sync_Interop/notes_mirror | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:sync.state | tldw_chatbook/Sync_Interop/sync_state_repository | _MEMORY | sqlite/unsupported |
| sqlite:tts.profile_store | tldw_chatbook/TTS/profile_schema | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:tts.profile_store_descriptor | tldw_chatbook/TTS/profile_schema | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:tts.profile_candidate | tldw_chatbook/TTS/profile_schema | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:tts.profile_candidate_upgrade | tldw_chatbook/TTS/profile_schema | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:tts.profile_store_version_peek | tldw_chatbook/TTS/profile_schema | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:tts.profile_backup | tldw_chatbook/TTS/profile_repository | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:tts.profile_migration_backup | tldw_chatbook/TTS/profile_repository | _PRIVATE_AND_READ_ONLY | sqlite/unsupported |
| sqlite:tts.profile_migration_boundary | tldw_chatbook/TTS/profile_migration_candidate | _PRIVATE_MEMORY_AND_READ_ONLY | sqlite/unsupported |
| sqlite:tts.profile_migration_publication | tldw_chatbook/TTS/profile_migration_publication | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:tts.profile_migration_recovery | tldw_chatbook/TTS/profile_migration_recovery | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:tts.profile_migration_publication_descriptor | tldw_chatbook/TTS/profile_migration_publication | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:tts.profile_migration_recovery_descriptor | tldw_chatbook/TTS/profile_migration_recovery | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:tts.profile_restore_stage | tldw_chatbook/TTS/profile_repository | _PRIVATE_AND_READ_ONLY | sqlite/unsupported |
| sqlite:tts.profile_recovery | tldw_chatbook/TTS/profile_repository | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:tts.profile_snapshot | tldw_chatbook/TTS/profile_repository | _READ_ONLY_URI | sqlite/unsupported |
| sqlite:tamagotchi.sqlite | tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage | _PRIVATE_OR_MEMORY | sqlite/unsupported |
| sqlite:utils.legacy_user_database_path | tldw_chatbook/Utils/paths | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:utils.project_databases_directory | tldw_chatbook/Utils/paths | _PRIVATE_FILE | sqlite/unsupported |
| sqlite:writing.local | tldw_chatbook/Writing_Interop/local_writing_service | _PRIVATE_OR_MEMORY | sqlite/domain-recovery-qualified |

## Exact persistence producer symbols

A row is a reviewed **source candidate**, not a claim its entire module is durable.
The refreshed census contains 1027 rows. Its qualified symbol is the locator-consumer
evidence; where canonical root selection is not yet extracted, the cohort table explicitly records that resolver
blocker. New calls in an existing symbol also change the expected count and fail.

| Module | Producer symbol | Call | Count | Classification | Resolver / adapter cohort |
| --- | --- | --- | --- | --- | --- |
| tldw_chatbook/Agents/project_instruction_resolver.py | _read_candidate | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Agents/run_log.py | RunLogWriter._write_bytes | open | 1 | qualified | agents.history |
| tldw_chatbook/Agents/run_log.py | RunLogWriter._write_bytes | write | 1 | qualified | agents.history |
| tldw_chatbook/Agents/run_log.py | RunLogWriter.bind | mkdir | 2 | qualified | agents.history |
| tldw_chatbook/Agents/run_log.py | RunLogWriter.bind | write_text | 1 | qualified | agents.history |
| tldw_chatbook/Audio/recording_service.py | AudioRecordingService._pyaudio_recording_loop | open | 1 | process_artifact | process |
| tldw_chatbook/Audio/recording_service.py | AudioRecordingService._save_audio_file | open | 1 | process_artifact | process |
| tldw_chatbook/Audio_Services_Interop/local_audio_services_service.py | LocalAudioServicesService._persist_history | mkdir | 1 | unsupported | files |
| tldw_chatbook/Audio_Services_Interop/local_audio_services_service.py | LocalAudioServicesService._persist_history | write_text | 1 | unsupported | files |
| tldw_chatbook/Backup_Recovery/chat_source_participants.py | sidecar_descriptor | open | 2 | generic_boundary | concrete chat source lifetime; runtime/capture validation pending |
| tldw_chatbook/Backup_Recovery/chat_source_participants.py | write_text | write | 1 | generic_boundary | concrete chat source lifetime; runtime/capture validation pending |
| tldw_chatbook/Backup_Recovery/crypto.py | _open_regular | open | 1 | generic_boundary | generic |
| tldw_chatbook/Backup_Recovery/crypto.py | _write_all | write | 1 | generic_boundary | generic |
| tldw_chatbook/Backup_Recovery/inventory.py | discover | open | 1 | generic_boundary | generic |
| tldw_chatbook/Backup_Recovery/journal.py | Journal._flush_record | open | 1 | generic_boundary | backup_recovery_publication |
| tldw_chatbook/Backup_Recovery/journal.py | Journal._locked | open | 1 | generic_boundary | backup_recovery_journal |
| tldw_chatbook/Backup_Recovery/journal.py | observe_artifact | open | 1 | generic_boundary | backup_recovery_journal |
| tldw_chatbook/Backup_Recovery/journal.py | observe_artifact.read | open | 1 | generic_boundary | backup_recovery_journal |
| tldw_chatbook/Backup_Recovery/journal.py | observe_artifact.recheck | open | 1 | generic_boundary | backup_recovery_journal |
| tldw_chatbook/Backup_Recovery/profile_catalog.py | ProfileCatalog.register | open | 1 | generic_boundary | backup_recovery_profile_catalog |
| tldw_chatbook/Backup_Recovery/publication.py | _flush_original | open | 1 | generic_boundary | backup_recovery_publication |
| tldw_chatbook/Backup_Recovery/publication.py | _retire | open | 1 | generic_boundary | backup_recovery_publication |
| tldw_chatbook/Backup_Recovery/publication.py | _installed_metadata | open | 1 | generic_boundary | backup_recovery_publication |
| tldw_chatbook/Backup_Recovery/publication.py | finalize_candidate | os.unlink | 1 | generic_boundary | backup_control |
| tldw_chatbook/Backup_Recovery/publication.py | _validate_installed | mkdir | 2 | generic_boundary | backup_recovery_publication |
| tldw_chatbook/Backup_Recovery/publication.py | _validate_installed | open | 1 | generic_boundary | backup_recovery_publication |
| tldw_chatbook/Backup_Recovery/publication.py | _validate_installed | write | 1 | generic_boundary | backup_recovery_publication |
| tldw_chatbook/Backup_Recovery/staging.py | _copy | create_private_file | 1 | disposable | backup_restore_staging |
| tldw_chatbook/Backup_Recovery/staging.py | _copy | write | 1 | disposable | backup_restore_staging |
| tldw_chatbook/Backup_Recovery/staging.py | stage_restore | ZipFile | 1 | disposable | backup_restore_staging |
| tldw_chatbook/Backup_Recovery/staging.py | stage_restore | create_private_file | 2 | disposable | backup_restore_staging |
| tldw_chatbook/Backup_Recovery/staging.py | stage_restore | write | 2 | disposable | backup_restore_staging |
| tldw_chatbook/Backup_Recovery/staging.py | stage_restore | write_bytes | 1 | disposable | backup_restore_staging |
| tldw_chatbook/Backup_Recovery/staging.py | stage_restore | write_text | 1 | disposable | backup_restore_staging |
| tldw_chatbook/Backup_Recovery/mcp_source_participants.py | backup_corrupt | os.replace | 2 | generic_boundary | exact five MCP source lifetimes; custom sources ordinary; service/runtime pending |
| tldw_chatbook/Backup_Recovery/mcp_source_participants.py | reader | open | 1 | generic_boundary | exact five MCP source lifetimes; custom sources ordinary; service/runtime pending |
| tldw_chatbook/Backup_Recovery/mcp_source_participants.py | write_json | dump | 2 | generic_boundary | exact five MCP source lifetimes; custom sources ordinary; service/runtime pending |
| tldw_chatbook/Backup_Recovery/mcp_source_participants.py | write_json | mkdir | 1 | generic_boundary | exact five MCP source lifetimes; custom sources ordinary; service/runtime pending |
| tldw_chatbook/Backup_Recovery/mcp_source_participants.py | write_json | open | 1 | generic_boundary | exact five MCP source lifetimes; custom sources ordinary; service/runtime pending |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | export_character_card_to_png | makedirs | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | export_character_card_to_png | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | extract_json_from_image_file | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | import_and_save_character_from_file_with_outcome | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | inspect_character_card_tts_attachment | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | load_character_and_image | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | load_character_card_from_file | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | load_chat_history_from_file_and_save_to_db | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py | export_dictionary_to_file | write | 1 | unsupported | concrete chat source lifetime; runtime/capture validation pending |
| tldw_chatbook/Character_Chat/expression_set_io.py | _candidate_pairs | ZipFile | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | _detect_ext | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | _resolve_vpack_expression_set | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | _valid_image | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | build_expression_set_zip | ZipFile | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | resolve_local_expression_set | ZipFile | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/local_character_persona_service.py | LocalCharacterPersonaService._persist_personas | write_text | 1 | unsupported | concrete chat source lifetime; runtime/capture validation pending |
| tldw_chatbook/Character_Chat/local_chat_dictionary_service.py | LocalChatDictionaryService._persist_history | write_text | 1 | unsupported | concrete chat source lifetime; runtime/capture validation pending |
| tldw_chatbook/Chat/attachment_core.py | process_attachment_bytes | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/chat_conversation_service.py | ChatConversationService._save_rag_context_store | write_text | 1 | unsupported | concrete chat source lifetime; runtime/capture validation pending |

| tldw_chatbook/Chat/citation_trace_identity.py | KeyringCitationFingerprintKeyProvider.provision_key | set_password | 1 | unsupported | runtime.credentials; task14 |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._persist_existing_message | to_json | 3 | memory | serialization-into-db.chachanotes.primary; no-independent-file |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._persist_metadata_only | to_json | 2 | memory | serialization-into-db.chachanotes.primary; no-independent-file |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._persist_new_message | to_json | 3 | memory | serialization-into-db.chachanotes.primary; no-independent-file |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._persist_usage_only | to_json | 1 | memory | serialization-into-db.chachanotes.primary; no-independent-file |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._snapshot_roleplay_message_projection_write | to_json | 1 | memory | serialization-into-db.chachanotes.primary; no-independent-file |
| tldw_chatbook/Chat/console_context_repository.py | ConsoleContextRepository.finish_auxiliary_attempt | to_json | 2 | memory | serialization-into-db.chachanotes.primary; no-independent-file |
| tldw_chatbook/Chat/console_generate_video.py | _stage_pending_video | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_image_view.py | ConsoleImageRenderCache.prepare | open | 1 | memory | PIL-BytesIO-render-cache; no-independent-file |
| tldw_chatbook/Chat/prompt_template_manager.py | load_template | open | 1 | unsupported | files |
| tldw_chatbook/Chat/trajectory_export.py | write_trajectory_export | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/prompt_history.py | PromptHistory._history_io | write | 1 | unsupported | phase7 async source lifetime; runtime composition pending |
| tldw_chatbook/Backup_Recovery/raw_participants.py | _file | open | 2 | generic_boundary | fixed source raw native lifetime; runtime composition pending |
| tldw_chatbook/Backup_Recovery/raw_participants.py | _mkdirs | mkdir | 2 | generic_boundary | fixed source raw native lifetime; runtime composition pending |
| tldw_chatbook/Backup_Recovery/raw_participants.py | _mkdirs | open | 1 | generic_boundary | fixed source raw native lifetime; runtime composition pending |
| tldw_chatbook/Backup_Recovery/raw_participants.py | _remove_temporary | os.unlink | 2 | generic_boundary | fixed source raw native lifetime; runtime composition pending |
| tldw_chatbook/Backup_Recovery/raw_participants.py | _replace | os.replace | 2 | generic_boundary | fixed source raw native lifetime; runtime composition pending |
| tldw_chatbook/Backup_Recovery/raw_participants.py | _unlink | os.unlink | 1 | generic_boundary | fixed source raw native lifetime; runtime composition pending |
| tldw_chatbook/Chat_Grammars_Interop/local_chat_grammars_service.py | LocalChatGrammarsService._persist | dump | 1 | unsupported | phase4 raw source lifetime; runtime composition pending |
| tldw_chatbook/Chunking/chunking_templates.py | ChunkingTemplateManager.save_template | dump | 1 | unsupported | phase4 raw source lifetime; runtime composition pending |
| tldw_chatbook/Feedback_Interop/local_feedback_service.py | LocalFeedbackService._persist | dump | 1 | unsupported | phase4 raw source lifetime; runtime composition pending |
| tldw_chatbook/Notes/template_store.py | merge_templates | dump | 1 | unsupported | notes.templates phase8 raw source lifetime; runtime composition pending |
| tldw_chatbook/Widgets/emoji_picker.py | save_recent_emoji | dump | 1 | unsupported | phase4 raw source lifetime; runtime composition pending |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator.__init__ | secure_private_directory | 2 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._add_character_dependency | dump | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._add_character_dependency | mkdir | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._add_character_dependency | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_characters | dump | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_characters | mkdir | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_characters | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_conversations | dump | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_conversations | mkdir | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_conversations | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_kept_briefings | dump | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_kept_briefings | mkdir | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_kept_briefings | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_media | dump | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_media | mkdir | 2 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_media | open | 2 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_media | write | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_notes | mkdir | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_notes | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_notes | write | 8 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_prompts | dump | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_prompts | mkdir | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_prompts | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_readme | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_readme | write | 33 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_zip_archive | ZipFile | 1 | generic_boundary | chatbooks.archives-default-or-explicit-external-destination; participant-task10 |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_zip_archive | open | 1 | generic_boundary | chatbooks.archives-default-or-explicit-external-destination; participant-task10 |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_zip_archive | write | 1 | generic_boundary | chatbooks.archives-default-or-explicit-external-destination; participant-task10 |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._export_message_attachments | mkdir | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._export_message_attachments | write_bytes | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_conversation_citation_report | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_conversation_citation_report | write | 2 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_kept_briefing_report | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_kept_briefing_report | write | 16 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_message_citation_report_section | write | 15 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator.create_chatbook | dump | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator.create_chatbook | open | 1 | disposable | runtime.chatbook_scratch:data/temp/chatbooks; per-run-finally-cleanup |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter.__init__ | secure_private_directory | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._extract_private_archive | ZipFile | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._extract_private_archive | open | 2 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._extract_private_archive | secure_private_directory | 2 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._extract_private_archive | write | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_characters | open | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_conversations | open | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_kept_briefings | open | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_media | open | 2 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_notes | open | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_prompts | open | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter.import_chatbook | open | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter.preview_chatbook | open | 1 | generic_boundary | runtime.chatbook_scratch:data/temp/imports; external-input-or-extracted-per-run-read; destination-DB-owners |
| tldw_chatbook/Chatbooks/database_paths.py | secure_chatbook_directory | secure_private_directory | 1 | generic_boundary | chatbooks.archives-default-or-explicit-external-destination |
| tldw_chatbook/Chatbooks/local_chatbook_service.py | LocalChatbookService._load_registry | open | 1 | unsupported | chatbooks.registry; participant-task10 |
| tldw_chatbook/Chunking/Chunk_Lib.py | load_document | open | 1 | unsupported | files |
| tldw_chatbook/Chunking/engine/chunker.py | Chunker.chunk_file_stream | open | 1 | unsupported | files |
| tldw_chatbook/Coding/code_mapper.py | SimpleIO.read_text | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/DB/AgentRuns_DB.py | AgentRunsDB | inherits:BaseDB | 1 | qualified | db.agent_runs |
| tldw_chatbook/DB/ChaChaNotes_DB.py | CharactersRAGDB._get_thread_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/ChaChaNotes_DB.py | CharactersRAGDB.backup_database | backup_connection_to_private | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Client_Media_DB_v2.py | MediaDatabase._get_thread_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Client_Media_DB_v2.py | MediaDatabase.backup_database | backup_connection_to_private | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Client_Media_DB_v2.py | check_database_integrity | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Client_Media_DB_v2.py | import_obsidian_note_to_db | dump | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Evals_DB.py | EvalsDB._get_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Library_Collections_DB.py | LibraryCollectionsDB | inherits:BaseDB | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Library_Ingest_Jobs_DB.py | LibraryIngestJobsDB | inherits:BaseDB | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Library_Ingest_Jobs_DB.py | LibraryIngestJobsDB._get_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Prompts_DB.py | PromptsDatabase._get_thread_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Prompts_DB.py | PromptsDatabase.backup_database | backup_connection_to_private | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Prompts_DB.py | export_prompt_keywords_to_csv | open | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Prompts_DB.py | export_prompts_formatted | ZipFile | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Prompts_DB.py | export_prompts_formatted | open | 2 | unsupported | sqlite |
| tldw_chatbook/DB/Prompts_DB.py | export_prompts_formatted | write | 2 | unsupported | sqlite |
| tldw_chatbook/DB/RAG_Indexing_DB.py | RAGIndexingDB._get_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Subscriptions_DB.py | SubscriptionsDB | inherits:BaseDB | 1 | qualified | db.subscriptions |
| tldw_chatbook/DB/Subscriptions_DB.py | SubscriptionsDB._get_connection | connect_private_sqlite | 1 | qualified | db.subscriptions |
| tldw_chatbook/DB/Subscriptions_DB.py | ensure_site_configs_schema | connect_private_sqlite | 1 | qualified | db.subscriptions |
| tldw_chatbook/DB/Workspace_DB.py | WorkspaceDB | inherits:BaseDB | 1 | qualified | db.workspaces |
| tldw_chatbook/DB/base_db.py | BaseDB._get_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/Evals/config_loader.py | EvalConfigLoader.save | dump | 1 | generic_boundary | eval.definitions phase8 exact default source; explicit exports ordinary; runtime composition pending |
| tldw_chatbook/Evals/dataset_loader.py | DatasetLoader._load_csv_dataset | open | 1 | external_input | evals |
| tldw_chatbook/Evals/dataset_loader.py | DatasetLoader._load_json_dataset | open | 1 | external_input | evals |
| tldw_chatbook/Evals/dataset_validator.py | DatasetValidator._load_dataset | open | 3 | external_input | evals |
| tldw_chatbook/Evals/eval_orchestrator.py | EvaluationOrchestrator.create_task_from_template | mkdir | 1 | external_input | evals |
| tldw_chatbook/Evals/eval_orchestrator.py | EvaluationOrchestrator.export_results | dump | 1 | external_input | evals |
| tldw_chatbook/Evals/eval_orchestrator.py | EvaluationOrchestrator.export_results | open | 2 | external_input | evals |
| tldw_chatbook/Evals/eval_orchestrator.py | quick_eval | mkdir | 1 | external_input | evals |
| tldw_chatbook/Evals/eval_runner.py | DatasetLoader._load_csv_dataset | open | 1 | external_input | evals |
| tldw_chatbook/Evals/eval_runner.py | DatasetLoader._load_json_dataset | open | 1 | external_input | evals |
| tldw_chatbook/Evals/eval_templates.py | EvalTemplateManager.create_sample_dataset | dump | 1 | external_input | evals |
| tldw_chatbook/Evals/eval_templates.py | EvalTemplateManager.create_sample_dataset | open | 1 | external_input | evals |
| tldw_chatbook/Evals/eval_templates.py | EvalTemplateManager.export_template_as_file | dump | 2 | external_input | evals |
| tldw_chatbook/Evals/eval_templates.py | EvalTemplateManager.export_template_as_file | open | 2 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_ab_test_csv | open | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_ab_test_json | dump | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_ab_test_json | open | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_ab_test_latex | open | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_ab_test_latex | write | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_ab_test_markdown | open | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_ab_test_markdown | write | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_run_csv | open | 2 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_run_json | dump | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_run_json | open | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_run_markdown | open | 1 | external_input | evals |
| tldw_chatbook/Evals/exporters.py | EvaluationExporter._export_run_markdown | write | 1 | external_input | evals |
| tldw_chatbook/Evals/specialized_runners.py | CodeExecutionRunner._execute_code | write_text | 1 | process_artifact | evals |
| tldw_chatbook/Evals/task_loader.py | TaskLoader._detect_file_format | open | 1 | external_input | evals |
| tldw_chatbook/Evals/task_loader.py | TaskLoader._detect_format | open | 1 | external_input | evals |
| tldw_chatbook/Evals/task_loader.py | TaskLoader._load_csv_task | open | 1 | external_input | evals |
| tldw_chatbook/Evals/task_loader.py | TaskLoader._load_custom_task | open | 1 | external_input | evals |
| tldw_chatbook/Evals/task_loader.py | TaskLoader._load_eleuther_task | open | 1 | external_input | evals |
| tldw_chatbook/Evals/task_loader.py | TaskLoader._load_huggingface_task | open | 1 | external_input | evals |
| tldw_chatbook/Evals/task_loader.py | TaskLoader.export_task | dump | 4 | external_input | evals |
| tldw_chatbook/Evals/task_loader.py | TaskLoader.export_task | open | 3 | external_input | evals |
| tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py | ChatImageHandler.get_image_info | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py | ChatImageHandler.prepare_image_payload | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py | ChatImageHandler.validate_image_data | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py | handle_start_llamacpp_server_button_pressed | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py | handle_start_llamafile_server_button_pressed | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_mlx_lm.py | handle_start_mlx_server_button_pressed | write | 2 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_copy_model_button_pressed | write | 4 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_create_model_button_pressed | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_delete_model_button_pressed | write | 6 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_embeddings_button_pressed | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_list_models_button_pressed | write | 5 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_ps_button_pressed | write | 4 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_pull_model_button_pressed | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_push_model_button_pressed | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_show_model_button_pressed | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py | handle_ollama_start_service_button_pressed | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_onnx.py | handle_start_onnx_server_button_pressed | write | 2 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py | handle_transformers_list_local_models_button_pressed | write | 5 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py | handle_start_vllm_server_button_pressed | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py | STTSEventHandler._deliver_generation_failure.deliver | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py | STTSEventHandler._deliver_generation_success.deliver | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py | STTSEventHandler._record_successful_sample_evidence | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py | STTSEventHandler._update_generation_progress.update | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py | STTSEventHandler.handle_audiobook_generate | write | 5 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py | STTSEventHandler.handle_audiobook_generate.progress_callback | write | 4 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py | TTSEventHandler._append_tts_artifact_chunk | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py | TTSEventHandler._append_tts_artifact_chunk | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py | TTSEventHandler._stream_response_via_sink | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/notes_events.py | load_note_templates | open | 1 | external_input | shipped immutable fallback template read; user source uses Notes.template_store |
| tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py | ComfyUIImageAdapter._download_output | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py | _load_packaged_workflow | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Image_Generation/adapters/image_format_utils.py | maybe_convert_format | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Image_Generation/request_validation.py | _validate_reference_image_content | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_db.py | open_connection | connect_private_sqlite | 1 | qualified | kanban.local |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.connection | connect | 1 | generic_boundary | phase6 exact service scope; native borrowers retained |
| tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py | save_summary_to_file | makedirs | 1 | unsupported | miscellaneous |
| tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py | save_summary_to_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py | save_summary_to_file | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py | save_summary_to_file | makedirs | 1 | unsupported | miscellaneous |
| tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py | save_summary_to_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py | save_summary_to_file | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/LLM_Calls/Summarization_General_Lib.py | extract_metadata_and_content | open | 1 | unsupported | files |
| tldw_chatbook/LLM_Calls/Summarization_General_Lib.py | extract_text_from_input | open | 1 | unsupported | files |
| tldw_chatbook/LLM_Calls/realtime/openai_session.py | OpenAIRealtimeSession.connect | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/LLM_Calls/realtime/transport.py | WsTransport.connect | connect | 1 | process_artifact | process |
| tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py | ModelCatalogDiskStore.load_into | open | 1 | disposable | diagnostics |
| tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py | ModelCatalogDiskStore.save | mkdir | 1 | disposable | diagnostics |
| tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py | ModelCatalogDiskStore.save | write_bytes | 1 | disposable | diagnostics |
| tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py | _process_markup_or_plain_text | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py | ingest_text_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py | process_mobi | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py | process_zip_of_epubs | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Document_Processing_Lib.py | process_rtf | open | 1 | unsupported | files |
| tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py | extract_images_from_pdf | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py | extract_images_from_pdf | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py | extract_visual_features | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py | preprocess_image_for_ocr | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py | process_image | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/OCR_Backends.py | DocextOCRBackend.process_image | open | 3 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/OCR_Backends.py | DocextOCRBackend.process_pdf | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/OCR_Backends.py | DocextOCRBackend.process_pdf | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/OCR_Backends.py | EasyOCRBackend.process_image | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/OCR_Backends.py | PaddleOCRBackend.process_image | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/OCR_Backends.py | TesseractOCRBackend.process_image | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py | extract_metadata_from_pdf | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py | extract_text_and_format_from_pdf | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py | process_pdf | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py | process_pdf | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/audio_processing.py | LocalAudioProcessor._process_single_audio | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/audio_processing.py | LocalAudioProcessor.download_audio_file | mkdir | 2 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/audio_processing.py | LocalAudioProcessor.download_audio_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/diarization_service.py | _lazy_import_silero_vad | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/parakeet_v2_installer.py | _sha256 | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/parakeet_v2_installer.py | install_verified_parakeet_v2 | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/parakeet_v2_installer.py | install_verified_parakeet_v2 | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/parakeet_v2_installer.py | install_verified_parakeet_v2 | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/parakeet_v2_installer.py | install_verified_parakeet_v2 | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend._transcribe_buffer_with_parakeet_mlx | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend._transcribe_with_parakeet_mlx | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend._transcribe_with_parakeet_onnx | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend._transcribe_with_remote_whisper | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend.transcribe_buffer | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend.transcribe_buffer | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _has_known_parakeet_v2_receipt | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/video_processing.py | LocalVideoProcessor._process_single_video | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/video_processing.py | LocalVideoProcessor._write_temp_cookiefile | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Logging_Config.py | PrivateRotatingFileHandler.__init__ | secure_private_directory | 1 | diagnostics | diagnostics |
| tldw_chatbook/Logging_Config.py | PrivateRotatingFileHandler._harden_existing_generations | open_private_binary | 1 | diagnostics | diagnostics |
| tldw_chatbook/Logging_Config.py | PrivateRotatingFileHandler._open | open_private_text_append_stream | 1 | diagnostics | diagnostics |
| tldw_chatbook/Logging_Config.py | RichLogHandler._process_log_queue | write | 2 | diagnostics | diagnostics |
| tldw_chatbook/MCP/client.py | _StdioJSONRPCConnection._send_message | write | 1 | process_artifact | process-transport |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog._migrate_generation | atomic_private_write_bytes | 1 | qualified | mcp.history |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog._read_bytes | open_private_binary | 1 | qualified | mcp.history |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog._secure_parent | secure_private_directory | 1 | qualified | mcp.history |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog.append | atomic_private_write_bytes | 2 | qualified | mcp.history |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog.append | open_private_text_append | 1 | qualified | mcp.history |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog.append | write | 1 | qualified | mcp.history |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._build_reading_export_response | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._extract_ebook_text | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._extract_pdf_text | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._hash_local_ingestion_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._load_reading_import_csv | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._sync_archive_snapshot_source_items | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._sync_archive_snapshot_source_items | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/generation_templates.py | _load_directory_templates | open | 1 | unsupported | files |
| tldw_chatbook/Media_Creation/image_generation_service.py | ImageGenerationService._setup_output_directory | mkdir | 3 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/image_generation_service.py | ImageGenerationService.generate_custom | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/image_generation_service.py | ImageGenerationService.generate_custom | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/image_generation_service.py | ImageGenerationService.initialize | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/swarmui_client.py | SwarmUIClient.__aenter__ | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/swarmui_client.py | SwarmUIClient.generate_image | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/swarmui_client.py | SwarmUIClient.get_image | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/swarmui_client.py | SwarmUIClient.get_models | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/swarmui_client.py | SwarmUIClient.get_session | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Creation/swarmui_client.py | SwarmUIClient.health_check | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Media_Playback/frame_source.py | AvFrameSource._open | open | 1 | process_artifact | process |
| tldw_chatbook/Model_Artifacts/acquisition.py | ArtifactAcquisitionService._fetch_artifact | mkdir | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/acquisition.py | ArtifactAcquisitionService._fetch_one_file | mkdir | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/acquisition.py | ArtifactAcquisitionService._hash_staged_file | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/acquisition.py | ArtifactAcquisitionService._reconcile_durable_bytes | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/fetch.py | stream_fetch | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/fetch.py | stream_fetch | write | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/gguf_admission.py | open_local_gguf | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/leases.py | ArtifactOperationLease._acquire_native_until | mkdir | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/leases.py | ArtifactOperationLease._acquire_native_until | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._copy_payload | mkdir | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._copy_payload | open | 2 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._copy_payload | write | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._create_install_staging | mkdir | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._download_stage_for | mkdir | 2 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._download_stage_ownership | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._ensure_owned_directory | mkdir | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._read_active | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._read_download_stage_marker | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._read_manifest | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._read_readiness | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._validate_download_stage_state.scan | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._verify_payload | open | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService.import_local_gguf | open | 2 | unsupported | models |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService.import_local_gguf | write | 1 | unsupported | models |
| tldw_chatbook/Models/evaluation_state.py | EvaluationState.load_from_file | open | 1 | external_input | evals |
| tldw_chatbook/Models/evaluation_state.py | EvaluationState.save_to_file | dump | 1 | external_input | evals |
| tldw_chatbook/Models/evaluation_state.py | EvaluationState.save_to_file | open | 1 | external_input | evals |
| tldw_chatbook/Notes/file_notes_git_network.py | _LayoutBuilder.directory | mkdir | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_network.py | _LayoutBuilder.file | open | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_network.py | _LayoutBuilder.file | write | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_network.py | _capture_ssh_trust_source | open | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_network.py | _file_digest | open | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_service.py | AsyncGitProcessRunner._write_process_stdin | write | 1 | process_artifact | process-transport |
| tldw_chatbook/Notes/file_notes_git_service.py | FileNotesGitService._commit_local_state_is_supported | open | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory._capture_entry | open | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory.capture_index | open | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory.create_directory | mkdir | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory.create_file | open | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory.create_file | write | 1 | generic_boundary | external-git-source-or-disposable-private-proof |
| tldw_chatbook/Notes/file_notes_replica.py | FileNotesReplica._get_connection | connect_private_sqlite | 1 | qualified | notes.file_notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.create_file | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.create_file | write | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.export_exact_file | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.export_exact_file | write | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.restore_file | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.restore_file | write | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_copy | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_copy | write | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_file | write | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | _read_regular_file | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/git_process_containment.py | _WindowsAsyncChildProcess.communicate.write_input | write | 1 | process_artifact | process-transport |
| tldw_chatbook/Notes/note_import_discovery.py | _inspect_selected_path | open | 2 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/note_import_discovery.py | _open_verified_directory | open | 2 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/note_import_discovery.py | _read_discovered_source_posix | open | 3 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/note_import_discovery.py | _scan_child_directory | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/note_import_discovery.py | _verify_lexical_source_binding | open | 2 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/note_import_receipts.py | NoteImportReceiptRepository._connect | connect_private_sqlite | 1 | qualified | notes.sync_state |
| tldw_chatbook/Notes/sync_engine.py | NotesSyncEngine._write_file_info | write_text | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot.__enter__ | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot._read_file | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot._verified_child_directory | mkdir | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot._verified_child_directory | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot._write_all | write | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot.write_text | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notifications/client_notifications_db.py | ClientNotificationsDB | inherits:BaseDB | 1 | qualified | notifications.client |
| tldw_chatbook/Notifications/client_notifications_db.py | ClientNotificationsDB._get_connection | connect_private_sqlite | 1 | qualified | notifications.client |
| tldw_chatbook/Notifications/event_state_repository.py | EventStateRepository | inherits:BaseDB | 1 | qualified | runtime.event_state |
| tldw_chatbook/Notifications/event_state_repository.py | EventStateRepository._get_connection | connect_private_sqlite | 1 | qualified | runtime.event_state |
| tldw_chatbook/Persona_Visual/runtime.py | _validated_portrait | open | 1 | unsupported | assets |
| tldw_chatbook/Prompt_Management/Prompts_Interop.py | <module> | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Prompt_Management/Prompts_Interop.py | <module> | write | 2 | unsupported | miscellaneous |
| tldw_chatbook/Prompt_Management/Prompts_Interop.py | import_prompts_from_files | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager.__init__ | mkdir | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager._load_custom_profiles | open | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager._migrate_legacy_blob | dump | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager._migrate_legacy_blob | open | 2 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager._save_one | dump | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager._save_one | open | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager.end_experiment | dump | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager.end_experiment | open | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager.start_experiment | dump | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager.start_experiment | mkdir | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager.start_experiment | open | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/eval/gating.py | GatingConfig.to_yaml | dump | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/eval/gating.py | GatingConfig.to_yaml | mkdir | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/eval/gating.py | GatingConfig.to_yaml | write_text | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/eval/regression.py | RegressionDetector._ensure_dir | mkdir | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/eval/regression.py | RegressionDetector._save_atomic | dump | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/pipeline_builder_simple.py | load_pipelines_from_toml | copy2 | 1 | qualified | rag |
| tldw_chatbook/RAG_Search/pipeline_builder_simple.py | load_pipelines_from_toml | mkdir | 1 | qualified | rag |
| tldw_chatbook/RAG_Search/pipeline_builder_simple.py | load_pipelines_from_toml | open | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.export_pipeline_config | dump | 1 | qualified | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.export_pipeline_config | open | 1 | qualified | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.load_pipeline_config | copy2 | 1 | qualified | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.load_pipeline_config | mkdir | 1 | qualified | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.load_pipeline_config | open | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/simplified/collection_indexes.py | _client | PersistentClient | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/simplified/rag_service.py | RAGService._connect_chacha_readonly | connect_private_sqlite | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/simplified/rag_service.py | RAGService._connect_prompts_readonly | connect_private_sqlite | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/simplified/vector_store.py | ChromaVectorStore.__init__ | mkdir | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/simplified/vector_store.py | ChromaVectorStore.client | PersistentClient | 1 | unsupported | rag |
| tldw_chatbook/Research_Interop/local_research_service.py | LocalResearchService._connect | connect_private_sqlite | 2 | unsupported | miscellaneous |
| tldw_chatbook/STT/parakeet_external.py | ExternalParakeetVerifier._verify_uncached | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/STT/parakeet_onnx.py | _wav_duration | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/STT/transcribe_cpp.py | _read_normalized_wav | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Scheduling/db/scheduled_tasks_db.py | ScheduledTasksDB | inherits:BaseDB | 1 | qualified | db.scheduled_tasks |
| tldw_chatbook/Skills_Interop/atomic_write.py | write_bytes_atomic | write_bytes | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/atomic_write.py | write_text_atomic | write_text | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService._load_index | open | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService._plan_for_script | open | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService._read_zip_member_bounded | open | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService._save_index | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService._script_output_root | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService._script_scratch_root | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService._write_bytes_atomic | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService._write_text_atomic | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService.create_skill | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService.export_skill | ZipFile | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService.import_skill | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService.import_skill_directory | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/local_skills_service.py | LocalSkillsService.import_skill_file | ZipFile | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/project_skills_discovery.py | _entry_for | open | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/project_skills_prompt.py | ProjectSkillsPromptLedger.record | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/project_skills_prompt.py | ProjectSkillsPromptLedger.record | write_text | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/skill_remote_fetch.py | re_root_skill_zip | ZipFile | 2 | unsupported | skills |
| tldw_chatbook/Skills_Interop/skill_trust_store.py | KeyringSkillTrustGenerationMarkerStore.save_marker | set_password | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/skill_trust_store.py | KeyringSkillTrustKeyCache.save_keys | set_password | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/skill_trust_store.py | _atomic_write_bytes | write_bytes | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/skill_trust_store.py | _atomic_write_json | write_text | 1 | unsupported | skills |
| tldw_chatbook/Skills_Interop/skill_trust_store.py | _ensure_trust_directory | mkdir | 1 | unsupported | skills |
| tldw_chatbook/Study_Interop/local_study_service.py | LocalStudyService.import_flashcards_json_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Subscriptions/briefing_audio.py | _looks_like_wav | open | 1 | qualified | subscriptions.assets |
| tldw_chatbook/Subscriptions/briefing_audio.py | briefing_audio_dir | secure_private_directory | 1 | qualified | subscriptions.assets |
| tldw_chatbook/Subscriptions/briefing_export.py | _copy_episode_audio_file | open | 2 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Subscriptions/briefing_export.py | _write_feed_xml_atomically | open | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Subscriptions/briefing_export.py | _write_feed_xml_atomically | write | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Subscriptions/site_config_manager.py | SiteConfigManager.export_configs | write_text | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Sync_Interop/notes_mirror.py | NotesMirror.__init__ | connect_private_sqlite | 1 | memory | dormant-default-memory-no-installed-callers |
| tldw_chatbook/Sync_Interop/sync_state_repository.py | SyncStateRepository | inherits:BaseDB | 1 | qualified | runtime.sync_state |
| tldw_chatbook/Sync_Interop/sync_state_repository.py | SyncStateRepository._get_connection | connect_private_sqlite | 1 | qualified | runtime.sync_state |
| tldw_chatbook/TTS/TTS_Generation.py | TTSService._commit_voice_setup_default | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/TTS_Generation.py | TTSService._prepared_provider_read | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/TTS_Generation.py | TTSService._restart_audio_cpp | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/TTS_Generation.py | TTSService._run_preferences_publication | write | 2 | unsupported | tts |
| tldw_chatbook/TTS/TTS_Generation.py | TTSService._transition_audio_cpp_lifecycle | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_artifact_catalog.py | load_audio_cpp_artifact_source_manifest | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | AudioCppGeneratedLaunchArtifact.validate | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | _create_artifact | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | _create_artifact | open | 3 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | _create_artifact | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | _create_artifact | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_managed_config.py | _read_server_json | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_package_scanner.py | _inspect_file | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_player.py | SimpleAudioPlayer.pause | connect | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_player.py | SimpleAudioPlayer.pause | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_player.py | SimpleAudioPlayer.resume | connect | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_player.py | SimpleAudioPlayer.resume | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_service.py | AudioService._convert_with_soundfile | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_service.py | AudioService.create_m4b_with_chapters | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/audiobook_generator.py | AudioBookGenerator._combine_chapters | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/audiobook_generator.py | AudioBookGenerator._combine_chapters | write | 2 | unsupported | tts |
| tldw_chatbook/TTS/audiobook_generator.py | AudioBookGenerator._generate_chapter_audio | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/audiobook_generator.py | AudioBookGenerator._generate_chapter_audio | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend.__init__ | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend._combine_audio_with_crossfade | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend._send_command | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend._tensor_to_wav_bytes | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend._transcribe_audio | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend.list_voices_with_metadata | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend.save_reference_voice | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend.save_reference_voice_with_metadata | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend.save_reference_voice_with_metadata | dump | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend.save_reference_voice_with_metadata | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | suppress_output | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_isolated.py | ChatterboxIsolatedBackend._read_response | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_isolated.py | ChatterboxIsolatedBackend._send_command | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_process.py | <module> | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_process.py | main | open | 5 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_process.py | send_response | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.create_profile | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.create_profile | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.export_profile | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.export_profile | dump | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.export_profile | mkdir | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.export_profile | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.export_profile | write | 7 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.import_profile | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.import_profile | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.import_profile | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.load_profiles | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.save_profiles | dump | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox_voice_manager.py | ChatterboxVoiceManager.save_profiles | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs.py | HiggsAudioTTSBackend.__init__ | mkdir | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs.py | HiggsAudioTTSBackend._load_voice_profiles | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs.py | HiggsAudioTTSBackend._prepare_messages | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs.py | HiggsAudioTTSBackend._save_voice_profiles | dump | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs.py | HiggsAudioTTSBackend._save_voice_profiles | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs.py | HiggsAudioTTSBackend.create_voice_profile | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs.py | HiggsAudioTTSBackend.create_voice_profile | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.__init__ | mkdir | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager._create_backup | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.create_profile | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.create_profile | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.export_profile | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.export_profile | dump | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.export_profile | mkdir | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.export_profile | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.export_profile | write | 6 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.import_profile | copy2 | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.import_profile | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.import_profile | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.load_profiles | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.restore_from_backup | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.save_profiles | dump | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/higgs_voice_manager.py | HiggsVoiceProfileManager.save_profiles | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend.__init__ | secure_private_directory | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._download_model_if_needed | makedirs | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._download_model_if_needed | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._download_model_if_needed | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._download_voice_if_needed | makedirs | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._download_voice_if_needed | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._download_voice_if_needed | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._generate_onnx_with_timestamps | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._generate_pytorch_with_timestamps | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._initialize_onnx | makedirs | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._initialize_onnx | write | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend._load_saved_blends | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/backends/kokoro.py | KokoroTTSBackend.initialize | mkdir | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/voice_manager_base.py | VoiceManagerBase.__init__ | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_native.py | _native_open | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _append_journal | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _immutable_validate | connect_private_sqlite_descriptor | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _prepare_parent | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _write_new_journal | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_recovery.py | _validate_authoritative_targets | connect_private_sqlite_descriptor | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_recovery.py | recover_profile_migration_publication | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_audio.py | _read_regular_source | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _clone_open | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _create_materialization_sync | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _create_materialization_sync | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _prepare_runtime_root_sync | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_storage.py | write_reference_blob | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | _BackupNativeState.fsync_file | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_backup_to | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_backup_to | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_create_recovery_backup | backup_open_connections_to_private | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_create_recovery_backup | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_create_recovery_backup | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_exact_schema_version | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_online_backup | backup_open_connections_to_private | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_publish_migrated_store | connect_private_sqlite | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_schema_version_for_restore | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_validate_restore_source | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_validate_standalone_snapshot | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | _fsync_directory | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | _fsync_file | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _CandidateValidationJob.pin_parent | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _copy_source_to_snapshot | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _open_candidate_source | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | open_exact_current_profile_store | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | open_exact_current_profile_store | connect_private_sqlite_descriptor | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _ExactCurrentProfileConnection.open_descriptor | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | open_profile_store | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | peek_profile_store_schema_version | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _validate_profile_candidate | connect_private_sqlite | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_store_lock.py | ProfileStoreLease.acquire | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/sample_audio_validation.py | _read_bounded_regular_file | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/sample_audio_validation.py | compressed_audio_has_decodable_frame | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/sample_audio_validation.py | wav_has_complete_frames | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/utils/download_models.py | ModelDownloader.__init__ | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/utils/download_models.py | ModelDownloader.download_file | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/utils/download_models.py | ModelDownloader.download_file | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/utils/download_models.py | ModelDownloader.download_file | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/utils/download_models.py | ModelDownloader.download_kokoro_model | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_blend_paths.py | write_private_json | atomic_private_write_text | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_codec.py | _encode_bundle | ZipFile | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_codec.py | _stream_member | write | 2 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_codec.py | _stream_member.write | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _bundle_open | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _cleanup_operation | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _copy_and_inspect | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _create_operation | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _prepare_root_sync | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _publish_sync | write | 1 | unsupported | tts |
| tldw_chatbook/Third_Party/aider/waiting.py | Spinner._supports_unicode | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Third_Party/aider/waiting.py | Spinner.end | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Third_Party/aider/waiting.py | Spinner.step | write | 2 | unsupported | miscellaneous |
| tldw_chatbook/Tools/_grep_worker.py | main | dump | 3 | unsupported | miscellaneous |
| tldw_chatbook/Tools/_grep_worker.py | run_search | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Tools/file_operation_tools.py | WriteFileTool.execute | mkdir | 1 | external_input | external |
| tldw_chatbook/Tools/file_operation_tools.py | WriteFileTool.execute | open | 2 | external_input | external |
| tldw_chatbook/Tools/file_operation_tools.py | WriteFileTool.execute | write | 2 | external_input | external |
| tldw_chatbook/Tools/file_operation_tools.py | _tool_sandbox_root | mkdir | 1 | external_input | external |
| tldw_chatbook/Tools/local_tool_impls.py | edit_file | open | 1 | unsupported | external |
| tldw_chatbook/Tools/local_tool_impls.py | edit_file | write_bytes | 1 | external_input | external |
| tldw_chatbook/Tools/local_tool_impls.py | read_file | open | 1 | unsupported | external |
| tldw_chatbook/Tools/local_tool_impls.py | write_file | write_bytes | 1 | external_input | external |
| tldw_chatbook/Tools/patch_tool_impls.py | patch_files | open | 1 | unsupported | external |
| tldw_chatbook/Tools/patch_tool_impls.py | patch_files | write_bytes | 1 | external_input | external |
| tldw_chatbook/Tools/web_tool_impls.py | _describe_archive | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/Tools/web_tool_impls.py | _describe_image | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Tools/web_tool_impls.py | _extract_pdf_text | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Chatbooks_Window.py | ChatbooksWindow._refresh_chatbooks | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Chatbooks_Window_Improved.py | ChatbooksWindowImproved._scan_chatbooks | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/CodeRepoCopyPasteWindow.py | CodeRepoCopyPasteWindow._export_to_zip_worker | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/CodeRepoCopyPasteWindow.py | CodeRepoCopyPasteWindow._export_to_zip_worker | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/CodeRepoCopyPasteWindow.py | CodeRepoCopyPasteWindow._export_to_zip_worker | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/CodeRepoCopyPasteWindow.py | CodeRepoCopyPasteWindow.generate_compilation._read_selected_files | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/CodeRepoCopyPasteWindow.py | CodeRepoCopyPasteWindow.handle_node_selected._read_local_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Console_Modules/image.py | ConsoleImageController._h3_reference_from_snapshot | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Console_Modules/message.py | ConsoleMessageController._save_console_message_image._write_images_to_disk | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Console_Modules/message.py | ConsoleMessageController._save_console_message_image._write_images_to_disk | write_bytes | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Console_Modules/message.py | ConsoleMessageController._serialize_console_message | to_json | 3 | unsupported | miscellaneous |
| tldw_chatbook/UI/Console_Modules/video.py | ConsoleVideoController._copy_pending_video_external | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Console_Modules/video.py | ConsoleVideoController._copy_pending_video_external | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/UI/Console_Modules/video.py | ConsoleVideoController._save_console_video_copy._copy_to_disk | copy2 | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Console_Modules/video.py | ConsoleVideoController._save_console_video_copy._copy_to_disk | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Dictation_Window_Improved.py | ImprovedDictationWindow._export_as_markdown | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Dictation_Window_Improved.py | ImprovedDictationWindow._export_as_markdown | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Dictation_Window_Improved.py | ImprovedDictationWindow._export_as_markdown | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Dictation_Window_Improved.py | ImprovedDictationWindow._export_as_text | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Dictation_Window_Improved.py | ImprovedDictationWindow._export_as_text | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Dictation_Window_Improved.py | ImprovedDictationWindow._export_as_text | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Evals/results_grid.py | ResultsGrid._write_export_file | write_text | 2 | unsupported | miscellaneous |
| tldw_chatbook/UI/LLM_Management_Window.py | LLMManagementWindow._populate_help_text | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/UI/Logs_Window.py | LogsWindow._render_view | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Logs_Window.py | LogsWindow.append_record | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/STTS_Window.py | AudioBookGenerationWidget._generate_audiobook | write | 4 | unsupported | miscellaneous |
| tldw_chatbook/UI/STTS_Window.py | AudioBookGenerationWidget._handle_export_location | copy2 | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/STTS_Window.py | AudioBookGenerationWidget._handle_file_selection | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/STTS_Window.py | AudioBookGenerationWidget._preview_chapter_audio | write | 2 | unsupported | miscellaneous |
| tldw_chatbook/UI/STTS_Window.py | AudioBookGenerationWidget._update_voice_options | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/STTS_Window.py | AudioBookGenerationWidget.audiobook_generation_complete | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/chat_screen.py | ChatScreen._begin_console_realtime_reply_audio | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/chat_screen.py | ChatScreen._connect_console_realtime | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/chat_screen.py | ChatScreen._write_sidebar_state_snapshot | dump | 1 | unsupported | phase7 async source lifetime; runtime composition pending |
| tldw_chatbook/UI/Screens/image_gen_demo_screen.py | ImageGenDemoScreen._render_result | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/library_screen.py | LibraryScreen._write_library_note_export_file | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/library_screen.py | LibraryScreen._write_library_prompt_export_file | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/settings_config_adapter.py | SettingsConfigAdapter.validate_config_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/settings_image_gen_defaults.py | load_user_image_generation_table | open | 1 | unsupported | files |
| tldw_chatbook/UI/Screens/settings_screen.py | SettingsScreen._handle_about_link | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/settings_video_gen_defaults.py | load_user_video_generation_table | open | 1 | unsupported | files |
| tldw_chatbook/UI/Screens/watchlists_collections_screen.py | WatchlistsCollectionsScreen._open_item_in_browser | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py | AudioCppRuntimeCard.apply_observation | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_playback_mixin.py | SpeechPlaybackMixin._generation_complete | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_playback_mixin.py | SpeechPlaybackMixin._handle_audio_export | copy2 | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_playback_mixin.py | SpeechPlaybackMixin._store_delivered_artifact | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_settings_mixin.py | SpeechSettingsMixin._export_voice_blends | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_settings_mixin.py | SpeechSettingsMixin._handle_export_file | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_settings_mixin.py | SpeechSettingsMixin._handle_export_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_settings_mixin.py | SpeechSettingsMixin._handle_import_file | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_settings_mixin.py | SpeechSettingsMixin._load_kokoro_voice_blends | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_settings_mixin.py | SpeechSettingsMixin._show_add_voice_blend_dialog | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_settings_mixin.py | SpeechSettingsMixin._update_default_voice_options | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Speech/speech_synthesis_mixin.py | SpeechSynthesisMixin._generate_tts | write | 4 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_single_worker | copy_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_single_worker | create_private_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_single_worker | secure_private_directory | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_worker | copy_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_worker | secure_private_directory | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._check_single_worker | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._export_characters_worker | create_private_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._export_characters_worker | secure_private_directory | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._export_conversations_worker | create_private_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._export_conversations_worker | secure_private_directory | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._export_notes_worker | create_private_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._export_notes_worker | secure_private_directory | 2 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._get_schema_version | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._integrity_worker | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._perform_database_restore | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._restore_single_database | secure_private_directory | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._restore_single_worker | copy_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._restore_single_worker | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._vacuum_single_worker | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._vacuum_worker | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._write_backup_manifest | create_private_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._write_backup_manifest | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow.on_markdown_link_clicked | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Voice_Cloning_Window.py | VoiceCloningWindow._test_generate_voice | write | 2 | unsupported | miscellaneous |
| tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py | ProgressStep._export_chatbook | secure_private_directory | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py | VoiceSetupStep._play_sample | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/stts_profile_library.py | STTSProfileLibrary._write_profile_export | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Splash_Screens/custom/custom_image.py | CustomImageEffect._load_and_convert_image | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | FileProcessor.detect_encoding | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | FileProcessor.read_file_content | open | 3 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | ensure_directory_exists | makedirs | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | generate_unique_identifier | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | safe_read_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | save_segments_to_json | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | save_segments_to_json | makedirs | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | save_segments_to_json | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | save_to_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | save_to_file | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/Utils.py | verify_checksum | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_copy | copy2 | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_copy | mkdir | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_bytes | mkdir | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_bytes | write | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_text | mkdir | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_text | write | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager.__init__ | makedirs | 1 | unsupported | tokenizers.custom; participant-task10 |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager._load_mappings | open | 1 | unsupported | tokenizers.custom; participant-task10 |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager.install_tokenizer | copy2 | 1 | unsupported | tokenizers.custom; participant-task10 |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager.save_mappings | dump | 1 | unsupported | tokenizers.custom; participant-task10 |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager.save_mappings | open | 1 | unsupported | tokenizers.custom; participant-task10 |
| tldw_chatbook/Utils/egress.py | guarded_fetch_requests | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/fd_protection.py | protect_file_descriptors | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/file_handlers.py | DataFileHandler._process_csv | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/file_handlers.py | DataFileHandler._process_json | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/file_handlers.py | DataFileHandler._process_yaml | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/file_handlers.py | DataFileHandler._process_yaml | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/instance_lock.py | acquire_profile_instance_lock | open | 1 | process_artifact | runtime.instance_lock; PID-and-portalocker-only |
| tldw_chatbook/Utils/instance_lock.py | acquire_profile_instance_lock | write | 1 | process_artifact | runtime.instance_lock; PID-and-portalocker-only |
| tldw_chatbook/Utils/log_widget_manager.py | LogWidgetManager.update_log | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/note_importers.py | CSVImporter.parse_file | open | 1 | unsupported | files |
| tldw_chatbook/Utils/note_importers.py | JSONImporter.parse_file | open | 1 | unsupported | files |
| tldw_chatbook/Utils/note_importers.py | YAMLImporter.parse_file | open | 1 | unsupported | files |
| tldw_chatbook/Utils/paths.py | get_project_databases_dir | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/secure_temp_files.py | create_secure_temp_file | write | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/secure_temp_files.py | secure_delete_file | open | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/secure_temp_files.py | secure_delete_file | write | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/ui_responsiveness_artifacts.py | write_responsiveness_artifacts | mkdir | 1 | diagnostics | diagnostics |
| tldw_chatbook/Utils/ui_responsiveness_artifacts.py | write_responsiveness_artifacts | write_text | 5 | diagnostics | diagnostics |
| tldw_chatbook/Video_Generation/video_store.py | VideoStore._atomic_publish | mkdir | 1 | process_artifact | process |
| tldw_chatbook/Video_Generation/video_store.py | VideoStore._atomic_publish | write | 1 | process_artifact | process |
| tldw_chatbook/Video_Generation/video_store.py | VideoStore._ensure_safe_root | mkdir | 1 | process_artifact | process |
| tldw_chatbook/Video_Generation/video_store.py | VideoStore._root_lease | mkdir | 1 | process_artifact | process |
| tldw_chatbook/Video_Generation/video_store.py | VideoStore._root_lease | open | 1 | process_artifact | process |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | create_filtered_sitemap | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | load_bookmarks | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | load_hashes | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | recursive_scrape | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | recursive_scrape.save_progress | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | recursive_scrape.save_progress | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | save_hashes | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | save_hashes | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | scrape_and_convert_with_filter | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | scrape_and_convert_with_filter | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Web_Scraping/Article_Scraper/importers.py | _load_from_chromium_json | open | 1 | unsupported | files |
| tldw_chatbook/Web_Scraping/Article_Scraper/importers.py | _load_from_firefox_html | open | 1 | unsupported | files |
| tldw_chatbook/Web_Scraping/Confluence/confluence_main.py | scrape_confluence_with_config | open | 1 | unsupported | files |
| tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py | _private_cookie_clone | open | 2 | external_input | external |
| tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py | _private_cookie_clone | secure_private_directory | 1 | external_input | external |
| tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py | get_chrome_cookies | connect_private_sqlite | 1 | external_input | external |
| tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py | get_chrome_cookies | open | 1 | unsupported | external |
| tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py | get_edge_cookies | connect_private_sqlite | 1 | external_input | external |
| tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py | get_edge_cookies | open | 1 | unsupported | external |
| tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py | get_firefox_cookies | connect_private_sqlite | 1 | external_input | external |
| tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py | get_safari_cookies | open | 1 | unsupported | external |
| tldw_chatbook/Widgets/Chat_Widgets/chat_message_enhanced.py | ChatMessageEnhanced._render_pixelated | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Chat_Widgets/chat_message_enhanced.py | ChatMessageEnhanced._render_regular | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Console/console_context_modal.py | ConsoleContextModal._save_json | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Console/console_context_modal.py | ConsoleContextModal._save_json | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Console/console_transcript.py | ConsoleMarkdownMessage._open_link | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Console/console_transcript.py | ConsoleTranscript._append_paint_log | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Console/console_transcript.py | ConsoleTranscript._append_paint_log | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | JSONStorage._create_backup | write | 1 | generic_boundary | tamagotchi.config phase8 exact default ConfigFileStorage; custom JSON ordinary; runtime composition pending |
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | JSONStorage._write_data | dump | 1 | generic_boundary | tamagotchi.config phase8 exact default ConfigFileStorage; custom JSON ordinary; runtime composition pending |
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | SQLiteStorage._connect | connect_private_sqlite | 2 | generic_boundary | tamagotchi.config-via-JSONStorage-or-dormant-custom-path |
| tldw_chatbook/Widgets/activity_log.py | ActivityLogWidget._write_json_export | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/activity_log.py | ActivityLogWidget._write_text_export | write | 3 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/activity_log.py | ActivityLogWidget.export_log | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/audio_troubleshooting_dialog.py | AudioTroubleshootingDialog._apply_and_close | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/audio_troubleshooting_dialog.py | AudioTroubleshootingDialog._initialize_audio | write | 5 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/audio_troubleshooting_dialog.py | AudioTroubleshootingDialog._start_level_test | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/audio_troubleshooting_dialog.py | AudioTroubleshootingDialog._stop_level_test | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/audio_troubleshooting_dialog.py | AudioTroubleshootingDialog.on_select_changed | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/chunk_preview_modal.py | ChunkPreviewModal.export_preview | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/chunk_preview_modal.py | ChunkPreviewModal.export_preview | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/file_extraction_dialog.py | FileExtractionDialog._save_files | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/project_skills_import_modal.py | _read_loose_skill_file_sync | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.on_export_theme | dump | 1 | generic_boundary | caller-selected-external-theme-export |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.on_save_theme | dump | 1 | unsupported | ui.themes phase8 raw source lifetime; runtime composition pending |
| tldw_chatbook/Workspaces/change_retention.py | prune_change_history | mkdir | 2 | qualified | workspaces.change_tracking |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo._locked._Lock.__enter__ | mkdir | 2 | qualified | workspaces.change_tracking |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo.ensure_initialized | mkdir | 3 | qualified | workspaces.change_tracking |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo.ensure_initialized | write_text | 1 | qualified | workspaces.change_tracking |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo.snapshot | open | 1 | qualified | workspaces.change_tracking |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo.snapshot | write | 3 | qualified | workspaces.change_tracking |
| tldw_chatbook/Writing_Interop/local_writing_service.py | LocalWritingService._connect | connect_private_sqlite | 2 | unsupported | miscellaneous |
| tldw_chatbook/app.py | <module> | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/app.py | TldwCli._display_buffered_logs | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/app.py | TldwCli._ensure_tts_profile_repository.open_repository | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/app.py | TldwCli._setup_buffered_logging.PersistentLogHandler.emit | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/app.py | _generated_css_is_stale | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/app.py | _generated_css_is_stale._sha256 | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/app.py | _ingest_pool_real_stderr | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/app.py | _load_css_build_manifest | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/app.py | _save_css_build_manifest | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/app.py | main_cli_runner | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/config.py | _config_interprocess_lock | create_private_text | 1 | unsupported | config |
| tldw_chatbook/config.py | _config_interprocess_lock | open_private_text_append_stream | 1 | unsupported | config |
| tldw_chatbook/config.py | _load_cli_config_bootstrap_unlocked | create_private_text | 2 | unsupported | config |
| tldw_chatbook/config.py | _load_cli_config_bootstrap_unlocked | open_private_binary | 1 | unsupported | config |
| tldw_chatbook/config.py | _load_cli_config_bootstrap_unlocked | secure_private_directory | 1 | unsupported | config |
| tldw_chatbook/config.py | _prepare_config_parent | secure_private_directory | 1 | unsupported | config |
| tldw_chatbook/config.py | _read_cli_config_serialized_unlocked | open_private_binary | 1 | unsupported | config |
| tldw_chatbook/config.py | _read_raw_cli_config_unlocked | open_private_binary | 1 | unsupported | config |
| tldw_chatbook/config.py | _try_read_cli_config_serialized_unlocked | open_private_binary | 1 | unsupported | config |
| tldw_chatbook/config.py | _write_raw_cli_config_unlocked | atomic_private_write_text | 1 | unsupported | config |
| tldw_chatbook/config.py | _write_serialized_config_artifact_unlocked | atomic_private_write_text | 1 | unsupported | config |
| tldw_chatbook/config.py | get_model_cache_dir | mkdir | 1 | unsupported | config |
| tldw_chatbook/config.py | get_user_data_dir | secure_private_directory | 2 | unsupported | config |
| tldw_chatbook/config.py | load_openai_mappings | open | 1 | unsupported | config |
| tldw_chatbook/config.py | load_settings | mkdir | 1 | unsupported | config |
| tldw_chatbook/config.py | read_cli_config_backup_serialized | open_private_binary | 1 | unsupported | config |
| tldw_chatbook/css/build_css.py | _file_sha256 | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/css/build_css.py | build_css | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/css/build_css.py | build_css | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/css/build_css.py | build_screen_css | write_text | 2 | unsupported | miscellaneous |
| tldw_chatbook/css/build_css.py | build_widget_defaults | write_text | 2 | unsupported | miscellaneous |
| tldw_chatbook/css/build_css.py | write_build_manifest | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/runtime_policy/server_credentials.py | KeyringServerCredentialStore._delete_index_record | delete_password | 1 | unsupported | runtime |
| tldw_chatbook/runtime_policy/server_credentials.py | KeyringServerCredentialStore._save_index | set_password | 1 | unsupported | runtime |
| tldw_chatbook/runtime_policy/server_credentials.py | KeyringServerCredentialStore.delete_scoped_secret | delete_password | 3 | unsupported | runtime |
| tldw_chatbook/runtime_policy/server_credentials.py | KeyringServerCredentialStore.set_scoped_secret | set_password | 1 | unsupported | runtime |
| tldw_chatbook/runtime_policy/source_state.py | RuntimeSourceStateStore.load | open_private_binary | 1 | qualified | runtime.source_state |
| tldw_chatbook/runtime_policy/source_state.py | RuntimeSourceStateStore.save | atomic_private_write_text | 1 | qualified | runtime.source_state |
| tldw_chatbook/tldw_api/utils.py | prepare_files_for_httpx | open | 1 | unsupported | server |

## Targeted evidence

`Tests/Backup_Recovery/test_inventory.py` exercises real SQLite/files, custom and
multiple profiles, shared hardlinks/symlinks, unknown durable files, required
absence, external-root exclusion, malformed config, dependencies, frozen tuples,
reserved context injection and fresh-process no-bootstrap/no-mutation discovery.
`Tests/Architecture/test_backup_owner_inventory.py` checks every exact producer row,
SQLite policy ID and synthetic new-producer/count-change negative controls.
Existing profile-path, retired-owner, effective-config and private-SQLite guards
remain required. Their actual commands/results are in the task implementation notes.

Capture qualification, DB-contained reference discovery, semantic validation,
relocation, credential sanitization and runtime activation remain explicit owner
cohort work. This census grants no complete-backup, replacement or release claim.

## Maintenance and native storage producers (TASK-31987)

Admission owns private control-root registry records and persistent lock files; pending remap records are durable recovery evidence, never disposable or imported authority. Capture/relocation remains unsupported until the control owner qualifies. Native creation/publication routines are generic checked boundaries; calling owners must inventory their output, staging, and retained recovery files. The qualification reader reads installed metadata only.

| Module | Qualified symbol | Call | Count | Classification | Cohort |
| --- | --- | --- | --- | --- | --- |
| tldw_chatbook/Backup_Recovery/native_files.py | _flush_private_tree | open | 1 | generic_boundary | backup_native_storage |
| tldw_chatbook/Backup_Recovery/native_files.py | create_private_directory | mkdir | 1 | generic_boundary | backup_native_storage |
| tldw_chatbook/Backup_Recovery/native_files.py | create_private_file | open | 1 | generic_boundary | backup_native_storage |
| tldw_chatbook/Backup_Recovery/native_files.py | pinned_directory | open | 2 | generic_boundary | backup_native_storage |
| tldw_chatbook/Backup_Recovery/native_files.py | publish_new | open | 1 | generic_boundary | backup_native_storage |

## Native publication census extension (TASK-31987)

The concrete `renameatx_np` callable assignment and `os.replace` seams are now counted, including imported OS aliases. These are cumulative syntactic candidates, not runtime binding inference; ordinary string replace calls do not receive an OS-publication label. Existing publication symbols retain their previously reviewed owner/cohort classifications below; newly surfaced Media, legacy-settings, and TTS symbols remain unsupported. This adds no blanket owner exemption and does not qualify legacy no-replace implementations for recovery.

| Module | Qualified symbol | Call | Count | Classification | Cohort |
| --- | --- | --- | --- | --- | --- |
| tldw_chatbook/Backup_Recovery/native_files.py | _rename_new | renameatx_np | 1 | generic_boundary | backup_native_storage |
| tldw_chatbook/Chat/trajectory_export.py | write_trajectory_export | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_zip_archive | os.replace | 1 | generic_boundary | chatbooks.archives-default-or-explicit-external-destination; participant-task10 |
| tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py | ModelCatalogDiskStore.save | os.replace | 1 | disposable | diagnostics |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._default_url_file_downloader | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._copy_payload | os.replace | 1 | unsupported | models |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_file | os.replace | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager._migrate_legacy_blob | os.replace | 1 | unsupported | rag |
| tldw_chatbook/Subscriptions/briefing_export.py | _write_feed_xml_atomically | os.replace | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/TTS/profile_migration_namespace.py | _rename_noreplace | renameatx_np | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_backup_to | os.replace | 1 | unsupported | tts |
| tldw_chatbook/UI/Console_Modules/video.py | ConsoleVideoController._copy_pending_video_external | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_databases | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_worker | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_text | os.replace | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_bytes | os.replace | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_copy | os.replace | 1 | generic_boundary | generic |
| tldw_chatbook/Video_Generation/video_store.py | VideoStore._commit_sibling | os.replace | 1 | process_artifact | process |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | recursive_scrape.save_progress | os.replace | 1 | unsupported | miscellaneous |

## Local write-intent retirement census (TASK-31987 review fix)

The control owner now exclusively creates bounded version-1 before/after write intents through `Admission._write_new_record`, and retires a matching intent through `os.unlink` only after the registry file/parent full native barriers complete. Intent removal is a persistence boundary, not disposable cleanup authority. The census now recognizes concrete OS unlink calls/import aliases, including the existing removal candidates below. Existing unsupported/generic classifications are retained per producer; newly surfaced symbols remain unsupported within their known cohort (the secure temporary-file helper remains a generic boundary). No new capture/exclusion qualification is implied.

| Module | Qualified symbol | Call | Count | Classification | Cohort |
| --- | --- | --- | --- | --- | --- |
| tldw_chatbook/Chat/trajectory_export.py | write_trajectory_export | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py | extract_text_from_image | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/OCR_Backends.py | DocextOCRBackend.process_pdf | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend.transcribe | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend.transcribe_buffer | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend._transcribe_with_parakeet_mlx | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend._transcribe_buffer_with_parakeet_mlx | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/video_processing.py | LocalVideoProcessor._discard_temp_cookiefile | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_file | os.unlink | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_copy | os.unlink | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.export_exact_file | os.unlink | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.create_file | os.unlink | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.move_file | os.unlink | 2 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.delete_file | os.unlink | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.restore_file | os.unlink | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot.write_text | os.unlink | 1 | external_input | external-notes-workspace-or-export |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | AudioCppGeneratedLaunchArtifact.cleanup | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | _remove_partial_artifact | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_service.py | AudioService.create_m4b_with_chapters | os.unlink | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend._transcribe_audio | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_namespace.py | remove_zero_reusable_tombstone | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _create_materialization_sync | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _sweep_orphans | os.unlink | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _cleanup_materialization_sync | os.unlink | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _CandidateValidationJob.cleanup.remove_snapshot | os.unlink | 2 | unsupported | tts |
| tldw_chatbook/UI/Console_Modules/video.py | ConsoleVideoController._copy_pending_video_external | os.unlink | 2 | unsupported | miscellaneous |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_text | os.unlink | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_bytes | os.unlink | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_copy | os.unlink | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/secure_temp_files.py | secure_temp_file | os.unlink | 1 | generic_boundary | generic |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | scrape_entire_site | os.unlink | 1 | unsupported | miscellaneous |


## Startup admission integration (TASK-31988)

The fixed default config namespace now owns `recovery-bootstrap/`, including
version-1 private pending/profile associations, a scope-neutral `unbound-owner`,
and the separate stable `admission/` authority. These local records are recovery
authority, excluded from imported archive authority. Missing/corrupt known state
is retained and fenced, never repaired by ordinary startup. Pending control roots
may be elsewhere; they cannot choose a second admission authority.

The following conservative read-open candidates have been reviewed individually:

| Module | Symbol | Call | Count | Classification | Cohort |
| --- | --- | --- | --- | --- | --- |
| tldw_chatbook/Backup_Recovery/bootstrap.py | _read | open | 1 | generic_boundary | backup_control_reader |
| tldw_chatbook/Backup_Recovery/bootstrap.py | _fingerprint | open | 1 | generic_boundary | backup_selector_reader |

`DB/private_sqlite.py` retains its exact two raw connection sites and the
registered owner inventory. Connection lifetime admission wraps that seam without
changing target selection or trusted-parent enforcement. Memory databases and
verified immutable descriptor views remain classified exemptions. Only browser
cookie clone sources (`cookies.chrome`, `cookies.edge`, `cookies.firefox`) have the
explicit `foreign_read_only_source` exemption; preserving a source file's mode
does not exempt owned RAG, Watchlists, or TTS reads from recovery fences.

The shared private file create/atomic-write/append/read-harden/directory boundaries
participate, with append streams retaining admission through successful close.
This does not qualify every raw writer in this census. Unsupported raw-owner
cohorts still block Complete backup and replacement.

## Core recovery SQLite cohort (TASK-31989)

The five `recovery.core.*` backup authorities preserve all SQLite rows, FTS state and
BLOB assets through a WAL-aware snapshot. Pure discovery resolves custom paths
without constructing/opening domain stores. Required later-owned file assets and
operational activation stay explicit dependencies. See [core qualification](backup-recovery-core-owners.md).

| sqlite:recovery.core.chachanotes | tldw_chatbook/DB/recovery_core | _PRIVATE_AND_READ_ONLY | sqlite/core-recovery-qualified |
| sqlite:recovery.core.media | tldw_chatbook/DB/recovery_core | _PRIVATE_AND_READ_ONLY | sqlite/core-recovery-qualified |
| sqlite:recovery.core.prompts | tldw_chatbook/DB/recovery_core | _PRIVATE_AND_READ_ONLY | sqlite/core-recovery-qualified |
| sqlite:recovery.core.library_collections | tldw_chatbook/DB/recovery_core | _PRIVATE_AND_READ_ONLY | sqlite/core-recovery-qualified |
| sqlite:recovery.core.library_ingest_jobs | tldw_chatbook/DB/recovery_core | _PRIVATE_AND_READ_ONLY | sqlite/core-recovery-qualified |
| tldw_chatbook/DB/recovery_core.py | _CoreAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-admission-and-core-recovery |
| tldw_chatbook/DB/recovery_core.py | _CoreAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-admission-and-core-recovery |
| tldw_chatbook/DB/recovery_core.py | _CoreAdapter.validate_dependencies | connect_private_sqlite | 2 | generic_boundary | native-admission-and-core-recovery |

| tldw_chatbook/Backup_Recovery/activation.py | _flush_existing | open | 1 | generic_boundary | backup_activation_records |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._create_lock | open | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._open | open | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._write_new_record | open | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._write_new_record | write | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._write | os.replace | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._write | os.unlink | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/control_records.py | _publish_activation_record | os.replace | 1 | generic_boundary | backup_control |
| tldw_chatbook/Backup_Recovery/control_records.py | _bind_activation | os.unlink | 1 | generic_boundary | backup_control |
| tldw_chatbook/DB/private_sqlite.py | _connect_registered_sqlite | connect | 1 | generic_boundary | generic |
| tldw_chatbook/DB/private_sqlite.py | _open_artifact_fd | open | 1 | generic_boundary | generic |
| tldw_chatbook/DB/private_sqlite.py | _prepare_windows_artifact | open | 1 | generic_boundary | generic |
| tldw_chatbook/DB/private_sqlite.py | migrate_profile_store_to_candidate | connect_private_sqlite_descriptor | 1 | generic_boundary | generic |
| tldw_chatbook/DB/private_sqlite.py | open_canonical_profile_migration_destination | secure_private_directory | 1 | generic_boundary | generic |
| tldw_chatbook/DB/private_sqlite.py | open_profile_migration_boundary_destination | secure_private_directory | 1 | generic_boundary | generic |

## Domain recovery capture/validation census (TASK-31990)

| tldw_chatbook/Research_Interop/recovery.py | _Adapter.validate | connect_private_sqlite | 1 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Research_Interop/recovery.py | _Adapter.capture | copy_private_sqlite | 1 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Writing_Interop/recovery.py | _Adapter.validate | connect_private_sqlite | 1 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Writing_Interop/recovery.py | _Adapter.capture | copy_private_sqlite | 1 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Study_Interop/recovery.py | _SharedAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-domain-recovery |
| sqlite:recovery.domain.research | tldw_chatbook/Research_Interop/recovery | _PRIVATE_AND_READ_ONLY | sqlite/domain-recovery-qualified |
| sqlite:recovery.domain.writing | tldw_chatbook/Writing_Interop/recovery | _PRIVATE_AND_READ_ONLY | sqlite/domain-recovery-qualified |
| sqlite:recovery.domain.evals | tldw_chatbook/Evals/recovery | _PRIVATE_AND_READ_ONLY | sqlite/domain-recovery-qualified |
| sqlite:recovery.domain.study | tldw_chatbook/Study_Interop/recovery | _READ_ONLY_URI | sqlite/domain-recovery-qualified |
| tldw_chatbook/Evals/recovery.py | _Adapter.validate | connect_private_sqlite | 1 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Evals/recovery.py | _Adapter.capture | copy_private_sqlite | 1 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Evals/recovery.py | _Adapter.validate_dependencies | connect_private_sqlite | 2 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Backup_Recovery/storage_admission.py | copy_capture_file | open | 2 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Backup_Recovery/storage_admission.py | copy_capture_file | write | 1 | generic_boundary | native-domain-recovery |

## Local research, writing, study and evaluation qualification

TASK-31990 implements [ADR-126](../decisions/126-complete-local-backup-and-recovery.md).
The four `recovery_adapters()` factories are import-light installed declarations;
executor composition registers them with `owner_registry.register`. They neither
instantiate stores nor start servers, engines, evaluations or models. The package
public exports retain their identities through lazy resolution. Existing feature
flags never suppress a discovered durable store, and custom database selectors use
`profile_paths.database_path` without fallback creation.

| Logical owner | Actual source and capture | Qualified schema / references |
| --- | --- | --- |
| research.local | Canonical research_db_path; full SQLite snapshot, including sessions, runs, events, checkpoints, artifacts, soft deletions and lease columns | Exact current PRAGMA user_version 1; genuine pre-lease v0 SQL from b6ba7d013^; declared installed ALTER transition 0→1; FK/integrity validation |
| writing.local | Canonical writing_db_path; full SQLite snapshot of projects, hierarchy, characters, world information, relationships, analyses, citations, versions and trash | Exact installed unversioned layout (PRAGMA user_version 0); no invented historical schema/version; FK/integrity validation |
| study.local / quiz.local | One shared db.chachanotes.primary physical payload, never separate selective exports | Core exact v42 schema, BLOB size/reference checks including deleted rows, study session references and historical quiz snapshot/answer identities |
| db.evals | Canonical evals_db_path; EvalsDB also owns word benches and character probes, datasets, models, run groups, snapshots, outputs, metrics, annotations, review state and AB comparisons | Exact installed v5; SQLite FK/integrity checks; selected-profile current character/model references; historical run snapshots retained independently |
| eval.definitions | Exact installed `Evals/config/eval_config.yaml`, including local edits made by EvalConfigLoader.save with its default path | Bounded 16 MiB safe portable YAML mapping; unsafe tags, recursive aliases and nonportable/deep structures refuse; bytes retained exactly, new file mode 0600 |

The frozen SQL catalogs are in each corresponding recovery module and were generated
from real constructors under Tests isolation at task 6 base 8ea52cfc0. All tables,
indexes, triggers and FTS/shadow-table SQL are included in sorted type/name order.
The old research fixture contains all five research tables, lacks lease columns,
and is independent of the current constructor. Only its declared installed
migration SQL is executed in the migration test; backup itself never migrates.
Other historical/physical schema variants remain explicitly unsupported. Research
artifacts and writing bodies are stored inside SQLite, not in an inferred asset
folder. Flashcard assets use `flashcard-asset://` UUIDs and stored BLOB bytes; moving
the physical database preserves those locators. Arbitrary prose, citations, dataset
source paths and URL strings are not rewritten or activated as managed paths.

Core/study/quiz declarations initially name one stable group per selected profile.
Inventory validates every original declared group's physical identity **before**
merging the exact installed cohort across profiles. Same-file/hardlink aliases merge
using a hash of final logical IDs, distinct files remain separate, and mismatches or
forged labels refuse. Inodes are proof only and never enter the stable scope digest.
The capture executor must deduplicate each physical group and supply the same staged
candidate identity for its logical members. A second independently copied database
cannot satisfy the shared semantic dependency check.

### Exact evaluation producer dispositions

The executable source rows above classify each concrete open/write site. Their
owner/path interpretation is as follows; external selections can be included through
the later explicit external-root option, and never justify a recursive home scan.

| Producer(s) | Path authority and durable result |
| --- | --- |
| EvalConfigLoader._load_config / save | No-argument path is the installed config file above and is baseline owned data. Explicit constructor/save path arguments identify external selected files; the application has no separate persisted custom-loader selector. Save holds ordinary admission through its complete write. |
| EvaluationOrchestrator._initialize_database | Uses get_evals_db_path or an explicit caller database argument; the application's wired instance uses the canonical selector. eval.orchestrator_parent is a historical parent-boundary declaration, not a second database payload. |
| EvaluationOrchestrator.create_task_from_template | Stores the task in EvalsDB. Writes a sample dataset only when output_dir is explicitly supplied; those bytes are external selected output. |
| EvaluationOrchestrator.export_results / quick_eval | export_results requires output_path; quick_eval writes only for an explicit output_dir. Durable original results already live in EvalsDB. |
| EvalTemplateManager.export_template_as_file / create_sample_dataset | Both require explicit output_path. No hidden default output folder. |
| TaskLoader.export_task | Explicit path argument; generated template dictionaries otherwise remain in memory or are persisted in EvalsDB by the orchestrator. |
| EvaluationExporter._export_ab_test_csv / _json / _latex / _markdown and _export_run_csv / _json / _markdown | All receive the explicit output_path from EvaluationExporter.export; none select an application data directory. |
| DatasetLoader._load_csv_dataset / _load_json_dataset in dataset_loader.py and eval_runner.py; DatasetValidator._load_dataset | Read explicit task/dataset sources; inline samples and their durable metadata are retained in EvalsDB. External source files/Hugging Face datasets retain inert references. |
| TaskLoader._detect_file_format / _detect_format / _load_csv_task / _load_custom_task / _load_eleuther_task / _load_huggingface_task | Read caller-selected task/config files; no owned durable output at these sites. |
| CodeExecutionRunner._execute_code | Writes code in its temporary execution directory; process artifact, not durable evaluation history. Capture never imports or runs it. |
| EvaluationState.save_to_file / load_from_file | Explicit filepath argument; no current production caller chooses a hidden fixed destination. Caller-selected state files are external outputs/inputs. |
| word_bench.storage and character_probe.storage | Use the supplied EvalsDB persistence API. Real recovery fixtures retain bench definitions, character run snapshots, replies, annotations and explicit review state. |
| ServerEvaluationsService / server research/writing/study/quiz services | Server-owned recovery boundary; no server fetch, engine startup, or mirror mistaken for local authority. Local stores remain baseline even when server mode is selected. |

### Admission, file scope and activation boundaries

All SQLite copies retain the reviewed fixed native authority, intact local binding,
selected namespaces plus `bootstrap.unbound`, exact sources and private staging.
Capture uses only the default SQLite factory. Ordinary research/writing file
operations now close their actual handles after commit/rollback instead of relying
on garbage collection; their persistent in-memory connection behavior is unchanged.
Every production `_connect()` call consumes rows/scalars inside its operation context.

Only `eval.definitions` is registered for the operation-private raw helpers
`copy_capture_file` and `_read_recovery_file`. No streams escape. They pin no-follow
parents, verify regular nonaliased files and exact native scope identity, enforce
actual streamed-byte limits, refuse overwrite and retire tracked FDs before returning.
The copy is cancellable and fsynced; interrupted candidates remain private executor
cleanup artifacts. The reader uses ordinary admission outside capture and never
opens arbitrary out-of-scope files during capture. Neither helper widens enrollment.
The default package config file therefore must be included in ordinary inventory
and namespace enrollment before capture. Only portable content is restored by this
owner; source executable permissions or ACLs are not granted by recovered YAML.
Directory/archive metadata qualification remains the container owner's responsibility.

Owner activation remains required: preserved lease/run state, model definitions,
provider configuration and execution snapshots are data, not permission to resume.
Task 14 must inspect evaluation configuration/provider mappings and DB model/config
JSON for managed credential locations before portable redaction or encrypted inclusion;
this task does not claim secret-free captures or perform credential sanitization.
Task 13 still owns sealed candidate/schema budgets and migration execution safeguards;
the 16 MiB YAML bound is an owner refusal boundary, not the global archive budget.
Native retirement includes raw source/destination parent directories and all
private traversal cleanup, including trusted-symlink transitions. The optional
private close callback follows `bootstrap.pinned_directory` through
`private_paths._open_verified_parent` / `_follow_trusted_symlink`; default callers
retain their existing behavior. The raw owner latches actual close uncertainty
before cleanup can retry an FD number. Ordinary missing/refused paths with
successful cleanup release their lease normally. Child-local simulated native-close
failures leave real directory FDs usable and demonstrate independent maintenance
exclusion until process exit. This is simulated-failure evidence, not actual OS or
hardware failure qualification; that release limit remains with task 6's native-close
limit. No Complete backup/publication/replacement UI is qualified by this cohort alone.

Research, writing and EvalsDB retain their literal registered connection/copy sites.
`DB/recovery_sqlite.py` only shares schema/FK/integrity validation and capture guards;
it neither opens storage nor chooses dynamic SQLite authority. The exact source and
copy census rows therefore remain unchanged by this consolidation.

## Task 8 installed operational producer additions

| tldw_chatbook/Agents/recovery.py | _RunLogs.discover | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Backup_Recovery/file_inventory.py | _inventory_tree.walk | open | 2 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _consume_recovery_file | open | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/DB/recovery_operations.py | _AgentRunsAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/DB/recovery_operations.py | _AgentRunsAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/DB/recovery_operations.py | _SubscriptionsAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/DB/recovery_operations.py | _SubscriptionsAdapter.discover | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/DB/recovery_operations.py | _SubscriptionsAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/DB/recovery_operations.py | _SubscriptionsAdapter.validate_dependencies | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/DB/recovery_operations.py | _WorkspacesAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/DB/recovery_operations.py | _WorkspacesAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Kanban_Interop/recovery.py | _KanbanAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Kanban_Interop/recovery.py | _KanbanAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notes/recovery.py | _FileNotesAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notes/recovery.py | _FileNotesAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notes/recovery.py | _ReceiptsAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notes/recovery.py | _ReceiptsAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notes/recovery.py | _SyncBindings.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notifications/recovery.py | _EventsAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notifications/recovery.py | _EventsAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notifications/recovery.py | _NotificationsAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Notifications/recovery.py | _NotificationsAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Scheduling/recovery.py | _ScheduledTasksAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Scheduling/recovery.py | _ScheduledTasksAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Sync_Interop/recovery.py | _SyncAdapter.capture | copy_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| tldw_chatbook/Sync_Interop/recovery.py | _SyncAdapter.validate | connect_private_sqlite | 1 | generic_boundary | native-operational-recovery |
| sqlite:recovery.operations.workspaces | tldw_chatbook/DB/recovery_operations | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.agent_runs | tldw_chatbook/DB/recovery_operations | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.subscriptions | tldw_chatbook/DB/recovery_operations | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.scheduled_tasks | tldw_chatbook/Scheduling/recovery | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.notifications | tldw_chatbook/Notifications/recovery | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.events | tldw_chatbook/Notifications/recovery | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.sync | tldw_chatbook/Sync_Interop/recovery | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.file_notes | tldw_chatbook/Notes/recovery | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.receipts | tldw_chatbook/Notes/recovery | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.agent_logs | tldw_chatbook/Agents/recovery | _READ_ONLY_URI | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.kanban | tldw_chatbook/Kanban_Interop/recovery | _PRIVATE_AND_READ_ONLY | sqlite/operational-recovery-qualified |
| sqlite:recovery.operations.note_bindings | tldw_chatbook/Notes/recovery | _READ_ONLY_URI | sqlite/operational-recovery-qualified |

## Task 8 qualified operational owners (TASK-31991)

This section supersedes the initial unsupported cohort entries **only for the exact
owners and producer rows marked `qualified` above**. It does not qualify RAG/vector
indexes (task 19), remaining generic file owners (task 9), managed credentials
(task 14), cross-platform metadata, or complete backup service composition. Adapters
are inert installed declarations; optional-disabled features still retain their
present files. Absent required SQLite stores remain missing-required; absent exact
optional raw files are unused, while unknown app-owned children block completeness.

| Owner | Installed selector and capture | Schema / dependencies | Activation claims retained as historical evidence |
| --- | --- | --- | --- |
| db.workspaces | workspaces_db_path; full WorkspaceDB | schema_version MAX 2, exact catalog/FK/quick_check | workspace IDs, active flags, runtime binding locators and allow_write metadata, handoff audits |
| db.agent_runs | agent_runs.db beside selected ChaChaNotes DB | schema_version MAX 12, exact catalog; core dependency | running/queued status, resumed/parent run IDs, budgets, definitions, snapshots/change notes |
| db.subscriptions | subscriptions_db_path; full SubscriptionsDB | schema_version MAX 1; two complete exact layouts: base and SiteConfigManager's core42 hybrid; hybrid requires embedded db_schema_version42; complete audio rows require exact profile payload IDs | URLs, enabled state, credential scopes, briefing/script/audio generation records, site definitions |
| db.scheduled_tasks | scheduled_tasks_db_path | schema_version MAX 3, exact catalog/FK | enabled schedules, queued/running jobs, histories, claimed delivery/execution state |
| notifications.client | notifications_db_path | schema_version MAX 1, exact catalog/FK | notification payloads and dispatch/delivery metadata |
| runtime.event_state | profile/tldw_chatbook_event_state.db | schema_version MAX 1, exact catalog/FK | stream cursors, principal/profile/source authority, observer status |
| runtime.sync_state | profile/tldw_chatbook_sync_state.db | schema_version MAX 4, exact catalog/FK | identity mappings, device/source/principal scope, uploads/downloads/conflicts and cursors |
| kanban.local | profile/tldw_chatbook_kanban.db | local_kanban_schema_meta schema_version1, exact catalog/FK; get_storage_status uses this same store | boards/cards/checklists/comments/activities, content identity |
| notes.file_notes | profile/file_notes.sqlite; full FileNotesReplica including FTS internals and raw revisions | installed unstamped schema, literal SELECT 0 only as validation dispatch; no version written | raw bytes, tombstones, protected paths, pre-edit/recovery snapshots; disk roots are historical text |
| notes.sync_state | profile/tldw_chatbook_notes_sync_state.db; actual NoteImportReceiptRepository | PRAGMA user_version1, exact catalog/FK; selected core dependency | approval/session IDs, pending payload/folder/membership effects, receipts and retries |
| notes.sync_bindings | shared physical ChaChaNotes payload | exact core42 and membership ownership semantics; same-profile core dependency and proven original physical identity | manual stays manual; managed owner_id/owner_active stay intact; legacy sync_sessions/conflicts/logs remain inert |
| mcp.local / mcp.targets / mcp.context | profile/local_mcp_store.json, mcp_server_targets.json, unified_mcp_context.json | checked bounded JSON objects, immutable bytes; 16MiB each | server commands, environment placeholders, tool profiles/targets/session bindings |
| mcp.permissions | profile/mcp_permissions.json and exact .bak | checked opaque recovery bytes, 16MiB each; corrupt backup is preserved evidence | historical allow/deny rules never become approvals |
| mcp.history | profile/mcp_execution_log.jsonl and exact .1 | checked opaque bytes, 256GiB/member | execution history; capture never invokes read_recent generation migration |
| runtime.source_state | runtime_policy.json beside selected config | checked bounded JSON object, 16MiB | selected server/local source, last-online state and reconnection scope |
| subscriptions.assets | profile/briefing_audio, explicit tree topology and all present files | checked opaque bytes; complete DB file_path dependencies use exact profile IDs and staged candidates | historical generated audio; no synthesis, export or cleanup replay |
| workspaces.change_tracking | profile/change_review, exact ShadowRepoService default | bounded no-follow topology, all retained Git bytes; no Git process during discovery/capture | commit/object/index/config/lock evidence never authorizes hooks, worktree restoration or stale-lock takeover |
| agents.history | exact configured tool sandbox plus local_filesystem WorkspaceDB and retained AgentRuns change roots, dotted and legacy run_log_dir_name subtrees only | checked opaque bounded files; DB locators must qualify; no arbitrary workspace contents | run logs/manifests/segments remain evidence, never tool permission |
| tamagotchi.config | ConfigFileStorage default .config/tldw_chatbook/tamagotchi_pets.json (APPDATA on Windows), exact timestamp backups | checked opaque bytes, 16MiB each; unknown siblings block | pet state and corrupt recovery backups; no constructor repair/replay |

Every owner above declares activation_required=True. `relocate` validates and
preserves immutable evidence; it neither claims fresh live allocation nor applies
imported path mappings. Tasks 17/20/21 must create/revalidate fresh live claims in
staged restore and the task20 durable activation ledger. Task25 is only the minimal
recovery launcher. Imported fields alone are never authorization to execute, access
external files, reconnect, rerun jobs or promote managed membership to manual.

ADR-021 describes an intended File Notes database/projection/recovery split. The
installed implementation has **one file_notes.sqlite replica/revision store**, not
the planned file_notes.db/notes_recovery.db pair. No phantom files or missing-required
rows were invented. ADR-029/030/036/059/060 ownership and selective export exclusions
remain intact. The installed NoteImportReceiptRepository holds import effects and
receipts; planned lasting-sync roots/journals are not yet present. Legacy lasting
sync fields in ChaChaNotes remain separately owned by notes.sync_bindings.

`JSONStorage`/`SQLiteStorage` are exported arbitrary-path capabilities with no
installed production construction sites; BaseTamagotchi defaults to MemoryStorage.
The exact call-site guard invalidates this unused classification if wiring appears.
NotesMirror likewise has no installed construction sites and defaults to memory.
No fabricated private SQLite filename or capture policy was added for these dormant
capabilities. Generic file tools operate on external/user-selected data; only the
application-owned log subtrees within workspace roots belong to agents.history.

Raw copying/checking retains task7 native FD/parent retirement and the fixed
application authority, selected namespaces plus bootstrap.unbound, exact enrolled
sources and private staging. `_check_recovery_file` streams bounded chunks and
observes cancellation without collecting bytes. Opaque validation proves safe byte
access, **not** semantic authority or a runnable Git/audio/permission schema. Tree
metadata is a task9 prerequisite and directory entries carry explicit same-profile
parent dependencies; unrelated owner/root overlaps still fail.

Ordinary MCP saves/migrations/appends, run-log writes, pet JSON saves/backups, shadow
Git operations and retention hold admission for their full mutation/process lifetime.
Briefing atomic writes already use the checked private writer; cleanup unlink now
also admits the whole operation. The operational process test stalls an actual Git
wrapper and proves maintenance cannot enter before its child process exits.

Fixture capture deliberately drains old EventStateRepository._get_connection and
SyncStateRepository._get_connection native cycles using gc.collect before enrolling
fixture authority: their per-call `with connection` commits but does not close, and
.close only handles memory connections. This **does not qualify runtime maintenance
drain**. Task10 owns coordinated live-resource closure; no complete backup capability
may infer production drain from fixture GC.

## Configuration and durable-file recovery census (TASK-31992)

| tldw_chatbook/DB/recovery_core.py | _CoreAdapter.discover | connect_private_sqlite | 1 | generic_boundary | qualified-core-reference-discovery |
| tldw_chatbook/Persona_Visual/recovery.py | _Assets._references | connect_private_sqlite | 1 | generic_boundary | qualified-persona-reference-discovery |
| tldw_chatbook/TTS/recovery.py | _Profiles.validate | connect_private_sqlite | 1 | generic_boundary | qualified-tts-schema-validation |
| tldw_chatbook/TTS/recovery.py | _Profiles.capture | copy_private_sqlite | 1 | generic_boundary | qualified-tts-capture |

| sqlite:recovery.files.tts | tldw_chatbook/TTS/recovery | _PRIVATE_AND_READ_ONLY | sqlite/durable-file-reference-qualified |
| sqlite:recovery.files.persona | tldw_chatbook/Persona_Visual/recovery | _READ_ONLY_URI | sqlite/durable-file-reference-qualified |

Task9 refines only the exact installed roots below. The source-symbol census above
remains exhaustive: a raw writer's `unsupported` row is **not** silently promoted to
ordinary-writer qualification by a recovery copy/read policy. The exact remaining
installed data qualifications are `rag.projections` / `rag.definitions` (task19)
and `runtime.credentials` (task14). Task10 owns full ordinary-writer lifetimes;
task11 owns intentional deletion and retained-reference service outcomes. There is
no runtime owner named `miscellaneous`: that older census bucket contains broad
syntactic candidates, including caller-selected exports, memory serialization and
process I/O. Its identified installed durable defaults are now mapped below.
Unknown app-data children, unknown skill/model siblings, custom acquisition
formats, and unresolved current or retained locators still block complete inventory.
No home scan or broad cache/backup-name exclusion was added. Earlier qualified
SQLite/operational owners keep their own adapters.

### Installed file policies and task10 participant handoff

`data` below is the pure selected `profile_paths.user_data_dir(config)`, including
installed `paths`/`Paths` aliases and user-folder selection. `config-parent` is the
explicit active config's parent. Every listed ordinary raw writer must enroll its
**full mutation lifetime**, including remove/rename/cleanup/child-process work,
before task10 may remove the corresponding `participant_pending` declaration or
release startup holds. A read-only/copy ID grants no ordinary-writer bypass.

| Installed owner | Exact selector / bytes policy | Actual source producer or selector; participant obligation |
| --- | --- | --- |
| config; config.history | Active config plus exact `.toml.bak` and `config_backup_YYYYMMDD_HHMMSS.toml` siblings; corrupt histories retained as bytes | `config._get_effective_config_path`, `config._write_serialized_config_artifact_unlocked`; existing checked config save admission remains. Credential policy applies equally to history in task14. |
| personas | data/tldw_chatbook_personas.json | `app._wire_character_persona_services`, `build_persona_service` and whole cached/core operations; exact source participant installed, runtime/capture validation pending. |
| chat.dictionary_history | data/tldw_chatbook_chat_dictionary_history.json | `build_dictionary_service`, whole core/history operations and concrete local scope jobs; exact source participant installed, runtime/capture validation pending. |
| chat.rag_context | data/tldw_chatbook_chat_rag_context.json | `ChatConversationService` compatibility writer/readers and actual bound citation migration companion; source participants installed, runtime/capture validation pending. |
| chat.grammars; feedback; audio.history | data/tldw_chatbook_chat_grammars.json; tldw_chatbook_feedback.json; tldw_chatbook_audio_history.json | `LocalChatGrammarsService._persist`, `LocalFeedbackService._persist`, `LocalAudioServicesService._persist_history`; participant pending. |
| chat.dictionaries | data/chat_dicts, including all retained files and empty topology | `config.load_settings` and exact `Chat_Dictionary_Lib` parser/import/export/listing operations; actual core `chat_dictionaries.file_path` remains a required group when present. Source participants installed, runtime/capture validation pending; caller files never widen this root. |
| chunking.templates | data/chunking_templates | `ChunkingTemplateManager._get_user_templates_dir`, `ChunkingTemplateManager.save_template`; participant pending. |
| notes.templates | config-parent/note_templates.json | `Notes.template_store` effective config selector and shared RMW; actual CLI and `_import_template_files` worker use phase8 source lifetime; runtime composition pending. |
| chat.prompts | installed package/Chat/prompt_templates | `Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR`, `load_template`; inert internal definitions. |
| generation.styles | data/image_generation_styles | `Media_Creation.generation_templates` installed template directory; participant pending. |
| tokenizers.custom | canonical ~/.config/tldw_cli/tokenizers, exact installed tokenizer owner root | `Utils.custom_tokenizers.CustomTokenizerManager.__init__`, `install_tokenizer`, `save_mappings`; participant pending. No environment model-cache inference. |
| skills | data/skills; known tldw_chatbook_skills.json, skills/, trust/ | Runtime `default_local_skills_store_dir` now delegates the pure recovery selector. `LocalSkillsService._save_index`, `_write_bytes_atomic`, `_write_text_atomic`, create/import/remove/script lifecycle; `Skills_Interop.atomic_write` and trust-store `_atomic_write_bytes`/`_atomic_write_json` require full participant lifetimes. Trust manifests/grants/snapshots are historical bytes, never imported execution authority. |
| persona.assets | data/persona_visual | `publish_persona_visual`, publication candidate cleanup, importer and authoring-workspace materialization/cleanup; concrete source/native lifetime implemented in phase12, aggregate app/headless qualification pending. Qualified core42 `persona_visual_assets` current/retained locators require exact source size/hash. |
| persona.visual_identity | data/visual_identities | `Character_Chat.visual_identity.publish_visual_identity_candidate` and cleanup/materialization; actual manual pack assets plus preview locators. Concrete source/native publication, readers, cleanup and UI callers implemented in phase13; aggregate app/headless qualification pending. Unknown source kinds refuse. |
| persona.visual_identity_builtin | Only referenced built-in files and ancestors under installed package/assets | Exact owning pack `source_kind=builtin`, core42 schema, checked relative path/size/SHA256. No whole-package traversal. Captured bytes never authorize installation or overwriting package assets. |
| chat.attachments | Same physical core DB, message_attachments.data and core image BLOBs | Exact core/attachment cohort with physical identity first. Delegates checked core capture/schema; no new raw writer. |
| tts.profile_store; tts.references | Canonical configured TTS DB, exact installed schema4; clone bytes/transcript/recipe references retained in DB | `_Profiles.capture` uses literal recovery.files.tts and existing checked SQLite snapshot. Exact TTS physical cohort; reference count/hash verified. Runtime `TTSProfileRepository` coordinated worker drain belongs task10. |
| tts.voices | Installed CHATTERBOX_VOICE_DIR and HiggsSettings.voice_samples_dir selectors, configured/default Kokoro blends directory, active config-parent/kokoro_voice_blends.json | `ChatterboxTTSBackend`, `HiggsVoiceProfileManager` create/import/save/restore/backup, `KokoroTTSBackend` saved blends and `voice_blend_paths.write_private_json`; participant pending. Voice payloads are baseline, not model opt-in. |
| TTS migration state | Exact journal/candidate/rollback constants from profile_migration_journal and DB-name.pre-v3.sqlite3/.pre-v4.sqlite3 | Existing state is an explicit unsafe pending-operation blocker; absence is recorded. No constructor, journal replay, migration, or blanket disposable classification. |
| models.artifacts | data/models topology; only models/managed installed descriptor layout qualified; unknown siblings unsupported | `managed_model_artifact_root` delegates pure selector. `ModelArtifactService.install`, import, acquire/activate/remove/reconcile and their stage/lease lifetimes remain participant pending. |
| Model recipes/state | Valid installed ArtifactDescriptor manifests and active/ready files retained inert by default; exact lock artifacts excluded | Locally present model_id variants only; no required downloadable alternatives. No recipe/state activation. |
| Selected model payloads | Opt-in model_id and exact ArtifactRef dependency closure, installed sizes/hashes; no generic links | Direct manifest dependency edges use final profile-qualified IDs. `_Artifacts.validate_dependencies` rechecks declared staged payloads and dependent manifest identity. Every selected manifest in the closure must be validated by task15/17. |
| Model staging/resume | Empty staging topology retained; nonempty staging is a specific unsafe pending-operation blocker | Actual `ModelArtifactService._download_stage_for` creates download-stage.json/payload/state; `_create_install_staging` owns install candidates. Discovery neither removes nor replays them. Unwired/unknown layouts cannot be selected as qualified model bytes. |
| generation.assets | data/generated_images/saved baseline; generated_images/temp and generated_videos opt-in | `Media_Creation.image_generation_service.ImageGenerationService._setup_output_directory`, `generate_custom`, `save_generation`, `cleanup_temp_images` and `VideoStore._atomic_publish`/`_commit_sibling`. Included assets retain participant pending; selected temporary media also retains explicit task11 reference-catalog qualification pending. |
| diagnostics.logs | Exact installed logging.log_filename under data plus numeric rotations, opt-in | `app` logging selector; defaults do not read diagnostic content. Unknown diagnostic/artifact layouts remain unsupported, not dropped. |
| cache.model_catalog | Exact data/model_catalog_cache.json excluded | `app` installed provider catalog cache selector; this does not exclude other caches. |
| external.files | Only explicit selected roots, including empty topology and ordinary metadata | No general symlink/mount/alias following. Missing selected roots are unavailable. |
| recovery.output | INTERNAL planned_output_root only; positive absence below pinned parent and no required baseline overlap | `_planned_output_exclusion`; existing empty paths refuse. No public extra CaptureOptions key, caller proof bit, output creation, or arbitrary existing output exclusion. Actual created output/control/rollback owners must independently qualify in tasks15/16/18. |

### Metadata, configuration and staged dependency contracts

`FileMetadata` is frozen, version1: final profile-qualified root/parent IDs,
bounded relative path, exact file/directory kind, observed ordinary mode bits,
mtime_ns, and private/external policy. Root/parent IDs remap with their item IDs;
private policy retains source mode evidence so later preview can explain restoring
private directories as0700. No foreign ownership or privilege restoration.
Metadata is preview evidence only; task15 must reobserve pinned sources at capture.
Scope digest includes topology/policy/selections/exclusions, not mode/mtime churn.

Actual Darwin tests exercise ACL, xattr, flags/links/special objects and private
metadata. Linux detection is implemented using fd listxattr but not host-qualified
here. Other platforms return unavailable capability. Unsupported metadata never
means presumed absent; selected builtins use the same walker with a bounded
private selected-path set, and an empty set never expands to the entire tree.

`managed_secret_locations` returns only known sensitive mapping-key tuples using
the installed sensitive-key predicate; no values, keyring access or decryption.
`remap_config_locations` accepts **exact installed section.key names**, comprising
all `profile_paths.DATABASE_PATHS` database selectors, paths.data_dir/Paths.data_dir,
notes.sync_directory, console.workspace_root, llm_management.model_download_dir,
embedding_config.model_cache_dir, app_tts.CHATTERBOX_VOICE_DIR,
app_tts.KOKORO_VOICE_BLENDS_DIR, HiggsSettings.voice_samples_dir. Unknown keys or
conflicting alias targets refuse. Arbitrary prose/unrecognized values are preserved.
These are pure parsed-mapping transformations, not mutation authority.
OwnerAdapter.relocate remains validation-only until task17 staged mutation authority.

Persona current and retained references are not deleted by a raw deleted flag.
Manual previews have no installed source digest column: require checked bytes now;
the archive capture supplies their content digest. Builtin bytes require the actual
installed digest, never a version label. Task11/17 must resolve staged builtin
bytes into validated dependency groups/new approved roots; incompatible installed
bytes require refusal or explicit remap, never package overwrite. All staged peer
lookups use exact declared final profile IDs, not suffix searches or archive paths.

TTS public package exports remain exact and lazy; public resolution admits startup.
Both actual TTS -m routes and the nested backend package guard before optional
imports; direct chatterbox protection remains. Persona/Skills exact exports are
lazy and model-store discovery preserves the no-inference/no-HTTP import seam.
No startup hold is released as a shortcut for a raw writer lacking tokens.

### Task9 source-census reconciliation: retained defaults and exact scratch

| Installed owner | Exact selector and capture policy | Installed source; task10 full mutation lifetime |
| --- | --- | --- |
| chat.prompt_history | data/prompt_history.jsonl, all persisted history bytes | `Chat.prompt_history.default_prompt_history_path`, `PromptHistory._history_io` append and capped rewrite, source-owned async queued/native lifetime; runtime composition pending. |
| ui.state | active config-parent/ui_state.toml, persisted sidebar state | `ChatScreen._load_sidebar_state`, `_write_sidebar_state_snapshot`, source-owned debounce/flush jobs; runtime composition pending. |
| ui.emoji_recents | active config-parent/recent_emojis.json, persisted recents | `Widgets.emoji_picker._recent_emojis_path`, `save_recent_emoji`; pending. |
| ui.themes | active config-parent/themes, every retained file and empty directory | `SettingsThemeEditor.__init__`, `on_save_theme`, `_delete_user_theme`; phase8 source lifetime, runtime composition pending. `on_export_theme` writes a separate external Downloads destination, never authority to scan home. |
| chatbooks.registry | `database_path(config, "prompts_db_path").with_name("tldw_chatbook_chatbooks.json")`, baseline opaque registry including provenance/outbox and retained records | `TldwCli._build_chatbook_db_paths` always supplies Prompts; `LocalChatbookService._default_registry_path` selects it first; `_load_registry`/`_save_registry`, create/update/delete and provenance mutations require full read-modify-save lifetime in10. Final dependency IDs are this profile's config and db.prompts.primary. |
| chatbooks.archives | data/chatbooks, every retained ordinary saved bundle and empty directory; baseline opaque bytes regardless of extension/name | `Chatbooks.database_paths.get_private_chatbooks_dir`, Chatbook creation wizard default destination and `ChatbookCreator._create_zip_archive`; full creator publish/cancel/cleanup lifetime pending10. ZIP contents are not opened or activated by recovery. A bundle does not become a recovery output merely because its source DB is captured or its filename resembles a backup. |
| runtime.instance_lock | exact data/.instance.lock, excluded PID/portalocker state | `Utils.instance_lock.acquire_profile_instance_lock`; never restore a PID/held lock. Wrong-kind paths, links and unsupported metadata refuse. |
| runtime.chatbook_scratch | retain data/temp topology; only data/temp/chatbooks and data/temp/imports have disposable purpose; other siblings unsupported | Creator `__init__` selects chatbooks, `create_chatbook` uses per-run mkdtemp and `_cleanup_run` in finally (471–475). Importer `__init__` selects imports, `_create_extract_dir` allocates per-run input extraction; both `preview_chatbook` (248–250) and `import_chatbook` (432–434) remove it in finally. Neither writes a durable recovery journal or catalog there. Destination DB mutations and the separate registry retain their own owners. Existing wrong-kind/linked/special paths do not inherit exclusions. Task10 drains creator/importer full lifetimes before capture; no cleanup is executed by discovery. |

The earlier `miscellaneous` scanner label is not an unassigned installed file cohort.
Examples reconciled above include JSON serialization into the core DB
(`ConsoleChatStore._persist_*`, `ConsoleContextRepository.finish_auxiliary_attempt`),
PIL/BytesIO memory rendering (`ConsoleImageRenderCache.prepare`), the already captured
`chat.rag_context`, `chunking.templates`, and `tokenizers.custom`, plus credentials14.
Generic exports such as `write_trajectory_export`, `ConsoleContextModal._save_json`,
`SettingsThemeEditor.on_export_theme`, library-note/prompt exports, and explicit
`LocalChatbookService.export_chatbook.output_path` are caller destinations; they
confer no app-owned root and are not silently added to baseline. A registry
`file_path` is an inert external locator unless its actual destination is already
inside the installed data/chatbooks owner or an explicitly selected external root.
This distinction preserves the registry bytes without arbitrary external traversal.
Actual existing unknown formats remain path-specific unsupported declarations,
not an indefinite generic omission of any identified installed durable default.


### Task9 review fix1: exclusion evidence and semantic directory dependencies

`file_inventory._inventory_root(root, *, owner, external)` is a private metadata
inspection of exactly one pinned inode. It reuses `_inventory_tree(root_only=True)`
with the same no-follow identity/kind/metadata checks and closes its metadata
handle before returning. It never visits child entries. `root_only` and
`selected_paths` are mutually exclusive; public/full-tree behavior and the builtin
empty-selected-set contract remain unchanged. This helper grants no capture,
exclusion or external-path authority.

Only installed config owner policy applies exclusions: `_excluded_root` requires
an observed regular file for catalog-cache/log artifacts and an observed directory
for unselected generated_videos. Positive leaf absence under a checked parent can
remain excluded; an unobserved parent is unavailable. Wrong-kind, linked, special,
unsupported-metadata or unavailable results remain refusals. Directory-shaped
cache/log paths are refused without walking their unrelated children. Failed
log-rotation namespace inspection produces unavailable owner evidence. In the
already enumerated generated-images tree, the unselected temp directory and its
checked children can be excluded, but every unsupported/unavailable entry remains
unchanged, and a regular file replacing the temp directory is unsupported.

Each persona semantic root (`persona.assets`, `persona.visual_identity`,
`persona.visual_identity_builtin`) declares its exact profile-qualified
`db.chachanotes.primary` peer before referenced payload edges. Staged validation
requires that edge in the root's dependencies before consulting its candidate;
an overbroad candidate map cannot supply undeclared authority. Tasks15/17 must
invoke `validate_dependencies` on semantic **included_directory** items as well as
file payloads. Creating directory topology instead of calling raw-file capture
must not bypass the graph check. These are declared semantic dependency cycles
with the core owner, to be resolved as complete groups, not recursive execution.

## Task 10 maintenance foundation — integration remains incomplete

`Admission.pause_requested(namespaces)` observes checked native registry/group/gate
contention without waiting for native locks. A true result is only a conservative
pause hint; it never acknowledges producer retirement or grants capture authority.
The same Admission instance remembers the gate identities used by normal admission
and refuses replaced gates or changed observed groups. A responder must reuse its
actual holder's authority, retain its native normal lease through real owner drain,
and fail closed on missing, pending, unsafe or otherwise uncertain evidence.

The process lease coordinator now counts pending native acquisitions before waiting
outside its RLock. Last-token retirement also joins outside the lock; retiring or
ambiguous native holds stay observable in `_retiring_holds` until positive retirement.
Future drain must include both `_holds` and `_retiring_holds`, as well as installed
producer-operation and native-handle tokens. Neither aggregate count is owner proof.
Startup acquisition coalesces concurrent callers without waiting under the RLock.
Startup holds still last until process exit; no release/reacquire or responder is
installed by this foundation.

`Participant` and `require_participant_coverage` define the planned protocol and
refuse uncovered path-bearing items. No passive owner exemptions or installed
participant factories are qualified yet. All preceding `participant_pending`
annotations and producer obligations remain in force, including actual Event/Sync
file connections, raw writers, cross-store publication and dirty editor boundaries.
Complete backup and replacement remain unavailable. Targeted foundation evidence:
16 focused tests; 79 combined participant/admission/bootstrap checks; 379 required
SQLite/census/service guards passed, with the existing Windows-only guard skipped
on this Mac. These overlapping runs do not prove full Task10 owner coverage.

### Task10 phase2: local gate and first native repository cohort

The private process-local pause gate now counts acquisition attempts before root,
bootstrap and authority lookup, every ordinary StorageLease, live installed
repository operations, and both active/retiring native holds. A timeout or canceled
pre-authority wait keeps its reservation until the actual caller returns; late
allocation cannot pass the closed gate. Actual holders retain their original
Admission instance and root/key for pause probing. Unresolved pre-root work makes
the local pause conservatively process-wide, reducing availability across roots.

The installed EventStateRepository and SyncStateRepository transaction scopes now
bind runtime.event_state/runtime.sync_state to the actual repository object and
selected path. File contexts commit/rollback and close the native connection on
its creating thread; memory contexts retain their persistent connection. Direct
private SQLite handles remain independently counted until native close. The actual
file seam still uses the existing db.base SQLite policy; no inferred stack/caller
owner or source-table exemption was added. Existing syntactic source rows remain
unchanged and the required census guards passed.

An operation's descendant authority is a live registered object bound to PID,
actual Thread object, asyncio Task identity, exact selected path/parent identity
and original native scope. Copied/stale tokens and cross-thread/task transfers
refuse. No ContextVar or caller owner string grants this authority. Ordinary
factory construction retains admission until final native retirement, including
constructors that close/reinitialize self or retain a handle then raise. Ambiguous
allocation/close failure conservatively retains the hold and may require restart.

This is still an incomplete Task10 cohort. No polling responder, startup release,
all-owner runtime coverage, service-wide pause, or Complete backup capability is
installed. Local zero counts describe observed cohort drainage only; advancing
requires participant_runtime_coverage_incomplete refusal and leaves startup
protection intact. Other retained SQLite owners, raw/cross-store producers,
app/headless composition, dirty editors, and async/thread handoff remain required
before any participant_pending promotion or startup retirement. Controller
rulings52–53 refine ADR-126 without changing capture custom-factory restrictions.

Phase2 verification also exposed a pre-phase2 combined-order lifecycle issue:
the required service-composition tests followed by core capture/bootstrap tests
produce recovery_scope_uncertain and late app.py7804 callbacks on both the current
code and clean phase BASE0f4ff4ba. Isolated core capture passes (62 tests). App7771
installs process-global ingestion hooks with bound notifiers; ingestion_indexing.py
provides uninstall_media_ingest_hook/reset_ingestion_indexer, but app has no matching
calls. The remaining Task10 app/lifecycle cohort must qualify actual callback,
worker and concurrent bootstrap authority ownership; exact causality is still to
be resolved. A generic stop/cancel is not a safe drain, and this debt is not moved
to Task26. Full commands/traces and blocked network-attempt disclosure are in the
phase2 execution report.

### Task10 phase3: core native lifetimes, with explicit borrower retirement

Under ADR-126 and controller rulings54–55, exact installed `CharactersRAGDB`,
`MediaDatabase`, `PromptsDatabase`, `LibraryCollectionsDB` and `LibraryIngestJobsDB`
instances now bind their selected paths to actual ordinary SQLite leases. The
private SQLite weak lookup is identity lookup only; the core participant strongly
retains every registered native handle until successful explicit native close.
Actual ordinary leases, pending acquisitions and retiring holds remain independent
blockers, including unregistered/failed constructor and unmatched factory returns.
The source census call counts are unchanged; no table row or semantic owner is
promoted to complete runtime coverage by these declarations.

Managed transaction scopes preserve existing native connection/cursor APIs,
including ChaCha nested/borrowed ownership and Collections read transactions.
A pause closes new cached getters and transaction admission. Existing exact
operations may finish; transaction exit never revokes an escaped native borrower.
Different installed participants may obtain independent ordinary admission only
while the local/target gates are open; they inherit no outer descendant authority.
A pause racing that fresh acquisition refuses it and leaves the outer scope live.

Explicit source-owner close reserves the source cache against new managed work,
closes outside coordinator locks and preserves references on failure. Per-thread
caches retire independently; Ingest's one retained connection protects operations
from all accessing threads and must close on its actual creating thread. Idle
foreign-thread caches, escaped cursors/connections and borrowed raw transactions
remain blockers until their owner explicitly retires them. A dead affine source
thread requires restart if its handle was not retired before exit. Memory remains
memory-owned; uninstalled subclasses retain ordinary behavior without installed
participant or descendant authority. Direct/qualified/imported production subclass
inspection found none in this cohort; the coordinated-reader test subclass remains
ordinary and unqualified.

This bounded phase is **not all-thread caller lifecycle qualification**. Required
next boundaries include `LibraryScreen._run_library_service_call` and its actual
Collections/Prompts/Media/ChaCha callers, the synchronous local services that return
materialized results to those pool jobs, `TldwCli._run_library_ingest_queue` plus
`LibraryIngestJobRegistry._persist`/`_persist_delete`/`requeue`, and the UI-owned
`_library_ingest_jobs_store` created by `_restore_ingest_jobs`. The current
`on_unmount` close follows parser shutdown and is not reusable pause/resume.
`ChatbookImporter.import_chatbook` owns multi-database transactions; Creator,
Persona publication and other DB-plus-file producers still require their full
operation and source-thread retirement boundary. A generic executor hook, source
name, zero count, GC, or `check_same_thread=False` does not prove borrower release.

Task31993 remains In Progress/all AC unchecked. Startup is retained, runtime
coverage still refuses, and no app/headless responder or Complete capability is
installed. The phase2 combined-order app callback/bootstrap failure remains an
explicit Task10 app/lifecycle obligation. The separately reproduced pre-phase3
Media historical-schema fixture belongs to later schema/migration qualification.

### Task10 phase4 — installed raw source lifetimes, runtime composition still pending

`Backup_Recovery/raw_participants.py` now enrolls the actual Feedback/Chat Grammar
services, default user ChunkingTemplateManager directory, and emoji module's selected
profile file. This is source-bound ordinary admission, not a capture permission or
runtime-coverage promotion. Every existing `participant_pending` remains in place.

Feedback and Grammar constructors/direct `_load`/`_persist` and full create/update/
delete methods admit the exact target, fixed `.tmp` publication path and every missing
parent before cached state or disk changes. A process-local physical-path lock covers
these synchronous bodies; rejection/failure restores `_records` and `_next_id`.
An existing `.tmp` is preserved/refused. Newly created sidecars have actual inode
ownership; successful rename consumes that ownership, so cleanup cannot unlink the
next process's same-name sidecar. This preserves last-writer-wins cached-store
semantics across independent service instances; it does not add cross-process CRUD
transactions or refresh stale cached records.

Native directory pins/descriptor-relative operations retain checked ancestor identity
through mkdir/open/rename/unlink. Actual file descriptors and text/native wrappers
remain strongly recorded until explicit native close. Failed close, ambiguous
publication, or failed owned-sidecar cleanup retain evidence and ordinary native
exclusion; later conflicting mutations refuse. They require explicit recovery/restart,
not garbage collection, a zero operation count, or a worker cancellation flag.

`ChunkingTemplateManager._get_user_templates_dir`, `save_template`, and direct
`_load_template_from_file` are gated. Built-in/custom directories retain ordinary
selected-path use; they receive no installed default-owner coverage or pause
descendant authority. The inventory remains `data/chunking_templates`, never package
templates or a caller-selected directory. Missing explicit custom save directories
still fail rather than being silently created. Linked/nonregular native targets
refuse before truncation; template names cannot escape the selected directory.

`save_recent_emoji` acquires its full read/modify/write lifetime on the actual app-owned
worker. Picker dismissal and Future/Worker cancellation do not retire that native
thread. Refusal before IO stays best-effort/silent; a still-running or uncertain
worker remains a drain blocker. Unserialized last-write-wins recents remain deliberate.
The direct read helper also holds its actual file lifetime. A changing selector
cannot bind one profile while writing another.

Local drain/fork accounting includes raw operations, pending selector/native/lock
acquisition, actual leases and failed retirement. Tokens have registry-backed exact
PID/Thread/asyncio Task identity, fixed scope and original native holder; copied,
foreign-task/thread, same-name/forged callback and sibling paths grant no authority.
Different core/raw operations get independent admission only while gates remain open.
No startup hold is retired and `_LocalPause.require_runtime_coverage` still refuses.
App/headless aggregation, remaining raw/compound/SQLite callers, dirty editors and
whole-task all-owner review remain required. Native filesystem evidence here is for
this POSIX host; unqualified platforms are not advertised as covered.

On platforms missing raw pinning primitives, explicit capability detection selects
ordinary path IO before any source access (ruling57). Existing verified-disjoint
bootstrap and enrolled-scope checks still govern all actual leases, including any
stronger native lease available. This mode keeps exact source/sidecar/resource
accounting but installs no raw participant or across-pause helper authority. Active
or uncertain work blocks drain; positive ordinary retirement does not qualify
capture, runtime coverage or startup retirement. Simulated missing-primitives
functional tests are not Windows qualification. Existing parent aliases are accepted
only while the selected path resolves to the same positively checked physical parent;
changed targets still refuse. Capture link policy is unchanged.

### Task10 phase5 — operational/domain SQLite lifetimes, caller jobs still pending

Exact installed WorkspaceDB (`db.workspaces`), AgentRunsDB (`db.agent_runs`),
ClientNotificationsDB (`notifications.client`), ScheduledTasksDB
(`db.scheduled_tasks`), LocalResearchService (`research.local`) and
LocalWritingService (`writing.local`) now register actual selected-source native
leases with the existing repository participant. The four BaseDB-backed file
routes really use native policy `db.base`; Research/Writing use their literal
service policies. This is an exact installed-type association, not authority for
BaseDB subclasses. Memory and Research external-DB injection remain unqualified.
Relative Research/Writing file paths bind once lexically at construction; memory
sentinels and existing alias/trust refusal stay intact. No schema, migration,
archive-owner classification or syntactic source count changes in this phase.

Workspace/AgentRuns/Notifications count existing connection and transaction scopes;
new cached/raw getters are gated. Their per-thread cached native connections remain
live after scope exit and may only retire through explicit actual source-thread
close. Close errors and failed live probes retain references. ScheduledTasks counts
its existing closing read scopes, transaction commit/rollback/finally-close, and
actual schema migration sequence at the same selected path. AgentRuns
`get_run_fresh` counts its separate read through its original finally.close.
Research/Writing count each lexical `_connection` scope through their existing
file-only `_OperationConnection` commit/rollback/native close. Their public close
still owns memory only; escaped raw `_connect` objects and uncertain setup/close
remain blockers. No high-level service method becomes a compound operation merely
because it calls several such scopes.

Controller-requested consistency checks reproduced the late raw-allocation/setup
gate race in the five earlier core owners plus Event/Sync. Their concrete getters
now recheck before returning a new native allocation, with positive close only of
that exact unpublished allocation on refusal. Collections' direct fresh getter
and Event/Sync's raw getters are also gated; Event/Sync file handles register their
actual `db.base` native policy. Existing borrowers are never revoked to make drain
succeed. Ingest keeps the exact new allocation local until setup completes, so a
concurrent cache change cannot redirect refusal cleanup to another native handle.

Concrete required next Task10 job/lifecycle boundaries:

- Workspace app composition (`app.py` local_workspace_db),
  `LocalWorkspaceRegistryService` CRUD/handoff methods, and source-thread callers
  returning to worker pools must positively retire WorkspaceDB caches after all
  native consumers finish. `LocalNotificationHomeActiveWorkAdapter.refresh_active_work_cache_async`
  dispatches `_compute_active_work_fields` with `asyncio.to_thread`; its notification
  reads likewise require actual worker retirement, not cancellation of the awaiter.
- `Chat/console_runtime.py` agent bridge construction owns sibling AgentRunsDB;
  `Agents/agent_service.py::_run_one`, settings_agents_panel `_derive_runs_db`, and
  `Workspaces/change_retention.py::run_retention_for_app` (constructs sibling AgentRunsDB
  before pruning) need whole source jobs and explicit cached-handle retirement.
  Agent DB scopes do not cover run transcripts/change snapshots/publication.
- `NotificationDispatchService.dispatch` reads settings, writes the inbox, and
  performs transient delivery as separate steps. ClientNotifications worker caches
  must close on their worker thread; higher-level research/scheduling/agent jobs
  must account for required notification followups across stores.
- `SchedulingService` async CRUD/sync/run_reminder_now and scheduler/automation
  execution remain full-job cohorts. Fresh SQLite closure does not prove queued
  dispatch, server followups or cross-store delivery has finished.
- `LocalResearchService._update_run` reads, writes/event-records, rereads and invokes
  `_dispatch_terminal_run_notification` in separate scopes. `LocalResearchEngine`
  execute_run/keepalive/pipeline offloads remain full-job work. External DB injection
  is never local Research authority. Writing create_version/restore_version,
  `_version_payload_for`/`_next_version_number` and analysis helpers compose multiple
  database scopes; service-wide coherence and caller ownership are not granted.

Actual app/headless composition, other persistence cohorts, all-source runtime
coverage, startup retirement/reacquisition and one independent whole-Task10 review
remain required. All participant_pending classifications stay; no production
responder, Complete backup or replacement exposure. The phase2 clean-BASE
combined-order app callback/bootstrap issue remains Task10 app/lifecycle work.
The separately baselined malformed Media historical-schema fixture remains Task13
schema/migration qualification, without a schema-stamp workaround here.

### Task10 phase6 — remaining local SQLite source lifetimes

The exact loaded production EvalsDB, SubscriptionsDB, FileNotesReplica,
NoteImportReceiptRepository and LocalKanbanService types now use the installed
repository/native identity protocol. These five and SiteConfigManager are looked
up only in their fixed, already loaded production modules; generic getters do not
import these additional source modules or the manager's config/encryption stack.
This is narrow lazy dispatch, not optional-dependency or runtime-coverage proof.

| Source scope | Semantic participant | Native policy and actual retirement |
| --- | --- | --- |
| EvalsDB lexical schema/CRUD/read `connection()` | db.evals | db.evals; retained `_local.connection` closes explicitly on its creating thread after managed work. |
| SubscriptionsDB schema, transaction, direct reads | db.subscriptions | db.base for writes; db.subscriptions.agent_read only for fixed read-only mode, revalidated on every association/use. Retained `_local.conn` requires positive source-thread retirement. |
| FileNotesReplica locked schema/read/transaction | notes.file_notes | notes.file_notes_replica; retained shared `_connection` refuses foreign-thread close and all active managed borrowers. RLock/check_same_thread=False does not revoke escaped native users. |
| NoteImportReceiptRepository transaction | notes.sync_state | notes.sync_state; existing BEGIN/schema/commit-or-rollback/finally.close retained. `_database_path` binds once, exposed read-only as db_path. Memory stays rejected by the existing file-only policy. |
| LocalKanbanService connection/transaction/schema | kanban.local | kanban.local; actual service owns each lexical native scope through its original finally.close. Public `local_kanban_db.open_connection`/schema/transaction helpers stay ordinarily fenced without source authority from a path. |
| SiteConfigManager initial table setup only | db.subscriptions.site_configs | db.subscriptions.site_configs; native setup closes before its independent CharactersRAGDB construction/use. Same-path source identities never share or reassign native lease authority. Standalone ensure_site_configs_schema stays ordinary. |

FileNotes directory construction is two independent safe stages. Its file selector
is bound before the constructor-only raw directory route admits the fixed database
path and actually missing parent targets. Existing ancestor identity is checked and
pinned before descriptor-relative mkdir; portable ordinary behavior remains
unqualified. Positive directory-resource retirement precedes a fresh repository
admission. Pause before creation has no directory effect; pause during admitted
mkdir may leave only those successfully admitted directories, then refuses before
SQLite. Retargeting between stages refuses. Directory close uncertainty retains
actual resources/leases and blocks maintenance; no cleanup removes existing dirs.
The raw constructor's internal notes.file_notes_replica policy identity is not a
second semantic coverage claim or a transfer of authority into the SQLite stage.

All new allocations register before source setup and recheck admission before raw
return. The actual Kanban service passes its exact source to the native helper;
standalone helper calls retain only ordinary native accounting. A real DELETE-to-WAL
mutation after source pause exposed and verified this pre-PRAGMA gate boundary. Failed setup closes exactly that unpublished native allocation, or retains
uncertain evidence. Cached live errors preserve references; only positive native
retirement allows file-cache reuse. Native cursors/connections retain their existing
contracts; Evals/Subscriptions managed exit does not close escaped cached users.
FileNotes closes its cursor at its existing transaction exit; memory retains its
original closed-handle behavior. Receipts/Kanban commit/rollback/close boundaries
are unchanged. Source scopes are not whole job or cross-store publication scopes.
Subscriptions reads an omitted default threshold before its DB operation, preserving
cold-config values and explicit overrides; a subsequent pause still refuses the
fresh DB mutation. Exact pre-existing plain/joined SQL literal payloads are retained,
including SQLite's stored DDL bytes; no catalog/migration exceptions were added.

Concrete remaining caller handoffs (required later Task10 work):

- `Evals/eval_orchestrator.py::_initialize_database` and actual run execution retain
  EvalsDB; `Event_Handlers/eval_db_operations.py::EvalDBOperations` and its
  `get_eval_db_ops` singleton retain another source. Their actual workers, result/
  asset publication and snapshots must finish/materialize native consumers and
  positively close each creating-thread cache before idle.
- `Widgets/Library/library_file_notes_workspace.py::_build_runtime` constructs and
  retains FileNotesReplica even when its awaiting mount is cancelled. Its
  `shutdown` dispatches `_close_owned_replica` with `asyncio.to_thread`; this need
  not be the native creating thread. `_close_owned_replica` currently clears the
  workspace reference after service/replica close without proof of native retirement.
  Integrate these actual source workers and FileNotesService/session/file/Git/autosave
  boundaries, dirty draft/save-discard state and root commit before retiring startup.
  A foreign close now truthfully leaves registered native evidence blocking drain.
- `Notes/note_import_executor.py::NoteImportExecutor.execute/retry_failed` receives
  the exact receipt repository and composes its ledger with target note/folder/
  membership effects. Per-transaction retirement does not reconcile unfinished
  effects or qualify import/sync job coherence.
- `Subscriptions/site_config_manager.py::get_site_config_manager` retains a hybrid
  manager used by `Subscriptions/web_scraping_pipelines.py`. Real subscription,
  briefing/audio/asset, agent-read and scraping jobs need whole-job drainage and
  native source-thread close. Its setup-only participant never stands for the whole
  Subscriptions/core42 hybrid payload or replaces either real native source.
- `app.py` creates the installed LocalKanbanService. Its `copy_card` and other
  methods can compose multiple separately committed SQLite scopes/awaits. Actual
  app/policy/service jobs must own complete work; direct native helper users remain
  blockers until positive close rather than receiving path-derived authority.

Final composition must aggregate these concrete source routes and shared semantic
payloads. Full runtime census coverage, startup retirement/reacquisition, responders,
Complete/replacement exposure and independent whole-Task10 review remain unavailable.
Earlier phase2 app/bootstrap ordering and other named caller cohorts remain required;
this phase does not convert pending declarations into runtime coverage.


### Task10 phase7 — PromptHistory and ChatScreen async file lifetimes

The exact default `PromptHistory` instance and exact loaded production `ChatScreen`
are raw sources `chat.prompt_history` and `ui.state`. Default history path and
active config-parent UI-state path are resolved before queued dispatch and checked
again in the actual worker reservation. A bound source refuses selection changes;
it cannot become an ordinary custom source to bypass a closed gate. Originally
custom history paths/subclasses remain ordinarily fenced and uninstalled. Factory
return values, class-name/module-name labels and instance callbacks confer no
source authority. `UIState` remains the data class, not the on-disk owner.

`Backup_Recovery/async_file_participants.py::_FileJob` registers one pending
acquisition before selection/snapshot/cache changes for only these two routes.
It grants no transferable IO permission. The worker invokes the actual class
method and acquires its own fixed raw scope, including selected file, fixed `.tmp`
publication sidecar and any missing directories. A pause before native admission
can refuse a previously queued job with no source mutation. Thread/task/copied-job
provenance is checked; raw/native authority stays on the real worker thread.

A synchronized queued/running/cancelled transition proves never-started work cannot
enter its source, even if the executor later invokes the disabled callback. Running
cancellation retains source serialization until actual worker completion and
creator bookkeeping. The actual worker completion signal is independent of the
executor's cancellable asyncio wrapper; shutdown/failed enqueue/queued cancellation
retire their pending records positively. The worker never waits for an event-loop
acknowledgement. File/native close or publication uncertainty remains in the raw
registry and blocks drainage; a cancelled/finished Textual worker is not retirement.

History load and append share one async lock. Appends use immutable payloads and
publish cached entries only after successful IO; capped rewrites publish an owned
sidecar. A draft stashed during a pending append survives its completion. Native
append errors that may have changed durable bytes retain uncertain source evidence.
`persistence_safe_point()` waits for actual history bookkeeping and returns false
when `persistence_error` remains; it does not load, save or discard a live draft.

Sidebar debounce remains an exclusive Textual worker. A source lock lasts through
native result delivery, and revision comparison clears dirty only for the current
successful snapshot. `_flush_sidebar_state_now()->bool` waits on that source lock,
flushes the latest dirty UI preferences while ordinary admission permits, and
returns false on failure/refusal while preserving dirty/error state. Direct
`_load_sidebar_state` and `_save_sidebar_state` use fixed raw scopes too. TOML
read/merge/sidecar publication preserves unrelated sections and previous good bytes
on a failed write. Pending queued snapshots cannot redirect to another profile.

Actual caller handoff: `ConsolePromptsController._ensure_console_prompt_history`
retains the shared default instance; `console_prompt_history_factory` is an
external/test seam, not installed authority. Composer mount warm-load and exclusive
recall share the history lifetime, and accepted-send `_record_prompt_history`
remains best effort on persistence refusal. `ChatScreen.on_unmount` uses the same
flush API; later maintenance composition must explicitly observe its false result,
dirty revision and error, plus history's safe-point result. It must close/gate
future mutations and reconcile late dirty data, not infer drain from timers,
Textual flags or an emptied reference, and cannot flush after pause by claiming
new permission. Independent instances retain existing cached last-writer behavior;
this phase adds no cross-process CRUD transaction/cache-refresh protocol.

Post-worker creator bookkeeping performs no further history/sidebar persistence.
The existing process startup native hold remains until a future qualified
whole-process local drain observes all pending work retired; that hold must be
reacquired before reopening admission. Tests use actual startup plus an independent
native maintainer to verify this composition, without enabling startup retirement.
Persistent/queued diagnostic sinks still have their separate census/drain obligation;
logging is not claimed globally IO-free. All other caller/job cohorts, full runtime
coverage, app/headless pause composition and independent whole-Task10 review remain
required. Earlier phase2 combined-order callback/bootstrap and Task13 malformed
Media-schema fixture debts remain; this cohort's focused runs did not reproduce
them. No Complete/replacement capability or responder is enabled.

### Task10 phase8 — settings, definitions and shared note-template files

This cohort adds actual source lifetimes for `RuntimeSourceStateStore`, the installed
`EvalConfigLoader` default YAML, the shared `Notes.template_store` module,
`SettingsThemeEditor` and default `ConfigFileStorage`. Exact classes, module identity
and current selectors determine installed posture; custom Eval/JSON paths and theme
exports remain ordinary admission. Previously installed sources cannot retarget or
demote to ordinary to evade a closed gate. Runtime composition remains unavailable.

Each operation reserves before selection and holds its source path lock through
serialization, native IO and positive retirement. Runtime private binary/atomic
helpers discover only their live source operation internally; they have no new
public authority arguments. The random runtime temporary is selected before helper
side effects. Existing parent/posture algorithms remain, including ordinary
unverified-platform behavior. The runtime source tracks helper traversal descriptors,
streams, publication and cleanup identities; ambiguous close retains native evidence
and prevents drain, with no close retry or garbage-collection retirement.

Pet backup and theme directory membership use two-stage preflight. The initial
ordinary parent admission and directory pin precede enumeration of exact TOML or
`<stem>.backup_YYYYMMDD_HHMMSS.json` entries. Matching members are bounded, fixed and
individually admitted while the initial lease/pin remains held. Selection, pause,
parent and observed file identities are rechecked before mutation. New or replaced
entries are never pruned by a stale name. Pet timestamp collisions preserve the
existing same-second overwrite policy: the one backup contains the primary bytes
from immediately before the latest save. Exact bytes are copied through an admitted
exclusive backup sidecar and atomically published; stale unowned sidecars refuse.
This is not constructor recovery replay during archive capture.

Eval mutable `get()` results remain supported. `_persisted_config` and
`persistence_error` expose truthful draft state; failed reload/save, including a
final native directory-close failure, retain the previous draft/saved marker.
`persistence_safe_point()` reports `needs_user_save_or_discard`,
`persistence_failed`, or `ready`. The raw participant refuses drain for dirty/error
Eval state or a theme's actual `is_modified` flag. Theme save/delete success and tree
changes follow positive source completion; existing confirmation dialogs remain.
Exports never mark an unrelated draft saved.

The CLI gathers input before native admission, then shares `merge_templates` with
the actual template importer. The admitted RMW rereads the latest file under the
shared in-process path lock and refuses corrupt existing data. The actual async
importer is the third concrete `_FileJob` dispatch route, with queued/running
cancellation and pending result bookkeeping; final write failure never reports
imported templates as successes. `notes_events.load_note_templates` uses the same
user-file reader; shipped fallback reads remain separate immutable input.

Ordinary cross-process CRUD still permits concurrent shared native writers. A
ready-handshake, post-read template lost-update schedule fails on both phase8 and
immutable phase BASE 838b949ac7aa9c92988d79fe725025e8ffc21a79. This is an existing
limitation, not passing coverage or a new cross-process transaction guarantee.
Maintenance exclusion is tested independently; no global writer mutex was added.

Remaining handoffs: actual app/headless maintenance composition must retain these
source instances, close gates, observe dirty/error state and positively drain before
startup hold retirement. Non-template note import still has its separate DB/Notes/Git
compound worker lifetime; config persistence still uses ordinary private helpers and
portalocker and is not covered by the runtime-only hook. Persistent logging, other
pending census cohorts, the responder and independent whole-Task10 review remain.
No runtime/startup/Complete capability is enabled by this phase.

The separate ADR-040 profile-path guard currently has two preexisting failures in
untouched `TTS/recovery.py:111` and `:115`, for the literal legacy
`~/.config/tldw_cli/chatterbox_voices` and `~/.config/tldw_cli/higgs_voices` paths.
Both `test_production_profile_owned_path_inventory_is_exact` and
`test_cli_prints_the_real_source_census_and_enforces_it` fail on current code and
immutable phase8 BASE. Reconcile these concrete entries in the pending Task10 TTS
cohort; the passing backup producer census does not erase that separate debt.

### Task10 phase9 — configuration source and concrete private resources

The actual installed `tldw_chatbook.config` module now binds the `config` raw
participant through `Backup_Recovery.config_participants`; a module name supplied
by a caller is not authority. Its lexical effective selector is fixed for the
installed module. Environment retargeting or loss of native qualification cannot
demote an existing binding into an ordinary source. Pure `profile_paths` selectors
remain independent of config import; `admit_startup()` still runs before config's
runtime imports and module-bottom bootstrap.

The whole source operation reserves admission before selectors/process-lock waits,
uses the existing `_CONFIG_FILE_LOCK` RLock, and fixes current config, stable
`.toml.lock`, `.toml.bak`, selected snapshot and each random atomic temporary as
separate members before side effects. Snapshot selection belongs to
`config._config_snapshot_path`. Cross-process serialization remains portalocker;
nonblocking retries observe pause requests without holding a coordinator lock.
Unlock exceptions are logged; only positively successful stream and FD closes
retire native evidence. The lock entry is never deleted/recreated to release it.

Covered routes: import/load/cache/forced bootstrap; `_prepare_config_parent`;
raw and serialized readers; `.bak` reader; `_write_raw_cli_config_unlocked` and
`_write_serialized_config_artifact_unlocked` (including direct helper calls);
whole/raw replacement; exact set/delete and revisioned section writers;
`get_atomic_config_snapshot`/`get_runtime_config_snapshot`; encrypted shutdown
persistence; enable/disable/password-change. Serialization, encryption, atomic
publication, reload and generation bookkeeping remain inside the operation.
Same-source descendants reuse only already fixed members on the actual Thread and
Task. The private helper dependency remains a leaf: active-operation discovery
uses `sys.modules`, and accepts only config or the prior RuntimeSourceStateStore
source. Every actual helper descriptor and config lock/read stream remains strongly
tracked until positive native retirement; failure after close still retains evidence.
The directory verifier accepts only the source parent or declared source directories.

`load_settings`, `get_user_data_dir`, and `get_model_cache_dir` retain their actual
mkdir behavior. Canonically selected data/profile, chat-dictionaries and model-cache
directories use separate concrete admitted operations, with selected ancestors and
source checks. They keep enclosing config work live, acquire new scope only with
open gates, and cannot expand an admitted config operation after pause. Custom
config parents are verified only; only the default application-owned config parent
is created/hardened. Unsuccessful bootstrap returns its established uncached display
defaults and parse failure signal without creating data directories from those
unusable defaults. Existing storage Save remains restart-required (ADR004), and
credential precedence/format and three Settings commit models remain ADR012/033.

Failure restores prior cache/settings/password/first-profile marker references and
runtime generation under the same process RLock, including final raw-pin close
failure. This is memory restoration, not disk rollback: bytes can already have
committed, and `ConfigMutationResult(file_replaced=True, caches_reloaded=False,
failure_phase="cache_reload")` remains truthful. `_CONFIG_PERSISTENCE_ERROR` retains
actual operation failure through cached reads and clears on successful publication;
config participant drain also requires a usable cache and no recorded parse failure.
An uncertain native close cannot be repaired by a later call, GC or registry reset.

Later Task10 composition must retain this exact config module/participant, close its
gate together with actual Settings staged/raw editors, retain their dirty drafts,
observe config publication/error state, and drain source threads/native resources.
Settings/app/headless callers must handle refusal and committed-but-unpublished
results without reporting save success or automatically discarding drafts. Actual
retained DB services and model runtime remain separately owned and are not moved,
reconnected or qualified by directory setup. Full local drain must include pending
post-IO bookkeeping before any future startup retirement; startup reacquisition
must precede reopened gates. No app responder, runtime completeness, startup release,
`config.history` capture adapter or Complete capability is enabled by this cohort.

Phase9 generation clarification: derived operations fix and recheck `_CONFIG_GENERATION` as well as the canonical path. Existing process RLock excludes concurrent publication, and a real reentrant publication from a derived scope is already refused by fixed config-member checks before either config bytes or the model directory changes. The explicit generation check is supplemental defense, not evidence of a previously demonstrated race.

### Task10 phase10 — actual chat core/sidecar sources and dictionary files

`chat_source_participants.build_persona_service/build_dictionary_service` bind the
actual app sources to the exact configured registered CharactersRAGDB and canonical
sidecar before load. `bind_citation_services` validates the real app's service,
repository and migration relationship. `operation` simultaneously exposes the
existing validated core/raw discoveries only for that same fixed pair and execution
identity. Other source instances, companion changes, path/DB/profile retarget and
foreign Tasks cannot inherit it. The seven Persona cache categories, dictionary
history and compatibility citation cache retain prior usable state on failure;
known possible mixed effects keep a sticky drain blocker. No durable rollback,
automatic reconciliation or restart-as-repair claim is made. Ordinary custom,
subclass and explicit-memory sources remain unqualified.

`dictionary_source_job._DictionaryJob` covers the actual local scope service's
queued/running worker and creator outcome lifetime. Exact methods acquire their
own source-thread scopes; queued cancellation prevents source code, running and
repeated cancellation waits for the actual callback, and independently cancelled
executor wrappers are not completion evidence. A newly opened worker connection
is positively closed there; an existing worker borrower remains live. No caller
DB is closed by synchronous service cleanup. Actual app startup remains necessary
through creator pending bookkeeping and is not retired by this source slice.

`dictionary_file_participants` fixes actual module/config/DB, all input/output/temp
members and parent identities before parser/import/export/listing effects. Export
uses its materialized preflight record; folder setup reuses phase9 config behavior,
without new config mkdir/scope inside a paused combined operation. Copy preserves
bytes, mode/times, platform xattr semantics and BSD flags on retained descriptors;
Darwin libc.fchflags is the bounded compatibility binding. Unsupported primitives
keep ordinary unqualified behavior. A copied sidecar followed by core failure is
sticky despite later success. Caller export/input paths are ordinary admitted IO,
not new `chat.dictionaries` roots. External editors do not cooperate by implication.

Canonical citation migrations retain actual concurrent claim/generation fencing;
compatibility cache writers retain their actual source lock. Inventory/capture does
not invoke migration. Whole runtime safe points, dirty/error handling and startup
release/reacquire still need real app/headless integration. Persona/dictionary UI
import/export Textual workers (`UI/Screens/personas_screen.py`), visual authoring, other
remaining persistence cohorts and aggregate app shutdown are not retired here.
Source participants do not qualify payload validation or a Complete archive. The
phase2 app callback/bootstrap, phase8 TTS profile literals, Task13 Media historical
fixture and Task26 diagnostic census obligations remain explicit outstanding work.

### Task10 phase11 — actual MCP local stores and execution history

The exact five constructor-selected source families now participate through
`Backup_Recovery/mcp_source_participants.py`: `LocalMCPStore` (`mcp.local`),
`ConfiguredServerTargetStore` (`mcp.targets`), `UnifiedMCPContextStore`
(`mcp.context`), `MCPPermissionStore` (`mcp.permissions`) and `MCPExecutionLog`
(`mcp.history`). Constructor admission begins before default config/path lookup.
Binding captures the actual config module, selected profile and canonical data
path; later verification uses the selected config cache without creating config
folders. Previously installed path/profile/config/native-posture changes refuse.
Custom paths and subclasses keep ordinary behavior and cannot borrow installed
across-pause authority or expand the baseline inventory.

All actual instance read/RMW methods hold the same source operation, through
serialization, result and native cleanup. Target storage preserves its class RLock;
other families use their existing or new instance RLock. This does not add a
cross-process CRUD mutex or change the documented last-writer contract. JSON readers
and owned fixed-temp publications use retained descriptors for installed sources.
Permission reads preselect active, `.tmp` and `.bak` because corruption recovery
moves the source and replaces the prior backup. Missing and corrupt payload policy
is preserved for ordinary use; admission/native uncertainty propagates without
resetting policy. Caller `updated_at` is published only after successful native
retirement, under the source lock. Capture must never call corruption recovery.

History fixes active and `.1`, both private random temporaries and the app-owned
parent before effects. Only actual installed history discovers the four exact
private helper routes: secure parent, binary generation read, atomic generation
write, active append stream. Existing private posture checks remain. Native
PID/Thread/Task/scope validation and strong FD/stream ownership last until explicit
successful close. A temp consumed during migration may be used again by rotation
only if absent; foreign recreation is preserved. Source/destination expectations
are fixed before IO and update from the positively owned published inode, never
from a newly discovered pathname. Partial generation failure and uncertain native
completion retain sticky source error; unrelated reads/success cannot establish
repair. Metadata sanitization, torn-line migration, two-generation cap and newest
ordering remain. Capture uses inert bytes and never `read_recent()` migration.

Source-only native evidence runs in isolated test children: each explicitly retires
only its own known quiescent startup token. Actual permission/history close-before
and close-after ambiguity retains native exclusion; independent maintenance enters
after successful rotation's actual resource close. Separate existing startup plus
pending event-loop bookkeeping evidence remains required. These tests do not retire
application startup or qualify a whole async service, execution, subprocess,
credential or UI job. Exact-file native enrollment refuses unowned sidecars; no
None native hold grants across-pause source authority.

Concrete remaining Task10 boundaries (source observations, **not exclusions**):

- `app._wire_server_context_provider` constructs canonical targets, calls
  `upsert_legacy_config_target`, then selects the credential backend. Actual
  `_wire_watchlists_and_notifications_services` composition constructs canonical
  local/context stores; unified service properties
  retain history/permissions siblings of `local_service.store.path`.
  `MCP/server.py::_register_local_agent_tools` constructs canonical standalone
  permissions. Aggregate startup/shutdown/cancellation and credential selection
  still need their complete job boundary.
- `UnifiedMCPControlPlaneService._audit_downgrade_if_fresh` commits
  `mark_config_changed`, then appends a best-effort history record. If the append
  fails, the marker persists and the next resolution no longer retries that audit.
  Five store lifetimes do not solve a pause between these writes. Future concrete
  source/service integration must preadmit the actual pair or retain known failed
  publication evidence; no general transaction facade or journal is introduced.
- `execute_hub_tool`, `test_hub_tool`, `execute_advanced_tool` await real local/server
  work before `_record_tool_execution`; `LocalMCPControlService.connect_profile`
  awaits client connection/description before saving discovery, and `_describe_profile`
  also persists snapshots. Runtime requests/batches/tools/resources/prompts record
  activity after primary work. `_run_local_lifecycle` times out/cancels awaiting
  operations and then `_record_local_attempt` separately reads/writes runtime state.
  Exact pending dispatch, running cancellation, actual worker/transport/process
  completion, native borrowers and final audit/cache bookkeeping remain required;
  an awaiter's cancellation is not native completion.
- `_apply_server_access_context` and source/server/scope/section selectors replace
  `self.context` before `_persist_context`; service cache truth spans the store call
  and remains a later service-boundary requirement. Restoring context cannot connect
  or activate anything. Approval resolution, governance and authority scope values
  retain their current semantics and imported values never authorize execution.
- `runtime_policy/server_credentials.py` scoped secret set/delete plus credential
  index updates and runtime keyring callers remain the credential cohort. Only fake
  external backends may be used for its runtime tests; archive credential export,
  exclusion and rollback mapping remain Task14. Server-owned data stays separately
  excluded, and no live transport, tool execution or real keychain is tested here.

The diagnostic source guard was run and remains **failing on inherited clean-BASE
drift**. This phase changed only indentation of existing MCP diagnostic/sink call
segments; AST argument/message comparisons are identical. The three affected owner digests and history sink entries were reviewed against
the actual AST; after the old save-only wrapper indentation was removed, those
entries already matched the checked manifest, so no manifest change was needed. BASE/current immutable exports
and the exact remaining delta are external `/private/tmp/task10-phase11-diagnostic-*.json`.
This does not waive or claim the full diagnostic census green. Runtime coverage,
startup release/reacquire, responder and Complete/replacement exposure remain off.

### Task10 phase12 exact changed-source census

Persona Visual source/native participation is implemented for the exact canonical
`PersonaVisualRepository` core binding; `publish_persona_visual` and issued
publication cleanup; workspace create/stage/adopt/cleanup; import external read,
fixed extraction, source-root verification and cleanup; and secure/fallback asset
reads. `persona_visual_participants` keeps every operation, pre-body allocation
uncertainty and unretired FD in the existing storage pause/drain registry. It
provides source `close_admission/drain/resume`, `safe_point(profile)` and the actual
screen's `persona_visual_maintenance_state()`. These return source/dirty/pending
evidence; they do not qualify application capture or release startup admission.
Ordinary external sources receive no installed across-pause authority and remain
visible in storage's operation/pending registry while active.

The actual Persona screen reserves its snapshot/configure, staging, import,
preview, publication and cleanup lifetimes through real native callback/result
completion. It closes only a new source-thread DB borrower. Published native-close
errors retain materialized identities and error/cleanup state; only an actual
matching positively cleaned candidate clears its own blocker. Maintenance
inspection never invokes `_drain_persona_visual_authoring`'s user-discard route.

Phase12 handoff (resolved for bounded source/native jobs in phase13 below): screen
`_save_visual_identity_pack` still calls generic `_drain_to_thread` for
`publish_visual_identity_candidate` and `cleanup_visual_identity_publication_candidate`;
its actual candidate/style/stage/reference/provider generation/cancel paths and
source-thread borrowers still need independent source qualification. The shared
helper's cancellation fix grants no IO permission. `_save_character_worker`'s existing
`persist_character` callback also uses that lifetime helper and remains a separate
actual source/caller boundary. Existing Shared Visual Identity cancellation and
character persistence cancellation checks exercise the helper change only.
`Chat/attachment_core.py`, Character_Chat expression/visual-identity stores, image
provider jobs, app/headless aggregate quiescence, caches and the local responder
must not be marked passive or excluded by this Persona-only evidence. Runtime
asset reads here do not authorize arbitrary renderer/generation callbacks.

| Module | Symbol | Call | Count | Classification | Cohort |
| --- | --- | --- | ---: | --- | --- |
| tldw_chatbook/Backup_Recovery/persona_visual_participants.py | _open_native | open | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Backup_Recovery/persona_visual_participants.py | mkdir | mkdir | 2 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Backup_Recovery/persona_visual_participants.py | unlink | os.unlink | 2 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/assets.py | _decode_selected_frame | open | 1 | memory | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _decode | open | 1 | memory | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _write_private | write | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _write_workspace_asset | write | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | create_persona_visual_authoring_workspace | mkdir | 2 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | create_persona_visual_authoring_workspace | secure_private_directory | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/importer.py | _create_candidate | mkdir | 2 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/importer.py | _extract_assets | open | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/importer.py | _inspect_image | open | 1 | memory | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/importer.py | _member_digest | open | 1 | memory | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/importer.py | _private_staging_root | secure_private_directory | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/importer.py | _write_all | write | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/importer.py | import_persona_visual_pack | ZipFile | 1 | memory | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/publication.py | _write_private_file | write | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/publication.py | cleanup_persona_visual_publication_candidate | secure_private_directory | 2 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/publication.py | publish_persona_visual | mkdir | 2 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Persona_Visual/publication.py | publish_persona_visual | secure_private_directory | 1 | generic_boundary | concrete Persona Visual source/native lifetime; aggregate app qualification pending |

### Task10 phase13 exact changed-source census

| Module | Symbol | Call | Count | Classification | Cohort |
| --- | --- | --- | ---: | --- | --- |
| tldw_chatbook/Backup_Recovery/visual_identity_participants.py | _open_native | open | 1 | generic_boundary | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Backup_Recovery/visual_identity_participants.py | mkdir | mkdir | 2 | generic_boundary | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Backup_Recovery/visual_identity_participants.py | resource_stream | open | 1 | generic_boundary | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Backup_Recovery/visual_identity_participants.py | unlink | os.unlink | 2 | generic_boundary | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Character_Chat/visual_identity.py | _inspect_image_bytes | open | 1 | memory | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Character_Chat/visual_identity.py | _write_private_publication_file | write | 1 | generic_boundary | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Character_Chat/visual_identity.py | _write_private_publication_path | write | 1 | generic_boundary | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Character_Chat/visual_identity.py | cleanup_visual_identity_publication_candidate | secure_private_directory | 1 | generic_boundary | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/Character_Chat/visual_identity.py | publish_visual_identity_candidate | secure_private_directory | 2 | generic_boundary | concrete Shared Visual Identity source/native lifetime; aggregate app qualification pending |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._dictionary_export_worker | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._dictionary_export_worker | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._export_expression_set | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._export_expression_set | write_bytes | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._write_text_file | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/private_paths.py | _MCPAppendStream.write | write | 1 | generic_boundary | exact MCP history append native lifetime; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | _native_mkdir | mkdir | 2 | generic_boundary | private helper; exact visual secure-directory native bridge; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | _native_open | open | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | _prepare_application_owned_parent | secure_private_directory | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_bytes | os.replace | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_bytes | os.unlink | 2 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_bytes | write | 2 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_text | atomic_private_write_bytes | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | create_private_text | open | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | create_private_text | secure_private_directory | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | create_private_text | write | 2 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | open_private_binary | open | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | open_private_text_append | open_private_text_append_stream | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | open_private_text_append_stream | mkdir | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | open_private_text_append_stream | open | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |
| tldw_chatbook/Utils/private_paths.py | secure_private_directory | mkdir | 1 | generic_boundary | private helper; exact active runtime source adoption only; other sources ordinary |


### Task10 phase13 — reviewed Shared Visual Identity lifetimes and remaining callers

The new concrete bridge `Backup_Recovery/visual_identity_participants.py` uses
existing storage pending/native leases and registered core operations for actual
`Character_Chat/visual_identity.py` candidate creation/publication/cleanup, selected
manual/builtin reads and `ensure_builtin_samira`. Public instance operations in
`DB/VisualIdentity_DB.py` retain their actual core borrower. Publication fixes old
source members, new staging/final leaves and directories before effects; only
positive native retirement permits completion. Installed sources bind exact loaded
config/profile/core/candidate identities. Ordinary custom/portable resources do not
inherit installed authority. Public fork, binding/version, atomic replacement and
partial card-then-pack seed behavior are preserved. Native uncertainty is sticky.

`Utils/private_paths.py` discovers either concrete visual source only for the
existing `secure_private_directory` helper and fixed selected directories. Generic
file/callback APIs remain ordinary boundaries. Its shared raw and Persona Visual
checks pass. This source bridge does not introduce a global source mutex.

`PersonasScreen._visual_identity_thread` admits actual allowlisted source jobs and
retains new DB borrower retirement through repeated cancellation; preexisting
borrowers are preserved. `_save_visual_identity_pack` retains issued error/cleanup
references and reconciles committed results even when native close is uncertain.
`visual_identity_maintenance_state` and source `safe_point` preserve pending or
unsaved drafts, including canonical row restoration with no replacements yet.
`_restore_candidate_reaction_rows` qualifies only the exact original candidate and
canonical missing path-free rows under its lock, never arbitrary shape refresh.
Copied relpath strings retain only ordinary fresh cleanup behavior and cannot
clear an installed failure; exact original source/native cleanup is required.

Rulings76–78 narrow both public callback boundaries: the actual atomic replacement
invocation and both repository publication guards suppress only visual/core
discovery while retaining all native/pending owners. Three exact original
same-repository result-read edges can reuse their existing validated core scope;
public callbacks cannot. Truth-value evaluation remains inside suppression.
Source error plus borrower-close failure retains its original exception/token and
truthful result in the screen, including cancelled workers. The analogous earlier
Persona Visual callback seam is an explicit required Task10 follow-up, not changed
or qualified by this phase.

Concrete remaining Task10 routes (mandatory handoff, not owner exclusions):
- `PersonasScreen._generate_visual_identity_assets_admitted` calls
  `_run_visual_identity_generation_request` and its real image generation worker,
  adapter/provider/native runtime. This phase uses synthetic generation sentinels;
  UI pending/cancellation tracking does not qualify runtime termination or release.
- Persona Visual generation runtime, local actor service callbacks, remaining
  TTS repository/runtime and credential owners still need their concrete cohorts.
- Actual app/injected/eager/lazy and headless maintenance aggregation must collect
  both visual sources, dirty editors, pending jobs and retained failures; bind the
  existing startup/admission/responder lifetimes without automatic draft discard.
- Whole Task10 independent review and aggregate qualification remain required.
  Task31993 stays In Progress with all ACs unchecked; no startup/Complete release.

Exact diagnostics are compared against phase13 BASE b4537258. Only changed visual
source/UI digest rows may be refreshed after identical diagnostic-call AST proof;
inherited owner/config sink drift and the two TTS/recovery.py profile literals
remain recorded debt, not green evidence. Full command/results and private native
observer logs are in the phase13 dual report.

### Task10 phase14: TTS repository lifecycle foundation only

`TTSProfileRepository` now reserves ordinary open/CRUD/restore entry, seals new
requests separately from terminal state, and exposes private owner-loop
`_maintenance_close_admission`, `_maintenance_drain(deadline)` and
`_maintenance_resume`. Its existing executor owns all cleanup. Explicit result
completion markers survive the native Future's done callback; cancelled callers
cannot hide running work. Positive connection/ProfileStoreLease retirement precedes
CLOSED/generation advance. Timeout and uncertain cleanup retain the executor and
actual source references. The exact repository is a strong local raw-operation
blocker during drain/resume uncertainty; that record grants no IO or native lease.

Resume uses fresh admission and original source/path/inode checks on the worker.
Public close stays definitive, including its existing non-quarantine failure
contract; maintenance never clears a failure because a later close succeeded.
This phase supplies **no installed TTS source binding or across-pause file scope**.
The actual app constructor at `app.py::TldwCli.__init__`, shared
`_ensure_tts_profile_repository` open task and retained definitive
`_close_tts_profile_repository` task are unchanged. Their full app/headless safe
point and exact current config/profile association remain required Task10 work.
No syntax-owner classification or participant_pending entry is promoted.

Immediate TTS continuation (phase14b within Task10):

- Repository `_worker_open_if_proven_current`, `_worker_initialize_store`,
  `_worker_publish_migrated_store`, reusable tombstones and exact-schema readers,
  plus `profile_migration_{candidate,journal,namespace,publication,recovery}` need
  their finite source-selected native cohort before effects. Preserve existing
  schema4/lineage/lock/publication recovery policy; inspection never opens/migrates
  the ordinary repository. Literal fixed candidate/rollback slots do not grant a
  parent-directory scope, and active-specific journal names must be selected from
  the actual producer rather than guessed from the default journal constant.
- `_worker_backup_to` selects its randomized `mkstemp` output inside the worker;
  destination/source/sidecars/parent and temporary must be admitted as one actual
  operation. At the phase14 base, cleanup lost a failed-close destination. Phase14b
  fixes outer retention below; full source/finite subordinate admission remains
  immediate phase14c work. `_worker_create_recovery_backup`, restore candidate/source
  readers, publication/rebind, residual cleanup and fsync descriptors also require
  exact native-resource and durable-result qualification.
- `profile_reference_storage.write_reference_blob/read_reference_blob` hold BLOBs
  locally and report close failures; lifetime after failed close needs real native
  evidence. `profile_reference_audio`, `profile_reference_materialization` creation,
  validation/sweep/cleanup, and `voice_bundle_service` inspect/stage/commit/export
  own additional file/lock/stream/worker resources. Repository row/reference tests
  do not qualify those resource owners or their service-level compound lifetimes.
- `profile_service._run_owned_repository_call`, consumer/artifact mutation fences,
  availability/dependency/runtime methods; `STTSProfileLibrary` editor drafts,
  sanitized export worker and bundle UI; app/shared open/close/result handling;
  actual synthesis/player/audiobook/model/backend jobs all retain their own pending
  and native obligations. Also audit independently cancelled private lifecycle
  completion Tasks and the underlying open/restore/close/shutdown futures rather
  than treating wrapper cancellation as native retirement.
- The two unchanged `TTS/recovery.py::_Voices.discover` literals at lines111/115
  still fail the profile-owned path census. ADR040 intentionally keeps reusable
  Chatterbox/Higgs voices shared; actual backends retain those legacy shared roots.
  Resolve their canonical runtime/config/default selection and exact census rows
  in the immediate actual voice-manager/backend cohort. No path relocation or
  blanket exception is authorized, and the profile guard is not Green.

Evidence: final focused new lifecycle/shared run57passed, including20 new lifecycle
cases; independent source/native and known-startup observer cases use private
subprocesses and pipes. Ordinary selected open/close/cancellation/namespace cases
116passed;3 spawned concurrent-open cases failed before repository code because
host SemLock allocation raises OSError28. A standalone stdlib Event probe reproduces
that host limitation. Reference/backup integration96passed. Final affected source
census1passed; earlier combined architecture23passed/3failed included one temporary
new-call row from an intermediate resume implementation (now removed and rechecked)
and the two inherited voice literals. Counts overlap and are not additive. No full
suite or live model/voice/network work ran; full commands and limits are in the
identical external and scratch task-10-phase14-report.md. Task31993 remains In
Progress with all ACs unchecked; whole Task10 review belongs to the controller.

### Task10 phase14b: outer backup retention, source graph still unsupported

Under ruling80, `TTS/profile_repository.py::_BackupNativeState` represents only the
outer `_worker_backup_to` resources. The repository registers it before allocation,
retains the exact source connection/configured path/active path/admitted generation,
and immediately records the mkstemp FD/path/inode, destination SQLite handle, parent
pin, file-fsync FD, selected destination and independent ordinary StorageLease.
Close attempt and positive return are separate; failed resources are not retried
or forgotten by public close, cancellation, successful later backups or maintenance.
The existing raw-operation set holds strong pending/failure evidence, not file access
or native authority. The ordinary lease keeps its native admission until positive
outer retirement. No installed TTS source binding or across-pause access was added.

`_BackupNativeState.check_namespace` checks parent/main provenance before cleanup
and publication; outer cleanup does not unlink remaining unproven sidecars. The
normal native backup journal exists until SQLite close and keeps its original
ordinary behavior. Full native journal ownership is still unsupported. Successful
rename consumes the operation's temporary ownership; positively observed publication
and safe receipt metadata survive subsequent directory-fsync/parent-close failure.
The outer scope keeps its exact public error and internal source/cleanup outcomes.

The two new `open` census rows are individually unsupported TTS sources: the parent
pin in `_worker_backup_to` and file-fsync FD in `_BackupNativeState.fsync_file`.
Existing private-SQLite policy rows and every TTS recovery/participant classification
remain unchanged. Public `_fsync_file/_fsync_directory` still serve restore and are
not qualified by the outer backup implementation.

Immediate phase14c, before voice phase15, must cover:

- `_worker_validate_standalone_snapshot` → `profile_schema.validate_profile_candidate`:
  source FD, random mkdtemp directory, snapshot mkstemp/copy FD, upgrade/read SQLite,
  schema/domain validation and BLOB readers; then the additional immutable snapshot
  reader plus integrity/reference scan. Actual pause evidence confirms source-open
  and copy-file effects before the eventual subordinate SQLite refusal.
- `DB/private_sqlite.backup_open_connections_to_private`: delegated source pin,
  destination journal mode/online-copy/native journal and their failure/retirement
  graph. Outer main/parent checks do not establish native sidecar identity.
- Exact loaded app/config/profile/repository registration before source selection;
  finite main/sidecar/lock/migration candidate, rollback and active-specific journal
  admission; `_worker_create_recovery_backup` and restore stage/rebind/residual/fsync
  resources, with original schema/lineage/quarantine policy.
- Real `profile_reference_storage` BLOB closure; `TTSCloneReferenceMaterializer`
  creation/sweep/opaque handle cleanup; `voice_bundle_service` inspection sessions,
  stage/commit/export/invalidation/close; concrete profile-service/editor/export and
  app open/close/result safe points; independently cancelled private legacy lifecycle
  tasks and executor shutdown. Runtime/model/audio sources retain their own duties.

The two inherited shared-voice literals in `TTS/recovery.py`, diagnostic inventory
drift and phase14's three host SemLock failures remain explicit; this phase does
not rerun or relabel those checks. No startup, runtime or Complete promotion.

Phase14b guard follow-up: the private-SQLite inventory has a pre-existing stale
`test_explicit_exclusions_and_absence_of_async_owner_are_documented` assertion for
`JSONStorage._create_backup` in `Widgets/Tamagotchi/tamagotchi_storage.py`. It expects
`shutil.copy2`, but unchanged BASE/current code uses `raw._file`, bounded stream
copy, `raw._replace` and `raw._remove_temporary`. Exact BASE AST-clause reproduction
and full source/test byte identity are in the phase14b report. Controller owns the
mandatory Task10 guard correction after native continuation; this is separate from
Task26 diagnostic drift and is not a waiver. Source census itself passed.


### Task10 phase14c: ordinary candidate validation, still unsupported as installed source

`TTS/profile_schema.py::validate_profile_candidate(path, *, check_deadline=None)`
now owns a concrete synchronous `_CandidateValidationJob` before first callback or
file access. It grants no configured repository/source authority. Its private
`_validate_profile_candidate(..., job=...)` retains existing snapshot-copy, schema4
and disposable-only historical migration behavior. Actual source/snapshot FD,
private directory/main identities, parent pins, returned upgrade/read SQLite handles
and independent ordinary leases survive uncertain cleanup in `storage._raw_operations`.
Attempted and positive close are distinct; independent closes preserve control-flow
priority, and foreign/unlinked/substituted or unproven-sidecar namespace uncertainty
cannot be erased by unrelated success. Native missing-parent-pin detection preserves
ordinary platform behavior without a failure-triggered downgrade or qualification.
The source census names the exact new pin and unlink symbols, with unchanged
`unsupported / tts` classification.

Immediate continuation: authenticate app.py's actual configured TTSProfileRepository,
preselect finite original source/destination/snapshot/migration/restore names before
worker dispatch, and own the actual `DB/private_sqlite.py::_pin_sqlite_source` parent
and file FDs, backup journals, `profile_repository._worker_validate_standalone_snapshot`
additional immutable reader, historical migration subresources and real reference
BLOB lifetimes. The standalone job cannot stand in for that compound source. Restore
publication/rebinding, ProfileStoreLease residual handles, materializer/bundle sessions,
service/UI dirty state and runtime/audio/model/process work remain Task10. ADR040 shared
voice path census correction follows this immediate TTS graph, not a default-path move.
The inherited exact JSONStorage AST guard follow-up, three host SemLock failures and
combined-order/diagnostic debts retain their earlier separate assignments.

### Task10 phase14d: configured TTS source checks and delegated SQLite pins

The actual TldwCli constructor explicitly binds its original TTSProfileRepository
receiver to the loaded config module/functions, effective/cache config source,
selected profile and TTS database path. Source checks precede ordinary entry,
worker execution and result publication; first-open worker generation is checked
before effects. Harmless setting changes and same-source cache replacement remain
valid. Copied/subclass receivers and reloaded/retargeted sources receive no binding
permission; standalone/custom constructors remain ordinary. The repository constructor
still performs no config import or IO. Existing shared named app open/close tasks,
service retained calls and definitive close of original resources remain intact.
This is no new across-pause privilege or whole-app participant registration.

Shared DB/private_sqlite `_SQLiteSourcePinJob` retains ordinary exact source/parent
leases or the existing validated capture lease/scope resource association. Explicit
preflight propagation owns returned main/sidecar/parent descriptors before metadata;
final `_PinnedSQLiteSource` owns returned source file/parent descriptors. The trusted
directory verifier's explicit private open/close observers delegate original native
calls, including existing raw/visual attribution. Admission precedes allocation
accounting; returned descriptors are attached before metadata, and unreturned failures
remain separate from later successful opens. Independent retirement attempts run once;
uncertain allocation/close, original body errors and cleanup errors retain actual
resources and exclusion. Capture uncertainty retains its native maintenance session.
No capture policy, scope paths, registry or directory-descendant grant was expanded.

Ruling84 permits ordinary unpublished backup validation to return its truthful
refusal when pause prevents a new allocation, after actual worker/publication,
known temporary bytes and native resources retire. Private diagnostic evidence covers
aggregate local drain and later native entry; it does not release production startup,
discard drafts or claim a successful backup receipt. Complete/replacement remain
unavailable and Task31993 remains In Progress with every AC unchecked.

Mandatory immediate continuation still includes outer immutable reader namespace
association, destination journal provenance, current/migration wrappers and native
ProfileStoreLease, BLOBs, recovery backup, restore/publication/rebind, materializer,
bundle/service/dirty editor/runtime owners, then shared voice roots and remaining
app/headless/runtime aggregation. See the phase14d dual report for exact native
limits, source evidence and inherited SemLock/voice census/pet AST/combined-order/
diagnostic debt. This entry supersedes only the now-implemented source/pin portions
of earlier phase handoffs, never their remaining routes.


Ruling87 distinguishes confirmed native rejection from unknown provider outcome
with one explicit per-call outcome record at the original native-open boundary.
Only rejection by the exact original native primitive proves no descriptor; its
returned FD is recorded before raw registration or outer wrapper work. Known
wrapper failures can therefore retire their actual FDs. Substituted/delegated
unreturned failures retain uncertainty, regardless of exception type or later
successful opens. Existing raw/visual attribution and trust/scope rules remain.
This corrects the demonstrated trusted-alias regression: default TTS profile paths
can remain lexical, while custom TTS database selections resolve canonically.
Ordinary alias copies and actual adapter capture must both positively retire;
no normal supported alias is waived as an unqualified-platform limitation.


Ruling88 carries the same per-call outcome into the existing SQLite artifact-open
primitive, preserving its non-raw/visual attribution and unrelated callers. Concrete
preflight and final pins attach known returned FDs before wrapper errors, recognize
only original-native rejection as absence, and retain unknown substituted outcomes.
This repairs a confirmed disappearing-optional-sidecar retirement regression against
BASE. Real ordinary/capture copies cover all three suffixes; actual replacement retry
retires, while a successful retry after an unknown allocation cannot erase that
older uncertainty. Privacy, no-follow, generation/absence retries and source identity
checks remain unchanged. No new capture/source/descendant permission is introduced.

### Task10 phase14e continuation — outer snapshot and delegated journal

Both ordinary backup and recovery backup retain `_BackupNativeState` through actual
snapshot validation, parent/temp/fsync descriptors, native destination and uncertain
allocation/close outcomes. Recovery now explicitly opens its registered
`tts.profile_recovery` destination and calls the existing checked open-connections
helper. Its exact source-close failure/exclusive-lease handoff remains intact and is
not full current-wrapper/ProfileStoreLease retirement qualification.

The existing ordinary progress/completion path observes the exact native journal;
observed replacement prevents native close, and only SQLite retires its own normal
journal. Independent private native maintainers distinguish completed cleanup from
retained uncertain resources. Test-child startup retirement is explicit diagnostic
fixture cleanup, never production startup release. Exact source census updates add
only recovery's actual parent open and explicit destination/checked-helper sites;
all TTS classifications remain unsupported pending the complete source/runtime graph.

Current wrapper/dual SQLite/FD/sidecars, primary/residual profile locks, actual BLOB
read/write, migration candidates/rollback/tombstones/journals, restore rebind/quarantine,
materializers/bundle sessions, dirty UI/service/consumer/artifact and all voice/backend/
model/audio/process/app/headless/startup routes remain mandatory Task10. Inherited
three SemLock, two shared-voice path census, exact pet AST and combined-order debts
are unchanged; diagnostic drift remains Task26. No all-owner capability promotion.

The existing candidate job is associated with its outer record before native effects.
Standalone candidate cleanup also stops releasing later actual leases after its first
uncertain lease-close result; independent native admission verifies the retained hold.
Only an explicit per-call refusal at the original SQLite admission boundary clears
that current pending target attempt. Same-code errors after native allocation remain
unknown; standalone kwargs, factory contracts and capture/source scopes are unchanged.


### Task10 phase14g — live migration, publication, restore and rebind native outcomes

The concrete `_MigrationNativeState` retains ordinary leases in the existing live
lease set and an explicit repository operation set. Source-version readers,
initialized connections and actual restore candidates retain their own constructor
and first-close outcomes; an existing public compatibility retry cannot clear a
failed first native outcome. Ordinary refusal and positive cleanup retire. No new
ambient context, global SQLite/BLOB registry, native mutex or callback/capture grant
is introduced. Original literal registered SQLite call sites remain visible.

Publication, recovery, namespace and postinit helpers explicitly associate their
original native parent/file/descriptor-reader/journal/fsync operations with that
record. A supplied operation must match its exact source and known parent; recovery
may associate only exact validated journal-row leaves after original authority
checks. The returned-FD namespace API keeps its existing opaque destination owner
when no operation is supplied. Scalar standalone publication/recovery/postinit
operations own only actual supplied sources. Existing opaque authority checks still
decide whether any path operation is allowed.

Verified-descriptor SQLite calls bypass ordinary path-backed SQLite admission.
Their private per-call `_SQLiteDescriptorOutcome` now observes actual duplication,
native SQLite return, unreturned allocations, and independent duplicate/reader
retirement before later wrapper failures. The DB seam grants no TTS/path authority.
Historical real reference BLOBs remain owned by their native SQLite parent; actual
source-thread observations prove positive parent retirement without another registry.

The shared-open journal precheck has its own bounded source record. Exact-current
revalidation observes its existing parent traversal on the already-held exact owner,
without trying to obtain fresh admission during maintenance cleanup. Original
traversal and final-close primitives remain distinct. Positive closes permit fresh
OS descriptor-number reuse; uncertain closes never replay. Failed construction can
retire a positively query-only DELETE-mode current connection only with original
parent/main/security checks and both WAL/SHM absent. Full-current use still requires
the retained WAL/SHM pair; foreign namespace or unknown mode stays excluded.

Genuine v3 restore uses the existing version-aware reference validator; current-v4
validation remains unchanged. Real preflight/publishing pause cases verify exact
source and reference bytes, original result/journal phase, and repeated recovery
convergence after ordinary resume. A durable journal does not itself qualify capture,
and no publication continuation authority is granted. Exact census movement is the
11 old raw-open rows above into `_native_open`'s two actual branches; owner disposition
remains unsupported pending all remaining TTS/application producers.

ADR126 rulings100–108 govern this bounded phase. Materializer14h, bundles14i,
service/dirty UI/runtime14j, shared voices15, actual backend/model/audio/native-process
and full app/headless/startup aggregation remain Task10 work. Inherited SemLock,
shared-voice census, pet AST and combined-order debts remain; diagnostic drift is
Task26. Tests use private temporary sources and native observers on this host only.

### Task10 phase14h: exact clone-reference materializer and response assets

Rulings109–112 add concrete native outcomes to the actual materializer, retained
on ordinary StorageLease.native_owner and the original opaque handle/record.
New worker reservations precede queued to_thread work. Creation and orphan sweep
record finite root/owner/lock/asset selections; unknown parents and foreign roots
refuse before native admission. Traversal descriptors observed through optional
secure_private_directory callbacks remain resource outcomes only, never direct
child authority. Each allocation still obtains ordinary storage admission.

Actual native return/rejection, unique unresolved allocation attempts, first close
failures and namespace residue remain distinct. Positive independent cleanup does
not erase an ambiguous close; later terminal calls never retry that numeric FD.
Known unpublished records are cleaned by their original protocol. Failed-create
cleanup verifies original native identities and preserves substitutions, renamed
original bytes and siblings; residue stays a drain blocker, not disposable capture.
Final cleanup retains original control signals and existing body precedence.

The default build_default_tts_service receiver binds to its original configured
module/selector/profile/root; drift refuses new work and resume while original
handle cleanup remains possible. Its pre-binding get_user_data_dir still performs
its existing config/base/profile directory work. Binding itself performs no clone
sweep, adapter/model work, or archive capture. Explicit custom constructors and prebinding custom factory selectors remain
ordinary and unqualified by pathname coincidence.

The materializer exposes _maintenance_close_admission, async _maintenance_drain
(deadline), and async _maintenance_resume on its owner loop; terminal seal/close
remain distinct. Drain preserves live response WAVs and refuses while native,
queued/result, cleanup or namespace work remains. Actual generation's existing
response.add_cleanup(materialization.aclose) retains ownership until native
cleanup, including failure after response._closed and terminal service follow-up.

Source/native tests cover real files/locks and independent maintenance processes,
including declared-profile enrollment. They do not qualify all response producers
or consumers. TTSAudioResponse callback ordering/closed-field, managed response
registry removal, synthesis/native process/adapter/model/audio shutdown, service
maintenance composition, recovered runtime-root inventory/bytes, clone bundles,
profile-service dirty UI and shared voices remain immediate Task10 work. Table
rows remain unsupported at the whole-owner level. No runtime startup release,
Complete backup, replacement, or Task31993 completion follows from this cohort.

### Task10 phase14i: bundle native operations and exact lazy app source

`TTSVoiceBundlePortabilityService` now reserves concrete native operations before
queued workers and retains ordinary admission across actual prepare, inspection,
fingerprint and publication resources. Direct descriptor calls pass their exact
selected path to admission; external selected source/destination/parent paths do
not inherit runtime-root scope or enlarge enrollment. Original finite creator
edges and exact `_Operation` membership constrain selected leaves. Independent
native admission and root flock observers cover actual encoded private bundles.

The source census's former direct `open` rows now converge on `_bundle_open`;
its original `_open_bundle_descriptor` primitive records native returns before
outer failure. `_open_bundle_stream` and existing private-directory observers
preserve stream/traversal attribution. Buffered members do not own their native
FDs; unknown/failed streams retain the FD and never compete with a second closer.
First uncertain native/lease closes are not replayed. Unknown native allocations,
failed exact namespace cleanup and substituted/renamed originals retain blocking
ownership despite sanitized public outcomes. Proven original partial-create
cleanup can retire; foreign/restart residue is not guessed disposable.

Owner-loop `_maintenance_close_admission`, async `_maintenance_drain(deadline)`
and `_maintenance_resume` preserve pending in-memory review sessions and await
actual caller/worker/repository result lifetimes. They are distinct from terminal
`seal/close/wait_closed`. TTL/count, stale reviews, single consumption, explicit
inactive consent, original artifact-before-profile fence order and source
fingerprinting remain. Export acknowledgement is independent of maintenance
eligibility: the original publisher's exact durable inode evidence preserves
success through late cleanup failure; fingerprint success does not. Existing
no-replace, post-publication convergence and intentional pre-publication `0600`
temporary-file preservation are unchanged.

Only the actual lazy app factory binds `_ConfiguredBundleSource` to the original
configured repository, app/profile/dependency/coordinator objects, config selectors
and exact root. Original fence class/function identity is checked without custom
accessor execution. Explicit custom routes stay ordinary/unqualified. Tests include
real configured SQLite reference import/export, byte-identical encoded output,
source/remap/callback drift and cleanup after drift. Other native evidence uses
controlled repository values with actual bundle/filesystem/native operations;
it does not qualify the downstream model service or whole app startup.

All TTS installed census rows remain `unsupported / tts`. The surviving
`tts_clone_materializations` and `tts_voice_bundle_portability` roots still need
source-backed installed inventory classification, including unknown/foreign
residue. Profile service and dirty Profile Library UI, consumer/artifact/native
runtime, shared voice paths, existing combined app/source-order and other Task10
guard debts remain pending. No source-backed disposition or complete capability
is inferred from directory names, clean native workers or this cohort's tests.

| tldw_chatbook/TTS/sample_audio_validation.py | _open_sample_container | open | 1 | unsupported | tts |
| tldw_chatbook/UI/stts_profile_library.py | STTSProfileLibrary._write_profile_export | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/stts_profile_library.py | STTSProfileLibrary._write_profile_export | write | 1 | unsupported | miscellaneous |

### Task10 phase14j source/native and caller boundary

Original profile-service calls and synchronous evidence now retain complete caller accounting, and optional original sample validation holds actual selected FD/codec cleanup outcomes independently of content validity. The original app-created service has a pure configured-source relationship; standalone/custom callers retain ordinary semantics without inherited runtime authority. Library editor/conflict/submitted-on-pause drafts, actions, page publication, bundle review/invalidation and sanitized export have reversible local admission/readiness. Export remains caller-owned output with original overwrite semantics, never a baseline asset. Native uncertainty retains original outcomes and ordinary selected path/parent leases. Rows above remain unsupported at cohort level: syntactic call counts do not establish installed app-wide participation.

Same-process ordinary first authority initialization uses existing pending acquisition records and coordinator condition to avoid observing its own marker before registration; incomplete/foreign records still refuse. Exact archived shared-source BASE reproduces the prior interval. This does not qualify cross-process first startup, existing runtime startup release, synthesis/model/provider/player jobs, complete Speech caller UI publication, artifact backend ownership, shared voices/default root selection or any remaining Task10 cohort. Accepted combined-order six-factory failures and all recorded later-cohort debts remain mandatory. Task31993 stays In Progress with every AC unchecked.

Ruling119 refinement: the sanitized writer now pins the selected parent and target FD, verifies existing identity/type before truncation, creates absent output exclusively beneath that parent, and uses nonblocking opens to refuse a substituted FIFO without a reader. The stream uses closefd=False; its buffer/target FD and independent parent FD have explicit separate native retirement, with first/unknown outcomes retained. Late path replacement does not redirect writes to a foreign inode; no caller output is deleted. This adds concrete native effects beyond the syntactic alias census and does not promote the cohort classification.

Ruling120 keeps an upfront original-primitive capability check and the original ordinary pathname stream export on unavailable platforms. This route is explicitly unqualified for Library maintenance, retains complete stream/result lifetimes, and never follows a failed/unknown pinned allocation. No global supports_dir_fd metadata is mutated. A mounted private capability simulation verifies ordinary public success/bytes/close separately from refused Library qualification; it is not a real cross-platform run.

## Recovered media durable owner — TASK-31994 / ADR-126

Catalog v1, exact source profile/message/slug/type references, operation journal and validated versioned tombstones live under the profile recovered_media root. Ready payloads are baseline dependencies; intentional deletion is validated inside the catalog and does not emit caller-supplied deletion_validated flags. Unknown files and pending operations block complete capture. Temporary VideoStore TTL/session cleanup never traverses this root. Real consumers: VideoStore.resolve_state and Console video card/play/save; the enhanced image implementation is Widgets/Chat_Widgets/chat_message_enhanced.py behind the compatibility alias. Single-image ambiguity renders missing. Task10 composition remains incomplete.

| sqlite:recovered.media | tldw_chatbook/Backup_Recovery/recovered_media | _PRIVATE_FILE | recovered.media/durable |
| sqlite:recovery.recovered_media | tldw_chatbook/Backup_Recovery/recovered_media | _PRIVATE_AND_READ_ONLY | recovered.media/capture-validation |

| tldw_chatbook/Backup_Recovery/recovered_media.py | RecoveredMedia.__init__ | secure_private_directory | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | RecoveredMedia._connection | connect_private_sqlite | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | RecoveredMedia._finish_delete | os.unlink | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | RecoveredMedia.retain | atomic_private_write_bytes | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | _RecoveredAdapter.capture | copy_private_sqlite | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | _RecoveredAdapter.discover | connect_private_sqlite | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | _RecoveredAdapter.validate | connect_private_sqlite | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | _RecoveredAdapter.validate_dependencies | connect_private_sqlite | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | _RecoveredAdapter.validate_restore_dependencies | connect_private_sqlite | 1 | qualified | recovered.media |
| tldw_chatbook/Backup_Recovery/recovered_media.py | _read | open_private_binary | 1 | qualified | recovered.media |

## Original Task19 bounded RAG discovery and indexing SQLite

The inert factory is Backup_Recovery.rag_inventory.recovery_adapters. RAG_Search
runtime package startup remains unchanged. Nonempty definition files and Chroma
stores remain unsupported; empty/absent selectors do not construct runtime owners.
The indexing SQLite v0 catalog and native current-thread lifetime are qualified
independently; this does not establish restored retrieval readiness or Chroma
capture. Task19 source-generation/reconciliation gates remain outstanding.

| sqlite:recovery.rag_indexing | tldw_chatbook/Backup_Recovery/rag_indexing | _PRIVATE_AND_READ_ONLY | exact installed indexing SQLite schema/capture |
| tldw_chatbook/Backup_Recovery/rag_indexing.py | _Indexing.validate | connect_private_sqlite | 1 | qualified | db.rag_indexing |
| tldw_chatbook/Backup_Recovery/rag_indexing.py | _Indexing.capture | copy_private_sqlite | 1 | qualified | db.rag_indexing |

## Archive capture, validation and publication boundaries

These calls operate on reviewed input archives, private disposable staging, or the
explicit output destination. Their census classification does not qualify any
additional live storage owner or claim complete backup/restore readiness. Preview
copies preserve live main/WAL bytes; credential reconstruction operates only on
private staged copies. Model lease rows retain their existing cohort status.

| sqlite:recovery.credentials | tldw_chatbook/Backup_Recovery/credentials | _PRIVATE_FILE | disposable credential reconstruction |
| sqlite:recovery.validation | tldw_chatbook/DB/private_sqlite | _PRIVATE_AND_READ_ONLY | disposable imported candidate validation |
| sqlite:recovery.validation_schema | tldw_chatbook/Backup_Recovery/sqlite_validation | _MEMORY | installed schema reference |

| tldw_chatbook/Backup_Recovery/archive_reader.py | _inspect | ZipFile | 1 | generic_boundary | reviewed archive input |
| tldw_chatbook/Backup_Recovery/archive_reader.py | _regular | open | 1 | generic_boundary | reviewed archive input |
| tldw_chatbook/Backup_Recovery/archive_reader.py | acquire | create_private_file | 1 | disposable | private archive staging |
| tldw_chatbook/Backup_Recovery/archive_reader.py | acquire | write | 1 | disposable | private archive staging |
| tldw_chatbook/Backup_Recovery/archive_reader.py | verify_sealed | ZipFile | 1 | generic_boundary | private archive validation |
| tldw_chatbook/Backup_Recovery/archive_writer.py | _package | ZipFile | 1 | disposable | private output staging |
| tldw_chatbook/Backup_Recovery/archive_writer.py | _package | create_private_file | 1 | disposable | private output staging |
| tldw_chatbook/Backup_Recovery/archive_writer.py | _package | open | 1 | generic_boundary | captured payload input |
| tldw_chatbook/Backup_Recovery/archive_writer.py | _package | write | 1 | disposable | private output staging |
| tldw_chatbook/Backup_Recovery/capture.py | _capture_under_maintenance | create_private_file | 1 | disposable | private capture manifest |
| tldw_chatbook/Backup_Recovery/capture.py | _capture_under_maintenance | write | 1 | disposable | private capture manifest |
| tldw_chatbook/Backup_Recovery/credentials.py | _read | open_private_binary | 1 | generic_boundary | staged credential input |
| tldw_chatbook/Backup_Recovery/credentials.py | _rewrite_database | connect_private_sqlite | 1 | disposable | staged credential reconstruction |
| tldw_chatbook/Backup_Recovery/credentials.py | _rewrite_database | os.replace | 1 | disposable | staged credential reconstruction |
| tldw_chatbook/Backup_Recovery/credentials.py | _write | atomic_private_write_text | 1 | disposable | staged credential rewrite |
| tldw_chatbook/Backup_Recovery/sqlite_validation.py | _reference | connect_private_sqlite | 1 | memory | installed schema reference |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _CaptureScope.sqlite_target | mkdir | 2 | disposable | private native capture SQLite copy |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _CaptureScope.sqlite_target | open | 2 | generic_boundary | native-domain-recovery |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _CaptureScope.sqlite_target | write | 1 | disposable | private native capture SQLite copy |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _PreviewScope.sqlite_target | create_private_file | 1 | disposable | private preview SQLite copy |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _PreviewScope.sqlite_target | open | 1 | generic_boundary | identity-bound live SQLite copy input |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _PreviewScope.sqlite_target | write | 1 | disposable | private preview SQLite copy |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _write_staged_credential_file | open | 1 | disposable | private staged credential rewrite |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _write_staged_credential_file | os.replace | 1 | disposable | private staged credential rewrite |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _write_staged_credential_file | os.unlink | 1 | disposable | private staged credential cleanup |
| tldw_chatbook/Backup_Recovery/storage_admission.py | _write_staged_credential_file | write | 1 | disposable | private staged credential rewrite |
