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
options/budgets are not accepted by this API: the later service must include them
when composing its operation scope digest, never substitute this inventory digest
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
| miscellaneous | Exact remaining producer module/symbol below; caller-owned path resolution is unresolved rather than guessed | Durable versus external/process classification unresolved; dependency coverage unknown | unsupported / unsupported / unsupported / no activation |
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
The refreshed census contains 978 rows. Its qualified symbol is the locator-consumer
evidence; where canonical root selection is not yet extracted, the cohort table explicitly records that resolver
blocker. New calls in an existing symbol also change the expected count and fail.

| Module | Producer symbol | Call | Count | Classification | Resolver / adapter cohort |
| --- | --- | --- | --- | --- | --- |
| tldw_chatbook/Agents/project_instruction_resolver.py | _read_candidate | open | 1 | unsupported | agents |
| tldw_chatbook/Agents/run_log.py | RunLogWriter._write_bytes | open | 1 | unsupported | agents |
| tldw_chatbook/Agents/run_log.py | RunLogWriter._write_bytes | write | 1 | unsupported | agents |
| tldw_chatbook/Agents/run_log.py | RunLogWriter.bind | mkdir | 2 | unsupported | agents |
| tldw_chatbook/Agents/run_log.py | RunLogWriter.bind | write_text | 1 | unsupported | agents |
| tldw_chatbook/Audio/recording_service.py | AudioRecordingService._pyaudio_recording_loop | open | 1 | process_artifact | process |
| tldw_chatbook/Audio/recording_service.py | AudioRecordingService._save_audio_file | open | 1 | process_artifact | process |
| tldw_chatbook/Audio_Services_Interop/local_audio_services_service.py | LocalAudioServicesService._persist_history | mkdir | 1 | unsupported | files |
| tldw_chatbook/Audio_Services_Interop/local_audio_services_service.py | LocalAudioServicesService._persist_history | write_text | 1 | unsupported | files |
| tldw_chatbook/Backup_Recovery/crypto.py | _open_regular | open | 1 | generic_boundary | generic |
| tldw_chatbook/Backup_Recovery/crypto.py | _write_all | write | 1 | generic_boundary | generic |
| tldw_chatbook/Backup_Recovery/inventory.py | discover | open | 1 | generic_boundary | generic |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | export_character_card_to_png | makedirs | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | export_character_card_to_png | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | extract_json_from_image_file | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | import_and_save_character_from_file_with_outcome | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | inspect_character_card_tts_attachment | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | load_character_and_image | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | load_character_card_from_file | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/Character_Chat_Lib.py | load_chat_history_from_file_and_save_to_db | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py | export_dictionary_to_file | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py | export_dictionary_to_file | write | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py | get_chat_dicts_folder | mkdir | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py | import_dictionary_from_file | copy2 | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py | import_dictionary_from_file | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py | parse_user_dict_markdown_file | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | _candidate_pairs | ZipFile | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | _detect_ext | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | _resolve_vpack_expression_set | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | _valid_image | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | build_expression_set_zip | ZipFile | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/expression_set_io.py | resolve_local_expression_set | ZipFile | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/local_character_persona_service.py | LocalCharacterPersonaService._persist_personas | mkdir | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/local_character_persona_service.py | LocalCharacterPersonaService._persist_personas | write_text | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/local_chat_dictionary_service.py | LocalChatDictionaryService._persist_history | mkdir | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/local_chat_dictionary_service.py | LocalChatDictionaryService._persist_history | write_text | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _inspect_image_bytes | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _open_publication_chain | open | 2 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _read_builtin_asset | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _read_private_publication_file | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _read_private_publication_path | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _read_samira_resource | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _read_user_asset_fallback | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _read_user_asset_secure | open | 3 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _sync_publication_directory | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _write_private_publication_file | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _write_private_publication_file | write | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _write_private_publication_path | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | _write_private_publication_path | write | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | cleanup_visual_identity_publication_candidate | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | cleanup_visual_identity_publication_candidate | secure_private_directory | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | publish_visual_identity_candidate | open | 1 | unsupported | files |
| tldw_chatbook/Character_Chat/visual_identity.py | publish_visual_identity_candidate | secure_private_directory | 2 | unsupported | files |
| tldw_chatbook/Chat/attachment_core.py | process_attachment_bytes | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/chat_conversation_service.py | ChatConversationService._save_rag_context_store | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/chat_conversation_service.py | ChatConversationService._save_rag_context_store | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/citation_legacy_migration.py | CitationLegacyMigrationService._raw_sidecar | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/citation_trace_identity.py | KeyringCitationFingerprintKeyProvider.provision_key | set_password | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._persist_existing_message | to_json | 3 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._persist_metadata_only | to_json | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._persist_new_message | to_json | 3 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._persist_usage_only | to_json | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_chat_store.py | ConsoleChatStore._snapshot_roleplay_message_projection_write | to_json | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_context_repository.py | ConsoleContextRepository.finish_auxiliary_attempt | to_json | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_generate_video.py | _stage_pending_video | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/console_image_view.py | ConsoleImageRenderCache.prepare | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/prompt_history.py | PromptHistory._append_impl.write_history | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/prompt_history.py | PromptHistory._append_impl.write_history | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chat/prompt_history.py | PromptHistory._append_impl.write_history | write | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chat/prompt_history.py | PromptHistory.load.read_history | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat/prompt_template_manager.py | load_template | open | 1 | unsupported | files |
| tldw_chatbook/Chat/trajectory_export.py | write_trajectory_export | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chat_Grammars_Interop/local_chat_grammars_service.py | LocalChatGrammarsService._persist | mkdir | 1 | unsupported | files |
| tldw_chatbook/Chat_Grammars_Interop/local_chat_grammars_service.py | LocalChatGrammarsService._persist | write_text | 1 | unsupported | files |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator.__init__ | secure_private_directory | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._add_character_dependency | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._add_character_dependency | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._add_character_dependency | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_characters | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_characters | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_characters | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_conversations | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_conversations | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_conversations | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_kept_briefings | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_kept_briefings | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_kept_briefings | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_media | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_media | mkdir | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_media | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_media | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_notes | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_notes | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_notes | write | 8 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_prompts | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_prompts | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._collect_prompts | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_readme | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_readme | write | 33 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_zip_archive | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_zip_archive | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_zip_archive | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._export_message_attachments | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._export_message_attachments | write_bytes | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_conversation_citation_report | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_conversation_citation_report | write | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_kept_briefing_report | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_kept_briefing_report | write | 16 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._write_message_citation_report_section | write | 15 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator.create_chatbook | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator.create_chatbook | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter.__init__ | secure_private_directory | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._extract_private_archive | ZipFile | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._extract_private_archive | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._extract_private_archive | secure_private_directory | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._extract_private_archive | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_characters | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_conversations | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_kept_briefings | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_media | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_notes | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter._import_prompts | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter.import_chatbook | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/chatbook_importer.py | ChatbookImporter.preview_chatbook | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/database_paths.py | secure_chatbook_directory | secure_private_directory | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chatbooks/local_chatbook_service.py | LocalChatbookService._load_registry | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chunking/Chunk_Lib.py | load_document | open | 1 | unsupported | files |
| tldw_chatbook/Chunking/chunking_templates.py | ChunkingTemplateManager._get_user_templates_dir | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chunking/chunking_templates.py | ChunkingTemplateManager._load_template_from_file | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chunking/chunking_templates.py | ChunkingTemplateManager.save_template | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chunking/chunking_templates.py | ChunkingTemplateManager.save_template | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Chunking/engine/chunker.py | Chunker.chunk_file_stream | open | 1 | unsupported | files |
| tldw_chatbook/Coding/code_mapper.py | SimpleIO.read_text | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Config_Files/create_custom_template.py | create_custom_template | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Config_Files/create_custom_template.py | create_custom_template | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Config_Files/create_custom_template.py | create_custom_template | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/DB/AgentRuns_DB.py | AgentRunsDB | inherits:BaseDB | 1 | unsupported | sqlite |
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
| tldw_chatbook/DB/Subscriptions_DB.py | SubscriptionsDB | inherits:BaseDB | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Subscriptions_DB.py | SubscriptionsDB._get_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Subscriptions_DB.py | ensure_site_configs_schema | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/DB/Workspace_DB.py | WorkspaceDB | inherits:BaseDB | 1 | unsupported | sqlite |
| tldw_chatbook/DB/base_db.py | BaseDB._get_connection | connect_private_sqlite | 1 | unsupported | sqlite |
| tldw_chatbook/Evals/config_loader.py | EvalConfigLoader._load_config | open | 1 | generic_boundary | evals |
| tldw_chatbook/Evals/config_loader.py | EvalConfigLoader.save | dump | 1 | generic_boundary | evals |
| tldw_chatbook/Evals/config_loader.py | EvalConfigLoader.save | mkdir | 1 | generic_boundary | evals |
| tldw_chatbook/Evals/config_loader.py | EvalConfigLoader.save | open | 1 | generic_boundary | evals |
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
| tldw_chatbook/Event_Handlers/note_ingest_events.py | handle_ingest_notes_import_now_button_pressed.import_worker_notes | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/note_ingest_events.py | handle_ingest_notes_import_now_button_pressed.import_worker_notes | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/note_ingest_events.py | handle_ingest_notes_import_now_button_pressed.import_worker_notes | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Event_Handlers/notes_events.py | load_note_templates | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Feedback_Interop/local_feedback_service.py | LocalFeedbackService._persist | mkdir | 1 | unsupported | files |
| tldw_chatbook/Feedback_Interop/local_feedback_service.py | LocalFeedbackService._persist | write_text | 1 | unsupported | files |
| tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py | ComfyUIImageAdapter._download_output | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py | _load_packaged_workflow | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Image_Generation/adapters/image_format_utils.py | maybe_convert_format | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Image_Generation/request_validation.py | _validate_reference_image_content | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_db.py | open_connection | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.__init__ | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService._search_cards_raw | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService._search_result_for_card | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.copy_card_with_checklists | connect | 2 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.export_board | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_board | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_card | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_card_link_counts | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_checklist | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_checklist_item | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_comment | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_label | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_list | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.get_storage_status | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_board_activities | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_boards | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_card_activities | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_card_labels | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_card_links | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_cards | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_cards_by_linked_content | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_checklist_items | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_checklists | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_comments | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_labels | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.list_lists | connect | 1 | unsupported | miscellaneous |
| tldw_chatbook/Kanban_Interop/local_kanban_service.py | LocalKanbanService.transaction | connect | 1 | unsupported | miscellaneous |
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
| tldw_chatbook/MCP/client.py | _StdioJSONRPCConnection._send_message | write | 1 | unsupported | mcp |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog._migrate_generation | atomic_private_write_bytes | 1 | unsupported | mcp |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog._read_bytes | open_private_binary | 1 | unsupported | mcp |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog._secure_parent | secure_private_directory | 1 | unsupported | mcp |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog.append | atomic_private_write_bytes | 2 | unsupported | mcp |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog.append | open_private_text_append | 1 | unsupported | mcp |
| tldw_chatbook/MCP/execution_log.py | MCPExecutionLog.append | write | 1 | unsupported | mcp |
| tldw_chatbook/MCP/local_store.py | LocalMCPStore._read_payload | open | 1 | unsupported | mcp |
| tldw_chatbook/MCP/local_store.py | LocalMCPStore.save | dump | 1 | unsupported | mcp |
| tldw_chatbook/MCP/local_store.py | LocalMCPStore.save | mkdir | 1 | unsupported | mcp |
| tldw_chatbook/MCP/local_store.py | LocalMCPStore.save | open | 1 | unsupported | mcp |
| tldw_chatbook/MCP/permission_store.py | MCPPermissionStore.save | dump | 1 | unsupported | mcp |
| tldw_chatbook/MCP/permission_store.py | MCPPermissionStore.save | mkdir | 1 | unsupported | mcp |
| tldw_chatbook/MCP/permission_store.py | MCPPermissionStore.save | open | 1 | unsupported | mcp |
| tldw_chatbook/MCP/server_target_store.py | ConfiguredServerTargetStore._read_payload | open | 1 | unsupported | mcp |
| tldw_chatbook/MCP/server_target_store.py | ConfiguredServerTargetStore.save_targets | dump | 1 | unsupported | mcp |
| tldw_chatbook/MCP/server_target_store.py | ConfiguredServerTargetStore.save_targets | mkdir | 1 | unsupported | mcp |
| tldw_chatbook/MCP/server_target_store.py | ConfiguredServerTargetStore.save_targets | open | 1 | unsupported | mcp |
| tldw_chatbook/MCP/unified_context_store.py | UnifiedMCPContextStore._read_payload | open | 1 | unsupported | mcp |
| tldw_chatbook/MCP/unified_context_store.py | UnifiedMCPContextStore.save | dump | 1 | unsupported | mcp |
| tldw_chatbook/MCP/unified_context_store.py | UnifiedMCPContextStore.save | mkdir | 1 | unsupported | mcp |
| tldw_chatbook/MCP/unified_context_store.py | UnifiedMCPContextStore.save | open | 1 | unsupported | mcp |
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
| tldw_chatbook/Model_Artifacts/leases.py | ArtifactOperationLease._acquire_until | mkdir | 1 | unsupported | models |
| tldw_chatbook/Model_Artifacts/leases.py | ArtifactOperationLease._acquire_until | open | 1 | unsupported | models |
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
| tldw_chatbook/Notes/file_notes_git_network.py | _LayoutBuilder.directory | mkdir | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_network.py | _LayoutBuilder.file | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_network.py | _LayoutBuilder.file | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_network.py | _capture_ssh_trust_source | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_network.py | _file_digest | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_service.py | AsyncGitProcessRunner._write_process_stdin | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_service.py | FileNotesGitService._commit_local_state_is_supported | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory._capture_entry | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory.capture_index | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory.create_directory | mkdir | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory.create_file | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_git_service.py | _PrivatePushProofDirectory.create_file | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_replica.py | FileNotesReplica.__init__ | connect_private_sqlite | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_replica.py | FileNotesReplica.__init__ | mkdir | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.create_file | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.create_file | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.export_exact_file | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.export_exact_file | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.restore_file | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.restore_file | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_copy | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_copy | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_file | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | _read_regular_file | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/git_process_containment.py | _WindowsAsyncChildProcess.communicate.write_input | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/note_import_discovery.py | _inspect_selected_path | open | 2 | unsupported | notes |
| tldw_chatbook/Notes/note_import_discovery.py | _open_verified_directory | open | 2 | unsupported | notes |
| tldw_chatbook/Notes/note_import_discovery.py | _read_discovered_source_posix | open | 3 | unsupported | notes |
| tldw_chatbook/Notes/note_import_discovery.py | _scan_child_directory | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/note_import_discovery.py | _verify_lexical_source_binding | open | 2 | unsupported | notes |
| tldw_chatbook/Notes/note_import_receipts.py | NoteImportReceiptRepository._connect | connect_private_sqlite | 1 | unsupported | notes |
| tldw_chatbook/Notes/sync_engine.py | NotesSyncEngine._write_file_info | write_text | 1 | unsupported | notes |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot.__enter__ | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot._read_file | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot._verified_child_directory | mkdir | 1 | unsupported | notes |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot._verified_child_directory | open | 1 | unsupported | notes |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot._write_all | write | 1 | unsupported | notes |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot.write_text | open | 1 | unsupported | notes |
| tldw_chatbook/Notifications/client_notifications_db.py | ClientNotificationsDB | inherits:BaseDB | 1 | unsupported | miscellaneous |
| tldw_chatbook/Notifications/client_notifications_db.py | ClientNotificationsDB._get_connection | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/Notifications/event_state_repository.py | EventStateRepository | inherits:BaseDB | 1 | unsupported | miscellaneous |
| tldw_chatbook/Notifications/event_state_repository.py | EventStateRepository._get_connection | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/Persona_Visual/assets.py | _decode_selected_frame | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/assets.py | _open_profile_root | open | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/assets.py | _read_profile_file_fallback | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/assets.py | _read_profile_file_secure | open | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _decode | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _read_marker | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _write_private | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _write_private | write | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _write_workspace_asset | open | 6 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _write_workspace_asset | write | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | cleanup_persona_visual_authoring_workspace | open | 6 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | create_persona_visual_authoring_workspace | mkdir | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | create_persona_visual_authoring_workspace | secure_private_directory | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _create_candidate | mkdir | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _delete_candidate | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _extract_assets | open | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _inspect_image | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _member_digest | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _open_candidate | open | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _pin_source | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _private_staging_root | secure_private_directory | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _read_marker | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _write_all | write | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _write_private | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | import_persona_visual_pack | ZipFile | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _cleanup_marker_current | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _delete_pinned_directory | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _open_absolute_directory_chain | open | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _pin_publication_file | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _pin_source_asset | open | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _publication_files_current | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _source_entry_current | open | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _write_private_file | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _write_private_file | write | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | cleanup_persona_visual_publication_candidate | open | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | cleanup_persona_visual_publication_candidate | secure_private_directory | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | publish_persona_visual | mkdir | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | publish_persona_visual | open | 3 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | publish_persona_visual | secure_private_directory | 1 | unsupported | assets |
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
| tldw_chatbook/RAG_Search/pipeline_builder_simple.py | load_pipelines_from_toml | copy2 | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/pipeline_builder_simple.py | load_pipelines_from_toml | mkdir | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/pipeline_builder_simple.py | load_pipelines_from_toml | open | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.export_pipeline_config | dump | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.export_pipeline_config | open | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.load_pipeline_config | copy2 | 1 | unsupported | rag |
| tldw_chatbook/RAG_Search/pipeline_loader.py | PipelineLoader.load_pipeline_config | mkdir | 1 | unsupported | rag |
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
| tldw_chatbook/Scheduling/db/scheduled_tasks_db.py | ScheduledTasksDB | inherits:BaseDB | 1 | unsupported | miscellaneous |
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
| tldw_chatbook/Subscriptions/briefing_audio.py | _looks_like_wav | open | 1 | unsupported | subscriptions |
| tldw_chatbook/Subscriptions/briefing_audio.py | briefing_audio_dir | secure_private_directory | 1 | unsupported | subscriptions |
| tldw_chatbook/Subscriptions/briefing_export.py | _copy_episode_audio_file | open | 2 | unsupported | subscriptions |
| tldw_chatbook/Subscriptions/briefing_export.py | _write_feed_xml_atomically | open | 1 | unsupported | subscriptions |
| tldw_chatbook/Subscriptions/briefing_export.py | _write_feed_xml_atomically | write | 1 | unsupported | subscriptions |
| tldw_chatbook/Subscriptions/site_config_manager.py | SiteConfigManager.export_configs | write_text | 1 | unsupported | subscriptions |
| tldw_chatbook/Sync_Interop/notes_mirror.py | NotesMirror.__init__ | connect_private_sqlite | 1 | unsupported | miscellaneous |
| tldw_chatbook/Sync_Interop/sync_state_repository.py | SyncStateRepository | inherits:BaseDB | 1 | unsupported | miscellaneous |
| tldw_chatbook/Sync_Interop/sync_state_repository.py | SyncStateRepository._get_connection | connect_private_sqlite | 1 | unsupported | miscellaneous |
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
| tldw_chatbook/TTS/profile_migration_namespace.py | admit_zero_reusable_tombstone | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_namespace.py | move_exact_noreplace | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_namespace.py | open_new_or_reused_private_file | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_namespace.py | prepare_reusable_tombstone | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_namespace.py | remove_exact | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_namespace.py | remove_zero_reusable_tombstone | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_namespace.py | require_reusable_tombstone | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _append_journal | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _append_journal | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _immutable_validate | connect_private_sqlite_descriptor | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _open_exact | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _prepare_parent | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_publication.py | _write_new_journal | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_recovery.py | _open_leaf | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_recovery.py | _validate_authoritative_targets | connect_private_sqlite_descriptor | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_recovery.py | recover_profile_migration_publication | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_audio.py | _read_regular_source | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _create_materialization_sync | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _create_materialization_sync | open | 4 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _create_materialization_sync | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _prepare_runtime_root_sync | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _prepare_runtime_root_sync | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _sweep_orphans | open | 3 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _validate_materialization_sync | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_storage.py | write_reference_blob | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_backup_to | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_create_recovery_backup | backup_connection_to_private | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_exact_schema_version | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_online_backup | backup_open_connections_to_private | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_publish_migrated_store | connect_private_sqlite | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_schema_version_for_restore | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_validate_restore_source | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_validate_standalone_snapshot | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | _fsync_directory | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | _fsync_file | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _copy_source_to_snapshot | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _open_candidate_source | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _open_exact_store_sidecars | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | capture_post_init_profile_store_authority | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | open_exact_current_profile_store | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | open_exact_current_profile_store | connect_private_sqlite_descriptor | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | open_exact_current_profile_store | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | open_profile_store | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | peek_profile_store_schema_version | connect_private_sqlite | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | validate_profile_candidate | connect_private_sqlite | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_store_lock.py | ProfileStoreLease.acquire | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/sample_audio_validation.py | _read_bounded_regular_file | open | 1 | unsupported | tts |
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
| tldw_chatbook/TTS/voice_bundle_service.py | _copy_and_inspect | open | 4 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _copy_and_inspect | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _create_operation | mkdir | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _create_operation | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _create_operation_file | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _fingerprint_source_sync | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _prepare_root_sync | open | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _prepare_root_sync | secure_private_directory | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _publish_sync | open | 2 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _publish_sync | write | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _published_file_matches | open | 1 | unsupported | tts |
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
| tldw_chatbook/UI/Screens/chat_screen.py | ChatScreen._load_sidebar_state | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/chat_screen.py | ChatScreen._write_sidebar_state_snapshot | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/chat_screen.py | ChatScreen._write_sidebar_state_snapshot | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/chat_screen.py | ChatScreen._write_sidebar_state_snapshot | open | 2 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/image_gen_demo_screen.py | ImageGenDemoScreen._render_result | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/library_screen.py | LibraryScreen._write_library_note_export_file | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/library_screen.py | LibraryScreen._write_library_prompt_export_file | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._dictionary_export_worker | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._dictionary_export_worker | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._export_expression_set | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._export_expression_set | write_bytes | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Screens/personas_screen.py | PersonasScreen._write_text_file | write_text | 1 | unsupported | miscellaneous |
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
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager.__init__ | makedirs | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager._load_mappings | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager.install_tokenizer | copy2 | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager.save_mappings | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/custom_tokenizers.py | CustomTokenizerManager.save_mappings | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/egress.py | guarded_fetch_requests | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/fd_protection.py | protect_file_descriptors | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/file_handlers.py | DataFileHandler._process_csv | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/file_handlers.py | DataFileHandler._process_json | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/file_handlers.py | DataFileHandler._process_yaml | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/file_handlers.py | DataFileHandler._process_yaml | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/instance_lock.py | acquire_profile_instance_lock | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/instance_lock.py | acquire_profile_instance_lock | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/log_widget_manager.py | LogWidgetManager.update_log | write | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/note_importers.py | CSVImporter.parse_file | open | 1 | unsupported | files |
| tldw_chatbook/Utils/note_importers.py | JSONImporter.parse_file | open | 1 | unsupported | files |
| tldw_chatbook/Utils/note_importers.py | YAMLImporter.parse_file | open | 1 | unsupported | files |
| tldw_chatbook/Utils/paths.py | get_project_databases_dir | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/private_paths.py | _follow_trusted_symlink | open | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | _open_directory_component | open | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | _open_leaf_for_create | open | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | _open_verified_parent | open | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | _prepare_application_owned_parent | secure_private_directory | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_bytes | open | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_bytes | write | 2 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_text | atomic_private_write_bytes | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | create_private_text | open | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | create_private_text | secure_private_directory | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | create_private_text | write | 2 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | open_private_binary | open | 2 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | open_private_text_append | open_private_text_append_stream | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | open_private_text_append_stream | mkdir | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | open_private_text_append_stream | open | 2 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | secure_private_directory | mkdir | 2 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | secure_private_directory | open | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | verify_trusted_directory | open | 1 | generic_boundary | generic |
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
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | JSONStorage.__init__ | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | JSONStorage._create_backup | copy2 | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | JSONStorage._read_data | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | JSONStorage._write_data | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | JSONStorage._write_data | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py | SQLiteStorage._connect | connect_private_sqlite | 2 | unsupported | miscellaneous |
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
| tldw_chatbook/Widgets/emoji_picker.py | load_recent_emojis | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/emoji_picker.py | save_recent_emoji | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/emoji_picker.py | save_recent_emoji | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/emoji_picker.py | save_recent_emoji | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/file_extraction_dialog.py | FileExtractionDialog._save_files | write_text | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/project_skills_import_modal.py | _read_loose_skill_file_sync | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.__init__ | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor._load_user_themes | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.load_user_theme | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.on_export_theme | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.on_export_theme | mkdir | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.on_export_theme | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.on_save_theme | dump | 1 | unsupported | miscellaneous |
| tldw_chatbook/Widgets/settings_theme_editor.py | SettingsThemeEditor.on_save_theme | open | 1 | unsupported | miscellaneous |
| tldw_chatbook/Workspaces/change_retention.py | prune_change_history | mkdir | 2 | unsupported | workspaces |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo._locked._Lock.__enter__ | mkdir | 2 | unsupported | workspaces |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo.ensure_initialized | mkdir | 3 | unsupported | workspaces |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo.ensure_initialized | write_text | 1 | unsupported | workspaces |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo.snapshot | open | 1 | unsupported | workspaces |
| tldw_chatbook/Workspaces/change_tracking.py | ShadowRepo.snapshot | write | 3 | unsupported | workspaces |
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
| tldw_chatbook/runtime_policy/source_state.py | RuntimeSourceStateStore.load | open_private_binary | 1 | unsupported | runtime |
| tldw_chatbook/runtime_policy/source_state.py | RuntimeSourceStateStore.save | atomic_private_write_text | 1 | unsupported | runtime |
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
| tldw_chatbook/Chatbooks/chatbook_creator.py | ChatbookCreator._create_zip_archive | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py | ModelCatalogDiskStore.save | os.replace | 1 | disposable | diagnostics |
| tldw_chatbook/Media/local_media_reading_service.py | LocalMediaReadingService._default_url_file_downloader | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/Model_Artifacts/service.py | ModelArtifactService._copy_payload | os.replace | 1 | unsupported | models |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_file | os.replace | 1 | unsupported | notes |
| tldw_chatbook/RAG_Search/config_profiles.py | ConfigProfileManager._migrate_legacy_blob | os.replace | 1 | unsupported | rag |
| tldw_chatbook/Subscriptions/briefing_export.py | _write_feed_xml_atomically | os.replace | 1 | unsupported | subscriptions |
| tldw_chatbook/TTS/profile_migration_namespace.py | _rename_noreplace | renameatx_np | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_repository.py | TTSProfileRepository._worker_backup_to | os.replace | 1 | unsupported | tts |
| tldw_chatbook/UI/Console_Modules/video.py | ConsoleVideoController._copy_pending_video_external | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_databases | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/UI/Tools_Settings_Window.py | ToolsSettingsWindow._backup_worker | os.replace | 1 | unsupported | miscellaneous |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_text | os.replace | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_bytes | os.replace | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_copy | os.replace | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_bytes | os.replace | 1 | generic_boundary | generic |
| tldw_chatbook/Video_Generation/video_store.py | VideoStore._commit_sibling | os.replace | 1 | process_artifact | process |
| tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py | recursive_scrape.save_progress | os.replace | 1 | unsupported | miscellaneous |

## Local write-intent retirement census (TASK-31987 review fix)

The control owner now exclusively creates bounded version-1 before/after write intents through `Admission._write_new_record`, and retires a matching intent through `os.unlink` only after the registry file/parent full native barriers complete. Intent removal is a persistence boundary, not disposable cleanup authority. The census now recognizes concrete OS unlink calls/import aliases, including the existing removal candidates below. Existing unsupported/generic classifications are retained per producer; newly surfaced symbols remain unsupported within their known cohort (the secure temporary-file helper remains a generic boundary). No new capture/exclusion qualification is implied.

| Module | Qualified symbol | Call | Count | Classification | Cohort |
| --- | --- | --- | --- | --- | --- |
| tldw_chatbook/Character_Chat/visual_identity.py | _discard_pinned_directory | os.unlink | 1 | unsupported | files |
| tldw_chatbook/Chat/trajectory_export.py | write_trajectory_export | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py | extract_text_from_image | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/OCR_Backends.py | DocextOCRBackend.process_pdf | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend.transcribe | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend.transcribe_buffer | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend._transcribe_with_parakeet_mlx | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/transcription_service.py | _LegacyTranscriptionBackend._transcribe_buffer_with_parakeet_mlx | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Local_Ingestion/video_processing.py | LocalVideoProcessor._discard_temp_cookiefile | os.unlink | 1 | unsupported | miscellaneous |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_file | os.unlink | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.save_copy | os.unlink | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.export_exact_file | os.unlink | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.create_file | os.unlink | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.move_file | os.unlink | 2 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.delete_file | os.unlink | 1 | unsupported | notes |
| tldw_chatbook/Notes/file_notes_service.py | FileNotesService.restore_file | os.unlink | 1 | unsupported | notes |
| tldw_chatbook/Notes/sync_paths.py | PinnedSyncRoot.write_text | os.unlink | 1 | unsupported | notes |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | cleanup_persona_visual_authoring_workspace | os.unlink | 2 | unsupported | assets |
| tldw_chatbook/Persona_Visual/authoring_workspace.py | _write_workspace_asset | os.unlink | 1 | unsupported | assets |
| tldw_chatbook/Persona_Visual/importer.py | _delete_candidate | os.unlink | 3 | unsupported | assets |
| tldw_chatbook/Persona_Visual/publication.py | _delete_pinned_directory | os.unlink | 1 | unsupported | assets |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | AudioCppGeneratedLaunchArtifact.cleanup | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_cpp_guided_launch.py | _remove_partial_artifact | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/audio_service.py | AudioService.create_m4b_with_chapters | os.unlink | 2 | unsupported | tts |
| tldw_chatbook/TTS/backends/chatterbox.py | ChatterboxTTSBackend._transcribe_audio | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_migration_namespace.py | remove_zero_reusable_tombstone | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _create_materialization_sync | os.unlink | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _sweep_orphans | os.unlink | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_reference_materialization.py | _cleanup_materialization_sync | os.unlink | 2 | unsupported | tts |
| tldw_chatbook/TTS/profile_schema.py | _unlink_if_present | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/TTS/voice_bundle_service.py | _cleanup_operation | os.unlink | 1 | unsupported | tts |
| tldw_chatbook/UI/Console_Modules/video.py | ConsoleVideoController._copy_pending_video_external | os.unlink | 2 | unsupported | miscellaneous |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_text | os.unlink | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_write_bytes | os.unlink | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/atomic_file_ops.py | atomic_copy | os.unlink | 1 | generic_boundary | generic |
| tldw_chatbook/Utils/private_paths.py | atomic_private_write_bytes | os.unlink | 2 | generic_boundary | generic |
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

| tldw_chatbook/Backup_Recovery/admission.py | Admission._create_lock | open | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._open | open | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._write_new_record | open | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._write_new_record | write | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._write | os.replace | 1 | unsupported | backup_control |
| tldw_chatbook/Backup_Recovery/admission.py | Admission._write | os.unlink | 1 | unsupported | backup_control |
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
| tldw_chatbook/Backup_Recovery/storage_admission.py | _read_recovery_file | open | 1 | generic_boundary | native-domain-recovery |
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
Hard OS FD-close failure is conservatively latched and retains native exclusion but
was not runtime fault-induced here; this remains explicit release qualification, as
with the task 6 native-close limit. No Complete backup/publication/replacement UI is
qualified by this cohort alone.
