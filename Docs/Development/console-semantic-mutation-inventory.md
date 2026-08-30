# Console semantic mutation inventory

This document is the TASK-23113.2 baseline for the mutation-integrity boundary
decided by [ADR-097](../../backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md).
It inventories live Python routes that can mutate canonical `messages` rows or
message-owned sidecars. It does not change any writer. Task 6 adds the database
guard and coordinator; Task 7 routes the model-visible owners named here through
that coordinator.

## Classification

- **model-visible**: can change a provider-visible message envelope: role,
  content, image/attachment identity, continuation, thinking/reasoning replay,
  assistant generation state, or a selected generation's canonical projection.
  These routes require semantic-revision coordination. A mixed-purpose primitive
  such as `CharactersRAGDB.update_message` is model-visible because its allowlist
  permits semantic fields, even though some callers restrict it to presentation
  fields.
- **visibility/ownership-only**: changes selection, tree membership, variant
  ownership, or a soft-delete tombstone while retaining the semantic bytes. These
  routes update surface/ownership state, but do not create a new semantic revision
  merely for the visibility change.
- **presentation-only**: local usage, feedback/ranking, UI metadata, image-
  generation provenance, legacy exchange capture, or trajectory diagnostics.
  These values do not change the provider-neutral message envelope.

The structural test records 38 live SQL sink identities and 64 boundary call
identities: 66 model-visible, 11 visibility/ownership-only, and 25
presentation-only identities in total. These are route-layer identities, not 102
distinct user actions; a public action appears once at its boundary and again at
the SQL sink it reaches.

## Public owners and behavior

Public-owner table is manual guidance: the structural contract below synchronizes
the two exact census sections bidirectionally, while this table explains the
user-facing ownership chain and must be reviewed when a route changes.

| Behavior | Classification | Public owner(s) | Current sink |
|---|---|---|---|
| Console durable turn acceptance | model-visible | `ConsoleChatStore.commit_durable_turn`; `ChatPersistenceService.commit_durable_turn`; `ConsoleDispatchRepository.insert_with_messages` | inserts USER/assistant `messages` and USER `message_attachments` |
| Console generation settlement, stop/failure, retry state, and continuation handoff/recovery | model-visible | `ConsoleChatStore.settle_dispatch_recovery`, `transition_dispatch_recovery_for_retry`, provider-continuation restore actions; `ConsoleDispatchRepository` settlement/CAS methods | updates content, thinking, continuation, and `assistant_generation_state` |
| Console append and ordinary terminal persistence | model-visible | `ConsoleChatStore.append_message`, terminal-message methods, `persist_message_if_needed`; durable sink owner `ChatPersistenceService.create_message` | `CharactersRAGDB.add_message`, attachment rewrite, optional presentation sidecars |
| Edit and regeneration replacement | model-visible | `ConsoleChatStore.update_message_content`, `select_variant`, `finalize_variant_stream`, `persist_selected_generation`; durable sink owners `_persist_existing_message` and `ChatPersistenceService.update_message_content`/`replace_assistant_generation_projection` | canonical `messages` update and optional attachment rewrite |
| Image/video generation settlement | model-visible | `ConsoleChatStore.append_generation_message`, `append_video_message`, `_persist_terminal_generation` | `ChatPersistenceService.create_message` or selected-generation replacement |
| Attachment mutation and selected image variant | model-visible | `ConsoleChatStore.append_generation_variant`, `keep_generation_variant`; `ChatPersistenceService.append_message_attachment`, `keep_message_attachment` | attachment append or scalar/attachment swap; generation metadata re-key is presentation-only |
| Console fork and temporary promotion | model-visible | `ConsoleSessionController._commit_durable_console_chat_fork`; `ConsoleChatStore.promote_ephemeral_session`; `ChatPersistenceService.fork_console_conversation_bundle`, `promote_console_conversation_bundle` | creates each canonical message and its attachments in the destination conversation |
| Classic chat bulk save | model-visible plus visibility | `Chat_Functions.save_chat_history_to_db_wrapper`; `ChatPersistenceService.save_history` | create/update retained rows and soft-delete omitted rows |
| Classic character create/post/edit | model-visible | `Character_Chat_Lib.create_conversation`, `start_new_chat_session`, `add_message_to_conversation`, `post_message_to_conversation`, `edit_message_content` | `add_message` or semantic `update_message` |
| Character API create/update | model-visible | `LocalCharacterPersonaService.create_character_chat_message`, `update_character_chat_message` | `add_message` or semantic `update_message` |
| Console character-greeting projection and repair | model-visible | `ConsoleChatStore.persist_roleplay_projection_plan`; producer `_snapshot_roleplay_message_projection_write` | frozen `_RoleplayMessageProjectionWrite.writer` invokes `ChatPersistenceService.update_message_content` |
| Import | model-visible plus visibility | `ChatbookImporter.import_chatbook` via `_import_conversations`; `Character_Chat_Lib.load_chat_history_from_file_and_save_to_db` | creates messages/attachments; V2 graph restoration writes variant/selection/tombstone fields |
| Provider-continuation fallback create/update/discard | model-visible | `ConsoleChatStore.persist_provider_continuation_event`, `ConsoleChatStore.discard_provider_continuation` | `CharactersRAGDB.create_assistant_with_continuation` or `update_provider_continuation` through the persistence DB handle |
| Sync create/update/delete | model-visible plus visibility | `SyncEnvelopeApplier.apply` → `ChatSyncAdapter.apply` → `_ContinuationValidatingChatStore` → `CharactersRAGDB.append_chat_message`/`delete_chat_message` | create/update changes the canonical envelope; delete retains those bytes and changes only tombstone visibility/ownership plus graph epoch |
| Regeneration variants in legacy DB API | model-visible plus visibility | `CharactersRAGDB.create_message_variant`, `select_message_variant` | new variant row plus selected/total flags |
| Research completion handoff | model-visible | `insert_research_completion_message` | assistant `add_message` |
| Message and subtree soft delete, including Sync tombstones | visibility/ownership-only | `ConsoleChatStore.delete_message`; classic/persona delete APIs; `CharactersRAGDB.delete_chat_message` | retained semantic bytes and semantic-revision lineage stay unchanged; only tombstone visibility/ownership state and graph epoch advance |
| Usage, feedback/ranking, message UI metadata | presentation-only | `ConsoleChatStore.set_message_usage`, `ConsoleChatStore.set_message_feedback`, `ConsoleChatStore.set_message_metadata`; classic ranking; DB feedback helper | local/version-neutral metadata or restricted `update_message` call |
| Reasoning/tool trajectory diagnostics | presentation-only | `ConsoleChatStore.write_trajectory_rows`; `ChatPersistenceService.write_trajectory_rows`; `LibraryActivityContribution.write`, `LibraryPreparationContribution.write` | `message_trajectory_metadata`; tool/reasoning payload is diagnostic and is not replayed into provider kwargs |
| Legacy exchange capture and purge | presentation-only | `ConsoleChatStore.attach_message_exchanges`, terminal exchange-flush paths, `ConsoleChatStore.commit_full_capture_purge`; `ChatPersistenceService.append_message_exchanges`, `delete_full_exchanges_for_conversation` | `message_exchanges` only |
| Image-generation provenance | presentation-only | `ChatPersistenceService.create_message` generation metadata and attachment helpers | `message_generation_metadata`; the attachment bytes/selection remain separately model-visible |

`Chat_Functions.py` has no direct SQL message writer. Its one durable public
mutation route is the bulk-save wrapper above. `console_chat_store.py` likewise
does not open SQLite or issue message SQL; it owns in-memory state and delegates
durability to `ChatPersistenceService` or `ConsoleDispatchRepository`.

There is no separate canonical tool-call table in this database. Replayable
tool-loop state and model reasoning are stored in the `messages` row's
`provider_continuation_json` and `thinking_blocks_json` fields, so add, edit,
dispatch settlement, import, and Sync routes above cover those semantic values.
Tool markers and reasoning rows in `message_trajectory_metadata` are a separate
diagnostic projection and remain presentation-only.

The soft-delete SQL sinks and their direct public owners are classified as
visibility/ownership-only. They set tombstone state and advance the graph epoch
without changing retained semantic bytes, retiring the live locator, or creating
a semantic successor. Mixed Sync apply functions remain model-visible because
the same public function can also create or update canonical message content.

`rag_message_trace_owners` and citation trace tables are intentionally outside
this census. They bind citation provenance to a message/version for display and
verification but do not change the message envelope supplied to a provider;
their repository already owns a separate revision and lifecycle contract.
Likewise, conversation library policy can change future request preparation but
is conversation-owned configuration, not a mutation of an existing message
semantic revision. Later trace-header/provenance tasks capture those request
inputs at the call boundary.

## Hard deletion

There is currently no live public hard-delete route for a `messages` row or a
`conversations` row. Message deletion APIs are soft deletes. The schema has
`messages.conversation_id ... ON DELETE CASCADE`, so a future direct hard delete
of a conversation is also a hard delete of every owned message and sidecar. The
census therefore treats any live `DELETE FROM conversations` as an unclassified
`conversations(cascades-messages)` route, in addition to detecting direct
`DELETE FROM messages`. Task 6 must install a fail-closed guard for both direct
message deletion and cascade deletion before Task 7 can expose a hard-delete
owner. ADR-097 requires materialization and canonical deletion to commit in the
same coordinator transaction.

## Generated and dynamic SQL

The scanner parses module and function ASTs, ignores docstrings and line
numbers, and normalizes a route to
`path::qualified-function::verb:table`. It recognizes unquoted, quoted, and
schema-qualified target identifiers in literal SQL, local literal constants,
module literal constants used directly by `execute`/`executemany`, and f-strings
whose target table is literal. Leading SQL comments and `WITH`/CTE prefixes do
not hide the subsequent mutation template. A direct executor f-string or
concatenation with an unresolved dynamic mutation target fails the census
instead of being silently ignored; this includes a dynamic schema in
`UPDATE {schema}.messages`, which is not assumed safe merely because the final
table component is literal.

The test contains an exact, self-checking allowlist for 20 existing dynamic
executor call sites. Each review pins the normalized call template and source,
its proven non-chat domain, and an exact target set that the test mechanically
re-derives from the current production AST. The 20 sites form 18 target-evidence
families. Their deliberately narrow evidence forms are literal arguments at
private helper call sites, literal list/tuple/map table producers, and local or
imported string constants (including a literal f-string derived from an imported
constant). A normalized SHA-256 fingerprint pins the complete selected
producer/caller/import evidence, so changing its structure requires review even
when the resulting table names do not change. The derived set must exactly equal
the recorded `exact_targets`; adding `messages`, another non-chat table, or only
claiming a runtime-supplied table fails. It is not a function-wide exemption:
literal canonical actions in the same function remain in the census, and a
second dynamic call, changed template/source, canonical retarget, stale review,
or changed target evidence fails. The reviewed domains cover keyword/library,
media, evaluation, notification, research, TTS, and writing tables.

This evidence check is a deterministic structural contract, not general
interprocedural dataflow. It follows only the explicitly declared literal helper
calls, literal container assignment, or string-constant import chain for each
reviewed family. Reviewed container producers must have one immutable binding;
later rebinding, subscript writes/deletes, and mutating or unknown method calls
fail closed. Every load of those containers is parent-checked and is allowed only
as a direct loop iterator, read subscript, or receiver of the small explicit
read-only method set needed by production. Aliasing it, passing it to a call, or
extracting one of its methods fails. A scope-aware name visitor distinguishes
module loads from true function/lambda locals, evaluates comprehension iterables
in their progressively nested scope, and applies class-local bindings in
statement order. Definition-time defaults, keyword defaults, decorators, and
annotations are checked in their enclosing execution scope; an unresolved
module-container capture there or before a class-local binding fails closed.
Class namespace propagation is modeled only for direct body statements. Any
reviewed-symbol use or binding nested in a compound class statement (`if`, loop,
`try`, `with`, or `match`) fails closed rather than relying on branch-sensitive
namespace inference.
Reviewed local and imported string constants must likewise have exactly one
module-scope binding at every resolved step. Every load of a reviewed private
helper symbol must be the function of its recognized direct call; capturing a
bound method in a variable/container or passing it as a callback fails. `getattr`
names formed entirely from literal strings (including static f-strings and
concatenation) are resolved and checked. A truly runtime-dynamic `getattr` name
cannot prove that it excludes a reviewed helper, so it also fails closed anywhere
this target-evidence audit applies. Targets assembled at runtime, forwarded
through unspecified helpers, or supplied only by annotations/review metadata are
forbidden until a new narrow evidence form and regression are added.

Discovery is restricted to `execute`/`executemany` arguments resolved directly
or through lexical local/module SQL constants, plus the single explicit
SQL-return helper `CharactersRAGDB._messages_insert_statement`. Arbitrary
unexecuted strings (including diagnostic text) are not sinks. SQL returned by
any other helper or assigned through an unresolved local name is not inferred
and is forbidden for canonical message/sidecar mutation unless the scanner
contract is extended first. `_messages_insert_statement` is the intentional
dynamic-value case inside this census: its target is the literal `messages`
table while its column list and placeholders are generated from a fixed field
tuple, so the helper itself is classified as the `messages` INSERT sink and
`add_message` is enumerated at every recognized DB boundary call.

The boundary scanner recognizes DB calls only through the exact `db`, `self.db`,
`self._require_db()`, recognized local DB-handle aliases, the database class's
own `self`, and the explicit Sync forwarding store. `_require_db()` must be the
exact `self._require_db()` shape; unrelated receivers such as `widget.db` and
`widget._require_db()` are not DB boundaries. Persistence and Console dispatch
calls use their separate receiver allowlists. A local alias assigned from
`getattr(recognized_receiver, "literal_mutator_name", ...)` is tracked in lexical
order: a call uses the binding active at that point, reassignment can expose a
second recognized action, and any unrecognized reassignment invalidates the old
binding. This straight-line rule does not attempt branch-sensitive control-flow
merging. It pins `ConsoleChatStore.delete_message` even though that method
feature-detects `delete_message_subtree` through `deleter`.

Two-hop DB aliases are recognized only when a local name is assigned from
`getattr(persistence-or-persistence_service, "db", None)`, including the exact
none-guarded conditional form used by continuation persistence, and a second
literal `getattr` obtains an allowlisted DB mutator from that name. Chat Sync has
two additional explicit seams: literal `call_if_present(local_store,
"append_chat_message"/"delete_chat_message", ...)` calls inside
`ChatSyncAdapter.apply`, and same-name forwarding aliases on
`_ContinuationValidatingChatStore.store`. The fork controller's bound-method
handoff is recognized only when an allowlisted persistence method is passed as a
positional argument or named keyword value to
`ConsoleSessionController._commit_durable_console_chat_fork`'s `_run_fork_io`.
Other string dispatchers, forwarding stores, runner callbacks, and multi-hop DB
handle flows are not inferred.

One purpose-built carrier rule follows the character-greeting writer across its
frozen plan: the scanner derives the action stored in
`_RoleplayMessageProjectionWrite.writer`, then recognizes calls through items
iterated from an argument annotated as
`ConsoleRoleplayProjectionPersistencePlan.message_writes`. The recognized
producer alias is evaluated in lexical order, so multiple active writer
assignments are retained and an unrelated reassignment invalidates the stale
binding before a carrier is constructed. It does not perform general container
or object-attribute dataflow. Aliases copied through another variable, returned
from a helper, stored in any other carrier, or built from a runtime-computed
method name can still evade static discovery; those forms are forbidden for this
boundary unless a similarly narrow scanner contract and its limitations are
deliberately updated in the same change. Historical
`_migrate_from_*` bodies, schema DDL/triggers, and SQL
migration files are excluded because they are not live application mutation
routes and have their own migration/packaging tests. FTS and Sync-log triggers
are derived indexes/logs, not canonical semantic owners.

## Exact live SQL sink census

- `tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._reconcile_checkpoint_row_uncoordinated::sql:update:messages` — model-visible
- `tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._cas_state_uncoordinated::sql:update:messages` — model-visible
- `tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._handoff_to_provider_continuation_uncoordinated::sql:update:messages` — model-visible
- `tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository.insert_with_messages::sql:insert:message_attachments` — model-visible
- `tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository.insert_with_messages::sql:insert:messages` — model-visible
- `tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._normalize_provider_continuation_owner_uncoordinated::sql:update:messages` — model-visible
- `tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._settle_with_assistant_uncoordinated::sql:update:messages` — model-visible
- `tldw_chatbook/Chat/console_semantic_revision.py::SemanticRevisionCoordinator._mutate_message::sql:delete:messages` — model-visible
- `tldw_chatbook/Chat/library_activity.py::LibraryActivityContribution.write::sql:insert:message_trajectory_metadata` — presentation-only
- `tldw_chatbook/Chat/library_preparation.py::LibraryPreparationContribution.write::sql:insert:message_trajectory_metadata` — presentation-only
- `tldw_chatbook/Chatbooks/chatbook_importer.py::ChatbookImporter._import_conversations::sql:update:messages` — visibility/ownership-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._messages_insert_statement::sql:insert:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._update_message_uncoordinated::sql:update:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_chat_message.apply_sync_message::sql:update:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_message_attachment_with_metadata.append_attachment::sql:insert:message_attachments` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_message_attachment_with_metadata.append_attachment::sql:insert:message_generation_metadata` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_message_attachment_with_metadata.append_attachment::sql:update:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_message_exchanges_local::sql:insert:message_exchanges` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.create_assistant_with_continuation::sql:insert:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.create_message_variant::sql:insert:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.create_message_variant::sql:update:messages` — visibility/ownership-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.delete_full_exchanges_for_conversation::sql:delete:message_exchanges` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._replace_assistant_generation_projection_uncoordinated::sql:update:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.select_message_variant::sql:update:messages` — visibility/ownership-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._set_message_attachments_uncoordinated::sql:delete:message_attachments` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._set_message_attachments_uncoordinated::sql:insert:message_attachments` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.set_message_generation_metadata::sql:delete:message_generation_metadata` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.set_message_generation_metadata::sql:insert:message_generation_metadata` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._set_message_feedback_uncoordinated::sql:update:messages` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.soft_delete_message::sql:update:messages` — visibility/ownership-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.soft_delete_message_subtree::sql:update:messages` — visibility/ownership-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.swap_message_attachment_with_scalar.swap_attachment::sql:update:message_attachments` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.swap_message_attachment_with_scalar.swap_attachment::sql:update:message_generation_metadata` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.swap_message_attachment_with_scalar.swap_attachment::sql:update:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.update_message_metadata_local::sql:update:messages` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.update_message_usage_local::sql:update:messages` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._update_provider_continuation_uncoordinated::sql:update:messages` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.upsert_trajectory_rows::sql:insert:message_trajectory_metadata` — presentation-only

## Exact boundary-call census

- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py::add_message_to_conversation::call:db:add_message` — model-visible
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py::create_conversation::call:db:add_message` — model-visible
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py::edit_message_content::call:db:update_message` — model-visible
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py::load_chat_history_from_file_and_save_to_db::call:db:add_message` — model-visible
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py::post_message_to_conversation::call:db:add_message` — model-visible
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py::remove_message_from_conversation::call:db:soft_delete_message` — visibility/ownership-only
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py::set_message_ranking::call:db:update_message` — presentation-only
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py::start_new_chat_session::call:db:add_message` — model-visible
- `tldw_chatbook/Character_Chat/local_character_persona_service.py::LocalCharacterPersonaService.create_character_chat_message::call:db:add_message` — model-visible
- `tldw_chatbook/Character_Chat/local_character_persona_service.py::LocalCharacterPersonaService.delete_character_chat_message::call:db:soft_delete_message` — visibility/ownership-only
- `tldw_chatbook/Character_Chat/local_character_persona_service.py::LocalCharacterPersonaService.update_character_chat_message::call:db:update_message` — model-visible
- `tldw_chatbook/Chat/Chat_Functions.py::save_chat_history_to_db_wrapper::call:persistence:save_history` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.append_message_attachment::call:db:append_message_attachment_with_metadata` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.append_message_exchanges::call:db:append_message_exchanges_local` — presentation-only
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.commit_durable_turn::call:dispatch:insert_with_messages` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.create_message::call:db:add_message_with_semantic_sidecars` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.delete_full_exchanges_for_conversation::call:db:delete_full_exchanges_for_conversation` — presentation-only
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.delete_message_subtree::call:db:soft_delete_message_subtree` — visibility/ownership-only
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.fork_console_conversation_bundle::call:persistence:create_message` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.keep_message_attachment::call:db:swap_message_attachment_with_scalar` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.promote_console_conversation_bundle::call:persistence:create_message` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.replace_assistant_generation_projection::call:db:replace_assistant_generation_projection` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.save_history::call:db:soft_delete_message` — visibility/ownership-only
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.save_history::call:persistence:create_message` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.save_history::call:persistence:update_message_content` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.update_message_content.coordinated_update::call:db:update_message` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.update_message_content.coordinated_update::call:db:update_message_with_attachments` — model-visible
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.update_message_metadata::call:db:update_message_metadata_local` — presentation-only
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.update_message_usage::call:db:update_message_usage_local` — presentation-only
- `tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.write_trajectory_rows::call:db:upsert_trajectory_rows` — presentation-only
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._create_terminal_message::call:persistence:create_message` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._discard_provider_continuation::call:db:update_provider_continuation` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._normalize_restored_provider_continuation::call:dispatch:normalize_provider_continuation_owner` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_exchanges_only_locked::call:persistence:append_message_exchanges` — presentation-only
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_existing_message::call:persistence:update_message_content` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_generation_variant::call:persistence:replace_assistant_generation_projection` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_metadata_only::call:persistence:update_message_metadata` — presentation-only
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_new_message::call:persistence:create_message` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_usage_only::call:persistence:update_message_usage` — presentation-only
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._promote_ephemeral_session_atomically::call:persistence:promote_console_conversation_bundle` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._settle_dispatch_recovery::call:dispatch:settle_with_assistant` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._append_generation_variant::call:persistence:append_message_attachment` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.commit_durable_turn::call:persistence:commit_durable_turn` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.commit_full_capture_purge::call:persistence:delete_full_exchanges_for_conversation` — presentation-only
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._delete_message::call:persistence:delete_message_subtree` — visibility/ownership-only
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._keep_generation_variant::call:persistence:keep_message_attachment` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_provider_continuation_event::call:db:create_assistant_with_continuation` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_provider_continuation_event::call:db:update_provider_continuation` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_provider_continuation_event::call:dispatch:handoff_to_provider_continuation` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_roleplay_projection_plan::call:persistence:update_message_content` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.transition_dispatch_recovery_for_retry::call:dispatch:cas_state` — model-visible
- `tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.write_trajectory_rows::call:persistence:write_trajectory_rows` — presentation-only
- `tldw_chatbook/Chatbooks/chatbook_importer.py::ChatbookImporter._import_conversations::call:db:add_message` — model-visible
- `tldw_chatbook/Chatbooks/chatbook_importer.py::ChatbookImporter._import_conversations::call:db:add_message_with_semantic_sidecars` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._add_message_with_semantic_sidecars::call:db:set_message_generation_metadata` — presentation-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_chat_message::call:db:add_message` — model-visible
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.delete_chat_message::call:db:soft_delete_message` — visibility/ownership-only
- `tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.update_message_feedback::call:db:update_message` — presentation-only
- `tldw_chatbook/Research_Interop/chat_handoff.py::insert_research_completion_message::call:db:add_message` — model-visible
- `tldw_chatbook/Sync_Interop/domain_adapters/chat.py::ChatSyncAdapter.apply::call:db:append_chat_message` — model-visible
- `tldw_chatbook/Sync_Interop/domain_adapters/chat.py::ChatSyncAdapter.apply::call:db:delete_chat_message` — model-visible
- `tldw_chatbook/Sync_Interop/envelope_applier.py::_ContinuationValidatingChatStore.append_chat_message::call:db:append_chat_message` — model-visible
- `tldw_chatbook/Sync_Interop/envelope_applier.py::_ContinuationValidatingChatStore.delete_chat_message::call:db:delete_chat_message` — model-visible
- `tldw_chatbook/UI/Console_Modules/session.py::ConsoleSessionController._commit_durable_console_chat_fork::call:persistence:fork_console_conversation_bundle` — model-visible
