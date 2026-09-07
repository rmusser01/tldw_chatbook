# Task 0 Decision: System-Prompt Persistence Seam

Decision for the per-session Console system prompt feature (Library ▸ Prompts + Console
injection, `Docs/superpowers/plans/2026-07-12-library-prompts-console-injection.md`).
Consumed by Task 13 ("System-prompt plumbing + persistence").

## Decision

**Migration (option b).** Add `system_prompt TEXT NULL` to `conversations` via a new
schema migration, `_CURRENT_SCHEMA_VERSION` bump from 17 to 18. No existing column
qualifies for reuse.

## Step 1: Every free/metadata-ish column on `conversations`, and what it already means

`tldw_chatbook/DB/ChaChaNotes_DB.py:246` (`CREATE TABLE IF NOT EXISTS conversations`) defines
the v4 base columns (`id, root_id, forked_from_message_id, parent_conversation_id,
character_id, title, rating, created_at, last_modified, deleted, client_id, version`) — none
of these are free-form metadata (all structural/identity/audit fields).

Three later migrations added every other conversations column:

- `_MIGRATE_V12_TO_V13_SQL` (`ChaChaNotes_DB.py:1677-1813`), `ALTER TABLE` statements at
  `1680-1692`: `assistant_kind`, `assistant_id`, `persona_memory_mode`, `scope_type`,
  `workspace_id`, `state`, `topic_label`, `topic_label_source`, `topic_last_tagged_at`,
  `topic_last_tagged_message_id`, `cluster_id`, `source`, `external_ref`.
- `_MIGRATE_V13_TO_V14_SQL` (`ChaChaNotes_DB.py:1815-1935`), `ALTER TABLE` statements at
  `1818-1820`: `runtime_backend`, `discovery_owner`, `discovery_entity_id`.
- (V14→V17 migrations, `2612`-`2981`, touch other tables — flashcards, quizzes, local
  marks — not `conversations`.)

Grepped every column's readers (`grep -rn "<column>" tldw_chatbook/ --include=*.py | grep -v
Tests`, file counts below) and read the two DB methods that own their semantics end to end:

- `add_conversation` (`ChaChaNotes_DB.py:4194-4327`) normalizes and inserts every one of
  these columns explicitly: assistant identity via `_normalize_conversation_assistant_identity`
  (`4230-4235`), scope via `_normalize_scope` (`4236-4239`), state via
  `_normalize_conversation_state` (`4240`), topic-tagging fields via
  `_normalize_nullable_text`/`_normalize_topic_label_source` (`4241-4247`), runtime/discovery
  via `_normalize_conversation_runtime_visibility` (`4248-4252`). The INSERT column list and
  params are at `4255-4272`.
- `update_conversation` (`ChaChaNotes_DB.py:4813-5030`) re-derives the same fields from a
  `SELECT` of the current row (`4851-4862`) and only rewrites a column when its update key is
  present in `update_data` (per-field `if 'x' in update_data:` blocks, `4915-5001`), each tied
  to the same normalization/allowed-value functions as `add_conversation`.

Column semantics confirmed (docstring at `4211` calls these "assistant/scope/topic
metadata", not free text) and cross-file usage counts (non-test files):

| Column | Owning feature | Files referencing it |
|---|---|---|
| `assistant_kind`/`assistant_id` | persona/character assistant identity | 16 / 17 |
| `persona_memory_mode` | persona read-only/read-write memory | 11 |
| `scope_type`/`workspace_id` | Console workspace scoping | 49 / 79 |
| `state` | conversation kanban state (`in-progress`/`resolved`/…) | 410 |
| `topic_label`* | auto/manual topic tagging | 10 |
| `cluster_id` | conversation clustering | 9 |
| `source`/`external_ref` | import/dedup provenance tracking | 458 / 5 |
| `runtime_backend`/`discovery_owner`/`discovery_entity_id` | agent runtime + discovery visibility | 54 / 16 / 15 |

Every nullable TEXT column on `conversations` is already owned by a specific, actively-used
feature with its own allowed-value/normalization rules (e.g. `state` is constrained to
`_ALLOWED_CONVERSATION_STATES`, `topic_label_source` to `{"manual","auto"}` via
`_normalize_topic_label_source`, `ChaChaNotes_DB.py:4045-4052`). There is no unclaimed
"scratch" TEXT column.

## Step 2: Applying the qualifying rule

Rule: a column qualifies only if (1) TEXT-typed and nullable, (2) not consumed anywhere with
conflicting semantics, (3) round-trips through `update_conversation` with sync triggers
intact.

- `scope_type`, `state`, `runtime_backend`, `discovery_owner` fail (1): all four are
  `NOT NULL DEFAULT '...'` (`ChaChaNotes_DB.py:1683,1685,1818,1819`), not nullable.
- Every remaining nullable TEXT candidate (`assistant_kind`, `assistant_id`,
  `persona_memory_mode`, `workspace_id`, `topic_label`, `topic_label_source`,
  `topic_last_tagged_at`, `topic_last_tagged_message_id`, `cluster_id`, `source`,
  `external_ref`, `discovery_entity_id`) fails (2): each is already consumed with a specific,
  conflicting meaning by `add_conversation`/`update_conversation` and by dozens of call sites
  across the codebase (table above). Repurposing any of them (e.g. writing a system prompt
  into `source`) would corrupt that column's existing contract for every other feature that
  reads it.
- No column reaches step (3) because none passes (1) and (2) together.

**None qualifies → choose the migration**, per the brief's fallback.

## Step 3: The migration Task 13 will add

New column: `system_prompt TEXT` (nullable, no default) on `conversations`.

Model the migration on the most recent one, `_migrate_from_v16_to_v17` /
`_MIGRATE_V16_TO_V17_SQL` (`ChaChaNotes_DB.py:2097-2113` and `2957-2981`), since it is the
simplest additive precedent. Concretely, Task 13 must:

1. Bump `_CURRENT_SCHEMA_VERSION = 17` → `18` at `ChaChaNotes_DB.py:142`.
2. Add a new class string `_MIGRATE_V17_TO_V18_SQL` (place it after `_MIGRATE_V16_TO_V17_SQL`,
   i.e. after line 2113) containing:
   - `ALTER TABLE conversations ADD COLUMN system_prompt TEXT;`
   - Re-issued `conversations_sync_create`/`_update`/`_delete`/`_undelete` triggers — **these
     must be redefined**, because `system_prompt` needs to appear in the JSON `sync_log`
     payload and (for `_update`) in the trigger's `WHEN` clause, or edits to the system prompt
     silently fail to produce a sync-log entry. Model this on the V13→V14 migration
     (`ChaChaNotes_DB.py:1826-1928`), which added `runtime_backend`/`discovery_owner`/
     `discovery_entity_id` the same way: `DROP TRIGGER IF EXISTS conversations_sync_*` then
     `CREATE TRIGGER` with `system_prompt` added to each `json_object(...)` payload
     (`1843,1850,1886,1893` for the pattern in the live `_create`/`_update` triggers) and
     `OLD.system_prompt IS NOT NEW.system_prompt OR` added to the `conversations_sync_update`
     `WHEN` clause (pattern at `1859-1882`). `conversations_sync_delete`/`_undelete` (the live,
     final versions redefined by this same V13→V14 migration, at `1900-1908` and `1910-1928`
     respectively) don't need the new column in their `WHEN` clauses (they key off `deleted`
     only) but `_undelete` should still include it in its full-row `json_object` payload
     (pattern at `1916-1923`, which already lists `runtime_backend`/`discovery_owner`/
     `discovery_entity_id` the same way); `_delete`'s payload is deliberately minimal
     (`1900-1908`) and needs no change.
   - `UPDATE db_schema_version SET version = 18 WHERE schema_name = 'rag_char_chat_schema' AND
     version = 17;`
3. Add method `_migrate_from_v17_to_v18(self, conn)` (copy the structure of
   `_migrate_from_v16_to_v17`, `ChaChaNotes_DB.py:2957-2981`: `conn.executescript(...)`, then
   verify `_get_db_version(conn) == 18`, raising `SchemaError` otherwise).
4. Register it in the `migration_steps` dispatch dict at `ChaChaNotes_DB.py:3054-3068`:
   add `17: self._migrate_from_v17_to_v18,`.

### Exact read/write call sites Task 13 will touch

**Write — create** (new conversation, system prompt applied before first persistence):
- `tldw_chatbook/Chat/chat_persistence_service.py:38-75` —
  `ChatPersistenceService.create_conversation()`: add a `system_prompt: Optional[str] = None`
  parameter and add `"system_prompt": system_prompt` to the `self.db.add_conversation({...})`
  dict (currently built at `63-75`).
- `tldw_chatbook/Chat/console_chat_store.py:488-503` — `persist_session_if_needed()`: add a
  `system_prompt=` kwarg to the `self.persistence.create_conversation(...)` call
  (`496-502`), sourced from the session's settings (Task 13 adds
  `ConsoleSessionSettings.system_prompt`, per the plan's Task 13 file list).
- DB layer: `add_conversation` (`ChaChaNotes_DB.py:4194-4327`) needs a
  `system_prompt = self._normalize_nullable_text(conv_data.get('system_prompt'))` line near
  `4241-4247` (reuse the existing static helper at `4023-4029`, which strips-to-`None`; note it
  `.strip()`s the string, so a caller who wants to preserve exact leading/trailing whitespace in
  a prompt should not use it as-is — Task 13's implementer should confirm this is acceptable or
  write a dedicated normalizer), plus adding `system_prompt` to the INSERT column list and
  params tuple (`4255-4272`).

**Write — update** (system prompt changed on an already-persisted conversation, e.g. via a
future `/system` command): **no existing call site exists today.** Neither
`console_chat_store.py` nor `chat_persistence_service.py` calls `update_conversation` for
anything — `console_chat_store.py` only ever calls `create_conversation` once per session
(guarded by `if session.persisted_conversation_id is not None: return`,
`console_chat_store.py:491-492`). Task 13 must:
- Add a method to `ChatPersistenceService` (e.g. `update_conversation_system_prompt`) that
  calls `self.db.update_conversation(conversation_id, {"system_prompt": system_prompt},
  expected_version=...)`.
- Add the matching column handling inside DB `update_conversation`
  (`ChaChaNotes_DB.py:4813-5030`): include `system_prompt` in the `SELECT` at `4851-4862`, a
  `if 'system_prompt' in update_data:` normalize-and-carry-forward block (pattern at
  `4919-4922` for `topic_label`), and a `fields_to_update_sql`/`params_for_set_clause` append
  (pattern at `4981-4983`).
- Extend the `ConsoleChatPersistence` Protocol (`console_chat_store.py:26-58`) with this new
  method, and add a new call site in `console_chat_store.py` (there is none today) for whenever
  the session's system prompt is applied/changed after `persisted_conversation_id` is already
  set.

**Read — resume** (restoring a saved conversation into a Console session): once the column
exists, no DB-layer change is needed for reads — `get_conversation_by_id`
(`ChaChaNotes_DB.py:4401-4442`, `SELECT * FROM conversations WHERE id = ?` at `4418`) already
returns every column as a dict, and `get_conversation_metadata`
(`tldw_chatbook/Chat/chat_conversation_service.py:405-406`) passes that dict straight through
into the `get_conversation_tree()` result's `"conversation"` key
(`chat_conversation_service.py:564-585`). Task 13 only needs to change
`tldw_chatbook/UI/Screens/chat_screen.py:2162-2246`
(`_resume_console_workspace_conversation`): read `conversation.get("system_prompt")`
(`conversation` dict built at `2206-2208`) and fold it into the `ConsoleSessionSettings`
snapshot passed as `settings=` to `store.restore_persisted_session(...)`
(`2235-2241`; currently passes `self._active_console_session_settings()` unchanged at `2240`,
which does not carry the resumed conversation's own system prompt today).

## Rationale

Every nullable TEXT column already on `conversations` was added for one specific, currently
load-bearing feature (persona/character identity, workspace scoping, kanban state, topic
tagging, clustering, import provenance, or agent runtime/discovery visibility), and both
`add_conversation` and `update_conversation` hard-code that ownership — reusing any of them for
an unrelated system prompt would silently corrupt whichever feature already owns the column the
moment both features are used on the same conversation (e.g. a workspace conversation with a
custom system prompt would have to choose between `workspace_id` meaning "workspace" and
meaning "system prompt text"), so a new `system_prompt TEXT NULL` column via an additive
migration (following the exact V12→V13/V13→V14 precedent of `ALTER TABLE` + redefined
`conversations_sync_*` triggers + a schema-version bump) is the only seam that satisfies the
qualifying rule; it also costs nothing extra to wire up because the create/update code paths
already exist as an explicit per-column allow-list in both DB methods, and the read path is
already fully generic (`SELECT *`) so restoring a saved conversation's system prompt requires
no DB change at all, only a one-line read in `chat_screen.py`'s resume handler.
