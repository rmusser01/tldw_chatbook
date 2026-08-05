# Character Authority and Conversation Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task by task. Every
> production change below starts with a failing focused test.

**Goal:** Give local and server-backed character conversations a durable,
source-aware `(source, authority_id, character_id)` identity that can later
drive exact TTS profile assignment, without changing speech admission,
assignment mutation, Sync V2, or managed audio.cpp ownership.

**Architecture:** Extend the existing owners instead of introducing a parallel
identity system. `CharactersRAGDB` exposes its existing durable local authority
and persists one new nullable conversation column. `ConfiguredServerTarget`
owns a persisted random authority scope, while
`RuntimeServerContextProvider` derives the server-user authority through its
existing authenticated client and runtime revision fence. Native Console
sessions carry those same fields and construct the existing `CharacterRef`
only when all character authority is proven.

**Tech Stack:** Python 3.12, SQLite, Pydantic v2, asyncio, Textual 8, pytest,
Ruff, mypy, Backlog.md.

**Task:** `TASK-617.2`

**Parent:** `TASK-617`

**ADR required:** yes

**ADR path:**
`backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`

**Reason:** ADR-037 already governs the local/server authority encoding,
authenticated-context fencing, one-column conversation migration,
backup/restore behavior, privacy rules, and the explicit Sync V2 exclusion.
This task implements only its approved Slice 3A.2 decision, so no second ADR is
needed.

---

## Scope boundary

This plan implements only approved Slice 3A.2:

- a narrow DB-owned accessor for the existing `local_authority_id`;
- a persisted canonical UUIDv4 `authority_scope_id` on saved server targets;
- durable first-use upgrade for legacy targets, with malformed/duplicate scope
  values failing closed for authority only;
- the exact `server-user-v1:` authority encoding and revision/auth-context
  fenced identity lookup;
- nullable `conversations.assistant_authority_id`, schema v28, local-only
  legacy backfill, normalized CRUD, and raw database backup/restore coverage;
- explicit null provenance for current import/Sync-adjacent paths that cannot
  prove authority;
- source-aware native Console session identity, persistence, resume, and
  Roleplay character handoff behavior.

It does **not** implement:

- trusted message speech snapshots or speech admission changes (Slice 3A.3);
- assignment set/replace/detach operations (Slice 3A.4);
- assigned-profile resolution, character voice controls, or automatic speech;
- Persona `CharacterRef` inheritance or Persona-specific TTS;
- `assistant_authority_id` transport in Sync V2 or portable chatbooks;
- a generic `AssistantRef` hierarchy or second identity store;
- managed audio.cpp discovery, launch, supervision, restart, or shutdown.

## Fixed contracts

The server authority frame is exact:

```text
LP(b"tldw-chatbook.character-authority")
+ LP(b"1")
+ LP(canonical_authority_scope_id_ascii)
+ LP(canonical_user_id_ascii)
```

`LP(value)` is an unsigned four-byte big-endian length followed by `value`.
The stored result is:

```text
server-user-v1:<lowercase sha256 hex digest>
```

The target scope is an exact lowercase hyphenated UUIDv4. The authenticated
user ID is an integer from `1` through `2^63 - 1`. URLs, routing `server_id`,
labels, auth method, tokens, credential fingerprints, and origins do not enter
the authority frame.

Conversation character identity is:

```text
(runtime_backend, assistant_authority_id, assistant_id)
```

- local character: numeric `character_id`, canonical decimal `assistant_id`,
  and the authority owned by the same database;
- scoped server character: null `character_id`, bounded opaque `assistant_id`,
  and encoded server-user authority;
- unscoped server/imported character: null authority and no `CharacterRef`;
- Persona/generic: null authority and no `CharacterRef`.

The current Sync V2 message envelope and portable chatbook format remain
unchanged. They never infer authority from the receiver's active source.

## Supported baseline

Use the repository's existing environment:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python --version
```

Baseline:

```text
Python 3.12.11
```

The focused pre-change suite is green:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/MCP/test_server_target_store.py \
  Tests/RuntimePolicy/test_server_context_provider.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/DB/test_chachanotes_citation_provenance_migration.py \
  Tests/Chat/test_chat_conversation_service.py
```

Expected baseline:

```text
257 passed
```

---

## Task 1: Persist target authority scope without changing server routing

**Files:**

- Modify: `tldw_chatbook/MCP/unified_control_models.py`
- Modify: `tldw_chatbook/MCP/server_target_store.py`
- Modify: `Tests/MCP/test_server_target_store.py`

- [x] Add failing tests that prove:
  - newly bootstrapped configured targets receive a canonical UUIDv4 scope;
  - JSON round-trip preserves the scope;
  - target representations never expose the raw scope;
  - legacy missing scopes are generated, atomically saved, reloaded, and only
    then returned;
  - existing scope survives mutable label/URL/auth/status updates and legacy
    config upsert;
  - malformed or duplicate scopes fail authority acquisition without making
    ordinary target loading unusable;
  - a write/reload failure returns no ephemeral scope.
- [x] Add `authority_scope_id` to `ConfiguredServerTarget` wire persistence
  with `repr=False`. Deserialization alone may represent a legacy missing
  scope as null so ordinary server routing still loads; every newly created
  target receives a canonical UUIDv4 before its first persistence.
- [x] Add one store-owned `ensure_authority_scope_id(server_id)` admission
  method. Serialize same-process upgrades, validate all non-null scope values,
  persist via the existing temporary-file replacement, reload, and verify
  exact durability before returning.
- [x] Preserve an existing scope in `upsert_legacy_config_target`; assign a
  scope when a genuinely new legacy-config-backed target is created.
- [x] Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/MCP/test_server_target_store.py
```

## Task 2: Resolve exact server-user character authority with context fencing

**Files:**

- Modify: `tldw_chatbook/runtime_policy/server_context.py`
- Modify: `Tests/RuntimePolicy/test_server_context_provider.py`

- [x] Add failing tests for the exact framing/digest vector and validation
  bounds.
- [x] Add failing async tests proving the resolver:
  - requires the expected configured-target ID carried by the caller and
    rejects a mismatch before identity network I/O;
  - calls `get_current_user_profile(sections="identity")`;
  - extracts only a valid positive `user.id`;
  - remains stable across mutable target metadata and credential rotation when
    the target scope and returned user are unchanged;
  - separates two users on the same target;
  - caches only inside one matching runtime/authenticated client context;
  - rejects an in-flight response after runtime revision, target, account, or
    credential/client context changes;
  - reports bounded `server_identity_unavailable` failure for missing,
    malformed, ambiguous, or unavailable identity while leaving ordinary
    `build_client()`/server text operations usable.
- [x] Implement the encoder as a small pure helper and add one async
  `RuntimeServerContextProvider` authority resolver that accepts an expected
  configured-target ID. Before network I/O, capture and validate that exact
  target, its persisted scope, the bound client/authentication key, and the
  runtime revision. Revalidate all four after the identity response before
  caching or returning. Reuse `RuntimePolicyContext.snapshot()` and the
  existing client cache key; do not create another runtime context service.
- [x] Clear the one-entry authority cache wherever the existing authenticated
  client context is invalidated. Never log the scope, encoded authority,
  endpoint response, token, origin, or routing ID.
- [x] Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/RuntimePolicy/test_server_context_provider.py
```

## Task 3: Add schema v28 and normalized conversation provenance

**Files:**

- Add:
  `tldw_chatbook/DB/migrations/chachanotes_v27_to_v28_character_authority.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Add: `Tests/DB/test_chachanotes_character_authority_migration.py`
- Modify: `Tests/DB/test_chachanotes_active_leaf_migration.py`
- Modify: `Tests/DB/test_chachanotes_context_summary_migration.py`
- Modify: `Tests/DB/test_chachanotes_citation_provenance_migration.py`
- Modify: `Tests/ChaChaNotesDB/test_message_generation_metadata.py`

- [x] Add failing migration tests that prove:
  - the DB-owned local authority accessor returns one stable value across
    close/reopen;
  - v27→v28 adds only nullable `assistant_authority_id`;
  - eligible legacy local character rows are backfilled from that database's
    authority;
  - legacy server, Persona, and generic rows remain authority-null;
  - a failed migration rolls back the column/version/backfill together.
- [x] Add failing CRUD validation tests that prove:
  - new local character rows use canonical decimal identity and the local
    authority;
  - server character rows accept bounded opaque `assistant_id`, keep
    `character_id` null, and may remain explicitly authority-null;
  - Persona/generic rows cannot retain character authority;
  - create/read/update and list normalization round-trip the new field.
- [x] Advance `CharactersRAGDB` to schema v28, add the migration step, and add
  the narrow `get_local_authority_id()` accessor.
- [x] Normalize runtime source and assistant fields jointly. Treat an omitted
  local authority on an app-owned local-character create as provable from the
  same DB; preserve an explicitly supplied null for unproven import material.
- [x] Include the field in application-owned conversation CRUD while leaving
  existing Sync V2 trigger/envelope payloads unchanged.
- [x] Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_chachanotes_character_authority_migration.py \
  Tests/DB/test_chachanotes_citation_provenance_migration.py
```

## Task 4: Carry provenance through services, backup, and unproven import

**Files:**

- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_importer.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`
- Modify: `Tests/Chat/test_chat_conversation_service.py`
- Modify: `Tests/Chatbooks/test_chatbook_importer.py`
- Modify: `Tests/Sync_Interop/test_chat_outbox_producer.py`
- Modify: `Tests/DB/test_core_sqlite_owner_privacy.py`

- [x] Add failing service tests that pass and normalize
  `assistant_authority_id` without changing the server conversation wire
  contract.
- [x] Add a failing portable-chatbook import test showing imported character
  identity is explicitly authority-null rather than rebound to the receiving
  DB merely because a numeric character ID exists.
- [x] Add a focused Sync V2 regression using a local conversation row that
  contains `assistant_authority_id`; prove the existing message envelope
  neither transports that conversation field nor infers any authority from
  active runtime state. Production Sync V2 code remains unchanged.
- [x] Add a failing SQLite online-backup round-trip test showing the new column
  and value survive application-owned backup/reopen.
- [x] Thread the field through the local persistence facade and normalized
  local conversation rows. Explicitly pass null at the unproven chatbook
  import boundary. Do not add the field to chatbook export or Sync V2.
- [x] Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Sync_Interop/test_chat_outbox_producer.py \
  Tests/DB/test_core_sqlite_owner_privacy.py
```

## Task 5: Make native Console session identity source-aware

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [x] Add failing store tests for local, scoped-server, unscoped-server,
  Persona, and generic sessions. Only the two complete character cases may
  return the existing `CharacterRef`.
- [x] Add failing persistence and state round-trip tests for
  `runtime_backend`, `assistant_kind`, `assistant_id`, and
  `assistant_authority_id`, retaining `character_id` only as the local numeric
  compatibility projection.
- [x] Add failing persisted-resume tests showing server opaque IDs restore
  without consulting or inheriting the active server, and authority-null
  records stay unscoped.
- [x] Extend `ConsoleChatSession` and its create/restore/persist seams with the
  four identity fields and a narrow `character_ref()` projection.
- [x] Key the direct/plain-provider character gate on trusted
  `assistant_kind == "character"` rather than local-only `character_id`, while
  keeping local avatar/dictionary lookups numeric and local.
- [x] Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_chat_store.py \
  Tests/UI/test_console_native_chat_flow.py
```

## Task 6: Produce source-aware Roleplay character handoffs

**Files:**

- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [x] Add failing workbench tests showing Start Chat emits the selected
  runtime source and expected server target instead of hard-coded local
  metadata.
- [x] Add failing Console handoff tests showing:
  - local character authority comes only from the character DB accessor;
  - server authority comes only from the revision-fenced resolver for the
    exact target carried by the handoff;
  - identity endpoint failure still creates an explicitly unscoped server
    character session that can perform ordinary text chat;
  - a target mismatch never fetches the same ID from a different active
    server or assigns that server's authority;
  - Persona handoffs never produce a `CharacterRef`.
- [x] Make the existing Personas handoff source-aware. Resolve character card
  detail through the existing local/server scope service. The current server
  detail wire DTO remains numeric; opaque server character IDs are accepted
  at persisted/session identity boundaries and do not require widening that
  established API client contract in this slice.
- [x] Populate the Console session identity before seeding/persisting the
  greeting. Do not invoke profile assignment or alter TTS selection.
- [x] Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_personas_workbench.py \
  Tests/UI/test_console_native_chat_flow.py
```

## Task 7: Focused regression, static checks, and task closeout

**Files:**

- Modify:
  `backlog/tasks/task-617.2 - Establish-character-authority-and-conversation-provenance.md`
- Modify this plan only if implementation deviations must be documented

- [x] Run the union of every focused test file changed by Tasks 1–6.
- [x] Run existing profile and external audio.cpp regressions to prove this
  slice did not change selection or synthesis:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/TTS/test_profile_types.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/UI/test_stts_playground_audio_cpp.py
```

- [x] Run static checks only on changed Python files:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check <changed-python-files>
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy <changed-python-files>
git diff --check
```

- [x] Run the repository's required broader gate and distinguish inherited
  `dev` failures from task regressions; do not fix unrelated baseline debt in
  this PR.
- [x] Self-review the diff against every TASK-617.2 acceptance criterion,
  ADR-037 privacy rules, and the explicit exclusions above.
- [x] Check all acceptance criteria, add concise implementation notes with
  commands/results and any documented plan deviation, then set TASK-617.2 to
  Done only when the repository Definition of Done is actually satisfied.

## Implementation deviations

- Review of Task 6 exposed shared-pane ownership, stale callback, cancellation,
  and authentication-context races that could publish a character under the
  wrong source. The handoff files and tests were hardened within the approved
  provenance contract; no TTS selection, speech, assignment, or managed-server
  behavior was added.
- Final whole-slice review found the legacy Tavern/SillyTavern history importer
  outside the originally listed portable-chatbook boundary. It now explicitly
  persists null authority, with a real schema-v28 import/reopen regression that
  proves a same-named local card cannot yield a `CharacterRef`.
- The repository-wide gate could not collect because current `origin/dev`
  carries the same `StreamDone` import error. The exact five focused failures,
  Ruff diagnostics, and production mypy diagnostics were replayed against
  `origin/dev` and documented in the Backlog task rather than repaired here.
