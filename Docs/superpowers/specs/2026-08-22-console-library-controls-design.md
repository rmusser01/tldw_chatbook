# Console Library controls: per-conversation retrieval and assistant access

**Status:** Architecture approved; review corrections incorporated on 2026-08-22

**Task:** [TASK-19900](../../../backlog/tasks/task-19900%20-%20Make-Console-Library-controls-explicit-per-conversation.md)

**Decision:** [ADR-079](../../../backlog/decisions/079-console-library-conversation-authority.md)

**Plan:** [Implementation plan](../plans/2026-08-22-console-library-controls.md)

**PR:** [#1933](https://github.com/rmusser01/tldw_chatbook/pull/1933)

## Summary

Console can consult a user's Library in three ways, owned by three different
actors:

| Mechanism | Actor | Authority |
| --- | --- | --- |
| **Search Library** | User | Explicit action, always available |
| **Automatic retrieval** | Application | Per-conversation Never / Automatic |
| **Library tools** | Assistant | Per-conversation Blocked / Allowed |

The current interface blurs those mechanisms. A status chip reports staged
evidence, a global switch retrieves before every send, and the assistant gets
a built-in Library provider whenever agent tools run. The global
`direct_library_tools` value is also easy to misread: `false` does not disable
Library access; ADR-030 defines it as the RAG fallback.

This design gives automatic retrieval and assistant access independent,
device-local per-conversation controls. Manual search remains reachable in all
four combinations. The existing global Direct/RAG selector stays a selector.
Assistant-initiated reads become reviewable as minimized local activity, never
as staged evidence for a future prompt.

## Goals

- Make it obvious whether the application will retrieve before the next send
  and whether the assistant may initiate Library reads during the turn.
- Preserve established behavior for conversations that predate the upgrade,
  while shipping privacy-tight defaults for new local work.
- Keep a device's model-access policy out of synchronized conversation data.
- Ensure a blocked or unreadable policy cannot be bypassed through built-in,
  Skill, MCP, queued-turn, or subagent composition.
- Give automatic retrieval a truthful pre-dispatch failure flow rather than
  silently sending an ungrounded request.
- Let users review assistant Library operations without retaining duplicate
  source bodies or feeding activity back to the model.
- Separate policy editing, manual search, staged/cited evidence, and activity
  review so each surface has one job.

## Non-goals

- A per-call approval card for every built-in Library read. The conversation
  policy is the authorization boundary; MCP and other local tools keep their
  existing independent approval systems.
- Per-conversation selection among Direct and RAG. The existing global
  `direct_library_tools` selector remains authoritative when access is Allowed.
- Configurable automatic-retrieval source categories. Automatic retrieval has
  one fixed, predictable category set in this release.
- Synchronizing or exporting the policy as part of a Chatbook/conversation.
- Replacing canonical CitationTrace work from ADR-024.
- Reworking Library retrieval ranking, indexing, or source-resolution policy.
- Removing the dead legacy `get_rag_context_for_chat` helper.

## Delivery decomposition

The design is one coherent authority model, but implementation is too broad
for one atomic PR. Delivery is dependency-ordered so each review has one
primary failure surface:

| Task | Deliverable |
| --- | --- |
| TASK-19900.1 | Device-local policy/checkpoint schema, transactional legacy seed, Sync-v1/v2 message-state compatibility, repository CAS/coordinator, and lifecycle |
| TASK-19900.2 | Immutable turn authority, provider absence/selection, and permanent name reservation |
| TASK-19900.3 | Fixed-category automatic retrieval and pause/recovery send gate |
| TASK-19900.4 | Two-axis chip, policy/search split, source density, responsive/focus behavior |
| TASK-19900.5 | Minimized activity capture, projection, Inspector, and export redaction |
| TASK-19900.6 | Documentation, integrated targeted gates, mutation checks, and live qualification |

No child task is marked In Progress until the written spec and detailed
implementation plan are approved. Dependencies point only backward; storage
lands before runtime/UI, and qualification lands after every product slice.

## 1. User model

### 1.1 Three mechanisms, not one mode

Manual search is not a state. It is an explicit user action and remains
available whether automatic retrieval is Never or Automatic and whether the
assistant is Blocked or Allowed.

The two policy axes produce four useful states:

| Automatic retrieval | Assistant access | What happens |
| --- | --- | --- |
| Never | Blocked | Only the user can run Search Library. |
| Automatic | Blocked | The app prepares evidence before eligible sends; the assistant cannot initiate Library reads. |
| Never | Allowed | No pre-send retrieval; the assistant may use the selected Direct/RAG provider during its turn. |
| Automatic | Allowed | The app may stage evidence before dispatch and the assistant may make additional Library reads during the turn. |

This is deliberately not an Off/Manual/Auto enum. Such an enum cannot express
the middle two combinations and makes “Manual” sound unavailable in other
states.

### 1.2 Shipped and global defaults

Shipped defaults for a newly created local Console session are:

- Automatic retrieval: **Never**.
- Assistant Library access: **Blocked**.

Canonical Settings may change the defaults used for future local sessions:

- Existing `[chat_defaults].rag_auto_retrieve_on_send`, shipped `false`, is
  relabeled and treated as the new-session automatic-retrieval default.
- New `[console].assistant_library_access_default`, shipped `false`, is the
  new-session assistant-access default.

The canonical F9 Settings surface owns both future-session defaults and labels
them **New Console conversations**. Its copy states that changing either value
does not rewrite open sessions or saved conversations. The configuration
template, settings schema/validation, load/save path, and focused Settings tests
ship with the corresponding policy/UI tasks; no deprecated Settings surface is
updated.

The session captures both once when it is created. Later Settings changes do
not rewrite an open session or persisted conversation. The existing
`[console].direct_library_tools` setting is different: it is a live global
Direct-versus-RAG selector captured into each Allowed turn, not a policy
default and not an enable switch.

### 1.3 Existing and externally arrived conversations

- A conversation already present when the database upgrades is seeded once,
  inside that migration transaction, to its prior effective behavior: the
  supplied global automatic value and assistant access Allowed.
- A conversation inserted later by sync/import has no device-local authority.
  A missing row therefore means Never and Blocked, irrespective of global
  defaults.
- Opening a missing-row synced conversation does not insert a policy row until
  the user explicitly saves policy on this device. Safe defaults can remain a
  read-only effective value.

## 2. Ownership and components

The implementation adds narrow units instead of teaching the existing RAG
modal, conversation metadata helper, and tool registry to share ownership.

### 2.1 Policy model and repository

`Chat/console_library_policy.py` owns immutable values and validation:

- `ConsoleConversationLibraryPolicy`
- `ConsoleLibraryPolicySnapshot`
- `ConsoleLibraryPolicyDefaults`
- policy normalization and safe missing/error outcomes

`Chat/console_library_policy_repository.py` owns database reads, legacy
seeding, insert/update CAS, and deletion behavior. It has no Textual or provider
imports. Reads distinguish:

- row found and valid;
- row absent, effective safe defaults;
- unreadable/corrupt/error, effective safe defaults plus an explicit error.

The repository never returns a valid-looking Allowed policy from an exception.

### 2.2 Policy coordinator and session holder

An app/store-owned `ConsoleLibraryPolicyCoordinator` is the single in-process
authority for durable policy revisions. It:

- performs repository reads and writes off the Textual event loop;
- publishes every committed CAS result to all live sessions bound to the same
  durable conversation;
- re-reads durable policy at the execution linearization point so a second app
  process or stale tab cannot retain revoked authority;
- returns an explicit unavailable result on a read error rather than falling
  back to a cached Allowed value.

The execution read and immutable capture are one logical operation. A policy
commit after that point applies to the next executed turn, matching the
running-turn immutability rule. Subagents never perform another policy read.

`ConsoleChatSession` receives a dedicated policy holder containing:

- effective values;
- persisted revision when one exists;
- whether the user explicitly staged a choice;
- source/error state;
- whether a durable save is pending.

A new local session captures global defaults into this holder. A policy edit
on an empty tab does not create a conversation, but it makes the session
non-untouched so navigation/close behavior cannot silently discard an explicit
choice. Merely holding untouched shipped defaults does not force persistence.

### 2.3 Admission authority and resolved destination

Provider selection and provider destination resolve at different times, so the
design uses two detached immutable records instead of claiming a destination
before the gateway knows it.

`ConsoleTurnLibraryAuthority` is captured at actual execution, after the
coordinator's durable read. It contains:

- automatic-retrieval and assistant-access policy;
- policy revision/source state;
- selected Direct/RAG mode;
- fixed automatic source categories;
- relevant item-scope snapshot;
- selected provider/model/endpoint intent;
- activity attempt/run identifiers.

After `provider_gateway.resolve_for_send` applies configuration fallback,
endpoint normalization, and readiness checks, it produces an immutable
`ConsoleResolvedDestination` containing the resolved provider, model,
credential-free effective endpoint identity, and egress class. The supported
classes are `on_device`, `private_network`, `public_network`, and `unknown`.
Unknown/custom destinations are disclosed as external/unknown, never guessed
to be on-device from provider name or API-key presence.

The final `ConsoleTurnExecutionContext` combines those two records before
automatic preparation or agent-provider composition. That combined context is
the only runtime input for automatic admission and built-in Library-provider
composition.

### 2.4 Sidecar projections and dispatch recovery

Assistant activity reuses `message_trajectory_metadata` for storage but not
`derive_trajectory` for presentation. A pure
`Chat/library_activity.py` projection filters and groups `library_activity`
events by durable turn, branch, attempt, and actor. It has no Textual or DB
dependency.

`derive_trajectory` must explicitly treat `library_activity` as sidecar-only:
it is neither a message-owned row nor an ordinary nested trajectory record.
This prevents the event from displacing an anchor's timing or appearing twice.

Automatic zero-match and one-shot-bypass disclosures use a separate bounded
`library_preparation` sidecar event, also excluded from generic trajectory
ownership. `Chat/library_preparation.py` projects it onto the durable sent turn.
It stores outcome, attempt ID, result count, and fixed source categories only;
it never stores the query, titles, source IDs, snippets, or bodies.

Every accepted **durable manual or queued user-text turn** also creates a
temporary, device-local row in the dedicated
`console_dispatch_checkpoints` table, anchored to its empty assistant recovery
owner. Ephemeral and machine-origin turns do not write this table. This
operational row is not trajectory history: no
trajectory projection or ADR-067 exporter reads the table, it has no sync or
import mapping, and it is atomically deleted when the assistant reaches a
durable terminal state. A stored revision and state columns own its `accepted`
and `dispatch_started` CAS transitions. Its bounded strict-schema payloads
contain preparation/attempt identity, frozen authority, credential-free
resolved destination identity, origin/queue-entry identity, and
presence/opaque-reference flags needed to decide whether recovery is
possible—never the draft, prefill text, evidence bodies/snippets, attachment
bytes, credentials, or a serialized provider request.

## 3. Device-local persistence

### 3.1 Tables

The current ChaChaNotes schema is v44. Implementation is expected to add a
v44→v45 migration after re-checking the schema head immediately before coding.

```sql
CREATE TABLE console_conversation_library_policy (
    conversation_id TEXT PRIMARY KEY
        REFERENCES conversations(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    schema_version INTEGER NOT NULL DEFAULT 1
        CHECK(schema_version > 0),
    auto_retrieve_on_send INTEGER NOT NULL DEFAULT 0
        CHECK(auto_retrieve_on_send IN (0, 1)),
    assistant_library_access INTEGER NOT NULL DEFAULT 0
        CHECK(assistant_library_access IN (0, 1)),
    policy_revision INTEGER NOT NULL DEFAULT 1
        CHECK(policy_revision > 0),
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE messages ADD COLUMN assistant_generation_state TEXT
    CHECK(
        assistant_generation_state IS NULL OR
        assistant_generation_state IN (
            'accepted', 'dispatch_started', 'continuation_active',
            'complete', 'stopped', 'failed', 'discarded'
        )
    );

CREATE TABLE console_dispatch_checkpoints (
    assistant_message_id TEXT PRIMARY KEY
        REFERENCES messages(id) ON DELETE CASCADE,
    user_message_id TEXT NOT NULL
        REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE CASCADE,
    schema_version INTEGER NOT NULL DEFAULT 1
        CHECK(schema_version > 0),
    preparation_id TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL,
    state TEXT NOT NULL
        CHECK(state IN ('accepted', 'dispatch_started')),
    checkpoint_revision INTEGER NOT NULL DEFAULT 1
        CHECK(checkpoint_revision > 0),
    user_message_version INTEGER NOT NULL
        CHECK(user_message_version > 0),
    assistant_message_version INTEGER NOT NULL
        CHECK(assistant_message_version > 0),
    origin TEXT NOT NULL CHECK(origin IN ('manual', 'queued')),
    queue_entry_id TEXT,
    frozen_authority_json TEXT NOT NULL,
    resolved_destination_json TEXT NOT NULL,
    reconstructability_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_console_dispatch_checkpoint_conversation
    ON console_dispatch_checkpoints(conversation_id);
```

Both tables use device-local persistence:

- no sync columns or triggers;
- no conversation-metadata mirror;
- no search/FTS/index projection;
- no Chatbook export/import mapping;
- policy is retained while a conversation is soft-deleted and cascade-deleted
  only when the local conversation row is permanently purged;
- a dispatch checkpoint exists only while its assistant owner is nonterminal
  and cascades with that owner or conversation.

`messages.assistant_generation_state` is not device-local. It participates in
the message row's existing version/hash, Sync v1/v2, `.chatbook`, active-path
JSON, and deletion/conflict contracts. `NULL` preserves historical semantics
(historical assistant rows are treated as complete **unless** canonical
`provider_continuation_json` is active; non-assistant rows must remain `NULL`).
A valid active ADR-063 continuation is authoritative regardless of NULL or a
stale new-state value. The loader exposes continuation recovery first and then
lazily normalizes the field to `continuation_active` under expected message
version/`deleted = 0`, recording ordinary sync intent. Recovery actions remain
disabled while that normalization result is unresolved. A successful write
must return the committed message version/hash and atomically rebind the
in-memory ADR-063 recovery handle before Resume or Discard is enabled. A known
rolled-back write failure may leave recovery available with its original
version only after a fresh read confirms the same valid continuation and
`deleted = 0`; a CAS miss always re-reads and reconciles. A missing/deleted row,
changed continuation identity, or invalid continuation hides the stale actions
and is quarantined rather than invoked. If the same valid continuation remains
at a newer version, the loader refreshes the handle and retries/binds under
that observed version before enabling actions. New durable assistant owners use
the closed vocabulary above.
Text/Markdown renderers turn an empty terminal or unresolved owner into bounded
literal status copy instead of exporting a blank row.

`ConsoleDispatchCheckpoint` is a strict typed model. Its three JSON columns use
canonical encoders, exact keys/types, and small byte caps; the repository also
verifies that the USER and assistant roles and conversation ownership match in
the same transaction. This avoids an unbounded or cross-conversation recovery
payload without requiring a provider request body.

### 3.2 Transactional legacy seed

The migration module does not read TOML or application config. Every production
entry point that can open and migrate `CharactersRAGDB` supplies one sanitized
`ConsoleLibraryMigrationSeed` containing the effective pre-upgrade global
automatic-retrieval boolean. Fresh databases do not require a legacy seed. If
a v44 database reaches the v44→v45 step without a valid seed, migration raises
`SchemaError` before DDL or version advancement instead of guessing.

One config-layer helper resolves and validates that seed; database openers call
the helper rather than duplicating default/coercion logic. The DB constructor
accepts only the resulting typed value and remains config-independent.

The migration-capable schema initializer must acquire `BEGIN IMMEDIATE`
**before its first schema-version read**. The current deferred outer
transaction cannot be upgraded by a nested migration block, so this task
changes that runner boundary rather than placing `BEGIN IMMEDIATE` only inside
the v44→v45 function. The lock is held only for schema inspection/migration and
is committed promptly when the database is already current.

Inside that same transaction, the v44→v45 step:

- adds the nullable message state, creates both tables and the checkpoint index;
- replaces all four final message Sync-v1 create, update, delete, and undelete
  trigger definitions so payloads carry the closed state and the update
  trigger watches it; the historical v4 bootstrap schema remains unchanged,
  because fresh databases traverse migrations and receive these definitions in
  v45 after the column exists;
- inserts a final policy row for every active or soft-deleted conversation
  present in that transaction;
- writes the supplied automatic value and `assistant_library_access = 1`,
  matching the previously always-composed Library provider;
- advances the schema version.

The same foundation delivery updates Sync-v2's message record, source reader,
and envelope/proof normalization before any changed Sync-v1 intent can reach
the outbox. New payloads preserve exact source-proof equality with
`assistant_generation_state` present, including explicit `NULL` for ordinary
messages. Older Sync-v1/v2 records that legitimately lack the field normalize
that one missing key to `NULL` before the same equality/proof checks; unknown,
malformed, or mismatched fields remain rejected. This compatibility seam ships
with v45 in TASK-19900.1 rather than waiting for the runtime send-gate task.

A crash or statement failure rolls the whole step back, including the message
column, both tables, the index, and version change. Concurrent processes
serialize on SQLite's write lock; the first successful migrator's seed is the
single upgrade value and
later openers observe v45 without reseeding. A conversation synced or imported
after the migration transaction receives no policy row and cannot be captured
by a later broad backfill. Migration tests exercise missing/invalid seed, two
competing openers, crash rollback, retry with a different supplied value, and
the pre-read locking behavior for fresh, already-current, v44, and older-schema
opens.

### 3.3 Reads and writes

Policy edits use optimistic compare-and-swap:

- update `WHERE conversation_id = ? AND policy_revision = ?`;
- create a previously absent local row with a conditional insert whose unique
  conflict is reported as a revision conflict, never silently overwritten;
- increment revision only on success;
- distinguish missing row, stale revision, corrupt row, and database error;
- never publish the candidate in session state before the durable write
  succeeds.

The policy modal remains open on failure. A stale revision offers Reload and
Compare/Retry; a missing conversation reports that it no longer exists. No
path displays Saved until the repository confirms the committed revision.

The same foundation repository owns explicit dispatch-checkpoint primitives:

- `insert_with_messages(cursor, ...)` writes the USER, empty assistant owner,
  and `accepted` checkpoint through the caller's transaction;
- `read_for_session(...)` validates and returns at most one active-path owner;
- `cas_state(..., expected_state, expected_revision, ...)` updates state,
  attempt, checkpoint revision, stored message versions, and matching assistant
  generation state with conditional USER/assistant `messages.version` and
  `deleted = 0` guards;
- `settle_with_assistant(..., expected_state, expected_revision,
  expected_user_version, expected_assistant_version, ...)` writes the assistant terminal state,
  content/metadata, message version/hash/sync intent, and deletes the checkpoint
  in one transaction;
- `handoff_to_provider_continuation(...)` atomically writes ADR-063's validated
  `provider_continuation_json`, sets `continuation_active`, bumps message
  version/hash/sync intent, and deletes the expected dispatch checkpoint before
  any tool executes;
- no generic upsert may overwrite state or revision.

### 3.4 First persistence and temporary sessions

An empty durable-capable tab stages policy in memory. On first conversation
creation for a **new local session**, the holder's captured policy is inserted
in the same transaction as the conversation row even when the user did not
edit it; otherwise later resume would mistake the row for externally arrived
missing policy and lose the captured global defaults. Opening an already
persisted synced/imported conversation with a missing policy row remains
write-free until an explicit policy save. Failure rolls back first conversation
creation rather than creating a row that can briefly execute under
missing-policy semantics. The transaction may stage a candidate conversation
ID and auto-title, but it does not mutate `session.persisted_conversation_id`
or the session title until commit succeeds; failure leaves both at their exact
pre-send values.

Ephemeral sessions retain policy, preparation, dispatch recovery, and assistant
activity in memory and never create a checkpoint-table row. Their in-memory
dispatch analogue uses the same states/actions but does not claim crash
durability. Promotion is disabled while an ephemeral preparation or dispatch
analogue is `preparing`, paused, `committing`, `accepted`, or
`dispatch_started`; the Save action explains `Finish or discard the pending
turn before saving.` Once no unresolved owner remains, promotion is one
transaction covering:

- the conversation row;
- the policy row;
- promoted messages and active lineage;
- queued `library_activity` and `library_preparation` rows.

Any failure rolls the complete promotion back and restores the session's
ephemeral identity, policy holder, messages, activity, and retryability. It
must not leave a non-ephemeral session pointing at a partial durable bundle.

Temporary sessions may use Assistant Allowed only for a built-in Library
provider authenticated by the runtime and only for the exact permanent
read-only reserved-name set. The ephemeral gate does not add `"library"` as a
broad audited source: an unknown future Library name or a third-party provider
claiming that source remains blocked.

### 3.5 Deletion, restore, and lifecycle

Soft deletion sets the existing conversation tombstone and makes policy and
activity inert but retains both for Undo/restore. Restoring the conversation
restores the same local policy and sidecars. A synchronized tombstone follows
the same rule. Only permanent local deletion of the conversation row cascades
policy and trajectory sidecars.

| Lifecycle event | Policy/activity result |
| --- | --- |
| New local session | Capture current future-session defaults in memory. |
| First durable send | Insert conversation, policy, sent message, and any preparation disclosure atomically; publish session ID/title only after commit. |
| v44→v45 upgrade | Seed every then-existing active/soft-deleted conversation in the migration transaction. |
| Later sync/import arrival | No policy row; effective Never/Blocked without a write. |
| Durable immediate/queued execution | Re-read through the coordinator, then freeze authority at execution. |
| Concurrent policy save | CAS winner publishes its committed revision to all same-process holders. |
| Temporary execution | Holder authority and dispatch recovery remain in memory only; exact audited read-only Library names may run when Allowed. |
| Temporary promotion | Refuse while preparation/dispatch recovery is unresolved; otherwise persist conversation, policy, lineage, completed preparation, and activity atomically. |
| Soft delete / restore | Retain and later resume the same policy and sidecars. |
| Permanent purge | Foreign-key cascade removes policy and trajectory sidecars. |
| Chatbook export/import | Policy and operational dispatch checkpoint excluded; activity/preparation trajectory export follows bounded redaction and import is inert. |

## 4. Runtime policy

### 4.1 Execution linearization and immutability

Turn preparation has two explicit capture points:

- an immediate send performs its durable policy read after send admission;
- a queued send performs that read after dequeue, so it sees changes made
  before it actually runs;
- the coordinator read plus `ConsoleTurnLibraryAuthority` construction is the
  policy linearization point;
- gateway resolution then freezes `ConsoleResolvedDestination` and produces
  the final execution context before retrieval or provider composition;
- a running/preparing turn never observes later policy, selector, provider,
  scope, or Settings changes;
- every subagent spawned by the turn inherits the same final context rather
  than resolving the conversation or destination again.

The primary and subagents cannot independently widen Library authority. A
subagent's own narrower allow-list may still remove Library tools.

### 4.2 Four combinations

At execution, the snapshot drives two separate gates:

1. `auto_retrieve_on_send` controls only the pre-dispatch retrieval stage.
2. `assistant_library_access` controls only built-in Library-provider
   composition.

Neither gate infers from the other. Manual search is outside both gates.

### 4.3 Built-in provider composition

When assistant access is Blocked, `_library_provider_for_context` returns no
provider. No Library schema or callable is visible to the model, primary
agent, or subagents.

When Allowed:

- `direct_library_tools = true` composes `LibraryToolProvider`, with ADR-030's
  18 list/get/search tools over Media, Notes, Prompts, Skills, Conversations,
  and Collections.
- `direct_library_tools = false` composes `LibraryRagToolProvider`, exactly
  `search_library_rag`, over Notes, Media, and Conversations.

The selector is captured into the turn. Changing Settings during a run affects
the next executed turn.

For a temporary conversation, composition also supplies an authenticated
built-in-Library marker to the ephemeral call gate. The gate admits only names
in the permanent reserved set and only when the turn authority says Allowed.
Adding `source="library"` alone never grants temporary-session execution.

### 4.4 Permanent name reservation

Export one canonical immutable reserved-name set derived from the union of:

- the keys of `Library.library_tool_contract.LIBRARY_TOOL_DESCRIPTORS`;
- `Agents.library_rag_tool_provider.RAG_TOOL_NAME`.

Skill and MCP collision filtering uses this set in all conversations and both
selector modes, even when no Library provider is registered. Registration
order or a dormant policy can never allow a third-party tool to claim a
built-in identity.

An inventory ratchet asserts that the descriptor registry still contains 18
names and the union still contains 19. A descriptor addition therefore fails
review until its reservation and ephemeral-read audit are explicitly accepted;
no second hand-maintained list can drift silently.

This reservation does not grant access and does not suppress unrelated MCP
Library tools with different names outside the existing ADR-030 overlap list.
MCP policy remains owned by its server/profile principal. Workspace/file tools
remain owned by `local:__local__` under ADR-032.

### 4.5 Provider egress disclosure

Allowed means Library results may enter the active model's request history.
Before a send, the policy modal labels provider/model/endpoint as selected
intent and states that the effective destination resolves on send. After the
gateway resolves, the expanded runtime line and any preparation card use only
`ConsoleResolvedDestination`: on this device, private network, public network,
or external/unknown. No API-key heuristic or provider-name allowlist may label
a destination on-device.

If either policy axis can place Library data in a model request—Automatic
retrieval or Assistant Allowed—and the resolved destination changes from
`on_device` to any other class, Console updates the expanded runtime line and
shows a persistent, non-blocking inline disclosure before dispatch. This also
covers Automatic + Blocked, where application-retrieved evidence can still
leave the device. Here persistent means an inline state that survives ordinary
repaint/navigation for that live session until the send settles or the
destination changes again, not a toast and not a new synced acknowledgment
record. The policy is preserved; the provider change is not silently
interpreted as consent revocation or renewal.

## 5. Manual Search Library

Manual search remains available from the composer/actions regardless of both
policy axes and regardless of assistant tool availability.

The manual surface owns:

- query;
- source-category toggles supported by the current Library Search/RAG seam;
- Search action;
- current item-scope summary;
- result/recovery state.

The query is always prefilled with the exact composer draft. Remove
`_console_draft_looks_like_rag_query`; no heuristic decides whether user text
looks like a query. The user may edit the search query without changing the
draft.

Source toggles are labeled **This search only**. They affect that manual run
and may be retained as harmless search-surface state, but they never become
automatic-retrieval policy. Manual results continue through the existing
staged-evidence bundle and are visible as `Sources — next send`.

## 6. Automatic pre-send retrieval

### 6.1 Admission

Automatic retrieval runs only when all are true:

- the executing turn authority says Automatic;
- the send is eligible plain user text under the current send contract;
- no explicit evidence bundle is already staged for this send;
- policy and RAG readiness are readable enough to execute;
- the send has not chosen the one-shot bypass.

Commands, approvals, regenerations, and other existing excluded send kinds
remain excluded. Explicit staged evidence wins because it expresses a more
specific user choice and avoids duplicate retrieval/cost.

### 6.2 Query and source categories

The executed draft is the query. The category set is fixed:

- Notes
- Media
- Conversations

Automatic retrieval never reads the manual search modal's source toggles.

The existing resolved conversation/workspace item scope still narrows exact
Note and Media eligibility. Under current scope semantics, an active item
scope excludes Conversations. The UI discloses this as runtime scope detail;
it does not mutate the fixed category policy.

### 6.3 Pre-dispatch lifecycle

The app-owned `ConsoleRuntime`'s `ConsoleChatStore` owns one in-memory
`ConsoleTurnPreparation` per affected session; the controller owns transitions
and the screen only projects them.
Every admitted immediate or queued text turn uses its commit substate so first
persistence cannot bypass the policy/USER atomicity rules. An Automatic turn
also traverses the retrieval preparation and pause states; a Never turn enters
`ready` after authority/destination resolution without displaying or running
Library retrieval.
The record survives screen replacement/navigation and contains a preparation
ID, current attempt ID, session/origin/queue-entry identity, executed draft,
authority and resolved destination, transient echo ID, staged attachment and
evidence identities, one-shot prefill identity, queue/generation authority,
the pre-send title state, and the ordinary-session persistence identity. It
contains no durable secret or provider request body.

The state machine is:

```text
preparing -> ready -> committing -> accepted -> dispatch_started -> dispatched
          -> paused(retrieval) -> preparing  (Retry)
                               -> ready      (Send once without Library)
          -> cancelled
committing -> paused(persistence) -> committing  (Retry)
accepted -> settled  (Discard)
dispatch_started -> settled | dispatch_started  (Discard | Retry anyway)
```

Transitions use compare-and-set on preparation ID/state. Only one transition
may enter `committing`; repeated Retry/Bypass/Cancel events are harmless.
The stored pause reason determines the available actions; the two paused
states are not interchangeable. Cancel is valid from `preparing`, either
`paused` reason, or `ready`; while `committing`,
recovery actions are disabled or idempotently ignored until the transaction
settles. An Automatic send displays `Preparing Library context…` before
provider dispatch and keeps the user's draft, attachments, explicit evidence,
and one-shot prefill staged until commit. The optimistic USER echo remains
transient and excluded from provider history.

Outcomes:

| Outcome | Behavior |
| --- | --- |
| Evidence found | Stage the bundle into the same exact request, disclose the count, then dispatch. |
| Zero matches | Dispatch without evidence and retain `0 matches · sent without Library evidence` on the sent turn. |
| Timeout or service failure | Pause before dispatch; show Retry, Send once without Library, and Cancel. |
| User chooses Retry | Re-run preparation against the same executed draft and immutable turn policy, with a new attempt ID. |
| User chooses Send once without Library | Dispatch this turn without evidence; retain bypass disclosure; do not change policy. |
| User chooses Cancel | Dispatch nothing; preserve the draft and policy. |

At `committing`, the controller revalidates queue/generation authority and
gateway readiness against the same selected provider/model/endpoint intent. A
different resolved destination invalidates the preparation and leaves it
paused; it is never substituted silently.

For a durable conversation, first-persistence identity and auto-title are
transaction inputs, not eager session mutations. The persistence helper stages
any new conversation ID and computed title without publishing them to
`ConsoleChatSession`; only a successful SQLite commit may publish them.
Existing helpers that currently publish an ID or title before commit must be
refactored to this boundary. One in-process commit then:

1. atomically persists first-conversation identity/policy, the USER turn, an
   empty assistant recovery owner with `assistant_generation_state = accepted`, its `accepted`
   `console_dispatch_checkpoints` row, and any `library_preparation`
   disclosure needed for zero-match/bypass;
2. after SQLite commit, publishes the durable conversation ID/title and the
   USER/assistant owners, transitions the in-memory preparation to `accepted`,
   and clears only the captured attachments/evidence/prefill;
3. acknowledges the exact queue entry and fires the accepted hook through
   idempotent effects keyed by preparation ID;
4. CAS-transitions the durable checkpoint and assistant message from
   `accepted` to `dispatch_started`, guarded by checkpoint revision, both
   stored message versions, and both `messages.deleted = 0`, immediately before
   invoking the provider with the already-prepared request;
5. streams into the same assistant owner, then uses
   `settle_with_assistant(...)` to write its terminal content/status/metadata
   plus version/hash/sync intent and delete the operational checkpoint in one
   transaction.

If a supported provider returns a complete tool-call batch before terminal
content, ownership hands off to ADR-063 in one transaction: validate and write
`provider_continuation_json` on this same assistant, set
`assistant_generation_state = continuation_active`, bump the message
version/hash and sync intent, and delete the expected-revision dispatch
checkpoint **before any tool executes**. From that commit onward only ADR-063
Resume/Take over/Discard is advertised. If the handoff fails, no tool executes
and the dispatch checkpoint remains `dispatch_started`. A reasoning-only
complete continuation is written inside the ordinary terminal settlement
instead of creating a second owner. ADR-063's configured Sync-v2 outbox
projection remains part of its fail-closed pre-tool boundary after the local
handoff commit. ADR-063 completion/Discard also sets the closed assistant
generation terminal state in the same existing message-version transaction.

Any database failure before SQLite commit rolls back the transaction, leaves
the ordinary or first-persistence session ID/title unchanged, restores the
captured staged state, and transitions `committing -> paused` with generic
Retry and Cancel recovery. `Send once without Library` is offered only for an
Automatic turn paused by retrieval timeout/failure; it is never shown for a
Never turn or for a persistence failure.

Every failure after SQLite commit leaves the assistant/checkpoint as the
durable recovery owner; it never creates a second USER or assistant row. On
reload, `accepted` means no provider invocation was attempted and offers
**Retry response** and **Discard**. Retry revalidates the same resolved
destination identity and frozen authority, uses the durable USER/attachments,
and re-runs automatic retrieval under a new attempt when required. Because the
checkpoint does not retain a provider request body, Retry is disabled with a
literal reason if one-shot prefill or transient evidence cannot be reconstructed
exactly. `dispatch_started` means delivery is indeterminate and offers
**Retry anyway** with a duplicate-request warning or **Discard**; it is never
automatically replayed.

Discard also calls `settle_with_assistant(...)`, atomically marking the
assistant owner interrupted and deleting the expected-revision checkpoint; a
failure changes neither. It keeps the durable USER turn. For a queued origin,
durable acceptance means the exact entry is never returned to pending: it is
acknowledged at most once, later entries remain paused while the checkpoint is
unresolved, and Retry or Discard settles that same accepted entry before queue
advancement. Recovery after restart is projected before any queue is allowed to
advance.

Loader reconciliation is deterministic:

| Durable rows | Loader result |
| --- | --- |
| Valid active `provider_continuation_json` and no dispatch checkpoint, with any NULL/new assistant state | ADR-063 continuation is authoritative; expose its recovery surface with actions disabled, lazily normalize `continuation_active` under message-version/deletion CAS, and publish the committed version/hash to the recovery handle before enabling actions. CAS conflict re-reads and reconciles changed/deleted ownership rather than preserving stale actions. |
| Valid checkpoint + matching `accepted`/`dispatch_started` assistant state and versions | Hydrate the checkpoint state's recovery actions. |
| Dispatch checkpoint + active `provider_continuation_json` on the same assistant | ADR-063 continuation wins; transactionally set `continuation_active`, bump message version/sync intent, delete the stale dispatch checkpoint, and never offer dispatch Retry/Discard. |
| Valid checkpoint + matching terminal assistant | Terminal assistant wins; delete the stale checkpoint transactionally and never offer retry. |
| Checkpoint whose USER/assistant is missing, cross-conversation, or wrong-role | Quarantine as a persistence error; never invoke a provider or delete unrelated messages. |
| No local checkpoint + `continuation_active` but no valid active continuation | Quarantine as continuation corruption; never ordinary-load it or invoke a provider. |
| No local checkpoint + `accepted`/`dispatch_started` | Project inert source-device pending/interrupted state; never invoke a provider automatically. |
| No checkpoint + terminal/NULL assistant state | Use ordinary message loading; no dispatch-recovery action is inferred from content alone. |

Because normal terminal completion and Discard update the assistant and delete
the checkpoint atomically, a crash cannot expose a durable discarded state with
a nonterminal assistant. The ADR-063 handoff likewise makes dispatch and tool
continuation ownership mutually exclusive. Fault injection nevertheless covers
every statement, both precedence reconciliations, and the message-version or
soft-delete conflict paths.

The empty assistant owner is therefore never an unexplained synchronized row.
Sync v1/v2, `.chatbook`, active-path JSON, and trajectory message projections
carry `assistant_generation_state` through the existing whole-message
version/hash contract, but never carry the device-local dispatch checkpoint.
On another device or after import, `accepted` renders
`Response accepted on another device; waiting for dispatch.` and
`dispatch_started` renders
`Response delivery status is unknown on the source device.` These states are
inert: they cannot expose source-device Retry/Discard. An explicit **Retry as a
new response** may create a sibling assistant variant without mutating or
claiming the source owner. Terminal publication replaces the state through the
ordinary message conflict rules. Text/Markdown exports render the literal
state copy; JSON/Chatbook exports include the closed state field, so none emits
a blank pending assistant without explanation.

Terminal settlement uses the closed message state as its durable discriminator:

| Outcome | Durable assistant state and context behavior |
| --- | --- |
| Complete, including empty content | `complete`; empty content renders/exports explicit empty-response copy and contributes no empty provider-history item. |
| User stop | `stopped`; retain bounded partial content and existing stopped-turn provider-history behavior. |
| Provider/preparation failure after acceptance | `failed`; retain bounded visible failure/partial copy and exclude it from later provider history. |
| Discard recovery owner | `discarded`; retain the USER, render literal discarded copy, and exclude the assistant from provider history. |

Normal provider-history construction excludes `accepted` and
`dispatch_started`; `continuation_active` is included only through ADR-063's
validated provider-specific continuation projection. Thus the durable recovery
owner never becomes an empty assistant prompt item.

Each transition requires the expected checkpoint revision, stored USER and
assistant `messages.version`, matching assistant state, and both
`deleted = 0`; concurrent message mutation or deletion returns Conflict and
leaves every row unchanged.

A process crash after durable commit but before provider dispatch therefore
leaves a visible accepted recovery owner. A crash after `dispatch_started` is
truthfully indeterminate; the design does not claim distributed exactly-once
delivery to an external model provider.

An ephemeral turn follows the same in-memory transitions and actions but skips
the SQLite transaction/checkpoint APIs. Its recovery survives screen
replacement only for the life of the app runtime, and unresolved recovery
blocks promotion as defined in §3.4.

Cancel removes the transient optimistic echo and restores the captured staged
state. For a manual send it returns the draft to the composer. For a queued send
it releases the exact entry back to pending, pauses queue advancement, and does
not copy the draft into the foreground composer. Closing the owning session or
shutting down settles the preparation through the same cancellation path before
store teardown. A paused queued preparation remains owned by its session across
ordinary navigation and blocks later queue entries from dispatching.

Retry keeps the same preparation, draft, policy revision, selector, scope, and
provider intent but creates a new retrieval attempt ID. It does not silently
refresh policy. To apply a later policy or provider-selection change, Cancel
and execute the turn again.

### 6.4 Durable sent-turn disclosure

Zero-match and one-shot-bypass outcomes that actually dispatch persist one
bounded device-local `library_preparation` sidecar row with the USER turn. It is
the owner of `0 matches · sent without Library evidence` and the equivalent
bypass disclosure across repaint, branch selection, and restart. A cancelled
preparation creates neither a durable USER row nor a sidecar event.

Default trajectory export reduces this event to outcome/result count/source
categories; full opt-in contains the same bounded payload because the event
never retains query or source identity. Import treats it as inert view data.

## 7. Assistant Library activity

### 7.1 What is recorded

Each completed or provider-level refused built-in Library operation creates a
versioned, bounded payload in a `library_activity` sidecar row. A
conversation-policy Blocked state registers no provider and therefore creates
no synthetic activity event:

```json
{
  "version": 1,
  "attempt_id": "opaque",
  "run_id": "opaque",
  "actor": {
    "kind": "primary|subagent",
    "run_id": "opaque",
    "parent_run_id": "opaque-or-null"
  },
  "library_provider": "direct|rag",
  "operation": "library_search_notes",
  "status": "succeeded|empty|blocked|failed",
  "result_count": 3,
  "query_preview": "bounded text or null",
  "source_refs": [
    {"type": "note", "id": "opaque", "title": "bounded title"}
  ],
  "error_code": null,
  "error_summary": null
}
```

Bounds are service-owned and applied before persistence:

- fixed maximum query/error/title characters;
- fixed maximum reference count;
- opaque canonical IDs only;
- no bodies, snippets, excerpts, embeddings, binary data, filesystem paths,
  credentials, provider request bodies, or arbitrary exception strings.

The sidecar's top-level message/conversation/turn/seq fields provide durable
ownership and ordering. Model/provider fields describe the model turn; the
payload's `library_provider` describes Direct versus RAG.

### 7.2 Capture boundary

Capture occurs at the trusted built-in Library-provider result seam, before
provider-result truncation and before the result reaches the model. This is
the only point that sees authoritative operation identity and structured
source references without parsing lossy marker text.

The event first enters a thread-safe, turn-owned memory sink in
`ConsoleChatStore` (or an app-owned runtime service with the same lifetime),
never a mounted Screen. If minimization or sink admission fails, the Library
result is withheld and the tool receives a bounded failure; authorization
cannot proceed without review capture.

Durable persistence may occur with the turn's other sidecar writes. A transient
database failure leaves the bounded event in that store-owned retry buffer.
Exhaustion does not retroactively fail an otherwise completed model turn, but
the Inspector shows **Library activity not saved in this session** with Retry
for as long as the buffer remains live. Navigation or screen replacement cannot
lose it; session close/promotion/shutdown performs a final bounded flush. A
process crash may lose an unsaved buffer, which is why this is reviewability,
not an audit-grade durability claim. Logs contain only event IDs, sizes,
statuses, and error categories.

### 7.3 Anchoring, branches, and temporary work

- Anchor each event to the durable user/system turn opener, not an ephemeral
  tool marker or whichever assistant message happens to finish last.
- `turn_id`, `attempt_id`, `run_id`, actor kind, and parent-run identity
  distinguish retries and subagents.
- The selected-turn projection follows the active message lineage. Activity
  from another branch remains stored and appears only when that branch/turn is
  selected through the appropriate historical view.
- Ephemeral sessions keep activity in memory and persist it as part of atomic
  promotion.

### 7.4 Presentation and export

Activity never becomes:

- staged evidence;
- a Sources entry;
- next-send context;
- a system/user message;
- provider history;
- synchronized data.

In this design, **device-local persistence** means the database rows are never
synced automatically. ADR-067 still permits an explicit user-initiated
trajectory export to copy their bounded/redacted representation out of the
device; “local-only” is not used as a no-egress promise.

The generic trajectory ledger does not render it as another tool row. The
Selected turn Inspector uses the separate projection and shows operation,
actor, mode, status, count, time, and bounded references.

ADR-067 exports raw sidecar rows in an explicit trajectory export. Its default
redaction path must recognize `library_activity` and replace query/source
details with bounded operation/status/count previews. Full activity details
are included only in the existing explicit full-export opt-in. Import treats
unknown/new fields as inert data and never executes source references.
This export contract applies to historical `library_activity` and
`library_preparation` events only. Rows from the separate temporary operational
`console_dispatch_checkpoints` table are never selected by either default or
full export serialization.

## 8. Console interface

### 8.1 Fixed two-axis status chip

The status strip uses exactly one fixed-order policy chip:

```text
Library · Auto off · Agent blocked
Library · Auto on · Agent blocked
Library · Auto off · Agent allowed
Library · Auto on · Agent allowed
```

Spacing may normalize to the renderer, but noun and axis order never change.
Do not mix `Never`, `Automatic`, `Off`, `On`, `Blocked`, and `Allowed` between
chip variants.

Exceptional unreadable state is explicit:

```text
Library: blocked · policy unavailable
```

Runtime readiness, item scope, Direct/RAG selection, provider destination, and
staged source counts are not extra chip axes. They appear in expanded details
and their existing dedicated surfaces.

### 8.2 Library Access modal

Activating the policy chip opens a policy-only modal. It contains:

- saved-row status: `This device · policy saved locally · not synced`;
- missing-row status: `This device · no policy saved · defaults to Never / Blocked`;
- Automatic retrieval radio rows: Never / Automatic;
- Assistant Library access radio rows: Blocked / Allowed;
- resolved Direct/RAG explanation when Allowed;
- category disclosure: Direct covers all six Library categories; RAG covers
  Notes, Media, and Conversations;
- selected provider/model intent before execution and the most recent resolved
  on-device/private-network/public-network/external-unknown disclosure;
- explicit Save and Cancel;
- persistent Saving, Saved, Conflict, Unavailable, and Error feedback near
  the controls.

Use text-valued radio rows, not unlabeled switches. Save is disabled until the
draft differs from its loaded revision and while a save is in flight. A
conflict offers Reload and Compare/Retry. Read failure disables policy editing
until Retry; it does not fabricate Never/Blocked as if successfully loaded.

For an unpersisted session, the status line says
`Temporary session · applies until close or saved`. Save commits to the holder,
not the database, and the modal says so.

Escape, backdrop, and Cancel use ADR-031 safe dismissal. A clean modal closes.
A dirty modal asks whether to discard; it never saves or discards merely
because Escape/backdrop was pressed. Focus starts on the first radio group,
returns to the opening chip after close, and moves to the error/recovery status
when a save fails.

### 8.3 Canonical Settings

F9 Settings exposes a **New Console conversations** group with the Never /
Automatic and Blocked / Allowed defaults. It explicitly says changes affect
only sessions created afterward. The global Direct/RAG selector remains a
separate control labeled as tool mode, not access. No control is added to
`Tools_Settings_Window.py` or the deprecated enhanced settings sidebar.

### 8.4 Search Library modal

The existing combined modal becomes a search-only surface:

- exact composer draft prefill;
- editable query;
- manual source-category toggles labeled `This search only`;
- current item-scope summary;
- Search Library and Cancel actions;
- retrieval status/recovery.

It contains no standing policy switch. Running a search never changes either
conversation policy axis.

### 8.5 Sources and Selected turn

The source tray uses one primary row per source:

```text
✓ Q3 planning notes                         note
✓ turbine-maintenance-log                  media
⚠ vendor contract                         note
```

Activation/expansion reveals snippet, authority, freshness, and source action.
Do not spend three or four visible rows per source in the ten-row tray.

Terminology is fixed:

- `Sources — next send` for staged evidence;
- `Cited sources (N)` for the selected sent answer;
- `Library activity (N actions)` for assistant operations.

The conversation Inspector gains a **Selected turn** group containing Cited
sources and Library activity. A message-level activity affordance selects that
turn and focuses the activity subsection. The empty state is explicit:
`No Library activity for this turn.` Activity is not added as another
top-level rail peer.

### 8.6 Responsive and literal rendering

Replace the modal's fixed width with a viewport-fitting bound and a bounded
scroll body. Keep actions visible/pinned when content overflows. At narrow
widths, radio rows, disclosures, and actions stack without horizontal
clipping. Tests mount the production screen hierarchy and stylesheet bundle
and assert painted containment, not only declared dimensions.

Long provider/model labels, translated copy expanded by at least 30%, CJK,
emoji, combining marks, and RTL-shaped text wrap or elide inside their owning
row without moving pinned actions off-screen. Every selected/blocked/saving/
conflict state remains text-labeled; color is never its only carrier. Repeated
Save/Retry activation while an operation is in flight is disabled or
idempotently ignored, and focus/status announcements identify the resulting
state.

All dynamic query, title, error, source, activity, and recovery text renders
with `markup=False` or the equivalent literal-text boundary. The adjacent
`console_staged_context.py` recovery sink is included when this flow touches
it so a Library title such as `[red]` cannot become formatting.

## 9. Failure behavior

| Failure | User-visible behavior | Runtime behavior |
| --- | --- | --- |
| v44→v45 seed absent/invalid or migration fails | Database upgrade error; normal recovery guidance | Whole migration rolls back; no partial policy/checkpoint table or version |
| Policy read fails/corrupt row | `Library: blocked · policy unavailable` | Never/Blocked for the attempted turn |
| Policy save fails | Modal stays open; Error + Retry | Prior committed revision remains active |
| Policy revision conflict | Conflict + Reload/Compare/Retry | No candidate publication |
| Conversation deleted during save | “Conversation no longer exists” | No row recreation |
| Auto retrieval timeout/failure | Pause with Retry / Send once without / Cancel | No provider dispatch until user decides |
| Auto retrieval zero matches | Persistent 0-match disclosure on turn | Dispatch without evidence |
| User cancels manual preparation | Draft/attachments/evidence/prefill restored | Transient echo removed; no provider dispatch |
| User cancels queued preparation | Queue entry remains pending | Exact claim released; later entries remain paused |
| Destination changes during retry | Preparation remains paused with new-destination notice | No silent context substitution or dispatch |
| Library provider blocked | Tool absent | No schema/call/result |
| Activity memory capture fails | Bounded tool failure | Result withheld from model |
| Activity durable save exhausts retries | Inspector “not saved in this session” + Retry | Store-owned buffer retained; completed turn remains completed |
| First-send/USER/assistant/checkpoint/preparation commit fails | Generic persistence failure with Retry / Cancel | Whole transaction rolls back; session ID/title and staged state remain pre-send; no Library bypass action |
| Accepted post-commit effect fails before provider invocation | Assistant recovery owner with Retry response / Discard | Same USER/assistant/checkpoint retained; effects retry idempotently; queue stays paused |
| Crash/failure after dispatch-start marker | Delivery status unknown; Retry anyway / Discard | Never auto-replay; reuse same assistant owner and warn of duplicate-request risk |
| Terminal completion/Discard settlement fails | Persistence recovery error; prior recovery remains | Assistant terminal update and checkpoint deletion both roll back; loader never observes a half-settled pair |
| Tool-continuation handoff fails | Continuation persistence error; dispatch recovery remains | No tool executes; no competing continuation owner is published |
| Assistant version changed/deleted during CAS or settlement | Conflict/reload recovery | Checkpoint and message remain unchanged; no unrelated row is revived |
| Remote/import contains unresolved assistant state without local checkpoint | Inert source-device status; optional Retry as new response | Never expose source Retry/Discard or auto-invoke provider |
| Ephemeral turn is unresolved during Save | `Finish or discard the pending turn before saving.` | No promotion write; in-memory recovery remains intact |
| Provider changes on-device→external while either Library axis is enabled | New destination re-disclosed | Policy preserved; later turn uses newly resolved destination |
| Conversation soft-deleted/restored | Hidden, then restored with prior controls | Policy/activity retained and inert while deleted |

## 10. Security and privacy invariants

- Manual search authority never grants automatic or assistant authority.
- Missing, corrupt, or unreadable policy never grants access.
- The trusted built-in namespace is reserved independently of catalog
  availability.
- Durable policy is re-read at execution; a cached Allowed holder is never the
  authority after a concurrent writer commits Blocked.
- Primary and subagents share one immutable maximum authority; a child cannot
  re-read globals to widen it.
- Assistant access governs only built-in Library providers. MCP and local
  tools remain under their independent permission principals and disclosure.
- Conversation policy is device-local and excluded from sync, import/export,
  metadata mirrors, FTS, diagnostics, and provider payloads.
- Activity and preparation disclosure use device-local persistence, are
  minimized before storage, literal-rendered, never synced, and
  default-redacted on explicit trajectory export.
- The operational dispatch checkpoint is device-local, temporary, excluded
  from both trajectory export modes, and contains no request body or content;
  recovery never fabricates exact replay when transient inputs are unavailable.
- Only the closed assistant generation state crosses sync/export. Remote/import
  unresolved owners are inert, and an ADR-063 continuation atomically replaces
  dispatch recovery before any tool can execute.
- Automatic retrieval uses a fixed category set and does not retain a manual
  query/filter as standing hidden policy.
- “Send once without Library” is request-scoped and visible; its disclosure is
  durable but it never persists as or weakens standing policy.
- New/changed policy, automatic-send, built-in Library-provider, preparation,
  and activity paths log no Library queries, titles, IDs, source bodies,
  excerpts, tool results, or arbitrary exception strings. This design does not
  make a repository-wide claim about untouched legacy Library logging.

## 11. Verification strategy

The repository requires targeted verification unless the owner separately
opts into a full sweep.

### 11.1 Migration and repository

- Historical v44 fixture proves the message state column, both
  policy/checkpoint tables, and the index are absent before migration and
  structurally complete afterward; historical NULL assistant state reads as
  complete except for valid active ADR-063 continuation, and non-assistant NULL
  remains valid.
- Fresh and migrated schema tests inspect all four message Sync-v1 triggers:
  create/delete/undelete payloads include assistant state, update watches and
  serializes it, and migration rerun/idempotence cannot leave an older trigger.
- Sync-v2 foundation tests prove each fresh/migrated create, update, delete, and
  undelete intent still passes exact source proof and outbox projection with an
  explicit NULL or closed state; older payloads missing only this field
  normalize to NULL, while unknown, malformed, or mismatched data still fails
  closed.
- The migration-capable initializer obtains `BEGIN IMMEDIATE` before its first
  version read; fresh, current, v44, and older-schema opens prove no nested
  deferred-transaction upgrade remains.
- Migration requires a sanitized seed, writes final rows only for active and
  soft-deleted conversation IDs present in its transaction, and advances v45
  atomically.
- Missing/invalid seed and injected statement failures leave schema/version
  unchanged; retry is deterministic.
- Two concurrent openers with different seeds prove the first successful
  migrator wins and no later reseed occurs.
- A conversation inserted after migration is never legacy-seeded.
- Existing conversations become prior-auto + Allowed; new/missing rows resolve
  Never + Blocked.
- Corrupt rows/read failures are explicit and fail closed.
- CAS success, stale revision, missing conversation, and concurrent writers.
- First conversation creation and policy insert are atomic; injected failures
  at every conversation/policy/USER/assistant/checkpoint/preparation write
  boundary leave both the database and in-memory session persistence
  identity/title unchanged and allow a clean retry.
- Ephemeral promotion success and injected failures at every write boundary
  prove complete rollback/restoration; unresolved ephemeral preparation or
  dispatch recovery visibly blocks promotion without a write.
- Soft delete/tombstone, restore, and permanent purge retain or cascade
  policy/activity exactly as the lifecycle table defines.

### 11.2 Runtime and catalog

- All four policy combinations.
- Allowed Direct exposes exactly the 18 ADR-030 tools.
- Allowed RAG exposes exactly `search_library_rag`.
- Blocked exposes neither provider.
- Every 19-name collision is rejected in Blocked, Direct, and RAG modes for
  Skills and MCP.
- The reserved set is derived from the descriptor registry plus RAG constant;
  inventory drift fails a ratchet test.
- Same-process publication and second-process execution re-read prevent stale
  Allowed holders; commits after the linearization point affect the next turn.
- Policy/selector/provider changes during a turn do not affect that turn.
- Queued turns capture at execution.
- Primary and subagents share the snapshot; child narrowing still works.
- Read failure never constructs a Library provider.
- Temporary Allowed admits only the authenticated 19-name audited read-only
  provider; source spoofing, unknown names, and Blocked remain refused.
- Gateway-resolved loopback, private-network, public-network, default-cloud,
  custom, malformed, and unknown destinations classify conservatively;
  on-device-to-external disclosure is required for Automatic + Blocked as well
  as every Assistant Allowed combination.

Authorization/name-reservation tests are mutation-checked: removing the gate
or static reservation must turn the test red.

### 11.3 Automatic retrieval

- Eligible plain sends, excluded send kinds, and explicit staged-evidence skip.
- Exact draft query and fixed Notes/Media/Conversations categories.
- Item scope narrows Notes/Media and excludes Conversations when active.
- Success stages the bundle into the exact dispatched request.
- Cancel prevents dispatch and preserves the draft.
- Timeout/failure pauses; Retry, one-shot bypass, and Cancel have distinct
  outcomes and do not change policy.
- Zero matches dispatches without evidence and leaves persistent disclosure.
- No path silently falls through from failure to provider dispatch.
- Repeated/racing Retry, Bypass, Cancel, session close, shutdown, and queue
  cancellation prove one in-process commit and no double dispatch.
- Never-turn persistence failures expose generic Retry/Cancel only; Automatic
  retrieval failures alone expose Send once without Library.
- Manual cancel restores the exact draft/attachments/evidence/prefill and
  removes the transient echo; queued cancel releases the same entry and blocks
  later queue advancement.
- Navigation/screen replacement preserves a paused preparation; a changed
  gateway destination cannot be substituted during retry.
- Zero-match/bypass sidecars persist atomically with the sent USER turn, remain
  branch-correct after restart, and never retain query/source identity.
- Failures/crashes injected after SQLite commit, after session/message
  publication, after staged-state clearing, after queue acceptance, after the
  accepted hook, after `dispatch_started`, and at provider invocation hydrate
  one durable recovery owner without duplicate USER/assistant rows or duplicate
  queue acceptance.
- Terminal completion and Discard fault injection proves the assistant update
  and checkpoint deletion are atomic. Loader reconciliation covers
  checkpoint+nonterminal, checkpoint+terminal, invalid ownership/role, and no
  checkpoint without inferring replay from an ordinary empty assistant.
- Complete-with-content, empty complete, stopped, failed, and discarded
  settlements prove closed state, literal rendering, provider-history
  inclusion/exclusion, expected message version, and `deleted = 0` conflicts.
- Crash/failure injection before and after ADR-063 tool-continuation creation
  proves exactly one recovery owner: no tool executes before handoff commits,
  continuation wins any dual-owner reconciliation, and reasoning-only complete
  continuation settles with the terminal message.
- Pre-v45 local rows, older-sync payloads, and `.chatbook` imports containing a
  valid active ADR-063 continuation plus NULL/missing generation state always
  expose continuation recovery; lazy normalization is version/deletion guarded
  and a successful normalization returns/publishes the committed version/hash
  before Resume/Discard. Tests exercise both actions after normalization,
  known rolled-back write failure with a confirming re-read, and racing message
  version/deletion or continuation replacement; stale actions are never left
  enabled for a changed owner.
- Runtime state transitions consume the foundation's Sync-v1/v2
  record/source/envelope contract. Active-path JSON, `.chatbook`, text/Markdown,
  and trajectory message projections prove unresolved state is never a blank
  unexplained row, checkpoints never cross devices/exports, remote/imported
  state is inert, and terminal publication follows whole-message
  conflict/version rules.
- `accepted` recovery supports Retry response/Discard; `dispatch_started`
  supports warned Retry anyway/Discard; unavailable transient inputs disable
  retry with a reason, later queue entries stay paused, and discard settles the
  exact accepted entry without returning it to pending.
- Operational dispatch checkpoints never enter sync, either trajectory export
  mode, provider context, or generic trajectory presentation, and are removed
  only after a durable assistant terminal state.
- Ephemeral turns use only the runtime-owned in-memory analogue, survive
  navigation without claiming crash durability, and cannot be promoted while
  preparation or dispatch recovery is unresolved.
- Rollback mode disables new policy/runtime behavior while preserving loader,
  continuation precedence, inert projection, and Discard-only draining; a build
  without the drain refuses v45 writes.

### 11.4 Activity

- Direct and RAG success/empty/blocked/failure events.
- Capture occurs before model delivery and before result truncation.
- Query/title/reference/error bounds and forbidden-content scans.
- Primary/subagent/run/attempt attribution under concurrency.
- Durable turn-opener anchoring, active-branch selection, and ephemeral
  promotion.
- Capture failure withholds result; durable failure remains in the app/store
  buffer across navigation, retries, shows warning, and receives a final flush
  on close/promotion/shutdown.
- `derive_trajectory` timing/message ownership is unchanged with activity rows
  present; separate projection returns them once.
- No activity in staged context, prompt construction, provider history, sync,
  ordinary logs, or Sources.
- Default trajectory export redacts activity details; full opt-in retains the
  bounded event.

### 11.5 Interface and live verification

- Exact chip grammar for four states plus unavailable state.
- Policy modal Save/Cancel, dirty Escape/backdrop, conflict, error, temp-state,
  saved-row/missing-row truth, provider disclosure, disabled reasons, and focus
  restoration.
- Editing policy on an empty session marks it non-untouched for navigation and
  close warnings; untouched captured defaults retain pristine replacement.
- Canonical Settings exposes only future-session defaults, keeps Direct/RAG
  separate, and round-trips through the config template/schema.
- Search modal exact draft prefill, no heuristic, and “This search only”
  filters.
- One painted row per source with expandable details.
- Selected-turn Cited sources/Library activity grouping and focus handoff.
- Literal dynamic text, including Rich/Textual markup-shaped titles/errors.
- Long/expanded, CJK, emoji, combining-mark, and RTL-shaped labels remain
  contained and readable; repeated Save/Retry activation cannot duplicate work.
- Narrow and standard viewports under the production hierarchy and complete
  stylesheet bundle; action containment and rendered-frame assertions.
- A deterministic recording provider drives exact queue/subagent/destination
  branches through the production UI; a real configured provider walkthrough
  is reserved for user-visible egress/result confirmation rather than model
  tool-choice nondeterminism.
- Scratch-profile real Console walkthrough covering policy persistence,
  automatic success/zero/failure recovery, Blocked/Allowed tool catalog,
  Direct/RAG selection, activity review, provider change disclosure, restart,
  and soft delete/restore. Permanent-purge cascade remains a repository-level
  verification because Console currently has no hard-delete action.

## 12. Rollout and rollback

Rollout is fail-closed. The v44→v45 migration receives one required sanitized
legacy seed and either commits the schema, final legacy rows, and version
together or rolls back together. No startup initializer, pending marker, or
recurring backfill exists.

Rollback disables policy editing, automatic retrieval, and built-in Library
composition while retaining the v45 column/tables and sidecar rows. The minimal
checkpoint loader, ADR-063 precedence handoff, inert remote/import projection,
and atomic **Discard** settlement remain enabled until
`console_dispatch_checkpoints` is empty; Retry/Retry-anyway may be hidden in
rollback mode, but existing rows are never abandoned or blindly deleted. A
rollback build that cannot provide this drain refuses to open v45 for writes.
Manual Search Library remains available. Older application versions must not
be asked to down-migrate or reinterpret the v45 database; use the normal
migration backup and downgrade procedure.

## ADR check

**ADR required:** yes

**ADR path:** `backlog/decisions/079-console-library-conversation-authority.md`

**Reason:** the design changes local schema/migration, sync and ownership,
assistant permission/runtime composition, privacy retention, and long-lived
Console control/review structure.
