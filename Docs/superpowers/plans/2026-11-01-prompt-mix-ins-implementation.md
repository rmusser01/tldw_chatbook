# Console Prompt Mix-ins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users combine literal Before/After response modifiers through the existing composer Prompts action, preserving clean history and exact retry instructions.

**Architecture:** Extend the existing Prompt artifact/source contracts and Prompt Workbench. Chat runtime owners capture per-chat selections into immutable send inputs; one pure composition function changes only the active user envelope. Existing conversation transactions retain operational retry snapshots independently of diagnostic traces, and provider adapters preserve eligible clean-history caching.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, SQLite/FTS5, existing Pydantic/source adapters, pytest/pytest-asyncio. No new dependency.

**Spec:** [Approved Console prompt mix-ins design](../specs/2026-11-01-prompt-mix-ins-design.md)

**Decision:** [ADR-286](../../../backlog/decisions/286-console-prompt-mix-ins.md)

**Planning task:** [TASK-38535](../../../backlog/tasks/task-38535%20-%20Plan-Console-prompt-mix-in-implementation.md)

**Baseline:** Design commit `13ded5544d`; source mapping inspected against the working checkout with HEAD `78e297e889`. Unrelated staged work is not a committed prerequisite. Prompts schema 5 and ChaChaNotes schema 81 were inspected; other work has staged later ChaChaNotes migrations. Allocate actual migration versions at execution, after checking integrated code, all refs and active worktrees. The descriptive SQL paths below deliberately reserve no numeric version.

**Status:** Implemented on dev with schema86/Prompts6. Fresh PR3346 review and both scoped correction reviews are complete; all findings are closed. TASK-38544 is Done. The PR remains a draft and normal hosted checks are separate.

```text
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: Direct implementation of the accepted design. Amend the ADR/spec before
changing instruction lifetime, retry behavior, source authority or retention.
```

## Global Constraints

The following quoted requirements are copied from the approved spec and apply to every task.

- "Each mix-in has one literal instruction body and one Before/After placement."
- "There are no variable forms, conditions, bundles, or automatic activation."
- "Console access is part of the existing composer **Prompts** button and Prompt Workbench."
- "There is no separate Mix-ins button or manager dialog."
- "Library edits affect future sends."
- "Running and queued sends retain their captured definitions."
- "Retry uses the original attempt's mix-ins; regeneration and edit-and-resend use current choices."
- "New chats start with none selected."
- "No cross-device chat-selection or retry-record sync protocol is introduced in v1."
- "No mix-ins means the original payload is unchanged."
- "Temporary attempts retain the same structure in memory."
- "The existing queue remains process-memory-only and retains its existing capacity, pause and quit behavior."
- "Use targeted tests touching the changed owners; a full suite requires separate user authorization."
- "Do not silently enable caching, change TTLs, force cache writes or assume every OpenAI-compatible endpoint accepts provider-specific fields."

Additional governing repository constraints: parameterized SQL; validated trust boundaries; optional-dependency checks; existing permission/source policies; no instruction bodies in logs or generic metadata; workers for operations over 100 ms; source TCSS plus `build_css.py`, never hand-edit the generated bundle. Python/UI values use existing design tokens and component patterns. Screens do not claim terminal-convention keys or shadow global bindings.

Read each linked Backlog task before implementation. Move it to In Progress and add its own Implementation Plan, including the ADR check, before changing product files. To Do tasks intentionally have no premature Implementation Plan or Implementation Notes. Use the user's chosen execution mode; no subagents are dispatched by this document itself. Preserve unrelated work and commit only each slice's paths/hunks.

## Delivery order and review boundaries

| Slice | Backlog task | Dependencies | Independently reviewable result |
| --- | --- | --- | --- |
| 1 | [TASK-38536](../../../backlog/tasks/task-38536%20-%20Add-typed-literal-mix-in-artifacts-to-the-local-prompt-library.md) | Planning | Strict local Mix-in records and safe existing consumers |
| 2 | [TASK-38537](../../../backlog/tasks/task-38537%20-%20Preserve-mix-in-type-across-portable-records-and-server-capabilities.md) | 1 | Portable definitions and truthful source capabilities |
| 3 | [TASK-38538](../../../backlog/tasks/task-38538%20-%20Persist-chat-mix-in-preferences-and-immutable-attempt-snapshots.md) | 1 | Transactional preference/attempt storage with lifecycle coverage |
| 4 | [TASK-38539](../../../backlog/tasks/task-38539%20-%20Capture-chat-mix-in-selections-during-manual-and-queue-admission.md) | 2, 3 | Runtime selection and immutable manual/queued capture |
| 5 | [TASK-38540](../../../backlog/tasks/task-38540%20-%20Compose-current-send-mix-ins-and-preserve-exact-retry-behavior.md) | 4 | Provider composition, clean history and original/current resend semantics |
| 6 | [TASK-38541](../../../backlog/tasks/task-38541%20-%20Preserve-eligible-clean-history-caches-for-mix-in-sends.md) | 5 | Capability-gated stable-history cache boundaries |
| 7 | [TASK-38542](../../../backlog/tasks/task-38542%20-%20Author-and-browse-dedicated-mix-ins-in-Library-Prompts.md) | 2, 4 | Existing Library editor/filter/handoff support |
| 8 | [TASK-38543](../../../backlog/tasks/task-38543%20-%20Manage-chat-mix-ins-through-the-existing-composer-Prompts-action.md) | 5, 7 | Existing Workbench controls, count, inspection and recovery copy |
| 9 | [TASK-38544](../../../backlog/tasks/task-38544%20-%20Qualify-the-integrated-prompt-mix-in-lifecycle-and-document-delivery.md) | 6, 8 | Integrated verification, live evidence and accurate user documentation |

Each slice is one intended PR. Keep supporting fixtures, documentation and migrations with their owner slice. Slices 2/3 and 6/7 can be implemented independently after their predecessors, but do not modify overlapping shared files concurrently. A later source discovery that requires splitting a slice must first update its AC and this plan; do not turn one task into an unrelated cleanup campaign.

## Source and file ownership map

Paths below are repository-relative. New interfaces in this plan are contracts to implement, not existing APIs. Retain existing module names and surrounding patterns; do not rewrite the controller or create a second engine.

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Prompt_Management/prompt_artifact_models.py` | Add `LiteralMixinDefinition` and the Mix-in discriminator/decoded union |
| `tldw_chatbook/Prompt_Management/prompt_artifact_codec.py` | Strict schema-v3 dispatch; existing v1/v2 behavior unchanged |
| `tldw_chatbook/Prompt_Management/prompt_source_capabilities.py` | Explicit Mix-in type/kind capability; ordinary Prompt eligibility remains narrow |
| `tldw_chatbook/DB/Prompts_DB.py` and `DB/migrations/prompts_literal_mixins.sql` | Extend local constraints and UUID/revision snapshot reads without losing existing records/FTS/history |
| Existing Prompt source/portable adapters | Normalize Mix-in records, preserve type and refuse unsupported sources |
| **New** `tldw_chatbook/Chat/console_prompt_mixins.py` | Immutable reference/selection/attempt values, preference codec, pure ordering and literal content composition |
| **New** `tldw_chatbook/DB/console_prompt_mixins.py` | Caller-cursor SQL for preferences, immutable used revisions, attempt records, copy/fork and purge |
| `tldw_chatbook/DB/migrations/chachanotes_prompt_mixins.sql` | Operational tables, bounds, FK/index/immutability rules; executed by the actual next migration |
| **New** `tldw_chatbook/Chat/console_prompt_mixin_service.py` | Runtime-owned selection orchestration, source resolution, capture, save recovery and transaction contributions |
| Existing store/context/received-turn/queue owners | Carry immutable values and exact-origin ownership through existing admission |
| Existing prepared-request/controller/gateway/provenance owners | Compose the verified current envelope once, count it, bind capture and retain per-attempt inputs |
| Existing provider adapter | Translate a semantic clean-history cache boundary only when supported |
| **New** `tldw_chatbook/Widgets/Prompts/prompt_mixin_editor.py` | Small shared literal editor using existing form patterns |
| **New** `tldw_chatbook/Widgets/Console/console_prompt_mixins_view.py` | Workbench selection view; renders snapshots and emits intents, owns no scheduling/storage |
| Existing Library/Console controllers and CSS modules | Narrow wiring and token-backed layout; existing Prompts route remains the entry point |

Do not create a standalone catalog, source registry, database, permission store, renderer, parameter engine or generic pipeline abstraction. These three focused Python modules and two widgets separate responsibilities that already cross runtime/storage/UI boundaries.

## Contract ledger

### Library record: Slice 1

Allocate literal mix-ins schema version **3**, leaving server-v1 and Console-v2 unchanged. The exact canonical JSON shape is:

```json
{"kind":"text_mixin","schema_version":3,"body":"Respond in noir style.\n","placement":"before"}
```

Store `artifact_type="mixin"`, `prompt_format="structured"`, `prompt_schema_version=3`. Decode only that matching type/kind/version with exact field types; reject unknown fields, boolean versions, blank bodies and unknown placements. Body limits use the selected source's existing compiled-text/definition/request limits. Preserve accepted whitespace literally. Compatibility projection is `system_prompt=""`, `user_prompt=body`; the definition remains canonical and all ordinary execution consumers must gate on `artifact_type == "prompt"`.

Add `LiteralMixinDefinition(body: str, placement: Literal["before", "after"], kind: Literal["text_mixin"] = "text_mixin", schema_version: Literal[3] = 3)` in the existing model module. Hide body in representations. `DecodedPromptArtifact.definition` becomes `BlockArtifactDefinition | LiteralMixinDefinition | None`; add state `supported_v3`. Existing `supported_v2` callers must narrow to block definitions, not assume every supported artifact has lanes. Keep malformed/foreign/unsupported states and raw stored values distinguishable. Serialization uses explicit kind/type dispatch.

Add `PromptsDatabase.fetch_mixin_snapshot(record_uuids: tuple[str, ...]) -> tuple[dict[str, Any], ...]` for one local read snapshot, and `PromptScopeService.read_mixin_records(*, mode: PromptBackend | str, record_uuids: tuple[str, ...]) -> tuple[dict[str, Any], ...]` as the policy/capability-checked async entry point. These read only the requested UUIDs, return UUID/version/definition together, and refuse missing/deleted/unsupported records. Respect SQLite parameter limits through the existing batching patterns while keeping one read transaction. Slice 1 implements local reads; Slice 2 permits server reads only when existing advertised revision-qualified operations can fulfill the same contract. The runtime checks source authority identity before and after the awaited call.

The effective read-only `PromptSourceCapabilities.mixin_snapshot_reads` flag is True locally and False for current server sources. Advertised Mix-in CRUD does not imply capture support; enable/send UI must honor the separate read capability. Current server mutation APIs also cannot honor `expected_version`, so their effective conditional-update capability is False and supplied write preconditions refuse before they can be discarded. No new endpoint or user option was added.

### Runtime values: Slice 3

Create these frozen values in `Chat/console_prompt_mixins.py`; all body/name-bearing fields are representation-hidden. Use existing input validation and strict codecs, not coercion.

```text
MixinPlacement = Literal["before", "after"]

MixinSourceRef(authority_key: str, backend: Literal["local", "server"], record_uuid: str)
MixinSelection(source: MixinSourceRef, enabled: bool, name: str,
               placement: MixinPlacement, last_used_version: int | None = None)
MixinPreferences(schema_version: int = 1, revision: int = 0,
                 entries: tuple[MixinSelection, ...] = ())
MixinSnapshot(source: MixinSourceRef, record_version: int, name: str,
              definition: LiteralMixinDefinition)
MixinAttemptSnapshot(items: tuple[MixinSnapshot, ...] = ())
MixinAttemptWrite(attempt_id: str, user_message_id: str,
                  assistant_message_id: str, snapshot: MixinAttemptSnapshot,
                  retry_of_attempt_id: str | None = None)
MixinPreferenceRead(status: Literal["absent", "supported", "unsupported", "corrupt"],
                    preferences: MixinPreferences | None, raw: object)
MixinReplayRead(status: Literal["legacy", "recorded", "unavailable"],
                snapshot: MixinAttemptSnapshot | None)

def decode_mixin_preferences(raw: object) -> MixinPreferenceRead: ...
def encode_mixin_preferences(value: MixinPreferences) -> str: ...
def reconcile_mixin_placements(
    preferences: MixinPreferences, snapshots: tuple[MixinSnapshot, ...]
) -> MixinPreferences: ...
def compose_mixin_content(
    content: str | list[dict[str, Any]], snapshot: MixinAttemptSnapshot
) -> str | list[dict[str, Any]]: ...
```

The signatures above describe APIs; their bodies are implemented in the owning slices. `raw` is retained only for unknown-format preservation and never exposed in diagnostics. An absent preference reads as empty, while corrupt/future content is unavailable and preserved until an explicit user reset. A recorded-empty attempt is `recorded` plus `MixinAttemptSnapshot(())`, never the same as `unavailable`. Define bounded `MixinPreferenceConflictError`, `MixinSnapshotUnavailableError` and `MixinSourceChangedError` alongside the owner that raises them; messages contain category codes rather than supplied instruction text.

`authority_key` identifies the configured source/profile through existing application scope identity. It contains no credential or credential-bearing URL. A record UUID alone is not authority. Resolve a local UUID with a UUID query, never by treating it as a name. Source changes invalidate pending reads and handoffs.

### Storage and transaction seam: Slice 3

`DB/console_prompt_mixins.py` exposes caller-cursor functions; it does not open, commit or rollback independent transactions:

```python
def read_preferences(cursor: sqlite3.Cursor, conversation_id: str) -> MixinPreferenceRead: ...
def save_preferences(cursor: sqlite3.Cursor, conversation_id: str, *,
                     expected_revision: int, value: MixinPreferences) -> MixinPreferences: ...
def write_attempt(cursor: sqlite3.Cursor, conversation_id: str, *,
                  batch: MixinAttemptWrite, native_message_ids: Mapping[str, str]) -> None: ...
def read_attempt(cursor: sqlite3.Cursor, conversation_id: str, *,
                 attempt_id: str) -> MixinReplayRead: ...
def copy_conversation_mixins(cursor: sqlite3.Cursor, *, source_conversation_id: str,
                             destination_conversation_id: str,
                             message_ids: Mapping[str, str]) -> None: ...
```

The four operational tables are `console_prompt_mixin_preferences`, `console_prompt_mixin_revisions`, `console_prompt_mixin_attempts`, and `console_prompt_mixin_attempt_items`. Preferences have a conversation PK/FK, schema discriminator, optimistic revision and bounded content-free selection JSON. Used revisions hold scoped immutable snapshot IDs plus source identity/version/name/placement/exact body. Attempts identify conversation, native attempt, user/assistant anchors, optional retry predecessor and expected item count. Message anchors are validated for conversation ownership at insertion and become NULL on hard message deletion; they do not cascade away an attempt still owned by the conversation. Ordered items reference same-conversation snapshots with unique `(conversation_id, attempt_id, ordinal)` and matching composite FKs. Zero items is an explicit valid attempt record.

Use conversation ownership and cascading hard-purge deletion. Copy/fork copies each distinct referenced used revision once into the destination's ownership and remaps message anchors in the existing fork transaction; it does not copy active preferences. This avoids a new global reference-counting service and makes source purge safe. Immutable-body update guards, foreign-key checks and scoped lookup enforce ownership. Use a bounded digest index only as a lookup accelerator and compare complete stored snapshot values before reuse; never rely on digest equality alone or expose text-derived hashes in diagnostics. Snapshot text follows the existing library compiled-text limit (`LOCAL_COMPILED_LANE_LIMIT`, 20,000 characters at this baseline); serialized definitions use `LOCAL_DEFINITION_LIMIT` (256,000 UTF-8 bytes). Preference JSON uses the existing `LOCAL_REQUEST_LIMIT` (512,000 UTF-8 bytes), not a new configurable capacity. Total requests still obey existing context budgets. Reject duplicate references, invalid flags and oversized records rather than silently dropping entries; server values must also fit these client operational-storage bounds before admission.

The existing `ConsoleTransactionWriter` deliberately accepts only restricted INSERT statements. Do not widen it to arbitrary SQL or expose its cursor. Add one narrow `append_mixin_attempt(*, batch: MixinAttemptWrite, native_message_ids: Mapping[str, str]) -> None` operation, implemented by its private cursor owner using `write_attempt`. `MixinAttemptContribution` in the service module implements existing `write`/`write_exact` contribution contracts and passes only exact native-to-durable anchors. Reuse the existing durable-acceptance fingerprint contract for these body-bearing contributions without logging fingerprints or source bodies.

Keep body-free attempt identity and a `mixin_snapshot_schema_version=1` discriminator in the existing message/checkpoint projection so a lost new-format record cannot masquerade as a legacy empty set. Classification at read is: marker absent on pre-feature data means legacy; marker present requires a valid corresponding record and item count. Retry creates a new attempt linked to the chosen previous attempt while retaining both records. Tool steps/transport retries within that attempt reuse it. Classify new message locators in the existing voice-promotion locator census and use exact remapping in promotion/fork paths.

### Runtime service: Slice 4

`ConsolePromptMixinService` uses the existing store, PromptScopeService and ChatPersistenceService through constructor injection; it creates no engine or autonomous worker. Its public seam is:

```python
def preferences(self, session_id: str) -> MixinPreferences: ...
def replace_preferences(self, session_id: str, *, expected_revision: int,
                        value: MixinPreferences) -> MixinPreferences: ...
async def capture(self, session_id: str, *, expected_revision: int) -> MixinAttemptSnapshot: ...
async def retry_snapshot(self, session_id: str, *, assistant_message_id: str) -> MixinAttemptSnapshot: ...
```

`replace_preferences` performs the owner-thread live CAS; persistence goes through the existing bounded settings-save worker/recovery pattern and carries exact conversation/incarnation/preference revision. `capture` snapshots references synchronously before its first await, reads only enabled definitions through source policy, checks source and selection revisions before publishing, reconciles observed placement edits, and returns immutable items in Before/After order. Disabled unavailable entries need no body read. Existing source reads must return the UUID, version and body as one record; local multi-record reads use one DB read snapshot. A server lacking truthful revision-qualified reads cannot be enabled for mix-in capture: show unsupported capability rather than inventing an endpoint or claiming a global cross-source transaction.

Extend `ConsoleTurnConfigurationSnapshot` with `prompt_mixins: MixinAttemptSnapshot`, representation-hidden and default-empty. Mirror it through existing custody copies, frozen-context constructors, recovery serialization, received-turn preparation and queued prompt ownership. A queued entry stores the accepted snapshot inside its existing custody request; keep body-free queue render projections unchanged. Original Retry loads its recorded snapshot into the new context; it never invokes current-library capture. New automatic origins receive empty mix-ins rather than capturing human selections. Pure composition uses the verified current user anchor/active-request owner, not the last `role=user` found after hooks or tool-result conversion.

## Before starting Slice 1

- [ ] Read the spec/ADR, `Docs/Architecture/{chat-pipeline,console,library-shell,database-layer,llm-providers,backup-recovery}.md`, ADR-040/053/057/097/098/208/210, and relevant `backlog/docs/lessons-*.md` entries. UI slices also read `design-language.md` and `component-patterns.md`.
- [ ] Record the exact root, HEAD, `git status --short`, integrated schema versions and task ID claims. Use a clean isolated execution worktree if required by the execution skill/user; do not relocate or clean the current staged work as an incidental planning step.
- [ ] Run the existing affected baseline before attributing failures to new work:

```bash
.venv/bin/python -m pytest -q --tb=short Tests/Prompt_Management/test_prompt_artifact_codec.py Tests/Prompts_DB/test_prompts_db_retained_history.py Tests/Chat/test_console_received_admission.py Tests/Chat/test_console_prepared_request.py
```

Expected: the named tests execute their actual bodies. If pre-existing bootstrap/storage admission or environment failures prevent this, record exact node IDs and establish a same-tree baseline as required by the testing lessons; do not bypass the fence or broaden into unrelated repairs. All later commands run from the execution checkout with Python >=3.12. Root `pyproject.toml` now owns asyncio auto mode; `Tests/UI/pytest.ini` no longer exists. Use `pytest-timeout` settings, not an unavailable shell `timeout` command.

## Task 1 — Local typed artifacts (TASK-38536)

**Files:** modify `Prompt_Management/{prompt_artifact_models,prompt_artifact_codec,prompt_block_compiler,prompt_source_capabilities,prompt_scope_service,local_prompt_service,server_prompt_adapter,Prompts_Interop}.py`, `DB/Prompts_DB.py`, and existing Prompt eligibility checks in `UI/Console_Modules/prompts.py` / `Widgets/Console/console_prompts_modal.py`. Create `DB/migrations/prompts_literal_mixins.sql` and `Tests/Prompt_Management/test_prompt_mixins.py`; extend `Tests/Prompts_DB/test_prompts_db_retained_history.py` and `Tests/DB/test_migration_file_sources.py`. Product paths in this and subsequent slices are prefixed by `tldw_chatbook/`.

**Consumes:** existing Prompt/Recipe model, source limits and local versioned write/read boundaries.
**Produces:** the schema-v3 `LiteralMixinDefinition`/`supported_v3` contract, local CRUD/filter/snapshot-read support, explicit local capability and safe ordinary-Prompt consumers.

- [x] **1. Add a failing strict/literal decode regression.** Include the complete record; no variable renderer may touch its body.

```python
def test_mixin_v3_keeps_literal_body_and_has_no_system_lane():
    from tldw_chatbook.Prompt_Management.prompt_artifact_codec import decode_prompt_artifact
    body = '  <svg>{pose}</svg>\n<style>.x { color: red; }</style>\n'
    definition = {"kind": "text_mixin", "schema_version": 3,
                  "body": body, "placement": "after"}
    decoded = decode_prompt_artifact({
        "artifact_type": "mixin", "prompt_format": "structured",
        "prompt_schema_version": 3, "prompt_definition": definition,
        "system_prompt": "", "user_prompt": body,
    })
    assert decoded.state == "supported_v3"
    assert decoded.definition.body == body
    assert (decoded.compiled_system, decoded.compiled_user) == ("", body)
```

- [x] **2. Run the new node.** ` .venv/bin/python -m pytest -q Tests/Prompt_Management/test_prompt_mixins.py::test_mixin_v3_keeps_literal_body_and_has_no_system_lane` must fail because Mix-in is unsupported. Add parameterized cases for boolean/foreign versions, extra keys, whitespace-only body, unknown placement, and type/kind mismatch.
- [x] **3. Implement explicit dispatch and local services.** Use the contract ledger; the new branch precedes v1/v2 dispatch and never feeds a Mix-in into block compilation. Audit all artifact-type and `supported_v2` consumers with `rg`; change recipe-only exclusions to positive Prompt eligibility where appropriate. Add a type filter through service/DB count, browse and search so pagination totals match displayed types. Add authority-qualified UUID reads without name fallback. Source writes compile from the canonical definition and perform existing optimistic checks.

```python
if artifact_type == "mixin":
    return decode_literal_mixin_v3(record, raw=raw)
# Existing legacy/server-v1/Console-v2 dispatch follows unchanged.
```

Define `decode_literal_mixin_v3(record, *, raw)` in the existing codec with the same decoded-state failure style. Validate body/placement through `LiteralMixinDefinition`; never echo invalid values in errors.

- [x] **4. Migrate the local CHECK constraint without data loss.** Execute the SQL file through an actual new migration entry, increment the integrated schema version, and transactionally rebuild the Prompts table with the same columns/identities and `artifact_type IN ('prompt','recipe','mixin')`. Preserve keyword/collection FKs, FTS tables/triggers, existing indices, sync-log rows and retained-history order. Do not edit `sqlite_master` or disable safety globally. Add a real old-schema fixture using the existing retained-history seed pattern and compare all pre-existing records and sync payload bytes after migration; force a mid-migration failure and verify rollback.
- [x] **5. Verify targeted compatibility.** Run the new file, artifact codec/block compiler/source tests, retained-history tests, normal Console prompt application tests and migration-file guard. Expected: old Prompt/Recipe cases remain unchanged; mix-ins can save/search/restore but normal insert/System/Recipe paths refuse them. Run scoped lint/format for changed Python files, update the library/database architecture notes, self-review, and commit only this slice: `feat: add literal mix-in prompt artifacts`.

## Task 2 — Portability and server/source boundaries (TASK-38537)

**Files:** modify `Prompt_Management/{prompt_chatbook_record,prompt_markdown_export,Prompts_Interop,prompt_source_capabilities,server_prompt_adapter,server_prompt_service,prompt_scope_service}.py`, `tldw_api/prompt_chatbook_schemas.py`, existing Chatbook Prompt collection/import call sites; extend `Tests/Prompt_Management/{test_prompt_chatbook_record,test_server_prompt_service,test_prompt_scope_service}.py`, `Tests/Chatbooks/test_chatbook_prompt_round_trip.py`, `Tests/Library/test_prompt_export_roundtrip.py`, `Tests/tldw_api/test_prompt_chatbook_schemas.py`.

**Consumes:** Slice 1 model/codec and existing source capability/portable record codec.
**Produces:** mix-ins use portable record version **2**; existing Prompt/Recipe exports remain version 1. Server schemas permit known Mix-ins, while source operations still require advertised type/kind and trustworthy revision reads.

- [x] **1. Add failing Chatbook and Markdown regressions.** Use a complete detail mapping and assert exact definition preservation through existing encoder/decoder names.

```python
def test_mixin_chatbook_v2_preserves_exact_definition():
    import json
    from tldw_chatbook.Prompt_Management.prompt_chatbook_record import (
        encode_chatbook_prompt_record, decode_chatbook_prompt_record,
    )
    raw = json.dumps({"kind": "text_mixin", "schema_version": 3,
                      "body": "\n{style}\n### TITLE ###\n", "placement": "before"})
    detail = {"name": "Literal", "author": None, "details": None,
              "system_prompt": "", "user_prompt": "\n{style}\n### TITLE ###\n",
              "keywords": [], "artifact_type": "mixin", "prompt_format": "structured",
              "prompt_schema_version": 3, "prompt_definition": raw}
    encoded = encode_chatbook_prompt_record(detail)
    assert encoded["record_version"] == 2
    restored = decode_chatbook_prompt_record(encoded)
    assert restored["artifact_type"] == "mixin"
    assert restored["prompt_definition"] == raw
```

- [x] **2. Run the new nodes and observe unsupported type/version failures.** Retain existing version-2-invalid tests by making their fixture a v1 Prompt rather than globally accepting any version-2 shape.
- [x] **3. Implement version dispatch and a distinct Markdown envelope.** Reuse the v2 portable record inside fixed Markdown framing:

````markdown
# Chatbook mix-in

```json
{"record_schema":"tldw-chatbook-prompt","record_version":2,"artifact_type":"mixin"}
```
````

The example shows framing only; the real JSON line contains every field required by the portable codec, including the exact stored definition. Emit one canonical JSON line so body/name newlines and `### TITLE ###` strings remain escaped. Do not emit the legacy `### TITLE/SYSTEM/USER ###` envelope for mix-ins. Detect the fixed new envelope before legacy parsing and route its record through the strict codec. Unknown/malformed versions refuse without raw snippets or fallback. Test the pre-feature legacy parser behavior with a small frozen compatibility fixture: it must not create an ordinary Prompt from the new envelope. Existing legacy export/import remains unchanged.
- [x] **4. Extend server schemas/capabilities without inventing server support.** Permit `mixin` in known client schemas and advertise `(3, 'text_mixin')` only when the actual source does. Require source identity, UUID and immutable revision evidence for capture. Existing get/version APIs may satisfy that; if not, mark Mix-in selection unavailable until the server supports a truthful contract. An advertised type alone does not permit guessing a new endpoint or using an unversioned/name-based read. Test missing capabilities, unsupported schema, changed source, denied policy, renamed UUID and failed conditional writes. Import never enables chat state.
- [x] **5. Verify the listed codec/Chatbook/server/schema/Markdown files, scoped lint and user export documentation.** Expected: legacy/v1 fixtures unchanged, v2 Mix-ins exact, malformed imports pre-mutation, old sources visibly unavailable. Commit only this slice: `feat: preserve mix-in portability and source capabilities`.

## Task 3 — Preferences and operational retry ownership (TASK-38538)

**Files:** create `Chat/console_prompt_mixins.py`, `Chat/console_prompt_mixin_service.py` (transaction contribution first), `DB/console_prompt_mixins.py`, `DB/migrations/chachanotes_prompt_mixins.sql`; create `Tests/DB/test_console_prompt_mixins.py`, `Tests/Chat/test_console_prompt_mixin_values.py`, `Tests/Backup_Recovery/test_prompt_mixin_restore.py`. Modify `DB/ChaChaNotes_DB.py`, `Chat/{chat_persistence_service,console_transaction_contribution,message_metadata,console_chat_fork,console_conversation_hydration}.py`, the existing fork/promotion bundle owner and `DB/recovery_core.py` schema policy. Inspect `Backup_Recovery/{inventory,restore_groups,local_content_lifetime}.py`; modify only if existing whole-DB cohort ownership does not cover the new tables. Extend existing fork/promotion locator and migration/packaging tests.

**Consumes:** Slice 1 strict definition; existing caller-owned durable turn/promotion/fork transaction capabilities.
**Produces:** runtime value/preference codecs, four operational tables, exact attempt repository functions and `MixinAttemptContribution`/writer operation from the ledger. The contribution type lives in the new service module; create that focused contribution first, then extend the same module with orchestration in Slice 4.

- [x] **1. Add a real SQLite rollback test and strict preference-value tests.** Reuse the existing fixture rather than mock a DB or seed a weakened schema:

```python
import pytest
from Tests.Chat.test_console_trace_runtime import make_database as make_database
from tldw_chatbook.Chat.console_prompt_mixins import MixinPreferences
from tldw_chatbook.DB.console_prompt_mixins import read_preferences, save_preferences

def test_preference_write_rolls_back_with_parent_transaction(make_database):
    db = make_database(":memory:", "mixin-test")
    cid = db.add_conversation({"title": "Mix-ins"})
    assert cid
    with pytest.raises(RuntimeError), db.transaction() as cursor:
        save_preferences(cursor, cid, expected_revision=0, value=MixinPreferences())
        raise RuntimeError("abort parent")
    with db.transaction() as cursor:
        assert read_preferences(cursor, cid).status == "absent"
```

If the existing conversation constructor requires additional fields at the integrated baseline, copy its real fixture inputs instead of weakening constraints. Add future-format preservation, duplicate source references, exact booleans, source UUID validation and repr/body-leak regressions.

- [x] **2. Run the new storage node and observe missing-owner/schema failure.** Add a migrated-database case and a test that deliberately removes an item row from a marked new-format attempt: reading it must be unavailable, never recorded-empty.
- [x] **3. Implement the runtime values, preference codec and SQL file.** Use the contract ledger. Add actual migration registration/version and schema-policy/packaging entries. `save_preferences` uses the exact expected preference revision and preserves unrelated conversation state. `write_attempt` validates its anchors belong to the accepted conversation, reuses equal immutable snapshots within that conversation, records empty attempts explicitly, and refuses a reused attempt ID with different content. Use one parent transaction, parameterized SQL and fixed category exceptions. Add update guards to immutable snapshots and ordered-item count/FK validation.
- [x] **4. Wire the existing transaction contribution seam.** Add the single narrow writer method, its lifetime revocation checks, and the contribution's exact native ID writes. Persist body-free attempt/schema markers with the accepted message/checkpoint; include the contribution in durable acceptance identity. Extend fork mapping and voice locator census; copy distinct used snapshots once per destination and start its preferences empty. Promotion includes temporary records once. Hard purge cascades within the owner; soft delete does not erase retry instructions. No trace setting is consulted to retain these operational records.
- [x] **5. Verify storage, capture-off reopen, duplicate-attempt rejection, FK ownership, commit rollback, fork/source purge and temporary promotion.** Run the three new test files plus `Tests/Chat/test_console_chat_fork_persistence.py`, `Tests/Chat/test_console_durable_turn_acceptance.py`, applicable exact-ID promotion tests and migration/packaging checks. Recovery tests must operate on disposable candidates through existing backup/restore service owners. Update database/backup architecture docs, self-review and commit: `feat: retain chat mix-in preferences and attempt snapshots`.

Reviewed implementation: `7528f8e3c4` plus fix `e4560ff180`; targeted storage/fork/recovery evidence and baseline qualifications are in TASK-38538. Both review findings were cleared.

## Task 4 — Runtime selection and admission capture (TASK-38539)

**Files:** extend `Chat/{console_prompt_mixins,console_prompt_mixin_service,console_chat_store,console_turn_context,console_configuration_capture,console_configuration_preparation,console_received_turn,console_turn_preparation,console_prompt_queue,console_prompt_queue_coordinator,console_chat_controller,console_runtime}.py`; wire the existing app-owned `prompt_scope_service` through `app.py` / `UI/Screens/chat_screen.py` only where runtime composition requires it, and extend `UI/Console_Modules/session.py` for exact-origin hydration/save state. Create `Tests/Chat/test_console_prompt_mixin_admission.py`; extend existing received-admission, execution-context and queue tests.

**Consumes:** Slice 2 source contracts; Slice 3 values/repository/contribution and existing received claim/custody/configuration APIs.
**Produces:** service methods in the ledger, per-chat live preference CAS/save recovery, and frozen `prompt_mixins` on execution/custody/queued inputs. Composition/dispatch is integrated in Slice 5.

- [x] **1. Write pure order and held-admission regressions.** For placement reconciliation, construct exact values without a widget:

```python
def test_placement_edit_moves_entry_after_existing_after_entries():
    from tldw_chatbook.Chat.console_prompt_mixins import (
        MixinSourceRef, MixinSelection, MixinPreferences, MixinSnapshot,
        reconcile_mixin_placements,
    )
    from tldw_chatbook.Prompt_Management.prompt_artifact_models import LiteralMixinDefinition
    a = MixinSourceRef("profile-test", "local", "00000000-0000-4000-8000-000000000001")
    b = MixinSourceRef("profile-test", "local", "00000000-0000-4000-8000-000000000002")
    prefs = MixinPreferences(entries=(
        MixinSelection(a, True, "A", "before"),
        MixinSelection(b, False, "B", "after"),
    ))
    updated = MixinSnapshot(a, 2, "A", LiteralMixinDefinition("A body", "after"))
    reconciled = reconcile_mixin_placements(prefs, (updated,))
    assert tuple(row.source for row in reconciled.entries) == (b, a)
    assert reconciled.entries[0].enabled is False
```

For asynchronous capture, use `asyncio.Event` barriers around a source read in the existing received-turn/controller harness. Start a send in chat A, hold its read, switch to B and change B's choices, release A, and assert A's original references are used. A change to A's own pending preference revision must refuse stale preparation and preserve its draft. Do not use sleep timing as evidence.

- [x] **2. Run new nodes before implementation.** Expected: missing runtime capture/reconciliation behavior. Add changed-library-version, source swap, unavailable enabled/disabled, two sends racing, selection-save failure and stale save retry cases.
- [x] **3. Implement selection intents and bounded persistence.** Route exact session ID/incarnation and expected preference revision through the service/store. Reuse existing settings-component save status and retry UI state instead of adding a second timer. Library placement reconciliation changes only the observed entry's group and moves it last; toggles do not reorder. Closing the Workbench keeps live choices. Unknown preference schemas remain preserved/unavailable.
- [x] **4. Thread the frozen snapshot through received preparation, configuration and queue custody.** Capture selected references before first await; source reads and validation run in existing preparation workers with finite deadlines/cancellation and stale-result fences. Queue acknowledgment happens only after coherent capture; full/stale/refused admission leaves the draft. Drain and tool continuations carry the frozen value without resolving the library again. Copy serialization of body-bearing inputs only into existing private operational custody/recovery owners, never render projections. Manual/authorized queued origins may capture; machine follow-ups/goal/wake origins get explicit empty values. Do not change ordinary queue capacity or its process lifetime.
- [x] **5. Verify new admission tests plus `Tests/Chat/test_console_received_admission.py`, `test_console_turn_execution_context.py`, `test_console_prompt_queue.py`, `test_console_prompt_queue_coordinator.py` and relevant configuration worker tests.** Expected: tab/source/preference races refuse correctly; already queued items survive later toggle/edit/delete with original bodies. Update Console architecture and commit: `feat: freeze mix-ins during Console admission`.

Reviewed implementation: `32df824beb` plus fix `a0c2c4bc32`; Task4 source-witness finding cleared. Targeted evidence, complete-input/queue-edit contracts and resource qualifications are recorded in TASK-38539.

## Task 5 — Composition, clean history and resend semantics (TASK-38540)

**Files:** extend `Chat/{console_prompt_mixins,console_prompt_mixin_service,console_prepared_request,console_chat_controller,console_turn_resend,console_trace_provenance,console_trace_final_values,console_dispatch_checkpoint,console_provider_gateway,console_history_budget,console_cost_tracker}.py`, `UI/Console_Modules/{context_spend,inspector_data}.py` for preview/projection wiring and the existing agent prepared-request seam in `Chat/console_agent_bridge.py`. Create `Tests/Chat/test_console_prompt_mixin_composition.py`, `Tests/Chat/test_console_prompt_mixin_retry.py`; extend `Tests/Chat/{test_console_prepared_request,test_console_trace_current_turn_transforms,test_console_regenerate_branching,test_console_edit_resend,test_console_turn_resend}.py`.

**Consumes:** frozen attempt snapshots and transaction contributions from Slices 3/4; existing active-request ownership/provenance and resend gates.
**Produces:** `compose_mixin_content`, exactly-once verified active-envelope integration, original Retry/current regeneration and correct preview/accounting/history/trace behavior.

- [x] **1. Add failing literal, multipart and no-op tests.** The pure function never scans conversation history or interprets instruction text:

```python
def test_before_user_after_preserves_braces_and_whitespace():
    from tldw_chatbook.Chat.console_prompt_mixins import (
        MixinSourceRef, MixinSnapshot, MixinAttemptSnapshot, compose_mixin_content,
    )
    from tldw_chatbook.Prompt_Management.prompt_artifact_models import LiteralMixinDefinition
    ref = MixinSourceRef("profile-test", "local", "00000000-0000-4000-8000-000000000001")
    snap = MixinAttemptSnapshot((
        MixinSnapshot(ref, 1, "Before", LiteralMixinDefinition("  {style}\n", "before")),
    ))
    assert compose_mixin_content("message", snap) == "  {style}\n\n\nmessage"
    assert compose_mixin_content("message", MixinAttemptSnapshot()) == "message"
```

Add a distinct second source for After/many-entry cases. For multipart input, retain the exact original part sequence between added text blocks, including image URL/bytes references; verify the original list/dicts are not mutated.

- [x] **2. Add controller-boundary original/current tests before wiring.** Extend the existing `trace_harness`/controller gateway capture fixtures so adapters see the actual prepared request. Send with body A, edit library to B and disable it, fail the assistant, then Retry: adapter receives A and the transcript still holds original user text. Regenerate after enabling B: adapter receives B. Repeat with original-empty/current-enabled, capture off, DB close/reopen and deleted library source. Marked missing/corrupt snapshots must refuse before adapter entry. The test must prove an unrelated tool/history `role=user` row is never selected as the injection anchor.
- [x] **3. Implement pure composition and integrate at the common prepared-request owner after existing transforms.** String behavior is `"\n\n".join(before + [content] + after)` without stripping body content; multipart content adds text blocks around original parts. The common owner uses the exact current-user anchor, then retains a semantic applied-attempt marker outside provider wire values so agent re-preparation cannot wrap twice. Tool-loop and transport continuation reuse that prepared envelope. Do not scatter separate implementations through every send route or mutate canonical stored messages.
- [x] **4. Select original/current snapshots by action and persist before dispatch.** `retry_message`/owned failed Resend loads the message's marked attempt snapshot; regeneration, edit-and-resend and explicit new-turn Continue capture current state. A refused echo that never had an accepted attempt follows ordinary new-send capture. Stopped-empty Resend with a marked accepted attempt also uses its original snapshot; never infer policy only from display text. New attempts retain a new identity and previous-attempt link. Wire the contribution into first persistence, retry, regenerate and settlement/promotion seams before provider entry. Preserve uncertain-outcome recovery rather than retrying an ambiguous dispatch automatically.
- [x] **5. Bind trace provenance and accounting to exact prepared values.** Add a known mix-in transform/section kind through existing descriptors and final-value alignment; do not pass metadata to providers. Trace credential/PII masking applies to projections, while exact operational storage remains separate and disclosed. Next-send/cost/token projections include literal bodies; provisional preview never calls the model or executes retrieval/tools. Context overflow preserves the draft and offers existing recovery without silently dropping mix-ins. History/compaction/retrieval use clean canonical messages and ignore usage receipts.
- [x] **6. Verify new files and the five existing integration owners listed above, including plain, character, agent/tool, streaming and non-streaming paths.** Expected: correct original/current semantics after restart, no duplicate wrapping, no historical instructions/receipt leakage, no body in diagnostics, and no adapter call on snapshot-save failure. Update chat-pipeline/Console architecture, self-review and commit: `feat: compose per-send mix-ins with exact retry history`.

Reviewed implementation: `a125761794` with fixes `8c551fe41f` and `98c32189b6`; both review rounds cleared. TASK-38540 records exact targeted evidence, unchanged BASE failures, unreleased-v82 proof-column amendment and current-choice recovery boundary.

## Task 6 — Stable-history cache boundaries (TASK-38541)

**Files:** modify `Chat/{console_prepared_request,console_provider_gateway}.py`, `LLM_Calls/LLM_API_Calls.py`, canonical `Chat/Chat_Functions.py` parameter projection only if needed; create `Tests/Chat/test_console_prompt_mixin_caching.py`; extend `Tests/Chat/test_anthropic_caching_degrade.py`, `Tests/Character_Chat/test_prompt_cache_review_regressions.py` and existing cache-usage tests.

**Consumes:** exactly composed request and active/history ownership from Slice 5.
**Produces:** representation-hidden semantic boundary `clean_history_message_index: int | None` on `PreparedConsoleRequest`, derived after history windowing; a verified `ConsoleCacheBoundary(clean_history_message_index: int, semantic_structure_fingerprint: str)` value carries that boundary through provider resolution/preparation, never in raw model text. Define this small frozen value in `console_prepared_request.py`; its fingerprint is ephemeral and representation-hidden. Revalidate after provider-specific message conversion. If conversion cannot map it unambiguously, omit the optional history marker and report its unavailable status; never guess a new marker on active content. No boundary is inferred by searching user text.

- [x] **1. Add a failing canonical adapter payload test.** Extend the existing mocked HTTP adapter harness, not a duplicate cache implementation. Prepare two turns with different mix-ins and assert unchanged system/tool/clean-history content and a marker at the end of the last clean historical message. The marker must not land on the mix-in-bearing active message or an earlier text block removed by budgeting. With no eligible history boundary, emit no new history marker.

```python
def test_cache_degrade_does_not_change_mixin_instructions():
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import _without_cache_control
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "older", "cache_control": {"type": "ephemeral"}}
    ]}, {"role": "user", "content": [{"type": "text", "text": "{style}\n\nnew"}]}]
    stripped = _without_cache_control(messages)
    assert stripped[-1]["content"][0]["text"] == "{style}\n\nnew"
    assert stripped[0]["content"][0]["text"] == "older"
```

The pure degrade check supplements the required real adapter-boundary regression; it is not evidence that caching works.

- [x] **2. Run the boundary regression and observe missing/history-tail behavior.** Add cases for token-window remapping, message coalescing, attachments, no history, all mix-ins off, system/tool cache disabled, 1-hour TTL rejection/degrade, tool loops and custom OpenAI endpoints.
- [x] **3. Translate the semantic boundary only in supported adapters.** Anthropic can use at most four breakpoints: existing system, tool, clean history and optional active tail. Preserve existing configuration and degrade fallback. OpenAI's stable `prompt_cache_key` remains based on stable system/tools; do not include selected mix-ins, UUIDs or timestamps. Use explicit OpenAI history breakpoints only when actual model/endpoint support is established through existing capabilities; otherwise retain automatic caching and disclose its limits. Do not add speculative cache-option kwargs to every compatible provider.
- [x] **4. Recheck current official provider docs at implementation time.** Cache support/pricing is mutable. Prefer existing provider documentation tooling; verify minimums, limits and request fields for the exact chosen model. Mocked payload correctness proves placement, not a hit. Do not change TTLs or force writes for a test metric.
- [x] **5. Run new/affected caching and usage tests, scoped lint and provider architecture updates.** Expected: eligible stable prefixes stay identical, no unsupported endpoint fields, existing switches/degrade behavior preserved, correctness identical on a miss. Commit: `perf: preserve clean-history cache boundaries for mix-ins`.

Reviewed implementation: `817618048a` plus fix `2cc49ffc82`; exact-end mapping finding cleared. TASK-38541 records real adapter payload/usage checks, current official documentation, preserved configuration and unavailable live-hit qualification.

## Task 7 — Library editor, filters and guarded handoff (TASK-38542)

**Files:** create `Widgets/Prompts/prompt_mixin_editor.py`; modify `Widgets/Library/{library_prompts_canvas,library_prompt_work_pane}.py`, `Library/library_prompts_state.py`, `UI/Library_Modules/{library_prompts_controller,library_prompt_browse_controller}.py`, `UI/Navigation/pending_handoff_store.py` and existing sanitized Console target projection/consumer in `UI/Console_Modules/prompts.py`. Create `Tests/UI/test_library_prompt_mixins.py`; extend `Tests/Library/test_library_prompts_state.py` and `Tests/UI/test_library_prompts_reader.py`. Reuse form classes; extend source `css/features/_library_panels.tcss` only for layout not expressible by existing classes.

**Consumes:** typed records/source support, runtime service selection intents and existing exact-target/one-shot Prompt handoff machinery.
**Produces:** a simple `PromptMixinEditor` and explicit Mix-in type filtering; a dedicated typed enable intent, not a normal Prompt-application payload.

`PromptMixinEditor(name: str, definition: LiteralMixinDefinition, **widget_kwargs)` exposes `read_name() -> str` and `read_definition() -> LiteralMixinDefinition`, with stable field IDs `mixin-name`, `mixin-body`, `mixin-placement`. Its events request save/cancel; it never calls DB/provider/runtime. Parent controllers retain source identity, expected library version, dirty state and save authority. Existing metadata fields remain parent-owned.

- [x] **1. Add failing literal editor and type-filter UI tests.** Use the production CSS and existing UI harness lifecycle/cleanup; this small Harness example defines its own mounted surface:

```python
from textual.app import App, ComposeResult
from textual.widgets import TextArea
from tldw_chatbook.Prompt_Management.prompt_artifact_models import LiteralMixinDefinition
from tldw_chatbook.Widgets.Prompts.prompt_mixin_editor import PromptMixinEditor

class EditorHarness(App):
    def compose(self) -> ComposeResult:
        yield PromptMixinEditor(name="Scene", definition=LiteralMixinDefinition("{pose}\n", "after"))

async def test_editor_preserves_literal_body():
    app = EditorHarness()
    async with app.run_test(size=(80, 24)):
        editor = app.query_one(PromptMixinEditor)
        assert editor.query_one("#mixin-body", TextArea).text == "{pose}\n"
        assert editor.read_definition().body == "{pose}\n"
```

The integrated Library harness must additionally cover real save and conditional conflict, create/edit/dirty Escape, source switches and filtered pagination totals.

- [x] **2. Run the new nodes before implementation.** Expected: absent editor/filter/enable intent. Add absent Console target, expired handoff, tab switch and newer handoff supersession cases.
- [x] **3. Implement the small editor and reuse Library CRUD/version/dirty-save owners.** Use one literal TextArea and existing Before/After form choice, not the block editor or parameter dialog. Positive artifact-type dispatch chooses this editor. Search/type filter count and page together; render unavailable source capabilities truthfully. User body braces and section-like text remain literal across save/export/reopen.
- [x] **4. Add guarded Enable in current chat.** Extend the existing pending handoff family with a source-qualified `ConsoleMixinEnableRequest` holding the validated reference, target session/incarnation, preference revision and existing expiry. No instruction body or executable grant enters routing metadata. The Console consumer resolves/validates via the runtime service, appends at the appropriate group's end, enables an existing entry without duplicating/reordering it, and never changes composer/System. Preserve one-shot claim/ack/release and stale-target refusal. Retain the existing open-Console remedy when there is no valid target.
- [x] **5. Verify the new UI file, Library state/reader, existing handoff regressions and Markdown/Chatbook paths from Slice 2.** Expected: mix-in-specific editor/enable behavior with unchanged normal Prompt/Recipe flows. Run token/bundle checks if CSS changes, update Library guide and commit: `feat: author and enable library prompt mix-ins`.

Reviewed implementation: `398e79d50b` plus fix `f0e1d640c4`; invalid-action recovery finding cleared, bounded native finish approved. TASK-38542 retains exact targeted evidence, source/capture manifest, source-fence and blank-draft interface amendments and inherited qualifications.

## Task 8 — Existing Prompts Workbench controls and inspection (TASK-38543)

**Files:** create `Widgets/Console/console_prompt_mixins_view.py`; modify `Widgets/Console/{console_prompts_modal,console_prompts_browse,console_prompts_state,console_composer_menu_modal}.py`, existing composer Prompts renderer/action in `Widgets/Console/console_composer_bar.py` if it owns that renderer, `UI/Console_Modules/prompts.py`, existing Next send/selected-turn inspection projection and message-action copy. Modify `css/features/_console_panels.tcss` only if needed, then rebuild. Create `Tests/UI/test_console_prompt_mixins.py`; extend `Tests/UI/{test_console_prompts_modal,test_console_composer_menu}.py` and existing Next send/retry-inspection tests.

**Consumes:** Library editor/enable path and runtime service/attempt inspection values. Does not read source DBs or schedule sends directly.
**Produces:** `ConsolePromptMixinsView` inside the existing Workbench, typed intents for exact-origin preference changes and body-free count/recovery projections. Add mode `mixins` to the existing `ConsolePromptsState` navigation/focus contract. Reuse the shared editor from Slice 7.

At this baseline the composer's existing Prompt entry is `ACTION_PROMPTS` in `console_composer_menu_modal.py`, labeled `Browse Prompt Library…`, consumed by the existing Console prompt controller. Reuse that action and any existing Prompts button renderer at execution. Do not add a second standalone button to reconcile the user's wording. Keep ordinary browse/edit/improve navigation available in the same Workbench; Mix-ins is an additional section, not a replacement prompt picker.

- [x] **1. Add failing mounted Workbench interactions.** With existing `Tests/UI/test_console_prompts_modal.py` setup, open the actual Prompts action, enter Mix-ins, add one Before and two After entries, disable the first After, move the second above it, close/reopen, and assert selected order/enabled flags in the real store. Show source-sorted library results in the opposite order and assert selection order is unchanged. Switch chats while a captured Workbench transaction is open and verify it cannot mutate the newly viewed chat.

```python
# Assertions after the real controls emit their intents into the runtime owner:
prefs = mixin_service.preferences(origin_session_id)
assert [entry.name for entry in prefs.entries] == ["Style", "Diagram", "Summary"]
assert [entry.enabled for entry in prefs.entries] == [True, True, False]
```

Here `mixin_service` and `origin_session_id` come from the mounted production runtime setup added to that UI test, not a widget-local replacement store. Include keyboard Tab/Space/Enter and Move up/down buttons, and wait on actual worker completion signals rather than arbitrary sleeps.

- [x] **2. Run new UI nodes and observe missing section/count/intents.** Add failure cases for deleted enabled/disabled records, updated placement/body, live-unsaved save failure and retry, dirty editor close, original nonempty/empty Retry labels, and missing historical snapshots. Counts represent enabled selections even when an enabled entry is unavailable; the unavailable warning remains visible.
- [x] **3. Render Before/After rows from runtime projections and route typed intents.** Add, toggle, move, remove and inspect actions carry originating session/incarnation and preference revision. Disabled rows retain position. Remove only removes selection; inspect/edit delegates to existing editor/source authority. Close/Escape retains selection changes and restores opener focus. Library edit is separately saved/discarded. Do not attach new global/terminal-convention keybindings.
- [x] **4. Integrate count, preview and historical inspection.** Update the existing Prompts action/renderer and accessible copy; narrow menu access remains the same route. Next send uses Slice 5's exact composition projection and shows update/unavailable/unsaved states. Historical usage expands stored snapshots, never current library bodies. Nonempty Retry reads `Retry with original mix-ins` and identifies its captured set; missing snapshots offer explicit current-choice regeneration. No usage receipt is inserted into transcript text.
- [x] **5. Verify at 160x45, 100x30 and 80x24 using production styles, mouse and keyboard.** Run the three UI files plus applicable Next send/message action tests; run `Tests/UI/test_design_token_governance.py` and `Tests/UI/test_css_bundle_sync_guard.py` if styles changed. Rebuild via `.venv/bin/python tldw_chatbook/css/build_css.py`, never edit generated CSS. Update the Console guide and commit: `feat: manage chat mix-ins in the Prompt Workbench`.

## Task 9 — Integrated qualification and documentation (TASK-38544)

**Files:** create `Tests/Chat/test_console_prompt_mixin_journey.py` and a focused QA evidence document `Docs/superpowers/qa/2026-11-01-prompt-mix-ins/verification.md`; extend `Tests/Backup_Recovery/test_prompt_mixin_restore.py` and the Slice 8 UI journeys only for demonstrated integration gaps. Update `Docs/Architecture/{chat-pipeline,console,library-shell,database-layer,llm-providers}.md`, `Docs/User_Guide/console.md`, relevant existing prompt-library guide, task implementation notes and this plan's completed-step/evidence state. No new test app or framework.

**Consumes:** all earlier slices; the exact integrated source hashes, existing isolated profile fixtures and production runtime/DB/provider boundaries.
**Produces:** honest automated/live evidence and user documentation of shipped behavior; no additional functionality.

- [x] **1. Add one production-owner integration journey with deterministic provider responses.** Use existing isolated bootstrap/profile fixtures and the adapter-entry recording pattern in `test_console_trace_current_turn_transforms.py`. Explicitly set capture off; create an ordinary and a character chat; use actual local PromptScopeService, store, controller and gateway. Exercise queue A, edit library to B, switch tabs, drain A, toggle off, Retry A, regenerate off, re-enable B, regenerate B and reopen. Assert adapter inputs, clean stored messages, separate attempt snapshots and expected source versions after each action.

```text
Accepted queued send -> captured A
Library edited B / selection off -> queued dispatch still A
Retry that failed dispatch -> original A, with original label
Regenerate while off -> no mix-in
Regenerate after re-enable -> B
Next independent message -> no historical A/B injection
Reopen saved chat -> selected references/states/order restored
New/copy/fork chat -> no active selection; inherited receipts inspect correctly
```

Do not infer success solely from mock calls: assert the persisted receipt anchors and captured provider semantic request agree. Extend the journey with stopped-empty Resend, explicit Continue, missing new-format snapshot and a body-bearing tool successor. Force snapshot-save failure and assert zero adapter entries; force post-dispatch settlement failure and verify existing uncertain recovery, not blind resend.

- [x] **2. Run only the changed slice modules and their named predecessor regressions.** Commands are the per-slice node/module commands above, not `pytest` on the full repository. Run DB restore/promotion/fork and migration-file packaging checks on disposable databases. Run scoped formatter/linter using the repository's configured tools; if an existing diagnostic is present, compare the same file/node against the exact pre-change baseline and report it. Do not mark the task Done on a narrower replacement harness that bypasses a failed native owner.
- [x] **3. Live-check the actual app with isolated data and production styles.** Create a private `mktemp -d` profile under the platform temporary directory, not a sticky `/tmp` ancestor. Launch through the existing isolated app/served QA launcher with child-only profile/config/data environment overrides; preserve the user's real shell environment and data. Record launch command, exact source hash and screenshot locations. Exercise existing Prompts action, keyboard toggles/order, literal scene instruction, preview, queue, Retry/regenerate, reopen and visible source/save failures at the three sizes from Slice 8. Verify teardown retires borrowed runtimes, DBs and clients; do not wait for unrelated perpetual workers.
- [x] **4. Qualify caches only with available real credentials and usage.** Use a stable tools/system/history prefix long enough for the exact model's documented cache minimum; compare actual cache-read/write counters across an unchanged turn, toggle and library edit. Record provider/model/settings and TTL/kill-switch state without credentials or bodies in logs. If credentials/capabilities are absent, retain deterministic payload tests and explicitly state live cache qualification is unavailable; do not claim savings. SVG/HTML is verified as instructions/text unless an existing renderer or permitted tool was actually exercised. No new rendering or automatic generation is introduced.
- [x] **5. Close documentation and task hygiene with evidence.** The QA record lists exact targeted commands/results, executed versus blocked journeys, migrations/restore checks, UI captures, live provider evidence and remaining limitations. All tasks require checked AC, implementation notes, scoped lint/format and ADR-286 links before CLI Done. If a runtime acceptance criterion is blocked, leave that task open and report the missing evidence. Add a lessons entry only for a demonstrated generalizable incident. Commit the qualification/docs changes: `test: qualify the integrated prompt mix-in lifecycle`.


Task9 implementation evidence (2026-11-01–02): [durable QA](../qa/2026-11-01-prompt-mix-ins/verification.md). Historical feature249, predecessor540 and packaging1 pass; final covering51 and fork14 pass (overlapping counts). Actual isolated UI/loopback and current Preview checks are source-qualified. External cache proof remains unavailable. Original-base attribution was narrowly extended for archive fixture and sparse-policy cases. All steps and final scoped re-review are complete; TASK-38544 is reopened for current-dev integration and fresh independent review. Disclosed limitations remain in QA.

## Spec coverage and handoff checklist

| Approved spec section | Delivery owner |
| --- | --- |
| 1: literal instruction-only scope and exclusions | All slices; 1 positive type validation, 5 origin/permission boundaries, 9 release check |
| 2: existing owners and no duplicate subsystem | File map; 3 existing transaction writer, 4 existing runtime, 7/8 existing surfaces |
| 3: artifact, source, import/export compatibility | 1 and 2; 7 editor/filter/guarded enable |
| 4: existing Prompts control, Before/After order, keyboard | 4 pure/live state; 7/8 UI and exact-origin handoff |
| 5: library edits, queue, reopen, new/copy/fork, retry/current | 3 storage; 4 capture; 5 action selection; 8 labels; 9 integrated journey |
| 6: current envelope, clean transcript/history, multipart, preview | 5 and 9 |
| 7: exact capture-off operational storage, privacy, backup/fork/purge | 3 and 5; 8 inspection; 9 restore/reopen evidence |
| 8: clean-prefix caching and provider policy | 6 and live-evidence limits in 9 |
| 9: failures, draft preservation, stale and unavailable state | 1/2 boundary checks; 3 durable rollback; 4 admission; 5 dispatch; 7/8 recovery |
| 10: targeted automated/live verification | Every slice's regressions; 9 integrated qualification |
| 11: ADR and atomic implementation handoff | Existing Backlog records and per-slice review gates |

- [x] Before marking a slice complete, verify changed callers/exports/source limits and run its targeted regression gates on the final tree.
- [x] Recheck public type/signature names against this ledger; update consumers and tests in the same slice if a reviewed contract must change.
- [x] Do not introduce implementation scope absent from the approved spec. Revise AC/spec/ADR first for a material boundary change.
- [x] Chosen execution mode: user-authorized subagent-driven slices with independent review between slices.
