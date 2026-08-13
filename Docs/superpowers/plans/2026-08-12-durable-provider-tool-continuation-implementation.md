# Durable Provider Tool-Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the minimum private provider context needed to resume interrupted native function-tool runs safely and to replay provider-required reasoning without exposing it in the visible transcript or re-executing completed side effects.

**Architecture:** Add one validated, bounded `provider_continuation_json` field to the existing assistant message/variant row and include it in the same mutation, version, trigger-intent, sync, branch, deletion, and export ownership as that row. Keep a pure canonical checkpoint module between provider adapters and the agent runtime. The runtime reports lifecycle transitions through one typed callback; the Console store owns transactions, Sync-v2 projection, interrupted-state UI, and explicit Resume/Discard/Take over. The foundation exposes canonical owner groups only; each provider adapter translates those groups to its own wire history. The existing history budget retains or evicts each visible owner and its canonical private rounds as one unit.

**Tech Stack:** Python 3.11+, SQLite/FTS5, existing ChaChaNotes schema migrations and trigger-backed `sync_log`, Sync v2 encrypted outbox, Textual 8.x Console UI, pytest/pytest-asyncio/Hypothesis, existing agent runtime and `.chatbook` import/export system.

---

## Design Sources And ADR Check

- Approved design: `Docs/superpowers/specs/2026-08-12-durable-provider-tool-continuation-design.md`
- Canonical decision: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`
- Backlog source of truth: `backlog/tasks/task-15675 - Add-durable-provider-tool-continuation-checkpoints.md`
- Related decisions: ADR-006, ADR-012, ADR-020, ADR-026, and ADR-062.

ADR required: yes

ADR path: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`

Reason: This changes the durable message schema, sync/export contracts, provider/runtime handoff, and side-effect recovery policy. ADR-063 already records the approved boundary, so no additional ADR is needed.

## Scope Guardrails

- Add one nullable message column, not a provider-state table or open metadata bag.
- Store canonical validated fields only; never store credentials, headers, raw provider bodies, arbitrary response items, logs, usage, or UI copy.
- Do not create a second agent loop or let provider adapters execute tools.
- Never execute on open/import/sync. Only an explicit Resume action may move a restored `pending` call toward fresh approval and execution.
- Treat restored `executing` calls as ambiguous and blocked. Never infer that they failed or retry them automatically.
- Keep Sync v1 source intent in ChaChaNotes and Sync v2 outbox state in its existing separate repository. Do not collapse the databases or require remote acknowledgement before a local side effect.
- Preserve visible-message behavior for conversations with no checkpoint. Legacy exports/imports remain readable.
- Do not expose private continuation in FTS, rendering, ordinary summaries, logs, errors, usage, text/Markdown exports, or whole-dict JSON serialization.
- Reuse the existing history budget and branch/variant ownership; do not add a second context manager.

## Branch And Baseline Discipline

- [x] Before implementation, fetch and rebase this documentation branch onto the then-current `origin/dev`:

  ```bash
  git fetch origin
  git rebase origin/dev
  ```

- [x] Record the rebased SHA and run identical clean-base and feature-branch baselines for the DB, runtime, sync, Console, and Chatbook suites named below. Localhost-bind failures must be rerun outside the managed socket sandbox; do not count a sandbox `PermissionError` as product evidence.
- [x] Verify TASK-15675 is already In Progress and that its structured Implementation Plan links this document and ADR-063 before production edits:

  ```bash
  backlog task 15675 --plain
  ```

- [x] For every cycle: add the focused test, run it RED for the intended missing behavior, make the smallest production change, rerun GREEN, then run the task regression set. Never commit a red state.

## Canonical Interfaces To Implement

Create a pure `tldw_chatbook/Chat/provider_continuation.py` module. Keep provider-specific wire fields out of it.

```python
ContinuationProvider = Literal["moonshot", "zai", "deepseek"]
ContinuationProtocol = Literal["chat_completions", "responses"]
ContinuationState = Literal["active", "complete"]
ContinuationCallState = Literal[
    "pending", "executing", "completed", "failed"
]

class ContinuationValidationError(ValueError): ...

@dataclass(frozen=True)
class ProviderContinuationCheckpoint:
    schema_version: Literal[1]
    checkpoint_revision: int
    provider: ContinuationProvider
    protocol: ContinuationProtocol
    model: str
    api_base_url: str
    state: ContinuationState
    rounds: tuple[ContinuationRound, ...]

def parse_provider_continuation_json(
    value: object,
) -> ProviderContinuationCheckpoint: ...

@dataclass(frozen=True)
class SafeContinuationRead:
    checkpoint: ProviderContinuationCheckpoint | None
    warning: str | None = None

@dataclass(frozen=True)
class ContinuationOwnerGroup:
    owner_message_id: str
    checkpoint: ProviderContinuationCheckpoint
    rounds: tuple[ContinuationRound, ...]

@dataclass(frozen=True)
class ContinuationRestoreTarget:
    provider: str
    model: str
    protocol: str
    api_base_url: str

def validate_continuation_restore(
    checkpoint: ProviderContinuationCheckpoint,
    target: ContinuationRestoreTarget,
) -> None: ...

def read_provider_continuation_json(value: object) -> SafeContinuationRead: ...

def dump_provider_continuation_json(
    checkpoint: ProviderContinuationCheckpoint | None,
) -> str | None: ...

def transition_provider_call(
    checkpoint: ProviderContinuationCheckpoint,
    *,
    call_id: str,
    expected_revision: int,
    target: ContinuationCallState,
    result: ContinuationResult | None = None,
) -> ProviderContinuationCheckpoint: ...

def continuation_owner_group(
    visible_message: Mapping[str, Any],
    checkpoint: ProviderContinuationCheckpoint | None,
) -> ContinuationOwnerGroup: ...
```

`parse_provider_continuation_json` is the strict mutation boundary: it enforces the exact V1 schema, provider/protocol pairing, call-ID uniqueness across all rounds, legal state/result combinations, round ordering, reasoning-only exception, JSON depth/node/string/count/total-byte bounds, and raises a context-free `ContinuationValidationError` for every invalid value. `read_provider_continuation_json` is the only tolerant read/import wrapper: it converts invalid or unknown-version private data to `checkpoint=None` plus one bounded safe warning while leaving visible content usable. `validate_continuation_restore` compares provider, model, protocol/API mode, and normalized base exactly before Resume/Take over or private-group exposure. Neither helper renders provider wire rows. Mutation APIs raise typed validation/conflict errors before writing.

### Task 1: Implement The Pure Canonical Checkpoint And Owner Group

**Files:**

- Create: `tldw_chatbook/Chat/provider_continuation.py`
- Create: `Tests/Chat/test_provider_continuation.py`
- Modify: `Tests/Chat/test_sensitive_llm_logging.py`

- [x] **Cycle 1A — schema RED:** add fixtures for one active tool round, a completed/failed result round, and the Kimi K3 complete reasoning-only final round. Add a parameter matrix rejecting unknown/extra keys, wrong scalar types, non-finite numbers, unsupported provider/protocol pairs, duplicate call IDs across rounds, missing results, results on pending/executing calls, empty tool rounds outside the K3 exception, and malformed exact JSON arguments.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_provider_continuation.py -k "schema or invalid or bounds"
  ```

  Expected RED: module/import is absent. Implement frozen dataclasses, exact-key validation, iterative depth/node accounting, immutable copies, safe typed errors, and canonical JSON serialization; rerun GREEN.
- [x] **Cycle 1B — transitions RED:** add `test_call_transition_state_machine_is_revision_checked` covering only `pending→executing→completed|failed`, idempotent exact replay of the same terminal transition, stale revision rejection, and every illegal transition. Implement `transition_provider_call`; rerun GREEN.
- [x] **Cycle 1C — restore-target/owner-group RED:** add a frozen target fixture and reject an exact mismatch in provider, model, protocol/API mode, or normalized base before Resume/Take over/private-group exposure. Then bind the validated checkpoint to exactly one visible assistant owner and return canonical immutable rounds without producing Chat- or Responses-shaped rows. Keep repeated function names with unique call IDs valid and expose the owner/group key needed by the existing history budget. Provider-specific replay eligibility and wire translation belong to TASK-15676/15677.
- [x] **Cycle 1D — privacy/mutation RED:** assert input mappings/lists are unchanged, canary credentials/raw provider bodies never appear in serialized data, exception text, captured logs, `repr`, or traceback chains, and every collection/string limit fails before allocation grows unbounded.
- [x] Run full focused tests and static checks:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_provider_continuation.py Tests/Chat/test_sensitive_llm_logging.py -k "continuation or sensitive"
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Chat/provider_continuation.py Tests/Chat/test_provider_continuation.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/Chat/provider_continuation.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/Chat/provider_continuation.py Tests/Chat/test_provider_continuation.py Tests/Chat/test_sensitive_llm_logging.py
  git commit -m "feat(chat): define durable provider continuation"
  ```

### Task 2: Add Schema V36 And Atomic Message Persistence

**Files:**

- Create: `tldw_chatbook/DB/migrations/chachanotes_v36_to_v37_provider_continuation.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py`
- Create: `Tests/DB/test_chachanotes_provider_continuation_migration.py`
- Create: `Tests/ChaChaNotesDB/test_provider_continuation.py`
- Modify: `Tests/Chat/test_chat_conversation_service.py`

- [x] **Cycle 2A — migration RED:** create a v35 fixture DB with message variants, deleted rows, metadata, sync triggers, and FTS rows. Assert migration 35→36 adds nullable `provider_continuation_json`, preserves every row/trigger/index, and leaves legacy values NULL. Run the migration node RED, add migration registration at schema version 36, rerun GREEN.
- [x] Recreate the message sync triggers in the migration so create/update payloads and their change predicate include `provider_continuation_json`; do not add it to FTS content or local-only metadata paths.
- [x] **Cycle 2B — CRUD RED:** add round-trip tests through `add_message`, `get_message_by_id`, conversation reads, tree/branch reads, and `normalize_message_row`. Permit blank assistant content only when a valid checkpoint is supplied; continue rejecting an otherwise blank message.
- [x] Add the column to every explicit message projection and insert/update statement. Do not use `SELECT *` or append arbitrary caller dictionaries.
- [x] **Cycle 2C — transaction RED:** add a crash-injection test around a new dedicated API:

  ```python
  def create_assistant_with_continuation(
      self,
      *,
      message_id: str,
      conversation_id: str,
      parent_message_id: str | None,
      content: str,
      provider_continuation_json: str,
      expected_conversation_version: int | None = None,
  ) -> str: ...

  def update_provider_continuation(
      self,
      *,
      message_id: str,
      expected_message_version: int,
      provider_continuation_json: str | None,
      content: str | None = None,
      deleted: bool | None = None,
  ) -> bool: ...
  ```

  Assert the caller-preallocated UUID is preserved exactly and that `add_message`-compatible failure semantics remain `str | None` at the lower boundary. The assistant row and trigger-written `sync_log` intent appear together or neither appears; stale versions change neither. Use the existing DB transaction/context and message version/hash path.
- [x] **Cycle 2D — discard semantics RED:** prove whole-message optimistic transactions implement: blank checkpoint-created row → clear checkpoint and tombstone; visible row → clear checkpoint and keep content; both bump version/hash and produce exactly one sync intent. Deletion/edit/regenerate/variant selection must never re-parent the checkpoint to a sibling.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/DB/test_chachanotes_provider_continuation_migration.py Tests/ChaChaNotesDB/test_provider_continuation.py Tests/Chat/test_chat_conversation_service.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/DB/migrations/chachanotes_v36_to_v37_provider_continuation.sql tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Chat/chat_conversation_service.py Tests/DB/test_chachanotes_provider_continuation_migration.py Tests/ChaChaNotesDB/test_provider_continuation.py Tests/Chat/test_chat_conversation_service.py
  git commit -m "feat(db): persist provider continuation on messages"
  ```

### Task 3: Extend Sync V1 Payloads And Reconcile Durable Sync V2 Intent

**Files:**

- Modify: `tldw_chatbook/Sync_Interop/envelope_builder.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_applier.py`
- Modify: `tldw_chatbook/Sync_Interop/chat_outbox_producer.py`
- Modify: `tldw_chatbook/Sync_Interop/sync_state_repository.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `Tests/Sync_Interop/test_envelope_builder.py`
- Modify: `Tests/Sync_Interop/test_envelope_applier.py`
- Modify: `Tests/Sync_Interop/test_chat_outbox_producer.py`
- Modify: `Tests/Sync_Interop/test_sync_state_repository.py`
- Create: `Tests/Sync_Interop/test_provider_continuation_reconciliation.py`

- [x] **Cycle 3A — envelope RED:** extend chat envelope tests so encrypted payload contains `provider_continuation_json` beside role/content, while clear routing metadata and logs contain no private data. Invalid private data is dropped safely on apply while visible content still applies; valid data attaches only to the same stable message/variant.
- [x] Add an optional canonical serialized field to `SyncEnvelopeBuilder.build_chat_message` and its applier. Do not put private data in routing metadata.
- [x] **Cycle 3B — source reader RED:** add a narrow `ChatSyncIntentSource` protocol to `ChatSyncV2OutboxProducer` and implement it on `ChaChaNotes_DB`. It reads an already-committed message sync intent/snapshot only when stable message ID, message version, and payload hash all match; it never accepts a caller-reconstructed payload as proof.
- [x] **Cycle 3C — atomic outbox receipt/schema RED:** bump `SYNC_STATE_SCHEMA_VERSION` from 3 to 4 and add an existing-v3 reopen/upgrade fixture proving every prior table/row remains. Add `sync_v2_source_projection_receipts` to `SyncStateRepository`, keyed by `(source_scope_key, dataset_id, domain, source_entity_id, source_version, source_payload_hash)` and storing the resulting `client_envelope_id`. Add one repository transaction that inserts/upserts the existing outbox envelope and its receipt together. The receipt also carries/validates server profile, authenticated principal, and workspace scope through `source_scope_key`. Two reconciliations of one source version/hash produce one outbox row and one receipt; a later version/hash creates a distinct envelope/receipt because the existing `client_envelope_id` includes the payload hash—it does not replace the earlier version.
- [x] Keep the ChaChaNotes source intent; this task does not invent a separate delete/ack protocol for `sync_log`. The atomic receipt is the bridge acknowledgement. Add crash tests for: source commit before projection, process death after atomic outbox+receipt before the producer returns, and restart reconciliation. On restart the retained source plus exact receipt is idempotent; a receipt without its referenced outbox row is invalid and blocks execution.
- [x] **Cycle 3D — side-effect barrier RED:** parameterize Sync v2 absent, configured with durable repository, configured with in-memory repository, and configured-but-unavailable. Before a pending tool transitions to `executing`, require the exact same-transaction ChaChaNotes intent and, when Sync v2 is configured, an exact scoped receipt whose `client_envelope_id` resolves to the durable outbox row for the same message ID/version/payload hash. Unconfigured sync proceeds locally; configured durable projection proceeds without waiting for remote ack; memory/unavailable/mismatched scope or hash blocks with safe actionable copy.
- [x] Add a narrow result such as:

  ```python
  @dataclass(frozen=True)
  class ContinuationDurabilityResult:
      ready: bool
      reason: str = ""

  def ensure_provider_continuation_durable(
      self,
      *,
      message_id: str,
      message_version: int,
      payload_hash: str,
  ) -> ContinuationDurabilityResult: ...
  ```

  Keep repository-type/durability knowledge at the producer/store boundary, not in `agent_runtime`.
- [x] **Cycle 3E — conflict/clear RED:** test two-device whole-record conflicts, Discard propagation, deletion, branch variants, and a later visible edit. No field-level merge; winning message version owns content plus checkpoint as one record.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_envelope_builder.py Tests/Sync_Interop/test_envelope_applier.py Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Sync_Interop/test_sync_state_repository.py Tests/Sync_Interop/test_provider_continuation_reconciliation.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/Sync_Interop/envelope_builder.py tldw_chatbook/Sync_Interop/envelope_applier.py tldw_chatbook/Sync_Interop/chat_outbox_producer.py tldw_chatbook/Sync_Interop/sync_state_repository.py tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Chat/console_chat_store.py Tests/Sync_Interop/test_envelope_builder.py Tests/Sync_Interop/test_envelope_applier.py Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Sync_Interop/test_sync_state_repository.py Tests/Sync_Interop/test_provider_continuation_reconciliation.py
  git commit -m "feat(sync): reconcile provider continuation intent"
  ```

### Task 4: Add Runtime Lifecycle Hooks Before Any Tool Side Effect

**Files:**

- Modify: `tldw_chatbook/Agents/agent_models.py`
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `Tests/Agents/test_agent_runtime.py`
- Modify: `Tests/Agents/test_agent_service.py`
- Create: `Tests/Agents/test_provider_continuation_runtime.py`

- [x] Extend `ModelTurn` with a typed optional checkpoint candidate, not a free-form metadata dictionary:

  ```python
  @dataclass(frozen=True)
  class ModelTurn:
      text: str = ""
      tool_calls: tuple[ToolCall, ...] = ()
      assistant_message: dict | None = None
      tokens: int = 0
      provider_continuation: ProviderContinuationCheckpoint | None = None
  ```

- [x] Add one narrow typed `LoopDeps` callback whose default preserves existing callers. Model the payload as this exact closed frozen union rather than a string kind or metadata dictionary:

  ```python
  @dataclass(frozen=True)
  class ContinuationEventContext:
      owner_message_id: str | None
      run_id: str
      agent_kind: Literal["primary", "subagent", "fleet"]
      durability: Literal["persistent", "ephemeral"]

  @dataclass(frozen=True)
  class ToolBatchReady:
      context: ContinuationEventContext
      checkpoint: ProviderContinuationCheckpoint
      expected_checkpoint_revision: int | None

  @dataclass(frozen=True)
  class ToolCallExecuting:
      context: ContinuationEventContext
      call_id: str
      expected_checkpoint_revision: int

  @dataclass(frozen=True)
  class ToolCallFinished:
      context: ContinuationEventContext
      call_id: str
      expected_checkpoint_revision: int
      target_state: Literal["completed", "failed"]
      result: ContinuationResult

  @dataclass(frozen=True)
  class FinalContinuation:
      context: ContinuationEventContext
      checkpoint: ProviderContinuationCheckpoint
      expected_checkpoint_revision: int | None
      assistant_content: str

  ProviderContinuationEvent = (
      ToolBatchReady | ToolCallExecuting | ToolCallFinished | FinalContinuation
  )

  persist_provider_continuation: Callable[
      [ProviderContinuationEvent], None
  ] = lambda event: None
  ```

  `expected_checkpoint_revision=None` is valid only for the first `ToolBatchReady` create or the first `FinalContinuation` creation of the approved tool-free Kimi K3 complete reasoning-only checkpoint on the preallocated owner. Every other update supplies the exact current revision and conflicts rather than merging. `FinalContinuation.assistant_content` is atomically checked/written with the complete checkpoint. Persistent primary events must carry and match the run's preallocated assistant owner. Persistent subagent/fleet tool batches fail closed before side effects until they have a distinct durable assistant owner; they never overwrite the primary owner's checkpoint. Explicitly ephemeral child runs may carry no owner, continue with in-memory continuation only, and are labelled non-resumable.

- [x] **Cycle 4A — ordering RED:** record callback-event/dispatch order and assert `ToolBatchReady` succeeds once before review/approval or any invocation, then `ToolCallExecuting` succeeds immediately before the actual call, and `ToolCallFinished` succeeds before the result is appended and before the next model request. A raised persistence callback must execute zero subsequent tools and return a safe non-success outcome.
- [x] **Cycle 4B — multi-call/cancel/ownership RED:** two-call batches persist the complete batch once, then transition each call independently. Cancellation after a partial streamed call executes zero tools; cancellation after call A completes never repeats A and leaves B pending. Duplicate call IDs are rejected before persistence or execution. Concurrent primary/subagent/fleet events cannot cross-write owners; persistent child calls without their own owner stop before execution, while explicitly ephemeral child runs stay in memory and are reported non-resumable.
- [x] **Cycle 4C — restore RED:** feed restored completed/failed/executing/pending checkpoints through an explicit resume entry. Completed/failed results are replayed only; executing blocks as ambiguous; pending pauses until Resume and then traverses the ordinary fresh approval hook before execution. Open/import/sync constructors do not call the loop.
- [x] **Cycle 4D — final reasoning create/update RED:** a tool-free Kimi K3 final `ModelTurn` emits `FinalContinuation(expected_checkpoint_revision=None)` before `RUN_DONE` and atomically creates the sole complete reasoning-only checkpoint on the preallocated assistant owner without a duplicate visible row. A post-tool K3 final round supplies the exact existing revision and appends/updates the same owner. Reject `None` for every other provider/state or when a checkpoint already exists; reject stale revisions. Other provider policies persist only their documented states.
- [x] Mutation guard: temporarily swap the callback order so invocation precedes `ToolCallExecuting`; the ordering test must fail. Restore and rerun GREEN.
- [x] Run runtime regressions including review-hook, cancellation, cycle detection, fleets, and native tools:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Agents/test_provider_continuation_runtime.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_agent_service.py Tests/Agents/test_native_tools.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_provider_continuation_runtime.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_service.py
  git commit -m "feat(agents): persist native tool lifecycle barriers"
  ```

### Task 5: Bind Checkpoints To Console Messages And Explicit Recovery UI

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/Chat/test_console_provider_continuation.py`
- Create: `Tests/UI/test_console_provider_continuation_recovery.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`

- [x] Add a typed `provider_continuation` field to `ConsoleChatMessage`; read/write it through the dedicated persistence APIs. Do not place it in `MessageMetadata`, rendering content, `tool_output_full`, or usage.
- [x] **Cycle 5A — forced owner RED:** when the first complete native tool batch arrives with empty visible content, assert the store force-creates one assistant row with empty content and the active checkpoint in the same DB transaction before any tool invocation. Later stream text updates that same row.
- [x] Wire the runtime hooks from Task 4 through `ConsoleAgentBridge` to the store. Freeze provider/model/protocol/normalized base in the first checkpoint; Resume resolves the current credential through ordinary readiness but may not change those frozen fields.
- [x] **Cycle 5B — restore UI RED:** load active checkpoints and assert the message renders an interrupted private-state affordance without reasoning, arguments, results, or raw IDs. Actions are `Resume` and `Discard`; opening the screen, selecting the conversation, syncing, importing, and switching variants execute zero tools.
- [x] **Cycle 5C — Resume RED:** pending calls require explicit Resume and then the existing approval path. Restored executing calls show ambiguous-state recovery and remain blocked. Completed/failed calls are replayed to the provider and never dispatched.
- [x] **Cycle 5D — Discard RED:** invoke the optimistic whole-message transaction. Blank checkpoint-only owners become tombstoned; visible owners retain content. The active path, sibling counts, selected variant, and persisted message IDs remain correct. A stale concurrent edit yields a conflict and does not partially clear.
- [x] **Cycle 5E — new-turn guard RED:** sending a new user message while an interrupted checkpoint exists never resumes, takes over, discards, or reassigns that checkpoint implicitly. Pin the exact blocking/recovery copy and prove zero tool execution.
- [x] **Cycle 5F — durability mode RED:** an explicitly selected ephemeral conversation may continue in memory after checkpoint persistence is unavailable, visibly labels the run non-resumable, writes no durable checkpoint, and cannot later present Resume. Persistent conversations fail closed at the same boundary.
- [x] **Cycle 5G — cross-device Take over RED:** a checkpoint marked as remotely active exposes a distinct `Take over` action, not ordinary Resume and not “send another message.” It warns that another device may still be running, repeats the same frozen-field/current-credential/conflict checks, requires fresh approval, and makes no distributed exactly-once claim. Merely typing/sending a new user message executes zero interrupted work.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_continuation.py Tests/UI/test_console_provider_continuation_recovery.py Tests/Chat/test_console_agent_bridge.py Tests/UI/test_console_native_chat_flow.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_provider_continuation.py Tests/UI/test_console_provider_continuation_recovery.py Tests/Chat/test_console_agent_bridge.py
  git commit -m "feat(console): resume durable provider tool runs"
  ```

### Task 6: Expand And Budget Provider History Atomically

**Files:**

- Modify: `tldw_chatbook/Chat/console_history_budget.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Agents/run_log_eviction.py`
- Modify: `Tests/Chat/test_console_history_budget.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/Agents/test_run_log_eviction.py`
- Create: `Tests/Chat/test_provider_continuation_history.py`

- [x] Add one outbound grouping helper that treats a visible owner plus its canonical continuation rounds as a single budget group. Keep `ConsoleChatMessage` history canonical; never splice provider-private or provider-wire rows into the stored visible transcript.
- [x] **Cycle 6A — grouping RED:** call `validate_continuation_restore` first and prove exact provider/model/protocol/normalized-base matching, owner attachment, branch selection, and canonical round ordering without rendering vendor wire history. The helper emits a `ContinuationOwnerGroup` consumed by provider adapters in TASK-15676/15677; unrelated providers receive no private group.
- [x] **Cycle 6B — token accounting RED:** private reasoning, assistant call JSON, and tool results contribute to the existing counter. If over budget, drop the oldest whole owner group—visible row plus every private row—never only private or only visible. Preserve existing system/current-turn pins and recent-round floors.
- [x] Keep the generic budget API small: add an optional group key/predicate or pre-grouped rows rather than a second trimming algorithm. QwenCloud and all providers without continuation must remain byte-identical.
- [x] **Cycle 6C — mutation/eviction RED:** mutate the group key so private rows can detach; the test must fail. Restore and rerun. Add branch selection and deleted-owner cases.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_provider_continuation_history.py Tests/Chat/test_console_history_budget.py Tests/Chat/test_console_provider_gateway.py Tests/Agents/test_run_log_eviction.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_history_budget.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Agents/run_log_eviction.py Tests/Chat/test_provider_continuation_history.py Tests/Chat/test_console_history_budget.py Tests/Chat/test_console_provider_gateway.py Tests/Agents/test_run_log_eviction.py
  git commit -m "feat(chat): budget private provider history atomically"
  ```

### Task 7: Make `.chatbook` Graph-Preserving And Other Exports Explicit

**Files:**

- Modify: `tldw_chatbook/Chatbooks/chatbook_models.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_creator.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_importer.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `Tests/Chatbooks/test_chatbook_models.py`
- Modify: `Tests/Chatbooks/test_chatbook_creator.py`
- Modify: `Tests/Chatbooks/test_chatbook_importer.py`
- Modify: `Tests/Chatbooks/test_chatbook_integration.py`
- Modify: `Tests/Chatbooks/test_chatbook_properties.py`
- Modify: `Tests/Chatbooks/test_chatbook_unit.py`
- Modify: `Tests/Chatbooks/test_chatbook_functionality.py`
- Modify: `Tests/Chatbooks/test_server_chatbook_service.py`
- Create: `Tests/Chatbooks/test_provider_continuation_roundtrip.py`
- Create: `Tests/Chat/test_provider_continuation_privacy.py`

- [x] **Cycle 7A — activate V2 manifest RED:** `ChatbookVersion.V2` already exists; make newly created exports select it while retaining V1 reads. V2 conversation export must include local stable message IDs, parent IDs, variant turn/index/count, selected/active leaf, role, content, deletion eligibility, ordering, and validated private continuation. Existing V1 fixtures and service compatibility remain unchanged.
- [x] **Cycle 7B — graph remap RED:** export a branched conversation with two assistant variants and a checkpoint on one owner. Import into a DB with colliding IDs; assert deterministic new IDs, remapped parents/selected leaf, preserved sibling order, and the checkpoint attached only after the owner map is complete. Opening the import executes zero tools.
- [x] **Cycle 7C — invalid/legacy RED:** corrupt version/provider/protocol/order/call pairing/bounds independently. Import visible messages, discard only private data, and emit one safe warning without canary contents. A V1/older flat export that cannot reconstruct an owner imports visible content and drops private continuation.
- [x] **Cycle 7D — ordinary JSON RED:** replace the existing whole-message-dict append with an explicit active-path projection. Include only the approved optional private field and graph-lite owner identifiers required for warning-safe reattachment. Never leak transient UI fields, credentials, metadata bags, or off-path variants.
- [x] **Cycle 7E — exclusion RED:** assert text, Markdown, FTS/search, render models, logs, errors, usage payloads, summaries, titles, clipboard copy, and run logs contain none of the private reasoning/call/result canaries.
- [x] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chatbooks/test_chatbook_models.py Tests/Chatbooks/test_chatbook_creator.py Tests/Chatbooks/test_chatbook_importer.py Tests/Chatbooks/test_chatbook_integration.py Tests/Chatbooks/test_chatbook_properties.py Tests/Chatbooks/test_chatbook_unit.py Tests/Chatbooks/test_chatbook_functionality.py Tests/Chatbooks/test_server_chatbook_service.py Tests/Chatbooks/test_provider_continuation_roundtrip.py Tests/Chat/test_provider_continuation_privacy.py
  ```

- [x] Commit:

  ```bash
  git add tldw_chatbook/Chatbooks/chatbook_models.py tldw_chatbook/Chatbooks/chatbook_creator.py tldw_chatbook/Chatbooks/chatbook_importer.py tldw_chatbook/Chat/Chat_Functions.py Tests/Chatbooks/test_chatbook_models.py Tests/Chatbooks/test_chatbook_creator.py Tests/Chatbooks/test_chatbook_importer.py Tests/Chatbooks/test_chatbook_integration.py Tests/Chatbooks/test_chatbook_properties.py Tests/Chatbooks/test_chatbook_unit.py Tests/Chatbooks/test_chatbook_functionality.py Tests/Chatbooks/test_server_chatbook_service.py Tests/Chatbooks/test_provider_continuation_roundtrip.py Tests/Chat/test_provider_continuation_privacy.py
  git commit -m "feat(chatbook): preserve provider continuation ownership"
  ```

### Task 8: Prove Crash Boundaries, Complete Documentation, And Close The Task

**Files:**

- Create: `Tests/Chat/test_provider_continuation_crash_recovery.py`
- Modify: `README.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/library/import-and-export.md`
- Modify: `backlog/tasks/task-15675 - Add-durable-provider-tool-continuation-checkpoints.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md` only if implementation produces a new, incident-backed reusable lesson.

- [x] Add deterministic crash injection at every approved boundary: before assistant/checkpoint commit, after it but before Sync-v2 projection, after projection but before acknowledgement, before `executing`, during a side effect, after result commit, and before next provider request. Assert the exact restored state and zero duplicate completed tool calls.
- [x] Add property/mutation coverage for branch/variant ownership, bounds, unknown versions, whole-record conflicts, replay policy, and atomic eviction. A mutation that re-executes `completed`, treats `executing` as pending, or detaches private history from its owner must fail.
- [x] Document private-state behavior, Resume/Discard semantics, restored ambiguity, current-credential resolution, sync durability, and export compatibility without claiming provider-side retention or exposing reasoning.
- [x] Run the complete touched surface:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/DB Tests/ChaChaNotesDB Tests/Agents Tests/Chat Tests/Chatbooks Tests/Sync_Interop
  ```

  If this is impractically broad on the implementation machine, run each directory separately and record exact counts. Any localhost suite blocked by sandbox policy must be rerun outside that restriction.
- [x] Run the full repository suite and static checks (the user stopped the
  repository run at 86%; per direction it was not restarted, is excluded from
  passing evidence, and the directly related 356-test matrix plus settled
  touched surfaces provide closeout evidence):

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
  git diff --check origin/dev...HEAD
  git diff --name-only -z --diff-filter=ACM origin/dev...HEAD -- '*.py' | xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check
  git diff --name-only -z --diff-filter=ACM origin/dev...HEAD -- '*.py' | xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/Chat/provider_continuation.py tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Sync_Interop/chat_outbox_producer.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook
  ```

- [x] Self-review every TASK-15675 acceptance criterion and ADR-063 invariant. Confirm no provider adapter, API default, vendor built-in tool, legacy Settings surface, or paid test was added in this PR.
- [x] Only after all criteria and evidence are complete, check every AC individually, write concise observed Implementation Notes with exact tests/deviations into the task, verify the rendered task, and only then mark Done:

  ```bash
  backlog task edit 15675 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7 --check-ac 8 --check-ac 9 --check-ac 10
  # Use apply_patch to add Implementation Notes from the observed results.
  backlog task 15675 --plain
  backlog task edit 15675 -s Done
  ```

- [x] Commit closeout:

  ```bash
  git add README.md Docs/User_Guide/console/agent-runs-and-tools.md Docs/User_Guide/library/import-and-export.md Docs/superpowers/plans/2026-08-12-durable-provider-tool-continuation-implementation.md "backlog/tasks/task-15675 - Add-durable-provider-tool-continuation-checkpoints.md" Tests/Chat/test_provider_continuation_crash_recovery.py
  git commit -m "docs(chat): document durable provider continuation"
  ```

#### Task 8 observed verification

- The original joined crash-recovery module passed `9 passed` across all seven
  approved crash boundaries. Final review then added production runtime/store
  crash hooks and restart assertions, exact lifecycle/terminal Sync-v2
  projections, inbound delete handling, current-intent clear/delete
  reconciliation during production restore, and policy-specific prior sidecars
  routed through the gateway's ordinary history budget. Specification and
  quality review approved the resulting contract at `0d4ac3f6b`.
- The final named continuation matrix passed `356 passed, 2 warnings in
  41.39s`. It covered canonical storage and migration, runtime lifecycle and
  eviction, Console persistence and explicit recovery, sync reconciliation,
  privacy and history budgeting, and `.chatbook` ownership round trips.
- Settled touched-surface evidence was: Agents `1386 passed`; Chat effective
  `4536 passed, 3 verified baseline failures, 64 skipped` after focused fixes;
  Chatbooks `243 passed, 1 skipped`; Sync Interop `250 passed`; DB `37 verified
  pre-TASK-15675 failures` after the three branch-caused fixture failures were
  fixed and their focused 14-test set passed; ChaChaNotesDB `187 passed, 2
  verified baseline failures`.
- A full-repository pytest run was started and advanced to 86%, but the user
  explicitly stopped broad testing. It produced no terminal summary and is not
  counted as completion evidence; per user direction, it was not restarted.
- The Task 8 Python file passes Ruff lint, Ruff format check, and `py_compile`.
  All changed Python files other than `agent_service.py` and `chat_screen.py`
  pass Ruff lint; those two files' 31 findings reproduce exactly on
  `origin/dev`. Ruff format initially reported 18 files: 17 reproduce on
  `origin/dev`, while the sole branch-added formatter deviation was corrected
  in `3b2a5ea85` with its 57 runtime tests passing. Mypy passed the three prescribed
  continuation core files, and both working-tree and branch `git diff --check`
  passed.
- No live or paid provider request was made. The diff adds no provider adapter,
  provider/API default, vendor built-in tool, or legacy Settings surface.

## PR Boundary And Handoff

- This is PR 1 of 3. Open it against `dev` and merge it before starting TASK-15676.
- PR description must link ADR-063, TASK-15675, this plan, the approved spec, schema migration evidence, crash matrix, and any baseline-only failures.
- After merge, create the TASK-15676 branch from the latest `dev`; do not stack provider implementation on an unmerged schema branch.
