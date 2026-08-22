# Console Library Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each Console conversation explicit, device-local automatic-retrieval and assistant-Library authority, a crash-honest send gate, and separate evidence/activity review surfaces.

**Architecture:** Six dependency-ordered deliveries establish schema and synchronization compatibility first, freeze authority and provider destination at execution second, add the recoverable pre-dispatch state machine third, then build UI, minimized activity review, and qualification. Policy and operational checkpoints remain device-local; only the closed assistant-generation state follows the existing whole-message sync/export contract. Pure models and projections stay outside Textual and database modules, while the app-owned store/coordinator own mutable lifetime and off-loop persistence.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5 schema v45, frozen dataclasses/enums and structural protocols, existing Console agent/Library/RAG/trajectory seams, pytest + pytest-asyncio + Hypothesis, Ruff, Backlog.md.

**Spec:** `Docs/superpowers/specs/2026-08-22-console-library-controls-design.md`

## Global Constraints

- ADR required: **yes**.
- ADR path: `backlog/decisions/079-console-library-conversation-authority.md`.
- Reason: storage/migration, sync/export, authority, runtime composition, privacy, and long-lived Console recovery/UI contracts change together.
- Delivery tasks are `TASK-19900.1` through `TASK-19900.6`; start them in dependency order and do not mark a child In Progress until its delivery begins.
- Re-read `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`, and `backlog/docs/lessons-backlog-hygiene.md` at the start of every child delivery.
- Recheck the ChaChaNotes schema head before coding. This plan expects v44 and allocates v45; renumber the migration, tests, and documentation together if the head changed.
- Keep historical `_FULL_SCHEMA_SQL_V4` unchanged. Fresh databases traverse v4 through v45.
- Acquire `BEGIN IMMEDIATE` before the first schema-version read for fresh, current, and migratable database opens.
- Never launch a schema-changing branch against the user's real data. Migration verification uses `:memory:` or `tmp_path`; live QA uses an explicitly isolated data directory only after the schema branch is current across participating worktrees.
- Ask the owner before a full repository test sweep. Run the targeted commands in this plan by default.
- Shipped new-session defaults are `[chat_defaults].rag_auto_retrieve_on_send = false` and `[console].assistant_library_access_default = false`.
- Existing v44 conversations seed the supplied legacy automatic value and assistant access Allowed. Later missing policy rows resolve Never/Blocked without a write.
- Automatic categories are exactly `("notes", "media", "conversations")`; active item scope narrows Notes/Media and excludes Conversations.
- Direct mode exposes the 18 names in `LIBRARY_TOOL_DESCRIPTORS`; RAG mode exposes only `search_library_rag`; their derived union of 19 names is always reserved.
- The status chip copy is exactly `Library · Auto off · Agent blocked`, `Library · Auto on · Agent blocked`, `Library · Auto off · Agent allowed`, or `Library · Auto on · Agent allowed`; unreadable policy is `Library: blocked · policy unavailable`.
- `Send once without Library` appears only after Automatic retrieval timeout/failure. Persistence failure and Never turns offer Retry/Cancel only.
- All user/source/provider/error/recovery strings render literally (`markup=False` or an equivalent boundary); color is never the only state carrier.
- The UI is an **Operate** surface: keyboard-first, dense, terminal-native, explicit about local/synced/external authority, and recoverable without reading logs.
- Policy and `console_dispatch_checkpoints` never sync or export. `assistant_generation_state` does sync/export. `library_activity` and `library_preparation` are local sidecars, default-redacted in explicit trajectory export, and inert on import.
- New/changed paths log identifiers, sizes, states, and error categories only—never queries, titles, source IDs/bodies/snippets, tool results, credentials, provider requests, or arbitrary exception text.

---

## File Responsibility Map

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Chat/console_library_policy.py` | Frozen policy/default/migration-seed/holder/read-result/turn-authority contracts and fail-closed normalization. No DB, Textual, or provider imports. |
| `tldw_chatbook/Chat/assistant_generation_state.py` | Closed assistant state vocabulary, role-aware normalization, literal remote/import status, and provider-history eligibility. |
| `tldw_chatbook/Chat/console_library_policy_repository.py` | Parameterized policy reads, conditional insert, revision CAS, and permanent-delete ownership over `CharactersRAGDB`. |
| `tldw_chatbook/Chat/console_library_policy_coordinator.py` | Off-loop reads/writes, same-process holder publication, and execution-time fresh capture. |
| `tldw_chatbook/Chat/console_dispatch_checkpoint.py` | Strict checkpoint model, canonical bounded JSON codecs, recovery/reconstructability contracts, and CAS result types. |
| `tldw_chatbook/Chat/console_dispatch_repository.py` | Transactional accepted insert, state CAS, terminal settlement, ADR-063 handoff, and loader reconciliation. |
| `tldw_chatbook/Chat/console_transaction_contribution.py` | Generic insert-only transaction-writer capability and sidecar contribution protocol used by first persistence and ephemeral promotion. It names no activity/preparation event kind and exposes no raw cursor/connection. |
| `tldw_chatbook/DB/migrations/chachanotes_v44_to_v45_console_library_policy.sql` | Nullable assistant state, device-local policy/checkpoint tables/index, and all four final Sync-v1 message triggers. |
| `tldw_chatbook/DB/ChaChaNotes_DB.py` | v45 runner/migration, migration seed input, message sync proof, low-level transaction operations, and trajectory rows. |
| `tldw_chatbook/config.py` and every production `CharactersRAGDB` opener | One sanitized migration-seed helper and shipped future-session defaults. DB code never reads TOML. |
| `tldw_chatbook/Sync_Interop/chat_outbox_producer.py` | Carry explicit assistant state in committed Sync-v2 source records. |
| `tldw_chatbook/Sync_Interop/envelope_builder.py` / `envelope_applier.py` | Build/apply explicit state and normalize only an older missing state key to `NULL`. |
| `tldw_chatbook/Chat/chat_persistence_service.py` | Narrow adapter for policy/checkpoint/atomic-turn operations and generic transaction contributions. |
| `tldw_chatbook/Chat/console_chat_store.py` | Session holders, preparation/checkpoint/activity lifetime, atomic identity publication, recovery hydration, and promotion rollback. |
| `tldw_chatbook/Chat/console_library_destination.py` | Pure endpoint classification into on-device/private/public/unknown and credential-free destination identity. |
| `tldw_chatbook/Chat/console_turn_context.py` | Detached pre-gateway configuration snapshot plus final execution context containing frozen policy authority and resolved destination. |
| `tldw_chatbook/Agents/tool_catalog.py` | Canonical permanent Library-name reservation and collision filtering independent of provider registration. |
| `tldw_chatbook/Agents/library_tool_provider.py` / `library_rag_tool_provider.py` | Policy-gated built-in providers and trusted pre-delivery activity-capture seam. |
| `tldw_chatbook/Agents/run_context.py` / `agent_service.py` | Current run/actor identity for primary/subagent activity attribution. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | Compose the captured provider, apply permanent collisions, and inherit one authority/activity sink across subagents. |
| `tldw_chatbook/Chat/console_turn_preparation.py` | Pure preparation state machine, pause/action matrix, CAS transitions, and reconstructability decisions. |
| `tldw_chatbook/Chat/library_preparation.py` | Bounded zero-match/bypass sidecar codec and pure sent-turn projection. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Admission, fresh authority capture, gateway resolution, retrieval, atomic acceptance, dispatch transition, provider invocation, and recovery actions. |
| `tldw_chatbook/Chat/console_prompt_queue_coordinator.py` | Precommit claim release, postcommit acknowledgement-once, unresolved-owner pause, and settlement advancement. |
| `tldw_chatbook/Chat/library_activity.py` | Bounded activity minimization/codec and pure selected-turn/branch projection. |
| `tldw_chatbook/Chat/console_library_activity_buffer.py` | Thread-safe store-owned pending-persistence buffer and bounded retry/final flush state. |
| `tldw_chatbook/Chat/trajectory.py` / `trajectory_export.py` / `trajectory_import.py` | Sidecar-only exclusion, default redaction, bounded full export, and inert import. |
| `tldw_chatbook/UI/Console_Modules/library_policy.py` | Screen-facing policy controller with explicit dependencies; no widget composition. |
| `tldw_chatbook/UI/Console_Modules/retrieval.py` | Manual Search Library only after automatic preparation moves to the turn controller. |
| `tldw_chatbook/UI/Console_Modules/right_rail.py` | Selected turn group and compact staged-source region. |
| `tldw_chatbook/UI/Console_Modules/wiring.py` | Construct the new policy controller with late-binding dependencies; do not grow `ChatScreen.__init__`. |
| `tldw_chatbook/Widgets/Console/console_library_access_modal.py` | Policy-only modal with safe dismissal, CAS feedback, and focus recovery. |
| `tldw_chatbook/Widgets/Console/console_library_search_modal.py` | Search-only query/source/item-scope surface with exact draft prefill. |
| `tldw_chatbook/Widgets/Console/console_status_chips.py` | One two-axis Library policy chip and opener event. |
| `tldw_chatbook/Widgets/Console/console_staged_context.py` | One primary row per source with activated detail and literal recovery text. |
| `tldw_chatbook/Widgets/Console/console_transcript.py` | Cited-source terminology, assistant-state literal rows, and message activity affordance. |
| `tldw_chatbook/UI/Screens/settings_library_rag_defaults.py` / `settings_screen.py` | Canonical future-session defaults; keep Direct/RAG as a separate live selector. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Thin event routing and snapshot projection only; new behavior lives in Console modules/widgets. |
| `tldw_chatbook/css/components/_agentic_terminal.tcss` | Viewport-fitting modals, pinned actions, stacked narrow layout, literal-state styling, and focus. |
| `tldw_chatbook/css/tldw_cli_modular.tcss` | Generated stylesheet output only; regenerate through the repository CSS script. |

## Cross-Delivery Interface Ledger

The names below are frozen for all six deliveries. If implementation discovers a type conflict, update this plan, the spec, and every consuming task before writing a second spelling.

```python
class ConsoleAutoRetrieve(str, Enum):
    NEVER = "never"
    AUTOMATIC = "automatic"

class ConsoleAssistantLibraryAccess(str, Enum):
    BLOCKED = "blocked"
    ALLOWED = "allowed"

class AssistantGenerationState(str, Enum):
    ACCEPTED = "accepted"
    DISPATCH_STARTED = "dispatch_started"
    CONTINUATION_ACTIVE = "continuation_active"
    COMPLETE = "complete"
    STOPPED = "stopped"
    FAILED = "failed"
    DISCARDED = "discarded"

AUTOMATIC_LIBRARY_SOURCE_TYPES: tuple[str, ...] = (
    "notes", "media", "conversations"
)
```

```python
@dataclass(frozen=True, slots=True)
class ConsoleLibraryMigrationSeed:
    auto_retrieve_on_send: bool

@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyDefaults:
    auto_retrieve: ConsoleAutoRetrieve
    assistant_access: ConsoleAssistantLibraryAccess

@dataclass(frozen=True, slots=True)
class ConsoleConversationLibraryPolicy:
    conversation_id: str
    auto_retrieve: ConsoleAutoRetrieve
    assistant_access: ConsoleAssistantLibraryAccess
    policy_revision: int
    updated_at: str

@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicySnapshot:
    auto_retrieve: ConsoleAutoRetrieve
    assistant_access: ConsoleAssistantLibraryAccess
    policy_revision: int | None
    source: Literal["new_session", "durable", "missing", "temporary", "unavailable"]
    error_code: str | None = None

@dataclass(slots=True)
class ConsoleLibraryPolicyHolder:
    snapshot: ConsoleLibraryPolicySnapshot
    explicitly_staged: bool = False
    save_pending: bool = False

@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyCandidate:
    auto_retrieve: ConsoleAutoRetrieve
    assistant_access: ConsoleAssistantLibraryAccess

@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyReadResult:
    snapshot: ConsoleLibraryPolicySnapshot
    durable_policy: ConsoleConversationLibraryPolicy | None

class ConsoleLibraryPolicyWriteStatus(str, Enum):
    COMMITTED = "committed"
    CONFLICT = "conflict"
    MISSING_CONVERSATION = "missing_conversation"
    UNAVAILABLE = "unavailable"

@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyWriteResult:
    status: ConsoleLibraryPolicyWriteStatus
    snapshot: ConsoleLibraryPolicySnapshot
```

```python
@dataclass(frozen=True, slots=True)
class ConsoleLibraryItemScopeSnapshot:
    note_ids: tuple[str, ...]
    media_ids: tuple[str, ...]
    conversations_allowed: bool

@dataclass(frozen=True, slots=True)
class ConsoleProviderIntent:
    provider: str
    model: str | None
    endpoint: str | None

@dataclass(frozen=True, slots=True)
class ConsoleTurnLibraryAuthority:
    policy: ConsoleLibraryPolicySnapshot
    direct_library_tools: bool
    source_types: tuple[str, ...]
    scope_snapshot: ConsoleLibraryItemScopeSnapshot
    provider_intent: ConsoleProviderIntent
    attempt_id: str

class ConsoleEgressClass(str, Enum):
    ON_DEVICE = "on_device"
    PRIVATE_NETWORK = "private_network"
    PUBLIC_NETWORK = "public_network"
    UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class ConsoleResolvedDestination:
    provider: str
    model: str | None
    endpoint_identity: str
    egress_class: ConsoleEgressClass

    @property
    def identity_key(self) -> tuple[str, str | None, str, ConsoleEgressClass]:
        return (self.provider, self.model, self.endpoint_identity, self.egress_class)
```

```python
@dataclass(frozen=True, slots=True)
class ConsoleTurnConfigurationSnapshot:
    session_id: str
    provider_selection: ConsoleProviderSelection
    session_settings: ConsoleSessionSettings | None
    workspace_roots: tuple[str, ...]
    capabilities: Mapping[str, object]
    rag_defaults: Mapping[str, object]
    tool_configuration: Mapping[str, object]
    provider_payload_settings: Mapping[str, object]

    @property
    def effective_model(self) -> str | None:
        return (
            self.provider_selection.explicit_model
            or self.provider_selection.configured_model
        )

@dataclass(frozen=True, slots=True)
class ConsoleTurnExecutionContext:
    configuration: ConsoleTurnConfigurationSnapshot
    library_authority: ConsoleTurnLibraryAuthority
    resolved_destination: ConsoleResolvedDestination

    @property
    def session_id(self) -> str:
        return self.configuration.session_id

    @property
    def effective_model(self) -> str | None:
        return self.configuration.effective_model

    @property
    def provider_selection(self) -> ConsoleProviderSelection:
        return self.configuration.provider_selection

    @property
    def session_settings(self) -> ConsoleSessionSettings | None:
        return self.configuration.session_settings

    @property
    def workspace_roots(self) -> tuple[str, ...]:
        return self.configuration.workspace_roots

    @property
    def capabilities(self) -> Mapping[str, object]:
        return self.configuration.capabilities

    @property
    def rag_defaults(self) -> Mapping[str, object]:
        return self.configuration.rag_defaults

    @property
    def tool_configuration(self) -> Mapping[str, object]:
        return self.configuration.tool_configuration

    @property
    def provider_payload_settings(self) -> Mapping[str, object]:
        return self.configuration.provider_payload_settings
```

```python
class ConsoleDispatchCheckpointState(str, Enum):
    ACCEPTED = "accepted"
    DISPATCH_STARTED = "dispatch_started"

class ConsoleTurnPreparationState(str, Enum):
    PREPARING = "preparing"
    READY = "ready"
    COMMITTING = "committing"
    ACCEPTED = "accepted"
    DISPATCH_STARTED = "dispatch_started"
    DISPATCHED = "dispatched"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    SETTLED = "settled"

class ConsolePreparationPauseKind(str, Enum):
    RETRIEVAL = "retrieval"
    PERSISTENCE = "persistence"
    DESTINATION_CHANGED = "destination_changed"

@dataclass(frozen=True, slots=True)
class ConsoleDispatchReconstructability:
    attachments_reconstructable: bool
    evidence_reconstructable: bool
    prefill_reconstructable: bool
    opaque_reference: str | None

@dataclass(frozen=True, slots=True)
class ConsoleDispatchCheckpoint:
    assistant_message_id: str
    user_message_id: str
    conversation_id: str
    preparation_id: str
    attempt_id: str
    state: ConsoleDispatchCheckpointState
    checkpoint_revision: int
    user_message_version: int
    assistant_message_version: int
    origin: Literal["manual", "queued"]
    queue_entry_id: str | None
    frozen_authority: ConsoleTurnLibraryAuthority
    resolved_destination: ConsoleResolvedDestination
    reconstructability: ConsoleDispatchReconstructability

@dataclass(frozen=True, slots=True)
class ConsoleTurnPreparation:
    preparation_id: str
    attempt_id: str
    session_id: str
    origin: Literal["manual", "queued"]
    queue_entry_id: str | None
    executed_draft: str
    execution_context: ConsoleTurnExecutionContext
    transient_user_message_id: str | None
    attachment_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    prefill_id: str | None
    queue_generation: int | None
    pre_send_title: str
    pre_send_conversation_id: str | None
    state: ConsoleTurnPreparationState
    pause_kind: ConsolePreparationPauseKind | None
    one_shot_bypass: bool
    ephemeral: bool

@dataclass(frozen=True, slots=True)
class ConsolePreparationTransition:
    preparation_id: str
    expected_state: ConsoleTurnPreparationState
    new_state: ConsoleTurnPreparationState
    pause_kind: ConsolePreparationPauseKind | None
    new_attempt_id: str | None

@dataclass(frozen=True, slots=True)
class ConsolePreparationOutcome:
    preparation_id: str
    attempt_id: str
    state: ConsoleTurnPreparationState
    evidence_bundle: EvidenceBundle | None
    contribution: LibraryPreparationContribution | None
    error_code: str | None
```

```python
@dataclass(frozen=True, slots=True)
class ConsoleDurableTurnAcceptance:
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    parent_message_id: str | None
    user_content: str
    attachments: tuple[Mapping[str, object], ...]
    preparation_id: str
    attempt_id: str
    origin: Literal["manual", "queued"]
    queue_entry_id: str | None
    frozen_authority: ConsoleTurnLibraryAuthority
    resolved_destination: ConsoleResolvedDestination
    reconstructability: ConsoleDispatchReconstructability
    contributions: tuple[ConsoleTransactionContribution, ...]

@dataclass(frozen=True, slots=True)
class ConsoleDispatchTransition:
    assistant_message_id: str
    expected_state: ConsoleDispatchCheckpointState
    expected_checkpoint_revision: int
    expected_user_message_version: int
    expected_assistant_message_version: int
    new_state: ConsoleDispatchCheckpointState
    new_attempt_id: str

@dataclass(frozen=True, slots=True)
class ConsoleAssistantSettlement:
    assistant_message_id: str
    expected_checkpoint_state: ConsoleDispatchCheckpointState
    expected_checkpoint_revision: int
    expected_user_message_version: int
    expected_assistant_message_version: int
    terminal_state: Literal["complete", "stopped", "failed", "discarded"]
    content: str
    metadata_json: str | None

@dataclass(frozen=True, slots=True)
class ConsoleContinuationHandoff:
    assistant_message_id: str
    expected_checkpoint_revision: int
    expected_user_message_version: int
    expected_assistant_message_version: int
    provider_continuation_json: str

class ConsoleDispatchResultStatus(str, Enum):
    COMMITTED = "committed"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    QUARANTINED = "quarantined"

@dataclass(frozen=True, slots=True)
class ConsoleDispatchReadResult:
    status: ConsoleDispatchResultStatus
    checkpoint: ConsoleDispatchCheckpoint | None
    error_code: str | None = None

@dataclass(frozen=True, slots=True)
class ConsoleDispatchWriteResult:
    status: ConsoleDispatchResultStatus
    checkpoint: ConsoleDispatchCheckpoint | None
    committed_message_version: int | None
    committed_payload_hash: str | None

@dataclass(frozen=True, slots=True)
class ConsoleDurableTurnCommit:
    identity: ConsoleStagedConversationIdentity
    user_message_id: str
    user_message_version: int
    assistant_message_id: str
    assistant_message_version: int
    checkpoint: ConsoleDispatchCheckpoint
```

```python
@dataclass(frozen=True, slots=True)
class LibraryPreparationEvent:
    version: Literal[1]
    outcome: Literal["zero_matches", "bypassed"]
    attempt_id: str
    result_count: int
    source_types: tuple[Literal["notes", "media", "conversations"], ...]

@dataclass(frozen=True, slots=True)
class LibraryPreparationView:
    turn_id: str
    outcome: Literal["zero_matches", "bypassed"]
    result_count: int
    source_types: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class LibraryPreparationContribution:
    event: LibraryPreparationEvent
    owner_message_key: Literal["user"] = "user"

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Persist one bounded sidecar through the caller-owned transaction."""

@dataclass(frozen=True, slots=True)
class LibraryActivitySourceRef:
    source_type: str
    source_id: str
    title: str

@dataclass(frozen=True, slots=True)
class LibraryActivityEvent:
    version: Literal[1]
    event_id: str
    attempt_id: str
    run_id: str
    actor_kind: Literal["primary", "subagent"]
    parent_run_id: str | None
    library_provider: Literal["direct", "rag"]
    operation: str
    status: Literal["succeeded", "empty", "blocked", "failed"]
    result_count: int
    query_preview: str | None
    source_refs: tuple[LibraryActivitySourceRef, ...]
    error_code: str | None
    error_summary: str | None

@dataclass(frozen=True, slots=True)
class LibraryActivityCandidate:
    attempt_id: str
    actor_kind: Literal["primary", "subagent"]
    run_id: str
    parent_run_id: str | None
    library_provider: Literal["direct", "rag"]
    operation: str
    arguments: Mapping[str, object]
    structured_result: object
    failure_code: str | None

@dataclass(frozen=True, slots=True)
class LibraryActivityView:
    turn_id: str
    events: tuple[LibraryActivityEvent, ...]
    unsaved_count: int
    save_error_code: str | None

@dataclass(frozen=True, slots=True)
class LibraryActivityContribution:
    events: tuple[LibraryActivityEvent, ...]
    owner_message_key: Literal["user"] = "user"

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Persist bounded activity rows through the promotion transaction."""

@dataclass(frozen=True, slots=True)
class LibraryActivityFlushResult:
    status: Literal["saved", "pending", "failed"]
    saved_count: int
    pending_count: int
    error_code: str | None
```

Canonical service-owned bounds:

```python
CHECKPOINT_AUTHORITY_MAX_BYTES = 4096
CHECKPOINT_DESTINATION_MAX_BYTES = 2048
CHECKPOINT_RECONSTRUCTABILITY_MAX_BYTES = 2048
LIBRARY_PREPARATION_MAX_BYTES = 1024
LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS = 160
LIBRARY_ACTIVITY_TITLE_MAX_CHARS = 160
LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS = 240
LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS = 200
LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT = 8
LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES = 8192
```

---

## Delivery 1 — Device-local foundation (`TASK-19900.1`)

### Task 1: Start TASK-19900.1 and pin the base

**Files:**
- Modify: `backlog/tasks/task-19900.1 - Add-device-local-Console-Library-policy-foundation.md`
- Read: the spec, ADR-079, and three lessons named in Global Constraints

**Interfaces:**
- Consumes: approved spec and ADR-079.
- Produces: an In Progress Backlog task with this plan linked in its task-local Implementation Plan.

- [ ] **Step 1: Recheck collisions and branch state.** Run `git status --short --branch`, `git branch -a --format='%(refname:short)' | rg -i 'library.*control|console.*rag'`, and `gh api -X GET /search/issues -f q='repo:rmusser01/tldw_chatbook is:pr is:open "Console Library"'`. Stop and reconcile any competing implementation.
- [ ] **Step 2: Confirm schema and test baseline.** Run `rg -n '_CURRENT_SCHEMA_VERSION = ' tldw_chatbook/DB/ChaChaNotes_DB.py`; expected: `44`. Then run `python -m pytest Tests/ChaChaNotesDB/test_migration_atomicity.py Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Chat/test_console_chat_store.py -q`; expected: pass before feature edits.
- [ ] **Step 3: Start the child correctly.** Run `backlog task edit 19900.1 -a @codex -s "In Progress" --plan "Implement Delivery 1 from Docs/superpowers/plans/2026-08-22-console-library-controls.md with RED-first migration, sync, repository, coordinator, and lifecycle checkpoints; ADR: backlog/decisions/079-console-library-conversation-authority.md."` Then verify with `backlog task 19900.1 --plain`.
- [ ] **Step 4: Commit task metadata.** Stage only the task file and commit `docs: start TASK-19900.1 Library policy foundation`.

### Task 2: Add policy, assistant-state, and migration-seed value contracts

**Files:**
- Create: `tldw_chatbook/Chat/console_library_policy.py`
- Create: `tldw_chatbook/Chat/assistant_generation_state.py`
- Create: `Tests/Chat/test_console_library_policy.py`
- Create: `Tests/Chat/test_assistant_generation_state.py`
- Modify: `tldw_chatbook/config.py:3110-3150,6751-6850`
- Modify: `Tests/test_config_console_defaults.py`

**Interfaces:**
- Consumes: enum/dataclass names and bounds in the interface ledger.
- Produces: `ConsoleLibraryMigrationSeed`, `ConsoleLibraryPolicyDefaults`, `ConsoleLibraryPolicySnapshot`, `ConsoleLibraryPolicyHolder`, `normalize_policy_read()`, `normalize_assistant_generation_state()`, and `load_console_library_migration_seed(app_config: Mapping[str, Any] | None = None) -> ConsoleLibraryMigrationSeed`.

- [ ] **Step 1: Write failing pure-model tests.** Pin shipped Never/Blocked defaults, legacy seed strict-bool validation, valid/absent/corrupt/error policy outcomes, holder explicit-stage behavior, role-aware assistant-state normalization, NULL + valid ADR-063 continuation precedence, and literal copy for unresolved imported states.
- [ ] **Step 2: Write failing config tests.** Pin template values `false`, strict load/save round trips, malformed values falling to shipped safe defaults for new sessions, and migration-seed resolution matching the pre-upgrade effective automatic value.
- [ ] **Step 3: Run RED.** Run `python -m pytest Tests/Chat/test_console_library_policy.py Tests/Chat/test_assistant_generation_state.py Tests/test_config_console_defaults.py -q`; expected: import/behavior failures.
- [ ] **Step 4: Implement the frozen contracts.** Keep parsing explicit and closed:

```python
def normalize_assistant_generation_state(
    *, role: object, raw_state: object, has_valid_active_continuation: bool
) -> AssistantGenerationState | None:
    if str(role or "").lower() != "assistant":
        return None
    if has_valid_active_continuation:
        return AssistantGenerationState.CONTINUATION_ACTIVE
    if raw_state is None:
        return None
    return AssistantGenerationState(str(raw_state))
```

`ConsoleLibraryMigrationSeed.__post_init__` rejects non-`bool` values; the config helper performs coercion before construction. Policy error codes are bounded machine codes, never exception text.
- [ ] **Step 5: Run GREEN and lint.** Run the Step-3 tests, then `python -m ruff check tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/assistant_generation_state.py tldw_chatbook/config.py Tests/Chat/test_console_library_policy.py Tests/Chat/test_assistant_generation_state.py Tests/test_config_console_defaults.py`.
- [ ] **Step 6: Commit.** Commit `feat(console): define Library policy contracts`.

### Task 3: Thread the sanitized migration seed through every production database opener

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:2930-2980`
- Modify: `tldw_chatbook/config.py`, `Notes/Notes_Library.py`, `RAG_Search/pipeline_functions_simple.py`, `RAG_Search/backfill.py`, `Chat/document_generator.py`, `MCP/server.py`, `Chatbooks/chatbook_creator.py`, `Chatbooks/chatbook_importer.py`, `Subscriptions/site_config_manager.py`, `UI/ChatbookCreationWindow.py`, and `UI/Tools_Settings_Window.py`
- Create: `Tests/DB/test_chachanotes_console_library_migration_seed_openers.py`

**Interfaces:**
- Consumes: `ConsoleLibraryMigrationSeed` and `load_console_library_migration_seed()` from Task 2.
- Produces: constructor keyword `console_library_migration_seed: ConsoleLibraryMigrationSeed | None = None` on `CharactersRAGDB.__init__` and no unseeded production v44 opener.

- [ ] **Step 1: Write the failing constructor/opener audit.** AST-scan production Python files for direct `CharactersRAGDB` calls and assert every call supplies `console_library_migration_seed=`. Add constructor tests proving fresh/current databases accept `None`, while a v44 migration rejects it before DDL.
- [ ] **Step 2: Run RED.** Run `python -m pytest Tests/DB/test_chachanotes_console_library_migration_seed_openers.py -q`; expected: opener audit and constructor behavior fail.
- [ ] **Step 3: Add the explicit constructor parameter and update all openers.** Each opener calls the one config helper; the DB module imports only the typed seed model and never calls config:

```python
db = CharactersRAGDB(
    db_path,
    client_id,
    console_library_migration_seed=load_console_library_migration_seed(),
)
```

Tests and historical-fixture helpers may pass a seed directly or deliberately omit it to test rejection.
- [ ] **Step 4: Re-run and lint.** Run the Step-2 test and `python -m ruff check` with the eleven opener files listed above plus `tldw_chatbook/DB/ChaChaNotes_DB.py`.
- [ ] **Step 5: Commit.** Commit `refactor(db): require explicit Library migration seed`.

### Task 4: Add the atomic v44-to-v45 migration and Sync-v1 triggers

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v44_to_v45_console_library_policy.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:450,5816-6100`
- Modify: `Tests/ChaChaNotesDB/historical_bootstrap.py`
- Create: `Tests/DB/test_chachanotes_console_library_policy_migration.py`
- Modify: `Tests/ChaChaNotesDB/test_migration_atomicity.py`

**Interfaces:**
- Consumes: typed migration seed and closed state strings.
- Produces: schema v45, `console_conversation_library_policy`, `console_dispatch_checkpoints`, `messages.assistant_generation_state`, index `idx_console_dispatch_checkpoint_conversation`, and final create/update/delete/undelete message triggers.

- [ ] **Step 1: Build a real v44 fixture and write RED migration assertions.** Assert all new objects/column are absent before open; afterward assert exact columns, CHECK clauses, foreign keys, no local-table sync triggers, all four final message triggers serialize state, and the update trigger watches state.
- [ ] **Step 2: Write lock/seed/rollback/concurrency RED tests.** Cover `BEGIN IMMEDIATE` before the first `_get_db_version` call on fresh/current/v44/older opens; missing/invalid seed; injected failure after every migration statement; two file-backed concurrent openers with opposite seeds; retry with another seed; active and soft-deleted seed rows; and a post-migration inserted conversation remaining rowless.
- [ ] **Step 3: Run RED.** Run `python -m pytest Tests/DB/test_chachanotes_console_library_policy_migration.py Tests/ChaChaNotesDB/test_migration_atomicity.py -q`; expected: v45/locking tests fail.
- [ ] **Step 4: Implement SQL and runner changes.** Set `_CURRENT_SCHEMA_VERSION = 45`, add map entry `44`, change the outer schema context to `TransactionContextManager(self, immediate=True)`, validate the seed only when entering 44→45, replace four trigger definitions in the migration, seed with one parameterized insert, and guard `44 -> 45` with rowcount 1. Never add the column to `_FULL_SCHEMA_SQL_V4`.
- [ ] **Step 5: Prove rollback and convergence.** Re-run Step 3 three times; the concurrent-opener test must assert one complete seed wins and the loser observes v45 without reseeding.
- [ ] **Step 6: Lint and diff-check.** Run `python -m ruff check tldw_chatbook/DB/ChaChaNotes_DB.py Tests/DB/test_chachanotes_console_library_policy_migration.py Tests/ChaChaNotesDB/test_migration_atomicity.py` and `git diff --check`.
- [ ] **Step 7: Commit.** Commit `feat(db): add Console Library policy schema v45`.

### Task 5: Carry assistant generation state through Sync-v2 and export/import contracts

**Files:**
- Modify: `tldw_chatbook/Sync_Interop/chat_outbox_producer.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_builder.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_applier.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:14240-14670`
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_creator.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_importer.py`
- Modify: `tldw_chatbook/Chat/document_generator.py`
- Modify: `tldw_chatbook/Chat/trajectory_export.py`
- Modify: `Tests/Sync_Interop/test_chat_outbox_producer.py`, `test_envelope_builder.py`, `test_envelope_applier.py`, `test_provider_continuation_reconciliation.py`
- Modify: `Tests/Chatbooks/test_provider_continuation_roundtrip.py`
- Create: `Tests/Chat/test_assistant_generation_state_roundtrip.py`

**Interfaces:**
- Consumes: `AssistantGenerationState` and migrated Sync-v1 payloads.
- Produces: `ChatSyncIntentRecord.assistant_generation_state: str | None`, keyword `assistant_generation_state: str | None = None` on `SyncEnvelopeBuilder.build_chat_message`, exact source-proof normalization, active JSON/Chatbook state, and literal text/Markdown pending status.

- [ ] **Step 1: Write RED sync compatibility tests.** For create/update/delete/undelete, pin explicit `NULL` and each closed assistant state through Sync-v1 proof and Sync-v2 outbox equality. Older payloads missing only the state normalize to `NULL`; unknown keys, malformed states, non-assistant non-NULL, and source/envelope mismatch return unavailable/rejected.
- [ ] **Step 2: Write RED export/import tests.** JSON/Chatbook/active-path round-trip the field; text/Markdown render literal accepted/dispatch-started/empty-complete copy; legacy missing field remains compatible; no exporter selects `console_dispatch_checkpoints`.
- [ ] **Step 3: Run RED.** Run the four Sync tests, `Tests/Chatbooks/test_provider_continuation_roundtrip.py`, and `Tests/Chat/test_assistant_generation_state_roundtrip.py`; expected: state omissions/failures.
- [ ] **Step 4: Implement one normalizer at each untrusted boundary.** New envelopes always contain the field, including `None`. Compatibility adds only the missing key:

```python
def normalize_legacy_chat_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("assistant_generation_state", None)
    return normalized
```

Validate exact allowed keys after normalization; do not use `setdefault` for any other field.
- [ ] **Step 5: Run GREEN, lint, and inspect source proof.** Re-run Step 3, scoped Ruff, and `rg -n 'assistant_generation_state' tldw_chatbook/Sync_Interop tldw_chatbook/Chatbooks tldw_chatbook/Chat/trajectory_export.py` to confirm every planned seam is present.
- [ ] **Step 6: Commit.** Commit `feat(sync): carry assistant generation state`.

### Task 6: Implement policy/checkpoint repositories, coordinator, and generic transaction contributions

**Files:**
- Create: `tldw_chatbook/Chat/console_library_policy_repository.py`
- Create: `tldw_chatbook/Chat/console_library_policy_coordinator.py`
- Create: `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- Create: `tldw_chatbook/Chat/console_dispatch_repository.py`
- Create: `tldw_chatbook/Chat/console_transaction_contribution.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Create: `Tests/ChaChaNotesDB/test_console_library_policy_repository.py`
- Create: `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`
- Create: `Tests/Chat/test_console_library_policy_coordinator.py`
- Create: `Tests/Chat/test_console_transaction_contribution.py`

**Interfaces:**
- Consumes: v45 schema, strict models, existing DB transaction/message version/hash/sync primitives, ADR-063 codec.
- Produces the exact methods below:

```python
class ConsoleTransactionWriter(Protocol):
    def execute(self, statement: str, parameters: tuple[object, ...], /) -> None:
        """Execute one parameterized INSERT through the caller transaction."""

    def executemany(
        self,
        statement: str,
        parameter_rows: Iterable[tuple[object, ...]],
        /,
    ) -> None:
        """Execute parameterized INSERT rows through the caller transaction."""

class ConsoleTransactionContribution(Protocol):
    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Write through the caller-owned capability without committing."""

class ConsoleLibraryPolicyRepository:
    def read(self, conversation_id: str) -> ConsoleLibraryPolicyReadResult:
        """Read one policy or an explicit fail-closed outcome."""

    def insert(self, conversation_id: str, candidate: ConsoleLibraryPolicyCandidate) -> ConsoleLibraryPolicyWriteResult:
        """Conditionally insert revision one without overwriting a race winner."""

    def compare_and_swap(self, conversation_id: str, expected_revision: int, candidate: ConsoleLibraryPolicyCandidate) -> ConsoleLibraryPolicyWriteResult:
        """Commit exactly one expected revision or report conflict."""

class ConsoleDispatchRepository:
    def insert_with_messages(self, cursor: sqlite3.Cursor, acceptance: ConsoleDurableTurnAcceptance) -> ConsoleDispatchCheckpoint:
        """Insert one USER, assistant owner, and accepted checkpoint."""

    def read_for_session(self, conversation_id: str) -> ConsoleDispatchReadResult:
        """Read and validate at most one active-path recovery owner."""

    def cas_state(self, transition: ConsoleDispatchTransition) -> ConsoleDispatchWriteResult:
        """Apply an expected-revision accepted/dispatch-started transition."""

    def settle_with_assistant(self, settlement: ConsoleAssistantSettlement) -> ConsoleDispatchWriteResult:
        """Commit terminal assistant state and delete its checkpoint atomically."""

    def handoff_to_provider_continuation(self, handoff: ConsoleContinuationHandoff) -> ConsoleDispatchWriteResult:
        """Commit ADR-063 ownership and remove dispatch ownership atomically."""

class ConsoleLibraryPolicyCoordinator:
    def register_holder(self, session_id: str, conversation_id: str | None, holder: ConsoleLibraryPolicyHolder) -> None:
        """Bind one live holder for same-process committed publication."""

    def unregister_holder(self, session_id: str) -> None:
        """Remove one closed session holder."""

    async def load(self, session_id: str, conversation_id: str) -> ConsoleLibraryPolicyReadResult:
        """Read durable policy off-loop and publish its effective result."""

    async def save(self, session_id: str, candidate: ConsoleLibraryPolicyCandidate) -> ConsoleLibraryPolicyWriteResult:
        """Commit one insert/CAS and publish only the committed snapshot."""

    async def capture_for_execution(self, session_id: str) -> ConsoleLibraryPolicySnapshot:
        """Perform the execution-time durable read and return frozen authority."""
```

- [ ] **Step 1: Write RED policy repository tests.** Cover valid/absent/corrupt/error reads, conditional insert race, update CAS success/conflict, missing/deleted conversation, no candidate publication, soft-delete retention/restore, and hard-purge cascade.
- [ ] **Step 2: Write RED checkpoint codec/ownership tests.** Pin exact JSON keys/types/order/byte caps; reject request text, source snippets, credentials, bad roles, cross-conversation owners, duplicate active-path owners, invalid states, and generic upsert behavior.
- [ ] **Step 3: Write RED atomic checkpoint tests.** Inject failure at USER, assistant, checkpoint, state-CAS, terminal-content, sync-intent, checkpoint-delete, continuation-write, and handoff-delete statements. Assert all-or-nothing plus expected checkpoint revision, USER/assistant versions, matching assistant state, and `deleted = 0`.
- [ ] **Step 4: Write RED coordinator tests.** Use two holders for one conversation and two repositories over one file DB. Assert off-loop execution, same-process publication only after commit, fresh execution read defeating stale Allowed, unavailable read producing Never/Blocked, and a commit after capture affecting only the next capture.
- [ ] **Step 5: Run RED.** Run the four new test files; expected: missing modules/contracts.
- [ ] **Step 6: Implement minimal repositories and coordinator.** Use parameterized SQL and typed result variants. `settle_with_assistant` and `handoff_to_provider_continuation` must write message content/state/version/hash/sync intent and delete the expected checkpoint in one `transaction(immediate=True)`.
- [ ] **Step 7: Implement the generic contribution seam.** Contributions receive only an insert-only `ConsoleTransactionWriter`, committed conversation ID candidate, and message-ID map. Its complete accepted SQL grammar is one `INSERT INTO simple_table (simple_column, ...) VALUES (?, ...)` statement with ordinary whitespace, one VALUES row, unquoted/unqualified ASCII identifiers, and equal non-zero column/placeholder/tuple arity; `executemany` requires at least one same-arity tuple. It rejects conflict modifiers/clauses (including REPLACE/IGNORE and both ON CONFLICT actions), literals/comments standing in for placeholders, INSERT...SELECT, RETURNING, multiple VALUES rows, quoted/dynamic identifiers, and every extra clause. The writer exposes no raw cursor/connection, authorizer, transaction/savepoint/ATTACH/DETACH control, commit/rollback, repository/session/publication state, or connection factory. Contribution errors propagate through the caller-owned `BEGIN IMMEDIATE` transaction. This is an API capability boundary for trusted in-process components, not a hostile-code sandbox.
- [ ] **Step 8: Run GREEN, lint, and mutation probes.** Re-run Step 5; temporarily invert missing/error fail-closed and remove a checkpoint version predicate one at a time, confirm named tests fail, then restore the implementation. Run scoped Ruff.
- [ ] **Step 9: Commit.** Commit `feat(console): add Library policy and dispatch repositories`.

### Task 7: Integrate holders, atomic first persistence, and promotion rollback

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:206-340,516-760,927-1510,5421-5845`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Create: `Tests/Chat/test_console_chat_store_library_policy.py`
- Create: `Tests/Chat/test_console_chat_store_atomic_promotion.py`

**Interfaces:**
- Consumes: holder/coordinator, transaction contributions, repository primitives.
- Produces: `ConsoleChatSession.library_policy_holder`, store-owned coordinator registration, `stage_first_persistence()`, `publish_committed_identity()`, contribution-aware `promote_ephemeral_session()`, and unresolved-operation promotion guard.

- [ ] **Step 1: Write RED session lifecycle tests.** New local session captures current defaults; untouched defaults do not make a pristine tab dirty; an explicit empty-tab edit does; first local persistence inserts policy even when not edited; restored missing-row conversation stays write-free Never/Blocked until explicit save.
- [ ] **Step 2: Write RED staged-identity tests.** Inject conversation/policy write failures and assert `persisted_conversation_id`, title, scope holder, message IDs, attachments, and policy holder remain byte-for-byte pre-call. A retry creates one conversation and publishes ID/title only after commit.
- [ ] **Step 3: Write RED promotion tests.** Promotion refuses unresolved preparation/checkpoint analogue before any write; success persists policy/full lineage/contributions atomically; each injected write failure restores ephemeral identity, policy, messages, scope, contributions, and retryability.
- [ ] **Step 4: Run RED.** Run the two new files plus `Tests/Chat/test_console_chat_store.py`; expected: lifecycle/atomicity failures.
- [ ] **Step 5: Refactor eager mutation out of first persistence.** Introduce immutable staging rather than assigning session fields inside `create_conversation`:

```python
@dataclass(frozen=True, slots=True)
class ConsoleStagedConversationIdentity:
    conversation_id: str
    title: str

def publish_committed_identity(
    self, session_id: str, identity: ConsoleStagedConversationIdentity
) -> None:
    session = self._session_or_raise(session_id)
    session.persisted_conversation_id = identity.conversation_id
    session.title = identity.title
```

The publishing method is called only after the transaction context exits successfully.
- [ ] **Step 6: Integrate holder/coordinator ownership.** Register/unregister holders on restore/create/close, publish committed saves to same-process siblings, and have runtime construction share one coordinator per app/store.
- [ ] **Step 7: Run GREEN and targeted foundation battery.** Run `Tests/Chat/test_console_library_policy.py`, `Tests/Chat/test_assistant_generation_state.py`, `Tests/DB/test_chachanotes_console_library_migration_seed_openers.py`, `Tests/DB/test_chachanotes_console_library_policy_migration.py`, `Tests/ChaChaNotesDB/test_migration_atomicity.py`, `Tests/ChaChaNotesDB/test_console_library_policy_repository.py`, `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`, `Tests/Chat/test_console_library_policy_coordinator.py`, `Tests/Chat/test_console_transaction_contribution.py`, `Tests/Chat/test_console_chat_store_library_policy.py`, and `Tests/Chat/test_console_chat_store_atomic_promotion.py` together, then scoped Ruff and `git diff --check`.
- [ ] **Step 8: Finish TASK-19900.1 hygiene.** Check every acceptance criterion, add concise Implementation Notes naming schema/sync/repository/lifecycle changes and targeted evidence, update a lessons file only if an actual reusable incident occurred, then run `backlog task edit 19900.1 -s Done` only when all DoD items are true.
- [ ] **Step 9: Commit.** Commit `feat(console): persist Library policy atomically`.

---

## Delivery 2 — Runtime authority and provider composition (`TASK-19900.2`)

### Task 8: Start TASK-19900.2 and split pre-gateway configuration from final execution context

**Files:**
- Modify: `backlog/tasks/task-19900.2 - Enforce-Console-Library-policy-at-the-agent-runtime-boundary.md`
- Modify: `tldw_chatbook/Chat/console_turn_context.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:3330-3520,8520-8600`
- Modify: `Tests/Chat/test_console_turn_execution_context.py`, `Tests/UI/test_console_auto_rag_on_send.py`, `Tests/UI/test_console_composer_menu.py`, and `Tests/UI/test_console_harness_config_honesty.py`
- Create: `Tests/Chat/test_console_turn_library_authority.py`

**Interfaces:**
- Consumes: `ConsoleLibraryPolicyCoordinator.capture_for_execution(session_id: str) -> ConsoleLibraryPolicySnapshot`.
- Produces: `ConsoleTurnConfigurationSnapshot.capture` for pre-gateway config and final `ConsoleTurnExecutionContext(configuration, library_authority, resolved_destination)`.

- [ ] **Step 1: Start the child and record baseline.** Put TASK-19900.2 In Progress with a task-local link to this delivery. Run `python -m pytest Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_agent_bridge.py -q`; expected: pass.
- [ ] **Step 2: Write RED capture-order tests.** Assert immediate capture occurs after admission, queued capture after dequeue, fresh policy read precedes authority construction, gateway resolution precedes final context, and selector/scope/provider changes after final construction do not mutate it.
- [ ] **Step 3: Run RED.** Run `Tests/Chat/test_console_turn_library_authority.py`, `Tests/Chat/test_console_turn_execution_context.py`, `Tests/UI/test_console_auto_rag_on_send.py`, `Tests/UI/test_console_composer_menu.py`, and `Tests/UI/test_console_harness_config_honesty.py`; expected: missing split types.
- [ ] **Step 4: Perform the mechanical type split.** Move current fields/capture logic to `ConsoleTurnConfigurationSnapshot`. The final type wraps it plus authority/destination and exposes compatibility read-only properties required by existing consumers. Do not permit `None` authority or destination in the final constructor.
- [ ] **Step 5: Capture fixed authority.** Set `source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES`, copy item scope and provider intent, use the fresh durable policy result, and capture `direct_library_tools` once. A failed policy read constructs Never/Blocked with source `unavailable`.
- [ ] **Step 6: Run GREEN, lint, and commit.** Run Step 3 plus scoped controller/turn-context tests and Ruff. Commit `refactor(console): freeze Library turn authority`.

### Task 9: Classify the resolved provider destination conservatively

**Files:**
- Create: `tldw_chatbook/Chat/console_library_destination.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py:634-710,1694-1810`
- Create: `Tests/Chat/test_console_library_destination.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`

**Interfaces:**
- Consumes: ready `ConsoleProviderResolution` after endpoint normalization.
- Produces: `resolve_console_destination(resolution: ConsoleProviderResolution) -> ConsoleResolvedDestination` and a credential-free stable `identity_key` comparison.

- [ ] **Step 1: Write RED table tests.** Cover `localhost`, `127.0.0.0/8`, `::1`, Unix/local transports if supported, RFC1918 IPv4, link-local, unique-local IPv6, public IP, default cloud URL, custom URL with credentials/path/query, malformed URL, hostname without provable address class, and missing endpoint.
- [ ] **Step 2: Pin conservative expectations.** Loopback/local transport is `on_device`; literal private/link-local addresses are `private_network`; literal public and canonical cloud endpoints are `public_network`; unresolved/malformed/custom hostnames without proof are `unknown`. Provider name and API-key presence never affect classification.
- [ ] **Step 3: Run RED.** Run the new destination tests and gateway tests.
- [ ] **Step 4: Implement standard-library parsing.** Use `urllib.parse` and `ipaddress`; normalize scheme/host/port only, strip userinfo/path/query/fragment, and return a bounded external/unknown label on parse failure.
- [ ] **Step 5: Add runtime disclosure state.** When either policy axis can place Library data in the request and destination changes from on-device to another class, store a persistent session runtime disclosure until send settlement or another destination change; never rewrite policy.
- [ ] **Step 6: Run GREEN, lint, and commit.** Commit `feat(console): disclose resolved Library egress`.

### Task 10: Reserve all Library names and enforce provider absence/selection

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py:2590-3000`
- Modify: `tldw_chatbook/Agents/library_tool_provider.py`
- Modify: `tldw_chatbook/Agents/library_rag_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:2822-3050,3584-3740`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:5610-5650`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:5530-5600`
- Modify: `Tests/Agents/test_tool_catalog_owner_cache.py`
- Modify: `Tests/Agents/test_library_tool_provider.py`
- Create: `Tests/Agents/test_library_name_reservation.py`
- Create: `Tests/Chat/test_console_library_runtime_policy.py`

**Interfaces:**
- Consumes: final execution context and existing Direct/RAG providers.
- Produces: `LIBRARY_RESERVED_TOOL_NAMES: frozenset[str]`, policy-aware `_library_provider_for_context()`, and authenticated ephemeral provider marker `BuiltinLibraryAuthority`.

```python
@dataclass(frozen=True, slots=True)
class BuiltinLibraryAuthority:
    provider_instance_id: str
    reserved_names: frozenset[str]
    assistant_access: ConsoleAssistantLibraryAccess
```

- [ ] **Step 1: Write RED inventory/reservation tests.** Assert descriptor count 18, union count 19, and every Skill/MCP collision is filtered in Blocked, Direct, and RAG modes even when no Library provider is registered.
- [ ] **Step 2: Write RED provider matrix tests.** Blocked/unavailable returns no provider/schema/callable for primary and subagent; Allowed+Direct lists exactly 18; Allowed+RAG lists exactly one; a child may narrow but not widen parent authority.
- [ ] **Step 3: Write RED ephemeral gate tests.** Allowed plus authenticated built-in marker admits only the exact reserved read-only names. `source="library"` spoofing, unreserved future names, third-party provider identity, and Blocked all fail.
- [ ] **Step 4: Run RED.** Run the four scoped files; expected: reservation and gate failures.
- [ ] **Step 5: Derive one immutable reservation.** Export:

```python
LIBRARY_RESERVED_TOOL_NAMES = frozenset(
    (*LIBRARY_TOOL_DESCRIPTORS.keys(), RAG_TOOL_NAME)
)
```

Pass it into both Skill and MCP collision sets before optional provider registration. Do not maintain another list.
- [ ] **Step 6: Gate provider construction from final authority.** `_library_provider_for_context` returns `None` unless `assistant_access is ALLOWED`; otherwise choose Direct/RAG from the captured selector. Thread the same provider and authority through the bridge's parent/child run construction.
- [ ] **Step 7: Run GREEN and mutation checks.** Remove the Blocked gate and permanent reservation separately and confirm named tests fail; restore, run scoped Ruff, and commit `feat(agents): enforce Console Library authority`.

### Task 11: Verify and close TASK-19900.2

**Files:**
- Modify: runtime tests and `backlog/tasks/task-19900.2 - Enforce-Console-Library-policy-at-the-agent-runtime-boundary.md`

**Interfaces:**
- Consumes: `ConsoleTurnExecutionContext`, `ConsoleResolvedDestination`, `LIBRARY_RESERVED_TOOL_NAMES`, and policy-aware provider construction from Tasks 8–10.
- Produces: one integrated runtime gate with no cached-authority bypass.

- [ ] **Step 1: Add concurrent/frozen integration cases.** Use two DB handles and two sessions to prove a second-process Blocked commit defeats a stale Allowed holder; a commit after capture affects the next turn; queued capture happens at execution; subagents inherit; Direct/RAG selector changes during a run do not alter that run.
- [ ] **Step 2: Add destination disclosure combinations.** Test Automatic+Blocked as well as both Allowed combinations; verify Unknown never renders on-device copy.
- [ ] **Step 3: Run the delivery battery.** Run `python -m pytest Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_runtime_policy.py Tests/Agents/test_library_name_reservation.py Tests/Agents/test_library_tool_provider.py Tests/Chat/test_console_agent_bridge.py -q`, then scoped Ruff and `git diff --check`.
- [ ] **Step 4: Complete Backlog hygiene.** Check ACs, add implementation notes/evidence, record only an incident-backed lesson, mark Done only after DoD.
- [ ] **Step 5: Commit.** Commit `test(console): qualify Library runtime authority`.

---

## Delivery 3 — Truthful automatic retrieval and dispatch recovery (`TASK-19900.3`)

### Task 12: Start TASK-19900.3 and implement pure preparation/sidecar state

**Files:**
- Modify: `backlog/tasks/task-19900.3 - Make-automatic-Console-Library-retrieval-a-truthful-send-gate.md`
- Create: `tldw_chatbook/Chat/console_turn_preparation.py`
- Create: `tldw_chatbook/Chat/library_preparation.py`
- Create: `Tests/Chat/test_console_turn_preparation.py`
- Create: `Tests/Chat/test_library_preparation.py`
- Modify: `tldw_chatbook/Chat/trajectory.py`
- Modify: `tldw_chatbook/Chat/trajectory_export.py`
- Modify: `tldw_chatbook/Chat/trajectory_import.py`

**Interfaces:**
- Consumes: final execution context, checkpoint states, generic contribution protocol.
- Produces: `ConsoleTurnPreparation`, `ConsolePreparationTransition`, `LibraryPreparationEvent`, `LibraryPreparationContribution`, and `project_library_preparation(rows, active_turn_ids) -> tuple[LibraryPreparationView, ...]`.

- [ ] **Step 1: Start the child and baseline current retrieval/controller tests.** Put TASK-19900.3 In Progress with this plan link. Run `python -m pytest Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_rag_settings_modal.py Tests/Chat/test_console_prompt_queue_coordinator.py -q`.
- [ ] **Step 2: Write RED state-machine tests.** Pin every legal transition/action in the spec, reject illegal/repeated/racing transitions by preparation ID/state, distinguish retrieval/persistence/destination pause, disable actions during committing, and prove Never moves directly to ready without a Library status.
- [ ] **Step 3: Write RED sidecar tests.** Zero-match and bypass serialize only version/outcome/attempt/result-count/fixed categories within 1024 bytes; cancelled/failure events write nothing; generic trajectory derives no row; default/full export remain equally bounded; import is inert.
- [ ] **Step 4: Run RED.** Run both new files and trajectory export/import tests.
- [ ] **Step 5: Implement pure contracts.** The action matrix must be data, not widget conditionals:

```python
PAUSE_ACTIONS: Mapping[ConsolePreparationPauseKind, tuple[str, ...]] = {
    ConsolePreparationPauseKind.RETRIEVAL: ("retry", "bypass", "cancel"),
    ConsolePreparationPauseKind.PERSISTENCE: ("retry", "cancel"),
    ConsolePreparationPauseKind.DESTINATION_CHANGED: ("retry", "cancel"),
}
```

Mark `library_preparation` as sidecar-only beside nested trajectory kinds so it cannot displace an anchor.
- [ ] **Step 6: Run GREEN, lint, and commit.** Commit `feat(console): model Library turn preparation`.

### Task 13: Replace fail-open auto RAG with fixed-category pre-dispatch preparation

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/retrieval.py:700-830`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:3330-3890`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Create: `Tests/Chat/test_console_automatic_library_preparation.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**
- Consumes: `ConsoleTurnExecutionContext`, preparation model, current RAG service/scope resolver, staged evidence bundle.
- Produces: `prepare_library_for_turn(preparation_id: str) -> ConsolePreparationOutcome` and store methods `begin_preparation`, `compare_and_set_preparation`, `preparation_for_session`, `cancel_preparation`.

- [ ] **Step 1: Write RED admission tests.** Eligible plain manual/queued user text with Automatic runs; commands, approvals, regeneration, wake/machine input, ineligible kinds, explicit evidence, Never, and bypass skip. Query equals executed draft. Source types equal the fixed tuple independent of manual modal state.
- [ ] **Step 2: Write RED scope/outcome tests.** Active scope narrows exact Note/Media IDs and excludes Conversations; success injects the bundle into the exact same prepared request; zero dispatches with contribution; timeout/failure pauses and never calls provider; Retry reuses frozen authority/draft with a new attempt ID; bypass changes no policy.
- [ ] **Step 3: Write RED cancel/race tests.** Manual Cancel restores exact draft/attachments/evidence/prefill and removes transient echo; queued Cancel releases the exact claim without foreground composer copy; racing Retry/Bypass/Cancel/close/shutdown produces one winner and zero provider calls before acceptance.
- [ ] **Step 4: Run RED.** Run the new test file plus current controller/retrieval tests.
- [ ] **Step 5: Move automatic ownership out of the screen retrieval controller.** Delete its standing config read and fail-open notices. Keep manual search methods. The controller creates/store-registers preparation before retrieval and uses the final context only.
- [ ] **Step 6: Implement explicit outcomes.** Timeout/service failure returns paused state; it never logs exception text or falls through. Zero and bypass add `LibraryPreparationContribution`; evidence success attaches the exact sealed bundle to the prepared request.
- [ ] **Step 7: Run GREEN, mutation-check no-dispatch, lint, and commit.** Delete the early return on retrieval failure and confirm `test_retrieval_failure_never_dispatches` fails; restore. Commit `feat(console): gate sends on Library preparation`.

### Task 14: Commit USER, assistant owner, checkpoint, and disclosures atomically

**Files:**
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:5421-5845`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:3690-3860`
- Create: `Tests/Chat/test_console_durable_turn_acceptance.py`
- Create: `Tests/Chat/test_console_first_send_atomicity.py`

**Interfaces:**
- Consumes: `ConsoleDispatchRepository.insert_with_messages`, staged identity, transaction contributions, preparation in `ready`.
- Produces: `commit_durable_turn(acceptance: ConsoleDurableTurnAcceptance) -> ConsoleDurableTurnCommit` and idempotent postcommit effects keyed by `preparation_id`.

- [ ] **Step 1: Write RED write-boundary injection tests.** Fail conversation, policy, USER, assistant, checkpoint, and preparation contribution writes separately. Assert no DB rows/version advance, no published ID/title/owners, exact staged-state restoration, `paused(persistence)`, Retry/Cancel only, and clean retry with no duplicate.
- [ ] **Step 2: Write RED postcommit-boundary tests.** Fail identity publication, staged clearing, queue acknowledgement, accepted hook, prompt-history hook, checkpoint transition, and provider-call entry separately. Assert the same durable USER/assistant/checkpoint owns recovery and no second row is created.
- [ ] **Step 3: Run RED.** Run both new files.
- [ ] **Step 4: Implement one transaction and delayed publication.** The transaction writes new conversation/policy if needed, USER, empty assistant with `accepted`, checkpoint, and contributions. It returns immutable committed IDs/versions/hash but mutates no session object until the context exits.
- [ ] **Step 5: Make postcommit effects idempotent.** Track completion bits by `preparation_id`; queue acknowledgement and accepted hook may be retried but occur at most once. Clear only captured attachment/evidence/prefill identities.
- [ ] **Step 6: Run GREEN, lint, and commit.** Commit `feat(console): accept durable turns atomically`.

### Task 15: Add dispatch recovery, terminal settlement, and queue semantics

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py:630-690`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:1248-1510,4140-4220,6410-6760`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:3040-3308,9600-10400,11540-12020`
- Modify: `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- Modify: `tldw_chatbook/UI/Console_Modules/prompt_queue.py`
- Create: `tldw_chatbook/UI/Console_Modules/dispatch_recovery.py`
- Create: `Tests/Chat/test_console_dispatch_recovery.py`
- Create: `Tests/Chat/test_console_dispatch_queue_recovery.py`
- Create: `Tests/UI/test_console_dispatch_recovery.py`

**Interfaces:**
- Consumes: durable checkpoint repository, preparation model, queue claim/authorization, final destination identity.
- Produces: accepted Retry response/Discard; dispatch-started Retry anyway/Discard; inert remote/import state; exact queue pause/settlement; in-memory ephemeral analogue.

- [ ] **Step 1: Write RED loader matrix tests.** Cover every row in spec §6.3: valid continuation precedence, valid checkpoint, both owners, terminal+stale checkpoint, bad roles/cross-conversation/missing owners, orphan continuation_active, checkpoint-free accepted/dispatch-started inert state, and ordinary terminal/NULL.
- [ ] **Step 2: Write RED recovery-action tests.** Accepted retry revalidates frozen destination/authority and reconstructability; unreconstructable prefill/evidence disables retry with literal reason. Dispatch-started never auto-replays and warns duplicate risk. Discard atomically writes `discarded` plus checkpoint deletion while retaining USER.
- [ ] **Step 3: Write RED queue tests.** Precommit Cancel releases exact entry; postcommit accepted entry is acknowledged at most once and never returns pending; later entries stay paused across restart until Retry/Discard settles; queue advances once after settlement.
- [ ] **Step 4: Write RED ephemeral tests.** Same in-memory actions and states, no checkpoint table row, survival across screen replacement, loss only at app-runtime end, unresolved state blocks promotion with exact copy.
- [ ] **Step 5: Run RED.** Run the three new files.
- [ ] **Step 6: Implement hydration before queue advancement.** Store loads and reconciles continuation/checkpoint ownership before prompt queue wake. Quarantine invalid pairs with bounded codes and never invoke provider or delete unrelated rows.
- [ ] **Step 7: Implement CAS action paths.** Transition accepted→dispatch_started immediately before provider call. Reuse the same assistant owner. Settlement passes expected checkpoint revision plus both message versions and deletion guards.
- [ ] **Step 8: Render recovery literally.** UI-neutral state owns labels/reasons; Textual widgets only project it with idempotently disabled in-flight buttons.
- [ ] **Step 9: Run GREEN, lint, and commit.** Commit `feat(console): recover accepted Library turns`.

### Task 16: Integrate ADR-063 handoff, history filtering, and all message projections

**Files:**
- Modify: `tldw_chatbook/Chat/provider_continuation.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:1360-1510,4140-4220,6480-6760`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:3040-3308,12780-12880`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Chat/console_prepared_request.py`
- Modify: `tldw_chatbook/Sync_Interop/chat_outbox_producer.py`, `tldw_chatbook/Sync_Interop/envelope_builder.py`, `tldw_chatbook/Sync_Interop/envelope_applier.py`, `tldw_chatbook/Chat/chat_conversation_service.py`, `tldw_chatbook/Chat/Chat_Functions.py`, `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`, `tldw_chatbook/Chatbooks/chatbook_creator.py`, `tldw_chatbook/Chatbooks/chatbook_importer.py`, `tldw_chatbook/Chat/document_generator.py`, and `tldw_chatbook/Chat/trajectory_export.py`
- Modify: `Tests/Chat/test_provider_continuation_crash_recovery.py`
- Modify: `Tests/Chat/test_provider_continuation_history.py`
- Create: `Tests/Chat/test_console_dispatch_continuation_handoff.py`
- Create: `Tests/Chat/test_console_assistant_generation_history.py`

**Interfaces:**
- Consumes: checkpoint handoff/settlement, assistant state normalization, existing ADR-063 continuation validation.
- Produces: exclusive dispatch→continuation ownership before tool execution, lazy legacy normalization/rebind, and closed-state provider-history policy.

- [ ] **Step 1: Write RED pre-tool handoff tests.** Complete tool-call batch validates and atomically writes continuation_active/version/hash/sync intent and deletes expected checkpoint before any tool invoke. Every injected handoff failure executes zero tools and leaves dispatch recovery.
- [ ] **Step 2: Write RED legacy normalization tests.** Active continuation with NULL/stale state exposes actions disabled, normalizes under expected version/deleted CAS, rebinds committed version/hash, then enables Resume/Discard. Rollback re-reads unchanged owner; conflict re-reads and hides/quarantines changed/deleted/replaced owner.
- [ ] **Step 3: Write RED history/terminal tests.** Exclude accepted, dispatch_started, failed, discarded, and empty complete. Include continuation_active only through valid ADR-063 projection. Preserve bounded stopped partial content. Empty complete renders explicit copy.
- [ ] **Step 4: Write RED remote/import tests.** Accepted/dispatch_started without local checkpoint are inert literal source-device rows; source Retry/Discard absent; Retry as new response creates sibling and leaves original unchanged.
- [ ] **Step 5: Run RED.** Run the four scoped continuation/history files plus Task-5 sync/export round-trip tests.
- [ ] **Step 6: Implement the ownership handoff and lazy normalization.** Reuse repository transactions; do not call old `update_provider_continuation` then checkpoint delete as separate writes.
- [ ] **Step 7: Implement one history predicate.** Export and provider builders call `assistant_state_allows_provider_history(state, has_valid_continuation, content)` instead of local conditionals.
- [ ] **Step 8: Run GREEN and mutation probes.** Remove pre-tool checkpoint deletion ordering and one message-version/deletion predicate separately; confirm named tests fail; restore and run Ruff.
- [ ] **Step 9: Commit.** Commit `feat(console): hand dispatch recovery to continuations`.

### Task 17: Verify and close TASK-19900.3

**Files:**
- Modify: `backlog/tasks/task-19900.3 - Make-automatic-Console-Library-retrieval-a-truthful-send-gate.md`

**Interfaces:**
- Consumes: Tasks 12–16.
- Produces: one crash-honest automatic/never send path.

- [ ] **Step 1: Run the targeted delivery battery.** Run `Tests/Chat/test_console_turn_preparation.py`, `Tests/Chat/test_library_preparation.py`, `Tests/Chat/test_console_automatic_library_preparation.py`, `Tests/Chat/test_console_durable_turn_acceptance.py`, `Tests/Chat/test_console_first_send_atomicity.py`, `Tests/Chat/test_console_dispatch_recovery.py`, `Tests/Chat/test_console_dispatch_queue_recovery.py`, `Tests/UI/test_console_dispatch_recovery.py`, `Tests/Chat/test_console_dispatch_continuation_handoff.py`, `Tests/Chat/test_console_assistant_generation_history.py`, `Tests/Chat/test_console_chat_controller.py`, `Tests/Chat/test_console_prompt_queue_coordinator.py`, `Tests/Chatbooks/test_provider_continuation_roundtrip.py`, `Tests/Sync_Interop/test_chat_outbox_producer.py`, `Tests/Sync_Interop/test_envelope_builder.py`, `Tests/Sync_Interop/test_envelope_applier.py`, and `Tests/Sync_Interop/test_provider_continuation_reconciliation.py`.
- [ ] **Step 2: Run fault-injection cases three times.** Expected: stable pass with no duplicate conversation/USER/assistant, no provider invocation before dispatch_started, and no queue advancement with unresolved recovery.
- [ ] **Step 3: Run scoped Ruff and `git diff --check`.** Do not launch the real profile and do not run the full suite without owner opt-in.
- [ ] **Step 4: Complete child AC/notes/lessons/Done hygiene.** Include the exact targeted commands and migration isolation in Implementation Notes.
- [ ] **Step 5: Commit.** Commit `test(console): qualify Library send recovery`.

---

## Delivery 4 — Policy, search, source, and responsive UI (`TASK-19900.4`)

### Task 18: Start TASK-19900.4 and build the two-axis chip plus Library Access modal

**Files:**
- Modify: `backlog/tasks/task-19900.4 - Split-Console-Library-policy-search-and-source-surfaces.md`
- Modify: `tldw_chatbook/Chat/console_display_state.py:538-640`
- Modify: `tldw_chatbook/Widgets/Console/console_status_chips.py:176-215`
- Create: `tldw_chatbook/Widgets/Console/console_library_access_modal.py`
- Create: `tldw_chatbook/UI/Console_Modules/library_policy.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Chat/test_console_display_state.py`
- Modify: `Tests/UI/test_console_status_chips.py`
- Create: `Tests/UI/test_console_library_access_modal.py`

**Interfaces:**
- Consumes: holder/coordinator snapshots and resolved destination runtime state.
- Produces: `ConsoleLibraryPolicyDisplayState`, `ConsoleLibraryChip.OpenRequested`, `ConsoleLibraryAccessModal`, and `ConsoleLibraryPolicyController`.

```python
@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyDisplayState:
    chip_label: str
    source_status: str
    auto_retrieve_label: Literal["Never", "Automatic"]
    assistant_access_label: Literal["Blocked", "Allowed"]
    provider_intent_label: str
    resolved_destination_label: str
    feedback: Literal["idle", "saving", "saved", "conflict", "unavailable", "error"]
    feedback_copy: str
    save_enabled: bool
    editing_enabled: bool
```

- [ ] **Step 1: Start the child and baseline UI tests.** Put TASK-19900.4 In Progress with this plan link. Run current status-chip, RAG-modal, staged-context, right-rail, Settings, and screen-size-ratchet tests.
- [ ] **Step 2: Write RED chip tests.** Pin the four exact strings/unavailable string, one chip only, fixed noun/axis order, no readiness/scope/mode/source count axes, keyboard/click opener parity, and focus return.
- [ ] **Step 3: Write RED modal state tests.** Cover durable saved/missing, temporary, Saving/Saved/Conflict/Unavailable/Error, dirty/clean Escape/backdrop/Cancel, Save disabled when clean/in-flight, Reload and Compare/Retry, missing conversation, failed save preserving prior committed holder, and literal provider/category copy.
- [ ] **Step 4: Run RED.** Run the three scoped test files.
- [ ] **Step 5: Implement pure state then widgets.** The controller constructor lists late-binding dependencies and is created in `wiring.py`. The modal uses text-valued RadioSet rows, explicit Save/Cancel, no config reads, and `SafeModalDismissMixin`.
- [ ] **Step 6: Keep screen growth below the ratchet.** `ChatScreen` only routes events and snapshots; policy load/save/focus logic belongs in the controller/widget files.
- [ ] **Step 7: Run GREEN, lint, screen ratchet, and commit.** Commit `feat(console): add Library access controls`.

### Task 19: Split Search Library from policy and add canonical future-session Settings

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_library_search_modal.py`
- Remove after imports migrate: `tldw_chatbook/Widgets/Console/console_rag_settings_modal.py`
- Modify: `tldw_chatbook/UI/Console_Modules/retrieval.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:9750-9870`
- Modify: `tldw_chatbook/UI/Screens/settings_library_rag_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_rag_profile_adapter.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:6900-7010,14580-14720,19320-19380`
- Modify: `tldw_chatbook/config.py`
- Replace: `Tests/UI/test_console_rag_settings_modal.py` with `Tests/UI/test_console_library_search_modal.py`
- Modify: `Tests/UI/test_settings_library_rag_defaults.py`, `test_settings_rag_profile_adapter.py`, `test_settings_rag_profile_region.py`, `test_console_library_tool_setting.py`

**Interfaces:**
- Consumes: current composer draft, manual source filters, item-scope summary, future-session config defaults.
- Produces: `ConsoleLibrarySearchResult(query, run, source_types)`, exact draft prefill, and Settings fields `rag_auto_retrieve_on_send` / `assistant_library_access_default` distinct from live `direct_library_tools`.

```python
@dataclass(frozen=True, slots=True)
class ConsoleLibrarySearchResult:
    query: str
    run: bool
    source_types: tuple[str, ...]
```

- [ ] **Step 1: Write RED search-only tests.** Exact composer draft always prefills—including text the removed heuristic rejected—query edits do not alter draft, source controls say `This search only`, Search/Cancel work, standing policy is absent and unchanged.
- [ ] **Step 2: Write RED Settings tests.** The `New Console conversations` group round-trips Never/Automatic and Blocked/Allowed, explains future-only scope, keeps Direct/RAG in a separate tool-mode row, and adds no control to deprecated Settings files.
- [ ] **Step 3: Run RED.** Run the replacement modal tests and four Settings files.
- [ ] **Step 4: Implement the search-only result and delete the query heuristic.** Remove `_console_draft_looks_like_rag_query`, auto-retrieve switch IDs/callbacks, and the screen's immediate config writer. Preserve harmless per-search filter state.
- [ ] **Step 5: Implement canonical Settings load/save.** Extend the existing defaults dataclass and adapter; use RadioSet/text-state rows rather than unlabeled Switches. Direct/RAG remains live and separate.
- [ ] **Step 6: Run GREEN, scan deprecated surfaces, lint, and commit.** `rg -n 'assistant_library_access_default|New Console conversations' tldw_chatbook/UI/Tools_Settings_Window.py tldw_chatbook/Widgets/enhanced_settings_sidebar.py` must return no matches. Commit `feat(console): separate Library search and defaults`.

### Task 20: Compact source rows and add Selected turn terminology/affordances

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py:640-950`
- Modify: `tldw_chatbook/Widgets/Console/console_staged_context.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:5680-5730`
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:12000-12150,14620-14920`
- Modify: `Tests/UI/test_console_staged_context.py`
- Modify: `Tests/UI/test_console_right_rail.py`
- Modify: `Tests/UI/test_console_transcript_selection_contract.py`

**Interfaces:**
- Consumes: staged evidence references, citation rows, selected message/turn ID.
- Produces: one `ConsoleSourcePrimaryRow` per source, activated detail rows, `Sources — next send`, `Cited sources (N)`, and Selected turn subsection focus event.

```python
@dataclass(frozen=True, slots=True)
class ConsoleSourcePrimaryRow:
    source_id: str
    status: Literal["ready", "warning", "blocked"]
    title: str
    source_type: str
    snippet: str
    authority: str
    freshness: str
    action_label: str
```

- [ ] **Step 1: Write RED compactness tests.** Ten references produce ten primary rows, not provenance-row multiplication. Each row renders status/title/type on one line; activation alone reveals snippet/authority/freshness/action.
- [ ] **Step 2: Write RED terminology/selection tests.** Staged/cited/activity labels are exact; a message activity affordance selects that turn and focuses the Inspector subsection; citation affordance focuses Cited sources; empty states remain explicit.
- [ ] **Step 3: Write literal text tests.** `[red]`, CJK, emoji, combining marks, and RTL-shaped strings render as text in source, recovery, title, and activity-adjacent sinks.
- [ ] **Step 4: Run RED.** Run the three scoped files.
- [ ] **Step 5: Implement primary/detail models and routing.** Details remain mounted/collapsible or lazily composed by source ID; do not cache removable child widgets in the rail region.
- [ ] **Step 6: Run GREEN, lint, and commit.** Commit `feat(console): clarify Library source review`.

### Task 21: Make both modals responsive and verify the production composition

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_console_library_controls_render.py`
- Modify: `Tests/Architecture/test_screen_size_ratchet.py` only to lower, never raise, a ceiling earned by extraction

**Interfaces:**
- Consumes: production `ChatScreen`, full stylesheet bundle, both modals, policy/search/source states.
- Produces: viewport-fit width, bounded scrolling body, pinned visible actions, stacked narrow layout, stable focus, and painted containment evidence.

- [ ] **Step 1: Write RED production-mount render tests.** Mount the real screen hierarchy and full CSS at standard and narrow sizes. Assert modal/content/action rectangles stay inside viewport, body scrolls, actions remain visible, and no horizontal clipping occurs.
- [ ] **Step 2: Add expanded-content matrix.** Render provider/model/copy at +30%, CJK, emoji, combining marks, RTL-shaped text, policy conflict/error, blocked/saving/selected states, and ten source rows. Assert labels remain painted and focusable.
- [ ] **Step 3: Run RED.** Run `python -m pytest Tests/UI/test_console_library_controls_render.py -q`; expected: fixed-width/overflow failures.
- [ ] **Step 4: Implement component CSS and regenerate bundle.** Use viewport-relative max width, `min-width: 0`, bounded scroll body, and a pinned action row. At narrow width stack radio/disclosure/action rows without dimension-changing hover/focus. Use semantic tokens only.
- [ ] **Step 5: Verify disabled readability and safe dismissal.** Test disabled reasons as text, repeated Save/Retry idempotence, dirty backdrop/Escape confirmation, clean close, opener focus restoration, and error focus movement.
- [ ] **Step 6: Run GREEN and UI quality checks.** Run `python tldw_chatbook/css/build_css.py`, `python tldw_chatbook/css/check_bundle_sync.py`, and `python -m pytest Tests/UI/test_console_library_controls_render.py Tests/UI/test_console_status_chips.py Tests/UI/test_console_library_access_modal.py Tests/UI/test_console_library_search_modal.py Tests/UI/test_console_staged_context.py Tests/UI/test_console_right_rail.py Tests/Architecture/test_screen_size_ratchet.py -q`. Then run exactly one detector pass: `node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs --json tldw_chatbook/Widgets/Console/console_library_access_modal.py tldw_chatbook/Widgets/Console/console_library_search_modal.py tldw_chatbook/Widgets/Console/console_status_chips.py tldw_chatbook/Widgets/Console/console_staged_context.py tldw_chatbook/UI/Console_Modules/right_rail.py tldw_chatbook/css/components/_agentic_terminal.tcss`.
- [ ] **Step 7: Close TASK-19900.4 and commit.** Complete ACs/notes/evidence/lessons/Done, then commit `test(console): qualify Library control surfaces`.

---

## Delivery 5 — Minimized assistant Library activity (`TASK-19900.5`)

### Task 22: Start TASK-19900.5 and implement bounded activity capture contracts

**Files:**
- Modify: `backlog/tasks/task-19900.5 - Capture-and-review-minimized-assistant-Library-activity.md`
- Create: `tldw_chatbook/Chat/library_activity.py`
- Modify: `tldw_chatbook/Agents/run_context.py`
- Modify: `tldw_chatbook/Agents/agent_service.py:2150-2405`
- Modify: `tldw_chatbook/Agents/library_tool_provider.py`
- Modify: `tldw_chatbook/Agents/library_rag_tool_provider.py`
- Create: `Tests/Chat/test_library_activity.py`
- Create: `Tests/Agents/test_library_activity_capture.py`

**Interfaces:**
- Consumes: turn/attempt identity, built-in Direct/RAG structured results, run actor context.
- Produces: `CurrentRunActor`, `LibraryActivityCandidate`, `LibraryActivityEvent`, `minimize_library_activity()`, and `LibraryActivitySink = Callable[[LibraryActivityEvent], None]`.

- [ ] **Step 1: Start child and baseline providers.** Put TASK-19900.5 In Progress with this plan link. Run direct/RAG provider tests and agent-service run-context tests.
- [ ] **Step 2: Write RED minimization tests.** Pin version, attempt/run/actor/mode/operation/status/count; cap preview/title/error/ID/reference count/bytes; reject bodies/snippets/excerpts/paths/credentials/provider request fields/arbitrary exceptions; log only event ID/size/status/error category.
- [ ] **Step 3: Write RED attribution tests.** Bind primary and subagent run IDs plus parent ID through tool invocation threads; retry attempts remain distinct; no bound actor causes a bounded capture failure, not unattributed success.
- [ ] **Step 4: Write RED trusted-boundary tests.** Direct and RAG capture after authoritative structured result but before result truncation/model delivery. Sink/minimization failure returns existing `ERROR_STORAGE_ERROR` with literal copy `Library result withheld because activity could not be recorded.` and withholds the original result. Conversation-policy Blocked registers no provider/event.
- [ ] **Step 5: Run RED.** Run both new files and provider tests.
- [ ] **Step 6: Extend run context with actor identity.** Bind and reset this exact value around every provider invocation:

```python
@dataclass(frozen=True, slots=True)
class CurrentRunActor:
    kind: Literal["primary", "subagent"]
    run_id: str
    parent_run_id: str | None
```

- [ ] **Step 7: Add recorder callbacks to both providers.** Call minimization/sink before serializing or returning the model-visible `ToolResult`; on failure return `review_capture_failed` without payload.
- [ ] **Step 8: Run GREEN, mutation-check capture withholding, lint, and commit.** Commit `feat(agents): capture minimized Library activity`.

### Task 23: Add the store-owned buffer, sidecar projection, promotion, and export redaction

**Files:**
- Create: `tldw_chatbook/Chat/console_library_activity_buffer.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/trajectory.py`
- Modify: `tldw_chatbook/Chat/trajectory_export.py`
- Modify: `tldw_chatbook/Chat/trajectory_import.py`
- Create: `Tests/Chat/test_console_library_activity_buffer.py`
- Create: `Tests/Chat/test_library_activity_projection.py`
- Modify: trajectory export/import tests

**Interfaces:**
- Consumes: bounded events, generic contribution protocol, trajectory rows, active branch/turn selection.
- Produces: `ConsoleLibraryActivityBuffer.admit/flush/retry/final_flush`, `LibraryActivityContribution`, and `project_library_activity(rows, active_turn_ids, selected_turn_id) -> LibraryActivityView`.

```python
class ConsoleLibraryActivityBuffer:
    def admit(self, session_id: str, turn_id: str, event: LibraryActivityEvent) -> None:
        """Retain one already-minimized event under the owning session/turn."""

    def flush(self, session_id: str) -> LibraryActivityFlushResult:
        """Persist the current bounded batch and remove only confirmed rows."""

    def retry(self, session_id: str) -> LibraryActivityFlushResult:
        """Retry the same retained batch without duplicating confirmed rows."""

    def final_flush(self, session_id: str) -> LibraryActivityFlushResult:
        """Perform the bounded close, promotion, or shutdown flush."""
```

- [ ] **Step 1: Write RED concurrency/buffer tests.** Concurrent provider threads admit unique ordered events; transient persistence failures retain the exact bounded batch across navigation; Retry flushes once; exhaustion exposes `Library activity not saved in this session`; close/promotion/shutdown perform one bounded final flush.
- [ ] **Step 2: Write RED anchoring/projection tests.** Anchor to durable opener, distinguish attempt/run/parent actor, filter active lineage/selected turn, retain other branches without showing them, and prove generic trajectory neither displaces nor duplicates anchors.
- [ ] **Step 3: Write RED promotion rollback tests.** Ephemeral events remain memory-only; atomic promotion contributes rows; each failure restores buffer/session/event identities and leaves no partial rows.
- [ ] **Step 4: Write RED export tests.** Default export retains operation/status/count plus bounded preview and removes query/source details; full opt-in retains only the already bounded event; preparation uses its own bounded serializer; checkpoint table is never selected; import remains inert.
- [ ] **Step 5: Run RED.** Run the two new files plus trajectory export/import tests.
- [ ] **Step 6: Implement lock-owned buffer and pure projection.** Buffer keys are `(session_id, turn_id, attempt_id, run_id, event_id)`; persistence removes rows only after repository confirmation. Never bind Screen lifetime.
- [ ] **Step 7: Register `library_activity` as sidecar-only.** `derive_trajectory` skips it entirely; separate projection parses exact v1 payloads and returns bounded corrupt-row status for malformed data.
- [ ] **Step 8: Run GREEN, lint, and commit.** Commit `feat(console): persist Library activity sidecars`.

### Task 24: Add Selected turn activity review and close TASK-19900.5

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/UI/test_console_library_activity.py`
- Modify: `Tests/UI/test_console_right_rail.py`
- Modify: `Tests/UI/test_console_transcript_selection_contract.py`
- Modify: `backlog/tasks/task-19900.5 - Capture-and-review-minimized-assistant-Library-activity.md`

**Interfaces:**
- Consumes: `LibraryActivityView`, selected turn, buffer save state.
- Produces: `Library activity (N actions)`, explicit empty/not-saved/retry states, and message affordance focus.

- [ ] **Step 1: Write RED Inspector tests.** Selected turn contains Cited sources then Library activity; operation/actor/mode/status/count/time/bounded refs render; empty state exact; unsaved buffer state and Retry visible; activity is not a top-level rail peer.
- [ ] **Step 2: Write RED message affordance tests.** Activating activity on an assistant selects the owning turn and focuses activity; branch changes reproject; citations remain separate; activity never appears in staged Sources.
- [ ] **Step 3: Write RED literal/overflow tests.** Long/malicious titles and errors stay literal and bounded; 8 refs fit/scroll without moving pinned rail actions.
- [ ] **Step 4: Run RED, implement thin projection wiring, and run GREEN.** Keep persistence retry in the store/controller, not the widget.
- [ ] **Step 5: Run Delivery-5 battery and quality gates.** Run activity model/provider/buffer/projection/UI/trajectory tests, scoped Ruff, screen ratchet, and `git diff --check`.
- [ ] **Step 6: Mutation-check minimization and separation.** Remove a source-body rejection and sidecar-only exclusion separately; confirm named tests fail; restore.
- [ ] **Step 7: Complete child AC/notes/lessons/Done and commit.** Commit `test(console): qualify Library activity review`.

---

## Delivery 6 — Documentation and production-path qualification (`TASK-19900.6`)

### Task 25: Start TASK-19900.6 and reconcile user/developer documentation

**Files:**
- Modify: `backlog/tasks/task-19900.6 - Qualify-and-document-Console-Library-conversation-controls.md`
- Modify: `Docs/User_Guide/console.md`, `Docs/User_Guide/console/context-and-rag.md`, `Docs/User_Guide/settings/rag.md`, `Docs/User_Guide/library/search-and-rag.md`, `Docs/User_Guide/library/import-and-export.md`, `Docs/User_Guide/library/media-and-conversations.md`, `Docs/Development/Agent-Tools/local-library-tools.md`, `Docs/Design/MCP.md`, `Docs/User_Guide/mcp.md`, and `README.md` only where their current claims intersect this feature
- Modify: `Docs/superpowers/specs/2026-08-22-console-library-controls-design.md` only for verified implementation deviations approved by the owner
- Create: `Tests/Docs/test_console_library_controls_docs.py`

**Interfaces:**
- Consumes: completed product behavior from Deliveries 1–5.
- Produces: consistent three-mechanism/two-axis/device-local/fixed-category/destination/recovery documentation.

- [ ] **Step 1: Start the child and inventory claims.** Put TASK-19900.6 In Progress with this plan link. Generate a bounded list of docs containing old enable/disable or global-auto wording.
- [ ] **Step 2: Write a documentation truth table.** Every relevant doc must say manual search is always available, automatic and assistant axes are independent, Direct/RAG is a selector, policy/checkpoint are local, assistant state syncs, remote unresolved state is inert, and ADR-063 owns continuation after handoff.
- [ ] **Step 3: Update only stale claims.** Preserve unrelated prose and do not document UI/control names not present in production.
- [ ] **Step 4: Write and run the documentation contract test.** `Tests/Docs/test_console_library_controls_docs.py` loads the ten files above, asserts required three-mechanism/two-axis/device-local/Direct-RAG/recovery statements, parses Markdown links relative to each file, skips only `http://`/`https://`, and asserts every local target exists. Run `python -m pytest Tests/Docs/test_console_library_controls_docs.py -q` plus `rg -n 'TASK-19900|ADR-079|2026-08-22-console-library-controls' Docs backlog README.md`.
- [ ] **Step 5: Commit.** Commit `docs(console): explain per-conversation Library controls`.

### Task 26: Add deterministic production-UI scenarios and run integrated targeted gates

**Files:**
- Create: `Tests/Fixtures/console_library_recording_provider.py`
- Create: `Tests/UI/test_console_library_controls_workflow.py`
- Create: `Tests/Integration/test_console_library_control_integration.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`, `Tests/Chat/test_console_prompt_queue_coordinator.py`, `Tests/Chat/test_console_agent_bridge.py`, `Tests/UI/test_console_status_chips.py`, `Tests/UI/test_console_right_rail.py`, and `Tests/UI/test_console_library_controls_render.py`

**Interfaces:**
- Consumes: production ChatScreen/controller/store/gateway/agent bridge and isolated SQLite databases.
- Produces: deterministic queued/subagent/destination/retrieval/recovery UI scenarios independent of arbitrary model tool choice.

- [ ] **Step 1: Implement a recording provider fixture.** It returns scripted readiness, zero/success/failure retrieval, token chunks, tool-call batch, continuation, timeout, and terminal outcomes while recording call count/order/destination/request metadata without retaining Library bodies.
- [ ] **Step 2: Drive all four policy combinations through production composition.** Assert provider schemas/calls, automatic preparation, manual search availability, exact chip/modal copy, activity count, and destination disclosure.
- [ ] **Step 3: Drive queue/subagent/restart/recovery branches.** Cover precommit cancel, accepted restart, dispatch-started restart, Retry/Discard, queue pause/advance, subagent inherited/narrowed authority, and continuation-exclusive handoff.
- [ ] **Step 4: Run the integrated targeted battery.** Run the exact command below and capture test count/time/output in TASK-19900.6 notes.

```bash
python -m pytest \
  Tests/Chat/test_console_library_policy.py \
  Tests/Chat/test_assistant_generation_state.py \
  Tests/DB/test_chachanotes_console_library_migration_seed_openers.py \
  Tests/DB/test_chachanotes_console_library_policy_migration.py \
  Tests/ChaChaNotesDB/test_migration_atomicity.py \
  Tests/ChaChaNotesDB/test_console_library_policy_repository.py \
  Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py \
  Tests/Chat/test_assistant_generation_state_roundtrip.py \
  Tests/Chat/test_console_library_policy_coordinator.py \
  Tests/Chat/test_console_transaction_contribution.py \
  Tests/Chat/test_console_chat_store_library_policy.py \
  Tests/Chat/test_console_chat_store_atomic_promotion.py \
  Tests/Chat/test_console_turn_library_authority.py \
  Tests/Chat/test_console_library_destination.py \
  Tests/Agents/test_library_name_reservation.py \
  Tests/Chat/test_console_library_runtime_policy.py \
  Tests/Chat/test_console_turn_preparation.py \
  Tests/Chat/test_library_preparation.py \
  Tests/Chat/test_console_automatic_library_preparation.py \
  Tests/Chat/test_console_durable_turn_acceptance.py \
  Tests/Chat/test_console_first_send_atomicity.py \
  Tests/Chat/test_console_dispatch_recovery.py \
  Tests/Chat/test_console_dispatch_queue_recovery.py \
  Tests/UI/test_console_dispatch_recovery.py \
  Tests/Chat/test_console_dispatch_continuation_handoff.py \
  Tests/Chat/test_console_assistant_generation_history.py \
  Tests/UI/test_console_library_access_modal.py \
  Tests/UI/test_console_library_search_modal.py \
  Tests/UI/test_console_library_controls_render.py \
  Tests/Chat/test_library_activity.py \
  Tests/Agents/test_library_activity_capture.py \
  Tests/Chat/test_console_library_activity_buffer.py \
  Tests/Chat/test_library_activity_projection.py \
  Tests/UI/test_console_library_activity.py \
  Tests/Docs/test_console_library_controls_docs.py \
  Tests/UI/test_console_library_controls_workflow.py \
  Tests/Integration/test_console_library_control_integration.py \
  Tests/Sync_Interop/test_chat_outbox_producer.py \
  Tests/Sync_Interop/test_envelope_builder.py \
  Tests/Sync_Interop/test_envelope_applier.py \
  Tests/Sync_Interop/test_provider_continuation_reconciliation.py \
  Tests/Chatbooks/test_provider_continuation_roundtrip.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_prompt_queue_coordinator.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/UI/test_settings_library_rag_defaults.py \
  Tests/UI/test_settings_rag_profile_adapter.py \
  Tests/UI/test_settings_rag_profile_region.py \
  Tests/Architecture/test_screen_size_ratchet.py \
  -q
```

- [ ] **Step 5: Run scoped static checks.** Run `python -m ruff check Tests/Fixtures/console_library_recording_provider.py Tests/UI/test_console_library_controls_workflow.py Tests/Integration/test_console_library_control_integration.py Tests/Docs/test_console_library_controls_docs.py`, `python tldw_chatbook/css/build_css.py`, `python tldw_chatbook/css/check_bundle_sync.py`, `python -m pytest Tests/UI/test_css_bundle_sync_guard.py Tests/Architecture/test_screen_size_ratchet.py -q`, and `git diff --check`. The product modules were already Ruff-checked in their owning delivery; do not replace those per-delivery gates with this fixture-only command.
- [ ] **Step 6: Run counterfactual checks.** Mutation-check authorization, static reservation, retrieval no-dispatch, activity minimization, continuation pre-tool handoff, message-version/deletion CAS, sync/export normalization, rollback drain, and literal rendering; record each named test that turns red.
- [ ] **Step 7: Commit.** Commit `test(console): cover Library control workflows`.

### Task 27: Run isolated live qualification and close the work

**Files:**
- Modify: `backlog/tasks/task-19900.6 - Qualify-and-document-Console-Library-conversation-controls.md`
- Modify: `backlog/tasks/task-19900 - Make-Console-Library-controls-explicit-per-conversation.md`
- Modify child task files only for final AC/notes/status hygiene
- Modify `backlog/docs/lessons-*.md` only if qualification reveals an incident-backed reusable lesson

**Interfaces:**
- Consumes: completed code, deterministic gates, isolated profile/data directory.
- Produces: live evidence, all child DoD, parent implementation notes, and merge-ready branch.

- [ ] **Step 1: Establish isolation before launch.** Create a temporary profile/data directory with `mktemp -d`, set both config and data-directory overrides explicitly, print resolved DB paths, and assert none points under the user's normal data directory. Do not use only `TLDW_CONFIG_PATH` as isolation.
- [ ] **Step 2: Run the live checklist.** Through the real Console verify first persistence/restart, four policy states, Direct/RAG selection, exact manual draft prefill, automatic success/zero/failure Retry/Bypass/Cancel, Never persistence Retry/Cancel only, queued behavior, deterministic subagent behavior, activity review, on-device→external disclosure, soft delete/restore, and unresolved accepted/dispatch-started recovery.
- [ ] **Step 3: Verify permanent purge at repository level.** Console has no hard-delete action; use a `tmp_path` database test to assert conversation FK cascade removes policy/checkpoint/trajectory sidecars. Do not invent a live UI step.
- [ ] **Step 4: Ask before the full suite.** Present the integrated targeted evidence and ask whether the owner wants `python -m pytest` across the repository. If approved, run it in the isolated environment and record exact failures/passes; if declined, record that targeted verification was the approved scope.
- [ ] **Step 5: Close TASK-19900.6.** Check ACs, add concise implementation notes and evidence, complete ADR/lessons/docs review, and mark Done only when DoD holds.
- [ ] **Step 6: Close the parent.** Re-read every child with `backlog task <id> --plain`, confirm all ACs checked/Done, update parent ACs and final implementation notes, recheck open PR/task/ADR identifiers, then mark TASK-19900 Done.
- [ ] **Step 7: Run final verification-before-completion.** Run the agreed test/static commands, `git status --short`, `git diff --check`, and inspect the final diff for secrets, logs, sync/export leakage, and accidental deprecated-Settings edits.
- [ ] **Step 8: Commit closeout.** Commit `docs(console): close Library controls delivery`.

---

## Spec Coverage Index

| Approved spec section | Implemented and verified by |
| --- | --- |
| §1 User model/defaults/existing arrivals | Tasks 2, 4, 7, 18, 19 |
| §2 Ownership/components/immutable destination/sidecars | Tasks 2, 6–10, 12, 22, 23 |
| §3 Schema/seed/CAS/first persistence/promotion/lifecycle | Tasks 3–7, 14–16, 23 |
| §4 Execution freshness/provider composition/reservation/egress | Tasks 8–11 |
| §5 Manual Search Library | Task 19 |
| §6 Automatic admission/query/state machine/recovery/disclosure | Tasks 12–17 |
| §7 Activity capture/anchoring/presentation/export | Tasks 22–24 |
| §8 Chip/access/settings/search/sources/responsive rendering | Tasks 18–21, 24 |
| §9 Failure behavior | Repository fault injection in Tasks 4, 6, 7, 14–16, 23 and UI recovery tests in Tasks 18, 21, 24 |
| §10 Security/privacy invariants | Tasks 5, 6, 8–10, 12–16, 22, 23 plus mutation gates in Task 26 |
| §11 Verification strategy | Per-delivery batteries in Tasks 7, 11, 17, 21, 24 and integrated/live qualification in Tasks 26–27 |

---

## Delivery Commit and Review Boundaries

Each child task is one reviewable PR/delivery. Do not begin a dependent child until its predecessor is integrated or the worktree is rebased onto the predecessor's exact final commit.

1. `TASK-19900.1`: schema v45, Sync-v1/v2 compatibility, repositories/coordinator, and lifecycle only.
2. `TASK-19900.2`: immutable runtime authority, egress, provider selection, reservation, and ephemeral gate only.
3. `TASK-19900.3`: automatic preparation, atomic acceptance, checkpoint recovery, queue, continuation handoff, and projections only.
4. `TASK-19900.4`: chip/modal/Settings/search/source/responsive UI only.
5. `TASK-19900.5`: minimized activity capture/buffer/projection/Inspector/export only.
6. `TASK-19900.6`: docs, deterministic integration, mutation evidence, isolated live QA, and closure only.

At every boundary, run `superpowers:requesting-code-review` against the child acceptance criteria and resolve technically verified findings before proceeding. At final completion, use `superpowers:verification-before-completion` and then `superpowers:finishing-a-development-branch`.
