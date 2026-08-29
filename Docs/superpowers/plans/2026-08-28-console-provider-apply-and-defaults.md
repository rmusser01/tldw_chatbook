# Console Provider Apply and Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make Console provider/model Apply commit to the exact originating chat, persist safe chat-owned settings, and add truthful per-model and new-chat default actions with explicit retryable failure states.

**Architecture:** Introduce one UI-neutral submission/origin contract shared by the quick popover and full Console Settings modal. ConsoleChatStore exposes a synchronous exact-session live commit that both modals invoke before dismissal, followed by a separate post-close durability step; ChatPersistenceService owns the safe generation metadata merge and the existing context-policy repository continues to own compaction. A focused locked config mutation patches literal model-profile paths and global chat defaults atomically, while ChatScreen coordinates post-close persistence/default work and exposes failures.

**Tech Stack:** Python 3.11, Textual 8.x, dataclasses/enums, TOML config mutation, SQLite conversation metadata, pytest/pytest-asyncio, Textual Pilot.

---

## Governing Contract

- Design: Docs/superpowers/specs/2026-08-27-console-provider-apply-persistence-design.md
- ADR required: yes
- ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md
- Reason: the feature establishes long-lived ownership boundaries for conversation metadata, model-profile defaults, global new-chat defaults, and runtime publication.
- Backlog task: backlog/tasks/task-22515 - Make-Console-provider-Apply-update-and-persist-conversation-settings.md
- Plan review: three independent review passes completed; all findings are addressed in this plan.
- Canonical settings surface: tldw_chatbook/UI/Screens/settings_screen.py remains the F9 settings owner; do not add features to legacy Tools_Settings_Window.py or enhanced_settings_sidebar.py.
- Verification constraint: run only the targeted commands below. Ask the user before any full pytest sweep.
- Live/config safety: config tests must use isolated temporary config paths. Never run a bare script that can write the real user config.
- UI geometry: mounted tests must use Tests.UI.consolidated_css.ConsolidatedCSSApp and production hierarchy/CSS.
- Dirty-worktree safety: before each task, record git status --short and inspect the
  named paths. Use git add -p -- for every tracked file that was already dirty; use
  exact git add -- only for newly created files. Never stage a directory. Before
  every commit run git diff --cached --check and inspect git diff --cached to prove
  no unrelated user hunk is staged.

## Task 1: Add the shared submission contract and controller-owned rebase

**Files:**

- Create: tldw_chatbook/Chat/console_settings_apply.py
- Modify: tldw_chatbook/Chat/console_session_settings.py:174-575
- Modify: tldw_chatbook/Chat/console_chat_controller.py:6680-6845
- Create: Tests/Chat/test_console_settings_apply.py
- Modify: Tests/Chat/test_console_session_settings.py

- [ ] **Step 1: Write failing tests for origin validation and field provenance**

Cover these cases:

- an origin stores the exact session ID and the persisted conversation ID observed when the modal opened;
- None to a first persisted ID is allowed for the same session;
- an already-persisted origin rebound to a different conversation ID is rejected;
- an origin captured while unsaved is rejected after an explicit rebind even when
  the new binding is its first non-null conversation ID;
- a missing/closed session is rejected;
- provider or model change rebases untouched generation fields from the target defaults;
- explicitly dirty fields survive rebasing and are marked dirty;
- target base_url replaces the source endpoint;
- fields not supported by the target are cleared;
- keyed A to B to A drafts restore the prior deliberate A edits;
- quick and full default masks are exact and compaction/system prompt/base_url are not profile fields.

Use immutable value objects with assertions against exact field sets. Start with:

    QUICK_MODEL_DEFAULT_FIELDS = frozenset({"temperature", "streaming"})
    FULL_MODEL_DEFAULT_FIELDS = frozenset({
        "temperature", "top_p", "min_p", "top_k", "max_tokens", "seed",
        "presence_penalty", "frequency_penalty", "reasoning_effort",
        "reasoning_summary", "verbosity", "thinking_effort",
        "thinking_budget_tokens", "streaming",
    })

- [ ] **Step 2: Run the new tests and confirm they fail for the missing module**

Run:

    pytest -q Tests/Chat/test_console_settings_apply.py Tests/Chat/test_console_session_settings.py -k "default_profile or rebase or origin"

Expected: collection/import failure for console_settings_apply before implementation.

- [ ] **Step 3: Implement the UI-neutral contract**

Add these public shapes in console_settings_apply.py:

    class ConsoleSettingsAction(str, Enum):
        APPLY_TO_CHAT = "apply_to_chat"
        SAVE_MODEL_DEFAULT = "save_model_default"
        MAKE_NEW_CHAT_DEFAULT = "make_new_chat_default"

    @dataclass(frozen=True, slots=True)
    class ConsoleSettingsOrigin:
        session_id: str
        persisted_conversation_id: str | None
        conversation_binding_revision: int

    @dataclass(frozen=True, slots=True)
    class ConsoleSettingsDraftState:
        settings: ConsoleSessionSettings
        context_policy_overrides: ConsoleContextPolicyOverrides
        field_drafts: tuple[ConsoleSettingsFieldDraft, ...]
        model_drafts: tuple[ConsoleModelDraft, ...]
        endpoint_draft: ConsoleEndpointDraft | None

    class ConsoleSettingsFieldProvenance(str, Enum):
        INHERITED = "inherited"
        EXPLICIT = "explicit"
        CARRIED = "carried"

    @dataclass(frozen=True, slots=True)
    class ConsoleSettingsFieldDraft:
        name: str
        effective_value: object | None
        profile_override: object | None
        provenance: ConsoleSettingsFieldProvenance
        dirty: bool

    @dataclass(frozen=True, slots=True)
    class ConsoleEndpointDraft:
        value: str
        bound_provider_config_key: str
        dirty: bool
        checked: bool

    @dataclass(frozen=True, slots=True)
    class ConsoleSettingsSubmission:
        submission_id: str
        action: ConsoleSettingsAction
        origin: ConsoleSettingsOrigin
        draft: ConsoleSettingsDraftState
        user_display_name_override: str | None
        default_field_mask: frozenset[str]

    @dataclass(frozen=True, slots=True)
    class ConsoleSettingsLiveCommit:
        submission_id: str
        session_id: str
        persisted_conversation_id: str | None
        conversation_binding_revision: int
        generation_revision: int
        context_policy_revision: int
        settings: ConsoleSessionSettings
        context_policy_overrides: ConsoleContextPolicyOverrides

    @dataclass(frozen=True, slots=True)
    class ConsoleSettingsCommittedSubmission:
        submission: ConsoleSettingsSubmission
        live_commit: ConsoleSettingsLiveCommit

Keep transition-to-full distinct from a submission:

    @dataclass(frozen=True, slots=True)
    class ConsoleSettingsTransfer:
        origin: ConsoleSettingsOrigin
        draft: ConsoleSettingsDraftState

Add pure helpers:

- remember_model_draft(state): replace one exact provider/literal-model key;
- field masks/constants used by both UI surfaces.

The field draft deliberately keeps effective_value separate from profile_override.
Apply freezes effective_value into the conversation. A blank/inherited profile
control keeps the resolved effective value for Apply but sets profile_override to
None so a default action deletes that exact model-profile field. This separation is
required for blank-to-inherit and full streaming Inherit semantics. When the
submitting surface is quick settings, temperature and streaming have no Inherit
control: build their profile_override from the displayed effective value even when
their provenance is inherited. Only the full surface may submit None for those
fields to delete an override.

Do not import Textual in this module. Do not persist anything here.

- [ ] **Step 4: Make ConsoleChatController the sole rebase owner**

Add:

    def rebase_console_settings_draft(
        self,
        state: ConsoleSettingsDraftState,
        *,
        provider: str,
        model: str | None,
        app_config: Mapping[str, object],
        exposed_fields: frozenset[str],
    ) -> ConsoleSettingsDraftState

The controller builds the target's complete effective defaults, clears the source
endpoint and unsupported provider-specific fields, restores an exact keyed draft
when present, and overlays only dirty supported fields. Widgets receive an injected
DraftRebaser callback backed by this method; they do not import or implement rebase
logic. The injected live_committer invokes this controller seam again against the
current runtime config immediately before ConsoleChatStore commits, so a stale
modal/config snapshot cannot bypass the owner.

- [ ] **Step 5: Expose one target-default builder from console_session_settings.py**

Add a small wrapper that accepts app_config, provider, and literal model ID, delegates to default_console_session_settings, and returns a fresh ConsoleSessionSettings snapshot. Keep the existing precedence:

    api_settings.<provider>.model_defaults[exact model]
      -> console.provider_defaults.<provider>
      -> chat_defaults
      -> provider settings

Use existing provider/model normalization and capability helpers; do not create a second precedence engine.

- [ ] **Step 6: Run tests and commit**

Run:

    pytest -q Tests/Chat/test_console_settings_apply.py Tests/Chat/test_console_session_settings.py

Expected: all selected tests pass.

Commit:

    git add -p -- tldw_chatbook/Chat/console_session_settings.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_session_settings.py
    git add -- tldw_chatbook/Chat/console_settings_apply.py Tests/Chat/test_console_settings_apply.py
    git diff --cached --check
    git diff --cached
    git commit -m "feat: define Console settings apply contract"

## Task 2: Add the safe conversation-generation metadata codec

**Files:**

- Create: tldw_chatbook/Chat/console_generation_settings_metadata.py
- Modify: tldw_chatbook/Chat/chat_persistence_service.py:1-330,650-730
- Modify: tldw_chatbook/Chat/console_conversation_hydration.py:280-345
- Create: Tests/Chat/test_console_generation_settings_metadata.py
- Modify: Tests/Chat/test_console_conversation_hydration.py
- Modify: Tests/Chat/test_console_chat_store.py:1998-2335,3331-3404

- [ ] **Step 1: Write failing codec and merge tests**

Pin a versioned metadata object under one owned key:

    "console_generation_settings": {
        "version": 1,
        "provider": "OpenAI",
        "model": "gpt-5",
        "temperature": 0.3,
        "streaming": true
    }

Test exact safe fields, finite numbers, strict booleans/integers, length bounds,
unknown-field rejection, distinct absent/invalid/future-version results, malformed
JSON, and preservation of unrelated metadata siblings. A writer must refuse to
replace invalid or future-owned data and emit one bounded warning per conversation
per session. Assert base_url, credentials, system_prompt, source, compaction, and
display-name data are never serialized by this codec.

- [ ] **Step 2: Run the codec tests and observe failure**

Run:

    pytest -q Tests/Chat/test_console_generation_settings_metadata.py Tests/Chat/test_console_conversation_hydration.py

Expected: missing codec/import failures.

- [ ] **Step 3: Implement the codec**

In console_generation_settings_metadata.py define:

    CONSOLE_GENERATION_SETTINGS_METADATA_KEY = "console_generation_settings"
    CONSOLE_GENERATION_SETTINGS_VERSION = 1

    @dataclass(frozen=True, slots=True)
    class ConsoleGenerationSettingsSnapshot:
        provider: str
        model: str | None
        temperature: float | None
        top_p: float | None
        min_p: float | None
        top_k: int | None
        max_tokens: int | None
        seed: int | None
        presence_penalty: float | None
        frequency_penalty: float | None
        reasoning_effort: str | None
        reasoning_summary: str | None
        verbosity: str | None
        thinking_effort: str | None
        thinking_budget_tokens: int | None
        streaming: bool

    def snapshot_from_session_settings(
        settings: ConsoleSessionSettings
    ) -> ConsoleGenerationSettingsSnapshot
    def parse_console_generation_settings(metadata: object) -> ConsoleGenerationSettingsReadResult
    def merge_console_generation_settings(metadata: object, snapshot: ConsoleGenerationSettingsSnapshot) -> dict[str, object]

ConsoleGenerationSettingsReadResult has exact ABSENT, VALID, INVALID, and
UNSUPPORTED_VERSION statuses and carries a snapshot only for VALID. Use strict
JSON-object parsing equivalent to ChatPersistenceService._initial_metadata_object.
Reject non-finite floats and bool-as-int. Keep a literal allowlist; never serialize
the dataclass with asdict.

- [ ] **Step 4: Add bounded merge-safe persistence methods**

Add to ChatPersistenceService:

    def get_conversation_generation_settings(
        self, conversation_id: str
    ) -> ConsoleGenerationSettingsReadResult

    def update_conversation_generation_settings(
        self,
        *,
        conversation_id: str,
        snapshot: ConsoleGenerationSettingsSnapshot,
        expected_snapshot: ConsoleGenerationSettingsSnapshot | None,
    ) -> ConsoleGenerationSettingsWriteResult

Implement a complete-owned-snapshot CAS without changing the approved v1 metadata
allowlist. A missing owned object matches only expected_snapshot=None; a valid
object must exactly equal expected_snapshot across every safe field. INVALID or
UNSUPPORTED_VERSION refuses the write without mutation.
Conversation-version conflicts may be retried only after a fresh read proves the
owned snapshot is still the expected base; merge unrelated siblings from that fresh
record. If the owned snapshot changed, return SUPERSEDED/CONFLICT and never write
the older snapshot. Return MISSING for a missing conversation and propagate only
the final sibling-only ConflictError.

- [ ] **Step 5: Restore settings and the durable safe snapshot together**

Replace the settings-only apply_resume_settings_overrides result with:

    @dataclass(frozen=True, slots=True)
    class ConsoleGenerationSettingsHydration:
        settings: ConsoleSessionSettings
        durable_snapshot: ConsoleGenerationSettingsSnapshot | None
        metadata_status: ConsoleGenerationSettingsReadStatus

    def hydrate_console_generation_settings(
        app_config: Mapping[str, object],
        conversation: Mapping[str, object],
    ) -> ConsoleGenerationSettingsHydration

Hydration:

1. parses the generation snapshot;
2. derives current defaults for the saved provider/model from current app_config;
3. overlays only the saved safe fields;
4. resolves base_url from current config, never metadata;
5. then applies the conversation system prompt and pinned prefill through their existing owners.

Task 5 callers pass app_config rather than an active-chat-derived settings base and
thread durable_snapshot/status into ConsoleChatStore.restore_persisted_session.
Store seeds generation_durable_snapshot from this result. INVALID and
UNSUPPORTED_VERSION use config-derived live settings, retain the original metadata
untouched, and expose one bounded warning when the conversation is restored or its
Model section is first shown.

- [ ] **Step 6: Run tests and commit**

Run:

    pytest -q Tests/Chat/test_console_generation_settings_metadata.py
    pytest -q Tests/Chat/test_console_conversation_hydration.py Tests/Chat/test_console_chat_store.py -k "metadata or generation_settings or roleplay"

Expected: selected tests pass and unrelated metadata remains semantically preserved.

Commit:

    git add -p -- tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_conversation_hydration.py Tests/Chat/test_console_conversation_hydration.py Tests/Chat/test_console_chat_store.py
    git add -- tldw_chatbook/Chat/console_generation_settings_metadata.py Tests/Chat/test_console_generation_settings_metadata.py
    git diff --cached --check
    git diff --cached
    git commit -m "feat: persist safe Console generation settings"

## Task 3: Make ConsoleChatStore the exact-origin live and durable owner

**Files:**

- Modify: tldw_chatbook/Chat/console_chat_store.py:698-870,4019-4090,4198-4235,8420-8750
- Modify: tldw_chatbook/Chat/console_context_repository.py:180-275
- Modify: tldw_chatbook/Chat/chat_persistence_service.py:190-230
- Modify: Tests/Chat/test_console_chat_store.py
- Create: Tests/Chat/test_console_settings_apply_store.py
- Create: Tests/Chat/test_console_context_policy_cas.py

- [ ] **Step 1: Write failing store tests**

Test:

- a submission updates the exact origin while another session is active;
- a missing session or persisted-ID rebound is rejected without mutation;
- binding revision rejects an explicit unsaved-session rebind while accepting the
  normal first-persistence publication;
- duplicate activation of the same submission commits once;
- generation and context-policy live values both remain after either durable write fails;
- failures are stored per component and settings revision;
- Retry writes only the failed component and only if its revision is still current;
- a newer Apply supersedes an older failure;
- two rapid Applies serialize per session and the older generation/context write
  cannot become durable after the newer one;
- an external owned-base change causes CAS refusal rather than overwrite;
- ordinary Apply does not clear an app-global default failure;
- unsaved sessions stage both components without creating an empty conversation;
- first persistence includes generation metadata in the create write and flushes context policy through its current repository;
- temporary sessions do not write; promotion uses the staged settings on first persistence.
- a resumed conversation seeds context_policy_durable_revision from
  ContextPolicyReadResult.revision, then a new policy snapshot succeeds from N to
  N+1 without a false failure row.

- [ ] **Step 2: Run the store tests and observe failure**

Run:

    pytest -q Tests/Chat/test_console_settings_apply_store.py Tests/Chat/test_console_context_policy_cas.py
    pytest -q Tests/Chat/test_console_chat_store.py -k "settings_apply or generation_settings or context_policy or temporary"

Expected: failures for missing store orchestration/revision state.

- [ ] **Step 3: Add bounded session revision and failure state**

Extend ConsoleChatSession with:

    conversation_binding_revision: int = 0
    generation_settings_revision: int = 0
    context_policy_revision: int = 0
    generation_durable_snapshot: ConsoleGenerationSettingsSnapshot | None = None
    context_policy_durable_revision: int | None = None
    new_chat_default_generation: int = 0
    settings_persistence_failures: dict[ConsoleSettingsComponent, ConsoleSettingsPersistenceFailure] = field(default_factory=dict)
    applied_settings_submission_ids: deque[str] = field(default_factory=lambda: deque(maxlen=32))
    generation_metadata_status: ConsoleGenerationSettingsReadStatus = ConsoleGenerationSettingsReadStatus.ABSENT
    generation_metadata_warning_shown: bool = False

The failure record must contain component, its component revision, and the exact
safe snapshot/overrides needed for retry; it must never contain secrets or
base_url. Increment the generation revision for any live generation-settings
replacement and the policy revision for any live context-policy replacement,
including edits made outside these modals, so a newer component edit always
supersedes its old retry. Retry increments neither revision. Keep submission
deduplication bounded to the session lifetime.

Origin validation compares both conversation ID and conversation_binding_revision.
The ordinary first-persistence publication is the only None-to-ID operation that
preserves the binding revision. Restore, repurpose, explicit handoff rebind, or any
other operation that changes which durable conversation a live session represents
must advance the binding revision before publishing the ID. Add explicit store
helpers for first publication versus rebind; remove direct assignments from restore
paths. Test an origin captured at None, explicit rebind to an ID, and rejection even
though the origin's captured ID was None.

- [ ] **Step 4: Implement exact-origin apply and retry**

Add:

    def capture_console_settings_origin(
        self, session_id: str
    ) -> ConsoleSettingsOrigin

    def session_user_display_name_override(
        self, session_id: str
    ) -> str | None

    def commit_console_settings_live(
        self, submission: ConsoleSettingsSubmission
    ) -> ConsoleSettingsLiveCommit

    async def persist_console_settings_commit_serialized(
        self, commit: ConsoleSettingsLiveCommit
    ) -> ConsoleSettingsPersistenceOutcome

    async def retry_console_settings_persistence(
        self,
        *,
        session_id: str,
        component: ConsoleSettingsComponent,
        revision: int,
    ) -> bool

The live method must:

1. resolve submission.origin.session_id directly, never active_session_id;
2. validate the persisted identity transition;
3. deduplicate one submission ID or live-commit token;
4. replace settings with source="user" while preserving the origin's system prompt;
5. set context overrides through an in-memory-only store seam;
6. increment both live component revisions and return before any database/config I/O.

persist_console_settings_commit_serialized owns one asyncio.Lock per session.
Inside the lock it rechecks identity/component revisions, captures the current
generation durable snapshot and context durable revision, performs adapter I/O in
asyncio.to_thread, then returns to the
UI loop to publish new durable bases or failures. Every Apply and Retry uses this
same serializer. A stale commit exits before I/O. Generation writes use Task 2's
complete-snapshot CAS; context-policy writes use the CAS seam below. Thus an older worker,
a conflict retry, or a stale Retry cannot become durable after a newer Apply.

Extend ConsoleContextRepository with save_policy_if_revision, covering insert,
update, and empty-row delete with an expected policy_revision predicate. Return a
typed WRITTEN, CONFLICT, or MISSING result and the new durable revision. Route
ChatPersistenceService.update_conversation_context_policy through that seam for
settings Apply while leaving unrelated established callers compatible.

Change _resolve_context_policy_on_resume to assign both result.overrides and
result.revision to the session, including revision=None when no row exists. Every
successful established policy write publishes the returned durable revision;
successful deletion publishes None. This seed/update happens independently of the
process-local context_policy_revision used to supersede stale UI retries.

- [ ] **Step 5: Put generation metadata into first persistence**

In persist_session_if_needed, merge the staged generation snapshot into initial
metadata passed to create_conversation. Preserve the
existing speech/roleplay/prefill contributors and atomic library-policy path. Do
not create a second conversation row or a follow-up generation write when initial
metadata was accepted. Publish generation_durable_snapshot as the exact inserted
safe snapshot after creation.

Patch _promote_ephemeral_session_atomically separately: include the same generation
metadata object in conversation_kwargs["metadata"] before
promote_console_conversation_bundle. After atomic promotion, publish its durable
generation revision and flush the current context policy through its existing
owner. Change _flush_context_policy_on_first_persist to return/publish a typed
outcome so a create or promotion failure enters the current component ledger
instead of only logging. Successful first-persistence and promotion policy writes
publish their returned durable revision, including None after a successful empty
policy delete.

When publish_committed_identity changes None to the staged first ID, keep that transition valid for any modal that opened while unsaved. Do not allow any later A to B rebound.

- [ ] **Step 6: Run tests and commit**

Run:

    pytest -q Tests/Chat/test_console_settings_apply_store.py Tests/Chat/test_console_context_policy_cas.py
    pytest -q Tests/Chat/test_console_chat_store.py -k "settings_apply or generation_settings or context_policy or temporary or promote"

Expected: selected tests pass.

Commit:

    git add -p -- tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_context_repository.py tldw_chatbook/Chat/chat_persistence_service.py Tests/Chat/test_console_chat_store.py
    git add -- Tests/Chat/test_console_settings_apply_store.py Tests/Chat/test_console_context_policy_cas.py
    git diff --cached --check
    git diff --cached
    git commit -m "feat: apply Console settings to exact chat"

## Task 4: Add locked literal-path model-default mutation

**Files:**

- Modify: tldw_chatbook/config.py:5885-6165
- Create: tldw_chatbook/Chat/console_settings_defaults.py
- Modify: Tests/test_config_delete_settings.py
- Create: Tests/Chat/test_console_settings_defaults.py

- [ ] **Step 1: Write failing exact-path and default-intent tests**

Test:

- model IDs containing dots, slashes, colons, and brackets remain one literal mapping key;
- quick save sets/deletes only temperature and streaming;
- full save sets/deletes only the full field mask;
- blank/inherit deletes the exact override and preserves unexposed profile fields;
- sibling profiles and unrelated concurrent config edits survive;
- Make Default atomically patches the profile plus chat_defaults.provider/model;
- Save Model Default does not change chat_defaults.provider/model;
- only full Make Default with an explicitly dirty checked endpoint patches the provider endpoint;
- pre-replace failure returns Not written to disk and a retryable original intent;
- post-replace cache failure returns Saved on disk; running app refresh failed and a cache-refresh-only continuation;
- a newer explicit intent supersedes the old retry generation;
- an unconfigured/blocked provider cannot become the new-chat default even if a
  caller bypasses the disabled UI action.
- raw provider aliases are selected only from authoritative user TOML while
  readiness uses the locked effective mapping, including a concurrent edit that
  removes required configuration before lock acquisition.

- [ ] **Step 2: Run and observe failure**

Run:

    pytest -q Tests/test_config_delete_settings.py Tests/Chat/test_console_settings_defaults.py

Expected: failures for missing literal tuple-path mutation/default service.

- [ ] **Step 3: Extend the atomic writer with literal section paths**

Refactor apply_settings_mutation_to_cli_config around one private locked
implementation and expose a literal transaction builder:

    @dataclass(frozen=True, slots=True)
    class LiteralSettingsMutation:
        section_values: Mapping[tuple[str, ...], Mapping[str, object]]
        delete_keys: Mapping[tuple[str, ...], Collection[str]]

    @dataclass(frozen=True, slots=True)
    class AtomicLiteralMutationSnapshot:
        generation: int
        raw_values: Mapping[str, object]
        effective_values: Mapping[str, object]

    @dataclass(frozen=True, slots=True)
    class LiteralConfigMutationResult:
        file_replaced: bool
        caches_reloaded: bool
        settings_view: Mapping[str, object] | None
        failure_phase: str | None

    def apply_literal_settings_transaction_to_cli_config(
        mutation_builder: Callable[[AtomicLiteralMutationSnapshot], LiteralSettingsMutation],
        *,
        mutation_precondition: Callable[[], bool] | None = None,
    ) -> LiteralConfigMutationResult

Invoke mutation_builder once under the config lock with both the exact raw
authoritative TOML mapping and the effective merged/decrypted mapping, before
changing the in-memory config. It must be synchronous and
side-effect-free; contain exceptions as before_replace failures. Tuple elements are
literal TOML mapping keys and are never split on punctuation. Validate the returned
paths and set/delete overlap before writing. Reuse the existing lock, authoritative
reread, temp file, os.replace, permission preservation, and cache publication. Keep
current callers and dotted-section behavior unchanged.

- [ ] **Step 4: Implement the focused defaults service**

In console_settings_defaults.py define:

    class ConsoleDefaultSavePhase(str, Enum):
        BEFORE_REPLACE = "before_replace"
        CACHE_PUBLICATION = "cache_publication"

    @dataclass(frozen=True, slots=True)
    class ConsoleDefaultMutationIntent:
        generation: int
        action: ConsoleSettingsAction
        provider_config_key: str
        literal_model_id: str
        field_mask: frozenset[str]
        values: Mapping[str, object | None]
        endpoint_patch: ConsoleEndpointPatch | None

    @dataclass(frozen=True, slots=True)
    class ConsoleDefaultMutationOutcome:
        intent_generation: int
        file_replaced: bool
        runtime_published: bool
        settings_view: Mapping[str, object] | None
        failure_phase: ConsoleDefaultSavePhase | None

    def apply_console_default_intent(intent: ConsoleDefaultMutationIntent) -> ConsoleDefaultMutationOutcome

    @dataclass(frozen=True, slots=True)
    class RuntimeConfigPublicationResult:
        published: bool
        settings_view: Mapping[str, object] | None
        failure_phase: str | None

    class ConsoleDefaultRecoveryAction(str, Enum):
        RETRY_SAVE = "retry_save"
        DISCARD_RETRY = "discard_retry"
        REFRESH_RUNNING_APP = "refresh_running_app"
        DISMISS_REFRESH = "dismiss_refresh"

    @dataclass(frozen=True, slots=True)
    class ConsoleDefaultRecoveryRequest:
        action: ConsoleDefaultRecoveryAction
        intent_generation: int

    def refresh_console_runtime_after_saved_default() -> RuntimeConfigPublicationResult

The defaults service supplies the locked builder. Against raw_values it resolves
the canonical provider alias to the existing raw api_settings section name and
selects that section's already-configured endpoint key. Against effective_values it
reruns provider/model readiness while still locked, then builds one literal
mutation:

- profile path: ("api_settings", provider_config_key, "model_defaults", literal_model_id);
- global path for Make Default: ("chat_defaults",) with provider/model;
- endpoint path only for authorized full Make Default.

This locked resolution must not create a second aliased provider section after an
external edit. It also lets a Retry preserve unrelated edits made after the first
attempt while patching only the immutable intent's owned fields.

None means delete the exact profile field. Streaming Inherit maps to delete; On/Off map to strict booleans. After file replacement, never retry the disk mutation for a cache-publication failure.

Build values from ConsoleSettingsFieldDraft.profile_override, not from the
conversation's frozen effective values. For the quick mask, first materialize
temperature and streaming profile_override from their displayed effective values
because quick has no Inherit state. Intersect the requested mask with fields
actually exposed and supported for the selected provider/model. Require a
ConsoleEndpointDraft whose provider binding matches the authoritative target and
whose dirty and checked flags are both true before including an endpoint. Reject
Make Default when the selected provider/model is not send-ready under the locked
effective snapshot. This guard must observe a concurrent edit that removes the
provider's required configuration before the writer acquires the lock.

- [ ] **Step 5: Add endpoint preview safety**

Implement a pure parser that returns only sanitized host/port authority plus:

- Local for localhost and loopback literals;
- LAN for private/link-local literals and .local;
- Remote for public IP literals;
- Remote/unknown for other hostnames.

Reject userinfo. Preserve a syntactically valid explicit port in copy such as
192.168.1.20:8080, but never return scheme, path, query, fragment, credentials, or
resolved IP. Classify using the host only and never perform DNS.

On immediate success, apply_console_default_intent returns the freshly published
settings mapping in ConsoleDefaultMutationOutcome.settings_view. A before-replace
or cache-publication failure returns settings_view=None. Only the latter recovery
path calls refresh_console_runtime_after_saved_default, which performs its locked reread, cache
publication, and load_settings rebuild off the UI thread and returns the complete
fresh settings mapping. The caller only assigns settings_view to
app_instance.app_config on the UI thread. It never writes the config file.

- [ ] **Step 6: Run tests and commit**

Run:

    pytest -q Tests/test_config_delete_settings.py Tests/Chat/test_console_settings_defaults.py

Expected: all selected tests pass, including multiprocessing/concurrent sibling preservation.

Commit:

    git add -p -- tldw_chatbook/config.py Tests/test_config_delete_settings.py
    git add -- tldw_chatbook/Chat/console_settings_defaults.py Tests/Chat/test_console_settings_defaults.py
    git diff --cached --check
    git diff --cached
    git commit -m "feat: patch Console model defaults atomically"

## Task 5: Make every eligible blank chat use published global defaults

**Files:**

- Modify: tldw_chatbook/UI/Console_Modules/session.py:2152-2185,2291-2455
- Modify: tldw_chatbook/UI/Console_Modules/workspace.py:3600-3645
- Modify: tldw_chatbook/Chat/console_chat_controller.py:6680-6735
- Modify: tldw_chatbook/Chat/console_launch_wake.py:215-235,330-355
- Modify: Tests/UI/test_console_session_controller.py
- Modify: Tests/UI/test_console_session_settings.py:2279-2533
- Modify: Tests/UI/test_console_launch_wake.py
- Modify: Tests/Chat/test_console_conversation_hydration.py

- [ ] **Step 1: Write failing new-chat and resume tests**

Cover Ctrl+T, temporary creation, workspace-created blank chat, and initial pristine Console. After runtime publication each must derive chat_defaults provider/model plus the exact model profile.

Also prove existing/open chats are unchanged and explicit Duplicate, Branch, Continue, and handoff creation paths retain their source-provided settings. A persisted resume must derive from the saved conversation provider/model, not the currently active chat.

Add regression coverage proving an already-open blocked pristine chat is not
silently refreshed to the new default after Make Default, while the existing
task-177 setup/configuration recovery still refreshes eligible unused blocked chats
for non-default configuration changes. Cover launch-wake hydration and a
missing-catalog custom model/unconfigured saved provider remaining explicit.

- [ ] **Step 2: Run and observe failure**

Run:

    pytest -q Tests/UI/test_console_session_controller.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_launch_wake.py Tests/Chat/test_console_conversation_hydration.py -k "new_chat or temporary or workspace or pristine or resume or duplicate or branch or handoff or default or wake"

Expected: current active-settings cloning fails the eligible-new-chat assertions.

- [ ] **Step 3: Add a config-only blank-chat builder**

Add blank_console_session_settings(app_config) in console_session_settings.py. It
must call default_console_session_settings(app_config) with no provider/model
overrides. UI wrappers may read current app_config but must not consult
_effective_console_provider_model, active controller controls, or another session.

- [ ] **Step 4: Change every eligible blank-new-chat path**

In _create_native_console_session_from_active_context:

    defaults = self._blank_console_session_settings()
    controller.new_session(
        settings=defaults,
        canonical_settings_baseline=defaults,
        ephemeral=ephemeral,
    )

In UI/Console_Modules/workspace.py remove inherited_settings from workspace-created
blank sessions and use the same config-only builder plus canonical baseline. Route
initial pristine ensure through it. Do not call _active_console_session_settings
for blank creation. Keep explicit source-settings Duplicate/Branch/Continue/handoff
paths unchanged.

Update _console_session_settings_for_resume to return Task 2's
ConsoleGenerationSettingsHydration from current app_config, not an active-session
base. The resume call passes hydration.settings,
hydration.durable_snapshot, and hydration.metadata_status into
restore_persisted_session. Preserve system prompt and current endpoint resolution.

Update console_launch_wake.py to the same hydration signature and provider-first
resume behavior, including the durable owned revision/status arguments.

- [ ] **Step 5: Fence already-open sessions from explicit default publication**

TldwCli owns a monotonic console_new_chat_default_generation. Every new blank
session captures it in ConsoleChatSession.new_chat_default_generation. Successful
Make Default increments the app generation, but never mutates existing sessions.
_maybe_refresh_stale_default_console_settings must return the existing settings
when the session predates the current explicit-default generation. Non-default
setup/configuration refreshes do not increment this generation, preserving the
existing task-177 recovery behavior.

- [ ] **Step 6: Run tests and commit**

Run:

    pytest -q Tests/UI/test_console_session_controller.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_launch_wake.py Tests/Chat/test_console_conversation_hydration.py -k "new_chat or temporary or workspace or pristine or resume or duplicate or branch or handoff or default or wake"

Expected: selected tests pass.

Commit:

    git add -p -- tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/workspace.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_launch_wake.py Tests/UI/test_console_session_controller.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_launch_wake.py Tests/Chat/test_console_conversation_hydration.py
    git diff --cached --check
    git diff --cached
    git commit -m "fix: seed blank Console chats from saved defaults"

## Task 6: Rebuild the quick Provider popover around explicit actions

**Files:**

- Modify: tldw_chatbook/Widgets/Console/console_model_popover.py:1-455
- Modify: Tests/UI/test_console_context_controls.py
- Modify: Tests/UI/test_console_model_apply_chips.py
- Modify: Tests/UI/test_console_modal_dismissal.py
- Modify: Tests/UI/test_console_resize_reflow.py

- [ ] **Step 1: Write failing mounted interaction tests**

Using ConsolidatedCSSApp and production CSS, test:

- main footer order and labels: Cancel, Full settings…, Defaults…, Apply to this chat;
- Defaults replaces the footer with Save as model default, Make default for new chats, Back;
- compaction-stays-with-this-chat copy is visible in Defaults;
- scope copy shows Applies to this conversation plus the exact unsaved-first-message
  or temporary-until-promoted durability line;
- blocked/unconfigured provider disables Make default for new chats in the quick
  Defaults chooser with an explanation;
- valid mouse and keyboard activation produce the same ConsoleSettingsCommittedSubmission and dismiss once after the live commit;
- invalid/NaN/infinite/out-of-range temperature stays open and displays an error;
- Cancel/Escape returns None;
- hierarchical Escape leaves Defaults first, then closes the popover;
- provider and model changes rebase target defaults, mark deliberate edits, drop stale endpoint, and restore A to B to A drafts;
- Full settings returns ConsoleSettingsTransfer without applying;
- 60x24 and 72x24 keep actions reachable, ordered, and not overlapping;
- mouse capture is released before dismissal and a deferred callback after teardown is harmless.

- [ ] **Step 2: Run and observe failures**

Run:

    pytest -q Tests/UI/test_console_context_controls.py Tests/UI/test_console_model_apply_chips.py Tests/UI/test_console_modal_dismissal.py Tests/UI/test_console_resize_reflow.py -k "popover or model_apply or defaults"

Expected: failures for missing buttons, explicit errors, and submission types.

- [ ] **Step 3: Implement popover state and result semantics**

Change the modal result type to ConsoleSettingsCommittedSubmission | ConsoleSettingsTransfer | None. Constructor inputs must include origin, app_config, the initial ConsoleSettingsDraftState, the chat scope/durability copy, a DraftRebaser callback backed by ConsoleChatController, and a synchronous live_committer callback that revalidates through the controller before ConsoleChatStore.commit_console_settings_live.

Replace the silent invalid-temperature fallback with the same blocking validation rule as full Settings and add #console-popover-error.

All four submit buttons must call one _submit(action) method. _submit validates,
builds the exact action/mask, releases mouse capture if needed, and calls
live_committer before dismissing. A rejected missing/closed/rebound origin reports
Chat closed; nothing applied, dismisses without a committed result, and cannot
start default work; ordinary field validation still keeps the surface open. A
successful commit is wrapped with its submission as
ConsoleSettingsCommittedSubmission, then the modal marks itself dismissed once and
dismisses. Defaults/Back only changes local view state.

Provider/model handlers must save the current keyed draft before calling the
injected controller rebaser. Mark temperature dirty on input changes and streaming
dirty only on explicit toggle. The widget never builds target defaults or decides
field support itself.

Render carried deliberate edits visibly beside the affected controls with the
literal marker Edited — carried from {provider}/{model}; inherited controls use an
Inherited marker. The marker is derived from field provenance, not guessed from
value equality.

- [ ] **Step 4: Implement bounded layout and Escape**

Keep one scrollable body and pinned action footer. At narrow width allow action rows to wrap or stack without clipping. Defaults is a local substate, so Escape returns to main state before SafeModalDismissMixin closes the screen.

- [ ] **Step 5: Run tests and commit**

Run:

    pytest -q Tests/UI/test_console_context_controls.py Tests/UI/test_console_model_apply_chips.py Tests/UI/test_console_modal_dismissal.py Tests/UI/test_console_resize_reflow.py -k "popover or model_apply or defaults"

Expected: selected tests pass at both terminal sizes.

Commit:

    git add -p -- tldw_chatbook/Widgets/Console/console_model_popover.py Tests/UI/test_console_context_controls.py Tests/UI/test_console_model_apply_chips.py Tests/UI/test_console_modal_dismissal.py Tests/UI/test_console_resize_reflow.py
    git diff --cached --check
    git diff --cached
    git commit -m "feat: add explicit Console provider actions"

## Task 7: Give full Console Settings the same contract and safe endpoint opt-in

**Files:**

- Modify: tldw_chatbook/Widgets/Console/console_settings_modal.py:75-215,279-520,1435-1810,1850-2350
- Modify: Tests/UI/test_console_session_settings.py:2877-5112
- Modify: Tests/UI/test_console_context_controls.py

- [ ] **Step 1: Write failing full-modal tests**

Test:

- Apply to this chat returns the same submission type and full context-policy overrides;
- Save as model default uses FULL_MODEL_DEFAULT_FIELDS;
- Make default for new chats uses the full mask and can opt into a dirty endpoint;
- Save Model Default can never persist endpoint;
- endpoint checkbox is disabled until endpoint is explicitly dirty;
- endpoint draft remains bound to its raw provider identity through A to B to A and
  quick-to-full transfer; a mismatched binding cannot be checked or saved;
- preview contains only sanitized host/classification;
- streaming cycles Inherit to On to Off and Inherit deletes the profile override;
- quick-to-full transfer restores all quick draft values, compaction, dirty provenance, origin, and keyed drafts;
- provider/model switches call the same injected controller-owned rebase seam;
- no modal handler writes config directly;
- Cancel/Escape does not apply.
- blocked/unconfigured provider readiness disables Make default for new chats with
  an explanatory status while leaving Apply to this chat available under its
  existing readiness rules.
- an app-global default failure renders the same exact intent summary/actions in
  full Console Settings as the Model rail.
- Retry/Discard/Refresh/Dismiss in the full modal emit a typed request with the
  exact intent generation, update the app-owned state, and refresh the modal region
  without relying on ChatScreen button bubbling.

- [ ] **Step 2: Run and observe failures**

Run:

    pytest -q Tests/UI/test_console_session_settings.py Tests/UI/test_console_context_controls.py -k "save_default or transfer or endpoint or streaming or provider_change or model_change or recovery"

Expected: current _save_as_default direct config writer and boolean streaming behavior fail.

- [ ] **Step 3: Replace direct persistence with discriminated results**

Remove the save_settings_to_cli_config import, _default_persist_sections, and
direct async config write from the modal. Apply and both default buttons validate,
invoke the same injected live_committer as the quick popover, and dismiss only
after it returns a successful ConsoleSettingsLiveCommit. ChatScreen performs
durable conversation persistence and any default mutation from the committed
result after close.

Accept an optional ConsoleSettingsTransfer and the app-owned
ConsoleDefaultDurabilityState in the constructor. Seed all controls, context
overrides, field provenance, endpoint provider binding/dirty/checked state, and
keyed provider/model drafts from the transfer. Render the same before-replace or
runtime-refresh exact intent summary and recovery actions exposed by the rail.
Also inject:

    DefaultRecoveryHandler = Callable[
        [ConsoleDefaultRecoveryRequest],
        Awaitable[ConsoleDefaultDurabilityState],
    ]

Each full-modal recovery button awaits this handler, replaces its local recovery
state with the returned app-owned snapshot, reenables valid controls, and remains
open. ChatScreen supplies the handler and performs Retry/Discard/Refresh/Dismiss;
recovery requests are not part of the normal Apply/dismiss result union.

Use one _submission_for_action method for Apply, Save Model Default, and Make Default.

- [ ] **Step 4: Add endpoint opt-in and inherited streaming**

Track ConsoleEndpointDraft separately from provider-derived endpoint
initialization. Show a Checkbox only for full Make Default semantics; it remains
unchecked by default and disables whenever its bound provider does not match the
selected provider. Its label/adjacent Static must render:

    Also save connection: host.example:8443 · Remote/unknown

using only Task 4's pure sanitizer. No URL details may enter copy or logs.

Represent streaming draft as None | True | False, render Inherit/On/Off, and build the effective conversation value before live Apply while retaining None provenance for profile deletion.

- [ ] **Step 5: Run tests and commit**

Run:

    pytest -q Tests/UI/test_console_session_settings.py Tests/UI/test_console_context_controls.py -k "save_default or transfer or endpoint or streaming or provider_change or model_change or exact_origin or recovery"

Expected: selected tests pass.

Commit:

    git add -p -- tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_context_controls.py
    git diff --cached --check
    git diff --cached
    git commit -m "feat: unify full Console settings actions"

## Task 8: Coordinate exact-origin Apply, default saves, and visible recovery

**Files:**

- Modify: tldw_chatbook/app.py:6935-6985
- Modify: tldw_chatbook/UI/Screens/chat_screen.py:2535-2725,3535-3605,4920-4985,19650-19730
- Modify: tldw_chatbook/UI/Console_Modules/left_rail.py:1645-1715
- Modify: Tests/UI/test_console_session_settings.py:5210-5415
- Modify: Tests/UI/test_console_model_apply_chips.py
- Modify: Tests/UI/test_console_rail_sections.py
- Modify: Tests/UI/test_console_parallel_runs.py

- [ ] **Step 1: Write failing coordinator and recovery tests**

Test:

- origin is captured before awaiting model-catalog resolution;
- switching tabs during that await or while the modal is open still applies to origin;
- a closed/rebound origin is rejected and the modal cannot claim success;
- both quick and full results call one coordinator;
- live Apply closes before default disk work finishes;
- a slow/failing conversation metadata or context-policy write neither delays nor
  prevents the requested model/global default mutation after the shared live commit;
- an execution context captured before the live commit retains its old settings,
  while one resolved afterward sees the new settings;
- each session failure row names generation or compaction/context policy and offers Retry;
- conversation Retry only affects the matching current session/component revision;
- default pre-replace failure is visible across Console tabs with Retry default save/Discard retry;
- post-replace failure is visible as Saved on disk; running app refresh failed with Refresh running app/Dismiss;
- cache-only refresh never rewrites disk;
- newer default intent supersedes older retry state;
- initial success, successful Retry, and successful Refresh each publish one fresh
  settings mapping and increment the new-chat-default generation at most once for
  the matching Make Default intent;
- ordinary Apply leaves app-global default failure untouched;
- modal teardown and duplicate callbacks cannot double-commit.
- successful default actions emit two independent receipts: This chat updated plus
  the exact model-profile or eligible-new-chat scope saved.

- [ ] **Step 2: Run and observe failures**

Run:

    pytest -q Tests/UI/test_console_session_settings.py Tests/UI/test_console_model_apply_chips.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_parallel_runs.py -k "exact_origin or persistence_failure or default_failure or model_apply"

Expected: failures for active-session callback routing and missing recovery actions.

- [ ] **Step 3: Capture origin before every await**

In action_open_console_model_popover and _open_console_settings:

    store = self._ensure_console_chat_store()
    session_id = store.active_session_id
    if session_id is None:
        return
    origin = store.capture_console_settings_origin(session_id)
    settings = store.session_settings(session_id)
    context_policy = store.session_context_policy_overrides(session_id)
    user_display_name = store.session_user_display_name_override(session_id)

Capture settings, policy, and system prompt from that same session before model-catalog awaits. Bind callback with the origin/transfer result; never look up active_session_id when applying.

- [ ] **Step 4: Put default durability state on the application lifetime**

Initialize a ConsoleDefaultDurabilityState holder beside app_config in TldwCli.
The holder contains only the newest immutable default intent generation and either
the before-replace retry record or the after-replace runtime-refresh record.
ChatScreen reads this app-owned holder, so navigation/unmount and Console tab
switches cannot make a global failure disappear.

Initialize console_new_chat_default_generation at zero in the same app lifetime.
Only a fully published Make Default increments it.

ConsoleDefaultDurabilityState also tracks the one current
runtime_published_intent_generation. Its accept_runtime_publication method is
idempotent and succeeds only for the newest intent generation. After the returned
settings mapping is assigned to app_instance.app_config, call this method; increment
console_new_chat_default_generation exactly once when that intent's action is Make
Default. The same path is used for initial success, a successful Retry after
before-replace failure, and Refresh running app after cache-publication failure.

Reserve a new default-intent generation synchronously before dispatching its
worker. The config mutation_precondition rechecks that generation under the config
lock, and the UI publishes an outcome only if the same generation is still newest.
A later explicit default action therefore supersedes both an older pending Retry
and an older in-flight result without letting either replace the new holder state.

- [ ] **Step 5: Add one coordinator**

Add an async _coordinate_console_settings_submission method:

1. inject a live_committer that asks ConsoleChatController to revalidate/rebase the
   submitted draft against the current app_config, then calls
   store.commit_console_settings_live; accept only its
   ConsoleSettingsCommittedSubmission;
2. refresh the origin's visible surfaces if mounted, without switching tabs;
3. immediately start the conversation durability task through
   store.persist_console_settings_commit_serialized and, for a default action, an
   independent asyncio.to_thread apply_console_default_intent task; await and
   publish their outcomes independently so neither gates the other;
4. for a full-modal submission, pass the captured display-name override to its
   existing exact-session roleplay owner and retain that owner's current warning
   behavior; do not add it to the new generation/context failure ledger;
5. on full default success, assign the default mutation outcome's fresh
   settings_view to
   app_instance.app_config on the UI thread, then call the idempotent
   accept_runtime_publication path above; do not resync or mutate any already-open
   session;
6. record each conversation/default failure independently and only for its current
   component revision or default intent generation.

Full settings transfer reopens ConsoleSettingsModal with the same origin/draft and no live call.

- [ ] **Step 6: Make runtime refresh cache-only and app-visible**

When file replacement succeeded but either config cache publication or
app_instance.app_config publication failed, retain only the already-saved intent
identity and a runtime-refresh action. Refresh running app must call a public
config helper that locks, rereads, and republishes the existing file without
writing it, then assign the returned fresh settings view to app_instance.app_config.
It must never call apply_console_default_intent or any disk mutation.

Run that helper in asyncio.to_thread. The helper returns
RuntimeConfigPublicationResult; only the final app_instance.app_config assignment
runs on the UI thread, followed by the same idempotent
accept_runtime_publication path used by initial success and Retry.

- [ ] **Step 7: Replace the rail recovery Static with explicit state/actions**

In left_rail.py, keep provider readiness and persistence recovery as separate rows inside the Model section. Add compact buttons with IDs for:

- retry conversation generation;
- retry compaction/context policy;
- retry default save;
- discard default retry;
- refresh running app;
- dismiss runtime-refresh failure.

Only mount/show actions valid for current state. The default row includes the exact
action, provider, literal model, field scope, and sanitized optional authority.
Button events route to the store/default service with exact revision/generation
tokens. ChatScreen exposes the same routing as a typed DefaultRecoveryHandler
injected into full Console Settings; both surfaces return the updated
ConsoleDefaultDurabilityState from the one app owner.

- [ ] **Step 8: Rebuild consolidated CSS if bundled declarations change**

If left_rail or modal CSS uses BUNDLED_CSS/BUNDLED_SCREEN_CSS, run:

    python tldw_chatbook/css/build_css.py

Then run:

    pytest -q Tests/UI/test_widget_css_consolidation.py Tests/UI/test_css_class_coverage_contract.py
    git diff --name-only -- tldw_chatbook/css/widget_defaults_self.tcss tldw_chatbook/css/widget_defaults_scoped.tcss tldw_chatbook/css/screen_css_self.tcss tldw_chatbook/css/screen_css_scoped.tcss tldw_chatbook/css/tldw_cli_modular.tcss

Expected: generated CSS is current and class coverage passes. Record only the
tracked generated paths printed by git diff; .css-build-manifest.json is ignored
builder state and must not be staged.

- [ ] **Step 9: Run coordinator tests and commit**

Run:

    pytest -q Tests/UI/test_console_session_settings.py Tests/UI/test_console_model_apply_chips.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_parallel_runs.py -k "exact_origin or persistence_failure or default_failure or model_apply"

Expected: selected tests pass.

Commit:

    git add -p -- tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/left_rail.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_model_apply_chips.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_parallel_runs.py
    # Add -p only the tracked generated paths printed by the preceding diff:
    git add -p -- tldw_chatbook/css/widget_defaults_self.tcss tldw_chatbook/css/widget_defaults_scoped.tcss tldw_chatbook/css/screen_css_self.tcss tldw_chatbook/css/screen_css_scoped.tcss tldw_chatbook/css/tldw_cli_modular.tcss
    git diff --cached --check
    git diff --cached
    git commit -m "feat: expose Console settings recovery actions"

## Task 9: Add end-to-end Console behavior coverage

**Files:**

- Create: Tests/UI/test_console_provider_apply_defaults_flow.py
- Modify: Tests/UI/test_console_resize_reflow.py
- Modify: Tests/ProductionApp/test_provider_selection_ownership.py

- [ ] **Step 1: Add production-hierarchy interaction tests**

Use real Console screen composition, ConsolidatedCSSApp, temporary config, and in-memory SQLite. Exercise:

- mouse Apply to this chat, close, next send context uses selected provider/model;
- keyboard activation follows the same method;
- a context captured before Apply keeps the old settings while a later capture uses
  the new settings;
- persisted close/resume restores generation settings and compaction;
- unsaved ordinary chat stages then persists both owners;
- temporary chat remains non-durable, then promotion persists;
- Save Model Default survives a simulated restart without changing global provider/model;
- Make Default changes every eligible new-chat entry point immediately and after simulated restart;
- existing/open and explicit source-derived chats remain unchanged;
- pre-replace and post-replace failures show exact copy/actions;
- quick compaction changes preserve every other policy override, and a stale policy
  Retry cannot restore values superseded by a newer full-context edit;
- blocked quick defaults, missing-catalog custom models, and unconfigured saved
  providers remain explicit rather than silently substituted;
- successful default actions show separate chat-updated and exact-default-scope
  receipts;
- 60x24 and 72x24 contain no overlap/clipping and every action is keyboard reachable.

Stub provider execution; do not contact a real LLM or write the real config.

- [ ] **Step 2: Run the end-to-end target**

Run:

    pytest -q Tests/UI/test_console_provider_apply_defaults_flow.py Tests/UI/test_console_resize_reflow.py Tests/ProductionApp/test_provider_selection_ownership.py

Expected: all selected tests pass.

- [ ] **Step 3: Run the complete targeted feature matrix**

Run:

    pytest -q \
      Tests/Chat/test_console_settings_apply.py \
      Tests/Chat/test_console_generation_settings_metadata.py \
      Tests/Chat/test_console_settings_apply_store.py \
      Tests/Chat/test_console_context_policy_cas.py \
      Tests/Chat/test_console_settings_defaults.py \
      Tests/Chat/test_console_conversation_hydration.py \
      Tests/Chat/test_console_session_settings.py \
      Tests/Chat/test_console_chat_store.py \
      Tests/UI/test_console_provider_apply_defaults_flow.py \
      Tests/UI/test_console_context_controls.py \
      Tests/UI/test_console_model_apply_chips.py \
      Tests/UI/test_console_modal_dismissal.py \
      Tests/UI/test_console_session_controller.py \
      Tests/UI/test_console_launch_wake.py \
      Tests/UI/test_console_session_settings.py \
      Tests/UI/test_console_rail_sections.py \
      Tests/UI/test_console_resize_reflow.py \
      Tests/UI/test_console_parallel_runs.py \
      Tests/ProductionApp/test_provider_selection_ownership.py \
      Tests/test_config_delete_settings.py

Expected: all targeted feature tests pass. If runtime is excessive, split by Chat/UI/config but do not silently omit a listed file.

- [ ] **Step 4: Run static and diff checks**

Run:

    python -m compileall -q tldw_chatbook/Chat tldw_chatbook/Widgets/Console tldw_chatbook/UI/Console_Modules tldw_chatbook/UI/Screens/chat_screen.py
    ruff check tldw_chatbook/Chat/console_settings_apply.py tldw_chatbook/Chat/console_generation_settings_metadata.py tldw_chatbook/Chat/console_settings_defaults.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_conversation_hydration.py tldw_chatbook/Widgets/Console/console_model_popover.py tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/left_rail.py tldw_chatbook/UI/Screens/chat_screen.py
    git diff --check

Expected: zero errors.

- [ ] **Step 5: Self-review against the acceptance criteria**

Review the diff for:

- any use of active_session_id after modal open;
- any conversation metadata serialization of endpoint/secrets;
- any compaction value in model/global defaults;
- any whole model_defaults replacement;
- any cache-failure retry that rewrites disk;
- any UI success copy before live commit;
- any bare real-config or real-network verification.

Commit:

    git add -p -- Tests/UI/test_console_resize_reflow.py Tests/ProductionApp/test_provider_selection_ownership.py
    git add -- Tests/UI/test_console_provider_apply_defaults_flow.py
    git diff --cached --check
    git diff --cached
    git commit -m "test: cover Console provider apply and defaults"

## Task 10: Finish backlog and implementation documentation

**Files:**

- Modify: backlog/tasks/task-22515 - Make-Console-provider-Apply-update-and-persist-conversation-settings.md
- Modify: Docs/superpowers/specs/2026-08-27-console-provider-apply-persistence-design.md if implementation deviations were approved
- Modify: backlog/docs/lessons-testing-evidence.md or backlog/docs/lessons-live-verification.md only if this work produced a concrete reusable incident

- [ ] **Step 1: Record implementation notes**

Add a concise Implementation Notes section to TASK-22515 covering:

- the shared submission/exact-origin owner;
- safe metadata and context-policy ownership;
- literal-path default mutation and failure phases;
- eligible new-chat behavior;
- modified/added files;
- targeted verification evidence;
- ADR-095.

- [ ] **Step 2: Check every acceptance criterion only after evidence exists**

Change each task checkbox from [ ] to [x] only when the targeted test or inspected behavior proves it.

- [ ] **Step 3: Complete task hygiene**

After every criterion, targeted test, static check, documentation update, and self-review is complete:

    backlog task edit 22515 -s Done

Do not mark Done early. Do not invent a lessons-learned entry.

- [ ] **Step 4: Commit final documentation**

Run:

    git diff --check
    git status --short

Commit:

    git add -p -- "backlog/tasks/task-22515 - Make-Console-provider-Apply-update-and-persist-conversation-settings.md" Docs/superpowers/specs/2026-08-27-console-provider-apply-persistence-design.md
    # If and only if one exact lessons file was deliberately changed:
    git add -p -- backlog/docs/lessons-testing-evidence.md
    git diff --cached --check
    git diff --cached
    git commit -m "docs: complete Console provider apply task"

## Optional Full Sweep

The repository requires explicit user approval before a full test sweep. After all targeted verification passes, ask:

    The targeted Console/provider/default test matrix passes. Do you want the full pytest suite before integration?

Run pytest only if the user opts in.
