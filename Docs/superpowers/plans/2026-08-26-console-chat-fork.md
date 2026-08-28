# Console Chat Fork from a Message Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user fork an independently owned Console chat through one selected stable message, choose the fork name in a compact confirmation dialog, open the result immediately, and leave the source chat untouched.

**Architecture:** Add a pure, allowlisted fork projection and fence contract beside the Console chat domain model. The store issues and revalidates the exact active-lineage and generated-image selection fence; a dedicated persistence bundle commits durable ancestry, messages, supported sidecars, governed citation links, policy, and sanitized project context in one SQLite transaction; only then does the session controller publish and activate the new live session. Temporary forks use the same projection but remain detached in memory and later promote as independent roots. UI orchestration stays in the decomposed Console controllers, with a presentation-only modal and captured-target overflow menu.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5, pytest, pytest-asyncio, Textual Pilot, stdlib `dataclasses`/`hashlib`/`json`/`uuid`.

**Approved design:** `Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md`

**Backlog task:** `TASK-23088`

**ADR required:** yes; no new ADR.

**ADR path:** `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md`

**Reason:** ADR-092 already governs the storage, identity, security-authority, and cross-module copy contract. This plan implements that approved decision without changing its architecture or the database schema.

---

## Invariants to preserve in every task

- The fork boundary is the source session's canonical active lineage prefix, inclusive of the selected message. Never derive it from rendered transcript rows or the whole hidden tree.
- Issuing, validating, committing, registering, and activating a fork never mutates source title, settings, active leaf, selected text/image variants, durable rows, scratch, approvals, runs, or recovery state.
- USER/ASSISTANT content, sent attachments, selected visible variants, declarative settings, Library/RAG policy, and sanitized project-instruction controls are allowlisted. Unknown or unsupported owners fail before commit.
- Tool/activity rows, drafts, pending attachments, usage, context summaries, pinned/one-shot prefill, continuations, recovery, approvals, scratch, resolved project-instruction bodies, and live video authority are excluded.
- Durable data is committed before a live session becomes visible. A precommit error publishes nothing; a postcommit activation error reports and reopens the same preallocated fork instead of creating another.
- Temporary sources produce temporary forks. A non-ephemeral source that has not acquired durable IDs produces an immediately saved independent-root fork without persisting the source.
- No new dependency and no schema migration are permitted for this feature.

## File responsibility map

### Domain and store

- Create `tldw_chatbook/Chat/console_chat_fork.py` — frozen eligibility, fence, snapshot, projected-message, governed-citation-link, and commit-result records; title normalization and deterministic fingerprints; explicit field allowlist.
- Reuse `derive_console_session_title(..., max_length=60)` from `tldw_chatbook/Chat/console_chat_models.py`; no change or fork state is needed in the mutable message model.
- Modify `tldw_chatbook/Chat/console_message_actions.py` — Fork availability/dispatch plus stable primary, overflow, and media action groups.
- Modify `tldw_chatbook/Chat/console_chat_store.py` — store-derived eligibility, active-prefix fence issue/revalidation, pure snapshot staging, detached registration, and sanitized temporary promotion.
- Modify `tldw_chatbook/Chat/console_conversation_hydration.py` — carry persona-memory identity through the canonical durable-resume path.
- Modify `tldw_chatbook/UI/Console_Modules/session.py` — round-trip the currently missing persona-memory field through screen-session serialization before fork orchestration lands.
- Modify `tldw_chatbook/Chat/console_project_instructions.py` — one pure helper that clears the notice key while retaining the three declarative controls.

### Persistence and governed sidecars

- Modify `tldw_chatbook/Chat/chat_persistence_service.py` — expose existing lineage fields and add the dedicated idempotent atomic fork bundle; move project-context JSON into atomic promotion.
- Modify `tldw_chatbook/Chat/citation_trace_repository.py` — link a fresh fork-message owner to an existing active immutable trace only after source and target revision/body/fingerprint validation.
- No migration and no generic deep-copy helper.

### Console UI

- Create `tldw_chatbook/Widgets/Console/console_fork_chat_modal.py` — naming/confirmation and six-state presentation only.
- Create `tldw_chatbook/Widgets/Console/console_message_more_menu.py` — captured message/action IDs and bounded menu focus/teardown.
- Modify `tldw_chatbook/Widgets/Console/__init__.py` — export the two widgets.
- Modify `tldw_chatbook/Widgets/Console/console_transcript.py` — direct action order, `f`, menu mounting/teardown/focus fallback, stable dispatch, and ineligible help.
- Modify `tldw_chatbook/Widgets/Console/console_generation_card.py` and `console_video_card.py` — keep media actions with their media cards.
- Modify `tldw_chatbook/UI/Console_Modules/image.py` — generation selection revision/fingerprint capture and validation.
- Modify `tldw_chatbook/UI/Console_Modules/message.py` — route Fork/More by captured message ID and delegate the operation to the session controller.
- Modify `tldw_chatbook/UI/Console_Modules/session.py` — modal lifecycle, cancellable validation generation, commit/register, activation, and created-not-opened recovery.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py` — named session/message/image callbacks only.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py` — delegation and synchronized F1/Inspector help copy only; no fork business logic.
- Modify `tldw_chatbook/Widgets/Console/console_edit_message_modal.py` — clarify that Edit & resend creates a response branch in the same chat.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` only if the new production widgets need shared styling.

### Tests and docs

- Create `Tests/Chat/test_console_chat_fork.py` — pure projection, allowlist, fence, eligibility, IDs, and source immutability.
- Create `Tests/Chat/test_console_chat_fork_persistence.py` — real-SQLite atomicity, ancestry, idempotency, reload, project context, citations, and degradation.
- Modify `Tests/Chat/test_console_message_actions.py`.
- Modify `Tests/Chat/test_console_image_controller.py`.
- Modify `Tests/Chat/test_console_chat_store_project_instructions.py`.
- Modify `Tests/Chat/test_citation_trace_repository.py`.
- Create `Tests/UI/test_console_fork_chat_modal.py`.
- Modify `Tests/UI/test_console_native_transcript.py`, `test_console_message_controller.py`, and `test_console_session_controller.py`.
- Create `Tests/integration/test_console_chat_fork_flow.py` — production-shaped controller/store/persistence journey without a provider.
- Modify `Docs/User_Guide/console/chat-basics.md` and `Docs/User_Guide/console/branching-and-rewind.md`.

## Task 1: Pin the pure fork contract and action eligibility

**Files:**

- Create: `Tests/Chat/test_console_chat_fork.py`
- Modify: `Tests/Chat/test_console_message_actions.py`
- Create: `tldw_chatbook/Chat/console_chat_fork.py`
- Modify: `tldw_chatbook/Chat/console_message_actions.py`
- Modify: `tldw_chatbook/Chat/console_project_instructions.py`

- [ ] **Step 1: Write failing domain tests for bounded titles and sanitized authority**

Cover these exact outcomes:

```python
assert default_fork_title("Research notes") == "Forked from Research notes"
assert default_fork_title("") == "Untitled chat — fork"
assert len(normalize_fork_title("x" * 100)) == 60
assert sanitize_fork_project_instruction_state(source) == replace(
    source, project_instruction_notice_key=None
)
```

Reject blank normalized titles and prove the helper uses
`derive_console_session_title(..., max_length=CONSOLE_FORK_TITLE_MAX_LENGTH)`
rather than introducing a second truncation algorithm.

- [ ] **Step 2: Write failing action-service tests**

Assert:

- eligible complete USER and ASSISTANT rows contain `fork` immediately before `regenerate`;
- non-empty stopped/failed ASSISTANT rows contain it, while pending, streaming, discarded, and empty failed rows expose a disabled reason;
- durable eligibility can be refused by a store-derived reason when any active-prefix node lacks a persisted ID;
- primary actions are Copy, Speak/Stop when applicable, Edit, Fork, Regenerate/Retry, Continue, More; overflow is Save as, Helpful, Not helpful, Delete; media actions are returned separately;
- TOOL/activity rows retain their specialized controls and never expose Fork or More;
- `dispatch("fork", message)` returns `status="fork_requested"` and the exact target message ID.

Pass a frozen `ConsoleForkEligibility` into action resolution. Do not let the action service infer durable lineage from presentation fields.

- [ ] **Step 3: Run the focused tests and verify RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_message_actions.py -q
```

Expected: collection/import failures for the new contract, then missing Fork/group assertions.

- [ ] **Step 4: Add the smallest pure contract**

Use frozen, slotted dataclasses and immutable tuples. The public shapes should remain narrow:

```python
CONSOLE_FORK_TITLE_MAX_LENGTH = 60
ConsoleForkDurability = Literal[
    "temporary", "durable", "unsaved_persistable"
]

@dataclass(frozen=True, slots=True)
class ConsoleForkEligibility:
    eligible: bool
    reason: str = ""

@dataclass(frozen=True, slots=True)
class ConsoleForkLineageFence:
    native_message_id: str
    persisted_message_id: str | None
    native_parent_id: str | None
    role: ConsoleMessageRole
    status: ConsoleMessageStatus
    visible_content: str
    visible_variant_id: str | None
    sibling_identity: tuple[str, ...]
    persisted_revision: int | None
    attachment_fingerprint: str

@dataclass(frozen=True, slots=True)
class ConsoleForkFence:
    source_session_id: str
    source_conversation_id: str | None
    source_conversation_version: int | str | None
    source_durability: ConsoleForkDurability
    source_title: str
    source_configuration_fingerprint: str
    boundary_message_id: str
    lineage: tuple[ConsoleForkLineageFence, ...]
    image_selections: tuple[ConsoleForkImageSelectionFence, ...]

@dataclass(frozen=True, slots=True)
class ConsoleForkConfigurationSnapshot:
    workspace_id: str
    settings: ConsoleSessionSettings  # copied with pinned_prefill=None
    rag_scope: RagScope | None
    context_policy_overrides: ConsoleContextPolicyOverrides
    library_policy: ConsoleLibraryPolicyCandidate
    runtime_backend: str
    assistant_kind: str | None
    assistant_id: str | None
    assistant_authority_id: str | None
    persona_memory_mode: str | None
    character_id: int | None
    character_name: str | None
    user_display_name_override: str | None
    character_system_template: str | None
    speech_preferences: ConsoleSpeechPreferences
    project_instruction_state: ProjectInstructionControlState

ConsoleForkCitationState = Literal["active_required", "unavailable", "none"]

@dataclass(frozen=True, slots=True)
class ConsoleForkCitationLink:
    source_persisted_message_id: str
    source_revision: int
    state: ConsoleForkCitationState

@dataclass(frozen=True, slots=True)
class ConsoleChatForkSnapshot:
    fork_session_id: str
    fork_conversation_id: str | None
    title: str
    source_session_id: str
    source_conversation_id: str | None
    source_boundary_persisted_message_id: str | None
    durable: bool
    messages: tuple[ConsoleForkProjectedMessage, ...]
    configuration: ConsoleForkConfigurationSnapshot
    citation_links: tuple[ConsoleForkCitationLink, ...]
```

Fingerprint bounded canonical JSON with a domain-separated SHA-256 helper. Never hash `repr()`, rendered Rich/Textual objects, raw file paths, scratch data, or permission state.

- [ ] **Step 5: Implement the action groups and project-control sanitizer**

Add `fork_requested` to `ConsoleActionStatus`, add `f Fork` to the guide, preserve the current Speak/Stop swap, and split the existing action tuple through small filtering helpers rather than duplicating action resolution. Media actions must not be included in the generic primary/overflow row.

- [ ] **Step 6: Run GREEN and commit**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_project_instructions.py -q
git diff --check
git add Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_message_actions.py \
  tldw_chatbook/Chat/console_chat_fork.py \
  tldw_chatbook/Chat/console_message_actions.py \
  tldw_chatbook/Chat/console_project_instructions.py
git commit -m "feat: define Console chat fork projection"
```

## Task 2: Fence the canonical active lineage and stage an independent snapshot

**Files:**

- Modify: `Tests/Chat/test_console_chat_fork.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`
- Modify: `Tests/Chat/test_console_conversation_hydration.py`
- Modify: `Tests/UI/test_console_session_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_fork.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/console_conversation_hydration.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`

- [ ] **Step 1: Add failing store tests for the exact boundary**

Build a real in-memory store tree with a sibling branch and messages after the selected boundary. Assert that `issue_fork_fence(message_id)` contains only `active_path_message_ids(session_id)` through the selected ID, inclusive. Include selected text variants and assert that the fence captures their identity/content without calling Keep or changing the source selection.

Add negative tests for:

- selected ID not on the active path;
- USER/ASSISTANT status/content eligibility;
- durable source where any included node lacks `persisted_message_id`;
- changed configuration fingerprint, conversation version, durability mode, or displayed title;
- off-path, later, TOOL/activity, pending, and discarded state;
- a content, status, parent, selected-variant, sibling-set, attachment, persisted-ID, or persisted-version change after issue.

- [ ] **Step 2: Pin source immutability and fresh ownership**

Before snapshot staging, serialize all source session/message/store indices relevant to the feature. After staging both durable and temporary snapshots, assert byte-equivalent serialized source state and fresh fork session, message, parent, turn, text-variant, attachment-owner, and generation-owner IDs. The snapshot must set usage, draft, pending attachments, one-shot prefill, context summary/boundary, continuation/recovery, todo/run state, scratch/approval authority, and live video keys to absent/default values.

Assert the typed configuration snapshot includes and fences the exact allowlist: title, Workspace ID, `ConsoleSessionSettings` with `pinned_prefill=None`, RAG scope, sparse context/compaction overrides, effective Library values seeded as a fresh candidate, runtime/assistant/persona/character identity, system/user-display identity, speech preferences, and sanitized project-instruction controls. Changing any included field stales the fence; changing draft, one-shot or pinned prefill, usage, scratch, approval, run, or presentation state does not cause that excluded value to enter the fingerprint or snapshot.

Add the currently missing `persona_memory_mode` typed field to `ConsoleChatSession` and thread it through `create_session()`, `restore_persisted_session()`, `ChatPersistenceService.create_conversation()`, first persistence/promotion kwargs, `hydrate_console_conversation()` in `console_conversation_hydration.py`, and `ConsoleSessionController` serialize/restore. Add a persona-backed create → screen snapshot → restore and real-SQLite first-persist → canonical hydration → resumed-session round trip. Do not query durable state ad hoc midway through projection.

- [ ] **Step 3: Run the new nodes and verify RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_session_controller.py -q
```

Expected: missing issue/revalidate/stage APIs.

- [ ] **Step 4: Implement issue/revalidate as one store-owned read contract**

Add public methods with no UI dependency:

```python
def fork_eligibility(self, message_id: str) -> ConsoleForkEligibility: ...
def issue_fork_fence(
    self,
    message_id: str,
    *,
    image_selections: Sequence[ConsoleForkImageSelectionFence] = (),
) -> ConsoleForkFence: ...
def validate_fork_fence(
    self,
    fence: ConsoleForkFence,
    *,
    image_selections: Sequence[ConsoleForkImageSelectionFence] = (),
) -> bool: ...
def stage_fork_snapshot(
    self,
    fence: ConsoleForkFence,
    *,
    title: str,
    fork_session_id: str,
    fork_conversation_id: str | None,
) -> ConsoleChatForkSnapshot: ...
```

Build the active prefix from canonical tree indices, not `messages_for_session()` and not `_tree_nodes_parent_first()`. Re-read every captured field during validation and ensure the source conversation version, durability mode, displayed title, configuration fingerprint, and selected boundary's active-prefix position still match. The configuration fingerprint covers exactly the typed `ConsoleForkConfigurationSnapshot` fields above; it replaces any nonexistent omnibus revision without adopting the store's over-broad payload revision.

- [ ] **Step 5: Add detached postcommit registration**

Construct all new `ConsoleChatSession` and `ConsoleChatMessage` values locally, validate parent relationships and ID uniqueness, then publish the complete session to store indices in one synchronous method. Support `activate=False`; the controller activates only after successful registration. On an exception, no partial fork session/index remains.

For a temporary fork, `persisted_conversation_id` and ancestry are absent and `ephemeral=True`. For a saved independent root from a non-ephemeral unsaved source, `ephemeral=False`, `root_id` equals the preallocated target conversation ID, and parent/fork columns are null. For a durable source, preserve ancestry only in the durable conversation record—not as shared live ownership.

- [ ] **Step 6: Run GREEN and commit**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_chat_store_tree.py \
  Tests/Chat/test_console_chat_store_sibling.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_session_controller.py -q
git diff --check
git add Tests/Chat/test_console_chat_fork.py Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_session_controller.py \
  tldw_chatbook/Chat/console_chat_fork.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/Chat/console_conversation_hydration.py \
  tldw_chatbook/UI/Console_Modules/session.py
git commit -m "feat: fence Console fork snapshots"
```

## Task 3: Fence selected generated images and degrade video truthfully

**Files:**

- Modify: `Tests/Chat/test_console_image_controller.py`
- Modify: `Tests/Chat/test_console_chat_fork.py`
- Modify: `Tests/UI/test_console_message_controller.py`
- Modify: `tldw_chatbook/UI/Console_Modules/image.py`
- Modify: `tldw_chatbook/UI/Console_Modules/message.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/Chat/console_chat_fork.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`

- [ ] **Step 1: Write failing image-fence tests**

For each included generated-image message, capture:

```python
(native_message_id, selected_position, browse_revision, attachment_meta_fingerprint)
```

Assert that changing the browsed position, Keep state, attachment order/data digest, generation metadata, deleting a variant, or cleaning up browse state makes validation fail. An unrelated image outside the prefix must not stale the fence.

- [ ] **Step 2: Write failing media projection tests**

Assert that only the selected generated-image attachment and matching generation metadata are rebuilt under the fork message's fresh owner/position. Ordinary sent attachments retain position/order/content/display name after the canonical size/type/data-presence checks; a missing/corrupt required attachment or generated image fails before commit and never falls back to a source filesystem path. A video message copies text and a bounded expired-video tombstone/disclosure, but never file bytes, a path, a video-store key, cleanup ownership, or playable state. Remap `source_image_message_id` only when its source image is inside the snapshot; otherwise clear it.

- [ ] **Step 3: Run RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_image_controller.py \
  Tests/Chat/test_console_chat_fork.py \
  Tests/UI/test_console_message_controller.py -q
```

- [ ] **Step 4: Implement image-controller capture and validation**

Keep the existing screen-owned browse map, add an image-controller-owned monotonic revision per message, and bump it on browse, Keep, delete, subtree removal, and cleanup. Expose named capture/validate/invalidate callbacks. Route message/subtree deletion to the invalidation callback through `message.py` and `wiring.py`; neither the session controller nor message controller may reach through screen-owned browse state.

Compute attachment/meta fingerprints in `console_chat_fork.py`. Return no raw bytes in the fence; bytes are copied only while the store stages the validated snapshot.

- [ ] **Step 5: Run GREEN and commit**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_image_controller.py \
  Tests/Chat/test_console_generation_store.py \
  Tests/Chat/test_console_chat_fork.py \
  Tests/UI/test_console_message_controller.py -q
git diff --check
git add Tests/Chat/test_console_image_controller.py \
  Tests/Chat/test_console_chat_fork.py \
  Tests/UI/test_console_message_controller.py \
  tldw_chatbook/UI/Console_Modules/image.py \
  tldw_chatbook/UI/Console_Modules/message.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/Chat/console_chat_fork.py \
  tldw_chatbook/Chat/console_chat_store.py
git commit -m "feat: fence Console fork media choices"
```

## Task 4: Commit durable forks and governed sidecars atomically

**Files:**

- Create: `Tests/Chat/test_console_chat_fork_persistence.py`
- Modify: `Tests/Chat/test_citation_trace_repository.py`
- Modify: `Tests/Chat/test_console_chat_store_project_instructions.py`
- Modify: `Tests/Chat/test_console_chat_store_atomic_promotion.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/citation_trace_repository.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`

- [ ] **Step 1: Write real-SQLite RED tests for ancestry and rollback**

Use the repository's real in-memory `ChaChaNotes_DB`, not mocks. A durable fork must persist:

```text
id = preallocated target conversation id
parent_conversation_id = source conversation id
forked_from_message_id = source boundary persisted message id
root_id = source.root_id or source.id
active_leaf_message_id = copied boundary message id
```

All copied messages use preallocated fresh IDs and copied parent IDs. Inject failure after conversation insert, a middle message, attachment/generation sidecar, citation link, policy insert, context-policy/project-context updates, and active-leaf update; every case must roll back all target rows and leave source row counts/content/version/active leaf unchanged.

Add a barrier race: complete the app-loop fence validation, mutate one captured durable source message/version (and separately the conversation version), then enter the fork transaction. Cursor-scoped source checks must reject both races before target insertion.

Persist the generated-video tombstone with no bytes/path/store key. In real SQLite reload tests, remap an in-snapshot `source_image_message_id` to the fork image ID, clear an out-of-snapshot reference, and reject missing/corrupt required attachment or generated-image payloads before the transaction starts.

- [ ] **Step 2: Write idempotency and source-kind tests**

Calling the bundle again with the same target ID and immutable fork identity returns `already_committed=True` and the same preallocated message map. Simulate an exception after commit and prove `resolve_console_fork_commit()` finds that same bundle; prove absence permits same-ID retry and a conflicting root/parent/boundary/title/active-leaf fails closed. A non-ephemeral unsaved source creates a saved target whose `root_id` equals its own preallocated ID and whose parent/fork columns are null, without creating a source conversation. A temporary source performs no DB write.

- [ ] **Step 3: Write governed citation tests**

Create one real active source trace/owner. The fork should add a fresh message-owner row that points to the same immutable trace/payload identity. Snapshot each source message as `active_required`, `unavailable`, or `none`. An `active_required` trace that becomes revoked, loses payloads, or fails body/revision/profile/fingerprint validation before commit must abort before publication. Only a trace already canonically unavailable in the confirmed snapshot may commit without an owner link. The operation must never clone trace/payload rows or invent provenance from citation display text.

Temporary forks retain textual citation markers only, expose no governed owner, and later promotion does not reconstruct one.

- [ ] **Step 4: Write project-context atomicity tests**

The committed project-context JSON retains only enabled, binding ID, and locator fingerprint, with a null notice key. The same fork transaction must also persist the copied sparse `ConsoleContextPolicyOverrides`; it may not rely on `_flush_context_policy_on_first_persist`. Inject either write failure and assert the entire durable fork/promotion rolls back. Reload must reproduce every persisted configuration category: Workspace/scope, `ConsoleSessionSettings` metadata and system prompt, RAG scope, context policy, fresh Library policy values/revision, role/persona identity, speech preferences, and sanitized project controls. Update the existing ephemeral-promotion test to prove there is no postcommit project-context or context-policy writer.

- [ ] **Step 5: Run RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_chat_fork_persistence.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_console_chat_store_project_instructions.py \
  Tests/Chat/test_console_chat_store_atomic_promotion.py -q
```

- [ ] **Step 6: Implement one dedicated persistence transaction**

Extend `create_conversation()` only enough to forward the schema's existing `root_id`, `parent_conversation_id`, and `forked_from_message_id` fields. Add:

```python
def fork_console_conversation_bundle(
    self,
    *,
    snapshot: ConsoleChatForkSnapshot,
    conversation_kwargs: Mapping[str, object],
    policy_candidate: ConsoleLibraryPolicyCandidate,
    project_context_json: str,
) -> ConsoleForkCommitResult: ...

def resolve_console_fork_commit(
    self,
    snapshot: ConsoleChatForkSnapshot,
) -> ConsoleForkCommitResult | None: ...
```

The bundle owns one outer `transaction(immediate=True)`. Before target insertion, the same cursor re-reads the captured source conversation version and every source durable message's ID/version/body/deleted state and rejects a mismatch. It then validates or creates the target conversation, inserts all messages/sidecars, links citations, writes sparse context policy and sanitized project context with the same cursor, sets the active leaf, and returns only durable identities after commit. The resolver performs the same root/parent/boundary/title/active-leaf identity check without writing and returns matching, absent, or raises on collision. SQLite atomicity is the proof for the rest of the bundle; do not add per-row digests or an operation table. Do not reuse the unrelated Workspace membership `fork_conversation_into_workspace()` method.

- [ ] **Step 7: Add one citation repository seam**

Implement a cursor-scoped `link_fork_message_owner(...)` that composes the repository's existing active-owner body/revision/fingerprint validation with `link_cache_message_owner()`. Its inputs identify source and target messages plus the confirmed citation state; it never accepts caller-supplied trace bodies or rendered citation objects. An `active_required` link must still validate and link or abort the transaction. Only confirmed `unavailable`/`none` entries omit a link. Keep revocation and payload availability checks fail-closed.

- [ ] **Step 8: Make ordinary temporary promotion project-context atomic**

Add both `project_context_json` and `context_policy_overrides` to `promote_console_conversation_bundle()` and write both before commit through the outer transaction's cursor. In `_promote_ephemeral_session_atomically()`, pass the sanitized JSON and sparse overrides in the bundle, remove the postcommit `_persist_project_instruction_state(session)` call, and bypass/remove `_flush_context_policy_on_first_persist(session)` for this promotion path. This is required so a temporary fork cannot later weaken the feature's atomicity contract.

Add a postcommit Workspace membership test at the store/controller seam: a registry failure preserves the committed conversation, records the incumbent retryable projection state, and reconciliation links that same conversation without a second fork transaction.

- [ ] **Step 9: Run GREEN and commit**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_chat_fork_persistence.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/Chat/test_console_chat_store_project_instructions.py \
  Tests/Chat/test_console_chat_store_atomic_promotion.py \
  Tests/Chat/test_chat_persistence_service.py -q
git diff --check
git add Tests/Chat/test_console_chat_fork_persistence.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_console_chat_store_project_instructions.py \
  Tests/Chat/test_console_chat_store_atomic_promotion.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  tldw_chatbook/Chat/console_chat_store.py
git commit -m "feat: commit Console chat forks atomically"
```

## Task 5: Add the direct Fork action and captured-target More menu

**Files:**

- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Tests/Chat/test_console_generation_card.py`
- Modify: `Tests/Chat/test_console_video_message.py`
- Create: `tldw_chatbook/Widgets/Console/console_message_more_menu.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/Widgets/Console/console_generation_card.py`
- Modify: `tldw_chatbook/Widgets/Console/console_video_card.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/UI/Console_Modules/message.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [ ] **Step 1: Write production-shaped action-row RED tests**

Using `Tests.UI.consolidated_css.ConsolidatedCSSApp`, mount real selected USER, ASSISTANT, stopped/failed, generated-image, video, and TOOL rows. Assert direct order and that the row fits at `120x35`, `100x30`, and `80x24` without clipping or horizontal scroll. Assert visible ineligible help for each disabled condition; `f` repeats that reason.

- [ ] **Step 2: Write More-menu target and focus RED tests**

Split lifecycle from dispatch. First, open More for message A and prove selection change to B, row recomposition/removal, transcript refresh, Escape, and click-away detach the menu with no action event. Separately choose Helpful for A, race a selection change after the choice, and assert the already captured event still carries A's message/action IDs. Await popup detachment before posting that event, and assert downstream dispatch/modal work starts only after no menu widget remains.

Focus fallback order is exact opener button if still mounted, selected row otherwise, composer last. Test complete keyboard traversal and dismissal without relying on disabled-button focus or hover.

- [ ] **Step 3: Pin media-card ownership**

Generation image navigation/Keep/View/Save and video Play/Save remain reachable on their card surfaces and are absent from the generic selected-message row. TOOL/activity controls remain direct and unchanged.

- [ ] **Step 4: Run RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/Chat/test_console_generation_card.py \
  Tests/Chat/test_console_video_message.py -q
```

- [ ] **Step 5: Implement the bounded menu and action row**

`ConsoleMessageMoreMenu` owns immutable `message_id` plus the four action IDs. On choice, it awaits Textual detachment/removal, then posts one event carrying both values; it exposes no store/controller access. `ConsoleTranscript` owns mount/unmount and focus fallback. Add the `f` binding beside `c/e/r`, the Fork tooltip, and guide text. Preserve event-button attributes as the primary stable parse seam.

Because Fork eligibility is store-owned, add the narrow message-controller/wiring/screen delegation in this task and pass the resulting frozen `ConsoleForkEligibility` into transcript action resolution. `chat_screen.py` receives delegation plus F1/Inspector help-copy updates only—no fork persistence or orchestration logic. Assert the button, `f`, guide, tooltip, F1 help, More labels, and action IDs remain synchronized.

- [ ] **Step 6: Move media controls and regenerate CSS if needed**

Use the existing card widgets; do not create a second media toolbar framework. If source TCSS changes, run:

```bash
../../.venv/bin/python -B -m tldw_chatbook.css.build_css
```

Never edit `tldw_chatbook/css/tldw_cli_modular.tcss` by hand.

- [ ] **Step 7: Run GREEN and commit**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/Chat/test_console_generation_card.py \
  Tests/Chat/test_console_video_message.py \
  Tests/Chat/test_console_message_actions.py -q
git diff --check
git add Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/Chat/test_console_generation_card.py \
  Tests/Chat/test_console_video_message.py \
  tldw_chatbook/Widgets/Console/console_message_more_menu.py \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  tldw_chatbook/Widgets/Console/console_generation_card.py \
  tldw_chatbook/Widgets/Console/console_video_card.py \
  tldw_chatbook/Widgets/Console/__init__.py \
  tldw_chatbook/UI/Console_Modules/message.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "feat: add Console fork message action"
```

If neither CSS file changed, omit both from `git add`.

## Task 6: Build the naming modal and cancellable controller workflow

**Files:**

- Create: `Tests/UI/test_console_fork_chat_modal.py`
- Modify: `Tests/UI/test_console_message_controller.py`
- Modify: `Tests/UI/test_console_session_controller.py`
- Create: `tldw_chatbook/Widgets/Console/console_fork_chat_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_edit_message_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/UI/Console_Modules/message.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`

- [ ] **Step 1: Write modal-state RED tests**

Use the production stylesheet harness at all three target sizes. Cover:

1. `editing` — selected editable default title, boundary excerpt, destination, citation/video/exclusion disclosure, Confirm and Cancel;
2. `validating` — spinner/status, Cancel/Escape/backdrop permitted, Confirm fenced;
3. `committing` — controls disabled, Escape/backdrop explain that commit is finishing;
4. `precommit_error` — entered title retained and retryable;
5. `stale_source` — exact source-changed copy and no commit;
6. `created_not_opened` — saved/temporary identity displayed with a single Open action that reuses it.

Enter accepts only from editing/precommit-error with a nonblank normalized title. The input is bounded to 60 characters. Double click/Enter produces one confirmation result.

- [ ] **Step 2: Write barrier-controlled cancellation RED test**

Pause validation on an `asyncio.Event`, cancel the modal, release the late validator, and assert no commit/store registration/activation call occurred. Then run a second request and prove the older validation generation cannot commit the newer request. This test must exercise the real controller generation check on the app loop, not merely cancel a cooperative coroutine.

- [ ] **Step 3: Write controller ordering and recovery RED tests**

Assert this sequence:

```text
capture fence -> show/confirm -> validate -> durable commit OR temp register
-> idempotent Workspace projection when durable -> register durable live session
-> activate fork
```

No store publication precedes a durable commit. A deterministic persistence failure, or an ambiguous failure whose target-ID reconciliation proves absence, stays precommit-error and leaves no fork; Retry uses the same IDs. An exception after SQLite commit must reconcile to the matching created fork and continue without a second bundle call. A conflicting target row fails closed.

For durable results, hydration/registration failure after commit switches the modal to created-not-opened without removing the durable fork; activation failure leaves the registered target discoverable. For temporary results, detached-session registration failure leaves no fork at all, while activation failure preserves exactly one registered target session for Open retry. Retrying Open always reuses the same conversation/session ID. Successful activation leaves the original session registered/open and unchanged.

Also inject a postcommit Workspace membership projection failure. The committed fork must keep its identity, record/reuse the existing idempotent reconciliation seam, and never duplicate the durable bundle. Hydration/activation may continue with a truthful pending-membership notice if that is the incumbent workspace behavior; otherwise the modal uses created-not-opened and Open retries projection before activation.

- [ ] **Step 4: Run RED**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_fork_chat_modal.py \
  Tests/UI/test_console_message_controller.py \
  Tests/UI/test_console_session_controller.py -q
```

- [ ] **Step 5: Implement a presentation-only modal**

Keep the modal's public contract to immutable initial summary data, a result containing the normalized title or cancellation, and explicit state-update methods. It must not read the store, write persistence, or create sessions. Reuse `SafeModalDismissMixin` and the edit modal's stale-key guard pattern.

Change Edit modal wording from “forks a new branch” to “creates a new response branch in this chat” so the two concepts remain distinct.

- [ ] **Step 6: Implement controller orchestration with named callbacks**

`ConsoleMessageController` handles the `fork_requested` result by calling `request_console_chat_fork(message_id)`. `ConsoleSessionController` owns a monotonic `_fork_validation_generation`, preallocated IDs, worker execution, modal state transitions, persistence, postcommit registration, and `_activate_native_console_session()`.

On any non-definitive durable write exception, call `resolve_console_fork_commit()` before showing an error: matching means created, absent permits retry with the same ID, and collision is terminal. Never infer rollback from the exception alone.

Wire image fence capture/validation and session/message delegation through constructor callbacks in `wiring.py`; do not reach through `ChatScreen` internals. Operations expected to exceed 100 ms run through the existing worker seam, but all store revalidation and generation-token comparisons occur on the app loop immediately before commit.

- [ ] **Step 7: Run GREEN and commit**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_fork_chat_modal.py \
  Tests/UI/test_console_message_controller.py \
  Tests/UI/test_console_session_controller.py \
  Tests/UI/test_console_native_chat_flow.py -q
git diff --check
git add Tests/UI/test_console_fork_chat_modal.py \
  Tests/UI/test_console_message_controller.py \
  Tests/UI/test_console_session_controller.py \
  tldw_chatbook/Widgets/Console/console_fork_chat_modal.py \
  tldw_chatbook/Widgets/Console/console_edit_message_modal.py \
  tldw_chatbook/Widgets/Console/__init__.py \
  tldw_chatbook/UI/Console_Modules/message.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/UI/Console_Modules/wiring.py
git commit -m "feat: orchestrate Console chat forks"
```

## Task 7: Prove the complete fork journey and document it

**Files:**

- Create: `Tests/integration/test_console_chat_fork_flow.py`
- Modify: `Docs/User_Guide/console/chat-basics.md`
- Modify: `Docs/User_Guide/console/branching-and-rewind.md`
- Modify: `Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md`
- Modify: `backlog/tasks/task-23088 - Fork-Console-chat-from-a-selected-message.md`

- [ ] **Step 1: Write the provider-free integration test**

Seed a durable Console tree with USER/ASSISTANT messages, a sibling, attachment, generated-image choice, project controls, Library/RAG settings, and active citation. Fork from a middle USER and ASSISTANT boundary, rename one fork, switch among source/forks, close/recreate the store, reload durable forks, and verify exact prefixes and ancestry. Assert the source DB and live snapshot captured before forking are unchanged after both operations.

Add a temporary-to-temporary fork and later promotion; reload it as an independent root with sanitized project controls, textual citation markers only, and unavailable-video tombstones.

Exercise live authority isolation, not just absent snapshot fields: source and fork allocate different scratch roots and lease generations; approvals, MCP/local-tool grants, recovery owners, and active runs remain source-only; the retained Workspace/project binding is re-resolved through normal validation before use; no resolved instruction body, raw folder path, attachment bytes, permission decision, or secret appears in the snapshot, notification, diagnostic, or captured log.

- [ ] **Step 2: Run the integration test and targeted regression set**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/integration/test_console_chat_fork_flow.py \
  Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_chat_fork_persistence.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_edit_message_modal.py \
  Tests/Chat/test_console_regenerate_branching.py \
  Tests/UI/test_console_fork_chat_modal.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_message_controller.py \
  Tests/UI/test_console_session_controller.py -q
```

Do not run the full repository suite unless the user explicitly opts in.

- [ ] **Step 3: Update user documentation**

In chat basics, show the new direct row order, `f` shortcut, More contents, and media-card actions. In branching/rewind, add a distinct “Fork chat from here” section explaining:

- inclusive active-lineage boundary and untouched source;
- editable default name and quick Enter acceptance;
- saved versus temporary behavior;
- independent-root rule for later saving a temporary fork;
- excluded tools/runs/scratch/approvals/recovery/usage;
- governed citation behavior and temporary textual markers;
- video is unavailable in the fork unless the user separately saved a copy.

- [ ] **Step 4: Perform one isolated live local TUI journey**

Use a temporary HOME/XDG/config/data location and deterministic seeded local DB; do not point the app at the developer's normal data. At `120x35` and `80x24`, exercise keyboard and pointer paths, rename a fork, switch all tabs, restart, and verify source/fork reload. Capture the exact command, viewport, DB fixture, and observations in the task Implementation Notes. No provider call is required.

- [ ] **Step 5: Run static checks limited to changed Python files**

```bash
../../.venv/bin/python -B -m ruff check \
  tldw_chatbook/Chat/console_chat_fork.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  tldw_chatbook/Chat/console_message_actions.py \
  tldw_chatbook/UI/Console_Modules/image.py \
  tldw_chatbook/UI/Console_Modules/message.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/Widgets/Console/console_fork_chat_modal.py \
  tldw_chatbook/Widgets/Console/console_message_more_menu.py \
  tldw_chatbook/Widgets/Console/console_transcript.py
git diff --check
```

- [ ] **Step 6: Self-review against ADR-092 and close task hygiene only after evidence exists**

Review the diff specifically for source mutation, deep-copy use, cross-DB “atomic” claims, broad session cloning, copied authority, missing cancellation generation checks, postcommit duplicate creation, and actions that disappear at 80 columns.

Then:

- check every Acceptance Criterion in `TASK-23088`;
- add concise Implementation Notes listing approach, trade-offs, modified files, targeted test/static/live evidence, and ADR-092;
- set task status to Done using a direct file edit if the five-digit Backlog CLI bug remains;
- add a lessons entry only if implementation produced a genuinely reusable incident;
- recheck `TASK-23088` uniqueness across local/remote refs and worktrees immediately before merge.

- [ ] **Step 7: Commit final integration/docs**

```bash
git add Tests/integration/test_console_chat_fork_flow.py \
  Docs/User_Guide/console/chat-basics.md \
  Docs/User_Guide/console/branching-and-rewind.md \
  Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md \
  'backlog/tasks/task-23088 - Fork-Console-chat-from-a-selected-message.md'
git commit -m "test: verify Console chat fork flow"
```

## Completion gate

Before claiming completion, record fresh evidence for:

- focused domain/action tests;
- real-SQLite atomicity, idempotency, citation, and project-context tests;
- generated-image and video degradation tests;
- production-shaped modal/menu/focus/layout tests at `120x35`, `100x30`, and `80x24`;
- provider-free reload integration test;
- changed-file Ruff and `git diff --check`;
- isolated live TUI journey;
- clean task/ADR/spec/plan links and a unique Backlog task ID.

The full test suite is intentionally outside this plan unless the user opts in.
