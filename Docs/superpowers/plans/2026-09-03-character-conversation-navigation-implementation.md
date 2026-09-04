# Character Conversation Navigation and Local Meaning Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Apply
> `superpowers:test-driven-development` before every behavior change and
> `superpowers:verification-before-completion` before every completion claim.
> Each numbered task is one independently reviewed pull request and one Backlog
> task; do not combine adjacent tasks.

**Goal:** Let first-time and expert users find, inspect, repair, search, and
resume exact local character conversations from Console Context, Console
`Ctrl+K`, and Roleplay, with optional explicitly enabled local Meaning search.

**Architecture:** One data-profile-scoped projection owns resolved and
unresolved identity, selected-branch eligibility, Keyword indexing, paging, and
repair preconditions. Context, the switcher, Roleplay, and Library consume typed
read models and one Console-owned activation coordinator; none invent identity
or mutate transcripts. A separate opt-in semantic subsystem consumes the same
eligible documents, stores embeddings without plaintext, and publishes only
atomic ready generations through bounded direct ANN retrieval.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5, asyncio workers,
dataclasses and enums, ChromaDB HNSW with cosine distance, local embedding
providers, pytest/pytest-asyncio, Ruff, modular TCSS and generated CSS bundles.

**Spec:**
`Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md`

**Backlog sequence:**

1. `TASK-31241` — decisions and contract alignment
2. `TASK-31242` — shared projection and Keyword index
3. `TASK-31243` — navigation recovery and Roleplay vertical slice
4. `TASK-31244` — Console Context Character section
5. `TASK-31245` — `Ctrl+K` Character chats
6. `TASK-31246` — local semantic foundation
7. `TASK-31247` — Settings lifecycle and Roleplay Meaning
8. `TASK-31248` — `Ctrl+K` Meaning and integrated qualification

## Global Constraints

- Work on one task at a time from current `origin/dev`; re-sweep every remote
  ref and worktree for task, ADR, and schema allocations immediately before
  filing or rebasing each PR.
- The first release searches local conversations in the active Data Profile
  only. It does not browse, count, index, search, or resume cached or live server
  conversations.
- `data_authority_id` means the database's durable `local_authority_id`; it is
  never an absolute database path, display label, or RAG configuration-profile
  identifier.
- Character cards are eligible; Personas are not.
- Keyword and Meaning consume only the selected visible user/assistant branch.
  System, thinking, tool, attachment, non-selected-branch, deleted, invalid,
  and inaccessible content is excluded.
- Existing `messages_fts` is not eligible for reuse. Character Keyword search
  owns a separate versioned derived generation.
- Meaning is explicit, local, opt-in, default-off, network-free, and direct ANN
  retrieval. No remote embedding fallback is permitted.
- Context remains Keyword-only. Meaning first ships end-to-end in Roleplay and
  reaches the switcher only in Task 8.
- Enter in Character surfaces activates the exact immutable highlighted
  conversation. It never substitutes a same-name card, current card, current
  profile, or fallback conversation.
- Blank Active-mode Enter in `Ctrl+K` keeps the incumbent MRU-other-tab
  behavior.
- Every user-facing PR owns its 52×20 keyboard, pointer, focus, truncation,
  empty, failure, and primary-action evidence; Task 8 adds integrated evidence
  rather than replacing the owning gates.
- Use workers for database, embedding, or filesystem operations that can exceed
  100 ms. Keep event-loop slices at or below 50 ms and expose busy state within
  100 ms.
- Run only targeted and reachable suites unless the user separately approves a
  full repository sweep.
- Edit source TCSS, run `../../.venv/bin/python -m tldw_chatbook.css.build_css`,
  and verify bundles with
  `../../.venv/bin/python -m tldw_chatbook.css.check_bundle_sync` whenever a UI
  task changes styling.
- Do not launch schema-changing code against the user's real profile. Live TUI
  runs use a disposable `HOME`, XDG paths, `TLDW_CONFIG_PATH`, and `[paths].data_dir`,
  then verify the decoy profile remains byte-identical.

## Programme ADR Check

ADR required: yes

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: the programme introduces durable derived indexes, a local identity and
repair contract, cross-screen activation and draft-veto boundaries, semantic
consent/lifecycle policy, and long-lived navigation structure. `ADR-116` is the
current collision-free allocation; Task 1 must re-sweep before creating it and
renumber the unshipped path plus every reference if an older claimant lands.

Task 1 also amends ADR-004
(`backlog/decisions/004-personas-destination-native-workbench.md`), ADR-030,
ADR-037, ADR-046, ADR-083, and ADR-085.
ADR-031 and ADR-033 remain linked, unchanged authorities.

## File Map

### Governance

- Create
  `backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`.
- Modify `backlog/decisions/004-personas-destination-native-workbench.md`
  plus ADR-030, ADR-037, ADR-046, ADR-083, and ADR-085 as named in the
  programme ADR check.
- Keep the approved spec and this plan linked from every Backlog task.

### Shared identity, projection, and Keyword search

- Create
  `tldw_chatbook/Character_Chat/character_conversation_navigation.py` for typed
  identities, immutable presentation rows, query pages, and the application
  service facade.
- Create `tldw_chatbook/DB/character_conversation_search.py` for the selected-
  branch eligibility projector, FTS generation repository, deterministic
  backfill, repair candidate lookup, and compare-and-set repair.
- Modify `tldw_chatbook/DB/ChaChaNotes_DB.py` only for schema migration wiring,
  transaction-owned revision/outbox writes, and narrow database entry points.

### Shared navigation and recovery

- Create `tldw_chatbook/Chat/console_conversation_activation.py` for the typed
  activation state machine and result.
- Create `tldw_chatbook/UI/Navigation/character_conversation_navigation.py` for
  Roleplay, Library-repair, and return-focus navigation payloads.
- Create
  `tldw_chatbook/UI/Library_Modules/library_character_repair_controller.py` for
  repair presentation and mutation coordination.
- Modify the incumbent Console workspace opener, Roleplay conversations
  controller/widgets, Personas screen, Library screen, and main navigation.

### Console Context and switcher

- Create `tldw_chatbook/Widgets/Console/console_character_context.py` for the
  bounded Character accordion and global Keyword results.
- Create `tldw_chatbook/UI/Console_Modules/character_context.py` for Context
  loading, preference, restoration, and navigation ownership.
- Modify `tldw_chatbook/Chat/console_switcher_state.py`,
  `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py`, and
  `tldw_chatbook/UI/Screens/chat_screen.py` for the third switcher mode and typed
  activation.

### Local semantic subsystem

- Create `tldw_chatbook/RAG_Search/character_conversations/contracts.py` for
  manifests, chunks, jobs, typed query outcomes, and lifecycle states.
- Create `tldw_chatbook/RAG_Search/character_conversations/chunking.py` for
  deterministic eligible-document chunking.
- Create `tldw_chatbook/RAG_Search/character_conversations/vector_store.py` for
  embeddings-only, generation-isolated Chroma operations.
- Create `tldw_chatbook/RAG_Search/character_conversations/index_service.py` for
  initial build, rebuild, outbox replay, reconciliation, pause/resume/cancel,
  deletion, and atomic readiness.
- Create `tldw_chatbook/RAG_Search/character_conversations/query_service.py` for
  local model validation and bounded direct ANN aggregation.
- Create `tldw_chatbook/RAG_Search/character_conversations/settings.py` for
  staged configuration values and saved lifecycle configuration.

### Settings and documentation

- Create
  `tldw_chatbook/Widgets/Settings_Widgets/character_chat_search_panel.py` and
  mount it through canonical `tldw_chatbook/UI/Screens/settings_screen.py`.
- Modify `tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py`
  and its widgets for Keyword/Meaning strategy and status.
- Create `tldw_chatbook/UI/Console_Modules/character_switcher_search.py` for the
  120 ms two-leg presentation coordinator.
- Modify `Docs/User_Guide/console/sessions-tabs-workspaces.md`,
  `Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md`, and
  `Docs/User_Guide/settings/rag.md` in the owning UI tasks.

---

### Task 1: Align Decisions and Contracts — TASK-31241

**Files:**

- Create:
  `backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`
- Modify: `backlog/decisions/004-personas-destination-native-workbench.md`
- Modify: `backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md`
- Modify:
  `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`
- Modify:
  `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`
- Modify:
  `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`
- Modify:
  `backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md`
- Modify:
  `backlog/tasks/task-31241 - Align-character-conversation-navigation-decisions.md`

**Interfaces:**

- Consumes: the approved design and preserved ADR-031/ADR-033 contracts.
- Produces: ADR-116's exact local authority, identity union, selected-branch
  corpus, activation, repair, consent, generation, and surface-ownership rules.

ADR required: yes

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: this task creates the programme's architectural authority before code.

- [ ] **Step 1: Reserve the governance identifiers against current history**

Run the remote/worktree task and ADR sweeps from
`backlog/docs/lessons-backlog-hygiene.md`, fetch current `origin/dev`, and record
that `TASK-31241` and ADR-116 have no older claimant. If either collides, apply
the older-arrival rule and change every unshipped reference in this plan, spec,
and task chain before continuing.

- [ ] **Step 2: Put TASK-31241 in progress and attach its executable plan**

Run:

```bash
backlog task edit 31241 -s "In Progress"
backlog task edit 31241 --plan "1. Reconfirm task and ADR allocations.\n2. Create ADR-116 from the approved design.\n3. Amend 004-personas-destination-native-workbench.md, ADR-030, ADR-037, ADR-046, ADR-083, and ADR-085.\n4. Link preserved ADR-031 and ADR-033.\n5. Run documentation and reference checks."
```

- [ ] **Step 3: Write ADR-116 before amending its consumers**

The Decision section must state these exact owned contracts:

```text
Identity: ResolvedLocalCharacterKey | UnresolvedConversationKey
Activation: Console-owned, cancellable before commit_started, result typed
Repair: Library-only, same-data-authority, explicit confirmation, CAS
Keyword: separate selected-branch FTS generation
Meaning: local-only, opt-in, embeddings-only, atomic ready generations
Surfaces: Context bounded; Ctrl+K operational; Roleplay complete browse
```

Include rejected alternatives for cached-server first release, broad
`messages_fts`, lexical reranking, remote embeddings, surface-owned resume,
name-based repair, and silent late-result reordering.

- [ ] **Step 4: Amend the six existing ADRs with narrow dated sections**

Each amendment links ADR-116 and `TASK-31241` and changes only its owned seam:

```text
ADR-004 (`004-personas-destination-native-workbench.md`)
         -> Roleplay complete per-character browse versus Library global/archive ownership
ADR-030  -> derived FTS/vector generations and authoritative invalidation
ADR-037  -> data_authority_id and typed resolved/unresolved character identity
ADR-046  -> exact typed activation and aggregate Roleplay draft veto
ADR-083  -> always-composed Context Character section after Conversations
ADR-085  -> third Ctrl+K mode and mounted activation state machine
```

- [ ] **Step 5: Verify preserved contracts and link integrity**

Run:

```bash
rg -n "ADR-031|031-tui-keybinding|ADR-033|033-settings" \
  backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md \
  Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
git diff --check
```

Expected: both preserved ADRs are linked, no amendment claims ownership of
reserved global keys or Settings transaction labels, and the diff check exits
zero.

- [ ] **Step 6: Commit only the governance slice**

```bash
git add backlog/decisions/004-personas-destination-native-workbench.md \
  backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md \
  backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md \
  backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md \
  backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md \
  backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md \
  backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md \
  "backlog/tasks/task-31241 - Align-character-conversation-navigation-decisions.md"
git commit -m "docs: align character conversation navigation decisions"
```

### Task 2: Build Shared Projection and Keyword Index — TASK-31242

**Files:**

- Create:
  `tldw_chatbook/Character_Chat/character_conversation_navigation.py`
- Create: `tldw_chatbook/DB/character_conversation_search.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `Tests/Character_Chat/test_character_conversation_navigation.py`
- Create: `Tests/DB/test_character_conversation_search_migration.py`
- Create: `Tests/DB/test_character_conversation_search_projection.py`
- Create: `Tests/DB/test_character_conversation_selected_branch_fts.py`
- Modify:
  `backlog/tasks/task-31242 - Build-character-conversation-projection-and-Keyword-index.md`

**Interfaces:**

- Consumes: ADR-116 and the existing `CharactersRAGDB.get_local_authority_id()`
  authority.
- Produces these immutable contracts for Tasks 3–8:

```python
class UnavailableCharacterReason(StrEnum):
    MISSING_CARD = "missing_card"
    DELETED_CARD = "deleted_card"
    MISSING_CHARACTER_AUTHORITY_LINK = "missing_character_authority_link"
    AMBIGUOUS_LEGACY_LINK = "ambiguous_legacy_link"

@dataclass(frozen=True)
class ResolvedLocalCharacterKey:
    data_authority_id: str
    character_id: int

@dataclass(frozen=True)
class UnresolvedConversationKey:
    data_authority_id: str
    conversation_id: str

@dataclass(frozen=True)
class LocalCharacterConversationTarget:
    character: ResolvedLocalCharacterKey
    conversation_id: str

CharacterConversationKey = ResolvedLocalCharacterKey | UnresolvedConversationKey

@dataclass(frozen=True)
class CharacterConversationCursor:
    last_modified: str
    conversation_id: str

@dataclass(frozen=True)
class CharacterConversationRow:
    row_key: str
    target: LocalCharacterConversationTarget | None
    unresolved: UnresolvedConversationKey | None
    unavailable_reason: UnavailableCharacterReason | None
    character_label: str
    title: str
    last_modified: str
    is_current: bool
    selected_excerpt: str

@dataclass(frozen=True)
class CharacterConversationGroup:
    key: CharacterConversationKey
    character_label: str
    rows: tuple[CharacterConversationRow, ...]
    total: int
    is_current: bool

@dataclass(frozen=True)
class EligibleConversationDocument:
    target: LocalCharacterConversationTarget
    title: str
    body: str
    source_revision: int
    eligibility_digest: str

@dataclass(frozen=True)
class CharacterConversationPage:
    rows: tuple[CharacterConversationRow, ...]
    total: int
    next_cursor: CharacterConversationCursor | None
    data_revision: int

class CharacterKeywordIndexStatus(StrEnum):
    ABSENT = "absent"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"

@dataclass(frozen=True)
class CharacterRepairCandidate:
    key: ResolvedLocalCharacterKey
    display_name: str
    version: int

@dataclass(frozen=True)
class CharacterRepairRequest:
    unresolved: UnresolvedConversationKey
    replacement: ResolvedLocalCharacterKey
    expected_conversation_version: int

class CharacterRepairResult(StrEnum):
    APPLIED = "applied"
    STALE_VERSION = "stale_version"
    NOT_FOUND = "not_found"
    INVALID_CANDIDATE = "invalid_candidate"

class CharacterConversationNavigationService:
    def recent_groups(self, *, group_limit: int = 4, row_limit: int = 5) -> tuple[CharacterConversationGroup, ...]: ...
    def keyword_search(self, query: str, *, offset: int = 0, limit: int = 50) -> CharacterConversationPage: ...
    def page_for_character(self, key: ResolvedLocalCharacterKey, *, cursor: CharacterConversationCursor | None = None, limit: int = 20) -> CharacterConversationPage: ...
    def repair_candidates(self, key: UnresolvedConversationKey) -> tuple[CharacterRepairCandidate, ...]: ...
    def repair(self, request: CharacterRepairRequest) -> CharacterRepairResult: ...
    def ensure_keyword_index(self) -> CharacterKeywordIndexStatus: ...
```

ADR required: no

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: this task directly implements ADR-116's already-decided identity,
storage, projection, and Keyword boundaries.

- [ ] **Step 1: Start TASK-31242 and record the current schema allocation**

Rebase after Task 1 merges, verify the current ChaChaNotes schema version, then
put the task in progress. The present plan advances v65 to v66; if `dev` has
advanced, append one new monotonic version rather than editing a shipped
migration. Record the final version in the Backlog implementation plan.

- [ ] **Step 2: Write failing identity and normalization tests**

In `Tests/Character_Chat/test_character_conversation_navigation.py`, add:

```python
def test_resolved_and_unresolved_row_keys_cannot_collide(): ...
def test_identity_rejects_blank_overlong_casefolded_or_path_derived_values(): ...
def test_unavailable_reason_changes_without_changing_unresolved_identity(): ...
def test_persona_rows_are_never_character_conversation_targets(): ...
```

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Character_Chat/test_character_conversation_navigation.py -v
```

Expected: collection fails because the new module and types do not exist.

- [ ] **Step 3: Implement the closed identity union and presentation models**

Implement the contracts above with validators that accept character IDs from 1
through `2**63 - 1` and nonblank authority/conversation IDs of at most 256 UTF-8
bytes. Serialization emits an explicit version and `resolved_local_character`
or `unresolved_conversation` tag; unknown versions and tags raise `ValueError`.

- [ ] **Step 4: Write failing selected-branch projector tests**

Seed one branched conversation with unique user, assistant, system, thinking,
tool, attachment, deleted, and non-selected canaries. Add:

```python
def test_projector_emits_only_selected_visible_user_and_assistant_path(): ...
def test_projector_fails_closed_for_cycle_dangling_parent_and_cross_conversation_parent(): ...
def test_projector_uses_unique_leaf_only_for_legacy_linear_conversation(): ...
def test_projector_digest_changes_when_selected_eligible_content_changes(): ...
```

Run the file and confirm all four tests fail before implementation.

- [ ] **Step 5: Implement one transaction-snapshot eligibility projector**

Add `SelectedBranchEligibilityProjector.project(conversation_id)` in
`character_conversation_search.py`. Traverse root-to-selected-leaf, include only
visible `user` and `assistant` bodies, join in deterministic message order, and
return `None` for any invalid or ambiguous graph. Both Keyword and later Meaning
must consume the returned `EligibleConversationDocument`; neither may re-query
message bodies independently.

- [ ] **Step 6: Write failing v66 migration and fresh-schema parity tests**

Assert a genuine current-version fixture upgrades once and fresh creation has
the same tables, columns, indexes, and triggers:

```text
character_conversation_search_documents
character_conversation_fts
character_conversation_search_generations
character_conversation_search_revision
```

The content table stores only selected eligible title/body material, authority,
conversation, character, digest, revision, and generation. The FTS table is a
separate external-content index; no test may find the new canaries in
`messages_fts` as evidence for this feature.

- [ ] **Step 7: Implement the additive migration and narrow DB entry points**

Wire one guarded migration in `ChaChaNotes_DB.py`, keep DDL in
`character_conversation_search.py`, and expose transaction-bound methods for
revision read/increment, generation state, projection pages, repair CAS, and
incremental document replacement. Do not run the backfill in the constructor.

- [ ] **Step 8: Write failing authority, ordering, paging, and repair tests**

Add tests named:

```python
def test_same_numeric_ids_in_two_authorities_never_merge(): ...
def test_recent_groups_force_current_then_sort_other_groups_by_latest_chat(): ...
def test_character_page_keyset_has_no_skip_or_repeat(): ...
def test_keyword_search_is_local_only_and_revalidates_data_revision(): ...
def test_unique_legacy_link_backfills_but_ambiguous_link_stays_unavailable(): ...
def test_repair_candidates_stay_in_authority_and_repair_uses_expected_version(): ...
```

Use real file-backed and in-memory SQLite fixtures; include server-shaped
canaries and assert they remain absent.

- [ ] **Step 9: Implement projection, deterministic backfill, and CAS repair**

Use section-first ordering so each character group remains contiguous. Keep
reason state outside identity. `repair()` must update only the exact unresolved
conversation under expected conversation version and authority, increment the
search revision in the same transaction, and return typed `APPLIED`,
`STALE_VERSION`, `NOT_FOUND`, or `INVALID_CANDIDATE`.

- [ ] **Step 10: Add and verify dormant Keyword index construction**

`ensure_keyword_index()` may synchronously report `ABSENT`, `BUILDING`,
`READY`, or `FAILED` and may schedule work only when explicitly called. Add an
architecture test proving app import and startup never call it. Backfill batches
128 conversations and emits progress after 128 records or one second.

- [ ] **Step 11: Run the Task 2 verification gate**

```bash
../../.venv/bin/python -m pytest \
  Tests/Character_Chat/test_character_conversation_navigation.py \
  Tests/DB/test_character_conversation_search_migration.py \
  Tests/DB/test_character_conversation_search_projection.py \
  Tests/DB/test_character_conversation_selected_branch_fts.py -v
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Character_Chat/character_conversation_navigation.py \
  tldw_chatbook/DB/character_conversation_search.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  Tests/Character_Chat/test_character_conversation_navigation.py \
  Tests/DB/test_character_conversation_search_migration.py \
  Tests/DB/test_character_conversation_search_projection.py \
  Tests/DB/test_character_conversation_selected_branch_fts.py
git diff --check
```

Expected: every targeted test and Ruff check passes; no UI file changes.

- [ ] **Step 12: Commit the projection slice**

```bash
git add tldw_chatbook/Character_Chat/character_conversation_navigation.py \
  tldw_chatbook/DB/character_conversation_search.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  Tests/Character_Chat/test_character_conversation_navigation.py \
  Tests/DB/test_character_conversation_search_migration.py \
  Tests/DB/test_character_conversation_search_projection.py \
  Tests/DB/test_character_conversation_selected_branch_fts.py \
  "backlog/tasks/task-31242 - Build-character-conversation-projection-and-Keyword-index.md"
git commit -m "feat: add character conversation keyword projection"
```

### Task 3: Add Trusted Navigation Recovery and Roleplay Browse — TASK-31243

**Files:**

- Create: `tldw_chatbook/Chat/console_conversation_activation.py`
- Create:
  `tldw_chatbook/UI/Navigation/character_conversation_navigation.py`
- Create:
  `tldw_chatbook/UI/Library_Modules/library_character_repair_controller.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify:
  `tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify:
  `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- Modify:
  `tldw_chatbook/Widgets/Persona_Widgets/personas_conversation_transcript_widget.py`
- Modify:
  `tldw_chatbook/UI/Library_Modules/library_conversations_controller.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/css/components/_workbench.tcss`
- Create: `Tests/Chat/test_console_conversation_activation.py`
- Create: `Tests/UI/test_character_conversation_navigation_payloads.py`
- Create: `Tests/UI/test_library_character_repair.py`
- Create: `Tests/UI/test_roleplay_character_conversation_browse.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Modify:
  `backlog/tasks/task-31243 - Add-trusted-character-navigation-recovery-and-Roleplay-browse.md`

**Interfaces:**

- Consumes: Task 2's typed targets, projection pages, repair contracts, and
  dormant Keyword-index adapter.
- Produces the only activation and navigation contracts later surfaces call:

```python
class ConsoleActivationPhase(StrEnum):
    IDLE = "idle"
    OPENING_CANCELLABLE = "opening_cancellable"
    COMMITTING = "committing"
    FAILURE_VISIBLE = "failure_visible"

class ConsoleActivationResultKind(StrEnum):
    OPENED = "opened"
    CANCELLED_PRECOMMIT = "cancelled_precommit"
    NOT_FOUND = "not_found"
    DATA_PROFILE_CHANGED = "data_profile_changed"
    CHARACTER_UNAVAILABLE = "character_unavailable"
    FAILED = "failed"

@dataclass(frozen=True)
class ConsoleConversationActivationResult:
    kind: ConsoleActivationResultKind
    target: LocalCharacterConversationTarget
    commit_started: bool

@dataclass(frozen=True)
class RoleplayReturnTarget:
    screen_id: str
    focus_id: str

@dataclass(frozen=True)
class RoleplayCharacterConversationLink:
    character: ResolvedLocalCharacterKey
    conversation_id: str | None = None
    query: str = ""
    return_target: RoleplayReturnTarget | None = None

@dataclass(frozen=True)
class LibraryCharacterRepairContext:
    unresolved: UnresolvedConversationKey
    expected_conversation_version: int
    historical_display_snapshot: str
    return_target: RoleplayReturnTarget

@dataclass(frozen=True)
class RoleplayDraftSnapshot:
    form_dirty: bool
    character_visual_dirty: bool
    persona_visual_dirty: bool
    attachments_dirty: bool
    inflight_save_domains: tuple[str, ...]
```

ADR required: no

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: Task 3 implements the ADR's previously decided activation, draft-veto,
repair, and surface-ownership contracts.

- [ ] **Step 1: Start TASK-31243 after Task 2 merges**

Rebase, confirm Task 2's exported names, put TASK-31243 in progress, and copy
this task's numbered steps into the task Implementation Plan. Do not change the
projection schema in this PR.

- [ ] **Step 2: Write failing navigation-payload validation tests**

Add tests proving payloads reject unknown versions, blank/overlong IDs,
nonlocal source tags, mismatched authority components, and invalid focus IDs.
Round trips must preserve the exact resolved or unresolved typed key without
carrying prompts, transcript bodies, credentials, card bodies, or filesystem
paths.

- [ ] **Step 3: Implement the versioned payload module and navigation keys**

Add constants for Roleplay character-conversation, Library repair, and return
focus contexts. Parse each once at the destination, clear it after acceptance,
and keep failed/declined navigation recoverable without replaying mutation.

- [ ] **Step 4: Write failing activation state-machine tests**

In `Tests/Chat/test_console_conversation_activation.py`, add:

```python
async def test_cancel_before_commit_changes_no_console_state(): ...
async def test_escape_after_commit_started_is_ignored(): ...
async def test_failed_postcommit_open_rolls_back_to_exact_prior_session(): ...
async def test_double_activate_shares_one_attempt_and_one_runtime_session(): ...
async def test_success_requires_exact_target_current_and_visible(): ...
async def test_profile_or_character_change_never_substitutes_a_target(): ...
```

Run this file and confirm failures precede production changes.

- [ ] **Step 5: Implement the Console-owned activation coordinator**

Wrap the incumbent `_resume_console_workspace_conversation()` hydration seam.
Capture the current session before work; revalidate authority/revision/card
before `commit_started`; make that assignment the linearization point; ignore
cancel afterward; and return `OPENED` only after store identity, mounted screen,
visible transcript, and composer focus all agree. Post-commit failure removes
only a new partial runtime session and repaints the exact prior session.

- [ ] **Step 6: Write failing aggregate Roleplay draft-veto tests**

Cover form edits, character visual changes, shared Persona visual changes,
attachments, and in-flight saves in one snapshot. Assert:

```python
async def test_save_and_continue_waits_until_every_domain_is_clean(): ...
async def test_partial_save_failure_preserves_failed_and_unsaved_domains(): ...
async def test_discard_names_and_clears_every_aggregate_draft_domain(): ...
async def test_stay_preserves_every_domain_and_focus(): ...
async def test_inflight_save_is_awaited_then_resnapshotted(): ...
```

- [ ] **Step 7: Implement the app-owned pre-navigation coordinator**

Define one `RoleplayDraftSnapshot` with explicit fields for the five domains.
Delegate save/discard to incumbent owners, collect domain-labeled failures, and
navigate only after a fresh snapshot is clean. The dialog copy lists affected
domains and exposes exactly `Save and continue`, `Discard and continue`, and
`Stay`.

- [ ] **Step 8: Write failing Library repair interaction tests**

Mount Library with a typed unresolved context and prove candidate rows come only
from the same data authority, no name match is preselected, old versus selected
identity is visible, confirmation is required, stale CAS focuses Refresh, and a
successful repair returns to the requested source anchor.

- [ ] **Step 9: Implement Library-owned repair presentation**

Keep Context, the switcher, and Roleplay read-only. The controller calls Task
2's `repair_candidates()` and `repair()`, maps typed failures to stable copy,
and invalidates Keyword/semantic candidates only after `APPLIED`. A cancelled
repair leaves both navigation context and conversation unchanged.

- [ ] **Step 10: Write failing Roleplay browse and compact-flow tests**

Add production-shaped tests for:

```text
Character list -> Card workspace -> Conversations -> Preview
deep link       -> Conversations with requested row focused
Back            -> exact prior pane and focus anchor
Enter preview   -> typed Console activation
View all        -> complete local keyset history, 20 rows per page
Keyword         -> selected-branch local corpus only
```

At 52×20, assert exactly one pane is visible; at wider sizes the incumbent
side-by-side layout may remain.

- [ ] **Step 11: Extend the incumbent Roleplay controller and widgets**

Replace direct character-ID listing with Task 2's exact key and page cursor.
Add query generation, requested-row focus, selected preview, `Back to Console`,
and stable reverse navigation. Preserve card editor, visual/attachment,
import/export, `Send transcript to Console draft`, and `Open in Library`
handlers as distinct actions.

- [ ] **Step 12: Build CSS and run the Task 3 gate**

```bash
../../.venv/bin/python -m tldw_chatbook.css.build_css
../../.venv/bin/python -m tldw_chatbook.css.check_bundle_sync
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_conversation_activation.py \
  Tests/UI/test_character_conversation_navigation_payloads.py \
  Tests/UI/test_library_character_repair.py \
  Tests/UI/test_roleplay_character_conversation_browse.py \
  Tests/UI/test_personas_workbench.py -v
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Chat/console_conversation_activation.py \
  tldw_chatbook/UI/Navigation/character_conversation_navigation.py \
  tldw_chatbook/UI/Library_Modules/library_character_repair_controller.py \
  tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py \
  Tests/Chat/test_console_conversation_activation.py \
  Tests/UI/test_character_conversation_navigation_payloads.py \
  Tests/UI/test_library_character_repair.py \
  Tests/UI/test_roleplay_character_conversation_browse.py
git diff --check
```

- [ ] **Step 13: Perform isolated real-TUI verification and commit**

Use scratch profile/config/data roots. At 52×20 and 120×50, walk deep link,
dirty draft Save/Discard/Stay, preview, exact resume, unavailable repair,
failure recovery, and Back focus. Record terminal cells and screenshots in the
task notes, then commit only Task 3 production, tests, CSS, docs, and task file
with message `feat: add trusted character conversation navigation`.

### Task 4: Add Character Conversations to Console Context — TASK-31244

**Files:**

- Create: `tldw_chatbook/Widgets/Console/console_character_context.py`
- Create: `tldw_chatbook/UI/Console_Modules/character_context.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/css/screen_agentic_console.tcss`
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md`
- Create: `Tests/UI/test_console_character_context.py`
- Create: `Tests/UI/test_console_character_context_geometry.py`
- Modify: `Tests/UI/test_console_context_rail_content.py`
- Modify:
  `backlog/tasks/task-31244 - Add-Character-conversations-to-Console-Context.md`

**Interfaces:**

- Consumes: Task 2 projection/Keyword search and Task 3 Roleplay, Library, and
  activation payloads.
- Produces `ConsoleCharacterContextState`,
  `ConsoleCharacterContextController.refresh()`, and a capability-gated query
  handoff seam used only after Task 5 installs the switcher mode.

```python
@dataclass(frozen=True)
class ConsoleCharacterContextState:
    groups: tuple[CharacterConversationGroup, ...]
    query: str
    search_rows: tuple[CharacterConversationRow, ...]
    expanded_key: CharacterConversationKey | None
    loading: bool
    error: str

class ConsoleCharacterContextController:
    async def refresh(self) -> None: ...
    async def search(self, query: str) -> None: ...
    async def activate(self, target: LocalCharacterConversationTarget) -> ConsoleConversationActivationResult: ...
    def open_roleplay(self, link: RoleplayCharacterConversationLink) -> None: ...
    def open_repair(self, context: LibraryCharacterRepairContext) -> None: ...
```

ADR required: no

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: the Context position, bounds, accordion behavior, and read-only
ownership are already decided by ADR-116 and amended ADR-083.

- [ ] **Step 1: Start TASK-31244 and pin the no-continuation boundary**

After Task 3 merges, put the task in progress and add an implementation-plan
line stating this PR must not render `Continue search in Character chats`.

- [ ] **Step 2: Write failing pure state tests**

Add tests for four-header selection, current-character inclusion with zero
chats, unavailable-slot consumption, five-row group caps, total counts,
date ordering, one-expanded-group invariant, and stable typed row keys.

- [ ] **Step 3: Implement the Context state/controller seam**

Fetch no more than four groups, five rows per group, and eight global Keyword
results. Fence async commits by controller generation and Data Profile revision.
Capture browse expanded key, focus ID, and scroll offset before search; restore
them when query clears or Escape exits search.

- [ ] **Step 4: Write failing disclosure-preference tests**

Cover these exact cases:

```text
new key absent + character context exists -> current/most-recent opens
explicit marker present                  -> stored Boolean wins
legacy Boolean without marker            -> preserve until manual toggle
responsive rail collapse                 -> never persists disclosure
no cards and no chats                     -> Character · No chats
```

- [ ] **Step 5: Implement versioned explicit disclosure persistence**

Persist `character_disclosure_explicit` plus the existing disclosure Boolean
through the canonical config writer only after a user toggle. Keep
`legacy-preserve` in memory until that first toggle; do not rewrite config on
read or responsive collapse.

- [ ] **Step 6: Write failing mounted hierarchy and interaction tests**

Assert Character is mounted immediately after Conversations and before Model
regardless of avatar setting. Verify `No character chats yet` plus `Open
Roleplay`, `View all N in Roleplay` for every nonempty resolved group, `Start in
Console` for a current zero-chat group, repair-only unavailable rows, global
search, exact Enter, pointer activation, loading, and failure recovery.

- [ ] **Step 7: Implement the always-composed Character widget**

Use native Textual buttons/list rows with one section scroll owner. Avatar
preference controls only image rendering. Cell-truncate labels without changing
accessible names. The collapsed summary is `Character · {name} · {N} chats` or
`Character · No chats`. Ordinary groups end in `View all N in Roleplay`; the
unavailable group is `Chats with unavailable characters` and ends in
`View all N in Library`. Search rows have no snippets. Route activation,
Roleplay, and repair through Task 3 typed contracts; do not call DB mutation
from the widget.

- [ ] **Step 8: Add source CSS and exact geometry assertions**

At 52×20 prove the rail fallback state does not claim the future switcher
capability. At 72×35, 80×24, and 120×50 prove children remain within the Context
pane, focus is visible, action labels are not clipped into ambiguity, and no
more than the bounded rows mount.

- [ ] **Step 9: Update Console guidance and run the Task 4 gate**

Document Character cards, saved character conversations, Console tabs, local-
only scope, View all, and unavailable recovery. Then run:

```bash
../../.venv/bin/python -m tldw_chatbook.css.build_css
../../.venv/bin/python -m tldw_chatbook.css.check_bundle_sync
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_character_context.py \
  Tests/UI/test_console_character_context_geometry.py \
  Tests/UI/test_console_context_rail_content.py -v
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/Console/console_character_context.py \
  tldw_chatbook/UI/Console_Modules/character_context.py \
  Tests/UI/test_console_character_context.py \
  Tests/UI/test_console_character_context_geometry.py
git diff --check
```

- [ ] **Step 10: Perform isolated Context walkthrough and commit**

Walk first-use, returning-user, empty, search/clear, unavailable, exact resume,
View all, avatar-off, and narrow-collapse states in the real TUI. Record cell
dimensions and focus outcomes, then commit only Task 4 files with message
`feat(console): add character conversation context`.

### Task 5: Add Character Chats to the Ctrl+K Switcher — TASK-31245

**Files:**

- Modify: `tldw_chatbook/Chat/console_switcher_state.py`
- Modify:
  `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/character_context.py`
- Modify: `tldw_chatbook/Widgets/Console/console_character_context.py`
- Modify: `tldw_chatbook/css/screen_agentic_console.tcss`
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md`
- Create: `Tests/UI/test_console_character_switcher.py`
- Create: `Tests/UI/test_console_character_switcher_geometry.py`
- Modify: `Tests/UI/test_console_session_switcher_trust.py`
- Modify: `Tests/UI/test_console_activity_switcher.py`
- Modify: `Tests/UI/test_console_modal_dismissal.py`
- Modify:
  `backlog/tasks/task-31245 - Add-Character-chats-mode-to-CtrlK-switcher.md`

**Interfaces:**

- Consumes: Task 2 Keyword pages, Task 3 activation, and Task 4's dormant
  Context query-handoff seam.
- Produces `SwitcherMode.CHARACTER_CHATS`, a Character query store independent
  from Active/History, and the complete narrow-terminal fallback. It does not
  expose Meaning.

```python
class SwitcherMode(str, Enum):
    ACTIVE = "active"
    HISTORY = "history"
    CHARACTER_CHATS = "character_chats"

@dataclass(frozen=True)
class ConsoleSwitcherCharacterResult:
    row_key: str
    target: LocalCharacterConversationTarget | None
    unresolved: UnresolvedConversationKey | None
    character_label: str
    title: str
    relative_time: str
    absolute_time: str
    selected_excerpt: str
```

ADR required: no

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: the third mode and trust invariants are already owned by ADR-116 and
amended ADR-085; ADR-031 continues to authorize F3 locally.

- [ ] **Step 1: Start TASK-31245 and capture incumbent trust tests**

Rebase after Task 4, put the task in progress, and run the existing switcher
trust, activity, and dismissal files before edits. Record their pass counts so
the third mode cannot weaken blank Enter, strict F2, stable pointer target,
selection movement, or safe dismissal behavior.

- [ ] **Step 2: Write failing mode and query-ownership tests**

Add:

```python
def test_f3_cycles_active_history_character_and_back(): ...
def test_active_and_history_share_query_but_character_query_is_independent(): ...
def test_active_zero_match_widens_with_explicit_history_label(): ...
def test_character_zero_match_never_widens(): ...
def test_character_f2_is_a_noop_with_truthful_hint(): ...
```

- [ ] **Step 3: Extend pure switcher state without semantic concepts**

Add the third enum member and result type. Preserve section-first ordering for
Active. Character rows sort by last activity descending with stable identity as
the final tie. Validate query through the existing length/control-character
boundary before calling Task 2.

- [ ] **Step 4: Write failing activation and pointer-stability tests**

Prove single pointer press captures its immutable row before await, Enter
freezes the committed highlight, double Enter shares one activation, Escape
cancels only precommit, profile/deletion failure leaves the prior tab active,
and the modal closes only after an `OPENED` result for the exact target.

- [ ] **Step 5: Wire the modal to Task 3 activation states**

Render `Opening…` while `OPENING_CANCELLABLE`, ignore further activation input
while in flight, switch to non-cancellable `COMMITTING` at the coordinator
callback, and leave typed failure visible with its prescribed Refresh, Library,
or return-to-profile action. Active native-tab choices retain their incumbent
path and MRU-other semantics.

- [ ] **Step 6: Write failing 52×20 grammar and focus tests**

Assert this exact vertical budget: top border, padding, mode row, search row,
scope row, divider, eight result rows representing four two-line results, two
selected-detail rows, action/paging row, hint/Cancel row, padding, bottom
border. Tab order is modes → search → results → actions → Cancel; initial focus
is search.

- [ ] **Step 7: Implement compact and wider switcher presentation**

At compact width, inline Active group status tokens rather than adding group
headers. In Character chats, show character/title and recency/state on each
two-line result; render excerpt and absolute timestamp only in the fixed
selected-detail region. Use Rich/Textual cell measurement for truncation. Keep
50 fetched rows per page and four visible rows at 52×20. The scope line reads
`This profile · Local chats` and never implies cached or remote coverage.

- [ ] **Step 8: Enable Context continuation only after capability exists**

Expose a capability method from `chat_screen.py`. Task 4's controller renders
`Continue search in Character chats` only when the mode is installed; transfer
the validated query and focus search. No Meaning label, model state, or semantic
action may appear in this PR.

- [ ] **Step 9: Build CSS and run the Task 5 gate**

```bash
../../.venv/bin/python -m tldw_chatbook.css.build_css
../../.venv/bin/python -m tldw_chatbook.css.check_bundle_sync
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_character_switcher.py \
  Tests/UI/test_console_character_switcher_geometry.py \
  Tests/UI/test_console_session_switcher_trust.py \
  Tests/UI/test_console_activity_switcher.py \
  Tests/UI/test_console_modal_dismissal.py -v
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Chat/console_switcher_state.py \
  tldw_chatbook/Widgets/Console/console_session_switcher_modal.py \
  Tests/UI/test_console_character_switcher.py \
  Tests/UI/test_console_character_switcher_geometry.py
git diff --check
```

- [ ] **Step 10: Run real keyboard/pointer verification and commit**

At 52×20, walk blank Active Enter, F3 cycle, separate queries, Active widening,
History paging, Character exact resume, unavailable recovery, pointer press,
pre/postcommit Escape, and Cancel. Repeat the target-identity checks at 120×50,
record evidence, then commit with message
`feat(console): add character chats to session switcher`.

### Task 6: Build the Opt-In Local Semantic Foundation — TASK-31246

**Files:**

- Create: `tldw_chatbook/RAG_Search/character_conversations/__init__.py`
- Create: `tldw_chatbook/RAG_Search/character_conversations/contracts.py`
- Create: `tldw_chatbook/RAG_Search/character_conversations/chunking.py`
- Create:
  `tldw_chatbook/RAG_Search/character_conversations/vector_store.py`
- Create:
  `tldw_chatbook/RAG_Search/character_conversations/index_service.py`
- Create:
  `tldw_chatbook/RAG_Search/character_conversations/query_service.py`
- Modify: `tldw_chatbook/DB/character_conversation_search.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create:
  `Tests/RAG_Search/character_conversations/test_character_semantic_contracts.py`
- Create:
  `Tests/RAG_Search/character_conversations/test_character_semantic_chunking.py`
- Create:
  `Tests/RAG_Search/character_conversations/test_character_semantic_vector_store.py`
- Create:
  `Tests/RAG_Search/character_conversations/test_character_semantic_index_service.py`
- Create:
  `Tests/RAG_Search/character_conversations/test_character_semantic_query_service.py`
- Modify:
  `backlog/tasks/task-31246 - Build-opt-in-local-character-conversation-semantic-index.md`

**Interfaces:**

- Consumes: Task 2's `EligibleConversationDocument`, data revision, and
  authoritative invalidation events.
- Produces a default-off backend for Task 7 and Task 8:

```python
class CharacterSemanticQueryStatus(StrEnum):
    RESULTS = "results"
    UNAVAILABLE = "unavailable"
    DAMAGED = "damaged"
    QUERY_ERROR = "query_error"

class CharacterSemanticJobState(StrEnum):
    ABSENT = "absent"
    WAITING_FOR_INITIAL_INDEX = "waiting_for_initial_index"
    BUILDING = "building"
    READY = "ready"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DAMAGED = "damaged"
    STORAGE_FULL = "storage_full"
    MODEL_UNAVAILABLE = "model_unavailable"

class CharacterSemanticMaintenanceAction(StrEnum):
    INDEX = "index"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    REBUILD = "rebuild"
    DELETE = "delete"

@dataclass(frozen=True)
class CharacterSemanticIndexConfig:
    model_id: str
    storage_path: str

@dataclass(frozen=True)
class CharacterEmbeddingChunk:
    chunk_id: str
    target: LocalCharacterConversationTarget
    source_revision: int
    eligibility_digest: str
    ordinal: int
    embedding: tuple[float, ...]

@dataclass(frozen=True)
class CharacterEmbeddingHit:
    chunk_id: str
    target: LocalCharacterConversationTarget
    source_revision: int
    eligibility_digest: str
    distance: float

@dataclass(frozen=True)
class CharacterSemanticConversationHit:
    target: LocalCharacterConversationTarget
    distance: float


@dataclass(frozen=True)
class CharacterSearchManifest:
    version: int
    data_authority_id: str
    generation_id: str
    model_id: str
    dimension: int
    normalized: bool
    chunk_policy_version: int
    eligibility_policy_version: int
    projection_version: int
    metric: Literal["cosine"]
    distance_semantics_version: int
    aggregation_version: int

@dataclass(frozen=True)
class CharacterSemanticQueryResult:
    status: CharacterSemanticQueryStatus
    rows: tuple[CharacterSemanticConversationHit, ...]
    generation_id: str | None
    message: str

@dataclass(frozen=True)
class CharacterSemanticIndexStatus:
    state: CharacterSemanticJobState
    indexed_conversations: int
    eligible_conversations: int
    active_job_id: str | None
    detail: str
    primary_action: CharacterSemanticMaintenanceAction

class CharacterConversationVectorStore:
    def replace_conversation(self, manifest: CharacterSearchManifest, chunks: Sequence[CharacterEmbeddingChunk]) -> None: ...
    def query(self, manifest: CharacterSearchManifest, embedding: Sequence[float], *, limit: int) -> tuple[CharacterEmbeddingHit, ...]: ...
    def delete_generation(self, manifest: CharacterSearchManifest) -> None: ...

class CharacterConversationSemanticIndex:
    async def index_existing(self, config: CharacterSemanticIndexConfig) -> str: ...
    async def rebuild(self, config: CharacterSemanticIndexConfig) -> str: ...
    async def pause(self, job_id: str) -> None: ...
    async def resume(self, job_id: str) -> None: ...
    async def cancel(self, job_id: str) -> None: ...
    async def delete_all(self, data_authority_id: str) -> None: ...
    async def reconcile(self, data_authority_id: str) -> CharacterSemanticIndexStatus: ...
```

ADR required: no

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: Task 6 implements the ADR's derived-index, privacy, generation, and
direct-query choices. Its additive semantic state follows amended ADR-030.

- [ ] **Step 1: Start TASK-31246 and allocate the additive schema version**

Rebase after Task 5, inspect current schema, and append exactly one migration
for durable semantic manifests, jobs, outbox, and ready fences. The present
sequence advances v66 to v67; never edit a version already present on `dev`.

- [ ] **Step 2: Write failing manifest and outcome tests**

Assert exact serialization, rejection of unknown versions/metrics/dimensions,
and the distinction between `RESULTS(())`, `UNAVAILABLE`, `DAMAGED`, and
`QUERY_ERROR`. Compatibility must include every manifest field shown above.

- [ ] **Step 3: Implement contracts and deterministic chunking**

Chunk only Task 2's eligible `body`, carry target, source revision, eligibility
digest, ordinal, and content hash, and never carry excluded roles or attachment
text. Make chunk boundaries deterministic for the same manifest and input. Add
tests for Unicode, long messages, empty eligible body, and bounded chunk count.

- [ ] **Step 4: Write failing no-plaintext vector-store tests**

Use a fake Chroma collection that records every argument. Prove
`replace_conversation()` supplies `ids`, `embeddings`, and safe metadata but
never `documents`; metadata must not contain title, excerpt, message body,
prompt, tool payload, or attachment text. Assert collection identity includes
data authority and generation and pins `hnsw:space=cosine`.

- [ ] **Step 5: Implement the narrow Chroma adapter**

Reuse existing Chroma client construction and naming utilities only. Do not
subclass or call the generic document-oriented `VectorStore`. Validate vector
dimension and finite values at the boundary; map collection absence,
incompatible metadata, corruption, and query exceptions to distinct typed
errors.

- [ ] **Step 6: Write failing atomic-generation and outbox tests**

Add real-SQLite tests for initial build, rebuild, partial failure, pause,
cancel, restart/resume, storage-full, model removal, delete, and reconciliation.
Inject failure after replacement chunks but before ready-fence advance and
prove the whole conversation is suppressed until idempotent replay completes.
Rebuild failure must leave the prior generation queryable.

- [ ] **Step 7: Implement durable jobs, per-conversation fences, and cutover**

Authoritative conversation transactions increment
`character_search_revision` and append an outbox event. The worker writes a
complete replacement, verifies count/digest/revision, advances that
conversation's ready fence, then removes old chunks. Initial/rebuild jobs mark a
new generation ready only after every included conversation is fenced; cutover
is one SQLite transaction. Candidate suppression happens immediately from
authoritative revision state, before vector cleanup.

- [ ] **Step 8: Write failing direct ANN query tests**

Use a deterministic embedding fake and assert: one query embedding, first 200
chunks, at most one refill to 400, maximum 50 conversations, lowest raw cosine
distance per conversation, and stable target tie-break. Include a semantic
canary sharing no lexical token with its query to prove there is no Keyword
prefilter or lexical rerank.

- [ ] **Step 9: Implement local-provider validation and query aggregation**

Reject remote provider kinds and models requiring a network fetch. Read only
the ready compatible manifest, query the vector store, filter every hit through
current authority/revision/digest fences, aggregate by minimum raw distance,
and join titles/excerpts from the authoritative projection only after ranking.
Any backend failure returns a non-RESULTS status.

- [ ] **Step 10: Prove default-off and unreachable production wiring**

Add an import/startup architecture test showing no screen imports
`index_service` or `query_service`, no constructor opens Chroma, no background
job starts, and no user-facing control references Meaning. Cold index/query
tests replace network transports with fail-on-call fakes.

- [ ] **Step 11: Run privacy, lifecycle, and performance gates**

```bash
../../.venv/bin/python -m pytest \
  Tests/RAG_Search/character_conversations/test_character_semantic_contracts.py \
  Tests/RAG_Search/character_conversations/test_character_semantic_chunking.py \
  Tests/RAG_Search/character_conversations/test_character_semantic_vector_store.py \
  Tests/RAG_Search/character_conversations/test_character_semantic_index_service.py \
  Tests/RAG_Search/character_conversations/test_character_semantic_query_service.py -v
../../.venv/bin/python -m ruff check \
  tldw_chatbook/RAG_Search/character_conversations \
  Tests/RAG_Search/character_conversations
git diff --check
```

Add a 10k-conversation/250k-message indexing fixture and record batch size 128,
progress cadence, event-loop slice, wall time, and extra RSS excluding loaded
model/native cache. Extra RSS must remain below 256 MiB.

- [ ] **Step 12: Commit the unreachable semantic backend**

Commit only semantic backend, DB migration/wiring, targeted tests, ADR/task
references, and task notes with message
`feat(rag): add local character conversation semantic index`. Verify no Settings,
Roleplay, Context, or switcher file changed.

### Task 7: Add Settings Lifecycle and Roleplay Meaning — TASK-31247

**Files:**

- Create: `tldw_chatbook/RAG_Search/character_conversations/settings.py`
- Create:
  `tldw_chatbook/Widgets/Settings_Widgets/character_chat_search_panel.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py`
- Modify:
  `tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py`
- Modify:
  `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/css/screen_agentic_settings.tcss`
- Modify: `tldw_chatbook/css/components/_workbench.tcss`
- Modify: `Docs/User_Guide/settings/rag.md`
- Modify:
  `Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md`
- Create: `Tests/RAG_Search/character_conversations/test_character_semantic_settings.py`
- Create: `Tests/UI/test_settings_character_chat_search.py`
- Create: `Tests/UI/test_settings_character_chat_search_geometry.py`
- Create: `Tests/UI/test_roleplay_character_meaning_search.py`
- Modify:
  `backlog/tasks/task-31247 - Add-Character-chat-search-controls-and-Roleplay-Meaning.md`

**Interfaces:**

- Consumes: Task 6 lifecycle/index/query services and Task 3 Roleplay browser.
- Produces saved semantic configuration, the canonical Settings control surface,
  and the first reachable end-to-end Meaning search in Roleplay.

```python
@dataclass(frozen=True)
class CharacterSemanticSavedConfig:
    enabled_for_future: bool
    model_id: str
    storage_path: str

    def to_index_config(self) -> CharacterSemanticIndexConfig: ...

@dataclass(frozen=True)
class CharacterSemanticSettingsDraft:
    enabled_for_future: bool
    model_id: str
    storage_path: str

```

ADR required: no

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: Task 7 implements ADR-116's consent/lifecycle UI while preserving
ADR-033's staged Save/Revert versus immediate reviewed-action model.

- [ ] **Step 1: Start TASK-31247 and baseline Settings commit-model tests**

Rebase after Task 6, put the task in progress, and run the existing Settings
draft/save/revert, Library RAG, and narrow-layout tests. Record baseline pass
counts before mounting the new panel.

- [ ] **Step 2: Write failing staged-versus-immediate settings tests**

Add tests proving model and `Keep future chats indexed` are staged, Save is
authoritative, Revert affects only staged fields, and maintenance actions always
use saved configuration. Index/Rebuild/Delete are disabled while relevant
fields differ from saved values; Pause/Resume/Cancel stay active for the current
saved-config job and ignore the draft.

- [ ] **Step 3: Implement saved config and draft adapters**

Validate local model IDs and confined storage paths without importing the model
or opening Chroma. Return immutable saved and draft objects. Save writes both
fields atomically through the canonical Settings writer; failure leaves the
previous saved object and the visible draft intact.

- [ ] **Step 4: Write failing lifecycle presentation tests**

Map every Task 6 state to exact status, one contextual primary action, and
recovery:

```text
ABSENT                    -> Index existing chats
WAITING_FOR_INITIAL_INDEX -> waiting copy; Index existing when saved config valid
BUILDING                  -> Pause or Cancel
READY                     -> Rebuild or Delete index
PAUSED                    -> Resume or Cancel
CANCELLED / FAILED        -> Retry or Delete index
DAMAGED                   -> Rebuild or Delete index
STORAGE_FULL              -> Free space, Retry, or Delete index
MODEL_UNAVAILABLE         -> Choose installed local model, then Save
```

Delete must be blocked by a dirty relevant draft. Successful Delete disables
the saved future-index preference and refreshes original plus draft values.

- [ ] **Step 5: Build the canonical Settings panel**

Mount `Character chat search` inside existing `SettingsCategoryId.LIBRARY_RAG`
under the visible RAG destination; keep stable category ID `library-rag`. Show
summary and `Local only · Nothing is uploaded` before advanced details. Use one
vertical scroll owner at 52×20. Keep Library RAG backfill labels, accelerators,
and handlers separate.

- [ ] **Step 6: Wire immediate maintenance actions with reviewed labels**

Buttons call Task 6 through a single-flight worker. Before Index, Rebuild, or
Delete, show ADR-033's immediate-action review language and the saved config
being used. Pause/Resume/Cancel act on the visible `active_job_id`. Busy state
appears within 100 ms; late job callbacks are fenced by screen instance, data
authority, and job ID.

- [ ] **Step 7: Write failing Roleplay Meaning behavior tests**

Add:

```python
async def test_roleplay_keyword_and_meaning_are_focusable_distinct_strategies(): ...
async def test_meaning_unavailable_routes_to_settings_without_enabling_it(): ...
async def test_meaning_returns_semantic_canary_without_shared_query_token(): ...
async def test_meaning_uses_selected_user_assistant_branch_only(): ...
async def test_meaning_failure_preserves_keyword_results_and_selection(): ...
async def test_profile_change_discards_late_meaning_generation(): ...
```

- [ ] **Step 8: Add the first reachable Meaning slice to Roleplay**

Show `Keyword` and `Meaning` as focusable strategy controls in the Conversations
pane. Meaning is enabled only for a compatible ready local generation. An
unavailable control stays focusable, explains the state in the selected detail
line, and routes to Settings. Query through Task 6; render authoritative titles
and selected preview only after current-revision revalidation.

- [ ] **Step 9: Add 52×20 and accessibility coverage**

Prove the Settings screen has one scroll owner, status precedes actions, the
primary action is reachable by keyboard and pointer, disabled reasons are
visible, advanced details start collapsed, and Save/Revert remain unambiguous.
In Roleplay, prove strategy, search, results, preview, and Back remain reachable
in the one-pane progression with selected-result detail announced once.

- [ ] **Step 10: Update Settings and Roleplay guidance**

Document local-only privacy, selected-branch exclusions, the difference between
Index existing and Keep future, all lifecycle controls, storage deletion,
Keyword versus Meaning, unavailable-model routing, and the fact that Library
RAG backfill is unrelated.

- [ ] **Step 11: Build CSS and run the Task 7 gate**

```bash
../../.venv/bin/python -m tldw_chatbook.css.build_css
../../.venv/bin/python -m tldw_chatbook.css.check_bundle_sync
../../.venv/bin/python -m pytest \
  Tests/RAG_Search/character_conversations/test_character_semantic_settings.py \
  Tests/UI/test_settings_character_chat_search.py \
  Tests/UI/test_settings_character_chat_search_geometry.py \
  Tests/UI/test_roleplay_character_meaning_search.py \
  Tests/UI/test_settings_panel_scoped_updates.py \
  Tests/UI/test_settings_narrow_layout.py -v
../../.venv/bin/python -m ruff check \
  tldw_chatbook/RAG_Search/character_conversations/settings.py \
  tldw_chatbook/Widgets/Settings_Widgets/character_chat_search_panel.py \
  Tests/RAG_Search/character_conversations/test_character_semantic_settings.py \
  Tests/UI/test_settings_character_chat_search.py \
  Tests/UI/test_settings_character_chat_search_geometry.py \
  Tests/UI/test_roleplay_character_meaning_search.py
git diff --check
```

- [ ] **Step 12: Run isolated lifecycle/Roleplay UAT and commit**

With a disposable profile and real installed local model, walk absent → index →
pause → resume → ready → Roleplay Meaning → rebuild failure with prior ready →
delete. Repeat dirty Save/Revert/maintenance combinations at 52×20 and one
standard size. Verify no network call and no real-profile mutation, record
evidence, then commit with message
`feat: add character chat semantic controls and Roleplay search`.

### Task 8: Integrate Ctrl+K Meaning and Qualify the Programme — TASK-31248

**Files:**

- Create:
  `tldw_chatbook/UI/Console_Modules/character_switcher_search.py`
- Modify:
  `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Chat/console_switcher_state.py`
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md`
- Modify: `Docs/User_Guide/settings/rag.md`
- Create: `Tests/UI/test_console_character_meaning_search.py`
- Create: `Tests/UI/test_character_navigation_integrated.py`
- Create: `Tests/Benchmarks/test_character_conversation_search_benchmark.py`
- Create:
  `Docs/superpowers/qa/character-conversation-navigation/README.md`
- Modify:
  `backlog/tasks/task-31248 - Integrate-Meaning-into-CtrlK-and-harden-character-navigation.md`

**Interfaces:**

- Consumes: the complete Tasks 1–7 stack.
- Produces the final two-leg switcher coordinator and integrated evidence:

```python
class CharacterSearchStrategy(StrEnum):
    KEYWORD = "keyword"
    MEANING = "meaning"

@dataclass(frozen=True)
class CharacterSwitcherSearchSnapshot:
    query: str
    strategy: CharacterSearchStrategy
    rows: tuple[ConsoleSwitcherCharacterResult, ...]
    painted_source: CharacterSearchStrategy | None
    meaning_results_pending_apply: bool
    selected_row_key: str | None
    status: str

class CharacterSwitcherSearchCoordinator:
    async def search(self, query: str, *, generation: int) -> None: ...
    def apply_meaning_results(self) -> CharacterSwitcherSearchSnapshot: ...
    def cancel(self) -> None: ...
```

ADR required: no

ADR path:
`backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`

Reason: Task 8 integrates the already-approved two-leg presentation and closes
cross-surface evidence without adding a new architecture boundary.

- [ ] **Step 1: Start TASK-31248 and baseline the full reachable feature stack**

Rebase after Task 7, put the task in progress, and run the complete targeted
suites from Tasks 2–7 once. Record failures attributable to current `dev`
separately; do not weaken or delete an incumbent assertion to make this task
green.

- [ ] **Step 2: Write failing two-leg state-table tests**

Use a controllable clock and futures to cover every approved ordering:

```text
Keyword running / Meaning ready
Meaning unavailable before search
Keyword nonempty before 120 ms, vector nonempty before gate
Keyword nonempty at gate, vector still running
Vector nonempty before Keyword
Vector nonempty after Keyword painted
Vector zero after Keyword painted
Vector failure after Keyword painted
Keyword zero while vector still healthy
profile/query/modal generation changes during either leg
```

Assert only one list paints automatically and losing legs never merge.

- [ ] **Step 3: Implement the 120 ms coordinator**

Start Keyword and vector tasks together. Before first paint, accept a valid
vector result immediately; hold a nonempty Keyword result until 120 ms; at the
gate paint the best valid available leg. After paint, never change row order
automatically. Store late nonempty vector rows behind
`meaning_results_pending_apply`; zero or failed Meaning updates status only.
Fence every callback by modal instance, query, Data Profile revision, and search
generation.

- [ ] **Step 4: Write failing selection-preservation and focus tests**

Prove `Apply Meaning results` retains the selected target when present, chooses
the first row only when the old target is absent, keeps keyboard focus on the
result/detail region, and never activates a row until a new explicit Enter or
pointer press. The action has visible copy and no F4 binding.

- [ ] **Step 5: Mount Keyword and Meaning strategy controls in Ctrl+K**

Insert focusable `Keyword` and `Meaning` controls on the Character scope row.
Unavailable Meaning remains focusable and routes to
Settings > RAG > Character chat search without enabling or downloading. Add
`Apply Meaning results` only when pending. Keep tab order modes → search →
strategy → results → actions → Cancel and retain the 52×20 total row budget.

- [ ] **Step 6: Write integrated race and trust tests**

Drive Context query transfer, Ctrl+K Meaning, Roleplay View all/Meaning,
Unavailable repair, dirty-draft veto, exact Console activation, index delete,
conversation delete, character unlink, and Data Profile switch. Every race must
either open the exact original target or leave the previous trustworthy state;
none may substitute a card/conversation or display stale/mixed semantic rows.

- [ ] **Step 7: Build the fixed benchmark corpus and query set**

Create a versioned fixture generator for 10,000 conversations and 250,000
messages with 30 query IDs: eight title Keyword, eight body Keyword, eight
semantic without shared token, three no-match, and three Unicode/long cases.
Run five warmups and ten recorded repetitions per query, 300 measurements total.
Compute P95 by nearest rank and emit hardware, OS, Python, SQLite, model,
manifest digest, corpus digest, and raw timings.

- [ ] **Step 8: Enforce latency and correctness benchmark gates**

`Tests/Benchmarks/test_character_conversation_search_benchmark.py` must assert
Keyword P95 ≤300 ms and Meaning P95 ≤2 s on the documented qualification host.
Every query also asserts expected identities, no server/cached canaries, no
excluded-role canaries, deterministic ties, and semantic-only direct retrieval.
Performance failure is a task failure, not a threshold edit.

- [ ] **Step 9: Run integrated 52×20 and equal-cell terminal verification**

Use production CSS and disposable profile state. Record Textual cell geometry
at 52×20, 72×35, 80×24, and 120×50. Run equal row/column scenarios in iTerm2
and Windows Terminal per the incumbent terminal-evidence protocol. Reject
captures with client/tmux clipping and inspect every visible descendant against
its pane, not only top-level shell widths.

- [ ] **Step 10: Run the moderated first-use check**

With at least three participants unfamiliar with the feature, record whether at
least two independently find and resume an existing character chat within two
minutes, explain Character card versus saved conversation versus open Console
tab, and recover from one unavailable row. Any failed criterion returns its
owning UI task to revision before Task 8 completes.

- [ ] **Step 11: Finish user guidance and reproducible QA evidence**

Update Console and Settings docs with Meaning readiness, the 120 ms behavior,
late-result apply action, local-only corpus, exact resume, repair, and privacy.
Write `Docs/superpowers/qa/character-conversation-navigation/README.md` with
fixture commands, scratch paths, app revision, model/manifest digest, terminal
cells, results, screenshots, benchmark raw-data path, and first-use protocol.

- [ ] **Step 12: Run the final targeted programme gate**

Run every new test file from Tasks 2–8 plus the incumbent switcher trust,
activity, modal dismissal, Context rail, Personas workbench, Settings commit-
model, narrow-layout, migration, CSS-bundle, and startup-import gates touched by
the sequence. Run Ruff across all changed Python/test paths and
`git diff --check`. Ask the user before expanding this to the full repository
suite.

- [ ] **Step 13: Rebase, requalify, self-review, and commit**

Rebase onto current `origin/dev`; rerun generated artifacts and every affected
targeted gate because startup and diagnostic manifests may change without a
direct file conflict. Compare every spec acceptance item to a passing test or
recorded UAT artifact. Commit with message
`feat(console): integrate character conversation meaning search`.

## Spec Coverage Map

| Approved design area | Owning task | Primary evidence |
| --- | --- | --- |
| Local-only scope, terminology, ownership, ADR alignment | 1 | ADR/reference checks |
| Typed identity, unavailable classification, selected-branch eligibility, Keyword FTS, paging, repair CAS | 2 | Real-SQLite migration/projection/eligibility tests |
| Cancellable exact activation, aggregate draft veto, Library repair, complete Roleplay browse | 3 | Race tests, mounted interaction tests, isolated TUI |
| Four-header/five-row Context accordion, disclosure migration, global Keyword, Roleplay links | 4 | Pure state, mounted hierarchy, geometry, first-use walkthrough |
| Active/History/Character chats switchboard, MRU-other trust, compact grammar | 5 | Incumbent trust regressions, new mode tests, real keyboard/pointer pass |
| Local embeddings-only generations, outbox fences, atomic cutover, bounded direct ANN | 6 | Privacy, crash, lifecycle, direct-semantic, memory tests |
| Consent and maintenance controls, Settings commit model, first Roleplay Meaning slice | 7 | Settings state matrix, lifecycle UAT, Roleplay semantic tests |
| 120 ms first paint, explicit late apply, integrated races, benchmarks, terminal and moderated evidence | 8 | Deterministic clock tests, 10k/250k benchmark, integrated UAT |

## Plan Self-Review Checklist

- [x] Every approved acceptance item maps to at least one numbered task and a
  named automated or live evidence step.
- [x] Every later interface name exactly matches the producer declaration in an
  earlier task.
- [x] Each Backlog task depends only on the immediately preceding, already-
  created task.
- [x] No PR exposes a UI capability before its backing service and recovery
  path exist.
- [x] Context stays Keyword-only through Task 8.
- [x] Task 6 remains default-off and unreachable until Task 7.
- [x] Task 7 is the first end-to-end Meaning slice; Task 8 only integrates the
  switcher and programme evidence.
- [x] No step authorizes server/cached-server scope, remote embeddings,
  plaintext vector documents, lexical semantic reranking, name-based repair,
  or fallback target activation.
- [x] Every schema change appends after the latest shipped version and includes
  immediately-preceding real-fixture plus fresh-schema parity tests.
- [x] Every UI task builds and checks generated CSS and performs isolated
  production-styled verification at 52×20.
- [x] Each task adds its Backlog Implementation Notes, checks all acceptance
  criteria, records ADR status, and moves to Done only after its own Definition
  of Done is satisfied.
