# Console `/rewind` Summarize-from-here Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add branch-safe manual prefix and inclusive range memory to Console
`/rewind`, with one durable effective-memory selector, exact provider request
projection, derived-only resume-safe banners, and no transcript mutation.

**Architecture:** Extend schema v54 additively with memory-scope and
append-mostly branch-selection tables. Keep persistence in
`console_context_repository.py`, pure unit/planning/selection/projection rules
in `console_context_compaction.py`, provider serialization in
`console_prepared_request.py`, admission/orchestration in
`console_chat_controller.py`, and presentation in the existing Console
widgets. Both manual directions share one bounded one-call service and exact
CAS commit path; generated memory replaces the effective branch selection
without deactivating sibling state or rewriting the legacy compatibility pair.

**Tech Stack:** Python 3.11+, SQLite/FTS5 migrations, Textual 8.x, pytest,
pytest-asyncio, Ruff.

**Parent task:** [TASK-575](../../../backlog/tasks/task-575%20-%20Console-rewind-add-a-Summarize-from-here-complement-to-Summarize-up-to-here.md)

**Spec:** [Console `/rewind` Summarize-from-here design](../specs/2026-08-28-console-rewind-summarize-from-here-design.md)

**ADR required:** yes

**ADR path:** [ADR-052](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md)

**Reason:** The task changes durable memory scope, branch selection, atomic
replacement, provider-context projection, and the long-lived `/rewind` UX.
ADR-052 was amended and independently re-reviewed before this plan.

## Global constraints

- Preserve the transcript and branch tree byte-for-byte. Memory, selection,
  and banners are derived state only.
- Keep both new tables local-only: no sync columns, sync triggers, payloads,
  or server DTOs.
- The current implementation baseline is schema v54 on `origin/dev`
  `3a3383123e`; use v55 unless a newer schema lands before implementation, in
  which case rebase first and mechanically advance every migration reference.
- Use one normative complete-unit helper everywhere. Never round, skip, or
  partially summarize an incomplete unit.
- A manual action makes exactly zero or one auxiliary provider call. It does
  not retry, recursively summarize, fold old memory into its input, or trim a
  raw span to make it fit.
- Generated memory is an app-owned semantic segment. Support only
  distinct-role and single-preamble provider mappings; never fall back to an
  ordinary user row.
- Every async result must pass runtime revalidation and the exact SQLite CAS
  transaction before it can become selected.
- Do not log transcript or summary bodies. The auxiliary ledger remains
  content-free for success, failure, cancellation, and stale outcomes.
- Apply test-driven development within each task: add the focused failing
  test, run it and observe the intended failure, make the smallest production
  change, then rerun the focused test before the task commit.
- Use targeted test files during implementation. The repository requires
  asking the user before a full-suite sweep.

---

## Task 1: Add v55 local memory-scope and branch-selection schema

**Files:**

- Create: `tldw_chatbook/DB/migrations/chachanotes_v54_to_v55_console_memory_scope_selection.sql`
- Create: `Tests/DB/test_chachanotes_v55_console_memory_selection_migration.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/DB/sql_validation.py`
- Verify: `Tests/DB/test_schema_table_allowlist_guard.py`

- [ ] Write migration tests first for a fresh database and an upgraded v54
  fixture. Assert schema version 55, the explicit unique parent key on
  `(id, conversation_id)`, both new tables, their checks/FKs/indexes, and no
  sync-log or server-facing artifacts.
- [ ] Add a v54-stamped partial/re-entry fixture and prove the guarded runner
  reaches v55 without duplicate rows, reordered backfill, or partial state.
- [ ] Add upgrade fixtures containing active/inactive generated memories,
  valid/invalid captured leaves, and a legacy conversation summary. Assert
  deterministic scope backfill (`prefix`, `automatic`, no selection anchor),
  non-suppressing select-event backfill in original memory insertion order,
  and inert records when no valid activation anchor exists.
- [ ] Add constraints tests proving: automatic scope is prefix with a null
  anchor; manual scope has an anchor; select has a memory; reset has no
  memory; sequence is database-owned and monotonic; activation, selected
  memory, and scope pairs cannot cross conversations.
- [ ] Add deletion tests proving referenced message/memory hard deletion is
  restrictive, while whole-conversation deletion cascades scope and selection
  rows. Ordinary message soft deletion remains the supported path.
- [ ] Add a migration-integrity test that injects one FK violation and proves
  `PRAGMA foreign_key_check` is fetched, the migration raises, the transaction
  rolls back, and the v55 stamp is not written.
- [ ] Run the new test file and confirm it fails because v55 does not exist:

  ```bash
  ../../.venv/bin/python -m pytest Tests/DB/test_chachanotes_v55_console_memory_selection_migration.py -q
  ```

- [ ] Implement the additive schema with these durable shapes:

  ```text
  console_conversation_memory_scopes(
      memory_id, conversation_id, coverage_kind, origin_kind,
      selection_anchor_message_id
  )
  console_conversation_memory_selections(
      sequence INTEGER PRIMARY KEY AUTOINCREMENT,
      selection_id UNIQUE, conversation_id, activation_message_id,
      selected_memory_id NULLABLE, event_kind, suppresses_legacy,
      created_at, revision, active
  )
  ```

  Use composite same-conversation foreign keys, restrictive message/memory
  deletion, and conversation-level cascade exactly as specified.
- [ ] Add `_migrate_from_v54_to_v55`, update `_CURRENT_SCHEMA_VERSION` and the
  migration dispatch table, run fetched `PRAGMA foreign_key_check` before the
  version stamp, and preserve atomic rollback behavior.
- [ ] Add both table names to the `chachanotes` `VALID_TABLES` allowlist.
- [ ] Run:

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/DB/test_chachanotes_v55_console_memory_selection_migration.py \
    Tests/DB/test_chachanotes_console_context_memory_migration.py \
    Tests/DB/test_schema_table_allowlist_guard.py -q
  ../../.venv/bin/python scripts/check_schema_table_allowlist.py
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/DB Tests/DB
  git commit -m "feat(db): add branch-scoped console memory selection"
  ```

---

## Task 2: Model scope, selection events, and one effective-memory decision

**Files:**

- Modify: `tldw_chatbook/Chat/console_context_repository.py`
- Modify: `tldw_chatbook/Chat/console_context_compaction.py`
- Create: `Tests/Chat/test_console_memory_selection.py`
- Modify: `Tests/DB/test_chachanotes_console_context_memory_migration.py`

- [ ] Write pure selector tests for prefix/range scope validation, newest
  branch-valid event by descending database sequence, sibling-event skipping,
  reset terminality, invalid selected-memory fail-open to raw, and no fallback
  to an older generated selection.
- [ ] Cover legacy precedence explicitly: valid legacy wins when the branch
  head is absent/non-suppressing; a suppressing manual/reset head overrides
  it; an invalid/off-lineage legacy pair falls through to the generated head.
- [ ] Run the selector tests and confirm imports/contracts are missing:

  ```bash
  ../../.venv/bin/python -m pytest Tests/Chat/test_console_memory_selection.py -q
  ```

- [ ] Add validated string enums and immutable records in
  `console_context_repository.py`:

  ```python
  class MemoryCoverageKind(str, Enum):
      PREFIX = "prefix"
      RANGE = "range"

  class MemoryOriginKind(str, Enum):
      AUTOMATIC = "automatic"
      MANUAL_REWIND = "manual_rewind"

  class MemorySelectionKind(str, Enum):
      SELECT = "select"
      RESET = "reset"

  @dataclass(frozen=True, slots=True)
  class ConsoleMemoryScopeRecord: ...

  @dataclass(frozen=True, slots=True)
  class ConsoleMemorySelectionRecord: ...
  ```

  Keep summary text excluded from repr, validate positive revisions/sequences,
  and reject contradictory scope/event combinations at the Python boundary.
- [ ] Add bounded repository readers for one scope and newest-first active
  selection candidates. Decode corrupt derived rows as ineligible rather than
  guessing or raising into request dispatch.
- [ ] Add repository round-trip tests for prefix/range scopes and select/reset
  events, including an assertion that these writes add no `sync_log` rows.
- [ ] Replace `select_valid_memory(...)` with one pure
  `select_effective_memory(...)` decision that receives active durable
  snapshots, decoded selection candidates/scopes, and an explicitly validated
  legacy snapshot or no-legacy sentinel. Return a typed result distinguishing
  `raw`, `legacy_prefix`, `generated_prefix`, and `generated_range`, together
  with the applicable branch head/fence.
- [ ] Keep the existing conservative prefix digest for both scopes. Validate
  prefix boundaries and both range anchors against the active lineage before
  returning generated memory.
- [ ] Run:

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/Chat/test_console_memory_selection.py \
    Tests/Chat/test_console_context_compaction.py \
    Tests/DB/test_chachanotes_console_context_memory_migration.py -q
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_context_repository.py \
    tldw_chatbook/Chat/console_context_compaction.py \
    Tests/Chat/test_console_memory_selection.py \
    Tests/DB/test_chachanotes_console_context_memory_migration.py
  git commit -m "feat(console): select effective branch memory"
  ```

---

## Task 3: Make selection, reset, and undo exact atomic transactions

**Files:**

- Modify: `tldw_chatbook/Chat/console_context_repository.py`
- Modify: `Tests/Chat/test_console_memory_selection.py`
- Modify: `Tests/DB/test_chachanotes_console_context_memory_migration.py`

- [ ] Add failing repository tests for two simultaneous no-memory jobs (only
  one wins), exact generated-head revision mismatch, exact legacy
  boundary/summary-digest mismatch, changed cursor/leaf/message parent,
  changed version/deletion/variant/attachment digest, and unrelated sibling
  events not invalidating the captured branch.
- [ ] Add reset tests proving current reset appends a suppressing tombstone,
  generated memories stay immutable/available, legacy is not cleared, and
  the returned token names the exact tombstone ID/revision.
- [ ] Add undo tests proving only the current applicable tombstone at the
  expected revision is deactivated; a later select/reset expires the token.
  Add Reset-all tests proving the legacy pair is cleared and all selection
  events/memory rows are deactivated with revision bumps.
- [ ] Run the tests and observe the old record-global active-bit behavior fail:

  ```bash
  ../../.venv/bin/python -m pytest Tests/Chat/test_console_memory_selection.py -q
  ```

- [ ] Introduce immutable transaction inputs rather than a loose parameter
  list:

  ```python
  @dataclass(frozen=True, slots=True)
  class MemorySelectionFence:
      effective_kind: str
      legacy_boundary_message_id: str | None
      legacy_summary_digest: str | None
      selection_sequence: int | None
      selection_id: str | None
      selection_revision: int | None
      memory_id: str | None
      memory_revision: int | None

  @dataclass(frozen=True, slots=True)
  class PersistedLineageFenceRow:
      message_id: str
      parent_message_id: str | None
      version: int
      deleted: bool
      content_digest: str
      selected_variant_id: str | None
      selected_variant_index: int | None
      attachment_digests: tuple[str, ...]

  @dataclass(frozen=True, slots=True)
  class BranchMemoryCommit:
      memory: ConsoleMemoryRecord
      scope: ConsoleMemoryScopeRecord
      selection: ConsoleMemorySelectionRecord
      expected_effective: MemorySelectionFence
      expected_branch_head: MemorySelectionFence
      expected_cursor: tuple[str, str | None]
      durable_lineage: tuple[PersistedLineageFenceRow, ...]
  ```

  Keep the persistence fence type in the repository module so it does not
  import the compaction module and create a circular dependency.

- [ ] Implement `commit_memory_selection_if_current(commit) -> bool` as one
  non-awaiting SQLite transaction. Rebuild the active lineage from the
  persisted cursor, compare every durable fact and the exact applicable head,
  compare legacy only when admitted, then insert memory, scope, and selection
  atomically. Never deactivate the previous branch/sibling memory.
- [ ] Replace record-global current reset/undo with
  `append_current_branch_reset_if_current(...)` and
  `undo_current_branch_reset_if_current(...)`. Retain `active` only for coarse
  availability and Reset-all compatibility.
- [ ] Make automatic selection inherit the applicable head's
  `suppresses_legacy` bit (default false); force manual select/reset true.
- [ ] Run the repository/migration focused files from Tasks 1-3.
- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_context_repository.py \
    Tests/Chat/test_console_memory_selection.py \
    Tests/DB/test_chachanotes_console_context_memory_migration.py
  git commit -m "feat(console): commit branch memory with exact CAS"
  ```

---

## Task 4: Unify complete durable units and manual prefix/range planning

**Files:**

- Modify: `tldw_chatbook/Chat/console_context_compaction.py`
- Modify: `tldw_chatbook/Chat/console_prepared_request.py`
- Create: `Tests/Chat/test_console_manual_memory_planning.py`
- Modify: `Tests/Chat/test_console_context_compaction.py`
- Modify: `Tests/Chat/test_console_prepared_request.py`

- [ ] Add table-driven failing tests for the normative complete-unit helper:
  persisted user start, positive versions, provider-visible rows, complete
  assistant/tool groups, and terminal assistant `status="complete"`. Assert
  refusal for unanswered, generating/stopped/failed, orphan/partial tool,
  deleted, ephemeral, system, and missing-version candidates with no rounding.
- [ ] Extend `DurableMessageSnapshot` only with durable facts needed by that
  predicate/CAS (parent, status, deletion, provider visibility, selected
  variant and attachment digests). Include those facts in its digest and
  content-free provenance payload.
- [ ] Replace `compactable_units_after`'s permissive role grouping with one
  `complete_durable_units(...)` result used by manual and automatic planners.
  Keep leading system/identity and seeded greeting outside the unit set.
- [ ] Add failing prefix-plan tests for every complete unit strictly before
  the selected prompt and range-plan tests for the inclusive selected prompt
  through the complete current assistant leaf. Assert stable raw role/tool
  envelopes, exact anchors/boundary, no prior memory in auxiliary input, and
  zero-call reasons for invalid/incomplete anchors or oversized input.
- [ ] Add one `ManualMemoryPlan` with `coverage_kind`, raw selected units,
  boundary/start anchors, immutable auxiliary messages, output cap, canonical
  before/after projections, token counts, and provenance. Expose two thin
  constructors, `plan_manual_prefix(...)` and `plan_manual_range(...)`, over
  the same core.
- [ ] Add the fixed app-owned idle request sentinel to
  `console_prepared_request.py`. Build both manual comparison artifacts with
  the exact current system/identity plus that sentinel and
  `apply_safety_window=False`: before is all authoritative raw history with no
  old memory; after applies only the candidate memory and exact retained raw
  rows.
- [ ] Enforce progress only when after < before, covered raw savings exceed
  wrapper/body cost, and after fits safe provider capacity. Keep wrapper tags
  and transcript data in separate immutable prompt/data envelopes.
- [ ] Run:

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/Chat/test_console_manual_memory_planning.py \
    Tests/Chat/test_console_context_compaction.py \
    Tests/Chat/test_console_prepared_request.py -q
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_context_compaction.py \
    tldw_chatbook/Chat/console_prepared_request.py \
    Tests/Chat/test_console_manual_memory_planning.py \
    Tests/Chat/test_console_context_compaction.py \
    Tests/Chat/test_console_prepared_request.py
  git commit -m "feat(console): plan exact manual memory ranges"
  ```

---

## Task 5: Execute both manual directions through one bounded service

**Files:**

- Modify: `tldw_chatbook/Chat/console_context_compaction.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Replace/extend: `Tests/Chat/test_console_rewind_summarize.py`
- Modify: `Tests/Chat/test_console_context_compaction.py`

- [ ] Rewrite the old rolling-summary expectations as failing parity tests:
  up-to uses only authoritative complete raw units, never folds legacy/prior
  memory, never truncates to the old 12k span, and commits a manual prefix
  scope/selection without writing the legacy pair.
- [ ] Add from-here service tests for exact inclusive range, one provider call,
  empty/envelope/over-cap/non-improving output rejection, cancellation ledger,
  and every stale admission fence. Assert old memory stays effective during
  the call and no partial database writes occur.
- [ ] Run:

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/Chat/test_console_rewind_summarize.py \
    Tests/Chat/test_console_context_compaction.py -q
  ```

- [ ] Add a `summarize_manual(...)` entry point to
  `ConsoleCompactionService` that accepts a precomputed manual plan and
  `BranchMemoryCommit` admission. It must start one content-free auxiliary
  ledger entry, dispatch exactly once, validate reserved tags/envelopes and
  output cap, re-evaluate canonical progress, run the supplied runtime fence,
  and finish success/failed/cancelled/stale without logging content.
- [ ] Refactor `ConsoleChatController.summarize_up_to(message_id)` onto the
  shared durable snapshot/planner/service path and add
  `summarize_from(message_id)`. Capture native/persisted start/end IDs,
  session, payload/identity/policy/provider/model/prompt, effective selection,
  legacy digest, cursor, lineage, versions, variants, attachments, and leaf
  before the call; recheck runtime facts immediately before the repository
  transaction.
- [ ] Preserve the editable `console.rewind_summarize` prompt as the only
  prompt and route provider work through the existing gateway with ordinary
  tools/sources/skills/world-info disabled.
- [ ] Return stable controller results/copy for provider-not-ready, busy,
  invalid start/end, no complete units, too large for one call,
  non-improving, failed, cancelled, and stale cases.
- [ ] Run the focused files again and commit:

  ```bash
  git add tldw_chatbook/Chat/console_context_compaction.py \
    tldw_chatbook/Chat/console_chat_controller.py \
    Tests/Chat/test_console_rewind_summarize.py \
    Tests/Chat/test_console_context_compaction.py
  git commit -m "feat(console): summarize rewind scopes through one service"
  ```

---

## Task 6: Project effective prefix/range memory into every request path

**Files:**

- Modify: `tldw_chatbook/Chat/console_context_compaction.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_prepared_request.py`
- Modify: `Tests/Chat/test_console_rewind_summarize.py`
- Modify: `Tests/Chat/test_console_prepared_request.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`

- [ ] Add failing pure projection tests for valid prefix and inclusive range
  scopes, missing/reversed/off-lineage/cross-conversation anchors, and the leak
  boundary. For a range, retain leading system/early raw rows, remove start
  through end, retain later durable rows/active request, and attach exactly one
  app-owned memory segment.
- [ ] Add retry/regenerate/continue/edit tests before, inside, at, and after a
  range. Payloads lacking the end boundary must remain byte-identical raw and
  must not receive future memory.
- [ ] Add direct and agent provider tests for distinct-role and
  single-preamble adapters, exact preview/dispatch artifact identity, private
  anchor stripping, and removal of thinking/continuation/attachment/tool
  sidecars owned only by removed rows. Assert no ordinary-user fallback.
- [ ] Run the tests and observe the current prefix-only projection fail.
- [ ] Add one pure `project_effective_memory(...)` function in
  `console_context_compaction.py`. It consumes annotated provider rows plus a
  validated effective-memory result and returns either the exact transformed
  rows/app-memory segment or the original raw input/no-memory result.
- [ ] Update `_apply_conversation_memory_preflight` and snapshot/preview paths
  to read repository candidates, validate legacy once, select effective state
  once, and use that immutable projection for both estimation and dispatch.
  Remove the independent legacy and generated-summary transforms so memory is
  never layered twice.
- [ ] While legacy is effective, make Ask/Automatic/Compact-now zero-call
  ineligible; keep deterministic safety/failure policy and recovery copy.
- [ ] Run:

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/Chat/test_console_rewind_summarize.py \
    Tests/Chat/test_console_prepared_request.py \
    Tests/Chat/test_console_provider_gateway.py \
    Tests/Chat/test_console_context_compaction.py -q
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat Tests/Chat
  git commit -m "feat(console): project branch memory without leaks"
  ```

---

## Task 7: Add ordered range-to-prefix automatic compaction and lifecycle controls

**Files:**

- Modify: `tldw_chatbook/Chat/console_context_compaction.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Widgets/Console/console_context_controls.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `Tests/Chat/test_console_context_compaction.py`
- Modify: `Tests/UI/test_console_context_controls.py`
- Modify: `Tests/Chat/test_console_session_settings.py`

- [ ] Add failing planner tests where effective memory is a range: mandatory
  complete early units first, one sealed prior-memory provenance unit second,
  then the largest consecutive eligible later prefix. Assert the prior body is
  present once in the auxiliary envelope but absent from
  `selected_units_json`, which contains only a content-free provenance marker.
- [ ] Test indivisibility and zero-call behavior when early units plus sealed
  memory do not fit, carry-forward handling for later units, no later-unit
  case using the range end boundary, and success producing an ordinary
  automatic prefix with inherited legacy suppression.
- [ ] Implement a distinct range-to-prefix branch in `plan_compaction`; do not
  reuse normal wire preamble order as the summarizer chronology. Preserve
  exact current-effective versus candidate-prefix progress accounting.
- [ ] Replace Settings/current-memory state with the typed effective-memory
  result so generated prefix/range and legacy manual prefix are labeled
  honestly. Show manual range start/end and unavailable legacy provenance
  without inventing provider/model data.
- [ ] Wire current reset, exact undo token, and separately confirmed Reset all
  to the Task 3 repository transactions. Ensure current reset suppresses
  legacy only on descendants, while Reset all clears legacy globally and has
  no undo.
- [ ] Run:

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/Chat/test_console_context_compaction.py \
    Tests/UI/test_console_context_controls.py \
    Tests/Chat/test_console_session_settings.py -q
  ```

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat tldw_chatbook/Widgets/Console \
    Tests/Chat Tests/UI/test_console_context_controls.py
  git commit -m "feat(console): compact and manage range memory"
  ```

---

## Task 8: Add the `/rewind` choice and guarded Textual worker flow

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_rewind_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Chat/test_console_rewind_modal.py`
- Modify: `Tests/UI/test_console_rewind_restore.py`

- [ ] Add failing modal tests for stable action IDs and exact order:
  `Restore to here`, `Summarize up to here`, `Summarize from here`, `Never
  mind`. Assert the new choice preserves the selected prompt ID/text and that
  cancel/focus behavior is unchanged. Both summary actions show `Uses the
  active model once`; when memory is effective they also show `Replaces
  current conversation memory`.
- [ ] Add failing screen tests proving both summary choices use exclusive
  workers, refuse while sending/streaming/compacting, guard the captured
  session/selection against later changes, preserve composer/draft state, and
  show actionable too-large/stale/error copy without transcript mutation.
  Mount at 80 columns as well as the normal harness width and assert the full
  action/cost copy remains keyboard-reachable without forbidden bindings.
- [ ] Run:

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/Chat/test_console_rewind_modal.py \
    Tests/UI/test_console_rewind_restore.py -q
  ```

- [ ] Add `KIND_SUMMARIZE_FROM = "summarize-from"` and its stable button ID,
  retaining the existing frozen `ConsoleRewindChoice` contract.
- [ ] Route `_apply_console_rewind_choice` to symmetric
  `_summarize_console_up_to` / `_summarize_console_from` exclusive workers.
  Reuse the existing refusal gate and controller result handling; always
  refocus safely when the modal/worker exits. Use `Summarizing selected
  range...` while the new worker runs and `Conversation memory updated.` on
  success.
- [ ] Run the focused modal/screen tests again and commit:

  ```bash
  git add tldw_chatbook/Widgets/Console/console_rewind_modal.py \
    tldw_chatbook/UI/Screens/chat_screen.py \
    Tests/Chat/test_console_rewind_modal.py \
    Tests/UI/test_console_rewind_restore.py
  git commit -m "feat(console): add summarize-from-here rewind action"
  ```

---

## Task 9: Render one derived scope banner and restore it on resume

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify: `Tests/UI/test_console_resume_active_path.py`
- Modify: `Tests/UI/test_console_context_controls.py`

- [ ] Add failing transcript tests for one immutable presentation value with
  prefix/range kind, render anchor, start/end IDs, and copy. Prefix renders
  above the selected prompt; range renders above its inclusive start. Missing
  anchors render no banner rather than guessing.
- [ ] Assert plain export, persisted transcript rows, branch tree, message
  count, and message IDs are unchanged. Switching sessions/branches replaces
  or clears the banner without rebuilding unrelated rows.
- [ ] Add restart/resume tests for generated prefix, generated range, legacy
  prefix, sibling branch selection, reset tombstone, corrupt scope, and
  dangling anchors. Effective selection after resume must match send-time
  projection and expose only one banner.
- [ ] Run:

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/UI/test_console_native_transcript.py \
    Tests/UI/test_console_resume_active_path.py \
    Tests/UI/test_console_context_controls.py -q
  ```

- [ ] Replace `summary_boundary_message_id`/`set_summary_boundary` with a
  frozen `ConsoleMemoryBannerPresentation | None` and one setter. Render copy:
  keep `⤵ Earlier turns summarized for context — full history above` for
  prefix; render `Context uses a summary of turns #N-#M - full transcript
  remains visible.` for a manual range, where N/M are user-turn ordinals
  derived from the validated active lineage rather than database IDs.
- [ ] During transcript sync/resume, resolve the same effective-memory result
  used by dispatch and derive the presentation; never persist banner rows or
  infer scope from old fields.
- [ ] Run the focused UI files again and commit:

  ```bash
  git add tldw_chatbook/Widgets/Console/console_transcript.py \
    tldw_chatbook/UI/Screens/chat_screen.py \
    Tests/UI/test_console_native_transcript.py \
    Tests/UI/test_console_resume_active_path.py \
    Tests/UI/test_console_context_controls.py
  git commit -m "feat(console): restore derived memory scope banners"
  ```

---

## Task 10: Focused integration, static checks, documentation, and task closeout

**Files:**

- Modify: `backlog/tasks/task-575 - Console-rewind-add-a-Summarize-from-here-complement-to-Summarize-up-to-here.md`
- Modify if an incident generalizes: `backlog/docs/lessons-testing-evidence.md`
- Modify if an incident generalizes: `backlog/docs/lessons-live-verification.md`

- [ ] Rebase on the latest `origin/dev`, inspect schema-version drift and
  resolve it before final verification. Confirm the working tree contains only
  TASK-575 changes.
- [ ] Run the focused cross-layer suite (not the full repository suite):

  ```bash
  ../../.venv/bin/python -m pytest \
    Tests/DB/test_chachanotes_v55_console_memory_selection_migration.py \
    Tests/DB/test_chachanotes_console_context_memory_migration.py \
    Tests/DB/test_schema_table_allowlist_guard.py \
    Tests/Chat/test_console_memory_selection.py \
    Tests/Chat/test_console_manual_memory_planning.py \
    Tests/Chat/test_console_context_compaction.py \
    Tests/Chat/test_console_prepared_request.py \
    Tests/Chat/test_console_provider_gateway.py \
    Tests/Chat/test_console_rewind_modal.py \
    Tests/Chat/test_console_rewind_summarize.py \
    Tests/Chat/test_console_session_settings.py \
    Tests/UI/test_console_rewind_restore.py \
    Tests/UI/test_console_context_controls.py \
    Tests/UI/test_console_native_transcript.py \
    Tests/UI/test_console_resume_active_path.py -q
  ```

- [ ] Run static and migration guards on the changed production/test files:

  ```bash
  ../../.venv/bin/python -m ruff check \
    tldw_chatbook/Chat/console_context_repository.py \
    tldw_chatbook/Chat/console_context_compaction.py \
    tldw_chatbook/Chat/console_prepared_request.py \
    tldw_chatbook/Chat/console_chat_controller.py \
    tldw_chatbook/Widgets/Console/console_context_controls.py \
    tldw_chatbook/Widgets/Console/console_settings_modal.py \
    tldw_chatbook/Widgets/Console/console_rewind_modal.py \
    tldw_chatbook/Widgets/Console/console_transcript.py \
    tldw_chatbook/UI/Screens/chat_screen.py \
    Tests/DB/test_chachanotes_v55_console_memory_selection_migration.py \
    Tests/DB/test_chachanotes_console_context_memory_migration.py \
    Tests/Chat/test_console_memory_selection.py \
    Tests/Chat/test_console_manual_memory_planning.py \
    Tests/Chat/test_console_context_compaction.py \
    Tests/Chat/test_console_prepared_request.py \
    Tests/Chat/test_console_provider_gateway.py \
    Tests/Chat/test_console_rewind_modal.py \
    Tests/Chat/test_console_rewind_summarize.py \
    Tests/Chat/test_console_session_settings.py \
    Tests/UI/test_console_rewind_restore.py \
    Tests/UI/test_console_context_controls.py \
    Tests/UI/test_console_native_transcript.py \
    Tests/UI/test_console_resume_active_path.py
  ../../.venv/bin/python scripts/check_schema_table_allowlist.py
  git diff --check
  ```

- [ ] Create an isolated scratch `TLDW_CONFIG_PATH` whose `[paths].data_dir` is
  also a scratch directory, print and verify the exact SQLite path before
  launch, and never open the schema-bumping branch against the shared
  development database.
- [ ] In that isolated mounted Console, test narrow and normal widths: create
  a branched durable chat, run each manual direction through one auxiliary
  completion, inspect the next-send provider payload, close/resume, verify the
  restored banner/payload, regenerate before/inside/after a range, and exercise
  reset/undo/reset-all. Capture exactly one memory segment and no private IDs.
  Record unavailable live-provider evidence honestly; do not substitute UI
  screenshots for provider-boundary evidence.
- [ ] Self-review against every verification bullet in the approved spec,
  ADR-052, keybinding conventions, security/privacy rules, and these global
  constraints. Search for placeholders and old split-brain paths:

  ```bash
  rg -n "TODO|FIXME|NotImplemented|summary_boundary_message_id|select_valid_memory" \
    tldw_chatbook Tests
  git diff origin/dev...HEAD --stat
  git diff origin/dev...HEAD --check
  ```

- [ ] Ask the user whether they want the full pytest suite. Do not run it
  without that opt-in.
- [ ] Check all TASK-575 acceptance criteria, add concise Implementation Notes
  with exact test/live evidence and ADR-052 link, and mark the task Done only
  when every Definition-of-Done item is satisfied:

  ```bash
  backlog task edit 575 -s Done --notes "Implemented branch-safe manual prefix/range memory; see implementation notes for evidence."
  ```

- [ ] Commit closeout documentation:

  ```bash
  git add 'backlog/tasks/task-575 - Console-rewind-add-a-Summarize-from-here-complement-to-Summarize-up-to-here.md'
  # If a genuine incident required a lesson, add only that exact lesson file.
  git commit -m "docs(console): close TASK-575 summarize-from-here"
  ```

## Required review gates

- Migration review: additive v55 shape, deterministic backfill, fetched FK
  audit, restrictive individual deletes, local-only table census, and
  conversation cascade.
- Transaction review: exact no-head/generated/legacy fences, persisted lineage
  reconstruction, branch-local event ordering, sibling independence, and
  reset/undo expiry.
- Provider review: one-call manual bound, canonical idle progress, no prior
  memory in manual input, range-to-prefix chronology, two supported wire
  contracts, raw fail-open, and no private anchor leakage.
- UI review: exact `/rewind` order/copy, busy/stale/failure behavior, one
  derived banner, resume/branch correctness, focus, and narrow geometry.
- Privacy review: no transcript/summary content in repr, logs, errors, or the
  auxiliary ledger.
- Completion review: focused automated evidence, static checks, optional full
  suite only with user approval, task notes/AC/ADR hygiene, and a clean branch.
