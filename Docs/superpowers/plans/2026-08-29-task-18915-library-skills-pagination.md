# Library Skills Pagination and Trust Recovery Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every local Skill reachable through exact 20-row Library pages while preserving literal pre-slice filtering, deterministic trust sorting, and truthful source-wide trust recovery.

**Architecture:** Keep paging state owned by the Skills source. Extend `LocalSkillsService.list_skills` into the exact local page boundary, add immutable validation/display contracts beside the existing Skills pure state, and add a dedicated non-visual Skills browse controller patterned after the Prompt controller. The Library screen wires that controller to the retained Skills canvas; the broad local-source snapshot remains available to rail/landing consumers but can no longer overwrite Skills canvas rows. Reuse `library_pager_state.py` for all pager copy and disabled-state derivation.

**Tech Stack:** Python 3.11+, Textual 8.x, Pydantic response schemas, pytest/pytest-asyncio, Ruff.

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/067-library-top-level-pagination-contracts.md`

**Reason:** This task implements the accepted Skills tranche without changing data ownership, storage, security authority, or cross-module architecture.

---

## Task 1: Pin the exact local Skills page contract

**Files:**

- Modify: `Tests/Skills/test_local_skills_service.py`
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `tldw_chatbook/tldw_api/skills_schemas.py`

- [ ] Add failing service tests with more than 40 Skills proving literal case-insensitive name/description filtering occurs before the 20-row slice, body/supporting-file/argument/metadata/trust-only markers never match, and the filtered `total`, `limit`, `offset`, and `count` are exact.
- [ ] Add failing status/name sort tests proving deterministic normalized-name tie-breaking and trust-blocked metadata survives in each summary.
- [ ] Add failing source-wide trust tests proving `blocked_total` and `first_blocked_skill_name` are computed from the complete classified index and do not change with page or filter.
- [ ] Extend the response schema with the two source-wide trust fields and add validated `query`/`sort` inputs to `LocalSkillsService.list_skills`; classify the complete index once, filter/sort summaries, then slice.
- [ ] Run the new service tests and existing `Tests/Skills/test_local_skills_service.py` file.

## Task 2: Add fail-closed pure Skills page state

**Files:**

- Modify: `Tests/Library/test_library_skills_state.py`
- Modify: `tldw_chatbook/Library/library_skills_state.py`

- [ ] Add failing tests for a normalized local-only `SkillBrowseScope`, request fingerprint/token matching, exact coordinate/cardinality validation, stable unique Skill identities, source-wide blocked metadata, clamped-page semantics, and late-result rejection.
- [ ] Add failing tests for fresh, loading, first-load error, page-failure, scope-failure, and stale retained-row pager states using `build_library_pager_display`.
- [ ] Implement immutable browse scope/result/page validation and extend `SkillsListState` to consume one already-filtered page rather than re-filtering broad snapshots.
- [ ] Make malformed rows, duplicated/blank identities, contradictory totals, coordinates, trust metadata, and non-JSON-like retained values fail closed.
- [ ] Run the Skills pure-state tests and shared pager tests.

## Task 3: Add the source-owned Skills browse controller

**Files:**

- Create: `tldw_chatbook/UI/Library_Modules/library_skills_browse_controller.py`
- Modify: `tldw_chatbook/UI/Library_Modules/__init__.py`
- Create: `Tests/UI/test_library_skills_browse_controller.py`

- [ ] Add failing controller tests for service signature/explicit local mode, generations, page and scope failure copy, one automatic out-of-range clamp, retry, unmount fencing, stale retained rows, and late-result rejection.
- [ ] Implement a dedicated controller that owns requested/applied scope, retained page rows, freshness, exact totals, source-wide trust recovery metadata, and pager display state.
- [ ] Preserve only last successfully applied scope for restoration; exclude failed drafts, loading state, errors, rows, and requested pages that never applied.
- [ ] Run the controller and pure-state suites.

## Task 4: Wire the retained Skills canvas and pager

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py`
- Modify: `Tests/UI/test_library_skills_canvas.py`
- Modify: `Tests/UI/test_library_skills_reader.py`

- [ ] Immediately before these UI edits, load Impeccable's `reference/craft-floor.md` and preserve the existing terminal-native density, focus grammar, and independent Items/Work layout.
- [ ] Add failing mounted tests for exact title/range/page copy, in-pane Previous/Next/Try again controls, visible disabled reasons, filter-focus restoration, sort/filter resetting to page 1, page-control focus, row 20 reachability, and pager containment at 100x30 and 170x48.
- [ ] Wire controller construction, service dispatch, filter/sort/page/retry actions, active-route checks, navigation invalidation, and applied-scope restoration into `LibraryScreen`.
- [ ] Render only the controller's bounded page in `LibrarySkillsListCanvas`; keep the pager inside the list pane with a `1fr`/`min-height: 0` scroll viewport and disable row/pager actions whenever state is stale or loading.
- [ ] Replace the current page-local trust header/count and Review target with controller-owned source-wide `blocked_total` and `first_blocked_skill_name`, opening the stable name directly even when it is off-page.
- [ ] Prove the legacy broad local-source snapshot can still refresh rail/landing consumers without replacing applied Skills rows.

## Task 5: Make mutations truthful under refresh failure

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_skills_canvas.py`
- Modify: `Tests/UI/test_library_skills_reader.py`

- [ ] Add failing tests for create/import/save/delete/trust mutations invalidating the active read and refreshing the full applied scope.
- [ ] Add failing committed-mutation/failed-refresh tests proving the known mutation is reconciled locally, exact totals disappear, all row/pager actions are disabled with visible stale reasons, and Try again or a new scope request restores fresh authority.
- [ ] Route successful mutations through the controller refresh; on refresh failure retain only validated locally reconciled rows and a committed-but-stale explanation.
- [ ] Run the focused mutation, trust, canvas, and reader suites.

## Task 6: Verify races, geometry, and isolated live behavior

**Files:**

- Create or modify the smallest existing TASK-1891x live/geometry harness under `Tests/UI/` or `scripts/` after inspecting the incumbent harness.
- Modify: `backlog/tasks/task-18915 - Page-Library-Skills-with-source-wide-trust-recovery.md`
- Modify only if a generalizable incident occurs: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] Run mounted race tests proving an older generation and a post-unmount completion cannot patch the current/freshly re-entered screen.
- [ ] Run 100x30 and 170x48 geometry tests with at least 45 Skills, covering page 1, page 2, final page, filter, focus, source-wide off-page Review, loading/error/retry, and stale recovery.
- [ ] Mutation-check the source-wide trust guard by temporarily deriving blocked count/Review target from the visible page, prove the off-page trust test fails, restore immediately, and rerun green.
- [ ] Mutation-check stale action safety by temporarily enabling one stale row action, prove the safety test fails, restore immediately, and rerun green.
- [ ] Run the isolated live TUI harness with its scratch profile/config/data paths created before imports; verify resolved paths/open handles are scratch-owned and real config/data manifests remain byte-identical.
- [ ] Run focused Ruff, the targeted owner suites, Impeccable's mechanical detector once over the changed UI targets, `git diff --check`, and a final diff review. Do not run the full repository suite without explicit user opt-in.
- [ ] Check every acceptance criterion, add concise Implementation Notes and verification evidence, and move TASK-18915 to Done only after all Definition-of-Done requirements are satisfied.
