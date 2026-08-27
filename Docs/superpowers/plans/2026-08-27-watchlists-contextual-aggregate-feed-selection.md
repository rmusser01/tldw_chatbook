# Watchlists Contextual Aggregate Feed Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Apply superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before claiming success.

**Goal:** Let users expand All Sources, Unassigned, and All Unread and select an individual feed while preserving the selected parent context, Reader query identity, and manual pane/filter preferences.

**Architecture:** Extend the existing immutable `TreeScope` with an explicit source-parent context, keep root and watchlist expansion as separate screen-owned session state, and make the already-atomic tree snapshot own complete all/unassigned source rows plus existing bulk per-source counts. The tree remains a pure navigation renderer; the screen owns async scope commit, Reader predicates, filter parking, server-backend availability, and reconciliation.

**Tech Stack:** Python 3.11+, Textual 8.x reactives/messages, existing `WatchlistBundleService`, pytest/pytest-asyncio, Ruff.

**ADR required:** no  
**ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`  
**Reason:** ADR-042 already defines the long-lived Reader-first navigation, stable snapshot, and atomic scope-commit boundaries this task extends. The change adds contextual occurrences within those boundaries without changing storage, service ownership, or runtime contracts.

---

### Task 1: Repair the inherited Reader test contract and establish a clean focused baseline

**Files:**
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`

**Step 1: Update the two stale recovery test doubles**

Change only the two inherited tests that still stub/assert the retired `list_items()` path so they return and assert `list_reader_items_page()` with a `WatchlistItemPage`. Do not add production compatibility for the retired API.

**Step 2: Preserve and classify the pre-repair evidence**

The identical focused command was already run before any test repair: 317 tests passed and only `test_server_backed_read_recovers_through_the_normal_local_load_path` plus `test_failed_switch_to_local_retries_the_normal_load_path` failed because their doubles returned an `AsyncMock` instead of `WatchlistItemPage`. Keep this evidence in the task notes; do not describe the post-repair run as the original baseline.

**Step 3: Run the focused post-repair selection**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/Subscriptions/test_watchlist_normalizers.py \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlist_tree.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  --tb=short
```

Expected: PASS; record that the only prior failures were stale test doubles inherited from PR #2113.

**Step 4: Commit**

```bash
git add Tests/Watchlists/test_watchlists_collections_screen.py
git commit -m "test: align Watchlists recovery with typed reader pages"
```

### Task 2: Define contextual source scopes and independent aggregate expansion

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`
- Modify: `Tests/Watchlists/test_watchlist_tree.py`

**Step 1: Write failing scope and rendering tests**

Add tests proving:

- `TreeScope(kind="source")` carries `parent_context` of `all`, `unassigned`, `unread`, or `watchlist`.
- All Sources, Unassigned, and All Unread each have a separate caret target and expansion state.
- Aggregate children use occurrence-qualified widget ids and post the exact contextual scope.
- All Unread renders only positive-count sources, except for the selected/pending zero-count pin.
- A source selected below one occurrence does not mark another occurrence active.
- Remove-from-watchlist is enabled only for a `parent_context="watchlist"` source.
- Aggregate and watchlist children are case-insensitively sorted by name with source id as the stable tie-breaker.
- Clicking only a caret changes expansion; clicking the adjacent label changes scope without toggling expansion.
- Scope commit and recompose retain focus on the activated caret, label, or feed-child control when that exact occurrence remains mounted.
- Expanded empty branches reuse the existing All Sources no-library state (`No Watchlists sources yet.`) and render the new exact contextual rows `No unassigned feeds` and `No unread feeds`.

Run each new nodeid immediately after writing it and confirm it fails for the missing contract before implementing that behavior.

**Step 2: Implement the smallest tree contract**

- Add a `SourceParentContext` literal and `parent_context` field to `TreeScope`.
- Split expansion into `expanded_root_kinds` and `expanded_watchlist_ids` reactives/messages.
- Pass complete all/unassigned source rows into `WatchlistTree`; derive unread children from the all-source snapshot plus bulk counts.
- Reuse one source-row renderer for aggregate and watchlist occurrences, with context-qualified ids and exact active matching.
- Keep Today and Starred as leaf smart feeds; add no pagination or virtualization.
- Render deterministic contextual empty rows and restore focus by occurrence id for caret, label, and feed-child controls after scope commit/recompose when possible.

Implement in behavior-sized GREEN slices: scope identity first, then root expansion/caret separation, then sorting/empty rows/focus, rerunning only the corresponding nodeids after each slice.

**Step 3: Run focused tree tests**

```bash
.venv/bin/python -m pytest -q Tests/Watchlists/test_watchlist_tree.py --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py Tests/Watchlists/test_watchlist_tree.py
git commit -m "feat: add contextual aggregate feed nodes"
```

### Task 3: Make the screen snapshot own complete aggregate feed data

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`

**Step 1: Write failing snapshot and persistence tests**

Add tests proving:

- One tree load obtains watchlists, complete all-source rows, complete unassigned rows, watchlist counts, and bulk source counts without per-source queries.
- Root and watchlist expansion survive section/tab changes and tree refresh independently for the screen session.
- Tree refresh updates mounted source rows and counts without rebuilding unrelated rails.
- Complete aggregate rows do not reuse the capped `_loaded_sources` management snapshot.
- Slow/fast overlapping refreshes publish only the newest generation.
- A branch refresh failure retains that branch's last exact snapshot with stale/error indication, or replaces only that branch with a failure row; unrelated branches remain intact and the user is notified once per failure episode.

Run the new tests and confirm RED.

**Step 2: Implement screen-owned tree data**

- Add `_tree_all_source_rows`, `_tree_unassigned_source_rows`, `_tree_expanded_root_kinds`, and `_tree_expanded_watchlist_ids`.
- Extend `_load_tree_data()` to acquire both complete row lists from `WatchlistBundleService` alongside existing bulk counts off the Textual event loop.
- Seed the tree factory with the two row snapshots and both expansion sets.
- Mirror the typed expansion message back to the screen and preserve it across recomposes and tab changes.
- Publish snapshots with a monotonic generation check so older work cannot overwrite a newer refresh.
- Preserve the last exact snapshot for a failed branch and expose stale/failure state without clearing unrelated branches; deduplicate notifications per failure episode.

Implement and prove these as separate RED/GREEN slices: snapshot ownership/query count, expansion persistence, then off-loop generation/failure publication.

**Step 3: Run focused snapshot tests**

```bash
.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py
git commit -m "feat: own aggregate feed snapshots in Watchlists"
```

### Task 4: Preserve contextual Reader predicates, breadcrumbs, and paging identity

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_pagination.py`
- Modify: `Tests/Watchlists/test_watchlists_workbench.py`

**Step 1: Write failing query authority tests**

Cover each contextual source occurrence:

- All Sources child: `source_id` only.
- Unassigned child: `source_id` plus `unassigned_only=True`.
- All Unread child: `source_id` plus `status="new"`.
- Watchlist child: `source_id` plus contextual `watchlist_id` membership.
- Query/page keys differ for the same source under different parents.
- Breadcrumbs include the exact parent label and source label.
- Pending, failed, and superseded scope requests do not relabel committed rows or active occurrence.
- A failed contextual request names both the attempted occurrence and the retained committed scope.
- With no feed selected, the Reader renders exactly `Select a feed to display it here.`

Run the new tests and confirm RED.

**Step 2: Implement contextual query authority**

- Include `parent_context` in `_items_page_key()`.
- Make `_items_scope_query()` emit the approved contextual predicates.
- Resolve source labels from the tree snapshot, not per-occurrence service calls.
- Include the aggregate/watchlist parent in source breadcrumbs.
- Keep `_request_tree_scope()` and `_replace_items_snapshot()` as the only async commit route in Read.
- Update the ContentPane empty copy to the approved exact string without changing loaded-item presentation.

Implement and test in separate RED/GREEN slices: query/page identity, breadcrumb/active authority, pending-failure messaging, then empty Reader copy.

**Step 3: Run focused Reader tests**

```bash
.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_workbench.py \
  --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/UI/Watchlists_Modules/content_pane.py Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_pagination.py Tests/Watchlists/test_watchlists_workbench.py
git commit -m "feat: preserve contextual feed query authority"
```

### Task 5: Park manual filters and local contextual scope honestly

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/article_list.py`
- Modify: `Tests/Watchlists/test_watchlist_tree.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_article_list.py`

**Step 1: Write failing filter/backend tests**

Add tests proving:

- A successful All Unread source commit forces effective Unread while parking the prior manual filter.
- Leaving unread context after a successful commit restores the parked manual filter; failed/superseded navigation changes neither.
- The single effective-Unread decision drives query kwargs, removal of conflicting multi-status kwargs, paging/in-flight identity, contextual empty copy, and the visible status control.
- While unread context is committed, the status control displays Unread, is disabled, and explains why; the parked manual value remains unchanged.
- Aggregate branches remain expandable on every Watchlists sub-screen.
- On local-backed management tabs, child activation commits contextual scope and management projections atomically, invalidates cached Feed Items/Reader authority, and performs no hidden item query.
- On server-backed management tabs, individual feed children are disabled with exact tooltip `Individual feed selection is available in Read or the Local backend.`
- On server-backed management tabs no child gesture is emitted, the heading remains explicitly unscoped, no local/server source-id comparison occurs, and the parked local contextual scope is not styled active.
- Returning to local Read reloads a fresh Reader snapshot.

Run the new tests and confirm RED.

**Step 2: Implement filter and backend parking**

- Track one parked manual Reader filter separately from the effective unread-context filter.
- Change it only in the successful scope commit callback.
- Centralize the effective status decision and feed it to query construction, context keys, ArticleList control state/copy, and empty-state copy.
- Expose a tree selection-disabled reason derived from section/backend while keeping expansion enabled.
- Suppress active contextual styling when server management is authoritative without discarding the parked local scope.
- Reuse the existing management cache invalidation and fresh local Read load path.

Implement and prove as separate RED/GREEN slices: effective filter authority, local-management commit/invalidation, then server-management disable/parking.

**Step 3: Run focused tests**

```bash
.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlist_tree.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py tldw_chatbook/UI/Watchlists_Modules/article_list.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists/test_watchlist_tree.py Tests/Watchlists/test_watchlists_article_list.py Tests/Watchlists/test_watchlists_collections_screen.py
git commit -m "feat: park Watchlists contextual feed preferences"
```

### Task 6: Reconcile membership/deletion changes and unread zero-count pins

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`
- Modify: `Tests/Watchlists/test_watchlist_tree.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`

**Step 1: Write failing reconciliation tests**

Prove:

- Deleted selected source falls back to its nearest existing parent.
- A selected watchlist child removed from that watchlist falls back to the watchlist.
- A selected Unassigned child that becomes assigned falls back to Unassigned.
- A selected or pending All Unread child remains visibly pinned when opening it marks its last unread item read.
- Once selection/pending authority leaves, the zero-count unread child disappears.
- Opening the last unread item retains the open row and Reader while the zero-count child is pinned; Mark unread restores positive membership.
- Failed mark-read/mark-unread writes preserve status, badge, membership, and selection.
- Collapsing/reopening retains the selected/pending pin.
- Focus, both expansion sets, Feed Items position, and Reader position remain stable through tree-data refresh.
- An invalid pending scope is discarded without prematurely committing a fallback; an invalid committed scope falls back to its nearest valid parent.

Run the new tests and confirm RED.

**Step 2: Implement one reconciliation helper**

- Reconcile committed and pending source scopes against the newly loaded tree snapshot before publishing it.
- Compute the nearest valid parent fallback from `parent_context`.
- Supply the selected/pending unread source id as the sole zero-count presentation pin.
- Do not mutate stored counts or invent synthetic service rows.

Implement and prove as separate RED/GREEN slices: committed/pending validity, unread pin lifecycle, then write-failure and view-position preservation.

**Step 3: Run focused reconciliation tests**

```bash
.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlist_tree.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists/test_watchlist_tree.py Tests/Watchlists/test_watchlists_collections_screen.py
git commit -m "fix: reconcile contextual feed navigation"
```

### Task 7: Focused verification, UI audit, and task documentation

**Files:**
- Modify: `backlog/tasks/task-22451 - Add-contextual-aggregate-feed-selection-to-Watchlists-Navigation.md`
- Modify only if warranted: `backlog/docs/lessons-testing-evidence.md`

**Step 1: Run the affected Watchlists selection only**

```bash
.venv/bin/python -m pytest -q \
  Tests/Subscriptions/test_watchlist_bundle_service.py \
  Tests/Subscriptions/test_watchlist_normalizers.py \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlist_tree.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  Tests/Watchlists/test_watchlists_workbench.py \
  --tb=short
```

Expected: PASS. Do not run the full repository suite.

**Step 2: Run modified-file Ruff**

```bash
.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py \
  tldw_chatbook/UI/Watchlists_Modules/article_list.py \
  tldw_chatbook/UI/Watchlists_Modules/content_pane.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlist_tree.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  Tests/Watchlists/test_watchlists_workbench.py
.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py \
  tldw_chatbook/UI/Watchlists_Modules/article_list.py \
  tldw_chatbook/UI/Watchlists_Modules/content_pane.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlist_tree.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  Tests/Watchlists/test_watchlists_workbench.py
```

Expected: PASS.

**Step 3: Run the Impeccable mechanical detector and branch checks**

```bash
node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs --json \
  tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py
git diff --check origin/dev...HEAD
git status --short
```

Expected: no actionable detector findings, no whitespace errors, and only intended changes.

**Step 4: Review and document**

- Review the branch diff against all six acceptance criteria.
- Check every AC in the task file.
- Add concise Implementation Notes, including targeted test/lint evidence and the inherited test-contract repair.
- Add a lessons entry only if this task uncovers a genuinely new, evidenced trap.
- Mark TASK-22451 Done only after all Definition-of-Done requirements are satisfied.
- Use `backlog task edit 22451 -s Done` for the final status transition.

**Step 5: Commit documentation**

```bash
git add Docs/superpowers/plans/2026-08-27-watchlists-contextual-aggregate-feed-selection.md "backlog/tasks/task-22451 - Add-contextual-aggregate-feed-selection-to-Watchlists-Navigation.md"
git commit -m "docs: complete TASK-22451"
```

**Step 6: Re-run final branch hygiene after the documentation commit**

```bash
git diff --check origin/dev...HEAD
git status --short
```

Expected: no whitespace errors and a clean worktree.
