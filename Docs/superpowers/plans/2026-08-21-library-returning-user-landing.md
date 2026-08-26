# Returning-User Library Landing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Use superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before each commit or completion claim.

**Goal:** Give populated Library profiles a wide, action-oriented landing that truthfully resumes the last successfully applied source scope, exposes only current recoverable problems, reuses trustworthy cached summaries under “From your Library,” and preserves the compact rail-first workflow.

**Architecture:** Extend the existing retained `LibraryLandingCanvas` and `LibraryScreen` state projection rather than adding a router, controller, service, or storage layer. A detached Continue receipt travels only through the existing runtime-compatible `ScreenStateStore` snapshot; it is separate from the active route, contains no record content, and restores the already-supported source scope fields. The receipt is captured synchronously from the current user-admitted route only when that route already owns a successfully applied state, so a late older request cannot become “last used.” Current failed-import/stale attention is derived from the live screen owners and is never saved. Existing cached source summaries remain the sole “From your Library” input, so landing composition performs no I/O.

**Tech Stack:** Python 3.11+, Textual 8.x, frozen dataclasses, pytest/Textual Pilot, Ruff.

---

## Design and ownership constraints

- The approved design is `Docs/superpowers/specs/2026-08-20-library-lifecycle-progressive-disclosure-design.md`, specifically “Returning-User Landing.”
- ADR required: no.
- ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`.
- Reason: this is the already-accepted returning-user presentation under Library’s existing lifecycle, config, source-authority, and screen-snapshot owners. It adds no schema, service contract, runtime boundary, or cross-screen manifest.
- `ScreenStateStore` remains process-memory-only and runtime-identity-scoped. Do not persist Continue in `config.toml`, and do not persist attention state anywhere.
- The landing may project existing state; it must not query a database/service, rank records, or start a refresh from `compose()` or `_library_landing_canvas_state()`.
- Continue eligibility is exact: Conversations, Database Notes, Media, Prompts, Skills, Collections, and Search. Create/edit/detail views, File Notes, Import, Export, Media Trash, and Study/Flashcard/Quiz handoffs are excluded from this slice.
- “Last successfully applied” means the current user-admitted eligible route at snapshot time whose existing owner already has an authoritative applied state. An older async success cannot replace a newer admitted route merely because it finishes later.
- A prior item/detail view may be reduced to its source list before activation, but the landing never claims the item was deleted. Deletion and page-clamp copy belongs to the source owner after its authoritative read.
- At wide size, F6’s content target is Continue when present, otherwise the first available landing action. No new single-letter binding is added. At compact size the rail remains the sole navigation owner; hidden landing focus transfers to the corresponding source/recovery/quick-action rail row and is restored on widening only when no newer user focus exists.
- User direction: run only modified/touched landing components and direct owners. Do not run repository-wide pytest.

### Approved presentation

```text
WIDE · POPULATED

CONTINUE
> Media · type: audio · page 2
  Item views resume at the source list.

NEEDS ATTENTION
! Media list may be out of date.                         [Retry]

FROM YOUR LIBRARY
  Note · Reading list
  Media · Quarterly report.pdf

QUICK ACTIONS
[Import…] [New note] [Search]

COMPACT

LIBRARY
> existing rail-first navigation
  (no second landing navigation layer)
```

Sections with no trustworthy data are omitted rather than rendered empty.

## Task 1: Define the retained landing presentation contract

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_entry_canvases.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Modify: `Tests/UI/test_library_entry_compose_once.py`

**Step 1: Write failing retained-owner tests**

Add exact tests for:

- a populated wide state rendering `Continue`, optional neutral source-list adjustment copy, `Needs attention`, `From your Library`, and `Quick actions` in that order;
- absent Continue/attention/summary inputs producing no empty headings or inert buttons;
- `None`/missing sections preserving the existing three quick-action buttons exactly once;
- `sync_state()` preserving action identity and focus when only copy or cached summaries change;
- a structural section-set change restoring semantic focus only when the user has not moved elsewhere;
- markup-bearing, CJK, emoji, combining, and mixed-direction labels rendering as text without breaking widget IDs.
- dynamic attention appearing or clearing updating readable status copy without color as its only carrier.

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_library_entry_compose_once.py \
  -k 'library and landing and (continue or attention or from_library or quick or focus)'
```

Expected RED: new state types/fields and section widgets do not exist.

**Step 2: Add the smallest display-only state**

In `library_entry_canvases.py`:

- add frozen display records for one Continue action and one attention action;
- extend `LibraryLandingCanvasState` with optional Continue/attention fields; present sections are the state key, while `LibraryScreen` remains the single lifecycle/responsive owner and passes none at compact size;
- keep `LibraryLandingRecentItem` as the existing cached-summary authority while presenting its section as “From your Library”;
- render headings only when their section has data;
- keep Import/New note/Search on the existing `.library-hub-action` route path;
- intentionally change Quick Actions DOM/Tab order from the incumbent Import → Search → New note contract to approved Import → New note → Search and update the existing ordering assertion in the same RED/GREEN step;
- attach only fixed action metadata to Continue/attention buttons—never arbitrary callable state;
- extend the existing widget-set key and deferred-sync guard instead of adding another sync mechanism.

Export only the new public display records from `Widgets/Library/__init__.py`.

**Step 3: Make the retained-owner tests green**

Run the exact Task 1 command again.

**Step 4: Prove the section omission guard**

Temporarily render all four headings unconditionally. Confirm the no-data test fails, restore immediately, and rerun that exact node green.

**Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_entry_canvases.py \
  tldw_chatbook/Widgets/Library/__init__.py \
  Tests/UI/test_library_entry_compose_once.py
git commit -m "feat(library): compose returning landing sections"
```

## Task 2: Remember Continue separately from the active route

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/State/test_screen_state_store.py`

**Step 1: Write failing snapshot/restore tests**

Cover the real `save_state()`/`restore_state()` pair:

- a fresh Media page-2/type/query/sort applied result produces a detached primitive Continue receipt carrying the full applied scope, while the restored screen’s active row is the landing;
- Prompt applied page/filter/sort and Conversation applied page/query round-trip exactly; no nonexistent Conversation sort field is introduced;
- Database Notes browse carries sort/filter but not selected note/editor content; File Notes is explicitly ineligible in this slice;
- a viewer/editor item identity is not a Continue target; restore falls back to the source list and records only the neutral “Item views resume at the source list” adjustment;
- malformed row IDs, booleans-as-pages, overflow pages, and invalid sort/filter shapes fail closed to a valid source default or omit Continue;
- the direct `ScreenStateStore` owner rejects the receipt when runtime identity is incompatible before `LibraryScreen.restore_state()` is called;
- the exact allowlist accepts only Conversations, Database Notes, Media, Prompts, Skills, Collections, and Search, while create/edit/detail, File Notes, Import, Export, Trash, and handoff rows are rejected;
- crossed Media→Prompt and Prompt→Conversation cases prove an older success resolving last cannot replace the newer user-admitted route at snapshot time;
- old snapshots without the new receipt retain the legacy selected-row restore contract used by deep links and compatibility tests;
- the receipt contains no title, body, query diagnostics, raw exception, private path, or arbitrary object.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py Tests/State/test_screen_state_store.py \
  -k 'library and (landing or screen_state) and (continue or restore or snapshot or scope or runtime)'
```

Expected RED: Continue has no independent receipt and restored state still reopens the saved active row.

**Step 2: Add one validated primitive receipt seam**

In `LibraryScreen`:

- initialize one optional screen-owned Continue receipt;
- build it synchronously at the existing `save_state()` boundary from the current user-admitted eligible route only when that route’s existing owner already has an authoritative applied state; otherwise retain the previous valid receipt rather than promoting a failed/uninitialized route;
- use the exact allowlist above: settled Conversation page, Database Notes loaded browse scope, Media/Prompt controller applied result, loaded Skills/Collections, or settled Search state; never admit a generic shell row or “static route”;
- carry route ID plus the minimum primitive scope already restored by existing source code;
- record only whether an item/detail view was reduced to its source list, never whether the item was deleted;
- save the receipt beside existing Library screen state;
- on a new-format restore, restore all source scope fields but leave `_library_selected_row_id` empty so the landing owns the first paint;
- on a legacy mapping without the receipt key, preserve today’s selected-row behavior;
- validate through the existing `_restore_library_media_scope`, `_restore_library_prompts_scope`, Conversation page normalizer, and one fixed Continue-eligible row set instead of the general shell table;
- keep runtime compatibility coverage in `Tests/State/test_screen_state_store.py`, where the owner can actually reject the snapshot before screen restoration.

Do not write the receipt to Library rail preferences or lifecycle config.

**Step 3: Make snapshot/restore tests green**

Run the exact Task 2 command again.

**Step 4: Prove the “separate from active route” invariant**

Temporarily restore `_library_selected_row_id` from the Continue receipt. Confirm the returning-landing test fails because the source canvas opens directly, restore immediately, and rerun green.

**Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py Tests/State/test_screen_state_store.py
git commit -m "feat(library): retain a separate continue receipt"
```

## Task 3: Project Continue, truthful fallback, and existing route dispatch

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`

**Step 1: Write mounted RED tests**

Use the production `LibraryHarness` and source fakes to prove:

- a returning populated screen first paints the landing and Continue describes the last applied full scope;
- pressing Continue uses `_select_library_rail_row` and the existing source controller, restoring Media/Prompt page/filter/sort, Conversation page/query, and Database Notes sort/filter rather than page 1/default scope;
- a prior item view opens its source list with neutral copy, never a fabricated selection or pre-read deletion claim;
- an actually deleted prior item is disclosed only after the source owner’s authoritative detail/list read, while a still-existing item never receives deletion copy;
- an out-of-range page uses the source owner’s existing one-clamp behavior and the landing makes no pre-read clamp/deletion claim;
- a Database Notes Continue reapplies the validated filter after generic navigation admission so `_select_library_rail_row` cannot clear it before the targeted canvas sync;
- File Notes receipts are rejected and never strand the screen on `#library-file-notes-loading`;
- current focus moved after a retained landing sync is not stolen by a stale Continue callback;
- direct navigation context and command-palette/deep-link routes still bypass the landing.

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py \
  -k 'library and landing and continue'
```

Expected RED: no Continue projection or handler exists.

**Step 2: Wire the screen projection and handler**

- project the validated receipt into the landing display state only for Expanded/Graduated wide presentation;
- derive readable labels from the fixed Library shell row metadata and normalized scope fields, never from private record content;
- use one `#library-hub-continue` handler that delegates to the existing guarded rail-row admission path;
- let each source owner perform its existing authoritative read/clamp/failure handling;
- clear any selected record identity before list fallback and preserve only the receipt’s neutral source-list adjustment copy;
- for Database Notes only, reapply the validated sort/filter at the narrow Continue seam after generic route admission and before targeted canvas sync; do not alter general Notes rail behavior;
- use the retained owner’s generation/route guard and existing semantic focus restore path.

**Step 3: Make mounted Continue tests green**

Run the exact Task 3 command again.

**Step 4: Prove full-scope dispatch**

Temporarily force Continue to page 1. Confirm the page-2 mounted test fails on the service-call offset/scope assertion, restore immediately, and rerun green.

**Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py
git commit -m "feat(library): resume the last applied source scope"
```

## Task 4: Add screen-owned attention without persistence

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_library_entry_compose_once.py`

**Step 1: Write failing ownership/recovery tests**

Cover only recoveries already owned by the current screen:

- the most recent recoverable failed ingest job projects a `Review` action that opens the existing Import/job recovery surface;
- Media, Prompt, or Conversation stale state projects one deterministic `Retry` action that opens the owning source and uses its existing retry/refresh path;
- fresh, initial-loading, hard-error-without-recovery, dismissed failure, and superseded generation states produce no Needs attention section;
- saving/restoring a screen serializes no attention record, failed-job copy, stale copy, ID, path, or exception;
- a late stale/failure result from an invalidated screen cannot add attention to a new landing;
- pressing Review/Retry after a newer user focus/route move does not overwrite that intent.
- `mark_failed`, `requeue`, and `dismiss` update or remove Needs attention while the same retained landing owner remains mounted;
- attention appearance/clearance changes readable status copy and does not rely on color alone.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py Tests/UI/test_library_entry_compose_once.py \
  -k 'library and landing and attention'
```

Expected RED: no landing attention projection/action exists.

**Step 2: Derive one current attention item**

- inspect only live screen/controller/registry state already in memory;
- choose one item with a documented deterministic priority (recoverable failed import, then active stale source in fixed Library source order);
- expose fixed operation metadata and bounded user copy only;
- route Review/Retry through existing Import or source handlers;
- extend the existing ingest registry listener to targeted-sync the landing only when its bounded attention signature changes, guarded by current route, screen generation, and newer user focus; do not start a broad snapshot refresh for failure/requeue/dismiss;
- never add attention fields to `save_state()` and never infer it from a prior restart snapshot.

**Step 3: Make attention tests green**

Run the exact Task 4 command again.

**Step 4: Prove non-persistence**

Temporarily include attention in `save_state()`. Confirm the restart test fails its forbidden-key/private-sentinel assertion, restore immediately, and rerun green.

**Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py Tests/UI/test_library_entry_compose_once.py
git commit -m "feat(library): surface current landing recovery"
```

## Task 5: Preserve cached “From your Library” summaries, quick-action ownership, and no-I/O composition

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_library_entry_compose_once.py`

**Step 1: Add no-I/O and authority RED tests**

- seed only `_local_source_records`/counts and assert “From your Library” renders the existing first trustworthy summary per supported source in fixed source order, without implying cross-source chronology;
- omit missing IDs, unresolved records, empty raw summaries, and every summary when the global source snapshot failed; do not claim per-source unavailability because the current snapshot has no such authority;
- add/use a raw optional-summary helper for landing projection so the incumbent list fallback “Untitled source” is never presented as trustworthy cached summary;
- install service/DB spies, establish the normal mounted-screen refresh baseline, then invoke `_library_landing_canvas_state()` and retained landing sync; assert those operations add zero reads/scans/ranking calls. Use a directly mounted `LibraryLandingCanvas` for pure composition assertions;
- press Import, New note, and Search and assert the exact existing route handlers/canvas owners and semantic first focus;
- gate a landing recompose, move focus, and assert completion preserves the newer focus.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py Tests/UI/test_library_entry_compose_once.py \
  -k 'library and landing and (from_library or quick or no_io or focus)'
```

Expected RED: the returning section contract or no-I/O guard is not yet complete.

**Step 2: Keep projection read-only and reuse handlers**

- retain `_hub_recent_items()` as the only cached-summary source while omitting records without a raw trustworthy title;
- do not sort beyond its fixed existing source order and do not add timestamps/relative-time work;
- extend the existing `.library-hub-action` path rather than adding per-button route logic;
- use the landing retained-sync callback and route/generation guard for any structural update.

**Step 3: Make authority tests green**

Run the exact Task 5 command again.

**Step 4: Prove no-I/O composition**

Temporarily call one source list method from `_library_landing_canvas_state()`. Confirm the spy test fails immediately, restore, and rerun green.

**Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py Tests/UI/test_library_entry_compose_once.py
git commit -m "fix(library): keep landing summaries source owned"
```

## Task 6: Verify wide/compact geometry, keyboard order, and compatibility

**Files:**
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_library_entry_compose_once.py`
- Modify only if mounted RED requires it: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate only if CSS changes: generated CSS bundle produced by `tldw_chatbook/css/build_css.py`

**Step 1: Add production-hierarchy geometry tests**

At exact 170x48 with `TldwCli.CSS_PATH`, assert:

- Continue, present attention, “From your Library,” and all three quick actions paint inside the landing pane with Continue as the strongest content-stage action, recovery using the existing callout grammar, cached summaries quiet, and Quick Actions secondary;
- headings precede their owned rows and long safe labels do not cover actions;
- actual Tab traversal follows Continue → attention action (when present) → cached summary rows → Import → New note → Search, then leaves/wraps according to the existing screen contract;
- F6 from the rail enters Continue when present (otherwise the first available landing action); no new single-letter binding or false footer hint is added;
- activation and actual supported return paths preserve semantic focus; do not invent a landing Back path solely for this test.

At exact 100x30, assert:

- compact remains rail-first and the returning landing does not create a second competing navigation layer;
- rail routes remain visible, reachable, and truthful;
- resizing 170x48 → 100x30 transfers focus from a hidden landing control to its semantic rail owner (Continue → receipt source row, attention → recovery source row, summaries → source row, Quick Actions → matching rail row), removes hidden landing controls from Tab/F6 order, and preserves receipt/scope/recovery state;
- resizing back to 170x48 restores the prior landing identity only when no newer compact-stage focus move occurred;
- CJK, emoji, combining, markup, and mixed-direction text stay compositor-contained by terminal cell width, not merely present in widget labels.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py Tests/UI/test_library_entry_compose_once.py \
  -k 'library and landing and (geometry or keyboard or focus or resize or continue or attention)'
```

**Step 2: Prefer no CSS change**

First run against the existing landing/toolbar CSS. If a production-hierarchy RED proves clipping, make the smallest component CSS change, regenerate the bundle, and rerun the exact geometry nodes. Do not hand-edit generated CSS.

If CSS changes, run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
.venv/bin/python -m pytest -q Tests/UI/test_css_build_integrity.py
```

**Step 3: Run the complete touched/direct-owner gate**

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_entry_compose_once.py \
  Tests/UI/test_library_shell.py \
  Tests/State/test_screen_state_store.py \
  -k 'library and (landing or screen_state) and (continue or attention or from_library or quick or restore or scope or runtime or geometry or keyboard or focus or resize or retry)'
```

Do not run bare `pytest` or the repository-wide suite.

**Step 4: Static verification**

```bash
.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/Library/library_entry_canvases.py \
  tldw_chatbook/Widgets/Library/__init__.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_entry_compose_once.py \
  Tests/UI/test_library_shell.py \
  Tests/State/test_screen_state_store.py
git diff --check origin/dev...HEAD
git diff --check
```

Run Ruff format-check only on touched files that already conform at the task base; do not accept unrelated whole-file formatting churn in legacy `library_screen.py` or the large shell test.

**Step 5: Self-review and closeout**

- inspect `git diff --name-status origin/dev...HEAD` and ensure every changed file is in the task boundary;
- verify no title/body/path/query/exception appears in Continue or attention persistence/logging, and no deletion/unavailability claim appears before authoritative source evidence;
- verify all Task 19641 ACs against exact evidence;
- update `Docs/User_Guide/library.md` with the returning landing and ASCII-only layout;
- add concise Implementation Notes to TASK-19641, document the no-full-suite deviation, and check every AC;
- add a lessons entry only if implementation uncovers a genuinely reusable incident;
- mark Done through Backlog CLI only after final focused verification and review.

**Step 6: Commit closeout**

```bash
git add Docs/User_Guide/library.md \
  "backlog/tasks/task-19641 - Build-the-returning-user-Library-landing.md"
git commit -m "docs(library): document returning landing"
```

## Final review checklist

- Continue is a source route/full-scope receipt, never a record snapshot.
- New-format restoration lands on the landing; legacy/deep-link snapshots remain compatible.
- Item/detail selection falls back to the source’s valid list scope with neutral pre-read adjustment copy; actual invalid/deleted/clamped outcomes are named only by the authoritative source after its read.
- Attention is current-screen-only, recoverable, bounded, and absent after restart.
- “From your Library” uses existing cached summaries in fixed source order and landing composition performs zero source reads.
- Import/New note/Search and all recovery actions use existing route owners.
- 170x48 is the populated wide landing; 100x30 remains rail-first.
- Continue is restricted to the exact eligible source allowlist; Database Notes is supported and File Notes is explicitly excluded in this slice.
- Structural updates are route/generation guarded; late older success cannot become Continue, hidden compact focus transfers semantically, and newer user focus wins.
- Only touched/direct-owner tests are claimed; no repository-wide pytest claim is made.
