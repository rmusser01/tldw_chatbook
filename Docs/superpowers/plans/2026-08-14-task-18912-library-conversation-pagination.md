# TASK-18912 Library Conversation Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every top-level Conversation reachable through an exact, resilient 20-item Library pager and land the one pure pager-display function reused by the later Prompt, Media, Skills, and Collections tasks.

**Architecture:** Keep Conversation storage, service, screen request state, Textual workers, canvas widgets, selection, and navigation in their current owners. Add one pure immutable pager-display calculation, repair the existing Conversation count/page query into one coherent read transaction, add a bounded stable-ID owning-page locator, and harden the screen's requested/applied/freshness lifecycle. Do not introduce a generic pager widget, generic source controller, root application paging state, or new dependency.

**Tech Stack:** Python 3.11+, SQLite/FTS5, Textual 8.x, dataclasses, pytest/Pilot, Ruff.

---

## Authority and constraints

- Task: `backlog/tasks/task-18912 - Standardize-Library-pager-display-and-harden-Conversation-paging.md`
- Design: `Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md`
- Architecture: `backlog/decisions/067-library-top-level-pagination-contracts.md` (Accepted), ADR-003, ADR-030, and ADR-033.
- Worktree: `.worktrees/library-top-level-pagination`; branch `codex/library-top-level-pagination`. Task 0 rebases onto latest `origin/dev`; the exact task base is always `git merge-base HEAD origin/dev` after that rebase, never the older planning SHA.
- ADR required: yes.
- ADR path: `backlog/decisions/067-library-top-level-pagination-contracts.md`.
- Reason: this task changes the cross-module Conversation page and stable-ID locator contracts and establishes the pure shared display contract.
- Use @superpowers:test-driven-development for each behavior change, @superpowers:systematic-debugging for unexpected failures, @textual-tui for mounted focus/worker behavior, @ponytail for the smallest source-owned implementation, and @superpowers:verification-before-completion before any success claim.
- Fixed page size is 20. Ordinary browse pages use exact `limit`/`offset`; stable-ID locators return a validated resolved page/offset.
- No production edit may precede its focused RED. Apply inverse mutations only with `apply_patch`, run the named oracle, and restore immediately.
- Preserve unrelated dirty work. Stage only files selected by this plan.
- Do not start the repository-wide gate while another broad pytest process is running. Compare a red broad gate with the exact post-rebase task merge base in an isolated worktree by node names, not counts.

## File map

Create:

- `tldw_chatbook/Library/library_pager_state.py` — one pure immutable pager-display calculation; no source records, workers, widgets, or event messages.
- `Tests/Library/test_library_pager_state.py` — exhaustive pure first/middle/final/empty/uninitialized/loading/stale display tests.

Modify:

- `tldw_chatbook/DB/ChaChaNotes_DB.py` — share the Conversation WHERE/order construction, execute count and rows in one read transaction, and add the bounded stable-ID owning-page query.
- `tldw_chatbook/Chat/chat_conversation_service.py` — expose validated ordinary and locator envelopes and retain summary enrichment.
- `tldw_chatbook/Chat/chat_conversation_scope_service.py` — route the additive locator call through the existing local/server policy boundary without changing mode ownership.
- `tldw_chatbook/Library/library_conversations_state.py` — consume `LibraryPagerDisplay`, validate source summaries before row construction, and retain Conversation-specific row/selection/preview state.
- `tldw_chatbook/UI/Screens/library_screen.py` — own requested/applied Conversation scope, freshness, generation, retry, focus intent, selection notice, restore, clamping, unmount fencing, broad-snapshot isolation, and deep-link application.
- `tldw_chatbook/Widgets/Library/library_conversations_canvas.py` — render exact title/range, visible disabled reasons, Retry, and the existing independent list viewport using Conversation-specific IDs.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` — keep the Conversation list at `1fr/min-height: 0` and pager/status visible at supported sizes.
- `tldw_chatbook/css/tldw_cli_modular.tcss` — regenerate from component CSS; reject timestamp-only churn.
- `Tests/DB/test_search_conversations_fts.py` — coherent snapshot, deterministic order, locator, and malformed coordinate coverage around the real SQLite service.
- `Tests/Chat/test_chat_conversation_service.py` — exact envelope/locator projection and validation behavior.
- `Tests/Library/test_library_conversations_state.py` — source-specific row identity validation and integration with the pure pager display.
- `Tests/Library/test_library_conversations_visibility.py` — all-scope visibility and concurrent count/row coherence.
- `Tests/UI/test_library_shell.py` — real mounted request lifecycle, retry, focus, navigation, restore, races, deep links, stale recovery, and broad-snapshot isolation.
- `Tests/UI/test_library_multiselect_conversations.py` — current-page selection clearing and stale-action safety.
- `Tests/UI/test_destination_shells.py` — update real destination service-contract fakes for the additive locator seam when required.
- `Docs/User_Guide/library/media-and-conversations.md` — document the 20-item Conversation pager and recovery/selection behavior.
- `backlog/tasks/task-18912 - Standardize-Library-pager-display-and-harden-Conversation-paging.md` — implementation plan link, checked AC, evidence, ADR, notes, and Done status at closeout only.

Do not modify Prompt, Media, Skill, Collection, Notes-tree, Trash, or Collection-member production owners in TASK-18912.

---

### Task 0: Rebase, read repository lessons, and freeze the baseline

**Files:** None

- [x] **Step 1: Read the task, accepted design/ADR, and required lessons**

Run:

```bash
backlog task 18912 --plain
cat backlog/docs/lessons-testing-evidence.md
cat backlog/docs/lessons-live-verification.md
cat backlog/docs/lessons-backlog-hygiene.md
```

Then read the complete spec, ADR-067, and this plan. Expected: task is In
Progress; ADR-067 is Accepted; no requirement conflicts.

- [x] **Step 2: Rebase the planning commit onto the latest dev and prove import provenance**

Run serially:

```bash
git status --short --branch
git fetch origin dev
git rebase origin/dev
PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import pathlib, tldw_chatbook; p=pathlib.Path(tldw_chatbook.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path.cwd().resolve())'
git merge-base HEAD origin/dev
git status --short --branch
```

Expected: clean `codex/library-top-level-pagination`, import beneath this
worktree, no unresolved rebase changes, and the printed merge base equals
`git rev-parse origin/dev` immediately after rebase. Record that printed SHA in
the task evidence. If latest dev changes any selected owner contract, amend and
re-review the plan before production edits.

- [x] **Step 3: Capture the current focused baseline before adding tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_conversations_state.py Tests/Library/test_library_conversations_visibility.py Tests/Chat/test_chat_conversation_service.py -q --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_multiselect_conversations.py -q -k 'conversation' --tb=short
```

Expected: record exact passed/failed node names. Do not call a failure baseline
unless the identical node reproduces on the rebased `origin/dev` commit in an
isolated worktree.

---

### Task 1: Add the pure pager-display contract

**Files:**

- Create: `tldw_chatbook/Library/library_pager_state.py`
- Create: `Tests/Library/test_library_pager_state.py`

- [x] **Step 1: Write pure RED tests for every display state**

Add tests that construct the display calculation directly and assert:

```python
def test_fresh_middle_page_has_exact_range_and_enabled_boundaries():
    display = build_library_pager_display(
        applied_page=2,
        requested_page=2,
        page_size=20,
        row_count=20,
        total=45,
        freshness="fresh",
    )
    assert display.title_count == 45
    assert display.range_copy == "21-40 of 45"
    assert display.page_copy == "Page 2 of 3"
    assert display.previous_disabled is False
    assert display.next_disabled is False


def test_uninitialized_failure_never_fabricates_zero_total():
    display = build_library_pager_display(
        applied_page=None,
        requested_page=1,
        page_size=20,
        row_count=0,
        total=None,
        freshness="uninitialized",
        error_copy="Couldn't load conversations.",
    )
    assert display.title_count is None
    assert display.range_copy == "No page loaded · Total unavailable"
    assert display.retry_visible is True


def test_stale_display_suppresses_exact_metadata_and_actions():
    display = build_library_pager_display(
        applied_page=3,
        requested_page=3,
        page_size=20,
        row_count=5,
        total=None,
        freshness="stale",
        stale_copy="Source changed again; try again.",
    )
    assert display.title_count is None
    assert display.range_copy == "List may be out of date"
    assert display.previous_disabled is True
    assert display.next_disabled is True
    assert display.retry_visible is True
```

Cover first, final, exact-multiple, one-row, successfully empty, initial loading, page-only loading with last-good metadata, recoverable failure, and both stale reasons. Assert visible disabled reasons (`Already on the first page.`, `No more results.`, `Page is loading.`, or unknown boundary) rather than tooltip-only state.

- [x] **Step 2: Run the pure tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_pager_state.py -q --tb=short
```

Expected: collection/import failure because `library_pager_state.py` and `build_library_pager_display` do not exist.

- [x] **Step 3: Implement only the pure immutable display function**

Create this minimal public shape, with exact input validation and no source/widget knowledge:

```python
from dataclasses import dataclass
from typing import Literal

PageFreshness = Literal["uninitialized", "fresh", "stale"]


@dataclass(frozen=True)
class LibraryPagerDisplay:
    title_count: int | None
    range_copy: str
    page_copy: str
    status_copy: str
    previous_disabled: bool
    next_disabled: bool
    previous_reason: str
    next_reason: str
    retry_visible: bool


def build_library_pager_display(
    *,
    applied_page: int | None,
    requested_page: int,
    page_size: int,
    row_count: int,
    total: int | None,
    freshness: PageFreshness,
    loading: bool = False,
    error_copy: str = "",
    stale_copy: str = "",
) -> LibraryPagerDisplay:
    ...
```

Reject booleans and invalid integers. Exact totals exist only while fresh. Derive all range/page/disabled/retry copy in this function. Do not add a generic response validator, request state, widget, composer, callback, or source abstraction.

- [x] **Step 4: Run pure tests and Ruff**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_pager_state.py -q --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Library/library_pager_state.py Tests/Library/test_library_pager_state.py
```

Expected: all pass.

- [x] **Step 5: Commit the pure foundation**

```bash
git add tldw_chatbook/Library/library_pager_state.py Tests/Library/test_library_pager_state.py
git commit -m "feat(library): add pure pager display state"
```

---

### Task 2: Make Conversation pages coherent and add the stable-ID locator

**Files:**

- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_scope_service.py`
- Modify: `Tests/DB/test_search_conversations_fts.py`
- Modify: `Tests/Chat/test_chat_conversation_service.py`
- Modify: `Tests/Library/test_library_conversations_visibility.py`

- [x] **Step 1: Write DB REDs for one-snapshot pages and owning-page lookup**

Use a real file-backed `CharactersRAGDB` with more than 40 rows, equal timestamps, global/workspace rows, soft deletion, and an FTS match. Assert ordinary pages return exact 20/20/5 partitions ordered by `last_modified DESC, id DESC` without skips or duplicates.

Add a coordinated two-connection WAL test: pause the reader after count evaluation, commit a second-connection insert/delete, release the page read, and assert total and items still describe one transaction snapshot. Instrument only a test seam around the two statements; do not weaken production transaction ownership.

Add a locator test:

```python
located = db.locate_conversation_page(
    target_id,
    scope_type="all",
    limit=20,
)
assert located["offset"] == 20
assert located["target_index"] == 24
assert located["total"] == 45
assert target_id in {row["id"] for row in located["rows"]}
assert located["rows"][located["target_index"] - located["offset"]]["id"] == target_id
assert len(located["rows"]) == 20
```

Cover first/final pages, equal timestamps, deleted/unavailable target, invalid stable ID, and exact page alignment. The locator must not walk pages or return an unbounded list.

- [x] **Step 2: Write service REDs for exact ordinary and locator envelopes**

Assert `list_conversations` retains:

```python
{
    "items": [...],
    "pagination": {
        "limit": 20,
        "offset": 20,
        "total": 45,
        "has_more": True,
    },
}
```

Add `locate_conversation_page(conversation_id, ..., limit=20)` and assert its envelope carries zero-based `target_index`, resolved aligned `offset`, derived one-based `page`, exact `total`, bounded `items`, and the requested ID at local position `target_index - offset`. Validate `offset == (target_index // limit) * limit`. Update scope-service tests to prove local routing/off-loop behavior and explicit server unsupported behavior unless a real server locator already exists; do not fake server support.

- [x] **Step 3: Run the focused DB/service REDs**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_search_conversations_fts.py Tests/Chat/test_chat_conversation_service.py Tests/Library/test_library_conversations_visibility.py -q -k 'coherent or page or locator or scope_all' --tb=short
```

Expected: locator tests fail because no locator exists; the concurrent-write test demonstrates the current separate count/page statements can observe a mixed result.

- [x] **Step 4: Implement the minimal shared WHERE/order seam and transaction**

Extract one private filter builder used by both ordinary and locator reads. In `search_conversations_page`, execute count and rows on the same connection:

```python
with self.transaction() as conn:
    total_row = conn.execute(count_query, tuple(params)).fetchone()
    rows = conn.execute(page_query, (*params, limit, offset)).fetchall()
```

The locator uses one coherent window/rank query under the identical WHERE and `ORDER BY last_modified DESC, id DESC`, returns zero-based `target_index`, computes `resolved_offset = (target_index // limit) * limit`, and returns only that page plus exact total. Validate `limit`, target ID, page-local position, and aligned resolved coordinates before returning. Keep SQL parameterized.

- [x] **Step 5: Add the service/scope method without a new controller**

`ChatConversationService.locate_conversation_page()` normalizes/enriches the bounded rows exactly as `list_conversations()` does. `ChatConversationScopeService.locate_conversation_page()` applies the existing detail/list policy boundary and uses the same local off-loop rule; it owns no paging state.

- [x] **Step 6: Run focused and owner service suites**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_search_conversations_fts.py Tests/Chat/test_chat_conversation_service.py Tests/Library/test_library_conversations_visibility.py -q --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Chat/chat_conversation_service.py tldw_chatbook/Chat/chat_conversation_scope_service.py Tests/DB/test_search_conversations_fts.py Tests/Chat/test_chat_conversation_service.py Tests/Library/test_library_conversations_visibility.py
```

Expected: all pass.

- [x] **Step 7: Commit the coherent service contract**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Chat/chat_conversation_service.py tldw_chatbook/Chat/chat_conversation_scope_service.py Tests/DB/test_search_conversations_fts.py Tests/Chat/test_chat_conversation_service.py Tests/Library/test_library_conversations_visibility.py
git commit -m "feat(library): add coherent conversation page locator"
```

---

### Task 3: Validate Conversation summaries and adopt the pure pager display

**Files:**

- Modify: `tldw_chatbook/Library/library_conversations_state.py`
- Modify: `Tests/Library/test_library_conversations_state.py`

- [x] **Step 1: Replace tolerant-malformed tests with fail-closed REDs**

Preserve harmless field fallbacks for a valid identified row, but require the page itself to fail before rendering when an item is not a mapping, lacks a stable Conversation ID, or duplicates an ID. Add exact expected-cardinality and coordinate checks through a Conversation-specific normalized-page function.

```python
with pytest.raises(ValueError, match="stable conversation identity"):
    validate_library_conversation_page(
        {"items": [{"title": "missing id"}],
         "pagination": {"limit": 20, "offset": 0, "total": 1}},
        requested_limit=20,
        requested_offset=0,
    )
```

Cover invalid booleans, negative totals, unequal limit/offset echoes, undersized non-final pages, oversized pages, duplicate IDs, and a valid empty page.

- [x] **Step 2: Run state REDs**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_conversations_state.py -q --tb=short
```

Expected: new validation tests fail because the current builder silently skips malformed rows and derives pager copy itself.

- [x] **Step 3: Implement source-specific normalization and display composition**

Add an immutable validated Conversation envelope (or tuple return) in `library_conversations_state.py`; keep it Conversation-specific. Replace range/page/disabled arithmetic in `build_library_conversations_state()` with `build_library_pager_display(...)`. Extend `LibraryConversationsCanvasState` with the pure display and source-specific selection notice/action-disable fields rather than duplicating pager math.

Do not sort page records again in the state builder: the DB/service ordering is authoritative. Preserve service order and use stable IDs only for validation/selection.

- [x] **Step 4: Run state tests and Ruff**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_pager_state.py Tests/Library/test_library_conversations_state.py -q --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Library/library_pager_state.py tldw_chatbook/Library/library_conversations_state.py Tests/Library/test_library_pager_state.py Tests/Library/test_library_conversations_state.py
```

Expected: all pass.

- [x] **Step 5: Commit state integration**

```bash
git add tldw_chatbook/Library/library_conversations_state.py Tests/Library/test_library_conversations_state.py
git commit -m "refactor(library): validate conversation page display"
```

---

### Task 4: Harden screen request, restore, retry, selection, and deep-link lifecycle

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_library_multiselect_conversations.py`
- Modify: `Tests/UI/test_destination_shells.py` when its real harness must expose the locator

- [x] **Step 1: Write mounted REDs for requested/applied scope and first failure**

Using the real `LibraryHarness` and mounted controls, assert:

- initial load renders `Loading page 1…` without a title count;
- first failure keeps the filter, exposes retryable screen state, and renders `No page loaded · Total unavailable` while both pagers are disabled; Task 5 owns mounting `#library-conversations-retry` in the source canvas;
- a failed scope change retains old rows/applied title/range, the requested input text, and `Filter wasn’t applied; showing previous results.`;
- Retry uses the requested scope, while a later successful authoritative request atomically replaces applied scope;
- save/restore persists only the last successful applied page/query, never records, loading/error, failed requested scope, or unsubmitted drafts;
- restore rejects bool/string/zero/negative/overflow pages before service dispatch and re-fetches a valid page.

- [x] **Step 2: Write mounted REDs for focus, selection, and visible reasons**

Assert filter focus survives submission. On Next, capture the invoking control before loading; page 2 returns focus to Next, while the final page moves focus to Previous because Next is disabled. The visible pager status contains the boundary/loading reason.

Enter Conversation Select mode, select rows, then page/filter. Assert mode exits, IDs clear, `Selection cleared.` is visible, and no invisible selection can export. Add a stale state test proving row/bulk/Previous/Next actions are disabled while Retry and filter recovery remain enabled.

- [x] **Step 3: Write race, unmount, broad-snapshot, and double-shrink REDs**

Use gates/events rather than sleeps:

- old filter/page result finishes after a new result and cannot apply;
- a request finishes after Library unmount and cannot touch the old or fresh screen;
- a late broad Library snapshot updates rail/landing/RAG consumers but cannot replace a dedicated Conversation page;
- a page becomes out of range and automatically reloads the last valid page once;
- the source shrinks again before that one follow-up result: no third call starts, last-good records become stale, exact total/range disappears, actions disable, and Retry/scope recovery is visible.

- [x] **Step 4: Rewrite the deep-link RED around deterministic owning-page behavior**

Update `test_library_conversation_id_context_opens_off_page_conversation` to use the real locator envelope and 45 rows. Assert target 25 reports `target_index=24`, resolves to page 2 with `21-40 of 45 · Page 2 of 3`, appears at page-local index 4, is selected in service order, and no extra row is injected. Add malformed locator cases: missing target, divergent target index/local position, wrong/unaligned offset, wrong rank-derived page, duplicate ID, and wrong cardinality fail inside the Conversation canvas with Retry.

- [x] **Step 5: Run the exact RED group**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_multiselect_conversations.py -q -k 'conversation and (retry or applied_scope or focus or selection or stale or unmount or broad_snapshot or shrink or context)' --tb=short
```

Expected: new nodes fail for missing freshness/requested-page/retry/focus/locator behavior; every test teardown releases its gates and waits only for the relevant Conversation worker group.

- [x] **Step 6: Implement the minimal source-owned screen state**

Keep existing fields where practical and add only:

```python
self._library_conversation_requested_page = 1
self._library_conversation_requested_query = ""
self._library_conversation_freshness = "uninitialized"
self._library_conversation_stale_copy = ""
self._library_conversation_selection_notice = ""
self._library_conversation_focus_after_apply = ""
```

Preparation records requested scope, invalidates generation, captures focus intent, clears current-page selection with notice, and publishes loading without mutating applied page/query/total. The loader validates the envelope before applying all applied fields together. Malformed/service failures remain in the canvas and expose retryable state.

Implement at most one automatic limit/offset clamp. A second out-of-range response enters stale state exactly as the design specifies. Malformed/service failures expose retryable screen state without crossing into the Task 5 canvas-rendering owner. `on_unmount` increments the Conversation generation and cancels screen-owned debounce/timers without claiming to stop an already-running thread call.

Save only applied page/query. Restore exact non-boolean positive integers after checked offset arithmetic; otherwise page 1. Do not restore records, freshness, loading, errors, stale copy, requested-but-unapplied scope, or selection.

- [x] **Step 7: Replace point-lookup/prepend deep linking with the locator**

In `_open_library_item_by_id("conversations", ...)`, preserve dirty-editor admission and navigation precedence, then call the source locator. Validate and atomically apply its page before selecting the target. Remove the page-1 prepend/truncate branch and any total fabrication. Unavailable targets retain the existing warning and retryable state; Task 5 mounts the Retry control.

- [x] **Step 8: Run the exact mounted group and focused Conversation suites**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_multiselect_conversations.py -q -k 'conversation' --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_pager_state.py Tests/Library/test_library_conversations_state.py Tests/Library/test_library_conversations_visibility.py Tests/Chat/test_chat_conversation_service.py -q --tb=short
```

Expected: all Conversation/pager tests pass.

- [x] **Step 9: Commit screen lifecycle behavior**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py Tests/UI/test_library_multiselect_conversations.py Tests/UI/test_destination_shells.py
git commit -m "fix(library): harden conversation page lifecycle"
```

---

### Task 5: Render the exact Conversation pager and verify geometry

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_conversations_canvas.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_library_shell.py`

- [x] **Step 1: Write canvas/geometry REDs at 100×30 and 170×48**

Assert the title uses the authoritative total rather than `len(rows)`, first-load title has no count, Retry is source-owned, disabled labels use the existing non-colour marker, and boundary/loading reasons are visible in mounted status text.

At both sizes assert:

```python
assert list_view.region.bottom <= pager.region.top
assert pager.region.bottom <= list_pane.region.bottom
assert pager.region.left >= list_pane.region.left
assert pager.region.right <= list_pane.region.right
```

Scroll the independent viewport to row 20 while pager identity/region remains fixed. Exercise page 2/final focus fallback through real Button presses.

- [x] **Step 2: Run geometry REDs**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py -q -k 'conversation and (geometry or pager or row_20 or disabled_reason or retry)' --tb=short
```

Expected: new title/retry/visible-reason assertions fail before canvas changes.

- [x] **Step 3: Implement source-specific rendering only**

Keep `#library-conversations-list` as the independent `VerticalScroll`. Render the pure display fields with existing Conversation IDs, plus `#library-conversations-retry`. Use the approved title count and status copy. Do not create a generic pager widget or composer.

Add only shared styling classes that describe layout, not behavior/state. Keep list `height: 1fr; min-height: 0`, pager `height: auto`, status centered, and controls within the list pane.

- [x] **Step 4: Regenerate CSS and reject timestamp-only churn**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```

Inspect the generated diff. If only a timestamp changes, restore it; otherwise
require exact semantic parity between component and bundle. Never hand-edit the
generated bundle.

- [x] **Step 5: Run mounted/geometry suites and Ruff**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_multiselect_conversations.py -q -k 'conversation' --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Widgets/Library/library_conversations_canvas.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py Tests/UI/test_library_multiselect_conversations.py
git diff --check
```

Expected: all pass and no pager escapes either supported host.

- [x] **Step 6: Commit canvas/CSS behavior**

```bash
git add tldw_chatbook/Widgets/Library/library_conversations_canvas.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_shell.py
git commit -m "feat(library): render resilient conversation pager"
```

---

### Task 6: Run inverse mutations, owner gates, isolated live verification, and closeout

**Files:**

- Modify: `Docs/User_Guide/library/media-and-conversations.md`
- Modify: `backlog/tasks/task-18912 - Standardize-Library-pager-display-and-harden-Conversation-paging.md`

- [x] **Step 1: Run and restore required inverse mutations**

Use `apply_patch`; run the exact oracle; restore immediately; rerun green:

1. Remove the Conversation generation equality guard; stale-result race must fail.
2. Restore deep-link prepend behavior; deterministic owning-page test must fail.
3. Split count and row statements across independent DB calls; coordinated-write coherence test must fail.
4. Allow duplicate/missing IDs to be silently dropped; malformed-page test must fail.
5. Leave one stale row/bulk action enabled; stale-action safety test must fail.
6. Allow a second automatic clamp; double-shrink bounded-call test must fail.

Record exact failed assertion/output and restored-green command in implementation notes.

- [x] **Step 2: Run focused and owner suites**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_pager_state.py Tests/Library/test_library_conversations_state.py Tests/Library/test_library_conversations_visibility.py Tests/DB/test_search_conversations_fts.py Tests/Chat/test_chat_conversation_service.py -q --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_multiselect_conversations.py Tests/UI/test_destination_shells.py -q -k 'conversation' --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py -q --tb=short
```

Expected: all task-local and owner tests pass. Any broad failure is classified
against the exact post-rebase task base returned by
`git merge-base HEAD origin/dev` in a read-only isolated worktree before
proceeding; no baseline label is accepted without identical node reproduction.

- [x] **Step 3: Perform isolated live verification at both sizes**

Create a fresh scratch profile before importing app modules. Set exact scratch `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH`, `[paths].data_dir`, and `TLDW_TEST_MODE=1`; keep stderr attached to the PTY. Fingerprint the real config/data roots before and after and require byte-identical manifests. Prove the TUI PID has no real-profile DB/config handles.

Seed 45 synthetic Conversations with one unique oldest-page search marker. At 100×30 and 170×48 prove:

- page 1 `1-20 of 45 · Page 1 of 3`, row 20 reachable, pager fixed;
- page 2 `21-40`, final page `41-45`, boundary disabled reason visible;
- full-source search finds only the page-3 marker and clear restores page 1;
- current-page selection clears with notice when paging;
- off-page stable-ID navigation opens the deterministic owning page;
- a controlled recoverable failure retains last good rows and Retry works;
- focus follows the invoking/opposite pager contract.

Capture synthetic-only evidence under a task-specific scratch/evidence directory. Stop the TUI cleanly and prove fingerprints unchanged.

- [x] **Step 4: Update user documentation and task notes**

Document the 20-item Conversation convention, exact range, full-source filter, current-page selection clearing, disabled reasons, and Retry behavior. In the task file, add concise Implementation Notes listing approach, decisions, modified files, ADR-067, automated/mutation/geometry/live evidence, and any plan deviations.

- [x] **Step 5: Run final static and repository gates**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Library/library_pager_state.py tldw_chatbook/Library/library_conversations_state.py tldw_chatbook/Chat/chat_conversation_service.py tldw_chatbook/Chat/chat_conversation_scope_service.py tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_conversations_canvas.py Tests/Library/test_library_pager_state.py Tests/Library/test_library_conversations_state.py Tests/Library/test_library_conversations_visibility.py Tests/DB/test_search_conversations_fts.py Tests/Chat/test_chat_conversation_service.py Tests/UI/test_library_shell.py Tests/UI/test_library_multiselect_conversations.py Tests/UI/test_destination_shells.py
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=short
```

Expected: Ruff and diff checks pass; full suite passes. If full suite is red, stop closeout and follow the exact-base comparison rule above.

- [x] **Step 6: Review, close task hygiene, and commit docs**

Request independent spec and quality reviews. Resolve every Critical/Important finding and rerun affected gates. Only after all Definition-of-Done requirements are proven:

```bash
backlog task edit 18912 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7
backlog task edit 18912 -s Done --notes "Implemented source-owned exact Conversation paging per ADR-067; see implementation notes and recorded automated, mutation, geometry, privacy, and isolated live evidence."
git add Docs/User_Guide/library/media-and-conversations.md "backlog/tasks/task-18912 - Standardize-Library-pager-display-and-harden-Conversation-paging.md"
git commit -m "docs(library): close conversation pagination task"
```

Expected: every AC checked, Implementation Notes present, TASK-18912 status Done, worktree clean, and TASK-18913 through TASK-18916 remain independent To Do tasks ready for their own plans.
