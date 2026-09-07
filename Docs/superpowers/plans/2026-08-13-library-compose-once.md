# Library Compose-Once-Per-Visit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first Library composition authoritative and reconcile every automatically started entry result through retained rail/canvas owners with zero whole-screen recomposes.

**Architecture:** `LibraryScreen` remains the entry-lifecycle and route authority. A focused snapshot-cache module validates and clones the app-scoped cache before composition, while a strict screen-owned reconciliation router tracks state generation, rendered generation, dirtiness, and route identity. Existing canvas `sync_state()` seams plus retained landing/handoff owners project changes below the screen boundary; route-specific entry workers use the same strict contract.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich, pytest, pytest-asyncio, Ruff, Backlog.md.

## Global Constraints

- During the automatic entry lifecycle, `LibraryScreen.compose_content()` is called exactly once per visit.
- Destination construction, mount, cache seed, source timeout, fresh reconciliation, retry, and automatically launched route workers must never call `LibraryScreen.refresh(recompose=True)` or `LibraryScreen.recompose()`.
- Canvas-owned `sync_state()` may recompose only the retained canvas that owns the changed presentation.
- User-initiated interactions remain out of scope, but they supersede stale entry-route ownership.
- A valid cache is cloned and applied after constructor field initialization and before `restore_state()`/first composition.
- Cache validation recognizes Notes/Media/Conversations record tuples, Prompts `(count, ())`, and Skills `(count, context_payload)`.
- No mutable container reachable through the known snapshot schema may be shared between the app cache and a screen.
- Equal fresh data performs zero DOM work only when rendered generation equals state generation and reconciliation is not dirty.
- A failed strict update marks the generation dirty; an equal later result may retry it.
- Targeted updates preserve semantic focus, selection, and scroll when the same enabled target remains.
- Deferred work rechecks screen attachment, generation, and route identity before touching the DOM.
- DOM remove/mount operations stay on the Textual message pump, not in an independent worker group.
- Preserve current cache TTL, source queries, page sizes, service ownership, cancellation groups, copy, and screen-per-visit navigation.
- Do not add dependencies, persistence, schema changes, or a new service boundary.
- Render evidence must inspect the compositor/exported frame, not only widget state or geometry.
- UAT uses a scratch `TLDW_CONFIG_PATH` whose `[paths].data_dir` is also scratch, plus before/after fingerprints of the real profile.
- Use the parent `dev` worktree virtual environment for pytest commands.
- ADR required: no.
- ADR path: N/A.
- ADR reason: this applies existing cache, screen/rail/canvas ownership, and targeted-update contracts without changing a durable or cross-module boundary.

## File Structure

- Create `tldw_chatbook/UI/Library_Modules/library_snapshot_cache.py`: snapshot type aliases, schema validation, and copy-on-read/copy-on-write cloning.
- Create `tldw_chatbook/Widgets/Library/library_entry_canvases.py`: retained Landing and Study handoff owners with in-place state synchronization.
- Modify `tldw_chatbook/Widgets/Library/library_collections_panel.py`: add complete panel `sync_state()` for entry-result projection.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: pre-compose cache seed, strict generation/router state, semantic focus capture, structural canvas replacement, and entry-worker routing.
- Create `Tests/Library/test_library_snapshot_cache.py`: pure schema, malformed-input, and nested-alias tests.
- Create `Tests/UI/test_library_entry_compose_once.py`: mounted lifecycle, entry-worker, identity, focus/scroll, race, retry, compositor, and latency evidence.
- Modify `Tests/UI/test_library_shell.py`: retain existing cache/restored-route contracts while removing assertions that depend on a screen-level recompose.
- Modify `Tests/UI/test_library_canvas_sync_defects.py`: extend portable focus/scroll coverage from Notes to snapshot-owned canvases.
- Modify `backlog/tasks/task-15459 - Library---compose-once-per-visit.md`: implementation plan link, completed criteria, verification evidence, ADR disposition, and implementation notes.

---

### Task 1: Schema-Aware Cache and Pre-Compose Seed

**Files:**

- Create: `tldw_chatbook/UI/Library_Modules/library_snapshot_cache.py`
- Create: `Tests/Library/test_library_snapshot_cache.py`
- Create: `Tests/UI/test_library_entry_compose_once.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py:6434-6680`

**Interfaces:**

- Produces: `LibrarySourceSnapshot`, the six-field snapshot alias consumed by `LibraryScreen`.
- Produces: `clone_library_source_snapshot(snapshot: object) -> LibrarySourceSnapshot | None`.
- Produces: `LibraryScreen._seed_local_source_snapshot_from_cache(*, now: float | None = None) -> bool`.
- Consumes: `LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS`, app attributes `_library_source_snapshot_cache` and `_library_source_snapshot_cache_stamp`, and `_apply_local_source_snapshot(..., schedule_reconcile=False)` from Task 2's final signature. Until Task 2 lands, add the keyword with a default of `True` and keep existing mounted behavior.

- [ ] **Step 1: Verify no parallel implementation has landed since the design checkpoint**

Run from the isolated worktree:

```powershell
gh pr list --state all --search "15459" --json number,title,state,headRefName
git fetch -q origin
git log --oneline (git merge-base origin/dev HEAD)..origin/dev -- `
  tldw_chatbook/UI/Screens/library_screen.py `
  Tests/UI/test_library_shell.py `
  backlog/tasks/'task-15459 - Library---compose-once-per-visit.md'
```

Expected: no independent TASK-15459 implementation or overlapping upstream change. If one exists, stop execution and reconcile it before editing.

- [ ] **Step 2: Record the pre-change repeat-visit baseline**

Add the harness below to `Tests/UI/test_library_entry_compose_once.py`. It records calls at the requirement boundary rather than inferring them from child mounts:

```python
from __future__ import annotations

import statistics
import time

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)


@pytest.mark.asyncio
async def test_warm_repeat_visit_composes_once_before_fresh_reconcile(monkeypatch):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    calls: list[LibraryScreen] = []
    original = LibraryScreen.compose_content

    def counted_compose(screen):
        calls.append(screen)
        yield from original(screen)

    monkeypatch.setattr(LibraryScreen, "compose_content", counted_compose)
    samples: list[float] = []
    revisits: list[LibraryScreen] = []
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        first = _active_library_screen(host)
        await _wait_for_library_shell(first, pilot)
        await host.pop_screen()
        await pilot.pause()
        for _ in range(5):
            revisit = LibraryScreen(app)
            revisits.append(revisit)
            started = time.perf_counter()
            await host.push_screen(revisit)
            await _wait_for_library_shell(revisit, pilot)
            samples.append((time.perf_counter() - started) * 1000)
            await host.pop_screen()
            await pilot.pause()
    print(
        f"warm_visit_median_ms={statistics.median(samples):.3f} "
        f"min_ms={min(samples):.3f} max_ms={max(samples):.3f} n={len(samples)}"
    )
    assert all(calls.count(revisit) == 1 for revisit in revisits)
```

Run:

```powershell
& (Resolve-Path ..\..\.venv\Scripts\python.exe) -m pytest -p no:cacheprovider `
  Tests/UI/test_library_entry_compose_once.py::test_warm_repeat_visit_composes_once_before_fresh_reconcile `
  -vv -s
```

Expected on the pre-change implementation: FAIL because the second screen composes more than once; preserve the printed timing and compose count in the task evidence.

- [ ] **Step 3: Write failing pure cache-schema and deep-isolation tests**

Add tests that cover both special entries and nested mutable records:

```python
from tldw_chatbook.UI.Library_Modules.library_snapshot_cache import (
    clone_library_source_snapshot,
)


def _snapshot():
    return (
        {
            "notes": ({"id": "n1", "meta": {"tags": ["a"]}},),
            "media": ({"id": "m1"},),
            "conversations": ({"id": "c1"},),
            "prompts": (2, ()),
            "skills": (
                1,
                {
                    "available_skills": [{"name": "alpha", "tags": ["safe"]}],
                    "blocked_skills": [],
                },
            ),
        },
        {"notes": 1, "media": 1, "conversations": 1},
        {"notes": True, "media": True, "conversations": True},
        None,
        None,
        {"study_decks": 1, "flashcards_due": 2, "quizzes": 3},
    )


def test_clone_accepts_real_prompt_and_skill_shapes_without_aliasing():
    original = _snapshot()
    cloned = clone_library_source_snapshot(original)
    assert cloned == original
    assert cloned is not original
    assert cloned[0] is not original[0]
    assert cloned[0]["notes"][0] is not original[0]["notes"][0]
    assert cloned[0]["skills"][1] is not original[0]["skills"][1]
    assert cloned[0]["skills"][1]["available_skills"] is not original[0]["skills"][1]["available_skills"]


@pytest.mark.parametrize(
    "malformed",
    [None, (), ({},), ({"notes": []}, {}, {}, None, None, {})],
)
def test_clone_rejects_malformed_outer_or_source_shapes(malformed):
    assert clone_library_source_snapshot(malformed) is None
```

Run:

```powershell
& (Resolve-Path ..\..\.venv\Scripts\python.exe) -m pytest -p no:cacheprovider `
  Tests/Library/test_library_snapshot_cache.py -vv
```

Expected: FAIL at collection because `library_snapshot_cache` does not exist.

- [ ] **Step 4: Implement the schema-aware clone helper**

Create `library_snapshot_cache.py` with the exact public boundary below. Use `copy.deepcopy` only after validating the known outer/source shapes, so malformed arbitrary objects are never traversed as snapshots:

```python
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

LibrarySourceSnapshot = tuple[
    dict[str, tuple[Any, ...]],
    dict[str, int],
    dict[str, bool],
    str | None,
    Any,
    dict[str, int | None],
]

_RECORD_SOURCES = ("notes", "media", "conversations")


def clone_library_source_snapshot(snapshot: object) -> LibrarySourceSnapshot | None:
    if not isinstance(snapshot, tuple) or len(snapshot) != 6:
        return None
    records, counts, total_known, lookup_error, recovery_state, study_counts = snapshot
    if not all(isinstance(value, Mapping) for value in (records, counts, total_known, study_counts)):
        return None
    if any(not isinstance(records.get(source), tuple) for source in _RECORD_SOURCES):
        return None
    if any(not isinstance(record, Mapping) for source in _RECORD_SOURCES for record in records[source]):
        return None
    if any(not isinstance(counts.get(source), int) for source in _RECORD_SOURCES):
        return None
    if any(not isinstance(total_known.get(source), bool) for source in _RECORD_SOURCES):
        return None
    if any(
        study_counts.get(key) is not None and not isinstance(study_counts.get(key), int)
        for key in ("study_decks", "flashcards_due", "quizzes")
    ):
        return None
    if lookup_error is not None and not isinstance(lookup_error, str):
        return None
    prompts = records.get("prompts")
    skills = records.get("skills")
    if not isinstance(prompts, tuple) or len(prompts) != 2 or not isinstance(prompts[1], tuple):
        return None
    if not isinstance(skills, tuple) or len(skills) != 2 or not isinstance(skills[1], Mapping):
        return None
    if prompts[0] is not None and not isinstance(prompts[0], int):
        return None
    if skills[0] is not None and not isinstance(skills[0], int):
        return None
    cloned = copy.deepcopy(snapshot)
    return (
        dict(cloned[0]),
        dict(cloned[1]),
        dict(cloned[2]),
        cloned[3],
        cloned[4],
        dict(cloned[5]),
    )
```

- [ ] **Step 5: Move cache application before composition**

In `LibraryScreen.__init__`, initialize every snapshot/router field first, then call:

```python
self._seed_local_source_snapshot_from_cache()
```

Add:

```python
def _seed_local_source_snapshot_from_cache(self, *, now: float | None = None) -> bool:
    stamp = getattr(self.app_instance, "_library_source_snapshot_cache_stamp", None)
    if not isinstance(stamp, (int, float)):
        return False
    age = (time.monotonic() if now is None else now) - float(stamp)
    if age < 0 or age >= LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS:
        return False
    snapshot = clone_library_source_snapshot(
        getattr(self.app_instance, "_library_source_snapshot_cache", None)
    )
    if snapshot is None:
        return False
    self._apply_local_source_snapshot(*snapshot, schedule_reconcile=False)
    return True
```

Delete the cache read/apply and explicit `self.refresh(recompose=True)` block from `on_mount()`. On successful fresh lookup, write `clone_library_source_snapshot(snapshot)` to the app cache and never store the screen's live containers.

- [ ] **Step 6: Pin constructor/restore ordering and third-visit isolation**

Extend `test_library_entry_compose_once.py` with a cache seeded before screen construction. Assert `_library_loaded` and cached counts before mounting, call `restore_state()`, and prove restored filter/selection wins. Extend the pure cache test by mutating nested Note and Skills lists in the second clone and asserting a third clone still equals the original.

Run:

```powershell
& (Resolve-Path ..\..\.venv\Scripts\python.exe) -m pytest -p no:cacheprovider `
  Tests/Library/test_library_snapshot_cache.py `
  Tests/UI/test_library_shell.py -k "snapshot_cache or repeat_visit or restore_state" `
  Tests/UI/test_library_entry_compose_once.py::test_warm_repeat_visit_composes_once_before_fresh_reconcile `
  -vv -s
```

Expected: pure cache and ordering tests PASS; the compose-once test remains RED only for the fresh mounted reconcile, which Task 2 removes.

- [ ] **Step 7: Run scoped Ruff and commit Task 1**

```powershell
& 'C:\Python312\Scripts\ruff.exe' check `
  tldw_chatbook/UI/Library_Modules/library_snapshot_cache.py `
  Tests/Library/test_library_snapshot_cache.py `
  Tests/UI/test_library_entry_compose_once.py
git diff --check
git add -- `
  tldw_chatbook/UI/Library_Modules/library_snapshot_cache.py `
  tldw_chatbook/UI/Screens/library_screen.py `
  Tests/Library/test_library_snapshot_cache.py `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_shell.py
git commit -m "perf(library): seed source cache before compose"
```

---

### Task 2: Strict Generation-Gated Snapshot Reconciliation

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py:1318-1504, 6229-6425`
- Modify: `Tests/UI/test_library_entry_compose_once.py`
- Modify: `Tests/UI/test_library_shell.py:23648-23735`

**Interfaces:**

- Produces: `LibraryEntryReconcileResult(Enum)` with `APPLIED`, `ALREADY_CURRENT`, `SUPERSEDED`, and `FAILED`.
- Produces: `LibraryScreen._library_entry_route_key() -> tuple[object, ...]`.
- Produces: `_sync_library_canvas(..., allow_screen_fallback: bool = True) -> bool`.
- Produces: `LibraryScreen._schedule_library_entry_reconcile(generation: int, route_key: tuple[object, ...]) -> None`.
- Produces: `async LibraryScreen._reconcile_library_entry_state(generation: int, route_key: tuple[object, ...]) -> LibraryEntryReconcileResult`.
- Produces: `LibraryScreen._complete_library_entry_reconcile(generation: int, route_key: tuple[object, ...]) -> None`.
- Changes: `_apply_local_source_snapshot(..., schedule_reconcile: bool = True) -> bool`, returning whether normalized presentation changed.

- [ ] **Step 1: Write failing changed/equal/dirty generation tests**

Use a mounted Conversations route and record the screen/rail/host/canvas identities. Patch both whole-screen APIs to append calls after first paint rather than raising inside broad exception handlers:

```python
refresh_calls: list[bool] = []
recompose_calls: list[LibraryScreen] = []

original_refresh = LibraryScreen.refresh
original_recompose = LibraryScreen.recompose

def recorded_refresh(screen, *regions, **kwargs):
    refresh_calls.append(bool(kwargs.get("recompose")))
    return original_refresh(screen, *regions, **kwargs)

async def recorded_recompose(screen):
    recompose_calls.append(screen)
    return await original_recompose(screen)
```

Add three assertions:

1. A changed snapshot updates rail and Conversations canvas while all four captured owners retain identity.
2. An equal clean snapshot updates cache timestamp but calls neither canvas `sync_state()` nor either screen recompose API.
3. Marking `_library_entry_reconcile_dirty = True` makes the same equal snapshot call targeted sync once and clear dirtiness.

Run the three node IDs. Expected: FAIL because state/rendered generations and strict routing do not exist.

- [ ] **Step 2: Make the shared canvas helper strict-capable**

Change the helper boundary without changing legacy callers. Retain the current
kind dispatch and state construction verbatim, initialize `canvas: Widget | None
= None` before its `try`, return `True` immediately after the existing
`canvas.sync_state(...)`, and replace the exception tail with:

```python
except Exception:
    logger.opt(exception=True).debug(f"Library {kind} canvas sync failed.")
    if allow_screen_fallback:
        screen.refresh(recompose=True)
        if then is not None:
            screen.call_after_refresh(then)
    elif then is not None and isinstance(canvas, PostRecomposeCallback):
        canvas.queue_after_recompose(None)
    return False
```

Keep `allow_screen_fallback=True` as the default so out-of-scope user-interaction callers retain current behavior. Every entry-lifecycle caller added by this plan passes `False` explicitly.

- [ ] **Step 3: Add state/rendered generations and a route key**

Initialize these class-safe fields in `__init__` before cache seeding:

```python
self._library_snapshot_state_generation = 0
self._library_snapshot_rendered_generation = 0
self._library_entry_reconcile_dirty = False
self._library_entry_reconcile_pending: tuple[int, tuple[object, ...]] | None = None
self._library_entry_reconcile_retry_generation: int | None = None
```

Implement a route key containing the fields that select the mounted owner:

```python
def _library_entry_route_key(self) -> tuple[object, ...]:
    return (
        self._library_selected_row_id,
        self._library_notes_source,
        self._library_notes_view,
        self._library_media_view,
        self._library_prompts_view,
        self._library_skills_view,
        self._selected_note_id,
        self._selected_media_id,
        self._selected_prompt_id,
        self._selected_skill_name,
    )
```

- [ ] **Step 4: Split state mutation from mounted projection**

Normalize and compare all six snapshot fields before assignment. On changed presentation, increment state generation and set `_library_entry_reconcile_dirty = True` before scheduling projection. On a clean equal result, return `False` after cache freshness is written. On a dirty equal result, retain the generation and schedule repair. Constructor cache seeding passes `schedule_reconcile=False` and sets rendered generation equal to state generation because the first compose will project that state.

Schedule DOM work on the screen pump:

```python
def _schedule_library_entry_reconcile(
    self, generation: int, route_key: tuple[object, ...]
) -> None:
    pending = (generation, route_key)
    if self._library_entry_reconcile_pending == pending:
        return
    self._library_entry_reconcile_pending = pending
    self.call_later(self._reconcile_library_entry_state, generation, route_key)
```

Do not use `run_worker` for the remove/mount portion; Pilot's message-pump drain does not wait for independent worker DOM mutations.

- [ ] **Step 5: Implement strict rail/header/list routing**

The reconcile method must first reject detached, stale-generation, and stale-route work. Build shell state once, call `LibraryRail.sync_state()`, update `#library-header-line` only when text differs, and route snapshot-owned list surfaces through `_sync_library_canvas(..., allow_screen_fallback=False)`. Search/RAG keeps its current yield-free scope/run-gate synchronizer. Rail-only surfaces return `APPLIED` after rail/header sync.

For direct in-place patches with no child recompose, complete immediately:

```python
def _complete_library_entry_reconcile(
    self, generation: int, route_key: tuple[object, ...]
) -> None:
    if generation != self._library_snapshot_state_generation:
        return
    if route_key != self._library_entry_route_key():
        return
    self._library_snapshot_rendered_generation = generation
    self._library_entry_reconcile_dirty = False
    self._library_entry_reconcile_pending = None
```

For canvas-owned `sync_state()`, keep reconciliation dirty and queue
`_complete_library_entry_reconcile(...)` through that canvas's
`PostRecomposeCallback`. Returning from `sync_state()` means the recompose was
requested, not that its children rendered. Structural remove/mount paths await
mount and post-mount reseed before calling the completion method. This prevents an
asynchronous canvas failure or teardown from falsely marking stale DOM clean.

On a still-current missing target, schedule one `call_later` retry for the generation. On the second failure, set dirty, clear pending/retry markers, and return `FAILED`. Route changes return `SUPERSEDED` without logging an error.

- [ ] **Step 6: Update the old Notes churn regression to assert the new contract**

`test_library_note_recompose_and_fifty_route_cycles_return_to_baseline` currently applies the exact same snapshot five times and waits for a replaced `TextArea`. Replace that loop with:

```python
prior = screen.query_one("#library-note-body", TextArea)
for _ in range(5):
    screen._apply_local_source_snapshot(
        dict(screen._local_source_records),
        dict(screen._local_source_counts),
        dict(screen._local_source_total_known),
        screen._library_lookup_error,
        screen._library_lookup_recovery_state,
        dict(screen._library_study_counts),
    )
    await pilot.pause()
    assert screen.query_one("#library-note-body", TextArea) is prior
```

Keep the rest of the ownership/timer assertions unchanged. This turns the test from pinning the removed bug into pinning the equal-clean no-op.

- [ ] **Step 7: Run Task 2 tests and mutation-check both forbidden APIs**

```powershell
& (Resolve-Path ..\..\.venv\Scripts\python.exe) -m pytest -p no:cacheprovider `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_shell.py -k "snapshot_cache or source_snapshot or recompose_and_fifty" `
  Tests/UI/test_library_canvas_scoped_sync.py `
  Tests/UI/test_library_canvas_sync_defects.py `
  -vv -s
```

Expected: PASS. Mutation-check by temporarily restoring `self.refresh(recompose=True)` in the changed-snapshot path and then `await self.recompose()` in the deferred path; the whole-screen API spies must fail for each mutation. Restore product code and rerun the same nodes to PASS.

- [ ] **Step 8: Run scoped Ruff and commit Task 2**

```powershell
& 'C:\Python312\Scripts\ruff.exe' check `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_canvas_sync_defects.py
& 'C:\Python312\Scripts\ruff.exe' check `
  tldw_chatbook/UI/Screens/library_screen.py `
  Tests/UI/test_library_shell.py `
  --ignore E721,F401
git diff --check
git add -- `
  tldw_chatbook/UI/Screens/library_screen.py `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_shell.py
git commit -m "perf(library): reconcile snapshots below screen"
```

---

### Task 3: Retained Landing/Handoff Owners and Portable Focus

**Files:**

- Create: `tldw_chatbook/Widgets/Library/library_entry_canvases.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:7686-7754, 7895-8430`
- Modify: `Tests/UI/test_library_entry_compose_once.py`
- Modify: `Tests/UI/test_library_canvas_sync_defects.py`

**Interfaces:**

- Produces: frozen `LibraryLandingRecentItem`, `LibraryLandingCanvasState`, and `LibraryStudyHandoffCanvasState` dataclasses.
- Produces: `LibraryLandingCanvas.sync_state(state: LibraryLandingCanvasState) -> None`.
- Produces: `LibraryStudyHandoffCanvas.sync_state(state: LibraryStudyHandoffCanvasState) -> None`.
- Produces: frozen `LibraryEntryFocusIdentity(widget_id: str = "", source_id: str = "", scroll_offset: tuple[int, int] | None = None)`.
- Produces: `LibraryScreen._capture_library_entry_focus() -> LibraryEntryFocusIdentity | None` and `_restore_library_entry_focus(identity, *, generation, route_key) -> None`.
- Produces: `LibraryScreen._finish_library_entry_canvas_sync(identity, *, generation, route_key) -> None`, the single post-recompose callback that restores focus/scroll and then marks the generation rendered.

- [ ] **Step 1: Write failing retained-owner tests**

Mount Landing with counts/recents, capture the three action Buttons, focus Search, apply a changed source snapshot, and assert:

```python
assert screen.query_one("#library-hub-action-import") is import_button
assert screen.query_one("#library-hub-action-search") is search_button
assert screen.query_one("#library-hub-action-new-note") is new_note_button
assert screen.focused is search_button
assert "Conversations (2)" in str(screen.query_one("#library-hub-counts").renderable)
```

Repeat on Study handoff, capturing its Open button while readiness/context changes. Assert the compositor paints the updated count/readiness copy and that the button remains the same object.

Expected: FAIL because Landing and handoff content are currently composed inline by the screen and have no `sync_state()` owner.

- [ ] **Step 2: Implement retained entry canvases**

Create state dataclasses containing only display data and stable dispatch identities. `LibraryLandingCanvas.compose()` mounts the three static actions once, a counts `Static`, and a recents container. `sync_state()` updates counts and replaces only recent-row children. `LibraryStudyHandoffCanvas.compose()` always mounts purpose, context, owner, recovery, and Open action; absent context uses `display=False` instead of omitting the widget. Its `sync_state()` updates text, display, and recovery classes without replacing the Open button.

The state synchronization boundaries are:

```python
def sync_state(self, state: LibraryLandingCanvasState) -> None:
    self.state = state
    self.query_one("#library-hub-counts", Static).update(state.counts_line)
    self.call_later(self._replace_recent_rows)


def sync_state(self, state: LibraryStudyHandoffCanvasState) -> None:
    self.state = state
    self.query_one("#library-study-handoff-purpose", Static).update(state.purpose)
    context = self.query_one("#library-study-handoff-context", Static)
    context.update(state.context)
    context.display = bool(state.context)
    self.query_one("#library-study-handoff-owner", Static).update(state.owner)
    recovery = self.query_one("#library-study-handoff-recovery", Static)
    recovery.update(state.recovery)
    recovery.set_class(state.blocked, "ds-recovery-callout")
    recovery.set_class(state.blocked, "is-blocked")
```

`_replace_recent_rows()` re-reads `self.state.recent_items` after its first await
and once again after mounting. Multiple queued calls therefore converge on the
latest state instead of mounting an older tuple captured before the yield.

Build both state objects from existing `_hub_counts_line()`, `_hub_recent_items()`, and `_study_handoff_copy()` output so copy and source ordering remain unchanged.

- [ ] **Step 3: Replace inline Landing/handoff composition with the retained widgets**

`compose_content()` yields one `LibraryLandingCanvas` or `LibraryStudyHandoffCanvas` child. Extend the strict router to query that exact owner and call `sync_state()`. Do not add a nested card or extra visible wrapper; preserve current IDs/classes on the child controls so event handlers and CSS continue to resolve.

- [ ] **Step 4: Extend semantic focus/scroll capture across list canvases**

Capture the currently focused descendant before any strict canvas sync. Use a stable widget ID when present; for row buttons, store their existing record/source identity attribute. Capture the active canvas/scroll region's `(scroll_x, scroll_y)` when supported. Queue restoration with the canvas's `PostRecomposeCallback`, not `screen.call_after_refresh`.

Restoration must verify:

```python
if generation != self._library_snapshot_state_generation:
    return
if route_key != self._library_entry_route_key():
    return
if self.focused is not outgoing_focus and self.focused is not None:
    return
```

Then resolve the same enabled semantic target and restore scroll with `animate=False`. Notes continues to compose its richer existing identity with this portable outer contract.

Queue exactly one callback on the canvas so `PostRecomposeCallback`'s replace-latest semantics cannot make focus restoration and rendered-generation completion displace each other:

```python
def _finish_library_entry_canvas_sync(
    self,
    identity: LibraryEntryFocusIdentity | None,
    *,
    generation: int,
    route_key: tuple[object, ...],
) -> None:
    if identity is not None:
        self._restore_library_entry_focus(
            identity, generation=generation, route_key=route_key
        )
    self._complete_library_entry_reconcile(generation, route_key)
```

Pass this composed callback as `_sync_library_canvas(..., then=finish,
allow_screen_fallback=False)`. The helper's existing Notes callback composition
runs the richer Notes restore first and this completion second.

- [ ] **Step 5: Add Conversations/Media/Prompts/Skills focus and scroll regressions**

Parameterize one test over the four canvases. Focus a stable filter/action/row, set a nonzero scroll offset where the canvas owns scrolling, apply changed entry data, wait for the canvas's own post-recompose callback, and assert focus/scroll and screen/canvas identity. Add a missing-row arm proving focus is not yanked to an unrelated replacement.

Run:

```powershell
& (Resolve-Path ..\..\.venv\Scripts\python.exe) -m pytest -p no:cacheprovider `
  Tests/UI/test_library_entry_compose_once.py -k "landing or handoff or focus or scroll" `
  Tests/UI/test_library_canvas_sync_defects.py `
  Tests/UI/test_library_canvas_scoped_sync.py `
  -vv
```

Expected: PASS.

- [ ] **Step 6: Render-check compact and wide owner geometry**

At `(60, 20)` and `(170, 48)`, export the SVG or inspect `screen._compositor.render_strips()`. Assert the three Landing actions or Study Open action are painted inside the viewport before and after synchronization. Widget existence alone is not the oracle.

- [ ] **Step 7: Run Ruff and commit Task 3**

```powershell
& 'C:\Python312\Scripts\ruff.exe' check `
  tldw_chatbook/Widgets/Library/library_entry_canvases.py `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_canvas_sync_defects.py
& 'C:\Python312\Scripts\ruff.exe' check `
  tldw_chatbook/UI/Screens/library_screen.py --ignore E721,F401
git diff --check
git add -- `
  tldw_chatbook/Widgets/Library/library_entry_canvases.py `
  tldw_chatbook/UI/Screens/library_screen.py `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_canvas_sync_defects.py
git commit -m "perf(library): retain entry canvas controls"
```

---

### Task 4: Route Every Automatic Entry Worker In Place

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_collections_panel.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:5256-5405, 9009-9370, 9655-9780, 25587-25602, 26519-26632`
- Modify: `Tests/UI/test_library_entry_compose_once.py`
- Modify: `Tests/UI/test_library_shell.py:16744-17218`

**Interfaces:**

- Produces: `LibraryCollectionsPanel.sync_state(state, *, name_value: str, description_value: str, delete_pending: bool) -> None`.
- Produces: `async LibraryScreen._replace_library_canvas_child(widget: Widget, *, generation: int, route_key: tuple[object, ...]) -> LibraryEntryReconcileResult`.
- Changes: `LibraryScreen._open_library_item_by_id(source_type: str, record_id: str, *, entry_origin: bool = False) -> None`.
- Changes: `LibraryScreen._open_pending_library_source()` passes `entry_origin=True`.

- [ ] **Step 1: Write a parameterized entry-worker compose-count matrix**

Add mounted cases for restored/deep-linked Prompts list, Collections, Skills list, Notes editor, Media viewer, Export, and pending opens for media/notes/conversation/prompt. Each case must:

1. arrange the initial route before mount;
2. capture the first screen/rail/host/active-owner identities;
3. wait on the worker's actual terminal state with a bounded condition, not only `pilot.pause()`;
4. assert `calls.count(screen) == 1`;
5. assert no recorded `refresh(recompose=True)` or `recompose()` call; and
6. assert only the route-owned child changes when loading structurally transitions.

Run the matrix. Expected: FAIL for Prompt browse, Collections, Skills trust posture, Media detail, and pending-open branches that still call whole-screen refresh/recompose.

- [ ] **Step 2: Convert Prompt and Skills completion paths**

Change `_sync_library_prompts_browse_result()` and `_load_library_skills_trust_posture()` to call `_sync_library_canvas(..., allow_screen_fallback=False)` only while their captured route identity is still current. Prompt loading and terminal results use the controller's existing request token as the stale-result authority; do not add a parallel token.

- [ ] **Step 3: Add complete Collections panel synchronization**

`LibraryCollectionsPanel.sync_state()` copies every compose input before `refresh(recompose=True)` on the panel itself:

```python
def sync_state(
    self,
    state: LibraryCollectionsPanelState,
    *,
    name_value: str,
    description_value: str,
    delete_pending: bool,
) -> None:
    self.state = state
    self.name_value = name_value
    self.description_value = description_value
    self.delete_pending = delete_pending
    self.refresh(recompose=True)
```

Change `_sync_collections_panel()` to query the retained panel and call this method. Entry lifecycle calls must not use `await self.recompose()` or `self.refresh(recompose=True)`.

- [ ] **Step 4: Convert restored media/note detail and Export completion**

Export already patches its always-mounted fields; add the route/generation guard and keep that path. Notes continues through `_sync_library_canvas("notes", allow_screen_fallback=False)`.

For Media, extract the existing viewer-construction branch into `_build_library_media_active_child() -> Widget`, and use `_replace_library_canvas_child()` to replace only the loading child with `LibraryMediaViewer` or the list canvas. The helper hides/removes old children, mounts the new child, then re-reads current state and synchronizes once more after mount to close the state-read-before-await gap.

- [ ] **Step 5: Make pending source opens strict without changing live actions**

Pass `entry_origin=True` only from `_open_pending_library_source()`. For that branch, replace each `self.refresh(recompose=True)`/`await self.recompose()` with the strict active-child replacement. For Conversations, do not call `_select_library_rail_row()` because its general user-interaction fallback may recompose the screen; set the same route fields, build shell state, apply rail/header selection, and mount/sync the Conversations canvas through the strict helper.

The ordinary Search/RAG Open action continues to call `_open_library_item_by_id(..., entry_origin=False)` and retains current user-interaction behavior in this task.

- [ ] **Step 6: Prove stale entry results cannot overwrite a new route**

Gate each representative service call, switch the screen route before releasing it, then assert the new route's owner and focus are unchanged. Cover at least Prompt token, Skills posture, Collections snapshot, Media detail, and one pending source open. The worker may settle state it owns, but its DOM result must return `SUPERSEDED`.

- [ ] **Step 7: Run the full entry-worker and companion route suites**

```powershell
& (Resolve-Path ..\..\.venv\Scripts\python.exe) -m pytest -p no:cacheprovider `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_shell.py -k "restored or deep_link or snapshot_cache or prompt or collections or skills or media_viewer or export" `
  Tests/UI/test_library_prompt_browse_controller.py `
  Tests/UI/test_library_prompt_collections.py `
  Tests/UI/test_library_skills_canvas.py `
  Tests/UI/test_library_media_side_by_side.py `
  -vv
```

Expected: PASS with no whole-screen compose calls in the entry matrix.

- [ ] **Step 8: Mutation-check each formerly broad worker and commit Task 4**

For Prompt, Collections, Skills, Media detail, and pending-open, temporarily restore its former whole-screen call one at a time. The corresponding parameterized case must fail on the exact API spy. Restore and rerun to PASS.

```powershell
& 'C:\Python312\Scripts\ruff.exe' check `
  tldw_chatbook/Widgets/Library/library_collections_panel.py `
  Tests/UI/test_library_entry_compose_once.py
& 'C:\Python312\Scripts\ruff.exe' check `
  tldw_chatbook/UI/Screens/library_screen.py `
  Tests/UI/test_library_shell.py `
  --ignore E721,F401
git diff --check
git add -- `
  tldw_chatbook/Widgets/Library/library_collections_panel.py `
  tldw_chatbook/UI/Screens/library_screen.py `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_shell.py
git commit -m "perf(library): update entry workers in place"
```

---

### Task 5: Race Closure, Performance Evidence, UAT, and Task Closeout

**Files:**

- Modify: `Tests/UI/test_library_entry_compose_once.py`
- Modify: `backlog/tasks/task-15459 - Library---compose-once-per-visit.md`
- Modify if a generalizable incident occurs: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

**Interfaces:**

- Consumes all Task 1-4 interfaces.
- Produces no new product API; this task proves, documents, and closes the agreed contract.

- [ ] **Step 1: Add deterministic mount/race/failure coverage**

Use `asyncio.Event` barriers, not sleeps, to cover:

- worker completion while Mount dispatch is still active;
- timeout state followed by fresh success;
- two changed generations where only the newer generation renders;
- route switch before a queued reconcile;
- first targeted failure, one next-turn retry, then dirty stop;
- equal data repairing a dirty generation; and
- detached/unmounted completion as a no-op.

Record reconcile results and target-sync calls after the fact. Do not raise sentinel assertions from inside broad `except Exception` paths.

- [ ] **Step 2: Prove the retry and equality guards are non-vacuous**

Mutation-check three guards independently:

1. remove the generation comparison, expecting the two-generation race to fail;
2. remove the dirty bypass, expecting equal-data repair to fail; and
3. allow a second retry to schedule, expecting the bounded retry-count assertion to fail.

Restore each mutation and rerun its node to PASS before continuing.

- [ ] **Step 3: Run the complete focused product-path matrix**

```powershell
& (Resolve-Path ..\..\.venv\Scripts\python.exe) -m pytest -p no:cacheprovider `
  Tests/Library/test_library_snapshot_cache.py `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_canvas_scoped_sync.py `
  Tests/UI/test_library_canvas_sync_defects.py `
  Tests/UI/test_library_shell.py `
  Tests/UI/test_library_prompt_browse_controller.py `
  Tests/UI/test_library_prompt_collections.py `
  Tests/UI/test_library_skills_canvas.py `
  Tests/UI/test_library_media_side_by_side.py `
  Tests/Widgets/Library/test_library_rail.py `
  Tests/Widgets/Library/test_library_conversations_canvas.py `
  Tests/Widgets/Library/test_library_notes_canvas.py `
  -vv
```

Expected: PASS. If the literal command hits a baseline Windows-only collection issue, rerun the exact failing modules against `origin/dev` and document byte-identical attribution; do not change TTS/media-playback platform modules.

- [ ] **Step 4: Record identical before/after warm-visit samples**

Run the warm-repeat benchmark test with at least five iterations in one process, reporting median, minimum, maximum, sample count, compose count, and screen/rail/host/canvas identities. Use the same seeded records and `LIBRARY_TEST_SIZE` as the Task 1 baseline. The post-change result must show one `compose_content()` call and zero whole-screen recompose requests.

- [ ] **Step 5: Run isolated rendered UAT at compact and wide sizes**

Create a scratch directory outside the real profile, write a parse-validated config with both `TLDW_CONFIG_PATH` and `[paths].data_dir` pointing into it, set `TLDW_TEST_MODE=1`, disable model-catalog networking, and fingerprint the real config/data roots before launch.

At `(60, 20)` and `(170, 48)`, drive:

- warm Landing repeat visit with fresh reconciliation;
- cold Conversations loading to rows;
- restored Media viewer detail;
- restored Prompts, Collections, Skills, Notes, and Export;
- focus held on an action/filter while fresh data lands; and
- one error/timeout-to-success transition.

For each, inspect the compositor/exported SVG for visible updated copy and controls, and record object identities plus compose/recompose counters. Revalidate scratch TOML after shutdown and compare real-profile fingerprints before deleting the recovery snapshot. Restore on unexplained drift and do not claim causality without a controlled reproduction.

- [ ] **Step 6: Run static checks and the full suite**

```powershell
& 'C:\Python312\Scripts\ruff.exe' check `
  tldw_chatbook/UI/Library_Modules/library_snapshot_cache.py `
  tldw_chatbook/Widgets/Library/library_entry_canvases.py `
  tldw_chatbook/Widgets/Library/library_collections_panel.py `
  Tests/Library/test_library_snapshot_cache.py `
  Tests/UI/test_library_entry_compose_once.py `
  Tests/UI/test_library_canvas_sync_defects.py
& 'C:\Python312\Scripts\ruff.exe' check `
  tldw_chatbook/UI/Screens/library_screen.py `
  Tests/UI/test_library_shell.py `
  --ignore E721,F401
git diff --check origin/dev...HEAD
& (Resolve-Path ..\..\.venv\Scripts\python.exe) -m pytest -q
```

Record unfiltered Ruff output for existing files before using the documented `E721,F401` baseline exclusions. Any new-line finding must be fixed. Attribute full-suite Windows collection failures only after reproducing the exact failures against unchanged `origin/dev` files.

- [ ] **Step 7: Self-review the complete branch**

Review `git diff origin/dev...HEAD` for:

- any entry-lifecycle call to either whole-screen recompose API;
- equality paths that can strand dirty rendered state;
- DOM work scheduled in an independent worker;
- state read before an awaited remove/mount without a post-mount reseed;
- focus restoration that can override intervening user focus;
- cache aliases in nested Note/Skills containers;
- stale worker results lacking generation/route/token guards; and
- tests that assert widget state without rendered-frame evidence where visibility is claimed.

Fix findings with a red regression first and rerun the affected focused slice.

- [ ] **Step 8: Close TASK-15459 manually and commit evidence**

Because the Backlog CLI silently mishandles five-digit task IDs, edit the task file directly. Check all four acceptance criteria, add concise Implementation Notes, include exact test/Ruff/UAT/performance evidence, and retain:

```text
ADR required: no
ADR path: N/A
Reason: Applies existing Library cache and retained screen/rail/canvas ownership contracts.
```

Add a lessons entry only if implementation produced a new evidenced, reusable incident. Then:

```powershell
git diff --check
git add -- `
  Tests/UI/test_library_entry_compose_once.py `
  backlog/tasks/'task-15459 - Library---compose-once-per-visit.md'
git commit -m "docs(library): close task 15459"
git status --short --branch
```

Expected: task status `Done`, four checked criteria, implementation notes and evidence present, and a clean worktree.

---

## Final Verification Checklist

- [ ] Cache is valid before first compose on warm visits.
- [ ] Every automatic entry route calls `compose_content()` exactly once.
- [ ] Both whole-screen recompose APIs remain unused throughout entry lifecycle work.
- [ ] Equal clean data performs zero DOM work.
- [ ] Dirty equal data repairs the rendered generation.
- [ ] Screen, rail, canvas host, and unaffected active owners retain identity.
- [ ] Semantic focus/selection/scroll survive when their target remains.
- [ ] Route/generation/token guards discard stale work.
- [ ] Compact and wide compositor evidence shows updated controls/copy.
- [ ] Warm-visit before/after measurements use identical data and dimensions.
- [ ] Focused suites, scoped Ruff, diff-check, and attributable full-suite results are recorded.
- [ ] Real profile fingerprints are unchanged after UAT.
- [ ] TASK-15459 is `Done` with AC 4/4, ADR disposition, and Implementation Notes.
