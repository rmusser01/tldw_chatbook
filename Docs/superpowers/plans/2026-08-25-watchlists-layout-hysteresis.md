# Watchlists Layout Hysteresis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize every Watchlists responsive pane boundary with Library-parity four-column hysteresis while preserving immediate manual actions, mode-local pane priority, focus, and saved preferences.

**Architecture:** Extend the pure Watchlists layout resolver with optional previous effective state and deterministic reopening. Keep a separate responsive baseline and mode-local priority lease in the screen controller, then derive Article Focus as an overlay before publishing only genuinely changed desired layouts to the existing workbench.

**Tech Stack:** Python 3.11+, Textual 8.x reactive widgets/messages, pytest with Textual Pilot, frozen dataclasses.

---

## Scope and file map

- `tldw_chatbook/UI/Watchlists_Modules/region_layout.py` owns pure nominal collapse and hysteretic reopening. It remains Textual-free.
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` owns width authority, responsive history, explicit transition causes, priority-lease lifecycle, Article Focus overlay, request suppression, and rollback snapshots.
- `Tests/Watchlists/test_watchlists_responsive_layout.py` proves the pure boundary policy in both directions, including priority-adjusted multi-pane order.
- `Tests/Watchlists/test_watchlists_collections_screen.py` proves the screen-width authority without mounting a full destination.
- `Tests/Watchlists/test_watchlists_scoped_rebuilds.py` proves mounted request, focus, Article Focus, priority parking, and rollback behavior.
- `Tests/Watchlists/test_watchlists_cold_open_layout.py` proves responsive history and the priority lease remain transient across app restarts.
- `backlog/tasks/task-22211 - Watchlists-responsive-layout-needs-hysteresis-at-its-collapse-boundaries.md` records the accepted plan, verification evidence, completed criteria, and implementation notes.

No new module or shared split-pane abstraction is introduced. Watchlists duplicates the semantic constant rather than importing Library internals; parity is behavioral and test-backed.

## ADR assessment

**ADR required:** no  
**ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`  
**Reason:** The change stabilizes ADR-042's existing preferred-versus-effective responsive policy and follows the accepted Library precedent. It does not change storage, ownership, dependencies, cross-module contracts, or the long-lived pane structure.

### Task 1: Add pure four-column hysteresis policy

**Files:**
- Modify: `Tests/Watchlists/test_watchlists_responsive_layout.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/region_layout.py:48-175`

- [ ] **Step 1: Extend the test helper and write failing boundary tests**

Update the local helper so tests can supply responsive history:

```python
def resolve(
    preferred: RegionLayout,
    width: int,
    *,
    read_mode: bool = True,
    article_focus: bool = False,
    priority_target: Region | None = None,
    previous: RegionLayout | None = None,
) -> RegionLayout:
    return region_layout.resolve_effective_layout(
        preferred,
        width=width,
        read_mode=read_mode,
        article_focus=article_focus,
        priority_target=priority_target,
        previous=previous,
    )
```

Add focused tests that assert:

```python
def test_read_inspector_boundary_has_four_column_hysteresis_both_directions():
    preferred = RegionLayout()
    collapsed = resolve(preferred, 144)
    assert collapsed.is_collapsed(Region.RIGHT_RAIL)

    for width in (145, 144, 146, 145, 147, 148):
        next_layout = resolve(preferred, width, previous=collapsed)
        assert next_layout == collapsed
        collapsed = next_layout

    opened = resolve(preferred, 149, previous=collapsed)
    assert not opened.is_collapsed(Region.RIGHT_RAIL)
    for width in (148, 147, 146, 145):
        opened = resolve(preferred, width, previous=opened)
        assert not opened.is_collapsed(Region.RIGHT_RAIL)
    assert resolve(preferred, 144, previous=opened).is_collapsed(
        Region.RIGHT_RAIL
    )
```

Add table-driven bidirectional sequences for every remaining representative
boundary. Each case must start open, collapse one column below its nominal threshold,
remain collapsed through `T + 3`, reopen at `T + 4`, remain open while shrinking
back to `T`, and recollapse at `T - 1`:

- Read Navigation: nominal `T=115`, buffered reopen at 119;
- Read Feed Items: nominal `T=91`, buffered reopen at 95;
- management Inspector: nominal `T=108`, buffered reopen at 112;
- management Navigation: nominal `T=78`, buffered reopen at 82.

Also add tests for:

- all-collapsed Read reopening at 95 (Feed Items), 119 (Navigation), and 149 (Inspector);
- all-collapsed management reopening at 82 (Navigation) and 112 (Inspector);
- a large width jump reopening every pane whose own buffered boundary was crossed;
- an active Inspector priority using the reverse of the adjusted collapse order;
- `previous=None` preserving every existing nominal breakpoint;
- manual preferred collapses and Article Focus remaining authoritative.

- [ ] **Step 2: Run the pure layout tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_watchlists_responsive_layout.py -q
```

Expected: FAIL because `resolve_effective_layout` does not accept `previous` and `LAYOUT_HYSTERESIS_WIDTH` does not exist.

- [ ] **Step 3: Implement the minimal pure resolver change**

Add the Watchlists-local semantic constant:

```python
LAYOUT_HYSTERESIS_WIDTH = 4
```

Add `previous: RegionLayout | None = None` as a keyword-only resolver argument. Preserve the existing nominal collapse loop, then return its result immediately when `previous is None`.

For a historical resolution:

```python
nominal_open = {region for region in mounted if region not in nominal_collapsed}
accepted_open = {
    region for region in nominal_open if not previous.is_collapsed(region)
}
for region in reversed(candidates):
    if region not in nominal_open or not previous.is_collapsed(region):
        continue
    candidate_open = accepted_open | {region}
    reopen_width = (
        CENTRE_COMFORT_WIDTH
        + len(mounted) * PANE_GRIP_WIDTH
        + sum(PANE_MINIMUM_WIDTHS[item] for item in candidate_open)
    )
    if width >= reopen_width + LAYOUT_HYSTERESIS_WIDTH:
        accepted_open.add(region)

return RegionLayout(collapsed=frozenset(set(mounted) - accepted_open))
```

Keep the Article Focus early return before hysteresis. Build `candidates` with the existing priority-target move before reversing it, so reopening uses the priority-adjusted order. Do not iterate a set when order affects the result.

- [ ] **Step 4: Run the pure layout tests and verify GREEN**

Run the same command from Step 2.

Expected: all tests in `test_watchlists_responsive_layout.py` PASS.

- [ ] **Step 5: Commit the pure policy**

```bash
git add Tests/Watchlists/test_watchlists_responsive_layout.py tldw_chatbook/UI/Watchlists_Modules/region_layout.py
git commit -m "fix(watchlists): add responsive layout hysteresis"
```

### Task 2: Establish invariant width authority and responsive controller history

**Files:**
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:18-22, 410-435, 1008-1030, 3038-3120`

- [ ] **Step 1: Write failing width-authority and no-op request tests**

In `test_watchlists_collections_screen.py`, use a minimal fake receiver with a Textual `Size` and call `_available_layout_width` unbound. Assert a positive `self.size.width` is returned without querying the workbench, while zero returns `None`:

```python
def test_layout_width_uses_only_positive_screen_allocation():
    receiver = SimpleNamespace(size=Size(145, 50))
    assert (
        WatchlistsCollectionsScreen._available_layout_width(receiver) == 145
    )
    receiver.size = Size(0, 50)
    assert WatchlistsCollectionsScreen._available_layout_width(receiver) is None
```

In `test_watchlists_scoped_rebuilds.py`, mount the screen, explicitly establish a 144-column responsive baseline, wrap `workbench.request_region_layout`, and feed 145–148 plus repeated ±1 widths through the passive resize path. Assert:

- `_responsive_region_layout` stays Inspector-collapsed;
- the request wrapper receives no calls inside the dead band;
- the mounted region identity and keyboard focus do not change;
- 149 produces exactly one request and one Inspector expansion;
- shrinking to 145 produces no request, while 144 produces exactly one collapse request.

Add a zero-width controller test that first calls recomputation with no responsive history, then after a positive baseline. In both zero-width cases assert no history change, no token increment, and no workbench request.

- [ ] **Step 2: Run the new controller tests and verify RED**

Run only the new node IDs:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Watchlists/test_watchlists_collections_screen.py::test_layout_width_uses_only_positive_screen_allocation \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py::test_resize_hysteresis_suppresses_sub_band_layout_requests \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py::test_zero_width_never_seeds_or_replaces_responsive_history \
  -q
```

Expected: FAIL because workbench width remains authoritative, responsive history is not separate, and equal effective layouts still allocate request tokens.

- [ ] **Step 3: Add explicit recompute causes and responsive baseline state**

Import `Literal` and define:

```python
LayoutRecomputeCause = Literal["initial", "resize", "explicit", "article_focus"]

@dataclass(frozen=True)
class ResponsivePriorityLease:
    target: Region
    read_mode: bool
```

Initialize:

```python
self._responsive_region_layout: RegionLayout | None = None
self._responsive_priority_lease: ResponsivePriorityLease | None = None
```

Replace `_available_layout_width` with the invariant authority:

```python
def _available_layout_width(self) -> int | None:
    """Return the positive screen allocation, never descendant content width."""
    width = self.size.width
    return width if width > 0 else None
```

Make every `_recompute_effective_layout` caller supply a required `cause` keyword. Use:

- `cause="initial"` from `on_mount`;
- `cause="resize"` from `on_resize`;
- `cause="article_focus"` from `action_article_focus`;
- `cause="explicit"` from preference changes, rollback, section changes, and section reconciliation fallback.

Inside recomputation:

1. Return immediately when width is `None`; do not mutate responsive/effective state or tokens.
2. For `article_focus`, preserve an existing `_responsive_region_layout`; only seed a nominal baseline if none exists.
3. For `resize`, pass `_responsive_region_layout` as `previous`; for `initial` and `explicit`, pass `None`.
4. Resolve and store the responsive baseline with `article_focus=False`.
5. Derive Article Focus as an effective overlay without replacing the responsive baseline.
6. Compare the result with the controller's current `_effective_region_layout`. Update the responsive baseline even when the visible overlay is unchanged, but return before item-state capture, token allocation, and workbench request if desired effective layout is equal.

- [ ] **Step 4: Run the new controller tests and relevant existing layout tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Watchlists/test_watchlists_collections_screen.py::test_layout_width_uses_only_positive_screen_allocation \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py::test_resize_hysteresis_suppresses_sub_band_layout_requests \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py::test_zero_width_never_seeds_or_replaces_responsive_history \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py::test_resize_derives_effective_layout_without_persisting_preference \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py::test_mounted_layout_cycles_preserve_complete_reader_and_list_state \
  -q
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit width authority and request suppression**

```bash
git add Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py
git commit -m "fix(watchlists): stabilize responsive layout requests"
```

### Task 3: Complete priority-lease, Article Focus, and rollback lifecycle

**Files:**
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`
- Modify: `Tests/Watchlists/test_watchlists_cold_open_layout.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:410-435, 1015-1030, 3054-3120, 3423-3515, 4400-4460, 4710-4755`

- [ ] **Step 1: Write failing intent-lifecycle tests**

Add mounted tests for these exact contracts:

1. At a responsive dead-band width, manually opening Inspector creates `ResponsivePriorityLease(Region.RIGHT_RAIL, read_mode=True)`, opens immediately, and a same-width passive event keeps it open.
2. Article Focus hides every side pane without changing `_responsive_region_layout` or the lease; exiting restores the responsive baseline. Resizing while focused updates the hidden responsive baseline without issuing visible workbench requests.
3. Switching to management parks a Read lease; switching back resumes it. A management resize must not clear the parked Read lease. A different manual open on Read replaces it; closing its target clears it.
4. The lease clears only when a passive resize in its originating mode reaches the hysteresis-stabilized all-preferred-fit result.
5. A failed real manual expansion restores the prior preferred layout, Article Focus state, and full lease. An explicit preference no-op returns no token and leaves `_manual_layout_rollback is None` rather than reusing `_current_layout_request_token`.
6. No responsive history or priority lease survives construction of a fresh screen in `test_watchlists_cold_open_layout.py`.
7. Extend `test_section_factory_failure_rolls_back_mode_and_can_retry` so a
   parked Read priority lease exists before the failing management swap and is
   unchanged after rollback and retry.

Extend `test_layout_intent_dataclasses_use_pascal_case_names` to include `ResponsivePriorityLease` and update direct `ManualLayoutRollback` fixtures to use `priority_lease_before`.

- [ ] **Step 2: Run the lifecycle tests and verify RED**

Run the new node IDs plus the existing failure/supersession cases:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  Tests/Watchlists/test_watchlists_cold_open_layout.py \
  Tests/Watchlists/test_watchlists_collections_screen.py::test_layout_intent_dataclasses_use_pascal_case_names \
  -k "priority_lease or article_focus or failed_manual_expansion or layout_acknowledgements or cold_open or section_factory_failure" \
  -q
```

Expected: new tests FAIL because the controller still stores only a target region, clears it outside its originating mode, and attaches rollback to a fallback token.

- [ ] **Step 3: Implement the lease and rollback contract**

Replace `ManualLayoutRollback.priority_before` with:

```python
priority_lease_before: ResponsivePriorityLease | None
```

Use a helper or one local expression to expose the active target only in the lease's originating mode:

```python
lease = self._responsive_priority_lease
priority_target = (
    lease.target if lease is not None and lease.read_mode == read_mode else None
)
```

During passive resize, clear the lease only when it is active in the current mode and the hysteretic unprioritized layout equals the preferred collapse set for mounted panes. Parked leases are left untouched.

In `_toggle_preferred_region`:

- capture the entire old lease;
- clear Article Focus before applying the manual action;
- create/replace the lease on an explicit open using the current mode;
- clear it only when the manual close targets that same active lease;
- call recomputation with `cause="explicit"`;
- create `ManualLayoutRollback` only when recomputation returns a real token;
- persist a genuinely changed preference even when no DOM request was needed.

Update `_next_layout_request_token` to re-key an existing in-flight rollback with its full lease snapshot. Update failure handling to restore the full lease before explicit recomputation. Do not modify the lease in failed section-swap fallback handling.

- [ ] **Step 4: Run lifecycle and regression tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Watchlists/test_watchlists_responsive_layout.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_cold_open_layout.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  -k "layout or resize or priority or article_focus or region or grip or cold_open or section_factory_failure" \
  -q
```

Expected: all selected changed-functionality tests PASS. Do not run the full repository suite.

- [ ] **Step 5: Commit the completed controller lifecycle**

```bash
git add Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py Tests/Watchlists/test_watchlists_cold_open_layout.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py
git commit -m "fix(watchlists): preserve responsive pane priority"
```

### Task 4: Focused verification, documentation, and task closeout

**Files:**
- Modify: `backlog/tasks/task-22211 - Watchlists-responsive-layout-needs-hysteresis-at-its-collapse-boundaries.md`
- Verify only: all Python and test files changed in Tasks 1-3

- [ ] **Step 1: Run the complete changed-functionality test set**

Run only the affected Watchlists layout files:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Watchlists/test_watchlists_responsive_layout.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_cold_open_layout.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  -k "layout or resize or priority or article_focus or region or grip or cold_open or section_factory_failure" \
  -q
```

Expected: all selected tests PASS with no errors or failures.

- [ ] **Step 2: Run static checks limited to modified code**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Watchlists_Modules/region_layout.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_responsive_layout.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_cold_open_layout.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Perform a bounded self-review**

Inspect `git diff origin/dev...HEAD` and confirm:

- manual `region_layout` persistence is unchanged by passive resize;
- every recompute call names its cause;
- only positive screen width seeds responsive history;
- Article Focus never replaces the responsive baseline;
- mode-local priority parks across sections and is cleared only in its origin mode;
- equality suppression occurs before token allocation and focus capture;
- rollback state exists only for a real request token;
- no Library import or shared split-pane abstraction was introduced.

- [ ] **Step 4: Close the Backlog task only after evidence is green**

The Backlog Implementation Plan and ADR assessment were recorded before Task 1 began.
Check all three acceptance criteria, add concise Implementation Notes naming the pure
resolver, invariant width authority, responsive baseline/lease, request suppression,
targeted tests, and modified files, then set status to Done through the Backlog CLI.

Do not add a lessons entry unless implementation reveals a new evidenced trap beyond the existing performance-review and testing lessons.

- [ ] **Step 5: Commit closeout metadata**

```bash
git add "backlog/tasks/task-22211 - Watchlists-responsive-layout-needs-hysteresis-at-its-collapse-boundaries.md"
git commit -m "docs(watchlists): close task 22211"
```

- [ ] **Step 6: Verify Backlog CLI resolution**

```bash
backlog task 22211 --plain
```

Expected: the printed file path is the TASK-22211 hysteresis task in this worktree;
status is Done; all three acceptance criteria are checked; and Implementation Plan,
ADR assessment, and Implementation Notes are present.
