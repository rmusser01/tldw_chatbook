# Library Media Return Settlement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Apply `superpowers:test-driven-development` to every behavior change and `superpowers:verification-before-completion` before each completion claim.

**Goal:** Make normal-Media returns deterministically restore the retained page, semantic row, final focus identity, and exact list scroll after a Trash round trip or authoritative same-scope recompose.

**Architecture:** Correct the Media presentation producer first so replacement stages use the adaptive Media class contract from their first frame. Add one Media-owned row-scroll geometry message derived from the owner's public Textual `Resize`, then let `LibraryScreen` settle an immutable, generation-fenced return request only when current-owner geometry proves the retained offset. Scroll is applied and verified before final focus; bounded failure is honest and non-recursive.

**Tech Stack:** Python 3.11+, Textual 8.x, immutable dataclasses, pytest/pytest-asyncio, Ruff, existing Library live harnesses.

**Spec:** `Docs/superpowers/specs/2026-08-30-library-media-return-settlement-design.md`

**ADR required:** yes

**ADR path:** `backlog/decisions/104-library-media-return-settlement-boundary.md`

**Reason:** The repair establishes a durable cross-module boundary between a Media widget's public geometry event and `LibraryScreen` navigation/focus authority, and amends the adaptive reader lifecycle.

## Global constraints

- `LibraryScreen` owns receipts, request identity, presentation epoch, authority fences, timeout, final-focus policy, and settlement outcomes. The Media scroll widget reports geometry only.
- Compose must project the correct Media stage classes before yielding the replacement shell. Post-mount reconciliation is equality-based and does not create work when unchanged.
- Textual `Resize` is the sole geometry-readiness producer after the application presentation gate. Sleeps, callback counts, recursive scheduling, geometry polling, and framework-private layout signals are prohibited.
- A presentation-changing projection records the current owner's latest geometry revision as an exclusive floor. A queued pre-change message cannot settle the new epoch.
- The content signature is applied scope plus ordered stable Media IDs. The logical layout signature excludes replacement widget identity, compose generation, presentation epoch, and transient geometry.
- An unchanged revision must restore the exact retained offset. A proven content/layout revision may clamp once and record a non-exact outcome. Timeout never proves shrink.
- Apply and verify scroll before final focus. Viewer/list returns finish on the semantic row; Trash round trips finish on the captured normal-Media control when it remains valid.
- After exact settlement, the retained receipt and existing two-second outer arm remain available for a later authoritative recompose in that window. User input, foreign focus, route change, or deadline disarms immediately.
- Restore retains the normal-Media page and marks it stale pending refresh; it never inserts or reranks a restored record.
- Do not add a dependency, schema change, shared cross-reader coordinator, Notes behavior change, or TCSS change without new evidence and an updated approved design.
- The existing uncommitted `Tests/UI/test_library_media_trash.py` repair and `Tests/Live/test_library_media_trash_paging_closeout.py` harness are Task 7 closeout work. Do not edit them in Tasks 1–3.
- Do not run repository-wide pytest without fresh user approval. Use only the focused and production-shaped gates below.

## File and ownership map

**Production owners**

- `tldw_chatbook/UI/Screens/library_screen.py` — Media presentation projection, presentation epoch, immutable request, authority fences, deadline, scroll/focus commit, and settlement outcomes.
- `tldw_chatbook/Widgets/Library/library_media_canvas.py` — Media-specific row-scroll owner and Resize-derived geometry message.

**Focused verification owners**

- Create `Tests/UI/test_library_media_return_settlement.py` — presentation, geometry protocol, authority, focus order, and bounded-failure tests isolated from Task 7's dirty files.
- Verify `Tests/UI/test_library_media_side_by_side.py` unchanged — production-shaped cross-reader exact-return consumer.
- Close out the existing `Tests/UI/test_library_media_trash.py` repair and existing `Tests/Live/test_library_media_trash_paging_closeout.py` harness only in Task 4.

## Approved-design coverage map

| Design obligation | Production owner | Primary proof |
| --- | --- | --- |
| Correct first-frame and same-size-recompose presentation | Task 1, `LibraryScreen` projection | First-frame and same-size-recompose focused tests |
| Public producer-owned geometry; no timing readiness | Task 2, `LibraryMediaRowScroll` | Resize payload/revision and no-premature-focus tests |
| Epoch floor rejects queued pre-change proof | Tasks 2–3, `LibraryScreen` | Exclusive-floor and queued-message inverse tests |
| Unchanged revision restores exact scroll before final focus | Tasks 2–3, settlement commit | `(0, 42)` consumer, focus-order, and five fresh runs |
| Viewer row policy and Trash captured-control policy | Task 3, final-focus resolver | Viewer/Trash/fallback focused tests |
| Content/layout revision and bounded failure stay honest | Task 3, authority/deadline paths | Revision, one-shot clamp, warning, and no-requeue tests |
| Existing paging/mutation contract and four sizes remain intact | Task 4 closeout | 108-test pure gate, cross-reader gate, four-size live harness |

---

### Task 1: Project the correct Media presentation before layout

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Create: `Tests/UI/test_library_media_return_settlement.py`

**Interfaces:**

Add one idempotent projection shared by compose and the Media adaptive-shell lifecycle path:

```python
def _project_library_media_stage_classes(self, shell_grid: Widget) -> bool:
    """Project effective Media stage classes; return whether they changed."""

def _reconcile_library_media_stage_presentation(self) -> bool:
    """Equality-reconcile the mounted Media stage; return whether it changed."""
```

- [ ] **Step 1: Write RED first-frame and same-size-recompose tests.**

In `Tests/UI/test_library_media_return_settlement.py`, mount the production `LibraryScreen` through the same lightweight app/host seam used by nearby Library tests. Assert both:

1. the first composed Media `#library-shell-grid` has `library-adaptive-compact` according to `_library_notes_compact` and does not have `library-notes-compact`; and
2. after forcing the wrong legacy class and performing a same-terminal-size authoritative recompose, the replacement Media stage is correct without waiting for a screen resize.

Keep the app context open inside each test; do not return mounted widgets from a helper after its `run_test()` context has exited.

- [ ] **Step 2: Run the new nodes and confirm the intended RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_return_settlement.py \
  -k 'first_frame or same_size_recompose'
```

Expected: assertions fail because the Media replacement stage can retain the Notes compact contract. No unrelated import or fixture failure is acceptable evidence.

- [ ] **Step 3: Implement the smallest shared projection.**

In `library_screen.py`:

1. remove `library-notes-compact` for the Media stage;
2. set `library-adaptive-compact` to the current effective compact projection represented by `_library_notes_compact`;
3. return whether either class actually changed;
4. call the projection directly in the Media compose branch before yielding the outer shell; and
5. call the mounted reconciliation from the existing Media adaptive-shell lifecycle handler before its current shell-width synchronization.

When mounted reconciliation changes classes, request one public layout refresh. Task 2
adds presentation-epoch advancement around this already-tested seam when the settlement
state exists. Do not reload data, write preferences, or recompose solely because
equality already holds.

- [ ] **Step 4: Verify Task 1.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_return_settlement.py \
  -k 'first_frame or same_size_recompose'

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_return_settlement.py

git diff --check
```

Expected: focused tests and Ruff pass; no Task 7 protected file changed during this task.

- [ ] **Step 5: Commit Task 1.**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_return_settlement.py
git commit -m "fix: project Media adaptive presentation"
```

---

### Task 2: Settle viewer returns from current-owner geometry

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_media_return_settlement.py`
- Verify unchanged: `Tests/UI/test_library_media_side_by_side.py`

**Interfaces:**

In `library_screen.py`, replace the existing viewer tuple and
`_LibraryMediaTrashReturn` with one frozen, transient receipt captured while the normal
Media list and its physical coordinate system still own the screen:

```python
_LibraryMediaFinalFocusPolicy: TypeAlias = Literal["row", "control"]


@dataclass(frozen=True)
class _LibraryMediaReturnReceipt:
    stable_id: str
    scroll_offset: tuple[int, int] | None
    content_signature: tuple[object, ...]
    layout_signature: tuple[object, ...]
    final_focus_policy: _LibraryMediaFinalFocusPolicy
    final_focus_identity: str | None
```

This is an in-memory normalization of the two existing return shapes, not a persisted
or cross-reader contract. Viewer capture uses row policy; Trash capture uses control
policy and its current normal-Media control ID.

Add exact signature producers in `LibraryScreen`:

```python
def _library_media_content_signature(self) -> tuple[object, ...]:
    """Return applied normal-Media scope plus ordered stable row IDs."""

def _library_media_layout_signature(self) -> tuple[object, ...]:
    """Return terminal allocation plus pure effective Media pane layout."""
```

The layout signature must not contain mounted object IDs, compose/lifecycle generation,
presentation epoch, or live `size`/`virtual_size`/`container_size` payloads.

In `library_media_canvas.py`, add immutable geometry data, an owner-identifying application message, and one Media-specific scroll owner:

```python
@dataclass(frozen=True)
class LibraryMediaRowGeometry:
    revision: int
    size: Size
    virtual_size: Size
    container_size: Size | None


class LibraryMediaRowGeometryChanged(Message):
    def __init__(
        self,
        owner: "LibraryMediaRowScroll",
        geometry: LibraryMediaRowGeometry,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.geometry = geometry


class LibraryMediaRowScroll(VerticalScroll):
    latest_geometry: LibraryMediaRowGeometry | None

    def on_resize(self, event: events.Resize) -> None:
        """Publish distinct, monotonically revised owner geometry after reflow."""
```

Import `events` and `Message` from Textual and `Size` from
`textual.geometry`; do not mirror Textual geometry in a second application type.

In `library_screen.py`, add one frozen request value rather than parallel mutable flags:

```python
@dataclass(frozen=True)
class _LibraryMediaReturnSettlement:
    request_id: int
    receipt: _LibraryMediaReturnReceipt
    final_focus_policy: _LibraryMediaFinalFocusPolicy
    final_focus_identity: str | None
    focus_intent_generation: int
    compose_generation: int
    media_lifecycle_generation: int
    presentation_epoch: int
    content_signature: tuple[object, ...]
    layout_signature: tuple[object, ...]
    route_identity: str
    media_view_identity: str
    shell_identity: int | None
    items_host_identity: int | None
    owner_identity: int | None
    exclusive_geometry_floor: int
```

Use the repository's actual route types when they are more specific than the
illustrative route strings above. The receipt's signatures remain the authority for
physical-coordinate validity; the request also captures current signatures so every
attempt can distinguish unchanged, proven revision, and stale authority.

- [ ] **Step 1: Write RED geometry and focus-order tests.**

Add tests proving:

- viewer and Trash capture normalize into frozen receipts with signatures taken before
  leaving the normal Media list;
- `LibraryMediaRowScroll` increments a revision and posts its concrete owner plus `size`, `virtual_size`, and `container_size` for distinct Resize geometry;
- a viewer/list return request does not publish final row focus before eligible geometry;
- geometry from the current owner at a revision above the exclusive floor applies `(0, 42)` exactly and only then focuses the semantic row; and
- duplicate delivery of the same revision is idempotent.

Retain the existing consumer assertion in `test_library_media_side_by_side.py`; do not weaken `(0, 42)` to the observed transient clamp.

- [ ] **Step 2: Confirm RED, including the established consumer failure.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_return_settlement.py \
  -k 'geometry or viewer_return or duplicate_revision'

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_side_by_side.py \
  -k 'media_recompose_restores_exact_row_and_scroll_after_other_reader_round_trip'
```

Expected: the new protocol tests fail because it does not exist, and the established consumer can reproduce `(0, 33) != (0, 42)` in a fresh process.

- [ ] **Step 3: Implement the Media geometry producer.**

Subclass only the Media row `VerticalScroll`. On public `events.Resize`, compare all three geometry fields with the last payload. For a distinct payload, increment an owner-local integer revision, retain the frozen value, and post `LibraryMediaRowGeometryChanged(self, geometry)`. Handle that message in `LibraryScreen` with `@on(LibraryMediaRowGeometryChanged)` and stop it after resolving the current owner. The widget must not inspect route, page, receipt, focus, or screen epoch and must not schedule follow-up callbacks.

- [ ] **Step 4: Implement exact viewer settlement in `LibraryScreen`.**

1. Normalize viewer and Trash captures into `_LibraryMediaReturnReceipt` while the normal list is current; preserve the same semantic row, offset, and captured control behavior.
2. Give `LibraryScreen` a monotonic presentation epoch and request ID.
3. When presentation changes, advance the epoch and record the current owner's latest revision as an exclusive floor.
4. Arm an immutable row-policy request from the retained viewer receipt instead of immediately declaring focus/scroll success. Keep the existing Trash final-focus continuation until Task 3 replaces it, so Task 2 does not silently change that policy before its RED coverage exists.
5. After recompose, rebuild the current request from the still-authoritative retained receipt with fresh compose/lifecycle/owner identities.
6. On the owner geometry message, reject unless request object identity, all captured generations, route/sub-view, Items-open state, current ancestry, receipt, applied content signature, logical layout signature, owner identity, live payload equality, and `revision > exclusive_geometry_floor` all hold.
7. If geometry was already published before the request armed, evaluate `owner.latest_geometry` immediately only when its revision remains above the current epoch floor and every authority fence holds.
8. Resolve the semantic row and prove the exact retained offset is within the current horizontal/vertical maxima.
9. In one synchronous commit, set the unanimated integer scroll offset, verify equality, apply programmatic row focus with `scroll_visible=False` when supported, then verify focus did not change the offset.
10. Record `exact-settled` for that request/revision. Consume only the ephemeral request; retain the receipt and existing outer focus arm for any later authoritative recompose inside its window.

Do not use `call_later()`, `call_after_refresh()`, sleeps, polling, or a fixed event count as readiness proof.

- [ ] **Step 5: Prove fresh-process determinism.**

Run the exact consumer five times with isolated pytest temp roots:

```bash
for run in 1 2 3 4 5; do
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/UI/test_library_media_side_by_side.py \
    -k 'media_recompose_restores_exact_row_and_scroll_after_other_reader_round_trip' \
    --basetemp="/tmp/task18918-media-return-${run}"
done
```

Expected: five independent passes, each preserving `(0, 42)`.

- [ ] **Step 6: Verify Task 2 and commit.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_return_settlement.py \
  Tests/UI/test_library_media_side_by_side.py \
  -k 'media and (presentation or geometry or return or recompose)'

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_return_settlement.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py

git diff --check
git add tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_return_settlement.py
git commit -m "fix: settle Media returns on owner geometry"
```

---

### Task 3: Preserve Trash focus policy and fail stale work closed

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_media_return_settlement.py`
- Verify unchanged: `Tests/UI/test_library_media_side_by_side.py`

**Required outcomes:**

- `exact-settled` — exact scroll and requested final focus committed.
- `exact-scroll-focus-fallback` — exact scroll committed, captured control unavailable under a proven layout, documented fallback focused.
- `clamped-after-revision` — content/layout signature authoritatively changed, so one explicit clamp and final-focus policy committed.
- `clamped-after-settlement-failure` — unchanged request reached its deadline with all non-geometry fences valid, then one honest geometry-based clamp/focus fallback committed.
- `layout-settlement-failed` — unchanged request reached its deadline but no truthful commit was possible.

Represent these as one transient, request-ID-keyed last outcome for assertions and
metadata-only diagnostics, for example:

```python
_LibraryMediaSettlementOutcome: TypeAlias = Literal[
    "exact-settled",
    "exact-scroll-focus-fallback",
    "clamped-after-revision",
    "clamped-after-settlement-failure",
    "layout-settlement-failed",
]

_library_media_last_settlement_outcome: (
    tuple[int, _LibraryMediaSettlementOutcome, int | None] | None
)
```

The tuple is `(request_id, outcome, owner_geometry_revision)`. It is not persisted,
shown as user content, or used as readiness authority.

- [ ] **Step 1: Write the stale-authority and Trash-policy RED matrix.**

Add tests for:

1. a presentation revision at or below the exclusive floor cannot settle, including a custom message queued before the class change;
2. a Trash Back return applies exact list scroll before focusing the captured normal-Media control and retains the selected row;
3. an unavailable captured control falls back to the semantic row or existing safe Media-list target and records `exact-scroll-focus-fallback`;
4. a proven content-signature or logical-layout-signature change clamps once and records `clamped-after-revision`;
5. an unchanged request with no eligible geometry reaches one deadline fallback/failure with no re-enqueued attempt;
6. duplicate message, old owner, detached owner, replaced shell/Items host, stale request ID, stale compose/lifecycle/focus generation, foreign route, Trash still active, Items collapsed, user input, foreign focus, and unmount cannot settle; and
7. a later authoritative recompose inside the two-second outer arm may create a new request after a prior exact settlement, while user takeover cancels that ability.

Use direct event/message delivery where possible. The tests must assert state/outcome and absence of queued follow-up behavior, not sleep for layout.

- [ ] **Step 2: Confirm the new matrix is RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_return_settlement.py \
  -k 'exclusive_floor or trash_focus or fallback or revision or deadline or stale or takeover or unmount'
```

Expected: each new behavior fails for the missing policy/authority path rather than fixture setup.

- [ ] **Step 3: Complete the final-focus policy.**

Arm normal viewer/list returns with `final_focus_policy="row"`. Arm Trash Back with `final_focus_policy="control"` and the captured normal-Media control identity. Remove any independent scheduled `_restore_library_media_trash_return_focus` path; the settlement commit becomes the single ordering owner.

For a current exact request, resolve both the semantic row and requested final target before mutation, apply and verify exact scroll, then set and verify final focus under the programmatic-focus guard. If the captured control is invalid only because a proven responsive layout removed it, use the documented fallback without claiming full exact settlement.

- [ ] **Step 4: Implement revision and deadline outcomes.**

For an authoritative content/layout signature change, use current-owner geometry to clamp once, apply the request's final-focus policy, and record `clamped-after-revision`. Do not classify widget replacement or compose/presentation generation alone as a logical revision.

Bind the existing two-second deadline to the ABA-safe request ID. At expiry:

- if any non-geometry fence fails, clear silently;
- otherwise attempt at most one current-geometry clamped scroll plus final-focus fallback;
- record `clamped-after-settlement-failure` only if that commit succeeds, else `layout-settlement-failed`;
- emit one metadata-only warning for the current request; and
- clear the ephemeral request, retained arm, and timer without re-enqueueing work.

Route change, user input/foreign focus, another Back request, or unmount synchronously invalidates the obsolete request.

- [ ] **Step 5: Run focused and cross-reader verification.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_return_settlement.py \
  Tests/UI/test_library_media_side_by_side.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_return_settlement.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile \
  tldw_chatbook/UI/Screens/library_screen.py

git diff --check
```

Expected: all focused settlement and existing cross-reader nodes pass; the protected Task 7 files remain untouched by this task.

- [ ] **Step 6: Commit Task 3.**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_return_settlement.py
git commit -m "fix: preserve Media return authority"
```

---

### Task 4: Run production-shaped closeout and finish TASK-18918

**Files:**

- Include existing modification: `Tests/UI/test_library_media_trash.py`
- Include existing file: `Tests/Live/test_library_media_trash_paging_closeout.py`
- Modify: `Docs/User_Guide/library.md`
- Modify: `backlog/tasks/task-18918 - Add-paged-recovery-viewing-to-Library-Media-Trash.md`
- Modify if incident-backed: `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Review and preserve the existing Task 7 closeout changes.**

Confirm `Tests/UI/test_library_media_trash.py` contains only the one-line test-fake compatibility repair already made, and review the existing live harness rather than recreating it. Check that the live harness exercises more than 40 records and the exact 160×50, 120×35, 100×30, and 80×24 sizes.

- [ ] **Step 2: Run the pure TASK-18918 gate.**

Run the established pure gate unchanged and expect 108 passing tests:

```bash
/usr/bin/env \
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-18918-media-trash-paging \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_client_media_trash_pagination.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  Tests/Library/test_library_media_trash_state.py \
  Tests/UI/test_library_media_trash_browse_controller.py \
  Tests/UI/test_library_media_browse_controller.py \
  -k 'media_trash or library_media_trash or permanently_delete_media_item or mark_stale_after_trash_restore'
```

Expected: `108 passed, 191 deselected`. Record the actual warning count and duration.
If the count changes, investigate and record the reason in Implementation Notes rather
than silently replacing the established evidence.

- [ ] **Step 3: Run the mounted production-shaped cross-reader gate.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_return_settlement.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_adaptive_reader_shell.py \
  Tests/UI/test_library_adaptive_reader_closeout.py
```

Expected: all pass, including exact Media return scroll, Trash focus policy, and unchanged Notes/other-reader behavior.

- [ ] **Step 4: Run the live four-size walkthrough.**

```bash
/usr/bin/env \
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-18918-media-trash-paging \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Live/test_library_media_trash_paging_closeout.py -s
```

At 160×50, 120×35, 100×30, and 80×24 verify: all Trash pages are reachable; filters/pager/Retry and both confirmations remain operable; Library and Items collapse independently; collapsing Library expands Items so full titles/details gain width; Restore retains normal Media's page but marks it stale; Back restores selected row, exact list scroll, and captured normal-Media control; viewer returns finish on the row.

- [ ] **Step 5: Run inverse-matrix and static checks.**

Exercise the existing mutation/request-generation inverse matrix plus the new queued-pre-epoch geometry, old-owner, detached-owner, focus-before-scroll, duplicate-revision, user-takeover, route-change, and deadline inverses.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_return_settlement.py \
  Tests/UI/test_library_media_trash.py \
  Tests/Live/test_library_media_trash_paging_closeout.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Live/test_library_media_trash_paging_closeout.py

git diff --check
```

- [ ] **Step 6: Update documentation and task evidence.**

Update the Library guide with the shipped Trash paging/recovery behavior and deterministic return semantics. If the failed callback-turn approaches produced a reusable lesson not already captured, add the concrete incident to `backlog/docs/lessons-testing-evidence.md`: same-size Media recompose kept the wrong presentation and four scheduler-based remedies remained nondeterministic; only producer-owned Resize evidence closed the gate.

In TASK-18918:

- check every acceptance criterion;
- record the original paging work plus the ADR-104 settlement amendment;
- list targeted test commands and exact results, five fresh exact-return passes, and all four live sizes;
- document the Task 7 baseline test-fake repair;
- link the spec, both plans, ADR-067, and ADR-104; and
- set status to Done only after all evidence is green.

- [ ] **Step 7: Commit closeout.**

```bash
git add Tests/UI/test_library_media_trash.py \
  Tests/Live/test_library_media_trash_paging_closeout.py \
  Docs/User_Guide/library.md \
  backlog/docs/lessons-testing-evidence.md \
  "backlog/tasks/task-18918 - Add-paged-recovery-viewing-to-Library-Media-Trash.md"
git commit -m "docs: close Media Trash paging task"
```

If the lessons file did not require an incident-backed change, omit it from `git add`.

## Completion criteria

- The first Media frame and same-size recomposes use the adaptive Media presentation contract.
- Exact unchanged-revision returns settle only from current-owner Resize-derived geometry above the presentation epoch's exclusive floor.
- Scroll is exact and observable before final focus; viewer returns focus the row and Trash returns focus the captured valid control.
- Stale authority cannot settle, proven revisions and deadline failures are labelled honestly, and no readiness polling or callback-count contract exists.
- The established exact-return consumer passes five fresh isolated runs.
- Focused pure/mounted gates and live 160×50, 120×35, 100×30, and 80×24 walkthroughs pass.
- TASK-18918, ADR-104, user documentation, and any incident-backed lesson are complete.
