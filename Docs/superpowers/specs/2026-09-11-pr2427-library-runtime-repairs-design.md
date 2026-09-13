# PR 2427: remaining Library runtime repairs

Status: design direction and written spec approved by the user on 2026-09-11.
Task: TASK-31932, steps166–168. Base: `81dadcc80bc92988f9def977992708a96c36b6a5`.

## Goal and limits

Repair the four remaining failing cases from the original32, including both
deterministically reproduced lifecycle races. The28 test-contract corrections
remain intact. This is one bounded Library regression-repair batch, not a new
layout, ownership model, or navigation policy.

ADR required: no.
ADR path: existing `backlog/decisions/086-library-adaptive-reader-shell.md` and
`backlog/decisions/141-library-work-retirement-and-in-place-sync-followups.md`.
Reason: restore existing mounted-state, retirement, footer and focus contracts
without changing a service boundary, persistence, dependencies or cleanup authority.
TASK-31223, TASK-17653 and TASK-32228 govern the footer and focus details.

## Approved approach

Use the existing projection and focus owners. Rejected alternatives are catching
missing-widget errors broadly, extending arbitrary delays, and introducing a
second lifecycle/focus coordinator. Those would hide ordering failures or add
ownership without addressing the observed causes.

### 1. Notes projection before child mount

`LibraryNotesController._load_library_note_backlinks` can finish after the work
pane adopts editor mode but before its editor children mount. Parent mounted
state therefore does not establish that `apply_session_state` can query them.

In `Widgets/Library/library_notes_canvas.py`, retain the newest immutable
presentation state and compact flag at the existing `apply_session_state`
boundary, but defer child-widget writes until the editor subtree is ready.
Existing `on_mount`/post-recompose wiring applies the retained state; no new
timer, worker, callback queue or whole-screen fallback is added. Preserve the
controller's synchronization guard, generation fencing, dirty tracking, focused
field authority and editor-ready signaling. Superseded state must not overwrite
a later snapshot when mounting completes.

### 2. Files projection during teardown

`LibraryFileNotesWorkspace._initialize` can resume after descendants are removed
but before parent Unmount clears `_active`. The reproduced state is mounted and
attached, yet `is_running=False`, with no path-label widget remaining.

In `Widgets/Library/library_file_notes_workspace.py`, extend the existing
control-projection lifecycle guard to reject that public stopped state. Keep
runtime acquisition, service/replica adoption and existing shutdown ownership
unchanged: an early return must not abandon an acquired resource. Review sibling
projection entry points reached by the controlled initialization path; extend
the same existing guards only where the reproduction demonstrates a need.
No broad exception catch, global cleanup or new disposal owner is introduced.

### 3. Notes footer registration and retired indicator

`LibraryScreen._register_footer_shortcuts` currently publishes generic help after
the Notes helper has published the correct region-specific help. Pane visibility
refresh is the confirmed last writer. Select the existing Notes-aware helper at
this shared registration boundary; it already falls back to generic Library
help for other routes. Preserve persistence across footer replacement, emergency
return updates, width tiers and typing-context filtering.

In `UI/Library_Modules/library_notes_controller.py`, the ancillary visibility
publisher must not reveal the retired token indicator. Keep it hidden and empty
under TASK-17653 while preserving word-count/DB contents and their existing
responsive policy. Do not restore retired token data or redesign footer chrome.

### 4. Empty Conversations focus landing

The corrected test starts from an attached current rail row. Even then, the
registered entry-focus channel leaves focus None after background recompose:
it searches for a filter absent from a genuinely empty page. That page already
offers the enabled `#library-conversations-empty-console` recovery action.

Extend the existing empty-list fallback in `UI/Screens/library_screen.py` to
recognize that recovery action. Retain filter priority for filtered misses,
existing nonempty row selection, attached/focusable readiness, bounded retry,
channel expiry, generation guards and newer-user-focus vetoes. Do not change the
generic stand-down policy or create another focus channel. Focusing the action
must not activate it or navigate to Console.

## Verification contract

- Add deterministic mounted regressions at the real service/mount/teardown
  boundaries, adapting the existing diagnostic gates into owned tests. Verify
  the intended ordering is actually reached; release every gate in finalizers.
- Notes: hold child mounting while real backlinks finish, require no worker
  failure, latest snapshot visible after mount, editor-ready behavior and working
  subsequent edits. Include a superseding-state control.
- Files: release initialization during descendant teardown, require no missing
  widget failure, exact existing resource shutdown and successful retained-owner
  remount where supported. Do not substitute attachment alone for running state.
- Footer: retain exact navigator/select/create/sync/editor/typing hints through
  pane visibility and recompose; require retired token hidden/empty and preserve
  unrelated route hints, word/DB contents and emergency-return behavior.
- Conversations: require an attached recovery target after empty-page recompose,
  filter focus for a filtered miss, existing nonempty focus behavior, and no
  stealing of a newer explicit focus intent. Initial test focus must be attached.
- Observe RED before runtime edits, then GREEN. Rerun the original32 with renamed
  node IDs, followed by complete affected files and scoped static/derived checks.
  Observe terminal native descriptors using the established observation-only
  runner and fresh per-user temporary roots. Do not force GC or clear pytest's
  final traceback to manufacture cleanup evidence.

## Evidence and handoff

The original B run finished986 passed/32 failed, with zero final SQLite/locks.
Combined targeted evidence verifies28 contract corrections; complete affected
files are not yet requalified. Controlled failures are recorded in
`/private/tmp/pr2427-lifecycle-proof.log` and
`/private/tmp/pr2427-focus-current.log`; footer diagnostics and the full evidence
ledger are linked from `backlog/docs/pr-2427-rebase-reconciliation.md`.

After written-spec approval, record the implementation plan in this existing
worktree, obtain its independent review, implement test-first, and review the
result. No further scope/design approval is needed for routine edits within this
spec. Stop only for a materially different remedy or missing authority.

Do not rebase during qualification or claim PR readiness from these targeted
repairs. Separate size/preload/CSS and resource-owner findings, newest-dev
integration, final PR comments/checks and normal protected merge remain open.
Do not raise caps, weaken assertions, run a repository-wide sweep, mutate the
outer checkout, or touch the unrelated reader-paydown plan.
