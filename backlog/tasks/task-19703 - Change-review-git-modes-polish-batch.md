---
id: TASK-19703
title: 'Change review git modes: polish batch from whole-branch review'
status: Done
assignee: []
created_date: '2026-08-21'
labels:
  - console
  - change-review
dependencies:
  - TASK-16801
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three non-blocking items the TASK-16801 whole-branch review raised, batched so
they do not each cost a separate branch. None changes behaviour a user relies
on today; all three are places where a failure would be reported less honestly
than the rest of the feature manages.

1. The detection worker's final `call_from_thread` is unguarded, while every
   other landing in the screen routes through the shared helper that catches
   only Textual's teardown `RuntimeError`. During app teardown this one can
   raise a logged worker failure instead of exiting quietly.
2. Both git modals' `_submit` handlers wrap their body in a bare `except`, so a
   genuine bug inside submit makes the confirm button silently do nothing —
   the user presses it and no feedback appears at all.
3. The merge/rebase/cherry-pick in-progress refusal is enforced by the engine's
   guard step, so the user only learns about it after filling in and submitting
   the commit modal. The design spec places that check before the modal opens,
   and the User Guide's wording implies the earlier placement.

Item 3 was a deliberate implementation choice (it avoids a private import and a
time-of-check/time-of-use window), so resolving it may mean amending the spec
and the Guide rather than moving the check — either is acceptable, but the
three artefacts should agree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The detection worker's landing uses the same teardown-only guard as every other landing, so a real bug there surfaces as a traceback rather than a debug line
- [x] #2 A bug raised inside either git modal's submit reports itself to the user instead of leaving the confirm button silently inert; a test proves it
- [x] #3 The in-progress refusal's placement is consistent across the code, the design spec and the User Guide — whichever placement is chosen
<!-- AC:END -->

## Implementation Notes

Three items, all from TASK-16801's whole-branch review; each is a place a
failure was reported less honestly than the rest of the feature manages.

**AC #1 — the detection landing.** `_dispatch_git_detection`'s
`call_from_thread` was the one landing not routed through `_land_on_ui`,
the shared helper that tolerates ONLY Textual's teardown `RuntimeError`.
Now routed, so a genuine bug in `_land_git_detection` surfaces as a
`WorkerFailed` traceback instead of dying quietly on the worker thread.

**AC #2 — both modals' submit.** Each `_submit` wraps its body in a broad
`except` so it can never raise into a Textual handler — correct — but then
returned SILENTLY, so pressing Commit (or Push) did nothing at all: no
action, no error, no dismissal, indistinguishable from a dead button. The
broad catch stays; the failure is now visible. The commit modal reuses its
inline `#change-git-commit-error` Static; the push modal has no such widget
and reports through `notify` rather than inventing one (plus CSS) for a
path only a bug reaches. Both proven red first and mutation-checked:
removing the two reports fails both new tests.

**AC #3 — the in-progress refusal's placement.** Code, spec and Guide
disagreed. The spec listed the merge/rebase/cherry-pick check as a
modal-open gate (step 3) and the User Guide told users it fired "before the
dialog even opens"; it actually ships as the engine's `in-progress-check`
step, i.e. at confirm time. **Resolved by amending the documents, not the
code.** The engine is the right home: the repository can enter a merge
between modal-open and submit, so a pre-modal check could only ever be
advisory — and an advisory check that passes and is then refused at submit
is more confusing than one honest refusal, while duplicating it invites the
two copies to drift. The Guide sentence bundled this together with the
active-run refusal, which genuinely IS pre-modal, so the two are now stated
separately with the reason for the difference. Guide stamped honestly: not
driven live, corrected by reading the shipped code against the page's own
claim.

**Qodo round (PR #1958):** one substantive finding, and it was right in a
way that improved the fix. `_land_on_ui` tolerated `RuntimeError` because
that is Textual's teardown signal — but `call_from_thread` ALSO re-raises
whatever the callback raised, so a genuine bug that happens to be a
`RuntimeError` was read as shutdown and vanished. Routing detection through
that helper (AC #1) therefore narrowed visibility for that one exception
type. `App.is_running` is the real discriminator — it is exactly what
`call_from_thread` consults before raising "App is not running" — so the
guard now re-raises when the app is still alive.

Two consequences worth recording. First, the current-mode worker carried a
SECOND copy of the same teardown rule in a local `_land` helper, which the
sharpening would have left behind — so a landing bug still vanished on the
very path the fix targeted. It now delegates to the one implementation.
Second, the existing teardown test simulated shutdown by raising
`RuntimeError("App is not running")` while the app was still running; that
proxy is exactly what the new discriminator (correctly) treats as a bug, so
it was rewritten to drive `_land_on_ui` directly against a stub app that is
genuinely not running.

Declined, with reasoning on the PR: the `Args:`-section docstring request
(zero tests in these files document pytest fixtures that way) and the
raw-exception-in-UI concern (this is a single-user local TUI whose banner
already prints absolute paths and git stderr by design; a generic message
with detail hidden in a log the user will not find is less honest, not
more).

**Files:** `tldw_chatbook/UI/Screens/change_review_screen.py`,
`Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md`,
`Docs/User_Guide/console/agent-runs-and-tools.md`,
`Tests/UI/test_change_review_commit_ui.py`,
`Tests/UI/test_change_review_push_ui.py`.
