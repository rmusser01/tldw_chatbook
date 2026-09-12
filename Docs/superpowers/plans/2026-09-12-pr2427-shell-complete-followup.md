# PR 2427 complete Shell follow-up implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development, followed
> by independent spec and quality reviews. Root alone writes Git.

**Goal:** Reconcile five confirmed stale Shell assertions without weakening
empty recovery, responsive focus, exact scrolling or fixed-height contracts.

**Architecture:** Test-only updates follow accepted TASK-32350 scope summaries
and TASK-32360 toolbar rebuilding. Keep existing runtime owners and all caps.
The sixth finding (Media Retry focus) remains a separate diagnosis, not an
assumed test correction.

**Tech Stack:** Python, pytest, real Textual Pilot, macOS native FD observation.

ADR required: no.
ADR path: N/A; existing TASK-32350, TASK-32360 and TASK-32175 contracts.
Reason: test-only reconciliation with accepted behavior, no new architecture.

## Baseline and boundaries

- Worktree: `.worktrees/pr2427-review-recovery`, published HEAD `440b2be6b2`.
- Corresponding task: TASK-31932, AC3 remains open.
- Preserve unrelated untracked reader-paydown plan without reading it.
- Complete Shell: `/private/tmp/pr2427-resize-shell-complete.CAd9yu`,
  862 passed / 6 failed, zero final SQLite and instance-lock handles.
- Fresh targeted RED: per-user temporary directory
  `pr2427-shell-six-red.XNCEuZwa9l`, 7 passed / same 6 failed in 71.91s.
  Its final inventory differs from the complete-file result: 20 descriptors,
  including 13 SQLite and one instance lock attributed to the final failing
  semantic-role case's app factory. Keep this separate lifetime finding open
  until exact-owner cleanup is independently verified; passing full-file
  teardown cannot establish isolated failure-unwind cleanup.
- Observation-only Notes trace: per-user temporary directory
  `pr2427-notes-shape-trace.hswYsWxrJ4/trace.jsonl`. The old purpose and note
  row are detached, although Textual retains `is_mounted=True`; current same-ID
  widgets differ. At 100x30 the navigator has measured width42, no overflow row,
  and height16. At width80 its initial height is9, not10. Changing both width
  and height crosses TASK-32360's shape boundary, so it cannot isolate surplus.

## Task 1: Preserve the tested contracts across accepted presentation changes

Only implementation file: `Tests/UI/test_library_shell.py`.

- [x] Media empty recovery: retain exact absent pager/detail selectors and all
  real painted recovery geometry, Tab/Enter activation, filtered reset and
  destination assertions. For Media only, replace the whole-frame `0 of 0`
  prohibition with the exact scope Static content
  `Media · 0 of 0 · all types · sort: Newest` and an explicit absence of
  `Item 0-0 of 0` pager copy. Keep the original `0 of 0` prohibition for
  Conversations and Prompts, plus the shared Page1of1/selection-copy exclusions.
- [x] Purpose: parameterize the existing breakpoint test over normal and
  selection navigator state. Normal follows the shape-changing rebuild and
  reacquires the live purpose Static at each breakpoint; retain original
  canvas identity, hidden-zero-height compact copy and visible-positive-height
  wide copy assertions. Selection must retain the same purpose object across
  the round trip (the neighboring selection-label control already pins the
  unchanged shape). Explicitly assert replacement/detachment in the normal
  case and same-object identity in selection. If measurement contradicts that
  no-shape-change premise, stop and report before changing expectations.
- [x] Surplus allocation: parameterize terminal width80 and100 separately;
  each starts at height24 and grows to30 at the SAME width, isolating height
  from a toolbar-shape transition. Retain navigator/editor/context cases,
  exact owner identity, every fixed-row height, and the exact six-row growth
  in only the named scroll owner. For navigator width80 include the accepted
  one-row browse overflow in the fixed map and expect owner9→15; width100
  requires no overflow and owner10→16. Editor remains10→16 and context12→18
  at both widths. Confirm these values on the real tree; do not use tolerances
  or derive expected sizes from the measured values. No shared-helper changes.
- [x] Full semantic-role journey: after the note-placement round trip,
  reacquire the row by its exact original ID, assert the same placement ID,
  live attachment and identity with the actual focus target, then press that
  current row. Preserve the original per-role focus/scroll comparisons and
  every later editor/title/preview/context/create-template transition.
- [x] Run all parameters of the four changed test functions and neighboring
  compact-label, breakpoint-editor and resize adverse controls. Use fresh
  report directories; never alter production code or the native observer.
  If later assertions reveal another issue, diagnose it before proposing a fix.
- [x] Self-review exact diff, run scoped Ruff and whitespace checks. Obtain
  independent spec review then independent quality review, addressing findings.

## Task 2: Qualification and progress preservation

- [ ] Freeze all changes after the separate Retry diagnosis is resolved or
  explicitly left open. Run complete affected Shell file (targeted file, not
  a full repository sweep) with the unchanged native observer and fresh root.
- [ ] Record exact terminal counts, warnings and native resource inventory
  separately. Never call the file qualified while a test remains failing.
- [ ] Update this plan and TASK-31932, then commit/publish only reviewed changes.
  Keep latest-dev integration, separate Console fixture resources, unchanged
  size/preload/CSS limits and final-head PR review/checks open.

## Separate Retry diagnosis (runtime repair not yet approved)

Observation-only trace `/private/tmp/pr2427-retry-trace-native.apr8fF/trace.jsonl`
disproves a canvas-readiness-only explanation. At monotonic463517.532482 the
canvas has no pending recompose callback, the current Retry is attached with
positive geometry, and readiness passes. At463517.535491 Retry receives focus;
at463517.542918 the real entry retry timer calls
`_focus_library_list_entry_if_current` and takes focus to the type filter.
Retry is still the same live widget. Its queued DescendantFocus has not yet
disarmed the request, and correctly becomes stale after the timer steals focus.

The guarded continuation checks only pending state and generation, while
TASK-2856 promises not to override the user's focus takeover. Waiting longer
or holding the timer only in this test would leave the runtime race untested.
The proposed bounded repair checks live focus ownership at the existing
ordinary-entry continuation, preserving the original anchor, legitimate
programmatic/row entry and separate semantic Media-return authority. A
deterministic regression must deliver the captured real timer callback after
Retry receives focus but before its focus event is consumed. User design
approval was requested; no runtime change has been implemented for this issue.

## Task 1 implementation evidence

Only the four specified Shell test functions/parameter declarations change.
Media retains pager absence and real recovery activation while checking the
accepted exact scope summary. Normal Notes purpose checks replacement and
attachment, selection checks retained identity, and both retain exact copy and
visibility geometry. Six fixed-width surplus cases retain exact owner identity,
fixed row sizes and six-row growth. The full semantic-role journey now activates
the live same-placement row, preserving every later focus and scroll assertion.

`/private/tmp/pr2427-shell-followup-four.1iyedj` passes all **15 cases in42.02s**,
exit zero, with three installed-dependency warnings. Final native inventory:
six descriptors, zero SQLite and zero instance locks. This is success-path
qualification, not proof that the prior isolated assertion-failure lifetime is
closed.

The expanded frozen-source run at
`/private/tmp/pr2427-shell-followup-expanded.0vxG57` passes **25/25 in166.78s**,
with three dependency warnings, exit zero, six final descriptors and zero
SQLite/instance locks. It includes all15 cases, the compact-label and editor
round-trip neighbors, exact breakpoint/same-side guards, 50-cycle presentation,
both 50-sequence same-side controls and high-frequency resize. Source hash:
`6cc1bb7ca31d7581917f60bb2daaf480d90c5f7c6973b9fee5257a06d87d4022`.

Independent spec and quality reviews approve the actual diff without findings.
Root confirms fatal Ruff (`E9,F63,F7,F82`) and whitespace checks pass. This
reviewed targeted slice may be checkpointed while the separate runtime design
approval is pending; complete Shell qualification and TASK-31932 AC3 remain
open. No runtime, CSS, shared fixture/helper or size budget changed in this slice.
