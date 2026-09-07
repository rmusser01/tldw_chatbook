---
id: TASK-2703
title: 'Console Edit Message modal: action buttons invisible in real terminals'
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31'
updated_date: '2026-08-14 02:03'
labels:
  - console
  - bug
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In a real terminal, the Console "Edit Message" modal renders its header,
explanation, and editor — but the **Cancel / Save / Edit & resend** buttons
never paint. The `#console-edit-message-actions` row's space is reserved
(blank rows between the editor and the modal's bottom border) and the
buttons ARE focusable — Tab/Tab/Enter activates one and the modal closes —
so the feature works blind, but a mouse user has no visible way to Save or
Edit & resend, and a keyboard user gets no focus feedback.

Reproduced on dev @ ff435772c (G1 user-guide verification session,
2026-07-31): tmux, both 235×52 and 200×50, two separate app instances,
opened via transcript selection → `e` on a USER message. Crucially, the
same flow **headless under `app.run_test(size=(200, 50))` appeared healthy
under geometry-only inspection** (`display=True`, non-zero regions, and
on-screen coordinates). A later real-bundle compositor probe showed why that
was a false positive: the fixed editor height pushes the USER action row
outside the opaque modal content region, so its button cells never paint and
center hit-tests resolve to the modal. The shorter non-USER shape still paints
its actions, although its full action region may overhang by one row. The
regression therefore needs compositor-cell, containment, and hit-test evidence,
not mounted/display geometry alone. Other Console modals remain unaffected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancel / Save / Edit & resend are visible in a real terminal (tmux and a normal TTY) for both USER and non-USER targets of the modal.
- [x] #2 Focus is visibly indicated when Tab reaches each button.
- [x] #3 A regression check exists that would catch a live-terminal-only disappearance (at minimum: a note in the test explaining why the headless assertion is insufficient, plus a geometry assertion that holds under the real stylesheet).
- [x] #4 The User Guide quirk note in `Docs/User_Guide/console/branching-and-rewind.md` is updated/removed to match the fixed behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: this is a localized Textual layout/rendering correction that changes no
storage, ownership, interface, security, dependency, or long-lived UX boundary.

1. Add independent real-bundle compositor paint, containment, and hit-test RED
   coverage for USER and non-USER modal shapes at the reported terminal sizes;
   evaluate ordinary/focused contrast after containment is repaired.
2. Replace only the editor's fixed height with remaining-space sizing and prove
   the fixed-height regression by mutation.
3. Add modal-scoped paint/focus CSS only if a separate post-containment RED
   proves it necessary; rebuild the generated bundle through the existing tool.
4. Verify both shapes and every focus step through tmux and a separate PTY with
   scratch state and before/after isolation fingerprints.
5. Remove the obsolete guide workaround, run bounded behavior/static/UI review
   plus the user-approved scoped verification, record evidence, and close the
   task only after final candidate verification.

Detailed execution plan:
`Docs/superpowers/plans/2026-08-13-task-2703-console-edit-modal-paint.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the modal-scoped layout correction and removed the obsolete guide workaround.

- RED/GREEN and mutation: real-bundle compositor tests reproduced missing USER paint, center hits, and containment. Flexible editor sizing passed all 61 owned modal tests. Restoring the fixed height failed the exact 22 layout/paint oracles; neutralizing the focus cue failed all 10 focus cases, and broad sibling focus styling failed 6 isolation cases. No CSS source or generated-bundle change was required.
- Interaction: every action label is painted and owns its mouse hit target; incumbent mouse activation tests remain green. An explicit mounted keyboard regression proves Tab reaches Save and Enter returns the expected edit result.
- Live terminal evidence: private tmux and a separate direct Expect PTY covered 200x50 and 235x52, USER and non-USER shapes, with 28/28 state records and 28/28 raw ANSI/SGR captures. All labels were contained, hit-testable, and painted; 20/20 focused actions had a non-color cue, siblings stayed ordinary, and minimum measured contrast was 4.578406:1.
- Isolation: live drivers used scratch HOME, XDG roots, config, and data with refresh disabled. Real-path manifests were byte-identical before/after (SHA-256 9ca80c9ba949beceab0d4452c62659e7fe2ff295539478132f1e1273ae3d1850); retained evidence manifest SHA-256 is f2b69003df15ef17b05800f3dc4207eea74c9640dfb29d4831fc7510756f700e.
- User-approved scoped verification: the bounded matrix produced 68 passes and one inherited integration exception; the exact native Console matrix passed 4/4. Tests/integration/test_console_edit_resend_e2e.py::test_console_edit_and_resend_full_lifecycle_persist_resume_swipe failed identically at TASK-2703 HEAD and exact pre-task base 0d718e7fb, at the same post-resume transcript assertion. The user explicitly approved closing with this exception and directed all discovered broader-suite failures to a separate follow-up PR from latest dev. The superseded full suite was terminated at 28% and is not claimed as TASK-2703 evidence.
- Static and review: Ruff lint, py_compile, CSS bundle sync, and cumulative/working diff checks passed. One whole-file Ruff format line-wrap is inherited at 0d718e7fb and outside the TASK-2703 hunk. The Impeccable detector returned no findings; independent spec and correctness/accessibility reviews approved with no findings.
- Latest-dev integration (2026-08-19): rebased the 11 task commits cleanly onto `origin/dev` at `a1d6df3f8`. The exact bounded modal/wiring/integration matrix now passes 69/69, including the formerly inherited integration failure, and the four exact native Console nodes pass 4/4. Ruff lint, `py_compile`, CSS bundle reproduction, and diff checks pass. Ruff format still reports only the same whole-file baseline drift reproduced from current `origin/dev`; the task changes only the two-line TCSS height rule.
- PR #1842 review (2026-08-20): Qodo's three test-quality findings were applied without changing production behavior. Terminal evidence sizes and the 3:1 contrast threshold now have one source of truth; compositor cropping rejects off-screen regions with a descriptive assertion (the new regression was RED with the prior `IndexError`); and the focus oracle runs once per size/modal shape while retaining every per-action focus, contrast, cue, and sibling-isolation check. The final related matrix passes 68/68; Ruff lint/format, `py_compile`, CSS bundle reproduction, and diff checks pass.
- Scope review found no outer-modal, copy, DOM, handler, global Button, dependency, config, logging, or generated CSS change. Existing render-frame, terminal-capture, focus-style, scratch-isolation, and Backlog CLI lessons already cover the incidents, so no duplicate lesson was added.
- ADR required: no. This remains a localized Textual layout correction with no architectural boundary change.
<!-- SECTION:NOTES:END -->
