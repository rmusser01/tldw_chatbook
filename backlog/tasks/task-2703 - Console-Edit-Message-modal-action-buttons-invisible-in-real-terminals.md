---
id: TASK-2703
title: 'Console Edit Message modal: action buttons invisible in real terminals'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31'
updated_date: '2026-08-13 20:54'
labels:
  - console
  - bug
  - ui
dependencies: []
---

## Description (the why)

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

## Acceptance Criteria (the what)

- [ ] Cancel / Save / Edit & resend are visible in a real terminal (tmux
      and a normal TTY) for both USER and non-USER targets of the modal.
- [ ] Focus is visibly indicated when Tab reaches each button.
- [ ] A regression check exists that would catch a live-terminal-only
      disappearance (at minimum: a note in the test explaining why the
      headless assertion is insufficient, plus a geometry assertion that
      holds under the real stylesheet).
- [ ] The User Guide quirk note in
      `Docs/User_Guide/console/branching-and-rewind.md` is updated/removed
      to match the fixed behavior.

## Implementation Plan (the how)

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
   plus the required full suite, record evidence, and close the task only after
   final candidate verification.

Detailed execution plan:
`Docs/superpowers/plans/2026-08-13-task-2703-console-edit-modal-paint.md`.
