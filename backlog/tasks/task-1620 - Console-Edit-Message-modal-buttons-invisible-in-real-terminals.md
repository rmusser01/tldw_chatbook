---
id: TASK-1620
title: 'Console Edit Message modal: action buttons invisible in real terminals'
status: To Do
assignee: []
created_date: '2026-07-31'
labels: [console, bug, ui]
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
same flow **headless under `app.run_test(size=(200, 50))` renders the
buttons correctly** (probe showed `display=True`, non-zero regions,
fully on-screen y=36..39) — so UI tests cannot catch this, and the bug is
specific to the live terminal driver/paint path. Other Console modals
(Rewind, Console Settings) paint their button rows fine in the same
terminal, so it is something about this modal's actions-row styling, not a
general modal problem.

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
