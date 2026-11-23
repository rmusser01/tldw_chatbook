---
id: TASK-36952
title: 'Work stream: Library screen Phase A2 — controllers without the screen'
status: Done
created_date: 2026-10-26 01:13
updated_date: '2026-11-23 20:00'
labels:
- architecture-review-2026-10-25
- library
- architecture
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`library_screen.py` is 37,542 lines (the class 36,411) with 1,354 methods. That is 5,312 file lines and 96 methods more than when the decomposition program closed at 32,230 lines. Its `__init__` builds 21 controllers with 413 lambdas, the media and notes controllers take the screen plus 92 and 95 lambdas, and tests pin method bodies to the class (370 unbound calls in 46 files; 66 files reach controllers through the screen). Related: TASK-32463, TASK-35309.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F2; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The media subsystem's controller is constructed without the screen and tested directly
- [x] #2 The lambda and unbound-call ratchets are lower than at filing
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-11-23. Four subtasks, all Done: TASK-36952.1, .2 and .3 (the media method bodies moved into `LibraryMediaController`, in .3.1 to .3.8), and .4.

**AC #1.**
- TASK-41757 (PR #3719) replaced the controller's 43 screen-method callbacks with four typed shell-service protocols on `LibraryMediaHost`. The controller now reaches its host only through that protocol and its 30 named accessors.
- TASK-36952.4 builds the controller on a stand-in host (a bare Textual `Screen`, not `LibraryScreen`) and drives one member of each protocol directly. Its test module never loads `library_screen`.
- Production still constructs the controller with `LibraryScreen`. Owner ruling, 2026-11-22: "The controller still holds the screen object; that is accepted." Owner ruling, 2026-11-23, on how this task closes: "Add a direct test, then close".

**AC #2.** Both ratchets are below their filing measurements, by the pins' own definitions in `Tests/Architecture/test_screen_size_ratchet.py`:
- `_LIBRARY_INIT_LAMBDAS`: 416 measured at dev `26b2e677dd` (the review's 413 counted the controllers only), now 354.
- `_LIBRARY_UNBOUND_TEST_CALLS`: 724 measured at dev `26b2e677dd`, now 511. The description's 370 counted only the named private `_x(fake_self, ...)` subset of the same census.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The media controller owns its bodies, reaches the screen only through typed protocols, and is tested on a stand-in host. The Library lambda and unbound-call ratchets fell from 416 to 354 and from 724 to 511.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
