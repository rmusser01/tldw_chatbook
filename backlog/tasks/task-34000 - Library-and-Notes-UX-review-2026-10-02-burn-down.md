---
id: TASK-34000
title: >-
  Library + Library ▸ Notes UX review 2026-10-02: burn down the verified
  first-timer and power-user findings
status: To Do
assignee: []
created_date: '2026-10-03 09:00'
labels:
  - library
  - notes
  - ux-review-2026-10-02
  - umbrella
dependencies: []
references:
  - qa/notes-library-ux-review-2026-10-02/README.md
  - qa/notes-library-ux-review-2026-10-02/report.md
  - qa/notes-library-ux-review-2026-10-02/findings.md
  - qa/notes-library-ux-review-2026-10-02/findings.json
  - qa/notes-library-ux-review-2026-10-02/improvements/ranked.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: a Sr-designer/HCI review of Library and its Notes sub-destination (origin/dev 2d34cbf80d) walked seven personas through real first-time and power-user workflows in the live app. A first-timer, a Notes power user, a Library operator, a keyboard-only low-vision user, a stress tester and a researcher each hit failures. Library scored 17/40 and Notes 16/40 on Nielsen's heuristics; both prior critiques had 25/40, but most of the drop is new coverage (quit, sync edge cases, export, the Library → Console loop) rather than regressions. The review produced 103 findings. An independent refute-first verifier re-reproduced every one at HEAD, and each was triaged against the backlog and ADRs; none was refuted.

The worst failures break the product's local-first promise:
- Notes loses typed text on Ctrl+Q.
- One ordinary edit jams a sync folder while the UI says "Ready".
- Export silently overwrites files.
- Media Export… freezes the app.
- Analysis can never find its provider.
- The Study hand-off crashes.

Owner priority (2026-10-03): the data-integrity P0s land first, in this order:
1. TASK-34000.1 (N-01)
2. TASK-34000.2 (N-02)
3. TASK-34000.3 (N-11)
4. TASK-32633 for N-03. Its delete_note and restore_note paths should land first.

The other P0s follow: .4 (L-01), .5 (L-02), .6 (S-05).

Owner rulings recorded in the subtasks:
- Media "Use in Console" links the item to the active workspace on use (TASK-34000.24).
- Note links get a `[[` picker, and a typed `[[Title]]` that matches exactly one note records a backlink (TASK-34000.11).
Still open for an owner ruling (see report.md "Questions"):
- ADR-031 save/key grammar: N-33 in .40, and S-28.
- A side-by-side reading desk: S-02 in .25 is scoped to preserving state. A desk needs an ADR-086/084 amendment and per-screen approval.

Subtasks: .1–.27 each hold one P0 or P1 finding. .28–.41 batch the P2/P3 findings by surface.

These 12 findings are owned by existing tasks, so no new subtask was filed. Re-verify each against the review evidence when its owner closes:
- N-03 → TASK-32633
- L-18 → TASK-32381, TASK-32382
- N-18 → TASK-32451
- S-19 → TASK-33620.7, TASK-33621.20
- L-12 → TASK-32304, TASK-31568
- L-14 → TASK-32384, TASK-28021
- S-11 → TASK-32650, TASK-31571, TASK-31569
- S-13 → TASK-32649, TASK-32301
- L-37 → TASK-32307
- S-24 → TASK-32573, TASK-32451, TASK-32378
- S-26 → TASK-32590, TASK-32607, TASK-32589
- S-28 → TASK-31569

Partly owned findings are filed as subtasks for the uncovered remainder only: L-02 (TASK-28018), N-05 (TASK-32513/32514), N-16 and N-35 (TASK-32578), N-14 (TASK-32635).

L-29 and S-23 cite Done tasks (task-31635; TASK-32126 and TASK-32355) but still reproduce. Their residue is filed in .39 and .41.

Evidence: qa/notes-library-ux-review-2026-10-02/ holds:
- report.md: scores, journeys, roadmap.
- findings.md and findings.json: the ledger.
- journeys/: the persona step logs.
- verify/: the independent re-reproductions.
- improvements/ranked.md: the ranked improvement roadmap. Strategic items there need ADR or owner rulings before they can become tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four data-integrity P0s (TASK-34000.1, .2, .3 and TASK-32633's delete/restore paths for N-03) are Done, each with a regression test that failed on 2d34cbf80d
- [ ] #2 All remaining P0 subtasks (TASK-34000.4, .5, .6) are Done
- [ ] #3 Every P1 subtask (TASK-34000.7–.27) is Done, or explicitly deferred with an owner ruling recorded in the subtask
- [ ] #4 The 12 findings owned by existing tasks are re-checked against the review evidence when their owners close, and any residue is filed
- [ ] #5 A follow-up dual-assessment critique of Library and Notes runs after the P0 and P1 subtasks land and records its scores against this review's 17/40 and 16/40 baselines
<!-- AC:END -->
