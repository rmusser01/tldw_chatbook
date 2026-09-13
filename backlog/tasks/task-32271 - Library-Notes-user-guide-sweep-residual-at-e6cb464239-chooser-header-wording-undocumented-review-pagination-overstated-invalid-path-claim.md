---
id: TASK-32271
title: >-
  Library Notes user guide sweep residual at e6cb464239: chooser header
  wording, undocumented review pagination, overstated invalid-path claim
status: In Progress
assignee: []
created_date: '2026-09-10 18:05'
updated_date: '2026-09-13 00:36'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The task-32141 guide sweep landed and the evidence assessor's 38-row conformance table is mostly VERIFIED -- Import once's exclusivity, the Obsidian skip rules, Info Properties, the delete copy and receipt shape are all verbatim-accurate. Three claims are still off, and none of them belongs to a code task in this batch:

- the guide says the chooser header "reads **Add from files**" until you choose; live it reads "Add files to Library notes.";
- the review's pagination (`Page 1 of 3` with per-page group counts) is entirely undocumented;
- "An invalid path shows an inline reason and leaves the dialog open" is overstated while the reason is painted into the dialog border (the code side of that is task-32251; this is the doc stamp).

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Merge origin/dev (7159fc0b99, every wave-3 code group) into the docs branch.
2. Enumerate every guide claim the wave touched: `git log 4a14b3f36f..origin/dev` on the five guides plus the "Verified against" stamps for 32143, 32146, 32186, 32242-32272, 32294-32298, 32389/32390, 32467.
3. Walk each claim live on dev at 235x52 and 100x30 (60x24 for the below-64-column claim) on a seeded scratch profile with a git-backed vault under `$HOME/.cache/tldw-crit/t12/`; one capture per claim under `wave3-caps/docs-sweep/`.
4. Fix every contradiction with the supersession treatment; add the undocumented pager; consolidate duplicate stamps; refresh the stamp with the commit walked.
5. Riders for defects found; renumber the colliding task-32472; run Tests/Docs and the backlog-id uniqueness test; inventory + bundle checks; PR.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each claim above matches the live surface, or is corrected in the guide
- [ ] #2 The 'Verified against' stamp on `Docs/User_Guide/library/notes.md` is refreshed with the commit it was checked at
<!-- AC:END -->
