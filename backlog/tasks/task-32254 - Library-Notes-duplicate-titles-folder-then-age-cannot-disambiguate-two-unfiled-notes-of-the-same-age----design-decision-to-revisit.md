---
id: TASK-32254
title: >-
  Library Notes duplicate titles: folder-then-age cannot disambiguate two
  unfiled notes of the same age -- design decision to revisit
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - design-decision
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed as a design decision to revisit, **not** as a defect. Task-32137 shipped exactly as specified and `test_duplicate_titles_render_folder_and_age_suffixes` pins the behaviour with rows aged 2h and 5d.

The spec hole: two notes titled `Reading list`, both unfiled and both the same age, render byte-identically as `Reading list . Unfiled . 2m` (`R/caps/01`). Unfiled-and-recent is the default state for precisely the notes a user is most likely to have just created twice, so the discriminator is a no-op in the case it is most needed. The guide's claim ("each row also names its folder -- 'Reading list . Unfiled . 2h'") is technically true and functionally false.

Proposed discriminator: fall back to the first line of the body, or to an absolute timestamp, when folder and age both match. This changes a pinned test, so it needs a product decision first.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user decision is recorded on whether folder-then-age is sufficient
- [ ] #2 If revised: two rows sharing title, folder and age are distinguishable at a glance, and `test_duplicate_titles_render_folder_and_age_suffixes` is extended to cover that case rather than only the 2h/5d one
<!-- AC:END -->
