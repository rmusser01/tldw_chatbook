---
id: TASK-32263
title: >-
  Library Notes Import once rewrites wikilinks into note-uuid identifiers
  inside the user's own prose -- design decision to revisit
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - design-decision
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed as a design decision to revisit; task-32129 specified the rewrite, so this is not a defect against the shipped spec.

The importer rewrites `[[wikilinks]]` to `note://df8a9c4b-...`, putting 36-character machine identifiers inside sentences the user wrote. No other tool -- Obsidian included -- can read them back, and Export Markdown then produces a file that is no longer portable, which is the opposite of the local-first promise the same screen makes in copy.

Alternative worth deciding on: keep `[[wikilinks]]` verbatim in the body and resolve them at render time against the imported batch. Same navigation, none of the lock-in, and it removes the UUID scar from the user's prose.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user decision is recorded on rewrite-at-import versus resolve-at-render
- [ ] #2 If revised: an imported note's body round-trips through Export Markdown with its links intact and readable by Obsidian
<!-- AC:END -->
