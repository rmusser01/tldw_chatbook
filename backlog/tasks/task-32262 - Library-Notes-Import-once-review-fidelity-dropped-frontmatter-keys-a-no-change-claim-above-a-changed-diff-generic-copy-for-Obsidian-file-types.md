---
id: TASK-32262
title: >-
  Library Notes Import once review fidelity: dropped frontmatter keys, a no-
  change claim above a changed diff, generic copy for Obsidian file types
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:24'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four review-surface fidelity gaps found on the 71-file vault import:

- non-tag frontmatter keys (`mood: ok`) are dropped silently, with no review row and no receipt line;
- "Unchanged repeat" rows print `Content: no change` directly above a diff showing a changed line -- the classification compares source-to-source while the diff compares stored-note-to-raw-source;
- the repeat-import folder-collision panel opens already showing "That folder name already exists" against a field the user has not touched, with no radio pre-selected;
- `.canvas`, `.png` and `.pdf` all get the generic "This file type is not supported" while `.obsidian` and `.trash` get vault-aware copy; an Obsidian canvas deserves "Obsidian canvas - not a note".

The Obsidian work (task-32129) is otherwise the best-verified part of the screen -- every prior Obsidian complaint is answered and the vault is byte-identical by shasum through Check, Import and an in-app edit -- which is what makes these four visible.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Frontmatter keys that are not imported are reported in the review or the receipt
- [ ] #2 A row classified as no change does not display a diff showing a change: both use one comparison basis
- [ ] #3 The collision panel does not paint an error against an untouched field, and a default is pre-selected
- [ ] #4 Obsidian-specific file types get vault-aware copy rather than the generic unsupported string
- [ ] #5 Covered by tests for the no-change/diff agreement and for the dropped-key report
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Report non-tag frontmatter keys the import drops (parser records them, review row states them).\n2. Show the existing-note diff only on a row whose action writes content, so a no-change row never carries a changed diff.\n3. Pre-select the safe collision default and stop pre-arming the rename error.\n4. Vault-aware unsupported copy for .canvas and the other non-note types.\n5. RED/GREEN tests per item.
<!-- SECTION:PLAN:END -->
