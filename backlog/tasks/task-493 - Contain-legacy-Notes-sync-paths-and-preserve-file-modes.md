---
id: TASK-493
title: Contain legacy Notes sync paths and preserve file modes
status: Done
assignee:
  - '@codex'
created_date: '2026-07-23 14:23'
updated_date: '2026-07-24 14:40'
labels:
  - security
  - privacy
  - notes
  - sync
dependencies:
  - TASK-943
references:
  - backlog/decisions/029-local-private-data-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep legacy Notes synchronization inside a pinned canonical root, preserve existing disk permissions, and retain compatibility with records stored under the originally selected root spelling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The lexical selected root remains a lookup alias while a pinned canonical root identity governs every filesystem access in a sync pass.
- [x] #2 POSIX traversal opens the root once and walks descendants relative to verified directory descriptors without following final or intermediate links.
- [x] #3 Descendant symlinks, junctions/reparse points, multiply linked regular files, cross-device nested mounts, and resolved escapes are never imported or written.
- [x] #4 Existing symlink-spelled sync metadata is matched safely and normalized to the canonical root only after a successful sync update.
- [x] #5 Replacing an existing note preserves its permission bits; new synchronized notes are created as `0600` on POSIX.
- [x] #6 Unsupported Windows containment or replacement checks skip the affected entry with an honest per-file diagnostic rather than claiming safety.
- [x] #7 One rejected entry does not abort unrelated safe files.
- [x] #8 Behavioral tests cover outside and in-root links, hardlinks, directory links, nested-device simulation, final-target and intermediate-parent replacement races, lexical-root compatibility, and `0600`/`0640`/`0644` mode behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md (existing)
Reason: ADR-029 already establishes the legacy Notes containment and mode-preservation policy; TASK-493 implements it without changing ADR-021 authority/recovery scope.

1. Add failing selected-root alias, descendant link/hardlink/mount, race, skip-isolation, Windows fail-closed, and 0600/0640/0644 mode tests.
2. Add a Notes-specific pinned canonical-root descriptor boundary for verified scan/read/private-create/mode-preserving atomic replacement.
3. Route the legacy sync engine through one pinned root per pass while querying lexical and canonical metadata aliases and normalizing only after successful updates.
4. Run focused/broad Notes verification, a canonical /private/tmp sentinel probe, self-review, task closeout, and an isolated commit.

Detailed plan: Docs/superpowers/plans/2026-07-24-legacy-notes-sync-containment.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-029 legacy Notes containment with a Notes-specific PinnedSyncRoot. Each pass resolves and opens the selected root once, scans/reads/writes through verified relative descriptors, rejects symlink/reparse, hardlink, non-regular, cross-device, escape, and identity-race cases, and closes the descriptor on every terminal path. Writes create 0700 descendant directories, create new files as 0600, preserve existing permission bits, and recheck parent/final identity before atomic rename. The engine queries lexical and canonical root spellings, publishes canonical metadata only after successful updates, distinguishes rejected entries from confirmed deletion, and continues safe siblings with bounded skip reasons.

Verification: 32 focused containment/sync tests; complete Notes suite 150 passed/1 skipped; containment plus private-path regression group 67 passed; changed-file Ruff, compileall, and git diff --check. A canonical /private/tmp sentinel probe passed all nine assertions for selected-root aliasing, safe sibling import, outside symlink/hardlink exclusion, 0700/0600 private creation, and 0640 replacement preservation.

ADR: existing backlog/decisions/029-local-private-data-boundary.md; no new ADR required. Detailed plan: Docs/superpowers/plans/2026-07-24-legacy-notes-sync-containment.md.
<!-- SECTION:NOTES:END -->
