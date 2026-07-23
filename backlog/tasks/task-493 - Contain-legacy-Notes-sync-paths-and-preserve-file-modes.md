---
id: TASK-493
title: Contain legacy Notes sync paths and preserve file modes
status: To Do
assignee: []
created_date: '2026-07-23 14:23'
updated_date: '2026-07-23 14:23'
labels:
  - security
  - privacy
  - notes
  - sync
dependencies:
  - TASK-488
references:
  - backlog/decisions/022-local-private-data-boundary.md
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
- [ ] #1 The lexical selected root remains a lookup alias while a pinned canonical root identity governs every filesystem access in a sync pass.
- [ ] #2 POSIX traversal opens the root once and walks descendants relative to verified directory descriptors without following final or intermediate links.
- [ ] #3 Descendant symlinks, junctions/reparse points, multiply linked regular files, cross-device nested mounts, and resolved escapes are never imported or written.
- [ ] #4 Existing symlink-spelled sync metadata is matched safely and normalized to the canonical root only after a successful sync update.
- [ ] #5 Replacing an existing note preserves its permission bits; new synchronized notes are created as `0600` on POSIX.
- [ ] #6 Unsupported Windows containment or replacement checks skip the affected entry with an honest per-file diagnostic rather than claiming safety.
- [ ] #7 One rejected entry does not abort unrelated safe files.
- [ ] #8 Behavioral tests cover outside and in-root links, hardlinks, directory links, nested-device simulation, final-target and intermediate-parent replacement races, lexical-root compatibility, and `0600`/`0640`/`0644` mode behavior.
<!-- AC:END -->
