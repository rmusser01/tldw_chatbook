---
id: TASK-32523
title: Define native reasoning prefill values and reservation state
status: To Do
assignee: []
created_date: '2026-09-13 03:18'
labels: []
dependencies:
  - TASK-32521
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish exact literal seed validation and one revision-owned lifecycle shared by every Console send path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exact nonblank seeds up to 4000 Unicode code points preserve accepted whitespace; invalid text and malformed versioned values fail without revealing seed bodies.
- [ ] #2 Next-send precedence, disable, clear, exclusive reservation, retry retention, and stale completion obey ADR-159 without consuming newer revisions.
- [ ] #3 Unknown provider combinations remain Unverified and capability results distinguish native continuation, tools, and simultaneous response prefill.
<!-- AC:END -->
