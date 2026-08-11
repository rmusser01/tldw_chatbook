---
id: TASK-15459
title: Library: compose once per visit
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: a warm Library revisit composes 2-3 times — the initial compose with pre-cache state, an explicit `refresh(recompose=True)` after applying the app-scoped snapshot cache (`library_screen.py` on_mount, `:4992`), and again when `_refresh_local_source_snapshot` reconciles; worker chains (trust-posture etc.) each end in another full-screen recompose (`:7577-7598`). Since screens are rebuilt on every tab switch, this cost is paid on every visit.

Fix direction: seed the cached snapshot in `__init__`/`restore_state` (both run before mount, `app.py:7899/:7922`) so the FIRST compose already renders cached data, then make the reconcile a targeted update. Stability constraint: the on_mount comment explains why it currently recomposes — the restore ordering is subtle; pin the cached-then-fresh data behavior with tests before reordering. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A warm revisit composes exactly once before data reconcile, and reconcile updates in place (or at most one scoped recompose only when data actually changed) — evidence
- [ ] #2 Cached-then-fresh rendering behavior preserved (tests)
- [ ] #3 Library visit latency before/after recorded
<!-- AC:END -->
