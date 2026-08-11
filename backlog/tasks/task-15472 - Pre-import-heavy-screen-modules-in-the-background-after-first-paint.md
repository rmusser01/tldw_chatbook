---
id: TASK-15472
title: Pre-import heavy screen modules in the background after first paint
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - navigation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: the first visit to a route imports the whole screen module synchronously on the UI thread inside the FIFO-locked navigation worker (`UI/Navigation/screen_registry.py:39-52/:256-278` -> `import_module`): `chat_screen.py` is ~19.9k lines, `library_screen.py` ~26k, `settings_screen.py` ~18.9k. July measured ~161 ms pure import for chat at 11k lines; it has nearly doubled since — plausibly ~1 s on constrained hardware, paid on the first click to each tab and serializing any queued navigation behind the lock.

Fix direction: after first paint, pre-import the top routes from a background THREAD at idle (imports are thread-safe and idempotent; a warm `sys.modules` hit makes `load_screen_class` free). Stability constraints: must not compete with first paint (idle delay), must not change import-error surfacing, and must not break the test seams that patch screens through module aliases (task-3023). Related umbrella: task-2902. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After idle pre-import, the first click to a pre-imported tab spends ~0 ms in import_module (evidence)
- [ ] #2 Cold start and first paint unchanged (A/B measurement)
- [ ] #3 Import errors surface identically to today; test suites that patch screen modules stay green
<!-- AC:END -->
