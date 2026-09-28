---
id: TASK-32744
title: Resolve the deferred environment polling boot-closure regression test
status: To Do
assignee: []
created_date: '2026-09-17 17:38'
updated_date: '2026-09-17 17:38'
labels:
  - llamacpp
  - catapult-review
  - test-debt
  - bug
dependencies:
  - TASK-32721
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore trustworthy deferred-I/O coverage for an independently reproduced environment-poll assertion failure.

### Scope

Investigate Tests/Packaging/test_console_interaction_boot_closure.py::test_console_defers_setup_and_environment_io_until_requested, where the second environment poll has an unexpected queued-job count.

### Explicit exclusions

No assumed llama.cpp regression, startup-budget increase, arbitrary sleep, full-suite cleanup or unconditional environment probing at boot.

### Architecture gate

ADR required: no new ADR for a contract-preserving repair. ADR path: backlog/decisions/036-application-service-composition-lifecycle.md. Reason: preserve lazy startup and owned worker behavior while correcting the proven cause.

### Affected areas

- `Tests/Packaging/test_console_interaction_boot_closure.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/Packaging/test_console_interaction_boot_closure.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduction separates child-program deferred-import assertions from first/second poll worker admission and records current and baseline outcomes.
- [ ] #2 The actual scheduling contract determines the expected jobs; the repair uses explicit scheduler gates or owner state rather than timing sleeps or relaxed assertions.
- [ ] #3 Startup still defers setup/environment I/O until requested and repeated requests cannot create duplicate owned work or run it on the UI thread.
- [ ] #4 The complete named file and any directly affected startup/preload guards pass with unchanged budgets; the milestone verification report links the cause and result.
<!-- AC:END -->
