---
id: TASK-32732
title: Qualify llama.cpp runtime lifecycle and native platform support
status: To Do
assignee: []
created_date: '2026-09-17 17:30'
updated_date: '2026-09-17 17:30'
labels:
  - llamacpp
  - catapult-review
  - runtimes
  - verification
dependencies:
  - TASK-32731
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn runtime support claims into recorded evidence, including the Windows pipe gap from the first milestone.

### Scope

A reproducible opt-in matrix and evidence report for the platform lanes approved by the runtime ADR: custom/managed launch, inference, install failure, rollback, process/pipe cleanup and model/runtime deletion fences.

### Explicit exclusions

No benchmark ranking, automatic snapshot reuse, claims for unexecuted platforms or downloading large models as an implicit test prerequisite.

### Architecture gate

ADR required: no new ADR. ADR path: backlog/decisions/114-llamacpp-lab-console-connection-authority.md. Reason: qualification preserves runtime-design ownership, ADR-025 leases and ADR-165 diagnostics.

### Affected areas

- `Tests/LLM_Management/test_llamacpp_live_smoke.py`
- `Tests/LLM_Management/test_llamacpp_diagnostics.py`
- `Tests/LLM_Management/test_llamacpp_runtime_live.py (new)`
- `Docs/superpowers/reviews/`
- `Docs/User_Guide/lab.md`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_runtime_live.py Tests/LLM_Management/test_llamacpp_live_smoke.py Tests/LLM_Management/test_llamacpp_diagnostics.py Tests/LLM_Management/test_server_lifecycle_resources.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A support table names exact OS/architecture/backend/runtime/model combinations and separates passed, failed and not-run evidence; an unexecuted required lane prevents completion or must be explicitly descoped in the ADR and task.
- [ ] #2 Supported lanes exercise real health/exact model/inference through production launch and adoption, then Stop and rollback, retaining the old installation after a failing candidate.
- [ ] #3 Native Windows child-pipe tests cover large output, cancellation, child exit and monitor failure, or Windows is explicitly left unqualified; POSIX fixtures cannot stand in for this evidence.
- [ ] #4 Retained live processes keep runtime/model leases when monitoring fails and release them only after verified death; unrelated servers and user assets remain untouched.
- [ ] #5 Harnesses use isolated temporary profiles/ports, explicit asset selection and owned cleanup; guides publish reproducible commands and actual limits without treating skips as successful qualification.
<!-- AC:END -->
