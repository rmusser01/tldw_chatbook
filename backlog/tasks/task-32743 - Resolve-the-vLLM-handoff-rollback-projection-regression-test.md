---
id: TASK-32743
title: Resolve the vLLM handoff rollback projection regression test
status: To Do
assignee: []
created_date: '2026-09-17 17:37'
updated_date: '2026-09-17 17:37'
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
Restore meaningful rollback coverage for a pre-existing failure reproduced during llama.cpp milestone verification.

### Scope

Investigate Tests/UI/test_console_provider_apply_defaults_flow.py::test_vllm_console_handoff_restores_projections_when_rollback_sync_fails[_sync_console_chat_core_state], which expected two sync calls but observed one when fault injection occurred before adoption.

### Explicit exclusions

No assumed production bug, weakening rollback assertions, broad Console fixture cleanup or changes to accepted endpoint/default ownership.

### Architecture gate

ADR required: no new ADR for a contract-preserving repair. ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md. Reason: reproduce and repair test setup or implementation under the existing transaction contract.

### Affected areas

- `Tests/UI/test_console_provider_apply_defaults_flow.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_provider_apply_defaults_flow.py Tests/UI/test_llamacpp_consumers.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The recorded node is reproduced against the current checkout and a baseline without the llama.cpp change, or evidence records that it was independently resolved.
- [ ] #2 Fault injection distinguishes pre-adoption projection construction from transaction/rollback failure; a successful control proves the actual handoff path was reached.
- [ ] #3 The test or a proven production defect is repaired without losing provider/model/endpoint/generation and UI projection rollback assertions.
- [ ] #4 The named file and neighboring llama.cpp consumer tests pass; the verification report links the root cause, command and outcome instead of retaining an unexplained exclusion.
<!-- AC:END -->
