---
id: TASK-32729
title: Add runtime-aware llama.cpp device controls and advanced help
status: To Do
assignee: []
created_date: '2026-09-17 17:28'
updated_date: '2026-09-17 17:28'
labels:
  - llamacpp
  - catapult-review
  - runtimes
  - feature
dependencies:
  - TASK-32725
  - TASK-32728
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent stale static help and expose device selection supported by the binary the user actually chose.

### Scope

Use selected-runtime capability evidence for existing structured tuning, session-local device selection and searchable help. Keep help parsing bounded and unsupported/unknown capability distinct.

### Explicit exclusions

No universal dynamic flag editor, persistent hardware inventory, automatic multi-GPU policy, tensor-split optimizer, or silent profile rewrites.

### Architecture gate

ADR required: no additional ADR beyond the runtime-design decision and ADR-165. ADR path: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md. Reason: tuning-only profiles remain unchanged; device selection and hardware observations stay session-local.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_runtime_probe.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_profiles.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_runtime_probe.py Tests/LLM_Management/test_llamacpp_profiles.py Tests/LLM_Management/test_llamacpp_launch_contract.py Tests/UI/test_llamacpp_runtime_setup.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Controls/help use evidence for the exact selected binary and invalidate on replacement; known unsupported flags are diagnosed before spawn without deleting saved tuning.
- [ ] #2 Unavailable help/device evidence is labeled unknown with a deliberate expert path, rather than fabricated support or an implicit compatibility pass.
- [ ] #3 Device selection accepts only current observed admissible device identifiers, stays session-local and rejects conflicts with expert device options.
- [ ] #4 Changing runtime/device evidence cannot silently retarget a running launch, reuse a stale device selection or modify Console defaults.
- [ ] #5 Searchable bounded help and keyboard controls pass mounted tests for old/new/unknown capabilities; no persisted GPU UUIDs, raw probe output or aggregated VRAM estimates are introduced.
<!-- AC:END -->
