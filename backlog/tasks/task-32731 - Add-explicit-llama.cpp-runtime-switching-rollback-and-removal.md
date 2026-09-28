---
id: TASK-32731
title: Add explicit llama.cpp runtime switching rollback and removal
status: To Do
assignee: []
created_date: '2026-09-17 17:29'
updated_date: '2026-09-17 17:29'
labels:
  - llamacpp
  - catapult-review
  - runtimes
  - feature
dependencies:
  - TASK-32730
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recover from a failed runtime upgrade without losing a known usable installation or deleting files still in use.

### Scope

Select the next-launch runtime, explicitly return to a last-known-good installation, and remove unused Chatbook-owned installations with truthful blocked/recovery states.

### Explicit exclusions

No live binary replacement, automatic process restart, automatic rollback after an uncertain failure, or deletion of custom external executables.

### Architecture gate

ADR required: no additional ADR beyond the TASK-32727 decision. ADR path: N/A until that decision is published; implementation must link the published path. Reason: implement its explicit selection, rollback and lifecycle contract.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_runtimes.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_runtime_install.py (new)`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py`
- `tldw_chatbook/LLM_Management/snapshot_admission.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_runtimes.py Tests/LLM_Management/test_llamacpp_runtime_install.py Tests/LLM_Management/test_server_lifecycle_resources.py Tests/LLM_Management/test_snapshot_admission.py Tests/UI/test_llamacpp_runtime_setup.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selection changes only the next launch and clearly distinguishes the active runtime; replacing a live process requires ordinary explicit Stop and Start.
- [ ] #2 Last-known-good evidence is tied to exact runtime/model/launch qualification and retained across failed installation or qualification; rollback selects the retained identity without claiming all models are compatible.
- [ ] #3 Removal is limited to owned unused installations, remains blocked while an exact process lease is retained and cannot escape the runtime store through links or stale identifiers.
- [ ] #4 Changing or rolling back a runtime invalidates old readiness and incompatible snapshot evidence without deleting snapshot records or rewriting Console defaults.
- [ ] #5 Failed-install, failed-start, stubborn-child, concurrent removal and rollback tests pass; the guide documents recovery and custom-entry unregister versus owned-file removal.
<!-- AC:END -->
