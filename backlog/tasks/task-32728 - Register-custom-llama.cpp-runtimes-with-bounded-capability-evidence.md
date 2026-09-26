---
id: TASK-32728
title: Register custom llama.cpp runtimes with bounded capability evidence
status: To Do
assignee: []
created_date: '2026-09-17 17:27'
updated_date: '2026-09-17 17:27'
labels:
  - llamacpp
  - catapult-review
  - runtimes
  - feature
dependencies:
  - TASK-32727
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make installed custom binaries selectable with trustworthy identity and evidence instead of relying on a transient executable field.

### Scope

A device-local runtime registry and Lab registration/selection workflow under the approved runtime decision. Collect bounded version/help/device evidence off the UI thread and bind launched processes to the selected runtime identity.

### Explicit exclusions

No managed downloading, automatic probing of arbitrary discovered executables, persistent model paths or model/profile association table.

### Architecture gate

ADR required: yes, supplied by the TASK-32727 decision. ADR path: N/A until that decision is published; copy its actual canonical path into the execution plan before implementation. Reason: executable registration and runtime identity are new durable authority.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_runtimes.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_runtime_probe.py (new)`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`
- `tldw_chatbook/UI/LLM_Management_Window.py`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_runtimes.py Tests/LLM_Management/test_llamacpp_runtime_probe.py Tests/LLM_Management/test_server_lifecycle_resources.py Tests/UI/test_llamacpp_runtime_setup.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Explicit registration and selection survive restart through the approved bounded/versioned/private store; corrupt documents and stale writes remain recoverable without silent replacement.
- [ ] #2 User-selected executables receive fixed-argv no-shell probes with deadlines, output caps and cancellation; malformed or unavailable results remain unknown.
- [ ] #3 Replacing an executable invalidates cached identity/capability/qualification evidence; the next launch revalidates the selected identity.
- [ ] #4 The current launch owns its exact runtime claim until process death, including failed monitoring or cancellation; unregistering a custom entry never deletes the user executable.
- [ ] #5 UI distinguishes registered, executable and model-qualified evidence, retains Browse/Detect compatibility and keeps paths/raw output out of Console and global logs; store/probe/lifecycle/mounted checks pass.
<!-- AC:END -->
