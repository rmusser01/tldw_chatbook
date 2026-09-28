---
id: TASK-32726
title: Make advanced llama.cpp launch transport readiness explicit
status: To Do
assignee: []
created_date: '2026-09-17 17:26'
updated_date: '2026-09-17 17:26'
labels:
  - llamacpp
  - catapult-review
  - foundation
  - feature
dependencies:
  - TASK-32721
  - TASK-32723
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Avoid probing a misleading HTTP root when expert launch options select TLS or an API prefix.

### Scope

Recognize supported transport-changing argument forms during local launch admission. Derive a safe exact readiness target when sufficient evidence exists; otherwise give an explicit unsupported automatic-verification state and the existing endpoint-check recovery route.

### Explicit exclusions

No TLS verification bypass, new certificate store, proxy/router orchestration, or expansion of snapshot management beyond ADR-119.

### Architecture gate

ADR required: no new ADR. ADR path: backlog/decisions/114-llamacpp-lab-console-connection-authority.md. Reason: correct ordinary readiness target selection within the accepted endpoint contract; ADR-119 retains its narrower snapshot transport.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_connection.py`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`
- `tldw_chatbook/LLM_Management/snapshot_admission.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_launch_contract.py Tests/LLM_Management/test_llamacpp_connection.py Tests/LLM_Management/test_snapshot_admission.py Tests/UI/test_llamacpp_setup_view.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Supported TLS/prefix argument forms never cause automatic verification to silently probe the default HTTP root.
- [ ] #2 Supported target derivation keeps the exact prefix, reserved model alias and launch generation; ambiguous or unsupported transport reports a bounded recovery message without claiming readiness.
- [ ] #3 HTTPS uses normal certificate verification and exact-endpoint credentials; redirects cannot change the verification target.
- [ ] #4 External endpoint verification remains available without gaining local Stop authority; late results and remounts retain current ownership fences.
- [ ] #5 Loopback HTTP/prefix and local TLS fixture checks cover success, wrong certificate and stale evidence; snapshots remain unavailable for unsupported transports and the guide states the boundary.
<!-- AC:END -->
