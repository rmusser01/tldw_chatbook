---
id: TASK-32737
title: Add explicit llama.cpp projector selection with leases and snapshot identity
status: To Do
assignee: []
created_date: '2026-09-17 17:33'
updated_date: '2026-09-17 17:33'
labels:
  - llamacpp
  - catapult-review
  - models
  - feature
dependencies:
  - TASK-32734
  - TASK-32735
  - TASK-32721
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide a safe first-class multimodal setup path instead of requiring raw projector arguments.

### Scope

Managed and session-local external projector selection, pairing suggestions with evidence, exact source admission and dual-artifact process leases integrated with existing snapshots and launch controls.

### Explicit exclusions

No automatic pairing or download from filename guesses, audio-support claims, automatic snapshot reuse or persisted external projector paths.

### Architecture gate

ADR required: yes, supplied by the TASK-32735 decision. ADR path: N/A until that decision is published; implementation must link the actual path and ADR-119. Reason: companion-artifact authority and compatibility identity.

### Affected areas

- `tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py`
- `tldw_chatbook/Model_Artifacts/service.py`
- `tldw_chatbook/Model_Artifacts/leases.py`
- `tldw_chatbook/LLM_Management/snapshot_admission.py`
- `tldw_chatbook/LLM_Management/snapshot_models.py`
- `tldw_chatbook/UI/LLM_Management_Window.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_projector_sources.py Tests/LLM_Management/test_gguf_server_sources.py Tests/LLM_Management/test_snapshot_admission.py Tests/UI/test_llamacpp_projector_setup.py Tests/LLM_Management/test_snapshot_live.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can explicitly select/remove a managed projector or choose a session-only external source; source identity and pairing evidence are shown without presenting heuristics as compatibility proof.
- [ ] #2 Managed model and projector leases are acquired before spawn and held by the exact process until death; partial failure, cancel, restart and stale cleanup cannot release a live artifact early.
- [ ] #3 Typed projector selection conflicts with equivalent raw flags before spawn; existing explicit raw usage keeps its current behavior when the picker is unset.
- [ ] #4 Snapshot admission records and compares the actual projector identity; unknown or changed identity cannot publish/restore incompatible snapshots, and old records remain inspectable.
- [ ] #5 Targeted lease/source/snapshot tests and mounted selection pass; isolated real image inference plus save/restart/restore controls qualify one exact runtime/model/projector combination without extending audio claims.
<!-- AC:END -->
