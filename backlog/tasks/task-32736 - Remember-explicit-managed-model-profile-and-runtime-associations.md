---
id: TASK-32736
title: Remember explicit managed-model profile and runtime associations
status: To Do
assignee: []
created_date: '2026-09-17 17:32'
updated_date: '2026-09-17 17:33'
labels:
  - llamacpp
  - catapult-review
  - models
  - feature
dependencies:
  - TASK-32728
  - TASK-32735
  - TASK-32722
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user recall a preferred tuning profile and runtime for an exact managed model without surprising implicit changes.

### Scope

Implement the approved private association store and explicit Lab remember/recall/remove actions using opaque managed artifact, profile and runtime references.

### Explicit exclusions

No external path-keyed map, automatic server start, executable/model paths inside tuning profiles, automatic Console sampling changes or projector feature.

### Architecture gate

ADR required: yes, supplied by the TASK-32735 decision. ADR path: N/A until that decision is published; copy its canonical path into the execution plan. Reason: durable cross-owner references need that explicit contract.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_associations.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_profiles.py`
- `tldw_chatbook/LLM_Management/llamacpp_runtimes.py (new)`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_associations.py Tests/LLM_Management/test_llamacpp_profiles.py Tests/UI/test_llamacpp_setup_view.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An explicitly saved association survives restart for the exact managed artifact identity; selecting a different revision or shard group cannot inherit it by filename.
- [ ] #2 Recall is visible and does not overwrite an edited draft without an explicit action; it affects the next launch and never starts/restarts a server.
- [ ] #3 Missing profiles/runtimes/artifacts produce a recoverable unresolved association and support explicit removal without deleting referenced resources.
- [ ] #4 External model/projector paths, credentials, endpoints and Console sampling never enter the association document or tuning profiles; stale concurrent saves do not lose newer changes.
- [ ] #5 Persistence, removal and source-switch tests plus mounted remember/recall flows pass; the guide explains association ownership and runtime-default behavior.
<!-- AC:END -->
