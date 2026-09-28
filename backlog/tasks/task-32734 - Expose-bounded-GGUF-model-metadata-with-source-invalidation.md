---
id: TASK-32734
title: Expose bounded GGUF model metadata with source invalidation
status: To Do
assignee: []
created_date: '2026-09-17 17:31'
updated_date: '2026-09-17 17:31'
labels:
  - llamacpp
  - catapult-review
  - models
  - feature
dependencies:
  - TASK-32721
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users understand a selected model without treating file metadata as runtime compatibility or available serving context.

### Scope

Extend the existing bounded GGUF reader with safe parameter/size metadata, training context and modality hints; show those fields for managed and explicitly selected external models with source-bound ephemeral caching.

### Explicit exclusions

No new parser, remote range-header scanner, durable external path cache, automatic pairing or compatibility claims from names and hints.

### Architecture gate

ADR required: no new ADR. ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md. Reason: bounded structural observations remain within current artifact authority; retain ADR-080 estimation semantics.

### Affected areas

- `tldw_chatbook/Model_Artifacts/gguf_admission.py`
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py`
- `tldw_chatbook/UI/LLM_Management_Window.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_gguf_admission.py Tests/LLM_Management/test_gguf_source_modes.py Tests/UI/test_llm_gguf_source_modes.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Useful parameter/size, training-context and modality fields are shown only when correctly typed and admissible; absent or malformed optional metadata stays unknown without inventing values.
- [ ] #2 Total parser budgets remain bounded for keys, strings, arrays and tensors, including useful keys appearing after the first 128 entries.
- [ ] #3 Cached observations bind to exact managed artifact or current external source identity and invalidate on file replacement or source selection changes; external paths are not persisted.
- [ ] #4 Training context is labeled separately from configured/observed serving context and memory scenarios; modality hints offer setup guidance without asserting successful inference.
- [ ] #5 Adversarial GGUF and source-change tests plus mounted model details pass; existing STT structural consumers and managed import remain compatible.
<!-- AC:END -->
