---
id: TASK-32730
title: Install selected llama.cpp releases into an owned runtime store
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
  - TASK-32728
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow explicit installation of a supported release while preserving provenance and the currently usable runtime.

### Scope

Release selection, bounded transfer, safe staging/extraction, integrity/provenance recording, crash recovery and atomic publication into the separate executable store approved by the runtime ADR.

### Explicit exclusions

No background updates, source compilation, arbitrary repository/package execution, auto-start after install, or replacement model downloader.

### Architecture gate

ADR required: yes, supplied by the TASK-32727 decision. ADR path: N/A until that decision is published; implementation must link the published path. Reason: acquisition trust and installation lifecycle are governed by that decision.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_runtime_install.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_runtimes.py (new)`
- `tldw_chatbook/Model_Artifacts/fetch.py (reuse where compatible)`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_runtime_install.py Tests/LLM_Management/test_llamacpp_runtimes.py Tests/Model_Artifacts/test_stream_fetch.py Tests/UI/test_llamacpp_runtime_setup.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An explicit selected release/asset can be installed only within the approved source and platform policy; UI records exact release/build/source and the verification actually available.
- [ ] #2 Interrupted/cancelled/corrupt downloads and interrupted extraction never publish an executable installation or alter the active/last-known-good runtime.
- [ ] #3 Archive traversal, escaping links, decompression/entry limits, insufficient space and unexpected layouts fail safely with bounded diagnostics and owned staging cleanup.
- [ ] #4 Published installations are immutable under their identity and cannot overwrite an in-use runtime; executable capability checks do not imply model qualification.
- [ ] #5 Fixture transfer/extraction/crash-recovery tests and mounted install/cancel/retry flow pass; an opt-in real supported release install is recorded with exact provenance before shipping the supported lane.
<!-- AC:END -->
