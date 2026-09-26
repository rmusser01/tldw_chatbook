---
id: TASK-32733
title: Show bounded llama.cpp runtime properties and optional metrics
status: To Do
assignee: []
created_date: '2026-09-17 17:30'
updated_date: '2026-09-17 17:31'
labels:
  - llamacpp
  - catapult-review
  - observability
  - feature
dependencies:
  - TASK-32721
  - TASK-32723
  - TASK-32709
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the effective serving configuration and available performance counters visible without collecting prompts or duplicating Console capacity discovery.

### Scope

An exact-target, generation-fenced Lab view of effective context, slots and allowlisted throughput/cache observations. Reuse the existing context resolver contract and add bounded reads of supported optional properties/metrics.

### Explicit exclusions

No automatic enabling of metrics on an existing server, raw slot/prompt bodies, persisted telemetry, monitoring daemon or new Console context precedence.

### Architecture gate

ADR required: no new ADR while observations remain bounded, ephemeral and read-only. ADR path: backlog/decisions/114-llamacpp-lab-console-connection-authority.md. Reason: existing observability privacy plus ADR-052 context provenance; endpoint polling limits must be pinned in the execution plan.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_observations.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_connection.py`
- `tldw_chatbook/Chat/console_context_window.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_observations.py Tests/LLM_Management/test_llamacpp_connection.py Tests/Chat/test_console_context_window.py Tests/UI/test_llamacpp_setup_view.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The view distinguishes configured context, observed per-slot/total capacity and unknown values; it agrees with the existing server-first Console resolver for equivalent evidence.
- [ ] #2 Reads use exact endpoint/model/credential identity, bounded bodies/deadlines, no cross-target redirects and generation fences; remount, replacement, authentication change and cancellation cannot publish stale data.
- [ ] #3 Only allowlisted numeric properties/counters are retained; raw server bodies, prompt text, labels that reveal paths and credential material never reach UI, copy, persistence or logs.
- [ ] #4 Optional missing/disabled/malformed metrics display unavailable with observation timestamps; counter resets and aggregation semantics do not produce invented throughput or cache-reuse claims.
- [ ] #5 Refresh is explicit or bounded while the active view is visible, stops on disposal and does not alter the server; transport/parser/mounted tests cover both supported and unavailable endpoints.
<!-- AC:END -->
