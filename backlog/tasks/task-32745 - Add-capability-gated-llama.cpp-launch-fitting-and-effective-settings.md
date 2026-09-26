---
id: TASK-32745
title: Add capability-gated llama.cpp launch fitting and effective settings
status: To Do
assignee: []
created_date: '2026-09-17 17:34'
updated_date: '2026-09-17 17:43'
labels:
  - llamacpp
  - catapult-review
  - tuning
  - feature
dependencies:
  - TASK-32729
  - TASK-32734
  - TASK-32738
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users fit a chosen model and configuration to selected hardware while keeping uncertainty and resulting settings visible.

### Scope

A separate launch-time guidance view and explicit supported runtime-fit controls for the selected binary; reuse model metadata and bounded observations to show configured and effective values.

### Explicit exclusions

No new universal memory formula, automatic profile overwrites, background trial launches, summed GPUs, speculative decoding preset or changes to download ratings.

### Architecture gate

ADR required: yes, supplied by the TASK-32738 decision. ADR path: N/A until that decision is published; execution must link its canonical path and ADR-080/119. Reason: fitting and effective-state eligibility follow the approved new contract.

### Affected areas

- `tldw_chatbook/LLM_Management/llamacpp_fit.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_runtime_probe.py (new)`
- `tldw_chatbook/LLM_Management/llamacpp_observations.py (new)`
- `tldw_chatbook/LLM_Management/snapshot_admission.py`
- `tldw_chatbook/LLM_Management/llamacpp_profiles.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_fit.py Tests/LLM_Management/test_snapshot_admission.py Tests/Model_Artifacts/test_machine_memory.py Tests/UI/test_llamacpp_fit.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fit controls appear only with compatible selected-runtime evidence and require an explicit action; unsupported/unknown builds retain ordinary manual launch.
- [ ] #2 The view names the selected model/runtime/context/cache/slots/devices and labels estimates versus observations without adding dedicated device capacities or double-counting unified memory.
- [ ] #3 Effective post-launch values are generation-bound and displayed separately from profile inputs; fit never silently rewrites a saved profile or active Console defaults.
- [ ] #4 Missing compatibility-relevant effective values keep snapshot Save/Restore unavailable; changed values invalidate previous compatibility evidence.
- [ ] #5 Targeted supported/unknown/OOM/cancel/stale-result tests and mounted controls pass; one opt-in real fit launch records effective settings and snapshot eligibility, with no guarantee inferred from an estimate.
<!-- AC:END -->

## Renumbering provenance

Created through Backlog CLI as TASK-32739 at 2026-09-17 17:34 UTC. The final sweep found a concurrent TASK-32739, Preserve model discovery selection and result ownership in Settings, created at 17:35 UTC in the component-pattern-library worktree and already In Progress. This uncommitted planning record voluntarily moved to TASK-32745 after fresh all-ref/all-worktree filename and content checks, preserving the active implementation in the other worktree despite this record being one minute older. Its roadmap links were updated; the original number belongs to that Settings task. This is a local allocation adjustment, not a rewrite of published Git history.
