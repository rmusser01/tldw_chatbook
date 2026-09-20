---
id: TASK-32878
title: Sync GGUF source-evidence workflow node IDs with the test file
status: To Do
assignee: []
created_date: '2026-09-20 09:30'
labels:
  - ci
  - dev-blocker
dependencies: []
references:
  - .github/workflows/task-2062-2-gguf-source-evidence.yml
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dev-side CI breakage (found blocking PRs #2746/#2747/#2750, 2026-09-20): the GGUF source-evidence workflow's full-app loop references node IDs the test file no longer collects. Evidence: the workflow at origin/dev lines 119-121 lists test_llm_gguf_source_modes.py::test_lazy_mount_cached_inventory_preserves_handoff_awaiting_fresh_read and ::test_inventory_during_lazy_selector_mount_is_replayed_after_leaving_pane[False]/[True], but the test file at origin/dev defines both as plain unparametrized async defs (:518/:578) -- CI logs show "ERROR: not found ... (no match in any of [<Module test_llm_gguf_source_modes.py>])" on all three OSes (run 35491798980). Every open PR that runs this workflow against current dev fails it; the branch content is irrelevant (verified: identical branch content passed the job against an older dev). Fix on dev: either restore the parametrization the workflow expects or update the three node IDs to what the file collects -- and confirm the async tests collect under the private-profile child (PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, no pytest-asyncio).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The workflow's node list matches what Tests/UI/test_llm_gguf_source_modes.py collects at dev, and the GGUF source-evidence job is green on dev and on PRs #2746/#2747/#2750
<!-- AC:END -->
