---
id: TASK-2101
title: 'LLM destination action census fails on dev: rail-key sets diverged'
status: Done
assignee: []
created_date: '2026-08-04 01:42'
labels:
  - models
  - test-failure
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/ProductionApp/test_llm_destination_actions.py::test_llm_destination_action_census_is_complete_and_removed_controls_are_absent fails on dev (line ~682): the census's expected rail-key set (contains download-models, ollama, ...) no longer matches the actual Models rail (curated, installed, ..., mlx-lm). Found during TASK-2062 Phase 3 Task 5; verified pre-existing by running the identical test on a checkout with zero Phase-3 code. Some post-#1185 dev merge changed the rail or the census without the other. NOTE: TASK-2062 Task 7 (in flight on feat/task-2062-model-browser-phase-3) will REMOVE download-models from the rail, which will change this census again — whoever fixes this should coordinate with that branch or land after it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census test passes on dev with an expected set matching the real rail
- [ ] #2 Root cause stated: which merge changed the rail or census without the other
<!-- AC:END -->

## Closeout audit 2026-10-11

Verdict: DONE (verified against origin/dev @ a3c47c6c72).

Evidence:
- AC1: the census expected set now matches the real rail — `Tests/ProductionApp/test_llm_destination_actions.py:782-796` pins exactly the production `MODELS_RAIL_SECTIONS` view keys (llama-cpp, llamafile, ollama, vllm, onnx, transformers, mlx-lm, curated, installed, external, remote; verified identical to `tldw_chatbook/UI/Screens/llm_screen.py:250-272`). The divergence this task filed was reconciled when the task-2062.3 rail rework landed (commit `4762eaeb0d`, "retire Transformers direct downloader", reworked the census in lockstep with the rail).
- AC1 ticked. AC2 left unticked: no standalone root-cause statement for which post-#1185 merge changed the rail without the census was ever recorded.
