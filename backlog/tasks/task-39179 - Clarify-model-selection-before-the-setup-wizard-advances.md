---
id: TASK-39179
title: Clarify model selection before the setup wizard advances
status: Done
assignee: []
created_date: 2026-11-11 11:03
labels:
- uat
- onboarding
- providers
dependencies: []
priority: low
updated_date: 2026-11-11 20:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Quick setup can successfully test a local provider and discover exactly one recommended model, then leave the provider and model incomplete when the user advances without selecting the radio row. The Summary's generic provider error makes the successful connection look lost.

### Reproduction

1. On local `origin/dev` `6bd97b112f`, start Quick setup on a fresh profile. Select **llama.cpp** and enter an OpenAI-compatible local endpoint whose `/v1/models` lists exactly one model.
2. Click **Test connection** and confirm **Connected — llama.cpp returned 1 model**.
3. Click **Next** to open **Pick a default model**. The only row is marked **recommended** and its radio is empty.
4. Click **Next** without selecting the row. Continue past Voice and Protect.
5. Observe **Provider — no credentials or saved endpoint** and **Default model — not selected** on Summary. Go back, select the model, then advance. Summary now offers **Start chatting**.

### Evidence

- [Successful provider test](../../qa/uat-core-workflows-2026-11-11/wide-provider-tested.txt)
- [Unselected recommended model](../../qa/uat-core-workflows-2026-11-11/wide-model-choice.txt)
- [Summary after advancing without a model](../../qa/uat-core-workflows-2026-11-11/wide-summary.txt)
- [Summary after explicitly selecting the model](../../qa/uat-core-workflows-2026-11-11/wide-summary-ready.txt)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 When a connected provider lists one model, the Model step makes clear what will happen if the user presses Next without selecting it.
- [x] #2 If the user skips model selection, Summary identifies the missing model and accurately describes the provider connection or save state.
- [x] #3 Selecting the discovered model and finishing setup leaves Console ready for the first send.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: No
ADR path: N/A
Reason: Clarify the existing optional model selection and staged setup state.

Cover skipped and explicitly selected models, explain the Next action and report the staged provider accurately, then verify setup and the first-send handoff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Model step explains what Next does with an unselected model. Summary names the staged provider and explains why setup remains unsaved when the model step is skipped. The selected-model path still saves and hands off to Console. The one-model regression passed in the combined run and the final-copy rerun, and the focused Console handoff test passed. Existing ADR-217 applies.
<!-- SECTION:NOTES:END -->
