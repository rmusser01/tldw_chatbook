---
id: TASK-34844
title: Kobold summarizer ignores its temp and system_message parameters
status: Done
assignee:
  - '@robert'
created_date: '2026-10-12 12:10'
labels: [bug, providers]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-34812 review: summarize_with_kobold accepts temp and system_message but never reads either — the payload hardcodes "temperature": 0.7 (Local_Summarization_Lib.py:638) and the prompt ignores the system message. The TASK-34812 routing fix was therefore behaviorally inert for those two parameters. Either wire them into the payload/prompt (matching sibling local summarizers) or delete the dead parameters so the next reader does not assume they matter; pick one and pin it with a test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 temp and system_message are either honored in the kobold payload/prompt or removed from the signature, with the choice recorded in Implementation Notes.
- [x] #2 A test pins the chosen behavior.
<!-- AC:END -->


## Implementation Notes

**Wired, with default-path preservation** (PR on fix/task-34844-kobold-dead-params). Kobold's generate-style API has no system role, so a PROVIDED system_message is prepended to the prompt (`system\n\n{prompt}`); None keeps the historical prompt byte-identical at the function seam. Composition note (review round 1): via `analyze()`, the substituted default system message is always prepended, so every analyze-routed kobold summary's prompt gains the system prefix by design — matching the llama/vllm/ooba siblings; byte-preservation holds only for direct calls passing None. temp is honored when provided (`float(temp)`), with the historical 0.7 retained as the default. Sibling convention checked against vllm (which uses the shared default string + messages array — not applicable to kobold's single-prompt shape).

Pins: Tests/LLM_Calls/test_kobold_parameter_wiring.py — defaults preserve the historical payload byte-for-byte, provided temp honored, provided system_message prepended; driven through the real function with the transport and config seams stubbed. Battery with kobold config/redirect/leak suites + dispatch routing pins: 23/23.

ADR: not required (parameter honoring inside one provider function; no architectural decision).
