---
id: TASK-31753
title: Update rewind integration gateway to the current compaction protocol
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:11'
updated_date: '2026-09-05 22:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore truthful end-to-end rewind summary evidence where the transport double predates required prepared-request and auxiliary-completion capabilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The real persisted restore/edit/summarize/resume journey reaches current summary admission and preserves exact span, outgoing payload, durable-memory and restart assertions.
- [ ] #2 The gateway fixture implements current prepared-request and auxiliary-completion contracts without bypassing production persistence, memory or provider guards.
- [ ] #3 The complete rewind integration file and related summary checks pass, with the prior save-first refusal documented as baseline.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; test-only current provider protocol alignment. 1. Read task and compare _SequencedCapturingGateway with canonical SummaryGateway in test_console_rewind_summarize.py. 2. Preserve real DB/store/controller lifecycle and sequenced payload capture while adding canonical prepare_chat_request and complete_auxiliary behavior. 3. Run complete rewind integration and relevant summary files; retain exact accepted/span/persist/resume assertions or report genuine product issues. 4. Root reviews a separate scoped fixture commit after purehelper stage.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current prepared-request, typed resolution, auxiliary completion and branch-memory assertions are implemented. The complete rewind integration passes after rebasing in a 46-test follow-up selection; all 73 summary tests passed in the initial rebased selection. Exact span, preamble, transcript, durable-memory restart and no-leak assertions remain. Pre-rebase combined 78-test run passed with an existing aggregate descriptor warning, not resource closure. TASK-31757 records the exposed runtime defect. No new ADR.
<!-- SECTION:NOTES:END -->
