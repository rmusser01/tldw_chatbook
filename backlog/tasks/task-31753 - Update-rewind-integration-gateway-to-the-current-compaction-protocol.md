---
id: TASK-31753
title: Update rewind integration gateway to the current compaction protocol
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:11'
updated_date: '2026-09-05 22:15'
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
Current prepared-request, typed provider-resolution and auxiliary-completion fixture protocols are implemented. Real savings require a compressible prefix; assertions now follow current branch-memory storage instead of retired legacy summary fields. This exposed TASK-31757, the live-parent snapshot defect. The full integration file passes in the 203-test rewind/settings selection; final full summary-fence verification is pending. ADR not required: test-only alignment.
<!-- SECTION:NOTES:END -->
