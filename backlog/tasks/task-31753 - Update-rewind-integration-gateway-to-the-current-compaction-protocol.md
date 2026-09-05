---
id: TASK-31753
title: Update rewind integration gateway to the current compaction protocol
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:11'
updated_date: '2026-09-05 23:09'
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
- [x] #4 Owning-turn summary and RAG context regression uses real durable storage and the current auxiliary gateway while preserving captured-provider and RAG assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; test-only current provider protocol alignment. 1. Read task and compare _SequencedCapturingGateway with canonical SummaryGateway in test_console_rewind_summarize.py. 2. Preserve real DB/store/controller lifecycle and sequenced payload capture while adding canonical prepare_chat_request and complete_auxiliary behavior. 3. Run complete rewind integration and relevant summary files; retain exact accepted/span/persist/resume assertions or report genuine product issues. 4. Root reviews a separate scoped fixture commit after purehelper stage.

Continuation: reproduce the owning-turn context save-first refusal; reuse the
existing SummaryGateway preparation/auxiliary implementation with a typed,
captured-selection resolver and real file-backed persistence. Persist compressible
historical rows and clean up controller/worker-owned connections. Run complete
context, summary and rewind integration files. ADR required: no; the same existing
compaction and provider contracts, with test-only fixture repairs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current prepared-request, typed resolution, auxiliary completion and branch-memory assertions are implemented. The complete rewind integration passes after rebasing in a 46-test follow-up selection; all 73 summary tests passed in the initial rebased selection. Exact span, preamble, transcript, durable-memory restart and no-leak assertions remain. Pre-rebase combined 78-test run passed with an existing aggregate descriptor warning, not resource closure. TASK-31757 records the exposed runtime defect. No new ADR.

Continuation: owning-turn summary/RAG test now uses a real file-backed ChaChaNotes DB and workspace registry, completed durable turns and existing SummaryGateway preparation/auxiliary behavior. Exact captured-provider and RAG configuration assertions remain, plus auxiliary captured-model. Refusals reproduced sequentially: missing durable service, missing workspace registry, then incomplete selection anchor; no production gate bypassed. Controller and DB/registry resources are explicitly released. Final seven complete Chat files including summary/context/rewind integration:205passed58.30s, with the preexisting209 aggregate descriptor warning. Scoped independent review clear. This task remains In Progress pending the broader closeout already recorded.
<!-- SECTION:NOTES:END -->
