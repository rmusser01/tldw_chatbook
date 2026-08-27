---
id: TASK-22863
title: Coordinate durable Watchlists check and briefing operations
status: To Do
assignee: []
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:17'
labels:
  - watchlists
  - console
  - async
  - briefings
dependencies:
  - TASK-22859
  - TASK-22860
  - TASK-22861
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-console-watchlists-commands-and-operations.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Accept source checks and briefing generation quickly, execute them under application-owned supervision, and expose exact durable receipt status without tying work to a screen lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Domain services expose public accept/execute APIs that commit or resolve the durable receipt before returning; tools do not call private briefing helpers or drive Textual widgets.
- [ ] #2 `watchlists_check_sources` accepts at most 50 canonical source IDs or one collection of at most 50 members, returns existing active receipts on duplicates, and executes no more than four checks concurrently.
- [ ] #3 `watchlists_generate_briefing` acknowledges one durable generating receipt, returns the existing active receipt on a duplicate, and continues generation independently of Console navigation/tool timeout.
- [ ] #4 Every accepted response contains the exact poll tool/arguments, suggested bounded backoff, canonical receipt/entity IDs, and terminal-state set.
- [ ] #5 The app-owned coordinator schedules work on the running application loop, retains strong task references, consumes terminal exceptions, and never uses `asyncio.run()` against app-owned services.
- [ ] #6 Shutdown stops acceptance, applies bounded cancellation/reconciliation, and leaves every accepted receipt in a truthful terminal or restart-recoverable state.
- [ ] #7 Console receipt cards refresh from durable status and expose Runs/Artifacts inspection plus Retry/Cancel only when the domain can honor those actions.
- [ ] #8 Concurrency, navigation survival, timeout, shutdown, restart reconciliation, scrubbed failure, provider dispatch, and Textual card tests pass.
<!-- AC:END -->
