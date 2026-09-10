---
id: TASK-32194
title: Fix capture for planning-prefixed agent tool calls
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 19:51'
updated_date: '2026-09-10 04:25'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore captured local-model tool continuations when the assistant explains a plan before calling a tool.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capture On admits valid planning-prefixed tool calls and preserves exact provider message content.
- [x] #2 Plain assistant text and malformed look-alike fences retain correct classifications; native calls remain supported.
- [x] #3 Targeted request and real controller trace regressions pass without weakening provenance validation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: align existing classifiers under the accepted provenance contract.
1. Promote the isolated failing reproduction to a repository regression.
2. Reuse the canonical fenced-tool parser in the agent artifact classifier.
3. Verify captured tool continuation and successor sends; investigate local-history failure shapes independently.
4. Run targeted checks, review and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Capture classifies tool fences with the same shared parser as request partitioning. Planning text and malformed look-alikes preserve exact payloads; native calls are unchanged. Updated bridge and real-controller discovery tests.

RED: six expected classifier/controller failures. GREEN: 21 focused checks; independent review ran 22 including native control. Real LAN llama.cpp captured greeting, calculator 17*19 and following answer passed using disposable databases. Broader trace run: 456 passed, two pre-existing display/redaction failures reproduced with original bridge/runtime modules. Scoped Ruff checks and formatting pass; no new ADR (existing ADR097). Old failed-empty history has its own TASK-32197. Lessons recorded in lessons-live-verification.md.
Maintainer review: rebased on dev 26cdfb42ad and renumbered to avoid merged task IDs. Independent parser/runtime review passed 20 focused cases; canonical parser usage also matches console_prepared_request. No new lint findings.
<!-- SECTION:NOTES:END -->
