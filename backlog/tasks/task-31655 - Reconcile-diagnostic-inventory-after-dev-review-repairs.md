---
id: TASK-31655
title: Reconcile diagnostic inventory after dev review repairs
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:12'
updated_date: '2026-09-05 18:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the diagnostic inventory and summarization boundary pins accurate after reviewed dev changes and behavior-preserving controller extractions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every changed diagnostic owner is explained by reviewed source deltas, with no new private payload or sink authority.
- [x] #2 Checked and generated diagnostic inventories agree and summarization boundary mutation tests pass.
- [x] #3 Scoped static and diagnostic checks pass without widening exemptions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare checked inventory against rebuilt current sources and trace each changed statement against latestdev.
2. Wait for Console/Library extraction boundaries to settle, then reconcile moved diagnostic ownership and regenerate the reviewed inventory.
3. Refresh only normalized manifest-boundary pins after auditing nonsummarization deltas; retain all summarization site and schema assertions.
4. Run inventory checks, full summarization privacy tests including mutants, and scoped static checks.
ADR required: no
ADR path: N/A
Reason: preserve established diagnostic privacy checks and reviewed ownership; no new logging authority or schema.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audited all9new controller diagnostic owners and bothscreen reductions against rebased dev93388ba69b. Aggregate moved statements remain222before/222after (Console118to79; Library104to100); statement content/arguments match after whitespace normalization except submission watchdog self._watchdog_seconds, whose wiring resolves the same screen constant. No added sink or private-payload authority. Regenerated574to583owner inventory and only the two normalized summarization boundary hashes; owned summmarization sites/schema unchanged. Full diagnostic inventory+summarization privacy including mutation tests327passed410.60s, /private/tmp/tldw-review-diagnostic-final-20260905.xml. Full test-file Ruff, JSONparse, diff whitespace and self-review pass. No ADR required for reviewed ownership accounting.
<!-- SECTION:NOTES:END -->
