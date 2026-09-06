---
id: TASK-31904
title: Give provider-grammar adapter fixture its required assistant owner
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:49'
updated_date: '2026-09-05 19:54'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the empty-tool-catalog provider-grammar regression focused on transport serialization now that every streaming adapter owns an assistant thinking capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The adapter fixture reaches the serializer sentinel with native transport enabled and an empty tool catalog while retaining the real thinking-owner validation.
- [x] #2 The complete provider-grammar test file and scoped static checks pass without production changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the unchanged failure at ThinkingCapture construction before the serializer sentinel.
2. Supply a stable opaque fixture assistant owner ID; preserve the ExpectedStop serializer assertion and native_tools/tools inputs.
3. Run the complete provider-grammar file and scoped static checks; independently review before committing.
ADR required: no
ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md
Reason: Test-only fixture admission correction under the existing assistant-generation ownership contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated only the native-provider grammar adapter fixture with its required stable assistant owner. The intercepted serializer still proves native grammar without dispatching. ADR check: no new ADR; preserves ADR-090 ownership. Before: both whole affected files had 62 passes and the two diagnosed failures. After both independent fixture corrections: 64 passed in 18.72s (/private/tmp/tldw-31741-31742-final.xml). Whole-file Ruff lint/format and whitespace checks passed; parent reviewed the exact fixture diff without findings.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31741 was renumbered to TASK-31904 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
