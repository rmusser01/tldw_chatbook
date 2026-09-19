---
id: TASK-32836
title: Keep MCP inspector readiness guidance with its server context
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 05:08'
updated_date: '2026-09-19 05:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep server readiness explanations and actions from appearing above unrelated tool, permission, audit or finding details.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The readiness badge, explanation and actions are hidden together while a detail view is present.
- [x] #2 Background readiness updates stay hidden until the last detail clears, then reveal current server guidance and actions.
- [x] #3 Targeted tests and private dark/light native captures verify detail transitions without changing server action routing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce guidance/action leakage for all four detail types and during readiness refresh/partial clear.
2. Extend the existing single visibility owner to the complete readiness block; preserve content updates and action routing.
3. Verify targeted inspector/workbench regressions, design governance, independent review and private dark/light native transitions. Save a bounded draft PR against dev.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine repair of the existing inspector detail-visibility contract from TASK-2270; no new persistent state, runtime boundary or application structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The inspector now hides the readiness badge, explanation and server actions together while any tool, permission, Audit or finding detail is shown. The existing single visibility owner still controls restoration after the last detail clears; background updates keep hidden content current. No action routing, permissions, tokens, CSS or runtime boundaries changed. Existing ADR-150/161 and TASK-2270 apply; no new ADR required.

62 targeted cases pass: nine new regressions, 17 inspector cases, ten workbench cases and 26 governance cases. Seven preflight guards pass; no introduced Ruff diagnostics; new files/changed helper formatted. Independent review found no blocker. All 16 private native captures were inspected across dark/light 80x24 and 170x48; clean shutdown, released lock, ten healthy databases, unchanged defaults and matching final source hashes pass. Native scope uses real catalog plus one synthetic audit metadata record, no tool execution. Initial red evidence and unrelated pytest cleanup warnings remain in the QA receipt.

Updated inspector visibility code, additive tests and review ledgers. QA: Docs/superpowers/qa/2026-09-18-mcp-inspector-guidance/README.md. Separate Audit selection/filter PRs remain independent. Next is Audit-to-tool/permission navigation; current-head CI and final visual approval still gate merge, and the wider component review stays open.
<!-- SECTION:NOTES:END -->
