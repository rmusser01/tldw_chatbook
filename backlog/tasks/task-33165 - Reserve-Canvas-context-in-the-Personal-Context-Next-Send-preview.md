---
id: TASK-33165
title: Reserve Canvas context in the Personal Context Next Send preview
status: Done
assignee:
  - '@codex'
created_date: '2026-09-28 05:52'
updated_date: '2026-09-29 18:18'
labels:
  - memory
  - canvas
  - context-budget
dependencies: []
references:
  - tldw_chatbook/Chat/console_chat_controller.py
  - tldw_chatbook/Chat/console_agent_bridge.py
documentation:
  - >-
    backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
  - backlog/decisions/202-dependency-aware-personal-context-forgetting.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Canvas tools are enabled and the model input budget is tight, the Personal Context Inspector can report records as selected even though the dispatched request omits them. Make the preview reflect the complete request budget while preserving the disposable nature of inspection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A native production preview regression reproduces the Canvas-enabled budget mismatch before the fix.
- [x] #2 With identical profile, model, message and tool inputs, the Inspector and dispatched request agree on selected records and their available input budget, including Canvas schemas and runtime guidance.
- [x] #3 Preview does not register an executable Canvas run, create staging records, grant mutation authority, or change the selected Canvas.
- [x] #4 Disabled, unavailable and stale Canvas states preserve their existing fail-closed behavior, with native Python >=3.12 targeted verification and a recorded ADR check.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md; backlog/decisions/202-dependency-aware-personal-context-forgetting.md
Reason: bounded correction of disposable first-request budgeting under existing Canvas ownership and Personal Context publication boundaries; document the pure-schema preview contract in ADR-121 before implementation.
1. Reproduce missing Canvas reservation through the real controller/bridge preview and real Personal Context selection, with synthetic private state and native Python 3.12.
2. Reuse the Canvas schema loader and shared disclosure planner for a data-only preview catalog. Preserve live registry authentication, schema order, persona narrowing, native/fenced protocol and progressive discovery; do not register a preview run or issue Canvas authority.
3. Reuse capture_interactive_owner/validate_interactive_owner and exact session identity to fence late publication against close, disable, durable/temporary replacement, promotion and branch changes. Carry the guard through project-preview assembly and Inspector async validation; release diagnostic sidecars only at final snapshot publication. Keep schema loading incremental inside the shared guarded probe, including denied and unreadable schemas.
4. Verify production preview versus dispatched first-request budgets/records and absent run/staging/selection effects, plus disabled/unavailable/stale controls. Run only affected native tests and scoped static checks.
5. Obtain one fresh final review, update receipts/task notes and the existing PR against dev. Record current-dev qualification debt and any newly advanced base separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected disposable Personal Context budgeting using the existing Canvas owner profile, ordered catalog metadata and a lazy read-only schema loader inside the shared guarded planner. Live Canvas provider authentication and execution authority remain unchanged. One shared currentness callback fences exact session identity, Canvas enablement/controller/interactive owner and branch through final snapshot publication and later Inspector validation; diagnostic sidecars publish only after that fence.

Native Python 3.12.11 reproduced the original 123565 versus 121793 token mismatch on unchanged head 295ee2a02a. Seven fresh-review regressions failed before correction (unreadable/denied schemas, closure/disable/replacement during project preview, durable replacement and later Inspector validation). Final affected groups passed 93 context/Inspector cases and 367 agent/Canvas cases, followed by 19 shared-probe checks after simplifying the loop. All 41 exact private-profile children passed without skips; parent/child application and profile-library imports were verified against this PR worktree. Three legacy suites now use the existing bootstrap-profile marker; the unchanged control reproduced their config-source setup failure. Targeted lint, regression-file formatting, AST-preserving helper formatting and whitespace pass. ADR-121 and ADR-202 apply; no new policy or activation was selected. Current-dev integration qualification remains separately tracked.
<!-- SECTION:NOTES:END -->
