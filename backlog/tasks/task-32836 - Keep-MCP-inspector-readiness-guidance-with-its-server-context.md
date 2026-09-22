---
id: TASK-32836
title: Keep MCP inspector readiness guidance with its server context
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 05:08'
updated_date: '2026-09-22 16:22'
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
Resume saved PR2722 only after approved PR2721 is confirmed merged. Use the actual merged dev tree and a fresh codex/ follow-up branch; preserve unrelated work.

ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: extend the existing inspector readiness visibility owner to its explanation and actions; no new state, authority, storage or application structure.

1. Retain PR2721 merge/check/review receipts and mark TASK-32835 Done after actual merge verification.
2. Apply saved PR2722 commit 4b77c4b33c21affd1f258b7c084490cd65e566c3 on the fresh branch. Retain current and historical review-document checkpoints; independently inspect any product conflict. Reopen TASK-32836 before follow-up edits.
3. Verify all four detail types, background readiness refresh, partial clearing and server-action routing. Run targeted inspector/workbench/guidance, shared runner admission and appropriate design/artifact checks; record upstream debt without broadening the product fix.
4. Bring the saved native runner onto the established shared CLI/private-profile bootstrap, pin checkout before imports, use warm_up_image_protocol, block network, log privately and record module/source provenance. Require positive painted geometry for controls. Add shared CLI admission coverage.
5. Run real native Servers -> Audit -> Tools -> Servers transitions at 80x24 and 170x48 in dark/light. Inspect captures and confirm guidance hides/restores without tool execution. Verify normal shutdown, released lock, healthy private databases, unchanged defaults/sentinels and source/runner hashes.
6. Publish receipts through Docs/superpowers/qa/export_receipts.py with stable host-path placeholders and original/export hashes. Label saved qualification historical. Obtain independent review and update existing PR2722 against dev, retaining TASK-32836 In Progress pending its own fresh visual approval, current-head CI/Qodo and final dev review. Do not include later Audit navigation or other screens.

7. Post-approval closeout: rebase onto current dev, inspect current-head CI and accumulated Qodo. Reproduce Test Tool mouse failures with event/geometry diagnostics; settle scheduled animations before test Run clicks and assert the actual hit without changing product code or execution expectations. Rerun affected Workbench tests, guidance and incoming startup/admission tests, refresh private native captures/lifecycle and compare to approved appearance. Existing ADR150/161 applies; no new ADR for test synchronization. Preserve failed logs; merge only after final-head CI/review/current-dev verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The inspector now hides the readiness badge, explanation and server actions together while any tool, permission, Audit or finding detail is shown. The existing single visibility owner still controls restoration after the last detail clears; background updates keep hidden content current. No action routing, permissions, tokens, CSS or runtime boundaries changed. Existing ADR-150/161 and TASK-2270 apply; no new ADR required.

62 targeted cases pass: nine new regressions, 17 inspector cases, ten workbench cases and 26 governance cases. Seven preflight guards pass; no introduced Ruff diagnostics; new files/changed helper formatted. Independent review found no blocker. All 16 private native captures were inspected across dark/light 80x24 and 170x48; clean shutdown, released lock, ten healthy databases, unchanged defaults and matching final source hashes pass. Native scope uses real catalog plus one synthetic audit metadata record, no tool execution. Initial red evidence and unrelated pytest cleanup warnings remain in the QA receipt.

Updated inspector visibility code, additive tests and review ledgers. QA: Docs/superpowers/qa/2026-09-18-mcp-inspector-guidance/README.md. Separate Audit selection/filter PRs remain independent. Next is Audit-to-tool/permission navigation; current-head CI and final visual approval still gate merge, and the wider component review stays open.

Current-dev integration starts at actual PR2721 merge3722a857; both conflicting documentation histories preserved and saved qualification labeled historical. Product applied cleanly. Updated only QA compatibility: captured Cancel operation and current inspector.clear_mode_view helper, shared runner admission/bootstrap/provenance/network/paint checks. 201 distinct targeted passes; one inherited dimension-ratchet failure with all12 declarations identical to base. Nine artifact guards pass after the declared-input network retry; no introduced Ruff diagnostics. Sixteen fresh native captures across dark/light compact/wide inspected; all14 sources plus runner match; private lifecycle clean and no network/tools executed. Independent review has no remaining findings. Current evidence: Docs/superpowers/qa/2026-09-18-mcp-inspector-guidance/current-dev/README.md. Existing ADR150/161, no new ADR. Remains In Progress pending fresh visual approval and current-head CI/Qodo/dev closeout.

Owner approved the current gallery and continuation. Conflict-free rebase onto dev5cdc9ddda00a4112ec06cf8b81fd23bd31d9edb7; inspector source identical to approved b423. Five CI Test Tool failures traced to synthetic mouse clicks during focus scrolling; test helper now settles scheduled animations and asserts real mouse hit. Redaction test requires a Failed outcome. 359 Workbench/guidance plus231 incoming startup/private-path/runner admission cases pass; delayed five-case probe and two strengthened redaction cases pass. Nine artifact guards pass, no introduced Ruff diagnostics, changed ranges formatted. Four fresh native cells/16captures and clean lifecycle pass; all16 inspector regions match approval, two central scrollbar fractional glyph differences disclosed. QA: Docs/superpowers/qa/2026-09-18-mcp-inspector-guidance/current-dev/approved-closeout/README.md. Existing ADR150/161; remains In Progress pending final-head CI/Qodo/dev and actual merge.

Addressed all six Qodo rule findings on86efdced97: document guidance/capture/probe helpers, group/format QA imports, replace the diagnostic environment delay with its recorded35ms constant. Explicit check=False preserves lifecycle process-check default. All9 guidance and6 diagnostic cases pass, four files pass Ruff and capture comparison unchanged. AST receipt confirms no product/test logic changes. Publication manifests retain raw and original-export hashes with explicit review transformations and new hashes; final-head review/CI still required.
<!-- SECTION:NOTES:END -->
