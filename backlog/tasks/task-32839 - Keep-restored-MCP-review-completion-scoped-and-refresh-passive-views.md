---
id: TASK-32839
title: Keep restored MCP review completion scoped and refresh passive views
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 06:18'
updated_date: '2026-09-20 05:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make restored-root review finish without overwriting later navigation or leaving historical catalog views visible after fresh defaults are accepted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Accepted review retains its native write to completion while later navigation or service replacement prevents stale UI updates and notifications.
- [x] #2 Successful current-view review refreshes passive server, tool, permission and Audit displays without connecting or granting tools.
- [x] #3 Confirmation, cancellation and changed-root rejection remain intact, with targeted real-owner tests and native dark/light evidence.
- [x] #4 Selecting another Permissions row or policy profile while an accepted review is pending preserves the newer selection and suppresses the superseded completion receipt.
- [x] #5 The native evidence runner rejects invalid arguments and unsafe profiles before app startup or output writes, using the existing shared validation boundary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce late approved-write completion after navigation and stale passive catalog displays using the actual isolated restore/review owner.
2. Retain the existing review token through accepted work; gate UI completion by its owning view and refresh passive local canvases after a current successful review. Preserve owner approval, file checks and retained native cancellation.
3. Run focused new and existing real-owner controls/activation tests, static checks, guards and independent review. Verify private native dark/light confirmation, cancellation and success; save an independent draft PR against dev.
4. Integrate merged PR2749, retain both report histories and reproduce same-mode Permissions row/profile changes during a held native write, including round trips.
5. Invalidate the existing recovery receipt at the accepted selection boundary, preserve native completion, and rerun the affected real-owner, selection/navigation and native visual checks before saving updated PR2727.
6. Address Qodo runner findings by reusing shared native CLI validation, documenting the entry point, extending existing boundary tests and requalifying a fresh native run.
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine caller-lifetime and passive UI repair under existing restored-authority policy; no owner policy, storage or service contract changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retained restored-review UI ownership through the native writer and all subsequent render steps, suppressing stale receipts after mode/screen round trips, service changes and pending synchronization. Current completion clears historical projections and synchronizes Servers, Tools, Permissions and Audit without reload, connection or grants. Added 18 real-owner regressions; two existing footer tests now use the established private-profile wrapper after their original setup failed before UI creation.

45 distinct targeted cases, seven preflight guards, unchanged static-analysis baseline and independent review pass. Sixteen native dark/light captures at 120x40 and 170x48 verify cancellation, confirmation, Ask/local defaults and retained historical bytes. Native exit0, released lock, ten healthy private databases, zero conversations/messages and unchanged user defaults are recorded; one unrelated restored Evals owner-enrollment diagnostic remains explicitly outside qualification.

Evidence and continuation: Docs/superpowers/qa/2026-09-18-mcp-restored-roots/README.md and both design-system/MCP ledgers. No new ADR: existing ADR126 restored authority and ADR150/161 UI governance apply. No new lesson needed; the existing private-profile lifetime guidance covers the fixture issue. Independent draft PR against dev; no merge before current-head CI and final visual approval. Catalog repopulation/status guidance after review and broader Permissions actions remain separate follow-up review scope.

Integrated onto merged PR2749/dev 65d79cc2b7, retaining both report histories with no product conflicts. Four added real-owner keyboard cases reproduced late completion overwriting same-mode Permissions row/profile selection, including round trips. Validated row and changed-profile selection now retire the existing UI receipt before awaiting, while the accepted native write completes unchanged.

Current integration: 85 targeted cases and all seven guards pass; no new static diagnostics, changed ranges/new files formatted, independent final review found no blockers and verified current source/export hashes. Sixteen fresh native captures qualify cancellation, confirmation and Ask/local publication at both theme/size combinations. Exit0, released lock, ten healthy private databases, zero conversations/messages and three unchanged default files verified. One separately documented Evals enrollment diagnostic remains. Updated the native evidence runner to use the application image warm-up after an early private-library import failure; preserved historical runner/evidence and recorded the lesson.

Current integration notes/gallery: Docs/superpowers/qa/2026-09-18-mcp-restored-roots/CURRENT-DEV-REVIEW.md. Existing ADR126/150/161 apply; no new ADR. Implementation complete; PR2727 still requires current-head CI, accumulated review and its own final visual approval before merge. Long-path presentation, status/catalog repopulation and connected runtime remain separate review scope.

Qodo review of bb824eba9a found no product bugs and two runner rule findings. Plan: reuse native_runner_args.py, document the entry point, extend the existing runner-input regressions, and verify a fresh native lifecycle. This reuses existing QA validation and changes no app behavior or architectural boundary.

Addressed both Qodo runner findings using the existing shared native_runner_args.py validation boundary and an explicit main() CLI/exit contract. Seven new runner-entry cases failed before the fix; all70 shared boundary checks pass. Product source/tests remain unchanged from the85-case qualified head bb824eba9a (155 distinct passes total). A fresh corrected-runner native journey passed all16captures and full lifecycle, with13 native source hashes and runner hash verified. Independent delta review found no blockers. Evidence manifest/gallery refreshed; prior integrated captures remain in bb824eba9a. Final-head CI/review and owner visual approval remain before merge.
<!-- SECTION:NOTES:END -->
