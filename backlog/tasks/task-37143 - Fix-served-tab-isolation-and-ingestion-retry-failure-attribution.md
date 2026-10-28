---
id: TASK-37143
title: Fix served tab isolation and ingestion retry failure attribution
status: Done
assignee:
  - '@codex'
created_date: '2026-10-27 23:04'
updated_date: '2026-10-28 01:15'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two browser tabs sharing a login must remain independently usable. Retried imports must not lose successful transcripts when a prior failure has no STT attempt metadata, and local Parakeet failures must provide useful content-free attribution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Two tabs sharing a login can use independent terminal children and matching Canvas state; closing or refreshing one does not disrupt the other, and unrelated login sessions cannot access their Canvas routes.
- [x] #2 Retried imports without prior STT failure provenance can complete; valid predecessor attempt provenance is preserved and malformed partial STT lineage is never emitted.
- [x] #3 Parakeet failures identify a bounded failure stage and safe underlying exception category in the parent application log, without exposing native messages, paths, source content, credentials or tracebacks.
- [x] #4 Focused regression and existing related tests pass, with new regressions included in the appropriate required CI lane.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (web terminal ownership); no additional ADR for the retry guard and bounded STT diagnostic extension.
ADR path: backlog/decisions/251-served-terminal-page-isolation.md (amends ADR-121/233); existing ADR-025 governs STT lineage/privacy.
Reason: correct per-page terminal ownership under shared device authentication while retaining Canvas security fences. Retry lineage and diagnostics retain existing persistence/privacy boundaries.

1. Web: write a real same-cookie two-tab WebSocket regression and Canvas ownership controls; observe failure. Render a fresh page ID, scope it beneath the authenticated login, and route terminal/Canvas requests independently without changing login/logout authority. Preserve exact-child cleanup and cross-login denial. Record the ownership decision in an ADR.
2. Retry: reproduce a successful transcript being rejected on a retry whose previous job lacks an STT failure attempt; omit incomplete STT lineage while retaining the Library job retry link, and preserve complete valid lineage.
3. Diagnostics: add typed, finite failure-stage and exception-category attribution from Parakeet runtime through the worker failure to the parent log. Cover decoding, VAD/inference, normalization/provenance and model loading. No native exception text, source/paths, secrets, traceback, or model downloads in tests.
4. Verify focused web/Canvas, STT executor/runtime/provenance, ingestion options/retry and diagnostic tests; run Ruff, format, census and diagnostic inventory checks. Ensure regression files are in required CI targets. Review all edits, finalize task notes and ACs. No full suite, push, PR or merge is included in this follow-up.

CI follow-up: reproduce the UI-ready census breach and assert STT diagnostics stay off the app-import and first-paint paths; defer diagnostic-only imports at executor validation, parse failure construction and parent failure handling; re-run focused import/census and diagnostic regressions, obtain fresh review, then publish and re-arm. ADR required: no; existing ADR-097 governs deferral, with no budget or boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented independent per-page terminal/Canvas scopes under shared login authentication, preserving strict child ownership, login-wide logout/expiry, legacy compatibility and cache-safe script updates. Full Canvas retirement frees finite slots; every awaited snapshot revalidates its child, broker and enabled state to prevent orphaned launches after closure or disable. ADR-251 records the decision.

Retries keep their Library job relationship and emit STT predecessor fields only with an actual failed-attempt snapshot. Both absent and complete predecessor cases now persist successful normalized transcript provenance.

Added typed finite Parakeet failure stage/exception categories through runtime, spawned worker, audio/video parse adapters and the parent log/job detail. Codes, recovery actions, cancellation, lazy VAD evaluation order and provenance schemas remain unchanged. Native messages, paths, source content, tracebacks and arbitrary class names are excluded. Existing ADR-025 governs STT lineage/privacy. Short-file recognize includes upstream-owned decoding/preprocessing, documented as the inference boundary.

Regression failures were observed before fixes: second-tab refusal, missing page ID transport, retry ValueError, missing attribution/parse propagation, Canvas slot exhaustion. Fresh independent review found one Important late-poll teardown race; real held-snapshot tests reproduced stale ready and HTTP500 cases before the post-await revalidation fixed them. No other Critical, Important or Minor findings.

Validation: final post-review combined checks: 86 passed; existing Canvas ownership/control-loss follow-up: 3 passed. Related STT runtime/executor: 118 passed, 1 existing symlink-unavailable skip; submission/retry/provenance/parent diagnostics: 55 passed; required CI workflow contracts: 130 passed. Web/gateway compatibility: 59 passed plus 2 additional Chrome flows (4 Chrome flows overall). Existing deprecation warnings only. Four new regression targets are wired into required PR Fast Lane and its exact invocation pin.

Scoped Ruff, changed-range formatting, whitespace, task uniqueness/readability, ADR numbering and UI census guards pass. Existing E402 startup-fence and E721 strict-type rules remain excluded in their existing production modules. Reviewed the sole log statement change (48 calls unchanged; two finite enum arguments added), curated only that owner inventory digest, and verified no inventory drift.

Architecture docs, plan and the observed capacity/late-response lesson are updated. No full-suite sweep, model download, actual native ONNX inference, commit, push, PR or merge in this follow-up. A fresh user transcription attempt is needed to attribute the original Parakeet failure with the new diagnostics.

Publication follow-up authorized by the user: commit these fixes and regression tests, open a PR against dev, rebase, perform a fresh Superpowers code review, and merge once CI is green. Fresh pre-commit regression verification passed all 86 affected checks. Current origin/dev remains 773cc9729cfbc9c13c14b56724fcb02efd3a54a3, the implementation base; merge queue mode is on. Full local suite remains outside the requested verification scope.

PR #3282 CI follow-up: reproduced the app-import regression, then deferred STT diagnostics at the three startup import edges. Legacy envelopes and parse errors keep diagnostics unloaded; typed failure validation, pickling and parent attribution remain covered. Added an explicit first-paint absence guard without changing any budget or snapshot. Focused verification: 38 passed plus the Windows census at 1032/1031; an isolated origin/dev f9baac8706 archive measured the same 1032 with windows_files resident and diagnostics absent, establishing inherited platform behavior. Eight additional protocol/parse/retry cases passed; independent Superpowers review passed 35 diagnostic/import cases with no findings. Scoped Ruff, changed-range formatting and whitespace pass; diagnostic inventory has no drift (699 owners, 16 sinks). Rebased onto dev with exact original patch equivalence. ADR required: no; existing ADR-097 deferral policy applies. Linux PR CI must confirm the original 1032-to-1031 regression is resolved before merge.
<!-- SECTION:NOTES:END -->
