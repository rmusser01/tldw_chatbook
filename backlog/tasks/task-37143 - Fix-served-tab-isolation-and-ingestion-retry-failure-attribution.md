---
id: TASK-37143
title: Fix served tab isolation and ingestion retry failure attribution
status: Done
assignee:
  - '@codex'
created_date: '2026-10-27 23:04'
updated_date: '2026-10-28 05:21'
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

CI dependency follow-up: current-head PR Fast Lane reports 17 served-tab setup failures because aiohttp_jinja2 is absent from its minimal install. Pin the existing web templating helper in the common lane dependency contract, observe RED, add that one helper to the shared install commands, and validate the web regressions with the actual lane install in a scratch environment plus focused CI contract tests. Preserve all targets, budgets and timeouts; no new production dependency or ADR (routine CI provisioning correction). Obtain a fresh independent review, commit/push with lease while auto-merge is disabled, and wait for new-head CI before re-arming.

UI CI follow-up: shard 3 fails a readiness-helper regression before its stale-proof action because it assumes the already-selected Console is still mount-blocked. Reproduce that valid already-ready order deterministically using the real selection helper, then correct only this test precondition while retaining its own-sync deferral/render and enabled-send assertions. Verify the adjacent lock-contention case and a mutation that bypasses the stale-proof retry still fails. No production behavior, shard, budget or timeout change; routine test-harness correction, no ADR required. Obtain independent review and rerun focused cases in the minimal lane environment before publishing.

The affected-file run reproduced the same already-ready precondition failure in the adjacent lock-contention case. Configure its missing-key connection through the existing persisted provider fixture before mount, preserving the blocked-to-enabled assertion. Leave three separate Windows trace-capture refusal cases unchanged; those passed Linux CI and precede or are isolated from this test edit.
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

CI dependency follow-up: current-head PR Fast Lane failed 17 served-tab setups because the base-only lane lacked aiohttp_jinja2 (3217 other tests passed). Added only the existing web runtime helper to all seven matching shared lane install commands and both dependency pins; their updated pin first failed three cases. A fresh environment from the updated exact package list, with pytest 9.2.0, passed all 184 focused web/STT/retry/startup and CI contract checks (four existing aiohttp warnings). Scoped Ruff, changed-range formatting and whitespace pass. No targets, budgets, timeouts, shard assignments or production requirements changed; no new ADR required. Rebased onto dev 323cf5d9e7 with both feature commits range-diff equivalent. Fresh independent review found no issues. Updated plan and testing-evidence lesson with the incident. New-head CI, including all UI shards under the shared install, must pass before merge.

UI CI follow-up: shard3 passed175 cases and failed the stale-proof regression at an unstable mount-block precondition. Explicit real selection reproduced the exact failure before fault injection. The final test starts ready, injects expiry, and retains its own deferred-then-rendered sync and enabled-Send assertions; targeted pytest9 lane case passed. In-memory early-return mutation failed at syncs=[False], establishing non-vacuity without helper source changes. Fresh independent review found no issues. Broader Windows file run:13passed4failed, modified case passed; untouched lock mount-state assumption and3 trace-capture refusals passed Linux shard and remain local verification limitations. The canonical missing-key experiment did not control the always-ready gateway and was fully removed; final scope is only the stale-proof case. No production, helper, timeout, budget or shard changes; no ADR required. Ruff, changed-range format and whitespace pass. Plan and evidence lesson updated; new-head Linux CI must pass before merge.
<!-- SECTION:NOTES:END -->
