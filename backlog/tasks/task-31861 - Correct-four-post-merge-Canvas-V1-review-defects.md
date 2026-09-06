---
id: TASK-31861
title: Correct four post-merge Canvas V1 review defects
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 13:36'
updated_date: '2026-09-06 15:17'
labels:
  - canvas
  - review
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct four independently reproduced defects in the merged Canvas V1 implementation before adding Mermaid support. Restore the accepted execution-profile, historical-selection, archive identity, and virtual form-state contracts without broadening runtime authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unknown or unsupported imported runtime profiles remain source-inspectable and exportable but cannot compile or execute in native or served Canvas.
- [x] #2 The next assistant Canvas read or update honors the live session historical selection and preserves branch reachability and stale-authority fences.
- [x] #3 Same-identity Canvas archive restore rejects any divergent canonical conversation or message graph atomically while exact restores remain idempotent.
- [x] #4 Untouched textarea and select defaults have matching rendered and virtual values in real Chromium; edits and reconstruction preserve supported form behavior.
- [x] #5 All four findings have permanent regressions with observed RED then GREEN, focused preservation tests and independent scoped review, without weakened sandbox or performance limits.
- [x] #6 Qodo feedback on PR #2459 is verified and addressed: bounded typed archive comparison including citations, structured unsupported-profile refusal, shared profile constant, and documented public contracts; targeted regressions and scoped review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: direct corrections to accepted behavior; no new authority, schema, or runtime capability.

### Task 1: Correct the four verified V1 findings

Scope: all acceptance criteria of TASK-31861. Historical pre-rebase base: 017cf826c, repair branch codex/canvas-v1-review-fixes in .worktrees/canvas-v1. Spec: Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md (runtime profiles, sections 8.2, 9 and 14); existing ADR-121 applies.

Global Constraints: strict zero egress; no native global, parent DOM, network, filesystem, cookies or new runtime powers. No profile guessing/conversion. Preserve source inspection/export, exact owner/incarnation/branch/expected-parent fences, atomic archives, existing quotas and scheduling boundaries. No V2, unrelated refactoring, schema or dependency changes. Preserve all existing worktrees, refs and ignored evidence. Root alone executes Git, tests, interpreters, browser and formatters. Workers use static reads and apply_patch only, no subagents. Targeted tests only, no full suite or user database.

1. Add permanent regression tests first, report exact selectors to root, and wait for root-observed RED before implementation. Repeat per finding if practical. Use existing real SQLite and real Chromium harnesses; mocks only at external agent seams.
2. P1: native_authority.resolve_render_plan and Web_Server/serve.py compile source while discarding the stored runtime_profile. Carry and validate profile at native and private served read boundaries before compilation; imported canvas-v99 must stay inert and readable/exportable. Inspect gateway read representation and preserve stale-plan fences.
3. P2: console_chat_controller registers CanvasScope with selected IDs None. Hand the authoritative live historical selection to the next assistant run, preserving reachability and incarnation fences. Reproduce three real submit_draft turns (create r1, update r2, pin r1 through NativeConsoleCanvasAuthority, then actual provider read/update expected r1); expect first HTML and a staged branch, not second HTML/conflict.
4. P2: chatbook_importer same-identity preflight compares only Canvas rows and then skips the whole conversation. Compare canonical conversation and message graph plus Canvas under the existing transaction; reject any divergence atomically while exact restore is idempotent. Seed/export the real Canvas graph fixture, update an originating message with db.update_message and preserve_descendants=True, then restore: it must fail without mutation. Preserve import-as-new behavior.
5. P2: canvas_runtime_worker.js installNode initializes values from attributes only. Untouched textarea text and select selected-option defaults render correctly but canvas.submit reports empty values. Correct the supported virtual form-state initialization and maintenance; real Chromium must submit hello/b for textarea hello and second selected option b, preserve edits/reconstruction, and pass zero-egress observations. Use virtual DOM only and retain bounded operations.
6. Root runs focused RED/GREEN plus preservation tests, package integrity regeneration/checks as required, lint and formatting checks limited to touched code. Worker self-reviews and records evidence. Independent scoped review verifies all four corrections and new breakage; root performs final task hygiene and reports any remaining limitations.

Report concerns before expanding architectural authority. Do not commit/push/merge from a worker.

### Task 2: Address Qodo review of PR #2459

ADR required: no new ADR
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: preserve the existing bounded, atomic archive and fail-closed runtime contracts.

Base: 78f8801b09d5e92f43b5fa94b00aa222cb00b61a (rebased onto dev 43d7b93a8). All Task 1 Global Constraints apply, including root-only execution and no worker subagents. Task 1 is complete; do not repeat that review.

1. Verify Qodo comments 3944264229, 3944264231, 3944264232, 3944264236, 3944264239 and 3944264244 against code. Read the relevant existing models and citation exporter before choosing minimal corrections.
2. Add permanent behavioral regressions and provide exact selectors to root; wait for observed RED before product edits. Same-identity graph reads must enforce the existing message ceiling at SQL fetch time (limit plus one), malformed incoming comparison shapes must reject through bounded typed validation before downstream dictionary operations, and citation-bearing archive fields must participate in canonical equality. Exact restores with citations stay idempotent and all divergences reject atomically. Preserve attachment comparison and graph validation.
3. Translate expected unsupported-profile failures at the plan HTTP route to its bounded plan_unavailable response in both native and served modes; do not swallow unexpected arbitrary faults. Reuse a shared supported runtime profile constant without weakening rejection or moving heavy imports onto startup.
4. Document Args/Returns/applicable Raises contracts of bind_selection_resolver, capture_selected_scope, capture_selection_owner, and canvas_active_path_message_ids, without behavior changes.
5. Root verifies focused RED/GREEN and preservation suites, preflight, changed-code lint/format and independent scoped rereview; records evidence, replies to Qodo, pushes with exact lease after rebase, waits for current-head review and required checks, and merges only when clear. Preserve all refs and worktrees.

Citation implementation clarification: use an optional read-only projection through the existing citation conversation service/migration and inject a repository that reads existing keys without provisioning. Preserve the current bounded 100 visible-context-row export projection. If legacy fingerprint reconciliation would be required, reject same-identity comparison conservatively without key, sidecar, or journal writes; ordinary exporter behavior remains unchanged. Verify those non-mutation guarantees with regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected all four post-merge V1 findings under existing ADR-121 (no new ADR, schema, dependency, authority, quotas, or scheduling changes): native/served stored-profile gates; live historical selection with stale-owner and atomic-promotion fences; canonical conversation/message/Canvas archive equality under writer transactions; and virtual textarea/select defaults, dirty state and reflected disabled controls. Native screen and dispatch now share the existing durable-only SYSTEM-row projection rule via ConsoleChatStore. Permanent regressions were observed RED then GREEN. Independent scoped review addressed the four findings; its two new regressions were reproduced, fixed, and independently re-reviewed with spec compliance pass and code quality approval, no remaining findings. Local repair commits:40cd4e653 and53ae7d5bd. Consolidated targeted Python verification:413 passed. Final Chromium security/native workflows:34 passed,2 optional browser-engine skips (Firefox/WebKit unavailable). Asset verification:21 passed including cached reproducibility; final formatted control regressions:6 passed. Counts overlap and are not summed. Existing RequestsDependencyWarning remains. Static comparison found no new lint diagnostic signatures (622 inherited/current across touched Python files); changed ranges formatted, diff-check and runtime integrity passed. Updated runtime compatibility docs; preserved all worktrees, refs and ignored reports/probes. No full repository sweep, user database operations, remote publication, PR or merge. Mermaid V2 remains next after integration.

PR #2459 Qodo follow-up: bounded the same-identity SQL query at the existing
ceiling plus one; added a strict typed envelope before graph semantics; included
canonical exported citations without provisioning keys or writing journals;
deferred writable citation setup beyond exact-restore skip; shared the runtime
profile across native, served and archive gates; documented public contracts;
and mapped only unsupported profiles to plan_unavailable. The existing gateway
already bounded generic faults as 503, so Qodo's raw-500 claim was corrected.
Read-only citation reconciliation refusal follows existing ADR-121; it may
require ordinary local reconciliation before exact restore. No new ADR, schema,
dependencies, quotas, runtime powers, or V2 work.

Independent scoped review passed all six findings with no new actionable issues.
532 targeted Python tests passed; post-format and latest-dev rebase focused
verification passed 163 cases each. Chromium security/native checks:34 passed,
2 optional-engine skips; existing Requests warning remains. Ruff baseline/current
243/243, no new diagnostic signatures; 41 changed ranges format clean, diff-check
and all six preflight guards pass. Rebased cleanly onto dev a9e13f4e3 with all
six patches unchanged; correction commit713ba8cc6. Rechecked404refs/68worktrees
for TASK-31861 ownership; preserved all worktrees, refs and ignored evidence.
Remote current-head Qodo/CI and merge remain integration gates, not claimed here.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from TASK-31825 to TASK-31861 on 2026-09-06 before PR publication.
The refreshed remote branch `origin/codex/dev-test-review-20260904` already
claims TASK-31825 for synthesized leading-system trace provenance. A sweep of
396 local/remote refs and 68 worktrees found 31860 as the highest task ID.
The original review reports and recovery refs retain their historical task ID
and commit hashes intentionally. Rebase onto dev `5894f4755` preserved the
repair patches: `40cd4e653` became `2c575574c`, and `53ae7d5bd` became
`c306d3be7`. No Canvas behavior changed during renumbering or rebase.

## PR preparation verification

Rebased onto dev `5894f4755`; `git range-diff` confirms all four replayed
commits preserve their original patches. Post-rebase verification passed:
413 targeted Python tests, eight Chromium native-workflow/form cases, and all
six repository preflight guards. The existing Requests dependency warning is
unchanged. Earlier statements about no publication describe the implementation
checkpoint; the user subsequently authorized pushing and creating a PR against
dev. Merge and Mermaid V2 implementation are not part of this publication step.
