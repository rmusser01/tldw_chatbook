---
id: TASK-34785
title: Stop Console transcript activity updates after view detachment
status: Done
assignee:
  - '@codex'
created_date: '2026-10-10 20:50'
updated_date: '2026-10-10 21:22'
labels:
  - console
  - bug
  - ci
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/2882'
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/38084008605/job/114306575704
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required UI lane can crash while a pending question/approval journey removes and remounts its Console view. Ensure transcript reconciliation stops touching retired widgets so navigation and pending-decision recovery remain usable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real Console view removal during an awaited activity update produces no MountError or further mounts into retired widgets.
- [x] #2 Mounted activity updates and the existing pending question/approval sibling/remount journey retain their behavior.
- [x] #3 A deterministic regression fails before repair; targeted checks, static analysis, preflight and independent review verify the fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/190-console-incremental-display-projections.md
Reason: repair stale DOM mutation within the existing mounted presentation lifecycle, without changing storage, authority or UI design.

1. Preserve the hosted failure and reproduce actual widget removal during an awaited activity update; compare the unchanged base implementation.
2. Reuse existing attachment/closing checks to stop the shared transcript mutation path after detachment; retain normal projection updates.
3. Verify the deterministic regression and affected activity/pending-projection modules, static checks and preflight; obtain independent review before publication.
4. Required CI wiring: the Assistant-turn test module was not in the UI census; the earlier gated-module description was incorrect. Add only the three new regression node IDs (ten nested detach parameters, one lock-resume case, two composition parameters), raise the literal census ratchet from 175 to 178, verify the existing node/shard checker contracts and preflight, then obtain independent review of the CI-only commit. Retain all 18 uncensused config-profile baseline failures; do not add or alter those unrelated controls. Existing ADR190 applies and no new ADR is required for this existing CI census mechanism.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Transcript mutation now checks actual attachment, closing/pruning state and expected parent ownership before and after nested activity/action/detail/adjunct awaits. Private Assistant/activity synchronization reports completion so retired work does not publish row signatures. The existing refresh lock also serializes Textual recompose; a refresh arriving during first composition queues one transcript-owned on_mount refresh, retaining changed activity after the screen caches its projection key. No style, storage, permission, runtime-owner or public caller contract changes. Existing ADR190 applies; no new ADR.

The exact hosted MountError is reproduced outside the repository on both immutable 732d1de8c29ede5c03d0a0263d4f9cacf29e0321 and base c2d6dcc3016708bec3473450016177b2ff602fb3. Their transcript source is byte-identical (SHA256 0f7e5c84964c5b6a2d683e71ff3e65d5859ff95ddc661b2a52104f66479b4779). Real recompose leaves the fresh turn attached while its activity stack is still awaiting composition. Separate real removal across lock, stale child removal and inter-mount boundaries produces inherited IndexError and stale signature publication. Raw probes: /private/tmp/pr2882-transcript-detach-probe.py; pr2882-transcript-composition-{head,base}.json; pr2882-transcript-detach-{head,base}.json.

Initial regression selection is NON-GREEN13FAIL: eleven genuine detach/mount/signature failures plus two app-context harness failures. Correcting only the composition harness preserves the actual Textual context and yields RED2 exact MountError. Initial standalone recovery_scope_uncertain and the first unchanged-signature probe oracle are retained separately. Final current-source selection PASS15 in3.68s: thirteen new deterministic lifecycle cases plus two existing real live-tool controls. Earlier GREEN15 and tightened lock-oracle GREEN2 are separate receipts, never summed. XML/logs: /private/tmp/pr2882-transcript-lifecycle-red.{xml,log}; pr2882-transcript-composition-red.{xml,log}; pr2882-transcript-lifecycle-final.{xml,log}. Existing cleanup warnings remain retained.

Scoped Ruff baseline27/current27/zeroNEW; fatal diagnostics zero and formatter all three files pass. Broader trace modules, the actual pending-projection journey, generated-artifact preflight and independent immutable review remain pending under root ownership. Task stays In Progress with acceptance criteria unchecked until that qualification completes.

Required CI coverage correction: test_console_assistant_turn.py was not previously gated. Appending exactly three regression node IDs now includes all 13 new lifecycle cases without adding unrelated controls. All original 175 census entries retain their positions and shard assignments; the floor rises to 178. Runtime and test source remain identical to reviewed commit fbe14192c7147450e7dfdbe88b9eee10e7b94403. Collection confirms 13 cases; the census checker passes at 178; the three affected Tests/CI contract modules pass 218 tests in 4.22s. Normal hash-pinned derived-artifact preflight exits zero with all checks passed. Receipts: /private/tmp/pr2882-transcript-census-collect.log; pr2882-transcript-census-check.log; pr2882-transcript-census-ci.{xml,log}; pr2882-transcript-census-preflight.log.

The broader current-source selection is NON-GREEN: 119 passed and 18 failed in 211.99s, including all nine trace modules (45 cases), the actual pending-projection journey and all 13 new lifecycle cases passing. Each of the 18 failures refuses a raw config source change before the requested widget assertions. Replaying exactly those 18 node IDs under immutable c2d6dcc3016708bec3473450016177b2ff602fb3 lifecycle methods and the same original 12-module collection context yields 18 failures, zero errors and 119 deselections in 24.32s; every failure is RecoveryRequired: raw_source_selection_changed. The outside-repository source-pin plugin preserves the real config getter, conftest and profile admission. Separate two-node collections produced two setup errors and are retained as non-qualifying harness evidence. No config bypass, registry reset, unrelated control change or full sweep. Receipts: /private/tmp/tldw-pr2882-lifecycle-integrated.{xml,log}; pr2882-profile-failure-nodeids.json; pr2882-profile-baseline-all18.{xml,log}; pr2882_source_pin_plugin.py.

CI-checker static qualification: two inherited Ruff findings (EXE001 and PIE810), zero new findings; fatal diagnostics pass. Its existing formatter drift is unchanged from fbe (32 formatting change-lines), so the checker formatter check remains NON-GREEN and no incidental reformat is included. Runtime/test scoped static receipts above remain valid. Git diff whitespace check passes.

Independent immutable runtime review by review_3029 found no actionable defect in fbe14192c7147450e7dfdbe88b9eee10e7b94403: first-mount/recompose projection retention and post-await detach guards preserve unconsumed signatures. The peer also provisionally verified this CI diff, exact old-entry preservation, 13-case collection, 218 contract passes and all 18 baseline failures. Final immutable CI review and task closeout remain under root ownership; status stays In Progress and acceptance criteria remain unchecked.

Root qualification on immutable fbe14192c7147450e7dfdbe88b9eee10e7b94403: the original nine trace/dispatch modules pass all45 cases. The broader12-module selection is retained as NON-GREEN119PASS/18FAIL in211.99s, with no collection errors. Its119 passes include all13 new lifecycle cases,47 existing Assistant-turn cases and the real pending question/approval sibling-view/remount journey (13 pending-projection cases pass). All18 failures stop at real config admission before UI assertions and reproduce with exact case IDs, identical12-module collection context and immutable c2 lifecycle methods; they remain inherited profile-harness failures, not suppressed or claimed repaired. XML/log: /private/tmp/tldw-pr2882-lifecycle-integrated.{xml,log}; exact baseline proof: /private/tmp/pr2882-profile-baseline-all18.xml. Artifact preflight passes on both runtimefbe and CI follow-upb2d; the final178-node census and218 CI contract cases pass.

Independent review_3029 clears immutable runtimefbe14192c7147450e7dfdbe88b9eee10e7b94403 and CI-onlyb2d24076ca1af97b4a020aa7d90448da9420d547. Runtime/Textual composition and retirement boundaries, screen cached-key preservation, all old census/shard positions, new13-case collection and exact baseline scope are reviewed. All acceptance criteria are qualified. This final task closeout changes metadata only; runtime/test bytes remain identical to reviewedfbe and census/checker bytes to reviewedb2d. Required hosted CI still gates publication integration; no full suite or live provider call was run.
<!-- SECTION:NOTES:END -->
