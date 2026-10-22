---
id: TASK-36067
title: Promote Console send verification controls into regression tests
status: Done
assignee:
  - '@codex'
created_date: '2026-10-22 00:18'
updated_date: '2026-10-22 05:52'
labels: []
dependencies: []
type: task
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the useful frame-identity and held-worker cleanup assertions from PR3189 in the remote test suite so local diagnostic artifacts can be deleted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A deterministic recycled-address regression exercises the current replay observer and verifies distinct retained invocation identities and empty live mappings after returns.
- [x] #2 Source-held and clear-held fixture failures retire every real worker and remain separately selectable without changing existing journey assertions or limits.
- [x] #3 Targeted native verification and independent review complete, useful tests are published in a PR, and PR3189 local evidence and merged branch/worktree are cleaned without touching unrelated work.
- [x] #4 The mapped response-rule Stop journey tolerates valid deferred control publication, still rejects persistently unavailable Stop, and preserves original action/worker-failure assertions and CI limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only synchronization and cleanup within the existing checked presentation/replay contract (TASK-35444/TASK-35645); product boundaries and committed limits remain unchanged.

1. Preserve the original promotion and cleanup implementation notes and AC1-3.
2. Diagnose the exact queue-rebased ea19 UI7 failure using complete failed-job output, current response-rule/control rendering and untouched native targeted variants.
3. Establish a deterministic real deferral/held-control reproduction before choosing a fix; distinguish the source phase/run label from published Stop readiness and retain a persistent-broken-control negative check.
4. Repair only the owning test precondition/cleanup if the real deferred rendering contract is confirmed. Keep genuine worker failures visible, original Stop/draft/helper-retirement assertions, 5-second action limits, production sources and CI allocation unchanged.
5. Run proportionate targeted native cases plus the promoted controls and relevant CI contracts, compare lint against exact ea19 baseline, review the complete diff and obtain a fresh independent skill review of the new head.
6. Commit/push the disarmed synchronized branch; recheck exact head/threads then re-arm under the human-authorized independent-review substitution. Continue monitor and merge-only cleanup; delete local diagnostic evidence rather than archiving it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Promoted three useful local controls from merged PR #3189 into the normal test collection. The new CI test extracts only the current replay observer, its predicate and original monitoring helpers into an isolated namespace, forces a reused address key, and checks 26 distinct completion tokens, no retained frames, and empty live mappings after each return. The projection helper exposes an optional failure-injection callback; the two private-profile cases fail while the source or clear gate is held and require all captured real workers to have exited. The clear case captures both source and sibling workers.

The ten tracked test files changed by #3189 exactly match landed dev cb442df67c. Existing projection assertions and its 16 other functions are unchanged; removing the optional callback statements and argument restores the original helper AST. No production files, workflow, census, shard placement, dependencies or committed timeouts changed. ADR required: no; test-only controls within existing boundaries.

Verification:
- Final native selection: all 3 new cases passed, 2 existing dependency warnings, 235.62s (242.34s launcher); mounted source/clear calls 112.70s/112.36s.
- Focused CI P0/census/task-ID contracts, including the new observer: 244 passed, 1 existing warning, 90.22s.
- An in-memory mutation replacing fresh observer tokens with the recycled numeric address failed the new assertion; unchanged current source passed. No historical Git revision is required by the committed test.
- Ruff: new CI test 0 findings; projection test baseline/current both 16, none introduced. New-file formatting and git diff whitespace checks pass.
- Native runs used the shared .venv Python and verified native-private temp/profile parents; the 900-second allowance was CLI-only. No full suite, repeated-run qualification or future hosted result is claimed.

An initial observer harness failure came from a missing isolated lock binding and was corrected. The initial clear case also failed at the existing sibling-card setup wait, before the injected clear failure; no root cause was established. The same case then passed isolated (137.00s call) and in the final three-case run (112.36s). Temporary diagnostics were removed; this is not a claim that the earlier setup failure or general UI timing qualification was fixed.

Independent working-diff and exact committed-head review at 699a0b869c380213566d6b3ded6411b3a63a8b6b found no actionable issues. Tests are published in https://github.com/rmusser01/tldw_chatbook/pull/3203. Hosted checks are pending; auto-merge is not armed.

Cleanup: original local/remote codex/fix-console-send-recovery branch, worktree and its parent private profiles are deleted. The 19,418-file evidence copy, baseline copies, caches and logs were deleted at the user's corrected request. The archived recovery ref now points only to the merged tracked implementation head 28883f3c37. Main checkout HEAD/status and existing stashes were unchanged; unrelated worktrees, branches and drafts were left alone. All new-run native-private directories and 1,665 generated cache/input artifacts were also deleted before closeout. The new active worktree contains only committed test/task sources, with no retained local evidence.

Queue integration ea19 failed UI7 job119643219360 in Derived39723849662: mapped learning/Stop size0 saw drafting/run label but Stop.display False after one pilot pause; finally wait_for_complete then raised WorkerCancelled. Other18 lanes passed; required aggregate119649097628 failed its prerequisite. Auto-merge disabled before pulling queue-rewritten branch with rebase; investigation started without rerunning hosted CI or changing production/limits.

UI7 follow-up: source tracing showed that refresh_rule_controls publishes drafting copy immediately while checked control presentation can legitimately defer. The test now waits at most five seconds for the real Stop button to be mounted, displayed, visible, enabled, have positive geometry and be the compositor hit at Pilot's default click origin. It then performs exactly one original Pilot click and retains every original draft, phase, unsettled-helper and cleanup assertion. Existing helper-entry/cleanup limits remain five seconds; no production/workflow/census/allocation/dependency limit changed.

Cleanup waits each app worker, accepting WorkerCancelled only for this Console node's console-sync group, which coalesced replay may replace. Parent-task cancellation is re-raised first; other node/group cancellations and WorkerFailed still propagate. The hosted trace proves an original hidden-Stop assertion followed by WorkerCancelled; it does not identify the cancelled worker's node/group.

Verification: exact native mapped cases at both sizes plus observer/P0/census/task-ID/response-rule CI contracts: 300 passed, 2 existing dependency warnings in 135.61s (141.26s launcher); mapped calls 31.61s/30.26s. CLI-only timeout=900; committed limits unchanged. Other 13 top-level response-rule functions remain AST-exact against ea19. Ruff baseline/current each have the same pre-existing E721, zero introduced; changed-function formatting and diff checks pass. Formatting preserved the tested AST.

Deterministic diagnostics held the real maintenance gate immediately after helper entry: old observation failed with phase drafting/run copy present/Stop hidden; finite replay release passed the repaired journey (37.87s pytest), and persistent hold failed specifically at the five-second readiness wait (39.72s). A visibility-only intermediate attempt still returned False from the single click; geometry/hit-target readiness corrected that staged attempt. Diagnostic-only staging replaced the first idle pause and allowed 90 seconds for helper-entry preparation; final 300-case native verification used the exact committed five-second waits. Real Textual Worker probes verified expected-sync acceptance and propagation of parent cancellation, other-node/group cancellation and WorkerFailed; omitting the parent guard swallowed cancellation. Disposable AST staging wrappers are evidence, not additional regression sources, and will be deleted.

Untouched ea19 native baseline was 1 failed/1 passed: its size0 failed earlier at the original helper-entry wait, before Stop readiness; cause unestablished. Final exact mapped cases pass, without claiming to repair that earlier setup failure or general timing qualification. Initial probe restoration/idle-wait attempts were rejected and are not passing evidence. Existing source-held/clear-held promoted controls were not repeated in this follow-up because their source is unchanged and ea19 UI5 passed; the observer was rerun with the focused contracts. No full suite or future hosted success claimed.

Independent working-diff review found no actionable findings after fixing its parent-cancellation finding. The testing-evidence lesson records the observed replay/click-target distinction and cancellation guard. ADR required: no, test-only precondition and cleanup within existing boundaries. Exact committed-head review remains required before push/re-arm; the human-authorized independent-review substitution persists. Auto-merge is currently off; queue integration will receive fresh hosted CI after publication.
<!-- SECTION:NOTES:END -->
