---
id: TASK-36067
title: Promote Console send verification controls into regression tests
status: Done
assignee:
  - '@codex'
created_date: '2026-10-22 00:18'
updated_date: '2026-10-22 00:40'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Promote existing test-only controls without changing product boundaries or runtime behavior.

1. Confirm the ten merged test files exactly match remote dev; classify local profiles, logs, baseline source and logging wrappers as disposable evidence.
2. Add deterministic current-observer recycled-address coverage in Tests/CI, using isolated AST extraction and real sys.monitoring/asyncio with no historical Git dependency.
3. Add separately selectable source/clear held-gate failure cases to the existing projection test file, using a test-helper callback and real worker retirement. Preserve all existing assertions, timeouts, CI census and shard allocation.
4. Run the three new cases and focused CI contracts with native private temp/profile parents, validate a failing observer identity mutation, check lint delta and diff, obtain independent review, then commit and create the test-only PR.
5. Delete all PR3189 local evidence copies and native profile directories; keep only merged tracked history in its archived recovery reference, then delete its local/remote branch. Leave unrelated checkouts, refs, stashes and drafts alone.
<!-- SECTION:PLAN:END -->

## Implementation Notes

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
