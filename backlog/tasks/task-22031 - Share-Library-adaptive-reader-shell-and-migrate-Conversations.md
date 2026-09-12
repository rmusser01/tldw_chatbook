---
id: TASK-22031
title: Share Library adaptive reader shell and migrate Conversations
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 23:24'
updated_date: '2026-08-25 13:27'
labels:
  - library
  - ui
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract the shipped Media reader structure into the Library-local adaptive shell and migrate Conversations as its first additional consumer. Preserve Media domain behavior while adding the approved list comfort expansion and complete read-only conversation work pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media uses the shared Library-local shell with its existing modes actions selection loading recovery and preference compatibility preserved
- [x] #2 Library and Conversations list are independently collapsible while the conversation work pane remains mounted
- [x] #3 Collapsing Library expands the destination list toward its comfort cap without changing saved widths
- [x] #4 Conversations exposes the complete saved transcript with Read and Info modes Find and Open in Console
- [x] #5 Selected and loaded conversation identity stay truthful under rapid traversal stale workers deletion and retry
- [x] #6 Shared Library preferences and Media legacy fallback follow ADR-086 without responsive preference writes
- [x] #7 Automated geometry race capability and Media regression tests pass with a live TUI walkthrough at representative terminal sizes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory Media and Conversations capabilities
2. Extract and prove the shared shell
3. Migrate shared preferences and adaptive geometry
4. Add the fenced Conversations reader
5. Run automated and live verification

ADR required: yes
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md
Reason: implements the accepted Library structural boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the Library-local pure adaptive layout policy and retained three-region shell, then moved Media onto the shared structure without changing its domain modes or selection/recovery behavior.
- Added the permanent Conversations reader with complete transcript loading, Read and Info modes, Find, Open in Console, bulk read-only preview, deletion/retry truth, and generation-fenced selected-versus-loaded identity.
- Added shared Library and destination-specific preference normalization, Settings ownership, legacy Media fallback, latest-intent persistence, rollback, peer synchronization, and delayed Settings reconciliation under ADR-086.
- Added focused state, shell, geometry, Settings, persistence, race, and Media regression coverage plus the capability inventory and 105-file live evidence bundle in `Docs/superpowers/reviews/evidence/task-22031/`.
- Post-rebase verification: the 101-test reader gate and 82-test layout/config gate passed; the 920-test affected gate produced 907 passes plus the same 13 `test_library_entry_compose_once.py` failures reproduced on `origin/dev`; the architecture gate produced 431 passes plus the known `test_prompt_and_skill_row_handlers_route_to_their_canvas` baseline failure. Targeted Ruff/format, `compileall`, generated CSS synchronization, and diff checks passed.
- The repository-wide xdist run collected 60,142 tests and reached 99% after 4h13m, but was interrupted after three recovered worker exits, a stalled tail, and file-descriptor growth. Its affected failures were reproduced serially: branch-owned Conversations/config tests passed, while the one branch-local stale Media hysteresis oracle was corrected and independently re-reviewed. Existing repo-wide Ruff/format debt remains unchanged in scope (510 Ruff findings; 1,687 files would reformat).
- Live verification passed for Conversations and Media at 160x50, 120x35, 100x30, and 80x24, including collapse/restore, list comfort expansion, focus/footer behavior, long transcript Find, rapid selection, error/retry, deletion, bulk preview, and preference-race recovery. Final independent review: APPROVE with no remaining Critical or Important findings.
- Reviewed and regenerated the required production diagnostic inventory after rebase. The two added fixed Library warnings and the generalized Settings warning contain no user content, secrets, filesystem paths, or URLs; the resulting inventory changes are limited to the expected Library/Settings fingerprints and the Library +2 call count.
- Triaged all three delayed Qodo comments: added environment-first resolution and regression coverage for every shared/destination reader preference, and made the Conversations request default consume the shared page-size constant. Rejected the reported Items-priority overflow as a false positive because the existing width-60 regression proves the outer cap returns `(0, 32, 18)` without overflow.
- No new lesson entry was needed; the existing interrupted-pytest/`lastfailed` lesson governed the incomplete repository-wide result instead of treating stale cache nodes as current evidence.
<!-- SECTION:NOTES:END -->
