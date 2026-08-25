---
id: TASK-22211
title: >-
  Watchlists responsive layout needs hysteresis at its collapse boundaries
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 19:52'
labels:
  - performance
  - watchlists
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22211).

New with PR #2063. `UI/Watchlists_Modules/region_layout.py:132-175`:
`resolve_effective_layout` applies bare width thresholds with no `previous` state, and
`on_resize` recomputes per Textual Resize event. Crossing 145 columns by ONE cell flips
the right rail: region factory + mount/remove pair per flip
(`watchlists_workbench.py:226-309`), repeated per Resize during a drag. This is the
documented sub-2-cell width-flap trap; the Library media reader carries the fix
(`LAYOUT_HYSTERESIS_WIDTH = 4`, `Library/library_media_reader_state.py:16`, `:341-355`)
and Watchlists does not. Aggravator (medium confidence): `_available_layout_width` prefers
`workbench.size.width` (`watchlists_collections_screen.py:2999`), which is
scrollbar-sensitive — a scrollbar toggle at the boundary could flap the layout with no
user resize.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Repeated +/-1-cell width changes at a collapse boundary cause no mount/remove churn (hysteresis test at the boundary, both directions)
- [x] The width source is not flappable by a scrollbar toggle, or a code-level guard absorbs sub-hysteresis changes (the repo rule: never trust a CSS-only guard)
- [x] Approach consistent with the Library reader's hysteresis precedent
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add pure four-column hysteresis to the Watchlists resolver with bidirectional tests at every Read and management boundary, including priority-adjusted reopening order.
2. Make positive screen allocation the sole width authority, keep separate responsive history, require explicit recompute causes, and suppress equal-layout workbench requests.
3. Add a mode-local priority lease; preserve and park it across Article Focus and tab changes; harden explicit-open, focus, rollback, failed-section-swap, and cold-restart behavior with mounted tests.
4. Run only changed-functionality Watchlists tests and static checks for modified files, self-review the scoped diff, record evidence and Implementation Notes, then close TASK-22211.

ADR required: no
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: bounded stabilization of ADR-042 responsive policy using the existing Library hysteresis precedent; no new storage, ownership, dependency, or long-lived pane architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented pure four-column resolver hysteresis using the Library precedent, with positive screen width as the invariant width source and a separate responsive baseline. Added a mode-local priority lease that parks across sections, remains separate from Article Focus, clears only in its origin mode past the dead band, and is restored with the full baseline on token-correlated manual-apply rollback; failed section swaps also restore the rendered responsive baseline. Equal effective requests return before focus capture or token allocation. Review found and fixed stale rollback-token reuse when an explicit layout request is suppressed as an equal no-op; test_suppressed_manual_preference_change_does_not_reuse_stale_token verifies a later stale failure cannot roll back the current preference. Review also found and fixed a stale responsive baseline after a failed narrow-width section swap; test_section_factory_failure_rolls_back_mode_and_can_retry verifies the baseline is reconciled to the rendered fallback and the section can be retried successfully. Verification: targeted Watchlists changed-functionality suite passed 75 tests with 111 deselected and 2 dependency warnings; Ruff passed for all modified Python and test files; git diff --check passed. Modified implementation and coverage: tldw_chatbook/UI/Watchlists_Modules/region_layout.py, tldw_chatbook/UI/Screens/watchlists_collections_screen.py, Tests/Watchlists/test_watchlists_responsive_layout.py, Tests/Watchlists/test_watchlists_collections_screen.py, Tests/Watchlists/test_watchlists_cold_open_layout.py, and Tests/Watchlists/test_watchlists_scoped_rebuilds.py. Supporting design, plan, critique, and this task record were updated. No persistence or storage schema, CSS, or shared split-pane framework changes were made. ADR required: no; existing ADR 042 governs the reader-first responsive policy.
<!-- SECTION:NOTES:END -->
