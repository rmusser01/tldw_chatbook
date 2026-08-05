---
id: TASK-1975
title: 'Change review: root budgets, oversize excludes, snapshot retention/GC'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
  - performance
dependencies:
  - TASK-1970
  - TASK-1971
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the substrate bounded. Registration scan enforces max_files/max_total_bytes per root — over budget disables tracking for that root with honest copy ('narrow the root or add excludes'), never a silent half-track. Git cannot exclude by size, so the scan appends files over max_file_bytes to info/exclude dynamically and the review discloses 'N oversized files untracked'. Retention: prune change_snapshots rows past retention_days (default 30), then reflog expire + git gc --prune on the shadow repo, on the existing maintenance path; orphaned shadow repos (root removed) GC'd by age.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An over-budget root registers with tracking disabled and the stated copy; an in-budget root tracks
- [x] #2 A file exceeding max_file_bytes lands in info/exclude, is absent from diffs, and the untracked count is disclosed in the review
- [x] #3 Retention run prunes rows older than the cutoff AND shrinks the shadow repo (object count measured before/after)
- [x] #4 A shadow repo whose root no longer exists is removed by GC
- [x] #5 All knobs read from the flat [change_review] section (NOT dotted-nested -- the get_cli_setting dotted form has dropped defaults before) with env-var overrides per repo convention
- [x] #6 An oversized file CREATED during a turn is excluded at E-snapshot time and disclosed, never committed to the shadow store
- [x] #7 A history row whose snapshots were pruned renders 'pruned by retention' instead of erroring
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `change_bounds.py`: flat `[change_review]` knob readers (max_files, max_total_bytes, max_file_bytes, retention_days) with env overrides; root budget scan; oversize scan appending to info/exclude (idempotent, disclosed count persisted per snapshot row)
2. Wire budgets into `ChangeTurnTracker.begin_turn` (over-budget root -> tracking_error copy 'narrow the root or add excludes') and oversize scan into every `snapshot()` call (B AND E — AC#6 mid-turn creations)
3. AgentRunsDB: `untracked_oversize` column on change_snapshots (additive, CREATE-on-open migration discipline); provider/screen disclose 'N oversized files untracked'
4. Retention: `prune_change_history(now)` — delete rows past retention_days, reflog expire + gc --prune on each shadow repo, rm orphaned repo dirs by age; hook onto the existing maintenance path
5. Screen: a selected history row whose shas no longer resolve renders 'pruned by retention' not a traceback
6. TDD throughout against real git; sabotage every first-try pass
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three new/changed layers, all against real git:

**`Workspaces/change_bounds.py` (new):** flat `[change_review]` knob reader
(`change_review_setting` — env `TLDW_CHANGE_REVIEW_*` beats config beats
default; unparseable values fall back) and `scan_root` (walk with the same
directory prunes as the shadow repo's forced excludes, symlinks never
followed, early-abort once either budget cap crosses, oversize list
root-relative).

**Enforcement:** `ChangeTurnTracker._baseline` budget-gates each root BEFORE
any snapshot ("root over change-tracking budget (N+ files / M+ bytes) —
narrow the root or add excludes; tracking disabled for this turn").
`ShadowRepo.snapshot` re-scans every call and appends oversized files to
`info/exclude` (static block rewritten first by ensure_initialized, so
entries never accumulate; `_exclude_pattern` escapes glob chars — paths are
data), exposing the set as `last_oversize_excluded`. `end_turn` filters
oversized paths out of the force-add carve-out (force-add defeats ignore
rules, not the size cap), attaches `untracked_oversize` to records, and
emits a ZERO-change record when the only turn event was a new oversized
file (a stable oversize set stays cardless — noise control). Documented
limit: excludes only stop untracked files; a file committed small that
later grew stays tracked and shows in diffs.

**Retention (`Workspaces/change_retention.py`, new):** `prune_change_history`
= row prune by `retention_days` cutoff → per-container sweep of the shadow
data dir (`<hash>/git` layout): root vanished + aged past retention → rmtree
(orphan GC); root alive but NO remaining rows → reset the whole container
(ancestry keeps every snapshot reachable, so gc alone can never shrink a
live chain — reset is the shrink move, next snapshot re-inits); rows remain
→ reflog expire + `gc --prune=now` under the repo lock. `run_retention_for_app`
wraps the production layout (runs DB sits beside ChaChaNotes) and never
raises. Scheduled in `app.schedule_media_cleanup` on its OWN timer BEFORE
the media-enabled early return — disabling media cleanup must not silently
disable snapshot retention; `retention_days <= 0` disables inside the pass.

**Disclosure/pruned rendering:** `change_snapshots.untracked_oversize`
column (idempotent-ALTER-on-open migration, tested against a pre-column
file), bridge threads the count through, Review screen banners "N oversized
file(s) untracked" (AC#2) and — via `ShadowRepo.has_snapshot` +
`provider.snapshots_pruned` — renders "history … was pruned by retention"
instead of the generic diff-unavailable copy (AC#7).

24 new tests in `Tests/Workspaces/test_change_bounds.py` + 2 screen tests;
noise-control branch and the app runner sabotage-verified; 240 green across
all change-review suites.
<!-- SECTION:NOTES:END -->
