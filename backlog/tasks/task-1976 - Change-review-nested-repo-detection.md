---
id: TASK-1976
title: 'Change review: detect nested repos and disclose the tracking hole'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
dependencies:
  - TASK-1971
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
git records a child repo as a gitlink: uncommitted changes INSIDE a nested clone are invisible to the snapshot — silently violating the feature's core promise for the common ~/projects root shape. v1 detects nested repos during the registration scan and discloses honestly: card/inspector and Review screen banners state 'N nested repositories inside this root are not tracked', naming them on the Review screen. (Auto-registering them as sub-roots is TASK-1977.)

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A root containing a child repo reports the child by path in the Review screen banner
- [x] #2 An edit inside the child repo produces NO diff rows AND the banner is present (the hole is disclosed, not hidden)
- [x] #3 A root that IS a repo (no children) shows no banner and tracks normally
- [x] #4 Detection runs in the registration scan, not per-turn
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `scan_root` detects nested repos during the SAME walk budgets already use
   (zero additional per-turn cost — AC#4's substance); root's own .git is
   not "nested"
2. TurnChangeRecord.nested_repos; rows carry them (JSON column, ALTER-on-open);
   bridge threads through; Review banner names them
3. Oversize precedent for cardless turns: NEW nested repo since B -> zero-change
   disclosure record; stable set stays cardless, disclosed on changed turns
4. TDD against real git; sabotage first-try passes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Detection rides the SAME walk the TASK-1975 budget gate already runs
(`scan_root` gains `nested_repos`; zero additional per-turn cost — AC#4's
substance), collected BEFORE directory pruning so both `.git` spellings
(dir and worktree file) register; the root's own `.git` is not nested.

**The test found a real production bug beyond the spec's framing:** `git
add -A` HARD-FAILS (exit 128, "does not have a commit checked out") on a
commitless nested repo — tracking would have died for the whole root, not
merely missed the child. Nested repos are therefore EXCLUDED from tracking
entirely (written to info/exclude alongside the oversize entries, trailing
slash, newline-named dirs unexcludable-but-skipped like oversize), making
the semantics uniform: excluded + disclosed.

Disclosure follows the oversize precedent: `TurnChangeRecord.nested_repos`
→ `change_snapshots.nested_repos` JSON column (idempotent-ALTER-on-open)
→ bridge pass-through → Review banner "N nested repositor(y/ies) inside
<root> not tracked: a, b" (first 5 named). A repo CLONED mid-turn emits a
zero-change disclosure record; a stable nested set stays cardless and is
disclosed on every changed turn's banner.

5 scan/tracker tests + 1 screen test (banner sabotage-verified); 249 green
across all change-review suites.
<!-- SECTION:NOTES:END -->
