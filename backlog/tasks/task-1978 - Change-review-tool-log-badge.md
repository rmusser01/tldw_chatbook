---
id: TASK-1978
title: 'Change review: ''changed outside direct file tools'' badge'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - console
  - change-review
  - agents
dependencies:
  - TASK-1971
  - TASK-1973
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cross-reference the run's recorded steps (AgentRunsDB) with the turn's changed files: a file no recorded file tool touched gets a `⚠ changed outside direct file tools` badge in the tree — turning the B..E attribution limit into signal (script side effects and external writers become visible AS SUCH). Copy is exact: 'outside direct file tools', never 'not by the agent' — script writes are agent work too, and badge absence is not proof of tool provenance.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A write_file-modified file carries no badge; a file created by a script the agent ran carries the badge
- [x] #2 Badge text matches the spec copy exactly and renders in monochrome
- [x] #3 A run with no recorded steps (older data) renders without badges rather than badging everything
- [x] #4 Badge derivation is tested against real recorded step shapes, not hand-built dicts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Provider `tool_touched_relpaths(row)`: run's persisted steps (real
   asdict(AgentStep) shape) through the SAME `tool_touched_paths` extractor
   the force-add carve-out uses, relativized to the row's root; None when
   the run has no recorded steps (older data -> no badges, AC#3)
2. `_leaf_label` appends the exact spec copy as a dim (monochrome) Text span
   for changed files absent from the touched set
3. RED tests: badged vs unbadged rows against a real fixture turn with
   steps appended via the production serializer; no-steps run bannerless;
   exact copy + dim style pinned
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Provider gains `tool_touched_relpaths(row)`: the run's persisted steps
(the real `asdict(AgentStep)` shape `AgentService._persist` stores) run
through the SAME `ChangeTurnTracker.tool_touched_paths` extractor the
force-add carve-out uses — badge and carve-out can never disagree about
provenance. Paths relativize to the row's root; `None` when the run has
no recorded steps, and the screen then renders NO badges (AC#3,
sabotage-verified: badging-everything failed the guard test).

`_leaf_label` appends the exact spec copy "⚠ changed outside direct file
tools" as a dim (monochrome) Text span for changed files absent from the
touched set; deletions and renames badge honestly (no file tool can
delete or rename — a true 'outside direct file tools' change). Derivation
is memoized per row and exception-guarded — a badge must never break the
review.

Tests: badged-vs-unbadged against a real fixture turn with a step
appended via the production serializer (real AgentStep dataclass through
dataclasses.asdict — AC#4), stepless-run guard, exact-copy + dim-span
pin. 261 green across all change-review suites before push.
<!-- SECTION:NOTES:END -->
