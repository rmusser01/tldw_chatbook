---
id: TASK-13214
title: >-
  Library shadow-name drift guard is red on generate-video/stream-video, masking
  later drift
status: Done
assignee: []
created_date: '2026-08-10 00:29'
updated_date: '2026-08-16 10:05'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Library/test_library_skills_state.py::test_shadow_name_set_stays_in_sync_with_real_sources fails on origin/dev: ConsoleCommandRegistry names not covered: {'stream-video', 'generate-video'} — both present in console_command_grammar.py but absent from _SHADOWED_BUILTIN_NAMES (library_skills_state.py). Introduced by the video-generation work. Impact beyond the two names: the guard asserts three subsets IN ORDER, so whichever fires first masks every gap underneath it. Demonstrated during supervisor-fleet PR 2a — two newly added runtime tools (wait_agents/check_agents) were missing from the same set, and the RUNTIME_TOOL_NAMES assertion fired first, completely hiding the video gap until the tool names were fixed. While this guard is red, anyone adding a runtime tool or console command gets no drift signal at all, which is precisely the erosion the test's own message warns about ('do not accept this as a baseline failure (task-580)').
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 generate-video and stream-video are covered by _SHADOWED_BUILTIN_NAMES (or deliberately exempted with a documented reason)
- [x] #2 The guard passes on a clean dev checkout
- [x] #3 The assertion reports ALL uncovered names across the three sources in one failure rather than short-circuiting on the first subset, so one gap cannot mask another
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Still red at 2026-08-11 on feat/rag-p2a-instrument-renewal (785ce369c) — re-verified directly, not inferred. Met again during TASK-15020/B1 while running Tests/Library whole (1742 passed, 1 failed); nothing in that arc's diff is in the test's import path. Sibling pre-existing failures surfaced in the same sweep and filed separately: TASK-15500 (scope-pipeline notify copy, 4 tests) and TASK-15501 (Console left-rail composition). Recording the second independent sighting because AC#3's masking argument is what makes this High: while this guard is red, no one adding a console command or runtime tool gets a drift signal.

Third sighting, 2026-08-16, on `chore/rag-16688-16788-residue` (base dev `c2f30862c`, HEAD `a6f04ea68`) during TASK-16688's `Tests/Library` battery — re-verified directly (`pytest Tests/Library/test_library_skills_state.py` → 1 failed, 15 passed), not inferred. The guard is still red, but the uncovered name has CHANGED: `ConsoleCommandRegistry names not covered: {'research'}`. generate-video/stream-video are now covered (AC#1 satisfied on dev), so what is left is the next gap underneath them — AC#3's masking argument playing out exactly as written. `research` is the `/research` Console command added by `e1f3a4424` (task-16481, "deliver completed research runs into the originating chat"), an ancestor of that branch's base. Not fixed there: out of that arc's scope, and this task already owns the guard. Recorded here rather than filed as a duplicate.
<!-- SECTION:NOTES:END -->

**Fourth sighting-adjacent finding (2026-08-16, the 16688/16788 batch's
final review, F6):** `expand_document` (TASK-16174's gated builtin) is
MISSING from `_SHADOWED_BUILTIN_NAMES`, although every other config-gated
builtin is listed there precisely because this drift guard cannot see
gated tools — a skill could shadow it undetected. Same family as the
`research` gap above; whoever repairs the guard should add it in the same
pass.

**CLOSED 2026-08-16 (fix/task-13214-shadow-guard).** AC#1 was already
satisfied on dev (generate-video/stream-video covered by the video arc);
what remained was the masked tail. AC#3 shipped as the structural fix:
the guard now collects FOUR sources (RUNTIME_TOOL_NAMES, the live
catalog, **the gate table via `gateable_builtin_tools()` — so gated-OFF
tools are no longer invisible (F6)**, and the Console command registry)
and asserts ONCE, reporting every gap across every source. The RED run
proved the design: it surfaced BOTH remaining gaps simultaneously —
`research` (masked) and `expand_document` (invisible to the old guard by
construction). Both added to `_SHADOWED_BUILTIN_NAMES` with reasons.
AC#2: `Tests/Library/` 1995 passed / 2 skipped / **0 failed** — the
standing dev red is gone. Behaviour note: skills named `research` or
`expand_document` are now correctly flagged as shadowing builtins (the
set's consumer at `library_skills_state.py:287` is the shadow-warning
path — that is the feature working).
