---
id: TASK-582
title: Expose skill-script sandbox limits as configuration
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 14:35'
updated_date: '2026-07-25 20:54'
labels:
  - skills
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The skill-script sandbox budget (`ScriptRunLimits` in `tldw_chatbook/Skills_Interop/skill_script_runner.py`) is a set of hardcoded defaults: 10s CPU, 512 MiB address space, 128 open files, 8 MiB max file size, 60s wall clock, 64 KiB retained output per stream. The design anticipated these being overridable from a `[skills]` config block, but only `script_scratch_root` was actually wired, and the feature documentation currently states plainly that the limits are not configurable.

The defaults are deliberately conservative, which means a legitimate long-running or output-heavy skill script has no way to be accommodated short of a code change. Conversely, a user who wants a tighter budget cannot impose one.

Note the trap this must avoid: `get_cli_setting("skills", {})` silently returns `{}` for any section name without a dot (config.py), so the section-dict form would make every knob permanently unreachable. The three-argument form is required, and the existing `script_scratch_root` read is the working precedent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each ScriptRunLimits field can be overridden from a `[skills]` config section, falling back to the current default when unset
- [x] #2 Overrides are read via the three-argument get_cli_setting form, with a test that would fail if the unreachable section-dict form were used
- [x] #3 Out-of-range or non-numeric values are rejected in favour of the default rather than producing an unbounded or zero budget
- [x] #4 A wall-clock override cannot exceed the confirm/run-budget envelope in a way that strands the agent run
- [x] #5 Docs/Features/Skills-Script-Execution.md is updated: the limits table stops saying the values are not configurable, and documents each knob
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added resolve_script_run_limits(), which builds the sandbox budget from ScriptRunLimits defaults plus any [skills] overrides, and wired run_skill_script to use it instead of a bare default.

Six knobs, one per field: script_cpu_seconds, script_address_space_bytes, script_open_files, script_file_size_bytes, script_wall_clock_seconds, script_output_cap_bytes. Omitting any key keeps its default.

AC#3, the part that matters: a value that is non-numeric, non-positive, or non-finite is REJECTED in favour of the default rather than applied. The governing rule is that a misconfigured limit must never end up MORE permissive than the default — script_cpu_seconds = 0 yields the 10s default, not an unlimited run. bool is excluded explicitly since it is an int subclass and 'true' is a config mistake, not a budget of 1.

AC#4: script_wall_clock_seconds is clamped to MAX_SCRIPT_WALL_CLOCK_SECONDS = 600. A run holds a worker thread and sits inside the agent's own run budget, so an unbounded value would strand the turn rather than merely permit a slow script.

AC#2: the reachability trap is pinned by a test whose fake get_cli_setting mirrors config.py's real behaviour of returning the default for a non-str key — so if the implementation ever switches to the section-dict form (get_cli_setting('skills', {})), which silently returns {} for any section without a dot, the override assertion fails rather than the knobs going quietly dead.

Tests: Tests/Skills/test_script_limits_config.py (10) written RED-first, all 10 failing before the resolver existed. Tests/Skills 369 passed, ruff clean (also removed a local ScriptRunLimits import that the change made unused).

Files: tldw_chatbook/Skills_Interop/local_skills_service.py, Tests/Skills/test_script_limits_config.py, Docs/Features/Skills-Script-Execution.md
<!-- SECTION:NOTES:END -->
