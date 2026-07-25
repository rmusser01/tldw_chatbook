---
id: TASK-582
title: Expose skill-script sandbox limits as configuration
status: To Do
assignee: []
created_date: '2026-07-25 14:35'
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
- [ ] #1 Each ScriptRunLimits field can be overridden from a `[skills]` config section, falling back to the current default when unset
- [ ] #2 Overrides are read via the three-argument get_cli_setting form, with a test that would fail if the unreachable section-dict form were used
- [ ] #3 Out-of-range or non-numeric values are rejected in favour of the default rather than producing an unbounded or zero budget
- [ ] #4 A wall-clock override cannot exceed the confirm/run-budget envelope in a way that strands the agent run
- [ ] #5 Docs/Features/Skills-Script-Execution.md is updated: the limits table stops saying the values are not configurable, and documents each knob
<!-- AC:END -->
