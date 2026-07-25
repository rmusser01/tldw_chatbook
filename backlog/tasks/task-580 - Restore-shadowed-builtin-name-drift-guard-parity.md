---
id: TASK-580
title: Restore shadowed-builtin-name drift-guard parity
status: To Do
assignee: []
created_date: '2026-07-25 14:35'
labels:
  - skills
  - tests
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Library` has a failing drift-guard test: `_SHADOWED_BUILTIN_NAMES` in `tldw_chatbook/Library/library_skills_state.py` is missing the `rewind` and `generate-image` console commands, which arrived with other merged features (commits 73ed08aa8 and 62f0f918c).

The guard exists so a user-installed skill cannot silently shadow a built-in command name. Every runtime tool that joined `RUNTIME_TOOL_NAMES` has correctly been added to this set, so the runtime side is in order — the gap is on the console-command side, where two commands were added without updating the list.

This is a genuine (if small) correctness gap, not just a red test: a skill named `rewind` or `generate-image` would currently not be recognised as shadowing a built-in. It has been carried as an accepted baseline failure across several branches, which erodes the signal value of the suite.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `rewind` and `generate-image` are present in `_SHADOWED_BUILTIN_NAMES`
- [ ] #2 The drift-guard test in Tests/Library passes without any accepted-baseline caveat
- [ ] #3 A skill named after either command is handled the same way as one shadowing any other built-in
- [ ] #4 The guard's failure message points at what to update, so the next command addition fixes it rather than accepting it as a baseline
<!-- AC:END -->
