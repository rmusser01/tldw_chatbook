---
id: TASK-580
title: Restore shadowed-builtin-name drift-guard parity
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 14:35'
updated_date: '2026-07-25 20:20'
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
- [x] #1 `rewind` and `generate-image` are present in `_SHADOWED_BUILTIN_NAMES`
- [x] #2 The drift-guard test in Tests/Library passes without any accepted-baseline caveat
- [x] #3 A skill named after either command is handled the same way as one shadowing any other built-in
- [x] #4 The guard's failure message points at what to update, so the next command addition fixes it rather than accepting it as a baseline
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added 'rewind' and 'generate-image' to _SHADOWED_BUILTIN_NAMES, clearing a drift-guard failure that had been carried as an accepted baseline across several branches.

Not just a red test: the set is what skill_name_shadows_builtin consults, so a skill named after either command was silently NOT recognised as shadowing a built-in, unlike every other console command. Small, but a real correctness gap.

AC#4 (make the guard self-service): all three assertions in the drift test now name the file to edit and say explicitly not to accept the failure as a baseline. The previous message named only the missing strings, which is what let it be waved through repeatedly — the guard fired correctly every time, it just did not tell anyone what to do about it.

AC#3: added a behaviour test asserting skill_name_shadows_builtin() returns the normalized name for both commands, including the whitespace/case-normalizing path. Pinned by literal name rather than re-deriving from the registry, so it keeps failing if someone removes them — re-deriving would make the test tautological against the very set under test.

Tests: Tests/Library 16 in the module and 1025 across Tests/Library + Tests/Skills, all green with no baseline caveat for the first time in this work stream. ruff clean.

Files: tldw_chatbook/Library/library_skills_state.py, Tests/Library/test_library_skills_state.py
<!-- SECTION:NOTES:END -->
