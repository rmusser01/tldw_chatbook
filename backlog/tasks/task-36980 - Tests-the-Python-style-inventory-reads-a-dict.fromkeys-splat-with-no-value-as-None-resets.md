---
id: TASK-36980
title: >-
  Tests: the Python style inventory reads a dict.fromkeys splat with no value as
  None resets
status: To Do
assignee: []
created_date: '2026-10-26 02:52'
updated_date: '2026-10-26 02:54'
labels:
  - tests
  - adr-161
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.7's run-tick change rewrote the Send-reason strip's literal `set_styles(width=None, height=None)` as `set_styles(**dict.fromkeys(stale))`, where `stale` is a list computed from `styles.inline.has_rule`. The behaviour was the same, but ADR-161's AST style inventory (Tests/UI/python_style_inventory.py) records any `**` keyword as prop `*` with a runtime value. test_python_style_ratchet's HARD-ZERO floor therefore rejected it, and the required check stayed red across five pushed heads (see the lessons-testing-evidence entry 'A syntax-level guard cannot read a computed value'). The branch fixed it by spelling two guarded literal-None resets.

`dict.fromkeys(<names>)` with no value argument, or with a literal `None` value, maps every key to `None` whatever the names are. So it is statically a set of None resets, and resets are independently legal under ADR-161. The inventory could record it as a reset (per name when the names are a literal collection of strings, `*` otherwise), so the next refactor of a guarded reset does not trip the same floor. This is optional hardening; the lesson already covers the trap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A `set_styles(**dict.fromkeys(<names>))` call, with no value argument or a literal `None` value, is recorded as a reset and is not a violation, whether `<names>` is a computed expression or a literal collection
- [ ] #2 When `<names>` is a literal tuple or list of strings, the reset is recorded once per named property; otherwise it is recorded as prop `*`
- [ ] #3 A splat over any other expression, or a `dict.fromkeys` with a non-None value argument, is still recorded as an opaque runtime write and still fails the floor
- [ ] #4 Tests/UI/test_component_pattern_governance.py has a case for each outcome above, and the whole file passes
<!-- AC:END -->
