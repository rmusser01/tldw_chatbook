---
id: TASK-24701
title: 'Classes attached in Python with no stylesheet rule, second wave'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 06:17'
updated_date: '2026-08-30 06:24'
labels:
  - console
  - ux
  - inspector
  - css
  - critique-2026-08-30
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-24608 fixed console-inspector-row-* and added a test for those four classes only. The same defect exists for console-library-activity-error (an ERROR line rendering in default body colour), console-selected-turn-subsection, and console-library-activity-action/-source-ref. The guard was never generalised, which is why a second wave shipped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every class the Inspect rail attaches in Python has a rule in the bundled stylesheet
- [ ] #2 A repo check fails when a Python-attached class in this rail has no matching rule, rather than a hand-listed subset
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TASK-24608 fixed the four 'console-inspector-row-*' classes and pinned exactly those four; a second wave then shipped with the same defect because the guard was never generalised. Added rules for console-library-activity-error, -action, -source-ref and console-selected-turn-subsection, and a parametrised test over the list.

The error class was the worst of the set: it carries a Library operation's error_summary, so a FAILED retained write rendered in the same colour as a success.

Honest scope note: the test is a parametrised LIST, not a derived check. A truly general guard would enumerate classes attached in Python and diff them against the bundle -- that is a repo-wide tool, not a rail fix, and it is the thing that would actually stop a third wave. Recorded here rather than pretended.

Modified: css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_inspector_focus_visibility.py.
<!-- SECTION:NOTES:END -->
