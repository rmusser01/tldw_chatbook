---
id: TASK-24608
title: Inspector row status classes have no CSS rule anywhere in the repo
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:54'
updated_date: '2026-08-30 01:39'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
  - css
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every run-inspector row is built with a console-inspector-row-<status> class and the modifier is swapped on every in-place update, but grep across all stylesheets returns zero matches for console-inspector-row, including the base class. For the Provider row this is harmless because the status word is the value. For Sources and Approvals it is not: Approvals carries status blocked when the pending count is above zero while its text reads only 'N pending', so a pending approval renders identically to none pending. The class has been attached, swapped and covered by passing tests for as long as the rule has been missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Blocked, running and ready inspector rows are visually distinguishable using semantic status tokens
- [x] #2 Colour is reinforcement only; every blocked row still reads as blocked from its text alone
- [x] #3 A repo check fails when a class attached in Python has no matching rule in the stylesheet bundle
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
grep -rn 'console-inspector-row' tldw_chatbook/css/ returned zero matches repo-wide -- not the base class, not one modifier -- while the class was built by f-string, swapped on every in-place update, and asserted by an existing test. The channel had never painted anything.

Two changes. (1) The class is now normalized through normalize_console_source_status, so the set of selectors is CLOSED to ready/running/blocked/muted instead of open to whatever string a producer invents; an arbitrary status can no longer mint a selector nothing styles. (2) Those four now have rules against $ds-status-* tokens, matching .console-staged-source-status, which had modelled this correctly all along one screen away.

Colour stays reinforcement: every row still reads as blocked from its own words, so The Label Before Color Rule holds. 'muted' deliberately keeps the default row colour -- an unrecognised status must not borrow the authority of a measured one.

The visible cost was on rows whose TEXT does not repeat the status: 'Approvals: 3 pending' carries status blocked and was pixel-identical to 'Approvals: 0 pending'.

Guard: a parametrised test asserts each of the four classes has a rule in the bundled stylesheet, plus a nine-case test pinning the raw-to-normalized mapping.

Modified: tldw_chatbook/Widgets/Console/console_run_inspector.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_run_inspector.py.
<!-- SECTION:NOTES:END -->
