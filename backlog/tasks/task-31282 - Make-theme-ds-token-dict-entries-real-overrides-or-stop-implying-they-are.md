---
id: TASK-31282
title: Make theme ds-token dict entries real overrides (or stop implying they are)
status: To Do
assignee: []
created_date: '2026-09-04 19:03'
labels:
  - themes
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-31264 established that a theme's variables dict cannot override any ds-* token defined in a tcss source (per-source variable scope: file tokens win). The 13 themes carrying 'full ds-* token sets' are therefore documentation, not behavior — misleading for the next theme author. Two honest exits: (a) move ds token defaults out of tcss into TldwCli.get_css_variables() so theme dicts genuinely win — blocked on ~35 test harnesses that parse tldw_cli_modular.tcss standalone and would hit unresolved variables; or (b) keep formulas as the only mechanism and trim/annotate the inert dict entries. Decide, then do one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A theme author can tell from the code whether a ds-* dict entry has runtime effect (either it does, or the inert entries are removed/marked)
- [ ] #2 If (a): the ~35 bundle-parsing test harnesses still pass; if (b): themes.py documents the formula mechanism at the top
<!-- AC:END -->
