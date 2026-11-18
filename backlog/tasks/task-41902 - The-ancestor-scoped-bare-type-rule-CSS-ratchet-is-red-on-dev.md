---
id: TASK-41902
title: "The ancestor-scoped bare type rule CSS ratchet is red on dev"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
- css
priority: medium
references:
- Tests/Performance/test_textual_css_fastpath.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-16 census stream: `test_ancestor_scoped_bare_type_rule_count_is_a_ratchet` reads `270 <= 268` at dev d0486978dd; dev's own Perf Guard push run 44370291559 fails it the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two new ancestor-scoped bare type rules are rewritten or removed, or justified under ADR-097 without raising the ratchet
- [ ] #2 The node passes and the commit that added them is named in the notes
<!-- AC:END -->
