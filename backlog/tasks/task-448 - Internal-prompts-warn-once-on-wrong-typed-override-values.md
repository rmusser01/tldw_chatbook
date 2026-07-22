---
id: TASK-448
title: 'Internal prompts: warn once on wrong-typed override values'
status: To Do
assignee: []
created_date: '2026-07-21 20:05'
labels:
  - internal-prompts
  - polish
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In `tldw_chatbook/Internal_Prompts/resolver.py`, `_extract_text` silently returns None (treated as "no override") when a config override value is neither a str nor a `{text, ...}` table — e.g. a stray TOML int/array from a hand-edit. The never-raises contract is correct, but the silent drop gives no signal that a user's intended override was ignored. Add a warn-once (keyed like the other resolver warnings) when a present-but-wrong-typed override/legacy value is discarded, so misconfiguration is visible in logs. Surfaced as deferred minor m1 during the P1 whole-branch review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A present override value of a non-str/non-table type logs exactly one warning per prompt id and still falls back to the shipped default (never raises)
- [ ] #2 A correctly-typed override (str or {text,...} table) and a genuinely-absent override both produce NO new warning
- [ ] #3 A unit test in Tests/Internal_Prompts/ covers the wrong-typed-value warn-once path
<!-- AC:END -->
