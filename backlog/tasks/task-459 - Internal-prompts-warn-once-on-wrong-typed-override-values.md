---
id: TASK-459
title: 'Internal prompts: warn once on wrong-typed override values'
status: Done
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
- [x] #1 A present override value of a non-str/non-table type logs exactly one warning per prompt id and still falls back to the shipped default (never raises)
- [x] #2 A correctly-typed override (str or {text,...} table) and a genuinely-absent override both produce NO new warning
- [x] #3 A unit test in Tests/Internal_Prompts/ covers the wrong-typed-value warn-once path
<!-- AC:END -->

## Implementation Notes

resolver.get_internal_prompt warns once (`:type` key) when an override is present but neither str nor dict (e.g. hand-edited int/array); never raises, falls back to default. 3 new resolver tests.
