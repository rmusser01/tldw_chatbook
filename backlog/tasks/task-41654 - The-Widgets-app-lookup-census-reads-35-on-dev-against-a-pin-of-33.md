---
id: TASK-41654
title: "The Widgets app-lookup census reads 35 on dev against a pin of 33"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- architecture
- tests
priority: medium
references:
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Architecture/app_lookup_census.tsv
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** `Tests/Architecture/test_app_string_lookup_census.py::test_app_string_lookups_match_their_pin[Widgets]` fails on dev 45b1949ca3 and later: 35 literal `getattr`/`hasattr` lookups on the app object against a pin of 33. It passed on 5e3c8b2bd4, so the two new lookups arrived with the round-15 merges #3594..#3609. The census says not to raise the pin.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two new lookups reach their service through a typed attribute or explicit accessor
- [x] #2 The Widgets census passes at its existing pin
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (census row only; no rule or design change).

1. Re-measure the Widgets census on current dev (PR #3627 / TASK-41800 may already have replaced the two lookups).
2. If still 35: find the two literal lookups #3594..#3609 added and read them through a typed attribute or explicit accessor. If already 33: confirm the fix meets AC #1, mutation-check it (restore the pre-fix file -> 35), and close with the evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Already fixed on dev before this stream started; no code change here (checked 2026-11-18 on dev d0486978dd).

- **Cause:** #3599 (TASK-41114, 2889f98c89) added `getattr(app, "persona_actor_pack_coordinator", None)` and `getattr(self.app, "actor_pack_recovery_error", None)` to `Widgets/Persona_Widgets/actor_pack_recovery_notice.py`, taking Widgets from 33 to 35.
- **Fix (landed in #3627, TASK-41800, commit 9ee8aaa9e8):** both became plain attribute reads. `TldwCli.__init__` wires both unconditionally in `_wire_character_persona_services` (`app_service_wiring.py`): the coordinator is a `PersonaActorPackCoordinator` (also typed on `ApplicationServices` in `app_services.py`), and `actor_pack_recovery_error` is annotated `str | None`. That meets AC #1 (typed attribute) and AC #2.
- **Evidence:** `Tests/Architecture/test_app_string_lookup_census.py` passed 55 of 55 serially on d0486978dd, alone, with Widgets at its pin of 33.
- **Mutation check:** restoring the pre-fix notice (`9ee8aaa9e8^`) in the worktree and re-running the scan gave Widgets 35. Restoring the current file gave 33.
- **Files:** this task file only.
<!-- SECTION:NOTES:END -->
