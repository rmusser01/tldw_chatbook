---
id: TASK-21106
title: >-
  Move Actor_Packs recovery out of app __init__ - it also crashes the test app factory and disarms the CSS cliff guard
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - test-integrity
  - actor-packs
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21106).

`_wire_character_persona_services` (app.py:6010 -> 6612) calls
`persona_actor_pack_coordinator.recover()` during `TldwCli.__init__`, running synchronous
SQLite on the app thread every boot. Consequence found live: `Tests/UI/app_factory.py`'s
`_build_test_app()` crashes (ChaChaNotes DB unassigned in that harness), which makes
`test_full_destination_tour_stays_under_the_parse_cache_cliff` FAIL on dev - the CSS cliff
guard is disarmed, invisibly, since CI has not completed since June. Additionally
`Actor_Packs/creation.py:17` imports `tldw_api.character_persona_schemas` (79 pydantic models,
~34 ms) eagerly, contradicting the task-285 deferral comment at app.py:7518. The recover()
docstring only requires running before affected surfaces mount - Personas mount satisfies it.

## Acceptance Criteria

- [ ] Actor-pack recovery runs at Personas-surface mount (or an equivalent pre-surface seam), not inside TldwCli.__init__; recovery semantics preserved
- [ ] The character_persona_schemas import is TYPE_CHECKING/function-local; a sys.modules assertion pins it off the app import path
- [ ] `test_full_destination_tour_stays_under_the_parse_cache_cliff` runs green on dev again (guard re-armed), with its measured source count recorded in the task
