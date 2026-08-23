---
id: TASK-21106
title: >-
  Move Actor_Packs recovery out of app __init__ - it also crashes the test app factory and disarms the CSS cliff guard
status: Done
assignee:
  - '@claude'
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

- [x] Actor-pack recovery runs at Personas-surface mount (or an equivalent pre-surface seam), not inside TldwCli.__init__; recovery semantics preserved
- [x] The character_persona_schemas import is TYPE_CHECKING/function-local; a sys.modules assertion pins it off the app import path
- [x] `test_full_destination_tour_stays_under_the_parse_cache_cliff` runs green on dev again (guard re-armed), with its measured source count recorded in the task

## Implementation Plan

Consumer census (who reads recovery-affected state):

- Recovery reconciles `actor_pack_persona_intents` + `actor_portable_identities`
  (read ONLY by `PersonaActorPackCoordinator` / `ActorPackCreationService`; the
  only external holder is `personas_screen.py:6593` via
  `app.actor_pack_creation_service`) and can compensate half-applied writes in
  the Persona JSON profile store (read via `LocalCharacterPersonaService` by the
  Personas screen / CCP handlers, the Persona Buddy controller, and Console
  persona-session paths through `character_persona_scope_service`).
  `VisualIdentityRepository.get_active_actor_pack` (Console avatar path) reads
  `visual_identity_*` tables, which recovery never touches.

Steps:

1. Coordinator: add an idempotent, thread-safe `ensure_recovered()`
   (threading.Lock + attempted flag; caches the result, records the failure
   category like `__init__` did, guarantees at-most-once even on unexpected
   exceptions). `create_persona()` calls it first so a mutation can never
   run against unreconciled intents regardless of scheduling (it already runs
   on a `_drain_to_thread` worker thread).
2. app.py: `_wire_character_persona_services` stops calling `recover()`
   (construction becomes DB-free); add `ensure_actor_pack_recovery()` — safe
   from any thread, early-returns when `chachanotes_db` is None (the test
   harness), delegates to the coordinator and maps the outcome onto
   `actor_pack_recovery_error` with the same log lines as before.
3. Kick recovery on a background thread from `_schedule_deferred_startup_work`
   (the sanctioned post-first-paint seam) so on every real boot it completes
   ahead of any user-driven persona interaction (Console attach, Buddy).
4. Hard gate the primary surface: Personas screen `_load_after_mount` awaits
   `asyncio.to_thread(app.ensure_actor_pack_recovery)` before any
   library/persona read. Once-per-app-session holds because the guard lives on
   the coordinator (screens are never cached; re-mounts re-call the idempotent
   ensure).
5. creation.py: make the `character_persona_schemas` import function-local in
   the two `model_validate` sites (same pattern as
   `local_character_persona_service.py`); the existing fresh-subprocess
   allowlist test `Tests/Utils/test_tldw_api_schema_deferral.py::
   test_app_import_schema_submodule_set_is_within_allowlist` is the sys.modules
   pin (currently RED with exactly this module leaking) — strengthen it with an
   explicit `character_persona_schemas not in loaded` assertion.
6. Tests: new unit coverage for `ensure_recovered` once-semantics +
   create-persona gating + app-level ensure wiring; re-run
   `Tests/UI/test_widget_css_consolidation.py` (record the cliff test's
   measured source count), `Tests/Performance/test_ui_latency_guardrails.py`,
   `Tests/Actor_Packs`, `Tests/Utils/test_tldw_api_schema_deferral.py`,
   `Tests/UI/test_screen_navigation.py` (99 baseline reds from the same
   factory crash), plus a full `--collect-only` sweep.

Baseline on `0f9638cef` (logs in `test-logs/`):

- `Tests/UI/test_widget_css_consolidation.py`: **1 failed** (the cliff test;
  `AttributeError: 'NoneType' object has no attribute 'execute_query'` at
  `repository.py:276` via `app.py:6612 recover()`), 30 passed.
- `Tests/UI/test_screen_navigation.py`: **99 failed, 31 passed** — same crash.
- `Tests/Utils/test_tldw_api_schema_deferral.py`: **1 failed** (allowlist test,
  leaking exactly `tldw_chatbook.tldw_api.character_persona_schemas`), 3 passed.
- `Tests/Performance/test_ui_latency_guardrails.py`: 2 passed.
- `Tests/Actor_Packs`: 94 passed.

## Implementation Notes

Recovery moved off the construction path onto a once-per-app-session gate; the
schema import went function-local; the CSS cliff guard is re-armed.

**Approach.**

- `PersonaActorPackCoordinator.ensure_recovered()` (persona_coordinator.py):
  idempotent, thread-safe (threading.Lock + attempted flag), caches the result,
  records `actor_pack_recovery_failed` instead of raising (matching how
  `__init__` always absorbed it), and consumes its single attempt even on an
  unexpected exception so a broken store cannot retry-loop. `create_persona()`
  now calls it first, so a mutation can never be admitted against unreconciled
  intents regardless of caller ordering (it already runs on a worker thread via
  `_drain_to_thread`).
- app.py: `_wire_character_persona_services` no longer calls `recover()`
  (construction is DB-free — this is what un-crashes `_build_test_app`, whose
  ChaChaNotes DB is None). New `ensure_actor_pack_recovery()` maps the
  coordinator outcome onto `actor_pack_recovery_error` with the exact
  `__init__`-era log lines, and skips when there is no ChaChaNotes DB.
  `_schedule_deferred_startup_work` kicks it on a thread worker right after
  first paint, ahead of any user-driven persona interaction.
- Personas surface (`personas_screen.py::_load_after_mount`): awaits
  `asyncio.to_thread(app.ensure_actor_pack_recovery)` before the first
  library/persona read. Screens are never cached, so the once-guard living on
  the coordinator (not the screen) is what keeps this once-per-session.
- Consumer census (who reads recovery-affected state): the intents/identities
  tables are read only by the coordinator/creation service (sole external
  holder: personas_screen.py:6593); the persona JSON store is read by the
  Personas screen/CCP handlers, Persona Buddy, and Console persona paths via
  `character_persona_scope_service`. Console's avatar path reads
  `visual_identity_*` tables, which recovery never touches. Personas gets the
  hard await-gate; mutations get the coordinator self-gate; Console reads and
  Buddy (both user-driven, post-first-paint) are covered by the deferred kick.
- `Actor_Packs/creation.py`: the `character_persona_schemas` import (79
  pydantic models, ~34 ms) moved function-local to its two `model_validate`
  sites (the `local_character_persona_service.py` pattern). The existing
  fresh-subprocess allowlist test is the sys.modules pin and was RED with
  exactly this module; strengthened with an explicit
  `character_persona_schemas not in loaded` assertion.

**Tests** (base `0f9638cef` → after; logs in `test-logs/`):

- `Tests/UI/test_widget_css_consolidation.py`: 1 failed/30 passed →
  **31 passed**. The cliff test measured **47 live stylesheet sources** after
  the full 13-destination tour (cap: 64-cache with the guard asserting < 56).
- `Tests/UI/test_screen_navigation.py`: 99 failed/31 passed → **130 passed**
  (all 99 were the same `recover()` AttributeError through `_build_test_app`).
- `Tests/Utils/test_tldw_api_schema_deferral.py`: 1 failed/3 passed →
  **4 passed**.
- `Tests/Actor_Packs`: 94 → **98 passed** (4 new coordinator tests; the
  create-persona gate test was A/B-verified red without the `ensure_recovered`
  call in `create_persona`).
- New `Tests/UI/test_actor_pack_recovery_seam.py`: **8 passed** — construction
  runs no recovery; ensure skips without a DB; outcome mapping; real-SQLite
  recovery through the app method; mounted real app proves the deferred worker
  fires; mounted Personas harness proves recovery lands before the first
  library read (A/B-verified red with the gate disabled).
- `Tests/Performance/test_ui_latency_guardrails.py`: 2 passed (unchanged).
- Personas UI suites (`test_personas_workbench.py` +
  `test_actor_pack_creation_workflow.py`): 357 passed;
  `test_character_persona_scope_service.py`: 54 passed;
  `Tests/Architecture/test_actor_pack_boundary.py`: 5 passed.
- Full `pytest Tests --collect-only`: 55041 collected, 29 errors — all
  pre-existing (missing optional deps: numpy/audio/TTS stacks, plus the dev
  red below); none in files this task touched.
- Live verification: real app booted from an isolated sandbox profile,
  navigated to Roleplay via the palette; screen mounted, library loaded, and
  the app log contained zero "Actor Pack recovery" failure lines.

**Updated dev pins that asserted the retired ordering** (both were
source-inspection tests requiring `.recover()` inside the wiring):
`Tests/Actor_Packs/test_actor_pack_creation.py` (now
`test_app_defers_recovery_and_the_mutation_path_self_gates`) and
`Tests/UI/test_console_runtime_ownership.py::test_actor_pack_recovery_precedes_character_persona_surfaces`
(now pins all three new seams).

**Discovered, deliberately not fixed** (pre-existing dev red, out of AC
scope):
`Tests/UI/test_console_runtime_ownership.py::test_app_fences_console_then_drains_buddy_before_profile_teardown`
builds a skeletal `object.__new__(TldwCli)` that never sets
`notes_sync_runtime_owner`, but `_shutdown_app_owned_lifecycles`
(app.py:12178 on this branch) now awaits `_shutdown_notes_sync_runtime()`
first — AttributeError before the console fence, so the 2 s wait times out.
Proven pre-existing: the shutdown chain and that test's fixture set are
byte-identical to HEAD, and none of this task's hunks touch shutdown.
