---
id: TASK-40141
title: Ratchet non-literal app-object lookups so a literal cannot hide behind a variable
status: Done
created_date: 2026-11-10 20:10
labels:
- architecture
- tests
- follow-up
priority: low
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3490#issuecomment-6777451323
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Architecture/app_lookup_census.tsv
- Tests/Architecture/app_private_attrs.txt
- tldw_chatbook/UI/Screens/settings_screen.py
dependencies: []
updated_date: 2026-11-12 15:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36942.4 ratchets string-named attribute lookups on the app object and fails any new underscore-prefixed name that is not in a frozen allowance. It counts only lookups whose name is a string literal; lookups whose name comes from a variable, a constant or an expression are left out of both the count and the private-name rule, a gap its docstring records (45 such lookups at the time).

That gap lets a change look like progress while moving a lookup out of view. Dev commit eced473b58 (TASK-39133) moved one literal lookup of `_vllm_connection_owner` in `settings_screen.py` behind a parameter, shared with the llama.cpp owner. The counted UI lookups fell from 1262 to 1261 with no check pointing at the cause; both owner names stay visible to the private-name rule only because other modules still look them up literally. The pin was not lowered, so the PR Fast Lane went red for every PR on dev until PR #3492 lowered it to 1261.

The blind spot is already real: seven private app attributes are reached only through module constants, are not in the allowance, and so are invisible to the private-name rule today (`_console_fleet_unseen_revision`, `_console_pending_interrupts`, `_console_reaction_preview_coordinator`, `_library_last_export_directory`, `_rag_service_generation`, `_rag_service_stamp`, `_tldw_library_ingest_options_cache`). Measured on dev ae9f9e8d21 with the census's own receiver rule, and unchanged on 4a9c4c1132: 47 non-literal lookups (UI 24, Event_Handlers 9, Chat 6, Backup_Recovery 3, `<root>` 2, RAG_Search 2, Scheduling 1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A new non-literal lookup on an app-like receiver fails an architecture test that names the file and says how to reach the service explicitly instead.
- [x] #2 Rewriting a counted literal lookup as a non-literal one fails, rather than reading as a reduction of the literal count.
- [x] #3 Every non-literal lookup that exists when the guard lands is recorded where a reviewer can see it, and that record can only shrink; a lookup that goes away fails until it is removed from the record.
- [x] #4 Each private app attribute reached only non-literally is either recorded together with the lookup that reaches it or reached explicitly; the frozen private-name allowance does not grow.
- [x] #5 The guard runs in the same required lane as the existing census.
- [x] #6 A negative control shows the guard fails on a planted non-literal lookup.
<!-- AC:END -->

## Implementation Plan

ADR required: no. This extends an existing architecture test (TASK-36942.4) with a shrink-only record of the same shape as its census; no product behaviour or design choice changes.

Re-measured on base 1455ceeb96: still 47 non-literal lookups (UI 24, Event_Handlers 9, Chat 6, Backup_Recovery 3, `<root>` 2, RAG_Search 2, Scheduling 1), and the same seven private attributes reached only through module constants. Every lookup whose name comes from a parameter, loop variable or dict reaches a public name or an allowed private one (`_vllm_connection_owner`, `_llamacpp_connection_owner`).

1. Reproduce: a planted `getattr(app, name)`, and a literal lookup rewritten through a constant, both pass the current census (the rewrite reads as a literal-count reduction and drops out of the private-name rule).
2. Extend `_scan` in `Tests/Architecture/test_app_string_lookup_census.py` (same walk, same receiver rule) to collect every non-literal lookup as (file, name expression, reached name), resolving an UPPER_CASE constant with one package-wide string value to the attribute it names.
3. Record the 47 lookups in `Tests/Architecture/app_nonliteral_lookups.tsv`; a new row fails naming the file and how to reach the service explicitly; a recorded row no longer found fails until removed. The seven private attributes appear in the record beside their lookups; `app_private_attrs.txt` is unchanged.
4. Negative controls: planted lookups (parameter, and a literal rewritten through a constant) fail the guard while the literal census reads a reduction.
5. Same file, so it runs in the PR Fast Lane (architecture-review tests) job that already runs the census.

## Implementation Notes

The census's own walk and receiver rule now also collect every non-literal `getattr`/`hasattr` on an app-like receiver, recorded in `Tests/Architecture/app_nonliteral_lookups.tsv` as file, name expression, and the attribute it reaches (`-` when the name is not a constant). The record is compared as a multiset: a new row fails, naming the file and asking for an explicit service instead of a new row; a recorded row no longer found fails until it is removed. Line numbers are left out so unrelated edits do not churn it.

- Reproduction (base 1455ceeb96): a planted `getattr(app, name)`, and a literal `getattr(app, "_planted_private")` rewritten through a constant, both pass the old census; the rewrite reads as a literal count of 1 -> 0 and the private name leaves the private-name rule.
- Re-measured on the base: 47 lookups, unchanged from ae9f9e8d21. A literal rewritten in place (the eced473b58 shape) is a new row, so it fails `test_no_new_nonliteral_app_lookups` beside the pin's "lower it" message.
- Private names: an UPPER_CASE name (or `self.NAME`) with one module- or class-level string value package-wide resolves to that value. The seven private attributes reached only through constants appear in the record beside their lookups; `app_private_attrs.txt` is unchanged. Every lookup through a parameter, loop variable or dict was read by hand: each reaches a public name or `_vllm_connection_owner`/`_llamacpp_connection_owner`, which are already allowed.
- Known ceiling (a `ponytail:` comment in the test): a literal passed into an existing helper's parameter still reaches the app unseen, since its row stays `-`. Following call sites would close it.
- Negative control: `test_a_literal_rewritten_as_a_name_fails_rather_than_reading_as_a_reduction[constant|parameter]` plants the rewrite in a temporary package. It asserts the literal census reads a reduction and the guard fails naming `UI/planted.py`, and for the constant case the private name it reaches. With the guard disabled both cases fail (mutation check). A real planted file in `tldw_chatbook/UI/` failed the guard with its path, and a bogus record row failed with "remove those rows".
- Lane: the guard lives in the same file, so it runs in `pr-fast-lane-arch` (`.github/workflows/derived-artifacts.yml`, pinned by `ARCH_REVIEW_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`) with no workflow change.
- Files: `Tests/Architecture/test_app_string_lookup_census.py`, `Tests/Architecture/app_nonliteral_lookups.tsv` (new), and `tldw_chatbook/app_services.py` (docstring of `_optional`, which said its reads stay out of the census).
- Verified 2026-11-13 on base 1455ceeb96. The census file passed 22 nodes at base and passes 25 at head. `./scripts/preflight.sh` passed. Serial `Tests/Architecture`: 1351 passed, 3 failed, and the same 3 fail at base: `test_persistent_diagnostic_inventory::test_reviewed_diagnostic_changes_are_metadata_only` (agent_service.py) and `test_timer_path_static_update_inventory` x2 (project_index_binding_card.py).
