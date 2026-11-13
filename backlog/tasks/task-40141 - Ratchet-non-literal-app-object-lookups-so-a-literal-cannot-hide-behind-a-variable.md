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

The census's own walk and receiver rule now also follow names that reach the app object:

- **Literals flowing into a lookup count as lookups.** A literal passed for an existing helper's name parameter (at any call site, positional or keyword, with `self`/`cls` skipped) counts in the pins and the private-name rule. So does each item of a literal tuple or list that a `for` loop or comprehension binds to the name, including tuple unpacking. Moving a literal lookup into `_optional(app, "...")` or a loop is therefore no longer a reduction. The five helpers found are `_optional`, `current_app_value`, `_storage_path_value`, `_consume_verified_default_intent` and `_consume_verified_console_intent`.
- **Every non-literal lookup is recorded.** `Tests/Architecture/app_nonliteral_lookups.tsv` holds 47 rows: file, name expression, and the attribute reached. The record is compared as a multiset. A new row fails, naming the file and asking for an explicit service; the message also says to update a row on a pure rename or move. A recorded row no longer found fails until it is removed. Line numbers are left out.
- **Constants resolve per module and through explicit imports.** An UPPER_CASE name (or `self.NAME`) resolves only when every binding of it in its module, or in the module it is imported from, is the same string. Assign, AnnAssign and AugAssign all count, and a binding that is not a string makes the row `-`. So a rebinding cannot hide a private name, and a same-named constant in an unrelated module changes nothing.

Evidence:

- **Reproduction (base 1455ceeb96):** a planted `getattr(app, name)`, and a literal rewritten through a constant, both passed the old census; the rewrite read as a literal count of 1 -> 0 and dropped the private name.
- **Re-measure:** 47 non-literal lookups, unchanged from ae9f9e8d21.
- **Pins re-measured:** `<root>` 78 -> 97 (17 `_optional` callers, 2 loop items), Backup_Recovery 20 -> 34 (tuple-unpacked loop in `runtime_maintenance.py`), UI 1261 -> 1290 (8 helper call sites, 21 loop items).
- **Private names:** no new private names, so `app_private_attrs.txt` is unchanged. The seven private attributes reached only through constants appear in the record beside their lookups.
- **Negative controls:**
  - `test_a_literal_rewritten_as_a_name_fails_rather_than_reading_as_a_reduction[constant|parameter]`
  - `test_a_literal_moved_into_a_helper_or_loop_is_still_counted[helper|helper-keyword|method|loop|unpacked-loop]`
  - `test_a_constant_resolves_through_its_module_and_imports_only`
- **Mutation checks:** each of these failed its controls. Guard off: both rewrite cases. Helper call-site counting off: 3 cases plus the `<root>`/UI pins. Loop counting off: 2 cases plus 3 pins. AnnAssign ignored: the constant test.
- **The review's two reproductions on the real tree:** `workflows_screen.py:249` rewritten as `_optional(self.app_instance, "_workflow_database_path")` keeps all 31 nodes green, with the lookup still counted and its private name still seen. `CONSOLE_RUNTIME_ATTR: str = "_console_hidden_runtime"` added under `Chat/console_runtime.py:189` fails the record test.
- **Lane:** the guard is in the same file, so it runs in `pr-fast-lane-arch` (`.github/workflows/derived-artifacts.yml`, pinned by `ARCH_REVIEW_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`).
- **Scan cost:** the scan takes about +35% CPU, about 14 s -> 19 s at load 50. The scopes come from the main walk rather than a parent map, which had cost +50%.
- **Known ceiling (a `ponytail:` comment):** a name taken from a dict, a module-level tuple, a nested loop or a call's return value is not followed, so its row reaches `-` and a literal added there is not counted. Helper call sites match by callee name, so an unrelated same-named call would over-count.
- **Files:**
  - `Tests/Architecture/test_app_string_lookup_census.py`
  - `Tests/Architecture/app_lookup_census.tsv`
  - `Tests/Architecture/app_nonliteral_lookups.tsv` (new)
  - `tldw_chatbook/app_services.py` (the `_optional` docstring)
- Verified 2026-11-13 on base 1455ceeb96 (review round). The census file passed 22 nodes at base and passes 31 at head. `./scripts/preflight.sh` passed. Serial `Tests/Architecture`: 1357 passed, 3 failed, and the same 3 fail at base: `test_persistent_diagnostic_inventory::test_reviewed_diagnostic_changes_are_metadata_only` (agent_service.py) and `test_timer_path_static_update_inventory` x2 (project_index_binding_card.py).
