---
id: TASK-41643
title: "Console received-route wiring still checks collaborators by code identity"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- security
- architecture
priority: medium
references:
- tldw_chatbook/UI/Console_Modules/wiring.py
- Tests/Architecture/test_code_identity_ratchet.py
dependencies:
- TASK-41319
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** This is a second ADR-356 rule 4 follow-up from TASK-41319. `UI/Console_Modules/wiring.py` still decides the received route by identity:
- `_CONSOLE_RECEIVED_SCREEN_SELECTORS`;
- the `__func__` comparison against `_CONSOLE_TURN_CONTEXT_BUILDER`;
- the scratch-provider and RAG-reader `__code__` checks;
- `_CONSOLE_RECEIVED_QUEUE_METHOD`;
- `composer_run_controls`' `_CONSOLE_PROMPT_QUEUE_DISPATCH`/`__code__` and `_CONSOLE_PREPARING_SESSION_READERS`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The received route decides from registrations or runtime-owned state, not code identity
- [x] #2 Wrapped collaborators behave as stock; foreign ones are refused
- [x] #3 The code-identity ratchet is lowered by the removed reads
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR-356 rule 4 already decides this; the change applies it to the received route the way TASK-41319 did for Send preparation (registration + per-instance override refusal).

Per site (`W` = `UI/Console_Modules/wiring.py::receive_console_visible_intent`, `C` = `UI/Console_Modules/composer_run_controls.py`):

| Site | Decision |
|---|---|
| W `_CONSOLE_RECEIVED_SCREEN_SELECTORS` (`__func__`/`__code__`) | REPLACE: the constant becomes method names; the screen must resolve each to its own bound method with no per-instance override (new `own_methods` beside `stock_native_methods` in `Chat/console_received_dispatch.py`) |
| W `__func__` vs `_CONSOLE_TURN_CONTEXT_BUILDER` | REPLACE: `own_methods(owner, ...)` (the owner's type is already exact) |
| W scratch provider `__code__` | REPLACE: the slot is already in `_console_received_live_adapters` (what the wiring registered); keep partial/args/keywords, drop func/code |
| W RAG readers `__code__` (`_CONSOLE_RECEIVED_RAG_READERS`) | REPLACE: delete; the received route and the stock accessor both call the current chat_screen readers, so a wrapped reader gives both the same answer. The accessor slots stay registered and must be the stock accessors |
| W `_CONSOLE_RECEIVED_QUEUE_METHOD` | REPLACE: `own_methods(controller, ("queue_prompt",))` |
| C `_CONSOLE_PROMPT_QUEUE_DISPATCH`/`__func__`/`__code__` | REPLACE: exact queue type + `own_methods(queue, ("dispatch",))`; the adapters stay registered |
| C `_CONSOLE_PREPARING_SESSION_READERS` (+ the context-provider `__func__` and the RAG loop) | REPLACE: names + `own_methods`; the context provider must `==` the session's bound builder |

Steps: 1) a new UI test drives the real received route with each collaborator wrapped by `functools.wraps` (red on base where it was declined) and with a per-instance/foreign callback (refused); 2) apply the table, delete what becomes unused (`_CONSOLE_RECEIVED_RAG_READERS`), keep definitions other routes still read (`_STOCK_CONSOLE_SCRATCH_SNAPSHOT`, `_CONSOLE_TURN_CONTEXT_BUILDER`, `_CONSOLE_RECEIVED_QUEUE_METHOD` for turn_admission/prompt_queue/received_dispatch); 3) retarget tests that pin a code-identity decline (`test_console_preparing_current_sources.py` session rows, `test_screen_fallback_preserves_replaced_class_dispatch_abi`) to per-instance state changes; 4) lower the ratchet by the measured removed reads; covering files base vs head, `Tests/Architecture` serially, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The received Send route and the two composer seams that feed it now decide from the screen's live-adapter registration and from per-instance state, not code identity (ADR-356 rule 4). No ADR exception was needed and no ADR changed.

- **Helper.** `own_methods(owner, names)` (beside `stock_native_methods` in `Chat/console_received_dispatch.py`): each name must resolve to a method bound to `owner` and must not sit in `vars(owner)`. A `functools.wraps` class method passes; a callback put on that one object does not.
- **Received route (`wiring.receive_console_visible_intent`).** Screen selectors and the session's turn-context builder go through `own_methods`; `_CONSOLE_RECEIVED_SCREEN_SELECTORS` is now a tuple of names. The scratch slot drops its func/code check: `_console_received_live_adapters` (checked just above) already holds it to the partial the wiring registered. The `_CONSOLE_RECEIVED_RAG_READERS` loop is deleted with the constant: the route and the stock accessor both call the current chat_screen readers, so a wrapped reader answers both the same; the accessor slots must still be the stock accessors. `queue_prompt` uses `own_methods`.
- **Composer (`composer_run_controls`).** The fallback hands the pressed inputs to a stock-typed queue whose `dispatch` is not overridden on the instance (was `_CONSOLE_PROMPT_QUEUE_DISPATCH` `__func__`/`__code__`). The Preparing poll checks the selectors and `_CONSOLE_PREPARING_SESSION_READERS` (now names) with `own_methods`, drops the RAG loop, and compares the controller's turn-context provider with `==` against the session's bound builder.
- **Consequence.** A replaced ChatScreen class method is now treated as stock; the route does not call the selectors, so a test that needs a custom gate puts it on the screen instance (`test_a_refused_acknowledged_enter_repaints_its_refusal` retargeted). A builder wrapped after the chat controller was built drops only the Preparing display shortcut (the `==` check) to the FULL poll; Send is unaffected.
- **Tests.** New `Tests/UI/test_console_received_route_wrapped_collaborators.py`: the real route with a wrapped screen selector, turn-context builder, scratch snapshot or RAG scope reader returns a turn like stock (4 cases red on base: "the received route declined"); an instance selector/builder override or a replaced scratch/RAG slot is refused and the foreign callback never runs; `own_methods` unit. `test_screen_fallback_gives_a_wrapped_class_dispatch_the_pressed_inputs` (red on base: no `_captured_inputs`). Retargeted (D2-style): the ABI control now overrides `dispatch` on the queue instance (renamed `..._preserves_instance_dispatch_override_abi`); in `test_preparing_defining_sources_decline_changed_callbacks` the session rows assert a wrapped class method still qualifies and an instance override still declines (the screen/controller rows keep their code mutations, out of scope); two tests switch `stock_native_methods(screen, selectors)` to `own_methods`.
- **Queue site.** The UI half of the `queue_prompt` check is covered through the shared `own_methods` unit test; the runtime half by the review-round transitions test below.
- **Mutation.** Dropping `name not in fields` from `own_methods` fails the unit test and the `instance_selector`/`instance_turn_context_builder` cases.
- **Ratchet.** `test_code_identity_ratchet.py` PINS: `__func__` -5, `__code__` -10, `__defaults__` -1, `__kwdefaults__` -2 (wiring.py, composer_run_controls.py, chat_screen.py, session.py). Chat ratchet unchanged.
- **Review round (2026-11-18).** (1) The route admitted a wrapped `queue_prompt` but the runtime's received queue admission (`console_received_dispatch._run_received_intent_bound`) and the captured-input queue admission (`prompt_queue` `queue_kwargs`) still compared code, so every received Send while a turn was live ended in `RecoveryRequired`; both now use `own_methods`, and the unread `_CONSOLE_RECEIVED_QUEUE_METHOD` and `_CONSOLE_PROMPT_QUEUE_DISPATCH` are deleted. New `test_wrapped_class_queue_prompt_queues_the_received_turn_like_stock` (`Tests/Chat/test_console_received_intent_transitions.py`). (2) The scratch slot must be registered for the current `screen._session` (a swapped-in same-class copy was admitted), and the route requires `type(screen) is ChatScreen` (a subclass's own gate was never called); the Preparing poll already refuses a subclass in `poll_source`. New foreign cases `swapped_session_scratch`, `subclass_screen_gate`. (3) `own_methods` treats an object without an instance dict as having no override (it raised `TypeError`), and the queue check requires an exact `ConsoleChatController`. (4) The Preparing poll compares the turn-context provider only when it is a bound method (a foreign object's `__eq__`/`__ne__` ran). Each new test was red on 33de5eaab and is green after. Ratchet: `__func__` -1, `__code__` -3 more; Chat ratchet `__code__` 49 -> 47. A class-level replacement of a ChatScreen selector is still treated as stock by design (ADR-356 rule 4).
- **Left (follow-ups).** The mounted-capture path in `turn_admission.py` (`stock_mounted_capture`, `precapture`, `stock_rag_depth_current`, `is_stock_session_provider`) and the Preparing screen/controller readers still compare code; `stock_native_methods` still does for its other callers.
- Files: `Chat/console_received_dispatch.py`, `Chat/console_chat_controller.py`, `UI/Console_Modules/wiring.py`, `UI/Console_Modules/composer_run_controls.py`, `UI/Console_Modules/prompt_queue.py` (queue admission + dead record only), `UI/Console_Modules/session.py`, `UI/Screens/chat_screen.py`; tests above plus `test_console_received_intent_feedback.py`, `test_console_staged_evidence_sessions.py`, `test_console_send_acknowledgement.py`; `Tests/Architecture/test_code_identity_ratchet.py`.
- **New-test evidence.** `test_console_received_route_wrapped_collaborators.py` + the wrapped-dispatch control + the retargeted preparing test against base product: 7 failed (4 wrapped route cases "declined", `own_methods` import, wrapped dispatch, wrapped capture-selection row), 6 passed (stock, 4 foreign, instance ABI); head 13 passed. Serial: 10 passed in 74 s at load ~50-68 (alone).
- **Evidence (2026-11-18, load 50-90, base = detached dev d0486978dd).** Covering fast set (17 files incl. both new ones, `-n 4`): base 28 failed/155 passed, head 29 failed/176 passed; the one head-only node was `test_a_refused_acknowledged_enter_repaints_its_refusal` (a class-level foreign gate, retargeted above; passes alone). Slow skill set (genuine capture, stock-owned, config rebuild, context owned, managed acceptance, `-n 4`): base 2 failed (`send_enter`/`send_button`, as TASK-41319 recorded)/28 passed; head after the test-API updates: the same 2 fail, `custom_affinity`/context-owned/`shutdown` pass on rerun (`-n 3`; `shutdown` had hit `sqlite_retirement_incomplete:AgentRuns` once under `-n 4`). Wider net (77 `Tests/UI/test_console_*` send/received/queue/skill/preparing/composer files + 6 more, `-n 4`): base 288 failed/1142 passed, head 290 failed/1161 passed (200 of the shared failures are `RecoveryRequired: raw_source_selection_changed` in `build_test_app_config`, the fixture's config participant, same on both). Head-only nodes: 8; rerun at `-n 4` 5 passed; `test_enter_after_a_tab_click_during_a_sync_pass_sends_to_the_clicked_chat` and `test_console_blocked_send_retains_exact_recovery_after_runtime_custody` fail on base too; `test_an_inspector_opened_after_chat_switches_matches_one_open_throughout` passes alone on head and base. `Tests/Architecture` serially: 4 failed/1475 passed; rerun of the 4 on base and head: the same 3 fail on both (`test_reviewed_diagnostic_changes_are_metadata_only`, two timer-path inventory tests), the 4th passed. `./scripts/preflight.sh` passed. Ruff: no new findings in touched files (base vs head diff empty).
<!-- SECTION:NOTES:END -->
