---
id: TASK-41651
title: "A spawned child's native tool round fails on a provider-continuation endpoint, with capture on or off"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- agents
- trace
priority: medium
references:
- tldw_chatbook/Agents/agent_service.py
dependencies:
- TASK-41310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** While verifying TASK-41310, a spawned child that made a native tool call against an endpoint that uses provider continuations failed before any trace was written. It fails the same way with capture off, so the trace work is not the cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A spawned child completes a native tool round on a provider-continuation endpoint
- [x] #2 A test reproduces the failure on dev and passes after the fix, with capture on and off
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (applies ADR-063's existing rule that an ownerless run's continuation is ephemeral; no new policy)

1. Reproduce through the TASK-41310 harness (real controller, store, bridge, gateway, trace runtime; socket double): the child's first reply is a native `tool_calls` call, Capture On and Off. Show it red on dev.
2. Find which `_continuation_error` site fires and why.
3. Fix where every child launch path (spawn, send_to_agent resume, workflow child) routes through: `AgentService._run_one`.
4. Prove the child's checkpoint reaches the provider (not dropped) and never reaches the host's persistence or trace custody; mutation-check each part of the fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** Every child launch path (spawn, `send_to_agent` resume, workflow child) handed the child the parent's continuation durability (`persistent`) and no owner. A child has no assistant row: the Console store persists only a primary's continuation, and a persistent context without an owner fails `_valid_continuation_context`. So the child's first native tool batch stopped at `_phase_persist_tool_batch` with "Provider continuation could not be persisted", before any provider call was traced and whatever the capture mode. Present since `11bd94dbbf` (2026-08-12, the native tool lifecycle barriers). With durability fixed alone, the child's follow-up then failed in `make_call_model` ("Active provider continuation requires an owner"): its in-memory checkpoint needs an owner id and the owner key the Console adapter reads.

**Fix** (`AgentService._run_one`, the one place all child launches meet): a non-primary run gets `continuation_durability="ephemeral"`, owner id = its own run id, owner key = `AgentService.child_continuation_owner_key`, which `_prepare_native_reply` passes (the adapter's key, or the canonical one when that is none; see the second configuration below). The host's `persist_provider_continuation`, trace `acknowledge_provider_continuation` and `expand_provider_continuation` are wired for the primary only: a child's events would otherwise reach the shared adapter and reset the primary's live trace custody (`_live_native_custody`, `_live_native_persisted_event`) from another thread. This matches the runtime's existing rule (`test_cycle_4b_persistent_missing_owner_stops_but_ephemeral_is_non_resumable`: an ownerless subagent runs ephemeral, non-resumable). The checkpoint is kept, not dropped, so providers that need the round's reasoning replayed (`hosted_provider_engine._apply_continuations`) still get it.

**Evidence** (`Tests/Chat/test_console_spawn_subagent_trace_capture.py::test_child_native_tool_round_completes[capture_on|capture_off]`, the TASK-41310 harness: real controller, store, bridge, gateway, trace runtime; socket double; the child answers its first request with a native `tool_calls` call):
- Red on dev d0486978dd (product files checked out, test kept): both fail, child run `error` with "Provider continuation could not be persisted"; the other 10 tests in the file pass.
- Green after: the child sends `[system,user]` then `[system,user,assistant,tool]`, run `done`; its follow-up carries one `active` checkpoint to the provider handler; the host store sees only `primary` continuation events; under Capture On both child calls are `complete` in its own linked segment.
- Mutations (each fails both cases; rerun alone, not while editing): no durability override ("could not be persisted"), no owner id or no owner key ("requires an owner"), host callbacks wired for children (store sees `fleet` events).
- File (with TASK-41652's test): 16 passed at `-n 4`; serially alone 16 passed in 112 s (load 68-77).
- Covering set (20 files: Agents service/continuation/fleet/redirect/code-review context, Plugins continuation, Chat agent bridge, provider continuation, Kimi/ZAI native tools, stop-mid-tool, dispatch handoff, crash recovery, named agent model, default send capture, fleet wake, engine swap, group merge, this file, mix-in journey) at `-n 4`: head 7 failed / 928 passed, base d0486978dd 7 failed / 922 passed, the same 7 nodes (5 `test_console_agent_bridge` usage/lifecycle, 2 `test_console_named_agent_model` `RecoveryRequired('raw_source_selection_changed')`); the 6 extra passes are the new tests.
- `Tests/Architecture` serially: 4 failed / 1475 passed. Three also fail on base (`test_persistent_diagnostic_inventory::test_reviewed_diagnostic_changes_are_metadata_only`, `test_timer_path_static_update_inventory::{test_no_timer_path_update_defaults_to_layout_true,test_clock_root_set_is_pinned}`); the fourth, `test_task_15103_review_ledger_canonical_schema_and_arithmetic`, hit pytest-timeout (300 s) under load and passes alone on head (90 s).
- `./scripts/preflight.sh`: all checks passed.

**Files:** `tldw_chatbook/Agents/agent_service.py`, `tldw_chatbook/Chat/console_agent_preparation.py`, `Tests/Chat/test_console_spawn_subagent_trace_capture.py`, `Tests/Architecture/test_module_size_ratchet.py` (agent_service 7680 -> 7692, +12), `Docs/Architecture/agent-runtime.md`, `Docs/Architecture/llm-providers.md`.

**Same AC, second configuration (coordinator review 2026-11-18): the parent's endpoint has no continuation protocol.** Reproduced with the parent on `openai` (legacy handler, no protocol) and the child routed by an `AgentDefinition` (`provider="custom-ep:spawntrace"`) to the engine endpoint: `test_child_native_round_when_the_parent_endpoint_has_no_continuations[capture_on|capture_off]`, red on `85f8571c0f` with "Active provider continuation requires an owner". Three parent-derived values blocked it, each fixed in one place:
- The child's owner key came from the parent (none): `AgentService` now takes `child_continuation_owner_key` (renamed from `continuation_owner_key`, used only for children); Console passes the adapter's key, or `CONTINUATION_OWNER_KEY` when that is none.
- The adapter matched groups only by its own key (none): `chat_call_impl` falls back to `CONTINUATION_OWNER_KEY` for a sub-agent request ("Every continuation group must attach to one request owner" otherwise). The primary's key is unchanged.
- `prepare_provider_continuation_request` required the parent's continuation target, so the child's checkpoint never reached the gateway and its final turn returned no continuation ("could not be persisted"). It is now just "the gateway can prepare requests": a primary without a target has no owner key and may hold no sidecar groups (`run_turn` refuses a sidecar without a target), so its requests never carry groups and are unchanged.
- Green after, Capture On and Off: the child sends both requests, its follow-up carries one `active` checkpoint, run `done`, under Capture On both child calls are `complete` in its own segment. Mutations (each fails exactly those two cases): no canonical default in preparation ("requires an owner"), no adapter fallback ("must attach to one request owner"), flag tied to the target again ("could not be persisted").
- Re-run after this change: the file 18 passed at `-n 4`; the 20-file covering set plus `test_module_size_ratchet`, `test_agents_chat_boundary` and `test_code_identity_ratchet` at `-n 4`: 7 failed / 1161 passed, the same 7 nodes as base d0486978dd.
- Files: `tldw_chatbook/Chat/console_streaming_request.py` (+3), `tldw_chatbook/Chat/console_agent_preparation.py`, `tldw_chatbook/Agents/agent_service.py` (rename only, size unchanged), the test file (`parent_openai` harness option), `Docs/Architecture/agent-runtime.md`.
**Review round (2026-11-18, four Minors):**
- A host that prepares continuation requests but passes no `child_continuation_owner_key` would only fail later, at a child's second request ("requires an owner"). `AgentService.__init__` now refuses that construction (`ValueError`). Rejected alternative: one Agents-owned child key the adapter always reads for sub-agents. It broke `test_console_agent_bridge.py::test_builtin_child_binds_its_own_native_continuation_under_custom_parent` and `test_code_review_context.py::test_gateway_admission_counts_private_continuation_and_effective_fallback`, which pin that a child's groups use the adapter's own key. Test: `test_a_service_that_prepares_continuations_needs_a_child_owner_key`. Three direct constructions that set the flag now pass the key (`Tests/Agents/test_agent_service.py`, `Tests/Agents/test_code_review_context.py`, `Tests/Chat/test_kimi_zai_provider_contract.py`).
- Accepted: a child's native round on a continuation endpoint now records its model step as "Ephemeral tool continuation is non-resumable." instead of the model text (`agent_runtime.py`, the same summary an ephemeral primary already records).
- `Docs/Architecture/llm-providers.md`: the child clause follows the pre-upgrade parenthetical and names `AgentService._run_one`.
- Tests record each applied checkpoint's call ids: the child's follow-up carries only `call_child_1`, and every primary request carries only its own `call_spawn_1` checkpoint (present in the same-endpoint case).
<!-- SECTION:NOTES:END -->
