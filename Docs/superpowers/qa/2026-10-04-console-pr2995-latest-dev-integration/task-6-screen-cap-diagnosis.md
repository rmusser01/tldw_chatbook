# Inherited ChatScreen cap and llama.cpp adoption diagnosis

Source remains9ed370f5668d82d9e6a0108835aa87136a57ce74. This is a read-only follow-up to the frozen model-config report; no production repair or extraction has started.

Last compliant first-parent source74965f694f435c590366ec66d671d3506d440ebc is25218 lines/759 method definitions. Pinned057 is25392/764, df2 is25391/764 and the integrated feature is25448/764. Exact [history](task-6-modelconfig-screen-cap-history.json), [eight additions/three removals](task-6-modelconfig-screen-cap-additions.json), and [immutable cap failure](task-6-modelconfig-screen-cap-upstream-red.json) retain the evidence. Count method definitions, including property getter/setter definitions, rather than unique names.

## New inherited defect confirmed before extraction

ADR114 requires Use in Console to adopt a verified llama.cpp provider/model/endpoint for the active session only. Existing positive controls are `Tests/UI/test_llamacpp_consumers.py::test_console_adopts_verified_llamacpp_for_active_session_only` and the actual setup-view Use-in-Console journey. The setup view stages LLAMACPP_CONSOLE and navigates to Chat. ChatScreen mount/resume register only the vLLM consumer; search finds no production call to `consume_pending_llamacpp_console_intent`.

The public llama wrapper passes a typed Llama intent/current-owner callback to `_consume_verified_console_intent`, but that body checks `VllmConsoleIntent`, reads `_vllm_connection_owner`, calls the vLLM predicate and builds vLLM settings. The parameters that should select these boundaries are unused. This is an inherited functional defect, not a safe generic transaction to move under another name.

The [immutable df2 diagnostic](task-6-screen-cap-llama-private-diagnostic.json) calls the real public wrapper and real helper, actual PendingHandoffStore and exact current verified Llama target. Its assertions prove the owner considers the intent current, the call returnsFalse, the mutation boundary is never entered and the claim remains pending. The diagnostic PASSES because it asserts the defect; it does not claim adoption succeeds. Production hash stays exact. Initial standalone probe stopped at missing private config parent; ordinary pytest probe stopped at raw_source_selection_changed. Both setup diagnostics remain retained; the existing private_profile_test helper admitted the final child without guard changes. [Safe logs/XML](task-6-screen-cap-safe-evidence-manifest.json).

The existing `test_llamacpp_console_adoption_failure_restores_session_and_requeues` only checksFalse, unchanged settings and pending claim, so it can pass before its injected post-adoption failure is reached. A repair would need an explicit entered-mutation assertion, a successful real positive adoption and mount/resume dispatch controls.

## Scope decision required

A behavior-preserving extraction cannot be described as fixing this adoption defect. Root must choose and plan the functional correction separately under ADR114/117 and DESIGN§7 before code. Proposed existing-controller placement remains ConsoleSessionController for verified session adoption, with explicit late-bound dependencies and direct consumer updates. Keep HooksController DOM-free; existing wiring owns UI callback adapters. Do not retain screen aliases/descriptors to hide method count, waive ownership assertions, raise caps or accidentally change durable-default/rollback/claim-recovery policy.
