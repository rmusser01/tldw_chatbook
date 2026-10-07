# Task 9 provider-await guard diagnosis

Read-only diagnosis at HEAD `d0aba50877540427ff00492b2afcde7c073e63ed`; Task9 BASE `cd15d174553bf77583a3b0894da245cd991bd1fe`. No tests, package imports, source/index/HEAD edits, fetch, or broad review were performed. This ignored SDD note is the only write. Root must record repair scope before implementation.

## Finding and evidence

The guard is accurate. `Tests/Chat/test_console_first_chat_handoff.py:86` rejects direct `await self.provider_gateway.resolve_for_send(...)` outside the existing bounded helper. HEAD contains exactly these two direct awaits:

- `console_chat_controller.py:7538`, `effective_thinking_history_policy_for_session`: argument remains `self._provider_selection_for_session(session_id)`.
- `console_chat_controller.py:20507`, `_build_personal_context_snapshot`: argument remains the supplied `provider_selection`.

Both complete methods are unchanged from Task9 BASE. Shell-extracted method ranges have identical BASE/HEAD SHA256 values: thinking `d54944087c84983f657ee38b70b83d83a9137d829cab2d0193cd42aff000ba67`; snapshot `b5b9347ba1c84d23cec36cf389a91f3260d14bf6f7634e6b3733af8f571e0ded`. The existing `task-9-report.md` records BASE reproduction of the same static failure. This note adds source attribution, not a new execution result.

The gateway protocol (`console_chat_controller.py:3294`) promises readiness resolution, with no deadline. The real gateway (`console_provider_gateway.py:4196`) directly awaits `_resolve_for_send_unclassified`. Its cloud branch (`:4574`) awaits `asyncio.to_thread(get_provider_readiness, ...)` without an aggregate caller deadline. Local HTTP probes have 5-second request timeouts (`:3910`, `:7343`), and optional serving metadata is refreshed in the background; those provisions do not bound the complete protocol call. A hanging protocol implementation therefore wedges either surface. This is a real inherited gateway-wait risk, not an inaccurate test.

Actual callers: ChatScreen awaits the thinking result before opening conversation settings (`chat_screen.py:3354`); `build_context_snapshot` awaits the Personal Context builder for an eligible agent preview (`console_chat_controller.py:20190`).

## Smallest production repair

Route only the two calls through the existing helper:

```python
resolution = await self._resolve_for_send_bounded(
    self._provider_selection_for_session(session_id)
)
```

```python
resolution = await self._resolve_for_send_bounded(provider_selection)
```

Keep selection evaluation inside the existing `try`, with the same expression/object, order, and arguments. Retain every existing exception, readiness, continuation, and preview branch. No gateway API, helper, caller signature, timeout constant, source guard, UI, permission, or owner change is needed.

The helper (`console_chat_controller.py:3974`) uses the existing `PROVIDER_VALIDATION_TIMEOUT_SECONDS = 30.0`, returns `ready=False` and actionable timeout copy, and propagates cancellation. Thinking already returns the normalized saved policy on exception or not-ready; it must neither overwrite that preference nor promote it to Required after failed resolution. Preview already returns `_empty_profile_context_snapshot()` on not-ready or uncertain assembly; it must skip subsequent profile/tool composition and preview planning after timeout. Preserve its existing `getattr(resolution, "ready", True)` default, distinct from thinking's False default. Keep the helper's reason/copy intact; these two read-only callers intentionally retain their existing fallback result contracts.

The bounded seam also records its existing `provider_resolution` stage. No diagnostic statement or new logging payload is introduced. `asyncio.wait_for` preserves the project's cooperative cancellation contract; it does not forcibly terminate a synchronous readiness worker or bound all later snapshot/service work. Those separate spans are outside this repair.

Actual current controller count is 29329 against 29367. The first substitution has equal line count; formatting the second call as one line removes two lines. Projection: **29327/29367, 40 lines headroom**, with unchanged screen and import budgets. This is a prediction; the implementation owner must measure formatted source and run the existing cap node. No cap increase or compression is needed.

ADR required: no.
ADR path: N/A; existing TASK-21145/TASK-21150 define this bounded seam, ADR-090 governs saved versus effective thinking policy, ADR-102 governs immutable profile snapshots, and ADR-220 retains bounded controller orchestration.
Reason: routine repair of two inherited bypasses of the existing deadline contract.

## Exact affected coverage

Existing selectors directly relevant to this repair:

- `Tests/Chat/test_console_first_chat_handoff.py::test_no_unbounded_resolve_for_send_awaits_remain` — retain its scanner and assertion unchanged.
- `Tests/Chat/test_console_first_chat_handoff.py::test_provider_validation_reaches_terminal_state_in_bounded_time` — existing helper timeout/copy contract.
- `Tests/Chat/test_console_context_policy_lifecycle.py::test_effective_thinking_policy_uses_send_continuation_groups` — all three existing parameter rows preserve compatible complete Required, active Exclude, and changed-model Exclude behavior.
- `Tests/Chat/test_console_personal_context_snapshot.py::test_agent_next_send_uses_one_pinned_snapshot_without_double_append` — one exact profile snapshot, single service read, reserved budget, and one inserted block.
- `Tests/Chat/test_console_personal_context_snapshot.py::test_agent_next_send_reserves_the_live_library_schemas` — exact live library provider/authority and reservation.
- `Tests/Chat/test_console_personal_context_snapshot.py::test_agent_next_send_uses_selected_project_root_for_local_schemas` — retained project selection forwarding.

The UI settings tests stub `effective_thinking_history_policy_for_session`; they do not cover the real gateway wait and need no changes for this repair. Planner-only tests do not exercise either newly bounded call. Existing compaction timeout coverage is unchanged and can carry if already qualified; no broader compaction/provider/UI replay is needed solely for these substitutions.

Missing behavioral cases to scope before source work:

1. In the thinking-policy owner, a positively entered hanging gateway with a short controller deadline returns the saved normalized policy; the stored preference stays unchanged and continuation replay is not consulted. Capture the requested session selection to prove the requested session's settings remain the argument source.
2. In the Personal Context snapshot owner, call the actual `_build_personal_context_snapshot` with a non-None service/builder, valid session, callable preview planner, and supplied selection object. A positively entered hanging gateway must receive that exact object and return the empty snapshot within an outer failure bound; downstream profile/tool composition and the preview planner must remain uncalled. Positive prerequisites prevent an early empty-return path from producing a false pass.

Use the existing test profile boundaries and unchanged original assertions. A small parametrized cancellation control for these two actual methods can additionally prove that explicit cancellation after gateway entry propagates `CancelledError`, rather than being converted to a fallback; cancel only after the positive entry witness to avoid confusing an earlier deadline with cancellation. No new sleeps, timeout policy, skip/xfail, config/conftest exception, or guard waiver is proposed.
