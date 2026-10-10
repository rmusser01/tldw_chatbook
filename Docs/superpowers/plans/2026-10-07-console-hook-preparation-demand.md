# Demand-driven Console hook context preparation

Task: TASK-34563.12. Base: 41e369c1eb09bb82d2892f68e2c8f068341d60d0.

ADR required: yes, existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: implements its demand-driven domain I/O rule while preserving ADR-197 hook consent and lifecycle authority.

## Goal and design

In the existing prepare_hooks_v2 owner, omit a workspace/context capture whose result has no consumer. Keep the original permissions.v2_configuration reconciliation and any applicable native-plugin hook_configuration call. Immediately before _hooks_v2_context_key, return None only when the review is ready, no v2 handlers or invalid admissions exist, native plugin definitions are absent, and this session has no engine, lifecycle entry or configured-owner entry. Recheck the existing disposed/session fence before this early return.

Do not infer absence from enabled=False, empty targets alone, or a closed-but-retained engine. Existing owners continue context comparison, invalidation and rebuilding. Unready/unknown/invalid state keeps existing refusal behavior. No cache, new task system, configuration switch, native wrapper or alternate authority path is needed. Preserve supported signatures and source-bound stock reads. No native lifetime or overall latency completion follows from skipping one unused call.

Older diagnostic whole-method spans were 0.800–1.024 s; they justify investigating this domain but do not isolate removable context cost or establish savings. Qualification uses original call counts and supported behavior on current source. All native checks stay sequential; no full suite or concurrent timing.

## Ownership and execution

- Baseline verification lane owns only Tests/Chat/test_console_hook_preparation_demand.py. Write a focused real-owner test with passive wrappers counting original v2_configuration and _hooks_v2_context_key calls, real private configuration, actual runtime disposal, and no external hook command execution. Cold empty config must reconcile but count zero context entries; include owner-present fallback and unready/invalid or fenced cases. Reuse existing fixtures where their actual cleanup/platform behavior fits.
- Shared preparation lane performs source-only design/test-boundary review. No edits or native runs.
- Controller integration lane owns only the small guard in Chat/console_runtime.py after root observes the original cold-empty call-count RED. Do not reformat unrelated blocks or change deadlines.
- Root owns this plan, task/report, one-at-a-time native checks and integration commit.

## Checks

- [x] Review the exact owner absence/permission boundary before product edits.
- [x] Run the cold-empty original-call-count test on unchanged base and observe meaningful RED.
- [x] Add only the demand guard; run focused new checks plus existing absent-v2/invalid and permission-epoch controls.
- [x] Run one actual mounted Send control after implementation; inspect source/current process-retirement receipts and scoped static deltas.
- [x] Record measured operation counts and qualifications, review diff and commit only tested scope.

Existing controls: Tests/Chat/test_hooks_v2_lifecycle.py::test_absent_v2_control_does_not_treat_invalid_config_as_absence; Tests/Agents/test_hook_permissions.py::{test_v2_review_is_persistent_and_legacy_targets_remain_separate,test_changed_v2_policy_and_revoke_retire_exact_epoch,test_invalid_enabled_guard_refuses_even_without_a_valid_engine_target}; Tests/UI/test_console_native_chat_flow.py::test_console_successful_send_does_not_leave_empty_send_tooltip.

Full lifecycle tests that execute shell command hooks are not substitutes for qualified Windows controls. The original unawaited Canvas fixture warning, existing static budgets and prepromotion native ownership gaps remain separately recorded under .11. No primary checkout edits, push or merge.

Source review confirms actual membership absence in all three owner maps plus `engine is None` (preserving injected getters and retained entries). Original cold-empty RED on unchanged41e369c1eb observed one fresh original v2_configuration call and one unused original context-key result; failed only the expected zero-context assertion. The contained run retired normally with force/overflow/lookup races0 and no unawaited/destroyed Task warning. Evidence: .superpowers/sdd/2026-10-07-console-hook-preparation-demand/red-audit.json.

Final source review clear;15 targeted PASS with0failures/errors/skips. Cold original review count remains1 and context count falls1→0. Final source matches frozen hashes; runtime lint0 and unchanged38baseline formatter transformations. Both contained runs retired normally without force/overflow/lookup races; final run has no unawaited/destroyed-task warning. Report: Docs/Development/2026-10-07-console-hook-preparation-demand-verification.md. No elapsed savings or broader responsiveness/native lifetime completion claimed.
