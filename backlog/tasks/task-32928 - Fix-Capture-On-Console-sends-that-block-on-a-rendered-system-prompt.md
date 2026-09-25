---
id: TASK-32928
title: Fix Capture-On Console sends that block on a rendered system prompt
status: Done
assignee:
  - '@dsh'
created_date: '2026-09-24 19:46'
updated_date: '2026-09-24 19:46'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Console send in a chat with Capture On was durably accepted and then blocked
before the provider was ever contacted, so the chat looked dead: the user's
message was committed and visible, no reply ever arrived, and no recovery
affordance appeared to explain or escape the state. Two independent defects
produced that experience - capture admission refused the request the turn had
just been accepted with, and the pause raised for that refusal was filtered out
of the transcript UI, leaving the failure both undiagnosable from the log and
invisible on screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Capture-On Console send whose provider vector begins with a system prompt reaches `provider_entry` and completes, instead of blocking after durable acceptance with `provider_started=False`.
- [x] #2 A post-commit TRACE_PROVENANCE pause renders a recovery callout offering Retry capture, Send without capture and Cancel send.
- [x] #3 Regression tests cover both defects, including a guard that the pre-fix descriptor is still rejected by the `system` provenance category, so the fix cannot silently become the expectation.
- [x] #4 Focused suites and `./scripts/preflight.sh` show no regression attributable to this change; pre-existing failures in the touched files reproduce at baseline.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/092-console-full-semantic-capture-policy.md (with backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md)
Reason: repairs a defect inside the accepted capture/provenance contract - no storage, schema, permission, provider-boundary or application-structure decision is made, and the durable trace surface is unchanged.

1. Reproduce a real Capture-On first send offline with a system prompt present, and A/B the pre-fix descriptor against the fixed one.
2. Return the provenance source the `system` category accepts for a leading, non-memory system row, mirroring `build_console_request`'s own classification rather than restating it.
3. Project the TRACE_PROVENANCE pause into the existing recovery callout, whose controller-side entrypoint and all three actions already accept that pause kind.
4. Add regressions for both defects, then run the focused suites and `./scripts/preflight.sh` and account for every failure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Incident (2026-09-24, live). Every Capture-On send in a Console chat was accepted, committed, and then blocked before dispatch. `tldw_cli_app.log` for the 19:24:00 attempt: `durable_commit status=succeeded` (with `Added conversation ID: 93cd5fe0-...`), then `phase=trace_provenance status=failed error_category=validation exception_type=TraceProvenanceAlignmentError`, then `controller_submit status=blocked`. Healthy sends instead continue `trace_reservation entered/succeeded -> trace_dispatch_commit -> provider_entry`. The earlier 16:43-run sends worked; the 18:03-run send and every send after it failed with identical code on disk, so the trigger was payload state, not a code change.

Root cause. `_build_durable_trace_request` gave every provider row without a saved revision a `ProviderArtifactTraceProvenance(ACTIVE_REQUEST, policy)` descriptor. `ConsoleRequestProvenance` files the leading, non-memory `system` rows under its `system` category, which admits only a saved revision or `TraceProvenanceSource.RENDERED_SYSTEM` (`Chat/console_trace_provenance.py:989`). The session now sends a system prompt, so the vector `["system", "user"]` raised `TraceProvenanceAlignmentError: trace provenance category mismatch: system` after durable acceptance - provider never called. The 16:43 sends carried no unsaved system message, which is why the same code passed then.

Second defect. `trace_call_recovery_state` (`UI/Console_Modules/provider_continuation_recovery.py`) projected only TRACE_CALL and TEMPORARY_CAPTURE pauses, while `ConsoleChatController.trace_call_recovery_preparation`, `retry_library_preparation` (-> `_retry_durable_trace_provenance`), `send_without_capture` and `cancel_library_preparation` all accept TRACE_PROVENANCE. The pause therefore rendered nothing: no callout, no actions, no explanation.

Fixes. `_rendered_trace_source_for_row` returns RENDERED_SYSTEM for a leading, non-memory system row and ACTIVE_REQUEST otherwise, mirroring `build_console_request`'s classification and matching the precedent already in `_build_speculative_voice_capture_request`; RENDERED_SYSTEM already has first-class durable handling (`Chat/console_trace_service.py:1417`). The callout projection now includes TRACE_PROVENANCE, so the existing card ("Trace capture blocked", with Retry capture / Send without capture / Cancel send) appears for that pause. The two pause-entry points also record `record_send_stage("trace_provenance", "failed", error=exc)`, which walks `__cause__`/`__context__` past the `from None` erasure - that is what named this failure at 19:24 and it reuses the content-free categories registered by TASK-32926.

Evidence. Offline A/B of a real Capture-On first send with a system prompt (temp DB, stubbed provider), payload roles `['system', 'user']`: pre-fix (helper forced to ACTIVE_REQUEST) printed `TRACE BUILD RAISED: TraceProvenanceAlignmentError: trace provenance category mismatch: system` and returned `accepted=True provider_started=False status=BLOCKED copy='Trace provenance could not be saved. Retry, Send without capture, or Cancel.'`; with the fix it printed `TRACE BUILD OK: capture_durability=durable` and returned `accepted=True provider_started=True`. Live after the fix: 19:42:31 `0865ec0a` and 19:43:04 `d1d42ef1` both ran `durable_commit succeeded -> trace_reservation entered/succeeded -> trace_dispatch_commit entered -> provider_entry entered -> controller_submit status=completed` (9.3s and 7.7s), with conversation `f04b7954` holding `hello` -> 129-char reply and `tell me a short story` -> 1272-char reply, so first and follow-up turns both pass with capture still On. Tests: `Tests/Chat/test_console_prepared_request.py` 39 passed (37 baseline + 2 new); targeted batch of that file with `Tests/Chat/test_console_trace_provenance.py`, `Tests/UI/test_console_trace_capture_recovery_flow.py`, `Tests/UI/test_console_provider_continuation_recovery.py` 99 passed / 3 failed, all three the environmental `RecoveryRequired: raw_source_selection_changed` fixture error that reproduces with this change stashed. `./scripts/preflight.sh` reports only the pre-existing Canvas Mermaid asset and duplicate `task-32187` failures; the production diagnostic inventory verifies clean.

Files: `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/UI/Console_Modules/provider_continuation_recovery.py`, `Tests/Chat/test_console_prepared_request.py`, `Tests/UI/test_console_trace_call_recovery.py`. Also regenerated `Docs/security/production-diagnostic-inventory.json` after reviewing its one drifted row (a code-owned enum value from the sibling speech-snapshot warning, no user content). The fix is in the `dev` working tree, not committed.

Verification gaps, stated plainly. `Tests/UI/test_console_trace_call_recovery.py` errors wholesale in this environment (`RecoveryRequired`, baseline 5 errors before the new tests were added), so its two projection assertions were additionally verified directly by script: TRACE_PROVENANCE and TRACE_CALL project `temporary_capture=False`, TEMPORARY_CAPTURE projects `True`, and RETRIEVAL/PERSISTENCE still project `None`. The repository's frozen trace censuses (`CONSOLE_REQUEST_ROUTE_CENSUS`, `CONSOLE_GATEWAY_CALLSITE_CENSUS`) pin source line numbers and are already stale at HEAD in files untouched here (`console_context_compaction.py` pinned 483 while its marker sits at 2218), so this change leaves them alone rather than folding an unrelated regeneration into it.

Separately repaired on this branch, not part of this fix: the non-frozen RAG capture path logged `Console RAG capture unavailable; reason=capture_provider_failure` on every send because `_capture_rag_context` inspected the wired provider, saw it accept a second positional argument, and called it with two, while the wired `ConsoleRuntime._capture_frozen_console_staged_rag(draft, turn_context, launch)` required three - so that path silently attached no evidence. That seam now wires a runtime-owned lease-free two-argument capture and reports the failure cause. It was benign for this incident: successful sends before and after the fix logged it too.
<!-- SECTION:NOTES:END -->
