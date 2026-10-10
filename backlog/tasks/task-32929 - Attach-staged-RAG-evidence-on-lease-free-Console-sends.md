---
id: TASK-32929
title: Attach staged RAG evidence on lease-free Console sends
status: Done
assignee:
  - '@dsh'
created_date: '2026-09-24 19:59'
updated_date: '2026-09-24 19:59'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every Console send logged `Console RAG capture unavailable; reason=capture_provider_failure` and attached no retrieved evidence, for weeks, while the log blamed the provider rather than naming a fault. The lease-free capture seam had been wired to a callable with a different signature, so the capture raised on every send; the controller swallowed the cause, which kept the mismatch invisible and left the staged-evidence feature silently inert. The failure is benign for delivery - sends still complete - which is exactly why it survived so long.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A lease-free Console send captures the evidence launch the runtime has staged and attaches its rendered context.
- [x] #2 The capture seam the controller is wired to accepts the `(draft, turn_context)` call the controller derives from its signature, and the lease-bound frozen capture keeps its three-argument contract on the same owner.
- [x] #3 A failed capture releases nothing, so evidence the turn never received stays staged for the next attempt.
- [x] #4 A capture failure is reported with its exception type and a send diagnostic carrying the root cause, instead of only `reason=capture_provider_failure`.
- [x] #5 Focused suites and `./scripts/preflight.sh` show no regression attributable to this change; pre-existing failures in the touched files reproduce at baseline.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/092-console-full-semantic-capture-policy.md
Reason: repairs a wiring defect inside the accepted capture contract; no storage, permission, provider-boundary or UX-structure decision is made, and the staged-evidence ownership model is unchanged.

1. Reproduce the failure: assert the exact provider `ensure_chat_controller` hands the controller and show it does not accept `(draft, turn_context)`.
2. Give `ConsoleRuntime` a lease-free two-argument capture that reads the runtime's own staged launch and releases it on the exact revision that produced a non-empty context, then wire that seam.
3. Name the cause at both capture-failure handlers instead of dropping it.
4. Add regressions for the wiring, the capture/release behaviour, the failed-capture case and the reporting, then run the focused suites and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Incident (2026-09-24, live). `tldw_cli_app.log` carried `Console RAG capture unavailable; reason=capture_provider_failure` on every send, including the successful ones (19:42:31 and 19:43:04 completed, 19:48:40 blocked on an unrelated trace surface refusal), so it read as a symptom of whatever else was failing. The cause was a signature mismatch: `_capture_rag_context` inspects `self._rag_capture_provider`, sees it accept a second positional argument, and calls it with `(draft, turn_context)`, but `ConsoleRuntime.ensure_chat_controller` wired `_capture_frozen_console_staged_rag(draft, turn_context, launch)` - the lease-bound capture - so the call raised `TypeError: ConsoleRuntime._capture_frozen_console_staged_rag() missing 1 required positional argument: 'launch'` on every send (reproduced directly against the wired callable). Because the handler logged only `reason=capture_provider_failure` and dropped the exception, this stayed invisible for weeks; `Tests/UI/test_console_auto_rag_on_send.py` only exercised the view's own two-argument seam, which was never the wired one.

Fix. `ConsoleRuntime._capture_console_staged_rag(draft, turn_context=None)` reads this runtime's staged slot (`snapshot_console_staged_evidence`), captures through `capture_console_staged_evidence_for_chat`, and calls `release_console_staged_evidence(launch, result, revision=...)` - so the launch is released only for the exact revision that produced a non-empty context, the cleared slot and the sent-source receipt stay consistent, and a failed capture leaves the evidence staged for the next attempt. `ensure_chat_controller` now wires that seam; the frozen path still resolves `_capture_frozen_console_staged_rag` through the same owner (`provider.__self__`), so lease-bound sends are unaffected. Both capture-failure handlers now log `exception_type=` and call `record_send_stage("rag_capture", "failed", error=exc)`, which walks `__cause__`/`__context__` to the root cause and classifies it content-free.

Evidence. New suite `Tests/Chat/test_console_rag_capture_provider.py`, 5 passed. The wiring test is genuinely RED on the pre-fix code: with the old wiring restored it fails `assert <function ConsoleRuntime._capture_frozen_console_staged_rag> is <function ConsoleRuntime._capture_console_staged_rag>`, and the arity bind in the same test raises on the three-argument seam. The behavioural tests cover capture-and-release, failed-capture-keeps-staged, the reporting path (a `TypeError` provider produces a `phase=rag_capture status=failed` diagnostic with `exception_type=TypeError`), and the composed path where the controller seam drives the newly wired runtime provider, attaches the rendered context, releases the launch, and records no failure. Batch of the new suite with `test_console_auto_rag_on_send.py`, `test_console_runtime_lazy_voice.py`, `test_console_runtime_lifetime.py` and `test_console_wave6_inventory.py`: 74 passed, 5 failed, 4 errors, and the identical 5 failures reproduce with this change stashed (`test_console_wave6_inventory.py` ratchets over an untouched `chat_screen.py`, and both `test_console_runtime_lifetime.py` failures plus the 4 setup errors are the environmental `RecoveryRequired: raw_source_selection_changed` fixture failure). `Docs/security/production-diagnostic-inventory.json` regenerated after reviewing the two reworded statements - they interpolate only `len(draft)` and `type(exc).__name__`.

Files: `tldw_chatbook/Chat/console_runtime.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `Tests/Chat/test_console_rag_capture_provider.py`, `Docs/security/production-diagnostic-inventory.json`. Related: TASK-32928 fixed the trace-provenance block that made the same sends look unresponsive; this seam was the separate defect its notes recorded as unfixed. The changes are in the `dev` working tree, not committed.

Not addressed here: whether retrieval-backed context is worth attaching on this machine. The media database holds one document, `MediaChunks` is empty, 40 chunks remain in `UnvectorizedMediaChunks`, and `[console] rag_auto_retrieve_on_send` is false - so repairing the seam removes the false error and restores the feature, but a turn still attaches evidence only when the user stages it explicitly.
<!-- SECTION:NOTES:END -->
