---
id: TASK-34352
title: >-
  Console controller finds its staged-evidence hooks by attribute name, and a
  resumed turn consumes evidence staged while it was paused
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-03 18:37'
updated_date: '2026-10-04 07:59'
labels:
  - console
  - rag
  - follow-up-33940
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33940.4 found that ConsoleRuntime wired a three-argument capture into a two-argument seam, so every send raised a `TypeError` that a broad `except` hid. The same coupling is still in place elsewhere. `Chat/console_chat_controller.py` reaches its staged-evidence collaborators by taking `self._rag_capture_provider.__self__` and calling `getattr` on it for private method names, in `_has_explicit_staged_evidence`, `_snapshot_staged_evidence` and `_capture_frozen_rag_context`. A missing name returns "unsupported" silently.

On dev `01a2020981` the only controller the app builds is ConsoleRuntime's, and its capture provider is bound to ConsoleRuntime. ConsoleRuntime does not define `_snapshot_console_staged_evidence` or `_release_frozen_console_staged_rag`; they exist only on `ConsoleRetrievalController`, whose capture seam the runtime overrides. Investigation (2026-10-03) established:

- Every production send (composer, queued, wake) is admitted through `ConsoleRuntime.accept_turn`, which pins the staged launch in the custody request and passes explicit capture and release hooks. The name-based fallbacks are reached only by controllers built without the runtime, which today means tests.
- A queued prompt's custody request never carries a launch, so evidence staged while a prompt is queued stays staged for the next manual send. This is coherent and is recorded rather than changed.
- **One real gap:** a composer send admitted with nothing staged records "nothing frozen" (`staged_evidence_launch is not None`), while its first attempt is treated as frozen. If that turn pauses on a card (for example a temporary chat's Capture-On card) and is resumed, it takes the live capture route and consumes evidence the user staged for their next message while the card was open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test against the runtime-owned controller establishes whether an ordinary composer send pins its staged evidence at dispatch, and the finding is recorded in the task notes
- [x] #2 A turn admitted with nothing staged that pauses and is resumed does not consume evidence staged while it was paused; that evidence stays staged for the next message
- [x] #3 The controller receives its staged-evidence collaborators as explicit constructor dependencies; no `__self__` or private-name `getattr` lookup remains for them
- [x] #4 A turn that holds a staged launch with no capture collaborator fails loudly when the lease is built, instead of silently sending without the evidence
- [ ] #5 Existing staged-evidence, queue-custody and dispatch-recovery tests pass, compared against dev `01a2020981` for any suite that already fails locally
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish with the runtime-owned controller: composer sends pin at dispatch (passes on dev); a paused+resumed turn admitted with nothing staged consumes later evidence (RED on dev).
2. Fix the resume gap: a custodied turn admitted with a frozen capture seam freezes 'nothing staged' too, matching its first-attempt route.
3. Replace the three __self__/getattr lookups with explicit constructor params (staged_evidence_snapshot, frozen_rag_capture); leases resolve their capture at construction and refuse a launch with no capture.
4. Move the tests that relied on name-guessing onto explicit wiring via one test helper.
5. Compare the staged-evidence / queue / dispatch-recovery suites against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Investigation (AC1): every production send (composer, queued, wake) is admitted through ConsoleRuntime.accept_turn. accept_turn pins the staged launch in the custody request and passes explicit capture and release hooks (wiring.py, ConsoleRuntime._run_custodied_turn). Test test_a_composer_send_uses_the_evidence_staged_at_dispatch passes on dev: evidence staged after dispatch stays staged for the next message. The three name-guessed fallbacks (__self__ + getattr for _has_staged_evidence / _snapshot_console_staged_evidence / _release_frozen_console_staged_rag / _capture_frozen_console_staged_rag) were reachable only from controllers built without a runtime, which means tests. Queued prompts never carry a launch (queue_prompt builds its custody request without one), so evidence staged while a prompt is queued stays for the next manual send. That is recorded, not changed. Correction to PR #2975's reply to Qodo #1: that live-seam fix covered the controller's own _submit_queued_entry, which production does not use (the runtime binds _submit_queued_turn).

The real gap (AC2): a custodied turn admitted with nothing staged recorded staged_evidence_frozen = False, even though its first attempt took the frozen route. If it paused on a card (for example a temporary chat's Capture-On card) and was resumed, it took the live route and consumed evidence the user staged for the next message while the card was open. RED on dev (sent HELIX-0042). Fixed: a custodied turn admitted with a frozen capture seam freezes 'nothing staged' too.

AC3/AC4: the controller takes staged_evidence_snapshot and frozen_rag_capture as explicit constructor dependencies; no __self__ or private-name getattr remains for them. _PreparedEvidenceLease refuses at construction a staged launch with no capture collaborator (RuntimeError), instead of the old silent drop at capture time. The tests that wired a ConsoleRetrievalController owner by attribute name now pass it explicitly (_retrieval_evidence_owner / _wire_retrieval_evidence_owner in Tests/Chat/test_console_automatic_library_preparation.py; round1/round3/round4 import them).

Tests: Tests/Chat/test_console_staged_evidence_dispatch_pin.py (2, runtime-owned controller via runtime.accept_turn, private profiles).
<!-- SECTION:NOTES:END -->
