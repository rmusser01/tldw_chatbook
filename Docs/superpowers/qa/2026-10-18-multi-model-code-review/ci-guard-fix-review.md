# TASK-35582 CI guard repair: scoped review

What this change does: It defers the review coordinator, ledger attachment, and annotation-only imports until use, and replaces the report modal's bare button selector with an explicit class. It also declares the existing modal launch/export edges in the inventory.

## Must fix

1. **[P2] Concurrent first sends can leave the ledger attached to a different review owner** (`tldw_chatbook/Chat/console_chat_controller.py:24968–24980`, connected registration guard at `25011–25023`).

   **What this is:** Each session's native provider composition runs through `asyncio.to_thread(compose_native_review_provider, ...)` at controller line 27079. That function obtains `controller.code_review` before binding the accepted request. Concurrent sessions therefore can enter this new lazy getter on different worker threads.

   **Problem:** Initialization has no serialization. A valid interleaving is: A and B both observe `_code_review is None`; B constructs/publishes coordinator B and attaches B's bound authority methods to the shared ledger; A then finishes its constructor assignment and overwrites `_code_review` with coordinator A. A's registration sees that the ledger is already `_review_ledger`, so it skips replacing the authority. The controller now creates boundaries through A while the ledger validates/transitions through B. B does not own A's boundary registry, so A's otherwise valid continuation is refused. Callback wiring and owner publication also occur before initial registration is complete. Eager construction before the worker submissions previously prevented this first-use overlap.

   **Fix:** Serialize the lazy construction, callback setup and initial attachment, including concurrent registration/bridge-swap access. A controller `RLock` can retain the existing reentrant registration/getter calls and exact same bound authority methods. Keep the import deferred. Add one constructor-barrier control with two first-use callers; assert one owner identity and that the registered authority methods belong to that returned owner.

   **If we skip it:** Simultaneous first sends in different Console sessions can leave review continuations permanently attached to the wrong in-memory issuer. The failure is a refusal, not an authority expansion. This is a static concurrency finding; no reproduction or test was executed by this reviewer, as instructed.

The ordinary send already obtains `runs_db.automatic_work` and establishes the work chain before threaded native composition (`_stream_assistant_response`, around lines 24563–24577). Consequently a cold-ledger filesystem lookup is not needed for the interleaving above, and is not claimed as the cause.

## Other scoped checks

- On a single first-use path, the property attaches the ledger before returning the coordinator, and the existing bridge swap still unregisters the old ledger and resets runtime authority before attaching the new one. Same-name bridge callback registration remains replacement rather than duplication. The finding is the newly concurrent first-use publication, not a proposed change to those contracts.
- Stop/close/shutdown still fence/close through the same coordinator methods, and shutdown unregisters the recorded authority. No separate cleanup defect was found in the seven-path diff.
- Both trace dataclasses retain deferred annotations (`from __future__ import annotations`); the two non-null witness branches still import and enforce the exact runtime witness type. No runtime evaluation of these dataclasses' annotation strings was found in the connected code.
- Settings imports now cover every remaining runtime use of `ReviewTeam`, `ReviewRefused` and `parse_review_team`. No unresolved annotation or exception name was found.
- All five existing modal action buttons carry `code-review-action`; the generated selector matches the source selector and retains the same token-backed values. The root inventory includes the report modal and its actual export edge to `ConsoleSaveMarkdownModal`; safe-dismissal inheritance and Escape handling remain present.

**Verdict: fix finding 1 before committing this repair.** Other original-feature reviews remain reused; this is not a request for another cohort or broad review.

## Evidence and limits

Reviewed the current seven-path uncommitted diff on candidate `cee6fdcf26`, its directly connected controller/coordinator/ledger/fan-out code, the author report `/private/tmp/task35582-bounded-recovery/ci-guard-fix-report.md`, and the root modal-inventory changes. The author reports the three exact CI ratchets individually green, affected direct controls green, and the root reports two modal inventory nodes green plus the declared-Python fatal/undefined/duplicate static guard green. These results were reused, not independently rerun here. The new lazy-owner control checks sequential cache identity; it does not exercise simultaneous first use.

Not checked: tests, live providers, credentials, spending, full-feature qualification, or the eventual commit. No edits were made except this report. The pending 66-request/3-child allowance was not used.


## Finding 1 closeout — corrected

**Finding 1 is closed. No remaining actionable finding in this incremental correction.** The original finding above is retained as the incident record; its earlier fix-before-commit verdict is superseded by this closeout.

Reviewed only the added controller `RLock`, the connected getter/ledger registration critical sections, and the constructor-overlap regression. The getter now takes `_code_review_lock` before checking the cache and retains it through construction, callback setup and initial registration. Every competing getter must acquire the same lock before reading or returning the cached owner. `_register_fleet_wake` takes that same lock around ledger lookup, old authority removal/runtime reset and new authority registration. This prevents the reported A/B construction and attachment interleaving while retaining the original bound methods and bridge-swap behavior.

The lock is reentrant: getter → `_register_fleet_wake` → getter obtains it again on the same thread, so the existing registration calls do not self-deadlock. The reviewed ledger register/unregister functions mutate their in-memory registry under the existing progress lock without invoking authority callbacks; coordinator reset/close does not call back into the controller getter. No newly introduced reverse acquisition of `_code_review_lock`, await, UI handoff, or worker join was found in these connected critical sections. The outer first-use registration still seeds only the existing body-free progress hints after its ledger block; those callbacks schedule intake rather than synchronously waiting for the app loop. No immediate new lock-order/deadlock issue was found at the corrected seams.

The constructor-overlap control forces the first constructor to remain pending while another worker attempts the getter, then asserts one constructor, identical returned/cache owner identity, and all three ledger callback owners matching that identity. The retained RED log shows the precise original mismatch (first/cache owner A versus second/callback owner B), **1 failed in 1.12s**. The retained GREEN log shows this control, sequential lazy identity and native default-two-child continuation control all passing, **3 passed in 6.24s**. These existing runs were inspected, not rerun by this reviewer.

Frozen source hashes match the updated author report: controller `ea051c4a88cd9c5bbff5299b0eb9100151b9e646e1c1b50300a92a19bdec56c8`; authority test `511238e9015baffcbb87ffbbd4d398d68d1e685425536c0f0a79eb352026d61e`.

**Updated scoped verdict: the CI-guard repair can proceed to the root's delivery steps.** This closes only finding 1 and reuses the earlier import/modal/inventory review. It does not close AC24 or authorize provider execution.

Not checked: no tests were rerun, and no prior feature, provider, budget, or broad lock audit was repeated. No source changes or new spending were made; only this closeout was appended.
