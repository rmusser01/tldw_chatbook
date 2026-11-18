---
id: TASK-41309
title: "A queued Send made before the Console attach finishes may be refused as 'Original queued skill preparation changed; send again.'"
status: Done
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- bug
priority: low
references:
- tldw_chatbook/Chat/console_configuration_preparation.py
- Tests/UI/test_console_send_draft_snapshot.py
- https://github.com/rmusser01/tldw_chatbook/pull/3541
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40714 (PR #3541, review M4) found it in a test and has not shown it in the product. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** A user who sends while another run is active gets a queued Send. If that Send is made before startup's Console attach reconciliation finishes, it may be refused with "Original queued skill preparation changed; send again." The user then has to send again for no reason they can see. Queueing needs an active run, so the likely real-world trigger is a restored session whose run is still running at startup. That case is unproven.

**Evidence.**
- `Chat/console_configuration_preparation.py:2481-2514` (`seal_pending_queue_capture`) raises the refusal at `:2512-2514`. It does so when the queued configuration took the deferred skill capture and the witness no longer points at it, or it is no longer pending.
- TASK-40714's notes: `Tests/UI/test_console_send_draft_snapshot.py::test_console_active_run_queues_without_a_screen_worker` fakes an active run right after the composer mounts. When its dispatch ran before `_reconcile_console_after_attach` finished, the skill sources were still cold, so the Send took the deferred skill capture and was refused. Delaying the reconciliation by 2 s made it fail 2 of 2 times. It also failed once on the bound default path, so it is not specific to an unbound MCP store. The test now waits for the attach to settle, so it no longer covers this case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reproduction through a real or restored active run (not a faked one) records whether a queued Send made before the attach finishes is refused
- [x] #2 If it is refused, that Send is either delivered once the skill sources are ready or refused with copy that says what to wait for, pinned by a test that is red on dev; if it is not, the notes say what prevents it
<!-- AC:END -->

## Implementation Plan

ADR required: no. A one-function correction to a queue-seal identity check; no design choice.

## Implementation Notes

- Reproduced through a REAL run, not a faked one: a real first Send streams through a held gateway (`WaitingGateway`) while `_reconcile_console_after_attach` is held open, then a second Send queues. It is refused with "Original queued skill preparation changed; send again." (red on base, `Tests/UI/test_console_send_draft_snapshot.py::test_console_queued_send_made_before_attach_finishes_is_not_refused`). So no restored session is needed: any Send made while the attach is unfinished reaches it, because a Send is not gated on the attach.
- Root cause is not timing. `PromptQueue._with_user_origin` always stamps `recovery_user_origin` onto a COPY of the captured snapshot, so `seal_pending_queue_capture`'s `witness[-4] is not configuration` identity check can never hold for a deferred (cold-skill) capture. The warm path takes no deferred capture, which is why it only appeared before the attach settled.
- Fix (`Chat/console_configuration_preparation.py`): the seal accepts the copy when it equals the witnessed original apart from `recovery_user_origin` (`dataclasses.replace` + `==`), and re-points the sealed witness's original-configuration slot at the copy, so the ordinary tree's later `record.request.configuration is original` custody check still holds when the queued turn runs. Any other difference is still refused.
- Delivery: the queued Send is admitted and its skill capture resolves when the turn starts (the existing deferred path), so nothing waits on the attach.
- Files: `tldw_chatbook/Chat/console_configuration_preparation.py`, `Tests/UI/test_console_send_draft_snapshot.py`.
- Verification limit: the harness cannot run a queued turn to completion even on the warm path (`ValueError: Local chat dictionary backend is unavailable`), so the test asserts the handoff directly: inside the seal, the sealed witness's original slot is the dispatcher's copy (so the ordinary tree's `record.request.configuration is original` check holds), a copy differing in any field but `recovery_user_origin` is still refused, and a witness with no original is refused with the intended RuntimeError (guarded; `replace(None, ...)` would have been a TypeError).
- Follow-up (coordinator to file): a runtime-owned token instead of structural equality.
