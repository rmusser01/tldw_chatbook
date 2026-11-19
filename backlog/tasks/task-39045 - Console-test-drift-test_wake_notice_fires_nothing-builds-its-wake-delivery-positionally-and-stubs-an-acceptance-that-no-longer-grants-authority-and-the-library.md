---
id: TASK-39045
title: "Console test drift: test_wake_notice_fires_nothing builds its wake delivery positionally and stubs an acceptance that no longer grants authority, and the library-preparation file has four reds and eight unbounded provider waits"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-07 06:31
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_chat_controller.py
- Tests/Chat/test_console_automatic_library_preparation.py
- tldw_chatbook/Chat/console_fleet_wake.py
- tldw_chatbook/Chat/console_automatic_submission.py
- tldw_chatbook/Chat/console_chat_controller.py
dependencies:
- TASK-38947
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on Console tests PR #3434 (head 16c99d2d9f), which lands TASK-38947. Line numbers are dev's; ones marked #3434 are from that head. No PR lane runs either file.

**`TestUserPromptSubmitHooks::test_wake_notice_fires_nothing`** (`Tests/Chat/test_console_chat_controller.py:12099`; #3434 :12130).
- Run plain, it fails at both trees on TASK-38111's path (`accepted=False, 'Hooks unavailable…'`).
- Under a temporary `bootstrap_profile` marker it fails with `PermissionError: agent-wake sends require coordinator-issued wake authority` (`console_chat_controller.py:9871`).
- That check predates the test (d42f2e31a1, 2026-08-13; the test came in 54cc387dc9, 2026-09-12). The drift is two later commits:
  - 3176bebc22 (2026-10-03) inserted `slot_token` before `authorization` in `_WakeDelivery` (`console_fleet_wake.py:250-254`). The test's positional `_WakeDelivery(session.id, token)` (`:12128`) now binds the token to `slot_token`.
  - 09c38664b4 (2026-10-11) added a second gate that requires `wake_authorization.accepted` (`console_automatic_submission.py:63-68`). The test's `accept` stub (`:12129-12131`) sets only `acceptance_started`; the real accept sets `accepted = True` (`console_fleet_wake.py:962`).
- A control probe of the verbatim body, under the marker:
  - as written: the first gate refuses it;
  - with a keyword-built delivery: the second gate refuses it ("Automatic source requires live wake authority");
  - with both fixes: it passes.
- TASK-38111's 2026-11-06 note records this PermissionError as a separate error it does not own.

**`Tests/Chat/test_console_automatic_library_preparation.py`.**
- Four nodes are red on dev and on #3434 (`-n 4`, 4 failed / 86 passed, about 17 s):
  - `test_text_with_attachment_is_automatic_eligible_and_preserved_until_ready`: `IndexError: list index out of range` at `:1371` (#3434 :1383).
  - `test_skill_refusal_cleans_preparation_and_allows_next_send`: `TypeError: …refuse() got an unexpected keyword argument 'invocation_source'` (raised from `console_chat_controller.py:10874`), so the stub predates the new keyword.
  - `test_recovery_uses_frozen_staged_inputs_and_leaves_new_state_staged[retry]` and `[bypass]`: the private-profile child fails `assert None is not None`.
- TASK-35992 (Done) recorded three of them as pre-existing, "refused by the existing image-capability test setup before retrieval", and left them unchanged. No open task owns them.
- Eight `await gateway.started.wait()` calls have no bound, at `:1068`, `:1089`, `:1260`, `:1573`, `:1790`, `:2289`, `:2346` and `:2619` (#3434 :1080, :1101, :1272, :1585, :1802, :2301, :2358, :2631); `:3226` (`await stream_started.wait()` in `hold_stream`, #3434 :3238) is the same. A ninth, the helper's own wait at `:649`, is bounded by #3434.
- TASK-38947 bounds the helper `_paused_queued_send` (`:633`). The bound is inline (#3434 :652-662: `asyncio.wait((started, chain), timeout=30, FIRST_COMPLETED)`, then `pytest.fail` with the chain's result), not a reusable function.
- None of the four reds is a bare-wait test, so these waits are latent: a turn that ends before reaching the provider hangs until the 300 s timeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Under its file's profile recipe, `test_wake_notice_fires_nothing` gets past both wake-authority gates and passes on what its name says: a wake notice fires no UserPromptSubmit hook; the notes show it red as written (the PermissionError)
- [x] #2 The four library-preparation reds are each classified (test drift naming the commit, or a product defect with a test red before its fix) and pass; each is shown red first
- [x] #3 When a turn ends before the provider starts, every library-preparation test that waits for the provider fails within a bounded time and names the turn's result; under a temporary edit that refuses before the provider, each affected test fails that way instead of at the 300 s timeout
- [x] #4 The library-preparation file passes whole at `-n 4`, with the count recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the reds on base 8b762b52fa (wake test PermissionError; four library-preparation reds).
2. Wake test: build `_WakeDelivery` by keyword and let the accept stub grant `accepted` as the real accept does.
3. Library preparation: trace each red to its commit; repair the stale seams (vision capability seam, skill-substitution stub signature).
4. Replace every unbounded provider-start wait with one reusable bounded helper that fails naming the turn's result; reuse it in `_paused_queued_send`.
5. Negative control: a temporary source edit that refuses every turn before the provider; each affected test must fail fast naming that result.
6. Run the file whole at -n 4.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five reds were test drift; no product change. Base 8b762b52fa, 2026-11-07.

**Wake test (AC#1).** `test_wake_notice_fires_nothing` now builds `_WakeDelivery(session.id, authorization=token)` by keyword (3176bebc22 put `slot_token` before `authorization`), and its accept stub sets `authorization.accepted = True` as the real `accept` does (09c38664b4 gates the automatic source on it). It runs under the file's `bootstrap_profile` module mark. Red as written on base: `PermissionError: agent-wake sends require coordinator-issued wake authority`. Green after.

**Library-preparation reds (AC#2), each red on base first.**
- `test_text_with_attachment_is_automatic_eligible_and_preserved_until_ready` (`IndexError` at `service.calls[0]`) and `test_recovery_uses_frozen_staged_inputs_and_leaves_new_state_staged[retry]`/`[bypass]` (`assert None is not None` in the private-profile child): drift from 3e618d7119 (2026-10-15). It moved the turn's capability capture into `console_configuration_capture`, so the vision gate reads that module's `is_vision_capable`. Patching only `controller_module.is_vision_capable` left the image turn blocked ("vision-model can't accept images") before Library preparation ran. The new `_vision_capable(monkeypatch)` patches both seams. (The controller's comment at the gate still calls its own copy "the documented monkeypatch seam"; it is stale. Left alone: product comment, outside this task.)
- `test_skill_refusal_cleans_preparation_and_allows_next_send` (`TypeError ... invocation_source`): drift from 0327cb5cf9 (2026-10-15), which added the call's `invocation_source`/`reference_offsets` keywords. The stub now takes the real signature.

**Bounded provider waits (AC#3).** One helper, `_until_provider_starts(started, turn, timeout=30)`, replaces all eight bare `await gateway.started.wait()` calls, the `hold_stream` wait, and TASK-38947's inline bound in `_paused_queued_send`. When the turn finishes first it fails with the turn's result or exception; when neither happens it cancels the turn and fails after 30 s. Negative control: a temporary edit made `ConsoleChatController._apply_skill_substitution` refuse every turn. The 15 provider-waiting tests (29 nodes: the 8 direct waits plus the 7 `_paused_queued_send` users) all failed in 34.6 s at load 69-70, each with `Failed: the turn ended before the provider: ConsoleSubmitResult(... visible_copy='r12t control: refused before the provider' ...)`, and none reached the timeout. The `hold_stream` wait waits on a task the test owns, not on a turn, so that control does not apply to it.

**Whole file (AC#4).** `Tests/Chat/test_console_automatic_library_preparation.py` plus the wake node at `-n 4`: 91 passed in 23.6 s at load 24-34. The file alone is 90 passed; base was 4 failed, 86 passed.

**Files.** `Tests/Chat/test_console_automatic_library_preparation.py`, `Tests/Chat/test_console_chat_controller.py`.

**Proposed follow-ups (not done here).**
- The same file has 14 more unbounded waits on boundaries before the provider (`service.entered`, `held.entered`, `boundary.entered`, `gateway.both_entered`). Under the refuse-everything control, the whole-file run hung on them until it was stopped. `_until_provider_starts` covers them as written.
- The same vision-seam drift (3e618d7119) leaves `Tests/Chat/test_console_chat_controller.py::test_submit_draft_sends_image_parts_when_vision_capable` red, and three tests in `Tests/Chat/test_console_edit_resend.py` are red, causes not read. No lane runs either file.
- Neither file is on a PR lane. They are non-UI (`Tests/Chat`), so lane placement is for the lanes stream.
<!-- SECTION:NOTES:END -->
