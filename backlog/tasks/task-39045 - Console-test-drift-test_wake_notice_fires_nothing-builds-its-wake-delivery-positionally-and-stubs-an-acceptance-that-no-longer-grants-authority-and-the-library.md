---
id: TASK-39045
title: "Console test drift: test_wake_notice_fires_nothing builds its wake delivery positionally and stubs an acceptance that no longer grants authority, and the library-preparation file has four reds and eight unbounded provider waits"
status: To Do
created_date: 2026-11-06 10:09
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
- [ ] #1 Under its file's profile recipe, `test_wake_notice_fires_nothing` gets past both wake-authority gates and passes on what its name says: a wake notice fires no UserPromptSubmit hook; the notes show it red as written (the PermissionError)
- [ ] #2 The four library-preparation reds are each classified (test drift naming the commit, or a product defect with a test red before its fix) and pass; each is shown red first
- [ ] #3 When a turn ends before the provider starts, every library-preparation test that waits for the provider fails within a bounded time and names the turn's result; under a temporary edit that refuses before the provider, each affected test fails that way instead of at the 300 s timeout
- [ ] #4 The library-preparation file passes whole at `-n 4`, with the count recorded
<!-- AC:END -->
