---
id: TASK-41649
title: "Accept a queued Send's capture by a runtime-owned token, not structural equality"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- security
priority: low
references:
- tldw_chatbook/UI/Console_Modules/prompt_queue.py
- tldw_chatbook/Chat/console_configuration_preparation.py
dependencies:
- TASK-41309
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41309 fixed queued Sends made before the Console attach finished. `PromptQueue._with_user_origin` copies the captured snapshot, so the seal's identity check could never hold. The seal (`Chat/console_configuration_preparation.py`, near `:2304`) now accepts the copy when it differs only by `recovery_user_origin`. That is structural equality, and ADR-356 rule 4 prefers runtime-owned authority. A token issued with the capture and carried by the queue would prove the copy came from the queue.

Also noted: queued-turn completion cannot be tested in the harness (the local chat dictionary backend is unavailable there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The queue hands the seal a runtime-owned token issued with the capture, and the seal accepts on the token
- [x] #2 A structurally equal copy without the token is refused (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (ADR-356 rule 4 already prefers runtime-owned authority; this replaces a structural comparison with it).

1. Test first (red on base): extend TASK-41309's queued-Send test in `Tests/UI/test_console_send_draft_snapshot.py` so the real seal is also handed a structurally equal copy of the queued configuration that the runtime did not issue; it must be refused, and the real queued Send must still be sealed.
2. Add `stamp_queue_user_origin(configuration, origin)` to `Chat/console_configuration_preparation.py`: it makes the origin-stamped copy and, when the current capture witness's original is that exact configuration, registers the copy in the witness's original slot. The registered copy is the token: issued by the runtime that owns the capture, carried by the queue as the configuration it passes to `queue_prompt`.
3. `PromptQueue._with_user_origin` stamps through it instead of `dataclasses.replace`.
4. `seal_pending_queue_capture` accepts on the token alone (`witness[-4] is configuration`); the `dataclasses.replace` + `==` comparison and the sealed-slot re-point are removed.
5. Mutation check, covering files, `Tests/Architecture` serially, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The seal now accepts a queued Send's configuration only as the copy the runtime registered for the queue, not by structural equality (ADR-356 rule 4).

- **Token.** New `stamp_queue_user_origin(configuration, origin)` in `Chat/console_configuration_preparation.py` makes the origin-stamped copy and, when the current capture witness's original is that exact configuration, puts the copy in the witness's original slot. That registered copy is the token: issued by the module that owns the capture witness, carried by the queue as the configuration it hands `queue_prompt`, checked by the seal by identity. `PromptQueue._with_user_origin` (`UI/Console_Modules/prompt_queue.py`) stamps through it instead of `dataclasses.replace`.
- **Seal.** `seal_pending_queue_capture` refuses a deferred skill capture unless `witness[-4] is configuration`; the `replace(...) != configuration` comparison, the `witness[-4] is None` guard (now covered by the identity check) and TASK-41309's sealed-slot re-point are gone, because the witness already holds the queue's copy. The sealed tuple is `(*witness[:-3], None, *witness[-2:])` again.
- **Deviation from the wording.** No separate token object is threaded through `queue_prompt`: a free-standing token would prove only that the caller held it, not that the configuration it accompanies is the queue's copy. Registering the copy itself binds the proof to the exact object and adds no parameter to the queue path.
- **Test (AC #2).** TASK-41309's `Tests/UI/test_console_send_draft_snapshot.py::test_console_queued_send_made_before_attach_finishes_is_not_refused` drives a real queued Send while the attach is held; its seal spy now also hands the real seal `replace(configuration)` (a structurally equal copy that is not the registered one), which must be refused, and the real queued Send must still be sealed with the dispatcher's copy. Red on the unfixed seal: `Failed: DID NOT RAISE <class 'RuntimeError'>` at the new case. Green on head. Mutation: with the stamp's witness registration disabled, the real queued Send is refused ("Original queued skill preparation changed; send again."), so the test fails.
- **Evidence.** Same covering runs as TASK-41644 (shared branch): no head-only failure after alone reruns; `Tests/Architecture` 3 failures identical on base; preflight passed.
- **Lane (not edited here).** The node above is in no lane (the file's three spoken-send nodes are in `scripts/ui_pr_gate_census.txt`); the file is `bootstrap_profile`, the node takes ~18 s call / ~42 s alone under load 35-40.
- Files: `tldw_chatbook/Chat/console_configuration_preparation.py`, `tldw_chatbook/UI/Console_Modules/prompt_queue.py`, `Tests/UI/test_console_send_draft_snapshot.py`.
<!-- SECTION:NOTES:END -->
