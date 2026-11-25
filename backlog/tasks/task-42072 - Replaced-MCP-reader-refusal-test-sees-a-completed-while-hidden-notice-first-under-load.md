---
id: TASK-42072
title: "Replaced-MCP-reader refusal test sees a 'completed while hidden' notice first under load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 01:10
updated_date: '2026-11-25 14:02'
labels:
- follow-up
- tests
- flaky
priority: medium
references:
- Tests/UI/test_console_send_without_mcp_store.py
- tldw_chatbook/Chat/console_runtime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-42071 local review on 2026-11-19, at dev a6df4d8d48 plus that fix. The test `Tests/UI/test_console_send_without_mcp_store.py::test_unbound_store_with_a_replaced_mcp_reader_is_still_refused` was laned for the first time by the TASK-41630..41655 burn-down.

**Observed failures.** It failed in 2 of 3 local `-n 4` runs of its file, while a serial `Tests/Architecture` run loaded the machine. In those runs the first recorded notice was "A Console turn completed while hidden. Return to Console to review." rather than "Console preparation source unavailable; Send again." It passed alone, in 3 of 3 quiet runs on that branch and 3 of 3 on dev, and in the CI UI Fast Lane run that exposed TASK-42071.

**Where the notice comes from.** `ConsoleRuntime._notify_terminal_receipt` (`Chat/console_runtime.py`) sends that notice only for a COMPLETED terminal receipt that it does not count as on screen.

**Two possible causes, unresolved:**
- A stray receipt arrives first.
- Under load, the replaced-reader Send is not refused at all.

Telling them apart matters, because the second would be a product bug (the test's second assertion, `gateway.sent_messages == []`, never ran in those failures).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The source of the "completed while hidden" receipt in a failing run is identified, with the evidence recorded
- [x] #2 A replaced MCP reader is shown to refuse the Send under load, or the product is fixed so that it does (red first)
- [x] #3 The test passes in 10 of 10 loaded `-n 4` runs of its file
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC#3 first, on unmodified dev: ten `-n 4` runs of the file while serial `Tests/Architecture` runs load the machine.
2. AC#1: in a throwaway worktree, disable TASK-42018's conftest receipt clear (the 11-19 state of this file) and run the file under the same load with a probe plugin that logs, per worker process, the unseen receipt marks at each test's start and end and every `_notify_terminal_receipt` call; the target test also logs its notices and `gateway.sent_messages`.
3. AC#2: from the same probe, check that the Send was refused even in the failing runs (refusal notice arrives, no provider call); a pass at dev means both assertions ran.
4. Notes only unless the Send is ever not refused (then a red-first product fix).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Verdict: cause 1, a receipt left by the previous test on the same worker. The replaced-reader Send was refused every time, so cause 2 is ruled out.** TASK-42018's conftest clear (a91a9645c4) already removes the cause, so this task changes no code or test.

**AC#3 (dev 2959b48314, unmodified).** Ten `-n 4` runs of the whole file ran while one serial `Tests/Architecture` run (its `test_*.py` files) loaded the machine throughout. Load1 at run start was 24-54, raised further by this session's other runs. The target passed 10 of 10. The whole file passed 9 of 10. Run 6 (load 45-53) failed a different node, `test_draft_snapshot_sends_without_a_bound_mcp_store[test_spoken_send_sends_its_capture_and_keeps_text_typed_after_it-no-pinned-io-agent]`, with `timed out waiting for _speak_armed_send.<locals>.<lambda>` (the 15 s wait for the spoken send's worker). Its `-non-default-data-dir-agent` sibling failed the same way once in the probe runs below. Neither is this task's test; both are reported here, not fixed.

**AC#1, the source.** The probe ran in a throwaway worktree of the same dev, with the conftest call `_clear_bootstrap_console_receipts()` replaced by `pass`. That is the file's 11-19 state: a6df4d8d48 had neither the clear nor TASK-41900's local workarounds. An uncommitted pytest plugin logged, per worker process:
- the `console_unseen:` marks in the shared ChaChaNotes database at each test's setup start and after its teardown;
- every `ConsoleRuntime._notify_terminal_receipt` call.

The target also logged when it installed its notice recorder, its notices and `len(gateway.sent_messages)`.
- **Four natural `-n 4` runs (load 32-57).** 20 of 24 `test_ordinary_send_reaches_the_provider_and_saves_the_reply` cases left a new unseen receipt in their worker's database (their app closes before the reply renders). Every later test on that worker announced exactly those receipt ids. The target passed in all four. Today's file order puts the three synchronous `test_turn_sources_*` cases ahead of it on its worker, and they leave no receipt. In a6df4d8d48's file, the target came straight after the ordinary-send cases.
- **Forced order in one process (`-p no:xdist`, the `[bound-gateway]` ordinary send and then the target), six runs at load 30-53.** The ordinary send left a receipt in four. Each receipt was absent at the ordinary send's start, present after its teardown and present at the target's start. It was announced during the target (`_notify_terminal_receipt` returned True), but 0.36-1.01 s before the target installed its recorder, so the target passed. That is the race: the target fails only when the mount-time announcement lands after `_record_notices` is installed.
- **A slower host's order, forced.** Holding the target's first attention-marks read for 1.0 s (one-shot, on its worker thread) moves the announcement past the recorder. Then 2 of 3 runs failed with the 11-19 assertion, `['A Console turn completed while hidden. Return to Console to review.'] != ['Console preparation source unavailable; Send again.']`. The third run ended before the refresh read again. (A 1.5 s hold on every read, tried first, let no announcement land before the test ended: 3 of 3 passed.) In the first failure, receipt `45fffcd9-...` of chat `ab26899e-...` was absent at the ordinary send's start and present after its teardown and at the target's start. It was announced in the target 0.98 s after the recorder was installed.
- **Control.** The same held pair with dev's conftest restored: 3 of 3 passed. No receipt was left at the target's start.

**AC#2, the refusal.** Every probe run of the target (4 natural, 6 forced, 6 held, 3 control) found `gateway.sent_messages` empty and received the refusal notice. In the two failing held runs the notices were the stray "completed while hidden" and then "Console preparation source unavailable; Send again.", with 0 messages sent. So the 11-19 failures were the stray notice alone, and the Send was refused. In the ten AC#3 passes both assertions ran and held: the refusal check is the test's last statement.

No lesson added. TASK-42018's and TASK-42138's notes already record the leak and its cure. The probe worktree and plugin were deleted.

Files: this task file only.
<!-- SECTION:NOTES:END -->
