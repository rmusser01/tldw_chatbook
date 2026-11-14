---
id: TASK-39158
title: "test_height_resize_requests_one_coalesced_context_reconcile counts two context reconciles for one height resize on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: '2026-11-10 18:40'
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_resize_reflow.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-39132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39132's covering runs, round-12 branch fu/r12-cprod). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08.

- `Tests/UI/test_console_resize_reflow.py::test_height_resize_requests_one_coalesced_context_reconcile` is a plain test, so locally it stops at `RecoveryRequired`.
- Under a temporary `bootstrap_profile` marker it fails `assert 2 == 1`, on both dev and the stack. The marker came from a scratch plugin; runs were through run.sh at `-n 4` and load ~22-27.
- TASK-39132's notes record the same under a temporary `@private_profile_test`, with and without that task's change, 3 runs each.
- The file has one UI census entry (`test_model_summary_sync_invalidates_mounted_context_allocation`). This node runs on no lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The second reconcile is explained, and fixed or re-pinned: a product regression (one height resize should coalesce to one reconcile) is fixed with this test red before, and drift is re-pinned with the commit that changed the count named
- [x] #2 The node runs plain through the suite's existing profile seams, on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

1. Run the node under a temporary node-level `bootstrap_profile` marker on base and read where the second reconcile comes from (count callers of the reconcile request during one height resize).
2. Find the commit that added the second request; classify regression (fix the product, test red before) or drift (re-pin, commit named).
3. Keep the node-level `@pytest.mark.bootstrap_profile` so it runs plain; time it serially three times (lane wiring by the round's parent).

## Implementation Notes

**Drift, re-pinned; no product change.** Base ae9f9e8d21 with only a node-level `@pytest.mark.bootstrap_profile` added: `assert 2 == 1` at the count (run.sh, load ~59).

- **The second reconcile** is the Character radio's follow-up, added by b333cebdfe (2026-10-23, "frame living portraits with truthful radio voice status"). The resize's requests (screen height adaptation, workspace sync, rail resize) all coalesce into one pass. That pass's `_reconcile_character_avatar_geometry` finds the voice meter no longer fits at 30 rows, calls `PortraitRadio.set_compact(True)`, which changes the Character body's height, and requests exactly one follow-up (`left_rail.py`, `call_after_refresh(self.request_allocation_reconcile)`). Probe: `set_compact` was called once, between run 1 and run 2; before the resize `voice-compact` was off, after it on. The flip is idempotent, so no further pass follows (the test's stable-runs check still holds).
- Re-pin: a `set_compact` spy; `compact_flips == [True]` and `reconcile_runs == 1 + len(compact_flips)`. Control (temporary product edit, reverted): without the radio's follow-up request, `assert 1 == (1 + 1)` at that line.
- The count fix exposed a second drift further down: `last_header` was `#console-rail-section-header-character`, which was the last Context section until 96f24f0efa (2026-09-05) moved Character to third; after `scroll_end` its header sat at y=-16, outside the viewport. It now reads the last descriptor's header (`CONTEXT_SECTION_DESCRIPTORS[-1]`, Details).
- The node keeps its node-level `bootstrap_profile` marker (like its neighbours), so it runs under its plain command; the file's census node `test_model_summary_sync_invalidates_mounted_context_allocation` stays unmarked.

**Lane facts for the parent:** serial `--timeout=180 --tb=short -q`, three runs: 1 passed in 26.3 / 19.7 / 26.2 s pytest, 37 / 27 / 37 s wall, load 65 / 63 / 57 on 18 CPUs.

Files: `Tests/UI/test_console_resize_reflow.py`.

**Lane (AC#2).** The node runs plain under its node-level `bootstrap_profile` marker. It is a node entry at the UI census tail; the file's other census node stays plain. Serial pytest times: 26.3 s, 19.7 s and 26.2 s (load 57-74). The census replay charged it 27 s, and `--rebalance 13.3` kept every shard at 13.1 pytest min or less over the 22 cached PR runs.
