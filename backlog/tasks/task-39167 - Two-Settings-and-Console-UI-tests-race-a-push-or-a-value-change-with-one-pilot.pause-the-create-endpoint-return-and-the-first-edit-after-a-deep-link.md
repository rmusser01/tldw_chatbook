---
id: TASK-39167
title: "Two Settings and Console UI tests race a push or a value change with one pilot.pause(): the create-endpoint return and the first edit after a deep link"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:36
labels:
- follow-up
- tests
- settings
- console
- flaky
priority: low
references:
- Tests/UI/test_console_endpoint_discovery.py
- Tests/UI/test_settings_deep_link_first_edit.py
dependencies:
- TASK-38961
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-38961's notes, round-12 branch fu/r12-ctests; and the settings stream). Checked by reading on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08; line numbers are dev 3a6498ae14's (re-read 2026-11-09, both test files unchanged since b590a15afd).

**The create-endpoint return.** `test_console_endpoint_discovery.py::test_create_endpoint_returns_to_responsive_settings_with_settled_evidence[outcome0]` failed 2 of 10 whole-file runs at load 45-161 (TASK-38961's notes).
- The test (`:125`) pushes the modal (`await app.push_screen(modal)`, `:139`), waits one `pilot.pause()`, presses New (`#console-settings-endpoint-new`, `:141`), waits one more `pilot.pause()`, then asserts that `app.screen` is the template modal and queries `#endpoint-template-name` (`:144-145`), later calling `_pick_in_pick_mode` (`:153`). A press or query that lands before the pushed screen has composed is the race TASK-38961's notes name.
- The node is not laned. Other nodes of the file are: 4 UI census entries on dev (7 on the round-12 stack), and 9 node ids on the workflow lane (`.github/workflows/derived-artifacts.yml:681-689` on dev 3a6498ae14).

**The first edit after a deep link.** `test_settings_deep_link_first_edit.py::test_the_first_edit_after_a_deep_link_keeps_the_linked_provider` failed once at load ~100 with `'api_key' == 'claude_subscription'` (settings stream).
- It sets `auth.value = "claude_subscription"` and waits one `pilot.pause()` (`:94-95`) before it asserts the staged value.
- The file is a whole-file UI census entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each test waits on the state it then reads (the composed modal's field; the staged auth value), with a bound that fails naming it
- [x] #2 Each passes 20 of 20 runs at `-n 4`, with load recorded, at least ten of them at a one-minute load of 40 or more
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Endpoint test: after `push_screen(modal)`, wait (bounded, named) for the modal's New button to be mounted before pressing it; after New, wait for the template modal to be the screen with `#endpoint-template-name` mounted before querying it.
2. Deep-link test: after `auth.value = "claude_subscription"`, wait (bounded, named) for the staged auth value before asserting it.
3. Show each wait's failure names its state under a temporary break of the condition.
4. 20 runs of each node at `-n 4` with load recorded (ten or more at load >= 40).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Base ae9f9e8d21, 2026-11-10. Test-only. Every run went through run.sh; loads are one-minute averages before and after each run.

**Create-endpoint return.** Each press now waits, bounded at 15 s and naming its state, for its own end condition:
- after the push, the modal's New button mounted;
- after New, the template modal is the screen with its fields composed (`#endpoint-template-create` mounted), before the name and URL are written;
- `_pick_in_pick_mode`: the switcher's Find mounted (TASK-38961's `_mounted_pick_popover`), then the highlighted row is a pickable pair for the current query, then Enter;
- the return to the modal on `custom-ep:report-probe`; the Streaming draft following its Select; `_settled` is bounded by 15 s of wall time instead of 30 pauses.

The pick step was the one that missed. With only the push and template waits converted, the node still failed 4 of 12 runs at load 66-85, with "the settings modal returning on the created endpoint never happened within 15 s". A probe in that failure showed the pick-mode switcher still open, with a pickable highlighted row, Find reading 'Report probe ' and no worker left. A probe before Enter, over 12 later runs (24 picks, all green), found no rows built yet in 9 of the 24 picks. So Enter sometimes landed before the rows were built; `_activate(None)` refuses and nothing is picked. Control: a temporary edit that delayed the pick-mode row build by 1 s turned the node red with the old helper and green with the new one (load 78-89).

**Deep-link first edit.** The auth-source wait reads the staged value until it is `claude_subscription`, then `api_key`, and fails as "the staged auth source never became … within 15 s: <value>". Negative control: staging of `claude_subscription` temporarily skipped -> "the staged auth source never became 'claude_subscription' within 15 s: 'api_key'". In the first campaign this test also failed once (load ~75) at the save step: `assert len(mutations) == 1` read `[]` after `_settle`. The save step now waits for the key field's focus, for Esc to release it, then for the save's config mutation, before `_settle` and the unchanged assertions. Which of the two (Esc, or the save's arrival) was late in that one run was not determined.

**Runs (AC#2).** 20 of 20 at `-n 4`, both endpoint cases and the deep-link test in each run, at load 47-83 (so all 20 at 40 or more). Whole files at `-n 4` (load 56-66): 62 passed, 1 failed. The failure is `test_shared_evidence_change_refreshes_the_console_once`, also red on the clean base worktree `fu-base-r13` with the same command; TASK-35567 owns it.

**Files.** `Tests/UI/test_console_endpoint_discovery.py`, `Tests/UI/test_settings_deep_link_first_edit.py`.
<!-- SECTION:NOTES:END -->
