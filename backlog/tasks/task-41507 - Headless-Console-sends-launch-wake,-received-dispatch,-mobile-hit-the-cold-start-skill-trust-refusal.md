---
id: TASK-41507
title: "Headless Console sends (launch-wake, received dispatch, mobile) hit the cold-start skill trust refusal"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 08:36
labels:
- follow-up
- console
- skills
- headless
priority: high
references:
- tldw_chatbook/Chat/console_received_dispatch.py
- tldw_chatbook/Web_Server/mobile_service.py
- tldw_chatbook/Chat/console_chat_controller.py
dependencies:
- TASK-41138
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-41138 (round 15) made a cold-Home Buddy send build the skill trust service before capture, because a send otherwise refuses with `RecoveryRequired('console_capture_source_not_ready')`. The same refusal very likely reaches the other headless senders, which call `controller.submit_draft` without building the trust service: `console_received_dispatch.py:793` and `:802`, `Web_Server/mobile_service.py:1162`, and launch-wake deliveries from a cold Console (found in the testinfra fix round; untested).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A launch-wake delivery, a received dispatch and a mobile send into a cold Console each reach the provider (or a visible, correct refusal), never `console_capture_source_not_ready`
- [x] #2 The trust service is built in the shared submit path or each caller, with one decision recorded in the notes
- [x] #3 The launch-wake and received-dispatch paths each have a test that starts cold and is red before the fix; the mobile path has a cold test showing a send reaches the provider (no defect on that path, reason recorded)
<!-- AC:END -->

## Implementation Plan

1. Reproduce each named path from a cold start and record where it refuses (RecoveryRequired spy on the turn's exception).
2. Decide where the trust service is built (shared submit path vs each caller) and record why.
3. Move Buddy's TASK-41138 helper into Chat/ and call it from the shared path and from any caller whose capture runs before that path.
4. One cold test per path, red before the fix (negative controls by temporarily removing each call), green after.
5. Covering suites base vs head, ratchets (module size), preflight.

## Implementation Notes

**Where each path stood at base (reproduced, not assumed).** Refusal reasons came from a temporary exception spy (reverted).
- **Mobile: never refused.** `MobileChatService` runs its own engine. Its app is `HeadlessAppContext`, which has no skills scope or local skills service, so `_ready_attribute` has no trust slot to refuse on. `Tests/Web_Server/test_mobile_service.py` builds a fresh engine per test and sends to the provider (for example `test_agent_round_trip_with_approval`): 50 passed at base and after.
- **Launch wake: refused.** Launching into Home (no Console, trust unbuilt), `ConsoleFleetWakeCoordinator._deliver` called the runtime submitter `_submit_fleet_wake`, which captures synchronously: `resolve_runtime_turn_configuration_snapshot` -> `remember_pending_mcp_capture` -> `_source_references` -> `_ready_attribute`. That raised `console_capture_source_not_ready` 12 times in one run, and no wake reached the provider. The existing launch tests start on Library, which builds the trust as it starts, so they never saw this (lesson added).
- **Received dispatch: refuses only through its capture.** `console_received_dispatch.py`'s submits (`:793`, `:802`) pass the configuration captured at acceptance, so they never capture or refuse themselves. The capture comes first: the wake's above, a goal iteration's `_submit_goal_iteration`, or a queued prompt's on a mounted Console. A launch wake is a received dispatch: its turn runs through `_run_custodied_turn` -> `submit_draft(configuration=...)` at `:802`.
- **Direct submit: refused.** A `controller.submit_draft` without a configuration captures in `_submit_draft_body` and refused the same way while the trust was unbuilt. That covers Buddy (TASK-41138), the wake's no-runtime branch, and a submit right after Console mounts, before skill discovery has built the trust. `Tests/UI/test_console_launch_wake.py`'s seeding send was red at base for this reason: 3 of 11 nodes, the file on no lane.

**Decision (AC#2): the shared submit path.** `ConsoleChatController.submit_draft` builds the trust whenever it captures its own configuration (no configuration and no resumed preparation), before `_submit_draft_lifecycle` and inside its diagnostic scope. It calls `publish_cold_skill_trust(app)`, which is Buddy's TASK-41138 helper moved to `Chat/console_configuration_preparation.py`, next to `_ready_attribute`. The helper now builds exactly when `_ready_attribute` would refuse, and returns at once for any app without a local skills service (mobile, test doubles). It consults no skill-listing policy (see the review round below).
- Why the shared path: every sender that captures its own configuration goes through that one entry. Buddy's own call is deleted (the shared path covers it), and callers added later are covered without a copy.
- One extra call site: the fleet wake's runtime submitter captures synchronously before any submit, so `_deliver` awaits the same helper before it chooses a chain or claims anything. A build failure there is retried after the coordinator's retry delay (review round below).
- Mobile needs nothing.

**Tests (AC#3).**
- New `Tests/UI/test_console_launch_wake.py::test_a_launch_into_home_builds_the_skill_trust_its_wake_captures`. It launches into Home, asserts the app's trust slot is empty, then asserts the owed wake reaches the provider with the child's result and the run is stamped delivered. Red/green:
  - `_deliver`'s call replaced by `pass` (submit-path build kept): red, "the cold launch delivered no wake turn", with the 12 `console_capture_source_not_ready` above.
  - With the submit-path build also removed: red at the seeding send (`console_capture_source_not_ready`).
  - All four product files at base: red, "the cold launch delivered no wake turn".
  - Both present: green.
- New `test_the_cold_trust_build_runs_only_where_capture_refuses` (same file, 4 cases, in-process). A ready slot, or no factory, builds nothing; an empty slot with a factory builds, with or without a denied local skill listing; no app is a no-op. At base it cannot import the helper.
- The same test covers received dispatch (the wake's custodied turn).
- `Tests/UI/test_buddy_conversation_modal.py::test_cold_home_saved_buddy_bootstraps_only_after_explicit_open` now runs through the shared path: red with the submit-path build removed (the send failed in `controller_submit`), green with it.
- Mobile has no cold refusal, so no mobile test can be red before a fix. **Coordinator ruling (2026-11-18):** AC#3 was reworded (red-before for the launch-wake and received-dispatch paths; for mobile, a cold test showing the send reaches the provider, with the reason recorded). That test already exists and runs on a PR lane (`dreams-boundary`): `Tests/Web_Server/test_mobile_service.py::test_agent_round_trip_with_approval`. It builds a fresh `MobileChatService` over an empty data folder with the `AgentModeGateway` double (`Tests/console_provider_doubles.py`, no transport), sends, and asserts the provider saw the tool result and the final answer reached the transcript. Passed before and after this change (the file: 50 passed at base, and in the 189-pass covering run after). Reason it cannot be red: the engine's `HeadlessAppContext` has no skills scope or local skills service, so `_ready_attribute` has no trust slot to refuse on.

**Runs.**
- Before the review round, at `-n 4` (load ~17): launch wake, Buddy modal, fleet wake retry, fleet wake, wrapped skill collaborators, cold trust preparation, mobile service and the cardless compaction file gave 189 passed.
- `Tests/UI/test_console_launch_wake.py` whole at `-n 4`: 11 passed after the fix (3 red at base), then 12 with the new test.
- The covering suites at `-n 4` (load ~21-33): fleet wake, fleet wake retry, configuration preparation, cold trust preparation, wrapped skill collaborators, Buddy modal, launch wake and mobile service gave 207 passed and 1 failed. The failure, `test_console_configuration_preparation.py::test_ready_local_skill_capture_runs_original_body_on_worker_and_matches_snapshot` (`mcp_definition_maximum` empty), also fails with the base product files: 2 of 2.
- `Tests/Architecture/test_module_size_ratchet.py`: the controller row rises 28761 -> 28764, the exact growth, with a dated note (owner rule for fixes over a size ratchet).
- `Tests/Architecture` serially: 1471 passed and 5 failed: the Widgets app-lookup pin, `test_no_new_upward_import` (`config.py` -> `Chat.console_provider_endpoints`), the diagnostic inventory's metadata-only check, and two timer-path inventory checks. All 5 fail the same with the base product files, and none touches this change.

**Docs.** The User Guide (`console/agent-runs-and-tools.md`) already says an owed wake is delivered at the next launch without opening Console. This makes that true when the app starts on Home, so no doc change.

**Lane.** `Tests/UI/test_console_launch_wake.py` is a whole UI census entry at the tail. Measured before the in-process case was added: 12 passed in 148.6 s serially with the lane's flags (load 14-17). The 4 in-process cases add about 2 s.

Files: `tldw_chatbook/Chat/console_configuration_preparation.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/Chat/console_fleet_wake.py`, `tldw_chatbook/UI/Navigation/buddy_conversation.py`, `Tests/UI/test_console_launch_wake.py`, `Tests/Architecture/test_module_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md`.

**Review r16co-rev1 (fix round 1).**
- **Critical: the policy gate is removed.** The scope-policy check this branch had added before the build refused cold sends that work at base. A server runtime source denies `skills.context.list.local`, yet the capture never asks that policy (it reads `local._visible_records()`), and the composer's preparation builds this same service without it. The reviewer's denial plugin: the cold Buddy test passed at dev and failed at the gated head with `PolicyDeniedError` (Buddy showed its generic "The reply could not finish"), and the wake looped (5,239 `_deliver` calls in 22 s). So the earlier note that a denied policy "refuses the send with the policy's own error" described a regression; it was wrong.
  - The helper test's deny case is now "a denied listing still builds".
  - `test_a_launch_into_home_builds_the_skill_trust_its_wake_captures` gains `[True]`: after seeding, `ServicePolicyEnforcer.require_allowed` denies only `skills.context.list.local`. The test first proves the denial is live on the launch app's own scope, then expects the wake delivered.
  - Red with the gate put back: `[True]` "the cold launch delivered no wake turn", with "wake delivery failed (exception_type=PolicyDeniedError)" filling about 780 KB of the run's output (roughly 6,000 lines). The helper test's two build cases also fail (`['skills.context.list.local', 'build'] != ['build']`, and a raised denial). Green without it.
- **Important: a delivery that fails before choosing a chain backs off.** Before, `_deliver`'s `except Exception` "paused" it as `_pause(cid, None, "interrupted_work", review=True)`. With no chain that saved nothing and blocked nothing, and it cleared `_retry_after[cid]`, so the `finally`'s `retry_soon()` re-delivered at once. The `ledger is None` branch had the same shape (`"history_unavailable"`).
  - Both now call `_back_off(cid)`, which sets `_retry_after[cid] = now + RETRY_DELAY_SECONDS`. Failures after a chain is chosen keep their durable `interrupted_work` pause.
  - New `test_a_wake_that_fails_before_its_claim_retries_after_the_delay` (same file) forces `publish_cold_skill_trust` to raise on a cold launch. Over a 2.5 s window it requires 2 to 4 tries, at least 0.9 x the delay apart, and nothing at the provider. Red with the old `except` path: "220 tries in 2.5 s". Green after.
- **Runs (fix round 1).**
  - Covering set at `-n 4` (load 29-43): launch wake, Buddy modal, fleet wake retry, fleet wake, wrapped skill collaborators, cold trust preparation, mobile service, cardless compaction holds, automatic wake recovery and the automatic-work pause UI file gave 212 passed and 13 failed. All 13 were the pause UI file failing with the local `RecoveryRequired('raw_source_selection_changed')`; that file is on no lane and passes 14 of 14 under a temporary bootstrap-profile plugin (removed).
  - With the reviewer's denial plugins on the cold Buddy node, the launch-wake file and fleet wake retry:
    - `r16co_rev1_plugin` (deny everywhere): 21 passed and 1 failed at `-n 4`, `test_a_launch_built_controller_is_not_sticky_when_console_opens`. It then passed 2 of 2 serially under the same plugin and the whole file passed 18 of 18 twice more at `-n 4`, so it is not reproduced.
    - `r16co_rev1_plugin2` (deny only inside `_deliver`): 22 passed.
    - `_deliver` ran at most 3 times in any node, the 3 in the forced-failure backoff test.
  - Census time: the launch-wake file, now 18 nodes, took 208.4 s serially with the UI lane's flags at load 25-36.
