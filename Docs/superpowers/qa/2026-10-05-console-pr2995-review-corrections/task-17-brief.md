### Task 17: Integrate latest-dev command timing, origin and captured-draft fixes

ADR required: no new ADR
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/219-console-chat-destinations-and-bounded-starts.md; backlog/decisions/220-console-human-decision-coordination-ownership.md.
Reason: compose landed TASK-33622.16 with the approved native/runtime/decision ownership, with no authority, schema, dependency, loading or UI-token boundary change.

One fresh implementer owns source/index/HEAD through a clean handoff. Read own-SDD task-17-latest-dev-preflight.md and task-17-latest-dev-preflight-selection.json for exact incoming declarations, composition candidates, 19-path hashes, protected owners, source-bound QA pins and selected IDs. Published source BASE06cfe2f6a30236dcd8f81893ebf49bc3d0b78036 descends from previous dev a78a9a900b4901e33c031f830dd2d80224d5147d; selected incoming dev8c4dfe59a243ce0cec8e131aff3935646c64b298 adds10commits/19paths. Root commits requirement metadata before dispatch. Do not read other plans' SDD directories or change the shared human checkout. No children, external replies/push/arming/merge or branch cleanup.

- [ ] Pin root metadata BASE and recovery ref/bundle; rebase once onto selected8c4dfe59, preserving all incoming commits and feature commits. If remote dev advances, report its actual delta without expanding or silently retargeting. Preserve any unexpected historical replay conflict and return its exact hunk; no blanket ours/theirs.
- [ ] Verify16 incoming files equal selected-dev whole-file bytes. Compose exactly wiring.build_console_controllers skill append kwargs/session_id forwarding, the three disjoint changed ChatScreen methods and production-diagnostic-inventory command_handoff row. Verify selected sketches/hashes and exact upstream declaration AST carry. Retain every other feature method/helper, including literal native automatic start, accepted receipt/physical custody, runtime Close/source fencing, pending handoff edits/clears and the Task16 typed-answer marker/adapter.
- [ ] Run only the22 exact IDs below, in the selection's four groups, on the actual integrated source. Use shared Python3.12 at /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python, WT PYTHONPATH, canonical private pytest profiles, fresh basetemps, -p no:randomly and existing300s timeout. Record actual collection/IDs/full argv/source hashes/complete stdout+stderr/log/JUnit/exits/warnings. Freeze any failure before proposing a same-node comparison or source repair. No original58/35/61/27/cohort/full-owner replay, full suite, installs, blanket bootstrap plugin, new skip/XFAIL, warning suppression or time/budget increases.
- [ ] Retain controller29301/store22344/interrupt6479/compaction4185/screen25218lines+759methods and ready1033/preimport557+425347LOC+135111bytes/app686/CSS608090 limits. Actual composed screen25192/759 has26line slack; no cap row change is justified. Unchanged controllers/store/interrupt/compaction and loading/runtime/import/timer/CSS/route owners carry by exact hashes. Two helpers stay lazy, so no new loading/payload node is selected. Historical35loading and actual Task16 payload557/415370/127527 remain source-specific evidence.
- [ ] Preserve all tracked unowned/QA bytes, original1983 QA digest and original2118-entry ZIP at63166118bytes/SHA256acecbdfb6ddbafe6df679c39f9f137f2bfb49a1f87b62556e1345021f8384c84. Preserve the existing527 public follow-up files and all previous manifests/rulings/review packages byte-for-byte. Run fatal Ruff on actual incoming Python paths, scoped added-hunk formatter assessment and whitespace; retain inherited formatting/warnings rather than reflowing owners.
- [ ] Freeze task-17-report.md plus compact additive task-17-safe-evidence manifest/source+QA carry maps. Keep reports/maps/receipts exact without copying earlier bulk archives. Commit only actual needed composition overlays; no empty source commit. Return clean source/index/HEAD and closed owned processes. Root performs one independent scoped integration review before Task18 Qodo repairs, any separately justified Stop correction, one additive publication and fresh current-head Qodo/CI/PerfGuard/latest-dev/normal merge gates.

**Selected exact nodes (22):**

command_origin_and_exact_wiring:

- Tests/UI/test_console_command_origin_chat.py::test_a_named_system_prompt_never_lands_on_the_chat_switched_to[enter]
- Tests/UI/test_console_command_origin_chat.py::test_a_named_system_prompt_never_lands_on_the_chat_switched_to[send-button]
- Tests/UI/test_console_command_origin_chat.py::test_a_named_system_prompt_takes_only_the_draft_its_send_captured[appended]
- Tests/UI/test_console_command_origin_chat.py::test_a_named_system_prompt_takes_only_the_draft_its_send_captured[replaced]
- Tests/UI/test_console_command_origin_chat.py::test_a_command_answers_in_the_chat_it_was_sent_from[doctor]
- Tests/UI/test_console_command_origin_chat.py::test_a_command_answers_in_the_chat_it_was_sent_from[skills]
- Tests/UI/test_console_command_origin_chat.py::test_a_command_answers_in_the_chat_it_was_sent_from[fewer-permission-prompts]
- Tests/UI/test_console_command_origin_chat.py::test_a_command_never_starts_in_a_chat_it_was_not_sent_from

command_timing_cancellation_and_draft:

- Tests/UI/test_console_video_send_freeze.py::test_the_cost_confirm_answers_its_keys_and_escape_starts_nothing[lazy-enter]
- Tests/UI/test_console_video_send_freeze.py::test_the_cost_confirm_answers_its_keys_and_escape_starts_nothing[lazy-send-button]
- Tests/UI/test_console_video_send_freeze.py::test_the_cost_confirm_answers_its_keys_and_escape_starts_nothing[eager-enter]
- Tests/UI/test_console_video_send_freeze.py::test_the_cost_confirm_answers_its_keys_and_escape_starts_nothing[eager-send-button]
- Tests/UI/test_console_video_send_freeze.py::test_another_command_runs_while_a_generation_is_in_flight
- Tests/UI/test_console_video_send_freeze.py::test_a_command_whose_modal_wait_is_cancelled_ends_quietly
- Tests/UI/test_console_video_send_freeze.py::test_stop_keeps_a_draft_typed_during_the_generation[enter]
- Tests/UI/test_console_video_send_freeze.py::test_stop_keeps_a_draft_typed_during_the_generation[send-button]
- Tests/UI/test_console_video_send_freeze.py::test_a_failed_generation_never_writes_into_the_chat_switched_to[typed]
- Tests/UI/test_console_video_send_freeze.py::test_a_failed_image_batch_still_offers_a_command_its_take_left_behind

changed_visible_send_plain_answer_control:

- Tests/UI/test_console_ask_user_typed_answers.py::test_typed_answer_resolves_a_real_round_through_the_real_card_and_composer

composed_static_owners:

- Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/chat_screen.py]
- Tests/Architecture/test_screen_size_ratchet.py::test_budget_is_not_left_slack_after_a_wave[tldw_chatbook/UI/Screens/chat_screen.py]
- Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged


### Task17 extension: Record the reviewed five existing cleanup diagnostics

ADR required: no
ADR path: N/A for an inventory-only correction under existing persistent diagnostic governance.
Reason: the selected diagnostic node reveals unchanged Task15 source has11 calls while its committed row still records6; existing statement review proves exactly five safe additions.

- [ ] Resume the same Task17 implementer after root metadata handoff. Initial21pass/1fail report is frozen at task-17-report-before-inventory-fix.md; original brief at task-17-brief-before-inventory-fix.md; original task-17-safe-evidence/manifest.json and all64 artifacts remain byte-identical. Map that manifest's report SHA explicitly to the frozen initial report without rewriting history. No21 passing-case replay.
- [ ] Read task-17-diagnostic-statements-review.txt: existing --statements --since100fa9d819 proves five added warning calls with fixed phases restriction, restriction_fallback, absence, abandonment and run_state plus type(exception).__name__. No removed/reworded/captured-exception/private-content/path/URL/sink change. Verify this exact drift before modifying metadata; unexpected drift returns to root.
- [ ] Use the existing scripts/check_persistent_diagnostic_inventory.py --write only after that review. Assert the sole JSON data change is the console_chat_start.py owner row call_count6→11 and diagnostic_digestec602b5bd3b0fafe72cc→568a6855dc033a46aebc; derived TASK492 summary1437→1442 is not a stored waiver. Every other row/sink/privacy/method/threshold and exact upstream command_handoff row stays unchanged. No production or test edit and no inventory-check bypass.
- [ ] Run only Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged with unchanged canonical profile/Python/PYTHONPATH/-p no:randomly/fresh basetemp/300s bounds, preserving original failure and all21 passes at54a. Retain complete actual argv/source/outputs/XML/exit; source whitespace and exact unowned/QA/ZIP carry. Freeze a separate compact task-17-diagnostic-fix-safe-evidence/manifest; append report with initial byte-exact prefix and transparent initial-report alias. Commit only the reviewed inventory row and return clean source ownership/closed processes. Independent Task17 scoped review uses the full original BASE through final source, including this bounded extension.


