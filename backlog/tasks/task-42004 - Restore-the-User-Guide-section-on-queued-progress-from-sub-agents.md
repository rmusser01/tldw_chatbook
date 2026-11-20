---
id: TASK-42004
title: "Restore the User Guide section on queued progress from sub-agents"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:34
labels:
- follow-up
- docs
- console
priority: low
references:
- Docs/User_Guide/console/agent-runs-and-tools.md
- tldw_chatbook/Agents/fleet_message_tools.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r17dc-rev1 (round 17). Squash 6a6242ce3a removed "Queued progress from sub-agents" (`report_to_supervisor`, Agent → Progress: N queued, Discard selected), added by 827b622638. The feature still ships (`Agents/fleet_message_tools.py`, the `console_agent_progress_modal.py` Queued progress view). The newer "Queued child progress" section describes the same queue in its saved-chat form.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The section is restored and reconciled with "Queued child progress", every sentence checked against current code and cited in the notes
<!-- AC:END -->

## Implementation Plan

1. Recover "Queued progress from sub-agents" from `git show 6a6242ce3a^:Docs/User_Guide/console/agent-runs-and-tools.md`.
2. Check each sentence against `Agents/fleet_message_tools.py`, `Agents/fleet_messages.py`, `console_agent_progress_modal.py`, the left rail's Progress button and the navigation counts; drop or rewrite what durable progress (7212692c8b) and peer messages changed.
3. Merge it with the newer "Queued child progress" section into one section under the restored heading, keeping the newer section's durable-queue and wake facts and fixing its "Read progress" label, which no code shows.
4. Run the doc tests that read this page; preflight before the final commit.

## Implementation Notes

- **One section.** "Queued progress from sub-agents" (827b622638, removed by 6a6242ce3a) and "Queued child progress" (7212692c8b, written after durable progress shipped) described the same queue. The restored heading now holds both, in the newer section's place after "Steering a running sub-agent", so it reads straight into "Messages between live siblings". No link pointed at `#queued-child-progress`.
- **What changed from the old text, and why:**
  - "Queued means accepted into this process's inbox", "a session-only queue, with no durable inbox", "closing its last open session or restarting the runtime loses the queue": false for saved chats since durable progress (ADR-199; `MessageInbox._append_locked` writes through `FleetProgressRepository`, `Agents/fleet_messages.py:389`). Kept only for temporary chats.
  - "Children cannot address each other": false since `send_to_peer` (`fleet_message_tools.py` `SEND_TO_PEER_SCHEMA`); now points at the sibling section.
  - "Progress does not wake an idle supervisor": false with `[agents] autowake_enabled` on; replaced by the newer section's wake paragraph.
  - View title "Queued progress (this session)" is now "Queued progress" (`console_agent_progress_modal.py:67`).
  - The newer section's **Read progress** label does not exist in code and never did (`git log -S 'Read progress' -- tldw_chatbook`: no commits); the entry point is the Agent section's **Progress: N queued** button.
  - Dropped, not verified: "if the same saved conversation is open in more than one session, those sessions share queued progress". Dropped as implementation detail: the newer section's "navigation remains responsive while a saved report change finishes during chat close, restore or app shutdown" (pinned by `Tests/UI/test_console_durable_progress.py::test_durable_discard_contention_keeps_mounted_reads_and_heartbeat_responsive`).
- **Sources for each kept claim** (base d85e90bc9e):
  - Fleet children get `report_to_supervisor`, the primary gets `read_agent_messages`: `Agents/agent_native_execution.py:1259` (`progress_sender_factory` for fleet children), `Agents/agent_service.py:5510-5545`.
  - 2,000 characters, up to four reports per collect, collection removes them, receipt "queued", report and read instructions: `fleet_messages.py:27` `MAX_MESSAGE_CHARS`, `:37` `MAX_COLLECTED`; `fleet_message_tools.py:33-66` (`REPORT_INSTRUCTIONS`, `READ_INSTRUCTIONS`, both schemas), `report()` receipt.
  - No interruption of `wait_agents`: `AgentRunTools.wait_agents`, `Agents/agent_run_tools.py:852`, blocks on children only; reports only enter the inbox.
  - Entry point **Progress: N queued** in the Agent rail section: `UI/Console_Modules/left_rail.py:2857`, `:2918`; wired at `UI/Screens/chat_screen.py:14886` to `open_fleet_progress` (`UI/Console_Modules/agent.py:1829`).
  - View copy and behaviour: arrows/Space status line `:87`, "Inspecting report N · selected for discard / not selected" `:234`, full body via `_literal` `:22`, 0.5 s refresh `:135`, "Discard selected (N)" `:255`, discard result "Existing history copies remain." `:311`, unavailable/queue-full copy `:166-168`, discard keeps running after close (thread worker; `on_unmount` `:137` only bumps the generation); scope copy "Inspection does not collect reports or wake an idle supervisor" `:69` (all `Widgets/Console/console_agent_progress_modal.py`). Owner check that makes a stale view unavailable: `require_current_owner`, `agent.py:1859`.
  - Discard removes only selected ids and refunds no lifetime sends: `MessageInbox.discard`, `fleet_messages.py:457`; lifetime 32 `:31`; per-child/per-chat/runtime limits `:29`, `:33`, `:35`; committed view during a save, `snapshot` `:445`.
  - Navigation **Progress: N** and **[N Sub-Agents]**: `Widgets/Console/console_workspace_tree.py:385`, `console_workspace_context.py:711-713`, `console_character_context.py:160`.
  - Agent mode off, pruned children, concurrent arrival, lifetime not refunded, attention unchanged, unsaved navigation counts, collection refreshing an open view, Save keeping an open view's checks: `Tests/UI/test_console_agent_progress.py` (`test_inspect_select_concurrent_arrival_discard_preserves_exact_ids_and_lifetime`, `test_unsaved_mode_off_progress_entry_and_captured_owner_survive_navigation`, `test_navigation_keeps_progress_separate_from_historical_counts_for_unsaved_sessions`, `test_model_collection_refreshes_open_modal_without_discard_or_wake`, `test_console_progress_navigation_pruned_access_and_actual_paint`, `test_real_live_report_save_preserves_open_selection_and_navigation`).
  - Saved/temporary chats, reopen, close revocation, capacity deferral, chain filter, checkpoint: `Tests/Agents/test_durable_fleet_messages.py` (`test_saved_reopen_preserves_fifo_and_revokes_old_capabilities`, `test_explicit_save_commits_chat_and_pending_reports_atomically`, `test_runtime_capacity_defers_saved_bodies_without_losing_navigation_counts`, `test_collection_checkpoint_ambiguity_stops_receipts_and_replay`), `Tests/Agents/test_fleet_messages.py` (`test_automatic_chain_filter_skips_unknown_and_foreign_fifo`, `test_two_phase_close_revokes_immediately_and_cannot_remove_replacement`), `Tests/UI/test_console_durable_progress.py::test_explicit_progress_retry_loads_deferred_saved_reports_after_capacity_frees`. Storage takes no trace-capture setting: `_append_locked` writes unconditionally.
  - Wakes: gated by `autowake_enabled` in `_priority_allows` (`Chat/console_fleet_wake.py:54`, `:805`); notice lists ids only, `compose_progress_wake_notice` `:87`; shared budget, coalescing, priority, slots, cancel on read, once-only, restart review: `Tests/Chat/test_progress_wakes.py` (`test_committed_progress_wakes_once_with_ids_and_fresh_read_notice`, `test_progress_and_completion_coalesce_but_collection_before_claim_cancels_hint`, `test_progress_waits_for_manual_priority_and_reserved_slot`, `test_fourth_progress_report_stays_pending_when_shared_generations_exhaust`, `test_completed_temporary_wake_save_then_restart_requires_review`, `test_restart_progress_attempt_is_review_only_and_report_stays_readable`).
  - Copies outside the queue: `READ_AGENT_MESSAGES_SCHEMA` ("provider history/private continuation may retain copies"); ADR-063 file exists at `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`.
- **Checks:** `Tests/Docs` + `Tests/MCP/test_mcp_documentation_contract.py` through the wrapper, `-n 4`: 119 passed.
- File: `Docs/User_Guide/console/agent-runs-and-tools.md`.
