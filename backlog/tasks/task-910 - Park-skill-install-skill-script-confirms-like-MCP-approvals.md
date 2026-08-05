---
id: TASK-910
title: Park skill-install/skill-script confirms like MCP approvals
status: Done
assignee: []
created_date: '2026-07-27 03:55'
updated_date: '2026-07-27 18:40'
labels:
  - console
  - agents
  - approvals
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The parallel-agents train (per-session runs) made background runs able to raise skill-install and skill-script confirm cards. Both bridges remain single-slot: a background run's confirm card mounts OVER the currently viewed tab, and switching sessions denies the pending confirm (deny-on-any-switch). MCP approvals got full park/badge/toast/round-identity treatment; the two skill-confirm bridges (request_skill_install_confirm, request_skill_script_confirm in console_chat_controller.py) did not. Fail-closed today, but the interruption and spurious denies degrade the multi-agent UX.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A background session's skill-install/skill-script confirm does not mount over the viewed tab; it parks with the needs-approval marker and a single toast, mounting on visit.
- [x] #2 Switching sessions no longer denies another session's pending confirm; only that session's own stop/shutdown does.
- [x] #3 Confirm decisions carry round identity so a decision cannot resolve a different session's confirm.
- [x] #4 Never auto-approve; timeout behavior unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read task-910 brief + request_mcp_approvals/resolve_pending_approval/switch_session end-to-end as the reference park/mount/retain/badge/toast/round-id implementation.
2. Convert request_skill_install_confirm from a single global event/decision slot to a round registry (_pending_skill_install_rounds, keyed by request_id, mirroring _pending_skill_script_rounds) so two different sessions can each have their own concurrent install confirm.
3. Add park/mount/retain to both request_skill_install_confirm and request_skill_script_confirm: is_parked gate, set_run_pending_approval badge, retained payload maps (_parked_skill_install_payloads/_parked_skill_script_payloads) populated for every session-attributed round (mounted or parked), and reuse the existing park_pending_approval/_park_console_approval UI bridge (same badge+toast machinery/copy as MCP) instead of inventing new UI.
4. Give resolve_pending_skill_install the same round-identity contract resolve_pending_skill_script already has (strict request_id match, silent no-op on mismatch/None).
5. Remove _deny_pending_skill_install_on_context_change/_deny_pending_skill_script_on_context_change and their calls from switch_session; replace with re-derive-from-parked-map calls (mirroring the MCP approval re-derive), and add the same re-derive to new_session/close_session.
6. Update SkillInstallConfirmCard to echo back request_id on InstallDecided (mirroring SkillScriptConfirmCard), and update ChatScreen's handler to thread it through.
7. TDD: rewrite/extend controller-level tests (install + script) for park/mount/re-mount/cross-session-stop-safety/shutdown-still-denies/stale-round-id-safe-no-op, and add UI-level end-to-end park tests to Tests/UI/test_console_parallel_runs.py mirroring the MCP badge+toast+mount test.
8. Run the skill-confirm + mcp-approval suites + parallel-runs + native-chat-flow gates; fix regressions; update the backlog task file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Converted both skill-confirm bridges (request_skill_install_confirm, request_skill_script_confirm) to the same park/mount/retain/badge/toast/round-id contract request_mcp_approvals already had.

Approach:
- request_skill_install_confirm: replaced the single global _pending_skill_install_event/_pending_skill_install_decision pair with a round registry (_pending_skill_install_rounds, keyed by a fresh request_id, guarded by a new _pending_skill_install_lock) so two different sessions can each carry a concurrent install confirm without clobbering each other -- mirrors _pending_skill_script_rounds' existing task-581 design.
- Both bridges gained an is_parked gate (session_id present and != store.active_session_id): a parked round sets the shared badge (set_run_pending_approval) and fires the toast via the SAME UI bridge request_mcp_approvals already used (park_pending_approval / ChatScreen._park_console_approval) -- no new UI surface, no new toast copy. A mounted-or-parked round's payload is retained unconditionally in a new per-bridge map (_parked_skill_install_payloads / _parked_skill_script_payloads) so switch_session/new_session/close_session can re-derive the mounted card on every activation, exactly like _parked_approval_payloads.
- resolve_pending_skill_install now takes request_id and strict-matches the armed round (mirrors resolve_pending_skill_script's existing contract) -- a mismatched/unknown/omitted id is a silent no-op, never an auto-approve.
- Removed _deny_pending_skill_install_on_context_change / _deny_pending_skill_script_on_context_change and their unconditional calls from switch_session (the pre-parking "any switch denies" behavior); switch_session/new_session/close_session now re-derive from the parked-payload maps instead. Cancellation continues to flow exclusively through _is_session_cancelled (owning session's own cancel event) or the never-reset _shutdown_requested -- unchanged from before this task.
- Teardown clears the mounted card only when this round's session is still the viewed one AND no sibling round for the SAME session remains armed, reconciling task-581's original same-session multi-round guard with the new cross-session parking case (a parked/background round's teardown must never blank the viewed session's own card).
- SkillInstallConfirmCard.InstallDecided now echoes request_id (mirroring SkillScriptConfirmCard.ScriptDecided); ChatScreen.handle_console_skill_install_decided threads it through.

Testing: TDD -- extended/rewrote controller-level tests in Tests/UI/test_console_skill_install_confirm.py and Tests/Chat/test_console_skill_script_confirm.py (park-for-background-session, mount-on-visit, switch-away-and-back re-mounts, unrelated-session-stop does not deny, owning-session cancel/shutdown still denies, stale/unknown/omitted round-id is a safe no-op) plus two new UI-level end-to-end tests in Tests/UI/test_console_parallel_runs.py driving the REAL controller bridge through a genuine worker thread and asserting on the real #chat-skill-install-card/#chat-skill-script-card widgets (park -> badge -> one toast -> mount on visit -> re-mount on revisit -> resolve). Removed the two pre-parking "any switch/context-change denies" tests (test_switch_session_denies_a_pending_skill_script_confirm, test_context_change_denies_a_pending_confirm, test_context_change_denies_every_armed_round) and replaced with shutdown-based equivalents, since AC#2 makes their old assertion false.

Verification: ran the skill-confirm + mcp-approval suites + Tests/UI/test_console_parallel_runs.py together (255 passed, 3 pre-existing failures unrelated to this task -- confirmed identical on unmodified HEAD via git stash) and Tests/UI/test_console_native_chat_flow.py (192 passed, 18 failed; confirmed all 18 also fail on unmodified HEAD, which additionally has one more flaky failure not seen on this branch -- no regression).

Deviation: none from the AC contract. Reused the MCP toast copy/helper verbatim (no new copy needed, since the skill-confirm cards are not a genuinely different surface).

Files: tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/Widgets/Chat_Widgets/skill_install_confirm_card.py, Tests/UI/test_console_skill_install_confirm.py, Tests/Chat/test_console_skill_script_confirm.py, Tests/Chat/test_skill_script_concurrent_confirms.py, Tests/UI/test_console_parallel_runs.py.
<!-- SECTION:NOTES:END -->
