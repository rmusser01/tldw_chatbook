---
id: TASK-34867
title: Console send permanently blocked after project-instruction preflight refusal
status: Done
assignee: []
created_date: '2026-09-12 19:21'
updated_date: '2026-09-12 21:30'
labels:
  - console
  - bug
  - project-instructions
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a fresh profile (any default-workspace Console session) the first send raises the project-instruction preflight with zero eligible bindings; the default workspace can never hold runtime bindings by task-069 design so 'Disable' is the only choice. Choosing Disable (or cancelling) returns ConsoleSubmitResult(accepted=False) from _run_agent_reply WITHOUT terminalizing the run state or resolving the pending assistant placeholder; the send gate then sticks at 'Preparing... / Wait for this turn to be accepted before queueing a message.' forever and the session composer is bricked. Found via live headless smoke test on feat/custom-endpoint-registry; reproduced in persona and generic sessions alike (counterfactual: not persona-related). Root cause: the three preflight early-return branches in console_chat_controller.py (~11552 and 11570-11577) skip the cleanup every other refusal path performs (_set_run_state(ConsoleRunState.blocked(...)) plus placeholder resolution). The default-workspace binding veto in registry_service.list_runtime_bindings is a deliberate invariant and stays. To be fixed on branch feat/custom-endpoint-registry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After preflight Disable the run state reaches a terminal blocked state and the send button returns to Send;After preflight cancel the same release behavior holds;The pending assistant placeholder no longer lingers as pending after refusal;A subsequent send in the same session reaches the provider;The queued-dispatch refusal path (DISPATCH_REFUSED) still returns its claim and pauses as before;Regression tests cover the disable and cancel refusal branches;Live headless verification: fresh profile send then Disable then composer unblocks then second send reaches loopback provider
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find how the existing stop/cancel path resolves a pending assistant placeholder and reuse that store API\n2. Add a controller helper that terminalizes the run state (ConsoleRunState.blocked) plus resolves the placeholder plus returns ConsoleSubmitResult(accepted=False)\n3. Call it from the three preflight early-return branches in _run_agent_reply (~11552 disable-callback-missing, ~11570 disable, ~11575 cancel)\n4. Regression tests: disable and cancel branches terminalize the run, placeholder not pending, next send allowed; queued DISPATCH_REFUSED path unchanged\n5. Live headless verification of the original fresh-profile stall repro\nADR required: no\nADR path: N/A (bug fix inside task-069 machinery; backlog/decisions/069-console-project-instruction-local-state-and-preflight.md linked)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The three preflight early-return branches in _run_agent_reply (setup-callback-missing, disable, cancel) now call a new controller helper _refuse_turn_for_project_setup, which resolves the pending assistant placeholder via _ensure_assistant_placeholder + store.mark_message_failed (mirroring the RUN_CANCELLED path), terminalizes the run with _set_run_state(ConsoleRunState.blocked(visible_copy)), and returns the same ConsoleSubmitResult(accepted=False, should_clear_draft=False, visible_copy=...) strings as before — no user-visible copy changes. The default-workspace binding veto in registry_service.list_runtime_bindings is untouched (task-069 invariant). Tests: 3 new refusal-probe tests in Tests/Chat/test_console_agent_project_instructions.py (disable/cancel/no-callback) assert terminalization, placeholder resolution, and that a following send is admitted; all 3 fail when the fix is reverted (mutation-checked). Adjacent suites green: full project-instructions file 31 passed, queue suites 59 passed (DISPATCH_REFUSED unchanged), provider-continuation 41 passed. Live headless verification on feat/custom-endpoint-registry: real TldwCli + real on-disk DB + loopback OpenAI server — fresh-profile send raised ProjectInstructionSetupModal with the gate held (occupies_slot=True, zero provider posts), Disable released the gate (tooltip back to 'Type a message to send.'), and the second send reached POST /v1/chat/completions; a persona session reproduced the same modal+release. Probe file removed after the run; real user config untouched (md5 verified). ADR required: no — bug fix inside the task-069 preflight machinery (backlog/decisions/069-console-project-instruction-local-state-and-preflight.md). Files: tldw_chatbook/Chat/console_chat_controller.py, Tests/Chat/test_console_agent_project_instructions.py.
<!-- SECTION:NOTES:END -->
