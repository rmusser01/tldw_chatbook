---
id: TASK-32275
title: First agent send after restart stalls ~70 s with no activity copy
status: Done
assignee: []
created_date: '2026-09-10 19:10'
updated_date: '2026-09-10 21:35'
labels:
  - console
  - agents
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a warm profile, the first Console agent send after an app restart showed an empty assistant row and 'Run: Agent running.' for about 70 seconds before the provider was called; a later send in the same instance took about 5 seconds. Nothing on screen said what the app was waiting on. The built-in MCP server spawn/discovery is the suspected cause; not traced. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the delay is identified and either removed or bounded by a visible timeout with a stated reason.
- [x] #2 While pre-provider setup runs, the assistant row or status strip says what is happening (for example connecting tools) instead of staying blank.
- [x] #3 A diagnostic event or test pins the pre-provider setup time budget.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on a warm profile with timing marks between "console agent
   reply start" and "Routing to endpoint".
2. Locate the wait with evidence (log marks plus a native stack sample).
3. Write failing tests for the bound and for the missing activity copy.
4. Bound the located wait; add the fifth activity state.
5. Re-run the live check and report the new gap.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Traced, bounded and made visible. The suspected cause (built-in MCP server spawn / catalog composition) was wrong: instrumented live, compose_catalog took 15 ms and the local provider 5 ms. The wait is the lazy Personal Context bootstrap, resolved on the FIRST agent send of a process via ConsoleChatController._personal_context_service -> app.get_personal_context_service() -> PersonalContextRepository.__init__, whose key protector reads the OS credential store; a sample(1) of the stuck process put 1686/1686 samples in SecItemAdd -> SecKeychainItemCreateFromContent -> makeLoginAuthUI -> AuthorizationCopyRights, the macOS Keychain authorization UI, waited on synchronously with no timeout (cold profile 3.7 s of a 5 s gap; warm restart still waiting after 6 minutes). Fix: _personal_context_service now wraps its asyncio.to_thread in asyncio.wait_for(..., CONSOLE_PRE_PROVIDER_SETUP_BUDGET_SECONDS) (10 s, constant comment states the reason and the 3.7 s healthy measurement it is sized against) and logs one WARNING naming the budget when it fires -- the function already promised 'personalization never blocks chat', but that promise was enforced against exceptions only and a hang is not an exception; giving up yields None, exactly what every other failure there yields, so the turn runs without profile tools instead of not at all, and the abandoned worker thread caches the service for a later send if the OS ever answers. Visibility: AgentLiveSnapshot gained a 'setup' status plus setup_started_at, ConsoleAgentBridge.begin_setup_phase/end_setup_phase mark it, live_snapshot short-circuits to it, and console_turn_activity_text renders 'Connecting tools... - <elapsed>' as its fifth state (docstring table updated); _run_agent_reply runs the measured setup span inside _pre_provider_setup_phase, an async context manager that always clears the mark, starting at the personal-context resolution rather than at catalog composition because that is where the time is (20 ms vs 3.7 s). Live before/after on the same wedged scratch profile: >6 min with a blank assistant row -> 11 s, with the row reading 'Connecting tools... - 3s', '6s', '9s' and then the approval card at ~12 s. Modified: Chat/console_chat_controller.py, Chat/console_agent_bridge.py, UI/Console_Modules/agent.py, Docs/User_Guide/console/agent-runs-and-tools.md, Docs/security/production-diagnostic-inventory.json, and the three matching test files.
<!-- SECTION:NOTES:END -->
