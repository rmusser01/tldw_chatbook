---
id: TASK-761
title: Restore Console agent dictionary send integration baseline
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 17:57'
updated_date: '2026-07-27 13:50'
labels:
  - console
  - chat-dictionaries
  - baseline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Console agent send integration contract so a conversation dictionary is applied before the agent bridge receives provider messages, eliminating the deterministic failure inherited from dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent-path Console sends apply the active conversation dictionary before agent dispatch,Provider-path dictionary behavior remains unchanged,The exact agent dictionary integration regression passes offline,Focused Console controller and dictionary tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Investigated before writing any code, per DoD. Production code was already
correct: `_apply_chat_dictionaries` runs in `submit_draft` (and every other
send path) before `_stream_assistant_response` -> `_run_agent_reply` ->
`self._agent_bridge.run_reply`, so `agent_messages = list(provider_messages)`
in `console_chat_controller.py` already carries the substituted text. No
production code was changed.

The deterministic failure was in the test suite: `Tests/UI/test_console_dictionary_send_integration.py::test_native_send_applies_conversation_dictionary_agent_branch`
(the "exact agent dictionary integration regression" test) failed 100% of
the time with `KeyError: 'agent_messages'`. Root cause: its `_fake_run_reply`
double had drifted out of sync with `ConsoleAgentBridge.run_reply`'s real
keyword-only signature -- later Console work (`provider_stream_signals` from
citation-repair streaming, `request_skill_script_confirm` from the
skill-script HITL gate) added new required-by-call kwargs that the fake never
accepted, so the real controller's call raised `TypeError` inside the fake,
which `_run_agent_reply`'s `except Exception` handler swallowed into a
"failed" run -- `captured["agent_messages"]` was never set, and the test's
own assertion then raised a misleading `KeyError` instead of surfacing the
real `TypeError`. Fixed by adding the two missing keyword parameters to the
fake so it mirrors the current bridge contract.

Added a second, stronger regression test,
`test_agent_path_applies_dictionary_before_bridge_sees_messages` in
`Tests/Chat/test_console_chat_controller.py`, that pins ORDER explicitly via
a shared event log (dictionary-applier call vs. bridge call), not just final
content -- a change that called the bridge before the dictionary would fail
this test even if content happened to be correct some other way. Verified it
actually pins the contract: temporarily commented out the
`_apply_chat_dictionaries` call at the `submit_draft` call site (simulating
"not applied on the agent path"), reran, observed
`AssertionError: assert ['bridge_called'] == ['dictionary_applied', 'bridge_called']`,
then restored the file from a backup and confirmed `git diff` against HEAD
was empty.

Files touched (tests only):
- Tests/UI/test_console_dictionary_send_integration.py -- added
  `provider_stream_signals=None` and `request_skill_script_confirm=None` to
  `_fake_run_reply`'s signature.
- Tests/Chat/test_console_chat_controller.py -- added
  `test_agent_path_applies_dictionary_before_bridge_sees_messages`.

Testing: Tests/Chat/test_console_agent_bridge.py,
test_console_agent_swap.py, test_console_chat_controller.py,
test_console_dictionary_application.py, Tests/UI/test_console_dictionary_send_integration.py,
Tests/UI/test_console_native_chat_flow.py -- 484 passed, 1 failed
(`test_console_conversation_browser_search_ignores_stale_results`, unrelated
to chat dictionaries/agent bridge, passes standalone in isolation --
pre-existing flake, not touched by this change).
<!-- SECTION:NOTES:END -->
