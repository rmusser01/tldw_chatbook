---
id: TASK-2536
title: gate_error blocked-run body still says the tool is set to Off
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 09:48'
updated_date: '2026-08-06 18:11'
labels:
  - mcp
  - honesty
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR-T3 Task 3 fixed the *decision note* for the Hub's synthetic `gate_error` gate
(`_resolve_test_gate()`'s fail-closed deny when the permission resolver itself raises):
it now reads the honest `_UNKNOWN_ORIGIN_SENTENCE` ("Permission state could not be
resolved.") instead of "This tool is set to Off." — that was task-2270's rider.

But the LOUD body text above that honest sentence is a different string, and it was
not touched. `_TOOL_TEST_BLOCKED_TEXT` (`mcp_workbench.py:304`, "Blocked — this tool
is set to Off in Permissions.") is shared verbatim by two call sites: the real
deny-gate short-circuit (where the tool genuinely is set to Off — the text is true
there) and the synthesized `gate_error` path (`mcp_workbench.py:3329`, where the
resolver failed and the tool's actual state was never determined — the text is false
there). So a `gate_error` blocked run now shows two lines one after another that
directly contradict each other: the loud body asserting "this tool is set to Off",
and the honest decision note beneath it saying the state could not be resolved.
Fixing the decision note alone did not close the rider — it moved the lie one line
up.

`Tests/UI/test_mcp_workbench.py::test_gate_check_exception_fails_closed` currently
pins `_TOOL_TEST_BLOCKED_TEXT` for the `gate_error` case, so this needs an authorized
change to that pin, not just a code fix.

(Escalated from PR-T3 Task 3's review; filed here so it isn't lost even if the
whole-branch review wave picks it up first.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A blocked run whose gate is the synthesized `gate_error` deny (resolver raised)
      does not render body text asserting the tool is set to Off.
- [x] #2 The blocked-run body and the decision note beneath it no longer contradict each
      other for the `gate_error` case.
- [x] #3 The real deny-gate short-circuit (tool genuinely set to Off) keeps rendering
      `_TOOL_TEST_BLOCKED_TEXT` unchanged.
- [x] #4 `test_gate_check_exception_fails_closed` is updated as a named, authorized
      contract change rather than left pinning the false body text.
- [x] #5 Additive regression test covers the fixed `gate_error` body text directly.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a gate_error-aware blocked body text constant in mcp_workbench.py.
2. Make the deny short-circuit choose between the genuine-deny text and the new honest text based on gate.origin.
3. Suppress the decision note for the gate_error case (body already carries the reason), mirroring _run_tool_test()'s own refusal precedent.
4. Update the two affected tests (one pre-authorized contract change, one made coherent with the new design).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added _TOOL_TEST_BLOCKED_UNKNOWN_TEXT (mcp_workbench.py) as the honest gate_error body copy; the deny short-circuit in on_mcp_inspector_tool_test_requested() now branches on gate.origin=="gate_error" to pick it over _TOOL_TEST_BLOCKED_TEXT, and passes decision_note=None for that case (body already states the reason, mirroring _run_tool_test()'s own no-double-say precedent for refusals) instead of calling _decision_note() again. _decision_note() itself is unchanged and still covered directly by its own unit test. Coverage came from updating the two existing end-to-end tests to assert the new honest text/empty note (not a brand-new test function) -- test_gate_check_exception_fails_closed is the brief's one pre-authorized pin change; test_gate_check_exception_decision_note_uses_honest_unresolved_sentence was also updated (not on the pre-authorized list, but explicitly named and directed by the brief's own no-double-say design) and flagged in the fix-round report. Verified RED against the unmodified code before implementing. Files: tldw_chatbook/UI/MCP_Modules/mcp_workbench.py, Tests/UI/test_mcp_workbench.py.
<!-- SECTION:NOTES:END -->
