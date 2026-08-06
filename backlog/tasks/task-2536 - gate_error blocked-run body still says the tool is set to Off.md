---
id: task-2536
title: gate_error blocked-run body still says the tool is set to Off
status: To Do
assignee: []
created_date: '2026-08-06 09:48'
labels:
  - mcp
  - honesty
dependencies: []
priority: medium
---

## Description

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

## Acceptance Criteria

- [ ] A blocked run whose gate is the synthesized `gate_error` deny (resolver raised)
      does not render body text asserting the tool is set to Off.
- [ ] The blocked-run body and the decision note beneath it no longer contradict each
      other for the `gate_error` case.
- [ ] The real deny-gate short-circuit (tool genuinely set to Off) keeps rendering
      `_TOOL_TEST_BLOCKED_TEXT` unchanged.
- [ ] `test_gate_check_exception_fails_closed` is updated as a named, authorized
      contract change rather than left pinning the false body text.
- [ ] Additive regression test covers the fixed `gate_error` body text directly.
