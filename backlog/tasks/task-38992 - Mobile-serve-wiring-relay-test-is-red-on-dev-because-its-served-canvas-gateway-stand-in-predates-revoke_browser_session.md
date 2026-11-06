---
id: TASK-38992
title: "Mobile serve-wiring relay test is red on dev because its served-canvas gateway stand-in predates revoke_browser_session"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- tests
- web-server
priority: low
references:
- Tests/Web_Server/test_mobile_service.py
- tldw_chatbook/Web_Server/serve.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05 (load 7-10).

`Tests/Web_Server/test_mobile_service.py::test_serve_wiring_attaches_relay_only_when_broker_exists` (`:2418`) fails 3 of 3 runs alone. The failure is `AttributeError: 'types.SimpleNamespace' object has no attribute 'revoke_browser_session'`, raised at `tldw_chatbook/Web_Server/serve.py:1814` from the test's `host.unbind_served_browser("browser-1", "child-a")` (`:2448`).

The cause is test drift:
- The test's `_RelayWiringHost` gives `_served_canvas_gateway` a `types.SimpleNamespace` that has only `mark_browser_session_unavailable` (`:2402-2404`).
- df22692d70 (2026-10-27, "isolate served tabs") made `unbind_served_browser` (`serve.py:1808-1814`) also call `self._served_canvas_gateway.revoke_browser_session(browser_session_id)`.
- The test file last changed on 2026-10-17, before that commit.

No PR lane runs `test_mobile_service.py`; it is in neither the census nor any workflow target. No branch due to merge before this one touches the file or `serve.py`. TASK-38972 covers a different defect in the same file's shutdown path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_serve_wiring_attaches_relay_only_when_broker_exists passes 10 of 10 runs alone on dev
- [ ] #2 The test checks that unbinding a served browser revokes that browser session on the served-canvas gateway, and fails under a temporary source edit that drops the revoke call
- [ ] #3 Every served-canvas gateway stand-in in Tests/Web_Server supplies the methods serve.py calls on that gateway, or fails at construction and names the method that is missing
<!-- AC:END -->
