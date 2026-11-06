---
id: TASK-38992
title: "Mobile serve-wiring relay test is red on dev, because its served-canvas gateway stand-in predates revoke_browser_session"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 19:16
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
- [x] #1 test_serve_wiring_attaches_relay_only_when_broker_exists passes 10 of 10 runs alone on dev
- [x] #2 The test checks that unbinding a served browser revokes that browser session on the served-canvas gateway, and fails under a temporary source edit that drops the revoke call
- [x] #3 Every served-canvas gateway stand-in in Tests/Web_Server supplies the methods serve.py calls on that gateway, or fails at construction and names the method that is missing
<!-- AC:END -->

## Implementation Plan

1. Add one served-canvas gateway stand-in builder for `Tests/Web_Server`: `create_autospec(CanvasGateway, instance=True, spec_set=True)`, so every method `serve.py` can call exists with its real signature. A method the test does not supply raises and names itself when called. Supplying a name `CanvasGateway` lacks fails at construction.
2. Replace the three hand-written stand-ins with it (`test_mobile_service.py`, `test_served_shell_browser.py`, and `test_canvas_kill_switch.py` twice), keeping each test's own behaviour (recorded `aclose` order). Drop the obsolete `mark_browser_session_unavailable`, which `serve.py` no longer calls on this gateway.
3. `test_serve_wiring_attaches_relay_only_when_broker_exists` asserts that unbinding revokes exactly that browser session on the gateway. Show it red under a temporary edit that drops the revoke call, then run it 10 of 10 alone.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** `Tests/served_canvas_gateway_double.py` adds `served_gateway_double(**methods)`, which is `create_autospec(CanvasGateway, instance=True, spec_set=True)`.
- Every public gateway method exists with its real signature, and every call is recorded.
- A method passed in `methods` runs as that method. Any other raises `AssertionError("unexpected served-canvas gateway call: <name>")` when called.
- At construction it AST-scans `serve.py` for every attribute read on `self._served_canvas_gateway` (`served_gateway_calls`). Any of those, or any name passed in `methods`, that `CanvasGateway` lacks raises `AttributeError` naming it.

**Changed stand-ins.** The three hand-written stand-ins in `Tests/Web_Server` now use it:
- `test_mobile_service.py`'s `_RelayWiringHost` supplies `revoke_browser_session`. It no longer supplies `mark_browser_session_unavailable`, which `serve.py` does not call on this gateway.
- `test_served_shell_browser.py`'s `_GatewayStub` supplies `aclose`.
- The two `Gateway` classes in `test_canvas_kill_switch.py` keep their recorded `aclose` order.
- `test_served_tab_isolation.py` already uses the real `CanvasGateway`.

**Test (AC#2).** `test_serve_wiring_attaches_relay_only_when_broker_exists` now asserts that `revoke_browser_session` is not called before the unbind and is called exactly once, with `"browser-1"`, by `unbind_served_browser("browser-1", "child-a")`. It was red under a temporary edit of `serve.py` that dropped the revoke call: `Expected 'revoke_browser_session' to be called once. Called 0 times.`

**Test (AC#3).** `test_served_gateway_double_supplies_what_serve_calls_or_names_the_gap` checks three things:
- the double has every method `serve.py` calls on that gateway;
- an unsupplied method raises, named;
- a name `CanvasGateway` lacks fails at construction, named.

Under a temporary edit that made `serve.py` call `forget_browser_session` instead, every stand-in failed at construction with `AttributeError: CanvasGateway has no method(s): ['forget_browser_session']`: this test and both kill-switch tests.

**Runs (AC#1).** `test_serve_wiring_attaches_relay_only_when_broker_exists` alone passed 10 of 10 (load 52-87). It failed at base in the same session.

The whole file at `-n 4` at load 52-69: 31 passed and 2 failed on this branch; 31 passed and 2 failed at base. Here the two were relay tests that time out on their 5 s waits. At base they were this test and a relay test. Under `--storage-lease-guard`, serially at load 74-85, it gave 33 passed and 1 relay timeout, with no lease finding. `test_canvas_kill_switch.py`, `test_served_shell_browser.py` (Playwright is installed here, so its headless-browser tests ran) and `test_served_tab_isolation.py`: 23 passed serially.

**Files.** `Tests/served_canvas_gateway_double.py` (new), `Tests/Web_Server/test_mobile_service.py`, `Tests/Web_Server/test_canvas_kill_switch.py`, `Tests/Web_Server/test_served_shell_browser.py`.
<!-- SECTION:NOTES:END -->
