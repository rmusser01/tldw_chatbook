---
id: TASK-40321
title: "The shared UI test app factory gives the app a SimpleNamespace runtime policy, so skill discovery takes a different path than in the shipped app"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- console
- harness
priority: medium
references:
- Tests/UI/app_factory.py
- tldw_chatbook/runtime_policy/bootstrap.py
- tldw_chatbook/UI/Console_Modules/skill.py
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948: its 100 ms harness had to take a custom path because of this). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Every app_factory-based UI test runs a skill-discovery path the shipped app does not. Off-loop fixes such as TASK-38948's skill discovery are therefore invisible to latency tests built on this factory, and a test can pass or fail for reasons that do not exist in production.

**Evidence.** `Tests/UI/app_factory.py:375-382` `fake_runtime_policy` builds `SimpleNamespace(state=RuntimeSourceState(...), persist=lambda: None)` and patches it over `tldw_chatbook.app.load_runtime_policy_for_app` (`:461-462`), from 913ec3101c (task-1458). The real type is `runtime_policy/bootstrap.py:122` `RuntimePolicyContext` (`__slots__`, constructor `(state, store, *, publish=None)`). Skill discovery's stock-source check needs `type(runtime_policy) is context_type` (`UI/Console_Modules/skill.py:482`, `_CONTEXT_CLASSES` at `:519-528`), so under the fake, discovery stays on the UI loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The shared factory gives the app a real RuntimePolicyContext backed by a sandboxed store, so skill discovery takes the shipped route
- [ ] #2 A test shows discovery runs off the UI loop under the factory and fails if the factory goes back to a stand-in object
- [ ] #3 The UI census files that build on the factory still pass (or each red is classified)
<!-- AC:END -->
