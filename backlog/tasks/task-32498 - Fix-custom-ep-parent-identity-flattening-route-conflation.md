---
id: TASK-32498
title: 'Fix custom-ep parent identity flattening conflating same-family built-in child routes (ADR-147 follow-up)'
status: To Do
assignee: []
created_date: '2026-09-12 07:55'
labels:
  - agents
  - console
  - llm-routing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32477 (agent provider routing, final-review finding I-2). The gateway flattens custom-ep resolutions to the family execution key (console_provider_gateway.py:1648-1653: provider=identity.execution_key; family map custom_endpoint_registry.py:293-304). So when the PARENT runs on custom-ep:X whose family key is F, both the resolver's parent_provider and the adapter's parent_endpoint (console_agent_bridge.py:2046-2050) see "F". A child explicitly routed (preset/default/allowlisted override) to the BUILT-IN provider F then fails the rerouted check (console_agent_bridge.py:2057) and silently streams from the parent's custom-ep resolution while its run-row snapshot claims resolved_provider=F with base_url NULL — violating the branch's no-silent-fallback rule and snapshot honesty. Narrow and fail-safe (user's own endpoint, child's own params still apply). Invisible to current tests because fakes use execution_key=selection.provider, preserving the slug, and all routing tests use built-in parents. Fix direction: the adapter/bridge must compare the child's target against the SELECTION-level provider id (retain it alongside the flattened execution key), not the flattened one. Add a regression test with a custom-ep parent and a same-family built-in child route.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Child explicitly routed to built-in provider F while parent runs on a custom-ep with family key F actually streams from built-in F's configured endpoint
- [ ] #2 Run-row snapshot matches where the bytes actually went in that scenario
- [ ] #3 Inherit children of custom-ep parents keep current (correct) behavior — parent resolution stream
- [ ] #4 Regression test with custom-ep parent + same-family built-in child route
<!-- AC:END -->
