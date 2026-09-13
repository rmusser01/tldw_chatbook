---
id: TASK-32498
title: 'Fix custom-ep parent identity flattening conflating same-family built-in child routes (ADR-147 follow-up)'
status: Done
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
- [x] #1 Child explicitly routed to built-in provider F while parent runs on a custom-ep with family key F actually streams from built-in F's configured endpoint
- [x] #2 Run-row snapshot matches where the bytes actually went in that scenario
- [x] #3 Inherit children of custom-ep parents keep current (correct) behavior — parent resolution stream
- [x] #4 Regression test with custom-ep parent + same-family built-in child route
<!-- AC:END -->

## Implementation Notes

Fixed in the qodo PR-2651 High commit on feat/agent-provider-routing:

- `ConsoleProviderResolution.selected_provider` carries the raw
  `custom-ep:<slug>` id (populated only for registry-endpoint selections;
  plain/alias selections keep `""`, so their behavior is byte-identical).
- The streaming adapter's same-target decision keys on `selected_provider`
  before falling back to execution_key/provider: a child routed to the
  built-in family of a custom-ep parent re-resolves independently (AC#1);
  an inheriting child names the slug and overlays the parent resolution
  (AC#3).
- Bridge plan + `run_turn(parent_raw_provider=...)` thread the raw identity
  into `AgentService._run_one`, whose spawn closure passes it as
  `parent_provider` to `resolve_spawn_target`: inheriting children snapshot
  slug + registry base_url + entry params (AC#2); continuations re-freeze
  the now-correct snapshot.
- Regression tests: gateway selected_provider population (3), adapter
  same-family reroute + inherit overlay (2), spawn integration snapshot +
  legacy fallback (2) (AC#4).

Files: console_provider_gateway.py, console_agent_bridge.py,
agent_service.py, test_console_provider_gateway.py,
test_console_agent_bridge.py, test_agent_routing_integration.py.
