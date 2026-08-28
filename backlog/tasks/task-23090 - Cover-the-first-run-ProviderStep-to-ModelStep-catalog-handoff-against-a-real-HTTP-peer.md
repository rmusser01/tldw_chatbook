---
id: TASK-23090
title: >-
  Cover the first-run ProviderStep-to-ModelStep catalog handoff against a real
  HTTP peer
status: To Do
assignee: []
created_date: '2026-08-28 15:22'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2158's regression tests are unit-level: they mock the transport or call _model_ids_from_discovery_result directly. One live-contract test now drives the Test button against a real gzip-capable local server, but nothing exercises the full ProviderStep-to-ModelStep handoff end to end, so a future change that drops or trims a production-sized catalog between discovery and the rendered picker would still pass CI. This is the exact gap that let three defects ship and required a manual live walk to find.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A live-contract test drives the wizard from provider selection to a rendered Model-step picker against a real local HTTP peer
- [ ] #2 The test asserts the full production-sized catalog reaches the picker, including the last (newest) model id
- [ ] #3 The test fails if a bound between discovery and the picker trims or rejects the catalog
<!-- AC:END -->

## Known blocker (measured, 2026-08-28)

A first attempt got the real HTTP peer and the wizard talking -- the probe
request was observed carrying `Accept-Encoding: identity` -- but discovery
itself short-circuited before any request:

    DBG staged: {'api_settings': {'llama_cpp': {'api_url':
      'http://127.0.0.1:61462/v1/chat/completions', 'credential_source':
      'draft', 'api_key': '...'}}}
    DBG prov outcomes: {...: ('error', 'missing_endpoint')}
    DBG hits: []

`missing_endpoint` here is the `resolve_provider_list_key(...) == "missing"`
branch ("No matching provider model list exists in [providers]"), not a bad
URL -- `_resolve_endpoint` accepts all three endpoint forms in isolation.
Seeding `[providers]` in the fresh test config did not clear it, so the
harness needs the provider-catalog wiring understood before this test can
assert on the picker. That investigation is the work, not an afterthought.
