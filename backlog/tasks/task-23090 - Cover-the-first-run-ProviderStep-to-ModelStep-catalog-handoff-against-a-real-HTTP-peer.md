---
id: TASK-23090
title: >-
  Cover the first-run ProviderStep-to-ModelStep catalog handoff against a real
  HTTP peer
status: Done
assignee:
  - '@claude'
created_date: '2026-08-28 15:22'
updated_date: '2026-08-28 21:29'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find why resolve_provider_list_key reports 'missing' in the fresh harness.\n2. Make the provider resolvable the way a real config does.\n3. Drive ProviderStep to ModelStep against the real HTTP peer and assert the full catalog reaches the picker.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Blocker in the task body was misdiagnosed: the 'missing_endpoint' was the [providers] branch, but the cause was not the config file. The app's LocalLLMProviderCatalogService loader reads self.providers_models, snapshotted at app init, and a fresh test app has an empty snapshot -- so discovery reported 'no matching provider model list' before issuing any request no matter what was written to [providers]. Setting app.providers_models = get_cli_providers_and_models() after the config write clears it. Test drives ProviderStep (llama_cpp, typed endpoint) to ModelStep against a real threaded HTTP peer serving 128 models. Pins two defects, each confirmed by reverting it: the wizard's typed-result bound (revert to 100 -> discovery result rejected) and the discovery module's identity encoding (remove header -> invalid_response). Explicitly does NOT pin settings_endpoint_probe's encoding fix, since this path never presses Test; the sibling test covers that. Also recorded: the picker renders models[:20] by design, so the catalog-bound fix took the step from zero models to a real list, not from 100 to 128 visible.
<!-- SECTION:NOTES:END -->
