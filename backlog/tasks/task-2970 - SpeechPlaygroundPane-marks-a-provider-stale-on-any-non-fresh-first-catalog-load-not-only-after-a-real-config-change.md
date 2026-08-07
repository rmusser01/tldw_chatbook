---
id: TASK-2970
title: >-
  SpeechPlaygroundPane marks a provider stale on any non-fresh first catalog
  load, not only after a real config change
status: To Do
assignee: []
created_date: '2026-08-07 04:19'
labels:
  - ui
  - speech
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A first-ever (not-yet-configuration-changed) audio.cpp/OpenAI/etc. catalog load whose reported health.fresh is False gets the generic 'settings changed; refresh models' recovery copy in the Playground's provider-status line, instead of the accurate, state-specific copy (e.g. 'settings are being applied' for reconfiguring, 'catalog is stale' for a naturally-stale available catalog). Found and isolated during TASK-2951's coverage-porting pass: SpeechCatalogMixin._load_provider_catalog_worker adds the provider to _stale_providers whenever the freshly-fetched catalog itself reports health.fresh is False, unconditionally, on every catalog application -- not only ones that followed a real provider-configuration change. _catalog_health_copy checks _stale_providers before health.state, so the copy implies a change that never happened. Confirmed absent from the retired TTSPlaygroundWidget: its equivalent success path (STTS_Window.py, pre-deletion, verified via git show HEAD) did an unconditional self._stale_providers.discard(provider_id) with no health.fresh branch at all -- the divergence is new, introduced when the mixin-based rebuild independently reimplemented this path. _catalog_health_copy itself is byte-for-byte identical between the two; only what populates _stale_providers diverged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A first (or otherwise non-configuration-change) catalog load whose health.fresh is False shows the accurate state-specific recovery copy (reconfiguring -> settings are being applied; a naturally-stale available catalog -> catalog is stale), not the generic settings-changed copy
- [ ] #2 The two xfail(strict=True) parametrizations in Tests/UI/test_speech_playground_pane_lifecycle.py::test_audio_cpp_health_states_use_fixed_safe_recovery_copy (health2, health3) are un-xfailed and pass
- [ ] #3 SpeechCatalogMixin._load_provider_catalog_worker only adds a provider to _stale_providers when a catalog is genuinely superseding a previously-fresher one (e.g. a real configuration-revision change), not on every catalog application regardless of history
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed from TASK-2951 (second TTSPlaygroundWidget deletion pass); full evidence trail (both code paths diffed, git show HEAD comparison) lives in the xfail reasons in Tests/UI/test_speech_playground_pane_lifecycle.py.
<!-- SECTION:NOTES:END -->
