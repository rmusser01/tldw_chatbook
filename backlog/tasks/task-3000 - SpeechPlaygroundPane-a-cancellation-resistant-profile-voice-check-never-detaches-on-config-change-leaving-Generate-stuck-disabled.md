---
id: TASK-3000
title: >-
  SpeechPlaygroundPane: a cancellation-resistant profile voice check never
  detaches on config change, leaving Generate stuck disabled
status: To Do
assignee: []
created_date: '2026-08-07 05:40'
labels:
  - ui
  - speech
  - tech-debt
dependencies:
  - TASK-2951
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fail-closed availability gap in the exact-profile generate-readiness path, distinct from the two fail-open CRITICALs task-2951 fixed (a keyboard bypass of a disabled Generate button, and an enabled Generate button that silently did nothing). This one leaves the button wrongly stuck DISABLED rather than wrongly reachable. Found and ported as a probe (against both pre-fix and post-fix task-2951 code, same result both times -- pre-existing, NOT introduced or worsened by task-2951's readiness-gate unification) during task-2951's own re-review: test_configuration_change_detaches_cancellation_resistant_profile_voice_gate, retired with TTSPlaygroundWidget's test file and never re-ported. Scenario: an exact profile preset's voice validation is in flight and its underlying provider request is cancellation-resistant (ignores cancellation, keeps running); a provider-configuration change arrives (mark_provider_configuration_changed) followed by a failed catalog reload (_load_provider_catalog raising). Nothing in mark_provider_configuration_changed, the failed-reload exception path, or _catalog_failure clears _profile_voice_validation_token -- so it never returns to None, availability never settles to a real state (unverified/available/unavailable), and Generate stays disabled indefinitely even though the provider is otherwise healthy and the user has no way to recover short of leaving and re-entering the Playground. The retired TTSPlaygroundWidget's equivalent path detached the token on exactly this sequence, settled availability at 'unverified', and allowed one warned exact attempt (matching the same one-warned-attempt contract task-2951's CRITICAL 2 fix restored for the plain naturally-stale-catalog case) -- this is that same contract, just for the cancellation-resistant-in-flight-validation entry point specifically.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 _profile_voice_validation_token detaches (returns to None) when a provider-configuration change is followed by a failed catalog reload while an exact profile's voice validation is in flight
- [ ] #2 Profile availability settles to an honest state (unverified, matching the retired widget's behavior for this sequence) instead of staying stuck mid-validation
- [ ] #3 Generate becomes enabled again and one warned exact attempt is allowed, matching the one-warned-attempt contract already restored for the naturally-stale-catalog case (task-2951 CRITICAL 2)
- [ ] #4 test_configuration_change_detaches_cancellation_resistant_profile_voice_gate is ported from the retired widget's test suite as a real (non-xfail) assertion against SpeechPlaygroundPane and passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed from task-2951's re-review (second TTSPlaygroundWidget deletion pass). Confirmed via an independent probe against both pre-fix and post-fix task-2951 code with identical results, so this is pre-existing and outside that task's own scope, not a regression it introduced. Retired widget reference: STTS_Window.py (pre-deletion) mark_provider_configuration_changed / the TTSProviderReconfiguringError-and-preset branch inside the voice-discovery exception handler, and the widget's own test_configuration_change_detaches_cancellation_resistant_profile_voice_gate.
<!-- SECTION:NOTES:END -->
