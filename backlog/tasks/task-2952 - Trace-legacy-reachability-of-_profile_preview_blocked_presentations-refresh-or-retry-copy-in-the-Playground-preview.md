---
id: TASK-2952
title: >-
  Trace legacy reachability of _profile_preview_blocked_presentation's
  refresh-or-retry copy in the Playground preview
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 02:09'
updated_date: '2026-08-07 06:06'
labels:
  - ui
  - speech
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SpeechProfileMixin._profile_preview_blocked_presentation (tldw_chatbook/UI/Speech/speech_profile_mixin.py:178-218) returns bounded recovery copy for the adopted-preset preview banner and runs BEFORE the honest no-catalog-check branch that voice-profiles slice 2 task 3 added to _sync_profile_preview_status. Three of its returns say 'Profile preview blocked -- refresh or retry from Voice profiles.' / '...TTS settings changed; refresh models.' for state == "unverified", with no check on the preset's provider class (recovery_action or provider_id) -- exactly the false recovery promise this whole slice exists to remove for legacy providers. It was found and left unfixed by task 3's implementer (per its report), who explicitly did not trace whether any of its three unverified-returning branches (expected_revision is None; current_revision != expected_revision; not self._catalog_generation_allowed) is reachable for a legacy-provider preset in practice, as opposed to being audio.cpp-only like several other sites this slice found and left alone with a documented reachability trace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every one of the three 'unverified'-returning branches in _profile_preview_blocked_presentation is traced to determine whether it is reachable when preset.provider_id is a legacy (non-audio_cpp) provider
- [x] #2 If a branch is legacy-reachable, its copy is made honest for legacy providers (no refresh/retry promise), consistent with the vocabulary this slice established (preset_has_no_catalog_check helper or equivalent), with a regression test pinning the legacy-vs-audio_cpp divergence
- [x] #3 If a branch is proven legacy-unreachable, that is documented in a code comment with the trace evidence, matching the pattern task 1 used for _PROFILE_UNVERIFIED_COPY
- [x] #4 The finding and its resolution (fixed or documented-unreachable, per branch) are recorded in the task's Implementation Notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Static trace of the three branches in _profile_preview_blocked_presentation against the registry/catalog/preset lifecycle to determine which app-level events reach each with a legacy preset adopted.
2. Empirical probes (Tests/UI/test_task2952_probe.py, throwaway) driving the real SpeechPlaygroundPane via FakeTTSService to confirm/refute each branch's reachability and whether refresh genuinely resolves it, for both audio_cpp and legacy (openai) presets.
3. Per branch: if honest for both classes, add a code comment citing the trace; if legacy gets a false promise, fix the copy using preset_has_no_catalog_check and add a TDD regression test in the permanent suite (test_speech_playground_pane.py, matching slice-2 task 3 conventions).
4. Delete or fold the throwaway probe file once conclusions are captured as permanent pins.
5. Run gates: speech pane/profile test files, ruff, repo-wide --collect-only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Traced all three "unverified"-styled returns in _profile_preview_blocked_presentation against the registry/catalog/preset lifecycle, confirmed live via Tests/UI/test_speech_playground_pane.py (mutation-checked). Result: none needed a copy fix -- no legacy-specific false promise was found in any of the three branches (AC #2's fix condition never triggered), but each conclusion is pinned as a regression test per AC #2's spirit and AC #3.

Branch 1, `expected_revision is None`: LEGACY-UNREACHABLE (and audio.cpp-unreachable -- not a class distinction). `_load_provider_catalog_worker` sets `_profile_configuration_revision` synchronously, before any `await`, the moment it targets the preset's own provider; production always targets it on the very first load since both provider classes are registered unconditionally (`adapter_bootstrap.build_default_tts_service`, `legacy_bridge.legacy_provider_specs`) and `_profile_preset` is fixed at construction, never reassigned. Pinned: test_adopted_preset_preview_never_shows_a_null_revision_refresh.

Branch 2, `current_revision != expected_revision`: HONEST FOR BOTH CLASSES. Fires on genuine config drift (the provider's registry configuration_revision bumping after the preview loaded) -- pure registry bookkeeping, not catalog content, so refresh re-syncs it for legacy exactly as it does for audio.cpp. Confirmed live: bumping a legacy provider's revision produces the "TTS settings changed; refresh models" banner, and pressing Refresh resolves it into the honest per-class copy. Pinned: test_adopted_preset_preview_revision_mismatch_refresh_recovers_for_both_classes.

Branch 3, `not self._catalog_generation_allowed`: HONEST FOR BOTH CLASSES (no legacy-specific divergence). The one live path found is `_load_provider_voices_worker`'s TTSRegistryClosedError (non-reconfiguring) branch, which forces the flag False without an _apply_controls recompute; the generic-exception and reconfiguring branches both self-heal. TTSRegistryClosedError comes from TTSAdapterRegistry._closed, a registry-wide one-way seal -- identical machinery for every provider class. Pinned as a cross-class symmetry check: test_adopted_preset_preview_registry_closed_during_voice_fetch_is_symmetric asserts audio_cpp and legacy produce the byte-identical banner from the same failure.

Method: empirical probes (throwaway Tests/UI/test_task2952_probe.py, deleted) drove the real SpeechPlaygroundPane through FakeTTSService across full mount lifecycles and targeted fault injection, instrumented with a spy on _profile_preview_blocked_presentation recording state at every call. Conclusions were converted into permanent pins in test_speech_playground_pane.py and a trace docstring on _profile_preview_blocked_presentation itself (matching the _PROFILE_UNVERIFIED_COPY comment pattern). Mutation-tested both directions: disabling the revision assignment failed 4/5 new tests (branches 1+2); injecting an artificial legacy-only copy into branch 3 failed the symmetry pin -- both reverted after confirming.

Files: tldw_chatbook/UI/Speech/speech_profile_mixin.py (trace docstring only, no behavior change), Tests/UI/test_speech_playground_pane.py (+3 pins, +TTSRegistryClosedError import).

Gates: targeted speech pane/profile suites (170 passed, 2 pre-existing xfail unrelated to this change) + Tests/UI/test_stts_profile_library.py + test_studio_tts_preferences.py (90 passed); ruff clean on touched files; repo-wide --collect-only (31734 tests collected, 0 errors).
<!-- SECTION:NOTES:END -->
