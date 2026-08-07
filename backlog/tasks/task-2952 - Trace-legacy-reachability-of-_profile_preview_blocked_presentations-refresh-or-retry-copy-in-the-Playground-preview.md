---
id: TASK-2952
title: >-
  Trace legacy reachability of _profile_preview_blocked_presentation's
  refresh-or-retry copy in the Playground preview
status: To Do
assignee: []
created_date: '2026-08-07 02:09'
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
- [ ] #1 Every one of the three 'unverified'-returning branches in _profile_preview_blocked_presentation is traced to determine whether it is reachable when preset.provider_id is a legacy (non-audio_cpp) provider
- [ ] #2 If a branch is legacy-reachable, its copy is made honest for legacy providers (no refresh/retry promise), consistent with the vocabulary this slice established (preset_has_no_catalog_check helper or equivalent), with a regression test pinning the legacy-vs-audio_cpp divergence
- [ ] #3 If a branch is proven legacy-unreachable, that is documented in a code comment with the trace evidence, matching the pattern task 1 used for _PROFILE_UNVERIFIED_COPY
- [ ] #4 The finding and its resolution (fixed or documented-unreachable, per branch) are recorded in the task's Implementation Notes
<!-- AC:END -->
