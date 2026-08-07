---
id: TASK-3000
title: >-
  SpeechPlaygroundPane: a cancellation-resistant profile voice check never
  detaches on config change, leaving Generate stuck disabled
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 05:40'
updated_date: '2026-08-07 13:09'
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
- [x] #1 _profile_voice_validation_token detaches (returns to None) when a provider-configuration change is followed by a failed catalog reload while an exact profile's voice validation is in flight
- [x] #2 Profile availability settles to an honest state (unverified, matching the retired widget's behavior for this sequence) instead of staying stuck mid-validation
- [x] #3 Generate becomes enabled again and one warned exact attempt is allowed, matching the one-warned-attempt contract already restored for the naturally-stale-catalog case (task-2951 CRITICAL 2)
- [x] #4 test_configuration_change_detaches_cancellation_resistant_profile_voice_gate is ported from the retired widget's test suite as a real (non-xfail) assertion against SpeechPlaygroundPane and passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Port test_configuration_change_detaches_cancellation_resistant_profile_voice_gate from the retired widget's pre-deletion test file (git show <sha>~1:Tests/UI/test_stts_playground_audio_cpp.py) into test_speech_playground_pane_lifecycle.py, targeting SpeechPlaygroundPane/_PaneHost with the FakeTTSService's existing voice_started_by_request/voice_gates/voice_ignore_cancellation machinery. Run it to confirm genuine RED against current code (button stuck disabled, token never clears).
2. In SpeechCatalogMixin.mark_provider_configuration_changed, port the retired widget's token-detach block: when the currently in-flight _profile_voice_validation_token belongs to the provider whose configuration just changed, clear it (set to None) immediately, mirroring the retired widget's own mark_provider_configuration_changed verbatim (git show <sha>~1:tldw_chatbook/UI/STTS_Window.py).
3. Re-run the ported test to confirm GREEN -- reuse the existing _catalog_failure preset branch (already settles availability to "unverified" and computes generation_allowed via _project_profile_preset_controls, per task-2951's CRITICAL-2 fix) rather than adding a parallel settling path.
4. Interaction/mutation check against TASK-2970: re-run 2970's tests after 3000's change and vice versa; revert each fix independently and confirm only its own test(s) fail.
5. Run the full gate list, ruff + format --check on touched files, repo-wide --collect-only; update the task file (AC boxes, notes) and mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported test_configuration_change_detaches_cancellation_resistant_profile_voice_gate
from the retired widget's pre-deletion test file (git show
f560217fb~1:Tests/UI/test_stts_playground_audio_cpp.py) into
Tests/UI/test_speech_playground_pane_lifecycle.py, targeting SpeechPlaygroundPane/
_PaneHost via the existing FakeTTSService voice_started_by_request/voice_gates/
voice_ignore_cancellation machinery (Tests/UI/speech_playground_fixtures.py already
supported cancellation-resistant per-request-key gating, no fixture changes needed).
Confirmed genuine RED against current code first: _profile_voice_validation_token
stayed set (a real CatalogRequestToken, not None) and the assertion failed exactly as
predicted by tracing _generation_readiness_error's preset branch, which blocks
unconditionally on that token being non-None.

Fix: ported the retired widget's own mark_provider_configuration_changed token-detach
block verbatim (git show f560217fb~1:tldw_chatbook/UI/STTS_Window.py) -- when the
currently in-flight _profile_voice_validation_token belongs to the provider whose
configuration just changed, clear it to None immediately, before any reload even
starts. This alone was sufficient: the existing _catalog_failure preset branch
(task-2951's CRITICAL-2 fix) already settles _profile_effective_availability to
"unverified" and computes _catalog_generation_allowed via
_project_profile_preset_controls on the subsequent failed reload, so once the token
stops blocking _generation_readiness_error, that already-correct settling path takes
over on its own -- no parallel settling path was added, per the task's own framing
("reuse that path, don't invent a parallel one").

Interaction/mutation checks against TASK-2970 (both touch mark_provider_configuration_
changed / the same success-path region of _load_provider_catalog_worker):
- Mutation: temporarily removed the token-detach block -- confirmed ONLY this ported
  test failed; TASK-2970's health2/health3 stayed green. Restored.
- Interaction: re-ran TASK-2970's tests after this fix (green) and this test after
  TASK-2970's fix (green); both fixes coexist in the full targeted suite with no
  cross-contamination.

Gates: targeted Speech/TTS suite (Tests/UI/test_speech_playground_pane_lifecycle.py,
test_speech_playground_pane.py, test_stts_playground_catalog.py,
Tests/TTS/test_stts_audio_cpp_generation.py) 205 passed (both fixes applied); full
Tests/UI/test_speech_*.py + test_stts_*.py + Tests/TTS/ + Tests/TTS_Events/ sweep:
2860 passed, 16 skipped (optional deps), 1 pre-existing failure unrelated to this
change (test_first_time_audio_cpp_setup_lab_generation_and_console_handoff, already
documented pre-existing in task-2951's own notes). Repo-wide --collect-only: 31874
collected, 0 errors. ruff check + format --check clean on touched files.

Files: tldw_chatbook/UI/Speech/speech_catalog_mixin.py (mark_provider_configuration_
changed), Tests/UI/test_speech_playground_pane_lifecycle.py (new ported test).

### Review round: trigger-wording correction + two regression-guard tests

Coordinator review approved the fix as correct but flagged two things.

First, a wording correction (no code change): this task's own description and my
Implementation Notes above frame the trigger as "config change, followed by a failed
catalog reload" -- narrower than what the fix actually depends on. The token detaches
the moment mark_provider_configuration_changed runs for its own provider, full stop;
no reload -- successful, failed, or never attempted at all -- has to follow. The
failed reload in the ported test exists only to demonstrate that Generate *recovers*
afterward (via the pre-existing _catalog_failure settling path), which is a separate
guarantee from the detach itself. This isn't an ad-hoc relaxation: the event stream
feeding mark_provider_configuration_changed is already scoped upstream to genuinely
*applied* changes -- on_stts_provider_configuration_changed (stts_events.py
:2092-2100) forwards every STTSProviderConfigurationChanged event verbatim, and that
event is only ever posted by _post_applied_settings_changes (stts_events.py
:1584-1598) for providers whose settings-save status == "applied".

Second, two coverage-pinning tests were added to Tests/UI/test_speech_playground_pane_
lifecycle.py to pin the fix's actual scope (only these two conditions -- no others --
detach the token), rather than relying only on the reviewer's own throwaway probes:
- test_unrelated_catalog_failure_does_not_detach_in_flight_profile_voice_token: voice
  validation in flight, NO config-change event anywhere, an unrelated catalog reload
  failure -- token and disabled button both untouched. Mutation-checked by widening
  _load_provider_catalog_worker's except Exception handler to clear
  self._profile_voice_validation_token unconditionally instead of only its own
  locally-reserved profile_voice_token; RED under that mutation (only this test),
  GREEN restored.
- test_configuration_change_for_unrelated_provider_does_not_detach_profile_voice_token:
  mark_provider_configuration_changed for a DIFFERENT provider while the token belongs
  to audio_cpp -- token and disabled button both untouched. Mutation-checked by
  removing the pending_voice_token.provider_id == provider_id guard; RED under that
  mutation (only this test), GREEN restored.

Both mutation checks were run together with health0-3 (TASK-2970) and the originally
ported TASK-3000 test in one pytest invocation each, confirming no cross-
contamination. Full 4-file targeted gate after adding both tests: 208 passed (205 +
3, including TASK-2970's own new positive-branch test from the same review round).
ruff check + format --check clean. No production-code changes in this round --
mark_provider_configuration_changed's fix from the original pass is unchanged.
<!-- SECTION:NOTES:END -->
