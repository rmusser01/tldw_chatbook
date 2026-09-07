---
id: TASK-2970
title: >-
  SpeechPlaygroundPane marks a provider stale on any non-fresh first catalog
  load, not only after a real config change
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 04:19'
updated_date: '2026-08-07 13:05'
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
- [x] #1 A first (or otherwise non-configuration-change) catalog load whose health.fresh is False shows the accurate state-specific recovery copy (reconfiguring -> settings are being applied; a naturally-stale available catalog -> catalog is stale), not the generic settings-changed copy
- [x] #2 The two xfail(strict=True) parametrizations in Tests/UI/test_speech_playground_pane_lifecycle.py::test_audio_cpp_health_states_use_fixed_safe_recovery_copy (health2, health3) are un-xfailed and pass
- [x] #3 SpeechCatalogMixin._load_provider_catalog_worker only adds a provider to _stale_providers when a catalog is genuinely superseding a previously-fresher one (e.g. a real configuration-revision change), not on every catalog application regardless of history
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Un-xfail the health2/health3 parametrizations in test_audio_cpp_health_states_use_fixed_safe_recovery_copy; run to confirm genuine RED against current code.
2. In SpeechCatalogMixin._load_provider_catalog_worker's success path, capture the provider's previously-stored configuration revision (from _catalog_configuration_revisions) before it is overwritten by this load's own revision.
3. Replace the unconditional `if catalog.health.fresh: discard else: add` with: discard when fresh; add only when NOT fresh AND a previous revision was recorded AND it differs from this load's revision (a genuine, real configuration-revision change this load hasn't caught up to yet); otherwise discard. This preserves _mark_stale_catalog_result and mark_provider_configuration_changed's own unconditional adds (untouched, out of scope) while fixing only this success-path branch.
4. Re-run the two un-xfailed tests to confirm GREEN; blast-radius grep every _stale_providers reader/writer and run the full Speech/TTS gate list to confirm mark_provider_configuration_changed's own genuine-config-change copy is unaffected.
5. Update the task file (AC boxes, Implementation Notes) and mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Un-xfailed health2/health3 in test_audio_cpp_health_states_use_fixed_safe_recovery_copy;
confirmed genuine RED (both asserted "settings changed; refresh models" instead of the
state-specific copy) against pre-fix code before touching production code.

Root cause confirmed via git show f560217fb~1:tldw_chatbook/UI/STTS_Window.py: the
retired widget's equivalent success path did an unconditional
self._stale_providers.discard(provider_id) with NO health.fresh branch at all -- the
mixin-based rebuild independently reimplemented this path and added an
if/else on catalog.health.fresh that marks ANY non-fresh load "stale", including the
very first one, before any configuration could possibly have changed.

Fix (SpeechCatalogMixin._load_provider_catalog_worker, success path): capture the
provider's previously-stored configuration revision (_catalog_configuration_revisions,
before this load overwrites it) and only add to _stale_providers when health.fresh is
False AND a previous revision was recorded AND it differs from this load's own
revision -- a genuine, real configuration-revision change this fetch has not caught up
to yet. Otherwise (first load ever, or a non-fresh repeat load with no revision
change) discard, matching the retired widget for the reachable case AC#1/#2 cover.
This is a narrower rule than the retired widget's own (unconditional discard, no
add path at all on this branch), chosen because AC#3's own wording specifically
calls out "genuinely superseding a previously-fresher one (e.g. a real
configuration-revision change)" as still warranting the add -- the configuration-
revision transition is the literal, most precise signal for that, and it cannot
misfire on the "flaky fresh flag with no real change" case a catalog-freshness-only
comparison would.

_mark_stale_catalog_result and mark_provider_configuration_changed (the two other,
explicit _stale_providers.add() call sites) are untouched -- out of this task's scope,
confirmed still producing the settings-changed copy via the existing
test_configuration_change_marks_catalog_stale_without_connecting and
test_catalog_result_is_discarded_when_configuration_revision_changes (both still
green).

Mutation-checked: temporarily reverted the fix back to the unconditional
if/else -- confirmed ONLY health2/health3 failed, the sibling TASK-3000 test and
health0/health1 stayed green; restored.

Gates: targeted Speech/TTS suite (Tests/UI/test_speech_playground_pane_lifecycle.py,
test_speech_playground_pane.py, test_stts_playground_catalog.py,
Tests/TTS/test_stts_audio_cpp_generation.py) 205 passed; full Tests/UI/test_speech_*.py
+ test_stts_*.py + Tests/TTS/ + Tests/TTS_Events/ sweep: 2860 passed, 16 skipped
(optional deps), 1 pre-existing failure unrelated to this change
(test_first_time_audio_cpp_setup_lab_generation_and_console_handoff, already documented
pre-existing in task-2951's own notes, confirmed against the unmodified base commit
there). Repo-wide --collect-only: 31874 collected, 0 errors. ruff check + format
--check clean on touched files.

Files: tldw_chatbook/UI/Speech/speech_catalog_mixin.py,
Tests/UI/test_speech_playground_pane_lifecycle.py (un-xfail + comment rewrite).

### Review round: positive-branch coverage added

Coordinator review approved the fix but flagged that health2/health3 only pin the
negative case (a first-ever non-fresh load must NOT be marked stale) -- AC#3's
positive branch (a genuine second-load supersession MUST still be marked stale) had
no dedicated test of its own. Added
test_second_load_after_genuine_config_change_marks_provider_stale to
Tests/UI/test_speech_playground_pane_lifecycle.py: mounts normally (records
configuration revision 1), bumps service.revisions["audio_cpp"] to 2, drives a
second, successful (token-current) reload whose catalog still reports fresh=False,
and asserts _stale_providers gains the provider with the "settings changed" status
copy. Mutation-checked by disabling the elif genuine-supersession branch in
_load_provider_catalog_worker (falls through to unconditional discard); RED under
that mutation (only this test), GREEN restored -- health2/health3 and both TASK-3000
tests unaffected in either direction. No production-code change in this round.
<!-- SECTION:NOTES:END -->
