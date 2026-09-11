---
id: TASK-32349
title: >-
  Library: a new profile with a pre-written config lands on the full rail with
  'Back to Get started' orphaned
status: Done
assignee: []
created_date: '2026-09-11 06:15'
updated_date: '2026-09-11 07:21'
labels:
  - library
  - onboarding
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An empty profile whose config.toml existed before first launch opens the full nine-destination rail (Media (0) … Collections (0)) with 'Back to Get started' at the bottom, a destination the user never saw (B D3 cap 63/64, A cap 04). Cause PROVEN in critique #8: coerce_library_lifecycle(raw=None, is_new_profile=False) resolves EXPANDED; the same happens to a real user who completes setup and quits before visiting Library. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A profile with no content yet lands on Get started regardless of whether its config file pre-dates the first Library visit
- [x] #2 'Back to Get started' is offered only after the user has seen Get started
- [x] #3 Pinned
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin three lifecycle-settle cases in a new Tests/UI/test_library_crit10_onboarding.py (unstored EXPANDED + all-empty -> STARTER; stored 'expanded' stays EXPANDED with Back offered; returning user with content -> GRADUATED).
2. Demote an UNSTORED EXPANDED to UNKNOWN in the all-empty settle branch of _apply_library_onboarding_evidence so the existing aggregate resolves STARTER.
3. Cross-reference the demotion from coerce_library_lifecycle's docstring (no code change there).
4. Update the one opposing pin (test_library_shell.py legacy/corrupt parametrize) and the docs row for 'Back to Get started'.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
An EXPANDED that nobody chose is the no-flash default `coerce_library_lifecycle(raw=None, is_new_profile=False)` returns, not a decision -- and nothing ever took it back, so an empty profile whose config.toml pre-dated its first Library visit opened the full nine-row rail and offered 'Back to Get started', a view it had never shown.

Fix is in the settle, not the default (the default must stay EXPANDED so a returning, populated profile does not flash the compact rail while the six-source evidence read is in flight). In `_apply_library_onboarding_evidence`'s all-EMPTY branch, an EXPANDED with no stored lifecycle falls back to UNKNOWN and the existing `aggregate_library_lifecycle` resolves it to STARTER. A stored 'expanded' (a real Explore press) and a returning user with content (HAS_USER_CONTENT graduates in the branch above) are untouched.

Deviation from the plan: storage is re-read via `_load_library_lifecycle_value()` at settle time instead of reusing the construction-time `_library_lifecycle_was_stored` snapshot. That snapshot never updates, so an Explore press followed by any later evidence round (a screen resume) would have demoted the user's choice back to Get started mid-session; `explore_library_lifecycle` mirrors 'expanded' into the same config this read consults, so the re-read gets it right. Pinned by test_explore_survives_the_next_evidence_read_in_the_same_session.

Pin updated, not weakened: test_library_onboarding_legacy_and_corrupt_preferences_open_expanded now carries a per-case settled value. Both cases still OPEN Expanded -- an absent value is the no-flash default, a corrupt one coerces to it -- and both settle to STARTER once the six-source evidence is all-EMPTY, because neither is a decision anybody made. Only a stored "expanded" is preserved: the guard compares the stored VALUE, not the mere presence of the key (review round 1, finding 3 -- the first cut tested presence and let a corrupt value keep the orphaned Back row). The constructor pin test_library_real_existing_config_without_lifecycle_defaults_expanded is untouched and green.

Live-verified on a purpose-built empty profile with a pre-written config (captures 01-04 under scratchpad crit10/wave/onboarding-import/caps): lands on Get started with the compact rail and no Back row; Explore reveals the Back row; the choice survives a restart.

Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/Library/library_rail_state.py (docstring cross-reference only), Tests/UI/test_library_crit10_onboarding.py (new), Tests/UI/test_library_shell.py, Docs/User_Guide/library.md
<!-- SECTION:NOTES:END -->
