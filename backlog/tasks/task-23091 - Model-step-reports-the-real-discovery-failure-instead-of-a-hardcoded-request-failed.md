---
id: TASK-23091
title: >-
  Model step reports the real discovery failure instead of a hardcoded 'request
  failed'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-28 15:45'
updated_date: '2026-08-28 15:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When ProviderStep hands off a failed discovery without a typed outcome in hand, ModelStep hardcodes the failure category to 'request failed'. The user is then told to check whether the server is running even when the real cause was a rejected API key, which is both unactionable and contradicts the provider-aware error copy added for UAT M-4. This masked the true cause for most of the TASK-23089 investigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A discovery that failed on authentication renders the authentication copy, not 'Couldn't reach the server'
- [ ] #2 The generic 'request failed' wording is used only when no typed outcome is available
- [ ] #3 Regression coverage pins the category derived from a recorded provider outcome
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Derive the category from the ProviderStep's recorded typed outcome when one exists.\n2. Keep 'request failed' only as the no-outcome fallback.\n3. Cover both paths with tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ProviderStep already records the typed ModelDiscoveryResult for a selection, so _handed_off_failure_category recovers the real category from it on both handoff paths; 'request failed' now appears only when no typed outcome exists, and a malformed recorded outcome degrades to the generic wording rather than raising. Tests pin all three paths and assert the recovered category still reaches classify_discovery_failure's AUTH branch (a correct category that no longer selects the auth copy would be worthless); the auth case was confirmed to fail against the previous behavior. Also refreshed the production diagnostic inventory pin in a separate commit -- that drift is inherited from dev (preflight fails identically on pristine origin/dev), reviewed per protocol, with a note to the owner that both new statements interpolate exception text that can carry a filesystem path.
<!-- SECTION:NOTES:END -->
