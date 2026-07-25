---
id: TASK-527
title: >-
  Remote install UX riders: policy-denial copy, silent bad-token degrade, suggested-name quirk
status: To Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-07-24 14:10'
labels:
  - skills
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three small UX gaps from PR #831 reviews: (1) a PolicyDeniedError during URL install surfaces as generic 'Could not import that skill.' because the outcome translator discards user_message ('Remote skill installs are disabled.'); (2) GitHubAPIClient.get_branches never raises (bare except returns ['main','master']), so a bad configured token on an ambiguous /tree/ URL silently degrades to a wrong-guess 404 instead of a clear auth message (partially mitigated by the slash-branch 404 hint); (3) a slash-branch tree URL with no subdir yields suggested_name from the branch segment instead of the repo name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Policy-denied URL installs show the policy's user-facing message in the import status line.
- [ ] #2 A 401/403 from the branch-listing call on an ambiguous tree URL produces a token/auth-specific message rather than a silent fallback guess.
- [ ] #3 suggested_name for /tree/<slash-branch> URLs without a subdir falls back to the repo name.
<!-- AC:END -->
