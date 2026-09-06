---
id: TASK-31805
title: Home 'Model - Ready' and Settings Overview overstate readiness with no credential present
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ux
  - settings
  - home
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). On a fresh profile with no API key anywhere, Home reports 'Model: Ready' and Settings Overview shows an OpenAI status implying usability; an actual send then fails with the key-required error. Readiness surfaces should reflect the same resolve_provider_api_key check the send path uses (see the CLAUDE.md configuration notes on readiness/spend agreement).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Home and Settings Overview report not-ready when no valid credential resolves for the selected provider.
- [ ] #2 Readiness surfaces and the send path share one credential check.
<!-- AC:END -->
