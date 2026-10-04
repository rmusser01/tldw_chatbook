---
id: TASK-34332
title: 'Endpoint field keeps every character typed after picking a local provider'
status: To Do
assignee: []
created_date: '2026-10-03 16:30'
labels:
  - first-run-wizard
  - provider-step
  - ux-review-2026-10-02
dependencies:
  - TASK-34100.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Characters typed into the Provider step's Endpoint field soon after picking llama.cpp are dropped. The URL ends up truncated, and Next is then refused. This is the "users lose keystrokes" symptom that review finding cross-cutting-17 describes. TASK-34100.1's ACs did not cover it, and it is not a regression: base loses more characters.

Evidence (TASK-34100.1 review round 2, live, real app, files under .worktrees/setup-wizard-ux-qa/evidence/):
- Base 3c439d606e, Tab then End, 30 Backspaces, then the URL typed as one burst: the field holds "http://127.0" with "Endpoint URL must include a valid host and port." (g1-v2-base80/02-base-endpoint-truncated.txt).
- Branch under a main-thread sampler, same sequence: "http://127.0.0.1:" with "Endpoint URL must not include an empty port." Five Ctrl+N presses were then refused with "The provider settings are invalid." (g1-v2-prof/00-branch-endpoint-truncated.txt).
- The branch without the sampler kept the whole value (g1-v2-fk120/04, g1-v2-fm80/06), so the loss depends on how busy the UI loop is while the user types.

The likely cause is something that rewrites `#setup-provider-endpoint` while the user types: a selection-change or validation handler that sets `.value`, or a re-render. TASK-34100.6 owns the endpoint fields; coordinate there if it lands first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The code path that drops or overwrites typed Endpoint characters is identified and recorded
- [ ] #2 A Pilot test that picks llama.cpp and types a full URL with no pauses ends with exactly that URL in the field; it fails on the pre-fix code
- [ ] #3 Verified live on a fresh isolated profile with the UI loop under load (for example during the localhost scan): the typed URL survives and Next accepts it
<!-- AC:END -->
