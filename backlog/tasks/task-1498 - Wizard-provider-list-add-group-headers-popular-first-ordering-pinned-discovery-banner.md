---
id: TASK-1498
title: >-
  Wizard provider list: add group headers, popular-first ordering, pinned
  discovery banner
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:55'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: Cloud/Local grouping exists only as sort order — no headers, no descriptions; discovery success renders below the fold so the 'we'll look for them' promise never visibly pays off. Depends on task-1495 for visibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cloud and Local sections have visible headers
- [ ] #2 Detected local server banner is pinned at the top of the step when discovery succeeds
- [ ] #3 A short popular-providers group (3-4) renders before the full alphabetical list
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Sections: Popular (openai/anthropic/ollama/llama_cpp when present) then Cloud/Local/Other alphabetical, headers as Statics inside the RadioSet; discovery banner+button pinned above the list, display-hidden until a server is found (kept the 1495 row budget: key input stays visible); scoped .setup-provider .hidden display:none (the bare .hidden convention was inert here). Note: live-contract tests are sensitive to real listeners on 8080/11434 — a stray UAT fake-server polluted two tests during development; consider injected discovery fakes for those tests later.
<!-- SECTION:NOTES:END -->
