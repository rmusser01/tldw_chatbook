---
id: TASK-2311
title: Briefing failures surface their reason proactively
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: Generate with no provider configured yields a bare "failed" row —
no toast, no reason. The actual cause ("OpenAI API Key is required but not
found") only appears after clicking the row and reading the detail region,
and the provider defaulted to openai silently. A user who skipped first-run
setup hits their first configuration cliff with zero guidance.

UAT finding F39.

## Acceptance Criteria (the what)

- [ ] A failed generation surfaces its reason at failure time (toast or
      inline, markup=False) without requiring row selection.
- [ ] Configuration-class failures point the user at where to fix it
      (Settings), naming the provider that was attempted.
- [ ] The provider used for generation is visible before generating.
