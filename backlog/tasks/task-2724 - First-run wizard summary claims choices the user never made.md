---
id: TASK-2724
title: First-run wizard summary claims choices the user never made
status: To Do
assignee: []
created_date: '2026-08-06 17:00'
labels:
  - first-run
  - wizard
  - ux
  - uat
  - honesty
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Walking the first-run wizard on `origin/dev` `b0185749c` with Ctrl+N only — no provider picked, no key entered, no model typed — the step-4 Summary reports:

- `✗ Provider — no credentials or saved endpoint` (correct)
- `✓ Default model — gpt-5.6-terra` (false ✓: nothing was chosen; this is the hard-coded OpenAI default from `config.py`, presented as a completed setup step — with no provider at all)
- `✓ RAG — embedding model: e5-small-v2` while the header paragraph of the SAME screen says "Left at recommended defaults: tools off, **RAG off**, …" (the two lines contradict each other)

The summary is the wizard's honesty surface — the ✓/✗/– vocabulary implies "what you set up vs skipped". Reporting a config-file fallback as a ✓ "Default model" next to a ✗ provider misstates the state the user will actually land in (Console shows the provider-setup wall), and the RAG line disagrees with the recommended-defaults sentence directly above it.

Skipped/never-touched steps should render as `–` (like Speech/Tools/Theme already do), with defaults labeled as defaults.

Evidence: wizard Summary pane capture, 2026-08-06 UAT session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] A wizard walk with no selections shows no ✓ rows; untouched steps render with the – (default/skipped) marker and name the default they fall back to.
- [ ] The RAG summary row and the recommended-defaults header cannot disagree (single source of truth for the RAG on/off state).
- [ ] A model row only shows ✓ when the user actually selected or typed a model.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
