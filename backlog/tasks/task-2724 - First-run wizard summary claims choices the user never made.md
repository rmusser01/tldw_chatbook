---
id: TASK-2724
title: First-run wizard summary claims choices the user never made
status: Done
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Existing pinned test blesses provider✓+template-model→✓, so the fix targets exactly the observed absurdity: ✗ provider with ✓ model/RAG.
2. RED: config shaped like the live run (template model + template RAG model + setup_started, no credentials) must produce zero ✓ rows.
3. GREEN: model row requires `provider_ok` for ✓ (typed-model-without-provider shows the name with –); RAG row gets the same template guard the theme row already has (`_TEMPLATE_DEFAULT_RAG_MODEL`).
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] A wizard walk with no provider configured shows no ✓ rows: the model and RAG template defaults render with the – marker, naming the default they fall back to.
- [x] The RAG row never claims ✓ for the untouched template embedding model, so it cannot contradict the header's "RAG off" sentence (AC narrowed from "single source of truth" — the header stays static text; the row is the moving part that lied).
- [x] The model row never shows ✓ while no provider is configured; a user-typed model without a provider is shown by name with the – marker and a "takes effect once a provider is connected" note (AC narrowed: with a provider configured, the accepted template model keeps its existing pinned ✓ — see test_rows_reflect_persisted_state).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`build_summary_rows` already had template-poisoning guards for theme/provider/speech but not for model or RAG, so the template defaults merged in at config load (`gpt-5.6-terra`, `e5-small-v2`) earned ✓ on a walk where nothing was chosen. Fix mirrors the existing `_TEMPLATE_DEFAULT_THEME` pattern: `_TEMPLATE_DEFAULT_MODEL` / `_TEMPLATE_DEFAULT_RAG_MODEL` constants; the model row now requires a configured provider for ✓ (a user-typed model without a provider is named with "takes effect once a provider is connected"; the bare template value reads "not selected"); the untouched template RAG model reads "off by default — embedding model e5-small-v2", agreeing with the header sentence instead of contradicting it. ACs were narrowed BEFORE implementation (recorded inline in the AC text) because the existing pinned test `test_rows_reflect_persisted_state` blesses ✓ for provider-configured + accepted-template-model — this task fixes the observed ✗-provider/✓-model absurdity without relitigating that. Tests: `TestSummaryTemplateHonesty` (5 tests, 4 watched RED; the fifth is the user-selected-RAG guard). Files: tldw_chatbook/UI/Wizards/first_run_setup_state.py, Tests/Wizards/test_first_run_setup_state.py.
<!-- SECTION:NOTES:END -->
