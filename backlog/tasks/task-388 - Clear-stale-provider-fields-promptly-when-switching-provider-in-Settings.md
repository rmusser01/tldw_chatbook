---
id: TASK-388
title: Clear stale provider fields promptly when switching provider in Settings
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
900ms after selecting llama.cpp, the form still showed Model=gpt-4o, Endpoint=https://api.openai.com/v1, 'set OPENAI_API_KEY or paste a local key' and readiness 'OpenAI / gpt-4o' (j1-10). A few seconds later the dependent fields settled to llama.cpp values (j1-12). During the window the screen asserts a provider/credential combination that never existed.

**Repro:** Providers & Models -> open Provider select -> choose llama.cpp -> read the form within ~1s, then again after ~3s.

**Verifier note:** Not covered: task-214 is the reverse direction (Settings shows boot-time selection until manual reselect after a config write), settings-input-select-fix is the Input→Select interaction, and task-290 covers the mount-time recompose storm, not provider-switch dependent-field latency. The observation (OpenAI model/endpoint/key text persisting ~1-3s after selecting llama.cpp before dependent widgets settle, j1-10 vs j1-12) is a transient async-recompose staleness window during which the form asserts a provider/credential combination that never existed. Plausible given the screen's dependent-field rebuild pattern; not verified to the exact mechanism. P3: transient, self-corrects, no data written.

**Source:** Console UX expert review 2026-07-20 (finding j1-provider-switch-stale-window; P3, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-10-llamacpp-selected.png`, `j1-12-after-provider-switch-settle.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dependent fields update atomically with the provider selection, or show a brief loading placeholder instead of the previous provider's values
<!-- AC:END -->
