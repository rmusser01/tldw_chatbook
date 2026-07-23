---
id: TASK-366
title: Fix Test Provider reporting ready and blocked simultaneously while testing the saved URL
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With a deliberately dead draft endpoint (http://127.0.0.1:9098) typed into the Endpoint field, Test Provider returned: 'Provider test | llama_cpp is ready. No API key is required. | model=missing | LLAMA_CPP_API_KEY=<redacted> | Endpoint: api_settings.llama_cpp.api_url=http://localhost:8080/completion | status=blocked'. It (a) says ready AND blocked, (b) cites the SAVED config URL (localhost:8080/completion) while the readiness panel right below shows the draft URL 127.0.0.1:9098, and (c) never flags that the typed URL is unreachable. The stale 'blocked' message also persisted on screen after settings were later saved successfully (j1-34) until Test was manually re-run.

**Repro:** Settings > Providers & Models -> Provider=llama.cpp -> Endpoint=http://127.0.0.1:9098 (unsaved) -> click Test Provider -> read result line rows 38-39.

**Verifier note:** Code-confirmed. _provider_readiness_test_report (settings_screen.py:5257-5311) concatenates readiness.user_message ('llama_cpp is ready…', computed from SAVED app_config) with status=blocked when the model field is empty — the contradictory line is structural. The endpoint line calls _provider_endpoint_summary(provider) with no draft value (5292 → 4785), so it prints the saved api_settings URL while the form shows a different draft. The task-191 live probe (_provider_live_probe_base_url, 5318+) runs only after a PASSING readiness test, so an unreachable draft URL is never flagged when the model is missing. Not covered by provider-test-outcome-toast or task-191 (both shipped the surface, neither covers these defects). Downgraded P1→P2: the toast summary itself renders an unambiguous verdict ('Provider test failed: … Also set a default model.'); the detail row and draft-vs-saved mismatch mislead but do not hard-block.

**Source:** Console UX expert review 2026-07-20 (finding j1-test-provider-contradicts-itself; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-13-bad-url-typed.png`, `j1-14-test-bad-url-immediate.png`, `j1-16-test-bad-url-full.png`, `j1-34-after-save-category.png`, `j1-35-test-after-save.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One unambiguous verdict about the endpoint currently shown in the form ('Could not reach http://127.0.0.1:9098 — connection refused'), and stale results cleared or marked outdated when inputs change or are saved
<!-- AC:END -->
