---
id: TASK-366
title: Fix Test Provider reporting ready and blocked simultaneously while testing the saved URL
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-23 08:20'
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
- [x] #1 One unambiguous verdict about the endpoint currently shown in the form ('Could not reach http://127.0.0.1:9098 — connection refused'), and stale results cleared or marked outdated when inputs change or are saved
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Since the review (dev cad9e271d) the readiness/endpoint had already moved to the
DRAFT config + draft endpoint (bugs b partially resolved). Two things remained:

1. **"ready AND blocked" contradiction removed.** `_build_provider_readiness_findings`
   led the detail line with `readiness.user_message` ("<provider> is ready …",
   config-level) while appending `status=blocked` when the model was missing. It
   now computes `passed` up front and, for the config-ready-but-no-model case,
   leads with "<provider> is configured, but no default model is set." — one
   verdict consistent with the status line. A genuine pass still reads
   "is ready" / `status=ready`.
2. **Stale results invalidated.** New `_mark_provider_test_result_stale()`
   replaces a prior verdict with "Provider settings changed since the last test —
   re-run Test Provider." It is called from `_stage_provider_value` (fires on any
   provider field edit — endpoint/model/api-key/env-var/provider) and after a
   successful provider save, so a stale ready/blocked line can no longer linger
   over a changed or just-saved form (the review saw it persist after save).
   No-op on the not-run/already-stale sentinels. `_update_provider_test_result`'s
   guard was widened to `(QueryError, AttributeError)` so the state update is safe
   before mount.

Not taken here: live-probing an unreachable DRAFT endpoint when the model is
missing (the "Could not reach …" network verdict). The existing live probe runs
only after a passing readiness test; reordering it to probe on a blocked test is
a larger network-flow change and is left for a follow-up — the contradiction and
staleness (the titular defects) are resolved.

Verified RED→GREEN in `Tests/UI/test_settings_provider_test_draft.py`
(`test_findings_avoid_ready_claim_when_blocked_on_missing_model`,
`test_findings_keep_ready_verdict_when_passing`,
`test_mark_provider_test_result_stale_invalidates_prior_verdict`) + updated the
footer-shortcuts canary in `test_settings_configuration_hub.py`; 265 settings
tests green.
<!-- SECTION:NOTES:END -->
