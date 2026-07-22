---
id: TASK-432
title: Provider test reports draft values and the Test control is clickable
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-22 17:49'
labels:
  - settings
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live in Settings > Providers & Models: with an unsaved draft endpoint (:9099) the test correctly exercised the DRAFT (endpoint reachable, 1 model - nothing was listening on the displayed URL) but the evidence line printed the stale saved value api_settings.llama_cpp.api_url=http://localhost:8080/completion, so the proof contradicts what was tested. Separately, "Test Provider" renders like a control but clicking it does nothing; the test only runs via the 't' category hotkey and only when focus is outside an input, and nothing explains this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Test results display the exact endpoint/model/key-source values that were used in the test (draft values when testing a draft)
- [x] #2 The Test control is activatable by mouse click
- [x] #3 The 't' hotkey behavior and its focus requirement are discoverable or removed as the sole path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Discovery: dev had moved since the review (dc196563f) — AC#2 (clickable Test button) and AC#3 (non-hotkey path) were already delivered by task-189's #settings-test-provider button. The real remaining defect was AC#1, extended (per user's 'full draft-honesty' choice) to ALL tested values.

AC#1 (settings_screen.py): the Test now evaluates and honestly reports the unsaved DRAFT provider config.
- overlay_provider_draft_config(app_config, *, provider_save_key, endpoint_key, draft_endpoint, draft_env_var, draft_api_key): pure, deep-copies config, overlays only non-None draft fields (empty string overlays = explicit clear); never mutates the input.
- _provider_test_staged_config(provider): overlays only the DIRTY draft fields (SettingsDraft.dirty_keys — a @property, accessed without parens; the app's own unsaved signal) and feeds the merged config to get_provider_readiness (unchanged pure fn), so ready/api_key_source/env_var reflect the draft. Returns app_config unchanged when nothing is dirty.
- _build_provider_readiness_findings(...): resolved-input (no widgets → unit-testable on a bare screen) evidence builder with per-value (draft) provenance tags — endpoint '(draft)', 'api_key_source=draft api_key (unsaved)', env-var '(draft env var)', model '(draft)'. The draft API-key VALUE is never placed in a finding (only the source label); redaction is per-finding, skipping ONLY the one constant literal marker (whole-string redact_secret_text mangled it — its regex reads api_key...=value as a secret). Empirically proven no secret leaks.

AC#2/AC#3: no production change (already wired). Pilot tests pin the real Button.press→handle_test_provider→action_settings_test_category(allow_text_entry_focus=True) path; AC#3 gets an honest companion test (test_t_hotkey_does_not_run_test_while_input_focused) proving the 't' hotkey no-ops while an input is focused (real keypress typed into input + direct guard call that fails if the guard is removed) — which is WHY the clickable button is needed.

Tests (Tests/UI/test_settings_provider_test_draft.py): 13 total — 5 pure overlay, 4 bare-instance findings (incl. draft-key-value-never-printed), 4 pilot (AC#2, AC#3-from-focus, AC#3-hotkey-no-op, draft-endpoint wiring). All RED→GREEN; mutation-checked. No-draft path byte-identical (20 existing hub provider tests green).

Follow-up filed TASK-483 (pre-existing: env-var VALUES redacted only by name-pattern; harden to present/missing or positional redaction).

Files: tldw_chatbook/UI/Screens/settings_screen.py, Tests/UI/test_settings_provider_test_draft.py.
<!-- SECTION:NOTES:END -->
