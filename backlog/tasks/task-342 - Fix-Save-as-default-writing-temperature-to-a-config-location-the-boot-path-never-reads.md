---
id: TASK-342
title: Fix Save-as-default writing temperature to a config location the boot path never reads
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With temperature set to 0.88, clicking 'Save as default' closed the modal with no confirmation toast. Config diff of the isolated home: [chat_defaults] got provider="llama_cpp" and streaming=true, while temperature=0.88, model and top_k were written to [api_settings.llama_cpp]; [chat_defaults].temperature stayed 0.6. A fresh app process shows rail 'Temperature 0.60' — the 0.88 the user saved 'as default' never comes back. The modal's help text ('Save as default also writes provider + streaming defaults to config') technically warns, but the button still accepts and writes the temperature edit somewhere inert.

**Repro:** 1. Rail > Configure; set Temperature (e.g. 0.88). 2. Click 'Save as default'. 3. Inspect ~/.config/tldw_cli/config.toml: temperature written under [api_settings.llama_cpp], [chat_defaults].temperature unchanged (0.6). 4. Restart app -> rail shows Temperature 0.60.

**Verifier note:** Confirmed real write/read priority bug, not in any ledger item or backlog task. Save-as-default writes temperature (and other PROVIDER_DEFAULT_PERSIST_FIELDS) to [api_settings.<provider>] (console_settings_modal.py:712-746; the docstring at line 718 explicitly claims this is 'the source build_default_console_session_settings reads on the next boot'), but the read path (console_session_settings.py:391-397, _float_setting_from_sources first-hit-wins) checks chat_defaults BEFORE provider settings, and the default config template ships chat_defaults.temperature=0.6 (config.py:2736) — so the saved value is permanently shadowed and silently reverts on restart, exactly as the reviewer's config diff showed. Also no success/failure toast on save. P1 upheld: explicit user save intent silently lost.

**Source:** Console UX expert review 2026-07-20 (finding j5-save-as-default-temperature-lost; P1, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J5 settings journey. Evidence: `j5-73-after-save-as-default.png`, `j5-74-rail-after-restart.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Save as default should either round-trip all shown values, or the dialog should state that sampling values are session-only and not write them to config at all. Silent acceptance followed by silent reversion after restart is the worst combination
<!-- AC:END -->
