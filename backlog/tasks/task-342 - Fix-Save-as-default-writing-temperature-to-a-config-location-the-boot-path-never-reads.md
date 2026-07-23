---
id: TASK-342
title: >-
  Fix Save-as-default writing temperature to a config location the boot path
  never reads
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 03:34'
labels:
  - console
  - ux
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
- [x] #1 Save as default should either round-trip all shown values, or the dialog should state that sampling values are session-only and not write them to config at all. Silent acceptance followed by silent reversion after restart is the worst combination
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Align build_default_console_session_settings source precedence with the documented Save-as-default contract: (model_profile, provider_settings, chat_defaults) — provider-scoped saves beat global defaults, matching how model already resolves
2. TDD: precedence unit test + parametrized round-trip over PROVIDER_DEFAULT_PERSIST_FIELDS
<!-- SECTION:PLAN:END -->

## Implementation Notes

Two deliberate-but-conflicting contracts existed: f14d22dc3 (review
feedback, pinned by tests) ranks chat_defaults ABOVE [api_settings.*]
scalars so factory provider templates can't shadow user-tuned globals —
which made Save-as-default's sampling writes into [api_settings.<provider>]
permanently inert (the docstring claimed it was the boot source).

Fix honors both: Save-as-default now writes sampling values to
**[console.provider_defaults.<provider>]** — a section that only ever
contains Console-saved defaults — and the boot builder ranks it
(model_profile, console_saved_defaults, chat_defaults, provider_settings).
Model/endpoint writes stay in api_settings (model already resolves
provider-first). The f14d22dc3 protection is untouched and its tests stay
green; a first attempt that blanket-reordered provider_settings above
chat_defaults was caught by exactly those tests and reverted.

Verified: precedence unit tests over all PROVIDER_DEFAULT_PERSIST_FIELDS +
factory-shadow control (RED first); write-path test updated to the new
section; against a real config file, api_settings temperature=0.88 alone
boots 0.6 (protection intact) while console.provider_defaults 0.88 boots
0.88 (round-trip fixed; the review's j5-74 evidence showed this reverting).
Note: defaults saved by the modal BEFORE this fix sit inert in
api_settings and need one re-save to migrate. Files:
`Chat/console_session_settings.py`, `Widgets/Console/console_settings_modal.py`.
