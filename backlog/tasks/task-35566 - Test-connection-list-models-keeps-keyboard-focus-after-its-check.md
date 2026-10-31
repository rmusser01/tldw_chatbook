---
id: TASK-35566
title: Test connection & list models keeps keyboard focus after its check
status: To Do
created_date: 2026-10-18 17:42
dependencies: []
labels:
- model-config-redesign
- follow-up
- console
priority: low
references:
- tldw_chatbook/Widgets/Console/console_settings_modal.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's live capture phase (2026-10-18) and confirmed in triage to predate Phase 9: it reproduces at the phase base 111f8e48fc.

In Chat settings, pressing Enter on "Test connection & list models" disables the button while the check runs. Textual then blurs it, so nothing is focused. When the result re-enables the button (`_apply_connection_probe_result`, `_apply_model_discovery_result`), focus is not restored. Shift+Tab then starts from the footer's end, and getting back to the Endpoint field takes 8 presses instead of 2.

Live evidence: Phase 9 capture 01i, and the triage's base capture 01h-base.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After the check finishes, keyboard focus is back on Test connection & list models, unless the user moved focus elsewhere meanwhile
- [ ] #2 Shift+Tab twice from there reaches the Endpoint field
- [ ] #3 A mounted test pins both, with a stubbed connection test
<!-- AC:END -->
