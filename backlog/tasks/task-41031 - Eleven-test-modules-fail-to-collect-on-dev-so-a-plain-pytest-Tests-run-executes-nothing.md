---
id: TASK-41031
title: "Eleven test modules fail to collect on dev, so a plain pytest Tests run executes nothing"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- ci
priority: high
references:
- Tests/Chat/test_console_thinking.py
- Tests/Evals/test_source_reader_sources.py
- Tests/Web_Server/test_webauthn_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`pytest Tests --collect-only` on dev ecad404bd4 (2026-11-15) collects 137,259 tests and stops with "Interrupted: 11 errors during collection". A plain `pytest Tests` run therefore executes no tests at all. The errors are:
- **Eight modules import code that does not exist on dev.** Git's add record for all eight is 3a6263c8a7 (2026-10-12, "complete the pooled-seam sweep"):
  - `Tests/Chat/test_console_thinking.py` imports `Chat.console_thinking`.
  - `Tests/Chat/test_local_qwen_reasoning.py` and `Tests/Chat/test_local_reasoning_toggle.py` import `LocalReasoningReplay` from `Chat.local_reasoning`.
  - The three `Tests/Evals/test_source_reader_*.py` import `Evals.source_reader`.
  - `Tests/UI/test_persona_buddy_host.py` imports `UI.Navigation.persona_buddy_host`.
  - `Tests/UI/test_persona_buddy_view.py` imports `Widgets.Persona_Widgets.persona_buddy`.
- **One module imports a helper another test file dropped.** `Tests/UI/test_product_maturity_phase1_empty_setup_states.py` imports `RaisingSkillsScopeService` from `Tests/UI/test_destination_shells.py`, which no longer defines it.
- **Two modules import an optional dependency unconditionally.** `Tests/Web_Server/test_web_auth_integration.py` and `Tests/Web_Server/test_webauthn_service.py` import `fido2`, which only the `web` and `all-tools` extras install, so they break collection in a venv without those extras.

TASK-34100.14's notes counted the same 11 errors and called them pre-existing; no task owns them. These modules run on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `pytest Tests --collect-only` reports zero collection errors on dev in a venv with the dev extras only
- [ ] #2 Each test of an absent module is deleted, or retargeted to the code that replaced it, with the choice recorded per file in the task notes
- [ ] #3 The two WebAuthn modules skip, rather than error, when fido2 is not installed
<!-- AC:END -->
