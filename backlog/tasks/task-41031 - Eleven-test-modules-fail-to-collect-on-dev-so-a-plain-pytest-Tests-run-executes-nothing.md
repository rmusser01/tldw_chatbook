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
  - `Tests/Evals/test_source_reader_comparison.py`, `test_source_reader_deepseek_pricing.py` and `test_source_reader_sources.py` import `Evals.source_reader`. That package belongs to TASK-32029 (In Progress) and exists only on `codex/release-0.2.4-20261107` (PR #3475; restored in b1590c86de).
  - `Tests/UI/test_persona_buddy_host.py` imports `UI.Navigation.persona_buddy_host`.
  - `Tests/UI/test_persona_buddy_view.py` imports `Widgets.Persona_Widgets.persona_buddy`.
- **One module imports a helper another test file dropped.** `Tests/UI/test_product_maturity_phase1_empty_setup_states.py` imports `RaisingSkillsScopeService` from `Tests/UI/test_destination_shells.py`, which no longer defines it.
- **Two modules import an optional dependency unconditionally.** `Tests/Web_Server/test_web_auth_integration.py` and `Tests/Web_Server/test_webauthn_service.py` import `fido2`, which only the `web` and `all-tools` extras install, so they break collection in a venv without those extras.

TASK-34100.14's notes counted the same 11 errors and called them pre-existing. TASK-32029 owns the three source_reader modules; no task owns the other eight. These modules run on no PR lane.

Why high: `.github/workflows/test.yml` runs on every push to main, and its core job runs `pytest Tests --ignore=Tests/UI` without `--continue-on-collection-errors`, so the eight non-UI collection errors (three in Tests/Chat, three in Tests/Evals, two in Tests/Web_Server) abort all six core shards there. nightly-deep survives only because it passes that flag.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `pytest Tests --collect-only` reports zero collection errors on dev in a venv with the dev extras only
- [ ] #2 Each test of an absent module is deleted, retargeted to the code that replaced it, or kept and collected because its module lands on dev (the source_reader modules go through TASK-32029); the choice is recorded per file in the task notes
- [ ] #3 The two WebAuthn modules skip, rather than error, when fido2 is not installed
<!-- AC:END -->
