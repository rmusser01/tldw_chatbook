---
id: TASK-41031
title: Eleven test modules fail to collect on dev, so a plain pytest Tests run executes
  nothing
status: In Progress
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
assignee:
- '@codex'
updated_date: 2026-11-17 20:26
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
- [x] #2 Each test of an absent module is deleted, retargeted to the code that replaced it, or kept and collected because its module lands on dev (the source_reader modules go through TASK-32029); the choice is recorded per file in the task notes
- [x] #3 The two WebAuthn modules skip, rather than error, when fido2 is not installed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR paths: existing ADR-090 and ADR-360. Reason: repair tests and comprehensive CI setup for shipped owners without restoring abandoned API shapes or promoting the unshipped SourceReader experiment. Verify each absent import and original add provenance, port unique thinking/Skills/Buddy behaviors to maintained owners before retiring obsolete collectors, install the declared web extra for full qualification, and verify targeted parallel collection and runtime cases. Record every file disposition; do not mark Done before zero-error complete-tree collection is established.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-11-17 light-CI release qualification run44150524535 reproduced the core collection failures. Commit3a6263c8a7 added these collectors when their imported APIs were already absent at its parent. Reuse test-only ports from31ea25ae74: transport eligibility (14 cases), durable hidden-thinking replay on/off/on, and late same-answer fleet-usage reattachment. All16 replacement cases pass on current dev-derived source (5.78s). Retire Tests/Chat/test_console_thinking.py, test_local_qwen_reasoning.py and test_local_reasoning_toggle.py only after retaining those ports; canonical thinking/capture/privacy/history suites remain. Defer the six stray SourceReader files with active unshipped TASK-32029: Tests/Evals/test_source_reader.py, test_source_reader_comparison.py, test_source_reader_deepseek_pricing.py, test_source_reader_experiment.py, test_source_reader_sources.py and test_source_reader_transport.py. The experiment and its owner branch/evidence remain active; this CI change does not import its unmerged product implementation, change its task or claim it canceled. UI collector disposition and full collection verification remain pending.
Current replacement coverage on 2026-11-17:190 Buddy preference/resolution/widget/overlay/native tool/realtime/splash cases passed126.25s. The five exact fallback regressions failed before47b9b5800b and pass after its scoped source correction. The two old UI collectors Tests/UI/test_persona_buddy_host.py and test_persona_buddy_view.py are retired after transferring current-owner assertions from6ee08cc4e9,e646ff7313,44fa516f32. Tests/Persona_Visual/test_persona_buddy.py is also a lazy absent-owner collector: priority/lease behavior is covered by current Persona_Buddy controller tests, authority/cancellation/fallback by resolution tests, and validated persistence by preferences tests. Its persona_buddy.json symlink/FIFO/private-file assertions target superseded storage; current incumbent-config persistence is the owner, not an equivalent claim about that removed file. Standalone Skills setup imports and parameter in test_product_maturity_phase1_empty_setup_states.py are retargeted to the maintained Library Skills retry/recovery-copy/no-row/no-handoff assertions frombab59abb18. The3 relevant Skills/Library/Watchlists controls pass using their actual bootstrap profile. Comprehensive core explicitly installs the declared web extra rather than skipping WebAuthn coverage. Full dev-only collection acceptance is still pending, and no experiment/product task is marked complete.
Final local collector verification on the repaired tree:138445 tests collected with zero collection errors in329.52s. This executed collection only, not the full application suite. The two WebAuthn modules skip explicitly without their optional dependency; with the declared dependency in temporary scratch,43 real auth/ceremony cases pass2.57s. The preserved Buddy suites pass190 cases126.25s; source ratchets231 cases9.29s and policy/workflow/package-metadata contracts781 cases77.71s pass. AC1 remains open for fresh dev-only environment confirmation; no assertion execution sweep or installed-optional closure is inferred from collection.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
