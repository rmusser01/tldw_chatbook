---
id: TASK-14820
title: Ingest forecast and consent must come from one truthful computation
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
updated_date: '2026-08-10 21:41'
labels:
  - library
  - ingest
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 of the 2026-08-10 re-critique (23/40; snapshot `.impeccable/critique/2026-08-10T20-43-44Z__chatbook-widgets-library-library-ingest-canvas-py.md`). The ingest surface's central promise — the commit-point forecast — is wrong by roughly half on the archetypal mixed folder, and it contradicts the consent line two rows above it.

`commit_summary_line` computes `will_import = supported_total - will_match` and counts only empty files as failures; a file whose type group has UNMET `required_features` (no pdf/audio/ebook/OCR tooling installed) is still counted as "will import", even though the preflight has already emitted a warning naming that exact missing dependency. The inline consent line uses a completely separate computation (`count_warning_affected_files`), so the two numbers are derived independently and disagree on screen.

Observed live in two independent sessions on different fixtures: `15 will import` sitting two rows above `⚠ Press Start again to import anyway — 7 files may fail`, delivering `8 imported · 5 skipped · 8 failed`; and `10 will import · 3 will skip · 2 will fail` delivering `1 done · 3 skipped · 10 failed · 1 matched`. The optimistic number is the one that persists on screen.

This is the forecast→receipt honesty loop that two prior critiques named as this surface's signature strength. It is now its central defect: the arcs made the forecast more detailed without making it more truthful.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The commit forecast and the inline consent line are derived from ONE computation and can never state different numbers for the same staged selection
- [x] #2 Files whose type group has unmet required tooling are forecast as failures (not imports), with the reason distinguishable from the empty-file reason (e.g. "N need tooling, M empty")
- [x] #3 A mixed folder staged on an install lacking optional backends produces a forecast whose import/skip/fail counts match the actual receipt tally
- [x] #4 The forecast remains visible (not blanked) while a gate blocks Start, so a blocked user does not lose the numbers they were reasoning about
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add `classify_missing_features(group, missing)` to `ingest_capabilities.py` -- the one place that knows which of a group's features are REQUIRED vs optional; reuse it instead of re-deriving the split at the call sites.
2. Add ONE forecast in `library_ingest_state.py`: frozen `IngestForecast` (will_import / will_match / match_capped / will_skip / will_fail_tooling / will_fail_empty / at_risk / tooling_groups) built by `build_ingest_forecast(preflight)` from the pre-flight's OWN warnings (not a fresh environment probe, so forecast and warning wall can never disagree).
3. Derive BOTH lines from that one object: `forecast_summary_line()` (commit line) and `forecast_consent_line()` (inline two-press consent). `count_warning_affected_files` becomes a thin wrapper over the forecast so no second computation survives.
4. Files in a group whose REQUIRED feature is warned forecast as failures with a distinguishable reason ('N need tooling, M empty'); a group with only an OPTIONAL feature warned stays an import but counts as at-risk for the consent line.
5. AC#4: stop blanking the commit line when a gate blocks Start -- it is suppressed only for path/pre-flight errors and for no-selection. (Deliberately supersedes task-3305 MI-16's hide-on-option-error rule, whose real defect was STALENESS; the gate updater already syncs both lines in one pass.)
6. GOVERNANCE TEST: stage a real mixed folder on this venv (no pdf/audio/ebook/OCR backends), compute the forecast, run it through the real submit path (`_IngestRunnerHarness`: real `submit_library_ingest_job` + real `run_parse_job` + real `persist_parsed_media` + real MediaDatabase) and assert the forecast's import/skip/fail counts EQUAL the terminal job outcomes.
7. Keep the state-layer unit suite + inline-consent suite green; update the two suites' copy pins that encode the old (untruthful) numbers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Built ONE forecast and made both lines render from it.

**`IngestForecast` + `build_ingest_forecast` (library_ingest_state.py)** — a frozen dataclass (will_import / will_match / match_capped / will_skip / will_fail_tooling / will_fail_empty / at_risk / tooling_groups) computed once per state build. `forecast_summary_line` renders the commit line, `forecast_consent_line` renders the armed gate line, and `count_warning_affected_files` is now a two-line read of `forecast.consent_affected` — that function BEING a second, independent computation is what let the two lines disagree on screen. There is no longer any arithmetic outside the forecast.

**Truthfulness (AC#2)** — the forecast is keyed off the pre-flight's OWN warnings (not a fresh environment probe), so what it counts and what the warning wall says are the same fact. `ingest_capabilities.classify_missing_features(group, missing)` splits those warned features into the group's REQUIRED and OPTIONAL ones — the required/optional distinction is declared on `TypeGroupCapabilities`, and every consumer previously re-derived it by unioning both tuples. A group with an unmet REQUIRED feature forecasts every one of its files as a failure ('N need tooling'); an unmet OPTIONAL one leaves them imports but marks them `at_risk`, which is what still reads 'may fail' in the consent line.

**AC#4** — the commit line no longer blanks when a gate closes. This deliberately supersedes task-3305 MI-16 (which hid it on an option error); MI-16's real defect was a STALE line, and the gate updater already syncs the forecast and the gate line in one in-place pass, so they move together. `Tests/UI/test_library_shell.py::test_commit_line_hides_while_option_error_gate_blocks` was rewritten (and renamed `…stays_visible_and_synced…`) to encode the new rule plus MI-16's surviving half (widget identity across the in-place path).

**GOVERNANCE (AC#3)** — `Tests/integration/test_library_ingest_flow.py::test_forecast_counts_equal_the_real_receipt_for_a_mixed_folder` stages a real 7-file mixed folder on this venv (no pdf/ebook/OCR backends — asserted up front, so the fixture cannot pass vacuously), computes the forecast from the REAL `analyze_path`, then runs the SAME folder through the real submit path (`submit_library_ingest_job` -> `run_parse_job` -> `persist_parsed_media` -> real `MediaDatabase`; only the process pool is the runner suite's in-process stand-in) and asserts (will_import, will_skip, will_fail) == (done, skipped, failed) plus which files landed in which bucket. Result: forecast '2 will import · 1 will skip · 4 will fail (3 need tooling, 1 empty)' == receipt 2 done / 1 skipped / 4 failed. Reverting the required-tooling branch makes that same test print '5 will import' against 2 actual — the original live defect, reproduced.

Modified: `Library/ingest_capabilities.py`, `Library/library_ingest_state.py`, `Tests/Library/test_ingest_capabilities.py`, `Tests/Library/test_library_ingest_state.py`, `Tests/UI/test_library_ingest_inline_consent.py`, `Tests/UI/test_library_shell.py`, `Tests/integration/test_library_ingest_flow.py`, `Docs/User_Guide/library/import-and-export.md`.
**xhigh review round (2026-08-10)** — three defects in the shipped forecast, all found by reading it against a backend and a cap it never knew about.

*The forecast did not know which backend it forecast (worst).* `build_ingest_forecast` subtracted LOCAL tooling gaps unconditionally, while `targets_server` was computed in `build_library_ingest_state` and never passed in. Live consequence: server mode + 5 .mp3 + no local audio extra rendered "0 will import · 5 will fail (need tooling)" and "5 files will fail without more tooling" for a batch `_submit_server_ingest_job` would have handed to the server, which never loads a local parser. The deleted `will_import = supported_total - will_match` was at least backend-agnostic, so for every server user this arc made the line worse than the defect it fixed. The forecast now takes `targets_server`; under it the local gaps are dropped entirely (`will_fail_tooling`/`at_risk`/`tooling_groups` all empty) and NO claim is made about the server's own tooling, because this process cannot know it (task-3309: forwarded extras are unverified). What is knowable is what gets sent, so the line says exactly that: "5 will be sent to the server · server tooling isn't checked from here".

*Consent followed the warnings, not the forecast.* `start_confirm_active` (and the screen's arming branch) keyed off the bare presence of `preflight.warnings`, so a server run — which reads the same local warnings and stakes nothing on them — demanded a second press for a confirm line with no reason to state. Both now key off `forecast.consent_affected`, the same field the confirm sentence renders from.

*AC#4 was over-applied.* Un-gating the commit line also un-gated it for `registry_available=False` / `media_db_available=False`, where it promised imports beside a permanently dead Start. AC#4 is about a BLOCKED-but-real selection keeping its numbers (option errors, armed consent — still true, still pinned by `test_commit_line_stays_visible_and_synced…`); a runtime with no import path at all is a different state and now renders no forecast.

*A capped probe's arithmetic.* `match_capped` made `will_match` a floor and therefore `will_import` a CEILING, but only the match half was hedged — "5 will import · at least 20 will match" is arithmetic a user can catch out. The hedge now propagates: "at most 5 will import · at least 20 will match".

Mutation-checked: reverting the `targets_server` branch turns the four server tests red; the governance test `test_forecast_counts_equal_the_real_receipt_for_a_mixed_folder` stays green throughout (local mode is unchanged). Modified: `Library/library_ingest_state.py`, `UI/Screens/library_screen.py`, `Tests/Library/test_library_ingest_state.py`, `Tests/UI/test_library_ingest_inline_consent.py`, `Docs/User_Guide/library/import-and-export.md`.
<!-- SECTION:NOTES:END -->
