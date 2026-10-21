---
id: TASK-35597
title: Make Tools-hub UI suites PR-gate-lane compatible, then census them
status: Done
created_date: 2026-10-19 16:38
updated_date: 2026-10-21 02:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3140's pre-merge review tried to close the coverage gap by adding test_settings_tools_hub, test_mcp_permissions_handoff, test_settings_footer_hints, test_screen_footer_hints, and test_screen_navigation to scripts/ui_pr_gate_census.txt. On CI those entries failed en masse: the app-constructing tests hit RecoveryRequired(raw_source_selection_changed) when run plain in one lane process (they need the @private_profile_test per-test child-runner isolation), and two pre-existing CSS drift guards failed on first in-lane execution (tldw_cli_modular.tcss lost footer section markers; AppFooterStatus.BUNDLED_CSS vs _widgets.tcss KEEP-IN-SYNC drift — both fail on dev's own tree too, for the footer owners). The entries were reverted; the gap is real but must be closed by making the suites lane-safe first (private-profile markers or per-test profile isolation), then re-adding them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 App-constructing tools-hub/port suites run green in a plain PR-gate lane process,Census re-adds the five suites with the floor raised deliberately,CSS drift findings handed to the footer owners or fixed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. INVESTIGATED (findings): Root cause of the lane failures is ADR-126 config-participant admission, not the suites' logic. Tests/conftest.py binds the tldw_chatbook.config module as a raw participant to the SESSION bootstrap sandbox at import time (eager Chunking/config_profiles imports). isolate_test_environment then redirects TLDW_CONFIG_PATH to a per-test tmp_path for every file NOT on its keep-list; the first guarded config read in such a test re-registers the config source with the new path and _register_raw_participant fails closed with RecoveryRequired(raw_source_selection_changed) (raw_participants.py:307). Verified: dev's censused app-constructing suites survive the lane via @pytest.mark.bootstrap_profile (60+ censused files; e.g. test_settings_configuration_hub.py marks its mounting tests per-node and is censused by node id) or the conftest file allowlist; @private_profile_test works because each case runs in a fresh child interpreter.
2. Fix mechanism: per-test @pytest.mark.bootstrap_profile on the app/harness-constructing tests of test_settings_tools_hub.py (5), test_settings_footer_hints.py (7), test_screen_footer_hints.py (6); module-level pytestmark for test_screen_navigation.py (108 of 141 plain-run nodes fail; private_profile would cost 108 child boots x 5-20s = far over shard budget, marker is the established in-lane pattern). test_mcp_permissions_handoff.py is already fully @private_profile_test and passes plain - no change.
3. CSS guards (test_screen_footer_hints.py): both fail on dev. (i) built-bundle marker guard: the bundle deliberately strips source comments (build_css without_source_comments, policy guarded by test_css_generated_comments.py, landed ae9093a714) so markers can never return via rebuild - adapt the guard to extract the footer selectors structurally from the bundle instead. (ii) BUNDLED_CSS parity guard: _widgets.tcss migrated to -* tokens in the same commit; BUNDLED_CSS must keep raw literals (its documented purpose is stylesheet-less harnesses where -* do not resolve - settings_tools_hub.py DEFAULT_CSS comment) - teach the guard to resolve -* tokens from core/_variables.tcss before comparing. Verified all 7 drifted values resolve to exactly the BUNDLED_CSS literals.
4. Verify: .venv/bin/python -m pytest <five files> -q -p no:cacheprovider (one process, mirroring the lane) green; run Tests/UI/test_design_token_governance.py after CSS-adjacent changes.
5. Census: append the green files to scripts/ui_pr_gate_census.txt with a task-35597 comment; run scripts/check_ui_pr_gate_census.py.
6. Commit (fix + census), mark task Done with Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Lane-safe via bootstrap_profile/private_profile_test; 2 CSS guards fixed guard-side; census: 3 whole files + 112 node ids (523 entries, checker OK); lane-mirroring run 145 passed; 36 dev-red tests documented and left ungated for owners.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made the five Tools-hub UI suites PR-gate-lane safe via the repo's
bootstrap_profile / private_profile_test isolation mechanisms, repaired two
CSS drift guards on the guard side (structural bundle extraction; $ds-*
token resolution), re-pinned two stale masked assertions, and re-censused:
3 whole files + 112 node-id entries (11 footer-hint nodes, 101 navigation
nodes). Lane-mirroring plain run over every added census entry: 145 passed.
36 dev-red tests across two files stay ungated and documented for their
owners (masked-red rot, red under fresh profiles too).
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
