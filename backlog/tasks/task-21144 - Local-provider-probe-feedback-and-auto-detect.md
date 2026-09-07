---
id: TASK-21144
title: Local provider probe feedback and auto-detect
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:14'
updated_date: '2026-08-25 15:03'
labels:
  - ux
  - wizard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings P-6, P-7, P-8 (findings.md): Detect and Test give zero feedback when no local server is running (byte-identical frames); the subtitle promises auto-detection that never happens; Detect vs Test is unexplained.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting a local provider auto-probes (debounced, short timeout) with a visible in-progress state
- [x] #2 Every probe ends in a visible result: found endpoint, or a not-found message naming the address tried
- [x] #3 Probe buttons are labeled by outcome (find server / test address)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace Detect/Test handlers + probe-status widget; find why not-found renders nothing\n2. Add in-progress + result states for both probes\n3. Auto-probe on local provider selection (debounced, short timeout)\n4. Relabel by outcome; tests; live tmux with no server + with a fake local server
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Investigation first: on the current build (post-F-1/21139 and post-keyboard/21142), Detect already reported 'Searching…'/'No local endpoints found.', selection already auto-ran discovery with a 'Couldn't discover models for X. You can continue anyway.' status, and Test already rendered '✗ …' verdicts — the UAT-observed silence decomposed into (a) the F-1 focus soft-lock eating the button presses and (b) the shared #setup-provider-probe-status Static composed at the panel's very bottom, below the fold at 40-row terminals. Remaining fixes made: the status Static moved above the Authentication collapsible, directly under the connection controls it reports on (verified live at 140x40: renders 4 rows below the buttons); buttons relabeled by outcome — 'Find local servers' / 'Test connection' (P-8). New app-level contract test (loopback_network marker) pins: selection-time feedback appears unprompted, Test ends in a visible verdict, status adjacency, and the labels. Suites: 872 passed.

Files: FirstRunSetupWizard.py, Tests/UI/test_first_run_wizard_live_contract.py.
<!-- SECTION:NOTES:END -->
