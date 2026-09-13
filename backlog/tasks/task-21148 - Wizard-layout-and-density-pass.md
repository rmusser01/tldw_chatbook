---
id: TASK-21148
title: Wizard layout and density pass
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:15'
updated_date: '2026-08-25 23:01'
labels:
  - ux
  - wizard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings P-1, V-1, V-2, Z-1, Z-2, Z-3, F-2, F-3, G-2, N-6, S-4 (findings.md): steps overflow at 140x40 while carrying rows of decorative dead space; step titles scroll away first; the Voice step leads with plumbing and hides Test and Hear below the fold; the full-track tracker drops all step titles and truncates step 10 to '1'; 80x24 shows one provider row with no guidance; tool switches take 4 rows each; the step total changes mid-flight when Protect joins; the summary config path wraps mid-character.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 140x40, each quick-track step shows its title, primary content, and primary action without scrolling
- [x] #2 Voice step leads with a one-line purpose and Test and Hear; endpoint/model/format/speed live under an Advanced disclosure
- [x] #3 Full-track tracker keeps step titles at 140 cols and renders two-digit step numbers
- [x] #4 Below a minimum size the wizard shows an enlarge-terminal hint; at 80x24 every step remains operable
- [x] #5 Protect appears in the quick track from the start (marked skipped when keyless); the step total never changes mid-flight
- [x] #6 Summary config path never wraps mid-character
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Summary path middle-truncation (pure helper)\n2. Sub-minimum-size hint line in wizard chrome\n3. Tracker: titles under boxes (fits 10 items at 140), two-digit numbers\n4. Density CSS pass + dock step titles (pinned)\n5. Protect always in both tracks; keyless state copy; update count pins\n6. Voice step: outcome-first layout, plumbing under Advanced collapsible\n7. Suites + live sweeps at 140x40 and 80x24
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Six sub-changes, all live-verified at 140x40 and 80x24 plus 883-test suites green:

S-4: middle_truncate_path (pure) — summary config path one-lines with an ellipsis; footer test updated to the truncation contract.
Z-2: #setup-size-hint in wizard chrome (on_resize, <100x30) — self-hides at the styles level too, since stylesheet-less test harnesses have no .hidden rule (a docked row that only pretends to hide shifts every geometry below it — caught by the suite).
F-2/F-3: tracker titles now stack UNDER the number boxes (item cost = max(number, title) so all 11 full-track titles fit at 140; two-digit boxes width:auto). Root cause of live titles STILL vanishing after the restack: stale FirstRunSetupWizard-scoped height:4/height:3 caps in _wizards.tcss from the old horizontal design — consolidated to auto. Widget carries DEFAULT_CSS so bare harnesses hold the layout.
P-1/Z-3 density: progress/step padding 2->1, Collapsible Contents padding + field-label margins zeroed inside steps — Provider now shows title, full list, probe status AND the whole key panel simultaneously at 140x40 (the density delta even broke a test's overflow premise at 100x24 — its terminal shrunk to 100x18 to keep proving scroll-into-view).
N-6: Protect always on both tracks (active_step_ids ignores key_entered); keyless ProtectKeysStep hides the password button and says 'No API keys saved yet — nothing to protect'; Welcome quick label renamed within its 61-char no-wrap budget ('provider, model, voice, protection'); focus-walk helper mirrors production's nav fallback for focusable-less steps; ~14 count/order pins updated.
V-1/V-2: Voice leads with purpose line + sample + Test and Hear + Use-as-default; endpoint/auth/model/voice/format/speed under an 'Advanced — endpoint, model & output' Collapsible (ids unchanged); the 12-way scroll-reachable contract rewritten to primary-immediate + advanced-after-expand.

Files: FirstRunSetupWizard.py, first_run_setup_state.py, _wizards.tcss (+regenerated), Tests/Wizards/{state,wizard}, Tests/UI/live-contract, Docs/User_Guide/First_Run_Setup.md.
<!-- SECTION:NOTES:END -->
