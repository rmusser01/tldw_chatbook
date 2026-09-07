---
id: TASK-21143
title: 'Provider trust chain: probe outcomes drive tracker, model step, and summary'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:14'
updated_date: '2026-08-25 14:48'
labels:
  - ux
  - wizard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings S-1, M-2, N-7, M-1, M-4, P-5 (findings.md): a key that fails authentication still yields tracker checkmarks, a completed wizard, and a Summary reading 'checkmark Provider / checkmark Default model'; the model step's failure row offers a Retry that cannot succeed and never points back to the fix; connection errors are category-generic. Probe outcomes must be in-memory only (not in the persisted draft) and ride the existing provider invalidation fences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a failed auth probe, advancing from the Model step requires explicit confirmation
- [x] #2 Tracker shows an attention state (not a checkmark) for provider/model steps whose probe failed
- [x] #3 Summary shows the failure ('key failed an authentication check') and makes Review provider setup the primary action
- [x] #4 Auth failures point back to the Provider step; connection failures for ollama/llama.cpp name the server and how to start it
- [x] #5 Fixing the key clears all stale failure state (fences respected); outcomes never persist to the setup draft
- [x] #6 State transforms covered by unit tests in first_run_setup_state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure: classify_discovery_failure + summary-actions/rows extensions in first_run_setup_state (+unit tests)\n2. ModelStep: current_probe_failure(); auth vs connection row copy (provider-aware for ollama/llama.cpp); Retry hidden on auth; ack flag reset on identity change\n3. Container: confirm_before_advance gate -> confirmation dialog -> advance; provider_probe_failure(); Provider-step inline notice on Back\n4. Tracker: attention state (build_setup_progress attention_ids + CSS + glyph)\n5. Summary: row override + review_provider primary on probe failure\n6. App-level tests with stubbed failed outcomes; suites; live tmux with real bad key
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pure layer (first_run_setup_state): classify_discovery_failure (auth/connection/none from discovery state+category), apply_probe_failure_to_summary_rows (overlays ROW_ATTENTION + honest detail on a config-'configured' Provider row only), build_first_run_summary_actions gains provider_probe_failed (flips primary to review_provider), build_setup_progress gains attention_ids (completed steps downgrade to 'attention'). 5 unit tests.

Wizard: ModelStep records _rendered_probe_failure at render (staleness-guarded by discovery-key match — current_probe_failure() returns '' the moment identity changes, riding the existing invalidation fences; nothing persists to the draft); failure rows are auth-aware ('Authentication failed — this API key was rejected. Go Back to fix it…', Retry hidden — it cannot fix a rejected key) and provider-aware for connection failures (ollama/llama.cpp name the server and how to start it); SetupStep.confirm_before_advance + container one-shot _advance_confirmed gate Next behind a 'Continue anyway?' dialog; container.provider_probe_failure() feeds tracker attention ('!' amber glyph, new CSS state), the Provider step's pinned notice on return (P-5), and Summary (row overlay + primary flip).

Verified live end-to-end against the real Anthropic API with a fake key: 401 -> auth row + no Retry -> 'Continue anyway?' gate -> tracker shows ! on Provider/Model -> Summary reads '✗ Provider — saved, but the key failed an authentication check' with 'Review provider setup' primary. App-level tests cover the full chain incl. cancel-does-not-advance and confirm-advances-once. Four old-copy test pins updated. Suites: 871 passed. User guide updated.

Files: first_run_setup_state.py, FirstRunSetupWizard.py, _wizards.tcss (+regenerated modular), Tests/Wizards/test_first_run_setup_state.py, Tests/Wizards/test_first_run_setup_wizard.py, Docs/User_Guide/First_Run_Setup.md.
<!-- SECTION:NOTES:END -->
