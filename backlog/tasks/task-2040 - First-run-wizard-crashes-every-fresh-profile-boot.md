---
id: TASK-2040
title: >-
  First-run wizard crashes every fresh-profile boot (app_instance read before init)
status: Done
assignee: []
created_date: '2026-08-03 01:30'
labels:
  - first-run
  - wizard
  - crash
  - p0
priority: high
dependencies: []
---

## Description (the why)

Every fresh profile's FIRST app boot crashed the entire app:
`SetupWizardContainer.__init__` builds its steps (`_create_steps()`)
BEFORE calling the base `WizardContainer.__init__` that assigns
`app_instance`, and `SpeechSetupStep.__init__`
(`FirstRunSetupWizard.py:1173`) reads
`self.wizard.app_instance.app_config` at construction time →
`AttributeError: 'SetupWizardContainer' object has no attribute
'app_instance'` → app exits. The SECOND boot works because the crashed
first boot already persisted `first_run.setup_started`, changing the
offer path — which made the crash look like a transient. Found during the
2026-08-02 ingest-UAT live verification (two "transient" first-launch
deaths on fresh profiles, captured on the third).

## Acceptance Criteria (the what)

- [x] A fresh profile's first boot reaches the setup wizard instead of
      crashing (live-verified: wizard renders on first launch).
- [x] `SetupWizardContainer(app_instance)` construction is pinned by a
      regression test that fails on the pre-fix code (verified red→green
      via patch swap).

## Implementation Notes

One assignment: `self.app_instance = app_instance` at the top of
`SetupWizardContainer.__init__`, before `_create_steps()` (the base class
re-assigns the same value harmlessly). Regression test in
`Tests/UI/test_first_run_wizard_live_contract.py`. Wizard suite 15/15.
The likely origin is the recent SpeechSetupStep addition reading
`app_config` at construction; any future step doing the same is now safe.
