---
id: TASK-364
title: Show the current model in the quick Change model popover and label its temperature input
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The palette command 'Console: Change model… — Quick provider/model/temperature switch (Alt+M)' opens a compact 'Model' popover containing: a Provider select showing the raw key 'llama_cpp' (the full modal shows 'llama.cpp'), a Model select showing the placeholder 'Select' instead of the active model (local-gemma), a 'Search all models…' input, a bare unlabeled input containing '0.6', static text 'Streaming: on', and Full settings…/Apply buttons.

**Repro:** 1. Click transcript, Ctrl+P, type 'change model', activate 'Console: Change model…'. 2. Observe popover: Model select = 'Select' placeholder (session model is local-gemma), temperature input has no label, provider shows 'llama_cpp'.

**Verifier note:** Real bug with confirmed mechanism, beyond the tracked raw-key issue: ConsoleModelPopover.compose seeds the model Select with the session model (console_model_popover.py:113-115) and build_console_model_options injects the current model into options, BUT the provider Select's mount-time Select.Changed fires _provider_changed (lines 143-159), which rebuilds model options with current_model=None and calls set_options() — resetting the model Select to BLANK and wiping the prefill. Matches j5-76 exactly (model shows 'Select' while rail shows local-gemma). Unlabeled temperature Input (placeholder invisible once a value is present) also verified at lines 122-128. The 'llama_cpp' raw provider key slice is KNOWN (task-194) — the popover ledger item and gap-not-exercised-2026-07 confirm this surface was never live-verified, so NEW not REGRESSION.

**Source:** Console UX expert review 2026-07-20 (finding j5-quick-model-popover-empty; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J5 settings journey. Evidence: `j5-76-model-popover.png`, `j5-71-palette-model.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A quick-switch surface must reflect the current model (that's what the user is switching FROM), label its temperature input, and use the same provider display names as the full settings modal. As-is a user cannot confirm the active model and could Apply with 'Select'
<!-- AC:END -->
