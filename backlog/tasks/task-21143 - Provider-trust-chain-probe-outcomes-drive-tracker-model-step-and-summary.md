---
id: TASK-21143
title: 'Provider trust chain: probe outcomes drive tracker, model step, and summary'
status: To Do
assignee: []
created_date: '2026-08-25 06:14'
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
- [ ] #1 After a failed auth probe, advancing from the Model step requires explicit confirmation
- [ ] #2 Tracker shows an attention state (not a checkmark) for provider/model steps whose probe failed
- [ ] #3 Summary shows the failure ('key failed an authentication check') and makes Review provider setup the primary action
- [ ] #4 Auth failures point back to the Provider step; connection failures for ollama/llama.cpp name the server and how to start it
- [ ] #5 Fixing the key clears all stale failure state (fences respected); outcomes never persist to the setup draft
- [ ] #6 State transforms covered by unit tests in first_run_setup_state
<!-- AC:END -->
