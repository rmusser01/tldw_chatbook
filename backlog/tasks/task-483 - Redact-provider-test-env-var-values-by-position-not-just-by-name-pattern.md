---
id: TASK-483
title: 'Redact provider-test env-var values by position, not just by name pattern'
status: To Do
assignee: []
created_date: '2026-07-22 17:48'
labels:
  - settings
  - security
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Settings provider Test evidence prints an env-var's value (from os.environ) and relies on redact_secret_text, which only redacts when the assignment KEY name matches API_KEY|TOKEN|SECRET|PASSWORD. A credential env var with a custom name (e.g. MY_LLAMA_CRED), or credentials embedded in an endpoint URL, print their raw value. Pre-existing; TASK-432 marginally widened reachability by flowing unsaved draft env-var names into the printed line. Harden so the test never echoes a raw credential value.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Provider Test evidence reports env-var presence as present/missing (or a positionally-redacted value) instead of echoing the raw os.environ value
- [ ] #2 A custom-named credential env var (name not matching the secret pattern) never has its value printed in the Test evidence
<!-- AC:END -->
