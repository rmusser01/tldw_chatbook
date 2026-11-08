---
id: TASK-39149
title: "The trajectory export masks sk- keys but leaves bare sk_live_, sk_test_, rk_live_ and rk_test_ keys in an exported trace"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- security
- privacy
- trace
- export
priority: medium
references:
- tldw_chatbook/Chat/trajectory_export.py
- Tests/Chat/test_trajectory_export.py
- tldw_chatbook/Utils/log_sanitizer.py
dependencies:
- TASK-39123
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (backup stream: TASK-39123's notes on round-12 branch fu/r12-backup list this as not covered). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are dev's; the module is the same on the stack.

- `_CREDENTIAL_VALUE_RES` (`Chat/trajectory_export.py:301-314`) has one key-prefix pattern, `\bsk-[A-Za-z0-9_-]{8,}` (`:308`). `_credential_text` (`:340-344`) applies these patterns to exported strings, and `_has_credential` decides with it.
- A probe on dev and on the stack used synthetic values and no network:
  - `_credential_text("sk-abcdefgh12345678")` returns `[credential redacted]`;
  - the same call returns `sk_live_abcdefgh12345678`, `sk_test_…`, `rk_live_…` and `rk_test_…` unchanged.
- This is separate from Console trace storage. TASK-39123 (round 12) moved stored traces to `credentials-v2`, whose patterns live in `Utils/log_sanitizer.py` (`trace_credentials_v2_spans`, round-12 branch). The export does not read trace storage and keeps its own regex.
- No PR lane runs `Tests/Chat/test_trajectory_export.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An exported trajectory masks bare `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` keys wherever it masks `sk-` keys; a test is red on dev, where the value is returned unchanged
- [ ] #2 `sk-` keys, private keys, bearer and basic tokens, AWS key ids and URL user info are still masked as before
- [ ] #3 The new tests run on a PR lane (never the fast PR contract), with three timings of that step recorded
<!-- AC:END -->
