---
id: TASK-39123
title: "Console trace storage's credentials-v1 projection leaves bare sk_live_, sk_test_, rk_live_ and rk_test_ keys unmasked, and v1 is versioned, so stored traces need a credentials-v2 projection"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- security
- console
- tracing
- needs-owner-decision
priority: medium
references:
- tldw_chatbook/Chat/console_trace_redaction.py
- tldw_chatbook/Utils/log_sanitizer.py
- tldw_chatbook/Chat/console_semantic_revision.py
- tldw_chatbook/DB/recovery_core_schema.py
- Tests/Utils/test_log_sanitizer.py
dependencies:
- TASK-39061
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39061's notes (round-11 branch fu/r11-arch) defer the trace side: "Credentials-v1 traces therefore still do not mask these keys". Line numbers are dev's.

**What the projection masks.**
- The trace projection's `_SECRET_TEXT` (`Chat/console_trace_redaction.py:51-56`) knows `sk-(?:live-)?…` plus a labelled-value rule.
- It then runs `sanitize_trace_credentials_v1` (`:708`). That reads `_STANDALONE_CREDENTIALS` (`Utils/log_sanitizer.py:104-116`), whose key forms are all `sk-`.
- fu/r11-arch adds the underscore forms only for diagnostics (`_DIAGNOSTIC_ONLY_CREDENTIALS`, applied in `_sanitize_diagnostic_text`), not for traces.

**Probe.** A pure probe on dev with synthetic 40-character bodies:

| Input | Masked |
|---|---|
| bare `sk_live_`, `sk_test_`, `rk_live_` or `rk_test_` keys in prose | no |
| a tool-result value `{'stripe': 'sk_live_…'}` | no |
| a nested message value `{"messages":[{"content":"key is sk_test_…"}]}` | no |
| `api_key=sk_live_…` (the label rule) | yes |
| `sk-…`, `sk-live-…` | yes |

**Why v1 cannot simply grow.**
- Each trace policy persists `credential_filter_version` (`DB/recovery_core_schema.py:248`).
- Revisions bind only results whose version equals `CREDENTIAL_FILTER_VERSION` (`console_semantic_revision.py:922`, `:968`).
- `Tests/Utils/test_log_sanitizer.py:87` pins the v1 projection.
- So new patterns added in place would change what an already stored v1 projection means.
- Tool results and provider envelopes are where such keys can appear without a label.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 New traces are stored under a credentials-v2 projection that masks bare `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` keys of realistic length in prose, in a tool-result string and in a nested JSON trace value; a test is red before the change on each case
- [ ] #2 Identifiers that only share a prefix (for example `sk_live_mode`) stay readable under v2
- [ ] #3 v1 output is byte-identical: the v1 pins stay green, and existing v1 policies and revisions still read
- [ ] #4 The owner decides how existing credentials-v1 policies are treated once v2 exists; the decision is recorded
- [ ] #5 The new tests run on a PR lane (never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->
