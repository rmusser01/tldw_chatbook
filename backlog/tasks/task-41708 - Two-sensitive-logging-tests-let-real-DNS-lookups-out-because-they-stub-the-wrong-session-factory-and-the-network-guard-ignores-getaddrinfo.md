---
id: TASK-41708
title: "Two sensitive-logging tests let real DNS lookups out because they stub the wrong session factory and the network guard ignores getaddrinfo"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- tests
- security
priority: high
references:
- Tests/Chat/test_sensitive_llm_logging.py
- Tests/network_guard.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-41522 implementer and confirmed by its review (round 16). `chat_with_openai` has used `create_owned_default_session` since 2026-10-12, but `Tests/Chat/test_sensitive_llm_logging.py` (`test_sensitive_openai_http_error_log_and_exception_are_metadata_only` and the two `test_auxiliary_pins_*` tests) stubs only `create_default_session` and `create_pooled_default_session`. The adapter builds a real session; the socket guard blocks the connection, but `Tests/network_guard.py` does not guard `getaddrinfo`, so real DNS lookups of `api.openai.test` and `api.openai.com` ran during the covering run (no HTTP request was sent). The three tests are red on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three tests stub the session factory the adapter actually uses and pass
- [ ] #2 The network guard refuses a DNS lookup of a non-loopback name during tests, and a test shows it red before the change
- [ ] #3 No test in the suite needs a real DNS lookup to pass, or each one that does is listed in the notes with why
<!-- AC:END -->
