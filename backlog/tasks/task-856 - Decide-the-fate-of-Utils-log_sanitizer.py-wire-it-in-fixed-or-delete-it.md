---
id: TASK-856
title: 'Decide the fate of Utils/log_sanitizer.py: wire it in fixed, or delete it'
status: To Do
assignee: []
created_date: '2026-07-27 04:35'
labels:
  - security
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Utils/log_sanitizer.py has zero production importers -- a repo-wide grep for log_sanitizer under tldw_chatbook/ and Tests/ finds only Tests/Utils/test_security_enhancements.py. The 40+ sanitize_string call sites elsewhere in the app resolve to a different function in Utils/input_validation.py. So every literal in this module protects nothing at runtime today; only the test file exercises it, making that test green and vacuous.

Independent of the dead-import problem, the module's rules are wrong in both directions. log_sanitizer.py:16 labels the pattern claude-[a-zA-Z0-9-]+ as matching "Anthropic keys", but claude-* is the model-id prefix (e.g. claude-opus-4-20250514), not a key format; a reproduction showed a log line naming the model claude-opus-4-20250514 gets its model name destroyed (replaced with ***ANTHROPIC_KEY***) while a real Anthropic key (sk-ant-api03-...) and a real OpenAI key (sk-proj-...) both survive sanitize_string unredacted, because :15's sk-[a-zA-Z0-9]{20,} character class excludes "-" and stops at "sk-ant"/"sk-proj". sanitize_dict() similarly redacts openai_api_key and auth_token but passes x-api-key, cohere_api_key, api_token, secret_key, and refresh_token through untouched. SENSITIVE_FIELDS also names ten literals that appear nowhere in this codebase (aws_access_key_id, connection_string, private_key, etc.) and misses roughly 30 real ones. If this module is ever wired in as-is, it will strip model names from logs while letting real credentials through.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is made and implemented: either log_sanitizer.py is deleted (along with its now-vacuous test), or it is fixed (correct key-format regexes, a field list that matches the app's real key names, an inverted Anthropic-key rule corrected) and wired into at least the log paths that currently rely on Utils/input_validation.sanitize_string for secret redaction
- [ ] #2 If kept, a test builds its expected-redacted-field list from the app's real config key names (not the module's own literal list) and confirms every real provider key format (openai, anthropic, cohere, etc.) is actually redacted from a sample log line
- [ ] #3 If deleted, its test file is removed and no remaining code references the module
<!-- AC:END -->
