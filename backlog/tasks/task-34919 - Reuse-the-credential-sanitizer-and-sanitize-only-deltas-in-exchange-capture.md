---
id: TASK-34919
title: Reuse-the-credential-sanitizer-and-sanitize-only-deltas-in-exchange-capture
status: To Do
assignee: []
created_date: '2026-10-13 14:47'
labels:
  - performance
  - ui-perf
  - wave-2-streaming
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every visible stream chunk flows through record_exchange_content to sanitize_capture_value_with_omission (console_exchange_capture.py:505-516), which constructs a fresh CredentialSanitizer (validation loop in __init__), runs regex credential detection, then walks the result twice more (stub_binary_strings, _remove_nested_credentials) — all under _exchange_lock. On the llama.cpp path this runs on the EVENT LOOP (console_provider_gateway.py:6391); on hosted paths it serializes concurrent exchanges through the lock. Exchange capture defaults ON, so this caps chunk rate on fast local models and adds latency to every token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No CredentialSanitizer or compiled pattern set is constructed per chunk — sanitizer state reused per exchange,Sanitization of accumulated lists is delta-based or amortized rather than full re-sanitize per chunk,llama.cpp path no longer sanitizes per chunk on the event loop,Exchange capture content and redaction semantics unchanged (existing capture tests pass)
<!-- AC:END -->
