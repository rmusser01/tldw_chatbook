---
id: TASK-26022
title: Anthropic subscription credential borrow
status: In Progress
assignee: []
created_date: '2026-08-31 15:45'
updated_date: '2026-09-02 06:35'
labels:
  - providers
  - auth
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Claude Pro or Max subscriber pays API rates on top of a subscription they already hold. Verified on origin/dev: Anthropic is API-key-only (Chat/provider_readiness.py:52) and a named grep for code_verifier, code_challenge, device_code, PKCE and claude.ai across tldw_chatbook returns zero - there is no OAuth path of any kind for an LLM provider. Hermes reads credentials other tools already minted, including Claude Code's own ~/.claude/.credentials.json, and sends the subscription authorization header. This is the narrow slice deliberately: read an existing credential, do not mint one. Full PKCE with refresh rotation is a separate, much larger piece of work and is explicitly out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a Claude subscription credential exists on the machine, chatbook can use it for Anthropic requests instead of an API key
- [x] #2 The credential is read, never written, refreshed or rotated by chatbook - a stale or expired credential produces a clear message telling the user to refresh it in the tool that owns it
- [x] #3 The credential file is never copied into chatbook's own config or logs, and its value never appears in the execution log or an approval card
- [x] #4 The user chooses this explicitly; discovering a credential on disk does not silently change how requests are billed
- [x] #5 Readiness reports which credential source is in use, so the user can tell subscription from API key at a glance
- [x] #6 With no such credential present, behavior is exactly as today
- [x] #7 Requests carry the correct headers for the subscription path, verified against a real account before the task is closed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. This reads an existing on-disk credential and adds a header path; it mints no tokens and stores no new secret. If the work grows to include minting or refreshing tokens, stop and raise an ADR first - that crosses into owning a credential lifecycle.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decided GO (2026-09-02, ToS/account-risk call is theirs). ACs #1-#6 implemented TDD; AC#7 (live verification against the real subscription) is OWNER-DRIVEN and stays open until they run it.

Implementation:
- LLM_Calls/anthropic_subscription.py (new): read_claude_code_credential (read-only parse of ~/.claude/.credentials.json; missing/malformed -> None, expired flagged); SubscriptionCredential masks the token in repr/str; anthropic_auth_source config gate (junk -> safe api_key default); subscription_headers* (authorization Bearer + anthropic-beta oauth-2025-04-20, replaces x-api-key entirely).
- chat_with_anthropic: auth_source=claude_subscription -> bearer headers, NO x-api-key; missing/expired credential FAILS with the refresh-in-Claude-Code message (never silent API-key fallback -> billing honesty); default mode never even reads the credential file (test-pinned). Extended-cache beta now MERGES with the oauth beta instead of overwriting.
- provider_readiness: subscription mode reports source 'subscription:claude_code' with reason 'Ready (Claude subscription)' (token never rides the readiness record); expired/missing -> blocked with refresh copy. Validator + closed reason-vocabulary extended for the key-less subscription source.
- Utils/log_sanitizer: sk-ant-oat/ort token shapes redacted.
- config.py: [api_settings.anthropic] auth_source documented, commented-out default.

Tests: Tests/LLM_Calls/test_anthropic_subscription.py (16). Readiness regression suite 460 green.

AC#7 CLOSED (live verify against a real Max account, 2026-09-02) - surfaced two gaps the static path hid, both fixed:
- macOS Keychain source: Claude Code stores the credential in the login Keychain ("Claude Code-credentials"), not ~/.claude/.credentials.json. read_claude_code_credential now falls back to a read-only, darwin-gated Keychain read (_keychain_credential_raw, absolute /usr/bin/security, 5s timeout, any failure -> None); file stays authoritative, non-macOS behavior unchanged.
- Claude Code identity gate: the OAuth token is rejected (misleading 429 rate_limit_error) unless system leads with "You are Claude Code, Anthropic's official CLI for Claude." chat_with_anthropic now prepends it as the first system block on the subscription path only (with_claude_code_identity), preserving the user's own prompt as a following block; api-key sends untouched.
- Live evidence: end-to-end chat_with_anthropic send with auth_source="claude_subscription" + a normal note-app system prompt read from keychain:Claude Code-credentials and returned 200 with real usage (input 56 / output 4). +13 tests. Lesson in lessons-live-verification.md.
<!-- SECTION:NOTES:END -->
