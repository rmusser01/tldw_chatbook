---
id: TASK-26022
title: Anthropic subscription credential borrow
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
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
- [ ] #1 When a Claude subscription credential exists on the machine, chatbook can use it for Anthropic requests instead of an API key
- [ ] #2 The credential is read, never written, refreshed or rotated by chatbook - a stale or expired credential produces a clear message telling the user to refresh it in the tool that owns it
- [ ] #3 The credential file is never copied into chatbook's own config or logs, and its value never appears in the execution log or an approval card
- [ ] #4 The user chooses this explicitly; discovering a credential on disk does not silently change how requests are billed
- [ ] #5 Readiness reports which credential source is in use, so the user can tell subscription from API key at a glance
- [ ] #6 With no such credential present, behavior is exactly as today
- [ ] #7 Requests carry the correct headers for the subscription path, verified against a real account before the task is closed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. This reads an existing on-disk credential and adds a header path; it mints no tokens and stores no new secret. If the work grows to include minting or refreshing tokens, stop and raise an ADR first - that crosses into owning a credential lifecycle.
<!-- SECTION:PLAN:END -->
