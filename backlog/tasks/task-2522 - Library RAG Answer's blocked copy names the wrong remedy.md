---
id: task-2522
title: Library RAG Answer's blocked copy names the wrong remedy
status: To Do
assignee: []
created_date: '2026-08-06 02:18'
labels:
  - library
  - rag
  - ux
dependencies: []
priority: medium
---

## Description

When Library RAG Answer's Run button is blocked because no credential resolves for the configured provider, the
panel shows: **"Blocked | Select a provider/model before asking for a RAG answer."** (`Library/library_rag_
state.py:1103`, recovery pointer "Console controls"). That copy tells the user to pick a provider/model — but
after PR-T2 Task 7, this exact branch is only reached when a provider/model IS already selected
(`resolve_library_rag_answer_provider()` resolved a name) and only the *credential* is missing. The remedy the
user actually needs is a completely different action, and the code already knows it.

`Chat/provider_readiness.get_provider_readiness(...)` returns a `ProviderReadiness` whose `.recovery` field
carries the real, specific fix — e.g. `"Set ANTHROPIC_API_KEY or add api_key under [api_settings.anthropic]."`
(`Chat/provider_readiness.py:256-260`). `library_rag_answer_provider_ready()` (`Library/library_rag_answer_
service.py:180-216`) discards this: it collapses the whole `ProviderReadiness` down to `.ready: bool` before
returning, so the specific recovery text never reaches the gate that builds the blocked-copy message.

For a PR whose whole point is honesty at paid moments, telling the user to do the one thing they've already done
(pick a provider) instead of the one thing they haven't (set a credential) undermines that goal.

## Acceptance Criteria

- [ ] When Library RAG Answer is blocked specifically because the resolved provider's credential doesn't
      resolve (endpoint named, `get_provider_readiness(...).ready is False`), the blocked message's recovery
      text is derived from `ProviderReadiness.recovery` (or equivalent real remedy) instead of the generic
      "Select a provider/model before asking for a RAG answer."
- [ ] The empty/missing-endpoint case (no provider selected at all) keeps its existing "Select a provider/model"
      copy — this task only fixes the case where a provider *is* selected but its credential is missing
- [ ] A regression test pins the credential-missing blocked message naming the actual missing key/config path,
      not a generic "select a provider" instruction
