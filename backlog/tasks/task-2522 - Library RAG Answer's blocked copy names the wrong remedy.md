---
id: TASK-2522
title: Library RAG Answer's blocked copy names the wrong remedy
status: Done
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

- [x] When Library RAG Answer is blocked specifically because the resolved provider's credential doesn't
      resolve (endpoint named, `get_provider_readiness(...).ready is False`), the blocked message's recovery
      text is derived from `ProviderReadiness.recovery` (or equivalent real remedy) instead of the generic
      "Select a provider/model before asking for a RAG answer."
- [x] The empty/missing-endpoint case (no provider selected at all) keeps its existing "Select a provider/model"
      copy — this task only fixes the case where a provider *is* selected but its credential is missing
- [x] A regression test pins the credential-missing blocked message naming the actual missing key/config path,
      not a generic "select a provider" instruction

## Implementation Notes

Closed in-branch by PR-T2's post-review fix wave (review round 3, finding I1) rather than as a follow-up: the
whole-branch review ruled this a user-facing regression of this branch.

`library_rag_answer_provider_gate()` (`tldw_chatbook/Library/library_rag_answer_service.py`) is a new single
`resolve -> readiness -> name` pass returning a `LibraryRagProviderGate` that keeps `ProviderReadiness.recovery`
alongside the name; `library_rag_answer_provider_ready()` is now a boolean view of it. The remedy is threaded to
`LibraryRagQueryState.from_values` as a DISTINCT optional `provider_credential_recovery` argument -- a message,
never a second readiness flag, so Task 4's invariant (readiness derived solely from `provider_name`) is intact
and a remedy can never make a blocked state look ready. The RAG-mode blocked branch now forks: a named-but-
uncredentialed provider gets the real remedy (owner "LLM provider credential"), and the genuinely-unselected
case keeps the original copy verbatim. The remedy is markup-escaped, since it embeds `[api_settings.<provider>]`
and both sinks (run-button tooltip, blocked callout/recovery `Static`s) render Rich markup.

Also removed the double resolution at both `UI/Screens/library_screen.py` call sites (one gate call each).

Pins: `Tests/Library/test_library_rag_state.py` (`test_unselected_provider_keeps_the_select_a_provider_copy`,
`test_named_but_uncredentialed_provider_shows_the_real_remedy`, `test_credential_remedy_is_markup_escaped_for_
its_rendering_sinks`, `test_credential_remedy_cannot_make_a_blocked_state_look_ready`, `test_panel_state_
threads_the_credential_remedy_into_query_state`), `Tests/Library/test_library_rag_answer_service.py` (three gate
tests), and the mounted-UI pin `Tests/UI/test_library_shell.py::test_library_shell_search_rag_mode_blocks_run_
when_endpoint_named_but_credential_missing`, whose assertion was inverted from the old copy to the new remedy.
`Docs/User_Guide/library/search-and-rag.md` now documents both blocked cases.
