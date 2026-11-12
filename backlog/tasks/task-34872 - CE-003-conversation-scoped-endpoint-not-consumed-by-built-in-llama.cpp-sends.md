---
id: TASK-34872
title: 'CE-003: conversation-scoped endpoint not consumed by built-in llama.cpp sends'
status: To Do
assignee: []
created_date: '2026-09-13 14:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleSettingsModal defined _base_url_changed twice in its class body; the later definition (model-discovery evidence invalidation, PR #2365) shadowed the earlier one that marks the endpoint draft dirty. A typed conversation-scoped Base URL therefore never produced a dirty live-bound ConsoleEndpointDraft, so _submission_for_action stripped the endpoint and the controller rebase reset settings.base_url to the configured provider URL: 'Use for this conversation' committed without any endpoint and the next send resolved the stale api_settings.llama_cpp.api_url (UAT CE-003: blocked at http://localhost:8080 while the field verifiably held http://127.0.0.1:9099; gateway and selection layers were verified correct). Full evidence chain and repro: .superpowers/ce003-investigation.md; fix implemented TDD in worktree commit 084015028a (merge the two handlers into one).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Typing a Base URL in Conversation settings marks the endpoint draft dirty and live-bound (bound to the active provider config key)
- [ ] Regression test: modal typing dirties `_endpoint_draft` with the typed value and bound provider key
- [ ] Regression test: apply-to-chat persists the typed llama.cpp endpoint into store session settings via the real rebase + live-commit path
- [ ] A send whose session settings carry a base_url differing from the configured api_url resolves and probes the session base_url (gateway + selection precedence verified)
- [ ] Targeted suites green (modal settings, settings apply/store, provider gateway) with no new failures vs baseline
<!-- AC:END -->
