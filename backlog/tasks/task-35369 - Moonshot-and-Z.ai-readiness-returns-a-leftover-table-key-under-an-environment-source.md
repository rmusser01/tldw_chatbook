---
id: TASK-35369
title: Moonshot and Z.ai readiness returns a leftover table key under an environment source
status: To Do
created_date: 2026-10-17 19:00
dependencies:
- TASK-33001.14
labels:
- model-config-redesign
- follow-up
- credentials
priority: low
references:
- tldw_chatbook/Chat/provider_readiness.py
- tldw_chatbook/LLM_Calls/moonshot.py
- tldw_chatbook/LLM_Calls/zai.py
- backlog/decisions/012-provider-credential-settings-boundary.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-2 review of Phase 9 Task 17 (2026-10-17), pre-existing. For Moonshot and Z.ai, `get_provider_readiness` takes the key `resolve_provider_credential` chose and then re-resolves it through `_resolved_hosted_api_key`, which calls `resolve_moonshot_request` / `resolve_zai_request` with no `explicit_api_key`. Those resolvers read the table's `api_key` before any variable and never read `credential_source`. So under "environment", with a key left in the table and the named variable exported, readiness returns the leftover key but labels it `env:<VAR>`, while a summary (TASK-33001.14) sends the named variable, as ADR-012 rules. Readiness feeds Console's call wiring, so the Console can send the key the user's "environment" choice set aside. Only a hand edit reaches this state, because Settings deletes the table key on an "environment" save. Probed at 7f66f6874a with the real loader on a scratch config (`api_key = "sk-stored-x"`, `api_key_env_var = "MOONSHOT_SUMMARY_KEY"`, `credential_source = "environment"`, the variable exported as `sk-named`): the summary sent `sk-named`; readiness gave `sk-stored-x` with `api_key_source` `env:MOONSHOT_SUMMARY_KEY`. Z.ai behaved the same.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Under credential_source "environment" with a key left in `[api_settings.moonshot]` or `[api_settings.zai]` and the named variable exported, readiness returns the named variable's key, as its `api_key_source` says, proved by a real-loader test on a scratch config for both providers
- [ ] #2 The hosted resolvers still validate the key readiness hands them (an invalid key still reports invalid settings), and a stored-key or no-source table resolves as it does today, pinned by the existing readiness tests
- [ ] #3 ADR-012's TASK-33001.14 amendment no longer records this as a difference from readiness
<!-- AC:END -->
