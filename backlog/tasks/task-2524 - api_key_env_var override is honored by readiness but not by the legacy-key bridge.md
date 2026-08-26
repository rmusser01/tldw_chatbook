---
id: TASK-2524
title: api_key_env_var override is honored by readiness but not by the legacy-key bridge
status: To Do
assignee: []
created_date: '2026-08-06 02:20'
labels:
  - config
  - provider-readiness
  - bug
dependencies: []
priority: low
---

## Description

`Chat/provider_readiness.get_provider_readiness()` lets a per-provider config override which environment
variable holds the credential: `env_var_value = provider_settings.get("api_key_env_var")`
(`provider_readiness.py:210-215`), falling back to the conventional `<PROVIDER>_API_KEY` name only when that
key isn't set. `config.py`'s `_normalize_legacy_provider_api_key()` (added by PR-T2 Task 7) does not have this
flexibility — it is called once per provider with a single hardcoded `env_var` string (the conventional name,
from the `_LEGACY_PROVIDER_API_KEY_BRIDGE` tuple, e.g. `("anthropic", "anthropic_api_key",
"ANTHROPIC_API_KEY")`) and only ever checks `os.getenv(env_var)` against that fixed name (`config.py:934-936`).

This is pre-existing behavior (the bridge is new in PR-T2, but the underlying gap between "readiness honors
`api_key_env_var`" and "the spend path doesn't" predates it), but PR-T2's new credential gate on the Library
side makes the failure mode worse: a user who sets `api_key_env_var = "MY_ANTHROPIC"` under
`[api_settings.anthropic]` now gets `get_provider_readiness(...).ready == True` (readiness correctly reads the
override and finds the variable) — so PR-T2's new gate lets Library RAG Answer *run* — but the legacy
`anthropic_api` dict the actual provider call reads (`LLM_Calls/LLM_API_Calls.py:1218-1219`) still has
`api_key: None`, because the bridge never looked at `MY_ANTHROPIC`. The call then fails deep inside the
provider call with a `ChatConfigurationError` instead of being caught by any gate — a confusing mid-run failure
where the up-front check said everything was fine.

## Acceptance Criteria

- [ ] `_normalize_legacy_provider_api_key` (or its caller) honors a per-provider `api_key_env_var` override the
      same way `get_provider_readiness` does, falling back to the conventional `<PROVIDER>_API_KEY` name only
      when no override is configured
- [ ] A regression test: `api_key_env_var = "MY_ANTHROPIC"` under `[api_settings.anthropic]`, with `MY_ANTHROPIC`
      set in the environment and no other credential source, resolves the same key for both `get_provider_
      readiness` AND the legacy `anthropic_api` spend dict (no `ChatConfigurationError`, no gate/spend
      disagreement)
- [ ] No regression to the existing conventional-env-var-name resolution path
