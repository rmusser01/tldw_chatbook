---
id: TASK-2523
title: Legacy-key bridge's modern-key lookup is exact-key while the readiness reader's is normalized
status: To Do
assignee: []
created_date: '2026-08-06 02:19'
labels:
  - config
  - provider-readiness
  - bug
dependencies: []
priority: low
---

## Description

PR-T2 Task 7 added `config.py`'s `_normalize_legacy_provider_api_key()` (`tldw_chatbook/config.py:820-946`) to
make Console's readiness check and the actual provider spend agree on one credential. Its modern-config lookup
is `provider_table = api_settings_section.get(provider_key)` (`config.py:927`) — an **exact-key** dict lookup
against the already-normalized `provider_key` (e.g. `"openai"`).

`Chat/provider_readiness.py`'s own reader, `_provider_settings_for_key()` (`provider_readiness.py:140-155`),
does not do an exact-key lookup — it iterates every key in `api_settings` and compares each one's *normalized*
form (`provider_config_key(str(configured_provider))`) against `provider_key`, so it matches
`[api_settings.OpenAI]`, `[api_settings.open-ai]`, etc. just as well as `[api_settings.openai]`.

A hand-edited config with a non-canonically-cased or -punctuated section header — e.g. `[api_settings.OpenAI]`
(capital O, matching the shipped `[providers]` display name) — makes the bridge's exact-key `.get("openai")`
miss it entirely. If a legacy `[API] openai_api_key` value also exists, the bridge then falls through to that
legacy value and writes a **new, duplicate** `api_settings["openai"]` table (lowercase) alongside the existing
`api_settings["OpenAI"]` table. `get_provider_readiness` (which reads with the normalized comparison) still
finds the original `OpenAI` table first and reports ready with *that* key, while the legacy `<provider>_api`
spend dict (fed by the bridge's return value) ends up with the *other* key — the exact split PR-T2 exists to
close, reopened by a hand-edited section header.

This is a hand-edit-only scenario today (the shipped default config and the Settings screen both write the
canonical lowercase key), so it has not been observed causing user harm, but it is a real, findable seam.

## Acceptance Criteria

- [ ] `_normalize_legacy_provider_api_key`'s modern-config lookup uses the same normalized-key matching
      `_provider_settings_for_key` uses (reuse that function, or an equivalent normalized lookup), instead of
      an exact-key `dict.get`
- [ ] A regression test: `[api_settings.OpenAI]` (non-canonical casing) with a real `api_key` is recognized by
      the bridge as the modern value (no duplicate lowercase table written, no fallback to a legacy `[API]`
      value that might also be present)
- [ ] No change to the precedence or write-back behavior already established by PR-T2 Task 7 for the canonical
      lowercase case
