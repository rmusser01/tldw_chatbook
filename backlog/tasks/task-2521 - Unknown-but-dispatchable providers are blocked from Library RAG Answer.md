---
id: TASK-2521
title: Unknown-but-dispatchable providers are blocked from Library RAG Answer
status: Done
assignee: []
created_date: '2026-08-06 02:17'
labels:
  - library
  - rag
  - config
  - bug
dependencies: []
priority: medium
---

## Description

PR-T2 Task 7 made `library_rag_answer_provider_ready()` (`tldw_chatbook/Library/library_rag_answer_service.py`)
ask `Chat/provider_readiness.get_provider_readiness(...)` instead of merely checking that an endpoint name was
configured — closing the "spends money while Console shows a blocking wall" split. But
`get_provider_readiness` reports `reason="Unknown provider"` (and `ready=False`) whenever the normalized
provider key is not in `KNOWN_PROVIDER_KEYS` (`Chat/provider_readiness.py:10-47`, the union of
`PROVIDERS_REQUIRING_API_KEY_KEYS` and `KEYLESS_PROVIDER_KEYS`) and no `api_settings` table exists for it.

`Chat_Functions.py`'s `API_CALL_HANDLERS` dispatch table (`tldw_chatbook/Chat/Chat_Functions.py:106-136`)
happily dispatches several provider keys that are absent from `KNOWN_PROVIDER_KEYS` once normalized through
`provider_config_key()`: `"custom-openai-api"` and `"custom-openai-api-2"` (hyphens normalize to
`custom_openai_api` / `custom_openai_api_2`, neither of which matches the underscored `"custom"` / `"custom_2"`
entries in `KEYLESS_PROVIDER_KEYS`), and bare `"mlx_lm"` (only `"local_mlx_lm"` is listed).

A self-hoster with `default_api_endpoint = "custom-openai-api"` (or `mlx_lm`) had a working Library RAG Answer
before PR-T2 — the old gate only checked that an endpoint name was set. After PR-T2, `resolve_library_rag_
answer_provider()` still resolves the name, but `library_rag_answer_provider_ready()` now returns `False` for
it, so the Run button is permanently disabled with no way to configure past it (there is no credential to add —
the provider is genuinely keyless/unknown-to-the-readiness-table, not missing a key).

## Acceptance Criteria

- [x] `custom-openai-api`, `custom-openai-api-2`, and `mlx_lm` (and any other key present in `API_CALL_HANDLERS`
      but absent from `KNOWN_PROVIDER_KEYS` once normalized) resolve as ready in `get_provider_readiness` (or
      `library_rag_answer_provider_ready()` specifically) when no credential is required for them, instead of
      being reported as an unknown/blocked provider
- [x] A regression test pins that a user with one of these endpoints configured as `default_api_endpoint` sees
      Library RAG Answer's Run button enabled
- [x] Providers that genuinely require a credential and are simply unrecognized still report not-ready (this
      fix does not weaken the "unknown provider" rejection for providers that actually need a key)

## Implementation Notes

Closed in-branch by PR-T2's post-review fix wave (review round 3, finding I2) rather than as a follow-up: the
whole-branch review ruled this a user-facing regression of this branch, not a pre-existing gap.

`custom_openai_api`, `custom_openai_api_2` and `mlx_lm` were added to `KEYLESS_PROVIDER_KEYS`
(`tldw_chatbook/Chat/provider_readiness.py`) -- the normalized forms of the `API_CALL_HANDLERS` keys
`"custom-openai-api"`, `"custom-openai-api-2"` and `"mlx_lm"`. This widens the keyless set only; nothing about
the "unknown provider" rejection for credential-requiring providers changed.

Pins (`Tests/Chat/test_provider_readiness.py`): `test_dispatchable_hyphenated_provider_spellings_stay_ready`
(all three spellings, parametrized) and `test_widening_the_keyless_set_did_not_weaken_credential_rejection`
(AC 3 -- anthropic still "Missing API key", an invented provider still "Unknown provider").
`Tests/Library/test_library_rag_answer_service.py::test_provider_gate_accepts_a_keyless_dispatchable_endpoint_
spelling` covers AC 2 at the seam the Library run gate consumes: the gate returns the endpoint NAME, which is
exactly the value `provider_name=` derives the Run button's enabled state from (that derivation is itself
pinned in `Tests/Library/test_library_rag_state.py`). A mounted-UI variant was judged redundant against that
chain rather than added.
