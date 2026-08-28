---
id: TASK-23089
title: >-
  First-run model discovery sends no Authorization header, so a valid cloud key
  always fails
status: To Do
assignee: []
created_date: '2026-08-28 06:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live first-run setup with a real, working OpenAI key still dead-ends at the Model step. The outbound models request carries no Authorization header at all, so it 401s and the step reports a failure for a perfectly good credential. This is the defect that keeps cloud first-run from working end to end; it is separate from the encoding and catalog-bound fixes in PR #2158, which are necessary but not sufficient.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A key typed into the Provider step reaches model discovery as an Authorization header
- [ ] #2 First-run setup with a valid OpenAI key lists real models at the Model step
- [ ] #3 The Model step reports the true failure category instead of a hardcoded 'request failed' on the ProviderStep handoff path
- [ ] #4 A regression test drives discovery through the wizard's own staged-settings shape and asserts the credential reaches the request
<!-- AC:END -->

## Evidence (live, 2026-08-27)

Real OpenAI key, isolated profile, key typed into the Provider step via the
real TUI. Model step showed:

    Couldn't reach the server (request failed). Check it's running, then
    Retry — or enter a model ID below.

Retry reproduced it. The key itself is good: `chat_api_call` returned a
completion, and `discover_openai_compatible_models(api_key=...)` returned
128 models when the credential is passed directly.

Driving the wizard's own path — `LocalLLMProviderCatalogService.discover_models`
with the staged-settings shape `_first_run_discovery_staged_settings` builds
for a `draft` credential (`{api_base_url, credential_source: "draft", api_key}`)
— with an httpx spy on the outgoing request:

    status: error | kind: missing_credentials
      request -> https://api.openai.com/v1/models   Authorization header present: False

So the staged draft credential is dropped before the request is built.
`_resolve_hosted_provider` (which does read staged credentials) is only
reached for `_STRICT_HOSTED_PROVIDER_KEYS`, and `openai` is not in that set
(`"openai" in _STRICT_HOSTED_PROVIDER_KEYS` -> False). Any cloud provider
outside that set whose `/models` requires auth will behave the same way.

Second, smaller defect on the same screen: the true cause never reaches the
user. `FirstRunSetupWizard.py` hardcodes `failure_category = "request failed"`
on the ProviderStep handoff path (two sites, ~3455 and ~3484), so an
authentication failure is rendered as "Couldn't reach the server ... Check
it's running" — telling the user to check a server when the problem is the
credential, and defeating the provider-aware error copy added for UAT M-4.

## Notes

- Not fixed here deliberately: changing which providers honor staged draft
  credentials is credential-resolution semantics and wants its own review.
- PR #2158 fixes two other defects on this path (gzip encoding, 100-model
  bound). Both are required for this flow to work, neither is sufficient.
